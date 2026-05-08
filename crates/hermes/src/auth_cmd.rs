use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{Duration as ChronoDuration, Utc};
use clap::{Args, Subcommand, ValueEnum};
use getrandom::fill as fill_random;
use hermes_core::{
    AuthStatusSummary, HermesContext, LoadedConfig, OPENROUTER_BASE_URL, clear_provider_auth_state,
    get_active_auth_provider, get_auth_status_summary, get_provider_profile,
    normalize_provider_alias, resolve_codex_access_token,
    resolve_google_gemini_runtime_credentials, resolve_minimax_oauth_runtime_credentials,
    resolve_nous_runtime_credentials, resolve_qwen_runtime_credentials,
};
use reqwest::Url;
use reqwest::blocking::Client;
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value;
use sha2::{Digest, Sha256};

use crate::config_cmd::save_env_value;
use crate::python_bridge::{project_root, resolve_repo_python};

const CONFIG_FALLBACK_LOGOUT_PROVIDERS: &[&str] = &[
    "nous",
    "openai-codex",
    "minimax-oauth",
    "google-gemini-cli",
    "qwen-oauth",
    "spotify",
];

const DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL: &str = "https://accounts.spotify.com";
const DEFAULT_SPOTIFY_API_BASE_URL: &str = "https://api.spotify.com/v1";
const DEFAULT_SPOTIFY_REDIRECT_URI: &str = "http://127.0.0.1:43827/spotify/callback";
const DEFAULT_SPOTIFY_SCOPE: &str = "user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-recently-played playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private user-library-read user-library-modify";
const SPOTIFY_DOCS_URL: &str =
    "https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify";
const SPOTIFY_DASHBOARD_URL: &str = "https://developer.spotify.com/dashboard";
const ANTHROPIC_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const DEFAULT_ANTHROPIC_OAUTH_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const DEFAULT_ANTHROPIC_OAUTH_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const ANTHROPIC_OAUTH_REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const ANTHROPIC_OAUTH_SCOPES: &str = "org:create_api_key user:profile user:inference";
const ANTHROPIC_OAUTH_USER_AGENT: &str = "claude-cli/0.0.0 (external, cli)";
const DEFAULT_GOOGLE_OAUTH_AUTHORIZE_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const DEFAULT_GOOGLE_OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const DEFAULT_GOOGLE_OAUTH_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v1/userinfo";
const GOOGLE_OAUTH_SCOPES: &str = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";
const GOOGLE_OAUTH_REDIRECT_HOST: &str = "127.0.0.1";
const GOOGLE_OAUTH_CALLBACK_PATH: &str = "/oauth2callback";
const GOOGLE_OAUTH_DEFAULT_REDIRECT_PORT: u16 = 8085;
const GOOGLE_DEFAULT_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
const GOOGLE_DEFAULT_CLIENT_SECRET: &str = "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";
const MINIMAX_OAUTH_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
const MINIMAX_OAUTH_SCOPE: &str = "group_id profile model.completion";
const MINIMAX_OAUTH_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:user_code";
const DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL: &str = "https://api.minimax.io";
const DEFAULT_MINIMAX_OAUTH_INFERENCE_BASE_URL: &str = "https://api.minimax.io/anthropic";
const DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL: &str = "https://api.minimaxi.com";
const DEFAULT_MINIMAX_OAUTH_CN_INFERENCE_BASE_URL: &str = "https://api.minimaxi.com/anthropic";
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_CODEX_OAUTH_ISSUER: &str = "https://auth.openai.com";
const DEFAULT_CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const DEFAULT_NOUS_PORTAL_URL: &str = "https://portal.nousresearch.com";
const DEFAULT_NOUS_INFERENCE_URL: &str = "https://inference-api.nousresearch.com/v1";
const DEFAULT_NOUS_CLIENT_ID: &str = "hermes-cli";
const DEFAULT_NOUS_SCOPE: &str = "inference:mint_agent_key";

const AUTH_ADD_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from types import SimpleNamespace\n",
    "from hermes_cli.auth_commands import auth_add_command\n",
    "raw_timeout = os.environ.get('HERMES_AUTH_ADD_TIMEOUT', '').strip()\n",
    "args = SimpleNamespace(\n",
    "    provider=os.environ.get('HERMES_AUTH_ADD_PROVIDER', ''),\n",
    "    auth_type=(os.environ.get('HERMES_AUTH_ADD_TYPE', '').strip() or None),\n",
    "    label=(os.environ.get('HERMES_AUTH_ADD_LABEL', '').strip() or None),\n",
    "    api_key=(os.environ.get('HERMES_AUTH_ADD_API_KEY', '').strip() or None),\n",
    "    portal_url=(os.environ.get('HERMES_AUTH_ADD_PORTAL_URL', '').strip() or None),\n",
    "    inference_url=(os.environ.get('HERMES_AUTH_ADD_INFERENCE_URL', '').strip() or None),\n",
    "    client_id=(os.environ.get('HERMES_AUTH_ADD_CLIENT_ID', '').strip() or None),\n",
    "    scope=(os.environ.get('HERMES_AUTH_ADD_SCOPE', '').strip() or None),\n",
    "    no_browser=(os.environ.get('HERMES_AUTH_ADD_NO_BROWSER', '0') == '1'),\n",
    "    timeout=(float(raw_timeout) if raw_timeout else None),\n",
    "    insecure=(os.environ.get('HERMES_AUTH_ADD_INSECURE', '0') == '1'),\n",
    "    ca_bundle=(os.environ.get('HERMES_AUTH_ADD_CA_BUNDLE', '').strip() or None),\n",
    ")\n",
    "auth_add_command(args)\n",
);

const AUTH_REMOVE_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from types import SimpleNamespace\n",
    "from hermes_cli.auth_commands import auth_remove_command\n",
    "args = SimpleNamespace(\n",
    "    provider=os.environ.get('HERMES_AUTH_REMOVE_PROVIDER', ''),\n",
    "    target=os.environ.get('HERMES_AUTH_REMOVE_TARGET', ''),\n",
    ")\n",
    "auth_remove_command(args)\n",
);

#[derive(Subcommand, Debug)]
pub enum AuthCommand {
    Add(AuthAddArgs),
    List { provider: Option<String> },
    Remove(AuthRemoveArgs),
    Reset { provider: String },
    Status { provider: String },
    Logout { provider: String },
    Spotify(SpotifyAuthArgs),
}

#[derive(Args, Debug)]
pub struct LogoutArgs {
    #[arg(long)]
    pub provider: Option<String>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct AuthAddArgs {
    pub provider: String,
    #[arg(long = "type", value_parser = ["oauth", "api-key", "api_key"])]
    pub auth_type: Option<String>,
    #[arg(long)]
    pub label: Option<String>,
    #[arg(long = "api-key")]
    pub api_key: Option<String>,
    #[arg(long = "portal-url")]
    pub portal_url: Option<String>,
    #[arg(long = "inference-url")]
    pub inference_url: Option<String>,
    #[arg(long = "client-id")]
    pub client_id: Option<String>,
    #[arg(long)]
    pub scope: Option<String>,
    #[arg(long = "no-browser", default_value_t = false)]
    pub no_browser: bool,
    #[arg(long)]
    pub timeout: Option<f64>,
    #[arg(long, default_value_t = false)]
    pub insecure: bool,
    #[arg(long = "ca-bundle")]
    pub ca_bundle: Option<String>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct AuthRemoveArgs {
    pub provider: String,
    pub target: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum, Default)]
pub enum SpotifyAuthAction {
    #[default]
    Login,
    Status,
    Logout,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SpotifyAuthArgs {
    #[arg(value_enum, default_value_t = SpotifyAuthAction::Login)]
    pub spotify_action: SpotifyAuthAction,
    #[arg(long = "client-id")]
    pub client_id: Option<String>,
    #[arg(long = "redirect-uri")]
    pub redirect_uri: Option<String>,
    #[arg(long)]
    pub scope: Option<String>,
    #[arg(long = "no-browser", default_value_t = false)]
    pub no_browser: bool,
    #[arg(long)]
    pub timeout: Option<f64>,
}

pub fn print_auth(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<AuthCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        Some(AuthCommand::Add(args)) => {
            print_auth_add(context, &args)?;
        }
        Some(AuthCommand::List { provider }) => {
            print_auth_list(context, provider.as_deref())?;
        }
        Some(AuthCommand::Remove(args)) => {
            print_auth_remove(context, &args)?;
        }
        Some(AuthCommand::Reset { provider }) => {
            print_auth_reset(context, &provider)?;
        }
        Some(AuthCommand::Status { provider }) => {
            let status = get_auth_status_summary(context.hermes_home().as_path(), &provider)?;
            println!(
                "{}",
                render_status(status, active_provider(context, loaded)?)
            );
        }
        Some(AuthCommand::Logout { provider }) => {
            logout_provider(context, loaded, Some(provider.as_str()))?;
        }
        Some(AuthCommand::Spotify(args)) => {
            print_auth_spotify(context, loaded, args)?;
        }
        None => {
            return Err(
                "auth requires a subcommand: add|list|remove|reset|status|logout|spotify".into(),
            );
        }
    }
    Ok(())
}

pub fn print_logout(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: LogoutArgs,
) -> Result<(), Box<dyn Error>> {
    logout_provider(context, loaded, args.provider.as_deref())
}

fn print_auth_add(context: &HermesContext, args: &AuthAddArgs) -> Result<(), Box<dyn Error>> {
    if let Some(custom_provider) =
        resolve_custom_provider_api_key_target(context.config_path().as_path(), args)?
    {
        return native_auth_add_custom_api_key(context, args, &custom_provider);
    }
    if should_use_native_auth_add(args) {
        return native_auth_add_api_key(context, args);
    }
    if should_use_native_anthropic_oauth_add(args) {
        return native_auth_add_anthropic_oauth(context, args);
    }
    if should_use_native_google_gemini_oauth_add(args) {
        return native_auth_add_google_gemini_oauth(context, args);
    }
    if should_use_native_minimax_oauth_add(args) {
        return native_auth_add_minimax_oauth(context, args);
    }
    if should_use_native_runtime_oauth_add(args) {
        return native_auth_add_runtime_oauth(context, args);
    }
    if should_use_native_codex_oauth_add(args) {
        return native_auth_add_openai_codex_oauth(context, args);
    }
    run_python_auth_add(args)
}

fn print_auth_list(
    context: &HermesContext,
    provider_filter: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let requested = provider_filter.and_then(normalize_provider_name);
    let pool = load_credential_pool(context.hermes_home().as_path())?;
    print!("{}", render_auth_list(&pool, requested.as_deref()));
    Ok(())
}

fn render_auth_list(
    pool: &std::collections::BTreeMap<String, Vec<PoolEntry>>,
    provider_filter: Option<&str>,
) -> String {
    let requested = provider_filter.and_then(normalize_provider_name);
    let mut providers = if let Some(provider) = requested {
        vec![provider]
    } else {
        let mut names = pool.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    };
    providers.retain(|provider| {
        pool.get(provider)
            .is_some_and(|entries| !entries.is_empty())
    });
    let mut lines = Vec::new();
    for (idx, provider) in providers.iter().enumerate() {
        let Some(entries) = pool.get(provider) else {
            continue;
        };
        let mut entries = entries.clone();
        entries.sort_by_key(|entry| entry.priority);
        let current_id = peek_entry_id(&entries);
        lines.push(format!("{provider} ({} credentials):", entries.len()));
        for (index, entry) in entries.iter().enumerate() {
            let marker = if current_id.as_deref() == Some(entry.id.as_str()) {
                "← "
            } else {
                "  "
            };
            let source = display_source(&entry.source);
            let status = format_exhausted_status(entry);
            lines.push(format!(
                "  #{}  {:<20} {:<7} {}{} {}",
                index + 1,
                entry.label,
                entry.auth_type,
                source,
                status,
                marker
            ));
        }
        if idx + 1 < providers.len() {
            lines.push(String::new());
        }
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

fn print_auth_reset(context: &HermesContext, provider: &str) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(provider)
        .ok_or("provider is required. Example: `hermes auth reset openrouter`.")?;
    let count = reset_auth_pool_statuses(context.hermes_home().as_path(), &provider)?;
    println!("Reset status on {count} {provider} credentials");
    Ok(())
}

fn print_auth_remove(context: &HermesContext, args: &AuthRemoveArgs) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth remove openrouter 1`.")?;
    let target = args.target.trim();
    if target.is_empty() {
        return Err("credential target is required".into());
    }
    let pool = load_credential_pool(context.hermes_home().as_path())?;
    let entries = pool.get(&provider).cloned().unwrap_or_default();
    let (index, matched, error) = resolve_auth_remove_target(&entries, target);
    let Some(entry) = matched else {
        return Err(format!(
            "{} Provider: {}.",
            error.unwrap_or("No credential target provided.".to_string()),
            provider
        )
        .into());
    };
    let Some(index) = index else {
        return Err(format!(
            "{} Provider: {}.",
            error.unwrap_or("No credential target provided.".to_string()),
            provider
        )
        .into());
    };
    if let Some(result) =
        try_native_auth_remove(context.hermes_home().as_path(), &provider, index, entry)?
    {
        println!(
            "Removed {} credential #{} ({})",
            provider, index, result.label
        );
        for line in result.cleaned {
            println!("{line}");
        }
        for line in result.hints {
            println!("{line}");
        }
        return Ok(());
    }
    if entry_source_is_manual(&entry.source) {
        remove_manual_auth_entry(context.hermes_home().as_path(), &provider, index)?;
        println!(
            "Removed {} credential #{} ({})",
            provider, index, entry.label
        );
        return Ok(());
    }
    run_python_auth_remove(&provider, target)
}

fn print_auth_spotify(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: SpotifyAuthArgs,
) -> Result<(), Box<dyn Error>> {
    match args.spotify_action {
        SpotifyAuthAction::Login => run_native_spotify_login(context, &args),
        SpotifyAuthAction::Status => {
            let status = get_auth_status_summary(context.hermes_home().as_path(), "spotify")?;
            println!(
                "{}",
                render_status(status, active_provider(context, loaded)?)
            );
            Ok(())
        }
        SpotifyAuthAction::Logout => logout_provider(context, loaded, Some("spotify")),
    }
}

pub(crate) fn run_default_spotify_login(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    run_native_spotify_login_with_io(context, &SpotifyAuthArgs::default(), input, output)
}

#[derive(Default)]
struct SpotifyCallbackResult {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

fn should_use_native_auth_add(args: &AuthAddArgs) -> bool {
    let provider = normalize_provider_name(&args.provider)
        .unwrap_or_else(|| normalize_provider_alias(&args.provider));
    let Some(profile) = get_provider_profile(&provider) else {
        return false;
    };
    if profile.name == "custom" || profile.auth_type != "api_key" {
        return false;
    }
    auth_add_uses_api_key_shape(args)
}

fn should_use_native_runtime_oauth_add(args: &AuthAddArgs) -> bool {
    let Some(provider) = normalize_provider_name(&args.provider) else {
        return false;
    };
    if !matches!(provider.as_str(), "qwen-oauth" | "nous" | "openai-codex") {
        return false;
    }
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") | None => {
            if provider == "nous" {
                return args.api_key.is_none() && !args.insecure && args.ca_bundle.is_none();
            }
            args.api_key.is_none()
                && args.portal_url.is_none()
                && args.inference_url.is_none()
                && args.client_id.is_none()
                && args.scope.is_none()
                && !args.no_browser
                && args.timeout.is_none()
                && !args.insecure
                && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn should_use_native_google_gemini_oauth_add(args: &AuthAddArgs) -> bool {
    let Some(provider) = normalize_provider_name(&args.provider) else {
        return false;
    };
    if provider != "google-gemini-cli" {
        return false;
    }
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") | None => {
            args.api_key.is_none()
                && args.portal_url.is_none()
                && args.inference_url.is_none()
                && args.client_id.is_none()
                && args.scope.is_none()
                && !args.insecure
                && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn should_use_native_minimax_oauth_add(args: &AuthAddArgs) -> bool {
    let Some(provider) = normalize_provider_name(&args.provider) else {
        return false;
    };
    if provider != "minimax-oauth" {
        return false;
    }
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") | None => {
            args.api_key.is_none() && !args.insecure && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn should_use_native_anthropic_oauth_add(args: &AuthAddArgs) -> bool {
    let Some(provider) = normalize_provider_name(&args.provider) else {
        return false;
    };
    if provider != "anthropic" {
        return false;
    }
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") | None => {
            args.api_key.is_none()
                && args.portal_url.is_none()
                && args.inference_url.is_none()
                && args.client_id.is_none()
                && args.scope.is_none()
                && !args.insecure
                && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn should_use_native_codex_oauth_add(args: &AuthAddArgs) -> bool {
    let Some(provider) = normalize_provider_name(&args.provider) else {
        return false;
    };
    if provider != "openai-codex" {
        return false;
    }
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") | None => {
            args.api_key.is_none()
                && args.portal_url.is_none()
                && args.inference_url.is_none()
                && args.client_id.is_none()
                && args.scope.is_none()
                && !args.insecure
                && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn native_auth_add_api_key(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add openrouter --api-key ...`.")?;
    let profile =
        get_provider_profile(&provider).ok_or_else(|| format!("Unknown provider: {provider}"))?;
    let api_key = args
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("No API key provided.")?;
    clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
    let mut auth_store = load_auth_store_json(context.hermes_home().as_path())?;
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let (entry_count, label) = {
        let root = auth_store
            .as_object_mut()
            .ok_or("auth store is not a JSON object")?;
        if !root.contains_key("version") {
            root.insert("version".to_string(), JsonValue::from(1));
        }
        if !root.contains_key("providers") {
            root.insert(
                "providers".to_string(),
                JsonValue::Object(Default::default()),
            );
        }
        let pool = ensure_json_object(root, "credential_pool")?;
        let provider_entries = pool
            .entry(provider.clone())
            .or_insert_with(|| JsonValue::Array(Vec::new()));
        let provider_entries = provider_entries
            .as_array_mut()
            .ok_or("credential_pool entry is not an array")?;
        let existing = provider_entries
            .iter()
            .filter_map(parse_pool_entry)
            .collect::<Vec<_>>();
        let next_index = existing.len() + 1;
        let priority = existing
            .iter()
            .map(|entry| entry.priority)
            .max()
            .unwrap_or(-1)
            + 1;
        let label = requested_label
            .clone()
            .unwrap_or_else(|| format!("api-key-{next_index}"));
        let mut entry = serde_json::Map::new();
        entry.insert("id".to_string(), JsonValue::String(generate_short_id()));
        entry.insert("label".to_string(), JsonValue::String(label.clone()));
        entry.insert(
            "auth_type".to_string(),
            JsonValue::String("api_key".to_string()),
        );
        entry.insert("priority".to_string(), JsonValue::from(priority));
        entry.insert(
            "source".to_string(),
            JsonValue::String("manual".to_string()),
        );
        entry.insert(
            "access_token".to_string(),
            JsonValue::String(api_key.to_string()),
        );
        if !profile.base_url.trim().is_empty() {
            entry.insert(
                "base_url".to_string(),
                JsonValue::String(profile.base_url.to_string()),
            );
        }
        provider_entries.push(JsonValue::Object(entry));
        (provider_entries.len(), label)
    };
    save_auth_store_json(context.hermes_home().as_path(), &auth_store)?;
    println!(
        "Added {} credential #{}: \"{}\"",
        provider, entry_count, label
    );
    Ok(())
}

fn native_auth_add_custom_api_key(
    context: &HermesContext,
    args: &AuthAddArgs,
    custom_provider: &ResolvedCustomProvider,
) -> Result<(), Box<dyn Error>> {
    let api_key = args
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("No API key provided.")?;
    clear_provider_suppressions(
        context.hermes_home().as_path(),
        custom_provider.pool_key.as_str(),
    )?;
    let mut auth_store = load_auth_store_json(context.hermes_home().as_path())?;
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let (entry_count, label) = {
        let root = auth_store
            .as_object_mut()
            .ok_or("auth store is not a JSON object")?;
        if !root.contains_key("version") {
            root.insert("version".to_string(), JsonValue::from(1));
        }
        if !root.contains_key("providers") {
            root.insert(
                "providers".to_string(),
                JsonValue::Object(Default::default()),
            );
        }
        let pool = ensure_json_object(root, "credential_pool")?;
        let provider_entries = pool
            .entry(custom_provider.pool_key.clone())
            .or_insert_with(|| JsonValue::Array(Vec::new()));
        let provider_entries = provider_entries
            .as_array_mut()
            .ok_or("credential_pool entry is not an array")?;
        let existing = provider_entries
            .iter()
            .filter_map(parse_pool_entry)
            .collect::<Vec<_>>();
        let next_index = existing.len() + 1;
        let priority = existing
            .iter()
            .map(|entry| entry.priority)
            .max()
            .unwrap_or(-1)
            + 1;
        let label = requested_label
            .clone()
            .unwrap_or_else(|| format!("api-key-{next_index}"));
        let mut entry = serde_json::Map::new();
        entry.insert("id".to_string(), JsonValue::String(generate_short_id()));
        entry.insert("label".to_string(), JsonValue::String(label.clone()));
        entry.insert(
            "auth_type".to_string(),
            JsonValue::String("api_key".to_string()),
        );
        entry.insert("priority".to_string(), JsonValue::from(priority));
        entry.insert(
            "source".to_string(),
            JsonValue::String("manual".to_string()),
        );
        entry.insert(
            "access_token".to_string(),
            JsonValue::String(api_key.to_string()),
        );
        if let Some(base_url) = custom_provider
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            entry.insert(
                "base_url".to_string(),
                JsonValue::String(base_url.trim_end_matches('/').to_string()),
            );
        }
        provider_entries.push(JsonValue::Object(entry));
        (provider_entries.len(), label)
    };
    save_auth_store_json(context.hermes_home().as_path(), &auth_store)?;
    println!(
        "Added {} credential #{}: \"{}\"",
        custom_provider.pool_key, entry_count, label
    );
    Ok(())
}

fn native_auth_add_anthropic_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    native_auth_add_anthropic_oauth_with_io(context, args, &mut input, &mut output)
}

fn native_auth_add_anthropic_oauth_with_io(
    context: &HermesContext,
    args: &AuthAddArgs,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add anthropic`.")?;
    if provider != "anthropic" {
        return run_python_auth_add(args);
    }
    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let code_verifier = anthropic_code_verifier()?;
    let code_challenge = spotify_code_challenge(&code_verifier);
    let authorize_url = anthropic_build_authorize_url(&code_verifier, &code_challenge)?;

    writeln!(output)?;
    writeln!(
        output,
        "Authorize Hermes with your Claude Pro/Max subscription."
    )?;
    writeln!(output)?;
    writeln!(output, "Open this URL to authorize Hermes:")?;
    writeln!(output, "{authorize_url}")?;
    writeln!(output)?;
    output.flush()?;

    if !args.no_browser && !is_remote_session() {
        if try_open_browser(&authorize_url)? {
            writeln!(output, "Browser opened for Anthropic authorization.")?;
        } else {
            writeln!(
                output,
                "Could not open the browser automatically; use the URL above."
            )?;
        }
        writeln!(output)?;
        output.flush()?;
    }

    writeln!(
        output,
        "After authorizing, you'll see a code. Paste it below."
    )?;
    output.flush()?;
    let raw_code = prompt_line(input, output, "Authorization code")?;
    let raw_code = raw_code.trim();
    if raw_code.is_empty() {
        return Err("No authorization code provided.".into());
    }
    let (code, returned_state) = anthropic_split_pasted_code(raw_code);
    let tokens = anthropic_exchange_code_for_tokens(
        code,
        returned_state,
        &code_verifier,
        args.timeout.unwrap_or(15.0).max(1.0),
    )?;

    clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
    let label =
        requested_label.unwrap_or_else(|| label_from_token(&tokens.access_token, &default_label));
    let count = add_auth_pool_entry(
        context.hermes_home().as_path(),
        &provider,
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source: "manual:hermes_pkce".to_string(),
            access_token: tokens.access_token,
            refresh_token: Some(tokens.refresh_token),
            base_url: get_provider_profile("anthropic")
                .map(|profile| profile.base_url.trim().to_string())
                .filter(|value| !value.is_empty()),
            expires_at_ms: Some(tokens.expires_at_ms),
            last_refresh: None,
        },
    )?;
    writeln!(
        output,
        "Added anthropic OAuth credential #{}: \"{}\"",
        count, label
    )?;
    output.flush()?;
    Ok(())
}

fn native_auth_add_google_gemini_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    native_auth_add_google_gemini_oauth_with_io(context, args, &mut input, &mut output)
}

fn native_auth_add_google_gemini_oauth_with_io(
    context: &HermesContext,
    args: &AuthAddArgs,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add google-gemini-cli`.")?;
    if provider != "google-gemini-cli" {
        return run_python_auth_add(args);
    }

    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    if google_gemini_oauth_path(context.hermes_home().as_path()).exists()
        && let Ok(creds) =
            resolve_google_gemini_runtime_credentials(context.hermes_home().as_path())
    {
        return finalize_google_gemini_auth_add(
            context,
            requested_label.as_deref(),
            &default_label,
            &creds.access_token,
            Some(&creds.refresh_token),
            None,
            creds.email.as_str(),
        );
    }

    let code_verifier = google_code_verifier()?;
    let code_challenge = spotify_code_challenge(&code_verifier);
    let state_nonce = google_state_nonce()?;
    let client_id = google_client_id();
    let client_secret = google_client_secret();
    let callback_timeout_seconds = validated_timeout_seconds(args.timeout, 300.0)?;
    let request_timeout_seconds = args.timeout.unwrap_or(20.0).max(1.0);

    let (callback_result, redirect_uri) = if args.no_browser || is_remote_session() {
        let redirect_uri = google_default_redirect_uri();
        let authorize_url =
            google_build_authorize_url(&client_id, &redirect_uri, &state_nonce, &code_challenge)?;
        writeln!(output)?;
        writeln!(
            output,
            "Open this URL to authorize Hermes with Google Gemini CLI:"
        )?;
        writeln!(output, "{authorize_url}")?;
        writeln!(output)?;
        writeln!(
            output,
            "After signing in, paste the full callback URL or just the code."
        )?;
        output.flush()?;
        (google_prompt_pasted_callback(input, output)?, redirect_uri)
    } else {
        let (listener, redirect_uri) = google_bind_callback_listener()?;
        let authorize_url =
            google_build_authorize_url(&client_id, &redirect_uri, &state_nonce, &code_challenge)?;
        writeln!(output)?;
        writeln!(
            output,
            "Opening your browser to sign in to Google Gemini CLI..."
        )?;
        writeln!(
            output,
            "If it does not open automatically, visit:\n  {authorize_url}"
        )?;
        writeln!(output)?;
        output.flush()?;
        if try_open_browser(&authorize_url)? {
            writeln!(output, "Browser opened for Google authorization.")?;
        } else {
            writeln!(
                output,
                "Could not open the browser automatically; use the URL above."
            )?;
        }
        writeln!(output)?;
        output.flush()?;

        let callback = match google_wait_for_callback(listener, callback_timeout_seconds)? {
            Some(callback) => callback,
            None => {
                writeln!(
                    output,
                    "Timed out waiting for the local callback. Paste the callback URL or code below."
                )?;
                output.flush()?;
                google_prompt_pasted_callback(input, output)?
            }
        };
        (callback, redirect_uri)
    };

    if let Some(error) = callback_result.error {
        let detail = callback_result.error_description.unwrap_or(error);
        return Err(format!("Google authorization failed: {detail}").into());
    }
    if callback_result.state.as_deref() != Some(state_nonce.as_str()) {
        return Err("Google authorization failed: state mismatch.".into());
    }
    let code = callback_result
        .code
        .ok_or("Google authorization failed: missing authorization code.")?;
    let token_payload = google_exchange_code_for_tokens(
        &client_id,
        &client_secret,
        &code,
        &redirect_uri,
        &code_verifier,
        request_timeout_seconds,
    )?;
    let access_token = json_string(&token_payload, "access_token")
        .ok_or("Google token response did not include an access_token.")?;
    let refresh_token = json_string(&token_payload, "refresh_token")
        .ok_or("Google token response did not include a refresh_token.")?;
    let expires_in = token_payload
        .get("expires_in")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0)
        .max(0);
    let email = google_fetch_user_email(access_token, request_timeout_seconds).unwrap_or_default();
    save_google_gemini_oauth_file(
        context.hermes_home().as_path(),
        access_token,
        refresh_token,
        expires_in,
        &email,
    )?;
    finalize_google_gemini_auth_add(
        context,
        requested_label.as_deref(),
        &default_label,
        access_token,
        Some(refresh_token),
        Some(expires_in),
        &email,
    )
}

fn native_auth_add_minimax_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    native_auth_add_minimax_oauth_with_io(context, args, &mut output)
}

fn native_auth_add_minimax_oauth_with_io(
    context: &HermesContext,
    args: &AuthAddArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add minimax-oauth`.")?;
    if provider != "minimax-oauth" {
        return run_python_auth_add(args);
    }

    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    if provider_state_exists(context.hermes_home().as_path(), &provider)?
        && !has_explicit_minimax_runtime_overrides(args)
    {
        let creds = resolve_minimax_oauth_runtime_credentials(context.hermes_home().as_path())?;
        let label = requested_label
            .unwrap_or_else(|| label_from_token(&creds.access_token, &default_label));
        clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
        let stored_count = add_auth_pool_entry(
            context.hermes_home().as_path(),
            &provider,
            NewPoolEntry {
                label: label.clone(),
                auth_type: "oauth".to_string(),
                source: "manual:minimax_oauth".to_string(),
                access_token: creds.access_token,
                refresh_token: None,
                base_url: non_empty_trimmed_owned(&creds.base_url),
                expires_at_ms: None,
                last_refresh: None,
            },
        )?;
        writeln!(
            output,
            "Added minimax-oauth OAuth credential #{}: \"{}\"",
            stored_count, label
        )?;
        output.flush()?;
        return Ok(());
    }

    let timeout_seconds = validated_timeout_seconds(args.timeout, 15.0)?;
    let portal_base_url = resolve_minimax_portal_base_url(args)?;
    let inference_base_url = resolve_minimax_inference_base_url(args, &portal_base_url)?;
    let client_id = resolve_minimax_client_id(args);
    let scope = resolve_minimax_scope(args);
    let code_verifier = minimax_code_verifier()?;
    let code_challenge = spotify_code_challenge(&code_verifier);
    let state_nonce = minimax_state_nonce()?;
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()?;

    writeln!(output, "Starting Hermes login via MiniMax OAuth...")?;
    writeln!(output, "Portal: {portal_base_url}")?;

    let code_data = minimax_request_user_code(
        &client,
        &portal_base_url,
        &client_id,
        &scope,
        &code_challenge,
        &state_nonce,
    )?;
    let verification_url = json_string(&code_data, "verification_uri")
        .ok_or("MiniMax OAuth response missing verification_uri.")?
        .to_string();
    let user_code = json_string(&code_data, "user_code")
        .ok_or("MiniMax OAuth response missing user_code.")?
        .to_string();
    let expired_in = json_i64(code_data.get("expired_in"))
        .filter(|value| *value > 0)
        .ok_or("MiniMax OAuth response missing expired_in.")?;
    let interval_ms = json_i64(code_data.get("interval"));

    writeln!(output)?;
    writeln!(output, "To continue:")?;
    writeln!(output, "  1. Open: {verification_url}")?;
    writeln!(output, "  2. If prompted, enter code: {user_code}")?;
    if !args.no_browser && !is_remote_session() {
        if try_open_browser(&verification_url)? {
            writeln!(output, "  (Opened browser for verification)")?;
        } else {
            writeln!(
                output,
                "  Could not open browser automatically -- use the URL above."
            )?;
        }
    }
    writeln!(output, "Waiting for approval...")?;
    output.flush()?;

    let token_data = minimax_poll_token(
        &client,
        &portal_base_url,
        &client_id,
        &user_code,
        &code_verifier,
        expired_in,
        interval_ms,
    )?;
    let access_token = json_string(&token_data, "access_token")
        .ok_or("MiniMax OAuth token payload missing access_token.")?
        .to_string();
    let refresh_token = json_string(&token_data, "refresh_token")
        .ok_or("MiniMax OAuth token payload missing refresh_token.")?
        .to_string();
    let expires_in = json_i64(token_data.get("expired_in"))
        .filter(|value| *value > 0)
        .ok_or("MiniMax OAuth token payload missing expired_in.")?;
    let token_type = json_string(&token_data, "token_type").map(ToOwned::to_owned);
    let resource_url = json_string(&token_data, "resource_url").map(ToOwned::to_owned);
    save_minimax_provider_state(
        context.hermes_home().as_path(),
        &portal_base_url,
        &inference_base_url,
        &client_id,
        &scope,
        &access_token,
        &refresh_token,
        expires_in,
        token_type.as_deref(),
        resource_url.as_deref(),
    )?;

    clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
    let label = requested_label.unwrap_or_else(|| label_from_token(&access_token, &default_label));
    let expires_at_ms = Some(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_millis() as i64)
            .unwrap_or(0)
            .saturating_add(expires_in.saturating_mul(1000)),
    );
    let stored_count = add_auth_pool_entry(
        context.hermes_home().as_path(),
        &provider,
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source: "manual:minimax_oauth".to_string(),
            access_token,
            refresh_token: Some(refresh_token),
            base_url: Some(inference_base_url),
            expires_at_ms,
            last_refresh: None,
        },
    )?;
    writeln!(
        output,
        "Added minimax-oauth OAuth credential #{}: \"{}\"",
        stored_count, label
    )?;
    if let Some(message) = json_string(&token_data, "notification_message")
        && !message.trim().is_empty()
    {
        writeln!(output, "Note from MiniMax: {}", message.trim())?;
    }
    output.flush()?;
    Ok(())
}

fn native_auth_add_openai_codex_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    native_auth_add_openai_codex_oauth_with_io(context, args, &mut output)
}

fn native_auth_add_openai_codex_oauth_with_io(
    context: &HermesContext,
    args: &AuthAddArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add openai-codex`.")?;
    if provider != "openai-codex" {
        return run_python_auth_add(args);
    }

    let issuer = codex_oauth_issuer();
    let device_url = format!("{}/codex/device", issuer.trim_end_matches('/'));
    let usercode_endpoint = format!(
        "{}/api/accounts/deviceauth/usercode",
        issuer.trim_end_matches('/')
    );
    let poll_endpoint = format!(
        "{}/api/accounts/deviceauth/token",
        issuer.trim_end_matches('/')
    );
    let redirect_uri = format!("{}/deviceauth/callback", issuer.trim_end_matches('/'));
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(
            args.timeout.unwrap_or(15.0).max(1.0),
        ))
        .build()?;
    let device_response = client
        .post(usercode_endpoint)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }))
        .send()
        .map_err(|error| format!("Failed to request Codex device code: {error}"))?;
    if device_response.status().as_u16() != 200 {
        return Err(format!(
            "Codex device code request returned status {}.",
            device_response.status().as_u16()
        )
        .into());
    }
    let device_payload: JsonValue = device_response.json()?;
    let device_payload = device_payload
        .as_object()
        .ok_or("Codex device code response was not a JSON object.")?;
    let user_code = json_string(device_payload, "user_code")
        .ok_or("Codex device code response missing user_code.")?
        .to_string();
    let device_auth_id = json_string(device_payload, "device_auth_id")
        .ok_or("Codex device code response missing device_auth_id.")?
        .to_string();
    let poll_interval = device_payload
        .get("interval")
        .and_then(JsonValue::as_i64)
        .unwrap_or(5)
        .max(1) as u64;

    writeln!(output, "Signing in to OpenAI Codex...")?;
    writeln!(
        output,
        "(Hermes creates its own session — won't affect Codex CLI or VS Code)"
    )?;
    writeln!(output)?;
    writeln!(output, "To continue, follow these steps:")?;
    writeln!(output)?;
    writeln!(output, "  1. Open this URL in your browser:")?;
    writeln!(output, "     {device_url}")?;
    writeln!(output)?;
    writeln!(output, "  2. Enter this code:")?;
    writeln!(output, "     {user_code}")?;
    writeln!(output)?;
    writeln!(output, "Waiting for sign-in... (press Ctrl+C to cancel)")?;
    output.flush()?;

    let started = std::time::Instant::now();
    let max_wait = Duration::from_secs_f64(codex_oauth_max_wait_seconds());
    let (authorization_code, code_verifier) = loop {
        let poll_response = client
            .post(&poll_endpoint)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            }))
            .send()
            .map_err(|error| format!("Codex device auth polling failed: {error}"))?;
        match poll_response.status().as_u16() {
            200 => {
                let payload: JsonValue = poll_response.json()?;
                let payload = payload
                    .as_object()
                    .ok_or("Codex device auth poll response was not a JSON object.")?;
                let authorization_code = json_string(payload, "authorization_code")
                    .map(ToOwned::to_owned)
                    .ok_or("Codex device auth response missing authorization_code.")?;
                let code_verifier = json_string(payload, "code_verifier")
                    .map(ToOwned::to_owned)
                    .ok_or("Codex device auth response missing code_verifier.")?;
                break (authorization_code, code_verifier);
            }
            403 | 404 => {
                if started.elapsed() >= max_wait {
                    return Err("Codex login timed out after waiting for device approval.".into());
                }
                thread::sleep(Duration::from_secs(poll_interval));
            }
            status => {
                return Err(format!("Codex device auth polling returned status {status}.").into());
            }
        }
    };
    let token_response = client
        .post(codex_oauth_token_url())
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", authorization_code.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("client_id", CODEX_OAUTH_CLIENT_ID),
            ("code_verifier", code_verifier.as_str()),
        ])
        .send()
        .map_err(|error| format!("Codex token exchange failed: {error}"))?;
    if token_response.status().as_u16() != 200 {
        return Err(format!(
            "Codex token exchange returned status {}.",
            token_response.status().as_u16()
        )
        .into());
    }
    let token_payload: JsonValue = token_response.json()?;
    let token_payload = token_payload
        .as_object()
        .ok_or("Codex token exchange response was not a JSON object.")?;
    let access_token = json_string(token_payload, "access_token")
        .ok_or("Codex token exchange did not return an access_token.")?
        .to_string();
    let refresh_token = json_string(token_payload, "refresh_token").map(ToOwned::to_owned);
    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| label_from_token(&access_token, &default_label));
    clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
    let stored_count = add_auth_pool_entry(
        context.hermes_home().as_path(),
        &provider,
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source: "manual:device_code".to_string(),
            access_token,
            refresh_token,
            base_url: Some(codex_base_url()),
            expires_at_ms: None,
            last_refresh: Some(codex_now_rfc3339()),
        },
    )?;
    writeln!(
        output,
        "Added openai-codex OAuth credential #{}: \"{}\"",
        stored_count, label
    )?;
    output.flush()?;
    Ok(())
}

fn native_auth_add_runtime_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add qwen-oauth`.")?;
    if provider == "nous" {
        return native_auth_add_nous_oauth(context, args);
    }
    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    let (source, access_token, refresh_token, base_url, derived_label) = match provider.as_str() {
        "qwen-oauth" => {
            let creds = resolve_qwen_runtime_credentials()?;
            let label = label_from_token(&creds.access_token, &default_label);
            (
                "manual:qwen_cli".to_string(),
                creds.access_token,
                None,
                non_empty_trimmed_owned(&creds.base_url),
                label,
            )
        }
        "nous" => {
            if !provider_state_exists(context.hermes_home().as_path(), "nous")? {
                return run_python_auth_add(args);
            }
            let creds =
                resolve_nous_runtime_credentials(context.hermes_home().as_path(), 300, 15.0)?;
            let mut auth_store = load_auth_store_json(context.hermes_home().as_path())?;
            let mut state = provider_state_json(&auth_store, "nous")
                .ok_or("Nous auth state is missing after runtime resolution.")?;
            if let Some(label) = requested_label.as_deref() {
                state.insert("label".to_string(), JsonValue::String(label.to_string()));
                store_provider_state(&mut auth_store, "nous", state.clone())?;
                save_auth_store_json(context.hermes_home().as_path(), &auth_store)?;
            }
            let access_token = json_string(&state, "access_token")
                .ok_or("Nous auth state is missing access_token after runtime resolution.")?
                .to_string();
            let label = json_string(&state, "label")
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| label_from_token(&access_token, &default_label));
            (
                "device_code".to_string(),
                access_token,
                json_string(&state, "refresh_token").map(ToOwned::to_owned),
                non_empty_trimmed_owned(&creds.base_url),
                label,
            )
        }
        "openai-codex" => {
            if provider_pool_has_entries(context.hermes_home().as_path(), "openai-codex")?
                || !provider_state_exists(context.hermes_home().as_path(), "openai-codex")?
            {
                return native_auth_add_openai_codex_oauth(context, args);
            }
            let access_token = resolve_codex_access_token(context.hermes_home().as_path())?;
            let auth_store = load_auth_store_json(context.hermes_home().as_path())?;
            let state = provider_state_json(&auth_store, "openai-codex")
                .ok_or("Codex auth state is missing after runtime resolution.")?;
            let tokens = state
                .get("tokens")
                .and_then(JsonValue::as_object)
                .ok_or("Codex auth state is missing tokens after runtime resolution.")?;
            let label = label_from_token(&access_token, &default_label);
            (
                "device_code".to_string(),
                access_token,
                json_string(tokens, "refresh_token").map(ToOwned::to_owned),
                get_provider_profile("openai-codex")
                    .map(|profile| profile.base_url.trim().to_string())
                    .filter(|value| !value.is_empty()),
                label,
            )
        }
        _ => return run_python_auth_add(args),
    };
    clear_provider_suppressions(context.hermes_home().as_path(), &provider)?;
    let label = requested_label.unwrap_or(derived_label);
    let stored_count = add_auth_pool_entry(
        context.hermes_home().as_path(),
        &provider,
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source,
            access_token,
            refresh_token,
            base_url,
            expires_at_ms: None,
            last_refresh: None,
        },
    )?;
    println!(
        "Added {} OAuth credential #{}: \"{}\"",
        provider, stored_count, label
    );
    Ok(())
}

fn native_auth_add_nous_oauth(
    context: &HermesContext,
    args: &AuthAddArgs,
) -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    native_auth_add_nous_oauth_with_io(context, args, &mut output)
}

fn native_auth_add_nous_oauth_with_io(
    context: &HermesContext,
    args: &AuthAddArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let provider = normalize_provider_name(&args.provider)
        .ok_or("provider is required. Example: `hermes auth add nous`.")?;
    if provider != "nous" {
        return run_python_auth_add(args);
    }

    let timeout_seconds = validated_timeout_seconds(args.timeout, 15.0)?;
    let next_index = next_provider_entry_index(context.hermes_home().as_path(), &provider)?;
    let default_label = oauth_default_label(&provider, next_index);
    let requested_label = args
        .label
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    if provider_state_exists(context.hermes_home().as_path(), "nous")? {
        if has_explicit_nous_runtime_overrides(args) {
            let auth_store = load_auth_store_json(context.hermes_home().as_path())?;
            let mut state = provider_state_json(&auth_store, "nous")
                .ok_or("Nous auth state is missing before runtime update.")?;
            state.insert(
                "portal_base_url".to_string(),
                JsonValue::String(resolve_nous_portal_base_url(args)?),
            );
            state.insert(
                "inference_base_url".to_string(),
                JsonValue::String(resolve_nous_inference_base_url(args)?),
            );
            state.insert(
                "client_id".to_string(),
                JsonValue::String(resolve_nous_client_id(args)),
            );
            state.insert(
                "scope".to_string(),
                JsonValue::String(resolve_nous_scope(args)),
            );
            if let Some(label) = requested_label.as_deref() {
                state.insert("label".to_string(), JsonValue::String(label.to_string()));
            }
            let _ = auth_store;
            seed_nous_provider_state_and_resolve(
                context.hermes_home().as_path(),
                state,
                timeout_seconds,
            )?;
        } else {
            resolve_nous_runtime_credentials(
                context.hermes_home().as_path(),
                300,
                timeout_seconds,
            )?;
        }
        let (stored_count, label) = finalize_nous_auth_add(
            context.hermes_home().as_path(),
            requested_label.as_deref(),
            &default_label,
        )?;
        writeln!(
            output,
            "Added nous OAuth credential #{}: \"{}\"",
            stored_count, label
        )?;
        output.flush()?;
        return Ok(());
    }

    if let Some(shared) = read_shared_nous_state() {
        let path = nous_shared_store_path();
        writeln!(output)?;
        if path.exists() {
            writeln!(
                output,
                "Found existing Nous OAuth credentials at {}",
                path.display()
            )?;
        } else {
            writeln!(output, "Found existing shared Nous OAuth credentials")?;
        }
        writeln!(
            output,
            "Rehydrating Nous session from shared credentials..."
        )?;
        output.flush()?;
        let import_state = build_nous_shared_import_state(&shared);
        match seed_nous_provider_state_and_resolve(
            context.hermes_home().as_path(),
            import_state,
            timeout_seconds,
        ) {
            Ok(()) => {
                let (stored_count, label) = finalize_nous_auth_add(
                    context.hermes_home().as_path(),
                    requested_label.as_deref(),
                    &default_label,
                )?;
                writeln!(
                    output,
                    "Imported nous OAuth credentials #{}: \"{}\"",
                    stored_count, label
                )?;
                output.flush()?;
                return Ok(());
            }
            Err(_) => {
                writeln!(
                    output,
                    "Could not refresh shared credentials — falling back to device-code login."
                )?;
                output.flush()?;
            }
        }
    }

    let portal_base_url = resolve_nous_portal_base_url(args)?;
    let requested_inference_url = resolve_nous_inference_base_url(args)?;
    let client_id = resolve_nous_client_id(args);
    let scope = resolve_nous_scope(args);
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds))
        .build()?;

    let mut device_form = vec![("client_id".to_string(), client_id.clone())];
    if !scope.trim().is_empty() {
        device_form.push(("scope".to_string(), scope.clone()));
    }
    let device_response = client
        .post(format!(
            "{}/api/oauth/device/code",
            portal_base_url.trim_end_matches('/')
        ))
        .header("Accept", "application/json")
        .form(&device_form)
        .send()
        .map_err(|error| format!("Failed to request Nous device code: {error}"))?;
    let device_status = device_response.status();
    let device_body = device_response.text()?;
    if !device_status.is_success() {
        return Err(format!(
            "Nous device code request returned status {}.",
            device_status.as_u16()
        )
        .into());
    }
    let device_payload = serde_json::from_str::<JsonValue>(&device_body)?;
    let device_payload = device_payload
        .as_object()
        .ok_or("Nous device code response was not a JSON object.")?;
    let device_code = json_string(device_payload, "device_code")
        .ok_or("Nous device code response missing device_code.")?
        .to_string();
    let user_code = json_string(device_payload, "user_code")
        .ok_or("Nous device code response missing user_code.")?
        .to_string();
    let verification_url = json_string(device_payload, "verification_uri_complete")
        .or_else(|| json_string(device_payload, "verification_uri"))
        .ok_or("Nous device code response missing verification_uri.")?
        .to_string();
    let expires_in = json_i64(device_payload.get("expires_in"))
        .unwrap_or(0)
        .max(1);
    let mut poll_interval = json_i64(device_payload.get("interval"))
        .unwrap_or(5)
        .clamp(1, 30);

    writeln!(output, "Starting Hermes login via Nous...")?;
    writeln!(output, "Portal: {portal_base_url}")?;
    writeln!(output)?;
    writeln!(output, "To continue:")?;
    writeln!(output, "  1. Open: {verification_url}")?;
    writeln!(output, "  2. If prompted, enter code: {user_code}")?;
    if !args.no_browser && !is_remote_session() {
        if try_open_browser(&verification_url)? {
            writeln!(output, "  (Opened browser for verification)")?;
        } else {
            writeln!(
                output,
                "  Could not open browser automatically — use the URL above."
            )?;
        }
    }
    writeln!(
        output,
        "Waiting for approval (polling every {}s)...",
        poll_interval
    )?;
    output.flush()?;

    let started = std::time::Instant::now();
    let token_payload = loop {
        if started.elapsed().as_secs() >= expires_in as u64 {
            return Err("Timed out waiting for device authorization.".into());
        }
        let token_response = client
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
            .map_err(|error| format!("Nous device auth polling failed: {error}"))?;
        let status = token_response.status();
        let body = token_response.text()?;
        let payload = serde_json::from_str::<JsonValue>(&body).unwrap_or(JsonValue::Null);
        if status.is_success() {
            let payload = payload
                .as_object()
                .ok_or("Nous token response was not a JSON object.")?
                .clone();
            if json_string(&payload, "access_token").is_none() {
                return Err("Nous token response missing access_token.".into());
            }
            break payload;
        }
        let Some(error_payload) = payload.as_object() else {
            return Err(format!("Nous token exchange returned status {}.", status.as_u16()).into());
        };
        let error_code = json_string(error_payload, "error").unwrap_or_default();
        if error_code == "authorization_pending" {
            thread::sleep(Duration::from_secs(poll_interval as u64));
            continue;
        }
        if error_code == "slow_down" {
            poll_interval = (poll_interval + 1).min(30);
            thread::sleep(Duration::from_secs(poll_interval as u64));
            continue;
        }
        let description = json_string(error_payload, "error_description")
            .unwrap_or("Unknown authentication error");
        if !error_code.is_empty() {
            return Err(format!("{error_code}: {description}").into());
        }
        return Err(format!("Nous token exchange returned status {}.", status.as_u16()).into());
    };

    let now = Utc::now();
    let token_expires_in = json_i64(token_payload.get("expires_in"))
        .unwrap_or(0)
        .max(0);
    let mut resolved_inference_url = requested_inference_url.clone();
    if let Some(portal_inference_url) = token_payload
        .get("inference_base_url")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        resolved_inference_url = normalize_http_url(portal_inference_url, "inference URL")?;
        if resolved_inference_url != requested_inference_url {
            writeln!(
                output,
                "Using portal-provided inference URL: {resolved_inference_url}"
            )?;
            output.flush()?;
        }
    }

    let mut state = JsonMap::new();
    state.insert(
        "portal_base_url".to_string(),
        JsonValue::String(portal_base_url.clone()),
    );
    state.insert(
        "inference_base_url".to_string(),
        JsonValue::String(resolved_inference_url),
    );
    state.insert("client_id".to_string(), JsonValue::String(client_id));
    state.insert(
        "scope".to_string(),
        JsonValue::String(
            json_string(&token_payload, "scope")
                .unwrap_or(scope.as_str())
                .to_string(),
        ),
    );
    state.insert(
        "token_type".to_string(),
        JsonValue::String(
            json_string(&token_payload, "token_type")
                .unwrap_or("Bearer")
                .to_string(),
        ),
    );
    state.insert(
        "access_token".to_string(),
        JsonValue::String(
            json_string(&token_payload, "access_token")
                .ok_or("Nous token response missing access_token.")?
                .to_string(),
        ),
    );
    if let Some(refresh_token) = json_string(&token_payload, "refresh_token") {
        state.insert(
            "refresh_token".to_string(),
            JsonValue::String(refresh_token.to_string()),
        );
    }
    state.insert(
        "obtained_at".to_string(),
        JsonValue::String(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    state.insert(
        "expires_at".to_string(),
        JsonValue::String(
            (now + ChronoDuration::seconds(token_expires_in))
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        ),
    );
    state.insert("expires_in".to_string(), JsonValue::from(token_expires_in));
    state.insert("agent_key".to_string(), JsonValue::Null);
    state.insert("agent_key_id".to_string(), JsonValue::Null);
    state.insert("agent_key_expires_at".to_string(), JsonValue::Null);
    state.insert("agent_key_expires_in".to_string(), JsonValue::Null);
    state.insert("agent_key_reused".to_string(), JsonValue::Null);
    state.insert("agent_key_obtained_at".to_string(), JsonValue::Null);
    if let Some(label) = requested_label.as_deref() {
        state.insert("label".to_string(), JsonValue::String(label.to_string()));
    }

    seed_nous_provider_state_and_resolve(context.hermes_home().as_path(), state, timeout_seconds)?;
    let (stored_count, label) = finalize_nous_auth_add(
        context.hermes_home().as_path(),
        requested_label.as_deref(),
        &default_label,
    )?;
    writeln!(
        output,
        "Added nous OAuth credential #{}: \"{}\"",
        stored_count, label
    )?;
    output.flush()?;
    Ok(())
}

fn run_python_auth_add(args: &AuthAddArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_AUTH_PYTHON"))
        .ok_or("could not find a Python interpreter for auth")?;
    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_AUTH_ADD_PROVIDER", args.provider.trim())
        .env(
            "HERMES_AUTH_ADD_TYPE",
            args.auth_type.as_deref().unwrap_or(""),
        )
        .env("HERMES_AUTH_ADD_LABEL", args.label.as_deref().unwrap_or(""))
        .env(
            "HERMES_AUTH_ADD_API_KEY",
            args.api_key.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_ADD_PORTAL_URL",
            args.portal_url.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_ADD_INFERENCE_URL",
            args.inference_url.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_ADD_CLIENT_ID",
            args.client_id.as_deref().unwrap_or(""),
        )
        .env("HERMES_AUTH_ADD_SCOPE", args.scope.as_deref().unwrap_or(""))
        .env(
            "HERMES_AUTH_ADD_NO_BROWSER",
            if args.no_browser { "1" } else { "0" },
        )
        .env(
            "HERMES_AUTH_ADD_TIMEOUT",
            args.timeout
                .map(|value| value.to_string())
                .unwrap_or_default(),
        )
        .env(
            "HERMES_AUTH_ADD_INSECURE",
            if args.insecure { "1" } else { "0" },
        )
        .env(
            "HERMES_AUTH_ADD_CA_BUNDLE",
            args.ca_bundle.as_deref().unwrap_or(""),
        )
        .arg("-c")
        .arg(AUTH_ADD_BOOTSTRAP);
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("auth add", status).into())
}

fn run_python_auth_remove(provider: &str, target: &str) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_AUTH_PYTHON"))
        .ok_or("could not find a Python interpreter for auth")?;
    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_AUTH_REMOVE_PROVIDER", provider)
        .env("HERMES_AUTH_REMOVE_TARGET", target)
        .arg("-c")
        .arg(AUTH_REMOVE_BOOTSTRAP);
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("auth remove", status).into())
}

struct NativeAuthRemoveResult {
    label: String,
    cleaned: Vec<String>,
    hints: Vec<String>,
}

type NativeRemovePlan = (bool, Vec<String>, Vec<String>, Vec<String>);

fn try_native_auth_remove(
    hermes_home: &Path,
    provider: &str,
    index: usize,
    entry: &PoolEntry,
) -> Result<Option<NativeAuthRemoveResult>, Box<dyn Error>> {
    let source = entry.source.trim();
    let (clear_provider_state, suppress_sources, cleaned, hints) = match provider {
        "copilot" if source == "gh_cli" || source.starts_with("env:") => (
            false,
            vec![
                "gh_cli".to_string(),
                "env:COPILOT_GITHUB_TOKEN".to_string(),
                "env:GH_TOKEN".to_string(),
                "env:GITHUB_TOKEN".to_string(),
            ],
            Vec::new(),
            vec![
                "Suppressed all copilot token sources (gh_cli + env vars) — they will not be re-seeded.".to_string(),
                "Note: Your gh CLI / shell environment is unchanged.".to_string(),
                "Run `hermes auth add copilot` to re-enable if needed.".to_string(),
            ],
        ),
        "anthropic" if source == "claude_code" => (
            false,
            vec![source.to_string()],
            Vec::new(),
            vec![
                "Suppressed claude_code credential — it will not be re-seeded.".to_string(),
                "Note: Claude Code credentials still live in ~/.claude/.credentials.json".to_string(),
                "Run `hermes auth add anthropic` to re-enable if needed.".to_string(),
            ],
        ),
        "anthropic" if source == "hermes_pkce" => (
            false,
            vec![source.to_string()],
            remove_hermes_pkce_file(hermes_home)?,
            Vec::new(),
        ),
        "nous" if source == "device_code" => (
            true,
            vec![source.to_string()],
            vec![format!("Cleared {provider} OAuth tokens from auth store")],
            Vec::new(),
        ),
        "openai-codex" if source == "device_code" || source.ends_with(":device_code") => {
            let mut suppress_sources = vec!["device_code".to_string()];
            if source != "device_code" {
                suppress_sources.push(source.to_string());
            }
            (
                true,
                suppress_sources,
                vec![format!("Cleared {provider} OAuth tokens from auth store")],
                vec![
                    "Suppressed openai-codex device_code source — it will not be re-seeded."
                        .to_string(),
                    "Note: Codex CLI credentials still live in ~/.codex/auth.json".to_string(),
                    "Run `hermes auth add openai-codex` to re-enable if needed.".to_string(),
                ],
            )
        }
        "qwen-oauth" if source == "qwen-cli" => (
            false,
            vec![source.to_string()],
            Vec::new(),
            vec![
                "Suppressed qwen-cli credential — it will not be re-seeded.".to_string(),
                "Note: Qwen CLI credentials still live in ~/.qwen/oauth_creds.json".to_string(),
                "Run `hermes auth add qwen-oauth` to re-enable if needed.".to_string(),
            ],
        ),
        "minimax-oauth" if source == "oauth" || source == "manual:minimax_oauth" => (
            true,
            vec![source.to_string()],
            vec![format!("Cleared {provider} OAuth tokens from auth store")],
            Vec::new(),
        ),
        "google-gemini-cli" if source == "manual:google_pkce" => (
            true,
            vec![source.to_string()],
            remove_google_gemini_oauth_file(hermes_home)?,
            vec![
                "Suppressed Google Gemini CLI credential — it will not be re-added automatically."
                    .to_string(),
                "Run `hermes auth add google-gemini-cli` to re-enable if needed.".to_string(),
            ],
        ),
        _ if source.starts_with("env:") => remove_env_source_native(hermes_home, provider, source)?,
        _ if source.starts_with("config:") || source == "model_config" => (
            false,
            vec![source.to_string()],
            Vec::new(),
            vec![
                format!("Suppressed {source} — it will not be re-seeded."),
                "Note: The underlying value in config.yaml is unchanged.  Edit it directly if you want to remove the credential from disk.".to_string(),
            ],
        ),
        _ => return Ok(None),
    };

    apply_native_auth_remove_changes(
        hermes_home,
        provider,
        index,
        clear_provider_state,
        &suppress_sources,
    )?;

    Ok(Some(NativeAuthRemoveResult {
        label: entry.label.clone(),
        cleaned,
        hints,
    }))
}

fn run_native_spotify_login(
    context: &HermesContext,
    args: &SpotifyAuthArgs,
) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    run_native_spotify_login_with_io(context, args, &mut input, &mut output)
}

pub(crate) fn run_native_spotify_login_with_io(
    context: &HermesContext,
    args: &SpotifyAuthArgs,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let existing_state =
        spotify_provider_state(&load_auth_store_json(context.hermes_home().as_path())?);
    let redirect_uri = spotify_redirect_uri(args.redirect_uri.as_deref(), existing_state.as_ref());
    let client_id = match spotify_client_id(args.client_id.as_deref(), existing_state.as_ref()) {
        Ok(value) => value,
        Err(error) if error == "spotify_client_id_missing" => {
            spotify_interactive_setup_with_io(context, input, output, &redirect_uri)?
        }
        Err(error) => return Err(error.into()),
    };
    let scope = spotify_scope_string(args.scope.as_deref().or_else(|| {
        existing_state
            .as_ref()
            .and_then(|state| json_string(state, "scope"))
    }));
    let accounts_base_url = spotify_accounts_base_url(existing_state.as_ref());
    let api_base_url = spotify_api_base_url(existing_state.as_ref());
    let timeout_seconds = args.timeout.unwrap_or(180.0).max(5.0);
    let code_verifier = spotify_code_verifier()?;
    let code_challenge = spotify_code_challenge(&code_verifier);
    let state_nonce = spotify_random_token(16)?;
    let authorize_url = spotify_build_authorize_url(
        &client_id,
        &redirect_uri,
        &scope,
        &state_nonce,
        &code_challenge,
        &accounts_base_url,
    )?;

    writeln!(output, "Starting Spotify PKCE login...")?;
    writeln!(output, "Client ID: {client_id}")?;
    writeln!(output, "Redirect URI: {redirect_uri}")?;
    writeln!(
        output,
        "Make sure this redirect URI is allow-listed in your Spotify app settings."
    )?;
    writeln!(output)?;
    writeln!(output, "Open this URL to authorize Hermes:")?;
    writeln!(output, "{authorize_url}")?;
    writeln!(output)?;
    writeln!(output, "Full setup guide: {SPOTIFY_DOCS_URL}")?;
    writeln!(output)?;
    output.flush()?;

    if !args.no_browser && !is_remote_session() {
        if try_open_browser(&authorize_url)? {
            writeln!(output, "Browser opened for Spotify authorization.")?;
        } else {
            writeln!(
                output,
                "Could not open the browser automatically; use the URL above."
            )?;
        }
        output.flush()?;
    }

    let callback = spotify_wait_for_callback(&redirect_uri, timeout_seconds)?;
    if let Some(error) = callback.error {
        let detail = callback.error_description.unwrap_or(error);
        return Err(format!("Spotify authorization failed: {detail}").into());
    }
    if callback.state.as_deref() != Some(state_nonce.as_str()) {
        return Err("Spotify authorization failed: state mismatch.".into());
    }
    let code = callback
        .code
        .ok_or("Spotify authorization failed: missing authorization code.")?;
    let token_payload = spotify_exchange_code_for_tokens(
        &client_id,
        &code,
        &redirect_uri,
        &code_verifier,
        &accounts_base_url,
        args.timeout.unwrap_or(20.0).max(1.0),
    )?;
    let spotify_state = spotify_token_payload_to_state(
        &token_payload,
        &client_id,
        &redirect_uri,
        &scope,
        &accounts_base_url,
        &api_base_url,
        existing_state.as_ref(),
    );
    let mut auth_store = load_auth_store_json(context.hermes_home().as_path())?;
    store_provider_state(&mut auth_store, "spotify", spotify_state)?;
    save_auth_store_json(context.hermes_home().as_path(), &auth_store)?;

    writeln!(output, "Spotify login successful!")?;
    writeln!(
        output,
        "  Auth state: {}",
        context.hermes_home().join("auth.json").display()
    )?;
    writeln!(output, "  Provider state saved under providers.spotify")?;
    writeln!(output, "  Docs: {SPOTIFY_DOCS_URL}")?;
    output.flush()?;
    Ok(())
}

fn render_status(status: AuthStatusSummary, active: Option<String>) -> String {
    let mut lines = Vec::new();
    lines.push(format!("provider={}", status.provider));
    lines.push(format!("name={}", status.display_name));
    lines.push(format!("configured={}", status.configured));
    lines.push(format!("logged_in={}", status.logged_in));
    lines.push(format!(
        "active={}",
        active.as_deref() == Some(status.provider.as_str())
    ));
    if let Some(source) = status.source {
        lines.push(format!("source={source}"));
    }
    if let Some(path) = status.auth_path {
        lines.push(format!("auth_path={path}"));
    }
    if let Some(detail) = status.detail {
        lines.push(format!("detail={detail}"));
    }
    lines.join("\n")
}

fn logout_provider(
    context: &HermesContext,
    loaded: &LoadedConfig,
    explicit_provider: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let target = explicit_provider
        .and_then(non_empty_trimmed)
        .or_else(|| active_provider(context, loaded).ok().flatten())
        .or_else(|| config_logout_fallback(loaded));

    let Some(target) = target else {
        println!("logged_out=false");
        println!("detail=no provider is currently logged in");
        return Ok(());
    };

    let cleared = clear_provider_auth_state(context.hermes_home().as_path(), &target)?;
    let reset = reset_config_provider(context.config_path().as_path(), &target)?;
    if cleared || reset {
        println!("logged_out=true");
        println!("provider={target}");
        println!("config_provider=auto");
        println!("fallback_provider=openrouter");
    } else {
        println!("logged_out=false");
        println!("provider={target}");
        println!("detail=no auth state found");
    }
    Ok(())
}

fn active_provider(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<Option<String>, Box<dyn Error>> {
    Ok(get_active_auth_provider(context.hermes_home().as_path())?
        .or_else(|| config_logout_fallback(loaded)))
}

fn config_logout_fallback(loaded: &LoadedConfig) -> Option<String> {
    let provider = loaded.configured_model_provider()?;
    CONFIG_FALLBACK_LOGOUT_PROVIDERS
        .iter()
        .any(|candidate| *candidate == provider)
        .then_some(provider)
}

fn reset_config_provider(path: &Path, target: &str) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    let raw = fs::read_to_string(path)?;
    if raw.trim().is_empty() {
        return Ok(false);
    }
    let mut parsed = serde_yaml::from_str::<Value>(&raw)?;
    let Some(root) = parsed.as_mapping_mut() else {
        return Ok(false);
    };
    let Some(model) = root
        .get_mut(Value::String("model".to_string()))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(false);
    };
    let current_provider = model
        .get(Value::String("provider".to_string()))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if current_provider.as_deref() != Some(target) {
        return Ok(false);
    }
    model.insert(
        Value::String("provider".to_string()),
        Value::String("auto".to_string()),
    );
    if model.contains_key(Value::String("base_url".to_string())) {
        model.insert(
            Value::String("base_url".to_string()),
            Value::String(OPENROUTER_BASE_URL.to_string()),
        );
    }
    let rendered = serde_yaml::to_string(&parsed)?;
    atomic_write(path, rendered.as_bytes())?;
    Ok(true)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status code {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn spotify_provider_state(auth_store: &JsonValue) -> Option<JsonMap<String, JsonValue>> {
    auth_store
        .get("providers")
        .and_then(JsonValue::as_object)
        .and_then(|providers| providers.get("spotify"))
        .and_then(JsonValue::as_object)
        .cloned()
}

fn store_provider_state(
    auth_store: &mut JsonValue,
    provider_id: &str,
    state: JsonMap<String, JsonValue>,
) -> Result<(), Box<dyn Error>> {
    let root = auth_store
        .as_object_mut()
        .ok_or("auth store is not a JSON object")?;
    let providers = root
        .entry("providers".to_string())
        .or_insert_with(|| JsonValue::Object(JsonMap::new()));
    let providers = providers
        .as_object_mut()
        .ok_or("providers is not a JSON object")?;
    providers.insert(provider_id.to_string(), JsonValue::Object(state));
    Ok(())
}

fn json_string<'a>(mapping: &'a JsonMap<String, JsonValue>, key: &str) -> Option<&'a str> {
    mapping
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn env_trimmed(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn spotify_scope_string(raw_scope: Option<&str>) -> String {
    let scope_text = raw_scope.unwrap_or(DEFAULT_SPOTIFY_SCOPE).trim();
    let mut seen = std::collections::BTreeSet::new();
    let mut scopes = Vec::new();
    for scope in scope_text.split_whitespace() {
        if seen.insert(scope.to_string()) {
            scopes.push(scope.to_string());
        }
    }
    if scopes.is_empty() {
        DEFAULT_SPOTIFY_SCOPE.to_string()
    } else {
        scopes.join(" ")
    }
}

fn spotify_client_id(
    explicit: Option<&str>,
    state: Option<&JsonMap<String, JsonValue>>,
) -> Result<String, &'static str> {
    for candidate in [
        explicit
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        env_trimmed("HERMES_SPOTIFY_CLIENT_ID"),
        env_trimmed("SPOTIFY_CLIENT_ID"),
        state
            .and_then(|state| json_string(state, "client_id"))
            .map(ToOwned::to_owned),
    ] {
        if let Some(value) = candidate {
            return Ok(value);
        }
    }
    Err("spotify_client_id_missing")
}

fn spotify_redirect_uri(
    explicit: Option<&str>,
    state: Option<&JsonMap<String, JsonValue>>,
) -> String {
    for candidate in [
        explicit
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        env_trimmed("HERMES_SPOTIFY_REDIRECT_URI"),
        env_trimmed("SPOTIFY_REDIRECT_URI"),
        state
            .and_then(|state| json_string(state, "redirect_uri"))
            .map(ToOwned::to_owned),
        Some(DEFAULT_SPOTIFY_REDIRECT_URI.to_string()),
    ] {
        if let Some(value) = candidate {
            return value;
        }
    }
    DEFAULT_SPOTIFY_REDIRECT_URI.to_string()
}

fn spotify_api_base_url(state: Option<&JsonMap<String, JsonValue>>) -> String {
    for candidate in [
        env_trimmed("HERMES_SPOTIFY_API_BASE_URL"),
        state
            .and_then(|state| json_string(state, "api_base_url"))
            .map(ToOwned::to_owned),
        Some(DEFAULT_SPOTIFY_API_BASE_URL.to_string()),
    ] {
        if let Some(value) = candidate {
            let trimmed = value.trim_end_matches('/').to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    DEFAULT_SPOTIFY_API_BASE_URL.to_string()
}

fn spotify_accounts_base_url(state: Option<&JsonMap<String, JsonValue>>) -> String {
    for candidate in [
        env_trimmed("HERMES_SPOTIFY_ACCOUNTS_BASE_URL"),
        state
            .and_then(|state| json_string(state, "accounts_base_url"))
            .map(ToOwned::to_owned),
        Some(DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()),
    ] {
        if let Some(value) = candidate {
            let trimmed = value.trim_end_matches('/').to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()
}

fn spotify_random_token(byte_len: usize) -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_SPOTIFY_TEST_STATE") {
        return Ok(override_value);
    }
    let mut bytes = vec![0u8; byte_len];
    fill_random(&mut bytes).map_err(|error| format!("failed to generate random bytes: {error}"))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn spotify_code_verifier() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_SPOTIFY_TEST_VERIFIER") {
        return Ok(override_value);
    }
    spotify_random_token(64).map(|value| value.chars().take(128).collect())
}

fn spotify_code_challenge(code_verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(code_verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hasher.finalize())
}

fn spotify_build_authorize_url(
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    state: &str,
    code_challenge: &str,
    accounts_base_url: &str,
) -> Result<String, Box<dyn Error>> {
    let mut url = Url::parse(&format!(
        "{}/authorize",
        accounts_base_url.trim_end_matches('/')
    ))?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("client_id", client_id);
        query.append_pair("response_type", "code");
        query.append_pair("redirect_uri", redirect_uri);
        query.append_pair("scope", scope);
        query.append_pair("state", state);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("code_challenge", code_challenge);
    }
    Ok(url.to_string())
}

fn spotify_validate_redirect_uri(
    redirect_uri: &str,
) -> Result<(String, u16, String), Box<dyn Error>> {
    let parsed = Url::parse(redirect_uri)?;
    if parsed.scheme() != "http" {
        return Err(
            "Spotify PKCE redirect_uri must use http://localhost or http://127.0.0.1.".into(),
        );
    }
    let host = parsed.host_str().unwrap_or_default();
    if host != "127.0.0.1" && host != "localhost" {
        return Err("Spotify PKCE redirect_uri must point to localhost or 127.0.0.1.".into());
    }
    let port = parsed
        .port()
        .ok_or("Spotify PKCE redirect_uri must include an explicit localhost port.")?;
    let path = if parsed.path().is_empty() {
        "/".to_string()
    } else {
        parsed.path().to_string()
    };
    Ok((host.to_string(), port, path))
}

fn spotify_wait_for_callback(
    redirect_uri: &str,
    timeout_seconds: f64,
) -> Result<SpotifyCallbackResult, Box<dyn Error>> {
    let (host, port, expected_path) = spotify_validate_redirect_uri(redirect_uri)?;
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|error| {
        format!("Could not bind Spotify callback server on {host}:{port}: {error}")
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("failed to configure Spotify callback server: {error}"))?;
    let deadline = std::time::Instant::now() + Duration::from_secs_f64(timeout_seconds.max(5.0));
    let mut buffer = [0u8; 8192];
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let size = stream.read(&mut buffer)?;
                let request = String::from_utf8_lossy(&buffer[..size]);
                let mut result = SpotifyCallbackResult::default();
                let first_line = request.lines().next().unwrap_or_default();
                if let Some(target) = first_line
                    .strip_prefix("GET ")
                    .and_then(|value| value.split_whitespace().next())
                {
                    let parsed = Url::parse(&format!("http://localhost{target}"))?;
                    if parsed.path() == expected_path {
                        let mut response_body =
                            "Spotify authorization received. You can close this tab.".to_string();
                        for (key, value) in parsed.query_pairs() {
                            match key.as_ref() {
                                "code" => result.code = Some(value.into_owned()),
                                "state" => result.state = Some(value.into_owned()),
                                "error" => result.error = Some(value.into_owned()),
                                "error_description" => {
                                    result.error_description = Some(value.into_owned())
                                }
                                _ => {}
                            }
                        }
                        if result.error.is_some() {
                            response_body =
                                "Spotify authorization failed. You can close this tab.".to_string();
                        }
                        let html = format!("<html><body><h1>{}</h1></body></html>", response_body);
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n<html><body><h1>{}</h1></body></html>",
                            html.len(),
                            response_body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        return Ok(result);
                    }
                }
                let response = b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 10\r\n\r\nNot found.";
                let _ = stream.write_all(response);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(format!("Spotify callback server failed: {error}").into()),
        }
    }
    Err("Spotify authorization timed out waiting for the local callback.".into())
}

fn spotify_exchange_code_for_tokens(
    client_id: &str,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
    accounts_base_url: &str,
    timeout_seconds: f64,
) -> Result<JsonMap<String, JsonValue>, Box<dyn Error>> {
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()?;
    let response = client
        .post(format!(
            "{}/api/token",
            accounts_base_url.trim_end_matches('/')
        ))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("client_id", client_id),
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", code_verifier),
        ])
        .send()
        .map_err(|error| format!("Spotify token exchange failed: {error}"))?;
    if response.status().as_u16() >= 400 {
        let detail = response.text().unwrap_or_default();
        let suffix = if detail.trim().is_empty() {
            String::new()
        } else {
            format!(" Response: {}", detail.trim())
        };
        return Err(format!("Spotify token exchange failed.{suffix}").into());
    }
    let payload: JsonValue = response.json()?;
    let payload = payload
        .as_object()
        .cloned()
        .ok_or("Spotify token response was not a JSON object.")?;
    if json_string(&payload, "access_token").is_none() {
        return Err("Spotify token response did not include an access_token.".into());
    }
    Ok(payload)
}

fn spotify_token_payload_to_state(
    token_payload: &JsonMap<String, JsonValue>,
    client_id: &str,
    redirect_uri: &str,
    requested_scope: &str,
    accounts_base_url: &str,
    api_base_url: &str,
    previous_state: Option<&JsonMap<String, JsonValue>>,
) -> JsonMap<String, JsonValue> {
    let now = Utc::now();
    let expires_in = token_payload
        .get("expires_in")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0)
        .max(0);
    let expires_at = now + ChronoDuration::seconds(expires_in);
    let mut state = previous_state.cloned().unwrap_or_default();
    state.insert(
        "client_id".to_string(),
        JsonValue::String(client_id.to_string()),
    );
    state.insert(
        "redirect_uri".to_string(),
        JsonValue::String(redirect_uri.to_string()),
    );
    state.insert(
        "accounts_base_url".to_string(),
        JsonValue::String(accounts_base_url.to_string()),
    );
    state.insert(
        "api_base_url".to_string(),
        JsonValue::String(api_base_url.to_string()),
    );
    state.insert(
        "scope".to_string(),
        JsonValue::String(requested_scope.to_string()),
    );
    state.insert(
        "granted_scope".to_string(),
        JsonValue::String(
            json_string(token_payload, "scope")
                .unwrap_or(requested_scope)
                .to_string(),
        ),
    );
    state.insert(
        "token_type".to_string(),
        JsonValue::String(
            json_string(token_payload, "token_type")
                .unwrap_or("Bearer")
                .to_string(),
        ),
    );
    state.insert(
        "access_token".to_string(),
        JsonValue::String(
            json_string(token_payload, "access_token")
                .unwrap_or("")
                .to_string(),
        ),
    );
    let refresh_token = json_string(token_payload, "refresh_token")
        .map(ToOwned::to_owned)
        .or_else(|| {
            previous_state
                .and_then(|state| json_string(state, "refresh_token"))
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default();
    state.insert(
        "refresh_token".to_string(),
        JsonValue::String(refresh_token),
    );
    state.insert(
        "obtained_at".to_string(),
        JsonValue::String(now.to_rfc3339()),
    );
    state.insert(
        "expires_at".to_string(),
        JsonValue::String(expires_at.to_rfc3339()),
    );
    state.insert("expires_in".to_string(), JsonValue::from(expires_in));
    state.insert(
        "auth_type".to_string(),
        JsonValue::String("oauth_pkce".to_string()),
    );
    state
}

fn spotify_interactive_setup_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    redirect_uri_hint: &str,
) -> Result<String, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "{}", "=".repeat(70))?;
    writeln!(output, "Spotify first-time setup")?;
    writeln!(output, "{}", "=".repeat(70))?;
    writeln!(output)?;
    writeln!(
        output,
        "Spotify requires every user to register their own lightweight"
    )?;
    writeln!(
        output,
        "developer app. This takes about two minutes and only has to be"
    )?;
    writeln!(output, "done once per machine.")?;
    writeln!(output)?;
    writeln!(output, "Full guide: {SPOTIFY_DOCS_URL}")?;
    writeln!(output)?;
    writeln!(output, "Steps:")?;
    writeln!(
        output,
        "  1. Opening {SPOTIFY_DASHBOARD_URL} in your browser..."
    )?;
    writeln!(output, "  2. Click 'Create app' and fill in:")?;
    writeln!(output, "       App name:     anything (e.g. hermes-agent)")?;
    writeln!(output, "       Description:  anything")?;
    writeln!(output, "       Redirect URI: {redirect_uri_hint}")?;
    writeln!(output, "       API/SDK:      Web API")?;
    writeln!(output, "  3. Agree to the terms, click Save.")?;
    writeln!(
        output,
        "  4. Open the app's Settings page and copy the Client ID."
    )?;
    writeln!(output, "  5. Paste it below.")?;
    writeln!(output)?;
    output.flush()?;
    if !is_remote_session() {
        let _ = try_open_browser(SPOTIFY_DASHBOARD_URL)?;
    }

    let raw = prompt_line(input, output, "Spotify Client ID")?;
    let client_id = raw.trim();
    if client_id.is_empty() {
        return Err("Spotify setup cancelled: empty Client ID.".into());
    }
    save_env_value(context.env_path(), "HERMES_SPOTIFY_CLIENT_ID", client_id)?;
    if redirect_uri_hint != DEFAULT_SPOTIFY_REDIRECT_URI {
        save_env_value(
            context.env_path(),
            "HERMES_SPOTIFY_REDIRECT_URI",
            redirect_uri_hint,
        )?;
    }
    writeln!(output)?;
    writeln!(
        output,
        "Saved HERMES_SPOTIFY_CLIENT_ID to {}",
        context.env_path().display()
    )?;
    writeln!(output)?;
    output.flush()?;
    Ok(client_id.to_string())
}

fn prompt_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    prompt: &str,
) -> Result<String, Box<dyn Error>> {
    write!(output, "{prompt}: ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err("interactive input closed".into());
    }
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
}

fn is_remote_session() -> bool {
    std::env::var_os("SSH_CLIENT").is_some() || std::env::var_os("SSH_TTY").is_some()
}

fn try_open_browser(url: &str) -> Result<bool, Box<dyn Error>> {
    let commands: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("open", &[])]
    } else if cfg!(target_os = "windows") {
        &[("cmd", &["/C", "start", ""])]
    } else {
        &[("xdg-open", &[])]
    };
    for (program, args) in commands {
        let mut command = Command::new(program);
        command.args(*args).arg(url);
        match command.status() {
            Ok(status) if status.success() => return Ok(true),
            Ok(_) => continue,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("failed to open browser: {error}").into()),
        }
    }
    Ok(false)
}

#[derive(Debug, Clone)]
struct PoolEntry {
    id: String,
    label: String,
    auth_type: String,
    priority: i64,
    source: String,
    last_status: Option<String>,
    last_status_at: Option<f64>,
    last_error_code: Option<i64>,
    last_error_reason: Option<String>,
    last_error_message: Option<String>,
    last_error_reset_at: Option<f64>,
}

fn load_credential_pool(
    hermes_home: &Path,
) -> Result<std::collections::BTreeMap<String, Vec<PoolEntry>>, Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    if !auth_path.exists() {
        return Ok(std::collections::BTreeMap::new());
    }
    let raw = fs::read_to_string(auth_path)?;
    if raw.trim().is_empty() {
        return Ok(std::collections::BTreeMap::new());
    }
    let parsed: JsonValue = serde_json::from_str(&raw)?;
    let mut result = std::collections::BTreeMap::new();
    let Some(pool) = parsed.get("credential_pool").and_then(JsonValue::as_object) else {
        return Ok(result);
    };
    for (provider, entries) in pool {
        let normalized = normalize_provider_name(provider).unwrap_or_else(|| provider.to_string());
        let Some(sequence) = entries.as_array() else {
            continue;
        };
        let parsed_entries = sequence
            .iter()
            .filter_map(parse_pool_entry)
            .collect::<Vec<_>>();
        result.insert(normalized, parsed_entries);
    }
    Ok(result)
}

fn reset_auth_pool_statuses(hermes_home: &Path, provider: &str) -> Result<usize, Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    if !auth_path.exists() {
        return Ok(0);
    }
    let raw = fs::read_to_string(&auth_path)?;
    if raw.trim().is_empty() {
        return Ok(0);
    }
    let mut parsed: JsonValue = serde_json::from_str(&raw)?;
    let Some(root) = parsed.as_object_mut() else {
        return Ok(0);
    };
    let Some(pool) = root
        .get_mut("credential_pool")
        .and_then(JsonValue::as_object_mut)
    else {
        return Ok(0);
    };
    let Some(entries) = pool.get_mut(provider).and_then(JsonValue::as_array_mut) else {
        return Ok(0);
    };
    let mut count = 0usize;
    for entry in entries {
        let Some(mapping) = entry.as_object_mut() else {
            continue;
        };
        let changed = mapping
            .get("last_status")
            .is_some_and(|value| !value.is_null())
            || mapping
                .get("last_status_at")
                .is_some_and(|value| !value.is_null())
            || mapping
                .get("last_error_code")
                .is_some_and(|value| !value.is_null());
        if changed {
            count += 1;
        }
        for key in [
            "last_status",
            "last_status_at",
            "last_error_code",
            "last_error_reason",
            "last_error_message",
            "last_error_reset_at",
        ] {
            mapping.insert(key.to_string(), JsonValue::Null);
        }
    }
    if count > 0 {
        let rendered = serde_json::to_string_pretty(&parsed)?;
        atomic_write(&auth_path, rendered.as_bytes())?;
    }
    Ok(count)
}

fn parse_pool_entry(value: &JsonValue) -> Option<PoolEntry> {
    let mapping = value.as_object()?;
    Some(PoolEntry {
        id: mapping
            .get("id")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_default(),
        label: mapping
            .get("label")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_default(),
        auth_type: mapping
            .get("auth_type")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "api_key".to_string()),
        priority: mapping
            .get("priority")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0),
        source: mapping
            .get("source")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "manual".to_string()),
        last_status: mapping
            .get("last_status")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
        last_status_at: mapping.get("last_status_at").and_then(JsonValue::as_f64),
        last_error_code: mapping.get("last_error_code").and_then(JsonValue::as_i64),
        last_error_reason: mapping
            .get("last_error_reason")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
        last_error_message: mapping
            .get("last_error_message")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
        last_error_reset_at: mapping
            .get("last_error_reset_at")
            .and_then(JsonValue::as_f64),
    })
}

fn normalize_provider_name(provider: &str) -> Option<String> {
    let normalized = normalize_provider_alias(provider);
    if normalized.trim().is_empty() {
        return None;
    }
    Some(normalized)
}

fn peek_entry_id(entries: &[PoolEntry]) -> Option<String> {
    entries
        .iter()
        .find(|entry| !entry_is_exhausted(entry))
        .map(|entry| entry.id.clone())
}

fn entry_is_exhausted(entry: &PoolEntry) -> bool {
    if entry.last_status.as_deref() != Some("exhausted") {
        return false;
    }
    let now = now_unix_seconds();
    if let Some(reset_at) = entry.last_error_reset_at
        && reset_at > now
    {
        return true;
    }
    if let Some(status_at) = entry.last_status_at {
        return status_at + exhausted_ttl_seconds(entry.last_error_code) > now;
    }
    false
}

fn exhausted_ttl_seconds(error_code: Option<i64>) -> f64 {
    match error_code {
        Some(429) => 60.0 * 60.0,
        _ => 60.0 * 60.0,
    }
}

fn display_source(source: &str) -> &str {
    source.strip_prefix("manual:").unwrap_or(source)
}

fn format_exhausted_status(entry: &PoolEntry) -> String {
    if entry.last_status.as_deref() != Some("exhausted") {
        return String::new();
    }
    let reason = entry
        .last_error_reason
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let message = entry
        .last_error_message
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let (label, retryable) = if entry.last_error_code == Some(429)
        || ["rate_limit", "usage_limit", "quota", "exhausted"]
            .iter()
            .any(|token| reason.contains(token))
        || ["rate limit", "usage limit", "quota", "too many requests"]
            .iter()
            .any(|token| message.contains(token))
    {
        ("rate-limited", true)
    } else if matches!(entry.last_error_code, Some(401 | 403))
        || [
            "invalid_token",
            "invalid_grant",
            "unauthorized",
            "forbidden",
            "auth",
        ]
        .iter()
        .any(|token| reason.contains(token))
        || [
            "unauthorized",
            "forbidden",
            "expired",
            "revoked",
            "invalid token",
            "authentication",
        ]
        .iter()
        .any(|token| message.contains(token))
    {
        ("auth failed", false)
    } else {
        ("exhausted", true)
    };
    let code = entry
        .last_error_code
        .map(|value| format!(" ({value})"))
        .unwrap_or_default();
    if !retryable {
        return format!(" {label}{code} (re-auth may be required)");
    }
    let Some(remaining) = exhausted_remaining_seconds(entry) else {
        return format!(" {label}{code}");
    };
    if remaining <= 0.0 {
        return format!(" {label}{code} (ready to retry)");
    }
    format!(
        " {label}{code} ({} left)",
        format_duration(remaining.ceil() as u64)
    )
}

fn exhausted_remaining_seconds(entry: &PoolEntry) -> Option<f64> {
    let now = now_unix_seconds();
    if let Some(reset_at) = entry.last_error_reset_at {
        return Some((reset_at - now).max(0.0));
    }
    entry
        .last_status_at
        .map(|status_at| (status_at + exhausted_ttl_seconds(entry.last_error_code) - now).max(0.0))
}

fn now_unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

fn format_duration(seconds: u64) -> String {
    let minutes = seconds / 60;
    let secs = seconds % 60;
    let hours = minutes / 60;
    let minutes = minutes % 60;
    let days = hours / 24;
    let hours = hours % 24;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

fn normalize_auth_type(raw: Option<&str>) -> Option<&'static str> {
    match raw.map(str::trim).filter(|value| !value.is_empty()) {
        Some("api-key" | "api_key") => Some("api_key"),
        Some("oauth") => Some("oauth"),
        Some(_) => None,
        None => None,
    }
}

fn load_auth_store_json(hermes_home: &Path) -> Result<JsonValue, Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    if !auth_path.exists() {
        return Ok(serde_json::json!({
            "version": 1,
            "providers": {}
        }));
    }
    let raw = fs::read_to_string(auth_path)?;
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({
            "version": 1,
            "providers": {}
        }));
    }
    Ok(serde_json::from_str(&raw)?)
}

fn save_auth_store_json(hermes_home: &Path, value: &JsonValue) -> Result<(), Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    let rendered = serde_json::to_string_pretty(value)?;
    atomic_write(&auth_path, rendered.as_bytes())
}

fn clear_provider_suppressions(hermes_home: &Path, provider: &str) -> Result<bool, Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let Some(root) = auth_store.as_object_mut() else {
        return Ok(false);
    };
    let Some(suppressed) = root
        .get_mut("suppressed_sources")
        .and_then(JsonValue::as_object_mut)
    else {
        return Ok(false);
    };
    if suppressed.remove(provider).is_none() {
        return Ok(false);
    }
    if suppressed.is_empty() {
        root.remove("suppressed_sources");
    }
    save_auth_store_json(hermes_home, &auth_store)?;
    Ok(true)
}

struct NewPoolEntry {
    label: String,
    auth_type: String,
    source: String,
    access_token: String,
    refresh_token: Option<String>,
    base_url: Option<String>,
    expires_at_ms: Option<i64>,
    last_refresh: Option<String>,
}

fn ensure_json_object<'a>(
    root: &'a mut serde_json::Map<String, JsonValue>,
    key: &str,
) -> Result<&'a mut serde_json::Map<String, JsonValue>, Box<dyn Error>> {
    let value = root
        .entry(key.to_string())
        .or_insert_with(|| JsonValue::Object(Default::default()));
    value
        .as_object_mut()
        .ok_or_else(|| format!("{key} is not a JSON object").into())
}

fn generate_short_id() -> String {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let hex = format!("{unique:x}");
    if hex.len() <= 6 {
        hex
    } else {
        hex[hex.len() - 6..].to_string()
    }
}

fn next_provider_entry_index(hermes_home: &Path, provider: &str) -> Result<usize, Box<dyn Error>> {
    let auth_store = load_auth_store_json(hermes_home)?;
    let count = auth_store
        .get("credential_pool")
        .and_then(JsonValue::as_object)
        .and_then(|pool| pool.get(provider))
        .and_then(JsonValue::as_array)
        .map(|entries| entries.len())
        .unwrap_or(0);
    Ok(count + 1)
}

fn provider_pool_has_entries(hermes_home: &Path, provider: &str) -> Result<bool, Box<dyn Error>> {
    let auth_store = load_auth_store_json(hermes_home)?;
    Ok(auth_store
        .get("credential_pool")
        .and_then(JsonValue::as_object)
        .and_then(|pool| pool.get(provider))
        .and_then(JsonValue::as_array)
        .is_some_and(|entries| !entries.is_empty()))
}

fn provider_state_exists(hermes_home: &Path, provider: &str) -> Result<bool, Box<dyn Error>> {
    let auth_store = load_auth_store_json(hermes_home)?;
    Ok(provider_state_json(&auth_store, provider).is_some())
}

fn provider_state_json(
    auth_store: &JsonValue,
    provider: &str,
) -> Option<JsonMap<String, JsonValue>> {
    auth_store
        .get("providers")
        .and_then(JsonValue::as_object)
        .and_then(|providers| providers.get(provider))
        .and_then(JsonValue::as_object)
        .cloned()
}

fn apply_native_auth_remove_changes(
    hermes_home: &Path,
    provider: &str,
    index: usize,
    clear_provider_state: bool,
    suppress_sources: &[String],
) -> Result<(), Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let root = auth_store
        .as_object_mut()
        .ok_or("auth store is not a JSON object")?;

    let Some(pool) = root
        .get_mut("credential_pool")
        .and_then(JsonValue::as_object_mut)
    else {
        return Err(format!("No credential #{}.", index).into());
    };
    let Some(entries) = pool.get_mut(provider).and_then(JsonValue::as_array_mut) else {
        return Err(format!("No credential #{}.", index).into());
    };
    if index == 0 || index > entries.len() {
        return Err(format!("No credential #{}.", index).into());
    }
    entries.remove(index - 1);
    for (priority, entry) in entries.iter_mut().enumerate() {
        if let Some(mapping) = entry.as_object_mut() {
            mapping.insert("priority".to_string(), JsonValue::from(priority as i64));
        }
    }

    if clear_provider_state {
        if let Some(providers) = root.get_mut("providers").and_then(JsonValue::as_object_mut) {
            providers.remove(provider);
        }
        if root.get("active_provider").and_then(JsonValue::as_str) == Some(provider) {
            root.insert("active_provider".to_string(), JsonValue::Null);
        }
    }

    if !suppress_sources.is_empty() {
        let suppressed = root
            .entry("suppressed_sources".to_string())
            .or_insert_with(|| JsonValue::Object(JsonMap::new()));
        let suppressed = suppressed
            .as_object_mut()
            .ok_or("suppressed_sources is not a JSON object")?;
        let provider_sources = suppressed
            .entry(provider.to_string())
            .or_insert_with(|| JsonValue::Array(Vec::new()));
        let provider_sources = provider_sources
            .as_array_mut()
            .ok_or("suppressed_sources entry is not an array")?;
        for source in suppress_sources {
            if !provider_sources
                .iter()
                .any(|value| value.as_str() == Some(source.as_str()))
            {
                provider_sources.push(JsonValue::String(source.clone()));
            }
        }
    }

    save_auth_store_json(hermes_home, &auth_store)
}

fn remove_hermes_pkce_file(hermes_home: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let oauth_file = hermes_home.join(".anthropic_oauth.json");
    if !oauth_file.exists() {
        return Ok(Vec::new());
    }
    fs::remove_file(&oauth_file)?;
    Ok(vec![
        "Cleared Hermes Anthropic OAuth credentials".to_string(),
    ])
}

fn remove_google_gemini_oauth_file(hermes_home: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let oauth_file = hermes_home.join("auth").join("google_oauth.json");
    if !oauth_file.exists() {
        return Ok(Vec::new());
    }
    fs::remove_file(&oauth_file)?;
    Ok(vec![
        "Cleared Google Gemini CLI OAuth credentials".to_string(),
    ])
}

fn remove_env_source_native(
    hermes_home: &Path,
    provider: &str,
    source: &str,
) -> Result<NativeRemovePlan, Box<dyn Error>> {
    let env_var = source
        .strip_prefix("env:")
        .ok_or("invalid env source")?
        .trim();
    if env_var.is_empty() {
        return Ok((false, vec![source.to_string()], Vec::new(), Vec::new()));
    }
    let env_path = hermes_home.join(".env");
    let env_in_process = std::env::var_os(env_var).is_some();
    let env_in_dotenv = env_file_contains_key(&env_path, env_var)?;
    let shell_exported = env_in_process && !env_in_dotenv;
    let removed = remove_env_file_key(&env_path, env_var)?;
    let mut cleaned = Vec::new();
    if removed {
        cleaned.push(format!("Cleared {env_var} from .env"));
    }
    let hints = if shell_exported {
        vec![
            format!(
                "Note: {env_var} is still set in your shell environment (not in ~/.hermes/.env)."
            ),
            "  Unset it there (shell profile, systemd EnvironmentFile, launchd plist, etc.) or it will keep being visible to Hermes.".to_string(),
            format!(
                "  The pool entry is now suppressed — Hermes will ignore {env_var} until you run `hermes auth add {provider}`."
            ),
        ]
    } else {
        vec![format!(
            "Suppressed env:{env_var} — it will not be re-seeded even if the variable is re-exported later."
        )]
    };
    Ok((false, vec![source.to_string()], cleaned, hints))
}

fn env_file_contains_key(path: &Path, key: &str) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    let text = fs::read_to_string(path)?;
    Ok(text.lines().any(|line| {
        line.trim()
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
    }))
}

fn remove_env_file_key(path: &Path, key: &str) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    let lines = fs::read_to_string(path)?
        .lines()
        .map(|line| format!("{line}\n"))
        .collect::<Vec<_>>();
    let mut changed = false;
    let filtered = lines
        .into_iter()
        .filter(|line| {
            let matches = line
                .strip_prefix(key)
                .is_some_and(|rest| rest.starts_with('='));
            changed |= matches;
            !matches
        })
        .collect::<Vec<_>>();
    if changed {
        atomic_write(path, filtered.concat().as_bytes())?;
        unsafe { std::env::remove_var(key) };
    }
    Ok(changed)
}

fn add_auth_pool_entry(
    hermes_home: &Path,
    provider: &str,
    entry: NewPoolEntry,
) -> Result<usize, Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let root = auth_store
        .as_object_mut()
        .ok_or("auth store is not a JSON object")?;
    if !root.contains_key("version") {
        root.insert("version".to_string(), JsonValue::from(1));
    }
    if !root.contains_key("providers") {
        root.insert(
            "providers".to_string(),
            JsonValue::Object(Default::default()),
        );
    }
    let pool = ensure_json_object(root, "credential_pool")?;
    let provider_entries = pool
        .entry(provider.to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()));
    let provider_entries = provider_entries
        .as_array_mut()
        .ok_or("credential_pool entry is not an array")?;
    let existing = provider_entries
        .iter()
        .filter_map(parse_pool_entry)
        .collect::<Vec<_>>();
    let priority = existing
        .iter()
        .map(|existing| existing.priority)
        .max()
        .unwrap_or(-1)
        + 1;
    let mut payload = JsonMap::new();
    payload.insert("id".to_string(), JsonValue::String(generate_short_id()));
    payload.insert("label".to_string(), JsonValue::String(entry.label));
    payload.insert("auth_type".to_string(), JsonValue::String(entry.auth_type));
    payload.insert("priority".to_string(), JsonValue::from(priority));
    payload.insert("source".to_string(), JsonValue::String(entry.source));
    payload.insert(
        "access_token".to_string(),
        JsonValue::String(entry.access_token),
    );
    if let Some(refresh_token) = entry.refresh_token.filter(|value| !value.trim().is_empty()) {
        payload.insert(
            "refresh_token".to_string(),
            JsonValue::String(refresh_token),
        );
    }
    if let Some(base_url) = entry.base_url.filter(|value| !value.trim().is_empty()) {
        payload.insert("base_url".to_string(), JsonValue::String(base_url));
    }
    if let Some(expires_at_ms) = entry.expires_at_ms {
        payload.insert("expires_at_ms".to_string(), JsonValue::from(expires_at_ms));
    }
    if let Some(last_refresh) = entry.last_refresh.filter(|value| !value.trim().is_empty()) {
        payload.insert("last_refresh".to_string(), JsonValue::String(last_refresh));
    }
    provider_entries.push(JsonValue::Object(payload));
    let count = provider_entries.len();
    save_auth_store_json(hermes_home, &auth_store)?;
    Ok(count)
}

fn upsert_auth_pool_entry_by_sources(
    hermes_home: &Path,
    provider: &str,
    entry: NewPoolEntry,
    replace_sources: &[&str],
) -> Result<usize, Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let root = auth_store
        .as_object_mut()
        .ok_or("auth store is not a JSON object")?;
    if !root.contains_key("version") {
        root.insert("version".to_string(), JsonValue::from(1));
    }
    if !root.contains_key("providers") {
        root.insert(
            "providers".to_string(),
            JsonValue::Object(Default::default()),
        );
    }
    let pool = ensure_json_object(root, "credential_pool")?;
    let provider_entries = pool
        .entry(provider.to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()));
    let provider_entries = provider_entries
        .as_array_mut()
        .ok_or("credential_pool entry is not an array")?;

    let mut first_match_index = None;
    let mut preserved_id = None;
    let mut filtered = Vec::with_capacity(provider_entries.len());
    for (index, value) in provider_entries.iter().enumerate() {
        let matches = value
            .as_object()
            .and_then(|mapping| mapping.get("source"))
            .and_then(JsonValue::as_str)
            .is_some_and(|source| replace_sources.iter().any(|candidate| *candidate == source));
        if matches {
            if first_match_index.is_none() {
                first_match_index = Some(index);
                preserved_id = value
                    .as_object()
                    .and_then(|mapping| mapping.get("id"))
                    .and_then(JsonValue::as_str)
                    .map(ToOwned::to_owned);
            }
            continue;
        }
        filtered.push(value.clone());
    }

    let insert_at = first_match_index
        .unwrap_or(filtered.len())
        .min(filtered.len());
    let mut payload = JsonMap::new();
    payload.insert(
        "id".to_string(),
        JsonValue::String(preserved_id.unwrap_or_else(generate_short_id)),
    );
    payload.insert("label".to_string(), JsonValue::String(entry.label));
    payload.insert("auth_type".to_string(), JsonValue::String(entry.auth_type));
    payload.insert("source".to_string(), JsonValue::String(entry.source));
    payload.insert(
        "access_token".to_string(),
        JsonValue::String(entry.access_token),
    );
    if let Some(refresh_token) = entry.refresh_token.filter(|value| !value.trim().is_empty()) {
        payload.insert(
            "refresh_token".to_string(),
            JsonValue::String(refresh_token),
        );
    }
    if let Some(base_url) = entry.base_url.filter(|value| !value.trim().is_empty()) {
        payload.insert("base_url".to_string(), JsonValue::String(base_url));
    }
    if let Some(expires_at_ms) = entry.expires_at_ms {
        payload.insert("expires_at_ms".to_string(), JsonValue::from(expires_at_ms));
    }
    if let Some(last_refresh) = entry.last_refresh.filter(|value| !value.trim().is_empty()) {
        payload.insert("last_refresh".to_string(), JsonValue::String(last_refresh));
    }
    filtered.insert(insert_at, JsonValue::Object(payload));
    for (priority, value) in filtered.iter_mut().enumerate() {
        if let Some(mapping) = value.as_object_mut() {
            mapping.insert("priority".to_string(), JsonValue::from(priority as i64));
            for key in [
                "last_status",
                "last_status_at",
                "last_error_code",
                "last_error_reason",
                "last_error_message",
                "last_error_reset_at",
            ] {
                mapping.remove(key);
            }
        }
    }
    *provider_entries = filtered;
    save_auth_store_json(hermes_home, &auth_store)?;
    Ok(insert_at + 1)
}

fn oauth_default_label(provider: &str, count: usize) -> String {
    format!("{provider}-oauth-{count}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedCustomProvider {
    pool_key: String,
    provider_key: Option<String>,
    base_url: Option<String>,
}

fn auth_add_uses_api_key_shape(args: &AuthAddArgs) -> bool {
    match normalize_auth_type(args.auth_type.as_deref()) {
        Some("oauth") => false,
        Some("api_key") | None => {
            args.api_key
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
                && args.portal_url.is_none()
                && args.inference_url.is_none()
                && args.client_id.is_none()
                && args.scope.is_none()
                && !args.no_browser
                && args.timeout.is_none()
                && !args.insecure
                && args.ca_bundle.is_none()
        }
        _ => false,
    }
}

fn resolve_custom_provider_api_key_target(
    config_path: &Path,
    args: &AuthAddArgs,
) -> Result<Option<ResolvedCustomProvider>, Box<dyn Error>> {
    if !auth_add_uses_api_key_shape(args) {
        return Ok(None);
    }
    resolve_custom_provider_target(config_path, &args.provider)
}

fn resolve_custom_provider_target(
    config_path: &Path,
    raw_provider: &str,
) -> Result<Option<ResolvedCustomProvider>, Box<dyn Error>> {
    let normalized = raw_provider.trim().to_ascii_lowercase();
    if normalized.is_empty() || matches!(normalized.as_str(), "or" | "open-router") {
        return Ok(None);
    }
    if !config_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(config_path)?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let parsed = serde_yaml::from_str::<Value>(&raw)?;
    let requested = normalized
        .strip_prefix("custom:")
        .unwrap_or(normalized.as_str())
        .trim();
    if requested.is_empty() {
        return Ok(None);
    }

    if let Some(resolved) = parsed
        .as_mapping()
        .and_then(|root| root.get(yaml_string_value("custom_providers")))
        .and_then(Value::as_sequence)
        .and_then(|entries| {
            entries.iter().find_map(|entry| {
                custom_provider_entry(entry.as_mapping()?, "").and_then(|resolved| {
                    custom_provider_matches(requested, &resolved).then_some(resolved)
                })
            })
        })
    {
        return Ok(Some(resolved));
    }

    Ok(parsed
        .as_mapping()
        .and_then(|root| root.get(yaml_string_value("providers")))
        .and_then(Value::as_mapping)
        .and_then(|providers| {
            providers.iter().find_map(|(key, value)| {
                let provider_key = key.as_str()?.trim();
                custom_provider_entry(value.as_mapping()?, provider_key).and_then(|resolved| {
                    custom_provider_matches(requested, &resolved).then_some(resolved)
                })
            })
        }))
}

fn custom_provider_entry(
    mapping: &serde_yaml::Mapping,
    provider_key: &str,
) -> Option<ResolvedCustomProvider> {
    let base_url = yaml_string_alias(mapping, &["base_url", "url", "api", "baseUrl"])?;
    let name = yaml_string_alias(mapping, &["name"])
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| provider_key.trim().to_string());
    let normalized_name = normalize_custom_provider_key(&name);
    if normalized_name.is_empty() {
        return None;
    }
    let provider_key = provider_key
        .trim()
        .to_ascii_lowercase()
        .replace(' ', "-")
        .trim()
        .to_string();
    Some(ResolvedCustomProvider {
        pool_key: format!("custom:{normalized_name}"),
        provider_key: if provider_key.is_empty() {
            None
        } else {
            Some(provider_key)
        },
        base_url: Some(base_url.trim().trim_end_matches('/').to_string()),
    })
}

fn custom_provider_matches(requested: &str, provider: &ResolvedCustomProvider) -> bool {
    let requested = normalize_custom_provider_key(requested);
    if requested.is_empty() {
        return false;
    }
    if provider.pool_key == format!("custom:{requested}") {
        return true;
    }
    provider
        .provider_key
        .as_deref()
        .is_some_and(|provider_key| provider_key == requested)
}

fn normalize_custom_provider_key(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(' ', "-")
}

fn yaml_string_alias(mapping: &serde_yaml::Mapping, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        mapping
            .get(yaml_string_value(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn yaml_string_value(key: &str) -> Value {
    Value::String(key.to_string())
}

fn has_explicit_nous_runtime_overrides(args: &AuthAddArgs) -> bool {
    args.portal_url
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
        || args
            .inference_url
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
        || args
            .client_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
        || args
            .scope
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
}

fn has_explicit_minimax_runtime_overrides(args: &AuthAddArgs) -> bool {
    args.portal_url
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
        || args
            .inference_url
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
        || args
            .client_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
        || args
            .scope
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
}

fn validated_timeout_seconds(raw: Option<f64>, default: f64) -> Result<f64, Box<dyn Error>> {
    match raw {
        Some(value) if value > 0.0 => Ok(value),
        Some(_) => Err("timeout must be greater than 0.".into()),
        None => Ok(default),
    }
}

fn normalize_http_url(raw: &str, field: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} cannot be empty.").into());
    }
    let parsed = Url::parse(trimmed).map_err(|error| format!("Invalid {field}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!("{field} must use http or https.").into());
    }
    Ok(trimmed.trim_end_matches('/').to_string())
}

fn resolve_nous_portal_base_url(args: &AuthAddArgs) -> Result<String, Box<dyn Error>> {
    let candidate = args
        .portal_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_trimmed("HERMES_PORTAL_BASE_URL"))
        .or_else(|| env_trimmed("NOUS_PORTAL_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_NOUS_PORTAL_URL.to_string());
    normalize_http_url(&candidate, "portal URL")
}

fn resolve_nous_inference_base_url(args: &AuthAddArgs) -> Result<String, Box<dyn Error>> {
    let candidate = args
        .inference_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_trimmed("NOUS_INFERENCE_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_NOUS_INFERENCE_URL.to_string());
    normalize_http_url(&candidate, "inference URL")
}

fn resolve_nous_client_id(args: &AuthAddArgs) -> String {
    args.client_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| DEFAULT_NOUS_CLIENT_ID.to_string())
}

fn resolve_nous_scope(args: &AuthAddArgs) -> String {
    args.scope
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| DEFAULT_NOUS_SCOPE.to_string())
}

fn resolve_minimax_portal_base_url(args: &AuthAddArgs) -> Result<String, Box<dyn Error>> {
    let candidate = args
        .portal_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL.to_string());
    normalize_http_url(&candidate, "portal URL")
}

fn resolve_minimax_inference_base_url(
    args: &AuthAddArgs,
    portal_base_url: &str,
) -> Result<String, Box<dyn Error>> {
    let candidate = args
        .inference_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| match portal_base_url {
            DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL => {
                DEFAULT_MINIMAX_OAUTH_CN_INFERENCE_BASE_URL.to_string()
            }
            DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL => {
                DEFAULT_MINIMAX_OAUTH_INFERENCE_BASE_URL.to_string()
            }
            other => format!("{}/anthropic", other.trim_end_matches('/')),
        });
    normalize_http_url(&candidate, "inference URL")
}

fn resolve_minimax_client_id(args: &AuthAddArgs) -> String {
    args.client_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| MINIMAX_OAUTH_CLIENT_ID.to_string())
}

fn resolve_minimax_scope(args: &AuthAddArgs) -> String {
    args.scope
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| MINIMAX_OAUTH_SCOPE.to_string())
}

fn json_i64(value: Option<&JsonValue>) -> Option<i64> {
    match value {
        Some(JsonValue::Number(number)) => number.as_i64(),
        Some(JsonValue::String(text)) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn seed_nous_provider_state_and_resolve(
    hermes_home: &Path,
    state: JsonMap<String, JsonValue>,
    timeout_seconds: f64,
) -> Result<(), Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    let existed_before = auth_path.exists();
    let original_auth = load_auth_store_json(hermes_home)?;
    let mut seeded_auth = original_auth.clone();
    store_provider_state(&mut seeded_auth, "nous", state)?;
    save_auth_store_json(hermes_home, &seeded_auth)?;
    match resolve_nous_runtime_credentials(hermes_home, 300, timeout_seconds) {
        Ok(_) => Ok(()),
        Err(error) => {
            if existed_before {
                save_auth_store_json(hermes_home, &original_auth)?;
            } else if auth_path.exists() {
                let _ = fs::remove_file(&auth_path);
            }
            Err(Box::new(error))
        }
    }
}

fn finalize_nous_auth_add(
    hermes_home: &Path,
    requested_label: Option<&str>,
    default_label: &str,
) -> Result<(usize, String), Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let mut state = provider_state_json(&auth_store, "nous")
        .ok_or("Nous auth state is missing after runtime resolution.")?;
    if let Some(label) = requested_label
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        state.insert("label".to_string(), JsonValue::String(label.to_string()));
        store_provider_state(&mut auth_store, "nous", state.clone())?;
        save_auth_store_json(hermes_home, &auth_store)?;
    }
    let access_token = json_string(&state, "access_token")
        .ok_or("Nous auth state is missing access_token after runtime resolution.")?
        .to_string();
    let label = json_string(&state, "label")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| label_from_token(&access_token, default_label));
    write_shared_nous_state(&state);
    clear_provider_suppressions(hermes_home, "nous")?;
    let stored_count = upsert_auth_pool_entry_by_sources(
        hermes_home,
        "nous",
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source: "device_code".to_string(),
            access_token,
            refresh_token: json_string(&state, "refresh_token").map(ToOwned::to_owned),
            base_url: json_string(&state, "inference_base_url").map(ToOwned::to_owned),
            expires_at_ms: None,
            last_refresh: None,
        },
        &["device_code", "manual:device_code"],
    )?;
    Ok((stored_count, label))
}

fn nous_shared_store_path() -> PathBuf {
    if let Some(path) = env_trimmed("HERMES_SHARED_AUTH_DIR") {
        return PathBuf::from(path).join("nous_auth.json");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".hermes")
        .join("shared")
        .join("nous_auth.json")
}

fn write_shared_nous_state(state: &JsonMap<String, JsonValue>) {
    let Some(access_token) = json_string(state, "access_token") else {
        return;
    };
    let Some(refresh_token) = json_string(state, "refresh_token") else {
        return;
    };
    let mut payload = JsonMap::new();
    payload.insert("_schema".to_string(), JsonValue::from(1));
    payload.insert(
        "access_token".to_string(),
        JsonValue::String(access_token.to_string()),
    );
    payload.insert(
        "refresh_token".to_string(),
        JsonValue::String(refresh_token.to_string()),
    );
    payload.insert(
        "token_type".to_string(),
        JsonValue::String(
            json_string(state, "token_type")
                .unwrap_or("Bearer")
                .to_string(),
        ),
    );
    payload.insert(
        "scope".to_string(),
        JsonValue::String(
            json_string(state, "scope")
                .unwrap_or(DEFAULT_NOUS_SCOPE)
                .to_string(),
        ),
    );
    payload.insert(
        "client_id".to_string(),
        JsonValue::String(
            json_string(state, "client_id")
                .unwrap_or(DEFAULT_NOUS_CLIENT_ID)
                .to_string(),
        ),
    );
    payload.insert(
        "portal_base_url".to_string(),
        JsonValue::String(
            json_string(state, "portal_base_url")
                .unwrap_or(DEFAULT_NOUS_PORTAL_URL)
                .to_string(),
        ),
    );
    payload.insert(
        "inference_base_url".to_string(),
        JsonValue::String(
            json_string(state, "inference_base_url")
                .unwrap_or(DEFAULT_NOUS_INFERENCE_URL)
                .to_string(),
        ),
    );
    payload.insert(
        "obtained_at".to_string(),
        state.get("obtained_at").cloned().unwrap_or(JsonValue::Null),
    );
    payload.insert(
        "expires_at".to_string(),
        state.get("expires_at").cloned().unwrap_or(JsonValue::Null),
    );
    payload.insert(
        "updated_at".to_string(),
        JsonValue::String(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    let path = nous_shared_store_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(rendered) = serde_json::to_string_pretty(&JsonValue::Object(payload)) {
        let _ = atomic_write(&path, rendered.as_bytes());
    }
}

fn read_shared_nous_state() -> Option<JsonMap<String, JsonValue>> {
    let path = nous_shared_store_path();
    if !path.is_file() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    let payload = serde_json::from_str::<JsonValue>(&raw).ok()?;
    let payload = payload.as_object()?.clone();
    json_string(&payload, "refresh_token")?;
    json_string(&payload, "access_token")?;
    Some(payload)
}

fn build_nous_shared_import_state(
    shared: &JsonMap<String, JsonValue>,
) -> JsonMap<String, JsonValue> {
    let mut state = JsonMap::new();
    state.insert(
        "access_token".to_string(),
        JsonValue::String(
            json_string(shared, "access_token")
                .unwrap_or_default()
                .to_string(),
        ),
    );
    state.insert(
        "refresh_token".to_string(),
        JsonValue::String(
            json_string(shared, "refresh_token")
                .unwrap_or_default()
                .to_string(),
        ),
    );
    state.insert(
        "client_id".to_string(),
        JsonValue::String(
            json_string(shared, "client_id")
                .unwrap_or(DEFAULT_NOUS_CLIENT_ID)
                .to_string(),
        ),
    );
    state.insert(
        "portal_base_url".to_string(),
        JsonValue::String(
            json_string(shared, "portal_base_url")
                .unwrap_or(DEFAULT_NOUS_PORTAL_URL)
                .to_string(),
        ),
    );
    state.insert(
        "inference_base_url".to_string(),
        JsonValue::String(
            json_string(shared, "inference_base_url")
                .unwrap_or(DEFAULT_NOUS_INFERENCE_URL)
                .to_string(),
        ),
    );
    state.insert(
        "token_type".to_string(),
        JsonValue::String(
            json_string(shared, "token_type")
                .unwrap_or("Bearer")
                .to_string(),
        ),
    );
    state.insert(
        "scope".to_string(),
        JsonValue::String(
            json_string(shared, "scope")
                .unwrap_or(DEFAULT_NOUS_SCOPE)
                .to_string(),
        ),
    );
    state.insert(
        "obtained_at".to_string(),
        shared
            .get("obtained_at")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    state.insert(
        "expires_at".to_string(),
        JsonValue::String("1970-01-01T00:00:00Z".to_string()),
    );
    state.insert("agent_key".to_string(), JsonValue::Null);
    state.insert("agent_key_id".to_string(), JsonValue::Null);
    state.insert("agent_key_expires_at".to_string(), JsonValue::Null);
    state.insert("agent_key_expires_in".to_string(), JsonValue::Null);
    state.insert("agent_key_reused".to_string(), JsonValue::Null);
    state.insert("agent_key_obtained_at".to_string(), JsonValue::Null);
    state
}

fn label_from_token(token: &str, fallback: &str) -> String {
    let Some(claims) = decode_jwt_claims(token) else {
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

fn anthropic_authorize_base_url() -> String {
    env_trimmed("HERMES_AUTH_ANTHROPIC_AUTHORIZE_URL")
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_OAUTH_AUTHORIZE_URL.to_string())
}

fn anthropic_token_url() -> String {
    env_trimmed("HERMES_AUTH_ANTHROPIC_TOKEN_URL")
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_OAUTH_TOKEN_URL.to_string())
}

fn anthropic_code_verifier() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_ANTHROPIC_TEST_VERIFIER") {
        return Ok(override_value);
    }
    spotify_random_token(32)
}

fn anthropic_build_authorize_url(
    state: &str,
    code_challenge: &str,
) -> Result<String, Box<dyn Error>> {
    let mut url = Url::parse(&anthropic_authorize_base_url())?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("code", "true");
        query.append_pair("client_id", ANTHROPIC_OAUTH_CLIENT_ID);
        query.append_pair("response_type", "code");
        query.append_pair("redirect_uri", ANTHROPIC_OAUTH_REDIRECT_URI);
        query.append_pair("scope", ANTHROPIC_OAUTH_SCOPES);
        query.append_pair("code_challenge", code_challenge);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("state", state);
    }
    Ok(url.to_string())
}

fn anthropic_split_pasted_code(raw: &str) -> (&str, &str) {
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

fn anthropic_exchange_code_for_tokens(
    code: &str,
    state: &str,
    code_verifier: &str,
    timeout_seconds: f64,
) -> Result<AnthropicOauthTokens, Box<dyn Error>> {
    if code.is_empty() {
        return Err("Anthropic authorization failed: missing authorization code.".into());
    }
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()?;
    let response = client
        .post(anthropic_token_url())
        .header("Content-Type", "application/json")
        .header("User-Agent", ANTHROPIC_OAUTH_USER_AGENT)
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
            "code_verifier": code_verifier,
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
        return Err(format!("Anthropic token exchange failed.{suffix}").into());
    }
    let payload: JsonValue = response.json()?;
    let payload = payload
        .as_object()
        .ok_or("Anthropic token response was not a JSON object.")?;
    let access_token = json_string(payload, "access_token")
        .ok_or("Anthropic token response did not include an access_token.")?
        .to_string();
    let refresh_token = json_string(payload, "refresh_token")
        .unwrap_or("")
        .to_string();
    let expires_in = payload
        .get("expires_in")
        .and_then(JsonValue::as_i64)
        .unwrap_or(3600)
        .max(0);
    let expires_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0)
        .saturating_add(expires_in.saturating_mul(1000));
    Ok(AnthropicOauthTokens {
        access_token,
        refresh_token,
        expires_at_ms,
    })
}

fn google_authorize_url() -> String {
    env_trimmed("HERMES_AUTH_GOOGLE_AUTH_URL")
        .unwrap_or_else(|| DEFAULT_GOOGLE_OAUTH_AUTHORIZE_URL.to_string())
}

fn google_token_url() -> String {
    env_trimmed("HERMES_AUTH_GOOGLE_TOKEN_URL")
        .unwrap_or_else(|| DEFAULT_GOOGLE_OAUTH_TOKEN_URL.to_string())
}

fn google_userinfo_url() -> String {
    env_trimmed("HERMES_AUTH_GOOGLE_USERINFO_URL")
        .unwrap_or_else(|| DEFAULT_GOOGLE_OAUTH_USERINFO_URL.to_string())
}

fn google_client_id() -> String {
    env_trimmed("HERMES_GEMINI_CLIENT_ID").unwrap_or_else(|| GOOGLE_DEFAULT_CLIENT_ID.to_string())
}

fn google_client_secret() -> String {
    env_trimmed("HERMES_GEMINI_CLIENT_SECRET")
        .unwrap_or_else(|| GOOGLE_DEFAULT_CLIENT_SECRET.to_string())
}

fn google_code_verifier() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_GOOGLE_TEST_VERIFIER") {
        return Ok(override_value);
    }
    spotify_random_token(64).map(|value| value.chars().take(128).collect())
}

fn google_state_nonce() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_GOOGLE_TEST_STATE") {
        return Ok(override_value);
    }
    spotify_random_token(16)
}

fn google_default_redirect_uri() -> String {
    format!(
        "http://{}:{}{}",
        GOOGLE_OAUTH_REDIRECT_HOST, GOOGLE_OAUTH_DEFAULT_REDIRECT_PORT, GOOGLE_OAUTH_CALLBACK_PATH
    )
}

fn google_gemini_oauth_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join("auth").join("google_oauth.json")
}

fn google_build_authorize_url(
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    code_challenge: &str,
) -> Result<String, Box<dyn Error>> {
    let mut url = Url::parse(&google_authorize_url())?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("client_id", client_id);
        query.append_pair("redirect_uri", redirect_uri);
        query.append_pair("response_type", "code");
        query.append_pair("scope", GOOGLE_OAUTH_SCOPES);
        query.append_pair("state", state);
        query.append_pair("code_challenge", code_challenge);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("access_type", "offline");
        query.append_pair("prompt", "consent");
    }
    Ok(url.to_string())
}

fn google_bind_callback_listener() -> Result<(TcpListener, String), Box<dyn Error>> {
    let listener = TcpListener::bind((
        GOOGLE_OAUTH_REDIRECT_HOST,
        GOOGLE_OAUTH_DEFAULT_REDIRECT_PORT,
    ))
    .or_else(|_| TcpListener::bind((GOOGLE_OAUTH_REDIRECT_HOST, 0)))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("failed to configure Google callback server: {error}"))?;
    let port = listener.local_addr()?.port();
    Ok((
        listener,
        format!(
            "http://{}:{}{}",
            GOOGLE_OAUTH_REDIRECT_HOST, port, GOOGLE_OAUTH_CALLBACK_PATH
        ),
    ))
}

fn google_wait_for_callback(
    listener: TcpListener,
    timeout_seconds: f64,
) -> Result<Option<SpotifyCallbackResult>, Box<dyn Error>> {
    let deadline = std::time::Instant::now() + Duration::from_secs_f64(timeout_seconds.max(5.0));
    let mut buffer = [0u8; 8192];
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let size = stream.read(&mut buffer)?;
                let request = String::from_utf8_lossy(&buffer[..size]);
                let first_line = request.lines().next().unwrap_or_default();
                if let Some(target) = first_line
                    .strip_prefix("GET ")
                    .and_then(|value| value.split_whitespace().next())
                {
                    let parsed = Url::parse(&format!("http://localhost{target}"))?;
                    if parsed.path() == GOOGLE_OAUTH_CALLBACK_PATH {
                        let mut result = SpotifyCallbackResult::default();
                        for (key, value) in parsed.query_pairs() {
                            match key.as_ref() {
                                "code" => result.code = Some(value.into_owned()),
                                "state" => result.state = Some(value.into_owned()),
                                "error" => result.error = Some(value.into_owned()),
                                "error_description" => {
                                    result.error_description = Some(value.into_owned())
                                }
                                _ => {}
                            }
                        }
                        let message = if result.error.is_some() {
                            "Google authorization failed. You can close this tab."
                        } else {
                            "Google authorization received. You can close this tab."
                        };
                        let html = format!("<html><body><h1>{message}</h1></body></html>");
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                            html.len(),
                            html
                        );
                        let _ = stream.write_all(response.as_bytes());
                        return Ok(Some(result));
                    }
                }
                let response =
                    b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 10\r\n\r\nNot found.";
                let _ = stream.write_all(response);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return Ok(None);
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(format!("Google callback server failed: {error}").into()),
        }
    }
}

fn google_prompt_pasted_callback(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<SpotifyCallbackResult, Box<dyn Error>> {
    let raw = prompt_line(input, output, "Callback URL or code")?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Google setup cancelled: empty authorization code.".into());
    }
    let mut result = SpotifyCallbackResult::default();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let parsed = Url::parse(trimmed)?;
        for (key, value) in parsed.query_pairs() {
            match key.as_ref() {
                "code" => result.code = Some(value.into_owned()),
                "state" => result.state = Some(value.into_owned()),
                "error" => result.error = Some(value.into_owned()),
                "error_description" => result.error_description = Some(value.into_owned()),
                _ => {}
            }
        }
        return Ok(result);
    }
    if let Some(query) = trimmed.strip_prefix('?') {
        let parsed = Url::parse(&format!(
            "http://localhost{GOOGLE_OAUTH_CALLBACK_PATH}?{query}"
        ))?;
        for (key, value) in parsed.query_pairs() {
            match key.as_ref() {
                "code" => result.code = Some(value.into_owned()),
                "state" => result.state = Some(value.into_owned()),
                "error" => result.error = Some(value.into_owned()),
                "error_description" => result.error_description = Some(value.into_owned()),
                _ => {}
            }
        }
        return Ok(result);
    }
    result.code = Some(trimmed.to_string());
    Ok(result)
}

fn google_exchange_code_for_tokens(
    client_id: &str,
    client_secret: &str,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
    timeout_seconds: f64,
) -> Result<JsonMap<String, JsonValue>, Box<dyn Error>> {
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()?;
    let response = client
        .post(google_token_url())
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("code_verifier", code_verifier),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("redirect_uri", redirect_uri),
        ])
        .send()
        .map_err(|error| format!("Google token exchange failed: {error}"))?;
    if response.status().as_u16() >= 400 {
        let detail = response.text().unwrap_or_default();
        let suffix = if detail.trim().is_empty() {
            String::new()
        } else {
            format!(" Response: {}", detail.trim())
        };
        return Err(format!("Google token exchange failed.{suffix}").into());
    }
    let payload: JsonValue = response.json()?;
    payload
        .as_object()
        .cloned()
        .ok_or_else(|| "Google token response was not a JSON object.".into())
}

fn google_fetch_user_email(
    access_token: &str,
    timeout_seconds: f64,
) -> Result<String, Box<dyn Error>> {
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()?;
    let response = client
        .get(google_userinfo_url())
        .query(&[("alt", "json")])
        .header("Authorization", format!("Bearer {access_token}"))
        .send()
        .map_err(|error| format!("Google userinfo request failed: {error}"))?;
    if response.status().as_u16() >= 400 {
        return Ok(String::new());
    }
    let payload: JsonValue = response.json()?;
    Ok(payload
        .as_object()
        .and_then(|mapping| json_string(mapping, "email"))
        .unwrap_or("")
        .to_string())
}

fn save_google_gemini_oauth_file(
    hermes_home: &Path,
    access_token: &str,
    refresh_token: &str,
    expires_in: i64,
    email: &str,
) -> Result<(), Box<dyn Error>> {
    let oauth_path = google_gemini_oauth_path(hermes_home);
    if let Some(parent) = oauth_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let expires_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0)
        .saturating_add(expires_in.max(60).saturating_mul(1000));
    let payload = serde_json::json!({
        "refresh": refresh_token,
        "access": access_token,
        "expires": expires_ms,
        "email": email,
    });
    let rendered = format!("{}\n", serde_json::to_string_pretty(&payload)?);
    atomic_write(&oauth_path, rendered.as_bytes())
}

fn finalize_google_gemini_auth_add(
    context: &HermesContext,
    requested_label: Option<&str>,
    default_label: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_in: Option<i64>,
    email: &str,
) -> Result<(), Box<dyn Error>> {
    clear_provider_suppressions(context.hermes_home().as_path(), "google-gemini-cli")?;
    let label = requested_label
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            if email.trim().is_empty() {
                default_label.to_string()
            } else {
                email.trim().to_string()
            }
        });
    let expires_at_ms = expires_in.map(|seconds| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_millis() as i64)
            .unwrap_or(0)
            .saturating_add(seconds.max(60).saturating_mul(1000))
    });
    let count = add_auth_pool_entry(
        context.hermes_home().as_path(),
        "google-gemini-cli",
        NewPoolEntry {
            label: label.clone(),
            auth_type: "oauth".to_string(),
            source: "manual:google_pkce".to_string(),
            access_token: access_token.to_string(),
            refresh_token: refresh_token.map(ToOwned::to_owned),
            base_url: None,
            expires_at_ms,
            last_refresh: None,
        },
    )?;
    println!(
        "Added google-gemini-cli OAuth credential #{}: \"{}\"",
        count, label
    );
    Ok(())
}

fn minimax_code_verifier() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_MINIMAX_TEST_VERIFIER") {
        return Ok(override_value);
    }
    spotify_random_token(64).map(|value| value.chars().take(96).collect())
}

fn minimax_state_nonce() -> Result<String, Box<dyn Error>> {
    if let Some(override_value) = env_trimmed("HERMES_AUTH_MINIMAX_TEST_STATE") {
        return Ok(override_value);
    }
    spotify_random_token(16)
}

fn minimax_request_user_code(
    client: &Client,
    portal_base_url: &str,
    client_id: &str,
    scope: &str,
    code_challenge: &str,
    state: &str,
) -> Result<JsonMap<String, JsonValue>, Box<dyn Error>> {
    let response = client
        .post(format!(
            "{}/oauth/code",
            portal_base_url.trim_end_matches('/')
        ))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .header(
            "x-request-id",
            spotify_random_token(12).unwrap_or_else(|_| "hermes-minimax".to_string()),
        )
        .form(&[
            ("response_type", "code"),
            ("client_id", client_id),
            ("scope", scope),
            ("code_challenge", code_challenge),
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
        return Err(format!("MiniMax OAuth authorization failed: {detail}").into());
    }
    let payload = serde_json::from_str::<JsonValue>(&body).map_err(|error| {
        format!("MiniMax OAuth authorization response was invalid JSON: {error}")
    })?;
    let payload = payload
        .as_object()
        .cloned()
        .ok_or("MiniMax OAuth authorization response was not a JSON object.")?;
    for field in ["user_code", "verification_uri", "expired_in"] {
        if !payload.contains_key(field) {
            return Err(format!("MiniMax OAuth response missing field: {field}").into());
        }
    }
    if json_string(&payload, "state") != Some(state) {
        return Err("MiniMax OAuth state mismatch (possible CSRF).".into());
    }
    Ok(payload)
}

fn minimax_poll_token(
    client: &Client,
    portal_base_url: &str,
    client_id: &str,
    user_code: &str,
    code_verifier: &str,
    expired_in: i64,
    interval_ms: Option<i64>,
) -> Result<JsonMap<String, JsonValue>, Box<dyn Error>> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0);
    let deadline = if expired_in > now_ms / 2 {
        UNIX_EPOCH + Duration::from_millis(expired_in.max(1) as u64)
    } else {
        SystemTime::now() + Duration::from_secs(expired_in.max(1) as u64)
    };
    let interval = Duration::from_millis(interval_ms.unwrap_or(2000).max(2000) as u64);

    while SystemTime::now() < deadline {
        let response = client
            .post(format!(
                "{}/oauth/token",
                portal_base_url.trim_end_matches('/')
            ))
            .header("Content-Type", "application/x-www-form-urlencoded")
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
                .and_then(|base_resp| json_string(base_resp, "status_msg"))
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| body.trim().to_string());
            let detail = if detail.is_empty() {
                "unknown".to_string()
            } else {
                detail
            };
            return Err(format!("MiniMax OAuth error: {detail}").into());
        }

        match json_string(&payload, "status") {
            Some("success") => {
                if !["access_token", "refresh_token", "expired_in"]
                    .iter()
                    .all(|field| payload.get(*field).is_some())
                {
                    return Err(
                        "MiniMax OAuth success payload missing required token fields.".into(),
                    );
                }
                return Ok(payload);
            }
            Some("error") => {
                return Err("MiniMax OAuth reported an error. Please try again later.".into());
            }
            _ => thread::sleep(interval),
        }
    }

    Err("MiniMax OAuth timed out before authorization completed.".into())
}

fn save_minimax_provider_state(
    hermes_home: &Path,
    portal_base_url: &str,
    inference_base_url: &str,
    client_id: &str,
    scope: &str,
    access_token: &str,
    refresh_token: &str,
    expires_in: i64,
    token_type: Option<&str>,
    resource_url: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let obtained_at = Utc::now();
    let expires_at = (obtained_at + ChronoDuration::seconds(expires_in.max(1)))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let mut state = JsonMap::new();
    state.insert(
        "provider".to_string(),
        JsonValue::String("minimax-oauth".to_string()),
    );
    state.insert(
        "region".to_string(),
        JsonValue::String(
            match portal_base_url {
                DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL => "cn",
                _ => "global",
            }
            .to_string(),
        ),
    );
    state.insert(
        "portal_base_url".to_string(),
        JsonValue::String(portal_base_url.to_string()),
    );
    state.insert(
        "inference_base_url".to_string(),
        JsonValue::String(inference_base_url.to_string()),
    );
    state.insert(
        "client_id".to_string(),
        JsonValue::String(client_id.to_string()),
    );
    state.insert("scope".to_string(), JsonValue::String(scope.to_string()));
    state.insert(
        "access_token".to_string(),
        JsonValue::String(access_token.to_string()),
    );
    state.insert(
        "refresh_token".to_string(),
        JsonValue::String(refresh_token.to_string()),
    );
    state.insert(
        "obtained_at".to_string(),
        JsonValue::String(obtained_at.to_rfc3339()),
    );
    state.insert("expires_at".to_string(), JsonValue::String(expires_at));
    state.insert("expires_in".to_string(), JsonValue::from(expires_in.max(1)));
    if let Some(value) = token_type.map(str::trim).filter(|value| !value.is_empty()) {
        state.insert(
            "token_type".to_string(),
            JsonValue::String(value.to_string()),
        );
    }
    if let Some(value) = resource_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        state.insert(
            "resource_url".to_string(),
            JsonValue::String(value.to_string()),
        );
    }
    store_provider_state(&mut auth_store, "minimax-oauth", state)?;
    save_auth_store_json(hermes_home, &auth_store)
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
        .or_else(|| {
            get_provider_profile("openai-codex")
                .map(|profile| profile.base_url.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "https://chatgpt.com/backend-api/codex".to_string())
}

fn codex_oauth_max_wait_seconds() -> f64 {
    env_trimmed("HERMES_AUTH_CODEX_MAX_WAIT_SECONDS")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or(15.0 * 60.0)
}

fn codex_now_rfc3339() -> String {
    Utc::now().to_rfc3339().replace("+00:00", "Z")
}

fn decode_jwt_claims(token: &str) -> Option<JsonMap<String, JsonValue>> {
    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload.as_bytes()).ok()?;
    serde_json::from_slice::<JsonValue>(&decoded)
        .ok()?
        .as_object()
        .cloned()
}

fn non_empty_trimmed_owned(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn resolve_auth_remove_target<'a>(
    entries: &'a [PoolEntry],
    target: &str,
) -> (Option<usize>, Option<&'a PoolEntry>, Option<String>) {
    let raw = target.trim();
    if raw.is_empty() {
        return (
            None,
            None,
            Some("No credential target provided.".to_string()),
        );
    }
    for (idx, entry) in entries.iter().enumerate() {
        if entry.id == raw {
            return (Some(idx + 1), Some(entry), None);
        }
    }
    let label_matches = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.label.trim().eq_ignore_ascii_case(raw))
        .collect::<Vec<_>>();
    if label_matches.len() == 1 {
        let (idx, entry) = label_matches[0];
        return (Some(idx + 1), Some(entry), None);
    }
    if label_matches.len() > 1 {
        return (
            None,
            None,
            Some(format!(
                "Ambiguous credential label \"{}\". Use the numeric index or entry id instead.",
                raw
            )),
        );
    }
    if let Ok(index) = raw.parse::<usize>() {
        if (1..=entries.len()).contains(&index) {
            return (Some(index), entries.get(index - 1), None);
        }
        return (None, None, Some(format!("No credential #{}.", index)));
    }
    (
        None,
        None,
        Some(format!("No credential matching \"{}\".", raw)),
    )
}

fn entry_source_is_manual(source: &str) -> bool {
    let normalized = source.trim().to_ascii_lowercase();
    normalized == "manual" || normalized.starts_with("manual:")
}

fn remove_manual_auth_entry(
    hermes_home: &Path,
    provider: &str,
    index: usize,
) -> Result<(), Box<dyn Error>> {
    let mut auth_store = load_auth_store_json(hermes_home)?;
    let root = auth_store
        .as_object_mut()
        .ok_or("auth store is not a JSON object")?;
    let Some(pool) = root
        .get_mut("credential_pool")
        .and_then(JsonValue::as_object_mut)
    else {
        return Err(format!("No credential #{}.", index).into());
    };
    let Some(entries) = pool.get_mut(provider).and_then(JsonValue::as_array_mut) else {
        return Err(format!("No credential #{}.", index).into());
    };
    if index == 0 || index > entries.len() {
        return Err(format!("No credential #{}.", index).into());
    }
    entries.remove(index - 1);
    for (priority, entry) in entries.iter_mut().enumerate() {
        if let Some(mapping) = entry.as_object_mut() {
            mapping.insert("priority".to_string(), JsonValue::from(priority as i64));
        }
    }
    save_auth_store_json(hermes_home, &auth_store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::HermesContext;
    use serde_json::json;
    use std::io::{Cursor, Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-auth-{label}-{unique}"))
    }

    fn spawn_spotify_token_server(
        response: JsonValue,
    ) -> (String, Arc<Mutex<String>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let request_body = Arc::new(Mutex::new(String::new()));
        let request_body_clone = Arc::clone(&request_body);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 8192];
            let size = stream.read(&mut buffer).unwrap();
            let request = String::from_utf8_lossy(&buffer[..size]).to_string();
            let body = request
                .split("\r\n\r\n")
                .nth(1)
                .unwrap_or_default()
                .to_string();
            *request_body_clone.lock().unwrap() = body;
            let payload = serde_json::to_string(&response).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (base_url, request_body, handle)
    }

    fn spawn_google_oauth_server(
        access_token: &str,
        refresh_token: &str,
        email: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let email = email.to_string();
        let handle = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                let first_line = request.lines().next().unwrap_or_default().to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let (status_line, payload) = if first_line.starts_with("POST /token ") {
                    (
                        "HTTP/1.1 200 OK",
                        json!({
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                            "expires_in": 3600,
                            "token_type": "Bearer"
                        })
                        .to_string(),
                    )
                } else if first_line.starts_with("GET /userinfo?alt=json ") {
                    (
                        "HTTP/1.1 200 OK",
                        json!({
                            "email": email
                        })
                        .to_string(),
                    )
                } else {
                    (
                        "HTTP/1.1 404 Not Found",
                        json!({
                            "error": "not_found"
                        })
                        .to_string(),
                    )
                };
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_minimax_oauth_server(
        access_token: &str,
        refresh_token: &str,
        state: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let state = state.to_string();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let payload = match idx {
                    0 => json!({
                        "user_code": "MINIMAX-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "expired_in": 60,
                        "interval": 100,
                        "state": state
                    }),
                    _ => json!({
                        "status": "success",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "expired_in": 3600,
                        "token_type": "Bearer",
                        "resource_url": "group-123",
                        "notification_message": "quota synced"
                    }),
                }
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_codex_oauth_server(
        access_token: &str,
        refresh_token: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let handle = thread::spawn(move || {
            for idx in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let (status_line, payload) = match idx {
                    0 => (
                        "HTTP/1.1 200 OK",
                        json!({
                            "user_code": "USER-CODE",
                            "device_auth_id": "device-auth-123",
                            "interval": 1
                        })
                        .to_string(),
                    ),
                    1 => (
                        "HTTP/1.1 200 OK",
                        json!({
                            "authorization_code": "auth-code-123",
                            "code_verifier": "verifier-xyz"
                        })
                        .to_string(),
                    ),
                    _ => (
                        "HTTP/1.1 200 OK",
                        json!({
                            "access_token": access_token,
                            "refresh_token": refresh_token
                        })
                        .to_string(),
                    ),
                };
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_nous_oauth_server(
        access_token: &str,
        refresh_token: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let payload = match idx {
                    0 => json!({
                        "device_code": "nous-device-code",
                        "user_code": "NOUS-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "verification_uri_complete": format!("{base_for_thread}/verify?code=NOUS-CODE"),
                        "expires_in": 60,
                        "interval": 1
                    }),
                    1 => json!({
                        "access_token": access_token,
                        "refresh_token": refresh_token,
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
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_nous_shared_import_server(
        access_token: &str,
        refresh_token: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let payload = match idx {
                    0 => json!({
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "Bearer",
                        "scope": "inference:mint_agent_key offline_access",
                        "expires_in": 3600,
                        "inference_base_url": format!("{base_for_thread}/refreshed-inference/v1")
                    }),
                    _ => json!({
                        "api_key": "nous-agent-key-imported",
                        "key_id": "agent-key-import-123",
                        "expires_at": "2999-01-03T00:00:00Z",
                        "expires_in": 86400,
                        "inference_base_url": format!("{base_for_thread}/runtime-inference/v1"),
                        "reused": false
                    }),
                }
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    #[test]
    fn reset_config_provider_switches_model_provider_to_auto() {
        let path = temp_path("config").join("config.yaml");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "model:\n  provider: openai-codex\n  base_url: https://chatgpt.com/backend-api/codex\n",
        )
        .unwrap();
        assert!(reset_config_provider(&path, "openai-codex").unwrap());
        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("provider: auto"));
        assert!(written.contains(OPENROUTER_BASE_URL));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn logout_uses_active_provider_and_clears_state() {
        let home = temp_path("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            r#"{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token":"a","refresh_token":"r"}}}}"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "model:\n  provider: openai-codex\n",
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = context.load_config_document().unwrap();
        logout_provider(&context, &loaded, None).unwrap();
        let auth = fs::read_to_string(home.join("auth.json")).unwrap();
        assert!(auth.contains("\"active_provider\": null"));
        let config = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config.contains("provider: auto"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn spotify_login_persists_native_provider_state() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("spotify-login");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (accounts_base_url, request_body, server) = spawn_spotify_token_server(json!({
            "access_token": "spotify-access",
            "refresh_token": "spotify-refresh",
            "token_type": "Bearer",
            "scope": "user-read-playback-state",
            "expires_in": 3600
        }));
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let redirect_uri = format!(
            "http://127.0.0.1:{}/spotify/callback",
            probe.local_addr().unwrap().port()
        );
        drop(probe);
        let callback_url = format!("{}?code=test-code&state=test-state", redirect_uri);
        let callback_thread = thread::spawn(move || {
            for _ in 0..30 {
                if reqwest::blocking::get(&callback_url).is_ok() {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
        });

        unsafe {
            std::env::set_var("HERMES_AUTH_SPOTIFY_TEST_STATE", "test-state");
            std::env::set_var("HERMES_AUTH_SPOTIFY_TEST_VERIFIER", "fixed-verifier");
            std::env::set_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL", &accounts_base_url);
            std::env::set_var("HERMES_SPOTIFY_API_BASE_URL", "https://api.spotify.test/v1");
        }
        let mut input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();
        let result = run_native_spotify_login_with_io(
            &context,
            &SpotifyAuthArgs {
                spotify_action: SpotifyAuthAction::Login,
                client_id: Some("client-123".to_string()),
                redirect_uri: Some(redirect_uri.clone()),
                scope: Some("user-read-playback-state".to_string()),
                no_browser: true,
                timeout: Some(5.0),
            },
            &mut input,
            &mut output,
        );
        unsafe {
            std::env::remove_var("HERMES_AUTH_SPOTIFY_TEST_STATE");
            std::env::remove_var("HERMES_AUTH_SPOTIFY_TEST_VERIFIER");
            std::env::remove_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL");
            std::env::remove_var("HERMES_SPOTIFY_API_BASE_URL");
        }

        result.unwrap();
        callback_thread.join().unwrap();
        server.join().unwrap();
        let body = request_body.lock().unwrap().clone();
        assert!(body.contains("client_id=client-123"));
        assert!(body.contains("grant_type=authorization_code"));
        assert!(body.contains("code=test-code"));
        assert!(body.contains("code_verifier=fixed-verifier"));
        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let state = &persisted["providers"]["spotify"];
        assert_eq!(state["client_id"], "client-123");
        assert_eq!(state["redirect_uri"], redirect_uri);
        assert_eq!(state["access_token"], "spotify-access");
        assert_eq!(state["refresh_token"], "spotify-refresh");
        assert_eq!(state["api_base_url"], "https://api.spotify.test/v1");
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Starting Spotify PKCE login..."));
        assert!(rendered.contains("Spotify login successful!"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn spotify_interactive_setup_saves_client_id() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("spotify-setup");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        unsafe { std::env::set_var("SSH_TTY", "1") };
        let mut input = Cursor::new(b"client-xyz\n".to_vec());
        let mut output = Vec::new();

        let client_id = spotify_interactive_setup_with_io(
            &context,
            &mut input,
            &mut output,
            "http://127.0.0.1:44991/spotify/callback",
        )
        .unwrap();

        unsafe { std::env::remove_var("SSH_TTY") };
        assert_eq!(client_id, "client-xyz");
        let env_text = fs::read_to_string(home.join(".env")).unwrap();
        assert!(env_text.contains("HERMES_SPOTIFY_CLIENT_ID=client-xyz"));
        assert!(
            env_text
                .contains("HERMES_SPOTIFY_REDIRECT_URI=http://127.0.0.1:44991/spotify/callback")
        );
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Spotify first-time setup"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn spotify_redirect_validation_rejects_invalid_scheme() {
        let error = spotify_validate_redirect_uri("https://example.com/callback")
            .unwrap_err()
            .to_string();
        assert!(error.contains("redirect_uri must use http://localhost or http://127.0.0.1"));
    }

    #[test]
    fn auth_list_shows_current_and_exhausted_entries() {
        let home = temp_path("auth-list");
        fs::create_dir_all(&home).unwrap();
        let now = now_unix_seconds();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "rate1",
                            "label": "rate-limited",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual:dashboard",
                            "last_status": "exhausted",
                            "last_status_at": now,
                            "last_error_code": 429
                        },
                        {
                            "id": "ok2",
                            "label": "ready",
                            "auth_type": "oauth",
                            "priority": 1,
                            "source": "manual"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        let pool = load_credential_pool(context.hermes_home().as_path()).unwrap();
        let rendered = render_auth_list(&pool, Some("openrouter"));
        assert!(rendered.contains("openrouter (2 credentials):"));
        assert!(rendered.contains("#1  rate-limited"));
        assert!(rendered.contains("rate-limited (429)"));
        assert!(rendered.contains("#2  ready"));
        assert!(rendered.contains("oauth"));
        assert!(rendered.contains("←"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_reset_clears_pool_status_fields() {
        let home = temp_path("auth-reset");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "a1",
                            "label": "primary",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "last_status": "exhausted",
                            "last_status_at": 123.0,
                            "last_error_code": 429,
                            "last_error_reason": "rate_limit",
                            "last_error_message": "Too many requests",
                            "last_error_reset_at": 456.0
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        let count =
            reset_auth_pool_statuses(context.hermes_home().as_path(), "openrouter").unwrap();
        assert_eq!(count, 1);
        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["openrouter"][0];
        assert!(entry["last_status"].is_null());
        assert!(entry["last_status_at"].is_null());
        assert!(entry["last_error_code"].is_null());
        assert!(entry["last_error_reason"].is_null());
        assert!(entry["last_error_message"].is_null());
        assert!(entry["last_error_reset_at"].is_null());
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_api_key_saves_native_pool_entry() {
        let home = temp_path("auth-add");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "suppressed_sources": {
                    "openrouter": ["env:OPENROUTER_API_KEY"]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        native_auth_add_api_key(
            &context,
            &AuthAddArgs {
                provider: "or".to_string(),
                auth_type: Some("api-key".to_string()),
                label: Some("primary".to_string()),
                api_key: Some("sk-openrouter".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["openrouter"][0];
        assert_eq!(entry["label"], "primary");
        assert_eq!(entry["auth_type"], "api_key");
        assert_eq!(entry["source"], "manual");
        assert_eq!(entry["access_token"], "sk-openrouter");
        assert_eq!(entry["base_url"], OPENROUTER_BASE_URL);
        assert!(
            persisted
                .get("suppressed_sources")
                .and_then(JsonValue::as_object)
                .and_then(|sources| sources.get("openrouter"))
                .is_none()
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_custom_api_key_from_provider_key_stays_native() {
        let home = temp_path("auth-add-custom");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "providers:\n  demo-endpoint:\n    name: Demo Provider\n    api: https://demo.example/v1/\n",
        )
        .unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "suppressed_sources": {
                    "custom:demo-provider": ["config:demo-provider"]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "demo-endpoint".to_string(),
                label: Some("primary".to_string()),
                api_key: Some("sk-demo".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["custom:demo-provider"][0];
        assert_eq!(entry["label"], "primary");
        assert_eq!(entry["auth_type"], "api_key");
        assert_eq!(entry["source"], "manual");
        assert_eq!(entry["access_token"], "sk-demo");
        assert_eq!(entry["base_url"], "https://demo.example/v1");
        assert!(
            persisted
                .get("suppressed_sources")
                .and_then(JsonValue::as_object)
                .and_then(|sources| sources.get("custom:demo-provider"))
                .is_none()
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_anthropic_oauth_uses_native_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-add-anthropic");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "suppressed_sources": {
                    "anthropic": ["claude_code", "hermes_pkce"]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (token_url, request_body, server) = spawn_spotify_token_server(json!({
            "access_token": "header.eyJlbWFpbCI6ImNsYXVkZUBleGFtcGxlLmNvbSJ9.sig",
            "refresh_token": "anth-refresh",
            "expires_in": 1800
        }));

        unsafe {
            std::env::set_var(
                "HERMES_AUTH_ANTHROPIC_AUTHORIZE_URL",
                "https://claude.test/oauth/authorize",
            );
            std::env::set_var("HERMES_AUTH_ANTHROPIC_TOKEN_URL", &token_url);
            std::env::set_var("HERMES_AUTH_ANTHROPIC_TEST_VERIFIER", "anth-verifier");
        }
        let mut input = Cursor::new(b"grant-code#returned-state\n".to_vec());
        let mut output = Vec::new();
        let result = native_auth_add_anthropic_oauth_with_io(
            &context,
            &AuthAddArgs {
                provider: "anthropic".to_string(),
                auth_type: Some("oauth".to_string()),
                no_browser: true,
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
            &mut input,
            &mut output,
        );
        unsafe {
            std::env::remove_var("HERMES_AUTH_ANTHROPIC_AUTHORIZE_URL");
            std::env::remove_var("HERMES_AUTH_ANTHROPIC_TOKEN_URL");
            std::env::remove_var("HERMES_AUTH_ANTHROPIC_TEST_VERIFIER");
        }

        result.unwrap();
        server.join().unwrap();
        let body = request_body.lock().unwrap().clone();
        assert!(body.contains("\"grant_type\":\"authorization_code\""));
        assert!(body.contains("\"client_id\":\"9d1c250a-e61b-44d9-88ed-5944d1962f5e\""));
        assert!(body.contains("\"code\":\"grant-code\""));
        assert!(body.contains("\"state\":\"returned-state\""));
        assert!(body.contains("\"code_verifier\":\"anth-verifier\""));

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["anthropic"][0];
        assert_eq!(entry["label"], "claude@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:hermes_pkce");
        assert_eq!(entry["refresh_token"], "anth-refresh");
        assert!(entry["expires_at_ms"].as_i64().unwrap() > 0);
        assert!(
            persisted
                .get("suppressed_sources")
                .and_then(JsonValue::as_object)
                .and_then(|sources| sources.get("anthropic"))
                .is_none()
        );
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Authorize Hermes with your Claude Pro/Max subscription."));
        assert!(rendered.contains("Added anthropic OAuth credential #1"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_oauth_uses_python_fallback() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = temp_path("auth-add-python");
        let log_path = temp.join("auth-add.log");
        let python = temp.join("python3");
        fs::create_dir_all(&temp).unwrap();
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'provider=%s\\ntype=%s\\nclient_id=%s\\n' \"$HERMES_AUTH_ADD_PROVIDER\" \"$HERMES_AUTH_ADD_TYPE\" \"$HERMES_AUTH_ADD_CLIENT_ID\" > \"{}\"\nprintf '%s\\n' \"$@\" >> \"{}\"\nexit 0\n",
                log_path.display(),
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe { std::env::set_var("HERMES_AUTH_PYTHON", &python) };
        let result = run_python_auth_add(&AuthAddArgs {
            provider: "openai-codex".to_string(),
            auth_type: Some("oauth".to_string()),
            client_id: Some("client-abc".to_string()),
            ..AuthAddArgs::default()
        });
        unsafe { std::env::remove_var("HERMES_AUTH_PYTHON") };

        result.unwrap();
        let logged = fs::read_to_string(log_path).unwrap();
        assert!(logged.contains("provider=openai-codex"));
        assert!(logged.contains("type=oauth"));
        assert!(logged.contains("client_id=client-abc"));
        assert!(logged.contains("-c"));
        assert!(logged.contains("auth_add_command"));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn auth_add_google_gemini_oauth_uses_native_runtime() {
        let home = temp_path("auth-add-google");
        let auth_dir = home.join("auth");
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
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "google-gemini-cli".to_string(),
                auth_type: Some("oauth".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["google-gemini-cli"][0];
        assert_eq!(entry["label"], "dev@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:google_pkce");
        assert_eq!(entry["access_token"], "google-fresh");
        assert_eq!(entry["refresh_token"], "google-refresh");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_google_gemini_oauth_without_state_uses_native_pkce_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-add-google-device");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "suppressed_sources": {
                    "google-gemini-cli": ["manual:google_pkce"]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (base_url, requests, server) = spawn_google_oauth_server(
            "google-access-123",
            "google-refresh-123",
            "gemini@example.com",
        );

        unsafe {
            std::env::set_var(
                "HERMES_AUTH_GOOGLE_AUTH_URL",
                format!("{base_url}/authorize"),
            );
            std::env::set_var("HERMES_AUTH_GOOGLE_TOKEN_URL", format!("{base_url}/token"));
            std::env::set_var(
                "HERMES_AUTH_GOOGLE_USERINFO_URL",
                format!("{base_url}/userinfo"),
            );
            std::env::set_var("HERMES_AUTH_GOOGLE_TEST_STATE", "google-state-123");
            std::env::set_var("HERMES_AUTH_GOOGLE_TEST_VERIFIER", "fixed-google-verifier");
            std::env::set_var("HERMES_GEMINI_CLIENT_ID", "test-google-client");
            std::env::set_var("HERMES_GEMINI_CLIENT_SECRET", "test-google-secret");
        }
        let mut input = Cursor::new(
            b"http://127.0.0.1:8085/oauth2callback?code=google-code-123&state=google-state-123\n"
                .to_vec(),
        );
        let mut output = Vec::new();
        let result = native_auth_add_google_gemini_oauth_with_io(
            &context,
            &AuthAddArgs {
                provider: "google-gemini-cli".to_string(),
                auth_type: Some("oauth".to_string()),
                no_browser: true,
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
            &mut input,
            &mut output,
        );
        unsafe {
            std::env::remove_var("HERMES_AUTH_GOOGLE_AUTH_URL");
            std::env::remove_var("HERMES_AUTH_GOOGLE_TOKEN_URL");
            std::env::remove_var("HERMES_AUTH_GOOGLE_USERINFO_URL");
            std::env::remove_var("HERMES_AUTH_GOOGLE_TEST_STATE");
            std::env::remove_var("HERMES_AUTH_GOOGLE_TEST_VERIFIER");
            std::env::remove_var("HERMES_GEMINI_CLIENT_ID");
            std::env::remove_var("HERMES_GEMINI_CLIENT_SECRET");
        }

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].starts_with("POST /token "));
        assert!(captured[0].contains("grant_type=authorization_code"));
        assert!(captured[0].contains("code=google-code-123"));
        assert!(captured[0].contains("code_verifier=fixed-google-verifier"));
        assert!(captured[0].contains("client_id=test-google-client"));
        assert!(captured[0].contains("client_secret=test-google-secret"));
        assert!(
            captured[0].contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A8085%2Foauth2callback")
        );
        assert!(captured[1].starts_with("GET /userinfo?alt=json "));
        assert!(
            captured[1]
                .to_ascii_lowercase()
                .contains("authorization: bearer google-access-123")
        );

        let oauth_state: JsonValue = serde_json::from_str(
            &fs::read_to_string(home.join("auth").join("google_oauth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(oauth_state["access"], "google-access-123");
        assert_eq!(oauth_state["refresh"], "google-refresh-123");
        assert_eq!(oauth_state["email"], "gemini@example.com");
        assert!(oauth_state["expires"].as_i64().unwrap() > 0);

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["google-gemini-cli"][0];
        assert_eq!(entry["label"], "gemini@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:google_pkce");
        assert_eq!(entry["access_token"], "google-access-123");
        assert_eq!(entry["refresh_token"], "google-refresh-123");
        assert!(entry["expires_at_ms"].as_i64().unwrap() > 0);
        assert!(
            persisted
                .get("suppressed_sources")
                .and_then(JsonValue::as_object)
                .and_then(|sources| sources.get("google-gemini-cli"))
                .is_none()
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Open this URL to authorize Hermes with Google Gemini CLI:"));
        assert!(rendered.contains(&format!("{base_url}/authorize")));
        assert!(
            rendered.contains("After signing in, paste the full callback URL or just the code.")
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_qwen_oauth_uses_native_runtime() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-add-qwen-home");
        let qwen_dir = home.join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "header.eyJlbWFpbCI6InF3ZW5AZXhhbXBsZS5jb20ifQ.sig",
                "refresh_token": "qwen-refresh",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let previous_home = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &home) };

        let result = print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "qwen-oauth".to_string(),
                auth_type: Some("oauth".to_string()),
                ..AuthAddArgs::default()
            },
        );

        match previous_home {
            Some(value) => unsafe { std::env::set_var("HOME", value) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        result.unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["qwen-oauth"][0];
        assert_eq!(entry["label"], "qwen@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:qwen_cli");
        assert_eq!(entry["base_url"], "https://portal.qwen.ai/v1");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_minimax_oauth_uses_native_runtime() {
        let home = temp_path("auth-add-minimax");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "minimax-oauth": {
                        "access_token": "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pQGV4YW1wbGUuY29tIn0.sig",
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
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "minimax-oauth".to_string(),
                auth_type: Some("oauth".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["minimax-oauth"][0];
        assert_eq!(entry["label"], "mini@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:minimax_oauth");
        assert_eq!(entry["base_url"], "https://api.minimax.io/anthropic");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_minimax_oauth_without_state_uses_native_login_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-add-minimax-login");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "suppressed_sources": {
                    "minimax-oauth": ["manual:minimax_oauth"]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (portal_base_url, requests, server) = spawn_minimax_oauth_server(
            "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pbWF4QGV4YW1wbGUuY29tIn0.sig",
            "minimax-refresh-2",
            "minimax-state-123",
        );

        unsafe {
            std::env::set_var("HERMES_AUTH_MINIMAX_TEST_VERIFIER", "minimax-verifier");
            std::env::set_var("HERMES_AUTH_MINIMAX_TEST_STATE", "minimax-state-123");
        }
        let mut output = Vec::new();
        let result = native_auth_add_minimax_oauth_with_io(
            &context,
            &AuthAddArgs {
                provider: "minimax-oauth".to_string(),
                auth_type: Some("oauth".to_string()),
                portal_url: Some(portal_base_url.clone()),
                inference_url: Some(format!("{portal_base_url}/anthropic")),
                client_id: Some("minimax-client-test".to_string()),
                scope: Some("group_id profile model.completion".to_string()),
                no_browser: true,
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
            &mut output,
        );
        unsafe {
            std::env::remove_var("HERMES_AUTH_MINIMAX_TEST_VERIFIER");
            std::env::remove_var("HERMES_AUTH_MINIMAX_TEST_STATE");
        }

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].starts_with("POST /oauth/code "));
        assert!(captured[0].contains("response_type=code"));
        assert!(captured[0].contains("client_id=minimax-client-test"));
        assert!(captured[0].contains("scope=group_id+profile+model.completion"));
        assert!(captured[0].contains("code_challenge_method=S256"));
        assert!(captured[0].contains("state=minimax-state-123"));
        assert!(captured[1].starts_with("POST /oauth/token "));
        assert!(
            captured[1].contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Auser_code")
        );
        assert!(captured[1].contains("client_id=minimax-client-test"));
        assert!(captured[1].contains("user_code=MINIMAX-CODE"));
        assert!(captured[1].contains("code_verifier=minimax-verifier"));

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let state = &persisted["providers"]["minimax-oauth"];
        assert_eq!(state["portal_base_url"], portal_base_url);
        assert_eq!(
            state["inference_base_url"],
            JsonValue::String(format!("{portal_base_url}/anthropic"))
        );
        assert_eq!(state["client_id"], "minimax-client-test");
        assert_eq!(state["scope"], "group_id profile model.completion");
        assert_eq!(
            state["access_token"],
            "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pbWF4QGV4YW1wbGUuY29tIn0.sig"
        );
        assert_eq!(state["refresh_token"], "minimax-refresh-2");
        assert_eq!(state["token_type"], "Bearer");
        assert_eq!(state["resource_url"], "group-123");

        let entry = &persisted["credential_pool"]["minimax-oauth"][0];
        assert_eq!(entry["label"], "minimax@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:minimax_oauth");
        assert_eq!(entry["refresh_token"], "minimax-refresh-2");
        assert_eq!(entry["base_url"], format!("{portal_base_url}/anthropic"));
        assert!(entry["expires_at_ms"].as_i64().unwrap() > 0);
        assert!(
            persisted
                .get("suppressed_sources")
                .and_then(JsonValue::as_object)
                .and_then(|sources| sources.get("minimax-oauth"))
                .is_none()
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Starting Hermes login via MiniMax OAuth..."));
        assert!(rendered.contains(&format!("Portal: {portal_base_url}")));
        assert!(rendered.contains("Waiting for approval..."));
        assert!(rendered.contains("Added minimax-oauth OAuth credential #1"));
        assert!(rendered.contains("Note from MiniMax: quota synced"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_nous_oauth_uses_native_runtime_with_existing_state() {
        let home = temp_path("auth-add-nous");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "header.eyJlbWFpbCI6Im5vdXNAZXhhbXBsZS5jb20ifQ.sig",
                        "refresh_token": "nous-refresh",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "client_id": "hermes-cli",
                        "portal_base_url": "https://portal.nous.test",
                        "inference_base_url": "https://inference.nous.test/v1",
                        "agent_key": "nous-agent-key",
                        "agent_key_expires_at": "2999-01-01T00:00:00Z",
                        "label": "Nous Main"
                    }
                },
                "credential_pool": {
                    "nous": [
                        {
                            "id": "legacy01",
                            "label": "legacy-nous",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:device_code",
                            "access_token": "legacy-access",
                            "refresh_token": "legacy-refresh",
                            "base_url": "https://legacy.nous.test/v1"
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "nous".to_string(),
                auth_type: Some("oauth".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entries = persisted["credential_pool"]["nous"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["label"], "Nous Main");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "device_code");
        assert_eq!(
            entry["access_token"],
            "header.eyJlbWFpbCI6Im5vdXNAZXhhbXBsZS5jb20ifQ.sig"
        );
        assert_eq!(entry["refresh_token"], "nous-refresh");
        assert_eq!(entry["base_url"], "https://inference.nous.test/v1");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_openai_codex_oauth_uses_native_runtime_with_existing_state() {
        let home = temp_path("auth-add-codex");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "header.eyJlbWFpbCI6ImNvZGV4QGV4YW1wbGUuY29tIn0.sig",
                            "refresh_token": "codex-refresh"
                        },
                        "last_refresh": "2026-05-08T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "openai-codex".to_string(),
                auth_type: Some("oauth".to_string()),
                ..AuthAddArgs::default()
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["openai-codex"][0];
        assert_eq!(entry["label"], "codex@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "device_code");
        assert_eq!(
            entry["access_token"],
            "header.eyJlbWFpbCI6ImNvZGV4QGV4YW1wbGUuY29tIn0.sig"
        );
        assert_eq!(entry["refresh_token"], "codex-refresh");
        assert_eq!(entry["base_url"], "https://chatgpt.com/backend-api/codex");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_openai_codex_oauth_without_state_uses_native_device_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-add-codex-device");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (issuer, requests, server) = spawn_codex_oauth_server(
            "header.eyJlbWFpbCI6ImNvZGV4LWRldmljZUBleGFtcGxlLmNvbSJ9.sig",
            "codex-refresh-2",
        );

        unsafe {
            std::env::set_var("HERMES_AUTH_CODEX_ISSUER", &issuer);
            std::env::set_var(
                "HERMES_AUTH_CODEX_TOKEN_URL",
                format!("{issuer}/oauth/token"),
            );
            std::env::set_var("HERMES_AUTH_CODEX_MAX_WAIT_SECONDS", "5");
            std::env::set_var(
                "HERMES_CODEX_BASE_URL",
                "https://codex.example.test/backend-api/codex",
            );
        }
        let mut output = Vec::new();
        let result = native_auth_add_openai_codex_oauth_with_io(
            &context,
            &AuthAddArgs {
                provider: "openai-codex".to_string(),
                auth_type: Some("oauth".to_string()),
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
            &mut output,
        );
        unsafe {
            std::env::remove_var("HERMES_AUTH_CODEX_ISSUER");
            std::env::remove_var("HERMES_AUTH_CODEX_TOKEN_URL");
            std::env::remove_var("HERMES_AUTH_CODEX_MAX_WAIT_SECONDS");
            std::env::remove_var("HERMES_CODEX_BASE_URL");
        }

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 3);
        assert!(captured[0].starts_with("POST /api/accounts/deviceauth/usercode "));
        assert!(captured[0].contains("\"client_id\":\"app_EMoamEEZ73f0CkXaXp7hrann\""));
        assert!(captured[1].starts_with("POST /api/accounts/deviceauth/token "));
        assert!(captured[1].contains("\"device_auth_id\":\"device-auth-123\""));
        assert!(captured[1].contains("\"user_code\":\"USER-CODE\""));
        assert!(captured[2].starts_with("POST /oauth/token "));
        assert!(captured[2].contains("grant_type=authorization_code"));
        assert!(captured[2].contains("code=auth-code-123"));
        assert!(captured[2].contains("code_verifier=verifier-xyz"));

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entry = &persisted["credential_pool"]["openai-codex"][0];
        assert_eq!(entry["label"], "codex-device@example.com");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["source"], "manual:device_code");
        assert_eq!(entry["refresh_token"], "codex-refresh-2");
        assert_eq!(
            entry["base_url"],
            "https://codex.example.test/backend-api/codex"
        );
        assert!(entry["last_refresh"].as_str().is_some());
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Signing in to OpenAI Codex..."));
        assert!(rendered.contains("Added openai-codex OAuth credential #1"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_add_nous_oauth_without_state_uses_native_device_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = temp_path("auth-add-nous-device");
        let home = temp.join("home");
        let shared = temp.join("shared");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (portal_base_url, requests, server) = spawn_nous_oauth_server(
            "header.eyJlbWFpbCI6Im5vdXMtZGV2aWNlQGV4YW1wbGUuY29tIn0.sig",
            "nous-refresh-2",
        );
        unsafe { std::env::set_var("HERMES_SHARED_AUTH_DIR", &shared) };
        let result = print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "nous".to_string(),
                auth_type: Some("oauth".to_string()),
                label: Some("Nous Device".to_string()),
                portal_url: Some(portal_base_url.clone()),
                inference_url: Some(format!("{portal_base_url}/requested-inference/v1")),
                client_id: Some("hermes-cli-test".to_string()),
                scope: Some("inference:mint_agent_key offline_access".to_string()),
                no_browser: true,
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
        );
        unsafe { std::env::remove_var("HERMES_SHARED_AUTH_DIR") };

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 3);
        assert!(captured[0].starts_with("POST /api/oauth/device/code "));
        assert!(captured[0].contains("client_id=hermes-cli-test"));
        assert!(captured[0].contains("scope=inference%3Amint_agent_key+offline_access"));
        assert!(captured[1].starts_with("POST /api/oauth/token "));
        assert!(
            captured[1]
                .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
        );
        assert!(captured[1].contains("device_code=nous-device-code"));
        assert!(captured[2].starts_with("POST /api/oauth/agent-key "));
        assert!(captured[2].contains("header.eyJlbWFpbCI6Im5vdXMtZGV2aWNlQGV4YW1wbGUuY29tIn0.sig"));

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let state = &persisted["providers"]["nous"];
        assert_eq!(state["label"], "Nous Device");
        assert_eq!(state["client_id"], "hermes-cli-test");
        assert_eq!(
            state["portal_base_url"],
            JsonValue::String(portal_base_url.clone())
        );
        assert_eq!(
            state["inference_base_url"],
            JsonValue::String(format!("{portal_base_url}/runtime-inference/v1"))
        );
        assert_eq!(state["refresh_token"], "nous-refresh-2");
        assert_eq!(state["agent_key"], "nous-agent-key");

        let entries = persisted["credential_pool"]["nous"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["label"], "Nous Device");
        assert_eq!(entry["source"], "device_code");
        assert_eq!(entry["auth_type"], "oauth");
        assert_eq!(entry["refresh_token"], "nous-refresh-2");
        assert_eq!(
            entry["base_url"],
            JsonValue::String(format!("{portal_base_url}/runtime-inference/v1"))
        );

        let shared_state: JsonValue =
            serde_json::from_str(&fs::read_to_string(shared.join("nous_auth.json")).unwrap())
                .unwrap();
        assert_eq!(shared_state["refresh_token"], "nous-refresh-2");
        assert_eq!(
            shared_state["inference_base_url"],
            JsonValue::String(format!("{portal_base_url}/runtime-inference/v1"))
        );
        assert!(shared_state.get("agent_key").is_none());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn auth_add_nous_oauth_imports_shared_state_before_device_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = temp_path("auth-add-nous-shared");
        let home = temp.join("home");
        let shared = temp.join("shared");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&shared).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let (portal_base_url, requests, server) = spawn_nous_shared_import_server(
            "header.eyJlbWFpbCI6Im5vdXMtaW1wb3J0QGV4YW1wbGUuY29tIn0.sig",
            "shared-refresh-2",
        );
        fs::write(
            shared.join("nous_auth.json"),
            serde_json::to_string_pretty(&json!({
                "_schema": 1,
                "access_token": "stale-shared-access",
                "refresh_token": "shared-refresh-1",
                "token_type": "Bearer",
                "scope": "inference:mint_agent_key offline_access",
                "client_id": "shared-client",
                "portal_base_url": portal_base_url,
                "inference_base_url": format!("{portal_base_url}/shared-inference/v1"),
                "obtained_at": "2026-01-01T00:00:00Z",
                "expires_at": "2026-01-01T00:00:00Z"
            }))
            .unwrap(),
        )
        .unwrap();

        unsafe { std::env::set_var("HERMES_SHARED_AUTH_DIR", &shared) };
        let result = print_auth_add(
            &context,
            &AuthAddArgs {
                provider: "nous".to_string(),
                auth_type: Some("oauth".to_string()),
                label: Some("Imported Nous".to_string()),
                timeout: Some(5.0),
                ..AuthAddArgs::default()
            },
        );
        unsafe { std::env::remove_var("HERMES_SHARED_AUTH_DIR") };

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].starts_with("POST /oauth/token "));
        assert!(captured[0].contains("grant_type=refresh_token"));
        assert!(captured[0].contains("client_id=shared-client"));
        assert!(captured[0].contains("refresh_token=shared-refresh-1"));
        assert!(captured[1].starts_with("POST /api/oauth/agent-key "));
        assert!(captured[1].contains("header.eyJlbWFpbCI6Im5vdXMtaW1wb3J0QGV4YW1wbGUuY29tIn0.sig"));

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let state = &persisted["providers"]["nous"];
        assert_eq!(state["label"], "Imported Nous");
        assert_eq!(state["client_id"], "shared-client");
        assert_eq!(state["refresh_token"], "shared-refresh-2");
        assert_eq!(state["agent_key"], "nous-agent-key-imported");
        assert_eq!(
            state["inference_base_url"],
            JsonValue::String(format!("{portal_base_url}/runtime-inference/v1"))
        );
        let entries = persisted["credential_pool"]["nous"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["source"], "device_code");
        assert_eq!(entries[0]["label"], "Imported Nous");
        assert_eq!(entries[0]["refresh_token"], "shared-refresh-2");
        let rewritten_shared: JsonValue =
            serde_json::from_str(&fs::read_to_string(shared.join("nous_auth.json")).unwrap())
                .unwrap();
        assert_eq!(rewritten_shared["refresh_token"], "shared-refresh-2");
        assert_eq!(
            rewritten_shared["inference_base_url"],
            JsonValue::String(format!("{portal_base_url}/runtime-inference/v1"))
        );
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn auth_remove_manual_entry_stays_native() {
        let home = temp_path("auth-remove-native");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "a1",
                            "label": "primary",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "sk-a"
                        },
                        {
                            "id": "b2",
                            "label": "backup",
                            "auth_type": "api_key",
                            "priority": 1,
                            "source": "manual",
                            "access_token": "sk-b"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "openrouter".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let entries = persisted["credential_pool"]["openrouter"]
            .as_array()
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["label"], "backup");
        assert_eq!(entries[0]["priority"], 0);
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_unsupported_source_uses_python_fallback() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = temp_path("auth-remove-python");
        let log_path = temp.join("auth-remove.log");
        let python = temp.join("python3");
        let home = temp_path("auth-remove-home");
        fs::create_dir_all(&temp).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "ext1",
                            "label": "external",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "external-helper",
                            "access_token": "sk-ext"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'provider=%s\\ntarget=%s\\n' \"$HERMES_AUTH_REMOVE_PROVIDER\" \"$HERMES_AUTH_REMOVE_TARGET\" > \"{}\"\nprintf '%s\\n' \"$@\" >> \"{}\"\nexit 0\n",
                log_path.display(),
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        unsafe { std::env::set_var("HERMES_AUTH_PYTHON", &python) };
        let result = print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "openrouter".to_string(),
                target: "1".to_string(),
            },
        );
        unsafe { std::env::remove_var("HERMES_AUTH_PYTHON") };

        result.unwrap();
        let logged = fs::read_to_string(log_path).unwrap();
        assert!(logged.contains("provider=openrouter"));
        assert!(logged.contains("target=1"));
        assert!(logged.contains("-c"));
        assert!(logged.contains("auth_remove_command"));
        let _ = fs::remove_dir_all(temp);
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_env_source_stays_native_and_clears_dotenv() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("auth-remove-env");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join(".env"),
            "OPENROUTER_API_KEY=sk-env\nOTHER_KEY=keep\n",
        )
        .unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "env1",
                            "label": "OPENROUTER_API_KEY",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "env:OPENROUTER_API_KEY",
                            "access_token": "sk-env"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "openrouter".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let env_text = fs::read_to_string(home.join(".env")).unwrap();
        assert!(!env_text.contains("OPENROUTER_API_KEY="));
        assert!(env_text.contains("OTHER_KEY=keep"));
        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert_eq!(
            persisted["suppressed_sources"]["openrouter"][0],
            "env:OPENROUTER_API_KEY"
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_copilot_env_stays_native_and_suppresses_all_sources() {
        let home = temp_path("auth-remove-copilot");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "copilot": [
                        {
                            "id": "cop1",
                            "label": "GH_TOKEN",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "env:GH_TOKEN",
                            "access_token": "gh-token"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "copilot".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        let suppressed = persisted["suppressed_sources"]["copilot"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(JsonValue::as_str)
            .collect::<Vec<_>>();
        assert!(suppressed.contains(&"gh_cli"));
        assert!(suppressed.contains(&"env:COPILOT_GITHUB_TOKEN"));
        assert!(suppressed.contains(&"env:GH_TOKEN"));
        assert!(suppressed.contains(&"env:GITHUB_TOKEN"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_anthropic_claude_code_stays_native() {
        let home = temp_path("auth-remove-claude");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "cc1",
                            "label": "claude_code",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "claude_code",
                            "access_token": "cc-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "anthropic".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert_eq!(
            persisted["suppressed_sources"]["anthropic"][0],
            "claude_code"
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_anthropic_hermes_pkce_stays_native() {
        let home = temp_path("auth-remove-hermes-pkce");
        fs::create_dir_all(&home).unwrap();
        fs::write(home.join(".anthropic_oauth.json"), "{\"ok\":true}\n").unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "pk1",
                            "label": "hermes_pkce",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "hermes_pkce",
                            "access_token": "pkce-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "anthropic".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        assert!(!home.join(".anthropic_oauth.json").exists());
        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert_eq!(
            persisted["suppressed_sources"]["anthropic"][0],
            "hermes_pkce"
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_custom_config_source_stays_native() {
        let home = temp_path("auth-remove-custom-config");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "custom:demo": [
                        {
                            "id": "cfg1",
                            "label": "demo",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "config:demo",
                            "access_token": "cfg-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "custom:demo".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert_eq!(
            persisted["suppressed_sources"]["custom:demo"][0],
            "config:demo"
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_nous_device_code_stays_native_and_clears_state() {
        let home = temp_path("auth-remove-nous");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "active_provider": "nous",
                "providers": {
                    "nous": {
                        "access_token": "nous-access",
                        "refresh_token": "nous-refresh"
                    }
                },
                "credential_pool": {
                    "nous": [
                        {
                            "id": "nous1",
                            "label": "Nous Main",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "device_code",
                            "access_token": "nous-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "nous".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert!(persisted["providers"].get("nous").is_none());
        assert_eq!(persisted["active_provider"], JsonValue::Null);
        assert_eq!(persisted["suppressed_sources"]["nous"][0], "device_code");
        assert!(
            persisted["credential_pool"]["nous"]
                .as_array()
                .is_some_and(|entries| entries.is_empty())
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_openai_codex_manual_device_code_stays_native() {
        let home = temp_path("auth-remove-codex");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "codex-access",
                            "refresh_token": "codex-refresh"
                        }
                    }
                },
                "credential_pool": {
                    "openai-codex": [
                        {
                            "id": "codex1",
                            "label": "Codex Main",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:device_code",
                            "access_token": "codex-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "openai-codex".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert!(persisted["providers"].get("openai-codex").is_none());
        assert_eq!(
            persisted["suppressed_sources"]["openai-codex"][0],
            "device_code"
        );
        assert_eq!(
            persisted["suppressed_sources"]["openai-codex"][1],
            "manual:device_code"
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_qwen_cli_stays_native_and_suppresses_source() {
        let home = temp_path("auth-remove-qwen");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "qwen-oauth": [
                        {
                            "id": "qwen1",
                            "label": "Qwen CLI",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "qwen-cli",
                            "access_token": "qwen-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "qwen-oauth".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert_eq!(persisted["suppressed_sources"]["qwen-oauth"][0], "qwen-cli");
        assert!(
            persisted["credential_pool"]["qwen-oauth"]
                .as_array()
                .is_some_and(|entries| entries.is_empty())
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_minimax_manual_oauth_stays_native_and_clears_state() {
        let home = temp_path("auth-remove-minimax");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "active_provider": "minimax-oauth",
                "providers": {
                    "minimax-oauth": {
                        "access_token": "minimax-access",
                        "refresh_token": "minimax-refresh"
                    }
                },
                "credential_pool": {
                    "minimax-oauth": [
                        {
                            "id": "minimax1",
                            "label": "MiniMax OAuth",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:minimax_oauth",
                            "access_token": "minimax-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "minimax-oauth".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert!(persisted["providers"].get("minimax-oauth").is_none());
        assert_eq!(persisted["active_provider"], JsonValue::Null);
        assert_eq!(
            persisted["suppressed_sources"]["minimax-oauth"][0],
            "manual:minimax_oauth"
        );
        assert!(
            persisted["credential_pool"]["minimax-oauth"]
                .as_array()
                .is_some_and(|entries| entries.is_empty())
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn auth_remove_google_gemini_cli_stays_native_and_clears_oauth_file() {
        let home = temp_path("auth-remove-google-gemini");
        fs::create_dir_all(home.join("auth")).unwrap();
        fs::write(
            home.join("auth").join("google_oauth.json"),
            json!({
                "refresh": "google-refresh",
                "access": "google-access",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "active_provider": "google-gemini-cli",
                "providers": {
                    "google-gemini-cli": {
                        "note": "placeholder"
                    }
                },
                "credential_pool": {
                    "google-gemini-cli": [
                        {
                            "id": "google1",
                            "label": "Google Gemini CLI",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:google_pkce",
                            "access_token": "google-access"
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_auth_remove(
            &context,
            &AuthRemoveArgs {
                provider: "google-gemini-cli".to_string(),
                target: "1".to_string(),
            },
        )
        .unwrap();

        let persisted: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("auth.json")).unwrap()).unwrap();
        assert!(persisted["providers"].get("google-gemini-cli").is_none());
        assert_eq!(persisted["active_provider"], JsonValue::Null);
        assert_eq!(
            persisted["suppressed_sources"]["google-gemini-cli"][0],
            "manual:google_pkce"
        );
        assert!(
            persisted["credential_pool"]["google-gemini-cli"]
                .as_array()
                .is_some_and(|entries| entries.is_empty())
        );
        assert!(!home.join("auth").join("google_oauth.json").exists());
        let _ = fs::remove_dir_all(home);
    }
}
