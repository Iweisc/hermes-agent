use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::{Command, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::{
    AuthStatusSummary, HermesContext, LoadedConfig, OPENROUTER_BASE_URL, clear_provider_auth_state,
    get_active_auth_provider, get_auth_status_summary,
};
use serde_json::Value as JsonValue;
use serde_yaml::Value;

use crate::python_bridge::{project_root, resolve_repo_python};

const CONFIG_FALLBACK_LOGOUT_PROVIDERS: &[&str] = &[
    "nous",
    "openai-codex",
    "minimax-oauth",
    "google-gemini-cli",
    "qwen-oauth",
    "spotify",
];

const SPOTIFY_AUTH_LOGIN_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from types import SimpleNamespace\n",
    "from hermes_cli.auth import login_spotify_command\n",
    "raw_timeout = os.environ.get('HERMES_AUTH_SPOTIFY_TIMEOUT', '').strip()\n",
    "args = SimpleNamespace(\n",
    "    client_id=(os.environ.get('HERMES_AUTH_SPOTIFY_CLIENT_ID', '').strip() or None),\n",
    "    redirect_uri=(os.environ.get('HERMES_AUTH_SPOTIFY_REDIRECT_URI', '').strip() or None),\n",
    "    scope=(os.environ.get('HERMES_AUTH_SPOTIFY_SCOPE', '').strip() or None),\n",
    "    no_browser=(os.environ.get('HERMES_AUTH_SPOTIFY_NO_BROWSER', '0') == '1'),\n",
    "    timeout=(float(raw_timeout) if raw_timeout else None),\n",
    ")\n",
    "login_spotify_command(args)\n",
);

#[derive(Subcommand, Debug)]
pub enum AuthCommand {
    List { provider: Option<String> },
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
        Some(AuthCommand::List { provider }) => {
            print_auth_list(context, provider.as_deref())?;
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
            return Err("auth requires a subcommand: list|reset|status|logout|spotify".into());
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

fn print_auth_spotify(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: SpotifyAuthArgs,
) -> Result<(), Box<dyn Error>> {
    match args.spotify_action {
        SpotifyAuthAction::Login => run_python_spotify_login(&args),
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

pub(crate) fn run_default_spotify_login() -> Result<(), Box<dyn Error>> {
    run_python_spotify_login(&SpotifyAuthArgs::default())
}

fn run_python_spotify_login(args: &SpotifyAuthArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_AUTH_PYTHON"))
        .ok_or("could not find a Python interpreter for auth")?;
    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env(
            "HERMES_AUTH_SPOTIFY_CLIENT_ID",
            args.client_id.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_SPOTIFY_REDIRECT_URI",
            args.redirect_uri.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_SPOTIFY_SCOPE",
            args.scope.as_deref().unwrap_or(""),
        )
        .env(
            "HERMES_AUTH_SPOTIFY_NO_BROWSER",
            if args.no_browser { "1" } else { "0" },
        )
        .env(
            "HERMES_AUTH_SPOTIFY_TIMEOUT",
            args.timeout
                .map(|value| value.to_string())
                .unwrap_or_default(),
        )
        .arg("-c")
        .arg(SPOTIFY_AUTH_LOGIN_BOOTSTRAP);
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("auth spotify", status).into())
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
    let trimmed = provider.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.as_str() {
        "or" | "open-router" => Some("openrouter".to_string()),
        _ => Some(trimmed),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::HermesContext;
    use serde_json::json;
    use std::path::PathBuf;

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-auth-{label}-{unique}"))
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
    fn spotify_login_uses_direct_python_launcher() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = temp_path("spotify-login");
        let log_path = temp.join("spotify-auth.log");
        let python = temp.join("python3");
        fs::create_dir_all(&temp).unwrap();
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'client_id=%s\\nredirect_uri=%s\\nscope=%s\\nno_browser=%s\\ntimeout=%s\\n' \"$HERMES_AUTH_SPOTIFY_CLIENT_ID\" \"$HERMES_AUTH_SPOTIFY_REDIRECT_URI\" \"$HERMES_AUTH_SPOTIFY_SCOPE\" \"$HERMES_AUTH_SPOTIFY_NO_BROWSER\" \"$HERMES_AUTH_SPOTIFY_TIMEOUT\" > \"{}\"\nprintf '%s\\n' \"$@\" >> \"{}\"\nexit 0\n",
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
        let result = run_python_spotify_login(&SpotifyAuthArgs {
            spotify_action: SpotifyAuthAction::Login,
            client_id: Some("client-123".to_string()),
            redirect_uri: Some("http://127.0.0.1:43827/spotify/callback".to_string()),
            scope: Some("user-read-playback-state".to_string()),
            no_browser: true,
            timeout: Some(9.5),
        });
        unsafe { std::env::remove_var("HERMES_AUTH_PYTHON") };

        result.unwrap();
        let logged = fs::read_to_string(log_path).unwrap();
        assert!(logged.contains("client_id=client-123"));
        assert!(logged.contains("redirect_uri=http://127.0.0.1:43827/spotify/callback"));
        assert!(logged.contains("scope=user-read-playback-state"));
        assert!(logged.contains("no_browser=1"));
        assert!(logged.contains("timeout=9.5"));
        assert!(logged.contains("-c"));
        assert!(logged.contains("login_spotify_command"));
        let _ = fs::remove_dir_all(temp);
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
}
