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
            return Err("auth requires a subcommand: status|logout|spotify".into());
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

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::HermesContext;
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
}
