use std::error::Error;

use clap::Subcommand;
use hermes_core::{
    AuthStatusSummary, HermesContext, LoadedConfig, ModelOverrides, get_active_auth_provider,
    get_auth_status_summary, get_provider_profile, infer_api_mode_from_base_url,
    list_provider_profiles, normalize_model_for_provider, normalize_provider_alias,
    resolve_provider_api_mode,
};
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};

#[derive(Subcommand, Debug)]
pub enum ModelCommand {
    Show,
    Set {
        model: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long = "base-url")]
        base_url: Option<String>,
        #[arg(long = "api-mode")]
        api_mode: Option<String>,
        #[arg(long = "clear-base-url")]
        clear_base_url: bool,
        #[arg(long = "clear-api-mode")]
        clear_api_mode: bool,
    },
    Providers {
        #[arg(long = "configured-only")]
        configured_only: bool,
    },
}

pub fn print_model(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<ModelCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(ModelCommand::Show) {
        ModelCommand::Show => print_model_show(context, loaded)?,
        ModelCommand::Set {
            model,
            provider,
            base_url,
            api_mode,
            clear_base_url,
            clear_api_mode,
        } => set_model(
            context,
            loaded,
            SetModelArgs {
                model,
                provider,
                base_url,
                api_mode,
                clear_base_url,
                clear_api_mode,
            },
        )?,
        ModelCommand::Providers { configured_only } => {
            print_providers(context, loaded, configured_only)?
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SetModelArgs {
    model: String,
    provider: Option<String>,
    base_url: Option<String>,
    api_mode: Option<String>,
    clear_base_url: bool,
    clear_api_mode: bool,
}

#[derive(Debug)]
struct ProviderRow {
    name: String,
    aliases: String,
    auth_type: String,
    base_url: String,
    configured: bool,
    logged_in: bool,
    source: Option<String>,
    detail: Option<String>,
    current: bool,
    active_auth: bool,
}

fn print_model_show(context: &HermesContext, loaded: &LoadedConfig) -> Result<(), Box<dyn Error>> {
    println!(
        "configured_model={}",
        loaded
            .configured_model_name()
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "configured_provider={}",
        loaded
            .configured_model_provider()
            .unwrap_or_else(|| "auto".to_string())
    );
    println!(
        "configured_base_url={}",
        loaded
            .configured_model_base_url()
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "configured_api_mode={}",
        loaded
            .configured_model_api_mode()
            .unwrap_or_else(|| "(not set)".to_string())
    );
    if let Some(active_auth) = get_active_auth_provider(context.hermes_home().as_path())? {
        println!("active_auth_provider={active_auth}");
    }

    match context.resolve_model_runtime(loaded, &ModelOverrides::default()) {
        Ok(runtime) => {
            println!("resolved_model={}", runtime.model);
            println!("resolved_provider={}", runtime.provider);
            println!("resolved_base_url={}", runtime.base_url);
            println!("resolved_api_mode={}", runtime.api_mode);
            println!("resolved_auth_type={}", runtime.auth_type);
            println!("resolved_api_key_present={}", !runtime.api_key.is_empty());
            println!("resolved_default_headers={}", runtime.default_headers.len());
        }
        Err(error) => {
            println!("runtime_error={error}");
        }
    }
    Ok(())
}

fn set_model(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: SetModelArgs,
) -> Result<(), Box<dyn Error>> {
    let model_input = sanitize_model_input(&args.model)?;
    let explicit_provider = args
        .provider
        .as_deref()
        .map(validate_provider_input)
        .transpose()?;
    let explicit_base_url = args
        .base_url
        .as_deref()
        .map(validate_base_url_input)
        .transpose()?;
    let explicit_api_mode = args
        .api_mode
        .as_deref()
        .map(validate_api_mode_input)
        .transpose()?;

    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let mut model_cfg = take_model_mapping(&mut root);
    let previous_provider = mapping_string(&model_cfg, "provider");
    let effective_provider = explicit_provider
        .clone()
        .or_else(|| loaded.configured_model_provider());
    let normalized_model = effective_provider
        .as_deref()
        .filter(|provider| *provider != "auto")
        .map(|provider| normalize_model_for_provider(&model_input, provider))
        .unwrap_or_else(|| model_input.clone());

    insert_mapping_string(&mut model_cfg, "default", &normalized_model);

    if let Some(provider) = explicit_provider.as_deref() {
        insert_mapping_string(&mut model_cfg, "provider", provider);
        if args.clear_base_url {
            remove_mapping_key(&mut model_cfg, "base_url");
        } else if let Some(base_url) = explicit_base_url.as_deref() {
            insert_mapping_string(&mut model_cfg, "base_url", base_url);
        } else if let Some(profile) = get_provider_profile(provider) {
            if !profile.base_url.trim().is_empty() {
                insert_mapping_string(&mut model_cfg, "base_url", profile.base_url);
            } else if previous_provider.as_deref() != Some(provider) {
                remove_mapping_key(&mut model_cfg, "base_url");
            }
        }
    } else if args.clear_base_url {
        remove_mapping_key(&mut model_cfg, "base_url");
    } else if let Some(base_url) = explicit_base_url.as_deref() {
        insert_mapping_string(&mut model_cfg, "base_url", base_url);
    }

    let api_mode_value = if args.clear_api_mode {
        None
    } else if let Some(api_mode) = explicit_api_mode.as_deref() {
        Some(api_mode.to_string())
    } else {
        let current_provider = mapping_string(&model_cfg, "provider");
        let current_base_url = mapping_string(&model_cfg, "base_url");
        current_provider
            .as_deref()
            .filter(|provider| !provider.is_empty() && *provider != "auto")
            .and_then(|provider| {
                resolve_provider_api_mode(provider, &normalized_model)
                    .map(str::to_string)
                    .or_else(|| {
                        current_base_url
                            .as_deref()
                            .and_then(infer_api_mode_from_base_url)
                            .map(str::to_string)
                    })
                    .or_else(|| {
                        get_provider_profile(provider).map(|profile| profile.api_mode.to_string())
                    })
            })
            .or_else(|| {
                current_base_url
                    .as_deref()
                    .and_then(infer_api_mode_from_base_url)
                    .map(str::to_string)
            })
    };
    if args.clear_api_mode {
        remove_mapping_key(&mut model_cfg, "api_mode");
    } else if let Some(api_mode) = api_mode_value {
        insert_mapping_string(&mut model_cfg, "api_mode", &api_mode);
    } else if explicit_provider.is_some() {
        remove_mapping_key(&mut model_cfg, "api_mode");
    }

    root.insert(yaml_key("model"), Value::Mapping(model_cfg));
    write_yaml_mapping(&context.config_path(), &root)?;

    let reloaded = context.load_config_document()?;
    println!("updated=true");
    println!("config_path={}", context.config_path().display());
    print_model_show(context, &reloaded)
}

fn print_providers(
    context: &HermesContext,
    loaded: &LoadedConfig,
    configured_only: bool,
) -> Result<(), Box<dyn Error>> {
    let current_provider = loaded.configured_model_provider();
    let active_auth = get_active_auth_provider(context.hermes_home().as_path())?;
    let mut rows = list_provider_profiles()
        .iter()
        .map(|profile| {
            let status = get_auth_status_summary(context.hermes_home().as_path(), profile.name)?;
            Ok(provider_row(
                profile.name,
                profile.aliases,
                profile.auth_type,
                profile.base_url,
                status,
                current_provider.as_deref() == Some(profile.name),
                active_auth.as_deref() == Some(profile.name),
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    rows.sort_by(|left, right| {
        right
            .current
            .cmp(&left.current)
            .then_with(|| right.active_auth.cmp(&left.active_auth))
            .then_with(|| right.logged_in.cmp(&left.logged_in))
            .then_with(|| right.configured.cmp(&left.configured))
            .then_with(|| left.name.cmp(&right.name))
    });

    for row in rows {
        if configured_only && !row.current && !row.active_auth && !row.configured && !row.logged_in
        {
            continue;
        }
        let marker = provider_marker(&row);
        println!(
            "{marker} {name:<18} auth={auth:<18} configured={configured:<5} logged_in={logged_in:<5}",
            name = row.name,
            auth = row.auth_type,
            configured = row.configured,
            logged_in = row.logged_in,
        );
        if !row.aliases.is_empty() {
            println!("  aliases={}", row.aliases);
        }
        if !row.base_url.is_empty() {
            println!("  base_url={}", row.base_url);
        }
        if let Some(source) = row.source.as_deref() {
            println!("  source={source}");
        }
        if let Some(detail) = row.detail.as_deref() {
            println!("  detail={detail}");
        }
    }
    Ok(())
}

fn provider_row(
    name: &str,
    aliases: &[&str],
    auth_type: &str,
    base_url: &str,
    status: AuthStatusSummary,
    current: bool,
    active_auth: bool,
) -> ProviderRow {
    ProviderRow {
        name: name.to_string(),
        aliases: aliases.join(","),
        auth_type: auth_type.to_string(),
        base_url: base_url.to_string(),
        configured: status.configured,
        logged_in: status.logged_in,
        source: status.source,
        detail: status.detail,
        current,
        active_auth,
    }
}

fn provider_marker(row: &ProviderRow) -> &'static str {
    match (row.current, row.active_auth) {
        (true, true) => "*@",
        (true, false) => "* ",
        (false, true) => " @",
        (false, false) => "  ",
    }
}

fn sanitize_model_input(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("model cannot be empty".into());
    }
    if trimmed.contains('\n') || trimmed.contains('\r') {
        return Err("model cannot contain newlines".into());
    }
    Ok(trimmed.to_string())
}

fn validate_provider_input(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("provider cannot be empty".into());
    }
    if trimmed.eq_ignore_ascii_case("auto") {
        return Ok("auto".to_string());
    }
    let canonical = normalize_provider_alias(trimmed);
    if get_provider_profile(&canonical).is_none() {
        return Err(format!("provider '{trimmed}' is not recognized").into());
    }
    Ok(canonical)
}

fn validate_base_url_input(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("base URL cannot be empty".into());
    }
    if trimmed.contains(char::is_whitespace) || trimmed.contains('\n') || trimmed.contains('\r') {
        return Err("base URL cannot contain whitespace".into());
    }
    if !trimmed.contains("://") {
        return Err("base URL must include a scheme like https://".into());
    }
    Ok(trimmed.trim_end_matches('/').to_string())
}

fn validate_api_mode_input(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim().to_ascii_lowercase();
    if !matches!(
        trimmed.as_str(),
        "chat_completions" | "anthropic_messages" | "codex_responses" | "bedrock_converse"
    ) {
        return Err(format!("unsupported api_mode '{value}'").into());
    }
    Ok(trimmed)
}

fn take_model_mapping(root: &mut Mapping) -> Mapping {
    match root.remove(&yaml_key("model")) {
        Some(Value::Mapping(mapping)) => mapping,
        Some(Value::String(model)) => {
            let mut mapping = Mapping::new();
            let trimmed = model.trim();
            if !trimmed.is_empty() {
                mapping.insert(yaml_key("default"), Value::String(trimmed.to_string()));
            }
            mapping
        }
        _ => Mapping::new(),
    }
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(yaml_key(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn insert_mapping_string(mapping: &mut Mapping, key: &str, value: &str) {
    mapping.insert(yaml_key(key), Value::String(value.to_string()));
}

fn remove_mapping_key(mapping: &mut Mapping, key: &str) {
    let key = yaml_key(key);
    mapping.remove(&key);
}

fn yaml_key(value: &str) -> Value {
    Value::String(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-model-{label}-{unique}"))
    }

    #[test]
    fn set_model_persists_provider_defaults() {
        let home = temp_path("set");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_model(
            &context,
            &context.load_config_document().unwrap(),
            SetModelArgs {
                model: "gpt-5".to_string(),
                provider: Some("openai".to_string()),
                base_url: None,
                api_mode: None,
                clear_base_url: false,
                clear_api_mode: false,
            },
        )
        .unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("default: gpt-5"));
        assert!(written.contains("provider: openai"));
        assert!(written.contains("base_url: https://api.openai.com/v1"));
        assert!(written.contains("api_mode: chat_completions"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn set_model_normalizes_provider_aliases_and_api_modes() {
        let home = temp_path("copilot");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_model(
            &context,
            &context.load_config_document().unwrap(),
            SetModelArgs {
                model: "anthropic/claude-sonnet-4.6".to_string(),
                provider: Some("github".to_string()),
                base_url: None,
                api_mode: None,
                clear_base_url: false,
                clear_api_mode: false,
            },
        )
        .unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("provider: copilot"));
        assert!(written.contains("default: claude-sonnet-4.6"));
        assert!(written.contains("api_mode: anthropic_messages"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn set_model_can_clear_stale_custom_base_url() {
        let home = temp_path("clear");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "model:\n  default: qwen3\n  provider: custom\n  base_url: https://localhost:11434/v1\n  api_mode: chat_completions\n",
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_model(
            &context,
            &context.load_config_document().unwrap(),
            SetModelArgs {
                model: "claude-sonnet-4.6".to_string(),
                provider: Some("anthropic".to_string()),
                base_url: None,
                api_mode: None,
                clear_base_url: false,
                clear_api_mode: true,
            },
        )
        .unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("provider: anthropic"));
        assert!(written.contains("base_url: https://api.anthropic.com"));
        assert!(!written.contains("https://localhost:11434/v1"));
        assert!(!written.contains("api_mode:"));
        let _ = fs::remove_dir_all(home);
    }
}
