use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::{HermesContext, LoadedConfig};
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::python_bridge::{project_root, resolve_repo_python};

#[derive(Subcommand, Debug)]
pub enum MemoryCommand {
    Setup,
    Status,
    Off,
    Reset(ResetArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ResetArgs {
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[arg(long, value_enum, default_value_t = ResetTarget::All)]
    pub target: ResetTarget,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ResetTarget {
    #[value(name = "all")]
    All,
    #[value(name = "memory")]
    Memory,
    #[value(name = "user")]
    User,
}

#[derive(Debug, Clone)]
struct ProviderInfo {
    name: String,
    description: String,
}

#[derive(Debug, Clone)]
struct MemoryFile {
    name: &'static str,
    description: &'static str,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SetupMode {
    NativeGeneric,
    NativeHindsight,
    PythonHook,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FieldKind {
    String,
    Bool,
    Integer,
    Float,
}

#[derive(Debug, Clone)]
struct SetupField {
    key: &'static str,
    description: &'static str,
    kind: FieldKind,
    default: Option<String>,
    secret: bool,
    env_var: Option<&'static str>,
    url: Option<&'static str>,
    required: bool,
}

#[derive(Debug, Clone)]
struct SetupProvider {
    name: String,
    description: String,
    hint: String,
    mode: SetupMode,
    fields: Vec<SetupField>,
}

#[derive(Debug, Clone, PartialEq)]
enum SetupValue {
    String(String),
    Bool(bool),
    Integer(i64),
    Float(f64),
}

impl SetupField {
    fn default_value(&self) -> Option<SetupValue> {
        self.default
            .as_deref()
            .and_then(|raw| parse_setup_value(self.kind, raw).ok())
    }
}

impl SetupValue {
    fn display(&self) -> String {
        match self {
            Self::String(value) => value.clone(),
            Self::Bool(value) => value.to_string(),
            Self::Integer(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
        }
    }

    fn env_string(&self) -> String {
        self.display()
    }

    fn to_yaml(&self) -> Value {
        match self {
            Self::String(value) => Value::String(value.clone()),
            Self::Bool(value) => Value::Bool(*value),
            Self::Integer(value) => {
                serde_yaml::to_value(*value).unwrap_or_else(|_| Value::String(value.to_string()))
            }
            Self::Float(value) => {
                serde_yaml::to_value(*value).unwrap_or_else(|_| Value::String(value.to_string()))
            }
        }
    }

    fn to_json(&self) -> JsonValue {
        match self {
            Self::String(value) => JsonValue::String(value.clone()),
            Self::Bool(value) => JsonValue::Bool(*value),
            Self::Integer(value) => JsonValue::from(*value),
            Self::Float(value) => JsonValue::from(*value),
        }
    }
}

pub fn print_memory(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<MemoryCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(MemoryCommand::Status) {
        MemoryCommand::Setup => print_memory_setup(context, loaded),
        MemoryCommand::Status => {
            println!("{}", render_status(context, loaded));
            Ok(())
        }
        MemoryCommand::Off => {
            disable_external_provider(context)?;
            println!();
            println!("  ✓ Memory provider: built-in only");
            println!("  Saved to config.yaml");
            println!();
            Ok(())
        }
        MemoryCommand::Reset(args) => reset_memory_files(context, args),
    }
}

fn print_memory_setup(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<(), Box<dyn Error>> {
    let providers = discover_memory_setup_providers(context);
    if providers.is_empty() {
        println!("\n  No memory provider plugins detected.");
        println!(
            "  Install a plugin to {}/plugins/ and try again.\n",
            context.display_hermes_home()
        );
        return Ok(());
    }

    let Some(selection) =
        prompt_provider_selection(&providers, loaded.config.memory.provider.trim())?
    else {
        println!("  Cancelled.\n");
        return Ok(());
    };

    match selection {
        0 => {
            disable_external_provider(context)?;
            println!("\n  ✓ Memory provider: built-in only");
            println!("  Saved to config.yaml\n");
            Ok(())
        }
        index => {
            let provider = providers
                .get(index.saturating_sub(1))
                .ok_or("invalid memory provider selection")?;
            match provider.mode {
                SetupMode::NativeGeneric => run_native_provider_setup(context, provider),
                SetupMode::NativeHindsight => run_hindsight_provider_setup(context, provider),
                SetupMode::PythonHook => {
                    println!(
                        "\n  Handing off to the provider-specific setup for {}.\n",
                        provider.name
                    );
                    bridge_memory_setup_provider(&provider.name)
                }
            }
        }
    }
}

fn discover_memory_setup_providers(context: &HermesContext) -> Vec<SetupProvider> {
    discover_memory_providers(context)
        .into_iter()
        .map(|provider| setup_provider_spec(context, provider))
        .collect()
}

fn setup_provider_spec(context: &HermesContext, provider: ProviderInfo) -> SetupProvider {
    let display_home = context.display_hermes_home();
    let (mode, fields) = match provider.name.as_str() {
        "byterover" => (
            SetupMode::NativeGeneric,
            vec![SetupField {
                key: "api_key",
                description: "ByteRover API key (optional, for cloud sync)",
                kind: FieldKind::String,
                default: None,
                secret: true,
                env_var: Some("BRV_API_KEY"),
                url: Some("https://app.byterover.dev"),
                required: false,
            }],
        ),
        "holographic" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: "db_path",
                    description: "SQLite database path",
                    kind: FieldKind::String,
                    default: Some(format!("{display_home}/memory_store.db")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "auto_extract",
                    description: "Auto-extract facts at session end",
                    kind: FieldKind::Bool,
                    default: Some(String::from("false")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "default_trust",
                    description: "Default trust score for new facts",
                    kind: FieldKind::Float,
                    default: Some(String::from("0.5")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "hrr_dim",
                    description: "HRR vector dimensions",
                    kind: FieldKind::Integer,
                    default: Some(String::from("1024")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
            ],
        ),
        "mem0" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: "api_key",
                    description: "Mem0 Platform API key",
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some("MEM0_API_KEY"),
                    url: Some("https://app.mem0.ai"),
                    required: true,
                },
                SetupField {
                    key: "user_id",
                    description: "User identifier",
                    kind: FieldKind::String,
                    default: Some(String::from("hermes-user")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "agent_id",
                    description: "Agent identifier",
                    kind: FieldKind::String,
                    default: Some(String::from("hermes")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "rerank",
                    description: "Enable reranking for recall",
                    kind: FieldKind::Bool,
                    default: Some(String::from("true")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                },
            ],
        ),
        "openviking" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: "endpoint",
                    description: "OpenViking server URL",
                    kind: FieldKind::String,
                    default: Some(String::from("https://app.openviking.ai")),
                    secret: false,
                    env_var: Some("OPENVIKING_ENDPOINT"),
                    url: None,
                    required: true,
                },
                SetupField {
                    key: "api_key",
                    description: "OpenViking API key (leave blank for local dev mode)",
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some("OPENVIKING_API_KEY"),
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "account",
                    description: "OpenViking tenant account ID",
                    kind: FieldKind::String,
                    default: Some(String::from("default")),
                    secret: false,
                    env_var: Some("OPENVIKING_ACCOUNT"),
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "user",
                    description: "OpenViking user ID within the account",
                    kind: FieldKind::String,
                    default: Some(String::from("default")),
                    secret: false,
                    env_var: Some("OPENVIKING_USER"),
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "agent",
                    description: "OpenViking agent ID within the account",
                    kind: FieldKind::String,
                    default: Some(String::from("hermes")),
                    secret: false,
                    env_var: Some("OPENVIKING_AGENT"),
                    url: None,
                    required: false,
                },
            ],
        ),
        "retaindb" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: "api_key",
                    description: "RetainDB API key",
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some("RETAINDB_API_KEY"),
                    url: Some("https://retaindb.com"),
                    required: true,
                },
                SetupField {
                    key: "base_url",
                    description: "API endpoint",
                    kind: FieldKind::String,
                    default: Some(String::from("https://api.retaindb.com")),
                    secret: false,
                    env_var: Some("RETAINDB_BASE_URL"),
                    url: None,
                    required: false,
                },
                SetupField {
                    key: "project",
                    description: "Project identifier",
                    kind: FieldKind::String,
                    default: Some(String::new()),
                    secret: false,
                    env_var: Some("RETAINDB_PROJECT"),
                    url: None,
                    required: false,
                },
            ],
        ),
        "supermemory" => (
            SetupMode::NativeGeneric,
            vec![SetupField {
                key: "api_key",
                description: "Supermemory API key",
                kind: FieldKind::String,
                default: None,
                secret: true,
                env_var: Some("SUPERMEMORY_API_KEY"),
                url: Some("https://supermemory.ai"),
                required: true,
            }],
        ),
        "hindsight" => (SetupMode::NativeHindsight, Vec::new()),
        "honcho" => (SetupMode::PythonHook, Vec::new()),
        _ => (SetupMode::PythonHook, Vec::new()),
    };

    let hint = if mode == SetupMode::PythonHook {
        String::from("custom setup")
    } else {
        render_setup_hint(&fields)
    };

    SetupProvider {
        name: provider.name,
        description: provider.description,
        hint,
        mode,
        fields,
    }
}

fn render_setup_hint(fields: &[SetupField]) -> String {
    let has_secret = fields.iter().any(|field| field.secret);
    let has_non_secret = fields.iter().any(|field| !field.secret);
    if fields.is_empty() {
        String::from("no setup needed")
    } else if has_secret && has_non_secret {
        String::from("API key / local")
    } else if has_secret {
        String::from("requires API key")
    } else {
        String::from("local")
    }
}

fn prompt_provider_selection(
    providers: &[SetupProvider],
    current_provider: &str,
) -> Result<Option<usize>, Box<dyn Error>> {
    let default_index = providers
        .iter()
        .position(|provider| provider.name == current_provider)
        .map(|index| index + 1)
        .unwrap_or(0);

    loop {
        println!("\nMemory provider setup");
        println!("────────────────────────────────────────");
        println!("  0) Built-in only — MEMORY.md / USER.md (default)");
        for (index, provider) in providers.iter().enumerate() {
            let active = if provider.name == current_provider {
                " ← active"
            } else {
                ""
            };
            let summary = if provider.description.trim().is_empty() {
                provider.hint.clone()
            } else {
                format!("{} — {}", provider.hint, provider.description)
            };
            println!("  {}) {} — {}{}", index + 1, provider.name, summary, active);
        }

        let prompt = format!("Select provider [{default_index}]: ");
        let Some(input) = read_prompt_line(&prompt)? else {
            return Ok(None);
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(Some(default_index));
        }
        if matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "built-in" | "builtin" | "built_in" | "none" | "default"
        ) {
            return Ok(Some(0));
        }
        if let Ok(index) = trimmed.parse::<usize>() {
            if index <= providers.len() {
                return Ok(Some(index));
            }
        }
        if let Some((index, _)) = providers
            .iter()
            .enumerate()
            .find(|(_, provider)| provider.name.eq_ignore_ascii_case(trimmed))
        {
            return Ok(Some(index + 1));
        }
        println!("  Invalid selection. Enter a number or provider name.");
    }
}

fn run_native_provider_setup(
    context: &HermesContext,
    provider: &SetupProvider,
) -> Result<(), Box<dyn Error>> {
    let existing = load_existing_provider_values(context, &provider.name)?;
    let mut provider_values = BTreeMap::new();
    let mut env_updates = BTreeMap::new();

    if !provider.fields.is_empty() {
        println!("\n  Configuring {}:\n", provider.name);
    }

    for field in &provider.fields {
        let existing_value = existing
            .get(field.key)
            .cloned()
            .or_else(|| load_existing_env_value(field));

        let Some(value) = prompt_setup_field(field, existing_value.clone())? else {
            println!("  Cancelled.\n");
            return Ok(());
        };

        if field.secret {
            let rendered = value.env_string();
            if !rendered.trim().is_empty() {
                if let Some(env_var) = field.env_var {
                    env_updates.insert(env_var.to_string(), rendered);
                }
            }
            continue;
        }

        provider_values.insert(field.key.to_string(), value.clone());
        let rendered = value.env_string();
        if let Some(env_var) = field.env_var {
            if !rendered.trim().is_empty() {
                env_updates.insert(env_var.to_string(), rendered);
            }
        }
    }

    save_provider_activation(context, &provider.name, &provider_values)?;
    persist_native_provider_state(context, &provider.name, &provider_values)?;
    for (key, value) in &env_updates {
        save_env_value(context.env_path(), key, value)?;
    }

    println!("\n  Memory provider: {}", provider.name);
    println!("  Activation saved to config.yaml");
    if !provider_values.is_empty() {
        println!("  Provider config saved");
    }
    if !env_updates.is_empty() {
        println!("  API keys saved to .env");
    }
    println!("\n  Start a new session to activate.\n");
    Ok(())
}

fn run_hindsight_provider_setup(
    context: &HermesContext,
    provider: &SetupProvider,
) -> Result<(), Box<dyn Error>> {
    let existing_values = load_existing_provider_values(context, &provider.name)?;
    let existing_env = load_simple_env(context.env_path());

    println!("\n  Configuring {}:\n", provider.name);

    let mode = prompt_choice(
        "  Select mode",
        &[
            ("cloud", "Hindsight Cloud API"),
            ("local_embedded", "Run Hindsight locally"),
            ("local_external", "Existing Hindsight instance"),
        ],
        existing_string_value(&existing_values, "mode")
            .as_deref()
            .unwrap_or("cloud"),
    )?;

    ensure_hindsight_dependencies(&mode)?;

    let mut provider_values = BTreeMap::new();
    let mut env_updates = BTreeMap::new();
    provider_values.insert(String::from("mode"), SetupValue::String(mode.clone()));

    match mode.as_str() {
        "cloud" => {
            let api_url = prompt_string_value(
                "  API URL",
                existing_string_value(&existing_values, "api_url")
                    .as_deref()
                    .unwrap_or("https://api.hindsight.vectorize.io"),
            )?;
            provider_values.insert(String::from("api_url"), SetupValue::String(api_url));

            let api_key = prompt_secret_with_existing(
                "  API key",
                existing_env.get("HINDSIGHT_API_KEY").map(String::as_str),
                false,
            )?;
            if !api_key.trim().is_empty() {
                env_updates.insert(String::from("HINDSIGHT_API_KEY"), api_key);
            }
        }
        "local_external" => {
            let api_url = prompt_string_value(
                "  Hindsight API URL",
                existing_string_value(&existing_values, "api_url")
                    .as_deref()
                    .unwrap_or("http://localhost:8888"),
            )?;
            provider_values.insert(String::from("api_url"), SetupValue::String(api_url));

            let api_key = prompt_secret_with_existing(
                "  API key (optional)",
                existing_env.get("HINDSIGHT_API_KEY").map(String::as_str),
                true,
            )?;
            if !api_key.trim().is_empty() {
                env_updates.insert(String::from("HINDSIGHT_API_KEY"), api_key);
            }
        }
        "local_embedded" => {
            let llm_provider = prompt_choice(
                "  Select LLM provider",
                &[
                    ("openai", "default model: gpt-4o-mini"),
                    ("anthropic", "default model: claude-haiku-4-5"),
                    ("gemini", "default model: gemini-2.5-flash"),
                    ("groq", "default model: openai/gpt-oss-120b"),
                    ("openrouter", "default model: qwen/qwen3.5-9b"),
                    ("minimax", "default model: MiniMax-M2.7"),
                    ("ollama", "default model: gemma3:12b"),
                    ("lmstudio", "default model: local-model"),
                    ("openai_compatible", "custom OpenAI-compatible endpoint"),
                ],
                existing_string_value(&existing_values, "llm_provider")
                    .as_deref()
                    .unwrap_or("openai"),
            )?;
            provider_values.insert(
                String::from("llm_provider"),
                SetupValue::String(llm_provider.clone()),
            );

            if llm_provider == "openai_compatible" {
                let base_url = prompt_string_value(
                    "  LLM endpoint URL",
                    existing_string_value(&existing_values, "llm_base_url")
                        .as_deref()
                        .unwrap_or("http://127.0.0.1:8080/v1"),
                )?;
                provider_values.insert(String::from("llm_base_url"), SetupValue::String(base_url));
            } else if llm_provider == "openrouter" {
                provider_values.insert(
                    String::from("llm_base_url"),
                    SetupValue::String(String::from("https://openrouter.ai/api/v1")),
                );
            }

            let default_model = hindsight_default_model(&llm_provider);
            let llm_model = prompt_string_value(
                "  LLM model",
                existing_string_value(&existing_values, "llm_model")
                    .as_deref()
                    .unwrap_or(default_model),
            )?;
            provider_values.insert(String::from("llm_model"), SetupValue::String(llm_model));

            let llm_api_key = prompt_secret_with_existing(
                "  LLM API key",
                existing_env
                    .get("HINDSIGHT_LLM_API_KEY")
                    .map(String::as_str),
                false,
            )?;
            if !llm_api_key.trim().is_empty() {
                env_updates.insert(String::from("HINDSIGHT_LLM_API_KEY"), llm_api_key);
            } else if let Some(existing) = existing_env.get("HINDSIGHT_LLM_API_KEY") {
                env_updates.insert(String::from("HINDSIGHT_LLM_API_KEY"), existing.clone());
            }
        }
        _ => return Err(format!("unsupported hindsight mode: {mode}").into()),
    }

    let bank_id = prompt_string_value(
        "  Memory bank name",
        existing_string_value(&existing_values, "bank_id")
            .as_deref()
            .unwrap_or("hermes"),
    )?;
    provider_values.insert(String::from("bank_id"), SetupValue::String(bank_id));

    let recall_budget = prompt_choice(
        "  Recall budget",
        &[
            ("low", "lightweight"),
            ("mid", "balanced"),
            ("high", "thorough"),
        ],
        existing_string_value(&existing_values, "recall_budget")
            .as_deref()
            .unwrap_or("mid"),
    )?;
    provider_values.insert(
        String::from("recall_budget"),
        SetupValue::String(recall_budget),
    );

    let timeout = prompt_integer_value(
        "  Timeout seconds",
        existing_integer_value(&existing_values, "timeout").unwrap_or(120),
    )?;
    provider_values.insert(String::from("timeout"), SetupValue::Integer(timeout));
    env_updates.insert(String::from("HINDSIGHT_TIMEOUT"), timeout.to_string());

    if mode == "local_embedded" {
        let idle_timeout = prompt_integer_value(
            "  Idle timeout seconds",
            existing_integer_value(&existing_values, "idle_timeout").unwrap_or(300),
        )?;
        provider_values.insert(
            String::from("idle_timeout"),
            SetupValue::Integer(idle_timeout),
        );
        env_updates.insert(
            String::from("HINDSIGHT_IDLE_TIMEOUT"),
            idle_timeout.to_string(),
        );
    }

    save_provider_activation(context, &provider.name, &provider_values)?;
    persist_native_provider_state(context, &provider.name, &provider_values)?;
    for (key, value) in &env_updates {
        save_env_value(context.env_path(), key, value)?;
    }

    if mode == "local_embedded" {
        materialize_hindsight_embedded_profile_env(context, &provider_values, &env_updates)?;
    }

    println!("\n  Memory provider: {}", provider.name);
    println!("  Activation saved to config.yaml");
    println!("  Provider config saved");
    if !env_updates.is_empty() {
        println!("  API keys saved to .env");
    }
    println!("\n  Start a new session to activate.\n");
    Ok(())
}

fn prompt_setup_field(
    field: &SetupField,
    existing_value: Option<SetupValue>,
) -> Result<Option<SetupValue>, Box<dyn Error>> {
    loop {
        let prompt = field_prompt_label(field, existing_value.as_ref());
        let input = if field.secret {
            read_secret_line(&prompt)?
        } else {
            read_prompt_line(&prompt)?
        };
        let Some(raw) = input else {
            return Ok(None);
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if let Some(value) = existing_value.clone().or_else(|| field.default_value()) {
                return Ok(Some(value));
            }
            if field.required {
                println!("  {} is required.", field.description);
                continue;
            }
            return Ok(Some(SetupValue::String(String::new())));
        }
        match parse_setup_value(field.kind, trimmed) {
            Ok(value) => return Ok(Some(value)),
            Err(error) => {
                println!("  {error}");
            }
        }
    }
}

fn prompt_choice(
    label: &str,
    options: &[(&str, &str)],
    current: &str,
) -> Result<String, Box<dyn Error>> {
    println!("{label}:");
    let default_index = options
        .iter()
        .position(|(value, _)| value.eq_ignore_ascii_case(current))
        .unwrap_or(0);
    for (index, (value, description)) in options.iter().enumerate() {
        let active = if index == default_index {
            " ← current"
        } else {
            ""
        };
        println!("    {}) {} — {}{}", index + 1, value, description, active);
    }

    loop {
        let prompt = format!("  Select [{}]: ", default_index + 1);
        let Some(input) = read_prompt_line(&prompt)? else {
            return Err("setup cancelled".into());
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(options[default_index].0.to_string());
        }
        if let Ok(index) = trimmed.parse::<usize>()
            && index >= 1
            && index <= options.len()
        {
            return Ok(options[index - 1].0.to_string());
        }
        if let Some((value, _)) = options
            .iter()
            .find(|(value, _)| value.eq_ignore_ascii_case(trimmed))
        {
            return Ok((*value).to_string());
        }
        println!("  Invalid selection.");
    }
}

fn prompt_string_value(label: &str, current: &str) -> Result<String, Box<dyn Error>> {
    let prompt = format!("  {} [{}]: ", label, current);
    let Some(input) = read_prompt_line(&prompt)? else {
        return Err("setup cancelled".into());
    };
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(current.to_string());
    }
    Ok(trimmed.to_string())
}

fn prompt_integer_value(label: &str, current: i64) -> Result<i64, Box<dyn Error>> {
    loop {
        let prompt = format!("  {} [{}]: ", label, current);
        let Some(input) = read_prompt_line(&prompt)? else {
            return Err("setup cancelled".into());
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(current);
        }
        match trimmed.parse::<i64>() {
            Ok(value) => return Ok(value),
            Err(_) => println!("  expected an integer"),
        }
    }
}

fn prompt_secret_with_existing(
    label: &str,
    existing: Option<&str>,
    allow_blank: bool,
) -> Result<String, Box<dyn Error>> {
    let prompt = if let Some(existing) = existing.filter(|value| !value.trim().is_empty()) {
        format!(
            "  {} (current: {}, blank to keep): ",
            label,
            mask_secret(existing)
        )
    } else {
        format!("  {}: ", label)
    };
    let Some(input) = read_secret_line(&prompt)? else {
        return Err("setup cancelled".into());
    };
    let trimmed = input.trim();
    if trimmed.is_empty() {
        if allow_blank {
            return Ok(String::new());
        }
        return Ok(existing.unwrap_or_default().to_string());
    }
    Ok(trimmed.to_string())
}

fn field_prompt_label(field: &SetupField, existing_value: Option<&SetupValue>) -> String {
    if field.secret {
        if let Some(value) = existing_value {
            return format!(
                "  {} (current: {}, blank to keep): ",
                field.description,
                mask_secret(&value.env_string())
            );
        }
        if let Some(url) = field.url {
            println!("  Get yours at {url}");
        }
        return format!("  {}: ", field.description);
    }

    if let Some(value) = existing_value {
        return format!("  {} [{}]: ", field.description, value.display());
    }
    if let Some(value) = field.default_value() {
        return format!("  {} [{}]: ", field.description, value.display());
    }
    format!("  {}: ", field.description)
}

fn read_prompt_line(prompt: &str) -> Result<Option<String>, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
}

fn read_secret_line(prompt: &str) -> Result<Option<String>, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let can_hide = cfg!(unix) && io::stdin().is_terminal();
    let echo_disabled = if can_hide {
        Command::new("stty")
            .arg("-echo")
            .status()
            .ok()
            .is_some_and(|status| status.success())
    } else {
        false
    };

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;

    if echo_disabled {
        let _ = Command::new("stty").arg("echo").status();
        println!();
    }

    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
}

fn parse_setup_value(kind: FieldKind, raw: &str) -> Result<SetupValue, String> {
    match kind {
        FieldKind::String => Ok(SetupValue::String(raw.to_string())),
        FieldKind::Bool => parse_bool_value(raw).map(SetupValue::Bool),
        FieldKind::Integer => raw
            .parse::<i64>()
            .map(SetupValue::Integer)
            .map_err(|_| format!("expected an integer for '{raw}'")),
        FieldKind::Float => raw
            .parse::<f64>()
            .map(SetupValue::Float)
            .map_err(|_| format!("expected a number for '{raw}'")),
    }
}

fn parse_bool_value(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(format!("expected true/false for '{raw}'")),
    }
}

fn load_existing_env_value(field: &SetupField) -> Option<SetupValue> {
    let env_var = field.env_var?;
    let raw = std::env::var(env_var).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    parse_setup_value(field.kind, trimmed).ok()
}

fn load_simple_env(path: PathBuf) -> BTreeMap<String, String> {
    if !path.exists() {
        return BTreeMap::new();
    }
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn load_existing_provider_values(
    context: &HermesContext,
    provider_name: &str,
) -> Result<BTreeMap<String, SetupValue>, Box<dyn Error>> {
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let mut values = BTreeMap::new();
    let Some(memory) = get_mapping_child(&root, "memory") else {
        return Ok(values);
    };
    let Some(provider) = get_mapping_child(memory, provider_name) else {
        return Ok(values);
    };
    for (key, value) in provider {
        let Some(key) = key.as_str() else {
            continue;
        };
        if let Some(parsed) = yaml_to_setup_value(value) {
            values.insert(key.to_string(), parsed);
        }
    }
    Ok(values)
}

fn save_provider_activation(
    context: &HermesContext,
    provider_name: &str,
    provider_values: &BTreeMap<String, SetupValue>,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let memory = ensure_mapping_child(&mut root, "memory")?;
    memory.insert(
        Value::String(String::from("provider")),
        Value::String(provider_name.to_string()),
    );

    let provider_mapping = setup_values_to_yaml_mapping(provider_values);
    if !provider_mapping.is_empty() {
        memory.insert(
            Value::String(provider_name.to_string()),
            Value::Mapping(provider_mapping.clone()),
        );
    }
    if provider_name == "holographic" {
        let plugins = ensure_mapping_child(&mut root, "plugins")?;
        plugins.insert(
            Value::String(String::from("hermes-memory-store")),
            Value::Mapping(provider_mapping),
        );
    }

    write_yaml_mapping(&context.config_path(), &root)
}

fn persist_native_provider_state(
    context: &HermesContext,
    provider_name: &str,
    provider_values: &BTreeMap<String, SetupValue>,
) -> Result<(), Box<dyn Error>> {
    match provider_name {
        "hindsight" => write_json_config(
            &context.hermes_home().join("hindsight").join("config.json"),
            provider_values,
        ),
        "mem0" => write_json_config(&context.hermes_home().join("mem0.json"), provider_values),
        "supermemory" => write_json_config(
            &context.hermes_home().join("supermemory.json"),
            provider_values,
        ),
        _ => Ok(()),
    }
}

fn write_json_config(
    path: &Path,
    values: &BTreeMap<String, SetupValue>,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut root = if path.exists() {
        serde_json::from_str::<JsonValue>(&fs::read_to_string(path)?)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()
    } else {
        JsonMap::new()
    };

    for (key, value) in values {
        root.insert(key.clone(), value.to_json());
    }

    fs::write(
        path,
        serde_json::to_string_pretty(&JsonValue::Object(root))?,
    )?;
    Ok(())
}

fn ensure_hindsight_dependencies(mode: &str) -> Result<(), Box<dyn Error>> {
    let dependency = if mode == "local_embedded" {
        "hindsight-all"
    } else {
        "hindsight-client>=0.4.22"
    };
    let Some(uv_path) = resolve_binary("uv") else {
        println!("  uv not found — skipping automatic dependency install");
        return Ok(());
    };

    let status = Command::new(uv_path)
        .args([
            "pip",
            "install",
            "--python",
            &std::env::current_exe()
                .ok()
                .and_then(|_| std::env::var("PYTHON").ok())
                .unwrap_or_else(|| String::from("python3")),
            "--quiet",
            "--upgrade",
            dependency,
        ])
        .status();
    match status {
        Ok(status) if status.success() => println!("  Dependencies up to date"),
        Ok(_) => println!("  Dependency install failed — continue manually if needed"),
        Err(error) => println!("  Dependency install failed: {error}"),
    }
    Ok(())
}

fn resolve_binary(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let path = dir.join(name);
            path.is_file().then_some(path)
        })
    })
}

fn hindsight_default_model(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "claude-haiku-4-5",
        "gemini" => "gemini-2.5-flash",
        "groq" => "openai/gpt-oss-120b",
        "openrouter" => "qwen/qwen3.5-9b",
        "minimax" => "MiniMax-M2.7",
        "ollama" => "gemma3:12b",
        "lmstudio" => "local-model",
        "openai_compatible" => "your-model-name",
        _ => "gpt-4o-mini",
    }
}

fn existing_string_value(values: &BTreeMap<String, SetupValue>, key: &str) -> Option<String> {
    match values.get(key) {
        Some(SetupValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn existing_integer_value(values: &BTreeMap<String, SetupValue>, key: &str) -> Option<i64> {
    match values.get(key) {
        Some(SetupValue::Integer(value)) => Some(*value),
        _ => None,
    }
}

fn materialize_hindsight_embedded_profile_env(
    context: &HermesContext,
    provider_values: &BTreeMap<String, SetupValue>,
    env_updates: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let home = dirs::home_dir().unwrap_or_else(|| context.home_dir().to_path_buf());
    let profile_env = home.join(".hindsight").join("profiles").join("hermes.env");
    if let Some(parent) = profile_env.parent() {
        fs::create_dir_all(parent)?;
    }

    let llm_provider = existing_string_value(provider_values, "llm_provider")
        .unwrap_or_else(|| String::from("openai"));
    let daemon_provider = if matches!(llm_provider.as_str(), "openai_compatible" | "openrouter") {
        "openai".to_string()
    } else {
        llm_provider.clone()
    };
    let mut lines = vec![
        format!("HINDSIGHT_API_LLM_PROVIDER={daemon_provider}"),
        format!(
            "HINDSIGHT_API_LLM_API_KEY={}",
            env_updates
                .get("HINDSIGHT_LLM_API_KEY")
                .cloned()
                .unwrap_or_default()
        ),
        format!(
            "HINDSIGHT_API_LLM_MODEL={}",
            existing_string_value(provider_values, "llm_model").unwrap_or_default()
        ),
        String::from("HINDSIGHT_API_LOG_LEVEL=info"),
    ];
    if let Some(base_url) = existing_string_value(provider_values, "llm_base_url")
        && !base_url.trim().is_empty()
    {
        lines.push(format!("HINDSIGHT_API_LLM_BASE_URL={base_url}"));
    }
    if let Some(idle_timeout) = existing_integer_value(provider_values, "idle_timeout") {
        lines.push(format!(
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT={idle_timeout}"
        ));
    }
    fs::write(profile_env, format!("{}\n", lines.join("\n")))?;
    Ok(())
}

fn ensure_mapping_child<'a>(
    root: &'a mut Mapping,
    key: &str,
) -> Result<&'a mut Mapping, Box<dyn Error>> {
    let key_value = Value::String(key.to_string());
    if !root.contains_key(&key_value) {
        root.insert(key_value.clone(), Value::Mapping(Mapping::new()));
    }
    let child = root
        .get_mut(&key_value)
        .ok_or_else(|| format!("missing mapping: {key}"))?;
    if !matches!(child, Value::Mapping(_)) {
        *child = Value::Mapping(Mapping::new());
    }
    child
        .as_mapping_mut()
        .ok_or_else(|| format!("failed to initialize mapping: {key}").into())
}

fn get_mapping_child<'a>(root: &'a Mapping, key: &str) -> Option<&'a Mapping> {
    root.get(Value::String(key.to_string()))
        .and_then(Value::as_mapping)
}

fn setup_values_to_yaml_mapping(values: &BTreeMap<String, SetupValue>) -> Mapping {
    let mut mapping = Mapping::new();
    for (key, value) in values {
        mapping.insert(Value::String(key.clone()), value.to_yaml());
    }
    mapping
}

fn yaml_to_setup_value(value: &Value) -> Option<SetupValue> {
    match value {
        Value::String(text) => Some(SetupValue::String(text.clone())),
        Value::Bool(boolean) => Some(SetupValue::Bool(*boolean)),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                return Some(SetupValue::Integer(integer));
            }
            number.as_f64().map(SetupValue::Float)
        }
        Value::Tagged(tagged) => yaml_to_setup_value(&tagged.value),
        _ => None,
    }
}

fn mask_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::from("set");
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    if chars.len() <= 4 {
        return String::from("set");
    }
    let suffix = chars[chars.len() - 4..].iter().collect::<String>();
    format!("...{suffix}")
}

fn bridge_memory_setup_provider(provider_name: &str) -> Result<(), Box<dyn Error>> {
    let trimmed = provider_name.trim();
    if trimmed.is_empty() {
        return Err("memory provider name cannot be empty".into());
    }

    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_MEMORY_PYTHON"))
        .ok_or("could not find a Python interpreter for memory setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_MEMORY_PROVIDER", trimmed)
        .arg("-c")
        .arg(MEMORY_SETUP_PROVIDER_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("memory", status).into())
}

const MEMORY_SETUP_PROVIDER_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.memory_setup import cmd_setup_provider\n",
    "cmd_setup_provider(os.environ['HERMES_MEMORY_PROVIDER'])\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn render_status(context: &HermesContext, loaded: &LoadedConfig) -> String {
    let mut lines = Vec::new();
    let provider_name = loaded.config.memory.provider.trim();
    let installed = discover_memory_providers(context);

    lines.push(String::from("Memory status"));
    lines.push(String::from("────────────────────────────────────────"));
    lines.push(String::from("  Built-in:  always active"));
    lines.push(format!(
        "  Provider:  {}",
        if provider_name.is_empty() {
            "(none — built-in only)".to_string()
        } else {
            provider_name.to_string()
        }
    ));
    lines.push(format!(
        "  Stores:    memory={} user_profile={}",
        loaded.config.memory.memory_enabled, loaded.config.memory.user_profile_enabled
    ));

    if !provider_name.is_empty() {
        if let Some(config) = loaded
            .cfg_get(&["memory", provider_name])
            .and_then(Value::as_mapping)
        {
            if !config.is_empty() {
                lines.push(String::new());
                lines.push(format!("  {provider_name} config:"));
                for (key, value) in config {
                    let Some(key) = key.as_str() else {
                        continue;
                    };
                    lines.push(format!("    {key}: {}", render_value(value)));
                }
            }
        }

        let found = installed
            .iter()
            .any(|provider| provider.name == provider_name);
        lines.push(String::new());
        lines.push(format!(
            "  Plugin:    {}",
            if found {
                "installed ✓"
            } else {
                "NOT installed ✗"
            }
        ));
        if !found {
            lines.push(format!(
                "  Install the '{provider_name}' memory plugin to {}/plugins/",
                context.display_hermes_home()
            ));
        }
    }

    if !installed.is_empty() {
        lines.push(String::new());
        lines.push(String::from("  Installed plugins:"));
        for provider in installed {
            let active = if provider.name == provider_name {
                " ← active"
            } else {
                ""
            };
            if provider.description.trim().is_empty() {
                lines.push(format!("    • {}{}", provider.name, active));
            } else {
                lines.push(format!(
                    "    • {}  ({}){}",
                    provider.name, provider.description, active
                ));
            }
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

fn disable_external_provider(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let memory_key = Value::String(String::from("memory"));
    let provider_key = Value::String(String::from("provider"));

    let memory = if let Some(Value::Mapping(mapping)) = root.get_mut(&memory_key) {
        mapping
    } else {
        root.insert(memory_key.clone(), Value::Mapping(Default::default()));
        root.get_mut(&memory_key)
            .and_then(Value::as_mapping_mut)
            .ok_or("failed to initialize memory config mapping")?
    };
    memory.insert(provider_key, Value::String(String::new()));
    write_yaml_mapping(&context.config_path(), &root)
}

fn reset_memory_files(context: &HermesContext, args: ResetArgs) -> Result<(), Box<dyn Error>> {
    let mem_dir = context.hermes_home().join("memories");
    let candidates = selected_files(args.target);
    let existing = candidates
        .iter()
        .filter_map(|file| {
            let path = mem_dir.join(file.name);
            path.exists().then_some((path, file.clone()))
        })
        .collect::<Vec<_>>();

    if existing.is_empty() {
        println!(
            "\n  Nothing to reset — no memory files found in {}/memories/\n",
            context.display_hermes_home()
        );
        return Ok(());
    }

    println!();
    println!("  This will permanently erase the following memory files:");
    for (path, file) in &existing {
        let size = path.metadata().map(|meta| meta.len()).unwrap_or(0);
        println!(
            "    ◆ {} ({}) — {} bytes",
            file.name, file.description, size
        );
    }

    if !args.yes {
        let confirmed = prompt_yes("\n  Type 'yes' to confirm: ")?;
        if !confirmed {
            println!("  Cancelled.\n");
            return Ok(());
        }
    }

    for (path, file) in &existing {
        fs::remove_file(path)?;
        println!("  ✓ Deleted {} ({})", file.name, file.description);
    }
    println!();
    println!("  Memory reset complete. New sessions will start with a blank slate.");
    println!(
        "  Files were in: {}/memories/\n",
        context.display_hermes_home()
    );
    Ok(())
}

fn selected_files(target: ResetTarget) -> Vec<MemoryFile> {
    match target {
        ResetTarget::All => vec![
            MemoryFile {
                name: "MEMORY.md",
                description: "agent notes",
            },
            MemoryFile {
                name: "USER.md",
                description: "user profile",
            },
        ],
        ResetTarget::Memory => vec![MemoryFile {
            name: "MEMORY.md",
            description: "agent notes",
        }],
        ResetTarget::User => vec![MemoryFile {
            name: "USER.md",
            description: "user profile",
        }],
    }
}

fn prompt_yes(prompt: &str) -> Result<bool, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(false);
    }
    Ok(line.trim().eq_ignore_ascii_case("yes"))
}

fn discover_memory_providers(context: &HermesContext) -> Vec<ProviderInfo> {
    let bundled = project_root().join("plugins").join("memory");
    let user = context.hermes_home().join("plugins");
    discover_memory_providers_from_roots(&bundled, Some(&user))
}

fn discover_memory_providers_from_roots(
    bundled_root: &Path,
    user_root: Option<&Path>,
) -> Vec<ProviderInfo> {
    let mut results = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    for (root, require_marker) in [
        (bundled_root, false),
        (user_root.unwrap_or(Path::new("")), true),
    ] {
        if root.as_os_str().is_empty() || !root.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !path.is_dir() || name.starts_with(['.', '_']) || seen.contains(&name) {
                continue;
            }
            let init = path.join("__init__.py");
            if !init.exists() {
                continue;
            }
            if require_marker && !looks_like_memory_provider(&init) {
                continue;
            }
            let description = read_plugin_description(&path).unwrap_or_default();
            seen.insert(name.clone());
            results.push(ProviderInfo { name, description });
        }
    }

    results.sort_by(|left, right| left.name.cmp(&right.name));
    results
}

fn looks_like_memory_provider(init_file: &Path) -> bool {
    let Ok(source) = fs::read_to_string(init_file) else {
        return false;
    };
    let prefix = source.chars().take(8_192).collect::<String>();
    prefix.contains("register_memory_provider") || prefix.contains("MemoryProvider")
}

fn read_plugin_description(dir: &Path) -> Option<String> {
    let plugin_yaml = dir.join("plugin.yaml");
    let text = fs::read_to_string(plugin_yaml).ok()?;
    let parsed = serde_yaml::from_str::<Value>(&text).ok()?;
    parsed
        .as_mapping()
        .and_then(|mapping| mapping.get(Value::String(String::from("description"))))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn render_value(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        Value::Sequence(_) | Value::Mapping(_) => serde_yaml::to_string(value)
            .unwrap_or_default()
            .trim()
            .replace('\n', " "),
        Value::Tagged(tagged) => render_value(&tagged.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(test)]
    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    #[cfg(test)]
    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    #[test]
    fn discovery_prefers_bundled_provider_name_collisions() {
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let user = temp.path().join("user");
        fs::create_dir_all(bundled.join("honcho")).unwrap();
        fs::create_dir_all(user.join("honcho")).unwrap();
        fs::create_dir_all(user.join("custom")).unwrap();

        fs::write(
            bundled.join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            bundled.join("honcho").join("plugin.yaml"),
            "description: Bundled honcho\n",
        )
        .unwrap();
        fs::write(
            user.join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            user.join("custom").join("__init__.py"),
            "def register_memory_provider(ctx):\n    pass\n",
        )
        .unwrap();

        let providers = discover_memory_providers_from_roots(&bundled, Some(&user));
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].name, "custom");
        assert_eq!(providers[1].name, "honcho");
        assert_eq!(providers[1].description, "Bundled honcho");
    }

    #[test]
    fn reset_target_selection_matches_expected_files() {
        assert_eq!(selected_files(ResetTarget::Memory).len(), 1);
        assert_eq!(selected_files(ResetTarget::User)[0].name, "USER.md");
        assert_eq!(selected_files(ResetTarget::All).len(), 2);
    }

    #[test]
    fn disable_external_provider_clears_provider_field() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "memory:\n  provider: honcho\n  honcho:\n    workspace: test\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        disable_external_provider(&context).unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("provider: \"\"") || written.contains("provider: ''"));
        assert!(written.contains("workspace: test"));
    }

    #[test]
    fn setup_provider_spec_marks_custom_setups() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));

        let hindsight = setup_provider_spec(
            &context,
            ProviderInfo {
                name: String::from("hindsight"),
                description: String::from("Hindsight"),
            },
        );
        let mem0 = setup_provider_spec(
            &context,
            ProviderInfo {
                name: String::from("mem0"),
                description: String::from("Mem0"),
            },
        );

        assert_eq!(hindsight.mode, SetupMode::NativeHindsight);
        assert_eq!(mem0.mode, SetupMode::NativeGeneric);
        assert!(mem0.fields.iter().any(|field| field.key == "api_key"));
    }

    #[test]
    fn save_provider_activation_and_state_write_native_files() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        let mut values = BTreeMap::new();
        values.insert(
            String::from("user_id"),
            SetupValue::String(String::from("alice")),
        );
        values.insert(
            String::from("agent_id"),
            SetupValue::String(String::from("hermes")),
        );
        values.insert(String::from("rerank"), SetupValue::Bool(false));

        save_provider_activation(&context, "mem0", &values).unwrap();
        persist_native_provider_state(&context, "mem0", &values).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: mem0"));
        assert!(config_text.contains("user_id: alice"));
        assert!(config_text.contains("rerank: false"));

        let json_text = fs::read_to_string(home.join("mem0.json")).unwrap();
        assert!(json_text.contains("\"user_id\": \"alice\""));
        assert!(json_text.contains("\"rerank\": false"));
    }

    #[test]
    fn persist_hindsight_state_writes_profile_config() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        let mut values = BTreeMap::new();
        values.insert(
            String::from("mode"),
            SetupValue::String(String::from("local_embedded")),
        );
        values.insert(
            String::from("llm_provider"),
            SetupValue::String(String::from("openrouter")),
        );
        values.insert(
            String::from("llm_model"),
            SetupValue::String(String::from("qwen/qwen3.5-9b")),
        );
        values.insert(String::from("idle_timeout"), SetupValue::Integer(300));

        persist_native_provider_state(&context, "hindsight", &values).unwrap();

        let json_text = fs::read_to_string(home.join("hindsight").join("config.json")).unwrap();
        assert!(json_text.contains("\"mode\": \"local_embedded\""));
        assert!(json_text.contains("\"llm_provider\": \"openrouter\""));
    }

    #[test]
    fn materialize_hindsight_embedded_profile_env_writes_expected_vars() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let old_home = env::var_os("HOME");
        set_env_var("HOME", temp.path());

        let mut values = BTreeMap::new();
        values.insert(
            String::from("llm_provider"),
            SetupValue::String(String::from("openrouter")),
        );
        values.insert(
            String::from("llm_model"),
            SetupValue::String(String::from("qwen/qwen3.5-9b")),
        );
        values.insert(
            String::from("llm_base_url"),
            SetupValue::String(String::from("https://openrouter.ai/api/v1")),
        );
        values.insert(String::from("idle_timeout"), SetupValue::Integer(300));

        let mut env_updates = BTreeMap::new();
        env_updates.insert(
            String::from("HINDSIGHT_LLM_API_KEY"),
            String::from("secret-key"),
        );

        materialize_hindsight_embedded_profile_env(&context, &values, &env_updates).unwrap();

        let env_text = fs::read_to_string(
            temp.path()
                .join(".hindsight")
                .join("profiles")
                .join("hermes.env"),
        )
        .unwrap();
        assert!(env_text.contains("HINDSIGHT_API_LLM_PROVIDER=openai"));
        assert!(env_text.contains("HINDSIGHT_API_LLM_API_KEY=secret-key"));
        assert!(env_text.contains("HINDSIGHT_API_LLM_MODEL=qwen/qwen3.5-9b"));
        assert!(env_text.contains("HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1"));
        assert!(env_text.contains("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=300"));

        match old_home {
            Some(value) => set_env_var("HOME", value),
            None => remove_env_var("HOME"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bridge_memory_setup_provider_uses_python_override() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'provider=%s\\n' \"$HERMES_MEMORY_PROVIDER\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_MEMORY_PYTHON", &fake_python);
        bridge_memory_setup_provider("hindsight").unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("provider=hindsight"));

        remove_env_var("HERMES_MEMORY_PYTHON");
    }

    #[test]
    fn render_status_shows_active_provider_and_installed_plugins() {
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("plugins").join("memory");
        fs::create_dir_all(bundled.join("honcho")).unwrap();
        fs::write(
            bundled.join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            bundled.join("honcho").join("plugin.yaml"),
            "description: Honcho provider\n",
        )
        .unwrap();

        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let loaded = LoadedConfig {
            path: home.join("config.yaml"),
            raw: serde_yaml::from_str(
                "memory:\n  provider: honcho\n  honcho:\n    workspace: local\n",
            )
            .unwrap(),
            config: hermes_core::HermesConfig {
                memory: hermes_core::MemoryConfig {
                    provider: String::from("honcho"),
                    ..Default::default()
                },
                ..Default::default()
            },
            warnings: Vec::new(),
        };
        let output = render_status(&context, &loaded);
        assert!(output.contains("Provider:  honcho"));
        assert!(output.contains("workspace: local"));
        assert!(output.contains("Installed plugins:"));
    }
}
