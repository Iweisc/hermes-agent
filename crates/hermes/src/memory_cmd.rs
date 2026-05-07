use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::{HermesContext, LoadedConfig};
use serde::Deserialize;
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
    NativeHoncho,
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
    key: String,
    description: String,
    kind: FieldKind,
    default: Option<String>,
    secret: bool,
    env_var: Option<String>,
    url: Option<String>,
    required: bool,
    choices: Vec<String>,
    when: BTreeMap<String, String>,
    default_from: Option<DefaultFrom>,
}

#[derive(Debug, Clone)]
struct SetupProvider {
    name: String,
    description: String,
    hint: String,
    mode: SetupMode,
    fields: Vec<SetupField>,
    bridge_save_config: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum SetupValue {
    String(String),
    Bool(bool),
    Integer(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
struct DefaultFrom {
    field: String,
    map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct BridgedProviderMetadata {
    #[serde(default)]
    has_post_setup: bool,
    #[serde(default)]
    has_save_config: bool,
    #[serde(default)]
    schema: Vec<BridgedSchemaField>,
}

#[derive(Debug, Deserialize)]
struct BridgedSchemaField {
    key: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    default: Option<JsonValue>,
    #[serde(default)]
    secret: bool,
    #[serde(default)]
    env_var: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    choices: Vec<String>,
    #[serde(default)]
    when: BTreeMap<String, JsonValue>,
    #[serde(default)]
    default_from: Option<BridgedDefaultFrom>,
}

#[derive(Debug, Deserialize)]
struct BridgedDefaultFrom {
    field: String,
    #[serde(default)]
    map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct MemoryPluginManifest {
    #[serde(default)]
    pip_dependencies: Vec<String>,
    #[serde(default)]
    external_dependencies: Vec<ExternalDependency>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalDependency {
    #[serde(default)]
    name: String,
    #[serde(default)]
    install: String,
    #[serde(default)]
    check: String,
}

impl SetupField {
    fn default_value(
        &self,
        current_values: &BTreeMap<String, SetupValue>,
        existing_values: &BTreeMap<String, SetupValue>,
    ) -> Option<SetupValue> {
        if let Some(default_from) = &self.default_from
            && let Some(reference) =
                merged_setup_value_display(current_values, existing_values, &default_from.field)
            && let Some(mapped) = default_from.map.get(reference.as_str())
        {
            return parse_setup_value(self.kind, mapped).ok();
        }
        self.default
            .as_deref()
            .and_then(|raw| parse_setup_value(self.kind, raw).ok())
    }

    fn applies_to(
        &self,
        current_values: &BTreeMap<String, SetupValue>,
        existing_values: &BTreeMap<String, SetupValue>,
    ) -> bool {
        self.when.iter().all(|(key, expected)| {
            merged_setup_value_display(current_values, existing_values, key)
                .is_some_and(|actual| actual == *expected)
        })
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
            ensure_memory_setup_dependencies(context, &provider.name)?;
            match provider.mode {
                SetupMode::NativeGeneric => run_native_provider_setup(context, provider),
                SetupMode::NativeHoncho => run_honcho_provider_setup(context, provider),
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
    let (mode, fields, bridge_save_config) = match provider.name.as_str() {
        "byterover" => (
            SetupMode::NativeGeneric,
            vec![SetupField {
                key: String::from("api_key"),
                description: String::from("ByteRover API key (optional, for cloud sync)"),
                kind: FieldKind::String,
                default: None,
                secret: true,
                env_var: Some(String::from("BRV_API_KEY")),
                url: Some(String::from("https://app.byterover.dev")),
                required: false,
                choices: Vec::new(),
                when: BTreeMap::new(),
                default_from: None,
            }],
            false,
        ),
        "holographic" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: String::from("db_path"),
                    description: String::from("SQLite database path"),
                    kind: FieldKind::String,
                    default: Some(format!("{display_home}/memory_store.db")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("auto_extract"),
                    description: String::from("Auto-extract facts at session end"),
                    kind: FieldKind::Bool,
                    default: Some(String::from("false")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("default_trust"),
                    description: String::from("Default trust score for new facts"),
                    kind: FieldKind::Float,
                    default: Some(String::from("0.5")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("hrr_dim"),
                    description: String::from("HRR vector dimensions"),
                    kind: FieldKind::Integer,
                    default: Some(String::from("1024")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
            ],
            false,
        ),
        "mem0" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: String::from("api_key"),
                    description: String::from("Mem0 Platform API key"),
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some(String::from("MEM0_API_KEY")),
                    url: Some(String::from("https://app.mem0.ai")),
                    required: true,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("user_id"),
                    description: String::from("User identifier"),
                    kind: FieldKind::String,
                    default: Some(String::from("hermes-user")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("agent_id"),
                    description: String::from("Agent identifier"),
                    kind: FieldKind::String,
                    default: Some(String::from("hermes")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("rerank"),
                    description: String::from("Enable reranking for recall"),
                    kind: FieldKind::Bool,
                    default: Some(String::from("true")),
                    secret: false,
                    env_var: None,
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
            ],
            false,
        ),
        "openviking" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: String::from("endpoint"),
                    description: String::from("OpenViking server URL"),
                    kind: FieldKind::String,
                    default: Some(String::from("https://app.openviking.ai")),
                    secret: false,
                    env_var: Some(String::from("OPENVIKING_ENDPOINT")),
                    url: None,
                    required: true,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("api_key"),
                    description: String::from(
                        "OpenViking API key (leave blank for local dev mode)",
                    ),
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some(String::from("OPENVIKING_API_KEY")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("account"),
                    description: String::from("OpenViking tenant account ID"),
                    kind: FieldKind::String,
                    default: Some(String::from("default")),
                    secret: false,
                    env_var: Some(String::from("OPENVIKING_ACCOUNT")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("user"),
                    description: String::from("OpenViking user ID within the account"),
                    kind: FieldKind::String,
                    default: Some(String::from("default")),
                    secret: false,
                    env_var: Some(String::from("OPENVIKING_USER")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("agent"),
                    description: String::from("OpenViking agent ID within the account"),
                    kind: FieldKind::String,
                    default: Some(String::from("hermes")),
                    secret: false,
                    env_var: Some(String::from("OPENVIKING_AGENT")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
            ],
            false,
        ),
        "retaindb" => (
            SetupMode::NativeGeneric,
            vec![
                SetupField {
                    key: String::from("api_key"),
                    description: String::from("RetainDB API key"),
                    kind: FieldKind::String,
                    default: None,
                    secret: true,
                    env_var: Some(String::from("RETAINDB_API_KEY")),
                    url: Some(String::from("https://retaindb.com")),
                    required: true,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("base_url"),
                    description: String::from("API endpoint"),
                    kind: FieldKind::String,
                    default: Some(String::from("https://api.retaindb.com")),
                    secret: false,
                    env_var: Some(String::from("RETAINDB_BASE_URL")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
                SetupField {
                    key: String::from("project"),
                    description: String::from("Project identifier"),
                    kind: FieldKind::String,
                    default: Some(String::new()),
                    secret: false,
                    env_var: Some(String::from("RETAINDB_PROJECT")),
                    url: None,
                    required: false,
                    choices: Vec::new(),
                    when: BTreeMap::new(),
                    default_from: None,
                },
            ],
            false,
        ),
        "supermemory" => (
            SetupMode::NativeGeneric,
            vec![SetupField {
                key: String::from("api_key"),
                description: String::from("Supermemory API key"),
                kind: FieldKind::String,
                default: None,
                secret: true,
                env_var: Some(String::from("SUPERMEMORY_API_KEY")),
                url: Some(String::from("https://supermemory.ai")),
                required: true,
                choices: Vec::new(),
                when: BTreeMap::new(),
                default_from: None,
            }],
            false,
        ),
        "honcho" => (SetupMode::NativeHoncho, Vec::new(), false),
        "hindsight" => (SetupMode::NativeHindsight, Vec::new(), false),
        _ => match load_bridged_provider_spec(context, &provider.name) {
            Ok(bridged) => (bridged.mode, bridged.fields, bridged.bridge_save_config),
            Err(_) => (SetupMode::PythonHook, Vec::new(), false),
        },
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
        bridge_save_config,
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

fn load_bridged_provider_spec(
    context: &HermesContext,
    provider_name: &str,
) -> Result<SetupProvider, Box<dyn Error>> {
    let metadata = load_bridged_provider_metadata(context, provider_name)?;
    if metadata.has_post_setup {
        return Ok(SetupProvider {
            name: provider_name.to_string(),
            description: String::new(),
            hint: String::from("custom setup"),
            mode: SetupMode::PythonHook,
            fields: Vec::new(),
            bridge_save_config: false,
        });
    }

    let fields = metadata
        .schema
        .into_iter()
        .map(bridged_schema_field_to_setup_field)
        .collect::<Result<Vec<_>, _>>()?;
    let hint = render_setup_hint(&fields);

    Ok(SetupProvider {
        name: provider_name.to_string(),
        description: String::new(),
        hint,
        mode: SetupMode::NativeGeneric,
        fields,
        bridge_save_config: metadata.has_save_config,
    })
}

fn load_bridged_provider_metadata(
    _context: &HermesContext,
    provider_name: &str,
) -> Result<BridgedProviderMetadata, Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_MEMORY_PYTHON"))
        .ok_or("could not find a Python interpreter for memory setup")?;

    let output = Command::new(&python)
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_MEMORY_PROVIDER", provider_name)
        .arg("-c")
        .arg(MEMORY_SETUP_METADATA_BOOTSTRAP)
        .output()?;
    if !output.status.success() {
        return Err(exit_status_message("memory metadata", output.status).into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    serde_json::from_str::<BridgedProviderMetadata>(stdout.trim()).map_err(|error| {
        format!("invalid memory provider metadata for {provider_name}: {error}").into()
    })
}

fn bridged_schema_field_to_setup_field(
    field: BridgedSchemaField,
) -> Result<SetupField, Box<dyn Error>> {
    if field.key.trim().is_empty() {
        return Err("memory setup field key cannot be empty".into());
    }
    let when = field
        .when
        .into_iter()
        .map(|(key, value)| (key, json_value_to_string(value)))
        .collect::<BTreeMap<_, _>>();

    Ok(SetupField {
        key: field.key.clone(),
        description: field.description.unwrap_or_else(|| field.key.clone()),
        kind: infer_field_kind(&field.default, &field.choices),
        default: field.default.and_then(json_value_to_string_option),
        secret: field.secret,
        env_var: field.env_var,
        url: field.url,
        required: field.required,
        choices: field.choices,
        when,
        default_from: field.default_from.map(|value| DefaultFrom {
            field: value.field,
            map: value.map,
        }),
    })
}

fn infer_field_kind(default: &Option<JsonValue>, choices: &[String]) -> FieldKind {
    if let Some(default) = default {
        match default {
            JsonValue::Bool(_) => return FieldKind::Bool,
            JsonValue::Number(number) if number.is_i64() || number.is_u64() => {
                return FieldKind::Integer;
            }
            JsonValue::Number(_) => return FieldKind::Float,
            _ => {}
        }
    }
    if !choices.is_empty()
        && choices
            .iter()
            .all(|choice| parse_bool_value(choice).is_ok())
    {
        return FieldKind::Bool;
    }
    FieldKind::String
}

fn json_value_to_string_option(value: JsonValue) -> Option<String> {
    match value {
        JsonValue::Null => None,
        JsonValue::Bool(boolean) => Some(boolean.to_string()),
        JsonValue::Number(number) => Some(number.to_string()),
        JsonValue::String(text) => Some(text),
        other => Some(other.to_string()),
    }
}

fn json_value_to_string(value: JsonValue) -> String {
    json_value_to_string_option(value).unwrap_or_default()
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
        if !field.applies_to(&provider_values, &existing) {
            continue;
        }
        let existing_value = existing
            .get(field.key.as_str())
            .cloned()
            .or_else(|| load_existing_env_value(field));
        let default_value = field.default_value(&provider_values, &existing);

        let Some(value) = prompt_setup_field(field, existing_value.clone(), default_value)? else {
            println!("  Cancelled.\n");
            return Ok(());
        };

        if field.secret {
            let rendered = value.env_string();
            if !rendered.trim().is_empty() {
                if let Some(env_var) = &field.env_var {
                    env_updates.insert(env_var.clone(), rendered);
                }
            }
            continue;
        }

        provider_values.insert(field.key.clone(), value.clone());
        let rendered = value.env_string();
        if let Some(env_var) = &field.env_var {
            if !rendered.trim().is_empty() {
                env_updates.insert(env_var.clone(), rendered);
            }
        }
    }

    save_provider_activation(context, &provider.name, &provider_values)?;
    persist_native_provider_state(context, &provider.name, &provider_values)?;
    if provider.bridge_save_config && !provider_values.is_empty() {
        bridge_memory_save_config(context, &provider.name, &provider_values)?;
    }
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

fn run_honcho_provider_setup(
    context: &HermesContext,
    provider: &SetupProvider,
) -> Result<(), Box<dyn Error>> {
    let mut config = load_honcho_setup_config(context);
    let existing_env = load_simple_env(context.env_path());
    let host_key = honcho_host_key(context);
    let current_host = config
        .get("hosts")
        .and_then(|value| value.as_object())
        .and_then(|hosts| hosts.get(&host_key))
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();

    println!("\n  Configuring {}:\n", provider.name);

    ensure_honcho_dependency()?;

    let current_deploy = if honcho_base_url(&config).is_some_and(|url| is_local_base_url(&url)) {
        "local"
    } else {
        "cloud"
    };
    let deploy = prompt_choice(
        "  Deployment",
        &[
            ("cloud", "Honcho cloud (api.honcho.dev)"),
            ("local", "Self-hosted Honcho server"),
        ],
        current_deploy,
    )?;
    let is_local = deploy == "local";

    if is_local {
        let base_url = prompt_http_url(
            "  Base URL",
            honcho_base_url(&config)
                .as_deref()
                .unwrap_or("http://localhost:8000"),
        )?;
        config.insert(String::from("baseUrl"), serde_json::Value::String(base_url));
        println!("  Local connections will skip auth automatically.");
    } else {
        config.remove("baseUrl");
        config.remove("base_url");

        let existing_key = current_host
            .get("apiKey")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| honcho_api_key(&config))
            .or_else(|| existing_env.get("HONCHO_API_KEY").cloned())
            .unwrap_or_default();
        let api_key = prompt_secret_with_existing(
            "  Honcho API key",
            (!existing_key.trim().is_empty()).then_some(existing_key.as_str()),
            false,
        )?;
        if api_key.trim().is_empty() {
            return Err("No API key configured. Set one and run setup again.".into());
        }
        save_env_value(context.env_path(), "HONCHO_API_KEY", &api_key)?;
    }

    config.remove("apiKey");

    let user_default = std::env::var("USER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| String::from("user"));
    let peer_name = prompt_string_value(
        "  Your name (user peer)",
        current_host
            .get("peerName")
            .and_then(|value| value.as_str())
            .or_else(|| config.get("peerName").and_then(|value| value.as_str()))
            .unwrap_or(user_default.as_str()),
    )?;

    let ai_default = if host_key == "hermes" {
        "hermes".to_string()
    } else {
        host_key
            .strip_prefix("hermes.")
            .map(str::to_string)
            .unwrap_or_else(|| String::from("hermes"))
    };
    let ai_peer = prompt_string_value(
        "  AI peer name",
        current_host
            .get("aiPeer")
            .and_then(|value| value.as_str())
            .or_else(|| config.get("aiPeer").and_then(|value| value.as_str()))
            .unwrap_or(ai_default.as_str()),
    )?;

    let workspace = prompt_string_value(
        "  Workspace ID",
        current_host
            .get("workspace")
            .and_then(|value| value.as_str())
            .or_else(|| config.get("workspace").and_then(|value| value.as_str()))
            .unwrap_or("hermes"),
    )?;

    let observation_mode = prompt_choice(
        "  Observation mode",
        &[
            ("directional", "each AI peer builds its own view"),
            ("unified", "shared pool across peers"),
        ],
        current_host
            .get("observationMode")
            .and_then(|value| value.as_str())
            .or_else(|| {
                config
                    .get("observationMode")
                    .and_then(|value| value.as_str())
            })
            .unwrap_or("directional"),
    )?;

    let write_frequency = prompt_honcho_write_frequency(
        current_host
            .get("writeFrequency")
            .or_else(|| config.get("writeFrequency"))
            .unwrap_or(&serde_json::Value::String(String::from("async"))),
    )?;

    let recall_mode = prompt_choice(
        "  Recall mode",
        &[
            ("hybrid", "auto-injected context + Honcho tools"),
            ("context", "auto-injected context only"),
            ("tools", "Honcho tools only"),
        ],
        current_host
            .get("recallMode")
            .and_then(|value| value.as_str())
            .or_else(|| config.get("recallMode").and_then(|value| value.as_str()))
            .unwrap_or("hybrid"),
    )?;

    let context_tokens = prompt_honcho_context_tokens(
        current_host
            .get("contextTokens")
            .or_else(|| config.get("contextTokens"))
            .and_then(honcho_integer),
    )?;

    let dialectic_cadence = prompt_integer_value(
        "  Dialectic cadence",
        current_host
            .get("dialecticCadence")
            .or_else(|| config.get("dialecticCadence"))
            .and_then(honcho_integer)
            .unwrap_or(2),
    )?;
    let dialectic_cadence = std::cmp::max(dialectic_cadence, 1);

    let reasoning_level = prompt_choice(
        "  Reasoning level",
        &[
            ("minimal", "quick factual lookups"),
            ("low", "straightforward questions"),
            ("medium", "multi-aspect synthesis"),
            ("high", "complex behavioral patterns"),
            ("max", "audit-level analysis"),
        ],
        current_host
            .get("dialecticReasoningLevel")
            .and_then(|value| value.as_str())
            .or_else(|| {
                config
                    .get("dialecticReasoningLevel")
                    .and_then(|value| value.as_str())
            })
            .unwrap_or("low"),
    )?;

    let session_strategy = prompt_choice(
        "  Session strategy",
        &[
            ("per-session", "start clean each run"),
            ("per-directory", "reuse per directory"),
            ("per-repo", "reuse per git repository"),
            ("global", "single shared session"),
        ],
        current_host
            .get("sessionStrategy")
            .and_then(|value| value.as_str())
            .or_else(|| {
                config
                    .get("sessionStrategy")
                    .and_then(|value| value.as_str())
            })
            .unwrap_or("per-session"),
    )?;
    let preserve_save_messages = current_host.contains_key("saveMessages");

    let host = ensure_honcho_host_block(&mut config, &host_key);
    host.remove("apiKey");
    host.insert(
        String::from("peerName"),
        serde_json::Value::String(peer_name),
    );
    host.insert(String::from("aiPeer"), serde_json::Value::String(ai_peer));
    host.insert(
        String::from("workspace"),
        serde_json::Value::String(workspace),
    );
    host.insert(
        String::from("observationMode"),
        serde_json::Value::String(observation_mode),
    );
    host.insert(String::from("writeFrequency"), write_frequency);
    host.insert(
        String::from("recallMode"),
        serde_json::Value::String(recall_mode),
    );
    match context_tokens {
        Some(value) => {
            host.insert(
                String::from("contextTokens"),
                serde_json::Value::from(value),
            );
        }
        None => {
            host.remove("contextTokens");
        }
    }
    host.insert(
        String::from("dialecticCadence"),
        serde_json::Value::from(dialectic_cadence),
    );
    host.insert(
        String::from("dialecticReasoningLevel"),
        serde_json::Value::String(reasoning_level),
    );
    host.insert(
        String::from("sessionStrategy"),
        serde_json::Value::String(session_strategy),
    );
    host.insert(String::from("enabled"), serde_json::Value::Bool(true));
    if !preserve_save_messages {
        host.insert(String::from("saveMessages"), serde_json::Value::Bool(true));
    }

    save_honcho_config(context, &config)?;
    save_memory_provider_only(context, "honcho")?;

    println!("\n  Memory provider: {}", provider.name);
    println!("  Activation saved to config.yaml");
    println!(
        "  Honcho config saved to {}",
        context.hermes_home().join("honcho.json").display()
    );
    if !is_local {
        println!("  API key saved to .env");
    }
    println!("\n  Start a new session to activate.\n");
    Ok(())
}

fn prompt_setup_field(
    field: &SetupField,
    existing_value: Option<SetupValue>,
    default_value: Option<SetupValue>,
) -> Result<Option<SetupValue>, Box<dyn Error>> {
    if !field.choices.is_empty() && !field.secret {
        return prompt_setup_choice_field(field, existing_value, default_value).map(Some);
    }
    loop {
        let prompt = field_prompt_label(field, existing_value.as_ref(), default_value.as_ref());
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
            if let Some(value) = existing_value.clone().or_else(|| default_value.clone()) {
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

fn prompt_setup_choice_field(
    field: &SetupField,
    existing_value: Option<SetupValue>,
    default_value: Option<SetupValue>,
) -> Result<SetupValue, Box<dyn Error>> {
    let current = existing_value
        .as_ref()
        .or(default_value.as_ref())
        .map(SetupValue::display)
        .unwrap_or_else(|| field.choices[0].clone());
    let choice = prompt_dynamic_choice(
        format!("  {}", field.description).as_str(),
        &field.choices,
        current.as_str(),
    )?;
    parse_setup_value(field.kind, choice.as_str())
        .map_err(|error| format!("invalid choice for {}: {error}", field.key).into())
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

fn prompt_http_url(label: &str, current: &str) -> Result<String, Box<dyn Error>> {
    loop {
        let value = prompt_string_value(label, current)?;
        if value.starts_with("http://") || value.starts_with("https://") {
            return Ok(value);
        }
        println!("  Invalid URL — must start with http:// or https://.");
    }
}

fn prompt_honcho_write_frequency(
    current: &serde_json::Value,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let current_display = honcho_display_value(current);
    loop {
        let raw = prompt_string_value("  Write frequency", current_display.as_str())?;
        if let Some(value) = parse_honcho_write_frequency(&raw) {
            return Ok(value);
        }
        println!("  Enter async, turn, session, or a non-negative integer.");
    }
}

fn prompt_honcho_context_tokens(current: Option<i64>) -> Result<Option<i64>, Box<dyn Error>> {
    let current_display = current
        .map(|value| value.to_string())
        .unwrap_or_else(|| String::from("uncapped"));
    loop {
        let raw = prompt_string_value("  Context tokens", current_display.as_str())?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" | "uncapped" | "no limit" => return Ok(None),
            _ => match raw.trim().parse::<i64>() {
                Ok(value) if value >= 0 => return Ok(Some(value)),
                _ => println!("  Enter a non-negative integer or 'uncapped'."),
            },
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

fn field_prompt_label(
    field: &SetupField,
    existing_value: Option<&SetupValue>,
    default_value: Option<&SetupValue>,
) -> String {
    if field.secret {
        if let Some(value) = existing_value {
            return format!(
                "  {} (current: {}, blank to keep): ",
                field.description,
                mask_secret(&value.env_string())
            );
        }
        if let Some(url) = field.url.as_deref() {
            println!("  Get yours at {url}");
        }
        return format!("  {}: ", field.description);
    }

    if let Some(value) = existing_value {
        return format!("  {} [{}]: ", field.description, value.display());
    }
    if let Some(value) = default_value {
        return format!("  {} [{}]: ", field.description, value.display());
    }
    format!("  {}: ", field.description)
}

fn prompt_dynamic_choice(
    label: &str,
    options: &[String],
    current: &str,
) -> Result<String, Box<dyn Error>> {
    println!("{label}:");
    let default_index = options
        .iter()
        .position(|value| value.eq_ignore_ascii_case(current))
        .unwrap_or(0);
    for (index, value) in options.iter().enumerate() {
        let active = if index == default_index {
            " ← current"
        } else {
            ""
        };
        println!("    {}) {}{}", index + 1, value, active);
    }

    loop {
        let prompt = format!("  Select [{}]: ", default_index + 1);
        let Some(input) = read_prompt_line(&prompt)? else {
            return Err("setup cancelled".into());
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(options[default_index].clone());
        }
        if let Ok(index) = trimmed.parse::<usize>()
            && index >= 1
            && index <= options.len()
        {
            return Ok(options[index - 1].clone());
        }
        if let Some(value) = options
            .iter()
            .find(|value| value.eq_ignore_ascii_case(trimmed))
        {
            return Ok(value.clone());
        }
        println!("  Invalid selection.");
    }
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
    let env_var = field.env_var.as_ref()?;
    let raw = std::env::var(env_var).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    parse_setup_value(field.kind, trimmed).ok()
}

fn merged_setup_value_display(
    current_values: &BTreeMap<String, SetupValue>,
    existing_values: &BTreeMap<String, SetupValue>,
    key: &str,
) -> Option<String> {
    current_values
        .get(key)
        .or_else(|| existing_values.get(key))
        .map(SetupValue::display)
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

fn save_memory_provider_only(
    context: &HermesContext,
    provider_name: &str,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let memory = ensure_mapping_child(&mut root, "memory")?;
    memory.insert(
        Value::String(String::from("provider")),
        Value::String(provider_name.to_string()),
    );
    write_yaml_mapping(&context.config_path(), &root)
}

fn ensure_hindsight_dependencies(mode: &str) -> Result<(), Box<dyn Error>> {
    let dependency = if mode == "local_embedded" {
        "hindsight-all"
    } else {
        "hindsight-client>=0.4.22"
    };
    if let Some(python) = resolve_memory_python() {
        install_missing_memory_python_dependencies(&python, &[dependency.to_string()])?;
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

fn ensure_honcho_dependency() -> Result<(), Box<dyn Error>> {
    if let Some(python) = resolve_memory_python() {
        install_missing_memory_python_dependencies(&python, &[String::from("honcho-ai>=2.0.1")])?;
    }
    Ok(())
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

fn load_honcho_setup_config(context: &HermesContext) -> BTreeMap<String, serde_json::Value> {
    for path in honcho_config_candidates(context) {
        if path.exists()
            && let Ok(text) = fs::read_to_string(&path)
            && let Ok(serde_json::Value::Object(map)) =
                serde_json::from_str::<serde_json::Value>(&text)
        {
            return map.into_iter().collect();
        }
    }
    BTreeMap::new()
}

fn honcho_config_candidates(context: &HermesContext) -> Vec<PathBuf> {
    let mut paths = vec![context.hermes_home().join("honcho.json")];
    if let Some(home) = dirs::home_dir() {
        let default_path = home.join(".hermes").join("honcho.json");
        if !paths.contains(&default_path) {
            paths.push(default_path);
        }
        let legacy = home.join(".honcho").join("config.json");
        if !paths.contains(&legacy) {
            paths.push(legacy);
        }
    }
    paths
}

fn save_honcho_config(
    context: &HermesContext,
    config: &BTreeMap<String, serde_json::Value>,
) -> Result<(), Box<dyn Error>> {
    let path = context.hermes_home().join("honcho.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&serde_json::Value::Object(
                config.clone().into_iter().collect(),
            ))?
        ),
    )?;
    Ok(())
}

fn honcho_host_key(context: &HermesContext) -> String {
    let profile = context.current_profile_name();
    if matches!(profile.as_str(), "default" | "custom") {
        String::from("hermes")
    } else {
        format!("hermes.{profile}")
    }
}

fn ensure_honcho_host_block<'a>(
    config: &'a mut BTreeMap<String, serde_json::Value>,
    host_key: &str,
) -> &'a mut serde_json::Map<String, serde_json::Value> {
    let hosts = config
        .entry(String::from("hosts"))
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !hosts.is_object() {
        *hosts = serde_json::Value::Object(serde_json::Map::new());
    }
    let hosts = hosts.as_object_mut().expect("hosts object");
    let entry = hosts
        .entry(host_key.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        *entry = serde_json::Value::Object(serde_json::Map::new());
    }
    entry.as_object_mut().expect("host object")
}

fn honcho_base_url(config: &BTreeMap<String, serde_json::Value>) -> Option<String> {
    config
        .get("baseUrl")
        .or_else(|| config.get("base_url"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn is_local_base_url(url: &str) -> bool {
    ["localhost", "127.0.0.1", "::1"]
        .iter()
        .any(|needle| url.contains(needle))
}

fn honcho_api_key(config: &BTreeMap<String, serde_json::Value>) -> Option<String> {
    config
        .get("apiKey")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn honcho_display_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::Bool(boolean) => boolean.to_string(),
        _ => String::new(),
    }
}

fn parse_honcho_write_frequency(raw: &str) -> Option<serde_json::Value> {
    if let Ok(value) = raw.trim().parse::<i64>() {
        if value >= 0 {
            return Some(serde_json::Value::from(value));
        }
        return None;
    }
    match raw.trim() {
        "async" | "turn" | "session" => Some(serde_json::Value::String(raw.trim().to_string())),
        _ => None,
    }
}

fn honcho_integer(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(number) => number.as_i64(),
        serde_json::Value::String(text) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
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

fn bridge_memory_save_config(
    context: &HermesContext,
    provider_name: &str,
    provider_values: &BTreeMap<String, SetupValue>,
) -> Result<(), Box<dyn Error>> {
    let trimmed = provider_name.trim();
    if trimmed.is_empty() {
        return Err("memory provider name cannot be empty".into());
    }

    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_MEMORY_PYTHON"))
        .ok_or("could not find a Python interpreter for memory setup")?;
    let values_json = serde_json::to_string(&setup_values_to_json_object(provider_values))?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_MEMORY_PROVIDER", trimmed)
        .env("HERMES_MEMORY_PROVIDER_VALUES_JSON", values_json)
        .env(
            "HERMES_MEMORY_HOME",
            context.hermes_home().display().to_string(),
        )
        .arg("-c")
        .arg(MEMORY_SAVE_CONFIG_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("memory save-config", status).into())
}

fn setup_values_to_json_object(values: &BTreeMap<String, SetupValue>) -> JsonValue {
    let mut root = JsonMap::new();
    for (key, value) in values {
        root.insert(key.clone(), value.to_json());
    }
    JsonValue::Object(root)
}

const MEMORY_SETUP_METADATA_BOOTSTRAP: &str = concat!(
    "import json\n",
    "import os\n",
    "from plugins.memory import load_memory_provider\n",
    "provider = load_memory_provider(os.environ['HERMES_MEMORY_PROVIDER'])\n",
    "if provider is None:\n",
    "    raise SystemExit(2)\n",
    "payload = {\n",
    "    'has_post_setup': hasattr(provider, 'post_setup'),\n",
    "    'has_save_config': hasattr(provider, 'save_config'),\n",
    "    'schema': provider.get_config_schema() if hasattr(provider, 'get_config_schema') else [],\n",
    "}\n",
    "print(json.dumps(payload, default=str))\n",
);

const MEMORY_SETUP_PROVIDER_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.memory_setup import cmd_setup_provider\n",
    "cmd_setup_provider(os.environ['HERMES_MEMORY_PROVIDER'])\n",
);

const MEMORY_SAVE_CONFIG_BOOTSTRAP: &str = concat!(
    "import json\n",
    "import os\n",
    "from plugins.memory import load_memory_provider\n",
    "provider = load_memory_provider(os.environ['HERMES_MEMORY_PROVIDER'])\n",
    "if provider is None:\n",
    "    raise SystemExit(2)\n",
    "if hasattr(provider, 'save_config'):\n",
    "    provider.save_config(\n",
    "        json.loads(os.environ['HERMES_MEMORY_PROVIDER_VALUES_JSON']),\n",
    "        os.environ['HERMES_MEMORY_HOME'],\n",
    "    )\n",
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

fn find_memory_provider_dir(context: &HermesContext, provider_name: &str) -> Option<PathBuf> {
    let bundled = project_root()
        .join("plugins")
        .join("memory")
        .join(provider_name);
    if bundled.is_dir() && bundled.join("__init__.py").exists() {
        return Some(bundled);
    }

    let user = context.hermes_home().join("plugins").join(provider_name);
    if user.is_dir()
        && user.join("__init__.py").exists()
        && looks_like_memory_provider(&user.join("__init__.py"))
    {
        return Some(user);
    }
    None
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

fn read_memory_plugin_manifest(dir: &Path) -> Option<MemoryPluginManifest> {
    let plugin_yaml = dir.join("plugin.yaml");
    let text = fs::read_to_string(plugin_yaml).ok()?;
    serde_yaml::from_str::<MemoryPluginManifest>(&text).ok()
}

fn ensure_memory_setup_dependencies(
    context: &HermesContext,
    provider_name: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(provider_dir) = find_memory_provider_dir(context, provider_name) else {
        return Ok(());
    };
    let Some(manifest) = read_memory_plugin_manifest(&provider_dir) else {
        return Ok(());
    };

    if let Some(python) = resolve_memory_python() {
        install_missing_memory_python_dependencies(&python, &manifest.pip_dependencies)?;
    }
    print_missing_external_dependency_hints(&manifest.external_dependencies);
    Ok(())
}

fn resolve_memory_python() -> Option<PathBuf> {
    let root = project_root();
    resolve_repo_python(&root, Some("HERMES_MEMORY_PYTHON"))
}

fn install_missing_memory_python_dependencies(
    python: &Path,
    dependencies: &[String],
) -> Result<(), Box<dyn Error>> {
    if dependencies.is_empty() {
        return Ok(());
    }

    let missing = dependencies
        .iter()
        .filter(|dependency| !python_import_available(python, dependency))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    println!("\n  Installing dependencies: {}", missing.join(", "));

    let Some(uv_path) = resolve_binary("uv") else {
        println!("  uv not found — cannot install dependencies");
        return Ok(());
    };

    let output = Command::new(uv_path)
        .args(["pip", "install", "--python"])
        .arg(python)
        .arg("--quiet")
        .args(&missing)
        .output()?;

    if output.status.success() {
        println!("  Dependencies installed");
        return Ok(());
    }

    println!("  Dependency install failed — continue manually if needed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(line) = stderr.lines().find(|line| !line.trim().is_empty()) {
        println!("  {}", line.trim());
    }
    Ok(())
}

fn python_import_available(python: &Path, dependency: &str) -> bool {
    let import_name = python_dependency_import_name(dependency);
    Command::new(python)
        .arg("-c")
        .arg(format!("import {import_name}"))
        .output()
        .ok()
        .is_some_and(|output| output.status.success())
}

fn python_dependency_import_name(dependency: &str) -> String {
    let package = dependency
        .split(['<', '>', '=', '!', '~', '['])
        .next()
        .unwrap_or(dependency)
        .trim();
    match package {
        "honcho-ai" => String::from("honcho"),
        "mem0ai" => String::from("mem0"),
        "hindsight-client" => String::from("hindsight_client"),
        "hindsight-all" => String::from("hindsight"),
        other => other.replace('-', "_"),
    }
}

fn print_missing_external_dependency_hints(dependencies: &[ExternalDependency]) {
    for dependency in dependencies {
        let check = dependency.check.trim();
        let install = dependency.install.trim();
        if check.is_empty() || install.is_empty() {
            continue;
        }
        if external_dependency_is_available(check) {
            continue;
        }

        let name = dependency.name.trim();
        if name.is_empty() {
            println!("  Missing external dependency. Install with: {install}");
        } else {
            println!("  Missing external dependency '{name}'. Install with: {install}");
        }
    }
}

fn external_dependency_is_available(check: &str) -> bool {
    let Some(binary) = check.split_whitespace().next() else {
        return true;
    };
    resolve_binary(binary).is_some()
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
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        crate::cli_test_env_lock()
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
        let honcho = setup_provider_spec(
            &context,
            ProviderInfo {
                name: String::from("honcho"),
                description: String::from("Honcho"),
            },
        );

        assert_eq!(hindsight.mode, SetupMode::NativeHindsight);
        assert_eq!(honcho.mode, SetupMode::NativeHoncho);
        assert_eq!(mem0.mode, SetupMode::NativeGeneric);
        assert!(mem0.fields.iter().any(|field| field.key == "api_key"));
    }

    #[test]
    fn setup_field_uses_default_from_and_when_matches_existing_values() {
        let field = SetupField {
            key: String::from("llm_model"),
            description: String::from("LLM model"),
            kind: FieldKind::String,
            default: Some(String::from("gpt-4o-mini")),
            secret: false,
            env_var: None,
            url: None,
            required: false,
            choices: vec![String::from("gpt-4o-mini"), String::from("qwen/qwen3.5-9b")],
            when: BTreeMap::from([(String::from("mode"), String::from("local"))]),
            default_from: Some(DefaultFrom {
                field: String::from("llm_provider"),
                map: BTreeMap::from([(
                    String::from("openrouter"),
                    String::from("qwen/qwen3.5-9b"),
                )]),
            }),
        };
        let current = BTreeMap::new();
        let existing = BTreeMap::from([
            (
                String::from("mode"),
                SetupValue::String(String::from("local")),
            ),
            (
                String::from("llm_provider"),
                SetupValue::String(String::from("openrouter")),
            ),
        ]);

        assert!(field.applies_to(&current, &existing));
        assert_eq!(
            field.default_value(&current, &existing),
            Some(SetupValue::String(String::from("qwen/qwen3.5-9b")))
        );
    }

    #[test]
    fn read_memory_plugin_manifest_parses_dependency_lists() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("plugin.yaml"),
            "pip_dependencies:\n  - mem0ai\nexternal_dependencies:\n  - name: brv\n    install: curl install\n    check: brv --version\n",
        )
        .unwrap();

        let manifest = read_memory_plugin_manifest(temp.path()).unwrap();
        assert_eq!(manifest.pip_dependencies, vec!["mem0ai"]);
        assert_eq!(manifest.external_dependencies.len(), 1);
        assert_eq!(manifest.external_dependencies[0].name, "brv");
    }

    #[test]
    fn python_dependency_import_name_maps_special_cases() {
        assert_eq!(python_dependency_import_name("honcho-ai>=2.0.1"), "honcho");
        assert_eq!(python_dependency_import_name("mem0ai"), "mem0");
        assert_eq!(
            python_dependency_import_name("hindsight-client>=0.4.22"),
            "hindsight_client"
        );
        assert_eq!(python_dependency_import_name("supermemory"), "supermemory");
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
    fn save_honcho_config_writes_profile_scoped_host_block() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes").join("profiles").join("coder");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        let mut config = BTreeMap::new();
        config.insert(
            String::from("baseUrl"),
            serde_json::Value::String(String::from("http://localhost:8000")),
        );
        let host = ensure_honcho_host_block(&mut config, &honcho_host_key(&context));
        host.insert(
            String::from("peerName"),
            serde_json::Value::String(String::from("alice")),
        );
        host.insert(
            String::from("workspace"),
            serde_json::Value::String(String::from("hermes")),
        );
        host.insert(String::from("enabled"), serde_json::Value::Bool(true));

        save_honcho_config(&context, &config).unwrap();
        save_memory_provider_only(&context, "honcho").unwrap();

        let honcho_text = fs::read_to_string(home.join("honcho.json")).unwrap();
        assert!(honcho_text.contains("\"baseUrl\": \"http://localhost:8000\""));
        assert!(honcho_text.contains("\"hermes.coder\""));
        assert!(honcho_text.contains("\"peerName\": \"alice\""));

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: honcho"));
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
    #[cfg(unix)]
    fn setup_provider_spec_loads_dynamic_metadata_for_schema_only_provider() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        fs::write(
            &fake_python,
            "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  cat <<'EOF'\n\
{\"has_post_setup\": false, \"has_save_config\": true, \"schema\": [{\"key\": \"mode\", \"description\": \"Mode\", \"default\": \"cloud\", \"choices\": [\"cloud\", \"local\"]}, {\"key\": \"token\", \"description\": \"Access token\", \"secret\": true, \"env_var\": \"CUSTOM_TOKEN\"}]}\n\
EOF\n\
  exit 0\n\
fi\n\
exit 9\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_MEMORY_PYTHON", &fake_python);
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let provider = setup_provider_spec(
            &context,
            ProviderInfo {
                name: String::from("custom"),
                description: String::from("Custom"),
            },
        );

        assert_eq!(provider.mode, SetupMode::NativeGeneric);
        assert!(provider.bridge_save_config);
        assert_eq!(provider.fields.len(), 2);
        assert_eq!(provider.fields[0].choices, vec!["cloud", "local"]);
        assert_eq!(provider.fields[1].env_var.as_deref(), Some("CUSTOM_TOKEN"));

        remove_env_var("HERMES_MEMORY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn setup_provider_spec_keeps_post_setup_providers_on_python_hook() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        fs::write(
            &fake_python,
            "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf '%s\\n' '{\"has_post_setup\": true, \"has_save_config\": true, \"schema\": [{\"key\": \"token\"}]}'\n\
  exit 0\n\
fi\n\
exit 9\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_MEMORY_PYTHON", &fake_python);
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let provider = setup_provider_spec(
            &context,
            ProviderInfo {
                name: String::from("custom"),
                description: String::from("Custom"),
            },
        );

        assert_eq!(provider.mode, SetupMode::PythonHook);
        assert!(provider.fields.is_empty());
        assert!(!provider.bridge_save_config);

        remove_env_var("HERMES_MEMORY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn ensure_memory_setup_dependencies_installs_missing_pip_dependencies() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("demo");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            "class Demo(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "pip_dependencies:\n  - missingpkg>=1.0\n",
        )
        .unwrap();

        let fake_python = temp.path().join("python3");
        fs::write(
            &fake_python,
            "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  case \"$2\" in\n\
    *\"import missingpkg\"*) exit 1 ;;\n\
    *) exit 0 ;;\n\
  esac\n\
fi\n\
exit 9\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let fake_uv = temp.path().join("uv");
        let uv_log = temp.path().join("uv.log");
        fs::write(
            &fake_uv,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
                uv_log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_uv).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_uv, perms).unwrap();

        let old_path = env::var_os("PATH");
        set_env_var(
            "PATH",
            format!(
                "{}:{}",
                temp.path().display(),
                env::var("PATH").unwrap_or_default()
            ),
        );
        set_env_var("HERMES_MEMORY_PYTHON", &fake_python);

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        ensure_memory_setup_dependencies(&context, "demo").unwrap();

        let output = fs::read_to_string(&uv_log).unwrap();
        assert!(output.contains("pip install --python"));
        assert!(output.contains("missingpkg>=1.0"));

        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        remove_env_var("HERMES_MEMORY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn bridge_memory_save_config_uses_python_override() {
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
  printf 'home=%s\\n' \"$HERMES_MEMORY_HOME\" >> '{}'\n\
  printf 'values=%s\\n' \"$HERMES_MEMORY_PROVIDER_VALUES_JSON\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display(),
                log.display(),
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let mut values = BTreeMap::new();
        values.insert(
            String::from("workspace"),
            SetupValue::String(String::from("alpha")),
        );
        values.insert(String::from("enabled"), SetupValue::Bool(true));

        set_env_var("HERMES_MEMORY_PYTHON", &fake_python);
        bridge_memory_save_config(&context, "custom", &values).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("provider=custom"));
        assert!(output.contains(format!("home={}", home.display()).as_str()));
        assert!(output.contains("\"workspace\":\"alpha\""));
        assert!(output.contains("\"enabled\":true"));

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
