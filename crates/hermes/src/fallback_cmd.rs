use std::error::Error;
use std::io::{self, BufRead, Write};

use clap::Subcommand;
use hermes_core::{
    LoadedConfig, get_provider_profile, infer_api_mode_from_base_url, normalize_model_for_provider,
    normalize_provider_alias, resolve_provider_api_mode,
};
use serde_yaml::{Mapping, Sequence, Value};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::model_cmd::prompt_model_selection_with_io;

#[derive(Subcommand, Debug)]
pub enum FallbackCommand {
    #[command(alias = "ls")]
    List,
    Add {
        model: Option<String>,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long = "base-url")]
        base_url: Option<String>,
        #[arg(long = "api-mode")]
        api_mode: Option<String>,
    },
    #[command(alias = "rm")]
    Remove { index: Option<usize> },
    Clear {
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FallbackEntry {
    provider: String,
    model: String,
    base_url: Option<String>,
    api_mode: Option<String>,
}

pub fn print_fallback(
    config_path: &std::path::Path,
    loaded: &LoadedConfig,
    command: Option<FallbackCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(FallbackCommand::List) {
        FallbackCommand::List => print_chain(config_path, loaded)?,
        FallbackCommand::Add {
            model,
            provider,
            base_url,
            api_mode,
        } => match (model, provider) {
            (Some(model), Some(provider)) => add_fallback(
                config_path,
                loaded,
                &model,
                &provider,
                base_url.as_deref(),
                api_mode.as_deref(),
            )?,
            (None, None) => {
                let stdin = io::stdin();
                let stdout = io::stdout();
                let mut input = stdin.lock();
                let mut output = stdout.lock();
                add_fallback_interactive_with_io(config_path, loaded, &mut input, &mut output)?;
            }
            _ => {
                return Err("fallback add requires both <model> and --provider, or neither".into());
            }
        },
        FallbackCommand::Remove { index } => {
            if let Some(index) = index {
                remove_fallback(config_path, index)?;
            } else {
                let stdin = io::stdin();
                let stdout = io::stdout();
                let mut input = stdin.lock();
                let mut output = stdout.lock();
                remove_fallback_with_io(config_path, &mut input, &mut output)?;
            }
        }
        FallbackCommand::Clear { yes } => clear_fallbacks(config_path, yes)?,
    }
    Ok(())
}

fn print_chain(config_path: &std::path::Path, loaded: &LoadedConfig) -> Result<(), Box<dyn Error>> {
    let root = read_raw_yaml_mapping(config_path)?;
    let chain = read_chain(&root);

    println!();
    if chain.is_empty() {
        println!("  No fallback providers configured.");
        println!();
        println!("  Add one with:  hermes fallback add <model> --provider <provider>");
        println!();
        return Ok(());
    }

    if let Some(primary) = describe_primary(loaded) {
        println!("  Primary:   {primary}");
        println!();
    }
    println!(
        "  Fallback chain ({} {}):",
        chain.len(),
        if chain.len() == 1 { "entry" } else { "entries" }
    );
    for (index, entry) in chain.iter().enumerate() {
        println!("    {}. {}", index + 1, format_entry(entry));
    }
    println!();
    println!("  Tried in order when the primary fails (rate-limit, 5xx, connection errors).");
    println!("  Config: {}", config_path.display());
    println!();
    Ok(())
}

fn add_fallback(
    config_path: &std::path::Path,
    loaded: &LoadedConfig,
    model_input: &str,
    provider_input: &str,
    base_url_input: Option<&str>,
    api_mode_input: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let provider = validate_provider(provider_input)?;
    let model_input = validate_model(model_input)?;
    let normalized_model = normalize_model_for_provider(&model_input, &provider);
    let base_url = resolve_base_url(&provider, base_url_input)?;
    let api_mode = resolve_api_mode(
        &provider,
        &normalized_model,
        base_url.as_deref(),
        api_mode_input,
    )?;
    let new_entry = FallbackEntry {
        provider: provider.clone(),
        model: normalized_model,
        base_url,
        api_mode,
    };

    if let Some(primary) = primary_entry(loaded) {
        if primary.provider == new_entry.provider && primary.model == new_entry.model {
            return Err(format!(
                "selected fallback matches the current primary ({})",
                format_entry(&new_entry)
            )
            .into());
        }
    }

    let mut root = read_raw_yaml_mapping(config_path)?;
    let mut chain = read_chain(&root);
    if chain
        .iter()
        .any(|entry| entry.provider == new_entry.provider && entry.model == new_entry.model)
    {
        println!();
        println!(
            "  {} is already in the fallback chain — skipped.",
            format_entry(&new_entry)
        );
        return Ok(());
    }

    chain.push(new_entry.clone());
    write_chain(&mut root, &chain);
    write_yaml_mapping(config_path, &root)?;

    println!();
    println!("  Added fallback: {}", format_entry(&new_entry));
    println!(
        "  Chain is now {} {} long.",
        chain.len(),
        if chain.len() == 1 { "entry" } else { "entries" }
    );
    println!();
    println!(
        "  Run `hermes fallback list` to view, or `hermes fallback remove <index>` to delete."
    );
    Ok(())
}

fn add_fallback_interactive_with_io(
    config_path: &std::path::Path,
    loaded: &LoadedConfig,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "Hermes Fallback Picker")?;
    writeln!(
        output,
        "  Select a provider + model to append to the fallback chain."
    )?;
    let Some(selection) = prompt_model_selection_with_io(loaded, input, output)? else {
        writeln!(output, "  No fallback added.")?;
        return Ok(());
    };
    add_fallback(
        config_path,
        loaded,
        &selection.model,
        &selection.provider,
        selection.base_url.as_deref(),
        selection.api_mode.as_deref(),
    )
}

fn remove_fallback(config_path: &std::path::Path, index: usize) -> Result<(), Box<dyn Error>> {
    if index == 0 {
        return Err("fallback index must be 1 or greater".into());
    }
    let mut root = read_raw_yaml_mapping(config_path)?;
    let mut chain = read_chain(&root);
    if chain.is_empty() {
        println!();
        println!("  No fallback providers configured — nothing to remove.");
        println!();
        return Ok(());
    }
    if index > chain.len() {
        return Err(format!(
            "fallback index {} is out of range (chain length {})",
            index,
            chain.len()
        )
        .into());
    }

    let removed = chain.remove(index - 1);
    write_chain(&mut root, &chain);
    write_yaml_mapping(config_path, &root)?;

    println!();
    println!("  Removed fallback: {}", format_entry(&removed));
    if chain.is_empty() {
        println!("  Fallback chain is now empty.");
    } else {
        println!(
            "  Chain is now {} {} long.",
            chain.len(),
            if chain.len() == 1 { "entry" } else { "entries" }
        );
    }
    println!();
    Ok(())
}

fn remove_fallback_with_io(
    config_path: &std::path::Path,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(config_path)?;
    let mut chain = read_chain(&root);
    if chain.is_empty() {
        writeln!(output)?;
        writeln!(
            output,
            "  No fallback providers configured — nothing to remove."
        )?;
        writeln!(output)?;
        return Ok(());
    }

    writeln!(output)?;
    writeln!(output, "Select a fallback to remove:")?;
    for (index, entry) in chain.iter().enumerate() {
        writeln!(output, "  {}. {}", index + 1, format_entry(entry))?;
    }
    writeln!(output, "  q. Cancel")?;

    loop {
        let response = prompt_line(input, output, "Selection")?;
        let trimmed = response.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("q") {
            writeln!(output, "  Cancelled — no change.")?;
            return Ok(());
        }
        let Ok(index) = trimmed.parse::<usize>() else {
            writeln!(output, "  Invalid selection: '{trimmed}'.")?;
            continue;
        };
        if !(1..=chain.len()).contains(&index) {
            writeln!(output, "  Selection must be between 1 and {}.", chain.len())?;
            continue;
        }

        let removed = chain.remove(index - 1);
        write_chain(&mut root, &chain);
        write_yaml_mapping(config_path, &root)?;
        writeln!(output)?;
        writeln!(output, "  Removed fallback: {}", format_entry(&removed))?;
        if chain.is_empty() {
            writeln!(output, "  Fallback chain is now empty.")?;
        } else {
            writeln!(
                output,
                "  Chain is now {} {} long.",
                chain.len(),
                if chain.len() == 1 { "entry" } else { "entries" }
            )?;
        }
        writeln!(output)?;
        return Ok(());
    }
}

fn clear_fallbacks(config_path: &std::path::Path, yes: bool) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(config_path)?;
    let chain = read_chain(&root);
    if chain.is_empty() {
        println!();
        println!("  No fallback providers configured — nothing to clear.");
        println!();
        return Ok(());
    }

    if !yes {
        println!();
        println!(
            "  Current fallback chain ({} {}):",
            chain.len(),
            if chain.len() == 1 { "entry" } else { "entries" }
        );
        for (index, entry) in chain.iter().enumerate() {
            println!("    {}. {}", index + 1, format_entry(entry));
        }
        println!();
        print!("  Clear all entries? [y/N]: ");
        io::stdout().flush()?;
        let mut response = String::new();
        io::stdin().read_line(&mut response)?;
        let response = response.trim().to_ascii_lowercase();
        if !matches!(response.as_str(), "y" | "yes") {
            println!("  Cancelled — no change.");
            return Ok(());
        }
    }

    write_chain(&mut root, &[]);
    write_yaml_mapping(config_path, &root)?;
    println!();
    println!("  Fallback chain cleared.");
    println!();
    Ok(())
}

fn describe_primary(loaded: &LoadedConfig) -> Option<String> {
    let model = loaded.configured_model_name()?;
    let provider = loaded.configured_model_provider();
    Some(match provider {
        Some(provider) => format!("{model}  (via {provider})"),
        None => model,
    })
}

fn primary_entry(loaded: &LoadedConfig) -> Option<FallbackEntry> {
    let provider = loaded.configured_model_provider()?;
    let model = loaded.configured_model_name()?;
    let base_url = loaded.configured_model_base_url();
    let api_mode = loaded.configured_model_api_mode();
    Some(FallbackEntry {
        provider: provider.clone(),
        model: normalize_model_for_provider(&model, &provider),
        base_url,
        api_mode,
    })
}

fn format_entry(entry: &FallbackEntry) -> String {
    let suffix = entry
        .base_url
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(|value| format!("  [{value}]"))
        .unwrap_or_default();
    format!("{}  (via {}){}", entry.model, entry.provider, suffix)
}

fn read_chain(root: &Mapping) -> Vec<FallbackEntry> {
    if let Some(chain) = root
        .get(Value::String("fallback_providers".to_string()))
        .and_then(Value::as_sequence)
    {
        let entries = parse_entry_sequence(chain);
        if !entries.is_empty() {
            return entries;
        }
    }

    match root.get(Value::String("fallback_model".to_string())) {
        Some(Value::Mapping(mapping)) => parse_entry_mapping(mapping).into_iter().collect(),
        Some(Value::Sequence(sequence)) => parse_entry_sequence(sequence),
        _ => Vec::new(),
    }
}

fn parse_entry_sequence(sequence: &Sequence) -> Vec<FallbackEntry> {
    sequence
        .iter()
        .filter_map(Value::as_mapping)
        .filter_map(parse_entry_mapping)
        .collect()
}

fn parse_entry_mapping(mapping: &Mapping) -> Option<FallbackEntry> {
    let provider = mapping_string(mapping, "provider")?;
    let model = mapping_string(mapping, "model")?;
    Some(FallbackEntry {
        provider,
        model,
        base_url: mapping_string(mapping, "base_url"),
        api_mode: mapping_string(mapping, "api_mode"),
    })
}

fn write_chain(root: &mut Mapping, chain: &[FallbackEntry]) {
    root.insert(
        Value::String("fallback_providers".to_string()),
        Value::Sequence(
            chain
                .iter()
                .map(|entry| {
                    let mut mapping = Mapping::new();
                    mapping.insert(
                        Value::String("provider".to_string()),
                        Value::String(entry.provider.clone()),
                    );
                    mapping.insert(
                        Value::String("model".to_string()),
                        Value::String(entry.model.clone()),
                    );
                    if let Some(base_url) = entry.base_url.as_deref() {
                        if !base_url.is_empty() {
                            mapping.insert(
                                Value::String("base_url".to_string()),
                                Value::String(base_url.to_string()),
                            );
                        }
                    }
                    if let Some(api_mode) = entry.api_mode.as_deref() {
                        if !api_mode.is_empty() {
                            mapping.insert(
                                Value::String("api_mode".to_string()),
                                Value::String(api_mode.to_string()),
                            );
                        }
                    }
                    Value::Mapping(mapping)
                })
                .collect(),
        ),
    );
    root.remove(Value::String("fallback_model".to_string()));
}

fn validate_provider(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("provider cannot be empty".into());
    }
    let provider = normalize_provider_alias(trimmed);
    if provider == "auto" {
        return Err("fallback provider cannot be auto".into());
    }
    if get_provider_profile(&provider).is_none() {
        return Err(format!("provider '{trimmed}' is not recognized").into());
    }
    Ok(provider)
}

fn validate_model(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("model cannot be empty".into());
    }
    if trimmed.contains('\n') || trimmed.contains('\r') {
        return Err("model cannot contain newlines".into());
    }
    Ok(trimmed.to_string())
}

fn resolve_base_url(
    provider: &str,
    explicit: Option<&str>,
) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(base_url) = explicit {
        let trimmed = base_url.trim();
        if trimmed.is_empty() {
            return Err("base URL cannot be empty".into());
        }
        if trimmed.contains(char::is_whitespace) || trimmed.contains('\n') || trimmed.contains('\r')
        {
            return Err("base URL cannot contain whitespace".into());
        }
        if !trimmed.contains("://") {
            return Err("base URL must include a scheme like https://".into());
        }
        return Ok(Some(trimmed.trim_end_matches('/').to_string()));
    }

    let profile = get_provider_profile(provider).ok_or("provider profile missing")?;
    if provider == "custom" || profile.base_url.trim().is_empty() {
        return Err(format!("provider '{provider}' requires --base-url").into());
    }
    Ok(Some(profile.base_url.trim_end_matches('/').to_string()))
}

fn resolve_api_mode(
    provider: &str,
    model: &str,
    base_url: Option<&str>,
    explicit: Option<&str>,
) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(api_mode) = explicit {
        let mode = api_mode.trim().to_ascii_lowercase();
        if !matches!(
            mode.as_str(),
            "chat_completions" | "anthropic_messages" | "codex_responses" | "bedrock_converse"
        ) {
            return Err(format!("unsupported api_mode '{api_mode}'").into());
        }
        return Ok(Some(mode));
    }
    if let Some(mode) = resolve_provider_api_mode(provider, model) {
        return Ok(Some(mode.to_string()));
    }
    if let Some(mode) = base_url.and_then(infer_api_mode_from_base_url) {
        return Ok(Some(mode.to_string()));
    }
    let profile = get_provider_profile(provider).ok_or("provider profile missing")?;
    Ok(Some(profile.api_mode.to_string()))
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(Value::String(key.to_string()))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn prompt_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    label: &str,
) -> Result<String, Box<dyn Error>> {
    write!(output, "{label}: ")?;
    output.flush()?;
    let mut response = String::new();
    let read = input.read_line(&mut response)?;
    if read == 0 {
        return Ok(String::new());
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-fallback-{label}-{unique}"))
    }

    #[test]
    fn read_chain_accepts_legacy_mapping() {
        let mut root = Mapping::new();
        let mut legacy = Mapping::new();
        legacy.insert(
            Value::String("provider".to_string()),
            Value::String("openrouter".to_string()),
        );
        legacy.insert(
            Value::String("model".to_string()),
            Value::String("gpt-5.4".to_string()),
        );
        root.insert(
            Value::String("fallback_model".to_string()),
            Value::Mapping(legacy),
        );
        let chain = read_chain(&root);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "openrouter");
        assert_eq!(chain[0].model, "gpt-5.4");
    }

    #[test]
    fn write_chain_migrates_to_fallback_providers() {
        let mut root = Mapping::new();
        root.insert(
            Value::String("fallback_model".to_string()),
            Value::Mapping(Mapping::new()),
        );
        write_chain(
            &mut root,
            &[FallbackEntry {
                provider: "nous".to_string(),
                model: "hermes-4".to_string(),
                base_url: None,
                api_mode: Some("chat_completions".to_string()),
            }],
        );
        assert!(
            root.get(Value::String("fallback_model".to_string()))
                .is_none()
        );
        let chain = root
            .get(Value::String("fallback_providers".to_string()))
            .and_then(Value::as_sequence)
            .unwrap();
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn add_fallback_rejects_duplicate_entries() {
        let home = temp_path("duplicate");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "model:\n  default: gpt-5.4\n  provider: openai\nfallback_providers:\n  - provider: openrouter\n    model: anthropic/claude-sonnet-4.6\n",
        )
        .unwrap();
        let context =
            hermes_core::HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = context.load_config_document().unwrap();
        add_fallback(
            &context.config_path(),
            &loaded,
            "anthropic/claude-sonnet-4.6",
            "openrouter",
            None,
            None,
        )
        .unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert_eq!(written.matches("provider: openrouter").count(), 1);
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn remove_fallback_migrates_legacy_chain() {
        let home = temp_path("remove");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "fallback_model:\n  provider: openrouter\n  model: gpt-5.4\n",
        )
        .unwrap();
        remove_fallback(&home.join("config.yaml"), 1).unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("fallback_providers: []"));
        assert!(!written.contains("fallback_model:"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn fallback_add_without_args_is_parseable() {
        let parsed = FallbackCommand::augment_subcommands(clap::Command::new("fallback"))
            .try_get_matches_from(["fallback", "add"])
            .unwrap();
        assert!(parsed.subcommand_matches("add").is_some());
    }

    #[test]
    fn fallback_remove_without_index_is_parseable() {
        let parsed = FallbackCommand::augment_subcommands(clap::Command::new("fallback"))
            .try_get_matches_from(["fallback", "remove"])
            .unwrap();
        assert!(parsed.subcommand_matches("remove").is_some());
    }

    #[test]
    fn add_fallback_interactive_uses_model_picker_selection() {
        let home = temp_path("interactive-add");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "model:\n  default: gpt-5\n  provider: openai\n",
        )
        .unwrap();
        let context =
            hermes_core::HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = context.load_config_document().unwrap();
        let mut input = Cursor::new(b"openrouter\nanthropic/claude-sonnet-4.6\n".to_vec());
        let mut output = Vec::new();
        add_fallback_interactive_with_io(&context.config_path(), &loaded, &mut input, &mut output)
            .unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("provider: openrouter"));
        assert!(written.contains("model: anthropic/claude-sonnet-4.6"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn remove_fallback_with_io_removes_selected_entry() {
        let home = temp_path("interactive-remove");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "fallback_providers:\n  - provider: openrouter\n    model: gpt-5.4\n  - provider: nous\n    model: hermes-4\n",
        )
        .unwrap();
        let mut input = Cursor::new(b"2\n".to_vec());
        let mut output = Vec::new();
        remove_fallback_with_io(&home.join("config.yaml"), &mut input, &mut output).unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("gpt-5.4"));
        assert!(!written.contains("hermes-4"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn remove_fallback_with_io_cancel_keeps_chain() {
        let home = temp_path("interactive-remove-cancel");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "fallback_providers:\n  - provider: openrouter\n    model: gpt-5.4\n",
        )
        .unwrap();
        let mut input = Cursor::new(b"q\n".to_vec());
        let mut output = Vec::new();
        remove_fallback_with_io(&home.join("config.yaml"), &mut input, &mut output).unwrap();
        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("gpt-5.4"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Cancelled"));
        let _ = fs::remove_dir_all(home);
    }
}
