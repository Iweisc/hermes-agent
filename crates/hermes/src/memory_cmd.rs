use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::{HermesContext, LoadedConfig};
use serde_yaml::Value;

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::python_bridge::{launch_python_main_command, project_root};

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

pub fn print_memory(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<MemoryCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(MemoryCommand::Status) {
        MemoryCommand::Setup => launch_python_main_command(
            "memory",
            &[String::from("setup")],
            Some("HERMES_MEMORY_PYTHON"),
            &[],
        ),
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
    use tempfile::TempDir;

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
