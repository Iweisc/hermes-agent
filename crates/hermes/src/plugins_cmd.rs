use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::python_bridge::{launch_python_main_command, project_root};

#[derive(Subcommand, Debug)]
pub enum PluginsCommand {
    Install(InstallArgs),
    Update {
        name: String,
    },
    #[command(aliases = ["rm", "uninstall"])]
    Remove {
        name: String,
    },
    #[command(alias = "ls")]
    List,
    Enable {
        name: String,
    },
    Disable {
        name: String,
    },
}

#[derive(Args, Debug, Clone)]
pub struct InstallArgs {
    pub identifier: String,
    #[arg(short = 'f', long)]
    pub force: bool,
    #[arg(long, conflicts_with = "no_enable")]
    pub enable: bool,
    #[arg(long = "no-enable", conflicts_with = "enable")]
    pub no_enable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PluginEntry {
    name: String,
    version: String,
    description: String,
    source: String,
    path: PathBuf,
}

pub fn print_plugins(
    context: &HermesContext,
    command: Option<PluginsCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => launch_python_main_command("plugins", &[], Some("HERMES_PLUGINS_PYTHON"), &[]),
        Some(PluginsCommand::Install(args)) => bridge_install(args),
        Some(PluginsCommand::Update { name }) => bridge_update(&name),
        Some(PluginsCommand::List) => print_list(context),
        Some(PluginsCommand::Enable { name }) => enable_plugin(context, &name),
        Some(PluginsCommand::Disable { name }) => disable_plugin(context, &name),
        Some(PluginsCommand::Remove { name }) => remove_plugin(context, &name),
    }
}

fn bridge_install(args: InstallArgs) -> Result<(), Box<dyn Error>> {
    let identifier = args.identifier.trim();
    if identifier.is_empty() {
        return Err("plugin identifier cannot be empty".into());
    }
    let mut argv = vec![String::from("install"), identifier.to_string()];
    if args.force {
        argv.push(String::from("--force"));
    }
    if args.enable {
        argv.push(String::from("--enable"));
    }
    if args.no_enable {
        argv.push(String::from("--no-enable"));
    }
    launch_python_main_command("plugins", &argv, Some("HERMES_PLUGINS_PYTHON"), &[])
}

fn bridge_update(name: &str) -> Result<(), Box<dyn Error>> {
    let name = name.trim();
    if name.is_empty() {
        return Err("plugin name cannot be empty".into());
    }
    launch_python_main_command(
        "plugins",
        &[String::from("update"), name.to_string()],
        Some("HERMES_PLUGINS_PYTHON"),
        &[],
    )
}

fn print_list(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let entries = discover_all_plugins(context)?;
    if entries.is_empty() {
        println!("No plugins installed.");
        println!("Install with: hermes plugins install owner/repo");
        return Ok(());
    }

    let enabled = load_plugin_set(context, "enabled")?;
    let disabled = load_plugin_set(context, "disabled")?;

    println!(
        "{:<24} {:<12} {:<12} {:<10} Description",
        "Name", "Status", "Version", "Source"
    );
    println!(
        "{:<24} {:<12} {:<12} {:<10} -----------",
        "------------------------", "------------", "------------", "----------"
    );
    for entry in entries {
        let status = if disabled.contains(&entry.name) {
            "disabled"
        } else if enabled.contains(&entry.name) {
            "enabled"
        } else {
            "not enabled"
        };
        println!(
            "{:<24} {:<12} {:<12} {:<10} {}",
            entry.name, status, entry.version, entry.source, entry.description
        );
    }
    println!();
    println!("Interactive toggle: hermes plugins");
    println!("Enable/disable:     hermes plugins enable|disable <name>");
    println!("Plugins are opt-in by default.");
    Ok(())
}

fn enable_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let name = resolve_existing_plugin_name(context, raw_name)?
        .ok_or_else(|| format!("plugin '{}' is not installed or bundled", raw_name.trim()))?;
    let mut enabled = load_plugin_set(context, "enabled")?;
    let mut disabled = load_plugin_set(context, "disabled")?;

    if enabled.contains(&name) && !disabled.contains(&name) {
        println!("Plugin '{name}' is already enabled.");
        return Ok(());
    }

    enabled.insert(name.clone());
    disabled.remove(&name);
    save_plugin_set(context, "enabled", &enabled)?;
    save_plugin_set(context, "disabled", &disabled)?;
    println!("Enabled plugin '{name}'. Takes effect on next session.");
    Ok(())
}

fn disable_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let name = resolve_existing_plugin_name(context, raw_name)?
        .ok_or_else(|| format!("plugin '{}' is not installed or bundled", raw_name.trim()))?;
    let mut enabled = load_plugin_set(context, "enabled")?;
    let mut disabled = load_plugin_set(context, "disabled")?;

    if !enabled.contains(&name) && disabled.contains(&name) {
        println!("Plugin '{name}' is already disabled.");
        return Ok(());
    }

    enabled.remove(&name);
    disabled.insert(name.clone());
    save_plugin_set(context, "enabled", &enabled)?;
    save_plugin_set(context, "disabled", &disabled)?;
    println!("Disabled plugin '{name}'. Takes effect on next session.");
    Ok(())
}

fn remove_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let plugins_dir = user_plugins_dir(context)?;
    let (canonical_name, target) =
        resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
            format!(
                "plugin '{}' not found in {}",
                raw_name.trim(),
                plugins_dir.display()
            )
        })?;
    fs::remove_dir_all(&target)?;
    println!(
        "Removed plugin '{canonical_name}' from {}",
        plugins_dir.display()
    );
    Ok(())
}

fn discover_all_plugins(context: &HermesContext) -> Result<Vec<PluginEntry>, Box<dyn Error>> {
    let mut seen = BTreeMap::<String, PluginEntry>::new();

    let bundled_root = bundled_plugins_dir();
    if bundled_root.is_dir() {
        for child in fs::read_dir(&bundled_root)? {
            let child = child?;
            let path = child.path();
            if !path.is_dir() {
                continue;
            }
            let dir_name = child.file_name().to_string_lossy().to_string();
            if dir_name == "memory" || dir_name == "context_engine" {
                continue;
            }
            let Some(entry) = plugin_entry_from_dir(&path, "bundled")? else {
                continue;
            };
            seen.entry(entry.name.clone()).or_insert(entry);
        }
    }

    let user_root = user_plugins_dir(context)?;
    if user_root.is_dir() {
        for child in fs::read_dir(&user_root)? {
            let child = child?;
            let path = child.path();
            if !path.is_dir() {
                continue;
            }
            let source = if path.join(".git").exists() {
                "git"
            } else {
                "user"
            };
            let Some(entry) = plugin_entry_from_dir(&path, source)? else {
                continue;
            };
            seen.insert(entry.name.clone(), entry);
        }
    }

    Ok(seen.into_values().collect())
}

fn plugin_entry_from_dir(path: &Path, source: &str) -> Result<Option<PluginEntry>, Box<dyn Error>> {
    let manifest_path = manifest_path(path);
    let Some(manifest_path) = manifest_path else {
        return Ok(None);
    };
    let text = fs::read_to_string(manifest_path)?;
    let parsed = serde_yaml::from_str::<Value>(&text).unwrap_or(Value::Null);
    let mapping = parsed.as_mapping();
    let name = mapping
        .and_then(|mapping| mapping.get(Value::String(String::from("name"))))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
        });
    if name.is_empty() {
        return Ok(None);
    }
    let version = mapping
        .and_then(|mapping| mapping.get(Value::String(String::from("version"))))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let description = mapping
        .and_then(|mapping| mapping.get(Value::String(String::from("description"))))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    Ok(Some(PluginEntry {
        name: name.to_string(),
        version,
        description,
        source: source.to_string(),
        path: path.to_path_buf(),
    }))
}

fn manifest_path(path: &Path) -> Option<PathBuf> {
    let yaml = path.join("plugin.yaml");
    if yaml.exists() {
        return Some(yaml);
    }
    let yml = path.join("plugin.yml");
    yml.exists().then_some(yml)
}

fn bundled_plugins_dir() -> PathBuf {
    if let Some(value) = std::env::var_os("HERMES_BUNDLED_PLUGINS") {
        let trimmed = value.to_string_lossy().trim().to_string();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    project_root().join("plugins")
}

fn user_plugins_dir(context: &HermesContext) -> Result<PathBuf, Box<dyn Error>> {
    let path = context.hermes_home().join("plugins");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn resolve_existing_plugin_name(
    context: &HermesContext,
    raw_name: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let requested = validate_plugin_name(raw_name)?;
    if let Some((name, _)) = resolve_installed_plugin(context, &requested)? {
        return Ok(Some(name));
    }

    let bundled_root = bundled_plugins_dir();
    if bundled_root.is_dir() {
        let direct = bundled_root.join(&requested);
        if direct.is_dir() && manifest_path(&direct).is_some() {
            return Ok(Some(requested));
        }
        for child in fs::read_dir(bundled_root)? {
            let child = child?;
            let path = child.path();
            if !path.is_dir() {
                continue;
            }
            let Some(entry) = plugin_entry_from_dir(&path, "bundled")? else {
                continue;
            };
            if entry.name == requested {
                return Ok(Some(entry.name));
            }
        }
    }

    Ok(None)
}

fn resolve_installed_plugin(
    context: &HermesContext,
    raw_name: &str,
) -> Result<Option<(String, PathBuf)>, Box<dyn Error>> {
    let requested = validate_plugin_name(raw_name)?;
    let plugins_dir = user_plugins_dir(context)?;
    let direct = plugins_dir.join(&requested);
    if direct.is_dir() {
        return Ok(Some((requested, direct)));
    }

    for child in fs::read_dir(&plugins_dir)? {
        let child = child?;
        let path = child.path();
        if !path.is_dir() {
            continue;
        }
        let Some(entry) = plugin_entry_from_dir(&path, "user")? else {
            continue;
        };
        if entry.name == requested {
            return Ok(Some((entry.name, path)));
        }
    }

    Ok(None)
}

fn load_plugin_set(
    context: &HermesContext,
    field: &str,
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let raw = read_raw_yaml_mapping(&context.config_path())?;
    let plugins = raw
        .get(Value::String(String::from("plugins")))
        .and_then(Value::as_mapping);
    let values = plugins
        .and_then(|mapping| mapping.get(Value::String(field.to_string())))
        .and_then(Value::as_sequence)
        .cloned()
        .unwrap_or_default();
    Ok(values
        .into_iter()
        .filter_map(|value| value.as_str().map(str::trim).map(ToOwned::to_owned))
        .filter(|value| !value.is_empty())
        .collect())
}

fn save_plugin_set(
    context: &HermesContext,
    field: &str,
    values: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let plugins_key = Value::String(String::from("plugins"));
    let field_key = Value::String(field.to_string());

    let plugins = if let Some(Value::Mapping(mapping)) = root.get_mut(&plugins_key) {
        mapping
    } else {
        root.insert(plugins_key.clone(), Value::Mapping(Mapping::new()));
        root.get_mut(&plugins_key)
            .and_then(Value::as_mapping_mut)
            .ok_or("failed to initialize plugins config mapping")?
    };
    plugins.insert(
        field_key,
        Value::Sequence(
            values
                .iter()
                .cloned()
                .map(Value::String)
                .collect::<Vec<_>>(),
        ),
    );
    write_yaml_mapping(&context.config_path(), &root)
}

fn validate_plugin_name(raw_name: &str) -> Result<String, Box<dyn Error>> {
    let name = raw_name.trim();
    if name.is_empty() {
        return Err("plugin name cannot be empty".into());
    }
    if matches!(name, "." | "..") {
        return Err("plugin name cannot reference the plugins directory itself".into());
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err("plugin name cannot contain path separators or traversal".into());
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn discover_plugins_prefers_user_over_bundled() {
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let user = temp.path().join("user");
        fs::create_dir_all(bundled.join("demo")).unwrap();
        fs::create_dir_all(user.join("demo")).unwrap();
        fs::create_dir_all(user.join("other")).unwrap();

        fs::write(
            bundled.join("demo").join("plugin.yaml"),
            "name: demo\nversion: 1.0.0\ndescription: bundled\n",
        )
        .unwrap();
        fs::write(
            user.join("demo").join("plugin.yaml"),
            "name: demo\nversion: 2.0.0\ndescription: user\n",
        )
        .unwrap();
        fs::write(user.join("demo").join(".git"), "").unwrap();
        fs::write(
            user.join("other").join("plugin.yml"),
            "name: other\ndescription: user other\n",
        )
        .unwrap();

        let old_override = std::env::var_os("HERMES_BUNDLED_PLUGINS");
        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().join(".hermes")));
        fs::create_dir_all(context.hermes_home().join("plugins")).unwrap();
        fs::rename(
            user.join("demo"),
            context.hermes_home().join("plugins").join("demo"),
        )
        .unwrap();
        fs::rename(
            user.join("other"),
            context.hermes_home().join("plugins").join("other"),
        )
        .unwrap();

        let entries = discover_all_plugins(&context).unwrap();
        assert_eq!(entries.len(), 2);
        let demo = entries.iter().find(|entry| entry.name == "demo").unwrap();
        assert_eq!(demo.version, "2.0.0");
        assert_eq!(demo.source, "git");

        match old_override {
            Some(value) => unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", value) },
            None => unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") },
        }
    }

    #[test]
    fn enable_disable_roundtrip_updates_config() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(home.join("plugins").join("demo")).unwrap();
        fs::write(
            home.join("plugins").join("demo").join("plugin.yaml"),
            "name: demo\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        enable_plugin(&context, "demo").unwrap();
        let config = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config.contains("enabled:"));
        assert!(config.contains("- demo"));

        disable_plugin(&context, "demo").unwrap();
        let config = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config.contains("disabled:"));
        assert!(config.contains("- demo"));
    }

    #[test]
    fn remove_plugin_resolves_manifest_name() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(home.join("plugins").join("disk-cleanup")).unwrap();
        fs::write(
            home.join("plugins")
                .join("disk-cleanup")
                .join("plugin.yaml"),
            "name: cleanup-helper\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        remove_plugin(&context, "cleanup-helper").unwrap();
        assert!(!home.join("plugins").join("disk-cleanup").exists());
    }

    #[test]
    fn validate_plugin_name_rejects_traversal() {
        assert!(validate_plugin_name("../bad").is_err());
        assert!(validate_plugin_name("bad/name").is_err());
        assert!(validate_plugin_name("").is_err());
    }
}
