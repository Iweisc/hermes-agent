use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use serde_yaml::{Mapping, Value};
use tempfile::TempDir;

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::python_bridge::{project_root, resolve_repo_python};

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

#[derive(Debug, Clone)]
struct EnvSpec {
    name: String,
    description: String,
    url: String,
    secret: bool,
}

const SUPPORTED_MANIFEST_VERSION: i64 = 1;

pub fn print_plugins(
    context: &HermesContext,
    command: Option<PluginsCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => bridge_plugins(None, &[]),
        Some(PluginsCommand::Install(args)) => install_plugin(context, args),
        Some(PluginsCommand::Update { name }) => update_plugin(context, &name),
        Some(PluginsCommand::List) => print_list(context),
        Some(PluginsCommand::Enable { name }) => enable_plugin(context, &name),
        Some(PluginsCommand::Disable { name }) => disable_plugin(context, &name),
        Some(PluginsCommand::Remove { name }) => remove_plugin(context, &name),
    }
}

fn bridge_plugins(action: Option<&str>, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_PLUGINS_PYTHON"))
        .ok_or("could not find a Python interpreter for plugins")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_PLUGINS_ACTION", action.unwrap_or(""))
        .arg("-c")
        .arg(PLUGINS_BOOTSTRAP)
        .args(passthrough);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("plugins", status).into())
}

const PLUGINS_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "import sys\n",
    "from hermes_cli.plugins_cmd import plugins_command\n",
    "action = (os.environ.get('HERMES_PLUGINS_ACTION') or '').strip()\n",
    "parser = argparse.ArgumentParser(prog='hermes plugins')\n",
    "parser.set_defaults(plugins_action=(action or None))\n",
    "if action == 'install':\n",
    "    parser.add_argument('identifier')\n",
    "    parser.add_argument('--force', '-f', action='store_true')\n",
    "    group = parser.add_mutually_exclusive_group()\n",
    "    group.add_argument('--enable', action='store_true')\n",
    "    group.add_argument('--no-enable', action='store_true')\n",
    "elif action == 'update':\n",
    "    parser.add_argument('name')\n",
    "elif action:\n",
    "    raise SystemExit(f'unsupported plugins action: {action}')\n",
    "plugins_command(parser.parse_args(sys.argv[1:]))\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn install_plugin(context: &HermesContext, args: InstallArgs) -> Result<(), Box<dyn Error>> {
    let identifier = args.identifier.trim();
    if identifier.is_empty() {
        return Err("plugin identifier cannot be empty".into());
    }

    let git_url = resolve_git_url(identifier)?;
    if git_url.starts_with("http://") || git_url.starts_with("file://") {
        println!("Warning: using insecure/local URL scheme: {git_url}");
    }
    println!("Cloning {git_url}...");

    let (target, manifest, installed_name) = install_plugin_core(context, identifier, args.force)?;
    copy_example_files(&target)?;
    prompt_plugin_env_vars(context, &manifest)?;
    display_after_install(&target, identifier)?;

    let should_enable = resolve_install_enable_choice(&args, &installed_name)?;
    if should_enable {
        let mut enabled = load_plugin_set(context, "enabled")?;
        let mut disabled = load_plugin_set(context, "disabled")?;
        enabled.insert(installed_name.clone());
        disabled.remove(&installed_name);
        save_plugin_set(context, "enabled", &enabled)?;
        save_plugin_set(context, "disabled", &disabled)?;
        println!("Enabled plugin '{installed_name}'.");
    } else {
        println!("Plugin installed but not enabled.");
        println!("Run `hermes plugins enable {installed_name}` to activate.");
    }

    println!("Restart the gateway for the plugin to take effect:");
    println!("  hermes gateway restart");
    Ok(())
}

fn update_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let (name, target) = resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
        format!(
            "plugin '{}' not found in {}",
            raw_name.trim(),
            user_plugins_dir(context)
                .ok()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| String::from("plugins"))
        )
    })?;
    if !target.join(".git").exists() {
        return Err(format!(
            "plugin '{name}' was not installed from git (no .git directory). Cannot update."
        )
        .into());
    }

    println!("Updating {name}...");
    let output = git_pull_plugin_dir(&target)?;
    copy_example_files(&target)?;

    if output.contains("Already up to date") {
        println!("Plugin '{name}' is already up to date.");
    } else {
        println!("Plugin '{name}' updated.");
        if !output.trim().is_empty() {
            println!("{output}");
        }
    }
    Ok(())
}

fn install_plugin_core(
    context: &HermesContext,
    identifier: &str,
    force: bool,
) -> Result<(PathBuf, Mapping, String), Box<dyn Error>> {
    let git_url = resolve_git_url(identifier)?;
    let plugins_dir = user_plugins_dir(context)?;
    let temp = TempDir::new()?;
    let temp_target = temp.path().join("plugin");

    run_git_clone(&git_url, &temp_target)?;

    let manifest = read_manifest_mapping(&temp_target)?;
    let plugin_name = manifest
        .get(Value::String(String::from("name")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| repo_name_from_url(&git_url));

    validate_manifest_version(&manifest, &plugin_name)?;
    let safe_name = validate_plugin_name(&plugin_name)?;
    let target = plugins_dir.join(&safe_name);

    if target.exists() {
        if !force {
            return Err(format!(
                "plugin '{plugin_name}' already exists. Use --force or run `hermes plugins update {plugin_name}`."
            )
            .into());
        }
        fs::remove_dir_all(&target)?;
    }

    fs::rename(&temp_target, &target)?;

    if manifest_path(&target).is_none() && !target.join("__init__.py").exists() {
        println!(
            "Warning: {} has no plugin.yaml / __init__.py; it may not be a valid Hermes plugin.",
            plugin_name
        );
    }

    let installed_manifest = read_manifest_mapping(&target)?;
    let installed_name = installed_manifest
        .get(Value::String(String::from("name")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&safe_name)
        .to_string();

    Ok((target, installed_manifest, installed_name))
}

fn resolve_git_url(identifier: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = identifier.trim();
    if trimmed.starts_with("https://")
        || trimmed.starts_with("http://")
        || trimmed.starts_with("git@")
        || trimmed.starts_with("ssh://")
        || trimmed.starts_with("file://")
    {
        return Ok(trimmed.to_string());
    }

    let parts = trimmed.trim_matches('/').split('/').collect::<Vec<_>>();
    if parts.len() == 2 && parts.iter().all(|part| !part.trim().is_empty()) {
        return Ok(format!(
            "https://github.com/{}/{}.git",
            parts[0].trim(),
            parts[1].trim()
        ));
    }

    Err(
        format!("invalid plugin identifier: '{trimmed}'. Use a Git URL or owner/repo shorthand.")
            .into(),
    )
}

fn repo_name_from_url(url: &str) -> String {
    let mut trimmed = url.trim().trim_end_matches('/').to_string();
    if trimmed.ends_with(".git") {
        trimmed.truncate(trimmed.len() - 4);
    }
    let last = trimmed
        .rsplit('/')
        .next()
        .unwrap_or(trimmed.as_str())
        .to_string();
    last.rsplit(':').next().unwrap_or(last.as_str()).to_string()
}

fn run_git_clone(git_url: &str, target: &Path) -> Result<(), Box<dyn Error>> {
    let output = Command::new("git")
        .args(["clone", "--depth", "1", git_url])
        .arg(target)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err("git is not installed or not in PATH.".into());
        }
        Err(error) => return Err(error.into()),
    };
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Err(format!(
        "Git clone failed:\n{}",
        if !stderr.is_empty() { stderr } else { stdout }
    )
    .into())
}

fn git_pull_plugin_dir(target: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .current_dir(target)
        .args(["pull", "--ff-only"])
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err("git is not installed or not in PATH.".into());
        }
        Err(error) => return Err(error.into()),
    };
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Err(if !stderr.is_empty() { stderr } else { stdout }.into())
}

fn read_manifest_mapping(path: &Path) -> Result<Mapping, Box<dyn Error>> {
    let Some(manifest_path) = manifest_path(path) else {
        return Ok(Mapping::new());
    };
    let text = fs::read_to_string(manifest_path)?;
    let parsed = serde_yaml::from_str::<Value>(&text)?;
    match parsed {
        Value::Mapping(mapping) => Ok(mapping),
        Value::Null => Ok(Mapping::new()),
        _ => Err(format!("{} must contain a YAML mapping", path.display()).into()),
    }
}

fn validate_manifest_version(manifest: &Mapping, plugin_name: &str) -> Result<(), Box<dyn Error>> {
    let Some(value) = manifest.get(Value::String(String::from("manifest_version"))) else {
        return Ok(());
    };
    let version = match value {
        Value::Number(number) => number.as_i64().ok_or_else(|| {
            format!("Plugin '{plugin_name}' has invalid manifest_version '{number}'.")
        })?,
        Value::String(text) => text.trim().parse::<i64>().map_err(|_| {
            format!(
                "Plugin '{plugin_name}' has invalid manifest_version '{}'.",
                text.trim()
            )
        })?,
        _ => {
            return Err(format!(
                "Plugin '{plugin_name}' has invalid manifest_version; expected an integer."
            )
            .into());
        }
    };
    if version > SUPPORTED_MANIFEST_VERSION {
        return Err(format!(
            "Plugin '{plugin_name}' requires manifest_version {version}, but this installer only supports up to {SUPPORTED_MANIFEST_VERSION}. Run `hermes update` to update Hermes."
        )
        .into());
    }
    Ok(())
}

fn copy_example_files(plugin_dir: &Path) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(plugin_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(real_name) = file_name.strip_suffix(".example") else {
            continue;
        };
        let target = plugin_dir.join(real_name);
        if target.exists() {
            continue;
        }
        fs::copy(&path, &target)?;
        println!("  Created {real_name} from {file_name}");
    }
    Ok(())
}

fn prompt_plugin_env_vars(
    context: &HermesContext,
    manifest: &Mapping,
) -> Result<(), Box<dyn Error>> {
    let specs = parse_manifest_env_specs(manifest);
    if specs.is_empty() {
        return Ok(());
    }

    let missing = specs
        .into_iter()
        .filter(|spec| {
            std::env::var(&spec.name)
                .ok()
                .is_none_or(|value| value.trim().is_empty())
        })
        .collect::<Vec<_>>();

    if missing.is_empty() {
        return Ok(());
    }

    let plugin_name = manifest
        .get(Value::String(String::from("name")))
        .and_then(Value::as_str)
        .unwrap_or("this plugin");
    println!("\n{plugin_name} requires the following environment variables:\n");

    for spec in missing {
        if !spec.description.trim().is_empty() {
            println!("  {} — {}", spec.name, spec.description);
        } else {
            println!("  {}", spec.name);
        }
        if !spec.url.trim().is_empty() {
            println!("  Get yours at: {}", spec.url);
        }

        let prompt = format!("  {}: ", spec.name);
        let value = if spec.secret {
            read_prompt(prompt.as_str(), true)?
        } else {
            read_prompt(prompt.as_str(), false)?
        };
        let Some(value) = value else {
            println!(
                "\n  Skipped (you can set these later in {}/.env)\n",
                context.display_hermes_home()
            );
            return Ok(());
        };
        let trimmed = value.trim();
        if trimmed.is_empty() {
            println!(
                "  Skipped (set {} in {}/.env later)",
                spec.name,
                context.display_hermes_home()
            );
            continue;
        }

        save_env_value(context.env_path(), &spec.name, trimmed)?;
        unsafe {
            std::env::set_var(&spec.name, trimmed);
        }
        println!("  Saved to {}/.env", context.display_hermes_home());
    }
    println!();
    Ok(())
}

fn parse_manifest_env_specs(manifest: &Mapping) -> Vec<EnvSpec> {
    let Some(value) = manifest.get(Value::String(String::from("requires_env"))) else {
        return Vec::new();
    };
    let Some(items) = value.as_sequence() else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|item| match item {
            Value::String(name) => {
                let trimmed = name.trim();
                (!trimmed.is_empty()).then(|| EnvSpec {
                    name: trimmed.to_string(),
                    description: String::new(),
                    url: String::new(),
                    secret: false,
                })
            }
            Value::Mapping(mapping) => {
                let name = mapping
                    .get(Value::String(String::from("name")))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())?;
                Some(EnvSpec {
                    name: name.to_string(),
                    description: mapping
                        .get(Value::String(String::from("description")))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    url: mapping
                        .get(Value::String(String::from("url")))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    secret: mapping
                        .get(Value::String(String::from("secret")))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                })
            }
            _ => None,
        })
        .collect()
}

fn display_after_install(plugin_dir: &Path, identifier: &str) -> Result<(), Box<dyn Error>> {
    let after_install = plugin_dir.join("after-install.md");
    if after_install.exists() {
        println!();
        println!("{}", fs::read_to_string(after_install)?);
        println!();
    } else {
        println!();
        println!("Plugin installed: {identifier}");
        println!("Location: {}", plugin_dir.display());
        println!();
    }
    Ok(())
}

fn resolve_install_enable_choice(
    args: &InstallArgs,
    installed_name: &str,
) -> Result<bool, Box<dyn Error>> {
    if args.enable {
        return Ok(true);
    }
    if args.no_enable {
        return Ok(false);
    }
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Ok(false);
    }

    let prompt = format!("  Enable '{installed_name}' now? [y/N]: ");
    let Some(answer) = read_prompt(&prompt, false)? else {
        return Ok(false);
    };
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn read_prompt(prompt: &str, secret: bool) -> Result<Option<String>, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let echo_disabled = if secret && cfg!(unix) && io::stdin().is_terminal() {
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
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::process::Command;
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

    fn run_git(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
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

    #[test]
    fn install_plugin_clones_repo_and_enables_when_requested() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("demo-plugin");
        init_plugin_repo(
            &repo,
            "name: demo\nversion: 1.0.0\ndescription: demo plugin\n",
        );
        fs::write(repo.join("config.yaml.example"), "demo: true\n").unwrap();
        run_git(&repo, &["add", "."]);
        run_git(&repo, &["commit", "-m", "add example"]);

        let home = temp.path().join(".hermes");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        install_plugin(
            &context,
            InstallArgs {
                identifier: format!("file://{}", repo.display()),
                force: false,
                enable: true,
                no_enable: false,
            },
        )
        .unwrap();

        let plugin_dir = home.join("plugins").join("demo");
        assert!(plugin_dir.join("plugin.yaml").exists());
        assert!(plugin_dir.join("config.yaml").exists());

        let config = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config.contains("enabled:"));
        assert!(config.contains("- demo"));
    }

    #[test]
    fn update_plugin_pulls_latest_changes() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("demo-plugin");
        init_plugin_repo(
            &repo,
            "name: demo\nversion: 1.0.0\ndescription: demo plugin\n",
        );

        let home = temp.path().join(".hermes");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        install_plugin(
            &context,
            InstallArgs {
                identifier: format!("file://{}", repo.display()),
                force: false,
                enable: false,
                no_enable: true,
            },
        )
        .unwrap();

        fs::write(
            repo.join("plugin.yaml"),
            "name: demo\nversion: 2.0.0\ndescription: updated plugin\n",
        )
        .unwrap();
        fs::write(repo.join("new.env.example"), "DEMO=1\n").unwrap();
        run_git(&repo, &["add", "."]);
        run_git(&repo, &["commit", "-m", "update plugin"]);

        update_plugin(&context, "demo").unwrap();

        let plugin_dir = home.join("plugins").join("demo");
        let manifest = fs::read_to_string(plugin_dir.join("plugin.yaml")).unwrap();
        assert!(manifest.contains("version: 2.0.0"));
        assert!(plugin_dir.join("new.env").exists());
    }

    #[test]
    #[cfg(unix)]
    fn bridge_plugins_uses_python_override_for_bare_command() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  shift 2\n\
  printf 'action=%s argv=%s\\n' \"$HERMES_PLUGINS_ACTION\" \"$*\" >> '{}'\n\
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

        set_env_var("HERMES_PLUGINS_PYTHON", &fake_python);
        bridge_plugins(None, &[]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action= argv="));

        remove_env_var("HERMES_PLUGINS_PYTHON");
    }
}
