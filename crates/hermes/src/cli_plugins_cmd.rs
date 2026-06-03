//! `hermes plugins` CLI subcommand — install, update, remove, and list plugins.
//!
//! Native Rust port of `hermes_cli/plugins_cmd.py`.
//!
//! Plugins are installed from Git repositories into `~/.hermes/plugins/`.
//! Supports full URLs and `owner/repo` shorthand (resolves to GitHub).
//!
//! After install, if the plugin ships an `after-install.md` file it is
//! rendered to stdout. Otherwise a default confirmation is shown. The
//! interactive curses UI of the Python original is replaced by an
//! equivalent text-based prompt flow (the gateway/TUI surfaces remain).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Args, Subcommand};
use hermes_core::{HermesContext, is_effectively_enabled};
use serde_yaml::{Mapping, Value};
use tempfile::TempDir;

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::plugin_runtime::{
    PluginSource, discover_context_engines as discover_context_engine_plugins,
    discover_general_plugins, discover_memory_providers as discover_memory_provider_plugins,
};
use crate::python_bridge::project_root;

/// Minimum manifest version this installer understands.
///
/// Plugins may declare `manifest_version: 1` in plugin.yaml; future breaking
/// changes to the manifest schema bump this.
const SUPPORTED_MANIFEST_VERSION: i64 = 1;

// ---------------------------------------------------------------------------
// CLI argument shapes
// ---------------------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum PluginsCommand {
    /// Install a plugin from a Git URL or owner/repo shorthand.
    Install(InstallArgs),
    /// Update an installed plugin by pulling latest from its git remote.
    Update { name: String },
    /// Remove an installed plugin by name.
    #[command(aliases = ["rm", "uninstall"])]
    Remove { name: String },
    /// List all plugins (bundled + user) with enabled/disabled state.
    #[command(alias = "ls")]
    List,
    /// Add a plugin to the enabled allow-list.
    Enable { name: String },
    /// Remove a plugin from the enabled allow-list.
    Disable { name: String },
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

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A discovered plugin row used by `list` / interactive toggle / dashboard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginEntry {
    pub name: String,
    pub version: String,
    pub description: String,
    pub source: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
struct EnvSpec {
    name: String,
    description: String,
    url: String,
    secret: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProviderInfo {
    name: String,
    description: String,
}

/// `(name, description)` option presented in provider/engine pickers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderOption {
    pub name: String,
    pub description: String,
}

/// JSON-serializable-style result from a non-interactive dashboard install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashboardInstallResult {
    pub plugin_name: String,
    pub warnings: Vec<String>,
    pub missing_env: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashboardToggleResult {
    pub name: String,
    pub unchanged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashboardUpdateResult {
    pub name: String,
    pub output: String,
    pub unchanged: bool,
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch `hermes plugins` subcommands. `None` launches the interactive
/// composite toggle UI (text-based).
pub fn print_plugins(
    context: &HermesContext,
    command: Option<PluginsCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => toggle_plugins(context),
        Some(PluginsCommand::Install(args)) => install_plugin(context, args),
        Some(PluginsCommand::Update { name }) => update_plugin(context, &name),
        Some(PluginsCommand::List) => print_list(context),
        Some(PluginsCommand::Enable { name }) => enable_plugin(context, &name),
        Some(PluginsCommand::Disable { name }) => disable_plugin(context, &name),
        Some(PluginsCommand::Remove { name }) => remove_plugin(context, &name),
    }
}

// ---------------------------------------------------------------------------
// Dashboard (web) entry points
// ---------------------------------------------------------------------------

pub fn dashboard_list_plugins(context: &HermesContext) -> Result<Vec<PluginEntry>, Box<dyn Error>> {
    discover_all_plugins(context)
}

pub fn dashboard_plugin_sets(
    context: &HermesContext,
) -> Result<(BTreeSet<String>, BTreeSet<String>), Box<dyn Error>> {
    Ok((
        load_plugin_set(context, "enabled")?,
        load_plugin_set(context, "disabled")?,
    ))
}

pub fn dashboard_memory_provider_options(context: &HermesContext) -> Vec<ProviderOption> {
    discover_memory_providers(context)
        .into_iter()
        .map(|provider| ProviderOption {
            name: provider.name,
            description: provider.description,
        })
        .collect()
}

pub fn dashboard_context_engine_options(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    discover_context_engines(context)
}

pub fn dashboard_current_memory_provider(
    context: &HermesContext,
) -> Result<String, Box<dyn Error>> {
    current_memory_provider(context)
}

pub fn dashboard_current_context_engine(context: &HermesContext) -> Result<String, Box<dyn Error>> {
    current_context_engine(context)
}

pub fn dashboard_save_memory_provider(
    context: &HermesContext,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    save_memory_provider(context, name)
}

pub fn dashboard_save_context_engine(
    context: &HermesContext,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    save_context_engine(context, name)
}

/// Non-interactive install for the web dashboard.
pub fn dashboard_install_plugin(
    context: &HermesContext,
    identifier: &str,
    force: bool,
    enable: bool,
) -> Result<DashboardInstallResult, Box<dyn Error>> {
    let mut warnings = Vec::new();
    // Mirror the Python: resolve_git_url failures are swallowed here (the
    // warning gathering is best-effort); the real error surfaces from core.
    if let Ok(git_url) = resolve_git_url(identifier)
        && (git_url.starts_with("http://") || git_url.starts_with("file://"))
    {
        warnings.push(String::from(
            "Insecure URL scheme; prefer https:// or git@ for production installs.",
        ));
    }

    let (target, manifest, installed_name) = install_plugin_core(context, identifier, force)?;
    copy_example_files_silent(&target)?;
    let missing_env = missing_manifest_env_names(context, &manifest);
    if enable {
        let mut enabled_set = load_plugin_set(context, "enabled")?;
        let mut disabled_set = load_plugin_set(context, "disabled")?;
        enabled_set.insert(installed_name.clone());
        disabled_set.remove(&installed_name);
        save_plugin_set(context, "enabled", &enabled_set)?;
        save_plugin_set(context, "disabled", &disabled_set)?;
    }
    Ok(DashboardInstallResult {
        plugin_name: installed_name,
        warnings,
        missing_env,
        enabled: enable,
    })
}

/// Enable or disable a plugin in `config.yaml` (runtime allow/deny lists).
pub fn dashboard_set_agent_plugin_enabled(
    context: &HermesContext,
    raw_name: &str,
    enabled: bool,
) -> Result<DashboardToggleResult, Box<dyn Error>> {
    let name = resolve_existing_plugin_name(context, raw_name)?
        .ok_or_else(|| format!("plugin '{}' is not installed or bundled", raw_name.trim()))?;
    let mut enabled_set = load_plugin_set(context, "enabled")?;
    let mut disabled_set = load_plugin_set(context, "disabled")?;

    if enabled {
        if enabled_set.contains(&name) && !disabled_set.contains(&name) {
            return Ok(DashboardToggleResult {
                name,
                unchanged: true,
            });
        }
        enabled_set.insert(name.clone());
        disabled_set.remove(&name);
    } else {
        if !enabled_set.contains(&name) && disabled_set.contains(&name) {
            return Ok(DashboardToggleResult {
                name,
                unchanged: true,
            });
        }
        enabled_set.remove(&name);
        disabled_set.insert(name.clone());
    }

    save_plugin_set(context, "enabled", &enabled_set)?;
    save_plugin_set(context, "disabled", &disabled_set)?;
    Ok(DashboardToggleResult {
        name,
        unchanged: false,
    })
}

/// `git pull` inside `~/.hermes/plugins/<name>`.
pub fn dashboard_update_user_plugin(
    context: &HermesContext,
    raw_name: &str,
) -> Result<DashboardUpdateResult, Box<dyn Error>> {
    let plugins_dir = user_plugins_dir(context)?;
    let (name, target) = resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
        format!(
            "plugin '{}' not found in {}",
            raw_name.trim(),
            plugins_dir.display()
        )
    })?;
    if !target.join(".git").exists() {
        return Err(format!("plugin '{name}' is not a git checkout; cannot pull updates.").into());
    }
    let output = git_pull_plugin_dir(&target)?;
    copy_example_files_silent(&target)?;
    Ok(DashboardUpdateResult {
        unchanged: output.contains("Already up to date"),
        name,
        output,
    })
}

/// Delete a plugin tree under `~/.hermes/plugins/` only. Returns the
/// canonical plugin name removed.
pub fn dashboard_remove_user_plugin(
    context: &HermesContext,
    raw_name: &str,
) -> Result<String, Box<dyn Error>> {
    let plugins_dir = user_plugins_dir(context)?;

    // Refuse to remove anything that resolves to a bundled plugin.
    for entry in discover_all_plugins(context)? {
        if entry.name == raw_name.trim() && entry.source == "bundled" {
            return Err("Bundled plugins cannot be removed from the dashboard.".into());
        }
    }

    let (canonical_name, target) =
        resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
            format!(
                "plugin '{}' not found in {}",
                raw_name.trim(),
                plugins_dir.display()
            )
        })?;
    fs::remove_dir_all(&target)?;
    Ok(canonical_name)
}

// ---------------------------------------------------------------------------
// Interactive composite UI (text-based)
// ---------------------------------------------------------------------------

fn toggle_plugins(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let entries = discover_all_plugins(context)?;
    let enabled = load_plugin_set(context, "enabled")?;
    let disabled = load_plugin_set(context, "disabled")?;
    let memory_options = discover_memory_providers(context)
        .into_iter()
        .map(|provider| ProviderOption {
            name: provider.name,
            description: provider.description,
        })
        .collect::<Vec<_>>();
    let context_options = discover_context_engines(context)?;

    if entries.is_empty() && memory_options.is_empty() && context_options.is_empty() {
        println!("No plugins installed and no provider categories available.");
        println!("Install with: hermes plugins install owner/repo");
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        println!("Interactive mode requires a terminal.");
        return Ok(());
    }

    if !entries.is_empty() {
        configure_general_plugins(context, &entries, &enabled, &disabled)?;
    }
    if !memory_options.is_empty() {
        configure_memory_provider(context, &memory_options)?;
    }
    if !context_options.is_empty() || current_context_engine(context)? != "compressor" {
        configure_context_engine(context, &context_options)?;
    }

    println!("Changes take effect on next session.");
    Ok(())
}

// ---------------------------------------------------------------------------
// install / update
// ---------------------------------------------------------------------------

fn install_plugin(context: &HermesContext, args: InstallArgs) -> Result<(), Box<dyn Error>> {
    let identifier = args.identifier.trim();
    if identifier.is_empty() {
        return Err("plugin identifier cannot be empty".into());
    }

    let git_url = resolve_git_url(identifier)?;
    if git_url.starts_with("http://") || git_url.starts_with("file://") {
        println!(
            "Warning: Using insecure/local URL scheme. \
             Consider using https:// or git@ for production installs."
        );
    }
    println!("Cloning {git_url}...");

    let (target, manifest, installed_name) = install_plugin_core(context, identifier, args.force)?;

    if manifest_path(&target).is_none() && !target.join("__init__.py").exists() {
        println!(
            "Warning: {} doesn't contain plugin.yaml or __init__.py. \
             It may not be a valid Hermes plugin.",
            installed_name
        );
    }

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
        println!("Plugin {installed_name} enabled.");
    } else {
        println!(
            "Plugin installed but not enabled. \
             Run `hermes plugins enable {installed_name}` to activate."
        );
    }

    println!("Restart the gateway for the plugin to take effect:");
    println!("  hermes gateway restart");
    Ok(())
}

fn update_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let plugins_dir_display = user_plugins_dir(context)
        .ok()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| String::from("plugins"));
    let (name, target) = resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
        format!(
            "plugin '{}' not found in {}",
            raw_name.trim(),
            plugins_dir_display
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
        println!("Plugin {name} is already up to date.");
    } else {
        println!("Plugin {name} updated.");
        if !output.trim().is_empty() {
            println!("{}", output.trim());
        }
    }
    Ok(())
}

/// Clone Git plugin into `~/.hermes/plugins`.
///
/// Returns `(target_dir, installed_manifest, canonical_name)`.
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
                "plugin '{plugin_name}' already exists. Use force reinstall \
                 or run `hermes plugins update {plugin_name}`."
            )
            .into());
        }
        fs::remove_dir_all(&target)?;
    }

    rename_dir(&temp_target, &target)?;

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

/// Rename a directory, falling back to a recursive copy across filesystems
/// (the temp dir may be on a different device than `~/.hermes`).
fn rename_dir(from: &Path, to: &Path) -> Result<(), Box<dyn Error>> {
    if fs::rename(from, to).is_ok() {
        return Ok(());
    }
    copy_dir_recursive(from, to)?;
    let _ = fs::remove_dir_all(from);
    Ok(())
}

fn copy_dir_recursive(from: &Path, to: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if file_type.is_symlink() {
            #[cfg(unix)]
            {
                let target = fs::read_link(&src)?;
                std::os::unix::fs::symlink(target, &dst)?;
            }
            #[cfg(not(unix))]
            {
                fs::copy(&src, &dst)?;
            }
        } else {
            fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// URL / name helpers
// ---------------------------------------------------------------------------

/// Turn an identifier into a cloneable Git URL.
///
/// - Full URL: `https://...`, `http://...`, `git@...`, `ssh://...`, `file://...`
/// - Shorthand: `owner/repo` -> `https://github.com/owner/repo.git`
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
        format!("Invalid plugin identifier: '{trimmed}'. Use a Git URL or owner/repo shorthand.")
            .into(),
    )
}

/// Extract the repo name from a Git URL for the plugin directory name.
fn repo_name_from_url(url: &str) -> String {
    let mut trimmed = url.trim().trim_end_matches('/').to_string();
    if trimmed.ends_with(".git") {
        trimmed.truncate(trimmed.len() - 4);
    }
    // Last path component.
    let last = trimmed
        .rsplit('/')
        .next()
        .unwrap_or(trimmed.as_str())
        .to_string();
    // Handle ssh-style urls: git@github.com:owner/repo -> repo
    last.rsplit(':').next().unwrap_or(last.as_str()).to_string()
}

/// Validate a plugin name; reject path-traversal sequences.
fn validate_plugin_name(raw_name: &str) -> Result<String, Box<dyn Error>> {
    let name = raw_name.trim();
    if name.is_empty() {
        return Err("Plugin name must not be empty.".into());
    }
    if matches!(name, "." | "..") {
        return Err(format!(
            "Invalid plugin name '{name}': must not reference the plugins directory itself."
        )
        .into());
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!(
            "Invalid plugin name '{name}': must not contain path separators or traversal."
        )
        .into());
    }
    Ok(name.to_string())
}

// ---------------------------------------------------------------------------
// git subprocess wrappers
// ---------------------------------------------------------------------------

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
    let msg = if !stderr.is_empty() { stderr } else { stdout };
    Err(if msg.is_empty() {
        String::from("git pull failed.")
    } else {
        msg
    }
    .into())
}

// ---------------------------------------------------------------------------
// manifest reading
// ---------------------------------------------------------------------------

fn manifest_path(path: &Path) -> Option<PathBuf> {
    let yaml = path.join("plugin.yaml");
    if yaml.exists() {
        return Some(yaml);
    }
    let yml = path.join("plugin.yml");
    yml.exists().then_some(yml)
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
            format!(
                "Plugin '{plugin_name}' has invalid manifest_version '{number}' (expected an integer)."
            )
        })?,
        Value::String(text) => text.trim().parse::<i64>().map_err(|_| {
            format!(
                "Plugin '{plugin_name}' has invalid manifest_version '{}' (expected an integer).",
                text.trim()
            )
        })?,
        _ => {
            return Err(format!(
                "Plugin '{plugin_name}' has invalid manifest_version (expected an integer)."
            )
            .into());
        }
    };
    if version > SUPPORTED_MANIFEST_VERSION {
        return Err(format!(
            "Plugin '{plugin_name}' requires manifest_version {version}, but this installer only \
             supports up to {SUPPORTED_MANIFEST_VERSION}. Run `hermes update` to update Hermes."
        )
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// .example file copying
// ---------------------------------------------------------------------------

fn copy_example_files(plugin_dir: &Path) -> Result<(), Box<dyn Error>> {
    let created = copy_example_files_with_report(plugin_dir)?;
    for real_name in created {
        println!("  Created {real_name} from {real_name}.example");
    }
    Ok(())
}

fn copy_example_files_silent(plugin_dir: &Path) -> Result<(), Box<dyn Error>> {
    let _ = copy_example_files_with_report(plugin_dir)?;
    Ok(())
}

/// Copy any `*.example` files to their real names if they don't already exist.
/// e.g. `config.yaml.example` -> `config.yaml`. Returns the created names.
fn copy_example_files_with_report(plugin_dir: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut created = Vec::new();
    let read = match fs::read_dir(plugin_dir) {
        Ok(read) => read,
        Err(_) => return Ok(created),
    };
    for entry in read {
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
        if real_name.is_empty() {
            continue;
        }
        let target = plugin_dir.join(real_name);
        if target.exists() {
            continue;
        }
        // Best-effort, matching the Python warning-on-failure behaviour.
        match fs::copy(&path, &target) {
            Ok(_) => created.push(real_name.to_string()),
            Err(error) => {
                println!("Warning: Failed to copy {file_name}: {error}");
            }
        }
    }
    Ok(created)
}

// ---------------------------------------------------------------------------
// env-var prompting
// ---------------------------------------------------------------------------

/// `requires_env` accepts either bare strings or `{name, description, url, secret}`.
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
        let value = read_prompt(prompt.as_str(), spec.secret)?;
        let Some(value) = value else {
            // EOF / interrupt: abort prompting entirely.
            println!(
                "\n  Skipped (you can set these later in {}/.env)",
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

/// Return declared `requires_env` names that are unset both in the live
/// environment and in `~/.hermes/.env`.
fn missing_manifest_env_names(context: &HermesContext, manifest: &Mapping) -> Vec<String> {
    let file_env = load_env_file_values(&context.env_path());
    parse_manifest_env_specs(manifest)
        .into_iter()
        .filter(|spec| {
            std::env::var(&spec.name)
                .ok()
                .is_none_or(|value| value.trim().is_empty())
                && file_env
                    .get(&spec.name)
                    .is_none_or(|value| value.trim().is_empty())
        })
        .map(|spec| spec.name)
        .collect()
}

fn load_env_file_values(path: &Path) -> BTreeMap<String, String> {
    let Ok(raw) = fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    raw.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                return None;
            }
            let (key, value) = trimmed.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// install presentation
// ---------------------------------------------------------------------------

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

/// Map the argparse tri-state: `--enable` -> true, `--no-enable` -> false,
/// neither -> prompt (when interactive) / false (non-interactive).
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

/// Read a single line from stdin after printing `prompt`. Returns `None` on
/// EOF (matching the Python `EOFError`/`KeyboardInterrupt` skip behaviour).
/// When `secret` is set and on a Unix tty, echo is disabled via `stty`.
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

// ---------------------------------------------------------------------------
// interactive configuration sub-flows
// ---------------------------------------------------------------------------

fn configure_general_plugins(
    context: &HermesContext,
    entries: &[PluginEntry],
    enabled: &BTreeSet<String>,
    disabled: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let plugins = discover_general_plugins(context)?;
    println!("\nGeneral Plugins");
    for (index, entry) in entries.iter().enumerate() {
        let plugin = plugins
            .iter()
            .find(|plugin| plugin.name == entry.name)
            .ok_or_else(|| format!("plugin '{}' disappeared during configuration", entry.name))?;
        let marker = if is_effectively_enabled(plugin, enabled, disabled) {
            "x"
        } else {
            " "
        };
        let mut label = entry.name.clone();
        if !entry.description.trim().is_empty() {
            label.push_str(" — ");
            label.push_str(entry.description.trim());
        }
        if plugin.kind.is_auto_enabled(plugin.source) {
            label.push_str(" [auto]");
        } else if entry.source == "bundled" {
            label.push_str(" [bundled]");
        }
        println!("  {:>2}. [{}] {}", index + 1, marker, label);
    }

    let Some(input) = read_prompt(
        "\nToggle plugin numbers separated by spaces or commas (Enter to keep): ",
        false,
    )?
    else {
        return Ok(());
    };

    let mut chosen = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let plugin = plugins.iter().find(|plugin| plugin.name == entry.name)?;
            is_effectively_enabled(plugin, enabled, disabled).then_some(index)
        })
        .collect::<BTreeSet<_>>();

    for token in input.split(|ch: char| ch.is_ascii_whitespace() || ch == ',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let index = trimmed
            .parse::<usize>()
            .map_err(|_| format!("invalid plugin selection: {trimmed}"))?;
        if index == 0 || index > entries.len() {
            return Err(format!("plugin selection {index} is out of range").into());
        }
        let zero_based = index - 1;
        if !chosen.insert(zero_based) {
            chosen.remove(&zero_based);
        }
    }

    // The new allow-list is the set of plugin names that were checked; anything
    // not checked is explicitly disabled so it remains off.
    let mut new_enabled = BTreeSet::new();
    let mut new_disabled = disabled.clone();
    for (index, entry) in entries.iter().enumerate() {
        let plugin = plugins
            .iter()
            .find(|plugin| plugin.name == entry.name)
            .ok_or_else(|| format!("plugin '{}' disappeared during configuration", entry.name))?;
        if chosen.contains(&index) {
            if !plugin.kind.is_auto_enabled(plugin.source) {
                new_enabled.insert(entry.name.clone());
            }
            new_disabled.remove(&entry.name);
        } else {
            new_enabled.remove(&entry.name);
            new_disabled.insert(entry.name.clone());
        }
    }

    if &new_enabled != enabled || &new_disabled != disabled {
        save_plugin_set(context, "enabled", &new_enabled)?;
        save_plugin_set(context, "disabled", &new_disabled)?;
        println!(
            "General plugins: {} enabled, {} disabled.",
            new_enabled.len(),
            entries.len().saturating_sub(new_enabled.len())
        );
    } else {
        println!("General plugins unchanged.");
    }

    Ok(())
}

fn configure_memory_provider(
    context: &HermesContext,
    providers: &[ProviderOption],
) -> Result<(), Box<dyn Error>> {
    let current = current_memory_provider(context)?;
    println!("\nMemory Provider");
    println!(
        "   0. built-in (default){}",
        if current.is_empty() { " [current]" } else { "" }
    );
    for (index, provider) in providers.iter().enumerate() {
        let current_marker = if provider.name == current {
            " [current]"
        } else {
            ""
        };
        if provider.description.trim().is_empty() {
            println!("  {:>2}. {}{}", index + 1, provider.name, current_marker);
        } else {
            println!(
                "  {:>2}. {} — {}{}",
                index + 1,
                provider.name,
                provider.description.trim(),
                current_marker
            );
        }
    }

    let Some(input) = read_prompt("Select memory provider (Enter to keep): ", false)? else {
        return Ok(());
    };
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let selected = if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "0" | "built-in" | "builtin" | "none" | "default"
    ) {
        String::new()
    } else if let Ok(index) = trimmed.parse::<usize>() {
        let provider = providers
            .get(
                index
                    .checked_sub(1)
                    .ok_or("memory provider selection must be >= 1")?,
            )
            .ok_or_else(|| format!("memory provider selection {index} is out of range"))?;
        provider.name.clone()
    } else {
        providers
            .iter()
            .find(|provider| provider.name.eq_ignore_ascii_case(trimmed))
            .map(|provider| provider.name.clone())
            .ok_or_else(|| format!("unknown memory provider: {trimmed}"))?
    };

    if selected != current {
        save_memory_provider(context, &selected)?;
        println!(
            "Memory provider set to {}.",
            if selected.is_empty() {
                "built-in"
            } else {
                selected.as_str()
            }
        );
    }
    Ok(())
}

fn configure_context_engine(
    context: &HermesContext,
    engines: &[ProviderOption],
) -> Result<(), Box<dyn Error>> {
    let current = current_context_engine(context)?;
    println!("\nContext Engine");
    println!(
        "   0. compressor (default){}",
        if current == "compressor" {
            " [current]"
        } else {
            ""
        }
    );
    for (index, engine) in engines.iter().enumerate() {
        let current_marker = if engine.name == current {
            " [current]"
        } else {
            ""
        };
        if engine.description.trim().is_empty() {
            println!("  {:>2}. {}{}", index + 1, engine.name, current_marker);
        } else {
            println!(
                "  {:>2}. {} — {}{}",
                index + 1,
                engine.name,
                engine.description.trim(),
                current_marker
            );
        }
    }

    let Some(input) = read_prompt("Select context engine (Enter to keep): ", false)? else {
        return Ok(());
    };
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let selected = if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "0" | "compressor" | "default"
    ) {
        String::from("compressor")
    } else if let Ok(index) = trimmed.parse::<usize>() {
        let engine = engines
            .get(
                index
                    .checked_sub(1)
                    .ok_or("context engine selection must be >= 1")?,
            )
            .ok_or_else(|| format!("context engine selection {index} is out of range"))?;
        engine.name.clone()
    } else {
        engines
            .iter()
            .find(|engine| engine.name.eq_ignore_ascii_case(trimmed))
            .map(|engine| engine.name.clone())
            .ok_or_else(|| format!("unknown context engine: {trimmed}"))?
    };

    if selected != current {
        save_context_engine(context, &selected)?;
        println!("Context engine set to {selected}.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

fn print_list(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let plugins = discover_general_plugins(context)?;
    if plugins.is_empty() {
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
    for plugin in plugins {
        let source = if plugin.source == PluginSource::User && plugin.path.join(".git").exists() {
            "git"
        } else {
            plugin.source.as_str()
        };
        let status = if disabled.contains(&plugin.name) {
            "disabled"
        } else if plugin.kind.is_auto_enabled(plugin.source) {
            "auto"
        } else if enabled.contains(&plugin.name) {
            "enabled"
        } else {
            "not enabled"
        };
        println!(
            "{:<24} {:<12} {:<12} {:<10} {}",
            plugin.name, status, plugin.version, source, plugin.description
        );
    }
    println!();
    println!("Interactive toggle: hermes plugins");
    println!("Enable/disable:     hermes plugins enable|disable <name>");
    println!("Plugins are opt-in by default — only 'enabled' plugins load.");
    Ok(())
}

// ---------------------------------------------------------------------------
// enable / disable / remove
// ---------------------------------------------------------------------------

fn enable_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let name = resolve_existing_plugin_name(context, raw_name)?
        .ok_or_else(|| format!("Plugin '{}' is not installed or bundled.", raw_name.trim()))?;
    let mut enabled = load_plugin_set(context, "enabled")?;
    let mut disabled = load_plugin_set(context, "disabled")?;
    let plugins = discover_general_plugins(context)?;
    let plugin = plugins
        .iter()
        .find(|plugin| plugin.name == name)
        .ok_or_else(|| format!("Plugin '{}' is not installed or bundled.", raw_name.trim()))?;

    if is_effectively_enabled(plugin, &enabled, &disabled) {
        println!("Plugin '{name}' is already enabled.");
        return Ok(());
    }

    if plugin.kind.is_auto_enabled(plugin.source) {
        enabled.remove(&name);
    } else {
        enabled.insert(name.clone());
    }
    disabled.remove(&name);
    save_plugin_set(context, "enabled", &enabled)?;
    save_plugin_set(context, "disabled", &disabled)?;
    println!("Plugin {name} enabled. Takes effect on next session.");
    Ok(())
}

fn disable_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let name = resolve_existing_plugin_name(context, raw_name)?
        .ok_or_else(|| format!("Plugin '{}' is not installed or bundled.", raw_name.trim()))?;
    let mut enabled = load_plugin_set(context, "enabled")?;
    let mut disabled = load_plugin_set(context, "disabled")?;
    let plugins = discover_general_plugins(context)?;
    let plugin = plugins
        .iter()
        .find(|plugin| plugin.name == name)
        .ok_or_else(|| format!("Plugin '{}' is not installed or bundled.", raw_name.trim()))?;

    if !is_effectively_enabled(plugin, &enabled, &disabled) && disabled.contains(&name) {
        println!("Plugin '{name}' is already disabled.");
        return Ok(());
    }

    enabled.remove(&name);
    disabled.insert(name.clone());
    save_plugin_set(context, "enabled", &enabled)?;
    save_plugin_set(context, "disabled", &disabled)?;
    println!("Plugin {name} disabled. Takes effect on next session.");
    Ok(())
}

fn remove_plugin(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let plugins_dir = user_plugins_dir(context)?;
    let (canonical_name, target) =
        resolve_installed_plugin(context, raw_name)?.ok_or_else(|| {
            let installed = installed_dir_names(&plugins_dir);
            format!(
                "Plugin '{}' not found in {}.\nInstalled plugins: {}",
                raw_name.trim(),
                plugins_dir.display(),
                if installed.is_empty() {
                    String::from("(none)")
                } else {
                    installed.join(", ")
                }
            )
        })?;
    fs::remove_dir_all(&target)?;
    println!(
        "Plugin {canonical_name} removed from {}",
        plugins_dir.display()
    );
    Ok(())
}

fn installed_dir_names(plugins_dir: &Path) -> Vec<String> {
    let Ok(read) = fs::read_dir(plugins_dir) else {
        return Vec::new();
    };
    let mut names = read
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().to_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// discovery
// ---------------------------------------------------------------------------

fn discover_all_plugins(context: &HermesContext) -> Result<Vec<PluginEntry>, Box<dyn Error>> {
    let mut entries = discover_general_plugins(context)?
        .into_iter()
        .map(|plugin| PluginEntry {
            name: plugin.name,
            version: plugin.version,
            description: plugin.description,
            source: if plugin.source == PluginSource::User && plugin.path.join(".git").exists() {
                String::from("git")
            } else {
                plugin.source.as_str().to_string()
            },
            path: plugin.path,
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(entries)
}

fn plugin_entry_from_dir(path: &Path, source: &str) -> Result<Option<PluginEntry>, Box<dyn Error>> {
    let Some(manifest_path) = manifest_path(path) else {
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

/// Bundled plugins dir: `$HERMES_BUNDLED_PLUGINS` (Nix) or `<repo>/plugins`.
pub fn bundled_plugins_dir() -> PathBuf {
    if let Some(value) = std::env::var_os("HERMES_BUNDLED_PLUGINS") {
        let trimmed = value.to_string_lossy().trim().to_string();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    project_root().join("plugins")
}

/// The user plugins directory, creating it if needed.
pub fn user_plugins_dir(context: &HermesContext) -> Result<PathBuf, Box<dyn Error>> {
    let path = context.hermes_home().join("plugins");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn discover_memory_providers(context: &HermesContext) -> Vec<ProviderInfo> {
    discover_memory_provider_plugins(context)
        .unwrap_or_default()
        .into_iter()
        .map(|provider| ProviderInfo {
            name: provider.name,
            description: provider.description,
        })
        .collect()
}

fn discover_context_engines(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    Ok(discover_context_engine_plugins(context)?
        .into_iter()
        .map(|provider| ProviderOption {
            name: provider.name,
            description: provider.description,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// config read/write — memory provider & context engine
// ---------------------------------------------------------------------------

fn current_memory_provider(context: &HermesContext) -> Result<String, Box<dyn Error>> {
    let raw = read_raw_yaml_mapping(&context.config_path())?;
    Ok(raw
        .get(Value::String(String::from("memory")))
        .and_then(Value::as_mapping)
        .and_then(|mapping| mapping.get(Value::String(String::from("provider"))))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string())
}

fn save_memory_provider(context: &HermesContext, name: &str) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let key = Value::String(String::from("memory"));
    let memory = if let Some(Value::Mapping(mapping)) = root.get_mut(&key) {
        mapping
    } else {
        root.insert(key.clone(), Value::Mapping(Mapping::new()));
        root.get_mut(&key)
            .and_then(Value::as_mapping_mut)
            .ok_or("failed to initialize memory config mapping")?
    };
    memory.insert(
        Value::String(String::from("provider")),
        Value::String(name.trim().to_string()),
    );
    write_yaml_mapping(&context.config_path(), &root)
}

fn current_context_engine(context: &HermesContext) -> Result<String, Box<dyn Error>> {
    let raw = read_raw_yaml_mapping(&context.config_path())?;
    Ok(raw
        .get(Value::String(String::from("context")))
        .and_then(Value::as_mapping)
        .and_then(|mapping| mapping.get(Value::String(String::from("engine"))))
        .and_then(Value::as_str)
        .unwrap_or("compressor")
        .trim()
        .to_string())
}

fn save_context_engine(context: &HermesContext, name: &str) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let key = Value::String(String::from("context"));
    let context_mapping = if let Some(Value::Mapping(mapping)) = root.get_mut(&key) {
        mapping
    } else {
        root.insert(key.clone(), Value::Mapping(Mapping::new()));
        root.get_mut(&key)
            .and_then(Value::as_mapping_mut)
            .ok_or("failed to initialize context config mapping")?
    };
    context_mapping.insert(
        Value::String(String::from("engine")),
        Value::String(name.trim().to_string()),
    );
    write_yaml_mapping(&context.config_path(), &root)
}

// ---------------------------------------------------------------------------
// resolution helpers
// ---------------------------------------------------------------------------

fn resolve_existing_plugin_name(
    context: &HermesContext,
    raw_name: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let requested = validate_plugin_name(raw_name)?;
    Ok(discover_general_plugins(context)?
        .into_iter()
        .find(|plugin| plugin.name == requested)
        .map(|plugin| plugin.name))
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

// ---------------------------------------------------------------------------
// enabled/disabled config lists
// ---------------------------------------------------------------------------

pub fn load_plugin_set(
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
    // Sorted output (BTreeSet iterates sorted) to match Python `sorted(...)`.
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

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn test_env_lock() -> &'static Mutex<()> {
        crate::cli_test_env_lock()
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
        assert!(status.success(), "git {args:?} failed");
    }

    fn init_plugin_repo(repo: &Path, manifest: &str) {
        fs::create_dir_all(repo).unwrap();
        run_git(repo, &["init"]);
        fs::write(repo.join("plugin.yaml"), manifest).unwrap();
        run_git(repo, &["add", "."]);
        run_git(repo, &["commit", "-m", "init"]);
    }

    #[test]
    fn resolve_git_url_handles_shorthand_and_full_urls() {
        assert_eq!(
            resolve_git_url("owner/repo").unwrap(),
            "https://github.com/owner/repo.git"
        );
        assert_eq!(
            resolve_git_url("https://example.com/x/y.git").unwrap(),
            "https://example.com/x/y.git"
        );
        assert_eq!(
            resolve_git_url("git@github.com:owner/repo.git").unwrap(),
            "git@github.com:owner/repo.git"
        );
        assert!(resolve_git_url("not-valid").is_err());
        assert!(resolve_git_url("a/b/c").is_err());
    }

    #[test]
    fn repo_name_from_url_strips_git_suffix_and_path() {
        assert_eq!(
            repo_name_from_url("https://github.com/owner/repo.git"),
            "repo"
        );
        assert_eq!(repo_name_from_url("https://github.com/owner/repo/"), "repo");
        assert_eq!(repo_name_from_url("git@github.com:owner/repo.git"), "repo");
    }

    #[test]
    fn validate_plugin_name_rejects_traversal() {
        assert!(validate_plugin_name("../bad").is_err());
        assert!(validate_plugin_name("bad/name").is_err());
        assert!(validate_plugin_name("bad\\name").is_err());
        assert!(validate_plugin_name("").is_err());
        assert!(validate_plugin_name(".").is_err());
        assert!(validate_plugin_name("..").is_err());
        assert_eq!(validate_plugin_name(" good ").unwrap(), "good");
    }

    #[test]
    fn parse_manifest_env_specs_handles_both_formats() {
        let manifest: Mapping = serde_yaml::from_str(
            "requires_env:\n  - SIMPLE_KEY\n  - name: RICH_KEY\n    description: A key\n    url: https://x\n    secret: true\n",
        )
        .unwrap();
        let specs = parse_manifest_env_specs(&manifest);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "SIMPLE_KEY");
        assert!(!specs[0].secret);
        assert_eq!(specs[1].name, "RICH_KEY");
        assert_eq!(specs[1].description, "A key");
        assert_eq!(specs[1].url, "https://x");
        assert!(specs[1].secret);
    }

    #[test]
    fn validate_manifest_version_rejects_future_versions() {
        let manifest: Mapping = serde_yaml::from_str("manifest_version: 99\n").unwrap();
        assert!(validate_manifest_version(&manifest, "demo").is_err());

        let ok: Mapping = serde_yaml::from_str("manifest_version: 1\n").unwrap();
        assert!(validate_manifest_version(&ok, "demo").is_ok());

        let bad: Mapping = serde_yaml::from_str("manifest_version: abc\n").unwrap();
        assert!(validate_manifest_version(&bad, "demo").is_err());

        let none: Mapping = serde_yaml::from_str("name: demo\n").unwrap();
        assert!(validate_manifest_version(&none, "demo").is_ok());
    }

    #[test]
    fn copy_example_files_creates_missing_only() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path();
        fs::write(dir.join("config.yaml.example"), "a: 1\n").unwrap();
        fs::write(dir.join("secret.env.example"), "K=v\n").unwrap();
        // Pre-existing real file should not be overwritten.
        fs::write(dir.join("secret.env"), "K=existing\n").unwrap();

        let created = copy_example_files_with_report(dir).unwrap();
        assert!(created.contains(&String::from("config.yaml")));
        assert!(!created.contains(&String::from("secret.env")));
        assert_eq!(
            fs::read_to_string(dir.join("config.yaml")).unwrap(),
            "a: 1\n"
        );
        assert_eq!(
            fs::read_to_string(dir.join("secret.env")).unwrap(),
            "K=existing\n"
        );
    }

    #[test]
    fn discover_plugins_prefers_user_over_bundled() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        fs::create_dir_all(bundled.join("demo")).unwrap();
        fs::write(
            bundled.join("demo").join("plugin.yaml"),
            "name: demo\nversion: 1.0.0\ndescription: bundled\n",
        )
        .unwrap();

        let old_override = std::env::var_os("HERMES_BUNDLED_PLUGINS");
        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().join(".hermes")));
        let user_plugins = context.hermes_home().join("plugins");
        fs::create_dir_all(user_plugins.join("demo")).unwrap();
        fs::write(
            user_plugins.join("demo").join("plugin.yaml"),
            "name: demo\nversion: 2.0.0\ndescription: user\n",
        )
        .unwrap();
        fs::write(user_plugins.join("demo").join(".git"), "").unwrap();
        fs::create_dir_all(user_plugins.join("other")).unwrap();
        fs::write(
            user_plugins.join("other").join("plugin.yml"),
            "name: other\ndescription: user other\n",
        )
        .unwrap();

        let entries = discover_all_plugins(&context).unwrap();
        let demo = entries.iter().find(|entry| entry.name == "demo").unwrap();
        assert_eq!(demo.version, "2.0.0");
        assert_eq!(demo.source, "git");
        assert!(entries.iter().any(|entry| entry.name == "other"));

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
    fn install_plugin_refuses_existing_without_force() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("demo-plugin");
        init_plugin_repo(&repo, "name: demo\nversion: 1.0.0\n");

        let home = temp.path().join(".hermes");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let args = InstallArgs {
            identifier: format!("file://{}", repo.display()),
            force: false,
            enable: false,
            no_enable: true,
        };
        install_plugin(&context, args.clone()).unwrap();
        let err = install_plugin(&context, args).unwrap_err();
        assert!(err.to_string().contains("already exists"));
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
    fn save_memory_and_context_provider_updates_config() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));

        save_memory_provider(&context, "mem0").unwrap();
        save_context_engine(&context, "compressor").unwrap();

        assert_eq!(current_memory_provider(&context).unwrap(), "mem0");
        assert_eq!(current_context_engine(&context).unwrap(), "compressor");
    }

    #[test]
    fn missing_manifest_env_names_reports_unset_only() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        fs::write(home.join(".env"), "SET_IN_FILE=value\n").unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));

        let manifest: Mapping = serde_yaml::from_str(
            "requires_env:\n  - SET_IN_FILE\n  - SET_IN_ENV\n  - TOTALLY_MISSING\n",
        )
        .unwrap();

        unsafe { std::env::set_var("SET_IN_ENV", "x") };
        let missing = missing_manifest_env_names(&context, &manifest);
        unsafe { std::env::remove_var("SET_IN_ENV") };

        assert!(missing.contains(&String::from("TOTALLY_MISSING")));
        assert!(!missing.contains(&String::from("SET_IN_FILE")));
        assert!(!missing.contains(&String::from("SET_IN_ENV")));
    }

    #[test]
    fn dashboard_remove_blocks_bundled() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        fs::create_dir_all(bundled.join("demo")).unwrap();
        fs::write(bundled.join("demo").join("plugin.yaml"), "name: demo\n").unwrap();

        let old_override = std::env::var_os("HERMES_BUNDLED_PLUGINS");
        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().join(".hermes")));
        fs::create_dir_all(context.hermes_home().join("plugins")).unwrap();

        let err = dashboard_remove_user_plugin(&context, "demo").unwrap_err();
        assert!(
            err.to_string()
                .contains("Bundled plugins cannot be removed")
        );

        match old_override {
            Some(value) => unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", value) },
            None => unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") },
        }
    }

    #[test]
    fn load_plugin_set_filters_blanks() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - alpha\n    - \"\"\n    - beta\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let set = load_plugin_set(&context, "enabled").unwrap();
        assert!(set.contains("alpha"));
        assert!(set.contains("beta"));
        assert_eq!(set.len(), 2);
    }
}
