use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use hermes_core::{
    HermesContext, ToolDefinition, ToolRuntime, discover_hook_registrations_from_source,
    discover_tool_definitions_from_source,
};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

use crate::plugins_cmd::{bundled_plugins_dir, load_plugin_set, user_plugins_dir};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PluginSource {
    Bundled,
    User,
    Project,
}

impl PluginSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bundled => "bundled",
            Self::User => "user",
            Self::Project => "project",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PluginKind {
    Standalone,
    Backend,
    Exclusive,
    Platform,
    ModelProvider,
    ContextEngine,
}

impl PluginKind {
    pub(crate) fn is_auto_enabled(self, source: PluginSource) -> bool {
        source == PluginSource::Bundled && matches!(self, Self::Backend | Self::Platform)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PluginCliCommand {
    pub(crate) name: String,
    pub(crate) help: String,
    pub(crate) description: String,
    pub(crate) plugin_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlatformSurface {
    pub(crate) key: String,
    pub(crate) label: String,
    pub(crate) required_env: Vec<String>,
    pub(crate) install_hint: Option<String>,
    pub(crate) emoji: String,
    pub(crate) has_setup_fn: bool,
    pub(crate) plugin_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DashboardSurface {
    pub(crate) name: String,
    pub(crate) label: String,
    pub(crate) description: String,
    pub(crate) icon: String,
    pub(crate) version: String,
    pub(crate) entry: String,
    pub(crate) css: Option<String>,
    pub(crate) api: Option<String>,
    pub(crate) slots: Vec<String>,
    pub(crate) tab_path: String,
    pub(crate) tab_position: String,
    pub(crate) tab_override: Option<String>,
    pub(crate) tab_hidden: bool,
    pub(crate) source: PluginSource,
    pub(crate) dashboard_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderOption {
    pub(crate) name: String,
    pub(crate) description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiscoveredPlugin {
    pub(crate) key: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) description: String,
    pub(crate) author: String,
    pub(crate) kind: PluginKind,
    pub(crate) source: PluginSource,
    pub(crate) path: PathBuf,
    pub(crate) requires_env: Vec<String>,
    pub(crate) provides_tools: Vec<String>,
    pub(crate) provides_hooks: Vec<String>,
    pub(crate) tool_definitions: Vec<ToolDefinition>,
    pub(crate) cli_commands: Vec<PluginCliCommand>,
    pub(crate) platforms: Vec<PlatformSurface>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct PluginCatalog {
    pub(crate) plugins: Vec<DiscoveredPlugin>,
    pub(crate) dashboards: Vec<DashboardSurface>,
    pub(crate) memory_providers: Vec<ProviderOption>,
    pub(crate) context_engines: Vec<ProviderOption>,
}

#[derive(Debug, Default, Deserialize)]
struct DashboardManifest {
    name: Option<String>,
    label: Option<String>,
    description: Option<String>,
    icon: Option<String>,
    version: Option<String>,
    entry: Option<String>,
    css: Option<String>,
    api: Option<String>,
    slots: Option<Vec<String>>,
    tab: Option<DashboardTab>,
}

#[derive(Debug, Default, Deserialize)]
struct DashboardTab {
    path: Option<String>,
    position: Option<String>,
    override_path: Option<String>,
    hidden: Option<bool>,
}

pub(crate) fn attach_python_plugin_runtime(
    context: &HermesContext,
    runtime: ToolRuntime,
) -> Result<ToolRuntime, Box<dyn Error>> {
    let enabled = load_plugin_set(context, "enabled")?;
    let disabled = load_plugin_set(context, "disabled")?;
    let has_enabled_general_plugins = discover_general_plugins(context)?
        .into_iter()
        .any(|plugin| is_effectively_enabled(&plugin, &enabled, &disabled));
    if !has_enabled_general_plugins {
        return Ok(runtime);
    }
    hermes_core::attach_python_plugin_runtime(&context.hermes_home(), runtime)
        .map_err(|error| -> Box<dyn Error> { error.into() })
}

pub(crate) fn discover_plugin_catalog(
    context: &HermesContext,
) -> Result<PluginCatalog, Box<dyn Error>> {
    let bundled = bundled_plugins_dir();
    let user = user_plugins_dir(context)?;
    let project = project_plugins_dir();

    let mut winners = BTreeMap::<String, DiscoveredPlugin>::new();
    for (root, source) in [
        (bundled.clone(), PluginSource::Bundled),
        (user.clone(), PluginSource::User),
        (project.clone(), PluginSource::Project),
    ] {
        if !root.is_dir() {
            continue;
        }
        for plugin in scan_plugin_tree(&root, source)? {
            winners.insert(plugin.key.clone(), plugin);
        }
    }

    let mut dashboards = BTreeMap::<String, DashboardSurface>::new();
    for (root, source) in [
        (user.clone(), PluginSource::User),
        (bundled.clone(), PluginSource::Bundled),
        (project.clone(), PluginSource::Project),
    ] {
        if !root.is_dir() {
            continue;
        }
        for dashboard in scan_dashboard_tree(&root, source)? {
            dashboards
                .entry(dashboard.name.clone())
                .or_insert(dashboard);
        }
    }

    let mut memory = BTreeMap::<String, ProviderOption>::new();
    let mut context_engines = BTreeMap::<String, ProviderOption>::new();
    for plugin in discover_memory_provider_plugins(context)? {
        memory.entry(plugin.name.clone()).or_insert(ProviderOption {
            name: plugin.name,
            description: plugin.description,
        });
    }
    for plugin in winners.values() {
        if plugin.kind == PluginKind::ContextEngine {
            context_engines
                .entry(plugin.name.clone())
                .or_insert(ProviderOption {
                    name: plugin.name.clone(),
                    description: plugin.description.clone(),
                });
        }
    }

    let mut plugins = winners.into_values().collect::<Vec<_>>();
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));

    Ok(PluginCatalog {
        plugins,
        dashboards: dashboards.into_values().collect(),
        memory_providers: memory.into_values().collect(),
        context_engines: context_engines.into_values().collect(),
    })
}

pub(crate) fn discover_general_plugins(
    context: &HermesContext,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    let mut plugins = discover_plugin_catalog(context)?
        .plugins
        .into_iter()
        .filter(|plugin| {
            !matches!(
                plugin.kind,
                PluginKind::Exclusive | PluginKind::ModelProvider | PluginKind::ContextEngine
            )
        })
        .collect::<Vec<_>>();
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));
    Ok(plugins)
}

pub(crate) fn discover_enabled_platform_plugins(
    context: &HermesContext,
) -> Result<Vec<PlatformSurface>, Box<dyn Error>> {
    let catalog = discover_plugin_catalog(context)?;
    let enabled = load_plugin_set(context, "enabled")?;
    let disabled = load_plugin_set(context, "disabled")?;
    let mut surfaces = Vec::new();
    for plugin in catalog.plugins {
        if plugin.platforms.is_empty() {
            continue;
        }
        if !is_effectively_enabled(&plugin, &enabled, &disabled) {
            continue;
        }
        surfaces.extend(plugin.platforms);
    }
    surfaces.sort_by(|left, right| left.key.cmp(&right.key));
    Ok(surfaces)
}

pub(crate) fn discover_enabled_cli_commands(
    context: &HermesContext,
) -> Result<Vec<PluginCliCommand>, Box<dyn Error>> {
    let catalog = discover_plugin_catalog(context)?;
    let enabled = load_plugin_set(context, "enabled")?;
    let disabled = load_plugin_set(context, "disabled")?;
    let mut seen = BTreeSet::new();
    let mut commands = Vec::new();

    for plugin in catalog.plugins {
        if !is_effectively_enabled(&plugin, &enabled, &disabled) {
            continue;
        }
        for command in plugin.cli_commands {
            if seen.insert(command.name.clone()) {
                commands.push(command);
            }
        }
    }

    if let Some(memory_command) = discover_active_memory_cli(context)? {
        if seen.insert(memory_command.name.clone()) {
            commands.push(memory_command);
        }
    }

    commands.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(commands)
}

pub(crate) fn discover_memory_providers(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    let mut providers = discover_memory_provider_plugins(context)?
        .into_iter()
        .map(|provider| ProviderOption {
            name: provider.name,
            description: provider.description,
        })
        .collect::<Vec<_>>();
    providers.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(providers)
}

pub(crate) fn discover_memory_provider_plugins(
    context: &HermesContext,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    let mut plugins = Vec::new();
    let mut seen = BTreeSet::new();
    for (root, source, prefix, require_marker) in [
        (
            bundled_plugins_dir().join("memory"),
            PluginSource::Bundled,
            "memory",
            false,
        ),
        (user_plugins_dir(context)?, PluginSource::User, "", true),
    ] {
        for plugin in scan_memory_provider_root(&root, source, prefix, require_marker)? {
            if seen.insert(plugin.name.clone()) {
                plugins.push(plugin);
            }
        }
    }
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));
    Ok(plugins)
}

pub(crate) fn find_memory_provider_plugin(
    context: &HermesContext,
    provider_name: &str,
) -> Result<Option<DiscoveredPlugin>, Box<dyn Error>> {
    let name = provider_name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    Ok(discover_memory_provider_plugins(context)?
        .into_iter()
        .find(|plugin| plugin.name == name))
}

pub(crate) fn discover_context_engines(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    let mut providers = discover_plugin_catalog(context)?.context_engines;
    providers.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(providers)
}

pub(crate) fn is_effectively_enabled(
    plugin: &DiscoveredPlugin,
    enabled: &BTreeSet<String>,
    disabled: &BTreeSet<String>,
) -> bool {
    if disabled.contains(&plugin.name) || disabled.contains(&plugin.key) {
        return false;
    }
    if plugin.kind.is_auto_enabled(plugin.source) {
        return true;
    }
    enabled.contains(&plugin.name) || enabled.contains(&plugin.key)
}

fn project_plugins_dir() -> PathBuf {
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS") {
        Path::new(".").join(".hermes").join("plugins")
    } else {
        PathBuf::new()
    }
}

fn env_var_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn scan_plugin_tree(
    root: &Path,
    source: PluginSource,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    let mut plugins = Vec::new();
    scan_plugin_tree_level(root, source, "", 0, &mut plugins)?;
    Ok(plugins)
}

fn scan_plugin_tree_level(
    root: &Path,
    source: PluginSource,
    prefix: &str,
    depth: usize,
    out: &mut Vec<DiscoveredPlugin>,
) -> Result<(), Box<dyn Error>> {
    if !root.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let manifest_path = plugin_manifest_path(&path);
        if let Some(manifest_path) = manifest_path {
            if let Some(plugin) = parse_plugin_manifest(&manifest_path, &path, prefix, source)? {
                out.push(plugin);
            }
            continue;
        }
        if depth >= 1 {
            continue;
        }
        let next_prefix = if prefix.is_empty() {
            dir_name
        } else {
            format!("{prefix}/{dir_name}")
        };
        scan_plugin_tree_level(&path, source, &next_prefix, depth + 1, out)?;
    }

    Ok(())
}

fn scan_memory_provider_root(
    root: &Path,
    source: PluginSource,
    prefix: &str,
    require_marker: bool,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    let mut plugins = Vec::new();
    if !root.is_dir() {
        return Ok(plugins);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(plugin) = parse_memory_provider_dir(&path, prefix, source, require_marker)? else {
            continue;
        };
        plugins.push(plugin);
    }
    Ok(plugins)
}

fn scan_dashboard_tree(
    root: &Path,
    source: PluginSource,
) -> Result<Vec<DashboardSurface>, Box<dyn Error>> {
    let mut dashboards = Vec::new();
    if !root.is_dir() {
        return Ok(dashboards);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest = path.join("dashboard").join("manifest.json");
        if !manifest.exists() {
            continue;
        }
        let text = fs::read_to_string(&manifest)?;
        let parsed = serde_json::from_str::<DashboardManifest>(&text)?;
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let name = parsed
            .name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&dir_name)
            .to_string();
        let tab = parsed.tab.unwrap_or_default();
        let tab_path = tab
            .path
            .as_deref()
            .filter(|value| value.starts_with('/'))
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("/{name}"));
        let tab_position = tab
            .position
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("end")
            .to_string();
        dashboards.push(DashboardSurface {
            name: name.clone(),
            label: parsed.label.unwrap_or_else(|| name.clone()),
            description: parsed.description.unwrap_or_default(),
            icon: parsed.icon.unwrap_or_else(|| String::from("Puzzle")),
            version: parsed.version.unwrap_or_else(|| String::from("0.0.0")),
            entry: parsed
                .entry
                .unwrap_or_else(|| String::from("dist/index.js")),
            css: parsed.css,
            api: parsed.api,
            slots: parsed.slots.unwrap_or_default(),
            tab_path,
            tab_position,
            tab_override: tab.override_path.filter(|value| value.starts_with('/')),
            tab_hidden: tab.hidden.unwrap_or(false),
            source,
            dashboard_dir: path.join("dashboard"),
        });
    }
    Ok(dashboards)
}

fn parse_plugin_manifest(
    manifest_path: &Path,
    plugin_dir: &Path,
    prefix: &str,
    source: PluginSource,
) -> Result<Option<DiscoveredPlugin>, Box<dyn Error>> {
    let text = fs::read_to_string(manifest_path)?;
    let parsed = serde_yaml::from_str::<YamlValue>(&text).unwrap_or(YamlValue::Null);
    let mapping = match parsed.as_mapping() {
        Some(mapping) => mapping,
        None => return Ok(None),
    };
    let dir_name = plugin_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_string();
    let key = if prefix.is_empty() {
        dir_name.clone()
    } else {
        format!("{prefix}/{dir_name}")
    };
    let name = mapping_string(mapping, "name").unwrap_or_else(|| dir_name.clone());
    if name.trim().is_empty() {
        return Ok(None);
    }

    let source_text = read_plugin_source_files(plugin_dir);
    let kind = determine_plugin_kind(&key, mapping_string(mapping, "kind"), &source_text);
    let tool_definitions = discover_tool_definitions(&source_text);
    let cli_commands = discover_cli_commands(&source_text, &name);
    let platforms = discover_platform_surfaces(&source_text, &name);

    Ok(Some(DiscoveredPlugin {
        key,
        name,
        version: mapping_string(mapping, "version").unwrap_or_default(),
        description: mapping_string(mapping, "description").unwrap_or_default(),
        author: mapping_string(mapping, "author").unwrap_or_default(),
        kind,
        source,
        path: plugin_dir.to_path_buf(),
        requires_env: mapping_string_list(mapping, "requires_env"),
        provides_tools: {
            let mut tools = mapping_string_list(mapping, "provides_tools");
            for definition in &tool_definitions {
                if !tools.iter().any(|tool| tool == &definition.name) {
                    tools.push(definition.name.clone());
                }
            }
            tools
        },
        provides_hooks: {
            let mut hooks = mapping_string_list(mapping, "provides_hooks");
            if hooks.is_empty() {
                hooks = mapping_string_list(mapping, "hooks");
            }
            for hook in discover_hook_registrations(&source_text) {
                if !hooks.iter().any(|existing| existing == &hook) {
                    hooks.push(hook);
                }
            }
            hooks
        },
        tool_definitions,
        cli_commands,
        platforms,
    }))
}

fn parse_memory_provider_dir(
    plugin_dir: &Path,
    prefix: &str,
    source: PluginSource,
    require_marker: bool,
) -> Result<Option<DiscoveredPlugin>, Box<dyn Error>> {
    let init_file = plugin_dir.join("__init__.py");
    if !init_file.exists() {
        return Ok(None);
    }
    let dir_name = plugin_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if dir_name.is_empty() || dir_name.starts_with(['.', '_']) {
        return Ok(None);
    }

    let key = if prefix.is_empty() {
        dir_name.clone()
    } else {
        format!("{prefix}/{dir_name}")
    };
    let source_text = read_plugin_source_files(plugin_dir);
    if source_text.trim().is_empty() {
        return Ok(None);
    }

    let manifest = plugin_manifest_path(plugin_dir)
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|text| serde_yaml::from_str::<YamlValue>(&text).ok())
        .and_then(|value| value.as_mapping().cloned());
    let raw_kind = manifest
        .as_ref()
        .and_then(|mapping| mapping_string(mapping, "kind"));
    let kind = determine_plugin_kind(&key, raw_kind, &source_text);
    if kind != PluginKind::Exclusive {
        return Ok(None);
    }
    if require_marker
        && !source_text.contains("register_memory_provider")
        && !source_text.contains("MemoryProvider")
    {
        return Ok(None);
    }

    let description = manifest
        .as_ref()
        .and_then(|mapping| mapping_string(mapping, "description"))
        .unwrap_or_default();
    Ok(Some(DiscoveredPlugin {
        key,
        name: dir_name,
        version: manifest
            .as_ref()
            .and_then(|mapping| mapping_string(mapping, "version"))
            .unwrap_or_default(),
        description,
        author: manifest
            .as_ref()
            .and_then(|mapping| mapping_string(mapping, "author"))
            .unwrap_or_default(),
        kind,
        source,
        path: plugin_dir.to_path_buf(),
        requires_env: manifest
            .as_ref()
            .map(|mapping| mapping_string_list(mapping, "requires_env"))
            .unwrap_or_default(),
        provides_tools: Vec::new(),
        provides_hooks: Vec::new(),
        tool_definitions: Vec::new(),
        cli_commands: Vec::new(),
        platforms: Vec::new(),
    }))
}

fn determine_plugin_kind(key: &str, raw_kind: Option<String>, source_text: &str) -> PluginKind {
    match raw_kind
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "backend" => return PluginKind::Backend,
        "exclusive" => return PluginKind::Exclusive,
        "platform" => return PluginKind::Platform,
        "model-provider" => return PluginKind::ModelProvider,
        "standalone" | "" => {}
        _ => {}
    }

    if key.starts_with("memory/")
        || source_text.contains("register_memory_provider")
        || source_text.contains("MemoryProvider")
    {
        return PluginKind::Exclusive;
    }
    if key.starts_with("model-providers/")
        || (source_text.contains("register_provider") && source_text.contains("ProviderProfile"))
    {
        return PluginKind::ModelProvider;
    }
    if key.starts_with("platforms/") || source_text.contains("register_platform") {
        return PluginKind::Platform;
    }
    if key.starts_with("image_gen/")
        || key.starts_with("observability/")
        || source_text.contains("register_image_gen_provider")
    {
        return PluginKind::Backend;
    }
    if key.starts_with("context_engine/")
        || source_text.contains("register_context_engine")
        || source_text.contains("ContextEngine")
    {
        return PluginKind::ContextEngine;
    }
    PluginKind::Standalone
}

fn read_plugin_source_files(plugin_dir: &Path) -> String {
    let mut parts = Vec::new();
    let Ok(entries) = fs::read_dir(plugin_dir) else {
        return String::new();
    };
    let mut files = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|value| value.to_str()) == Some("py")
        })
        .collect::<Vec<_>>();
    files.sort();
    for file in files {
        if let Ok(text) = fs::read_to_string(file) {
            parts.push(text);
        }
    }
    parts.join("\n")
}

fn discover_cli_commands(source_text: &str, plugin_name: &str) -> Vec<PluginCliCommand> {
    let mut commands = Vec::new();
    let mut blocks = extract_call_blocks(source_text, "ctx.register_cli_command");
    if blocks.is_empty() {
        blocks = extract_call_blocks(source_text, "register_cli_command");
    }
    for block in blocks {
        let Some(name) = extract_keyword_string(&block, "name") else {
            continue;
        };
        let help = extract_keyword_string(&block, "help").unwrap_or_default();
        let description = extract_keyword_string(&block, "description").unwrap_or_default();
        commands.push(PluginCliCommand {
            name,
            help,
            description,
            plugin_name: plugin_name.to_string(),
        });
    }
    commands
}

fn discover_hook_registrations(source_text: &str) -> Vec<String> {
    discover_hook_registrations_from_source(source_text)
}

fn discover_tool_definitions(source_text: &str) -> Vec<ToolDefinition> {
    discover_tool_definitions_from_source(source_text)
}

fn discover_platform_surfaces(source_text: &str, plugin_name: &str) -> Vec<PlatformSurface> {
    let mut platforms = Vec::new();
    let mut blocks = extract_call_blocks(source_text, "ctx.register_platform");
    if blocks.is_empty() {
        blocks = extract_call_blocks(source_text, "register_platform");
    }
    for block in blocks {
        let Some(key) = extract_keyword_string(&block, "name") else {
            continue;
        };
        platforms.push(PlatformSurface {
            label: extract_keyword_string(&block, "label").unwrap_or_else(|| titleize(&key)),
            required_env: extract_keyword_string_list(&block, "required_env"),
            install_hint: extract_keyword_string(&block, "install_hint"),
            emoji: extract_keyword_string(&block, "emoji").unwrap_or_default(),
            has_setup_fn: keyword_present(&block, "setup_fn"),
            key,
            plugin_name: plugin_name.to_string(),
        });
    }
    platforms
}

fn discover_active_memory_cli(
    context: &HermesContext,
) -> Result<Option<PluginCliCommand>, Box<dyn Error>> {
    let provider = fs::read_to_string(context.config_path())
        .ok()
        .and_then(|text| serde_yaml::from_str::<YamlValue>(&text).ok())
        .and_then(|value| {
            value
                .as_mapping()
                .and_then(|mapping| mapping.get(YamlValue::String(String::from("memory"))))
                .and_then(YamlValue::as_mapping)
                .and_then(|mapping| mapping.get(YamlValue::String(String::from("provider"))))
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default();
    if provider.is_empty() {
        return Ok(None);
    }
    let candidates = [
        bundled_plugins_dir().join("memory").join(&provider),
        user_plugins_dir(context)?.join(&provider),
    ];
    let plugin_dir = candidates
        .into_iter()
        .find(|path| path.join("cli.py").exists());
    let Some(plugin_dir) = plugin_dir else {
        return Ok(None);
    };
    let description = plugin_manifest_path(&plugin_dir)
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|text| serde_yaml::from_str::<YamlValue>(&text).ok())
        .and_then(|value| {
            value
                .as_mapping()
                .and_then(|mapping| mapping_string(mapping, "description"))
        })
        .unwrap_or_default();
    Ok(Some(PluginCliCommand {
        name: provider.clone(),
        help: if description.is_empty() {
            format!("Manage {provider} memory plugin")
        } else {
            description.clone()
        },
        description,
        plugin_name: provider,
    }))
}

fn plugin_manifest_path(path: &Path) -> Option<PathBuf> {
    let yaml = path.join("plugin.yaml");
    if yaml.exists() {
        return Some(yaml);
    }
    let yml = path.join("plugin.yml");
    yml.exists().then_some(yml)
}

fn mapping_string(mapping: &serde_yaml::Mapping, key: &str) -> Option<String> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn mapping_string_list(mapping: &serde_yaml::Mapping, key: &str) -> Vec<String> {
    let Some(value) = mapping.get(YamlValue::String(key.to_string())) else {
        return Vec::new();
    };
    match value {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(|item| match item {
                YamlValue::String(text) => {
                    let trimmed = text.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                }
                YamlValue::Mapping(mapping) => mapping
                    .get(YamlValue::String(String::from("name")))
                    .and_then(YamlValue::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(ToOwned::to_owned),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_call_blocks(source: &str, func_name: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let needle = func_name.as_bytes();
    let bytes = source.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let Some(found) = find_bytes(bytes, needle, index) else {
            break;
        };
        let mut cursor = found + needle.len();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'(' {
            index = found + needle.len();
            continue;
        }
        let open = cursor;
        let mut depth = 0usize;
        let mut string_delim = None::<u8>;
        let mut escaped = false;
        while cursor < bytes.len() {
            let ch = bytes[cursor];
            if let Some(delim) = string_delim {
                if escaped {
                    escaped = false;
                } else if ch == b'\\' {
                    escaped = true;
                } else if ch == delim {
                    string_delim = None;
                }
                cursor += 1;
                continue;
            }
            match ch {
                b'\'' | b'"' => string_delim = Some(ch),
                b'(' => depth += 1,
                b')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        blocks.push(source[open + 1..cursor].to_string());
                        cursor += 1;
                        break;
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        index = cursor;
    }
    blocks
}

fn extract_keyword_string(block: &str, key: &str) -> Option<String> {
    let pattern = format!(
        r#"(?s)\b{}\s*=\s*(?:"([^"]*)"|'([^']*)')"#,
        regex::escape(key)
    );
    let regex = Regex::new(&pattern).ok()?;
    regex
        .captures(block)
        .and_then(|captures| captures.get(1).or_else(|| captures.get(2)))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_keyword_string_list(block: &str, key: &str) -> Vec<String> {
    let pattern = format!(r#"(?s)\b{}\s*=\s*\[(.*?)\]"#, regex::escape(key));
    let Some(regex) = Regex::new(&pattern).ok() else {
        return Vec::new();
    };
    let Some(captures) = regex.captures(block) else {
        return Vec::new();
    };
    let Some(body) = captures.get(1) else {
        return Vec::new();
    };
    let Some(string_re) = Regex::new(r#"(?:"([^"]*)"|'([^']*)')"#).ok() else {
        return Vec::new();
    };
    string_re
        .captures_iter(body.as_str())
        .filter_map(|caps| {
            caps.get(1)
                .or_else(|| caps.get(2))
                .map(|value| value.as_str().trim().to_string())
        })
        .filter(|value| !value.is_empty())
        .collect()
}

fn keyword_present(block: &str, key: &str) -> bool {
    let pattern = format!(r#"\b{}\s*="#, regex::escape(key));
    Regex::new(&pattern)
        .map(|regex| regex.is_match(block))
        .unwrap_or(false)
}

fn find_bytes(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    haystack[start..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|index| start + index)
}

fn titleize(value: &str) -> String {
    value
        .split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use hermes_core::{dispatch_tool, get_tool_definitions_with_runtime};
    use tempfile::TempDir;

    fn write_plugin(dir: &Path, manifest: &str, source: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("plugin.yaml"), manifest).unwrap();
        fs::write(dir.join("__init__.py"), source).unwrap();
    }

    #[test]
    fn extract_cli_and_platform_surfaces_from_source() {
        let source = r#"
def register(ctx):
    ctx.register_cli_command(
        name="meet",
        help="Join a meeting",
        description="Meet tools",
    )
    ctx.register_platform(
        name="irc",
        label="IRC",
        required_env=["IRC_SERVER", "IRC_CHANNEL"],
        install_hint="stdlib only",
        setup_fn=interactive_setup,
        emoji="💬",
    )
"#;
        let commands = discover_cli_commands(source, "demo");
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "meet");
        assert_eq!(commands[0].help, "Join a meeting");

        let platforms = discover_platform_surfaces(source, "demo");
        assert_eq!(platforms.len(), 1);
        assert_eq!(platforms[0].key, "irc");
        assert_eq!(platforms[0].label, "IRC");
        assert_eq!(platforms[0].required_env, vec!["IRC_SERVER", "IRC_CHANNEL"]);
        assert!(platforms[0].has_setup_fn);
    }

    #[test]
    fn extract_tool_and_hook_surfaces_from_source() {
        let source = r#"
DEMO_SCHEMA = {
    "name": "demo_tool",
    "description": "Demo tool",
    "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
}

def register(ctx):
    ctx.register_tool(
        name="demo_tool",
        toolset="demo",
        schema=DEMO_SCHEMA,
        handler=handle_demo,
        emoji="plug",
    )
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
"#;
        let tools = discover_tool_definitions(source);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "demo_tool");
        assert_eq!(tools[0].toolset, "demo");
        assert_eq!(tools[0].description, "Demo tool");
        assert_eq!(tools[0].emoji, "plug");
        assert_eq!(
            tools[0].schema["parameters"]["type"],
            serde_json::json!("object")
        );

        let hooks = discover_hook_registrations(source);
        assert_eq!(hooks, vec![String::from("pre_llm_call")]);
    }

    #[test]
    fn extract_tuple_registered_tools_from_source() {
        let source = r#"
MEET_JOIN_SCHEMA = {
    "name": "meet_join",
    "description": "Join a meeting",
    "parameters": {"type": "object", "properties": {}},
}
MEET_STATUS_SCHEMA = {
    "name": "meet_status",
    "description": "Report meeting status",
    "parameters": {"type": "object", "properties": {}},
}
_TOOLS = (
    ("meet_join", MEET_JOIN_SCHEMA, handle_meet_join, "📞"),
    ("meet_status", MEET_STATUS_SCHEMA, handle_meet_status, "🟢"),
)

def register(ctx):
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="google_meet",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
"#;
        let tools = discover_tool_definitions(source);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "meet_join");
        assert_eq!(tools[0].toolset, "google_meet");
        assert_eq!(tools[0].description, "Join a meeting");
        assert_eq!(tools[1].name, "meet_status");
        assert_eq!(tools[1].description, "Report meeting status");
    }

    #[test]
    fn catalog_discovers_nested_plugins_and_dashboard_manifests() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let home = temp.path().join(".hermes");
        let user_plugins = home.join("plugins");
        fs::create_dir_all(&user_plugins).unwrap();

        write_plugin(
            &bundled.join("platforms").join("irc"),
            "name: irc-platform\nkind: platform\ndescription: IRC\n",
            "from .adapter import register\n",
        );
        fs::write(
            bundled.join("platforms").join("irc").join("adapter.py"),
            r#"
def register(ctx):
    ctx.register_platform(name="irc", label="IRC", required_env=["IRC_SERVER"], emoji="💬")
"#,
        )
        .unwrap();
        write_plugin(
            &bundled.join("image_gen").join("openai"),
            "name: openai\nkind: backend\ndescription: OpenAI images\n",
            "def register(ctx):\n    pass\n",
        );
        write_plugin(
            &bundled.join("memory").join("honcho"),
            "name: honcho\ndescription: Memory\n",
            "class MemoryProvider:\n    pass\n",
        );
        fs::create_dir_all(bundled.join("example-dashboard").join("dashboard")).unwrap();
        fs::write(
            bundled
                .join("example-dashboard")
                .join("dashboard")
                .join("manifest.json"),
            r#"{"name":"example","label":"Example","entry":"dist/index.js","slots":["sidebar"]}"#,
        )
        .unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        let catalog = discover_plugin_catalog(&context).unwrap();
        assert!(
            catalog
                .plugins
                .iter()
                .any(|plugin| plugin.name == "irc-platform")
        );
        assert!(catalog.plugins.iter().any(|plugin| plugin.name == "openai"));
        assert!(
            catalog
                .memory_providers
                .iter()
                .any(|provider| provider.name == "honcho")
        );
        assert!(
            catalog
                .dashboards
                .iter()
                .any(|dashboard| dashboard.name == "example")
        );

        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn memory_provider_discovery_prefers_bundled_and_accepts_manifestless_user_plugins() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let home = temp.path().join(".hermes");
        let user_plugins = home.join("plugins");
        fs::create_dir_all(bundled.join("memory").join("honcho")).unwrap();
        fs::create_dir_all(user_plugins.join("honcho")).unwrap();
        fs::create_dir_all(user_plugins.join("custom")).unwrap();

        fs::write(
            bundled.join("memory").join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("plugin.yaml"),
            "description: Bundled honcho\n",
        )
        .unwrap();
        fs::write(
            user_plugins.join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            user_plugins.join("custom").join("__init__.py"),
            "def register_memory_provider(ctx):\n    pass\n",
        )
        .unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let providers = discover_memory_provider_plugins(&context).unwrap();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].name, "custom");
        assert_eq!(providers[0].description, "");
        assert_eq!(providers[1].name, "honcho");
        assert_eq!(providers[1].description, "Bundled honcho");
        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn enabled_cli_commands_include_active_memory_provider() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let bundled = temp.path().join("bundled");
        fs::create_dir_all(home.join("plugins")).unwrap();
        fs::create_dir_all(bundled.join("memory").join("honcho")).unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("plugin.yaml"),
            "name: honcho\ndescription: Honcho provider\n",
        )
        .unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("cli.py"),
            "def register_cli(subparser):\n    pass\n",
        )
        .unwrap();
        fs::write(home.join("config.yaml"), "memory:\n  provider: honcho\n").unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let commands = discover_enabled_cli_commands(&context).unwrap();
        assert!(commands.iter().any(|command| command.name == "honcho"));
        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn attach_python_plugin_runtime_bridges_tools_and_hooks() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("demo");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: demo\ndescription: Demo plugin\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
import json


def register(ctx):
    ctx.register_tool(
        name="demo_tool",
        toolset="demo_tools",
        schema={
            "name": "demo_tool",
            "description": "Demo tool",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {
                "success": True,
                "echo": args.get("message", ""),
                "task_id": kwargs.get("task_id", ""),
            }
        ),
        description="Demo tool",
        emoji="plug",
    )

    def pre_llm_call(**kwargs):
        return {"context": "dynamic context"}

    def pre_tool_call(**kwargs):
        args = kwargs.get("args") or {}
        if kwargs.get("tool_name") == "demo_tool" and args.get("message") == "blocked":
            return {"action": "block", "message": "blocked by plugin"}
        return None

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo\n",
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let runtime = attach_python_plugin_runtime(
            &context,
            ToolRuntime::new(temp.path()).with_hermes_home(home),
        )
        .unwrap();

        let enabled_toolsets = vec![String::from("demo_tools")];
        let definitions =
            get_tool_definitions_with_runtime(Some(&enabled_toolsets), None, Some(&runtime));
        assert!(definitions.iter().any(|tool| tool.name == "demo_tool"));

        let hook_results = runtime.invoke_hook(
            "pre_llm_call",
            &serde_json::json!({
                "session_id": "session-1",
                "user_message": "hi",
                "conversation_history": [],
                "is_first_turn": true,
                "model": "test-model",
                "platform": "",
                "sender_id": "",
            }),
        );
        assert!(hook_results.iter().any(|result| {
            result
                .get("context")
                .and_then(JsonValue::as_str)
                .is_some_and(|value| value == "dynamic context")
        }));

        let result = dispatch_tool(
            "demo_tool",
            serde_json::json!({"message": "hello"}),
            &runtime,
        );
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed.get("echo").and_then(JsonValue::as_str),
            Some("hello")
        );

        let blocked = dispatch_tool(
            "demo_tool",
            serde_json::json!({"message": "blocked"}),
            &runtime,
        );
        let parsed_blocked: JsonValue = serde_json::from_str(&blocked).unwrap();
        assert_eq!(
            parsed_blocked.get("error").and_then(JsonValue::as_str),
            Some("blocked by plugin")
        );
    }

    #[test]
    fn attach_python_plugin_runtime_keeps_hook_only_plugins_active() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("observer");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: observer\ndescription: Hook only plugin\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
def register(ctx):
    def pre_llm_call(**kwargs):
        return {"context": "hook-only context"}
    ctx.register_hook("pre_llm_call", pre_llm_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - observer\n",
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let runtime = attach_python_plugin_runtime(
            &context,
            ToolRuntime::new(temp.path()).with_hermes_home(home),
        )
        .unwrap();

        let hook_results = runtime.invoke_hook(
            "pre_llm_call",
            &serde_json::json!({
                "session_id": "session-1",
                "user_message": "hi",
                "conversation_history": [],
                "is_first_turn": true,
                "model": "test-model",
                "platform": "",
                "sender_id": "",
            }),
        );
        assert!(hook_results.iter().any(|result| {
            result
                .get("context")
                .and_then(JsonValue::as_str)
                .is_some_and(|value| value == "hook-only context")
        }));
    }
}
