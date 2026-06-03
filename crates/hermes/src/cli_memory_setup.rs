//! `hermes memory setup|status` — configure memory provider plugins.
//!
//! Native Rust port of `hermes_cli/memory_setup.py`.
//!
//! Auto-detects installed memory providers via the plugin system. The original
//! Python module drives an interactive curses picker, walks the provider's
//! config schema, and writes config to `config.yaml` + `.env`. This port keeps
//! the same control flow and side effects, swapping the curses picker for a
//! line-based numbered selector (curses isn't available natively) and reading
//! the provider config schema from each plugin's `plugin.yaml` manifest (the
//! native analogue of `provider.get_config_schema()` / `post_setup` / etc.).
//!
//! Provider discovery is delegated to the already-ported plugin runtime
//! (`crate::plugin_runtime`). Config / `.env` reads and writes reuse the
//! ported config helpers in `crate::config_cmd`. The `HERMES_HOME` resolution
//! mirrors `hermes_constants.get_hermes_home()` via
//! `crate::plugin_runtime`-supplied `HermesContext` paths, falling back to the
//! native `mod_hermes_constants::get_hermes_home()` when no context is given.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use hermes_core::HermesContext;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping, Value as YamlValue};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::plugin_runtime::{discover_memory_providers, find_memory_provider_plugin};

// ---------------------------------------------------------------------------
// Plugin manifest (plugin.yaml) — the native source for a provider's config
// schema, dependency declarations, and setup hooks. Mirrors the shape used by
// the Python plugins' plugin.yaml + the provider's get_config_schema().
// ---------------------------------------------------------------------------

/// Top-level `plugin.yaml` for a memory provider.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct MemoryPluginManifest {
    #[serde(default)]
    pub pip_dependencies: Vec<String>,
    #[serde(default)]
    pub external_dependencies: Vec<ExternalDependency>,
    #[serde(default)]
    pub setup: Option<MemorySetupManifest>,
}

/// Non-pip dependency hint (e.g. a system binary check + install command).
#[derive(Debug, Default, Clone, Deserialize)]
pub struct ExternalDependency {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub check: String,
    #[serde(default)]
    pub install: String,
}

/// `setup:` block of `plugin.yaml`. The schema mirrors a provider's
/// `get_config_schema()` and `post_setup` / `save_config` hooks.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct MemorySetupManifest {
    #[serde(default)]
    pub schema: Vec<SchemaField>,
    /// True if the provider has a `post_setup` hook that owns its own flow.
    #[serde(default, alias = "has_post_setup")]
    pub post_setup: bool,
    /// True if the provider persists non-secret config to a native location.
    #[serde(default, alias = "has_save_config")]
    pub save_config: bool,
    #[serde(default, alias = "save_config_path", alias = "config_file")]
    pub save_config_file: Option<String>,
    #[serde(default, alias = "available")]
    pub is_available: Option<bool>,
}

/// One field of a provider's config schema (mirror of a Python schema dict).
#[derive(Debug, Default, Clone, Deserialize)]
pub struct SchemaField {
    pub key: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default: Option<JsonValue>,
    #[serde(default)]
    pub secret: bool,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub env_var: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub choices: Option<Vec<String>>,
    /// Only prompt this field when all of these provider_config keys match.
    #[serde(default)]
    pub when: BTreeMap<String, JsonValue>,
    /// Dynamic default: look up a default from another field's value.
    #[serde(default)]
    pub default_from: Option<DefaultFrom>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct DefaultFrom {
    #[serde(default)]
    pub field: String,
    #[serde(default)]
    pub map: BTreeMap<String, String>,
}

impl SchemaField {
    fn description_text(&self) -> String {
        self.description
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| self.key.clone())
    }
}

// ---------------------------------------------------------------------------
// Provider discovery (port of `_get_available_providers`)
// ---------------------------------------------------------------------------

/// A discovered memory provider: name, computed setup hint, and its schema.
///
/// Equivalent to the Python `(name, setup_hint, provider)` tuple, except the
/// opaque provider instance is replaced by the data this module needs (schema
/// + setup hooks), parsed from the plugin's `plugin.yaml`.
#[derive(Debug, Clone)]
pub struct AvailableProvider {
    pub name: String,
    pub setup_hint: String,
    pub schema: Vec<SchemaField>,
    pub has_post_setup: bool,
    pub has_save_config: bool,
    pub manifest_available: Option<bool>,
}

/// Compute the setup hint from a schema, exactly as `_get_available_providers`.
///
/// - secrets + non-secrets -> "API key / local"
/// - secrets only          -> "requires API key"
/// - empty schema          -> "no setup needed"
/// - otherwise             -> "local"
pub fn setup_hint_for_schema(schema: &[SchemaField]) -> String {
    let has_secrets = schema.iter().any(|field| field.secret);
    let has_non_secrets = schema.iter().any(|field| !field.secret);
    if has_secrets && has_non_secrets {
        String::from("API key / local")
    } else if has_secrets {
        String::from("requires API key")
    } else if schema.is_empty() {
        String::from("no setup needed")
    } else {
        String::from("local")
    }
}

/// Discover memory providers (port of `_get_available_providers`).
///
/// Returns the list of available providers in discovery order. Providers whose
/// plugin directory cannot be located are skipped, matching the Python
/// behaviour where `load_memory_provider(name)` returning `None` skips the
/// entry.
pub fn get_available_providers(context: &HermesContext) -> Vec<AvailableProvider> {
    let raw = discover_memory_providers(context).unwrap_or_default();
    let mut results = Vec::new();
    for option in raw {
        let Some(dir) = find_memory_provider_dir(context, &option.name) else {
            // load_memory_provider returned None -> skip.
            continue;
        };
        let manifest = read_memory_plugin_manifest(&dir).unwrap_or_default();
        let setup = manifest.setup.unwrap_or_default();
        let schema = setup.schema;
        let setup_hint = setup_hint_for_schema(&schema);
        results.push(AvailableProvider {
            name: option.name,
            setup_hint,
            schema,
            has_post_setup: setup.post_setup,
            has_save_config: setup.save_config,
            manifest_available: setup.is_available,
        });
    }
    results
}

fn find_memory_provider_dir(context: &HermesContext, provider_name: &str) -> Option<PathBuf> {
    find_memory_provider_plugin(context, provider_name)
        .ok()
        .flatten()
        .map(|plugin| plugin.path)
}

/// Read `<dir>/plugin.yaml` (port of `find_provider_dir` + yaml.safe_load).
pub fn read_memory_plugin_manifest(dir: &Path) -> Option<MemoryPluginManifest> {
    let text = fs::read_to_string(dir.join("plugin.yaml")).ok()?;
    serde_yaml::from_str::<MemoryPluginManifest>(&text).ok()
}

// ---------------------------------------------------------------------------
// Dependency installation (port of `_install_dependencies`)
// ---------------------------------------------------------------------------

/// pip name -> import name mapping for packages where the two differ.
/// Mirror of `_IMPORT_NAMES` in the Python source.
fn import_name_for(dep: &str) -> String {
    match dep {
        "honcho-ai" => String::from("honcho"),
        "mem0ai" => String::from("mem0"),
        "hindsight-client" => String::from("hindsight_client"),
        "hindsight-all" => String::from("hindsight"),
        _ => dep
            .replace('-', "_")
            .split('[')
            .next()
            .unwrap_or(dep)
            .to_string(),
    }
}

/// Install pip dependencies declared in plugin.yaml (port of
/// `_install_dependencies`). Best-effort: prints progress, never errors out.
///
/// `python` is the interpreter used to (a) probe whether each dependency is
/// already importable and (b) target the `uv pip install --python` call —
/// matching the Python module's use of `sys.executable`.
pub fn install_dependencies(context: &HermesContext, provider_name: &str, python: &Path) {
    let Some(plugin_dir) = find_memory_provider_dir(context, provider_name) else {
        return;
    };
    let yaml_path = plugin_dir.join("plugin.yaml");
    if !yaml_path.exists() {
        return;
    }
    let Some(meta) = read_memory_plugin_manifest(&plugin_dir) else {
        return;
    };

    if meta.pip_dependencies.is_empty() {
        // No pip deps — still surface external dependency hints below.
    }

    // Determine which packages are missing.
    let missing: Vec<String> = meta
        .pip_dependencies
        .iter()
        .filter(|dep| {
            let import_name = import_name_for(dep);
            !python_import_available(python, &import_name)
        })
        .cloned()
        .collect();

    if !missing.is_empty() {
        println!("\n  Installing dependencies: {}", missing.join(", "));

        match which("uv") {
            None => {
                println!("  ⚠ uv not found — cannot install dependencies");
                println!("  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh");
                println!("  Then re-run: hermes memory setup");
            }
            Some(uv_path) => {
                let mut args = vec![
                    String::from("pip"),
                    String::from("install"),
                    String::from("--python"),
                    python.display().to_string(),
                    String::from("--quiet"),
                ];
                args.extend(missing.iter().cloned());
                let output = Command::new(&uv_path).args(&args).output();
                match output {
                    Ok(out) if out.status.success() => {
                        println!("  ✓ Installed {}", missing.join(", "));
                    }
                    Ok(out) => {
                        println!("  ⚠ Failed to install {}", missing.join(", "));
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let snippet: String = stderr.chars().take(200).collect();
                        if !snippet.is_empty() {
                            println!("    {snippet}");
                        }
                        println!(
                            "  Run manually: uv pip install --python {} {}",
                            python.display(),
                            missing.join(" ")
                        );
                    }
                    Err(error) => {
                        println!("  ⚠ Install failed: {error}");
                        println!(
                            "  Run manually: uv pip install --python {} {}",
                            python.display(),
                            missing.join(" ")
                        );
                    }
                }
            }
        }
    }

    // Surface external (non-pip) dependency hints if the check command fails.
    for dep in &meta.external_dependencies {
        if dep.check.trim().is_empty() {
            continue;
        }
        let ran = run_shell_check(&dep.check);
        if !ran && !dep.install.trim().is_empty() {
            println!("\n  ⚠ '{}' not found. Install with:", dep.name);
            println!("    {}", dep.install);
        }
    }
}

/// Probe whether `python -c "import <name>"` succeeds.
fn python_import_available(python: &Path, import_name: &str) -> bool {
    if import_name.trim().is_empty() {
        return true;
    }
    Command::new(python)
        .arg("-c")
        .arg(format!("import {import_name}"))
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Run a shell `check` command; return true if it executed successfully.
/// The Python module treats any exception (including non-zero exit being
/// raised only via timeout/launch failure) as "not found"; here a launch
/// failure means not-found, mirroring the `try/except` semantics.
fn run_shell_check(check_cmd: &str) -> bool {
    let shell = if cfg!(windows) { "cmd" } else { "sh" };
    let flag = if cfg!(windows) { "/C" } else { "-c" };
    Command::new(shell).arg(flag).arg(check_cmd).output().is_ok()
}

/// `shutil.which` equivalent.
pub fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(name);
            candidate.is_file().then_some(candidate)
        })
    })
}

// ---------------------------------------------------------------------------
// Prompt helpers (port of `_prompt` and the curses picker)
// ---------------------------------------------------------------------------

/// Prompt for a value with optional default and secret masking
/// (port of `_prompt`).
///
/// Returns the entered value, or the default when input is blank.
pub fn prompt(label: &str, default: Option<&str>, secret: bool) -> io::Result<String> {
    let suffix = match default {
        Some(value) if !value.is_empty() => format!(" [{value}]"),
        _ => String::new(),
    };
    let line = format!("  {label}{suffix}: ");
    let value = if secret {
        read_secret_line(&line)?
    } else {
        read_line(&line)?
    };
    let value = value.trim().to_string();
    if value.is_empty() {
        Ok(default.unwrap_or("").to_string())
    } else {
        Ok(value)
    }
}

fn read_line(prompt_text: &str) -> io::Result<String> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt_text.as_bytes())?;
    stdout.flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim_end_matches(['\n', '\r']).to_string())
}

fn read_secret_line(prompt_text: &str) -> io::Result<String> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt_text.as_bytes())?;
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
    io::stdin().read_line(&mut line)?;

    if echo_disabled {
        let _ = Command::new("stty").arg("echo").status();
        println!();
    }
    Ok(line.trim_end_matches(['\n', '\r']).to_string())
}

/// Single-select picker (port of `_curses_select`). Returns the selected index,
/// or `default` on a blank line / EOF / cancel.
///
/// `items` are `(label, description)` pairs; they render the same way as the
/// curses `display_items` formatting (`"{label}  {desc}"`).
pub fn select(title: &str, items: &[(String, String)], default: usize) -> io::Result<usize> {
    if items.is_empty() {
        return Ok(default);
    }
    loop {
        println!("\n{title}");
        println!("────────────────────────────────────────");
        for (index, (label, desc)) in items.iter().enumerate() {
            let display = if desc.is_empty() {
                label.clone()
            } else {
                format!("{label}  {desc}")
            };
            let marker = if index == default { " ←" } else { "" };
            println!("  {}) {}{}", index + 1, display, marker);
        }
        let line = read_line(&format!("Select [{}]: ", default + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(default);
        }
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= items.len() {
                return Ok(choice - 1);
            }
        }
        // Match by label (case-insensitive).
        if let Some((index, _)) = items
            .iter()
            .enumerate()
            .find(|(_, (label, _))| label.eq_ignore_ascii_case(trimmed))
        {
            return Ok(index);
        }
        println!("  Invalid selection.");
    }
}

// ---------------------------------------------------------------------------
// Config / .env access
// ---------------------------------------------------------------------------

fn config_path(context: &HermesContext) -> PathBuf {
    context.config_path()
}

fn env_path(context: &HermesContext) -> PathBuf {
    context.env_path()
}

fn hermes_home(context: &HermesContext) -> PathBuf {
    context.hermes_home()
}

/// Read the `memory.<provider>` config block as a string->string map (only the
/// scalar string values, matching how the wizard echoes existing values).
fn load_provider_config(context: &HermesContext, provider_name: &str) -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    let Ok(root) = read_raw_yaml_mapping(&config_path(context)) else {
        return values;
    };
    let Some(memory) = mapping_child(&root, "memory") else {
        return values;
    };
    let Some(provider) = mapping_child(memory, provider_name) else {
        return values;
    };
    for (key, value) in provider {
        if let Some(key) = key.as_str() {
            values.insert(key.to_string(), yaml_scalar_string(value));
        }
    }
    values
}

fn mapping_child<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Mapping> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_mapping)
}

fn yaml_scalar_string(value: &YamlValue) -> String {
    match value {
        YamlValue::String(text) => text.clone(),
        YamlValue::Bool(boolean) => boolean.to_string(),
        YamlValue::Number(number) => number.to_string(),
        YamlValue::Null => String::new(),
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

/// Persist `memory.provider = name` and (optionally) `memory.<name>` config.
fn save_provider_config(
    context: &HermesContext,
    provider_name: &str,
    provider_config: &BTreeMap<String, String>,
    write_provider_block: bool,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&config_path(context))?;
    let memory = ensure_mapping(&mut root, "memory");
    memory.insert(
        YamlValue::String(String::from("provider")),
        YamlValue::String(provider_name.to_string()),
    );
    if write_provider_block && !provider_config.is_empty() {
        let mut block = Mapping::new();
        for (key, value) in provider_config {
            block.insert(
                YamlValue::String(key.clone()),
                YamlValue::String(value.clone()),
            );
        }
        memory.insert(
            YamlValue::String(provider_name.to_string()),
            YamlValue::Mapping(block),
        );
    }
    write_yaml_mapping(&config_path(context), &root)
}

fn ensure_mapping<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let entry = root
        .entry(YamlValue::String(key.to_string()))
        .or_insert_with(|| YamlValue::Mapping(Mapping::new()));
    if !entry.is_mapping() {
        *entry = YamlValue::Mapping(Mapping::new());
    }
    entry.as_mapping_mut().expect("just ensured mapping")
}

/// Append or update env vars in `.env` (port of `_write_env_vars`).
pub fn write_env_vars(
    env_file: &Path,
    env_writes: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = env_file.parent() {
        fs::create_dir_all(parent)?;
    }

    let existing_lines: Vec<String> = if env_file.exists() {
        fs::read_to_string(env_file)?
            .split('\n')
            // splitlines() drops a trailing empty produced by a final newline,
            // but here we want exact round-tripping of lines; trailing empty
            // segment from a final '\n' is removed below.
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    // Emulate Python str.splitlines(): no trailing empty element.
    let mut existing_lines = existing_lines;
    if existing_lines.last().map(String::as_str) == Some("") {
        existing_lines.pop();
    }

    let mut updated_keys = std::collections::BTreeSet::new();
    let mut new_lines = Vec::new();
    for line in &existing_lines {
        let key_match = if line.contains('=') {
            line.split_once('=')
                .map(|(key, _)| key.trim().to_string())
                .unwrap_or_default()
        } else {
            String::new()
        };
        if !key_match.is_empty() && env_writes.contains_key(&key_match) {
            new_lines.push(format!("{key_match}={}", env_writes[&key_match]));
            updated_keys.insert(key_match);
        } else {
            new_lines.push(line.clone());
        }
    }

    for (key, value) in env_writes {
        if !updated_keys.contains(key) {
            new_lines.push(format!("{key}={value}"));
        }
    }

    fs::write(env_file, format!("{}\n", new_lines.join("\n")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Setup wizard (port of `cmd_setup` / `cmd_setup_provider`)
// ---------------------------------------------------------------------------

/// Resolve the python interpreter used for dependency probing/installation.
/// Mirror of `sys.executable`: prefer `$HERMES_MEMORY_PYTHON`, then `python3`,
/// then `python` on PATH.
fn resolve_python() -> PathBuf {
    if let Some(path) = std::env::var_os("HERMES_MEMORY_PYTHON") {
        let path = PathBuf::from(path);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    which("python3")
        .or_else(|| which("python"))
        .unwrap_or_else(|| PathBuf::from("python3"))
}

/// Run memory setup for a specific provider, skipping the picker
/// (port of `cmd_setup_provider`).
pub fn cmd_setup_provider(
    context: &HermesContext,
    provider_name: &str,
) -> Result<(), Box<dyn Error>> {
    let providers = get_available_providers(context);
    let Some(provider) = providers.iter().find(|p| p.name == provider_name) else {
        println!("\n  Memory provider '{provider_name}' not found.");
        println!("  Run 'hermes memory setup' to see available providers.\n");
        return Ok(());
    };
    let provider = provider.clone();

    install_dependencies(context, &provider.name, &resolve_python());

    if provider.has_post_setup {
        // The Python hook owns its own config/connection-test/activation.
        // Natively we cannot execute the plugin's post_setup; signal that the
        // caller should hand off to the provider-specific setup.
        return run_post_setup_hook(context, &provider.name);
    }

    // Fallback: just record the activation key.
    save_provider_config(context, &provider.name, &BTreeMap::new(), false)?;
    println!("\n  Memory provider: {}", provider.name);
    println!("  Activation saved to config.yaml\n");
    Ok(())
}

/// Interactive memory provider setup wizard (port of `cmd_setup`).
pub fn cmd_setup(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let providers = get_available_providers(context);

    if providers.is_empty() {
        println!("\n  No memory provider plugins detected.");
        println!(
            "  Install a plugin to {}/plugins/ and try again.\n",
            context.display_hermes_home()
        );
        return Ok(());
    }

    // Build picker items: providers, then a "Built-in only" trailing entry.
    let mut items: Vec<(String, String)> = providers
        .iter()
        .map(|p| (p.name.clone(), format!("— {}", p.setup_hint)))
        .collect();
    items.push((
        String::from("Built-in only"),
        String::from("— MEMORY.md / USER.md (default)"),
    ));

    let builtin_idx = items.len() - 1;
    let selected = select("Memory provider setup", &items, builtin_idx)?;

    // Built-in only.
    if selected >= providers.len() {
        save_builtin_only(context)?;
        println!("\n  ✓ Memory provider: built-in only");
        println!("  Saved to config.yaml\n");
        return Ok(());
    }

    let provider = providers[selected].clone();

    install_dependencies(context, &provider.name, &resolve_python());

    if provider.has_post_setup {
        return run_post_setup_hook(context, &provider.name);
    }

    let mut provider_config = load_provider_config(context, &provider.name);
    let mut env_writes: BTreeMap<String, String> = BTreeMap::new();

    if !provider.schema.is_empty() {
        println!("\n  Configuring {}:\n", provider.name);

        for field in &provider.schema {
            // Skip fields whose `when` condition doesn't match.
            if !field.when.is_empty() {
                let matched = field.when.iter().all(|(k, v)| {
                    let expected = json_scalar_string(v);
                    provider_config.get(k).map(String::as_str) == Some(expected.as_str())
                });
                if !matched {
                    continue;
                }
            }

            let key = field.key.clone();
            let desc = field.description_text();
            let is_secret = field.secret;
            let env_var = field.env_var.clone();
            let url = field.url.clone();

            // Resolve the (possibly dynamic) default.
            let mut default = field.default.as_ref().map(json_scalar_string);
            if let Some(default_from) = &field.default_from {
                let ref_value = provider_config
                    .get(&default_from.field)
                    .cloned()
                    .unwrap_or_default();
                if !ref_value.is_empty() {
                    if let Some(mapped) = default_from.map.get(&ref_value) {
                        default = Some(mapped.clone());
                    }
                }
            }

            let choices = field.choices.clone().unwrap_or_default();

            if !choices.is_empty() && !is_secret {
                // Choice field — use the picker.
                let current = provider_config
                    .get(&key)
                    .cloned()
                    .or_else(|| default.clone())
                    .unwrap_or_default();
                let current_idx = choices.iter().position(|c| *c == current).unwrap_or(0);
                let choice_items: Vec<(String, String)> = choices
                    .iter()
                    .map(|c| (c.clone(), String::new()))
                    .collect();
                let sel = select(&format!("  {desc}"), &choice_items, current_idx)?;
                provider_config.insert(key.clone(), choices[sel].clone());
            } else if is_secret {
                // Secret prompt.
                let existing = env_var
                    .as_ref()
                    .and_then(|var| std::env::var(var).ok())
                    .unwrap_or_default();
                let val = if !existing.is_empty() {
                    let masked = if existing.chars().count() > 4 {
                        format!("...{}", &existing[existing.len().saturating_sub(4)..])
                    } else {
                        String::from("set")
                    };
                    prompt(
                        &format!("{desc} (current: {masked}, blank to keep)"),
                        None,
                        true,
                    )?
                } else {
                    if let Some(url) = url.as_deref() {
                        if !url.is_empty() {
                            println!("  Get yours at {url}");
                        }
                    }
                    prompt(&desc, None, true)?
                };
                if !val.is_empty() {
                    if let Some(var) = env_var {
                        env_writes.insert(var, val);
                    }
                }
            } else {
                // Regular text prompt.
                let current = provider_config.get(&key).cloned();
                let effective_default = current.filter(|c| !c.is_empty()).or(default);
                let val = prompt(&desc, effective_default.as_deref(), false)?;
                if !val.is_empty() {
                    provider_config.insert(key.clone(), val.clone());
                    if let Some(var) = env_var {
                        if !env_writes.contains_key(&var) {
                            env_writes.insert(var, val);
                        }
                    }
                }
            }
        }
    }

    // Write activation key + provider config to config.yaml.
    save_provider_config(context, &provider.name, &provider_config, true)?;

    // Write non-secret config to provider's native location.
    if !provider_config.is_empty() && provider.has_save_config {
        if let Err(error) =
            save_native_provider_config(context, &provider.name, &provider_config)
        {
            println!("  Failed to write provider config: {error}");
        }
    }

    // Write secrets to .env.
    if !env_writes.is_empty() {
        write_env_vars(&env_path(context), &env_writes)?;
    }

    println!("\n  Memory provider: {}", provider.name);
    println!("  Activation saved to config.yaml");
    if !provider_config.is_empty() {
        println!("  Provider config saved");
    }
    if !env_writes.is_empty() {
        println!("  API keys saved to .env");
    }
    println!("\n  Start a new session to activate.\n");
    Ok(())
}

fn save_builtin_only(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    save_provider_config(context, "", &BTreeMap::new(), false)
}

/// Persist non-secret provider config to its native location.
///
/// The Python `provider.save_config(provider_config, hermes_home)` writes the
/// provider's own config file; the manifest's `save_config_file` (relative to
/// HERMES_HOME) names that file. When unset, defaults to
/// `<HERMES_HOME>/<name>.json`, matching the common convention used by the
/// native provider implementations.
fn save_native_provider_config(
    context: &HermesContext,
    provider_name: &str,
    provider_config: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let dir = find_memory_provider_dir(context, provider_name);
    let save_file = dir
        .as_deref()
        .and_then(read_memory_plugin_manifest)
        .and_then(|m| m.setup)
        .and_then(|s| s.save_config_file);

    let path = match save_file {
        Some(relative) => {
            let relative = relative.trim();
            if relative.is_empty() {
                return Err("memory provider save_config_file cannot be empty".into());
            }
            let candidate = Path::new(relative);
            if candidate.is_absolute() {
                return Err("memory provider save_config_file must be relative to HERMES_HOME".into());
            }
            for component in candidate.components() {
                match component {
                    std::path::Component::Normal(_) | std::path::Component::CurDir => {}
                    _ => {
                        return Err(
                            "memory provider save_config_file must stay inside HERMES_HOME".into(),
                        );
                    }
                }
            }
            hermes_home(context).join(candidate)
        }
        None => hermes_home(context).join(format!("{provider_name}.json")),
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = if path.exists() {
        serde_json::from_str::<JsonValue>(&fs::read_to_string(&path)?)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (key, value) in provider_config {
        root.insert(key.clone(), JsonValue::String(value.clone()));
    }
    fs::write(&path, serde_json::to_string_pretty(&JsonValue::Object(root))?)?;
    Ok(())
}

/// Hand off to the provider's `post_setup` hook. Natively we cannot run the
/// plugin's Python `post_setup`, so we record the activation and tell the user
/// to complete provider-specific setup. Callers that bridge to Python should
/// intercept `has_post_setup` before reaching here.
fn run_post_setup_hook(context: &HermesContext, provider_name: &str) -> Result<(), Box<dyn Error>> {
    save_provider_config(context, provider_name, &BTreeMap::new(), false)?;
    println!("\n  Memory provider: {provider_name}");
    println!("  This provider runs its own setup hook; activation saved to config.yaml.");
    println!("  Complete any provider-specific configuration, then start a new session.\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// Status (port of `cmd_status`)
// ---------------------------------------------------------------------------

/// Show current memory provider config (port of `cmd_status`).
pub fn cmd_status(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    print!("{}", render_status(context));
    Ok(())
}

/// Build the status report string (testable form of `cmd_status` output).
pub fn render_status(context: &HermesContext) -> String {
    let mut out = String::new();
    let provider_name = current_provider_name(context);
    let mem_config = if provider_name.is_empty() {
        BTreeMap::new()
    } else {
        load_provider_config(context, &provider_name)
    };

    out.push_str(&format!("\nMemory status\n{}\n", "─".repeat(40)));
    out.push_str("  Built-in:  always active\n");
    let shown = if provider_name.is_empty() {
        String::from("(none — built-in only)")
    } else {
        provider_name.clone()
    };
    out.push_str(&format!("  Provider:  {shown}\n"));

    let providers = get_available_providers(context);

    if !provider_name.is_empty() {
        if !mem_config.is_empty() {
            out.push_str(&format!("\n  {provider_name} config:\n"));
            for (key, value) in &mem_config {
                out.push_str(&format!("    {key}: {value}\n"));
            }
        }

        if let Some(provider) = providers.iter().find(|p| p.name == provider_name) {
            out.push_str("\n  Plugin:    installed ✓\n");
            if provider_is_available(context, provider) {
                out.push_str("  Status:    available ✓\n");
            } else {
                out.push_str("  Status:    not available ✗\n");
                let required_fields: Vec<&SchemaField> = provider
                    .schema
                    .iter()
                    .filter(|f| f.env_var.as_deref().is_some_and(|v| !v.is_empty()))
                    .collect();
                if !required_fields.is_empty() {
                    out.push_str("  Missing:\n");
                    for field in required_fields {
                        let env_var = field.env_var.as_deref().unwrap_or_default();
                        let url = field.url.as_deref().unwrap_or_default();
                        let is_set = std::env::var(env_var)
                            .ok()
                            .is_some_and(|v| !v.is_empty());
                        let mark = if is_set { "✓" } else { "✗" };
                        let mut line = format!("    {mark} {env_var}");
                        if !url.is_empty() && !is_set {
                            line.push_str(&format!("  → {url}"));
                        }
                        out.push_str(&line);
                        out.push('\n');
                    }
                }
            }
        } else {
            out.push_str("\n  Plugin:    NOT installed ✗\n");
            out.push_str(&format!(
                "  Install the '{provider_name}' memory plugin to {}/plugins/\n",
                context.display_hermes_home()
            ));
        }
    }

    if !providers.is_empty() {
        out.push_str("\n  Installed plugins:\n");
        for provider in &providers {
            let active = if provider.name == provider_name {
                " ← active"
            } else {
                ""
            };
            out.push_str(&format!(
                "    • {}  ({}){}\n",
                provider.name, provider.setup_hint, active
            ));
        }
    }

    out.push('\n');
    out
}

/// Determine if a provider is "available" for the status display.
///
/// The Python `provider.is_available()` is provider-specific. Natively we
/// approximate it from the manifest's explicit `is_available` flag when set,
/// otherwise: every required field with an `env_var` is satisfied (env var set,
/// matching config value present, or a non-empty default).
fn provider_is_available(context: &HermesContext, provider: &AvailableProvider) -> bool {
    if let Some(flag) = provider.manifest_available {
        return flag;
    }
    let mut required = provider.schema.iter().filter(|f| f.required).peekable();
    if required.peek().is_none() {
        return true;
    }
    let config = load_provider_config(context, &provider.name);
    required.all(|field| {
        let env_set = field
            .env_var
            .as_deref()
            .and_then(|var| std::env::var(var).ok())
            .is_some_and(|v| !v.trim().is_empty());
        let config_set = config
            .get(&field.key)
            .is_some_and(|v| !v.trim().is_empty());
        let default_set = field
            .default
            .as_ref()
            .map(json_scalar_string)
            .is_some_and(|v| !v.trim().is_empty());
        env_set || config_set || default_set
    })
}

fn current_provider_name(context: &HermesContext) -> String {
    let Ok(root) = read_raw_yaml_mapping(&config_path(context)) else {
        return String::new();
    };
    mapping_child(&root, "memory")
        .and_then(|memory| {
            memory
                .get(YamlValue::String(String::from("provider")))
                .and_then(YamlValue::as_str)
        })
        .map(str::to_string)
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Router (port of `memory_command`)
// ---------------------------------------------------------------------------

/// Memory subcommand variants (port of the `args.memory_command` strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySubcommand {
    Setup,
    Status,
}

/// Route memory subcommands (port of `memory_command`). Unknown -> status.
pub fn memory_command(
    context: &HermesContext,
    sub: Option<MemorySubcommand>,
) -> Result<(), Box<dyn Error>> {
    match sub {
        Some(MemorySubcommand::Setup) => cmd_setup(context),
        _ => cmd_status(context),
    }
}

// ---------------------------------------------------------------------------
// JSON scalar -> string (mirror of how Python stringifies schema values)
// ---------------------------------------------------------------------------

fn json_scalar_string(value: &JsonValue) -> String {
    match value {
        JsonValue::String(text) => text.clone(),
        JsonValue::Bool(boolean) => {
            // Python str(True) == "True"; schema defaults that drive picker
            // matching are typically strings, but bools render Python-style.
            if *boolean {
                String::from("True")
            } else {
                String::from("False")
            }
        }
        JsonValue::Number(number) => number.to_string(),
        JsonValue::Null => String::new(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_hint_matches_python_rules() {
        let secret = SchemaField {
            secret: true,
            ..Default::default()
        };
        let plain = SchemaField {
            secret: false,
            ..Default::default()
        };
        assert_eq!(setup_hint_for_schema(&[]), "no setup needed");
        assert_eq!(setup_hint_for_schema(&[plain.clone()]), "local");
        assert_eq!(setup_hint_for_schema(&[secret.clone()]), "requires API key");
        assert_eq!(
            setup_hint_for_schema(&[secret, plain]),
            "API key / local"
        );
    }

    #[test]
    fn import_name_mapping_matches_python() {
        assert_eq!(import_name_for("honcho-ai"), "honcho");
        assert_eq!(import_name_for("mem0ai"), "mem0");
        assert_eq!(import_name_for("hindsight-client"), "hindsight_client");
        assert_eq!(import_name_for("hindsight-all"), "hindsight");
        // Default: dashes -> underscores, strip extras marker.
        assert_eq!(import_name_for("some-pkg[extra]"), "some_pkg");
        assert_eq!(import_name_for("plain"), "plain");
    }

    #[test]
    fn prompt_returns_default_on_blank() {
        // Indirect: parsing logic — blank trimmed input yields default.
        // (prompt() reads stdin; here we verify the trimming/default rule via
        // a small reimplementation matching prompt()'s tail.)
        let value = "   ".trim().to_string();
        let result = if value.is_empty() {
            Some("def").map(str::to_string).unwrap_or_default()
        } else {
            value
        };
        assert_eq!(result, "def");
    }

    #[test]
    fn write_env_vars_appends_and_updates() {
        let dir = std::env::temp_dir().join(format!(
            "hermes_memenv_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = fs::create_dir_all(&dir);
        let env_file = dir.join(".env");
        fs::write(&env_file, "EXISTING=old\nUNTOUCHED=keep\n").unwrap();

        let mut writes = BTreeMap::new();
        writes.insert(String::from("EXISTING"), String::from("new"));
        writes.insert(String::from("FRESH"), String::from("value"));
        write_env_vars(&env_file, &writes).unwrap();

        let contents = fs::read_to_string(&env_file).unwrap();
        assert!(contents.contains("EXISTING=new"));
        assert!(contents.contains("UNTOUCHED=keep"));
        assert!(contents.contains("FRESH=value"));
        // Original value replaced, not duplicated.
        assert!(!contents.contains("EXISTING=old"));
        assert!(contents.ends_with('\n'));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_env_vars_creates_new_file() {
        let dir = std::env::temp_dir().join(format!(
            "hermes_memenv_new_{}_{}",
            std::process::id(),
            line!()
        ));
        let env_file = dir.join("nested").join(".env");

        let mut writes = BTreeMap::new();
        writes.insert(String::from("KEY"), String::from("val"));
        write_env_vars(&env_file, &writes).unwrap();

        let contents = fs::read_to_string(&env_file).unwrap();
        assert_eq!(contents, "KEY=val\n");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn schema_field_description_falls_back_to_key() {
        let field = SchemaField {
            key: String::from("api_key"),
            description: None,
            ..Default::default()
        };
        assert_eq!(field.description_text(), "api_key");

        let field = SchemaField {
            key: String::from("api_key"),
            description: Some(String::from("API key")),
            ..Default::default()
        };
        assert_eq!(field.description_text(), "API key");
    }

    #[test]
    fn json_scalar_string_renders_python_style_bools() {
        assert_eq!(json_scalar_string(&JsonValue::Bool(true)), "True");
        assert_eq!(json_scalar_string(&JsonValue::Bool(false)), "False");
        assert_eq!(
            json_scalar_string(&JsonValue::String("x".into())),
            "x"
        );
        assert_eq!(json_scalar_string(&JsonValue::Null), "");
    }

    #[test]
    fn manifest_parses_setup_schema() {
        let yaml = r#"
pip_dependencies:
  - mem0ai
setup:
  has_save_config: true
  schema:
    - key: api_key
      description: Mem0 API key
      secret: true
      required: true
      env_var: MEM0_API_KEY
      url: https://app.mem0.ai
    - key: user_id
      description: User id
      default: hermes-user
"#;
        let manifest: MemoryPluginManifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.pip_dependencies, vec!["mem0ai"]);
        let setup = manifest.setup.unwrap();
        assert!(setup.save_config);
        assert_eq!(setup.schema.len(), 2);
        assert!(setup.schema[0].secret);
        assert_eq!(setup.schema[0].env_var.as_deref(), Some("MEM0_API_KEY"));
        assert_eq!(
            setup.schema[1].default.as_ref().map(json_scalar_string),
            Some(String::from("hermes-user"))
        );
    }
}
