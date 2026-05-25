use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;

use crate::{ToolDefinition, ToolRuntime, tool_error};

#[derive(Debug, Deserialize)]
struct BridgeResponse<T> {
    ok: bool,
    result: Option<T>,
    error: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct PluginCliDispatchResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone)]
struct PythonPluginBridge {
    python: PathBuf,
    root: PathBuf,
    cwd: PathBuf,
    hermes_home: PathBuf,
    state: Arc<Mutex<BridgeState>>,
}

#[derive(Debug, Default)]
struct BridgeState {
    process: Option<BridgeProcess>,
}

#[derive(Debug)]
struct BridgeProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<String>>,
    stderr_thread: Option<JoinHandle<()>>,
}

impl BridgeProcess {
    fn new(child: Child, stdin: ChildStdin, stdout: ChildStdout, stderr: ChildStderr) -> Self {
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        let stderr_thread = Some(spawn_stderr_collector(stderr, Arc::clone(&stderr_tail)));
        Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_tail,
            stderr_thread,
        }
    }

    fn send_request(&mut self, request: &str) -> Result<String, String> {
        self.stdin
            .write_all(request.as_bytes())
            .map_err(|error| format!("failed to write plugin runtime payload: {error}"))?;
        self.stdin
            .write_all(b"\n")
            .map_err(|error| format!("failed to terminate plugin runtime payload: {error}"))?;
        self.stdin
            .flush()
            .map_err(|error| format!("failed to flush plugin runtime payload: {error}"))?;
        let mut response_line = String::new();
        let read = self
            .stdout
            .read_line(&mut response_line)
            .map_err(|error| format!("failed to read plugin runtime bridge output: {error}"))?;
        if read == 0 {
            return Err("plugin runtime bridge exited unexpectedly".to_string());
        }
        Ok(response_line)
    }

    fn shutdown_with_message(&mut self, message: String) -> String {
        let stderr = self.stderr_snapshot();
        self.shutdown();
        if stderr.is_empty() {
            message
        } else {
            format!("{message}; stderr={stderr}")
        }
    }

    fn stderr_snapshot(&self) -> String {
        self.stderr_tail
            .lock()
            .ok()
            .map(|tail| tail.trim().to_string())
            .unwrap_or_default()
    }

    fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BridgeState {
    fn drop(&mut self) {
        if let Some(process) = self.process.as_mut() {
            process.shutdown();
        }
    }
}

impl PythonPluginBridge {
    fn new(hermes_home: &Path, cwd: &Path) -> Result<Self, String> {
        let root = project_root();
        let python =
            resolve_repo_python(&root, Some("HERMES_PLUGIN_RUNTIME_PYTHON")).ok_or_else(|| {
                "could not find a Python interpreter for plugin runtime dispatch".to_string()
            })?;
        Ok(Self {
            python,
            root,
            cwd: cwd.to_path_buf(),
            hermes_home: hermes_home.to_path_buf(),
            state: Arc::new(Mutex::new(BridgeState::default())),
        })
    }

    fn dispatch_tool(
        &self,
        tool_name: &str,
        args: &Value,
        runtime: &ToolRuntime,
    ) -> Result<String, String> {
        self.run(
            "dispatch",
            &json!({
                "tool_name": tool_name,
                "args": args,
                "task_id": runtime.current_session_id().unwrap_or_default(),
                "session_id": runtime.current_session_id().unwrap_or_default(),
                "enabled_tools": runtime
                    .available_tool_names()
                    .map(|names| names.iter().cloned().collect::<Vec<_>>())
                    .unwrap_or_default(),
            }),
        )
    }

    fn invoke_hook(&self, hook_name: &str, payload: &Value) -> Result<Vec<Value>, String> {
        self.run(
            "hook",
            &json!({
                "hook_name": hook_name,
                "payload": payload,
            }),
        )
    }

    fn run_cli_command(&self, argv: &[String]) -> Result<PluginCliDispatchResult, String> {
        self.run(
            "cli",
            &json!({
                "argv": argv,
            }),
        )
    }

    fn run<T: for<'de> Deserialize<'de>>(
        &self,
        action: &str,
        payload: &Value,
    ) -> Result<T, String> {
        let request = serde_json::to_string(&json!({
            "action": action,
            "payload": payload,
        }))
        .map_err(|error| format!("failed to serialize plugin runtime payload: {error}"))?;
        for attempt in 0..2 {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "plugin runtime bridge lock poisoned".to_string())?;
            if state.process.is_none() {
                state.process = Some(self.start_process()?);
            }
            let Some(process) = state.process.as_mut() else {
                return Err("plugin runtime bridge failed to start".to_string());
            };

            let response_line = match process.send_request(&request) {
                Ok(line) => line,
                Err(error) => {
                    let detail = process.shutdown_with_message(error);
                    state.process = None;
                    if attempt == 0 {
                        continue;
                    }
                    return Err(detail);
                }
            };

            let response = serde_json::from_str::<BridgeResponse<T>>(response_line.trim_end())
                .map_err(|error| {
                    format!(
                        "failed to decode plugin runtime bridge output: {error}; stdout={}",
                        response_line.trim()
                    )
                })?;
            if response.ok {
                return response
                    .result
                    .ok_or_else(|| "plugin runtime bridge returned no result".to_string());
            }
            return Err(
                if response
                    .error
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty()
                {
                    "plugin runtime bridge returned an unspecified error".to_string()
                } else {
                    response.error.unwrap_or_default()
                },
            );
        }
        Err("plugin runtime bridge failed after restart".to_string())
    }

    fn start_process(&self) -> Result<BridgeProcess, String> {
        let mut child = Command::new(&self.python)
            .current_dir(&self.cwd)
            .env("HERMES_HOME", &self.hermes_home)
            .env("PYTHONPATH", pythonpath_with_root(&self.root))
            .arg("-u")
            .arg("-c")
            .arg(PYTHON_PLUGIN_RUNTIME_BRIDGE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to start plugin runtime bridge: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "plugin runtime bridge stdin unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "plugin runtime bridge stdout unavailable".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "plugin runtime bridge stderr unavailable".to_string())?;
        Ok(BridgeProcess::new(child, stdin, stdout, stderr))
    }
}

const PYTHON_PLUGIN_RUNTIME_BRIDGE: &str = r#"
import json
import os
import sys
import traceback

from hermes_cli.plugins import discover_plugins, get_plugin_manager, invoke_hook
from tools.registry import registry


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()

discover_plugins()
manager = get_plugin_manager()

def _collect_cli_commands():
    commands = dict(getattr(manager, "_cli_commands", {}) or {})
    try:
        from plugins.memory import discover_plugin_cli_commands
        for info in discover_plugin_cli_commands() or []:
            name = str(info.get("name", "")).strip()
            if name:
                commands[name] = info
    except Exception:
        pass
    return commands

for raw in sys.stdin:
    text = raw.strip()
    if not text:
        continue
    try:
        request = json.loads(text)
        action = str(request.get("action", "")).strip()
        payload = request.get("payload") or {}
        if action == "dispatch":
            emit({
                "ok": True,
                "result": registry.dispatch(
                    payload.get("tool_name", ""),
                    payload.get("args") or {},
                    task_id=payload.get("task_id"),
                    session_id=payload.get("session_id"),
                    enabled_tools=payload.get("enabled_tools"),
                )
            })
        elif action == "hook":
            emit({
                "ok": True,
                "result": invoke_hook(
                    payload.get("hook_name", ""),
                    **(payload.get("payload") or {}),
                )
            })
        elif action == "cli":
            import argparse
            import contextlib
            import io

            argv = payload.get("argv") or []
            if not isinstance(argv, list) or not argv:
                raise ValueError("plugin CLI argv must be a non-empty list")
            argv = [str(item) for item in argv]
            command_name = argv[0].strip()
            commands = _collect_cli_commands()
            info = commands.get(command_name)
            if not info:
                raise ValueError(f"unknown plugin CLI command: {command_name}")
            parser = argparse.ArgumentParser(prog=f"hermes {command_name}")
            info["setup_fn"](parser)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            exit_code = 0
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    parsed = parser.parse_args(argv[1:])
                    handler = info.get("handler_fn") or getattr(parsed, "func", None)
                    if handler is None:
                        raise ValueError(f"plugin CLI command '{command_name}' has no handler")
                    result = handler(parsed)
                    if isinstance(result, bool):
                        exit_code = 0 if result else 1
                    elif isinstance(result, int):
                        exit_code = result
                    elif result is None:
                        exit_code = 0
                    else:
                        exit_code = 0
                except SystemExit as exc:
                    code = exc.code
                    exit_code = code if isinstance(code, int) else 1
            emit({
                "ok": True,
                "result": {
                    "exit_code": int(exit_code),
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue(),
                }
            })
        else:
            raise ValueError(f"unknown action: {action}")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({
            "ok": False,
            "error": str(exc),
        })
"#;

const PYTHON_PLUGIN_PLATFORM_SETUP: &str = concat!(
    "import os\n",
    "from gateway.platform_registry import platform_registry\n",
    "from hermes_cli.plugins import discover_plugins\n",
    "discover_plugins()\n",
    "target = os.environ['HERMES_GATEWAY_SETUP_PLATFORM'].strip()\n",
    "entry = platform_registry.get(target)\n",
    "if entry is None:\n",
    "    raise SystemExit(f'unknown plugin platform: {target}')\n",
    "if entry.source != 'plugin':\n",
    "    raise SystemExit(f'platform is not plugin-backed: {target}')\n",
    "if entry.setup_fn is None:\n",
    "    raise SystemExit(f'plugin platform has no setup function: {target}')\n",
    "entry.setup_fn()\n",
);

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn resolve_repo_python(project_root: &Path, override_env_var: Option<&str>) -> Option<PathBuf> {
    if let Some(env_var) = override_env_var
        && let Some(value) = std::env::var(env_var).ok()
        && !value.trim().is_empty()
    {
        return Some(PathBuf::from(value.trim()));
    }

    let candidates = [
        project_root.join(".venv").join(python_bin_name()),
        project_root.join("venv").join(python_bin_name()),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
            .join("hermes-agent")
            .join("venv")
            .join(python_bin_name()),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    which_on_path("python3").or_else(|| which_on_path("python"))
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn python_bin_name() -> &'static str {
    #[cfg(windows)]
    {
        "Scripts/python.exe"
    }
    #[cfg(not(windows))]
    {
        "bin/python"
    }
}

fn pythonpath_with_root(root: &Path) -> OsString {
    let mut paths = vec![root.to_path_buf()];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_else(|_| root.as_os_str().to_os_string())
}

fn spawn_stderr_collector(stderr: ChildStderr, stderr_tail: Arc<Mutex<String>>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or_default() > 0 {
            if let Ok(mut tail) = stderr_tail.lock() {
                tail.push_str(line.trim_end());
                tail.push('\n');
                if tail.len() > 8_192 {
                    let keep_from = tail.len().saturating_sub(8_192);
                    let trimmed = tail[keep_from..].to_string();
                    *tail = trimmed;
                }
            }
            line.clear();
        }
    })
}

pub fn attach_python_plugin_runtime(
    hermes_home: &Path,
    runtime: ToolRuntime,
) -> Result<ToolRuntime, String> {
    let plugins = discover_enabled_general_plugins(hermes_home, runtime.cwd());
    if plugins.is_empty() {
        return Ok(runtime);
    }
    let mut toolsets = BTreeMap::<String, Vec<String>>::new();
    let mut definitions = Vec::new();
    let mut dynamic_tool_names = BTreeSet::new();
    let mut active_hook_names = BTreeSet::new();
    for plugin in plugins {
        for hook in plugin.provides_hooks {
            active_hook_names.insert(hook);
        }
        for definition in plugin.tool_definitions {
            if dynamic_tool_names.insert(definition.name.clone()) {
                toolsets
                    .entry(definition.toolset.clone())
                    .or_default()
                    .push(definition.name.clone());
                definitions.push(definition);
            }
        }
    }
    let runtime = runtime.with_dynamic_tools(definitions, toolsets);
    attach_python_plugin_callbacks(hermes_home, runtime, dynamic_tool_names, active_hook_names)
}

pub fn dispatch_python_plugin_cli_command(
    hermes_home: &Path,
    cwd: &Path,
    argv: &[String],
) -> Result<PluginCliDispatchResult, String> {
    PythonPluginBridge::new(hermes_home, cwd)?.run_cli_command(argv)
}

pub fn run_python_plugin_platform_setup(
    hermes_home: &Path,
    cwd: &Path,
    platform_key: &str,
    accept_hooks: bool,
) -> Result<(), String> {
    let key = platform_key.trim();
    if key.is_empty() {
        return Err("plugin platform key cannot be empty".to_string());
    }

    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON")).ok_or_else(|| {
        "could not find a Python interpreter for plugin platform setup".to_string()
    })?;
    let mut command = Command::new(&python);
    command
        .current_dir(cwd)
        .env("HERMES_HOME", hermes_home)
        .env("PYTHONPATH", pythonpath_with_root(&root))
        .env("HERMES_GATEWAY_SETUP_PLATFORM", key)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }

    let status = command
        .arg("-c")
        .arg(PYTHON_PLUGIN_PLATFORM_SETUP)
        .status()
        .map_err(|error| format!("failed to start plugin platform setup: {error}"))?;
    if status.success() {
        return Ok(());
    }
    Err(match status.code() {
        Some(code) => format!("plugin platform setup exited with status {code}"),
        None => "plugin platform setup terminated by signal".to_string(),
    })
}

pub fn attach_python_plugin_callbacks(
    hermes_home: &Path,
    runtime: ToolRuntime,
    dynamic_tool_names: BTreeSet<String>,
    active_hook_names: BTreeSet<String>,
) -> Result<ToolRuntime, String> {
    let bridge = Arc::new(PythonPluginBridge::new(hermes_home, runtime.cwd())?);
    attach_python_plugin_callbacks_with_bridge(
        runtime,
        bridge,
        dynamic_tool_names,
        active_hook_names,
    )
}

fn attach_python_plugin_callbacks_with_bridge(
    runtime: ToolRuntime,
    bridge: Arc<PythonPluginBridge>,
    dynamic_tool_names: BTreeSet<String>,
    active_hook_names: BTreeSet<String>,
) -> Result<ToolRuntime, String> {
    let bridge_for_dispatch = bridge.clone();
    let bridge_for_hook = bridge;
    Ok(runtime
        .with_dynamic_tool_dispatch(move |name, args, runtime| {
            if !dynamic_tool_names.contains(name) {
                return None;
            }
            Some(
                bridge_for_dispatch
                    .dispatch_tool(name, args, runtime)
                    .unwrap_or_else(tool_error),
            )
        })
        .with_hook_invoke_callback(move |hook_name, payload, _runtime| {
            if !active_hook_names.contains(hook_name) {
                return Vec::new();
            }
            bridge_for_hook
                .invoke_hook(hook_name, payload)
                .unwrap_or_default()
        }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PluginSource {
    Bundled,
    User,
    Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PluginKind {
    Standalone,
    Backend,
    Exclusive,
    Platform,
    ModelProvider,
    ContextEngine,
}

impl PluginKind {
    fn is_auto_enabled(self, source: PluginSource) -> bool {
        source == PluginSource::Bundled && matches!(self, Self::Backend | Self::Platform)
    }
}

#[derive(Debug, Clone)]
struct DiscoveredPlugin {
    key: String,
    name: String,
    kind: PluginKind,
    source: PluginSource,
    provides_hooks: Vec<String>,
    tool_definitions: Vec<ToolDefinition>,
}

fn discover_enabled_general_plugins(hermes_home: &Path, cwd: &Path) -> Vec<DiscoveredPlugin> {
    let enabled = load_plugin_set(hermes_home, "enabled");
    let disabled = load_plugin_set(hermes_home, "disabled");
    let mut winners = BTreeMap::<String, DiscoveredPlugin>::new();
    for (root, source) in [
        (bundled_plugins_dir(), PluginSource::Bundled),
        (user_plugins_dir(hermes_home), PluginSource::User),
        (project_plugins_dir(cwd), PluginSource::Project),
    ] {
        if !root.is_dir() {
            continue;
        }
        for plugin in scan_plugin_tree(&root, source) {
            winners.insert(plugin.key.clone(), plugin);
        }
    }
    winners
        .into_values()
        .filter(|plugin| {
            !matches!(
                plugin.kind,
                PluginKind::Exclusive | PluginKind::ModelProvider | PluginKind::ContextEngine
            )
        })
        .filter(|plugin| {
            if disabled.contains(&plugin.name) || disabled.contains(&plugin.key) {
                return false;
            }
            if plugin.kind.is_auto_enabled(plugin.source) {
                return true;
            }
            enabled.contains(&plugin.name) || enabled.contains(&plugin.key)
        })
        .collect()
}

fn load_plugin_set(hermes_home: &Path, key: &str) -> BTreeSet<String> {
    let config_path = hermes_home.join("config.yaml");
    let text = match fs::read_to_string(config_path) {
        Ok(text) => text,
        Err(_) => return BTreeSet::new(),
    };
    let value = match serde_yaml::from_str::<YamlValue>(&text) {
        Ok(value) => value,
        Err(_) => return BTreeSet::new(),
    };
    value
        .as_mapping()
        .and_then(|mapping| mapping.get(YamlValue::String(String::from("plugins"))))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(YamlValue::String(key.to_string())))
        .and_then(YamlValue::as_sequence)
        .map(|items| {
            items
                .iter()
                .filter_map(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default()
}

fn bundled_plugins_dir() -> PathBuf {
    if let Some(override_path) = std::env::var_os("HERMES_BUNDLED_PLUGINS")
        && !override_path.is_empty()
    {
        return PathBuf::from(override_path);
    }
    project_root().join("plugins")
}

fn user_plugins_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("plugins")
}

fn project_plugins_dir(cwd: &Path) -> PathBuf {
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS") {
        cwd.join(".hermes").join("plugins")
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

fn scan_plugin_tree(root: &Path, source: PluginSource) -> Vec<DiscoveredPlugin> {
    let mut plugins = Vec::new();
    scan_plugin_tree_level(root, source, "", 0, &mut plugins);
    plugins
}

fn scan_plugin_tree_level(
    root: &Path,
    source: PluginSource,
    prefix: &str,
    depth: usize,
    out: &mut Vec<DiscoveredPlugin>,
) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if let Some(manifest_path) = plugin_manifest_path(&path) {
            if let Some(plugin) = parse_plugin_manifest(&manifest_path, &path, prefix, source) {
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
        scan_plugin_tree_level(&path, source, &next_prefix, depth + 1, out);
    }
}

fn plugin_manifest_path(path: &Path) -> Option<PathBuf> {
    let yaml = path.join("plugin.yaml");
    if yaml.exists() {
        return Some(yaml);
    }
    let yml = path.join("plugin.yml");
    yml.exists().then_some(yml)
}

fn parse_plugin_manifest(
    manifest_path: &Path,
    plugin_dir: &Path,
    prefix: &str,
    source: PluginSource,
) -> Option<DiscoveredPlugin> {
    let text = fs::read_to_string(manifest_path).ok()?;
    let parsed = serde_yaml::from_str::<YamlValue>(&text).unwrap_or(YamlValue::Null);
    let mapping = parsed.as_mapping()?;
    let dir_name = plugin_dir.file_name()?.to_string_lossy().to_string();
    let key = if prefix.is_empty() {
        dir_name.clone()
    } else {
        format!("{prefix}/{dir_name}")
    };
    let name = mapping_string(mapping, "name").unwrap_or_else(|| dir_name.clone());
    if name.trim().is_empty() {
        return None;
    }
    let source_text = read_plugin_source_files(plugin_dir);
    Some(DiscoveredPlugin {
        key: key.clone(),
        name,
        kind: determine_plugin_kind(&key, mapping_string(mapping, "kind"), &source_text),
        source,
        provides_hooks: discover_hook_registrations_from_source(&source_text),
        tool_definitions: discover_tool_definitions_from_source(&source_text),
    })
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
    files
        .into_iter()
        .filter_map(|file| fs::read_to_string(file).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

fn mapping_string(mapping: &serde_yaml::Mapping, key: &str) -> Option<String> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

pub fn discover_tool_definitions_from_source(source_text: &str) -> Vec<ToolDefinition> {
    let mut tools = discover_direct_tool_registrations(source_text);
    if tools.is_empty() {
        tools = discover_tuple_registered_tools(source_text);
    }
    tools
}

pub fn discover_hook_registrations_from_source(source_text: &str) -> Vec<String> {
    let mut hooks = Vec::new();
    let mut blocks = extract_call_blocks(source_text, "ctx.register_hook");
    if blocks.is_empty() {
        blocks = extract_call_blocks(source_text, "register_hook");
    }
    for block in blocks {
        let Some(name) = extract_first_string_argument(&block) else {
            continue;
        };
        if !hooks.iter().any(|existing| existing == &name) {
            hooks.push(name);
        }
    }
    hooks
}

fn discover_direct_tool_registrations(source_text: &str) -> Vec<ToolDefinition> {
    let schema_literals = extract_schema_assignments(source_text);
    let mut definitions = Vec::new();
    let mut blocks = extract_call_blocks(source_text, "ctx.register_tool");
    if blocks.is_empty() {
        blocks = extract_call_blocks(source_text, "register_tool");
    }
    for block in blocks {
        let Some(name) = extract_keyword_string(&block, "name") else {
            continue;
        };
        let Some(toolset) = extract_keyword_string(&block, "toolset") else {
            continue;
        };
        let emoji = extract_keyword_string(&block, "emoji").unwrap_or_default();
        let description = extract_keyword_string(&block, "description")
            .or_else(|| {
                extract_schema_literal(&block, &schema_literals)
                    .as_ref()
                    .and_then(schema_description)
            })
            .unwrap_or_default();
        let schema = extract_schema_literal(&block, &schema_literals).unwrap_or_else(|| {
            json!({
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}},
            })
        });
        definitions.push(ToolDefinition {
            name,
            toolset,
            description,
            emoji,
            schema,
        });
    }
    definitions
}

fn discover_tuple_registered_tools(source_text: &str) -> Vec<ToolDefinition> {
    let Some(tuple_name) = extract_loop_tuple_name(source_text) else {
        return Vec::new();
    };
    let entries = extract_tuple_entries(source_text, &tuple_name);
    let schema_literals = extract_schema_assignments(source_text);
    let mut definitions = Vec::new();
    for entry in entries {
        let Some(name) = entry.first().cloned() else {
            continue;
        };
        let Some(schema_name) = entry.get(1).cloned() else {
            continue;
        };
        let emoji = entry.get(3).cloned().unwrap_or_default();
        let Some(toolset) = extract_keyword_string_in_loop(source_text, "toolset") else {
            continue;
        };
        let schema = schema_literals
            .get(&schema_name)
            .cloned()
            .unwrap_or_else(|| {
                json!({
                    "name": name,
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                })
            });
        let description = schema_description(&schema).unwrap_or_default();
        definitions.push(ToolDefinition {
            name,
            toolset,
            description,
            emoji,
            schema,
        });
    }
    definitions
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

fn extract_first_string_argument(block: &str) -> Option<String> {
    let regex = Regex::new(r#"(?s)^\s*(?:"([^"]*)"|'([^']*)')"#).ok()?;
    regex
        .captures(block)
        .and_then(|captures| captures.get(1).or_else(|| captures.get(2)))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_schema_literal(block: &str, schema_literals: &BTreeMap<String, Value>) -> Option<Value> {
    let schema_name = extract_keyword_identifier(block, "schema")?;
    schema_literals.get(&schema_name).cloned().or_else(|| {
        extract_keyword_braced_value(block, "schema")
            .and_then(|value| parse_python_mapping_literal(&value))
    })
}

fn extract_schema_assignments(source_text: &str) -> BTreeMap<String, Value> {
    let mut schemas = BTreeMap::new();
    let regex = Regex::new(r#"(?m)^\s*([A-Z][A-Z0-9_]+)\s*(?::[^=]+)?=\s*\{"#).unwrap();
    for capture in regex.captures_iter(source_text) {
        let Some(name_match) = capture.get(1) else {
            continue;
        };
        let name = name_match.as_str().to_string();
        let start = capture.get(0).unwrap().end().saturating_sub(1);
        if let Some(body) = extract_balanced_braces(source_text, start)
            && let Some(value) = parse_python_mapping_literal(&body)
        {
            schemas.insert(name, value);
        }
    }
    schemas
}

fn extract_keyword_identifier(block: &str, key: &str) -> Option<String> {
    let pattern = format!(
        r#"(?m)\b{}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)"#,
        regex::escape(key)
    );
    let regex = Regex::new(&pattern).ok()?;
    regex
        .captures(block)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_keyword_braced_value(block: &str, key: &str) -> Option<String> {
    let pattern = format!(r#"\b{}\s*=\s*\{{"#, regex::escape(key));
    let regex = Regex::new(&pattern).ok()?;
    let found = regex.find(block)?;
    extract_balanced_braces(block, found.end().saturating_sub(1))
}

fn extract_balanced_braces(source: &str, open_index: usize) -> Option<String> {
    let bytes = source.as_bytes();
    if bytes.get(open_index) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut cursor = open_index;
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
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(source[open_index..=cursor].to_string());
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

fn parse_python_mapping_literal(value: &str) -> Option<Value> {
    let mut normalized = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if in_single {
            if escaped {
                normalized.push(ch);
                escaped = false;
                continue;
            }
            match ch {
                '\\' => {
                    normalized.push(ch);
                    escaped = true;
                }
                '\'' => {
                    normalized.push('"');
                    in_single = false;
                }
                '"' => normalized.push_str("\\\""),
                _ => normalized.push(ch),
            }
            continue;
        }
        if in_double {
            normalized.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double = false;
            }
            continue;
        }
        match ch {
            '\'' => {
                normalized.push('"');
                in_single = true;
            }
            '"' => {
                normalized.push(ch);
                in_double = true;
            }
            'T' if chars.clone().take(3).collect::<String>().as_str() == "rue" => {
                normalized.push_str("true");
                chars.next();
                chars.next();
                chars.next();
            }
            'F' if chars.clone().take(4).collect::<String>().as_str() == "alse" => {
                normalized.push_str("false");
                chars.next();
                chars.next();
                chars.next();
                chars.next();
            }
            'N' if chars.clone().take(3).collect::<String>().as_str() == "one" => {
                normalized.push_str("null");
                chars.next();
                chars.next();
                chars.next();
            }
            _ => normalized.push(ch),
        }
    }
    let normalized = Regex::new(r#",(\s*[}\]])"#)
        .ok()
        .map(|regex| regex.replace_all(&normalized, "$1").into_owned())
        .unwrap_or(normalized);
    serde_json::from_str(&normalized).ok()
}

fn schema_description(schema: &Value) -> Option<String> {
    schema
        .as_object()
        .and_then(|object| object.get("description"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn extract_loop_tuple_name(source_text: &str) -> Option<String> {
    let regex = Regex::new(
        r#"for\s+\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s+in\s+([A-Za-z_][A-Za-z0-9_]*)\s*:"#,
    )
    .ok()?;
    regex
        .captures(source_text)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().to_string())
}

fn extract_tuple_entries(source_text: &str, tuple_name: &str) -> Vec<Vec<String>> {
    let pattern = format!(r#"(?m)^\s*{}\s*=\s*\("#, regex::escape(tuple_name));
    let Some(regex) = Regex::new(&pattern).ok() else {
        return Vec::new();
    };
    let Some(found) = regex.find(source_text) else {
        return Vec::new();
    };
    let Some(body) = extract_balanced_parens(source_text, found.end().saturating_sub(1)) else {
        return Vec::new();
    };
    let entry_re = Regex::new(
        r#"(?s)\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*"([^"]*)"\s*\)"#,
    )
    .unwrap();
    entry_re
        .captures_iter(&body)
        .map(|caps| {
            vec![
                caps.get(1).unwrap().as_str().to_string(),
                caps.get(2).unwrap().as_str().to_string(),
                String::new(),
                caps.get(3).unwrap().as_str().to_string(),
            ]
        })
        .collect()
}

fn extract_balanced_parens(source: &str, open_index: usize) -> Option<String> {
    let bytes = source.as_bytes();
    if bytes.get(open_index) != Some(&b'(') {
        return None;
    }
    let mut depth = 0usize;
    let mut cursor = open_index;
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
                    return Some(source[open_index..=cursor].to_string());
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

fn extract_keyword_string_in_loop(source_text: &str, key: &str) -> Option<String> {
    extract_call_blocks(source_text, "ctx.register_tool")
        .into_iter()
        .find_map(|block| extract_keyword_string(&block, key))
}

fn find_bytes(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    haystack[start..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|index| start + index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::get_tool_definitions_with_runtime;

    #[test]
    fn attach_python_plugin_runtime_uses_static_tool_and_hook_discovery() {
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
        handler=lambda args, **kwargs: "{\"success\": true}",
        emoji="plug",
    )

    def pre_llm_call(**kwargs):
        return {"context": "static core hook"}

    ctx.register_hook("pre_llm_call", pre_llm_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo\n",
        )
        .unwrap();

        let runtime = attach_python_plugin_runtime(
            &home,
            ToolRuntime::new(temp.path()).with_hermes_home(&home),
        )
        .unwrap();

        let enabled_toolsets = vec![String::from("demo")];
        let definitions =
            get_tool_definitions_with_runtime(Some(&enabled_toolsets), None, Some(&runtime));
        assert!(definitions.iter().any(|tool| tool.name == "demo_tool"));

        let hook_results = runtime.invoke_hook(
            "pre_llm_call",
            &json!({
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
                .and_then(Value::as_str)
                .is_some_and(|value| value == "static core hook")
        }));
    }

    #[test]
    fn attach_python_plugin_runtime_reuses_persistent_bridge_process() {
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
COUNTER = 0

def register(ctx):
    def pre_llm_call(**kwargs):
        global COUNTER
        COUNTER += 1
        return {"count": COUNTER}

    ctx.register_hook("pre_llm_call", pre_llm_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo\n",
        )
        .unwrap();

        let runtime = attach_python_plugin_runtime(
            &home,
            ToolRuntime::new(temp.path()).with_hermes_home(&home),
        )
        .unwrap();

        let first = runtime.invoke_hook("pre_llm_call", &json!({"session_id": "session-1"}));
        let second = runtime.invoke_hook("pre_llm_call", &json!({"session_id": "session-1"}));
        assert_eq!(first[0].get("count").and_then(Value::as_i64), Some(1));
        assert_eq!(second[0].get("count").and_then(Value::as_i64), Some(2));
        assert!(
            runtime
                .invoke_hook("post_llm_call", &json!({"session_id": "session-1"}))
                .is_empty()
        );
    }

    #[test]
    fn dispatch_python_plugin_cli_command_runs_registered_plugin_command() {
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
def register(ctx):
    def setup_cli(parser):
        parser.add_argument("name")
        parser.add_argument("--warn", action="store_true")

    def handle_cli(args):
        import sys
        if args.warn:
            print(f"warn:{args.name}", file=sys.stderr)
            return 7
        print(f"hello {args.name}")
        return 0

    ctx.register_cli_command(
        name="demo",
        help="demo command",
        description="demo command",
        setup_fn=setup_cli,
        handler_fn=handle_cli,
    )
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo\n",
        )
        .unwrap();

        let success = dispatch_python_plugin_cli_command(
            &home,
            temp.path(),
            &[String::from("demo"), String::from("alice")],
        )
        .unwrap();
        assert_eq!(success.exit_code, 0);
        assert_eq!(success.stdout.trim(), "hello alice");
        assert!(success.stderr.trim().is_empty());

        let failure = dispatch_python_plugin_cli_command(
            &home,
            temp.path(),
            &[
                String::from("demo"),
                String::from("bob"),
                String::from("--warn"),
            ],
        )
        .unwrap();
        assert_eq!(failure.exit_code, 7);
        assert!(failure.stdout.trim().is_empty());
        assert_eq!(failure.stderr.trim(), "warn:bob");
    }
}
