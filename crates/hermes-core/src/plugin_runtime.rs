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

use crate::tools::ALWAYS_DYNAMIC_TOOLSET;
use crate::{ToolDefinition, ToolRuntime, tool_error};

const CREDENTIAL_SUFFIXES: [&str; 4] = ["_API_KEY", "_TOKEN", "_SECRET", "_KEY"];

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCliCommand {
    pub name: String,
    pub help: String,
    pub description: String,
    pub plugin_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformSurface {
    pub key: String,
    pub label: String,
    pub required_env: Vec<String>,
    pub install_hint: Option<String>,
    pub emoji: String,
    pub has_setup_fn: bool,
    pub plugin_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashboardSurface {
    pub name: String,
    pub label: String,
    pub description: String,
    pub icon: String,
    pub version: String,
    pub entry: String,
    pub css: Option<String>,
    pub api: Option<String>,
    pub slots: Vec<String>,
    pub tab_path: String,
    pub tab_position: String,
    pub tab_override: Option<String>,
    pub tab_hidden: bool,
    pub source: PluginSource,
    pub dashboard_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageGenProviderEnvVar {
    pub key: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub default: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageGenProviderModel {
    pub id: String,
    #[serde(default)]
    pub display: String,
    #[serde(default)]
    pub speed: String,
    #[serde(default)]
    pub strengths: String,
    #[serde(default)]
    pub price: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageGenPluginProvider {
    pub plugin_name: String,
    pub name: String,
    #[serde(default)]
    pub badge: String,
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub env_vars: Vec<ImageGenProviderEnvVar>,
    #[serde(default)]
    pub models: Vec<ImageGenProviderModel>,
    #[serde(default)]
    pub default_model: Option<String>,
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
        let mut command = Command::new(&self.python);
        command
            .current_dir(&self.cwd)
            .env("HERMES_HOME", &self.hermes_home)
            .env("PYTHONPATH", pythonpath_with_root(&self.root))
            .arg("-u")
            .arg("-c")
            .arg(PYTHON_PLUGIN_RUNTIME_BRIDGE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_hermes_env_file(&mut command, &self.hermes_home);
        let mut child = command
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

def _resolve_image_gen_provider():
    from agent.image_gen_registry import get_provider
    from hermes_cli.config import load_config

    configured = None
    try:
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("provider")
            if isinstance(value, str) and value.strip():
                configured = value.strip()
    except Exception:
        configured = None
    if not configured or configured == "fal":
        return configured, None

    provider = get_provider(configured)
    if provider is None:
        discover_plugins(force=True)
        provider = get_provider(configured)
    return configured, provider

_SELECTED_CONTEXT_ENGINE_LOADED = False
_SELECTED_CONTEXT_ENGINE = None

def _load_selected_context_engine():
    global _SELECTED_CONTEXT_ENGINE_LOADED, _SELECTED_CONTEXT_ENGINE
    if _SELECTED_CONTEXT_ENGINE_LOADED:
        return _SELECTED_CONTEXT_ENGINE

    engine_name = "compressor"
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("context") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("engine")
            if isinstance(value, str) and value.strip():
                engine_name = value.strip()
    except Exception:
        pass

    if not engine_name or engine_name == "compressor":
        return None

    engine = None
    try:
        from plugins.context_engine import load_context_engine
        engine = load_context_engine(engine_name)
    except Exception:
        engine = None

    if engine is None:
        try:
            from hermes_cli.plugins import get_plugin_context_engine
            candidate = get_plugin_context_engine()
            if candidate is not None and getattr(candidate, "name", "") == engine_name:
                engine = candidate
        except Exception:
            pass
    _SELECTED_CONTEXT_ENGINE = engine
    _SELECTED_CONTEXT_ENGINE_LOADED = True
    return _SELECTED_CONTEXT_ENGINE

def _dispatch_context_engine_tool(tool_name, args):
    engine = _load_selected_context_engine()
    if engine is None:
        raise ValueError(f"unknown tool: {tool_name}")
    names = {
        str(schema.get("name", "") or "").strip()
        for schema in (engine.get_tool_schemas() or [])
        if isinstance(schema, dict)
    }
    if tool_name not in names:
        raise ValueError(f"unknown tool: {tool_name}")
    result = engine.handle_tool_call(tool_name, args or {}, messages=[])
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)

for raw in sys.stdin:
    text = raw.strip()
    if not text:
        continue
    try:
        request = json.loads(text)
        action = str(request.get("action", "")).strip()
        payload = request.get("payload") or {}
        if action == "dispatch":
            tool_name = payload.get("tool_name", "")
            emit({
                "ok": True,
                "result": (
                    registry.dispatch(
                        tool_name,
                        payload.get("args") or {},
                        task_id=payload.get("task_id"),
                        session_id=payload.get("session_id"),
                        enabled_tools=payload.get("enabled_tools"),
                    )
                    if registry.get_entry(tool_name) is not None
                    else _dispatch_context_engine_tool(tool_name, payload.get("args") or {})
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

const PYTHON_PLUGIN_IMAGE_GEN_PROBE: &str = concat!(
    "import json\n",
    "from agent.image_gen_registry import get_provider\n",
    "from hermes_cli.config import load_config\n",
    "from hermes_cli.plugins import discover_plugins\n",
    "discover_plugins(force=True)\n",
    "configured = None\n",
    "cfg = load_config()\n",
    "section = cfg.get('image_gen') if isinstance(cfg, dict) else None\n",
    "if isinstance(section, dict):\n",
    "    value = section.get('provider')\n",
    "    if isinstance(value, str) and value.strip():\n",
    "        configured = value.strip()\n",
    "available = False\n",
    "if configured and configured != 'fal':\n",
    "    provider = get_provider(configured)\n",
    "    if provider is not None:\n",
    "        try:\n",
    "            available = bool(provider.is_available())\n",
    "        except Exception:\n",
    "            available = False\n",
    "print(json.dumps({'available': available}))\n",
);

const PYTHON_PLUGIN_IMAGE_GEN_GENERATE: &str = concat!(
    "import json\n",
    "import os\n",
    "from agent.image_gen_registry import get_provider\n",
    "from hermes_cli.config import load_config\n",
    "from hermes_cli.plugins import discover_plugins\n",
    "discover_plugins(force=True)\n",
    "prompt = os.environ.get('HERMES_IMAGE_GEN_PROMPT', '')\n",
    "aspect_ratio = os.environ.get('HERMES_IMAGE_GEN_ASPECT_RATIO', '')\n",
    "configured = None\n",
    "cfg = load_config()\n",
    "section = cfg.get('image_gen') if isinstance(cfg, dict) else None\n",
    "if isinstance(section, dict):\n",
    "    value = section.get('provider')\n",
    "    if isinstance(value, str) and value.strip():\n",
    "        configured = value.strip()\n",
    "if not configured or configured == 'fal':\n",
    "    result = {\n",
    "        'success': False,\n",
    "        'image': None,\n",
    "        'error': 'no plugin image generation provider is configured',\n",
    "        'error_type': 'provider_not_registered',\n",
    "    }\n",
    "else:\n",
    "    provider = get_provider(configured)\n",
    "    if provider is None:\n",
    "        result = {\n",
    "            'success': False,\n",
    "            'image': None,\n",
    "            'error': (\n",
    "                f\"image_gen.provider='{configured}' is set but no plugin registered \"\n",
    "                f\"that name. Run `hermes plugins list` to see available image gen backends.\"\n",
    "            ),\n",
    "            'error_type': 'provider_not_registered',\n",
    "        }\n",
    "    else:\n",
    "        try:\n",
    "            result = provider.generate(prompt=prompt, aspect_ratio=aspect_ratio)\n",
    "        except Exception as exc:\n",
    "            result = {\n",
    "                'success': False,\n",
    "                'image': None,\n",
    "                'error': f\"Provider '{getattr(provider, 'name', '?')}' error: {exc}\",\n",
    "                'error_type': 'provider_exception',\n",
    "            }\n",
    "        if not isinstance(result, dict):\n",
    "            result = {\n",
    "                'success': False,\n",
    "                'image': None,\n",
    "                'error': 'Provider returned a non-dict result',\n",
    "                'error_type': 'provider_contract',\n",
    "            }\n",
    "print(json.dumps(result, ensure_ascii=False, default=str))\n",
);

const PYTHON_PLUGIN_IMAGE_GEN_PROVIDERS: &str = concat!(
    "import json\n",
    "from agent.image_gen_registry import list_providers\n",
    "from hermes_cli.plugins import discover_plugins\n",
    "discover_plugins(force=True)\n",
    "rows = []\n",
    "for provider in list_providers():\n",
    "    if getattr(provider, 'name', None) == 'fal':\n",
    "        continue\n",
    "    try:\n",
    "        schema = provider.get_setup_schema() or {}\n",
    "    except Exception:\n",
    "        schema = {}\n",
    "    if not isinstance(schema, dict):\n",
    "        schema = {}\n",
    "    try:\n",
    "        models = provider.list_models() or []\n",
    "    except Exception:\n",
    "        models = []\n",
    "    if not isinstance(models, list):\n",
    "        models = []\n",
    "    try:\n",
    "        default_model = provider.default_model()\n",
    "    except Exception:\n",
    "        default_model = None\n",
    "    try:\n",
    "        available = bool(provider.is_available())\n",
    "    except Exception:\n",
    "        available = False\n",
    "    env_vars = []\n",
    "    for item in schema.get('env_vars', []) or []:\n",
    "        if not isinstance(item, dict):\n",
    "            continue\n",
    "        key = str(item.get('key', '') or '').strip()\n",
    "        if not key:\n",
    "            continue\n",
    "        env_vars.append({\n",
    "            'key': key,\n",
    "            'prompt': str(item.get('prompt', '') or ''),\n",
    "            'url': item.get('url'),\n",
    "            'default': item.get('default'),\n",
    "        })\n",
    "    model_rows = []\n",
    "    for item in models:\n",
    "        if not isinstance(item, dict):\n",
    "            continue\n",
    "        model_id = str(item.get('id', '') or '').strip()\n",
    "        if not model_id:\n",
    "            continue\n",
    "        model_rows.append({\n",
    "            'id': model_id,\n",
    "            'display': str(item.get('display', '') or ''),\n",
    "            'speed': str(item.get('speed', '') or ''),\n",
    "            'strengths': str(item.get('strengths', '') or ''),\n",
    "            'price': str(item.get('price', '') or ''),\n",
    "        })\n",
    "    rows.append({\n",
    "        'plugin_name': str(getattr(provider, 'name', '') or ''),\n",
    "        'name': str(schema.get('name', getattr(provider, 'display_name', getattr(provider, 'name', ''))) or ''),\n",
    "        'badge': str(schema.get('badge', '') or ''),\n",
    "        'tag': str(schema.get('tag', '') or ''),\n",
    "        'available': available,\n",
    "        'env_vars': env_vars,\n",
    "        'models': model_rows,\n",
    "        'default_model': default_model,\n",
    "    })\n",
    "print(json.dumps(rows, ensure_ascii=False, default=str))\n",
);

const PYTHON_CONTEXT_ENGINE_TOOLS: &str = r#"
import json

from hermes_cli.plugins import discover_plugins

discover_plugins()

def _load_selected_context_engine():
    engine_name = "compressor"
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("context") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("engine")
            if isinstance(value, str) and value.strip():
                engine_name = value.strip()
    except Exception:
        pass

    if not engine_name or engine_name == "compressor":
        return None

    engine = None
    try:
        from plugins.context_engine import load_context_engine
        engine = load_context_engine(engine_name)
    except Exception:
        engine = None

    if engine is None:
        try:
            from hermes_cli.plugins import get_plugin_context_engine
            candidate = get_plugin_context_engine()
            if candidate is not None and getattr(candidate, "name", "") == engine_name:
                engine = candidate
        except Exception:
            pass
    return engine

engine = _load_selected_context_engine()
definitions = []
if engine is not None:
    for schema in engine.get_tool_schemas() or []:
        if not isinstance(schema, dict):
            continue
        name = str(schema.get("name", "") or "").strip()
        if not name:
            continue
        definitions.append({
            "name": name,
            "toolset": "__always__",
            "description": str(schema.get("description", "") or ""),
            "emoji": "",
            "schema": schema,
        })

print(json.dumps(definitions, ensure_ascii=False))
"#;

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
    for definition in discover_selected_context_engine_tool_definitions(hermes_home, runtime.cwd())?
    {
        if dynamic_tool_names.insert(definition.name.clone()) {
            toolsets
                .entry(definition.toolset.clone())
                .or_default()
                .push(definition.name.clone());
            definitions.push(definition);
        }
    }
    if definitions.is_empty() && active_hook_names.is_empty() {
        return Ok(runtime);
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

pub fn python_plugin_image_gen_available(hermes_home: &Path, cwd: &Path) -> Result<bool, String> {
    #[derive(Deserialize)]
    struct ProbeResult {
        available: bool,
    }

    run_python_plugin_json(hermes_home, cwd, PYTHON_PLUGIN_IMAGE_GEN_PROBE, &[])
        .map(|result: ProbeResult| result.available)
}

pub fn dispatch_python_plugin_image_generate(
    hermes_home: &Path,
    cwd: &Path,
    prompt: &str,
    aspect_ratio: &str,
) -> Result<Value, String> {
    run_python_plugin_json(
        hermes_home,
        cwd,
        PYTHON_PLUGIN_IMAGE_GEN_GENERATE,
        &[
            ("HERMES_IMAGE_GEN_PROMPT", prompt),
            ("HERMES_IMAGE_GEN_ASPECT_RATIO", aspect_ratio),
        ],
    )
}

pub fn discover_plugin_image_gen_providers(
    hermes_home: &Path,
    cwd: &Path,
) -> Result<Vec<ImageGenPluginProvider>, String> {
    run_python_plugin_json(hermes_home, cwd, PYTHON_PLUGIN_IMAGE_GEN_PROVIDERS, &[])
}

fn discover_selected_context_engine_tool_definitions(
    hermes_home: &Path,
    cwd: &Path,
) -> Result<Vec<ToolDefinition>, String> {
    let mut definitions = run_python_plugin_json::<Vec<ToolDefinition>>(
        hermes_home,
        cwd,
        PYTHON_CONTEXT_ENGINE_TOOLS,
        &[],
    )?;
    for definition in &mut definitions {
        definition.toolset = ALWAYS_DYNAMIC_TOOLSET.to_string();
    }
    Ok(definitions)
}

fn run_python_plugin_json<T: for<'de> Deserialize<'de>>(
    hermes_home: &Path,
    cwd: &Path,
    script: &str,
    extra_env: &[(&str, &str)],
) -> Result<T, String> {
    let root = project_root();
    let python =
        resolve_repo_python(&root, Some("HERMES_PLUGIN_RUNTIME_PYTHON")).ok_or_else(|| {
            "could not find a Python interpreter for plugin runtime dispatch".to_string()
        })?;
    let mut command = Command::new(&python);
    command
        .current_dir(cwd)
        .env("HERMES_HOME", hermes_home)
        .env("PYTHONPATH", pythonpath_with_root(&root))
        .arg("-c")
        .arg(script)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped());
    apply_hermes_env_file(&mut command, hermes_home);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|error| format!("failed to start plugin runtime helper: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            match output.status.code() {
                Some(code) => format!("plugin runtime helper exited with status {code}"),
                None => "plugin runtime helper terminated by signal".to_string(),
            }
        } else {
            stderr
        });
    }
    serde_json::from_slice::<T>(&output.stdout).map_err(|error| {
        format!(
            "failed to decode plugin runtime helper output: {error}; stdout={}",
            String::from_utf8_lossy(&output.stdout).trim()
        )
    })
}

pub fn discover_enabled_plugin_cli_commands(
    hermes_home: &Path,
    cwd: &Path,
) -> Vec<PluginCliCommand> {
    let mut seen = BTreeSet::new();
    let mut commands = Vec::new();
    for plugin in discover_enabled_general_plugins(hermes_home, cwd) {
        for command in plugin.cli_commands {
            if seen.insert(command.name.clone()) {
                commands.push(command);
            }
        }
    }
    if let Some(memory_command) = discover_active_memory_cli_command(hermes_home)
        && seen.insert(memory_command.name.clone())
    {
        commands.push(memory_command);
    }
    commands.sort_by(|left, right| left.name.cmp(&right.name));
    commands
}

pub fn discover_enabled_plugin_platforms(hermes_home: &Path, cwd: &Path) -> Vec<PlatformSurface> {
    let mut surfaces = Vec::new();
    for plugin in discover_enabled_general_plugins(hermes_home, cwd) {
        if plugin.platforms.is_empty() {
            continue;
        }
        surfaces.extend(plugin.platforms);
    }
    surfaces.sort_by(|left, right| left.key.cmp(&right.key));
    surfaces
}

pub fn discover_scanned_plugins(hermes_home: &Path, cwd: &Path) -> Vec<DiscoveredPlugin> {
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
    winners.into_values().collect()
}

pub fn discover_memory_provider_plugins(hermes_home: &Path) -> Vec<DiscoveredPlugin> {
    let mut plugins = Vec::new();
    let mut seen = BTreeSet::new();
    for (root, source, prefix, require_marker) in [
        (
            bundled_plugins_dir().join("memory"),
            PluginSource::Bundled,
            "memory",
            false,
        ),
        (user_plugins_dir(hermes_home), PluginSource::User, "", true),
    ] {
        for plugin in scan_memory_provider_root(&root, source, prefix, require_marker) {
            if seen.insert(plugin.name.clone()) {
                plugins.push(plugin);
            }
        }
    }
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));
    plugins
}

pub fn discover_general_plugins(hermes_home: &Path, cwd: &Path) -> Vec<DiscoveredPlugin> {
    let mut plugins = discover_scanned_plugins(hermes_home, cwd)
        .into_iter()
        .filter(|plugin| {
            !matches!(
                plugin.kind,
                PluginKind::Exclusive | PluginKind::ModelProvider | PluginKind::ContextEngine
            )
        })
        .collect::<Vec<_>>();
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));
    plugins
}

pub fn discover_context_engine_plugins(hermes_home: &Path, cwd: &Path) -> Vec<DiscoveredPlugin> {
    let mut plugins = BTreeMap::<String, DiscoveredPlugin>::new();
    for plugin in discover_scanned_plugins(hermes_home, cwd) {
        if plugin.kind == PluginKind::ContextEngine {
            plugins.entry(plugin.name.clone()).or_insert(plugin);
        }
    }
    plugins.into_values().collect()
}

pub fn is_effectively_enabled(
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

pub fn discover_dashboard_surfaces(hermes_home: &Path, cwd: &Path) -> Vec<DashboardSurface> {
    let mut dashboards = BTreeMap::<String, DashboardSurface>::new();
    for (root, source) in [
        (user_plugins_dir(hermes_home), PluginSource::User),
        (bundled_plugins_dir(), PluginSource::Bundled),
        (project_plugins_dir(cwd), PluginSource::Project),
    ] {
        if !root.is_dir() {
            continue;
        }
        for dashboard in scan_dashboard_tree(&root, source) {
            dashboards
                .entry(dashboard.name.clone())
                .or_insert(dashboard);
        }
    }
    dashboards.into_values().collect()
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
    apply_hermes_env_file(&mut command, hermes_home);
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

fn apply_hermes_env_file(command: &mut Command, hermes_home: &Path) {
    for (key, value) in hermes_env_entries(hermes_home) {
        command.env(key, value);
    }
}

fn hermes_env_entries(hermes_home: &Path) -> Vec<(String, String)> {
    let path = hermes_home.join(".env");
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    let contents = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => String::from_utf8_lossy(&error.into_bytes()).into_owned(),
    };
    let mut entries = Vec::new();
    for line in contents.lines() {
        let Some((key, value)) = parse_env_line(line) else {
            continue;
        };
        if !is_valid_env_var_name(&key) {
            continue;
        }
        entries.push((key.clone(), sanitize_env_value(&key, &value)));
    }
    entries
}

fn parse_env_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let without_export = trimmed.strip_prefix("export ").unwrap_or(trimmed);
    let (key, raw_value) = without_export.split_once('=')?;
    Some((
        key.trim().to_string(),
        strip_matching_quotes(raw_value.trim()).to_string(),
    ))
}

fn strip_matching_quotes(value: &str) -> &str {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let first = bytes[0];
        let last = bytes[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}

fn is_valid_env_var_name(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn sanitize_env_value(key: &str, value: &str) -> String {
    if !CREDENTIAL_SUFFIXES
        .iter()
        .any(|suffix| key.ends_with(suffix))
    {
        return value.to_string();
    }
    value.chars().filter(|ch| ch.is_ascii()).collect()
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
pub enum PluginSource {
    Bundled,
    User,
    Project,
}

impl PluginSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bundled => "bundled",
            Self::User => "user",
            Self::Project => "project",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginKind {
    Standalone,
    Backend,
    Exclusive,
    Platform,
    ModelProvider,
    ContextEngine,
}

impl PluginKind {
    pub fn is_auto_enabled(self, source: PluginSource) -> bool {
        source == PluginSource::Bundled && matches!(self, Self::Backend | Self::Platform)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredPlugin {
    pub key: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub kind: PluginKind,
    pub source: PluginSource,
    pub path: PathBuf,
    pub requires_env: Vec<String>,
    pub provides_tools: Vec<String>,
    pub provides_hooks: Vec<String>,
    pub tool_definitions: Vec<ToolDefinition>,
    pub cli_commands: Vec<PluginCliCommand>,
    pub platforms: Vec<PlatformSurface>,
}

pub fn discover_enabled_general_plugins(hermes_home: &Path, cwd: &Path) -> Vec<DiscoveredPlugin> {
    let enabled = load_plugin_set(hermes_home, "enabled");
    let disabled = load_plugin_set(hermes_home, "disabled");
    discover_general_plugins(hermes_home, cwd)
        .into_iter()
        .filter(|plugin| is_effectively_enabled(plugin, &enabled, &disabled))
        .collect()
}

/// A plugin's listing entry: name, version, and effective enabled state.
#[derive(Debug, Clone)]
pub struct PluginListing {
    pub name: String,
    pub version: String,
    pub enabled: bool,
}

/// List all discovered general plugins with their effective enabled state.
/// Port of the `plugins.list` RPC (which enumerates the plugin manager's
/// plugins, including disabled ones, with name/version/enabled).
pub fn plugins_list(hermes_home: &Path, cwd: &Path) -> Vec<PluginListing> {
    let enabled = load_plugin_set(hermes_home, "enabled");
    let disabled = load_plugin_set(hermes_home, "disabled");
    discover_general_plugins(hermes_home, cwd)
        .into_iter()
        .map(|plugin| {
            let is_enabled = is_effectively_enabled(&plugin, &enabled, &disabled);
            PluginListing {
                name: plugin.name,
                version: plugin.version,
                enabled: is_enabled,
            }
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

fn scan_memory_provider_root(
    root: &Path,
    source: PluginSource,
    prefix: &str,
    require_marker: bool,
) -> Vec<DiscoveredPlugin> {
    let mut plugins = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return plugins;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(plugin) = parse_memory_provider_dir(&path, prefix, source, require_marker) else {
            continue;
        };
        plugins.push(plugin);
    }
    plugins
}

fn scan_dashboard_tree(root: &Path, source: PluginSource) -> Vec<DashboardSurface> {
    let mut dashboards = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return dashboards;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest = path.join("dashboard").join("manifest.json");
        if !manifest.exists() {
            continue;
        }
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<DashboardManifest>(&text) else {
            continue;
        };
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
    dashboards
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
    let cli_commands = discover_plugin_cli_commands_from_source(&source_text, &name);
    let platforms = discover_platform_surfaces_from_source(&source_text, &name);
    let tool_definitions = discover_tool_definitions_from_source(&source_text);
    let mut provides_tools = mapping_string_list(mapping, "provides_tools");
    for definition in &tool_definitions {
        if !provides_tools.iter().any(|tool| tool == &definition.name) {
            provides_tools.push(definition.name.clone());
        }
    }
    let mut provides_hooks = mapping_string_list(mapping, "provides_hooks");
    if provides_hooks.is_empty() {
        provides_hooks = mapping_string_list(mapping, "hooks");
    }
    for hook in discover_hook_registrations_from_source(&source_text) {
        if !provides_hooks.iter().any(|existing| existing == &hook) {
            provides_hooks.push(hook);
        }
    }
    Some(DiscoveredPlugin {
        key: key.clone(),
        cli_commands,
        name,
        version: mapping_string(mapping, "version").unwrap_or_default(),
        description: mapping_string(mapping, "description").unwrap_or_default(),
        author: mapping_string(mapping, "author").unwrap_or_default(),
        kind: determine_plugin_kind(&key, mapping_string(mapping, "kind"), &source_text),
        platforms,
        path: plugin_dir.to_path_buf(),
        requires_env: mapping_string_list(mapping, "requires_env"),
        provides_tools,
        provides_hooks,
        source,
        tool_definitions,
    })
}

fn parse_memory_provider_dir(
    plugin_dir: &Path,
    prefix: &str,
    source: PluginSource,
    require_marker: bool,
) -> Option<DiscoveredPlugin> {
    let init_file = plugin_dir.join("__init__.py");
    if !init_file.exists() {
        return None;
    }
    let dir_name = plugin_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if dir_name.is_empty() || dir_name.starts_with(['.', '_']) {
        return None;
    }

    let key = if prefix.is_empty() {
        dir_name.clone()
    } else {
        format!("{prefix}/{dir_name}")
    };
    let source_text = read_plugin_source_files(plugin_dir);
    if source_text.trim().is_empty() {
        return None;
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
        return None;
    }
    if require_marker
        && !source_text.contains("register_memory_provider")
        && !source_text.contains("MemoryProvider")
    {
        return None;
    }

    let description = manifest
        .as_ref()
        .and_then(|mapping| mapping_string(mapping, "description"))
        .unwrap_or_default();
    Some(DiscoveredPlugin {
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

fn discover_active_memory_cli_command(hermes_home: &Path) -> Option<PluginCliCommand> {
    let config_path = hermes_home.join("config.yaml");
    let provider = fs::read_to_string(config_path)
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
        return None;
    }
    let candidates = [
        bundled_plugins_dir().join("memory").join(&provider),
        user_plugins_dir(hermes_home).join(&provider),
    ];
    let plugin_dir = candidates
        .into_iter()
        .find(|path| path.join("cli.py").exists())?;
    let description = plugin_manifest_path(&plugin_dir)
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|text| serde_yaml::from_str::<YamlValue>(&text).ok())
        .and_then(|value| {
            value
                .as_mapping()
                .and_then(|mapping| mapping_string(mapping, "description"))
        })
        .unwrap_or_default();
    Some(PluginCliCommand {
        name: provider.clone(),
        help: if description.is_empty() {
            format!("Manage {provider} memory plugin")
        } else {
            description.clone()
        },
        description,
        plugin_name: provider,
    })
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

pub fn discover_plugin_cli_commands_from_source(
    source_text: &str,
    plugin_name: &str,
) -> Vec<PluginCliCommand> {
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

pub fn discover_platform_surfaces_from_source(
    source_text: &str,
    plugin_name: &str,
) -> Vec<PlatformSurface> {
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
            key: key.clone(),
            label: extract_keyword_string(&block, "label").unwrap_or_else(|| titleize(&key)),
            required_env: extract_keyword_string_list(&block, "required_env"),
            install_hint: extract_keyword_string(&block, "install_hint"),
            emoji: extract_keyword_string(&block, "emoji").unwrap_or_default(),
            has_setup_fn: keyword_present(&block, "setup_fn"),
            plugin_name: plugin_name.to_string(),
        });
    }
    platforms
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
    use tempfile::TempDir;

    use crate::{dispatch_tool, get_tool_definitions_with_runtime};

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
    fn attach_python_plugin_runtime_bridges_selected_context_engine_tools() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("demo-context");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: demo-context\ndescription: Demo context engine\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
import json

from agent.context_engine import ContextEngine

class DemoContextEngine(ContextEngine):
    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "demo-engine"

    def update_from_response(self, usage):
        return None

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def get_tool_schemas(self):
        return [{
            "name": "demo_engine_lookup",
            "description": "Demo context engine tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }]

    def handle_tool_call(self, name, args, **kwargs):
        self.calls += 1
        return json.dumps({
            "success": True,
            "query": (args or {}).get("query", ""),
            "calls": self.calls,
        })

def register(ctx):
    ctx.register_context_engine(DemoContextEngine())
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo-context\ncontext:\n  engine: demo-engine\n",
        )
        .unwrap();

        let runtime = attach_python_plugin_runtime(
            &home,
            ToolRuntime::new(temp.path()).with_hermes_home(&home),
        )
        .unwrap();

        let enabled_toolsets = vec![String::from("hermes-cli")];
        let definitions =
            get_tool_definitions_with_runtime(Some(&enabled_toolsets), None, Some(&runtime));
        assert!(
            definitions
                .iter()
                .any(|tool| tool.name == "demo_engine_lookup")
        );

        let result = dispatch_tool("demo_engine_lookup", json!({"query": "status"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["query"], json!("status"));
        assert_eq!(parsed["calls"], json!(1));

        let result = dispatch_tool("demo_engine_lookup", json!({"query": "again"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["query"], json!("again"));
        assert_eq!(parsed["calls"], json!(2));
        assert!(
            runtime
                .invoke_hook("pre_llm_call", &json!({"session_id": "s"}))
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

    #[test]
    fn discover_plugin_image_gen_providers_lists_bundled_backends() {
        let temp = TempDir::new().unwrap();
        let providers = discover_plugin_image_gen_providers(temp.path(), temp.path()).unwrap();
        let names = providers
            .iter()
            .map(|provider| provider.plugin_name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"openai"));
        assert!(names.contains(&"openai-codex"));
        assert!(names.contains(&"xai"));

        let openai = providers
            .iter()
            .find(|provider| provider.plugin_name == "openai")
            .expect("openai image provider");
        assert!(!openai.available);
        assert!(
            openai
                .env_vars
                .iter()
                .any(|item| item.key == "OPENAI_API_KEY")
        );
        assert!(
            openai
                .models
                .iter()
                .any(|model| model.id == "gpt-image-2-medium")
        );
    }
}
