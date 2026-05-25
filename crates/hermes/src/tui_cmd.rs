use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Args;
use hermes_core::{
    DelegateExecutor, HermesContext, LoadedConfig, MessageRecord, ModelOverrides, SessionCreate,
    SessionStore, ToolRuntime,
};
use serde_json::{Value as JsonValue, json};
use serde_yaml::Value as YamlValue;

use crate::python_bridge::project_root;

#[derive(Args, Debug, Clone, Default)]
pub struct TuiGatewayArgs {}

#[derive(Debug, Clone, Default)]
pub struct TuiLaunchOptions {
    pub continue_last: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub query: Option<String>,
    pub resume: Option<String>,
    pub toolsets: Option<String>,
    pub tui_dev: bool,
}

struct TuiGatewayServer {
    context: HermesContext,
    cwd: PathBuf,
    loaded: LoadedConfig,
    overrides: ModelOverrides,
    session_store: SessionStore,
    sessions: HashSet<String>,
    toolsets: Vec<String>,
}

pub fn launch_tui(
    _context: &HermesContext,
    session_store: &SessionStore,
    mut options: TuiLaunchOptions,
) -> Result<(), Box<dyn Error>> {
    let resume_session_id = resolve_launch_resume(session_store, &mut options)?;
    let tui_dir = resolve_tui_dir();
    let cwd = env::current_dir().unwrap_or_else(|_| project_root());
    let (program, args, run_cwd) = resolve_tui_command(&tui_dir, options.tui_dev)?;
    let current_exe = env::current_exe()?;

    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(run_cwd)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    command.env("HERMES_CWD", &cwd);
    command.env("HERMES_TUI_GATEWAY_BIN", &current_exe);
    command.env("HERMES_TUI_GATEWAY_ARGS_JSON", "[\"tui-gateway\"]");
    command.env(
        "NODE_ENV",
        if options.tui_dev {
            "development"
        } else {
            "production"
        },
    );
    command.env("HERMES_TUI_BACKEND", "rust");

    let node_options = env::var("NODE_OPTIONS").unwrap_or_default();
    let mut node_tokens = node_options
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if !node_tokens
        .iter()
        .any(|token| token.starts_with("--max-old-space-size="))
    {
        node_tokens.push(String::from("--max-old-space-size=8192"));
    }
    if !node_tokens.iter().any(|token| token == "--expose-gc") {
        node_tokens.push(String::from("--expose-gc"));
    }
    command.env("NODE_OPTIONS", node_tokens.join(" "));

    if let Some(value) = options
        .model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.env("HERMES_MODEL", value);
        command.env("HERMES_INFERENCE_MODEL", value);
    }
    if let Some(value) = options
        .provider
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.env("HERMES_TUI_PROVIDER", value);
        command.env("HERMES_INFERENCE_PROVIDER", value);
    }
    if let Some(value) = options
        .toolsets
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.env("HERMES_TUI_TOOLSETS", value);
    }
    if let Some(value) = options
        .query
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.env("HERMES_TUI_QUERY", value);
    }
    if let Some(value) = resume_session_id.as_deref() {
        command.env("HERMES_TUI_RESUME", value);
    }

    let status = command.status()?;
    std::process::exit(status.code().unwrap_or(1));
}

pub fn run_tui_gateway(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<(), Box<dyn Error>> {
    let mut server = TuiGatewayServer::new(context.clone(), loaded.clone())?;
    server.run()
}

impl TuiGatewayServer {
    fn new(context: HermesContext, loaded: LoadedConfig) -> Result<Self, Box<dyn Error>> {
        let cwd = env::var_os("HERMES_CWD")
            .map(PathBuf::from)
            .filter(|value| !value.as_os_str().is_empty())
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| project_root()));
        let toolsets = parse_env_list("HERMES_TUI_TOOLSETS")
            .into_iter()
            .filter(|name| hermes_core::validate_toolset(name))
            .collect::<Vec<_>>();
        let overrides = ModelOverrides {
            model: env_string("HERMES_MODEL").or_else(|| env_string("HERMES_INFERENCE_MODEL")),
            provider: env_string("HERMES_TUI_PROVIDER")
                .or_else(|| env_string("HERMES_INFERENCE_PROVIDER")),
            base_url: env_string("HERMES_BASE_URL"),
            api_key: env_string("HERMES_API_KEY"),
            api_mode: env_string("HERMES_API_MODE"),
        };
        let session_store = context.open_session_store()?;
        Ok(Self {
            context,
            cwd,
            loaded,
            overrides,
            session_store,
            sessions: HashSet::new(),
            toolsets,
        })
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let stdin = io::stdin();
        let mut stdout = io::stdout().lock();
        self.write_event(&mut stdout, "gateway.ready", None, json!({}))?;
        for line in stdin.lock().lines() {
            let raw = match line {
                Ok(value) => value,
                Err(error) => {
                    self.write_event(
                        &mut stdout,
                        "error",
                        None,
                        json!({ "message": error.to_string() }),
                    )?;
                    continue;
                }
            };
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            let request = match serde_json::from_str::<JsonValue>(trimmed) {
                Ok(value) => value,
                Err(error) => {
                    write_jsonrpc_error(
                        &mut stdout,
                        JsonValue::Null,
                        -32700,
                        &format!("invalid JSON-RPC payload: {error}"),
                    )?;
                    continue;
                }
            };
            self.handle_request(&mut stdout, request)?;
        }
        Ok(())
    }

    fn handle_request(
        &mut self,
        stdout: &mut impl Write,
        request: JsonValue,
    ) -> Result<(), Box<dyn Error>> {
        let id = request.get("id").cloned().unwrap_or(JsonValue::Null);
        let method = request
            .get("method")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "missing method".to_string());
        let params = request
            .get("params")
            .cloned()
            .unwrap_or_else(|| JsonValue::Object(Default::default()));
        let params = params.as_object().cloned().unwrap_or_default();
        let method = match method {
            Ok(value) => value,
            Err(error) => {
                write_jsonrpc_error(stdout, id, -32600, &error)?;
                return Ok(());
            }
        };

        let result = match method {
            "commands.catalog" => Ok(json!({
                "canon": {},
                "categories": [],
                "pairs": [],
                "skill_count": 0,
                "sub": {},
            })),
            "complete.path" | "complete.slash" => Ok(json!({
                "items": [],
                "replace_from": 0,
            })),
            "config.get" => self.handle_config_get(&params),
            "session.close" => self.handle_session_close(&params),
            "session.create" => self.handle_session_create(stdout, &params),
            "session.delete" => self.handle_session_delete(&params),
            "session.list" => self.handle_session_list(&params),
            "session.most_recent" => self.handle_session_most_recent(),
            "session.resume" => self.handle_session_resume(stdout, &params),
            "session.title" => self.handle_session_title(stdout, &params),
            "setup.status" => self.handle_setup_status(),
            "terminal.resize" => Ok(json!({"ok": true})),
            "prompt.submit" => self.handle_prompt_submit(stdout, &params),
            "command.dispatch" => Ok(json!({
                "output": "The Rust TUI backend does not support gateway slash dispatch yet.",
                "type": "exec",
            })),
            "slash.exec" => Ok(json!({
                "output": "The Rust TUI backend does not support slash worker execution yet.",
            })),
            other => Err(format!("unsupported JSON-RPC method '{other}'")),
        };

        match result {
            Ok(value) => write_jsonrpc_result(stdout, id, value)?,
            Err(error) => write_jsonrpc_error(stdout, id, -32601, &error)?,
        }
        Ok(())
    }

    fn handle_config_get(
        &self,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let key = params
            .get("key")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        match key {
            "full" => Ok(json!({
                "config": {
                    "display": self.display_config_json(),
                    "voice": {},
                }
            })),
            "mtime" => {
                let mtime = fs::metadata(&self.loaded.path)
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .and_then(system_time_seconds)
                    .unwrap_or(0.0);
                Ok(json!({ "mtime": mtime }))
            }
            other => Err(format!("unsupported config.get key '{other}'")),
        }
    }

    fn handle_setup_status(&self) -> Result<JsonValue, String> {
        let configured = self
            .context
            .resolve_model_runtime(&self.loaded, &self.overrides)
            .is_ok();
        Ok(json!({ "provider_configured": configured }))
    }

    fn handle_session_create(
        &mut self,
        stdout: &mut impl Write,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let cols = parse_u16(params.get("cols")).unwrap_or(80);
        let session_id = format!("tui_{:x}", unix_ts_nanos());
        self.ensure_session_row(&session_id)?;
        let _ = cols;
        self.sessions.insert(session_id.clone());
        let info = self.session_info(&session_id)?;
        self.write_event(stdout, "session.info", Some(&session_id), info.clone())
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "info": info,
            "session_id": session_id,
        }))
    }

    fn handle_session_resume(
        &mut self,
        stdout: &mut impl Write,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let requested = params
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "session.resume requires session_id".to_string())?;
        let resolved = self.resolve_session_key(requested)?;
        let cols = parse_u16(params.get("cols")).unwrap_or(80);
        let info = self.session_info(&resolved)?;
        let _ = cols;
        self.sessions.insert(resolved.clone());
        let messages = self
            .session_store
            .get_messages(&resolved)
            .map_err(|error| error.to_string())?
            .into_iter()
            .filter_map(message_record_to_gateway_row)
            .collect::<Vec<_>>();
        self.write_event(stdout, "session.info", Some(&resolved), info.clone())
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "info": info,
            "message_count": messages.len(),
            "messages": messages,
            "resumed": resolved,
            "session_id": requested,
        }))
    }

    fn handle_session_close(
        &mut self,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let session_id = params
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(session_id) = session_id {
            self.sessions.remove(session_id);
            let _ = self.session_store.end_session(session_id, "closed");
        }
        Ok(json!({"ok": true}))
    }

    fn handle_session_title(
        &mut self,
        stdout: &mut impl Write,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let session_id = params
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "session.title requires session_id".to_string())?;
        let title = params
            .get("title")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "session.title requires title".to_string())?;
        let resolved = self.resolve_session_key(session_id)?;
        self.session_store
            .set_session_title(&resolved, title)
            .map_err(|error| error.to_string())?;
        let info = self.session_info(&resolved)?;
        self.write_event(stdout, "session.info", Some(&resolved), info)
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "pending": false,
            "session_key": resolved,
            "title": title,
        }))
    }

    fn handle_session_list(
        &self,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let limit = params
            .get("limit")
            .and_then(JsonValue::as_i64)
            .unwrap_or(200)
            .clamp(1, 500);
        let sessions = self
            .session_store
            .search_sessions(None, limit, 0)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|row| {
                json!({
                    "id": row.id,
                    "message_count": row.message_count,
                    "preview": row.preview,
                    "source": row.source,
                    "started_at": row.started_at,
                    "title": row.title.unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({ "sessions": sessions }))
    }

    fn handle_session_delete(
        &mut self,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let session_id = params
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "session.delete requires session_id".to_string())?;
        let resolved = self.resolve_session_key(session_id)?;
        self.session_store
            .delete_session(&resolved)
            .map_err(|error| error.to_string())?;
        self.sessions.remove(&resolved);
        Ok(json!({ "deleted": resolved }))
    }

    fn handle_session_most_recent(&self) -> Result<JsonValue, String> {
        let latest = self
            .session_store
            .search_sessions(None, 1, 0)
            .map_err(|error| error.to_string())?
            .into_iter()
            .next();
        Ok(match latest {
            Some(row) => json!({
                "session_id": row.id,
                "source": row.source,
                "started_at": row.started_at,
                "title": row.title,
            }),
            None => json!({"session_id": JsonValue::Null}),
        })
    }

    fn handle_prompt_submit(
        &mut self,
        stdout: &mut impl Write,
        params: &serde_json::Map<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let session_id = params
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "prompt.submit requires session_id".to_string())?;
        let text = params
            .get("text")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "prompt.submit requires text".to_string())?;
        let resolved = self.resolve_session_key(session_id)?;
        self.write_event(stdout, "message.start", Some(&resolved), json!({}))
            .map_err(|error| error.to_string())?;

        let delegate = DelegateExecutor::new(
            self.context.clone(),
            self.loaded.clone(),
            "rust-tui",
            self.effective_toolsets(),
            self.overrides.clone(),
            self.cwd.clone(),
        );
        let runtime = ToolRuntime::new(self.cwd.clone())
            .with_hermes_home(self.context.hermes_home())
            .with_clarify_callback(|question, _choices| {
                Err(format!(
                    "Clarify is not supported by the Rust TUI backend yet: {question}"
                ))
            })
            .with_delegate_callback(move |request, parent_runtime| {
                delegate.execute(request, parent_runtime)
            });
        let result = self
            .context
            .run_chat_completions_turn(
                &self.loaded,
                text,
                &runtime,
                Some(&self.effective_toolsets()),
                &self.overrides,
                Some(&resolved),
                Some(&self.session_store),
            )
            .map_err(|error| error.to_string())?;
        let info = self.session_info(&resolved)?;
        self.write_event(stdout, "session.info", Some(&resolved), info.clone())
            .map_err(|error| error.to_string())?;
        self.write_event(
            stdout,
            "message.complete",
            Some(&resolved),
            json!({
                "text": result.final_response,
                "usage": info.get("usage").cloned().unwrap_or_else(|| json!({
                    "calls": result.api_calls,
                    "input": 0,
                    "output": 0,
                    "total": 0,
                })),
            }),
        )
        .map_err(|error| error.to_string())?;
        Ok(json!({ "ok": true }))
    }

    fn ensure_session_row(&self, session_id: &str) -> Result<(), String> {
        if self
            .session_store
            .get_session(session_id)
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Ok(());
        }
        let runtime = self
            .context
            .resolve_model_runtime(&self.loaded, &self.overrides)
            .map_err(|error| error.to_string())?;
        let tool_runtime =
            ToolRuntime::new(self.cwd.clone()).with_hermes_home(self.context.hermes_home());
        let system_prompt = self
            .context
            .render_system_prompt(&tool_runtime, &self.loaded.config.memory)
            .map_err(|error| error.to_string())?;
        self.session_store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: String::from("tui"),
                user_id: None,
                model: Some(runtime.model.clone()),
                model_config: Some(json!({
                    "api_mode": runtime.api_mode,
                    "base_url": runtime.base_url,
                    "provider": runtime.provider,
                })),
                system_prompt: Some(system_prompt),
                parent_session_id: None,
            })
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn resolve_session_key(&self, requested: &str) -> Result<String, String> {
        self.session_store
            .resolve_session_id(requested)
            .map_err(|error| error.to_string())?
            .or_else(|| {
                self.session_store
                    .resolve_session_by_title(requested)
                    .ok()
                    .flatten()
            })
            .ok_or_else(|| format!("No unique session matched '{requested}'."))
    }

    fn session_info(&self, session_id: &str) -> Result<JsonValue, String> {
        let runtime = self
            .context
            .resolve_model_runtime(&self.loaded, &self.overrides)
            .map_err(|error| error.to_string())?;
        let session = self
            .session_store
            .get_session(session_id)
            .map_err(|error| error.to_string())?;
        let title = session.and_then(|row| row.title);
        Ok(json!({
            "cwd": self.cwd.display().to_string(),
            "fast": false,
            "lazy": false,
            "mcp_servers": [],
            "model": runtime.model,
            "reasoning_effort": "",
            "release_date": "",
            "service_tier": "",
            "skills": self.available_skills(),
            "system_prompt": "",
            "tools": self.available_tools(),
            "update_behind": JsonValue::Null,
            "update_command": "",
            "usage": {
                "calls": 0,
                "input": 0,
                "output": 0,
                "total": 0,
            },
            "version": env!("CARGO_PKG_VERSION"),
            "title": title,
        }))
    }

    fn effective_toolsets(&self) -> Vec<String> {
        if self.toolsets.is_empty() {
            self.loaded.config.toolsets.clone()
        } else {
            self.toolsets.clone()
        }
    }

    fn available_tools(&self) -> JsonValue {
        let mut grouped = serde_json::Map::new();
        for name in self.effective_toolsets() {
            let tools = hermes_core::resolve_toolset(&name);
            if tools.is_empty() {
                continue;
            }
            grouped.insert(
                name,
                JsonValue::Array(tools.into_iter().map(JsonValue::String).collect()),
            );
        }
        JsonValue::Object(grouped)
    }

    fn available_skills(&self) -> JsonValue {
        let skills_dir = self.context.skills_dir();
        let mut grouped = serde_json::Map::new();
        let Ok(entries) = fs::read_dir(skills_dir) else {
            return JsonValue::Object(grouped);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let category = entry.file_name().to_string_lossy().to_string();
            let Ok(children) = fs::read_dir(path) else {
                continue;
            };
            let names = children
                .flatten()
                .filter(|child| child.path().is_dir())
                .map(|child| JsonValue::String(child.file_name().to_string_lossy().to_string()))
                .collect::<Vec<_>>();
            grouped.insert(category, JsonValue::Array(names));
        }
        JsonValue::Object(grouped)
    }

    fn display_config_json(&self) -> JsonValue {
        let mut display = serde_json::Map::new();
        display.insert(
            String::from("streaming"),
            JsonValue::Bool(self.loaded.config.display.streaming),
        );
        display.insert(
            String::from("tui_compact"),
            JsonValue::Bool(self.loaded.config.display.compact),
        );
        if let Some(value) = self.loaded.cfg_get(&["display"]) {
            if let JsonValue::Object(extra) = yaml_to_json(value) {
                for (key, value) in extra {
                    display.insert(key, value);
                }
            }
        }
        JsonValue::Object(display)
    }

    fn write_event(
        &self,
        stdout: &mut impl Write,
        event_type: &str,
        session_id: Option<&str>,
        payload: JsonValue,
    ) -> Result<(), io::Error> {
        let mut params = serde_json::Map::new();
        params.insert(
            String::from("type"),
            JsonValue::String(event_type.to_string()),
        );
        if let Some(session_id) = session_id {
            params.insert(
                String::from("session_id"),
                JsonValue::String(session_id.to_string()),
            );
        }
        params.insert(String::from("payload"), payload);
        serde_json::to_writer(
            &mut *stdout,
            &json!({
                "jsonrpc": "2.0",
                "method": "event",
                "params": JsonValue::Object(params),
            }),
        )?;
        stdout.write_all(b"\n")?;
        stdout.flush()
    }
}

fn resolve_launch_resume(
    session_store: &SessionStore,
    options: &mut TuiLaunchOptions,
) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(value) = options
        .resume
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return resolve_session_by_name_or_id(session_store, value);
    }
    if let Some(value) = options.continue_last.as_deref() {
        if value.trim().is_empty() || value == "__latest__" {
            return Ok(resolve_last_session(session_store, Some("tui"))?
                .or_else(|| resolve_last_session(session_store, None).ok().flatten()));
        }
        return resolve_session_by_name_or_id(session_store, value);
    }
    Ok(None)
}

fn resolve_session_by_name_or_id(
    session_store: &SessionStore,
    value: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(id) = session_store.resolve_session_id(value)? {
        return Ok(Some(id));
    }
    if let Some(id) = session_store.resolve_session_by_title(value)? {
        return Ok(Some(id));
    }
    Ok(None)
}

fn resolve_last_session(
    session_store: &SessionStore,
    source: Option<&str>,
) -> Result<Option<String>, Box<dyn Error>> {
    Ok(session_store
        .search_sessions(source, 1, 0)?
        .into_iter()
        .next()
        .map(|row| row.id))
}

fn resolve_tui_dir() -> PathBuf {
    env::var_os("HERMES_TUI_DIR")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| project_root().join("ui-tui"))
}

fn resolve_tui_command(
    tui_dir: &Path,
    tui_dev: bool,
) -> Result<(String, Vec<String>, PathBuf), Box<dyn Error>> {
    ensure_node_installed()?;
    ensure_tui_dependencies(tui_dir)?;
    if tui_dev {
        run_command(
            "npm",
            ["run", "build", "--prefix", "packages/hermes-ink"],
            tui_dir,
        )?;
        let tsx = tui_dir
            .join("node_modules")
            .join(".bin")
            .join(if cfg!(windows) { "tsx.cmd" } else { "tsx" });
        if tsx.is_file() {
            return Ok((
                tsx.display().to_string(),
                vec![String::from("src/entry.tsx")],
                tui_dir.to_path_buf(),
            ));
        }
        return Ok((
            String::from("npm"),
            vec![String::from("start")],
            tui_dir.to_path_buf(),
        ));
    }

    if build_needed(tui_dir) {
        run_command("npm", ["run", "build"], tui_dir)?;
    }
    Ok((
        String::from("node"),
        vec![tui_dir.join("dist").join("entry.js").display().to_string()],
        tui_dir.to_path_buf(),
    ))
}

fn ensure_node_installed() -> Result<(), Box<dyn Error>> {
    if which("node").is_none() {
        return Err("node not found on PATH".into());
    }
    if which("npm").is_none() {
        return Err("npm not found on PATH".into());
    }
    Ok(())
}

fn ensure_tui_dependencies(tui_dir: &Path) -> Result<(), Box<dyn Error>> {
    let node_modules = tui_dir.join("node_modules");
    if node_modules.is_dir() {
        return Ok(());
    }
    run_command(
        "npm",
        [
            "install",
            "--silent",
            "--no-fund",
            "--no-audit",
            "--progress=false",
        ],
        tui_dir,
    )
}

fn build_needed(tui_dir: &Path) -> bool {
    let entry = tui_dir.join("dist").join("entry.js");
    if !entry.is_file() {
        return true;
    }
    let entry_mtime = fs::metadata(&entry)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(system_time_seconds)
        .unwrap_or(0.0);
    let mut newest_source = 0.0_f64;
    for relative in [
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "tsconfig.build.json",
    ] {
        let path = tui_dir.join(relative);
        if let Some(mtime) = fs::metadata(path)
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .and_then(system_time_seconds)
        {
            newest_source = newest_source.max(mtime);
        }
    }
    newest_source = newest_source.max(latest_matching_mtime(&tui_dir.join("src"), &["ts", "tsx"]));
    newest_source = newest_source.max(latest_matching_mtime(
        &tui_dir.join("packages").join("hermes-ink").join("src"),
        &["ts", "tsx"],
    ));
    newest_source > entry_mtime
}

fn latest_matching_mtime(root: &Path, suffixes: &[&str]) -> f64 {
    let Ok(entries) = fs::read_dir(root) else {
        return 0.0;
    };
    let mut newest = 0.0_f64;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            newest = newest.max(latest_matching_mtime(&path, suffixes));
            continue;
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        if !suffixes.iter().any(|suffix| *suffix == extension) {
            continue;
        }
        if let Some(mtime) = fs::metadata(path)
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .and_then(system_time_seconds)
        {
            newest = newest.max(mtime);
        }
    }
    newest
}

fn run_command<I, S>(program: &str, args: I, cwd: &Path) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args = args
        .into_iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
    let status = Command::new(program)
        .args(&args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if status.success() {
        return Ok(());
    }
    Err(format!(
        "{program} {} failed with status {}",
        args.join(" "),
        status.code().unwrap_or(1)
    )
    .into())
}

fn which(program: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|raw| {
        env::split_paths(&raw).find_map(|dir| {
            let candidate = dir.join(program);
            if candidate.is_file() {
                return Some(candidate);
            }
            #[cfg(windows)]
            {
                let candidate = dir.join(format!("{program}.exe"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            None
        })
    })
}

fn message_record_to_gateway_row(record: MessageRecord) -> Option<JsonValue> {
    let text = render_message_content(record.content.as_ref())?;
    if record.role == "tool" {
        return Some(json!({
            "context": "",
            "name": record.tool_name.unwrap_or_else(|| String::from("tool")),
            "role": "tool",
        }));
    }
    Some(json!({
        "role": record.role,
        "text": text,
    }))
}

fn render_message_content(content: Option<&JsonValue>) -> Option<String> {
    match content {
        Some(JsonValue::String(text)) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Some(JsonValue::Object(map)) => map
            .get("text")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| serde_json::to_string(map).ok()),
        Some(JsonValue::Array(items)) => {
            let joined = items
                .iter()
                .filter_map(render_message_content_value)
                .collect::<Vec<_>>()
                .join("\n");
            (!joined.trim().is_empty()).then_some(joined)
        }
        _ => None,
    }
}

fn render_message_content_value(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        JsonValue::Object(map) => map
            .get("text")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

fn parse_env_list(name: &str) -> Vec<String> {
    env_string(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_u16(value: Option<&JsonValue>) -> Option<u16> {
    value
        .and_then(JsonValue::as_u64)
        .and_then(|raw| u16::try_from(raw).ok())
}

fn yaml_to_json(value: &YamlValue) -> JsonValue {
    match value {
        YamlValue::Bool(value) => JsonValue::Bool(*value),
        YamlValue::Number(value) => value
            .as_i64()
            .map(|number| JsonValue::Number(number.into()))
            .or_else(|| {
                serde_json::Number::from_f64(value.as_f64().unwrap_or_default())
                    .map(JsonValue::Number)
            })
            .unwrap_or(JsonValue::Null),
        YamlValue::String(value) => JsonValue::String(value.clone()),
        YamlValue::Sequence(items) => JsonValue::Array(items.iter().map(yaml_to_json).collect()),
        YamlValue::Mapping(mapping) => JsonValue::Object(
            mapping
                .iter()
                .filter_map(|(key, value)| {
                    key.as_str()
                        .map(|key| (key.to_string(), yaml_to_json(value)))
                })
                .collect(),
        ),
        _ => JsonValue::Null,
    }
}

fn write_jsonrpc_result(
    stdout: &mut impl Write,
    id: JsonValue,
    result: JsonValue,
) -> Result<(), io::Error> {
    serde_json::to_writer(
        &mut *stdout,
        &json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()
}

fn write_jsonrpc_error(
    stdout: &mut impl Write,
    id: JsonValue,
    code: i64,
    message: &str,
) -> Result<(), io::Error> {
    serde_json::to_writer(
        &mut *stdout,
        &json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message,
            },
        }),
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0)
}

fn system_time_seconds(value: SystemTime) -> Option<f64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_to_json_converts_nested_display_values() {
        let value = serde_yaml::from_str::<YamlValue>(
            "display:\n  tui_auto_resume_recent: true\n  sections:\n    thinking: expanded\n",
        )
        .unwrap();
        let json = yaml_to_json(&value);
        assert_eq!(
            json["display"]["tui_auto_resume_recent"],
            JsonValue::Bool(true)
        );
        assert_eq!(
            json["display"]["sections"]["thinking"],
            JsonValue::String(String::from("expanded"))
        );
    }

    #[test]
    fn render_message_content_handles_text_arrays() {
        let value = json!([
            {"text": "hello"},
            {"text": "world"}
        ]);
        assert_eq!(
            render_message_content(Some(&value)),
            Some(String::from("hello\nworld"))
        );
    }
}
