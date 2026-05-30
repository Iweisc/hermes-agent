use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local, Utc};
use hermes_core::{
    DelegateExecutor, HermesContext, HermesError, MessageAppend, MessageRecord, ModelOverrides,
    SessionCreate, SessionStore, ToolRuntime,
};
use serde_json::{Map, Value, json};
use url::Url;

const DENY_SOURCES: &[&str] = &["tool"];
const MAX_COMPLETION_ITEMS: usize = 30;

const COMMANDS_CATALOG_HELPER: &str = r#"
import json
import sys

from hermes_cli.commands import COMMAND_REGISTRY, SUBCOMMANDS, _build_description
from hermes_cli.config import read_raw_config

TUI_HIDDEN = {"new", "clear", "quit", "exit", "copy", "paste", "image", "commands", "approve", "deny", "sethome", "set-home", "update"}
TUI_EXTRA = [
    ("/compact", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    ("/mouse", "Toggle mouse/wheel tracking [on|off|toggle]", "TUI"),
]

all_pairs = []
canon = {}
categories = []
cat_map = {}
cat_order = []

for cmd in COMMAND_REGISTRY:
    if cmd.name in TUI_HIDDEN or cmd.gateway_only:
        continue
    c = f"/{cmd.name}"
    canon[c.lower()] = c
    for a in cmd.aliases:
        canon[f"/{a}".lower()] = c
    desc = _build_description(cmd)
    all_pairs.append([c, desc])
    if cmd.category not in cat_map:
        cat_map[cmd.category] = []
        cat_order.append(cmd.category)
    cat_map[cmd.category].append([c, desc])

for name, desc, cat in TUI_EXTRA:
    all_pairs.append([name, desc])
    if cat not in cat_map:
        cat_map[cat] = []
        cat_order.append(cat)
    cat_map[cat].append([name, desc])

warning = ""
try:
    qcmds = (read_raw_config() or {}).get("quick_commands", {}) or {}
    if isinstance(qcmds, dict) and qcmds:
        bucket = "User commands"
        if bucket not in cat_map:
            cat_map[bucket] = []
            cat_order.append(bucket)
        for qname, qc in sorted(qcmds.items()):
            if not isinstance(qc, dict):
                continue
            key = f"/{qname}"
            canon[key.lower()] = key
            qtype = qc.get("type", "")
            if qtype == "exec":
                default_desc = f"exec: {qc.get('command', '')}"
            elif qtype == "alias":
                default_desc = f"alias -> {qc.get('target', '')}"
            else:
                default_desc = qtype or "quick command"
            qdesc = str(qc.get("description") or default_desc)
            qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
            all_pairs.append([key, qdesc])
            cat_map[bucket].append([key, qdesc])
except Exception as e:
    warning = f"quick_commands discovery unavailable: {e}"

skill_count = 0
try:
    from agent.skill_commands import scan_skill_commands

    for key, info in sorted(scan_skill_commands().items()):
        desc = str(info.get("description", "Skill"))
        all_pairs.append([key, desc[:120] + ("…" if len(desc) > 120 else "")])
        skill_count += 1
except Exception as e:
    warning = f"skill discovery unavailable: {e}"

for cat in cat_order:
    categories.append({"name": cat, "pairs": cat_map[cat]})

print(json.dumps({
    "pairs": all_pairs,
    "sub": {k: list(v) for k, v in SUBCOMMANDS.items()},
    "canon": canon,
    "categories": categories,
    "skill_count": skill_count,
    "warning": warning,
}))
"#;

const SLASH_COMPLETION_HELPER: &str = r#"
import json
import sys

from hermes_cli.commands import SlashCommandCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text

from agent.skill_commands import get_skill_commands

payload = json.load(sys.stdin)
text = str(payload.get("text", ""))
completer = SlashCommandCompleter(skill_commands_provider=lambda: get_skill_commands())
doc = Document(text, len(text))
items = [{
    "text": c.text,
    "display": c.display or c.text,
    "meta": to_plain_text(c.display_meta) if c.display_meta else "",
} for c in completer.get_completions(doc, None)][:30]

text_lower = text.lower()
for extra in [
    {"text": "/compact", "display": "/compact", "meta": "Toggle compact display mode"},
    {"text": "/details", "display": "/details", "meta": "Control agent detail visibility"},
    {"text": "/logs", "display": "/logs", "meta": "Show recent gateway log lines"},
    {"text": "/mouse", "display": "/mouse", "meta": "Toggle mouse/wheel tracking [on|off|toggle]"},
]:
    if extra["text"].startswith(text_lower) and not any(item["text"] == extra["text"] for item in items):
        items.append(extra)

print(json.dumps({
    "items": items,
    "replace_from": text.rfind(" ") + 1 if " " in text else 1,
}))
"#;

const COMMAND_RESOLVE_HELPER: &str = r#"
import json
import sys

from hermes_cli.commands import resolve_command

payload = json.load(sys.stdin)
name = str(payload.get("name", ""))
result = resolve_command(name)
if result is None:
    raise SystemExit(2)
print(json.dumps({
    "canonical": result.name,
    "description": result.description,
    "category": result.category,
}))
"#;

const COMMAND_DISPATCH_HELPER: &str = r#"
import json
import subprocess
import sys

from agent.skill_commands import build_skill_invocation_message, scan_skill_commands
from hermes_cli.commands import resolve_command
from hermes_cli.config import read_raw_config
from hermes_cli.plugins import get_plugin_command_handler, resolve_plugin_command_result

payload = json.load(sys.stdin)
name = str(payload.get("name", "")).lstrip("/")
arg = str(payload.get("arg", ""))
session_key = str(payload.get("session_key", "") or "")

resolved = resolve_command(name)
if resolved is not None:
    name = resolved.name

qcmds = (read_raw_config() or {}).get("quick_commands", {}) or {}
if name in qcmds:
    qc = qcmds[name]
    if qc.get("type") == "exec":
        result = subprocess.run(
            qc.get("command", ""),
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = ((result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")).strip()[:4000]
        if result.returncode != 0:
            raise RuntimeError(output or f"quick command failed with exit code {result.returncode}")
        print(json.dumps({"type": "exec", "output": output}))
        raise SystemExit(0)
    if qc.get("type") == "alias":
        print(json.dumps({"type": "alias", "target": qc.get("target", "")}))
        raise SystemExit(0)

handler = get_plugin_command_handler(name)
if handler:
    result = resolve_plugin_command_result(handler(arg))
    print(json.dumps({"type": "plugin", "output": str(result or "")}))
    raise SystemExit(0)

cmds = scan_skill_commands()
key = f"/{name}"
if key in cmds:
    msg = build_skill_invocation_message(key, arg, task_id=session_key)
    if msg:
        print(json.dumps({
            "type": "skill",
            "message": msg,
            "name": cmds[key].get("name", name),
        }))
        raise SystemExit(0)

if name in ("queue", "q"):
    if not arg:
        raise ValueError("usage: /queue <prompt>")
    print(json.dumps({"type": "send", "message": arg}))
    raise SystemExit(0)

if name == "steer":
    if not arg:
        raise ValueError("usage: /steer <prompt>")
    print(json.dumps({"type": "send", "message": arg}))
    raise SystemExit(0)

raise RuntimeError(f"unknown command: {name}")
"#;

const SLASH_EXEC_HELPER: &str = r#"
import contextlib
import io
import json
import sys

import cli as cli_mod
from cli import HermesCLI
from rich.console import Console

payload = json.load(sys.stdin)
command = str(payload.get("command", "")).strip()
session_key = str(payload.get("session_key", "") or "")

if not command:
    raise ValueError("empty command")
if not command.startswith("/"):
    command = f"/{command}"

buf = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    cli = HermesCLI(model=None, compact=True, resume=session_key or None, verbose=False)

cli.console = Console(file=buf, force_terminal=True, width=120)
old = getattr(cli_mod, "_cprint", None)
if old is not None:
    cli_mod._cprint = lambda text: print(text)

try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        cli.process_command(command)
finally:
    if old is not None:
        cli_mod._cprint = old

print(json.dumps({"output": buf.getvalue().rstrip()}))
"#;

const INITIAL_SESSION_INFO_HELPER: &str = r#"
import json
import os
import sys

result = {
    "model": "",
    "tools": {},
    "skills": {},
    "cwd": os.getenv("TERMINAL_CWD", os.getcwd()),
    "lazy": True,
    "version": "",
    "release_date": "",
}

try:
    from tui_gateway.server import _resolve_model

    result["model"] = _resolve_model() or ""
except Exception:
    pass

try:
    from hermes_cli import __version__, __release_date__

    result["version"] = __version__
    result["release_date"] = __release_date__
except Exception:
    pass

sys.__stdout__.write(json.dumps(result))
sys.__stdout__.flush()
"#;

const GATEWAY_READY_HELPER: &str = r#"
import json
import sys

result = {"skin": {}}
try:
    from tui_gateway.server import resolve_skin

    result["skin"] = resolve_skin()
except Exception:
    pass

sys.__stdout__.write(json.dumps(result))
sys.__stdout__.flush()
"#;

const CONFIG_SET_HELPER: &str = r#"
import json
import sys

from tui_gateway.server import dispatch

payload = json.load(sys.stdin)
response = dispatch({
    "jsonrpc": "2.0",
    "id": "cfg",
    "method": "config.set",
    "params": payload,
})
sys.__stdout__.write(json.dumps(response))
sys.__stdout__.flush()
"#;

const CONFIG_GET_HELPER: &str = r#"
import json
import sys

from tui_gateway.server import dispatch

payload = json.load(sys.stdin)
response = dispatch({
    "jsonrpc": "2.0",
    "id": "cfg",
    "method": "config.get",
    "params": payload,
})
sys.__stdout__.write(json.dumps(response))
sys.__stdout__.flush()
"#;

const DISPATCH_RPC_HELPER: &str = r#"
import json
import sys

from tui_gateway.server import dispatch

payload = json.load(sys.stdin)
response = dispatch({
    "jsonrpc": "2.0",
    "id": "helper",
    "method": payload.get("method", ""),
    "params": payload.get("params", {}) or {},
})
sys.__stdout__.write(json.dumps(response))
sys.__stdout__.flush()
"#;

const SKIN_PAYLOAD_HELPER: &str = r#"
import json
import sys

from tui_gateway.server import resolve_skin

sys.__stdout__.write(json.dumps(resolve_skin()))
sys.__stdout__.flush()
"#;

#[derive(Debug, Clone)]
struct HelperContext {
    context: HermesContext,
    hermes_home: PathBuf,
    project_root: PathBuf,
    python: PathBuf,
    python_path: String,
    work_root: PathBuf,
}

#[derive(Debug, Clone)]
struct SessionBinding {
    attached_images: Vec<String>,
    child_id: String,
    cols: u64,
    info: Map<String, Value>,
    image_counter: u64,
    running: bool,
    store_session_id: Option<String>,
    store_session_dirty: bool,
    usage: Map<String, Value>,
}

#[derive(Debug, Clone)]
struct PendingRequest {
    local_session_id: Option<String>,
    method: String,
    resume_target: Option<String>,
    response_tx: Option<Sender<Value>>,
    suppress_output: bool,
}

#[derive(Debug, Clone)]
struct PromptRequestBinding {
    child_request_id: String,
    local_session_id: String,
}

#[derive(Debug, Default)]
struct ProxyState {
    child_to_local: HashMap<String, String>,
    next_internal_request: u64,
    next_local_session: u64,
    next_paste: u64,
    next_prompt_request: u64,
    pending: HashMap<String, PendingRequest>,
    prompt_requests: HashMap<String, PromptRequestBinding>,
    sessions: HashMap<String, SessionBinding>,
}

pub fn run(context: HermesContext) -> Result<(), Box<dyn Error>> {
    let store = Arc::new(context.open_session_store()?);
    let stdout = Arc::new(Mutex::new(io::stdout()));
    let state = Arc::new(Mutex::new(ProxyState::default()));
    let session_cwd = std::env::var("HERMES_CWD")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .display()
                .to_string()
        });
    let project_root = project_root();
    let python = resolve_repo_python(&project_root)
        .ok_or_else(|| "unable to resolve Python interpreter for tui_gateway proxy".to_string())?;
    let work_root = PathBuf::from(&session_cwd);
    let helper = HelperContext {
        context: context.clone(),
        hermes_home: context.hermes_home(),
        project_root: project_root.clone(),
        python: python.clone(),
        python_path: compose_python_path(&project_root),
        work_root,
    };

    let mut child = Command::new(&python)
        .args(["-m", "tui_gateway.worker"])
        .current_dir(&session_cwd)
        .env("HERMES_PYTHON_SRC_ROOT", &project_root)
        .env("PYTHONPATH", &helper.python_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let child_stdin = Arc::new(Mutex::new(
        child
            .stdin
            .take()
            .ok_or_else(|| "failed to capture child stdin".to_string())?,
    ));
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture child stdout".to_string())?;
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture child stderr".to_string())?;

    pipe_child_stdout(child_stdout, Arc::clone(&stdout), Arc::clone(&state));
    pipe_child_stderr(child_stderr);

    let ready_payload =
        run_python_helper_json(&helper, GATEWAY_READY_HELPER, None).unwrap_or_else(|_| json!({}));
    write_json(
        &stdout,
        &json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "gateway.ready", "payload": ready_payload},
        }),
    )?;

    let stdin = io::stdin();
    for raw in stdin.lock().lines() {
        let raw = raw?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => {
                write_json(
                    &stdout,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {"code": -32700, "message": "parse error"},
                    }),
                )?;
                continue;
            }
        };

        if let Some(response) =
            handle_native_request(&request, &store, &state, &helper, &stdout, &child_stdin)?
        {
            write_json(&stdout, &response)?;
            continue;
        }

        forward_request(&child_stdin, &state, &request)?;
    }

    Ok(())
}

fn handle_native_request(
    request: &Value,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
    stdout: &Arc<Mutex<io::Stdout>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return Ok(None);
    };
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let params = request
        .get("params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let response = match method {
        "session.create" => handle_session_create(id, &params, store, state, helper)?,
        "session.list" => {
            let limit = clamp_limit(params.get("limit").and_then(Value::as_i64).unwrap_or(200));
            let fetch_limit = std::cmp::max(limit * 2, 200);
            let deny = deny_sources();
            let rows = store.search_sessions(None, fetch_limit, 0)?;
            let sessions = rows
                .into_iter()
                .filter(|row| !deny.contains(row.source.trim().to_ascii_lowercase().as_str()))
                .take(limit as usize)
                .map(|row| {
                    json!({
                        "id": row.id,
                        "title": row.title.unwrap_or_default(),
                        "preview": row.preview,
                        "started_at": row.started_at,
                        "message_count": row.message_count,
                        "source": row.source,
                    })
                })
                .collect::<Vec<_>>();
            Some(ok_response(id, json!({ "sessions": sessions })))
        }
        "session.most_recent" => {
            let deny = deny_sources();
            let mut result = json!({ "session_id": Value::Null });
            for row in store.search_sessions(None, 200, 0)? {
                if deny.contains(row.source.trim().to_ascii_lowercase().as_str()) {
                    continue;
                }
                result = json!({
                    "session_id": row.id,
                    "title": row.title.unwrap_or_default(),
                    "started_at": row.started_at,
                    "source": row.source,
                });
                break;
            }
            Some(ok_response(id, result))
        }
        "session.delete" => {
            let Some(target) = params
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Ok(Some(error_response(id, 4006, "session_id required")));
            };
            if active_store_session_ids(state)?.contains(target) {
                return Ok(Some(error_response(
                    id,
                    4023,
                    "cannot delete an active session",
                )));
            }
            let deleted = store.delete_session(target)?;
            if !deleted {
                return Ok(Some(error_response(id, 4007, "session not found")));
            }
            Some(ok_response(id, json!({ "deleted": target })))
        }
        "session.resume" => handle_session_resume(id, &params, store, state, helper)?,
        "session.close" => handle_session_close(id, &params, store, state, child_stdin)?,
        "session.branch" => handle_session_branch(id, &params, store, state, helper)?,
        "session.title" => handle_session_title(id, &params, store, state)?,
        "session.save" => handle_session_save(id, &params, store, state)?,
        "session.undo" => handle_session_undo(id, &params, store, state, child_stdin)?,
        "session.compress" => handle_session_compress(id, &params, state, child_stdin)?,
        "session.history" => handle_session_history(id, &params, store, state)?,
        "rollback.list" => {
            handle_session_bound_child_request(id, "rollback.list", &params, state, child_stdin)?
        }
        "rollback.diff" => {
            handle_session_bound_child_request(id, "rollback.diff", &params, state, child_stdin)?
        }
        "rollback.restore" => {
            handle_session_bound_child_request(id, "rollback.restore", &params, state, child_stdin)?
        }
        "reload.mcp" => forward_child_request(id, "reload.mcp", &params, state, child_stdin, true)?,
        "browser.manage" => handle_browser_manage(id, &params, state, child_stdin)?,
        "tools.configure" => {
            forward_child_request(id, "tools.configure", &params, state, child_stdin, true)?
        }
        "skills.reload" => {
            forward_child_request(id, "skills.reload", &params, state, child_stdin, false)?
        }
        "skills.manage" => forward_child_request_timeout(
            id,
            "skills.manage",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(45),
        )?,
        "voice.toggle" => forward_child_request_timeout(
            id,
            "voice.toggle",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(20),
        )?,
        "voice.record" => forward_child_request_timeout(
            id,
            "voice.record",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(20),
        )?,
        "voice.tts" => forward_child_request_timeout(
            id,
            "voice.tts",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(20),
        )?,
        "shell.exec" => forward_child_request_timeout(
            id,
            "shell.exec",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(35),
        )?,
        "cli.exec" => forward_child_request_timeout(
            id,
            "cli.exec",
            &params,
            state,
            child_stdin,
            false,
            Duration::from_secs(240),
        )?,
        "plugins.list" => handle_helper_dispatch(id, "plugins.list", &params, helper)?,
        "config.show" => handle_helper_dispatch(id, "config.show", &params, helper)?,
        "insights.get" => handle_helper_dispatch(id, "insights.get", &params, helper)?,
        "cron.manage" => handle_helper_dispatch(id, "cron.manage", &params, helper)?,
        "tools.list" => forward_child_request(id, "tools.list", &params, state, child_stdin, true)?,
        "tools.show" => forward_child_request(id, "tools.show", &params, state, child_stdin, true)?,
        "toolsets.list" => {
            forward_child_request(id, "toolsets.list", &params, state, child_stdin, true)?
        }
        "agents.list" => {
            forward_child_request(id, "agents.list", &params, state, child_stdin, false)?
        }
        "session.usage" => handle_session_usage(id, &params, state)?,
        "session.status" => handle_session_status(id, &params, store, state, helper)?,
        "setup.status" => handle_helper_dispatch(id, "setup.status", &params, helper)?,
        "delegation.status" => {
            forward_child_request(id, "delegation.status", &params, state, child_stdin, false)?
        }
        "delegation.pause" => {
            forward_child_request(id, "delegation.pause", &params, state, child_stdin, false)?
        }
        "subagent.interrupt" => {
            forward_child_request(id, "subagent.interrupt", &params, state, child_stdin, false)?
        }
        "spawn_tree.save" => handle_helper_dispatch(id, "spawn_tree.save", &params, helper)?,
        "spawn_tree.list" => handle_helper_dispatch(id, "spawn_tree.list", &params, helper)?,
        "spawn_tree.load" => handle_helper_dispatch(id, "spawn_tree.load", &params, helper)?,
        "process.stop" => {
            forward_child_request(id, "process.stop", &params, state, child_stdin, false)?
        }
        "reload.env" => {
            forward_child_request(id, "reload.env", &params, state, child_stdin, false)?
        }
        "paste.collapse" => handle_paste_collapse(id, &params, state, helper)?,
        "model.options" => {
            forward_child_request(id, "model.options", &params, state, child_stdin, true)?
        }
        "model.save_key" => {
            forward_child_request(id, "model.save_key", &params, state, child_stdin, true)?
        }
        "model.disconnect" => {
            forward_child_request(id, "model.disconnect", &params, state, child_stdin, true)?
        }
        "config.get" => handle_config_get(id, &params, state, helper, child_stdin)?,
        "config.set" => handle_config_set(id, &params, state, helper, stdout, child_stdin)?,
        "prompt.background" => handle_prompt_background(id, &params, state, helper, stdout)?,
        "prompt.submit" => handle_prompt_submit(id, &params, state, child_stdin)?,
        "session.interrupt" => handle_session_interrupt(id, &params, state, child_stdin)?,
        "session.steer" => handle_session_steer(id, &params, state, child_stdin)?,
        "approval.respond" => handle_approval_respond(id, &params, state, helper)?,
        "clarify.respond" => handle_text_respond(
            id,
            &params,
            child_stdin,
            state,
            "clarify.respond",
            "request_id",
            "answer",
        )?,
        "sudo.respond" => handle_text_respond(
            id,
            &params,
            child_stdin,
            state,
            "sudo.respond",
            "request_id",
            "password",
        )?,
        "secret.respond" => handle_text_respond(
            id,
            &params,
            child_stdin,
            state,
            "secret.respond",
            "request_id",
            "value",
        )?,
        "terminal.resize" => handle_terminal_resize(id, &params, state, child_stdin)?,
        "clipboard.paste" => handle_clipboard_paste(id, &params, state, helper)?,
        "image.attach" => handle_image_attach(id, &params, state, helper)?,
        "input.detect_drop" => handle_input_detect_drop(id, &params, state, helper)?,
        "commands.catalog" => {
            let result = run_python_helper_json(helper, COMMANDS_CATALOG_HELPER, None)?;
            Some(ok_response(id, result))
        }
        "complete.path" => Some(ok_response(
            id,
            complete_path_response(
                params
                    .get("word")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &helper.work_root,
            ),
        )),
        "complete.slash" => {
            let text = params
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !text.starts_with('/') {
                Some(ok_response(id, json!({"items": []})))
            } else if let Some(details) = details_completions(text) {
                Some(ok_response(id, details))
            } else {
                let result = run_python_helper_json(
                    helper,
                    SLASH_COMPLETION_HELPER,
                    Some(&json!({"text": text})),
                )?;
                Some(ok_response(id, result))
            }
        }
        "command.resolve" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match run_python_helper_json(
                helper,
                COMMAND_RESOLVE_HELPER,
                Some(&json!({"name": name})),
            ) {
                Ok(result) => Some(ok_response(id, result)),
                Err(_) => Some(error_response(
                    id,
                    4011,
                    &format!("unknown command: {name}"),
                )),
            }
        }
        "command.dispatch" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arg = params
                .get("arg")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let session_key = params
                .get("session_id")
                .and_then(Value::as_str)
                .and_then(|local_id| lookup_store_session_id(state, local_id).ok().flatten())
                .unwrap_or_default();
            match run_python_helper_json(
                helper,
                COMMAND_DISPATCH_HELPER,
                Some(&json!({"name": name, "arg": arg, "session_key": session_key})),
            ) {
                Ok(result) => Some(ok_response(id, result)),
                Err(error) => Some(error_response(id, 4018, &error.to_string())),
            }
        }
        "slash.exec" => {
            let command = params
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if command.is_empty() {
                Some(error_response(id, 4004, "empty command"))
            } else {
                let session_key = params
                    .get("session_id")
                    .and_then(Value::as_str)
                    .and_then(|local_id| lookup_store_session_id(state, local_id).ok().flatten())
                    .unwrap_or_default();
                match run_python_helper_json(
                    helper,
                    SLASH_EXEC_HELPER,
                    Some(&json!({"command": command, "session_key": session_key})),
                ) {
                    Ok(result) => Some(ok_response(id, result)),
                    Err(error) => Some(error_response(id, 4018, &error.to_string())),
                }
            }
        }
        _ => None,
    };

    Ok(response)
}

fn clamp_limit(raw: i64) -> i64 {
    raw.clamp(1, 200)
}

fn handle_session_create(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let cols = params
        .get("cols")
        .and_then(Value::as_u64)
        .unwrap_or(80)
        .max(1);
    let info = initial_session_info(helper, None);
    let store_session_id = new_store_session_id();
    let model = info
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    store.create_session(&SessionCreate {
        id: store_session_id.clone(),
        source: "tui".to_string(),
        user_id: None,
        model,
        model_config: None,
        system_prompt: None,
        parent_session_id: None,
    })?;
    let session_id = create_local_session(state, Some(store_session_id), cols)?;
    if let Some(info_object) = info.as_object().cloned() {
        update_session_info_cache(state, &session_id, info_object.clone())?;
        if let Some(usage) = info_object.get("usage").and_then(Value::as_object) {
            update_session_usage_cache(state, &session_id, usage.clone())?;
        }
    }
    Ok(Some(ok_response(
        id,
        json!({
            "session_id": session_id,
            "info": info,
        }),
    )))
}

fn handle_session_resume(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(target) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(session) = resolve_resume_session(store, target)? else {
        return Ok(Some(error_response(id, 4007, "session not found")));
    };
    store.reopen_session(&session.id)?;
    let cols = params
        .get("cols")
        .and_then(Value::as_u64)
        .unwrap_or(80)
        .max(1);
    let info = initial_session_info(helper, session.model.as_deref());
    let messages = resume_messages(store, &session.id)?;
    let session_id = create_local_session(state, Some(session.id.clone()), cols)?;
    if let Some(info_object) = info.as_object().cloned() {
        update_session_info_cache(state, &session_id, info_object.clone())?;
        if let Some(usage) = info_object.get("usage").and_then(Value::as_object) {
            update_session_usage_cache(state, &session_id, usage.clone())?;
        }
    }
    Ok(Some(ok_response(
        id,
        json!({
            "session_id": session_id,
            "resumed": session.id,
            "message_count": messages.len(),
            "messages": messages,
            "info": info,
        }),
    )))
}

fn handle_session_close(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(ok_response(id, json!({"closed": false}))));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(ok_response(id, json!({"closed": false}))));
    };
    if !binding.child_id.is_empty() {
        send_internal_child_request(
            child_stdin,
            state,
            "session.close",
            json!({"session_id": binding.child_id}),
            Some(local_id.to_string()),
            None,
            None,
        )?;
    } else if let Some(store_session_id) = binding.store_session_id.as_deref() {
        if let Some(record) = store.get_session(store_session_id)? {
            if record.message_count == 0 && record.title.as_deref().unwrap_or_default().is_empty() {
                let _ = store.delete_session(store_session_id);
            } else {
                let _ = store.end_session(store_session_id, "closed");
            }
        }
    }
    release_local_session(state, local_id)?;
    Ok(Some(ok_response(id, json!({"closed": true}))))
}

fn handle_session_branch(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let Some(parent_session_id) = binding.store_session_id.as_deref() else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let history = store.get_messages(parent_session_id)?;
    if history.is_empty() {
        return Ok(Some(error_response(
            id,
            4008,
            "nothing to branch — send a message first",
        )));
    }

    let branch_name = params
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    let title = if branch_name.is_empty() {
        let current = store
            .get_session_title(parent_session_id)?
            .unwrap_or_else(|| "branch".to_string());
        store.get_next_title_in_lineage(&current)?
    } else {
        branch_name.to_string()
    };

    let store_session_id = new_store_session_id();
    let model = binding
        .info
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            store
                .get_session(parent_session_id)
                .ok()
                .flatten()
                .and_then(|row| row.model)
        });
    store.create_session(&SessionCreate {
        id: store_session_id.clone(),
        source: "tui".to_string(),
        user_id: None,
        model: model.clone(),
        model_config: None,
        system_prompt: None,
        parent_session_id: Some(parent_session_id.to_string()),
    })?;
    for message in history {
        store.append_message(&store_session_id, &message_record_to_append(&message))?;
    }
    store.set_session_title(&store_session_id, &title)?;

    let info = initial_session_info(helper, model.as_deref());
    let session_id = create_local_session(state, Some(store_session_id.clone()), binding.cols)?;
    if let Some(info_object) = info.as_object().cloned() {
        update_session_info_cache(state, &session_id, info_object.clone())?;
        if let Some(usage) = info_object.get("usage").and_then(Value::as_object) {
            update_session_usage_cache(state, &session_id, usage.clone())?;
        }
    }

    Ok(Some(ok_response(
        id,
        json!({
            "session_id": session_id,
            "title": title,
            "parent": parent_session_id,
        }),
    )))
}

fn resolve_resume_session(
    store: &SessionStore,
    target: &str,
) -> Result<Option<hermes_core::SessionRecord>, HermesError> {
    if let Some(session) = store.get_session(target)? {
        return Ok(Some(session));
    }
    if let Some(resolved) = store.resolve_session_by_title(target)? {
        return store.get_session(&resolved);
    }
    Ok(None)
}

fn initial_session_info(helper: &HelperContext, model_override: Option<&str>) -> Value {
    let mut info = run_python_helper_json(helper, INITIAL_SESSION_INFO_HELPER, None)
        .ok()
        .unwrap_or_else(|| {
            json!({
                "model": "",
                "tools": {},
                "skills": {},
                "cwd": helper.work_root.display().to_string(),
                "lazy": true,
                "version": env!("CARGO_PKG_VERSION"),
                "release_date": "",
            })
        });
    if let Some(model) = model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        && let Some(object) = info.as_object_mut()
    {
        object.insert("model".to_string(), json!(model));
    }
    info
}

fn new_store_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("tui_{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

fn handle_session_title(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(store_session_id) = lookup_clean_store_session_id(state, local_id)? else {
        return Ok(None);
    };
    let existing_row = match store.get_session(&store_session_id) {
        Ok(row) => row,
        Err(error) => return Ok(Some(error_response(id, 5007, &error.to_string()))),
    };
    let Some(existing_row) = existing_row else {
        return Ok(None);
    };

    if !params.contains_key("title") {
        return Ok(Some(ok_response(
            id,
            json!({
                "title": existing_row.title.unwrap_or_default(),
                "session_key": store_session_id,
            }),
        )));
    }

    let title = params
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if title.is_empty() {
        return Ok(Some(error_response(id, 4021, "title required")));
    }

    match store.set_session_title(&store_session_id, title) {
        Ok(true) => Ok(Some(ok_response(
            id,
            json!({"pending": false, "title": title}),
        ))),
        Ok(false) => Ok(Some(ok_response(
            id,
            json!({
                "pending": false,
                "title": existing_row.title.unwrap_or_else(|| title.to_string()),
            }),
        ))),
        Err(HermesError::State { detail, .. }) => Ok(Some(error_response(id, 4022, &detail))),
        Err(error) => Ok(Some(error_response(id, 5007, &error.to_string()))),
    }
}

fn handle_session_save(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(store_session_id) = lookup_store_session_id(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let messages = store
        .get_messages(&store_session_id)?
        .into_iter()
        .map(message_record_to_chat_message)
        .collect::<Result<Vec<_>, _>>()?;
    let model = binding
        .info
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            store
                .get_session(&store_session_id)
                .ok()
                .flatten()
                .and_then(|row| row.model)
        })
        .unwrap_or_default();
    let filename = std::env::current_dir()?.join(format!(
        "hermes_conversation_{}.json",
        current_filename_timestamp()
    ));
    fs::write(
        &filename,
        serde_json::to_vec_pretty(&json!({
            "model": model,
            "messages": messages,
        }))?,
    )?;
    Ok(Some(ok_response(
        id,
        json!({"file": filename.display().to_string()}),
    )))
}

fn handle_session_undo(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    if binding.running {
        return Ok(Some(error_response(
            id,
            4009,
            "session busy — /interrupt the current turn before /undo",
        )));
    }
    let Some(store_session_id) = binding.store_session_id.as_deref() else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };

    let records = store.get_messages(store_session_id)?;
    let (retained, removed) = trim_last_exchange(records);
    if removed <= 0 {
        return Ok(Some(ok_response(id, json!({"removed": 0}))));
    }

    store.clear_messages(store_session_id)?;
    for message in retained {
        store.append_message(store_session_id, &message_record_to_append(&message))?;
    }

    if let Some(child_id) = lookup_child_session_id(state, local_id)? {
        send_internal_child_request(
            child_stdin,
            state,
            "session.close",
            json!({"session_id": child_id}),
            Some(local_id.to_string()),
            None,
            Some("child.close"),
        )?;
    }

    Ok(Some(ok_response(id, json!({"removed": removed}))))
}

fn handle_session_usage(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    Ok(Some(ok_response(
        id,
        Value::Object(usage_payload(&binding)),
    )))
}

fn handle_session_status(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };

    let session_key = binding
        .store_session_id
        .clone()
        .unwrap_or_else(|| local_id.to_string());
    let record = binding
        .store_session_id
        .as_deref()
        .map(|session_id| store.get_session(session_id))
        .transpose()?
        .flatten();
    let usage = usage_payload(&binding);
    let model = binding
        .info
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| record.as_ref().and_then(|row| row.model.clone()))
        .unwrap_or_else(|| "(unknown)".to_string());
    let title = record
        .as_ref()
        .and_then(|row| row.title.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let created = record
        .as_ref()
        .map(|row| row.started_at)
        .unwrap_or_else(current_timestamp_seconds);
    let last_activity = if binding.running {
        current_timestamp_seconds()
    } else {
        created
    };

    let mut lines = vec![
        "Hermes TUI Status".to_string(),
        String::new(),
        format!("Session ID: {session_key}"),
        format!("Path: {}", helper.hermes_home.display()),
    ];
    if let Some(title) = title {
        lines.push(format!("Title: {title}"));
    }
    lines.extend([
        format!("Model: {model} (unknown)"),
        format!("Created: {}", format_timestamp(created)),
        format!("Last Activity: {}", format_timestamp(last_activity)),
        format!(
            "Tokens: {}",
            usage.get("total").and_then(Value::as_i64).unwrap_or(0)
        ),
        format!(
            "Agent Running: {}",
            if binding.running { "Yes" } else { "No" }
        ),
    ]);
    Ok(Some(ok_response(id, json!({"output": lines.join("\n")}))))
}

fn handle_session_compress(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    mark_store_session_dirty(state, local_id)?;
    ensure_child_session(child_stdin, state, local_id)?;
    let Some(child_id) = lookup_child_session_id(state, local_id)? else {
        return Ok(Some(error_response(
            id,
            4001,
            "child session not materialized",
        )));
    };

    let mut forwarded_params = params.clone();
    forwarded_params.insert("session_id".to_string(), json!(child_id));
    let response = send_blocking_child_request(
        child_stdin,
        state,
        "session.compress",
        Value::Object(forwarded_params),
        Some(local_id.to_string()),
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_session_history(
    id: Value,
    params: &Map<String, Value>,
    store: &SessionStore,
    state: &Arc<Mutex<ProxyState>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(store_session_id) = lookup_store_session_id(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let Some(_) = store.get_session(&store_session_id)? else {
        return Ok(Some(error_response(id, 4007, "session not found")));
    };
    let (count, messages) = history_messages(store, &store_session_id)?;
    Ok(Some(ok_response(
        id,
        json!({
            "count": count,
            "messages": messages,
        }),
    )))
}

fn handle_session_bound_child_request(
    id: Value,
    method: &str,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }
    ensure_child_session(child_stdin, state, local_id)?;
    let Some(child_id) = lookup_child_session_id(state, local_id)? else {
        return Ok(Some(error_response(
            id,
            4001,
            "child session not materialized",
        )));
    };

    let mut forwarded_params = params.clone();
    forwarded_params.insert("session_id".to_string(), json!(child_id));
    let response = send_blocking_child_request(
        child_stdin,
        state,
        method,
        Value::Object(forwarded_params),
        Some(local_id.to_string()),
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_browser_manage(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let mut forwarded_params = params.clone();
    let local_session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if let Some(local_id) = local_session_id.as_deref()
        && session_exists(state, local_id)?
    {
        ensure_child_session(child_stdin, state, local_id)?;
        if let Some(child_id) = lookup_child_session_id(state, local_id)? {
            forwarded_params.insert("session_id".to_string(), json!(child_id));
        } else {
            forwarded_params.remove("session_id");
        }
    }
    let response = send_blocking_child_request_timeout(
        child_stdin,
        state,
        "browser.manage",
        Value::Object(forwarded_params),
        local_session_id,
        Duration::from_secs(20),
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_helper_dispatch(
    id: Value,
    method: &str,
    params: &Map<String, Value>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let response = run_python_helper_json(
        helper,
        DISPATCH_RPC_HELPER,
        Some(&json!({"method": method, "params": params})),
    )
    .map_err(|error| format!("{method} helper failed: {error}"))?;
    Ok(Some(rebind_response_id(response, id)))
}

fn forward_child_request(
    id: Value,
    method: &str,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
    map_session_id: bool,
) -> Result<Option<Value>, Box<dyn Error>> {
    forward_child_request_timeout(
        id,
        method,
        params,
        state,
        child_stdin,
        map_session_id,
        Duration::from_secs(5),
    )
}

fn forward_child_request_timeout(
    id: Value,
    method: &str,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
    map_session_id: bool,
    timeout: Duration,
) -> Result<Option<Value>, Box<dyn Error>> {
    let mut forwarded_params = params.clone();
    if map_session_id
        && let Some(local_id) = params
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    {
        if session_exists(state, local_id)? {
            ensure_child_session(child_stdin, state, local_id)?;
            if let Some(child_id) = lookup_child_session_id(state, local_id)? {
                forwarded_params.insert("session_id".to_string(), json!(child_id));
            } else {
                forwarded_params.remove("session_id");
            }
        }
    }
    let response = send_blocking_child_request_timeout(
        child_stdin,
        state,
        method,
        Value::Object(forwarded_params),
        if map_session_id {
            params
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        } else {
            None
        },
        timeout,
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_config_get(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let key = params
        .get("key")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    let local_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if local_id.is_some()
        && requires_child_config_get(key)
        && let Some(child_id) = lookup_child_session_id(state, local_id.unwrap_or_default())?
    {
        let mut forwarded_params = params.clone();
        forwarded_params.insert("session_id".to_string(), json!(child_id));
        let response = send_blocking_child_request(
            child_stdin,
            state,
            "config.get",
            Value::Object(forwarded_params),
            Some(local_id.unwrap_or_default().to_string()),
        )?;
        return Ok(Some(rebind_response_id(response, id)));
    }

    let response = run_python_helper_json(helper, CONFIG_GET_HELPER, Some(&json!(params)))
        .map_err(|error| format!("config.get helper failed: {error}"))?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_config_set(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
    stdout: &Arc<Mutex<io::Stdout>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(key) = params
        .get("key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4002, "config key required")));
    };
    let local_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if local_id.is_some() && requires_child_config_set(key) {
        let local_id = local_id.unwrap_or_default();
        if !session_exists(state, local_id)? {
            return Ok(Some(error_response(id, 4001, "session not found")));
        }
        ensure_child_session(child_stdin, state, local_id)?;
        let Some(child_id) = lookup_child_session_id(state, local_id)? else {
            return Ok(Some(error_response(
                id,
                4001,
                "child session not materialized",
            )));
        };

        let mut forwarded_params = params.clone();
        forwarded_params.insert("session_id".to_string(), json!(child_id));
        let response = send_blocking_child_request(
            child_stdin,
            state,
            "config.set",
            Value::Object(forwarded_params),
            Some(local_id.to_string()),
        )?;
        return Ok(Some(rebind_response_id(response, id)));
    }

    let response = run_python_helper_json(helper, CONFIG_SET_HELPER, Some(&json!(params)))
        .map_err(|error| format!("config.set helper failed: {error}"))?;
    let response = rebind_response_id(response, id);
    if key == "skin" && response.get("result").is_some() {
        let skin =
            run_python_helper_json(helper, SKIN_PAYLOAD_HELPER, None).unwrap_or_else(|_| json!({}));
        write_json(
            stdout,
            &json!({
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "skin.changed",
                    "payload": skin,
                },
            }),
        )?;
    }
    Ok(Some(response))
}

fn handle_paste_collapse(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let text = params
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if text.is_empty() {
        return Ok(Some(error_response(id, 4004, "empty paste")));
    }

    let counter = next_paste_counter(state)?;
    let line_count = text.matches('\n').count() + 1;
    let paste_dir = helper.hermes_home.join("pastes");
    fs::create_dir_all(&paste_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let paste_file = paste_dir.join(format!("paste_{counter}_{timestamp}.txt"));
    fs::write(&paste_file, text)?;

    Ok(Some(ok_response(
        id,
        json!({
            "placeholder": format!("[Pasted text #{counter}: {line_count} lines → {}]", paste_file.display()),
            "path": paste_file.display().to_string(),
            "lines": line_count,
        }),
    )))
}

fn requires_child_config_get(key: &str) -> bool {
    matches!(key, "fast")
}

fn requires_child_config_set(key: &str) -> bool {
    matches!(
        key,
        "model" | "fast" | "verbose" | "yolo" | "reasoning" | "personality"
    )
}

fn handle_prompt_background(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
    stdout: &Arc<Mutex<io::Stdout>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let text = params
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    if text.is_empty() {
        return Ok(Some(error_response(id, 4012, "text required")));
    }

    let task_id = new_background_task_id();
    let event_task_id = task_id.clone();
    let parent = local_id.to_string();
    let context = helper.context.clone();
    let work_root = helper.work_root.clone();
    let stdout = Arc::clone(stdout);
    let model = binding
        .info
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    thread::spawn(move || {
        let result = run_background_prompt(&context, &work_root, &text, model);
        let body = match result {
            Ok(text) => text,
            Err(error) => format!("error: {error}"),
        };
        let _ = write_json(
            &stdout,
            &json!({
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "background.complete",
                    "session_id": parent,
                    "payload": {"task_id": event_task_id, "text": body},
                },
            }),
        );
    });

    Ok(Some(ok_response(id, json!({"task_id": task_id}))))
}

fn handle_session_interrupt(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    // Pending dangerous-command approvals live in the long-lived
    // `tui_gateway.worker` child process (tools.approval._gateway_queues),
    // not in this proxy. Forwarding `session.interrupt` below tears the agent
    // turn down in that process, where unregister_gateway_notify/clear_session
    // already cancel every blocked approval wait. A fresh `python3 -c` helper
    // here would only mutate empty module globals (resolved=0), so it is
    // omitted as a confirmed no-op.
    if !binding.child_id.is_empty() {
        send_internal_child_request(
            child_stdin,
            state,
            "session.interrupt",
            json!({"session_id": binding.child_id}),
            None,
            None,
            None,
        )?;
    }
    Ok(Some(ok_response(id, json!({"status": "interrupted"}))))
}

fn handle_prompt_submit(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    mark_store_session_dirty(state, local_id)?;
    ensure_child_session(child_stdin, state, local_id)?;
    sync_pending_images(child_stdin, state, local_id)?;
    let Some(child_id) = lookup_child_session_id(state, local_id)? else {
        return Ok(Some(error_response(
            id,
            4001,
            "child session not materialized",
        )));
    };

    let mut forwarded_params = params.clone();
    forwarded_params.insert("session_id".to_string(), json!(child_id));
    let response = send_blocking_child_request(
        child_stdin,
        state,
        "prompt.submit",
        Value::Object(forwarded_params),
        Some(local_id.to_string()),
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_session_steer(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Ok(Some(error_response(id, 4001, "session not found")));
    };
    let text = params
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        return Ok(Some(error_response(id, 4002, "text is required")));
    }
    if !binding.child_id.is_empty() {
        let response = send_blocking_child_request(
            child_stdin,
            state,
            "session.steer",
            json!({"session_id": binding.child_id, "text": text}),
            Some(local_id.to_string()),
        )?;
        return Ok(Some(rebind_response_id(response, id)));
    }
    Ok(Some(ok_response(
        id,
        json!({"status": "rejected", "text": text}),
    )))
}

fn handle_approval_respond(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    _helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    let choice = params
        .get("choice")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("deny");
    if !matches!(choice, "once" | "session" | "always" | "deny") {
        return Ok(Some(error_response(id, 4000, "invalid approval choice")));
    }

    if lookup_store_session_id(state, local_id)?.is_none() {
        return Ok(None);
    }

    // Pending dangerous-command approvals are queued inside the long-lived
    // `tui_gateway.worker` child process (tools.approval._gateway_queues), keyed
    // by session and unblocked there by its own `approval.respond` handler.
    // This proxy keeps no approval queue of its own, and the worker's
    // _ALLOWED_METHODS deliberately excludes `approval.respond`, so there is no
    // in-process queue here to resolve. Spawning a fresh `python3 -c` helper
    // would import tools.approval with empty module globals and always resolve
    // 0 entries. Return that count natively without the subprocess round-trip,
    // preserving the {"resolved": <int>} shape callers expect.
    Ok(Some(ok_response(id, json!({"resolved": 0}))))
}

fn handle_text_respond(
    id: Value,
    params: &Map<String, Value>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    method: &str,
    request_key: &str,
    value_key: &str,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(request_id) = params
        .get(request_key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(
            id,
            4009,
            &format!("{request_key} required"),
        )));
    };
    let Some(binding) = take_prompt_request(state, request_id)? else {
        return Ok(Some(error_response(
            id,
            4009,
            &format!("no pending {value_key} request"),
        )));
    };
    let value = params
        .get(value_key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let response = send_blocking_child_request(
        child_stdin,
        state,
        method,
        json!({request_key: binding.child_request_id, value_key: value}),
        None,
    )?;
    Ok(Some(rebind_response_id(response, id)))
}

fn handle_terminal_resize(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    child_stdin: &Arc<Mutex<ChildStdin>>,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    let cols = params
        .get("cols")
        .and_then(Value::as_u64)
        .unwrap_or(80)
        .max(1);
    set_session_cols(state, local_id, cols)?;
    if let Some(child_id) = lookup_child_session_id(state, local_id)? {
        send_internal_child_request(
            child_stdin,
            state,
            "terminal.resize",
            json!({"session_id": child_id, "cols": cols}),
            None,
            None,
            None,
        )?;
    }
    Ok(Some(ok_response(id, json!({"cols": cols}))))
}

fn handle_clipboard_paste(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    let image_counter = bump_session_image_counter(state, local_id)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = helper
        .hermes_home
        .join("images")
        .join(format!("clip_{timestamp}_{image_counter}.png"));
    let result = clipboard_paste_result(&path);
    if result
        .get("attached")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        queue_attached_image(state, local_id, &path.display().to_string())?;
        let count = queued_image_count(state, local_id)?;
        let mut payload = result.as_object().cloned().unwrap_or_default();
        payload.insert("count".to_string(), json!(count));
        return Ok(Some(ok_response(id, Value::Object(payload))));
    }
    Ok(Some(ok_response(id, result)))
}

/// Save the system clipboard image to `path` natively and build the response
/// payload. Mirrors the former `CLIPBOARD_PASTE_HELPER` Python helper:
/// `{"attached": true, "path", "name", "width", "height", "token_estimate"}`
/// on success (width/height/token_estimate omitted if the image can't be
/// decoded), or `{"attached": false, "message"}` when no image is found.
fn clipboard_paste_result(path: &Path) -> Value {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if !hermes_core::save_clipboard_image(path) {
        let message = if hermes_core::has_clipboard_image() {
            "Clipboard has image but extraction failed"
        } else {
            "No image found in clipboard"
        };
        return json!({"attached": false, "message": message});
    }

    let mut payload = Map::new();
    payload.insert("attached".to_string(), json!(true));
    payload.insert("path".to_string(), json!(path.display().to_string()));
    payload.insert(
        "name".to_string(),
        json!(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        ),
    );
    if let Some((width, height)) = hermes_core::image_dimensions(path) {
        payload.insert("width".to_string(), json!(width));
        payload.insert("height".to_string(), json!(height));
        payload.insert(
            "token_estimate".to_string(),
            json!(hermes_core::image_token_estimate(width, height)),
        );
    }
    Value::Object(payload)
}

fn handle_image_attach(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    let raw = params
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if raw.is_empty() {
        return Ok(Some(error_response(id, 4015, "path required")));
    }
    let (image_path, remainder) = match resolve_image_attachment(&raw, &helper.work_root) {
        Ok(result) => result,
        Err(message) => return Ok(Some(error_response(id, 4016, &message))),
    };

    queue_attached_image(state, local_id, &image_path.display().to_string())?;
    let count = queued_image_count(state, local_id)?;
    let mut payload = image_meta(helper, &image_path);
    payload.insert("attached".to_string(), json!(true));
    payload.insert("path".to_string(), json!(image_path.display().to_string()));
    payload.insert("count".to_string(), json!(count));
    payload.insert("remainder".to_string(), json!(remainder));
    payload.insert(
        "text".to_string(),
        json!(if remainder.is_empty() {
            format!(
                "[User attached image: {}]",
                image_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("image")
            )
        } else {
            remainder.clone()
        }),
    );
    Ok(Some(ok_response(id, Value::Object(payload))))
}

fn handle_input_detect_drop(
    id: Value,
    params: &Map<String, Value>,
    state: &Arc<Mutex<ProxyState>>,
    helper: &HelperContext,
) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(local_id) = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Some(error_response(id, 4006, "session_id required")));
    };
    if !session_exists(state, local_id)? {
        return Ok(Some(error_response(id, 4001, "session not found")));
    }

    let raw = params
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Some(drop) = detect_file_drop(raw, &helper.work_root) else {
        return Ok(Some(ok_response(id, json!({"matched": false}))));
    };
    if drop.is_image {
        queue_attached_image(state, local_id, &drop.path.display().to_string())?;
        let count = queued_image_count(state, local_id)?;
        let mut payload = image_meta(helper, &drop.path);
        payload.insert("matched".to_string(), json!(true));
        payload.insert("is_image".to_string(), json!(true));
        payload.insert("count".to_string(), json!(count));
        payload.insert("path".to_string(), json!(drop.path.display().to_string()));
        payload.insert(
            "text".to_string(),
            json!(if drop.remainder.is_empty() {
                format!(
                    "[User attached image: {}]",
                    drop.path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("image")
                )
            } else {
                drop.remainder.clone()
            }),
        );
        return Ok(Some(ok_response(id, Value::Object(payload))));
    }

    Ok(Some(ok_response(
        id,
        json!({
            "matched": true,
            "is_image": false,
            "name": drop.path.file_name().and_then(|name| name.to_str()).unwrap_or_default(),
            "text": if drop.remainder.is_empty() {
                format!("[User attached file: {}]", drop.path.display())
            } else {
                format!("[User attached file: {}]\n{}", drop.path.display(), drop.remainder)
            },
        }),
    )))
}

#[derive(Debug)]
struct DetectedDrop {
    path: PathBuf,
    is_image: bool,
    remainder: String,
}

fn deny_sources() -> HashSet<&'static str> {
    DENY_SOURCES.iter().copied().collect()
}

fn pipe_child_stdout(
    stdout: ChildStdout,
    sink: Arc<Mutex<io::Stdout>>,
    state: Arc<Mutex<ProxyState>>,
) {
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => match rewrite_child_line(&state, &line) {
                    Ok(Some(line)) => {
                        let _ = write_line(&sink, &line);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("[rust-tui-gateway] child stdout rewrite failed: {error}");
                        let _ = write_line(&sink, &line);
                    }
                },
                Err(error) => {
                    eprintln!("[rust-tui-gateway] child stdout read failed: {error}");
                    break;
                }
            }
        }
    });
}

fn pipe_child_stderr(stderr: ChildStderr) {
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(line) => eprintln!("{line}"),
                Err(error) => {
                    eprintln!("[rust-tui-gateway] child stderr read failed: {error}");
                    break;
                }
            }
        }
    });
}

fn forward_request(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    request: &Value,
) -> Result<(), Box<dyn Error>> {
    if request.get("method").and_then(Value::as_str) == Some("prompt.submit")
        && let Some(local_sid) = request
            .get("params")
            .and_then(Value::as_object)
            .and_then(|params| params.get("session_id"))
            .and_then(Value::as_str)
    {
        ensure_child_session(stdin, state, local_sid)?;
        sync_pending_images(stdin, state, local_sid)?;
    }
    let forwarded = rewrite_outbound_request(state, request)?;
    let mut stdin = stdin
        .lock()
        .map_err(|_| "failed to lock child stdin for forwarding")?;
    stdin.write_all(serde_json::to_string(&forwarded)?.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    Ok(())
}

fn write_json(sink: &Arc<Mutex<io::Stdout>>, value: &Value) -> Result<(), Box<dyn Error>> {
    write_line(sink, &serde_json::to_string(value)?)?;
    Ok(())
}

fn write_line(sink: &Arc<Mutex<io::Stdout>>, line: &str) -> io::Result<()> {
    let mut stdout = sink
        .lock()
        .map_err(|_| io::Error::other("stdout lock poisoned"))?;
    stdout.write_all(line.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()
}

fn ok_response(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn rebind_response_id(mut response: Value, id: Value) -> Value {
    if let Some(object) = response.as_object_mut() {
        object.insert("id".to_string(), id);
    }
    response
}

fn rewrite_outbound_request(
    state: &Arc<Mutex<ProxyState>>,
    request: &Value,
) -> Result<Value, Box<dyn Error>> {
    let mut forwarded = request.clone();
    let Some(object) = forwarded.as_object_mut() else {
        return Ok(forwarded);
    };
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let request_id = object.get("id").and_then(Value::as_str).map(str::to_string);
    let mut local_session_id = None;
    let mut resume_target = None;

    if let Some(params) = object.get_mut("params").and_then(Value::as_object_mut)
        && method != "session.resume"
        && method != "session.delete"
        && method != "session.list"
        && method != "session.most_recent"
        && method != "session.create"
        && method != "session.close"
    {
        if let Some(raw_sid) = params
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_string)
        {
            local_session_id = Some(raw_sid.clone());
            if method == "prompt.submit" {
                mark_store_session_dirty(state, &raw_sid)?;
            }
            if let Some(child_sid) = lookup_child_session_id(state, &raw_sid)? {
                params.insert("session_id".to_string(), Value::String(child_sid));
            }
        }
    } else if method == "session.resume" {
        resume_target = object
            .get("params")
            .and_then(Value::as_object)
            .and_then(|params| params.get("session_id"))
            .and_then(Value::as_str)
            .map(str::to_string);
    }

    if let Some(id) = request_id {
        let mut guard = state
            .lock()
            .map_err(|_| "proxy state lock poisoned while tracking request")?;
        guard.pending.insert(
            id,
            PendingRequest {
                local_session_id,
                method,
                resume_target,
                response_tx: None,
                suppress_output: false,
            },
        );
    }

    Ok(forwarded)
}

fn rewrite_child_line(
    state: &Arc<Mutex<ProxyState>>,
    line: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let mut message: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(_) => return Ok(Some(line.to_string())),
    };
    let Some(object) = message.as_object_mut() else {
        return Ok(Some(line.to_string()));
    };

    if object.get("method").and_then(Value::as_str) == Some("event") {
        rewrite_child_event(state, object)?;
    } else if let Some(id) = object.get("id").and_then(Value::as_str).map(str::to_string) {
        if rewrite_child_response(state, &id, object)? {
            return Ok(None);
        }
    }

    Ok(Some(serde_json::to_string(&message)?))
}

fn rewrite_child_event(
    state: &Arc<Mutex<ProxyState>>,
    object: &mut Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let Some(params) = object.get_mut("params").and_then(Value::as_object_mut) else {
        return Ok(());
    };
    let Some(child_sid) = params.get("session_id").and_then(Value::as_str) else {
        return Ok(());
    };
    let local_sid = if let Some(local_sid) = lookup_local_session_id(state, child_sid)? {
        Some(local_sid)
    } else {
        claim_pending_resume_child_session(state, child_sid)?
    };
    if let Some(local_sid) = local_sid.clone() {
        params.insert("session_id".to_string(), Value::String(local_sid));
    }
    if let Some(local_sid) = local_sid.clone() {
        cache_child_event_state(state, &local_sid, params)?;
    }
    strip_internal_session_key(params);
    if let Some(event_type) = params.get("type").and_then(Value::as_str)
        && matches!(
            event_type,
            "clarify.request" | "sudo.request" | "secret.request"
        )
        && let Some(payload) = params.get_mut("payload").and_then(Value::as_object_mut)
        && let Some(child_request_id) = payload
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_string)
        && let Some(local_sid) = local_sid
    {
        let local_request_id = bind_prompt_request(state, &local_sid, &child_request_id)?;
        payload.insert("request_id".to_string(), Value::String(local_request_id));
    }
    Ok(())
}

fn cache_child_event_state(
    state: &Arc<Mutex<ProxyState>>,
    local_session_id: &str,
    params: &Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let event_type = params
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match event_type {
        "session.info" => {
            if let Some(payload) = params.get("payload").and_then(Value::as_object) {
                if let Some(session_key) = payload
                    .get("session_key")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    update_store_session_id(state, local_session_id, session_key)?;
                }
                update_session_info_cache(state, local_session_id, payload.clone())?;
                if let Some(usage) = payload.get("usage").and_then(Value::as_object) {
                    update_session_usage_cache(state, local_session_id, usage.clone())?;
                }
            }
        }
        "message.start" => set_session_running(state, local_session_id, true)?,
        "message.complete" => {
            set_session_running(state, local_session_id, false)?;
            if let Some(payload) = params.get("payload").and_then(Value::as_object) {
                if let Some(session_key) = payload
                    .get("session_key")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    update_store_session_id(state, local_session_id, session_key)?;
                }
                if let Some(usage) = payload.get("usage").and_then(Value::as_object) {
                    update_session_usage_cache(state, local_session_id, usage.clone())?;
                }
            }
        }
        "error" => set_session_running(state, local_session_id, false)?,
        _ => {}
    }
    Ok(())
}

fn strip_internal_session_key(params: &mut Map<String, Value>) {
    let event_type = params
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(event_type, "session.info" | "message.complete") {
        return;
    }
    if let Some(payload) = params.get_mut("payload").and_then(Value::as_object_mut) {
        payload.remove("session_key");
    }
}

fn claim_pending_resume_child_session(
    state: &Arc<Mutex<ProxyState>>,
    child_id: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let candidate = {
        let guard = state
            .lock()
            .map_err(|_| "proxy state lock poisoned while claiming pending resume child")?;
        let mut candidates = guard
            .pending
            .values()
            .filter(|pending| {
                pending.method == "session.resume"
                    && pending.suppress_output
                    && pending.local_session_id.is_some()
            })
            .filter_map(|pending| {
                pending
                    .local_session_id
                    .as_ref()
                    .map(|local_id| (local_id.clone(), pending.resume_target.clone()))
            })
            .collect::<Vec<_>>();
        if candidates.len() == 1 {
            candidates.pop()
        } else {
            None
        }
    };
    let Some((local_id, resume_target)) = candidate else {
        return Ok(None);
    };
    attach_child_to_local(state, &local_id, child_id, resume_target)?;
    Ok(Some(local_id))
}

fn rewrite_child_response(
    state: &Arc<Mutex<ProxyState>>,
    request_id: &str,
    object: &mut Map<String, Value>,
) -> Result<bool, Box<dyn Error>> {
    let pending = {
        let mut guard = state
            .lock()
            .map_err(|_| "proxy state lock poisoned while resolving response")?;
        guard.pending.remove(request_id)
    };
    let Some(pending) = pending else {
        return Ok(false);
    };
    let Some(result) = object.get_mut("result").and_then(Value::as_object_mut) else {
        if let Some(response_tx) = pending.response_tx {
            let _ = response_tx.send(Value::Object(object.clone()));
        }
        match pending.method.as_str() {
            "session.close" => {
                if let Some(local_sid) = pending.local_session_id {
                    release_local_session(state, &local_sid)?;
                }
            }
            "child.close" => {
                if let Some(local_sid) = pending.local_session_id {
                    detach_child_session(state, &local_sid)?;
                }
            }
            _ => {}
        }
        return Ok(pending.suppress_output);
    };

    if let Some(local_sid) = pending.local_session_id.as_deref() {
        if let Some(session_key) = result.get("session_key").and_then(Value::as_str) {
            update_store_session_id(state, local_sid, session_key)?;
        } else if let Some(session_key) = result
            .get("info")
            .and_then(Value::as_object)
            .and_then(|info| info.get("session_key"))
            .and_then(Value::as_str)
        {
            update_store_session_id(state, local_sid, session_key)?;
        }
    }

    match pending.method.as_str() {
        "session.create" | "session.branch" => {
            if let Some(child_sid) = result
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                let local_sid = if let Some(local_sid) = pending.local_session_id {
                    attach_child_to_local(state, &local_sid, &child_sid, None)?;
                    local_sid
                } else {
                    bind_child_session(state, &child_sid, None)?
                };
                result.insert("session_id".to_string(), Value::String(local_sid));
            }
        }
        "session.resume" => {
            if let Some(child_sid) = result
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                let local_sid = if let Some(local_sid) = pending.local_session_id {
                    attach_child_to_local(
                        state,
                        &local_sid,
                        &child_sid,
                        pending.resume_target.clone(),
                    )?;
                    local_sid
                } else {
                    bind_child_session(state, &child_sid, pending.resume_target)?
                };
                result.insert("session_id".to_string(), Value::String(local_sid));
            }
        }
        "session.close" => {
            if let Some(local_sid) = pending.local_session_id {
                release_local_session(state, &local_sid)?;
            }
        }
        "child.close" => {
            if let Some(local_sid) = pending.local_session_id {
                detach_child_session(state, &local_sid)?;
            }
        }
        _ => {}
    }
    if let Some(response_tx) = pending.response_tx {
        let _ = response_tx.send(Value::Object(object.clone()));
    }
    Ok(pending.suppress_output)
}

fn bind_child_session(
    state: &Arc<Mutex<ProxyState>>,
    child_id: &str,
    store_session_id: Option<String>,
) -> Result<String, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while binding session")?;
    if let Some(local_sid) = guard.child_to_local.get(child_id).cloned() {
        if let Some(binding) = guard.sessions.get_mut(&local_sid)
            && store_session_id.is_some()
        {
            binding.store_session_id = store_session_id;
            binding.store_session_dirty = false;
        }
        return Ok(local_sid);
    }
    guard.next_local_session = guard.next_local_session.saturating_add(1);
    let local_sid = format!("rs_tui_{:08x}", guard.next_local_session);
    guard
        .child_to_local
        .insert(child_id.to_string(), local_sid.clone());
    guard.sessions.insert(
        local_sid.clone(),
        SessionBinding {
            attached_images: Vec::new(),
            child_id: child_id.to_string(),
            cols: 80,
            info: Map::new(),
            image_counter: 0,
            running: false,
            store_session_id,
            store_session_dirty: false,
            usage: default_usage_payload(""),
        },
    );
    Ok(local_sid)
}

fn create_local_session(
    state: &Arc<Mutex<ProxyState>>,
    store_session_id: Option<String>,
    cols: u64,
) -> Result<String, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while creating session")?;
    guard.next_local_session = guard.next_local_session.saturating_add(1);
    let local_sid = format!("rs_tui_{:08x}", guard.next_local_session);
    guard.sessions.insert(
        local_sid.clone(),
        SessionBinding {
            attached_images: Vec::new(),
            child_id: String::new(),
            cols,
            info: Map::new(),
            image_counter: 0,
            running: false,
            store_session_id,
            store_session_dirty: false,
            usage: default_usage_payload(""),
        },
    );
    Ok(local_sid)
}

fn attach_child_to_local(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    child_id: &str,
    store_session_id: Option<String>,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while attaching child session")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        let previous_child_id = binding.child_id.clone();
        binding.child_id = child_id.to_string();
        if let Some(store_session_id) = store_session_id {
            binding.store_session_id = Some(store_session_id);
            binding.store_session_dirty = false;
        }
        if !previous_child_id.is_empty() {
            guard.child_to_local.remove(&previous_child_id);
        }
        guard
            .child_to_local
            .insert(child_id.to_string(), local_id.to_string());
    }
    Ok(())
}

fn detach_child_session(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while detaching child session")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        let previous_child_id = std::mem::take(&mut binding.child_id);
        if !previous_child_id.is_empty() {
            guard.child_to_local.remove(&previous_child_id);
        }
    }
    Ok(())
}

fn release_local_session(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while releasing session")?;
    guard
        .prompt_requests
        .retain(|_, binding| binding.local_session_id != local_id);
    if let Some(binding) = guard.sessions.remove(local_id) {
        if !binding.child_id.is_empty() {
            guard.child_to_local.remove(&binding.child_id);
        }
    }
    Ok(())
}

fn lookup_child_session_id(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while looking up child session")?;
    Ok(guard
        .sessions
        .get(local_id)
        .and_then(|binding| (!binding.child_id.is_empty()).then(|| binding.child_id.clone())))
}

fn lookup_local_session_id(
    state: &Arc<Mutex<ProxyState>>,
    child_id: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while looking up local session")?;
    Ok(guard.child_to_local.get(child_id).cloned())
}

fn lookup_store_session_id(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while looking up store session")?;
    Ok(guard
        .sessions
        .get(local_id)
        .and_then(|binding| binding.store_session_id.clone()))
}

fn lookup_clean_store_session_id(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while looking up clean store session")?;
    Ok(guard.sessions.get(local_id).and_then(|binding| {
        if binding.store_session_dirty {
            None
        } else {
            binding.store_session_id.clone()
        }
    }))
}

fn update_store_session_id(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    store_session_id: &str,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating store session")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.store_session_id = Some(store_session_id.to_string());
        binding.store_session_dirty = false;
    }
    Ok(())
}

fn mark_store_session_dirty(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while marking store session dirty")?;
    if let Some(binding) = guard.sessions.get_mut(local_id)
        && binding.store_session_id.is_some()
    {
        binding.store_session_dirty = true;
    }
    Ok(())
}

fn session_exists(state: &Arc<Mutex<ProxyState>>, local_id: &str) -> Result<bool, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while checking session existence")?;
    Ok(guard.sessions.contains_key(local_id))
}

fn get_session_binding(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<Option<SessionBinding>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while cloning session binding")?;
    Ok(guard.sessions.get(local_id).cloned())
}

fn bind_prompt_request(
    state: &Arc<Mutex<ProxyState>>,
    local_session_id: &str,
    child_request_id: &str,
) -> Result<String, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while binding prompt request")?;
    guard.next_prompt_request = guard.next_prompt_request.saturating_add(1);
    let local_request_id = format!("rs_prompt_{:08x}", guard.next_prompt_request);
    guard.prompt_requests.insert(
        local_request_id.clone(),
        PromptRequestBinding {
            child_request_id: child_request_id.to_string(),
            local_session_id: local_session_id.to_string(),
        },
    );
    Ok(local_request_id)
}

fn take_prompt_request(
    state: &Arc<Mutex<ProxyState>>,
    local_request_id: &str,
) -> Result<Option<PromptRequestBinding>, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while taking prompt request")?;
    Ok(guard.prompt_requests.remove(local_request_id))
}

fn update_session_info_cache(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    info: Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating session info cache")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.info = info;
    }
    Ok(())
}

fn update_session_usage_cache(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    usage: Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating session usage cache")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.usage = usage;
    }
    Ok(())
}

fn set_session_running(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    running: bool,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating running state")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.running = running;
    }
    Ok(())
}

fn default_usage_payload(model: &str) -> Map<String, Value> {
    let mut usage = Map::new();
    usage.insert("model".to_string(), json!(model));
    usage.insert("calls".to_string(), json!(0));
    usage.insert("input".to_string(), json!(0));
    usage.insert("output".to_string(), json!(0));
    usage.insert("total".to_string(), json!(0));
    usage.insert("cache_read".to_string(), json!(0));
    usage.insert("cache_write".to_string(), json!(0));
    usage
}

fn usage_payload(binding: &SessionBinding) -> Map<String, Value> {
    let model = binding
        .info
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if binding.usage.is_empty() {
        default_usage_payload(&model)
    } else {
        let mut usage = binding.usage.clone();
        usage
            .entry("model".to_string())
            .or_insert_with(|| json!(model));
        usage
    }
}

fn current_timestamp_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn new_background_task_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("bg_{:06x}", (now.as_nanos() & 0x00ff_ffff) as u64)
}

fn current_filename_timestamp() -> String {
    let timestamp = current_timestamp_seconds();
    format_local_timestamp(timestamp, "%Y%m%d_%H%M%S")
        .unwrap_or_else(|| format!("{timestamp:.0}"))
}

fn format_timestamp(timestamp: f64) -> String {
    format_local_timestamp(timestamp, "%Y-%m-%d %H:%M").unwrap_or_else(|| format!("{timestamp:.0}"))
}

/// Format a Unix timestamp (seconds) in the local timezone, matching Python's
/// `datetime.fromtimestamp(ts).strftime(fmt)`.
fn format_local_timestamp(timestamp: f64, fmt: &str) -> Option<String> {
    let secs = timestamp.floor() as i64;
    let nanos = ((timestamp - timestamp.floor()) * 1_000_000_000.0).round() as u32;
    let utc = DateTime::<Utc>::from_timestamp(secs, nanos)?;
    Some(utc.with_timezone(&Local).format(fmt).to_string())
}

fn set_session_cols(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    cols: u64,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating cols")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.cols = cols;
    }
    Ok(())
}

fn bump_session_image_counter(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<u64, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while updating image counter")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.image_counter = binding.image_counter.saturating_add(1);
        return Ok(binding.image_counter);
    }
    Err("session not found".into())
}

fn queue_attached_image(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while queueing image")?;
    if let Some(binding) = guard.sessions.get_mut(local_id) {
        binding.attached_images.push(path.to_string());
    }
    Ok(())
}

fn queued_image_count(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<usize, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while reading queued images")?;
    Ok(guard
        .sessions
        .get(local_id)
        .map(|binding| binding.attached_images.len())
        .unwrap_or(0))
}

fn take_queued_images(
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while draining queued images")?;
    Ok(guard
        .sessions
        .get_mut(local_id)
        .map(|binding| std::mem::take(&mut binding.attached_images))
        .unwrap_or_default())
}

fn sync_pending_images(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(child_id) = lookup_child_session_id(state, local_id)? else {
        return Ok(());
    };
    let queued = take_queued_images(state, local_id)?;
    for path in queued {
        send_internal_child_request(
            stdin,
            state,
            "image.attach",
            json!({"session_id": child_id, "path": path}),
            None,
            None,
            None,
        )?;
    }
    Ok(())
}

fn ensure_child_session(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    local_id: &str,
) -> Result<(), Box<dyn Error>> {
    if lookup_child_session_id(state, local_id)?.is_some() {
        return Ok(());
    }
    let Some(binding) = get_session_binding(state, local_id)? else {
        return Err("session not found".into());
    };
    let Some(store_session_id) = binding.store_session_id else {
        return Err("store session missing for lazy child materialization".into());
    };
    send_internal_child_request(
        stdin,
        state,
        "session.resume",
        json!({"session_id": store_session_id, "cols": binding.cols}),
        Some(local_id.to_string()),
        Some(store_session_id),
        None,
    )?;
    for _ in 0..300 {
        if lookup_child_session_id(state, local_id)?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    Err("timed out waiting for child session materialization".into())
}

fn send_internal_child_request(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    method: &str,
    params: Value,
    local_session_id: Option<String>,
    resume_target: Option<String>,
    tracked_method: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let request_id = {
        let mut guard = state
            .lock()
            .map_err(|_| "proxy state lock poisoned while sending internal request")?;
        guard.next_internal_request = guard.next_internal_request.saturating_add(1);
        let request_id = format!("rs_internal_{:08x}", guard.next_internal_request);
        guard.pending.insert(
            request_id.clone(),
            PendingRequest {
                local_session_id,
                method: tracked_method.unwrap_or(method).to_string(),
                resume_target,
                response_tx: None,
                suppress_output: true,
            },
        );
        request_id
    };
    let request = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    });
    let mut stdin = stdin
        .lock()
        .map_err(|_| "failed to lock child stdin for internal request")?;
    stdin.write_all(serde_json::to_string(&request)?.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    Ok(())
}

fn send_blocking_child_request(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    method: &str,
    params: Value,
    local_session_id: Option<String>,
) -> Result<Value, Box<dyn Error>> {
    send_blocking_child_request_timeout(
        stdin,
        state,
        method,
        params,
        local_session_id,
        Duration::from_secs(5),
    )
}

fn send_blocking_child_request_timeout(
    stdin: &Arc<Mutex<ChildStdin>>,
    state: &Arc<Mutex<ProxyState>>,
    method: &str,
    params: Value,
    local_session_id: Option<String>,
    timeout: Duration,
) -> Result<Value, Box<dyn Error>> {
    let (response_tx, response_rx) = mpsc::channel();
    let request_id = {
        let mut guard = state
            .lock()
            .map_err(|_| "proxy state lock poisoned while sending blocking request")?;
        guard.next_internal_request = guard.next_internal_request.saturating_add(1);
        let request_id = format!("rs_internal_{:08x}", guard.next_internal_request);
        guard.pending.insert(
            request_id.clone(),
            PendingRequest {
                local_session_id,
                method: method.to_string(),
                resume_target: None,
                response_tx: Some(response_tx),
                suppress_output: true,
            },
        );
        request_id
    };
    let request = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    });
    {
        let mut stdin = stdin
            .lock()
            .map_err(|_| "failed to lock child stdin for blocking request")?;
        stdin.write_all(serde_json::to_string(&request)?.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
    }

    match response_rx.recv_timeout(timeout) {
        Ok(response) => Ok(response),
        Err(_) => {
            let mut guard = state
                .lock()
                .map_err(|_| "proxy state lock poisoned while clearing timed out request")?;
            guard
                .pending
                .remove(request["id"].as_str().unwrap_or_default());
            Err(format!("timed out waiting for child response to {method}").into())
        }
    }
}

fn resume_messages(store: &SessionStore, session_id: &str) -> Result<Vec<Value>, Box<dyn Error>> {
    Ok(history_messages(store, session_id)?.1)
}

fn history_messages(
    store: &SessionStore,
    session_id: &str,
) -> Result<(usize, Vec<Value>), Box<dyn Error>> {
    let lineage = session_lineage(store, session_id)?;
    let mut tool_call_args: HashMap<String, String> = HashMap::new();
    let mut raw_count = 0_usize;
    let mut messages = Vec::new();
    for lineage_id in lineage {
        for message in store.get_messages(&lineage_id)? {
            raw_count += 1;
            let role = message.role.trim();
            if !matches!(role, "user" | "assistant" | "tool" | "system") {
                continue;
            }
            if role == "assistant" {
                if let Some(tool_calls) = message.tool_calls.as_ref().and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        let Some(tc_id) = tool_call.get("id").and_then(Value::as_str) else {
                            continue;
                        };
                        let name = tool_call
                            .get("function")
                            .and_then(Value::as_object)
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        tool_call_args.insert(tc_id.to_string(), name.to_string());
                    }
                }
                let content = content_text(message.content.as_ref());
                if content.trim().is_empty() {
                    continue;
                }
                messages.push(json!({"role": "assistant", "text": content}));
                continue;
            }
            if role == "tool" {
                let name = message
                    .tool_call_id
                    .as_deref()
                    .and_then(|tool_call_id| tool_call_args.get(tool_call_id))
                    .cloned()
                    .or_else(|| message.tool_name.clone())
                    .unwrap_or_else(|| "tool".to_string());
                messages.push(json!({"role": "tool", "name": name, "context": ""}));
                continue;
            }
            let content = content_text(message.content.as_ref());
            if content.trim().is_empty() {
                continue;
            }
            messages.push(json!({"role": role, "text": content}));
        }
    }
    Ok((raw_count, messages))
}

fn session_lineage(store: &SessionStore, session_id: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut lineage = Vec::new();
    let mut current = Some(session_id.to_string());
    let mut seen = HashSet::new();
    while let Some(session_id) = current {
        if !seen.insert(session_id.clone()) {
            break;
        }
        let Some(record) = store.get_session(&session_id)? else {
            break;
        };
        lineage.push(record.id.clone());
        current = record.parent_session_id.clone();
    }
    lineage.reverse();
    Ok(lineage)
}

fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn run_background_prompt(
    context: &HermesContext,
    work_root: &Path,
    prompt: &str,
    model: Option<String>,
) -> Result<String, Box<dyn Error>> {
    let config = context.load_config_document()?;
    let session_store = context.open_session_store()?;
    let overrides = ModelOverrides {
        model,
        provider: None,
        base_url: None,
        api_key: None,
        api_mode: None,
    };
    let enabled_toolsets = config.config.toolsets.clone();
    let delegate = DelegateExecutor::new(
        context.clone(),
        config.clone(),
        "rust-background-delegate",
        enabled_toolsets.clone(),
        overrides.clone(),
        work_root.to_path_buf(),
    );
    let runtime = ToolRuntime::new(work_root.to_path_buf())
        .with_hermes_home(context.hermes_home())
        .with_clarify_callback(|question, _choices| {
            Err(format!(
                "Clarify is unavailable in background tasks: {question}"
            ))
        })
        .with_delegate_callback(move |request, parent_runtime| {
            delegate.execute(request, parent_runtime)
        });
    let result = context.run_chat_completions_turn(
        &config,
        prompt,
        &runtime,
        Some(&enabled_toolsets),
        &overrides,
        None,
        Some(&session_store),
    )?;
    Ok(result.final_response)
}

fn message_record_to_chat_message(message: MessageRecord) -> Result<Value, HermesError> {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String(message.role));
    map.insert(
        "content".to_string(),
        message.content.unwrap_or(Value::Null),
    );
    if let Some(tool_call_id) = message.tool_call_id {
        map.insert("tool_call_id".to_string(), Value::String(tool_call_id));
    }
    if let Some(tool_calls) = message.tool_calls {
        if !tool_calls.is_array() {
            return Err(HermesError::State {
                action: "saving session transcript",
                detail: "Stored tool_calls payload was not an array.".to_string(),
            });
        }
        map.insert("tool_calls".to_string(), tool_calls);
    }
    if let Some(reasoning_details) = message.reasoning_details {
        map.insert("reasoning_details".to_string(), reasoning_details);
    }
    if let Some(codex_reasoning_items) = message.codex_reasoning_items {
        map.insert("codex_reasoning_items".to_string(), codex_reasoning_items);
    }
    if let Some(codex_message_items) = message.codex_message_items {
        map.insert("codex_message_items".to_string(), codex_message_items);
    }
    Ok(Value::Object(map))
}

fn message_record_to_append(message: &MessageRecord) -> MessageAppend {
    MessageAppend {
        role: message.role.clone(),
        content: message.content.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message.tool_calls.clone(),
        tool_name: message.tool_name.clone(),
        token_count: message.token_count,
        finish_reason: message.finish_reason.clone(),
        reasoning: message.reasoning.clone(),
        reasoning_content: message.reasoning_content.clone(),
        reasoning_details: message.reasoning_details.clone(),
        codex_reasoning_items: message.codex_reasoning_items.clone(),
        codex_message_items: message.codex_message_items.clone(),
    }
}

fn trim_last_exchange(mut records: Vec<MessageRecord>) -> (Vec<MessageRecord>, i64) {
    let mut removed = 0_i64;
    while records
        .last()
        .is_some_and(|message| matches!(message.role.trim(), "assistant" | "tool"))
    {
        records.pop();
        removed += 1;
    }
    if records
        .last()
        .is_some_and(|message| message.role.trim() == "user")
    {
        records.pop();
        removed += 1;
    }
    (records, removed)
}

fn active_store_session_ids(
    state: &Arc<Mutex<ProxyState>>,
) -> Result<HashSet<String>, Box<dyn Error>> {
    let guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while reading active sessions")?;
    Ok(guard
        .sessions
        .values()
        .filter_map(|binding| binding.store_session_id.clone())
        .collect())
}

fn next_paste_counter(state: &Arc<Mutex<ProxyState>>) -> Result<u64, Box<dyn Error>> {
    let mut guard = state
        .lock()
        .map_err(|_| "proxy state lock poisoned while incrementing paste counter")?;
    guard.next_paste = guard.next_paste.saturating_add(1);
    Ok(guard.next_paste)
}

fn run_python_helper_json(
    helper: &HelperContext,
    script: &str,
    payload: Option<&Value>,
) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(&helper.python);
    command
        .arg("-c")
        .arg(script)
        .current_dir(&helper.work_root)
        .env("HERMES_HOME", &helper.hermes_home)
        .env("HERMES_PYTHON_SRC_ROOT", &helper.project_root)
        .env("PYTHONPATH", &helper.python_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;
    if let Some(input) = payload
        && let Some(stdin) = child.stdin.as_mut()
    {
        stdin.write_all(serde_json::to_string(input)?.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "python helper failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        )
        .into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    Ok(serde_json::from_str(stdout.trim())?)
}

fn image_meta(_helper: &HelperContext, path: &Path) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert(
        "name".to_string(),
        json!(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        ),
    );
    if let Ok((width, height)) = image::image_dimensions(path) {
        let token_estimate = u64::from((width + 511) / 512).max(1)
            * u64::from((height + 511) / 512).max(1)
            * 85;
        payload.insert("width".to_string(), json!(width));
        payload.insert("height".to_string(), json!(height));
        payload.insert("token_estimate".to_string(), json!(token_estimate));
    }
    payload
}

fn split_path_input(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (String::new(), String::new());
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    if matches!(chars.first(), Some('"') | Some('\'')) {
        let quote = chars[0];
        let mut pos = 1usize;
        while pos < chars.len() {
            let ch = chars[pos];
            if ch == '\\' && pos + 1 < chars.len() {
                pos += 2;
                continue;
            }
            if ch == quote {
                let token = chars[1..pos].iter().collect::<String>();
                let remainder = chars[pos + 1..]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .to_string();
                return (token, remainder);
            }
            pos += 1;
        }
        return (chars[1..].iter().collect::<String>(), String::new());
    }

    let mut pos = 0usize;
    while pos < chars.len() {
        let ch = chars[pos];
        if ch == '\\' && pos + 1 < chars.len() && chars[pos + 1] == ' ' {
            pos += 2;
        } else if ch == ' ' {
            break;
        } else {
            pos += 1;
        }
    }
    let token = chars[..pos].iter().collect::<String>().replace("\\ ", " ");
    let remainder = chars[pos..].iter().collect::<String>().trim().to_string();
    (token, remainder)
}

fn resolve_attachment_path(raw_path: &str, work_root: &Path) -> Option<PathBuf> {
    let mut token = raw_path.trim().to_string();
    if token.is_empty() {
        return None;
    }
    if (token.starts_with('"') && token.ends_with('"'))
        || (token.starts_with('\'') && token.ends_with('\''))
    {
        token = token[1..token.len().saturating_sub(1)].trim().to_string();
    }
    token = token.replace("\\ ", " ");
    if token.is_empty() {
        return None;
    }

    let mut candidate = if token.starts_with("file://") {
        Url::parse(&token).ok()?.to_file_path().ok()?
    } else {
        PathBuf::from(expand_home_path(&token))
    };
    if !candidate.is_absolute() {
        candidate = work_root.join(candidate);
    }
    let resolved = candidate.canonicalize().unwrap_or(candidate);
    resolved.is_file().then_some(resolved)
}

fn expand_home_path(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .display()
            .to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join(rest)
            .display()
            .to_string();
    }
    path.to_string()
}

fn starts_like_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with('~')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("file://")
        || value.starts_with("\"/")
        || value.starts_with("\"~")
        || value.starts_with("'/")
        || value.starts_with("'~")
        || value.starts_with("\"./")
        || value.starts_with("\"../")
        || value.starts_with("'./")
        || value.starts_with("'../")
        || value
            .chars()
            .nth(1)
            .zip(value.chars().nth(2))
            .map(|(a, b)| {
                a == ':'
                    && matches!(b, '\\' | '/')
                    && value
                        .chars()
                        .next()
                        .is_some_and(|first| first.is_ascii_alphabetic())
            })
            .unwrap_or(false)
        || (value.len() >= 4
            && matches!(value.chars().next(), Some('\'') | Some('"'))
            && value.chars().nth(2) == Some(':')
            && matches!(value.chars().nth(3), Some('\\') | Some('/'))
            && value
                .chars()
                .nth(1)
                .is_some_and(|first| first.is_ascii_alphabetic()))
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tiff" | "tif" | "svg" | "ico"
    )
}

fn resolve_image_attachment(raw: &str, work_root: &Path) -> Result<(PathBuf, String), String> {
    let dropped = detect_file_drop(raw, work_root);
    let (path, remainder) = if let Some(drop) = dropped {
        (drop.path, drop.remainder)
    } else {
        let (path_token, remainder) = split_path_input(raw);
        let Some(path) = resolve_attachment_path(&path_token, work_root) else {
            return Err(format!("image not found: {path_token}"));
        };
        (path, remainder)
    };
    if !is_supported_image(&path) {
        return Err(format!(
            "unsupported image: {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("image")
        ));
    }
    Ok((path, remainder))
}

fn detect_file_drop(user_input: &str, work_root: &Path) -> Option<DetectedDrop> {
    let stripped = user_input.trim();
    if stripped.is_empty() || !starts_like_path(stripped) {
        return None;
    }
    if let Some(path) = resolve_attachment_path(stripped, work_root) {
        return Some(DetectedDrop {
            is_image: is_supported_image(&path),
            path,
            remainder: String::new(),
        });
    }
    let (first_token, mut remainder) = split_path_input(stripped);
    let mut drop_path = resolve_attachment_path(&first_token, work_root);
    if drop_path.is_none()
        && stripped.contains(' ')
        && !matches!(stripped.chars().next(), Some('\'') | Some('"'))
    {
        for (idx, ch) in stripped.char_indices().rev() {
            if ch != ' ' {
                continue;
            }
            let candidate = stripped[..idx].trim_end();
            if let Some(resolved) = resolve_attachment_path(candidate, work_root) {
                drop_path = Some(resolved);
                remainder = stripped[idx + 1..].trim().to_string();
                break;
            }
        }
    }
    let path = drop_path?;
    Some(DetectedDrop {
        is_image: is_supported_image(&path),
        path,
        remainder,
    })
}

fn complete_path_response(word: &str, work_root: &Path) -> Value {
    if word.trim().is_empty() {
        return json!({"items": []});
    }

    let is_context = word.starts_with('@');
    let query = if is_context { &word[1..] } else { word };
    if is_context && query.is_empty() {
        return json!({
            "items": [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
        });
    }

    let (prefix_tag, path_part) = if is_context && matches!(query, "file" | "folder") {
        (query, "")
    } else if is_context && (query.starts_with("file:") || query.starts_with("folder:")) {
        let mut parts = query.splitn(2, ':');
        (
            parts.next().unwrap_or_default(),
            parts.next().unwrap_or_default(),
        )
    } else {
        ("", query)
    };

    if is_context && !path_part.is_empty() && !path_part.contains('/') && prefix_tag != "folder" {
        let mut ranked = Vec::new();
        for rel in list_repo_files(work_root) {
            let basename = Path::new(&rel)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            if basename.starts_with('.') && !path_part.starts_with('.') {
                continue;
            }
            if let Some(rank) = fuzzy_basename_rank(&basename, path_part) {
                ranked.push((rank, rel, basename));
            }
        }
        ranked.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.len().cmp(&right.1.len()))
                .then_with(|| left.1.cmp(&right.1))
        });

        let tag = if prefix_tag.is_empty() {
            "file"
        } else {
            prefix_tag
        };
        let items = ranked
            .into_iter()
            .take(MAX_COMPLETION_ITEMS)
            .map(|(_, rel, basename)| {
                json!({
                    "text": format!("@{tag}:{rel}"),
                    "display": basename,
                    "meta": Path::new(&rel).parent().and_then(Path::to_str).unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        return json!({"items": items});
    }

    let expanded = if path_part.is_empty() {
        ".".to_string()
    } else {
        normalize_completion_path(path_part)
    };
    let (search_dir_raw, matcher) = if expanded == "." || expanded.is_empty() {
        (".".to_string(), String::new())
    } else if expanded.ends_with('/') {
        (expanded.clone(), String::new())
    } else {
        (
            Path::new(&expanded)
                .parent()
                .and_then(Path::to_str)
                .unwrap_or(".")
                .to_string(),
            Path::new(&expanded)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string(),
        )
    };
    let search_dir = resolve_completion_path(work_root, &search_dir_raw);
    if !search_dir.is_dir() {
        return json!({"items": []});
    }

    let want_dir = prefix_tag == "folder";
    let matcher_lower = matcher.to_ascii_lowercase();
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir(&search_dir) {
        let mut names = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                Some((name, entry.path()))
            })
            .collect::<Vec<_>>();
        names.sort_by(|left, right| left.0.cmp(&right.0));

        for (name, full) in names {
            if !matcher.is_empty() && !name.to_ascii_lowercase().starts_with(&matcher_lower) {
                continue;
            }
            if is_context && prefix_tag.is_empty() && name.starts_with('.') {
                continue;
            }
            let is_dir = full.is_dir();
            if !prefix_tag.is_empty() && want_dir != is_dir {
                continue;
            }
            let rel = full
                .strip_prefix(work_root)
                .unwrap_or(&full)
                .to_string_lossy()
                .replace('\\', "/");
            let suffix = if is_dir { "/" } else { "" };

            let text = if is_context && !prefix_tag.is_empty() {
                format!("@{prefix_tag}:{rel}{suffix}")
            } else if is_context {
                format!("@{}:{rel}{suffix}", if is_dir { "folder" } else { "file" })
            } else if word.starts_with("~/") {
                let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
                let pretty = full
                    .strip_prefix(&home)
                    .unwrap_or(&full)
                    .to_string_lossy()
                    .replace('\\', "/");
                format!("~/{pretty}{suffix}")
            } else if word.starts_with("./") {
                format!("./{rel}{suffix}")
            } else {
                format!("{rel}{suffix}")
            };

            items.push(json!({
                "text": text,
                "display": format!("{name}{suffix}"),
                "meta": if is_dir { "dir" } else { "" },
            }));
            if items.len() >= MAX_COMPLETION_ITEMS {
                break;
            }
        }
    }

    json!({"items": items})
}

fn normalize_completion_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed == "~" {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .to_string_lossy()
            .to_string()
    } else if let Some(rest) = trimmed.strip_prefix("~/") {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join(rest)
            .to_string_lossy()
            .to_string()
    } else {
        trimmed.to_string()
    }
}

fn resolve_completion_path(work_root: &Path, raw: &str) -> PathBuf {
    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        candidate
    } else {
        work_root.join(candidate)
    }
}

fn list_repo_files(root: &Path) -> Vec<String> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file()
                && let Ok(rel) = path.strip_prefix(root)
            {
                files.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    files
}

fn fuzzy_basename_rank(name: &str, query: &str) -> Option<(u8, usize)> {
    if query.is_empty() {
        return None;
    }
    let nl = name.to_ascii_lowercase();
    let ql = query.to_ascii_lowercase();
    if nl == ql {
        return Some((0, name.len()));
    }
    if nl.starts_with(&ql) {
        return Some((1, name.len()));
    }

    let mut parts = Vec::new();
    let mut buffer = String::new();
    for ch in name.chars() {
        if matches!(ch, '-' | '_' | '.')
            || (ch.is_uppercase() && !buffer.is_empty() && !buffer.ends_with(char::is_uppercase))
        {
            if !buffer.is_empty() {
                parts.push(buffer.clone());
            }
            buffer.clear();
            if !matches!(ch, '-' | '_' | '.') {
                buffer.push(ch);
            }
        } else {
            buffer.push(ch);
        }
    }
    if !buffer.is_empty() {
        parts.push(buffer);
    }
    if parts
        .iter()
        .any(|part| part.to_ascii_lowercase().starts_with(&ql))
    {
        return Some((2, name.len()));
    }
    if nl.contains(&ql) {
        return Some((3, name.len()));
    }

    let mut query_chars = ql.chars();
    let mut current = query_chars.next()?;
    for ch in nl.chars() {
        if ch == current {
            if let Some(next) = query_chars.next() {
                current = next;
            } else {
                return Some((4, name.len()));
            }
        }
    }
    None
}

fn details_completions(text: &str) -> Option<Value> {
    if !text.to_ascii_lowercase().starts_with("/details") {
        return None;
    }
    let stripped = text.trim();
    if !stripped.is_empty() {
        let head = stripped
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !"/details".starts_with(&head) {
            return None;
        }
    }

    let mut body = text["/details".len()..].to_string();
    if body.starts_with(' ') {
        body.remove(0);
    }
    let parts = body
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let has_trailing_space = text.ends_with(' ');
    let sections = ["thinking", "tools", "subagents", "activity"];
    let modes = ["hidden", "collapsed", "expanded"];

    let item = |value: &str, meta: &str| json!({"text": value, "display": value, "meta": meta});
    let root_item = |value: &str, meta: &str, needs_leading_space: bool| {
        let text = if needs_leading_space {
            format!(" {value}")
        } else {
            value.to_string()
        };
        item(&text, meta)
    };

    let items = if body.is_empty() || (parts.is_empty() && has_trailing_space) {
        let mut results = modes
            .iter()
            .map(|mode| root_item(mode, "global mode", !has_trailing_space))
            .collect::<Vec<_>>();
        results.push(root_item("cycle", "cycle global mode", !has_trailing_space));
        results.extend(
            sections
                .iter()
                .map(|section| root_item(section, "section override", !has_trailing_space)),
        );
        results
    } else if parts.len() == 1 && !has_trailing_space {
        let prefix = parts[0].to_ascii_lowercase();
        [
            "hidden",
            "collapsed",
            "expanded",
            "cycle",
            "thinking",
            "tools",
            "subagents",
            "activity",
        ]
        .into_iter()
        .filter(|candidate| candidate.starts_with(&prefix) && *candidate != prefix)
        .map(|candidate| {
            item(
                candidate,
                if sections.contains(&candidate) {
                    "section override"
                } else if candidate == "cycle" {
                    "cycle global mode"
                } else {
                    "global mode"
                },
            )
        })
        .collect::<Vec<_>>()
    } else if parts.len() == 1 && has_trailing_space && sections.contains(&parts[0].as_str()) {
        let section = parts[0].to_ascii_lowercase();
        let mut results = modes
            .iter()
            .map(|mode| item(mode, &format!("set {section}")))
            .collect::<Vec<_>>();
        results.push(item("reset", &format!("clear {section} override")));
        results
    } else if parts.len() == 2 && !has_trailing_space && sections.contains(&parts[0].as_str()) {
        let section = parts[0].to_ascii_lowercase();
        let prefix = parts[1].to_ascii_lowercase();
        ["hidden", "collapsed", "expanded", "reset"]
            .into_iter()
            .filter(|candidate| candidate.starts_with(&prefix) && *candidate != prefix)
            .map(|candidate| {
                let meta = if candidate == "reset" {
                    format!("clear {section} override")
                } else {
                    format!("set {section}")
                };
                item(candidate, &meta)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    Some(json!({
        "items": items,
        "replace_from": if text.contains(' ') {
            text.rfind(' ').unwrap_or(0) + 1
        } else {
            text.len()
        }
    }))
}

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

fn resolve_repo_python(project_root: &Path) -> Option<PathBuf> {
    if let Some(value) = std::env::var("HERMES_PYTHON")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Some(PathBuf::from(value));
    }

    let candidates: [PathBuf; 3] = [
        project_root.join(".venv").join(python_bin_name()),
        project_root.join("venv").join(python_bin_name()),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
            .join("hermes-agent")
            .join("venv")
            .join(python_bin_name()),
    ];
    candidates
        .into_iter()
        .find(|candidate| candidate.exists())
        .or_else(|| which_on_path("python3"))
        .or_else(|| which_on_path("python"))
}

fn compose_python_path(project_root: &Path) -> String {
    let current = std::env::var("PYTHONPATH").unwrap_or_default();
    if current.trim().is_empty() {
        project_root.display().to_string()
    } else {
        let mut paths = vec![project_root.to_path_buf()];
        paths.extend(std::env::split_paths(&current));
        std::env::join_paths(paths)
            .ok()
            .and_then(|value| value.into_string().ok())
            .unwrap_or_else(|| format!("{}:{}", project_root.display(), current))
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, ChildStdout, Command, Stdio};
    use std::sync::{Mutex as StdMutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use hermes_core::{MessageAppend, SessionCreate};

    fn test_state() -> Arc<Mutex<ProxyState>> {
        Arc::new(Mutex::new(ProxyState::default()))
    }

    fn cwd_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    #[test]
    fn outbound_request_maps_local_session_id_to_child_session_id() {
        let state = test_state();
        {
            let mut guard = state.lock().unwrap();
            guard
                .child_to_local
                .insert("child-1".to_string(), "local-1".to_string());
            guard.sessions.insert(
                "local-1".to_string(),
                SessionBinding {
                    attached_images: Vec::new(),
                    child_id: "child-1".to_string(),
                    cols: 80,
                    info: Map::new(),
                    image_counter: 0,
                    running: false,
                    store_session_id: None,
                    store_session_dirty: false,
                    usage: default_usage_payload(""),
                },
            );
        }

        let rewritten = rewrite_outbound_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": "r1",
                "method": "prompt.submit",
                "params": {"session_id": "local-1", "text": "hi"},
            }),
        )
        .unwrap();

        assert_eq!(rewritten["params"]["session_id"], json!("child-1"));
        let guard = state.lock().unwrap();
        assert_eq!(
            guard.pending["r1"].local_session_id.as_deref(),
            Some("local-1")
        );
        assert_eq!(guard.pending["r1"].method, "prompt.submit");
    }

    #[test]
    fn session_create_response_binds_local_runtime_session_id() {
        let state = test_state();
        let _ = rewrite_outbound_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": "r-create",
                "method": "session.create",
                "params": {"cols": 80},
            }),
        )
        .unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","id":"r-create","result":{"session_id":"child-create","info":{"lazy":true}}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        let local_id = parsed["result"]["session_id"].as_str().unwrap();

        assert_eq!(local_id, "rs_tui_00000001");
        let guard = state.lock().unwrap();
        assert_eq!(guard.child_to_local["child-create"], "rs_tui_00000001");
        assert_eq!(guard.sessions["rs_tui_00000001"].child_id, "child-create");
    }

    #[test]
    fn child_events_rewrite_child_session_id_to_local_session_id() {
        let state = test_state();
        bind_child_session(&state, "child-2", Some("store-2".to_string())).unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"message.start","session_id":"child-2"}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();

        assert_eq!(parsed["params"]["session_id"], json!("rs_tui_00000001"));
    }

    #[test]
    fn session_resume_response_binds_local_runtime_session_id() {
        let state = test_state();
        let _ = rewrite_outbound_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": "r-resume",
                "method": "session.resume",
                "params": {"session_id": "stored-session", "cols": 80},
            }),
        )
        .unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","id":"r-resume","result":{"session_id":"child-resume","resumed":"stored-session","messages":[]}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();

        assert_eq!(parsed["result"]["session_id"], json!("rs_tui_00000001"));
        assert_eq!(parsed["result"]["resumed"], json!("stored-session"));
        let guard = state.lock().unwrap();
        assert_eq!(
            guard.sessions["rs_tui_00000001"]
                .store_session_id
                .as_deref(),
            Some("stored-session")
        );
    }

    #[test]
    fn early_child_event_during_internal_resume_claims_local_session_binding() {
        let state = test_state();
        let local_id = create_local_session(&state, Some("stored-early".to_string()), 80).unwrap();
        {
            let mut guard = state.lock().unwrap();
            guard.pending.insert(
                "rs_internal_resume".to_string(),
                PendingRequest {
                    local_session_id: Some(local_id.clone()),
                    method: "session.resume".to_string(),
                    resume_target: Some("stored-early".to_string()),
                    response_tx: None,
                    suppress_output: true,
                },
            );
        }

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"session.info","session_id":"child-early","payload":{"model":"demo"}}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();

        assert_eq!(parsed["params"]["session_id"], json!(local_id));
        let guard = state.lock().unwrap();
        assert_eq!(guard.child_to_local["child-early"], local_id);
        assert_eq!(guard.sessions[&local_id].child_id, "child-early");
    }

    #[test]
    fn session_create_native_precreates_store_backed_local_session() {
        let state = test_state();
        let store = test_store();
        let response = handle_session_create(
            json!("r-create-native"),
            &json!({"cols": 84}).as_object().unwrap(),
            &store,
            &state,
            &helper_context(),
        )
        .unwrap()
        .unwrap();

        let session_id = response["result"]["session_id"].as_str().unwrap();
        let store_id = lookup_store_session_id(&state, session_id)
            .unwrap()
            .expect("store session id");
        assert!(store.get_session(&store_id).unwrap().is_some());
        assert_eq!(response["result"]["info"]["lazy"], json!(true));
    }

    #[test]
    fn session_resume_native_returns_stored_messages() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-history".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-test".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "stored-history",
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(json!("hello")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store
            .append_message(
                "stored-history",
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(json!("hi")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        let response = handle_session_resume(
            json!("r-resume-native"),
            &json!({"session_id": "stored-history", "cols": 90})
                .as_object()
                .unwrap(),
            &store,
            &state,
            &helper_context(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(response["result"]["resumed"], json!("stored-history"));
        assert_eq!(response["result"]["message_count"], json!(2));
        assert_eq!(response["result"]["messages"][0]["text"], json!("hello"));
        assert_eq!(response["result"]["messages"][1]["text"], json!("hi"));
    }

    #[test]
    fn session_branch_native_creates_lazy_child_session_with_copied_history() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-branch-parent".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-branch".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .set_session_title("stored-branch-parent", "Parent title")
            .unwrap();
        for (role, content) in [("user", "hello"), ("assistant", "hi")] {
            store
                .append_message(
                    "stored-branch-parent",
                    &MessageAppend {
                        role: role.to_string(),
                        content: Some(json!(content)),
                        tool_call_id: None,
                        tool_calls: None,
                        tool_name: None,
                        token_count: None,
                        finish_reason: None,
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: None,
                        codex_reasoning_items: None,
                        codex_message_items: None,
                    },
                )
                .unwrap();
        }
        let local_id =
            create_local_session(&state, Some("stored-branch-parent".to_string()), 92).unwrap();

        let response = handle_session_branch(
            json!("r-branch"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
            &helper_context(),
        )
        .unwrap()
        .expect("native session.branch response");

        let new_local_id = response["result"]["session_id"].as_str().unwrap();
        assert_ne!(new_local_id, "rs_tui_00000001");
        assert_eq!(response["result"]["title"], json!("Parent title #2"));
        assert_eq!(response["result"]["parent"], json!("stored-branch-parent"));

        let branched_store_id = lookup_store_session_id(&state, new_local_id)
            .unwrap()
            .expect("branched store session id");
        let branched_record = store
            .get_session(&branched_store_id)
            .unwrap()
            .expect("branched session row");
        assert_eq!(
            branched_record.parent_session_id.as_deref(),
            Some("stored-branch-parent")
        );
        assert_eq!(branched_record.model.as_deref(), Some("gpt-branch"));
        assert_eq!(
            store
                .get_session_title(&branched_store_id)
                .unwrap()
                .as_deref(),
            Some("Parent title #2")
        );

        let copied = store.get_messages(&branched_store_id).unwrap();
        assert_eq!(copied.len(), 2);
        assert_eq!(copied[0].role, "user");
        assert_eq!(copied[0].content, Some(json!("hello")));
        assert_eq!(copied[1].role, "assistant");
        assert_eq!(copied[1].content, Some(json!("hi")));
        assert!(
            lookup_child_session_id(&state, new_local_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn session_branch_native_rejects_empty_session() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-empty-branch".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-empty".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        let local_id =
            create_local_session(&state, Some("stored-empty-branch".to_string()), 80).unwrap();

        let response = handle_session_branch(
            json!("r-empty-branch"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
            &helper_context(),
        )
        .unwrap()
        .expect("native session.branch response");

        assert_eq!(response["error"]["code"], json!(4008));
    }

    #[test]
    fn prompt_background_requires_text() {
        let state = test_state();
        let local_id = create_local_session(&state, Some("stored-bg".to_string()), 80).unwrap();
        let stdout = Arc::new(Mutex::new(io::stdout()));

        let response = handle_prompt_background(
            json!("r-bg-empty"),
            json!({"session_id": local_id, "text": ""})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
            &stdout,
        )
        .unwrap()
        .expect("prompt.background response");

        assert_eq!(response["error"]["code"], json!(4012));
    }

    #[test]
    fn prompt_background_rejects_missing_session() {
        let state = test_state();
        let stdout = Arc::new(Mutex::new(io::stdout()));

        let response = handle_prompt_background(
            json!("r-bg-missing"),
            json!({"session_id": "rs_tui_missing", "text": "hello"})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
            &stdout,
        )
        .unwrap()
        .expect("prompt.background response");

        assert_eq!(response["error"]["code"], json!(4001));
    }

    #[test]
    fn session_close_native_releases_local_session_binding() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-close".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        let local_id = create_local_session(&state, Some("stored-close".to_string()), 80).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let response = handle_session_close(
            json!("r-close"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
            &child_stdin,
        )
        .unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response.unwrap()["result"]["closed"], json!(true));
        let guard = state.lock().unwrap();
        assert!(guard.sessions.is_empty());
        assert!(guard.child_to_local.is_empty());
    }

    #[test]
    fn session_title_response_caches_store_session_id() {
        let state = test_state();
        bind_child_session(&state, "child-title", None).unwrap();
        let _ = rewrite_outbound_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": "r-title",
                "method": "session.title",
                "params": {"session_id": "rs_tui_00000001"},
            }),
        )
        .unwrap();

        let _ = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","id":"r-title","result":{"session_key":"stored-title","title":"demo"}}"#,
        )
        .unwrap()
        .unwrap();

        let guard = state.lock().unwrap();
        assert_eq!(
            guard.sessions["rs_tui_00000001"]
                .store_session_id
                .as_deref(),
            Some("stored-title")
        );
        assert!(!guard.sessions["rs_tui_00000001"].store_session_dirty);
    }

    #[test]
    fn prompt_submit_marks_cached_store_session_id_dirty() {
        let state = test_state();
        bind_child_session(&state, "child-dirty", Some("stored-dirty".to_string())).unwrap();

        let _ = rewrite_outbound_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": "r-dirty",
                "method": "prompt.submit",
                "params": {"session_id": "rs_tui_00000001", "text": "hi"},
            }),
        )
        .unwrap();

        let guard = state.lock().unwrap();
        assert!(guard.sessions["rs_tui_00000001"].store_session_dirty);
    }

    fn test_store() -> SessionStore {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hermes-rs-agent-tui-gateway-{nonce}-{}.db",
            std::process::id()
        ));
        SessionStore::open(path).unwrap()
    }

    fn dummy_child_stdin() -> (Child, Arc<Mutex<ChildStdin>>) {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        (child, Arc::new(Mutex::new(stdin)))
    }

    fn dummy_child_echo() -> (Child, Arc<Mutex<ChildStdin>>, BufReader<ChildStdout>) {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        (child, Arc::new(Mutex::new(stdin)), BufReader::new(stdout))
    }

    fn helper_context() -> HelperContext {
        helper_context_at(std::env::temp_dir())
    }

    fn helper_context_at(hermes_home: PathBuf) -> HelperContext {
        let project_root = project_root();
        HelperContext {
            context: HermesContext::new(std::env::temp_dir())
                .with_hermes_home_env(Some(hermes_home.clone())),
            hermes_home,
            python: resolve_repo_python(&project_root).unwrap_or_else(|| PathBuf::from("python3")),
            python_path: compose_python_path(&project_root),
            work_root: project_root.clone(),
            project_root,
        }
    }

    fn wait_for_pending_request_id(state: &Arc<Mutex<ProxyState>>, method: &str) -> String {
        for _ in 0..100 {
            if let Some(request_id) = state
                .lock()
                .unwrap()
                .pending
                .iter()
                .find_map(|(id, pending)| (pending.method == method).then(|| id.clone()))
            {
                return request_id;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out waiting for pending {method} request");
    }

    #[test]
    fn session_title_is_native_for_clean_store_session_id() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-native".to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .set_session_title("stored-native", "Native title")
            .unwrap();
        bind_child_session(&state, "child-native", Some("stored-native".to_string())).unwrap();

        let response = handle_session_title(
            json!("r-native-title"),
            json!({"session_id": "rs_tui_00000001"})
                .as_object()
                .unwrap(),
            &store,
            &state,
        )
        .unwrap();

        let response = response.expect("native session.title response");
        assert_eq!(response["result"]["title"], json!("Native title"));
        assert_eq!(response["result"]["session_key"], json!("stored-native"));
    }

    #[test]
    fn session_title_for_dirty_store_session_id_falls_back_to_child() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-stale".to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        bind_child_session(&state, "child-stale", Some("stored-stale".to_string())).unwrap();
        mark_store_session_dirty(&state, "rs_tui_00000001").unwrap();

        let response = handle_session_title(
            json!("r-stale-title"),
            json!({"session_id": "rs_tui_00000001"})
                .as_object()
                .unwrap(),
            &store,
            &state,
        )
        .unwrap();

        assert!(response.is_none());
    }

    #[test]
    fn session_save_native_writes_current_session_transcript_json() {
        let _cwd_guard = cwd_lock().lock().unwrap();
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-save".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-save".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "stored-save",
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(json!("hello")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        let local_id = create_local_session(&state, Some("stored-save".to_string()), 80).unwrap();

        let temp_root = std::env::temp_dir().join(format!(
            "hermes-rs-agent-save-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root).unwrap();
        let old_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&temp_root).unwrap();

        let response = handle_session_save(
            json!("r-save"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
        )
        .unwrap()
        .expect("native session.save response");

        std::env::set_current_dir(&old_cwd).unwrap();

        let file = response["result"]["file"].as_str().unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
        assert_eq!(saved["model"], json!("gpt-save"));
        assert_eq!(saved["messages"].as_array().unwrap().len(), 1);
        assert_eq!(saved["messages"][0]["role"], json!("user"));
        assert_eq!(saved["messages"][0]["content"], json!("hello"));
    }

    #[test]
    fn session_undo_native_rewrites_store_and_detaches_child_session() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-undo".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-undo".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        for (role, content) in [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
            ("assistant", "second answer"),
        ] {
            store
                .append_message(
                    "stored-undo",
                    &MessageAppend {
                        role: role.to_string(),
                        content: Some(json!(content)),
                        tool_call_id: None,
                        tool_calls: None,
                        tool_name: None,
                        token_count: None,
                        finish_reason: None,
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: None,
                        codex_reasoning_items: None,
                        codex_message_items: None,
                    },
                )
                .unwrap();
        }
        let local_id = create_local_session(&state, Some("stored-undo".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-undo", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "child.close");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"closed\":true}}}}"
                ),
            );
        });

        let response = handle_session_undo(
            json!("r-undo"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native session.undo response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["result"]["removed"], json!(2));
        assert!(
            lookup_child_session_id(&state, "rs_tui_00000001")
                .unwrap()
                .is_none()
        );
        assert!(session_exists(&state, "rs_tui_00000001").unwrap());

        let messages = store.get_messages("stored-undo").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, Some(json!("first question")));
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, Some(json!("first answer")));
    }

    #[test]
    fn session_info_event_reanchors_store_session_id_and_hides_internal_key() {
        let state = test_state();
        bind_child_session(&state, "child-info-key", Some("stored-stale".to_string())).unwrap();
        mark_store_session_dirty(&state, "rs_tui_00000001").unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"session.info","session_id":"child-info-key","payload":{"model":"gpt-live","session_key":"stored-live","usage":{"total":5}}}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();

        assert_eq!(parsed["params"]["session_id"], json!("rs_tui_00000001"));
        assert_eq!(
            parsed["params"]["payload"].get("session_key"),
            None,
            "internal session_key should not leak to the public event stream"
        );
        let guard = state.lock().unwrap();
        assert_eq!(
            guard.sessions["rs_tui_00000001"]
                .store_session_id
                .as_deref(),
            Some("stored-live")
        );
        assert!(!guard.sessions["rs_tui_00000001"].store_session_dirty);
    }

    #[test]
    fn message_complete_event_reanchors_store_session_id_and_hides_internal_key() {
        let state = test_state();
        bind_child_session(
            &state,
            "child-complete-key",
            Some("stored-before".to_string()),
        )
        .unwrap();
        mark_store_session_dirty(&state, "rs_tui_00000001").unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"message.complete","session_id":"child-complete-key","payload":{"text":"done","status":"complete","session_key":"stored-after","usage":{"calls":3,"total":21}}}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();

        assert_eq!(parsed["params"]["session_id"], json!("rs_tui_00000001"));
        assert_eq!(
            parsed["params"]["payload"].get("session_key"),
            None,
            "internal session_key should not leak to the public event stream"
        );
        let guard = state.lock().unwrap();
        assert_eq!(
            guard.sessions["rs_tui_00000001"]
                .store_session_id
                .as_deref(),
            Some("stored-after")
        );
        assert!(!guard.sessions["rs_tui_00000001"].store_session_dirty);
        assert_eq!(guard.sessions["rs_tui_00000001"].usage["calls"], json!(3));
        assert_eq!(guard.sessions["rs_tui_00000001"].usage["total"], json!(21));
    }

    #[test]
    fn session_interrupt_is_native_for_lazy_local_session() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-interrupt".to_string()), 80).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();

        let response = handle_session_interrupt(
            json!("r-interrupt"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native session.interrupt response");
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["result"]["status"], json!("interrupted"));
    }

    #[test]
    fn session_interrupt_with_child_sends_internal_interrupt_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-interrupt-child".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-interrupt", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();

        let response = handle_session_interrupt(
            json!("r-interrupt-child"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native session.interrupt response");
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["result"]["status"], json!("interrupted"));
        let guard = state.lock().unwrap();
        let pending = guard
            .pending
            .values()
            .find(|pending| pending.method == "session.interrupt")
            .expect("internal session.interrupt request");
        assert!(pending.suppress_output);
    }

    #[test]
    fn prompt_submit_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-submit-child".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-submit", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "prompt.submit");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"status\":\"streaming\"}}}}"
                ),
            );
        });

        let response = handle_prompt_submit(
            json!("r-submit-child"),
            json!({"session_id": local_id, "text": "hello"})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native prompt.submit response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-submit-child"));
        assert_eq!(response["result"]["status"], json!("streaming"));
        let guard = state.lock().unwrap();
        assert!(guard.sessions[&local_id].store_session_dirty);
    }

    #[test]
    fn session_compress_with_child_reanchors_store_session_from_nested_info() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-compress-before".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-compress", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "session.compress");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"status\":\"compressed\",\"removed\":2,\"info\":{{\"session_key\":\"stored-compress-after\",\"model\":\"demo/model\"}},\"messages\":[]}}}}"
                ),
            );
        });

        mark_store_session_dirty(&state, &local_id).unwrap();
        let response = handle_session_compress(
            json!("r-compress-child"),
            json!({"session_id": local_id, "focus_topic": "summary"})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("session.compress response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-compress-child"));
        assert_eq!(response["result"]["status"], json!("compressed"));
        assert_eq!(response["result"]["removed"], json!(2));
        let guard = state.lock().unwrap();
        assert_eq!(
            guard.sessions[&local_id].store_session_id.as_deref(),
            Some("stored-compress-after")
        );
        assert!(!guard.sessions[&local_id].store_session_dirty);
    }

    #[test]
    fn rollback_restore_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-rollback".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-rollback", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "rollback.restore");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"success\":true,\"history_removed\":2,\"restored_to\":\"abc123\"}}}}"
                ),
            );
        });

        let response = handle_session_bound_child_request(
            json!("r-rollback-restore"),
            "rollback.restore",
            json!({"session_id": local_id, "hash": "abc123"})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("rollback.restore response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-rollback-restore"));
        assert_eq!(response["result"]["success"], json!(true));
        assert_eq!(response["result"]["history_removed"], json!(2));
        assert_eq!(response["result"]["restored_to"], json!("abc123"));
    }

    #[test]
    fn browser_manage_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-browser".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-browser", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "browser.manage");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"connected\":true,\"url\":\"http://127.0.0.1:9222\"}}}}"
                ),
            );
        });

        let response = handle_browser_manage(
            json!("r-browser-manage"),
            json!({"action": "status", "session_id": local_id})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("browser.manage response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-browser-manage"));
        assert_eq!(response["result"]["connected"], json!(true));
        assert_eq!(response["result"]["url"], json!("http://127.0.0.1:9222"));
    }

    #[test]
    fn skills_manage_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "skills.manage");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"skills\":{{\"Built-in\":[\"demo\"]}}}}}}"
                ),
            );
        });

        let response = forward_child_request_timeout(
            json!("r-skills-manage"),
            "skills.manage",
            json!({"action": "list"}).as_object().unwrap(),
            &state,
            &child_stdin,
            false,
            Duration::from_secs(45),
        )
        .unwrap()
        .expect("skills.manage response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-skills-manage"));
        assert_eq!(response["result"]["skills"]["Built-in"][0], json!("demo"));
    }

    #[test]
    fn voice_toggle_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "voice.toggle");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"enabled\":true,\"tts\":false,\"record_key\":\"ctrl+b\"}}}}"
                ),
            );
        });

        let response = forward_child_request_timeout(
            json!("r-voice-toggle"),
            "voice.toggle",
            json!({"action": "status"}).as_object().unwrap(),
            &state,
            &child_stdin,
            false,
            Duration::from_secs(20),
        )
        .unwrap()
        .expect("voice.toggle response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-voice-toggle"));
        assert_eq!(response["result"]["enabled"], json!(true));
        assert_eq!(response["result"]["record_key"], json!("ctrl+b"));
    }

    #[test]
    fn voice_record_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "voice.record");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"status\":\"recording\"}}}}"
                ),
            );
        });

        let response = forward_child_request_timeout(
            json!("r-voice-record"),
            "voice.record",
            json!({"action": "start"}).as_object().unwrap(),
            &state,
            &child_stdin,
            false,
            Duration::from_secs(20),
        )
        .unwrap()
        .expect("voice.record response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-voice-record"));
        assert_eq!(response["result"]["status"], json!("recording"));
    }

    #[test]
    fn shell_exec_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "shell.exec");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"stdout\":\"hello\\n\",\"stderr\":\"\",\"code\":0}}}}"
                ),
            );
        });

        let response = forward_child_request_timeout(
            json!("r-shell-exec"),
            "shell.exec",
            json!({"command": "echo hello"}).as_object().unwrap(),
            &state,
            &child_stdin,
            false,
            Duration::from_secs(35),
        )
        .unwrap()
        .expect("shell.exec response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-shell-exec"));
        assert_eq!(response["result"]["stdout"], json!("hello\n"));
        assert_eq!(response["result"]["code"], json!(0));
    }

    #[test]
    fn session_history_is_native_and_includes_ancestors() {
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "parent-history".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-parent".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "parent-history",
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(json!("from parent")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store
            .create_session(&SessionCreate {
                id: "child-history".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-child".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: Some("parent-history".to_string()),
            })
            .unwrap();
        store
            .append_message(
                "child-history",
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(json!("")),
                    tool_call_id: None,
                    tool_calls: Some(json!([{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": "{}"
                        }
                    }])),
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store
            .append_message(
                "child-history",
                &MessageAppend {
                    role: "tool".to_string(),
                    content: Some(json!("{\"success\":true}")),
                    tool_call_id: Some("call-1".to_string()),
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store
            .append_message(
                "child-history",
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(json!("done")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        let state = test_state();
        let local_id = create_local_session(&state, Some("child-history".to_string()), 80).unwrap();
        let response = handle_session_history(
            json!("r-session-history"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
        )
        .unwrap()
        .expect("session.history response");

        assert_eq!(response["id"], json!("r-session-history"));
        assert_eq!(response["result"]["count"], json!(4));
        assert_eq!(
            response["result"]["messages"][0]["text"],
            json!("from parent")
        );
        assert_eq!(response["result"]["messages"][1]["role"], json!("tool"));
        assert_eq!(response["result"]["messages"][1]["name"], json!("terminal"));
        assert_eq!(response["result"]["messages"][2]["text"], json!("done"));
    }

    #[test]
    fn session_history_rejects_missing_store_session() {
        let store = test_store();
        let state = test_state();
        let local_id = create_local_session(&state, None, 80).unwrap();

        let response = handle_session_history(
            json!("r-session-history-missing"),
            json!({"session_id": local_id}).as_object().unwrap(),
            &store,
            &state,
        )
        .unwrap()
        .expect("missing session.history response");

        assert_eq!(response["error"]["code"], json!(4001));
    }

    #[test]
    fn config_set_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-config-child".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-config", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let stdout = Arc::new(Mutex::new(io::stdout()));
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "config.set");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"key\":\"reasoning\",\"value\":\"high\"}}}}"
                ),
            );
        });

        let response = handle_config_set(
            json!("r-config-child"),
            json!({"key": "reasoning", "session_id": local_id, "value": "high"})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
            &stdout,
            &child_stdin,
        )
        .unwrap()
        .expect("config.set response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-config-child"));
        assert_eq!(response["result"]["key"], json!("reasoning"));
        assert_eq!(response["result"]["value"], json!("high"));
    }

    #[test]
    fn config_set_helper_path_uses_isolated_hermes_home() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let hermes_home = std::env::temp_dir().join(format!(
            "hermes-rs-agent-config-home-{nonce}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&hermes_home).unwrap();
        let state = test_state();
        let stdout = Arc::new(Mutex::new(io::stdout()));
        let (_child, child_stdin) = dummy_child_stdin();

        let response = handle_config_set(
            json!("r-config-helper"),
            json!({"key": "busy", "value": "queue"})
                .as_object()
                .unwrap(),
            &state,
            &helper_context_at(hermes_home.clone()),
            &stdout,
            &child_stdin,
        )
        .unwrap()
        .expect("config.set response");

        assert_eq!(response["id"], json!("r-config-helper"));
        assert_eq!(response["result"]["key"], json!("busy"));
        assert_eq!(response["result"]["value"], json!("queue"));
        let config_text = fs::read_to_string(hermes_home.join("config.yaml")).unwrap();
        assert!(config_text.contains("busy_input_mode: queue"));
    }

    #[test]
    fn config_get_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-config-get-child".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-config-get", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "config.get");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"value\":\"fast\"}}}}"
                ),
            );
        });

        let response = handle_config_get(
            json!("r-config-get-child"),
            json!({"key": "fast", "session_id": local_id})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
            &child_stdin,
        )
        .unwrap()
        .expect("config.get response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-config-get-child"));
        assert_eq!(response["result"]["value"], json!("fast"));
    }

    #[test]
    fn config_get_helper_path_reads_isolated_hermes_home() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let hermes_home = std::env::temp_dir().join(format!(
            "hermes-rs-agent-config-get-home-{nonce}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&hermes_home).unwrap();
        let state = test_state();
        let stdout = Arc::new(Mutex::new(io::stdout()));
        let (_child, child_stdin) = dummy_child_stdin();
        let helper = helper_context_at(hermes_home.clone());

        let _ = handle_config_set(
            json!("r-config-helper-set"),
            json!({"key": "busy", "value": "queue"})
                .as_object()
                .unwrap(),
            &state,
            &helper,
            &stdout,
            &child_stdin,
        )
        .unwrap()
        .expect("config.set response");
        let response = handle_config_get(
            json!("r-config-helper-get"),
            json!({"key": "busy"}).as_object().unwrap(),
            &state,
            &helper,
            &child_stdin,
        )
        .unwrap()
        .expect("config.get response");
        let mtime = handle_config_get(
            json!("r-config-helper-mtime"),
            json!({"key": "mtime"}).as_object().unwrap(),
            &state,
            &helper,
            &child_stdin,
        )
        .unwrap()
        .expect("config.get mtime response");

        assert_eq!(response["id"], json!("r-config-helper-get"));
        assert_eq!(response["result"]["value"], json!("queue"));
        assert!(mtime["result"]["mtime"].as_f64().unwrap_or_default() > 0.0);
    }

    #[test]
    fn model_options_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-model-options".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-model-options", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "model.options");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"model\":\"demo/model\",\"provider\":\"demo\",\"providers\":[]}}}}"
                ),
            );
        });

        let response = forward_child_request(
            json!("r-model-options"),
            "model.options",
            json!({"session_id": local_id}).as_object().unwrap(),
            &state,
            &child_stdin,
            true,
        )
        .unwrap()
        .expect("model.options response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-model-options"));
        assert_eq!(response["result"]["model"], json!("demo/model"));
        assert_eq!(response["result"]["provider"], json!("demo"));
    }

    #[test]
    fn spawn_tree_helper_dispatch_uses_isolated_hermes_home() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let hermes_home = std::env::temp_dir().join(format!(
            "hermes-rs-agent-spawn-tree-home-{nonce}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&hermes_home).unwrap();
        let helper = helper_context_at(hermes_home.clone());

        let save = handle_helper_dispatch(
            json!("r-spawn-save"),
            "spawn_tree.save",
            json!({
                "session_id": "rs_tui_00000001",
                "finished_at": 1234.0,
                "started_at": 1200.0,
                "label": "demo",
                "subagents": [{"id": "sub-1"}]
            })
            .as_object()
            .unwrap(),
            &helper,
        )
        .unwrap()
        .expect("spawn_tree.save response");
        let path = save["result"]["path"].as_str().unwrap().to_string();
        let listed = handle_helper_dispatch(
            json!("r-spawn-list"),
            "spawn_tree.list",
            json!({"session_id": "rs_tui_00000001", "limit": 10})
                .as_object()
                .unwrap(),
            &helper,
        )
        .unwrap()
        .expect("spawn_tree.list response");
        let loaded = handle_helper_dispatch(
            json!("r-spawn-load"),
            "spawn_tree.load",
            json!({"path": path}).as_object().unwrap(),
            &helper,
        )
        .unwrap()
        .expect("spawn_tree.load response");

        assert_eq!(listed["id"], json!("r-spawn-list"));
        assert_eq!(listed["result"]["entries"][0]["label"], json!("demo"));
        assert_eq!(loaded["result"]["session_id"], json!("rs_tui_00000001"));
        assert_eq!(loaded["result"]["subagents"][0]["id"], json!("sub-1"));
    }

    #[test]
    fn paste_collapse_writes_paste_file_in_hermes_home() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let hermes_home = std::env::temp_dir().join(format!(
            "hermes-rs-agent-paste-home-{nonce}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&hermes_home).unwrap();
        let state = test_state();

        let response = handle_paste_collapse(
            json!("r-paste"),
            json!({"text": "one\ntwo"}).as_object().unwrap(),
            &state,
            &helper_context_at(hermes_home.clone()),
        )
        .unwrap()
        .expect("paste.collapse response");

        let path = response["result"]["path"].as_str().unwrap();
        let written = fs::read_to_string(path).unwrap();
        assert_eq!(written, "one\ntwo");
        assert_eq!(response["result"]["lines"], json!(2));
        assert!(
            response["result"]["placeholder"]
                .as_str()
                .unwrap()
                .contains("Pasted text #1")
        );
    }

    #[test]
    fn session_steer_rejects_lazy_local_session_without_child() {
        let state = test_state();
        let local_id = create_local_session(&state, Some("stored-steer".to_string()), 80).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();

        let response = handle_session_steer(
            json!("r-steer"),
            json!({"session_id": local_id, "text": "keep going"})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native session.steer response");
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["result"]["status"], json!("rejected"));
        assert_eq!(response["result"]["text"], json!("keep going"));
    }

    #[test]
    fn child_prompt_request_event_maps_request_id_to_local_id() {
        let state = test_state();
        bind_child_session(&state, "child-prompt", Some("stored-prompt".to_string())).unwrap();

        let rewritten = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"clarify.request","session_id":"child-prompt","payload":{"question":"Continue?","choices":["yes","no"],"request_id":"child-req-1"}}}"#,
        )
        .unwrap()
        .unwrap();
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        let request_id = parsed["params"]["payload"]["request_id"]
            .as_str()
            .unwrap()
            .to_string();

        assert!(request_id.starts_with("rs_prompt_"));
        let binding = state
            .lock()
            .unwrap()
            .prompt_requests
            .get(&request_id)
            .cloned()
            .expect("prompt binding");
        assert_eq!(binding.child_request_id, "child-req-1");
        assert_eq!(binding.local_session_id, "rs_tui_00000001");
    }

    #[test]
    fn session_steer_with_child_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-steer-child".to_string()), 80).unwrap();
        attach_child_to_local(&state, &local_id, "child-steer", None).unwrap();
        let (mut child, child_stdin) = dummy_child_stdin();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "session.steer");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"status\":\"queued\",\"text\":\"keep going\"}}}}"
                ),
            );
        });

        let response = handle_session_steer(
            json!("r-steer-child"),
            json!({"session_id": local_id, "text": "keep going"})
                .as_object()
                .unwrap(),
            &state,
            &child_stdin,
        )
        .unwrap()
        .expect("native session.steer response");
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-steer-child"));
        assert_eq!(response["result"]["status"], json!("queued"));
        assert_eq!(response["result"]["text"], json!("keep going"));
    }

    #[test]
    fn approval_respond_is_native_for_clean_store_session_id() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-approval".to_string()), 80).unwrap();

        let response = handle_approval_respond(
            json!("r-approval"),
            json!({"session_id": local_id, "choice": "deny"})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
        )
        .unwrap()
        .expect("native approval.respond response");

        assert_eq!(response["result"]["resolved"], json!(0));
    }

    #[test]
    fn session_usage_reads_cached_child_usage() {
        let state = test_state();
        bind_child_session(&state, "child-usage", Some("stored-usage".to_string())).unwrap();

        let _ = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"session.info","session_id":"child-usage","payload":{"model":"gpt-test","usage":{"calls":2,"input":11,"output":7,"total":18}}}}"#,
        )
        .unwrap()
        .unwrap();

        let response = handle_session_usage(
            json!("r-usage"),
            json!({"session_id": "rs_tui_00000001"})
                .as_object()
                .unwrap(),
            &state,
        )
        .unwrap()
        .expect("native session.usage response");

        assert_eq!(response["result"]["model"], json!("gpt-test"));
        assert_eq!(response["result"]["calls"], json!(2));
        assert_eq!(response["result"]["total"], json!(18));
    }

    #[test]
    fn session_status_uses_cached_runtime_state() {
        let state = test_state();
        let store = test_store();
        store
            .create_session(&SessionCreate {
                id: "stored-status".to_string(),
                source: "tui".to_string(),
                user_id: None,
                model: Some("gpt-status".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .set_session_title("stored-status", "Status title")
            .unwrap();
        bind_child_session(&state, "child-status", Some("stored-status".to_string())).unwrap();
        let _ = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"session.info","session_id":"child-status","payload":{"model":"gpt-live","usage":{"calls":1,"total":42}}}}"#,
        )
        .unwrap()
        .unwrap();
        let _ = rewrite_child_line(
            &state,
            r#"{"jsonrpc":"2.0","method":"event","params":{"type":"message.start","session_id":"child-status"}}"#,
        )
        .unwrap()
        .unwrap();

        let response = handle_session_status(
            json!("r-status"),
            json!({"session_id": "rs_tui_00000001"})
                .as_object()
                .unwrap(),
            &store,
            &state,
            &helper_context(),
        )
        .unwrap()
        .expect("native session.status response");
        let output = response["result"]["output"].as_str().unwrap();

        assert!(output.contains("Session ID: stored-status"));
        assert!(output.contains("Title: Status title"));
        assert!(output.contains("Model: gpt-live (unknown)"));
        assert!(output.contains("Tokens: 42"));
        assert!(output.contains("Agent Running: Yes"));
    }

    #[test]
    fn approval_respond_is_native_when_store_session_id_is_dirty() {
        let state = test_state();
        let local_id =
            create_local_session(&state, Some("stored-dirty-approval".to_string()), 80).unwrap();
        mark_store_session_dirty(&state, &local_id).unwrap();

        let response = handle_approval_respond(
            json!("r-approval-dirty"),
            json!({"session_id": local_id, "choice": "deny"})
                .as_object()
                .unwrap(),
            &state,
            &helper_context(),
        )
        .unwrap()
        .expect("native approval.respond response");

        assert_eq!(response["result"]["resolved"], json!(0));
    }

    #[test]
    fn clarify_respond_round_trips_through_blocking_internal_request() {
        let state = test_state();
        let prompt_request_id =
            bind_prompt_request(&state, "rs_tui_00000001", "child-req-1").unwrap();
        let (mut child, child_stdin, mut child_stdout) = dummy_child_echo();
        let state_for_thread = Arc::clone(&state);
        let responder = thread::spawn(move || {
            let request_id = wait_for_pending_request_id(&state_for_thread, "clarify.respond");
            let _ = rewrite_child_line(
                &state_for_thread,
                &format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":\"{request_id}\",\"result\":{{\"status\":\"ok\"}}}}"
                ),
            );
        });

        let response = handle_text_respond(
            json!("r-clarify"),
            json!({"request_id": prompt_request_id, "answer": "yes"})
                .as_object()
                .unwrap(),
            &child_stdin,
            &state,
            "clarify.respond",
            "request_id",
            "answer",
        )
        .unwrap()
        .expect("native clarify.respond response");
        let mut outbound = String::new();
        child_stdout.read_line(&mut outbound).unwrap();
        responder.join().unwrap();
        let _ = child.kill();
        let _ = child.wait();

        assert_eq!(response["id"], json!("r-clarify"));
        assert_eq!(response["result"]["status"], json!("ok"));
        let outbound: Value = serde_json::from_str(outbound.trim()).unwrap();
        assert_eq!(outbound["params"]["request_id"], json!("child-req-1"));
        assert_eq!(outbound["params"]["answer"], json!("yes"));
    }

    #[test]
    fn detect_file_drop_matches_repo_image_with_remainder() {
        let root = project_root();
        let input = "./website/static/img/logo.png describe this";
        let detected = detect_file_drop(input, &root).expect("detect drop");

        assert!(detected.is_image);
        assert!(detected.path.ends_with("website/static/img/logo.png"));
        assert_eq!(detected.remainder, "describe this");
    }

    #[test]
    fn image_attach_queues_image_and_returns_remainder() {
        let state = test_state();
        bind_child_session(&state, "child-image", Some("stored-image".to_string())).unwrap();
        let helper = helper_context();
        let response = handle_image_attach(
            json!("r-image"),
            json!({
                "session_id": "rs_tui_00000001",
                "path": "./website/static/img/logo.png summarize this",
            })
            .as_object()
            .unwrap(),
            &state,
            &helper,
        )
        .unwrap()
        .expect("image.attach response");

        assert_eq!(response["result"]["name"], json!("logo.png"));
        assert_eq!(response["result"]["remainder"], json!("summarize this"));
        assert_eq!(response["result"]["count"], json!(1));
        let guard = state.lock().unwrap();
        assert_eq!(guard.sessions["rs_tui_00000001"].attached_images.len(), 1);
    }

    #[test]
    fn complete_path_response_lists_matching_relative_directory() {
        let root = project_root();
        let result = complete_path_response("./ui-t", &root);
        let items = result["items"].as_array().unwrap();
        assert!(items.iter().any(|item| item["text"] == json!("./ui-tui/")));
    }

    #[test]
    fn details_completions_offer_section_matches() {
        let result = details_completions("/details th").unwrap();
        let items = result["items"].as_array().unwrap();
        assert!(items.iter().any(|item| item["text"] == json!("thinking")));
        assert_eq!(result["replace_from"], json!(9));
    }
}
