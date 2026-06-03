//! Kanban tools — structured tool-call surface for worker + orchestrator agents.
//!
//! Native Rust port of `tools/kanban_tools.py`.
//!
//! These tools are only registered into the model's schema when the agent is
//! running under the dispatcher (`HERMES_KANBAN_TASK` set) or when the current
//! profile has the `kanban` toolset enabled. A normal `hermes chat` session
//! sees **zero** kanban tools in its schema.
//!
//! Why tools instead of just shelling out to `hermes kanban`?
//!
//! 1. **Backend portability.** A worker whose terminal tool points at Docker /
//!    Modal / Singularity / SSH would run `hermes kanban complete …` inside the
//!    container, where `hermes` isn't installed and the DB isn't mounted. Tools
//!    run in the agent's process, so they always reach the local kanban DB
//!    regardless of terminal backend.
//! 2. **No shell-quoting footguns.** Structured tool args skip shlex+argparse.
//! 3. **Better errors.** Tool-call failures return structured JSON the model can
//!    reason about, not stderr strings it has to parse.
//!
//! Humans continue to use the CLI / dashboard / slash command. These tools are
//! ONLY for the worker agent's handoff back to the kernel.
//!
//! Each handler takes the parsed JSON arguments object and returns a JSON-encoded
//! result string (matching the Python `(args: dict, **kw) -> str` contract).

use std::env;

use serde_json::{json, Value};

use hermes_core::cli_kanban_db as kb;

// ---------------------------------------------------------------------------
// tool_error / ok shims
// ---------------------------------------------------------------------------

/// Mirror of `tools.registry.tool_error` (the single-arg form used here):
/// returns a JSON object `{"error": "<message>"}`.
pub fn tool_error(message: impl AsRef<str>) -> String {
    json!({ "error": message.as_ref() }).to_string()
}

/// `_ok(**fields)` — `{"ok": true, ...fields}`.
fn ok(fields: Value) -> String {
    let mut map = serde_json::Map::new();
    map.insert("ok".to_string(), Value::Bool(true));
    if let Value::Object(obj) = fields {
        for (k, v) in obj {
            map.insert(k, v);
        }
    }
    Value::Object(map).to_string()
}

// ---------------------------------------------------------------------------
// Gating
// ---------------------------------------------------------------------------

fn env_nonempty(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Tools are available when:
///
/// 1. `HERMES_KANBAN_TASK` is set (dispatcher-spawned worker), OR
/// 2. The current profile has `kanban` in its `toolsets` config.
///
/// Humans running `hermes chat` without the kanban toolset see zero kanban
/// tools. Workers spawned by the kanban dispatcher and orchestrator profiles
/// with the kanban toolset enabled see all seven.
pub fn check_kanban_mode() -> bool {
    if env_nonempty("HERMES_KANBAN_TASK").is_some() {
        return true;
    }

    // Check if the current profile has the kanban toolset enabled.
    // `load_config()` returns a `serde_yaml::Value`; mirrors the Python
    // `cfg.get("toolsets", [])` membership check.
    let cfg = hermes_core::cli_config::load_config();
    match cfg.get("toolsets") {
        Some(serde_yaml::Value::Sequence(seq)) => seq
            .iter()
            .any(|v| v.as_str() == Some("kanban")),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Resolve `task_id` arg or fall back to the env var the dispatcher set.
fn default_task_id(arg: Option<&str>) -> Option<String> {
    if let Some(a) = arg {
        if !a.is_empty() {
            return Some(a.to_string());
        }
    }
    env_nonempty("HERMES_KANBAN_TASK")
}

/// Return this worker's dispatcher run id when it is scoped to `task_id`.
fn worker_run_id(task_id: &str) -> Option<i64> {
    if env::var("HERMES_KANBAN_TASK").ok().as_deref() != Some(task_id) {
        return None;
    }
    let raw = env_nonempty("HERMES_KANBAN_RUN_ID")?;
    raw.parse::<i64>().ok()
}

/// Reject worker-driven destructive calls on foreign task IDs.
///
/// A process spawned by the dispatcher has `HERMES_KANBAN_TASK` set to its own
/// task id. Tools like `kanban_complete` / `kanban_block` / `kanban_heartbeat`
/// mutate run-lifecycle state, so a buggy or prompt-injected worker that passed
/// an explicit `task_id` for some other task could corrupt sibling or
/// cross-tenant runs.
///
/// Orchestrator profiles (kanban toolset enabled but **no** `HERMES_KANBAN_TASK`
/// in env) aren't subject to this check.
///
/// Returns `None` when the call is allowed, or a tool-error string when it must
/// be rejected. Callers should `return` the error verbatim.
fn enforce_worker_task_ownership(tid: &str) -> Option<String> {
    let env_tid = match env_nonempty("HERMES_KANBAN_TASK") {
        Some(v) => v,
        // Orchestrator or CLI context — no task-scope restriction.
        None => return None,
    };
    if tid != env_tid {
        return Some(tool_error(format!(
            "worker is scoped to task {env_tid}; refusing to mutate {tid}. \
             Use kanban_comment to hand off information to other tasks, or \
             kanban_create to spawn follow-up work."
        )));
    }
    None
}

/// Connect to the kanban DB (default board / path).
fn connect() -> kb::Result<rusqlite::Connection> {
    kb::connect(None, None)
}

// Arg-extraction helpers operating on the serde_json args object.

fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

/// "truthy non-empty string": present, a string, and not empty.
fn arg_nonempty_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    match arg_str(args, key) {
        Some(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn task_dict(t: &kb::Task) -> Value {
    json!({
        "id": t.id,
        "title": t.title,
        "body": t.body,
        "assignee": t.assignee,
        "status": t.status,
        "tenant": t.tenant,
        "priority": t.priority,
        "workspace_kind": t.workspace_kind,
        "workspace_path": t.workspace_path,
        "created_by": t.created_by,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "completed_at": t.completed_at,
        "result": t.result,
        "current_run_id": t.current_run_id,
    })
}

fn run_dict(r: &kb::Run) -> Value {
    json!({
        "id": r.id,
        "profile": r.profile,
        "status": r.status,
        "outcome": r.outcome,
        "summary": r.summary,
        "error": r.error,
        "metadata": r.metadata,
        "started_at": r.started_at,
        "ended_at": r.ended_at,
    })
}

/// Read a task's full state: task row, parents, children, comments, runs
/// (attempt history), and the last N events.
pub fn handle_show(args: &Value) -> String {
    let tid = match default_task_id(arg_str(args, "task_id")) {
        Some(t) => t,
        None => {
            return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
        }
    };

    let result: kb::Result<String> = (|| {
        let conn = connect()?;
        let task = match kb::get_task(&conn, &tid)? {
            Some(t) => t,
            None => return Ok(tool_error(format!("task {tid} not found"))),
        };
        let comments = kb::list_comments(&conn, &tid)?;
        let events = kb::list_events(&conn, &tid)?;
        let runs = kb::list_runs(&conn, &tid, false)?;
        let parents = kb::parent_ids(&conn, &tid)?;
        let children = kb::child_ids(&conn, &tid)?;
        let worker_context = kb::build_worker_context(&conn, &tid)?;

        // Cap events at the last 50 (full log via CLI).
        let events_tail: Vec<&kb::Event> = if events.len() > 50 {
            events[events.len() - 50..].iter().collect()
        } else {
            events.iter().collect()
        };

        let out = json!({
            "task": task_dict(&task),
            "parents": parents,
            "children": children,
            "comments": comments.iter().map(|c| json!({
                "author": c.author,
                "body": c.body,
                "created_at": c.created_at,
            })).collect::<Vec<_>>(),
            "events": events_tail.iter().map(|e| json!({
                "kind": e.kind,
                "payload": e.payload,
                "created_at": e.created_at,
                "run_id": e.run_id,
            })).collect::<Vec<_>>(),
            "runs": runs.iter().map(run_dict).collect::<Vec<_>>(),
            "worker_context": worker_context,
        });
        Ok(out.to_string())
    })();

    match result {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_show: {e}")),
    }
}

/// Mark the current task done with a structured handoff.
pub fn handle_complete(args: &Value) -> String {
    let tid = match default_task_id(arg_str(args, "task_id")) {
        Some(t) => t,
        None => {
            return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
        }
    };
    if let Some(err) = enforce_worker_task_ownership(&tid) {
        return err;
    }

    let summary = arg_str(args, "summary");
    let result = arg_str(args, "result");

    // created_cards: accept None, a single string, or a list of strings.
    let created_cards: Option<Vec<String>> = match args.get("created_cards") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => {
            // Accept a single id as a string for convenience, then normalise.
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Some(vec![])
            } else {
                Some(vec![trimmed.to_string()])
            }
        }
        Some(Value::Array(arr)) => {
            // Normalise: strings only, stripped, non-empty.
            let cleaned: Vec<String> = arr
                .iter()
                .map(value_to_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            Some(cleaned)
        }
        Some(other) => {
            return tool_error(format!(
                "created_cards must be a list of task ids, got {}",
                py_type_name(other)
            ))
        }
    };

    // metadata must be an object/dict if present.
    let metadata: Option<Value> = match args.get("metadata") {
        None | Some(Value::Null) => None,
        Some(v @ Value::Object(_)) => Some(v.clone()),
        Some(other) => {
            return tool_error(format!(
                "metadata must be an object/dict, got {}",
                py_type_name(other)
            ))
        }
    };

    let has_summary = summary.map(|s| !s.is_empty()).unwrap_or(false);
    let has_result = result.map(|s| !s.is_empty()).unwrap_or(false);
    if !(has_summary || has_result) {
        return tool_error("provide at least one of: summary (preferred), result");
    }

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        let opts = kb::CompleteOpts {
            result,
            summary,
            metadata,
            created_cards: created_cards.clone(),
            expected_run_id: worker_run_id(&tid),
        };
        match kb::complete_task(&conn, &tid, opts) {
            Err(kb::KanbanError::HallucinatedCards { phantom, .. }) => {
                // Structured rejection — surface the phantom ids so the worker
                // can retry with a corrected list or drop the field. Audit
                // event already landed in the DB.
                Ok(tool_error(format!(
                    "kanban_complete blocked: the following created_cards do not \
                     exist or were not created by this worker: {}. Either omit \
                     them, use only ids returned from successful kanban_create \
                     calls, or remove the created_cards field.",
                    phantom.join(", ")
                )))
            }
            Err(e) => Err(e),
            Ok(false) => Ok(tool_error(format!(
                "could not complete {tid} (unknown id or already terminal)"
            ))),
            Ok(true) => {
                let run = kb::latest_run(&conn, &tid)?;
                Ok(ok(json!({
                    "task_id": tid,
                    "run_id": run.map(|r| r.id),
                })))
            }
        }
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_complete: {e}")),
    }
}

/// Transition the task to blocked with a reason a human will read.
pub fn handle_block(args: &Value) -> String {
    let tid = match default_task_id(arg_str(args, "task_id")) {
        Some(t) => t,
        None => {
            return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
        }
    };
    if let Some(err) = enforce_worker_task_ownership(&tid) {
        return err;
    }
    let reason = match args.get("reason").and_then(|v| v.as_str()) {
        Some(r) if !r.trim().is_empty() => r,
        _ => return tool_error("reason is required — explain what input you need"),
    };

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        let ok_flag = kb::block_task(&conn, &tid, Some(reason), worker_run_id(&tid))?;
        if !ok_flag {
            return Ok(tool_error(format!(
                "could not block {tid} (unknown id or not in running/ready)"
            )));
        }
        let run = kb::latest_run(&conn, &tid)?;
        Ok(ok(json!({
            "task_id": tid,
            "run_id": run.map(|r| r.id),
        })))
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_block: {e}")),
    }
}

/// Signal that the worker is still alive during a long operation.
pub fn handle_heartbeat(args: &Value) -> String {
    let tid = match default_task_id(arg_str(args, "task_id")) {
        Some(t) => t,
        None => {
            return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
        }
    };
    if let Some(err) = enforce_worker_task_ownership(&tid) {
        return err;
    }
    let note = arg_str(args, "note");

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        let ok_flag = kb::heartbeat_worker(&conn, &tid, note, worker_run_id(&tid))?;
        if !ok_flag {
            return Ok(tool_error(format!(
                "could not heartbeat {tid} (unknown id or not running)"
            )));
        }
        Ok(ok(json!({ "task_id": tid })))
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_heartbeat: {e}")),
    }
}

/// Append a comment to a task's thread.
pub fn handle_comment(args: &Value) -> String {
    // Note: comment does NOT fall back to the env var — task_id stays explicit.
    let tid = match arg_nonempty_str(args, "task_id") {
        Some(t) => t.to_string(),
        None => {
            return tool_error(
                "task_id is required (use the current task id if that's what you \
                 mean — pulls from env but kept explicit here)",
            )
        }
    };
    let body = match args.get("body").and_then(|v| v.as_str()) {
        Some(b) if !b.trim().is_empty() => b.to_string(),
        _ => return tool_error("body is required"),
    };
    let author = arg_nonempty_str(args, "author")
        .map(|s| s.to_string())
        .or_else(|| env_nonempty("HERMES_PROFILE"))
        .unwrap_or_else(|| "worker".to_string());

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        let cid = kb::add_comment(&conn, &tid, &author, &body)?;
        Ok(ok(json!({ "task_id": tid, "comment_id": cid })))
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_comment: {e}")),
    }
}

/// Create a child task. Orchestrator workers use this to fan out.
pub fn handle_create(args: &Value) -> String {
    let title = match args.get("title").and_then(|v| v.as_str()) {
        Some(t) if !t.trim().is_empty() => t,
        _ => return tool_error("title is required"),
    };
    let assignee = match arg_nonempty_str(args, "assignee") {
        Some(a) => a,
        None => {
            return tool_error(
                "assignee is required — name the profile that should execute this \
                 task (the dispatcher will only spawn tasks with an assignee)",
            )
        }
    };
    let body = arg_str(args, "body");

    // parents: accept None/empty, a single string, or a list of strings.
    let parents: Vec<String> = match args.get("parents") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.to_string()],
        Some(Value::Array(arr)) => arr.iter().map(value_to_str).collect(),
        Some(other) => {
            return tool_error(format!(
                "parents must be a list of task ids, got {}",
                py_type_name(other)
            ))
        }
    };

    let tenant = arg_nonempty_str(args, "tenant")
        .map(|s| s.to_string())
        .or_else(|| env_nonempty("HERMES_TENANT"));

    // priority: int(priority) if not None else 0
    let priority: i64 = match args.get("priority") {
        None | Some(Value::Null) => 0,
        Some(v) => match value_to_int(v) {
            Some(p) => p,
            None => return tool_error(format!(
                "kanban_create: invalid priority {}",
                value_to_str(v)
            )),
        },
    };

    let workspace_kind = arg_nonempty_str(args, "workspace_kind").unwrap_or("scratch");
    let workspace_path = arg_str(args, "workspace_path");
    let triage = args
        .get("triage")
        .map(value_truthy)
        .unwrap_or(false);
    let idempotency_key = arg_str(args, "idempotency_key");

    // max_runtime_seconds: int(...) if not None else None
    let max_runtime_seconds: Option<i64> = match args.get("max_runtime_seconds") {
        None | Some(Value::Null) => None,
        Some(v) => match value_to_int(v) {
            Some(n) => Some(n),
            None => return tool_error(format!(
                "kanban_create: invalid max_runtime_seconds {}",
                value_to_str(v)
            )),
        },
    };

    // skills: accept None, a single string, or a list of strings.
    let skills: Option<Vec<String>> = match args.get("skills") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(vec![s.to_string()]),
        Some(Value::Array(arr)) => Some(arr.iter().map(value_to_str).collect()),
        Some(other) => {
            return tool_error(format!(
                "skills must be a list of skill names, got {}",
                py_type_name(other)
            ))
        }
    };

    let created_by = env_nonempty("HERMES_PROFILE").unwrap_or_else(|| "worker".to_string());

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        let opts = kb::CreateTask {
            title: title.trim(),
            body,
            assignee: Some(assignee),
            created_by: Some(&created_by),
            workspace_kind,
            workspace_path,
            tenant: tenant.as_deref(),
            priority,
            parents: parents.clone(),
            triage,
            idempotency_key,
            max_runtime_seconds,
            skills: skills.clone(),
        };
        let new_tid = kb::create_task(&conn, opts)?;
        let new_task = kb::get_task(&conn, &new_tid)?;
        Ok(ok(json!({
            "task_id": new_tid,
            "status": new_task.map(|t| t.status),
        })))
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_create: {e}")),
    }
}

/// Add a parent→child dependency edge after the fact.
pub fn handle_link(args: &Value) -> String {
    let parent_id = arg_nonempty_str(args, "parent_id");
    let child_id = arg_nonempty_str(args, "child_id");
    let (parent_id, child_id) = match (parent_id, child_id) {
        (Some(p), Some(c)) => (p, c),
        _ => return tool_error("both parent_id and child_id are required"),
    };

    let result_val: kb::Result<String> = (|| {
        let conn = connect()?;
        // ValueError in Python covers cycle + self-parent rejections; here both
        // map to KanbanError::Value, surfaced with the same prefix below.
        kb::link_tasks(&conn, parent_id, child_id)?;
        Ok(ok(json!({ "parent_id": parent_id, "child_id": child_id })))
    })();

    match result_val {
        Ok(s) => s,
        Err(e) => tool_error(format!("kanban_link: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Value coercion helpers (mirroring Python str()/int()/bool() semantics)
// ---------------------------------------------------------------------------

/// Python `str(value)` for the JSON values we expect inside arrays.
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Coerce to i64 the way Python `int(x)` would for ints / numeric strings.
fn value_to_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                // Python int(float) truncates toward zero.
                n.as_f64().map(|f| f.trunc() as i64)
            }
        }
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

/// Python truthiness for the JSON values used by `triage`.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Approximation of Python `type(x).__name__` for the error messages.
fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const DESC_TASK_ID_DEFAULT: &str = "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env \
     (the task the dispatcher spawned you to work on).";

pub fn kanban_show_schema() -> Value {
    json!({
        "name": "kanban_show",
        "description": "Read a task's full state — title, body, assignee, parent task handoffs, \
your prior attempts on this task if any, comments, and recent events. Use this to (re)orient \
yourself before starting work, especially on retries. The response includes a pre-formatted \
``worker_context`` string suitable for inclusion verbatim in your reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": DESC_TASK_ID_DEFAULT },
            },
            "required": [],
        },
    })
}

pub fn kanban_complete_schema() -> Value {
    json!({
        "name": "kanban_complete",
        "description": "Mark your current task done with a structured handoff for downstream \
workers and humans. Prefer ``summary`` for a human-readable 1-3 sentence description of what you \
did; put machine-readable facts in ``metadata`` (changed_files, tests_run, decisions, findings, \
etc). At least one of ``summary`` or ``result`` is required. If you created new tasks via \
``kanban_create`` during this run, list their ids in ``created_cards`` — the kernel verifies them \
so phantom references are caught before they leak into downstream automation.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": DESC_TASK_ID_DEFAULT },
                "summary": {
                    "type": "string",
                    "description": "Human-readable handoff, 1-3 sentences. Appears in Run History \
on the dashboard and in downstream workers' context.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Free-form dict of structured facts about this attempt — \
{\"changed_files\": [...], \"tests_run\": 12, \"findings\": [...]}. Surfaced to downstream workers \
alongside ``summary``.",
                },
                "result": {
                    "type": "string",
                    "description": "Short result log line (legacy field, maps to task.result). Use \
``summary`` instead when possible; this exists for compatibility with callers that still set \
--result on the CLI.",
                },
                "created_cards": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional structured manifest of task ids you created via \
``kanban_create`` during this run. The kernel verifies each id exists and was created by this \
worker's profile; any phantom id blocks the completion with an error listing what went wrong \
(auditable in the task's events). Only list ids you got back from a successful ``kanban_create`` \
call — do not invent or remember ids from prose. Omit the field if you did not create any cards.",
                },
            },
            "required": [],
        },
    })
}

pub fn kanban_block_schema() -> Value {
    json!({
        "name": "kanban_block",
        "description": "Transition the task to blocked because you need human input to proceed. \
``reason`` will be shown to the human on the board and included in context when someone unblocks \
you. Use for genuine blockers only — don't block on things you can resolve yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": DESC_TASK_ID_DEFAULT },
                "reason": {
                    "type": "string",
                    "description": "What you need answered, in one or two sentences. Don't paste \
the whole conversation; the human has the board and can ask follow-ups via comments.",
                },
            },
            "required": ["reason"],
        },
    })
}

pub fn kanban_heartbeat_schema() -> Value {
    json!({
        "name": "kanban_heartbeat",
        "description": "Signal that you're still alive during a long operation (training, \
encoding, large crawls). Call every few minutes so humans see liveness separately from PID \
checks. Pure side effect — no work changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": DESC_TASK_ID_DEFAULT },
                "note": {
                    "type": "string",
                    "description": "Optional short note describing current progress. Shown in the \
event log.",
                },
            },
            "required": [],
        },
    })
}

pub fn kanban_comment_schema() -> Value {
    json!({
        "name": "kanban_comment",
        "description": "Append a comment to a task's thread. Use for durable notes that should \
outlive this run (questions for the next worker, partial findings, rationale). Ephemeral reasoning \
doesn't belong here — use your normal response instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id. Required (may be your own task or another's — comment \
threads are per-task).",
                },
                "body": { "type": "string", "description": "Markdown-supported comment body." },
                "author": {
                    "type": "string",
                    "description": "Override author name. Defaults to the current profile \
(HERMES_PROFILE env).",
                },
            },
            "required": ["task_id", "body"],
        },
    })
}

pub fn kanban_create_schema() -> Value {
    json!({
        "name": "kanban_create",
        "description": "Create a new kanban task, optionally as a child of the current one (pass \
the current task id in ``parents``). Used by orchestrator workers to fan out — decompose work into \
child tasks with specific assignees, link them into a pipeline, then complete your own task. The \
dispatcher picks up the new tasks on its next tick and spawns the assigned profiles.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": { "type": "string", "description": "Short task title (required)." },
                "assignee": {
                    "type": "string",
                    "description": "Profile name that should execute this task (e.g. \
'researcher-a', 'reviewer', 'writer'). Required — tasks without an assignee are never dispatched.",
                },
                "body": {
                    "type": "string",
                    "description": "Opening post: full spec, acceptance criteria, links. The \
assigned worker reads this as part of its context.",
                },
                "parents": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Parent task ids. The new task stays in 'todo' until every \
parent reaches 'done'; then it auto-promotes to 'ready'. Typical fan-in: list all the researcher \
task ids when creating a synthesizer task.",
                },
                "tenant": {
                    "type": "string",
                    "description": "Optional namespace for multi-project isolation. Defaults to \
HERMES_TENANT env if set.",
                },
                "priority": {
                    "type": "integer",
                    "description": "Dispatcher tiebreaker. Higher = picked sooner when multiple \
ready tasks share an assignee.",
                },
                "workspace_kind": {
                    "type": "string",
                    "enum": ["scratch", "dir", "worktree"],
                    "description": "Workspace flavor: 'scratch' (fresh tmp dir, default), 'dir' \
(shared directory, requires absolute workspace_path), 'worktree' (git worktree).",
                },
                "workspace_path": {
                    "type": "string",
                    "description": "Absolute path for 'dir' or 'worktree' workspace. Relative paths \
are rejected at dispatch.",
                },
                "triage": {
                    "type": "boolean",
                    "description": "If true, task lands in 'triage' instead of 'todo' — a specifier \
profile is expected to flesh out the body before work starts.",
                },
                "idempotency_key": {
                    "type": "string",
                    "description": "If a non-archived task with this key already exists, return \
that task's id instead of creating a duplicate. Useful for retry-safe automation.",
                },
                "max_runtime_seconds": {
                    "type": "integer",
                    "description": "Per-task runtime cap. When exceeded, the dispatcher SIGTERMs \
the worker and re-queues the task with outcome='timed_out'.",
                },
                "skills": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Skill names to force-load into the dispatched worker (in \
addition to the built-in kanban-worker skill). Use this to pin a task to a specialist context — \
e.g. ['translation'] for a translation task, ['github-code-review'] for a reviewer task. The names \
must match skills installed on the assignee's profile.",
                },
            },
            "required": ["title", "assignee"],
        },
    })
}

pub fn kanban_link_schema() -> Value {
    json!({
        "name": "kanban_link",
        "description": "Add a parent→child dependency edge after both tasks already exist. The \
child won't promote to 'ready' until all parents are 'done'. Cycles and self-links are rejected.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": { "type": "string", "description": "Parent task id." },
                "child_id": { "type": "string", "description": "Child task id." },
            },
            "required": ["parent_id", "child_id"],
        },
    })
}

// ---------------------------------------------------------------------------
// Registration metadata
// ---------------------------------------------------------------------------

/// A registerable kanban tool: name, toolset, schema, emoji, and a dispatch
/// function. The integration layer wires these into `tool_registry` with
/// `check_fn = check_kanban_mode`.
pub struct KanbanToolDef {
    pub name: &'static str,
    pub toolset: &'static str,
    pub emoji: &'static str,
    pub schema: fn() -> Value,
    pub handler: fn(&Value) -> String,
}

/// All seven kanban tools, in registration order (matches the Python module).
pub fn tool_defs() -> Vec<KanbanToolDef> {
    vec![
        KanbanToolDef {
            name: "kanban_show",
            toolset: "kanban",
            emoji: "📋",
            schema: kanban_show_schema,
            handler: handle_show,
        },
        KanbanToolDef {
            name: "kanban_complete",
            toolset: "kanban",
            emoji: "✔",
            schema: kanban_complete_schema,
            handler: handle_complete,
        },
        KanbanToolDef {
            name: "kanban_block",
            toolset: "kanban",
            emoji: "⏸",
            schema: kanban_block_schema,
            handler: handle_block,
        },
        KanbanToolDef {
            name: "kanban_heartbeat",
            toolset: "kanban",
            emoji: "💓",
            schema: kanban_heartbeat_schema,
            handler: handle_heartbeat,
        },
        KanbanToolDef {
            name: "kanban_comment",
            toolset: "kanban",
            emoji: "💬",
            schema: kanban_comment_schema,
            handler: handle_comment,
        },
        KanbanToolDef {
            name: "kanban_create",
            toolset: "kanban",
            emoji: "➕",
            schema: kanban_create_schema,
            handler: handle_create,
        },
        KanbanToolDef {
            name: "kanban_link",
            toolset: "kanban",
            emoji: "🔗",
            schema: kanban_link_schema,
            handler: handle_link,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    // Env is process-global; serialise tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        unsafe {
            env::remove_var("HERMES_KANBAN_TASK");
            env::remove_var("HERMES_KANBAN_RUN_ID");
            env::remove_var("HERMES_PROFILE");
            env::remove_var("HERMES_TENANT");
        }
    }

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).expect("handler must return valid JSON")
    }

    #[test]
    fn ok_merges_fields() {
        let out = parse(&ok(json!({ "task_id": "t1", "run_id": 7 })));
        assert_eq!(out["ok"], json!(true));
        assert_eq!(out["task_id"], json!("t1"));
        assert_eq!(out["run_id"], json!(7));
    }

    #[test]
    fn tool_error_shape() {
        let out = parse(&tool_error("boom"));
        assert_eq!(out["error"], json!("boom"));
    }

    #[test]
    fn default_task_id_prefers_arg() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "env-task") };
        assert_eq!(default_task_id(Some("arg-task")).as_deref(), Some("arg-task"));
        // Empty arg falls through to env.
        assert_eq!(default_task_id(Some("")).as_deref(), Some("env-task"));
        assert_eq!(default_task_id(None).as_deref(), Some("env-task"));
        clear_env();
        assert_eq!(default_task_id(None), None);
    }

    #[test]
    fn worker_run_id_scoped() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var("HERMES_KANBAN_TASK", "t1");
            env::set_var("HERMES_KANBAN_RUN_ID", "42");
        }
        assert_eq!(worker_run_id("t1"), Some(42));
        // Foreign task → None even with run id set.
        assert_eq!(worker_run_id("t2"), None);
        // Non-numeric run id → None.
        unsafe { env::set_var("HERMES_KANBAN_RUN_ID", "abc") };
        assert_eq!(worker_run_id("t1"), None);
        clear_env();
    }

    #[test]
    fn ownership_blocks_foreign_task() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "mine") };
        assert!(enforce_worker_task_ownership("mine").is_none());
        let err = enforce_worker_task_ownership("theirs").expect("should reject");
        let v = parse(&err);
        assert!(v["error"].as_str().unwrap().contains("scoped to task mine"));
        // No env var → orchestrator context → always allowed.
        clear_env();
        assert!(enforce_worker_task_ownership("anything").is_none());
    }

    #[test]
    fn complete_requires_summary_or_result() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "t1") };
        let out = parse(&handle_complete(&json!({ "task_id": "t1" })));
        assert!(out["error"]
            .as_str()
            .unwrap()
            .contains("at least one of"));
        clear_env();
    }

    #[test]
    fn complete_rejects_bad_metadata() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "t1") };
        let out = parse(&handle_complete(
            &json!({ "task_id": "t1", "summary": "x", "metadata": "nope" }),
        ));
        assert!(out["error"].as_str().unwrap().contains("metadata must be an object"));
        assert!(out["error"].as_str().unwrap().contains("got str"));
        clear_env();
    }

    #[test]
    fn complete_rejects_bad_created_cards_type() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "t1") };
        let out = parse(&handle_complete(
            &json!({ "task_id": "t1", "summary": "x", "created_cards": 5 }),
        ));
        assert!(out["error"]
            .as_str()
            .unwrap()
            .contains("created_cards must be a list"));
        assert!(out["error"].as_str().unwrap().contains("got int"));
        clear_env();
    }

    #[test]
    fn complete_foreign_task_is_rejected_before_db() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "mine") };
        // Even with valid summary, a foreign task id is rejected by ownership.
        let out = parse(&handle_complete(
            &json!({ "task_id": "theirs", "summary": "done" }),
        ));
        assert!(out["error"].as_str().unwrap().contains("scoped to task mine"));
        clear_env();
    }

    #[test]
    fn block_requires_reason() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var("HERMES_KANBAN_TASK", "t1") };
        let out = parse(&handle_block(&json!({ "task_id": "t1", "reason": "   " })));
        assert!(out["error"].as_str().unwrap().contains("reason is required"));
        clear_env();
    }

    #[test]
    fn comment_requires_explicit_task_id() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        // task_id does NOT fall back to env for comment.
        unsafe { env::set_var("HERMES_KANBAN_TASK", "t1") };
        let out = parse(&handle_comment(&json!({ "body": "hi" })));
        assert!(out["error"].as_str().unwrap().contains("task_id is required"));
        clear_env();
    }

    #[test]
    fn comment_requires_body() {
        let out = parse(&handle_comment(&json!({ "task_id": "t1", "body": "  " })));
        assert!(out["error"].as_str().unwrap().contains("body is required"));
    }

    #[test]
    fn create_requires_title_and_assignee() {
        let out = parse(&handle_create(&json!({ "assignee": "writer" })));
        assert!(out["error"].as_str().unwrap().contains("title is required"));
        let out = parse(&handle_create(&json!({ "title": "Do it" })));
        assert!(out["error"].as_str().unwrap().contains("assignee is required"));
    }

    #[test]
    fn create_rejects_bad_skills_type() {
        let out = parse(&handle_create(
            &json!({ "title": "t", "assignee": "a", "skills": 3 }),
        ));
        assert!(out["error"].as_str().unwrap().contains("skills must be a list"));
        assert!(out["error"].as_str().unwrap().contains("got int"));
    }

    #[test]
    fn create_rejects_bad_parents_type() {
        let out = parse(&handle_create(
            &json!({ "title": "t", "assignee": "a", "parents": 3 }),
        ));
        assert!(out["error"].as_str().unwrap().contains("parents must be a list"));
    }

    #[test]
    fn link_requires_both_ids() {
        let out = parse(&handle_link(&json!({ "parent_id": "p" })));
        assert!(out["error"]
            .as_str()
            .unwrap()
            .contains("both parent_id and child_id are required"));
    }

    #[test]
    fn show_requires_task_id_without_env() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let out = parse(&handle_show(&json!({})));
        assert!(out["error"].as_str().unwrap().contains("task_id is required"));
        clear_env();
    }

    #[test]
    fn value_coercion_helpers() {
        assert_eq!(value_to_int(&json!(5)), Some(5));
        assert_eq!(value_to_int(&json!("12")), Some(12));
        assert_eq!(value_to_int(&json!(3.9)), Some(3));
        assert_eq!(value_to_int(&json!("nope")), None);
        assert!(value_truthy(&json!(true)));
        assert!(!value_truthy(&json!(false)));
        assert!(!value_truthy(&json!(0)));
        assert!(value_truthy(&json!("x")));
        assert_eq!(py_type_name(&json!(1)), "int");
        assert_eq!(py_type_name(&json!(1.5)), "float");
        assert_eq!(py_type_name(&json!("s")), "str");
        assert_eq!(py_type_name(&json!([])), "list");
        assert_eq!(py_type_name(&json!({})), "dict");
    }

    #[test]
    fn tool_defs_cover_seven_tools() {
        let defs = tool_defs();
        let names: Vec<&str> = defs.iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            vec![
                "kanban_show",
                "kanban_complete",
                "kanban_block",
                "kanban_heartbeat",
                "kanban_comment",
                "kanban_create",
                "kanban_link",
            ]
        );
        for d in &defs {
            assert_eq!(d.toolset, "kanban");
            // Schema must produce a name matching the def name.
            let schema = (d.schema)();
            assert_eq!(schema["name"], json!(d.name));
        }
    }
}
