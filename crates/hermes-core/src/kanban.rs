use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use rusqlite::{
    Connection, OptionalExtension, Row, Transaction, TransactionBehavior, params, params_from_iter,
};
use serde_json::{Map, Value, json};
use serde_yaml::Value as YamlValue;

use crate::{
    HermesContext, LoadedConfig,
    tools::{ToolRuntime, tool_error, tool_result},
};

const DEFAULT_BOARD: &str = "default";
pub const VALID_KANBAN_STATUSES: &[&str] = &[
    "triage", "todo", "ready", "running", "blocked", "done", "archived",
];
pub const VALID_WORKSPACE_KINDS: &[&str] = &["scratch", "dir", "worktree"];
const CONTEXT_MAX_PRIOR_RUNS: usize = 10;
const CONTEXT_MAX_COMMENTS: usize = 30;
const CONTEXT_MAX_FIELD_CHARS: usize = 4096;
const CONTEXT_MAX_BODY_CHARS: usize = 8192;
const DEFAULT_CLAIM_TTL_SECONDS: i64 = 15 * 60;
const DEFAULT_FAILURE_LIMIT: i64 = 5;
const DEFAULT_LOG_ROTATE_BYTES: u64 = 2 * 1024 * 1024;
const TERM_GRACE_POLL_COUNT: u32 = 10;
const TERM_GRACE_POLL_MILLIS: u64 = 500;
static TASK_ID_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn kanban_available() -> bool {
    if env::var("HERMES_KANBAN_TASK")
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return true;
    }
    kanban_enabled_in_config(&default_hermes_home())
}

pub fn kanban_show_schema() -> Value {
    json!({
        "name": "kanban_show",
        "description": "Read a kanban task's state, including its body, parent and child links, comments, events, runs, and a prebuilt worker context string.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the environment."
                }
            },
            "required": []
        }
    })
}

pub fn kanban_complete_schema() -> Value {
    json!({
        "name": "kanban_complete",
        "description": "Mark a kanban task done with a structured handoff. At least one of summary or result is required.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the environment."
                },
                "summary": {
                    "type": "string",
                    "description": "Human-readable handoff summary."
                },
                "metadata": {
                    "type": "object",
                    "description": "Structured machine-readable handoff data.",
                    "properties": {},
                    "additionalProperties": true
                },
                "result": {
                    "type": "string",
                    "description": "Legacy result string stored on the task row."
                },
                "created_cards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of task ids created by this worker during the run."
                }
            },
            "required": []
        }
    })
}

pub fn kanban_block_schema() -> Value {
    json!({
        "name": "kanban_block",
        "description": "Mark a kanban task blocked and record the human-facing reason.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the environment."
                },
                "reason": {
                    "type": "string",
                    "description": "Why the task is blocked."
                }
            },
            "required": ["reason"]
        }
    })
}

pub fn kanban_heartbeat_schema() -> Value {
    json!({
        "name": "kanban_heartbeat",
        "description": "Record a liveness heartbeat for a running kanban worker.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the environment."
                },
                "note": {
                    "type": "string",
                    "description": "Optional short progress note."
                }
            },
            "required": []
        }
    })
}

pub fn kanban_comment_schema() -> Value {
    json!({
        "name": "kanban_comment",
        "description": "Append a durable comment to a kanban task thread.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id to comment on."
                },
                "body": {
                    "type": "string",
                    "description": "Comment body."
                },
                "author": {
                    "type": "string",
                    "description": "Optional author override. Defaults to HERMES_PROFILE or worker."
                }
            },
            "required": ["task_id", "body"]
        }
    })
}

pub fn kanban_create_schema() -> Value {
    json!({
        "name": "kanban_create",
        "description": "Create a kanban task, optionally linked under parent tasks and pinned to a specific assignee profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Task title."
                },
                "assignee": {
                    "type": "string",
                    "description": "Optional profile that should execute the task."
                },
                "body": {
                    "type": "string",
                    "description": "Optional task body."
                },
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional parent task ids."
                },
                "tenant": {
                    "type": "string",
                    "description": "Optional tenant or namespace."
                },
                "priority": {
                    "type": "integer",
                    "description": "Optional priority."
                },
                "workspace_kind": {
                    "type": "string",
                    "enum": ["scratch", "dir", "worktree"],
                    "description": "Workspace mode."
                },
                "workspace_path": {
                    "type": "string",
                    "description": "Optional workspace path."
                },
                "triage": {
                    "type": "boolean",
                    "description": "If true, create the task in triage."
                },
                "idempotency_key": {
                    "type": "string",
                    "description": "Optional key used to dedupe retried creates."
                },
                "max_runtime_seconds": {
                    "type": "integer",
                    "description": "Optional runtime limit."
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional extra skill names to force-load for the worker."
                }
            },
            "required": ["title"]
        }
    })
}

pub fn kanban_link_schema() -> Value {
    json!({
        "name": "kanban_link",
        "description": "Link an existing parent task to an existing child task.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "Parent task id."
                },
                "child_id": {
                    "type": "string",
                    "description": "Child task id."
                }
            },
            "required": ["parent_id", "child_id"]
        }
    })
}

pub fn handle_kanban_show(args: &Value, runtime: &ToolRuntime) -> String {
    let task_id = match default_task_id(optional_non_empty_string(args, "task_id")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    let conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let task = match get_task(&conn, &task_id) {
        Ok(Some(value)) => value,
        Ok(None) => return tool_error(format!("task {task_id} not found")),
        Err(error) => return tool_error(error),
    };
    let comments = match list_comments(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let events = match list_events(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let runs = match list_runs(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let parents = match parent_ids(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let children = match child_ids(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let worker_context = match build_worker_context(&conn, &task_id) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    tool_result(json!({
        "task": task.to_json(),
        "parents": parents,
        "children": children,
        "comments": comments.into_iter().map(Comment::to_json).collect::<Vec<_>>(),
        "events": events.into_iter().rev().take(50).collect::<Vec<_>>().into_iter().rev().map(Event::to_json).collect::<Vec<_>>(),
        "runs": runs.into_iter().map(Run::to_json).collect::<Vec<_>>(),
        "worker_context": worker_context,
    }))
}

pub fn handle_kanban_complete(args: &Value, runtime: &ToolRuntime) -> String {
    let task_id = match default_task_id(optional_non_empty_string(args, "task_id")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(error) = enforce_worker_task_ownership(&task_id) {
        return error;
    }
    let summary = optional_string(args, "summary");
    let result = optional_string(args, "result");
    if summary.is_none() && result.is_none() {
        return tool_error("provide at least one of: summary, result");
    }
    let metadata = match optional_object(args, "metadata") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let created_cards = match optional_string_list(args, "created_cards") {
        Ok(value) => value.unwrap_or_default(),
        Err(error) => return tool_error(error),
    };
    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match complete_task(
        &mut conn,
        &task_id,
        result.as_deref(),
        summary.as_deref(),
        metadata.as_ref(),
        &created_cards,
        worker_run_id(&task_id),
    ) {
        Ok(true) => {
            let run_id = latest_run(&conn, &task_id).ok().flatten().map(|run| run.id);
            tool_result(json!({
                "ok": true,
                "task_id": task_id,
                "run_id": run_id,
            }))
        }
        Ok(false) => tool_error(format!(
            "could not complete {task_id} (unknown id or already terminal)"
        )),
        Err(error) => tool_error(error),
    }
}

pub fn handle_kanban_block(args: &Value, runtime: &ToolRuntime) -> String {
    let task_id = match default_task_id(optional_non_empty_string(args, "task_id")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(error) = enforce_worker_task_ownership(&task_id) {
        return error;
    }
    let reason = match required_non_empty_string(args, "reason") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match block_task(&mut conn, &task_id, &reason, worker_run_id(&task_id)) {
        Ok(true) => {
            let run_id = latest_run(&conn, &task_id).ok().flatten().map(|run| run.id);
            tool_result(json!({
                "ok": true,
                "task_id": task_id,
                "run_id": run_id,
            }))
        }
        Ok(false) => tool_error(format!(
            "could not block {task_id} (unknown id or not in running or ready)"
        )),
        Err(error) => tool_error(error),
    }
}

pub fn handle_kanban_heartbeat(args: &Value, runtime: &ToolRuntime) -> String {
    let task_id = match default_task_id(optional_non_empty_string(args, "task_id")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(error) = enforce_worker_task_ownership(&task_id) {
        return error;
    }
    let note = optional_string(args, "note");
    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match heartbeat_worker(
        &mut conn,
        &task_id,
        note.as_deref(),
        worker_run_id(&task_id),
    ) {
        Ok(true) => tool_result(json!({
            "ok": true,
            "task_id": task_id,
        })),
        Ok(false) => tool_error(format!(
            "could not heartbeat {task_id} (unknown id or not running)"
        )),
        Err(error) => tool_error(error),
    }
}

pub fn handle_kanban_comment(args: &Value, runtime: &ToolRuntime) -> String {
    let task_id = match required_non_empty_string(args, "task_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let body = match required_non_empty_string(args, "body") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let author = optional_string(args, "author")
        .or_else(|| env::var("HERMES_PROFILE").ok().and_then(non_empty_trimmed))
        .unwrap_or_else(|| "worker".to_string());
    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match add_comment(&mut conn, &task_id, &author, &body) {
        Ok(comment_id) => tool_result(json!({
            "ok": true,
            "task_id": task_id,
            "comment_id": comment_id,
        })),
        Err(error) => tool_error(error),
    }
}

pub fn handle_kanban_create(args: &Value, runtime: &ToolRuntime) -> String {
    let title = match required_non_empty_string(args, "title") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let assignee = optional_non_empty_string(args, "assignee");
    let body = optional_string(args, "body");
    let parents = match optional_string_list(args, "parents") {
        Ok(value) => value.unwrap_or_default(),
        Err(error) => return tool_error(error),
    };
    let tenant = optional_string(args, "tenant");
    let priority = match optional_i64(args, "priority") {
        Ok(value) => value.unwrap_or(0),
        Err(error) => return tool_error(error),
    };
    let workspace_kind =
        optional_string(args, "workspace_kind").unwrap_or_else(|| "scratch".to_string());
    if !VALID_WORKSPACE_KINDS.contains(&workspace_kind.as_str()) {
        return tool_error(format!(
            "workspace_kind must be one of {}",
            VALID_WORKSPACE_KINDS.join(", ")
        ));
    }
    let workspace_path = optional_string(args, "workspace_path");
    let triage = match optional_bool(args, "triage") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return tool_error(error),
    };
    let idempotency_key = optional_string(args, "idempotency_key");
    let max_runtime_seconds = match optional_i64(args, "max_runtime_seconds") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if max_runtime_seconds.is_some_and(|value| value <= 0) {
        return tool_error("max_runtime_seconds must be positive");
    }
    let skills = match optional_string_list(args, "skills") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let created_by = env::var("HERMES_PROFILE")
        .ok()
        .and_then(non_empty_trimmed)
        .unwrap_or_else(|| "worker".to_string());

    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match create_task(
        &mut conn,
        CreateTaskInput {
            title,
            body,
            assignee,
            parents,
            tenant,
            priority,
            workspace_kind,
            workspace_path,
            triage,
            idempotency_key,
            max_runtime_seconds,
            skills,
            created_by,
        },
    ) {
        Ok(task_id) => {
            let status = get_task(&conn, &task_id)
                .ok()
                .flatten()
                .map(|task| task.status)
                .unwrap_or_default();
            tool_result(json!({
                "ok": true,
                "task_id": task_id,
                "status": status,
            }))
        }
        Err(error) => tool_error(error),
    }
}

pub fn handle_kanban_link(args: &Value, runtime: &ToolRuntime) -> String {
    let parent_id = match required_non_empty_string(args, "parent_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let child_id = match required_non_empty_string(args, "child_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mut conn = match connect_kanban(runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    match link_tasks(&mut conn, &parent_id, &child_id) {
        Ok(()) => tool_result(json!({
            "ok": true,
            "parent_id": parent_id,
            "child_id": child_id,
        })),
        Err(error) => tool_error(error),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KanbanDispatchOptions {
    pub dry_run: bool,
    pub max_spawn: Option<usize>,
    pub failure_limit: Option<i64>,
}

impl Default for KanbanDispatchOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            max_spawn: None,
            failure_limit: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KanbanSpawnRecord {
    pub task_id: String,
    pub assignee: String,
    pub workspace_path: String,
    pub pid: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KanbanDispatchResult {
    pub reclaimed: usize,
    pub promoted: usize,
    pub spawned: Vec<KanbanSpawnRecord>,
    pub skipped_unassigned: Vec<String>,
    pub skipped_nonspawnable: Vec<String>,
    pub crashed: Vec<String>,
    pub timed_out: Vec<String>,
    pub auto_blocked: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KanbanRunResult {
    pub task_id: String,
    pub assignee: String,
    pub workspace_path: String,
    pub pid: Option<u32>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub body: Option<String>,
    pub assignee: Option<String>,
    pub status: String,
    pub priority: i64,
    pub tenant: Option<String>,
    pub workspace_kind: String,
    pub workspace_path: Option<String>,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub result: Option<String>,
    pub current_run_id: Option<i64>,
    pub max_runtime_seconds: Option<i64>,
    pub last_heartbeat_at: Option<i64>,
    pub consecutive_failures: i64,
    pub last_failure_error: Option<String>,
    pub skills: Vec<String>,
}

impl Task {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        let skills = row
            .get::<_, Option<String>>("skills")?
            .and_then(|value| serde_json::from_str::<Vec<String>>(&value).ok())
            .unwrap_or_default();
        Ok(Self {
            id: row.get("id")?,
            title: row.get("title")?,
            body: row.get("body")?,
            assignee: row.get("assignee")?,
            status: row.get("status")?,
            priority: row.get("priority")?,
            tenant: row.get("tenant")?,
            workspace_kind: row.get("workspace_kind")?,
            workspace_path: row.get("workspace_path")?,
            created_by: row.get("created_by")?,
            created_at: row.get("created_at")?,
            started_at: row.get("started_at")?,
            completed_at: row.get("completed_at")?,
            result: row.get("result")?,
            current_run_id: row.get("current_run_id")?,
            max_runtime_seconds: row.get("max_runtime_seconds")?,
            last_heartbeat_at: row.get("last_heartbeat_at")?,
            consecutive_failures: row.get("consecutive_failures")?,
            last_failure_error: row.get("last_failure_error")?,
            skills,
        })
    }

    fn to_json(self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "assignee": self.assignee,
            "status": self.status,
            "tenant": self.tenant,
            "priority": self.priority,
            "workspace_kind": self.workspace_kind,
            "workspace_path": self.workspace_path,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "current_run_id": self.current_run_id,
            "max_runtime_seconds": self.max_runtime_seconds,
            "last_heartbeat_at": self.last_heartbeat_at,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_error": self.last_failure_error,
            "skills": self.skills,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Run {
    pub id: i64,
    pub task_id: String,
    pub profile: Option<String>,
    pub status: String,
    pub outcome: Option<String>,
    pub summary: Option<String>,
    pub metadata: Option<Value>,
    pub error: Option<String>,
    pub started_at: i64,
    pub ended_at: Option<i64>,
}

impl Run {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        let metadata = row
            .get::<_, Option<String>>("metadata")?
            .and_then(|value| serde_json::from_str::<Value>(&value).ok());
        Ok(Self {
            id: row.get("id")?,
            task_id: row.get("task_id")?,
            profile: row.get("profile")?,
            status: row.get("status")?,
            outcome: row.get("outcome")?,
            summary: row.get("summary")?,
            metadata,
            error: row.get("error")?,
            started_at: row.get("started_at")?,
            ended_at: row.get("ended_at")?,
        })
    }

    fn to_json(self) -> Value {
        json!({
            "id": self.id,
            "task_id": self.task_id,
            "profile": self.profile,
            "status": self.status,
            "outcome": self.outcome,
            "summary": self.summary,
            "metadata": self.metadata,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Comment {
    pub id: i64,
    pub task_id: String,
    pub author: String,
    pub body: String,
    pub created_at: i64,
}

impl Comment {
    fn to_json(self) -> Value {
        json!({
            "id": self.id,
            "task_id": self.task_id,
            "author": self.author,
            "body": self.body,
            "created_at": self.created_at,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Event {
    pub id: i64,
    pub task_id: String,
    pub kind: String,
    pub payload: Option<Value>,
    pub created_at: i64,
    pub run_id: Option<i64>,
}

impl Event {
    fn to_json(self) -> Value {
        json!({
            "id": self.id,
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "created_at": self.created_at,
            "run_id": self.run_id,
        })
    }
}

pub struct CreateTaskInput {
    pub title: String,
    pub body: Option<String>,
    pub assignee: Option<String>,
    pub parents: Vec<String>,
    pub tenant: Option<String>,
    pub priority: i64,
    pub workspace_kind: String,
    pub workspace_path: Option<String>,
    pub triage: bool,
    pub idempotency_key: Option<String>,
    pub max_runtime_seconds: Option<i64>,
    pub skills: Option<Vec<String>>,
    pub created_by: String,
}

#[derive(Debug, Clone, Default)]
pub struct KanbanTaskQuery {
    pub assignee: Option<String>,
    pub status: Option<String>,
    pub tenant: Option<String>,
    pub include_archived: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KanbanTaskDetail {
    pub task: Task,
    pub parents: Vec<String>,
    pub children: Vec<String>,
    pub comments: Vec<Comment>,
    pub events: Vec<Event>,
    pub runs: Vec<Run>,
    pub latest_summary: Option<String>,
    pub worker_context: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KanbanBoardRecord {
    pub slug: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub color: String,
    pub created_at: Option<i64>,
    pub archived: bool,
    pub db_path: String,
    pub current: bool,
    pub task_count: i64,
    pub ready_count: i64,
    pub blocked_count: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KanbanBoardRemoval {
    pub slug: String,
    pub action: String,
    pub old_path: String,
    pub new_path: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KanbanNotifySubscription {
    pub task_id: String,
    pub platform: String,
    pub chat_id: String,
    pub thread_id: String,
    pub user_id: Option<String>,
    pub created_at: i64,
    pub last_event_id: i64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct KanbanBoardStats {
    pub by_status: BTreeMap<String, i64>,
    pub by_assignee: BTreeMap<String, BTreeMap<String, i64>>,
    pub oldest_ready_age_seconds: Option<i64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KanbanAssigneeRecord {
    pub name: String,
    pub on_disk: bool,
    pub counts: BTreeMap<String, i64>,
}

fn connect_kanban(runtime: &ToolRuntime) -> Result<Connection, String> {
    connect_kanban_path(runtime.hermes_home())
}

fn connect_kanban_path(hermes_home: &Path) -> Result<Connection, String> {
    let path = kanban_db_path(hermes_home)?;
    open_kanban_db_at_path(&path)
}

fn open_kanban_db_at_path(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("creating {} failed: {error}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .map_err(|error| format!("opening {} failed: {error}", path.display()))?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
    )
    .map_err(|error| format!("configuring kanban database failed: {error}"))?;
    init_kanban_db(&conn)?;
    Ok(conn)
}

pub fn open_kanban_db(hermes_home: &Path) -> Result<Connection, String> {
    connect_kanban_path(hermes_home)
}

pub fn kanban_db_path_for_home(hermes_home: &Path) -> Result<PathBuf, String> {
    kanban_db_path(hermes_home)
}

pub fn dispatch_kanban_once(
    context: &HermesContext,
    loaded: &LoadedConfig,
    options: KanbanDispatchOptions,
) -> Result<KanbanDispatchResult, String> {
    dispatch_kanban_once_with_spawn(context, loaded, options, None)
}

pub fn kanban_has_spawnable_ready(context: &HermesContext) -> Result<bool, String> {
    let conn = connect_kanban_path(&context.hermes_home())?;
    has_spawnable_ready(&conn, context)
}

pub fn run_kanban_task(
    context: &HermesContext,
    loaded: &LoadedConfig,
    task_id: &str,
    dry_run: bool,
) -> Result<KanbanRunResult, String> {
    let task_id =
        non_empty_trimmed(task_id.to_string()).ok_or_else(|| "task_id is required".to_string())?;
    let mut conn = connect_kanban_path(&context.hermes_home())?;
    let failure_limit = kanban_failure_limit(loaded);
    run_specific_task(
        &mut conn,
        context,
        &task_id,
        kanban_claim_ttl_seconds(loaded),
        dry_run,
        failure_limit,
        None,
    )
}

pub fn current_kanban_board(hermes_home: &Path) -> Result<String, String> {
    Ok(current_board_slug(hermes_home)?.unwrap_or_else(|| DEFAULT_BOARD.to_string()))
}

pub fn set_current_kanban_board(hermes_home: &Path, board: &str) -> Result<PathBuf, String> {
    let slug =
        normalize_board_slug(Some(board))?.ok_or_else(|| "board slug is required".to_string())?;
    let path = kanban_root(hermes_home).join("kanban").join("current");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("creating {} failed: {error}", parent.display()))?;
    }
    fs::write(&path, format!("{slug}\n"))
        .map_err(|error| format!("writing {} failed: {error}", path.display()))?;
    Ok(path)
}

pub fn clear_current_kanban_board(hermes_home: &Path) -> Result<(), String> {
    let path = kanban_root(hermes_home).join("kanban").join("current");
    let _ = fs::remove_file(path);
    Ok(())
}

fn board_dir_for_slug(hermes_home: &Path, slug: &str) -> PathBuf {
    kanban_root(hermes_home)
        .join("kanban")
        .join("boards")
        .join(slug)
}

fn board_db_path_from_slug(hermes_home: &Path, slug: &str) -> PathBuf {
    if slug == DEFAULT_BOARD {
        kanban_root(hermes_home).join("kanban.db")
    } else {
        board_dir_for_slug(hermes_home, slug).join("kanban.db")
    }
}

fn board_metadata_path_for_slug(hermes_home: &Path, slug: &str) -> PathBuf {
    board_dir_for_slug(hermes_home, slug).join("board.json")
}

fn default_board_name(slug: &str) -> String {
    let mut parts = Vec::new();
    for part in slug.replace('_', "-").split('-') {
        if part.is_empty() {
            continue;
        }
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            continue;
        };
        let mut word = first.to_ascii_uppercase().to_string();
        word.push_str(chars.as_str());
        parts.push(word);
    }
    if parts.is_empty() {
        slug.to_string()
    } else {
        parts.join(" ")
    }
}

fn board_task_counts(conn: &Connection) -> Result<(i64, i64, i64), String> {
    conn.query_row(
        "SELECT COUNT(*),
                SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END)
           FROM tasks
          WHERE status != 'archived'",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                row.get::<_, Option<i64>>(2)?.unwrap_or(0),
            ))
        },
    )
    .map_err(|error| format!("reading board task counts failed: {error}"))
}

fn read_board_metadata_for_slug(
    hermes_home: &Path,
    slug: &str,
) -> Result<KanbanBoardRecord, String> {
    let mut record = KanbanBoardRecord {
        slug: slug.to_string(),
        name: default_board_name(slug),
        description: String::new(),
        icon: String::new(),
        color: String::new(),
        created_at: None,
        archived: false,
        db_path: board_db_path_from_slug(hermes_home, slug)
            .display()
            .to_string(),
        current: current_kanban_board(hermes_home).ok().as_deref() == Some(slug),
        task_count: 0,
        ready_count: 0,
        blocked_count: 0,
    };
    let path = board_metadata_path_for_slug(hermes_home, slug);
    if let Ok(contents) = fs::read_to_string(&path)
        && let Ok(value) = serde_json::from_str::<Value>(&contents)
        && let Some(object) = value.as_object()
    {
        record.name = object
            .get("name")
            .and_then(Value::as_str)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| default_board_name(slug));
        record.description = object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        record.icon = object
            .get("icon")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        record.color = object
            .get("color")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        record.created_at = object.get("created_at").and_then(Value::as_i64);
        record.archived = object
            .get("archived")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    }
    let db_path = board_db_path_from_slug(hermes_home, slug);
    if db_path.exists()
        && let Ok(conn) = open_kanban_db_at_path(&db_path)
        && let Ok((task_count, ready_count, blocked_count)) = board_task_counts(&conn)
    {
        record.task_count = task_count;
        record.ready_count = ready_count;
        record.blocked_count = blocked_count;
    }
    Ok(record)
}

fn write_board_metadata_for_slug(
    hermes_home: &Path,
    slug: &str,
    name: Option<&str>,
    description: Option<&str>,
    icon: Option<&str>,
    color: Option<&str>,
    archived: Option<bool>,
) -> Result<KanbanBoardRecord, String> {
    let mut record = read_board_metadata_for_slug(hermes_home, slug)?;
    if let Some(value) = name {
        let trimmed = value.trim();
        record.name = if trimmed.is_empty() {
            default_board_name(slug)
        } else {
            trimmed.to_string()
        };
    }
    if let Some(value) = description {
        record.description = value.to_string();
    }
    if let Some(value) = icon {
        record.icon = value.to_string();
    }
    if let Some(value) = color {
        record.color = value.to_string();
    }
    if let Some(value) = archived {
        record.archived = value;
    }
    if record.created_at.is_none() {
        record.created_at = Some(now_ts());
    }
    let path = board_metadata_path_for_slug(hermes_home, slug);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("creating {} failed: {error}", parent.display()))?;
    }
    fs::write(
        &path,
        serde_json::to_string_pretty(&json!({
            "slug": record.slug,
            "name": record.name,
            "description": record.description,
            "icon": record.icon,
            "color": record.color,
            "created_at": record.created_at,
            "archived": record.archived,
        }))
        .map_err(|error| format!("serializing board metadata failed: {error}"))?,
    )
    .map_err(|error| format!("writing {} failed: {error}", path.display()))?;
    read_board_metadata_for_slug(hermes_home, slug)
}

pub fn kanban_board_exists(hermes_home: &Path, board: &str) -> Result<bool, String> {
    let slug =
        normalize_board_slug(Some(board))?.ok_or_else(|| "board slug is required".to_string())?;
    if slug == DEFAULT_BOARD {
        return Ok(true);
    }
    let dir = board_dir_for_slug(hermes_home, &slug);
    Ok(dir.is_dir() || dir.join("kanban.db").exists())
}

pub fn create_kanban_board(
    hermes_home: &Path,
    board: &str,
    name: Option<&str>,
    description: Option<&str>,
    icon: Option<&str>,
    color: Option<&str>,
) -> Result<KanbanBoardRecord, String> {
    let slug =
        normalize_board_slug(Some(board))?.ok_or_else(|| "board slug is required".to_string())?;
    fs::create_dir_all(board_dir_for_slug(hermes_home, &slug))
        .map_err(|error| format!("creating board {slug} failed: {error}"))?;
    open_kanban_db_at_path(&board_db_path_from_slug(hermes_home, &slug))?;
    write_board_metadata_for_slug(
        hermes_home,
        &slug,
        name,
        description,
        icon,
        color,
        Some(false),
    )
}

pub fn list_kanban_boards(
    hermes_home: &Path,
    include_archived: bool,
) -> Result<Vec<KanbanBoardRecord>, String> {
    let mut slugs = HashSet::new();
    slugs.insert(DEFAULT_BOARD.to_string());
    let root = kanban_root(hermes_home).join("kanban").join("boards");
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name == "_archived" {
                continue;
            }
            if let Some(slug) = normalize_board_slug(Some(name))? {
                slugs.insert(slug);
            }
        }
    }
    let mut boards = Vec::new();
    for slug in slugs {
        let board = read_board_metadata_for_slug(hermes_home, &slug)?;
        if include_archived || !board.archived {
            boards.push(board);
        }
    }
    boards.sort_by(|left, right| {
        (left.slug != DEFAULT_BOARD, left.slug.as_str())
            .cmp(&(right.slug != DEFAULT_BOARD, right.slug.as_str()))
    });
    Ok(boards)
}

pub fn remove_kanban_board(
    hermes_home: &Path,
    board: &str,
    archive: bool,
) -> Result<KanbanBoardRemoval, String> {
    let slug =
        normalize_board_slug(Some(board))?.ok_or_else(|| "board slug is required".to_string())?;
    if slug == DEFAULT_BOARD {
        return Err("default board cannot be removed".to_string());
    }
    let dir = board_dir_for_slug(hermes_home, &slug);
    if !dir.exists() {
        return Err(format!("board {slug:?} does not exist"));
    }
    let old_path = dir.display().to_string();
    let result = if archive {
        let archived_root = kanban_root(hermes_home)
            .join("kanban")
            .join("boards")
            .join("_archived");
        fs::create_dir_all(&archived_root)
            .map_err(|error| format!("creating {} failed: {error}", archived_root.display()))?;
        let target = archived_root.join(format!("{slug}-{}", now_ts()));
        fs::rename(&dir, &target)
            .map_err(|error| format!("archiving board {slug} failed: {error}"))?;
        KanbanBoardRemoval {
            slug: slug.clone(),
            action: String::from("archived"),
            old_path,
            new_path: Some(target.display().to_string()),
        }
    } else {
        fs::remove_dir_all(&dir)
            .map_err(|error| format!("deleting board {slug} failed: {error}"))?;
        KanbanBoardRemoval {
            slug: slug.clone(),
            action: String::from("deleted"),
            old_path,
            new_path: None,
        }
    };
    if current_kanban_board(hermes_home).ok().as_deref() == Some(slug.as_str()) {
        clear_current_kanban_board(hermes_home)?;
    }
    Ok(result)
}

pub fn rename_kanban_board(
    hermes_home: &Path,
    board: &str,
    name: &str,
) -> Result<KanbanBoardRecord, String> {
    let slug =
        normalize_board_slug(Some(board))?.ok_or_else(|| "board slug is required".to_string())?;
    if !kanban_board_exists(hermes_home, &slug)? {
        return Err(format!("board {slug:?} does not exist"));
    }
    write_board_metadata_for_slug(hermes_home, &slug, Some(name), None, None, None, None)
}

fn init_kanban_db(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(SCHEMA_SQL)
        .map_err(|error| format!("initializing kanban schema failed: {error}"))?;
    migrate_optional_columns(conn)
}

fn migrate_optional_columns(conn: &Connection) -> Result<(), String> {
    let task_cols = table_columns(conn, "tasks")?;
    for (name, ddl) in [
        ("tenant", "ALTER TABLE tasks ADD COLUMN tenant TEXT"),
        ("result", "ALTER TABLE tasks ADD COLUMN result TEXT"),
        (
            "idempotency_key",
            "ALTER TABLE tasks ADD COLUMN idempotency_key TEXT",
        ),
        (
            "consecutive_failures",
            "ALTER TABLE tasks ADD COLUMN consecutive_failures INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "worker_pid",
            "ALTER TABLE tasks ADD COLUMN worker_pid INTEGER",
        ),
        (
            "last_failure_error",
            "ALTER TABLE tasks ADD COLUMN last_failure_error TEXT",
        ),
        (
            "max_runtime_seconds",
            "ALTER TABLE tasks ADD COLUMN max_runtime_seconds INTEGER",
        ),
        (
            "last_heartbeat_at",
            "ALTER TABLE tasks ADD COLUMN last_heartbeat_at INTEGER",
        ),
        (
            "current_run_id",
            "ALTER TABLE tasks ADD COLUMN current_run_id INTEGER",
        ),
        (
            "workflow_template_id",
            "ALTER TABLE tasks ADD COLUMN workflow_template_id TEXT",
        ),
        (
            "current_step_key",
            "ALTER TABLE tasks ADD COLUMN current_step_key TEXT",
        ),
        ("skills", "ALTER TABLE tasks ADD COLUMN skills TEXT"),
    ] {
        if !task_cols.contains(name) {
            conn.execute_batch(ddl)
                .map_err(|error| format!("migrating tasks.{name} failed: {error}"))?;
        }
    }

    let event_cols = table_columns(conn, "task_events")?;
    if !event_cols.contains("run_id") {
        conn.execute_batch("ALTER TABLE task_events ADD COLUMN run_id INTEGER")
            .map_err(|error| format!("migrating task_events.run_id failed: {error}"))?;
    }

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
         CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
         CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant);
         CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key);
         CREATE INDEX IF NOT EXISTS idx_links_child ON task_links(child_id);
         CREATE INDEX IF NOT EXISTS idx_links_parent ON task_links(parent_id);
         CREATE INDEX IF NOT EXISTS idx_comments_task ON task_comments(task_id, created_at);
         CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id, created_at);
         CREATE INDEX IF NOT EXISTS idx_events_run ON task_events(run_id, id);
         CREATE INDEX IF NOT EXISTS idx_runs_task ON task_runs(task_id, started_at);
         CREATE INDEX IF NOT EXISTS idx_runs_status ON task_runs(status);",
    )
    .map_err(|error| format!("creating kanban indexes failed: {error}"))?;

    Ok(())
}

fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|error| format!("reading {table} columns failed: {error}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("reading {table} column names failed: {error}"))?;
    let mut columns = HashSet::new();
    for row in rows {
        columns.insert(row.map_err(|error| format!("reading {table} column failed: {error}"))?);
    }
    Ok(columns)
}

type KanbanSpawnFn = dyn Fn(&Task, &Path) -> Result<Option<u32>, String>;

fn dispatch_kanban_once_with_spawn(
    context: &HermesContext,
    loaded: &LoadedConfig,
    options: KanbanDispatchOptions,
    spawn_fn: Option<&KanbanSpawnFn>,
) -> Result<KanbanDispatchResult, String> {
    let mut conn = connect_kanban_path(&context.hermes_home())?;
    let ttl_seconds = kanban_claim_ttl_seconds(loaded);
    let failure_limit = options
        .failure_limit
        .filter(|value| *value > 0)
        .unwrap_or_else(|| kanban_failure_limit(loaded));

    let mut result = KanbanDispatchResult::default();
    result.reclaimed = release_stale_claims(&mut conn)?;
    result.crashed = detect_crashed_workers(&mut conn, failure_limit)?;
    result.timed_out = enforce_max_runtime(&mut conn, failure_limit)?;
    result.promoted = recompute_ready(&mut conn)?;

    let ready_tasks = {
        let mut stmt = conn
            .prepare(
                "SELECT * FROM tasks
                 WHERE status = 'ready' AND claim_lock IS NULL
                 ORDER BY priority DESC, created_at ASC",
            )
            .map_err(|error| format!("preparing ready-task query failed: {error}"))?;
        let rows = stmt
            .query_map([], Task::from_row)
            .map_err(|error| format!("querying ready tasks failed: {error}"))?;
        collect_rows(rows, "reading ready task")?
    };

    let mut spawned = 0_usize;
    for task in ready_tasks {
        if options.max_spawn.is_some_and(|limit| spawned >= limit) {
            break;
        }

        let Some(assignee) = task.assignee.as_deref() else {
            result.skipped_unassigned.push(task.id.clone());
            continue;
        };
        if !context.profile_exists(assignee) {
            result.skipped_nonspawnable.push(task.id.clone());
            continue;
        }

        if options.dry_run {
            let workspace = resolve_workspace(&task, &context.hermes_home())?;
            result.spawned.push(KanbanSpawnRecord {
                task_id: task.id.clone(),
                assignee: assignee.to_string(),
                workspace_path: workspace.display().to_string(),
                pid: None,
            });
            spawned += 1;
            continue;
        }

        let claimed = match claim_task(&mut conn, &task.id, ttl_seconds, None)? {
            Some(value) => value,
            None => continue,
        };
        let run = match run_claimed_task(&mut conn, context, &claimed, failure_limit, spawn_fn) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        if run.auto_blocked {
            result.auto_blocked.push(claimed.id.clone());
        }
        if let Some(record) = run.record {
            result.spawned.push(record);
            spawned += 1;
        }
    }

    Ok(result)
}

fn run_specific_task(
    conn: &mut Connection,
    context: &HermesContext,
    task_id: &str,
    ttl_seconds: i64,
    dry_run: bool,
    failure_limit: i64,
    spawn_fn: Option<&KanbanSpawnFn>,
) -> Result<KanbanRunResult, String> {
    let task = get_task(conn, task_id)?.ok_or_else(|| format!("unknown task {task_id}"))?;
    if task.status != "ready" {
        return Err(format!(
            "could not run {task_id} (task is not ready; status={})",
            task.status
        ));
    }
    let assignee = task
        .assignee
        .clone()
        .ok_or_else(|| format!("task {task_id} has no assignee"))?;
    if !context.profile_exists(&assignee) {
        return Err(format!(
            "task {task_id} is assigned to {assignee}, which is not a Hermes profile"
        ));
    }
    let workspace = resolve_workspace(&task, &context.hermes_home())?;
    if dry_run {
        return Ok(KanbanRunResult {
            task_id: task.id,
            assignee,
            workspace_path: workspace.display().to_string(),
            pid: None,
            dry_run: true,
        });
    }

    let claimed = claim_task(conn, task_id, ttl_seconds, None)?.ok_or_else(|| {
        format!("could not claim {task_id} (task is not ready or is already claimed)")
    })?;
    let run = run_claimed_task(conn, context, &claimed, failure_limit, spawn_fn)?;
    let Some(record) = run.record else {
        return Err(format!("task {} was claimed but not spawned", claimed.id));
    };
    Ok(KanbanRunResult {
        task_id: record.task_id,
        assignee: record.assignee,
        workspace_path: record.workspace_path,
        pid: record.pid,
        dry_run: false,
    })
}

#[derive(Debug, Default)]
struct RunClaimedTaskResult {
    record: Option<KanbanSpawnRecord>,
    auto_blocked: bool,
}

fn run_claimed_task(
    conn: &mut Connection,
    context: &HermesContext,
    task: &Task,
    failure_limit: i64,
    spawn_fn: Option<&KanbanSpawnFn>,
) -> Result<RunClaimedTaskResult, String> {
    let workspace = resolve_workspace(task, &context.hermes_home())?;
    set_workspace_path(conn, &task.id, &workspace)?;

    let pid = match spawn_fn {
        Some(spawn) => spawn(task, &workspace),
        None => default_spawn(task, &workspace, context),
    };
    match pid {
        Ok(pid) => {
            if let Some(pid) = pid {
                set_worker_pid(conn, &task.id, pid)?;
            }
            Ok(RunClaimedTaskResult {
                record: Some(KanbanSpawnRecord {
                    task_id: task.id.clone(),
                    assignee: task.assignee.clone().unwrap_or_default(),
                    workspace_path: workspace.display().to_string(),
                    pid,
                }),
                auto_blocked: false,
            })
        }
        Err(error) => {
            let auto_blocked = record_task_failure(
                conn,
                &task.id,
                &error,
                "spawn_failed",
                failure_limit,
                true,
                true,
                None,
            )?;
            Ok(RunClaimedTaskResult {
                record: None,
                auto_blocked,
            })
        }
    }
}

pub fn create_task(conn: &mut Connection, input: CreateTaskInput) -> Result<String, String> {
    let title = input.title.trim().to_string();
    if title.is_empty() {
        return Err("title is required".to_string());
    }
    let assignee = input.assignee.as_deref().and_then(normalize_profile_name);
    if !VALID_WORKSPACE_KINDS.contains(&input.workspace_kind.as_str()) {
        return Err(format!(
            "workspace_kind must be one of {}",
            VALID_WORKSPACE_KINDS.join(", ")
        ));
    }
    let created_by = non_empty_trimmed(input.created_by).unwrap_or_else(|| "worker".to_string());
    let parents = dedupe_strings(input.parents);
    let skills = input.skills.map(dedupe_strings);

    if let Some(idempotency_key) = input.idempotency_key.as_deref() {
        let existing = conn
            .query_row(
                "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
                [idempotency_key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| format!("querying idempotency key failed: {error}"))?;
        if let Some(task_id) = existing {
            return Ok(task_id);
        }
    }

    for _ in 0..2 {
        let task_id = new_task_id();
        let tx = begin_immediate(conn)?;
        let initial_status = if input.triage {
            "triage".to_string()
        } else if parents.is_empty() {
            "ready".to_string()
        } else {
            let missing = find_missing_parents(&tx, &parents)?;
            if !missing.is_empty() {
                return Err(format!("unknown parent task(s): {}", missing.join(", ")));
            }
            if all_parents_done(&tx, &parents)? {
                "ready".to_string()
            } else {
                "todo".to_string()
            }
        };

        let inserted = tx.execute(
            "INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by, created_at,
                workspace_kind, workspace_path, tenant, idempotency_key,
                max_runtime_seconds, skills
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                task_id,
                title,
                input.body,
                assignee,
                initial_status,
                input.priority,
                created_by,
                now_ts(),
                input.workspace_kind,
                input.workspace_path,
                input.tenant,
                input.idempotency_key,
                input.max_runtime_seconds,
                skills
                    .as_ref()
                    .map(|value| serde_json::to_string(value).unwrap_or_else(|_| "[]".to_string())),
            ],
        );
        match inserted {
            Ok(_) => {}
            Err(rusqlite::Error::SqliteFailure(error, _))
                if error.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_PRIMARYKEY
                    || error.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE =>
            {
                continue;
            }
            Err(error) => return Err(format!("creating task failed: {error}")),
        }

        for parent_id in &parents {
            tx.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                params![parent_id, task_id],
            )
            .map_err(|error| format!("linking parent {parent_id} failed: {error}"))?;
        }
        append_event(
            &tx,
            &task_id,
            "created",
            Some(json!({
                "assignee": assignee,
                "status": initial_status,
                "parents": parents,
                "tenant": input.tenant,
                "skills": skills,
            })),
            None,
        )?;
        tx.commit()
            .map_err(|error| format!("committing task creation failed: {error}"))?;
        return Ok(task_id);
    }

    Err("creating task failed due to repeated id collisions".to_string())
}

pub fn get_task(conn: &Connection, task_id: &str) -> Result<Option<Task>, String> {
    conn.query_row(
        "SELECT * FROM tasks WHERE id = ?",
        [task_id],
        Task::from_row,
    )
    .optional()
    .map_err(|error| format!("reading task {task_id} failed: {error}"))
}

pub fn link_tasks(conn: &mut Connection, parent_id: &str, child_id: &str) -> Result<(), String> {
    if parent_id == child_id {
        return Err("a task cannot depend on itself".to_string());
    }
    let tx = begin_immediate(conn)?;
    let missing = find_missing_parents(&tx, &[parent_id.to_string(), child_id.to_string()])?;
    if !missing.is_empty() {
        return Err(format!("unknown task(s): {}", missing.join(", ")));
    }
    if would_cycle(&tx, parent_id, child_id)? {
        return Err(format!(
            "linking {parent_id} -> {child_id} would create a cycle"
        ));
    }
    tx.execute(
        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
        params![parent_id, child_id],
    )
    .map_err(|error| format!("linking tasks failed: {error}"))?;
    let parent_status: String = tx
        .query_row(
            "SELECT status FROM tasks WHERE id = ?",
            [parent_id],
            |row| row.get(0),
        )
        .map_err(|error| format!("reading parent task failed: {error}"))?;
    if parent_status != "done" {
        tx.execute(
            "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
            [child_id],
        )
        .map_err(|error| format!("demoting linked child failed: {error}"))?;
    }
    append_event(
        &tx,
        child_id,
        "linked",
        Some(json!({
            "parent": parent_id,
            "child": child_id,
        })),
        None,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task link failed: {error}"))?;
    Ok(())
}

pub fn parent_ids(conn: &Connection, task_id: &str) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id")
        .map_err(|error| format!("preparing parent query failed: {error}"))?;
    let rows = stmt
        .query_map([task_id], |row| row.get::<_, String>(0))
        .map_err(|error| format!("querying parent ids failed: {error}"))?;
    collect_string_rows(rows, "reading parent id")
}

pub fn child_ids(conn: &Connection, task_id: &str) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id")
        .map_err(|error| format!("preparing child query failed: {error}"))?;
    let rows = stmt
        .query_map([task_id], |row| row.get::<_, String>(0))
        .map_err(|error| format!("querying child ids failed: {error}"))?;
    collect_string_rows(rows, "reading child id")
}

pub fn add_comment(
    conn: &mut Connection,
    task_id: &str,
    author: &str,
    body: &str,
) -> Result<i64, String> {
    let author = non_empty_trimmed(author.to_string())
        .ok_or_else(|| "comment author is required".to_string())?;
    let body = non_empty_trimmed(body.to_string())
        .ok_or_else(|| "comment body is required".to_string())?;
    let tx = begin_immediate(conn)?;
    let exists = tx
        .query_row("SELECT 1 FROM tasks WHERE id = ?", [task_id], |_| Ok(()))
        .optional()
        .map_err(|error| format!("checking task {task_id} failed: {error}"))?;
    if exists.is_none() {
        return Err(format!("unknown task {task_id}"));
    }
    tx.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
        params![task_id, author, body, now_ts()],
    )
    .map_err(|error| format!("adding comment failed: {error}"))?;
    let comment_id = tx.last_insert_rowid();
    append_event(
        &tx,
        task_id,
        "commented",
        Some(json!({
            "author": author,
            "len": body.len(),
        })),
        None,
    )?;
    tx.commit()
        .map_err(|error| format!("committing comment failed: {error}"))?;
    Ok(comment_id)
}

pub fn list_comments(conn: &Connection, task_id: &str) -> Result<Vec<Comment>, String> {
    let mut stmt = conn
        .prepare("SELECT id, task_id, author, body, created_at FROM task_comments WHERE task_id = ? ORDER BY created_at ASC")
        .map_err(|error| format!("preparing comment query failed: {error}"))?;
    let rows = stmt
        .query_map([task_id], |row| {
            Ok(Comment {
                id: row.get("id")?,
                task_id: row.get("task_id")?,
                author: row.get("author")?,
                body: row.get("body")?,
                created_at: row.get("created_at")?,
            })
        })
        .map_err(|error| format!("querying comments failed: {error}"))?;
    collect_rows(rows, "reading comment")
}

pub fn list_events(conn: &Connection, task_id: &str) -> Result<Vec<Event>, String> {
    let mut stmt = conn
        .prepare("SELECT id, task_id, kind, payload, created_at, run_id FROM task_events WHERE task_id = ? ORDER BY created_at ASC, id ASC")
        .map_err(|error| format!("preparing event query failed: {error}"))?;
    let rows = stmt
        .query_map([task_id], |row| {
            let payload = row
                .get::<_, Option<String>>("payload")?
                .and_then(|value| serde_json::from_str::<Value>(&value).ok());
            Ok(Event {
                id: row.get("id")?,
                task_id: row.get("task_id")?,
                kind: row.get("kind")?,
                payload,
                created_at: row.get("created_at")?,
                run_id: row.get("run_id")?,
            })
        })
        .map_err(|error| format!("querying events failed: {error}"))?;
    collect_rows(rows, "reading event")
}

pub fn list_runs(conn: &Connection, task_id: &str) -> Result<Vec<Run>, String> {
    let mut stmt = conn
        .prepare("SELECT * FROM task_runs WHERE task_id = ? ORDER BY started_at ASC, id ASC")
        .map_err(|error| format!("preparing run query failed: {error}"))?;
    let rows = stmt
        .query_map([task_id], Run::from_row)
        .map_err(|error| format!("querying runs failed: {error}"))?;
    collect_rows(rows, "reading run")
}

pub fn latest_run(conn: &Connection, task_id: &str) -> Result<Option<Run>, String> {
    conn.query_row(
        "SELECT * FROM task_runs WHERE task_id = ? ORDER BY started_at DESC, id DESC LIMIT 1",
        [task_id],
        Run::from_row,
    )
    .optional()
    .map_err(|error| format!("reading latest run failed: {error}"))
}

pub fn latest_summary(conn: &Connection, task_id: &str) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT summary FROM task_runs WHERE task_id = ? AND summary IS NOT NULL AND summary != '' ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        [task_id],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|error| format!("reading latest summary failed: {error}"))
}

pub fn complete_task(
    conn: &mut Connection,
    task_id: &str,
    result: Option<&str>,
    summary: Option<&str>,
    metadata: Option<&Value>,
    created_cards: &[String],
    expected_run_id: Option<i64>,
) -> Result<bool, String> {
    let verified_cards = if created_cards.is_empty() {
        Vec::new()
    } else {
        let (verified, phantom) = verify_created_cards(conn, task_id, created_cards)?;
        if !phantom.is_empty() {
            let tx = begin_immediate(conn)?;
            append_event(
                &tx,
                task_id,
                "completion_blocked_hallucination",
                Some(json!({
                    "phantom_cards": phantom,
                    "verified_cards": verified,
                    "summary_preview": summary.or(result).map(|value| value.lines().next().unwrap_or_default().chars().take(200).collect::<String>()),
                })),
                None,
            )?;
            tx.commit()
                .map_err(|error| format!("committing hallucination audit failed: {error}"))?;
            return Err(format!(
                "kanban_complete blocked: the following created_cards do not exist or were not created by this worker: {}. Either omit them, use only ids returned from successful kanban_create calls, or remove the created_cards field.",
                phantom.join(", ")
            ));
        }
        verified
    };

    let tx = begin_immediate(conn)?;
    let updated = if let Some(run_id) = expected_run_id {
        tx.execute(
            "UPDATE tasks SET status = 'done', result = ?, completed_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready', 'blocked') AND current_run_id = ?",
            params![result, now_ts(), task_id, run_id],
        )
    } else {
        tx.execute(
            "UPDATE tasks SET status = 'done', result = ?, completed_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready', 'blocked')",
            params![result, now_ts(), task_id],
        )
    }
    .map_err(|error| format!("updating task completion failed: {error}"))?;
    if updated != 1 {
        return Ok(false);
    }

    let handoff_summary = summary.or(result);
    let mut run_id = end_run(
        &tx,
        task_id,
        "done",
        "completed",
        handoff_summary,
        metadata,
        None,
    )?;
    if run_id.is_none() && (handoff_summary.is_some() || metadata.is_some()) {
        run_id = Some(synthesize_ended_run(
            &tx,
            task_id,
            "done",
            "completed",
            handoff_summary,
            metadata,
            None,
        )?);
    }
    let event_summary = handoff_summary
        .map(|value| truncate_line(value, 400))
        .filter(|value| !value.is_empty());
    let mut payload = Map::new();
    payload.insert(
        "result_len".to_string(),
        Value::Number(serde_json::Number::from(
            result.unwrap_or_default().len() as u64
        )),
    );
    payload.insert(
        "summary".to_string(),
        event_summary.map(Value::String).unwrap_or(Value::Null),
    );
    if !verified_cards.is_empty() {
        payload.insert("verified_cards".to_string(), json!(verified_cards));
    }
    append_event(
        &tx,
        task_id,
        "completed",
        Some(Value::Object(payload)),
        run_id,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task completion failed: {error}"))?;

    clear_failure_counter(conn, task_id)?;
    recompute_ready(conn)?;
    Ok(true)
}

pub fn edit_completed_task_result(
    conn: &mut Connection,
    task_id: &str,
    result: &str,
    summary: Option<&str>,
    metadata: Option<&Value>,
) -> Result<bool, String> {
    let task_id =
        non_empty_trimmed(task_id.to_string()).ok_or_else(|| "task_id is required".to_string())?;
    let result = result.trim();
    if result.is_empty() {
        return Err("result is required".to_string());
    }
    let handoff_summary = summary.unwrap_or(result);
    let tx = begin_immediate(conn)?;
    let status = tx
        .query_row("SELECT status FROM tasks WHERE id = ?", [&task_id], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|error| format!("reading task status failed: {error}"))?;
    if status.as_deref() != Some("done") {
        return Ok(false);
    }
    tx.execute(
        "UPDATE tasks SET result = ? WHERE id = ?",
        params![result, task_id],
    )
    .map_err(|error| format!("updating task result failed: {error}"))?;

    let mut run_id = tx
        .query_row(
            "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'completed' ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC LIMIT 1",
            [&task_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| format!("reading completed run failed: {error}"))?;
    if let Some(existing_run_id) = run_id {
        tx.execute(
            "UPDATE task_runs SET summary = ?, metadata = COALESCE(?, metadata) WHERE id = ?",
            params![
                handoff_summary,
                metadata.map(|value| {
                    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
                }),
                existing_run_id,
            ],
        )
        .map_err(|error| format!("updating completed run failed: {error}"))?;
    } else {
        run_id = Some(synthesize_ended_run(
            &tx,
            &task_id,
            "done",
            "completed",
            Some(handoff_summary),
            metadata,
            None,
        )?);
    }

    let event_summary = truncate_line(handoff_summary, 400);
    let fields = if metadata.is_some() {
        vec!["result", "summary", "metadata"]
    } else {
        vec!["result", "summary"]
    };
    append_event(
        &tx,
        &task_id,
        "edited",
        Some(json!({
            "fields": fields,
            "result_len": result.len(),
            "summary": (!event_summary.is_empty()).then_some(event_summary),
        })),
        run_id,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task edit failed: {error}"))?;
    Ok(true)
}

pub fn block_task(
    conn: &mut Connection,
    task_id: &str,
    reason: &str,
    expected_run_id: Option<i64>,
) -> Result<bool, String> {
    let tx = begin_immediate(conn)?;
    let updated = if let Some(run_id) = expected_run_id {
        tx.execute(
            "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready') AND current_run_id = ?",
            params![task_id, run_id],
        )
    } else {
        tx.execute(
            "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready')",
            [task_id],
        )
    }
    .map_err(|error| format!("updating task block failed: {error}"))?;
    if updated != 1 {
        return Ok(false);
    }
    let mut run_id = end_run(&tx, task_id, "blocked", "blocked", Some(reason), None, None)?;
    if run_id.is_none() {
        run_id = Some(synthesize_ended_run(
            &tx,
            task_id,
            "blocked",
            "blocked",
            Some(reason),
            None,
            None,
        )?);
    }
    append_event(
        &tx,
        task_id,
        "blocked",
        Some(json!({"reason": reason})),
        run_id,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task block failed: {error}"))?;
    Ok(true)
}

pub fn heartbeat_worker(
    conn: &mut Connection,
    task_id: &str,
    note: Option<&str>,
    expected_run_id: Option<i64>,
) -> Result<bool, String> {
    let tx = begin_immediate(conn)?;
    let now = now_ts();
    let updated = if let Some(run_id) = expected_run_id {
        tx.execute(
            "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running' AND current_run_id = ?",
            params![now, task_id, run_id],
        )
    } else {
        tx.execute(
            "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running'",
            params![now, task_id],
        )
    }
    .map_err(|error| format!("updating heartbeat failed: {error}"))?;
    if updated != 1 {
        return Ok(false);
    }
    let run_id = if let Some(run_id) = expected_run_id {
        Some(run_id)
    } else {
        current_run_id(&tx, task_id)?
    };
    if let Some(run_id) = run_id {
        tx.execute(
            "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
            params![now, run_id],
        )
        .map_err(|error| format!("updating run heartbeat failed: {error}"))?;
    }
    append_event(
        &tx,
        task_id,
        "heartbeat",
        note.map(|value| json!({"note": value})),
        run_id,
    )?;
    tx.commit()
        .map_err(|error| format!("committing heartbeat failed: {error}"))?;
    Ok(true)
}

pub fn recompute_ready(conn: &mut Connection) -> Result<usize, String> {
    let tx = begin_immediate(conn)?;
    let todo_ids = {
        let mut stmt = tx
            .prepare("SELECT id FROM tasks WHERE status = 'todo'")
            .map_err(|error| format!("preparing todo query failed: {error}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| format!("querying todo tasks failed: {error}"))?;
        collect_string_rows(rows, "reading todo task id")?
    };

    let mut promoted = 0_usize;
    for task_id in todo_ids {
        let parents = parent_ids(&tx, &task_id)?;
        if all_parents_done(&tx, &parents)? {
            let updated = tx
                .execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                    [task_id.as_str()],
                )
                .map_err(|error| format!("promoting task failed: {error}"))?;
            if updated == 1 {
                append_event(&tx, &task_id, "promoted", None, None)?;
                promoted += 1;
            }
        }
    }
    tx.commit()
        .map_err(|error| format!("committing ready recompute failed: {error}"))?;
    Ok(promoted)
}

pub fn build_worker_context(conn: &Connection, task_id: &str) -> Result<String, String> {
    let task = get_task(conn, task_id)?.ok_or_else(|| format!("unknown task {task_id}"))?;
    let mut lines = Vec::new();
    lines.push(format!("# Kanban task {}: {}", task.id, task.title));
    lines.push(String::new());
    lines.push(format!(
        "Assignee: {}",
        task.assignee.as_deref().unwrap_or("(unassigned)")
    ));
    lines.push(format!("Status: {}", task.status));
    if let Some(tenant) = task.tenant.as_deref() {
        lines.push(format!("Tenant: {tenant}"));
    }
    lines.push(format!(
        "Workspace: {} @ {}",
        task.workspace_kind,
        task.workspace_path.as_deref().unwrap_or("(unresolved)")
    ));
    lines.push(String::new());

    if let Some(body) = task.body.as_deref()
        && !body.trim().is_empty()
    {
        lines.push("## Body".to_string());
        lines.push(cap_text(body, CONTEXT_MAX_BODY_CHARS));
        lines.push(String::new());
    }

    let prior_runs = list_runs(conn, task_id)?
        .into_iter()
        .filter(|run| run.ended_at.is_some())
        .collect::<Vec<_>>();
    if !prior_runs.is_empty() {
        lines.push("## Prior attempts on this task".to_string());
        let shown = prior_runs
            .iter()
            .rev()
            .take(CONTEXT_MAX_PRIOR_RUNS)
            .cloned()
            .collect::<Vec<_>>();
        if prior_runs.len() > shown.len() {
            lines.push(format!(
                "_({} earlier attempts omitted; showing most recent {})_",
                prior_runs.len() - shown.len(),
                shown.len()
            ));
        }
        for (index, run) in shown.into_iter().rev().enumerate() {
            lines.push(format!(
                "### Attempt {} — {} ({}, {})",
                index + 1,
                run.outcome.clone().unwrap_or(run.status.clone()),
                run.profile
                    .clone()
                    .unwrap_or_else(|| "(unknown)".to_string()),
                format_ts(run.started_at)
            ));
            if let Some(summary) = run.summary.as_deref()
                && !summary.trim().is_empty()
            {
                lines.push(cap_text(summary, CONTEXT_MAX_FIELD_CHARS));
            }
            if let Some(error) = run.error.as_deref()
                && !error.trim().is_empty()
            {
                lines.push(format!(
                    "_error_: {}",
                    cap_text(error, CONTEXT_MAX_FIELD_CHARS)
                ));
            }
            if let Some(metadata) = run.metadata {
                let encoded = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
                lines.push(format!(
                    "_metadata_: `{}`",
                    cap_text(&encoded, CONTEXT_MAX_FIELD_CHARS)
                ));
            }
            lines.push(String::new());
        }
    }

    let parents = parent_ids(conn, task_id)?;
    if !parents.is_empty() {
        lines.push("## Parent handoffs".to_string());
        for parent_id in parents {
            let parent = match get_task(conn, &parent_id)? {
                Some(value) => value,
                None => continue,
            };
            lines.push(format!("### {} — {}", parent.id, parent.title));
            let summary = latest_summary(conn, &parent.id)?.or(parent.result.clone());
            if let Some(summary) = summary
                && !summary.trim().is_empty()
            {
                lines.push(cap_text(&summary, CONTEXT_MAX_FIELD_CHARS));
            } else {
                lines.push("_no recorded summary_".to_string());
            }
            lines.push(String::new());
        }
    }

    let comments = list_comments(conn, task_id)?;
    if !comments.is_empty() {
        lines.push("## Comments".to_string());
        let shown = comments
            .iter()
            .rev()
            .take(CONTEXT_MAX_COMMENTS)
            .cloned()
            .collect::<Vec<_>>();
        if comments.len() > shown.len() {
            lines.push(format!(
                "_({} older comments omitted; showing most recent {})_",
                comments.len() - shown.len(),
                shown.len()
            ));
        }
        for comment in shown.into_iter().rev() {
            lines.push(format!(
                "- [{}] {}: {}",
                format_ts(comment.created_at),
                comment.author,
                cap_text(&comment.body, CONTEXT_MAX_FIELD_CHARS)
            ));
        }
    }

    Ok(lines.join("\n").trim().to_string())
}

pub fn list_tasks(conn: &Connection, query: &KanbanTaskQuery) -> Result<Vec<Task>, String> {
    if let Some(status) = query.status.as_deref()
        && !VALID_KANBAN_STATUSES.contains(&status)
    {
        return Err(format!(
            "status must be one of {}",
            VALID_KANBAN_STATUSES.join(", ")
        ));
    }

    let mut sql = String::from("SELECT * FROM tasks WHERE 1=1");
    let mut params = Vec::<String>::new();
    if !query.include_archived {
        sql.push_str(" AND status != ?");
        params.push(String::from("archived"));
    }
    if let Some(assignee) = query.assignee.as_deref() {
        sql.push_str(" AND assignee = ?");
        params.push(assignee.to_string());
    }
    if let Some(status) = query.status.as_deref() {
        sql.push_str(" AND status = ?");
        params.push(status.to_string());
    }
    if let Some(tenant) = query.tenant.as_deref() {
        sql.push_str(" AND tenant = ?");
        params.push(tenant.to_string());
    }
    sql.push_str(
        " ORDER BY CASE status
            WHEN 'running' THEN 0
            WHEN 'ready' THEN 1
            WHEN 'todo' THEN 2
            WHEN 'triage' THEN 3
            WHEN 'blocked' THEN 4
            WHEN 'done' THEN 5
            ELSE 6 END,
          priority DESC,
          created_at ASC,
          id ASC",
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|error| format!("preparing task list failed: {error}"))?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), Task::from_row)
        .map_err(|error| format!("querying task list failed: {error}"))?;
    collect_rows(rows, "reading task")
}

pub fn kanban_task_detail(
    conn: &Connection,
    task_id: &str,
) -> Result<Option<KanbanTaskDetail>, String> {
    let Some(task) = get_task(conn, task_id)? else {
        return Ok(None);
    };
    Ok(Some(KanbanTaskDetail {
        task,
        parents: parent_ids(conn, task_id)?,
        children: child_ids(conn, task_id)?,
        comments: list_comments(conn, task_id)?,
        events: list_events(conn, task_id)?,
        runs: list_runs(conn, task_id)?,
        latest_summary: latest_summary(conn, task_id)?,
        worker_context: build_worker_context(conn, task_id)?,
    }))
}

pub fn unlink_tasks(
    conn: &mut Connection,
    parent_id: &str,
    child_id: &str,
) -> Result<bool, String> {
    let tx = begin_immediate(conn)?;
    let removed = tx
        .execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            params![parent_id, child_id],
        )
        .map_err(|error| format!("unlinking tasks failed: {error}"))?;
    if removed == 0 {
        return Ok(false);
    }
    append_event(
        &tx,
        child_id,
        "unlinked",
        Some(json!({"parent": parent_id, "child": child_id})),
        None,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task unlink failed: {error}"))?;
    recompute_ready(conn)?;
    Ok(true)
}

pub fn assign_task(
    conn: &mut Connection,
    task_id: &str,
    profile: Option<&str>,
) -> Result<bool, String> {
    let status = get_task(conn, task_id)?
        .map(|task| task.status)
        .ok_or_else(|| format!("unknown task {task_id}"))?;
    if status == "running" {
        return Err(format!(
            "cannot assign {task_id} while it is running; reclaim it first"
        ));
    }
    let assignee = profile.and_then(normalize_profile_name);
    let tx = begin_immediate(conn)?;
    let updated = tx
        .execute(
            "UPDATE tasks SET assignee = ? WHERE id = ? AND status != 'archived'",
            params![assignee, task_id],
        )
        .map_err(|error| format!("assigning task failed: {error}"))?;
    if updated == 0 {
        return Ok(false);
    }
    append_event(
        &tx,
        task_id,
        "assigned",
        Some(json!({"assignee": assignee})),
        None,
    )?;
    tx.commit()
        .map_err(|error| format!("committing assignment failed: {error}"))?;
    Ok(true)
}

pub fn unblock_task(conn: &mut Connection, task_id: &str) -> Result<bool, String> {
    let parents = parent_ids(conn, task_id)?;
    let next_status = if all_parents_done(conn, &parents)? {
        "ready"
    } else {
        "todo"
    };
    let tx = begin_immediate(conn)?;
    let updated = tx
        .execute(
            "UPDATE tasks
                SET status = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
              WHERE id = ? AND status = 'blocked'",
            params![next_status, task_id],
        )
        .map_err(|error| format!("unblocking task failed: {error}"))?;
    if updated == 0 {
        return Ok(false);
    }
    append_event(
        &tx,
        task_id,
        "unblocked",
        Some(json!({"status": next_status})),
        None,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task unblock failed: {error}"))?;
    Ok(true)
}

pub fn archive_task(conn: &mut Connection, task_id: &str) -> Result<bool, String> {
    let tx = begin_immediate(conn)?;
    let updated = tx
        .execute(
            "UPDATE tasks
                SET status = 'archived',
                    claim_lock = NULL,
                    claim_expires = NULL,
                    worker_pid = NULL
              WHERE id = ? AND status != 'archived'",
            [task_id],
        )
        .map_err(|error| format!("archiving task failed: {error}"))?;
    if updated == 0 {
        return Ok(false);
    }
    let run_id = end_run(
        &tx,
        task_id,
        "reclaimed",
        "reclaimed",
        Some("task archived with run still active"),
        None,
        None,
    )?;
    append_event(&tx, task_id, "archived", None, run_id)?;
    tx.commit()
        .map_err(|error| format!("committing task archive failed: {error}"))?;
    Ok(true)
}

pub fn reclaim_task(
    conn: &mut Connection,
    task_id: &str,
    reason: Option<&str>,
) -> Result<bool, String> {
    let task = get_task(conn, task_id)?.ok_or_else(|| format!("unknown task {task_id}"))?;
    if task.status != "running" {
        return Ok(false);
    }
    let tx = begin_immediate(conn)?;
    tx.execute(
        "UPDATE tasks
            SET status = 'ready',
                claim_lock = NULL,
                claim_expires = NULL,
                worker_pid = NULL,
                consecutive_failures = 0,
                last_failure_error = NULL
          WHERE id = ? AND status = 'running'",
        [task_id],
    )
    .map_err(|error| format!("reclaiming task failed: {error}"))?;
    let run_id = end_run(
        &tx,
        task_id,
        "ready",
        "reclaimed",
        reason,
        Some(&json!({"manual": true, "reason": reason})),
        None,
    )?;
    append_event(
        &tx,
        task_id,
        "reclaimed",
        Some(json!({"manual": true, "reason": reason})),
        run_id,
    )?;
    tx.commit()
        .map_err(|error| format!("committing task reclaim failed: {error}"))?;
    Ok(true)
}

pub fn reassign_task(
    conn: &mut Connection,
    task_id: &str,
    profile: Option<&str>,
    reclaim_first: bool,
    reason: Option<&str>,
) -> Result<bool, String> {
    let Some(task) = get_task(conn, task_id)? else {
        return Ok(false);
    };
    if task.status == "running" {
        if !reclaim_first || !reclaim_task(conn, task_id, reason)? {
            return Ok(false);
        }
    }
    assign_task(conn, task_id, profile)
}

pub fn board_stats(conn: &Connection) -> Result<KanbanBoardStats, String> {
    let mut stats = KanbanBoardStats::default();
    let mut by_status = conn
        .prepare(
            "SELECT status, COUNT(*)
               FROM tasks
              WHERE status != 'archived'
              GROUP BY status",
        )
        .map_err(|error| format!("preparing status stats failed: {error}"))?;
    let rows = by_status
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .map_err(|error| format!("querying status stats failed: {error}"))?;
    for row in rows {
        let (status, count) =
            row.map_err(|error| format!("reading status stat failed: {error}"))?;
        stats.by_status.insert(status, count);
    }

    let mut by_assignee = conn
        .prepare(
            "SELECT COALESCE(assignee, ''), status, COUNT(*)
               FROM tasks
              WHERE status != 'archived'
              GROUP BY COALESCE(assignee, ''), status",
        )
        .map_err(|error| format!("preparing assignee stats failed: {error}"))?;
    let rows = by_assignee
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|error| format!("querying assignee stats failed: {error}"))?;
    for row in rows {
        let (assignee, status, count) =
            row.map_err(|error| format!("reading assignee stat failed: {error}"))?;
        stats
            .by_assignee
            .entry(if assignee.is_empty() {
                String::from("(unassigned)")
            } else {
                assignee
            })
            .or_default()
            .insert(status, count);
    }

    stats.oldest_ready_age_seconds = conn
        .query_row(
            "SELECT MIN(created_at) FROM tasks WHERE status = 'ready'",
            [],
            |row| row.get::<_, Option<i64>>(0),
        )
        .optional()
        .map_err(|error| format!("reading oldest ready task failed: {error}"))?
        .flatten()
        .map(|created_at| now_ts().saturating_sub(created_at));
    Ok(stats)
}

pub fn known_assignees(
    conn: &Connection,
    context: &HermesContext,
) -> Result<Vec<KanbanAssigneeRecord>, String> {
    let mut on_disk = HashSet::new();
    on_disk.insert(String::from("default"));
    if let Ok(entries) = fs::read_dir(context.profiles_root()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if context.profile_exists(name) {
                on_disk.insert(name.to_string());
            }
        }
    }

    let mut counts: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    let mut stmt = conn
        .prepare(
            "SELECT assignee, status, COUNT(*) AS n
               FROM tasks
              WHERE status != 'archived' AND assignee IS NOT NULL
              GROUP BY assignee, status",
        )
        .map_err(|error| format!("preparing known assignees query failed: {error}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|error| format!("querying known assignees failed: {error}"))?;
    for row in rows {
        let (assignee, status, count) =
            row.map_err(|error| format!("reading known assignee failed: {error}"))?;
        counts.entry(assignee).or_default().insert(status, count);
    }

    let mut names = on_disk;
    names.extend(counts.keys().cloned());
    let mut records = names
        .into_iter()
        .map(|name| KanbanAssigneeRecord {
            on_disk: context.profile_exists(&name),
            counts: counts.remove(&name).unwrap_or_default(),
            name,
        })
        .collect::<Vec<_>>();
    records.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(records)
}

pub fn claim_task(
    conn: &mut Connection,
    task_id: &str,
    ttl_seconds: i64,
    claimer: Option<&str>,
) -> Result<Option<Task>, String> {
    let now = now_ts();
    let lock = claimer.map(ToOwned::to_owned).unwrap_or_else(claimer_id);
    let expires = now + ttl_seconds.max(1);
    let tx = begin_immediate(conn)?;

    if let Some(run_id) = tx
        .query_row(
            "SELECT current_run_id FROM tasks WHERE id = ? AND status = 'ready'",
            [task_id],
            |row| row.get::<_, Option<i64>>(0),
        )
        .optional()
        .map_err(|error| format!("reading stale run pointer failed: {error}"))?
        .flatten()
    {
        tx.execute(
            "UPDATE task_runs
                SET status = 'reclaimed',
                    outcome = 'reclaimed',
                    summary = COALESCE(summary, 'invariant recovery on re-claim'),
                    ended_at = ?,
                    claim_lock = NULL,
                    claim_expires = NULL,
                    worker_pid = NULL
              WHERE id = ? AND ended_at IS NULL",
            params![now, run_id],
        )
        .map_err(|error| format!("closing stale claimed run failed: {error}"))?;
    }

    let updated = tx
        .execute(
            "UPDATE tasks
                SET status = 'running',
                    claim_lock = ?,
                    claim_expires = ?,
                    started_at = COALESCE(started_at, ?),
                    last_heartbeat_at = NULL
              WHERE id = ?
                AND status = 'ready'
                AND claim_lock IS NULL",
            params![lock, expires, now, task_id],
        )
        .map_err(|error| format!("claiming task failed: {error}"))?;
    if updated != 1 {
        return Ok(None);
    }

    let run_meta = tx
        .query_row(
            "SELECT assignee, max_runtime_seconds, current_step_key FROM tasks WHERE id = ?",
            [task_id],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .map_err(|error| format!("reading claimed task metadata failed: {error}"))?;
    tx.execute(
        "INSERT INTO task_runs (
            task_id, profile, step_key, status,
            claim_lock, claim_expires, max_runtime_seconds, started_at
         ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
        params![
            task_id, run_meta.0, run_meta.2, lock, expires, run_meta.1, now,
        ],
    )
    .map_err(|error| format!("creating task run failed: {error}"))?;
    let run_id = tx.last_insert_rowid();
    tx.execute(
        "UPDATE tasks SET current_run_id = ? WHERE id = ?",
        params![run_id, task_id],
    )
    .map_err(|error| format!("recording current run failed: {error}"))?;
    append_event(
        &tx,
        task_id,
        "claimed",
        Some(json!({
            "lock": lock,
            "expires": expires,
            "run_id": run_id,
        })),
        Some(run_id),
    )?;
    tx.commit()
        .map_err(|error| format!("committing task claim failed: {error}"))?;
    get_task(conn, task_id)
}

pub fn release_stale_claims(conn: &mut Connection) -> Result<usize, String> {
    let now = now_ts();
    let stale_rows = {
        let mut stmt = conn
            .prepare(
                "SELECT id, claim_lock FROM tasks
                 WHERE status = 'running'
                   AND claim_expires IS NOT NULL
                   AND claim_expires < ?",
            )
            .map_err(|error| format!("preparing stale-claim query failed: {error}"))?;
        let rows = stmt
            .query_map([now], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|error| format!("querying stale claims failed: {error}"))?;
        collect_rows(rows, "reading stale claim")?
    };

    let mut reclaimed = 0_usize;
    for (task_id, claim_lock) in stale_rows {
        let tx = begin_immediate(conn)?;
        let updated = tx
            .execute(
                "UPDATE tasks
                    SET status = 'ready',
                        claim_lock = NULL,
                        claim_expires = NULL,
                        worker_pid = NULL,
                        last_heartbeat_at = NULL
                  WHERE id = ? AND status = 'running'",
                [task_id.as_str()],
            )
            .map_err(|error| format!("reclaiming stale task failed: {error}"))?;
        if updated == 1 {
            let run_id = end_run(
                &tx,
                &task_id,
                "reclaimed",
                "reclaimed",
                None,
                None,
                claim_lock
                    .as_deref()
                    .map(|value| format!("stale_lock={value}"))
                    .as_deref(),
            )?;
            append_event(
                &tx,
                &task_id,
                "reclaimed",
                Some(json!({
                    "stale_lock": claim_lock,
                })),
                run_id,
            )?;
            reclaimed += 1;
        }
        tx.commit()
            .map_err(|error| format!("committing stale-claim reclaim failed: {error}"))?;
    }
    Ok(reclaimed)
}

fn detect_crashed_workers(
    conn: &mut Connection,
    failure_limit: i64,
) -> Result<Vec<String>, String> {
    let host_prefix = format!("{}:", claimer_host());
    let rows = {
        let mut stmt = conn
            .prepare(
                "SELECT id, worker_pid, claim_lock FROM tasks
                 WHERE status = 'running' AND worker_pid IS NOT NULL",
            )
            .map_err(|error| format!("preparing crashed-worker query failed: {error}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(|error| format!("querying running workers failed: {error}"))?;
        collect_rows(rows, "reading running worker")?
    };

    let mut crashed = Vec::new();
    for (task_id, worker_pid, claim_lock) in rows {
        if !claim_lock
            .as_deref()
            .unwrap_or_default()
            .starts_with(&host_prefix)
        {
            continue;
        }
        if pid_alive(worker_pid) {
            continue;
        }

        let tx = begin_immediate(conn)?;
        let updated = tx
            .execute(
                "UPDATE tasks
                    SET status = 'ready',
                        claim_lock = NULL,
                        claim_expires = NULL,
                        worker_pid = NULL,
                        last_heartbeat_at = NULL
                  WHERE id = ? AND status = 'running'",
                [task_id.as_str()],
            )
            .map_err(|error| format!("reclaiming crashed worker failed: {error}"))?;
        if updated == 1 {
            let run_id = end_run(
                &tx,
                &task_id,
                "crashed",
                "crashed",
                None,
                Some(&json!({
                    "pid": worker_pid,
                    "claimer": claim_lock,
                })),
                Some(&format!("pid {worker_pid} not alive")),
            )?;
            append_event(
                &tx,
                &task_id,
                "crashed",
                Some(json!({
                    "pid": worker_pid,
                    "claimer": claim_lock,
                })),
                run_id,
            )?;
            tx.commit()
                .map_err(|error| format!("committing crashed-worker reclaim failed: {error}"))?;
            let _ = record_task_failure(
                conn,
                &task_id,
                &format!("pid {worker_pid} not alive"),
                "crashed",
                failure_limit,
                false,
                false,
                Some(json!({
                    "pid": worker_pid,
                    "claimer": claim_lock,
                })),
            )?;
            crashed.push(task_id);
            continue;
        }
        tx.commit()
            .map_err(|error| format!("committing crashed-worker update failed: {error}"))?;
    }
    Ok(crashed)
}

fn enforce_max_runtime(conn: &mut Connection, failure_limit: i64) -> Result<Vec<String>, String> {
    let now = now_ts();
    let host_prefix = format!("{}:", claimer_host());
    let rows = {
        let mut stmt = conn
            .prepare(
                "SELECT t.id,
                        t.worker_pid,
                        COALESCE(r.started_at, t.started_at) AS active_started_at,
                        t.max_runtime_seconds,
                        t.claim_lock
                   FROM tasks t
              LEFT JOIN task_runs r ON r.id = t.current_run_id
                  WHERE t.status = 'running'
                    AND t.max_runtime_seconds IS NOT NULL
                    AND COALESCE(r.started_at, t.started_at) IS NOT NULL
                    AND t.worker_pid IS NOT NULL",
            )
            .map_err(|error| format!("preparing max-runtime query failed: {error}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            })
            .map_err(|error| format!("querying max-runtime candidates failed: {error}"))?;
        collect_rows(rows, "reading max-runtime candidate")?
    };

    let mut timed_out = Vec::new();
    for (task_id, worker_pid, started_at, max_runtime_seconds, claim_lock) in rows {
        if !claim_lock
            .as_deref()
            .unwrap_or_default()
            .starts_with(&host_prefix)
        {
            continue;
        }
        let elapsed = now.saturating_sub(started_at);
        if elapsed < max_runtime_seconds {
            continue;
        }

        let sigkill = terminate_worker_pid(worker_pid);
        let tx = begin_immediate(conn)?;
        let updated = tx
            .execute(
                "UPDATE tasks
                    SET status = 'ready',
                        claim_lock = NULL,
                        claim_expires = NULL,
                        worker_pid = NULL,
                        last_heartbeat_at = NULL
                  WHERE id = ? AND status = 'running'",
                [task_id.as_str()],
            )
            .map_err(|error| format!("releasing timed-out task failed: {error}"))?;
        if updated == 1 {
            let payload = json!({
                "pid": worker_pid,
                "elapsed_seconds": elapsed,
                "limit_seconds": max_runtime_seconds,
                "sigkill": sigkill,
            });
            let run_id = end_run(
                &tx,
                &task_id,
                "timed_out",
                "timed_out",
                None,
                Some(&payload),
                Some(&format!(
                    "elapsed {elapsed}s > limit {max_runtime_seconds}s"
                )),
            )?;
            append_event(&tx, &task_id, "timed_out", Some(payload.clone()), run_id)?;
            tx.commit()
                .map_err(|error| format!("committing timed-out task failed: {error}"))?;
            let _ = record_task_failure(
                conn,
                &task_id,
                &format!("elapsed {elapsed}s > limit {max_runtime_seconds}s"),
                "timed_out",
                failure_limit,
                false,
                false,
                Some(payload),
            )?;
            timed_out.push(task_id);
            continue;
        }
        tx.commit()
            .map_err(|error| format!("committing timed-out update failed: {error}"))?;
    }
    Ok(timed_out)
}

fn default_spawn(
    task: &Task,
    workspace: &Path,
    context: &HermesContext,
) -> Result<Option<u32>, String> {
    let assignee = task
        .assignee
        .as_deref()
        .ok_or_else(|| format!("task {} has no assignee", task.id))?;
    let profile = normalize_profile_name(assignee)
        .ok_or_else(|| format!("task {} has an invalid assignee", task.id))?;
    let current_exe = std::env::current_exe()
        .map_err(|error| format!("resolving current executable failed: {error}"))?;
    let log_path = worker_log_path(&context.hermes_home(), task)?;
    rotate_worker_log(&log_path, DEFAULT_LOG_ROTATE_BYTES)?;
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("opening {} failed: {error}", log_path.display()))?;
    let stderr_file = log_file
        .try_clone()
        .map_err(|error| format!("cloning {} failed: {error}", log_path.display()))?;
    let prompt = kanban_worker_prompt(&task.id, workspace);

    let mut command = Command::new(current_exe);
    command
        .arg("-p")
        .arg(&profile)
        .arg("chat")
        .arg(prompt)
        .arg("--toolset")
        .arg("hermes-cli")
        .arg("--toolset")
        .arg("kanban")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr_file));
    if workspace.is_dir() {
        command.current_dir(workspace);
    }
    command.env("HERMES_KANBAN_TASK", &task.id);
    command.env("HERMES_KANBAN_WORKSPACE", workspace);
    if let Some(run_id) = task.current_run_id {
        command.env("HERMES_KANBAN_RUN_ID", run_id.to_string());
    }
    if let Ok(Some(claim_lock)) = current_claim_lock(&context.hermes_home(), &task.id) {
        command.env("HERMES_KANBAN_CLAIM_LOCK", claim_lock);
    }
    command.env("HERMES_KANBAN_DB", kanban_db_path(&context.hermes_home())?);
    if let Some(board) = current_board_slug(&context.hermes_home())? {
        command.env("HERMES_KANBAN_BOARD", board);
    }
    command.env(
        "HERMES_KANBAN_WORKSPACES_ROOT",
        workspaces_root(&context.hermes_home())?,
    );
    command.env("HERMES_PROFILE", profile);

    let child = command
        .spawn()
        .map_err(|error| format!("spawning kanban worker for {} failed: {error}", task.id))?;
    Ok(Some(child.id()))
}

fn current_claim_lock(hermes_home: &Path, task_id: &str) -> Result<Option<String>, String> {
    let conn = connect_kanban_path(hermes_home)?;
    conn.query_row(
        "SELECT claim_lock FROM tasks WHERE id = ?",
        [task_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .optional()
    .map(|value| value.flatten())
    .map_err(|error| format!("reading claim lock failed: {error}"))
}

fn set_workspace_path(conn: &mut Connection, task_id: &str, path: &Path) -> Result<(), String> {
    let tx = begin_immediate(conn)?;
    tx.execute(
        "UPDATE tasks SET workspace_path = ? WHERE id = ?",
        params![path.display().to_string(), task_id],
    )
    .map_err(|error| format!("persisting workspace path failed: {error}"))?;
    tx.commit()
        .map_err(|error| format!("committing workspace path failed: {error}"))?;
    Ok(())
}

fn resolve_workspace(task: &Task, hermes_home: &Path) -> Result<PathBuf, String> {
    match task.workspace_kind.as_str() {
        "scratch" => {
            let path = if let Some(raw) = task.workspace_path.as_deref() {
                let path = expand_user_path(raw)?;
                if !path.is_absolute() {
                    return Err(format!(
                        "task {} has non-absolute workspace_path {:?}; workspace paths must be absolute",
                        task.id, raw
                    ));
                }
                path
            } else {
                workspaces_root(hermes_home)?.join(&task.id)
            };
            fs::create_dir_all(&path)
                .map_err(|error| format!("creating {} failed: {error}", path.display()))?;
            Ok(path)
        }
        "dir" => {
            let raw = task.workspace_path.as_deref().ok_or_else(|| {
                format!(
                    "task {} has workspace_kind=dir but no workspace_path",
                    task.id
                )
            })?;
            let path = expand_user_path(raw)?;
            if !path.is_absolute() {
                return Err(format!(
                    "task {} has non-absolute workspace_path {:?}; use an absolute path",
                    task.id, raw
                ));
            }
            fs::create_dir_all(&path)
                .map_err(|error| format!("creating {} failed: {error}", path.display()))?;
            Ok(path)
        }
        "worktree" => {
            if let Some(raw) = task.workspace_path.as_deref() {
                let path = expand_user_path(raw)?;
                if !path.is_absolute() {
                    return Err(format!(
                        "task {} has non-absolute worktree path {:?}; use an absolute path",
                        task.id, raw
                    ));
                }
                Ok(path)
            } else {
                Ok(std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(".worktrees")
                    .join(&task.id))
            }
        }
        other => Err(format!("unknown workspace_kind: {other}")),
    }
}

fn workspaces_root(hermes_home: &Path) -> Result<PathBuf, String> {
    if let Some(path) = env::var("HERMES_KANBAN_WORKSPACES_ROOT")
        .ok()
        .and_then(non_empty_trimmed)
    {
        return expand_user_path(&path);
    }
    let root = kanban_root(hermes_home);
    let board = current_board_slug(hermes_home)?.unwrap_or_else(|| DEFAULT_BOARD.to_string());
    if board == DEFAULT_BOARD {
        Ok(root.join("kanban").join("workspaces"))
    } else {
        Ok(root
            .join("kanban")
            .join("boards")
            .join(board)
            .join("workspaces"))
    }
}

fn worker_log_path(hermes_home: &Path, task: &Task) -> Result<PathBuf, String> {
    let root = kanban_root(hermes_home);
    let board = current_board_slug(hermes_home)?.unwrap_or_else(|| DEFAULT_BOARD.to_string());
    let dir = if board == DEFAULT_BOARD {
        root.join("kanban").join("logs")
    } else {
        root.join("kanban").join("boards").join(board).join("logs")
    };
    fs::create_dir_all(&dir)
        .map_err(|error| format!("creating {} failed: {error}", dir.display()))?;
    Ok(dir.join(format!("{}.log", task.id)))
}

fn rotate_worker_log(path: &Path, max_bytes: u64) -> Result<(), String> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() <= max_bytes {
        return Ok(());
    }
    let rotated = PathBuf::from(format!("{}.1", path.display()));
    if rotated.exists() {
        let _ = fs::remove_file(&rotated);
    }
    fs::rename(path, &rotated)
        .map_err(|error| format!("rotating {} failed: {error}", path.display()))?;
    Ok(())
}

pub fn worker_log_path_for_task(hermes_home: &Path, task_id: &str) -> Result<PathBuf, String> {
    let task_id =
        non_empty_trimmed(task_id.to_string()).ok_or_else(|| "task_id is required".to_string())?;
    let root = kanban_root(hermes_home);
    let board = current_board_slug(hermes_home)?.unwrap_or_else(|| DEFAULT_BOARD.to_string());
    let dir = if board == DEFAULT_BOARD {
        root.join("kanban").join("logs")
    } else {
        root.join("kanban").join("boards").join(board).join("logs")
    };
    Ok(dir.join(format!("{task_id}.log")))
}

pub fn read_worker_log(
    hermes_home: &Path,
    task_id: &str,
    tail_bytes: Option<usize>,
) -> Result<Option<String>, String> {
    let path = worker_log_path_for_task(hermes_home, task_id)?;
    let Ok(bytes) = fs::read(&path) else {
        return Ok(None);
    };
    let slice = if let Some(limit) = tail_bytes {
        let start = bytes.len().saturating_sub(limit);
        &bytes[start..]
    } else {
        &bytes[..]
    };
    Ok(Some(String::from_utf8_lossy(slice).to_string()))
}

pub fn gc_worker_logs(hermes_home: &Path, older_than_seconds: i64) -> Result<usize, String> {
    let cutoff = now_ts().saturating_sub(older_than_seconds.max(0)) as u64;
    let dir = worker_log_path_for_task(hermes_home, "placeholder")?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "worker logs directory is unavailable".to_string())?;
    let mut removed = 0_usize;
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(0);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let modified = match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Ok(seconds) = modified.duration_since(UNIX_EPOCH) else {
            continue;
        };
        if seconds.as_secs() <= cutoff {
            if fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

pub fn gc_events(conn: &mut Connection, older_than_seconds: i64) -> Result<usize, String> {
    let cutoff = now_ts().saturating_sub(older_than_seconds.max(0));
    conn.execute(
        "DELETE FROM task_events
          WHERE created_at < ?
            AND task_id IN (
                SELECT id FROM tasks WHERE status IN ('done', 'archived')
            )",
        [cutoff],
    )
    .map_err(|error| format!("garbage-collecting task events failed: {error}"))
}

pub fn add_notify_sub(
    conn: &mut Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
) -> Result<(), String> {
    let task_id =
        non_empty_trimmed(task_id.to_string()).ok_or_else(|| "task_id is required".to_string())?;
    let platform = non_empty_trimmed(platform.to_string())
        .ok_or_else(|| "platform is required".to_string())?;
    let chat_id =
        non_empty_trimmed(chat_id.to_string()).ok_or_else(|| "chat_id is required".to_string())?;
    let thread_id = thread_id.unwrap_or("").trim().to_string();
    conn.execute(
        "INSERT INTO kanban_notify_subs (
            task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(task_id, platform, chat_id, thread_id)
        DO UPDATE SET user_id = excluded.user_id",
        params![task_id, platform, chat_id, thread_id, user_id, now_ts()],
    )
    .map_err(|error| format!("adding notify subscription failed: {error}"))?;
    Ok(())
}

pub fn list_notify_subs(
    conn: &Connection,
    task_id: Option<&str>,
) -> Result<Vec<KanbanNotifySubscription>, String> {
    let (sql, params): (&str, Vec<String>) = if let Some(task_id) = task_id {
        (
            "SELECT task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id
               FROM kanban_notify_subs
              WHERE task_id = ?
              ORDER BY task_id, platform, chat_id, thread_id",
            vec![task_id.to_string()],
        )
    } else {
        (
            "SELECT task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id
               FROM kanban_notify_subs
              ORDER BY task_id, platform, chat_id, thread_id",
            Vec::new(),
        )
    };
    let mut stmt = conn
        .prepare(sql)
        .map_err(|error| format!("preparing notify subscription query failed: {error}"))?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            Ok(KanbanNotifySubscription {
                task_id: row.get(0)?,
                platform: row.get(1)?,
                chat_id: row.get(2)?,
                thread_id: row.get(3)?,
                user_id: row.get(4)?,
                created_at: row.get(5)?,
                last_event_id: row.get(6)?,
            })
        })
        .map_err(|error| format!("querying notify subscriptions failed: {error}"))?;
    collect_rows(rows, "reading notify subscription")
}

pub fn remove_notify_sub(
    conn: &mut Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
) -> Result<bool, String> {
    let updated = conn
        .execute(
            "DELETE FROM kanban_notify_subs
              WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            params![task_id, platform, chat_id, thread_id.unwrap_or("")],
        )
        .map_err(|error| format!("removing notify subscription failed: {error}"))?;
    Ok(updated > 0)
}

fn record_task_failure(
    conn: &mut Connection,
    task_id: &str,
    error: &str,
    outcome: &str,
    failure_limit: i64,
    release_claim: bool,
    end_active_run: bool,
    event_payload_extra: Option<Value>,
) -> Result<bool, String> {
    let tx = begin_immediate(conn)?;
    let row = tx
        .query_row(
            "SELECT consecutive_failures, status FROM tasks WHERE id = ?",
            [task_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|error| format!("reading task failure state failed: {error}"))?;
    let Some((prior_failures, status)) = row else {
        return Ok(false);
    };
    let failures = prior_failures + 1;
    let bounded_error = truncate_line(error, 500);

    let mut blocked = false;
    if failures >= failure_limit {
        blocked = true;
        if release_claim {
            tx.execute(
                "UPDATE tasks
                    SET status = 'blocked',
                        claim_lock = NULL,
                        claim_expires = NULL,
                        worker_pid = NULL,
                        consecutive_failures = ?,
                        last_failure_error = ?
                  WHERE id = ? AND status IN ('running', 'ready')",
                params![failures, bounded_error, task_id],
            )
            .map_err(|error| format!("blocking failed task failed: {error}"))?;
        } else {
            tx.execute(
                "UPDATE tasks
                    SET status = 'blocked',
                        consecutive_failures = ?,
                        last_failure_error = ?
                  WHERE id = ? AND status IN ('ready', 'running', 'blocked')",
                params![failures, bounded_error, task_id],
            )
            .map_err(|error| format!("blocking task after repeated failures failed: {error}"))?;
        }
        let run_id = if end_active_run {
            end_run(
                &tx,
                task_id,
                "gave_up",
                "gave_up",
                None,
                Some(&json!({
                    "failures": failures,
                    "trigger_outcome": outcome,
                })),
                Some(&bounded_error),
            )?
        } else {
            None
        };
        let mut payload = Map::new();
        payload.insert("failures".to_string(), json!(failures));
        payload.insert("error".to_string(), Value::String(bounded_error.clone()));
        payload.insert(
            "trigger_outcome".to_string(),
            Value::String(outcome.to_string()),
        );
        if let Some(extra) = event_payload_extra
            && let Some(object) = extra.as_object()
        {
            for (key, value) in object {
                payload.insert(key.clone(), value.clone());
            }
        }
        append_event(
            &tx,
            task_id,
            "gave_up",
            Some(Value::Object(payload)),
            run_id,
        )?;
    } else {
        if release_claim {
            tx.execute(
                "UPDATE tasks
                    SET status = 'ready',
                        claim_lock = NULL,
                        claim_expires = NULL,
                        worker_pid = NULL,
                        consecutive_failures = ?,
                        last_failure_error = ?
                  WHERE id = ? AND status = 'running'",
                params![failures, bounded_error, task_id],
            )
            .map_err(|error| format!("releasing failed task failed: {error}"))?;
        } else {
            tx.execute(
                "UPDATE tasks
                    SET consecutive_failures = ?,
                        last_failure_error = ?
                  WHERE id = ?",
                params![failures, bounded_error, task_id],
            )
            .map_err(|error| format!("updating failure counter failed: {error}"))?;
        }
        if end_active_run {
            let run_id = end_run(
                &tx,
                task_id,
                outcome,
                outcome,
                None,
                Some(&json!({"failures": failures})),
                Some(&bounded_error),
            )?;
            append_event(
                &tx,
                task_id,
                outcome,
                Some(json!({
                    "error": bounded_error,
                    "failures": failures,
                })),
                run_id,
            )?;
        }
    }
    if status == "done" {
        return Ok(false);
    }
    tx.commit()
        .map_err(|error| format!("committing task failure failed: {error}"))?;
    Ok(blocked)
}

fn clear_failure_counter(conn: &Connection, task_id: &str) -> Result<(), String> {
    conn.execute(
        "UPDATE tasks
            SET consecutive_failures = 0,
                last_failure_error = NULL
          WHERE id = ?",
        [task_id],
    )
    .map_err(|error| format!("clearing failure counter failed: {error}"))?;
    Ok(())
}

fn set_worker_pid(conn: &mut Connection, task_id: &str, pid: u32) -> Result<(), String> {
    let tx = begin_immediate(conn)?;
    tx.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?",
        params![i64::from(pid), task_id],
    )
    .map_err(|error| format!("recording worker pid failed: {error}"))?;
    let run_id = current_run_id(&tx, task_id)?;
    if let Some(run_id) = run_id {
        tx.execute(
            "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
            params![i64::from(pid), run_id],
        )
        .map_err(|error| format!("recording run worker pid failed: {error}"))?;
    }
    append_event(&tx, task_id, "spawned", Some(json!({"pid": pid})), run_id)?;
    tx.commit()
        .map_err(|error| format!("committing worker pid failed: {error}"))?;
    Ok(())
}

fn has_spawnable_ready(conn: &Connection, context: &HermesContext) -> Result<bool, String> {
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT assignee FROM tasks
             WHERE status = 'ready'
               AND assignee IS NOT NULL
               AND claim_lock IS NULL",
        )
        .map_err(|error| format!("preparing spawnable-ready query failed: {error}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| format!("querying ready assignees failed: {error}"))?;
    let assignees = collect_string_rows(rows, "reading ready assignee")?;
    if assignees.is_empty() {
        return Ok(false);
    }
    Ok(assignees
        .iter()
        .any(|assignee| context.profile_exists(assignee)))
}

fn begin_immediate(conn: &mut Connection) -> Result<Transaction<'_>, String> {
    conn.transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| format!("starting write transaction failed: {error}"))
}

fn current_run_id(conn: &Connection, task_id: &str) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT current_run_id FROM tasks WHERE id = ?",
        [task_id],
        |row| row.get::<_, Option<i64>>(0),
    )
    .optional()
    .map(|value| value.flatten())
    .map_err(|error| format!("reading current run id failed: {error}"))
}

fn end_run(
    conn: &Connection,
    task_id: &str,
    status: &str,
    outcome: &str,
    summary: Option<&str>,
    metadata: Option<&Value>,
    error: Option<&str>,
) -> Result<Option<i64>, String> {
    let run_id = current_run_id(conn, task_id)?;
    let Some(run_id) = run_id else {
        return Ok(None);
    };
    conn.execute(
        "UPDATE task_runs SET status = ?, outcome = ?, summary = ?, error = ?, metadata = ?, ended_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ? AND ended_at IS NULL",
        params![
            status,
            outcome,
            summary,
            error,
            metadata.map(|value| serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())),
            now_ts(),
            run_id,
        ],
    )
    .map_err(|error| format!("closing run failed: {error}"))?;
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?",
        [task_id],
    )
    .map_err(|error| format!("clearing current run failed: {error}"))?;
    Ok(Some(run_id))
}

fn synthesize_ended_run(
    conn: &Connection,
    task_id: &str,
    status: &str,
    outcome: &str,
    summary: Option<&str>,
    metadata: Option<&Value>,
    error: Option<&str>,
) -> Result<i64, String> {
    let row = conn
        .query_row(
            "SELECT assignee, current_step_key FROM tasks WHERE id = ?",
            [task_id],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                ))
            },
        )
        .map_err(|error| format!("reading task for synthetic run failed: {error}"))?;
    let now = now_ts();
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, outcome, summary, error, metadata, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            task_id,
            row.0,
            row.1,
            status,
            outcome,
            summary,
            error,
            metadata.map(|value| serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())),
            now,
            now,
        ],
    )
    .map_err(|error| format!("creating synthetic run failed: {error}"))?;
    Ok(conn.last_insert_rowid())
}

fn verify_created_cards(
    conn: &Connection,
    completing_task_id: &str,
    claimed_ids: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let claimed = dedupe_strings(claimed_ids.to_vec());
    if claimed.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let assignee = conn
        .query_row(
            "SELECT assignee FROM tasks WHERE id = ?",
            [completing_task_id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .map_err(|error| format!("reading completing task failed: {error}"))?
        .flatten();
    let Some(assignee) = assignee else {
        return Ok((Vec::new(), claimed));
    };

    let placeholders = std::iter::repeat_n("?", claimed.len())
        .collect::<Vec<_>>()
        .join(", ");
    let mut stmt = conn
        .prepare(&format!(
            "SELECT id, created_by FROM tasks WHERE id IN ({placeholders})"
        ))
        .map_err(|error| format!("preparing created-cards query failed: {error}"))?;
    let rows = stmt
        .query_map(params_from_iter(claimed.iter()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })
        .map_err(|error| format!("querying created cards failed: {error}"))?;
    let found = collect_rows(rows, "reading created card")?
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    let linked_children = child_ids(conn, completing_task_id)?
        .into_iter()
        .collect::<HashSet<_>>();

    let mut verified = Vec::new();
    let mut phantom = Vec::new();
    for claimed_id in claimed {
        let Some(created_by) = found.get(&claimed_id) else {
            phantom.push(claimed_id);
            continue;
        };
        if created_by.as_deref() == Some(&assignee)
            || created_by.as_deref() == Some(completing_task_id)
            || linked_children.contains(&claimed_id)
        {
            verified.push(claimed_id);
        } else {
            phantom.push(claimed_id);
        }
    }
    Ok((verified, phantom))
}

fn would_cycle(conn: &Connection, parent_id: &str, child_id: &str) -> Result<bool, String> {
    let mut seen = HashSet::new();
    let mut stack = vec![child_id.to_string()];
    while let Some(node) = stack.pop() {
        if node == parent_id {
            return Ok(true);
        }
        if !seen.insert(node.clone()) {
            continue;
        }
        stack.extend(child_ids(conn, &node)?);
    }
    Ok(false)
}

fn find_missing_parents(conn: &Connection, parents: &[String]) -> Result<Vec<String>, String> {
    if parents.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = std::iter::repeat_n("?", parents.len())
        .collect::<Vec<_>>()
        .join(", ");
    let mut stmt = conn
        .prepare(&format!(
            "SELECT id FROM tasks WHERE id IN ({placeholders})"
        ))
        .map_err(|error| format!("preparing parent lookup failed: {error}"))?;
    let rows = stmt
        .query_map(params_from_iter(parents.iter()), |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("querying parents failed: {error}"))?;
    let present = collect_string_rows(rows, "reading parent task id")?
        .into_iter()
        .collect::<HashSet<_>>();
    Ok(parents
        .iter()
        .filter(|parent| !present.contains(parent.as_str()))
        .cloned()
        .collect())
}

fn all_parents_done(conn: &Connection, parents: &[String]) -> Result<bool, String> {
    if parents.is_empty() {
        return Ok(true);
    }
    let placeholders = std::iter::repeat_n("?", parents.len())
        .collect::<Vec<_>>()
        .join(", ");
    let mut stmt = conn
        .prepare(&format!(
            "SELECT status FROM tasks WHERE id IN ({placeholders})"
        ))
        .map_err(|error| format!("preparing parent status query failed: {error}"))?;
    let rows = stmt
        .query_map(params_from_iter(parents.iter()), |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("querying parent statuses failed: {error}"))?;
    for status in collect_string_rows(rows, "reading parent status")? {
        if status != "done" {
            return Ok(false);
        }
    }
    Ok(true)
}

fn append_event(
    conn: &Connection,
    task_id: &str,
    kind: &str,
    payload: Option<Value>,
    run_id: Option<i64>,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        params![
            task_id,
            run_id,
            kind,
            payload.map(|value| serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())),
            now_ts(),
        ],
    )
    .map_err(|error| format!("appending event {kind} failed: {error}"))?;
    Ok(())
}

fn kanban_enabled_in_config(hermes_home: &Path) -> bool {
    let path = hermes_home.join("config.yaml");
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(parsed) = serde_yaml::from_str::<YamlValue>(&contents) else {
        return false;
    };
    parsed
        .as_mapping()
        .and_then(|mapping| mapping.get(YamlValue::String("toolsets".to_string())))
        .and_then(YamlValue::as_sequence)
        .is_some_and(|sequence| {
            sequence.iter().any(|value| {
                value
                    .as_str()
                    .is_some_and(|text| text.trim().eq_ignore_ascii_case("kanban"))
            })
        })
}

fn kanban_db_path(hermes_home: &Path) -> Result<PathBuf, String> {
    if let Some(path) = env::var("HERMES_KANBAN_DB")
        .ok()
        .and_then(non_empty_trimmed)
    {
        return Ok(PathBuf::from(path));
    }

    let root = kanban_root(hermes_home);
    let board = env::var("HERMES_KANBAN_BOARD")
        .ok()
        .and_then(non_empty_trimmed)
        .map(|value| normalize_board_slug(Some(&value)))
        .transpose()?
        .flatten()
        .or_else(|| read_current_board(&root).ok().flatten())
        .unwrap_or_else(|| DEFAULT_BOARD.to_string());
    if board == DEFAULT_BOARD {
        Ok(root.join("kanban.db"))
    } else {
        Ok(root
            .join("kanban")
            .join("boards")
            .join(board)
            .join("kanban.db"))
    }
}

fn kanban_root(hermes_home: &Path) -> PathBuf {
    if let Some(path) = env::var("HERMES_KANBAN_HOME")
        .ok()
        .and_then(non_empty_trimmed)
    {
        return PathBuf::from(path);
    }
    shared_root_from_home(hermes_home)
}

fn shared_root_from_home(hermes_home: &Path) -> PathBuf {
    if hermes_home
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "profiles")
        && let Some(root) = hermes_home.parent().and_then(Path::parent)
    {
        return root.to_path_buf();
    }
    hermes_home.to_path_buf()
}

fn read_current_board(root: &Path) -> Result<Option<String>, String> {
    let path = root.join("kanban").join("current");
    let contents = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    normalize_board_slug(Some(contents.trim()))
}

fn current_board_slug(hermes_home: &Path) -> Result<Option<String>, String> {
    if let Some(board) = env::var("HERMES_KANBAN_BOARD")
        .ok()
        .and_then(non_empty_trimmed)
    {
        return normalize_board_slug(Some(&board));
    }
    read_current_board(&kanban_root(hermes_home))
}

fn normalize_board_slug(raw: Option<&str>) -> Result<Option<String>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let slug = raw.trim().to_ascii_lowercase();
    if slug.is_empty() {
        return Ok(None);
    }
    if slug.len() > 64 {
        return Err("board slug must be 64 chars or fewer".to_string());
    }
    let mut chars = slug.chars();
    let Some(first) = chars.next() else {
        return Ok(None);
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(format!("invalid board slug {slug:?}"));
    }
    if !slug
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_'))
    {
        return Err(format!("invalid board slug {slug:?}"));
    }
    Ok(Some(slug))
}

fn kanban_claim_ttl_seconds(loaded: &LoadedConfig) -> i64 {
    loaded
        .cfg_get(&["kanban", "claim_ttl_seconds"])
        .and_then(YamlValue::as_i64)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CLAIM_TTL_SECONDS)
}

fn kanban_failure_limit(loaded: &LoadedConfig) -> i64 {
    loaded
        .cfg_get(&["kanban", "failure_limit"])
        .and_then(YamlValue::as_i64)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_FAILURE_LIMIT)
}

fn default_hermes_home() -> PathBuf {
    env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

fn default_task_id(task_id: Option<String>) -> Result<String, String> {
    task_id
        .or_else(|| {
            env::var("HERMES_KANBAN_TASK")
                .ok()
                .and_then(non_empty_trimmed)
        })
        .ok_or_else(|| "task_id is required (or set HERMES_KANBAN_TASK in the env)".to_string())
}

fn worker_run_id(task_id: &str) -> Option<i64> {
    if env::var("HERMES_KANBAN_TASK")
        .ok()
        .and_then(non_empty_trimmed)
        .as_deref()
        != Some(task_id)
    {
        return None;
    }
    env::var("HERMES_KANBAN_RUN_ID")
        .ok()
        .and_then(non_empty_trimmed)
        .and_then(|value| value.parse::<i64>().ok())
}

fn enforce_worker_task_ownership(task_id: &str) -> Option<String> {
    let Some(env_task) = env::var("HERMES_KANBAN_TASK")
        .ok()
        .and_then(non_empty_trimmed)
    else {
        return None;
    };
    if env_task == task_id {
        None
    } else {
        Some(tool_error(format!(
            "worker is scoped to task {env_task}; refusing to mutate {task_id}. Use kanban_comment to hand off information to other tasks, or kanban_create to spawn follow-up work."
        )))
    }
}

fn normalize_profile_name(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_ascii_lowercase())
    }
}

fn new_task_id() -> String {
    let seq = TASK_ID_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("t_{:x}{:04x}", unix_ts_nanos(), seq & 0xffff)
}

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn unix_ts_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn format_ts(timestamp: i64) -> String {
    Local
        .timestamp_opt(timestamp, 0)
        .single()
        .map(|time| time.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| timestamp.to_string())
}

fn truncate_line(value: &str, limit: usize) -> String {
    value
        .trim()
        .split('\n')
        .next()
        .unwrap_or_default()
        .chars()
        .take(limit)
        .collect()
}

fn cap_text(value: &str, limit: usize) -> String {
    let trimmed = value.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_string();
    }
    let prefix = trimmed.chars().take(limit).collect::<String>();
    format!("{prefix}… [truncated]")
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for value in values {
        let Some(value) = non_empty_trimmed(value) else {
            continue;
        };
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }
    deduped
}

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&Row<'_>) -> rusqlite::Result<T>>,
    action: &str,
) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    for row in rows {
        values.push(row.map_err(|error| format!("{action} failed: {error}"))?);
    }
    Ok(values)
}

fn collect_string_rows(
    rows: rusqlite::MappedRows<'_, impl FnMut(&Row<'_>) -> rusqlite::Result<String>>,
    action: &str,
) -> Result<Vec<String>, String> {
    collect_rows(rows, action)
}

fn optional_string(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .and_then(|value| non_empty_trimmed(value.to_string()))
}

fn optional_non_empty_string(args: &Value, key: &str) -> Option<String> {
    optional_string(args, key)
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    optional_string(args, key).ok_or_else(|| format!("{key} is required"))
}

fn optional_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn optional_i64(args: &Value, key: &str) -> Result<Option<i64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .ok_or_else(|| format!("{key} must be an integer"))
            .map(Some),
        Some(_) => Err(format!("{key} must be an integer")),
    }
}

fn optional_object(args: &Value, key: &str) -> Result<Option<Value>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(Value::Object(object.clone()))),
        Some(_) => Err(format!("{key} must be an object")),
    }
}

fn optional_string_list(args: &Value, key: &str) -> Result<Option<Vec<String>>, String> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(vec![text.to_string()])),
        Value::Array(items) => {
            let mut values = Vec::new();
            for item in items {
                let Some(text) = item.as_str() else {
                    return Err(format!("{key} must contain only strings"));
                };
                values.push(text.to_string());
            }
            Ok(Some(values))
        }
        _ => Err(format!("{key} must be a string or array of strings")),
    }
}

fn non_empty_trimmed(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn claimer_host() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .and_then(non_empty_trimmed)
        .or_else(|| nixless_hostname().ok().and_then(non_empty_trimmed))
        .unwrap_or_else(|| "unknown".to_string())
}

fn claimer_id() -> String {
    format!("{}:{}", claimer_host(), std::process::id())
}

fn nixless_hostname() -> Result<String, String> {
    fs::read_to_string("/etc/hostname")
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("reading hostname failed: {error}"))
}

fn expand_user_path(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("path must not be empty".to_string());
    }
    if trimmed == "~" {
        return dirs::home_dir().ok_or_else(|| "home directory is unavailable".to_string());
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        return dirs::home_dir()
            .ok_or_else(|| "home directory is unavailable".to_string())
            .map(|home| home.join(rest));
    }
    Ok(PathBuf::from(trimmed))
}

fn kanban_worker_prompt(task_id: &str, workspace: &Path) -> String {
    format!(
        "Execute kanban task {task_id}. Call kanban_show first, work inside {} unless the task explicitly requires otherwise, heartbeat during long-running work, and finish with kanban_complete(summary=..., metadata=...) or kanban_block(reason=...).",
        workspace.display()
    )
}

#[cfg(unix)]
fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid as i32, 0) };
    if rc == 0 {
        true
    } else {
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

#[cfg(not(unix))]
fn pid_alive(_pid: i64) -> bool {
    true
}

#[cfg(unix)]
fn terminate_worker_pid(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    let _ = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    for _ in 0..TERM_GRACE_POLL_COUNT {
        if !pid_alive(pid) {
            return false;
        }
        sleep(Duration::from_millis(TERM_GRACE_POLL_MILLIS));
    }
    let rc = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    rc == 0
}

#[cfg(not(unix))]
fn terminate_worker_pid(_pid: i64) -> bool {
    false
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    current_run_id       INTEGER,
    workflow_template_id TEXT,
    current_step_key     TEXT,
    skills               TEXT
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn kanban_available_reads_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "toolsets:\n  - hermes-cli\n  - kanban\n",
        )
        .unwrap();

        let old_home = env::var_os("HERMES_HOME");
        let old_task = env::var_os("HERMES_KANBAN_TASK");
        unsafe {
            env::set_var("HERMES_HOME", temp.path());
            env::remove_var("HERMES_KANBAN_TASK");
        }
        assert!(kanban_available());
        match old_home {
            Some(value) => unsafe { env::set_var("HERMES_HOME", value) },
            None => unsafe { env::remove_var("HERMES_HOME") },
        }
        match old_task {
            Some(value) => unsafe { env::set_var("HERMES_KANBAN_TASK", value) },
            None => unsafe { env::remove_var("HERMES_KANBAN_TASK") },
        }
    }

    #[test]
    fn kanban_create_comment_complete_promotes_child() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("profiles").join("dev");
        fs::create_dir_all(&home).unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(&home);
        let mut conn = connect_kanban(&runtime).unwrap();

        let parent_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Parent".to_string(),
                body: Some("Research this change".to_string()),
                assignee: Some("researcher".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();
        let child_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Child".to_string(),
                body: Some("Use the parent handoff".to_string()),
                assignee: Some("writer".to_string()),
                parents: vec![parent_id.clone()],
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        assert_eq!(get_task(&conn, &child_id).unwrap().unwrap().status, "todo");
        add_comment(&mut conn, &child_id, "worker", "Wait for the parent result").unwrap();
        assert!(
            complete_task(
                &mut conn,
                &parent_id,
                None,
                Some("Parent finished with useful context"),
                Some(&json!({"changed_files":["src/lib.rs"]})),
                &[],
                None,
            )
            .unwrap()
        );

        let child = get_task(&conn, &child_id).unwrap().unwrap();
        assert_eq!(child.status, "ready");
        let context = build_worker_context(&conn, &child_id).unwrap();
        assert!(context.contains("Parent finished with useful context"));
        assert!(context.contains("Wait for the parent result"));
    }

    #[test]
    fn kanban_heartbeat_and_block_update_running_task() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let mut conn = connect_kanban(&runtime).unwrap();
        let task_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Running".to_string(),
                body: None,
                assignee: Some("worker".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        let tx = begin_immediate(&mut conn).unwrap();
        tx.execute(
            "UPDATE tasks SET status = 'running', started_at = ?, claim_lock = 'host:1', claim_expires = ?, worker_pid = 123 WHERE id = ?",
            params![now_ts(), now_ts() + 600, task_id],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, started_at) VALUES (?, ?, 'running', 'host:1', ?, ?)",
            params![task_id, "worker", now_ts() + 600, now_ts()],
        )
        .unwrap();
        let run_id = tx.last_insert_rowid();
        tx.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            params![run_id, task_id],
        )
        .unwrap();
        tx.commit().unwrap();

        assert!(
            heartbeat_worker(&mut conn, &task_id, Some("still working"), Some(run_id)).unwrap()
        );
        assert!(block_task(&mut conn, &task_id, "need input", Some(run_id)).unwrap());
        let latest = latest_run(&conn, &task_id).unwrap().unwrap();
        assert_eq!(latest.outcome.as_deref(), Some("blocked"));
        assert_eq!(latest.summary.as_deref(), Some("need input"));
        assert_eq!(
            get_task(&conn, &task_id).unwrap().unwrap().status,
            "blocked"
        );
    }

    #[test]
    fn kanban_complete_rejects_phantom_created_cards() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let mut conn = connect_kanban(&runtime).unwrap();
        let task_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Complete".to_string(),
                body: None,
                assignee: Some("worker".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        let error = complete_task(
            &mut conn,
            &task_id,
            None,
            Some("done"),
            None,
            &[String::from("t_deadbeef")],
            None,
        )
        .unwrap_err();
        assert!(error.contains("kanban_complete blocked"));
        let events = list_events(&conn, &task_id).unwrap();
        assert!(
            events
                .iter()
                .any(|event| event.kind == "completion_blocked_hallucination")
        );
    }

    #[test]
    fn kanban_dispatch_dry_run_lists_ready_task_without_claiming() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        context.create_profile("worker").unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let mut conn = connect_kanban(&runtime).unwrap();
        let task_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Dispatch me".to_string(),
                body: None,
                assignee: Some("worker".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        let result = dispatch_kanban_once_with_spawn(
            &context,
            &loaded,
            KanbanDispatchOptions {
                dry_run: true,
                max_spawn: None,
                failure_limit: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(result.spawned.len(), 1);
        assert_eq!(result.spawned[0].task_id, task_id);
        let task = get_task(&conn, &task_id).unwrap().unwrap();
        assert_eq!(task.status, "ready");
        assert!(task.current_run_id.is_none());
    }

    #[test]
    fn kanban_dispatch_claims_and_records_spawned_pid() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        context.create_profile("worker").unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let mut conn = connect_kanban(&runtime).unwrap();
        let task_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Spawn me".to_string(),
                body: None,
                assignee: Some("worker".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        let expected_task_id = task_id.clone();
        let spawn = move |task: &Task, workspace: &Path| -> Result<Option<u32>, String> {
            assert_eq!(task.id, expected_task_id);
            assert!(workspace.ends_with(&expected_task_id));
            Ok(Some(4242))
        };

        let result = dispatch_kanban_once_with_spawn(
            &context,
            &loaded,
            KanbanDispatchOptions::default(),
            Some(&spawn),
        )
        .unwrap();

        assert_eq!(result.spawned.len(), 1);
        assert_eq!(result.spawned[0].pid, Some(4242));
        let task = get_task(&conn, &task_id).unwrap().unwrap();
        assert_eq!(task.status, "running");
        let row = conn
            .query_row(
                "SELECT worker_pid, claim_lock, current_run_id FROM tasks WHERE id = ?",
                [task_id.as_str()],
                |row| {
                    Ok((
                        row.get::<_, Option<i64>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row.0, Some(4242));
        assert!(row.1.is_some());
        assert!(row.2.is_some());
    }

    #[test]
    fn kanban_dispatch_spawn_failure_trips_circuit_breaker() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "kanban:\n  failure_limit: 1\n",
        )
        .unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        context.create_profile("worker").unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let mut conn = connect_kanban(&runtime).unwrap();
        let task_id = create_task(
            &mut conn,
            CreateTaskInput {
                title: "Fail me".to_string(),
                body: None,
                assignee: Some("worker".to_string()),
                parents: Vec::new(),
                tenant: None,
                priority: 0,
                workspace_kind: "scratch".to_string(),
                workspace_path: None,
                triage: false,
                idempotency_key: None,
                max_runtime_seconds: None,
                skills: None,
                created_by: "worker".to_string(),
            },
        )
        .unwrap();

        let spawn = |_task: &Task, _workspace: &Path| -> Result<Option<u32>, String> {
            Err("spawn exploded".to_string())
        };

        let result = dispatch_kanban_once_with_spawn(
            &context,
            &loaded,
            KanbanDispatchOptions::default(),
            Some(&spawn),
        )
        .unwrap();

        assert_eq!(result.auto_blocked, vec![task_id.clone()]);
        let task = get_task(&conn, &task_id).unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        let events = list_events(&conn, &task_id).unwrap();
        assert!(events.iter().any(|event| event.kind == "gave_up"));
    }
}
