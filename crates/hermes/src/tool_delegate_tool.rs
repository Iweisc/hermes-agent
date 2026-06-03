//! Delegate Tool -- Subagent Architecture
//!
//! Spawns child AIAgent instances with isolated context, restricted toolsets,
//! and their own terminal sessions. Supports single-task and batch (parallel)
//! modes. The parent blocks until all children complete.
//!
//! Each child gets:
//!   - A fresh conversation (no parent history)
//!   - Its own task_id (own terminal session, file ops cache)
//!   - A restricted toolset (configurable, with blocked tools always stripped)
//!   - A focused system prompt built from the delegated goal + context
//!
//! The parent's context only sees the delegation call and the summary result,
//! never the child's intermediate tool calls or reasoning.
//!
//! Faithful port of `tools/delegate_tool.py`.
//!
//! The Python module spawns live `AIAgent` instances inside a thread pool. The
//! native AIAgent runtime is owned by other crates and is not available here as
//! a flat dependency, so this port reproduces the **portable, deterministic**
//! surface of the module verbatim:
//!   - delegation config readers (depth, concurrency, timeout, kill-switch …)
//!   - role normalisation
//!   - child system-prompt construction
//!   - toolset stripping / MCP preservation
//!   - delegation event types + legacy event mapping
//!   - the module-level pause flag and active-subagent registry
//!   - output-tail extraction and error-output detection
//!   - credential resolution (`_resolve_delegation_credentials`)
//!   - the OpenAI function-calling schema and its dynamic refresh
//!
//! Pieces that strictly require a live `AIAgent` (`_build_child_agent`,
//! `_run_single_child`, the executor loop in `delegate_task`) are exposed via
//! traits/parameters so a host crate can wire in its agent factory without
//! this module taking a heavy dependency.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Map, Value};

// The Python module imports `base_url_hostname` and `is_truthy_value` from
// `utils`. The native equivalents live in `hermes_core::mod_utils`, but that
// module is private to that crate (not re-exported). To keep this file a
// dependency-free flat module, the two small helpers are reproduced here
// verbatim from `crates/hermes-core/src/mod_utils.rs`.

/// Shared truthy string set (port of `utils.is_truthy_value`'s string branch).
const TRUTHY_STRINGS: &[&str] = &["1", "true", "yes", "on"];

/// Trim+lowercase membership test against [`TRUTHY_STRINGS`].
fn is_truthy_str(s: &str) -> bool {
    let normalized = s.trim().to_lowercase();
    TRUTHY_STRINGS.contains(&normalized.as_str())
}

/// Extract a lowercased hostname from a base URL. Bare hosts (no `://`) are
/// parsed by prefixing `//`. Trailing dots are stripped. Returns `""` when no
/// hostname can be extracted. Faithful copy of `mod_utils::base_url_hostname`.
fn base_url_hostname(base_url: &str) -> String {
    let raw = base_url.trim();
    if raw.is_empty() {
        return String::new();
    }
    let to_parse = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("//{raw}")
    };
    match url::Url::parse(&to_parse) {
        Ok(parsed) => parsed
            .host_str()
            .map(|h| h.to_lowercase().trim_end_matches('.').to_string())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Blocked tools / toolsets
// ---------------------------------------------------------------------------

/// Tools that children must never have access to.
pub const DELEGATE_BLOCKED_TOOLS: &[&str] = &[
    "delegate_task",  // no recursive delegation
    "clarify",        // no user interaction
    "memory",         // no writes to shared MEMORY.md
    "send_message",   // no cross-platform side effects
    "execute_code",   // children should reason step-by-step, not write scripts
];

/// Toolset names whose membership is excluded from the subagent-facing
/// capability hint string (`_get_subagent_toolset_names`).
pub const EXCLUDED_TOOLSET_NAMES: &[&str] = &["debugging", "safe", "delegation", "moa", "rl"];

/// Toolset names containing only blocked tools — removed for leaf children.
const BLOCKED_TOOLSET_NAMES: &[&str] = &["delegation", "clarify", "memory", "code_execution"];

pub const DEFAULT_MAX_CONCURRENT_CHILDREN: i64 = 3;
/// Flat by default: parent (0) -> child (1); grandchild rejected unless
/// `max_spawn_depth` raised.
pub const MAX_DEPTH: i64 = 1;
const MIN_SPAWN_DEPTH: i64 = 1;
const MAX_SPAWN_DEPTH_CAP: i64 = 3;

pub const DEFAULT_MAX_ITERATIONS: i64 = 50;
/// Seconds before a child agent is considered stuck.
pub const DEFAULT_CHILD_TIMEOUT: i64 = 600;
/// Seconds between parent activity heartbeats during delegation.
pub const HEARTBEAT_INTERVAL: i64 = 30;
/// 15 * 30s = 450s idle between turns -> stale.
pub const HEARTBEAT_STALE_CYCLES_IDLE: i64 = 15;
/// 40 * 30s = 1200s stuck on same tool -> stale.
pub const HEARTBEAT_STALE_CYCLES_IN_TOOL: i64 = 40;

pub fn default_toolsets() -> Vec<String> {
    vec!["terminal".to_string(), "file".to_string(), "web".to_string()]
}

// ---------------------------------------------------------------------------
// Delegation config
// ---------------------------------------------------------------------------

/// Mirror of the `delegation` config block. In Python this is loaded lazily via
/// `_load_config()` (CLI_CONFIG first, then persistent config). Here the host
/// supplies the resolved map; the readers below interpret it exactly as the
/// Python helpers do.
pub type DelegationConfig = Map<String, Value>;

fn cfg_get<'a>(cfg: &'a DelegationConfig, key: &str) -> Option<&'a Value> {
    cfg.get(key).filter(|v| !v.is_null())
}

/// Coerce a JSON value to an integer the way Python's `int(val)` would for
/// the values that flow through config (ints, floats, numeric strings, bools).
fn coerce_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        Value::String(s) => {
            let t = s.trim();
            if let Ok(i) = t.parse::<i64>() {
                Some(i)
            } else {
                t.parse::<f64>().ok().map(|f| f as i64)
            }
        }
        _ => None,
    }
}

fn coerce_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// `delegation.subagent_auto_approve` -> which approval callback to install in
/// a subagent worker thread. Returns true for auto-approve, false for
/// auto-deny (the safe default).
pub fn subagent_auto_approve(cfg: &DelegationConfig) -> bool {
    match cfg.get("subagent_auto_approve") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => is_truthy_str(s),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        _ => false,
    }
}

/// Read `delegation.max_concurrent_children`, falling back to the
/// `DELEGATION_MAX_CONCURRENT_CHILDREN` env var, then the default (3).
/// Only the floor (1) is enforced.
pub fn get_max_concurrent_children(cfg: &DelegationConfig) -> i64 {
    if let Some(val) = cfg_get(cfg, "max_concurrent_children") {
        if let Some(i) = coerce_int(val) {
            let result = i.max(1);
            if result > 10 {
                log::warn!(
                    "delegation.max_concurrent_children={result}: each child consumes API tokens independently. High values multiply cost linearly."
                );
            }
            return result;
        }
        log::warn!(
            "delegation.max_concurrent_children={val:?} is not a valid integer; using default {DEFAULT_MAX_CONCURRENT_CHILDREN}"
        );
        return DEFAULT_MAX_CONCURRENT_CHILDREN;
    }
    if let Ok(env_val) = std::env::var("DELEGATION_MAX_CONCURRENT_CHILDREN") {
        if !env_val.is_empty() {
            return env_val.trim().parse::<i64>().map(|i| i.max(1)).unwrap_or(DEFAULT_MAX_CONCURRENT_CHILDREN);
        }
    }
    DEFAULT_MAX_CONCURRENT_CHILDREN
}

/// Read `delegation.child_timeout_seconds`. Default 600s, floor 30s.
pub fn get_child_timeout(cfg: &DelegationConfig) -> f64 {
    if let Some(val) = cfg_get(cfg, "child_timeout_seconds") {
        if let Some(f) = coerce_float(val) {
            return f.max(30.0);
        }
        log::warn!(
            "delegation.child_timeout_seconds={val:?} is not a valid number; using default {DEFAULT_CHILD_TIMEOUT}"
        );
    }
    if let Ok(env_val) = std::env::var("DELEGATION_CHILD_TIMEOUT_SECONDS") {
        if let Ok(f) = env_val.trim().parse::<f64>() {
            return f.max(30.0);
        }
    }
    DEFAULT_CHILD_TIMEOUT as f64
}

/// Read `delegation.max_spawn_depth`, clamped to [1, 3]. Default 1 (flat).
pub fn get_max_spawn_depth(cfg: &DelegationConfig) -> i64 {
    let val = match cfg_get(cfg, "max_spawn_depth") {
        Some(v) => v,
        None => return MAX_DEPTH,
    };
    let ival = match coerce_int(val) {
        Some(i) => i,
        None => {
            log::warn!(
                "delegation.max_spawn_depth={val:?} is not a valid integer; using default {MAX_DEPTH}"
            );
            return MAX_DEPTH;
        }
    };
    let clamped = ival.clamp(MIN_SPAWN_DEPTH, MAX_SPAWN_DEPTH_CAP);
    if clamped != ival {
        log::warn!(
            "delegation.max_spawn_depth={ival} out of range [{MIN_SPAWN_DEPTH}, {MAX_SPAWN_DEPTH_CAP}]; clamping to {clamped}"
        );
    }
    clamped
}

/// Global kill switch for the orchestrator role (default true).
pub fn get_orchestrator_enabled(cfg: &DelegationConfig) -> bool {
    match cfg.get("orchestrator_enabled") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => {
            matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes" | "on")
        }
        _ => true,
    }
}

/// Whether narrowed child toolsets should keep the parent's MCP toolsets
/// (default true).
pub fn get_inherit_mcp_toolsets(cfg: &DelegationConfig) -> bool {
    match cfg.get("inherit_mcp_toolsets") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => is_truthy_str(s),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::Null) | None => true,
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Role normalisation
// ---------------------------------------------------------------------------

/// Normalise a caller-provided role to "leaf" or "orchestrator".
/// None/empty -> "leaf". Unknown strings coerce to "leaf" with a warning.
pub fn normalize_role(r: Option<&str>) -> &'static str {
    let r = match r {
        Some(s) if !s.is_empty() => s,
        _ => return "leaf",
    };
    match r.trim().to_lowercase().as_str() {
        "leaf" => "leaf",
        "orchestrator" => "orchestrator",
        _ => {
            log::warn!("Unknown delegate_task role={r:?}, coercing to 'leaf'");
            "leaf"
        }
    }
}

/// Honor the caller's role only when both the kill switch and the child's
/// depth allow it. Single point where role degrades to "leaf".
pub fn effective_role(
    role: &str,
    child_depth: i64,
    max_spawn_depth: i64,
    orchestrator_enabled: bool,
) -> &'static str {
    let orchestrator_ok = orchestrator_enabled && child_depth < max_spawn_depth;
    if role == "orchestrator" && orchestrator_ok {
        "orchestrator"
    } else {
        "leaf"
    }
}

// ---------------------------------------------------------------------------
// Toolset stripping / MCP preservation
// ---------------------------------------------------------------------------

/// Remove toolsets that contain only blocked tools.
pub fn strip_blocked_tools(toolsets: &[String]) -> Vec<String> {
    toolsets
        .iter()
        .filter(|t| !BLOCKED_TOOLSET_NAMES.contains(&t.as_str()))
        .cloned()
        .collect()
}

/// Return true for canonical MCP toolsets and their registered aliases.
///
/// The Python helper consults `tools.registry.get_toolset_alias_target`. Here
/// the alias resolver is injected so this stays dependency-free; pass a closure
/// returning the alias target (or `None`).
pub fn is_mcp_toolset_name(name: &str, alias_target: impl Fn(&str) -> Option<String>) -> bool {
    if name.is_empty() {
        return false;
    }
    if name.starts_with("mcp-") {
        return true;
    }
    alias_target(name)
        .map(|t| t.starts_with("mcp-"))
        .unwrap_or(false)
}

/// Append any parent MCP toolsets that are missing from a narrowed child.
pub fn preserve_parent_mcp_toolsets(
    child_toolsets: &[String],
    parent_toolsets: &BTreeSet<String>,
    alias_target: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    let mut preserved: Vec<String> = child_toolsets.to_vec();
    // parent_toolsets is a BTreeSet so iteration is already sorted (matches the
    // Python `sorted(parent_toolsets)`).
    for toolset_name in parent_toolsets {
        if is_mcp_toolset_name(toolset_name, &alias_target) && !preserved.contains(toolset_name) {
            preserved.push(toolset_name.clone());
        }
    }
    preserved
}

/// Compute the live toolset names suitable for subagent requests.
///
/// Excludes the excluded names, `hermes-*` composites, and toolsets where ALL
/// tools are in [`DELEGATE_BLOCKED_TOOLS`]. `all_toolsets` maps toolset name ->
/// its list of tool names (mirrors `toolsets.get_all_toolsets()`). Result is
/// sorted.
pub fn get_subagent_toolset_names(
    all_toolsets: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    let mut names: Vec<String> = all_toolsets
        .iter()
        .filter(|(name, tools)| {
            !EXCLUDED_TOOLSET_NAMES.contains(&name.as_str())
                && !name.starts_with("hermes-")
                && !tools.iter().all(|t| DELEGATE_BLOCKED_TOOLS.contains(&t.as_str()))
        })
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    names
}

/// `", ".join("'name'")` for the schema descriptions.
pub fn get_toolset_list_str(all_toolsets: &HashMap<String, Vec<String>>) -> String {
    get_subagent_toolset_names(all_toolsets)
        .iter()
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Child system prompt
// ---------------------------------------------------------------------------

/// Build a focused system prompt for a child agent.
///
/// When `role == "orchestrator"`, appends a delegation-capability block whose
/// depth note is literal truth grounded in the passed config.
pub fn build_child_system_prompt(
    goal: &str,
    context: Option<&str>,
    workspace_path: Option<&str>,
    role: &str,
    max_spawn_depth: i64,
    child_depth: i64,
) -> String {
    let mut parts: Vec<String> = vec![
        "You are a focused subagent working on a specific delegated task.".to_string(),
        String::new(),
        format!("YOUR TASK:\n{goal}"),
    ];
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            parts.push(format!("\nCONTEXT:\n{ctx}"));
        }
    }
    if let Some(ws) = workspace_path {
        if !ws.trim().is_empty() {
            parts.push(format!(
                "\nWORKSPACE PATH:\n{ws}\nUse this exact path for local repository/workdir operations unless the task explicitly says otherwise."
            ));
        }
    }
    parts.push(
        "\nComplete this task using the tools available to you. \
When finished, provide a clear, concise summary of:\n\
- What you did\n\
- What you found or accomplished\n\
- Any files you created or modified\n\
- Any issues encountered\n\n\
Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. \
If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n\
Be thorough but concise -- your response is returned to the parent agent as a summary."
            .to_string(),
    );
    if role == "orchestrator" {
        let child_note = if child_depth + 1 >= max_spawn_depth {
            "Your own children MUST be leaves (cannot delegate further) \
because they would be at the depth floor — you cannot pass \
role='orchestrator' to your own delegate_task calls."
        } else {
            "Your own children can themselves be orchestrators or leaves, \
depending on the `role` you pass to delegate_task. Default is \
'leaf'; pass role='orchestrator' explicitly when a child \
needs to further decompose its work."
        };
        parts.push(format!(
            "\n## Subagent Spawning (Orchestrator Role)\n\
You have access to the `delegate_task` tool and CAN spawn \
your own subagents to parallelize independent work.\n\n\
WHEN to delegate:\n\
- The goal decomposes into 2+ independent subtasks that can \
run in parallel (e.g. research A and B simultaneously).\n\
- A subtask is reasoning-heavy and would flood your context \
with intermediate data.\n\n\
WHEN NOT to delegate:\n\
- Single-step mechanical work — do it directly.\n\
- Trivial tasks you can execute in one or two tool calls.\n\
- Re-delegating your entire assigned goal to one worker \
(that's just pass-through with no value added).\n\n\
Coordinate your workers' results and synthesize them before \
reporting back to your parent. You are responsible for the \
final summary, not your workers.\n\n\
NOTE: You are at depth {child_depth}. The delegation tree \
is capped at max_spawn_depth={max_spawn_depth}. {child_note}"
        ));
    }
    parts.join("\n")
}

/// Best-effort local workspace hint for child prompts.
///
/// Returns a candidate only when it is a concrete absolute directory. The
/// candidate list (TERMINAL_CWD env, parent hints, terminal_cwd, cwd) is
/// supplied by the host; this function expands `~`, makes the path absolute,
/// and validates that it is an existing directory.
pub fn resolve_workspace_hint(candidates: &[Option<String>]) -> Option<String> {
    for candidate in candidates.iter().flatten() {
        if candidate.is_empty() {
            continue;
        }
        let expanded = expanduser(candidate);
        let abs = match std::path::Path::new(&expanded).canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // Fall back to a best-effort absolute join when canonicalize
                // fails (path may not exist yet), mirroring os.path.abspath.
                let p = std::path::Path::new(&expanded);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    match std::env::current_dir() {
                        Ok(cwd) => cwd.join(p),
                        Err(_) => continue,
                    }
                }
            }
        };
        if abs.is_absolute() && abs.is_dir() {
            return Some(abs.to_string_lossy().into_owned());
        }
    }
    None
}

fn expanduser(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~") {
        if rest.is_empty() || rest.starts_with('/') {
            if let Some(home) = dirs::home_dir() {
                return format!("{}{}", home.to_string_lossy(), rest);
            }
        }
    }
    path.to_string()
}

// ---------------------------------------------------------------------------
// Delegation progress event types
// ---------------------------------------------------------------------------

/// Formal event types emitted during delegation progress.
///
/// `_build_child_progress_callback` normalises incoming legacy strings to
/// these enum values via [`legacy_event_map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DelegateEvent {
    TaskSpawned,
    TaskProgress,
    TaskCompleted,
    TaskFailed,
    TaskThinking,
    TaskToolStarted,
    TaskToolCompleted,
}

impl DelegateEvent {
    /// The canonical wire string (`delegate.*`).
    pub fn as_str(&self) -> &'static str {
        match self {
            DelegateEvent::TaskSpawned => "delegate.task_spawned",
            DelegateEvent::TaskProgress => "delegate.task_progress",
            DelegateEvent::TaskCompleted => "delegate.task_completed",
            DelegateEvent::TaskFailed => "delegate.task_failed",
            DelegateEvent::TaskThinking => "delegate.task_thinking",
            DelegateEvent::TaskToolStarted => "delegate.tool_started",
            DelegateEvent::TaskToolCompleted => "delegate.tool_completed",
        }
    }

    /// Parse a canonical `delegate.*` string back into a variant.
    pub fn from_canonical(s: &str) -> Option<DelegateEvent> {
        match s {
            "delegate.task_spawned" => Some(DelegateEvent::TaskSpawned),
            "delegate.task_progress" => Some(DelegateEvent::TaskProgress),
            "delegate.task_completed" => Some(DelegateEvent::TaskCompleted),
            "delegate.task_failed" => Some(DelegateEvent::TaskFailed),
            "delegate.task_thinking" => Some(DelegateEvent::TaskThinking),
            "delegate.tool_started" => Some(DelegateEvent::TaskToolStarted),
            "delegate.tool_completed" => Some(DelegateEvent::TaskToolCompleted),
            _ => None,
        }
    }
}

/// Legacy event strings -> `DelegateEvent`. Incoming child-agent events use
/// the old names; the callback normalises them.
pub fn legacy_event_map(s: &str) -> Option<DelegateEvent> {
    match s {
        "_thinking" => Some(DelegateEvent::TaskThinking),
        "reasoning.available" => Some(DelegateEvent::TaskThinking),
        "tool.started" => Some(DelegateEvent::TaskToolStarted),
        "tool.completed" => Some(DelegateEvent::TaskToolCompleted),
        "subagent_progress" => Some(DelegateEvent::TaskProgress),
        _ => None,
    }
}

/// Resolve an arbitrary incoming event string to a [`DelegateEvent`] using the
/// same priority chain as `_build_child_progress_callback`: legacy map first,
/// then the canonical `delegate.*` parser. Returns `None` for unknown events
/// (which the callback drops).
pub fn resolve_event(event_type: &str) -> Option<DelegateEvent> {
    legacy_event_map(event_type).or_else(|| DelegateEvent::from_canonical(event_type))
}

// ---------------------------------------------------------------------------
// Output tail / error detection
// ---------------------------------------------------------------------------

/// One entry in a child's output tail: `{tool, preview, is_error}`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputTailEntry {
    pub tool: String,
    pub preview: String,
    pub is_error: bool,
}

/// Pull the last N tool-call results from a child's conversation messages.
///
/// `messages` is the conversation list (each item a JSON object with `role`,
/// `tool_calls`, `content`, `tool_call_id`, …). Mirrors `_extract_output_tail`.
pub fn extract_output_tail(
    messages: &Value,
    max_entries: usize,
    max_chars: usize,
) -> Vec<OutputTailEntry> {
    let messages = match messages.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };

    // First pass (forward): build tool_call_id -> tool_name map.
    let mut pending_call_by_id: HashMap<String, String> = HashMap::new();
    for msg in messages {
        let msg = match msg.as_object() {
            Some(m) => m,
            None => continue,
        };
        if msg.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(tcs) = msg.get("tool_calls").and_then(Value::as_array) {
                for tc in tcs {
                    let tc_id = tc.get("id").and_then(Value::as_str);
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .unwrap_or("tool");
                    if let Some(id) = tc_id {
                        pending_call_by_id.insert(id.to_string(), name.to_string());
                    }
                }
            }
        }
    }

    // Second pass (reverse): pick tool results, newest first.
    let mut tail: Vec<OutputTailEntry> = Vec::new();
    for msg in messages.iter().rev() {
        if tail.len() >= max_entries {
            break;
        }
        let msg = match msg.as_object() {
            Some(m) if m.get("role").and_then(Value::as_str) == Some("tool") => m,
            _ => continue,
        };
        let content = match msg.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        let is_error = looks_like_error_output(&content);
        let tool_name = msg
            .get("tool_call_id")
            .and_then(Value::as_str)
            .and_then(|id| pending_call_by_id.get(id))
            .cloned()
            .unwrap_or_else(|| "tool".to_string());
        let preview = char_truncate(&content, max_chars);
        tail.push(OutputTailEntry {
            tool: tool_name,
            preview,
            is_error,
        });
    }

    tail.reverse();
    tail
}

/// Conservative stderr/error detector for tool-result previews.
///
/// Faithful port of `_looks_like_error_output`.
pub fn looks_like_error_output(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    let head = content.trim_start();
    if head.starts_with('{') || head.starts_with('[') {
        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
            if let Some(obj) = parsed.as_object() {
                // `parsed.get("error")` truthy in Python: non-null, non-empty.
                if let Some(err) = obj.get("error") {
                    if is_truthy_json(err) {
                        return true;
                    }
                }
                let status = obj
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                if matches!(status.as_str(), "error" | "failed" | "failure" | "timeout") {
                    return true;
                }
            }
        }
    }

    let first = content
        .lines()
        .next()
        .map(|l| l.trim().to_lowercase())
        .unwrap_or_default();
    first.starts_with("error:")
        || first.starts_with("failed:")
        || first.starts_with("traceback ")
        || first.starts_with("exception:")
}

/// Python truthiness for the JSON values that flow through `error` keys.
fn is_truthy_json(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Truncate a string to at most `max_chars` Unicode scalar values, matching
/// Python's `content[:max_chars]` slicing semantics.
fn char_truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect()
    }
}

// ---------------------------------------------------------------------------
// Runtime state: pause flag + active subagent registry
// ---------------------------------------------------------------------------

static SPAWN_PAUSED: AtomicBool = AtomicBool::new(false);

/// Globally block/unblock new delegate_task spawns. Returns the new state.
pub fn set_spawn_paused(paused: bool) -> bool {
    SPAWN_PAUSED.store(paused, Ordering::SeqCst);
    paused
}

pub fn is_spawn_paused() -> bool {
    SPAWN_PAUSED.load(Ordering::SeqCst)
}

/// A TUI-facing snapshot record of a live subagent. Mirrors the dict the
/// Python registry stores (minus the live `agent` handle, which is never
/// exported).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubagentRecord {
    pub subagent_id: String,
    pub parent_id: Option<String>,
    pub depth: i64,
    pub goal: String,
    pub model: Option<String>,
    pub started_at: f64,
    pub status: String,
    pub tool_count: i64,
    pub last_tool: Option<String>,
}

fn active_subagents() -> &'static Mutex<HashMap<String, SubagentRecord>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, SubagentRecord>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a live subagent. No-op if `subagent_id` is empty.
pub fn register_subagent(record: SubagentRecord) {
    if record.subagent_id.is_empty() {
        return;
    }
    if let Ok(mut map) = active_subagents().lock() {
        map.insert(record.subagent_id.clone(), record);
    }
}

pub fn unregister_subagent(subagent_id: &str) {
    if let Ok(mut map) = active_subagents().lock() {
        map.remove(subagent_id);
    }
}

/// Update the per-subagent tool counter and last tool name on a relayed
/// `TASK_TOOL_STARTED` event. No-op if the id is unknown.
pub fn record_tool_started(subagent_id: &str, tool_count: i64, last_tool: &str) {
    if let Ok(mut map) = active_subagents().lock() {
        if let Some(rec) = map.get_mut(subagent_id) {
            rec.tool_count = tool_count;
            rec.last_tool = Some(last_tool.to_string());
        }
    }
}

/// Snapshot of the currently running subagent tree (copy, safe from any
/// thread). The live `agent` handle is intentionally never included.
pub fn list_active_subagents() -> Vec<SubagentRecord> {
    active_subagents()
        .lock()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_default()
}

/// Return whether a matching subagent exists. The Python `interrupt_subagent`
/// additionally calls `agent.interrupt(...)`; that side effect is performed by
/// the host via the returned record/id. Returns true if found.
pub fn subagent_exists(subagent_id: &str) -> bool {
    active_subagents()
        .lock()
        .map(|m| m.contains_key(subagent_id))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Credential resolution
// ---------------------------------------------------------------------------

/// Resolved delegation credentials. `command`/`args` are only populated when a
/// provider override is resolved via the runtime provider system.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DelegationCredentials {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_mode: Option<String>,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
}

/// A resolved runtime-provider bundle, as returned by
/// `hermes_cli.runtime_provider.resolve_runtime_provider`. Supplied by the host
/// when `delegation.provider` is configured.
#[derive(Debug, Clone, Default)]
pub struct RuntimeProvider {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_mode: Option<String>,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
}

fn cfg_str(cfg: &DelegationConfig, key: &str) -> Option<String> {
    let raw = cfg.get(key)?;
    let s = match raw {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Resolve credentials for subagent delegation from the delegation config.
///
/// Faithful port of `_resolve_delegation_credentials`. When `delegation.base_url`
/// is set, provider/api_mode are inferred from the host. When only
/// `delegation.provider` is set, the host's `resolve_runtime_provider`
/// implementation is invoked via `resolve_provider`. When neither is set, all
/// values are `None` so the child inherits from its parent.
///
/// `resolve_provider` mirrors `resolve_runtime_provider(requested=provider)`:
/// it returns `Ok(bundle)` or `Err(message)`.
pub fn resolve_delegation_credentials(
    cfg: &DelegationConfig,
    resolve_provider: impl Fn(&str) -> Result<RuntimeProvider, String>,
) -> Result<DelegationCredentials, String> {
    let configured_model = cfg_str(cfg, "model");
    let configured_provider = cfg_str(cfg, "provider");
    let configured_base_url = cfg_str(cfg, "base_url");
    let configured_api_key = cfg_str(cfg, "api_key");

    if let Some(base_url) = configured_base_url {
        // None -> inherited from parent in _build_child_agent.
        let api_key = configured_api_key;

        let base_lower = base_url.to_lowercase();
        let mut provider = "custom".to_string();
        let mut api_mode = "chat_completions".to_string();
        let host = base_url_hostname(&base_url);
        if host == "chatgpt.com" && base_lower.contains("/backend-api/codex") {
            provider = "openai-codex".to_string();
            api_mode = "codex_responses".to_string();
        } else if host == "api.anthropic.com" {
            provider = "anthropic".to_string();
            api_mode = "anthropic_messages".to_string();
        } else if base_lower.contains("api.kimi.com/coding") {
            provider = "custom".to_string();
            api_mode = "anthropic_messages".to_string();
        }

        return Ok(DelegationCredentials {
            model: configured_model,
            provider: Some(provider),
            base_url: Some(base_url),
            api_key,
            api_mode: Some(api_mode),
            command: None,
            args: None,
        });
    }

    let configured_provider = match configured_provider {
        Some(p) => p,
        None => {
            // No provider override — child inherits everything from parent.
            return Ok(DelegationCredentials {
                model: configured_model,
                ..Default::default()
            });
        }
    };

    // Provider is configured — resolve full credentials.
    let runtime = resolve_provider(&configured_provider).map_err(|exc| {
        format!(
            "Cannot resolve delegation provider '{configured_provider}': {exc}. \
Check that the provider is configured (API key set, valid provider name), \
or set delegation.base_url/delegation.api_key for a direct endpoint. \
Available providers: openrouter, nous, zai, kimi-coding, minimax."
        )
    })?;

    let api_key = runtime.api_key.clone().unwrap_or_default();
    if api_key.is_empty() {
        return Err(format!(
            "Delegation provider '{configured_provider}' resolved but has no API key. \
Set the appropriate environment variable or run 'hermes auth'."
        ));
    }

    Ok(DelegationCredentials {
        model: configured_model.or(runtime.model).filter(|s| !s.is_empty()),
        provider: runtime.provider,
        base_url: runtime.base_url,
        api_key: Some(api_key),
        api_mode: runtime.api_mode,
        command: runtime.command,
        args: Some(runtime.args.unwrap_or_default()),
    })
}

// ---------------------------------------------------------------------------
// Standard error payload
// ---------------------------------------------------------------------------

/// Mirrors `tools.registry.tool_error` — a JSON object with a single `error`
/// key. Non-ASCII characters are preserved verbatim (Python `ensure_ascii=False`).
pub fn tool_error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

// ---------------------------------------------------------------------------
// OpenAI Function-Calling Schema
// ---------------------------------------------------------------------------

const SCHEMA_DESCRIPTION: &str = "Spawn one or more subagents to work on tasks in isolated contexts. \
Each subagent gets its own conversation, terminal session, and toolset. \
Only the final summary is returned -- intermediate tool results \
never enter your context window.\n\n\
TWO MODES (one of 'goal' or 'tasks' is required):\n\
1. Single task: provide 'goal' (+ optional context, toolsets)\n\
2. Batch (parallel): provide 'tasks' array with up to delegation.max_concurrent_children items (default 3, configurable via config.yaml, no hard ceiling). \
All run concurrently and results are returned together. Nested delegation requires role='orchestrator' and delegation.max_spawn_depth >= 2.\n\n\
WHEN TO USE delegate_task:\n\
- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n\
- Tasks that would flood your context with intermediate data\n\
- Parallel independent workstreams (research A and B simultaneously)\n\n\
WHEN NOT TO USE (use these instead):\n\
- Mechanical multi-step work with no reasoning needed -> use execute_code\n\
- Single tool call -> just call the tool directly\n\
- Tasks needing user interaction -> subagents cannot use clarify\n\
- Durable long-running work that must outlive the current turn -> \
use cronjob (action='create') or terminal(background=True, \
notify_on_complete=True) instead. delegate_task runs SYNCHRONOUSLY \
inside the parent turn: if the parent is interrupted (user sends a \
new message, /stop, /new) the child is cancelled with status=\
'interrupted' and its work is discarded. Children cannot continue \
in the background.\n\n\
IMPORTANT:\n\
- Subagents have NO memory of your conversation. Pass all relevant \
info (file paths, error messages, constraints) via the 'context' field.\n\
- If the user is writing in a non-English language, or asked for \
output in a specific language / tone / style, say so in 'context' \
(e.g. \"respond in Chinese\", \"return output in Japanese\"). \
Otherwise subagents default to English and their summaries will \
contaminate your final reply with the wrong language.\n\
- Subagent summaries are SELF-REPORTS, not verified facts. A subagent \
that claims \"uploaded successfully\" or \"file written\" may be wrong. \
For operations with external side-effects (HTTP POST/PUT, remote \
writes, file creation at shared paths, publishing), require the \
subagent to return a verifiable handle (URL, ID, absolute path, HTTP \
status) and verify it yourself — fetch the URL, stat the file, read \
back the content — before telling the user the operation succeeded.\n\
- Leaf subagents (role='leaf', the default) CANNOT call: \
delegate_task, clarify, memory, send_message, execute_code.\n\
- Orchestrator subagents (role='orchestrator') retain \
delegate_task so they can spawn their own workers, but still \
cannot use clarify, memory, send_message, or execute_code. \
Orchestrators are bounded by delegation.max_spawn_depth \
(default 2) and can be disabled globally via \
delegation.orchestrator_enabled=false.\n\
- Each subagent gets its own terminal session (separate working directory and state).\n\
- Results are always returned as an array, one entry per task.";

const ROLE_DESCRIPTION: &str = "Role of the child agent. 'leaf' (default) = focused \
worker, cannot delegate further. 'orchestrator' = can \
use delegate_task to spawn its own workers. Requires \
delegation.max_spawn_depth >= 2 in config; ignored \
(treated as 'leaf') when the child would exceed \
max_spawn_depth or when \
delegation.orchestrator_enabled=false.";

const ACP_COMMAND_DESCRIPTION: &str = "Override ACP command for child agents (e.g. 'claude', 'copilot'). \
When set, children use ACP subprocess transport instead of inheriting \
the parent's transport. Enables spawning Claude Code (claude --acp --stdio) \
or other ACP-capable agents from any parent, including Discord/Telegram/CLI.";

const ACP_ARGS_DESCRIPTION: &str = "Arguments for the ACP command (default: ['--acp', '--stdio']). \
Only used when acp_command is set. Example: ['--acp', '--stdio', '--model', 'claude-opus-4-6']";

/// Build the `delegate_task` OpenAI function-calling schema, with the dynamic
/// `toolsets` descriptions filled in from the live toolset list.
///
/// Mirrors `DELEGATE_TASK_SCHEMA` after `_refresh_delegate_task_schema()`.
/// Pass the live toolset map (name -> tool names) to populate the dynamic
/// descriptions; pass an empty map to mirror the unrefreshed default (empty
/// list string).
pub fn delegate_task_schema(all_toolsets: &HashMap<String, Vec<String>>) -> Value {
    let toolset_list = get_toolset_list_str(all_toolsets);

    let top_toolsets_desc = format!(
        "Toolsets to enable for this subagent. \
Default: inherits your enabled toolsets. \
Available toolsets: {toolset_list}. \
Common patterns: ['terminal', 'file'] for code work, \
['web'] for research, ['browser'] for web interaction, \
['terminal', 'file', 'web'] for full-stack tasks."
    );
    let task_toolsets_desc = format!(
        "Toolsets for this specific task. \
Available: {toolset_list}. \
Use 'web' for network access, 'terminal' for shell, 'browser' for web interaction."
    );

    json!({
        "name": "delegate_task",
        "description": SCHEMA_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What the subagent should accomplish. Be specific and self-contained -- the subagent knows nothing about your conversation history."
                },
                "context": {
                    "type": "string",
                    "description": "Background information the subagent needs: file paths, error messages, project structure, constraints. The more specific you are, the better the subagent performs."
                },
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": top_toolsets_desc
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "description": "Task goal"},
                            "context": {"type": "string", "description": "Task-specific context"},
                            "toolsets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": task_toolsets_desc
                            },
                            "acp_command": {
                                "type": "string",
                                "description": "Per-task ACP command override (e.g. 'claude'). Overrides the top-level acp_command for this task only."
                            },
                            "acp_args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Per-task ACP args override."
                            },
                            "role": {
                                "type": "string",
                                "enum": ["leaf", "orchestrator"],
                                "description": "Per-task role override. See top-level 'role' for semantics."
                            }
                        },
                        "required": ["goal"]
                    },
                    "description": "Batch mode: tasks to run in parallel (limit configurable via delegation.max_concurrent_children, default 3). Each gets its own subagent with isolated context and terminal session. When provided, top-level goal/context/toolsets are ignored."
                },
                "role": {
                    "type": "string",
                    "enum": ["leaf", "orchestrator"],
                    "description": ROLE_DESCRIPTION
                },
                "acp_command": {
                    "type": "string",
                    "description": ACP_COMMAND_DESCRIPTION
                },
                "acp_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ACP_ARGS_DESCRIPTION
                }
            },
            "required": []
        }
    })
}

/// Delegation has no external requirements -- always available.
pub fn check_delegate_requirements() -> bool {
    true
}

// ---------------------------------------------------------------------------
// delegate_task request normalisation
// ---------------------------------------------------------------------------

/// A single normalised delegation task (from `goal`/`context`/… or one element
/// of the `tasks` batch array).
#[derive(Debug, Clone, PartialEq)]
pub struct DelegateTask {
    pub goal: String,
    pub context: Option<String>,
    pub toolsets: Option<Vec<String>>,
    pub role: String,
    pub acp_command: Option<String>,
    pub acp_args: Option<Vec<String>>,
}

/// Outcome of the up-front validation/normalisation `delegate_task` performs
/// before any child is built.
#[derive(Debug)]
pub enum DelegatePrep {
    /// A `tool_error(...)` JSON string the caller should return verbatim.
    Error(String),
    /// The validated task list (input order preserved) plus the resolved
    /// effective max_iterations and the resolved depth/max_spawn used.
    Ready {
        tasks: Vec<DelegateTask>,
        effective_max_iterations: i64,
        depth: i64,
        max_spawn_depth: i64,
    },
}

/// Top-level inputs to `delegate_task`, mirroring the Python signature's
/// scalar args (the `parent_agent` is the host's concern).
#[derive(Debug, Clone, Default)]
pub struct DelegateTaskArgs {
    pub goal: Option<String>,
    pub context: Option<String>,
    pub toolsets: Option<Vec<String>>,
    pub tasks: Option<Vec<DelegateTask>>,
    pub max_iterations: Option<i64>,
    pub acp_command: Option<String>,
    pub acp_args: Option<Vec<String>>,
    pub role: Option<String>,
}

/// Validate and normalise the `delegate_task` arguments exactly as the Python
/// front-half of `delegate_task` does (before child construction).
///
/// `depth` is the parent's `_delegate_depth`. This performs all the early
/// returns (pause, depth, batch size, missing goal) and produces a normalised
/// task list; spawning the children themselves is the host's responsibility.
pub fn prepare_delegate_task(
    args: &DelegateTaskArgs,
    cfg: &DelegationConfig,
    depth: i64,
) -> DelegatePrep {
    // Operator-controlled kill switch.
    if is_spawn_paused() {
        return DelegatePrep::Error(tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI \
(`p` in /agents) or the `delegation.pause` RPC before retrying.",
        ));
    }

    let top_role = normalize_role(args.role.as_deref());

    let max_spawn = get_max_spawn_depth(cfg);
    if depth >= max_spawn {
        return DelegatePrep::Error(
            json!({
                "error": format!(
                    "Delegation depth limit reached (depth={depth}, \
max_spawn_depth={max_spawn}). Raise \
delegation.max_spawn_depth in config.yaml if deeper \
nesting is required (cap: {MAX_SPAWN_DEPTH_CAP})."
                )
            })
            .to_string(),
        );
    }

    let default_max_iter = cfg_get(cfg, "max_iterations")
        .and_then(coerce_int)
        .unwrap_or(DEFAULT_MAX_ITERATIONS);
    if let Some(mi) = args.max_iterations {
        if mi != default_max_iter {
            log::debug!(
                "delegate_task: ignoring caller-supplied max_iterations={mi}; using delegation.max_iterations={default_max_iter} from config"
            );
        }
    }
    let effective_max_iter = default_max_iter;

    let max_children = get_max_concurrent_children(cfg);

    let task_list: Vec<DelegateTask> = match &args.tasks {
        Some(tasks) if !tasks.is_empty() => {
            if tasks.len() as i64 > max_children {
                return DelegatePrep::Error(tool_error(&format!(
                    "Too many tasks: {} provided, but max_concurrent_children is {max_children}. \
Either reduce the task count, split into multiple delegate_task calls, or increase \
delegation.max_concurrent_children in config.yaml.",
                    tasks.len()
                )));
            }
            tasks.clone()
        }
        _ => {
            match args.goal.as_deref().filter(|g| !g.trim().is_empty()) {
                Some(goal) => vec![DelegateTask {
                    goal: goal.to_string(),
                    context: args.context.clone(),
                    toolsets: args.toolsets.clone(),
                    role: top_role.to_string(),
                    acp_command: None,
                    acp_args: None,
                }],
                None => {
                    return DelegatePrep::Error(tool_error(
                        "Provide either 'goal' (single task) or 'tasks' (batch).",
                    ));
                }
            }
        }
    };

    if task_list.is_empty() {
        return DelegatePrep::Error(tool_error("No tasks provided."));
    }

    // Validate each task has a goal.
    for (i, task) in task_list.iter().enumerate() {
        if task.goal.trim().is_empty() {
            return DelegatePrep::Error(tool_error(&format!("Task {i} is missing a 'goal'.")));
        }
    }

    DelegatePrep::Ready {
        tasks: task_list,
        effective_max_iterations: effective_max_iter,
        depth,
        max_spawn_depth: max_spawn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(v: Value) -> DelegationConfig {
        v.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn normalize_role_cases() {
        assert_eq!(normalize_role(None), "leaf");
        assert_eq!(normalize_role(Some("")), "leaf");
        assert_eq!(normalize_role(Some("LEAF")), "leaf");
        assert_eq!(normalize_role(Some(" Orchestrator ")), "orchestrator");
        assert_eq!(normalize_role(Some("bogus")), "leaf");
    }

    #[test]
    fn effective_role_depth_and_killswitch() {
        // depth 1 < max 2, enabled -> orchestrator
        assert_eq!(effective_role("orchestrator", 1, 2, true), "orchestrator");
        // depth floor reached
        assert_eq!(effective_role("orchestrator", 2, 2, true), "leaf");
        // kill switch off
        assert_eq!(effective_role("orchestrator", 1, 2, false), "leaf");
        // leaf stays leaf
        assert_eq!(effective_role("leaf", 0, 3, true), "leaf");
    }

    #[test]
    fn max_concurrent_children_clamps_and_defaults() {
        assert_eq!(
            get_max_concurrent_children(&cfg(json!({"max_concurrent_children": 0}))),
            1
        );
        assert_eq!(
            get_max_concurrent_children(&cfg(json!({"max_concurrent_children": 7}))),
            7
        );
        // invalid -> default
        assert_eq!(
            get_max_concurrent_children(&cfg(json!({"max_concurrent_children": "abc"}))),
            DEFAULT_MAX_CONCURRENT_CHILDREN
        );
        // string int parses
        assert_eq!(
            get_max_concurrent_children(&cfg(json!({"max_concurrent_children": "5"}))),
            5
        );
    }

    #[test]
    fn child_timeout_floor_and_default() {
        assert_eq!(get_child_timeout(&cfg(json!({"child_timeout_seconds": 5}))), 30.0);
        assert_eq!(
            get_child_timeout(&cfg(json!({"child_timeout_seconds": 900}))),
            900.0
        );
        assert_eq!(get_child_timeout(&cfg(json!({}))), DEFAULT_CHILD_TIMEOUT as f64);
    }

    #[test]
    fn max_spawn_depth_clamps() {
        assert_eq!(get_max_spawn_depth(&cfg(json!({}))), MAX_DEPTH);
        assert_eq!(get_max_spawn_depth(&cfg(json!({"max_spawn_depth": 0}))), 1);
        assert_eq!(get_max_spawn_depth(&cfg(json!({"max_spawn_depth": 99}))), 3);
        assert_eq!(get_max_spawn_depth(&cfg(json!({"max_spawn_depth": 2}))), 2);
        assert_eq!(
            get_max_spawn_depth(&cfg(json!({"max_spawn_depth": "bad"}))),
            MAX_DEPTH
        );
    }

    #[test]
    fn orchestrator_enabled_parsing() {
        assert!(get_orchestrator_enabled(&cfg(json!({}))));
        assert!(!get_orchestrator_enabled(&cfg(json!({"orchestrator_enabled": false}))));
        assert!(get_orchestrator_enabled(&cfg(json!({"orchestrator_enabled": "on"}))));
        assert!(!get_orchestrator_enabled(
            &cfg(json!({"orchestrator_enabled": "no"}))
        ));
    }

    #[test]
    fn strip_blocked_tools_removes_blocked() {
        let input = vec![
            "terminal".to_string(),
            "delegation".to_string(),
            "memory".to_string(),
            "web".to_string(),
        ];
        assert_eq!(
            strip_blocked_tools(&input),
            vec!["terminal".to_string(), "web".to_string()]
        );
    }

    #[test]
    fn subagent_toolset_names_filters() {
        let mut all: HashMap<String, Vec<String>> = HashMap::new();
        all.insert("terminal".into(), vec!["terminal".into()]);
        all.insert("delegation".into(), vec!["delegate_task".into()]); // excluded by name
        all.insert("hermes-core".into(), vec!["foo".into()]); // hermes- prefix
        all.insert("onlyblocked".into(), vec!["clarify".into(), "memory".into()]); // all blocked
        all.insert("web".into(), vec!["web_search".into()]);
        let names = get_subagent_toolset_names(&all);
        assert_eq!(names, vec!["terminal".to_string(), "web".to_string()]);
    }

    #[test]
    fn mcp_preservation() {
        let mut parents = BTreeSet::new();
        parents.insert("mcp-github".to_string());
        parents.insert("terminal".to_string());
        parents.insert("aliased".to_string());
        let child = vec!["terminal".to_string()];
        let preserved = preserve_parent_mcp_toolsets(&child, &parents, |n| {
            if n == "aliased" {
                Some("mcp-aliased-target".to_string())
            } else {
                None
            }
        });
        assert!(preserved.contains(&"mcp-github".to_string()));
        assert!(preserved.contains(&"aliased".to_string()));
        assert!(preserved.contains(&"terminal".to_string()));
    }

    #[test]
    fn child_prompt_leaf_and_orchestrator() {
        let leaf = build_child_system_prompt("do X", Some("ctx"), None, "leaf", 2, 1);
        assert!(leaf.contains("YOUR TASK:\ndo X"));
        assert!(leaf.contains("CONTEXT:\nctx"));
        assert!(!leaf.contains("Orchestrator Role"));

        let orch = build_child_system_prompt("do X", None, Some("/tmp"), "orchestrator", 3, 1);
        assert!(orch.contains("WORKSPACE PATH:\n/tmp"));
        assert!(orch.contains("Orchestrator Role"));
        assert!(orch.contains("depth 1"));
        // child_depth+1 (2) < max (3) -> children can be orchestrators note
        assert!(orch.contains("can themselves be orchestrators"));

        let orch_floor =
            build_child_system_prompt("do X", None, None, "orchestrator", 2, 1);
        assert!(orch_floor.contains("MUST be leaves"));
    }

    #[test]
    fn event_resolution() {
        assert_eq!(resolve_event("_thinking"), Some(DelegateEvent::TaskThinking));
        assert_eq!(
            resolve_event("tool.started"),
            Some(DelegateEvent::TaskToolStarted)
        );
        assert_eq!(
            resolve_event("subagent_progress"),
            Some(DelegateEvent::TaskProgress)
        );
        assert_eq!(
            resolve_event("delegate.tool_started"),
            Some(DelegateEvent::TaskToolStarted)
        );
        assert_eq!(resolve_event("totally.unknown"), None);
        assert_eq!(
            DelegateEvent::TaskThinking.as_str(),
            "delegate.task_thinking"
        );
    }

    #[test]
    fn error_output_detection() {
        assert!(!looks_like_error_output(""));
        assert!(looks_like_error_output(r#"{"error": "boom"}"#));
        assert!(!looks_like_error_output(r#"{"error": ""}"#));
        assert!(!looks_like_error_output(r#"{"error": null}"#));
        assert!(looks_like_error_output(r#"{"status": "failed"}"#));
        assert!(!looks_like_error_output(r#"{"status": "ok"}"#));
        assert!(looks_like_error_output("Error: something went wrong"));
        assert!(looks_like_error_output("Traceback (most recent call last)"));
        // plain text containing the word error in the middle is NOT flagged
        assert!(!looks_like_error_output("the operation had no error at all"));
    }

    #[test]
    fn output_tail_pairs_by_id() {
        let messages = json!([
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "read_file"}},
                    {"id": "b", "function": {"name": "run_command"}}
                ]
            },
            {"role": "tool", "tool_call_id": "a", "content": "file contents here"},
            {"role": "tool", "tool_call_id": "b", "content": "{\"error\": \"nope\"}"}
        ]);
        let tail = extract_output_tail(&messages, 12, 8000);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].tool, "read_file");
        assert!(!tail[0].is_error);
        assert_eq!(tail[1].tool, "run_command");
        assert!(tail[1].is_error);
    }

    #[test]
    fn output_tail_respects_limits() {
        let messages = json!([
            {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "t"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "abcdefghij"}
        ]);
        let tail = extract_output_tail(&messages, 1, 4);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].preview, "abcd");
    }

    #[test]
    fn pause_flag_roundtrip() {
        assert!(set_spawn_paused(true));
        assert!(is_spawn_paused());
        assert!(!set_spawn_paused(false));
        assert!(!is_spawn_paused());
    }

    #[test]
    fn subagent_registry_lifecycle() {
        let rec = SubagentRecord {
            subagent_id: "sa-test-1".to_string(),
            depth: 0,
            goal: "g".to_string(),
            status: "running".to_string(),
            ..Default::default()
        };
        register_subagent(rec);
        assert!(subagent_exists("sa-test-1"));
        record_tool_started("sa-test-1", 3, "read_file");
        let snap = list_active_subagents();
        let found = snap.iter().find(|r| r.subagent_id == "sa-test-1").unwrap();
        assert_eq!(found.tool_count, 3);
        assert_eq!(found.last_tool.as_deref(), Some("read_file"));
        // agent handle never exported — record carries no live handle field.
        unregister_subagent("sa-test-1");
        assert!(!subagent_exists("sa-test-1"));
        // empty id is a no-op
        register_subagent(SubagentRecord::default());
        assert!(!subagent_exists(""));
    }

    #[test]
    fn credentials_base_url_inference() {
        let resolver = |_: &str| -> Result<RuntimeProvider, String> {
            panic!("provider resolver must not be called when base_url is set")
        };

        let anthropic = resolve_delegation_credentials(
            &cfg(json!({"base_url": "https://api.anthropic.com/v1"})),
            resolver,
        )
        .unwrap();
        assert_eq!(anthropic.provider.as_deref(), Some("anthropic"));
        assert_eq!(anthropic.api_mode.as_deref(), Some("anthropic_messages"));
        assert!(anthropic.api_key.is_none());

        let codex = resolve_delegation_credentials(
            &cfg(json!({"base_url": "https://chatgpt.com/backend-api/codex"})),
            resolver,
        )
        .unwrap();
        assert_eq!(codex.provider.as_deref(), Some("openai-codex"));
        assert_eq!(codex.api_mode.as_deref(), Some("codex_responses"));

        let kimi = resolve_delegation_credentials(
            &cfg(json!({"base_url": "https://api.kimi.com/coding/v1", "api_key": "k"})),
            resolver,
        )
        .unwrap();
        assert_eq!(kimi.provider.as_deref(), Some("custom"));
        assert_eq!(kimi.api_mode.as_deref(), Some("anthropic_messages"));
        assert_eq!(kimi.api_key.as_deref(), Some("k"));

        let custom = resolve_delegation_credentials(
            &cfg(json!({"base_url": "https://example.com/v1"})),
            resolver,
        )
        .unwrap();
        assert_eq!(custom.provider.as_deref(), Some("custom"));
        assert_eq!(custom.api_mode.as_deref(), Some("chat_completions"));
    }

    #[test]
    fn credentials_inherit_when_unset() {
        let resolver = |_: &str| -> Result<RuntimeProvider, String> { unreachable!() };
        let creds =
            resolve_delegation_credentials(&cfg(json!({"model": "m"})), resolver).unwrap();
        assert_eq!(creds.model.as_deref(), Some("m"));
        assert!(creds.provider.is_none());
        assert!(creds.base_url.is_none());
        assert!(creds.api_key.is_none());
    }

    #[test]
    fn credentials_provider_resolution() {
        let resolver = |p: &str| -> Result<RuntimeProvider, String> {
            assert_eq!(p, "nous");
            Ok(RuntimeProvider {
                model: Some("nous-model".into()),
                provider: Some("nous".into()),
                base_url: Some("https://nous.example".into()),
                api_key: Some("secret".into()),
                api_mode: Some("chat_completions".into()),
                command: None,
                args: Some(vec![]),
            })
        };
        let creds =
            resolve_delegation_credentials(&cfg(json!({"provider": "nous"})), resolver).unwrap();
        assert_eq!(creds.provider.as_deref(), Some("nous"));
        assert_eq!(creds.api_key.as_deref(), Some("secret"));
        // configured model absent -> falls back to runtime model
        assert_eq!(creds.model.as_deref(), Some("nous-model"));

        // empty api key -> error
        let bad_resolver = |_: &str| -> Result<RuntimeProvider, String> {
            Ok(RuntimeProvider {
                api_key: Some(String::new()),
                ..Default::default()
            })
        };
        let err = resolve_delegation_credentials(&cfg(json!({"provider": "x"})), bad_resolver)
            .unwrap_err();
        assert!(err.contains("no API key"));

        // resolver error wrapped
        let fail_resolver =
            |_: &str| -> Result<RuntimeProvider, String> { Err("kaboom".to_string()) };
        let err =
            resolve_delegation_credentials(&cfg(json!({"provider": "x"})), fail_resolver)
                .unwrap_err();
        assert!(err.contains("Cannot resolve delegation provider 'x': kaboom"));
    }

    #[test]
    fn schema_dynamic_descriptions() {
        let mut all: HashMap<String, Vec<String>> = HashMap::new();
        all.insert("terminal".into(), vec!["terminal".into()]);
        all.insert("web".into(), vec!["web_search".into()]);
        let schema = delegate_task_schema(&all);
        assert_eq!(schema["name"], "delegate_task");
        let toolsets_desc = schema["parameters"]["properties"]["toolsets"]["description"]
            .as_str()
            .unwrap();
        assert!(toolsets_desc.contains("'terminal'"));
        assert!(toolsets_desc.contains("'web'"));
        // required is empty
        assert_eq!(schema["parameters"]["required"], json!([]));
    }

    #[test]
    fn prepare_rejects_when_paused() {
        set_spawn_paused(true);
        let args = DelegateTaskArgs {
            goal: Some("do it".into()),
            ..Default::default()
        };
        let prep = prepare_delegate_task(&args, &cfg(json!({})), 0);
        set_spawn_paused(false);
        match prep {
            DelegatePrep::Error(e) => assert!(e.contains("paused")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn prepare_depth_limit() {
        let args = DelegateTaskArgs {
            goal: Some("g".into()),
            ..Default::default()
        };
        // default max_spawn = 1, depth 1 >= 1 -> error
        let prep = prepare_delegate_task(&args, &cfg(json!({})), 1);
        match prep {
            DelegatePrep::Error(e) => assert!(e.contains("depth limit")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn prepare_single_and_batch() {
        let args = DelegateTaskArgs {
            goal: Some("solo".into()),
            context: Some("c".into()),
            ..Default::default()
        };
        match prepare_delegate_task(&args, &cfg(json!({})), 0) {
            DelegatePrep::Ready { tasks, effective_max_iterations, .. } => {
                assert_eq!(tasks.len(), 1);
                assert_eq!(tasks[0].goal, "solo");
                assert_eq!(effective_max_iterations, DEFAULT_MAX_ITERATIONS);
            }
            _ => panic!("expected ready"),
        }

        // batch too large for default max_children (3)
        let big: Vec<DelegateTask> = (0..4)
            .map(|i| DelegateTask {
                goal: format!("t{i}"),
                context: None,
                toolsets: None,
                role: "leaf".into(),
                acp_command: None,
                acp_args: None,
            })
            .collect();
        let args = DelegateTaskArgs {
            tasks: Some(big),
            ..Default::default()
        };
        match prepare_delegate_task(&args, &cfg(json!({})), 0) {
            DelegatePrep::Error(e) => assert!(e.contains("Too many tasks")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn prepare_missing_goal() {
        let args = DelegateTaskArgs::default();
        match prepare_delegate_task(&args, &cfg(json!({})), 0) {
            DelegatePrep::Error(e) => assert!(e.contains("Provide either")),
            _ => panic!("expected error"),
        }

        let args = DelegateTaskArgs {
            tasks: Some(vec![DelegateTask {
                goal: "   ".into(),
                context: None,
                toolsets: None,
                role: "leaf".into(),
                acp_command: None,
                acp_args: None,
            }]),
            ..Default::default()
        };
        match prepare_delegate_task(&args, &cfg(json!({})), 0) {
            DelegatePrep::Error(e) => assert!(e.contains("missing a 'goal'")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn tool_error_shape() {
        assert_eq!(tool_error("boom"), r#"{"error":"boom"}"#);
    }
}
