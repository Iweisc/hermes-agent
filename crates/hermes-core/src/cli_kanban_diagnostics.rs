//! Kanban diagnostics — structured, actionable distress signals for tasks.
//!
//! A [`Diagnostic`] is a machine-readable description of something that's wrong
//! with a kanban task: a hallucinated card id, a spawn crash-loop, a task stuck
//! blocked for too long, etc. Each one carries:
//!
//! * A **kind** (canonical code; UI/tests match on this).
//! * A **severity** (`warning` / `error` / `critical`).
//! * A **title** (one-line human description) and **detail** (longer text).
//! * A list of **suggested actions** — structured entries the dashboard turns
//!   into buttons and the CLI turns into hints.
//!
//! Rules run over (task, recent events, recent runs) and emit diagnostics. They
//! are stateless and read-only — no DB writes. Callers compute diagnostics on
//! demand (on `/board` load, `/tasks/:id` fetch, or `hermes kanban
//! diagnostics`).
//!
//! This is a faithful native port of `hermes_cli/kanban_diagnostics.py`. The
//! Python original tolerated tasks/events/runs being sqlite3.Row, dataclasses,
//! or plain dicts. To preserve that flexibility every input here is a
//! [`serde_json::Value`] (typically a JSON object); field access goes through
//! [`field`] which mirrors the Python `_task_field` semantics.

use serde_json::{json, Map, Value};

/// Severity rungs, ordered least → most urgent. The UI colors them amber
/// (warning), orange (error), red (critical). Sorted outputs put critical first
/// so operators see the worst fires at the top.
pub const SEVERITY_ORDER: [&str; 3] = ["warning", "error", "critical"];

/// Known kinds (for the UI's filter / legend / i18n keys). Update when rules
/// are added.
pub const DIAGNOSTIC_KINDS: [&str; 5] = [
    "hallucinated_cards",
    "prose_phantom_refs",
    "repeated_failures",
    "repeated_crashes",
    "stuck_in_blocked",
];

/// A single recovery action attached to a diagnostic.
///
/// The `kind` determines how both the UI and CLI render it:
///
/// * `reclaim` / `reassign` — POST to the matching `/tasks/:id/*` endpoint;
///   dashboard wires into the existing recovery popover.
/// * `unblock` — PATCH status back to `ready` (for stuck-blocked diagnostics).
/// * `cli_hint` — print/copy a shell command (e.g. `hermes -p <profile> auth`).
///   No HTTP side effect.
/// * `open_docs` — deep-link to the docs URL named in `payload.url`.
/// * `comment` — nudge the operator to add a comment (for stuck-blocked tasks
///   that need human input).
///
/// `suggested = true` marks the action as the recommended first step; the UI
/// highlights it. Multiple actions can be suggested if they're equally valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticAction {
    pub kind: String,
    pub label: String,
    pub payload: Map<String, Value>,
    pub suggested: bool,
}

impl DiagnosticAction {
    /// Construct an action with an empty payload and `suggested = false`.
    pub fn new(kind: impl Into<String>, label: impl Into<String>) -> Self {
        DiagnosticAction {
            kind: kind.into(),
            label: label.into(),
            payload: Map::new(),
            suggested: false,
        }
    }

    /// Builder: set the payload.
    pub fn with_payload(mut self, payload: Map<String, Value>) -> Self {
        self.payload = payload;
        self
    }

    /// Builder: mark this action as suggested.
    pub fn suggested(mut self, suggested: bool) -> Self {
        self.suggested = suggested;
        self
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "kind": self.kind,
            "label": self.label,
            "payload": Value::Object(self.payload.clone()),
            "suggested": self.suggested,
        })
    }
}

/// One active distress signal on a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub kind: String,
    /// `"warning"` | `"error"` | `"critical"`.
    pub severity: String,
    pub title: String,
    pub detail: String,
    pub actions: Vec<DiagnosticAction>,
    pub first_seen_at: i64,
    pub last_seen_at: i64,
    pub count: i64,
    /// Optional: the run id this diagnostic is scoped to. `None` = task-wide.
    pub run_id: Option<i64>,
    /// Optional structured payload for the UI (phantom ids, failure count).
    pub data: Map<String, Value>,
}

impl Default for Diagnostic {
    fn default() -> Self {
        Diagnostic {
            kind: String::new(),
            severity: String::new(),
            title: String::new(),
            detail: String::new(),
            actions: Vec::new(),
            first_seen_at: 0,
            last_seen_at: 0,
            count: 1,
            run_id: None,
            data: Map::new(),
        }
    }
}

impl Diagnostic {
    pub fn to_dict(&self) -> Value {
        json!({
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "actions": self.actions.iter().map(|a| a.to_dict()).collect::<Vec<_>>(),
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "count": self.count,
            "run_id": self.run_id,
            "data": Value::Object(self.data.clone()),
        })
    }
}

// ---------------------------------------------------------------------------
// Rule helpers
// ---------------------------------------------------------------------------

/// Read a field from a task/event/run regardless of representation.
///
/// Python callers passed sqlite3.Row, dataclasses, or plain dicts. Here the
/// canonical representation is a JSON object, so this just does a keyed lookup,
/// returning `Value::Null` when the value is absent. A non-object input yields
/// `Value::Null` too (matching the Python "missing → default" behaviour).
pub fn field<'a>(obj: &'a Value, name: &str) -> &'a Value {
    match obj.get(name) {
        Some(v) => v,
        None => &Value::Null,
    }
}

/// Tolerate `event.payload` being either an object or a JSON string. Mirrors
/// Python `_parse_payload`: returns an empty object on anything unparseable.
fn parse_payload(ev: &Value) -> Map<String, Value> {
    let p = field(ev, "payload");
    match p {
        Value::Object(m) => m.clone(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        _ => Map::new(),
    }
}

fn event_kind(ev: &Value) -> String {
    match field(ev, "kind") {
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

/// Read an integer-ish field, coercing JSON numbers/strings to i64. Missing or
/// uncoercible values yield `0` — matching Python's `int(t or 0)`.
fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else {
                n.as_f64().map(|f| f as i64).unwrap_or(0)
            }
        }
        Value::String(s) => s.trim().parse::<i64>().ok().unwrap_or(0),
        Value::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn event_ts(ev: &Value) -> i64 {
    as_i64(field(ev, "created_at"))
}

/// Read a string field as an owned `String`, treating non-strings as empty.
fn as_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

/// Read an optional string: returns `None` when the value is JSON null/absent,
/// otherwise the string (non-string scalars stringify to their text form, but
/// in practice these fields are always strings or null).
fn opt_str(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Return events of `kind` that have no `completed`/`edited` event *strictly
/// after* them. Walks chronologically: each clean event resets the
/// accumulator; each matching event gets appended.
///
/// Events must be sorted by id (arrival order); callers pass the task's full
/// event list which the DB already returns in that order.
fn active_hallucination_events<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    let mut active: Vec<&Value> = Vec::new();
    for ev in events {
        let k = event_kind(ev);
        if k == "completed" || k == "edited" {
            active.clear();
        } else if k == kind {
            active.push(ev);
        }
    }
    active
}

/// Timestamp of the most recent clean completion / edit event.
///
/// Kept for general "has this task ever been successfully completed" lookups;
/// hallucination rules use [`active_hallucination_events`] instead because they
/// need strict ordering.
pub fn latest_clean_event_ts(events: &[Value]) -> i64 {
    let mut latest = 0;
    for ev in events {
        let k = event_kind(ev);
        if k == "completed" || k == "edited" {
            let t = event_ts(ev);
            if t > latest {
                latest = t;
            }
        }
    }
    latest
}

/// Standard always-available actions. Every diagnostic can offer these as
/// fallbacks regardless of kind — they're the two baseline recovery primitives
/// the kernel supports.
fn generic_recovery_actions(_task: &Value, running: bool) -> Vec<DiagnosticAction> {
    let mut out: Vec<DiagnosticAction> = Vec::new();
    if running {
        out.push(DiagnosticAction::new("reclaim", "Reclaim task"));
    }
    let mut payload = Map::new();
    payload.insert("reclaim_first".to_string(), Value::Bool(running));
    out.push(
        DiagnosticAction::new("reassign", "Reassign to different profile").with_payload(payload),
    );
    out
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Read an integer config value by key (coercing numbers/strings), falling back
/// to `default` when absent or uncoercible.
fn cfg_int(cfg: &Map<String, Value>, key: &str, default: i64) -> i64 {
    match cfg.get(key) {
        Some(v @ Value::Number(_)) => as_i64(v),
        Some(v @ Value::String(_)) => {
            let s = as_str(v);
            s.trim().parse::<i64>().ok().unwrap_or(default)
        }
        _ => default,
    }
}

/// Read a float config value by key, falling back to `default`.
fn cfg_float(cfg: &Map<String, Value>, key: &str, default: f64) -> f64 {
    match cfg.get(key) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok().unwrap_or(default),
        _ => default,
    }
}

/// Default diagnostics configuration, mirroring Python `DEFAULT_CONFIG`.
pub fn default_config() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("failure_threshold".to_string(), json!(3));
    // Legacy alias accepted at read time by the repeated-failures rule.
    m.insert("spawn_failure_threshold".to_string(), json!(3));
    m.insert("crash_threshold".to_string(), json!(2));
    m.insert("blocked_stale_hours".to_string(), json!(24));
    m
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

/// Truncate a string to at most `max` characters (by Unicode scalar), appending
/// `…` when it was longer. Mirrors `text[:max] + ("…" if len > max else "")`.
fn truncate_ellipsis(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > max {
        let mut out: String = chars[..max].iter().collect();
        out.push('…');
        out
    } else {
        s.to_string()
    }
}

/// Truncate to at most `max` characters, no ellipsis. Mirrors `text[:max]`.
fn head_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// First physical line of a string (`text.splitlines()[0]`), or empty.
fn first_line(s: &str) -> &str {
    s.split(['\n', '\r']).next().unwrap_or("")
}

// ---------------------------------------------------------------------------
// Rule implementations
// ---------------------------------------------------------------------------
//
// Each rule takes (task, events, runs, now_ts, cfg) and returns zero or more
// Diagnostic instances.

/// Blocked-hallucination gate fires: a worker called kanban_complete with
/// created_cards that didn't exist or weren't created by the completing
/// profile. Task stayed in its prior state; the operator needs to decide how to
/// proceed.
///
/// Auto-clears when a successful completion (or edit) follows the blocked
/// event.
fn rule_hallucinated_cards(
    task: &Value,
    events: &[Value],
    _runs: &[Value],
    _now: i64,
    _cfg: &Map<String, Value>,
) -> Vec<Diagnostic> {
    let hits = active_hallucination_events(events, "completion_blocked_hallucination");
    if hits.is_empty() {
        return Vec::new();
    }
    let mut phantom_ids: Vec<String> = Vec::new();
    let first = event_ts(hits[0]);
    let last = event_ts(hits[hits.len() - 1]);
    for ev in &hits {
        let payload = parse_payload(ev);
        if let Some(Value::Array(arr)) = payload.get("phantom_cards") {
            for pid in arr {
                if let Value::String(pid) = pid {
                    if !phantom_ids.contains(pid) {
                        phantom_ids.push(pid.clone());
                    }
                }
            }
        }
    }
    let running = as_str(field(task, "status")) == "running";
    let mut actions: Vec<DiagnosticAction> = Vec::new();
    actions.push(DiagnosticAction::new("comment", "Add a comment explaining what to do").suggested(false));
    actions.extend(generic_recovery_actions(task, running));
    let mut data = Map::new();
    data.insert(
        "phantom_ids".to_string(),
        Value::Array(phantom_ids.into_iter().map(Value::String).collect()),
    );
    vec![Diagnostic {
        kind: "hallucinated_cards".to_string(),
        severity: "error".to_string(),
        title: "Worker claimed cards that don't exist".to_string(),
        detail: "The completing worker declared created_cards that either didn't \
exist or weren't created by its profile. The completion was \
blocked and the task stayed in its prior state. \
Usually means the worker hallucinated ids instead of capturing \
return values from kanban_create."
            .to_string(),
        actions,
        first_seen_at: first,
        last_seen_at: last,
        count: hits.len() as i64,
        run_id: None,
        data,
    }]
}

/// Advisory prose-scan: the completion summary mentions `t_<hex>` ids that
/// don't resolve. Non-blocking; surfaced as a warning only.
///
/// Auto-clears when a fresh clean completion arrives AFTER the suspected event.
fn rule_prose_phantom_refs(
    task: &Value,
    events: &[Value],
    _runs: &[Value],
    _now: i64,
    _cfg: &Map<String, Value>,
) -> Vec<Diagnostic> {
    let hits = active_hallucination_events(events, "suspected_hallucinated_references");
    if hits.is_empty() {
        return Vec::new();
    }
    let mut phantom_refs: Vec<String> = Vec::new();
    for ev in &hits {
        let payload = parse_payload(ev);
        if let Some(Value::Array(arr)) = payload.get("phantom_refs") {
            for pid in arr {
                if let Value::String(pid) = pid {
                    if !phantom_refs.contains(pid) {
                        phantom_refs.push(pid.clone());
                    }
                }
            }
        }
    }
    let running = as_str(field(task, "status")) == "running";
    let mut data = Map::new();
    data.insert(
        "phantom_refs".to_string(),
        Value::Array(phantom_refs.into_iter().map(Value::String).collect()),
    );
    vec![Diagnostic {
        kind: "prose_phantom_refs".to_string(),
        severity: "warning".to_string(),
        title: "Completion summary references unknown task ids".to_string(),
        detail: "The completion summary mentions task ids that don't resolve \
in this board's database. The completion itself succeeded, \
but downstream consumers parsing the summary may be pointed \
at cards that never existed."
            .to_string(),
        actions: generic_recovery_actions(task, running),
        first_seen_at: event_ts(hits[0]),
        last_seen_at: event_ts(hits[hits.len() - 1]),
        count: hits.len() as i64,
        run_id: None,
        data,
    }]
}

/// Order runs by their `id` field ascending. Stable so that ties keep input
/// order (matching Python's stable `sorted`).
fn order_runs_by_id(runs: &[Value]) -> Vec<&Value> {
    let mut ordered: Vec<&Value> = runs.iter().collect();
    ordered.sort_by_key(|r| as_i64(field(r, "id")));
    ordered
}

/// Task's unified `consecutive_failures` counter is climbing — something about
/// this task+profile combo is broken and each retry fails the same way.
/// Triggers regardless of the specific failure mode.
///
/// Threshold: `cfg["failure_threshold"]` (default 3). Accepts the legacy
/// `spawn_failure_threshold` config key for back-compat. The unified counter
/// falls back to the legacy `spawn_failures` column when absent.
fn rule_repeated_failures(
    task: &Value,
    _events: &[Value],
    runs: &[Value],
    now: i64,
    cfg: &Map<String, Value>,
) -> Vec<Diagnostic> {
    let threshold = cfg_int(
        cfg,
        "failure_threshold",
        cfg_int(cfg, "spawn_failure_threshold", 3),
    );
    // Read the new unified counter name, with a fallback to the legacy column
    // name. Python distinguishes "absent" (Null) from a present value: only
    // when consecutive_failures is null do we fall back to spawn_failures.
    let cf = field(task, "consecutive_failures");
    let (failures, failures_present): (i64, bool) = if !cf.is_null() {
        (as_i64(cf), true)
    } else {
        let sf = field(task, "spawn_failures");
        // Python `_task_field(..., "spawn_failures", 0)` defaults to 0 (present).
        (as_i64(sf), true)
    };
    let _ = failures_present;
    if failures < threshold {
        return Vec::new();
    }

    // last_failure_error with fallback to last_spawn_error.
    let lfe = field(task, "last_failure_error");
    let last_err: Option<String> = if !lfe.is_null() {
        opt_str(lfe)
    } else {
        opt_str(field(task, "last_spawn_error"))
    };
    let assignee = field(task, "assignee");
    let assignee_str = as_str(assignee);

    // Classify the most recent failure by peeking at run outcomes.
    let ordered_runs = order_runs_by_id(runs);
    let mut most_recent_outcome: Option<String> = None;
    for r in ordered_runs.iter().rev() {
        let oc = as_str(field(r, "outcome"));
        if oc == "spawn_failed" || oc == "timed_out" || oc == "crashed" {
            most_recent_outcome = Some(oc);
            break;
        }
    }

    let mut actions: Vec<DiagnosticAction> = Vec::new();
    let recent = most_recent_outcome.as_deref();
    if recent == Some("spawn_failed") && !assignee_str.is_empty() && assignee_str != "default" {
        // Spawn is failing specifically — profile setup issue.
        let cmd = format!("hermes -p {} doctor", assignee_str);
        let mut p = Map::new();
        p.insert("command".to_string(), Value::String(cmd.clone()));
        actions.push(
            DiagnosticAction::new("cli_hint", format!("Verify profile: {}", cmd))
                .with_payload(p)
                .suggested(true),
        );
        let cmd2 = format!("hermes -p {} auth", assignee_str);
        let mut p2 = Map::new();
        p2.insert("command".to_string(), Value::String(cmd2.clone()));
        actions.push(
            DiagnosticAction::new("cli_hint", format!("Fix profile auth: {}", cmd2))
                .with_payload(p2),
        );
    } else if recent == Some("timed_out") || recent == Some("crashed") {
        // Worker got off the ground but died. Logs are the right place to
        // diagnose; reclaim/reassign are the recovery levers.
        let task_id = field(task, "id");
        if is_truthy(task_id) {
            let cmd = format!("hermes kanban log {}", value_to_plain(task_id));
            let mut p = Map::new();
            p.insert("command".to_string(), Value::String(cmd.clone()));
            actions.push(
                DiagnosticAction::new("cli_hint", format!("Check logs: {}", cmd))
                    .with_payload(p)
                    .suggested(true),
            );
        }
    }
    let running = as_str(field(task, "status")) == "running";
    actions.extend(generic_recovery_actions(task, running));

    let severity = if failures >= threshold * 2 {
        "critical"
    } else {
        "error"
    };
    let err_text = last_err.as_deref().unwrap_or("").trim().to_string();
    let err_snippet = if err_text.is_empty() {
        String::new()
    } else {
        truncate_ellipsis(&err_text, 500)
    };
    let outcome_label = match recent {
        Some("spawn_failed") => "spawn",
        Some("timed_out") => "timeout",
        Some("crashed") => "crash",
        _ => "failure",
    };
    let (title, detail) = if !err_snippet.is_empty() {
        let headline = head_chars(first_line(&err_snippet), 160);
        let title = format!("Agent {} x{}: {}", outcome_label, failures, headline);
        let detail = format!(
            "This task has failed {} times in a row \
(most recent: {}). Full last error:\n\n\
{}\n\n\
The dispatcher will keep retrying until the consecutive-\
failures counter trips the circuit breaker (default 5), \
at which point the task auto-blocks. Fix the root cause \
and reclaim to retry.",
            failures, outcome_label, err_snippet
        );
        (title, detail)
    } else {
        let title = format!("Agent {} x{} (no error recorded)", outcome_label, failures);
        let detail = format!(
            "This task has failed {} times in a row \
(most recent: {}) but no error text was \
captured. Check the suggested command or the worker log.",
            failures, outcome_label
        );
        (title, detail)
    };

    let mut data = Map::new();
    data.insert("consecutive_failures".to_string(), json!(failures));
    data.insert(
        "most_recent_outcome".to_string(),
        match &most_recent_outcome {
            Some(s) => Value::String(s.clone()),
            None => Value::Null,
        },
    );
    data.insert(
        "last_error".to_string(),
        match &last_err {
            Some(s) => Value::String(s.clone()),
            None => Value::Null,
        },
    );

    vec![Diagnostic {
        kind: "repeated_failures".to_string(),
        severity: severity.to_string(),
        title,
        detail,
        actions,
        first_seen_at: now,
        last_seen_at: now,
        count: failures,
        run_id: None,
        data,
    }]
}

/// The worker spawns fine but keeps crashing mid-run. Check the last N runs'
/// outcomes; N consecutive `crashed` without a successful `completed` means
/// something about the task + profile combo is broken.
///
/// Threshold: `cfg["crash_threshold"]` (default 2). Suppresses itself when the
/// unified repeated-failures rule is also about to fire, to avoid
/// double-flagging.
fn rule_repeated_crashes(
    task: &Value,
    _events: &[Value],
    runs: &[Value],
    now: i64,
    cfg: &Map<String, Value>,
) -> Vec<Diagnostic> {
    let failure_threshold = cfg_int(
        cfg,
        "failure_threshold",
        cfg_int(cfg, "spawn_failure_threshold", 3),
    );
    // `_task_field(task, "consecutive_failures", 0) or 0` — default 0, and a
    // falsy (0/null) value also collapses to 0.
    let unified_counter = as_i64(field(task, "consecutive_failures"));
    // Unified rule will catch this — let it handle to avoid double fire.
    if unified_counter >= failure_threshold {
        return Vec::new();
    }

    let threshold = cfg_int(cfg, "crash_threshold", 2);
    let ordered = order_runs_by_id(runs);
    // Count trailing consecutive 'crashed' outcomes.
    let mut consecutive: i64 = 0;
    let mut last_err: Option<String> = None;
    let mut last_err_set = false;
    for r in ordered.iter().rev() {
        let outcome = as_str(field(r, "outcome"));
        if outcome == "crashed" {
            consecutive += 1;
            if !last_err_set {
                last_err = opt_str(field(r, "error"));
                last_err_set = true;
            }
        } else if outcome == "completed" || outcome == "reclaimed" {
            // A success (or manual reclaim) breaks the streak.
            break;
        } else {
            // Other outcomes (timed_out, blocked, spawn_failed, gave_up) aren't
            // crash signals — don't count them, but they also don't break the
            // crash streak.
            continue;
        }
    }
    if consecutive < threshold {
        return Vec::new();
    }
    let task_id = field(task, "id");
    let mut actions: Vec<DiagnosticAction> = Vec::new();
    if is_truthy(task_id) {
        let cmd = format!("hermes kanban log {}", value_to_plain(task_id));
        let mut p = Map::new();
        p.insert("command".to_string(), Value::String(cmd.clone()));
        actions.push(
            DiagnosticAction::new("cli_hint", format!("Check logs: {}", cmd))
                .with_payload(p)
                .suggested(true),
        );
    }
    let running = as_str(field(task, "status")) == "running";
    actions.extend(generic_recovery_actions(task, running));
    let severity = if consecutive >= threshold * 2 {
        "critical"
    } else {
        "error"
    };
    let err_text = last_err.as_deref().unwrap_or("").trim().to_string();
    let err_snippet = if err_text.is_empty() {
        String::new()
    } else {
        truncate_ellipsis(&err_text, 500)
    };
    let (title, detail) = if !err_snippet.is_empty() {
        let headline = head_chars(first_line(&err_snippet), 160);
        let title = format!("Agent crashed {}x: {}", consecutive, headline);
        let detail = format!(
            "The last {} runs ended with outcome=crashed. \
Full last error:\n\n{}",
            consecutive, err_snippet
        );
        (title, detail)
    } else {
        let title = format!("Agent crashed {}x (no error recorded)", consecutive);
        let detail = format!(
            "The last {} runs ended with outcome=crashed but \
no error text was captured. Check the worker log for more.",
            consecutive
        );
        (title, detail)
    };

    let mut data = Map::new();
    data.insert("consecutive_crashes".to_string(), json!(consecutive));
    data.insert(
        "last_error".to_string(),
        match &last_err {
            Some(s) => Value::String(s.clone()),
            None => Value::Null,
        },
    );

    vec![Diagnostic {
        kind: "repeated_crashes".to_string(),
        severity: severity.to_string(),
        title,
        detail,
        actions,
        first_seen_at: now,
        last_seen_at: now,
        count: consecutive,
        run_id: None,
        data,
    }]
}

/// Task has been in `blocked` status for too long without a comment.
///
/// Threshold: `cfg["blocked_stale_hours"]` (default 24). Surfaced as a warning
/// so humans know there's a pending unblock.
fn rule_stuck_in_blocked(
    task: &Value,
    events: &[Value],
    _runs: &[Value],
    now: i64,
    cfg: &Map<String, Value>,
) -> Vec<Diagnostic> {
    let hours = cfg_float(cfg, "blocked_stale_hours", 24.0);
    let status = as_str(field(task, "status"));
    if status != "blocked" {
        return Vec::new();
    }
    // Find the most recent ``blocked`` event.
    let mut last_blocked_ts = 0i64;
    for ev in events {
        if event_kind(ev) == "blocked" {
            let t = event_ts(ev);
            if t > last_blocked_ts {
                last_blocked_ts = t;
            }
        }
    }
    if last_blocked_ts == 0 {
        return Vec::new();
    }
    let age_hours = (now - last_blocked_ts) as f64 / 3600.0;
    if age_hours < hours {
        return Vec::new();
    }
    // Any comment / unblock after the block breaks the "stale" signal.
    for ev in events {
        let k = event_kind(ev);
        if (k == "commented" || k == "unblocked") && event_ts(ev) > last_blocked_ts {
            return Vec::new();
        }
    }
    let actions: Vec<DiagnosticAction> = vec![
        DiagnosticAction::new("comment", "Add a comment / unblock the task").suggested(true),
    ];
    let age_int = age_hours as i64; // int() truncates toward zero
    let mut data = Map::new();
    data.insert("blocked_at".to_string(), json!(last_blocked_ts));
    data.insert("age_hours".to_string(), json!(round1(age_hours)));
    vec![Diagnostic {
        kind: "stuck_in_blocked".to_string(),
        severity: "warning".to_string(),
        title: format!("Task has been blocked for {}h", age_int),
        detail: format!(
            "This task transitioned to blocked {}h ago and \
has had no comments or unblock attempts since. Blocked tasks \
are waiting for human input — check the block reason and \
either unblock with feedback or answer with a comment.",
            age_int
        ),
        actions,
        first_seen_at: last_blocked_ts,
        last_seen_at: last_blocked_ts,
        count: 1,
        run_id: None,
        data,
    }]
}

/// Round to one decimal place, mirroring Python `round(x, 1)` for the values
/// these rules produce. Banker's rounding edge cases are not exercised by the
/// data here (durations), so this uses straightforward half-away rounding.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Truthiness for a task id, mirroring `if task_id:` in Python — None/empty
/// string/zero are falsy.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Render a JSON scalar as it would appear inside a command string. Strings
/// render without quotes; numbers/bools as their text; everything else empty.
fn value_to_plain(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Registry + public entry points
// ---------------------------------------------------------------------------

type RuleFn = fn(&Value, &[Value], &[Value], i64, &Map<String, Value>) -> Vec<Diagnostic>;

/// Registry — order matters: rules higher on the list render first when
/// severity ties (via the stable sort). Add new rules here.
const RULES: [RuleFn; 5] = [
    rule_hallucinated_cards,
    rule_prose_phantom_refs,
    rule_repeated_failures,
    rule_repeated_crashes,
    rule_stuck_in_blocked,
];

/// Index of a severity in [`SEVERITY_ORDER`], or `-1` if unknown.
fn severity_idx(sev: &str) -> i64 {
    SEVERITY_ORDER
        .iter()
        .position(|s| *s == sev)
        .map(|i| i as i64)
        .unwrap_or(-1)
}

/// Run every rule against a single task's state and return a severity-sorted
/// list of active diagnostics.
///
/// Sorting: critical first, then error, then warning; ties broken by
/// most-recent `last_seen_at`. The sort is stable, so registry order is the
/// final tiebreak (matching Python).
///
/// `now` defaults to the current unix time when `None`. `config` overlays onto
/// [`default_config`].
pub fn compute_task_diagnostics(
    task: &Value,
    events: &[Value],
    runs: &[Value],
    now: Option<i64>,
    config: Option<&Map<String, Value>>,
) -> Vec<Diagnostic> {
    let now_ts = now.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    });
    let mut cfg = default_config();
    if let Some(overlay) = config {
        for (k, v) in overlay {
            cfg.insert(k.clone(), v.clone());
        }
    }
    let mut out: Vec<Diagnostic> = Vec::new();
    for rule in RULES.iter() {
        // A broken rule must never crash the dashboard. The Python original
        // wrapped each rule in try/except; our rules don't panic on normal
        // input, but we still guard with catch_unwind to preserve the
        // "drop the diagnostic, never 500" contract.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rule(task, events, runs, now_ts, &cfg)
        }));
        if let Ok(diags) = result {
            out.extend(diags);
        }
    }
    // Stable sort: critical first, then error, then warning; ties by most
    // recent last_seen_at. Negated keys give descending order.
    out.sort_by(|a, b| {
        let ka = (-severity_idx(&a.severity), -a.last_seen_at);
        let kb = (-severity_idx(&b.severity), -b.last_seen_at);
        ka.cmp(&kb)
    });
    out
}

/// Highest severity present in the list, or `None` if empty. Useful for card
/// badges that need a single color.
pub fn severity_of_highest(diagnostics: &[Diagnostic]) -> Option<String> {
    let mut highest_idx: i64 = -1;
    let mut highest: Option<String> = None;
    for d in diagnostics {
        let idx = SEVERITY_ORDER
            .iter()
            .position(|s| *s == d.severity)
            .map(|i| i as i64)
            .unwrap_or(-1);
        if idx > highest_idx {
            highest_idx = idx;
            highest = Some(d.severity.clone());
        }
    }
    highest
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Value {
        v
    }

    #[test]
    fn no_diagnostics_for_clean_task() {
        let task = json!({"id": "t_1", "status": "ready", "consecutive_failures": 0});
        let diags = compute_task_diagnostics(&task, &[], &[], Some(1000), None);
        assert!(diags.is_empty());
    }

    #[test]
    fn hallucinated_cards_fires_and_collects_phantoms() {
        let task = json!({"id": "t_1", "status": "running"});
        let events = vec![obj(json!({
            "kind": "completion_blocked_hallucination",
            "created_at": 500,
            "payload": {"phantom_cards": ["t_aaa", "t_bbb"]}
        }))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(1000), None);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.kind, "hallucinated_cards");
        assert_eq!(d.severity, "error");
        assert_eq!(d.count, 1);
        assert_eq!(d.first_seen_at, 500);
        assert_eq!(
            d.data.get("phantom_ids").unwrap(),
            &json!(["t_aaa", "t_bbb"])
        );
        // running task gets a reclaim action + reassign + the comment action.
        assert!(d.actions.iter().any(|a| a.kind == "reclaim"));
        assert!(d.actions.iter().any(|a| a.kind == "comment"));
    }

    #[test]
    fn hallucination_payload_as_json_string() {
        let task = json!({"id": "t_1", "status": "running"});
        let events = vec![obj(json!({
            "kind": "completion_blocked_hallucination",
            "created_at": 7,
            "payload": "{\"phantom_cards\": [\"t_zzz\"]}"
        }))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(10), None);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].data.get("phantom_ids").unwrap(), &json!(["t_zzz"]));
    }

    #[test]
    fn hallucination_auto_clears_after_completed() {
        let task = json!({"id": "t_1", "status": "ready"});
        let events = vec![
            obj(json!({"kind": "completion_blocked_hallucination", "created_at": 1})),
            obj(json!({"kind": "completed", "created_at": 2})),
        ];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(100), None);
        assert!(diags.is_empty());
    }

    #[test]
    fn prose_phantom_refs_warning() {
        let task = json!({"id": "t_1", "status": "ready"});
        let events = vec![obj(json!({
            "kind": "suspected_hallucinated_references",
            "created_at": 3,
            "payload": {"phantom_refs": ["t_p1"]}
        }))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(50), None);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].kind, "prose_phantom_refs");
        assert_eq!(diags[0].severity, "warning");
        assert_eq!(diags[0].data.get("phantom_refs").unwrap(), &json!(["t_p1"]));
    }

    #[test]
    fn repeated_failures_threshold_and_error_severity() {
        let task = json!({
            "id": "t_9",
            "status": "running",
            "consecutive_failures": 3,
            "last_failure_error": "boom line one\nline two",
            "assignee": "prof"
        });
        let runs = vec![obj(json!({"id": 1, "outcome": "crashed", "error": "x"}))];
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.kind, "repeated_failures");
        assert_eq!(d.severity, "error");
        assert_eq!(d.count, 3);
        assert!(d.title.contains("Agent crash x3"));
        assert!(d.title.contains("boom line one"));
        assert_eq!(d.data.get("most_recent_outcome").unwrap(), &json!("crashed"));
    }

    #[test]
    fn repeated_failures_critical_at_double_threshold() {
        let task = json!({
            "id": "t_9",
            "status": "ready",
            "consecutive_failures": 6
        });
        let diags = compute_task_diagnostics(&task, &[], &[], Some(1000), None);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, "critical");
    }

    #[test]
    fn repeated_failures_spawn_failed_with_assignee_emits_doctor_hint() {
        let task = json!({
            "id": "t_9",
            "status": "running",
            "consecutive_failures": 3,
            "assignee": "myprofile"
        });
        let runs = vec![obj(json!({"id": 1, "outcome": "spawn_failed"}))];
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        let d = &diags[0];
        let hint = d
            .actions
            .iter()
            .find(|a| a.kind == "cli_hint")
            .expect("expected a cli_hint");
        assert!(hint.suggested);
        assert_eq!(
            hint.payload.get("command").unwrap(),
            &json!("hermes -p myprofile doctor")
        );
    }

    #[test]
    fn repeated_failures_legacy_spawn_failures_fallback() {
        // consecutive_failures absent -> fall back to spawn_failures.
        let task = json!({
            "id": "t_9",
            "status": "ready",
            "spawn_failures": 4,
            "last_spawn_error": "legacy err"
        });
        let diags = compute_task_diagnostics(&task, &[], &[], Some(1000), None);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].count, 4);
        assert!(diags[0].title.contains("legacy err"));
    }

    #[test]
    fn repeated_crashes_fires_below_unified_threshold() {
        let task = json!({"id": "t_3", "status": "running", "consecutive_failures": 0});
        let runs = vec![
            obj(json!({"id": 1, "outcome": "crashed", "error": "segfault"})),
            obj(json!({"id": 2, "outcome": "crashed", "error": "oom"})),
        ];
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.kind, "repeated_crashes");
        assert_eq!(d.count, 2);
        // last_err is the most-recent crash (highest id) -> "oom".
        assert!(d.title.contains("oom"));
        assert_eq!(d.data.get("last_error").unwrap(), &json!("oom"));
    }

    #[test]
    fn repeated_crashes_suppressed_when_unified_rule_fires() {
        let task = json!({"id": "t_3", "status": "running", "consecutive_failures": 3});
        let runs = vec![
            obj(json!({"id": 1, "outcome": "crashed"})),
            obj(json!({"id": 2, "outcome": "crashed"})),
        ];
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        // Only the unified repeated_failures rule, not repeated_crashes.
        assert!(diags.iter().any(|d| d.kind == "repeated_failures"));
        assert!(!diags.iter().any(|d| d.kind == "repeated_crashes"));
    }

    #[test]
    fn repeated_crashes_streak_broken_by_completed() {
        let task = json!({"id": "t_3", "status": "ready", "consecutive_failures": 0});
        let runs = vec![
            obj(json!({"id": 1, "outcome": "crashed"})),
            obj(json!({"id": 2, "outcome": "completed"})),
            obj(json!({"id": 3, "outcome": "crashed"})),
        ];
        // Trailing streak is just the id=3 crash -> below threshold 2.
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        assert!(diags.is_empty());
    }

    #[test]
    fn repeated_crashes_non_crash_non_break_outcomes_skipped() {
        let task = json!({"id": "t_3", "status": "ready", "consecutive_failures": 0});
        // timed_out is neither a crash nor a streak-breaker.
        let runs = vec![
            obj(json!({"id": 1, "outcome": "crashed"})),
            obj(json!({"id": 2, "outcome": "timed_out"})),
            obj(json!({"id": 3, "outcome": "crashed"})),
        ];
        let diags = compute_task_diagnostics(&task, &[], &runs, Some(1000), None);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].kind, "repeated_crashes");
        assert_eq!(diags[0].count, 2);
    }

    #[test]
    fn stuck_in_blocked_fires_after_threshold() {
        let now = 1_000_000;
        let blocked_at = now - 25 * 3600; // 25h ago
        let task = json!({"id": "t_b", "status": "blocked"});
        let events = vec![obj(json!({"kind": "blocked", "created_at": blocked_at}))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(now), None);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.kind, "stuck_in_blocked");
        assert_eq!(d.severity, "warning");
        assert_eq!(d.title, "Task has been blocked for 25h");
        assert_eq!(d.data.get("blocked_at").unwrap(), &json!(blocked_at));
    }

    #[test]
    fn stuck_in_blocked_cleared_by_later_comment() {
        let now = 1_000_000;
        let blocked_at = now - 25 * 3600;
        let task = json!({"id": "t_b", "status": "blocked"});
        let events = vec![
            obj(json!({"kind": "blocked", "created_at": blocked_at})),
            obj(json!({"kind": "commented", "created_at": blocked_at + 10})),
        ];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(now), None);
        assert!(diags.is_empty());
    }

    #[test]
    fn stuck_in_blocked_not_stale_yet() {
        let now = 1_000_000;
        let blocked_at = now - 3600; // 1h ago, threshold 24h
        let task = json!({"id": "t_b", "status": "blocked"});
        let events = vec![obj(json!({"kind": "blocked", "created_at": blocked_at}))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(now), None);
        assert!(diags.is_empty());
    }

    #[test]
    fn severity_sorting_critical_first() {
        // Build a task that fires repeated_failures (critical) and is also
        // blocked... but blocked + running can't coexist; use two separate
        // checks instead. Here we just sort a hand-built list.
        let warn = Diagnostic {
            severity: "warning".to_string(),
            last_seen_at: 100,
            ..Diagnostic::default()
        };
        let crit = Diagnostic {
            severity: "critical".to_string(),
            last_seen_at: 50,
            ..Diagnostic::default()
        };
        let err = Diagnostic {
            severity: "error".to_string(),
            last_seen_at: 75,
            ..Diagnostic::default()
        };
        let mut list = vec![warn, crit, err];
        list.sort_by(|a, b| {
            let ka = (-severity_idx(&a.severity), -a.last_seen_at);
            let kb = (-severity_idx(&b.severity), -b.last_seen_at);
            ka.cmp(&kb)
        });
        assert_eq!(list[0].severity, "critical");
        assert_eq!(list[1].severity, "error");
        assert_eq!(list[2].severity, "warning");
    }

    #[test]
    fn severity_of_highest_picks_critical() {
        let diags = vec![
            Diagnostic {
                severity: "warning".to_string(),
                ..Diagnostic::default()
            },
            Diagnostic {
                severity: "critical".to_string(),
                ..Diagnostic::default()
            },
            Diagnostic {
                severity: "error".to_string(),
                ..Diagnostic::default()
            },
        ];
        assert_eq!(severity_of_highest(&diags), Some("critical".to_string()));
        assert_eq!(severity_of_highest(&[]), None);
    }

    #[test]
    fn config_override_lowers_failure_threshold() {
        let task = json!({"id": "t_9", "status": "ready", "consecutive_failures": 1});
        let mut cfg = Map::new();
        cfg.insert("failure_threshold".to_string(), json!(1));
        let diags = compute_task_diagnostics(&task, &[], &[], Some(1000), Some(&cfg));
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].kind, "repeated_failures");
    }

    #[test]
    fn error_snippet_truncated_to_500_chars() {
        let long = "x".repeat(600);
        let task = json!({
            "id": "t_9",
            "status": "ready",
            "consecutive_failures": 3,
            "last_failure_error": long
        });
        let diags = compute_task_diagnostics(&task, &[], &[], Some(1000), None);
        let d = &diags[0];
        // detail contains the snippet with trailing ellipsis.
        assert!(d.detail.contains('…'));
    }

    #[test]
    fn to_dict_round_trips_shape() {
        let task = json!({"id": "t_1", "status": "running"});
        let events = vec![obj(json!({
            "kind": "completion_blocked_hallucination",
            "created_at": 500,
            "payload": {"phantom_cards": ["t_aaa"]}
        }))];
        let diags = compute_task_diagnostics(&task, &events, &[], Some(1000), None);
        let v = diags[0].to_dict();
        assert_eq!(v["kind"], json!("hallucinated_cards"));
        assert_eq!(v["severity"], json!("error"));
        assert!(v["actions"].is_array());
        assert_eq!(v["run_id"], Value::Null);
        assert_eq!(v["data"]["phantom_ids"], json!(["t_aaa"]));
    }
}
