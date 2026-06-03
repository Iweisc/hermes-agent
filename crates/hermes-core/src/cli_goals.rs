//! Persistent session goals — the Ralph loop for Hermes (native Rust port).
//!
//! A goal is a free-form user objective that stays active across turns. After
//! each turn completes, a small judge call asks an auxiliary model "is this
//! goal satisfied by the assistant's last response?". If not, Hermes feeds a
//! continuation prompt back into the same session and keeps working until the
//! goal is done, the turn budget is exhausted, the user pauses/clears it, or
//! the user sends a new message (which takes priority and pauses the loop).
//!
//! State is persisted in the SessionDB `state_meta` table keyed by
//! `goal:<session_id>` so `/resume` picks it up.
//!
//! Design notes / invariants (ported verbatim from `hermes_cli/goals.py`):
//!
//! - The continuation prompt is just a normal user message appended to the
//!   session. No system-prompt mutation, no toolset swap — caching stays intact.
//! - Judge failures are fail-OPEN: `continue`. A broken judge must not wedge
//!   progress; the turn budget is the backstop.
//! - When a real user message arrives mid-loop it preempts the continuation
//!   prompt and also pauses the goal loop for that turn.
//!
//! Nothing in this module touches the agent's system prompt or toolset.
//!
//! ## Port notes
//!
//! The Python module reaches into `agent.auxiliary_client.get_text_auxiliary_client`
//! to obtain an OpenAI-compatible chat client and fires a `chat.completions`
//! call. That client-construction surface lives elsewhere in the Rust port, so
//! here `judge_goal` is parameterised over a [`JudgeBackend`] trait: the pure
//! request-construction (`build_judge_messages`) and response-parsing
//! (`parse_judge_response`) logic — the parts that actually carry behaviour —
//! are reproduced exactly, while the transport is injected. A [`NoBackend`]
//! reproduces the "auxiliary client unavailable -> continue" path.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ──────────────────────────────────────────────────────────────────────
// Constants & defaults
// ──────────────────────────────────────────────────────────────────────

pub const DEFAULT_MAX_TURNS: i64 = 20;
pub const DEFAULT_JUDGE_TIMEOUT: f64 = 30.0;
/// Cap how much of the last response we send to the judge.
const JUDGE_RESPONSE_SNIPPET_CHARS: usize = 4000;
/// Cap how much of the goal text we send to the judge.
const JUDGE_GOAL_SNIPPET_CHARS: usize = 2000;

pub const CONTINUATION_PROMPT_TEMPLATE: &str = concat!(
    "[Continuing toward your standing goal]\n",
    "Goal: {goal}\n\n",
    "Continue working toward this goal. Take the next concrete step. ",
    "If you believe the goal is complete, state so explicitly and stop. ",
    "If you are blocked and need input from the user, say so clearly and stop."
);

pub const JUDGE_SYSTEM_PROMPT: &str = concat!(
    "You are a strict judge evaluating whether an autonomous agent has ",
    "achieved a user's stated goal. You receive the goal text and the ",
    "agent's most recent response. Your only job is to decide whether ",
    "the goal is fully satisfied based on that response.\n\n",
    "A goal is DONE only when:\n",
    "- The response explicitly confirms the goal was completed, OR\n",
    "- The response clearly shows the final deliverable was produced, OR\n",
    "- The response explains the goal is unachievable / blocked / needs ",
    "user input (treat this as DONE with reason describing the block).\n\n",
    "Otherwise the goal is NOT done — CONTINUE.\n\n",
    "Reply ONLY with a single JSON object on one line:\n",
    "{\"done\": <true|false>, \"reason\": \"<one-sentence rationale>\"}"
);

pub const JUDGE_USER_PROMPT_TEMPLATE: &str = concat!(
    "Goal:\n{goal}\n\n",
    "Agent's most recent response:\n{response}\n\n",
    "Is the goal satisfied?"
);

/// Build the continuation user message for a goal (mirrors the Python format()).
pub fn continuation_prompt(goal: &str) -> String {
    CONTINUATION_PROMPT_TEMPLATE.replace("{goal}", goal)
}

// ──────────────────────────────────────────────────────────────────────
// GoalState
// ──────────────────────────────────────────────────────────────────────

/// Serializable goal state stored per session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalState {
    pub goal: String,
    /// active | paused | done | cleared
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub turns_used: i64,
    #[serde(default = "default_max_turns")]
    pub max_turns: i64,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub last_turn_at: f64,
    /// "done" | "continue" | "skipped"
    #[serde(default)]
    pub last_verdict: Option<String>,
    #[serde(default)]
    pub last_reason: Option<String>,
    /// why we auto-paused (budget, etc.)
    #[serde(default)]
    pub paused_reason: Option<String>,
}

fn default_status() -> String {
    "active".to_string()
}

fn default_max_turns() -> i64 {
    DEFAULT_MAX_TURNS
}

impl Default for GoalState {
    fn default() -> Self {
        GoalState {
            goal: String::new(),
            status: default_status(),
            turns_used: 0,
            max_turns: DEFAULT_MAX_TURNS,
            created_at: 0.0,
            last_turn_at: 0.0,
            last_verdict: None,
            last_reason: None,
            paused_reason: None,
        }
    }
}

impl GoalState {
    pub fn new(goal: impl Into<String>) -> Self {
        GoalState {
            goal: goal.into(),
            ..Default::default()
        }
    }

    /// Serialize to JSON (Python `to_json`, `ensure_ascii=False`).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse stored JSON, tolerating missing/mistyped fields exactly like the
    /// Python `from_json` (which coerces with `int(... or default)` etc.).
    pub fn from_json(raw: &str) -> Result<Self, serde_json::Error> {
        let data: Value = serde_json::from_str(raw)?;
        Ok(GoalState::from_value(&data))
    }

    fn from_value(data: &Value) -> Self {
        let get_str = |k: &str| -> Option<String> {
            data.get(k)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        };
        // Mirror Python `int(data.get(k, default) or default)`: treat null,
        // 0, and missing as the default in the "or" sense only for max_turns;
        // for turns_used the Python uses `or 0` so 0/None both -> 0.
        let goal = get_str("goal").unwrap_or_default();
        let status = get_str("status").unwrap_or_else(default_status);
        let turns_used = coerce_int(data.get("turns_used"), 0).max(0);
        let max_turns_raw = coerce_int(data.get("max_turns"), DEFAULT_MAX_TURNS);
        // `int(x or DEFAULT)` -> if x is falsy (0 / null / missing) use DEFAULT.
        let max_turns = if max_turns_raw == 0 {
            DEFAULT_MAX_TURNS
        } else {
            max_turns_raw
        };
        let created_at = coerce_float(data.get("created_at"), 0.0);
        let last_turn_at = coerce_float(data.get("last_turn_at"), 0.0);
        // `data.get("last_verdict")` returns None if missing OR if explicitly
        // null; a non-string value would be passed through in Python but here
        // we only keep strings (verdicts are always strings in practice).
        let last_verdict = get_str("last_verdict");
        let last_reason = get_str("last_reason");
        let paused_reason = get_str("paused_reason");

        GoalState {
            goal,
            status,
            turns_used,
            max_turns,
            created_at,
            last_turn_at,
            last_verdict,
            last_reason,
            paused_reason,
        }
    }
}

/// Coerce a JSON value to i64 the way Python's `int(x or default)` would for
/// the goals fields: numbers are truncated, strings parsed, and any
/// falsy/None/missing collapses to `default`.
fn coerce_int(v: Option<&Value>, default: i64) -> i64 {
    match v {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                f as i64
            } else {
                default
            }
        }
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(default),
        Some(Value::Bool(b)) => {
            if *b {
                1
            } else {
                default
            }
        }
        _ => default,
    }
}

fn coerce_float(v: Option<&Value>, default: f64) -> f64 {
    match v {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(default),
        Some(Value::Bool(b)) => {
            if *b {
                1.0
            } else {
                default
            }
        }
        _ => default,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Persistence (SessionDB state_meta)
// ──────────────────────────────────────────────────────────────────────

fn meta_key(session_id: &str) -> String {
    format!("goal:{session_id}")
}

/// Minimal accessor over the SessionDB `state_meta` table.
///
/// The Python module caches one `SessionDB` per `hermes_home`. Here the caller
/// owns the connection (or path); a [`GoalStore`] wraps a single connection and
/// exposes the get/set/delete primitives the goal persistence needs. All
/// operations are best-effort and return `Result` so callers can choose to
/// log-and-ignore (matching Python's defensive `try/except`).
pub struct GoalStore {
    conn: Connection,
}

impl GoalStore {
    /// Open (or create) the state DB at `path` and ensure the `state_meta`
    /// table exists. Mirrors `SessionDB` only insofar as the goal code needs.
    pub fn open(path: &std::path::Path) -> rusqlite::Result<Self> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )?;
        Ok(GoalStore { conn })
    }

    /// Use an already-open connection (e.g. one shared with the SessionStore).
    pub fn from_connection(conn: Connection) -> rusqlite::Result<Self> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )?;
        Ok(GoalStore { conn })
    }

    /// In-memory store, handy for tests and non-standard launchers.
    pub fn in_memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )?;
        Ok(GoalStore { conn })
    }

    fn get_meta(&self, key: &str) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM state_meta WHERE key = ?1",
                [key],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .map(|opt| opt.flatten())
    }

    fn set_meta(&self, key: &str, value: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    /// Load the goal for a session, or `None` if none exists.
    /// Returns `None` (not an error) on empty session id, missing row, or parse
    /// failure — matching Python's defensive `load_goal`.
    pub fn load_goal(&self, session_id: &str) -> Option<GoalState> {
        if session_id.is_empty() {
            return None;
        }
        let raw = match self.get_meta(&meta_key(session_id)) {
            Ok(Some(raw)) => raw,
            Ok(None) => return None,
            Err(_) => return None,
        };
        if raw.is_empty() {
            return None;
        }
        GoalState::from_json(&raw).ok()
    }

    /// Persist a goal to SessionDB. No-op (Ok) if the session id is empty.
    pub fn save_goal(&self, session_id: &str, state: &GoalState) -> rusqlite::Result<()> {
        if session_id.is_empty() {
            return Ok(());
        }
        self.set_meta(&meta_key(session_id), &state.to_json())
    }

    /// Mark a goal cleared in the DB (preserved for audit, status=cleared).
    pub fn clear_goal(&self, session_id: &str) -> rusqlite::Result<()> {
        if let Some(mut state) = self.load_goal(session_id) {
            state.status = "cleared".to_string();
            self.save_goal(session_id, &state)?;
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────
// Judge
// ──────────────────────────────────────────────────────────────────────

/// Truncate `text` to `limit` chars (by Unicode scalar, matching Python str
/// slicing), appending the truncation marker the Python uses.
pub fn truncate(text: &str, limit: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}… [truncated]")
}

/// Verdict returned by the judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Done,
    Continue,
    Skipped,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Done => "done",
            Verdict::Continue => "continue",
            Verdict::Skipped => "skipped",
        }
    }
}

/// Parse the judge's reply. Fail-open to `(false, "<reason>")`.
/// Returns `(done, reason)`.
pub fn parse_judge_response(raw: &str) -> (bool, String) {
    if raw.is_empty() {
        return (false, "judge returned empty response".to_string());
    }

    let mut text = raw.trim().to_string();

    // Strip markdown code fences the model may wrap JSON in.
    if text.starts_with("```") {
        // Python `text.strip("`")` strips backticks from BOTH ends.
        text = text.trim_matches('`').to_string();
        // Peel off a leading json/JSON/etc tag (up to and incl. first newline).
        if let Some(nl) = text.find('\n') {
            text = text[nl + 1..].to_string();
        }
    }

    // First try: parse the whole blob.
    let mut data: Option<Value> = serde_json::from_str(text.trim()).ok();

    // Second try: pull the first JSON object out (non-greedy `\{.*?\}`,
    // DOTALL). We reproduce the smallest `{...}` span starting at the first
    // `{` — matching the non-greedy regex behaviour.
    if data.is_none() {
        if let Some(obj) = first_json_object(&text) {
            data = serde_json::from_str(&obj).ok();
        }
    }

    let obj = match data.as_ref().and_then(|v| v.as_object()) {
        Some(o) => o,
        None => {
            // Python uses repr() of the truncated raw, wrapped in quotes.
            return (
                false,
                format!("judge reply was not JSON: {:?}", truncate(raw, 200)),
            );
        }
    };

    let done = match obj.get("done") {
        Some(Value::String(s)) => {
            let lowered = s.trim().to_lowercase();
            matches!(lowered.as_str(), "true" | "yes" | "1" | "done")
        }
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::Null) | None => false,
        Some(other) => {
            // Python bool() of a non-empty container/object is True.
            !other.is_null()
        }
    };

    let reason = obj
        .get("reason")
        .map(json_reason_to_string)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "no reason provided".to_string());

    (done, reason)
}

/// `str(data.get("reason") or "")` — coerce the reason value to a string.
fn json_reason_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                String::new()
            }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Find the first `{...}` object substring using the same non-greedy semantics
/// as Python's `re.compile(r"\{.*?\}", re.DOTALL)`: from the first `{`, take the
/// shortest span ending at the next `}`.
fn first_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let rest = &text[start..];
    let end_rel = rest.find('}')?;
    Some(rest[..=end_rel].to_string())
}

/// Build the judge chat messages: `[system, user]` with the goal/response
/// truncated to the snippet caps. The returned tuples are `(role, content)`,
/// ready to feed into an OpenAI-compatible `chat.completions.create` call.
pub fn build_judge_messages(goal: &str, last_response: &str) -> Vec<(String, String)> {
    let prompt = JUDGE_USER_PROMPT_TEMPLATE
        .replace("{goal}", &truncate(goal, JUDGE_GOAL_SNIPPET_CHARS))
        .replace(
            "{response}",
            &truncate(last_response, JUDGE_RESPONSE_SNIPPET_CHARS),
        );
    vec![
        ("system".to_string(), JUDGE_SYSTEM_PROMPT.to_string()),
        ("user".to_string(), prompt),
    ]
}

/// Transport for the goal judge. Implementors fire the chat-completions call
/// against the auxiliary model and return the raw assistant text.
///
/// Returning `Err(JudgeError::Unavailable)` reproduces the Python "auxiliary
/// client unavailable / no client configured" path (verdict `continue`);
/// `Err(JudgeError::Api)` reproduces an API-call failure (also `continue`).
pub trait JudgeBackend {
    /// Send `messages` to the auxiliary model and return the raw reply text.
    ///
    /// `timeout` is the per-request timeout in seconds. Implementations are
    /// expected to use `temperature=0` and a small `max_tokens` (the Python
    /// uses 200) — the [`build_judge_messages`] output is passed straight in.
    fn judge(&self, messages: &[(String, String)], timeout: f64) -> Result<String, JudgeError>;
}

/// Reasons a judge backend cannot produce a verdict — all fail-open.
#[derive(Debug, Clone)]
pub enum JudgeError {
    /// Auxiliary client could not be imported/constructed or none configured.
    Unavailable(String),
    /// The API call itself failed; carries the error type name (Python uses
    /// `type(exc).__name__`).
    Api(String),
}

/// A backend that is always unavailable — reproduces the Python branch where
/// the auxiliary client cannot be obtained. Always yields verdict `continue`.
pub struct NoBackend;

impl JudgeBackend for NoBackend {
    fn judge(&self, _messages: &[(String, String)], _timeout: f64) -> Result<String, JudgeError> {
        Err(JudgeError::Unavailable(
            "auxiliary client unavailable".to_string(),
        ))
    }
}

/// Ask the auxiliary model whether the goal is satisfied.
///
/// Returns `(verdict, reason)` where verdict is [`Verdict::Done`],
/// [`Verdict::Continue`], or [`Verdict::Skipped`] (when the judge couldn't be
/// reached because the goal/response were empty in a way that short-circuits).
///
/// Deliberately fail-open: any backend error returns `(Continue, "...")` so a
/// broken judge doesn't wedge progress — the turn budget is the backstop.
pub fn judge_goal<B: JudgeBackend + ?Sized>(
    backend: &B,
    goal: &str,
    last_response: &str,
    timeout: f64,
) -> (Verdict, String) {
    if goal.trim().is_empty() {
        return (Verdict::Skipped, "empty goal".to_string());
    }
    if last_response.trim().is_empty() {
        // No substantive reply this turn — almost certainly not done yet.
        return (
            Verdict::Continue,
            "empty response (nothing to evaluate)".to_string(),
        );
    }

    let messages = build_judge_messages(goal, last_response);
    let raw = match backend.judge(&messages, timeout) {
        Ok(raw) => raw,
        Err(JudgeError::Unavailable(_)) => {
            return (
                Verdict::Continue,
                "auxiliary client unavailable".to_string(),
            );
        }
        Err(JudgeError::Api(type_name)) => {
            return (Verdict::Continue, format!("judge error: {type_name}"));
        }
    };

    let (done, reason) = parse_judge_response(&raw);
    let verdict = if done {
        Verdict::Done
    } else {
        Verdict::Continue
    };
    (verdict, reason)
}

/// Convenience: judge with the default timeout.
pub fn judge_goal_default<B: JudgeBackend + ?Sized>(
    backend: &B,
    goal: &str,
    last_response: &str,
) -> (Verdict, String) {
    judge_goal(backend, goal, last_response, DEFAULT_JUDGE_TIMEOUT)
}

// ──────────────────────────────────────────────────────────────────────
// GoalManager — the orchestration surface CLI + gateway talk to
// ──────────────────────────────────────────────────────────────────────

/// Decision returned by [`GoalManager::evaluate_after_turn`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalDecision {
    /// Current goal status after update (None if there was no goal).
    pub status: Option<String>,
    /// Caller should fire another turn.
    pub should_continue: bool,
    pub continuation_prompt: Option<String>,
    /// "done" | "continue" | "skipped" | "inactive"
    pub verdict: String,
    pub reason: String,
    /// User-visible one-liner to print/send.
    pub message: String,
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Per-session goal state + continuation decisions.
///
/// The CLI and gateway each hold one `GoalManager` per live session. The
/// manager borrows a [`GoalStore`] for persistence; the Python version reaches
/// into a module-level cached `SessionDB`, but here the store is explicit so
/// callers control connection lifetime.
pub struct GoalManager<'a> {
    pub session_id: String,
    pub default_max_turns: i64,
    store: &'a GoalStore,
    state: Option<GoalState>,
}

impl<'a> GoalManager<'a> {
    /// Construct a manager, loading any persisted goal for `session_id`.
    pub fn new(session_id: impl Into<String>, store: &'a GoalStore) -> Self {
        Self::with_default_max_turns(session_id, store, DEFAULT_MAX_TURNS)
    }

    pub fn with_default_max_turns(
        session_id: impl Into<String>,
        store: &'a GoalStore,
        default_max_turns: i64,
    ) -> Self {
        let session_id = session_id.into();
        // Python: `int(default_max_turns or DEFAULT_MAX_TURNS)`.
        let default_max_turns = if default_max_turns == 0 {
            DEFAULT_MAX_TURNS
        } else {
            default_max_turns
        };
        let state = store.load_goal(&session_id);
        GoalManager {
            session_id,
            default_max_turns,
            store,
            state,
        }
    }

    // --- introspection ------------------------------------------------

    pub fn state(&self) -> Option<&GoalState> {
        self.state.as_ref()
    }

    pub fn is_active(&self) -> bool {
        self.state.as_ref().map(|s| s.status == "active").unwrap_or(false)
    }

    pub fn has_goal(&self) -> bool {
        self.state
            .as_ref()
            .map(|s| s.status == "active" || s.status == "paused")
            .unwrap_or(false)
    }

    pub fn status_line(&self) -> String {
        let s = match &self.state {
            None => return "No active goal. Set one with /goal <text>.".to_string(),
            Some(s) => s,
        };
        if s.status == "cleared" {
            return "No active goal. Set one with /goal <text>.".to_string();
        }
        let turns = format!("{}/{} turns", s.turns_used, s.max_turns);
        match s.status.as_str() {
            "active" => format!("⊙ Goal (active, {turns}): {}", s.goal),
            "paused" => {
                let extra = match &s.paused_reason {
                    Some(r) if !r.is_empty() => format!(" — {r}"),
                    _ => String::new(),
                };
                format!("⏸ Goal (paused, {turns}{extra}): {}", s.goal)
            }
            "done" => format!("✓ Goal done ({turns}): {}", s.goal),
            other => format!("Goal ({other}, {turns}): {}", s.goal),
        }
    }

    // --- mutation -----------------------------------------------------

    /// Start a new standing goal. Errors if `goal` is empty after trimming.
    pub fn set(&mut self, goal: &str, max_turns: Option<i64>) -> Result<GoalState, GoalError> {
        let goal = goal.trim();
        if goal.is_empty() {
            return Err(GoalError::EmptyGoal);
        }
        // Python: `int(max_turns) if max_turns else self.default_max_turns`.
        let max_turns = match max_turns {
            Some(m) if m != 0 => m,
            _ => self.default_max_turns,
        };
        let state = GoalState {
            goal: goal.to_string(),
            status: "active".to_string(),
            turns_used: 0,
            max_turns,
            created_at: now_unix(),
            last_turn_at: 0.0,
            last_verdict: None,
            last_reason: None,
            paused_reason: None,
        };
        self.state = Some(state.clone());
        let _ = self.store.save_goal(&self.session_id, &state);
        Ok(state)
    }

    pub fn pause(&mut self, reason: &str) -> Option<GoalState> {
        let reason = if reason.is_empty() { "user-paused" } else { reason };
        let state = self.state.as_mut()?;
        state.status = "paused".to_string();
        state.paused_reason = Some(reason.to_string());
        let snapshot = state.clone();
        let _ = self.store.save_goal(&self.session_id, &snapshot);
        Some(snapshot)
    }

    pub fn resume(&mut self, reset_budget: bool) -> Option<GoalState> {
        let state = self.state.as_mut()?;
        state.status = "active".to_string();
        state.paused_reason = None;
        if reset_budget {
            state.turns_used = 0;
        }
        let snapshot = state.clone();
        let _ = self.store.save_goal(&self.session_id, &snapshot);
        Some(snapshot)
    }

    pub fn clear(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.status = "cleared".to_string();
            let snapshot = state.clone();
            let _ = self.store.save_goal(&self.session_id, &snapshot);
        }
        self.state = None;
    }

    pub fn mark_done(&mut self, reason: &str) {
        if let Some(state) = self.state.as_mut() {
            state.status = "done".to_string();
            state.last_verdict = Some("done".to_string());
            state.last_reason = Some(reason.to_string());
            let snapshot = state.clone();
            let _ = self.store.save_goal(&self.session_id, &snapshot);
        }
    }

    // --- the main entry point called after every turn -----------------

    /// Run the judge and update state. Return a [`GoalDecision`].
    ///
    /// `_user_initiated` distinguishes a real user prompt (true) from a
    /// continuation prompt we fed ourselves (false). Both increment
    /// `turns_used` because both consume model budget — so the flag does not
    /// change accounting here (it mirrors the Python parameter, kept for
    /// caller parity).
    pub fn evaluate_after_turn<B: JudgeBackend + ?Sized>(
        &mut self,
        backend: &B,
        last_response: &str,
        _user_initiated: bool,
    ) -> GoalDecision {
        // Inactive / no goal -> short-circuit.
        let is_active = self
            .state
            .as_ref()
            .map(|s| s.status == "active")
            .unwrap_or(false);
        if !is_active {
            let status = self.state.as_ref().map(|s| s.status.clone());
            return GoalDecision {
                status,
                should_continue: false,
                continuation_prompt: None,
                verdict: "inactive".to_string(),
                reason: "no active goal".to_string(),
                message: String::new(),
            };
        }

        // Count the turn that just finished.
        let (goal_text, turns_used, max_turns) = {
            let state = self.state.as_mut().expect("active implies Some");
            state.turns_used += 1;
            state.last_turn_at = now_unix();
            (state.goal.clone(), state.turns_used, state.max_turns)
        };

        let (verdict, reason) = judge_goal_default(backend, &goal_text, last_response);
        {
            let state = self.state.as_mut().expect("active implies Some");
            state.last_verdict = Some(verdict.as_str().to_string());
            state.last_reason = Some(reason.clone());
        }

        if verdict == Verdict::Done {
            {
                let state = self.state.as_mut().expect("active implies Some");
                state.status = "done".to_string();
            }
            self.persist();
            return GoalDecision {
                status: Some("done".to_string()),
                should_continue: false,
                continuation_prompt: None,
                verdict: "done".to_string(),
                reason: reason.clone(),
                message: format!("✓ Goal achieved: {reason}"),
            };
        }

        if turns_used >= max_turns {
            let paused_reason =
                format!("turn budget exhausted ({turns_used}/{max_turns})");
            {
                let state = self.state.as_mut().expect("active implies Some");
                state.status = "paused".to_string();
                state.paused_reason = Some(paused_reason);
            }
            self.persist();
            return GoalDecision {
                status: Some("paused".to_string()),
                should_continue: false,
                continuation_prompt: None,
                verdict: "continue".to_string(),
                reason,
                message: format!(
                    "⏸ Goal paused — {turns_used}/{max_turns} turns used. \
                     Use /goal resume to keep going, or /goal clear to stop."
                ),
            };
        }

        self.persist();
        let continuation = self.next_continuation_prompt();
        GoalDecision {
            status: Some("active".to_string()),
            should_continue: true,
            continuation_prompt: continuation,
            verdict: "continue".to_string(),
            reason: reason.clone(),
            message: format!(
                "↻ Continuing toward goal ({turns_used}/{max_turns}): {reason}"
            ),
        }
    }

    fn persist(&self) {
        if let Some(state) = &self.state {
            let _ = self.store.save_goal(&self.session_id, state);
        }
    }

    pub fn next_continuation_prompt(&self) -> Option<String> {
        let state = self.state.as_ref()?;
        if state.status != "active" {
            return None;
        }
        Some(continuation_prompt(&state.goal))
    }
}

/// Errors raised by goal mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalError {
    EmptyGoal,
}

impl std::fmt::Display for GoalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GoalError::EmptyGoal => write!(f, "goal text is empty"),
        }
    }
}

impl std::error::Error for GoalError {}

// A tiny holder so callers that want a per-home cache (like the Python
// `_DB_CACHE`) have a place to keep stores keyed by hermes_home path.
/// Cache of [`GoalStore`]s keyed by hermes_home path string, mirroring the
/// Python module-level `_DB_CACHE`. Not used internally; provided for callers
/// that open one store per profile.
#[derive(Default)]
pub struct GoalStoreCache {
    map: HashMap<String, GoalStore>,
}

impl GoalStoreCache {
    pub fn new() -> Self {
        GoalStoreCache::default()
    }

    /// Get-or-open a store for `home`/`db_path`. The closure builds the store
    /// only on a cache miss.
    pub fn get_or_open(
        &mut self,
        home: &str,
        db_path: &std::path::Path,
    ) -> rusqlite::Result<&GoalStore> {
        if !self.map.contains_key(home) {
            let store = GoalStore::open(db_path)?;
            self.map.insert(home.to_string(), store);
        }
        Ok(self.map.get(home).expect("just inserted"))
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend returning a canned reply.
    struct CannedBackend(String);
    impl JudgeBackend for CannedBackend {
        fn judge(&self, _m: &[(String, String)], _t: f64) -> Result<String, JudgeError> {
            Ok(self.0.clone())
        }
    }

    struct ApiFailBackend;
    impl JudgeBackend for ApiFailBackend {
        fn judge(&self, _m: &[(String, String)], _t: f64) -> Result<String, JudgeError> {
            Err(JudgeError::Api("TimeoutError".to_string()))
        }
    }

    #[test]
    fn goalstate_roundtrip() {
        let s = GoalState {
            goal: "ship it".to_string(),
            status: "active".to_string(),
            turns_used: 3,
            max_turns: 10,
            created_at: 1.5,
            last_turn_at: 2.5,
            last_verdict: Some("continue".to_string()),
            last_reason: Some("not done".to_string()),
            paused_reason: None,
        };
        let json = s.to_json();
        let back = GoalState::from_json(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn goalstate_from_json_defaults() {
        // Missing fields fall back to defaults; max_turns of 0 -> DEFAULT.
        let s = GoalState::from_json(r#"{"goal":"x","max_turns":0}"#).unwrap();
        assert_eq!(s.goal, "x");
        assert_eq!(s.status, "active");
        assert_eq!(s.turns_used, 0);
        assert_eq!(s.max_turns, DEFAULT_MAX_TURNS);
        assert_eq!(s.created_at, 0.0);
        assert!(s.last_verdict.is_none());
    }

    #[test]
    fn goalstate_from_json_string_numbers() {
        let s = GoalState::from_json(r#"{"goal":"x","turns_used":"5","max_turns":"30"}"#).unwrap();
        assert_eq!(s.turns_used, 5);
        assert_eq!(s.max_turns, 30);
    }

    #[test]
    fn truncate_behaviour() {
        assert_eq!(truncate("", 10), "");
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdef", 3), "abc… [truncated]");
    }

    #[test]
    fn parse_plain_json() {
        let (done, reason) = parse_judge_response(r#"{"done": true, "reason": "all good"}"#);
        assert!(done);
        assert_eq!(reason, "all good");
    }

    #[test]
    fn parse_code_fenced_json() {
        let raw = "```json\n{\"done\": false, \"reason\": \"keep going\"}\n```";
        let (done, reason) = parse_judge_response(raw);
        assert!(!done);
        assert_eq!(reason, "keep going");
    }

    #[test]
    fn parse_embedded_json_object() {
        let raw = "Here is my verdict: {\"done\": true, \"reason\": \"done now\"} thanks";
        let (done, reason) = parse_judge_response(raw);
        assert!(done);
        assert_eq!(reason, "done now");
    }

    #[test]
    fn parse_string_done_values() {
        for v in ["true", "yes", "1", "done", "TRUE", "Yes"] {
            let raw = format!("{{\"done\": \"{v}\", \"reason\": \"r\"}}");
            let (done, _) = parse_judge_response(&raw);
            assert!(done, "expected done for {v}");
        }
        let (done, _) = parse_judge_response(r#"{"done": "nope", "reason": "r"}"#);
        assert!(!done);
    }

    #[test]
    fn parse_empty_and_garbage() {
        let (done, reason) = parse_judge_response("");
        assert!(!done);
        assert_eq!(reason, "judge returned empty response");

        let (done, reason) = parse_judge_response("not json at all");
        assert!(!done);
        assert!(reason.starts_with("judge reply was not JSON"));
    }

    #[test]
    fn parse_missing_reason() {
        let (done, reason) = parse_judge_response(r#"{"done": true}"#);
        assert!(done);
        assert_eq!(reason, "no reason provided");
    }

    #[test]
    fn judge_empty_goal_skipped() {
        let (v, r) = judge_goal_default(&NoBackend, "   ", "response");
        assert_eq!(v, Verdict::Skipped);
        assert_eq!(r, "empty goal");
    }

    #[test]
    fn judge_empty_response_continue() {
        let (v, r) = judge_goal_default(&NoBackend, "goal", "  ");
        assert_eq!(v, Verdict::Continue);
        assert_eq!(r, "empty response (nothing to evaluate)");
    }

    #[test]
    fn judge_unavailable_continue() {
        let (v, r) = judge_goal_default(&NoBackend, "goal", "real response");
        assert_eq!(v, Verdict::Continue);
        assert_eq!(r, "auxiliary client unavailable");
    }

    #[test]
    fn judge_api_failure_continue() {
        let (v, r) = judge_goal_default(&ApiFailBackend, "goal", "real response");
        assert_eq!(v, Verdict::Continue);
        assert_eq!(r, "judge error: TimeoutError");
    }

    #[test]
    fn judge_done_verdict() {
        let backend = CannedBackend(r#"{"done": true, "reason": "complete"}"#.to_string());
        let (v, r) = judge_goal_default(&backend, "goal", "I finished the task");
        assert_eq!(v, Verdict::Done);
        assert_eq!(r, "complete");
    }

    #[test]
    fn build_messages_shape() {
        let msgs = build_judge_messages("g", "r");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].0, "system");
        assert_eq!(msgs[0].1, JUDGE_SYSTEM_PROMPT);
        assert_eq!(msgs[1].0, "user");
        assert!(msgs[1].1.contains("Goal:\ng"));
        assert!(msgs[1].1.contains("response:\nr"));
    }

    #[test]
    fn continuation_prompt_fills_goal() {
        let p = continuation_prompt("write tests");
        assert!(p.contains("Goal: write tests"));
        assert!(p.starts_with("[Continuing toward your standing goal]"));
    }

    fn store() -> GoalStore {
        GoalStore::in_memory().unwrap()
    }

    #[test]
    fn manager_set_and_persist() {
        let store = store();
        let mut mgr = GoalManager::new("sess1", &store);
        assert!(!mgr.has_goal());
        let st = mgr.set("do the thing", None).unwrap();
        assert_eq!(st.status, "active");
        assert_eq!(st.max_turns, DEFAULT_MAX_TURNS);
        assert!(mgr.is_active());

        // Reload from store -> persisted.
        let mgr2 = GoalManager::new("sess1", &store);
        assert!(mgr2.is_active());
        assert_eq!(mgr2.state().unwrap().goal, "do the thing");
    }

    #[test]
    fn manager_set_empty_errs() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        assert_eq!(mgr.set("   ", None), Err(GoalError::EmptyGoal));
    }

    #[test]
    fn manager_pause_resume_clear() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        mgr.set("g", Some(5)).unwrap();
        mgr.pause("");
        assert_eq!(mgr.state().unwrap().status, "paused");
        assert_eq!(mgr.state().unwrap().paused_reason.as_deref(), Some("user-paused"));
        assert!(mgr.has_goal());
        assert!(!mgr.is_active());

        mgr.resume(true);
        assert!(mgr.is_active());
        assert!(mgr.state().unwrap().paused_reason.is_none());

        mgr.clear();
        assert!(mgr.state().is_none());
        // Cleared status persists in store.
        assert_eq!(store.load_goal("s").unwrap().status, "cleared");
    }

    #[test]
    fn manager_status_line() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        assert_eq!(mgr.status_line(), "No active goal. Set one with /goal <text>.");
        mgr.set("ship", Some(4)).unwrap();
        assert_eq!(mgr.status_line(), "⊙ Goal (active, 0/4 turns): ship");
    }

    #[test]
    fn evaluate_inactive() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        let d = mgr.evaluate_after_turn(&NoBackend, "resp", true);
        assert_eq!(d.verdict, "inactive");
        assert!(!d.should_continue);
        assert!(d.continuation_prompt.is_none());
    }

    #[test]
    fn evaluate_done() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        mgr.set("g", Some(10)).unwrap();
        let backend = CannedBackend(r#"{"done": true, "reason": "finished"}"#.to_string());
        let d = mgr.evaluate_after_turn(&backend, "I am done", true);
        assert_eq!(d.verdict, "done");
        assert!(!d.should_continue);
        assert_eq!(d.status.as_deref(), Some("done"));
        assert_eq!(d.message, "✓ Goal achieved: finished");
        assert_eq!(mgr.state().unwrap().turns_used, 1);
    }

    #[test]
    fn evaluate_continue() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        mgr.set("g", Some(10)).unwrap();
        let backend = CannedBackend(r#"{"done": false, "reason": "more work"}"#.to_string());
        let d = mgr.evaluate_after_turn(&backend, "progress", true);
        assert_eq!(d.verdict, "continue");
        assert!(d.should_continue);
        assert!(d.continuation_prompt.is_some());
        assert_eq!(d.status.as_deref(), Some("active"));
        assert!(d.message.starts_with("↻ Continuing toward goal (1/10)"));
    }

    #[test]
    fn evaluate_budget_exhausted() {
        let store = store();
        let mut mgr = GoalManager::new("s", &store);
        mgr.set("g", Some(1)).unwrap();
        let backend = CannedBackend(r#"{"done": false, "reason": "keep going"}"#.to_string());
        let d = mgr.evaluate_after_turn(&backend, "progress", true);
        assert_eq!(d.verdict, "continue");
        assert!(!d.should_continue);
        assert_eq!(d.status.as_deref(), Some("paused"));
        assert!(d.message.starts_with("⏸ Goal paused — 1/1 turns used."));
        assert!(mgr.state().unwrap().paused_reason.as_deref().unwrap().contains("turn budget exhausted (1/1)"));
    }

    #[test]
    fn store_load_empty_session() {
        let store = store();
        assert!(store.load_goal("").is_none());
        assert!(store.load_goal("never-set").is_none());
    }
}
