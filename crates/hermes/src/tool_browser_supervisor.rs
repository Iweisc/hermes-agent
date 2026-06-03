//! Persistent CDP supervisor for browser dialog + frame detection.
//!
//! One [`CdpSupervisor`] runs per Hermes `task_id` that has a reachable CDP
//! endpoint. It holds a single persistent WebSocket to the backend, subscribes
//! to `Page` / `Runtime` / `Target` events on every attached session
//! (top-level page and every OOPIF / worker target that auto-attaches), and
//! surfaces observable state — pending dialogs and frame tree — through a
//! thread-safe snapshot object that tool handlers consume synchronously.
//!
//! This is a native Rust port of `tools/browser_supervisor.py`. The Python
//! original uses a dedicated asyncio loop on a daemon thread; here we use a
//! dedicated thread driving a synchronous [`tungstenite`] WebSocket with a
//! reconnecting read loop, matching the pattern already used in
//! `hermes-core/src/browser.rs`.
//!
//! The supervisor is NOT in the agent's tool schema. Its output reaches the
//! agent via two channels:
//!
//! 1. `browser_snapshot` merges supervisor state into its return payload.
//! 2. `browser_dialog` tool responds to a pending dialog by calling
//!    [`CdpSupervisor::respond_to_dialog`] on the active supervisor.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};
use url::Url;

// ── Config defaults ─────────────────────────────────────────────────────────

pub const DIALOG_POLICY_MUST_RESPOND: &str = "must_respond";
pub const DIALOG_POLICY_AUTO_DISMISS: &str = "auto_dismiss";
pub const DIALOG_POLICY_AUTO_ACCEPT: &str = "auto_accept";

pub const DEFAULT_DIALOG_POLICY: &str = DIALOG_POLICY_MUST_RESPOND;
pub const DEFAULT_DIALOG_TIMEOUT_S: f64 = 300.0;

/// Snapshot caps for `frame_tree` — keep payloads bounded on ad-heavy pages.
pub const FRAME_TREE_MAX_ENTRIES: usize = 30;
pub const FRAME_TREE_MAX_OOPIF_DEPTH: usize = 2;

/// Ring buffer of recent console-level events.
pub const CONSOLE_HISTORY_MAX: usize = 50;

/// Keep the last N closed dialogs in `recent_dialogs` so agents on backends
/// that auto-dismiss server-side (e.g. Browserbase) can still observe that a
/// dialog fired, even if they couldn't respond to it in time.
pub const RECENT_DIALOGS_MAX: usize = 20;

/// Magic host the injected dialog bridge XHRs to. Intercepted via the CDP
/// Fetch domain before any network resolution happens, so the hostname never
/// has to exist.
pub const DIALOG_BRIDGE_HOST: &str = "hermes-dialog-bridge.invalid";

/// Fetch URL pattern gated on the bridge host.
pub fn dialog_bridge_url_pattern() -> String {
    format!("http://{DIALOG_BRIDGE_HOST}/*")
}

/// Script injected into every frame via Page.addScriptToEvaluateOnNewDocument.
/// Overrides alert/confirm/prompt to round-trip through a sync XHR that we
/// intercept via Fetch.requestPaused.
pub const DIALOG_BRIDGE_SCRIPT: &str = r#"
(() => {
  if (window.__hermesDialogBridgeInstalled) return;
  window.__hermesDialogBridgeInstalled = true;
  const ENDPOINT = "http://hermes-dialog-bridge.invalid/";
  function ask(kind, message, defaultPrompt) {
    try {
      const xhr = new XMLHttpRequest();
      const params = new URLSearchParams({
        kind: String(kind || ""),
        message: String(message == null ? "" : message),
        default_prompt: String(defaultPrompt == null ? "" : defaultPrompt),
      });
      xhr.open("GET", ENDPOINT + "?" + params.toString(), false);  // sync
      xhr.send(null);
      if (xhr.status !== 200) return null;
      const body = xhr.responseText || "";
      let parsed;
      try { parsed = JSON.parse(body); } catch (e) { return null; }
      if (kind === "alert") return undefined;
      if (kind === "confirm") return Boolean(parsed && parsed.accept);
      if (kind === "prompt") {
        if (!parsed || !parsed.accept) return null;
        return parsed.prompt_text == null ? "" : String(parsed.prompt_text);
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  const realAlert   = window.alert;
  const realConfirm = window.confirm;
  const realPrompt  = window.prompt;
  window.alert   = function(message) { ask("alert",   message, ""); };
  window.confirm = function(message) {
    const r = ask("confirm", message, "");
    return r === null ? false : Boolean(r);
  };
  window.prompt  = function(message, def) {
    const r = ask("prompt", message, def == null ? "" : def);
    return r === null ? null : String(r);
  };
})();
"#;

/// The valid dialog policies.
pub fn valid_policies() -> [&'static str; 3] {
    [
        DIALOG_POLICY_MUST_RESPOND,
        DIALOG_POLICY_AUTO_DISMISS,
        DIALOG_POLICY_AUTO_ACCEPT,
    ]
}

fn is_valid_policy(policy: &str) -> bool {
    valid_policies().contains(&policy)
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── Data model ───────────────────────────────────────────────────────────────

/// A JS dialog currently open on some frame's session.
#[derive(Debug, Clone)]
pub struct PendingDialog {
    pub id: String,
    /// "alert" | "confirm" | "prompt" | "beforeunload"
    pub dialog_type: String,
    pub message: String,
    pub default_prompt: String,
    pub opened_at: f64,
    /// Which attached CDP session the dialog fired in.
    pub cdp_session_id: String,
    pub frame_id: Option<String>,
    /// When set, the dialog was captured via the bridge XHR path (Fetch
    /// domain). Response must be delivered via Fetch.fulfillRequest, NOT
    /// Page.handleJavaScriptDialog — the native dialog never fired.
    pub bridge_request_id: Option<String>,
}

impl PendingDialog {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.dialog_type,
            "message": self.message,
            "default_prompt": self.default_prompt,
            "opened_at": self.opened_at,
            "frame_id": self.frame_id,
        })
    }
}

/// A historical record of a dialog that was opened and then handled.
#[derive(Debug, Clone)]
pub struct DialogRecord {
    pub id: String,
    pub dialog_type: String,
    pub message: String,
    pub opened_at: f64,
    pub closed_at: f64,
    /// "agent" | "auto_policy" | "remote" | "watchdog"
    pub closed_by: String,
    pub frame_id: Option<String>,
}

impl DialogRecord {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.dialog_type,
            "message": self.message,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "closed_by": self.closed_by,
            "frame_id": self.frame_id,
        })
    }
}

/// One frame in the page's frame tree.
///
/// `is_oopif` means the frame has its own CDP target (separate process,
/// reachable via `cdp_session_id`). Same-origin / srcdoc iframes share the
/// parent process and have `is_oopif=false` + `cdp_session_id=None`.
#[derive(Debug, Clone)]
pub struct FrameInfo {
    pub frame_id: String,
    pub url: String,
    pub origin: String,
    pub parent_frame_id: Option<String>,
    pub is_oopif: bool,
    pub cdp_session_id: Option<String>,
    pub name: String,
}

impl FrameInfo {
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("frame_id".to_string(), Value::String(self.frame_id.clone()));
        object.insert("url".to_string(), Value::String(self.url.clone()));
        object.insert("origin".to_string(), Value::String(self.origin.clone()));
        object.insert("is_oopif".to_string(), Value::Bool(self.is_oopif));
        if let Some(sid) = self.cdp_session_id.as_ref() {
            if !sid.is_empty() {
                object.insert("session_id".to_string(), Value::String(sid.clone()));
            }
        }
        if let Some(parent) = self.parent_frame_id.as_ref() {
            if !parent.is_empty() {
                object.insert(
                    "parent_frame_id".to_string(),
                    Value::String(parent.clone()),
                );
            }
        }
        if !self.name.is_empty() {
            object.insert("name".to_string(), Value::String(self.name.clone()));
        }
        Value::Object(object)
    }
}

/// Ring buffer entry for console + exception traffic.
#[derive(Debug, Clone)]
pub struct ConsoleEvent {
    pub ts: f64,
    /// "log" | "error" | "warning" | "exception"
    pub level: String,
    pub text: String,
    pub url: Option<String>,
}

/// Read-only snapshot of supervisor state.
#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub pending_dialogs: Vec<PendingDialog>,
    pub recent_dialogs: Vec<DialogRecord>,
    pub frame_tree: Value,
    pub console_errors: Vec<ConsoleEvent>,
    /// False if supervisor is detached/stopped.
    pub active: bool,
    pub cdp_url: String,
    pub task_id: String,
}

impl SupervisorSnapshot {
    /// Serialize for inclusion in `browser_snapshot` output.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert(
            "pending_dialogs".to_string(),
            Value::Array(self.pending_dialogs.iter().map(|d| d.to_json()).collect()),
        );
        out.insert("frame_tree".to_string(), self.frame_tree.clone());
        if !self.recent_dialogs.is_empty() {
            out.insert(
                "recent_dialogs".to_string(),
                Value::Array(self.recent_dialogs.iter().map(|d| d.to_json()).collect()),
            );
        }
        Value::Object(out)
    }
}

// ── Supervisor shared state ───────────────────────────────────────────────────

/// Mutable supervisor state guarded by a single mutex (cross-thread reads).
#[derive(Default)]
struct SupervisorState {
    /// dialog id -> pending dialog. Ordered insertion preserved via `order`.
    pending_dialogs: HashMap<String, PendingDialog>,
    /// Insertion order of pending dialog ids (for deterministic candidate
    /// selection, mirroring Python's dict ordering).
    pending_order: Vec<String>,
    recent_dialogs: Vec<DialogRecord>,
    frames: HashMap<String, FrameInfo>,
    /// Insertion order of frame ids (mirrors Python dict iteration order).
    frame_order: Vec<String>,
    console_events: Vec<ConsoleEvent>,
    active: bool,
    /// Watchdog expiry deadlines: dialog id -> instant at which it auto-dismisses.
    dialog_deadlines: HashMap<String, Instant>,
    /// Monotonic id generator for dialogs (human-readable in snapshots).
    dialog_seq: u64,
}

impl SupervisorState {
    fn insert_pending(&mut self, dialog: PendingDialog) {
        if !self.pending_dialogs.contains_key(&dialog.id) {
            self.pending_order.push(dialog.id.clone());
        }
        self.pending_dialogs.insert(dialog.id.clone(), dialog);
    }

    fn remove_pending(&mut self, id: &str) -> Option<PendingDialog> {
        self.pending_order.retain(|x| x != id);
        self.dialog_deadlines.remove(id);
        self.pending_dialogs.remove(id)
    }

    fn ordered_pending(&self) -> Vec<PendingDialog> {
        self.pending_order
            .iter()
            .filter_map(|id| self.pending_dialogs.get(id).cloned())
            .collect()
    }

    fn insert_frame(&mut self, frame: FrameInfo) {
        if !self.frames.contains_key(&frame.frame_id) {
            self.frame_order.push(frame.frame_id.clone());
        }
        self.frames.insert(frame.frame_id.clone(), frame);
    }

    fn remove_frame(&mut self, id: &str) -> Option<FrameInfo> {
        self.frame_order.retain(|x| x != id);
        self.frames.remove(id)
    }

    fn ordered_frames(&self) -> Vec<FrameInfo> {
        self.frame_order
            .iter()
            .filter_map(|id| self.frames.get(id).cloned())
            .collect()
    }

    /// Move a pending dialog to the recent_dialogs ring buffer.
    fn archive_dialog(&mut self, dialog: &PendingDialog, closed_by: &str) {
        let record = DialogRecord {
            id: dialog.id.clone(),
            dialog_type: dialog.dialog_type.clone(),
            message: dialog.message.clone(),
            opened_at: dialog.opened_at,
            closed_at: now_secs(),
            closed_by: closed_by.to_string(),
            frame_id: dialog.frame_id.clone(),
        };
        self.recent_dialogs.push(record);
        if self.recent_dialogs.len() > RECENT_DIALOGS_MAX * 2 {
            let start = self.recent_dialogs.len() - RECENT_DIALOGS_MAX;
            self.recent_dialogs.drain(..start);
        }
    }

    fn push_console(&mut self, event: ConsoleEvent) {
        self.console_events.push(event);
        if self.console_events.len() > CONSOLE_HISTORY_MAX * 2 {
            let start = self.console_events.len() - CONSOLE_HISTORY_MAX;
            self.console_events.drain(..start);
        }
    }

    /// Build the capped frame_tree payload.
    fn build_frame_tree(&self) -> Value {
        let frames = self.ordered_frames();
        if frames.is_empty() {
            return json!({"top": null, "children": [], "truncated": false});
        }

        // Identify a top frame — one with no parent, preferring oopif=false.
        let tops: Vec<&FrameInfo> = frames
            .iter()
            .filter(|f| f.parent_frame_id.as_deref().unwrap_or("").is_empty())
            .collect();
        let top = tops
            .iter()
            .find(|f| !f.is_oopif)
            .copied()
            .or_else(|| tops.first().copied());

        let top = match top {
            Some(t) => t.clone(),
            None => return json!({"top": null, "children": [], "truncated": false}),
        };

        // BFS from top, capped by FRAME_TREE_MAX_ENTRIES and
        // FRAME_TREE_MAX_OOPIF_DEPTH for OOPIF branches.
        let mut children: Vec<Value> = Vec::new();
        let mut truncated = false;
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(top.frame_id.clone());

        let mut queue: std::collections::VecDeque<(FrameInfo, usize)> = frames
            .iter()
            .filter(|f| f.parent_frame_id.as_deref() == Some(top.frame_id.as_str()))
            .map(|f| (f.clone(), 1usize))
            .collect();

        while let Some((frame, depth)) = queue.pop_front() {
            if children.len() >= FRAME_TREE_MAX_ENTRIES {
                break;
            }
            if visited.contains(&frame.frame_id) {
                continue;
            }
            visited.insert(frame.frame_id.clone());
            if frame.is_oopif && depth > FRAME_TREE_MAX_OOPIF_DEPTH {
                truncated = true;
                continue;
            }
            children.push(frame.to_json());
            for f in frames.iter() {
                if f.parent_frame_id.as_deref() == Some(frame.frame_id.as_str())
                    && !visited.contains(&f.frame_id)
                {
                    queue.push_back((f.clone(), depth + 1));
                }
            }
        }
        if !queue.is_empty() {
            truncated = true;
        }

        json!({
            "top": top.to_json(),
            "children": children,
            "truncated": truncated,
        })
    }
}

// ── Commands sent to the supervisor thread ────────────────────────────────────

enum SupervisorCommand {
    /// Respond to a pending dialog. The CDP work happens on the supervisor
    /// thread; the result is sent back on `result_tx`.
    Respond {
        dialog: PendingDialog,
        accept: bool,
        prompt_text: String,
        result_tx: Sender<Result<(), String>>,
    },
    Stop,
}

// ── Supervisor core ───────────────────────────────────────────────────────────

/// One supervisor per (task_id, cdp_url) pair.
pub struct CdpSupervisor {
    pub task_id: String,
    pub cdp_url: String,
    pub dialog_policy: String,
    pub dialog_timeout_s: f64,

    state: Arc<Mutex<SupervisorState>>,
    /// Signaled when the supervisor first becomes ready (or fails to start).
    ready: Arc<(Mutex<ReadyState>, Condvar)>,
    command_tx: Mutex<Option<Sender<SupervisorCommand>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    stop_requested: Arc<Mutex<bool>>,
}

#[derive(Default)]
struct ReadyState {
    ready: bool,
    error: Option<String>,
}

impl CdpSupervisor {
    /// Construct a supervisor. Returns `Err` if `dialog_policy` is invalid.
    pub fn new(
        task_id: impl Into<String>,
        cdp_url: impl Into<String>,
        dialog_policy: impl Into<String>,
        dialog_timeout_s: f64,
    ) -> Result<Self, String> {
        let dialog_policy = dialog_policy.into();
        if !is_valid_policy(&dialog_policy) {
            let mut valid = valid_policies().to_vec();
            valid.sort_unstable();
            return Err(format!(
                "Invalid dialog_policy {dialog_policy:?}; must be one of {valid:?}"
            ));
        }
        Ok(Self {
            task_id: task_id.into(),
            cdp_url: cdp_url.into(),
            dialog_policy,
            dialog_timeout_s,
            state: Arc::new(Mutex::new(SupervisorState::default())),
            ready: Arc::new((Mutex::new(ReadyState::default()), Condvar::new())),
            command_tx: Mutex::new(None),
            thread: Mutex::new(None),
            stop_requested: Arc::new(Mutex::new(false)),
        })
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, SupervisorState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn is_thread_alive(&self) -> bool {
        let guard = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }

    // ── Public sync API ──────────────────────────────────────────────────────

    /// Launch the background loop and wait until attachment is complete.
    ///
    /// Returns whatever error attach failed with (connect error, bad
    /// WebSocket URL, CDP domain enable failure, etc.). On success, the
    /// supervisor is fully wired up.
    pub fn start(&self, timeout: Duration) -> Result<(), String> {
        if self.is_thread_alive() {
            return Ok(());
        }
        // Reset ready/stop state.
        {
            let (lock, _cv) = &*self.ready;
            let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            guard.ready = false;
            guard.error = None;
        }
        *self.stop_requested.lock().unwrap_or_else(|e| e.into_inner()) = false;

        let (command_tx, command_rx) = mpsc::channel();
        *self.command_tx.lock().unwrap_or_else(|e| e.into_inner()) = Some(command_tx);

        let worker = SupervisorWorker {
            task_id: self.task_id.clone(),
            cdp_url: self.cdp_url.clone(),
            dialog_policy: self.dialog_policy.clone(),
            dialog_timeout: Duration::from_secs_f64(self.dialog_timeout_s.max(0.0)),
            dialog_timeout_s: self.dialog_timeout_s,
            state: Arc::clone(&self.state),
            ready: Arc::clone(&self.ready),
            stop_requested: Arc::clone(&self.stop_requested),
            command_rx,
        };

        let task_id = self.task_id.clone();
        let handle = thread::Builder::new()
            .name(format!("cdp-supervisor-{task_id}"))
            .spawn(move || worker.run())
            .map_err(|e| format!("failed to spawn supervisor thread: {e}"))?;
        *self.thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);

        // Wait for ready signal.
        let (lock, cv) = &*self.ready;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + timeout;
        while !guard.ready {
            let now = Instant::now();
            if now >= deadline {
                drop(guard);
                self.stop(Duration::from_secs(5));
                return Err(format!(
                    "CDP supervisor did not attach within {:.0}s (cdp_url={}...)",
                    timeout.as_secs_f64(),
                    &self.cdp_url[..self.cdp_url.len().min(80)]
                ));
            }
            let wait = deadline - now;
            let (g, _timeout_result) = cv
                .wait_timeout(guard, wait)
                .unwrap_or_else(|e| e.into_inner());
            guard = g;
        }
        if let Some(err) = guard.error.clone() {
            drop(guard);
            self.stop(Duration::from_secs(5));
            return Err(err);
        }
        Ok(())
    }

    /// Cancel the supervisor task and join the thread.
    pub fn stop(&self, timeout: Duration) {
        *self.stop_requested.lock().unwrap_or_else(|e| e.into_inner()) = true;
        if let Some(tx) = self
            .command_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            let _ = tx.send(SupervisorCommand::Stop);
        }
        let handle = self
            .thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(handle) = handle {
            // Best-effort join with timeout: poll is_finished, then join.
            let deadline = Instant::now() + timeout;
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }
            let _ = handle.join();
        }
        self.lock_state().active = false;
    }

    /// Return an immutable snapshot of current state.
    pub fn snapshot(&self) -> SupervisorSnapshot {
        let guard = self.lock_state();
        let dialogs = guard.ordered_pending();
        let recent_start = guard.recent_dialogs.len().saturating_sub(RECENT_DIALOGS_MAX);
        let recent = guard.recent_dialogs[recent_start..].to_vec();
        let frames_tree = guard.build_frame_tree();
        let console_start = guard.console_events.len().saturating_sub(CONSOLE_HISTORY_MAX);
        let console = guard.console_events[console_start..].to_vec();
        let active = guard.active;
        SupervisorSnapshot {
            pending_dialogs: dialogs,
            recent_dialogs: recent,
            frame_tree: frames_tree,
            console_errors: console,
            active,
            cdp_url: self.cdp_url.clone(),
            task_id: self.task_id.clone(),
        }
    }

    /// Accept/dismiss a pending dialog. Sync bridge onto the supervisor thread.
    ///
    /// Returns `{"ok": true, "dialog": {...}}` on success,
    /// `{"ok": false, "error": "..."}` on a recoverable error (no dialog,
    /// ambiguous dialog_id, supervisor inactive).
    pub fn respond_to_dialog(
        &self,
        action: &str,
        prompt_text: Option<&str>,
        dialog_id: Option<&str>,
        timeout: Duration,
    ) -> Value {
        if action != "accept" && action != "dismiss" {
            return json!({
                "ok": false,
                "error": format!("action must be 'accept' or 'dismiss', got {action:?}"),
            });
        }

        let dialog = {
            let guard = self.lock_state();
            if !guard.active {
                return json!({"ok": false, "error": "supervisor is not active"});
            }
            let pending = guard.ordered_pending();
            if pending.is_empty() {
                return json!({"ok": false, "error": "no dialog is currently open"});
            }
            if let Some(did) = dialog_id {
                match guard.pending_dialogs.get(did) {
                    Some(d) => d.clone(),
                    None => {
                        let mut known: Vec<String> =
                            guard.pending_dialogs.keys().cloned().collect();
                        known.sort();
                        return json!({
                            "ok": false,
                            "error": format!(
                                "dialog_id {:?} not found (known: {:?})",
                                did, known
                            ),
                        });
                    }
                }
            } else if pending.len() > 1 {
                let candidates: Vec<String> = pending.iter().map(|d| d.id.clone()).collect();
                return json!({
                    "ok": false,
                    "error": format!(
                        "{} pending dialogs; specify dialog_id. Candidates: {:?}",
                        pending.len(),
                        candidates
                    ),
                });
            } else {
                pending[0].clone()
            }
        };

        let snapshot_copy = dialog.to_json();

        let (result_tx, result_rx) = mpsc::channel();
        let cmd = SupervisorCommand::Respond {
            dialog,
            accept: action == "accept",
            prompt_text: prompt_text.unwrap_or("").to_string(),
            result_tx,
        };

        {
            let guard = self.command_tx.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(tx) => {
                    if tx.send(cmd).is_err() {
                        return json!({"ok": false, "error": "supervisor loop is not running"});
                    }
                }
                None => {
                    return json!({"ok": false, "error": "supervisor loop is not running"});
                }
            }
        }

        match result_rx.recv_timeout(timeout) {
            Ok(Ok(())) => json!({"ok": true, "dialog": snapshot_copy}),
            Ok(Err(e)) => json!({"ok": false, "error": e}),
            Err(RecvTimeoutError::Timeout) => {
                json!({"ok": false, "error": "TimeoutError: dialog response timed out"})
            }
            Err(RecvTimeoutError::Disconnected) => {
                json!({"ok": false, "error": "supervisor loop is not running"})
            }
        }
    }
}

// ── Supervisor worker (runs on its own thread) ────────────────────────────────

struct SupervisorWorker {
    task_id: String,
    cdp_url: String,
    dialog_policy: String,
    dialog_timeout: Duration,
    dialog_timeout_s: f64,
    state: Arc<Mutex<SupervisorState>>,
    ready: Arc<(Mutex<ReadyState>, Condvar)>,
    stop_requested: Arc<Mutex<bool>>,
    command_rx: Receiver<SupervisorCommand>,
}

type CdpSocket = WebSocket<MaybeTlsStream<std::net::TcpStream>>;

impl SupervisorWorker {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, SupervisorState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn stop_requested(&self) -> bool {
        *self.stop_requested.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn signal_ready(&self, error: Option<String>) {
        let (lock, cv) = &*self.ready;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.ready {
            guard.ready = true;
            guard.error = error;
            cv.notify_all();
        }
    }

    fn ready_already_set(&self) -> bool {
        let (lock, _cv) = &*self.ready;
        lock.lock().unwrap_or_else(|e| e.into_inner()).ready
    }

    /// Top-level supervisor loop with reconnection.
    fn run(self) {
        let mut attempt: u64 = 0;
        let mut backoff = 0.5_f64;

        while !self.stop_requested() {
            let connect_result = self.connect_ws();
            let mut socket = match connect_result {
                Ok(s) => s,
                Err(e) => {
                    attempt += 1;
                    if !self.ready_already_set() {
                        // Never connected once — fatal for start().
                        self.signal_ready(Some(e));
                        self.lock_state().active = false;
                        return;
                    }
                    log::warn!(
                        "CDP supervisor {}: connect failed (attempt {}): {}",
                        self.task_id,
                        attempt,
                        e
                    );
                    self.sleep_interruptible(Duration::from_secs_f64(backoff.min(10.0)));
                    backoff = (backoff * 2.0).min(10.0);
                    continue;
                }
            };

            // Reset per-connection session state.
            // (We deliberately keep pending_dialogs and frames; they reconcile
            // as events arrive.)
            let mut conn = ConnectionState::default();

            let attach_result = self.attach_initial_page(&mut socket, &mut conn);
            match attach_result {
                Ok(()) => {
                    self.lock_state().active = true;
                    backoff = 0.5;
                    self.signal_ready(None);
                }
                Err(e) => {
                    if !self.ready_already_set() {
                        self.signal_ready(Some(e));
                        self.lock_state().active = false;
                        let _ = socket.close(None);
                        return;
                    }
                    log::warn!("CDP supervisor {}: attach failed: {}", self.task_id, e);
                    self.lock_state().active = false;
                    let _ = socket.close(None);
                    if self.stop_requested() {
                        return;
                    }
                    self.sleep_interruptible(Duration::from_secs_f64(backoff));
                    backoff = (backoff * 2.0).min(10.0);
                    continue;
                }
            }

            // Run the read loop until the socket drops or stop is requested.
            self.read_loop(&mut socket, &mut conn);

            self.lock_state().active = false;
            let _ = socket.close(None);

            if self.stop_requested() {
                return;
            }

            log::debug!(
                "CDP supervisor {}: reconnecting in {:.1}s...",
                self.task_id,
                backoff
            );
            self.sleep_interruptible(Duration::from_secs_f64(backoff));
            backoff = (backoff * 2.0).min(10.0);
        }

        self.lock_state().active = false;
    }

    fn sleep_interruptible(&self, dur: Duration) {
        let deadline = Instant::now() + dur;
        while Instant::now() < deadline {
            if self.stop_requested() {
                return;
            }
            thread::sleep(Duration::from_millis(50).min(dur));
        }
    }

    fn connect_ws(&self) -> Result<CdpSocket, String> {
        Url::parse(&self.cdp_url).map_err(|e| format!("Invalid CDP endpoint: {e}"))?;
        let (mut socket, _resp) =
            connect(&self.cdp_url).map_err(|e| format!("Connecting to CDP endpoint failed: {e}"))?;
        set_socket_read_timeout(&mut socket, Duration::from_millis(200));
        Ok(socket)
    }

    /// Find a page target, attach flattened session, enable domains, install
    /// dialog bridge.
    fn attach_initial_page(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
    ) -> Result<(), String> {
        let resp = self.cdp(socket, conn, "Target.getTargets", None, None, 10.0)?;
        let targets = resp
            .get("result")
            .and_then(|r| r.get("targetInfos"))
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let page_target = targets
            .iter()
            .find(|t| t.get("type").and_then(|v| v.as_str()) == Some("page"));

        let target_id = if let Some(target) = page_target {
            target
                .get("targetId")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "page target missing targetId".to_string())?
                .to_string()
        } else {
            let created = self.cdp(
                socket,
                conn,
                "Target.createTarget",
                Some(json!({"url": "about:blank"})),
                None,
                10.0,
            )?;
            created
                .get("result")
                .and_then(|r| r.get("targetId"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "createTarget returned no targetId".to_string())?
                .to_string()
        };

        let attach = self.cdp(
            socket,
            conn,
            "Target.attachToTarget",
            Some(json!({"targetId": target_id, "flatten": true})),
            None,
            10.0,
        )?;
        let page_session = attach
            .get("result")
            .and_then(|r| r.get("sessionId"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| "attachToTarget returned no sessionId".to_string())?
            .to_string();
        conn.page_session_id = Some(page_session.clone());

        self.cdp(socket, conn, "Page.enable", None, Some(&page_session), 10.0)?;
        self.cdp(
            socket,
            conn,
            "Runtime.enable",
            None,
            Some(&page_session),
            10.0,
        )?;
        self.cdp(
            socket,
            conn,
            "Target.setAutoAttach",
            Some(json!({
                "autoAttach": true,
                "waitForDebuggerOnStart": false,
                "flatten": true,
            })),
            Some(&page_session),
            10.0,
        )?;
        self.install_dialog_bridge(socket, conn, &page_session);
        Ok(())
    }

    /// Install the dialog-bridge init script + Fetch interceptor on a session.
    /// Best-effort; failures are logged and swallowed.
    fn install_dialog_bridge(&self, socket: &mut CdpSocket, conn: &mut ConnectionState, sid: &str) {
        if let Err(e) = self.cdp(
            socket,
            conn,
            "Page.addScriptToEvaluateOnNewDocument",
            Some(json!({"source": DIALOG_BRIDGE_SCRIPT, "runImmediately": true})),
            Some(sid),
            5.0,
        ) {
            log::debug!(
                "dialog bridge: addScriptToEvaluateOnNewDocument failed on sid={}: {}",
                &sid[..sid.len().min(16)],
                e
            );
        }
        if let Err(e) = self.cdp(
            socket,
            conn,
            "Fetch.enable",
            Some(json!({
                "patterns": [{
                    "urlPattern": dialog_bridge_url_pattern(),
                    "requestStage": "Request",
                }],
                "handleAuthRequests": false,
            })),
            Some(sid),
            5.0,
        ) {
            log::debug!(
                "dialog bridge: Fetch.enable failed on sid={}: {}",
                &sid[..sid.len().min(16)],
                e
            );
        }
        // Best-effort inject into already-loaded document.
        let _ = self.cdp(
            socket,
            conn,
            "Runtime.evaluate",
            Some(json!({"expression": DIALOG_BRIDGE_SCRIPT, "returnByValue": true})),
            Some(sid),
            3.0,
        );
    }

    /// Send a CDP command and synchronously await its response, dispatching any
    /// events that arrive in the meantime.
    fn cdp(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        method: &str,
        params: Option<Value>,
        session_id: Option<&str>,
        timeout_s: f64,
    ) -> Result<Value, String> {
        let call_id = conn.next_call_id;
        conn.next_call_id += 1;

        let mut payload = Map::new();
        payload.insert("id".to_string(), json!(call_id));
        payload.insert("method".to_string(), Value::String(method.to_string()));
        if let Some(p) = params {
            payload.insert("params".to_string(), p);
        }
        if let Some(sid) = session_id {
            if !sid.is_empty() {
                payload.insert("sessionId".to_string(), Value::String(sid.to_string()));
            }
        }
        let text = Value::Object(payload).to_string();
        socket
            .send(Message::Text(text.into()))
            .map_err(|e| format!("Sending CDP method {method} failed: {e}"))?;

        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
        loop {
            if Instant::now() >= deadline {
                return Err(format!("CDP method {method} timed out"));
            }
            match self.read_message(socket) {
                Ok(Some(msg)) => {
                    if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                        if id == call_id {
                            if let Some(err) = msg.get("error") {
                                return Err(format!("CDP error on id={call_id}: {err}"));
                            }
                            return Ok(msg);
                        }
                        // A response to some other in-flight call: ignore — in
                        // this synchronous model we only have one in flight.
                    } else if let Some(method_name) =
                        msg.get("method").and_then(|v| v.as_str()).map(String::from)
                    {
                        let params = msg.get("params").cloned().unwrap_or(json!({}));
                        let event_sid = msg
                            .get("sessionId")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        // Dispatch the event but we can't issue nested blocking
                        // CDP calls here without re-entrancy; queue child setup.
                        self.on_event(socket, conn, &method_name, &params, event_sid.as_deref());
                    }
                }
                Ok(None) => {
                    // Read timed out; keep waiting until deadline.
                    if self.stop_requested() {
                        return Err("supervisor stop requested".to_string());
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Read and dispatch incoming CDP frames; also services the command channel
    /// for dialog responses. Returns when the socket drops or stop is requested.
    fn read_loop(&self, socket: &mut CdpSocket, conn: &mut ConnectionState) {
        loop {
            if self.stop_requested() {
                return;
            }
            // Service any pending response commands first.
            self.drain_commands(socket, conn);

            match self.read_message(socket) {
                Ok(Some(msg)) => {
                    if let Some(method) = msg.get("method").and_then(|v| v.as_str()).map(String::from)
                    {
                        let params = msg.get("params").cloned().unwrap_or(json!({}));
                        let event_sid = msg
                            .get("sessionId")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        self.on_event(socket, conn, &method, &params, event_sid.as_deref());
                    }
                    // Stray command responses (id present) are ignored here.
                }
                Ok(None) => {
                    // Read timeout: check watchdog deadlines, then loop.
                    self.check_watchdogs(socket, conn);
                }
                Err(_) => {
                    // Socket closed / errored — exit so run() can reconnect.
                    return;
                }
            }
        }
    }

    fn drain_commands(&self, socket: &mut CdpSocket, conn: &mut ConnectionState) {
        loop {
            match self.command_rx.try_recv() {
                Ok(SupervisorCommand::Stop) => {
                    *self
                        .stop_requested
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = true;
                    return;
                }
                Ok(SupervisorCommand::Respond {
                    dialog,
                    accept,
                    prompt_text,
                    result_tx,
                }) => {
                    let res = self.handle_dialog_cdp(socket, conn, &dialog, accept, &prompt_text);
                    let _ = result_tx.send(res);
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
    }

    /// Auto-dismiss dialogs whose watchdog deadline elapsed (must_respond mode).
    fn check_watchdogs(&self, socket: &mut CdpSocket, conn: &mut ConnectionState) {
        let now = Instant::now();
        let expired: Vec<String> = {
            let guard = self.lock_state();
            guard
                .dialog_deadlines
                .iter()
                .filter(|(_, deadline)| now >= **deadline)
                .map(|(id, _)| id.clone())
                .collect()
        };
        for dialog_id in expired {
            self.dialog_timeout_expired(socket, conn, &dialog_id);
        }
    }

    fn read_message(&self, socket: &mut CdpSocket) -> Result<Option<Value>, String> {
        match socket.read() {
            Ok(Message::Text(text)) => serde_json::from_str::<Value>(text.as_ref())
                .map(Some)
                .map_err(|e| format!("Invalid JSON from CDP endpoint: {e}")),
            Ok(Message::Binary(bytes)) => serde_json::from_slice::<Value>(bytes.as_ref())
                .map(Some)
                .map_err(|e| format!("Invalid binary CDP frame: {e}")),
            Ok(Message::Ping(payload)) => {
                socket
                    .send(Message::Pong(payload))
                    .map_err(|e| format!("Responding to CDP ping failed: {e}"))?;
                Ok(None)
            }
            Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => Ok(None),
            Ok(Message::Close(_)) => Err("CDP connection closed".to_string()),
            Err(tungstenite::Error::Io(error))
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut =>
            {
                Ok(None)
            }
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                Err("CDP connection closed".to_string())
            }
            Err(e) => Err(format!("Reading CDP frame failed: {e}")),
        }
    }

    // ── Event dispatch ──────────────────────────────────────────────────────

    fn on_event(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        method: &str,
        params: &Value,
        session_id: Option<&str>,
    ) {
        match method {
            "Page.javascriptDialogOpening" => {
                self.on_dialog_opening(socket, conn, params, session_id)
            }
            "Page.javascriptDialogClosed" => self.on_dialog_closed(params, session_id),
            "Fetch.requestPaused" => self.on_fetch_paused(socket, conn, params, session_id),
            "Page.frameAttached" => self.on_frame_attached(params, session_id),
            "Page.frameNavigated" => self.on_frame_navigated(params, session_id),
            "Page.frameDetached" => self.on_frame_detached(params),
            "Target.attachedToTarget" => self.on_target_attached(socket, conn, params),
            "Target.detachedFromTarget" => self.on_target_detached(conn, params),
            "Runtime.consoleAPICalled" => self.on_console(params, "api"),
            "Runtime.exceptionThrown" => self.on_console(params, "exception"),
            _ => {}
        }
    }

    fn arm_watchdog(&self, dialog_id: &str) {
        let deadline = Instant::now() + self.dialog_timeout;
        self.lock_state()
            .dialog_deadlines
            .insert(dialog_id.to_string(), deadline);
    }

    fn on_dialog_opening(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        params: &Value,
        session_id: Option<&str>,
    ) {
        let seq = {
            let mut guard = self.lock_state();
            guard.dialog_seq += 1;
            guard.dialog_seq
        };
        let dialog = PendingDialog {
            id: format!("d-{seq}"),
            dialog_type: str_field(params, "type"),
            message: str_field(params, "message"),
            default_prompt: str_field(params, "defaultPrompt"),
            opened_at: now_secs(),
            cdp_session_id: session_id
                .map(String::from)
                .or_else(|| conn.page_session_id.clone())
                .unwrap_or_default(),
            frame_id: opt_str_field(params, "frameId"),
            bridge_request_id: None,
        };

        match self.dialog_policy.as_str() {
            DIALOG_POLICY_AUTO_DISMISS => {
                self.lock_state().archive_dialog(&dialog, "auto_policy");
                self.auto_handle_dialog(socket, conn, &dialog, false, "");
            }
            DIALOG_POLICY_AUTO_ACCEPT => {
                self.lock_state().archive_dialog(&dialog, "auto_policy");
                let prompt = dialog.default_prompt.clone();
                self.auto_handle_dialog(socket, conn, &dialog, true, &prompt);
            }
            _ => {
                let id = dialog.id.clone();
                self.lock_state().insert_pending(dialog);
                self.arm_watchdog(&id);
            }
        }
    }

    /// Send handleJavaScriptDialog for auto_dismiss/auto_accept.
    fn auto_handle_dialog(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        dialog: &PendingDialog,
        accept: bool,
        prompt_text: &str,
    ) {
        let mut params = Map::new();
        params.insert("accept".to_string(), Value::Bool(accept));
        if dialog.dialog_type == "prompt" {
            params.insert(
                "promptText".to_string(),
                Value::String(prompt_text.to_string()),
            );
        }
        let sid = if dialog.cdp_session_id.is_empty() {
            None
        } else {
            Some(dialog.cdp_session_id.as_str())
        };
        if let Err(e) = self.cdp(
            socket,
            conn,
            "Page.handleJavaScriptDialog",
            Some(Value::Object(params)),
            sid,
            5.0,
        ) {
            log::debug!("auto-handle CDP call failed for {}: {}", dialog.id, e);
        }
    }

    fn dialog_timeout_expired(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        dialog_id: &str,
    ) {
        let dialog = { self.lock_state().pending_dialogs.get(dialog_id).cloned() };
        let dialog = match dialog {
            Some(d) => d,
            None => {
                self.lock_state().dialog_deadlines.remove(dialog_id);
                return;
            }
        };
        log::warn!(
            "CDP supervisor {}: dialog {} ({}) auto-dismissed after {}s timeout",
            self.task_id,
            dialog_id,
            dialog.dialog_type,
            self.dialog_timeout_s
        );
        {
            let mut guard = self.lock_state();
            if guard.pending_dialogs.contains_key(dialog_id) {
                if let Some(d) = guard.remove_pending(dialog_id) {
                    guard.archive_dialog(&d, "watchdog");
                }
            }
        }
        if dialog.bridge_request_id.is_some() {
            self.fulfill_bridge_request(socket, conn, &dialog, false, "");
        } else {
            let sid = if dialog.cdp_session_id.is_empty() {
                None
            } else {
                Some(dialog.cdp_session_id.as_str())
            };
            if let Err(e) = self.cdp(
                socket,
                conn,
                "Page.handleJavaScriptDialog",
                Some(json!({"accept": false})),
                sid,
                5.0,
            ) {
                log::debug!("auto-dismiss failed for {}: {}", dialog_id, e);
            }
        }
    }

    /// Send the Page.handleJavaScriptDialog CDP command (agent path only).
    /// Routes to the bridge-fulfill path when captured via the injected XHR.
    fn handle_dialog_cdp(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        dialog: &PendingDialog,
        accept: bool,
        prompt_text: &str,
    ) -> Result<(), String> {
        if dialog.bridge_request_id.is_some() {
            self.fulfill_bridge_request(socket, conn, dialog, accept, prompt_text);
            let mut guard = self.lock_state();
            if guard.pending_dialogs.contains_key(&dialog.id) {
                if let Some(d) = guard.remove_pending(&dialog.id) {
                    guard.archive_dialog(&d, "agent");
                }
            }
            return Ok(());
        }

        let mut params = Map::new();
        params.insert("accept".to_string(), Value::Bool(accept));
        if dialog.dialog_type == "prompt" {
            params.insert(
                "promptText".to_string(),
                Value::String(prompt_text.to_string()),
            );
        }
        let sid = if dialog.cdp_session_id.is_empty() {
            None
        } else {
            Some(dialog.cdp_session_id.as_str())
        };
        let result = self.cdp(
            socket,
            conn,
            "Page.handleJavaScriptDialog",
            Some(Value::Object(params)),
            sid,
            5.0,
        );
        // Clear regardless — CDP error usually means the dialog already closed.
        {
            let mut guard = self.lock_state();
            if guard.pending_dialogs.contains_key(&dialog.id) {
                if let Some(d) = guard.remove_pending(&dialog.id) {
                    guard.archive_dialog(&d, "agent");
                }
            }
        }
        result.map(|_| ())
    }

    fn on_dialog_closed(&self, _params: &Value, session_id: Option<&str>) {
        let mut guard = self.lock_state();
        // Match by session id; clear the oldest non-bridge dialog on that
        // session.
        let candidate = guard
            .pending_order
            .iter()
            .filter_map(|id| guard.pending_dialogs.get(id))
            .find(|d| {
                let same_session = match session_id {
                    Some(s) => d.cdp_session_id == s,
                    None => d.cdp_session_id.is_empty(),
                };
                same_session && d.bridge_request_id.is_none()
            })
            .map(|d| d.id.clone());
        if let Some(did) = candidate {
            if let Some(d) = guard.remove_pending(&did) {
                guard.archive_dialog(&d, "remote");
            }
        }
    }

    /// Bridge XHR captured mid-flight — materialize as a pending dialog.
    fn on_fetch_paused(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        params: &Value,
        session_id: Option<&str>,
    ) {
        let url = params
            .get("request")
            .and_then(|r| r.get("url"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let request_id = match params.get("requestId").and_then(|v| v.as_str()) {
            Some(r) if !r.is_empty() => r.to_string(),
            _ => return,
        };

        if !url.contains(DIALOG_BRIDGE_HOST) {
            // Not ours — forward unchanged so the page sees its own request.
            let _ = self.cdp(
                socket,
                conn,
                "Fetch.continueRequest",
                Some(json!({"requestId": request_id})),
                session_id,
                3.0,
            );
            return;
        }

        let (kind, message, default_prompt) = parse_bridge_query(&url);
        let kind = if kind.is_empty() {
            "alert".to_string()
        } else {
            kind
        };

        let seq = {
            let mut guard = self.lock_state();
            guard.dialog_seq += 1;
            guard.dialog_seq
        };
        let dialog = PendingDialog {
            id: format!("d-{seq}"),
            dialog_type: kind,
            message,
            default_prompt: default_prompt.clone(),
            opened_at: now_secs(),
            cdp_session_id: session_id
                .map(String::from)
                .or_else(|| conn.page_session_id.clone())
                .unwrap_or_default(),
            frame_id: opt_str_field(params, "frameId"),
            bridge_request_id: Some(request_id),
        };

        match self.dialog_policy.as_str() {
            DIALOG_POLICY_AUTO_DISMISS => {
                self.lock_state().archive_dialog(&dialog, "auto_policy");
                self.fulfill_bridge_request(socket, conn, &dialog, false, "");
            }
            DIALOG_POLICY_AUTO_ACCEPT => {
                self.lock_state().archive_dialog(&dialog, "auto_policy");
                self.fulfill_bridge_request(socket, conn, &dialog, true, &default_prompt);
            }
            _ => {
                let id = dialog.id.clone();
                self.lock_state().insert_pending(dialog);
                self.arm_watchdog(&id);
            }
        }
    }

    /// Resolve a bridge XHR via Fetch.fulfillRequest so the page unblocks.
    fn fulfill_bridge_request(
        &self,
        socket: &mut CdpSocket,
        conn: &mut ConnectionState,
        dialog: &PendingDialog,
        accept: bool,
        prompt_text: &str,
    ) {
        let request_id = match dialog.bridge_request_id.as_ref() {
            Some(r) => r.clone(),
            None => return,
        };
        let payload = json!({
            "accept": accept,
            "prompt_text": if dialog.dialog_type == "prompt" { prompt_text } else { "" },
            "dialog_id": dialog.id,
        });
        let body = payload.to_string();
        use base64::Engine as _;
        let body_b64 = base64::engine::general_purpose::STANDARD.encode(body.as_bytes());
        let sid = if dialog.cdp_session_id.is_empty() {
            None
        } else {
            Some(dialog.cdp_session_id.as_str())
        };
        if let Err(e) = self.cdp(
            socket,
            conn,
            "Fetch.fulfillRequest",
            Some(json!({
                "requestId": request_id,
                "responseCode": 200,
                "responseHeaders": [
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "Access-Control-Allow-Origin", "value": "*"},
                ],
                "body": body_b64,
            })),
            sid,
            5.0,
        ) {
            log::debug!("bridge fulfill failed for {}: {}", dialog.id, e);
        }
    }

    // ── Frame / target tracking ─────────────────────────────────────────────

    fn on_frame_attached(&self, params: &Value, session_id: Option<&str>) {
        let frame_id = match opt_str_field(params, "frameId") {
            Some(f) if !f.is_empty() => f,
            _ => return,
        };
        let mut guard = self.lock_state();
        guard.insert_frame(FrameInfo {
            frame_id,
            url: String::new(),
            origin: String::new(),
            parent_frame_id: opt_str_field(params, "parentFrameId"),
            is_oopif: false,
            cdp_session_id: session_id.map(String::from),
            name: String::new(),
        });
    }

    fn on_frame_navigated(&self, params: &Value, session_id: Option<&str>) {
        let frame = params.get("frame").cloned().unwrap_or(json!({}));
        let frame_id = match opt_str_field(&frame, "id") {
            Some(f) if !f.is_empty() => f,
            _ => return,
        };
        let mut guard = self.lock_state();
        let existing = guard.frames.get(&frame_id).cloned();
        let origin = {
            let so = str_field(&frame, "securityOrigin");
            if so.is_empty() {
                str_field(&frame, "origin")
            } else {
                so
            }
        };
        let info = FrameInfo {
            frame_id: frame_id.clone(),
            url: str_field(&frame, "url"),
            origin,
            parent_frame_id: opt_str_field(&frame, "parentId")
                .or_else(|| existing.as_ref().and_then(|e| e.parent_frame_id.clone())),
            is_oopif: existing.as_ref().map(|e| e.is_oopif).unwrap_or(false),
            cdp_session_id: existing
                .as_ref()
                .and_then(|e| e.cdp_session_id.clone())
                .or_else(|| session_id.map(String::from)),
            name: {
                let n = str_field(&frame, "name");
                if n.is_empty() {
                    existing.as_ref().map(|e| e.name.clone()).unwrap_or_default()
                } else {
                    n
                }
            },
        };
        guard.insert_frame(info);
    }

    fn on_frame_detached(&self, params: &Value) {
        let frame_id = match opt_str_field(params, "frameId") {
            Some(f) if !f.is_empty() => f,
            _ => return,
        };
        let reason = {
            let r = str_field(params, "reason");
            if r.is_empty() { "remove".to_string() } else { r }
        }
        .to_lowercase();
        if reason == "swap" {
            return;
        }
        let mut guard = self.lock_state();
        if let Some(existing) = guard.frames.get(&frame_id) {
            // Keep OOPIF records even when the parent says the frame was
            // "removed" — the iframe is still visible, in a different process.
            if existing.is_oopif
                && existing
                    .cdp_session_id
                    .as_deref()
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
            {
                return;
            }
        }
        guard.remove_frame(&frame_id);
    }

    fn on_target_attached(&self, socket: &mut CdpSocket, conn: &mut ConnectionState, params: &Value) {
        let info = params.get("targetInfo").cloned().unwrap_or(json!({}));
        let sid = match opt_str_field(params, "sessionId") {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };
        let target_type = str_field(&info, "type");
        if target_type != "iframe" && target_type != "worker" {
            return;
        }
        conn.child_sessions
            .insert(sid.clone(), target_type.clone());

        if target_type == "iframe" {
            if let Some(target_id) = opt_str_field(&info, "targetId").filter(|s| !s.is_empty()) {
                let mut guard = self.lock_state();
                let existing = guard.frames.get(&target_id).cloned();
                guard.insert_frame(FrameInfo {
                    frame_id: target_id.clone(),
                    url: str_field(&info, "url"),
                    origin: String::new(),
                    parent_frame_id: existing
                        .as_ref()
                        .and_then(|e| e.parent_frame_id.clone()),
                    is_oopif: true,
                    cdp_session_id: Some(sid.clone()),
                    name: {
                        let title = str_field(&info, "title");
                        if title.is_empty() {
                            existing.as_ref().map(|e| e.name.clone()).unwrap_or_default()
                        } else {
                            title
                        }
                    },
                });
            }
        }

        // Enable domains on the child session synchronously.
        self.enable_child_domains(socket, conn, &sid);
    }

    fn enable_child_domains(&self, socket: &mut CdpSocket, conn: &mut ConnectionState, sid: &str) {
        if let Err(e) = (|| -> Result<(), String> {
            self.cdp(socket, conn, "Page.enable", None, Some(sid), 3.0)?;
            self.cdp(socket, conn, "Runtime.enable", None, Some(sid), 3.0)?;
            self.cdp(
                socket,
                conn,
                "Target.setAutoAttach",
                Some(json!({
                    "autoAttach": true,
                    "waitForDebuggerOnStart": false,
                    "flatten": true,
                })),
                Some(sid),
                3.0,
            )?;
            Ok(())
        })() {
            log::debug!("child session {} setup failed: {}", &sid[..sid.len().min(16)], e);
        }
        self.install_dialog_bridge(socket, conn, sid);
    }

    fn on_target_detached(&self, conn: &mut ConnectionState, params: &Value) {
        let sid = match opt_str_field(params, "sessionId") {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };
        conn.child_sessions.remove(&sid);
        let mut guard = self.lock_state();
        let ids: Vec<String> = guard.frame_order.clone();
        for fid in ids {
            if let Some(frame) = guard.frames.get(&fid) {
                if frame.cdp_session_id.as_deref() == Some(sid.as_str()) {
                    let mut updated = frame.clone();
                    updated.cdp_session_id = None;
                    guard.frames.insert(fid, updated);
                }
            }
        }
    }

    // ── Console / exception ring buffer ─────────────────────────────────────

    fn on_console(&self, params: &Value, level_from: &str) {
        let event = if level_from == "exception" {
            let details = params.get("exceptionDetails").cloned().unwrap_or(json!({}));
            ConsoleEvent {
                ts: now_secs(),
                level: "exception".to_string(),
                text: str_field(&details, "text"),
                url: opt_str_field(&details, "url"),
            }
        } else {
            let raw_level = {
                let r = str_field(params, "type");
                if r.is_empty() { "log".to_string() } else { r }
            };
            let level = match raw_level.as_str() {
                "error" | "assert" => "error",
                "warning" => "warning",
                _ => "log",
            };
            let args = params
                .get("args")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let mut parts: Vec<String> = Vec::new();
            for a in args.iter().take(4) {
                if a.is_object() {
                    let v = str_field(a, "value");
                    let text = if v.is_empty() {
                        str_field(a, "description")
                    } else {
                        v
                    };
                    parts.push(text);
                }
            }
            ConsoleEvent {
                ts: now_secs(),
                level: level.to_string(),
                text: parts.join(" "),
                url: None,
            }
        };
        self.lock_state().push_console(event);
    }
}

/// Per-connection (per-WebSocket) state held only on the worker thread.
struct ConnectionState {
    next_call_id: u64,
    page_session_id: Option<String>,
    /// session id -> target type ("iframe" | "worker")
    child_sessions: HashMap<String, String>,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            next_call_id: 1,
            page_session_id: None,
            child_sessions: HashMap::new(),
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn str_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn opt_str_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Parse the bridge XHR query string into (kind, message, default_prompt).
/// Mirrors Python's `parse_qs` taking the first value for each key.
fn parse_bridge_query(url: &str) -> (String, String, String) {
    let parsed = Url::parse(url);
    let mut kind = String::new();
    let mut message = String::new();
    let mut default_prompt = String::new();
    if let Ok(u) = parsed {
        let mut seen_kind = false;
        let mut seen_message = false;
        let mut seen_default = false;
        for (k, v) in u.query_pairs() {
            match k.as_ref() {
                "kind" if !seen_kind => {
                    kind = v.into_owned();
                    seen_kind = true;
                }
                "message" if !seen_message => {
                    message = v.into_owned();
                    seen_message = true;
                }
                "default_prompt" if !seen_default => {
                    default_prompt = v.into_owned();
                    seen_default = true;
                }
                _ => {}
            }
        }
    }
    (kind, message, default_prompt)
}

fn set_socket_read_timeout(socket: &mut CdpSocket, timeout: Duration) {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => {
            let _ = stream.set_read_timeout(Some(timeout));
        }
        MaybeTlsStream::Rustls(stream) => {
            let _ = stream.get_mut().set_read_timeout(Some(timeout));
        }
        _ => {}
    }
}

// ── Registry ──────────────────────────────────────────────────────────────────

/// Process-global (task_id → supervisor) map with idempotent start/stop.
pub struct SupervisorRegistry {
    by_task: Mutex<HashMap<String, Arc<CdpSupervisor>>>,
}

impl SupervisorRegistry {
    fn new() -> Self {
        Self {
            by_task: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<CdpSupervisor>>> {
        self.by_task.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Return the supervisor for `task_id` if running, else `None`.
    pub fn get(&self, task_id: &str) -> Option<Arc<CdpSupervisor>> {
        self.lock().get(task_id).cloned()
    }

    /// Idempotently ensure a supervisor is running for `(task_id, cdp_url)`.
    ///
    /// If a supervisor exists for this task but was bound to a different
    /// `cdp_url`, the old one is stopped and a fresh one is started.
    pub fn get_or_start(
        &self,
        task_id: &str,
        cdp_url: &str,
        dialog_policy: &str,
        dialog_timeout_s: f64,
        start_timeout: Duration,
    ) -> Result<Arc<CdpSupervisor>, String> {
        let existing = {
            let mut guard = self.lock();
            match guard.get(task_id).cloned() {
                Some(existing) => {
                    if existing.cdp_url == cdp_url && existing.is_thread_alive() {
                        return Ok(existing);
                    }
                    // URL changed or unhealthy — remove, fall through to recreate.
                    guard.remove(task_id);
                    Some(existing)
                }
                None => None,
            }
        };
        if let Some(existing) = existing {
            existing.stop(Duration::from_secs(5));
        }

        let supervisor = Arc::new(CdpSupervisor::new(
            task_id,
            cdp_url,
            dialog_policy,
            dialog_timeout_s,
        )?);
        supervisor.start(start_timeout)?;

        let mut guard = self.lock();
        // Guard against a concurrent get_or_start from another thread.
        if let Some(already) = guard.get(task_id).cloned() {
            if already.cdp_url == cdp_url {
                drop(guard);
                supervisor.stop(Duration::from_secs(5));
                return Ok(already);
            }
        }
        guard.insert(task_id.to_string(), Arc::clone(&supervisor));
        Ok(supervisor)
    }

    /// Stop and discard the supervisor for `task_id` if it exists.
    pub fn stop(&self, task_id: &str) {
        let supervisor = self.lock().remove(task_id);
        if let Some(supervisor) = supervisor {
            supervisor.stop(Duration::from_secs(5));
        }
    }

    /// Stop every running supervisor. For shutdown / test teardown.
    pub fn stop_all(&self) {
        let items: Vec<Arc<CdpSupervisor>> = {
            let mut guard = self.lock();
            let v = guard.values().cloned().collect();
            guard.clear();
            v
        };
        for supervisor in items {
            supervisor.stop(Duration::from_secs(5));
        }
    }
}

/// Process-global registry, mirroring Python's `SUPERVISOR_REGISTRY`.
pub fn supervisor_registry() -> &'static SupervisorRegistry {
    static REGISTRY: OnceLock<SupervisorRegistry> = OnceLock::new();
    REGISTRY.get_or_init(SupervisorRegistry::new)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dialog(id: &str, dtype: &str, session: &str) -> PendingDialog {
        PendingDialog {
            id: id.to_string(),
            dialog_type: dtype.to_string(),
            message: "msg".to_string(),
            default_prompt: String::new(),
            opened_at: 1.0,
            cdp_session_id: session.to_string(),
            frame_id: None,
            bridge_request_id: None,
        }
    }

    #[test]
    fn invalid_policy_rejected() {
        let err = CdpSupervisor::new("t1", "ws://x", "nope", 1.0).unwrap_err();
        assert!(err.contains("Invalid dialog_policy"));
        assert!(err.contains("auto_accept"));
    }

    #[test]
    fn valid_policies_accepted() {
        for p in valid_policies() {
            assert!(CdpSupervisor::new("t", "ws://x", p, 1.0).is_ok());
        }
    }

    #[test]
    fn pending_dialog_to_json_shape() {
        let mut d = make_dialog("d-1", "prompt", "sid");
        d.frame_id = Some("f1".to_string());
        d.default_prompt = "hi".to_string();
        let v = d.to_json();
        assert_eq!(v["id"], "d-1");
        assert_eq!(v["type"], "prompt");
        assert_eq!(v["message"], "msg");
        assert_eq!(v["default_prompt"], "hi");
        assert_eq!(v["frame_id"], "f1");
        // No bridge_request_id / cdp_session_id leaked into the json.
        assert!(v.get("bridge_request_id").is_none());
        assert!(v.get("cdp_session_id").is_none());
    }

    #[test]
    fn frame_info_to_json_omits_empties() {
        let f = FrameInfo {
            frame_id: "f1".to_string(),
            url: "http://x".to_string(),
            origin: "http://x".to_string(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        };
        let v = f.to_json();
        assert_eq!(v["frame_id"], "f1");
        assert_eq!(v["is_oopif"], false);
        assert!(v.get("session_id").is_none());
        assert!(v.get("parent_frame_id").is_none());
        assert!(v.get("name").is_none());

        let f2 = FrameInfo {
            frame_id: "f2".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: Some("f1".to_string()),
            is_oopif: true,
            cdp_session_id: Some("sid".to_string()),
            name: "child".to_string(),
        };
        let v2 = f2.to_json();
        assert_eq!(v2["session_id"], "sid");
        assert_eq!(v2["parent_frame_id"], "f1");
        assert_eq!(v2["name"], "child");
        assert_eq!(v2["is_oopif"], true);
    }

    #[test]
    fn build_frame_tree_empty() {
        let state = SupervisorState::default();
        let tree = state.build_frame_tree();
        assert_eq!(tree["top"], Value::Null);
        assert_eq!(tree["children"], json!([]));
        assert_eq!(tree["truncated"], false);
    }

    #[test]
    fn build_frame_tree_basic() {
        let mut state = SupervisorState::default();
        state.insert_frame(FrameInfo {
            frame_id: "top".to_string(),
            url: "http://top".to_string(),
            origin: "http://top".to_string(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        state.insert_frame(FrameInfo {
            frame_id: "child1".to_string(),
            url: "http://c1".to_string(),
            origin: "http://c1".to_string(),
            parent_frame_id: Some("top".to_string()),
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        let tree = state.build_frame_tree();
        assert_eq!(tree["top"]["frame_id"], "top");
        let children = tree["children"].as_array().unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0]["frame_id"], "child1");
        assert_eq!(tree["truncated"], false);
    }

    #[test]
    fn build_frame_tree_prefers_non_oopif_top() {
        let mut state = SupervisorState::default();
        state.insert_frame(FrameInfo {
            frame_id: "oopif-top".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: true,
            cdp_session_id: Some("s".to_string()),
            name: String::new(),
        });
        state.insert_frame(FrameInfo {
            frame_id: "real-top".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        let tree = state.build_frame_tree();
        assert_eq!(tree["top"]["frame_id"], "real-top");
    }

    #[test]
    fn build_frame_tree_oopif_depth_truncation() {
        let mut state = SupervisorState::default();
        state.insert_frame(FrameInfo {
            frame_id: "top".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        // Chain of oopifs: depth 1, 2, 3 (3 should be truncated).
        let mut parent = "top".to_string();
        for depth in 1..=3 {
            let id = format!("oopif-{depth}");
            state.insert_frame(FrameInfo {
                frame_id: id.clone(),
                url: String::new(),
                origin: String::new(),
                parent_frame_id: Some(parent.clone()),
                is_oopif: true,
                cdp_session_id: Some("s".to_string()),
                name: String::new(),
            });
            parent = id;
        }
        let tree = state.build_frame_tree();
        let children = tree["children"].as_array().unwrap();
        let ids: Vec<&str> = children
            .iter()
            .map(|c| c["frame_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"oopif-1"));
        assert!(ids.contains(&"oopif-2"));
        assert!(!ids.contains(&"oopif-3"));
        assert_eq!(tree["truncated"], true);
    }

    #[test]
    fn build_frame_tree_entries_cap() {
        let mut state = SupervisorState::default();
        state.insert_frame(FrameInfo {
            frame_id: "top".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        for i in 0..(FRAME_TREE_MAX_ENTRIES + 10) {
            state.insert_frame(FrameInfo {
                frame_id: format!("c{i}"),
                url: String::new(),
                origin: String::new(),
                parent_frame_id: Some("top".to_string()),
                is_oopif: false,
                cdp_session_id: None,
                name: String::new(),
            });
        }
        let tree = state.build_frame_tree();
        let children = tree["children"].as_array().unwrap();
        assert_eq!(children.len(), FRAME_TREE_MAX_ENTRIES);
        assert_eq!(tree["truncated"], true);
    }

    #[test]
    fn archive_and_ring_buffer_cap() {
        let mut state = SupervisorState::default();
        for i in 0..(RECENT_DIALOGS_MAX * 3) {
            let d = make_dialog(&format!("d-{i}"), "alert", "s");
            state.archive_dialog(&d, "remote");
        }
        // After exceeding 2x cap, it trims down to RECENT_DIALOGS_MAX.
        assert!(state.recent_dialogs.len() <= RECENT_DIALOGS_MAX * 2);
        assert!(state.recent_dialogs.len() >= RECENT_DIALOGS_MAX);
    }

    #[test]
    fn console_ring_buffer_cap() {
        let mut state = SupervisorState::default();
        for i in 0..(CONSOLE_HISTORY_MAX * 3) {
            state.push_console(ConsoleEvent {
                ts: i as f64,
                level: "log".to_string(),
                text: format!("e{i}"),
                url: None,
            });
        }
        assert!(state.console_events.len() <= CONSOLE_HISTORY_MAX * 2);
    }

    #[test]
    fn pending_ordering_preserved() {
        let mut state = SupervisorState::default();
        state.insert_pending(make_dialog("d-1", "alert", "a"));
        state.insert_pending(make_dialog("d-2", "confirm", "b"));
        state.insert_pending(make_dialog("d-3", "prompt", "c"));
        let ordered = state.ordered_pending();
        assert_eq!(ordered[0].id, "d-1");
        assert_eq!(ordered[1].id, "d-2");
        assert_eq!(ordered[2].id, "d-3");
        // Removing the middle one preserves order of the rest.
        state.remove_pending("d-2");
        let ordered = state.ordered_pending();
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].id, "d-1");
        assert_eq!(ordered[1].id, "d-3");
    }

    #[test]
    fn respond_invalid_action() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let r = sup.respond_to_dialog("frobnicate", None, None, Duration::from_secs(1));
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap().contains("must be 'accept'"));
    }

    #[test]
    fn respond_inactive_supervisor() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        // Never started → not active.
        let r = sup.respond_to_dialog("accept", None, None, Duration::from_secs(1));
        assert_eq!(r["ok"], false);
        assert_eq!(r["error"], "supervisor is not active");
    }

    #[test]
    fn respond_no_dialog_when_active() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        sup.lock_state().active = true;
        let r = sup.respond_to_dialog("accept", None, None, Duration::from_secs(1));
        assert_eq!(r["ok"], false);
        assert_eq!(r["error"], "no dialog is currently open");
    }

    #[test]
    fn respond_ambiguous_dialogs() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        {
            let mut g = sup.lock_state();
            g.active = true;
            g.insert_pending(make_dialog("d-1", "alert", "a"));
            g.insert_pending(make_dialog("d-2", "alert", "b"));
        }
        let r = sup.respond_to_dialog("accept", None, None, Duration::from_secs(1));
        assert_eq!(r["ok"], false);
        let err = r["error"].as_str().unwrap();
        assert!(err.contains("2 pending dialogs"));
        assert!(err.contains("specify dialog_id"));
    }

    #[test]
    fn respond_unknown_dialog_id() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        {
            let mut g = sup.lock_state();
            g.active = true;
            g.insert_pending(make_dialog("d-1", "alert", "a"));
        }
        let r = sup.respond_to_dialog("accept", None, Some("d-99"), Duration::from_secs(1));
        assert_eq!(r["ok"], false);
        let err = r["error"].as_str().unwrap();
        assert!(err.contains("d-99"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn parse_bridge_query_extracts_first_values() {
        let url = "http://hermes-dialog-bridge.invalid/?kind=confirm&message=Are+you+sure%3F&default_prompt=";
        let (kind, message, default_prompt) = parse_bridge_query(url);
        assert_eq!(kind, "confirm");
        assert_eq!(message, "Are you sure?");
        assert_eq!(default_prompt, "");
    }

    #[test]
    fn parse_bridge_query_prompt_default() {
        let url = "http://hermes-dialog-bridge.invalid/?kind=prompt&message=Name&default_prompt=Bob";
        let (kind, message, default_prompt) = parse_bridge_query(url);
        assert_eq!(kind, "prompt");
        assert_eq!(message, "Name");
        assert_eq!(default_prompt, "Bob");
    }

    #[test]
    fn snapshot_to_json_omits_empty_recent() {
        let snap = SupervisorSnapshot {
            pending_dialogs: vec![make_dialog("d-1", "alert", "s")],
            recent_dialogs: vec![],
            frame_tree: json!({"top": null, "children": [], "truncated": false}),
            console_errors: vec![],
            active: true,
            cdp_url: "ws://x".to_string(),
            task_id: "t".to_string(),
        };
        let v = snap.to_json();
        assert!(v.get("pending_dialogs").is_some());
        assert!(v.get("frame_tree").is_some());
        assert!(v.get("recent_dialogs").is_none());
        assert_eq!(v["pending_dialogs"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn snapshot_to_json_includes_recent() {
        let rec = DialogRecord {
            id: "d-1".to_string(),
            dialog_type: "alert".to_string(),
            message: "m".to_string(),
            opened_at: 1.0,
            closed_at: 2.0,
            closed_by: "agent".to_string(),
            frame_id: None,
        };
        let snap = SupervisorSnapshot {
            pending_dialogs: vec![],
            recent_dialogs: vec![rec],
            frame_tree: json!({"top": null, "children": [], "truncated": false}),
            console_errors: vec![],
            active: true,
            cdp_url: "ws://x".to_string(),
            task_id: "t".to_string(),
        };
        let v = snap.to_json();
        assert!(v.get("recent_dialogs").is_some());
        assert_eq!(v["recent_dialogs"][0]["closed_by"], "agent");
    }

    #[test]
    fn on_dialog_closed_archives_matching_session() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let worker = make_test_worker(&sup);
        {
            let mut g = worker.lock_state();
            g.insert_pending(make_dialog("d-1", "alert", "sessA"));
            g.insert_pending(make_dialog("d-2", "alert", "sessB"));
        }
        worker.on_dialog_closed(&json!({}), Some("sessA"));
        let g = worker.lock_state();
        assert!(!g.pending_dialogs.contains_key("d-1"));
        assert!(g.pending_dialogs.contains_key("d-2"));
        assert_eq!(g.recent_dialogs.len(), 1);
        assert_eq!(g.recent_dialogs[0].closed_by, "remote");
    }

    #[test]
    fn on_frame_detached_swap_is_noop() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let worker = make_test_worker(&sup);
        worker.lock_state().insert_frame(FrameInfo {
            frame_id: "f1".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        worker.on_frame_detached(&json!({"frameId": "f1", "reason": "swap"}));
        assert!(worker.lock_state().frames.contains_key("f1"));
    }

    #[test]
    fn on_frame_detached_keeps_live_oopif() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let worker = make_test_worker(&sup);
        worker.lock_state().insert_frame(FrameInfo {
            frame_id: "f1".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: true,
            cdp_session_id: Some("s1".to_string()),
            name: String::new(),
        });
        worker.on_frame_detached(&json!({"frameId": "f1", "reason": "remove"}));
        assert!(worker.lock_state().frames.contains_key("f1"));
    }

    #[test]
    fn on_frame_detached_removes_dead_frame() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let worker = make_test_worker(&sup);
        worker.lock_state().insert_frame(FrameInfo {
            frame_id: "f1".to_string(),
            url: String::new(),
            origin: String::new(),
            parent_frame_id: None,
            is_oopif: false,
            cdp_session_id: None,
            name: String::new(),
        });
        worker.on_frame_detached(&json!({"frameId": "f1", "reason": "remove"}));
        assert!(!worker.lock_state().frames.contains_key("f1"));
    }

    #[test]
    fn on_console_levels() {
        let sup = CdpSupervisor::new("t", "ws://x", DEFAULT_DIALOG_POLICY, 1.0).unwrap();
        let worker = make_test_worker(&sup);
        worker.on_console(
            &json!({"type": "error", "args": [{"value": "boom"}, {"description": "x"}]}),
            "api",
        );
        worker.on_console(
            &json!({"exceptionDetails": {"text": "TypeError", "url": "http://x"}}),
            "exception",
        );
        let g = worker.lock_state();
        assert_eq!(g.console_events.len(), 2);
        assert_eq!(g.console_events[0].level, "error");
        assert_eq!(g.console_events[0].text, "boom x");
        assert_eq!(g.console_events[1].level, "exception");
        assert_eq!(g.console_events[1].text, "TypeError");
        assert_eq!(g.console_events[1].url.as_deref(), Some("http://x"));
    }

    #[test]
    fn registry_get_missing() {
        let reg = SupervisorRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    /// Build a worker that shares the supervisor's state, for unit-testing
    /// event handlers without a live socket.
    fn make_test_worker(sup: &CdpSupervisor) -> SupervisorWorker {
        let (_tx, rx) = mpsc::channel();
        SupervisorWorker {
            task_id: sup.task_id.clone(),
            cdp_url: sup.cdp_url.clone(),
            dialog_policy: sup.dialog_policy.clone(),
            dialog_timeout: Duration::from_secs_f64(sup.dialog_timeout_s),
            dialog_timeout_s: sup.dialog_timeout_s,
            state: Arc::clone(&sup.state),
            ready: Arc::clone(&sup.ready),
            stop_requested: Arc::clone(&sup.stop_requested),
            command_rx: rx,
        }
    }
}
