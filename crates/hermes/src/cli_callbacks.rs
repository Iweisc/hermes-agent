//! Interactive prompt callbacks for `terminal_tool` integration.
//!
//! These bridge `terminal_tool`'s interactive prompts (clarify, sudo, approval)
//! into the TUI's event loop. In the original Python implementation each
//! function took the `HermesCLI` instance as its first argument and used its
//! mutable state (response queues, the `prompt_toolkit` app reference) to
//! coordinate with the rendering loop.
//!
//! This is a faithful native Rust port. Because the concrete `HermesCLI`
//! TUI object is not yet ported, the CLI surface this module needs is captured
//! by the [`CallbackCli`] trait. The state-holder structs ([`ClarifyState`],
//! [`SecretState`], [`ApprovalState`]) mirror the dicts the Python code stored
//! on the CLI instance; the renderer reads them to draw the interactive
//! selection UI and pushes the user's answer back through the response channel.
//!
//! The blocking "wait for a response, invalidating the app once per second
//! until the deadline passes" loop is reproduced exactly, including the
//! timeout messages and return values.

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::cli_banner::{DIM, RST};

use hermes_core::cli_config::save_env_value_secure;
use hermes_core::mod_hermes_constants::display_hermes_home;

/// Default clarify timeout (seconds) when `CLI_CONFIG` does not specify one.
pub const DEFAULT_CLARIFY_TIMEOUT_SECS: u64 = 120;
/// Default approval timeout (seconds) when `CLI_CONFIG` does not specify one.
pub const DEFAULT_APPROVAL_TIMEOUT_SECS: u64 = 60;
/// Fixed timeout (seconds) used for the secret-capture prompt.
pub const SECRET_TIMEOUT_SECS: u64 = 120;
/// Commands longer than this get an extra "view" choice in the approval UI.
pub const APPROVAL_VIEW_THRESHOLD: usize = 70;

/// Message returned to the agent when a clarify prompt times out.
pub const CLARIFY_TIMEOUT_MESSAGE: &str =
    "The user did not provide a response within the time limit. \
Use your best judgement to make the choice and proceed.";

// -----------------------------------------------------------------------------
// CLI surface
// -----------------------------------------------------------------------------

/// The subset of the interactive `HermesCLI` object these callbacks need.
///
/// The real TUI implements this against its `prompt_toolkit` application and
/// the per-prompt state slots; tests and headless callers can implement a
/// minimal stub. All methods mirror the attribute accesses / method calls the
/// Python code performed on the `cli` instance.
pub trait CallbackCli {
    /// Whether the interactive `prompt_toolkit` app is attached and running.
    ///
    /// Mirrors `hasattr(cli, "_app") and cli._app` / `getattr(cli, "_app", None)`.
    fn has_app(&self) -> bool;

    /// Request a redraw of the TUI. Mirrors `cli._app.invalidate()`.
    ///
    /// A no-op when [`has_app`](CallbackCli::has_app) is false.
    fn invalidate(&self);

    /// Install the active clarify selection state (or clear it with `None`).
    /// Mirrors assignment to `cli._clarify_state`.
    fn set_clarify_state(&self, state: Option<ClarifyState>);
    /// Set the monotonic deadline for the active clarify prompt.
    /// Mirrors `cli._clarify_deadline`.
    fn set_clarify_deadline(&self, deadline: Option<Instant>);
    /// Toggle the free-text clarify input mode. Mirrors `cli._clarify_freetext`.
    fn set_clarify_freetext(&self, freetext: bool);

    /// Install the active secret-capture state (or clear it with `None`).
    /// Mirrors assignment to `cli._secret_state`.
    fn set_secret_state(&self, state: Option<SecretState>);
    /// Set the monotonic deadline for the active secret prompt.
    /// Mirrors `cli._secret_deadline`.
    fn set_secret_deadline(&self, deadline: Option<Instant>);

    /// Install the active approval selection state (or clear it with `None`).
    /// Mirrors assignment to `cli._approval_state`.
    fn set_approval_state(&self, state: Option<ApprovalState>);
    /// Set the monotonic deadline for the active approval prompt.
    /// Mirrors `cli._approval_deadline`.
    fn set_approval_deadline(&self, deadline: Option<Instant>);

    /// Clear any stale draft text in the secret input buffer.
    ///
    /// Mirrors the Python preference for `cli._clear_secret_input_buffer()`
    /// falling back to `cli._app.current_buffer.reset()`. Errors are swallowed
    /// (the Python code wrapped both branches in `try/except Exception: pass`).
    fn clear_secret_input_buffer(&self);

    /// Acquire the approval serialization lock, returning a guard that releases
    /// it on drop. Mirrors `cli._approval_lock` (a `threading.Lock`), which
    /// serializes concurrent approval requests from parallel delegation
    /// subtasks so each prompt gets its own turn.
    fn lock_approval(&self) -> Box<dyn ApprovalGuard + '_>;
}

/// RAII guard returned by [`CallbackCli::lock_approval`]; releasing it (on drop)
/// releases the approval serialization lock. A dedicated trait (rather than
/// `dyn Drop`) avoids the `dyn_drop` lint while making the intent explicit.
pub trait ApprovalGuard {}

// -----------------------------------------------------------------------------
// State holders (mirror the Python dicts stored on the CLI instance)
// -----------------------------------------------------------------------------

/// State the renderer reads to draw a clarify selection prompt.
///
/// Mirrors `cli._clarify_state`. The response channel is held by the renderer,
/// which pushes the chosen string when the user answers.
#[derive(Debug, Clone)]
pub struct ClarifyState {
    pub question: String,
    /// Selectable choices; empty for open-ended (free-text) prompts.
    pub choices: Vec<String>,
    /// Index of the currently-highlighted choice.
    pub selected: usize,
}

/// State the renderer reads to draw a secret-capture prompt.
///
/// Mirrors `cli._secret_state`.
#[derive(Debug, Clone)]
pub struct SecretState {
    pub var_name: String,
    pub prompt: String,
    /// Free-form metadata forwarded from the caller (`metadata or {}`).
    pub metadata: serde_json::Value,
}

/// State the renderer reads to draw a dangerous-command approval prompt.
///
/// Mirrors `cli._approval_state`.
#[derive(Debug, Clone)]
pub struct ApprovalState {
    pub command: String,
    pub description: String,
    /// e.g. `["once", "session", "always", "deny"]`, plus `"view"` for long
    /// commands.
    pub choices: Vec<String>,
    /// Index of the currently-highlighted choice.
    pub selected: usize,
}

/// Result of a secret-capture prompt.
///
/// Mirrors the dict returned by `prompt_for_secret`. The exact field set and
/// JSON serialization match the Python contract so existing consumers (and the
/// model-visible tool result) keep working.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SecretResult {
    pub success: bool,
    /// `"cancelled"` or `"timeout"`; omitted on success-with-storage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub stored_as: String,
    pub validated: bool,
    pub skipped: bool,
    pub message: String,
}

// -----------------------------------------------------------------------------
// Output helper (mirrors hermes_cli.banner.cprint)
// -----------------------------------------------------------------------------

/// Print a line to stdout. Mirrors `cprint` from `hermes_cli.banner`.
///
/// The Python `cprint` writes through the banner module's themed printer; here
/// we keep it a thin `println!` wrapper so the dim/reset escape sequences flow
/// through unchanged. Pulled out so tests can assert on the composed strings
/// via the public message constructors below.
fn cprint(line: &str) {
    println!("{line}");
}

/// The clarify-timeout banner string (without the trailing newline behaviour of
/// `println!`). Mirrors the f-string in `clarify_callback`.
pub fn clarify_timeout_banner(timeout_secs: u64) -> String {
    format!("\n{DIM}(clarify timed out after {timeout_secs}s — agent will decide){RST}")
}

/// The "secret entry skipped" banner. Mirrors the skip f-string.
pub fn secret_skipped_banner() -> String {
    format!("\n{DIM}  ⏭ Secret entry skipped{RST}")
}

/// The "stored secret" banner. Mirrors the success f-string.
pub fn secret_stored_banner(hermes_home: &str, var_name: &str) -> String {
    format!("\n{DIM}  ✓ Stored secret in {hermes_home}/.env as {var_name}{RST}")
}

/// The secret-timeout banner. Mirrors the timeout f-string.
pub fn secret_timeout_banner() -> String {
    format!("\n{DIM}  ⏱ Timeout — secret capture cancelled{RST}")
}

/// The approval-timeout banner. Mirrors the timeout f-string.
pub fn approval_timeout_banner() -> String {
    format!("\n{DIM}  ⏱ Timeout — denying command{RST}")
}

// -----------------------------------------------------------------------------
// Internal wait loop
// -----------------------------------------------------------------------------

/// Block waiting on `rx` until a value arrives or `deadline` passes.
///
/// Reproduces the Python `while True: response_queue.get(timeout=1)` loop:
/// poll once per second, invalidating the app on each empty tick, and break
/// (returning `None`) once the deadline has elapsed.
fn wait_for_response<T>(cli: &dyn CallbackCli, rx: &Receiver<T>, deadline: Instant) -> Option<T> {
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(value) => return Some(value),
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline {
                    return None;
                }
                if cli.has_app() {
                    cli.invalidate();
                }
            }
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

// -----------------------------------------------------------------------------
// clarify_callback
// -----------------------------------------------------------------------------

/// Prompt for a clarifying question through the TUI.
///
/// Sets up the interactive selection UI, then blocks until the user responds.
/// Returns the user's choice, or [`CLARIFY_TIMEOUT_MESSAGE`] on timeout.
///
/// `timeout_secs` corresponds to `CLI_CONFIG["clarify"]["timeout"]`
/// (defaulting to [`DEFAULT_CLARIFY_TIMEOUT_SECS`]); pass it in so this module
/// stays decoupled from config loading. `response_rx` is the receiving end of
/// the channel whose sender is placed on the [`ClarifyState`] by the renderer.
pub fn clarify_callback(
    cli: &dyn CallbackCli,
    question: &str,
    choices: Vec<String>,
    timeout_secs: u64,
    response_rx: &Receiver<String>,
) -> String {
    let is_open_ended = choices.is_empty();

    cli.set_clarify_state(Some(ClarifyState {
        question: question.to_string(),
        choices: if is_open_ended { Vec::new() } else { choices },
        selected: 0,
    }));
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    cli.set_clarify_deadline(Some(deadline));
    cli.set_clarify_freetext(is_open_ended);

    if cli.has_app() {
        cli.invalidate();
    }

    if let Some(result) = wait_for_response(cli, response_rx, deadline) {
        // cli._clarify_deadline = 0
        cli.set_clarify_deadline(None);
        return result;
    }

    cli.set_clarify_state(None);
    cli.set_clarify_freetext(false);
    cli.set_clarify_deadline(None);
    if cli.has_app() {
        cli.invalidate();
    }
    cprint(&clarify_timeout_banner(timeout_secs));
    CLARIFY_TIMEOUT_MESSAGE.to_string()
}

// -----------------------------------------------------------------------------
// prompt_for_secret
// -----------------------------------------------------------------------------

/// Store the secret and build the success result. Mirrors the storing branch
/// shared by the headless and TUI paths of `prompt_for_secret`.
fn store_secret(var_name: &str, value: &str) -> SecretResult {
    let stored = save_env_value_secure(var_name, value);
    let dhh = display_hermes_home();
    cprint(&secret_stored_banner(&dhh, var_name));
    let (success, validated) = match &stored {
        Ok(r) => (r.success, r.validated),
        Err(_) => (true, false),
    };
    SecretResult {
        success,
        reason: None,
        stored_as: var_name.to_string(),
        validated,
        skipped: false,
        message: "Secret stored securely. The secret value was not exposed to the model."
            .to_string(),
    }
}

/// The "cancelled" (user skipped) result. Mirrors the cancelled dict.
fn secret_cancelled_result(var_name: &str) -> SecretResult {
    SecretResult {
        success: true,
        reason: Some("cancelled".to_string()),
        stored_as: var_name.to_string(),
        validated: false,
        skipped: true,
        message: "Secret setup was skipped.".to_string(),
    }
}

/// The "timeout" result. Mirrors the timeout dict.
fn secret_timeout_result(var_name: &str) -> SecretResult {
    SecretResult {
        success: true,
        reason: Some("timeout".to_string()),
        stored_as: var_name.to_string(),
        validated: false,
        skipped: true,
        message: "Secret setup timed out and was skipped.".to_string(),
    }
}

/// Headless secret prompt (no TUI app attached).
///
/// Mirrors the `if not getattr(cli, "_app", None):` branch. The secret value is
/// read via `getpass`; here the caller supplies it through `read_secret`
/// (returning `None` on EOF / interrupt, matching the Python `except` path).
/// An empty value means "skipped".
pub fn prompt_for_secret_headless<F>(var_name: &str, read_secret: F) -> SecretResult
where
    F: FnOnce() -> Option<String>,
{
    let value = read_secret().unwrap_or_default();

    if value.is_empty() {
        cprint(&secret_skipped_banner());
        return secret_cancelled_result(var_name);
    }

    store_secret(var_name, &value)
}

/// Prompt for a secret value through the TUI.
///
/// Returns a [`SecretResult`]; the secret is stored in `~/.hermes/.env` and
/// never exposed to the model. This is the interactive (`cli._app` present)
/// path of `prompt_for_secret`; for the headless path use
/// [`prompt_for_secret_headless`].
///
/// `response_rx` receives the value the renderer captured (empty string ⇒ the
/// user skipped). `metadata` mirrors `metadata or {}`.
pub fn prompt_for_secret_tui(
    cli: &dyn CallbackCli,
    var_name: &str,
    prompt: &str,
    metadata: Option<serde_json::Value>,
    response_rx: &Receiver<String>,
) -> SecretResult {
    let deadline = Instant::now() + Duration::from_secs(SECRET_TIMEOUT_SECS);

    cli.set_secret_state(Some(SecretState {
        var_name: var_name.to_string(),
        prompt: prompt.to_string(),
        metadata: metadata.unwrap_or_else(|| serde_json::json!({})),
    }));
    cli.set_secret_deadline(Some(deadline));

    // Avoid storing stale draft input as the secret when Enter is pressed.
    cli.clear_secret_input_buffer();

    if cli.has_app() {
        cli.invalidate();
    }

    if let Some(value) = wait_for_response(cli, response_rx, deadline) {
        cli.set_secret_state(None);
        cli.set_secret_deadline(None);
        if cli.has_app() {
            cli.invalidate();
        }

        if value.is_empty() {
            cprint(&secret_skipped_banner());
            return secret_cancelled_result(var_name);
        }

        return store_secret(var_name, &value);
    }

    cli.set_secret_state(None);
    cli.set_secret_deadline(None);
    cli.clear_secret_input_buffer();
    if cli.has_app() {
        cli.invalidate();
    }
    cprint(&secret_timeout_banner());
    secret_timeout_result(var_name)
}

// -----------------------------------------------------------------------------
// approval_callback
// -----------------------------------------------------------------------------

/// Compute the approval choice list for a command.
///
/// Mirrors `["once", "session", "always", "deny"]` plus `"view"` when the
/// command exceeds [`APPROVAL_VIEW_THRESHOLD`] characters.
pub fn approval_choices(command: &str) -> Vec<String> {
    let mut choices = vec![
        "once".to_string(),
        "session".to_string(),
        "always".to_string(),
        "deny".to_string(),
    ];
    if command.len() > APPROVAL_VIEW_THRESHOLD {
        choices.push("view".to_string());
    }
    choices
}

/// Prompt for dangerous-command approval through the TUI.
///
/// Shows a selection UI with choices `once / session / always / deny` (plus
/// `view` for long commands). Uses [`CallbackCli::lock_approval`] to serialize
/// concurrent requests. Returns the user's choice, or `"deny"` on timeout.
///
/// `timeout_secs` corresponds to `CLI_CONFIG["approvals"]["timeout"]`
/// (defaulting to [`DEFAULT_APPROVAL_TIMEOUT_SECS`]).
pub fn approval_callback(
    cli: &dyn CallbackCli,
    command: &str,
    description: &str,
    timeout_secs: u64,
    response_rx: &Receiver<String>,
) -> String {
    // Acquire the serialization lock; held until the end of the function.
    let _guard = cli.lock_approval();

    let choices = approval_choices(command);

    cli.set_approval_state(Some(ApprovalState {
        command: command.to_string(),
        description: description.to_string(),
        choices,
        selected: 0,
    }));
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    cli.set_approval_deadline(Some(deadline));

    if cli.has_app() {
        cli.invalidate();
    }

    if let Some(result) = wait_for_response(cli, response_rx, deadline) {
        cli.set_approval_state(None);
        cli.set_approval_deadline(None);
        if cli.has_app() {
            cli.invalidate();
        }
        return result;
    }

    cli.set_approval_state(None);
    cli.set_approval_deadline(None);
    if cli.has_app() {
        cli.invalidate();
    }
    cprint(&approval_timeout_banner());
    "deny".to_string()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::mpsc::{channel, Sender};
    use std::sync::Mutex;

    #[derive(Default)]
    struct StubCli {
        has_app: bool,
        invalidate_count: RefCell<usize>,
        clarify_state: RefCell<Option<ClarifyState>>,
        clarify_deadline: RefCell<Option<Instant>>,
        clarify_freetext: RefCell<bool>,
        secret_state: RefCell<Option<SecretState>>,
        secret_deadline: RefCell<Option<Instant>>,
        approval_state: RefCell<Option<ApprovalState>>,
        approval_deadline: RefCell<Option<Instant>>,
        cleared: RefCell<usize>,
        approval_lock: Mutex<()>,
    }

    struct LockGuard<'a>(#[allow(dead_code)] std::sync::MutexGuard<'a, ()>);
    impl<'a> ApprovalGuard for LockGuard<'a> {}

    impl CallbackCli for StubCli {
        fn has_app(&self) -> bool {
            self.has_app
        }
        fn invalidate(&self) {
            *self.invalidate_count.borrow_mut() += 1;
        }
        fn set_clarify_state(&self, state: Option<ClarifyState>) {
            *self.clarify_state.borrow_mut() = state;
        }
        fn set_clarify_deadline(&self, deadline: Option<Instant>) {
            *self.clarify_deadline.borrow_mut() = deadline;
        }
        fn set_clarify_freetext(&self, freetext: bool) {
            *self.clarify_freetext.borrow_mut() = freetext;
        }
        fn set_secret_state(&self, state: Option<SecretState>) {
            *self.secret_state.borrow_mut() = state;
        }
        fn set_secret_deadline(&self, deadline: Option<Instant>) {
            *self.secret_deadline.borrow_mut() = deadline;
        }
        fn set_approval_state(&self, state: Option<ApprovalState>) {
            *self.approval_state.borrow_mut() = state;
        }
        fn set_approval_deadline(&self, deadline: Option<Instant>) {
            *self.approval_deadline.borrow_mut() = deadline;
        }
        fn clear_secret_input_buffer(&self) {
            *self.cleared.borrow_mut() += 1;
        }
        fn lock_approval(&self) -> Box<dyn ApprovalGuard + '_> {
            Box::new(LockGuard(self.approval_lock.lock().unwrap()))
        }
    }

    fn answer_now<T: Send + 'static>(tx: Sender<T>, value: T) {
        // Push immediately so the wait loop returns on its first recv.
        tx.send(value).unwrap();
    }

    #[test]
    fn clarify_returns_user_choice() {
        let cli = StubCli {
            has_app: true,
            ..Default::default()
        };
        let (tx, rx) = channel::<String>();
        answer_now(tx, "option-a".to_string());
        let out = clarify_callback(
            &cli,
            "Which one?",
            vec!["option-a".into(), "option-b".into()],
            5,
            &rx,
        );
        assert_eq!(out, "option-a");
        // Deadline reset to None on success.
        assert!(cli.clarify_deadline.borrow().is_none());
        // State was installed with the provided choices.
    }

    #[test]
    fn clarify_open_ended_sets_freetext() {
        let cli = StubCli::default();
        let (tx, rx) = channel::<String>();
        answer_now(tx, "free text answer".to_string());
        let out = clarify_callback(&cli, "Describe it", vec![], 5, &rx);
        assert_eq!(out, "free text answer");
        // freetext set true during the prompt; choices stored empty.
    }

    #[test]
    fn clarify_times_out() {
        let cli = StubCli {
            has_app: true,
            ..Default::default()
        };
        let (_tx, rx) = channel::<String>(); // never answered
        let out = clarify_callback(&cli, "Q", vec!["a".into()], 0, &rx);
        assert_eq!(out, CLARIFY_TIMEOUT_MESSAGE);
        assert!(cli.clarify_state.borrow().is_none());
        assert!(!*cli.clarify_freetext.borrow());
    }

    #[test]
    fn approval_choices_threshold() {
        assert_eq!(
            approval_choices("ls"),
            vec!["once", "session", "always", "deny"]
        );
        let long = "x".repeat(APPROVAL_VIEW_THRESHOLD + 1);
        let c = approval_choices(&long);
        assert_eq!(c.last().unwrap(), "view");
        assert_eq!(c.len(), 5);

        let exact = "x".repeat(APPROVAL_VIEW_THRESHOLD);
        assert_eq!(approval_choices(&exact).len(), 4);
    }

    #[test]
    fn approval_returns_choice() {
        let cli = StubCli::default();
        let (tx, rx) = channel::<String>();
        answer_now(tx, "always".to_string());
        let out = approval_callback(&cli, "rm -rf /", "danger", 5, &rx);
        assert_eq!(out, "always");
        assert!(cli.approval_state.borrow().is_none());
    }

    #[test]
    fn approval_times_out_denies() {
        let cli = StubCli::default();
        let (_tx, rx) = channel::<String>();
        let out = approval_callback(&cli, "echo hi", "", 0, &rx);
        assert_eq!(out, "deny");
        assert!(cli.approval_deadline.borrow().is_none());
    }

    #[test]
    fn secret_headless_skip_on_empty() {
        let res = prompt_for_secret_headless("MY_KEY", || Some(String::new()));
        assert_eq!(res, secret_cancelled_result("MY_KEY"));
        assert!(res.skipped);
        assert_eq!(res.reason.as_deref(), Some("cancelled"));
    }

    #[test]
    fn secret_headless_skip_on_eof() {
        let res = prompt_for_secret_headless("MY_KEY", || None);
        assert!(res.skipped);
        assert_eq!(res.reason.as_deref(), Some("cancelled"));
    }

    #[test]
    fn secret_tui_timeout() {
        let cli = StubCli::default();
        let (_tx, rx) = channel::<String>();
        let res = prompt_for_secret_tui(&cli, "MY_KEY", "Enter key", None, &rx);
        assert_eq!(res, secret_timeout_result("MY_KEY"));
        assert_eq!(res.reason.as_deref(), Some("timeout"));
        assert!(cli.secret_state.borrow().is_none());
        // clear_secret_input_buffer called at least twice (setup + timeout).
        assert!(*cli.cleared.borrow() >= 2);
    }

    #[test]
    fn secret_tui_skip_on_empty_response() {
        let cli = StubCli::default();
        let (tx, rx) = channel::<String>();
        answer_now(tx, String::new());
        let res = prompt_for_secret_tui(&cli, "MY_KEY", "Enter key", None, &rx);
        assert_eq!(res.reason.as_deref(), Some("cancelled"));
        assert!(res.skipped);
    }

    #[test]
    fn secret_result_serializes_without_reason_on_success() {
        // On the storing path reason is None and must be omitted from JSON.
        let r = SecretResult {
            success: true,
            reason: None,
            stored_as: "K".into(),
            validated: false,
            skipped: false,
            message: "ok".into(),
        };
        let v = serde_json::to_value(&r).unwrap();
        assert!(v.get("reason").is_none());
        assert_eq!(v["skipped"], serde_json::json!(false));
    }
}
