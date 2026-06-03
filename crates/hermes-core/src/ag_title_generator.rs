//! Auto-generate short session titles from the first user/assistant exchange.
//!
//! Port of `agent/title_generator.py`.
//!
//! In Python this runs asynchronously after the first response is delivered so
//! it never adds latency to the user-facing reply. The Python module wires
//! itself to `agent.auxiliary_client.call_llm`, a `session_db` object with
//! `get_session_title` / `set_session_title`, and a couple of callbacks.
//!
//! To keep this Rust port self-contained (the auxiliary client's `call_llm`
//! entry point and the session DB are not yet ported as concrete callable
//! types here), those collaborators are abstracted behind small traits:
//!
//! * [`TitleLlm`]   — produces a raw title string from the prepared messages,
//!                    mirroring `call_llm(...).choices[0].message.content`.
//! * [`SessionTitleStore`] — `get_session_title` / `set_session_title`.
//!
//! Callers wire concrete implementations (backed by `crate::ag_auxiliary_client`
//! and the session database) when integrating. The pure logic — prompt
//! construction, snippet truncation, title cleanup, first-exchange detection —
//! is reproduced faithfully and is fully unit-tested.

/// Failure callback signature: `(task_name, error_message)`.
///
/// Mirrors the Python `FailureCallback = Callable[[str, BaseException], None]`.
/// Used to surface auxiliary failures to the user (e.g. via
/// `AIAgent._emit_auxiliary_failure`) so silent drops become visible instead
/// of piling up as NULL session titles.
pub type FailureCallback<'a> = dyn FnMut(&str, &str) + 'a;

/// Title callback signature: `(title)`.
///
/// Mirrors the Python `TitleCallback = Callable[[str], None]`.
pub type TitleCallback<'a> = dyn FnMut(&str) + 'a;

/// System prompt used to ask the model for a title. Verbatim from Python's
/// `_TITLE_PROMPT`.
pub const TITLE_PROMPT: &str = "Generate a short, descriptive title (3-7 words) for a conversation that starts with the \
following exchange. The title should capture the main topic or intent. \
Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes.";

/// A single chat message `{role, content}` as passed to the LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Options threaded into the auxiliary call, matching the Python keyword
/// arguments to `call_llm`.
#[derive(Debug, Clone)]
pub struct TitleRequest {
    pub task: &'static str,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: f64,
    pub timeout: f64,
}

/// Abstraction over `agent.auxiliary_client.call_llm`.
///
/// An implementation should perform the request and return the raw assistant
/// content (`response.choices[0].message.content`), or an `Err(message)` to
/// signal a failure (mirroring the Python `except Exception as e`).
pub trait TitleLlm {
    fn call_llm(&self, request: &TitleRequest) -> Result<String, String>;
}

/// Abstraction over the session database's title accessors.
pub trait SessionTitleStore {
    /// Mirrors `session_db.get_session_title(session_id)`. Returns the current
    /// title, or `None` if unset. An `Err` mirrors the Python call raising.
    fn get_session_title(&self, session_id: &str) -> Result<Option<String>, String>;

    /// Mirrors `session_db.set_session_title(session_id, title)`.
    fn set_session_title(&self, session_id: &str, title: &str) -> Result<(), String>;
}

/// Build the two-message request body for title generation.
///
/// Reproduces the Python truncation (`[:500]`) and message layout exactly.
pub fn build_title_messages(user_message: &str, assistant_response: &str) -> Vec<ChatMessage> {
    let user_snippet = truncate_chars(user_message, 500);
    let assistant_snippet = truncate_chars(assistant_response, 500);

    vec![
        ChatMessage::new("system", TITLE_PROMPT),
        ChatMessage::new(
            "user",
            format!("User: {user_snippet}\n\nAssistant: {assistant_snippet}"),
        ),
    ]
}

/// Truncate a string to at most `max` characters (Unicode scalar values),
/// matching Python's `s[:n]` slice semantics on `str`.
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Clean up a raw model title: strip surrounding quotes, drop a leading
/// `"title:"` prefix (case-insensitive), trim, and enforce an 80-char cap.
///
/// Reproduces the cleanup block from Python's `generate_title`. Returns `None`
/// when the resulting title is empty.
pub fn clean_title(raw: &str) -> Option<String> {
    // response content `.strip()`
    let mut title = raw.trim().to_string();

    // title.strip('"\'') — strips any leading/trailing run of " or ' chars.
    title = title
        .trim_matches(|c| c == '"' || c == '\'')
        .to_string();

    // if title.lower().startswith("title:"): title = title[6:].strip()
    if title.to_lowercase().starts_with("title:") {
        // Drop the first 6 *characters* (Python str slice), then strip.
        let after: String = title.chars().skip(6).collect();
        title = after.trim().to_string();
    }

    // Enforce reasonable length: if len > 80, title = title[:77] + "..."
    if title.chars().count() > 80 {
        let head: String = title.chars().take(77).collect();
        title = format!("{head}...");
    }

    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

/// Generate a session title from the first exchange.
///
/// Port of `generate_title`. Returns the cleaned title string, or `None` on
/// failure or empty result. `failure_callback` is invoked with
/// `("title generation", error_message)` when the LLM call fails.
pub fn generate_title(
    llm: &dyn TitleLlm,
    user_message: &str,
    assistant_response: &str,
    timeout: f64,
    mut failure_callback: Option<&mut FailureCallback<'_>>,
) -> Option<String> {
    let messages = build_title_messages(user_message, assistant_response);

    let request = TitleRequest {
        task: "title_generation",
        messages,
        max_tokens: 500,
        temperature: 0.3,
        timeout,
    };

    match llm.call_llm(&request) {
        Ok(content) => clean_title(&content),
        Err(e) => {
            log::warn!("Title generation failed: {e}");
            log::debug!("Title generation traceback");
            if let Some(cb) = failure_callback.as_mut() {
                // Python wraps the callback in try/except; here the callback
                // signature can't fail, so there is nothing to guard.
                cb("title generation", &e);
            }
            None
        }
    }
}

/// Default timeout used by `generate_title` when called from
/// `auto_title_session` (Python's `timeout: float = 30.0`).
pub const DEFAULT_TIMEOUT: f64 = 30.0;

/// Generate and set a session title if one doesn't already exist.
///
/// Port of `auto_title_session`. Intended to be called from a background
/// thread after the first exchange completes.
///
/// Silently skips if:
/// - `session_id` is empty,
/// - the session already has a title,
/// - title generation fails.
#[allow(clippy::too_many_arguments)]
pub fn auto_title_session(
    store: &dyn SessionTitleStore,
    llm: &dyn TitleLlm,
    session_id: &str,
    user_message: &str,
    assistant_response: &str,
    failure_callback: Option<&mut FailureCallback<'_>>,
    mut title_callback: Option<&mut TitleCallback<'_>>,
) {
    // `if not session_db or not session_id: return` — the store is always
    // present here (a reference); guard on the session id only.
    if session_id.is_empty() {
        return;
    }

    // Check if title already exists (user may have set one via /title before
    // the first response). Any error short-circuits to a silent return,
    // matching the Python `except Exception: return`.
    match store.get_session_title(session_id) {
        Ok(Some(existing)) if !existing.is_empty() => return,
        Ok(_) => {}
        Err(_) => return,
    }

    let title = match generate_title(
        llm,
        user_message,
        assistant_response,
        DEFAULT_TIMEOUT,
        failure_callback,
    ) {
        Some(t) => t,
        None => return,
    };

    match store.set_session_title(session_id, &title) {
        Ok(()) => {
            log::debug!("Auto-generated session title: {title}");
            if let Some(cb) = title_callback.as_mut() {
                cb(&title);
            }
        }
        Err(e) => {
            log::debug!("Failed to set auto-generated title: {e}");
        }
    }
}

/// Count `role == "user"` messages in the conversation history.
///
/// Helper extracted from `maybe_auto_title` so the first-exchange heuristic is
/// independently testable.
pub fn count_user_messages(conversation_history: &[ChatMessage]) -> usize {
    conversation_history
        .iter()
        .filter(|m| m.role == "user")
        .count()
}

/// Decide whether title generation should fire for this exchange.
///
/// Port of the guard logic in `maybe_auto_title` (before spawning the thread).
/// Returns `true` when a title should be generated.
///
/// Conditions (all must hold):
/// - `session_id`, `user_message`, and `assistant_response` are non-empty,
/// - the conversation has at most 2 user messages (first/second exchange).
pub fn should_auto_title(
    session_id: &str,
    user_message: &str,
    assistant_response: &str,
    conversation_history: &[ChatMessage],
) -> bool {
    if session_id.is_empty() || user_message.is_empty() || assistant_response.is_empty() {
        return false;
    }

    let user_msg_count = count_user_messages(conversation_history);
    user_msg_count <= 2
}

/// Fire title generation after the first exchange.
///
/// Port of `maybe_auto_title`. Unlike the Python version this runs
/// synchronously: the Python code spawns a daemon thread purely to keep the
/// auxiliary LLM call off the user-facing reply path. Callers that want the
/// same fire-and-forget behaviour should invoke this from their own background
/// task (e.g. `std::thread::spawn` or a tokio blocking task); the gating logic
/// is identical either way. Returns `true` if title generation was attempted.
#[allow(clippy::too_many_arguments)]
pub fn maybe_auto_title(
    store: &dyn SessionTitleStore,
    llm: &dyn TitleLlm,
    session_id: &str,
    user_message: &str,
    assistant_response: &str,
    conversation_history: &[ChatMessage],
    failure_callback: Option<&mut FailureCallback<'_>>,
    title_callback: Option<&mut TitleCallback<'_>>,
) -> bool {
    if !should_auto_title(session_id, user_message, assistant_response, conversation_history) {
        return false;
    }

    auto_title_session(
        store,
        llm,
        session_id,
        user_message,
        assistant_response,
        failure_callback,
        title_callback,
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    struct StubLlm {
        response: Result<String, String>,
    }
    impl TitleLlm for StubLlm {
        fn call_llm(&self, _request: &TitleRequest) -> Result<String, String> {
            self.response.clone()
        }
    }

    #[derive(Default)]
    struct MemStore {
        titles: RefCell<HashMap<String, String>>,
        get_should_err: bool,
        set_should_err: bool,
    }
    impl SessionTitleStore for MemStore {
        fn get_session_title(&self, session_id: &str) -> Result<Option<String>, String> {
            if self.get_should_err {
                return Err("boom".into());
            }
            Ok(self.titles.borrow().get(session_id).cloned())
        }
        fn set_session_title(&self, session_id: &str, title: &str) -> Result<(), String> {
            if self.set_should_err {
                return Err("nope".into());
            }
            self.titles
                .borrow_mut()
                .insert(session_id.to_string(), title.to_string());
            Ok(())
        }
    }

    #[test]
    fn build_messages_truncates_to_500_chars() {
        let long_user = "u".repeat(600);
        let long_asst = "a".repeat(700);
        let msgs = build_title_messages(&long_user, &long_asst);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, TITLE_PROMPT);
        assert_eq!(msgs[1].role, "user");
        let expected_user = "u".repeat(500);
        let expected_asst = "a".repeat(500);
        assert_eq!(
            msgs[1].content,
            format!("User: {expected_user}\n\nAssistant: {expected_asst}")
        );
    }

    #[test]
    fn build_messages_handles_empty_inputs() {
        let msgs = build_title_messages("", "");
        assert_eq!(msgs[1].content, "User: \n\nAssistant: ");
    }

    #[test]
    fn clean_strips_quotes_and_title_prefix() {
        assert_eq!(clean_title("\"Hello World\""), Some("Hello World".into()));
        assert_eq!(clean_title("'Hello World'"), Some("Hello World".into()));
        assert_eq!(clean_title("Title: My Topic"), Some("My Topic".into()));
        assert_eq!(clean_title("TITLE: My Topic"), Some("My Topic".into()));
        assert_eq!(clean_title("  spaced  "), Some("spaced".into()));
    }

    #[test]
    fn clean_empty_returns_none() {
        assert_eq!(clean_title(""), None);
        assert_eq!(clean_title("   "), None);
        assert_eq!(clean_title("\"\""), None);
    }

    #[test]
    fn clean_enforces_80_char_cap() {
        let raw = "x".repeat(100);
        let cleaned = clean_title(&raw).unwrap();
        assert_eq!(cleaned.chars().count(), 80);
        assert!(cleaned.ends_with("..."));
        assert_eq!(cleaned, format!("{}...", "x".repeat(77)));
    }

    #[test]
    fn clean_exactly_80_unchanged() {
        let raw = "y".repeat(80);
        assert_eq!(clean_title(&raw), Some(raw));
    }

    #[test]
    fn generate_title_success() {
        let llm = StubLlm {
            response: Ok("  \"Great Chat\"  ".into()),
        };
        let title = generate_title(&llm, "hi", "hello", 30.0, None);
        assert_eq!(title, Some("Great Chat".into()));
    }

    #[test]
    fn generate_title_failure_invokes_callback() {
        let llm = StubLlm {
            response: Err("402 payment required".into()),
        };
        let mut captured: Vec<(String, String)> = Vec::new();
        let mut cb = |task: &str, err: &str| {
            captured.push((task.to_string(), err.to_string()));
        };
        let title = generate_title(
            &llm,
            "hi",
            "hello",
            30.0,
            Some(&mut cb as &mut FailureCallback<'_>),
        );
        assert_eq!(title, None);
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, "title generation");
        assert!(captured[0].1.contains("402"));
    }

    #[test]
    fn auto_title_skips_when_title_exists() {
        let store = MemStore::default();
        store
            .titles
            .borrow_mut()
            .insert("s1".into(), "Existing".into());
        let llm = StubLlm {
            response: Ok("New Title".into()),
        };
        auto_title_session(&store, &llm, "s1", "u", "a", None, None);
        assert_eq!(store.titles.borrow().get("s1").unwrap(), "Existing");
    }

    #[test]
    fn auto_title_skips_on_get_error() {
        let store = MemStore {
            get_should_err: true,
            ..Default::default()
        };
        let llm = StubLlm {
            response: Ok("New".into()),
        };
        auto_title_session(&store, &llm, "s1", "u", "a", None, None);
        assert!(store.titles.borrow().is_empty());
    }

    #[test]
    fn auto_title_sets_and_calls_back() {
        let store = MemStore::default();
        let llm = StubLlm {
            response: Ok("Fresh Title".into()),
        };
        let mut seen: Vec<String> = Vec::new();
        {
            let mut cb = |t: &str| seen.push(t.to_string());
            auto_title_session(
                &store,
                &llm,
                "s1",
                "u",
                "a",
                None,
                Some(&mut cb as &mut TitleCallback<'_>),
            );
        }
        assert_eq!(store.titles.borrow().get("s1").unwrap(), "Fresh Title");
        assert_eq!(seen, vec!["Fresh Title".to_string()]);
    }

    #[test]
    fn auto_title_empty_session_id_returns() {
        let store = MemStore::default();
        let llm = StubLlm {
            response: Ok("X".into()),
        };
        auto_title_session(&store, &llm, "", "u", "a", None, None);
        assert!(store.titles.borrow().is_empty());
    }

    #[test]
    fn auto_title_set_error_no_callback() {
        let store = MemStore {
            set_should_err: true,
            ..Default::default()
        };
        let llm = StubLlm {
            response: Ok("X".into()),
        };
        let mut called = false;
        {
            let mut cb = |_t: &str| called = true;
            auto_title_session(
                &store,
                &llm,
                "s1",
                "u",
                "a",
                None,
                Some(&mut cb as &mut TitleCallback<'_>),
            );
        }
        assert!(!called);
    }

    #[test]
    fn should_auto_title_gating() {
        let hist = vec![ChatMessage::new("user", "hi")];
        assert!(should_auto_title("s1", "u", "a", &hist));
        // empty fields
        assert!(!should_auto_title("", "u", "a", &hist));
        assert!(!should_auto_title("s1", "", "a", &hist));
        assert!(!should_auto_title("s1", "u", "", &hist));
        // too many user messages
        let big = vec![
            ChatMessage::new("user", "1"),
            ChatMessage::new("assistant", "a"),
            ChatMessage::new("user", "2"),
            ChatMessage::new("user", "3"),
        ];
        assert!(!should_auto_title("s1", "u", "a", &big));
        // exactly 2 is allowed
        let two = vec![
            ChatMessage::new("user", "1"),
            ChatMessage::new("user", "2"),
        ];
        assert!(should_auto_title("s1", "u", "a", &two));
    }

    #[test]
    fn count_user_messages_only_user_role() {
        let hist = vec![
            ChatMessage::new("system", "s"),
            ChatMessage::new("user", "1"),
            ChatMessage::new("assistant", "a"),
            ChatMessage::new("user", "2"),
        ];
        assert_eq!(count_user_messages(&hist), 2);
        assert_eq!(count_user_messages(&[]), 0);
    }

    #[test]
    fn maybe_auto_title_attempts_when_gated_in() {
        let store = MemStore::default();
        let llm = StubLlm {
            response: Ok("Topic Title".into()),
        };
        let hist = vec![ChatMessage::new("user", "hi")];
        let attempted =
            maybe_auto_title(&store, &llm, "s1", "u", "a", &hist, None, None);
        assert!(attempted);
        assert_eq!(store.titles.borrow().get("s1").unwrap(), "Topic Title");
    }

    #[test]
    fn maybe_auto_title_skips_when_gated_out() {
        let store = MemStore::default();
        let llm = StubLlm {
            response: Ok("X".into()),
        };
        let attempted = maybe_auto_title(&store, &llm, "s1", "", "a", &[], None, None);
        assert!(!attempted);
        assert!(store.titles.borrow().is_empty());
    }
}
