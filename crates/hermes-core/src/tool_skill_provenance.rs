//! Skill write-origin provenance — task-local signal for distinguishing
//! agent-sediment skill writes from foreground user-directed writes.
//!
//! The curator only consolidates/prunes skills it autonomously created via the
//! background self-improvement review fork. Skills a user asks a foreground
//! agent to write belong to the user and must never be auto-curated.
//!
//! This module exposes a process/thread-local write origin that the agent loop
//! sets before each tool loop so tool handlers (e.g. skill_manage create) can
//! check whether they are executing inside the background-review fork.
//!
//! The Python original ([`tools/skill_provenance.py`]) used a
//! [`contextvars.ContextVar`]. Rust has no direct equivalent, so we model the
//! same set/reset/get contract with a thread-local plus a [`WriteOriginToken`]
//! that records the prior value to restore on `reset`.
//!
//! Usage mirrors the Python API:
//!
//! ```
//! use hermes_core::tool_skill_provenance::{
//!     set_current_write_origin, reset_current_write_origin,
//!     get_current_write_origin, BACKGROUND_REVIEW,
//! };
//!
//! let token = set_current_write_origin(BACKGROUND_REVIEW);
//! // ... tool runs here ...
//! assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);
//! reset_current_write_origin(token);
//! assert_eq!(get_current_write_origin(), "foreground");
//! ```

use std::cell::RefCell;

/// The default write origin: any tool call made by a regular (non-review)
/// agent, from the CLI, the gateway, cron, or a subagent.
pub const FOREGROUND: &str = "foreground";

/// The sentinel value the background review fork uses; mirrors run_agent.py's
/// `AIAgent._memory_write_origin` override in `_spawn_background_review()`.
/// Only skills created under this origin should be marked agent-created for
/// curator management.
pub const BACKGROUND_REVIEW: &str = "background_review";

thread_local! {
    static WRITE_ORIGIN: RefCell<String> = RefCell::new(FOREGROUND.to_string());
}

/// Opaque token returned by [`set_current_write_origin`]. The caller must pass
/// it back to [`reset_current_write_origin`] (ideally in a `finally`-style
/// scope guard) to restore the prior origin.
///
/// This mirrors `contextvars.Token`: it carries the value that was active
/// before the corresponding `set` call.
#[derive(Debug, Clone)]
pub struct WriteOriginToken {
    previous: String,
}

/// Bind the active write origin to the current context.
///
/// Returns a [`WriteOriginToken`] the caller must pass to
/// [`reset_current_write_origin`] in a finally/scope-exit path.
///
/// An empty `origin` falls back to [`FOREGROUND`], matching the Python
/// `origin or "foreground"` behavior.
pub fn set_current_write_origin(origin: &str) -> WriteOriginToken {
    let effective = if origin.is_empty() {
        FOREGROUND.to_string()
    } else {
        origin.to_string()
    };
    let previous = WRITE_ORIGIN.with(|cell| {
        let prev = cell.borrow().clone();
        *cell.borrow_mut() = effective;
        prev
    });
    WriteOriginToken { previous }
}

/// Restore the prior write origin context recorded in `token`.
pub fn reset_current_write_origin(token: WriteOriginToken) {
    WRITE_ORIGIN.with(|cell| {
        *cell.borrow_mut() = token.previous;
    });
}

/// Return the active write origin.
///
/// Default: `"foreground"` — any tool call made by a regular (non-review)
/// agent, from the CLI, the gateway, cron, or a subagent.
///
/// `"background_review"` — the self-improvement review fork; only skills
/// created under this origin should be marked agent-created for curator
/// management.
pub fn get_current_write_origin() -> String {
    WRITE_ORIGIN.with(|cell| cell.borrow().clone())
}

/// Convenience: `true` iff the current write origin is the background review
/// fork.
pub fn is_background_review() -> bool {
    get_current_write_origin() == BACKGROUND_REVIEW
}

/// RAII scope guard that sets the write origin on construction and restores
/// the prior value on drop. This is a Rust-idiomatic alternative to the
/// manual set/reset token dance and guarantees restoration even on panic.
///
/// ```
/// use hermes_core::tool_skill_provenance::{WriteOriginGuard, get_current_write_origin};
///
/// {
///     let _g = WriteOriginGuard::new("background_review");
///     assert_eq!(get_current_write_origin(), "background_review");
/// }
/// assert_eq!(get_current_write_origin(), "foreground");
/// ```
pub struct WriteOriginGuard {
    token: Option<WriteOriginToken>,
}

impl WriteOriginGuard {
    /// Set the active write origin and return a guard that restores the prior
    /// value when dropped.
    pub fn new(origin: &str) -> Self {
        WriteOriginGuard {
            token: Some(set_current_write_origin(origin)),
        }
    }
}

impl Drop for WriteOriginGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_write_origin(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each test runs on its own thread to isolate the thread-local. We also
    // explicitly reset to the default at the start to be defensive.
    fn reset_default() {
        let t = set_current_write_origin(FOREGROUND);
        let _ = t;
    }

    #[test]
    fn default_is_foreground() {
        reset_default();
        assert_eq!(get_current_write_origin(), FOREGROUND);
        assert!(!is_background_review());
    }

    #[test]
    fn set_and_get() {
        let token = set_current_write_origin(BACKGROUND_REVIEW);
        assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);
        assert!(is_background_review());
        reset_current_write_origin(token);
        assert_eq!(get_current_write_origin(), FOREGROUND);
        assert!(!is_background_review());
    }

    #[test]
    fn empty_falls_back_to_foreground() {
        let token = set_current_write_origin("");
        assert_eq!(get_current_write_origin(), FOREGROUND);
        reset_current_write_origin(token);
    }

    #[test]
    fn nested_set_reset_restores_prior() {
        reset_default();
        let outer = set_current_write_origin(BACKGROUND_REVIEW);
        assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);

        let inner = set_current_write_origin("custom_origin");
        assert_eq!(get_current_write_origin(), "custom_origin");

        // Resetting inner restores the background_review value, not default.
        reset_current_write_origin(inner);
        assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);

        reset_current_write_origin(outer);
        assert_eq!(get_current_write_origin(), FOREGROUND);
    }

    #[test]
    fn guard_restores_on_drop() {
        reset_default();
        {
            let _g = WriteOriginGuard::new(BACKGROUND_REVIEW);
            assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);
        }
        assert_eq!(get_current_write_origin(), FOREGROUND);
    }

    #[test]
    fn guard_restores_prior_non_default() {
        let outer = set_current_write_origin("layer_a");
        {
            let _g = WriteOriginGuard::new("layer_b");
            assert_eq!(get_current_write_origin(), "layer_b");
        }
        assert_eq!(get_current_write_origin(), "layer_a");
        reset_current_write_origin(outer);
    }

    #[test]
    fn thread_local_is_isolated_per_thread() {
        let _g = WriteOriginGuard::new(BACKGROUND_REVIEW);
        assert_eq!(get_current_write_origin(), BACKGROUND_REVIEW);

        let handle = std::thread::spawn(|| get_current_write_origin());
        // A fresh thread sees the default, not this thread's override.
        assert_eq!(handle.join().unwrap(), FOREGROUND);
    }
}
