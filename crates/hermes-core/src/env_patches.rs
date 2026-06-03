//! Monkey patches for making hermes-agent tools work inside async frameworks (Atropos).
//!
//! # Problem
//! Some tools use `asyncio.run()` internally (e.g., Modal backend via SWE-ReX,
//! `web_extract`). This crashes when called from inside Atropos's event loop because
//! `asyncio.run()` can't be nested.
//!
//! # Solution
//! The Modal environment now uses a dedicated `_AsyncWorker` thread internally,
//! making it safe for both CLI and Atropos use. No monkey-patching is required.
//!
//! This module is kept for backward compatibility. [`apply_patches`] is a no-op.
//!
//! # Usage
//! Call [`apply_patches`] once at startup. This is idempotent and safe to call
//! multiple times.
//!
//! This is a direct port of `environments/patches.py`.

use std::sync::atomic::{AtomicBool, Ordering};

/// Tracks whether the (no-op) patches have already been applied.
///
/// Mirrors the Python module-level `_patches_applied` flag.
static PATCHES_APPLIED: AtomicBool = AtomicBool::new(false);

/// Apply all monkey patches needed for Atropos compatibility.
///
/// This is a no-op kept for backward compatibility: async safety is now built
/// into the Modal environment itself. The call is idempotent — subsequent calls
/// after the first return immediately.
///
/// Returns `true` if this call performed the (no-op) application, `false` if the
/// patches had already been applied by an earlier call.
pub fn apply_patches() -> bool {
    // Equivalent to the Python early-return on `_patches_applied`.
    // `swap` atomically reads the prior value and sets the flag, so concurrent
    // callers race-free: exactly one observes `false`.
    if PATCHES_APPLIED.swap(true, Ordering::SeqCst) {
        return false;
    }

    log::debug!("apply_patches() called; no patches needed (async safety is built-in)");
    true
}

/// Whether [`apply_patches`] has been invoked at least once.
///
/// Exposed for tests and for callers that want to inspect application state.
pub fn patches_applied() -> bool {
    PATCHES_APPLIED.load(Ordering::SeqCst)
}

/// Reset the applied flag back to its initial state.
///
/// Primarily useful for tests that need to exercise the first-call path
/// repeatedly. There is no equivalent in the Python source, where the flag is a
/// process-lifetime global.
pub fn reset_patches_for_test() {
    PATCHES_APPLIED.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    // Serialize tests because they share the process-global atomic flag.
    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn first_call_applies_and_is_idempotent() {
        let _g = test_lock().lock().unwrap();
        reset_patches_for_test();

        assert!(!patches_applied(), "flag should start cleared");

        // First call performs the no-op application.
        assert!(apply_patches());
        assert!(patches_applied());

        // Subsequent calls are no-ops and report no fresh application.
        assert!(!apply_patches());
        assert!(!apply_patches());
        assert!(patches_applied());
    }

    #[test]
    fn reset_returns_to_unapplied() {
        let _g = test_lock().lock().unwrap();
        reset_patches_for_test();

        assert!(apply_patches());
        reset_patches_for_test();
        assert!(!patches_applied());

        // After reset, the first call applies again.
        assert!(apply_patches());
    }
}
