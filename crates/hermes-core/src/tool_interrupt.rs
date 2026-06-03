//! Per-thread interrupt signaling for all tools.
//!
//! Provides thread-scoped interrupt tracking so that interrupting one agent
//! session does not kill tools running in other sessions. This is critical
//! in the gateway where multiple agents run concurrently in the same process.
//!
//! The agent stores its execution thread ID at the start of `run_conversation()`
//! and passes it to [`set_interrupt`] / clear. Tools call [`is_interrupted`]
//! which checks the CURRENT thread — no argument needed.
//!
//! Usage in tools:
//! ```ignore
//! use hermes_core::tool_interrupt::is_interrupted;
//! if is_interrupted() {
//!     // return interrupted result with returncode 130
//! }
//! ```
//!
//! Port of the Python module `tools/interrupt.py`.
//!
//! ## Thread identity
//!
//! The Python original uses `threading.get_ident()` integer thread idents and
//! lets callers pass an explicit target ident. Rust's [`std::thread::ThreadId`]
//! is not convertible to/from a stable integer on stable Rust, so we expose a
//! [`Tid`] newtype around `ThreadId` to identify threads. [`current_tid`]
//! returns the identifier for the calling thread; callers that previously
//! captured a thread ident should capture a `Tid` instead.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::thread::ThreadId;

/// Opaque, hashable identifier for a thread.
///
/// Mirrors the Python `int` thread ident: it can be captured on one thread
/// (e.g. at the start of `run_conversation`) and later passed back into
/// [`set_interrupt`] to target that specific thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tid(pub ThreadId);

impl From<ThreadId> for Tid {
    fn from(id: ThreadId) -> Self {
        Tid(id)
    }
}

/// Returns the [`Tid`] of the calling thread.
///
/// Equivalent to `threading.current_thread().ident` in the Python original.
pub fn current_tid() -> Tid {
    Tid(std::thread::current().id())
}

/// Opt-in debug tracing — pairs with `HERMES_DEBUG_INTERRUPT` in
/// `tools/environments/base.py`. Enables per-call logging of set/check so the
/// caller thread, target thread, and current state are visible when
/// diagnosing "interrupt signaled but tool never saw it" reports.
fn debug_interrupt() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("HERMES_DEBUG_INTERRUPT").is_some())
}

/// Set of thread idents that have been interrupted.
fn interrupted_threads() -> &'static Mutex<HashSet<Tid>> {
    static THREADS: OnceLock<Mutex<HashSet<Tid>>> = OnceLock::new();
    THREADS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Set or clear interrupt for a specific thread.
///
/// * `active` — `true` to signal interrupt, `false` to clear it.
/// * `thread_id` — target thread. When `None`, targets the current thread
///   (backward compat for CLI/tests).
pub fn set_interrupt(active: bool, thread_id: Option<Tid>) {
    let tid = thread_id.unwrap_or_else(current_tid);
    let snapshot = {
        let mut set = interrupted_threads().lock().unwrap_or_else(|e| e.into_inner());
        if active {
            set.insert(tid);
        } else {
            set.remove(&tid);
        }
        if debug_interrupt() {
            Some(set.clone())
        } else {
            None
        }
    };
    if debug_interrupt() {
        log::info!(
            "[interrupt-debug] set_interrupt(active={}, target_tid={:?}) \
             called_from_tid={:?} current_set={:?}",
            active,
            tid,
            current_tid(),
            snapshot,
        );
    }
}

/// Check if an interrupt has been requested for the current thread.
///
/// Safe to call from any thread — each thread only sees its own interrupt
/// state.
pub fn is_interrupted() -> bool {
    let tid = current_tid();
    let set = interrupted_threads().lock().unwrap_or_else(|e| e.into_inner());
    set.contains(&tid)
}

// ---------------------------------------------------------------------------
// Backward-compatible _interrupt_event proxy
// ---------------------------------------------------------------------------
// Some legacy call sites (code_execution_tool, process_registry, tests)
// import _interrupt_event directly and call .is_set() / .set() / .clear().
// This shim maps those calls to the per-thread functions above so existing
// code keeps working while the underlying mechanism is thread-scoped.

/// Drop-in proxy that maps `threading.Event` methods to per-thread state.
///
/// Mirrors the Python `_ThreadAwareEventProxy`. Use [`interrupt_event`] to
/// obtain the shared instance equivalent to the module-level
/// `_interrupt_event`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadAwareEventProxy;

impl ThreadAwareEventProxy {
    /// Equivalent to `threading.Event.is_set()`.
    pub fn is_set(&self) -> bool {
        is_interrupted()
    }

    /// Equivalent to `threading.Event.set()` — signals interrupt on the
    /// current thread.
    pub fn set(&self) {
        set_interrupt(true, None);
    }

    /// Equivalent to `threading.Event.clear()` — clears interrupt on the
    /// current thread.
    pub fn clear(&self) {
        set_interrupt(false, None);
    }

    /// Not truly supported — returns current state immediately (matching the
    /// Python proxy, which ignores `timeout`).
    pub fn wait(&self, _timeout: Option<f64>) -> bool {
        self.is_set()
    }
}

/// The shared, thread-aware interrupt event proxy.
///
/// Equivalent to the module-level `_interrupt_event` singleton in Python.
pub fn interrupt_event() -> ThreadAwareEventProxy {
    ThreadAwareEventProxy
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    // Tests in this module mutate global per-thread interrupt state. They run
    // on distinct threads (the test harness uses multiple threads), but to
    // keep them deterministic we always clean up the current thread's state.

    #[test]
    fn current_thread_set_and_clear() {
        set_interrupt(false, None);
        assert!(!is_interrupted());

        set_interrupt(true, None);
        assert!(is_interrupted());

        set_interrupt(false, None);
        assert!(!is_interrupted());
    }

    #[test]
    fn interrupt_is_thread_scoped() {
        // Clear current thread.
        set_interrupt(false, None);

        // A different thread interrupting itself must NOT affect us.
        let handle = std::thread::spawn(|| {
            set_interrupt(true, None);
            assert!(is_interrupted());
            // leave the spawned thread's state set; it dies here.
        });
        handle.join().unwrap();

        assert!(!is_interrupted());
    }

    #[test]
    fn target_specific_thread_by_tid() {
        let (tx, rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        let handle = std::thread::spawn(move || {
            // Report our tid, then wait until the controller signals us.
            tx.send(current_tid()).unwrap();
            // Block until main thread has set our interrupt.
            done_rx.recv().unwrap();
            assert!(is_interrupted());
        });

        let worker_tid = rx.recv().unwrap();

        // Setting the worker's interrupt from this (main) thread must not
        // interrupt the main thread.
        set_interrupt(false, None);
        set_interrupt(true, Some(worker_tid));
        assert!(!is_interrupted());

        done_tx.send(()).unwrap();
        handle.join().unwrap();

        // Clean up worker's state (worker is gone, but tidy the global set).
        set_interrupt(false, Some(worker_tid));
    }

    #[test]
    fn event_proxy_maps_to_per_thread_state() {
        let ev = interrupt_event();
        ev.clear();
        assert!(!ev.is_set());
        assert!(!is_interrupted());

        ev.set();
        assert!(ev.is_set());
        assert!(is_interrupted());

        // wait() returns current state immediately and ignores timeout.
        assert!(ev.wait(Some(5.0)));
        assert!(ev.wait(None));

        ev.clear();
        assert!(!ev.is_set());
        assert!(!is_interrupted());
    }

    #[test]
    fn current_tid_is_stable_within_thread() {
        assert_eq!(current_tid(), current_tid());
    }
}
