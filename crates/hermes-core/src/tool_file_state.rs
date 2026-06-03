//! Cross-agent file state coordination.
//!
//! Prevents mangled edits when concurrent subagents (same process, same
//! filesystem) touch the same file. Complements the single-agent path-overlap
//! check elsewhere — this module catches the case where subagent B writes a
//! file that subagent A already read, so A's next write would overwrite B's
//! changes with stale content.
//!
//! # Design
//!
//! A process-wide singleton [`FileStateRegistry`] tracks, per resolved path:
//!
//!   * per-agent read stamps: `{task_id: {path: (mtime, read_ts, partial)}}`
//!   * last writer globally: `{path: (task_id, write_ts)}`
//!   * per-path lock for read→modify→write critical sections
//!
//! Three public hooks are used by the file tools:
//!
//!   * [`FileStateRegistry::record_read`] — called by read_file
//!   * [`FileStateRegistry::note_write`] — called after write_file / patch
//!   * [`FileStateRegistry::check_stale`] — called BEFORE write_file / patch
//!
//! Plus [`FileStateRegistry::lock_path`] — returns a per-path lock guard to
//! wrap the whole read→modify→write block. And
//! [`FileStateRegistry::writes_since`] for the subagent-completion reminder in
//! the delegate tool.
//!
//! All methods are no-ops when `HERMES_DISABLE_FILE_STATE_GUARD=1` is set.
//!
//! This module is intentionally separate from the per-task read tracker that
//! handles consecutive-read loop detection, which is a different concern.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Public stamp type: `(mtime, read_ts, partial)`.
///
/// `partial == true` when read_file returned a windowed view (offset > 1 or
/// limit < total_lines) — writes that happen after a partial read should still
/// warn so the model re-reads in full.
pub type ReadStamp = (f64, f64, bool);

/// Number of resolved-path entries retained per agent. Bounded to keep long
/// sessions from accumulating unbounded state. On overflow we drop the oldest
/// entries by insertion order.
const MAX_PATHS_PER_AGENT: usize = 4096;

/// Global last-writer map cap. Same policy.
const MAX_GLOBAL_WRITERS: usize = 4096;

/// An insertion-ordered map. Rust's `HashMap` does not preserve insertion
/// order (unlike Python's `dict` since 3.7), so we maintain an explicit order
/// vector to be able to drop the oldest entries on overflow.
#[derive(Default, Debug, Clone)]
struct OrderedMap<V> {
    map: HashMap<String, V>,
    order: Vec<String>,
}

impl<V> OrderedMap<V> {
    fn new() -> Self {
        OrderedMap {
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Insert or update a key. On first insert the key is appended to the
    /// insertion order; updates of existing keys keep their original position
    /// (matching Python `dict` semantics where re-assigning a key does not
    /// reorder it).
    fn insert(&mut self, key: String, value: V) {
        if !self.map.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.map.insert(key, value);
    }

    fn get(&self, key: &str) -> Option<&V> {
        self.map.get(key)
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn keys(&self) -> impl Iterator<Item = &String> {
        // Preserve insertion order, skipping any stale entries (should not
        // happen, but defensive).
        self.order.iter().filter(move |k| self.map.contains_key(*k))
    }

    fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.order
            .iter()
            .filter_map(move |k| self.map.get(k).map(|v| (k, v)))
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Trim to `limit` entries by dropping the oldest by insertion order.
    fn cap(&mut self, limit: usize) {
        let over = self.len().saturating_sub(limit);
        if over == 0 {
            return;
        }
        let mut removed = 0usize;
        let mut idx = 0usize;
        while removed < over && idx < self.order.len() {
            let key = self.order[idx].clone();
            if self.map.remove(&key).is_some() {
                removed += 1;
            }
            idx += 1;
        }
        // Drop the consumed prefix from the order vector.
        self.order.drain(0..idx);
    }
}

/// Guards `_reads` + `_last_writer`.
#[derive(Default)]
struct State {
    /// `{task_id: {path: ReadStamp}}`
    reads: HashMap<String, OrderedMap<ReadStamp>>,
    /// `{path: (task_id, write_ts)}`
    last_writer: OrderedMap<(String, f64)>,
}

/// Process-wide coordinator for cross-agent file edits.
pub struct FileStateRegistry {
    state: Mutex<State>,
    /// guards `path_locks`
    path_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl Default for FileStateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FileStateRegistry {
    pub fn new() -> Self {
        FileStateRegistry {
            state: Mutex::new(State::default()),
            path_locks: Mutex::new(HashMap::new()),
        }
    }

    // ── Path lock management ─────────────────────────────────────────
    fn lock_for(&self, resolved: &str) -> Arc<Mutex<()>> {
        let mut locks = self.path_locks.lock().unwrap();
        locks
            .entry(resolved.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Acquire the per-path lock for a read→modify→write section.
    ///
    /// Same process, same filesystem — threads on the same path serialize.
    /// Different paths proceed in parallel. The returned guard releases the
    /// lock when dropped.
    ///
    /// Note: callers must keep the returned guard (and the [`PathLock`] it
    /// borrows from) alive for the duration of the critical section.
    pub fn lock_path(&self, resolved: &str) -> PathLock {
        PathLock {
            lock: self.lock_for(resolved),
        }
    }

    // ── Read/write accounting ────────────────────────────────────────

    /// Record that `task_id` read `resolved`. No-op if the guard is disabled
    /// or the file's mtime cannot be read (and none was supplied).
    pub fn record_read(&self, task_id: &str, resolved: &str, partial: bool, mtime: Option<f64>) {
        if disabled() {
            return;
        }
        let mtime = match mtime.or_else(|| get_mtime(resolved)) {
            Some(m) => m,
            None => return,
        };
        let now = now_ts();
        let mut state = self.state.lock().unwrap();
        let agent_reads = state.reads.entry(task_id.to_string()).or_default();
        agent_reads.insert(resolved.to_string(), (mtime, now, partial));
        agent_reads.cap(MAX_PATHS_PER_AGENT);
    }

    /// Record a successful write.
    ///
    /// Updates the global last-writer map AND this agent's own read stamp (a
    /// write is an implicit read — the agent now knows the current content).
    pub fn note_write(&self, task_id: &str, resolved: &str, mtime: Option<f64>) {
        if disabled() {
            return;
        }
        let mtime = match mtime.or_else(|| get_mtime(resolved)) {
            Some(m) => m,
            None => return,
        };
        let now = now_ts();
        let mut state = self.state.lock().unwrap();
        state
            .last_writer
            .insert(resolved.to_string(), (task_id.to_string(), now));
        state.last_writer.cap(MAX_GLOBAL_WRITERS);
        // Writer's own view is now up-to-date.
        let agent_reads = state.reads.entry(task_id.to_string()).or_default();
        agent_reads.insert(resolved.to_string(), (mtime, now, false));
        agent_reads.cap(MAX_PATHS_PER_AGENT);
    }

    /// Return a model-facing warning if this write would be stale.
    ///
    /// Three staleness classes, in order of severity:
    ///
    ///   1. Sibling subagent wrote this file after this agent's last read.
    ///   2. External/unknown change (mtime differs from our last read).
    ///   3. Agent never read the file (write-without-read).
    ///
    /// Returns `None` when the write is safe. Does not error — callers decide
    /// whether to block or warn.
    pub fn check_stale(&self, task_id: &str, resolved: &str) -> Option<String> {
        if disabled() {
            return None;
        }

        let (stamp, last_writer) = {
            let state = self.state.lock().unwrap();
            let stamp = state
                .reads
                .get(task_id)
                .and_then(|m| m.get(resolved))
                .copied();
            let last_writer = state.last_writer.get(resolved).cloned();
            (stamp, last_writer)
        };

        // Case 3: never read AND we have no write record — net-new file or
        // first touch by this agent. Let existing sensitive-path and
        // file-exists logic handle it; nothing to warn about here.
        if stamp.is_none() && last_writer.is_none() {
            return None;
        }

        let current_mtime = match get_mtime(resolved) {
            Some(m) => m,
            // File doesn't exist — write will create it; not stale.
            None => return None,
        };

        // Case 1: sibling subagent modified after our last read.
        if let Some((ref writer_tid, writer_ts)) = last_writer {
            if writer_tid != task_id {
                match stamp {
                    None => {
                        return Some(format!(
                            "{resolved} was modified by sibling subagent \
                             {writer_tid:?} but this agent never read it. \
                             Read the file before writing to avoid overwriting \
                             the sibling's changes."
                        ));
                    }
                    Some((_, read_ts, _)) => {
                        if writer_ts > read_ts {
                            return Some(format!(
                                "{resolved} was modified by sibling subagent \
                                 {writer_tid:?} at {} — after this agent's last \
                                 read at {}. Re-read the file before writing.",
                                fmt_ts(writer_ts),
                                fmt_ts(read_ts)
                            ));
                        }
                    }
                }
            }
        }

        // Case 2: external / unknown modification (mtime drifted).
        if let Some((read_mtime, _read_ts, partial)) = stamp {
            if current_mtime != read_mtime {
                return Some(format!(
                    "{resolved} was modified since you last read it on disk \
                     (external edit or unrecorded writer). Re-read the file \
                     before writing."
                ));
            }
            if partial {
                return Some(format!(
                    "{resolved} was last read with offset/limit pagination \
                     (partial view). Re-read the whole file before overwriting \
                     it."
                ));
            }
        }

        // Case 3b: agent truly never read the file.
        if stamp.is_none() {
            return Some(format!(
                "{resolved} was not read by this agent. Read the file first so \
                 you can write an informed edit."
            ));
        }

        None
    }

    // ── Reminder helper for delegate tool ────────────────────────────

    /// Return `{writer_task_id: [paths]}` for writes done after `since_ts` by
    /// agents OTHER than `exclude_task_id`.
    ///
    /// Used by delegate_task to append a "subagent modified files the parent
    /// previously read" reminder to the delegation result.
    pub fn writes_since<I, S>(
        &self,
        exclude_task_id: &str,
        since_ts: f64,
        paths: I,
    ) -> HashMap<String, Vec<String>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if disabled() {
            return HashMap::new();
        }
        let paths_set: std::collections::HashSet<String> =
            paths.into_iter().map(|p| p.as_ref().to_string()).collect();
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        let state = self.state.lock().unwrap();
        for (p, (writer_tid, ts)) in state.last_writer.iter() {
            if writer_tid == exclude_task_id {
                continue;
            }
            if *ts < since_ts {
                continue;
            }
            if paths_set.contains(p) {
                out.entry(writer_tid.clone()).or_default().push(p.clone());
            }
        }
        out
    }

    /// Return the list of resolved paths this agent has read (insertion order).
    pub fn known_reads(&self, task_id: &str) -> Vec<String> {
        if disabled() {
            return Vec::new();
        }
        let state = self.state.lock().unwrap();
        state
            .reads
            .get(task_id)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    // ── Testing hooks ────────────────────────────────────────────────

    /// Reset all state. Intended for tests only.
    pub fn clear(&self) {
        {
            let mut state = self.state.lock().unwrap();
            state.reads.clear();
            state.last_writer.clear();
        }
        {
            let mut locks = self.path_locks.lock().unwrap();
            locks.clear();
        }
    }
}

/// RAII holder for a per-path lock. While alive, [`PathLock::acquire`] yields a
/// guard that serializes same-path critical sections.
pub struct PathLock {
    lock: Arc<Mutex<()>>,
}

impl PathLock {
    /// Acquire the underlying lock, returning a guard held for the critical
    /// section. Equivalent to entering the Python `with lock_path(...)` block.
    pub fn acquire(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().unwrap()
    }
}

// ── Module-level singleton + helpers ─────────────────────────────────

static REGISTRY: OnceLock<FileStateRegistry> = OnceLock::new();

/// Return the process-wide registry singleton.
pub fn get_registry() -> &'static FileStateRegistry {
    REGISTRY.get_or_init(FileStateRegistry::new)
}

/// Re-read each call so tests can toggle via the environment variable.
fn disabled() -> bool {
    std::env::var("HERMES_DISABLE_FILE_STATE_GUARD")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

/// Short relative wall-clock for error messages.
fn fmt_ts(ts: f64) -> String {
    // ts is a unix epoch seconds (UTC). Render local %H:%M:%S to match the
    // Python time.strftime("%H:%M:%S", time.localtime(ts)) behaviour.
    use chrono::{Local, TimeZone};
    let secs = ts.trunc() as i64;
    let nanos = ((ts - ts.trunc()) * 1_000_000_000.0).round() as u32;
    match Local.timestamp_opt(secs, nanos) {
        chrono::LocalResult::Single(dt) => dt.format("%H:%M:%S").to_string(),
        _ => Local
            .timestamp_opt(secs, 0)
            .single()
            .map(|dt| dt.format("%H:%M:%S").to_string())
            .unwrap_or_default(),
    }
}

/// Current wall-clock time as unix epoch seconds (float), matching
/// Python's `time.time()`.
fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// File modification time as unix epoch seconds (float). Returns `None` on any
/// filesystem error (missing file, permission, etc.) — matching Python's
/// `os.path.getmtime` raising `OSError`.
fn get_mtime(path: &str) -> Option<f64> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    modified
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .ok()
}

// ── Convenience wrappers (short names used at call sites) ─────────────

/// Convenience wrapper around [`FileStateRegistry::record_read`] on the
/// singleton.
pub fn record_read(task_id: &str, resolved_or_path: &str, partial: bool) {
    get_registry().record_read(task_id, resolved_or_path, partial, None);
}

/// Convenience wrapper around [`FileStateRegistry::note_write`] on the
/// singleton.
pub fn note_write(task_id: &str, resolved_or_path: &str) {
    get_registry().note_write(task_id, resolved_or_path, None);
}

/// Convenience wrapper around [`FileStateRegistry::check_stale`] on the
/// singleton.
pub fn check_stale(task_id: &str, resolved_or_path: &str) -> Option<String> {
    get_registry().check_stale(task_id, resolved_or_path)
}

/// Convenience wrapper around [`FileStateRegistry::lock_path`] on the
/// singleton.
pub fn lock_path(resolved_or_path: &str) -> PathLock {
    get_registry().lock_path(resolved_or_path)
}

/// Convenience wrapper around [`FileStateRegistry::writes_since`] on the
/// singleton.
pub fn writes_since<I, S>(
    exclude_task_id: &str,
    since_ts: f64,
    paths: I,
) -> HashMap<String, Vec<String>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    get_registry().writes_since(exclude_task_id, since_ts, paths)
}

/// Convenience wrapper around [`FileStateRegistry::known_reads`] on the
/// singleton.
pub fn known_reads(task_id: &str) -> Vec<String> {
    get_registry().known_reads(task_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Serialize tests that touch the shared env var so they don't race.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn temp_file(contents: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "hermes_fs_test_{}_{}.txt",
            std::process::id(),
            now_ts().to_bits()
        );
        p.push(unique);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn ordered_map_cap_drops_oldest() {
        let mut m: OrderedMap<i32> = OrderedMap::new();
        m.insert("a".into(), 1);
        m.insert("b".into(), 2);
        m.insert("c".into(), 3);
        m.cap(2);
        assert_eq!(m.len(), 2);
        assert!(m.get("a").is_none());
        assert!(m.get("b").is_some());
        assert!(m.get("c").is_some());
        let keys: Vec<_> = m.keys().cloned().collect();
        assert_eq!(keys, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn ordered_map_update_keeps_position() {
        let mut m: OrderedMap<i32> = OrderedMap::new();
        m.insert("a".into(), 1);
        m.insert("b".into(), 2);
        m.insert("a".into(), 99); // update — should not reorder
        let keys: Vec<_> = m.keys().cloned().collect();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(*m.get("a").unwrap(), 99);
    }

    #[test]
    fn net_new_file_no_warning() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        // Never read, no writer: case 3 returns None.
        assert!(reg.check_stale("t1", "/nonexistent/path/abc.txt").is_none());
    }

    #[test]
    fn write_without_read_warns() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        // Another agent wrote, but this agent never read it.
        reg.note_write("other", path, None);
        let warn = reg.check_stale("me", path);
        assert!(warn.is_some(), "expected sibling-write warning");
        assert!(warn.unwrap().contains("sibling subagent"));
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn read_then_write_no_warning_for_same_agent() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        reg.record_read("me", path, false, None);
        // Same agent, no external change → safe.
        assert!(reg.check_stale("me", path).is_none());
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn sibling_write_after_read_warns() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        // me reads with an explicit early read timestamp.
        reg.record_read("me", path, false, Some(get_mtime(path).unwrap()));
        // Force the stored read_ts to be in the past so the sibling write is
        // unambiguously "after". We re-record with a manual read by directly
        // simulating via note_write from the sibling after a small sleep.
        std::thread::sleep(std::time::Duration::from_millis(5));
        reg.note_write("sibling", path, None);
        let warn = reg.check_stale("me", path);
        assert!(warn.is_some(), "expected post-read sibling warning");
        assert!(warn.unwrap().contains("sibling subagent"));
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn partial_read_warns() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        reg.record_read("me", path, true, None);
        let warn = reg.check_stale("me", path);
        assert!(warn.is_some());
        assert!(warn.unwrap().contains("partial view"));
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn external_mtime_drift_warns() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        // Record a read with a stale mtime that won't match disk.
        reg.record_read("me", path, false, Some(1.0));
        let warn = reg.check_stale("me", path);
        assert!(warn.is_some());
        assert!(warn.unwrap().contains("modified since you last read"));
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn writes_since_filters_by_time_and_agent() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap().to_string();
        let before = now_ts();
        std::thread::sleep(std::time::Duration::from_millis(5));
        reg.note_write("sibling", &path, None);
        let res = reg.writes_since("parent", before, [path.clone()]);
        assert_eq!(res.get("sibling").map(|v| v.len()), Some(1));
        // Excluding the writer yields nothing.
        let res2 = reg.writes_since("sibling", before, [path.clone()]);
        assert!(res2.is_empty());
        // since_ts in the future yields nothing.
        let res3 = reg.writes_since("parent", now_ts() + 100.0, [path.clone()]);
        assert!(res3.is_empty());
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn known_reads_lists_paths() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        reg.record_read("me", path, false, None);
        assert_eq!(reg.known_reads("me"), vec![path.to_string()]);
        assert!(reg.known_reads("other").is_empty());
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn disabled_guard_is_noop() {
        let _g = env_lock();
        unsafe { std::env::set_var("HERMES_DISABLE_FILE_STATE_GUARD", "1"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        reg.note_write("other", path, None);
        reg.record_read("me", path, true, None);
        assert!(reg.check_stale("me", path).is_none());
        assert!(reg.known_reads("me").is_empty());
        assert!(reg.writes_since("x", 0.0, [path]).is_empty());
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn lock_path_serializes_same_path() {
        let _g = env_lock();
        let reg = FileStateRegistry::new();
        let l1 = reg.lock_path("/some/path");
        let _guard = l1.acquire();
        // Different path lock is independent and acquirable.
        let l2 = reg.lock_path("/other/path");
        let _guard2 = l2.acquire();
    }

    #[test]
    fn clear_resets_state() {
        let _g = env_lock();
        unsafe { std::env::remove_var("HERMES_DISABLE_FILE_STATE_GUARD"); }
        let reg = FileStateRegistry::new();
        let f = temp_file("hello");
        let path = f.to_str().unwrap();
        reg.record_read("me", path, false, None);
        reg.clear();
        assert!(reg.known_reads("me").is_empty());
        let _ = std::fs::remove_file(&f);
    }
}
