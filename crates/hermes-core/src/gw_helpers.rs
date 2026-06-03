//! Shared helper utilities for gateway platform adapters.
//!
//! Native Rust port of `gateway/platforms/helpers.py`. Extracts common
//! patterns duplicated across 5-7 adapters:
//! - message deduplication ([`MessageDeduplicator`]),
//! - markdown stripping ([`strip_markdown`]),
//! - thread participation tracking ([`ThreadParticipationTracker`]),
//! - phone number redaction ([`redact_phone`]).
//!
//! The Python `TextBatchAggregator` is intentionally NOT ported here: it is
//! built on `asyncio.Task` scheduling tightly coupled to the Python event loop
//! and the dynamic `MessageEvent` type. Its batching policy (the delay
//! selection) is exposed as a pure function [`batch_delay_for`] so callers can
//! reuse the timing logic against whatever async runtime they wire it into.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::atomic_json_write;

/// Returns the current Unix time in seconds as an `f64`, mirroring Python's
/// `time.time()`.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ─── Message Deduplication ──────────────────────────────────────────────────

/// TTL-based message deduplication cache.
///
/// Faithful port of the `MessageDeduplicator` class. Replaces the identical
/// `_seen_messages` / `_is_duplicate()` pattern previously duplicated in
/// discord, slack, dingtalk, wecom, weixin, mattermost, and feishu adapters.
///
/// ```
/// # use hermes_core::gw_helpers::MessageDeduplicator;
/// let mut dedup = MessageDeduplicator::new();
/// assert!(!dedup.is_duplicate("abc"));
/// assert!(dedup.is_duplicate("abc"));
/// ```
#[derive(Debug, Clone)]
pub struct MessageDeduplicator {
    seen: HashMap<String, f64>,
    max_size: usize,
    ttl: f64,
}

impl Default for MessageDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageDeduplicator {
    /// Default cache: 2000 entries, 300-second TTL (matches Python defaults).
    pub fn new() -> Self {
        Self::with_config(2000, 300.0)
    }

    /// Construct with explicit `max_size` and `ttl_seconds`.
    pub fn with_config(max_size: usize, ttl_seconds: f64) -> Self {
        Self {
            seen: HashMap::new(),
            max_size,
            ttl: ttl_seconds,
        }
    }

    /// Return `true` if `msg_id` was already seen within the TTL window.
    ///
    /// Empty ids are never duplicates (matches Python's `if not msg_id`).
    /// As a side-effect, records `msg_id` with the current timestamp when it
    /// is treated as new, and prunes the cache when it exceeds `max_size`.
    pub fn is_duplicate(&mut self, msg_id: &str) -> bool {
        if msg_id.is_empty() {
            return false;
        }
        let now = now_secs();
        if let Some(&ts) = self.seen.get(msg_id) {
            if now - ts < self.ttl {
                return true;
            }
            // Entry has expired — remove it and treat as new.
            self.seen.remove(msg_id);
        }
        self.seen.insert(msg_id.to_string(), now);

        if self.seen.len() > self.max_size {
            let cutoff = now - self.ttl;
            self.seen.retain(|_, &mut v| v > cutoff);
            if self.seen.len() > self.max_size {
                // TTL pruning alone does not cap the cache when every entry is
                // still fresh. Keep the newest entries so the helper's
                // max_size bound is enforced under sustained traffic.
                let mut items: Vec<(String, f64)> =
                    self.seen.drain().collect();
                // Sort ascending by timestamp, keep the newest `max_size`.
                items.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let start = items.len().saturating_sub(self.max_size);
                self.seen = items.into_iter().skip(start).collect();
            }
        }
        false
    }

    /// Clear all tracked messages.
    pub fn clear(&mut self) {
        self.seen.clear();
    }

    /// Current number of tracked entries (mainly for tests/diagnostics).
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

// ─── Text Batch Aggregation (timing policy only) ────────────────────────────

/// Default batch delay in seconds (Python `batch_delay=0.6`).
pub const DEFAULT_BATCH_DELAY: f64 = 0.6;
/// Default split delay in seconds (Python `split_delay=2.0`).
pub const DEFAULT_SPLIT_DELAY: f64 = 2.0;
/// Default split threshold in characters (Python `split_threshold=4000`).
pub const DEFAULT_SPLIT_THRESHOLD: usize = 4000;

/// Pure port of the delay-selection policy used by Python's
/// `TextBatchAggregator._flush`.
///
/// Returns `split_delay` when the last appended chunk looks like a split
/// message (its length is at least `split_threshold`); otherwise returns the
/// normal `batch_delay`. Callers wire this into their own async runtime.
pub fn batch_delay_for(
    last_chunk_len: usize,
    batch_delay: f64,
    split_delay: f64,
    split_threshold: usize,
) -> f64 {
    if last_chunk_len >= split_threshold {
        split_delay
    } else {
        batch_delay
    }
}

/// Return `true` if batching should be active (delay > 0), mirroring
/// `TextBatchAggregator.is_enabled`.
pub fn batching_enabled(batch_delay: f64) -> bool {
    batch_delay > 0.0
}

// ─── Markdown Stripping ─────────────────────────────────────────────────────

// Pre-compiled regexes for performance. `(?s)` enables DOTALL; `(?m)` enables
// MULTILINE, matching the Python `re.DOTALL` / `re.MULTILINE` flags.
static RE_BOLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\*\*(.+?)\*\*").unwrap());
static RE_ITALIC_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\*(.+?)\*").unwrap());
static RE_BOLD_UNDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)__(.+?)__").unwrap());
static RE_ITALIC_UNDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)_(.+?)_").unwrap());
static RE_CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"```[a-zA-Z0-9_+-]*\n?").unwrap());
static RE_INLINE_CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`(.+?)`").unwrap());
static RE_HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^#{1,6}\s+").unwrap());
static RE_LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap());
static RE_MULTI_NEWLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

/// Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.).
///
/// Faithful port of `strip_markdown`. Applies the same regex substitutions in
/// the same order and trims surrounding whitespace.
pub fn strip_markdown(text: &str) -> String {
    let mut out = RE_BOLD.replace_all(text, "$1").into_owned();
    out = RE_ITALIC_STAR.replace_all(&out, "$1").into_owned();
    out = RE_BOLD_UNDER.replace_all(&out, "$1").into_owned();
    out = RE_ITALIC_UNDER.replace_all(&out, "$1").into_owned();
    out = RE_CODE_BLOCK.replace_all(&out, "").into_owned();
    out = RE_INLINE_CODE.replace_all(&out, "$1").into_owned();
    out = RE_HEADING.replace_all(&out, "").into_owned();
    out = RE_LINK.replace_all(&out, "$1").into_owned();
    out = RE_MULTI_NEWLINE.replace_all(&out, "\n\n").into_owned();
    out.trim().to_string()
}

// ─── Thread Participation Tracking ──────────────────────────────────────────

/// Persistent tracking of threads the bot has participated in.
///
/// Faithful port of `ThreadParticipationTracker`. Replaces the identical
/// `_load`/`_save_participated_threads` + `_mark_thread_participated` pattern
/// previously duplicated in discord.py and matrix.py.
///
/// Insertion order is preserved (matching Python `dict` semantics) so that the
/// "keep the newest" pruning on save retains the most recently added threads.
#[derive(Debug, Clone)]
pub struct ThreadParticipationTracker {
    platform: String,
    max_tracked: usize,
    /// Ordered list of tracked thread ids (oldest first).
    threads: Vec<String>,
}

/// Default cap on tracked threads (matches Python `_MAX_TRACKED = 500`).
pub const DEFAULT_MAX_TRACKED: usize = 500;

impl ThreadParticipationTracker {
    /// Construct a tracker for `platform_name`, loading any persisted state
    /// from `<hermes_home>/<platform>_threads.json`. Uses the default cap of
    /// 500 tracked threads.
    pub fn new(platform_name: impl Into<String>) -> Self {
        Self::with_max(platform_name, DEFAULT_MAX_TRACKED)
    }

    /// Construct with an explicit `max_tracked` cap.
    pub fn with_max(platform_name: impl Into<String>, max_tracked: usize) -> Self {
        let platform = platform_name.into();
        let mut tracker = Self {
            platform,
            max_tracked,
            threads: Vec::new(),
        };
        tracker.threads = tracker.load();
        tracker
    }

    /// Path to the persistent state file for this platform.
    pub fn state_path(&self) -> PathBuf {
        get_hermes_home().join(format!("{}_threads.json", self.platform))
    }

    fn load(&self) -> Vec<String> {
        let path = self.state_path();
        if path.exists() {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(arr) = value.as_array() {
                        return arr
                            .iter()
                            .map(|v| match v {
                                // Mirror Python `str(thread_id)`: strings keep
                                // their value, other JSON scalars stringify.
                                serde_json::Value::String(s) => s.clone(),
                                other => json_scalar_to_str(other),
                            })
                            .collect();
                    }
                }
            }
        }
        Vec::new()
    }

    fn save(&mut self) {
        let path = self.state_path();
        if self.threads.len() > self.max_tracked {
            let start = self.threads.len() - self.max_tracked;
            self.threads = self.threads.split_off(start);
        }
        let data = serde_json::Value::Array(
            self.threads
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        );
        // Python passes `indent=None` (compact); the Rust port maps indent 0
        // to compact output.
        if let Err(e) = atomic_json_write(&path, &data, 0) {
            log::warn!(
                "[ThreadParticipationTracker] failed to persist {}: {}",
                path.display(),
                e
            );
        }
    }

    /// Mark `thread_id` as participated and persist. No-op (no write) if the
    /// thread is already tracked.
    pub fn mark(&mut self, thread_id: impl Into<String>) {
        let thread_id = thread_id.into();
        if !self.threads.iter().any(|t| t == &thread_id) {
            self.threads.push(thread_id);
            self.save();
        }
    }

    /// Return `true` if `thread_id` is currently tracked (Python `__contains__`).
    pub fn contains(&self, thread_id: &str) -> bool {
        self.threads.iter().any(|t| t == thread_id)
    }

    /// Clear all tracked threads (in memory only, matching Python `clear`,
    /// which does not persist).
    pub fn clear(&mut self) {
        self.threads.clear();
    }

    /// Number of tracked threads.
    pub fn len(&self) -> usize {
        self.threads.len()
    }

    /// Whether no threads are tracked.
    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }
}

/// Stringify a non-string JSON scalar the way Python's `str()` would for the
/// values typically found in these id lists (numbers, bools, null).
fn json_scalar_to_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ─── Phone Number Redaction ─────────────────────────────────────────────────

/// Redact a phone number for logging, preserving the leading digits and the
/// last digits.
///
/// Faithful port of `redact_phone`. The Python implementation slices by code
/// point; this version operates on Unicode scalar values (`char`s) so multi-
/// byte phone strings behave identically.
pub fn redact_phone(phone: &str) -> String {
    if phone.is_empty() {
        return "<none>".to_string();
    }
    let chars: Vec<char> = phone.chars().collect();
    let len = chars.len();
    if len <= 8 {
        if len > 4 {
            // phone[:2] + "****" + phone[-2:]
            let first: String = chars[..2].iter().collect();
            let last: String = chars[len - 2..].iter().collect();
            format!("{}****{}", first, last)
        } else {
            "****".to_string()
        }
    } else {
        // phone[:4] + "****" + phone[-4:]
        let first: String = chars[..4].iter().collect();
        let last: String = chars[len - 4..].iter().collect();
        format!("{}****{}", first, last)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_empty_never_duplicate() {
        let mut d = MessageDeduplicator::new();
        assert!(!d.is_duplicate(""));
        assert!(!d.is_duplicate(""));
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn dedup_detects_repeat() {
        let mut d = MessageDeduplicator::new();
        assert!(!d.is_duplicate("m1"));
        assert!(d.is_duplicate("m1"));
        assert!(d.is_duplicate("m1"));
        assert!(!d.is_duplicate("m2"));
    }

    #[test]
    fn dedup_expired_entry_is_new() {
        // TTL of 0 means nothing stays "fresh": now - ts < 0 is always false.
        let mut d = MessageDeduplicator::with_config(2000, 0.0);
        assert!(!d.is_duplicate("x"));
        // Second call: entry exists but now-ts (>=0) is not < 0, so expired.
        assert!(!d.is_duplicate("x"));
    }

    #[test]
    fn dedup_clear() {
        let mut d = MessageDeduplicator::new();
        d.is_duplicate("a");
        d.is_duplicate("b");
        assert_eq!(d.len(), 2);
        d.clear();
        assert_eq!(d.len(), 0);
        assert!(!d.is_duplicate("a"));
    }

    #[test]
    fn dedup_max_size_bound_under_fresh_traffic() {
        // All entries fresh (long TTL); cache must stay capped at max_size.
        let mut d = MessageDeduplicator::with_config(10, 10_000.0);
        for i in 0..100 {
            assert!(!d.is_duplicate(&format!("id-{i}")));
        }
        assert!(d.len() <= 10);
    }

    #[test]
    fn strip_markdown_bold_italic() {
        assert_eq!(strip_markdown("**bold**"), "bold");
        assert_eq!(strip_markdown("*italic*"), "italic");
        assert_eq!(strip_markdown("__bold__"), "bold");
        assert_eq!(strip_markdown("_italic_"), "italic");
    }

    #[test]
    fn strip_markdown_code_and_links() {
        assert_eq!(strip_markdown("`code`"), "code");
        assert_eq!(strip_markdown("[label](http://x.com)"), "label");
        assert_eq!(strip_markdown("```python\nx = 1\n```"), "x = 1");
    }

    #[test]
    fn strip_markdown_heading_and_newlines() {
        assert_eq!(strip_markdown("# Title"), "Title");
        assert_eq!(strip_markdown("### Sub"), "Sub");
        assert_eq!(strip_markdown("a\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn strip_markdown_trims() {
        assert_eq!(strip_markdown("   hello   "), "hello");
    }

    #[test]
    fn redact_phone_cases() {
        assert_eq!(redact_phone(""), "<none>");
        // len <= 4 -> "****"
        assert_eq!(redact_phone("1234"), "****");
        assert_eq!(redact_phone("12"), "****");
        // 4 < len <= 8 -> first2 + **** + last2
        assert_eq!(redact_phone("12345"), "12****45");
        assert_eq!(redact_phone("12345678"), "12****78");
        // len > 8 -> first4 + **** + last4
        assert_eq!(redact_phone("+1234567890"), "+123****7890");
    }

    #[test]
    fn batch_delay_policy() {
        assert_eq!(
            batch_delay_for(5000, 0.6, 2.0, 4000),
            2.0
        );
        assert_eq!(
            batch_delay_for(100, 0.6, 2.0, 4000),
            0.6
        );
        assert!(batching_enabled(0.6));
        assert!(!batching_enabled(0.0));
    }

    #[test]
    fn thread_tracker_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "gw_helpers_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        let mut t = ThreadParticipationTracker::new("testplat");
        assert!(!t.contains("a"));
        t.mark("a");
        t.mark("b");
        t.mark("a"); // duplicate, no growth
        assert!(t.contains("a"));
        assert!(t.contains("b"));
        assert_eq!(t.len(), 2);

        // Reload from disk picks up persisted state.
        let t2 = ThreadParticipationTracker::new("testplat");
        assert!(t2.contains("a"));
        assert!(t2.contains("b"));
        assert_eq!(t2.len(), 2);

        unsafe { std::env::remove_var("HERMES_HOME"); }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn thread_tracker_prunes_to_max() {
        let dir = std::env::temp_dir().join(format!(
            "gw_helpers_prune_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        let mut t = ThreadParticipationTracker::with_max("prunep", 3);
        for i in 0..6 {
            t.mark(format!("t{i}"));
        }
        // After save-time pruning, only the newest 3 remain.
        let reloaded = ThreadParticipationTracker::with_max("prunep", 3);
        assert_eq!(reloaded.len(), 3);
        assert!(reloaded.contains("t5"));
        assert!(reloaded.contains("t4"));
        assert!(reloaded.contains("t3"));
        assert!(!reloaded.contains("t0"));

        unsafe { std::env::remove_var("HERMES_HOME"); }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
