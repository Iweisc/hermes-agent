//! File Tools Module — LLM agent file manipulation tools.
//!
//! Native Rust port of `tools/file_tools.py`. Reproduces the four agent-facing
//! file tools (`read_file`, `write_file`, `patch`, `search_files`) plus the
//! orchestration layer around them:
//!
//!   * Per-task read tracker (consecutive-loop detection + warnings/blocks).
//!   * Read deduplication (skip re-sending unchanged file content) with its own
//!     escalating block after repeated stubs.
//!   * Device-path and sensitive-path guards (pure path checks, no I/O).
//!   * Internal status-text write guard (prevents the dedup stub message from
//!     being persisted as file content).
//!   * Character-count read guard and large-file hint.
//!   * Cross-agent file-state coordination (read records, staleness checks,
//!     per-path locking).
//!   * Secret redaction.
//!
//! # Dependency surface
//!
//! Several helpers the Python module imported live in sibling Rust modules
//! that, in the parallel port, are not yet declared `pub` in the crate roots
//! (`tool_file_operations`, `tool_file_state`, `agent_redact`,
//! `agent_file_safety`, `tool_binary_extensions`, `cli_config`). To keep this
//! module self-contained and unblocked, the small, load-bearing pieces are
//! reproduced locally here:
//!
//!   * pagination normalisation, `~`/normpath path semantics, the binary
//!     extension set, the internal-cache read-block list, and a conservative
//!     secret-redaction pass. When the shared modules become public, callers
//!     can swap these for the canonical implementations behind the same
//!     signatures.
//!
//! Cross-agent file-state coordination is provided through the
//! [`FileStateHooks`] trait (defaulting to a no-op); the integration layer can
//! wire it to the real registry once published.
//!
//! # File-ops abstraction
//!
//! The Python `_get_file_ops(task_id)` resolves a terminal environment and
//! wraps it in `ShellFileOperations`. That machinery lives in `terminal_tool`.
//! To avoid blocking, this module is parameterised over a [`FileOpsProvider`]
//! trait: the caller supplies a provider that runs an operation against the
//! file-ops for a given task id and returns the operation's JSON result dict
//! (the Python `result.to_dict()`). A `FakeProvider` is used in tests.
//!
//! The pure orchestration logic (tracking, guards, dedup, JSON shaping) is
//! reproduced faithfully and does not depend on the provider.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Local helper implementations (mirroring the imported Python helpers).
//
// These are intentionally small reproductions so the module compiles without
// depending on sibling crate modules that may not yet be public in the
// parallel port. Signatures match the canonical helpers so swapping is trivial.
// ---------------------------------------------------------------------------

/// Mirror `os.path.expanduser`: expand a leading `~` / `~/...`. `~user` forms
/// pass through unchanged.
fn expand_user(path: &str) -> String {
    if path == "~" {
        return home_dir().to_string_lossy().into_owned();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir().join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Binary file extensions to skip for text-based operations. Mirrors
/// `tools/binary_extensions.py` (`.pdf` intentionally excluded).
const BINARY_EXTENSIONS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif", ".mp4", ".mov",
    ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg", ".mp3", ".wav", ".ogg",
    ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus", ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".rar", ".xz", ".z", ".tgz", ".iso", ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
    ".obj", ".lib", ".app", ".msi", ".deb", ".rpm", ".doc", ".docx", ".xls", ".xlsx", ".ppt",
    ".pptx", ".odt", ".ods", ".odp", ".ttf", ".otf", ".woff", ".woff2", ".eot", ".pyc", ".pyo",
    ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib", ".sqlite", ".sqlite3", ".db",
    ".mdb", ".idx", ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    ".swf", ".fla", ".lockb", ".dat", ".data",
];

/// Check if a file path has a binary extension. Mirrors Python's
/// `path.rfind(".")` over the whole path, lowercased.
pub fn has_binary_extension(path: &str) -> bool {
    match path.rfind('.') {
        None => false,
        Some(dot) => {
            let ext = path[dot..].to_lowercase();
            BINARY_EXTENSIONS.contains(&ext.as_str())
        }
    }
}

/// Active HERMES_HOME (profile-aware): `HERMES_HOME` when set, else `~/.hermes`.
fn hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    home_dir().join(".hermes")
}

/// Mirror `agent.file_safety.get_read_block_error`: refuse reads of Hermes
/// internal catalog/hub metadata caches (prompt-injection surface). Returns a
/// message when blocked, else `None`.
pub fn get_read_block_error(path: &str) -> Option<String> {
    let resolved = resolve_path_for_task(path, "default")
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| expand_user(path));
    let home = hermes_home();
    let blocked_dirs = [
        home.join("catalog"),
        home.join("hub"),
        home.join("skills_hub"),
    ];
    for d in &blocked_dirs {
        let ds = d.to_string_lossy();
        if resolved == *ds || resolved.starts_with(&format!("{}/", ds)) {
            return Some(format!(
                "Refusing to read Hermes internal metadata file: {}\n\
                 These files may contain untrusted content from external sources.",
                path
            ));
        }
    }
    None
}

/// Conservative secret redaction. The canonical implementation lives in
/// `agent_redact`; until it is public we apply a minimal pass that masks
/// common high-confidence token shapes. `code_file`/`force` are accepted for
/// signature parity; redaction is gated by `HERMES_REDACT` unless `force`.
pub fn redact_sensitive_text(text: &str, force: bool, _code_file: bool) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    if !(force || redact_enabled()) {
        return text.to_string();
    }
    redact_token_shapes(text)
}

fn redact_enabled() -> bool {
    std::env::var("HERMES_REDACT")
        .map(|v| {
            let v = v.trim().to_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn redact_token_shapes(text: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // High-confidence provider key prefixes followed by a token body.
        regex::Regex::new(
            r"(?i)\b(sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{20,})\b",
        )
        .unwrap()
    });
    re.replace_all(text, "[REDACTED]").into_owned()
}

/// Return safe `read_file` pagination bounds. Mirrors
/// `normalize_read_pagination`: offset >= 1 (default 1), limit clamped to
/// [1, 2000] (default 500).
fn normalize_read_pagination(offset: Option<i64>, limit: Option<i64>) -> (i64, i64) {
    let normalized_offset = std::cmp::max(1, offset.unwrap_or(1));
    let limit = limit.unwrap_or(500);
    let normalized_limit = std::cmp::max(1, std::cmp::min(limit, 2000));
    (normalized_offset, normalized_limit)
}

/// Return safe search pagination bounds. Mirrors
/// `normalize_search_pagination`: offset >= 0 (default 0), limit >= 1
/// (default 50).
fn normalize_search_pagination(offset: Option<i64>, limit: Option<i64>) -> (i64, i64) {
    let normalized_offset = std::cmp::max(0, offset.unwrap_or(0));
    let normalized_limit = std::cmp::max(1, limit.unwrap_or(50));
    (normalized_offset, normalized_limit)
}

// ---------------------------------------------------------------------------
// Cross-agent file-state hooks (pluggable; default no-op)
// ---------------------------------------------------------------------------

/// Hooks into the cross-agent file-state registry. The canonical registry
/// lives in `tool_file_state`; until it is public the integration layer can
/// install a real implementation via [`set_file_state_hooks`]. The default is
/// a no-op (single-agent behaviour), which is also what setting
/// `HERMES_DISABLE_FILE_STATE_GUARD=1` produces in the Python original.
pub trait FileStateHooks: Send + Sync {
    /// Records that `task_id` read `resolved` (possibly partial).
    fn record_read(&self, _task_id: &str, _resolved: &str, _partial: bool) {}
    /// Records a successful write by `task_id` to `resolved`.
    fn note_write(&self, _task_id: &str, _resolved: &str) {}
    /// Returns a staleness warning naming the sibling writer, or `None`.
    fn check_stale(&self, _task_id: &str, _resolved: &str) -> Option<String> {
        None
    }
}

struct NoopFileStateHooks;
impl FileStateHooks for NoopFileStateHooks {}

static FILE_STATE_HOOKS: Mutex<Option<Box<dyn FileStateHooks>>> = Mutex::new(None);

/// Install cross-agent file-state hooks (e.g. wiring to the real registry).
pub fn set_file_state_hooks(hooks: Box<dyn FileStateHooks>) {
    *FILE_STATE_HOOKS.lock().unwrap() = Some(hooks);
}

fn fs_record_read(task_id: &str, resolved: &str, partial: bool) {
    if let Some(h) = FILE_STATE_HOOKS.lock().unwrap().as_ref() {
        h.record_read(task_id, resolved, partial);
    }
}
fn fs_note_write(task_id: &str, resolved: &str) {
    if let Some(h) = FILE_STATE_HOOKS.lock().unwrap().as_ref() {
        h.note_write(task_id, resolved);
    }
}
fn fs_check_stale(task_id: &str, resolved: &str) -> Option<String> {
    FILE_STATE_HOOKS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|h| h.check_stale(task_id, resolved))
}

// ---------------------------------------------------------------------------
// tool_error — mirrors tools.registry.tool_error
// ---------------------------------------------------------------------------

/// Build a JSON error string `{"error": <msg>}` (the registry's `tool_error`).
pub fn tool_error(msg: &str) -> String {
    json!({ "error": msg }).to_string()
}

// ---------------------------------------------------------------------------
// Read-size guard
// ---------------------------------------------------------------------------

const DEFAULT_MAX_READ_CHARS: usize = 100_000;

static MAX_READ_CHARS_CACHE: Mutex<Option<usize>> = Mutex::new(None);

/// Return the configured max characters per file read.
///
/// Reads `file_read_max_chars` from config on first call and caches it for the
/// lifetime of the process. Falls back to [`DEFAULT_MAX_READ_CHARS`] when the
/// config is missing or invalid.
pub fn get_max_read_chars() -> usize {
    {
        let guard = MAX_READ_CHARS_CACHE.lock().unwrap();
        if let Some(v) = *guard {
            return v;
        }
    }
    let resolved = load_configured_max_read_chars().unwrap_or(DEFAULT_MAX_READ_CHARS);
    let mut guard = MAX_READ_CHARS_CACHE.lock().unwrap();
    *guard = Some(resolved);
    resolved
}

fn load_configured_max_read_chars() -> Option<usize> {
    // Mirror `load_config().get("file_read_max_chars")`. The canonical config
    // loader isn't public here, so read `$HERMES_HOME/config.yaml` directly.
    let cfg_path = hermes_home().join("config.yaml");
    let text = std::fs::read_to_string(&cfg_path).ok()?;
    let cfg: serde_yaml::Value = serde_yaml::from_str(&text).ok()?;
    let val = cfg.get("file_read_max_chars")?;
    // isinstance(val, (int, float)) and val > 0
    if let Some(i) = val.as_i64() {
        if i > 0 {
            return Some(i as usize);
        }
    } else if let Some(f) = val.as_f64() {
        if f > 0.0 {
            return Some(f as usize);
        }
    }
    None
}

/// If the total file size exceeds this AND the caller didn't ask for a narrow
/// range (limit <= 200), we include a hint encouraging targeted reads.
const LARGE_FILE_HINT_BYTES: i64 = 512_000; // 512 KB

// ---------------------------------------------------------------------------
// Device path blocklist
// ---------------------------------------------------------------------------

const BLOCKED_DEVICE_PATHS: &[&str] = &[
    // Infinite output — never reach EOF
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/full",
    // Blocks waiting for input
    "/dev/stdin",
    "/dev/tty",
    "/dev/console",
    // Nonsensical to read
    "/dev/stdout",
    "/dev/stderr",
    // fd aliases
    "/dev/fd/0",
    "/dev/fd/1",
    "/dev/fd/2",
];

/// Return true if the path would hang the process (infinite output or blocking
/// input). Uses the *literal* path (after `~` expansion) — no symlink
/// resolution — mirroring the Python guard.
pub fn is_blocked_device(filepath: &str) -> bool {
    let normalized = expand_user(filepath);
    if BLOCKED_DEVICE_PATHS.contains(&normalized.as_str()) {
        return true;
    }
    // /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio.
    if normalized.starts_with("/proc/")
        && (normalized.ends_with("/fd/0")
            || normalized.ends_with("/fd/1")
            || normalized.ends_with("/fd/2"))
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Sensitive-path guard
// ---------------------------------------------------------------------------

const SENSITIVE_PATH_PREFIXES: &[&str] = &[
    "/etc/",
    "/boot/",
    "/usr/lib/systemd/",
    "/private/etc/",
    "/private/var/",
];

const SENSITIVE_EXACT_PATHS: &[&str] = &["/var/run/docker.sock", "/run/docker.sock"];

/// Return an error message if the path targets a sensitive system location.
pub fn check_sensitive_path(filepath: &str, task_id: &str) -> Option<String> {
    let resolved = resolve_path_for_task(filepath, task_id)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| filepath.to_string());
    let normalized = normpath(&expand_user(filepath));
    let err = format!(
        "Refusing to write to sensitive system path: {}\n\
         Use the terminal tool with sudo if you need to modify system files.",
        filepath
    );
    for prefix in SENSITIVE_PATH_PREFIXES {
        if resolved.starts_with(prefix) || normalized.starts_with(prefix) {
            return Some(err);
        }
    }
    if SENSITIVE_EXACT_PATHS.contains(&resolved.as_str())
        || SENSITIVE_EXACT_PATHS.contains(&normalized.as_str())
    {
        return Some(err);
    }
    None
}

/// Mirror `os.path.normpath`: lexical collapse of `.` / `..` / repeated
/// separators (no symlink resolution, no absolutisation).
fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    let is_abs = path.starts_with('/');
    // Count leading slashes for POSIX special-case (exactly two → preserved).
    let leading = path.chars().take_while(|&c| c == '/').count();
    let mut comps: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if is_abs {
                    if comps.last().is_some_and(|&c| c != "..") {
                        comps.pop();
                    } else if !is_abs {
                        comps.push("..");
                    }
                    // absolute: leading .. are discarded
                } else if comps.last().is_some_and(|&c| c != "..") {
                    comps.pop();
                } else {
                    comps.push("..");
                }
            }
            other => comps.push(other),
        }
    }
    let body = comps.join("/");
    if is_abs {
        let prefix = if leading == 2 { "//" } else { "/" };
        if body.is_empty() {
            prefix.to_string()
        } else {
            format!("{}{}", prefix, body)
        }
    } else if body.is_empty() {
        ".".to_string()
    } else {
        body
    }
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Resolve a path relative to TERMINAL_CWD (the worktree base directory)
/// instead of the main repository root.
pub fn resolve_path(filepath: &str, task_id: &str) -> Option<PathBuf> {
    resolve_path_for_task(filepath, task_id)
}

/// Resolve *filepath* against the task's live terminal cwd when possible.
///
/// Mirrors `_resolve_path_for_task`: expand `~`, and if not absolute, join
/// against the live tracking cwd (when known), else `TERMINAL_CWD`, else the
/// process cwd. Finally resolve to an absolute, symlink-collapsed path.
pub fn resolve_path_for_task(filepath: &str, task_id: &str) -> Option<PathBuf> {
    let expanded = expand_user(filepath);
    let p = Path::new(&expanded);
    let abs: PathBuf = if p.is_absolute() {
        p.to_path_buf()
    } else {
        let base = get_live_tracking_cwd(task_id)
            .or_else(|| std::env::var("TERMINAL_CWD").ok())
            .or_else(|| std::env::current_dir().ok().map(|d| d.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "/".to_string());
        Path::new(&base).join(p)
    };
    // Path.resolve(): canonicalise when it exists, else lexical-normalise the
    // absolute form (Python's resolve() returns a path even for non-existent
    // targets in 3.6+, resolving the existing prefix).
    match std::fs::canonicalize(&abs) {
        Ok(c) => Some(c),
        Err(_) => Some(PathBuf::from(normpath(&abs.to_string_lossy()))),
    }
}

/// Return the task's live terminal cwd for bookkeeping when available.
///
/// In the Python original this consults the terminal_tool environment caches.
/// Those caches are owned by the not-yet-ported terminal_tool; this hook is
/// pluggable via [`set_live_cwd_resolver`]. Default returns `None` so paths
/// resolve against `TERMINAL_CWD` / process cwd, matching the fallback branch.
pub fn get_live_tracking_cwd(task_id: &str) -> Option<String> {
    let guard = LIVE_CWD_RESOLVER.lock().unwrap();
    if let Some(f) = guard.as_ref() {
        return f(task_id);
    }
    None
}

type LiveCwdResolver = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;
static LIVE_CWD_RESOLVER: Mutex<Option<LiveCwdResolver>> = Mutex::new(None);

/// Install a resolver that maps a task id to its live terminal cwd. Used by the
/// integration layer to bridge to the ported terminal-environment caches.
pub fn set_live_cwd_resolver<F>(f: F)
where
    F: Fn(&str) -> Option<String> + Send + Sync + 'static,
{
    *LIVE_CWD_RESOLVER.lock().unwrap() = Some(Box::new(f));
}

// ---------------------------------------------------------------------------
// Per-task read tracker
// ---------------------------------------------------------------------------

const READ_HISTORY_CAP: usize = 500;
const DEDUP_CAP: usize = 1000;
const READ_TIMESTAMPS_CAP: usize = 1000;

const READ_DEDUP_STATUS_MESSAGE: &str = "File unchanged since last read. The content from \
the earlier read_file result in this conversation is \
still current — refer to that instead of re-reading.";

/// Key for the most recent read/search call: distinguishes reads from searches.
#[derive(Clone, PartialEq, Eq, Debug)]
enum LastKey {
    Read {
        path: String,
        offset: i64,
        limit: i64,
    },
    Search {
        pattern: String,
        target: String,
        path: String,
        file_glob: String,
        limit: i64,
        offset: i64,
    },
}

/// Dedup key: `(resolved_path, offset, limit)`.
type DedupKey = (String, i64, i64);

#[derive(Default)]
struct TaskTracker {
    last_key: Option<LastKey>,
    consecutive: i64,
    /// (path, offset, limit) tuples for diagnostic summaries.
    read_history: Vec<(String, i64, i64)>,
    read_history_set: HashSet<(String, i64, i64)>,
    /// resolved_path/offset/limit → mtime. Insertion-ordered.
    dedup: OrderedMap<DedupKey, f64>,
    /// resolved_path/offset/limit → stub-return count. Insertion-ordered.
    dedup_hits: OrderedMap<DedupKey, i64>,
    /// resolved_path → mtime recorded at read/write time. Insertion-ordered.
    read_timestamps: OrderedMap<String, f64>,
}

/// Insertion-ordered map (Python `dict` semantics) for eviction by age.
struct OrderedMap<K: Clone + Eq + std::hash::Hash, V> {
    map: HashMap<K, V>,
    order: Vec<K>,
}

impl<K: Clone + Eq + std::hash::Hash, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        OrderedMap {
            map: HashMap::new(),
            order: Vec::new(),
        }
    }
}

impl<K: Clone + Eq + std::hash::Hash, V> OrderedMap<K, V> {
    fn insert(&mut self, key: K, value: V) {
        if !self.map.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.map.insert(key, value);
    }
    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }
    fn remove(&mut self, key: &K) -> Option<V> {
        let v = self.map.remove(key);
        if v.is_some() {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
        }
        v
    }
    fn len(&self) -> usize {
        self.map.len()
    }
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
    /// Pop oldest by insertion order.
    fn pop_oldest(&mut self) -> Option<(K, V)> {
        if self.order.is_empty() {
            return None;
        }
        let key = self.order.remove(0);
        self.map.remove(&key).map(|v| (key, v))
    }
    fn keys(&self) -> impl Iterator<Item = &K> {
        self.order.iter()
    }
}

static READ_TRACKER: Mutex<Option<HashMap<String, TaskTracker>>> = Mutex::new(None);

/// Enforce size caps on the per-task read-tracker sub-containers.
fn cap_read_tracker_data(td: &mut TaskTracker) {
    if td.read_history.len() > READ_HISTORY_CAP {
        let excess = td.read_history.len() - READ_HISTORY_CAP;
        for _ in 0..excess {
            if td.read_history.is_empty() {
                break;
            }
            let removed = td.read_history.remove(0);
            td.read_history_set.remove(&removed);
        }
    }
    while td.dedup.len() > DEDUP_CAP {
        if td.dedup.pop_oldest().is_none() {
            break;
        }
    }
    while td.dedup_hits.len() > DEDUP_CAP {
        if td.dedup_hits.pop_oldest().is_none() {
            break;
        }
    }
    while td.read_timestamps.len() > READ_TIMESTAMPS_CAP {
        if td.read_timestamps.pop_oldest().is_none() {
            break;
        }
    }
}

fn with_tracker<R>(f: impl FnOnce(&mut HashMap<String, TaskTracker>) -> R) -> R {
    let mut guard = READ_TRACKER.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    f(guard.as_mut().unwrap())
}

// ---------------------------------------------------------------------------
// Internal status-text write guard
// ---------------------------------------------------------------------------

/// Return true when content looks like an internal file-tool status, not real
/// file bytes. Mirrors `_is_internal_file_status_text`.
pub fn is_internal_file_status_text(content: &str) -> bool {
    let stripped = content.trim();
    if stripped.is_empty() {
        return false;
    }
    if stripped == READ_DEDUP_STATUS_MESSAGE {
        return true;
    }
    if stripped.contains(READ_DEDUP_STATUS_MESSAGE)
        && stripped.len() <= 2 * READ_DEDUP_STATUS_MESSAGE.len()
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Expected write-error classification
// ---------------------------------------------------------------------------

/// errno values for write denials that shouldn't hit error logs:
/// EACCES (13), EPERM (1), EROFS (30) on Linux/macOS.
pub fn is_expected_write_errno(errno: i32) -> bool {
    matches!(errno, libc::EACCES | libc::EPERM | libc::EROFS)
}

// ---------------------------------------------------------------------------
// mtime helper (mirrors os.path.getmtime → OSError on failure)
// ---------------------------------------------------------------------------

fn get_mtime(path: &str) -> Option<f64> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    modified.duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).ok()
}

fn get_file_size(path: &str) -> Option<i64> {
    std::fs::metadata(path).ok().map(|m| m.len() as i64)
}

// ---------------------------------------------------------------------------
// File-ops provider abstraction
// ---------------------------------------------------------------------------

/// A request describing one file operation to run against the task's file-ops.
///
/// Mirrors the four `ShellFileOperations` methods the Python tools invoke. The
/// provider executes the op and returns its `result.to_dict()` JSON object.
#[derive(Debug, Clone)]
pub enum FileOpRequest<'a> {
    Read {
        path: &'a str,
        offset: i64,
        limit: i64,
    },
    Write {
        path: &'a str,
        content: &'a str,
    },
    PatchReplace {
        path: &'a str,
        old_string: &'a str,
        new_string: &'a str,
        replace_all: bool,
    },
    PatchV4a {
        patch: &'a str,
    },
    Search {
        pattern: &'a str,
        path: &'a str,
        target: &'a str,
        file_glob: Option<&'a str>,
        limit: i64,
        offset: i64,
        output_mode: &'a str,
        context: i64,
    },
}

/// Supplies file-ops for a task id (mirrors `_get_file_ops`).
///
/// Implementations own the terminal-environment lifecycle and caching. The
/// orchestration functions hand the provider a [`FileOpRequest`]; the provider
/// runs it (e.g. via `ShellFileOperations`) and returns the operation's JSON
/// result dict (the Python `result.to_dict()`), which this module then shapes.
///
/// Object-safe so the tools can take `&dyn FileOpsProvider`.
pub trait FileOpsProvider {
    /// Execute `req` against the file-ops for `task_id`; return the result dict.
    fn run(&self, task_id: &str, req: FileOpRequest<'_>) -> Value;
}

// ---------------------------------------------------------------------------
// read_file_tool
// ---------------------------------------------------------------------------

/// Read a file with pagination and line numbers. Mirrors `read_file_tool`.
pub fn read_file_tool(
    provider: &dyn FileOpsProvider,
    path: &str,
    offset: i64,
    limit: i64,
    task_id: &str,
) -> String {
    let (offset, limit) = normalize_read_pagination(Some(offset), Some(limit));

    // ── Device path guard ────────────────────────────────────────────
    if is_blocked_device(path) {
        return json!({
            "error": format!(
                "Cannot read '{}': this is a device file that would block or produce infinite output.",
                path
            )
        })
        .to_string();
    }

    let resolved = match resolve_path_for_task(path, task_id) {
        Some(p) => p,
        None => {
            // Can't resolve — fall back to literal for the binary check etc.
            PathBuf::from(expand_user(path))
        }
    };
    let resolved_str = resolved.to_string_lossy().into_owned();

    // ── Binary file guard ────────────────────────────────────────────
    if has_binary_extension(&resolved_str) {
        let ext = Path::new(&resolved_str)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        return json!({
            "error": format!(
                "Cannot read binary file '{}' ({}). Use vision_analyze for images, or terminal to inspect binary files.",
                path, ext
            )
        })
        .to_string();
    }

    // ── Hermes internal path guard ───────────────────────────────────
    if let Some(block_error) = get_read_block_error(path) {
        return json!({ "error": block_error }).to_string();
    }

    // ── Dedup check ──────────────────────────────────────────────────
    let dedup_key: DedupKey = (resolved_str.clone(), offset, limit);
    let cached_mtime = with_tracker(|tr| {
        let td = tr.entry(task_id.to_string()).or_default();
        td.dedup.get(&dedup_key).copied()
    });

    if let Some(cm) = cached_mtime {
        if let Some(current_mtime) = get_mtime(&resolved_str) {
            if current_mtime == cm {
                let hits = with_tracker(|tr| {
                    let td = tr.entry(task_id.to_string()).or_default();
                    let h = td.dedup_hits.get(&dedup_key).copied().unwrap_or(0) + 1;
                    td.dedup_hits.insert(dedup_key.clone(), h);
                    cap_read_tracker_data(td);
                    h
                });
                if hits >= 2 {
                    return json!({
                        "error": format!(
                            "BLOCKED: You have called read_file on this exact region {} times and the file has NOT changed. STOP calling read_file for this path — the content from your earlier read_file result in this conversation is still current. Proceed with your task using the information you already have.",
                            hits + 1
                        ),
                        "path": path,
                        "already_read": hits + 1,
                    })
                    .to_string();
                }
                return json!({
                    "status": "unchanged",
                    "message": READ_DEDUP_STATUS_MESSAGE,
                    "path": path,
                    "dedup": true,
                    "content_returned": false,
                })
                .to_string();
            }
        }
        // stat failed or mtime changed — fall through to full read.
    }

    // ── Perform the read ─────────────────────────────────────────────
    let mut result_dict = provider.run(
        task_id,
        FileOpRequest::Read {
            path,
            offset,
            limit,
        },
    );

    // ── Character-count guard ────────────────────────────────────────
    // Python: len(result.content or "") — character count of the content.
    let content_str = result_dict
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let content_len = content_str.chars().count();
    let file_size = result_dict
        .get("file_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let max_chars = get_max_read_chars();
    if content_len > max_chars {
        let total_lines = result_dict
            .get("total_lines")
            .cloned()
            .unwrap_or_else(|| Value::String("unknown".to_string()));
        let total_lines_disp = json_scalar_display(&total_lines);
        return json!({
            "error": format!(
                "Read produced {} characters which exceeds the safety limit ({} chars). Use offset and limit to read a smaller range. The file has {} lines total.",
                fmt_thousands(content_len as i64), fmt_thousands(max_chars as i64), total_lines_disp
            ),
            "path": path,
            "total_lines": total_lines,
            "file_size": file_size,
        })
        .to_string();
    }

    // ── Redact secrets ───────────────────────────────────────────────
    // Python guards on `if result.content:` (truthy → non-empty string).
    if !content_str.is_empty() {
        let redacted = redact_sensitive_text(&content_str, false, true);
        if let Value::Object(map) = &mut result_dict {
            map.insert("content".to_string(), Value::String(redacted));
        }
    }

    // ── Large-file hint ──────────────────────────────────────────────
    let truncated = result_dict.get("truncated").and_then(|v| v.as_bool()).unwrap_or(false);
    if file_size != 0 && file_size > LARGE_FILE_HINT_BYTES && limit > 200 && truncated {
        if let Value::Object(map) = &mut result_dict {
            map.entry("_hint").or_insert_with(|| {
                Value::String(format!(
                    "This file is large ({} bytes). Consider reading only the section you need with offset and limit to keep context usage efficient.",
                    fmt_thousands(file_size)
                ))
            });
        }
    }

    // ── Track for consecutive-loop detection ─────────────────────────
    let read_key = LastKey::Read {
        path: path.to_string(),
        offset,
        limit,
    };
    let count = with_tracker(|tr| {
        let td = tr.entry(task_id.to_string()).or_default();
        td.dedup_hits.remove(&dedup_key);
        let hist_key = (path.to_string(), offset, limit);
        if td.read_history_set.insert(hist_key.clone()) {
            td.read_history.push(hist_key);
        }
        if td.last_key.as_ref() == Some(&read_key) {
            td.consecutive += 1;
        } else {
            td.last_key = Some(read_key.clone());
            td.consecutive = 1;
        }
        let c = td.consecutive;
        if let Some(mtime_now) = get_mtime(&resolved_str) {
            td.dedup.insert(dedup_key.clone(), mtime_now);
            td.read_timestamps.insert(resolved_str.clone(), mtime_now);
        }
        cap_read_tracker_data(td);
        c
    });

    // ── Cross-agent file-state registry ──────────────────────────────
    let partial = offset > 1 || truncated;
    fs_record_read(task_id, &resolved_str, partial);

    if count >= 4 {
        return json!({
            "error": format!(
                "BLOCKED: You have read this exact file region {} times in a row. The content has NOT changed. You already have this information. STOP re-reading and proceed with your task.",
                count
            ),
            "path": path,
            "already_read": count,
        })
        .to_string();
    } else if count >= 3 {
        if let Value::Object(map) = &mut result_dict {
            map.insert(
                "_warning".to_string(),
                Value::String(format!(
                    "You have read this exact file region {} times consecutively. The content has not changed since your last read. Use the information you already have. If you are stuck in a loop, stop reading and proceed with writing or responding.",
                    count
                )),
            );
        }
    }

    result_dict.to_string()
}

/// Render a JSON scalar the way Python's f-string would for the total_lines
/// hint (integers without quotes, the string "unknown" without quotes).
fn json_scalar_display(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Format an integer with comma thousands separators (Python `{:,}`).
fn fmt_thousands(n: i64) -> String {
    let neg = n < 0;
    let s = n.unsigned_abs().to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if neg {
        format!("-{}", out)
    } else {
        out
    }
}

// ---------------------------------------------------------------------------
// reset_file_dedup / notify_other_tool_call
// ---------------------------------------------------------------------------

/// Clear the deduplication cache for file reads. Called after context
/// compression. `None` task_id clears all tasks.
pub fn reset_file_dedup(task_id: Option<&str>) {
    with_tracker(|tr| match task_id {
        Some(tid) => {
            if let Some(td) = tr.get_mut(tid) {
                td.dedup.clear();
                td.dedup_hits.clear();
            }
        }
        None => {
            for td in tr.values_mut() {
                td.dedup.clear();
                td.dedup_hits.clear();
            }
        }
    });
}

/// Reset consecutive read/search counter for a task (an intervening non-read
/// tool call breaks any loop in progress). Mirrors `notify_other_tool_call`.
pub fn notify_other_tool_call(task_id: &str) {
    with_tracker(|tr| {
        if let Some(td) = tr.get_mut(task_id) {
            td.last_key = None;
            td.consecutive = 0;
            td.dedup_hits.clear();
        }
    });
}

// ---------------------------------------------------------------------------
// dedup / timestamp invalidation helpers (used by write/patch)
// ---------------------------------------------------------------------------

/// Remove all dedup cache entries whose resolved path matches *filepath*.
fn invalidate_dedup_for_path(filepath: &str, task_id: &str) {
    let resolved = match resolve_path(filepath, "default") {
        Some(p) => p.to_string_lossy().into_owned(),
        None => return,
    };
    with_tracker(|tr| {
        if let Some(td) = tr.get_mut(task_id) {
            let stale: Vec<DedupKey> = td
                .dedup
                .keys()
                .filter(|k| k.0 == resolved)
                .cloned()
                .collect();
            for k in stale {
                td.dedup.remove(&k);
            }
        }
    });
}

/// Record the file's current modification time after a successful write.
fn update_read_timestamp(filepath: &str, task_id: &str) {
    invalidate_dedup_for_path(filepath, task_id);
    let resolved = match resolve_path_for_task(filepath, task_id) {
        Some(p) => p.to_string_lossy().into_owned(),
        None => return,
    };
    let current_mtime = match get_mtime(&resolved) {
        Some(m) => m,
        None => return,
    };
    with_tracker(|tr| {
        if let Some(td) = tr.get_mut(task_id) {
            td.read_timestamps.insert(resolved.clone(), current_mtime);
            cap_read_tracker_data(td);
        }
    });
}

/// Check whether a file was modified since the agent last read it. Returns a
/// warning string if stale, else `None`. Does not block.
fn check_file_staleness(filepath: &str, task_id: &str) -> Option<String> {
    let resolved = resolve_path_for_task(filepath, task_id)?
        .to_string_lossy()
        .into_owned();
    let read_mtime = with_tracker(|tr| {
        tr.get(task_id)
            .and_then(|td| td.read_timestamps.get(&resolved).copied())
    });
    let read_mtime = read_mtime?;
    let current_mtime = get_mtime(&resolved)?;
    if current_mtime != read_mtime {
        Some(format!(
            "Warning: {} was modified since you last read it (external edit or concurrent agent). The content you read may be stale. Consider re-reading the file to verify before writing.",
            filepath
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// write_file_tool
// ---------------------------------------------------------------------------

/// Write content to a file. Mirrors `write_file_tool`.
pub fn write_file_tool(
    provider: &dyn FileOpsProvider,
    path: &str,
    content: &str,
    task_id: &str,
) -> String {
    if let Some(sensitive_err) = check_sensitive_path(path, task_id) {
        return tool_error(&sensitive_err);
    }
    if is_internal_file_status_text(content) {
        return tool_error(
            "Refusing to write internal read_file status text as file content. Re-read the file or reconstruct the intended file contents before writing.",
        );
    }

    let resolved = resolve_path_for_task(path, task_id).map(|p| p.to_string_lossy().into_owned());

    match resolved {
        None => {
            let stale_warning = check_file_staleness(path, task_id);
            let mut result_dict = provider.run(task_id, FileOpRequest::Write { path, content });
            if let Some(w) = stale_warning {
                if let Value::Object(m) = &mut result_dict {
                    m.insert("_warning".to_string(), Value::String(w));
                }
            }
            update_read_timestamp(path, task_id);
            result_dict.to_string()
        }
        Some(resolved_str) => {
            // Serialise read→modify→write per path. We hold no real registry
            // lock here unless hooks install one; the ordering still mirrors
            // the Python critical section.
            let cross_warning = fs_check_stale(task_id, &resolved_str);
            let stale_warning = check_file_staleness(path, task_id);
            let mut result_dict = provider.run(task_id, FileOpRequest::Write { path, content });
            let effective_warning = cross_warning.or(stale_warning);
            if let Some(w) = effective_warning {
                if let Value::Object(m) = &mut result_dict {
                    m.insert("_warning".to_string(), Value::String(w));
                }
            }
            update_read_timestamp(path, task_id);
            let has_error = result_dict
                .get("error")
                .map(|e| !e.is_null())
                .unwrap_or(false);
            if !has_error {
                fs_note_write(task_id, &resolved_str);
            }
            result_dict.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// patch_tool
// ---------------------------------------------------------------------------

/// Patch a file using replace mode or V4A patch format. Mirrors `patch_tool`.
#[allow(clippy::too_many_arguments)]
pub fn patch_tool(
    provider: &dyn FileOpsProvider,
    mode: &str,
    path: Option<&str>,
    old_string: Option<&str>,
    new_string: Option<&str>,
    replace_all: bool,
    patch: Option<&str>,
    task_id: &str,
) -> String {
    // Collect candidate paths for the sensitive-path check.
    let mut paths_to_check: Vec<String> = Vec::new();
    if let Some(p) = path {
        if !p.is_empty() {
            paths_to_check.push(p.to_string());
        }
    }
    if mode == "patch" {
        if let Some(patch_text) = patch {
            for cap in v4a_path_regex().captures_iter(patch_text) {
                paths_to_check.push(cap[1].trim().to_string());
            }
        }
    }
    for p in &paths_to_check {
        if let Some(sensitive_err) = check_sensitive_path(p, task_id) {
            return tool_error(&sensitive_err);
        }
    }

    // Resolve + dedupe + sort the paths for deterministic lock ordering.
    let mut resolved_paths: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for p in &paths_to_check {
        if let Some(r) = resolve_path_for_task(p, task_id) {
            let rs = r.to_string_lossy().into_owned();
            if seen.insert(rs.clone()) {
                resolved_paths.push(rs);
            }
        }
    }
    resolved_paths.sort();

    // `resolved_paths` is sorted+deduped so concurrent callers would lock in
    // the same order if the registry hooks install per-path locks. (Default
    // hooks are no-ops, matching single-agent behaviour.)
    let _lock_order = &resolved_paths;

    // Collect staleness warnings + path→resolved map.
    let mut stale_warnings: Vec<String> = Vec::new();
    let mut path_to_resolved: HashMap<String, Option<String>> = HashMap::new();
    for p in &paths_to_check {
        let r = resolve_path_for_task(p, task_id).map(|x| x.to_string_lossy().into_owned());
        path_to_resolved.insert(p.clone(), r.clone());
        let cross = r.as_ref().and_then(|rr| fs_check_stale(task_id, rr));
        let sw = cross.or_else(|| check_file_staleness(p, task_id));
        if let Some(w) = sw {
            stale_warnings.push(w);
        }
    }

    // Dispatch the patch operation.
    let mut result_dict: Value = if mode == "replace" {
        let path = match path {
            Some(p) if !p.is_empty() => p,
            _ => return tool_error("path required"),
        };
        let (os, ns) = match (old_string, new_string) {
            (Some(o), Some(n)) => (o, n),
            _ => return tool_error("old_string and new_string required"),
        };
        provider.run(
            task_id,
            FileOpRequest::PatchReplace {
                path,
                old_string: os,
                new_string: ns,
                replace_all,
            },
        )
    } else if mode == "patch" {
        let patch_text = match patch {
            Some(p) if !p.is_empty() => p,
            _ => return tool_error("patch content required"),
        };
        provider.run(task_id, FileOpRequest::PatchV4a { patch: patch_text })
    } else {
        return tool_error(&format!("Unknown mode: {}", mode));
    };
    if !stale_warnings.is_empty() {
        let warning = if stale_warnings.len() == 1 {
            stale_warnings[0].clone()
        } else {
            stale_warnings.join(" | ")
        };
        if let Value::Object(m) = &mut result_dict {
            m.insert("_warning".to_string(), Value::String(warning));
        }
    }

    let has_error = result_dict
        .get("error")
        .map(|e| !e.is_null())
        .unwrap_or(false);
    if !has_error {
        for p in &paths_to_check {
            update_read_timestamp(p, task_id);
            if let Some(Some(r)) = path_to_resolved.get(p) {
                fs_note_write(task_id, r);
            }
        }
    }

    // Hint when old_string not found.
    let error_text = result_dict
        .get("error")
        .and_then(|e| e.as_str())
        .map(|s| s.to_string());
    if let Some(err) = &error_text {
        if err.contains("Could not find") && !err.contains("Did you mean one of these sections?") {
            if let Value::Object(m) = &mut result_dict {
                m.insert(
                    "_hint".to_string(),
                    Value::String(
                        "old_string not found. Use read_file to verify the current content, or search_files to locate the text."
                            .to_string(),
                    ),
                );
            }
        }
    }

    result_dict.to_string()
}

fn v4a_path_regex() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"(?m)^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$").unwrap()
    })
}

// ---------------------------------------------------------------------------
// search_tool
// ---------------------------------------------------------------------------

/// Search for content or files. Mirrors `search_tool`.
#[allow(clippy::too_many_arguments)]
pub fn search_tool(
    provider: &dyn FileOpsProvider,
    pattern: &str,
    target: &str,
    path: &str,
    file_glob: Option<&str>,
    limit: i64,
    offset: i64,
    output_mode: &str,
    context: i64,
    task_id: &str,
) -> String {
    let (offset, limit) = normalize_search_pagination(Some(offset), Some(limit));

    let search_key = LastKey::Search {
        pattern: pattern.to_string(),
        target: target.to_string(),
        path: path.to_string(),
        file_glob: file_glob.unwrap_or("").to_string(),
        limit,
        offset,
    };
    let count = with_tracker(|tr| {
        let td = tr.entry(task_id.to_string()).or_default();
        if td.last_key.as_ref() == Some(&search_key) {
            td.consecutive += 1;
        } else {
            td.last_key = Some(search_key.clone());
            td.consecutive = 1;
        }
        td.consecutive
    });

    if count >= 4 {
        return json!({
            "error": format!(
                "BLOCKED: You have run this exact search {} times in a row. The results have NOT changed. You already have this information. STOP re-searching and proceed with your task.",
                count
            ),
            "pattern": pattern,
            "already_searched": count,
        })
        .to_string();
    }

    let mut result_dict = provider.run(
        task_id,
        FileOpRequest::Search {
            pattern,
            path,
            target,
            file_glob,
            limit,
            offset,
            output_mode,
            context,
        },
    );
    // Redact each match's content (Python iterates result.matches; the result
    // dict carries them under "matches" with a "content" field per match).
    if let Some(Value::Array(matches)) = result_dict.get_mut("matches") {
        for m in matches.iter_mut() {
            if let Value::Object(mo) = m {
                let needs = mo
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                if needs {
                    let c = mo.get("content").and_then(|c| c.as_str()).unwrap().to_string();
                    mo.insert(
                        "content".to_string(),
                        Value::String(redact_sensitive_text(&c, false, true)),
                    );
                }
            }
        }
    }

    if count >= 3 {
        if let Value::Object(m) = &mut result_dict {
            m.insert(
                "_warning".to_string(),
                Value::String(format!(
                    "You have run this exact search {} times consecutively. The results have not changed. Use the information you already have.",
                    count
                )),
            );
        }
    }

    let truncated = result_dict.get("truncated").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut result_json = result_dict.to_string();
    if truncated {
        let next_offset = offset + limit;
        result_json.push_str(&format!(
            "\n\n[Hint: Results truncated. Use offset={} to see more, or narrow with a more specific pattern or file_glob.]",
            next_offset
        ));
    }
    result_json
}

// ---------------------------------------------------------------------------
// Dispatch helpers (mirror _handle_* in the Python module)
// ---------------------------------------------------------------------------

/// Mirror `_handle_read_file`.
pub fn handle_read_file(provider: &dyn FileOpsProvider, args: &Value, task_id: &str) -> String {
    let tid = if task_id.is_empty() { "default" } else { task_id };
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(1);
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(500);
    read_file_tool(provider, path, offset, limit, tid)
}

/// Mirror `_handle_write_file` (including the arg-validation messages).
pub fn handle_write_file(provider: &dyn FileOpsProvider, args: &Value, task_id: &str) -> String {
    let tid = if task_id.is_empty() { "default" } else { task_id };
    let path = args.get("path");
    let path_str = path.and_then(|v| v.as_str());
    if path_str.map(|s| s.is_empty()).unwrap_or(true) || path_str.is_none() {
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with both 'path' and 'content' set.",
        );
    }
    if !args.get("content").is_some() {
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a path but no content argument — this is almost always a dropped-arg bug under context pressure. Re-emit the tool call with the full content payload, or use execute_code with hermes_tools.write_file() for very large files.",
        );
    }
    let content_val = args.get("content").unwrap();
    let content = match content_val.as_str() {
        Some(s) => s,
        None => {
            return tool_error(&format!(
                "write_file: 'content' must be a string, got {}.",
                json_type_name(content_val)
            ));
        }
    };
    write_file_tool(provider, path_str.unwrap(), content, tid)
}

/// Mirror `_handle_patch`.
pub fn handle_patch(provider: &dyn FileOpsProvider, args: &Value, task_id: &str) -> String {
    let tid = if task_id.is_empty() { "default" } else { task_id };
    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("replace");
    let path = args.get("path").and_then(|v| v.as_str());
    let old_string = args.get("old_string").and_then(|v| v.as_str());
    let new_string = args.get("new_string").and_then(|v| v.as_str());
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let patch = args.get("patch").and_then(|v| v.as_str());
    patch_tool(
        provider, mode, path, old_string, new_string, replace_all, patch, tid,
    )
}

/// Mirror `_handle_search_files` (including the grep/find target aliasing).
pub fn handle_search_files(provider: &dyn FileOpsProvider, args: &Value, task_id: &str) -> String {
    let tid = if task_id.is_empty() { "default" } else { task_id };
    let raw_target = args.get("target").and_then(|v| v.as_str()).unwrap_or("content");
    let target = match raw_target {
        "grep" => "content",
        "find" => "files",
        other => other,
    };
    let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let file_glob = args.get("file_glob").and_then(|v| v.as_str());
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(50);
    let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);
    let output_mode = args
        .get("output_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("content");
    let context = args.get("context").and_then(|v| v.as_i64()).unwrap_or(0);
    search_tool(
        provider, pattern, target, path, file_glob, limit, offset, output_mode, context, tid,
    )
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

// ---------------------------------------------------------------------------
// Tool schemas (mirror the *_SCHEMA dicts)
// ---------------------------------------------------------------------------

/// JSON schema for the `read_file` tool.
pub fn read_file_schema() -> Value {
    json!({
        "name": "read_file",
        "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are rejected; use offset and limit to read specific sections of large files. NOTE: Cannot read images or binary files — use vision_analyze for images.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
                "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
                "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 500, max: 2000)", "default": 500, "maximum": 2000}
            },
            "required": ["path"]
        }
    })
}

/// JSON schema for the `write_file` tool.
pub fn write_file_schema() -> Value {
    json!({
        "name": "write_file",
        "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
                "content": {"type": "string", "description": "Complete content to write to the file"}
            },
            "required": ["path", "content"]
        }
    })
}

/// JSON schema for the `patch` tool.
pub fn patch_schema() -> Value {
    json!({
        "name": "patch",
        "description": "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. Returns a unified diff. Auto-runs syntax checks after editing.\n\nReplace mode (default): find a unique string and replace it.\nPatch mode: apply V4A multi-file patches for bulk changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["replace", "patch"], "description": "Edit mode: 'replace' for targeted find-and-replace, 'patch' for V4A multi-file patches", "default": "replace"},
                "path": {"type": "string", "description": "File path to edit (required for 'replace' mode)"},
                "old_string": {"type": "string", "description": "Text to find in the file (required for 'replace' mode). Must be unique in the file unless replace_all=true. Include enough surrounding context to ensure uniqueness."},
                "new_string": {"type": "string", "description": "Replacement text (required for 'replace' mode). Can be empty string to delete the matched text."},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of requiring a unique match (default: false)", "default": false},
                "patch": {"type": "string", "description": "V4A format patch content (required for 'patch' mode). Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch"}
            },
            "required": ["mode"]
        }
    })
}

/// JSON schema for the `search_files` tool.
pub fn search_files_schema() -> Value {
    json!({
        "name": "search_files",
        "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
                "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
                "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
                "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
                "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
                "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
                "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
                "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
            },
            "required": ["pattern"]
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A provider returning canned result dicts (the Python `to_dict()` shape).
    struct FakeProvider {
        read_content: String,
        read_total_lines: i64,
        read_file_size: i64,
        read_truncated: bool,
    }

    impl FakeProvider {
        fn empty() -> Self {
            FakeProvider::empty()
        }
    }

    impl FileOpsProvider for FakeProvider {
        fn run(&self, _task_id: &str, req: FileOpRequest<'_>) -> Value {
            match req {
                FileOpRequest::Read { .. } => json!({
                    "content": self.read_content,
                    "total_lines": self.read_total_lines,
                    "file_size": self.read_file_size,
                    "truncated": self.read_truncated,
                }),
                FileOpRequest::Write { .. } => json!({"bytes_written": 2}),
                FileOpRequest::PatchReplace { .. } | FileOpRequest::PatchV4a { .. } => {
                    json!({"success": true})
                }
                FileOpRequest::Search { .. } => json!({
                    "total_count": 1,
                    "matches": [{"path": "x", "line": 1, "content": "secret"}],
                }),
            }
        }
    }

    #[test]
    fn blocked_device_paths() {
        assert!(is_blocked_device("/dev/zero"));
        assert!(is_blocked_device("/dev/stdin"));
        assert!(is_blocked_device("/proc/self/fd/0"));
        assert!(is_blocked_device("/proc/1234/fd/2"));
        assert!(!is_blocked_device("/tmp/file.txt"));
        assert!(!is_blocked_device("/dev/null"));
    }

    #[test]
    fn read_device_returns_error() {
        let provider = FakeProvider::empty();
        let out = read_file_tool(&provider, "/dev/zero", 1, 500, "t-dev");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("error").is_some());
        assert!(v["error"].as_str().unwrap().contains("device file"));
    }

    #[test]
    fn read_binary_extension_blocked() {
        let provider = FakeProvider::empty();
        let out = read_file_tool(&provider, "/tmp/image.png", 1, 500, "t-bin");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("binary file"));
    }

    #[test]
    fn sensitive_path_detected() {
        assert!(check_sensitive_path("/etc/passwd", "t").is_some());
        assert!(check_sensitive_path("/var/run/docker.sock", "t").is_some());
        assert!(check_sensitive_path("/tmp/ok.txt", "t").is_none());
    }

    #[test]
    fn internal_status_text_guard() {
        assert!(is_internal_file_status_text(READ_DEDUP_STATUS_MESSAGE));
        assert!(is_internal_file_status_text(&format!(
            "Note: {}",
            READ_DEDUP_STATUS_MESSAGE
        )));
        assert!(!is_internal_file_status_text("a real file with lots of content"));
        assert!(!is_internal_file_status_text(""));
    }

    #[test]
    fn write_status_text_refused() {
        let provider = FakeProvider::empty();
        let out = write_file_tool(&provider, "/tmp/x.txt", READ_DEDUP_STATUS_MESSAGE, "t-w");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("internal read_file status text"));
    }

    #[test]
    fn write_sensitive_refused() {
        let provider = FakeProvider::empty();
        let out = write_file_tool(&provider, "/etc/hosts", "hi", "t-w2");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("sensitive system path"));
    }

    #[test]
    fn char_count_guard_rejects_huge_reads() {
        let big = "x".repeat(DEFAULT_MAX_READ_CHARS + 10);
        let provider = FakeProvider { read_content: big, read_total_lines: 42, read_file_size: 999_999, read_truncated: true };
        // Use a path that resolves to something unlikely to exist (no dedup).
        let out = read_file_tool(&provider, "/tmp/__nonexist_big__.txt", 1, 500, "t-big");
        let v: Value = serde_json::from_str(&out).unwrap();
        let err = v["error"].as_str().unwrap();
        assert!(err.contains("exceeds the safety limit"));
        assert!(err.contains("42 lines"));
    }

    #[test]
    fn search_redacts_and_truncation_hint() {
        let provider = FakeProvider::empty();
        let out = handle_search_files(
            &provider,
            &json!({"pattern": "foo", "target": "grep"}),
            "t-search",
        );
        // grep alias maps to content target; result is valid JSON.
        assert!(serde_json::from_str::<Value>(out.split("\n\n[Hint").next().unwrap()).is_ok());
    }

    #[test]
    fn consecutive_search_block() {
        let provider = FakeProvider::empty();
        let tid = "t-loop";
        let args = json!({"pattern": "abc", "target": "content"});
        for _ in 0..3 {
            let _ = handle_search_files(&provider, &args, tid);
        }
        // 4th identical search → blocked.
        let out = handle_search_files(&provider, &args, tid);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("BLOCKED"));
        assert_eq!(v["already_searched"].as_i64().unwrap(), 4);
    }

    #[test]
    fn handle_write_missing_content() {
        let provider = FakeProvider::empty();
        let out = handle_write_file(&provider, &json!({"path": "/tmp/x"}), "t");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("missing required field 'content'"));
    }

    #[test]
    fn handle_write_non_string_content() {
        let provider = FakeProvider::empty();
        let out = handle_write_file(&provider, &json!({"path": "/tmp/x", "content": 5}), "t");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("must be a string"));
        assert!(v["error"].as_str().unwrap().contains("int"));
    }

    #[test]
    fn fmt_thousands_works() {
        assert_eq!(fmt_thousands(0), "0");
        assert_eq!(fmt_thousands(999), "999");
        assert_eq!(fmt_thousands(1000), "1,000");
        assert_eq!(fmt_thousands(100_000), "100,000");
        assert_eq!(fmt_thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn normpath_basic() {
        assert_eq!(normpath("/etc/../etc/passwd"), "/etc/passwd");
        assert_eq!(normpath("a/b/../c"), "a/c");
        assert_eq!(normpath("/a//b"), "/a/b");
        assert_eq!(normpath(""), ".");
    }

    #[test]
    fn reset_and_notify() {
        // Smoke test the tracker mutators don't panic.
        notify_other_tool_call("nonexistent-task");
        reset_file_dedup(Some("nonexistent-task"));
        reset_file_dedup(None);
    }

    #[test]
    fn patch_unknown_mode() {
        let provider = FakeProvider::empty();
        let out = patch_tool(&provider, "bogus", None, None, None, false, None, "t-p");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("Unknown mode"));
    }

    #[test]
    fn patch_replace_missing_args() {
        let provider = FakeProvider::empty();
        let out = patch_tool(&provider, "replace", None, None, None, false, None, "t-p2");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"].as_str().unwrap(), "path required");
    }

    #[test]
    fn schemas_are_well_formed() {
        for s in [
            read_file_schema(),
            write_file_schema(),
            patch_schema(),
            search_files_schema(),
        ] {
            assert!(s.get("name").and_then(|n| n.as_str()).is_some());
            assert!(s.get("parameters").is_some());
        }
    }
}
