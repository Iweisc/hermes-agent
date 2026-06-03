//! Centralized logging setup for Hermes Agent.
//!
//! Native Rust port of `hermes_logging.py`.
//!
//! Provides a single [`setup_logging`] entry point that both the CLI and
//! gateway call early in their startup path. All log files live under
//! `~/.hermes/logs/` (profile-aware via [`crate::mod_hermes_constants::get_hermes_home`]).
//!
//! Log files produced:
//! * `agent.log`   — INFO+, all agent/tool/session activity (the main log)
//! * `errors.log`  — WARNING+, errors and warnings only (quick triage)
//! * `gateway.log` — INFO+, gateway-only events (created when `mode="gateway"`)
//!
//! All files use a rotating-file handler with redaction (via
//! [`crate::agent_redact::redact_log`]) so secrets are never written to disk.
//!
//! ## Component separation
//! `gateway.log` only receives records from `gateway.*` loggers — platform
//! adapters, session management, slash commands, delivery. `agent.log` remains
//! the catch-all (everything goes there).
//!
//! ## Session context
//! Call [`set_session_context`] at the start of a conversation and
//! [`clear_session_context`] when done. All log lines emitted on that thread
//! will include `[session_id]` for filtering/correlation.
//!
//! ## Port notes
//! Python's `logging` module has a global, process-wide hierarchy of loggers,
//! handlers and formatters. Rust's `log` crate has a single global logger.
//! Rather than wiring this into the `log` facade (which is the job of the
//! sibling `logging.rs` module / `crate::logging`), this module faithfully
//! reproduces the *behaviour and bookkeeping* of `hermes_logging.py`:
//!
//! * Log-directory creation, config-file defaults, and the resolved sizes /
//!   backup counts that the Python code computes.
//! * Idempotent handler registration keyed on resolved file path
//!   (`_add_rotating_handler`).
//! * Managed-mode (`is_managed`) group-writable chmod-after-create semantics
//!   (`_ManagedRotatingFileHandler`).
//! * The thread-local session context and the `session_tag` injection used by
//!   the format strings.
//! * The set of noisy third-party loggers to suppress and the verbose-mode
//!   toggle.
//!
//! Handlers here own their own file streams and perform size-based rotation
//! exactly like `RotatingFileHandler`, so emitting through them produces files
//! byte-for-byte comparable to the Python output (modulo the redactor).

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use chrono::Local;

use crate::agent_redact::redact_log;
use crate::mod_hermes_constants::{get_config_path, get_hermes_home};

// ---------------------------------------------------------------------------
// Constants mirroring the Python module
// ---------------------------------------------------------------------------

/// Default log format — includes timestamp, level, optional session tag,
/// logger name, and message. The `session_tag` field is guaranteed to exist on
/// every record (it is computed from the thread-local session context).
pub const LOG_FORMAT: &str = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s";

/// Verbose console format, with a short `%H:%M:%S` timestamp.
pub const LOG_FORMAT_VERBOSE: &str =
    "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s";

/// Third-party loggers that are noisy at DEBUG/INFO level.
pub const NOISY_LOGGERS: [&str; 14] = [
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
];

/// Logger name prefixes that belong to each component.
///
/// Used by the gateway-component filter and exposed for `hermes logs
/// --component`. Mirrors the Python `COMPONENT_PREFIXES` dict.
pub fn component_prefixes() -> HashMap<&'static str, Vec<&'static str>> {
    let mut m = HashMap::new();
    m.insert("gateway", vec!["gateway"]);
    m.insert(
        "agent",
        vec!["agent", "run_agent", "model_tools", "batch_runner"],
    );
    m.insert("tools", vec!["tools"]);
    m.insert("cli", vec!["hermes_cli", "cli"]);
    m.insert("cron", vec!["cron"]);
    m
}

// ---------------------------------------------------------------------------
// Log levels (mirroring Python's `logging` numeric levels)
// ---------------------------------------------------------------------------

/// A standard Python-`logging`-compatible level.
///
/// Numeric values match `logging.DEBUG` (10) … `logging.CRITICAL` (50) so that
/// the `root.level > level` comparisons in the Python source port directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    NotSet,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl LogLevel {
    /// Numeric value matching Python's `logging` levels.
    pub fn as_int(self) -> u32 {
        match self {
            LogLevel::NotSet => 0,
            LogLevel::Debug => 10,
            LogLevel::Info => 20,
            LogLevel::Warning => 30,
            LogLevel::Error => 40,
            LogLevel::Critical => 50,
        }
    }

    /// Parse a level name, mirroring `getattr(logging, level_name, logging.INFO)`.
    /// Unknown names fall back to [`LogLevel::Info`].
    pub fn from_name(name: &str) -> LogLevel {
        match name.to_ascii_uppercase().as_str() {
            "NOTSET" => LogLevel::NotSet,
            "DEBUG" => LogLevel::Debug,
            "INFO" => LogLevel::Info,
            "WARNING" | "WARN" => LogLevel::Warning,
            "ERROR" => LogLevel::Error,
            "CRITICAL" | "FATAL" => LogLevel::Critical,
            _ => LogLevel::Info,
        }
    }

    /// The canonical uppercase level name used in formatted output.
    pub fn name(self) -> &'static str {
        match self {
            LogLevel::NotSet => "NOTSET",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRITICAL",
        }
    }
}

// ---------------------------------------------------------------------------
// Module-global state mirroring the Python module-level globals
// ---------------------------------------------------------------------------

/// Sentinel tracking whether [`setup_logging`] has already run (mirrors
/// `_logging_initialized`). Idempotent — a second call is a no-op unless
/// `force = true`.
static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Thread-local per-conversation session context (mirrors `_session_context`).
    static SESSION_CONTEXT: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// The shared root logger registry, holding the set of attached handlers and
/// the effective root level. Stands in for `logging.getLogger()`.
static ROOT: OnceLock<Mutex<RootLogger>> = OnceLock::new();

fn root() -> &'static Mutex<RootLogger> {
    ROOT.get_or_init(|| Mutex::new(RootLogger::new()))
}

// ---------------------------------------------------------------------------
// Public session context API
// ---------------------------------------------------------------------------

/// Set the session ID for the current thread.
///
/// All subsequent log records on this thread will include `[session_id]` in
/// the formatted output. Call at the start of `run_conversation()`.
pub fn set_session_context(session_id: impl Into<String>) {
    SESSION_CONTEXT.with(|c| *c.borrow_mut() = Some(session_id.into()));
}

/// Clear the session ID for the current thread.
pub fn clear_session_context() {
    SESSION_CONTEXT.with(|c| *c.borrow_mut() = None);
}

/// Return the session tag for the current thread: `" [<sid>]"` or `""`.
///
/// This is the value `_session_record_factory` injects as `record.session_tag`.
pub fn current_session_tag() -> String {
    SESSION_CONTEXT.with(|c| match &*c.borrow() {
        Some(sid) if !sid.is_empty() => format!(" [{sid}]"),
        _ => String::new(),
    })
}

// ---------------------------------------------------------------------------
// Managed-mode detection (mirrors hermes_cli.config.is_managed)
// ---------------------------------------------------------------------------

const MANAGED_TRUE_VALUES: [&str; 3] = ["true", "1", "yes"];

/// Return the package manager owning this install, if any.
///
/// Mirrors `hermes_cli.config.get_managed_system`: honours `HERMES_MANAGED`
/// then falls back to a `.managed` marker file in `HERMES_HOME`.
pub fn get_managed_system() -> Option<String> {
    let raw = std::env::var("HERMES_MANAGED").unwrap_or_default();
    let raw = raw.trim();
    if !raw.is_empty() {
        let normalized = raw.to_ascii_lowercase();
        if MANAGED_TRUE_VALUES.contains(&normalized.as_str()) {
            return Some("NixOS".to_string());
        }
        // Python consults `_MANAGED_SYSTEM_NAMES` here; the marker-file path is
        // the only one that materially affects this module, so we preserve the
        // raw value as the fallback name (matching `.get(normalized, raw)`).
        return Some(raw.to_string());
    }

    let managed_marker = get_hermes_home().join(".managed");
    if managed_marker.exists() {
        return Some("NixOS".to_string());
    }
    None
}

/// Check if Hermes is running in package-manager-managed mode.
pub fn is_managed() -> bool {
    get_managed_system().is_some()
}

// ---------------------------------------------------------------------------
// Component filter
// ---------------------------------------------------------------------------

/// Only pass records whose logger name starts with one of `prefixes`.
///
/// Mirrors the Python `_ComponentFilter`, used to route gateway-specific
/// records to `gateway.log`.
#[derive(Debug, Clone)]
pub struct ComponentFilter {
    prefixes: Vec<String>,
}

impl ComponentFilter {
    pub fn new(prefixes: impl IntoIterator<Item = impl Into<String>>) -> ComponentFilter {
        ComponentFilter {
            prefixes: prefixes.into_iter().map(Into::into).collect(),
        }
    }

    /// `True` if `name` starts with any configured prefix
    /// (`str.startswith(tuple)` semantics).
    pub fn passes(&self, name: &str) -> bool {
        self.prefixes.iter().any(|p| name.starts_with(p.as_str()))
    }
}

// ---------------------------------------------------------------------------
// Rotating file handler (mirrors RotatingFileHandler + _ManagedRotatingFileHandler)
// ---------------------------------------------------------------------------

/// A size-rotating, redacting file handler.
///
/// Behaviourally equivalent to Python's `RotatingFileHandler` wrapped in
/// `_ManagedRotatingFileHandler`: it rotates when the file would exceed
/// `max_bytes`, keeps `backup_count` numbered backups, formats records with the
/// supplied format string, redacts the formatted line, and (in managed mode)
/// chmods new files to `0o660`.
pub struct RotatingFileHandler {
    base_filename: PathBuf,
    /// Fully-resolved path used for idempotent dedup, mirroring
    /// `Path(handler.baseFilename).resolve()`.
    resolved: PathBuf,
    max_bytes: u64,
    backup_count: u32,
    level: LogLevel,
    format: String,
    /// `%H:%M:%S`-style short timestamp when `Some`, else default `asctime`.
    short_time: bool,
    filter: Option<ComponentFilter>,
    managed: bool,
    /// Marker mirroring `handler._hermes_verbose`.
    verbose: bool,
}

/// Resolve a path for dedup purposes. Falls back to a normalised absolute path
/// when the file does not yet exist (`Path.resolve()` in Python does not
/// require existence).
fn resolve_path(path: &Path) -> PathBuf {
    match fs::canonicalize(path) {
        Ok(p) => p,
        Err(_) => {
            // Best-effort: join with cwd if relative, then leave as-is.
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()
                    .map(|c| c.join(path))
                    .unwrap_or_else(|_| path.to_path_buf())
            }
        }
    }
}

impl RotatingFileHandler {
    fn new(
        path: &Path,
        level: LogLevel,
        max_bytes: u64,
        backup_count: u32,
        format: &str,
        short_time: bool,
        filter: Option<ComponentFilter>,
    ) -> RotatingFileHandler {
        RotatingFileHandler {
            base_filename: path.to_path_buf(),
            resolved: resolve_path(path),
            max_bytes,
            backup_count,
            level,
            format: format.to_string(),
            short_time,
            filter,
            managed: is_managed(),
            verbose: false,
        }
    }

    /// Apply group-writable perms in managed mode (mirrors `_chmod_if_managed`).
    #[cfg(unix)]
    fn chmod_if_managed(&self) {
        if self.managed {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.base_filename, fs::Permissions::from_mode(0o660));
        }
    }

    #[cfg(not(unix))]
    fn chmod_if_managed(&self) {}

    /// Whether this record passes the handler level + optional component filter.
    fn accepts(&self, level: LogLevel, logger_name: &str) -> bool {
        if (level.as_int()) < self.level.as_int() {
            return false;
        }
        match &self.filter {
            Some(f) => f.passes(logger_name),
            None => true,
        }
    }

    /// Format a record the way the Python `Formatter` + `session_tag` would,
    /// then redact it. Returns the line *without* the trailing newline.
    fn format_record(&self, level: LogLevel, logger_name: &str, message: &str) -> String {
        let now = Local::now();
        let asctime = if self.short_time {
            now.format("%H:%M:%S").to_string()
        } else {
            // Python's default asctime: "YYYY-MM-DD HH:MM:SS,mmm".
            now.format("%Y-%m-%d %H:%M:%S,%3f").to_string()
        };
        let session_tag = current_session_tag();
        let line = self
            .format
            .replace("%(asctime)s", &asctime)
            .replace("%(levelname)s", level.name())
            .replace("%(session_tag)s", &session_tag)
            .replace("%(name)s", logger_name)
            .replace("%(message)s", message);
        redact_log(&line)
    }

    /// Whether writing `msg_len` more bytes should trigger a rollover, matching
    /// `RotatingFileHandler.shouldRollover`.
    fn should_rollover(&self, msg_len: u64) -> bool {
        if self.max_bytes == 0 {
            return false;
        }
        let cur = fs::metadata(&self.base_filename).map(|m| m.len()).unwrap_or(0);
        cur + msg_len >= self.max_bytes
    }

    /// Perform the numbered-suffix rollover (`.1`, `.2`, …), matching
    /// `RotatingFileHandler.doRollover`.
    fn do_rollover(&self) {
        if self.backup_count > 0 {
            for i in (1..self.backup_count).rev() {
                let sfn = self.suffixed(i);
                let dfn = self.suffixed(i + 1);
                if sfn.exists() {
                    let _ = fs::remove_file(&dfn);
                    let _ = fs::rename(&sfn, &dfn);
                }
            }
            let dfn = self.suffixed(1);
            let _ = fs::remove_file(&dfn);
            if self.base_filename.exists() {
                let _ = fs::rename(&self.base_filename, &dfn);
            }
        }
        self.chmod_if_managed();
    }

    fn suffixed(&self, n: u32) -> PathBuf {
        let mut name = self.base_filename.as_os_str().to_os_string();
        name.push(format!(".{n}"));
        PathBuf::from(name)
    }

    /// Emit a record: format + redact + (rollover) + append, mirroring
    /// `FileHandler.emit`/`RotatingFileHandler.emit`.
    fn emit(&self, level: LogLevel, logger_name: &str, message: &str) {
        if !self.accepts(level, logger_name) {
            return;
        }
        let mut line = self.format_record(level, logger_name, message);
        line.push('\n');
        let bytes = line.as_bytes();

        let exists_before = self.base_filename.exists();
        if self.should_rollover(bytes.len() as u64) {
            self.do_rollover();
        }

        if let Some(parent) = self.base_filename.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.base_filename)
        {
            let _ = f.write_all(bytes);
            // chmod on initial creation, mirroring _ManagedRotatingFileHandler._open.
            if !exists_before {
                self.chmod_if_managed();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Root logger registry
// ---------------------------------------------------------------------------

/// Stands in for the global `logging.getLogger()` root: its attached handlers,
/// its effective level, and the per-logger level overrides (used for the noisy
/// third-party suppression).
pub struct RootLogger {
    handlers: Vec<RotatingFileHandler>,
    level: LogLevel,
    /// Per-named-logger level overrides, mirroring
    /// `logging.getLogger(name).setLevel(...)`.
    named_levels: HashMap<String, LogLevel>,
}

impl RootLogger {
    fn new() -> RootLogger {
        RootLogger {
            handlers: Vec::new(),
            // A fresh root logger is at WARNING in Python; NOTSET here matches
            // the `root.level == logging.NOTSET` branch in setup_logging.
            level: LogLevel::NotSet,
            named_levels: HashMap::new(),
        }
    }

    /// Add a handler unless one is already attached for the same resolved path
    /// (mirrors the dedup loop in `_add_rotating_handler`).
    fn add_rotating_handler(&mut self, handler: RotatingFileHandler) {
        for existing in &self.handlers {
            if existing.resolved == handler.resolved {
                return;
            }
        }
        self.handlers.push(handler);
    }

    fn set_named_level(&mut self, name: &str, level: LogLevel) {
        self.named_levels.insert(name.to_string(), level);
    }
}

// ---------------------------------------------------------------------------
// Internal helper: _add_rotating_handler
// ---------------------------------------------------------------------------

/// Add a rotating handler to the root logger, skipping if one already exists
/// for the same resolved file path (idempotent). Mirrors
/// `_add_rotating_handler`.
#[allow(clippy::too_many_arguments)]
fn add_rotating_handler(
    root_logger: &mut RootLogger,
    path: &Path,
    level: LogLevel,
    max_bytes: u64,
    backup_count: u32,
    format: &str,
    short_time: bool,
    log_filter: Option<ComponentFilter>,
) {
    let resolved = resolve_path(path);
    for existing in &root_logger.handlers {
        if existing.resolved == resolved {
            return; // already attached
        }
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let handler = RotatingFileHandler::new(
        path,
        level,
        max_bytes,
        backup_count,
        format,
        short_time,
        log_filter,
    );
    root_logger.add_rotating_handler(handler);
}

// ---------------------------------------------------------------------------
// _read_logging_config
// ---------------------------------------------------------------------------

/// Best-effort read of `logging.*` from `config.yaml`.
///
/// Returns `(level, max_size_mb, backup_count)` — any may be `None`. Mirrors
/// `_read_logging_config`.
pub fn read_logging_config() -> (Option<String>, Option<u64>, Option<u32>) {
    let config_path = get_config_path();
    if !config_path.exists() {
        return (None, None, None);
    }
    let Ok(text) = fs::read_to_string(&config_path) else {
        return (None, None, None);
    };
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&text) else {
        return (None, None, None);
    };
    let log_cfg = value.get("logging");
    let Some(log_cfg) = log_cfg else {
        return (None, None, None);
    };
    if !log_cfg.is_mapping() {
        return (None, None, None);
    }
    let level = log_cfg
        .get("level")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let max_size_mb = log_cfg.get("max_size_mb").and_then(|v| v.as_u64());
    let backup_count = log_cfg
        .get("backup_count")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    (level, max_size_mb, backup_count)
}

// ---------------------------------------------------------------------------
// setup_logging
// ---------------------------------------------------------------------------

/// Options for [`setup_logging`], mirroring the keyword-only Python parameters.
#[derive(Debug, Default, Clone)]
pub struct SetupOptions {
    /// Override for the Hermes home directory (falls back to
    /// [`get_hermes_home`]).
    pub hermes_home: Option<PathBuf>,
    /// Minimum level for the `agent.log` handler (e.g. `"DEBUG"`, `"INFO"`).
    pub log_level: Option<String>,
    /// Max size of each log file in megabytes before rotation.
    pub max_size_mb: Option<u64>,
    /// Number of rotated backup files to keep.
    pub backup_count: Option<u32>,
    /// Caller context: `"cli"`, `"gateway"`, `"cron"`. `"gateway"` adds a
    /// `gateway.log` receiving only gateway-component records.
    pub mode: Option<String>,
    /// Re-run setup even if it has already been called.
    pub force: bool,
}

/// Configure the Hermes logging subsystem.
///
/// Safe to call multiple times — the second call is a no-op unless
/// `opts.force` is `true`. Returns the `logs/` directory where files are
/// written. Mirrors `setup_logging`.
pub fn setup_logging(opts: &SetupOptions) -> std::io::Result<PathBuf> {
    let home = opts
        .hermes_home
        .clone()
        .unwrap_or_else(get_hermes_home);
    let log_dir = home.join("logs");
    fs::create_dir_all(&log_dir)?;

    // Read config defaults (best-effort — config may not be loaded yet).
    let (cfg_level, cfg_max_size, cfg_backup) = read_logging_config();

    let level_name = opts
        .log_level
        .clone()
        .or(cfg_level)
        .unwrap_or_else(|| "INFO".to_string())
        .to_ascii_uppercase();
    let level = LogLevel::from_name(&level_name);
    let max_bytes = opts
        .max_size_mb
        .or(cfg_max_size)
        .filter(|v| *v != 0)
        .unwrap_or(5)
        * 1024
        * 1024;
    let backups = opts
        .backup_count
        .or(cfg_backup)
        .filter(|v| *v != 0)
        .unwrap_or(3);

    let r = root();
    let mut guard = r.lock().expect("root logger mutex poisoned");

    // --- agent.log (INFO+) — the main activity log ---
    add_rotating_handler(
        &mut guard,
        &log_dir.join("agent.log"),
        level,
        max_bytes,
        backups,
        LOG_FORMAT,
        false,
        None,
    );

    // --- errors.log (WARNING+) — quick triage log ---
    add_rotating_handler(
        &mut guard,
        &log_dir.join("errors.log"),
        LogLevel::Warning,
        2 * 1024 * 1024,
        2,
        LOG_FORMAT,
        false,
        None,
    );

    // --- gateway.log (INFO+, gateway component only) ---
    if opts.mode.as_deref() == Some("gateway") {
        let prefixes = component_prefixes();
        let gw = prefixes.get("gateway").cloned().unwrap_or_default();
        add_rotating_handler(
            &mut guard,
            &log_dir.join("gateway.log"),
            LogLevel::Info,
            5 * 1024 * 1024,
            3,
            LOG_FORMAT,
            false,
            Some(ComponentFilter::new(gw)),
        );
    }

    if LOGGING_INITIALIZED.load(Ordering::SeqCst) && !opts.force {
        return Ok(log_dir);
    }

    // Ensure root logger level is low enough for the handlers to fire.
    if guard.level == LogLevel::NotSet || guard.level.as_int() > level.as_int() {
        guard.level = level;
    }

    // Suppress noisy third-party loggers.
    for name in NOISY_LOGGERS {
        guard.set_named_level(name, LogLevel::Warning);
    }

    LOGGING_INITIALIZED.store(true, Ordering::SeqCst);
    Ok(log_dir)
}

/// Enable DEBUG-level console logging for `--verbose` / `-v` mode.
///
/// Called by the agent constructor when verbose logging is requested. Mirrors
/// `setup_verbose_logging`: it lowers the root level to DEBUG, suppresses noisy
/// third-party loggers, and keeps `rex-deploy` at INFO. Console output is
/// represented here by a verbose-marked handler so the bookkeeping (and the
/// duplicate-handler guard) ports faithfully.
pub fn setup_verbose_logging() {
    let r = root();
    let mut guard = r.lock().expect("root logger mutex poisoned");

    // Avoid adding duplicate verbose console handlers.
    if guard.handlers.iter().any(|h| h.verbose) {
        return;
    }

    // A "stream" handler is modelled as a verbose-marked handler. It writes to
    // a console-stand-in path under the logs dir only as a structural mirror;
    // the important behaviour for ports is the level/marker bookkeeping.
    let mut handler = RotatingFileHandler::new(
        Path::new(""), // no file backing — emit() is a structural mirror
        LogLevel::Debug,
        0,
        0,
        LOG_FORMAT_VERBOSE,
        true,
        None,
    );
    handler.verbose = true;
    handler.resolved = PathBuf::from(format!("<verbose-stream-{}>", guard.handlers.len()));
    guard.handlers.push(handler);

    // Lower root logger level so DEBUG records reach all handlers.
    if guard.level.as_int() > LogLevel::Debug.as_int() {
        guard.level = LogLevel::Debug;
    }

    // Keep third-party libraries at WARNING to reduce noise.
    for name in NOISY_LOGGERS {
        guard.set_named_level(name, LogLevel::Warning);
    }
    // rex-deploy at INFO for sandbox status.
    guard.set_named_level("rex-deploy", LogLevel::Info);
}

// ---------------------------------------------------------------------------
// Emit API
// ---------------------------------------------------------------------------

/// Emit a record through the configured root handlers, applying each handler's
/// level + component filter and the effective per-logger level override.
///
/// This is the analogue of `logging.getLogger(name).log(level, msg)` flowing to
/// the root handlers. It exists so other ported modules can drive the same
/// files the Python code would.
pub fn emit(level: LogLevel, logger_name: &str, message: &str) {
    let r = root();
    let guard = r.lock().expect("root logger mutex poisoned");

    // Effective level: the most specific named override that is a prefix of the
    // logger name, else the root level (mirrors logging's hierarchical
    // effective-level lookup for the suppression entries we install).
    let effective = effective_level(&guard, logger_name);
    if level.as_int() < effective.as_int() {
        return;
    }
    for handler in &guard.handlers {
        handler.emit(level, logger_name, message);
    }
}

/// Resolve the effective level for `logger_name`, honouring named overrides
/// (longest matching ancestor wins, as in the `logging` hierarchy) and falling
/// back to the root level (treating NOTSET as WARNING like Python).
fn effective_level(root_logger: &RootLogger, logger_name: &str) -> LogLevel {
    let mut best: Option<(usize, LogLevel)> = None;
    for (name, lvl) in &root_logger.named_levels {
        if logger_name == name || logger_name.starts_with(&format!("{name}.")) {
            let len = name.len();
            if best.map(|(l, _)| len > l).unwrap_or(true) {
                best = Some((len, *lvl));
            }
        }
    }
    if let Some((_, lvl)) = best {
        return lvl;
    }
    match root_logger.level {
        LogLevel::NotSet => LogLevel::Warning,
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_level_numeric_values_match_python() {
        assert_eq!(LogLevel::NotSet.as_int(), 0);
        assert_eq!(LogLevel::Debug.as_int(), 10);
        assert_eq!(LogLevel::Info.as_int(), 20);
        assert_eq!(LogLevel::Warning.as_int(), 30);
        assert_eq!(LogLevel::Error.as_int(), 40);
        assert_eq!(LogLevel::Critical.as_int(), 50);
    }

    #[test]
    fn level_from_name_fallback_to_info() {
        assert_eq!(LogLevel::from_name("debug"), LogLevel::Debug);
        assert_eq!(LogLevel::from_name("WARNING"), LogLevel::Warning);
        assert_eq!(LogLevel::from_name("bogus"), LogLevel::Info);
        assert_eq!(LogLevel::from_name(""), LogLevel::Info);
    }

    #[test]
    fn session_context_roundtrip() {
        clear_session_context();
        assert_eq!(current_session_tag(), "");
        set_session_context("sess-123");
        assert_eq!(current_session_tag(), " [sess-123]");
        // Empty string behaves like cleared, matching `if sid else ""`.
        set_session_context("");
        assert_eq!(current_session_tag(), "");
        set_session_context("sess-9");
        clear_session_context();
        assert_eq!(current_session_tag(), "");
    }

    #[test]
    fn component_filter_startswith_semantics() {
        let f = ComponentFilter::new(["gateway"]);
        assert!(f.passes("gateway"));
        assert!(f.passes("gateway.telegram"));
        assert!(!f.passes("agent.run"));

        let prefixes = component_prefixes();
        assert_eq!(prefixes["gateway"], vec!["gateway"]);
        assert!(prefixes["agent"].contains(&"batch_runner"));
        assert_eq!(prefixes["cli"], vec!["hermes_cli", "cli"]);
    }

    #[test]
    fn managed_system_honours_env_true_values() {
        unsafe {
            std::env::set_var("HERMES_MANAGED", "yes");
        }
        assert_eq!(get_managed_system().as_deref(), Some("NixOS"));
        assert!(is_managed());

        unsafe {
            std::env::set_var("HERMES_MANAGED", "apt");
        }
        assert_eq!(get_managed_system().as_deref(), Some("apt"));

        unsafe {
            std::env::remove_var("HERMES_MANAGED");
        }
    }

    #[test]
    fn rotating_handler_dedup_by_resolved_path() {
        let dir = std::env::temp_dir().join(format!("hermes_log_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let mut rl = RootLogger::new();
        let path = dir.join("agent.log");
        add_rotating_handler(&mut rl, &path, LogLevel::Info, 1024, 1, LOG_FORMAT, false, None);
        add_rotating_handler(&mut rl, &path, LogLevel::Info, 1024, 1, LOG_FORMAT, false, None);
        assert_eq!(rl.handlers.len(), 1, "duplicate handler must be skipped");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn handler_accepts_respects_level_and_filter() {
        let h = RotatingFileHandler::new(
            Path::new("/tmp/x.log"),
            LogLevel::Warning,
            0,
            0,
            LOG_FORMAT,
            false,
            Some(ComponentFilter::new(["gateway"])),
        );
        // below level
        assert!(!h.accepts(LogLevel::Info, "gateway.foo"));
        // at level but wrong component
        assert!(!h.accepts(LogLevel::Warning, "agent.foo"));
        // at level + right component
        assert!(h.accepts(LogLevel::Error, "gateway.foo"));
    }

    #[test]
    fn format_record_substitutes_all_fields() {
        clear_session_context();
        set_session_context("S1");
        let h = RotatingFileHandler::new(
            Path::new("/tmp/y.log"),
            LogLevel::Info,
            0,
            0,
            LOG_FORMAT,
            false,
            None,
        );
        let line = h.format_record(LogLevel::Info, "agent.run", "hello world");
        assert!(line.contains("INFO"));
        assert!(line.contains(" [S1]"));
        assert!(line.contains("agent.run:"));
        assert!(line.contains("hello world"));
        assert!(!line.contains("%("), "no unsubstituted placeholders: {line}");
        clear_session_context();
    }

    #[test]
    fn rotating_handler_emits_and_rotates() {
        let dir = std::env::temp_dir().join(format!("hermes_rot_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("agent.log");
        let h = RotatingFileHandler::new(&path, LogLevel::Info, 64, 2, LOG_FORMAT, false, None);
        for i in 0..20 {
            h.emit(LogLevel::Info, "agent", &format!("message number {i}"));
        }
        // Base file exists, and at least one backup was produced.
        assert!(path.exists());
        let backup1 = {
            let mut n = path.as_os_str().to_os_string();
            n.push(".1");
            PathBuf::from(n)
        };
        assert!(backup1.exists(), "expected rotated backup .1 to exist");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_logging_config_missing_is_none() {
        // With a nonexistent config path (via a temp HERMES_HOME), all None.
        let tmp = std::env::temp_dir().join(format!("hermes_cfg_none_{}", std::process::id()));
        let _ = fs::create_dir_all(&tmp);
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let (lvl, max, bk) = read_logging_config();
        assert!(lvl.is_none() && max.is_none() && bk.is_none());
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = fs::remove_dir_all(&tmp);
    }
}
