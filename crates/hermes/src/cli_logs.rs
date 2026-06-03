//! `hermes logs` — view and filter Hermes log files.
//!
//! Supports tailing, following, session filtering, level filtering,
//! component filtering, and relative time ranges. All log files live
//! under `~/.hermes/logs/`.
//!
//! Usage examples:
//!
//! ```text
//! hermes logs                    # last 50 lines of agent.log
//! hermes logs -f                 # follow agent.log in real time
//! hermes logs errors             # last 50 lines of errors.log
//! hermes logs gateway -n 100     # last 100 lines of gateway.log
//! hermes logs --level WARNING    # only WARNING+ lines
//! hermes logs --session abc123   # filter by session ID substring
//! hermes logs --component tools  # only tool-related lines
//! hermes logs --since 1h         # lines from the last hour
//! hermes logs --since 30m -f     # follow, starting 30 min ago
//! ```
//!
//! This is a faithful native Rust port of `hermes_cli/logs.py`.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Local, NaiveDateTime, TimeZone};
use regex::Regex;

// ---------------------------------------------------------------------------
// Dependency shims
// ---------------------------------------------------------------------------
//
// The Python module imports `get_hermes_home` / `display_hermes_home` from
// `hermes_constants` and `COMPONENT_PREFIXES` from `hermes_logging`. The
// home-dir helpers are already ported in `crate::cli_backup`; we re-export
// thin local wrappers so this module compiles standalone and stays in sync.

/// Return the Hermes home directory (`HERMES_HOME` env var, else `~/.hermes`).
fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
}

/// Return a user-friendly display string for the current `HERMES_HOME`.
///
/// Uses `~/` shorthand for readability when the home dir is under the
/// user's home directory.
fn display_hermes_home() -> String {
    let home = get_hermes_home();
    if let Some(user_home) = dirs::home_dir() {
        if let Ok(rel) = home.strip_prefix(&user_home) {
            return format!("~/{}", rel.display());
        }
    }
    home.display().to_string()
}

/// Component name → logger-name prefixes, mirroring
/// `hermes_logging.COMPONENT_PREFIXES`.
fn component_prefixes(component_lower: &str) -> Option<&'static [&'static str]> {
    match component_lower {
        "gateway" => Some(&["gateway"]),
        "agent" => Some(&["agent", "run_agent", "model_tools", "batch_runner"]),
        "tools" => Some(&["tools"]),
        "cli" => Some(&["hermes_cli", "cli"]),
        "cron" => Some(&["cron"]),
        _ => None,
    }
}

/// Sorted list of known component names (for error messages).
fn component_names() -> Vec<&'static str> {
    let mut v = vec!["agent", "cli", "cron", "gateway", "tools"];
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------------
// Known log files (name → filename)
// ---------------------------------------------------------------------------

/// Resolve a log alias to its filename (`agent`/`errors`/`gateway`).
pub fn log_filename(log_name: &str) -> Option<&'static str> {
    match log_name {
        "agent" => Some("agent.log"),
        "errors" => Some("errors.log"),
        "gateway" => Some("gateway.log"),
        _ => None,
    }
}

/// Sorted list of available log aliases.
fn log_aliases() -> Vec<&'static str> {
    let mut v = vec!["agent", "errors", "gateway"];
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------------
// Regexes
// ---------------------------------------------------------------------------

// Log line timestamp regex — matches "2026-04-05 22:35:00,123" or
// "2026-04-05 22:35:00" at the start of a line.
static TS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})").unwrap());

// Level extraction — matches " INFO ", " WARNING ", " ERROR ", " DEBUG ", " CRITICAL ".
static LEVEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s").unwrap());

// Logger name extraction — after level and optional session tag, the next
// non-space token before ":" is the logger name.
// Matches: "INFO gateway.run:" or "INFO [sess_abc] tools.terminal_tool:".
static LOGGER_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\s(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)(?:\s+\[.*?\])?\s+(\S+):").unwrap()
});

// Relative time string like "1h", "30m", "2d".
static SINCE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+)\s*([smhd])$").unwrap());

/// Level ordering for `>=` filtering.
fn level_order(level: &str) -> i32 {
    match level {
        "DEBUG" => 0,
        "INFO" => 1,
        "WARNING" => 2,
        "ERROR" => 3,
        "CRITICAL" => 4,
        _ => -1,
    }
}

/// Whether a level name is a valid level.
fn is_valid_level(level: &str) -> bool {
    matches!(level, "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL")
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Parse a relative time string like `1h`, `30m`, `2d` into a naive local
/// datetime cutoff. Returns `None` if the string can't be parsed.
pub fn parse_since(since_str: &str) -> Option<NaiveDateTime> {
    let s = since_str.trim().to_lowercase();
    let caps = SINCE_RE.captures(&s)?;
    let value: i64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str();
    let delta = match unit {
        "s" => chrono::Duration::seconds(value),
        "m" => chrono::Duration::minutes(value),
        "h" => chrono::Duration::hours(value),
        "d" => chrono::Duration::days(value),
        _ => return None,
    };
    let now = Local::now().naive_local();
    Some(now - delta)
}

/// Extract the leading timestamp from a log line. Returns `None` if not
/// parseable. Matches Python's `datetime.strptime(..., "%Y-%m-%d %H:%M:%S")`
/// on the captured `YYYY-MM-DD HH:MM:SS` portion (whitespace between date and
/// time is normalised to a single space, matching `\s+` capture semantics).
pub fn parse_line_timestamp(line: &str) -> Option<NaiveDateTime> {
    let caps = TS_RE.captures(line)?;
    let raw = caps.get(1)?.as_str();
    // The captured group may contain arbitrary whitespace between date and
    // time (the regex uses `\s+`). Normalise to a single space so strptime
    // succeeds the same way Python's `%Y-%m-%d %H:%M:%S` does (which treats a
    // run of whitespace in the format as matching a run of whitespace).
    let normalized = normalize_ts_whitespace(raw);
    NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M:%S").ok()
}

/// Collapse the whitespace run between the date and time parts into a single
/// space, so `chrono`'s `%Y-%m-%d %H:%M:%S` parse succeeds the same way
/// Python's `strptime` does (which treats a space in the format as matching a
/// run of whitespace). The capture group is always `<date><ws><time>`.
fn normalize_ts_whitespace(raw: &str) -> String {
    if let Some(ws_start) = raw.find(char::is_whitespace) {
        let after = &raw[ws_start..];
        let ws_len: usize = after
            .chars()
            .take_while(|c| c.is_whitespace())
            .map(char::len_utf8)
            .sum();
        let time_start = ws_start + ws_len;
        return format!("{} {}", &raw[..ws_start], &raw[time_start..]);
    }
    raw.to_string()
}

/// Extract the log level from a line.
pub fn extract_level(line: &str) -> Option<String> {
    LEVEL_RE
        .captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// Extract the logger name from a log line.
pub fn extract_logger_name(line: &str) -> Option<String> {
    LOGGER_NAME_RE
        .captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// Check whether a log line's logger name starts with any of `prefixes`.
fn line_matches_component(line: &str, prefixes: &[&str]) -> bool {
    match extract_logger_name(line) {
        Some(name) => prefixes.iter().any(|p| name.starts_with(p)),
        None => false,
    }
}

/// Set of active filters applied to log lines.
#[derive(Default, Clone)]
pub struct Filters {
    pub min_level: Option<String>,
    pub session_filter: Option<String>,
    pub since: Option<NaiveDateTime>,
    pub component_prefixes: Option<Vec<&'static str>>,
}

impl Filters {
    /// Whether any filter is active.
    pub fn has_filters(&self) -> bool {
        self.min_level.is_some()
            || self.session_filter.is_some()
            || self.since.is_some()
            || self.component_prefixes.is_some()
    }
}

/// Check if a log line passes all active filters.
pub fn matches_filters(line: &str, f: &Filters) -> bool {
    if let Some(since) = f.since {
        if let Some(ts) = parse_line_timestamp(line) {
            if ts < since {
                return false;
            }
        }
    }

    if let Some(ref min_level) = f.min_level {
        if let Some(level) = extract_level(line) {
            // Python: _LEVEL_ORDER.get(level, 0) — unknown levels map to 0.
            if level_order(&level).max(0) < level_order(min_level).max(0) {
                return false;
            }
        }
    }

    if let Some(ref session) = f.session_filter {
        if !line.contains(session.as_str()) {
            return false;
        }
    }

    if let Some(ref prefixes) = f.component_prefixes {
        if !line_matches_component(line, prefixes) {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Errors / result of tailing
// ---------------------------------------------------------------------------

/// Error conditions that mirror the Python CLI's `print(...) + sys.exit(1)`
/// behavior. The caller is responsible for printing and exiting; this keeps
/// the core logic testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogError {
    UnknownLog { name: String, available: String },
    LogFileNotFound { path: String, hint: String },
    InvalidSince { value: String },
    InvalidLevel { value: String },
    UnknownComponent { name: String, available: String },
    PermissionDenied { path: String },
}

impl LogError {
    /// Render the user-facing message(s), one per line (matching Python's
    /// multiple `print(...)` calls).
    pub fn messages(&self) -> Vec<String> {
        match self {
            LogError::UnknownLog { name, available } => vec![format!(
                "Unknown log: '{}'. Available: {}",
                name, available
            )],
            LogError::LogFileNotFound { path, hint } => {
                vec![format!("Log file not found: {}", path), hint.clone()]
            }
            LogError::InvalidSince { value } => vec![format!(
                "Invalid --since value: '{}'. Use format like '1h', '30m', '2d'.",
                value
            )],
            LogError::InvalidLevel { value } => vec![format!(
                "Invalid --level: '{}'. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
                value
            )],
            LogError::UnknownComponent { name, available } => vec![format!(
                "Unknown component: '{}'. Available: {}",
                name, available
            )],
            LogError::PermissionDenied { path } => {
                vec![format!("Permission denied: {}", path)]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public tail entry point
// ---------------------------------------------------------------------------

/// Options for [`tail_log`].
pub struct TailOptions {
    pub log_name: String,
    pub num_lines: usize,
    pub follow: bool,
    pub level: Option<String>,
    pub session: Option<String>,
    pub since: Option<String>,
    pub component: Option<String>,
}

impl Default for TailOptions {
    fn default() -> Self {
        TailOptions {
            log_name: "agent".to_string(),
            num_lines: 50,
            follow: false,
            level: None,
            session: None,
            since: None,
            component: None,
        }
    }
}

/// Read and display log lines, optionally following in real time.
///
/// On a fatal user-error condition this prints the relevant message(s) and
/// returns `Err(LogError)` so the caller can `process::exit(1)`, mirroring the
/// Python CLI's `print(...) + sys.exit(1)`.
pub fn tail_log(opts: &TailOptions) -> Result<(), LogError> {
    let filename = match log_filename(&opts.log_name) {
        Some(f) => f,
        None => {
            let err = LogError::UnknownLog {
                name: opts.log_name.clone(),
                available: log_aliases().join(", "),
            };
            print_messages(&err);
            return Err(err);
        }
    };

    let log_path = get_hermes_home().join("logs").join(filename);
    if !log_path.exists() {
        let err = LogError::LogFileNotFound {
            path: log_path.display().to_string(),
            hint: "(Logs are created when Hermes runs — try 'hermes chat' first)".to_string(),
        };
        print_messages(&err);
        return Err(err);
    }

    // Parse --since into a datetime cutoff.
    let mut since_dt = None;
    if let Some(ref since) = opts.since {
        if !since.is_empty() {
            match parse_since(since) {
                Some(dt) => since_dt = Some(dt),
                None => {
                    let err = LogError::InvalidSince {
                        value: since.clone(),
                    };
                    print_messages(&err);
                    return Err(err);
                }
            }
        }
    }

    let min_level = opts.level.as_ref().map(|l| l.to_uppercase());
    if let Some(ref ml) = min_level {
        if !is_valid_level(ml) {
            let err = LogError::InvalidLevel {
                value: opts.level.clone().unwrap_or_default(),
            };
            print_messages(&err);
            return Err(err);
        }
    }

    // Resolve component to logger name prefixes.
    let mut comp_prefixes: Option<Vec<&'static str>> = None;
    if let Some(ref component) = opts.component {
        if !component.is_empty() {
            let component_lower = component.to_lowercase();
            match component_prefixes(&component_lower) {
                Some(prefixes) => comp_prefixes = Some(prefixes.to_vec()),
                None => {
                    let err = LogError::UnknownComponent {
                        name: component.clone(),
                        available: component_names().join(", "),
                    };
                    print_messages(&err);
                    return Err(err);
                }
            }
        }
    }

    let filters = Filters {
        min_level: min_level.clone(),
        session_filter: opts.session.clone(),
        since: since_dt,
        component_prefixes: comp_prefixes.clone(),
    };

    // Read and display the tail.
    let lines = match read_tail(&log_path, opts.num_lines, &filters) {
        Ok(l) => l,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                let err = LogError::PermissionDenied {
                    path: log_path.display().to_string(),
                };
                print_messages(&err);
                return Err(err);
            }
            // Other IO errors: treat as empty (Python's fallback re-reads, but
            // a genuinely unreadable file would also surface here). Be lenient.
            Vec::new()
        }
    };

    // Print header.
    let mut filter_parts: Vec<String> = Vec::new();
    if let Some(ref ml) = min_level {
        filter_parts.push(format!("level>={}", ml));
    }
    if let Some(ref s) = opts.session {
        filter_parts.push(format!("session={}", s));
    }
    if let Some(ref c) = opts.component {
        filter_parts.push(format!("component={}", c));
    }
    if let Some(ref s) = opts.since {
        filter_parts.push(format!("since={}", s));
    }
    let filter_desc = if filter_parts.is_empty() {
        String::new()
    } else {
        format!(" [{}]", filter_parts.join(", "))
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if opts.follow {
        let _ = writeln!(
            out,
            "--- {}/logs/{}{} (Ctrl+C to stop) ---",
            display_hermes_home(),
            filename,
            filter_desc
        );
    } else {
        let _ = writeln!(
            out,
            "--- {}/logs/{}{} (last {}) ---",
            display_hermes_home(),
            filename,
            filter_desc,
            opts.num_lines
        );
    }

    for line in &lines {
        // Python prints with end="" — the stored lines retain their trailing
        // newline, so emit verbatim.
        let _ = write!(out, "{}", line);
    }
    let _ = out.flush();

    if !opts.follow {
        return Ok(());
    }

    // Follow mode — poll for new content. Ctrl+C terminates the process
    // (the Python version catches KeyboardInterrupt and prints "--- stopped
    // ---"; in Rust the default SIGINT handler exits, which is acceptable for
    // a CLI). This loop runs until the process is interrupted.
    follow_log(&log_path, &filters);

    Ok(())
}

fn print_messages(err: &LogError) {
    for m in err.messages() {
        println!("{}", m);
    }
}

// ---------------------------------------------------------------------------
// Tail reading
// ---------------------------------------------------------------------------

/// Read the last `num_lines` matching lines from a log file.
///
/// When filters are active, reads more raw lines to find enough matches.
pub fn read_tail(
    path: &Path,
    num_lines: usize,
    filters: &Filters,
) -> std::io::Result<Vec<String>> {
    if filters.has_filters() {
        // Read more lines to ensure we get enough after filtering.
        // For large files, read last 10K-ish lines and filter down.
        let want = std::cmp::max(num_lines.saturating_mul(20), 2000);
        let raw_lines = read_last_n_lines(path, want)?;
        let filtered: Vec<String> = raw_lines
            .into_iter()
            .filter(|l| matches_filters(l, filters))
            .collect();
        Ok(last_n(filtered, num_lines))
    } else {
        read_last_n_lines(path, num_lines)
    }
}

/// Return the last `n` elements of a vector (all if `n` exceeds length).
fn last_n(mut v: Vec<String>, n: usize) -> Vec<String> {
    if v.len() > n {
        let start = v.len() - n;
        v.drain(0..start);
    }
    v
}

/// Efficiently read the last `n` lines from a file.
///
/// For files under 1MB, reads the whole file. For larger files, reads chunks
/// from the end. Each returned line retains a trailing `\n` (matching Python's
/// `readlines()` semantics closely enough for printing).
pub fn read_last_n_lines(path: &Path, n: usize) -> std::io::Result<Vec<String>> {
    let result = read_last_n_lines_inner(path, n);
    match result {
        Ok(v) => Ok(v),
        Err(_) => {
            // Fallback: read entire file (Python's bare `except` fallback).
            let bytes = fs::read(path)?;
            let text = decode_lossy(&bytes);
            Ok(last_n(split_keepends(&text), n))
        }
    }
}

fn read_last_n_lines_inner(path: &Path, n: usize) -> std::io::Result<Vec<String>> {
    let meta = fs::metadata(path)?;
    let size = meta.len();
    if size == 0 {
        return Ok(Vec::new());
    }

    // For files up to 1MB, read the whole thing — simple and correct.
    if size <= 1_048_576 {
        let bytes = fs::read(path)?;
        let text = decode_lossy(&bytes);
        return Ok(last_n(split_keepends(&text), n));
    }

    // For large files, read chunks from the end.
    let mut f = fs::File::open(path)?;
    let mut chunk_size: u64 = 8192;
    // `lines` holds raw byte segments (split on b"\n"), oldest-first.
    let mut lines: Vec<Vec<u8>> = Vec::new();
    let mut pos: u64 = size;

    while pos > 0 && (lines.len() as u64) <= (n as u64) + 1 {
        let read_size = std::cmp::min(chunk_size, pos);
        pos -= read_size;
        f.seek(SeekFrom::Start(pos))?;
        let mut buf = vec![0u8; read_size as usize];
        f.read_exact(&mut buf)?;

        let chunk_lines: Vec<Vec<u8>> = split_bytes_newline(&buf);

        if !lines.is_empty() {
            // Merge the last partial segment of the new chunk with the first
            // partial segment of what we already have.
            let last_chunk = chunk_lines
                .last()
                .cloned()
                .unwrap_or_default();
            let merged = {
                let mut m = last_chunk;
                m.extend_from_slice(&lines[0]);
                m
            };
            lines[0] = merged;
            // Prepend chunk_lines[:-1].
            let prefix = &chunk_lines[..chunk_lines.len().saturating_sub(1)];
            let mut new_lines: Vec<Vec<u8>> = Vec::with_capacity(prefix.len() + lines.len());
            new_lines.extend_from_slice(prefix);
            new_lines.append(&mut lines);
            lines = new_lines;
        } else {
            lines = chunk_lines;
        }

        chunk_size = std::cmp::min(chunk_size * 2, 65536);
    }

    // Decode and return last N non-empty lines.
    let mut decoded: Vec<String> = Vec::new();
    for raw in &lines {
        if raw.iter().all(|b| b.is_ascii_whitespace()) {
            // Python: `if not raw.strip(): continue`. An all-whitespace (incl.
            // empty) segment is skipped. Note: non-ASCII whitespace is rare in
            // logs; treat ASCII whitespace as the strip set.
            continue;
        }
        let s = String::from_utf8_lossy(raw).into_owned();
        decoded.push(format!("{}\n", s));
    }
    Ok(last_n(decoded, n))
}

/// Decode bytes as UTF-8 with lossy replacement (mirrors
/// `errors="replace"`).
fn decode_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Split text into lines, keeping the trailing `\n` on each line (like
/// Python's `readlines()`).
fn split_keepends(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '\n' {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Split a byte buffer on `b"\n"` (mirrors Python `bytes.split(b"\n")`,
/// which produces N+1 segments for N separators, including trailing empty).
fn split_bytes_newline(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    for &b in buf {
        if b == b'\n' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(b);
        }
    }
    out.push(cur);
    out
}

// ---------------------------------------------------------------------------
// Follow mode
// ---------------------------------------------------------------------------

/// Poll a log file for new content and print matching lines.
///
/// Runs until the process is interrupted (Ctrl+C). Mirrors the Python
/// `_follow_log`: seek to EOF, then read line by line, sleeping 0.3s when
/// no new data is available.
pub fn follow_log(path: &Path, filters: &Filters) {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    // Seek to end.
    if f.seek(SeekFrom::End(0)).is_err() {
        return;
    }

    let stdout = std::io::stdout();
    let mut pending: Vec<u8> = Vec::new();

    loop {
        // Read available bytes.
        let mut buf = [0u8; 8192];
        match f.read(&mut buf) {
            Ok(0) => {
                // No new data: emit any complete pending line? Python's
                // readline() only returns a complete line (terminated by \n)
                // or, at EOF with no newline, the partial. To match the common
                // tail-follow behavior we only emit on newline boundaries.
                thread::sleep(Duration::from_millis(300));
            }
            Ok(read) => {
                pending.extend_from_slice(&buf[..read]);
                // Extract complete lines (terminated by \n).
                loop {
                    if let Some(idx) = pending.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = pending.drain(..=idx).collect();
                        let line = String::from_utf8_lossy(&line_bytes).into_owned();
                        if matches_filters(&line, filters) {
                            let mut out = stdout.lock();
                            let _ = write!(out, "{}", line);
                            let _ = out.flush();
                        }
                    } else {
                        break;
                    }
                }
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(300));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// list_logs
// ---------------------------------------------------------------------------

/// Print available log files with sizes (mirrors `list_logs`).
pub fn list_logs() {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let log_dir = get_hermes_home().join("logs");
    if !log_dir.exists() {
        let _ = writeln!(
            out,
            "No logs directory at {}/logs/",
            display_hermes_home()
        );
        return;
    }

    let _ = writeln!(out, "Log files in {}/logs/:\n", display_hermes_home());
    let mut found = false;

    let entries = match fs::read_dir(&log_dir) {
        Ok(rd) => rd,
        Err(_) => {
            let _ = writeln!(
                out,
                "No logs directory at {}/logs/",
                display_hermes_home()
            );
            return;
        }
    };

    // Collect and sort by name (matching Python's sorted(iterdir())).
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();

    for entry in &paths {
        let meta = match fs::metadata(entry) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let is_log = meta.is_file()
            && entry
                .extension()
                .map(|e| e == "log")
                .unwrap_or(false);
        if !is_log {
            continue;
        }

        let size = meta.len();
        let size_str = format_size(size);

        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let age_secs = now.saturating_sub(mtime);

        let age_str = if age_secs < 60 {
            "just now".to_string()
        } else if age_secs < 3600 {
            format!("{}m ago", age_secs / 60)
        } else if age_secs < 86400 {
            format!("{}h ago", age_secs / 3600)
        } else {
            // Format mtime as a date.
            Local
                .timestamp_opt(mtime as i64, 0)
                .single()
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_default()
        };

        let name = entry
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let _ = writeln!(out, "  {:<25} {:>8}   {}", name, size_str, age_str);
        found = true;
    }

    if !found {
        let _ = writeln!(
            out,
            "  (no log files yet — run 'hermes chat' to generate logs)"
        );
    }
}

/// Format a byte size as `B` / `KB` / `MB` (matching Python's formatting).
fn format_size(size: u64) -> String {
    if size < 1024 {
        format!("{}B", size)
    } else if size < 1024 * 1024 {
        format!("{:.1}KB", size as f64 / 1024.0)
    } else {
        format!("{:.1}MB", size as f64 / (1024.0 * 1024.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn test_parse_since_units() {
        let now = Local::now().naive_local();
        let s = parse_since("30m").unwrap();
        let diff = (now - s).num_minutes();
        assert!((29..=31).contains(&diff), "diff was {}", diff);

        let h = parse_since("2h").unwrap();
        let dh = (now - h).num_hours();
        assert!((1..=2).contains(&dh));

        let d = parse_since("3d").unwrap();
        let dd = (now - d).num_days();
        assert!((2..=3).contains(&dd));

        // With whitespace and uppercase.
        assert!(parse_since("  5 S  ").is_some());
    }

    #[test]
    fn test_parse_since_invalid() {
        assert!(parse_since("").is_none());
        assert!(parse_since("abc").is_none());
        assert!(parse_since("10x").is_none());
        assert!(parse_since("h").is_none());
    }

    #[test]
    fn test_parse_line_timestamp() {
        let ts = parse_line_timestamp("2026-04-05 22:35:00,123 INFO foo:").unwrap();
        assert_eq!(ts.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-04-05 22:35:00");

        let ts2 = parse_line_timestamp("2026-04-05 22:35:00 plain").unwrap();
        assert_eq!(ts2.format("%H:%M:%S").to_string(), "22:35:00");

        assert!(parse_line_timestamp("no timestamp here").is_none());
        assert!(parse_line_timestamp("2026-13-99 99:99:99 bad").is_none());
    }

    #[test]
    fn test_extract_level() {
        assert_eq!(
            extract_level("2026-04-05 22:35:00 INFO gateway.run: hi"),
            Some("INFO".to_string())
        );
        assert_eq!(
            extract_level("ts WARNING tools.x: warn"),
            Some("WARNING".to_string())
        );
        assert_eq!(extract_level("no level"), None);
    }

    #[test]
    fn test_extract_logger_name() {
        assert_eq!(
            extract_logger_name("2026-04-05 22:35:00 INFO gateway.run: hello"),
            Some("gateway.run".to_string())
        );
        // With session tag.
        assert_eq!(
            extract_logger_name("ts INFO [sess_abc] tools.terminal_tool: msg"),
            Some("tools.terminal_tool".to_string())
        );
        assert_eq!(extract_logger_name("ts INFO no colon here"), None);
    }

    #[test]
    fn test_line_matches_component() {
        let prefixes = component_prefixes("tools").unwrap();
        assert!(line_matches_component(
            "ts INFO tools.terminal_tool: x",
            prefixes
        ));
        assert!(!line_matches_component("ts INFO gateway.run: x", prefixes));
        assert!(!line_matches_component("no logger name", prefixes));
    }

    #[test]
    fn test_matches_filters_level() {
        let f = Filters {
            min_level: Some("WARNING".to_string()),
            ..Default::default()
        };
        assert!(!matches_filters("ts INFO foo: x", &f));
        assert!(matches_filters("ts ERROR foo: x", &f));
        assert!(matches_filters("ts WARNING foo: x", &f));
        // Lines without a level pass (level extraction returns None).
        assert!(matches_filters("no level here", &f));
    }

    #[test]
    fn test_matches_filters_session() {
        let f = Filters {
            session_filter: Some("abc123".to_string()),
            ..Default::default()
        };
        assert!(matches_filters("ts INFO [abc123] foo: x", &f));
        assert!(!matches_filters("ts INFO [xyz] foo: x", &f));
    }

    #[test]
    fn test_matches_filters_since() {
        let cutoff = NaiveDateTime::parse_from_str("2026-04-05 22:00:00", "%Y-%m-%d %H:%M:%S")
            .unwrap();
        let f = Filters {
            since: Some(cutoff),
            ..Default::default()
        };
        assert!(!matches_filters("2026-04-05 21:00:00 INFO foo: old", &f));
        assert!(matches_filters("2026-04-05 23:00:00 INFO foo: new", &f));
        // Lines without timestamp pass.
        assert!(matches_filters("no timestamp INFO foo: x", &f));
    }

    #[test]
    fn test_matches_filters_component() {
        let f = Filters {
            component_prefixes: Some(component_prefixes("agent").unwrap().to_vec()),
            ..Default::default()
        };
        assert!(matches_filters("ts INFO run_agent.loop: x", &f));
        assert!(!matches_filters("ts INFO gateway.run: x", &f));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(2048), "2.0KB");
        assert_eq!(format_size(1024 * 1024 + 1024 * 512), "1.5MB");
    }

    #[test]
    fn test_split_keepends() {
        let v = split_keepends("a\nb\nc");
        assert_eq!(v, vec!["a\n", "b\n", "c"]);
        let v2 = split_keepends("a\nb\n");
        assert_eq!(v2, vec!["a\n", "b\n"]);
        assert_eq!(split_keepends(""), Vec::<String>::new());
    }

    #[test]
    fn test_split_bytes_newline() {
        // Mirrors Python bytes.split(b"\n").
        assert_eq!(
            split_bytes_newline(b"a\nb\nc"),
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
        );
        assert_eq!(
            split_bytes_newline(b"a\nb\n"),
            vec![b"a".to_vec(), b"b".to_vec(), Vec::<u8>::new()]
        );
    }

    #[test]
    fn test_last_n() {
        let v: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let got = last_n(v.clone(), 3);
        assert_eq!(got, vec!["7", "8", "9"]);
        assert_eq!(last_n(v.clone(), 100).len(), 10);
        assert_eq!(last_n(Vec::new(), 5), Vec::<String>::new());
    }

    #[test]
    fn test_read_last_n_lines_small_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hermes_cli_logs_test_{}.log", std::process::id()));
        let mut f = fs::File::create(&path).unwrap();
        for i in 0..100 {
            writeln!(f, "2026-04-05 22:35:0{} INFO foo: line {}", i % 10, i).unwrap();
        }
        drop(f);

        let lines = read_last_n_lines(&path, 5).unwrap();
        assert_eq!(lines.len(), 5);
        assert!(lines[4].contains("line 99"));
        assert!(lines[0].contains("line 95"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_tail_with_filters() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hermes_cli_logs_filt_{}.log", std::process::id()));
        let mut f = fs::File::create(&path).unwrap();
        for i in 0..50 {
            let level = if i % 2 == 0 { "INFO" } else { "ERROR" };
            writeln!(f, "2026-04-05 22:35:00 {} foo: line {}", level, i).unwrap();
        }
        drop(f);

        let filters = Filters {
            min_level: Some("ERROR".to_string()),
            ..Default::default()
        };
        let lines = read_tail(&path, 5, &filters).unwrap();
        assert_eq!(lines.len(), 5);
        for l in &lines {
            assert!(l.contains("ERROR"));
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_log_filename() {
        assert_eq!(log_filename("agent"), Some("agent.log"));
        assert_eq!(log_filename("errors"), Some("errors.log"));
        assert_eq!(log_filename("gateway"), Some("gateway.log"));
        assert_eq!(log_filename("bogus"), None);
    }

    #[test]
    fn test_log_error_messages() {
        let e = LogError::UnknownLog {
            name: "x".to_string(),
            available: "agent, errors, gateway".to_string(),
        };
        assert_eq!(
            e.messages(),
            vec!["Unknown log: 'x'. Available: agent, errors, gateway".to_string()]
        );

        let e2 = LogError::LogFileNotFound {
            path: "/tmp/x.log".to_string(),
            hint: "hint".to_string(),
        };
        assert_eq!(e2.messages().len(), 2);
    }

    #[test]
    fn test_filters_has_filters() {
        assert!(!Filters::default().has_filters());
        let f = Filters {
            session_filter: Some("x".to_string()),
            ..Default::default()
        };
        assert!(f.has_filters());
    }
}
