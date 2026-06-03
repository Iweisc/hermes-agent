//! `hermes checkpoints` CLI subcommand.
//!
//! Native Rust port of `hermes_cli/checkpoints.py`.
//!
//! Gives users direct visibility and control over the filesystem checkpoint
//! store at `~/.hermes/checkpoints/`. Actions:
//!
//! ```text
//!     hermes checkpoints               # same as `status`
//!     hermes checkpoints status        # total size, project count, breakdown
//!     hermes checkpoints list          # per-project checkpoint counts + workdir
//!     hermes checkpoints prune [opts]  # force a sweep (ignores the 24h marker)
//!     hermes checkpoints clear [-f]    # nuke the entire base (asks first)
//!     hermes checkpoints clear-legacy  # delete just the legacy-* archives
//! ```
//!
//! Examples:
//!
//! ```text
//!     hermes checkpoints
//!     hermes checkpoints prune --retention-days 3 --max-size-mb 200
//!     hermes checkpoints clear -f
//! ```
//!
//! None of these require the agent to be running. Safe to call any time.
//!
//! This module mirrors the argparse dispatch layer of the Python module. The
//! heavy lifting (`store_status`, `prune_checkpoints`, `clear_all`,
//! `clear_legacy`) is delegated to [`crate::tool_checkpoint_manager`], which is
//! the native port of `tools/checkpoint_manager.py`.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use serde_json::Value;

use crate::tool_checkpoint_manager as ckpt;

// ---------------------------------------------------------------------------
// Argument structs (mirror the argparse subparsers)
// ---------------------------------------------------------------------------

/// Parsed `hermes checkpoints <COMMAND>` invocation.
///
/// `None` / bare `hermes checkpoints` maps to [`CheckpointsCommand::Status`]
/// with `limit = 20`, matching the Python `set_defaults(func=cmd_status)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointsCommand {
    /// Show total size, project count, and per-project breakdown.
    Status { limit: usize },
    /// Alias for `status`.
    List { limit: usize },
    /// Delete orphan/stale checkpoints and GC the store.
    Prune {
        retention_days: i64,
        max_size_mb: i64,
        keep_orphans: bool,
    },
    /// Delete the entire checkpoint base (all `/rollback` history).
    Clear { force: bool },
    /// Delete only the `legacy-<ts>/` archives from v1 migration.
    ClearLegacy { force: bool },
}

impl Default for CheckpointsCommand {
    fn default() -> Self {
        // Bare `hermes checkpoints` -> status with the argparse default limit.
        CheckpointsCommand::Status { limit: 20 }
    }
}

impl CheckpointsCommand {
    /// Default `--limit` for `status`/`list`.
    pub const DEFAULT_LIMIT: usize = 20;
    /// Default `--retention-days` for `prune`.
    pub const DEFAULT_RETENTION_DAYS: i64 = 7;
    /// Default `--max-size-mb` for `prune`.
    pub const DEFAULT_MAX_SIZE_MB: i64 = 500;

    /// Construct from a positional command name + flags, mirroring the argparse
    /// subparser defaults. Unknown commands fall back to `status`.
    pub fn parse(command: Option<&str>, args: &CheckpointsArgs) -> CheckpointsCommand {
        match command.map(str::trim) {
            None | Some("") | Some("status") => CheckpointsCommand::Status {
                limit: args.limit.unwrap_or(Self::DEFAULT_LIMIT),
            },
            Some("list") => CheckpointsCommand::List {
                limit: args.limit.unwrap_or(Self::DEFAULT_LIMIT),
            },
            Some("prune") => CheckpointsCommand::Prune {
                retention_days: args.retention_days.unwrap_or(Self::DEFAULT_RETENTION_DAYS),
                max_size_mb: args.max_size_mb.unwrap_or(Self::DEFAULT_MAX_SIZE_MB),
                keep_orphans: args.keep_orphans,
            },
            Some("clear") => CheckpointsCommand::Clear { force: args.force },
            Some("clear-legacy") => CheckpointsCommand::ClearLegacy { force: args.force },
            Some(_) => CheckpointsCommand::Status {
                limit: args.limit.unwrap_or(Self::DEFAULT_LIMIT),
            },
        }
    }
}

/// Raw flag bag used by [`CheckpointsCommand::parse`].
#[derive(Debug, Clone, Default)]
pub struct CheckpointsArgs {
    pub limit: Option<usize>,
    pub retention_days: Option<i64>,
    pub max_size_mb: Option<i64>,
    pub keep_orphans: bool,
    pub force: bool,
}

// ---------------------------------------------------------------------------
// Formatting helpers (faithful ports of _fmt_bytes / _fmt_ts / _fmt_age)
// ---------------------------------------------------------------------------

/// Human-readable byte size.
///
/// Port of Python `_fmt_bytes`: `B` is printed as an integer, every larger unit
/// with one decimal place. Climbs through KB/MB/GB/TB.
pub fn fmt_bytes(n: i64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = if n > 0 { n as f64 } else { 0.0 };
    let last = UNITS.len() - 1;
    for (i, unit) in UNITS.iter().enumerate() {
        if size < 1024.0 || i == last {
            if *unit == "B" {
                return format!("{} {unit}", size as i64);
            }
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

/// Format a unix timestamp as `%Y-%m-%d %H:%M` in local time.
///
/// Port of Python `_fmt_ts`: returns the em-dash sentinel on failure.
pub fn fmt_ts(ts: Option<f64>) -> String {
    let ts = match ts {
        Some(t) if t.is_finite() => t,
        _ => return "\u{2014}".to_string(),
    };
    Local
        .timestamp_opt(ts.trunc() as i64, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "\u{2014}".to_string())
}

/// Relative age string ("12s ago", "3m ago", "now", ...).
///
/// Port of Python `_fmt_age`. A `None`/non-finite timestamp yields the em-dash.
pub fn fmt_age(ts: Option<f64>) -> String {
    let ts = match ts {
        Some(t) if t.is_finite() => t,
        _ => return "\u{2014}".to_string(),
    };
    let age = now_secs() - ts;
    if !age.is_finite() {
        return "\u{2014}".to_string();
    }
    if age < 0.0 {
        return "now".to_string();
    }
    if age < 60.0 {
        return format!("{}s ago", age as i64);
    }
    if age < 3600.0 {
        return format!("{}m ago", (age / 60.0) as i64);
    }
    if age < 86400.0 {
        return format!("{}h ago", (age / 3600.0) as i64);
    }
    format!("{}d ago", (age / 86400.0) as i64)
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Truncate a workdir string to `max` chars, prefixing with `…` and keeping the
/// trailing `max-1` chars (mirrors Python `"…" + wd[-59:]` for a 60-col field).
fn truncate_workdir(wd: &str, max: usize) -> String {
    if wd.chars().count() <= max {
        return wd.to_string();
    }
    let keep = max.saturating_sub(1);
    let count = wd.chars().count();
    let tail: String = wd.chars().skip(count.saturating_sub(keep)).collect();
    format!("\u{2026}{tail}")
}

// ---------------------------------------------------------------------------
// Value accessors over the store_status() JSON shape
// ---------------------------------------------------------------------------

fn as_i64(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn opt_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Command handlers (port of cmd_* functions). Each returns the process exit
// code, matching the Python return values.
// ---------------------------------------------------------------------------

/// `hermes checkpoints status` (and the bare invocation).
///
/// Port of `cmd_status`. Always returns 0.
pub fn cmd_status(base: &Path, limit: usize) -> i32 {
    let info = ckpt::store_status(base);

    let base_str = info
        .get("base")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    println!("Checkpoint base: {base_str}");
    println!("Total size:      {}", fmt_bytes(as_i64(&info, "total_size_bytes")));
    println!("  store/         {}", fmt_bytes(as_i64(&info, "store_size_bytes")));
    println!("  legacy-*       {}", fmt_bytes(as_i64(&info, "legacy_size_bytes")));
    println!("Projects:        {}", as_i64(&info, "project_count"));

    let empty: Vec<Value> = Vec::new();
    let mut projects: Vec<&Value> = info
        .get("projects")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .collect();
    // Sort by last_touch descending (None/0 sorts last).
    projects.sort_by(|a, b| {
        let la = a.get("last_touch").and_then(opt_f64).unwrap_or(0.0);
        let lb = b.get("last_touch").and_then(opt_f64).unwrap_or(0.0);
        lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
    });

    if !projects.is_empty() {
        println!();
        println!(
            "  {:<60}  {:>7}  {:>12}  STATE",
            "WORKDIR", "COMMITS", "LAST TOUCH"
        );
        for p in projects.iter().take(limit) {
            let wd_raw = p.get("workdir").and_then(Value::as_str).unwrap_or("");
            let wd = if wd_raw.is_empty() {
                "(unknown)".to_string()
            } else {
                truncate_workdir(wd_raw, 60)
            };
            let exists = p.get("exists").and_then(Value::as_bool).unwrap_or(false);
            let state = if exists { "live" } else { "orphan" };
            let commits = p.get("commits").and_then(Value::as_i64).unwrap_or(0);
            let last = fmt_age(p.get("last_touch").and_then(opt_f64));
            println!("  {wd:<60}  {commits:>7}  {last:>12}  {state}");
        }
    }

    let legacy: Vec<&Value> = info
        .get("legacy_archives")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .collect();
    if !legacy.is_empty() {
        println!();
        println!("Legacy archives ({}):", legacy.len());
        // Sort by mtime descending.
        let mut sorted = legacy.clone();
        sorted.sort_by(|a, b| {
            let ma = a.get("mtime").and_then(opt_f64).unwrap_or(0.0);
            let mb = b.get("mtime").and_then(opt_f64).unwrap_or(0.0);
            mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
        });
        for arch in sorted {
            let name = arch.get("name").and_then(Value::as_str).unwrap_or("");
            let size = arch.get("size_bytes").and_then(Value::as_i64).unwrap_or(0);
            println!("  {name:<40}  {:>10}", fmt_bytes(size));
        }
        println!();
        println!("Clear with: hermes checkpoints clear-legacy");
    }
    0
}

/// `hermes checkpoints list` — terser alias for `status`. Port of `cmd_list`.
pub fn cmd_list(base: &Path, limit: usize) -> i32 {
    cmd_status(base, limit)
}

/// `hermes checkpoints prune`. Port of `cmd_prune`. Always returns 0.
pub fn cmd_prune(base: &Path, retention_days: i64, max_size_mb: i64, keep_orphans: bool) -> i32 {
    println!("Pruning checkpoint store\u{2026}");
    println!("  retention_days:    {retention_days}");
    println!("  delete_orphans:    {}", !keep_orphans);
    println!("  max_total_size_mb: {max_size_mb}");
    println!();

    let max_total = if max_size_mb > 0 { max_size_mb as u64 } else { 0 };
    let result = ckpt::prune_checkpoints(retention_days, !keep_orphans, base, max_total);

    println!("Scanned:         {}", result.scanned);
    println!("Deleted orphan:  {}", result.deleted_orphan);
    println!("Deleted stale:   {}", result.deleted_stale);
    println!("Errors:          {}", result.errors);
    println!("Bytes reclaimed: {}", fmt_bytes(result.bytes_freed as i64));
    0
}

/// `hermes checkpoints clear`. Port of `cmd_clear`.
///
/// Returns: 0 on success / nothing-to-do, 1 if aborted at the prompt, 2 if the
/// clear failed.
pub fn cmd_clear(base: &Path, force: bool) -> i32 {
    let info = ckpt::store_status(base);
    if as_i64(&info, "total_size_bytes") == 0 && !base.exists() {
        println!("Nothing to clear \u{2014} checkpoint base does not exist.");
        return 0;
    }

    let base_str = info.get("base").and_then(Value::as_str).unwrap_or("");
    let legacy_count = info
        .get("legacy_archives")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);

    println!("This will delete the ENTIRE checkpoint base at {base_str}");
    println!("  size:        {}", fmt_bytes(as_i64(&info, "total_size_bytes")));
    println!("  projects:    {}", as_i64(&info, "project_count"));
    println!("  legacy dirs: {legacy_count}");
    println!();
    println!("All /rollback history for every working directory will be lost.");
    if !force && !confirm("Proceed?") {
        println!("Aborted.");
        return 1;
    }

    let result = ckpt::clear_all(base);
    if result.deleted > 0 {
        println!("Cleared. Reclaimed {}.", fmt_bytes(result.bytes_freed as i64));
        return 0;
    }
    println!("Could not clear checkpoint base (see logs).");
    2
}

/// `hermes checkpoints clear-legacy`. Port of `cmd_clear_legacy`.
///
/// Returns: 0 on success / nothing-to-do, 1 if aborted at the prompt.
pub fn cmd_clear_legacy(base: &Path, force: bool) -> i32 {
    let info = ckpt::store_status(base);
    let empty: Vec<Value> = Vec::new();
    let legacy: Vec<&Value> = info
        .get("legacy_archives")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .collect();
    if legacy.is_empty() {
        println!("No legacy archives to clear.");
        return 0;
    }

    let total: i64 = legacy
        .iter()
        .map(|a| a.get("size_bytes").and_then(Value::as_i64).unwrap_or(0))
        .sum();
    println!(
        "Found {} legacy archive(s), total {}:",
        legacy.len(),
        fmt_bytes(total)
    );
    for arch in &legacy {
        let name = arch.get("name").and_then(Value::as_str).unwrap_or("");
        let size = arch.get("size_bytes").and_then(Value::as_i64).unwrap_or(0);
        println!("  {name:<40}  {:>10}", fmt_bytes(size));
    }
    println!();
    println!("Legacy archives hold pre-v2 per-project shadow repos, moved aside");
    println!("during the single-store migration. Delete when you're confident");
    println!("you don't need the old /rollback history.");
    if !force && !confirm("Delete all legacy archives?") {
        println!("Aborted.");
        return 1;
    }

    let result = ckpt::clear_legacy(base);
    println!(
        "Deleted {} archive(s), reclaimed {}.",
        result.deleted,
        fmt_bytes(result.bytes_freed as i64)
    );
    0
}

// ---------------------------------------------------------------------------
// Confirmation prompt (port of _confirm)
// ---------------------------------------------------------------------------

/// Interactive `[y/N]` prompt. Port of Python `_confirm`.
///
/// Returns `true` only for `y`/`yes` (case-insensitive). EOF or read error maps
/// to the Python `EOFError`/`KeyboardInterrupt` branch: prints a newline and
/// returns `false`.
pub fn confirm(prompt: &str) -> bool {
    let mut out = io::stdout().lock();
    let _ = write!(out, "{prompt} [y/N]: ");
    let _ = out.flush();

    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) => {
            // EOF -> mirror the Python `print()` then `return False`.
            println!();
            false
        }
        Ok(_) => {
            let resp = line.trim().to_lowercase();
            resp == "y" || resp == "yes"
        }
        Err(_) => {
            println!();
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch (port of register_cli + the func dispatch in argparse)
// ---------------------------------------------------------------------------

/// Resolve the default checkpoint base directory: `~/.hermes/checkpoints`.
pub fn default_checkpoint_base() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes").join("checkpoints")
}

/// Dispatch a parsed [`CheckpointsCommand`] against a checkpoint base directory,
/// returning the process exit code (matching the Python `func` return values).
pub fn run(base: &Path, command: CheckpointsCommand) -> i32 {
    match command {
        CheckpointsCommand::Status { limit } => cmd_status(base, limit),
        CheckpointsCommand::List { limit } => cmd_list(base, limit),
        CheckpointsCommand::Prune {
            retention_days,
            max_size_mb,
            keep_orphans,
        } => cmd_prune(base, retention_days, max_size_mb, keep_orphans),
        CheckpointsCommand::Clear { force } => cmd_clear(base, force),
        CheckpointsCommand::ClearLegacy { force } => cmd_clear_legacy(base, force),
    }
}

/// Convenience entry point: parse a positional command name + flags, then run
/// against the default `~/.hermes/checkpoints` base.
pub fn run_default(command: Option<&str>, args: &CheckpointsArgs) -> i32 {
    let base = default_checkpoint_base();
    run(&base, CheckpointsCommand::parse(command, args))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn fmt_bytes_matches_python() {
        assert_eq!(fmt_bytes(0), "0 B");
        assert_eq!(fmt_bytes(-5), "0 B");
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1023), "1023 B");
        assert_eq!(fmt_bytes(1024), "1.0 KB");
        assert_eq!(fmt_bytes(1536), "1.5 KB");
        assert_eq!(fmt_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(fmt_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(fmt_bytes(1024_i64.pow(4)), "1.0 TB");
        // Beyond TB still reports in TB.
        assert_eq!(fmt_bytes(1024_i64.pow(4) * 2048), "2048.0 TB");
    }

    #[test]
    fn fmt_age_buckets() {
        let now = now_secs();
        assert_eq!(fmt_age(None), "\u{2014}");
        assert_eq!(fmt_age(Some(f64::NAN)), "\u{2014}");
        assert_eq!(fmt_age(Some(now + 1000.0)), "now");
        assert_eq!(fmt_age(Some(now - 10.0)), "10s ago");
        assert_eq!(fmt_age(Some(now - 120.0)), "2m ago");
        assert_eq!(fmt_age(Some(now - 7200.0)), "2h ago");
        assert_eq!(fmt_age(Some(now - 3.0 * 86400.0)), "3d ago");
    }

    #[test]
    fn fmt_ts_invalid_returns_dash() {
        assert_eq!(fmt_ts(None), "\u{2014}");
        assert_eq!(fmt_ts(Some(f64::NAN)), "\u{2014}");
        // A valid timestamp produces a YYYY-MM-DD HH:MM-shaped string.
        let s = fmt_ts(Some(1_700_000_000.0));
        assert_eq!(s.len(), 16);
        assert!(s.contains('-') && s.contains(':'));
    }

    #[test]
    fn truncate_workdir_keeps_tail_with_ellipsis() {
        let short = "/a/b/c";
        assert_eq!(truncate_workdir(short, 60), short);

        let long: String = std::iter::repeat('x').take(80).collect();
        let out = truncate_workdir(&long, 60);
        assert!(out.starts_with('\u{2026}'));
        // 1 ellipsis + 59 tail chars.
        assert_eq!(out.chars().count(), 60);
    }

    #[test]
    fn parse_defaults_and_aliases() {
        let args = CheckpointsArgs::default();
        assert_eq!(
            CheckpointsCommand::parse(None, &args),
            CheckpointsCommand::Status { limit: 20 }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some(""), &args),
            CheckpointsCommand::Status { limit: 20 }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("status"), &args),
            CheckpointsCommand::Status { limit: 20 }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("list"), &args),
            CheckpointsCommand::List { limit: 20 }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("prune"), &args),
            CheckpointsCommand::Prune {
                retention_days: 7,
                max_size_mb: 500,
                keep_orphans: false,
            }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("clear"), &args),
            CheckpointsCommand::Clear { force: false }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("clear-legacy"), &args),
            CheckpointsCommand::ClearLegacy { force: false }
        );
        // Unknown command -> status (argparse would error, but the bare default
        // here mirrors `set_defaults(func=cmd_status)`).
        assert_eq!(
            CheckpointsCommand::parse(Some("bogus"), &args),
            CheckpointsCommand::Status { limit: 20 }
        );
    }

    #[test]
    fn parse_honours_flags() {
        let args = CheckpointsArgs {
            limit: Some(5),
            retention_days: Some(3),
            max_size_mb: Some(200),
            keep_orphans: true,
            force: true,
        };
        assert_eq!(
            CheckpointsCommand::parse(Some("status"), &args),
            CheckpointsCommand::Status { limit: 5 }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("prune"), &args),
            CheckpointsCommand::Prune {
                retention_days: 3,
                max_size_mb: 200,
                keep_orphans: true,
            }
        );
        assert_eq!(
            CheckpointsCommand::parse(Some("clear"), &args),
            CheckpointsCommand::Clear { force: true }
        );
    }

    #[test]
    fn status_on_missing_base_returns_zero() {
        let tmp =
            std::env::temp_dir().join(format!("cli_ckpt_status_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        assert_eq!(cmd_status(&tmp, 20), 0);
    }

    #[test]
    fn clear_on_missing_base_returns_zero() {
        let tmp =
            std::env::temp_dir().join(format!("cli_ckpt_clear_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // Nothing to clear, base does not exist -> 0 without prompting.
        assert_eq!(cmd_clear(&tmp, true), 0);
    }

    #[test]
    fn clear_legacy_on_missing_base_returns_zero() {
        let tmp =
            std::env::temp_dir().join(format!("cli_ckpt_clearleg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // store_status on a missing base yields no legacy archives.
        assert_eq!(cmd_clear_legacy(&tmp, true), 0);
    }

    #[test]
    fn clear_legacy_deletes_only_legacy_dirs() {
        let tmp =
            std::env::temp_dir().join(format!("cli_ckpt_legacy_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("legacy-20200101-000000")).unwrap();
        fs::write(tmp.join("legacy-20200101-000000/x"), b"data").unwrap();
        fs::create_dir_all(tmp.join("store")).unwrap();
        fs::write(tmp.join("store/y"), b"keep").unwrap();

        // force=true so no prompt; should report success and remove only legacy.
        assert_eq!(cmd_clear_legacy(&tmp, true), 0);
        assert!(!tmp.join("legacy-20200101-000000").exists());
        assert!(tmp.join("store").exists());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn prune_on_missing_base_returns_zero() {
        let tmp =
            std::env::temp_dir().join(format!("cli_ckpt_prune_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        assert_eq!(cmd_prune(&tmp, 7, 500, false), 0);
    }

    #[test]
    fn run_dispatches_status_variant() {
        let tmp = std::env::temp_dir().join(format!("cli_ckpt_run_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        assert_eq!(run(&tmp, CheckpointsCommand::Status { limit: 20 }), 0);
        assert_eq!(run(&tmp, CheckpointsCommand::List { limit: 20 }), 0);
    }

    #[test]
    fn default_base_ends_with_hermes_checkpoints() {
        let base = default_checkpoint_base();
        assert!(base.ends_with("checkpoints"));
        assert!(base
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n == ".hermes")
            .unwrap_or(false));
    }
}
