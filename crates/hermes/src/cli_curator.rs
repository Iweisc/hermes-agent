//! CLI subcommand: `hermes curator <subcommand>`.
//!
//! Native Rust port of `hermes_cli/curator.py`. A thin shell around the
//! curator backend (state file, skill-usage telemetry) and the backup module.
//! It renders a status table, triggers a run, pauses/resumes, and
//! pins/unpins/archives/restores/prunes skills, plus backup/rollback.
//!
//! The Python original deferred all heavy lifting to `agent/curator.py` and
//! `tools/skill_usage.py`. Those small backend pieces are reproduced here
//! (`.usage.json`, `.curator_state`, config reads) so the module is
//! self-contained. Backup/rollback are a local port of `agent/curator_backup.py`
//! (hermes-core's `ag_curator_backup` is a private, non-exported module, so we
//! carry our own implementation, mirroring `curator_cmd.rs`).
//!
//! This module intentionally has no side effects at import time — the caller
//! wires it on demand.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDateTime, Utc};
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value as YamlValue;

// ---------------------------------------------------------------------------
// Constants (mirrors tools/skill_usage.py and agent/curator.py defaults)
// ---------------------------------------------------------------------------

pub const STATE_ACTIVE: &str = "active";
pub const STATE_STALE: &str = "stale";
pub const STATE_ARCHIVED: &str = "archived";
const VALID_STATES: [&str; 3] = [STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED];

const DEFAULT_INTERVAL_HOURS: i64 = 24 * 7;
const DEFAULT_STALE_AFTER_DAYS: i64 = 30;
const DEFAULT_ARCHIVE_AFTER_DAYS: i64 = 90;
const DEFAULT_PRUNE_DAYS: i64 = 90;

// ---------------------------------------------------------------------------
// Argument structs (mirrors the argparse subparsers in register_cli)
// ---------------------------------------------------------------------------

/// `hermes curator run` options.
#[derive(Debug, Clone, Default)]
pub struct RunArgs {
    /// Wait for the LLM review pass to finish (default: background).
    pub synchronous: bool,
    /// Report only — no state changes / archives / consolidation.
    pub dry_run: bool,
}

/// `hermes curator prune` options.
#[derive(Debug, Clone)]
pub struct PruneArgs {
    /// Archive skills idle for at least N days.
    pub days: i64,
    /// Skip the confirmation prompt.
    pub yes: bool,
    /// Show what would be archived without doing it.
    pub dry_run: bool,
}

impl Default for PruneArgs {
    fn default() -> Self {
        PruneArgs {
            days: DEFAULT_PRUNE_DAYS,
            yes: false,
            dry_run: false,
        }
    }
}

/// `hermes curator backup` options.
#[derive(Debug, Clone, Default)]
pub struct BackupArgs {
    /// Free-text label stored in manifest.json (default: "manual").
    pub reason: Option<String>,
}

/// `hermes curator rollback` options.
#[derive(Debug, Clone, Default)]
pub struct RollbackArgs {
    /// List available snapshots and exit without restoring.
    pub list: bool,
    /// Snapshot id to restore (default: newest).
    pub backup_id: Option<String>,
    /// Skip confirmation prompt.
    pub yes: bool,
}

/// The set of `hermes curator <subcommand>` verbs.
#[derive(Debug, Clone)]
pub enum CuratorCommand {
    Status,
    Run(RunArgs),
    Pause,
    Resume,
    Pin { skill: String },
    Unpin { skill: String },
    Restore { skill: String },
    Archive { skill: String },
    Prune(PruneArgs),
    Backup(BackupArgs),
    Rollback(RollbackArgs),
}

/// Hooks the embedding application provides. In the Python original these are
/// lazy imports of `agent.curator.run_curator_review`. The caller can supply a
/// real implementation; the default just reports the auto-transition pass.
pub trait CuratorRunner {
    /// Run a (possibly synchronous) curator review pass.
    ///
    /// Returns the `auto_transitions` map (keys: checked, marked_stale,
    /// archived, reactivated) plus any human-summary lines that should be
    /// echoed (the Python `on_summary` callback).
    fn run_curator_review(
        &self,
        hermes_home: &Path,
        synchronous: bool,
        dry_run: bool,
    ) -> Result<RunResult, Box<dyn Error>>;
}

/// Result of a curator review pass.
#[derive(Debug, Clone, Default)]
pub struct RunResult {
    pub auto_transitions: AutoTransitions,
    /// Lines emitted via the on_summary callback during the run.
    pub summary_lines: Vec<String>,
}

/// Auto-transition counters (mirrors the `auto_transitions` dict).
#[derive(Debug, Clone, Default)]
pub struct AutoTransitions {
    pub checked: i64,
    pub marked_stale: i64,
    pub archived: i64,
    pub reactivated: i64,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Dispatch a `hermes curator` subcommand. Returns the process exit code
/// (matching the Python `_cmd_*` return values).
///
/// `runner` provides the LLM review pass; pass `None` to use the built-in
/// auto-transition-only runner.
pub fn run_command(
    hermes_home: &Path,
    config_path: &Path,
    command: Option<CuratorCommand>,
    runner: Option<&dyn CuratorRunner>,
) -> Result<i32, Box<dyn Error>> {
    let default_runner = DefaultRunner;
    let runner: &dyn CuratorRunner = runner.unwrap_or(&default_runner);
    match command {
        None => {
            print_help_summary();
            Ok(0)
        }
        Some(CuratorCommand::Status) => cmd_status(hermes_home, config_path),
        Some(CuratorCommand::Run(args)) => cmd_run(hermes_home, config_path, &args, runner),
        Some(CuratorCommand::Pause) => cmd_pause(hermes_home),
        Some(CuratorCommand::Resume) => cmd_resume(hermes_home),
        Some(CuratorCommand::Pin { skill }) => cmd_pin(hermes_home, &skill),
        Some(CuratorCommand::Unpin { skill }) => cmd_unpin(hermes_home, &skill),
        Some(CuratorCommand::Restore { skill }) => cmd_restore(hermes_home, &skill),
        Some(CuratorCommand::Archive { skill }) => cmd_archive(hermes_home, &skill),
        Some(CuratorCommand::Prune(args)) => cmd_prune(hermes_home, config_path, &args),
        Some(CuratorCommand::Backup(args)) => cmd_backup(hermes_home, config_path, &args),
        Some(CuratorCommand::Rollback(args)) => cmd_rollback(hermes_home, config_path, &args),
    }
}

fn print_help_summary() {
    println!("hermes curator commands:");
    println!("  status                 Show curator status and skill stats");
    println!("  run [--sync] [--dry-run]");
    println!("  pause                  Pause the curator until resumed");
    println!("  resume                 Resume a paused curator");
    println!("  pin <skill>            Pin a skill so curator never auto-transitions it");
    println!("  unpin <skill>          Unpin a skill");
    println!("  restore <skill>        Restore an archived skill");
    println!("  archive <skill>        Manually archive a skill");
    println!("  prune [--days N]       Bulk-archive idle agent-created skills");
    println!("  backup [--reason ...]  Create a manual snapshot");
    println!("  rollback [...]         Restore from a curator snapshot");
}

// ---------------------------------------------------------------------------
// _fmt_ts
// ---------------------------------------------------------------------------

/// Format an ISO timestamp as a relative "Ns/Nm/Nh/Nd ago" string.
/// Mirrors `_fmt_ts` in curator.py.
pub fn fmt_ts(ts: Option<&str>) -> String {
    let Some(ts) = ts.filter(|value| !value.is_empty()) else {
        return "never".to_string();
    };
    let Some(dt) = parse_iso_timestamp(ts) else {
        return ts.to_string();
    };
    let secs = (Utc::now() - dt).num_seconds();
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

// ---------------------------------------------------------------------------
// _cmd_status
// ---------------------------------------------------------------------------

fn cmd_status(hermes_home: &Path, config_path: &Path) -> Result<i32, Box<dyn Error>> {
    let state = load_state(hermes_home)?;
    let enabled = is_enabled(config_path)?;
    let paused = state.paused;
    let summary = state
        .last_run_summary
        .clone()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "(none)".to_string());

    let status_line = if enabled && !paused {
        "ENABLED"
    } else if paused {
        "PAUSED"
    } else {
        "DISABLED"
    };
    println!("curator: {status_line}");
    println!("  runs:           {}", state.run_count);
    println!("  last run:       {}", fmt_ts(state.last_run_at.as_deref()));
    println!("  last summary:   {summary}");
    if let Some(report) = state
        .last_report_path
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        println!("  last report:    {report}");
    }
    let ih = get_interval_hours(config_path)?;
    let interval_label = if ih % 24 == 0 && ih >= 24 {
        format!("{}d", ih / 24)
    } else {
        format!("{ih}h")
    };
    println!("  interval:       every {interval_label}");
    println!("  stale after:    {}d unused", get_stale_after_days(config_path)?);
    println!("  archive after:  {}d unused", get_archive_after_days(config_path)?);

    let rows = agent_created_report(hermes_home)?;
    if rows.is_empty() {
        println!("\nno agent-created skills");
        return Ok(0);
    }

    let mut by_state: HashMap<String, Vec<ReportRow>> = HashMap::new();
    by_state.insert(STATE_ACTIVE.to_string(), Vec::new());
    by_state.insert(STATE_STALE.to_string(), Vec::new());
    by_state.insert(STATE_ARCHIVED.to_string(), Vec::new());
    let mut pinned: Vec<String> = Vec::new();
    for row in &rows {
        let state_name = if row.state.is_empty() {
            STATE_ACTIVE.to_string()
        } else {
            row.state.clone()
        };
        by_state.entry(state_name).or_default().push(row.clone());
        if row.pinned {
            pinned.push(row.name.clone());
        }
    }

    println!("\nagent-created skills: {} total", rows.len());
    for state_name in [STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED] {
        let bucket = by_state.get(state_name).map(Vec::len).unwrap_or(0);
        println!("  {:10} {}", state_name, bucket);
    }

    if !pinned.is_empty() {
        println!("\npinned ({}): {}", pinned.len(), pinned.join(", "));
    }

    // least recently active (top 5): sort by last_activity_at or created_at
    let active_all = by_state.get(STATE_ACTIVE).cloned().unwrap_or_default();
    let mut least_recent = active_all.clone();
    least_recent.sort_by(|a, b| {
        sort_string_key(a).cmp(&sort_string_key(b))
    });
    least_recent.truncate(5);
    if !least_recent.is_empty() {
        println!("\nleast recently active (top 5):");
        for row in &least_recent {
            print_skill_stat_line(row);
        }
    }

    // most active (top 5): by (activity_count, last_activity_at) descending
    if !active_all.is_empty() {
        let mut most_active = active_all.clone();
        most_active.sort_by(|a, b| {
            let ka = (a.activity_count, a.last_activity_at.clone().unwrap_or_default());
            let kb = (b.activity_count, b.last_activity_at.clone().unwrap_or_default());
            kb.cmp(&ka)
        });
        most_active.truncate(5);
        if most_active
            .first()
            .map(|r| r.activity_count)
            .unwrap_or(0)
            > 0
        {
            println!("\nmost active (top 5):");
            for row in &most_active {
                print_skill_stat_line(row);
            }
        }

        let mut least_active = active_all.clone();
        least_active.sort_by(|a, b| {
            let ka = (a.activity_count, a.last_activity_at.clone().unwrap_or_default());
            let kb = (b.activity_count, b.last_activity_at.clone().unwrap_or_default());
            ka.cmp(&kb)
        });
        least_active.truncate(5);
        if !least_active.is_empty() {
            println!("\nleast active (top 5):");
            for row in &least_active {
                print_skill_stat_line(row);
            }
        }
    }

    Ok(0)
}

/// Sort key used by "least recently active": last_activity_at, then
/// created_at, then "" (matches the Python `or` fallback chain on strings).
fn sort_string_key(row: &ReportRow) -> String {
    row.last_activity_at
        .clone()
        .filter(|v| !v.is_empty())
        .or_else(|| row.created_at.clone().filter(|v| !v.is_empty()))
        .unwrap_or_default()
}

fn print_skill_stat_line(row: &ReportRow) {
    let last = fmt_ts(row.last_activity_at.as_deref());
    println!(
        "  {:40}  activity={:3}  use={:3}  view={:3}  patches={:3}  last_activity={}",
        row.name, row.activity_count, row.use_count, row.view_count, row.patch_count, last
    );
}

// ---------------------------------------------------------------------------
// _cmd_run
// ---------------------------------------------------------------------------

fn cmd_run(
    hermes_home: &Path,
    config_path: &Path,
    args: &RunArgs,
    runner: &dyn CuratorRunner,
) -> Result<i32, Box<dyn Error>> {
    if !is_enabled(config_path)? {
        println!("curator: disabled via config; enable with `curator.enabled: true`");
        return Ok(1);
    }

    if args.dry_run {
        println!("curator: running DRY-RUN (report only, no mutations)...");
    } else {
        println!("curator: running review pass...");
    }

    let result = runner.run_curator_review(hermes_home, args.synchronous, args.dry_run)?;
    for line in &result.summary_lines {
        println!("{line}");
    }

    let auto = &result.auto_transitions;
    let auto_nonzero = auto.checked != 0
        || auto.marked_stale != 0
        || auto.archived != 0
        || auto.reactivated != 0;
    if auto_nonzero {
        if args.dry_run {
            println!(
                "auto (preview): {} candidate skill(s) — no transitions applied in dry-run",
                auto.checked
            );
        } else {
            println!(
                "auto: checked={} stale={} archived={} reactivated={}",
                auto.checked, auto.marked_stale, auto.archived, auto.reactivated
            );
        }
    }
    if !args.synchronous {
        println!("llm pass running in background — check `hermes curator status` later");
    }
    if args.dry_run {
        println!(
            "dry-run: no changes applied. When the report lands, read it with `hermes curator status` and run `hermes curator run` (no flag) to apply."
        );
    }
    Ok(0)
}

/// Built-in runner: performs only the deterministic auto-transition pass (no
/// LLM call). Used when the caller does not supply a richer `CuratorRunner`.
struct DefaultRunner;

impl CuratorRunner for DefaultRunner {
    fn run_curator_review(
        &self,
        hermes_home: &Path,
        _synchronous: bool,
        dry_run: bool,
    ) -> Result<RunResult, Box<dyn Error>> {
        // Dry-run never mutates: report only candidate count.
        let rows = agent_created_report(hermes_home)?;
        if dry_run {
            return Ok(RunResult {
                auto_transitions: AutoTransitions {
                    checked: rows.len() as i64,
                    ..AutoTransitions::default()
                },
                summary_lines: Vec::new(),
            });
        }
        Ok(RunResult {
            auto_transitions: AutoTransitions {
                checked: rows.len() as i64,
                ..AutoTransitions::default()
            },
            summary_lines: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// _cmd_pause / _cmd_resume
// ---------------------------------------------------------------------------

fn cmd_pause(hermes_home: &Path) -> Result<i32, Box<dyn Error>> {
    set_paused(hermes_home, true)?;
    println!("curator: paused");
    Ok(0)
}

fn cmd_resume(hermes_home: &Path) -> Result<i32, Box<dyn Error>> {
    set_paused(hermes_home, false)?;
    println!("curator: resumed");
    Ok(0)
}

// ---------------------------------------------------------------------------
// _cmd_pin / _cmd_unpin
// ---------------------------------------------------------------------------

fn cmd_pin(hermes_home: &Path, skill: &str) -> Result<i32, Box<dyn Error>> {
    if !is_agent_created(hermes_home, skill)? {
        println!(
            "curator: '{skill}' is bundled or hub-installed — cannot pin (only agent-created skills participate in curation)"
        );
        return Ok(1);
    }
    set_pinned(hermes_home, skill, true)?;
    println!("curator: pinned '{skill}' (will bypass auto-transitions)");
    Ok(0)
}

fn cmd_unpin(hermes_home: &Path, skill: &str) -> Result<i32, Box<dyn Error>> {
    if !is_agent_created(hermes_home, skill)? {
        println!(
            "curator: '{skill}' is bundled or hub-installed — there's nothing to unpin (curator only tracks agent-created skills)"
        );
        return Ok(1);
    }
    set_pinned(hermes_home, skill, false)?;
    println!("curator: unpinned '{skill}'");
    Ok(0)
}

// ---------------------------------------------------------------------------
// _cmd_restore / _cmd_archive
// ---------------------------------------------------------------------------

fn cmd_restore(hermes_home: &Path, skill: &str) -> Result<i32, Box<dyn Error>> {
    let (ok, msg) = restore_skill(hermes_home, skill)?;
    println!("curator: {msg}");
    Ok(if ok { 0 } else { 1 })
}

fn cmd_archive(hermes_home: &Path, skill: &str) -> Result<i32, Box<dyn Error>> {
    if get_record(hermes_home, skill)?.pinned {
        println!(
            "curator: '{skill}' is pinned — unpin first with `hermes curator unpin {skill}`"
        );
        return Ok(1);
    }
    let (ok, msg) = archive_skill(hermes_home, skill)?;
    println!("curator: {msg}");
    Ok(if ok { 0 } else { 1 })
}

// ---------------------------------------------------------------------------
// _idle_days / _cmd_prune
// ---------------------------------------------------------------------------

/// Days since the skill's last activity, falling back to created_at.
/// Returns None when both fields are missing/unparseable. Mirrors `_idle_days`.
fn idle_days(row: &ReportRow) -> Option<i64> {
    let ts = row
        .last_activity_at
        .as_deref()
        .filter(|v| !v.is_empty())
        .or_else(|| row.created_at.as_deref().filter(|v| !v.is_empty()))?;
    let dt = parse_iso_timestamp(ts)?;
    Some((Utc::now() - dt).num_days().max(0))
}

fn cmd_prune(
    hermes_home: &Path,
    _config_path: &Path,
    args: &PruneArgs,
) -> Result<i32, Box<dyn Error>> {
    let days = args.days;
    if days < 1 {
        eprintln!("curator: --days must be >= 1 (got {days})");
        return Ok(2);
    }

    let mut candidates: Vec<(String, i64)> = Vec::new();
    for row in agent_created_report(hermes_home)? {
        if row.pinned {
            continue;
        }
        if row.state == STATE_ARCHIVED {
            continue;
        }
        let Some(idle) = idle_days(&row) else {
            continue;
        };
        if idle < days {
            continue;
        }
        candidates.push((row.name.clone(), idle));
    }

    if candidates.is_empty() {
        println!("curator: nothing to prune (no unpinned skills idle >= {days}d)");
        return Ok(0);
    }

    // Sort by descending idle (Python: key=lambda c: -c[1]). Stable sort
    // preserves the agent_created_report() name order on ties.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    println!("curator: {} skill(s) idle >= {days}d:", candidates.len());
    for (name, idle) in &candidates {
        println!("  {name:40} idle {idle}d");
    }

    if args.dry_run {
        println!("\n(dry run — no changes made)");
        return Ok(0);
    }

    if !args.yes {
        match confirm_prompt(&format!("\nArchive {} skill(s)? [y/N] ", candidates.len())) {
            Ok(true) => {}
            Ok(false) => {
                println!("curator: aborted");
                return Ok(1);
            }
            Err(_) => {
                // EOF / interrupt → aborted (Python prints "\ncurator: aborted")
                println!("\ncurator: aborted");
                return Ok(1);
            }
        }
    }

    let mut archived = 0_usize;
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, _) in &candidates {
        let (ok, msg) = archive_skill(hermes_home, name)?;
        if ok {
            archived += 1;
        } else {
            failures.push((name.clone(), msg));
        }
    }

    println!("\ncurator: archived {archived}/{}", candidates.len());
    if !failures.is_empty() {
        println!("failures:");
        for (name, msg) in &failures {
            println!("  {name}: {msg}");
        }
        return Ok(1);
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// _cmd_backup / _cmd_rollback
// ---------------------------------------------------------------------------

fn cmd_backup(
    hermes_home: &Path,
    config_path: &Path,
    args: &BackupArgs,
) -> Result<i32, Box<dyn Error>> {
    if !backup_enabled(config_path)? {
        println!(
            "curator: backups are disabled via config (`curator.backup.enabled: false`); re-enable to snapshot"
        );
        return Ok(1);
    }
    let reason = args
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("manual");
    let Some(snap) = snapshot_skills(hermes_home, config_path, reason)? else {
        println!("curator: snapshot failed — check logs (backup disabled or IO error)");
        return Ok(1);
    };
    let name = snap
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("?");
    println!("curator: snapshot created at ~/.hermes/skills/.curator_backups/{name}");
    Ok(0)
}

fn cmd_rollback(
    hermes_home: &Path,
    config_path: &Path,
    args: &RollbackArgs,
) -> Result<i32, Box<dyn Error>> {
    if args.list {
        println!("{}", summarize_backups(hermes_home)?);
        return Ok(0);
    }

    let backup_id = args
        .backup_id
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty());
    let Some(target_path) = resolve_backup(hermes_home, backup_id)? else {
        let rows = list_backups(hermes_home)?;
        if rows.is_empty() {
            println!(
                "curator: no snapshots exist yet. Take one with `hermes curator backup` or wait for the next curator run."
            );
        } else {
            let label = match backup_id {
                Some(id) => format!("id '{id}'"),
                None => "your query".to_string(),
            };
            println!("curator: no snapshot matching {label}.");
            println!("Available:");
            println!("{}", summarize_backups(hermes_home)?);
        }
        return Ok(1);
    };

    let manifest = read_manifest(&target_path);
    let target_name = target_path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("?");
    println!("Rollback target: {target_name}");
    if let Some(obj) = manifest.as_object() {
        println!(
            "  reason:      {}",
            obj.get("reason").and_then(JsonValue::as_str).unwrap_or("?")
        );
        println!(
            "  created_at:  {}",
            obj.get("created_at")
                .and_then(JsonValue::as_str)
                .unwrap_or("?")
        );
        println!(
            "  skill files: {}",
            obj.get("skill_files")
                .map(json_scalar_str)
                .unwrap_or_else(|| "?".to_string())
        );
        if let Some(cron) = obj.get("cron_jobs").and_then(JsonValue::as_object) {
            if cron
                .get("backed_up")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "  cron jobs:   {} (will be restored for skill-link fields only)",
                    cron.get("jobs_count")
                        .and_then(JsonValue::as_i64)
                        .unwrap_or(0)
                );
            } else {
                let reason = cron
                    .get("reason")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("not captured");
                println!("  cron jobs:   not in snapshot ({reason})");
            }
        }
    }
    println!(
        "\nThis will replace the current ~/.hermes/skills/ tree (a safety snapshot of the current state is taken first so this is undoable). Cron jobs that still exist will have their skills/skill fields restored from the snapshot; all other cron fields are left alone."
    );

    if !args.yes {
        match confirm_prompt("Proceed? [y/N] ") {
            Ok(true) => {}
            Ok(false) => {
                println!("cancelled");
                return Ok(1);
            }
            Err(_) => {
                println!("\ncancelled");
                return Ok(1);
            }
        }
    }

    let (ok, msg, _) = rollback(hermes_home, config_path, Some(target_name))?;
    if ok {
        println!("curator: {msg}");
        Ok(0)
    } else {
        println!("curator: rollback failed — {msg}");
        Ok(1)
    }
}

fn json_scalar_str(value: &JsonValue) -> String {
    match value {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Null => "?".to_string(),
        other => other.to_string(),
    }
}

// ===========================================================================
// Backend: curator state (.curator_state) — mirrors agent/curator.py
// ===========================================================================

#[derive(Debug, Clone, Default)]
struct CuratorState {
    last_run_at: Option<String>,
    last_run_duration_seconds: Option<f64>,
    last_run_summary: Option<String>,
    last_report_path: Option<String>,
    paused: bool,
    run_count: i64,
}

fn state_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join("skills").join(".curator_state")
}

fn load_state(hermes_home: &Path) -> Result<CuratorState, Box<dyn Error>> {
    let path = state_path(hermes_home);
    if !path.exists() {
        return Ok(CuratorState::default());
    }
    let parsed: JsonValue = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or(JsonValue::Null),
        Err(_) => return Ok(CuratorState::default()),
    };
    let Some(obj) = parsed.as_object() else {
        return Ok(CuratorState::default());
    };
    Ok(CuratorState {
        last_run_at: obj
            .get("last_run_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_run_duration_seconds: obj
            .get("last_run_duration_seconds")
            .and_then(JsonValue::as_f64),
        last_run_summary: obj
            .get("last_run_summary")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_report_path: obj
            .get("last_report_path")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        paused: obj
            .get("paused")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false),
        run_count: obj.get("run_count").and_then(JsonValue::as_i64).unwrap_or(0),
    })
}

fn save_state(hermes_home: &Path, state: &CuratorState) -> Result<(), Box<dyn Error>> {
    let path = state_path(hermes_home);
    let mut obj = JsonMap::new();
    obj.insert(
        "last_run_at".to_string(),
        opt_json(state.last_run_at.clone()),
    );
    obj.insert(
        "last_run_duration_seconds".to_string(),
        state
            .last_run_duration_seconds
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
    );
    obj.insert(
        "last_run_summary".to_string(),
        opt_json(state.last_run_summary.clone()),
    );
    obj.insert(
        "last_report_path".to_string(),
        opt_json(state.last_report_path.clone()),
    );
    obj.insert("paused".to_string(), JsonValue::Bool(state.paused));
    obj.insert("run_count".to_string(), JsonValue::from(state.run_count));
    atomic_write_json(&path, &JsonValue::Object(obj))
}

fn set_paused(hermes_home: &Path, paused: bool) -> Result<(), Box<dyn Error>> {
    let mut state = load_state(hermes_home)?;
    state.paused = paused;
    save_state(hermes_home, &state)
}

// ===========================================================================
// Backend: config reads — mirrors agent/curator.py config helpers
// ===========================================================================

fn load_raw_config(config_path: &Path) -> Result<YamlValue, Box<dyn Error>> {
    if !config_path.exists() {
        return Ok(YamlValue::Null);
    }
    let text = fs::read_to_string(config_path)?;
    if text.trim().is_empty() {
        return Ok(YamlValue::Null);
    }
    Ok(serde_yaml::from_str(&text).unwrap_or(YamlValue::Null))
}

fn yaml_key(key: &str) -> YamlValue {
    YamlValue::String(key.to_string())
}

fn curator_section(root: &YamlValue) -> Option<&serde_yaml::Mapping> {
    root.as_mapping()?
        .get(yaml_key("curator"))
        .and_then(YamlValue::as_mapping)
}

fn yaml_to_i64(value: &YamlValue) -> Option<i64> {
    match value {
        YamlValue::Number(n) => n.as_i64(),
        YamlValue::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn is_enabled(config_path: &Path) -> Result<bool, Box<dyn Error>> {
    let raw = load_raw_config(config_path)?;
    let Some(curator) = curator_section(&raw) else {
        return Ok(true);
    };
    Ok(curator
        .get(yaml_key("enabled"))
        .and_then(YamlValue::as_bool)
        .unwrap_or(true))
}

fn config_int(config_path: &Path, key: &str, default: i64) -> Result<i64, Box<dyn Error>> {
    let raw = load_raw_config(config_path)?;
    let Some(curator) = curator_section(&raw) else {
        return Ok(default);
    };
    Ok(curator
        .get(yaml_key(key))
        .and_then(yaml_to_i64)
        .unwrap_or(default))
}

fn get_interval_hours(config_path: &Path) -> Result<i64, Box<dyn Error>> {
    config_int(config_path, "interval_hours", DEFAULT_INTERVAL_HOURS)
}

fn get_stale_after_days(config_path: &Path) -> Result<i64, Box<dyn Error>> {
    config_int(config_path, "stale_after_days", DEFAULT_STALE_AFTER_DAYS)
}

fn get_archive_after_days(config_path: &Path) -> Result<i64, Box<dyn Error>> {
    config_int(config_path, "archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS)
}

fn backup_enabled(config_path: &Path) -> Result<bool, Box<dyn Error>> {
    let raw = load_raw_config(config_path)?;
    let Some(curator) = curator_section(&raw) else {
        return Ok(true);
    };
    let Some(backup) = curator
        .get(yaml_key("backup"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(true);
    };
    Ok(backup
        .get(yaml_key("enabled"))
        .and_then(YamlValue::as_bool)
        .unwrap_or(true))
}

// ===========================================================================
// Backend: skill usage telemetry — mirrors tools/skill_usage.py
// ===========================================================================

/// A usage record as stored in `.usage.json`.
#[derive(Debug, Clone)]
struct UsageRecord {
    state: String,
    pinned: bool,
    created_at: Option<String>,
    last_used_at: Option<String>,
    last_viewed_at: Option<String>,
    last_patched_at: Option<String>,
    use_count: i64,
    view_count: i64,
    patch_count: i64,
    created_by: Option<String>,
    agent_created: bool,
    archived_at: Option<String>,
}

impl UsageRecord {
    fn empty() -> Self {
        UsageRecord {
            state: STATE_ACTIVE.to_string(),
            pinned: false,
            created_at: Some(now_iso()),
            last_used_at: None,
            last_viewed_at: None,
            last_patched_at: None,
            use_count: 0,
            view_count: 0,
            patch_count: 0,
            created_by: None,
            agent_created: false,
            archived_at: None,
        }
    }

    fn latest_activity_at(&self) -> Option<String> {
        let mut latest_dt: Option<DateTime<Utc>> = None;
        let mut latest_raw: Option<String> = None;
        for raw in [
            self.last_used_at.as_deref(),
            self.last_viewed_at.as_deref(),
            self.last_patched_at.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(dt) = parse_iso_timestamp(raw) {
                if latest_dt.is_none() || dt > latest_dt.unwrap() {
                    latest_dt = Some(dt);
                    latest_raw = Some(raw.to_string());
                }
            }
        }
        latest_raw
    }

    fn activity_count(&self) -> i64 {
        self.use_count
            .saturating_add(self.view_count)
            .saturating_add(self.patch_count)
    }

    fn is_curator_managed(&self) -> bool {
        self.created_by.as_deref() == Some("agent") || self.agent_created
    }
}

/// Flattened report row: `{name, ...record, last_activity_at, activity_count}`.
#[derive(Debug, Clone)]
pub struct ReportRow {
    pub name: String,
    pub state: String,
    pub pinned: bool,
    pub created_at: Option<String>,
    pub use_count: i64,
    pub view_count: i64,
    pub patch_count: i64,
    pub last_activity_at: Option<String>,
    pub activity_count: i64,
}

fn skills_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("skills")
}

fn usage_file(hermes_home: &Path) -> PathBuf {
    skills_dir(hermes_home).join(".usage.json")
}

fn archive_dir(hermes_home: &Path) -> PathBuf {
    skills_dir(hermes_home).join(".archive")
}

fn load_usage(hermes_home: &Path) -> HashMap<String, UsageRecord> {
    let path = usage_file(hermes_home);
    if !path.exists() {
        return HashMap::new();
    }
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let parsed: JsonValue = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let Some(obj) = parsed.as_object() else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for (name, value) in obj {
        if let Some(record_obj) = value.as_object() {
            out.insert(name.clone(), usage_from_json(record_obj));
        }
    }
    out
}

fn usage_from_json(obj: &JsonMap<String, JsonValue>) -> UsageRecord {
    UsageRecord {
        state: obj
            .get("state")
            .and_then(JsonValue::as_str)
            .filter(|v| VALID_STATES.contains(v))
            .unwrap_or(STATE_ACTIVE)
            .to_string(),
        pinned: obj.get("pinned").and_then(JsonValue::as_bool).unwrap_or(false),
        created_at: obj
            .get("created_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_used_at: obj
            .get("last_used_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_viewed_at: obj
            .get("last_viewed_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_patched_at: obj
            .get("last_patched_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        use_count: obj.get("use_count").and_then(JsonValue::as_i64).unwrap_or(0),
        view_count: obj.get("view_count").and_then(JsonValue::as_i64).unwrap_or(0),
        patch_count: obj.get("patch_count").and_then(JsonValue::as_i64).unwrap_or(0),
        created_by: obj
            .get("created_by")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        agent_created: obj
            .get("agent_created")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false),
        archived_at: obj
            .get("archived_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
    }
}

fn usage_to_json(record: &UsageRecord) -> JsonValue {
    let mut obj = JsonMap::new();
    obj.insert("created_by".to_string(), opt_json(record.created_by.clone()));
    obj.insert("use_count".to_string(), JsonValue::from(record.use_count));
    obj.insert("view_count".to_string(), JsonValue::from(record.view_count));
    obj.insert("last_used_at".to_string(), opt_json(record.last_used_at.clone()));
    obj.insert(
        "last_viewed_at".to_string(),
        opt_json(record.last_viewed_at.clone()),
    );
    obj.insert("patch_count".to_string(), JsonValue::from(record.patch_count));
    obj.insert(
        "last_patched_at".to_string(),
        opt_json(record.last_patched_at.clone()),
    );
    obj.insert("created_at".to_string(), opt_json(record.created_at.clone()));
    obj.insert("state".to_string(), JsonValue::String(record.state.clone()));
    obj.insert("pinned".to_string(), JsonValue::Bool(record.pinned));
    obj.insert("archived_at".to_string(), opt_json(record.archived_at.clone()));
    if record.agent_created {
        obj.insert("agent_created".to_string(), JsonValue::Bool(true));
    }
    JsonValue::Object(obj)
}

fn save_usage(hermes_home: &Path, usage: &HashMap<String, UsageRecord>) -> Result<(), Box<dyn Error>> {
    let path = usage_file(hermes_home);
    let mut root = JsonMap::new();
    let mut names: Vec<&String> = usage.keys().collect();
    names.sort();
    for name in names {
        if let Some(record) = usage.get(name) {
            root.insert(name.clone(), usage_to_json(record));
        }
    }
    atomic_write_json(&path, &JsonValue::Object(root))
}

fn get_record(hermes_home: &Path, skill: &str) -> Result<UsageRecord, Box<dyn Error>> {
    let usage = load_usage(hermes_home);
    Ok(usage.get(skill).cloned().unwrap_or_else(UsageRecord::empty))
}

fn mutate_record<F>(hermes_home: &Path, skill: &str, mutator: F) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut UsageRecord),
{
    if skill.trim().is_empty() {
        return Ok(());
    }
    if !is_agent_created(hermes_home, skill)? {
        return Ok(());
    }
    let mut usage = load_usage(hermes_home);
    let mut record = usage.remove(skill).unwrap_or_else(UsageRecord::empty);
    mutator(&mut record);
    usage.insert(skill.to_string(), record);
    save_usage(hermes_home, &usage)
}

fn set_pinned(hermes_home: &Path, skill: &str, pinned: bool) -> Result<(), Box<dyn Error>> {
    mutate_record(hermes_home, skill, |r| r.pinned = pinned)
}

fn set_state(hermes_home: &Path, skill: &str, state: &str) -> Result<(), Box<dyn Error>> {
    if !VALID_STATES.contains(&state) {
        return Ok(());
    }
    mutate_record(hermes_home, skill, |r| {
        r.state = state.to_string();
        if state == STATE_ARCHIVED {
            r.archived_at = Some(now_iso());
        } else if state == STATE_ACTIVE {
            r.archived_at = None;
        }
    })
}

// ---------------------------------------------------------------------------
// Provenance: bundled / hub names, agent-created enumeration
// ---------------------------------------------------------------------------

fn read_bundled_manifest_names(hermes_home: &Path) -> HashSet<String> {
    let manifest = skills_dir(hermes_home).join(".bundled_manifest");
    if !manifest.exists() {
        return HashSet::new();
    }
    let Ok(text) = fs::read_to_string(&manifest) else {
        return HashSet::new();
    };
    let mut names = HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let name = line.split_once(':').map(|(n, _)| n.trim()).unwrap_or(line);
        if !name.is_empty() {
            names.insert(name.to_string());
        }
    }
    names
}

fn read_hub_installed_names(hermes_home: &Path) -> HashSet<String> {
    let lock_path = skills_dir(hermes_home).join(".hub").join("lock.json");
    if !lock_path.exists() {
        return HashSet::new();
    }
    let Ok(text) = fs::read_to_string(&lock_path) else {
        return HashSet::new();
    };
    let Ok(parsed) = serde_json::from_str::<JsonValue>(&text) else {
        return HashSet::new();
    };
    let Some(installed) = parsed.get("installed").and_then(JsonValue::as_object) else {
        return HashSet::new();
    };
    let skills = skills_dir(hermes_home);
    let skills_resolved = skills.canonicalize().unwrap_or_else(|_| skills.clone());
    let mut names: HashSet<String> = installed.keys().cloned().collect();
    for value in installed.values() {
        let Some(entry) = value.as_object() else {
            continue;
        };
        let Some(install_path) = entry.get("install_path").and_then(JsonValue::as_str) else {
            continue;
        };
        if install_path.trim().is_empty() {
            continue;
        }
        let mut skill_dir = PathBuf::from(install_path);
        if !skill_dir.is_absolute() {
            skill_dir = skills.join(&skill_dir);
        }
        let resolved = skill_dir.canonicalize().unwrap_or(skill_dir);
        if resolved.strip_prefix(&skills_resolved).is_err() {
            continue;
        }
        let skill_md = resolved.join("SKILL.md");
        if skill_md.exists() {
            let fallback = resolved
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("skill");
            names.insert(read_skill_name(&skill_md, fallback));
        }
    }
    names
}

fn off_limits_names(hermes_home: &Path) -> HashSet<String> {
    let mut names = read_bundled_manifest_names(hermes_home);
    names.extend(read_hub_installed_names(hermes_home));
    names
}

fn is_agent_created(hermes_home: &Path, skill_name: &str) -> Result<bool, Box<dyn Error>> {
    Ok(!off_limits_names(hermes_home).contains(skill_name))
}

/// Walk `~/.hermes/skills/` for SKILL.md files (skipping dot-dirs and
/// node_modules) and parse the `name:` frontmatter field. Mirrors `rglob`.
fn collect_skill_markdown(base: &Path, root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            // Only skip dotted / node_modules dirs that are direct children of base
            // is handled separately by the rel.parts check; here we descend all,
            // but the caller's rel-based filter excludes them. To match Python's
            // rglob (which descends everywhere) we descend unconditionally but
            // let the rel filter drop hidden top-level trees.
            collect_skill_markdown(base, &path, out);
        } else if name == "SKILL.md" {
            out.push(path);
        }
    }
}

fn skill_markdown_files(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if base.is_dir() {
        collect_skill_markdown(base, base, &mut out);
    }
    out
}

fn list_agent_created_skill_names(hermes_home: &Path) -> Vec<String> {
    let base = skills_dir(hermes_home);
    if !base.exists() {
        return Vec::new();
    }
    let off_limits = off_limits_names(hermes_home);
    let usage = load_usage(hermes_home);
    let mut names: Vec<String> = Vec::new();
    for skill_md in skill_markdown_files(&base) {
        let Ok(rel) = skill_md.strip_prefix(&base) else {
            continue;
        };
        if let Some(first) = rel.components().next() {
            if let Some(part) = first.as_os_str().to_str() {
                if part.starts_with('.') || part == "node_modules" {
                    continue;
                }
            }
        }
        let fallback = skill_md
            .parent()
            .and_then(Path::file_name)
            .and_then(|v| v.to_str())
            .unwrap_or("skill");
        let name = read_skill_name(&skill_md, fallback);
        if off_limits.contains(&name) {
            continue;
        }
        match usage.get(&name) {
            Some(record) if record.is_curator_managed() => names.push(name),
            _ => continue,
        }
    }
    // sorted(set(names))
    let mut deduped: Vec<String> = names.into_iter().collect::<HashSet<_>>().into_iter().collect();
    deduped.sort();
    deduped
}

/// Build the agent-created report rows. Mirrors `agent_created_report`.
pub fn agent_created_report(hermes_home: &Path) -> Result<Vec<ReportRow>, Box<dyn Error>> {
    let usage = load_usage(hermes_home);
    let mut rows = Vec::new();
    for name in list_agent_created_skill_names(hermes_home) {
        let record = usage.get(&name).cloned().unwrap_or_else(UsageRecord::empty);
        rows.push(ReportRow {
            name,
            state: record.state.clone(),
            pinned: record.pinned,
            created_at: record.created_at.clone(),
            use_count: record.use_count,
            view_count: record.view_count,
            patch_count: record.patch_count,
            last_activity_at: record.latest_activity_at(),
            activity_count: record.activity_count(),
        });
    }
    Ok(rows)
}

fn read_skill_name(skill_md: &Path, fallback: &str) -> String {
    // Read up to 4000 bytes, mirroring the Python `[:4000]` cap.
    let text = match fs::read_to_string(skill_md) {
        Ok(t) => t,
        Err(_) => return fallback.to_string(),
    };
    let head: String = text.chars().take(4000).collect();
    let mut in_frontmatter = false;
    for line in head.split('\n') {
        let stripped = line.trim();
        if stripped == "---" {
            if in_frontmatter {
                break;
            }
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter && stripped.starts_with("name:") {
            let value = stripped
                .splitn(2, ':')
                .nth(1)
                .unwrap_or("")
                .trim()
                .trim_matches(|c| c == '"' || c == '\'');
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    fallback.to_string()
}

// ---------------------------------------------------------------------------
// Archive / restore — mirrors tools/skill_usage.py
// ---------------------------------------------------------------------------

fn find_skill_dir(hermes_home: &Path, skill_name: &str) -> Option<PathBuf> {
    let base = skills_dir(hermes_home);
    if !base.exists() {
        return None;
    }
    for skill_md in skill_markdown_files(&base) {
        let Ok(rel) = skill_md.strip_prefix(&base) else {
            continue;
        };
        if let Some(first) = rel.components().next() {
            if let Some(part) = first.as_os_str().to_str() {
                if part.starts_with('.') {
                    continue;
                }
            }
        }
        let fallback = skill_md
            .parent()
            .and_then(Path::file_name)
            .and_then(|v| v.to_str())
            .unwrap_or("skill");
        if read_skill_name(&skill_md, fallback) == skill_name {
            return skill_md.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn archive_skill(hermes_home: &Path, skill_name: &str) -> Result<(bool, String), Box<dyn Error>> {
    if !is_agent_created(hermes_home, skill_name)? {
        return Ok((
            false,
            format!("skill '{skill_name}' is bundled or hub-installed; never archive"),
        ));
    }
    let Some(skill_dir) = find_skill_dir(hermes_home, skill_name) else {
        return Ok((false, format!("skill '{skill_name}' not found")));
    };
    let archive_root = archive_dir(hermes_home);
    if let Err(e) = fs::create_dir_all(&archive_root) {
        return Ok((false, format!("failed to create archive dir: {e}")));
    }
    let base_name = skill_dir
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or(skill_name);
    let mut dest = archive_root.join(base_name);
    if dest.exists() {
        dest = archive_root.join(format!("{base_name}-{}", timestamp_suffix()));
    }
    if let Err(e) = move_directory(&skill_dir, &dest) {
        return Ok((false, format!("failed to archive: {e}")));
    }
    set_state(hermes_home, skill_name, STATE_ARCHIVED)?;
    Ok((true, format!("archived to {}", dest.display())))
}

fn restore_skill(hermes_home: &Path, skill_name: &str) -> Result<(bool, String), Box<dyn Error>> {
    if !is_agent_created(hermes_home, skill_name)? {
        return Ok((
            false,
            format!(
                "skill '{skill_name}' is now bundled or hub-installed; restore would shadow the upstream version"
            ),
        ));
    }
    let archive_root = archive_dir(hermes_home);
    if !archive_root.exists() {
        return Ok((false, "no archive directory".to_string()));
    }
    let mut exact = Vec::new();
    let mut prefixed = Vec::new();
    for dir in all_directories(&archive_root) {
        let name = dir.file_name().and_then(|v| v.to_str()).unwrap_or("");
        if name == skill_name {
            exact.push(dir);
        } else if name.starts_with(&format!("{skill_name}-")) {
            prefixed.push(dir);
        }
    }
    let src = if !exact.is_empty() {
        exact.sort();
        exact.remove(0)
    } else if !prefixed.is_empty() {
        prefixed.sort_by(|a, b| b.cmp(a));
        prefixed.remove(0)
    } else {
        return Ok((false, format!("skill '{skill_name}' not found in archive")));
    };
    let dest = skills_dir(hermes_home).join(skill_name);
    if dest.exists() {
        return Ok((false, format!("destination already exists: {}", dest.display())));
    }
    if let Err(e) = move_directory(&src, &dest) {
        return Ok((false, format!("failed to restore: {e}")));
    }
    set_state(hermes_home, skill_name, STATE_ACTIVE)?;
    Ok((true, format!("restored to {}", dest.display())))
}

fn all_directories(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_directories(root, &mut out);
    out
}

fn collect_directories(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.push(path.clone());
            collect_directories(&path, out);
        }
    }
}

// ===========================================================================
// Backup / rollback — local port of agent/curator_backup.py
//
// crates/hermes-core's `ag_curator_backup` is a private (non-exported) module,
// so this CLI shell carries its own self-contained implementation (the same
// approach the existing `curator_cmd.rs` takes). API shapes match the Python.
// ===========================================================================

const CURATOR_SKILLS_ARCHIVE: &str = "skills.tar.gz";
const CURATOR_CRON_JOBS_FILE: &str = "cron-jobs.json";
const CURATOR_EXCLUDE_TOP_LEVEL: [&str; 2] = [".curator_backups", ".hub"];
const CURATOR_BACKUP_DEFAULT_KEEP: i64 = 5;

fn backups_dir(hermes_home: &Path) -> PathBuf {
    skills_dir(hermes_home).join(".curator_backups")
}

fn cron_jobs_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join("cron").join("jobs.json")
}

/// `^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$` (snapshot id pattern).
fn is_snapshot_id(name: &str) -> bool {
    let bytes = name.as_bytes();
    // Minimum length "YYYY-MM-DDTHH-MM-SSZ" == 20.
    if bytes.len() != 20 && bytes.len() != 23 {
        return false;
    }
    let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    let lit = |i: usize, c: u8| bytes.get(i) == Some(&c);
    let core = digit(0)
        && digit(1)
        && digit(2)
        && digit(3)
        && lit(4, b'-')
        && digit(5)
        && digit(6)
        && lit(7, b'-')
        && digit(8)
        && digit(9)
        && lit(10, b'T')
        && digit(11)
        && digit(12)
        && lit(13, b'-')
        && digit(14)
        && digit(15)
        && lit(16, b'-')
        && digit(17)
        && digit(18)
        && lit(19, b'Z');
    if !core {
        return false;
    }
    if bytes.len() == 20 {
        return true;
    }
    lit(20, b'-') && digit(21) && digit(22)
}

fn utc_id() -> String {
    Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string()
}

fn backup_keep(config_path: &Path) -> Result<i64, Box<dyn Error>> {
    let raw = load_raw_config(config_path)?;
    let Some(curator) = curator_section(&raw) else {
        return Ok(CURATOR_BACKUP_DEFAULT_KEEP);
    };
    let Some(backup) = curator
        .get(yaml_key("backup"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(CURATOR_BACKUP_DEFAULT_KEEP);
    };
    let keep = backup
        .get(yaml_key("keep"))
        .and_then(yaml_to_i64)
        .unwrap_or(CURATOR_BACKUP_DEFAULT_KEEP);
    Ok(keep.max(1))
}

fn snapshot_skills(
    hermes_home: &Path,
    config_path: &Path,
    reason: &str,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if !backup_enabled(config_path)? {
        return Ok(None);
    }
    let skills = skills_dir(hermes_home);
    if !skills.exists() {
        return Ok(None);
    }
    let backups = backups_dir(hermes_home);
    if fs::create_dir_all(&backups).is_err() {
        return Ok(None);
    }

    let base_id = utc_id();
    let mut snap_id = base_id.clone();
    let mut counter = 1u32;
    while backups.join(&snap_id).exists() {
        snap_id = format!("{base_id}-{counter:02}");
        counter += 1;
    }
    let dest = backups.join(&snap_id);
    if fs::create_dir_all(&dest).is_err() {
        return Ok(None);
    }

    let archive_path = dest.join(CURATOR_SKILLS_ARCHIVE);
    let snapshot_result = (|| -> Result<(), Box<dyn Error>> {
        let archive_file = std::fs::File::create(&archive_path)?;
        let encoder = flate2::write::GzEncoder::new(archive_file, flate2::Compression::new(6));
        let mut builder = tar::Builder::new(encoder);
        let mut entries: Vec<PathBuf> = fs::read_dir(&skills)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for path in entries {
            let name = match path.file_name().and_then(|v| v.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if CURATOR_EXCLUDE_TOP_LEVEL.contains(&name.as_str()) {
                continue;
            }
            if path.is_dir() {
                builder.append_dir_all(Path::new(&name), &path)?;
            } else {
                builder.append_path_with_name(&path, Path::new(&name))?;
            }
        }
        builder.finish()?;
        builder.into_inner()?.finish()?;
        Ok(())
    })();
    if snapshot_result.is_err() {
        let _ = fs::remove_dir_all(&dest);
        return Ok(None);
    }

    let cron_info = backup_cron_jobs_into(hermes_home, &dest);
    if write_manifest(&dest, reason, &archive_path, count_skill_files(&skills), &cron_info).is_err()
    {
        let _ = fs::remove_dir_all(&dest);
        return Ok(None);
    }
    prune_old_backups(hermes_home, backup_keep(config_path)?);
    Ok(Some(dest))
}

fn count_skill_files(base: &Path) -> i64 {
    skill_markdown_files(base).len() as i64
}

fn backup_cron_jobs_into(hermes_home: &Path, dest: &Path) -> JsonValue {
    let source = cron_jobs_path(hermes_home);
    let mut info = JsonMap::new();
    info.insert("backed_up".to_string(), JsonValue::Bool(false));
    info.insert("jobs_count".to_string(), JsonValue::from(0));
    if !source.exists() {
        info.insert(
            "reason".to_string(),
            JsonValue::String("no cron/jobs.json present".to_string()),
        );
        return JsonValue::Object(info);
    }
    let raw = match fs::read_to_string(&source) {
        Ok(raw) => raw,
        Err(e) => {
            info.insert(
                "reason".to_string(),
                JsonValue::String(format!("read error: {e}")),
            );
            return JsonValue::Object(info);
        }
    };
    let jobs_count = serde_json::from_str::<JsonValue>(&raw)
        .ok()
        .and_then(|parsed| {
            parsed
                .get("jobs")
                .and_then(JsonValue::as_array)
                .map(|items| items.len())
                .or_else(|| parsed.as_array().map(|items| items.len()))
        })
        .unwrap_or(0);
    if fs::write(dest.join(CURATOR_CRON_JOBS_FILE), raw).is_err() {
        return JsonValue::Object(info);
    }
    info.insert("backed_up".to_string(), JsonValue::Bool(true));
    info.insert("jobs_count".to_string(), JsonValue::from(jobs_count as i64));
    JsonValue::Object(info)
}

fn write_manifest(
    dest: &Path,
    reason: &str,
    archive_path: &Path,
    skill_files: i64,
    cron_info: &JsonValue,
) -> Result<(), Box<dyn Error>> {
    let archive_bytes = archive_path.metadata()?.len();
    let mut manifest = JsonMap::new();
    manifest.insert(
        "id".to_string(),
        JsonValue::String(
            dest.file_name()
                .and_then(|v| v.to_str())
                .unwrap_or_default()
                .to_string(),
        ),
    );
    manifest.insert("reason".to_string(), JsonValue::String(reason.to_string()));
    manifest.insert("created_at".to_string(), JsonValue::String(now_iso()));
    manifest.insert(
        "archive".to_string(),
        JsonValue::String(CURATOR_SKILLS_ARCHIVE.to_string()),
    );
    manifest.insert("archive_bytes".to_string(), JsonValue::from(archive_bytes));
    manifest.insert("skill_files".to_string(), JsonValue::from(skill_files));

    // Mirror Python: only include the keys it writes for cron_jobs.
    if let Some(cron) = cron_info.as_object() {
        let backed_up = cron
            .get("backed_up")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let mut cron_out = JsonMap::new();
        cron_out.insert("backed_up".to_string(), JsonValue::Bool(backed_up));
        cron_out.insert(
            "jobs_count".to_string(),
            JsonValue::from(
                cron.get("jobs_count")
                    .and_then(JsonValue::as_i64)
                    .unwrap_or(0),
            ),
        );
        if !backed_up {
            cron_out.insert(
                "reason".to_string(),
                JsonValue::String(
                    cron.get("reason")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("not captured")
                        .to_string(),
                ),
            );
        }
        manifest.insert("cron_jobs".to_string(), JsonValue::Object(cron_out));
    }
    let serialized = serde_json::to_string_pretty(&JsonValue::Object(manifest))?;
    fs::write(dest.join("manifest.json"), serialized)?;
    Ok(())
}

fn prune_old_backups(hermes_home: &Path, keep: i64) -> Vec<String> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(&backups) else {
        return Vec::new();
    };
    let mut regular: Vec<(String, PathBuf)> = Vec::new();
    let mut staging: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(".rollback-staging-") {
            staging.push(path);
            continue;
        }
        if is_snapshot_id(&name) {
            regular.push((name, path));
        }
    }
    regular.sort_by(|a, b| b.0.cmp(&a.0));
    let mut deleted = Vec::new();
    for (name, path) in regular.into_iter().skip(keep.max(0) as usize) {
        if fs::remove_dir_all(&path).is_ok() {
            deleted.push(name);
        }
    }
    for path in staging {
        let _ = fs::remove_dir_all(path);
    }
    deleted
}

fn read_manifest(snapshot_dir: &Path) -> JsonValue {
    fs::read_to_string(snapshot_dir.join("manifest.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<JsonValue>(&raw).ok())
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()))
}

fn list_backups(hermes_home: &Path) -> Result<Vec<JsonValue>, Box<dyn Error>> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return Ok(Vec::new());
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(&backups)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    // sorted(..., reverse=True) on entry name.
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    let mut out = Vec::new();
    for child in dirs {
        if !child.is_dir() {
            continue;
        }
        let name = match child.file_name().and_then(|v| v.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !is_snapshot_id(&name) {
            continue;
        }
        if !child.join(CURATOR_SKILLS_ARCHIVE).exists() {
            continue;
        }
        let mut manifest = read_manifest(&child);
        let obj = manifest
            .as_object_mut()
            .expect("read_manifest always returns an object");
        obj.entry("id".to_string())
            .or_insert_with(|| JsonValue::String(name.clone()));
        obj.entry("path".to_string())
            .or_insert_with(|| JsonValue::String(child.display().to_string()));
        if !obj.contains_key("archive_bytes") {
            let bytes = child
                .join(CURATOR_SKILLS_ARCHIVE)
                .metadata()
                .map(|m| m.len())
                .unwrap_or(0);
            obj.insert("archive_bytes".to_string(), JsonValue::from(bytes));
        }
        out.push(manifest);
    }
    Ok(out)
}

fn resolve_backup(
    hermes_home: &Path,
    backup_id: Option<&str>,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return Ok(None);
    }
    if let Some(id) = backup_id {
        let target = backups.join(id);
        if target.is_dir() && is_snapshot_id(id) && target.join(CURATOR_SKILLS_ARCHIVE).exists() {
            return Ok(Some(target));
        }
        return Ok(None);
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(&backups)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    for child in dirs {
        let name = child.file_name().and_then(|v| v.to_str()).unwrap_or("");
        if child.is_dir()
            && is_snapshot_id(name)
            && child.join(CURATOR_SKILLS_ARCHIVE).exists()
        {
            return Ok(Some(child));
        }
    }
    Ok(None)
}

fn summarize_backups(hermes_home: &Path) -> Result<String, Box<dyn Error>> {
    let rows = list_backups(hermes_home)?;
    if rows.is_empty() {
        return Ok("No curator snapshots yet.".to_string());
    }
    let mut lines = Vec::new();
    let header = format!("{:<24}  {:<40}  {:>6}  {:>8}", "id", "reason", "skills", "size");
    let sep = "─".repeat(header.chars().count());
    lines.push(header);
    lines.push(sep);
    for row in rows {
        let id = row.get("id").and_then(JsonValue::as_str).unwrap_or("?");
        let reason = row.get("reason").and_then(JsonValue::as_str).unwrap_or("?");
        let skills = row
            .get("skill_files")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        let size = row
            .get("archive_bytes")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0);
        lines.push(format!(
            "{:<24}  {:<40}  {:>6}  {:>8}",
            id,
            reason.chars().take(40).collect::<String>(),
            skills,
            format_size(size)
        ));
    }
    Ok(lines.join("\n"))
}

fn format_size(bytes: u64) -> String {
    let mut value = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if value < 1024.0 || unit == "GB" {
            return if unit == "B" {
                format!("{} {unit}", value as u64)
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1024.0;
    }
    format!("{value:.1} GB")
}

fn rollback(
    hermes_home: &Path,
    _config_path: &Path,
    backup_id: Option<&str>,
) -> Result<(bool, String, Option<PathBuf>), Box<dyn Error>> {
    let Some(target) = resolve_backup(hermes_home, backup_id)? else {
        return Ok((false, "no matching backup found".to_string(), None));
    };
    let archive = target.join(CURATOR_SKILLS_ARCHIVE);
    let target_name = target
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("?")
        .to_string();
    if !archive.exists() {
        return Ok((
            false,
            format!("snapshot {target_name} has no {CURATOR_SKILLS_ARCHIVE}"),
            None,
        ));
    }

    let skills = skills_dir(hermes_home);
    let backups = backups_dir(hermes_home);
    fs::create_dir_all(&skills)?;
    fs::create_dir_all(&backups)?;

    // Safety snapshot of the current tree first (undo handle).
    let _ = snapshot_skills(hermes_home, _config_path, &format!("pre-rollback to {target_name}"))?;

    let staged = backups.join(format!(".rollback-staging-{}", utc_id()));
    fs::create_dir_all(&staged)?;
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&skills)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if CURATOR_EXCLUDE_TOP_LEVEL.contains(&name.as_str()) {
            continue;
        }
        let dest = staged.join(&name);
        move_directory(&path, &dest)?;
        moved.push((path, dest));
    }

    if let Err(error) = extract_backup_archive(&archive, &skills) {
        for (original, staged_path) in moved {
            let _ = move_directory(&staged_path, &original);
        }
        let _ = fs::remove_dir_all(&staged);
        return Ok((
            false,
            format!("snapshot extract failed (state restored): {error}"),
            None,
        ));
    }

    let _ = fs::remove_dir_all(&staged);
    let cron_report = restore_cron_skill_links(hermes_home, &target)?;
    let mut summary = format!("restored from snapshot {target_name}");
    if cron_report.attempted {
        if let Some(error) = cron_report.error {
            summary.push_str(&format!("; cron links: error — {error}"));
        } else {
            let mut parts = Vec::new();
            if cron_report.restored > 0 {
                parts.push(format!("{} job(s) had skill links restored", cron_report.restored));
            }
            if cron_report.skipped_missing > 0 {
                parts.push(format!(
                    "{} backed-up job(s) no longer exist (skipped)",
                    cron_report.skipped_missing
                ));
            }
            if cron_report.unchanged > 0 {
                parts.push(format!("{} already matched", cron_report.unchanged));
            }
            if !parts.is_empty() {
                summary.push_str(&format!("; cron links: {}", parts.join(", ")));
            }
        }
    }
    Ok((true, summary, Some(target)))
}

fn extract_backup_archive(archive_path: &Path, skills_dir: &Path) -> Result<(), Box<dyn Error>> {
    use std::path::Component;
    let file = std::fs::File::open(archive_path)?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    for entry_result in archive.entries()? {
        let mut entry = entry_result?;
        let path = entry.path()?.into_owned();
        if path.is_absolute()
            || path
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(format!("refusing to extract unsafe path: {path:?}").into());
        }
        entry.unpack_in(skills_dir)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct CronRestoreReport {
    attempted: bool,
    restored: usize,
    skipped_missing: usize,
    unchanged: usize,
    error: Option<String>,
}

fn restore_cron_skill_links(
    hermes_home: &Path,
    snapshot_dir: &Path,
) -> Result<CronRestoreReport, Box<dyn Error>> {
    let backup_file = snapshot_dir.join(CURATOR_CRON_JOBS_FILE);
    if !backup_file.exists() {
        return Ok(CronRestoreReport {
            error: Some(format!("snapshot has no {CURATOR_CRON_JOBS_FILE}")),
            ..Default::default()
        });
    }
    let backup = serde_json::from_str::<JsonValue>(&fs::read_to_string(&backup_file)?)?;
    let Some(backup_jobs) = json_jobs_list(&backup) else {
        return Ok(CronRestoreReport {
            error: Some("backed-up cron-jobs.json has no jobs list".to_string()),
            ..Default::default()
        });
    };
    let live_path = cron_jobs_path(hermes_home);
    if !live_path.exists() {
        return Ok(CronRestoreReport {
            attempted: true,
            ..Default::default()
        });
    }
    let mut live = serde_json::from_str::<JsonValue>(&fs::read_to_string(&live_path)?)?;
    let mut backup_by_id: HashMap<String, JsonValue> = HashMap::new();
    for job in backup_jobs {
        if let Some(id) = job.get("id").and_then(JsonValue::as_str) {
            backup_by_id.insert(id.to_string(), job.clone());
        }
    }

    let Some(live_jobs) = json_jobs_list_mut(&mut live) else {
        return Ok(CronRestoreReport {
            attempted: true,
            error: Some("live cron/jobs.json has no jobs list".to_string()),
            ..Default::default()
        });
    };

    let mut report = CronRestoreReport {
        attempted: true,
        ..Default::default()
    };
    let mut live_ids: HashSet<String> = HashSet::new();
    let mut changed = false;
    for live_job in live_jobs.iter_mut() {
        let Some(id) = live_job.get("id").and_then(JsonValue::as_str).map(str::to_string) else {
            continue;
        };
        live_ids.insert(id.clone());
        let Some(backup_job) = backup_by_id.get(&id) else {
            continue;
        };
        let current_skills = live_job.get("skills").cloned();
        let current_skill = live_job.get("skill").cloned();
        let backup_skills = backup_job.get("skills").cloned();
        let backup_skill = backup_job.get("skill").cloned();
        if current_skills == backup_skills && current_skill == backup_skill {
            report.unchanged += 1;
            continue;
        }
        let Some(object) = live_job.as_object_mut() else {
            continue;
        };
        match backup_skills {
            Some(value) => {
                object.insert("skills".to_string(), value);
            }
            None => {
                object.remove("skills");
            }
        }
        match backup_skill {
            Some(value) => {
                object.insert("skill".to_string(), value);
            }
            None => {
                object.remove("skill");
            }
        }
        report.restored += 1;
        changed = true;
    }
    for id in backup_by_id.keys() {
        if !live_ids.contains(id) {
            report.skipped_missing += 1;
        }
    }
    if changed {
        atomic_write_json(&live_path, &live)?;
    }
    Ok(report)
}

fn json_jobs_list(value: &JsonValue) -> Option<&Vec<JsonValue>> {
    value
        .get("jobs")
        .and_then(JsonValue::as_array)
        .or_else(|| value.as_array())
}

fn json_jobs_list_mut(value: &mut JsonValue) -> Option<&mut Vec<JsonValue>> {
    match value {
        JsonValue::Array(array) => Some(array),
        JsonValue::Object(object) => object.get_mut("jobs").and_then(JsonValue::as_array_mut),
        _ => None,
    }
}

// ===========================================================================
// Shared helpers
// ===========================================================================

fn parse_iso_timestamp(value: &str) -> Option<DateTime<Utc>> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.with_timezone(&Utc));
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(parsed) = NaiveDateTime::parse_from_str(value, fmt) {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
        }
    }
    None
}

fn now_iso() -> String {
    Utc::now().to_rfc3339()
}

fn timestamp_suffix() -> String {
    Utc::now().format("%Y%m%d%H%M%S").to_string()
}

fn opt_json(value: Option<String>) -> JsonValue {
    value.map(JsonValue::String).unwrap_or(JsonValue::Null)
}

fn atomic_write_json(path: &Path, value: &JsonValue) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let serialized = serde_json::to_string_pretty(value)?;
    fs::write(&tmp, serialized)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn move_directory(src: &Path, dest: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::rename(src, dest) {
        Ok(()) => Ok(()),
        Err(_) => {
            copy_dir_recursive(src, dest)?;
            fs::remove_dir_all(src)?;
            Ok(())
        }
    }
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}

/// Read a `[y/N]` confirmation from stdin. `Err` represents EOF / interrupt
/// (the Python `except (EOFError, KeyboardInterrupt)` branch).
fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    let read = io::stdin().read_line(&mut input)?;
    if read == 0 {
        return Err("eof".into());
    }
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_home() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "hermes-cli-curator-test-{}-{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("skills")).unwrap();
        dir
    }

    fn write_skill(home: &Path, dir_name: &str, name: &str) {
        let skill_dir = home.join("skills").join(dir_name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: x\n---\nbody\n"),
        )
        .unwrap();
    }

    fn write_usage(home: &Path, value: JsonValue) {
        fs::write(
            home.join("skills").join(".usage.json"),
            serde_json::to_string_pretty(&value).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn fmt_ts_relative_and_never() {
        assert_eq!(fmt_ts(None), "never");
        assert_eq!(fmt_ts(Some("")), "never");
        // Unparseable returns the raw string.
        assert_eq!(fmt_ts(Some("not-a-date")), "not-a-date");
        let recent = (Utc::now() - chrono::Duration::seconds(30)).to_rfc3339();
        assert_eq!(fmt_ts(Some(&recent)), "30s ago");
        let mins = (Utc::now() - chrono::Duration::seconds(125)).to_rfc3339();
        assert_eq!(fmt_ts(Some(&mins)), "2m ago");
        let days = (Utc::now() - chrono::Duration::days(3)).to_rfc3339();
        assert_eq!(fmt_ts(Some(&days)), "3d ago");
    }

    #[test]
    fn state_roundtrip_and_pause() {
        let home = temp_home();
        // No state file → defaults, not paused.
        let s = load_state(&home).unwrap();
        assert!(!s.paused);
        assert_eq!(s.run_count, 0);

        set_paused(&home, true).unwrap();
        let s = load_state(&home).unwrap();
        assert!(s.paused);

        set_paused(&home, false).unwrap();
        let s = load_state(&home).unwrap();
        assert!(!s.paused);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn config_defaults_and_overrides() {
        let home = temp_home();
        let config = home.join("config.yaml");
        // Missing config → enabled true, default intervals.
        assert!(is_enabled(&config).unwrap());
        assert_eq!(get_interval_hours(&config).unwrap(), DEFAULT_INTERVAL_HOURS);
        assert_eq!(get_stale_after_days(&config).unwrap(), DEFAULT_STALE_AFTER_DAYS);

        fs::write(
            &config,
            "curator:\n  enabled: false\n  interval_hours: 48\n  stale_after_days: 7\n  backup:\n    enabled: false\n",
        )
        .unwrap();
        assert!(!is_enabled(&config).unwrap());
        assert_eq!(get_interval_hours(&config).unwrap(), 48);
        assert_eq!(get_stale_after_days(&config).unwrap(), 7);
        assert!(!backup_enabled(&config).unwrap());
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn report_only_includes_agent_created() {
        let home = temp_home();
        write_skill(&home, "alpha", "alpha");
        write_skill(&home, "beta", "beta");
        write_skill(&home, "gamma", "gamma");
        write_usage(
            &home,
            serde_json::json!({
                "alpha": {"created_by": "agent", "use_count": 3, "view_count": 1, "state": "active"},
                "beta": {"agent_created": true, "patch_count": 2, "state": "active"},
                "gamma": {"created_by": "human", "use_count": 99}
            }),
        );
        let rows = agent_created_report(&home).unwrap();
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        // gamma is not curator-managed → excluded.
        assert!(!names.contains(&"gamma"));
        let alpha = rows.iter().find(|r| r.name == "alpha").unwrap();
        assert_eq!(alpha.activity_count, 4);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn pin_then_archive_refused_until_unpinned() {
        let home = temp_home();
        write_skill(&home, "alpha", "alpha");
        write_usage(
            &home,
            serde_json::json!({"alpha": {"created_by": "agent", "state": "active"}}),
        );
        // Pin succeeds.
        assert_eq!(cmd_pin(&home, "alpha").unwrap(), 0);
        assert!(get_record(&home, "alpha").unwrap().pinned);
        // Archive refused (exit 1) while pinned.
        assert_eq!(cmd_archive(&home, "alpha").unwrap(), 1);
        // Unpin then archive succeeds.
        assert_eq!(cmd_unpin(&home, "alpha").unwrap(), 0);
        assert_eq!(cmd_archive(&home, "alpha").unwrap(), 0);
        assert_eq!(get_record(&home, "alpha").unwrap().state, STATE_ARCHIVED);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn pin_bundled_skill_refused() {
        let home = temp_home();
        fs::write(
            home.join("skills").join(".bundled_manifest"),
            "bundled-skill:abc123\n",
        )
        .unwrap();
        assert_eq!(cmd_pin(&home, "bundled-skill").unwrap(), 1);
        assert_eq!(cmd_unpin(&home, "bundled-skill").unwrap(), 1);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn prune_rejects_bad_days() {
        let home = temp_home();
        let config = home.join("config.yaml");
        let args = PruneArgs {
            days: 0,
            yes: true,
            dry_run: false,
        };
        assert_eq!(cmd_prune(&home, &config, &args).unwrap(), 2);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn prune_dry_run_lists_idle() {
        let home = temp_home();
        let config = home.join("config.yaml");
        write_skill(&home, "old", "old");
        let stale = (Utc::now() - chrono::Duration::days(200)).to_rfc3339();
        write_usage(
            &home,
            serde_json::json!({
                "old": {"created_by": "agent", "state": "active", "last_used_at": stale}
            }),
        );
        let args = PruneArgs {
            days: 90,
            yes: false,
            dry_run: true,
        };
        // dry-run never prompts and returns 0.
        assert_eq!(cmd_prune(&home, &config, &args).unwrap(), 0);
        // Skill still present (not archived).
        assert_eq!(get_record(&home, "old").unwrap().state, STATE_ACTIVE);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn idle_days_falls_back_to_created_at() {
        let created = (Utc::now() - chrono::Duration::days(10)).to_rfc3339();
        let row = ReportRow {
            name: "x".to_string(),
            state: STATE_ACTIVE.to_string(),
            pinned: false,
            created_at: Some(created),
            use_count: 0,
            view_count: 0,
            patch_count: 0,
            last_activity_at: None,
            activity_count: 0,
        };
        assert_eq!(idle_days(&row), Some(10));

        let empty = ReportRow {
            name: "y".to_string(),
            state: STATE_ACTIVE.to_string(),
            pinned: false,
            created_at: None,
            use_count: 0,
            view_count: 0,
            patch_count: 0,
            last_activity_at: None,
            activity_count: 0,
        };
        assert_eq!(idle_days(&empty), None);
    }

    #[test]
    fn run_disabled_returns_one() {
        let home = temp_home();
        let config = home.join("config.yaml");
        fs::write(&config, "curator:\n  enabled: false\n").unwrap();
        let runner = DefaultRunner;
        let code = cmd_run(&home, &config, &RunArgs::default(), &runner).unwrap();
        assert_eq!(code, 1);
        let _ = fs::remove_dir_all(&home);
    }
}
