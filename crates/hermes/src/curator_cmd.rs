use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use chrono::{DateTime, NaiveDateTime, Utc};
use clap::{Args, Subcommand};
use flate2::Compression;
use flate2::write::GzEncoder;
use hermes_core::HermesContext;
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value as YamlValue;
use tar::{Archive, Builder};

use crate::python_bridge::{project_root, resolve_repo_python};

const DEFAULT_INTERVAL_HOURS: i64 = 24 * 7;
const DEFAULT_STALE_AFTER_DAYS: i64 = 30;
const DEFAULT_ARCHIVE_AFTER_DAYS: i64 = 90;
const STATE_ACTIVE: &str = "active";
const STATE_STALE: &str = "stale";
const STATE_ARCHIVED: &str = "archived";
const VALID_STATES: [&str; 3] = [STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED];

#[derive(Subcommand, Debug)]
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

#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    #[arg(long = "sync", visible_alias = "synchronous", default_value_t = false)]
    pub synchronous: bool,
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,
}

#[derive(Args, Debug, Clone)]
pub struct PruneArgs {
    #[arg(long, default_value_t = 90)]
    pub days: i64,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,
}

#[derive(Args, Debug, Clone)]
pub struct BackupArgs {
    #[arg(long)]
    pub reason: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct RollbackArgs {
    #[arg(long = "list", default_value_t = false)]
    pub list: bool,
    #[arg(long = "id")]
    pub backup_id: Option<String>,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

#[derive(Debug, Clone)]
struct CuratorState {
    last_run_at: Option<String>,
    last_run_duration_seconds: Option<f64>,
    last_run_summary: Option<String>,
    last_report_path: Option<String>,
    paused: bool,
    run_count: i64,
}

#[derive(Debug, Clone)]
struct UsageRecord {
    name: String,
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

pub fn print_curator(
    context: &HermesContext,
    command: Option<CuratorCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => {
            print_help_summary();
            Ok(())
        }
        Some(CuratorCommand::Status) => print_status(context),
        Some(CuratorCommand::Run(args)) => print_curator_run(context, args),
        Some(CuratorCommand::Pause) => {
            set_paused(context, true)?;
            println!("curator: paused");
            Ok(())
        }
        Some(CuratorCommand::Resume) => {
            set_paused(context, false)?;
            println!("curator: resumed");
            Ok(())
        }
        Some(CuratorCommand::Pin { skill }) => set_pinned_command(context, &skill, true),
        Some(CuratorCommand::Unpin { skill }) => set_pinned_command(context, &skill, false),
        Some(CuratorCommand::Restore { skill }) => restore_command(context, &skill),
        Some(CuratorCommand::Archive { skill }) => archive_command(context, &skill),
        Some(CuratorCommand::Prune(args)) => prune_command(context, args),
        Some(CuratorCommand::Backup(args)) => print_curator_backup(context, args),
        Some(CuratorCommand::Rollback(args)) => print_curator_rollback(context, args),
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

fn print_curator_run(context: &HermesContext, args: RunArgs) -> Result<(), Box<dyn Error>> {
    if !curator_enabled(context)? {
        println!("curator: disabled via config; enable with `curator.enabled: true`");
        return Err("curator disabled".into());
    }

    if args.dry_run {
        println!("curator: running DRY-RUN (report only, no mutations)...");
    } else {
        println!("curator: running review pass...");
    }

    if !args.synchronous {
        launch_detached_curator_run(context, args.dry_run)?;
        println!("llm pass running in background — check `hermes curator status` later");
        if args.dry_run {
            println!(
                "dry-run: no changes applied. When the report lands, read it with `hermes curator status` and run `hermes curator run` (no flag) to apply."
            );
        }
        return Ok(());
    }

    let mut envs = vec![(
        "HERMES_CURATOR_RUN_DRY".to_string(),
        if args.dry_run { "1" } else { "0" }.to_string(),
    )];
    run_curator_python(CURATOR_RUN_SYNC_BOOTSTRAP, &mut envs)
}

fn launch_detached_curator_run(
    context: &HermesContext,
    dry_run: bool,
) -> Result<(), Box<dyn Error>> {
    let binary = env::var_os("HERMES_CURATOR_BINARY")
        .map(PathBuf::from)
        .unwrap_or(env::current_exe()?);
    let mut command = Command::new(binary);
    command
        .arg("curator")
        .arg("run")
        .arg("--sync")
        .env("HERMES_HOME", context.hermes_home())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if dry_run {
        command.arg("--dry-run");
    }
    #[cfg(unix)]
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let _child = command.spawn()?;
    Ok(())
}

fn print_curator_backup(context: &HermesContext, args: BackupArgs) -> Result<(), Box<dyn Error>> {
    if !curator_backup_enabled(&context)? {
        println!(
            "curator: backups are disabled via config (`curator.backup.enabled: false`); re-enable to snapshot"
        );
        return Err("curator backups disabled".into());
    }
    let reason = args
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("manual");
    let Some(snapshot) = snapshot_skills(&context, reason)? else {
        println!("curator: snapshot failed — check logs (backup disabled or IO error)");
        return Err("curator snapshot failed".into());
    };
    println!("curator: snapshot created at {}", snapshot.display());
    Ok(())
}

fn print_curator_rollback(
    context: &HermesContext,
    args: RollbackArgs,
) -> Result<(), Box<dyn Error>> {
    if args.list {
        println!("{}", summarize_backups(context)?);
        return Ok(());
    }

    let backup_id = args
        .backup_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(target) = resolve_backup(context, backup_id)? else {
        let rows = list_backups(context)?;
        if rows.is_empty() {
            println!(
                "curator: no snapshots exist yet. Take one with `hermes curator backup` or wait for the next curator run."
            );
        } else {
            let label = backup_id
                .map(|value| format!("id '{value}'"))
                .unwrap_or_else(|| "your query".to_string());
            println!("curator: no snapshot matching {label}.");
            println!("Available:");
            println!("{}", summarize_backups(context)?);
        }
        return Err("curator rollback snapshot not found".into());
    };

    let manifest = read_backup_manifest(&target);
    println!(
        "Rollback target: {}",
        target
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("?")
    );
    if let Some(reason) = manifest.get("reason").and_then(JsonValue::as_str) {
        println!("  reason:      {reason}");
    }
    if let Some(created_at) = manifest.get("created_at").and_then(JsonValue::as_str) {
        println!("  created_at:  {created_at}");
    }
    if let Some(skill_files) = manifest.get("skill_files").and_then(JsonValue::as_i64) {
        println!("  skill files: {skill_files}");
    }
    if let Some(cron) = manifest.get("cron_jobs").and_then(JsonValue::as_object) {
        if cron
            .get("backed_up")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false)
        {
            println!(
                "  cron jobs:   {} (will restore skill-link fields only)",
                cron.get("jobs_count")
                    .and_then(JsonValue::as_i64)
                    .unwrap_or(0)
            );
        } else if let Some(reason) = cron.get("reason").and_then(JsonValue::as_str) {
            println!("  cron jobs:   not in snapshot ({reason})");
        }
    }
    println!(
        "\nThis will replace the current ~/.hermes/skills/ tree. A safety snapshot of the current state is taken first so this is undoable. Cron jobs that still exist will have only their skills/skill fields restored from the snapshot."
    );
    if !args.yes && !confirm_prompt("Proceed? [y/N] ")? {
        println!("cancelled");
        return Err("curator rollback cancelled".into());
    }

    let (ok, msg, _) = rollback_skills(context, backup_id)?;
    if ok {
        println!("curator: {msg}");
        Ok(())
    } else {
        println!("curator: rollback failed — {msg}");
        Err(msg.into())
    }
}

fn run_curator_python(
    bootstrap: &str,
    extra_env: &mut Vec<(String, String)>,
) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CURATOR_PYTHON"))
        .ok_or("could not find a Python interpreter for curator")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string());
    for (key, value) in extra_env.drain(..) {
        command.env(key, value);
    }
    command.arg("-c").arg(bootstrap);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("curator", status).into())
}

const CURATOR_RUN_SYNC_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from agent import curator\n",
    "dry = (os.environ.get('HERMES_CURATOR_RUN_DRY') == '1')\n",
    "def _on_summary(msg):\n",
    "    print(msg)\n",
    "result = curator.run_curator_review(on_summary=_on_summary, synchronous=True, dry_run=dry)\n",
    "auto = result.get('auto_transitions', {}) or {}\n",
    "if dry:\n",
    "    print(f\"auto (preview): {auto.get('checked', 0)} candidate skill(s) — no transitions applied in dry-run\")\n",
    "else:\n",
    "    print(f\"auto: checked={auto.get('checked', 0)} stale={auto.get('marked_stale', 0)} archived={auto.get('archived', 0)} reactivated={auto.get('reactivated', 0)}\")\n",
    "if dry:\n",
    "    print(\"dry-run: no changes applied. When the report lands, read it with `hermes curator status` and run `hermes curator run` (no flag) to apply.\")\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn print_status(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let state = load_curator_state(context)?;
    let enabled = curator_enabled(context)?;
    let status_line = if enabled && !state.paused {
        "ENABLED"
    } else if state.paused {
        "PAUSED"
    } else {
        "DISABLED"
    };

    println!("curator: {status_line}");
    println!("  runs:           {}", state.run_count);
    println!("  last run:       {}", fmt_ts(state.last_run_at.as_deref()));
    println!(
        "  last summary:   {}",
        state.last_run_summary.as_deref().unwrap_or("(none)")
    );
    if let Some(report) = state
        .last_report_path
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        println!("  last report:    {report}");
    }
    println!(
        "  interval:       every {}",
        format_interval(curator_interval_hours(context)?)
    );
    println!(
        "  stale after:    {}d unused",
        curator_stale_after_days(context)?
    );
    println!(
        "  archive after:  {}d unused",
        curator_archive_after_days(context)?
    );

    let rows = agent_created_report(context)?;
    if rows.is_empty() {
        println!("\nno agent-created skills");
        return Ok(());
    }

    let mut active = Vec::new();
    let mut stale = Vec::new();
    let mut archived = Vec::new();
    let mut pinned = Vec::new();
    for row in &rows {
        match row.state.as_str() {
            STATE_ACTIVE => active.push(row.clone()),
            STATE_STALE => stale.push(row.clone()),
            STATE_ARCHIVED => archived.push(row.clone()),
            _ => active.push(row.clone()),
        }
        if row.pinned {
            pinned.push(row.name.clone());
        }
    }

    println!("\nagent-created skills: {} total", rows.len());
    println!("  active     {}", active.len());
    println!("  stale      {}", stale.len());
    println!("  archived   {}", archived.len());

    if !pinned.is_empty() {
        pinned.sort();
        println!("\npinned ({}): {}", pinned.len(), pinned.join(", "));
    }

    let mut least_recent = active.clone();
    least_recent.sort_by_key(|row| {
        row.last_activity_at()
            .or_else(|| row.created_at_dt())
            .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
    });
    least_recent.truncate(5);
    if !least_recent.is_empty() {
        println!("\nleast recently active (top 5):");
        for row in &least_recent {
            println!(
                "  {:40}  activity={:3}  use={:3}  view={:3}  patches={:3}  last_activity={}",
                row.name,
                row.activity_count(),
                row.use_count,
                row.view_count,
                row.patch_count,
                fmt_ts(
                    row.last_activity_at()
                        .map(|value| value.to_rfc3339())
                        .as_deref()
                ),
            );
        }
    }

    let mut most_active = active.clone();
    most_active.sort_by(|left, right| {
        let left_key = (left.activity_count(), left.last_activity_at());
        let right_key = (right.activity_count(), right.last_activity_at());
        right_key.cmp(&left_key)
    });
    most_active.truncate(5);
    if most_active
        .first()
        .is_some_and(|row| row.activity_count() > 0)
    {
        println!("\nmost active (top 5):");
        for row in &most_active {
            println!(
                "  {:40}  activity={:3}  use={:3}  view={:3}  patches={:3}  last_activity={}",
                row.name,
                row.activity_count(),
                row.use_count,
                row.view_count,
                row.patch_count,
                fmt_ts(
                    row.last_activity_at()
                        .map(|value| value.to_rfc3339())
                        .as_deref()
                ),
            );
        }
    }

    let mut least_active = active;
    least_active.sort_by(|left, right| {
        let left_key = (left.activity_count(), left.last_activity_at());
        let right_key = (right.activity_count(), right.last_activity_at());
        left_key.cmp(&right_key)
    });
    least_active.truncate(5);
    if !least_active.is_empty() {
        println!("\nleast active (top 5):");
        for row in &least_active {
            println!(
                "  {:40}  activity={:3}  use={:3}  view={:3}  patches={:3}  last_activity={}",
                row.name,
                row.activity_count(),
                row.use_count,
                row.view_count,
                row.patch_count,
                fmt_ts(
                    row.last_activity_at()
                        .map(|value| value.to_rfc3339())
                        .as_deref()
                ),
            );
        }
    }

    Ok(())
}

fn set_pinned_command(
    context: &HermesContext,
    raw_skill: &str,
    pinned: bool,
) -> Result<(), Box<dyn Error>> {
    let skill = validate_skill_name(raw_skill)?;
    if !is_agent_created(context, skill)? {
        if pinned {
            println!(
                "curator: '{}' is bundled or hub-installed — cannot pin (only agent-created skills participate in curation)",
                skill
            );
        } else {
            println!(
                "curator: '{}' is bundled or hub-installed — there's nothing to unpin (curator only tracks agent-created skills)",
                skill
            );
        }
        return Ok(());
    }
    mutate_usage_record(context, skill, |record| {
        record.pinned = pinned;
    })?;
    if pinned {
        println!("curator: pinned '{}' (will bypass auto-transitions)", skill);
    } else {
        println!("curator: unpinned '{}'", skill);
    }
    Ok(())
}

fn restore_command(context: &HermesContext, raw_skill: &str) -> Result<(), Box<dyn Error>> {
    let skill = validate_skill_name(raw_skill)?;
    let (ok, msg) = restore_skill(context, skill)?;
    println!("curator: {msg}");
    if ok { Ok(()) } else { Err(msg.into()) }
}

fn archive_command(context: &HermesContext, raw_skill: &str) -> Result<(), Box<dyn Error>> {
    let skill = validate_skill_name(raw_skill)?;
    if get_usage_record(context, skill)?.pinned {
        let msg = format!(
            "curator: '{}' is pinned — unpin first with `hermes curator unpin {}`",
            skill, skill
        );
        println!("{msg}");
        return Err(msg.into());
    }
    let (ok, msg) = archive_skill(context, skill)?;
    println!("curator: {msg}");
    if ok { Ok(()) } else { Err(msg.into()) }
}

fn prune_command(context: &HermesContext, args: PruneArgs) -> Result<(), Box<dyn Error>> {
    if args.days < 1 {
        return Err(format!("curator: --days must be >= 1 (got {})", args.days).into());
    }

    let mut candidates = agent_created_report(context)?
        .into_iter()
        .filter(|row| !row.pinned && row.state != STATE_ARCHIVED)
        .filter_map(|row| {
            let idle = idle_days(&row)?;
            (idle >= args.days).then_some((row.name, idle))
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        println!(
            "curator: nothing to prune (no unpinned skills idle >= {}d)",
            args.days
        );
        return Ok(());
    }

    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    println!(
        "curator: {} skill(s) idle >= {}d:",
        candidates.len(),
        args.days
    );
    for (name, idle) in &candidates {
        println!("  {:40} idle {}d", name, idle);
    }

    if args.dry_run {
        println!("\n(dry run — no changes made)");
        return Ok(());
    }

    if !args.yes && !confirm_prompt(&format!("\nArchive {} skill(s)? [y/N] ", candidates.len()))? {
        println!("curator: aborted");
        return Err("curator: aborted".into());
    }

    let mut archived = 0_usize;
    let mut failures = Vec::new();
    for (name, _) in &candidates {
        let (ok, msg) = archive_skill(context, name)?;
        if ok {
            archived += 1;
        } else {
            failures.push((name.clone(), msg));
        }
    }

    println!("\ncurator: archived {archived}/{}", candidates.len());
    if !failures.is_empty() {
        println!("failures:");
        for (name, msg) in failures {
            println!("  {name}: {msg}");
        }
        return Err("curator prune had failures".into());
    }
    Ok(())
}

fn load_curator_state(context: &HermesContext) -> Result<CuratorState, Box<dyn Error>> {
    let path = context.hermes_home().join("skills").join(".curator_state");
    if !path.exists() {
        return Ok(default_curator_state());
    }
    let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(path)?)?;
    let Some(object) = parsed.as_object() else {
        return Ok(default_curator_state());
    };
    let mut state = default_curator_state();
    state.last_run_at = object
        .get("last_run_at")
        .and_then(JsonValue::as_str)
        .map(str::to_string);
    state.last_run_duration_seconds = object
        .get("last_run_duration_seconds")
        .and_then(JsonValue::as_f64);
    state.last_run_summary = object
        .get("last_run_summary")
        .and_then(JsonValue::as_str)
        .map(str::to_string);
    state.last_report_path = object
        .get("last_report_path")
        .and_then(JsonValue::as_str)
        .map(str::to_string);
    state.paused = object
        .get("paused")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    state.run_count = object
        .get("run_count")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0);
    Ok(state)
}

fn save_curator_state(context: &HermesContext, state: &CuratorState) -> Result<(), Box<dyn Error>> {
    let path = context.hermes_home().join("skills").join(".curator_state");
    let mut object = JsonMap::new();
    object.insert(
        "last_run_at".to_string(),
        state
            .last_run_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "last_run_duration_seconds".to_string(),
        state
            .last_run_duration_seconds
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "last_run_summary".to_string(),
        state
            .last_run_summary
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "last_report_path".to_string(),
        state
            .last_report_path
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert("paused".to_string(), JsonValue::Bool(state.paused));
    object.insert("run_count".to_string(), JsonValue::from(state.run_count));
    atomic_write_json(&path, &JsonValue::Object(object))
}

fn default_curator_state() -> CuratorState {
    CuratorState {
        last_run_at: None,
        last_run_duration_seconds: None,
        last_run_summary: None,
        last_report_path: None,
        paused: false,
        run_count: 0,
    }
}

fn set_paused(context: &HermesContext, paused: bool) -> Result<(), Box<dyn Error>> {
    let mut state = load_curator_state(context)?;
    state.paused = paused;
    save_curator_state(context, &state)
}

fn curator_enabled(context: &HermesContext) -> Result<bool, Box<dyn Error>> {
    let raw = load_raw_config(context)?;
    let Some(root) = raw.as_mapping() else {
        return Ok(true);
    };
    let Some(curator) = root
        .get(&yaml_key("curator"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(true);
    };
    Ok(curator
        .get(&yaml_key("enabled"))
        .and_then(YamlValue::as_bool)
        .unwrap_or(true))
}

fn curator_interval_hours(context: &HermesContext) -> Result<i64, Box<dyn Error>> {
    curator_config_int(context, "interval_hours", DEFAULT_INTERVAL_HOURS)
}

fn curator_stale_after_days(context: &HermesContext) -> Result<i64, Box<dyn Error>> {
    curator_config_int(context, "stale_after_days", DEFAULT_STALE_AFTER_DAYS)
}

fn curator_archive_after_days(context: &HermesContext) -> Result<i64, Box<dyn Error>> {
    curator_config_int(context, "archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS)
}

fn curator_config_int(
    context: &HermesContext,
    key: &str,
    default: i64,
) -> Result<i64, Box<dyn Error>> {
    let raw = load_raw_config(context)?;
    let Some(root) = raw.as_mapping() else {
        return Ok(default);
    };
    let Some(curator) = root
        .get(&yaml_key("curator"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(default);
    };
    Ok(curator
        .get(&yaml_key(key))
        .and_then(yaml_value_to_i64)
        .unwrap_or(default))
}

fn load_raw_config(context: &HermesContext) -> Result<YamlValue, Box<dyn Error>> {
    if !context.config_path().exists() {
        return Ok(YamlValue::Null);
    }
    let text = fs::read_to_string(context.config_path())?;
    if text.trim().is_empty() {
        return Ok(YamlValue::Null);
    }
    Ok(serde_yaml::from_str(&text)?)
}

fn yaml_value_to_i64(value: &YamlValue) -> Option<i64> {
    match value {
        YamlValue::Number(number) => number.as_i64(),
        YamlValue::String(text) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn agent_created_report(context: &HermesContext) -> Result<Vec<UsageRecord>, Box<dyn Error>> {
    let usage = load_usage(context)?;
    let names = list_agent_created_skill_names(context, &usage)?;
    let mut rows = Vec::new();
    for name in names {
        let mut record = usage.get(&name).cloned().unwrap_or_else(empty_usage_record);
        record.name = name;
        rows.push(record);
    }
    Ok(rows)
}

fn list_agent_created_skill_names(
    context: &HermesContext,
    usage: &HashMap<String, UsageRecord>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let base = context.hermes_home().join("skills");
    if !base.exists() {
        return Ok(Vec::new());
    }
    let mut off_limits = bundled_skill_names(context)?;
    off_limits.extend(hub_installed_names(context)?);
    let mut names = HashSet::new();
    for skill_md in skill_markdown_files(&base)? {
        let rel = match skill_md.strip_prefix(&base) {
            Ok(rel) => rel,
            Err(_) => continue,
        };
        if rel
            .components()
            .next()
            .and_then(|component| component.as_os_str().to_str())
            .is_some_and(|part| part.starts_with('.') || part == "node_modules")
        {
            continue;
        }
        let skill_name = read_skill_name(
            &skill_md,
            skill_md
                .parent()
                .and_then(Path::file_name)
                .and_then(|v| v.to_str())
                .unwrap_or("skill"),
        );
        if off_limits.contains(&skill_name) {
            continue;
        }
        let Some(record) = usage.get(&skill_name) else {
            continue;
        };
        if !(record.created_by.as_deref() == Some("agent") || record.agent_created) {
            continue;
        }
        names.insert(skill_name);
    }
    let mut values = names.into_iter().collect::<Vec<_>>();
    values.sort();
    Ok(values)
}

fn is_agent_created(context: &HermesContext, skill_name: &str) -> Result<bool, Box<dyn Error>> {
    let mut off_limits = bundled_skill_names(context)?;
    off_limits.extend(hub_installed_names(context)?);
    Ok(!off_limits.contains(skill_name))
}

fn bundled_skill_names(context: &HermesContext) -> Result<HashSet<String>, Box<dyn Error>> {
    let manifest = context
        .hermes_home()
        .join("skills")
        .join(".bundled_manifest");
    if !manifest.exists() {
        return Ok(HashSet::new());
    }
    let mut names = HashSet::new();
    for line in fs::read_to_string(manifest)?.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let name = trimmed
            .split_once(':')
            .map(|(name, _)| name.trim())
            .unwrap_or(trimmed);
        if !name.is_empty() {
            names.insert(name.to_string());
        }
    }
    Ok(names)
}

fn hub_installed_names(context: &HermesContext) -> Result<HashSet<String>, Box<dyn Error>> {
    let lock_path = context
        .hermes_home()
        .join("skills")
        .join(".hub")
        .join("lock.json");
    if !lock_path.exists() {
        return Ok(HashSet::new());
    }
    let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(lock_path)?)?;
    let Some(installed) = parsed.get("installed").and_then(JsonValue::as_object) else {
        return Ok(HashSet::new());
    };
    let skills_dir = context.hermes_home().join("skills");
    let skills_root_resolved = skills_dir
        .canonicalize()
        .unwrap_or_else(|_| skills_dir.clone());
    let mut names = installed.keys().cloned().collect::<HashSet<_>>();
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
        let mut skill_dir = PathBuf::from(install_path.trim());
        if !skill_dir.is_absolute() {
            skill_dir = context.hermes_home().join("skills").join(skill_dir);
        }
        let resolved = skill_dir.canonicalize().unwrap_or(skill_dir);
        if resolved.strip_prefix(&skills_root_resolved).is_err() {
            continue;
        }
        let skill_md = resolved.join("SKILL.md");
        if skill_md.exists() {
            names.insert(read_skill_name(
                &skill_md,
                resolved
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("skill"),
            ));
        }
    }
    Ok(names)
}

fn skill_markdown_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut output = Vec::new();
    collect_skill_markdown_files(root, &mut output)?;
    Ok(output)
}

fn collect_skill_markdown_files(
    root: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if name.starts_with('.') || name == "node_modules" {
                continue;
            }
            collect_skill_markdown_files(&path, output)?;
        } else if name == "SKILL.md" {
            output.push(path);
        }
    }
    Ok(())
}

fn load_usage(context: &HermesContext) -> Result<HashMap<String, UsageRecord>, Box<dyn Error>> {
    let path = context.hermes_home().join("skills").join(".usage.json");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(path)?)?;
    let Some(object) = parsed.as_object() else {
        return Ok(HashMap::new());
    };
    let mut result = HashMap::new();
    for (name, value) in object {
        let Some(record) = value.as_object() else {
            continue;
        };
        result.insert(name.clone(), usage_record_from_json(name, record));
    }
    Ok(result)
}

fn save_usage(
    context: &HermesContext,
    usage: &HashMap<String, UsageRecord>,
) -> Result<(), Box<dyn Error>> {
    let path = context.hermes_home().join("skills").join(".usage.json");
    let mut root = JsonMap::new();
    let mut names = usage.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let Some(record) = usage.get(&name) else {
            continue;
        };
        root.insert(name, usage_record_to_json(record));
    }
    atomic_write_json(&path, &JsonValue::Object(root))
}

fn usage_record_from_json(name: &str, object: &JsonMap<String, JsonValue>) -> UsageRecord {
    UsageRecord {
        name: name.to_string(),
        state: object
            .get("state")
            .and_then(JsonValue::as_str)
            .filter(|value| VALID_STATES.contains(value))
            .unwrap_or(STATE_ACTIVE)
            .to_string(),
        pinned: object
            .get("pinned")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false),
        created_at: object
            .get("created_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_used_at: object
            .get("last_used_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_viewed_at: object
            .get("last_viewed_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        last_patched_at: object
            .get("last_patched_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        use_count: object
            .get("use_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0),
        view_count: object
            .get("view_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0),
        patch_count: object
            .get("patch_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0),
        created_by: object
            .get("created_by")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        agent_created: object
            .get("agent_created")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false),
        archived_at: object
            .get("archived_at")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
    }
}

fn usage_record_to_json(record: &UsageRecord) -> JsonValue {
    let mut object = JsonMap::new();
    object.insert(
        "created_by".to_string(),
        record
            .created_by
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert("use_count".to_string(), JsonValue::from(record.use_count));
    object.insert("view_count".to_string(), JsonValue::from(record.view_count));
    object.insert(
        "last_used_at".to_string(),
        record
            .last_used_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "last_viewed_at".to_string(),
        record
            .last_viewed_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "patch_count".to_string(),
        JsonValue::from(record.patch_count),
    );
    object.insert(
        "last_patched_at".to_string(),
        record
            .last_patched_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert(
        "created_at".to_string(),
        record
            .created_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    object.insert("state".to_string(), JsonValue::String(record.state.clone()));
    object.insert("pinned".to_string(), JsonValue::Bool(record.pinned));
    object.insert(
        "archived_at".to_string(),
        record
            .archived_at
            .clone()
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    );
    if record.agent_created {
        object.insert("agent_created".to_string(), JsonValue::Bool(true));
    }
    JsonValue::Object(object)
}

fn empty_usage_record() -> UsageRecord {
    UsageRecord {
        name: String::new(),
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

fn get_usage_record(context: &HermesContext, skill: &str) -> Result<UsageRecord, Box<dyn Error>> {
    let usage = load_usage(context)?;
    Ok(usage.get(skill).cloned().unwrap_or_else(empty_usage_record))
}

fn mutate_usage_record<F>(
    context: &HermesContext,
    skill: &str,
    mutator: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut UsageRecord),
{
    if skill.trim().is_empty() {
        return Ok(());
    }
    if !is_agent_created(context, skill)? {
        return Ok(());
    }
    let mut usage = load_usage(context)?;
    let mut record = usage.remove(skill).unwrap_or_else(empty_usage_record);
    record.name = skill.to_string();
    mutator(&mut record);
    usage.insert(skill.to_string(), record);
    save_usage(context, &usage)
}

fn set_skill_state(
    context: &HermesContext,
    skill: &str,
    state: &str,
) -> Result<(), Box<dyn Error>> {
    if !VALID_STATES.contains(&state) {
        return Ok(());
    }
    mutate_usage_record(context, skill, |record| {
        record.state = state.to_string();
        if state == STATE_ARCHIVED {
            record.archived_at = Some(now_iso());
        } else if state == STATE_ACTIVE {
            record.archived_at = None;
        }
    })
}

fn archive_skill(context: &HermesContext, skill: &str) -> Result<(bool, String), Box<dyn Error>> {
    if !is_agent_created(context, skill)? {
        return Ok((
            false,
            format!(
                "skill '{}' is bundled or hub-installed; never archive",
                skill
            ),
        ));
    }
    let Some(skill_dir) = find_skill_dir(context, skill)? else {
        return Ok((false, format!("skill '{}' not found", skill)));
    };
    let archive_root = context.hermes_home().join("skills").join(".archive");
    fs::create_dir_all(&archive_root)?;

    let base_name = skill_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(skill);
    let mut dest = archive_root.join(base_name);
    if dest.exists() {
        dest = archive_root.join(format!("{}-{}", base_name, timestamp_suffix()));
    }
    move_directory(&skill_dir, &dest)?;
    set_skill_state(context, skill, STATE_ARCHIVED)?;
    Ok((true, format!("archived to {}", dest.display())))
}

fn restore_skill(context: &HermesContext, skill: &str) -> Result<(bool, String), Box<dyn Error>> {
    if !is_agent_created(context, skill)? {
        return Ok((
            false,
            format!(
                "skill '{}' is now bundled or hub-installed; restore would shadow the upstream version",
                skill
            ),
        ));
    }
    let archive_root = context.hermes_home().join("skills").join(".archive");
    if !archive_root.exists() {
        return Ok((false, "no archive directory".to_string()));
    }
    let mut candidates = archive_candidates(&archive_root, skill)?;
    if candidates.is_empty() {
        return Ok((false, format!("skill '{}' not found in archive", skill)));
    }
    candidates.sort();
    let src = candidates.remove(0);
    let dest = context.hermes_home().join("skills").join(skill);
    if dest.exists() {
        return Ok((
            false,
            format!("destination already exists: {}", dest.display()),
        ));
    }
    move_directory(&src, &dest)?;
    set_skill_state(context, skill, STATE_ACTIVE)?;
    Ok((true, format!("restored to {}", dest.display())))
}

fn archive_candidates(archive_root: &Path, skill: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut exact = Vec::new();
    let mut prefixed = Vec::new();
    for entry in all_directories(archive_root)? {
        let name = entry
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        if name == skill {
            exact.push(entry);
        } else if name.starts_with(&format!("{skill}-")) {
            prefixed.push(entry);
        }
    }
    if !exact.is_empty() {
        return Ok(exact);
    }
    prefixed.sort_by(|left, right| right.cmp(left));
    Ok(prefixed)
}

fn all_directories(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut output = Vec::new();
    collect_directories(root, &mut output)?;
    Ok(output)
}

fn collect_directories(root: &Path, output: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            output.push(path.clone());
            collect_directories(&path, output)?;
        }
    }
    Ok(())
}

fn find_skill_dir(context: &HermesContext, skill: &str) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let base = context.hermes_home().join("skills");
    if !base.exists() {
        return Ok(None);
    }
    for skill_md in skill_markdown_files(&base)? {
        let rel = match skill_md.strip_prefix(&base) {
            Ok(rel) => rel,
            Err(_) => continue,
        };
        if rel
            .components()
            .next()
            .and_then(|component| component.as_os_str().to_str())
            .is_some_and(|part| part.starts_with('.'))
        {
            continue;
        }
        let fallback = skill_md
            .parent()
            .and_then(Path::file_name)
            .and_then(|value| value.to_str())
            .unwrap_or("skill");
        if read_skill_name(&skill_md, fallback) == skill {
            return Ok(skill_md.parent().map(Path::to_path_buf));
        }
    }
    Ok(None)
}

fn read_skill_name(skill_md: &Path, fallback: &str) -> String {
    let text = match fs::read_to_string(skill_md) {
        Ok(text) => text,
        Err(_) => return fallback.to_string(),
    };
    if !text.starts_with("---") {
        return fallback.to_string();
    }
    let tail = &text[3..];
    let Some(end_offset) = tail.find("\n---\n").or_else(|| tail.find("\n---\r\n")) else {
        return fallback.to_string();
    };
    let yaml_content = &tail[..end_offset];
    let Ok(parsed) = serde_yaml::from_str::<YamlValue>(yaml_content) else {
        return fallback.to_string();
    };
    parsed
        .as_mapping()
        .and_then(|mapping| mapping.get(&yaml_key("name")))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

fn validate_skill_name(raw: &str) -> Result<&str, Box<dyn Error>> {
    let value = raw.trim();
    if value.is_empty() {
        return Err("skill name cannot be empty".into());
    }
    Ok(value)
}

fn idle_days(record: &UsageRecord) -> Option<i64> {
    let dt = record
        .last_activity_at()
        .or_else(|| record.created_at_dt())?;
    Some((Utc::now() - dt).num_days().max(0))
}

impl UsageRecord {
    fn last_activity_at(&self) -> Option<DateTime<Utc>> {
        [
            self.last_used_at.as_deref(),
            self.last_viewed_at.as_deref(),
            self.last_patched_at.as_deref(),
        ]
        .into_iter()
        .flatten()
        .filter_map(parse_iso_timestamp)
        .max()
    }

    fn created_at_dt(&self) -> Option<DateTime<Utc>> {
        self.created_at.as_deref().and_then(parse_iso_timestamp)
    }

    fn activity_count(&self) -> i64 {
        self.use_count
            .saturating_add(self.view_count)
            .saturating_add(self.patch_count)
    }
}

fn parse_iso_timestamp(value: &str) -> Option<DateTime<Utc>> {
    if value.trim().is_empty() {
        return None;
    }
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }
    None
}

fn fmt_ts(value: Option<&str>) -> String {
    let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
        return "never".to_string();
    };
    let Some(dt) = parse_iso_timestamp(value) else {
        return value.to_string();
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

fn format_interval(hours: i64) -> String {
    if hours >= 24 && hours % 24 == 0 {
        format!("{}d", hours / 24)
    } else {
        format!("{hours}h")
    }
}

const CURATOR_DEFAULT_KEEP: usize = 5;
const CURATOR_SKILLS_ARCHIVE: &str = "skills.tar.gz";
const CURATOR_CRON_JOBS_FILE: &str = "cron-jobs.json";
const CURATOR_EXCLUDE_TOP_LEVEL: [&str; 2] = [".curator_backups", ".hub"];

#[derive(Debug, Clone)]
struct CuratorBackupRow {
    id: String,
    reason: String,
    archive_bytes: u64,
    skill_files: i64,
}

#[derive(Debug, Clone, Default)]
struct CronRestoreReport {
    attempted: bool,
    restored: usize,
    skipped_missing: usize,
    unchanged: usize,
    error: Option<String>,
}

fn curator_backup_enabled(context: &HermesContext) -> Result<bool, Box<dyn Error>> {
    let raw = load_raw_config(context)?;
    let Some(root) = raw.as_mapping() else {
        return Ok(true);
    };
    let Some(curator) = root
        .get(&yaml_key("curator"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(true);
    };
    let Some(backup) = curator
        .get(&yaml_key("backup"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(true);
    };
    Ok(backup
        .get(&yaml_key("enabled"))
        .and_then(YamlValue::as_bool)
        .unwrap_or(true))
}

fn curator_backup_keep(context: &HermesContext) -> Result<usize, Box<dyn Error>> {
    let raw = load_raw_config(context)?;
    let Some(root) = raw.as_mapping() else {
        return Ok(CURATOR_DEFAULT_KEEP);
    };
    let Some(curator) = root
        .get(&yaml_key("curator"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(CURATOR_DEFAULT_KEEP);
    };
    let Some(backup) = curator
        .get(&yaml_key("backup"))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(CURATOR_DEFAULT_KEEP);
    };
    let keep = backup
        .get(&yaml_key("keep"))
        .and_then(yaml_value_to_i64)
        .unwrap_or(CURATOR_DEFAULT_KEEP as i64)
        .max(1) as usize;
    Ok(keep)
}

fn curator_backups_dir(context: &HermesContext) -> PathBuf {
    context
        .hermes_home()
        .join("skills")
        .join(".curator_backups")
}

fn curator_skills_dir(context: &HermesContext) -> PathBuf {
    context.hermes_home().join("skills")
}

fn curator_cron_jobs_path(context: &HermesContext) -> PathBuf {
    context.hermes_home().join("cron").join("jobs.json")
}

fn curator_timestamp_id() -> String {
    Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string()
}

fn next_snapshot_id(backups: &Path) -> String {
    let base = curator_timestamp_id();
    let mut candidate = base.clone();
    let mut counter = 1;
    while backups.join(&candidate).exists() {
        candidate = format!("{base}-{counter:02}");
        counter += 1;
    }
    candidate
}

fn snapshot_skills(
    context: &HermesContext,
    reason: &str,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if !curator_backup_enabled(context)? {
        return Ok(None);
    }
    let skills = curator_skills_dir(context);
    if !skills.exists() {
        return Ok(None);
    }
    let backups = curator_backups_dir(context);
    fs::create_dir_all(&backups)?;
    let snapshot_id = next_snapshot_id(&backups);
    let dest = backups.join(snapshot_id);
    fs::create_dir_all(&dest)?;

    let archive_path = dest.join(CURATOR_SKILLS_ARCHIVE);
    let archive_file = File::create(&archive_path)?;
    let encoder = GzEncoder::new(archive_file, Compression::new(6));
    let mut builder = Builder::new(encoder);
    for entry in fs::read_dir(&skills)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if CURATOR_EXCLUDE_TOP_LEVEL.contains(&name_str.as_ref()) {
            continue;
        }
        if path.is_dir() {
            builder.append_dir_all(Path::new(name_str.as_ref()), &path)?;
        } else {
            builder.append_path_with_name(&path, Path::new(name_str.as_ref()))?;
        }
    }
    builder.finish()?;
    let encoder = builder.into_inner()?;
    encoder.finish()?;

    let cron_info = backup_cron_jobs_into(context, &dest)?;
    write_backup_manifest(
        &dest,
        reason,
        &archive_path,
        count_skill_files(&skills)?,
        &cron_info,
    )?;
    prune_old_backups(context, curator_backup_keep(context)?)?;
    Ok(Some(dest))
}

fn count_skill_files(root: &Path) -> Result<i64, Box<dyn Error>> {
    Ok(skill_markdown_files(root)?.len() as i64)
}

fn backup_cron_jobs_into(
    context: &HermesContext,
    dest: &Path,
) -> Result<JsonValue, Box<dyn Error>> {
    let source = curator_cron_jobs_path(context);
    let mut info = JsonMap::new();
    info.insert("backed_up".to_string(), JsonValue::Bool(false));
    info.insert("jobs_count".to_string(), JsonValue::from(0));
    if !source.exists() {
        info.insert(
            "reason".to_string(),
            JsonValue::String("no cron/jobs.json present".to_string()),
        );
        return Ok(JsonValue::Object(info));
    }

    let raw = match fs::read_to_string(&source) {
        Ok(raw) => raw,
        Err(error) => {
            info.insert(
                "reason".to_string(),
                JsonValue::String(format!("read error: {error}")),
            );
            return Ok(JsonValue::Object(info));
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
    fs::write(dest.join(CURATOR_CRON_JOBS_FILE), raw)?;
    info.insert("backed_up".to_string(), JsonValue::Bool(true));
    info.insert("jobs_count".to_string(), JsonValue::from(jobs_count as i64));
    Ok(JsonValue::Object(info))
}

fn write_backup_manifest(
    snapshot_dir: &Path,
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
            snapshot_dir
                .file_name()
                .and_then(|value| value.to_str())
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
    manifest.insert("cron_jobs".to_string(), cron_info.clone());
    atomic_write_json(
        &snapshot_dir.join("manifest.json"),
        &JsonValue::Object(manifest),
    )
}

fn prune_old_backups(context: &HermesContext, keep: usize) -> Result<Vec<String>, Box<dyn Error>> {
    let backups = curator_backups_dir(context);
    if !backups.exists() {
        return Ok(Vec::new());
    }
    let mut regular = Vec::new();
    let mut staging = Vec::new();
    for entry in fs::read_dir(&backups)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(".rollback-staging-") {
            staging.push(path);
            continue;
        }
        if path.join(CURATOR_SKILLS_ARCHIVE).exists() {
            regular.push((name, path));
        }
    }
    regular.sort_by(|left, right| right.0.cmp(&left.0));
    let mut deleted = Vec::new();
    for (_, path) in regular.into_iter().skip(keep) {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string();
        fs::remove_dir_all(&path)?;
        deleted.push(name);
    }
    for path in staging {
        let _ = fs::remove_dir_all(path);
    }
    Ok(deleted)
}

fn list_backups(context: &HermesContext) -> Result<Vec<CuratorBackupRow>, Box<dyn Error>> {
    let backups = curator_backups_dir(context);
    if !backups.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for entry in fs::read_dir(&backups)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !path.join(CURATOR_SKILLS_ARCHIVE).exists() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let manifest = read_backup_manifest(&path);
        rows.push(CuratorBackupRow {
            id: id.clone(),
            reason: manifest
                .get("reason")
                .and_then(JsonValue::as_str)
                .unwrap_or("?")
                .to_string(),
            archive_bytes: manifest
                .get("archive_bytes")
                .and_then(JsonValue::as_u64)
                .unwrap_or_else(|| {
                    path.join(CURATOR_SKILLS_ARCHIVE)
                        .metadata()
                        .map(|m| m.len())
                        .unwrap_or(0)
                }),
            skill_files: manifest
                .get("skill_files")
                .and_then(JsonValue::as_i64)
                .unwrap_or(0),
        });
    }
    rows.sort_by(|left, right| right.id.cmp(&left.id));
    Ok(rows)
}

fn summarize_backups(context: &HermesContext) -> Result<String, Box<dyn Error>> {
    let rows = list_backups(context)?;
    if rows.is_empty() {
        return Ok("No curator snapshots yet.".to_string());
    }
    let mut lines = Vec::new();
    lines.push(format!(
        "{:<24}  {:<40}  {:>6}  {:>8}",
        "id", "reason", "skills", "size"
    ));
    lines.push("─".repeat(lines[0].len()));
    for row in rows {
        lines.push(format!(
            "{:<24}  {:<40}  {:>6}  {:>8}",
            row.id,
            truncate_reason(&row.reason, 40),
            row.skill_files,
            format_size(row.archive_bytes)
        ));
    }
    Ok(lines.join("\n"))
}

fn truncate_reason(value: &str, max_len: usize) -> String {
    value.chars().take(max_len).collect::<String>()
}

fn resolve_backup(
    context: &HermesContext,
    backup_id: Option<&str>,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let backups = curator_backups_dir(context);
    if !backups.exists() {
        return Ok(None);
    }
    if let Some(backup_id) = backup_id {
        let target = backups.join(backup_id);
        return Ok(target
            .is_dir()
            .then_some(target.clone())
            .filter(|path| path.join(CURATOR_SKILLS_ARCHIVE).exists()));
    }
    Ok(list_backups(context)?
        .first()
        .map(|row| backups.join(&row.id))
        .filter(|path| path.join(CURATOR_SKILLS_ARCHIVE).exists()))
}

fn read_backup_manifest(snapshot_dir: &Path) -> JsonValue {
    fs::read_to_string(snapshot_dir.join("manifest.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<JsonValue>(&raw).ok())
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()))
}

fn rollback_skills(
    context: &HermesContext,
    backup_id: Option<&str>,
) -> Result<(bool, String, Option<PathBuf>), Box<dyn Error>> {
    let Some(target) = resolve_backup(context, backup_id)? else {
        return Ok((false, "no matching backup found".to_string(), None));
    };
    let archive = target.join(CURATOR_SKILLS_ARCHIVE);
    if !archive.exists() {
        return Ok((
            false,
            format!(
                "snapshot {} has no {}",
                target
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("?"),
                CURATOR_SKILLS_ARCHIVE
            ),
            None,
        ));
    }

    let skills = curator_skills_dir(context);
    let backups = curator_backups_dir(context);
    fs::create_dir_all(&skills)?;
    fs::create_dir_all(&backups)?;

    let _ = snapshot_skills(
        context,
        &format!(
            "pre-rollback to {}",
            target
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("?")
        ),
    )?;

    let staged = backups.join(format!(".rollback-staging-{}", curator_timestamp_id()));
    fs::create_dir_all(&staged)?;
    let mut moved = Vec::new();
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
    let cron_report = restore_cron_skill_links(context, &target)?;
    let mut summary = format!(
        "restored from snapshot {}",
        target
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("?")
    );
    if cron_report.attempted {
        if let Some(error) = cron_report.error {
            summary.push_str(&format!("; cron links: error — {error}"));
        } else {
            let mut parts = Vec::new();
            if cron_report.restored > 0 {
                parts.push(format!(
                    "{} job(s) had skill links restored",
                    cron_report.restored
                ));
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
    let file = File::open(archive_path)?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    for entry_result in archive.entries()? {
        let mut entry = entry_result?;
        let path = entry.path()?.into_owned();
        if path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(format!("refusing to extract unsafe path: {:?}", path).into());
        }
        entry.unpack_in(skills_dir)?;
    }
    Ok(())
}

fn restore_cron_skill_links(
    context: &HermesContext,
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
    let live_path = curator_cron_jobs_path(context);
    if !live_path.exists() {
        return Ok(CronRestoreReport {
            attempted: true,
            ..Default::default()
        });
    }
    let mut live = serde_json::from_str::<JsonValue>(&fs::read_to_string(&live_path)?)?;
    let Some(live_jobs) = json_jobs_list_mut(&mut live) else {
        return Ok(CronRestoreReport {
            attempted: true,
            error: Some("live cron/jobs.json has no jobs list".to_string()),
            ..Default::default()
        });
    };

    let mut backup_by_id = HashMap::new();
    for job in backup_jobs {
        let Some(id) = job.get("id").and_then(JsonValue::as_str) else {
            continue;
        };
        backup_by_id.insert(id.to_string(), job.clone());
    }

    let mut report = CronRestoreReport {
        attempted: true,
        ..Default::default()
    };
    let mut live_ids = HashSet::new();
    let mut changed = false;
    for live_job in live_jobs.iter_mut() {
        let Some(id) = live_job.get("id").and_then(JsonValue::as_str) else {
            continue;
        };
        live_ids.insert(id.to_string());
        let Some(backup_job) = backup_by_id.get(id) else {
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

fn atomic_write_json(path: &Path, value: &JsonValue) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&tmp, format!("{}\n", serde_json::to_string_pretty(value)?))?;
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

fn now_iso() -> String {
    Utc::now().to_rfc3339()
}

fn timestamp_suffix() -> String {
    Utc::now().format("%Y%m%d%H%M%S").to_string()
}

fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn yaml_key(key: &str) -> YamlValue {
    YamlValue::String(key.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(test)]
    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    #[cfg(test)]
    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-curator-{label}-{unique}"))
    }

    #[test]
    fn pause_and_resume_update_state() {
        let home = temp_path("pause");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_paused(&context, true).unwrap();
        assert!(load_curator_state(&context).unwrap().paused);
        set_paused(&context, false).unwrap();
        assert!(!load_curator_state(&context).unwrap().paused);
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn pin_and_unpin_update_usage_record() {
        let home = temp_path("pin");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join("demo")).unwrap();
        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nBody\n",
        )
        .unwrap();
        mutate_usage_record(&context, "demo", |record| {
            record.created_by = Some("agent".to_string());
        })
        .unwrap();
        set_pinned_command(&context, "demo", true).unwrap();
        assert!(get_usage_record(&context, "demo").unwrap().pinned);
        set_pinned_command(&context, "demo", false).unwrap();
        assert!(!get_usage_record(&context, "demo").unwrap().pinned);
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn archive_and_restore_round_trip() {
        let home = temp_path("archive");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join("demo")).unwrap();
        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nBody\n",
        )
        .unwrap();
        mutate_usage_record(&context, "demo", |record| {
            record.created_by = Some("agent".to_string());
        })
        .unwrap();

        let archived = archive_skill(&context, "demo").unwrap();
        assert!(archived.0);
        assert!(!home.join("skills").join("demo").exists());
        assert!(home.join("skills").join(".archive").join("demo").exists());
        assert_eq!(
            get_usage_record(&context, "demo").unwrap().state,
            STATE_ARCHIVED
        );

        let restored = restore_skill(&context, "demo").unwrap();
        assert!(restored.0);
        assert!(home.join("skills").join("demo").exists());
        assert_eq!(
            get_usage_record(&context, "demo").unwrap().state,
            STATE_ACTIVE
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn prune_dry_run_keeps_skill_in_place() {
        let home = temp_path("prune");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join("demo")).unwrap();
        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nBody\n",
        )
        .unwrap();
        let mut usage = HashMap::new();
        let mut record = empty_usage_record();
        record.name = "demo".to_string();
        record.created_by = Some("agent".to_string());
        record.created_at = Some("2024-01-01T00:00:00+00:00".to_string());
        usage.insert("demo".to_string(), record);
        save_usage(&context, &usage).unwrap();

        prune_command(
            &context,
            PruneArgs {
                days: 90,
                yes: false,
                dry_run: true,
            },
        )
        .unwrap();
        assert!(home.join("skills").join("demo").exists());
        assert!(!home.join("skills").join(".archive").join("demo").exists());
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn curator_run_sync_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("curator-run");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let fake_python = home.join("python3");
        let log = home.join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'run dry=%s\\n' \"$HERMES_CURATOR_RUN_DRY\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_CURATOR_PYTHON", &fake_python);
        print_curator_run(
            &context,
            RunArgs {
                synchronous: true,
                dry_run: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("run dry=1"));

        remove_env_var("HERMES_CURATOR_PYTHON");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn curator_run_async_spawns_detached_sync_child() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("curator-run-async");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let fake_binary = home.join("hermes");
        let log = home.join("curator.log");
        fs::write(
            &fake_binary,
            format!(
                "#!/bin/sh\nprintf 'argv=%s\\n' \"$*\" >> '{}'\nprintf 'home=%s\\n' \"$HERMES_HOME\" >> '{}'\n",
                log.display(),
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_binary).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_binary, perms).unwrap();

        set_env_var("HERMES_CURATOR_BINARY", &fake_binary);
        print_curator_run(
            &context,
            RunArgs {
                synchronous: false,
                dry_run: true,
            },
        )
        .unwrap();

        for _ in 0..20 {
            if log.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("argv=curator run --sync --dry-run"));
        assert!(output.contains(&format!("home={}", home.display())));

        remove_env_var("HERMES_CURATOR_BINARY");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn curator_backup_creates_snapshot_and_manifest() {
        let home = temp_path("curator-backup");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join("demo")).unwrap();
        fs::create_dir_all(home.join("cron")).unwrap();
        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nBody\n",
        )
        .unwrap();
        fs::write(
            home.join("cron").join("jobs.json"),
            r#"{"jobs":[{"id":"job-1","skills":["demo"]}],"updated_at":"now"}"#,
        )
        .unwrap();

        let snapshot = snapshot_skills(&context, "manual-snapshot")
            .unwrap()
            .unwrap();
        assert!(snapshot.join(CURATOR_SKILLS_ARCHIVE).exists());
        assert!(snapshot.join(CURATOR_CRON_JOBS_FILE).exists());
        let manifest = read_backup_manifest(&snapshot);
        assert_eq!(
            manifest["reason"],
            JsonValue::String("manual-snapshot".to_string())
        );
        assert_eq!(manifest["skill_files"], JsonValue::from(1));
        assert_eq!(manifest["cron_jobs"]["backed_up"], JsonValue::Bool(true));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn curator_rollback_restores_skills_and_cron_links() {
        let home = temp_path("curator-rollback");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join("demo")).unwrap();
        fs::create_dir_all(home.join("cron")).unwrap();
        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nOriginal\n",
        )
        .unwrap();
        fs::write(
            home.join("cron").join("jobs.json"),
            r#"{"jobs":[{"id":"job-1","skills":["demo"],"schedule":"daily"},{"id":"job-2","skills":["extra"]}],"updated_at":"now"}"#,
        )
        .unwrap();
        let snapshot = snapshot_skills(&context, "baseline").unwrap().unwrap();

        fs::write(
            home.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nChanged\n",
        )
        .unwrap();
        fs::write(
            home.join("cron").join("jobs.json"),
            r#"{"jobs":[{"id":"job-1","skills":["merged"],"schedule":"daily"},{"id":"job-3","skills":["new"]}],"updated_at":"later"}"#,
        )
        .unwrap();

        let snapshot_id = snapshot
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap()
            .to_string();
        let result = rollback_skills(&context, Some(&snapshot_id)).unwrap();
        assert!(result.0);
        let restored_skill =
            fs::read_to_string(home.join("skills").join("demo").join("SKILL.md")).unwrap();
        assert!(restored_skill.contains("Original"));

        let cron: JsonValue =
            serde_json::from_str(&fs::read_to_string(home.join("cron").join("jobs.json")).unwrap())
                .unwrap();
        let jobs = cron["jobs"].as_array().unwrap();
        let job_one = jobs
            .iter()
            .find(|job| job["id"] == JsonValue::String("job-1".to_string()))
            .unwrap();
        assert_eq!(
            job_one["skills"],
            JsonValue::Array(vec![JsonValue::String("demo".to_string())])
        );
        assert_eq!(job_one["schedule"], JsonValue::String("daily".to_string()));
        let job_three = jobs
            .iter()
            .find(|job| job["id"] == JsonValue::String("job-3".to_string()))
            .unwrap();
        assert_eq!(
            job_three["skills"],
            JsonValue::Array(vec![JsonValue::String("new".to_string())])
        );
        let backups = list_backups(&context).unwrap();
        assert!(
            backups
                .iter()
                .any(|row| row.reason.starts_with("pre-rollback to"))
        );
        let _ = fs::remove_dir_all(home);
    }
}
