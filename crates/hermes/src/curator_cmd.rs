use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDateTime, Utc};
use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value as YamlValue;

use crate::python_bridge::launch_python_main_command;

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
        Some(CuratorCommand::Run(args)) => bridge_run(args),
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
        Some(CuratorCommand::Backup(args)) => bridge_backup(args),
        Some(CuratorCommand::Rollback(args)) => bridge_rollback(args),
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

fn bridge_run(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let mut argv = vec![String::from("run")];
    if args.synchronous {
        argv.push(String::from("--sync"));
    }
    if args.dry_run {
        argv.push(String::from("--dry-run"));
    }
    bridge_curator(&argv)
}

fn bridge_backup(args: BackupArgs) -> Result<(), Box<dyn Error>> {
    let mut argv = vec![String::from("backup")];
    if let Some(reason) = args
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        argv.push(String::from("--reason"));
        argv.push(reason.to_string());
    }
    bridge_curator(&argv)
}

fn bridge_rollback(args: RollbackArgs) -> Result<(), Box<dyn Error>> {
    let mut argv = vec![String::from("rollback")];
    if args.list {
        argv.push(String::from("--list"));
    }
    if let Some(backup_id) = args
        .backup_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        argv.push(String::from("--id"));
        argv.push(backup_id.to_string());
    }
    if args.yes {
        argv.push(String::from("--yes"));
    }
    bridge_curator(&argv)
}

fn bridge_curator(argv: &[String]) -> Result<(), Box<dyn Error>> {
    launch_python_main_command("curator", argv, Some("HERMES_CURATOR_PYTHON"), &[])
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
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn bridge_run_args_preserve_flags() {
        let args = RunArgs {
            synchronous: true,
            dry_run: true,
        };
        let mut argv = vec![String::from("run")];
        if args.synchronous {
            argv.push(String::from("--sync"));
        }
        if args.dry_run {
            argv.push(String::from("--dry-run"));
        }
        assert_eq!(argv, vec!["run", "--sync", "--dry-run"]);
    }
}
