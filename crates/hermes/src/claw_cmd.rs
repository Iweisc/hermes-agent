use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::HermesContext;
use serde_json::Value as JsonValue;

use crate::backup::create_pre_migration_backup;

mod native_migration;
use native_migration::{MigrationItem, MigrationReport, run_native_apply, run_native_preview};

const OPENCLAW_DIR_NAMES: [&str; 3] = [".openclaw", ".clawdbot", ".moltbot"];
const OPENCLAW_CONFIG_FILE_NAMES: [&str; 3] = ["openclaw.json", "clawdbot.json", "moltbot.json"];

#[derive(Subcommand, Debug)]
pub enum ClawCommand {
    Migrate(MigrateArgs),
    #[command(alias = "clean")]
    Cleanup(CleanupArgs),
}

#[derive(Args, Debug, Clone)]
pub struct MigrateArgs {
    #[arg(long)]
    pub source: Option<PathBuf>,
    #[arg(long = "dry-run")]
    pub dry_run: bool,
    #[arg(long, value_enum, default_value_t = MigratePreset::Full)]
    pub preset: MigratePreset,
    #[arg(long)]
    pub overwrite: bool,
    #[arg(long = "migrate-secrets")]
    pub migrate_secrets: bool,
    #[arg(long = "no-backup")]
    pub no_backup: bool,
    #[arg(long = "workspace-target")]
    pub workspace_target: Option<PathBuf>,
    #[arg(long = "skill-conflict", value_enum, default_value_t = SkillConflict::Skip)]
    pub skill_conflict: SkillConflict,
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct CleanupArgs {
    #[arg(long)]
    pub source: Option<PathBuf>,
    #[arg(long = "dry-run")]
    pub dry_run: bool,
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum MigratePreset {
    #[value(name = "user-data")]
    UserData,
    #[value(name = "full")]
    Full,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum SkillConflict {
    #[value(name = "skip")]
    Skip,
    #[value(name = "overwrite")]
    Overwrite,
    #[value(name = "rename")]
    Rename,
}

pub fn print_claw(
    context: &HermesContext,
    command: Option<ClawCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        Some(ClawCommand::Migrate(args)) => print_migrate(context, args),
        Some(ClawCommand::Cleanup(args)) => print_cleanup(args),
        None => {
            println!("Usage: hermes claw <command> [options]");
            println!();
            println!("Commands:");
            println!("  migrate          Migrate settings from OpenClaw to Hermes");
            println!("  cleanup          Archive leftover OpenClaw directories after migration");
            println!();
            println!("Run `hermes help claw` for options.");
            Ok(())
        }
    }
}

fn print_migrate(context: &HermesContext, args: MigrateArgs) -> Result<(), Box<dyn Error>> {
    let source_dir = resolve_migrate_source(&args)?;
    if let Some(path) = args.workspace_target.as_ref() {
        if !path.is_absolute() {
            return Err("workspace-target must be an absolute path".into());
        }
    }

    if !source_dir.is_dir() {
        print_migration_banner();
        println!();
        println!("OpenClaw directory not found: {}", source_dir.display());
        println!("Make sure your OpenClaw installation is at the expected path.");
        println!("You can specify a custom path: hermes claw migrate --source /path/to/.openclaw");
        return Ok(());
    }

    if source_dir_is_empty(&source_dir) || source_dir_contains_only_empty_configs(&source_dir) {
        print_migration_banner();
        println!();
        println!("Nothing to migrate from OpenClaw.");
        return Ok(());
    }

    print_migration_banner();
    println!();
    println!("Migration Settings");
    println!("  Source:      {}", source_dir.display());
    println!("  Target:      {}", context.hermes_home().display());
    println!("  Preset:      {}", args.preset.as_str());
    println!(
        "  Overwrite:   {}",
        if args.overwrite {
            "yes"
        } else {
            "no (skip conflicts)"
        }
    );
    println!(
        "  Secrets:     {}",
        if args.migrate_secrets {
            "yes (allowlisted only)"
        } else {
            "no"
        }
    );
    if args.skill_conflict != SkillConflict::Skip {
        println!("  Skill conflicts: {}", args.skill_conflict.as_str());
    }
    if let Some(path) = args.workspace_target.as_ref() {
        println!("  Workspace:   {}", path.display());
    }

    warn_if_openclaw_running(args.yes)?;
    warn_if_gateway_running(context, args.yes)?;

    let preview_report = run_native_preview(context, source_dir.clone(), &args)?;
    let preview_count = preview_report.summary.get("migrated").copied().unwrap_or(0);
    let preview_conflicts = preview_report.summary.get("conflict").copied().unwrap_or(0);

    if preview_count == 0 && preview_conflicts == 0 {
        println!();
        println!("Nothing to migrate from OpenClaw.");
        print_migration_report(&preview_report, true);
        return Ok(());
    }

    println!();
    if preview_count > 0 {
        println!("Migration Preview — {preview_count} item(s) would be imported");
    } else {
        println!("Migration Preview — {preview_conflicts} conflict(s), nothing would be imported");
    }
    println!("No changes have been made yet. Review the list below:");
    print_migration_report(&preview_report, true);

    if args.dry_run {
        return Ok(());
    }

    if preview_conflicts > 0 && !args.overwrite {
        println!();
        println!("Plan has {preview_conflicts} conflict(s). Refusing to apply.");
        println!("Each conflict is an item whose target already exists in ~/.hermes/.");
        println!("Re-run with --overwrite to replace conflicting targets.");
        println!("Or re-run with --dry-run to review the full plan.");
        return Ok(());
    }

    println!();
    if !args.yes {
        if !io::stdin().is_terminal() {
            println!("Non-interactive session — preview only.");
            println!("To execute, re-run with: hermes claw migrate --yes");
            return Ok(());
        }
        if !confirm_prompt("Proceed with migration? [Y/n] ")? {
            println!("Migration cancelled.");
            return Ok(());
        }
    }

    let mut backup_archive = None;
    if !args.no_backup {
        match create_pre_migration_backup(context, 5) {
            Ok(path) => {
                backup_archive = path;
                if let Some(path) = backup_archive.as_ref() {
                    println!();
                    println!("Pre-migration backup: {}", path.display());
                    println!("Restore with: hermes import {}", path.display());
                }
            }
            Err(error) => {
                println!();
                println!("Could not create pre-migration backup: {error}");
                println!("Re-run with --no-backup to skip, or free up disk space.");
                return Ok(());
            }
        }
    }

    let report = match run_native_apply(context, source_dir, &args) {
        Ok(report) => report,
        Err(error) => {
            println!();
            println!("Migration failed: {error}");
            if let Some(path) = backup_archive.as_ref() {
                println!("A pre-migration backup is available at: {}", path.display());
                println!("Restore with: hermes import {}", path.display());
            }
            return Ok(());
        }
    };

    print_migration_report(&report, false);
    Ok(())
}

fn print_migration_banner() {
    println!();
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│          ⚕ Hermes — OpenClaw Migration                 │");
    println!("└─────────────────────────────────────────────────────────┘");
}

fn print_cleanup(args: CleanupArgs) -> Result<(), Box<dyn Error>> {
    println!();
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│          Hermes — OpenClaw Cleanup                     │");
    println!("└─────────────────────────────────────────────────────────┘");

    let dirs_to_check = if let Some(source) = args.source.as_ref() {
        validate_source(source)?;
        vec![source.clone()]
    } else {
        find_openclaw_dirs()
    };

    if dirs_to_check.is_empty() {
        println!();
        println!("No OpenClaw directories found. Nothing to clean up.");
        return Ok(());
    }

    let running = detect_openclaw_processes();
    if !running.is_empty() {
        println!();
        println!("OpenClaw appears to be still running:");
        for detail in &running {
            println!("  * {detail}");
        }
        println!("Archiving .openclaw/ while the service is active may cause it to immediately");
        println!("recreate an empty skeleton directory, destroying your config.");
        println!("Stop OpenClaw first: systemctl --user stop openclaw-gateway.service");
        println!();
        if !args.yes {
            if !io::stdin().is_terminal() {
                println!("Non-interactive session — aborting. Stop OpenClaw and re-run.");
                return Ok(());
            }
            if !confirm_prompt("Proceed anyway? [y/N] ")? {
                println!("Aborted. Stop OpenClaw first, then re-run: hermes claw cleanup");
                return Ok(());
            }
        }
    }

    let mut total_archived = 0usize;
    for source_dir in dirs_to_check {
        println!();
        println!("Found: {}", source_dir.display());
        let state_files = scan_workspace_state(&source_dir);
        let workspace_dirs = find_workspace_dirs(&source_dir);

        if !workspace_dirs.is_empty() {
            println!("Workspace directories: {}", workspace_dirs.len());
            for workspace in workspace_dirs.iter().take(5) {
                let detail = workspace_detail(workspace);
                println!(
                    "      {}/  ({detail})",
                    workspace.file_name().unwrap().to_string_lossy()
                );
            }
            if workspace_dirs.len() > 5 {
                println!("      ... and {} more", workspace_dirs.len() - 5);
            }
        }

        if !state_files.is_empty() {
            println!();
            println!("  {} state file(s) found:", state_files.len());
            for (_, desc) in state_files.iter().take(8) {
                println!("      {desc}");
            }
            if state_files.len() > 8 {
                println!("      ... and {} more", state_files.len() - 8);
            }
        }

        println!();
        let archive_path = archive_path_for(&source_dir);
        if args.dry_run {
            println!(
                "Would archive: {} → {}",
                source_dir.display(),
                archive_path.display()
            );
            continue;
        }
        if !args.yes && !io::stdin().is_terminal() {
            println!(
                "Non-interactive session — would archive: {}",
                source_dir.display()
            );
            println!("To execute, re-run with: hermes claw cleanup --yes");
            continue;
        }

        if args.yes || confirm_prompt(&format!("Archive {}? [Y/n] ", source_dir.display()))? {
            std::fs::rename(&source_dir, &archive_path)?;
            println!(
                "Archived: {} → {}",
                source_dir.display(),
                archive_path.display()
            );
            total_archived += 1;
        } else {
            println!("Skipped.");
        }
    }

    println!();
    if args.dry_run {
        println!(
            "Dry run complete. {} directory(ies) would be archived.",
            find_openclaw_dirs_count(args.source.as_ref())
        );
        println!("Run without --dry-run to archive them.");
    } else if total_archived > 0 {
        println!("Cleaned up {total_archived} OpenClaw directory(ies).");
        println!("Directories were renamed, not deleted. You can undo by renaming them back.");
    } else {
        println!("No directories were archived.");
    }
    Ok(())
}

fn validate_source(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.as_os_str().is_empty() {
        return Err("source path cannot be empty".into());
    }
    if !path.exists() {
        return Err(format!("source path not found: {}", path.display()).into());
    }
    if !path.is_dir() {
        return Err(format!("source path is not a directory: {}", path.display()).into());
    }
    Ok(())
}

fn resolve_migrate_source(args: &MigrateArgs) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = args.source.as_ref() {
        validate_source(path)?;
        return Ok(path.clone());
    }

    let Some(home) = dirs::home_dir() else {
        return Ok(PathBuf::from(".openclaw"));
    };
    let default = home.join(".openclaw");
    if default.is_dir() {
        return Ok(default);
    }
    for name in [".clawdbot", ".moltbot"] {
        let candidate = home.join(name);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Ok(default)
}

fn source_dir_is_empty(source_dir: &Path) -> bool {
    std::fs::read_dir(source_dir)
        .map(|mut entries| entries.next().is_none())
        .unwrap_or(false)
}

fn source_dir_contains_only_empty_configs(source_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(source_dir) else {
        return false;
    };

    let mut saw_config = false;
    for entry in entries {
        let Ok(entry) = entry else {
            return false;
        };
        let Ok(file_type) = entry.file_type() else {
            return false;
        };
        if !file_type.is_file() {
            return false;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return false;
        };
        if !OPENCLAW_CONFIG_FILE_NAMES.contains(&name) {
            return false;
        }
        if !config_file_is_empty_object(&entry.path()) {
            return false;
        }
        saw_config = true;
    }
    saw_config
}

fn config_file_is_empty_object(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    serde_json::from_str::<JsonValue>(trimmed)
        .ok()
        .and_then(|value| value.as_object().map(|object| object.is_empty()))
        .unwrap_or(false)
}

fn find_openclaw_dirs() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    OPENCLAW_DIR_NAMES
        .iter()
        .map(|name| home.join(name))
        .filter(|path| path.is_dir())
        .collect()
}

fn find_openclaw_dirs_count(source: Option<&PathBuf>) -> usize {
    source
        .map(|_| 1)
        .unwrap_or_else(|| find_openclaw_dirs().len())
}

fn scan_workspace_state(source_dir: &Path) -> Vec<(PathBuf, String)> {
    let mut findings = Vec::new();
    if !source_dir.exists() {
        return findings;
    }

    for name in ["todo.json", "sessions", "logs"] {
        let candidate = source_dir.join(name);
        if candidate.exists() {
            let kind = if candidate.is_dir() {
                "directory"
            } else {
                "file"
            };
            findings.push((candidate, format!("Root {kind}: {name}")));
        }
    }

    let Ok(children) = std::fs::read_dir(source_dir) else {
        return findings;
    };
    for child in children.flatten() {
        let path = child.path();
        let name = child.file_name();
        if !path.is_dir() || name.to_string_lossy().starts_with('.') {
            continue;
        }
        for state_name in ["todo.json", "sessions", "logs", "memory"] {
            let state_path = path.join(state_name);
            if state_path.exists() {
                let kind = if state_path.is_dir() {
                    "directory"
                } else {
                    "file"
                };
                let rel = state_path
                    .strip_prefix(source_dir)
                    .unwrap_or(&state_path)
                    .display()
                    .to_string();
                findings.push((state_path, format!("Workspace {kind}: {rel}")));
            }
        }
    }

    findings
}

fn find_workspace_dirs(source_dir: &Path) -> Vec<PathBuf> {
    let Ok(children) = std::fs::read_dir(source_dir) else {
        return Vec::new();
    };
    children
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            path.file_name()
                .map(|name| !name.to_string_lossy().starts_with('.'))
                .unwrap_or(false)
        })
        .filter(|path| {
            ["todo.json", "SOUL.md", "MEMORY.md", "USER.md"]
                .iter()
                .any(|name| path.join(name).exists())
        })
        .collect()
}

fn workspace_detail(workspace: &Path) -> String {
    let mut items = Vec::new();
    if workspace.join("todo.json").exists() {
        items.push("todo.json");
    }
    if workspace.join("sessions").is_dir() {
        items.push("sessions/");
    }
    if workspace.join("SOUL.md").exists() {
        items.push("SOUL.md");
    }
    if workspace.join("MEMORY.md").exists() {
        items.push("MEMORY.md");
    }
    if items.is_empty() {
        String::from("empty")
    } else {
        items.join(", ")
    }
}

fn archive_path_for(source_dir: &Path) -> PathBuf {
    let parent = source_dir.parent().unwrap_or_else(|| Path::new("."));
    let base_name = source_dir
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from("openclaw"));
    let mut candidate = parent.join(format!("{base_name}.pre-migration"));
    if !candidate.exists() {
        return candidate;
    }

    let timestamp = chrono::Local::now().format("%Y%m%d").to_string();
    candidate = parent.join(format!("{base_name}.pre-migration-{timestamp}"));
    if !candidate.exists() {
        return candidate;
    }

    let mut counter = 2usize;
    loop {
        let numbered = parent.join(format!("{base_name}.pre-migration-{timestamp}-{counter}"));
        if !numbered.exists() {
            return numbered;
        }
        counter += 1;
    }
}

fn detect_openclaw_processes() -> Vec<String> {
    let mut found = Vec::new();

    #[cfg(not(windows))]
    {
        if let Ok(output) = Command::new("systemctl")
            .args(["--user", "is-active", "openclaw-gateway.service"])
            .output()
            && String::from_utf8_lossy(&output.stdout).trim() == "active"
        {
            found.push(String::from("systemd service: openclaw-gateway.service"));
        }

        if let Ok(output) = Command::new("pgrep").args(["-fa", "openclaw"]).output()
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let lines = stdout
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter(|line| {
                    !line.contains("pgrep -fa openclaw")
                        && !line.contains("hermes claw cleanup")
                        && !line.contains("cargo run")
                })
                .collect::<Vec<_>>();
            let pids = lines
                .iter()
                .filter_map(|line| line.split_whitespace().next())
                .collect::<Vec<_>>();
            if !pids.is_empty() {
                found.push(format!("openclaw process(es) (PIDs: {})", pids.join(", ")));
            }
        }
    }

    #[cfg(windows)]
    {
        for exe in ["openclaw.exe", "clawd.exe"] {
            if let Ok(output) = Command::new("tasklist")
                .args(["/FI", &format!("IMAGENAME eq {exe}")])
                .output()
            {
                if String::from_utf8_lossy(&output.stdout)
                    .to_ascii_lowercase()
                    .contains(&exe.to_ascii_lowercase())
                {
                    found.push(format!("process: {exe}"));
                }
            }
        }
    }

    found
}

fn warn_if_openclaw_running(auto_yes: bool) -> Result<(), Box<dyn Error>> {
    let running = detect_openclaw_processes();
    if running.is_empty() {
        return Ok(());
    }

    println!();
    println!("OpenClaw appears to be running:");
    for detail in &running {
        println!("  * {detail}");
    }
    println!("Messaging platforms only allow one active session per bot token. If you continue,");
    println!("both OpenClaw and Hermes may try to use the same token, causing disconnects.");
    println!("Recommendation: stop OpenClaw before migrating.");
    println!();
    if auto_yes || io::stdin().is_terminal() {
        return Ok(());
    }
    println!("Non-interactive session — continuing to preview only.");
    Ok(())
}

fn warn_if_gateway_running(context: &HermesContext, auto_yes: bool) -> Result<(), Box<dyn Error>> {
    let Some(connected) = connected_gateway_platforms(context)? else {
        return Ok(());
    };
    if connected.is_empty() {
        return Ok(());
    }

    println!();
    println!(
        "Hermes gateway is running with active connections: {}",
        connected.join(", ")
    );
    println!("Migrating bot tokens while the gateway is active will cause conflicts.");
    println!("Recommendation: stop the gateway first with `hermes stop`.");
    println!();
    if auto_yes || io::stdin().is_terminal() {
        return Ok(());
    }
    println!("Non-interactive session — continuing to preview only.");
    Ok(())
}

fn connected_gateway_platforms(
    context: &HermesContext,
) -> Result<Option<Vec<String>>, Box<dyn Error>> {
    let pid_path = context.hermes_home().join("gateway.pid");
    let Some(pid) = read_gateway_pid(&pid_path) else {
        return Ok(None);
    };
    if !process_running(pid) {
        return Ok(None);
    }

    let state_path = context.hermes_home().join("gateway_state.json");
    if !state_path.exists() {
        return Ok(Some(Vec::new()));
    }
    let raw = std::fs::read_to_string(state_path)?;
    let state: JsonValue = serde_json::from_str(&raw)?;
    let connected = state
        .get("platforms")
        .and_then(JsonValue::as_object)
        .map(|platforms| {
            platforms
                .iter()
                .filter_map(|(name, info)| {
                    (info.get("state").and_then(JsonValue::as_str) == Some("connected"))
                        .then(|| name.clone())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(Some(connected))
}

fn read_gateway_pid(path: &Path) -> Option<i64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('{') {
        let value = serde_json::from_str::<JsonValue>(trimmed).ok()?;
        return value.get("pid").and_then(JsonValue::as_i64);
    }
    trimmed.parse::<i64>().ok()
}

fn process_running(pid: i64) -> bool {
    #[cfg(unix)]
    {
        let Ok(pid) = i32::try_from(pid) else {
            return false;
        };
        let status = unsafe { libc::kill(pid, 0) };
        status == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        let _ = pid;
        true
    }
}

fn print_migration_report(report: &MigrationReport, dry_run: bool) {
    let migrated = report.summary.get("migrated").copied().unwrap_or(0);
    let skipped = report.summary.get("skipped").copied().unwrap_or(0);
    let conflicts = report.summary.get("conflict").copied().unwrap_or(0);
    let errors = report.summary.get("error").copied().unwrap_or(0);

    println!();
    if dry_run {
        println!("Dry Run Results");
        println!("No files were modified. This is a preview of what would happen.");
    } else {
        println!("Migration Results");
    }
    println!("Preset: {}", report.preset);
    println!();

    print_items(
        "migrated",
        if dry_run { "Would migrate" } else { "Migrated" },
        &report.items,
    );
    print_items(
        "conflict",
        "Conflicts (skipped — use --overwrite to force)",
        &report.items,
    );
    print_items("skipped", "Skipped", &report.items);
    print_items("error", "Errors", &report.items);

    let mut parts = Vec::new();
    if migrated > 0 {
        parts.push(format!(
            "{migrated} {}",
            if dry_run { "would migrate" } else { "migrated" }
        ));
    }
    if conflicts > 0 {
        parts.push(format!("{conflicts} conflict(s)"));
    }
    if skipped > 0 {
        parts.push(format!("{skipped} skipped"));
    }
    if errors > 0 {
        parts.push(format!("{errors} error(s)"));
    }
    if parts.is_empty() {
        println!("Summary: Nothing to migrate.");
    } else {
        println!("Summary: {}", parts.join(", "));
    }

    if let Some(output_dir) = report.output_dir.as_ref() {
        println!("Full report saved to: {}", output_dir.display());
    }
    for warning in &report.warnings {
        println!("Warning: {warning}");
    }
    for step in &report.next_steps {
        println!("Next: {step}");
    }
}

fn print_items(status: &str, label: &str, items: &[MigrationItem]) {
    let matching = items
        .iter()
        .filter(|item| item.status == status)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return;
    }
    println!("  {label}:");
    for item in matching {
        let kind = item.kind.as_str();
        let tail = if matches!(status, "migrated" | "conflict") {
            item.destination.as_deref().unwrap_or_default()
        } else {
            item.reason.as_str()
        };
        if tail.is_empty() {
            println!("      {kind}");
        } else {
            println!("      {kind:<22} {tail}");
        }
    }
    println!();
}

fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(false);
    }
    let answer = line.trim();
    if answer.is_empty() {
        return Ok(true);
    }
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}

impl MigratePreset {
    fn as_str(self) -> &'static str {
        match self {
            Self::UserData => "user-data",
            Self::Full => "full",
        }
    }
}

impl SkillConflict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Overwrite => "overwrite",
            Self::Rename => "rename",
        }
    }
}

trait IsTerminal {
    fn is_terminal(&self) -> bool;
}

impl IsTerminal for io::Stdin {
    fn is_terminal(&self) -> bool {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe { libc::isatty(self.as_raw_fd()) == 1 }
        }
        #[cfg(windows)]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn test_env_lock() -> &'static Mutex<()> {
        crate::cli_test_env_lock()
    }

    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    fn test_context(temp: &TempDir) -> HermesContext {
        let hermes_home = temp.path().join(".hermes");
        fs::create_dir_all(&hermes_home).unwrap();
        HermesContext::new(temp.path()).with_hermes_home_env(Some(hermes_home))
    }

    #[cfg(unix)]
    fn install_fake_python(temp: &TempDir) -> PathBuf {
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let script = format!(
            "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
            temp.path().join("python.log").display()
        );
        for name in ["python", "python3"] {
            let path = bin_dir.join(name);
            fs::write(&path, &script).unwrap();
            let mut perms = fs::metadata(&path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&path, perms).unwrap();
        }
        bin_dir
    }

    #[test]
    fn archive_path_adds_timestamp_when_base_exists() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join(".openclaw");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(temp.path().join(".openclaw.pre-migration")).unwrap();

        let archive = archive_path_for(&source);
        assert!(
            archive
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".openclaw.pre-migration-")
        );
    }

    #[test]
    fn scan_workspace_state_finds_root_and_workspace_items() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join(".openclaw");
        std::fs::create_dir_all(source.join("proj").join("sessions")).unwrap();
        std::fs::write(source.join("todo.json"), b"{}").unwrap();
        std::fs::write(source.join("proj").join("SOUL.md"), b"x").unwrap();
        std::fs::write(source.join("proj").join("todo.json"), b"{}").unwrap();

        let findings = scan_workspace_state(&source);
        assert!(
            findings
                .iter()
                .any(|(_, desc)| desc == "Root file: todo.json")
        );
        assert!(
            findings
                .iter()
                .any(|(_, desc)| desc == "Workspace file: proj/todo.json")
        );
        assert_eq!(find_workspace_dirs(&source).len(), 1);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_without_openclaw_source_stays_native() {
        let _guard = test_env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = TempDir::new().unwrap();
        let context = test_context(&temp);
        let fake_bin = install_fake_python(&temp);
        let log = temp.path().join("python.log");

        let old_home = env::var_os("HOME");
        let old_path = env::var_os("PATH");
        set_env_var("HOME", temp.path());
        set_env_var("PATH", &fake_bin);
        let result = print_migrate(
            &context,
            MigrateArgs {
                source: None,
                dry_run: true,
                preset: MigratePreset::Full,
                overwrite: false,
                migrate_secrets: false,
                no_backup: false,
                workspace_target: None,
                skill_conflict: SkillConflict::Skip,
                yes: false,
            },
        );
        let python_called = log.exists();

        match old_home {
            Some(value) => set_env_var("HOME", value),
            None => remove_env_var("HOME"),
        }
        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_minimal_source_stays_native_without_optional_skill_script() {
        let _guard = test_env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = TempDir::new().unwrap();
        let context = test_context(&temp);
        let source = temp.path().join(".openclaw");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(source.join("workspace.default")).unwrap();
        fs::write(
            source.join("workspace.default").join("SOUL.md"),
            b"native soul",
        )
        .unwrap();
        let fake_bin = install_fake_python(&temp);
        let log = temp.path().join("python.log");
        let old_path = env::var_os("PATH");
        set_env_var("PATH", &fake_bin);
        let result = print_migrate(
            &context,
            MigrateArgs {
                source: Some(source),
                dry_run: true,
                preset: MigratePreset::Full,
                overwrite: false,
                migrate_secrets: false,
                no_backup: false,
                workspace_target: None,
                skill_conflict: SkillConflict::Skip,
                yes: false,
            },
        );
        let python_called = log.exists();

        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_empty_source_stays_native() {
        let _guard = test_env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = TempDir::new().unwrap();
        let context = test_context(&temp);
        let source = temp.path().join(".openclaw");
        fs::create_dir_all(&source).unwrap();
        let fake_bin = install_fake_python(&temp);
        let log = temp.path().join("python.log");
        let old_path = env::var_os("PATH");
        set_env_var("PATH", &fake_bin);
        let result = print_migrate(
            &context,
            MigrateArgs {
                source: Some(source),
                dry_run: true,
                preset: MigratePreset::Full,
                overwrite: false,
                migrate_secrets: false,
                no_backup: false,
                workspace_target: None,
                skill_conflict: SkillConflict::Skip,
                yes: false,
            },
        );
        let python_called = log.exists();

        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_config_only_source_stays_native() {
        let _guard = test_env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = TempDir::new().unwrap();
        let context = test_context(&temp);
        let source = temp.path().join(".openclaw");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("openclaw.json"), b"{}").unwrap();
        let fake_bin = install_fake_python(&temp);
        let log = temp.path().join("python.log");
        let old_path = env::var_os("PATH");
        set_env_var("PATH", &fake_bin);
        let result = print_migrate(
            &context,
            MigrateArgs {
                source: Some(source),
                dry_run: true,
                preset: MigratePreset::Full,
                overwrite: false,
                migrate_secrets: false,
                no_backup: false,
                workspace_target: None,
                skill_conflict: SkillConflict::Skip,
                yes: false,
            },
        );
        let python_called = log.exists();

        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_apply_creates_pre_migration_backup_without_python() {
        let _guard = test_env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = TempDir::new().unwrap();
        let context = test_context(&temp);
        let fake_bin = install_fake_python(&temp);
        let log = temp.path().join("python.log");
        let source = temp.path().join(".openclaw");
        fs::create_dir_all(source.join("workspace.default")).unwrap();
        fs::write(
            source.join("workspace.default").join("SOUL.md"),
            b"native soul",
        )
        .unwrap();

        let old_path = env::var_os("PATH");
        set_env_var("PATH", &fake_bin);
        print_migrate(
            &context,
            MigrateArgs {
                source: Some(source.clone()),
                dry_run: false,
                preset: MigratePreset::Full,
                overwrite: true,
                migrate_secrets: false,
                no_backup: false,
                workspace_target: None,
                skill_conflict: SkillConflict::Skip,
                yes: true,
            },
        )
        .unwrap();

        assert!(!log.exists());
        let backups = fs::read_dir(context.hermes_home().join("backups"))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("pre-migration-") && name.ends_with(".zip"))
            .collect::<Vec<_>>();
        assert_eq!(backups.len(), 1);
        assert_eq!(
            fs::read_to_string(context.hermes_home().join("SOUL.md")).unwrap(),
            "native soul"
        );
        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
    }
}
