use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Args, Subcommand, ValueEnum};

use crate::python_bridge::launch_python_main_command;

const OPENCLAW_DIR_NAMES: [&str; 3] = [".openclaw", ".clawdbot", ".moltbot"];

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

pub fn print_claw(command: Option<ClawCommand>) -> Result<(), Box<dyn Error>> {
    match command {
        Some(ClawCommand::Migrate(args)) => print_migrate(args),
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

fn print_migrate(args: MigrateArgs) -> Result<(), Box<dyn Error>> {
    if let Some(path) = args.source.as_ref() {
        validate_source(path)?;
    }
    if let Some(path) = args.workspace_target.as_ref() {
        if !path.is_absolute() {
            return Err("workspace-target must be an absolute path".into());
        }
    }

    let mut argv = Vec::new();
    if let Some(path) = args.source {
        argv.push(String::from("--source"));
        argv.push(path.display().to_string());
    }
    if args.dry_run {
        argv.push(String::from("--dry-run"));
    }
    if args.preset != MigratePreset::Full {
        argv.push(String::from("--preset"));
        argv.push(args.preset.as_str().to_string());
    }
    if args.overwrite {
        argv.push(String::from("--overwrite"));
    }
    if args.migrate_secrets {
        argv.push(String::from("--migrate-secrets"));
    }
    if args.no_backup {
        argv.push(String::from("--no-backup"));
    }
    if let Some(path) = args.workspace_target {
        argv.push(String::from("--workspace-target"));
        argv.push(path.display().to_string());
    }
    if args.skill_conflict != SkillConflict::Skip {
        argv.push(String::from("--skill-conflict"));
        argv.push(args.skill_conflict.as_str().to_string());
    }
    if args.yes {
        argv.push(String::from("--yes"));
    }

    let mut full = vec![String::from("migrate")];
    full.extend(argv);
    launch_python_main_command("claw", &full, Some("HERMES_CLAW_PYTHON"), &[])
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
    use tempfile::TempDir;

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
    fn build_migrate_args_preserves_requested_flags() {
        let args = MigrateArgs {
            source: Some(PathBuf::from("/tmp/openclaw")),
            dry_run: true,
            preset: MigratePreset::UserData,
            overwrite: true,
            migrate_secrets: true,
            no_backup: true,
            workspace_target: Some(PathBuf::from("/tmp/workspace")),
            skill_conflict: SkillConflict::Rename,
            yes: true,
        };

        let mut argv = vec![String::from("migrate")];
        if let Some(path) = args.source.as_ref() {
            argv.push(String::from("--source"));
            argv.push(path.display().to_string());
        }
        if args.dry_run {
            argv.push(String::from("--dry-run"));
        }
        if args.preset != MigratePreset::Full {
            argv.push(String::from("--preset"));
            argv.push(args.preset.as_str().to_string());
        }
        if args.overwrite {
            argv.push(String::from("--overwrite"));
        }
        if args.migrate_secrets {
            argv.push(String::from("--migrate-secrets"));
        }
        if args.no_backup {
            argv.push(String::from("--no-backup"));
        }
        if let Some(path) = args.workspace_target.as_ref() {
            argv.push(String::from("--workspace-target"));
            argv.push(path.display().to_string());
        }
        if args.skill_conflict != SkillConflict::Skip {
            argv.push(String::from("--skill-conflict"));
            argv.push(args.skill_conflict.as_str().to_string());
        }
        if args.yes {
            argv.push(String::from("--yes"));
        }

        assert_eq!(
            argv,
            vec![
                "migrate",
                "--source",
                "/tmp/openclaw",
                "--dry-run",
                "--preset",
                "user-data",
                "--overwrite",
                "--migrate-secrets",
                "--no-backup",
                "--workspace-target",
                "/tmp/workspace",
                "--skill-conflict",
                "rename",
                "--yes"
            ]
        );
    }
}
