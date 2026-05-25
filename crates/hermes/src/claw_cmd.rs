use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand, ValueEnum};
use serde_json::Value as JsonValue;

use crate::python_bridge::{project_root, resolve_repo_python};

const OPENCLAW_DIR_NAMES: [&str; 3] = [".openclaw", ".clawdbot", ".moltbot"];
const OPENCLAW_CONFIG_FILE_NAMES: [&str; 3] = ["openclaw.json", "clawdbot.json", "moltbot.json"];
const OPENCLAW_MIGRATION_SCRIPT_REL: [&str; 4] = [
    "migration",
    "openclaw-migration",
    "scripts",
    "openclaw_to_hermes.py",
];

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

    let root = project_root();
    let script_candidates = migration_script_candidates(&root, &detect_hermes_home());
    if !script_candidates.iter().any(|path| path.exists()) {
        print_migration_script_missing(&script_candidates);
        return Ok(());
    }

    if source_dir_is_empty(&source_dir) || source_dir_contains_only_empty_configs(&source_dir) {
        print_migration_banner();
        println!();
        println!("Nothing to migrate from OpenClaw.");
        return Ok(());
    }

    let python = resolve_repo_python(&root, Some("HERMES_CLAW_PYTHON"))
        .ok_or("could not find a Python interpreter for claw migrate")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env(
            "HERMES_CLAW_MIGRATE_DRY_RUN",
            if args.dry_run { "1" } else { "0" },
        )
        .env("HERMES_CLAW_MIGRATE_PRESET", args.preset.as_str())
        .env(
            "HERMES_CLAW_MIGRATE_OVERWRITE",
            if args.overwrite { "1" } else { "0" },
        )
        .env(
            "HERMES_CLAW_MIGRATE_SECRETS",
            if args.migrate_secrets { "1" } else { "0" },
        )
        .env(
            "HERMES_CLAW_MIGRATE_NO_BACKUP",
            if args.no_backup { "1" } else { "0" },
        )
        .env(
            "HERMES_CLAW_MIGRATE_SKILL_CONFLICT",
            args.skill_conflict.as_str(),
        )
        .env(
            "HERMES_CLAW_MIGRATE_SOURCE",
            source_dir.display().to_string(),
        )
        .env("HERMES_CLAW_MIGRATE_YES", if args.yes { "1" } else { "0" });
    if let Some(path) = args.workspace_target.as_ref() {
        command.env(
            "HERMES_CLAW_MIGRATE_WORKSPACE_TARGET",
            path.display().to_string(),
        );
    }
    command.arg("-c").arg(CLAW_MIGRATE_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("claw", status).into())
}

fn print_migration_banner() {
    println!();
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│          ⚕ Hermes — OpenClaw Migration                 │");
    println!("└─────────────────────────────────────────────────────────┘");
}

fn print_migration_script_missing(candidates: &[PathBuf]) {
    print_migration_banner();
    println!();
    println!("Migration script not found.");
    println!("Expected at one of:");
    for candidate in candidates {
        println!("  {}", candidate.display());
    }
    println!("Make sure the openclaw-migration skill is installed.");
}

const CLAW_MIGRATE_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "from hermes_cli.claw import _cmd_migrate\n",
    "_cmd_migrate(argparse.Namespace(\n",
    "    source=(os.environ.get('HERMES_CLAW_MIGRATE_SOURCE') or None),\n",
    "    dry_run=(os.environ.get('HERMES_CLAW_MIGRATE_DRY_RUN') == '1'),\n",
    "    preset=(os.environ.get('HERMES_CLAW_MIGRATE_PRESET') or 'full'),\n",
    "    overwrite=(os.environ.get('HERMES_CLAW_MIGRATE_OVERWRITE') == '1'),\n",
    "    migrate_secrets=(os.environ.get('HERMES_CLAW_MIGRATE_SECRETS') == '1'),\n",
    "    no_backup=(os.environ.get('HERMES_CLAW_MIGRATE_NO_BACKUP') == '1'),\n",
    "    workspace_target=(os.environ.get('HERMES_CLAW_MIGRATE_WORKSPACE_TARGET') or None),\n",
    "    skill_conflict=(os.environ.get('HERMES_CLAW_MIGRATE_SKILL_CONFLICT') or 'skip'),\n",
    "    yes=(os.environ.get('HERMES_CLAW_MIGRATE_YES') == '1'),\n",
    "))\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
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

fn migration_script_candidates(project_root: &Path, hermes_home: &Path) -> Vec<PathBuf> {
    vec![
        optional_skills_root(project_root)
            .join(OPENCLAW_MIGRATION_SCRIPT_REL.iter().collect::<PathBuf>()),
        hermes_home.join(
            ["skills"]
                .iter()
                .chain(OPENCLAW_MIGRATION_SCRIPT_REL.iter())
                .collect::<PathBuf>(),
        ),
    ]
}

fn optional_skills_root(project_root: &Path) -> PathBuf {
    env::var_os("HERMES_OPTIONAL_SKILLS")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("optional-skills"))
}

fn detect_hermes_home() -> PathBuf {
    env::var_os("HERMES_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/"))
                .join(".hermes")
        })
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
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
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
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let old_home = env::var_os("HOME");
        set_env_var("HOME", temp.path());
        set_env_var("HERMES_CLAW_PYTHON", &fake_python);
        let result = print_migrate(MigrateArgs {
            source: None,
            dry_run: true,
            preset: MigratePreset::Full,
            overwrite: false,
            migrate_secrets: false,
            no_backup: false,
            workspace_target: None,
            skill_conflict: SkillConflict::Skip,
            yes: false,
        });
        let python_called = log.exists();

        match old_home {
            Some(value) => set_env_var("HOME", value),
            None => remove_env_var("HOME"),
        }
        remove_env_var("HERMES_CLAW_PYTHON");
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_missing_script_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let source = temp.path().join(".openclaw");
        let optional_skills = temp.path().join("optional-skills");
        let hermes_home = temp.path().join(".hermes");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&optional_skills).unwrap();
        fs::create_dir_all(&hermes_home).unwrap();

        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let old_home = env::var_os("HERMES_HOME");
        let old_optional = env::var_os("HERMES_OPTIONAL_SKILLS");
        set_env_var("HERMES_HOME", &hermes_home);
        set_env_var("HERMES_OPTIONAL_SKILLS", &optional_skills);
        set_env_var("HERMES_CLAW_PYTHON", &fake_python);
        let result = print_migrate(MigrateArgs {
            source: Some(source),
            dry_run: true,
            preset: MigratePreset::Full,
            overwrite: false,
            migrate_secrets: false,
            no_backup: false,
            workspace_target: None,
            skill_conflict: SkillConflict::Skip,
            yes: false,
        });
        let python_called = log.exists();

        match old_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
        match old_optional {
            Some(value) => set_env_var("HERMES_OPTIONAL_SKILLS", value),
            None => remove_env_var("HERMES_OPTIONAL_SKILLS"),
        }
        remove_env_var("HERMES_CLAW_PYTHON");
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_empty_source_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let source = temp.path().join(".openclaw");
        let optional_skills = temp.path().join("optional-skills");
        let script = optional_skills
            .join("migration")
            .join("openclaw-migration")
            .join("scripts")
            .join("openclaw_to_hermes.py");
        let hermes_home = temp.path().join(".hermes");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(script.parent().unwrap()).unwrap();
        fs::write(&script, b"# placeholder").unwrap();
        fs::create_dir_all(&hermes_home).unwrap();

        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let old_home = env::var_os("HERMES_HOME");
        let old_optional = env::var_os("HERMES_OPTIONAL_SKILLS");
        set_env_var("HERMES_HOME", &hermes_home);
        set_env_var("HERMES_OPTIONAL_SKILLS", &optional_skills);
        set_env_var("HERMES_CLAW_PYTHON", &fake_python);
        let result = print_migrate(MigrateArgs {
            source: Some(source),
            dry_run: true,
            preset: MigratePreset::Full,
            overwrite: false,
            migrate_secrets: false,
            no_backup: false,
            workspace_target: None,
            skill_conflict: SkillConflict::Skip,
            yes: false,
        });
        let python_called = log.exists();

        match old_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
        match old_optional {
            Some(value) => set_env_var("HERMES_OPTIONAL_SKILLS", value),
            None => remove_env_var("HERMES_OPTIONAL_SKILLS"),
        }
        remove_env_var("HERMES_CLAW_PYTHON");
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_with_config_only_source_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let source = temp.path().join(".openclaw");
        let optional_skills = temp.path().join("optional-skills");
        let script = optional_skills
            .join("migration")
            .join("openclaw-migration")
            .join("scripts")
            .join("openclaw_to_hermes.py");
        let hermes_home = temp.path().join(".hermes");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("openclaw.json"), b"{}").unwrap();
        fs::create_dir_all(script.parent().unwrap()).unwrap();
        fs::write(&script, b"# placeholder").unwrap();
        fs::create_dir_all(&hermes_home).unwrap();

        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let old_home = env::var_os("HERMES_HOME");
        let old_optional = env::var_os("HERMES_OPTIONAL_SKILLS");
        set_env_var("HERMES_HOME", &hermes_home);
        set_env_var("HERMES_OPTIONAL_SKILLS", &optional_skills);
        set_env_var("HERMES_CLAW_PYTHON", &fake_python);
        let result = print_migrate(MigrateArgs {
            source: Some(source),
            dry_run: true,
            preset: MigratePreset::Full,
            overwrite: false,
            migrate_secrets: false,
            no_backup: false,
            workspace_target: None,
            skill_conflict: SkillConflict::Skip,
            yes: false,
        });
        let python_called = log.exists();

        match old_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
        match old_optional {
            Some(value) => set_env_var("HERMES_OPTIONAL_SKILLS", value),
            None => remove_env_var("HERMES_OPTIONAL_SKILLS"),
        }
        remove_env_var("HERMES_CLAW_PYTHON");
        result.unwrap();
        assert!(!python_called);
    }

    #[test]
    #[cfg(unix)]
    fn migrate_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        let source = temp.path().join(".openclaw");
        let workspace = temp.path().join("workspace");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("openclaw.json"),
            br#"{"model":{"provider":"openrouter"}}"#,
        )
        .unwrap();
        fs::create_dir_all(&workspace).unwrap();
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'source=%s dry=%s preset=%s overwrite=%s secrets=%s no_backup=%s workspace=%s conflict=%s yes=%s\\n' \\\n\
    \"$HERMES_CLAW_MIGRATE_SOURCE\" \"$HERMES_CLAW_MIGRATE_DRY_RUN\" \"$HERMES_CLAW_MIGRATE_PRESET\" \\\n\
    \"$HERMES_CLAW_MIGRATE_OVERWRITE\" \"$HERMES_CLAW_MIGRATE_SECRETS\" \"$HERMES_CLAW_MIGRATE_NO_BACKUP\" \\\n\
    \"$HERMES_CLAW_MIGRATE_WORKSPACE_TARGET\" \"$HERMES_CLAW_MIGRATE_SKILL_CONFLICT\" \"$HERMES_CLAW_MIGRATE_YES\" >> '{}'\n\
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

        set_env_var("HERMES_CLAW_PYTHON", &fake_python);
        print_migrate(MigrateArgs {
            source: Some(source.clone()),
            dry_run: true,
            preset: MigratePreset::UserData,
            overwrite: true,
            migrate_secrets: true,
            no_backup: true,
            workspace_target: Some(workspace.clone()),
            skill_conflict: SkillConflict::Rename,
            yes: true,
        })
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains(&format!(
            "source={} dry=1 preset=user-data overwrite=1 secrets=1 no_backup=1 workspace={} conflict=rename yes=1",
            source.display(),
            workspace.display()
        )));

        remove_env_var("HERMES_CLAW_PYTHON");
    }
}
