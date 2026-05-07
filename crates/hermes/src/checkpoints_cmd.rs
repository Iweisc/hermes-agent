use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use clap::Subcommand;
use hermes_core::HermesContext;
use serde::Deserialize;

const STORE_DIRNAME: &str = "store";
const PROJECTS_DIRNAME: &str = "projects";
const INDEXES_DIRNAME: &str = "indexes";
const LEGACY_PREFIX: &str = "legacy-";
const REFS_PREFIX: &str = "refs/hermes";

#[derive(Subcommand, Debug)]
pub enum CheckpointsCommand {
    Status {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Prune {
        #[arg(long = "retention-days", default_value_t = 7)]
        retention_days: u64,
        #[arg(long = "max-size-mb", default_value_t = 500)]
        max_size_mb: u64,
        #[arg(long = "keep-orphans")]
        keep_orphans: bool,
    },
    Clear {
        #[arg(short = 'f', long)]
        force: bool,
    },
    #[command(name = "clear-legacy")]
    ClearLegacy {
        #[arg(short = 'f', long)]
        force: bool,
    },
}

#[derive(Debug, Clone)]
struct CheckpointStoreStatus {
    base: PathBuf,
    store_size_bytes: u64,
    legacy_size_bytes: u64,
    total_size_bytes: u64,
    project_count: usize,
    projects: Vec<ProjectStatus>,
    legacy_archives: Vec<LegacyArchive>,
}

#[derive(Debug, Clone)]
struct ProjectStatus {
    hash: String,
    workdir: String,
    exists: bool,
    last_touch: Option<f64>,
    commits: usize,
}

#[derive(Debug, Clone)]
struct LegacyArchive {
    name: String,
    size_bytes: u64,
    mtime: f64,
}

#[derive(Debug, Clone, Default)]
struct PruneResult {
    scanned: usize,
    deleted_orphan: usize,
    deleted_stale: usize,
    errors: usize,
    bytes_freed: u64,
}

#[derive(Debug, Deserialize)]
struct ProjectMeta {
    workdir: Option<String>,
    last_touch: Option<f64>,
}

#[derive(Debug)]
struct GitCommandResult {
    success: bool,
    stdout: String,
}

pub fn print_checkpoints(
    context: &HermesContext,
    command: Option<CheckpointsCommand>,
) -> Result<(), Box<dyn Error>> {
    let base = context.hermes_home().join("checkpoints");
    match command.unwrap_or(CheckpointsCommand::Status { limit: 20 }) {
        CheckpointsCommand::Status { limit } | CheckpointsCommand::List { limit } => {
            print_status(&base, limit)?
        }
        CheckpointsCommand::Prune {
            retention_days,
            max_size_mb,
            keep_orphans,
        } => {
            println!("Pruning checkpoint store…");
            println!("  retention_days:    {retention_days}");
            println!("  delete_orphans:    {}", !keep_orphans);
            println!("  max_total_size_mb: {max_size_mb}");
            println!();
            let result = prune_checkpoints(&base, retention_days, !keep_orphans, max_size_mb);
            println!("Scanned:         {}", result.scanned);
            println!("Deleted orphan:  {}", result.deleted_orphan);
            println!("Deleted stale:   {}", result.deleted_stale);
            println!("Errors:          {}", result.errors);
            println!("Bytes reclaimed: {}", format_bytes(result.bytes_freed));
        }
        CheckpointsCommand::Clear { force } => {
            let status = collect_status(&base)?;
            if status.total_size_bytes == 0 && !base.exists() {
                println!("Nothing to clear — checkpoint base does not exist.");
                return Ok(());
            }

            println!(
                "This will delete the ENTIRE checkpoint base at {}",
                status.base.display()
            );
            println!("  size:        {}", format_bytes(status.total_size_bytes));
            println!("  projects:    {}", status.project_count);
            println!("  legacy dirs: {}", status.legacy_archives.len());
            println!();
            println!("All /rollback history for every working directory will be lost.");
            if !force && !confirm_prompt("Proceed? [y/N] ")? {
                println!("Aborted.");
                return Ok(());
            }

            let result = clear_all(&base);
            if result.deleted_stale > 0 {
                println!("Cleared. Reclaimed {}.", format_bytes(result.bytes_freed));
            } else {
                println!("Could not clear checkpoint base (see logs).");
                return Err("checkpoint clear failed".into());
            }
        }
        CheckpointsCommand::ClearLegacy { force } => {
            let status = collect_status(&base)?;
            if status.legacy_archives.is_empty() {
                println!("No legacy archives to clear.");
                return Ok(());
            }

            let total = status
                .legacy_archives
                .iter()
                .map(|archive| archive.size_bytes)
                .sum::<u64>();
            println!(
                "Found {} legacy archive(s), total {}:",
                status.legacy_archives.len(),
                format_bytes(total)
            );
            for archive in &status.legacy_archives {
                println!(
                    "  {:<40}  {:>10}",
                    archive.name,
                    format_bytes(archive.size_bytes)
                );
            }
            println!();
            println!("Legacy archives hold pre-v2 per-project shadow repos, moved aside");
            println!("during the single-store migration. Delete when you're confident");
            println!("you don't need the old /rollback history.");
            if !force && !confirm_prompt("Delete all legacy archives? [y/N] ")? {
                println!("Aborted.");
                return Ok(());
            }

            let result = clear_legacy(&base);
            println!(
                "Deleted {} archive(s), reclaimed {}.",
                result.deleted_stale,
                format_bytes(result.bytes_freed)
            );
        }
    }
    Ok(())
}

fn print_status(base: &Path, limit: usize) -> Result<(), Box<dyn Error>> {
    let status = collect_status(base)?;
    println!("Checkpoint base: {}", status.base.display());
    println!("Total size:      {}", format_bytes(status.total_size_bytes));
    println!("  store/         {}", format_bytes(status.store_size_bytes));
    println!(
        "  legacy-*       {}",
        format_bytes(status.legacy_size_bytes)
    );
    println!("Projects:        {}", status.project_count);

    if !status.projects.is_empty() {
        println!();
        println!(
            "  {:<60}  {:>7}  {:>12}  STATE",
            "WORKDIR", "COMMITS", "LAST TOUCH"
        );
        for project in status.projects.iter().take(limit) {
            let workdir = truncate_left(&project.workdir, 60);
            let state = if project.exists { "live" } else { "orphan" };
            println!(
                "  {:<60}  {:>7}  {:>12}  {}",
                workdir,
                project.commits,
                format_age(project.last_touch),
                state
            );
        }
    }

    if !status.legacy_archives.is_empty() {
        println!();
        println!("Legacy archives ({}):", status.legacy_archives.len());
        for archive in &status.legacy_archives {
            println!(
                "  {:<40}  {:>10}",
                archive.name,
                format_bytes(archive.size_bytes)
            );
        }
        println!();
        println!("Clear with: hermes checkpoints clear-legacy");
    }

    Ok(())
}

fn collect_status(base: &Path) -> Result<CheckpointStoreStatus, Box<dyn Error>> {
    let mut status = CheckpointStoreStatus {
        base: base.to_path_buf(),
        store_size_bytes: 0,
        legacy_size_bytes: 0,
        total_size_bytes: 0,
        project_count: 0,
        projects: Vec::new(),
        legacy_archives: Vec::new(),
    };

    if !base.exists() {
        return Ok(status);
    }

    let store = store_path(base);
    if store.exists() {
        status.store_size_bytes = dir_size_bytes(&store);
        if store.join("HEAD").exists() {
            for project in list_projects(&store)? {
                let commits =
                    git_ref_count(&store, &format!("{REFS_PREFIX}/{}", project.hash)).unwrap_or(0);
                status.projects.push(ProjectStatus { commits, ..project });
            }
            status.projects.sort_by(|left, right| {
                right
                    .last_touch
                    .unwrap_or_default()
                    .partial_cmp(&left.last_touch.unwrap_or_default())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
    status.project_count = status.projects.len();

    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with(LEGACY_PREFIX) {
            continue;
        }
        let size_bytes = dir_size_bytes(&path);
        let mtime = modified_ts(&path).unwrap_or_default();
        status.legacy_size_bytes = status.legacy_size_bytes.saturating_add(size_bytes);
        status.legacy_archives.push(LegacyArchive {
            name,
            size_bytes,
            mtime,
        });
    }
    status
        .legacy_archives
        .sort_by(|left, right| right.mtime.total_cmp(&left.mtime));

    status.total_size_bytes = dir_size_bytes(base);
    Ok(status)
}

fn list_projects(store: &Path) -> Result<Vec<ProjectStatus>, Box<dyn Error>> {
    let projects_dir = store.join(PROJECTS_DIRNAME);
    if !projects_dir.exists() {
        return Ok(Vec::new());
    }
    let mut projects = Vec::new();
    for entry in fs::read_dir(projects_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("json")) {
            continue;
        }
        let hash = match path.file_stem().and_then(OsStr::to_str) {
            Some(value) if !value.trim().is_empty() => value.to_string(),
            _ => continue,
        };
        let meta: ProjectMeta = match serde_json::from_slice(&fs::read(&path)?) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let workdir = meta.workdir.unwrap_or_default();
        projects.push(ProjectStatus {
            hash,
            exists: !workdir.is_empty() && Path::new(&workdir).exists(),
            workdir,
            last_touch: meta.last_touch,
            commits: 0,
        });
    }
    Ok(projects)
}

fn prune_checkpoints(
    base: &Path,
    retention_days: u64,
    delete_orphans: bool,
    max_total_size_mb: u64,
) -> PruneResult {
    let mut result = PruneResult::default();
    if !base.exists() {
        return result;
    }

    let size_before = dir_size_bytes(base);
    let cutoff = if retention_days > 0 {
        Some(now_ts() - retention_days as f64 * 86_400.0)
    } else {
        None
    };

    let entries = match fs::read_dir(base) {
        Ok(entries) => entries,
        Err(_) => return result,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name == STORE_DIRNAME {
            continue;
        }
        if name.starts_with(LEGACY_PREFIX) {
            let Some(cutoff) = cutoff else {
                continue;
            };
            let mtime = modified_ts(&path).unwrap_or_default();
            if mtime >= cutoff {
                continue;
            }
            let size = dir_size_bytes(&path);
            match fs::remove_dir_all(&path) {
                Ok(()) => {
                    result.bytes_freed = result.bytes_freed.saturating_add(size);
                    result.deleted_stale += 1;
                }
                Err(_) => result.errors += 1,
            }
            continue;
        }
        if !path.join("HEAD").exists() {
            continue;
        }

        result.scanned += 1;
        let reason = if delete_orphans && legacy_shadow_is_orphan(&path) {
            Some("orphan")
        } else if cutoff.is_some_and(|cutoff| newest_mtime(&path).unwrap_or_default() < cutoff) {
            Some("stale")
        } else {
            None
        };
        let Some(reason) = reason else {
            continue;
        };
        let size = dir_size_bytes(&path);
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                result.bytes_freed = result.bytes_freed.saturating_add(size);
                if reason == "orphan" {
                    result.deleted_orphan += 1;
                } else {
                    result.deleted_stale += 1;
                }
            }
            Err(_) => result.errors += 1,
        }
    }

    let store = store_path(base);
    if store.join("HEAD").exists() {
        let projects = list_projects(&store).unwrap_or_default();
        for project in projects {
            result.scanned += 1;
            let reason = if delete_orphans && (!project.exists || project.workdir.is_empty()) {
                Some("orphan")
            } else if project.last_touch.is_some_and(|last_touch| {
                cutoff.is_some_and(|limit| last_touch > 0.0 && last_touch < limit)
            }) {
                Some("stale")
            } else {
                None
            };
            let Some(reason) = reason else {
                continue;
            };

            let ref_name = format!("{REFS_PREFIX}/{}", project.hash);
            let _ = git_success(&store, ["update-ref", "-d", ref_name.as_str()]);

            let index_path = store.join(INDEXES_DIRNAME).join(&project.hash);
            if index_path.exists() {
                let _ = fs::remove_file(index_path);
            }
            let meta_path = store
                .join(PROJECTS_DIRNAME)
                .join(format!("{}.json", project.hash));
            if meta_path.exists() {
                let _ = fs::remove_file(meta_path);
            }
            if reason == "orphan" {
                result.deleted_orphan += 1;
            } else {
                result.deleted_stale += 1;
            }
        }

        run_gc(&store);

        if max_total_size_mb > 0 {
            let cap_bytes = max_total_size_mb.saturating_mul(1024 * 1024);
            for _ in 0..20 {
                if dir_size_bytes(&store) <= cap_bytes {
                    break;
                }
                let refs = git_lines(&store, ["for-each-ref", "--format=%(refname)", REFS_PREFIX])
                    .unwrap_or_default();
                if refs.is_empty() {
                    break;
                }

                let mut any_drop = false;
                for ref_name in refs {
                    let count = git_ref_count(&store, &ref_name).unwrap_or(0);
                    if count <= 1 {
                        continue;
                    }
                    let commits = git_lines(&store, ["rev-list", "--reverse", ref_name.as_str()])
                        .unwrap_or_default();
                    if commits.len() <= 1 {
                        continue;
                    }
                    let keep = &commits[1..];
                    let mut new_parent: Option<String> = None;
                    let mut failed = false;
                    for sha in keep {
                        let tree_ref = format!("{sha}^{{tree}}");
                        let Some(tree_sha) =
                            git_single_line(&store, ["rev-parse", tree_ref.as_str()])
                        else {
                            failed = true;
                            break;
                        };
                        let message =
                            git_single_line(&store, ["log", "--format=%s", "-1", sha.as_str()])
                                .unwrap_or_else(|| String::from("checkpoint"));
                        let mut args = vec![
                            String::from("commit-tree"),
                            tree_sha,
                            String::from("-m"),
                            message,
                            String::from("--no-gpg-sign"),
                        ];
                        if let Some(parent) = new_parent.as_ref() {
                            args.splice(2..2, [String::from("-p"), parent.clone()]);
                        }
                        let Some(new_sha) = git_single_line_vec(&store, args) else {
                            failed = true;
                            break;
                        };
                        new_parent = Some(new_sha);
                    }
                    if failed {
                        continue;
                    }
                    let Some(new_parent) = new_parent else {
                        continue;
                    };
                    let _ = git_success(
                        &store,
                        ["update-ref", ref_name.as_str(), new_parent.as_str()],
                    );
                    any_drop = true;
                }

                if !any_drop {
                    break;
                }
            }

            run_gc(&store);
        }
    }

    let size_after = dir_size_bytes(base);
    let reclaimed = size_before.saturating_sub(size_after);
    if reclaimed > result.bytes_freed {
        result.bytes_freed = reclaimed;
    }
    result
}

fn legacy_shadow_is_orphan(path: &Path) -> bool {
    let marker = path.join("HERMES_WORKDIR");
    let workdir = fs::read_to_string(marker)
        .ok()
        .map(|value| value.trim().to_string());
    match workdir {
        Some(path) if !path.is_empty() => !Path::new(&path).exists(),
        _ => true,
    }
}

fn clear_all(base: &Path) -> PruneResult {
    let mut result = PruneResult::default();
    if !base.exists() {
        return result;
    }
    result.bytes_freed = dir_size_bytes(base);
    match fs::remove_dir_all(base) {
        Ok(()) => result.deleted_stale = 1,
        Err(_) => {
            result.deleted_stale = 0;
            result.bytes_freed = 0;
            result.errors = 1;
        }
    }
    result
}

fn clear_legacy(base: &Path) -> PruneResult {
    let mut result = PruneResult::default();
    if !base.exists() {
        return result;
    }
    let Ok(entries) = fs::read_dir(base) else {
        return result;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if !path.is_dir() || !name.starts_with(LEGACY_PREFIX) {
            continue;
        }
        let size = dir_size_bytes(&path);
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                result.bytes_freed = result.bytes_freed.saturating_add(size);
                result.deleted_stale += 1;
            }
            Err(_) => result.errors += 1,
        }
    }
    result
}

fn store_path(base: &Path) -> PathBuf {
    base.join(STORE_DIRNAME)
}

fn git_ref_count(store: &Path, ref_name: &str) -> Option<usize> {
    let result = run_git(
        store,
        vec!["rev-list".into(), "--count".into(), ref_name.into()],
    )
    .ok()?;
    if !result.success {
        return None;
    }
    result.stdout.trim().parse::<usize>().ok()
}

fn git_lines<const N: usize>(store: &Path, args: [&str; N]) -> Option<Vec<String>> {
    let result = run_git(
        store,
        args.into_iter().map(str::to_string).collect::<Vec<_>>(),
    )
    .ok()?;
    if !result.success {
        return None;
    }
    Some(
        result
            .stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

fn git_single_line<const N: usize>(store: &Path, args: [&str; N]) -> Option<String> {
    let result = run_git(
        store,
        args.into_iter().map(str::to_string).collect::<Vec<_>>(),
    )
    .ok()?;
    if !result.success {
        return None;
    }
    let line = result.stdout.trim();
    (!line.is_empty()).then(|| line.to_string())
}

fn git_single_line_vec(store: &Path, args: Vec<String>) -> Option<String> {
    let result = run_git(store, args).ok()?;
    if !result.success {
        return None;
    }
    let line = result.stdout.trim();
    (!line.is_empty()).then(|| line.to_string())
}

fn git_success<const N: usize>(store: &Path, args: [&str; N]) -> bool {
    run_git(
        store,
        args.into_iter().map(str::to_string).collect::<Vec<_>>(),
    )
    .map(|result| result.success)
    .unwrap_or(false)
}

fn run_gc(store: &Path) {
    let _ = git_success(store, ["reflog", "expire", "--expire=now", "--all"]);
    let _ = git_success(store, ["gc", "--prune=now", "--quiet"]);
}

fn run_git(store: &Path, args: Vec<String>) -> io::Result<GitCommandResult> {
    let mut command = Command::new("git");
    command.args(&args);
    command.env("GIT_DIR", store);
    command.env("GIT_CONFIG_GLOBAL", devnull_path());
    command.env("GIT_CONFIG_SYSTEM", devnull_path());
    command.env("GIT_CONFIG_NOSYSTEM", "1");
    command.env_remove("GIT_NAMESPACE");
    command.env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES");
    if let Some(parent) = store.parent() {
        command.current_dir(parent);
    }
    let output = command.output()?;
    Ok(GitCommandResult {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
    })
}

fn newest_mtime(path: &Path) -> Option<f64> {
    let mut newest = modified_ts(path).unwrap_or_default();
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if let Some(mtime) = modified_ts(&child) {
                if mtime > newest {
                    newest = mtime;
                }
            }
            if child.is_dir() {
                stack.push(child);
            }
        }
    }
    (newest > 0.0).then_some(newest)
}

fn modified_ts(path: &Path) -> Option<f64> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    modified
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|value| value.as_secs_f64())
}

fn dir_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            } else if metadata.is_dir() {
                stack.push(child);
            }
        }
    }
    total
}

fn truncate_left(value: &str, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars {
        return value.to_string();
    }
    let tail = value
        .chars()
        .skip(count.saturating_sub(max_chars.saturating_sub(1)))
        .collect::<String>();
    format!("…{tail}")
}

fn format_bytes(size: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = size as f64;
    for (index, unit) in units.iter().enumerate() {
        if value < 1024.0 || index == units.len() - 1 {
            return if *unit == "B" {
                format!("{} {unit}", value as u64)
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1024.0;
    }
    String::from("0 B")
}

fn format_age(timestamp: Option<f64>) -> String {
    let Some(timestamp) = timestamp else {
        return "—".to_string();
    };
    let age = now_ts() - timestamp;
    if !age.is_finite() {
        return "—".to_string();
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
    if age < 86_400.0 {
        return format!("{}h ago", (age / 3600.0) as i64);
    }
    format!("{}d ago", (age / 86_400.0) as i64)
}

#[allow(dead_code)]
fn format_ts(timestamp: Option<f64>) -> String {
    let Some(timestamp) = timestamp else {
        return "—".to_string();
    };
    let seconds = timestamp.trunc() as i64;
    Local
        .timestamp_opt(seconds, 0)
        .single()
        .map(|value| value.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "—".to_string())
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
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
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}

fn devnull_path() -> &'static str {
    #[cfg(windows)]
    {
        "NUL"
    }
    #[cfg(not(windows))]
    {
        "/dev/null"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    use tempfile::TempDir;

    #[test]
    fn collect_status_reports_project_and_legacy_archive() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("checkpoints");
        let store = store_path(&base);
        fs::create_dir_all(store.join(PROJECTS_DIRNAME)).unwrap();
        init_bare_store(&store);

        let workdir = temp.path().join("project");
        fs::create_dir_all(&workdir).unwrap();
        let hash = "abcd1234efgh5678";
        fs::write(
            store.join(PROJECTS_DIRNAME).join(format!("{hash}.json")),
            serde_json::json!({
                "workdir": workdir,
                "created_at": now_ts(),
                "last_touch": now_ts(),
            })
            .to_string(),
        )
        .unwrap();
        seed_ref(&store, &format!("{REFS_PREFIX}/{hash}"));

        let legacy = base.join("legacy-111");
        fs::create_dir_all(&legacy).unwrap();
        fs::write(legacy.join("old.txt"), b"legacy").unwrap();

        let status = collect_status(&base).unwrap();
        assert_eq!(status.project_count, 1);
        assert_eq!(status.projects[0].commits, 1);
        assert_eq!(status.projects[0].workdir, workdir.display().to_string());
        assert_eq!(status.legacy_archives.len(), 1);
        assert!(status.total_size_bytes >= status.store_size_bytes);
    }

    #[test]
    fn prune_checkpoints_deletes_stale_project_ref_and_metadata() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("checkpoints");
        let store = store_path(&base);
        fs::create_dir_all(store.join(PROJECTS_DIRNAME)).unwrap();
        init_bare_store(&store);

        let workdir = temp.path().join("project");
        fs::create_dir_all(&workdir).unwrap();
        let hash = "abcd1234efgh5678";
        fs::write(
            store.join(PROJECTS_DIRNAME).join(format!("{hash}.json")),
            serde_json::json!({
                "workdir": workdir,
                "created_at": now_ts() - 864000.0,
                "last_touch": now_ts() - 864000.0,
            })
            .to_string(),
        )
        .unwrap();
        seed_ref(&store, &format!("{REFS_PREFIX}/{hash}"));

        let result = prune_checkpoints(&base, 7, false, 0);
        assert_eq!(result.deleted_stale, 1);
        assert!(
            !store
                .join(PROJECTS_DIRNAME)
                .join(format!("{hash}.json"))
                .exists()
        );
        assert!(git_ref_count(&store, &format!("{REFS_PREFIX}/{hash}")).is_none());
    }

    #[test]
    fn clear_legacy_removes_only_legacy_dirs() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("checkpoints");
        let legacy = base.join("legacy-123");
        let store = base.join(STORE_DIRNAME);
        fs::create_dir_all(&legacy).unwrap();
        fs::create_dir_all(&store).unwrap();
        fs::write(legacy.join("a.txt"), b"x").unwrap();
        fs::write(store.join("HEAD"), b"ref: refs/heads/master\n").unwrap();

        let result = clear_legacy(&base);
        assert_eq!(result.deleted_stale, 1);
        assert!(!legacy.exists());
        assert!(store.exists());
    }

    fn init_bare_store(store: &Path) {
        fs::create_dir_all(store.parent().unwrap()).unwrap();
        let output = Command::new("git")
            .args(["init", "--bare", store.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git init failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn seed_ref(store: &Path, ref_name: &str) {
        let tree = command_with_store(store)
            .arg("mktree")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                child.stdin.take();
                child.wait_with_output()
            })
            .unwrap();
        assert!(tree.status.success());
        let tree_sha = String::from_utf8_lossy(&tree.stdout).trim().to_string();

        let commit = command_with_store(store)
            .env("GIT_AUTHOR_NAME", "Hermes")
            .env("GIT_AUTHOR_EMAIL", "hermes@example.com")
            .env("GIT_COMMITTER_NAME", "Hermes")
            .env("GIT_COMMITTER_EMAIL", "hermes@example.com")
            .args([
                "commit-tree",
                tree_sha.as_str(),
                "-m",
                "checkpoint",
                "--no-gpg-sign",
            ])
            .output()
            .unwrap();
        assert!(
            commit.status.success(),
            "git commit-tree failed: {}",
            String::from_utf8_lossy(&commit.stderr)
        );
        let sha = String::from_utf8_lossy(&commit.stdout).trim().to_string();

        let update = command_with_store(store)
            .args(["update-ref", ref_name, sha.as_str()])
            .output()
            .unwrap();
        assert!(
            update.status.success(),
            "git update-ref failed: {}",
            String::from_utf8_lossy(&update.stderr)
        );
    }

    fn command_with_store(store: &Path) -> Command {
        let mut command = Command::new("git");
        command.env("GIT_DIR", store);
        command.env("GIT_CONFIG_GLOBAL", devnull_path());
        command.env("GIT_CONFIG_SYSTEM", devnull_path());
        command.env("GIT_CONFIG_NOSYSTEM", "1");
        command.current_dir(store.parent().unwrap());
        command
    }
}
