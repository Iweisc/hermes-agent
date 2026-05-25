use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Args;
use hermes_core::HermesContext;
use serde_yaml::Value as YamlValue;

use crate::backup::{create_pre_update_backup, create_quick_snapshot, format_size};
use crate::config_cmd::{migrate_config, read_raw_yaml_mapping};
use crate::dashboard_cmd::{ensure_dashboard_web_ui, stop_stale_dashboard_processes};
use crate::gateway_cmd::restart_gateways_after_update;
use crate::memory_cmd::sync_honcho_profiles;
use crate::python_bridge::{project_root, resolve_repo_python};
use crate::skills_cmd::sync_bundled_skills;

#[derive(Args, Debug, Clone)]
pub struct UpdateArgs {
    #[arg(long, default_value_t = false)]
    pub gateway: bool,
    #[arg(long, default_value_t = false)]
    pub check: bool,
    #[arg(long = "no-backup", default_value_t = false)]
    pub no_backup: bool,
    #[arg(long, default_value_t = false)]
    pub backup: bool,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

pub fn print_update(context: &HermesContext, args: UpdateArgs) -> Result<(), Box<dyn Error>> {
    if let Some(system) = get_managed_system(context) {
        eprintln!("{}", format_managed_message(&system, "update Hermes Agent"));
        return Ok(());
    }

    if args.check {
        return run_update_check();
    }

    print_update_apply(context, args)
}

fn print_update_apply(context: &HermesContext, args: UpdateArgs) -> Result<(), Box<dyn Error>> {
    print_update_apply_native(context, args)
}

fn print_update_apply_native(
    context: &HermesContext,
    args: UpdateArgs,
) -> Result<(), Box<dyn Error>> {
    let root = update_project_root();
    if !root.join(".git").exists() {
        return Err("Not a git repository — cannot update.".into());
    }

    let git_base = git_base_command();

    println!("⚕ Updating Hermes Agent...");
    println!();
    maybe_run_pre_update_backup(context, &args)?;
    println!("→ Fetching updates...");
    let fetch = run_git(&root, &git_base, &["fetch", "origin"])?;
    if !fetch.status.success() {
        return Err(map_fetch_error(&fetch.stderr).into());
    }

    let current_branch = run_git(&root, &git_base, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    if !current_branch.status.success() {
        let stderr = first_stderr_line(&current_branch.stderr).unwrap_or("git rev-parse failed");
        return Err(stderr.to_string().into());
    }
    let current_branch = current_branch.stdout.trim().to_string();

    let auto_stash_ref = if current_branch != "main" {
        let label = if current_branch == "HEAD" {
            String::from("detached HEAD")
        } else {
            format!("branch '{current_branch}'")
        };
        println!("  ⚠ Currently on {label} — switching to main for update...");
        let stash = stash_local_changes_if_needed(&root, &git_base)?;
        let checkout = run_git(&root, &git_base, &["checkout", "main"])?;
        if !checkout.status.success() {
            let stderr = first_stderr_line(&checkout.stderr).unwrap_or("git checkout main failed");
            return Err(stderr.to_string().into());
        }
        stash
    } else {
        stash_local_changes_if_needed(&root, &git_base)?
    };

    let prompt_for_restore = auto_stash_ref.is_some()
        && !args.yes
        && (args.gateway || (io::stdin().is_terminal() && io::stdout().is_terminal()));

    let rev = run_git(
        &root,
        &git_base,
        &["rev-list", "HEAD..origin/main", "--count"],
    )?;
    if !rev.status.success() {
        let stderr = first_stderr_line(&rev.stderr).unwrap_or("git rev-list failed");
        return Err(stderr.to_string().into());
    }
    let behind = rev.stdout.trim().parse::<u64>().unwrap_or(0);
    if behind == 0 {
        invalidate_update_cache(context);
        if let Some(stash_ref) = auto_stash_ref.as_deref() {
            let _ = restore_stashed_changes(
                context,
                &root,
                &git_base,
                stash_ref,
                prompt_for_restore,
                args.gateway,
            )?;
        }
        if !matches!(current_branch.as_str(), "main" | "HEAD") {
            let _ = run_git(&root, &git_base, &["checkout", current_branch.as_str()])?;
        }
        println!("✓ Already up to date!");
        return Ok(());
    }

    println!("→ Found {behind} new commit(s)");
    if let Some(snapshot_id) = create_quick_snapshot(context, Some("pre-update"))? {
        println!("  ✓ Pre-update snapshot: {snapshot_id}");
    }

    println!("→ Pulling updates...");
    let pull = run_git(&root, &git_base, &["pull", "--ff-only", "origin", "main"])?;
    if pull.status.success() {
    } else {
        println!("  ⚠ Fast-forward not possible, resetting to match origin/main...");
        let reset = run_git(&root, &git_base, &["reset", "--hard", "origin/main"])?;
        if !reset.status.success() {
            let stderr = first_stderr_line(&reset.stderr).unwrap_or("git reset failed");
            return Err(stderr.to_string().into());
        }
    }

    if let Some(stash_ref) = auto_stash_ref.as_deref() {
        let _ = restore_stashed_changes(
            context,
            &root,
            &git_base,
            stash_ref,
            prompt_for_restore,
            args.gateway,
        )?;
    }

    invalidate_update_cache(context);

    let removed = clear_bytecode_cache(&root)?;
    if removed > 0 {
        println!(
            "  ✓ Cleared {removed} stale __pycache__ director{}",
            if removed == 1 { "y" } else { "ies" }
        );
    }

    println!("→ Updating Python dependencies...");
    install_python_dependencies(&root)?;

    update_node_dependencies(&root)?;
    if let Err(error) = ensure_dashboard_web_ui(&root) {
        println!("  ⚠ Web UI build skipped: {error}");
    }

    sync_bundled_skills_after_update(context)?;
    sync_honcho_profiles_after_update(context)?;

    println!();
    println!("→ Checking configuration for new options...");
    migrate_config(context)?;

    if args.gateway {
        let _ = fs::write(context.hermes_home().join(".update_exit_code"), "0");
    }

    restart_gateways_after_update_native(context)?;
    stop_stale_dashboards_after_update()?;

    println!();
    println!("✓ Update complete!");
    for note in post_update_messages(args.gateway) {
        println!("  {note}");
    }

    Ok(())
}

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn run_update_check() -> Result<(), Box<dyn Error>> {
    let repo_dir = update_project_root();
    if !repo_dir.join(".git").exists() {
        return Err("Not a git repository — cannot check for updates.".into());
    }

    let git_base = git_base_command();

    println!("→ Fetching from upstream...");
    let upstream = run_git(&repo_dir, &git_base, &["fetch", "upstream"])?;
    let compare_branch = if upstream.status.success() {
        "upstream/main"
    } else {
        println!("→ Fetching from origin...");
        let origin = run_git(&repo_dir, &git_base, &["fetch", "origin"])?;
        if !origin.status.success() {
            return Err(map_fetch_error(&origin.stderr).into());
        }
        "origin/main"
    };

    let rev = run_git(
        &repo_dir,
        &git_base,
        &["rev-list", &format!("HEAD..{compare_branch}"), "--count"],
    )?;
    if !rev.status.success() {
        let stderr = first_stderr_line(&rev.stderr).unwrap_or("git rev-list failed");
        return Err(stderr.to_string().into());
    }
    let behind = rev.stdout.trim().parse::<u64>().unwrap_or(0);

    if behind == 0 {
        println!("✓ Already up to date.");
    } else {
        let commits_word = if behind == 1 { "commit" } else { "commits" };
        println!("⚕ Update available: {behind} {commits_word} behind {compare_branch}.");
        println!(
            "  Run '{}' to install.",
            recommended_update_command(context_managed_system())
        );
    }

    Ok(())
}

fn maybe_run_pre_update_backup(
    context: &HermesContext,
    args: &UpdateArgs,
) -> Result<(), Box<dyn Error>> {
    if args.no_backup {
        println!("◆ Pre-update backup: skipped (--no-backup)");
        println!();
        return Ok(());
    }

    let (enabled, keep) = read_update_backup_settings(context)?;
    if !enabled && !args.backup {
        return Ok(());
    }

    println!("◆ Creating pre-update backup...");
    match create_pre_update_backup(context, keep) {
        Ok(Some(path)) => {
            let size = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            println!("  Saved:    {} ({})", path.display(), format_size(size));
            println!("  Restore:  hermes import {}", path.display());
            println!("  Disable:  omit --backup (backups are off by default)");
            println!("            set updates.pre_update_backup: false in config.yaml");
            println!();
        }
        Ok(None) => {
            println!("  ⚠ Backup skipped (no files found or write failed); continuing update.");
            println!();
        }
        Err(error) => {
            println!("  ⚠ Backup failed: {error}");
            println!("  Continuing with update.");
            println!();
        }
    }
    Ok(())
}

fn sync_bundled_skills_after_update(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    println!();
    println!("→ Syncing bundled skills...");
    let result = sync_bundled_skills(context, true)?;
    if !result.copied.is_empty() {
        println!(
            "  + {} new: {}",
            result.copied.len(),
            result.copied.join(", ")
        );
    }
    if !result.updated.is_empty() {
        println!(
            "  ↑ {} updated: {}",
            result.updated.len(),
            result.updated.join(", ")
        );
    }
    if result.copied.is_empty() && result.updated.is_empty() {
        println!("  ✓ Skills are up to date");
    }

    let profile_homes = collect_profile_homes(context)?;
    if !profile_homes.is_empty() {
        println!();
        println!("→ Syncing bundled skills to all profiles...");
        for (name, home) in profile_homes {
            let profile_context = context.clone().with_hermes_home_env(Some(home));
            match sync_bundled_skills(&profile_context, true) {
                Ok(result) => {
                    let mut parts = Vec::new();
                    if !result.copied.is_empty() {
                        parts.push(format!("+{} new", result.copied.len()));
                    }
                    if !result.updated.is_empty() {
                        parts.push(format!("↑{} updated", result.updated.len()));
                    }
                    let status = if parts.is_empty() {
                        String::from("up to date")
                    } else {
                        parts.join(", ")
                    };
                    println!("  {name}: {status}");
                }
                Err(error) => {
                    println!("  {name}: error ({error})");
                }
            }
        }
    }
    Ok(())
}

fn sync_honcho_profiles_after_update(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let synced = sync_honcho_profiles(context)?;
    if synced > 0 {
        println!();
        println!("→ Syncing Honcho profiles...");
        println!("  ✓ Synced {synced} profile(s)");
    }
    Ok(())
}

fn restart_gateways_after_update_native(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let summary = restart_gateways_after_update(context)?;
    if summary.restarted_services.is_empty()
        && summary.restarted_profiles.is_empty()
        && summary.stopped_manual == 0
    {
        return Ok(());
    }

    println!();
    println!("→ Restarting gateways...");
    for service in summary.restarted_services {
        println!("  ✓ Restarted {service}");
    }
    if !summary.restarted_profiles.is_empty() {
        println!(
            "  ✓ Restarting manual gateway profile(s): {}",
            summary.restarted_profiles.join(", ")
        );
    }
    if summary.stopped_manual > 0 {
        println!(
            "  → Stopped {} manual gateway process(es)",
            summary.stopped_manual
        );
        println!("    Restart manually: hermes gateway run");
        if summary.stopped_manual > 1 {
            println!("    (or: hermes -p <profile> gateway run  for each profile)");
        }
    }
    Ok(())
}

fn stop_stale_dashboards_after_update() -> Result<(), Box<dyn Error>> {
    let _ = stop_stale_dashboard_processes("code updated")?;
    Ok(())
}

fn post_update_messages(gateway_mode: bool) -> Vec<String> {
    let _ = gateway_mode;
    Vec::new()
}

fn collect_profile_homes(
    context: &HermesContext,
) -> Result<Vec<(String, PathBuf)>, Box<dyn Error>> {
    let mut homes = Vec::new();
    let default_home = context.default_hermes_root();
    if default_home.is_dir() {
        homes.push((String::from("default"), default_home));
    }
    let profiles_root = context.profiles_root();
    if profiles_root.is_dir() {
        let mut entries = fs::read_dir(&profiles_root)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            homes.push((name, path));
        }
    }
    Ok(homes)
}

fn read_update_backup_settings(context: &HermesContext) -> Result<(bool, usize), Box<dyn Error>> {
    let mapping = read_raw_yaml_mapping(&context.config_path())?;
    let updates = mapping
        .get(&YamlValue::String(String::from("updates")))
        .and_then(YamlValue::as_mapping);
    let enabled = updates
        .and_then(|value| value.get(&YamlValue::String(String::from("pre_update_backup"))))
        .and_then(YamlValue::as_bool)
        .unwrap_or(false);
    let keep = updates
        .and_then(|value| value.get(&YamlValue::String(String::from("backup_keep"))))
        .and_then(YamlValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(5)
        .max(1);
    Ok((enabled, keep))
}

fn run_git(
    repo_dir: &Path,
    git_base: &[String],
    args: &[&str],
) -> Result<GitResult, Box<dyn Error>> {
    let mut command = Command::new(&git_base[0]);
    if git_base.len() > 1 {
        command.args(&git_base[1..]);
    }
    command.current_dir(repo_dir).args(args);
    let output = command.output()?;
    Ok(GitResult {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn map_fetch_error(stderr: &str) -> &'static str {
    if stderr.contains("Could not resolve host") || stderr.contains("unable to access") {
        "Network error — cannot reach the remote repository."
    } else if stderr.contains("Authentication failed") || stderr.contains("could not read Username")
    {
        "Authentication failed — check your git credentials or SSH key."
    } else if !stderr.trim().is_empty() {
        "Failed to fetch."
    } else {
        "Failed to fetch."
    }
}

fn first_stderr_line(stderr: &str) -> Option<&str> {
    stderr.lines().find(|line| !line.trim().is_empty())
}

fn update_project_root() -> PathBuf {
    std::env::var("HERMES_UPDATE_PROJECT_ROOT")
        .ok()
        .map(|value| PathBuf::from(value.trim()))
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(project_root)
}

fn git_base_command() -> Vec<String> {
    let binary = resolve_binary("HERMES_UPDATE_GIT", "git")
        .unwrap_or_else(|| PathBuf::from("git"))
        .display()
        .to_string();
    let mut git_base = vec![binary];
    if cfg!(windows) {
        git_base.extend([
            String::from("-c"),
            String::from("windows.appendAtomically=false"),
        ]);
    }
    git_base
}

fn resolve_binary(override_env: &str, default_name: &str) -> Option<PathBuf> {
    if let Some(value) = std::env::var(override_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Some(PathBuf::from(value));
    }
    which_on_path(default_name)
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn stash_local_changes_if_needed(
    repo_dir: &Path,
    git_base: &[String],
) -> Result<Option<String>, Box<dyn Error>> {
    let status = run_git(repo_dir, git_base, &["status", "--porcelain"])?;
    if !status.status.success() || status.stdout.trim().is_empty() {
        return Ok(None);
    }

    let unmerged = run_git(repo_dir, git_base, &["ls-files", "--unmerged"])?;
    if !unmerged.stdout.trim().is_empty() {
        println!("→ Clearing unmerged index entries from a previous conflict...");
        let _ = run_git(repo_dir, git_base, &["reset"])?;
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);
    println!("→ Local changes detected — stashing before update...");
    let stash = run_git(
        repo_dir,
        git_base,
        &[
            "stash",
            "push",
            "--include-untracked",
            "-m",
            &format!("hermes-update-autostash-{stamp}"),
        ],
    )?;
    if !stash.status.success() {
        let stderr = first_stderr_line(&stash.stderr).unwrap_or("git stash failed");
        return Err(stderr.to_string().into());
    }

    let stash_ref = run_git(repo_dir, git_base, &["rev-parse", "--verify", "refs/stash"])?;
    if !stash_ref.status.success() {
        let stderr = first_stderr_line(&stash_ref.stderr).unwrap_or("git rev-parse stash failed");
        return Err(stderr.to_string().into());
    }
    Ok(Some(stash_ref.stdout.trim().to_string()))
}

fn restore_stashed_changes(
    context: &HermesContext,
    repo_dir: &Path,
    git_base: &[String],
    stash_ref: &str,
    prompt_user: bool,
    gateway_mode: bool,
) -> Result<bool, Box<dyn Error>> {
    if prompt_user {
        println!();
        println!("⚠ Local changes were stashed before updating.");
        println!("  Restoring them may reapply local customizations onto the updated codebase.");
        println!("  Review the result afterward if Hermes behaves unexpectedly.");
        print!("Restore local changes now? [Y/n]: ");
        io::stdout().flush()?;
        let response = if gateway_mode {
            gateway_prompt(context, "Restore local changes now? [Y/n]", "y", 300)
        } else {
            let mut response = String::new();
            io::stdin().read_line(&mut response)?;
            response.trim().to_string()
        };
        let response = response.trim().to_ascii_lowercase();
        if !matches!(response.as_str(), "" | "y" | "yes") {
            println!("Skipped restoring local changes.");
            println!("Your changes are still preserved in git stash.");
            println!("Restore manually with: git stash apply {stash_ref}");
            return Ok(false);
        }
    }

    println!("→ Restoring local changes...");
    let restore = run_git(repo_dir, git_base, &["stash", "apply", stash_ref])?;
    let unmerged = run_git(
        repo_dir,
        git_base,
        &["diff", "--name-only", "--diff-filter=U"],
    )?;
    let has_conflicts = !unmerged.stdout.trim().is_empty();
    if !restore.status.success() || has_conflicts {
        println!("✗ Update pulled new code, but restoring local changes hit conflicts.");
        if !restore.stdout.trim().is_empty() {
            println!("{}", restore.stdout.trim());
        }
        if !restore.stderr.trim().is_empty() {
            println!("{}", restore.stderr.trim());
        }
        if has_conflicts {
            println!();
            println!("Conflicted files:");
            for file in unmerged
                .stdout
                .lines()
                .filter(|line| !line.trim().is_empty())
            {
                println!("  • {file}");
            }
        }
        println!();
        println!("Your stashed changes are preserved — nothing is lost.");
        println!("  Stash ref: {stash_ref}");
        let _ = run_git(repo_dir, git_base, &["reset", "--hard", "HEAD"])?;
        println!("Working tree reset to clean state.");
        println!("Restore your changes later with: git stash apply {stash_ref}");
        return Ok(false);
    }

    if let Some(selector) = resolve_stash_selector(repo_dir, git_base, stash_ref)? {
        let drop = run_git(repo_dir, git_base, &["stash", "drop", selector.as_str()])?;
        if !drop.status.success() {
            println!(
                "⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry."
            );
        }
    } else {
        println!(
            "⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop."
        );
    }

    println!("⚠ Local changes were restored on top of the updated codebase.");
    println!("  Review `git diff` / `git status` if Hermes behaves unexpectedly.");
    Ok(true)
}

fn gateway_prompt(
    context: &HermesContext,
    prompt_text: &str,
    default: &str,
    timeout_secs: u64,
) -> String {
    let prompt_path = context.hermes_home().join(".update_prompt.json");
    let response_path = context.hermes_home().join(".update_response");
    let _ = fs::remove_file(&response_path);
    let payload = serde_json::json!({
        "prompt": prompt_text,
        "default": default,
        "id": format!(
            "{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|value| value.as_nanos())
                .unwrap_or(0)
        ),
    });
    if let Ok(text) = serde_json::to_string(&payload) {
        let _ = atomic_write_json(&prompt_path, &text);
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs.max(1));
    while std::time::Instant::now() < deadline {
        if let Ok(answer) = fs::read_to_string(&response_path) {
            let _ = fs::remove_file(&response_path);
            let _ = fs::remove_file(&prompt_path);
            let trimmed = answer.trim();
            return if trimmed.is_empty() {
                default.to_string()
            } else {
                trimmed.to_string()
            };
        }
        sleep(Duration::from_millis(500));
    }

    let _ = fs::remove_file(&prompt_path);
    let _ = fs::remove_file(&response_path);
    println!("  (no response after {timeout_secs}s, using default: {default:?})");
    default.to_string()
}

fn atomic_write_json(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, content)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn resolve_stash_selector(
    repo_dir: &Path,
    git_base: &[String],
    stash_ref: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let stash_list = run_git(repo_dir, git_base, &["stash", "list", "--format=%gd %H"])?;
    if !stash_list.status.success() {
        return Ok(None);
    }
    for line in stash_list.stdout.lines() {
        let Some((selector, commit)) = line.split_once(' ') else {
            continue;
        };
        if commit.trim() == stash_ref {
            return Ok(Some(selector.trim().to_string()));
        }
    }
    Ok(None)
}

fn invalidate_update_cache(context: &HermesContext) {
    let default_home = context.default_hermes_root();
    let mut homes = vec![default_home.clone()];
    let profiles_root = default_home.join("profiles");
    if let Ok(entries) = fs::read_dir(&profiles_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                homes.push(path);
            }
        }
    }
    for home in homes {
        let cache = home.join(".update_check");
        let _ = fs::remove_file(cache);
    }
}

fn clear_bytecode_cache(root: &Path) -> Result<usize, Box<dyn Error>> {
    let mut removed = 0_usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                "venv" | ".venv" | "node_modules" | ".git" | ".worktrees"
            ) {
                continue;
            }
            if name == "__pycache__" {
                if fs::remove_dir_all(&path).is_ok() {
                    removed += 1;
                }
                continue;
            }
            stack.push(path);
        }
    }
    Ok(removed)
}

fn install_python_dependencies(root: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(uv) = resolve_binary("HERMES_UPDATE_UV", "uv") {
        let status = Command::new(&uv)
            .current_dir(root)
            .env("VIRTUAL_ENV", root.join("venv"))
            .arg("pip")
            .arg("install")
            .arg("-e")
            .arg(".[all]")
            .status()?;
        if status.success() {
            return Ok(());
        }
        return Err(exit_status_message("uv pip install", status).into());
    }

    let python = resolve_repo_python(root, Some("HERMES_UPDATE_PYTHON"))
        .ok_or("could not find a Python interpreter for dependency refresh")?;
    let pip_check = Command::new(&python)
        .current_dir(root)
        .args(["-m", "pip", "--version"])
        .status()?;
    if !pip_check.success() {
        let ensure = Command::new(&python)
            .current_dir(root)
            .args(["-m", "ensurepip", "--upgrade", "--default-pip"])
            .status()?;
        if !ensure.success() {
            return Err(exit_status_message("python -m ensurepip", ensure).into());
        }
    }
    let status = Command::new(&python)
        .current_dir(root)
        .args(["-m", "pip", "install", "-e", ".[all]"])
        .status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("python -m pip install", status).into())
}

fn update_node_dependencies(root: &Path) -> Result<(), Box<dyn Error>> {
    let Some(npm) = resolve_binary("HERMES_UPDATE_NPM", "npm") else {
        return Ok(());
    };

    let paths = [("repo root", root), ("ui-tui", &root.join("ui-tui"))];
    if !paths
        .iter()
        .any(|(_, path)| path.join("package.json").exists())
    {
        return Ok(());
    }

    println!("→ Updating Node.js dependencies...");
    for (label, path) in paths {
        if !path.join("package.json").exists() {
            continue;
        }
        let result = run_npm_install_deterministic(
            &npm,
            path,
            &["--silent", "--no-fund", "--no-audit", "--progress=false"],
        )?;
        if result.status.success() {
            println!("  ✓ {label}");
            continue;
        }

        println!("  ⚠ npm install failed in {label}");
        let stderr = String::from_utf8_lossy(&result.stderr);
        if let Some(line) = stderr.lines().rev().find(|line| !line.trim().is_empty()) {
            println!("    {line}");
        }
    }
    Ok(())
}

fn run_npm_install_deterministic(
    npm: &Path,
    cwd: &Path,
    extra_args: &[&str],
) -> Result<Output, Box<dyn Error>> {
    let lockfile = cwd.join("package-lock.json");
    if lockfile.exists() {
        let ci = Command::new(npm)
            .current_dir(cwd)
            .arg("ci")
            .args(extra_args)
            .output()?;
        if ci.status.success() {
            return Ok(ci);
        }
    }
    Ok(Command::new(npm)
        .current_dir(cwd)
        .arg("install")
        .args(extra_args)
        .output()?)
}

fn get_managed_system(context: &HermesContext) -> Option<String> {
    if let Ok(raw) = std::env::var("HERMES_MANAGED") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let normalized = trimmed.to_ascii_lowercase();
            return Some(match normalized.as_str() {
                "true" | "1" | "yes" => String::from("NixOS"),
                "brew" | "homebrew" => String::from("Homebrew"),
                "nix" | "nixos" => String::from("NixOS"),
                _ => trimmed.to_string(),
            });
        }
    }
    context
        .hermes_home()
        .join(".managed")
        .exists()
        .then_some(String::from("NixOS"))
}

fn context_managed_system() -> Option<&'static str> {
    let raw = std::env::var("HERMES_MANAGED").ok();
    match raw
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "brew" | "homebrew" => Some("Homebrew"),
            "true" | "1" | "yes" | "nix" | "nixos" => Some("NixOS"),
            _ => None,
        },
        None => None,
    }
}

fn recommended_update_command(managed_system: Option<&str>) -> &'static str {
    match managed_system {
        Some("Homebrew") => "brew upgrade hermes-agent",
        Some("NixOS") => "sudo nixos-rebuild switch",
        _ => "hermes update",
    }
}

fn format_managed_message(system: &str, action: &str) -> String {
    let raw = std::env::var("HERMES_MANAGED").unwrap_or_default();
    let normalized = raw.trim().to_ascii_lowercase();
    if system == "NixOS" {
        let env_hint = if matches!(normalized.as_str(), "true" | "1" | "yes") {
            "true"
        } else if raw.trim().is_empty() {
            "true"
        } else {
            raw.trim()
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by NixOS (HERMES_MANAGED={env_hint}).\nEdit services.hermes-agent.settings in your configuration.nix and run:\n  sudo nixos-rebuild switch"
        );
    }
    if system == "Homebrew" {
        let env_hint = if raw.trim().is_empty() {
            "homebrew"
        } else {
            raw.trim()
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by Homebrew (HERMES_MANAGED={env_hint}).\nUse:\n  brew upgrade hermes-agent"
        );
    }
    format!(
        "Cannot {action}: this Hermes installation is managed by {system}.\nUse your package manager to upgrade or reinstall Hermes."
    )
}

struct GitResult {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::TempDir;

    #[cfg(test)]
    fn test_env_lock() -> &'static std::sync::Mutex<()> {
        crate::cli_test_env_lock()
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
        std::env::temp_dir().join(format!("hermes-rs-update-{label}-{unique}"))
    }

    #[test]
    #[cfg(unix)]
    fn gateway_update_uses_prompt_files_and_writes_exit_code_marker() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join(".hermes");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        let log = temp.path().join("git.log");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&bin).unwrap();
        fs::write(
            &fake_git,
            format!(
                "#!/bin/sh\n\
printf 'git %s\\n' \"$*\" >> '{}'\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) printf ' M local.txt\\n'; exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
                log.display()
            ),
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        for path in [&fake_git, &fake_uv] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);

        let prompt_home = home.clone();
        let responder = std::thread::spawn(move || {
            let prompt_path = prompt_home.join(".update_prompt.json");
            let response_path = prompt_home.join(".update_response");
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while std::time::Instant::now() < deadline {
                if prompt_path.exists() {
                    fs::write(response_path, "n\n").unwrap();
                    return;
                }
                sleep(Duration::from_millis(100));
            }
            panic!("timed out waiting for update prompt");
        });

        print_update(
            &context,
            UpdateArgs {
                gateway: true,
                check: false,
                no_backup: false,
                backup: false,
                yes: false,
            },
        )
        .unwrap();
        responder.join().unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("git stash push --include-untracked"));
        assert!(!output.contains("git stash apply"));
        assert_eq!(
            fs::read_to_string(home.join(".update_exit_code")).unwrap(),
            "0"
        );

        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    fn post_update_messages_are_empty_for_plain_native_update() {
        assert!(post_update_messages(false).is_empty());
    }

    #[test]
    fn post_update_messages_are_empty_for_gateway_update_too() {
        assert!(post_update_messages(true).is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn native_update_runs_git_and_uv_and_migrates_config() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        let log = temp.path().join("actions.log");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&bin).unwrap();

        fs::write(
            &fake_git,
            format!(
                "#!/bin/sh\n\
printf 'git %s\\n' \"$*\" >> '{}'\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
                log.display()
            ),
        )
        .unwrap();
        fs::write(
            &fake_uv,
            format!(
                "#!/bin/sh\nprintf 'uv %s\\n' \"$*\" >> '{}'\nexit 0\n",
                log.display()
            ),
        )
        .unwrap();

        for path in [&fake_git, &fake_uv] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: false,
                yes: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("git fetch origin"));
        assert!(output.contains("git rev-list HEAD..origin/main --count"));
        assert!(output.contains("git pull --ff-only origin main"));
        assert!(output.contains("uv pip install -e .[all]"));
        assert!(context.config_path().exists());
        assert!(context.env_path().exists());

        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    #[cfg(unix)]
    fn native_update_backup_flag_writes_pre_update_zip() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&bin).unwrap();
        fs::write(home.join("config.yaml"), "display:\n  skin: slate\n").unwrap();

        fs::write(
            &fake_git,
            "#!/bin/sh\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        for path in [&fake_git, &fake_uv] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: true,
                yes: true,
            },
        )
        .unwrap();

        let backups = home.join("backups");
        let entries = fs::read_dir(&backups)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(entries.len(), 1);
        let backup_name = entries[0].file_name().to_string_lossy().to_string();
        assert!(backup_name.starts_with("pre-update-"));
        assert!(backup_name.ends_with(".zip"));

        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    #[cfg(unix)]
    fn native_update_syncs_bundled_skills_to_profiles() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let bundled = temp.path().join("bundled");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        let skill_dir = bundled.join("dev").join("demo");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(home.join("profiles").join("coder")).unwrap();
        fs::create_dir_all(&skill_dir).unwrap();
        fs::create_dir_all(&bin).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: demo skill\n---\nbody\n",
        )
        .unwrap();

        fs::write(
            &fake_git,
            "#!/bin/sh\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        for path in [&fake_git, &fake_uv] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);
        set_env_var("HERMES_BUNDLED_SKILLS", &bundled);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: false,
                yes: true,
            },
        )
        .unwrap();

        assert!(
            home.join("skills")
                .join("dev")
                .join("demo")
                .join("SKILL.md")
                .exists()
        );
        assert!(
            home.join("profiles")
                .join("coder")
                .join("skills")
                .join("dev")
                .join("demo")
                .join("SKILL.md")
                .exists()
        );
        assert!(home.join("skills").join(".bundled_manifest").exists());
        assert!(
            home.join("profiles")
                .join("coder")
                .join("skills")
                .join(".bundled_manifest")
                .exists()
        );

        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
        remove_env_var("HERMES_BUNDLED_SKILLS");
    }

    #[test]
    #[cfg(unix)]
    fn native_update_syncs_honcho_host_blocks_to_profiles() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(home.join("profiles").join("coder")).unwrap();
        fs::create_dir_all(&bin).unwrap();

        let honcho = serde_json::json!({
            "apiKey": "test-key",
            "workspace": "shared-root",
            "peerName": "alice",
            "hosts": {
                "hermes": {
                    "workspace": "shared-memory",
                    "peerName": "alice",
                    "enabled": true,
                    "writeFrequency": "session",
                    "recallMode": "hybrid"
                }
            }
        });
        fs::write(
            home.join("honcho.json"),
            format!("{}\n", serde_json::to_string_pretty(&honcho).unwrap()),
        )
        .unwrap();

        fs::write(
            &fake_git,
            "#!/bin/sh\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        for path in [&fake_git, &fake_uv] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: false,
                yes: true,
            },
        )
        .unwrap();

        let saved = serde_json::from_str::<serde_json::Value>(
            &fs::read_to_string(home.join("honcho.json")).unwrap(),
        )
        .unwrap();
        let coder = saved
            .get("hosts")
            .and_then(serde_json::Value::as_object)
            .and_then(|hosts| hosts.get("hermes.coder"))
            .and_then(serde_json::Value::as_object)
            .unwrap();
        assert_eq!(
            coder.get("aiPeer").and_then(serde_json::Value::as_str),
            Some("coder")
        );
        assert_eq!(
            coder.get("workspace").and_then(serde_json::Value::as_str),
            Some("shared-memory")
        );
        assert_eq!(
            coder.get("peerName").and_then(serde_json::Value::as_str),
            Some("alice")
        );
        assert_eq!(
            coder
                .get("writeFrequency")
                .and_then(serde_json::Value::as_str),
            Some("session")
        );
        assert_eq!(
            coder.get("enabled").and_then(serde_json::Value::as_bool),
            Some(true)
        );

        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    #[cfg(unix)]
    fn native_update_restarts_gateway_services() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let user_home = temp.path().join("user");
        let home = user_home.join(".hermes");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        let fake_systemctl = bin.join("systemctl");
        let log = home.join("systemctl.log");
        let profile_home = home.join("profiles").join("coder");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&profile_home).unwrap();
        let unit_dir = user_home.join(".config").join("systemd").join("user");
        fs::create_dir_all(&unit_dir).unwrap();
        fs::write(unit_dir.join("hermes-gateway.service"), "unit").unwrap();
        fs::write(unit_dir.join("hermes-gateway-coder.service"), "unit").unwrap();

        fs::write(
            &fake_git,
            "#!/bin/sh\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        fs::write(
            &fake_systemctl,
            format!(
                "#!/bin/sh\n\
printf '%s\\n' \"$*\" >> '{}'\n\
if [ \"$1\" = \"--user\" ] && [ \"$2\" = \"list-units\" ]; then\n\
  printf 'hermes-gateway.service loaded active running Hermes\\n'\n\
  printf 'hermes-gateway-coder.service loaded active running Hermes\\n'\n\
  exit 0\n\
fi\n\
if [ \"$1\" = \"--user\" ] && [ \"$2\" = \"is-active\" ]; then\n\
  printf 'active\\n'\n\
  exit 0\n\
fi\n\
if [ \"$1\" = \"list-units\" ]; then\n\
  exit 0\n\
fi\n\
if [ \"$1\" = \"--user\" ] && [ \"$2\" = \"show\" ]; then\n\
  printf '0\\n'\n\
  exit 0\n\
fi\n\
if [ \"$1\" = \"show\" ]; then\n\
  printf '0\\n'\n\
  exit 0\n\
fi\n\
exit 0\n",
                log.display()
            ),
        )
        .unwrap();
        for path in [&fake_git, &fake_uv, &fake_systemctl] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", bin.display(), original_path));

        let context = HermesContext::new(&user_home).with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: false,
                yes: true,
            },
        )
        .unwrap();

        let logged = fs::read_to_string(&log).unwrap();
        assert!(logged.contains("--user reload-or-restart hermes-gateway"));
        assert!(logged.contains("--user reload-or-restart hermes-gateway-coder"));

        set_env_var("PATH", original_path);
        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    #[cfg(unix)]
    fn native_update_restarts_manual_gateways_and_stops_dashboards() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let bin = temp.path().join("bin");
        let fake_git = bin.join("git");
        let fake_uv = bin.join("uv");
        let fake_gateway = bin.join("hermes");
        let log = home.join("gateway-restart.log");
        let profile_home = home.join("profiles").join("coder");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&profile_home).unwrap();

        fs::write(
            &fake_git,
            "#!/bin/sh\n\
case \"$1\" in\n\
  fetch) exit 0 ;;\n\
  rev-parse)\n\
    if [ \"$2\" = \"--abbrev-ref\" ]; then\n\
      printf 'main\\n'\n\
    else\n\
      printf 'deadbeef\\n'\n\
    fi\n\
    exit 0 ;;\n\
  status) exit 0 ;;\n\
  rev-list) printf '1\\n'; exit 0 ;;\n\
  pull) exit 0 ;;\n\
  stash) exit 0 ;;\n\
  ls-files) exit 0 ;;\n\
  diff) exit 0 ;;\n\
  checkout) exit 0 ;;\n\
  reset) exit 0 ;;\n\
esac\n\
exit 0\n",
        )
        .unwrap();
        fs::write(&fake_uv, "#!/bin/sh\nexit 0\n").unwrap();
        fs::write(
            &fake_gateway,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
                log.display()
            ),
        )
        .unwrap();
        for path in [&fake_git, &fake_uv, &fake_gateway] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let mut gateway = Command::new("bash")
            .args(["-c", "exec -a 'hermes gateway run' sleep 30"])
            .spawn()
            .unwrap();
        fs::write(
            profile_home.join("gateway.pid"),
            format!("{{\"pid\":{}}}\n", gateway.id()),
        )
        .unwrap();

        let mut unrelated_gateway = Command::new("bash")
            .args(["-c", "exec -a 'hermes gateway run' sleep 30"])
            .spawn()
            .unwrap();

        let mut dashboard = Command::new("bash")
            .current_dir(project_root())
            .args(["-c", "exec -a 'hermes dashboard' sleep 30"])
            .spawn()
            .unwrap();

        let mut unrelated_dashboard = Command::new("bash")
            .current_dir(temp.path())
            .args(["-c", "exec -a 'hermes dashboard' sleep 30"])
            .spawn()
            .unwrap();

        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_env_var("HERMES_UPDATE_PROJECT_ROOT", &root);
        set_env_var("HERMES_UPDATE_GIT", &fake_git);
        set_env_var("HERMES_UPDATE_UV", &fake_uv);
        set_env_var("HERMES_GATEWAY_BINARY", &fake_gateway);

        print_update(
            &context,
            UpdateArgs {
                gateway: false,
                check: false,
                no_backup: false,
                backup: false,
                yes: true,
            },
        )
        .unwrap();

        let gateway_status = gateway.wait().unwrap();
        let dashboard_status = dashboard.wait().unwrap();
        assert!(!gateway_status.success());
        assert!(!dashboard_status.success());
        assert!(unrelated_gateway.try_wait().unwrap().is_none());
        assert!(unrelated_dashboard.try_wait().unwrap().is_none());
        let _ = unrelated_gateway.kill();
        let _ = unrelated_gateway.wait();
        let _ = unrelated_dashboard.kill();
        let _ = unrelated_dashboard.wait();

        let logged = fs::read_to_string(&log).unwrap();
        assert!(logged.contains("--profile coder gateway run --replace"));

        remove_env_var("HERMES_GATEWAY_BINARY");
        remove_env_var("HERMES_UPDATE_PROJECT_ROOT");
        remove_env_var("HERMES_UPDATE_GIT");
        remove_env_var("HERMES_UPDATE_UV");
    }

    #[test]
    fn managed_marker_implies_nixos() {
        let home = temp_path("managed");
        fs::create_dir_all(&home).unwrap();
        fs::write(home.join(".managed"), "").unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        assert_eq!(get_managed_system(&context).as_deref(), Some("NixOS"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn formats_homebrew_message() {
        let _guard = test_env_lock().lock().unwrap();
        let old = std::env::var("HERMES_MANAGED").ok();
        unsafe { std::env::set_var("HERMES_MANAGED", "homebrew") };
        let message = format_managed_message("Homebrew", "update Hermes Agent");
        assert!(message.contains("Cannot update Hermes Agent"));
        assert!(message.contains("brew upgrade hermes-agent"));
        match old {
            Some(value) => unsafe { std::env::set_var("HERMES_MANAGED", value) },
            None => unsafe { std::env::remove_var("HERMES_MANAGED") },
        }
    }
}
