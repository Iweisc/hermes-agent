use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Args;
use hermes_core::HermesContext;

use crate::backup::create_quick_snapshot;
use crate::config_cmd::migrate_config;
use crate::dashboard_cmd::ensure_dashboard_web_ui;
use crate::python_bridge::{project_root, resolve_repo_python};

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
    if should_use_python_update_bridge(&args) {
        return print_update_apply_python(args);
    }
    print_update_apply_native(context, args)
}

fn should_use_python_update_bridge(args: &UpdateArgs) -> bool {
    args.gateway || args.backup || args.no_backup
}

fn print_update_apply_python(args: UpdateArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_UPDATE_PYTHON"))
        .ok_or("could not find a Python interpreter for update")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env(
            "HERMES_UPDATE_GATEWAY",
            if args.gateway { "1" } else { "0" },
        )
        .env(
            "HERMES_UPDATE_NO_BACKUP",
            if args.no_backup { "1" } else { "0" },
        )
        .env("HERMES_UPDATE_BACKUP", if args.backup { "1" } else { "0" })
        .env("HERMES_UPDATE_YES", if args.yes { "1" } else { "0" })
        .arg("-c")
        .arg(UPDATE_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("update", status).into())
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
        && io::stdin().is_terminal()
        && io::stdout().is_terminal();

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
            let _ = restore_stashed_changes(&root, &git_base, stash_ref, prompt_for_restore)?;
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
        let _ = restore_stashed_changes(&root, &git_base, stash_ref, prompt_for_restore)?;
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

    println!();
    println!("→ Checking configuration for new options...");
    migrate_config(context)?;

    println!();
    println!("✓ Update complete!");
    println!(
        "  Remaining Python-only update behavior: gateway restart flow, full pre-update backup flags, and bundled skill/profile sync."
    );
    println!("  Restart running gateways or dashboards manually if needed.");
    println!("    hermes gateway restart");
    println!("    hermes dashboard --port <port>");

    Ok(())
}

const UPDATE_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "from hermes_cli.main import _cmd_update_impl, _finalize_update_output, _install_hangup_protection\n",
    "gateway_mode = (os.environ.get('HERMES_UPDATE_GATEWAY') == '1')\n",
    "args = argparse.Namespace(\n",
    "    gateway=gateway_mode,\n",
    "    check=False,\n",
    "    no_backup=(os.environ.get('HERMES_UPDATE_NO_BACKUP') == '1'),\n",
    "    backup=(os.environ.get('HERMES_UPDATE_BACKUP') == '1'),\n",
    "    yes=(os.environ.get('HERMES_UPDATE_YES') == '1'),\n",
    ")\n",
    "state = _install_hangup_protection(gateway_mode=gateway_mode)\n",
    "try:\n",
    "    _cmd_update_impl(args, gateway_mode=gateway_mode)\n",
    "finally:\n",
    "    _finalize_update_output(state)\n",
);

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
    repo_dir: &Path,
    git_base: &[String],
    stash_ref: &str,
    prompt_user: bool,
) -> Result<bool, Box<dyn Error>> {
    if prompt_user {
        println!();
        println!("⚠ Local changes were stashed before updating.");
        println!("  Restoring them may reapply local customizations onto the updated codebase.");
        println!("  Review the result afterward if Hermes behaves unexpectedly.");
        print!("Restore local changes now? [Y/n]: ");
        io::stdout().flush()?;
        let mut response = String::new();
        io::stdin().read_line(&mut response)?;
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
    fn gateway_update_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        let context =
            HermesContext::new("/tmp").with_hermes_home_env(Some(temp.path().join(".hermes")));
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'gateway=%s no_backup=%s backup=%s yes=%s\\n' \\\n\
    \"$HERMES_UPDATE_GATEWAY\" \"$HERMES_UPDATE_NO_BACKUP\" \"$HERMES_UPDATE_BACKUP\" \"$HERMES_UPDATE_YES\" >> '{}'\n\
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

        set_env_var("HERMES_UPDATE_PYTHON", &fake_python);
        print_update(
            &context,
            UpdateArgs {
                gateway: true,
                check: false,
                no_backup: true,
                backup: true,
                yes: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("gateway=1 no_backup=1 backup=1 yes=1"));

        remove_env_var("HERMES_UPDATE_PYTHON");
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
