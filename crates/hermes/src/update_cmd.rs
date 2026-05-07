use std::error::Error;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

use clap::Args;
use hermes_core::HermesContext;

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

    print_update_apply(args)
}

fn print_update_apply(args: UpdateArgs) -> Result<(), Box<dyn Error>> {
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

const UPDATE_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "from hermes_cli.main import cmd_update\n",
    "cmd_update(argparse.Namespace(\n",
    "    gateway=(os.environ.get('HERMES_UPDATE_GATEWAY') == '1'),\n",
    "    check=False,\n",
    "    no_backup=(os.environ.get('HERMES_UPDATE_NO_BACKUP') == '1'),\n",
    "    backup=(os.environ.get('HERMES_UPDATE_BACKUP') == '1'),\n",
    "    yes=(os.environ.get('HERMES_UPDATE_YES') == '1'),\n",
    "))\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn run_update_check() -> Result<(), Box<dyn Error>> {
    let repo_dir = project_root();
    if !repo_dir.join(".git").exists() {
        return Err("Not a git repository — cannot check for updates.".into());
    }

    let mut git_base = vec![String::from("git")];
    if cfg!(windows) {
        git_base.extend([
            String::from("-c"),
            String::from("windows.appendAtomically=false"),
        ]);
    }

    println!("→ Fetching from upstream...");
    let upstream = run_git(&repo_dir, &git_base, ["fetch", "upstream"])?;
    let compare_branch = if upstream.status.success() {
        "upstream/main"
    } else {
        println!("→ Fetching from origin...");
        let origin = run_git(&repo_dir, &git_base, ["fetch", "origin"])?;
        if !origin.status.success() {
            return Err(map_fetch_error(&origin.stderr).into());
        }
        "origin/main"
    };

    let rev = run_git(
        &repo_dir,
        &git_base,
        ["rev-list", &format!("HEAD..{compare_branch}"), "--count"],
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

fn run_git<const N: usize>(
    repo_dir: &PathBuf,
    git_base: &[String],
    args: [&str; N],
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
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::TempDir;

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
        std::env::temp_dir().join(format!("hermes-rs-update-{label}-{unique}"))
    }

    #[test]
    #[cfg(unix)]
    fn update_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
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
        print_update_apply(UpdateArgs {
            gateway: true,
            check: false,
            no_backup: true,
            backup: true,
            yes: true,
        })
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("gateway=1 no_backup=1 backup=1 yes=1"));

        remove_env_var("HERMES_UPDATE_PYTHON");
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
