use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::Args;

use crate::python_bridge::project_root;

#[derive(Args, Debug, Clone)]
pub struct AcpArgs {
    #[arg(long)]
    pub accept_hooks: bool,
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AcpLaunchPlan {
    Binary(PathBuf),
    Cargo {
        cargo: PathBuf,
        project_root: PathBuf,
    },
}

pub fn print_acp(args: AcpArgs) -> Result<(), Box<dyn Error>> {
    let current_exe = env::current_exe()?;
    let path_binaries = which_on_path(binary_name()).into_iter().collect::<Vec<_>>();
    let root = project_root();
    let plan = resolve_acp_launch_plan(
        override_binary_path(),
        &current_exe,
        &path_binaries,
        which_on_path(cargo_name()),
        &root,
    )
    .ok_or_else(|| {
        "could not find a hermes-acp binary; set HERMES_ACP_BINARY or build the Rust ACP binary"
            .to_string()
    })?;

    let mut command = match plan {
        AcpLaunchPlan::Binary(path) => Command::new(path),
        AcpLaunchPlan::Cargo {
            cargo,
            project_root,
        } => {
            let mut command = Command::new(cargo);
            command.current_dir(project_root).args([
                "run",
                "-q",
                "-p",
                "hermes-rs-acp",
                "--bin",
                "hermes-acp",
                "--",
            ]);
            command
        }
    };
    if args.accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }
    command.args(&args.args);
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("acp", status).into())
}

fn resolve_acp_launch_plan(
    override_binary: Option<PathBuf>,
    current_exe: &Path,
    path_binaries: &[PathBuf],
    cargo_binary: Option<PathBuf>,
    project_root: &Path,
) -> Option<AcpLaunchPlan> {
    if let Some(path) = override_binary {
        return Some(AcpLaunchPlan::Binary(path));
    }

    for candidate in sibling_binary_candidates(current_exe) {
        if candidate.is_file() {
            return Some(AcpLaunchPlan::Binary(candidate));
        }
    }
    for candidate in path_binaries {
        if candidate.is_file() {
            return Some(AcpLaunchPlan::Binary(candidate.clone()));
        }
    }

    if project_root.join("Cargo.toml").is_file()
        && let Some(cargo) = cargo_binary
    {
        return Some(AcpLaunchPlan::Cargo {
            cargo,
            project_root: project_root.to_path_buf(),
        });
    }
    None
}

fn override_binary_path() -> Option<PathBuf> {
    let value = env::var_os("HERMES_ACP_BINARY")?;
    let value = value.to_string_lossy();
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

fn sibling_binary_candidates(current_exe: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = current_exe.parent() {
        candidates.push(dir.join(binary_name()));
        if let Some(parent) = dir.parent() {
            candidates.push(parent.join(binary_name()));
        }
    }
    candidates
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn binary_name() -> &'static str {
    #[cfg(windows)]
    {
        "hermes-acp.exe"
    }
    #[cfg(not(windows))]
    {
        "hermes-acp"
    }
}

fn cargo_name() -> &'static str {
    #[cfg(windows)]
    {
        "cargo.exe"
    }
    #[cfg(not(windows))]
    {
        "cargo"
    }
}

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Parser, Debug)]
    struct AcpHarness {
        #[command(flatten)]
        args: AcpArgs,
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        env::temp_dir().join(format!("hermes-rs-acp-{label}-{unique}"))
    }

    #[test]
    fn acp_args_preserve_passthrough_and_accept_hooks() {
        let parsed =
            AcpHarness::try_parse_from(["acp", "--accept-hooks", "status", "--detail", "--deep"])
                .unwrap();
        assert!(parsed.args.accept_hooks);
        assert_eq!(
            parsed.args.args,
            vec![
                "status".to_string(),
                "--detail".to_string(),
                "--deep".to_string()
            ]
        );
    }

    #[test]
    fn resolve_acp_launch_plan_prefers_sibling_binary() {
        let root = temp_path("sibling");
        let exe = root.join("target").join("debug").join("hermes");
        let sibling = exe.parent().unwrap().join(binary_name());
        fs::create_dir_all(sibling.parent().unwrap()).unwrap();
        fs::write(&sibling, b"#!/bin/sh\n").unwrap();

        let plan = resolve_acp_launch_plan(None, &exe, &[], None, Path::new("/missing")).unwrap();
        assert_eq!(plan, AcpLaunchPlan::Binary(sibling));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_acp_launch_plan_falls_back_to_cargo() {
        let root = temp_path("cargo");
        let exe = root.join("target").join("debug").join("hermes");
        let project_root = root.join("repo");
        let cargo = root.join(cargo_name());
        fs::create_dir_all(project_root.as_path()).unwrap();
        fs::write(project_root.join("Cargo.toml"), b"[workspace]\n").unwrap();
        fs::write(&cargo, b"#!/bin/sh\n").unwrap();

        let plan = resolve_acp_launch_plan(None, &exe, &[], Some(cargo.clone()), &project_root);
        assert_eq!(
            plan,
            Some(AcpLaunchPlan::Cargo {
                cargo,
                project_root,
            })
        );

        let _ = fs::remove_dir_all(root);
    }
}
