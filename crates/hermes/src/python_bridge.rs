use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

pub fn launch_python_main_command(
    command_name: &str,
    args: &[String],
    override_env_var: Option<&str>,
    extra_env: &[(String, String)],
) -> Result<(), Box<dyn Error>> {
    let project_root = project_root();
    let python = resolve_repo_python(&project_root, override_env_var)
        .ok_or("could not find a Python interpreter for compatibility launch")?;
    let mut command = Command::new(python);
    command
        .current_dir(&project_root)
        .env("PYTHONPATH", project_root.display().to_string())
        .arg("-m")
        .arg("hermes_cli.main")
        .arg(command_name);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.args(args);
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message(command_name, status).into())
}

pub fn resolve_repo_python(project_root: &Path, override_env_var: Option<&str>) -> Option<PathBuf> {
    if let Some(env_var) = override_env_var {
        if let Some(value) = std::env::var(env_var).ok() {
            if !value.trim().is_empty() {
                return Some(PathBuf::from(value.trim()));
            }
        }
    }

    let candidates = [
        project_root.join(".venv").join(python_bin_name()),
        project_root.join("venv").join(python_bin_name()),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
            .join("hermes-agent")
            .join("venv")
            .join(python_bin_name()),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    which_on_path("python3").or_else(|| which_on_path("python"))
}

pub fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
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

fn python_bin_name() -> &'static str {
    #[cfg(windows)]
    {
        "Scripts/python.exe"
    }
    #[cfg(not(windows))]
    {
        "bin/python"
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
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-pybridge-{label}-{unique}"))
    }

    #[test]
    fn project_root_contains_python_cli() {
        let root = project_root();
        assert!(root.join("hermes_cli").join("main.py").exists());
    }

    #[test]
    fn resolve_repo_python_prefers_local_venv() {
        let root = temp_path("venv");
        let python = root.join(".venv").join(python_bin_name());
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(&python, b"").unwrap();
        let resolved = resolve_repo_python(&root, None).unwrap();
        assert_eq!(resolved, python);
        let _ = fs::remove_dir_all(root);
    }
}
