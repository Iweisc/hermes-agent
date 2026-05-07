use std::error::Error;
use std::process::{Command, ExitStatus};

use crate::python_bridge::{project_root, resolve_repo_python};

pub fn print_whatsapp() -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_WHATSAPP_PYTHON"))
        .ok_or("could not find a Python interpreter for whatsapp setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-c")
        .arg(WHATSAPP_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("whatsapp", status).into())
}

const WHATSAPP_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "from hermes_cli.main import cmd_whatsapp\n",
    "cmd_whatsapp(argparse.Namespace())\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
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

    #[test]
    fn whatsapp_command_name_is_stable() {
        let result = print_whatsapp as fn() -> Result<(), Box<dyn Error>>;
        let _ = result;
    }

    #[test]
    #[cfg(unix)]
    fn whatsapp_uses_python_override() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'whatsapp bootstrap\\n' >> '{}'\n\
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

        set_env_var("HERMES_WHATSAPP_PYTHON", &fake_python);
        print_whatsapp().unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("whatsapp bootstrap"));

        remove_env_var("HERMES_WHATSAPP_PYTHON");
    }
}
