use std::error::Error;
use std::process::{Command, ExitStatus};

use clap::{Args, ValueEnum};

use crate::python_bridge::{project_root, resolve_repo_python};

#[derive(Args, Debug, Clone)]
pub struct SetupArgs {
    #[arg(value_enum)]
    pub section: Option<SetupSection>,
    #[arg(long = "non-interactive", default_value_t = false)]
    pub non_interactive: bool,
    #[arg(long, default_value_t = false)]
    pub reset: bool,
    #[arg(long, default_value_t = false)]
    pub reconfigure: bool,
    #[arg(long, default_value_t = false)]
    pub quick: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum SetupSection {
    #[value(name = "model")]
    Model,
    #[value(name = "tts")]
    Tts,
    #[value(name = "terminal")]
    Terminal,
    #[value(name = "gateway")]
    Gateway,
    #[value(name = "tools")]
    Tools,
    #[value(name = "agent")]
    Agent,
}

pub fn print_setup(args: SetupArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_SETUP_PYTHON"))
        .ok_or("could not find a Python interpreter for setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env(
            "HERMES_SETUP_NON_INTERACTIVE",
            if args.non_interactive { "1" } else { "0" },
        )
        .env("HERMES_SETUP_RESET", if args.reset { "1" } else { "0" })
        .env(
            "HERMES_SETUP_RECONFIGURE",
            if args.reconfigure { "1" } else { "0" },
        )
        .env("HERMES_SETUP_QUICK", if args.quick { "1" } else { "0" });
    if let Some(section) = args.section {
        command.env("HERMES_SETUP_SECTION", section.as_str());
    }
    command.arg("-c").arg(SETUP_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("setup", status).into())
}

impl SetupSection {
    fn as_str(self) -> &'static str {
        match self {
            SetupSection::Model => "model",
            SetupSection::Tts => "tts",
            SetupSection::Terminal => "terminal",
            SetupSection::Gateway => "gateway",
            SetupSection::Tools => "tools",
            SetupSection::Agent => "agent",
        }
    }
}

const SETUP_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "from hermes_cli.setup import run_setup_wizard\n",
    "args = argparse.Namespace(\n",
    "    section=(os.environ.get('HERMES_SETUP_SECTION') or None),\n",
    "    non_interactive=(os.environ.get('HERMES_SETUP_NON_INTERACTIVE') == '1'),\n",
    "    reset=(os.environ.get('HERMES_SETUP_RESET') == '1'),\n",
    "    reconfigure=(os.environ.get('HERMES_SETUP_RECONFIGURE') == '1'),\n",
    "    quick=(os.environ.get('HERMES_SETUP_QUICK') == '1'),\n",
    ")\n",
    "run_setup_wizard(args)\n",
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
    use clap::Parser;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    #[derive(Parser, Debug)]
    struct SetupHarness {
        #[command(flatten)]
        args: SetupArgs,
    }

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
    fn setup_args_preserve_section_and_flags() {
        let parsed = SetupHarness::try_parse_from([
            "setup",
            "gateway",
            "--non-interactive",
            "--reset",
            "--reconfigure",
            "--quick",
        ])
        .unwrap();
        assert_eq!(parsed.args.section, Some(SetupSection::Gateway));
        assert!(parsed.args.non_interactive);
        assert!(parsed.args.reset);
        assert!(parsed.args.reconfigure);
        assert!(parsed.args.quick);
    }

    #[test]
    fn setup_section_names_are_stable() {
        assert_eq!(SetupSection::Model.as_str(), "model");
        assert_eq!(SetupSection::Tts.as_str(), "tts");
        assert_eq!(SetupSection::Terminal.as_str(), "terminal");
        assert_eq!(SetupSection::Gateway.as_str(), "gateway");
        assert_eq!(SetupSection::Tools.as_str(), "tools");
        assert_eq!(SetupSection::Agent.as_str(), "agent");
    }

    #[test]
    #[cfg(unix)]
    fn setup_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s non_interactive=%s reset=%s reconfigure=%s quick=%s\\n' \\\n\
    \"$HERMES_SETUP_SECTION\" \"$HERMES_SETUP_NON_INTERACTIVE\" \"$HERMES_SETUP_RESET\" \\\n\
    \"$HERMES_SETUP_RECONFIGURE\" \"$HERMES_SETUP_QUICK\" >> '{}'\n\
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

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        print_setup(SetupArgs {
            section: Some(SetupSection::Gateway),
            non_interactive: true,
            reset: true,
            reconfigure: true,
            quick: true,
        })
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=gateway non_interactive=1 reset=1 reconfigure=1 quick=1"));

        remove_env_var("HERMES_SETUP_PYTHON");
    }
}
