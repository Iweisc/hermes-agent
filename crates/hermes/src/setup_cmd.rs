use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::process::{Command, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, ValueEnum};
use hermes_core::HermesContext;
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
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

pub fn print_setup(context: &HermesContext, args: SetupArgs) -> Result<(), Box<dyn Error>> {
    if should_use_native_agent_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_agent_setup(context, &mut ui);
    }
    print_setup_python(args)
}

fn print_setup_python(args: SetupArgs) -> Result<(), Box<dyn Error>> {
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

fn should_use_native_agent_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Agent))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
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

trait SetupUi {
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>>;
    fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>>;

    fn blank(&mut self) -> Result<(), Box<dyn Error>> {
        self.line("")
    }
}

struct TerminalUi;

impl SetupUi for TerminalUi {
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
        println!("{text}");
        Ok(())
    }

    fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let mut stdout = io::stdout();
        stdout.write_all(prompt.as_bytes())?;
        stdout.flush()?;
        let mut input = String::new();
        let read = io::stdin().read_line(&mut input)?;
        if read == 0 {
            return Err("setup cancelled".into());
        }
        Ok(input.trim().to_string())
    }
}

fn run_native_agent_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_max = get_nested_i64(&root, &["agent", "max_turns"]).unwrap_or(90);
    let current_mode = get_nested_string(&root, &["display", "tool_progress"])
        .unwrap_or_else(|| "all".to_string());
    let current_threshold = get_nested_f64(&root, &["compression", "threshold"]).unwrap_or(0.50);
    let current_reset_mode =
        get_nested_string(&root, &["session_reset", "mode"]).unwrap_or_else(|| "both".to_string());
    let current_idle = get_nested_i64(&root, &["session_reset", "idle_minutes"]).unwrap_or(1440);
    let current_hour = get_nested_i64(&root, &["session_reset", "at_hour"]).unwrap_or(4);

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Agent Settings")?;
    ui.line("Configure max turns, tool progress, compression, and session resets.")?;
    ui.blank()?;

    let max_turns = prompt_positive_i64(
        ui,
        "Max iterations",
        current_max,
        "Enter a positive integer.",
    )?;
    ensure_mapping(&mut root, "agent")
        .insert(yaml_key("max_turns"), serde_yaml::to_value(max_turns)?);
    root.remove(yaml_key("max_turns"));
    let _ = remove_env_key(&context.env_path(), "HERMES_MAX_ITERATIONS")?;
    ui.line(&format!("Max iterations set to {max_turns}"))?;

    ui.blank()?;
    ui.line("Tool progress mode: off, new, all, verbose")?;
    let tool_progress = prompt_enum_choice(
        ui,
        "Tool progress mode",
        &["off", "new", "all", "verbose"],
        &current_mode,
    )?;
    ensure_mapping(&mut root, "display").insert(
        yaml_key("tool_progress"),
        Value::String(tool_progress.clone()),
    );
    ui.line(&format!("Tool progress set to: {tool_progress}"))?;

    ui.blank()?;
    let compression_threshold = prompt_f64_range(
        ui,
        "Compression threshold (0.5-0.95)",
        current_threshold,
        0.5,
        0.95,
    )?;
    let compression = ensure_mapping(&mut root, "compression");
    compression.insert(yaml_key("enabled"), Value::Bool(true));
    compression.insert(
        yaml_key("threshold"),
        serde_yaml::to_value(compression_threshold)?,
    );
    ui.line(&format!(
        "Context compression threshold set to {compression_threshold}"
    ))?;

    ui.blank()?;
    ui.line("Session reset mode:")?;
    ui.line("  1. Inactivity + daily reset")?;
    ui.line("  2. Inactivity only")?;
    ui.line("  3. Daily only")?;
    ui.line("  4. Never auto-reset")?;
    let default_reset = match current_reset_mode.as_str() {
        "idle" => 2,
        "daily" => 3,
        "none" => 4,
        _ => 1,
    };
    let reset_choice = prompt_menu_choice(ui, "Select [1]: ", 4, default_reset)?;
    let session_reset = ensure_mapping(&mut root, "session_reset");
    match reset_choice {
        1 => {
            let idle = prompt_positive_i64(
                ui,
                "  Inactivity timeout (minutes)",
                current_idle,
                "Enter a positive integer.",
            )?;
            let hour = prompt_i64_range(
                ui,
                "  Daily reset hour (0-23, local time)",
                current_hour,
                0,
                23,
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("both".to_string()));
            session_reset.insert(yaml_key("idle_minutes"), serde_yaml::to_value(idle)?);
            session_reset.insert(yaml_key("at_hour"), serde_yaml::to_value(hour)?);
            ui.line(&format!(
                "Sessions reset after {idle} min idle or daily at {hour}:00"
            ))?;
        }
        2 => {
            let idle = prompt_positive_i64(
                ui,
                "  Inactivity timeout (minutes)",
                current_idle,
                "Enter a positive integer.",
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("idle".to_string()));
            session_reset.insert(yaml_key("idle_minutes"), serde_yaml::to_value(idle)?);
            ui.line(&format!("Sessions reset after {idle} min of inactivity"))?;
        }
        3 => {
            let hour = prompt_i64_range(
                ui,
                "  Daily reset hour (0-23, local time)",
                current_hour,
                0,
                23,
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("daily".to_string()));
            session_reset.insert(yaml_key("at_hour"), serde_yaml::to_value(hour)?);
            ui.line(&format!("Sessions reset daily at {hour}:00"))?;
        }
        4 => {
            session_reset.insert(yaml_key("mode"), Value::String("none".to_string()));
            ui.line("Sessions will never auto-reset. Use /reset manually when needed.")?;
        }
        _ => unreachable!(),
    }

    write_yaml_mapping(&context.config_path(), &root)?;
    ui.blank()?;
    ui.line("Agent Settings configuration complete!")?;
    Ok(())
}

fn prompt_positive_i64(
    ui: &mut dyn SetupUi,
    label: &str,
    current: i64,
    error_message: &str,
) -> Result<i64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<i64>() {
            Ok(value) if value > 0 => return Ok(value),
            _ => ui.line(error_message)?,
        }
    }
}

fn prompt_i64_range(
    ui: &mut dyn SetupUi,
    label: &str,
    current: i64,
    min: i64,
    max: i64,
) -> Result<i64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<i64>() {
            Ok(value) if value >= min && value <= max => return Ok(value),
            _ => ui.line(&format!("Enter a whole number between {min} and {max}."))?,
        }
    }
}

fn prompt_f64_range(
    ui: &mut dyn SetupUi,
    label: &str,
    current: f64,
    min: f64,
    max: f64,
) -> Result<f64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<f64>() {
            Ok(value) if value >= min && value <= max => return Ok(value),
            _ => ui.line(&format!("Enter a number between {min} and {max}."))?,
        }
    }
}

fn prompt_enum_choice(
    ui: &mut dyn SetupUi,
    label: &str,
    options: &[&str],
    current: &str,
) -> Result<String, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current.to_string());
        }
        let normalized = input.to_ascii_lowercase();
        if options.iter().any(|option| *option == normalized) {
            return Ok(normalized);
        }
        ui.line(&format!("Choose one of: {}", options.join(", ")))?;
    }
}

fn prompt_menu_choice(
    ui: &mut dyn SetupUi,
    prompt: &str,
    max: usize,
    default: usize,
) -> Result<usize, Box<dyn Error>> {
    loop {
        let input = ui.prompt(prompt)?;
        if input.is_empty() {
            return Ok(default);
        }
        match input.parse::<usize>() {
            Ok(value) if (1..=max).contains(&value) => return Ok(value),
            _ => ui.line(&format!("Enter a number between 1 and {max}."))?,
        }
    }
}

fn get_nested_i64(root: &Mapping, path: &[&str]) -> Option<i64> {
    get_nested_value(root, path).and_then(Value::as_i64)
}

fn get_nested_f64(root: &Mapping, path: &[&str]) -> Option<f64> {
    get_nested_value(root, path).and_then(Value::as_f64)
}

fn get_nested_string(root: &Mapping, path: &[&str]) -> Option<String> {
    get_nested_value(root, path)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn get_nested_value<'a>(root: &'a Mapping, path: &[&str]) -> Option<&'a Value> {
    let (last, parents) = path.split_last()?;
    let mut current = root;
    for segment in parents {
        current = current.get(yaml_key(segment))?.as_mapping()?;
    }
    current.get(yaml_key(last))
}

fn ensure_mapping<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let key_value = yaml_key(key).clone();
    let replace = !matches!(root.get(&key_value), Some(Value::Mapping(_)));
    if replace {
        root.insert(key_value.clone(), Value::Mapping(Mapping::new()));
    }
    match root.get_mut(&key_value) {
        Some(Value::Mapping(mapping)) => mapping,
        _ => unreachable!(),
    }
}

fn yaml_key(key: &str) -> Value {
    Value::String(key.to_string())
}

fn remove_env_key(path: &Path, key: &str) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    let original = fs::read_to_string(path)?;
    let mut kept = Vec::new();
    let mut removed = false;
    for line in original.lines() {
        if line
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
        {
            removed = true;
            continue;
        }
        kept.push(format!("{line}\n"));
    }
    if !removed {
        return Ok(false);
    }
    atomic_write(path, kept.concat().as_bytes())?;
    Ok(true)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp-{unique}"));
    let tmp = std::path::PathBuf::from(tmp);
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
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

    #[derive(Default)]
    struct TestUi {
        answers: Vec<String>,
        cursor: usize,
        output: String,
    }

    impl TestUi {
        fn new(answers: &[&str]) -> Self {
            Self {
                answers: answers.iter().map(|value| (*value).to_string()).collect(),
                cursor: 0,
                output: String::new(),
            }
        }
    }

    impl SetupUi for TestUi {
        fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
            self.output.push_str(text);
            self.output.push('\n');
            Ok(())
        }

        fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
            self.output.push_str(prompt);
            let answer = self
                .answers
                .get(self.cursor)
                .cloned()
                .ok_or("missing test answer")?;
            self.cursor += 1;
            Ok(answer)
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
        let context = HermesContext::new(temp.path());
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Gateway),
                non_interactive: true,
                reset: true,
                reconfigure: true,
                quick: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=gateway non_interactive=1 reset=1 reconfigure=1 quick=1"));

        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    fn native_agent_setup_updates_config_and_removes_legacy_env() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(
            context.config_path(),
            "agent:\n  max_turns: 90\ndisplay:\n  tool_progress: all\ncompression:\n  threshold: 0.6\nsession_reset:\n  mode: both\n  idle_minutes: 120\n  at_hour: 6\nmax_turns: 12\n",
        )
        .unwrap();
        fs::write(
            context.env_path(),
            "OPENAI_API_KEY=test\nHERMES_MAX_ITERATIONS=77\n",
        )
        .unwrap();

        let mut ui = TestUi::new(&["120", "verbose", "0.75", "2", "300"]);
        run_native_agent_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("max_turns: 120"));
        assert!(saved.contains("tool_progress: verbose"));
        assert!(saved.contains("enabled: true"));
        assert!(saved.contains("threshold: 0.75"));
        assert!(saved.contains("mode: idle"));
        assert!(saved.contains("idle_minutes: 300"));
        assert!(!saved.contains("\nmax_turns: 12\n"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("OPENAI_API_KEY=test"));
        assert!(!env_text.contains("HERMES_MAX_ITERATIONS"));
    }

    #[test]
    fn agent_setup_with_extra_flags_stays_on_python_path() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s quick=%s\\n' \"$HERMES_SETUP_SECTION\" \"$HERMES_SETUP_QUICK\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        #[cfg(unix)]
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        let context = HermesContext::new(temp.path());
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Agent),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=agent quick=1"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }
}
