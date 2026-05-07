use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, ValueEnum};
use hermes_core::{HermesContext, get_auth_status_summary};
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
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
    if should_use_native_tts_setup(context, &args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_tts_setup(context, &mut ui);
    }
    if should_use_native_terminal_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_terminal_setup(context, &mut ui);
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

fn should_use_native_tts_setup(context: &HermesContext, args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Tts))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
        && !nous_auth_present(context)
}

fn should_use_native_terminal_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Terminal))
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
    fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>>;

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

    fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let mut stdout = io::stdout();
        stdout.write_all(prompt.as_bytes())?;
        stdout.flush()?;

        let echo_disabled = if cfg!(unix) && io::stdin().is_terminal() {
            Command::new("stty")
                .arg("-echo")
                .status()
                .ok()
                .is_some_and(|status| status.success())
        } else {
            false
        };

        let mut input = String::new();
        let read = io::stdin().read_line(&mut input)?;

        if echo_disabled {
            let _ = Command::new("stty").arg("echo").status();
            println!();
        }

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

fn run_native_tts_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_provider =
        get_nested_string(&root, &["tts", "provider"]).unwrap_or_else(|| "edge".to_string());

    let providers = [
        ("edge", "Edge TTS"),
        ("elevenlabs", "ElevenLabs"),
        ("openai", "OpenAI TTS"),
        ("xai", "xAI TTS"),
        ("minimax", "MiniMax TTS"),
        ("mistral", "Mistral Voxtral TTS"),
        ("gemini", "Google Gemini TTS"),
        ("neutts", "NeuTTS"),
        ("kittentts", "KittenTTS"),
    ];

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Text-to-Speech")?;
    ui.line(&format!(
        "Current: {}",
        provider_label(&current_provider, &providers)
    ))?;
    ui.blank()?;
    for (index, (_, label)) in providers.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 1, label))?;
    }
    ui.line(&format!(
        "  {}. Keep current ({})",
        providers.len() + 1,
        provider_label(&current_provider, &providers)
    ))?;

    let selection = prompt_menu_choice(
        ui,
        "Select TTS provider: ",
        providers.len() + 1,
        providers.len() + 1,
    )?;
    if selection == providers.len() + 1 {
        ui.line(&format!(
            "Keeping current TTS provider: {}",
            provider_label(&current_provider, &providers)
        ))?;
        return Ok(());
    }

    let mut selected = providers[selection - 1].0.to_string();
    match selected.as_str() {
        "elevenlabs" => {
            if env_value("ELEVENLABS_API_KEY").is_none() {
                let api_key = ui.prompt_secret("ElevenLabs API key: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "ELEVENLABS_API_KEY", api_key.trim())?;
                    ui.line("ElevenLabs API key saved")?;
                }
            }
        }
        "openai" => {
            if env_value("VOICE_TOOLS_OPENAI_KEY").is_none()
                && env_value("OPENAI_API_KEY").is_none()
            {
                let api_key = ui.prompt_secret("OpenAI API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "VOICE_TOOLS_OPENAI_KEY", api_key.trim())?;
                    ui.line("OpenAI TTS API key saved")?;
                }
            }
        }
        "xai" => {
            if env_value("XAI_API_KEY").is_none() {
                let api_key = ui.prompt_secret("xAI API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "XAI_API_KEY", api_key.trim())?;
                    ui.line("xAI TTS API key saved")?;
                }
            }
            if selected == "xai" {
                let voice_id = ui.prompt("xAI voice_id (Enter for default eve): ")?;
                if !voice_id.trim().is_empty() {
                    ensure_mapping(ensure_mapping(&mut root, "tts"), "xai").insert(
                        yaml_key("voice_id"),
                        Value::String(voice_id.trim().to_string()),
                    );
                    ui.line(&format!("xAI voice_id set to: {}", voice_id.trim()))?;
                }
            }
        }
        "minimax" => {
            if env_value("MINIMAX_API_KEY").is_none() {
                let api_key = ui.prompt_secret("MiniMax API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "MINIMAX_API_KEY", api_key.trim())?;
                    ui.line("MiniMax TTS API key saved")?;
                }
            }
        }
        "mistral" => {
            if env_value("MISTRAL_API_KEY").is_none() {
                let api_key = ui.prompt_secret("Mistral API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "MISTRAL_API_KEY", api_key.trim())?;
                    ui.line("Mistral TTS API key saved")?;
                }
            }
        }
        "gemini" => {
            if env_value("GEMINI_API_KEY").is_none() && env_value("GOOGLE_API_KEY").is_none() {
                ui.line("Get a free API key at https://aistudio.google.com/app/apikey")?;
                let api_key = ui.prompt_secret("Gemini API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                } else {
                    save_env_value(context.env_path(), "GEMINI_API_KEY", api_key.trim())?;
                    ui.line("Gemini TTS API key saved")?;
                }
            }
        }
        "neutts" => {
            if !python_module_installed("neutts")? {
                ui.line("NeuTTS requires a Python package and espeak-ng.")?;
                if prompt_yes_no(ui, "Install NeuTTS dependencies now? [Y/n]: ", true)? {
                    if !install_neutts_deps(ui)? {
                        ui.line("NeuTTS installation incomplete. Falling back to Edge TTS.")?;
                        selected = "edge".to_string();
                    }
                } else {
                    ui.line("Skipping install. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                }
            } else {
                ui.line("NeuTTS is already installed")?;
            }
        }
        "kittentts" => {
            if !python_module_installed("kittentts")? {
                ui.line("KittenTTS is lightweight and requires no API key.")?;
                if prompt_yes_no(ui, "Install KittenTTS now? [Y/n]: ", true)? {
                    if !install_kittentts_deps(ui)? {
                        ui.line("KittenTTS installation incomplete. Falling back to Edge TTS.")?;
                        selected = "edge".to_string();
                    }
                } else {
                    ui.line("Skipping install. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                }
            } else {
                ui.line("KittenTTS is already installed")?;
            }
        }
        _ => {}
    }

    ensure_mapping(&mut root, "tts").insert(yaml_key("provider"), Value::String(selected.clone()));
    write_yaml_mapping(&context.config_path(), &root)?;
    ui.line(&format!(
        "TTS provider set to: {}",
        provider_label(&selected, &providers)
    ))?;
    Ok(())
}

fn run_native_terminal_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_backend =
        get_nested_string(&root, &["terminal", "backend"]).unwrap_or_else(|| "local".to_string());

    let mut choices = vec![
        ("local", "Local - run directly on this machine (default)"),
        (
            "docker",
            "Docker - isolated container with configurable resources",
        ),
        ("modal", "Modal - serverless cloud sandbox"),
        ("ssh", "SSH - run on a remote machine"),
        (
            "daytona",
            "Daytona - persistent cloud development environment",
        ),
        (
            "vercel_sandbox",
            "Vercel Sandbox - cloud microVM with snapshot filesystem persistence",
        ),
    ];
    if cfg!(target_os = "linux") {
        choices.push((
            "singularity",
            "Singularity/Apptainer - HPC-friendly container",
        ));
    }

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Terminal Backend")?;
    ui.line("Choose where Hermes runs shell commands and code.")?;
    ui.blank()?;
    for (index, (_, label)) in choices.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 1, label))?;
    }
    ui.line(&format!(
        "  {}. Keep current ({})",
        choices.len() + 1,
        current_backend
    ))?;

    let selected_index = prompt_menu_choice(
        ui,
        "Select terminal backend: ",
        choices.len() + 1,
        choices.len() + 1,
    )?;
    if selected_index == choices.len() + 1 {
        ui.line(&format!("Keeping current backend: {current_backend}"))?;
        return Ok(());
    }

    let selected_backend = choices[selected_index - 1].0;
    if matches!(selected_backend, "modal" | "daytona") {
        ui.line("Falling back to Python setup for this backend.")?;
        return print_setup_python(SetupArgs {
            section: Some(SetupSection::Terminal),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        });
    }

    let terminal = ensure_mapping(&mut root, "terminal");
    terminal.insert(
        yaml_key("backend"),
        Value::String(selected_backend.to_string()),
    );

    match selected_backend {
        "local" => configure_local_terminal(context, ui, terminal)?,
        "docker" => configure_docker_terminal(context, ui, terminal)?,
        "singularity" => configure_singularity_terminal(context, ui, terminal)?,
        "ssh" => configure_ssh_terminal(context, ui)?,
        "vercel_sandbox" => configure_vercel_terminal(ui, terminal)?,
        _ => unreachable!(),
    }

    save_env_value(context.env_path(), "TERMINAL_ENV", selected_backend)?;
    if selected_backend == "vercel_sandbox" {
        if let Some(runtime) = mapping_string(terminal, "vercel_runtime") {
            save_env_value(context.env_path(), "TERMINAL_VERCEL_RUNTIME", &runtime)?;
        }
    }
    write_yaml_mapping(&context.config_path(), &root)?;
    ui.blank()?;
    ui.line(&format!("Terminal backend set to: {selected_backend}"))?;
    Ok(())
}

fn configure_local_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Local")?;
    let current_cwd =
        mapping_string(terminal, "cwd").unwrap_or_else(|| context.home_dir().display().to_string());
    let cwd = ui.prompt(&format!("  Gateway working directory [{}]: ", current_cwd))?;
    if !cwd.trim().is_empty() {
        terminal.insert(yaml_key("cwd"), Value::String(cwd.trim().to_string()));
    }

    if env_value("SUDO_PASSWORD").is_some() {
        ui.line("Sudo password: configured")?;
    } else if prompt_yes_no(
        ui,
        "Enable sudo support? (stores password for apt install, etc.) [y/N]: ",
        false,
    )? {
        let sudo_password = ui.prompt_secret("  Sudo password: ")?;
        if !sudo_password.trim().is_empty() {
            save_env_value(context.env_path(), "SUDO_PASSWORD", sudo_password.trim())?;
            ui.line("Sudo password saved")?;
        }
    }
    Ok(())
}

fn configure_docker_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Docker")?;
    if !command_exists("docker") {
        ui.line("Docker not found in PATH. Install Docker: https://docs.docker.com/get-docker/")?;
    }
    let current_image = mapping_string(terminal, "docker_image")
        .unwrap_or_else(|| "nikolaik/python-nodejs:python3.11-nodejs20".to_string());
    let image = prompt_with_default(ui, "  Docker image", &current_image)?;
    terminal.insert(yaml_key("docker_image"), Value::String(image.clone()));
    save_env_value(context.env_path(), "TERMINAL_DOCKER_IMAGE", &image)?;
    prompt_container_resources(ui, terminal)
}

fn configure_singularity_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Singularity/Apptainer")?;
    if !command_exists("apptainer") && !command_exists("singularity") {
        ui.line(
            "Singularity/Apptainer not found in PATH. Install: https://apptainer.org/docs/admin/main/installation.html",
        )?;
    }
    let current_image = mapping_string(terminal, "singularity_image")
        .unwrap_or_else(|| "docker://nikolaik/python-nodejs:python3.11-nodejs20".to_string());
    let image = prompt_with_default(ui, "  Container image", &current_image)?;
    terminal.insert(yaml_key("singularity_image"), Value::String(image.clone()));
    save_env_value(context.env_path(), "TERMINAL_SINGULARITY_IMAGE", &image)?;
    prompt_container_resources(ui, terminal)
}

fn configure_ssh_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: SSH")?;
    let current_host = env_value("TERMINAL_SSH_HOST").unwrap_or_default();
    let host = prompt_with_default(ui, "  SSH host (hostname or IP)", &current_host)?;
    if !host.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_HOST", host.trim())?;
    }

    let default_user = std::env::var("USER").unwrap_or_default();
    let current_user = env_value("TERMINAL_SSH_USER").unwrap_or_else(|| default_user.clone());
    let user = prompt_with_default(ui, "  SSH user", &current_user)?;
    if !user.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_USER", user.trim())?;
    }

    let current_port = env_value("TERMINAL_SSH_PORT").unwrap_or_else(|| "22".to_string());
    let port = prompt_with_default(ui, "  SSH port", &current_port)?;
    if !port.trim().is_empty() && port.trim() != "22" {
        save_env_value(context.env_path(), "TERMINAL_SSH_PORT", port.trim())?;
    }

    let default_key = context.home_dir().join(".ssh").join("id_rsa");
    let current_key =
        env_value("TERMINAL_SSH_KEY").unwrap_or_else(|| default_key.display().to_string());
    let ssh_key = prompt_with_default(ui, "  SSH private key path", &current_key)?;
    if !ssh_key.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_KEY", ssh_key.trim())?;
    }

    if !host.trim().is_empty() && prompt_yes_no(ui, "  Test SSH connection? [Y/n]: ", true)? {
        let mut ssh = Command::new("ssh");
        ssh.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]);
        if !ssh_key.trim().is_empty() {
            ssh.args(["-i", ssh_key.trim()]);
        }
        if !port.trim().is_empty() && port.trim() != "22" {
            ssh.args(["-p", port.trim()]);
        }
        let destination = if !user.trim().is_empty() {
            format!("{}@{}", user.trim(), host.trim())
        } else {
            host.trim().to_string()
        };
        ssh.arg(destination).arg("echo ok");
        match ssh.output() {
            Ok(output) if output.status.success() => ui.line("  SSH connection successful!")?,
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = stderr.trim();
                if detail.is_empty() {
                    ui.line("  SSH connection failed.")?;
                } else {
                    ui.line(&format!("  SSH connection failed: {detail}"))?;
                }
            }
            Err(error) => ui.line(&format!("  SSH connection failed: {error}"))?,
        }
    }

    Ok(())
}

fn configure_vercel_terminal(
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    const SUPPORTED_RUNTIMES: &[&str] = &["node24", "node22", "python3.13"];

    ui.line("Terminal backend: Vercel Sandbox")?;
    ui.line("Cloud microVM sandboxes with snapshot-backed filesystem persistence.")?;
    ui.line("Requires the optional SDK: pip install 'hermes-agent[vercel]'")?;
    if !python_module_installed("vercel")? {
        ui.line("vercel SDK not detected; install it before using this backend.")?;
    }

    let current_runtime =
        mapping_string(terminal, "vercel_runtime").unwrap_or_else(|| "node24".to_string());
    let runtime = prompt_enum_choice(ui, "  Runtime", SUPPORTED_RUNTIMES, &current_runtime)?;
    terminal.insert(yaml_key("vercel_runtime"), Value::String(runtime.clone()));

    let current_persist = mapping_bool(terminal, "container_persistent").unwrap_or(true);
    let persist_default = if current_persist { "yes" } else { "no" };
    let persist = prompt_yes_no(
        ui,
        &format!("  Persist filesystem with snapshots? (yes/no) [{persist_default}]: "),
        current_persist,
    )?;
    terminal.insert(yaml_key("container_persistent"), Value::Bool(persist));

    let current_cpu = mapping_f64(terminal, "container_cpu").unwrap_or(1.0);
    let cpu = prompt_f64_range(ui, "  CPU cores", current_cpu, 0.1, 512.0)?;
    terminal.insert(yaml_key("container_cpu"), serde_yaml::to_value(cpu)?);

    let current_memory = mapping_i64(terminal, "container_memory").unwrap_or(5120);
    let memory = prompt_positive_i64(
        ui,
        "  Memory in MB (5120 = 5GB)",
        current_memory,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_memory"), serde_yaml::to_value(memory)?);

    terminal.insert(yaml_key("container_disk"), serde_yaml::to_value(51200_i64)?);
    Ok(())
}

fn prompt_container_resources(
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    let current_persist = mapping_bool(terminal, "container_persistent").unwrap_or(true);
    let persist_default = if current_persist { "yes" } else { "no" };
    let persist = prompt_yes_no(
        ui,
        &format!("  Persist filesystem across sessions? (yes/no) [{persist_default}]: "),
        current_persist,
    )?;
    terminal.insert(yaml_key("container_persistent"), Value::Bool(persist));

    let current_cpu = mapping_f64(terminal, "container_cpu").unwrap_or(1.0);
    let cpu = prompt_f64_range(ui, "  CPU cores", current_cpu, 0.1, 512.0)?;
    terminal.insert(yaml_key("container_cpu"), serde_yaml::to_value(cpu)?);

    let current_memory = mapping_i64(terminal, "container_memory").unwrap_or(5120);
    let memory = prompt_positive_i64(
        ui,
        "  Memory in MB (5120 = 5GB)",
        current_memory,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_memory"), serde_yaml::to_value(memory)?);

    let current_disk = mapping_i64(terminal, "container_disk").unwrap_or(51200);
    let disk = prompt_positive_i64(
        ui,
        "  Disk in MB (51200 = 50GB)",
        current_disk,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_disk"), serde_yaml::to_value(disk)?);
    Ok(())
}

fn prompt_with_default(
    ui: &mut dyn SetupUi,
    label: &str,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let input = ui.prompt(&format!("{label} [{current}]: "))?;
    if input.trim().is_empty() {
        return Ok(current.to_string());
    }
    Ok(input.trim().to_string())
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(yaml_key(key))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn mapping_bool(mapping: &Mapping, key: &str) -> Option<bool> {
    mapping.get(yaml_key(key)).and_then(Value::as_bool)
}

fn mapping_i64(mapping: &Mapping, key: &str) -> Option<i64> {
    mapping.get(yaml_key(key)).and_then(Value::as_i64)
}

fn mapping_f64(mapping: &Mapping, key: &str) -> Option<f64> {
    mapping.get(yaml_key(key)).and_then(Value::as_f64)
}

fn provider_label<'a>(provider: &str, providers: &'a [(&str, &'a str)]) -> &'a str {
    providers
        .iter()
        .find_map(|(key, label)| (*key == provider).then_some(*label))
        .unwrap_or("Custom")
}

fn prompt_yes_no(
    ui: &mut dyn SetupUi,
    prompt: &str,
    default_yes: bool,
) -> Result<bool, Box<dyn Error>> {
    loop {
        let input = ui.prompt(prompt)?;
        let normalized = input.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Ok(default_yes);
        }
        match normalized.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => ui.line("Please answer yes or no.")?,
        }
    }
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn nous_auth_present(context: &HermesContext) -> bool {
    get_auth_status_summary(&context.hermes_home(), "nous")
        .map(|status| status.logged_in)
        .unwrap_or(false)
}

fn setup_python_interpreter() -> Option<PathBuf> {
    resolve_repo_python(&project_root(), Some("HERMES_SETUP_PYTHON"))
}

fn python_module_installed(module: &str) -> Result<bool, Box<dyn Error>> {
    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    let status = Command::new(python)
        .arg("-c")
        .arg(format!(
            "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec({module:?}) else 1)"
        ))
        .status()?;
    Ok(status.success())
}

fn install_neutts_deps(ui: &mut dyn SetupUi) -> Result<bool, Box<dyn Error>> {
    if !command_exists("espeak-ng") {
        ui.line("NeuTTS requires espeak-ng for phonemization.")?;
        ui.line(&format!(
            "Install with: {}",
            if cfg!(target_os = "macos") {
                "brew install espeak-ng"
            } else if cfg!(target_os = "windows") {
                "choco install espeak-ng"
            } else {
                "sudo apt install espeak-ng"
            }
        ))?;
        if !prompt_yes_no(ui, "Install espeak-ng now? [Y/n]: ", true)? {
            return Ok(false);
        }
        let mut install = if cfg!(target_os = "macos") {
            let mut cmd = Command::new("brew");
            cmd.args(["install", "espeak-ng"]);
            cmd
        } else if cfg!(target_os = "windows") {
            let mut cmd = Command::new("choco");
            cmd.args(["install", "espeak-ng", "-y"]);
            cmd
        } else {
            let mut cmd = Command::new("sudo");
            cmd.args(["apt", "install", "-y", "espeak-ng"]);
            cmd
        };
        if !install.status().ok().is_some_and(|status| status.success()) {
            return Ok(false);
        }
        ui.line("espeak-ng installed")?;
    }

    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    ui.line("Installing neutts Python package...")?;
    let status = Command::new(python)
        .args(["-m", "pip", "install", "-U", "neutts[all]", "--quiet"])
        .status()?;
    Ok(status.success())
}

fn install_kittentts_deps(ui: &mut dyn SetupUi) -> Result<bool, Box<dyn Error>> {
    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    let wheel_url = "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl";
    ui.line("Installing kittentts Python package...")?;
    let status = Command::new(python)
        .args([
            "-m",
            "pip",
            "install",
            "-U",
            wheel_url,
            "soundfile",
            "--quiet",
        ])
        .status()?;
    Ok(status.success())
}

fn command_exists(command: &str) -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .map(|path| path.join(command))
        .any(|candidate| candidate.is_file())
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

        fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
            self.prompt(prompt)
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
    fn native_tts_setup_writes_provider_and_secret() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "tts:\n  provider: edge\n").unwrap();

        let mut ui = TestUi::new(&["3", "sk-tts-test"]);
        run_native_tts_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: openai"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("VOICE_TOOLS_OPENAI_KEY=sk-tts-test"));
    }

    #[test]
    fn native_terminal_setup_writes_docker_backend() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let mut ui = TestUi::new(&["2", "my-image:latest", "yes", "2", "4096", "20480"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: docker"));
        assert!(saved.contains("docker_image: my-image:latest"));
        assert!(saved.contains("container_persistent: true"));
        assert!(saved.contains("container_cpu: 2.0"));
        assert!(saved.contains("container_memory: 4096"));
        assert!(saved.contains("container_disk: 20480"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=docker"));
        assert!(env_text.contains("TERMINAL_DOCKER_IMAGE=my-image:latest"));
    }

    #[test]
    fn native_terminal_setup_writes_vercel_backend() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let mut ui = TestUi::new(&["6", "node22", "no", "1.5", "2048"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: vercel_sandbox"));
        assert!(saved.contains("vercel_runtime: node22"));
        assert!(saved.contains("container_persistent: false"));
        assert!(saved.contains("container_cpu: 1.5"));
        assert!(saved.contains("container_memory: 2048"));
        assert!(saved.contains("container_disk: 51200"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=vercel_sandbox"));
        assert!(env_text.contains("TERMINAL_VERCEL_RUNTIME=node22"));
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

    #[test]
    #[cfg(unix)]
    fn tts_setup_with_nous_auth_stays_on_python_path() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s\\n' \"$HERMES_SETUP_SECTION\" >> '{}'\n\
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

        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            r#"{"providers":{"nous":{"agent_key":"test-key"}}}"#,
        )
        .unwrap();

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Tts),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=tts"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn terminal_setup_cloud_backend_falls_back_to_python() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s\\n' \"$HERMES_SETUP_SECTION\" >> '{}'\n\
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
        fs::create_dir_all(context.hermes_home()).unwrap();

        let mut ui = TestUi::new(&["3"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=terminal"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }
}
