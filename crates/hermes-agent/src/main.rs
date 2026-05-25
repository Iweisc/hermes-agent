use std::error::Error;
use std::io::{self, Write};

use clap::{Parser, Subcommand};
use hermes_core::{
    DelegateExecutor, EnvLoadReport, HermesContext, LoadedConfig, LoggingMode, ModelOverrides,
    ToolRuntime,
};

mod tui_gateway;

#[derive(Parser, Debug)]
#[command(name = "hermes-agent", version, about = "Hermes agent Rust bootstrap")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Chat {
        prompt: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        base_url: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        api_mode: Option<String>,
        #[arg(long = "toolset")]
        toolsets: Vec<String>,
    },
    Env,
    Status,
    TuiGateway,
}

fn main() -> Result<(), Box<dyn Error>> {
    let detected = HermesContext::detect();
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let profile_override = detected.apply_profile_override(&raw_args)?;
    let context = match profile_override.hermes_home.clone() {
        Some(home) => detected.with_hermes_home_env(Some(home)),
        None => detected,
    };
    unsafe { std::env::set_var("HERMES_HOME", context.hermes_home()) };
    context.ensure_hermes_home()?;
    let env_report = context.load_hermes_dotenv(None)?;
    let config = context.load_config_document()?;
    let _logging = context.setup_logging(&config, LoggingMode::Cli)?;
    let session_store = context.open_session_store()?;
    emit_warnings(&env_report, &config);
    log::info!(
        target: "run_agent",
        "startup profile={} home={}",
        context.current_profile_name(),
        context.hermes_home().display()
    );
    let argv = std::iter::once(String::from("hermes-agent")).chain(profile_override.args);
    let cli = Cli::parse_from(argv);

    match cli.command.unwrap_or(Command::Status) {
        Command::Chat {
            prompt,
            session,
            model,
            provider,
            base_url,
            api_key,
            api_mode,
            toolsets,
        } => {
            let enabled_toolsets = if toolsets.is_empty() {
                config.config.toolsets.clone()
            } else {
                toolsets
            };
            let overrides = ModelOverrides {
                model,
                provider,
                base_url,
                api_key,
                api_mode,
            };
            let delegate = DelegateExecutor::new(
                context.clone(),
                config.clone(),
                "rust-delegate",
                enabled_toolsets.clone(),
                overrides.clone(),
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            );
            let runtime = ToolRuntime::default()
                .with_hermes_home(context.hermes_home())
                .with_clarify_callback(run_clarify_prompt)
                .with_delegate_callback(move |request| delegate.execute(request));
            let result = context.run_chat_completions_turn(
                &config,
                &prompt,
                &runtime,
                Some(&enabled_toolsets),
                &overrides,
                session.as_deref(),
                Some(&session_store),
            )?;
            println!("{}", result.final_response);
        }
        Command::Env => {
            println!("hermes_home={}", context.hermes_home().display());
            println!("config={}", context.config_path().display());
            println!("state_db={}", session_store.path().display());
            println!("active_profile={}", context.active_profile());
            println!("loaded_env_files={}", env_report.loaded_paths.len());
        }
        Command::Status => {
            println!("app=hermes-agent");
            println!("mode=rust-bootstrap");
            println!("current_profile={}", context.current_profile_name());
            println!("hermes_home={}", context.hermes_home().display());
            println!("config={}", config.path.display());
            println!("state_db={}", session_store.path().display());
            println!(
                "session_count={}",
                session_store.session_count().unwrap_or_default()
            );
            println!("loaded_env_files={}", env_report.loaded_paths.len());
            println!("logging_level={}", config.config.logging.level);
            println!("terminal_backend={}", config.config.terminal.backend);
            println!("toolsets={}", config.config.toolsets.join(","));
            println!("note=Rust chat_completions loop available via `hermes-agent chat`");
        }
        Command::TuiGateway => {
            tui_gateway::run(context)?;
        }
    }

    Ok(())
}

fn emit_warnings(env_report: &EnvLoadReport, config: &LoadedConfig) {
    for warning in &env_report.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "run_agent", "{warning}");
    }
    for warning in &config.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "run_agent", "{warning}");
    }
}

fn run_clarify_prompt(question: &str, choices: Option<&[String]>) -> Result<String, String> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "\n[clarify] {question}").map_err(|error| error.to_string())?;
    if let Some(choices) = choices {
        for (index, choice) in choices.iter().enumerate() {
            writeln!(stdout, "{}. {}", index + 1, choice).map_err(|error| error.to_string())?;
        }
        write!(stdout, "Choose 1-{} or type your answer: ", choices.len())
            .map_err(|error| error.to_string())?;
    } else {
        write!(stdout, "Answer: ").map_err(|error| error.to_string())?;
    }
    stdout.flush().map_err(|error| error.to_string())?;

    let mut line = String::new();
    let read = io::stdin()
        .read_line(&mut line)
        .map_err(|error| error.to_string())?;
    if read == 0 {
        return Err("No user input received.".to_string());
    }
    let answer = line.trim();
    if let Some(choices) = choices
        && let Ok(index) = answer.parse::<usize>()
        && (1..=choices.len()).contains(&index)
    {
        return Ok(choices[index - 1].clone());
    }
    if answer.is_empty() {
        return Err("No user input received.".to_string());
    }
    Ok(answer.to_string())
}
