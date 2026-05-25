use std::error::Error;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use hermes_core::{
    ApprovalRequest, DelegateExecutor, EnvLoadReport, GatewayEventBridge, HermesContext,
    LoadedConfig, LoggingMode, ModelOverrides, StepUpdate, ToolProgressUpdate, ToolRuntime,
};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};

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
        #[arg(long, default_value_t = false)]
        json_events: bool,
        #[arg(long, default_value_t = false)]
        gateway_events: bool,
    },
    Env,
    Status,
}

#[derive(Clone)]
struct ChatEventEmitter {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl ChatEventEmitter {
    fn stdout() -> Self {
        Self {
            writer: Arc::new(Mutex::new(Box::new(io::stdout()))),
        }
    }

    #[cfg(test)]
    fn from_writer(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
        }
    }

    fn emit<T: Serialize>(&self, event: &T) {
        let Ok(mut writer) = self.writer.lock() else {
            return;
        };
        let _ = serde_json::to_writer(&mut **writer, event);
        let _ = writer.write_all(b"\n");
        let _ = writer.flush();
    }
}

fn emit_tool_progress_event(emitter: &ChatEventEmitter, update: &ToolProgressUpdate) {
    let event = json!({
        "event": "tool_progress",
        "event_type": update.event_type,
        "function_name": update.function_name,
        "preview": update.preview,
        "function_args": update.function_args,
        "duration_ms": update.duration_ms,
        "is_error": update.is_error,
    });
    emitter.emit(&event);
}

fn emit_step_event(emitter: &ChatEventEmitter, update: &StepUpdate) {
    let event = json!({
        "event": "step",
        "iteration": update.iteration,
        "prev_tools": update.prev_tools,
    });
    emitter.emit(&event);
}

fn emit_clarify_event(emitter: &ChatEventEmitter, question: &str, choices: Option<&[String]>) {
    let event = json!({
        "event": "clarify_request",
        "question": question,
        "choices": choices,
    });
    emitter.emit(&event);
}

fn emit_approval_event(emitter: &ChatEventEmitter, request: &ApprovalRequest) {
    let event = json!({
        "event": "approval_request",
        "command": request.command,
        "description": request.description,
        "pattern_keys": request.pattern_keys,
        "choices": request.choices,
        "allow_permanent": request.allow_permanent,
    });
    emitter.emit(&event);
}

fn emit_gateway_events(emitter: &ChatEventEmitter, events: &[hermes_core::GatewayEventEnvelope]) {
    for event in events {
        emitter.emit(event);
    }
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
            json_events,
            gateway_events,
        } => {
            if json_events && gateway_events {
                return Err("--json-events and --gateway-events cannot be used together".into());
            }
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
            let structured_events = json_events || gateway_events;
            let event_emitter = structured_events.then(ChatEventEmitter::stdout);
            let gateway_bridge =
                gateway_events.then(|| Arc::new(Mutex::new(GatewayEventBridge::default())));
            let mut runtime = ToolRuntime::default()
                .with_hermes_home(context.hermes_home())
                .with_delegate_callback(move |request| delegate.execute(request));
            if let Some(emitter) = event_emitter.clone() {
                if let Some(bridge) = gateway_bridge.clone() {
                    emitter.emit(&bridge.lock().unwrap().gateway_ready());
                    emitter.emit(&bridge.lock().unwrap().message_start());
                    runtime = runtime
                        .with_tool_progress_callback({
                            let bridge = Arc::clone(&bridge);
                            let emitter = emitter.clone();
                            move |update| {
                                let events = bridge.lock().unwrap().on_tool_progress(update);
                                emit_gateway_events(&emitter, &events);
                            }
                        })
                        .with_step_callback({
                            let bridge = Arc::clone(&bridge);
                            let emitter = emitter.clone();
                            move |update| {
                                let events = bridge.lock().unwrap().on_step(update);
                                emit_gateway_events(&emitter, &events);
                            }
                        })
                        .with_clarify_request_callback({
                            let bridge = Arc::clone(&bridge);
                            let emitter = emitter.clone();
                            move |request| {
                                emitter.emit(&bridge.lock().unwrap().on_clarify_request(request))
                            }
                        })
                        .with_approval_request_callback({
                            let bridge = Arc::clone(&bridge);
                            let emitter = emitter.clone();
                            move |request| {
                                emitter.emit(&bridge.lock().unwrap().on_approval_request(request))
                            }
                        });
                } else {
                    runtime = runtime
                        .with_tool_progress_callback({
                            let emitter = emitter.clone();
                            move |update| emit_tool_progress_event(&emitter, update)
                        })
                        .with_step_callback({
                            let emitter = emitter.clone();
                            move |update| emit_step_event(&emitter, update)
                        })
                        .with_clarify_request_callback({
                            let emitter = emitter.clone();
                            move |request| {
                                emit_clarify_event(
                                    &emitter,
                                    &request.question,
                                    request.choices.as_deref(),
                                )
                            }
                        })
                        .with_approval_request_callback({
                            let emitter = emitter.clone();
                            move |request| emit_approval_event(&emitter, request)
                        });
                }
                runtime = runtime
                    .with_clarify_callback(run_clarify_prompt_stderr)
                    .with_approval_callback(run_approval_prompt_stderr);
            } else {
                runtime = runtime
                    .with_clarify_callback(run_clarify_prompt)
                    .with_approval_callback(run_approval_prompt);
            }
            let result = context.run_chat_completions_turn(
                &config,
                &prompt,
                &runtime,
                Some(&enabled_toolsets),
                &overrides,
                session.as_deref(),
                Some(&session_store),
            )?;
            if let Some(emitter) = event_emitter {
                if let Some(bridge) = gateway_bridge {
                    let event = bridge.lock().unwrap().on_final_response(&result);
                    emitter.emit(&event);
                } else {
                    let event = json!({
                        "event": "final_response",
                        "text": result.final_response,
                        "reasoning": result.reasoning,
                        "api_calls": result.api_calls,
                        "tool_calls": result.tool_calls,
                        "model": result.model,
                        "provider": result.provider,
                        "base_url": result.base_url,
                        "session_id": result.session_id,
                    });
                    emitter.emit(&event);
                }
            } else {
                println!("{}", result.final_response);
            }
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
    run_clarify_prompt_with_writer(&mut stdout, question, choices)
}

fn run_clarify_prompt_stderr(question: &str, choices: Option<&[String]>) -> Result<String, String> {
    let mut stderr = io::stderr().lock();
    run_clarify_prompt_with_writer(&mut stderr, question, choices)
}

fn run_clarify_prompt_with_writer<W: Write>(
    writer: &mut W,
    question: &str,
    choices: Option<&[String]>,
) -> Result<String, String> {
    writeln!(writer, "\n[clarify] {question}").map_err(|error| error.to_string())?;
    if let Some(choices) = choices {
        for (index, choice) in choices.iter().enumerate() {
            writeln!(writer, "{}. {}", index + 1, choice).map_err(|error| error.to_string())?;
        }
        write!(writer, "Choose 1-{} or type your answer: ", choices.len())
            .map_err(|error| error.to_string())?;
    } else {
        write!(writer, "Answer: ").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    read_clarify_answer(choices)
}

fn run_approval_prompt(request: &ApprovalRequest) -> Result<String, String> {
    let mut stdout = io::stdout().lock();
    run_approval_prompt_with_writer(&mut stdout, request)
}

fn run_approval_prompt_stderr(request: &ApprovalRequest) -> Result<String, String> {
    let mut stderr = io::stderr().lock();
    run_approval_prompt_with_writer(&mut stderr, request)
}

fn run_approval_prompt_with_writer<W: Write>(
    writer: &mut W,
    request: &ApprovalRequest,
) -> Result<String, String> {
    writeln!(writer, "\n[approval] {}", request.description).map_err(|error| error.to_string())?;
    writeln!(writer, "{}", request.command).map_err(|error| error.to_string())?;
    writeln!(writer).map_err(|error| error.to_string())?;
    for (index, choice) in request.choices.iter().enumerate() {
        writeln!(
            writer,
            "{}. {}",
            index + 1,
            approval_choice_label(choice, request.allow_permanent)
        )
        .map_err(|error| error.to_string())?;
    }
    write!(writer, "Choose 1-{}: ", request.choices.len()).map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    read_approval_answer(request)
}

fn read_clarify_answer(choices: Option<&[String]>) -> Result<String, String> {
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

fn read_approval_answer(request: &ApprovalRequest) -> Result<String, String> {
    let mut line = String::new();
    let read = io::stdin()
        .read_line(&mut line)
        .map_err(|error| error.to_string())?;
    if read == 0 {
        return Ok("deny".to_string());
    }

    let choice = line.trim();
    if let Ok(index) = choice.parse::<usize>()
        && (1..=request.choices.len()).contains(&index)
    {
        return Ok(request.choices[index - 1].clone());
    }
    Ok(choice.to_string())
}

fn approval_choice_label(choice: &str, allow_permanent: bool) -> &'static str {
    match choice {
        "once" => "Allow once",
        "session" => "Allow this session",
        "always" if allow_permanent => "Always allow",
        "always" => "Allow this session",
        _ => "Deny",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SharedBufferWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut locked = self.0.lock().unwrap();
            locked.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn chat_cli_parses_json_events_flag() {
        let cli = Cli::try_parse_from(["hermes-agent", "chat", "hello", "--json-events"]).unwrap();
        match cli.command {
            Some(Command::Chat {
                prompt,
                json_events,
                gateway_events,
                ..
            }) => {
                assert_eq!(prompt, "hello");
                assert!(json_events);
                assert!(!gateway_events);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn chat_cli_parses_gateway_events_flag() {
        let cli =
            Cli::try_parse_from(["hermes-agent", "chat", "hello", "--gateway-events"]).unwrap();
        match cli.command {
            Some(Command::Chat {
                prompt,
                json_events,
                gateway_events,
                ..
            }) => {
                assert_eq!(prompt, "hello");
                assert!(!json_events);
                assert!(gateway_events);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn chat_event_emitter_writes_json_lines() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let emitter = ChatEventEmitter::from_writer(Box::new(SharedBufferWriter(shared.clone())));

        emit_tool_progress_event(
            &emitter,
            &ToolProgressUpdate {
                event_type: String::from("tool.started"),
                function_name: Some(String::from("terminal")),
                preview: Some(String::from("rm -rf /tmp/demo")),
                function_args: Some(json!({"command": "rm -rf /tmp/demo"})),
                duration_ms: None,
                is_error: None,
            },
        );
        emit_approval_event(
            &emitter,
            &ApprovalRequest {
                command: String::from("rm -rf /tmp/demo"),
                description: String::from("dangerous command"),
                pattern_keys: vec![String::from("delete")],
                choices: vec![String::from("once"), String::from("deny")],
                allow_permanent: false,
            },
        );

        let rendered = String::from_utf8(shared.lock().unwrap().clone()).unwrap();
        let lines = rendered
            .lines()
            .map(|line| serde_json::from_str::<JsonValue>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], json!("tool_progress"));
        assert_eq!(lines[0]["function_name"], json!("terminal"));
        assert_eq!(lines[1]["event"], json!("approval_request"));
        assert_eq!(lines[1]["command"], json!("rm -rf /tmp/demo"));
    }

    #[test]
    fn chat_gateway_event_emitter_writes_gateway_lines() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let emitter = ChatEventEmitter::from_writer(Box::new(SharedBufferWriter(shared.clone())));
        let mut bridge = GatewayEventBridge::default();

        emitter.emit(&bridge.gateway_ready());
        emitter.emit(&bridge.message_start());
        let started = bridge.on_tool_progress(&ToolProgressUpdate {
            event_type: String::from("tool.started"),
            function_name: Some(String::from("terminal")),
            preview: Some(String::from("echo hi")),
            function_args: Some(json!({"command": "echo hi"})),
            duration_ms: None,
            is_error: None,
        });
        emit_gateway_events(&emitter, &started);
        let final_event = bridge.on_final_response(&hermes_core::AgentTurnResult {
            final_response: String::from("done"),
            reasoning: Some(String::from("thought process")),
            api_calls: 1,
            tool_calls: 1,
            model: String::from("test-model"),
            provider: String::from("custom"),
            base_url: String::from("http://localhost"),
            session_id: Some(String::from("session_123")),
        });
        emitter.emit(&final_event);

        let rendered = String::from_utf8(shared.lock().unwrap().clone()).unwrap();
        let lines = rendered
            .lines()
            .map(|line| serde_json::from_str::<JsonValue>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0]["type"], json!("gateway.ready"));
        assert_eq!(lines[1]["type"], json!("message.start"));
        assert_eq!(lines[2]["type"], json!("tool.start"));
        assert_eq!(lines[3]["type"], json!("tool.progress"));
        assert_eq!(lines[4]["type"], json!("message.complete"));
        assert_eq!(lines[4]["payload"]["text"], json!("done"));
        assert_eq!(lines[4]["payload"]["reasoning"], json!("thought process"));
    }
}
