mod acp_cmd;
mod auth_cmd;
mod backup;
mod checkpoints_cmd;
mod claw_cmd;
mod completion;
mod config_cmd;
mod curator_cmd;
mod dashboard_cmd;
mod debug;
mod doctor;
mod dump;
mod fallback_cmd;
mod gateway_cmd;
mod hooks;
mod insights_cmd;
mod login_cmd;
mod logs;
mod mcp_cmd;
mod mcp_server;
mod memory_cmd;
mod model_cmd;
mod native_api_server;
mod native_gateway_runtime;
mod native_webhook_server;
mod pairing_cmd;
mod plugins_cmd;
mod profile_cmd;
mod python_bridge;
mod setup_cmd;
mod skills_cmd;
mod skills_guard;
mod slack_cmd;
mod snapshot_cmd;
mod tools_cmd;
mod uninstall_cmd;
mod update_cmd;
mod webhook;
mod whatsapp_cmd;

use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use hermes_core::{
    DelegateExecutor, EnvLoadReport, HermesContext, KanbanDispatchOptions, LoadedConfig,
    LoggingMode, LoggingSetup, ModelOverrides, ToolRuntime, dispatch_kanban_once,
    get_tool_definitions, handle_cronjob, is_container, is_wsl, kanban_has_spawnable_ready,
    run_cron_job_now, run_due_cron_jobs, run_kanban_task,
};
use serde_json::{Value as JsonValue, json};

#[cfg(test)]
pub(crate) fn cli_test_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Parser, Debug)]
#[command(name = "hermes", version, about = "Hermes Rust bootstrap")]
struct Cli {
    #[arg(short = 'p', long = "profile", global = true)]
    _profile: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Paths,
    Version,
    Dump(dump::DumpArgs),
    Doctor(doctor::DoctorArgs),
    Debug {
        #[command(subcommand)]
        command: Option<debug::DebugCommand>,
    },
    Hooks {
        #[command(subcommand)]
        command: Option<hooks::HooksCommand>,
    },
    Login(login_cmd::LoginArgs),
    Fallback {
        #[command(subcommand)]
        command: Option<fallback_cmd::FallbackCommand>,
    },
    Model {
        #[command(subcommand)]
        command: Option<model_cmd::ModelCommand>,
    },
    Slack {
        #[command(subcommand)]
        command: Option<slack_cmd::SlackCommand>,
    },
    Webhook {
        #[command(subcommand)]
        command: Option<webhook::WebhookCommand>,
    },
    Completion(completion::CompletionArgs),
    Dashboard(dashboard_cmd::DashboardArgs),
    Gateway(gateway_cmd::GatewayArgs),
    Skills {
        #[command(subcommand)]
        command: Option<skills_cmd::SkillsCommand>,
    },
    Checkpoints {
        #[command(subcommand)]
        command: Option<checkpoints_cmd::CheckpointsCommand>,
    },
    Snapshot {
        #[command(subcommand)]
        command: Option<snapshot_cmd::SnapshotCommand>,
    },
    Plugins {
        #[command(subcommand)]
        command: Option<plugins_cmd::PluginsCommand>,
    },
    Curator {
        #[command(subcommand)]
        command: Option<curator_cmd::CuratorCommand>,
    },
    Memory {
        #[command(subcommand)]
        command: Option<memory_cmd::MemoryCommand>,
    },
    Mcp {
        #[command(subcommand)]
        command: Option<mcp_cmd::McpCommand>,
    },
    Insights(insights_cmd::InsightsArgs),
    Claw {
        #[command(subcommand)]
        command: Option<claw_cmd::ClawCommand>,
    },
    Acp(acp_cmd::AcpArgs),
    Logout(auth_cmd::LogoutArgs),
    Auth {
        #[command(subcommand)]
        command: Option<auth_cmd::AuthCommand>,
    },
    Setup(setup_cmd::SetupArgs),
    Config {
        #[command(subcommand)]
        command: Option<config_cmd::ConfigCommand>,
    },
    Pairing {
        #[command(subcommand)]
        command: Option<pairing_cmd::PairingCommand>,
    },
    Uninstall(uninstall_cmd::UninstallArgs),
    Backup(backup::BackupArgs),
    Import(backup::ImportArgs),
    Profile {
        #[command(subcommand)]
        command: Option<profile_cmd::ProfileCommand>,
    },
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
    Sessions {
        #[command(subcommand)]
        command: SessionsCommand,
    },
    Cron {
        #[command(subcommand)]
        command: Option<CronCommand>,
    },
    Logs(logs::LogsArgs),
    Kanban {
        #[command(subcommand)]
        command: KanbanCommand,
    },
    Tools(tools_cmd::ToolsArgs),
    Update(update_cmd::UpdateArgs),
    Whatsapp,
    Status,
}

#[derive(Subcommand, Debug)]
enum SessionsCommand {
    List {
        #[arg(long, default_value_t = 20)]
        limit: i64,
        #[arg(long, default_value_t = 0)]
        offset: i64,
        #[arg(long)]
        source: Option<String>,
    },
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: i64,
        #[arg(long, default_value_t = 0)]
        offset: i64,
        #[arg(long)]
        source: Option<String>,
    },
    Export {
        output: String,
        #[arg(long)]
        source: Option<String>,
        #[arg(long = "session-id")]
        session_id: Option<String>,
    },
    Rename {
        id: String,
        title: Vec<String>,
    },
    Delete {
        id: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    Prune {
        #[arg(long = "older-than", default_value_t = 90)]
        older_than: u64,
        #[arg(long)]
        source: Option<String>,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    Stats,
}

#[derive(Subcommand, Debug)]
enum CronCommand {
    List {
        #[arg(long = "all", default_value_t = false)]
        all: bool,
    },
    #[command(alias = "add")]
    Create(CronCreateArgs),
    Edit(CronEditArgs),
    Pause {
        job_id: String,
    },
    Resume {
        job_id: String,
    },
    Tick,
    Run {
        job_id: String,
    },
    #[command(alias = "rm", alias = "delete")]
    Remove {
        job_id: String,
    },
    Status,
}

#[derive(Args, Debug, Clone)]
struct CronCreateArgs {
    schedule: String,
    prompt: Option<String>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    deliver: Option<String>,
    #[arg(long)]
    repeat: Option<i64>,
    #[arg(long = "skill")]
    skills: Vec<String>,
    #[arg(long)]
    script: Option<String>,
    #[arg(long = "no-agent", default_value_t = false)]
    no_agent: bool,
    #[arg(long)]
    workdir: Option<String>,
}

#[derive(Args, Debug, Clone)]
struct CronEditArgs {
    job_id: String,
    #[arg(long)]
    schedule: Option<String>,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    deliver: Option<String>,
    #[arg(long)]
    repeat: Option<i64>,
    #[arg(long = "skill")]
    skills: Vec<String>,
    #[arg(long = "add-skill")]
    add_skills: Vec<String>,
    #[arg(long = "remove-skill")]
    remove_skills: Vec<String>,
    #[arg(long = "clear-skills", default_value_t = false)]
    clear_skills: bool,
    #[arg(long)]
    script: Option<String>,
    #[arg(long = "no-agent", action = clap::ArgAction::SetTrue)]
    no_agent: bool,
    #[arg(long = "agent", action = clap::ArgAction::SetTrue)]
    agent: bool,
    #[arg(long)]
    workdir: Option<String>,
}

#[derive(Subcommand, Debug)]
enum KanbanCommand {
    Tick {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        max_spawn: Option<usize>,
        #[arg(long)]
        failure_limit: Option<i64>,
    },
    Daemon {
        #[arg(long, default_value_t = 60.0)]
        interval: f64,
        #[arg(long)]
        max_spawn: Option<usize>,
        #[arg(long)]
        failure_limit: Option<i64>,
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long, short = 'v')]
        verbose: bool,
    },
    Run {
        id: String,
        #[arg(long)]
        dry_run: bool,
    },
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
    let logging = context.setup_logging(&config, LoggingMode::Cli)?;
    let session_store = context.open_session_store()?;
    emit_warnings(&env_report, &config);
    log::info!(
        target: "hermes_cli",
        "startup profile={} home={}",
        context.current_profile_name(),
        context.hermes_home().display()
    );
    let argv = std::iter::once(String::from("hermes")).chain(profile_override.args);
    let cli = Cli::parse_from(argv);

    match cli.command.unwrap_or(Command::Status) {
        Command::Paths => print_paths(&context, &config, &logging),
        Command::Version => dump::print_version(),
        Command::Dump(args) => dump::print_dump(&context, &config, args)?,
        Command::Doctor(args) => {
            doctor::print_doctor(&context, &env_report, &config, &session_store, args)?
        }
        Command::Debug { command } => debug::print_debug(&context, &config, command)?,
        Command::Hooks { command } => hooks::print_hooks(&context, &config, command)?,
        Command::Login(args) => login_cmd::print_login(args)?,
        Command::Fallback { command } => {
            fallback_cmd::print_fallback(&context.config_path(), &config, command)?
        }
        Command::Model { command } => model_cmd::print_model(&context, &config, command)?,
        Command::Dashboard(args) => dashboard_cmd::print_dashboard(args)?,
        Command::Gateway(args) => gateway_cmd::print_gateway(&context, args)?,
        Command::Skills { command } => skills_cmd::print_skills(&context, command)?,
        Command::Checkpoints { command } => checkpoints_cmd::print_checkpoints(&context, command)?,
        Command::Snapshot { command } => snapshot_cmd::print_snapshot(&context, command)?,
        Command::Plugins { command } => plugins_cmd::print_plugins(&context, command)?,
        Command::Curator { command } => curator_cmd::print_curator(&context, command)?,
        Command::Memory { command } => memory_cmd::print_memory(&context, &config, command)?,
        Command::Mcp { command } => mcp_cmd::print_mcp(&context, command)?,
        Command::Insights(args) => insights_cmd::print_insights(&context, args)?,
        Command::Claw { command } => claw_cmd::print_claw(command)?,
        Command::Acp(args) => acp_cmd::print_acp(args)?,
        Command::Slack { command } => slack_cmd::print_slack(&context, &config, command)?,
        Command::Webhook { command } => webhook::print_webhook(&context, &config, command)?,
        Command::Completion(args) => completion::print_completion(args)?,
        Command::Logout(args) => auth_cmd::print_logout(&context, &config, args)?,
        Command::Auth { command } => auth_cmd::print_auth(&context, &config, command)?,
        Command::Setup(args) => setup_cmd::print_setup(&context, args)?,
        Command::Config { command } => {
            config_cmd::print_config(&context, &env_report, &config, command)?
        }
        Command::Pairing { command } => pairing_cmd::print_pairing(&context, command)?,
        Command::Uninstall(args) => uninstall_cmd::print_uninstall(&context, args)?,
        Command::Backup(args) => backup::print_backup(&context, args)?,
        Command::Import(args) => backup::print_import(&context, args)?,
        Command::Profile { command } => profile_cmd::print_profile(&context, command)?,
        Command::Chat {
            prompt,
            session,
            model,
            provider,
            base_url,
            api_key,
            api_mode,
            toolsets,
        } => run_chat(
            &context,
            &config,
            &session_store,
            prompt,
            session,
            model,
            provider,
            base_url,
            api_key,
            api_mode,
            toolsets,
        )?,
        Command::Sessions { command } => print_sessions(&context, &session_store, command)?,
        Command::Cron { command } => print_cron(&context, &config, &session_store, command)?,
        Command::Logs(args) => logs::print_logs(&context, args)?,
        Command::Kanban { command } => print_kanban(&context, &config, command)?,
        Command::Tools(args) => tools_cmd::print_tools(&context, &config, args)?,
        Command::Update(args) => update_cmd::print_update(&context, args)?,
        Command::Whatsapp => whatsapp_cmd::print_whatsapp(&context)?,
        Command::Status => print_status(&context, &env_report, &config, &session_store),
    }

    Ok(())
}

fn print_paths(context: &HermesContext, config: &LoadedConfig, logging: &LoggingSetup) {
    print_kv("hermes_home", context.hermes_home());
    print_kv("default_root", context.default_hermes_root());
    print_kv("config", context.config_path());
    print_kv("log_dir", logging.log_dir.clone());
    print_kv("agent_log", logging.agent_log.clone());
    print_kv("errors_log", logging.errors_log.clone());
    print_kv("soul", context.hermes_home().join("SOUL.md"));
    print_kv("skills", context.skills_dir());
    print_kv("env", context.env_path());
    print_kv("optional_skills", context.optional_skills_dir(None));
    println!("config_logging_level={}", config.config.logging.level);
    match context.subprocess_home() {
        Some(path) => print_kv("subprocess_home", path),
        None => println!("subprocess_home=<disabled>"),
    }
}

fn print_sessions(
    context: &HermesContext,
    session_store: &hermes_core::SessionStore,
    command: SessionsCommand,
) -> Result<(), Box<dyn Error>> {
    match command {
        SessionsCommand::List {
            limit,
            offset,
            source,
        } => {
            let (limit, offset) = validate_pagination(limit, offset)?;
            let rows = session_store.search_sessions(source.as_deref(), limit, offset)?;
            for row in rows {
                println!(
                    "{}\tsource={}\tmodel={}\ttitle={}\tlast_active={:.3}\tpreview={}",
                    row.id,
                    row.source,
                    row.model.unwrap_or_default(),
                    row.title.unwrap_or_default(),
                    row.last_active,
                    row.preview
                );
            }
        }
        SessionsCommand::Search {
            query,
            limit,
            offset,
            source,
        } => {
            let (limit, offset) = validate_pagination(limit, offset)?;
            let source_filter = source.map(|item| vec![item]);
            let rows = session_store.search_messages(
                &query,
                source_filter.as_deref(),
                None,
                None,
                limit,
                offset,
            )?;
            for row in rows {
                println!(
                    "{}\tsession={}\trole={}\tsource={}\tsnippet={}",
                    row.id, row.session_id, row.role, row.source, row.snippet
                );
            }
        }
        SessionsCommand::Export {
            output,
            source,
            session_id,
        } => {
            if output.trim().is_empty() {
                return Err("sessions export output cannot be empty".into());
            }
            if source.is_some() && session_id.is_some() {
                return Err(
                    "sessions export does not accept both --source and --session-id".into(),
                );
            }

            if let Some(session_id) = session_id {
                let resolved = resolve_existing_session_id(session_store, &session_id)?;
                let export = session_store
                    .export_session(&resolved)?
                    .ok_or_else(|| format!("session '{session_id}' not found"))?;
                write_session_exports(&output, &[export])?;
            } else {
                let exports = session_store.export_all(source.as_deref())?;
                write_session_exports(&output, &exports)?;
            }
        }
        SessionsCommand::Rename { id, title } => {
            if title.is_empty() {
                return Err("sessions rename requires a non-empty title".into());
            }
            let resolved = resolve_existing_session_id(session_store, &id)?;
            let title = title.join(" ");
            let updated = session_store.set_session_title(&resolved, &title)?;
            println!("renamed={updated}");
            println!("id={resolved}");
            println!("title={title}");
        }
        SessionsCommand::Delete { id, yes } => {
            let resolved = resolve_existing_session_id(session_store, &id)?;
            if !yes
                && !confirm_prompt(&format!(
                    "Delete session '{resolved}' and all its messages? [y/N] "
                ))?
            {
                println!("cancelled=true");
                return Ok(());
            }
            let deleted = session_store.delete_session(&resolved)?;
            if deleted {
                remove_session_files(&context.hermes_home().join("sessions"), &resolved);
            }
            println!("deleted={deleted}");
            if deleted {
                println!("id={resolved}");
            }
        }
        SessionsCommand::Prune {
            older_than,
            source,
            yes,
        } => {
            let source_msg = source
                .as_deref()
                .map(|value| format!(" from '{value}'"))
                .unwrap_or_default();
            if !yes
                && !confirm_prompt(&format!(
                    "Delete all ended sessions older than {older_than} days{source_msg}? [y/N] "
                ))?
            {
                println!("cancelled=true");
                return Ok(());
            }
            let removed = session_store.prune_sessions(older_than, source.as_deref())?;
            let sessions_dir = context.hermes_home().join("sessions");
            for session_id in &removed {
                remove_session_files(&sessions_dir, session_id);
            }
            println!("pruned={}", removed.len());
        }
        SessionsCommand::Stats => {
            println!("total_sessions={}", session_store.session_count()?);
            println!("total_messages={}", session_store.message_count(None)?);
            for (source, count) in session_store.session_counts_by_source()? {
                println!("source={source}\tsessions={count}");
            }
            if let Ok(metadata) = fs::metadata(session_store.path()) {
                println!("database_size_bytes={}", metadata.len());
            }
        }
    }
    Ok(())
}

fn print_status(
    context: &HermesContext,
    env_report: &EnvLoadReport,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
) {
    println!("app=hermes");
    println!("mode=rust-bootstrap");
    println!("current_profile={}", context.current_profile_name());
    println!("hermes_home={}", context.hermes_home().display());
    println!("default_root={}", context.default_hermes_root().display());
    println!("config={}", config.path.display());
    println!("state_db={}", session_store.path().display());
    println!(
        "session_count={}",
        session_store.session_count().unwrap_or_default()
    );
    println!("loaded_env_files={}", env_report.loaded_paths.len());
    println!("terminal_backend={}", config.config.terminal.backend);
    println!(
        "tool_count={}",
        get_tool_definitions(
            Some(&[String::from("hermes-cli")]),
            disabled_memory_toolsets(&config.config.memory).as_deref(),
        )
        .len()
    );
    println!("logging_level={}", config.config.logging.level);
    println!("display_skin={}", config.config.display.skin);
    println!("redact_secrets={}", config.config.security.redact_secrets);
    println!("force_ipv4={}", config.config.network.force_ipv4);
    println!("termux={}", context.is_termux());
    println!("wsl={}", is_wsl());
    println!("container={}", is_container());
    if let Some(warning) = context.profile_fallback_warning() {
        println!("warning={warning}");
    }
    println!("note=Rust one-shot chat is available via `hermes chat`");
}

#[allow(clippy::too_many_arguments)]
fn run_chat(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
    prompt: String,
    session: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    api_mode: Option<String>,
    toolsets: Vec<String>,
) -> Result<(), Box<dyn Error>> {
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
    let disabled = disabled_memory_toolsets(&config.config.memory);
    let tool_names = get_tool_definitions(Some(&enabled_toolsets), disabled.as_deref())
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
    let delegate = DelegateExecutor::new(
        context.clone(),
        config.clone(),
        "rust-delegate",
        enabled_toolsets.clone(),
        overrides.clone(),
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    );
    let mut runtime = ToolRuntime::default()
        .with_hermes_home(context.hermes_home())
        .with_available_tool_names(tool_names)
        .with_clarify_callback(run_clarify_prompt)
        .with_delegate_callback(move |request| delegate.execute(request));
    let _ = runtime.load_memory_store(&config.config.memory);
    let result = context.run_chat_completions_turn(
        config,
        &prompt,
        &runtime,
        Some(&enabled_toolsets),
        &overrides,
        session.as_deref(),
        Some(session_store),
    )?;
    println!("{}", result.final_response);
    Ok(())
}

fn print_cron(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
    command: Option<CronCommand>,
) -> Result<(), Box<dyn Error>> {
    let base_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match command.unwrap_or(CronCommand::List { all: false }) {
        CronCommand::List { all } => print_cron_list(context, all),
        CronCommand::Create(args) => print_cron_create(context, args),
        CronCommand::Edit(args) => print_cron_edit(context, args),
        CronCommand::Pause { job_id } => print_cron_job_action(context, "pause", &job_id, "Paused"),
        CronCommand::Resume { job_id } => {
            print_cron_job_action(context, "resume", &job_id, "Resumed")
        }
        CronCommand::Tick => {
            let result = run_due_cron_jobs(context, config, session_store, &base_cwd)?;
            println!("due={}", result.due_count);
            println!("ran={}", result.ran_count);
            println!("succeeded={}", result.success_count);
            println!("failed={}", result.failure_count);
            println!("silent={}", result.silent_count);
            for item in result.results {
                println!(
                    "{}\tname={}\tsuccess={}\tsilent={}\tno_agent={}\tsession={}\toutput={}\terror={}",
                    item.job_id,
                    item.job_name,
                    item.success,
                    item.silent,
                    item.no_agent,
                    item.session_id.unwrap_or_default(),
                    item.output_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_default(),
                    item.error.unwrap_or_default(),
                );
            }
            Ok(())
        }
        CronCommand::Run { job_id } => {
            if job_id.trim().is_empty() {
                return Err("cron run requires a non-empty job id".into());
            }
            let item = run_cron_job_now(context, config, session_store, &base_cwd, &job_id)?;
            println!("job_id={}", item.job_id);
            println!("name={}", item.job_name);
            println!("success={}", item.success);
            println!("silent={}", item.silent);
            println!("no_agent={}", item.no_agent);
            println!("session={}", item.session_id.unwrap_or_default());
            println!(
                "output={}",
                item.output_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default()
            );
            println!("error={}", item.error.unwrap_or_default());
            if !item.final_response.is_empty() {
                println!("{}", item.final_response);
            }
            Ok(())
        }
        CronCommand::Remove { job_id } => {
            print_cron_job_action(context, "remove", &job_id, "Removed")
        }
        CronCommand::Status => print_cron_status(context),
    }
}

fn cron_tool_result(context: &HermesContext, args: JsonValue) -> Result<JsonValue, Box<dyn Error>> {
    let runtime = ToolRuntime::default().with_hermes_home(context.hermes_home());
    let raw = handle_cronjob(&args, &runtime);
    let value: JsonValue = serde_json::from_str(&raw)?;
    if let Some(error) = value.get("error").and_then(JsonValue::as_str) {
        return Err(error.to_string().into());
    }
    if !value
        .get("success")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        return Err("cron command failed".into());
    }
    Ok(value)
}

fn normalized_cli_strings(values: &[String]) -> Vec<String> {
    let mut output = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if !trimmed.is_empty() && !output.iter().any(|existing| existing == trimmed) {
            output.push(trimmed.to_string());
        }
    }
    output
}

fn print_cron_list(context: &HermesContext, show_all: bool) -> Result<(), Box<dyn Error>> {
    let result = cron_tool_result(
        context,
        json!({
            "action": "list",
            "include_disabled": show_all,
        }),
    )?;
    let jobs = result
        .get("jobs")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default();
    if jobs.is_empty() {
        println!("No scheduled jobs.");
        println!("Create one with 'hermes cron create ...' or the /cron command in chat.");
        return Ok(());
    }

    println!(
        "{:<14} {:<12} {:<18} {:<26} Name",
        "ID", "State", "Schedule", "Next Run"
    );
    println!(
        "{:<14} {:<12} {:<18} {:<26} ----",
        "--------------", "------------", "------------------", "--------------------------"
    );
    for job in &jobs {
        let job_id = json_string(job, "job_id").unwrap_or_else(|| String::from("?"));
        let state = json_string(job, "state").unwrap_or_else(|| String::from("?"));
        let schedule = json_string(job, "schedule").unwrap_or_else(|| String::from("?"));
        let next_run_at = json_string(job, "next_run_at").unwrap_or_default();
        let name = json_string(job, "name").unwrap_or_else(|| String::from("(unnamed)"));
        println!(
            "{:<14} {:<12} {:<18} {:<26} {}",
            truncate_plain(&job_id, 14),
            truncate_plain(&state, 12),
            truncate_plain(&schedule, 18),
            truncate_plain(&next_run_at, 26),
            name
        );
    }
    println!();
    println!("{} scheduled job(s)", jobs.len());
    Ok(())
}

fn print_cron_create(context: &HermesContext, args: CronCreateArgs) -> Result<(), Box<dyn Error>> {
    let mut payload = serde_json::Map::new();
    payload.insert(String::from("action"), json!("create"));
    payload.insert(String::from("schedule"), json!(args.schedule));
    insert_optional_string(&mut payload, "prompt", args.prompt);
    insert_optional_string(&mut payload, "name", args.name);
    insert_optional_string(&mut payload, "deliver", args.deliver);
    if let Some(repeat) = args.repeat {
        payload.insert(String::from("repeat"), json!(repeat));
    }
    let skills = normalized_cli_strings(&args.skills);
    if !skills.is_empty() {
        payload.insert(String::from("skills"), json!(skills));
    }
    insert_optional_string(&mut payload, "script", args.script);
    if args.no_agent {
        payload.insert(String::from("no_agent"), json!(true));
    }
    insert_optional_string(&mut payload, "workdir", args.workdir);

    let result = cron_tool_result(context, JsonValue::Object(payload))?;
    println!("Created job: {}", required_json_string(&result, "job_id")?);
    println!("  Name: {}", required_json_string(&result, "name")?);
    println!("  Schedule: {}", required_json_string(&result, "schedule")?);
    if let Some(skills) = result.get("skills").and_then(JsonValue::as_array)
        && !skills.is_empty()
    {
        println!("  Skills: {}", join_json_strings(skills));
    }
    if let Some(job) = result.get("job").and_then(JsonValue::as_object) {
        if let Some(script) = job.get("script").and_then(JsonValue::as_str) {
            println!("  Script: {script}");
        }
        if job
            .get("no_agent")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false)
        {
            println!("  Mode: no-agent (script stdout delivered directly)");
        }
        if let Some(workdir) = job.get("workdir").and_then(JsonValue::as_str) {
            println!("  Workdir: {workdir}");
        }
    }
    println!(
        "  Next run: {}",
        required_json_string(&result, "next_run_at")?
    );
    Ok(())
}

fn print_cron_edit(context: &HermesContext, args: CronEditArgs) -> Result<(), Box<dyn Error>> {
    let existing = find_cron_job(context, &args.job_id)?;
    let mut payload = serde_json::Map::new();
    payload.insert(String::from("action"), json!("update"));
    payload.insert(String::from("job_id"), json!(args.job_id));
    insert_optional_string(&mut payload, "schedule", args.schedule);
    insert_optional_string(&mut payload, "prompt", args.prompt);
    insert_optional_string(&mut payload, "name", args.name);
    insert_optional_string(&mut payload, "deliver", args.deliver);
    if let Some(repeat) = args.repeat {
        payload.insert(String::from("repeat"), json!(repeat));
    }

    if args.clear_skills {
        payload.insert(String::from("skills"), json!([]));
    } else {
        let replacement = normalized_cli_strings(&args.skills);
        if !replacement.is_empty() {
            payload.insert(String::from("skills"), json!(replacement));
        } else {
            let add = normalized_cli_strings(&args.add_skills);
            let remove = normalized_cli_strings(&args.remove_skills);
            if !add.is_empty() || !remove.is_empty() {
                let mut final_skills = existing_cron_skills(&existing);
                final_skills.retain(|skill| !remove.iter().any(|item| item == skill));
                for skill in add {
                    if !final_skills.iter().any(|existing| existing == &skill) {
                        final_skills.push(skill);
                    }
                }
                payload.insert(String::from("skills"), json!(final_skills));
            }
        }
    }

    insert_optional_string(&mut payload, "script", args.script);
    if args.no_agent && args.agent {
        return Err("cron edit accepts either --no-agent or --agent, not both".into());
    }
    if args.no_agent {
        payload.insert(String::from("no_agent"), json!(true));
    } else if args.agent {
        payload.insert(String::from("no_agent"), json!(false));
    }
    insert_optional_string(&mut payload, "workdir", args.workdir);

    let result = cron_tool_result(context, JsonValue::Object(payload))?;
    let updated = result
        .get("job")
        .ok_or("cron update result did not include job")?;
    println!("Updated job: {}", required_json_string(updated, "job_id")?);
    println!("  Name: {}", required_json_string(updated, "name")?);
    println!("  Schedule: {}", required_json_string(updated, "schedule")?);
    let skills = updated
        .get("skills")
        .and_then(JsonValue::as_array)
        .map(|values| join_json_strings(values))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| String::from("none"));
    println!("  Skills: {skills}");
    if let Some(script) = updated.get("script").and_then(JsonValue::as_str) {
        println!("  Script: {script}");
    }
    if updated
        .get("no_agent")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        println!("  Mode: no-agent (script stdout delivered directly)");
    }
    if let Some(workdir) = updated.get("workdir").and_then(JsonValue::as_str) {
        println!("  Workdir: {workdir}");
    }
    Ok(())
}

fn print_cron_job_action(
    context: &HermesContext,
    action: &str,
    job_id: &str,
    success_verb: &str,
) -> Result<(), Box<dyn Error>> {
    let result = cron_tool_result(
        context,
        json!({
            "action": action,
            "job_id": job_id,
        }),
    )?;
    let job = result
        .get("job")
        .or_else(|| result.get("removed_job"))
        .ok_or("cron result did not include job")?;
    let name = json_string(job, "name").unwrap_or_else(|| job_id.to_string());
    println!("{success_verb} job: {name} ({job_id})");
    if matches!(action, "resume" | "run")
        && let Some(next_run_at) = job.get("next_run_at").and_then(JsonValue::as_str)
    {
        println!("  Next run: {next_run_at}");
    }
    if action == "run" {
        println!("  It will run on the next scheduler tick.");
    }
    Ok(())
}

fn print_cron_status(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let result = cron_tool_result(
        context,
        json!({
            "action": "list",
            "include_disabled": false,
        }),
    )?;
    let jobs = result
        .get("jobs")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default();
    if jobs.is_empty() {
        println!("No active jobs");
        return Ok(());
    }

    println!("{} active job(s)", jobs.len());
    if let Some(next_run) = jobs
        .iter()
        .filter_map(|job| job.get("next_run_at").and_then(JsonValue::as_str))
        .min()
    {
        println!("Next run: {next_run}");
    }
    println!("Use `hermes gateway status` to check automatic scheduler availability.");
    Ok(())
}

fn find_cron_job(context: &HermesContext, job_id: &str) -> Result<JsonValue, Box<dyn Error>> {
    let result = cron_tool_result(
        context,
        json!({
            "action": "list",
            "include_disabled": true,
        }),
    )?;
    let Some(job) = result
        .get("jobs")
        .and_then(JsonValue::as_array)
        .and_then(|jobs| {
            jobs.iter().find(|job| {
                job.get("job_id")
                    .and_then(JsonValue::as_str)
                    .is_some_and(|id| id == job_id)
            })
        })
    else {
        return Err(format!("Job not found: {job_id}").into());
    };
    Ok(job.clone())
}

fn existing_cron_skills(job: &JsonValue) -> Vec<String> {
    let mut skills = job
        .get("skills")
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if skills.is_empty()
        && let Some(skill) = job.get("skill").and_then(JsonValue::as_str)
        && !skill.trim().is_empty()
    {
        skills.push(skill.trim().to_string());
    }
    skills
}

fn insert_optional_string(
    payload: &mut serde_json::Map<String, JsonValue>,
    key: &str,
    value: Option<String>,
) {
    if let Some(value) = value {
        payload.insert(key.to_string(), JsonValue::String(value));
    }
}

fn required_json_string(value: &JsonValue, key: &str) -> Result<String, Box<dyn Error>> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("cron result missing '{key}'").into())
}

fn json_string(value: &JsonValue, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

fn join_json_strings(values: &[JsonValue]) -> String {
    values
        .iter()
        .filter_map(JsonValue::as_str)
        .map(str::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn truncate_plain(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return value.chars().take(max_chars).collect();
    }
    format!(
        "{}...",
        value
            .chars()
            .take(max_chars.saturating_sub(3))
            .collect::<String>()
    )
}

fn print_kanban(
    context: &HermesContext,
    config: &LoadedConfig,
    command: KanbanCommand,
) -> Result<(), Box<dyn Error>> {
    struct PidFileGuard {
        path: PathBuf,
    }

    impl PidFileGuard {
        fn create(path: &Path) -> io::Result<Self> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, std::process::id().to_string())?;
            Ok(Self {
                path: path.to_path_buf(),
            })
        }
    }

    impl Drop for PidFileGuard {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    match command {
        KanbanCommand::Tick {
            dry_run,
            max_spawn,
            failure_limit,
        } => {
            if max_spawn.is_some_and(|value| value == 0) {
                return Err("max_spawn must be positive".into());
            }
            if failure_limit.is_some_and(|value| value <= 0) {
                return Err("failure_limit must be positive".into());
            }
            let result = dispatch_kanban_once(
                context,
                config,
                KanbanDispatchOptions {
                    dry_run,
                    max_spawn,
                    failure_limit,
                },
            )?;
            println!("dry_run={dry_run}");
            println!("reclaimed={}", result.reclaimed);
            println!("promoted={}", result.promoted);
            println!("spawned={}", result.spawned.len());
            println!("skipped_unassigned={}", result.skipped_unassigned.len());
            println!("skipped_nonspawnable={}", result.skipped_nonspawnable.len());
            println!("crashed={}", result.crashed.len());
            println!("timed_out={}", result.timed_out.len());
            println!("auto_blocked={}", result.auto_blocked.len());
            for item in result.spawned {
                println!(
                    "{}\tassignee={}\tworkspace={}\tpid={}",
                    item.task_id,
                    item.assignee,
                    item.workspace_path,
                    item.pid.map(|value| value.to_string()).unwrap_or_default()
                );
            }
        }
        KanbanCommand::Daemon {
            interval,
            max_spawn,
            failure_limit,
            pidfile,
            verbose,
        } => {
            if interval <= 0.0 {
                return Err("interval must be positive".into());
            }
            if max_spawn.is_some_and(|value| value == 0) {
                return Err("max_spawn must be positive".into());
            }
            if failure_limit.is_some_and(|value| value <= 0) {
                return Err("failure_limit must be positive".into());
            }
            let _pidfile_guard = match pidfile.as_deref() {
                Some(path) => Some(PidFileGuard::create(path)?),
                None => None,
            };

            let running = Arc::new(AtomicBool::new(true));
            let signal_flag = Arc::clone(&running);
            ctrlc::set_handler(move || {
                signal_flag.store(false, Ordering::SeqCst);
            })?;

            eprintln!(
                "Kanban dispatcher running (interval={}s, pid={}). Ctrl-C to stop.",
                interval,
                std::process::id()
            );
            let mut bad_ticks = 0_u32;
            let mut last_warn_at = 0_i64;
            let mut last_error_at = 0_i64;
            const HEALTH_WINDOW: u32 = 6;

            while running.load(Ordering::SeqCst) {
                match dispatch_kanban_once(
                    context,
                    config,
                    KanbanDispatchOptions {
                        dry_run: false,
                        max_spawn,
                        failure_limit,
                    },
                ) {
                    Ok(result) => {
                        let ready_pending = !result.skipped_unassigned.is_empty()
                            || kanban_has_spawnable_ready(context)?;
                        let spawned_any = !result.spawned.is_empty();
                        if ready_pending && !spawned_any {
                            bad_ticks += 1;
                        } else {
                            bad_ticks = 0;
                        }
                        if bad_ticks >= HEALTH_WINDOW {
                            let now = chrono::Local::now().timestamp();
                            if now - last_warn_at >= 300 {
                                eprintln!(
                                    "[{}] WARN dispatcher stuck: ready queue non-empty for {} consecutive ticks but 0 workers spawned successfully.",
                                    chrono::Local::now().format("%Y-%m-%d %H:%M"),
                                    bad_ticks
                                );
                                last_warn_at = now;
                            }
                        }
                        let did_work = result.reclaimed > 0
                            || !result.crashed.is_empty()
                            || !result.timed_out.is_empty()
                            || result.promoted > 0
                            || !result.spawned.is_empty()
                            || !result.auto_blocked.is_empty();
                        if verbose && did_work {
                            eprintln!(
                                "[{}] reclaimed={} crashed={} timed_out={} promoted={} spawned={} auto_blocked={}",
                                chrono::Local::now().format("%Y-%m-%d %H:%M"),
                                result.reclaimed,
                                result.crashed.len(),
                                result.timed_out.len(),
                                result.promoted,
                                result.spawned.len(),
                                result.auto_blocked.len(),
                            );
                        }
                    }
                    Err(error) => {
                        let now = chrono::Local::now().timestamp();
                        if now - last_error_at >= 30 {
                            eprintln!(
                                "[{}] ERROR dispatcher tick failed: {}",
                                chrono::Local::now().format("%Y-%m-%d %H:%M"),
                                error
                            );
                            last_error_at = now;
                        }
                    }
                }

                let mut slept = 0.0_f64;
                while running.load(Ordering::SeqCst) && slept < interval {
                    let chunk = (interval - slept).min(1.0);
                    sleep(Duration::from_secs_f64(chunk));
                    slept += chunk;
                }
            }
            eprintln!("(dispatcher stopped)");
        }
        KanbanCommand::Run { id, dry_run } => {
            let task_id = id.trim();
            if task_id.is_empty() {
                return Err("id must not be empty".into());
            }
            let result = run_kanban_task(context, config, task_id, dry_run)?;
            println!("task_id={}", result.task_id);
            println!("assignee={}", result.assignee);
            println!("workspace={}", result.workspace_path);
            println!("dry_run={}", result.dry_run);
            println!(
                "pid={}",
                result
                    .pid
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            );
        }
    }
    Ok(())
}

pub(crate) fn disabled_memory_toolsets(memory: &hermes_core::MemoryConfig) -> Option<Vec<String>> {
    (!memory.any_enabled()).then(|| vec![String::from("memory")])
}

fn print_kv(key: &str, value: PathBuf) {
    println!("{key}={}", value.display());
}

fn validate_pagination(limit: i64, offset: i64) -> Result<(i64, i64), Box<dyn Error>> {
    if limit < 0 {
        return Err("limit must be non-negative".into());
    }
    if offset < 0 {
        return Err("offset must be non-negative".into());
    }
    Ok((limit, offset))
}

fn resolve_existing_session_id(
    session_store: &hermes_core::SessionStore,
    candidate: &str,
) -> Result<String, Box<dyn Error>> {
    if candidate.trim().is_empty() {
        return Err("session id cannot be empty".into());
    }
    session_store
        .resolve_session_id(candidate)?
        .ok_or_else(|| format!("session '{candidate}' not found").into())
}

fn write_session_exports(
    output: &str,
    exports: &[hermes_core::ExportedSession],
) -> Result<(), Box<dyn Error>> {
    if output == "-" {
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        for export in exports {
            serde_json::to_writer(&mut writer, export)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        return Ok(());
    }

    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    for export in exports {
        serde_json::to_writer(&mut writer, export)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    println!("exported={}", exports.len());
    println!("output={output}");
    Ok(())
}

fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(false);
    }
    let answer = line.trim();
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}

pub(crate) fn run_clarify_prompt(
    question: &str,
    choices: Option<&[String]>,
) -> Result<String, String> {
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

fn remove_session_files(sessions_dir: &Path, session_id: &str) {
    for suffix in [".json", ".jsonl"] {
        let path = sessions_dir.join(format!("{session_id}{suffix}"));
        let _ = fs::remove_file(path);
    }
    if let Ok(entries) = fs::read_dir(sessions_dir) {
        let prefix = format!("request_dump_{session_id}_");
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with(&prefix) && name.ends_with(".json") {
                let _ = fs::remove_file(path);
            }
        }
    }
}

fn emit_warnings(env_report: &EnvLoadReport, config: &LoadedConfig) {
    for warning in &env_report.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "hermes_cli", "{warning}");
    }
    for warning in &config.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "hermes_cli", "{warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cron_cli_parses_python_compatible_crud_commands() {
        let cli = Cli::try_parse_from([
            "hermes",
            "cron",
            "add",
            "30m",
            "check status",
            "--name",
            "health",
            "--skill",
            "ops",
            "--script",
            "health.py",
            "--workdir",
            "/tmp",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Cron {
                command: Some(CronCommand::Create(args)),
            }) => {
                assert_eq!(args.schedule, "30m");
                assert_eq!(args.prompt.as_deref(), Some("check status"));
                assert_eq!(args.name.as_deref(), Some("health"));
                assert_eq!(args.skills, vec![String::from("ops")]);
                assert_eq!(args.script.as_deref(), Some("health.py"));
                assert_eq!(args.workdir.as_deref(), Some("/tmp"));
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "cron", "rm", "cron_123"]).unwrap();
        match cli.command {
            Some(Command::Cron {
                command: Some(CronCommand::Remove { job_id }),
            }) => assert_eq!(job_id, "cron_123"),
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn cron_tool_result_creates_and_lists_jobs_in_profile_home() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home));
        context.ensure_hermes_home().unwrap();

        let created = cron_tool_result(
            &context,
            json!({
                "action": "create",
                "schedule": "30m",
                "prompt": "Summarize deployment health",
                "name": "deploy-health",
                "skills": ["ops"],
            }),
        )
        .unwrap();
        assert_eq!(created["success"], JsonValue::Bool(true));
        assert_eq!(
            created["name"],
            JsonValue::String(String::from("deploy-health"))
        );

        let listed = cron_tool_result(
            &context,
            json!({
                "action": "list",
                "include_disabled": true,
            }),
        )
        .unwrap();
        assert_eq!(listed["count"], JsonValue::from(1));
        assert_eq!(
            listed["jobs"][0]["name"],
            JsonValue::String(String::from("deploy-health"))
        );
        assert_eq!(
            listed["jobs"][0]["skills"][0],
            JsonValue::String(String::from("ops"))
        );
    }
}
