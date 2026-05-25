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

use std::collections::BTreeMap;
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

use chrono::TimeZone;
use clap::{Args, Parser, Subcommand};
use hermes_core::{
    DelegateExecutor, EnvLoadReport, HermesContext, KanbanCreateTaskInput, KanbanDispatchOptions,
    KanbanTaskQuery, LoadedConfig, LoggingMode, LoggingSetup, ModelOverrides, ToolRuntime,
    VALID_KANBAN_STATUSES, VALID_WORKSPACE_KINDS, add_comment, add_notify_sub, archive_task,
    assign_task, block_task, board_stats, build_worker_context, claim_task, complete_task,
    create_kanban_board, create_task, current_kanban_board, dispatch_kanban_once,
    edit_completed_task_result, gc_events, gc_worker_logs, get_task, get_tool_definitions,
    handle_cronjob, heartbeat_worker, is_container, is_wsl, kanban_db_path_for_home,
    kanban_has_spawnable_ready, kanban_task_detail, known_assignees, link_tasks, list_events,
    list_kanban_boards, list_notify_subs, list_runs, list_tasks, open_kanban_db, read_worker_log,
    reassign_task, reclaim_task, release_stale_claims, remove_kanban_board, remove_notify_sub,
    rename_kanban_board, run_cron_job_now, run_due_cron_jobs, run_kanban_task,
    set_current_kanban_board, unblock_task, unlink_tasks,
};
use rusqlite::Connection;
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
        #[arg(long)]
        board: Option<String>,
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
    Init,
    Boards {
        #[command(subcommand)]
        command: Option<KanbanBoardsCommand>,
    },
    Create {
        title: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long = "parent")]
        parents: Vec<String>,
        #[arg(long, default_value = "scratch")]
        workspace: String,
        #[arg(long)]
        tenant: Option<String>,
        #[arg(long, default_value_t = 0)]
        priority: i64,
        #[arg(long)]
        triage: bool,
        #[arg(long = "idempotency-key")]
        idempotency_key: Option<String>,
        #[arg(long = "max-runtime")]
        max_runtime: Option<String>,
        #[arg(long = "created-by")]
        created_by: Option<String>,
        #[arg(long = "skill")]
        skills: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    #[command(alias = "ls")]
    List {
        #[arg(long, conflicts_with = "assignee")]
        mine: bool,
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        tenant: Option<String>,
        #[arg(long)]
        archived: bool,
        #[arg(long)]
        json: bool,
    },
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    Assign {
        id: String,
        profile: String,
    },
    Reclaim {
        id: String,
        #[arg(long)]
        reason: Option<String>,
    },
    Reassign {
        id: String,
        profile: String,
        #[arg(long)]
        reclaim: bool,
        #[arg(long)]
        reason: Option<String>,
    },
    Link {
        parent_id: String,
        child_id: String,
    },
    Unlink {
        parent_id: String,
        child_id: String,
    },
    Claim {
        id: String,
        #[arg(long)]
        ttl: Option<i64>,
    },
    Comment {
        id: String,
        text: Vec<String>,
        #[arg(long)]
        author: Option<String>,
    },
    Complete {
        ids: Vec<String>,
        #[arg(long)]
        summary: Option<String>,
        #[arg(long)]
        result: Option<String>,
        #[arg(long)]
        metadata: Option<String>,
    },
    Edit {
        id: String,
        #[arg(long)]
        result: String,
        #[arg(long)]
        summary: Option<String>,
        #[arg(long)]
        metadata: Option<String>,
    },
    Archive {
        ids: Vec<String>,
    },
    Block {
        task_id: String,
        reason: Vec<String>,
        #[arg(long = "ids", num_args = 1..)]
        ids: Vec<String>,
    },
    Unblock {
        ids: Vec<String>,
    },
    Heartbeat {
        id: String,
        #[arg(long)]
        note: Option<String>,
    },
    Context {
        id: String,
    },
    Runs {
        id: String,
        #[arg(long)]
        json: bool,
    },
    Stats {
        #[arg(long)]
        json: bool,
    },
    NotifySubscribe {
        id: String,
        #[arg(long)]
        platform: String,
        #[arg(long = "chat-id")]
        chat_id: String,
        #[arg(long = "thread-id")]
        thread_id: Option<String>,
        #[arg(long = "user-id")]
        user_id: Option<String>,
    },
    NotifyList {
        task_id: Option<String>,
        #[arg(long)]
        json: bool,
    },
    NotifyUnsubscribe {
        id: String,
        #[arg(long)]
        platform: String,
        #[arg(long = "chat-id")]
        chat_id: String,
        #[arg(long = "thread-id")]
        thread_id: Option<String>,
    },
    Log {
        id: String,
        #[arg(long = "tail", alias = "tail-bytes")]
        tail_bytes: Option<usize>,
    },
    Gc {
        #[arg(long = "event-retention-days", default_value_t = 30)]
        event_retention_days: i64,
        #[arg(long = "log-retention-days", default_value_t = 30)]
        log_retention_days: i64,
    },
    Assignees {
        #[arg(long)]
        json: bool,
    },
    Tail {
        id: String,
        #[arg(long, default_value_t = 1.0)]
        interval: f64,
    },
    Watch {
        #[arg(long)]
        assignee: Option<String>,
        #[arg(long)]
        tenant: Option<String>,
        #[arg(long)]
        kinds: Option<String>,
        #[arg(long, default_value_t = 0.5)]
        interval: f64,
    },
    #[command(alias = "diag")]
    Diagnostics {
        #[arg(long)]
        severity: Option<String>,
        #[arg(long = "task")]
        task_id: Option<String>,
        #[arg(long)]
        json: bool,
    },
    #[command(alias = "tick")]
    Dispatch {
        #[arg(long)]
        dry_run: bool,
        #[arg(long = "max")]
        max_spawn: Option<usize>,
        #[arg(long)]
        failure_limit: Option<i64>,
        #[arg(long)]
        json: bool,
    },
    Daemon {
        #[arg(long, default_value_t = 60.0)]
        interval: f64,
        #[arg(long = "max")]
        max_spawn: Option<usize>,
        #[arg(long)]
        failure_limit: Option<i64>,
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long, short = 'v')]
        verbose: bool,
        #[arg(long)]
        force: bool,
    },
    Run {
        id: String,
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand, Debug)]
enum KanbanBoardsCommand {
    #[command(alias = "ls")]
    List {
        #[arg(long)]
        json: bool,
        #[arg(long = "all")]
        all: bool,
    },
    #[command(alias = "new")]
    Create {
        slug: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        icon: Option<String>,
        #[arg(long)]
        color: Option<String>,
        #[arg(long)]
        switch: bool,
    },
    #[command(alias = "use")]
    Switch {
        slug: String,
    },
    #[command(alias = "current")]
    Show,
    Rename {
        slug: String,
        name: String,
    },
    #[command(name = "rm", alias = "remove", alias = "delete")]
    Remove {
        slug: String,
        #[arg(long = "delete")]
        delete: bool,
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
        Command::Kanban { board, command } => print_kanban(&context, &config, board, command)?,
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
    board: Option<String>,
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

    struct BoardEnvGuard {
        prior: Option<std::ffi::OsString>,
    }

    impl BoardEnvGuard {
        fn set(board: Option<&str>) -> Result<Option<Self>, Box<dyn Error>> {
            let Some(board) = board else {
                return Ok(None);
            };
            let trimmed = board.trim();
            if trimmed.is_empty() {
                return Err("board must not be empty".into());
            }
            let prior = std::env::var_os("HERMES_KANBAN_BOARD");
            unsafe { std::env::set_var("HERMES_KANBAN_BOARD", trimmed) };
            Ok(Some(Self { prior }))
        }
    }

    impl Drop for BoardEnvGuard {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(value) => unsafe { std::env::set_var("HERMES_KANBAN_BOARD", value) },
                None => unsafe { std::env::remove_var("HERMES_KANBAN_BOARD") },
            }
        }
    }

    let _board_env_guard = BoardEnvGuard::set(board.as_deref())?;

    match command {
        KanbanCommand::Init => {
            let _conn = open_kanban_db(&context.hermes_home())?;
            let path = kanban_db_path_for_home(&context.hermes_home())?;
            println!("Kanban DB initialized at {}", path.display());
            println!();
            let profiles = discover_kanban_profiles(context);
            if profiles.is_empty() {
                println!("No profiles found under ~/.hermes/profiles/.");
                println!("Create one with `hermes -p <name> setup` before assigning tasks.");
            } else {
                println!(
                    "Discovered {} profile(s) on disk; any of these can be an --assignee:",
                    profiles.len()
                );
                for profile in profiles {
                    println!("  {profile}");
                }
            }
            println!();
            println!("Next step: start the gateway so ready tasks actually get picked up.");
            println!("  hermes gateway start");
            println!();
            println!(
                "The gateway hosts an embedded dispatcher that ticks every 60 seconds\nby default (config: kanban.dispatch_interval_seconds). Without a\nrunning gateway, tasks stay in 'ready' forever."
            );
        }
        KanbanCommand::Boards { command } => {
            let board_command = command.unwrap_or(KanbanBoardsCommand::List {
                json: false,
                all: false,
            });
            match board_command {
                KanbanBoardsCommand::List { json, all } => {
                    let boards = list_kanban_boards(&context.hermes_home(), all)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&boards)?);
                    } else if boards.is_empty() {
                        println!("No boards");
                    } else {
                        for board in boards {
                            let current = if board.current { "*" } else { " " };
                            println!(
                                "{current} {}  ready={} blocked={} total={}  {}",
                                board.slug,
                                board.ready_count,
                                board.blocked_count,
                                board.task_count,
                                board.name
                            );
                        }
                    }
                }
                KanbanBoardsCommand::Create {
                    slug,
                    name,
                    description,
                    icon,
                    color,
                    switch,
                } => {
                    let board = create_kanban_board(
                        &context.hermes_home(),
                        &slug,
                        name.as_deref(),
                        description.as_deref(),
                        icon.as_deref(),
                        color.as_deref(),
                    )?;
                    if switch {
                        set_current_kanban_board(&context.hermes_home(), &board.slug)?;
                    }
                    println!("Created board: {}", board.slug);
                    if switch {
                        println!("current={}", board.slug);
                    }
                }
                KanbanBoardsCommand::Switch { slug } => {
                    if !hermes_core::kanban_board_exists(&context.hermes_home(), &slug)? {
                        return Err(format!("board {slug:?} does not exist").into());
                    }
                    set_current_kanban_board(&context.hermes_home(), &slug)?;
                    println!("{slug}");
                }
                KanbanBoardsCommand::Show => {
                    println!("{}", current_kanban_board(&context.hermes_home())?);
                }
                KanbanBoardsCommand::Rename { slug, name } => {
                    let board = rename_kanban_board(&context.hermes_home(), &slug, &name)?;
                    println!("Renamed board {} -> {}", board.slug, board.name);
                }
                KanbanBoardsCommand::Remove { slug, delete } => {
                    let result = remove_kanban_board(&context.hermes_home(), &slug, !delete)?;
                    println!("{} {}", result.action, result.slug);
                    if let Some(path) = result.new_path {
                        println!("path={path}");
                    }
                }
            }
        }
        KanbanCommand::Create {
            title,
            body,
            assignee,
            parents,
            workspace,
            tenant,
            priority,
            triage,
            idempotency_key,
            max_runtime,
            created_by,
            skills,
            json,
        } => {
            let (workspace_kind, workspace_path) = parse_kanban_workspace(&workspace)?;
            let max_runtime_seconds = max_runtime
                .as_deref()
                .map(parse_kanban_runtime_seconds)
                .transpose()?;
            let mut conn = open_kanban_db(&context.hermes_home())?;
            let task_id = create_task(
                &mut conn,
                KanbanCreateTaskInput {
                    title,
                    body,
                    assignee,
                    parents,
                    tenant,
                    priority,
                    workspace_kind,
                    workspace_path,
                    triage,
                    idempotency_key,
                    max_runtime_seconds,
                    skills: (!skills.is_empty()).then_some(skills),
                    created_by: created_by.unwrap_or_else(|| String::from("user")),
                },
            )?;
            let task = get_task(&conn, &task_id)?
                .ok_or_else(|| format!("created task {task_id} disappeared"))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&task)?);
            } else {
                println!(
                    "Created {}  ({}, assignee={})",
                    task.id,
                    task.status,
                    task.assignee.as_deref().unwrap_or("-")
                );
            }
        }
        KanbanCommand::List {
            mine,
            assignee,
            status,
            tenant,
            archived,
            json,
        } => {
            if let Some(status) = status.as_deref()
                && !VALID_KANBAN_STATUSES.contains(&status)
            {
                return Err(
                    format!("status must be one of {}", VALID_KANBAN_STATUSES.join(", ")).into(),
                );
            }
            let assignee = if mine {
                Some(kanban_cli_profile_name(context))
            } else {
                assignee
            };
            let mut conn = open_kanban_db(&context.hermes_home())?;
            let _ = release_stale_claims(&mut conn);
            let _ = hermes_core::recompute_ready(&mut conn);
            let tasks = list_tasks(
                &conn,
                &KanbanTaskQuery {
                    assignee,
                    status,
                    tenant,
                    include_archived: archived,
                },
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&tasks)?);
            } else if tasks.is_empty() {
                println!("No tasks");
            } else {
                for task in tasks {
                    println!(
                        "{}  {:8}  {:16}  {}",
                        task.id,
                        task.status,
                        task.assignee.as_deref().unwrap_or("(unassigned)"),
                        task.title
                    );
                }
            }
        }
        KanbanCommand::Show { id, json } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let detail =
                kanban_task_detail(&conn, &id)?.ok_or_else(|| format!("task {id} not found"))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&detail)?);
            } else {
                println!("{}  {}", detail.task.id, detail.task.title);
                println!("status={}", detail.task.status);
                println!(
                    "assignee={}",
                    detail.task.assignee.as_deref().unwrap_or("(unassigned)")
                );
                if let Some(summary) = detail.latest_summary.as_deref() {
                    println!("latest_summary={summary}");
                }
                if !detail.comments.is_empty() {
                    println!("comments={}", detail.comments.len());
                }
                if !detail.runs.is_empty() {
                    println!("runs={}", detail.runs.len());
                }
            }
        }
        KanbanCommand::Assign { id, profile } => {
            let profile = (!profile.eq_ignore_ascii_case("none")).then_some(profile.as_str());
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !assign_task(&mut conn, &id, profile)? {
                return Err(format!("task {id} not found").into());
            }
            println!("Assigned {id}");
        }
        KanbanCommand::Reclaim { id, reason } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !reclaim_task(&mut conn, &id, reason.as_deref())? {
                return Err(format!("cannot reclaim {id}").into());
            }
            println!("Reclaimed {id}");
        }
        KanbanCommand::Reassign {
            id,
            profile,
            reclaim,
            reason,
        } => {
            let profile = (!profile.eq_ignore_ascii_case("none")).then_some(profile.as_str());
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !reassign_task(&mut conn, &id, profile, reclaim, reason.as_deref())? {
                return Err(format!("cannot reassign {id}").into());
            }
            println!("Reassigned {id}");
        }
        KanbanCommand::Link {
            parent_id,
            child_id,
        } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            link_tasks(&mut conn, &parent_id, &child_id)?;
            println!("Linked {parent_id} -> {child_id}");
        }
        KanbanCommand::Unlink {
            parent_id,
            child_id,
        } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !unlink_tasks(&mut conn, &parent_id, &child_id)? {
                return Err(format!("no such link: {parent_id} -> {child_id}").into());
            }
            println!("Unlinked {parent_id} -> {child_id}");
        }
        KanbanCommand::Claim { id, ttl } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            let task = claim_task(&mut conn, &id, ttl.unwrap_or(15 * 60), None)?
                .ok_or_else(|| format!("cannot claim {id}"))?;
            println!(
                "{}  status={}  workspace={}",
                task.id,
                task.status,
                task.workspace_path.as_deref().unwrap_or("")
            );
        }
        KanbanCommand::Comment { id, text, author } => {
            if text.is_empty() {
                return Err("comment text is required".into());
            }
            let mut conn = open_kanban_db(&context.hermes_home())?;
            let author = author.unwrap_or_else(|| kanban_cli_profile_name(context));
            add_comment(&mut conn, &id, &author, &text.join(" "))?;
            println!("Commented on {id}");
        }
        KanbanCommand::Complete {
            ids,
            summary,
            result,
            metadata,
        } => {
            if ids.is_empty() {
                return Err("at least one task id is required".into());
            }
            if ids.len() > 1 && (summary.is_some() || metadata.is_some()) {
                return Err("--summary / --metadata can't be used with multiple ids".into());
            }
            let metadata = metadata.as_deref().map(parse_kanban_metadata).transpose()?;
            let mut conn = open_kanban_db(&context.hermes_home())?;
            for id in ids {
                if !complete_task(
                    &mut conn,
                    &id,
                    result.as_deref(),
                    summary.as_deref(),
                    metadata.as_ref(),
                    &[],
                    None,
                )? {
                    return Err(format!("cannot complete {id}").into());
                }
                println!("Completed {id}");
            }
        }
        KanbanCommand::Edit {
            id,
            result,
            summary,
            metadata,
        } => {
            let metadata = metadata.as_deref().map(parse_kanban_metadata).transpose()?;
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !edit_completed_task_result(
                &mut conn,
                &id,
                &result,
                summary.as_deref(),
                metadata.as_ref(),
            )? {
                return Err(format!("cannot edit {id} (unknown id or task is not done)").into());
            }
            println!("Edited {id}");
        }
        KanbanCommand::Archive { ids } => {
            if ids.is_empty() {
                return Err("at least one task id is required".into());
            }
            let mut conn = open_kanban_db(&context.hermes_home())?;
            for id in ids {
                if !archive_task(&mut conn, &id)? {
                    return Err(format!("cannot archive {id}").into());
                }
                println!("Archived {id}");
            }
        }
        KanbanCommand::Block {
            task_id,
            reason,
            ids,
        } => {
            let mut all_ids = Vec::with_capacity(ids.len() + 1);
            all_ids.push(task_id);
            all_ids.extend(ids);
            let author = kanban_cli_profile_name(context);
            let reason = reason.join(" ");
            let reason = reason.trim().to_string();
            if all_ids.is_empty() {
                return Err("at least one task id is required".into());
            }
            let reason_value = (!reason.is_empty()).then_some(reason.as_str());
            let mut conn = open_kanban_db(&context.hermes_home())?;
            for id in all_ids {
                if let Some(reason) = reason_value {
                    add_comment(&mut conn, &id, &author, &format!("BLOCKED: {reason}"))?;
                }
                if !block_task(&mut conn, &id, reason_value.unwrap_or("blocked"), None)? {
                    return Err(format!("cannot block {id}").into());
                }
                if let Some(reason) = reason_value {
                    println!("Blocked {id}: {reason}");
                } else {
                    println!("Blocked {id}");
                }
            }
        }
        KanbanCommand::Unblock { ids } => {
            if ids.is_empty() {
                return Err("at least one task id is required".into());
            }
            let mut conn = open_kanban_db(&context.hermes_home())?;
            for id in ids {
                if !unblock_task(&mut conn, &id)? {
                    return Err(format!("cannot unblock {id}").into());
                }
                println!("Unblocked {id}");
            }
        }
        KanbanCommand::Heartbeat { id, note } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !heartbeat_worker(&mut conn, &id, note.as_deref(), None)? {
                return Err(format!("cannot heartbeat {id}").into());
            }
            println!("Heartbeat recorded for {id}");
        }
        KanbanCommand::Context { id } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            println!("{}", build_worker_context(&conn, &id)?);
        }
        KanbanCommand::Runs { id, json } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let runs = list_runs(&conn, &id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&runs)?);
            } else if runs.is_empty() {
                println!("(no runs yet for {id})");
            } else {
                for run in runs {
                    println!(
                        "{}  {}  {}",
                        run.id,
                        run.outcome.as_deref().unwrap_or(&run.status),
                        run.summary.as_deref().unwrap_or("")
                    );
                }
            }
        }
        KanbanCommand::Stats { json } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let stats = board_stats(&conn)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                for (status, count) in stats.by_status {
                    println!("{status}={count}");
                }
                if let Some(age) = stats.oldest_ready_age_seconds {
                    println!("oldest_ready_age_seconds={age}");
                }
            }
        }
        KanbanCommand::NotifySubscribe {
            id,
            platform,
            chat_id,
            thread_id,
            user_id,
        } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            add_notify_sub(
                &mut conn,
                &id,
                &platform,
                &chat_id,
                thread_id.as_deref(),
                user_id.as_deref(),
            )?;
            println!("Subscribed {id}");
        }
        KanbanCommand::NotifyList { task_id, json } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let subs = list_notify_subs(&conn, task_id.as_deref())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&subs)?);
            } else if subs.is_empty() {
                println!("No subscriptions");
            } else {
                for sub in subs {
                    println!(
                        "{}  {}  {}  {}",
                        sub.task_id, sub.platform, sub.chat_id, sub.thread_id
                    );
                }
            }
        }
        KanbanCommand::NotifyUnsubscribe {
            id,
            platform,
            chat_id,
            thread_id,
        } => {
            let mut conn = open_kanban_db(&context.hermes_home())?;
            if !remove_notify_sub(&mut conn, &id, &platform, &chat_id, thread_id.as_deref())? {
                return Err(format!("no such subscription for {id}").into());
            }
            println!("Unsubscribed {id}");
        }
        KanbanCommand::Log { id, tail_bytes } => {
            match read_worker_log(&context.hermes_home(), &id, tail_bytes)? {
                Some(contents) => print!("{contents}"),
                None => println!("No log for {id}"),
            }
        }
        KanbanCommand::Gc {
            event_retention_days,
            log_retention_days,
        } => {
            if event_retention_days < 0 || log_retention_days < 0 {
                return Err("retention days must be non-negative".into());
            }
            let mut conn = open_kanban_db(&context.hermes_home())?;
            let removed_events = gc_events(&mut conn, event_retention_days * 24 * 3600)?;
            let removed_logs =
                gc_worker_logs(&context.hermes_home(), log_retention_days * 24 * 3600)?;
            println!("removed_events={removed_events}");
            println!("removed_logs={removed_logs}");
        }
        KanbanCommand::Assignees { json } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let assignees = known_assignees(&conn, context)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&assignees)?);
            } else if assignees.is_empty() {
                println!("No assignees");
            } else {
                for assignee in assignees {
                    let on_disk = if assignee.on_disk { "disk" } else { "board" };
                    println!(
                        "{}  {:4}  {}",
                        assignee.name,
                        on_disk,
                        serde_json::to_string(&assignee.counts)?
                    );
                }
            }
        }
        KanbanCommand::Tail { id, interval } => {
            if interval <= 0.0 {
                return Err("interval must be positive".into());
            }
            println!("Tailing events for {id}. Ctrl-C to stop.");
            let running = Arc::new(AtomicBool::new(true));
            let signal_flag = Arc::clone(&running);
            ctrlc::set_handler(move || {
                signal_flag.store(false, Ordering::SeqCst);
            })?;
            let mut last_id = 0_i64;
            while running.load(Ordering::SeqCst) {
                let conn = open_kanban_db(&context.hermes_home())?;
                for event in list_events(&conn, &id)? {
                    if event.id <= last_id {
                        continue;
                    }
                    let payload = event
                        .payload
                        .as_ref()
                        .map(|value| format!(" {value}"))
                        .unwrap_or_default();
                    println!(
                        "[{}] {}{}",
                        chrono::Local
                            .timestamp_opt(event.created_at, 0)
                            .single()
                            .map(|value| value.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| event.created_at.to_string()),
                        event.kind,
                        payload,
                    );
                    last_id = event.id;
                }
                drop(conn);
                sleep(Duration::from_secs_f64(interval.max(0.1)));
            }
            println!("(stopped)");
        }
        KanbanCommand::Watch {
            assignee,
            tenant,
            kinds,
            interval,
        } => {
            if interval <= 0.0 {
                return Err("interval must be positive".into());
            }
            let kinds = kinds.map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .collect::<std::collections::HashSet<_>>()
            });
            let running = Arc::new(AtomicBool::new(true));
            let signal_flag = Arc::clone(&running);
            ctrlc::set_handler(move || {
                signal_flag.store(false, Ordering::SeqCst);
            })?;
            println!("Watching kanban events. Ctrl-C to stop.");
            let conn = open_kanban_db(&context.hermes_home())?;
            let mut cursor: i64 =
                conn.query_row("SELECT COALESCE(MAX(id), 0) FROM task_events", [], |row| {
                    row.get(0)
                })?;
            drop(conn);
            while running.load(Ordering::SeqCst) {
                let conn = open_kanban_db(&context.hermes_home())?;
                let mut stmt = conn.prepare(
                    "SELECT e.id, e.task_id, e.kind, e.payload, e.created_at, t.assignee, t.tenant
                       FROM task_events e
                  LEFT JOIN tasks t ON t.id = e.task_id
                      WHERE e.id > ?
                      ORDER BY e.id ASC
                      LIMIT 200",
                )?;
                let rows = stmt.query_map([cursor], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                    ))
                })?;
                for row in rows {
                    let (id, task_id, kind, payload, created_at, row_assignee, row_tenant) = row?;
                    cursor = cursor.max(id);
                    if let Some(filter) = kinds.as_ref()
                        && !filter.contains(&kind)
                    {
                        continue;
                    }
                    if assignee
                        .as_deref()
                        .is_some_and(|value| row_assignee.as_deref() != Some(value))
                    {
                        continue;
                    }
                    if tenant
                        .as_deref()
                        .is_some_and(|value| row_tenant.as_deref() != Some(value))
                    {
                        continue;
                    }
                    println!(
                        "[{}] {:10} {:18} (@{}){}",
                        chrono::Local
                            .timestamp_opt(created_at, 0)
                            .single()
                            .map(|value| value.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| created_at.to_string()),
                        task_id,
                        kind,
                        row_assignee.as_deref().unwrap_or("-"),
                        payload
                            .as_deref()
                            .map(|value| format!(" {value}"))
                            .unwrap_or_default()
                    );
                }
                drop(stmt);
                drop(conn);
                sleep(Duration::from_secs_f64(interval.max(0.1)));
            }
            println!("(stopped)");
        }
        KanbanCommand::Diagnostics {
            severity,
            task_id,
            json,
        } => {
            let conn = open_kanban_db(&context.hermes_home())?;
            let mut entries = collect_kanban_diagnostics(&conn, task_id.as_deref())?;
            if let Some(severity) = severity.as_deref() {
                entries.retain(|entry| {
                    entry
                        .diagnostics
                        .iter()
                        .any(|diag| diag.severity == severity)
                });
                for entry in &mut entries {
                    entry.diagnostics.retain(|diag| diag.severity == severity);
                }
                entries.retain(|entry| !entry.diagnostics.is_empty());
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else if entries.is_empty() {
                println!("No active diagnostics on this board.");
            } else {
                let total: usize = entries.iter().map(|entry| entry.diagnostics.len()).sum();
                println!(
                    "{total} active diagnostic(s) across {} task(s):",
                    entries.len()
                );
                println!();
                for entry in entries {
                    println!(
                        "  {}  {:8}  @{:18}  {}",
                        entry.task_id,
                        entry.status,
                        entry
                            .assignee
                            .clone()
                            .unwrap_or_else(|| String::from("(unassigned)")),
                        entry.title
                    );
                    for diag in entry.diagnostics {
                        println!("    [{}] {}: {}", diag.severity, diag.kind, diag.title);
                        if !diag.data.is_empty() {
                            println!("       data: {}", serde_json::to_string(&diag.data)?);
                        }
                        for action in diag.actions.iter().filter(|action| action.suggested) {
                            println!("       -> {}", action.label);
                        }
                    }
                    println!();
                }
            }
        }
        KanbanCommand::Dispatch {
            dry_run,
            max_spawn,
            failure_limit,
            json,
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
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "reclaimed": result.reclaimed,
                        "crashed": result.crashed,
                        "timed_out": result.timed_out,
                        "auto_blocked": result.auto_blocked,
                        "promoted": result.promoted,
                        "spawned": result.spawned.iter().map(|item| json!({
                            "task_id": item.task_id,
                            "assignee": item.assignee,
                            "workspace": item.workspace_path,
                            "pid": item.pid,
                        })).collect::<Vec<_>>(),
                        "skipped_unassigned": result.skipped_unassigned,
                        "skipped_nonspawnable": result.skipped_nonspawnable,
                    }))?
                );
            } else {
                println!("Reclaimed:    {}", result.reclaimed);
                println!("Crashed:      {}", result.crashed.len());
                if !result.crashed.is_empty() {
                    println!("  {}", result.crashed.join(", "));
                }
                println!("Timed out:    {}", result.timed_out.len());
                if !result.timed_out.is_empty() {
                    println!("  {}", result.timed_out.join(", "));
                }
                println!("Auto-blocked: {}", result.auto_blocked.len());
                if !result.auto_blocked.is_empty() {
                    println!("  {}", result.auto_blocked.join(", "));
                }
                println!("Promoted:     {}", result.promoted);
                println!("Spawned:      {}", result.spawned.len());
                for item in result.spawned {
                    let dry_tag = if dry_run { " (dry)" } else { "" };
                    println!(
                        "  - {}  ->  {}  @ {}{}",
                        item.task_id,
                        item.assignee,
                        if item.workspace_path.is_empty() {
                            "-"
                        } else {
                            &item.workspace_path
                        },
                        dry_tag
                    );
                }
                if !result.skipped_unassigned.is_empty() {
                    println!(
                        "Skipped (unassigned): {}",
                        result.skipped_unassigned.join(", ")
                    );
                }
                if !result.skipped_nonspawnable.is_empty() {
                    println!(
                        "Skipped (non-spawnable assignee - terminal lane, OK): {}",
                        result.skipped_nonspawnable.join(", ")
                    );
                }
            }
        }
        KanbanCommand::Daemon {
            interval,
            max_spawn,
            failure_limit,
            pidfile,
            verbose,
            force,
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
            if !force {
                return Err(
                    "hermes kanban daemon is deprecated; rerun with --force or use `hermes gateway start`"
                        .into(),
                );
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
                "Kanban dispatcher running STANDALONE via --force (interval={}s, pid={}). Ctrl-C to stop.",
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

fn discover_kanban_profiles(context: &HermesContext) -> Vec<String> {
    let mut profiles = Vec::new();
    let Ok(entries) = fs::read_dir(context.profiles_root()) else {
        return profiles;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                profiles.push(trimmed.to_string());
            }
        }
    }
    profiles.sort();
    profiles.dedup();
    profiles
}

fn kanban_cli_profile_name(context: &HermesContext) -> String {
    std::env::var("HERMES_PROFILE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            let profile = context.current_profile_name();
            if profile == "default" {
                String::from("user")
            } else {
                profile
            }
        })
}

fn parse_kanban_metadata(value: &str) -> Result<JsonValue, Box<dyn Error>> {
    let parsed: JsonValue = serde_json::from_str(value)?;
    if parsed.is_object() {
        Ok(parsed)
    } else {
        Err("metadata must be a JSON object".into())
    }
}

fn parse_kanban_workspace(value: &str) -> Result<(String, Option<String>), Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "scratch" || trimmed == "worktree" {
        return Ok((
            if trimmed == "worktree" {
                String::from("worktree")
            } else {
                String::from("scratch")
            },
            None,
        ));
    }
    if let Some(path) = trimmed.strip_prefix("dir:") {
        let expanded = path.trim();
        if expanded.is_empty() {
            return Err("dir: workspace requires a path".into());
        }
        return Ok((String::from("dir"), Some(expanded.to_string())));
    }
    if VALID_WORKSPACE_KINDS.contains(&trimmed) {
        return Ok((trimmed.to_string(), None));
    }
    Err(format!(
        "workspace must be one of {} or dir:<path>",
        VALID_WORKSPACE_KINDS.join(", ")
    )
    .into())
}

fn parse_kanban_runtime_seconds(value: &str) -> Result<i64, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("max runtime must not be empty".into());
    }
    let split_at = trimmed
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, unit) = trimmed.split_at(split_at);
    let amount: i64 = digits
        .parse()
        .map_err(|_| format!("invalid max runtime {trimmed:?}"))?;
    if amount <= 0 {
        return Err("max runtime must be positive".into());
    }
    let seconds = match unit {
        "" | "s" => amount,
        "m" => amount * 60,
        "h" => amount * 60 * 60,
        "d" => amount * 60 * 60 * 24,
        _ => return Err(format!("invalid max runtime unit in {trimmed:?}").into()),
    };
    Ok(seconds)
}

#[derive(Debug, Clone, serde::Serialize)]
struct KanbanDiagnosticAction {
    kind: String,
    label: String,
    payload: BTreeMap<String, JsonValue>,
    suggested: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct KanbanDiagnostic {
    kind: String,
    severity: String,
    title: String,
    detail: String,
    actions: Vec<KanbanDiagnosticAction>,
    data: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct KanbanDiagnosticEntry {
    task_id: String,
    title: String,
    status: String,
    assignee: Option<String>,
    diagnostics: Vec<KanbanDiagnostic>,
}

fn diagnostic_action(
    kind: &str,
    label: impl Into<String>,
    payload: BTreeMap<String, JsonValue>,
    suggested: bool,
) -> KanbanDiagnosticAction {
    KanbanDiagnosticAction {
        kind: kind.to_string(),
        label: label.into(),
        payload,
        suggested,
    }
}

fn generic_diagnostic_actions(
    task_id: &str,
    running: bool,
    assignee: Option<&str>,
) -> Vec<KanbanDiagnosticAction> {
    let mut actions = Vec::new();
    if running {
        let mut payload = BTreeMap::new();
        payload.insert(String::from("task_id"), json!(task_id));
        actions.push(diagnostic_action(
            "reclaim",
            format!("Reclaim {task_id}"),
            payload,
            true,
        ));
    }
    let mut payload = BTreeMap::new();
    payload.insert(String::from("task_id"), json!(task_id));
    actions.push(diagnostic_action(
        "comment",
        format!("Comment on {task_id}"),
        payload,
        !running,
    ));
    if let Some(assignee) = assignee {
        let mut payload = BTreeMap::new();
        payload.insert(String::from("task_id"), json!(task_id));
        payload.insert(String::from("assignee"), json!(assignee));
        actions.push(diagnostic_action(
            "reassign",
            format!("Reassign {task_id}"),
            payload,
            false,
        ));
    }
    actions
}

fn collect_kanban_diagnostics(
    conn: &Connection,
    task_id: Option<&str>,
) -> Result<Vec<KanbanDiagnosticEntry>, Box<dyn Error>> {
    let tasks = if let Some(task_id) = task_id {
        match get_task(conn, task_id)? {
            Some(task) if task.status != "archived" => vec![task],
            Some(_) => Vec::new(),
            None => return Err(format!("no such task: {task_id}").into()),
        }
    } else {
        list_tasks(
            conn,
            &KanbanTaskQuery {
                include_archived: false,
                ..KanbanTaskQuery::default()
            },
        )?
    };

    let mut entries = Vec::new();
    for task in tasks {
        let Some(detail) = kanban_task_detail(conn, &task.id)? else {
            continue;
        };
        let mut diagnostics = Vec::new();
        let task_id = detail.task.id.clone();
        let running = detail.task.status == "running";

        let latest_completed_at = detail
            .events
            .iter()
            .filter(|event| event.kind == "completed" || event.kind == "edited")
            .map(|event| event.created_at)
            .max()
            .unwrap_or(0);
        if let Some(event) = detail.events.iter().rev().find(|event| {
            event.kind == "completion_blocked_hallucination"
                && event.created_at >= latest_completed_at
        }) {
            let phantom = event
                .payload
                .as_ref()
                .and_then(|payload| payload.get("phantom_cards"))
                .cloned()
                .unwrap_or(JsonValue::Array(Vec::new()));
            let verified = event
                .payload
                .as_ref()
                .and_then(|payload| payload.get("verified_cards"))
                .cloned()
                .unwrap_or(JsonValue::Array(Vec::new()));
            let mut data = BTreeMap::new();
            data.insert(String::from("phantom_ids"), phantom.clone());
            data.insert(String::from("verified_ids"), verified);
            diagnostics.push(KanbanDiagnostic {
                kind: String::from("hallucinated_cards"),
                severity: String::from("error"),
                title: String::from("Completion claimed cards that do not exist"),
                detail: String::from(
                    "A completion reported created_cards that were missing or not attributable to this worker. Remove the phantom ids or create the follow-up tasks explicitly before completing again.",
                ),
                actions: generic_diagnostic_actions(
                    &task_id,
                    running,
                    detail.task.assignee.as_deref(),
                ),
                data,
            });
        }

        if detail.task.consecutive_failures >= 3 {
            let most_recent_outcome =
                detail
                    .runs
                    .iter()
                    .rev()
                    .find_map(|run| match run.outcome.as_deref() {
                        Some("spawn_failed" | "timed_out" | "crashed") => run.outcome.clone(),
                        _ => None,
                    });
            let mut actions =
                generic_diagnostic_actions(&task_id, running, detail.task.assignee.as_deref());
            if let Some(outcome) = most_recent_outcome.as_deref() {
                if outcome == "spawn_failed"
                    && let Some(assignee) = detail.task.assignee.as_deref()
                {
                    let mut payload = BTreeMap::new();
                    payload.insert(
                        String::from("command"),
                        json!(format!("hermes -p {assignee} doctor")),
                    );
                    actions.insert(
                        0,
                        diagnostic_action(
                            "cli_hint",
                            format!("Verify profile: hermes -p {assignee} doctor"),
                            payload,
                            true,
                        ),
                    );
                } else if matches!(outcome, "timed_out" | "crashed") {
                    let mut payload = BTreeMap::new();
                    payload.insert(
                        String::from("command"),
                        json!(format!("hermes kanban log {task_id}")),
                    );
                    actions.insert(
                        0,
                        diagnostic_action(
                            "cli_hint",
                            format!("Check logs: hermes kanban log {task_id}"),
                            payload,
                            true,
                        ),
                    );
                }
            }
            let mut data = BTreeMap::new();
            data.insert(
                String::from("consecutive_failures"),
                json!(detail.task.consecutive_failures),
            );
            if let Some(outcome) = most_recent_outcome.clone() {
                data.insert(String::from("most_recent_outcome"), json!(outcome));
            }
            if let Some(error) = detail.task.last_failure_error.clone() {
                data.insert(String::from("last_error"), json!(error));
            }
            let err_snippet = detail
                .task
                .last_failure_error
                .clone()
                .unwrap_or_default()
                .chars()
                .take(160)
                .collect::<String>();
            diagnostics.push(KanbanDiagnostic {
                kind: String::from("repeated_failures"),
                severity: if detail.task.consecutive_failures >= 6 {
                    String::from("critical")
                } else {
                    String::from("error")
                },
                title: if err_snippet.is_empty() {
                    format!("Agent failure x{}", detail.task.consecutive_failures)
                } else {
                    format!(
                        "Agent failure x{}: {}",
                        detail.task.consecutive_failures, err_snippet
                    )
                },
                detail: detail
                    .task
                    .last_failure_error
                    .clone()
                    .unwrap_or_else(|| String::from("No error text was captured.")),
                actions,
                data,
            });
        }

        let mut trailing_crashes = 0_i64;
        let mut last_crash_error = None;
        for run in detail.runs.iter().rev() {
            match run.outcome.as_deref() {
                Some("crashed") => {
                    trailing_crashes += 1;
                    if last_crash_error.is_none() {
                        last_crash_error = run.error.clone();
                    }
                }
                Some("completed" | "reclaimed") => break,
                _ => {}
            }
        }
        if detail.task.consecutive_failures < 3 && trailing_crashes >= 2 {
            let mut actions =
                generic_diagnostic_actions(&task_id, running, detail.task.assignee.as_deref());
            let mut payload = BTreeMap::new();
            payload.insert(
                String::from("command"),
                json!(format!("hermes kanban log {task_id}")),
            );
            actions.insert(
                0,
                diagnostic_action(
                    "cli_hint",
                    format!("Check logs: hermes kanban log {task_id}"),
                    payload,
                    true,
                ),
            );
            let mut data = BTreeMap::new();
            data.insert(String::from("consecutive_crashes"), json!(trailing_crashes));
            if let Some(error) = last_crash_error.clone() {
                data.insert(String::from("last_error"), json!(error));
            }
            diagnostics.push(KanbanDiagnostic {
                kind: String::from("repeated_crashes"),
                severity: if trailing_crashes >= 4 {
                    String::from("critical")
                } else {
                    String::from("error")
                },
                title: format!("Agent crashed {trailing_crashes}x"),
                detail: last_crash_error
                    .unwrap_or_else(|| String::from("No error text was captured.")),
                actions,
                data,
            });
        }

        if detail.task.status == "blocked" {
            let latest_blocked = detail
                .events
                .iter()
                .filter(|event| event.kind == "blocked")
                .map(|event| event.created_at)
                .max()
                .unwrap_or(0);
            if latest_blocked > 0 {
                let cleared = detail.events.iter().any(|event| {
                    event.created_at > latest_blocked
                        && matches!(event.kind.as_str(), "commented" | "unblocked")
                });
                let age_hours = (chrono::Local::now().timestamp() - latest_blocked) / 3600;
                if !cleared && age_hours >= 24 {
                    let mut payload = BTreeMap::new();
                    payload.insert(String::from("task_id"), json!(task_id));
                    diagnostics.push(KanbanDiagnostic {
                        kind: String::from("stuck_in_blocked"),
                        severity: String::from("warning"),
                        title: format!("Task has been blocked for {age_hours}h"),
                        detail: String::from(
                            "This task has been blocked for a long time without a newer comment or unblock event.",
                        ),
                        actions: vec![diagnostic_action(
                            "comment",
                            "Add a comment / unblock the task",
                            payload,
                            true,
                        )],
                        data: {
                            let mut data = BTreeMap::new();
                            data.insert(String::from("age_hours"), json!(age_hours));
                            data
                        },
                    });
                }
            }
        }

        if !diagnostics.is_empty() {
            diagnostics.sort_by_key(|diag| match diag.severity.as_str() {
                "critical" => 0,
                "error" => 1,
                _ => 2,
            });
            entries.push(KanbanDiagnosticEntry {
                task_id: detail.task.id.clone(),
                title: detail.task.title.clone(),
                status: detail.task.status.clone(),
                assignee: detail.task.assignee.clone(),
                diagnostics,
            });
        }
    }
    Ok(entries)
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
