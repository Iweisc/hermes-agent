mod acp_cmd;
mod auth_cmd;
mod backup;
mod checkpoints_cmd;
mod claw_cmd;
mod completion;
mod config_cmd;
mod curator_cmd;
mod dashboard_cmd;
mod dashboard_server;
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
mod plugin_runtime;
mod plugins_cmd;
mod profile_cmd;
mod python_bridge;
mod setup_cmd;
mod skills_cmd;
mod skills_guard;
mod slack_cmd;
mod snapshot_cmd;
mod tools_cmd;
mod tui_cmd;
mod uninstall_cmd;
mod update_cmd;
mod webhook;
mod whatsapp_cmd;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufRead, BufWriter, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::TimeZone;
use clap::{Args, Parser, Subcommand};
use hermes_core::{
    DelegateExecutor, EnvLoadReport, HermesContext, KanbanCreateTaskInput, KanbanDispatchOptions,
    KanbanTaskQuery, LoadedConfig, LoggingMode, LoggingSetup, ModelOverrides, ToolRuntime,
    VALID_KANBAN_STATUSES, VALID_WORKSPACE_KINDS, add_comment, add_notify_sub, archive_task,
    assign_task, block_task, board_stats, build_worker_context, claim_task, complete_task,
    create_kanban_board, create_task, current_kanban_board, dispatch_kanban_once,
    edit_completed_task_result, gc_events, gc_worker_logs, get_active_auth_provider,
    get_auth_status_summary, get_task, get_tool_definitions, handle_cronjob, heartbeat_worker,
    is_container, is_wsl, kanban_db_path_for_home, kanban_has_spawnable_ready,
    kanban_task_detail, known_assignees, link_tasks, list_events, list_kanban_boards,
    list_notify_subs, list_provider_profiles, list_runs, list_tasks, load_skill_prompt_content,
    open_kanban_db, read_worker_log, reassign_task, reclaim_task, release_stale_claims,
    remove_kanban_board, remove_notify_sub, rename_kanban_board, run_cron_job_now,
    run_due_cron_jobs, run_kanban_task, set_current_kanban_board, unblock_task, unlink_tasks,
};
use rusqlite::Connection;
use serde_json::{Value as JsonValue, json};

use crate::python_bridge::{project_root, resolve_repo_python};

const CONTINUE_LATEST_SENTINEL: &str = "__hermes_continue_latest__";

const INHERITED_RELAUNCH_FLAGS: &[(&str, bool)] = &[
    ("--profile", true),
    ("-p", true),
    ("-m", true),
    ("--model", true),
    ("--provider", true),
    ("--accept-hooks", false),
    ("--skills", true),
    ("-s", true),
    ("--yolo", false),
    ("--pass-session-id", false),
    ("--ignore-user-config", false),
    ("--ignore-rules", false),
    ("--tui", false),
    ("--dev", false),
];

#[cfg(test)]
pub(crate) fn cli_test_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Args, Debug, Clone, Default)]
struct ChatArgs {
    #[arg(long = "query", global = true)]
    query: Option<String>,
    #[arg(short = 'r', long = "resume", global = true)]
    resume: Option<String>,
    #[arg(
        short = 'c',
        long = "continue",
        global = true,
        num_args = 0..=1,
        default_missing_value = CONTINUE_LATEST_SENTINEL
    )]
    continue_last: Option<String>,
    #[arg(short = 'm', long = "model", global = true)]
    model: Option<String>,
    #[arg(long = "provider", global = true)]
    provider: Option<String>,
    #[arg(long = "base-url", global = true)]
    base_url: Option<String>,
    #[arg(long = "api-key", global = true)]
    api_key: Option<String>,
    #[arg(long = "api-mode", global = true)]
    api_mode: Option<String>,
    #[arg(
        short = 's',
        long = "skills",
        global = true,
        value_delimiter = ',',
        action = clap::ArgAction::Append
    )]
    skills: Vec<String>,
    #[arg(
        long = "toolsets",
        global = true,
        value_delimiter = ',',
        action = clap::ArgAction::Append
    )]
    toolsets: Vec<String>,
    #[arg(short = 'w', long = "worktree", global = true, default_value_t = false)]
    worktree: bool,
    #[arg(long = "yolo", global = true, default_value_t = false)]
    yolo: bool,
    #[arg(long = "ignore-user-config", global = true, default_value_t = false)]
    ignore_user_config: bool,
}

#[derive(Parser, Debug)]
#[command(name = "hermes", version, about = "Hermes Rust bootstrap")]
struct Cli {
    #[arg(short = 'p', long = "profile", global = true)]
    _profile: Option<String>,
    #[command(flatten)]
    chat: ChatArgs,
    #[arg(long = "tui", global = true, default_value_t = false)]
    tui: bool,
    #[arg(long = "dev", global = true, default_value_t = false)]
    tui_dev: bool,
    #[arg(short = 'm', long = "model", global = true)]
    model: Option<String>,
    #[arg(long = "provider", global = true)]
    provider: Option<String>,
    #[arg(short = 't', long = "toolsets", global = true)]
    toolsets: Option<String>,
    #[arg(long = "resume", short = 'r', global = true)]
    resume: Option<String>,
    #[arg(
        long = "continue",
        short = 'c',
        global = true,
        num_args = 0..=1,
        default_missing_value = "__latest__"
    )]
    continue_last: Option<String>,
    #[arg(long = "query", short = 'q', global = true)]
    query: Option<String>,
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
    Approve(SlashCompatArgs),
    Deny(SlashCompatArgs),
    Fallback {
        #[command(subcommand)]
        command: Option<fallback_cmd::FallbackCommand>,
    },
    Busy(SlashCompatArgs),
    Browser(SlashCompatArgs),
    Clear(SlashCompatArgs),
    Compress(SlashCompatArgs),
    Commands(SlashCompatArgs),
    Copy(SlashCompatArgs),
    Footer(SlashCompatArgs),
    #[command(alias = "tasks")]
    Agents(SlashCompatArgs),
    #[command(alias = "bg", alias = "btw")]
    Background(SlashCompatArgs),
    #[command(alias = "fork")]
    Branch(SlashCompatArgs),
    #[command(alias = "provider")]
    Model {
        #[command(subcommand)]
        command: Option<model_cmd::ModelCommand>,
    },
    Fast(SlashCompatArgs),
    Gquota(SlashCompatArgs),
    Goal(SlashCompatArgs),
    History(SlashCompatArgs),
    Image(SlashCompatArgs),
    Indicator(SlashCompatArgs),
    #[command(alias = "reset")]
    New(SlashCompatArgs),
    Paste(SlashCompatArgs),
    Personality(SlashCompatArgs),
    #[command(alias = "q")]
    Queue(SlashCompatArgs),
    Redraw(SlashCompatArgs),
    Reasoning(SlashCompatArgs),
    Reload(SlashCompatArgs),
    #[command(alias = "reload_mcp")]
    ReloadMcp(SlashCompatArgs),
    #[command(alias = "reload_skills")]
    ReloadSkills(SlashCompatArgs),
    Rollback(SlashCompatArgs),
    Restart(gateway_cmd::GatewayServiceArgs),
    Resume(ResumeArgs),
    Retry(SlashCompatArgs),
    Save(SlashCompatArgs),
    #[command(alias = "set-home")]
    Sethome(SlashCompatArgs),
    Skin(SlashCompatArgs),
    #[command(alias = "sb")]
    Statusbar(SlashCompatArgs),
    Steer(SlashCompatArgs),
    Stop(SlashCompatArgs),
    Toolsets(SlashCompatArgs),
    Title(SlashCompatArgs),
    Undo(SlashCompatArgs),
    Usage(SlashCompatArgs),
    Verbose(SlashCompatArgs),
    Voice(SlashCompatArgs),
    Yolo(SlashCompatArgs),
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
    #[command(alias = "platforms")]
    Gateway(gateway_cmd::GatewayArgs),
    Skills {
        #[command(subcommand)]
        command: Option<skills_cmd::SkillsCommand>,
    },
    Checkpoints {
        #[command(subcommand)]
        command: Option<checkpoints_cmd::CheckpointsCommand>,
    },
    #[command(alias = "snap")]
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
    Topic(SlashCompatArgs),
    Chat {
        prompt: Vec<String>,
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
    #[command(hide = true)]
    TuiGateway(tui_cmd::TuiGatewayArgs),
    Update(update_cmd::UpdateArgs),
    Whatsapp,
    Status(StatusArgs),
}

#[derive(Subcommand, Debug)]
enum SessionsCommand {
    Browse {
        #[arg(long, default_value_t = 500)]
        limit: i64,
        #[arg(long)]
        source: Option<String>,
    },
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
    #[command(external_subcommand)]
    Compat(Vec<String>),
}

#[derive(Args, Debug, Clone)]
struct SlashCompatArgs {
    #[arg(long)]
    session: Option<String>,
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Args, Debug, Clone)]
struct ResumeArgs {
    #[arg(long, default_value_t = 50)]
    limit: i64,
    #[arg(long)]
    source: Option<String>,
    target: Vec<String>,
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

#[derive(Args, Debug, Clone, Default)]
struct StatusArgs {
    #[arg(long)]
    session: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let detected = HermesContext::detect();
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let profile_override = detected.apply_profile_override(&raw_args)?;
    let processed_args = coalesce_session_name_args(profile_override.args, known_subcommands());
    let context = match profile_override.hermes_home.clone() {
        Some(home) => detected.with_hermes_home_env(Some(home)),
        None => detected,
    };
    unsafe { std::env::set_var("HERMES_HOME", context.hermes_home()) };
    context.ensure_hermes_home()?;
    let env_report = context.load_hermes_dotenv(None)?;
    if argv_contains_flag(&processed_args, "--ignore-user-config") {
        unsafe { std::env::set_var("HERMES_IGNORE_USER_CONFIG", "1") };
    }
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
    apply_cli_runtime_env(
        &config,
        &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    );
    let argv = std::iter::once(String::from("hermes")).chain(processed_args);
    let cli = Cli::parse_from(argv);

    if cli.chat.yolo {
        unsafe { std::env::set_var("HERMES_YOLO_MODE", "1") };
    }

    if cli.tui {
        return tui_cmd::launch_tui(
            &context,
            &session_store,
            tui_cmd::TuiLaunchOptions {
                continue_last: cli.continue_last.clone(),
                model: cli.model.clone(),
                provider: cli.provider.clone(),
                query: cli.query.clone(),
                resume: cli.resume.clone(),
                toolsets: cli.toolsets.clone(),
                tui_dev: cli.tui_dev,
            },
        );
    }

    let default_chat = cli.command.is_none();
    match cli.command.unwrap_or(Command::Chat { prompt: Vec::new() }) {
        Command::Paths => print_paths(&context, &config, &logging),
        Command::Version => dump::print_version(),
        Command::Dump(args) => dump::print_dump(&context, &config, args)?,
        Command::Doctor(args) => {
            doctor::print_doctor(&context, &env_report, &config, &session_store, args)?
        }
        Command::Debug { command } => debug::print_debug(&context, &config, command)?,
        Command::Hooks { command } => hooks::print_hooks(&context, &config, command)?,
        Command::Login(args) => login_cmd::print_login(args)?,
        Command::Approve(args) => print_live_gateway_only_command(
            "approve",
            &args.args,
            "approve is only available for live pending approvals in a running gateway or TUI session.",
        )?,
        Command::Deny(args) => print_live_gateway_only_command(
            "deny",
            &args.args,
            "deny is only available for live pending approvals in a running gateway or TUI session.",
        )?,
        Command::Fallback { command } => {
            fallback_cmd::print_fallback(&context.config_path(), &config, command)?
        }
        Command::Agents(args) => print_slash_compat(&session_store, "agents", args)?,
        Command::Background(args) => print_slash_compat(&session_store, "background", args)?,
        Command::Branch(args) => print_slash_compat(&session_store, "branch", args)?,
        Command::Busy(args) => print_slash_compat(&session_store, "busy", args)?,
        Command::Browser(args) => print_slash_compat(&session_store, "browser", args)?,
        Command::Clear(args) => print_no_arg_slash_compat(&session_store, "clear", args)?,
        Command::Compress(args) => print_compress_compat(&session_store, args)?,
        Command::Commands(args) => print_slash_compat(&session_store, "commands", args)?,
        Command::Copy(args) => print_slash_compat(&session_store, "copy", args)?,
        Command::Footer(args) => print_slash_compat(&session_store, "footer", args)?,
        Command::Model { command } => model_cmd::print_model(&context, &config, command)?,
        Command::Fast(args) => print_slash_compat(&session_store, "fast", args)?,
        Command::Gquota(args) => print_slash_compat(&session_store, "gquota", args)?,
        Command::Goal(args) => print_goal_compat(&session_store, args)?,
        Command::History(args) => print_slash_compat(&session_store, "history", args)?,
        Command::Image(args) => print_image_compat(&session_store, args)?,
        Command::Indicator(args) => print_slash_compat(&session_store, "indicator", args)?,
        Command::New(args) => print_slash_compat(&session_store, "new", args)?,
        Command::Paste(args) => print_paste_compat(&session_store, args)?,
        Command::Personality(args) => print_slash_compat(&session_store, "personality", args)?,
        Command::Queue(args) => print_send_turn_compat(&session_store, "queue", args)?,
        Command::Redraw(args) => print_no_arg_slash_compat(&session_store, "redraw", args)?,
        Command::Reasoning(args) => print_slash_compat(&session_store, "reasoning", args)?,
        Command::Reload(args) => print_slash_compat(&session_store, "reload", args)?,
        Command::ReloadMcp(args) => print_slash_compat(&session_store, "reload-mcp", args)?,
        Command::ReloadSkills(args) => print_slash_compat(&session_store, "reload-skills", args)?,
        Command::Rollback(args) => print_slash_compat(&session_store, "rollback", args)?,
        Command::Restart(service) => gateway_cmd::print_gateway(
            &context,
            gateway_cmd::GatewayArgs {
                accept_hooks: false,
                command: Some(gateway_cmd::GatewayCommand::Restart(service)),
            },
        )?,
        Command::Resume(args) => print_resume(&session_store, args)?,
        Command::Retry(args) => print_retry_compat(&session_store, args)?,
        Command::Save(args) => print_slash_compat(&session_store, "save", args)?,
        Command::Sethome(args) => print_live_gateway_only_command(
            "sethome",
            &args.args,
            "sethome is only available from a running gateway chat context.",
        )?,
        Command::Skin(args) => print_slash_compat(&session_store, "skin", args)?,
        Command::Statusbar(args) => print_slash_compat(&session_store, "statusbar", args)?,
        Command::Steer(args) => print_send_turn_compat(&session_store, "steer", args)?,
        Command::Stop(args) => print_slash_compat(&session_store, "stop", args)?,
        Command::Toolsets(args) => print_slash_compat(&session_store, "toolsets", args)?,
        Command::Title(args) => print_slash_compat(&session_store, "title", args)?,
        Command::Undo(args) => print_undo_compat(&session_store, args)?,
        Command::Usage(args) => print_slash_compat(&session_store, "usage", args)?,
        Command::Verbose(args) => print_slash_compat(&session_store, "verbose", args)?,
        Command::Voice(args) => print_slash_compat(&session_store, "voice", args)?,
        Command::Yolo(args) => print_slash_compat(&session_store, "yolo", args)?,
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
        Command::Claw { command } => claw_cmd::print_claw(&context, command)?,
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
        Command::Topic(args) => print_live_gateway_only_command(
            "topic",
            &args.args,
            "topic is only available in Telegram private chats through the gateway.",
        )?,
        Command::Chat { prompt } => run_chat(
            &context,
            &config,
            &session_store,
            cli.chat.clone(),
            prompt,
            default_chat,
        )?,
        Command::Sessions { command } => print_sessions(&context, &session_store, command)?,
        Command::Cron { command } => print_cron(&context, &config, &session_store, command)?,
        Command::Logs(args) => logs::print_logs(&context, args)?,
        Command::Kanban { board, command } => print_kanban(&context, &config, board, command)?,
        Command::Tools(args) => tools_cmd::print_tools(&context, &config, args)?,
        Command::TuiGateway(_args) => tui_cmd::run_tui_gateway(&context, &config)?,
        Command::Update(args) => update_cmd::print_update(&context, args)?,
        Command::Whatsapp => whatsapp_cmd::print_whatsapp(&context)?,
        Command::Status(args) => {
            print_status(&context, &env_report, &config, &session_store, args)?
        }
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
        SessionsCommand::Browse { limit, source } => {
            let (limit, _) = validate_pagination(limit, 0)?;
            let rows = session_store.search_sessions(source.as_deref(), limit, 0)?;
            let stdin = io::stdin();
            let stdout = io::stdout();
            let mut input = stdin.lock();
            let mut output = stdout.lock();
            match browse_sessions_with_io(&rows, &mut input, &mut output)? {
                Some(session_id) => {
                    writeln!(output, "Resuming session: {session_id}")?;
                    output.flush()?;
                    launch_python_resume_session(&session_id)?;
                }
                None => {
                    writeln!(output, "Cancelled.")?;
                }
            }
        }
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
    args: StatusArgs,
) -> Result<(), Box<dyn Error>> {
    if let Some(requested) = args.session.as_deref() {
        let session_id = resolve_slash_compat_session_id(session_store, Some(requested))?;
        return launch_python_slash_command(&session_id, "status", &[]);
    }
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
    println!("note=Rust interactive chat is the default; use `hermes chat -q ...` for one-shot");
    Ok(())
}

fn run_chat(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
    chat: ChatArgs,
    prompt_parts: Vec<String>,
    default_chat: bool,
) -> Result<(), Box<dyn Error>> {
    unsafe { std::env::set_var("HERMES_SESSION_SOURCE", "cli") };
    let mut worktree_guard = if chat.worktree {
        Some(WorktreeGuard::create()?)
    } else {
        None
    };
    let active_cwd = worktree_guard
        .as_ref()
        .map(|guard| guard.path().to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    apply_cli_runtime_env(config, &active_cwd);

    let enabled_toolsets = if chat.toolsets.is_empty() {
        config.config.toolsets.clone()
    } else {
        unique_cli_strings(&chat.toolsets)
    };
    let overrides = ModelOverrides {
        model: chat.model.clone(),
        provider: chat.provider.clone(),
        base_url: chat.base_url.clone(),
        api_key: chat.api_key.clone(),
        api_mode: chat.api_mode.clone(),
    };

    let stdin_is_terminal = io::stdin().is_terminal();
    let stdout_is_terminal = io::stdout().is_terminal();
    let prompt = resolve_initial_prompt(chat.query.as_deref(), &prompt_parts)?;
    let stdin_prompt = read_prompt_from_stdin()?;
    let initial_prompt = prompt.or(stdin_prompt);

    if !has_any_chat_provider_configured(context, config, &overrides)? {
        print_chat_setup_required();
        if !stdin_is_terminal || !stdout_is_terminal {
            print_noninteractive_chat_setup_guidance();
            return Err("Hermes is not configured yet.".into());
        }
        let reply = prompt_user_input("Run setup now? [Y/n] ")?
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if matches!(reply.as_str(), "" | "y" | "yes") {
            setup_cmd::print_setup(
                context,
                setup_cmd::SetupArgs {
                    section: None,
                    non_interactive: false,
                    reset: false,
                    reconfigure: false,
                    quick: false,
                },
            )?;
            return Ok(());
        }
        println!();
        println!("You can run 'hermes setup' at any time to configure.");
        return Err("Hermes is not configured yet.".into());
    }

    let _ = skills_cmd::sync_bundled_skills(context, true);

    let mut session_hint = resolve_chat_session_hint(session_store, &chat)?;
    let runtime = build_chat_runtime(
        context,
        config,
        &enabled_toolsets,
        &overrides,
        &active_cwd,
        &chat.skills,
        worktree_guard.as_ref().map(WorktreeGuard::metadata),
    )?;

    if let Some(prompt) = initial_prompt {
        let result = execute_chat_turn(
            context,
            config,
            session_store,
            &runtime,
            &enabled_toolsets,
            &overrides,
            &prompt,
            session_hint.as_deref(),
        )?;
        println!("{}", result.final_response);
        return Ok(());
    }

    if !stdin_is_terminal || !stdout_is_terminal {
        return Err(
            "interactive chat requires a tty; pass -q/--query or pipe a prompt on stdin".into(),
        );
    }

    let runtime_model = context.resolve_model_runtime(config, &overrides)?;
    if default_chat {
        println!(
            "Hermes Rust CLI  model={}  provider={}",
            runtime_model.model, runtime_model.provider
        );
    }
    if let Some(resumed) = session_hint.as_deref() {
        println!("Resumed session: {resumed}");
    }
    if !chat.skills.is_empty() {
        println!(
            "Activated skills: {}",
            unique_cli_strings(&chat.skills).join(", ")
        );
    }
    if worktree_guard.is_some() {
        println!("Worktree: {}", active_cwd.display());
    }
    println!("Type /exit to quit.");

    loop {
        let Some(user_input) = prompt_user_input("> ")? else {
            println!();
            break;
        };
        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "/exit" | "/quit" | "exit" | "quit") {
            break;
        }
        if trimmed == "/help" {
            println!("Use /exit to quit, /help to show this message, or type a prompt.");
            continue;
        }

        let result = execute_chat_turn(
            context,
            config,
            session_store,
            &runtime,
            &enabled_toolsets,
            &overrides,
            trimmed,
            session_hint.as_deref(),
        )?;
        if let Some(next_session) = result.session_id.clone() {
            session_hint = Some(next_session);
        }
        println!();
        println!("{}", result.final_response);
        println!();
    }

    worktree_guard.take();
    Ok(())
}

fn execute_chat_turn(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
    runtime: &ToolRuntime,
    enabled_toolsets: &[String],
    overrides: &ModelOverrides,
    prompt: &str,
    session_hint: Option<&str>,
) -> Result<hermes_core::AgentTurnResult, Box<dyn Error>> {
    context
        .run_chat_completions_turn(
            config,
            prompt,
            runtime,
            Some(enabled_toolsets),
            overrides,
            session_hint,
            Some(session_store),
        )
        .map_err(Into::into)
}

fn build_chat_runtime(
    context: &HermesContext,
    config: &LoadedConfig,
    enabled_toolsets: &[String],
    overrides: &ModelOverrides,
    cwd: &Path,
    skills: &[String],
    worktree: Option<&WorktreeMetadata>,
) -> Result<ToolRuntime, Box<dyn Error>> {
    let disabled = disabled_memory_toolsets(&config.config.memory);
    let tool_names = get_tool_definitions(Some(enabled_toolsets), disabled.as_deref())
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
    let delegate = DelegateExecutor::new(
        context.clone(),
        config.clone(),
        "rust-delegate",
        enabled_toolsets.to_vec(),
        overrides.clone(),
        cwd.to_path_buf(),
    );
    let mut runtime = ToolRuntime::new(cwd)
        .with_hermes_home(context.hermes_home())
        .with_available_tool_names(tool_names)
        .with_clarify_callback(run_clarify_prompt)
        .with_delegate_callback(move |request, parent_runtime| {
            delegate.execute(request, parent_runtime)
        });

    let mut missing_skills = Vec::new();
    for skill in unique_cli_strings(skills) {
        match load_skill_prompt_content(&context.hermes_home(), &skill) {
            Ok(content) => {
                runtime =
                    runtime.with_system_prompt_addition(format!("Activated skill:\n{content}"));
            }
            Err(_) => missing_skills.push(skill),
        }
    }
    if !missing_skills.is_empty() {
        return Err(format!("Unknown skill(s): {}", missing_skills.join(", ")).into());
    }
    if let Some(worktree) = worktree {
        runtime = runtime.with_system_prompt_addition(format!(
            "[System note: You are working in an isolated git worktree at {}. Your branch is `{}`. Changes here do not affect the main working tree or other agents. The original repo is at {}.]",
            worktree.path.display(),
            worktree.branch,
            worktree.repo_root.display(),
        ));
    }
    let _ = runtime.load_memory_store(&config.config.memory);
    Ok(runtime)
}

fn resolve_initial_prompt(
    query: Option<&str>,
    prompt_parts: &[String],
) -> Result<Option<String>, Box<dyn Error>> {
    let query = query.and_then(non_empty_trimmed);
    let prompt = (!prompt_parts.is_empty()).then(|| prompt_parts.join(" "));
    let prompt = prompt.as_deref().and_then(non_empty_trimmed);
    match (query, prompt) {
        (Some(_), Some(_)) => Err("pass either --query or a chat prompt, not both".into()),
        (Some(query), None) => Ok(Some(query.to_string())),
        (None, Some(prompt)) => Ok(Some(prompt.to_string())),
        (None, None) => Ok(None),
    }
}

fn resolve_chat_session_hint(
    session_store: &hermes_core::SessionStore,
    chat: &ChatArgs,
) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(resume) = chat.resume.as_deref() {
        let resolved = resolve_session_name_or_id(session_store, resume)?;
        return resolved
            .map(Some)
            .ok_or_else(|| format!("No session matched '{resume}'.").into());
    }
    let Some(continue_value) = chat.continue_last.as_deref() else {
        return Ok(None);
    };
    if continue_value == CONTINUE_LATEST_SENTINEL {
        let resolved = resolve_last_session_id(session_store, "cli")?;
        return resolved
            .map(Some)
            .ok_or_else(|| "No previous CLI session found to continue.".into());
    }
    let resolved = resolve_session_name_or_id(session_store, continue_value)?;
    resolved
        .map(Some)
        .ok_or_else(|| format!("No session matched '{continue_value}'.").into())
}

fn resolve_session_name_or_id(
    session_store: &hermes_core::SessionStore,
    value: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if let Some(resolved) = session_store.resolve_session_id(trimmed)? {
        return Ok(Some(resolved));
    }
    if let Some(resolved) = session_store.resolve_session_by_title(trimmed)? {
        return Ok(Some(resolved));
    }
    Ok(None)
}

fn resolve_last_session_id(
    session_store: &hermes_core::SessionStore,
    source: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    Ok(session_store
        .search_sessions(Some(source), 1, 0)?
        .into_iter()
        .next()
        .map(|row| row.id))
}

fn has_any_chat_provider_configured(
    context: &HermesContext,
    config: &LoadedConfig,
    overrides: &ModelOverrides,
) -> Result<bool, Box<dyn Error>> {
    if overrides
        .api_key
        .as_deref()
        .and_then(non_empty_trimmed)
        .is_some()
        || overrides
            .base_url
            .as_deref()
            .and_then(non_empty_trimmed)
            .is_some()
        || config.configured_model_api_key().is_some()
        || config.configured_model_base_url().is_some()
    {
        return Ok(true);
    }

    for key in provider_env_keys() {
        if env_value_for_context(context, key).is_some() {
            return Ok(true);
        }
    }

    if get_active_auth_provider(context.hermes_home().as_path())?
        .as_deref()
        .is_some_and(inference_auth_provider)
    {
        return Ok(true);
    }

    for profile in list_provider_profiles() {
        if profile.auth_type == "api_key" {
            continue;
        }
        let status = get_auth_status_summary(context.hermes_home().as_path(), profile.name)?;
        if status.configured || status.logged_in {
            return Ok(true);
        }
    }

    Ok(false)
}

fn provider_env_keys() -> Vec<&'static str> {
    let mut keys = vec![
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    ];
    for profile in list_provider_profiles() {
        for key in profile.env_vars {
            if !keys.contains(key) {
                keys.push(key);
            }
        }
    }
    keys
}

fn inference_auth_provider(provider: &str) -> bool {
    let provider = provider.trim();
    !provider.is_empty()
        && list_provider_profiles()
            .iter()
            .any(|profile| profile.name == provider)
}

fn env_value_for_context(context: &HermesContext, key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| read_env_file_value(context.env_path().as_path(), key))
}

fn read_env_file_value(path: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((entry_key, entry_value)) = trimmed.split_once('=') else {
            continue;
        };
        if entry_key.trim() != key {
            continue;
        }
        let value = entry_value.trim().trim_matches(['"', '\'']);
        if value.is_empty() {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

fn print_chat_setup_required() {
    println!();
    println!("It looks like Hermes isn't configured yet -- no API keys or providers found.");
    println!();
    println!("  Run:  hermes setup");
    println!();
}

fn print_noninteractive_chat_setup_guidance() {
    println!("⚕ Hermes Setup — Non-interactive mode");
    println!();
    println!("  Running in a non-interactive environment (no TTY detected).");
    println!("  The interactive wizard cannot be used here.");
    println!();
    println!("  Configure Hermes using environment variables or config commands:");
    println!("    hermes config set model.provider custom");
    println!("    hermes config set model.base_url http://localhost:8080/v1");
    println!("    hermes config set model.default your-model-name");
    println!();
    println!("  Or set OPENROUTER_API_KEY / OPENAI_API_KEY in your environment.");
    println!("  Run 'hermes setup' in an interactive terminal to use the full wizard.");
    println!();
}

fn prompt_user_input(prompt: &str) -> Result<Option<String>, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}

fn read_prompt_from_stdin() -> Result<Option<String>, Box<dyn Error>> {
    if io::stdin().is_terminal() {
        return Ok(None);
    }
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(non_empty_trimmed(&buffer).map(ToOwned::to_owned))
}

fn unique_cli_strings(values: &[String]) -> Vec<String> {
    let mut output = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if !trimmed.is_empty() && !output.iter().any(|existing| existing == trimmed) {
            output.push(trimmed.to_string());
        }
    }
    output
}

fn known_subcommands() -> &'static [&'static str] {
    &[
        "paths",
        "version",
        "dump",
        "doctor",
        "debug",
        "hooks",
        "login",
        "fallback",
        "model",
        "slack",
        "webhook",
        "completion",
        "dashboard",
        "gateway",
        "skills",
        "checkpoints",
        "snapshot",
        "plugins",
        "curator",
        "memory",
        "mcp",
        "insights",
        "claw",
        "acp",
        "logout",
        "auth",
        "setup",
        "config",
        "pairing",
        "uninstall",
        "backup",
        "import",
        "profile",
        "chat",
        "sessions",
        "cron",
        "logs",
        "kanban",
        "tools",
        "update",
        "whatsapp",
        "status",
    ]
}

fn coalesce_session_name_args(args: Vec<String>, commands: &[&str]) -> Vec<String> {
    let mut output = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        let token = &args[index];
        output.push(token.clone());
        if matches!(token.as_str(), "-c" | "--continue" | "-r" | "--resume") {
            index += 1;
            let mut pieces = Vec::new();
            while index < args.len() {
                let next = &args[index];
                if next.starts_with('-') {
                    break;
                }
                if commands.iter().any(|command| *command == next) {
                    break;
                }
                pieces.push(next.clone());
                index += 1;
            }
            if !pieces.is_empty() {
                output.push(pieces.join(" "));
            }
            continue;
        }
        index += 1;
    }
    output
}

fn argv_contains_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn apply_cli_runtime_env(config: &LoadedConfig, cwd: &Path) {
    unsafe { std::env::set_var("TERMINAL_ENV", config.config.terminal.backend.trim()) };
    let backend = config.config.terminal.backend.trim().to_ascii_lowercase();
    let configured_cwd = config.config.terminal.cwd.trim();
    let effective_cwd = if backend == "local" {
        cwd.to_path_buf()
    } else if matches!(configured_cwd, "" | "." | "auto" | "cwd") {
        cwd.to_path_buf()
    } else {
        PathBuf::from(configured_cwd)
    };
    unsafe { std::env::set_var("TERMINAL_CWD", effective_cwd) };
}

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[derive(Debug, Clone)]
struct WorktreeMetadata {
    path: PathBuf,
    branch: String,
    repo_root: PathBuf,
}

struct WorktreeGuard {
    metadata: WorktreeMetadata,
    original_cwd: PathBuf,
}

impl WorktreeGuard {
    fn create() -> Result<Self, Box<dyn Error>> {
        let repo_root =
            git_repo_root()?.ok_or("--worktree requires being inside a git repository")?;
        let original_cwd = std::env::current_dir().unwrap_or_else(|_| repo_root.clone());
        let metadata = setup_worktree(&repo_root)?;
        std::env::set_current_dir(&metadata.path)?;
        unsafe { std::env::set_var("TERMINAL_CWD", &metadata.path) };
        Ok(Self {
            metadata,
            original_cwd,
        })
    }

    fn path(&self) -> &Path {
        &self.metadata.path
    }

    fn metadata(&self) -> &WorktreeMetadata {
        &self.metadata
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original_cwd);
        let _ = cleanup_worktree(&self.metadata);
    }
}

fn git_repo_root() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(non_empty_trimmed(&value).map(PathBuf::from))
}

fn setup_worktree(repo_root: &Path) -> Result<WorktreeMetadata, Box<dyn Error>> {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let worktree_name = format!("hermes-{:08x}", (suffix & 0xffff_ffff) as u64);
    let branch = format!("hermes/{worktree_name}");
    let worktrees_dir = repo_root.join(".worktrees");
    fs::create_dir_all(&worktrees_dir)?;
    ensure_worktrees_gitignore(repo_root)?;
    let path = worktrees_dir.join(&worktree_name);
    let output = ProcessCommand::new("git")
        .current_dir(repo_root)
        .args([
            "worktree",
            "add",
            path.to_string_lossy().as_ref(),
            "-b",
            branch.as_str(),
            "HEAD",
        ])
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("failed to create worktree: {stderr}").into());
    }
    copy_worktreeinclude_entries(repo_root, &path)?;
    Ok(WorktreeMetadata {
        path,
        branch,
        repo_root: repo_root.to_path_buf(),
    })
}

fn ensure_worktrees_gitignore(repo_root: &Path) -> Result<(), Box<dyn Error>> {
    let path = repo_root.join(".gitignore");
    let entry = ".worktrees/";
    let existing = fs::read_to_string(&path).unwrap_or_default();
    if existing.lines().any(|line| line.trim() == entry) {
        return Ok(());
    }
    let mut contents = existing;
    if !contents.is_empty() && !contents.ends_with('\n') {
        contents.push('\n');
    }
    contents.push_str(entry);
    contents.push('\n');
    fs::write(path, contents)?;
    Ok(())
}

fn copy_worktreeinclude_entries(
    repo_root: &Path,
    worktree_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let include_path = repo_root.join(".worktreeinclude");
    if !include_path.is_file() {
        return Ok(());
    }
    for line in fs::read_to_string(include_path)?.lines() {
        let entry = line.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        let source = repo_root.join(entry);
        let destination = worktree_path.join(entry);
        if source.is_file() {
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source, &destination)?;
        }
    }
    Ok(())
}

fn cleanup_worktree(worktree: &WorktreeMetadata) -> Result<(), Box<dyn Error>> {
    if !worktree.path.exists() {
        return Ok(());
    }
    let has_unpushed = ProcessCommand::new("git")
        .current_dir(&worktree.path)
        .args(["log", "--oneline", "HEAD", "--not", "--remotes"])
        .output()
        .ok()
        .is_some_and(|output| !String::from_utf8_lossy(&output.stdout).trim().is_empty());
    if has_unpushed {
        eprintln!(
            "Keeping worktree with unpushed commits: {}",
            worktree.path.display()
        );
        return Ok(());
    }
    let _ = ProcessCommand::new("git")
        .current_dir(&worktree.repo_root)
        .args([
            "worktree",
            "remove",
            worktree.path.to_string_lossy().as_ref(),
            "--force",
        ])
        .output();
    let _ = ProcessCommand::new("git")
        .current_dir(&worktree.repo_root)
        .args(["branch", "-D", worktree.branch.as_str()])
        .output();
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
        KanbanCommand::Compat(args) => {
            launch_python_kanban_command(&args)?;
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

fn resolve_resume_target(
    session_store: &hermes_core::SessionStore,
    candidate: &str,
) -> Result<String, Box<dyn Error>> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return Err("resume target cannot be empty".into());
    }
    if let Some(session_id) = session_store.resolve_session_id(candidate)? {
        return Ok(session_id);
    }
    if let Some(session_id) = session_store.resolve_session_by_title(candidate)? {
        return Ok(session_id);
    }
    Err(format!("session '{candidate}' not found").into())
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

fn print_slash_compat(
    session_store: &hermes_core::SessionStore,
    slash_name: &str,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    launch_python_slash_command(&session_id, slash_name, &args.args)
}

fn print_no_arg_slash_compat(
    session_store: &hermes_core::SessionStore,
    slash_name: &str,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    ensure_no_extra_args(slash_name, &args.args)?;
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    launch_python_slash_command(&session_id, slash_name, &[])
}

fn print_live_gateway_only_command(
    _command_name: &str,
    _args: &[String],
    message: &str,
) -> Result<(), Box<dyn Error>> {
    Err(message.into())
}

fn print_goal_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let goal_text = args.args.join(" ");
    if is_goal_control_command(&goal_text) {
        return launch_python_slash_command(&session_id, "goal", &args.args);
    }
    launch_python_slash_command(&session_id, "goal", &args.args)?;
    launch_python_chat_query(&session_id, goal_text.trim())
}

fn print_compress_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let focus_topic = args.args.join(" ");
    let result = launch_python_session_bridge_compress(&session_id, focus_topic.trim())?;
    if let Some(summary) = result.get("summary").and_then(JsonValue::as_str) {
        let summary = summary.trim();
        if !summary.is_empty() {
            println!("{summary}");
            return Ok(());
        }
    }
    let before = result
        .get("before_messages")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0);
    let after = result
        .get("after_messages")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0);
    println!("Compressed session history: {before} -> {after} messages.");
    Ok(())
}

fn print_send_turn_compat(
    session_store: &hermes_core::SessionStore,
    command_name: &str,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let prompt = ensure_nonempty_prompt(command_name, &args.args)?;
    launch_python_chat_query(&session_id, &prompt)
}

fn print_image_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let (image_path, prompt) = split_image_args(&args.args)?;
    launch_python_chat_turn(&session_id, prompt.as_deref(), Some(&image_path))
}

fn print_paste_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let prompt = joined_optional_prompt(&args.args);
    let image_path = launch_python_session_bridge_clipboard_save()?;
    launch_python_chat_turn(&session_id, prompt.as_deref(), Some(&image_path))
}

fn print_retry_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    ensure_no_extra_args("retry", &args.args)?;
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let Some(result) = session_store.truncate_last_user_turn(&session_id)? else {
        println!("No user message found to retry.");
        return Ok(());
    };
    let Some(message) = result.last_user_text.as_deref() else {
        println!("Last user message is empty.");
        return Ok(());
    };
    println!("Retrying: \"{}\"", truncate_preview(message, 60));
    launch_python_chat_query(&session_id, message)
}

fn print_undo_compat(
    session_store: &hermes_core::SessionStore,
    args: SlashCompatArgs,
) -> Result<(), Box<dyn Error>> {
    ensure_no_extra_args("undo", &args.args)?;
    let session_id = resolve_slash_compat_session_id(session_store, args.session.as_deref())?;
    let Some(result) = session_store.truncate_last_user_turn(&session_id)? else {
        println!("No user message found to undo.");
        return Ok(());
    };
    let removed = result
        .last_user_text
        .as_deref()
        .map(|text| format!(" Removed: \"{}\"", truncate_preview(text, 60)))
        .unwrap_or_default();
    println!("Undid {} message(s).{removed}", result.removed_count);
    println!(
        "  {} message(s) remaining in history.",
        result.remaining_count
    );
    Ok(())
}

fn print_resume(
    session_store: &hermes_core::SessionStore,
    args: ResumeArgs,
) -> Result<(), Box<dyn Error>> {
    let target = args.target.join(" ");
    let session_id = if target.trim().is_empty() {
        let (limit, _) = validate_pagination(args.limit, 0)?;
        let rows = session_store.search_sessions(args.source.as_deref(), limit, 0)?;
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut input = stdin.lock();
        let mut output = stdout.lock();
        match browse_sessions_with_io(&rows, &mut input, &mut output)? {
            Some(session_id) => {
                writeln!(output, "Resuming session: {session_id}")?;
                output.flush()?;
                session_id
            }
            None => {
                writeln!(output, "Cancelled.")?;
                return Ok(());
            }
        }
    } else {
        let session_id = resolve_resume_target(session_store, &target)?;
        println!("Resuming session: {session_id}");
        session_id
    };
    launch_python_resume_session(&session_id)
}

fn launch_python_resume_session(session_id: &str) -> Result<(), Box<dyn Error>> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Err("session id cannot be empty".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for interactive resume")?;
    let mut command = ProcessCommand::new(&python);
    let original_argv = std::env::args().skip(1).collect::<Vec<_>>();
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("hermes_cli.main");
    for arg in build_resume_python_args(session_id, &original_argv) {
        command.arg(arg);
    }
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(format!("resume command exited with status {status}").into())
}

fn launch_python_kanban_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() {
        return Err("kanban command cannot be empty".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for kanban compatibility command")?;
    let mut command = ProcessCommand::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("hermes_cli.main")
        .arg("kanban");
    for arg in args {
        command.arg(arg);
    }
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(format!("kanban command exited with status {status}").into())
}

fn resolve_slash_compat_session_id(
    session_store: &hermes_core::SessionStore,
    requested: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    if let Some(session_id) = requested {
        return resolve_existing_session_id(session_store, session_id);
    }
    Ok(session_store
        .search_sessions(None, 1, 0)?
        .into_iter()
        .next()
        .map(|row| row.id)
        .unwrap_or_else(|| String::from("rust-cli-compat")))
}

fn ensure_no_extra_args(command_name: &str, args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() {
        return Ok(());
    }
    Err(format!("usage: hermes {command_name} [--session SESSION]").into())
}

fn ensure_nonempty_prompt(command_name: &str, args: &[String]) -> Result<String, Box<dyn Error>> {
    let prompt = args.join(" ");
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return Err(format!("usage: hermes {command_name} <prompt> [--session SESSION]").into());
    }
    Ok(trimmed.to_string())
}

fn joined_optional_prompt(args: &[String]) -> Option<String> {
    let prompt = args.join(" ");
    let trimmed = prompt.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn split_image_args(args: &[String]) -> Result<(PathBuf, Option<String>), Box<dyn Error>> {
    let Some(path) = args.first() else {
        return Err("usage: hermes image <path> [prompt] [--session SESSION]".into());
    };
    let trimmed_path = path.trim();
    if trimmed_path.is_empty() {
        return Err("usage: hermes image <path> [prompt] [--session SESSION]".into());
    }
    Ok((
        PathBuf::from(trimmed_path),
        joined_optional_prompt(&args[1..]),
    ))
}

fn is_goal_control_command(goal_text: &str) -> bool {
    matches!(
        goal_text.trim().to_ascii_lowercase().as_str(),
        "" | "status" | "pause" | "resume" | "clear" | "stop" | "done"
    )
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let mut preview = text.chars().take(max_chars).collect::<String>();
    preview.push_str("...");
    preview
}

fn launch_python_slash_command(
    session_id: &str,
    slash_name: &str,
    args: &[String],
) -> Result<(), Box<dyn Error>> {
    let rendered = run_python_slash_command(session_id, slash_name, args)?;
    if !rendered.is_empty() {
        println!("{rendered}");
    }
    Ok(())
}

fn run_python_slash_command(
    session_id: &str,
    slash_name: &str,
    args: &[String],
) -> Result<String, Box<dyn Error>> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Err("session id cannot be empty".into());
    }
    let slash_name = slash_name.trim();
    if slash_name.is_empty() || slash_name.contains(char::is_whitespace) {
        return Err("slash command name must be non-empty and contain no whitespace".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for slash compatibility command")?;
    let mut command = ProcessCommand::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("tui_gateway.slash_worker")
        .arg("--session-key")
        .arg(session_id)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let payload = json!({
        "id": 1,
        "command": build_slash_command_text(slash_name, args),
    });
    if let Some(stdin) = child.stdin.as_mut() {
        writeln!(stdin, "{payload}")?;
    } else {
        return Err("slash worker stdin was not available".into());
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "slash command exited with status {}: {stderr}",
            output.status
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or("slash worker returned no response")?;
    let parsed: JsonValue = serde_json::from_str(response)?;
    if !parsed
        .get("ok")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        let error = parsed
            .get("error")
            .and_then(JsonValue::as_str)
            .unwrap_or("slash worker failed");
        return Err(error.to_string().into());
    }
    if let Some(rendered) = parsed.get("output").and_then(JsonValue::as_str) {
        return Ok(rendered.trim_end().to_string());
    }
    Ok(String::new())
}

fn launch_python_session_bridge_compress(
    session_id: &str,
    focus_topic: &str,
) -> Result<JsonValue, Box<dyn Error>> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Err("session id cannot be empty".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for compress compatibility command")?;
    let mut command = ProcessCommand::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("tui_gateway.session_bridge")
        .arg("compress")
        .arg("--session-key")
        .arg(session_id)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let trimmed_focus = focus_topic.trim();
    if !trimmed_focus.is_empty() {
        command.arg("--focus-topic").arg(trimmed_focus);
    }
    let output = command.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "compress bridge exited with status {}: {stderr}",
            output.status
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or("compress bridge returned no response")?;
    let parsed: JsonValue = serde_json::from_str(response)?;
    if !parsed
        .get("ok")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        let error = parsed
            .get("error")
            .and_then(JsonValue::as_str)
            .unwrap_or("compress bridge failed");
        return Err(error.to_string().into());
    }
    Ok(parsed.get("result").cloned().unwrap_or(JsonValue::Null))
}

fn launch_python_session_bridge_clipboard_save() -> Result<PathBuf, Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for clipboard compatibility command")?;
    let output = ProcessCommand::new(&python)
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("tui_gateway.session_bridge")
        .arg("clipboard-save")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "clipboard bridge exited with status {}: {stderr}",
            output.status
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or("clipboard bridge returned no response")?;
    let parsed: JsonValue = serde_json::from_str(response)?;
    if !parsed
        .get("ok")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        let error = parsed
            .get("error")
            .and_then(JsonValue::as_str)
            .unwrap_or("clipboard bridge failed");
        return Err(error.to_string().into());
    }
    let path = parsed
        .get("result")
        .and_then(|result| result.get("path"))
        .and_then(JsonValue::as_str)
        .ok_or("clipboard bridge returned no path")?;
    Ok(PathBuf::from(path))
}

fn launch_python_chat_query(session_id: &str, query: &str) -> Result<(), Box<dyn Error>> {
    launch_python_chat_turn(session_id, Some(query), None)
}

fn launch_python_chat_turn(
    session_id: &str,
    query: Option<&str>,
    image: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Err("session id cannot be empty".into());
    }
    let query = query.map(str::trim).filter(|value| !value.is_empty());
    if query.is_none() && image.is_none() {
        return Err("query or image is required".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_CLI_PYTHON"))
        .ok_or("could not find a Python interpreter for chat turn command")?;
    let mut command = ProcessCommand::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .arg("-m")
        .arg("hermes_cli.main")
        .arg("chat")
        .arg("--resume")
        .arg(session_id);
    if let Some(query) = query {
        command.arg("--query").arg(query);
    }
    if let Some(image) = image {
        command.arg("--image").arg(image);
    }
    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(format!("chat turn command exited with status {status}").into())
}

fn build_slash_command_text(slash_name: &str, args: &[String]) -> String {
    if args.is_empty() {
        return format!("/{slash_name}");
    }
    format!("/{} {}", slash_name, args.join(" "))
}

fn build_resume_python_args(session_id: &str, original_argv: &[String]) -> Vec<String> {
    let mut args = extract_inherited_relaunch_flags(original_argv);
    args.push("--resume".to_string());
    args.push(session_id.trim().to_string());
    args
}

fn extract_inherited_relaunch_flags(argv: &[String]) -> Vec<String> {
    let mut flags = Vec::new();
    let mut index = 0;
    while index < argv.len() {
        let arg = &argv[index];
        if let Some((key, _value)) = arg.split_once('=') {
            if INHERITED_RELAUNCH_FLAGS
                .iter()
                .any(|(flag, _)| *flag == key)
            {
                flags.push(arg.clone());
            }
            index += 1;
            continue;
        }

        if let Some((_, takes_value)) = INHERITED_RELAUNCH_FLAGS
            .iter()
            .find(|(flag, _)| *flag == arg)
        {
            flags.push(arg.clone());
            if *takes_value && index + 1 < argv.len() && !argv[index + 1].starts_with('-') {
                flags.push(argv[index + 1].clone());
                index += 1;
            }
        }
        index += 1;
    }
    flags
}

fn browse_sessions_with_io(
    rows: &[hermes_core::SessionSearchRow],
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<Option<String>, Box<dyn Error>> {
    if rows.is_empty() {
        writeln!(output, "No sessions found.")?;
        return Ok(None);
    }

    let mut filter = String::new();
    loop {
        let visible = filter_session_rows(rows, &filter);
        writeln!(output)?;
        writeln!(output, "Hermes Session Browser")?;
        writeln!(
            output,
            "  Enter a number to select, text to filter, or Enter/q to cancel."
        )?;
        if !filter.is_empty() {
            writeln!(output, "  Filter: {filter}")?;
        }
        writeln!(output)?;

        if visible.is_empty() {
            writeln!(output, "  No sessions match that filter.")?;
        } else {
            for (index, row) in visible.iter().enumerate() {
                writeln!(
                    output,
                    "  {}. {:<50} {:<10} {:<8} {}",
                    index + 1,
                    session_row_label(row),
                    relative_time_label(row.last_active),
                    row.source,
                    row.id
                )?;
            }
        }

        let response = prompt_line(input, output, "Selection")?;
        let trimmed = response.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("q") {
            return Ok(None);
        }
        if let Ok(index) = trimmed.parse::<usize>() {
            if (1..=visible.len()).contains(&index) {
                return Ok(Some(visible[index - 1].id.clone()));
            }
            writeln!(output, "Selection must be between 1 and {}.", visible.len())?;
            continue;
        }
        filter = trimmed.to_string();
    }
}

fn filter_session_rows<'a>(
    rows: &'a [hermes_core::SessionSearchRow],
    filter: &str,
) -> Vec<&'a hermes_core::SessionSearchRow> {
    let needle = filter.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return rows.iter().collect();
    }
    rows.iter()
        .filter(|row| {
            let title = row
                .title
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let preview = row.preview.to_ascii_lowercase();
            let source = row.source.to_ascii_lowercase();
            let id = row.id.to_ascii_lowercase();
            title.contains(&needle)
                || preview.contains(&needle)
                || source.contains(&needle)
                || id.contains(&needle)
        })
        .collect()
}

fn session_row_label(row: &hermes_core::SessionSearchRow) -> String {
    let label = row
        .title
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            if row.preview.trim().is_empty() {
                row.id.as_str()
            } else {
                row.preview.as_str()
            }
        });
    truncate_for_display(label, 50)
}

fn truncate_for_display(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim();
    let chars = trimmed.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return trimmed.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    format!("{}...", chars.into_iter().take(keep).collect::<String>())
}

fn relative_time_label(timestamp: f64) -> String {
    let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(value) => value.as_secs_f64(),
        Err(_) => return "unknown".to_string(),
    };
    let delta = (now - timestamp).max(0.0);
    if delta < 60.0 {
        return "just now".to_string();
    }
    if delta < 3600.0 {
        return format!("{}m ago", (delta / 60.0).floor() as u64);
    }
    if delta < 86_400.0 {
        return format!("{}h ago", (delta / 3600.0).floor() as u64);
    }
    format!("{}d ago", (delta / 86_400.0).floor() as u64)
}

fn prompt_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    label: &str,
) -> Result<String, Box<dyn Error>> {
    write!(output, "{label}: ")?;
    output.flush()?;

    let mut response = String::new();
    let read = input.read_line(&mut response)?;
    if read == 0 {
        return Ok(String::new());
    }
    Ok(response)
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
    use crate::python_bridge::{project_root, resolve_repo_python};
    use clap::CommandFactory;
    use hermes_core::{MessageAppend, SessionCreate};
    use serde::Deserialize;
    use serde_json::Value;
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn session_row(
        id: &str,
        title: Option<&str>,
        preview: &str,
        source: &str,
        last_active: f64,
    ) -> hermes_core::SessionSearchRow {
        hermes_core::SessionSearchRow {
            id: id.to_string(),
            source: source.to_string(),
            model: Some("test/model".to_string()),
            title: title.map(ToOwned::to_owned),
            started_at: last_active - 60.0,
            ended_at: None,
            end_reason: None,
            message_count: 3,
            tool_call_count: 0,
            last_active,
            preview: preview.to_string(),
        }
    }

    fn now_ts() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_secs_f64())
            .unwrap_or(0.0)
    }

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
    fn top_level_tui_flags_parse() {
        let cli = Cli::try_parse_from([
            "hermes",
            "--tui",
            "--dev",
            "--resume",
            "session-123",
            "--model",
            "anthropic/claude-sonnet-4.6",
            "--provider",
            "anthropic",
            "--toolsets",
            "web,terminal",
            "--query",
            "hello",
        ])
        .unwrap();
        assert!(cli.tui);
        assert!(cli.tui_dev);
        assert_eq!(cli.resume.as_deref(), Some("session-123"));
        assert_eq!(cli.model.as_deref(), Some("anthropic/claude-sonnet-4.6"));
        assert_eq!(cli.provider.as_deref(), Some("anthropic"));
        assert_eq!(cli.toolsets.as_deref(), Some("web,terminal"));
        assert_eq!(cli.query.as_deref(), Some("hello"));
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

    #[test]
    fn sessions_browse_subcommand_is_parseable() {
        let cli = Cli::try_parse_from(["hermes", "sessions", "browse", "--limit", "42"]).unwrap();
        match cli.command {
            Some(Command::Sessions {
                command: SessionsCommand::Browse { limit, source },
            }) => {
                assert_eq!(limit, 42);
                assert!(source.is_none());
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn top_level_model_and_snapshot_aliases_are_parseable() {
        let cli =
            Cli::try_parse_from(["hermes", "provider", "set", "gpt-5", "--provider", "openai"])
                .unwrap();
        match cli.command {
            Some(Command::Model {
                command:
                    Some(model_cmd::ModelCommand::Set {
                        model, provider, ..
                    }),
            }) => {
                assert_eq!(model, "gpt-5");
                assert_eq!(provider.as_deref(), Some("openai"));
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "snap", "create", "before", "deploy"]).unwrap();
        match cli.command {
            Some(Command::Snapshot {
                command: Some(snapshot_cmd::SnapshotCommand::Create { label }),
            }) => assert_eq!(label, vec![String::from("before"), String::from("deploy")]),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "platforms", "status"]).unwrap();
        match cli.command {
            Some(Command::Gateway(gateway_cmd::GatewayArgs {
                command: Some(gateway_cmd::GatewayCommand::Status(_)),
                ..
            })) => {}
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn kanban_slash_style_subcommands_parse_into_compat_passthrough() {
        let cli = Cli::try_parse_from(["hermes", "kanban", "list"]).unwrap();
        match cli.command {
            Some(Command::Kanban {
                command: KanbanCommand::Compat(args),
                ..
            }) => assert_eq!(args, vec![String::from("list")]),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "kanban", "boards", "list"]).unwrap();
        match cli.command {
            Some(Command::Kanban {
                command: KanbanCommand::Compat(args),
                ..
            }) => assert_eq!(args, vec![String::from("boards"), String::from("list")]),
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn slash_compat_commands_and_aliases_are_parseable() {
        let cli = Cli::try_parse_from(["hermes", "reasoning", "high"]).unwrap();
        match cli.command {
            Some(Command::Reasoning(SlashCompatArgs { session, args })) => {
                assert!(session.is_none());
                assert_eq!(args, vec![String::from("high")]);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "approve", "all", "session"]).unwrap();
        match cli.command {
            Some(Command::Approve(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("all"), String::from("session")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli =
            Cli::try_parse_from(["hermes", "browser", "connect", "ws://127.0.0.1:9222"]).unwrap();
        match cli.command {
            Some(Command::Browser(SlashCompatArgs { args, .. })) => assert_eq!(
                args,
                vec![String::from("connect"), String::from("ws://127.0.0.1:9222")]
            ),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "compress", "database", "schema"]).unwrap();
        match cli.command {
            Some(Command::Compress(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("database"), String::from("schema")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "clear"]).unwrap();
        match cli.command {
            Some(Command::Clear(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "reload_mcp"]).unwrap();
        match cli.command {
            Some(Command::ReloadMcp(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "tasks"]).unwrap();
        match cli.command {
            Some(Command::Agents(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "reset", "Sprint", "Branch"]).unwrap();
        match cli.command {
            Some(Command::New(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("Sprint"), String::from("Branch")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "gquota"]).unwrap();
        match cli.command {
            Some(Command::Gquota(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "goal", "pause"]).unwrap();
        match cli.command {
            Some(Command::Goal(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("pause")]);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "history"]).unwrap();
        match cli.command {
            Some(Command::History(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from([
            "hermes",
            "image",
            "/tmp/cat.png",
            "what",
            "do",
            "you",
            "see",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Image(SlashCompatArgs { args, .. })) => assert_eq!(
                args,
                vec![
                    String::from("/tmp/cat.png"),
                    String::from("what"),
                    String::from("do"),
                    String::from("you"),
                    String::from("see")
                ]
            ),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "paste", "describe", "this"]).unwrap();
        match cli.command {
            Some(Command::Paste(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("describe"), String::from("this")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "deny", "all"]).unwrap();
        match cli.command {
            Some(Command::Deny(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("all")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "redraw"]).unwrap();
        match cli.command {
            Some(Command::Redraw(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "status"]).unwrap();
        match cli.command {
            Some(Command::Status(StatusArgs { session })) => {
                assert!(session.is_none());
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "rollback", "diff", "2"]).unwrap();
        match cli.command {
            Some(Command::Rollback(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("diff"), String::from("2")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "status", "--session", "sess-1"]).unwrap();
        match cli.command {
            Some(Command::Status(StatusArgs { session })) => {
                assert_eq!(session.as_deref(), Some("sess-1"));
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "q", "follow", "up"]).unwrap();
        match cli.command {
            Some(Command::Queue(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("follow"), String::from("up")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "retry"]).unwrap();
        match cli.command {
            Some(Command::Retry(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "set-home"]).unwrap();
        match cli.command {
            Some(Command::Sethome(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "undo"]).unwrap();
        match cli.command {
            Some(Command::Undo(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "steer", "after", "tool"]).unwrap();
        match cli.command {
            Some(Command::Steer(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("after"), String::from("tool")])
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "save"]).unwrap();
        match cli.command {
            Some(Command::Save(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "stop"]).unwrap();
        match cli.command {
            Some(Command::Stop(SlashCompatArgs { args, .. })) => assert!(args.is_empty()),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "bg", "Summarize", "the", "logs"]).unwrap();
        match cli.command {
            Some(Command::Background(SlashCompatArgs { args, .. })) => assert_eq!(
                args,
                vec![
                    String::from("Summarize"),
                    String::from("the"),
                    String::from("logs")
                ]
            ),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "fork", "new", "branch", "title"]).unwrap();
        match cli.command {
            Some(Command::Branch(SlashCompatArgs { args, .. })) => assert_eq!(
                args,
                vec![
                    String::from("new"),
                    String::from("branch"),
                    String::from("title")
                ]
            ),
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "title", "--session", "sess-1", "My", "Session"])
            .unwrap();
        match cli.command {
            Some(Command::Title(SlashCompatArgs { session, args })) => {
                assert_eq!(session.as_deref(), Some("sess-1"));
                assert_eq!(args, vec![String::from("My"), String::from("Session")]);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }

        let cli = Cli::try_parse_from(["hermes", "topic", "help"]).unwrap();
        match cli.command {
            Some(Command::Topic(SlashCompatArgs { args, .. })) => {
                assert_eq!(args, vec![String::from("help")]);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn restart_command_is_parseable_with_gateway_flags() {
        let cli = Cli::try_parse_from(["hermes", "restart", "--system", "--all"]).unwrap();
        match cli.command {
            Some(Command::Restart(gateway_cmd::GatewayServiceArgs { system, all })) => {
                assert!(system);
                assert!(all);
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn resume_command_is_parseable_with_target_and_flags() {
        let cli = Cli::try_parse_from([
            "hermes", "resume", "--limit", "25", "--source", "cli", "Project", "Phoenix",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Resume(ResumeArgs {
                limit,
                source,
                target,
            })) => {
                assert_eq!(limit, 25);
                assert_eq!(source.as_deref(), Some("cli"));
                assert_eq!(
                    target,
                    vec![String::from("Project"), String::from("Phoenix")]
                );
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn browse_sessions_with_io_selects_filtered_entry() {
        let now = now_ts();
        let rows = vec![
            session_row("sess-1", Some("Alpha project"), "draft plan", "cli", now),
            session_row(
                "sess-2",
                Some("Beta project"),
                "review PR",
                "telegram",
                now - 7200.0,
            ),
        ];
        let mut input = Cursor::new(b"beta\n1\n".to_vec());
        let mut output = Vec::new();
        let selected = browse_sessions_with_io(&rows, &mut input, &mut output).unwrap();
        assert_eq!(selected.as_deref(), Some("sess-2"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Hermes Session Browser"));
        assert!(rendered.contains("Filter: beta"));
        assert!(rendered.contains("2h ago"));
    }

    #[test]
    fn browse_sessions_with_io_cancels_on_empty_input() {
        let now = now_ts();
        let rows = vec![session_row("sess-1", None, "hello world", "cli", now)];
        let mut input = Cursor::new(b"\n".to_vec());
        let mut output = Vec::new();
        let selected = browse_sessions_with_io(&rows, &mut input, &mut output).unwrap();
        assert!(selected.is_none());
    }

    #[test]
    fn browse_sessions_with_io_accepts_rows_from_store() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "20260525_000001_abc123";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                session_id,
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String("hello".to_string())),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        let rows = store.search_sessions(Some("cli"), 10, 0).unwrap();
        let mut input = Cursor::new(b"1\n".to_vec());
        let mut output = Vec::new();
        let selected = browse_sessions_with_io(&rows, &mut input, &mut output).unwrap();
        assert_eq!(selected.as_deref(), Some(session_id));
    }

    #[test]
    fn launch_python_resume_session_uses_python_override_and_resume_flag() {
        let temp = tempfile::TempDir::new().unwrap();
        let log_path = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\n",
            log_path.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_resume_session("sess-123");
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();
        let logged = fs::read_to_string(&log_path).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("hermes_cli.main"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains("sess-123"));
    }

    #[test]
    fn launch_python_kanban_command_uses_python_override_and_args() {
        let temp = tempfile::TempDir::new().unwrap();
        let log_path = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\n",
            log_path.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_kanban_command(&[String::from("boards"), String::from("list")]);
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();
        let logged = fs::read_to_string(&log_path).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("hermes_cli.main"));
        assert!(logged.contains("kanban"));
        assert!(logged.contains("boards"));
        assert!(logged.contains("list"));
    }

    #[test]
    fn launch_python_slash_command_uses_worker_module_and_json_payload() {
        let temp = tempfile::TempDir::new().unwrap();
        let argv_log = temp.path().join("argv.log");
        let stdin_log = temp.path().join("stdin.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\ncat > \"{}\"\nprintf '%s\\n' '{{\"id\":1,\"ok\":true,\"output\":\"compat ok\"}}'\n",
            argv_log.display(),
            stdin_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_slash_command(
            "sess-compat",
            "reasoning",
            &[String::from("high"), String::from("show")],
        );
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("tui_gateway.slash_worker"));
        assert!(logged.contains("--session-key"));
        assert!(logged.contains("sess-compat"));

        let payload = fs::read_to_string(&stdin_log).unwrap();
        assert!(payload.contains("\"id\":1"));
        assert!(payload.contains("\"command\":\"/reasoning high show\""));
    }

    #[test]
    fn print_no_arg_slash_compat_uses_worker_for_clear_and_redraw() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "clear-redraw";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let stdin_log = temp.path().join("stdin.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' -- CALL -- >> \"{argv}\"\nprintf '%s\\n' \"$@\" >> \"{argv}\"\ncat >> \"{stdin}\"\nprintf '%s\\n' '{{\"id\":1,\"ok\":true,\"output\":\"ok\"}}'\n",
            argv = argv_log.display(),
            stdin = stdin_log.display(),
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        print_no_arg_slash_compat(
            &store,
            "clear",
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: Vec::new(),
            },
        )
        .unwrap();
        print_no_arg_slash_compat(
            &store,
            "redraw",
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: Vec::new(),
            },
        )
        .unwrap();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert_eq!(logged.matches("CALL").count(), 2);
        assert!(logged.contains("tui_gateway.slash_worker"));
        assert!(logged.contains("--session-key"));
        assert!(logged.contains(session_id));

        let payload = fs::read_to_string(&stdin_log).unwrap();
        assert!(payload.contains("\"command\":\"/clear\""));
        assert!(payload.contains("\"command\":\"/redraw\""));
    }

    #[test]
    fn print_slash_compat_uses_worker_for_rollback_diff() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "rollback-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let stdin_log = temp.path().join("stdin.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{argv}\"\ncat > \"{stdin}\"\nprintf '%s\\n' '{{\"id\":1,\"ok\":true,\"output\":\"rollback ok\"}}'\n",
            argv = argv_log.display(),
            stdin = stdin_log.display(),
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        print_slash_compat(
            &store,
            "rollback",
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![String::from("diff"), String::from("2")],
            },
        )
        .unwrap();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("tui_gateway.slash_worker"));
        assert!(logged.contains("--session-key"));
        assert!(logged.contains(session_id));

        let payload = fs::read_to_string(&stdin_log).unwrap();
        assert!(payload.contains("\"command\":\"/rollback diff 2\""));
    }

    #[test]
    fn print_status_uses_worker_when_session_is_requested() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let env_report = EnvLoadReport::default();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "status-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let stdin_log = temp.path().join("stdin.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{argv}\"\ncat > \"{stdin}\"\nprintf '%s\\n' '{{\"id\":1,\"ok\":true,\"output\":\"status ok\"}}'\n",
            argv = argv_log.display(),
            stdin = stdin_log.display(),
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        print_status(
            &context,
            &env_report,
            &config,
            &store,
            StatusArgs {
                session: Some(session_id.to_string()),
            },
        )
        .unwrap();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("tui_gateway.slash_worker"));
        assert!(logged.contains("--session-key"));
        assert!(logged.contains(session_id));

        let payload = fs::read_to_string(&stdin_log).unwrap();
        assert!(payload.contains("\"command\":\"/status\""));
    }

    #[test]
    fn live_gateway_only_commands_report_explicit_unavailable_errors() {
        let approve = print_live_gateway_only_command(
            "approve",
            &[String::from("all")],
            "approve is only available for live pending approvals in a running gateway or TUI session.",
        )
        .unwrap_err();
        assert!(
            approve
                .to_string()
                .contains("approve is only available for live pending approvals")
        );

        let topic = print_live_gateway_only_command(
            "topic",
            &[String::from("help")],
            "topic is only available in Telegram private chats through the gateway.",
        )
        .unwrap_err();
        assert!(
            topic
                .to_string()
                .contains("topic is only available in Telegram private chats")
        );
    }

    #[test]
    fn launch_python_chat_query_uses_chat_resume_and_query() {
        let temp = tempfile::TempDir::new().unwrap();
        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\n",
            argv_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_chat_query("sess-chat", "Ship the fix");
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("hermes_cli.main"));
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains("sess-chat"));
        assert!(logged.contains("--query"));
        assert!(logged.contains("Ship the fix"));
    }

    #[test]
    fn launch_python_chat_turn_uses_chat_resume_query_and_image() {
        let temp = tempfile::TempDir::new().unwrap();
        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\n",
            argv_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_chat_turn(
            "sess-chat",
            Some("Describe the attachment"),
            Some(Path::new("/tmp/cat.png")),
        );
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("hermes_cli.main"));
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains("sess-chat"));
        assert!(logged.contains("--query"));
        assert!(logged.contains("Describe the attachment"));
        assert!(logged.contains("--image"));
        assert!(logged.contains("/tmp/cat.png"));
    }

    #[test]
    fn launch_python_session_bridge_compress_uses_bridge_module_and_focus_topic() {
        let temp = tempfile::TempDir::new().unwrap();
        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\nprintf '%s\\n' '{{\"ok\":true,\"result\":{{\"summary\":\"compressed ok\"}}}}'\n",
            argv_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_session_bridge_compress("sess-compress", "database schema");
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        assert_eq!(
            result.unwrap().get("summary").and_then(JsonValue::as_str),
            Some("compressed ok")
        );

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("tui_gateway.session_bridge"));
        assert!(logged.contains("compress"));
        assert!(logged.contains("--session-key"));
        assert!(logged.contains("sess-compress"));
        assert!(logged.contains("--focus-topic"));
        assert!(logged.contains("database schema"));
    }

    #[test]
    fn launch_python_session_bridge_clipboard_save_uses_bridge_module() {
        let temp = tempfile::TempDir::new().unwrap();
        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\nprintf '%s\\n' '{{\"ok\":true,\"result\":{{\"path\":\"/tmp/fake.png\"}}}}'\n",
            argv_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = launch_python_session_bridge_clipboard_save();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        assert_eq!(result.unwrap(), PathBuf::from("/tmp/fake.png"));

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("-m"));
        assert!(logged.contains("tui_gateway.session_bridge"));
        assert!(logged.contains("clipboard-save"));
    }

    #[test]
    fn print_send_turn_compat_uses_chat_query_for_queue_and_steer() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "send-turn-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' -- CALL -- >> \"{argv}\"\nprintf '%s\\n' \"$@\" >> \"{argv}\"\n",
            argv = argv_log.display(),
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        print_send_turn_compat(
            &store,
            "queue",
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![String::from("queued"), String::from("prompt")],
            },
        )
        .unwrap();
        print_send_turn_compat(
            &store,
            "steer",
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![String::from("steered"), String::from("prompt")],
            },
        )
        .unwrap();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert_eq!(logged.matches("CALL").count(), 2);
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains(session_id));
        assert!(logged.contains("--query"));
        assert!(logged.contains("queued prompt"));
        assert!(logged.contains("steered prompt"));
    }

    #[test]
    fn print_image_and_paste_compat_use_chat_turn_with_image_inputs() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "image-paste-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' -- CALL -- >> \"{argv}\"\nprintf '%s\\n' \"$@\" >> \"{argv}\"\nif [ \"$2\" = \"tui_gateway.session_bridge\" ]; then\n  printf '%s\\n' '{{\"ok\":true,\"result\":{{\"path\":\"/tmp/from-clipboard.png\"}}}}'\nfi\n",
            argv = argv_log.display(),
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        print_image_compat(
            &store,
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![
                    String::from("/tmp/direct.png"),
                    String::from("inspect"),
                    String::from("this"),
                ],
            },
        )
        .unwrap();
        print_paste_compat(
            &store,
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![String::from("describe"), String::from("clipboard")],
            },
        )
        .unwrap();
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert_eq!(logged.matches("CALL").count(), 3);
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains(session_id));
        assert!(logged.contains("--image"));
        assert!(logged.contains("/tmp/direct.png"));
        assert!(logged.contains("/tmp/from-clipboard.png"));
        assert!(logged.contains("--query"));
        assert!(logged.contains("inspect this"));
        assert!(logged.contains("describe clipboard"));
        assert!(logged.contains("tui_gateway.session_bridge"));
        assert!(logged.contains("clipboard-save"));
    }

    #[test]
    fn print_retry_compat_replays_last_user_turn_via_chat_query() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "retry-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        for message in [
            hermes_core::MessageAppend {
                role: String::from("user"),
                content: Some(json!("keep me")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("keep answer")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("user"),
                content: Some(json!("retry this exact prompt")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("drop answer")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ] {
            store.append_message(session_id, &message).unwrap();
        }

        let argv_log = temp.path().join("argv.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{}\"\n",
            argv_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = print_retry_compat(
            &store,
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: Vec::new(),
            },
        );
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains(session_id));
        assert!(logged.contains("--query"));
        assert!(logged.contains("retry this exact prompt"));

        let remaining = store.get_messages(session_id).unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].role, "user");
        assert_eq!(remaining[1].role, "assistant");
    }

    #[test]
    fn print_undo_compat_truncates_last_exchange() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "undo-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        for message in [
            hermes_core::MessageAppend {
                role: String::from("user"),
                content: Some(json!("keep me")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("keep answer")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("user"),
                content: Some(json!("drop me")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            hermes_core::MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("drop answer")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ] {
            store.append_message(session_id, &message).unwrap();
        }

        print_undo_compat(
            &store,
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: Vec::new(),
            },
        )
        .unwrap();

        let remaining = store.get_messages(session_id).unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].content, Some(json!("keep me")));
        assert_eq!(remaining[1].content, Some(json!("keep answer")));
    }

    #[test]
    fn print_goal_compat_kicks_off_non_control_goal_via_chat_query() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        let session_id = "goal-session";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let argv_log = temp.path().join("argv.log");
        let stdin_log = temp.path().join("stdin.log");
        let python = temp.path().join("fake-python");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' -- CALL -- >> \"{argv}\"\nprintf '%s\\n' \"$@\" >> \"{argv}\"\nif [ \"$2\" = \"tui_gateway.slash_worker\" ]; then\n  cat > \"{stdin}\"\n  printf '%s\\n' '{{\"id\":1,\"ok\":true,\"output\":\"goal ok\"}}'\nfi\n",
            argv = argv_log.display(),
            stdin = stdin_log.display()
        );
        fs::write(&python, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe {
            std::env::set_var("HERMES_CLI_PYTHON", &python);
        }
        let result = print_goal_compat(
            &store,
            SlashCompatArgs {
                session: Some(session_id.to_string()),
                args: vec![
                    String::from("pause"),
                    String::from("the"),
                    String::from("rollout"),
                ],
            },
        );
        unsafe {
            std::env::remove_var("HERMES_CLI_PYTHON");
        }
        result.unwrap();

        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(logged.contains("tui_gateway.slash_worker"));
        assert!(logged.contains("hermes_cli.main"));
        assert!(logged.contains("chat"));
        assert!(logged.contains("--resume"));
        assert!(logged.contains(session_id));
        assert!(logged.contains("--query"));
        assert!(logged.contains("pause the rollout"));

        let payload = fs::read_to_string(&stdin_log).unwrap();
        assert!(payload.contains("\"command\":\"/goal pause the rollout\""));
    }

    #[test]
    fn goal_control_command_detection_matches_python_semantics() {
        assert!(is_goal_control_command(""));
        assert!(is_goal_control_command("pause"));
        assert!(is_goal_control_command(" done "));
        assert!(!is_goal_control_command("pause the rollout"));
        assert!(!is_goal_control_command("ship the fix"));
    }

    #[test]
    fn resolve_slash_compat_session_id_prefers_requested_or_latest() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();

        let requested_error = resolve_slash_compat_session_id(&store, Some("missing"));
        assert!(requested_error.is_err());
        assert_eq!(
            resolve_slash_compat_session_id(&store, None).unwrap(),
            "rust-cli-compat"
        );

        let session_id = "20260525_000002_latest";
        store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("test/model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        assert_eq!(
            resolve_slash_compat_session_id(&store, None).unwrap(),
            session_id
        );
        assert_eq!(
            resolve_slash_compat_session_id(&store, Some(session_id)).unwrap(),
            session_id
        );
    }

    #[test]
    fn resolve_resume_target_supports_title_lineage_and_prefixes() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();

        for session_id in ["abc123", "abc999", "other1"] {
            store
                .create_session(&SessionCreate {
                    id: session_id.to_string(),
                    source: "cli".to_string(),
                    user_id: None,
                    model: Some("test/model".to_string()),
                    model_config: None,
                    system_prompt: None,
                    parent_session_id: None,
                })
                .unwrap();
        }
        store.set_session_title("abc123", "Project").unwrap();
        store.set_session_title("abc999", "Project #2").unwrap();

        assert_eq!(resolve_resume_target(&store, "other1").unwrap(), "other1");
        assert_eq!(resolve_resume_target(&store, "other").unwrap(), "other1");
        assert_eq!(resolve_resume_target(&store, "Project").unwrap(), "abc999");
        assert!(resolve_resume_target(&store, "missing").is_err());
    }

    #[test]
    fn extract_inherited_relaunch_flags_preserves_expected_flags() {
        let argv = vec![
            "--tui".to_string(),
            "--dev".to_string(),
            "--profile".to_string(),
            "work".to_string(),
            "-m".to_string(),
            "gpt-5".to_string(),
            "--provider=openrouter".to_string(),
            "sessions".to_string(),
            "browse".to_string(),
        ];
        assert_eq!(
            extract_inherited_relaunch_flags(&argv),
            vec![
                "--tui".to_string(),
                "--dev".to_string(),
                "--profile".to_string(),
                "work".to_string(),
                "-m".to_string(),
                "gpt-5".to_string(),
                "--provider=openrouter".to_string(),
            ]
        );
    }

    #[test]
    fn extract_inherited_relaunch_flags_skips_non_inherited_flags() {
        let argv = vec![
            "--worktree".to_string(),
            "--quiet".to_string(),
            "--tui".to_string(),
            "--skills".to_string(),
            "foo".to_string(),
            "sessions".to_string(),
            "browse".to_string(),
        ];
        assert_eq!(
            extract_inherited_relaunch_flags(&argv),
            vec![
                "--tui".to_string(),
                "--skills".to_string(),
                "foo".to_string(),
            ]
        );
    }

    #[test]
    fn build_resume_python_args_appends_resume_after_inherited_flags() {
        let argv = vec![
            "--tui".to_string(),
            "--profile".to_string(),
            "work".to_string(),
            "sessions".to_string(),
            "browse".to_string(),
        ];
        assert_eq!(
            build_resume_python_args("sess-42", &argv),
            vec![
                "--tui".to_string(),
                "--profile".to_string(),
                "work".to_string(),
                "--resume".to_string(),
                "sess-42".to_string(),
            ]
        );
    }

    #[test]
    fn build_slash_command_text_prefixes_and_joins_args() {
        assert_eq!(build_slash_command_text("usage", &[]), "/usage");
        assert_eq!(
            build_slash_command_text("reasoning", &[String::from("high"), String::from("show")]),
            "/reasoning high show"
        );
        assert_eq!(
            build_slash_command_text(
                "title",
                &[
                    String::from("Project"),
                    String::from("Phoenix"),
                    String::from("Plan")
                ]
            ),
            "/title Project Phoenix Plan"
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
    struct PythonCommandMatrix {
        flags: Vec<String>,
        subcommands: BTreeMap<String, PythonCommandMatrix>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CommandMatrix {
        flags: BTreeSet<String>,
        subcommands: BTreeMap<String, CommandMatrix>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CommandGap {
        path: String,
        python_only: Vec<String>,
        rust_only: Vec<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FlagGap {
        path: String,
        python_only: Vec<String>,
        rust_only: Vec<String>,
    }

    const PYTHON_COMMAND_MATRIX_SCRIPT: &str = r#"
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import hermes_cli.main as hermes_main


class _StopCapture(Exception):
    pass


captured = {}
original = argparse.ArgumentParser.parse_args


def _capture(parser, *args, **kwargs):
    captured["parser"] = parser
    raise _StopCapture()


def _walk(parser):
    flags = set()
    subcommands = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                if name == "help":
                    continue
                subcommands[name] = _walk(subparser)
        else:
            for option in action.option_strings:
                if option.startswith("--"):
                    flags.add(option)
    return {
        "flags": sorted(flags),
        "subcommands": subcommands,
    }


argparse.ArgumentParser.parse_args = _capture
argv0 = sys.argv[:]
sys.argv = ["hermes"]
try:
    hermes_main.main()
except _StopCapture:
    pass
finally:
    argparse.ArgumentParser.parse_args = original
    sys.argv = argv0

print(json.dumps(_walk(captured["parser"]), sort_keys=True))
"#;

    fn rust_command_matrix(command: &mut clap::Command) -> CommandMatrix {
        command.build();

        let mut flags = BTreeSet::new();
        for arg in command.get_arguments() {
            if arg.is_hide_set() {
                continue;
            }
            if let Some(long) = arg.get_long() {
                flags.insert(format!("--{long}"));
            }
        }

        let mut subcommands = BTreeMap::new();
        for subcommand in command.get_subcommands_mut() {
            if subcommand.is_hide_set() || subcommand.get_name() == "help" {
                continue;
            }
            let child = rust_command_matrix(subcommand);
            subcommands.insert(subcommand.get_name().to_string(), child.clone());
            for alias in subcommand.get_all_aliases() {
                subcommands.insert(alias.to_string(), child.clone());
            }
        }

        CommandMatrix { flags, subcommands }
    }

    fn python_command_matrix() -> PythonCommandMatrix {
        let project_root = project_root();
        let python = resolve_repo_python(&project_root, None).expect("python interpreter");
        let hermes_home = tempfile::TempDir::new().expect("temp hermes home");
        let output = std::process::Command::new(python)
            .arg("-c")
            .arg(PYTHON_COMMAND_MATRIX_SCRIPT)
            .current_dir(&project_root)
            .env("HERMES_HOME", hermes_home.path())
            .output()
            .expect("run python command matrix helper");

        assert!(
            output.status.success(),
            "python command matrix helper failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        serde_json::from_slice(&output.stdout).expect("parse python command matrix json")
    }

    fn normalize_python_matrix(node: PythonCommandMatrix) -> CommandMatrix {
        CommandMatrix {
            flags: node.flags.into_iter().collect(),
            subcommands: node
                .subcommands
                .into_iter()
                .map(|(name, child)| (name, normalize_python_matrix(child)))
                .collect(),
        }
    }

    fn diff_command_sets(
        path: &str,
        python: &CommandMatrix,
        rust: &CommandMatrix,
        gaps: &mut Vec<CommandGap>,
    ) {
        let python_names = python
            .subcommands
            .keys()
            .cloned()
            .collect::<BTreeSet<String>>();
        let rust_names = rust
            .subcommands
            .keys()
            .cloned()
            .collect::<BTreeSet<String>>();

        let python_only = python_names
            .difference(&rust_names)
            .cloned()
            .collect::<Vec<String>>();
        let rust_only = rust_names
            .difference(&python_names)
            .cloned()
            .collect::<Vec<String>>();
        if !python_only.is_empty() || !rust_only.is_empty() {
            gaps.push(CommandGap {
                path: path.to_string(),
                python_only,
                rust_only,
            });
        }

        for name in python_names.intersection(&rust_names) {
            let next_path = format!("{path} {name}");
            diff_command_sets(
                &next_path,
                python.subcommands.get(name).expect("python child"),
                rust.subcommands.get(name).expect("rust child"),
                gaps,
            );
        }
    }

    fn matrix_at_path<'a>(root: &'a CommandMatrix, path: &[&str]) -> &'a CommandMatrix {
        let mut current = root;
        for segment in path {
            current = current
                .subcommands
                .get(*segment)
                .unwrap_or_else(|| panic!("missing command path segment: {segment}"));
        }
        current
    }

    fn flag_gap_at_path(
        root_path: &[&str],
        python_root: &CommandMatrix,
        rust_root: &CommandMatrix,
        ignored: &[&str],
    ) -> FlagGap {
        let path = if root_path.is_empty() {
            String::from("hermes")
        } else {
            format!("hermes {}", root_path.join(" "))
        };
        let python = matrix_at_path(python_root, root_path);
        let rust = matrix_at_path(rust_root, root_path);
        let ignored = ignored
            .iter()
            .map(|flag| flag.to_string())
            .collect::<BTreeSet<String>>();

        let python_only = python
            .flags
            .difference(&rust.flags)
            .filter(|flag| !ignored.contains(*flag))
            .cloned()
            .collect::<Vec<String>>();
        let rust_only = rust
            .flags
            .difference(&python.flags)
            .filter(|flag| !ignored.contains(*flag))
            .cloned()
            .collect::<Vec<String>>();

        FlagGap {
            path,
            python_only,
            rust_only,
        }
    }

    #[test]
    fn python_rust_cli_command_matrix_is_current() {
        let python = normalize_python_matrix(python_command_matrix());
        let rust = rust_command_matrix(&mut Cli::command());
        let mut gaps = Vec::new();
        diff_command_sets("hermes", &python, &rust, &mut gaps);
        let expected = vec![
            CommandGap {
                path: String::from("hermes"),
                python_only: vec![],
                rust_only: vec![String::from("paths"), String::from("snapshot")],
            },
            CommandGap {
                path: String::from("hermes kanban"),
                python_only: vec![
                    "archive",
                    "assign",
                    "assignees",
                    "block",
                    "boards",
                    "claim",
                    "comment",
                    "complete",
                    "context",
                    "create",
                    "diag",
                    "diagnostics",
                    "dispatch",
                    "edit",
                    "gc",
                    "heartbeat",
                    "init",
                    "link",
                    "list",
                    "log",
                    "ls",
                    "notify-list",
                    "notify-subscribe",
                    "notify-unsubscribe",
                    "reassign",
                    "reclaim",
                    "runs",
                    "show",
                    "stats",
                    "tail",
                    "unblock",
                    "unlink",
                    "watch",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                rust_only: vec![String::from("run"), String::from("tick")],
            },
            CommandGap {
                path: String::from("hermes model"),
                python_only: vec![],
                rust_only: vec![
                    String::from("providers"),
                    String::from("set"),
                    String::from("show"),
                ],
            },
            CommandGap {
                path: String::from("hermes profile"),
                python_only: vec![],
                rust_only: vec![String::from("current"), String::from("path")],
            },
            CommandGap {
                path: String::from("hermes sessions"),
                python_only: vec![String::from("browse")],
                rust_only: vec![String::from("search")],
            },
            CommandGap {
                path: String::from("hermes tools"),
                python_only: vec![],
                rust_only: vec![String::from("ls"), String::from("run")],
            },
        ];
        assert_eq!(gaps, expected);
    }

    #[test]
    fn python_rust_root_and_chat_flag_matrix_is_current() {
        let python = normalize_python_matrix(python_command_matrix());
        let rust = rust_command_matrix(&mut Cli::command());
        let ignored = ["--profile"];

        assert_eq!(
            flag_gap_at_path(&[], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes"),
                python_only: vec![
                    "--accept-hooks",
                    "--continue",
                    "--dev",
                    "--ignore-rules",
                    "--ignore-user-config",
                    "--model",
                    "--oneshot",
                    "--pass-session-id",
                    "--provider",
                    "--resume",
                    "--skills",
                    "--toolsets",
                    "--tui",
                    "--worktree",
                    "--yolo",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                rust_only: vec![],
            }
        );

        assert_eq!(
            flag_gap_at_path(&["chat"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes chat"),
                python_only: vec![
                    "--accept-hooks",
                    "--checkpoints",
                    "--continue",
                    "--dev",
                    "--ignore-rules",
                    "--ignore-user-config",
                    "--image",
                    "--max-turns",
                    "--pass-session-id",
                    "--query",
                    "--quiet",
                    "--resume",
                    "--skills",
                    "--source",
                    "--toolsets",
                    "--tui",
                    "--verbose",
                    "--worktree",
                    "--yolo",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                rust_only: vec![
                    "--api-key",
                    "--api-mode",
                    "--base-url",
                    "--session",
                    "--toolset",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            }
        );
    }

    #[test]
    fn python_rust_gateway_and_tools_flag_matrix_is_current() {
        let python = normalize_python_matrix(python_command_matrix());
        let rust = rust_command_matrix(&mut Cli::command());
        let ignored = ["--profile"];

        assert_eq!(
            flag_gap_at_path(&["gateway"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes gateway"),
                python_only: vec![],
                rust_only: vec![],
            }
        );
        assert_eq!(
            flag_gap_at_path(&["gateway", "run"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes gateway run"),
                python_only: vec![],
                rust_only: vec![],
            }
        );

        for path in [
            ["gateway", "start"].as_slice(),
            ["gateway", "stop"].as_slice(),
            ["gateway", "restart"].as_slice(),
            ["gateway", "status"].as_slice(),
            ["gateway", "install"].as_slice(),
            ["gateway", "uninstall"].as_slice(),
            ["gateway", "setup"].as_slice(),
            ["gateway", "migrate-legacy"].as_slice(),
        ] {
            let gap = flag_gap_at_path(path, &python, &rust, &ignored);
            assert_eq!(
                gap,
                FlagGap {
                    path: format!("hermes {}", path.join(" ")),
                    python_only: vec![],
                    rust_only: vec![String::from("--accept-hooks")],
                }
            );
        }

        assert_eq!(
            flag_gap_at_path(&["tools"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes tools"),
                python_only: vec![],
                rust_only: vec![],
            }
        );
        assert_eq!(
            flag_gap_at_path(&["tools", "enable"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes tools enable"),
                python_only: vec![],
                rust_only: vec![],
            }
        );
        assert_eq!(
            flag_gap_at_path(&["tools", "disable"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes tools disable"),
                python_only: vec![],
                rust_only: vec![],
            }
        );
        assert_eq!(
            flag_gap_at_path(&["tools", "list"], &python, &rust, &ignored),
            FlagGap {
                path: String::from("hermes tools list"),
                python_only: vec![],
                rust_only: vec![String::from("--toolset")],
            }
        );
    }

    #[test]
    fn top_level_chat_flags_parse_without_subcommand() {
        let cli = Cli::try_parse_from([
            "hermes",
            "--resume",
            "Alpha Session",
            "--skills",
            "ops,git",
            "--toolsets",
            "web,terminal",
            "--yolo",
        ])
        .unwrap();

        assert!(cli.command.is_none());
        assert_eq!(cli.chat.resume.as_deref(), Some("Alpha Session"));
        assert_eq!(
            cli.chat.skills,
            vec![String::from("ops"), String::from("git")]
        );
        assert_eq!(
            cli.chat.toolsets,
            vec![String::from("web"), String::from("terminal")]
        );
        assert!(cli.chat.yolo);
    }

    #[test]
    fn coalesces_multiword_continue_before_subcommand() {
        let args = coalesce_session_name_args(
            vec![
                String::from("-c"),
                String::from("Project"),
                String::from("Alpha"),
                String::from("chat"),
                String::from("hello"),
            ],
            known_subcommands(),
        );
        assert_eq!(
            args,
            vec![
                String::from("-c"),
                String::from("Project Alpha"),
                String::from("chat"),
                String::from("hello"),
            ]
        );
    }

    #[test]
    fn resolve_chat_session_hint_supports_titles_and_continue_latest() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home));
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();

        store
            .create_session(&SessionCreate {
                id: String::from("sess-1"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store.set_session_title("sess-1", "Project Alpha").unwrap();

        let resumed = resolve_chat_session_hint(
            &store,
            &ChatArgs {
                resume: Some(String::from("Project Alpha")),
                ..ChatArgs::default()
            },
        )
        .unwrap();
        assert_eq!(resumed.as_deref(), Some("sess-1"));

        let continued = resolve_chat_session_hint(
            &store,
            &ChatArgs {
                continue_last: Some(CONTINUE_LATEST_SENTINEL.to_string()),
                ..ChatArgs::default()
            },
        )
        .unwrap();
        assert_eq!(continued.as_deref(), Some("sess-1"));
    }

    #[test]
    fn build_chat_runtime_rejects_unknown_skills() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();

        let error = build_chat_runtime(
            &context,
            &config,
            &config.config.toolsets,
            &ModelOverrides::default(),
            temp.path(),
            &[String::from("missing-skill")],
            None,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Unknown skill(s): missing-skill")
        );
    }

    #[test]
    fn chat_provider_detection_reads_env_file_and_auth_state() {
        let _guard = cli_test_env_lock().lock().unwrap();
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path().join("home");
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();

        fs::write(context.env_path(), "OPENROUTER_API_KEY=sk-test\n").unwrap();
        assert!(
            has_any_chat_provider_configured(&context, &config, &ModelOverrides::default())
                .unwrap()
        );

        fs::write(context.env_path(), "").unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            r#"{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token":"token"}}}}"#,
        )
        .unwrap();
        assert!(
            has_any_chat_provider_configured(&context, &config, &ModelOverrides::default())
                .unwrap()
        );
    }

    #[test]
    fn read_env_file_value_trims_quotes_and_missing_values() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join(".env");
        fs::write(&path, "OPENAI_API_KEY=\"sk-test\"\nEMPTY=\n").unwrap();

        assert_eq!(
            read_env_file_value(&path, "OPENAI_API_KEY").as_deref(),
            Some("sk-test")
        );
        assert_eq!(read_env_file_value(&path, "EMPTY"), None);
        assert_eq!(read_env_file_value(&path, "MISSING"), None);
    }

    #[test]
    fn inference_auth_provider_filters_non_inference_entries() {
        assert!(inference_auth_provider("openai-codex"));
        assert!(!inference_auth_provider("spotify"));
        assert!(!inference_auth_provider(""));
    }
}
