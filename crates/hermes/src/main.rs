mod auth_cmd;
mod backup;
mod completion;
mod config_cmd;
mod debug;
mod doctor;
mod dump;
mod hooks;
mod logs;

use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::sleep;
use std::time::Duration;

use clap::{Parser, Subcommand};
use hermes_core::{
    DelegateExecutor, EnvLoadReport, HermesContext, KanbanDispatchOptions, LoadedConfig,
    LoggingMode, LoggingSetup, ModelOverrides, ToolRuntime, dispatch_kanban_once, dispatch_tool,
    get_tool_definitions, is_container, is_wsl, kanban_has_spawnable_ready, run_cron_job_now,
    run_due_cron_jobs, run_kanban_task,
};

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
    Completion(completion::CompletionArgs),
    Logout(auth_cmd::LogoutArgs),
    Auth {
        #[command(subcommand)]
        command: Option<auth_cmd::AuthCommand>,
    },
    Config {
        #[command(subcommand)]
        command: Option<config_cmd::ConfigCommand>,
    },
    Backup(backup::BackupArgs),
    Import(backup::ImportArgs),
    Profile {
        #[command(subcommand)]
        command: Option<ProfileCommand>,
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
        command: CronCommand,
    },
    Logs(logs::LogsArgs),
    Kanban {
        #[command(subcommand)]
        command: KanbanCommand,
    },
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },
    Status,
}

#[derive(Subcommand, Debug)]
enum ProfileCommand {
    Current,
    Path { name: Option<String> },
    Create { name: String },
    Use { name: String },
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
    Tick,
    Run { id: String },
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

#[derive(Subcommand, Debug)]
enum ToolsCommand {
    List {
        #[arg(long)]
        toolset: Option<String>,
    },
    Run {
        name: String,
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
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
        Command::Completion(args) => completion::print_completion(args)?,
        Command::Logout(args) => auth_cmd::print_logout(&context, &config, args)?,
        Command::Auth { command } => auth_cmd::print_auth(&context, &config, command)?,
        Command::Config { command } => config_cmd::print_config(&context, &config, command)?,
        Command::Backup(args) => backup::print_backup(&context, args)?,
        Command::Import(args) => backup::print_import(&context, args)?,
        Command::Profile { command } => print_profile(&context, command)?,
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
        Command::Tools { command } => print_tools(&context, &config, command)?,
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

fn print_profile(
    context: &HermesContext,
    command: Option<ProfileCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(ProfileCommand::Current) {
        ProfileCommand::Current => {
            println!("active_profile={}", context.active_profile());
            println!("current_profile={}", context.current_profile_name());
            println!("display_home={}", context.display_hermes_home());
        }
        ProfileCommand::Path { name } => {
            let selected = name.unwrap_or_else(|| context.current_profile_name());
            println!("profile={selected}");
            println!("path={}", context.profile_dir(&selected)?.display());
        }
        ProfileCommand::Create { name } => {
            let path = context.create_profile(&name)?;
            println!("created={name}");
            println!("path={}", path.display());
        }
        ProfileCommand::Use { name } => {
            context.set_active_profile(&name)?;
            println!("active_profile={}", context.active_profile());
        }
    }
    Ok(())
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
    _context: &HermesContext,
    config: &LoadedConfig,
    session_store: &hermes_core::SessionStore,
    command: CronCommand,
) -> Result<(), Box<dyn Error>> {
    let base_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match command {
        CronCommand::Tick => {
            let result = run_due_cron_jobs(_context, config, session_store, &base_cwd)?;
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
        }
        CronCommand::Run { id } => {
            if id.trim().is_empty() {
                return Err("cron run requires a non-empty job id".into());
            }
            let item = run_cron_job_now(_context, config, session_store, &base_cwd, &id)?;
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
        }
    }
    Ok(())
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

fn print_tools(
    context: &HermesContext,
    config: &LoadedConfig,
    command: ToolsCommand,
) -> Result<(), Box<dyn Error>> {
    match command {
        ToolsCommand::List { toolset } => {
            let enabled = toolset.map(|name| vec![name]);
            let disabled = disabled_memory_toolsets(&config.config.memory);
            let tools = get_tool_definitions(enabled.as_deref(), disabled.as_deref());
            for tool in tools {
                println!(
                    "{}\ttoolset={}\temoji={}\tdescription={}",
                    tool.name, tool.toolset, tool.emoji, tool.description
                );
            }
        }
        ToolsCommand::Run { name, args, cwd } => {
            let parsed_args: serde_json::Value = serde_json::from_str(&args)?;
            if !parsed_args.is_object() {
                return Err("tools run --args must be a JSON object".into());
            }
            let disabled = disabled_memory_toolsets(&config.config.memory);
            let tool_names =
                get_tool_definitions(Some(&config.config.toolsets), disabled.as_deref())
                    .into_iter()
                    .map(|tool| tool.name)
                    .collect::<Vec<_>>();
            let delegate = DelegateExecutor::new(
                context.clone(),
                config.clone(),
                "rust-delegate",
                config.config.toolsets.clone(),
                ModelOverrides::default(),
                cwd.clone().unwrap_or_else(|| {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                }),
            );
            let runtime = match cwd {
                Some(path) => ToolRuntime::new(path),
                None => ToolRuntime::default(),
            }
            .with_hermes_home(context.hermes_home())
            .with_available_tool_names(tool_names)
            .with_clarify_callback(run_clarify_prompt)
            .with_delegate_callback(move |request| delegate.execute(request));
            let mut runtime = runtime;
            let _ = runtime.load_memory_store(&config.config.memory);
            println!("{}", dispatch_tool(&name, parsed_args, &runtime));
        }
    }
    Ok(())
}

fn disabled_memory_toolsets(memory: &hermes_core::MemoryConfig) -> Option<Vec<String>> {
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
