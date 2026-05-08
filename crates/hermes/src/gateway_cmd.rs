use std::env;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{self, BufRead, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
#[cfg(test)]
use std::sync::Mutex;
use std::thread::sleep;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use hermes_core::{HermesContext, is_container, is_wsl};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::config_cmd::save_env_value;
use crate::python_bridge::{project_root, resolve_repo_python};

const SERVICE_BASE: &str = "hermes-gateway";
const LEGACY_SERVICE_NAMES: &[&str] = &["hermes.service"];
const LEGACY_UNIT_EXECSTART_MARKERS: &[&str] = &[
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
];
const GATEWAY_PROCESS_PATTERNS: &[&str] = &[
    "hermes_cli.main gateway",
    "hermes_cli.main --profile",
    "hermes_cli.main -p",
    "hermes_cli/main.py gateway",
    "hermes_cli/main.py --profile",
    "hermes_cli/main.py -p",
    "hermes gateway run",
    "gateway/run.py",
];

#[derive(Args, Debug, Clone)]
pub struct GatewayArgs {
    #[arg(long, global = true, default_value_t = false)]
    pub accept_hooks: bool,
    #[command(subcommand)]
    pub command: Option<GatewayCommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum GatewayCommand {
    Run(GatewayRunArgs),
    Start(GatewayServiceArgs),
    Stop(GatewayServiceArgs),
    Restart(GatewayServiceArgs),
    Status(GatewayStatusArgs),
    Install(GatewayInstallArgs),
    Uninstall(GatewaySystemArgs),
    Setup,
    MigrateLegacy(GatewayMigrateLegacyArgs),
}

#[derive(Args, Debug, Clone)]
pub struct GatewayRunArgs {
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,
    #[arg(short = 'q', long = "quiet", default_value_t = false)]
    pub quiet: bool,
    #[arg(long, default_value_t = false)]
    pub replace: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GatewayServiceArgs {
    #[arg(long, default_value_t = false)]
    pub system: bool,
    #[arg(long, default_value_t = false)]
    pub all: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GatewayStatusArgs {
    #[arg(long, default_value_t = false)]
    pub deep: bool,
    #[arg(short = 'l', long = "full", default_value_t = false)]
    pub full: bool,
    #[arg(long, default_value_t = false)]
    pub system: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GatewayInstallArgs {
    #[arg(long, default_value_t = false)]
    pub force: bool,
    #[arg(long, default_value_t = false)]
    pub system: bool,
    #[arg(long = "run-as-user")]
    pub run_as_user: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct GatewaySystemArgs {
    #[arg(long, default_value_t = false)]
    pub system: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GatewayMigrateLegacyArgs {
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,
    #[arg(short = 'y', long = "yes", default_value_t = false)]
    pub yes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GatewaySnapshot {
    manager: String,
    service_installed: bool,
    service_running: bool,
    gateway_pids: Vec<i64>,
    service_scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegacyGatewayUnit {
    name: String,
    path: PathBuf,
    is_system: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GatewayProcess {
    pid: i64,
    command: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct GatewayUpdateRestartSummary {
    pub restarted_services: Vec<String>,
    pub restarted_profiles: Vec<String>,
    pub stopped_manual: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GatewaySetupPlatform {
    key: String,
    label: String,
    emoji: String,
    status: String,
    #[serde(default)]
    token_var: String,
    #[serde(default)]
    install_hint: Option<String>,
    #[serde(default)]
    setup_instructions: Vec<String>,
    #[serde(default)]
    required_env: Vec<String>,
    #[serde(default)]
    has_builtin_setup: bool,
    #[serde(default)]
    has_plugin_setup: bool,
    #[serde(default)]
    vars: Vec<GatewaySetupVar>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GatewaySetupVar {
    name: String,
    prompt: String,
    #[serde(default)]
    password: bool,
    #[serde(default)]
    help: String,
    #[serde(default)]
    is_allowlist: bool,
}

pub fn print_gateway(context: &HermesContext, args: GatewayArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        None => print_gateway_run(
            args.accept_hooks,
            GatewayRunArgs {
                verbose: 0,
                quiet: false,
                replace: false,
            },
        ),
        Some(GatewayCommand::Run(run)) => print_gateway_run(args.accept_hooks, run),
        Some(GatewayCommand::Start(service)) => {
            print_gateway_start(context, args.accept_hooks, service)
        }
        Some(GatewayCommand::Stop(service)) => {
            print_gateway_stop(context, args.accept_hooks, service)
        }
        Some(GatewayCommand::Restart(service)) => {
            print_gateway_restart(context, args.accept_hooks, service)
        }
        Some(GatewayCommand::Status(status)) => print_gateway_status(context, status),
        Some(GatewayCommand::Install(install)) => {
            print_gateway_install(context, args.accept_hooks, install)
        }
        Some(GatewayCommand::Uninstall(args2)) => print_gateway_uninstall(context, args2),
        Some(GatewayCommand::Setup) => print_gateway_setup(context, args.accept_hooks),
        Some(GatewayCommand::MigrateLegacy(args2)) => print_gateway_migrate_legacy(context, args2),
    }
}

pub(crate) fn restart_gateways_after_update(
    context: &HermesContext,
) -> Result<GatewayUpdateRestartSummary, Box<dyn Error>> {
    let mut summary = GatewayUpdateRestartSummary::default();
    let mut service_pids = collect_running_service_gateway_pids(context)?;

    for system in [false, true] {
        for service in list_active_systemd_gateway_services(system)? {
            match restart_named_systemd_service(system, &service) {
                Ok(()) => summary.restarted_services.push(service),
                Err(error) => {
                    eprintln!("  ⚠ Failed to restart {service}: {error}");
                }
            }
        }
    }

    let profile_homes = collect_profile_homes(context)?;
    if is_macos() {
        for (name, home) in &profile_homes {
            let profile_context = context.clone().with_hermes_home_env(Some(home.clone()));
            if launchd_plist_path(&profile_context).exists()
                && launchd_service_active(&profile_context)
            {
                match restart_launchd_service(&profile_context) {
                    Ok(()) => summary.restarted_services.push(if name == "default" {
                        launchd_label(&profile_context)
                    } else {
                        format!("{} ({name})", launchd_label(&profile_context))
                    }),
                    Err(error) => {
                        eprintln!(
                            "  ⚠ Failed to restart {}: {}",
                            launchd_label(&profile_context),
                            error
                        );
                    }
                }
            }
        }
    }

    service_pids.extend(collect_running_service_gateway_pids(context)?);

    let processes = find_all_gateway_processes()?;
    let mut manual_processes = Vec::new();
    for process in &processes {
        if !service_pids.contains(&process.pid) {
            manual_processes.push(process.clone());
        }
    }
    if manual_processes.is_empty() {
        return Ok(summary);
    }

    let mut mapped_profiles = Vec::new();
    for (name, home) in profile_homes {
        let profile_context = context.clone().with_hermes_home_env(Some(home.clone()));
        if has_any_systemd_unit(&profile_context)
            || (is_macos() && launchd_plist_path(&profile_context).exists())
        {
            continue;
        }
        let Some(pid) = gateway_pids_for_profile(&home)
            .into_iter()
            .find(|pid| manual_processes.iter().any(|process| process.pid == *pid))
        else {
            continue;
        };
        mapped_profiles.push((name, pid));
    }

    let killed = kill_gateway_processes(&manual_processes, false);
    if killed > 0 {
        let manual_pids = manual_processes
            .iter()
            .map(|process| process.pid)
            .collect::<Vec<_>>();
        let _ = wait_for_processes_exit(
            &manual_pids,
            Duration::from_secs(10),
            Some(Duration::from_secs(5)),
        );
    }

    for (profile, _) in &mapped_profiles {
        if launch_detached_profile_gateway_after_update(context, profile).is_ok() {
            summary.restarted_profiles.push(profile.clone());
        }
    }
    summary.stopped_manual = manual_processes.len().saturating_sub(mapped_profiles.len());

    Ok(summary)
}

fn print_gateway_start(
    context: &HermesContext,
    _accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        let processes = find_all_gateway_processes()?;
        let killed = kill_gateway_processes(&processes, false);
        if killed > 0 {
            println!("Killed {killed} stale gateway process(es) across all profiles");
            let _ = wait_for_processes_exit(
                &processes
                    .into_iter()
                    .map(|process| process.pid)
                    .collect::<Vec<_>>(),
                Duration::from_secs(10),
                Some(Duration::from_secs(5)),
            );
        }
    }

    if is_termux(context) {
        return Err(
            "Gateway service start is not supported on Termux. Run manually: hermes gateway run"
                .into(),
        );
    }

    if has_any_systemd_unit(context) {
        start_systemd_service(context, args.system)?;
        println!("Started {} service", gateway_service_name(context));
        return Ok(());
    }

    if is_macos() {
        start_launchd_service(context)?;
        println!("Started {} service", launchd_label(context));
        return Ok(());
    }

    if is_wsl() {
        return Err(
            "WSL detected but systemd is not available. Run `hermes gateway run` or use tmux."
                .into(),
        );
    }

    if is_container() {
        println!("Service start is not applicable inside a Docker container.");
        println!("Start the container itself or run `hermes gateway run`.");
        return Ok(());
    }

    Err("Gateway service start is not supported on this platform".into())
}

fn print_gateway_stop(
    context: &HermesContext,
    _accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        let service_stopped = stop_gateway_service_if_available(context, args.system);
        let processes = find_all_gateway_processes()?;
        let killed = kill_gateway_processes(&processes, false);
        let total = killed + usize::from(service_stopped);
        if total > 0 {
            println!("Stopped {total} gateway process(es) across all profiles");
            let _ = wait_for_processes_exit(
                &processes
                    .into_iter()
                    .map(|process| process.pid)
                    .collect::<Vec<_>>(),
                Duration::from_secs(10),
                Some(Duration::from_secs(5)),
            );
        } else {
            println!("No gateway processes found");
        }
        return Ok(());
    }

    if has_any_systemd_unit(context) {
        stop_systemd_service(context, args.system)?;
        println!("Stopped {} service", gateway_service_name(context));
        return Ok(());
    }

    if launchd_plist_path(context).exists() {
        stop_launchd_service(context)?;
        println!("Stopped {} service", launchd_label(context));
        return Ok(());
    }

    if stop_manual_gateway(context)? {
        println!("Stopped gateway for this profile");
    } else {
        println!("No gateway running for this profile");
    }
    Ok(())
}

fn print_gateway_restart(
    context: &HermesContext,
    accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        let service_stopped = stop_gateway_service_if_available(context, args.system);
        let processes = find_all_gateway_processes()?;
        let killed = kill_gateway_processes(&processes, false);
        let total = killed + usize::from(service_stopped);
        if total > 0 {
            println!("Stopped {total} gateway process(es) across all profiles");
        }
        let _ = wait_for_processes_exit(
            &processes
                .into_iter()
                .map(|process| process.pid)
                .collect::<Vec<_>>(),
            Duration::from_secs(10),
            Some(Duration::from_secs(5)),
        );
        println!("Starting gateway...");

        if has_any_systemd_unit(context) {
            start_systemd_service(context, args.system)?;
            println!("Started {} service", gateway_service_name(context));
            return Ok(());
        }
        if launchd_plist_path(context).exists() {
            start_launchd_service(context)?;
            println!("Started {} service", launchd_label(context));
            return Ok(());
        }
        return print_gateway_run(
            accept_hooks,
            GatewayRunArgs {
                verbose: 0,
                quiet: false,
                replace: false,
            },
        );
    }

    if has_any_systemd_unit(context) {
        restart_systemd_service(context, args.system)?;
        println!("Restarted {} service", gateway_service_name(context));
        return Ok(());
    }

    if launchd_plist_path(context).exists() {
        restart_launchd_service(context)?;
        println!("Restarted {} service", launchd_label(context));
        return Ok(());
    }

    let _ = stop_manual_gateway(context)?;
    println!("Starting gateway...");
    print_gateway_run(
        accept_hooks,
        GatewayRunArgs {
            verbose: 0,
            quiet: false,
            replace: false,
        },
    )
}

fn print_gateway_uninstall(
    context: &HermesContext,
    args: GatewaySystemArgs,
) -> Result<(), Box<dyn Error>> {
    if has_any_systemd_unit(context) {
        let removed = uninstall_systemd_service(context, args.system)?;
        if removed {
            println!("Uninstalled {} service", gateway_service_name(context));
        } else {
            println!("Gateway service is not installed");
        }
        return Ok(());
    }

    if launchd_plist_path(context).exists() {
        let removed = uninstall_launchd_service(context)?;
        if removed {
            println!("Uninstalled {} service", launchd_label(context));
        } else {
            println!("Gateway service is not installed");
        }
        return Ok(());
    }

    println!("Gateway service is not installed");
    Ok(())
}

fn print_gateway_install(
    context: &HermesContext,
    _accept_hooks: bool,
    args: GatewayInstallArgs,
) -> Result<(), Box<dyn Error>> {
    if args
        .run_as_user
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| value.is_empty())
    {
        return Err("run-as-user cannot be empty".into());
    }
    if args.run_as_user.is_some() && !args.system {
        return Err("--run-as-user requires --system".into());
    }
    if args.run_as_user.is_some() && !supports_systemd_services() {
        return Err("--run-as-user is only supported for systemd installs".into());
    }

    if is_termux(context) {
        return Err(
            "Gateway service installation is not supported on Termux. Run manually: hermes gateway run"
                .into(),
        );
    }

    if supports_systemd_services() {
        install_systemd_service(
            context,
            args.force,
            args.system,
            args.run_as_user.as_deref(),
        )?;
        println!(
            "Installed {} {} service",
            gateway_service_name(context),
            if args.system { "system" } else { "user" }
        );
        if args.system {
            let identity = system_service_identity(args.run_as_user.as_deref())?;
            println!("Configured to run as: {}", identity.username);
        }
        return Ok(());
    }

    if is_macos() {
        install_launchd_service(context, args.force)?;
        println!("Installed {} launchd service", launchd_label(context));
        return Ok(());
    }

    if is_wsl() {
        return Err(
            "WSL detected but systemd is not available. Enable systemd or run `hermes gateway run`."
                .into(),
        );
    }

    if is_container() {
        println!("Gateway service install is not applicable inside a Docker container.");
        println!("The gateway should run as the container's main process.");
        return Ok(());
    }

    Err("Gateway service installation is not supported on this platform".into())
}

fn print_gateway_migrate_legacy(
    context: &HermesContext,
    args: GatewayMigrateLegacyArgs,
) -> Result<(), Box<dyn Error>> {
    if !supports_systemd_services() && !is_macos() {
        println!("Legacy unit migration only applies to systemd-based Linux hosts.");
        return Ok(());
    }

    let legacy = find_legacy_gateway_units(context);
    if legacy.is_empty() {
        println!("No legacy Hermes gateway units found.");
        return Ok(());
    }

    println!();
    println!("Legacy Hermes gateway unit(s) found:");
    for unit in &legacy {
        let scope = if unit.is_system { "system" } else { "user" };
        println!("  {}  ({scope} scope)", unit.path.display());
    }
    println!();

    if args.dry_run {
        println!("(dry-run - nothing removed)");
        return Ok(());
    }

    if !args.yes && !confirm_legacy_removal()? {
        println!("Skipped. Run again with: hermes gateway migrate-legacy");
        return Ok(());
    }

    let (removed, remaining) = remove_legacy_gateway_units(&legacy)?;
    println!();
    if remaining.is_empty() {
        println!("Removed {removed} legacy unit(s).");
    } else {
        println!(
            "{} legacy unit(s) still present - see messages above.",
            remaining.len()
        );
    }
    Ok(())
}

fn print_gateway_status(
    context: &HermesContext,
    args: GatewayStatusArgs,
) -> Result<(), Box<dyn Error>> {
    let snapshot = gateway_snapshot(context, args.system);
    println!("profile={}", context.current_profile_name());
    println!("home={}", context.display_hermes_home());
    if snapshot.manager.starts_with("systemd") && snapshot.service_installed {
        println!("manager={}", snapshot.manager);
        println!(
            "service={}",
            if snapshot.service_running {
                "running"
            } else {
                "stopped"
            }
        );
        println!("service_name={}", gateway_service_name(context));
        println!(
            "unit_path={}",
            systemd_unit_path(context, snapshot.service_scope.as_deref() == Some("system"))
                .display()
        );
        if snapshot.has_process_service_mismatch() {
            println!("warning=process running but service is not active");
            println!("pids={}", format_pids(&snapshot.gateway_pids, None));
        }
        if args.deep {
            if let Some((linger, detail)) = systemd_linger_status() {
                if linger {
                    println!("linger=enabled");
                } else {
                    println!("linger=disabled");
                }
                if !detail.is_empty() {
                    println!("linger_detail={detail}");
                }
            }
        }
        if args.full {
            let service_name = gateway_service_name(context);
            let argv = if snapshot.service_scope.as_deref() == Some("system") {
                vec![
                    String::from("status"),
                    service_name,
                    String::from("--no-pager"),
                    String::from("-l"),
                ]
            } else {
                vec![
                    String::from("--user"),
                    String::from("status"),
                    service_name,
                    String::from("--no-pager"),
                    String::from("-l"),
                ]
            };
            run_optional_status_command("systemctl", &argv);
        }
    } else if is_macos() && snapshot.service_installed {
        println!("manager=launchd");
        println!(
            "service={}",
            if snapshot.service_running {
                "running"
            } else {
                "stopped"
            }
        );
        println!("plist_path={}", launchd_plist_path(context).display());
        if snapshot.has_process_service_mismatch() {
            println!("warning=process running but launchd is not active");
            println!("pids={}", format_pids(&snapshot.gateway_pids, None));
        }
        if args.full {
            run_optional_status_command(
                "launchctl",
                &[String::from("list"), launchd_label(context)],
            );
        }
    } else if !snapshot.gateway_pids.is_empty() {
        println!("manager={}", snapshot.manager);
        println!("running=true");
        println!("pids={}", format_pids(&snapshot.gateway_pids, None));
        println!("mode=manual");
        if is_termux(context) {
            println!("note=android may stop background jobs when termux is suspended");
        } else if is_wsl() {
            println!("note=manual foreground mode is recommended for wsl");
        }
    } else {
        println!("manager={}", snapshot.manager);
        println!("running=false");
        println!("mode=stopped");
    }

    let health = runtime_health_lines(context)?;
    if !health.is_empty() {
        println!("health:");
        for line in health {
            println!("  {line}");
        }
    }

    let others = other_profile_gateway_processes(context)?;
    if !others.is_empty() {
        println!("other_profiles:");
        for (name, pid) in others {
            println!("  {name}\tpid={pid}");
        }
    }

    Ok(())
}

fn print_gateway_run(accept_hooks: bool, args: GatewayRunArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON"))
        .ok_or("could not find a Python interpreter for gateway launch")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_GATEWAY_VERBOSE", args.verbose.to_string())
        .env("HERMES_GATEWAY_QUIET", if args.quiet { "1" } else { "0" })
        .env(
            "HERMES_GATEWAY_REPLACE",
            if args.replace { "1" } else { "0" },
        );
    if accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }
    command.arg("-c").arg(GATEWAY_RUN_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("gateway", status).into())
}

const GATEWAY_RUN_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.gateway import run_gateway\n",
    "run_gateway(\n",
    "    verbose=int(os.environ.get('HERMES_GATEWAY_VERBOSE', '0')),\n",
    "    quiet=os.environ.get('HERMES_GATEWAY_QUIET') == '1',\n",
    "    replace=os.environ.get('HERMES_GATEWAY_REPLACE') == '1',\n",
    ")\n",
);

fn print_gateway_setup(context: &HermesContext, accept_hooks: bool) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    run_gateway_setup_with_io(context, &mut input, &mut output, accept_hooks)
}

pub(crate) fn run_gateway_setup_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    accept_hooks: bool,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "┌─────────────────────────────────────────────────────────┐"
    )?;
    writeln!(
        output,
        "│             ⚕ Gateway Setup                            │"
    )?;
    writeln!(
        output,
        "├─────────────────────────────────────────────────────────┤"
    )?;
    writeln!(
        output,
        "│  Configure messaging platforms and the gateway service. │"
    )?;
    writeln!(
        output,
        "│  Press Ctrl+C at any time to exit.                     │"
    )?;
    writeln!(
        output,
        "└─────────────────────────────────────────────────────────┘"
    )?;
    writeln!(output)?;

    let snapshot = gateway_snapshot(context, false);
    if snapshot.service_installed && snapshot.service_running {
        writeln!(output, "Gateway service is installed and running.")?;
    } else if snapshot.service_installed {
        writeln!(output, "Gateway service is installed but not running.")?;
        if prompt_gateway_yes_no(input, output, "Start it now?", true)? {
            if let Err(error) = print_gateway_start(
                context,
                accept_hooks,
                GatewayServiceArgs {
                    system: false,
                    all: false,
                },
            ) {
                writeln!(output, "Start failed: {error}")?;
            }
        }
    } else {
        writeln!(output, "Gateway service is not installed yet.")?;
        writeln!(
            output,
            "You'll be offered to install it after configuring platforms."
        )?;
    }

    loop {
        writeln!(output)?;
        writeln!(output, "Messaging Platforms")?;
        let platforms = load_gateway_setup_metadata(accept_hooks)?;
        let mut labels = platforms
            .iter()
            .map(|platform| {
                format!(
                    "{} {} ({})",
                    platform.emoji, platform.label, platform.status
                )
            })
            .collect::<Vec<_>>();
        labels.push(String::from("Done"));
        let choice = prompt_gateway_menu_choice(
            input,
            output,
            "Select a platform to configure",
            &labels.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        if choice >= platforms.len() {
            break;
        }
        let platform = &platforms[choice];
        if gateway_platform_uses_native_standard_setup(platform) {
            configure_standard_gateway_platform_with_io(context, platform, input, output)?;
        } else {
            run_gateway_platform_setup_bridge(accept_hooks, &platform.key)?;
        }
    }

    let platforms = load_gateway_setup_metadata(accept_hooks)?;
    let any_configured = platforms
        .iter()
        .any(|platform| gateway_platform_status_is_progress(&platform.status));
    if !any_configured {
        writeln!(output)?;
        writeln!(
            output,
            "No platforms configured. Run 'hermes gateway setup' when ready."
        )?;
        writeln!(output)?;
        return Ok(());
    }

    writeln!(output)?;
    let snapshot = gateway_snapshot(context, false);
    if snapshot.service_running {
        if prompt_gateway_yes_no(
            input,
            output,
            "Restart the gateway to pick up changes?",
            true,
        )? {
            if let Err(error) = print_gateway_restart(
                context,
                accept_hooks,
                GatewayServiceArgs {
                    system: false,
                    all: false,
                },
            ) {
                writeln!(output, "Restart failed: {error}")?;
            }
        }
    } else if snapshot.service_installed {
        if prompt_gateway_yes_no(input, output, "Start the gateway service?", true)? {
            if let Err(error) = print_gateway_start(
                context,
                accept_hooks,
                GatewayServiceArgs {
                    system: false,
                    all: false,
                },
            ) {
                writeln!(output, "Start failed: {error}")?;
            }
        }
    } else if supports_systemd_services() || is_macos() {
        let service_kind = if supports_systemd_services() {
            "systemd"
        } else {
            "launchd"
        };
        let question = if is_wsl() {
            format!(
                "Install the gateway as a {service_kind} service? (note: services may not survive WSL restarts)"
            )
        } else {
            format!("Install the gateway as a {service_kind} service?")
        };
        if prompt_gateway_yes_no(input, output, question.as_str(), true)? {
            match print_gateway_install(
                context,
                accept_hooks,
                GatewayInstallArgs {
                    force: false,
                    system: false,
                    run_as_user: None,
                },
            ) {
                Ok(()) => {
                    if prompt_gateway_yes_no(input, output, "Start the service now?", true)? {
                        if let Err(error) = print_gateway_start(
                            context,
                            accept_hooks,
                            GatewayServiceArgs {
                                system: false,
                                all: false,
                            },
                        ) {
                            writeln!(output, "Start failed: {error}")?;
                        }
                    }
                }
                Err(error) => {
                    writeln!(output, "Install failed: {error}")?;
                }
            }
        } else {
            writeln!(output, "You can install later: hermes gateway install")?;
            if supports_systemd_services() {
                writeln!(
                    output,
                    "Or as a boot-time service: sudo hermes gateway install --system"
                )?;
            }
            writeln!(output, "Or run in foreground: hermes gateway run")?;
        }
    } else if is_wsl() {
        writeln!(output, "WSL detected but systemd is not running.")?;
        writeln!(output, "Run in foreground: hermes gateway run")?;
        writeln!(
            output,
            "For persistence:   tmux new -s hermes 'hermes gateway run'"
        )?;
        writeln!(
            output,
            "To enable systemd: add systemd=true to /etc/wsl.conf, then 'wsl --shutdown'"
        )?;
    } else if is_termux(context) {
        writeln!(output, "Termux does not use systemd/launchd services.")?;
        writeln!(output, "Run in foreground: hermes gateway run")?;
        writeln!(
            output,
            "Or start it manually in the background (best effort): nohup hermes gateway run >{}/logs/gateway.log 2>&1 &",
            context.display_hermes_home()
        )?;
    } else {
        writeln!(output, "Service install not supported on this platform.")?;
        writeln!(output, "Run in foreground: hermes gateway run")?;
    }

    writeln!(output)?;
    Ok(())
}

fn load_gateway_setup_metadata(
    accept_hooks: bool,
) -> Result<Vec<GatewaySetupPlatform>, Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON"))
        .ok_or("could not find a Python interpreter for gateway setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string());
    if accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }
    command.arg("-c").arg(GATEWAY_SETUP_METADATA_BOOTSTRAP);

    let output = command.output()?;
    if !output.status.success() {
        return Err(exit_status_message("gateway metadata", output.status).into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    serde_json::from_str::<Vec<GatewaySetupPlatform>>(stdout.trim())
        .map_err(|error| format!("invalid gateway setup metadata: {error}").into())
}

fn run_gateway_platform_setup_bridge(
    accept_hooks: bool,
    platform_key: &str,
) -> Result<(), Box<dyn Error>> {
    if platform_key.trim().is_empty() {
        return Err("gateway platform key cannot be empty".into());
    }
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON"))
        .ok_or("could not find a Python interpreter for gateway setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_GATEWAY_SETUP_PLATFORM", platform_key);
    if accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }
    command.arg("-c").arg(GATEWAY_SETUP_PLATFORM_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("gateway platform setup", status).into())
}

fn gateway_platform_uses_native_standard_setup(platform: &GatewaySetupPlatform) -> bool {
    !platform.vars.is_empty() && !platform.has_builtin_setup && !platform.has_plugin_setup
}

fn gateway_platform_status_is_progress(status: &str) -> bool {
    let lowered = status.trim().to_ascii_lowercase();
    !(lowered == "not configured"
        || lowered.starts_with("partially")
        || lowered.starts_with("plugin disabled"))
}

fn configure_standard_gateway_platform_with_io(
    context: &HermesContext,
    platform: &GatewaySetupPlatform,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "─── {} {} Setup ───",
        platform.emoji, platform.label
    )?;
    if !platform.setup_instructions.is_empty() {
        writeln!(output)?;
        for line in &platform.setup_instructions {
            writeln!(output, "  {line}")?;
        }
    }

    if let Some(existing) = read_effective_env_value(context, platform.token_var.as_str())
        .filter(|_| !platform.token_var.is_empty())
    {
        let _ = existing;
        writeln!(output)?;
        writeln!(output, "{} is already configured.", platform.label)?;
        if !prompt_gateway_yes_no(
            input,
            output,
            format!("Reconfigure {}?", platform.label).as_str(),
            false,
        )? {
            return Ok(());
        }
    }

    let mut allowlist_value: Option<String> = None;
    for var in &platform.vars {
        writeln!(output)?;
        if !var.help.trim().is_empty() {
            writeln!(output, "  {}", var.help.trim())?;
        }
        let existing = read_effective_env_value(context, &var.name);
        if !var.password
            && let Some(current) = existing.as_deref().filter(|value| !value.trim().is_empty())
        {
            writeln!(output, "  Current: {current}")?;
        }

        if var.is_allowlist {
            writeln!(output, "  The gateway denies all users by default.")?;
            writeln!(
                output,
                "  Enter user IDs to create an allowlist, or leave empty to choose another access mode."
            )?;
            let value = prompt_gateway_line(input, output, format!("  {}", var.prompt).as_str())?;
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                let cleaned = normalize_gateway_allowlist(&var.name, trimmed);
                save_env_value(context.env_path(), &var.name, &cleaned)?;
                remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
                writeln!(
                    output,
                    "  Saved — only these users can interact with the bot."
                )?;
                allowlist_value = Some(cleaned);
            } else {
                let access_choices = [
                    "Enable open access (anyone can message the bot)",
                    "Use DM pairing (unknown users request access, you approve later)",
                    "Skip for now (bot will deny all users until configured)",
                ];
                let access_idx = prompt_gateway_menu_choice(
                    input,
                    output,
                    "How should unauthorized users be handled?",
                    &access_choices,
                )?;
                match access_idx {
                    0 => {
                        save_env_value(context.env_path(), "GATEWAY_ALLOW_ALL_USERS", "true")?;
                        writeln!(output, "  Open access enabled.")?;
                    }
                    1 | 2 => {
                        remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
                        if access_idx == 1 {
                            writeln!(
                                output,
                                "  DM pairing mode selected. Approve codes with `hermes pairing approve`."
                            )?;
                        } else {
                            writeln!(
                                output,
                                "  Skipped — configure later with `hermes gateway setup`."
                            )?;
                        }
                    }
                    _ => {}
                }
            }
            continue;
        }

        let value = prompt_gateway_line(input, output, format!("  {}", var.prompt).as_str())?;
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            save_env_value(context.env_path(), &var.name, trimmed)?;
            writeln!(output, "  Saved {}", var.name)?;
        } else if var.name == platform.token_var {
            writeln!(
                output,
                "  Skipped — {} won't work without this.",
                platform.label
            )?;
            return Ok(());
        } else {
            writeln!(output, "  Skipped (can configure later)")?;
        }
    }

    if platform.key == "telegram"
        && let Some(allowlist) = allowlist_value
        && read_effective_env_value(context, "TELEGRAM_HOME_CHANNEL").is_none()
        && let Some(first_id) = allowlist
            .split(',')
            .map(str::trim)
            .find(|value| !value.is_empty())
        && prompt_gateway_yes_no(
            input,
            output,
            format!("Use your user ID ({first_id}) as the home channel?").as_str(),
            true,
        )?
    {
        save_env_value(context.env_path(), "TELEGRAM_HOME_CHANNEL", first_id)?;
        writeln!(output, "  Home channel set to {first_id}")?;
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    Ok(())
}

fn prompt_gateway_menu_choice(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    title: &str,
    choices: &[&str],
) -> Result<usize, Box<dyn Error>> {
    loop {
        writeln!(output, "{title}:")?;
        for (index, choice) in choices.iter().enumerate() {
            writeln!(output, "  {}. {}", index + 1, choice)?;
        }
        let response = prompt_gateway_line(input, output, "Enter a number")?;
        let trimmed = response.trim();
        if trimmed.is_empty() {
            writeln!(output, "Please enter a selection.")?;
            continue;
        }
        let Ok(index) = trimmed.parse::<usize>() else {
            writeln!(output, "Invalid selection: '{trimmed}'.")?;
            continue;
        };
        if !(1..=choices.len()).contains(&index) {
            writeln!(output, "Selection must be between 1 and {}.", choices.len())?;
            continue;
        }
        return Ok(index - 1);
    }
}

fn prompt_gateway_yes_no(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    prompt: &str,
    default: bool,
) -> Result<bool, Box<dyn Error>> {
    let suffix = if default { " [Y/n]" } else { " [y/N]" };
    loop {
        let response = prompt_gateway_line(input, output, format!("{prompt}{suffix}").as_str())?;
        let trimmed = response.trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            return Ok(default);
        }
        match trimmed.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => {
                writeln!(output, "Please answer yes or no.")?;
            }
        }
    }
}

fn prompt_gateway_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    prompt: &str,
) -> Result<String, Box<dyn Error>> {
    write!(output, "{prompt}: ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err("interactive input closed".into());
    }
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
}

fn read_effective_env_value(context: &HermesContext, key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| read_env_file_value(&context.env_path(), key))
}

fn read_env_file_value(path: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        if !rest.starts_with('=') {
            continue;
        }
        let value = rest[1..].trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn remove_env_key_if_present(path: &Path, key: &str) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return Ok(());
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
    if removed {
        fs::write(path, kept.concat())?;
    }
    Ok(())
}

fn normalize_gateway_allowlist(var_name: &str, value: &str) -> String {
    let cleaned = value.replace(' ', "");
    if !var_name.contains("DISCORD") {
        return cleaned;
    }
    cleaned
        .split(',')
        .filter_map(|entry| {
            let mut value = entry.trim();
            if value.is_empty() {
                return None;
            }
            if value.starts_with("<@") && value.ends_with('>') {
                value = value.trim_start_matches("<@").trim_start_matches('!');
                value = value.trim_end_matches('>');
            }
            if let Some(stripped) = value.strip_prefix("user:") {
                value = stripped;
            }
            (!value.is_empty()).then(|| value.to_string())
        })
        .collect::<Vec<_>>()
        .join(",")
}

const GATEWAY_SETUP_METADATA_BOOTSTRAP: &str = concat!(
    "import json\n",
    "from hermes_cli.gateway import _all_platforms, _platform_status, _builtin_setup_fn\n",
    "items = []\n",
    "for platform in _all_platforms():\n",
    "    entry = platform.get('_registry_entry')\n",
    "    vars_ = []\n",
    "    for spec in platform.get('vars') or []:\n",
    "        vars_.append({\n",
    "            'name': spec.get('name', ''),\n",
    "            'prompt': spec.get('prompt', ''),\n",
    "            'password': bool(spec.get('password', False)),\n",
    "            'help': spec.get('help', '') or '',\n",
    "            'is_allowlist': bool(spec.get('is_allowlist', False)),\n",
    "        })\n",
    "    items.append({\n",
    "        'key': platform.get('key', ''),\n",
    "        'label': platform.get('label', ''),\n",
    "        'emoji': platform.get('emoji', ''),\n",
    "        'status': _platform_status(platform),\n",
    "        'token_var': platform.get('token_var', '') or '',\n",
    "        'install_hint': platform.get('install_hint'),\n",
    "        'setup_instructions': list(platform.get('setup_instructions') or []),\n",
    "        'required_env': list(getattr(entry, 'required_env', []) or []),\n",
    "        'has_builtin_setup': _builtin_setup_fn(platform.get('key', '')) is not None,\n",
    "        'has_plugin_setup': bool(entry is not None and getattr(entry, 'setup_fn', None) is not None),\n",
    "        'vars': vars_,\n",
    "    })\n",
    "print(json.dumps(items))\n",
);

const GATEWAY_SETUP_PLATFORM_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.gateway import _all_platforms, _configure_platform\n",
    "target = os.environ['HERMES_GATEWAY_SETUP_PLATFORM']\n",
    "for platform in _all_platforms():\n",
    "    if platform.get('key') == target:\n",
    "        _configure_platform(platform)\n",
    "        break\n",
    "else:\n",
    "    raise SystemExit(f'unknown gateway platform: {target}')\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

impl GatewaySnapshot {
    fn has_process_service_mismatch(&self) -> bool {
        self.service_installed && !self.service_running && !self.gateway_pids.is_empty()
    }
}

fn gateway_snapshot(context: &HermesContext, system: bool) -> GatewaySnapshot {
    let pids = gateway_pids_for_profile(&context.hermes_home());
    if is_termux(context) {
        return GatewaySnapshot {
            manager: "Termux / manual process".to_string(),
            service_installed: false,
            service_running: false,
            gateway_pids: pids,
            service_scope: None,
        };
    }
    if is_linux() && is_container() && !supports_systemd_services() {
        return GatewaySnapshot {
            manager: "docker (foreground)".to_string(),
            service_installed: false,
            service_running: false,
            gateway_pids: pids,
            service_scope: None,
        };
    }
    if supports_systemd_services() {
        let selected_system = select_systemd_scope(context, system);
        let unit_path = systemd_unit_path(context, selected_system);
        let installed = unit_path.exists();
        let running = installed && systemd_service_active(context, selected_system);
        return GatewaySnapshot {
            manager: format!(
                "systemd ({})",
                if selected_system { "system" } else { "user" }
            ),
            service_installed: installed,
            service_running: running,
            gateway_pids: pids,
            service_scope: Some(if selected_system {
                "system".to_string()
            } else {
                "user".to_string()
            }),
        };
    }
    if is_macos() {
        let plist_path = launchd_plist_path(context);
        return GatewaySnapshot {
            manager: "launchd".to_string(),
            service_installed: plist_path.exists(),
            service_running: plist_path.exists() && launchd_service_active(context),
            gateway_pids: pids,
            service_scope: Some("launchd".to_string()),
        };
    }
    GatewaySnapshot {
        manager: "manual process".to_string(),
        service_installed: false,
        service_running: false,
        gateway_pids: pids,
        service_scope: None,
    }
}

fn gateway_service_name(context: &HermesContext) -> String {
    let home = context.hermes_home();
    let default_root = context.home_dir().join(".hermes");
    if home == default_root {
        return SERVICE_BASE.to_string();
    }
    let profiles_root = default_root.join("profiles");
    if let Ok(relative) = home.strip_prefix(&profiles_root) {
        let mut parts = relative.components();
        if let Some(first) = parts.next() {
            let candidate = first.as_os_str().to_string_lossy();
            if parts.next().is_none() && is_valid_profile_id(&candidate) {
                return format!("{SERVICE_BASE}-{candidate}");
            }
        }
    }
    let hash = Sha256::digest(home.display().to_string().as_bytes());
    let short = hash[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{SERVICE_BASE}-{short}")
}

fn systemd_unit_path(context: &HermesContext, system: bool) -> PathBuf {
    let service = gateway_service_name(context);
    if system
        && let Some(fake) = env::var_os("HERMES_FAKE_SYSTEMD_DIR")
            .map(PathBuf::from)
            .filter(|path| !path.as_os_str().is_empty())
    {
        return fake.join(format!("{service}.service"));
    }
    if system {
        PathBuf::from("/etc/systemd/system").join(format!("{service}.service"))
    } else {
        context
            .home_dir()
            .join(".config")
            .join("systemd")
            .join("user")
            .join(format!("{service}.service"))
    }
}

fn launchd_label(context: &HermesContext) -> String {
    let name = gateway_service_name(context);
    if name == SERVICE_BASE {
        "ai.hermes.gateway".to_string()
    } else {
        format!(
            "ai.hermes.gateway-{}",
            name.trim_start_matches(&format!("{SERVICE_BASE}-"))
        )
    }
}

fn launchd_plist_path(context: &HermesContext) -> PathBuf {
    context
        .home_dir()
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{}.plist", launchd_label(context)))
}

fn gateway_pids_for_profile(home: &Path) -> Vec<i64> {
    let pid_path = home.join("gateway.pid");
    let Some(pid) = read_gateway_pid(&pid_path) else {
        return Vec::new();
    };
    if process_running(pid) {
        vec![pid]
    } else {
        Vec::new()
    }
}

fn read_gateway_pid(path: &Path) -> Option<i64> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('{') {
        let value = serde_json::from_str::<JsonValue>(trimmed).ok()?;
        return value.get("pid").and_then(JsonValue::as_i64);
    }
    trimmed.parse::<i64>().ok()
}

fn runtime_health_lines(context: &HermesContext) -> Result<Vec<String>, Box<dyn Error>> {
    let path = context.hermes_home().join("gateway_state.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path)?;
    let value: JsonValue = serde_json::from_str(&raw)?;
    let Some(state) = value.as_object() else {
        return Ok(Vec::new());
    };
    let mut lines = Vec::new();
    if let Some(platforms) = state.get("platforms").and_then(JsonValue::as_object) {
        for (platform, pdata) in platforms {
            if pdata.get("state").and_then(JsonValue::as_str) == Some("fatal") {
                let message = pdata
                    .get("error_message")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("unknown error");
                lines.push(format!("⚠ {platform}: {message}"));
            }
        }
    }
    let gateway_state = state.get("gateway_state").and_then(JsonValue::as_str);
    let exit_reason = state.get("exit_reason").and_then(JsonValue::as_str);
    let restart_requested = state
        .get("restart_requested")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let active_agents = state
        .get("active_agents")
        .and_then(JsonValue::as_i64)
        .unwrap_or(0);
    match gateway_state {
        Some("startup_failed") => {
            if let Some(reason) = exit_reason {
                lines.push(format!("⚠ Last startup issue: {reason}"));
            }
        }
        Some("draining") => {
            let action = if restart_requested {
                "restart"
            } else {
                "shutdown"
            };
            lines.push(format!(
                "⏳ Gateway draining for {action} ({active_agents} active agent(s))"
            ));
        }
        Some("stopped") => {
            if let Some(reason) = exit_reason {
                lines.push(format!("⚠ Last shutdown reason: {reason}"));
            }
        }
        _ => {}
    }
    Ok(lines)
}

fn other_profile_gateway_processes(
    context: &HermesContext,
) -> Result<Vec<(String, i64)>, Box<dyn Error>> {
    let current = context.current_profile_name();
    let mut rows = Vec::new();
    let root = context.profiles_root();
    if !root.is_dir() {
        return Ok(rows);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if name == current || !is_valid_profile_id(&name) {
            continue;
        }
        let pid = read_gateway_pid(&path.join("gateway.pid"));
        if let Some(pid) = pid.filter(|pid| process_running(*pid)) {
            rows.push((name, pid));
        }
    }
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(rows)
}

fn find_all_gateway_processes() -> Result<Vec<GatewayProcess>, Box<dyn Error>> {
    let exclude = ancestor_pids();
    #[cfg(windows)]
    {
        let _ = exclude;
        Ok(Vec::new())
    }
    #[cfg(not(windows))]
    {
        let output = Command::new("ps")
            .args(["-A", "eww", "-o", "pid=,command="])
            .output()?;
        if !output.status.success() {
            return Ok(Vec::new());
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(parse_gateway_ps_processes(&stdout, &exclude))
    }
}

fn collect_profile_homes(
    context: &HermesContext,
) -> Result<Vec<(String, PathBuf)>, Box<dyn Error>> {
    let mut homes = Vec::new();
    let default_home = context.default_hermes_root();
    if default_home.is_dir() {
        homes.push((String::from("default"), default_home));
    }
    let profiles_root = context.profiles_root();
    if profiles_root.is_dir() {
        let mut entries = fs::read_dir(&profiles_root)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !is_valid_profile_id(&name) {
                continue;
            }
            homes.push((name, path));
        }
    }
    Ok(homes)
}

#[cfg(not(windows))]
fn parse_gateway_ps_processes(output: &str, exclude_pids: &[i64]) -> Vec<GatewayProcess> {
    output
        .lines()
        .filter_map(|line| {
            let stripped = line.trim();
            if stripped.is_empty() || stripped.contains("grep") {
                return None;
            }
            let mut parts = stripped.splitn(2, char::is_whitespace);
            let pid = parts.next()?.trim().parse::<i64>().ok()?;
            let command = parts.next().unwrap_or("").trim().to_string();
            gateway_process_matches(pid, &command, exclude_pids)
                .then_some(GatewayProcess { pid, command })
        })
        .collect()
}

fn gateway_process_matches(pid: i64, command: &str, exclude_pids: &[i64]) -> bool {
    pid > 0
        && !exclude_pids.contains(&pid)
        && GATEWAY_PROCESS_PATTERNS
            .iter()
            .any(|pattern| command.contains(pattern))
}

fn ancestor_pids() -> Vec<i64> {
    let mut ancestors = Vec::new();
    let mut pid = std::process::id() as i64;
    for _ in 0..64 {
        if pid <= 0 || ancestors.contains(&pid) {
            break;
        }
        ancestors.push(pid);
        let Some(parent) = parent_pid(pid) else {
            break;
        };
        pid = parent;
    }
    ancestors
}

fn parent_pid(pid: i64) -> Option<i64> {
    if pid <= 1 {
        return None;
    }
    let output = Command::new("ps")
        .args(["-o", "ppid=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let parent = stdout.trim().parse::<i64>().ok()?;
    (parent > 0).then_some(parent)
}

fn kill_gateway_processes(processes: &[GatewayProcess], force: bool) -> usize {
    let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
    let mut killed = 0;
    for process in processes {
        if !process_running(process.pid) {
            continue;
        }
        match signal_pid(process.pid, signal) {
            Ok(()) => killed += 1,
            Err(_) if !process_running(process.pid) => {}
            Err(error) => eprintln!("Failed to kill PID {}: {}", process.pid, error),
        }
    }
    killed
}

fn wait_for_processes_exit(pids: &[i64], timeout: Duration, force_after: Option<Duration>) -> bool {
    let mut pending = pids
        .iter()
        .copied()
        .filter(|pid| process_running(*pid))
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return true;
    }

    let start = Instant::now();
    let force_deadline = force_after.map(|value| start + value);
    let deadline = start + timeout;
    let mut force_sent = false;

    loop {
        pending.retain(|pid| process_running(*pid));
        if pending.is_empty() {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return pending.is_empty();
        }
        if !force_sent && force_deadline.is_some_and(|value| now >= value) {
            for pid in &pending {
                let _ = signal_pid(*pid, libc::SIGKILL);
            }
            force_sent = true;
        }
        sleep(Duration::from_millis(300));
    }
}

fn stop_gateway_service_if_available(context: &HermesContext, system: bool) -> bool {
    if has_any_systemd_unit(context) {
        return stop_systemd_service(context, system).is_ok();
    }
    if launchd_plist_path(context).exists() {
        return stop_launchd_service(context).is_ok();
    }
    false
}

fn list_active_systemd_gateway_services(system: bool) -> Result<Vec<String>, Box<dyn Error>> {
    if which_on_path("systemctl").is_none() {
        return Ok(Vec::new());
    }
    let output = build_systemctl_command(
        system,
        &[
            "list-units",
            "hermes-gateway*",
            "--plain",
            "--no-legend",
            "--no-pager",
        ],
    )
    .output()?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut services = Vec::new();
    for line in stdout.lines() {
        let Some(unit) = line.split_whitespace().next() else {
            continue;
        };
        if !unit.ends_with(".service") {
            continue;
        }
        let service = unit.trim_end_matches(".service").to_string();
        if !services.iter().any(|value| value == &service) {
            services.push(service);
        }
    }
    Ok(services)
}

fn restart_named_systemd_service(system: bool, service: &str) -> Result<(), Box<dyn Error>> {
    let output = build_systemctl_command(system, &["restart", service]).output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(command_failure_message("systemctl", &output).into())
}

fn systemd_main_pid(system: bool, service: &str) -> Option<i64> {
    let output =
        build_systemctl_command(system, &["show", service, "--property=MainPID", "--value"])
            .output()
            .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()
        .filter(|pid| *pid > 0)
}

fn collect_running_service_gateway_pids(
    context: &HermesContext,
) -> Result<Vec<i64>, Box<dyn Error>> {
    let mut pids = Vec::new();
    for system in [false, true] {
        for service in list_active_systemd_gateway_services(system)? {
            if let Some(pid) = systemd_main_pid(system, &service)
                && !pids.contains(&pid)
            {
                pids.push(pid);
            }
        }
    }
    if is_macos() {
        for (_, home) in collect_profile_homes(context)? {
            let profile_context = context.clone().with_hermes_home_env(Some(home.clone()));
            if !(launchd_plist_path(&profile_context).exists()
                && launchd_service_active(&profile_context))
            {
                continue;
            }
            for pid in gateway_pids_for_profile(&home) {
                if !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
        }
    }
    Ok(pids)
}

fn systemd_service_active(context: &HermesContext, system: bool) -> bool {
    if which_on_path("systemctl").is_none() {
        return false;
    }
    let mut command = Command::new("systemctl");
    if !system {
        command.arg("--user");
    }
    command
        .arg("is-active")
        .arg(gateway_service_name(context))
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|value| value.trim() == "active")
}

fn has_any_systemd_unit(context: &HermesContext) -> bool {
    systemd_unit_path(context, false).exists() || systemd_unit_path(context, true).exists()
}

fn require_systemd_service_installed(
    context: &HermesContext,
    system: bool,
    action: &str,
) -> Result<bool, Box<dyn Error>> {
    let selected_system = select_systemd_scope(context, system);
    let unit_path = systemd_unit_path(context, selected_system);
    if !unit_path.exists() {
        let scope_flag = if selected_system { " --system" } else { "" };
        let prefix = if selected_system { "sudo " } else { "" };
        return Err(
            format!(
                "Gateway service is not installed. Run: {prefix}hermes gateway install{scope_flag} before {action}."
            )
            .into(),
        );
    }
    if selected_system {
        require_root_for_system_service(action)?;
    }
    Ok(selected_system)
}

fn start_systemd_service(context: &HermesContext, system: bool) -> Result<(), Box<dyn Error>> {
    let selected_system = require_systemd_service_installed(context, system, "starting")?;
    run_systemctl(selected_system, &["start", &gateway_service_name(context)])?;
    Ok(())
}

fn stop_systemd_service(context: &HermesContext, system: bool) -> Result<(), Box<dyn Error>> {
    let selected_system = require_systemd_service_installed(context, system, "stopping")?;
    run_systemctl(selected_system, &["stop", &gateway_service_name(context)])?;
    Ok(())
}

fn restart_systemd_service(context: &HermesContext, system: bool) -> Result<(), Box<dyn Error>> {
    let selected_system = require_systemd_service_installed(context, system, "restarting")?;
    let _ = run_systemctl_allow_failure(
        selected_system,
        &["reset-failed", &gateway_service_name(context)],
    );
    run_systemctl(
        selected_system,
        &["reload-or-restart", &gateway_service_name(context)],
    )?;
    Ok(())
}

fn uninstall_systemd_service(
    context: &HermesContext,
    system: bool,
) -> Result<bool, Box<dyn Error>> {
    let selected_system = select_systemd_scope(context, system);
    let unit_path = systemd_unit_path(context, selected_system);
    if !unit_path.exists() {
        return Ok(false);
    }
    if selected_system {
        require_root_for_system_service("uninstall")?;
    }
    let service_name = gateway_service_name(context);
    let _ = run_systemctl_allow_failure(selected_system, &["stop", &service_name]);
    let _ = run_systemctl_allow_failure(selected_system, &["disable", &service_name]);
    fs::remove_file(&unit_path)?;
    let _ = run_systemctl_allow_failure(selected_system, &["daemon-reload"]);
    Ok(true)
}

fn legacy_unit_search_paths(context: &HermesContext) -> Vec<(bool, PathBuf)> {
    let mut paths = vec![(
        false,
        context
            .home_dir()
            .join(".config")
            .join("systemd")
            .join("user"),
    )];
    let system_base = env::var_os("HERMES_FAKE_SYSTEMD_DIR")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("/etc/systemd/system"));
    paths.push((true, system_base));
    paths
}

fn find_legacy_gateway_units(context: &HermesContext) -> Vec<LegacyGatewayUnit> {
    let mut units = Vec::new();
    for (is_system, base) in legacy_unit_search_paths(context) {
        for name in LEGACY_SERVICE_NAMES {
            let unit_path = base.join(name);
            let Ok(text) = fs::read_to_string(&unit_path) else {
                continue;
            };
            if !LEGACY_UNIT_EXECSTART_MARKERS
                .iter()
                .any(|marker| text.contains(marker))
            {
                continue;
            }
            units.push(LegacyGatewayUnit {
                name: (*name).to_string(),
                path: unit_path,
                is_system,
            });
        }
    }
    units
}

fn remove_legacy_gateway_units(
    legacy: &[LegacyGatewayUnit],
) -> Result<(usize, Vec<PathBuf>), Box<dyn Error>> {
    let mut removed = 0usize;
    let mut remaining = Vec::new();

    for unit in legacy {
        if unit.is_system && current_uid() != 0 {
            println!("System-scope legacy units require root to remove.");
            println!("  Re-run with: sudo hermes gateway migrate-legacy");
            remaining.push(unit.path.clone());
            continue;
        }

        if !unit.is_system {
            let _ = run_systemctl_allow_failure(false, &["stop", &unit.name]);
            let _ = run_systemctl_allow_failure(false, &["disable", &unit.name]);
        } else {
            let _ = run_systemctl_allow_failure(true, &["stop", &unit.name]);
            let _ = run_systemctl_allow_failure(true, &["disable", &unit.name]);
        }

        match fs::remove_file(&unit.path) {
            Ok(_) => {
                println!("  Removed {}", unit.path.display());
                removed += 1;
            }
            Err(error) => {
                println!("  Could not remove {}: {error}", unit.path.display());
                remaining.push(unit.path.clone());
            }
        }
    }

    let had_user = legacy.iter().any(|unit| !unit.is_system);
    let had_system = legacy.iter().any(|unit| unit.is_system);
    if had_user {
        let _ = run_systemctl_allow_failure(false, &["daemon-reload"]);
    }
    if had_system && current_uid() == 0 {
        let _ = run_systemctl_allow_failure(true, &["daemon-reload"]);
    }

    Ok((removed, remaining))
}

fn install_systemd_service(
    context: &HermesContext,
    force: bool,
    system: bool,
    run_as_user: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if system {
        require_root_for_system_service("install")?;
    }
    let unit_path = systemd_unit_path(context, system);
    let service_name = gateway_service_name(context);
    let unit = generate_systemd_unit(context, system, run_as_user)?;
    if unit_path.exists() && !force {
        let current = fs::read_to_string(&unit_path).unwrap_or_default();
        if current == unit {
            println!("Service already installed at: {}", unit_path.display());
            return Ok(());
        }
        return Err(format!(
            "Service already installed at: {}. Use --force to reinstall.",
            unit_path.display()
        )
        .into());
    }
    if let Some(parent) = unit_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&unit_path, unit)?;
    run_systemctl(system, &["daemon-reload"])?;
    run_systemctl(system, &["enable", &service_name])?;
    Ok(())
}

fn run_systemctl(system: bool, args: &[&str]) -> Result<(), Box<dyn Error>> {
    let output = build_systemctl_command(system, args).output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(command_failure_message("systemctl", &output).into())
}

fn run_systemctl_allow_failure(system: bool, args: &[&str]) -> Result<(), Box<dyn Error>> {
    let _ = build_systemctl_command(system, args).status()?;
    Ok(())
}

fn build_systemctl_command(system: bool, args: &[&str]) -> Command {
    let mut command = Command::new("systemctl");
    if !system {
        command.arg("--user");
    }
    command.args(args);
    command
}

fn generate_systemd_unit(
    context: &HermesContext,
    system: bool,
    run_as_user: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    let root = project_root();
    let mut gateway_binary = resolve_gateway_service_binary(context, None)?;
    let mut working_dir = root;
    let mut hermes_home = context.hermes_home();
    let profile_arg = gateway_profile_arg(context);
    let mut venv_dir = resolve_gateway_virtual_env(&working_dir);
    let mut path_entries = build_gateway_path_entries(&working_dir, venv_dir.as_deref());
    let mut wanted_by = "default.target";
    let mut identity = None;

    if system {
        let resolved = system_service_identity(run_as_user)?;
        gateway_binary =
            remap_path_for_target_user(&gateway_binary, context.home_dir(), &resolved.home_dir);
        working_dir =
            remap_path_for_target_user(&working_dir, context.home_dir(), &resolved.home_dir);
        hermes_home = remap_hermes_home_for_target_user(context, &resolved.home_dir);
        venv_dir = venv_dir
            .map(|path| remap_path_for_target_user(&path, context.home_dir(), &resolved.home_dir));
        path_entries = path_entries
            .into_iter()
            .map(|entry| {
                remap_path_for_target_user(
                    &PathBuf::from(&entry),
                    context.home_dir(),
                    &resolved.home_dir,
                )
            })
            .map(|path| path.display().to_string())
            .collect();
        append_user_local_paths(&resolved.home_dir, &mut path_entries);
        wanted_by = "multi-user.target";
        identity = Some(resolved);
    }

    path_entries.dedup();
    let sane_path = path_entries.join(":");
    let exec_start = if profile_arg.is_empty() {
        format!("{} gateway run --replace", gateway_binary.display())
    } else {
        format!(
            "{} {profile_arg} gateway run --replace",
            gateway_binary.display()
        )
    };

    let mut service_lines = vec![
        String::from("Type=simple"),
        format!("ExecStart={exec_start}"),
        format!("WorkingDirectory={}", working_dir.display()),
    ];
    if let Some(identity) = identity.as_ref() {
        service_lines.push(format!("User={}", identity.username));
        service_lines.push(format!("Group={}", identity.group_name));
        service_lines.push(format!(
            "Environment=\"HOME={}\"",
            identity.home_dir.display()
        ));
        service_lines.push(format!("Environment=\"USER={}\"", identity.username));
        service_lines.push(format!("Environment=\"LOGNAME={}\"", identity.username));
    }
    service_lines.push(format!("Environment=\"PATH={sane_path}\""));
    if let Some(venv_dir) = venv_dir.as_ref() {
        service_lines.push(format!(
            "Environment=\"VIRTUAL_ENV={}\"",
            venv_dir.display()
        ));
    }
    service_lines.extend([
        format!("Environment=\"HERMES_HOME={}\"", hermes_home.display()),
        String::from("Restart=always"),
        String::from("RestartSec=60"),
        String::from("RestartMaxDelaySec=300"),
        String::from("RestartSteps=5"),
        String::from("RestartForceExitStatus=75"),
        String::from("KillMode=mixed"),
        String::from("KillSignal=SIGTERM"),
        String::from("ExecReload=/bin/kill -USR1 $MAINPID"),
        String::from("TimeoutStopSec=330"),
        String::from("StandardOutput=journal"),
        String::from("StandardError=journal"),
    ]);

    Ok(format!(
        "[Unit]\nDescription=Hermes Agent Gateway - Messaging Platform Integration\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=0\n\n[Service]\n{}\n\n[Install]\nWantedBy={wanted_by}\n",
        service_lines.join("\n")
    ))
}

fn build_gateway_path_entries(root: &Path, venv_dir: Option<&Path>) -> Vec<String> {
    let mut path_entries = Vec::new();
    if let Some(venv_dir) = venv_dir {
        let venv_bin = venv_dir.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let rendered = venv_bin.display().to_string();
        if !rendered.is_empty() {
            path_entries.push(rendered);
        }
    }
    for candidate in [root.join("node_modules").join(".bin")] {
        let rendered = candidate.display().to_string();
        if !rendered.is_empty() {
            path_entries.push(rendered);
        }
    }
    if let Some(path) = env::var_os("PATH") {
        path_entries.extend(
            env::split_paths(&path)
                .map(|entry| entry.display().to_string())
                .filter(|entry| !entry.is_empty()),
        );
    }
    path_entries.dedup();
    path_entries
}

fn append_user_local_paths(home_dir: &Path, path_entries: &mut Vec<String>) {
    for suffix in [".local/bin", ".cargo/bin", "go/bin", ".npm-global/bin"] {
        let path = home_dir.join(suffix);
        if !path.exists() {
            continue;
        }
        let rendered = path.display().to_string();
        if !path_entries.iter().any(|entry| entry == &rendered) {
            path_entries.push(rendered);
        }
    }
}

fn remap_path_for_target_user(path: &Path, current_home: &Path, target_home: &Path) -> PathBuf {
    path.strip_prefix(current_home)
        .map(|relative| target_home.join(relative))
        .unwrap_or_else(|_| path.to_path_buf())
}

fn remap_hermes_home_for_target_user(context: &HermesContext, target_home: &Path) -> PathBuf {
    let current_hermes = context.hermes_home();
    let current_default = context.home_dir().join(".hermes");
    let target_default = target_home.join(".hermes");
    if current_hermes == current_default {
        return target_default;
    }
    current_hermes
        .strip_prefix(&current_default)
        .map(|relative| target_default.join(relative))
        .unwrap_or(current_hermes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SystemServiceIdentity {
    username: String,
    group_name: String,
    home_dir: PathBuf,
}

fn system_service_identity(
    run_as_user: Option<&str>,
) -> Result<SystemServiceIdentity, Box<dyn Error>> {
    if !is_linux() {
        return Err("system gateway install is not supported on this platform".into());
    }
    #[cfg(not(unix))]
    {
        let _ = run_as_user;
        Err("system gateway install is not supported on this platform".into())
    }
    #[cfg(unix)]
    {
        let username = run_as_user
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .or_else(|| env_nonempty("SUDO_USER"))
            .or_else(|| env_nonempty("USER"))
            .or_else(|| env_nonempty("LOGNAME"))
            .ok_or("Could not determine which user the gateway service should run as")?;
        if username == "root" && run_as_user.is_none() {
            return Err("Refusing to install the gateway system service as root; pass --run-as-user root to override".into());
        }
        if username == "root" {
            println!("Installing gateway service to run as root.");
        }
        let (gid, home_dir) = lookup_passwd(&username)?;
        let group_name = lookup_group_name(gid)?;
        Ok(SystemServiceIdentity {
            username,
            group_name,
            home_dir,
        })
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(unix)]
fn lookup_passwd(username: &str) -> Result<(libc::gid_t, PathBuf), Box<dyn Error>> {
    let username = CString::new(username)?;
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result = std::ptr::null_mut();
    let mut buffer = vec![0u8; 1024];
    loop {
        let status = unsafe {
            libc::getpwnam_r(
                username.as_ptr(),
                &mut pwd,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut result,
            )
        };
        if status == 0 {
            break;
        }
        if status == libc::ERANGE {
            buffer.resize(buffer.len() * 2, 0);
            continue;
        }
        return Err(std::io::Error::from_raw_os_error(status).into());
    }
    if result.is_null() {
        let requested = username.to_string_lossy().into_owned();
        return Err(format!("Unknown user: {requested}").into());
    }
    let home_dir = unsafe { CStr::from_ptr(pwd.pw_dir) }
        .to_str()
        .map(PathBuf::from)?;
    Ok((pwd.pw_gid, home_dir))
}

#[cfg(unix)]
fn lookup_group_name(gid: libc::gid_t) -> Result<String, Box<dyn Error>> {
    let mut grp: libc::group = unsafe { std::mem::zeroed() };
    let mut result = std::ptr::null_mut();
    let mut buffer = vec![0u8; 1024];
    loop {
        let status = unsafe {
            libc::getgrgid_r(
                gid,
                &mut grp,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut result,
            )
        };
        if status == 0 {
            break;
        }
        if status == libc::ERANGE {
            buffer.resize(buffer.len() * 2, 0);
            continue;
        }
        return Err(std::io::Error::from_raw_os_error(status).into());
    }
    if result.is_null() {
        return Err(format!("Unknown group id: {gid}").into());
    }
    Ok(unsafe { CStr::from_ptr(grp.gr_name) }.to_str()?.to_string())
}

fn launchd_service_active(context: &HermesContext) -> bool {
    Command::new("launchctl")
        .arg("list")
        .arg(launchd_label(context))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn start_launchd_service(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let plist_path = launchd_plist_path(context);
    if !plist_path.exists() {
        return Err("Gateway service is not installed. Run: hermes gateway install".into());
    }
    let target = launchd_target(context);
    let output = Command::new("launchctl")
        .args(["kickstart", &target])
        .output()?;
    if output.status.success() {
        return Ok(());
    }
    match output.status.code() {
        Some(3) | Some(113) => {
            let domain = launchd_domain();
            let plist = plist_path.display().to_string();
            run_launchctl(&["bootstrap", &domain, &plist])?;
            run_launchctl(&["kickstart", &target])?;
            Ok(())
        }
        _ => Err(command_failure_message("launchctl", &output).into()),
    }
}

fn stop_launchd_service(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let target = launchd_target(context);
    let output = Command::new("launchctl")
        .args(["bootout", &target])
        .output()?;
    if !(output.status.success() || matches!(output.status.code(), Some(3) | Some(113))) {
        return Err(command_failure_message("launchctl", &output).into());
    }
    let _ = wait_for_gateway_exit(
        &context.hermes_home(),
        Duration::from_secs(10),
        Some(Duration::from_secs(5)),
    );
    Ok(())
}

fn restart_launchd_service(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let plist_path = launchd_plist_path(context);
    if !plist_path.exists() {
        return Err("Gateway service is not installed. Run: hermes gateway install".into());
    }
    let target = launchd_target(context);
    let output = Command::new("launchctl")
        .args(["kickstart", "-k", &target])
        .output()?;
    if output.status.success() {
        return Ok(());
    }
    match output.status.code() {
        Some(3) | Some(113) => {
            let domain = launchd_domain();
            let plist = plist_path.display().to_string();
            run_launchctl(&["bootstrap", &domain, &plist])?;
            run_launchctl(&["kickstart", &target])?;
            Ok(())
        }
        _ => Err(command_failure_message("launchctl", &output).into()),
    }
}

fn uninstall_launchd_service(context: &HermesContext) -> Result<bool, Box<dyn Error>> {
    let plist_path = launchd_plist_path(context);
    if !plist_path.exists() {
        return Ok(false);
    }
    let target = launchd_target(context);
    let output = Command::new("launchctl")
        .args(["bootout", &target])
        .output()?;
    if !(output.status.success() || matches!(output.status.code(), Some(3) | Some(113))) {
        return Err(command_failure_message("launchctl", &output).into());
    }
    fs::remove_file(&plist_path)?;
    Ok(true)
}

fn install_launchd_service(context: &HermesContext, force: bool) -> Result<(), Box<dyn Error>> {
    let plist_path = launchd_plist_path(context);
    let plist = generate_launchd_plist(context)?;
    if plist_path.exists() && !force {
        let current = fs::read_to_string(&plist_path).unwrap_or_default();
        if current == plist {
            println!("Service already installed at: {}", plist_path.display());
            return Ok(());
        }
        return Err(format!(
            "Service already installed at: {}. Use --force to reinstall.",
            plist_path.display()
        )
        .into());
    }
    if let Some(parent) = plist_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&plist_path, plist)?;
    let domain = launchd_domain();
    let plist_arg = plist_path.display().to_string();
    run_launchctl(&["bootstrap", &domain, &plist_arg])?;
    Ok(())
}

fn run_launchctl(args: &[&str]) -> Result<(), Box<dyn Error>> {
    let output = Command::new("launchctl").args(args).output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(command_failure_message("launchctl", &output).into())
}

fn launchd_domain() -> String {
    format!("gui/{}", current_uid())
}

fn launchd_target(context: &HermesContext) -> String {
    format!("{}/{}", launchd_domain(), launchd_label(context))
}

fn generate_launchd_plist(context: &HermesContext) -> Result<String, Box<dyn Error>> {
    let root = project_root();
    let gateway_binary = resolve_gateway_service_binary(context, None)?;
    let working_dir = root.display().to_string();
    let hermes_home = context.hermes_home().display().to_string();
    let log_dir = context.hermes_home().join("logs");
    let profile_arg = gateway_profile_arg(context);
    let venv_dir = resolve_gateway_virtual_env(&root);
    let path_entries = build_gateway_path_entries(&root, venv_dir.as_deref());
    let sane_path = path_entries.join(":");

    let mut program_args = vec![gateway_binary.display().to_string()];
    if !profile_arg.is_empty() {
        program_args.extend(profile_arg.split_whitespace().map(str::to_string));
    }
    program_args.extend([
        String::from("gateway"),
        String::from("run"),
        String::from("--replace"),
    ]);
    let args_xml = program_args
        .iter()
        .map(|arg| format!("        <string>{}</string>", xml_escape(arg)))
        .collect::<Vec<_>>()
        .join("\n");

    let mut env_xml = vec![
        format!("        <key>PATH</key>\n        <string>{sane_path}</string>"),
        format!("        <key>HERMES_HOME</key>\n        <string>{hermes_home}</string>"),
    ];
    if let Some(venv_dir) = venv_dir.as_ref() {
        env_xml.insert(
            1,
            format!(
                "        <key>VIRTUAL_ENV</key>\n        <string>{}</string>",
                venv_dir.display()
            ),
        );
    }

    Ok(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>{}</string>\n\n    <key>ProgramArguments</key>\n    <array>\n{}\n    </array>\n\n    <key>WorkingDirectory</key>\n    <string>{working_dir}</string>\n\n    <key>EnvironmentVariables</key>\n    <dict>\n{}\n    </dict>\n\n    <key>RunAtLoad</key>\n    <true/>\n\n    <key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>\n\n    <key>StandardOutPath</key>\n    <string>{}/gateway.log</string>\n\n    <key>StandardErrorPath</key>\n    <string>{}/gateway.error.log</string>\n</dict>\n</plist>\n",
        launchd_label(context),
        args_xml,
        env_xml.join("\n"),
        log_dir.display(),
        log_dir.display()
    ))
}

fn resolve_gateway_service_binary(
    context: &HermesContext,
    target_home: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut binary = env::var_os("HERMES_GATEWAY_BINARY")
        .map(PathBuf::from)
        .unwrap_or(env::current_exe()?);
    if let Some(home_dir) = target_home {
        binary = remap_path_for_target_user(&binary, context.home_dir(), home_dir);
    }
    Ok(binary)
}

fn launch_detached_profile_gateway_after_update(
    context: &HermesContext,
    profile: &str,
) -> Result<(), Box<dyn Error>> {
    let gateway_binary = resolve_gateway_service_binary(context, None)?;
    let mut command = Command::new(gateway_binary);
    if profile != "default" {
        command.arg("--profile").arg(profile);
    }
    command
        .arg("gateway")
        .arg("run")
        .arg("--replace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let _child = command.spawn()?;
    Ok(())
}

fn stop_manual_gateway(context: &HermesContext) -> Result<bool, Box<dyn Error>> {
    let pids = gateway_pids_for_profile(&context.hermes_home());
    let Some(pid) = pids.first().copied() else {
        return Ok(false);
    };
    signal_pid(pid, libc::SIGTERM)?;
    let _ = wait_for_gateway_exit(
        &context.hermes_home(),
        Duration::from_secs(10),
        Some(Duration::from_secs(5)),
    );
    Ok(true)
}

fn gateway_profile_arg(context: &HermesContext) -> String {
    let profile = context.current_profile_name();
    if profile == "default" || profile == "custom" || !is_valid_profile_id(&profile) {
        String::new()
    } else {
        format!("--profile {profile}")
    }
}

fn derive_venv_dir(python: &Path) -> Option<PathBuf> {
    let parent = python.parent()?;
    let name = parent.file_name()?.to_string_lossy();
    if name != "bin" && name != "Scripts" {
        return None;
    }
    let venv_dir = parent.parent()?.to_path_buf();
    if venv_dir.join("pyvenv.cfg").exists() {
        return Some(venv_dir);
    }
    let file_name = venv_dir.file_name()?.to_string_lossy();
    if file_name == ".venv" || file_name == "venv" {
        return Some(venv_dir);
    }
    None
}

fn resolve_gateway_virtual_env(project_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env_nonempty("VIRTUAL_ENV") {
        return Some(PathBuf::from(value));
    }
    if let Some(venv_dir) = resolve_repo_python(project_root, Some("HERMES_GATEWAY_PYTHON"))
        .and_then(|python| derive_venv_dir(&python))
    {
        return Some(venv_dir);
    }
    for candidate in [
        project_root.join(".venv"),
        project_root.join("venv"),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
            .join("hermes-agent")
            .join("venv"),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
        .replace('\'', "&apos;")
}

fn confirm_legacy_removal() -> Result<bool, Box<dyn Error>> {
    print!("Remove these legacy units? [Y/n] ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_ascii_lowercase();
    Ok(choice.is_empty() || choice == "y" || choice == "yes")
}

fn wait_for_gateway_exit(
    hermes_home: &Path,
    timeout: Duration,
    force_after: Option<Duration>,
) -> bool {
    let start = Instant::now();
    let force_deadline = force_after.map(|value| start + value);
    let deadline = start + timeout;
    let mut force_sent = false;

    loop {
        if gateway_pids_for_profile(hermes_home).is_empty() {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return gateway_pids_for_profile(hermes_home).is_empty();
        }
        if !force_sent
            && force_deadline.is_some_and(|value| now >= value)
            && let Some(pid) = gateway_pids_for_profile(hermes_home).first().copied()
        {
            let _ = signal_pid(pid, libc::SIGKILL);
            force_sent = true;
        }
        sleep(Duration::from_millis(300));
    }
}

fn signal_pid(pid: i64, signal: i32) -> Result<(), Box<dyn Error>> {
    if pid <= 0 {
        return Err("invalid pid".into());
    }
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid as libc::pid_t, signal) };
        if result == 0 {
            return Ok(());
        }
        return Err(std::io::Error::last_os_error().into());
    }
    #[cfg(not(unix))]
    {
        let _ = signal;
        Err("process signalling is not supported on this platform".into())
    }
}

fn select_systemd_scope(context: &HermesContext, system: bool) -> bool {
    if system {
        return true;
    }
    systemd_unit_path(context, true).exists() && !systemd_unit_path(context, false).exists()
}

fn supports_systemd_services() -> bool {
    if !is_linux() || is_termux_detected() {
        return false;
    }
    if which_on_path("systemctl").is_none() {
        return false;
    }
    systemd_operational(false) || systemd_operational(true)
}

fn systemd_operational(system: bool) -> bool {
    let mut command = Command::new("systemctl");
    if !system {
        command.arg("--user");
    }
    command
        .arg("is-system-running")
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "running" | "degraded" | "starting" | "initializing"
            )
        })
}

fn systemd_linger_status() -> Option<(bool, String)> {
    if !is_linux() || is_termux_detected() {
        return None;
    }
    if which_on_path("loginctl").is_none() {
        return None;
    }
    let user = env::var("USER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("LOGNAME")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })?;
    let output = Command::new("loginctl")
        .args(["show-user", &user, "--property=Linger", "--value"])
        .output()
        .ok()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Some((false, detail));
    }
    let value = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_ascii_lowercase();
    match value.as_str() {
        "yes" | "true" | "1" => Some((true, String::new())),
        "no" | "false" | "0" => Some((false, String::new())),
        _ => Some((false, format!("unexpected loginctl output: {}", value))),
    }
}

fn require_root_for_system_service(action: &str) -> Result<(), Box<dyn Error>> {
    if current_uid() == 0 {
        return Ok(());
    }
    Err(
        format!("System gateway service {action} requires root. Run: sudo hermes gateway --system")
            .into(),
    )
}

fn current_uid() -> u32 {
    #[cfg(test)]
    if let Some(value) = env_nonempty("HERMES_TEST_EUID")
        && let Ok(parsed) = value.parse::<u32>()
    {
        return parsed;
    }
    #[cfg(unix)]
    {
        unsafe { libc::geteuid() as u32 }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

fn run_optional_status_command(binary: &str, args: &[String]) {
    let _ = Command::new(binary).args(args).status();
}

fn command_failure_message(command: &str, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stdout.is_empty() {
        return stdout;
    }
    match output.status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn process_running(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if result == 0 {
            return true;
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn format_pids(pids: &[i64], limit: Option<usize>) -> String {
    let render = match limit {
        Some(limit) => {
            let mut values = pids
                .iter()
                .take(limit)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if pids.len() > limit {
                values.push("...".to_string());
            }
            values
        }
        None => pids.iter().map(ToString::to_string).collect::<Vec<_>>(),
    };
    render.join(", ")
}

fn is_linux() -> bool {
    cfg!(target_os = "linux")
}

fn is_macos() -> bool {
    cfg!(target_os = "macos")
}

fn is_termux_detected() -> bool {
    env::var("TERMUX_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
        || env::var("PREFIX")
            .ok()
            .is_some_and(|value| value.contains("com.termux/files/usr"))
}

fn is_termux(context: &HermesContext) -> bool {
    let _ = context;
    is_termux_detected()
}

fn is_valid_profile_id(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes.iter().copied().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
    })
}

#[cfg(test)]
fn test_env_lock() -> &'static Mutex<()> {
    crate::cli_test_env_lock()
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::io::Cursor;
    use tempfile::TempDir;

    #[derive(Parser, Debug)]
    struct GatewayHarness {
        #[command(flatten)]
        args: GatewayArgs,
    }

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let ctx = HermesContext::new(&home).with_hermes_home_env(Some(home.join(".hermes")));
        (temp, ctx)
    }

    #[test]
    fn gateway_args_preserve_accept_hooks_and_status_flags() {
        let parsed = GatewayHarness::try_parse_from([
            "gateway",
            "--accept-hooks",
            "status",
            "--deep",
            "--full",
            "--system",
        ])
        .unwrap();
        assert!(parsed.args.accept_hooks);
        match parsed.args.command.unwrap() {
            GatewayCommand::Status(args) => {
                assert!(args.deep);
                assert!(args.full);
                assert!(args.system);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gateway_args_accept_hooks_after_subcommand() {
        let parsed = GatewayHarness::try_parse_from(["gateway", "run", "--accept-hooks"]).unwrap();
        assert!(parsed.args.accept_hooks);
        match parsed.args.command.unwrap() {
            GatewayCommand::Run(_) => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gateway_run_args_parse_replace_verbose_and_quiet() {
        let parsed =
            GatewayHarness::try_parse_from(["gateway", "run", "-vv", "--quiet", "--replace"])
                .unwrap();
        match parsed.args.command.unwrap() {
            GatewayCommand::Run(args) => {
                assert_eq!(args.verbose, 2);
                assert!(args.quiet);
                assert!(args.replace);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gateway_service_name_uses_profile_name_and_hashes_custom_paths() {
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.default_hermes_root()).unwrap();
        assert_eq!(gateway_service_name(&ctx), SERVICE_BASE);

        let profile_home = ctx.profiles_root().join("coder");
        let profile_ctx = ctx.clone().with_hermes_home_env(Some(profile_home));
        assert_eq!(gateway_service_name(&profile_ctx), "hermes-gateway-coder");

        let custom_home = ctx.home_dir().join("custom-home");
        let custom_ctx = ctx.clone().with_hermes_home_env(Some(custom_home));
        assert!(gateway_service_name(&custom_ctx).starts_with("hermes-gateway-"));
        assert_ne!(gateway_service_name(&custom_ctx), "hermes-gateway-coder");
    }

    #[test]
    fn gateway_snapshot_reports_manual_running_pid() {
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.hermes_home()).unwrap();
        let mut child = Command::new("sleep").arg("5").spawn().unwrap();
        fs::write(
            ctx.hermes_home().join("gateway.pid"),
            child.id().to_string(),
        )
        .unwrap();

        let snapshot = gateway_snapshot(&ctx, false);
        assert!(snapshot.gateway_pids.contains(&(child.id() as i64)));

        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn runtime_health_lines_extract_fatal_and_drain_messages() {
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.hermes_home()).unwrap();
        fs::write(
            ctx.hermes_home().join("gateway_state.json"),
            r#"{"gateway_state":"draining","restart_requested":true,"active_agents":2,"platforms":{"telegram":{"state":"fatal","error_message":"boom"}}}"#,
        )
        .unwrap();
        let lines = runtime_health_lines(&ctx).unwrap();
        assert!(lines.iter().any(|line| line.contains("telegram")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("draining for restart"))
        );
    }

    #[test]
    fn other_profile_gateway_processes_lists_running_named_profiles() {
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.profiles_root().join("builder")).unwrap();
        let mut child = Command::new("sleep").arg("5").spawn().unwrap();
        fs::write(
            ctx.profiles_root().join("builder").join("gateway.pid"),
            child.id().to_string(),
        )
        .unwrap();
        let rows = other_profile_gateway_processes(&ctx).unwrap();
        assert_eq!(rows, vec![("builder".to_string(), child.id() as i64)]);
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(not(windows))]
    fn parse_gateway_ps_processes_matches_run_commands() {
        let output = "\
123 /usr/bin/python -m hermes_cli.main gateway run --replace\n\
124 hermes chat hi\n\
125 /usr/bin/env HERMES_HOME=/tmp/x hermes gateway run\n";
        let processes = parse_gateway_ps_processes(output, &[999]);
        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].pid, 123);
        assert_eq!(processes[1].pid, 125);
    }

    #[test]
    #[cfg(unix)]
    fn gateway_run_uses_python_override_and_env_flags() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'accept=%s verbose=%s quiet=%s replace=%s\\n' \\\n\
    \"$HERMES_ACCEPT_HOOKS\" \"$HERMES_GATEWAY_VERBOSE\" \\\n\
    \"$HERMES_GATEWAY_QUIET\" \"$HERMES_GATEWAY_REPLACE\" >> '{}'\n\
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

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        print_gateway_run(
            true,
            GatewayRunArgs {
                verbose: 2,
                quiet: true,
                replace: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("accept=1 verbose=2 quiet=1 replace=1"));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn gateway_setup_metadata_uses_python_override_and_accept_hooks() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'metadata accept=%s platform=%s\\n' \"$HERMES_ACCEPT_HOOKS\" \"$HERMES_GATEWAY_SETUP_PLATFORM\" >> '{}'\n\
  cat <<'JSON'\n\
[{{\"key\":\"email\",\"label\":\"Email\",\"emoji\":\"@\",\"status\":\"not configured\",\"token_var\":\"EMAIL_ADDRESS\",\"vars\":[{{\"name\":\"EMAIL_ADDRESS\",\"prompt\":\"Email address\",\"help\":\"Mailbox address.\"}}]}}]\n\
JSON\n\
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

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        let metadata = load_gateway_setup_metadata(true).unwrap();

        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].key, "email");
        assert_eq!(metadata[0].label, "Email");
        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("metadata accept=1 platform="));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn gateway_setup_platform_bridge_uses_python_override_and_selected_key() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'platform accept=%s key=%s\\n' \"$HERMES_ACCEPT_HOOKS\" \"$HERMES_GATEWAY_SETUP_PLATFORM\" >> '{}'\n\
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

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        run_gateway_platform_setup_bridge(true, "telegram").unwrap();

        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("platform accept=1 key=telegram"));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    fn configure_standard_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = GatewaySetupPlatform {
            key: String::from("email"),
            label: String::from("Email"),
            emoji: String::from("@"),
            status: String::from("not configured"),
            token_var: String::from("EMAIL_ADDRESS"),
            install_hint: None,
            setup_instructions: vec![String::from("1. Use a dedicated mailbox.")],
            required_env: Vec::new(),
            has_builtin_setup: false,
            has_plugin_setup: false,
            vars: vec![
                GatewaySetupVar {
                    name: String::from("EMAIL_ADDRESS"),
                    prompt: String::from("Email address"),
                    password: false,
                    help: String::from("Mailbox address."),
                    is_allowlist: false,
                },
                GatewaySetupVar {
                    name: String::from("EMAIL_PASSWORD"),
                    prompt: String::from("Password"),
                    password: true,
                    help: String::from("App password."),
                    is_allowlist: false,
                },
                GatewaySetupVar {
                    name: String::from("EMAIL_IMAP_HOST"),
                    prompt: String::from("IMAP host"),
                    password: false,
                    help: String::from("IMAP server."),
                    is_allowlist: false,
                },
                GatewaySetupVar {
                    name: String::from("EMAIL_SMTP_HOST"),
                    prompt: String::from("SMTP host"),
                    password: false,
                    help: String::from("SMTP server."),
                    is_allowlist: false,
                },
                GatewaySetupVar {
                    name: String::from("EMAIL_ALLOWED_USERS"),
                    prompt: String::from("Allowed sender emails"),
                    password: false,
                    help: String::from("Trusted senders."),
                    is_allowlist: true,
                },
            ],
        };

        let mut input = Cursor::new(
            "bot@example.com\napp-password\nimap.example.com\nsmtp.example.com\nme@example.com,ops@example.com\n",
        );
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("EMAIL_ADDRESS=bot@example.com"));
        assert!(env_text.contains("EMAIL_PASSWORD=app-password"));
        assert!(env_text.contains("EMAIL_IMAP_HOST=imap.example.com"));
        assert!(env_text.contains("EMAIL_SMTP_HOST=smtp.example.com"));
        assert!(env_text.contains("EMAIL_ALLOWED_USERS=me@example.com,ops@example.com"));
    }

    #[test]
    #[cfg(unix)]
    fn gateway_install_rejects_run_as_user_without_system() {
        let (_temp, ctx) = test_context();
        let error = print_gateway_install(
            &ctx,
            false,
            GatewayInstallArgs {
                force: false,
                system: false,
                run_as_user: Some(String::from("alice")),
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--run-as-user requires --system"));
    }

    #[test]
    #[cfg(unix)]
    fn gateway_restart_all_kills_existing_process_and_relaunches() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'restart verbose=%s quiet=%s replace=%s\\n' \\\n\
    \"$HERMES_GATEWAY_VERBOSE\" \"$HERMES_GATEWAY_QUIET\" \"$HERMES_GATEWAY_REPLACE\" >> '{}'\n\
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

        let mut child = Command::new("bash")
            .args(["-lc", "exec -a 'hermes gateway run' sleep 30"])
            .spawn()
            .unwrap();

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Restart(GatewayServiceArgs {
                    system: false,
                    all: true,
                })),
            },
        )
        .unwrap();

        let status = child.wait().unwrap();
        assert!(!status.success());
        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("restart verbose=0 quiet=0 replace=0"));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    fn gateway_start_stop_restart_use_systemctl_for_user_units() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.hermes_home()).unwrap();
        let fake_bin = ctx.home_dir().join("bin");
        let log_path = ctx.home_dir().join("systemctl.log");
        fs::create_dir_all(&fake_bin).unwrap();
        let script = fake_bin.join("systemctl");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\ncase \"$*\" in\n  *is-system-running*) printf 'running\\n' ;;\nesac\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", fake_bin.display(), original_path));
        let unit_path = systemd_unit_path(&ctx, false);
        fs::create_dir_all(unit_path.parent().unwrap()).unwrap();
        fs::write(&unit_path, "unit").unwrap();

        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Start(GatewayServiceArgs {
                    system: false,
                    all: false,
                })),
            },
        )
        .unwrap();
        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Stop(GatewayServiceArgs {
                    system: false,
                    all: false,
                })),
            },
        )
        .unwrap();
        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Restart(GatewayServiceArgs {
                    system: false,
                    all: false,
                })),
            },
        )
        .unwrap();

        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("--user start hermes-gateway"));
        assert!(log.contains("--user stop hermes-gateway"));
        assert!(log.contains("--user reset-failed hermes-gateway"));
        assert!(log.contains("--user reload-or-restart hermes-gateway"));
        set_env_var("PATH", original_path);
    }

    #[test]
    fn gateway_stop_kills_manual_profile_process() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.hermes_home()).unwrap();
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        fs::write(
            ctx.hermes_home().join("gateway.pid"),
            format!("{{\"pid\":{}}}\n", child.id()),
        )
        .unwrap();

        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Stop(GatewayServiceArgs {
                    system: false,
                    all: false,
                })),
            },
        )
        .unwrap();

        let status = child.wait().unwrap();
        assert!(!status.success());
    }

    #[test]
    fn gateway_uninstall_removes_user_systemd_unit() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let fake_bin = ctx.home_dir().join("bin");
        let log_path = ctx.home_dir().join("systemctl-uninstall.log");
        fs::create_dir_all(&fake_bin).unwrap();
        let script = fake_bin.join("systemctl");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\ncase \"$*\" in\n  *is-system-running*) printf 'running\\n' ;;\nesac\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", fake_bin.display(), original_path));
        let unit_path = systemd_unit_path(&ctx, false);
        fs::create_dir_all(unit_path.parent().unwrap()).unwrap();
        fs::write(&unit_path, "unit").unwrap();

        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Uninstall(GatewaySystemArgs {
                    system: false,
                })),
            },
        )
        .unwrap();

        assert!(!unit_path.exists());
        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("--user stop hermes-gateway"));
        assert!(log.contains("--user disable hermes-gateway"));
        assert!(log.contains("--user daemon-reload"));
        set_env_var("PATH", original_path);
    }

    #[test]
    fn gateway_install_writes_user_systemd_unit() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let fake_bin = ctx.home_dir().join("bin");
        let log_path = ctx.home_dir().join("systemctl-install.log");
        fs::create_dir_all(&fake_bin).unwrap();
        let script = fake_bin.join("systemctl");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\ncase \"$*\" in\n  *is-system-running*) printf 'running\\n' ;;\nesac\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", fake_bin.display(), original_path));

        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::Install(GatewayInstallArgs {
                    force: false,
                    system: false,
                    run_as_user: None,
                })),
            },
        )
        .unwrap();

        let unit_path = systemd_unit_path(&ctx, false);
        let unit = fs::read_to_string(&unit_path).unwrap();
        let current_exe = env::current_exe().unwrap();
        assert!(unit.contains("gateway run --replace"));
        assert!(unit.contains(&current_exe.display().to_string()));
        assert!(!unit.contains("hermes_cli.main"));
        assert!(unit.contains("HERMES_HOME="));
        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("--user daemon-reload"));
        assert!(log.contains("--user enable hermes-gateway"));
        set_env_var("PATH", original_path);
    }

    #[test]
    fn gateway_install_writes_system_systemd_unit() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let fake_bin = ctx.home_dir().join("bin");
        let fake_systemd = ctx.home_dir().join("etc-systemd");
        let log_path = ctx.home_dir().join("systemctl-install-system.log");
        fs::create_dir_all(&fake_bin).unwrap();
        fs::create_dir_all(&fake_systemd).unwrap();
        let script = fake_bin.join("systemctl");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\ncase \"$*\" in\n  *is-system-running*) printf 'running\\n' ;;\nesac\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", fake_bin.display(), original_path));
        set_env_var("HERMES_FAKE_SYSTEMD_DIR", &fake_systemd);
        set_env_var("HERMES_TEST_EUID", "0");

        let run_as_user = env::var("USER")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("LOGNAME")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(|| String::from("root"));
        install_systemd_service(&ctx, false, true, Some(&run_as_user)).unwrap();

        let unit_path = systemd_unit_path(&ctx, true);
        let unit = fs::read_to_string(&unit_path).unwrap();
        let current_exe = env::current_exe().unwrap();
        assert!(unit.contains("gateway run --replace"));
        assert!(unit.contains(&current_exe.display().to_string()));
        assert!(!unit.contains("hermes_cli.main"));
        assert!(unit.contains(&format!("User={run_as_user}")));
        assert!(unit.contains("WantedBy=multi-user.target"));
        assert!(unit.contains("Environment=\"HOME="));
        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("daemon-reload"));
        assert!(log.contains("enable hermes-gateway"));

        remove_env_var("HERMES_TEST_EUID");
        remove_env_var("HERMES_FAKE_SYSTEMD_DIR");
        set_env_var("PATH", original_path);
    }

    #[test]
    fn generate_launchd_plist_contains_gateway_command_and_home() {
        let (_temp, ctx) = test_context();
        let plist = generate_launchd_plist(&ctx).unwrap();
        let current_exe = env::current_exe().unwrap();
        assert!(plist.contains("<key>ProgramArguments</key>"));
        assert!(plist.contains("gateway"));
        assert!(plist.contains("run"));
        assert!(plist.contains("--replace"));
        assert!(plist.contains(&xml_escape(&current_exe.display().to_string())));
        assert!(!plist.contains("hermes_cli.main"));
        assert!(plist.contains("HERMES_HOME"));
    }

    #[test]
    fn resolve_gateway_virtual_env_prefers_explicit_env() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let venv = temp.path().join("custom-venv");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&venv).unwrap();
        let original_home = env::var_os("HOME");
        set_env_var("HOME", &home);
        set_env_var("VIRTUAL_ENV", &venv);

        let resolved = resolve_gateway_virtual_env(&root).unwrap();
        assert_eq!(resolved, venv);

        remove_env_var("VIRTUAL_ENV");
        if let Some(home) = original_home {
            set_env_var("HOME", home);
        } else {
            remove_env_var("HOME");
        }
    }

    #[test]
    fn resolve_gateway_virtual_env_ignores_system_python_layout() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let home = temp.path().join("home");
        let python = temp.path().join("bin").join("python3");
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(&python, b"").unwrap();
        let original_home = env::var_os("HOME");
        set_env_var("HOME", &home);
        set_env_var("HERMES_GATEWAY_PYTHON", &python);
        remove_env_var("VIRTUAL_ENV");

        let resolved = resolve_gateway_virtual_env(&root);
        assert!(resolved.is_none());

        remove_env_var("HERMES_GATEWAY_PYTHON");
        if let Some(home) = original_home {
            set_env_var("HOME", home);
        } else {
            remove_env_var("HOME");
        }
    }

    #[test]
    fn find_legacy_gateway_units_filters_by_execstart_markers() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let user_dir = ctx.home_dir().join(".config").join("systemd").join("user");
        let fake_system = ctx.home_dir().join("etc-systemd");
        fs::create_dir_all(&user_dir).unwrap();
        fs::create_dir_all(&fake_system).unwrap();
        set_env_var("HERMES_FAKE_SYSTEMD_DIR", &fake_system);

        fs::write(
            user_dir.join("hermes.service"),
            "[Service]\nExecStart=/usr/bin/python -m hermes_cli.main gateway run\n",
        )
        .unwrap();
        fs::write(
            fake_system.join("hermes.service"),
            "[Service]\nExecStart=/usr/bin/other-daemon\n",
        )
        .unwrap();

        let units = find_legacy_gateway_units(&ctx);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].path, user_dir.join("hermes.service"));
        assert!(!units[0].is_system);
        remove_env_var("HERMES_FAKE_SYSTEMD_DIR");
    }

    #[test]
    fn migrate_legacy_removes_user_unit_and_reload() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let fake_bin = ctx.home_dir().join("bin");
        let log_path = ctx.home_dir().join("systemctl-legacy.log");
        let user_dir = ctx.home_dir().join(".config").join("systemd").join("user");
        fs::create_dir_all(&fake_bin).unwrap();
        fs::create_dir_all(&user_dir).unwrap();
        let script = fake_bin.join("systemctl");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\ncase \"$*\" in\n  *is-system-running*) printf 'running\\n' ;;\nesac\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let original_path = env::var("PATH").unwrap_or_default();
        set_env_var("PATH", format!("{}:{}", fake_bin.display(), original_path));
        let unit_path = user_dir.join("hermes.service");
        fs::write(
            &unit_path,
            "[Service]\nExecStart=/usr/bin/python -m hermes_cli.main gateway run\n",
        )
        .unwrap();

        print_gateway(
            &ctx,
            GatewayArgs {
                accept_hooks: false,
                command: Some(GatewayCommand::MigrateLegacy(GatewayMigrateLegacyArgs {
                    dry_run: false,
                    yes: true,
                })),
            },
        )
        .unwrap();

        assert!(!unit_path.exists());
        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("--user stop hermes.service"));
        assert!(log.contains("--user disable hermes.service"));
        assert!(log.contains("--user daemon-reload"));
        set_env_var("PATH", original_path);
    }

    #[test]
    fn select_systemd_scope_prefers_system_when_only_system_unit_exists() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let service = gateway_service_name(&ctx);
        let user_unit = systemd_unit_path(&ctx, false);
        let fake_systemd = ctx.home_dir().join("etc-systemd");
        fs::create_dir_all(fake_systemd.clone()).unwrap();
        set_env_var("HERMES_FAKE_SYSTEMD_DIR", &fake_systemd);
        fs::create_dir_all(user_unit.parent().unwrap()).unwrap();
        fs::write(fake_systemd.join(format!("{service}.service")), "x").unwrap();
        assert!(select_systemd_scope(&ctx, false));
        remove_env_var("HERMES_FAKE_SYSTEMD_DIR");
    }
}
