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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use clap::{Args, Subcommand};
use getrandom::fill as fill_random;
use hermes_core::{HermesContext, is_container, is_wsl};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use sha2::{Digest, Sha256};

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value};
use crate::native_api_server::maybe_run_native_api_server;
use crate::native_gateway_runtime::maybe_run_native_gateway_bundle;
use crate::native_webhook_server::maybe_run_native_webhook_server;
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
#[cfg(all(test, not(windows)))]
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

#[derive(Debug, Clone, Copy)]
struct GatewaySetupPlatformSpec {
    key: &'static str,
    label: &'static str,
    emoji: &'static str,
    token_var: &'static str,
    has_builtin_setup: bool,
    setup_instructions: &'static [&'static str],
    vars: &'static [GatewaySetupVarSpec],
}

#[derive(Debug, Clone, Copy)]
struct GatewaySetupVarSpec {
    name: &'static str,
    prompt: &'static str,
    password: bool,
    help: &'static str,
    is_allowlist: bool,
}

pub fn print_gateway(context: &HermesContext, args: GatewayArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        None => print_gateway_run(
            context,
            args.accept_hooks,
            GatewayRunArgs {
                verbose: 0,
                quiet: false,
                replace: false,
            },
        ),
        Some(GatewayCommand::Run(run)) => print_gateway_run(context, args.accept_hooks, run),
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
    let profile_homes = collect_profile_homes(context)?;

    for (name, home) in &profile_homes {
        let profile_context = context.clone().with_hermes_home_env(Some(home.clone()));
        for system in [false, true] {
            if !systemd_unit_path(&profile_context, system).exists()
                || !systemd_service_active(&profile_context, system)
            {
                continue;
            }
            let service = gateway_service_name(&profile_context);
            match restart_systemd_service(&profile_context, system) {
                Ok(()) => {
                    summary.restarted_services.push(if name == "default" {
                        service
                    } else {
                        format!("{service} ({name})")
                    });
                }
                Err(error) => {
                    eprintln!("  ⚠ Failed to restart {service}: {error}");
                }
            }
        }
    }

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

    let mut mapped_manual = Vec::new();
    for (name, home) in &profile_homes {
        let profile_context = context.clone().with_hermes_home_env(Some(home.clone()));
        if has_any_systemd_unit(&profile_context)
            || (is_macos() && launchd_plist_path(&profile_context).exists())
        {
            continue;
        }
        let Some(pid) = gateway_pids_for_profile(home).into_iter().next() else {
            continue;
        };
        mapped_manual.push((
            name.clone(),
            GatewayProcess {
                pid,
                command: String::new(),
            },
        ));
    }

    if mapped_manual.is_empty() {
        return Ok(summary);
    }

    let manual_processes = mapped_manual
        .iter()
        .map(|(_, process)| process.clone())
        .collect::<Vec<_>>();
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

    for (profile, _) in &mapped_manual {
        if launch_detached_profile_gateway_after_update(context, profile).is_ok() {
            summary.restarted_profiles.push(profile.clone());
        }
    }

    Ok(summary)
}

fn print_gateway_start(
    context: &HermesContext,
    _accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        let processes = known_profile_gateway_processes(context)?;
        let killed = kill_gateway_processes(&processes, false);
        if killed > 0 {
            println!("Killed {killed} stale gateway process(es) across all profiles");
            let pids = processes
                .iter()
                .map(|process| process.pid)
                .collect::<Vec<_>>();
            let _ = wait_for_processes_exit(
                &pids,
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
        let processes = known_profile_gateway_processes(context)?;
        let killed = kill_gateway_processes(&processes, false);
        let total = killed + usize::from(service_stopped);
        if total > 0 {
            println!("Stopped {total} gateway process(es) across all profiles");
            let pids = processes
                .iter()
                .map(|process| process.pid)
                .collect::<Vec<_>>();
            let _ = wait_for_processes_exit(
                &pids,
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
        let processes = known_profile_gateway_processes(context)?;
        let killed = kill_gateway_processes(&processes, false);
        let total = killed + usize::from(service_stopped);
        if total > 0 {
            println!("Stopped {total} gateway process(es) across all profiles");
        }
        let pids = processes
            .iter()
            .map(|process| process.pid)
            .collect::<Vec<_>>();
        let _ =
            wait_for_processes_exit(&pids, Duration::from_secs(10), Some(Duration::from_secs(5)));
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
            context,
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
        context,
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

fn print_gateway_run(
    context: &HermesContext,
    accept_hooks: bool,
    args: GatewayRunArgs,
) -> Result<(), Box<dyn Error>> {
    if maybe_run_native_gateway_bundle(context, &args)? {
        return Ok(());
    }
    if maybe_run_native_webhook_server(context, &args)? {
        return Ok(());
    }
    if maybe_run_native_api_server(context, &args)? {
        return Ok(());
    }
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

pub(crate) const GATEWAY_RUN_BOOTSTRAP: &str = concat!(
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
        let platforms = load_gateway_setup_metadata(context, accept_hooks)?;
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
        if configure_native_gateway_builtin_platform_with_io(context, platform, input, output)? {
            continue;
        } else if configure_native_gateway_plugin_platform_with_io(
            context, platform, input, output,
        )? {
            continue;
        } else if gateway_platform_uses_native_standard_setup(platform) {
            configure_standard_gateway_platform_with_io(context, platform, input, output)?;
        } else {
            run_gateway_platform_setup_bridge(accept_hooks, &platform.key)?;
        }
    }

    let platforms = load_gateway_setup_metadata(context, accept_hooks)?;
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

const NO_GATEWAY_SETUP_VARS: &[GatewaySetupVarSpec] = &[];

const TELEGRAM_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Open Telegram and message @BotFather",
    "2. Send /newbot and follow the prompts to create your bot",
    "3. Copy the bot token BotFather gives you",
    "4. To find your user ID: message @userinfobot — it replies with your numeric ID",
];

const TELEGRAM_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "TELEGRAM_BOT_TOKEN",
        prompt: "Bot token",
        password: true,
        help: "Paste the token from @BotFather (step 3 above).",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "TELEGRAM_ALLOWED_USERS",
        prompt: "Allowed user IDs (comma-separated)",
        password: false,
        help: "Paste your user ID from step 4 above.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "TELEGRAM_HOME_CHANNEL",
        prompt: "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
        password: false,
        help: "For DMs, this is your user ID. You can set it later by typing /set-home in chat.",
        is_allowlist: false,
    },
];

const DISCORD_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Go to https://discord.com/developers/applications → New Application",
    "2. Go to Bot → Reset Token → copy the bot token",
    "3. Enable: Bot → Privileged Gateway Intents → Message Content Intent",
    "4. Invite the bot to your server:",
    "   OAuth2 → URL Generator → check BOTH scopes:",
    "     - bot",
    "     - applications.commands  (required for slash commands!)",
    "   Bot Permissions: Send Messages, Read Message History, Attach Files",
    "   Copy the URL and open it in your browser to invite.",
    "5. Get your user ID: enable Developer Mode in Discord settings,",
    "   then right-click your name → Copy ID",
];

const DISCORD_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "DISCORD_BOT_TOKEN",
        prompt: "Bot token",
        password: true,
        help: "Paste the token from step 2 above.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "DISCORD_ALLOWED_USERS",
        prompt: "Allowed user IDs or usernames (comma-separated)",
        password: false,
        help: "Paste your user ID from step 5 above.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "DISCORD_HOME_CHANNEL",
        prompt: "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
        password: false,
        help: "Right-click a channel → Copy Channel ID (requires Developer Mode).",
        is_allowlist: false,
    },
];

const SLACK_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Go to https://api.slack.com/apps → Create New App",
    "2. Pick 'From an app manifest' — Hermes will write one below",
    "3. Enable Socket Mode and create an App-Level Token with connections:write",
    "4. Install the app to your workspace and copy the xoxb-... bot token",
    "5. Invite the bot to channels with /invite @YourBot",
    "6. Find your user ID: click your profile → three dots → Copy member ID",
];

const SLACK_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "SLACK_BOT_TOKEN",
        prompt: "Bot Token (xoxb-...)",
        password: true,
        help: "Paste the bot token from Slack after installing the app.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "SLACK_APP_TOKEN",
        prompt: "App Token (xapp-...)",
        password: true,
        help: "Paste the app-level token from Socket Mode.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "SLACK_ALLOWED_USERS",
        prompt: "Allowed user IDs (comma-separated)",
        password: false,
        help: "Paste your member ID from step 6 above.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "SLACK_HOME_CHANNEL",
        prompt: "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
        password: false,
        help: "Open the channel in Slack, copy its link, and use the C... channel ID.",
        is_allowlist: false,
    },
];

const MATRIX_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Works with any Matrix homeserver (self-hosted Synapse/Conduit/Dendrite or matrix.org)",
    "2. Create a bot user on your homeserver, or use your own account",
    "3. Get an access token from Element, or provide user ID + password",
    "4. For E2EE: set MATRIX_ENCRYPTION=true or enable it below",
    "5. Matrix user IDs look like @username:server and room IDs look like !abc123:server",
];

const SIGNAL_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Install signal-cli or run the bbernhard/signal-cli-rest-api container",
    "2. Link your Signal account and start the HTTP daemon",
    "3. Default local daemon URL: http://127.0.0.1:8080",
    "4. Signal account numbers should use E.164 format, e.g. +15551234567",
];

const DINGTALK_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Go to https://open-dev.dingtalk.com and create an application, or use device authorization",
    "2. Under Credentials, copy the AppKey (Client ID) and AppSecret (Client Secret)",
    "3. Enable Stream Mode under the bot settings",
    "4. Add the bot to a group chat or message it directly",
];

const FEISHU_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Create an app at https://open.feishu.cn/ or https://open.larksuite.com/",
    "2. Enable the Bot capability and copy the App ID and App Secret, or use QR setup below",
    "3. WebSocket mode is recommended because it does not require a public URL",
    "4. Restrict access with DM pairing or FEISHU_ALLOWED_USERS for production use",
];

const WECOM_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Create a smart robot in WeCom Application → Workspace → Smart Robot",
    "2. Select API Mode and copy the Bot ID and Secret, or use QR setup below",
    "3. The bot connects by WebSocket — no public callback URL is required",
    "4. Restrict access with WECOM_ALLOWED_USERS or DM pairing for production use",
];

const WHATSAPP_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Choose a separate bot number or personal self-chat mode",
    "2. Hermes uses the bundled WhatsApp bridge to show a QR code",
    "3. Scan the QR code from WhatsApp → Linked Devices → Link a Device",
    "4. Configure WHATSAPP_ALLOWED_USERS to restrict who can talk to Hermes",
];

const WEIXIN_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Hermes opens Tencent iLink QR login in this terminal",
    "2. Use WeChat to scan and confirm the QR code",
    "3. Hermes stores the returned account_id/token in your profile .env",
    "4. This adapter supports native text, image, video, and document delivery",
];

const QQBOT_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Register a QQ Bot application at https://q.qq.com or use QR setup below",
    "2. Note your App ID and App Secret from the application page",
    "3. Enable the required intents for direct, group, and guild messages",
    "4. Restrict access with DM pairing or QQ_ALLOWED_USERS for production use",
];

const EMAIL_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Use a dedicated email account for your Hermes agent",
    "2. For Gmail: enable 2FA, then create an App Password at",
    "   https://myaccount.google.com/apppasswords",
    "3. For other providers: use your email password or app-specific password",
    "4. IMAP must be enabled on your email account",
];

const EMAIL_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "EMAIL_ADDRESS",
        prompt: "Email address",
        password: false,
        help: "The email address Hermes will use (e.g., hermes@gmail.com).",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "EMAIL_PASSWORD",
        prompt: "Email password (or app password)",
        password: true,
        help: "For Gmail, use an App Password (not your regular password).",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "EMAIL_IMAP_HOST",
        prompt: "IMAP host",
        password: false,
        help: "e.g., imap.gmail.com for Gmail, outlook.office365.com for Outlook.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "EMAIL_SMTP_HOST",
        prompt: "SMTP host",
        password: false,
        help: "e.g., smtp.gmail.com for Gmail, smtp.office365.com for Outlook.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "EMAIL_ALLOWED_USERS",
        prompt: "Allowed sender emails (comma-separated)",
        password: false,
        help: "Only emails from these addresses will be processed.",
        is_allowlist: true,
    },
];

const SMS_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Create a Twilio account at https://www.twilio.com/",
    "2. Get your Account SID and Auth Token from the Twilio Console dashboard",
    "3. Buy or configure a phone number capable of sending SMS",
    "4. Set up your webhook URL for inbound SMS:",
    "   Twilio Console → Phone Numbers → Active Numbers → your number",
    "   → Messaging → A MESSAGE COMES IN → Webhook → https://your-server:8080/webhooks/twilio",
];

const SMS_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "TWILIO_ACCOUNT_SID",
        prompt: "Twilio Account SID",
        password: false,
        help: "Found on the Twilio Console dashboard.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "TWILIO_AUTH_TOKEN",
        prompt: "Twilio Auth Token",
        password: true,
        help: "Found on the Twilio Console dashboard (click to reveal).",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "TWILIO_PHONE_NUMBER",
        prompt: "Twilio phone number (E.164 format, e.g. +15551234567)",
        password: false,
        help: "The Twilio phone number to send SMS from.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "SMS_ALLOWED_USERS",
        prompt: "Allowed phone numbers (comma-separated, E.164 format)",
        password: false,
        help: "Only messages from these phone numbers will be processed.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "SMS_HOME_CHANNEL",
        prompt: "Home channel phone number (for cron/notification delivery, or empty)",
        password: false,
        help: "Phone number to deliver cron job results and notifications to.",
        is_allowlist: false,
    },
];

const HOMEASSISTANT_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Create a Long-Lived Access Token in Home Assistant",
    "   Profile → Security → Long-Lived Access Tokens",
    "2. Enter the token below and confirm your Home Assistant URL",
    "3. Hermes subscribes to Home Assistant state_changed events over WebSocket",
    "4. Configure watch_domains / watch_entities in config.yaml to receive events",
];

const HOMEASSISTANT_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "HASS_TOKEN",
        prompt: "Home Assistant Long-Lived Access Token",
        password: true,
        help: "Required. Create this in Profile → Security → Long-Lived Access Tokens.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "HASS_URL",
        prompt: "Home Assistant URL (default: http://homeassistant.local:8123)",
        password: false,
        help: "Leave empty to use the default local URL.",
        is_allowlist: false,
    },
];

const MATTERMOST_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. In Mattermost: Integrations → Bot Accounts → Add Bot Account",
    "   (System Console → Integrations → Bot Accounts must be enabled)",
    "2. Give it a username (e.g. hermes) and copy the bot token",
    "3. Works with any self-hosted Mattermost instance — enter your server URL",
    "4. To find your user ID: click your avatar (top-left) → Profile",
    "   Your user ID is displayed there — click it to copy.",
    "   ⚠ This is NOT your username — it's a 26-character alphanumeric ID.",
    "5. To get a channel ID: click the channel name → View Info → copy the ID",
];

const MATTERMOST_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "MATTERMOST_URL",
        prompt: "Server URL (e.g. https://mm.example.com)",
        password: false,
        help: "Your Mattermost server URL. Works with any self-hosted instance.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "MATTERMOST_TOKEN",
        prompt: "Bot token",
        password: true,
        help: "Paste the bot token from step 2 above.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "MATTERMOST_ALLOWED_USERS",
        prompt: "Allowed user IDs (comma-separated)",
        password: false,
        help: "Your Mattermost user ID from step 4 above.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "MATTERMOST_HOME_CHANNEL",
        prompt: "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
        password: false,
        help: "Channel ID where Hermes delivers cron results and notifications.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "MATTERMOST_REPLY_MODE",
        prompt: "Reply mode — 'off' for flat messages, 'thread' for threaded replies (default: off)",
        password: false,
        help: "off = flat channel messages, thread = replies nest under your message.",
        is_allowlist: false,
    },
];

const BLUEBUBBLES_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Install BlueBubbles on a Mac that will act as your iMessage server:",
    "   https://bluebubbles.app/",
    "2. Complete the BlueBubbles setup wizard and sign in with your Apple ID",
    "3. In BlueBubbles Settings → API, note the Server URL and password",
    "4. The server URL is typically http://<your-mac-ip>:1234",
    "5. Hermes connects via the BlueBubbles REST API and receives incoming messages via a local webhook",
    "6. To authorize users, use DM pairing: hermes pairing generate bluebubbles",
];

const BLUEBUBBLES_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "BLUEBUBBLES_SERVER_URL",
        prompt: "BlueBubbles server URL (e.g. http://192.168.1.10:1234)",
        password: false,
        help: "The URL shown in BlueBubbles Settings → API.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "BLUEBUBBLES_PASSWORD",
        prompt: "BlueBubbles server password",
        password: true,
        help: "The password shown in BlueBubbles Settings → API.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "BLUEBUBBLES_ALLOWED_USERS",
        prompt: "Pre-authorized phone numbers or iMessage IDs (comma-separated, or leave empty for DM pairing)",
        password: false,
        help: "Optional — pre-authorize specific users. Leave empty to use DM pairing instead.",
        is_allowlist: true,
    },
    GatewaySetupVarSpec {
        name: "BLUEBUBBLES_HOME_CHANNEL",
        prompt: "Home channel (phone number or iMessage ID for cron/notifications, or empty)",
        password: false,
        help: "Phone number or Apple ID to deliver cron results and notifications to.",
        is_allowlist: false,
    },
];

const WEBHOOK_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Enable the generic webhook adapter to receive external POST events",
    "2. Hermes listens on 0.0.0.0:8644 by default and validates HMAC signatures",
    "3. Define routes in config.yaml or create them dynamically with:",
    "   hermes webhook subscribe <name>",
    "4. Per-route secrets are preferred; WEBHOOK_SECRET sets a global fallback secret",
];

const WEBHOOK_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "WEBHOOK_ENABLED",
        prompt: "Enable webhook adapter now? (true)",
        password: false,
        help: "Required. Enter true to start the webhook listener.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WEBHOOK_PORT",
        prompt: "Webhook listen port (default: 8644)",
        password: false,
        help: "Optional. Leave empty to keep the default port.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WEBHOOK_SECRET",
        prompt: "Global HMAC secret (optional)",
        password: true,
        help: "Optional. Route-specific secrets can also be stored in config.yaml or webhook subscriptions.",
        is_allowlist: false,
    },
];

const API_SERVER_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Enable the OpenAI-compatible API server for external chat clients and automation",
    "2. By default it binds to 127.0.0.1:8642 and serves /v1 plus /health endpoints",
    "3. Set API_SERVER_KEY when exposing it beyond localhost",
    "4. Optional CORS origins let browser clients talk to the server directly",
];

const API_SERVER_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "API_SERVER_ENABLED",
        prompt: "Enable API server now? (true)",
        password: false,
        help: "Required. Enter true to expose the HTTP API server.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "API_SERVER_HOST",
        prompt: "Bind host (default: 127.0.0.1)",
        password: false,
        help: "Leave empty to keep the safe localhost default.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "API_SERVER_PORT",
        prompt: "API server port (default: 8642)",
        password: false,
        help: "Optional. Leave empty to keep the default port.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "API_SERVER_KEY",
        prompt: "API key (optional, but recommended off localhost)",
        password: true,
        help: "Recommended whenever the host is reachable from other machines.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "API_SERVER_CORS_ORIGINS",
        prompt: "Allowed CORS origins (comma-separated, optional)",
        password: false,
        help: "Example: https://chat.example.com,https://admin.example.com",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "API_SERVER_MODEL_NAME",
        prompt: "Advertised model name (optional)",
        password: false,
        help: "Override the model name returned by /v1/models if desired.",
        is_allowlist: false,
    },
];

const WECOM_CALLBACK_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Go to WeCom Admin Console → Applications → Create Self-Built App",
    "2. Note the Corp ID (top of admin console) and create a Corp Secret",
    "3. Under Receive Messages, configure the callback URL to point to your server",
    "4. Copy the Token and EncodingAESKey from the callback configuration",
    "5. The adapter runs an HTTP server — ensure the port is reachable from WeCom",
    "6. Restrict access with WECOM_CALLBACK_ALLOWED_USERS for production use",
];

const WECOM_CALLBACK_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_CORP_ID",
        prompt: "Corp ID",
        password: false,
        help: "Your WeCom enterprise Corp ID.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_CORP_SECRET",
        prompt: "Corp Secret",
        password: true,
        help: "The secret for your self-built application.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_AGENT_ID",
        prompt: "Agent ID",
        password: false,
        help: "The Agent ID of your self-built application.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_TOKEN",
        prompt: "Callback Token",
        password: true,
        help: "The Token from your WeCom callback configuration.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_ENCODING_AES_KEY",
        prompt: "Encoding AES Key",
        password: true,
        help: "The EncodingAESKey from your WeCom callback configuration.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_PORT",
        prompt: "Callback server port (default: 8645)",
        password: false,
        help: "Port for the HTTP callback server.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "WECOM_CALLBACK_ALLOWED_USERS",
        prompt: "Allowed user IDs (comma-separated, or empty)",
        password: false,
        help: "Restrict which WeCom users can interact with the app.",
        is_allowlist: true,
    },
];

const YUANBAO_SETUP_INSTRUCTIONS: &[&str] = &[
    "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
    "2. In the app, go to PAI → My Bot and create a new bot",
    "3. After the bot is created, copy the App ID and App Secret",
    "4. Enter them below and Hermes will connect automatically over WebSocket",
];

const YUANBAO_SETUP_VARS: &[GatewaySetupVarSpec] = &[
    GatewaySetupVarSpec {
        name: "YUANBAO_APP_ID",
        prompt: "App ID",
        password: false,
        help: "The App ID from your Yuanbao IM Bot credentials.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "YUANBAO_APP_SECRET",
        prompt: "App Secret",
        password: true,
        help: "The App Secret (used for HMAC signing) from your Yuanbao IM Bot.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "YUANBAO_BOT_ID",
        prompt: "Bot ID (optional)",
        password: false,
        help: "Optional. If omitted, Hermes can resolve it during sign-token flows when supported.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "YUANBAO_HOME_CHANNEL",
        prompt: "Home channel (optional, e.g. group:<group_code> or direct:<account_id>)",
        password: false,
        help: "Used for cron results and notifications.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "YUANBAO_API_DOMAIN",
        prompt: "API domain (optional, default: https://bot.yuanbao.tencent.com)",
        password: false,
        help: "Optional override for the Yuanbao HTTPS API base URL.",
        is_allowlist: false,
    },
    GatewaySetupVarSpec {
        name: "YUANBAO_WS_URL",
        prompt: "WebSocket URL (optional, default: wss://bot-wss.yuanbao.tencent.com/wss/connection)",
        password: false,
        help: "Optional override for the Yuanbao WebSocket endpoint.",
        is_allowlist: false,
    },
];

const IRC_REQUIRED_ENV: &[&str] = &["IRC_SERVER", "IRC_CHANNEL", "IRC_NICKNAME"];
const TEAMS_REQUIRED_ENV: &[&str] = &["TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"];

const GATEWAY_BUILTIN_PLATFORM_SPECS: &[GatewaySetupPlatformSpec] = &[
    GatewaySetupPlatformSpec {
        key: "telegram",
        label: "Telegram",
        emoji: "📱",
        token_var: "TELEGRAM_BOT_TOKEN",
        has_builtin_setup: false,
        setup_instructions: TELEGRAM_SETUP_INSTRUCTIONS,
        vars: TELEGRAM_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "discord",
        label: "Discord",
        emoji: "💬",
        token_var: "DISCORD_BOT_TOKEN",
        has_builtin_setup: false,
        setup_instructions: DISCORD_SETUP_INSTRUCTIONS,
        vars: DISCORD_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "slack",
        label: "Slack",
        emoji: "💼",
        token_var: "SLACK_BOT_TOKEN",
        has_builtin_setup: false,
        setup_instructions: SLACK_SETUP_INSTRUCTIONS,
        vars: SLACK_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "matrix",
        label: "Matrix",
        emoji: "🔐",
        token_var: "MATRIX_ACCESS_TOKEN",
        has_builtin_setup: false,
        setup_instructions: MATRIX_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "mattermost",
        label: "Mattermost",
        emoji: "💬",
        token_var: "MATTERMOST_TOKEN",
        has_builtin_setup: false,
        setup_instructions: MATTERMOST_SETUP_INSTRUCTIONS,
        vars: MATTERMOST_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "whatsapp",
        label: "WhatsApp",
        emoji: "📲",
        token_var: "WHATSAPP_ENABLED",
        has_builtin_setup: false,
        setup_instructions: WHATSAPP_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "signal",
        label: "Signal",
        emoji: "📡",
        token_var: "SIGNAL_HTTP_URL",
        has_builtin_setup: false,
        setup_instructions: SIGNAL_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "email",
        label: "Email",
        emoji: "📧",
        token_var: "EMAIL_ADDRESS",
        has_builtin_setup: false,
        setup_instructions: EMAIL_SETUP_INSTRUCTIONS,
        vars: EMAIL_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "sms",
        label: "SMS (Twilio)",
        emoji: "📱",
        token_var: "TWILIO_ACCOUNT_SID",
        has_builtin_setup: false,
        setup_instructions: SMS_SETUP_INSTRUCTIONS,
        vars: SMS_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "homeassistant",
        label: "Home Assistant",
        emoji: "🏠",
        token_var: "HASS_TOKEN",
        has_builtin_setup: false,
        setup_instructions: HOMEASSISTANT_SETUP_INSTRUCTIONS,
        vars: HOMEASSISTANT_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "dingtalk",
        label: "DingTalk",
        emoji: "💬",
        token_var: "DINGTALK_CLIENT_ID",
        has_builtin_setup: false,
        setup_instructions: DINGTALK_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "feishu",
        label: "Feishu / Lark",
        emoji: "🪽",
        token_var: "FEISHU_APP_ID",
        has_builtin_setup: false,
        setup_instructions: FEISHU_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "wecom",
        label: "WeCom (Enterprise WeChat)",
        emoji: "💬",
        token_var: "WECOM_BOT_ID",
        has_builtin_setup: false,
        setup_instructions: WECOM_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "wecom_callback",
        label: "WeCom Callback (Self-Built App)",
        emoji: "💬",
        token_var: "WECOM_CALLBACK_CORP_ID",
        has_builtin_setup: false,
        setup_instructions: WECOM_CALLBACK_SETUP_INSTRUCTIONS,
        vars: WECOM_CALLBACK_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "weixin",
        label: "Weixin / WeChat",
        emoji: "💬",
        token_var: "WEIXIN_ACCOUNT_ID",
        has_builtin_setup: false,
        setup_instructions: WEIXIN_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "bluebubbles",
        label: "BlueBubbles (iMessage)",
        emoji: "💬",
        token_var: "BLUEBUBBLES_SERVER_URL",
        has_builtin_setup: false,
        setup_instructions: BLUEBUBBLES_SETUP_INSTRUCTIONS,
        vars: BLUEBUBBLES_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "webhook",
        label: "Webhook",
        emoji: "🔗",
        token_var: "WEBHOOK_ENABLED",
        has_builtin_setup: false,
        setup_instructions: WEBHOOK_SETUP_INSTRUCTIONS,
        vars: WEBHOOK_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "qqbot",
        label: "QQ Bot",
        emoji: "🐧",
        token_var: "QQ_APP_ID",
        has_builtin_setup: false,
        setup_instructions: QQBOT_SETUP_INSTRUCTIONS,
        vars: NO_GATEWAY_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "yuanbao",
        label: "Yuanbao",
        emoji: "💎",
        token_var: "YUANBAO_APP_ID",
        has_builtin_setup: false,
        setup_instructions: YUANBAO_SETUP_INSTRUCTIONS,
        vars: YUANBAO_SETUP_VARS,
    },
    GatewaySetupPlatformSpec {
        key: "api_server",
        label: "API Server",
        emoji: "🌐",
        token_var: "API_SERVER_ENABLED",
        has_builtin_setup: false,
        setup_instructions: API_SERVER_SETUP_INSTRUCTIONS,
        vars: API_SERVER_SETUP_VARS,
    },
];

fn gateway_builtin_platform_specs() -> &'static [GatewaySetupPlatformSpec] {
    GATEWAY_BUILTIN_PLATFORM_SPECS
}

fn load_gateway_setup_metadata(
    context: &HermesContext,
    accept_hooks: bool,
) -> Result<Vec<GatewaySetupPlatform>, Box<dyn Error>> {
    let mut platforms = native_gateway_setup_metadata(context);
    for platform in load_gateway_plugin_setup_metadata(context, accept_hooks)? {
        if platforms
            .iter()
            .all(|existing| existing.key != platform.key)
        {
            platforms.push(platform);
        }
    }
    Ok(platforms)
}

fn native_gateway_setup_metadata(context: &HermesContext) -> Vec<GatewaySetupPlatform> {
    let mut platforms = gateway_builtin_platform_specs()
        .iter()
        .map(|spec| GatewaySetupPlatform {
            key: spec.key.to_string(),
            label: spec.label.to_string(),
            emoji: spec.emoji.to_string(),
            status: native_gateway_platform_status(context, spec),
            token_var: spec.token_var.to_string(),
            install_hint: None,
            setup_instructions: spec
                .setup_instructions
                .iter()
                .map(|line| (*line).to_string())
                .collect(),
            required_env: Vec::new(),
            has_builtin_setup: spec.has_builtin_setup,
            has_plugin_setup: false,
            vars: spec
                .vars
                .iter()
                .map(|var| GatewaySetupVar {
                    name: var.name.to_string(),
                    prompt: var.prompt.to_string(),
                    password: var.password,
                    help: var.help.to_string(),
                    is_allowlist: var.is_allowlist,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    platforms.extend(native_bundled_gateway_plugin_metadata(context));
    platforms
}

fn native_gateway_platform_status(
    context: &HermesContext,
    spec: &GatewaySetupPlatformSpec,
) -> String {
    let val = read_effective_env_value(context, spec.token_var);
    match spec.key {
        "whatsapp" => {
            if val
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("true"))
            {
                let session_file = context
                    .hermes_home()
                    .join("whatsapp")
                    .join("session")
                    .join("creds.json");
                if session_file.exists() {
                    return "configured + paired".to_string();
                }
                return "enabled, not paired".to_string();
            }
            "not configured".to_string()
        }
        "signal" => {
            let account = read_effective_env_value(context, "SIGNAL_ACCOUNT");
            if val.is_some() && account.is_some() {
                "configured".to_string()
            } else if val.is_some() || account.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "email" => {
            let password = read_effective_env_value(context, "EMAIL_PASSWORD");
            let imap = read_effective_env_value(context, "EMAIL_IMAP_HOST");
            let smtp = read_effective_env_value(context, "EMAIL_SMTP_HOST");
            if val.is_some() && password.is_some() && imap.is_some() && smtp.is_some() {
                "configured".to_string()
            } else if val.is_some() || password.is_some() || imap.is_some() || smtp.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "homeassistant" => {
            let url = read_effective_env_value(context, "HASS_URL");
            if val.is_some() {
                "configured".to_string()
            } else if url.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "matrix" => {
            let homeserver = read_effective_env_value(context, "MATRIX_HOMESERVER");
            let password = read_effective_env_value(context, "MATRIX_PASSWORD");
            if (val.is_some() || password.is_some()) && homeserver.is_some() {
                let e2ee = read_effective_env_value(context, "MATRIX_ENCRYPTION")
                    .map(|value| value.to_ascii_lowercase())
                    .is_some_and(|value| matches!(value.as_str(), "true" | "1" | "yes"));
                if e2ee {
                    "configured + E2EE".to_string()
                } else {
                    "configured".to_string()
                }
            } else if val.is_some() || password.is_some() || homeserver.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "feishu" => {
            let secret = read_effective_env_value(context, "FEISHU_APP_SECRET");
            if val.is_some() && secret.is_some() {
                "configured".to_string()
            } else if val.is_some() || secret.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "bluebubbles" => {
            let password = read_effective_env_value(context, "BLUEBUBBLES_PASSWORD");
            if val.is_some() && password.is_some() {
                "configured".to_string()
            } else if val.is_some() || password.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "wecom" => {
            let secret = read_effective_env_value(context, "WECOM_SECRET");
            if val.is_some() && secret.is_some() {
                "configured".to_string()
            } else if val.is_some() || secret.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "weixin" => {
            let token = read_effective_env_value(context, "WEIXIN_TOKEN");
            if val.is_some() && token.is_some() {
                "configured".to_string()
            } else if val.is_some() || token.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "webhook" => {
            let port = read_effective_env_value(context, "WEBHOOK_PORT");
            let secret = read_effective_env_value(context, "WEBHOOK_SECRET");
            if val
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("true"))
            {
                "configured".to_string()
            } else if port.is_some() || secret.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "qqbot" => {
            let secret = read_effective_env_value(context, "QQ_CLIENT_SECRET");
            if val.is_some() && secret.is_some() {
                "configured".to_string()
            } else if val.is_some() || secret.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "yuanbao" => {
            let secret = read_effective_env_value(context, "YUANBAO_APP_SECRET");
            if val.is_some() && secret.is_some() {
                "configured".to_string()
            } else if val.is_some() || secret.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "api_server" => {
            let key = read_effective_env_value(context, "API_SERVER_KEY");
            let host = read_effective_env_value(context, "API_SERVER_HOST");
            let port = read_effective_env_value(context, "API_SERVER_PORT");
            if val
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("true"))
                || key.is_some()
            {
                "configured".to_string()
            } else if host.is_some() || port.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        _ => {
            if val.is_some() {
                "configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
    }
}

fn native_bundled_gateway_plugin_metadata(context: &HermesContext) -> Vec<GatewaySetupPlatform> {
    vec![
        GatewaySetupPlatform {
            key: String::from("irc"),
            label: String::from("IRC"),
            emoji: String::from("💬"),
            status: native_gateway_plugin_platform_status(
                context,
                "irc",
                &[("IRC_SERVER", "server"), ("IRC_CHANNEL", "channel")],
            ),
            token_var: String::from("IRC_SERVER"),
            install_hint: Some(String::from("No extra packages needed (stdlib only)")),
            setup_instructions: Vec::new(),
            required_env: IRC_REQUIRED_ENV
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            has_builtin_setup: false,
            has_plugin_setup: true,
            vars: Vec::new(),
        },
        GatewaySetupPlatform {
            key: String::from("teams"),
            label: String::from("Microsoft Teams"),
            emoji: String::from("💼"),
            status: native_gateway_plugin_platform_status(
                context,
                "teams",
                &[
                    ("TEAMS_CLIENT_ID", "client_id"),
                    ("TEAMS_CLIENT_SECRET", "client_secret"),
                    ("TEAMS_TENANT_ID", "tenant_id"),
                ],
            ),
            token_var: String::from("TEAMS_CLIENT_ID"),
            install_hint: Some(String::from("pip install microsoft-teams-apps aiohttp")),
            setup_instructions: Vec::new(),
            required_env: TEAMS_REQUIRED_ENV
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            has_builtin_setup: false,
            has_plugin_setup: true,
            vars: Vec::new(),
        },
    ]
}

fn native_gateway_plugin_platform_status(
    context: &HermesContext,
    platform: &str,
    required: &[(&str, &str)],
) -> String {
    let configured = required
        .iter()
        .filter(|(env_key, extra_key)| {
            read_effective_env_value(context, env_key)
                .or_else(|| read_gateway_platform_extra_value(context, platform, extra_key))
                .is_some()
        })
        .count();
    if configured == required.len() {
        "configured".to_string()
    } else if configured > 0 {
        "partially configured".to_string()
    } else {
        "not configured".to_string()
    }
}

fn load_gateway_plugin_setup_metadata(
    context: &HermesContext,
    accept_hooks: bool,
) -> Result<Vec<GatewaySetupPlatform>, Box<dyn Error>> {
    if !gateway_has_enabled_user_plugins(context)? {
        return Ok(Vec::new());
    }
    let root = project_root();
    let Some(python) = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON")) else {
        return Ok(Vec::new());
    };

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string());
    if accept_hooks {
        command.env("HERMES_ACCEPT_HOOKS", "1");
    }
    command
        .arg("-c")
        .arg(GATEWAY_PLUGIN_SETUP_METADATA_BOOTSTRAP);

    let output = command.output()?;
    if !output.status.success() {
        return Err(exit_status_message("gateway plugin metadata", output.status).into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str::<Vec<GatewaySetupPlatform>>(trimmed)
        .map_err(|error| format!("invalid gateway plugin setup metadata: {error}").into())
}

fn gateway_has_enabled_user_plugins(context: &HermesContext) -> Result<bool, Box<dyn Error>> {
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let Some(plugins) = root
        .get(YamlValue::String(String::from("plugins")))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(false);
    };
    let Some(enabled) = plugins
        .get(YamlValue::String(String::from("enabled")))
        .and_then(YamlValue::as_sequence)
    else {
        return Ok(false);
    };
    Ok(enabled
        .iter()
        .filter_map(YamlValue::as_str)
        .map(str::trim)
        .any(|value| !value.is_empty()))
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
            if platform.key == "slack"
                && prompt_gateway_yes_no(
                    input,
                    output,
                    "Regenerate the Slack app manifest with the latest command list?",
                    true,
                )?
            {
                write_slack_manifest_for_gateway_setup(context, output)?;
            }
            return Ok(());
        }
    }

    if platform.key == "slack" {
        write_slack_manifest_for_gateway_setup(context, output)?;
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
            allowlist_value = configure_gateway_allowlist(
                context,
                input,
                output,
                &var.name,
                format!("  {}", var.prompt).as_str(),
            )?;
            continue;
        }

        loop {
            let value = prompt_gateway_line(input, output, format!("  {}", var.prompt).as_str())?;
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                if let Err(message) = validate_gateway_setup_value(platform, var, trimmed) {
                    writeln!(output, "  {message}")?;
                    continue;
                }
                let normalized = normalize_gateway_setup_value(platform, var, trimmed);
                save_env_value(context.env_path(), &var.name, &normalized)?;
                writeln!(output, "  Saved {}", var.name)?;
            } else if gateway_setup_var_is_required(platform, var) {
                writeln!(
                    output,
                    "  Skipped — {} won't work without this.",
                    platform.label
                )?;
                return Ok(());
            } else {
                writeln!(output, "  Skipped (can configure later)")?;
            }
            break;
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

    if platform.key == "bluebubbles" {
        configure_bluebubbles_advanced_settings(context, input, output)?;
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    Ok(())
}

fn gateway_setup_var_is_required(platform: &GatewaySetupPlatform, var: &GatewaySetupVar) -> bool {
    var.name == platform.token_var
        || (platform.key == "bluebubbles" && var.name == "BLUEBUBBLES_PASSWORD")
        || (platform.key == "yuanbao" && var.name == "YUANBAO_APP_SECRET")
}

fn configure_bluebubbles_advanced_settings(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  Advanced settings (defaults are fine for most setups):"
    )?;
    if prompt_gateway_yes_no(input, output, "Configure webhook listener settings?", false)? {
        let value = prompt_gateway_line(input, output, "  Webhook listener port (default: 8645)")?;
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<u16>() {
                Ok(port) if port > 0 => {
                    save_env_value(
                        context.env_path(),
                        "BLUEBUBBLES_WEBHOOK_PORT",
                        &port.to_string(),
                    )?;
                    writeln!(output, "  Webhook port set to {port}")?;
                }
                Ok(_) | Err(_) => {
                    writeln!(output, "  Invalid port number, using default 8645")?;
                }
            }
        }
    }
    writeln!(
        output,
        "  Requires the BlueBubbles Private API helper for typing indicators, read receipts, and tapback reactions."
    )?;
    writeln!(
        output,
        "  Install: https://docs.bluebubbles.app/helper-bundle/installation"
    )?;
    Ok(())
}

fn configure_gateway_allowlist(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    var_name: &str,
    prompt: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    writeln!(output, "  The gateway denies all users by default.")?;
    writeln!(
        output,
        "  Enter user IDs to create an allowlist, or leave empty to choose another access mode."
    )?;
    let value = prompt_gateway_line(input, output, prompt)?;
    let trimmed = value.trim();
    if !trimmed.is_empty() {
        let cleaned = normalize_gateway_allowlist(var_name, trimmed);
        save_env_value(context.env_path(), var_name, &cleaned)?;
        remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
        writeln!(
            output,
            "  Saved — only these users can interact with the bot."
        )?;
        return Ok(Some(cleaned));
    }

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
    Ok(None)
}

fn write_slack_manifest_for_gateway_setup(
    context: &HermesContext,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    match crate::slack_cmd::write_default_slack_manifest(context) {
        Ok(target) => {
            writeln!(output)?;
            writeln!(
                output,
                "  Slack app manifest written to: {}",
                target.display()
            )?;
            writeln!(
                output,
                "  Paste it into https://api.slack.com/apps → your app → App Manifest, then save and reinstall."
            )?;
        }
        Err(error) => {
            writeln!(output)?;
            writeln!(output, "  Couldn't write Slack manifest: {error}")?;
            writeln!(
                output,
                "  You can generate it manually later with: hermes slack manifest --write"
            )?;
        }
    }
    Ok(())
}

fn validate_gateway_setup_value(
    platform: &GatewaySetupPlatform,
    var: &GatewaySetupVar,
    value: &str,
) -> Result<(), &'static str> {
    if platform.key == "telegram"
        && var.name == "TELEGRAM_BOT_TOKEN"
        && !is_valid_telegram_bot_token(value)
    {
        return Err(
            "Invalid token format. Expected: <numeric_id>:<alphanumeric_hash> (for example, 123456789:ABCdefGHI-jklMNOpqrSTUvwxYZ).",
        );
    }
    if matches!(var.name.as_str(), "WEBHOOK_ENABLED" | "API_SERVER_ENABLED")
        && !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        )
    {
        return Err("Enter true, yes, or 1 to enable this platform.");
    }
    if matches!(
        var.name.as_str(),
        "WEBHOOK_PORT" | "API_SERVER_PORT" | "WECOM_CALLBACK_PORT"
    ) {
        let Ok(port) = value.trim().parse::<u16>() else {
            return Err("Invalid port number. Enter an integer between 1 and 65535.");
        };
        if port == 0 {
            return Err("Invalid port number. Enter an integer between 1 and 65535.");
        }
    }
    Ok(())
}

fn normalize_gateway_setup_value(
    platform: &GatewaySetupPlatform,
    var: &GatewaySetupVar,
    value: &str,
) -> String {
    if matches!(
        platform.key.as_str(),
        "bluebubbles" | "mattermost" | "matrix"
    ) && matches!(
        var.name.as_str(),
        "BLUEBUBBLES_SERVER_URL" | "MATTERMOST_URL" | "MATRIX_HOMESERVER"
    ) {
        return value.trim_end_matches('/').to_string();
    }
    if matches!(var.name.as_str(), "WEBHOOK_ENABLED" | "API_SERVER_ENABLED") {
        return String::from("true");
    }
    if matches!(
        var.name.as_str(),
        "WEBHOOK_PORT" | "API_SERVER_PORT" | "WECOM_CALLBACK_PORT"
    ) {
        return value
            .trim()
            .parse::<u16>()
            .map(|port| port.to_string())
            .unwrap_or_else(|_| value.trim().to_string());
    }
    if matches!(var.name.as_str(), "HASS_URL" | "YUANBAO_API_DOMAIN") {
        return value.trim_end_matches('/').to_string();
    }
    if var.name == "API_SERVER_CORS_ORIGINS" {
        return value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>()
            .join(",");
    }
    value.to_string()
}

fn is_valid_telegram_bot_token(value: &str) -> bool {
    let Some((bot_id, token)) = value.split_once(':') else {
        return false;
    };
    !bot_id.is_empty()
        && bot_id.bytes().all(|byte| byte.is_ascii_digit())
        && token.len() >= 30
        && token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn configure_native_gateway_builtin_platform_with_io(
    context: &HermesContext,
    platform: &GatewaySetupPlatform,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<bool, Box<dyn Error>> {
    match platform.key.as_str() {
        "whatsapp" => {
            crate::whatsapp_cmd::run_whatsapp_setup_with_io(context, input, output)?;
            Ok(true)
        }
        "matrix" => {
            configure_matrix_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "signal" => {
            configure_signal_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "dingtalk" => {
            configure_dingtalk_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "feishu" => {
            configure_feishu_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "wecom" => {
            configure_wecom_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "weixin" => {
            configure_weixin_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        "qqbot" => {
            configure_qqbot_gateway_platform_with_io(context, platform, input, output)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn configure_matrix_gateway_platform_with_io(
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

    let existing_auth = read_effective_env_value(context, "MATRIX_ACCESS_TOKEN")
        .or_else(|| read_effective_env_value(context, "MATRIX_PASSWORD"));
    if existing_auth.is_some() {
        writeln!(output)?;
        writeln!(output, "Matrix is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure Matrix?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    let homeserver = prompt_gateway_line(
        input,
        output,
        "Homeserver URL (e.g. https://matrix.example.org)",
    )?;
    if !homeserver.trim().is_empty() {
        let normalized = homeserver.trim().trim_end_matches('/').to_string();
        save_env_value(context.env_path(), "MATRIX_HOMESERVER", &normalized)?;
        writeln!(output, "  Saved MATRIX_HOMESERVER")?;
    }

    writeln!(output)?;
    writeln!(
        output,
        "  Auth: provide an access token (recommended), or user ID + password."
    )?;
    let token = prompt_gateway_line(
        input,
        output,
        "Access token (leave empty to use password login)",
    )?;
    if !token.trim().is_empty() {
        save_env_value(context.env_path(), "MATRIX_ACCESS_TOKEN", token.trim())?;
        let user_id = prompt_gateway_line(
            input,
            output,
            "User ID (@bot:server — optional, will be auto-detected)",
        )?;
        if !user_id.trim().is_empty() {
            save_env_value(context.env_path(), "MATRIX_USER_ID", user_id.trim())?;
        }
        writeln!(output, "  Matrix access token saved")?;
    } else {
        let user_id = prompt_gateway_line(input, output, "User ID (@bot:server)")?;
        if !user_id.trim().is_empty() {
            save_env_value(context.env_path(), "MATRIX_USER_ID", user_id.trim())?;
        }
        let password = prompt_gateway_line(input, output, "Password")?;
        if !password.trim().is_empty() {
            save_env_value(context.env_path(), "MATRIX_PASSWORD", password.trim())?;
            writeln!(output, "  Matrix credentials saved")?;
        }
    }

    let auth_configured =
        !token.trim().is_empty() || read_effective_env_value(context, "MATRIX_PASSWORD").is_some();
    if auth_configured {
        writeln!(output)?;
        let want_e2ee =
            prompt_gateway_yes_no(input, output, "Enable end-to-end encryption (E2EE)?", false)?;
        if want_e2ee {
            save_env_value(context.env_path(), "MATRIX_ENCRYPTION", "true")?;
            writeln!(output, "  E2EE enabled")?;
        }

        let matrix_pkg = if want_e2ee {
            "mautrix[encryption]"
        } else {
            "mautrix"
        };
        if !gateway_python_module_installed("mautrix")? {
            writeln!(output, "  Installing {matrix_pkg}...")?;
            if gateway_install_python_package(matrix_pkg)? {
                writeln!(output, "  {matrix_pkg} installed")?;
            } else {
                writeln!(
                    output,
                    "  Install failed — run manually: pip install '{matrix_pkg}'"
                )?;
            }
        }

        writeln!(output)?;
        writeln!(output, "  Matrix user IDs look like @username:server.")?;
        configure_gateway_allowlist(
            context,
            input,
            output,
            "MATRIX_ALLOWED_USERS",
            "Allowed user IDs (comma-separated, e.g. @you:server)",
        )?;

        writeln!(output)?;
        writeln!(
            output,
            "  Home Room: where Hermes delivers cron job results and notifications."
        )?;
        let home_room = prompt_gateway_line(
            input,
            output,
            "Home room ID (leave empty to set later with /set-home)",
        )?;
        if !home_room.trim().is_empty() {
            save_env_value(context.env_path(), "MATRIX_HOME_ROOM", home_room.trim())?;
        }
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    Ok(())
}

fn configure_signal_gateway_platform_with_io(
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

    let existing_url = read_effective_env_value(context, "SIGNAL_HTTP_URL");
    let existing_account = read_effective_env_value(context, "SIGNAL_ACCOUNT");
    if existing_url.is_some() && existing_account.is_some() {
        writeln!(output)?;
        writeln!(output, "Signal is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure Signal?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    if which_on_path("signal-cli").is_some() {
        writeln!(output, "  signal-cli found on PATH.")?;
    } else {
        writeln!(output, "  signal-cli not found on PATH.")?;
        writeln!(
            output,
            "  Signal requires signal-cli running as an HTTP daemon."
        )?;
        writeln!(
            output,
            "  Install: https://github.com/AsamK/signal-cli/releases"
        )?;
        writeln!(output, "  macOS: brew install signal-cli")?;
        writeln!(output, "  Docker: bbernhard/signal-cli-rest-api")?;
        writeln!(
            output,
            "  Start daemon: signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080"
        )?;
    }

    let default_url = existing_url
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("http://127.0.0.1:8080");
    let url = loop {
        writeln!(output)?;
        writeln!(
            output,
            "  Enter the URL where the signal-cli HTTP daemon is running."
        )?;
        let raw = prompt_gateway_line(
            input,
            output,
            format!("  HTTP URL [{default_url}]").as_str(),
        )?;
        let candidate = if raw.trim().is_empty() {
            default_url.to_string()
        } else {
            raw.trim().to_string()
        };
        match normalize_signal_http_url(&candidate) {
            Some(url) => break url,
            None => writeln!(output, "  Enter an http:// or https:// URL.")?,
        }
    };

    writeln!(output, "  Testing connection...")?;
    match probe_signal_http_daemon(&url) {
        Ok(200) => writeln!(output, "  signal-cli daemon is reachable.")?,
        Ok(status) => {
            writeln!(output, "  signal-cli responded with status {status}.")?;
            if !prompt_gateway_yes_no(input, output, "Continue anyway?", false)? {
                return Ok(());
            }
        }
        Err(error) => {
            writeln!(output, "  Could not reach signal-cli at {url}: {error}")?;
            if !prompt_gateway_yes_no(
                input,
                output,
                "Save this URL anyway? (you can start signal-cli later)",
                true,
            )? {
                return Ok(());
            }
        }
    }
    save_env_value(context.env_path(), "SIGNAL_HTTP_URL", &url)?;

    writeln!(output)?;
    writeln!(
        output,
        "  Enter your Signal account phone number in E.164 format."
    )?;
    let account = loop {
        let prompt = if let Some(existing) = existing_account
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            format!("  Account number [{existing}]")
        } else {
            String::from("  Account number")
        };
        let raw = prompt_gateway_line(input, output, &prompt)?;
        let candidate = if raw.trim().is_empty() {
            existing_account.clone().unwrap_or_default()
        } else {
            raw.trim().to_string()
        };
        if is_valid_signal_account(&candidate) {
            break candidate;
        }
        writeln!(
            output,
            "  Account number is required and should look like +15551234567."
        )?;
    };
    save_env_value(context.env_path(), "SIGNAL_ACCOUNT", &account)?;

    writeln!(output)?;
    writeln!(
        output,
        "  Enter phone numbers or UUIDs of allowed users (comma-separated)."
    )?;
    let existing_allowed = read_effective_env_value(context, "SIGNAL_ALLOWED_USERS");
    let default_allowed = existing_allowed
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&account);
    let allowed = prompt_gateway_line(
        input,
        output,
        format!("  Allowed users [{default_allowed}]").as_str(),
    )?;
    let allowed = if allowed.trim().is_empty() {
        default_allowed.to_string()
    } else {
        normalize_gateway_allowlist("SIGNAL_ALLOWED_USERS", allowed.trim())
    };
    save_env_value(context.env_path(), "SIGNAL_ALLOWED_USERS", &allowed)?;

    writeln!(output)?;
    if prompt_gateway_yes_no(
        input,
        output,
        "Enable group messaging? (disabled by default for security)",
        false,
    )? {
        writeln!(output)?;
        writeln!(output, "  Enter group IDs to allow, or * for all groups.")?;
        let existing_groups = read_effective_env_value(context, "SIGNAL_GROUP_ALLOWED_USERS");
        let default_groups = existing_groups
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("*");
        let groups = prompt_gateway_line(
            input,
            output,
            format!("  Group IDs [{default_groups}]").as_str(),
        )?;
        let groups = if groups.trim().is_empty() {
            default_groups.to_string()
        } else {
            groups.replace(' ', "")
        };
        save_env_value(context.env_path(), "SIGNAL_GROUP_ALLOWED_USERS", &groups)?;
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    writeln!(output, "  URL: {url}")?;
    writeln!(output, "  Account: {account}")?;
    writeln!(output, "  DM auth: SIGNAL_ALLOWED_USERS + DM pairing")?;
    let groups_enabled = read_effective_env_value(context, "SIGNAL_GROUP_ALLOWED_USERS").is_some();
    writeln!(
        output,
        "  Groups: {}",
        if groups_enabled {
            "enabled"
        } else {
            "disabled"
        }
    )?;
    Ok(())
}

fn normalize_signal_http_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(trimmed).ok()?;
    match parsed.scheme() {
        "http" | "https" => Some(trimmed.to_string()),
        _ => None,
    }
}

fn is_valid_signal_account(value: &str) -> bool {
    let trimmed = value.trim();
    let digits = trimmed.strip_prefix('+').unwrap_or_default();
    (8..=16).contains(&digits.len()) && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn probe_signal_http_daemon(url: &str) -> Result<u16, Box<dyn Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;
    let response = client
        .get(format!("{}/api/v1/check", url.trim_end_matches('/')))
        .send()?;
    Ok(response.status().as_u16())
}

fn configure_dingtalk_gateway_platform_with_io(
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

    if let Some(existing) = read_effective_env_value(context, "DINGTALK_CLIENT_ID") {
        writeln!(output)?;
        writeln!(
            output,
            "{} is already configured (Client ID: {existing}).",
            platform.label
        )?;
        if !prompt_gateway_yes_no(
            input,
            output,
            format!("Reconfigure {}?", platform.label).as_str(),
            false,
        )? {
            return Ok(());
        }
    }

    writeln!(output)?;
    let method = prompt_gateway_menu_choice(
        input,
        output,
        "Choose setup method",
        &[
            "Device authorization (scan/open DingTalk authorization link)",
            "Manual input (Client ID and Client Secret)",
        ],
    )?;

    let configured = if method == 0 {
        match dingtalk_device_authorize_with_io(output) {
            Ok((client_id, client_secret)) => {
                save_env_value(context.env_path(), "DINGTALK_CLIENT_ID", &client_id)?;
                save_env_value(context.env_path(), "DINGTALK_CLIENT_SECRET", &client_secret)?;
                save_env_value(context.env_path(), "DINGTALK_ALLOW_ALL_USERS", "true")?;
                writeln!(output, "  Device authorization successful.")?;
                true
            }
            Err(error) => {
                writeln!(output, "  Device authorization failed: {error}")?;
                writeln!(output, "  Continuing with manual input.")?;
                configure_dingtalk_manual_credentials(context, input, output)?
            }
        }
    } else {
        configure_dingtalk_manual_credentials(context, input, output)?
    };

    if configured {
        save_env_value(context.env_path(), "DINGTALK_ALLOW_ALL_USERS", "true")?;
        writeln!(output)?;
        writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    }
    Ok(())
}

fn configure_dingtalk_manual_credentials(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<bool, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  Enter the DingTalk application credentials from the developer console."
    )?;
    let Some(client_id) = prompt_gateway_required_line(
        input,
        output,
        "  AppKey (Client ID)",
        "Skipped — DingTalk won't work without a Client ID.",
    )?
    else {
        return Ok(false);
    };
    let Some(client_secret) = prompt_gateway_required_line(
        input,
        output,
        "  AppSecret (Client Secret)",
        "Skipped — DingTalk won't work without a Client Secret.",
    )?
    else {
        return Ok(false);
    };
    save_env_value(context.env_path(), "DINGTALK_CLIENT_ID", &client_id)?;
    save_env_value(context.env_path(), "DINGTALK_CLIENT_SECRET", &client_secret)?;
    Ok(true)
}

fn dingtalk_device_authorize_with_io(
    output: &mut dyn Write,
) -> Result<(String, String), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "  Initializing DingTalk device authorization...")?;
    writeln!(
        output,
        "  Note: the authorization page may be branded OpenClaw."
    )?;

    let base = env::var("DINGTALK_REGISTRATION_BASE_URL")
        .unwrap_or_else(|_| String::from("https://oapi.dingtalk.com"))
        .trim()
        .trim_end_matches('/')
        .to_string();
    if base.is_empty() {
        return Err("DingTalk registration base URL cannot be empty".into());
    }
    let source = env::var("DINGTALK_REGISTRATION_SOURCE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| String::from("openClaw"));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    let init = dingtalk_registration_api_post(
        &client,
        &base,
        "/app/registration/init",
        serde_json::json!({ "source": source }),
    )?;
    let nonce = dingtalk_json_string(&init, "nonce")
        .ok_or("DingTalk registration init response missing nonce")?;

    let begin = dingtalk_registration_api_post(
        &client,
        &base,
        "/app/registration/begin",
        serde_json::json!({ "nonce": nonce }),
    )?;
    let device_code = dingtalk_json_string(&begin, "device_code")
        .ok_or("DingTalk registration begin response missing device_code")?;
    let verification_uri = dingtalk_json_string(&begin, "verification_uri_complete")
        .ok_or("DingTalk registration begin response missing verification URI")?;
    let expires_in = dingtalk_json_u64(&begin, "expires_in").unwrap_or(7200);
    let interval = dingtalk_json_u64(&begin, "interval").unwrap_or(3).max(1);

    writeln!(output)?;
    writeln!(
        output,
        "  Open this link or scan it from DingTalk to authorize:"
    )?;
    writeln!(output, "  {verification_uri}")?;
    writeln!(output)?;
    writeln!(
        output,
        "  Waiting for authorization... (timeout: {expires_in}s)"
    )?;

    dingtalk_wait_for_registration_success(&client, &base, &device_code, interval, expires_in)
}

fn dingtalk_registration_api_post(
    client: &reqwest::blocking::Client,
    base: &str,
    path: &str,
    payload: JsonValue,
) -> Result<JsonValue, Box<dyn Error>> {
    let url = format!("{base}{path}");
    let response = client.post(&url).json(&payload).send()?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("DingTalk registration HTTP {status} at {path}").into());
    }
    let data = response.json::<JsonValue>()?;
    let errcode = data
        .get("errcode")
        .and_then(JsonValue::as_i64)
        .unwrap_or(-1);
    if errcode != 0 {
        let errmsg = data
            .get("errmsg")
            .and_then(JsonValue::as_str)
            .unwrap_or("unknown error");
        return Err(
            format!("DingTalk registration API error at {path}: {errmsg} ({errcode})").into(),
        );
    }
    Ok(data)
}

fn dingtalk_wait_for_registration_success(
    client: &reqwest::blocking::Client,
    base: &str,
    device_code: &str,
    interval_secs: u64,
    expires_in_secs: u64,
) -> Result<(String, String), Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(expires_in_secs.max(1));
    let interval = Duration::from_secs(interval_secs.max(1));
    while Instant::now() < deadline {
        sleep(interval);
        let poll = dingtalk_registration_api_post(
            client,
            base,
            "/app/registration/poll",
            serde_json::json!({ "device_code": device_code }),
        )?;
        let status = dingtalk_json_string(&poll, "status")
            .unwrap_or_else(|| String::from("UNKNOWN"))
            .to_ascii_uppercase();
        match status.as_str() {
            "WAITING" => continue,
            "SUCCESS" => {
                let client_id = dingtalk_json_string(&poll, "client_id")
                    .ok_or("DingTalk authorization succeeded without client_id")?;
                let client_secret = dingtalk_json_string(&poll, "client_secret")
                    .ok_or("DingTalk authorization succeeded without client_secret")?;
                return Ok((client_id, client_secret));
            }
            "FAIL" | "EXPIRED" | "UNKNOWN" => {
                let reason = dingtalk_json_string(&poll, "fail_reason").unwrap_or(status);
                return Err(format!("DingTalk authorization failed: {reason}").into());
            }
            _ => return Err(format!("Unexpected DingTalk authorization status: {status}").into()),
        }
    }
    Err("DingTalk authorization timed out, please retry".into())
}

fn dingtalk_json_string(value: &JsonValue, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn dingtalk_json_u64(value: &JsonValue, key: &str) -> Option<u64> {
    value.get(key).and_then(JsonValue::as_u64)
}

#[derive(Debug, Clone)]
struct FeishuSetupCredentials {
    app_id: String,
    app_secret: String,
    domain: String,
    open_id: Option<String>,
    bot_name: Option<String>,
}

fn configure_feishu_gateway_platform_with_io(
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

    let existing_app_id = read_effective_env_value(context, "FEISHU_APP_ID");
    let existing_secret = read_effective_env_value(context, "FEISHU_APP_SECRET");
    if existing_app_id.is_some() && existing_secret.is_some() {
        writeln!(output)?;
        writeln!(output, "Feishu / Lark is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure Feishu / Lark?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    let method = prompt_gateway_menu_choice(
        input,
        output,
        "How would you like to set up Feishu / Lark?",
        &[
            "Scan QR code to create a new bot automatically (recommended)",
            "Enter existing App ID and App Secret manually",
        ],
    )?;

    let mut used_qr = false;
    let credentials = if method == 0 {
        match feishu_qr_register_with_io(output) {
            Ok(Some(credentials)) => {
                used_qr = true;
                Some(credentials)
            }
            Ok(None) => {
                writeln!(
                    output,
                    "  QR setup did not complete. Continuing with manual input."
                )?;
                configure_feishu_manual_credentials(input, output)?
            }
            Err(error) => {
                writeln!(output, "  QR registration failed: {error}")?;
                writeln!(output, "  Continuing with manual input.")?;
                configure_feishu_manual_credentials(input, output)?
            }
        }
    } else {
        configure_feishu_manual_credentials(input, output)?
    };

    let Some(credentials) = credentials else {
        return Ok(());
    };

    save_env_value(context.env_path(), "FEISHU_APP_ID", &credentials.app_id)?;
    save_env_value(
        context.env_path(),
        "FEISHU_APP_SECRET",
        &credentials.app_secret,
    )?;
    save_env_value(context.env_path(), "FEISHU_DOMAIN", &credentials.domain)?;

    let connection_mode = if used_qr {
        String::from("websocket")
    } else {
        writeln!(output)?;
        let mode_idx = prompt_gateway_menu_choice(
            input,
            output,
            "Connection mode",
            &[
                "WebSocket (recommended — no public URL needed)",
                "Webhook (requires a reachable HTTP endpoint)",
            ],
        )?;
        if mode_idx == 1 {
            writeln!(output, "  Webhook defaults: 127.0.0.1:8765/feishu/webhook")?;
            writeln!(
                output,
                "  Override with FEISHU_WEBHOOK_HOST / FEISHU_WEBHOOK_PORT / FEISHU_WEBHOOK_PATH"
            )?;
            writeln!(
                output,
                "  For signature verification, set FEISHU_ENCRYPT_KEY and FEISHU_VERIFICATION_TOKEN"
            )?;
            String::from("webhook")
        } else {
            String::from("websocket")
        }
    };
    save_env_value(
        context.env_path(),
        "FEISHU_CONNECTION_MODE",
        &connection_mode,
    )?;

    if let Some(bot_name) = credentials.bot_name.as_deref() {
        writeln!(output)?;
        writeln!(output, "  Bot: {bot_name}")?;
    }

    configure_feishu_dm_policy(context, input, output, credentials.open_id.as_deref())?;
    configure_feishu_group_policy(context, input, output)?;

    writeln!(output)?;
    let home_channel = prompt_gateway_line(
        input,
        output,
        "  Home chat ID (optional, for cron/notifications)",
    )?;
    if !home_channel.trim().is_empty() {
        save_env_value(
            context.env_path(),
            "FEISHU_HOME_CHANNEL",
            home_channel.trim(),
        )?;
        writeln!(output, "  Home channel set to {}", home_channel.trim())?;
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    writeln!(output, "  App ID: {}", credentials.app_id)?;
    writeln!(output, "  Domain: {}", credentials.domain)?;
    Ok(())
}

fn configure_feishu_manual_credentials(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<Option<FeishuSetupCredentials>, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  Go to https://open.feishu.cn/ or https://open.larksuite.com/ for Lark."
    )?;
    writeln!(
        output,
        "  Create an app, enable the Bot capability, and copy the credentials."
    )?;

    let Some(app_id) = prompt_gateway_required_line(
        input,
        output,
        "  App ID",
        "Skipped — Feishu / Lark won't work without an App ID.",
    )?
    else {
        return Ok(None);
    };
    let Some(app_secret) = prompt_gateway_required_line(
        input,
        output,
        "  App Secret",
        "Skipped — Feishu / Lark won't work without an App Secret.",
    )?
    else {
        return Ok(None);
    };
    writeln!(output)?;
    let domain_idx = prompt_gateway_menu_choice(
        input,
        output,
        "Domain",
        &["feishu (China)", "lark (International)"],
    )?;
    let domain = if domain_idx == 1 { "lark" } else { "feishu" }.to_string();

    let bot_info = match feishu_probe_bot_raw(&app_id, &app_secret, &domain) {
        Ok(info) => {
            if let Some(name) = info.bot_name.as_deref() {
                writeln!(output, "  Credentials verified — bot: {name}")?;
            } else {
                writeln!(output, "  Credentials verified.")?;
            }
            Some(info)
        }
        Err(error) => {
            writeln!(output, "  Credential verification skipped: {error}")?;
            None
        }
    };

    Ok(Some(FeishuSetupCredentials {
        app_id,
        app_secret,
        domain,
        open_id: None,
        bot_name: bot_info.and_then(|info| info.bot_name),
    }))
}

fn configure_feishu_dm_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    default_open_id: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    let access_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should direct messages be authorized?",
        &[
            "Use DM pairing approval (recommended)",
            "Allow all direct messages",
            "Only allow listed user IDs",
        ],
    )?;
    match access_idx {
        0 => {
            save_env_value(context.env_path(), "FEISHU_ALLOW_ALL_USERS", "false")?;
            save_env_value(context.env_path(), "FEISHU_ALLOWED_USERS", "")?;
            writeln!(output, "  DM pairing enabled.")?;
        }
        1 => {
            save_env_value(context.env_path(), "FEISHU_ALLOW_ALL_USERS", "true")?;
            save_env_value(context.env_path(), "FEISHU_ALLOWED_USERS", "")?;
            writeln!(output, "  Open DM access enabled for Feishu / Lark.")?;
        }
        2 => {
            save_env_value(context.env_path(), "FEISHU_ALLOW_ALL_USERS", "false")?;
            let prompt =
                if let Some(open_id) = default_open_id.filter(|value| !value.trim().is_empty()) {
                    format!("  Allowed user IDs (comma-separated) [{open_id}]")
                } else {
                    String::from("  Allowed user IDs (comma-separated)")
                };
            let raw = prompt_gateway_line(input, output, &prompt)?;
            let allowed = if raw.trim().is_empty() {
                default_open_id.unwrap_or("").to_string()
            } else {
                normalize_gateway_allowlist("FEISHU_ALLOWED_USERS", raw.trim())
            };
            save_env_value(context.env_path(), "FEISHU_ALLOWED_USERS", &allowed)?;
            writeln!(output, "  Allowlist saved.")?;
        }
        _ => {}
    }
    Ok(())
}

fn configure_feishu_group_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    let group_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should group chats be handled?",
        &[
            "Respond only when @mentioned in groups (recommended)",
            "Disable group chats",
        ],
    )?;
    if group_idx == 0 {
        save_env_value(context.env_path(), "FEISHU_GROUP_POLICY", "open")?;
        writeln!(output, "  Group chats enabled (bot must be @mentioned).")?;
    } else {
        save_env_value(context.env_path(), "FEISHU_GROUP_POLICY", "disabled")?;
        writeln!(output, "  Group chats disabled.")?;
    }
    Ok(())
}

fn feishu_qr_register_with_io(
    output: &mut dyn Write,
) -> Result<Option<FeishuSetupCredentials>, Box<dyn Error>> {
    writeln!(output, "  Connecting to Feishu / Lark...")?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    let initial_domain = env::var("FEISHU_REGISTRATION_DOMAIN")
        .ok()
        .filter(|value| value.trim() == "lark")
        .unwrap_or_else(|| String::from("feishu"));

    let init = feishu_registration_post(&client, &initial_domain, &[("action", "init")])?;
    let supports_client_secret = init
        .get("supported_auth_methods")
        .and_then(JsonValue::as_array)
        .is_some_and(|methods| {
            methods
                .iter()
                .filter_map(JsonValue::as_str)
                .any(|method| method == "client_secret")
        });
    if !supports_client_secret {
        return Err("Feishu / Lark registration does not support client_secret auth".into());
    }

    let begin = feishu_registration_post(
        &client,
        &initial_domain,
        &[
            ("action", "begin"),
            ("archetype", "PersonalAgent"),
            ("auth_method", "client_secret"),
            ("request_user_info", "open_id"),
        ],
    )?;
    let device_code = feishu_json_string(&begin, "device_code")
        .ok_or("Feishu / Lark registration did not return a device_code")?;
    let mut qr_url = feishu_json_string(&begin, "verification_uri_complete").unwrap_or_default();
    if !qr_url.trim().is_empty() {
        qr_url = append_url_query_param(
            &append_url_query_param(&qr_url, "from", "hermes"),
            "tp",
            "hermes",
        );
    }
    let interval = feishu_json_u64(&begin, "interval").unwrap_or(5).max(1);
    let expire_in = feishu_json_u64(&begin, "expire_in").unwrap_or(600).max(1);
    let timeout_ms = feishu_env_u64("FEISHU_REGISTRATION_TIMEOUT_MS", expire_in * 1000).max(1);
    let poll_interval = Duration::from_millis(
        feishu_env_u64("FEISHU_REGISTRATION_POLL_INTERVAL_MS", interval * 1000).max(1),
    );

    if !qr_url.trim().is_empty() {
        writeln!(output)?;
        writeln!(output, "  Open this URL in Feishu / Lark on your phone:")?;
        writeln!(output, "  {qr_url}")?;
    }
    writeln!(output)?;
    writeln!(output, "  Waiting for QR scan confirmation...")?;

    let mut current_domain = initial_domain;
    let mut domain_switched = false;
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        let poll = feishu_registration_post(
            &client,
            &current_domain,
            &[
                ("action", "poll"),
                ("device_code", device_code.as_str()),
                ("tp", "ob_app"),
            ],
        )?;
        if feishu_nested_json_string(&poll, &["user_info", "tenant_brand"]).as_deref()
            == Some("lark")
            && !domain_switched
        {
            current_domain = String::from("lark");
            domain_switched = true;
        }
        if let (Some(app_id), Some(app_secret)) = (
            feishu_json_string(&poll, "client_id"),
            feishu_json_string(&poll, "client_secret"),
        ) {
            let open_id = feishu_nested_json_string(&poll, &["user_info", "open_id"]);
            let bot_info = feishu_probe_bot_raw(&app_id, &app_secret, &current_domain).ok();
            return Ok(Some(FeishuSetupCredentials {
                app_id,
                app_secret,
                domain: current_domain,
                open_id,
                bot_name: bot_info.and_then(|info| info.bot_name),
            }));
        }
        match feishu_json_string(&poll, "error").as_deref() {
            Some("access_denied" | "expired_token") => return Ok(None),
            _ => sleep(poll_interval),
        }
    }
    Ok(None)
}

fn feishu_registration_post(
    client: &reqwest::blocking::Client,
    domain: &str,
    params: &[(&str, &str)],
) -> Result<JsonValue, Box<dyn Error>> {
    let base = feishu_registration_base_url(domain);
    let response = client
        .post(format!("{base}/oauth/v1/app/registration"))
        .form(&params)
        .send()?;
    let text = response.text()?;
    Ok(serde_json::from_str(&text)?)
}

fn feishu_registration_base_url(domain: &str) -> String {
    if let Ok(value) = env::var("FEISHU_REGISTRATION_BASE_URL")
        && !value.trim().is_empty()
    {
        return value.trim().trim_end_matches('/').to_string();
    }
    match domain {
        "lark" => String::from("https://accounts.larksuite.com"),
        _ => String::from("https://accounts.feishu.cn"),
    }
}

fn feishu_open_base_url(domain: &str) -> String {
    if let Ok(value) = env::var("FEISHU_OPEN_BASE_URL")
        && !value.trim().is_empty()
    {
        return value.trim().trim_end_matches('/').to_string();
    }
    match domain {
        "lark" => String::from("https://open.larksuite.com"),
        _ => String::from("https://open.feishu.cn"),
    }
}

fn feishu_probe_bot_raw(
    app_id: &str,
    app_secret: &str,
    domain: &str,
) -> Result<FeishuSetupCredentials, Box<dyn Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let base = feishu_open_base_url(domain);
    let token = client
        .post(format!(
            "{base}/open-apis/auth/v3/tenant_access_token/internal"
        ))
        .json(&serde_json::json!({
            "app_id": app_id,
            "app_secret": app_secret,
        }))
        .send()?
        .json::<JsonValue>()?;
    let access_token = feishu_json_string(&token, "tenant_access_token")
        .ok_or("tenant access token not returned")?;
    let bot = client
        .get(format!("{base}/open-apis/bot/v3/info"))
        .bearer_auth(access_token)
        .send()?
        .json::<JsonValue>()?;
    if bot.get("code").and_then(JsonValue::as_i64) != Some(0) {
        return Err("bot info probe failed".into());
    }
    Ok(FeishuSetupCredentials {
        app_id: app_id.to_string(),
        app_secret: app_secret.to_string(),
        domain: domain.to_string(),
        open_id: None,
        bot_name: feishu_nested_json_string(&bot, &["bot", "app_name"])
            .or_else(|| feishu_nested_json_string(&bot, &["bot", "bot_name"]))
            .or_else(|| feishu_nested_json_string(&bot, &["data", "bot", "app_name"]))
            .or_else(|| feishu_nested_json_string(&bot, &["data", "bot", "bot_name"])),
    })
}

fn feishu_json_string(value: &JsonValue, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn feishu_json_u64(value: &JsonValue, key: &str) -> Option<u64> {
    value.get(key).and_then(JsonValue::as_u64)
}

fn feishu_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn feishu_nested_json_string(value: &JsonValue, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn configure_wecom_gateway_platform_with_io(
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

    let existing_bot_id = read_effective_env_value(context, "WECOM_BOT_ID");
    let existing_secret = read_effective_env_value(context, "WECOM_SECRET");
    if existing_bot_id.is_some() && existing_secret.is_some() {
        writeln!(output)?;
        writeln!(output, "WeCom is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure WeCom?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    let method = prompt_gateway_menu_choice(
        input,
        output,
        "How would you like to set up WeCom?",
        &[
            "Scan QR code to obtain Bot ID and Secret automatically (recommended)",
            "Enter existing Bot ID and Secret manually",
        ],
    )?;

    let configured = if method == 0 {
        match wecom_qr_scan_for_bot_info_with_io(output) {
            Ok(Some((bot_id, secret))) => {
                save_env_value(context.env_path(), "WECOM_BOT_ID", &bot_id)?;
                save_env_value(context.env_path(), "WECOM_SECRET", &secret)?;
                writeln!(output, "  QR scan successful. Bot ID and Secret saved.")?;
                true
            }
            Ok(None) => {
                writeln!(
                    output,
                    "  QR scan did not complete. Continuing with manual input."
                )?;
                configure_wecom_manual_credentials(context, input, output)?
            }
            Err(error) => {
                writeln!(output, "  QR scan failed: {error}")?;
                writeln!(output, "  Continuing with manual input.")?;
                configure_wecom_manual_credentials(context, input, output)?
            }
        }
    } else {
        configure_wecom_manual_credentials(context, input, output)?
    };

    if !configured {
        return Ok(());
    }

    configure_wecom_access_policy(context, input, output)?;

    writeln!(output)?;
    writeln!(output, "  Chat ID for scheduled results and notifications.")?;
    let home = prompt_gateway_line(
        input,
        output,
        "  Home chat ID (optional, for cron/notifications)",
    )?;
    if !home.trim().is_empty() {
        save_env_value(context.env_path(), "WECOM_HOME_CHANNEL", home.trim())?;
        writeln!(output, "  Home channel set to {}", home.trim())?;
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    Ok(())
}

fn configure_wecom_manual_credentials(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<bool, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  1. Go to WeCom Application → Workspace → Smart Robot → Create smart robots"
    )?;
    writeln!(output, "  2. Select API Mode")?;
    writeln!(
        output,
        "  3. Copy the Bot ID and Secret from the bot's credentials info"
    )?;
    writeln!(
        output,
        "  4. The bot connects via WebSocket — no public endpoint needed"
    )?;

    let Some(bot_id) = prompt_gateway_required_line(
        input,
        output,
        "  Bot ID",
        "Skipped — WeCom won't work without a Bot ID.",
    )?
    else {
        return Ok(false);
    };
    let Some(secret) = prompt_gateway_required_line(
        input,
        output,
        "  Secret",
        "Skipped — WeCom won't work without a Secret.",
    )?
    else {
        return Ok(false);
    };
    save_env_value(context.env_path(), "WECOM_BOT_ID", &bot_id)?;
    save_env_value(context.env_path(), "WECOM_SECRET", &secret)?;
    Ok(true)
}

fn configure_wecom_access_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  The gateway denies all users by default for security."
    )?;
    writeln!(
        output,
        "  Enter user IDs to create an allowlist, or leave empty."
    )?;
    let allowed = prompt_gateway_line(
        input,
        output,
        "  Allowed user IDs (comma-separated, or empty)",
    )?;
    if !allowed.trim().is_empty() {
        let cleaned = normalize_gateway_allowlist("WECOM_ALLOWED_USERS", allowed.trim());
        save_env_value(context.env_path(), "WECOM_ALLOWED_USERS", &cleaned)?;
        remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
        writeln!(
            output,
            "  Saved — only these users can interact with the bot."
        )?;
        return Ok(());
    }

    writeln!(output)?;
    let access_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should unauthorized users be handled?",
        &[
            "Enable open access (anyone can message the bot)",
            "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
            "Disable direct messages",
            "Skip for now (bot will deny all users until configured)",
        ],
    )?;
    match access_idx {
        0 => {
            save_env_value(context.env_path(), "WECOM_DM_POLICY", "open")?;
            save_env_value(context.env_path(), "GATEWAY_ALLOW_ALL_USERS", "true")?;
            writeln!(output, "  Open access enabled.")?;
        }
        1 => {
            save_env_value(context.env_path(), "WECOM_DM_POLICY", "pairing")?;
            remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
            writeln!(
                output,
                "  DM pairing mode selected. Approve codes with `hermes pairing approve`."
            )?;
        }
        2 => {
            save_env_value(context.env_path(), "WECOM_DM_POLICY", "disabled")?;
            remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
            writeln!(output, "  Direct messages disabled.")?;
        }
        3 => {
            remove_env_key_if_present(&context.env_path(), "GATEWAY_ALLOW_ALL_USERS")?;
            writeln!(
                output,
                "  Skipped — configure later with `hermes gateway setup`."
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn wecom_qr_scan_for_bot_info_with_io(
    output: &mut dyn Write,
) -> Result<Option<(String, String)>, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "  Requesting WeCom QR authorization code...")?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    let generated = wecom_get_json(&client, &wecom_qr_generate_url(), "generate QR code")?;
    let scode = wecom_nested_json_string(&generated, &["data", "scode"])
        .ok_or("WeCom QR response missing scode")?;
    let auth_url = wecom_nested_json_string(&generated, &["data", "auth_url"]);
    let page_url = wecom_qr_code_page_url(&scode);

    writeln!(output)?;
    writeln!(output, "  Open this URL in WeCom on your phone:")?;
    writeln!(output, "  {page_url}")?;
    if let Some(auth_url) = auth_url.filter(|value| value != &page_url) {
        writeln!(output, "  Auth URL: {auth_url}")?;
    }
    writeln!(output)?;
    writeln!(output, "  Waiting for QR scan confirmation...")?;

    let interval = Duration::from_millis(wecom_env_u64("WECOM_QR_POLL_INTERVAL_MS", 3000).max(1));
    let timeout_ms = wecom_env_u64("WECOM_QR_TIMEOUT_MS", 300_000).max(1);
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        let poll = wecom_get_json(&client, &wecom_qr_query_url(&scode), "query QR scan result")?;
        let status = wecom_nested_json_string(&poll, &["data", "status"])
            .unwrap_or_default()
            .to_ascii_lowercase();
        if status == "success" {
            let bot_id = wecom_nested_json_string(&poll, &["data", "bot_info", "botid"])
                .or_else(|| wecom_nested_json_string(&poll, &["data", "bot_info", "bot_id"]))
                .ok_or("WeCom QR success response missing bot ID")?;
            let secret = wecom_nested_json_string(&poll, &["data", "bot_info", "secret"])
                .ok_or("WeCom QR success response missing secret")?;
            return Ok(Some((bot_id, secret)));
        }
        if matches!(
            status.as_str(),
            "failed" | "fail" | "expired" | "cancelled" | "canceled"
        ) {
            return Err(format!("WeCom QR scan status: {status}").into());
        }
        sleep(interval);
    }
    Ok(None)
}

fn wecom_get_json(
    client: &reqwest::blocking::Client,
    url: &str,
    action: &str,
) -> Result<JsonValue, Box<dyn Error>> {
    let response = client.get(url).send()?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("WeCom {action} returned HTTP {status}").into());
    }
    Ok(response.json::<JsonValue>()?)
}

fn wecom_qr_generate_url() -> String {
    let base = env::var("WECOM_QR_GENERATE_URL")
        .unwrap_or_else(|_| String::from("https://work.weixin.qq.com/ai/qc/generate"))
        .trim()
        .to_string();
    if base.contains("source=") {
        base
    } else {
        append_url_query_param(&base, "source", "hermes")
    }
}

fn wecom_qr_query_url(scode: &str) -> String {
    let base = env::var("WECOM_QR_QUERY_URL")
        .unwrap_or_else(|_| String::from("https://work.weixin.qq.com/ai/qc/query_result"))
        .trim()
        .to_string();
    append_url_query_param(&base, "scode", scode)
}

fn wecom_qr_code_page_url(scode: &str) -> String {
    let base = env::var("WECOM_QR_CODE_PAGE")
        .unwrap_or_else(|_| {
            String::from("https://work.weixin.qq.com/ai/qc/gen?source=hermes&scode=")
        })
        .trim()
        .to_string();
    let encoded = percent_encode_url_component(scode);
    if base.contains("{scode}") {
        return base.replace("{scode}", &encoded);
    }
    if base.ends_with('=') || base.ends_with('/') {
        format!("{base}{encoded}")
    } else {
        append_url_query_param(&base, "scode", scode)
    }
}

fn append_url_query_param(url: &str, key: &str, value: &str) -> String {
    let separator = if url.contains('?') {
        if url.ends_with('?') || url.ends_with('&') {
            ""
        } else {
            "&"
        }
    } else {
        "?"
    };
    format!(
        "{url}{separator}{key}={}",
        percent_encode_url_component(value)
    )
}

fn percent_encode_url_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn wecom_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn wecom_nested_json_string(value: &JsonValue, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

const WEIXIN_DEFAULT_ILINK_BASE_URL: &str = "https://ilinkai.weixin.qq.com";
const WEIXIN_DEFAULT_CDN_BASE_URL: &str = "https://novac2c.cdn.weixin.qq.com/c2c";

#[derive(Debug, Clone)]
struct WeixinSetupCredentials {
    account_id: String,
    token: String,
    base_url: String,
    user_id: Option<String>,
}

fn configure_weixin_gateway_platform_with_io(
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

    let existing_account = read_effective_env_value(context, "WEIXIN_ACCOUNT_ID");
    let existing_token = read_effective_env_value(context, "WEIXIN_TOKEN");
    if existing_account.is_some() && existing_token.is_some() {
        writeln!(output)?;
        writeln!(output, "Weixin is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure Weixin?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    if !prompt_gateway_yes_no(input, output, "Start QR login now?", true)? {
        writeln!(output, "  Cancelled.")?;
        return Ok(());
    }

    let credentials = match weixin_qr_login_with_io(context, output) {
        Ok(Some(credentials)) => credentials,
        Ok(None) => {
            writeln!(output, "  QR login did not complete.")?;
            return Ok(());
        }
        Err(error) => {
            writeln!(output, "  QR login failed: {error}")?;
            return Ok(());
        }
    };

    save_env_value(
        context.env_path(),
        "WEIXIN_ACCOUNT_ID",
        &credentials.account_id,
    )?;
    save_env_value(context.env_path(), "WEIXIN_TOKEN", &credentials.token)?;
    if !credentials.base_url.trim().is_empty() {
        save_env_value(context.env_path(), "WEIXIN_BASE_URL", &credentials.base_url)?;
    }
    let cdn_base = read_effective_env_value(context, "WEIXIN_CDN_BASE_URL")
        .unwrap_or_else(|| WEIXIN_DEFAULT_CDN_BASE_URL.to_string());
    save_env_value(context.env_path(), "WEIXIN_CDN_BASE_URL", &cdn_base)?;

    configure_weixin_dm_policy(context, input, output, credentials.user_id.as_deref())?;
    configure_weixin_group_policy(context, input, output)?;
    if let Some(user_id) = credentials
        .user_id
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        writeln!(output)?;
        if prompt_gateway_yes_no(
            input,
            output,
            format!("Use your Weixin user ID ({user_id}) as the home channel?").as_str(),
            true,
        )? {
            save_env_value(context.env_path(), "WEIXIN_HOME_CHANNEL", user_id)?;
            writeln!(output, "  Home channel set to {user_id}")?;
        }
    }

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    writeln!(output, "  Account ID: {}", credentials.account_id)?;
    if let Some(user_id) = credentials.user_id.as_deref() {
        writeln!(output, "  User ID: {user_id}")?;
    }
    Ok(())
}

fn configure_weixin_dm_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    user_id: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    let access_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should direct messages be authorized?",
        &[
            "Use DM pairing approval (recommended)",
            "Allow all direct messages",
            "Only allow listed user IDs",
            "Disable direct messages",
        ],
    )?;
    match access_idx {
        0 => {
            save_env_value(context.env_path(), "WEIXIN_DM_POLICY", "pairing")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOW_ALL_USERS", "false")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOWED_USERS", "")?;
            writeln!(output, "  DM pairing enabled.")?;
        }
        1 => {
            save_env_value(context.env_path(), "WEIXIN_DM_POLICY", "open")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOW_ALL_USERS", "true")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOWED_USERS", "")?;
            writeln!(output, "  Open DM access enabled for Weixin.")?;
        }
        2 => {
            let prompt = if let Some(user_id) = user_id.filter(|value| !value.is_empty()) {
                format!("  Allowed Weixin user IDs (comma-separated) [{user_id}]")
            } else {
                String::from("  Allowed Weixin user IDs (comma-separated)")
            };
            let raw = prompt_gateway_line(input, output, &prompt)?;
            let allowlist = if raw.trim().is_empty() {
                user_id.unwrap_or("").to_string()
            } else {
                normalize_gateway_allowlist("WEIXIN_ALLOWED_USERS", raw.trim())
            };
            save_env_value(context.env_path(), "WEIXIN_DM_POLICY", "allowlist")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOW_ALL_USERS", "false")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOWED_USERS", &allowlist)?;
            writeln!(output, "  Weixin allowlist saved.")?;
        }
        3 => {
            save_env_value(context.env_path(), "WEIXIN_DM_POLICY", "disabled")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOW_ALL_USERS", "false")?;
            save_env_value(context.env_path(), "WEIXIN_ALLOWED_USERS", "")?;
            writeln!(output, "  Direct messages disabled.")?;
        }
        _ => {}
    }
    Ok(())
}

fn configure_weixin_group_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  Note: QR login connects an iLink bot identity, not a scriptable personal WeChat account."
    )?;
    writeln!(
        output,
        "  Ordinary WeChat groups usually cannot invite that identity; these settings only apply if iLink delivers group events."
    )?;
    let group_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should group chats be handled?",
        &[
            "Disable group chats (recommended)",
            "Allow all group chats",
            "Only allow listed group chat IDs",
        ],
    )?;
    match group_idx {
        0 => {
            save_env_value(context.env_path(), "WEIXIN_GROUP_POLICY", "disabled")?;
            save_env_value(context.env_path(), "WEIXIN_GROUP_ALLOWED_USERS", "")?;
            writeln!(output, "  Group chats disabled.")?;
        }
        1 => {
            save_env_value(context.env_path(), "WEIXIN_GROUP_POLICY", "open")?;
            save_env_value(context.env_path(), "WEIXIN_GROUP_ALLOWED_USERS", "")?;
            writeln!(
                output,
                "  All group chats enabled if iLink delivers group events."
            )?;
        }
        2 => {
            let raw = prompt_gateway_line(
                input,
                output,
                "  Allowed group chat IDs (comma-separated, not member user IDs)",
            )?;
            let allowlist = normalize_gateway_allowlist("WEIXIN_GROUP_ALLOWED_USERS", raw.trim());
            save_env_value(context.env_path(), "WEIXIN_GROUP_POLICY", "allowlist")?;
            save_env_value(context.env_path(), "WEIXIN_GROUP_ALLOWED_USERS", &allowlist)?;
            writeln!(output, "  Group allowlist saved.")?;
        }
        _ => {}
    }
    Ok(())
}

fn weixin_qr_login_with_io(
    context: &HermesContext,
    output: &mut dyn Write,
) -> Result<Option<WeixinSetupCredentials>, Box<dyn Error>> {
    let base = weixin_ilink_base_url();
    let request_timeout_ms = weixin_env_u64("WEIXIN_QR_TIMEOUT_MS", 35_000).max(1);
    let login_timeout_ms = weixin_env_u64("WEIXIN_QR_LOGIN_TIMEOUT_MS", 480_000).max(1);
    let poll_interval =
        Duration::from_millis(weixin_env_u64("WEIXIN_QR_POLL_INTERVAL_MS", 1000).max(1));
    let bot_type = env::var("WEIXIN_QR_BOT_TYPE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "3".to_string());

    let mut qr = weixin_fetch_qr(&base, &bot_type, request_timeout_ms)?;
    let mut current_base = base.clone();
    let deadline = Instant::now() + Duration::from_millis(login_timeout_ms);
    let mut refresh_count = 0;

    loop {
        let scan_data = qr.scan_url.as_deref().unwrap_or(&qr.qrcode);
        writeln!(output)?;
        writeln!(output, "  Use WeChat to scan this QR login URL:")?;
        writeln!(output, "  {scan_data}")?;

        while Instant::now() < deadline {
            let poll = match weixin_poll_qr_status(&current_base, &qr.qrcode, request_timeout_ms) {
                Ok(value) => value,
                Err(error) => {
                    writeln!(output, "  QR poll failed: {error}")?;
                    sleep(poll_interval);
                    continue;
                }
            };
            match poll {
                WeixinQrPoll::Wait => sleep(poll_interval),
                WeixinQrPoll::Scanned => {
                    writeln!(output, "  QR scanned. Confirm login in WeChat.")?;
                    sleep(poll_interval);
                }
                WeixinQrPoll::Redirect(url) => {
                    current_base = url;
                    sleep(poll_interval);
                }
                WeixinQrPoll::Expired => {
                    refresh_count += 1;
                    if refresh_count > 3 {
                        return Ok(None);
                    }
                    writeln!(
                        output,
                        "  QR code expired, refreshing... ({refresh_count}/3)"
                    )?;
                    qr = weixin_fetch_qr(&base, &bot_type, request_timeout_ms)?;
                    current_base = base.clone();
                    break;
                }
                WeixinQrPoll::Confirmed(credentials) => {
                    weixin_save_account(context, &credentials)?;
                    writeln!(output)?;
                    writeln!(
                        output,
                        "  Weixin QR login complete. account_id={}",
                        credentials.account_id
                    )?;
                    return Ok(Some(credentials));
                }
            }
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
    }
}

struct WeixinQrCode {
    qrcode: String,
    scan_url: Option<String>,
}

enum WeixinQrPoll {
    Wait,
    Scanned,
    Redirect(String),
    Expired,
    Confirmed(WeixinSetupCredentials),
}

fn weixin_fetch_qr(
    base_url: &str,
    bot_type: &str,
    timeout_ms: u64,
) -> Result<WeixinQrCode, Box<dyn Error>> {
    let endpoint = format!(
        "ilink/bot/get_bot_qrcode?bot_type={}",
        percent_encode_url_component(bot_type)
    );
    let data = weixin_api_get(base_url, &endpoint, timeout_ms)?;
    let qrcode = weixin_json_string(&data, "qrcode").ok_or("Weixin QR response missing qrcode")?;
    let scan_url = weixin_json_string(&data, "qrcode_img_content");
    Ok(WeixinQrCode { qrcode, scan_url })
}

fn weixin_poll_qr_status(
    base_url: &str,
    qrcode: &str,
    timeout_ms: u64,
) -> Result<WeixinQrPoll, Box<dyn Error>> {
    let endpoint = format!(
        "ilink/bot/get_qrcode_status?qrcode={}",
        percent_encode_url_component(qrcode)
    );
    let data = weixin_api_get(base_url, &endpoint, timeout_ms)?;
    let status = weixin_json_string(&data, "status").unwrap_or_else(|| "wait".to_string());
    match status.as_str() {
        "wait" => Ok(WeixinQrPoll::Wait),
        "scaned" => Ok(WeixinQrPoll::Scanned),
        "scaned_but_redirect" => {
            let redirect = weixin_json_string(&data, "redirect_host").unwrap_or_default();
            if redirect.trim().is_empty() {
                return Ok(WeixinQrPoll::Wait);
            }
            Ok(WeixinQrPoll::Redirect(weixin_redirect_base_url(&redirect)))
        }
        "expired" => Ok(WeixinQrPoll::Expired),
        "confirmed" => {
            let account_id = weixin_json_string(&data, "ilink_bot_id")
                .ok_or("Weixin QR status missing ilink_bot_id")?;
            let token = weixin_json_string(&data, "bot_token")
                .ok_or("Weixin QR status missing bot_token")?;
            let base_url = weixin_json_string(&data, "baseurl")
                .unwrap_or_else(|| WEIXIN_DEFAULT_ILINK_BASE_URL.to_string());
            let user_id = weixin_json_string(&data, "ilink_user_id");
            Ok(WeixinQrPoll::Confirmed(WeixinSetupCredentials {
                account_id,
                token,
                base_url,
                user_id,
            }))
        }
        _ => Ok(WeixinQrPoll::Wait),
    }
}

fn weixin_api_get(
    base_url: &str,
    endpoint: &str,
    timeout_ms: u64,
) -> Result<JsonValue, Box<dyn Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .build()?;
    let url = format!("{}/{}", base_url.trim_end_matches('/'), endpoint);
    let response = client
        .get(&url)
        .header("iLink-App-Id", "bot")
        .header("iLink-App-ClientVersion", "131584")
        .send()?;
    let status = response.status();
    let raw = response.text()?;
    if !status.is_success() {
        return Err(format!("iLink GET {endpoint} HTTP {status}: {}", raw.trim()).into());
    }
    Ok(serde_json::from_str(&raw)?)
}

fn weixin_save_account(
    context: &HermesContext,
    credentials: &WeixinSetupCredentials,
) -> Result<(), Box<dyn Error>> {
    let dir = context.hermes_home().join("weixin").join("accounts");
    fs::create_dir_all(&dir)?;
    let file_name = credentials
        .account_id
        .chars()
        .map(|ch| {
            if matches!(ch, '/' | '\\' | '\0') {
                '_'
            } else {
                ch
            }
        })
        .collect::<String>();
    let path = dir.join(format!("{file_name}.json"));
    let saved_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);
    let payload = serde_json::json!({
        "token": credentials.token,
        "base_url": credentials.base_url,
        "user_id": credentials.user_id.as_deref().unwrap_or(""),
        "saved_at": saved_at.to_string(),
    });
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&payload)?),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&path, perms)?;
    }
    Ok(())
}

fn weixin_ilink_base_url() -> String {
    env::var("WEIXIN_ILINK_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| WEIXIN_DEFAULT_ILINK_BASE_URL.to_string())
        .trim()
        .trim_end_matches('/')
        .to_string()
}

fn weixin_redirect_base_url(value: &str) -> String {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    }
}

fn weixin_json_string(value: &JsonValue, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn weixin_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
struct QqbotSetupCredentials {
    app_id: String,
    client_secret: String,
    user_openid: Option<String>,
}

fn configure_qqbot_gateway_platform_with_io(
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

    let existing_app_id = read_effective_env_value(context, "QQ_APP_ID");
    let existing_secret = read_effective_env_value(context, "QQ_CLIENT_SECRET");
    if existing_app_id.is_some() && existing_secret.is_some() {
        writeln!(output)?;
        writeln!(output, "QQ Bot is already configured.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure QQ Bot?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    let method = prompt_gateway_menu_choice(
        input,
        output,
        "How would you like to set up QQ Bot?",
        &[
            "Scan QR code to add bot automatically (recommended)",
            "Enter existing App ID and App Secret manually",
        ],
    )?;

    let credentials = if method == 0 {
        match qqbot_qr_register_with_io(output) {
            Ok(Some(credentials)) => Some(credentials),
            Ok(None) => {
                writeln!(
                    output,
                    "  QR setup did not complete. Continuing with manual input."
                )?;
                configure_qqbot_manual_credentials(input, output)?
            }
            Err(error) => {
                writeln!(output, "  QR registration failed: {error}")?;
                writeln!(output, "  Continuing with manual input.")?;
                configure_qqbot_manual_credentials(input, output)?
            }
        }
    } else {
        configure_qqbot_manual_credentials(input, output)?
    };

    let Some(credentials) = credentials else {
        return Ok(());
    };

    save_env_value(context.env_path(), "QQ_APP_ID", &credentials.app_id)?;
    save_env_value(
        context.env_path(),
        "QQ_CLIENT_SECRET",
        &credentials.client_secret,
    )?;

    configure_qqbot_dm_policy(context, input, output, credentials.user_openid.as_deref())?;
    configure_qqbot_home_channel(context, input, output, credentials.user_openid.as_deref())?;

    writeln!(output)?;
    writeln!(output, "{} {} configured!", platform.emoji, platform.label)?;
    writeln!(output, "  App ID: {}", credentials.app_id)?;
    Ok(())
}

fn configure_qqbot_manual_credentials(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<Option<QqbotSetupCredentials>, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(
        output,
        "  Go to https://q.qq.com to register a QQ Bot application."
    )?;
    writeln!(
        output,
        "  Note your App ID and App Secret from the application page."
    )?;
    let Some(app_id) = prompt_gateway_required_line(
        input,
        output,
        "  App ID",
        "Skipped — QQ Bot won't work without an App ID.",
    )?
    else {
        return Ok(None);
    };
    let Some(client_secret) = prompt_gateway_required_line(
        input,
        output,
        "  App Secret",
        "Skipped — QQ Bot won't work without an App Secret.",
    )?
    else {
        return Ok(None);
    };
    Ok(Some(QqbotSetupCredentials {
        app_id,
        client_secret,
        user_openid: None,
    }))
}

fn configure_qqbot_dm_policy(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    user_openid: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    let access_idx = prompt_gateway_menu_choice(
        input,
        output,
        "How should direct messages be authorized?",
        &[
            "Use DM pairing approval (recommended)",
            "Allow all direct messages",
            "Only allow listed user OpenIDs",
        ],
    )?;
    match access_idx {
        0 => {
            save_env_value(context.env_path(), "QQ_ALLOW_ALL_USERS", "false")?;
            if let Some(openid) = user_openid.filter(|value| !value.trim().is_empty()) {
                writeln!(output)?;
                if prompt_gateway_yes_no(
                    input,
                    output,
                    format!("Add yourself ({openid}) to the allow list?").as_str(),
                    true,
                )? {
                    save_env_value(context.env_path(), "QQ_ALLOWED_USERS", openid)?;
                    writeln!(output, "  Allow list set to {openid}")?;
                } else {
                    save_env_value(context.env_path(), "QQ_ALLOWED_USERS", "")?;
                }
            } else {
                save_env_value(context.env_path(), "QQ_ALLOWED_USERS", "")?;
            }
            writeln!(output, "  DM pairing enabled.")?;
        }
        1 => {
            save_env_value(context.env_path(), "QQ_ALLOW_ALL_USERS", "true")?;
            save_env_value(context.env_path(), "QQ_ALLOWED_USERS", "")?;
            writeln!(output, "  Open DM access enabled for QQ Bot.")?;
        }
        2 => {
            let prompt = if let Some(openid) = user_openid.filter(|value| !value.trim().is_empty())
            {
                format!("  Allowed user OpenIDs (comma-separated) [{openid}]")
            } else {
                String::from("  Allowed user OpenIDs (comma-separated)")
            };
            let raw = prompt_gateway_line(input, output, &prompt)?;
            let allowed = if raw.trim().is_empty() {
                user_openid.unwrap_or("").to_string()
            } else {
                normalize_gateway_allowlist("QQ_ALLOWED_USERS", raw.trim())
            };
            save_env_value(context.env_path(), "QQ_ALLOW_ALL_USERS", "false")?;
            save_env_value(context.env_path(), "QQ_ALLOWED_USERS", &allowed)?;
            writeln!(output, "  Allowlist saved.")?;
        }
        _ => {}
    }
    Ok(())
}

fn configure_qqbot_home_channel(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    user_openid: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if let Some(openid) = user_openid.filter(|value| !value.trim().is_empty()) {
        writeln!(output)?;
        if prompt_gateway_yes_no(
            input,
            output,
            format!("Use your QQ user ID ({openid}) as the home channel?").as_str(),
            true,
        )? {
            save_env_value(context.env_path(), "QQBOT_HOME_CHANNEL", openid)?;
            writeln!(output, "  Home channel set to {openid}")?;
        }
    } else {
        writeln!(output)?;
        let home_channel = prompt_gateway_line(
            input,
            output,
            "  Home channel OpenID (for cron/notifications, or empty)",
        )?;
        if !home_channel.trim().is_empty() {
            save_env_value(
                context.env_path(),
                "QQBOT_HOME_CHANNEL",
                home_channel.trim(),
            )?;
            writeln!(output, "  Home channel set to {}", home_channel.trim())?;
        }
    }
    Ok(())
}

fn qqbot_qr_register_with_io(
    output: &mut dyn Write,
) -> Result<Option<QqbotSetupCredentials>, Box<dyn Error>> {
    let timeout_ms = qqbot_env_u64("QQBOT_ONBOARD_TIMEOUT_MS", 600_000).max(1);
    let poll_interval =
        Duration::from_millis(qqbot_env_u64("QQBOT_ONBOARD_POLL_INTERVAL_MS", 2000).max(1));
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);

    for refresh_count in 0..=3 {
        let (task_id, aes_key) = qqbot_create_bind_task()?;
        let url = qqbot_connect_url(&task_id);
        writeln!(output)?;
        writeln!(output, "  Open this URL in QQ on your phone:")?;
        writeln!(output, "  {url}")?;

        while Instant::now() < deadline {
            match qqbot_poll_bind_result(&task_id) {
                Ok(QqbotBindPoll::Completed {
                    app_id,
                    encrypted_secret,
                    user_openid,
                }) => {
                    let client_secret = qqbot_decrypt_secret(&encrypted_secret, &aes_key)?;
                    writeln!(output)?;
                    writeln!(output, "  QR scan complete. App ID: {app_id}")?;
                    if let Some(openid) = user_openid.as_deref() {
                        writeln!(output, "  Scanner OpenID: {openid}")?;
                    }
                    return Ok(Some(QqbotSetupCredentials {
                        app_id,
                        client_secret,
                        user_openid,
                    }));
                }
                Ok(QqbotBindPoll::Expired) => {
                    if refresh_count >= 3 {
                        return Ok(None);
                    }
                    writeln!(
                        output,
                        "  QR code expired, refreshing... ({}/{})",
                        refresh_count + 1,
                        3
                    )?;
                    break;
                }
                Ok(QqbotBindPoll::Pending) | Ok(QqbotBindPoll::None) | Err(_) => {
                    sleep(poll_interval);
                }
            }
        }
    }
    Ok(None)
}

enum QqbotBindPoll {
    None,
    Pending,
    Completed {
        app_id: String,
        encrypted_secret: String,
        user_openid: Option<String>,
    },
    Expired,
}

fn qqbot_create_bind_task() -> Result<(String, String), Box<dyn Error>> {
    let key = qqbot_generate_bind_key()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let data = client
        .post(format!("{}/lite/create_bind_task", qqbot_portal_base_url()))
        .header("Accept", "application/json")
        .json(&serde_json::json!({ "key": key }))
        .send()?
        .json::<JsonValue>()?;
    qqbot_require_retcode_ok(&data, "create_bind_task")?;
    let task_id = data
        .get("data")
        .and_then(|data| data.get("task_id"))
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("create_bind_task response missing task_id")?;
    Ok((task_id.to_string(), key))
}

fn qqbot_poll_bind_result(task_id: &str) -> Result<QqbotBindPoll, Box<dyn Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let data = client
        .post(format!("{}/lite/poll_bind_result", qqbot_portal_base_url()))
        .header("Accept", "application/json")
        .json(&serde_json::json!({ "task_id": task_id }))
        .send()?
        .json::<JsonValue>()?;
    qqbot_require_retcode_ok(&data, "poll_bind_result")?;
    let body = data.get("data").unwrap_or(&JsonValue::Null);
    match body.get("status").and_then(JsonValue::as_i64).unwrap_or(0) {
        1 => Ok(QqbotBindPoll::Pending),
        2 => {
            let app_id = body
                .get("bot_appid")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or("poll_bind_result response missing bot_appid")?;
            let encrypted_secret = body
                .get("bot_encrypt_secret")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or("poll_bind_result response missing bot_encrypt_secret")?;
            let user_openid = body
                .get("user_openid")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            Ok(QqbotBindPoll::Completed {
                app_id: app_id.to_string(),
                encrypted_secret: encrypted_secret.to_string(),
                user_openid,
            })
        }
        3 => Ok(QqbotBindPoll::Expired),
        _ => Ok(QqbotBindPoll::None),
    }
}

fn qqbot_require_retcode_ok(value: &JsonValue, action: &str) -> Result<(), Box<dyn Error>> {
    if value.get("retcode").and_then(JsonValue::as_i64) == Some(0) {
        return Ok(());
    }
    let message = value
        .get("msg")
        .and_then(JsonValue::as_str)
        .unwrap_or(action);
    Err(format!("QQ Bot onboard {action} failed: {message}").into())
}

fn qqbot_generate_bind_key() -> Result<String, Box<dyn Error>> {
    let mut key = [0_u8; 32];
    fill_random(&mut key).map_err(|error| -> Box<dyn Error> {
        format!("QQ Bot bind key generation failed: {error:?}").into()
    })?;
    Ok(BASE64_STANDARD.encode(key))
}

fn qqbot_decrypt_secret(
    encrypted_base64: &str,
    key_base64: &str,
) -> Result<String, Box<dyn Error>> {
    let key = BASE64_STANDARD.decode(key_base64)?;
    if key.len() != 32 {
        return Err("QQ Bot bind key must decode to 32 bytes".into());
    }
    let raw = BASE64_STANDARD.decode(encrypted_base64)?;
    if raw.len() < 12 + 16 {
        return Err("QQ Bot encrypted secret is too short".into());
    }
    let nonce_bytes: [u8; 12] = raw[..12]
        .try_into()
        .map_err(|_| "QQ Bot encrypted secret has an invalid nonce")?;
    let mut ciphertext = raw[12..].to_vec();
    let unbound = ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, &key)
        .map_err(|_| "QQ Bot AES key initialization failed")?;
    let key = ring::aead::LessSafeKey::new(unbound);
    let plaintext = key
        .open_in_place(
            ring::aead::Nonce::assume_unique_for_key(nonce_bytes),
            ring::aead::Aad::empty(),
            &mut ciphertext,
        )
        .map_err(|_| "QQ Bot encrypted secret authentication failed")?;
    Ok(String::from_utf8(plaintext.to_vec())?)
}

fn qqbot_portal_base_url() -> String {
    if let Ok(value) = env::var("QQ_PORTAL_BASE_URL")
        && !value.trim().is_empty()
    {
        return value.trim().trim_end_matches('/').to_string();
    }
    let host = env::var("QQ_PORTAL_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| String::from("q.qq.com"));
    format!("https://{}", host.trim().trim_end_matches('/'))
}

fn qqbot_connect_url(task_id: &str) -> String {
    let template = env::var("QQBOT_QR_URL_TEMPLATE").unwrap_or_else(|_| {
        String::from(
            "https://q.qq.com/qqbot/openclaw/connect.html?task_id={task_id}&_wv=2&source=hermes",
        )
    });
    template.replace("{task_id}", &percent_encode_url_component(task_id))
}

fn qqbot_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn gateway_python_module_installed(module: &str) -> Result<bool, Box<dyn Error>> {
    let root = project_root();
    let Some(python) = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON")) else {
        return Ok(false);
    };
    let status = Command::new(python)
        .arg("-c")
        .arg(format!(
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module:?}) else 1)"
        ))
        .status()?;
    Ok(status.success())
}

fn gateway_install_python_package(package: &str) -> Result<bool, Box<dyn Error>> {
    let root = project_root();
    let Some(python) = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON")) else {
        return Ok(false);
    };
    let status = Command::new(python)
        .args(["-m", "pip", "install", "-U", package])
        .status()?;
    Ok(status.success())
}

fn configure_native_gateway_plugin_platform_with_io(
    context: &HermesContext,
    platform: &GatewaySetupPlatform,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<bool, Box<dyn Error>> {
    match platform.key.as_str() {
        "irc" => {
            configure_irc_gateway_platform_with_io(context, input, output)?;
            Ok(true)
        }
        "teams" => {
            configure_teams_gateway_platform_with_io(context, input, output)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn configure_irc_gateway_platform_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "─── 💬 IRC Setup ───")?;
    if let Some(existing_server) = read_effective_env_value(context, "IRC_SERVER") {
        writeln!(output)?;
        writeln!(output, "IRC is already configured for {existing_server}.")?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure IRC?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    writeln!(
        output,
        "  Connect Hermes to Libera.Chat, OFTC, ZNC, InspIRCd, or another IRC network."
    )?;

    let server = prompt_gateway_required_line(
        input,
        output,
        "  IRC server hostname (e.g. irc.libera.chat)",
        "Server is required — skipping IRC setup.",
    )?;
    let Some(server) = server else {
        return Ok(());
    };
    save_env_value(context.env_path(), "IRC_SERVER", &server)?;

    let use_tls = prompt_gateway_yes_no(input, output, "Use TLS (recommended)?", true)?;
    save_env_value(
        context.env_path(),
        "IRC_USE_TLS",
        if use_tls { "true" } else { "false" },
    )?;

    let default_port = if use_tls { "6697" } else { "6667" };
    let port = prompt_gateway_line(
        input,
        output,
        format!("  Port (default {default_port})").as_str(),
    )?;
    let port = port.trim();
    if !port.is_empty() {
        if port.parse::<u16>().is_ok() {
            save_env_value(context.env_path(), "IRC_PORT", port)?;
        } else {
            writeln!(output, "  Invalid port — using default {default_port}.")?;
        }
    } else if read_effective_env_value(context, "IRC_PORT").is_some() {
        save_env_value(context.env_path(), "IRC_PORT", "")?;
    }

    let nickname = prompt_gateway_required_line(
        input,
        output,
        "  Bot nickname (e.g. hermes-bot)",
        "Nickname is required — skipping IRC setup.",
    )?;
    let Some(nickname) = nickname else {
        return Ok(());
    };
    save_env_value(context.env_path(), "IRC_NICKNAME", &nickname)?;

    let channel = prompt_gateway_required_line(
        input,
        output,
        "  Channel to join (e.g. #hermes, comma-separated for multiple)",
        "Channel is required — skipping IRC setup.",
    )?;
    let Some(channel) = channel else {
        return Ok(());
    };
    save_env_value(context.env_path(), "IRC_CHANNEL", &channel)?;

    writeln!(output)?;
    writeln!(output, "  Optional authentication. Leave blank to skip.")?;
    if prompt_gateway_yes_no(input, output, "Configure a server password?", false)? {
        let server_password = prompt_gateway_line(input, output, "  Server password")?;
        if !server_password.trim().is_empty() {
            save_env_value(
                context.env_path(),
                "IRC_SERVER_PASSWORD",
                server_password.trim(),
            )?;
        }
    }
    if prompt_gateway_yes_no(input, output, "Identify with NickServ on connect?", false)? {
        let nickserv = prompt_gateway_line(input, output, "  NickServ password")?;
        if !nickserv.trim().is_empty() {
            save_env_value(context.env_path(), "IRC_NICKSERV_PASSWORD", nickserv.trim())?;
        }
    }

    writeln!(output)?;
    writeln!(
        output,
        "  IRC nicks are not authenticated; restrict access for shared channels."
    )?;
    if prompt_gateway_yes_no(
        input,
        output,
        "Allow all users in the channel to talk to the bot?",
        false,
    )? {
        save_env_value(context.env_path(), "IRC_ALLOW_ALL_USERS", "true")?;
        save_env_value(context.env_path(), "IRC_ALLOWED_USERS", "")?;
        writeln!(output, "  Open access enabled for IRC.")?;
    } else {
        save_env_value(context.env_path(), "IRC_ALLOW_ALL_USERS", "false")?;
        let allowed = prompt_gateway_line(
            input,
            output,
            "  Allowed nicks (comma-separated, leave empty to deny everyone)",
        )?;
        save_env_value(
            context.env_path(),
            "IRC_ALLOWED_USERS",
            &allowed.replace(' ', ""),
        )?;
    }

    writeln!(output)?;
    writeln!(output, "💬 IRC configured!")?;
    Ok(())
}

fn configure_teams_gateway_platform_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "─── 💼 Microsoft Teams Setup ───")?;
    if let Some(existing_id) = read_effective_env_value(context, "TEAMS_CLIENT_ID") {
        writeln!(output)?;
        writeln!(
            output,
            "Teams is already configured for app ID {existing_id}."
        )?;
        if !prompt_gateway_yes_no(input, output, "Reconfigure Teams?", false)? {
            return Ok(());
        }
    }

    writeln!(output)?;
    writeln!(output, "  Install and log in with the Teams CLI first:")?;
    writeln!(output, "    npm install -g @microsoft/teams.cli@preview")?;
    writeln!(output, "    teams login")?;
    writeln!(output, "  Expose port 3978 publicly, then create your bot:")?;
    writeln!(
        output,
        "    teams app create --name \"Hermes\" --endpoint \"https://<tunnel>/api/messages\""
    )?;

    let client_id = prompt_gateway_required_line(
        input,
        output,
        "  Client ID",
        "Client ID is required — skipping Teams setup.",
    )?;
    let Some(client_id) = client_id else {
        return Ok(());
    };
    save_env_value(context.env_path(), "TEAMS_CLIENT_ID", &client_id)?;

    let client_secret = prompt_gateway_required_line(
        input,
        output,
        "  Client secret",
        "Client secret is required — skipping Teams setup.",
    )?;
    let Some(client_secret) = client_secret else {
        return Ok(());
    };
    save_env_value(context.env_path(), "TEAMS_CLIENT_SECRET", &client_secret)?;

    let tenant_id = prompt_gateway_required_line(
        input,
        output,
        "  Tenant ID",
        "Tenant ID is required — skipping Teams setup.",
    )?;
    let Some(tenant_id) = tenant_id else {
        return Ok(());
    };
    save_env_value(context.env_path(), "TEAMS_TENANT_ID", &tenant_id)?;

    writeln!(output)?;
    writeln!(
        output,
        "  To find your AAD object ID for the allowlist: teams status --verbose"
    )?;
    if prompt_gateway_yes_no(
        input,
        output,
        "Restrict access to specific users? (recommended)",
        true,
    )? {
        let allowed =
            prompt_gateway_line(input, output, "  Allowed AAD object IDs (comma-separated)")?;
        save_env_value(
            context.env_path(),
            "TEAMS_ALLOWED_USERS",
            &allowed.replace(' ', ""),
        )?;
        remove_env_key_if_present(&context.env_path(), "TEAMS_ALLOW_ALL_USERS")?;
    } else {
        save_env_value(context.env_path(), "TEAMS_ALLOW_ALL_USERS", "true")?;
        writeln!(output, "  Open access enabled for Teams.")?;
    }

    writeln!(output)?;
    writeln!(output, "💼 Microsoft Teams configured!")?;
    Ok(())
}

fn prompt_gateway_required_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    prompt: &str,
    missing_message: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let value = prompt_gateway_line(input, output, prompt)?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        writeln!(output, "  {missing_message}")?;
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
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

fn read_gateway_platform_extra_value(
    context: &HermesContext,
    platform: &str,
    key: &str,
) -> Option<String> {
    let root = read_raw_yaml_mapping(&context.config_path()).ok()?;
    root.get(YamlValue::String(String::from("platforms")))
        .and_then(YamlValue::as_mapping)?
        .get(YamlValue::String(platform.to_string()))
        .and_then(YamlValue::as_mapping)?
        .get(YamlValue::String(String::from("extra")))
        .and_then(YamlValue::as_mapping)?
        .get(YamlValue::String(key.to_string()))
        .and_then(yaml_scalar_string)
        .filter(|value| !value.trim().is_empty())
}

fn yaml_scalar_string(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(text) => Some(text.trim().to_string()),
        YamlValue::Number(number) => Some(number.to_string()),
        YamlValue::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
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

const GATEWAY_PLUGIN_SETUP_METADATA_BOOTSTRAP: &str = concat!(
    "import json\n",
    "from hermes_cli.gateway import _platform_status\n",
    "from hermes_cli.plugins import discover_plugins\n",
    "discover_plugins()\n",
    "from gateway.platform_registry import platform_registry\n",
    "items = []\n",
    "for entry in platform_registry.plugin_entries():\n",
    "    platform = {\n",
    "        'key': entry.name,\n",
    "        'label': entry.label,\n",
    "        'emoji': entry.emoji,\n",
    "        'token_var': entry.required_env[0] if entry.required_env else '',\n",
    "        'install_hint': entry.install_hint,\n",
    "        '_registry_entry': entry,\n",
    "    }\n",
    "    items.append({\n",
    "        'key': entry.name,\n",
    "        'label': entry.label,\n",
    "        'emoji': entry.emoji,\n",
    "        'status': _platform_status(platform),\n",
    "        'token_var': platform['token_var'],\n",
    "        'install_hint': entry.install_hint,\n",
    "        'setup_instructions': [],\n",
    "        'required_env': list(getattr(entry, 'required_env', []) or []),\n",
    "        'has_builtin_setup': False,\n",
    "        'has_plugin_setup': bool(getattr(entry, 'setup_fn', None) is not None),\n",
    "        'vars': [],\n",
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

fn known_profile_gateway_processes(
    context: &HermesContext,
) -> Result<Vec<GatewayProcess>, Box<dyn Error>> {
    let mut processes = Vec::new();
    for (_, home) in collect_profile_homes(context)? {
        for pid in gateway_pids_for_profile(&home) {
            if processes
                .iter()
                .any(|process: &GatewayProcess| process.pid == pid)
            {
                continue;
            }
            processes.push(GatewayProcess {
                pid,
                command: String::new(),
            });
        }
    }
    Ok(processes)
}

#[cfg(all(test, not(windows)))]
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

#[cfg(all(test, not(windows)))]
fn gateway_process_matches(pid: i64, command: &str, exclude_pids: &[i64]) -> bool {
    pid > 0
        && !exclude_pids.contains(&pid)
        && GATEWAY_PROCESS_PATTERNS
            .iter()
            .any(|pattern| command.contains(pattern))
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
    use std::io::{Cursor, Read, Write};
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

    fn test_json_body_string(request: &[u8], key: &str) -> String {
        let request = String::from_utf8_lossy(request);
        let body = request.split("\r\n\r\n").nth(1).unwrap_or("").trim();
        let value: JsonValue = serde_json::from_str(body).unwrap();
        value
            .get(key)
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string()
    }

    fn qqbot_encrypt_secret_for_test(secret: &str, key_base64: &str) -> String {
        let key = BASE64_STANDARD.decode(key_base64).unwrap();
        let nonce = [7_u8; 12];
        let unbound = ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, &key).unwrap();
        let key = ring::aead::LessSafeKey::new(unbound);
        let mut in_out = secret.as_bytes().to_vec();
        key.seal_in_place_append_tag(
            ring::aead::Nonce::assume_unique_for_key(nonce),
            ring::aead::Aad::empty(),
            &mut in_out,
        )
        .unwrap();
        let mut raw = nonce.to_vec();
        raw.extend_from_slice(&in_out);
        BASE64_STANDARD.encode(raw)
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
        let context = HermesContext::new(temp.path());
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
            &context,
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
    fn gateway_setup_metadata_uses_native_standard_builtins_and_plugin_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_home, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(
            context.config_path(),
            "plugins:\n  enabled:\n    - custom-platform\n",
        )
        .unwrap();
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
[{{\"key\":\"customchat\",\"label\":\"Custom Chat\",\"emoji\":\"#\",\"status\":\"not configured\",\"token_var\":\"CUSTOM_TOKEN\",\"required_env\":[\"CUSTOM_TOKEN\"],\"has_plugin_setup\":true}}]\n\
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
        let metadata = load_gateway_setup_metadata(&context, true).unwrap();

        let telegram = metadata
            .iter()
            .find(|platform| platform.key == "telegram")
            .unwrap();
        assert_eq!(telegram.label, "Telegram");
        assert!(gateway_platform_uses_native_standard_setup(telegram));

        let discord = metadata
            .iter()
            .find(|platform| platform.key == "discord")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(discord));

        let mattermost = metadata
            .iter()
            .find(|platform| platform.key == "mattermost")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(mattermost));

        let slack = metadata
            .iter()
            .find(|platform| platform.key == "slack")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(slack));

        let bluebubbles = metadata
            .iter()
            .find(|platform| platform.key == "bluebubbles")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(bluebubbles));

        let homeassistant = metadata
            .iter()
            .find(|platform| platform.key == "homeassistant")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(homeassistant));

        let webhook = metadata
            .iter()
            .find(|platform| platform.key == "webhook")
            .unwrap();
        assert!(gateway_platform_uses_native_standard_setup(webhook));

        let whatsapp = metadata
            .iter()
            .find(|platform| platform.key == "whatsapp")
            .unwrap();
        assert!(!whatsapp.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(whatsapp));

        let matrix = metadata
            .iter()
            .find(|platform| platform.key == "matrix")
            .unwrap();
        assert!(!matrix.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(matrix));

        let signal = metadata
            .iter()
            .find(|platform| platform.key == "signal")
            .unwrap();
        assert!(!signal.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(signal));

        let dingtalk = metadata
            .iter()
            .find(|platform| platform.key == "dingtalk")
            .unwrap();
        assert!(!dingtalk.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(dingtalk));

        let feishu = metadata
            .iter()
            .find(|platform| platform.key == "feishu")
            .unwrap();
        assert!(!feishu.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(feishu));

        let wecom = metadata
            .iter()
            .find(|platform| platform.key == "wecom")
            .unwrap();
        assert!(!wecom.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(wecom));

        let weixin = metadata
            .iter()
            .find(|platform| platform.key == "weixin")
            .unwrap();
        assert!(!weixin.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(weixin));

        let qqbot = metadata
            .iter()
            .find(|platform| platform.key == "qqbot")
            .unwrap();
        assert!(!qqbot.has_builtin_setup);
        assert!(!gateway_platform_uses_native_standard_setup(qqbot));

        let email = metadata
            .iter()
            .find(|platform| platform.key == "email")
            .unwrap();
        assert_eq!(email.status, "not configured");
        assert!(gateway_platform_uses_native_standard_setup(email));

        let api_server = metadata
            .iter()
            .find(|platform| platform.key == "api_server")
            .unwrap();
        assert_eq!(api_server.status, "not configured");
        assert!(gateway_platform_uses_native_standard_setup(api_server));

        let irc = metadata
            .iter()
            .find(|platform| platform.key == "irc")
            .unwrap();
        assert_eq!(
            irc.required_env,
            vec!["IRC_SERVER", "IRC_CHANNEL", "IRC_NICKNAME"]
        );
        assert!(irc.has_plugin_setup);
        let custom = metadata
            .iter()
            .find(|platform| platform.key == "customchat")
            .unwrap();
        assert_eq!(custom.required_env, vec!["CUSTOM_TOKEN"]);
        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("metadata accept=1 platform="));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn gateway_setup_metadata_skips_plugin_bridge_without_enabled_user_plugins() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_home, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'called\\n' >> '{}'\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        let metadata = load_gateway_setup_metadata(&context, true).unwrap();

        assert!(metadata.iter().any(|platform| platform.key == "irc"));
        assert!(metadata.iter().any(|platform| platform.key == "teams"));
        assert!(!log.exists());

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    fn gateway_setup_metadata_reports_native_status() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        for key in [
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
            "HASS_TOKEN",
            "HASS_URL",
            "IRC_SERVER",
            "IRC_CHANNEL",
            "TEAMS_CLIENT_ID",
            "TEAMS_CLIENT_SECRET",
            "TEAMS_TENANT_ID",
            "API_SERVER_ENABLED",
            "API_SERVER_HOST",
            "API_SERVER_PORT",
            "API_SERVER_KEY",
            "API_SERVER_CORS_ORIGINS",
            "API_SERVER_MODEL_NAME",
            "BLUEBUBBLES_SERVER_URL",
            "BLUEBUBBLES_PASSWORD",
            "MATRIX_ACCESS_TOKEN",
            "MATRIX_PASSWORD",
            "MATRIX_HOMESERVER",
            "MATRIX_USER_ID",
            "MATRIX_ENCRYPTION",
            "MATRIX_ALLOWED_USERS",
            "MATRIX_HOME_ROOM",
            "SIGNAL_HTTP_URL",
            "SIGNAL_ACCOUNT",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "DINGTALK_CLIENT_ID",
            "DINGTALK_CLIENT_SECRET",
            "DINGTALK_ALLOW_ALL_USERS",
            "WHATSAPP_MODE",
            "WHATSAPP_ENABLED",
            "WHATSAPP_ALLOWED_USERS",
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_DOMAIN",
            "FEISHU_CONNECTION_MODE",
            "FEISHU_ALLOW_ALL_USERS",
            "FEISHU_ALLOWED_USERS",
            "FEISHU_GROUP_POLICY",
            "FEISHU_HOME_CHANNEL",
            "QQ_APP_ID",
            "QQ_CLIENT_SECRET",
            "QQ_ALLOW_ALL_USERS",
            "QQ_ALLOWED_USERS",
            "QQBOT_HOME_CHANNEL",
            "WECOM_BOT_ID",
            "WECOM_SECRET",
            "WECOM_ALLOWED_USERS",
            "WECOM_DM_POLICY",
            "WECOM_HOME_CHANNEL",
            "WEBHOOK_ENABLED",
            "WEBHOOK_PORT",
            "WEBHOOK_SECRET",
            "WEIXIN_ACCOUNT_ID",
            "WEIXIN_TOKEN",
            "WEIXIN_BASE_URL",
            "WEIXIN_CDN_BASE_URL",
            "WEIXIN_DM_POLICY",
            "WEIXIN_ALLOW_ALL_USERS",
            "WEIXIN_ALLOWED_USERS",
            "WEIXIN_GROUP_POLICY",
            "WEIXIN_GROUP_ALLOWED_USERS",
            "WEIXIN_HOME_CHANNEL",
            "YUANBAO_APP_ID",
            "YUANBAO_APP_SECRET",
            "YUANBAO_BOT_ID",
            "YUANBAO_HOME_CHANNEL",
            "YUANBAO_API_DOMAIN",
            "YUANBAO_WS_URL",
        ] {
            remove_env_var(key);
        }
        fs::create_dir_all(context.hermes_home().join("whatsapp").join("session")).unwrap();
        fs::write(
            context.hermes_home().join("whatsapp/session/creds.json"),
            "{}",
        )
        .unwrap();
        save_env_value(context.env_path(), "EMAIL_ADDRESS", "bot@example.com").unwrap();
        save_env_value(context.env_path(), "EMAIL_PASSWORD", "app-password").unwrap();
        save_env_value(context.env_path(), "EMAIL_IMAP_HOST", "imap.example.com").unwrap();
        save_env_value(context.env_path(), "EMAIL_SMTP_HOST", "smtp.example.com").unwrap();
        save_env_value(context.env_path(), "HASS_TOKEN", "ha-token").unwrap();
        save_env_value(context.env_path(), "WHATSAPP_ENABLED", "true").unwrap();
        save_env_value(context.env_path(), "WEBHOOK_ENABLED", "true").unwrap();
        save_env_value(context.env_path(), "API_SERVER_ENABLED", "true").unwrap();
        save_env_value(context.env_path(), "YUANBAO_APP_ID", "yb-app").unwrap();
        save_env_value(context.env_path(), "YUANBAO_APP_SECRET", "yb-secret").unwrap();
        fs::write(
            context.config_path(),
            "platforms:\n  irc:\n    extra:\n      server: irc.libera.chat\n      channel: '#hermes'\n",
        )
        .unwrap();
        save_env_value(context.env_path(), "TEAMS_CLIENT_ID", "teams-client").unwrap();

        let metadata = native_gateway_setup_metadata(&context);
        let email = metadata
            .iter()
            .find(|platform| platform.key == "email")
            .unwrap();
        assert_eq!(email.status, "configured");
        let whatsapp = metadata
            .iter()
            .find(|platform| platform.key == "whatsapp")
            .unwrap();
        assert_eq!(whatsapp.status, "configured + paired");
        let irc = metadata
            .iter()
            .find(|platform| platform.key == "irc")
            .unwrap();
        assert_eq!(irc.status, "configured");
        let teams = metadata
            .iter()
            .find(|platform| platform.key == "teams")
            .unwrap();
        assert_eq!(teams.status, "partially configured");
        let bluebubbles = metadata
            .iter()
            .find(|platform| platform.key == "bluebubbles")
            .unwrap();
        assert_eq!(bluebubbles.status, "not configured");
        let signal = metadata
            .iter()
            .find(|platform| platform.key == "signal")
            .unwrap();
        assert_eq!(signal.status, "not configured");
        let homeassistant = metadata
            .iter()
            .find(|platform| platform.key == "homeassistant")
            .unwrap();
        assert_eq!(homeassistant.status, "configured");
        let dingtalk = metadata
            .iter()
            .find(|platform| platform.key == "dingtalk")
            .unwrap();
        assert_eq!(dingtalk.status, "not configured");
        let feishu = metadata
            .iter()
            .find(|platform| platform.key == "feishu")
            .unwrap();
        assert_eq!(feishu.status, "not configured");
        let qqbot = metadata
            .iter()
            .find(|platform| platform.key == "qqbot")
            .unwrap();
        assert_eq!(qqbot.status, "not configured");
        let wecom = metadata
            .iter()
            .find(|platform| platform.key == "wecom")
            .unwrap();
        assert_eq!(wecom.status, "not configured");
        let weixin = metadata
            .iter()
            .find(|platform| platform.key == "weixin")
            .unwrap();
        assert_eq!(weixin.status, "not configured");
        let webhook = metadata
            .iter()
            .find(|platform| platform.key == "webhook")
            .unwrap();
        assert_eq!(webhook.status, "configured");
        let yuanbao = metadata
            .iter()
            .find(|platform| platform.key == "yuanbao")
            .unwrap();
        assert_eq!(yuanbao.status, "configured");
        let api_server = metadata
            .iter()
            .find(|platform| platform.key == "api_server")
            .unwrap();
        assert_eq!(api_server.status, "configured");
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
        run_gateway_platform_setup_bridge(true, "legacy-bridge-platform").unwrap();

        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("platform accept=1 key=legacy-bridge-platform"));

        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn configure_whatsapp_gateway_platform_uses_native_setup_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "WHATSAPP_MODE",
            "WHATSAPP_ENABLED",
            "WHATSAPP_ALLOWED_USERS",
            "HERMES_WHATSAPP_BRIDGE_DIR",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let bridge_dir = temp.path().join("wa-bridge");
        let fake_bin = temp.path().join("bin");
        fs::create_dir_all(&bridge_dir).unwrap();
        fs::create_dir_all(&fake_bin).unwrap();
        fs::write(bridge_dir.join("bridge.js"), "console.log('bridge');\n").unwrap();

        let command_log = temp.path().join("commands.log");
        let npm = fake_bin.join("npm");
        fs::write(
            &npm,
            format!(
                "#!/bin/sh\nprintf 'npm %s\\n' \"$*\" >> '{}'\nmkdir -p node_modules\n",
                command_log.display()
            ),
        )
        .unwrap();
        let node = fake_bin.join("node");
        fs::write(
            &node,
            format!(
                "#!/bin/sh\nprintf 'node %s\\n' \"$*\" >> '{}'\nSESSION=''\nPREV=''\nfor ARG in \"$@\"; do\n  if [ \"$PREV\" = '1' ]; then SESSION=\"$ARG\"; PREV=''; continue; fi\n  if [ \"$ARG\" = '--session' ]; then PREV='1'; fi\n done\nmkdir -p \"$SESSION\"\nprintf '{{}}' > \"$SESSION/creds.json\"\n",
                command_log.display()
            ),
        )
        .unwrap();
        for path in [&npm, &node] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let fake_python = temp.path().join("python3");
        let python_log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", python_log.display()),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let original_path = env::var_os("PATH");
        set_env_var(
            "PATH",
            format!(
                "{}:{}",
                fake_bin.display(),
                env::var("PATH").unwrap_or_default()
            ),
        );
        set_env_var("HERMES_WHATSAPP_BRIDGE_DIR", &bridge_dir);
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "whatsapp")
            .unwrap();
        let mut input = Cursor::new("1\n15551234567\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("WHATSAPP_MODE=bot"));
        assert!(env_text.contains("WHATSAPP_ENABLED=true"));
        assert!(env_text.contains("WHATSAPP_ALLOWED_USERS=15551234567"));
        assert!(
            context
                .hermes_home()
                .join("whatsapp")
                .join("session")
                .join("creds.json")
                .exists()
        );
        let commands = fs::read_to_string(command_log).unwrap();
        assert!(commands.contains("npm install --no-fund --no-audit --progress=false"));
        assert!(commands.contains("node"));
        assert!(!python_log.exists());

        match original_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        remove_env_var("HERMES_WHATSAPP_BRIDGE_DIR");
        remove_env_var("HERMES_GATEWAY_PYTHON");
        remove_env_var("HERMES_GATEWAY_SETUP_PLATFORM");
    }

    #[test]
    #[cfg(unix)]
    fn configure_matrix_gateway_platform_uses_native_setup_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
printf 'argv=%s platform=%s\\n' \"$*\" \"$HERMES_GATEWAY_SETUP_PLATFORM\" >> '{}'\n\
if [ \"$1\" = \"-c\" ]; then\n\
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
        remove_env_var("HERMES_GATEWAY_SETUP_PLATFORM");
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "matrix")
            .unwrap();
        let mut input = Cursor::new(
            "https://matrix.example.org/\n\n@bot:example.org\nmatrix-pass\ny\n@alice:example.org\n!home:example.org\n",
        );
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("MATRIX_HOMESERVER=https://matrix.example.org"));
        assert!(env_text.contains("MATRIX_USER_ID=@bot:example.org"));
        assert!(env_text.contains("MATRIX_PASSWORD=matrix-pass"));
        assert!(env_text.contains("MATRIX_ENCRYPTION=true"));
        assert!(env_text.contains("MATRIX_ALLOWED_USERS=@alice:example.org"));
        assert!(env_text.contains("MATRIX_HOME_ROOM=!home:example.org"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "matrix")
            .unwrap()
            .status;
        assert_eq!(status, "configured + E2EE");

        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("argv=-c"));
        assert!(log_text.contains("platform="));
        assert!(!log_text.contains("platform=matrix"));
        assert!(!log_text.contains("-m pip"));

        remove_env_var("HERMES_GATEWAY_PYTHON");
        remove_env_var("HERMES_GATEWAY_SETUP_PLATFORM");
    }

    #[test]
    #[cfg(unix)]
    fn configure_signal_gateway_platform_uses_native_setup_without_bridge() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "SIGNAL_HTTP_URL",
            "SIGNAL_ACCOUNT",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_bin = temp.path().join("bin");
        fs::create_dir_all(&fake_bin).unwrap();
        fs::write(fake_bin.join("signal-cli"), "#!/bin/sh\nexit 0\n").unwrap();
        let original_path = env::var_os("PATH");
        set_env_var("PATH", &fake_bin);

        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "signal")
            .unwrap();
        let mut input = Cursor::new(format!(
            "http://{addr}/\n+15551234567\n+15551234567, +15557654321\ny\ngroup-a, group-b\n"
        ));
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains(&format!("SIGNAL_HTTP_URL=http://{addr}")));
        assert!(env_text.contains("SIGNAL_ACCOUNT=+15551234567"));
        assert!(env_text.contains("SIGNAL_ALLOWED_USERS=+15551234567,+15557654321"));
        assert!(env_text.contains("SIGNAL_GROUP_ALLOWED_USERS=group-a,group-b"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "signal")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        if let Some(path) = original_path {
            set_env_var("PATH", path);
        } else {
            remove_env_var("PATH");
        }
        remove_env_var("HERMES_GATEWAY_PYTHON");
        remove_env_var("HERMES_GATEWAY_SETUP_PLATFORM");
    }

    #[test]
    #[cfg(unix)]
    fn configure_dingtalk_gateway_platform_uses_native_device_flow_without_bridge() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "DINGTALK_CLIENT_ID",
            "DINGTALK_CLIENT_SECRET",
            "DINGTALK_ALLOW_ALL_USERS",
            "DINGTALK_REGISTRATION_BASE_URL",
            "DINGTALK_REGISTRATION_SOURCE",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("DINGTALK_REGISTRATION_BASE_URL", format!("http://{addr}"));
        let server = std::thread::spawn(move || {
            let responses = [
                r#"{"errcode":0,"nonce":"nonce-1"}"#,
                r#"{"errcode":0,"device_code":"device-1","verification_uri_complete":"https://example.com/verify","expires_in":5,"interval":1}"#,
                r#"{"errcode":0,"status":"SUCCESS","client_id":"ding-client","client_secret":"ding-secret"}"#,
            ];
            for body in responses {
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut request = [0_u8; 2048];
                    let _ = stream.read(&mut request);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "dingtalk")
            .unwrap();
        let mut input = Cursor::new("1\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("DINGTALK_CLIENT_ID=ding-client"));
        assert!(env_text.contains("DINGTALK_CLIENT_SECRET=ding-secret"));
        assert!(env_text.contains("DINGTALK_ALLOW_ALL_USERS=true"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "dingtalk")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        remove_env_var("DINGTALK_REGISTRATION_BASE_URL");
        remove_env_var("DINGTALK_REGISTRATION_SOURCE");
        remove_env_var("HERMES_GATEWAY_PYTHON");
        remove_env_var("HERMES_GATEWAY_SETUP_PLATFORM");
    }

    #[test]
    #[cfg(unix)]
    fn configure_feishu_gateway_platform_uses_native_qr_flow_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_DOMAIN",
            "FEISHU_CONNECTION_MODE",
            "FEISHU_ALLOW_ALL_USERS",
            "FEISHU_ALLOWED_USERS",
            "FEISHU_GROUP_POLICY",
            "FEISHU_HOME_CHANNEL",
            "FEISHU_REGISTRATION_BASE_URL",
            "FEISHU_OPEN_BASE_URL",
            "FEISHU_REGISTRATION_POLL_INTERVAL_MS",
            "FEISHU_REGISTRATION_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("FEISHU_REGISTRATION_BASE_URL", format!("http://{addr}"));
        set_env_var("FEISHU_OPEN_BASE_URL", format!("http://{addr}"));
        set_env_var("FEISHU_REGISTRATION_POLL_INTERVAL_MS", "1");
        set_env_var("FEISHU_REGISTRATION_TIMEOUT_MS", "1000");
        let server = std::thread::spawn(move || {
            let responses = [
                r#"{"supported_auth_methods":["client_secret"]}"#,
                r#"{"device_code":"device-1","verification_uri_complete":"https://example.com/verify","user_code":"USER","interval":1,"expire_in":5}"#,
                r#"{"client_id":"feishu-app","client_secret":"feishu-secret","user_info":{"tenant_brand":"feishu","open_id":"ou-user"}}"#,
                r#"{"code":0,"tenant_access_token":"tenant-token"}"#,
                r#"{"code":0,"bot":{"app_name":"HermesBot","open_id":"ou-bot"}}"#,
            ];
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut next = 0;
            while next < responses.len() && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0_u8; 4096];
                        let _ = stream.read(&mut request);
                        let body = responses[next];
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        next += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "feishu")
            .unwrap();
        let mut input = Cursor::new("1\n3\nou-user, ou-other\n1\nhome-chat\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("FEISHU_APP_ID=feishu-app"));
        assert!(env_text.contains("FEISHU_APP_SECRET=feishu-secret"));
        assert!(env_text.contains("FEISHU_DOMAIN=feishu"));
        assert!(env_text.contains("FEISHU_CONNECTION_MODE=websocket"));
        assert!(env_text.contains("FEISHU_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("FEISHU_ALLOWED_USERS=ou-user,ou-other"));
        assert!(env_text.contains("FEISHU_GROUP_POLICY=open"));
        assert!(env_text.contains("FEISHU_HOME_CHANNEL=home-chat"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "feishu")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        for key in [
            "FEISHU_REGISTRATION_BASE_URL",
            "FEISHU_OPEN_BASE_URL",
            "FEISHU_REGISTRATION_POLL_INTERVAL_MS",
            "FEISHU_REGISTRATION_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }
    }

    #[test]
    fn configure_feishu_gateway_platform_writes_manual_credentials_and_policies() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_DOMAIN",
            "FEISHU_CONNECTION_MODE",
            "FEISHU_ALLOW_ALL_USERS",
            "FEISHU_ALLOWED_USERS",
            "FEISHU_GROUP_POLICY",
            "FEISHU_HOME_CHANNEL",
            "FEISHU_OPEN_BASE_URL",
        ] {
            remove_env_var(key);
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("FEISHU_OPEN_BASE_URL", format!("http://{addr}"));
        let server = std::thread::spawn(move || {
            let responses = [
                r#"{"code":0,"tenant_access_token":"tenant-token"}"#,
                r#"{"code":0,"bot":{"app_name":"ManualBot","open_id":"ou-bot"}}"#,
            ];
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut next = 0;
            while next < responses.len() && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0_u8; 4096];
                        let _ = stream.read(&mut request);
                        let body = responses[next];
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        next += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "feishu")
            .unwrap();
        let mut input = Cursor::new("2\nmanual-app\nmanual-secret\n2\n2\n1\n2\nhome-chat\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("FEISHU_APP_ID=manual-app"));
        assert!(env_text.contains("FEISHU_APP_SECRET=manual-secret"));
        assert!(env_text.contains("FEISHU_DOMAIN=lark"));
        assert!(env_text.contains("FEISHU_CONNECTION_MODE=webhook"));
        assert!(env_text.contains("FEISHU_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("FEISHU_ALLOWED_USERS="));
        assert!(env_text.contains("FEISHU_GROUP_POLICY=disabled"));
        assert!(env_text.contains("FEISHU_HOME_CHANNEL=home-chat"));

        remove_env_var("FEISHU_OPEN_BASE_URL");
    }

    #[test]
    #[cfg(unix)]
    fn configure_weixin_gateway_platform_uses_native_qr_flow_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "WEIXIN_ACCOUNT_ID",
            "WEIXIN_TOKEN",
            "WEIXIN_BASE_URL",
            "WEIXIN_CDN_BASE_URL",
            "WEIXIN_DM_POLICY",
            "WEIXIN_ALLOW_ALL_USERS",
            "WEIXIN_ALLOWED_USERS",
            "WEIXIN_GROUP_POLICY",
            "WEIXIN_GROUP_ALLOWED_USERS",
            "WEIXIN_HOME_CHANNEL",
            "WEIXIN_ILINK_BASE_URL",
            "WEIXIN_QR_POLL_INTERVAL_MS",
            "WEIXIN_QR_LOGIN_TIMEOUT_MS",
            "WEIXIN_QR_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("WEIXIN_ILINK_BASE_URL", format!("http://{addr}"));
        set_env_var("WEIXIN_QR_POLL_INTERVAL_MS", "1");
        set_env_var("WEIXIN_QR_LOGIN_TIMEOUT_MS", "1000");
        set_env_var("WEIXIN_QR_TIMEOUT_MS", "1000");
        let server = std::thread::spawn(move || {
            let responses = [
                r#"{"qrcode":"qr-1","qrcode_img_content":"http://scan.example/qr-1"}"#,
                r#"{"status":"scaned"}"#,
                r#"{"status":"confirmed","ilink_bot_id":"bot-account","bot_token":"bot-token","baseurl":"http://ilink.example.com","ilink_user_id":"wxid-user"}"#,
            ];
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut next = 0;
            while next < responses.len() && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0_u8; 4096];
                        let _ = stream.read(&mut request);
                        let body = responses[next];
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        next += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "weixin")
            .unwrap();
        let mut input = Cursor::new("\n3\n\n3\nroom-a, room-b\n\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("WEIXIN_ACCOUNT_ID=bot-account"));
        assert!(env_text.contains("WEIXIN_TOKEN=bot-token"));
        assert!(env_text.contains("WEIXIN_BASE_URL=http://ilink.example.com"));
        assert!(env_text.contains("WEIXIN_CDN_BASE_URL=https://novac2c.cdn.weixin.qq.com/c2c"));
        assert!(env_text.contains("WEIXIN_DM_POLICY=allowlist"));
        assert!(env_text.contains("WEIXIN_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("WEIXIN_ALLOWED_USERS=wxid-user"));
        assert!(env_text.contains("WEIXIN_GROUP_POLICY=allowlist"));
        assert!(env_text.contains("WEIXIN_GROUP_ALLOWED_USERS=room-a,room-b"));
        assert!(env_text.contains("WEIXIN_HOME_CHANNEL=wxid-user"));

        let account_path = context
            .hermes_home()
            .join("weixin")
            .join("accounts")
            .join("bot-account.json");
        let account =
            serde_json::from_str::<JsonValue>(&fs::read_to_string(account_path).unwrap()).unwrap();
        assert_eq!(
            account.get("token").and_then(JsonValue::as_str),
            Some("bot-token")
        );
        assert_eq!(
            account.get("base_url").and_then(JsonValue::as_str),
            Some("http://ilink.example.com")
        );
        assert_eq!(
            account.get("user_id").and_then(JsonValue::as_str),
            Some("wxid-user")
        );
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "weixin")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        for key in [
            "WEIXIN_ILINK_BASE_URL",
            "WEIXIN_QR_POLL_INTERVAL_MS",
            "WEIXIN_QR_LOGIN_TIMEOUT_MS",
            "WEIXIN_QR_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }
    }

    #[test]
    #[cfg(unix)]
    fn configure_qqbot_gateway_platform_uses_native_qr_flow_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "QQ_APP_ID",
            "QQ_CLIENT_SECRET",
            "QQ_ALLOW_ALL_USERS",
            "QQ_ALLOWED_USERS",
            "QQBOT_HOME_CHANNEL",
            "QQ_PORTAL_BASE_URL",
            "QQBOT_QR_URL_TEMPLATE",
            "QQBOT_ONBOARD_POLL_INTERVAL_MS",
            "QQBOT_ONBOARD_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("QQ_PORTAL_BASE_URL", format!("http://{addr}"));
        set_env_var(
            "QQBOT_QR_URL_TEMPLATE",
            format!("http://{addr}/connect?task_id={{task_id}}"),
        );
        set_env_var("QQBOT_ONBOARD_POLL_INTERVAL_MS", "1");
        set_env_var("QQBOT_ONBOARD_TIMEOUT_MS", "1000");
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut next = 0;
            let mut bind_key = String::new();
            while next < 2 && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0_u8; 4096];
                        let bytes = stream.read(&mut request).unwrap_or(0);
                        let body = if next == 0 {
                            bind_key = test_json_body_string(&request[..bytes], "key");
                            String::from(r#"{"retcode":0,"data":{"task_id":"task-1"}}"#)
                        } else {
                            let encrypted = qqbot_encrypt_secret_for_test("qq-secret", &bind_key);
                            format!(
                                r#"{{"retcode":0,"data":{{"status":2,"bot_appid":"qq-app","bot_encrypt_secret":"{}","user_openid":"user-open"}}}}"#,
                                encrypted
                            )
                        };
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        next += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "qqbot")
            .unwrap();
        let mut input = Cursor::new("1\n1\n\n\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("QQ_APP_ID=qq-app"));
        assert!(env_text.contains("QQ_CLIENT_SECRET=qq-secret"));
        assert!(env_text.contains("QQ_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("QQ_ALLOWED_USERS=user-open"));
        assert!(env_text.contains("QQBOT_HOME_CHANNEL=user-open"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "qqbot")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        for key in [
            "QQ_PORTAL_BASE_URL",
            "QQBOT_QR_URL_TEMPLATE",
            "QQBOT_ONBOARD_POLL_INTERVAL_MS",
            "QQBOT_ONBOARD_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }
    }

    #[test]
    fn configure_qqbot_gateway_platform_writes_manual_credentials_and_allowlist() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "QQ_APP_ID",
            "QQ_CLIENT_SECRET",
            "QQ_ALLOW_ALL_USERS",
            "QQ_ALLOWED_USERS",
            "QQBOT_HOME_CHANNEL",
            "QQ_PORTAL_BASE_URL",
        ] {
            remove_env_var(key);
        }

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "qqbot")
            .unwrap();
        let mut input =
            Cursor::new("2\nmanual-app\nmanual-secret\n3\nuser-a, user-b\nhome-openid\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("QQ_APP_ID=manual-app"));
        assert!(env_text.contains("QQ_CLIENT_SECRET=manual-secret"));
        assert!(env_text.contains("QQ_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("QQ_ALLOWED_USERS=user-a,user-b"));
        assert!(env_text.contains("QQBOT_HOME_CHANNEL=home-openid"));
    }

    #[test]
    #[cfg(unix)]
    fn configure_wecom_gateway_platform_uses_native_qr_flow_without_bridge() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "WECOM_BOT_ID",
            "WECOM_SECRET",
            "WECOM_ALLOWED_USERS",
            "WECOM_DM_POLICY",
            "WECOM_HOME_CHANNEL",
            "GATEWAY_ALLOW_ALL_USERS",
            "WECOM_QR_GENERATE_URL",
            "WECOM_QR_QUERY_URL",
            "WECOM_QR_CODE_PAGE",
            "WECOM_QR_POLL_INTERVAL_MS",
            "WECOM_QR_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!("#!/bin/sh\nprintf called >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        set_env_var("WECOM_QR_GENERATE_URL", format!("http://{addr}/generate"));
        set_env_var("WECOM_QR_QUERY_URL", format!("http://{addr}/query_result"));
        set_env_var(
            "WECOM_QR_CODE_PAGE",
            format!("http://{addr}/gen?source=hermes&scode="),
        );
        set_env_var("WECOM_QR_POLL_INTERVAL_MS", "1");
        set_env_var("WECOM_QR_TIMEOUT_MS", "1000");
        let server = std::thread::spawn(move || {
            let responses = [
                r#"{"data":{"scode":"scode-1","auth_url":"https://example.com/auth"}}"#,
                r#"{"data":{"status":"success","bot_info":{"botid":"wecom-bot","secret":"wecom-secret"}}}"#,
            ];
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut next = 0;
            while next < responses.len() && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0_u8; 2048];
                        let _ = stream.read(&mut request);
                        let body = responses[next];
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        next += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "wecom")
            .unwrap();
        let mut input = Cursor::new("1\nuser-a, user-b\nchat-1\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );
        server.join().unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("WECOM_BOT_ID=wecom-bot"));
        assert!(env_text.contains("WECOM_SECRET=wecom-secret"));
        assert!(env_text.contains("WECOM_ALLOWED_USERS=user-a,user-b"));
        assert!(env_text.contains("WECOM_HOME_CHANNEL=chat-1"));
        assert!(!env_text.contains("GATEWAY_ALLOW_ALL_USERS=true"));
        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "wecom")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
        assert!(!log.exists());

        for key in [
            "WECOM_QR_GENERATE_URL",
            "WECOM_QR_QUERY_URL",
            "WECOM_QR_CODE_PAGE",
            "WECOM_QR_POLL_INTERVAL_MS",
            "WECOM_QR_TIMEOUT_MS",
            "HERMES_GATEWAY_PYTHON",
            "HERMES_GATEWAY_SETUP_PLATFORM",
        ] {
            remove_env_var(key);
        }
    }

    #[test]
    fn configure_wecom_gateway_platform_writes_manual_credentials_and_pairing_policy() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        for key in [
            "WECOM_BOT_ID",
            "WECOM_SECRET",
            "WECOM_ALLOWED_USERS",
            "WECOM_DM_POLICY",
            "WECOM_HOME_CHANNEL",
            "GATEWAY_ALLOW_ALL_USERS",
        ] {
            remove_env_var(key);
        }

        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "wecom")
            .unwrap();
        let mut input = Cursor::new("2\nbot-id\nsecret\n\n2\nhome-chat\n");
        let mut output = Vec::new();

        assert!(
            configure_native_gateway_builtin_platform_with_io(
                &context,
                &platform,
                &mut input,
                &mut output,
            )
            .unwrap()
        );

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("WECOM_BOT_ID=bot-id"));
        assert!(env_text.contains("WECOM_SECRET=secret"));
        assert!(env_text.contains("WECOM_DM_POLICY=pairing"));
        assert!(env_text.contains("WECOM_HOME_CHANNEL=home-chat"));
        assert!(!env_text.contains("GATEWAY_ALLOW_ALL_USERS=true"));
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
    fn configure_telegram_gateway_platform_validates_and_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "telegram")
            .unwrap();

        let mut input =
            Cursor::new("bad-token\n123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcd\n111, 222\n111\n");
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Invalid token format"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TELEGRAM_BOT_TOKEN=123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcd"));
        assert!(env_text.contains("TELEGRAM_ALLOWED_USERS=111,222"));
        assert!(env_text.contains("TELEGRAM_HOME_CHANNEL=111"));
    }

    #[test]
    fn configure_slack_gateway_platform_writes_manifest_and_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "slack")
            .unwrap();

        let mut input = Cursor::new("xoxb-token\nxapp-token\nU123, U456\nC123CHANNEL\n");
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Slack app manifest written to:"));

        let manifest_text =
            fs::read_to_string(context.hermes_home().join("slack-manifest.json")).unwrap();
        assert!(manifest_text.contains("\"socket_mode_enabled\": true"));
        assert!(manifest_text.contains("\"command\": \"/hermes\""));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("SLACK_BOT_TOKEN=xoxb-token"));
        assert!(env_text.contains("SLACK_APP_TOKEN=xapp-token"));
        assert!(env_text.contains("SLACK_ALLOWED_USERS=U123,U456"));
        assert!(env_text.contains("SLACK_HOME_CHANNEL=C123CHANNEL"));
    }

    #[test]
    fn configure_slack_existing_token_can_refresh_manifest_without_reconfigure() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        save_env_value(context.env_path(), "SLACK_BOT_TOKEN", "xoxb-existing").unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "slack")
            .unwrap();

        let mut input = Cursor::new("n\n\n");
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        assert!(context.hermes_home().join("slack-manifest.json").exists());
        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("SLACK_BOT_TOKEN=xoxb-existing"));
        assert!(!env_text.contains("SLACK_APP_TOKEN="));
    }

    #[test]
    fn configure_bluebubbles_gateway_platform_writes_env_values_and_webhook_port() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "bluebubbles")
            .unwrap();

        let mut input = Cursor::new(
            "http://192.168.1.10:1234/\nsecret\n+15551234567, user@icloud.com\n+15551234567\ny\n8765\n",
        );
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Webhook port set to 8765"));
        assert!(rendered.contains("BlueBubbles Private API helper"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234"));
        assert!(env_text.contains("BLUEBUBBLES_PASSWORD=secret"));
        assert!(env_text.contains("BLUEBUBBLES_ALLOWED_USERS=+15551234567,user@icloud.com"));
        assert!(env_text.contains("BLUEBUBBLES_HOME_CHANNEL=+15551234567"));
        assert!(env_text.contains("BLUEBUBBLES_WEBHOOK_PORT=8765"));

        let status = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "bluebubbles")
            .unwrap()
            .status;
        assert_eq!(status, "configured");
    }

    #[test]
    fn configure_homeassistant_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "homeassistant")
            .unwrap();

        let mut input = Cursor::new("ha-token\nhttp://ha.example.local:8123/\n");
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("HASS_TOKEN=ha-token"));
        assert!(env_text.contains("HASS_URL=http://ha.example.local:8123"));
    }

    #[test]
    fn configure_webhook_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "webhook")
            .unwrap();

        let mut input = Cursor::new("yes\n8646\nshared-secret\n");
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("WEBHOOK_ENABLED=true"));
        assert!(env_text.contains("WEBHOOK_PORT=8646"));
        assert!(env_text.contains("WEBHOOK_SECRET=shared-secret"));
    }

    #[test]
    fn configure_api_server_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "api_server")
            .unwrap();

        let mut input = Cursor::new(
            "1\n0.0.0.0\n9000\nsuper-secret\nhttps://chat.example.com, https://admin.example.com\nhermes-edge\n",
        );
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("API_SERVER_ENABLED=true"));
        assert!(env_text.contains("API_SERVER_HOST=0.0.0.0"));
        assert!(env_text.contains("API_SERVER_PORT=9000"));
        assert!(env_text.contains("API_SERVER_KEY=super-secret"));
        assert!(env_text.contains(
            "API_SERVER_CORS_ORIGINS=https://chat.example.com,https://admin.example.com"
        ));
        assert!(env_text.contains("API_SERVER_MODEL_NAME=hermes-edge"));
    }

    #[test]
    fn configure_yuanbao_gateway_platform_writes_extended_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let platform = native_gateway_setup_metadata(&context)
            .into_iter()
            .find(|platform| platform.key == "yuanbao")
            .unwrap();

        let mut input = Cursor::new(
            "yb-app\nyb-secret\nyb-bot\ngroup:home123\nhttps://bot.yuanbao.tencent.com/\nwss://bot-wss.yuanbao.tencent.com/wss/connection\n",
        );
        let mut output = Vec::new();
        configure_standard_gateway_platform_with_io(&context, &platform, &mut input, &mut output)
            .unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("YUANBAO_APP_ID=yb-app"));
        assert!(env_text.contains("YUANBAO_APP_SECRET=yb-secret"));
        assert!(env_text.contains("YUANBAO_BOT_ID=yb-bot"));
        assert!(env_text.contains("YUANBAO_HOME_CHANNEL=group:home123"));
        assert!(env_text.contains("YUANBAO_API_DOMAIN=https://bot.yuanbao.tencent.com"));
        assert!(
            env_text.contains("YUANBAO_WS_URL=wss://bot-wss.yuanbao.tencent.com/wss/connection")
        );
    }

    #[test]
    fn configure_irc_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let mut input = Cursor::new(
            "irc.libera.chat\n\n\nhermes-bot\n#hermes\nn\ny\nnickpass\nn\nalice, bob\n",
        );
        let mut output = Vec::new();

        configure_irc_gateway_platform_with_io(&context, &mut input, &mut output).unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("IRC_SERVER=irc.libera.chat"));
        assert!(env_text.contains("IRC_USE_TLS=true"));
        assert!(env_text.contains("IRC_NICKNAME=hermes-bot"));
        assert!(env_text.contains("IRC_CHANNEL=#hermes"));
        assert!(env_text.contains("IRC_NICKSERV_PASSWORD=nickpass"));
        assert!(env_text.contains("IRC_ALLOW_ALL_USERS=false"));
        assert!(env_text.contains("IRC_ALLOWED_USERS=alice,bob"));
    }

    #[test]
    fn configure_teams_gateway_platform_writes_env_values() {
        let (_temp, context) = test_context();
        fs::create_dir_all(context.hermes_home()).unwrap();
        let mut input = Cursor::new("client-id\nclient-secret\ntenant-id\n\nuser-1, user-2\n");
        let mut output = Vec::new();

        configure_teams_gateway_platform_with_io(&context, &mut input, &mut output).unwrap();

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TEAMS_CLIENT_ID=client-id"));
        assert!(env_text.contains("TEAMS_CLIENT_SECRET=client-secret"));
        assert!(env_text.contains("TEAMS_TENANT_ID=tenant-id"));
        assert!(env_text.contains("TEAMS_ALLOWED_USERS=user-1,user-2"));
        assert!(!env_text.contains("TEAMS_ALLOW_ALL_USERS=true"));
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
        fs::create_dir_all(ctx.hermes_home()).unwrap();
        fs::write(
            ctx.hermes_home().join("gateway.pid"),
            format!("{{\"pid\":{}}}\n", child.id()),
        )
        .unwrap();

        let mut unrelated = Command::new("bash")
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
        assert!(process_running(unrelated.id() as i64));
        let _ = unrelated.kill();
        let _ = unrelated.wait();
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
