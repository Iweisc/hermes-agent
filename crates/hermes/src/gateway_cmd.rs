use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use hermes_core::{HermesContext, is_container, is_wsl};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::python_bridge::launch_python_main_command;

const SERVICE_BASE: &str = "hermes-gateway";

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

pub fn print_gateway(context: &HermesContext, args: GatewayArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        None => bridge_gateway(args.accept_hooks, &[String::from("run")]),
        Some(GatewayCommand::Run(run)) => bridge_gateway(args.accept_hooks, &bridge_run_args(run)),
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
            bridge_gateway(args.accept_hooks, &bridge_install_args(install))
        }
        Some(GatewayCommand::Uninstall(args2)) => print_gateway_uninstall(context, args2),
        Some(GatewayCommand::Setup) => bridge_gateway(args.accept_hooks, &[String::from("setup")]),
        Some(GatewayCommand::MigrateLegacy(args2)) => {
            bridge_gateway(args.accept_hooks, &bridge_migrate_legacy_args(args2))
        }
    }
}

fn print_gateway_start(
    context: &HermesContext,
    accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        return bridge_gateway(accept_hooks, &bridge_service_args("start", args));
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
    accept_hooks: bool,
    args: GatewayServiceArgs,
) -> Result<(), Box<dyn Error>> {
    if args.all {
        return bridge_gateway(accept_hooks, &bridge_service_args("stop", args));
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
        return bridge_gateway(accept_hooks, &bridge_service_args("restart", args));
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
    bridge_gateway(accept_hooks, &[String::from("run")])
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

fn bridge_gateway(accept_hooks: bool, argv: &[String]) -> Result<(), Box<dyn Error>> {
    let extra_env = if accept_hooks {
        vec![("HERMES_ACCEPT_HOOKS".to_string(), "1".to_string())]
    } else {
        Vec::new()
    };
    launch_python_main_command("gateway", argv, Some("HERMES_GATEWAY_PYTHON"), &extra_env)
}

fn bridge_run_args(args: GatewayRunArgs) -> Vec<String> {
    let mut argv = Vec::new();
    if args.verbose > 0 {
        argv.push("run".to_string());
    }
    for _ in 0..args.verbose {
        argv.push("--verbose".to_string());
    }
    if args.quiet {
        if argv.is_empty() {
            argv.push("run".to_string());
        }
        argv.push("--quiet".to_string());
    }
    if args.replace {
        if argv.is_empty() {
            argv.push("run".to_string());
        }
        argv.push("--replace".to_string());
    }
    argv
}

fn bridge_service_args(subcommand: &str, args: GatewayServiceArgs) -> Vec<String> {
    let mut argv = vec![subcommand.to_string()];
    if args.system {
        argv.push("--system".to_string());
    }
    if args.all {
        argv.push("--all".to_string());
    }
    argv
}

fn bridge_install_args(args: GatewayInstallArgs) -> Vec<String> {
    let mut argv = vec!["install".to_string()];
    if args.force {
        argv.push("--force".to_string());
    }
    if args.system {
        argv.push("--system".to_string());
    }
    if let Some(user) = args
        .run_as_user
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        argv.push("--run-as-user".to_string());
        argv.push(user.to_string());
    }
    argv
}

fn bridge_migrate_legacy_args(args: GatewayMigrateLegacyArgs) -> Vec<String> {
    let mut argv = vec!["migrate-legacy".to_string()];
    if args.dry_run {
        argv.push("--dry-run".to_string());
    }
    if args.yes {
        argv.push("--yes".to_string());
    }
    argv
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
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
    fn bridge_run_args_omits_run_for_default_invocation() {
        assert!(
            bridge_run_args(GatewayRunArgs {
                verbose: 0,
                quiet: false,
                replace: false,
            })
            .is_empty()
        );
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
    fn bridge_gateway_for_status_subcommands_preserves_argv() {
        let args = bridge_service_args(
            "restart",
            GatewayServiceArgs {
                system: true,
                all: true,
            },
        );
        assert_eq!(args, vec!["restart", "--system", "--all"]);
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
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"{}\"\nif [[ \"$*\" == *\"is-system-running\"* ]]; then\n  printf 'running\\n'\nfi\nexit 0\n",
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
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"{}\"\nif [[ \"$*\" == *\"is-system-running\"* ]]; then\n  printf 'running\\n'\nfi\nexit 0\n",
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
