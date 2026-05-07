use std::env;
use std::error::Error;
use std::fs;
#[cfg(not(windows))]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

use clap::{Args, Subcommand};
use hermes_core::{HermesContext, is_container, is_wsl};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::python_bridge::launch_python_main_command;

const SERVICE_BASE: &str = "hermes-gateway";

#[derive(Args, Debug, Clone)]
pub struct GatewayArgs {
    #[arg(long, default_value_t = false)]
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

pub fn print_gateway(
    context: &HermesContext,
    args: GatewayArgs,
) -> Result<(), Box<dyn Error>> {
    match args.command {
        None => bridge_gateway(args.accept_hooks, &[]),
        Some(GatewayCommand::Run(run)) => bridge_gateway(args.accept_hooks, &bridge_run_args(run)),
        Some(GatewayCommand::Start(service)) => {
            bridge_gateway(args.accept_hooks, &bridge_service_args("start", service))
        }
        Some(GatewayCommand::Stop(service)) => {
            bridge_gateway(args.accept_hooks, &bridge_service_args("stop", service))
        }
        Some(GatewayCommand::Restart(service)) => {
            bridge_gateway(args.accept_hooks, &bridge_service_args("restart", service))
        }
        Some(GatewayCommand::Status(status)) => print_gateway_status(context, status),
        Some(GatewayCommand::Install(install)) => {
            bridge_gateway(args.accept_hooks, &bridge_install_args(install))
        }
        Some(GatewayCommand::Uninstall(args2)) => {
            bridge_gateway(args.accept_hooks, &bridge_uninstall_args(args2))
        }
        Some(GatewayCommand::Setup) => bridge_gateway(args.accept_hooks, &[String::from("setup")]),
        Some(GatewayCommand::MigrateLegacy(args2)) => {
            bridge_gateway(args.accept_hooks, &bridge_migrate_legacy_args(args2))
        }
    }
}

fn print_gateway_status(
    context: &HermesContext,
    args: GatewayStatusArgs,
) -> Result<(), Box<dyn Error>> {
    let snapshot = gateway_snapshot(context, args.system);
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
            systemd_unit_path(context, snapshot.service_scope.as_deref() == Some("system")).display()
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
            run_optional_status_command(
                "systemctl",
                if snapshot.service_scope.as_deref() == Some("system") {
                    &["status", &gateway_service_name(context), "--no-pager", "-l"]
                } else {
                    &["--user", "status", &gateway_service_name(context), "--no-pager", "-l"]
                },
            );
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
            run_optional_status_command("launchctl", &["list", &launchd_label(context)]);
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
    if let Some(user) = args.run_as_user.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        argv.push("--run-as-user".to_string());
        argv.push(user.to_string());
    }
    argv
}

fn bridge_uninstall_args(args: GatewaySystemArgs) -> Vec<String> {
    let mut argv = vec!["uninstall".to_string()];
    if args.system {
        argv.push("--system".to_string());
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
    let default_root = context.default_hermes_root();
    if home == default_root {
        return SERVICE_BASE.to_string();
    }
    let profiles_root = context.profiles_root();
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
        format!("ai.hermes.gateway-{}", name.trim_start_matches(&format!("{SERVICE_BASE}-")))
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
            let action = if restart_requested { "restart" } else { "shutdown" };
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
        .or_else(|| env::var("LOGNAME").ok().filter(|value| !value.trim().is_empty()))?;
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

fn run_optional_status_command(binary: &str, args: &[&str]) {
    let _ = Command::new(binary).args(args).status();
}

fn process_running(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
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
    fn bridge_run_args_omits_run_for_default_invocation() {
        assert!(bridge_run_args(GatewayRunArgs {
            verbose: 0,
            quiet: false,
            replace: false,
        })
        .is_empty());
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
        assert!(lines.iter().any(|line| line.contains("draining for restart")));
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
    fn select_systemd_scope_prefers_system_when_only_system_unit_exists() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let service = gateway_service_name(&ctx);
        let user_unit = systemd_unit_path(&ctx, false);
        let system_unit = PathBuf::from("/etc/systemd/system").join(format!("{service}.service"));
        let fake_systemd = ctx.home_dir().join("etc-systemd");
        fs::create_dir_all(fake_systemd.clone()).unwrap();
        set_env_var("HERMES_FAKE_SYSTEMD_DIR", &fake_systemd);
        fs::create_dir_all(user_unit.parent().unwrap()).unwrap();
        fs::write(fake_systemd.join(format!("{service}.service")), "x").unwrap();
        assert!(select_systemd_scope(&ctx, false));
        remove_env_var("HERMES_FAKE_SYSTEMD_DIR");
    }
}
