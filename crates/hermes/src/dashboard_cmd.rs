use std::error::Error;
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

use clap::Args;

use crate::python_bridge::launch_python_main_command;

const DASHBOARD_PATTERNS: &[&str] = &[
    "hermes dashboard",
    "hermes_cli.main dashboard",
    "hermes_cli/main.py dashboard",
];

#[derive(Args, Debug)]
pub struct DashboardArgs {
    #[arg(long, default_value_t = 9119)]
    pub port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long = "no-open")]
    pub no_open: bool,
    #[arg(long)]
    pub insecure: bool,
    #[arg(long)]
    pub tui: bool,
    #[arg(long)]
    pub stop: bool,
    #[arg(long)]
    pub status: bool,
}

pub fn print_dashboard(args: DashboardArgs) -> Result<(), Box<dyn Error>> {
    if args.status {
        report_dashboard_status()?;
        return Ok(());
    }
    if args.stop {
        let pids = find_dashboard_pids()?;
        if pids.is_empty() {
            println!("No hermes dashboard processes running.");
            return Ok(());
        }
        let result = kill_dashboard_processes("requested via --stop")?;
        if result.failed.is_empty() {
            return Ok(());
        }
        return Err(format!(
            "failed to stop {} dashboard process(es)",
            result.failed.len()
        )
        .into());
    }
    launch_python_dashboard(args)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DashboardProcess {
    pid: i32,
    command: String,
}

#[derive(Debug, Default)]
struct StopResult {
    killed: Vec<i32>,
    failed: Vec<(i32, String)>,
}

fn report_dashboard_status() -> Result<(), Box<dyn Error>> {
    let processes = find_dashboard_processes()?;
    if processes.is_empty() {
        println!("No hermes dashboard processes running.");
        return Ok(());
    }
    println!("{} hermes dashboard process(es) running:", processes.len());
    for process in processes {
        if process.command.is_empty() {
            println!("    PID {}", process.pid);
        } else {
            println!("    PID {}: {}", process.pid, process.command);
        }
    }
    Ok(())
}

fn find_dashboard_pids() -> Result<Vec<i32>, Box<dyn Error>> {
    Ok(find_dashboard_processes()?
        .into_iter()
        .map(|process| process.pid)
        .collect())
}

fn find_dashboard_processes() -> Result<Vec<DashboardProcess>, Box<dyn Error>> {
    let self_pid = std::process::id() as i32;
    #[cfg(windows)]
    {
        let output = Command::new("wmic")
            .args(["process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"])
            .output();
        let Ok(output) = output else {
            return Ok(Vec::new());
        };
        if !output.status.success() {
            return Ok(Vec::new());
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Ok(parse_wmic_process_list(&stdout, self_pid));
    }
    #[cfg(not(windows))]
    {
        let output = Command::new("ps")
            .args(["-A", "-o", "pid=,command="])
            .output();
        let Ok(output) = output else {
            return Ok(Vec::new());
        };
        if !output.status.success() {
            return Ok(Vec::new());
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(parse_ps_process_list(&stdout, self_pid))
    }
}

#[cfg(not(windows))]
fn parse_ps_process_list(output: &str, self_pid: i32) -> Vec<DashboardProcess> {
    output
        .lines()
        .filter_map(|line| {
            let stripped = line.trim();
            if stripped.is_empty() || stripped.contains("grep") {
                return None;
            }
            let mut parts = stripped.splitn(2, char::is_whitespace);
            let pid = parts.next()?.trim().parse::<i32>().ok()?;
            let command = parts.next().unwrap_or("").trim().to_string();
            dashboard_matches(pid, &command, self_pid).then_some(DashboardProcess { pid, command })
        })
        .collect()
}

#[cfg(windows)]
fn parse_wmic_process_list(output: &str, self_pid: i32) -> Vec<DashboardProcess> {
    let mut processes = Vec::new();
    let mut current_command = String::new();
    for line in output.lines() {
        let stripped = line.trim();
        if let Some(command) = stripped.strip_prefix("CommandLine=") {
            current_command = command.trim().to_string();
            continue;
        }
        let Some(pid) = stripped.strip_prefix("ProcessId=") else {
            continue;
        };
        let Ok(pid) = pid.trim().parse::<i32>() else {
            continue;
        };
        if dashboard_matches(pid, &current_command, self_pid) {
            processes.push(DashboardProcess {
                pid,
                command: current_command.clone(),
            });
        }
    }
    processes
}

fn dashboard_matches(pid: i32, command: &str, self_pid: i32) -> bool {
    pid != self_pid
        && DASHBOARD_PATTERNS
            .iter()
            .any(|pattern| command.contains(pattern))
}

fn kill_dashboard_processes(reason: &str) -> Result<StopResult, Box<dyn Error>> {
    let pids = find_dashboard_pids()?;
    if pids.is_empty() {
        return Ok(StopResult::default());
    }
    println!();
    println!("⟲ Stopping {} dashboard process(es) ({reason})", pids.len());

    #[cfg(windows)]
    let result = kill_dashboard_processes_windows(&pids);
    #[cfg(not(windows))]
    let result = kill_dashboard_processes_unix(&pids);

    for pid in &result.killed {
        println!("    ✓ stopped PID {pid}");
    }
    for (pid, detail) in &result.failed {
        println!("    ✗ failed to stop PID {pid}: {detail}");
    }
    if !result.killed.is_empty() {
        println!("  Restart the dashboard when you're ready:");
        println!("    hermes dashboard --port <port>");
    }
    Ok(result)
}

#[cfg(windows)]
fn kill_dashboard_processes_windows(pids: &[i32]) -> StopResult {
    let mut result = StopResult::default();
    for pid in pids {
        match Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output()
        {
            Ok(output) if output.status.success() => result.killed.push(*pid),
            Ok(output) => {
                let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
                result.failed.push((
                    *pid,
                    if detail.is_empty() {
                        "taskkill failed".to_string()
                    } else {
                        detail
                    },
                ));
            }
            Err(error) => result.failed.push((*pid, error.to_string())),
        }
    }
    result
}

#[cfg(not(windows))]
fn kill_dashboard_processes_unix(pids: &[i32]) -> StopResult {
    let mut result = StopResult::default();
    let mut pending = Vec::new();

    for pid in pids {
        let rc = unsafe { libc::kill(*pid, libc::SIGTERM) };
        if rc == 0 {
            pending.push(*pid);
            continue;
        }
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(code) if code == libc::ESRCH => result.killed.push(*pid),
            _ => result.failed.push((*pid, err.to_string())),
        }
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !pending.is_empty() && std::time::Instant::now() < deadline {
        sleep(Duration::from_millis(100));
        let mut survivors = Vec::new();
        for pid in pending {
            let rc = unsafe { libc::kill(pid, 0) };
            if rc == 0 {
                survivors.push(pid);
                continue;
            }
            let err = std::io::Error::last_os_error();
            match err.raw_os_error() {
                Some(code) if code == libc::ESRCH => result.killed.push(pid),
                _ => survivors.push(pid),
            }
        }
        pending = survivors;
    }

    for pid in pending {
        let rc = unsafe { libc::kill(pid, libc::SIGKILL) };
        if rc == 0 {
            result.killed.push(pid);
            continue;
        }
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(code) if code == libc::ESRCH => result.killed.push(pid),
            _ => result.failed.push((pid, err.to_string())),
        }
    }
    result
}

fn launch_python_dashboard(args: DashboardArgs) -> Result<(), Box<dyn Error>> {
    let mut argv = vec![
        "--port".to_string(),
        args.port.to_string(),
        "--host".to_string(),
        args.host,
    ];
    if args.no_open {
        argv.push("--no-open".to_string());
    }
    if args.insecure {
        argv.push("--insecure".to_string());
    }
    if args.tui {
        argv.push("--tui".to_string());
    }
    launch_python_main_command("dashboard", &argv, Some("HERMES_DASHBOARD_PYTHON"), &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(windows))]
    fn parse_ps_process_list_matches_dashboard_patterns() {
        let output = "\
123 /usr/bin/python -m hermes_cli.main dashboard --port 9119\n\
124 hermes chat hi\n\
125 /usr/bin/env hermes dashboard --status\n";
        let processes = parse_ps_process_list(output, 999);
        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].pid, 123);
        assert_eq!(processes[1].pid, 125);
    }

    #[test]
    fn dashboard_matches_ignores_self_and_unrelated_commands() {
        assert!(!dashboard_matches(100, "hermes dashboard", 100));
        assert!(!dashboard_matches(101, "hermes chat dashboard notes", 100));
        assert!(dashboard_matches(
            102,
            "python -m hermes_cli.main dashboard --port 9119",
            100
        ));
    }
}
