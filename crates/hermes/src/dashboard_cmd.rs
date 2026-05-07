use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::Duration;

use clap::Args;

use crate::python_bridge::{project_root, resolve_repo_python};

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
    launch_dashboard(args)
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

fn launch_dashboard(args: DashboardArgs) -> Result<(), Box<dyn Error>> {
    validate_dashboard_args(&args)?;

    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_DASHBOARD_PYTHON"))
        .ok_or("could not find a Python interpreter for dashboard launch")?;

    ensure_dashboard_python_dependencies(&python, &root)?;
    ensure_dashboard_web_ui(&root)?;

    let embedded_chat = dashboard_embedded_chat_enabled(&args);
    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_DASHBOARD_HOST", args.host.trim())
        .env("HERMES_DASHBOARD_PORT", args.port.to_string())
        .env(
            "HERMES_DASHBOARD_OPEN_BROWSER",
            if args.no_open { "0" } else { "1" },
        )
        .env(
            "HERMES_DASHBOARD_ALLOW_PUBLIC",
            if args.insecure { "1" } else { "0" },
        )
        .env(
            "HERMES_DASHBOARD_EMBEDDED_CHAT",
            if embedded_chat { "1" } else { "0" },
        )
        .arg("-c")
        .arg(DASHBOARD_BOOTSTRAP);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("dashboard", status).into())
}

const DASHBOARD_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.web_server import start_server\n",
    "start_server(\n",
    "    host=os.environ['HERMES_DASHBOARD_HOST'],\n",
    "    port=int(os.environ['HERMES_DASHBOARD_PORT']),\n",
    "    open_browser=os.environ.get('HERMES_DASHBOARD_OPEN_BROWSER') == '1',\n",
    "    allow_public=os.environ.get('HERMES_DASHBOARD_ALLOW_PUBLIC') == '1',\n",
    "    embedded_chat=os.environ.get('HERMES_DASHBOARD_EMBEDDED_CHAT') == '1',\n",
    ")\n",
);

fn validate_dashboard_args(args: &DashboardArgs) -> Result<(), Box<dyn Error>> {
    if args.host.trim().is_empty() {
        return Err("--host must not be empty".into());
    }
    if args.host.chars().any(char::is_whitespace) {
        return Err("--host must not contain whitespace".into());
    }
    Ok(())
}

fn dashboard_embedded_chat_enabled(args: &DashboardArgs) -> bool {
    args.tui || env::var("HERMES_DASHBOARD_TUI").is_ok_and(|value| value.trim() == "1")
}

fn ensure_dashboard_python_dependencies(
    python: &Path,
    project_root: &Path,
) -> Result<(), Box<dyn Error>> {
    let status = Command::new(python)
        .current_dir(project_root)
        .arg("-c")
        .arg("import fastapi, uvicorn")
        .status()?;
    if status.success() {
        return Ok(());
    }

    eprintln!("Web UI dependencies not installed (need fastapi + uvicorn).");
    eprintln!("Re-install the package into this interpreter so metadata updates apply:");
    eprintln!("  cd {}", project_root.display());
    eprintln!("  {} -m pip install -e .", python.display());
    eprintln!("If `pip` is missing in this venv, use:  uv pip install -e .");
    Err("dashboard dependencies are missing".into())
}

fn ensure_dashboard_web_ui(project_root: &Path) -> Result<(), Box<dyn Error>> {
    if env::var_os("HERMES_WEB_DIST").is_some() {
        return Ok(());
    }

    let web_dir = project_root.join("web");
    if !web_dir.join("package.json").exists() {
        return Ok(());
    }
    if !web_ui_build_needed(&web_dir)? {
        return Ok(());
    }

    let Some(npm) = which_on_path("npm") else {
        eprintln!("Web UI frontend not built and npm is not available.");
        eprintln!("Install Node.js, then run:  cd web && npm install && npm run build");
        return Err("dashboard frontend is not built".into());
    };

    println!("→ Building web UI...");
    let install = run_npm_install_deterministic(&npm, &web_dir)?;
    if !install.status.success() {
        eprintln!("  ✗ Web UI npm install failed");
        eprintln!("  Run manually:  cd web && npm install && npm run build");
        return Err("dashboard frontend npm install failed".into());
    }

    let build = Command::new(&npm)
        .current_dir(&web_dir)
        .arg("run")
        .arg("build")
        .output()?;
    if !build.status.success() {
        eprintln!("  ✗ Web UI build failed");
        eprintln!("  Run manually:  cd web && npm install && npm run build");
        return Err("dashboard frontend build failed".into());
    }

    println!("  ✓ Web UI built");
    Ok(())
}

fn web_ui_build_needed(web_dir: &Path) -> Result<bool, Box<dyn Error>> {
    let dist_dir = web_dir
        .parent()
        .ok_or("web directory has no parent")?
        .join("hermes_cli")
        .join("web_dist");
    let manifest = dist_dir.join(".vite").join("manifest.json");
    let sentinel = if manifest.exists() {
        manifest
    } else {
        dist_dir.join("index.html")
    };
    if !sentinel.exists() {
        return Ok(true);
    }
    let dist_mtime = sentinel.metadata()?.modified()?;
    if web_source_newer_than(web_dir, dist_mtime)? {
        return Ok(true);
    }

    for meta in [
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "vite.config.ts",
        "vite.config.js",
    ] {
        let path = web_dir.join(meta);
        if path.exists() && path.metadata()?.modified()? > dist_mtime {
            return Ok(true);
        }
    }
    Ok(false)
}

fn web_source_newer_than(
    path: &Path,
    dist_mtime: std::time::SystemTime,
) -> Result<bool, Box<dyn Error>> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if file_type.is_dir() {
            if name == "node_modules" || name == "dist" {
                continue;
            }
            if web_source_newer_than(&entry_path, dist_mtime)? {
                return Ok(true);
            }
            continue;
        }

        if !file_type.is_file() {
            continue;
        }
        if !has_web_source_extension(&entry_path) {
            continue;
        }
        if entry.metadata()?.modified()? > dist_mtime {
            return Ok(true);
        }
    }
    Ok(false)
}

fn has_web_source_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext, "ts" | "tsx" | "js" | "jsx" | "css" | "html" | "vue"))
}

fn run_npm_install_deterministic(npm: &Path, web_dir: &Path) -> Result<Output, Box<dyn Error>> {
    let lockfile = web_dir.join("package-lock.json");
    if lockfile.exists() {
        let output = Command::new(npm)
            .current_dir(web_dir)
            .arg("ci")
            .arg("--silent")
            .output()?;
        if output.status.success() {
            return Ok(output);
        }
    }
    Ok(Command::new(npm)
        .current_dir(web_dir)
        .arg("install")
        .arg("--silent")
        .output()?)
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

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Parser, Debug)]
    struct DashboardHarness {
        #[command(flatten)]
        args: DashboardArgs,
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        env::temp_dir().join(format!("hermes-rs-dashboard-{label}-{unique}"))
    }

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe { env::set_var(key, value) };
    }

    fn remove_env_var(key: &str) {
        unsafe { env::remove_var(key) };
    }

    #[cfg(unix)]
    fn write_executable(path: &Path, contents: &str) {
        use std::os::unix::fs::PermissionsExt;

        fs::write(path, contents).unwrap();
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

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

    #[test]
    fn dashboard_args_parse_lifecycle_and_launch_flags() {
        let parsed = DashboardHarness::try_parse_from([
            "dashboard",
            "--port",
            "8123",
            "--host",
            "0.0.0.0",
            "--no-open",
            "--insecure",
            "--tui",
        ])
        .unwrap();
        assert_eq!(parsed.args.port, 8123);
        assert_eq!(parsed.args.host, "0.0.0.0");
        assert!(parsed.args.no_open);
        assert!(parsed.args.insecure);
        assert!(parsed.args.tui);
    }

    #[test]
    fn dashboard_embedded_chat_honors_env_fallback() {
        let _guard = test_env_lock().lock().unwrap();
        remove_env_var("HERMES_DASHBOARD_TUI");
        let args = DashboardArgs {
            port: 9119,
            host: String::from("127.0.0.1"),
            no_open: false,
            insecure: false,
            tui: false,
            stop: false,
            status: false,
        };
        assert!(!dashboard_embedded_chat_enabled(&args));

        set_env_var("HERMES_DASHBOARD_TUI", "1");
        assert!(dashboard_embedded_chat_enabled(&args));
        remove_env_var("HERMES_DASHBOARD_TUI");
    }

    #[test]
    fn web_ui_build_needed_detects_newer_source_file() {
        let temp = temp_path("build-needed");
        let web_dir = temp.join("web");
        let dist_dir = temp.join("hermes_cli").join("web_dist").join(".vite");
        fs::create_dir_all(&web_dir).unwrap();
        fs::create_dir_all(&dist_dir).unwrap();
        fs::write(web_dir.join("package.json"), "{}\n").unwrap();
        fs::write(dist_dir.join("manifest.json"), "{}\n").unwrap();

        std::thread::sleep(Duration::from_millis(20));
        fs::write(web_dir.join("src.tsx"), "export {};\n").unwrap();

        assert!(web_ui_build_needed(&web_dir).unwrap());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    #[cfg(unix)]
    fn launch_dashboard_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = temp_path("launch");
        let log = temp.join("python.log");
        let fake_python = temp.join("python3");
        fs::create_dir_all(&temp).unwrap();
        write_executable(
            &fake_python,
            &format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  case \"$2\" in\n\
    'import fastapi, uvicorn')\n\
      echo deps >> '{}'\n\
      exit 0\n\
      ;;\n\
    *)\n\
      printf 'launch host=%s port=%s open=%s insecure=%s chat=%s\\n' \\\n\
        \"$HERMES_DASHBOARD_HOST\" \"$HERMES_DASHBOARD_PORT\" \\\n\
        \"$HERMES_DASHBOARD_OPEN_BROWSER\" \"$HERMES_DASHBOARD_ALLOW_PUBLIC\" \\\n\
        \"$HERMES_DASHBOARD_EMBEDDED_CHAT\" >> '{}'\n\
      exit 0\n\
      ;;\n\
  esac\n\
fi\n\
exit 7\n",
                log.display(),
                log.display()
            ),
        );

        set_env_var("HERMES_DASHBOARD_PYTHON", &fake_python);
        set_env_var("HERMES_WEB_DIST", temp.join("built-dist"));
        set_env_var("HERMES_DASHBOARD_TUI", "1");

        launch_dashboard(DashboardArgs {
            port: 8123,
            host: String::from("0.0.0.0"),
            no_open: true,
            insecure: true,
            tui: false,
            stop: false,
            status: false,
        })
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("deps"));
        assert!(output.contains("launch host=0.0.0.0 port=8123 open=0 insecure=1 chat=1"));

        remove_env_var("HERMES_DASHBOARD_PYTHON");
        remove_env_var("HERMES_WEB_DIST");
        remove_env_var("HERMES_DASHBOARD_TUI");
        let _ = fs::remove_dir_all(temp);
    }
}
