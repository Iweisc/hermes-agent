use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tools::ToolRuntime;

const DEFAULT_TIMEOUT_SECS: u64 = 180;
const FOREGROUND_MAX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_LOG_LIMIT: usize = 200;
const MAX_LOG_LIMIT: usize = 2_000;
const DEFAULT_WAIT_TIMEOUT_SECS: u64 = 300;
const KILL_GRACE_MILLIS: u64 = 1_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessSession {
    id: String,
    pid: i32,
    pgid: i32,
    command: String,
    cwd: String,
    started_at: f64,
    log_path: PathBuf,
    status_path: PathBuf,
}

pub fn terminal_schema() -> Value {
    json!({
        "name": "terminal",
        "description": "Execute a shell command on the local machine. Supports foreground execution and persistent background jobs managed through the process tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "background": {
                    "type": "boolean",
                    "default": false,
                    "description": "Run the command as a managed background process"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": FOREGROUND_MAX_TIMEOUT_SECS,
                    "default": DEFAULT_TIMEOUT_SECS,
                    "description": "Foreground timeout in seconds"
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command"
                }
            },
            "required": ["command"]
        }
    })
}

pub fn process_schema() -> Value {
    json!({
        "name": "process",
        "description": "Manage background processes started by terminal(background=true). Actions: list, poll, log, wait, kill.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "log", "wait", "kill"],
                    "description": "Action to perform"
                },
                "session_id": {
                    "type": "string",
                    "description": "Process session id for actions other than list"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Wait timeout in seconds"
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Line offset for log pagination"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LOG_LIMIT,
                    "description": "Maximum number of log lines to return"
                }
            },
            "required": ["action"]
        }
    })
}

pub fn handle_terminal(args: &Value, runtime: &ToolRuntime) -> String {
    let command = match required_non_empty_string(args, "command") {
        Ok(command) => command,
        Err(error) => return json_error(error),
    };
    let background = match optional_bool(args, "background") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return json_error(error),
    };
    let timeout = match optional_u64(args, "timeout") {
        Ok(value) => value.unwrap_or(DEFAULT_TIMEOUT_SECS),
        Err(error) => return json_error(error),
    };
    if timeout == 0 || timeout > FOREGROUND_MAX_TIMEOUT_SECS {
        return json_error(format!(
            "timeout must be between 1 and {FOREGROUND_MAX_TIMEOUT_SECS}"
        ));
    }
    let workdir = match optional_non_empty_string(args, "workdir") {
        Ok(value) => value,
        Err(error) => return json_error(error),
    };
    let cwd = match resolve_workdir(runtime, workdir.as_deref()) {
        Ok(path) => path,
        Err(error) => return json_error(error),
    };

    if background {
        return spawn_background_command(&command, &cwd, runtime);
    }
    run_foreground_command(&command, &cwd, timeout)
}

pub fn handle_process(args: &Value, runtime: &ToolRuntime) -> String {
    let action = match required_non_empty_string(args, "action") {
        Ok(action) => action,
        Err(error) => return json_error(error),
    };
    match action.as_str() {
        "list" => list_processes(runtime),
        "poll" => with_session(args, runtime, poll_process),
        "log" => with_session(args, runtime, read_process_log),
        "wait" => with_session(args, runtime, wait_process),
        "kill" => with_session(args, runtime, kill_process),
        _ => json_error("action must be one of: list, poll, log, wait, kill"),
    }
}

pub fn terminal_available() -> bool {
    shell_available()
}

fn with_session(
    args: &Value,
    runtime: &ToolRuntime,
    handler: fn(&Value, &ToolRuntime, &ProcessSession) -> String,
) -> String {
    let session_id = match required_non_empty_string(args, "session_id") {
        Ok(session_id) => session_id,
        Err(error) => return json_error(error),
    };
    let session = match load_session(runtime, &session_id) {
        Ok(session) => session,
        Err(error) => return json_error(error),
    };
    handler(args, runtime, &session)
}

fn list_processes(runtime: &ToolRuntime) -> String {
    let root = processes_root(runtime);
    let mut sessions = Vec::new();
    if let Ok(entries) = fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            let meta = path.join("session.json");
            if !meta.is_file() {
                continue;
            }
            if let Ok(text) = fs::read_to_string(&meta)
                && let Ok(session) = serde_json::from_str::<ProcessSession>(&text)
            {
                sessions.push(session_summary(&session, None));
            }
        }
    }
    sessions.sort_by(|left, right| {
        right["started_at"]
            .as_f64()
            .partial_cmp(&left["started_at"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    json_result(json!({ "processes": sessions }))
}

fn poll_process(_args: &Value, _runtime: &ToolRuntime, session: &ProcessSession) -> String {
    let exit_code = read_exit_code(&session.status_path);
    let running = exit_code.is_none() && process_alive(session.pid);
    let output_tail = read_last_lines(&session.log_path, DEFAULT_LOG_LIMIT);
    json_result(json!({
        "session_id": session.id,
        "command": session.command,
        "pid": session.pid,
        "running": running,
        "exit_code": exit_code,
        "started_at": session.started_at,
        "cwd": session.cwd,
        "output_tail": output_tail,
    }))
}

fn read_process_log(args: &Value, _runtime: &ToolRuntime, session: &ProcessSession) -> String {
    let limit = match optional_usize(args, "limit") {
        Ok(value) => value.unwrap_or(DEFAULT_LOG_LIMIT),
        Err(error) => return json_error(error),
    };
    if limit == 0 || limit > MAX_LOG_LIMIT {
        return json_error(format!("limit must be between 1 and {MAX_LOG_LIMIT}"));
    }
    let offset = match optional_usize(args, "offset") {
        Ok(value) => value,
        Err(error) => return json_error(error),
    };
    let lines = read_all_lines(&session.log_path);
    let total = lines.len();
    let start = offset.unwrap_or_else(|| total.saturating_sub(limit));
    let rows = lines
        .into_iter()
        .skip(start)
        .take(limit)
        .collect::<Vec<_>>();
    json_result(json!({
        "session_id": session.id,
        "offset": start,
        "limit": limit,
        "total_lines": total,
        "lines": rows,
    }))
}

fn wait_process(args: &Value, _runtime: &ToolRuntime, session: &ProcessSession) -> String {
    let timeout = match optional_u64(args, "timeout") {
        Ok(value) => value.unwrap_or(DEFAULT_WAIT_TIMEOUT_SECS),
        Err(error) => return json_error(error),
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout);

    loop {
        let exit_code = read_exit_code(&session.status_path);
        if let Some(exit_code) = exit_code {
            let output_tail = read_last_lines(&session.log_path, DEFAULT_LOG_LIMIT);
            return json_result(json!({
                "session_id": session.id,
                "running": false,
                "exit_code": exit_code,
                "output_tail": output_tail,
            }));
        }
        if !process_alive(session.pid) {
            let output_tail = read_last_lines(&session.log_path, DEFAULT_LOG_LIMIT);
            return json_result(json!({
                "session_id": session.id,
                "running": false,
                "exit_code": Value::Null,
                "output_tail": output_tail,
                "warning": "process exited without writing a status file",
            }));
        }
        if std::time::Instant::now() >= deadline {
            let output_tail = read_last_lines(&session.log_path, DEFAULT_LOG_LIMIT);
            return json_result(json!({
                "session_id": session.id,
                "running": true,
                "exit_code": Value::Null,
                "output_tail": output_tail,
                "timeout": timeout,
            }));
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn kill_process(_args: &Value, _runtime: &ToolRuntime, session: &ProcessSession) -> String {
    let was_running = process_alive(session.pid);
    if was_running {
        let _ = signal_group(session.pgid, libc::SIGTERM);
        thread::sleep(Duration::from_millis(KILL_GRACE_MILLIS));
        if process_alive(session.pid) {
            let _ = signal_group(session.pgid, libc::SIGKILL);
        }
    }

    let mut exit_code = read_exit_code(&session.status_path);
    if exit_code.is_none() {
        let fallback = if was_running { 143 } else { -1 };
        let _ = fs::write(&session.status_path, fallback.to_string());
        exit_code = Some(fallback);
    }

    json_result(json!({
        "session_id": session.id,
        "killed": was_running,
        "running": process_alive(session.pid),
        "exit_code": exit_code,
    }))
}

fn spawn_background_command(command: &str, cwd: &Path, runtime: &ToolRuntime) -> String {
    let root = processes_root(runtime);
    if let Err(error) = fs::create_dir_all(&root) {
        return json_error(format!(
            "creating process directory {} failed: {error}",
            root.display()
        ));
    }

    let session_id = format!("proc_{:x}", unix_ts_nanos());
    let session_dir = root.join(&session_id);
    if let Err(error) = fs::create_dir_all(&session_dir) {
        return json_error(format!(
            "creating process session {} failed: {error}",
            session_dir.display()
        ));
    }

    let log_path = session_dir.join("output.log");
    let status_path = session_dir.join("exit_code");
    let meta_path = session_dir.join("session.json");
    let log_file = match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(file) => file,
        Err(error) => return json_error(format!("opening {} failed: {error}", log_path.display())),
    };
    let stderr_file = match log_file.try_clone() {
        Ok(file) => file,
        Err(error) => return json_error(format!("cloning log handle failed: {error}")),
    };

    let wrapped = format!(
        "{command}\ncode=$?\nprintf '%s' \"$code\" > {status}\nexit \"$code\"",
        status = shell_quote(&status_path.display().to_string()),
    );
    let mut child = match shell_command(&wrapped, cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return json_error(format!("starting background command failed: {error}")),
    };

    let pid = child.id() as i32;
    let session = ProcessSession {
        id: session_id.clone(),
        pid,
        pgid: pid,
        command: command.to_string(),
        cwd: cwd.display().to_string(),
        started_at: unix_ts_secs(),
        log_path,
        status_path,
    };
    if let Err(error) = fs::write(
        &meta_path,
        serde_json::to_vec_pretty(&session).unwrap_or_default(),
    ) {
        let _ = signal_group(pid, libc::SIGTERM);
        return json_error(format!("writing process metadata failed: {error}"));
    }
    let _ = child.stdin.take();
    drop(child);

    json_result(json!({
        "output": "Background process started",
        "session_id": session.id,
        "pid": session.pid,
        "exit_code": 0,
        "error": Value::Null,
    }))
}

fn run_foreground_command(command: &str, cwd: &Path, timeout: u64) -> String {
    let child = match shell_command(command, cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return json_error(format!("starting command failed: {error}")),
    };

    let pid = child.id() as i32;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let output = match rx.recv_timeout(Duration::from_secs(timeout)) {
        Ok(result) => match result {
            Ok(output) => output,
            Err(error) => return json_error(format!("waiting for command failed: {error}")),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = signal_group(pid, libc::SIGTERM);
            thread::sleep(Duration::from_millis(KILL_GRACE_MILLIS));
            if process_alive(pid) {
                let _ = signal_group(pid, libc::SIGKILL);
            }
            let timeout_output = rx
                .recv_timeout(Duration::from_secs(3))
                .ok()
                .and_then(Result::ok);
            let output = timeout_output
                .as_ref()
                .map(combine_output)
                .unwrap_or_default();
            return json_result(json!({
                "output": output.trim(),
                "exit_code": 124,
                "error": format!("command timed out after {timeout} seconds"),
                "status": "timeout",
            }));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return json_error("command worker disconnected unexpectedly");
        }
    };

    let combined = combine_output(&output);
    json_result(json!({
        "output": combined.trim(),
        "exit_code": output.status.code().unwrap_or_default(),
        "error": Value::Null,
        "status": if output.status.success() { "completed" } else { "failed" },
    }))
}

fn shell_command(command: &str, cwd: &Path) -> Command {
    let mut builder = Command::new("bash");
    builder.arg("-lc").arg(command).current_dir(cwd);
    // SAFETY: setsid is called in the child immediately before exec to create
    // a dedicated process group so timeouts and kill requests can signal the
    // whole subtree.
    unsafe {
        builder.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    builder
}

fn resolve_workdir(runtime: &ToolRuntime, raw: Option<&str>) -> Result<PathBuf, String> {
    let path = match raw {
        Some(raw) => runtime.resolve_path(raw)?,
        None => runtime.cwd().to_path_buf(),
    };
    if !path.exists() {
        return Err(format!("workdir does not exist: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("workdir is not a directory: {}", path.display()));
    }
    Ok(path)
}

fn load_session(runtime: &ToolRuntime, session_id: &str) -> Result<ProcessSession, String> {
    if !valid_session_id(session_id) {
        return Err("session_id must match proc_<hex>".to_string());
    }
    let meta_path = processes_root(runtime)
        .join(session_id)
        .join("session.json");
    let text = fs::read_to_string(&meta_path)
        .map_err(|error| format!("reading {} failed: {error}", meta_path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("invalid session metadata: {error}"))
}

fn session_summary(session: &ProcessSession, output_tail: Option<String>) -> Value {
    let exit_code = read_exit_code(&session.status_path);
    let running = exit_code.is_none() && process_alive(session.pid);
    json!({
        "session_id": session.id,
        "command": session.command,
        "pid": session.pid,
        "running": running,
        "exit_code": exit_code,
        "cwd": session.cwd,
        "started_at": session.started_at,
        "output_tail": output_tail,
    })
}

fn read_exit_code(path: &Path) -> Option<i32> {
    let text = fs::read_to_string(path).ok()?;
    text.trim().parse::<i32>().ok()
}

fn read_last_lines(path: &Path, limit: usize) -> String {
    let lines = read_all_lines(path);
    lines
        .into_iter()
        .rev()
        .take(limit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

fn read_all_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(ToOwned::to_owned)
        .collect()
}

fn combine_output(output: &Output) -> String {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

fn process_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(pid, 0) is the standard POSIX existence check and does
    // not actually send a signal.
    let rc = unsafe { libc::kill(pid, 0) };
    if rc == 0 {
        return true;
    }
    matches!(io::Error::last_os_error().raw_os_error(), Some(libc::EPERM))
}

fn signal_group(pgid: i32, signal: i32) -> io::Result<()> {
    if pgid <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "process group id must be positive",
        ));
    }
    // SAFETY: negative pid targets the process group created via setsid.
    let rc = unsafe { libc::kill(-pgid, signal) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn valid_session_id(value: &str) -> bool {
    value.starts_with("proc_")
        && value.len() > 5
        && value[5..].chars().all(|ch| ch.is_ascii_hexdigit())
}

fn processes_root(runtime: &ToolRuntime) -> PathBuf {
    runtime.hermes_home().join("processes")
}

fn shell_available() -> bool {
    Command::new("bash")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn unix_ts_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn json_error(message: impl Into<String>) -> String {
    json!({ "error": message.into() }).to_string()
}

fn json_result(value: Value) -> String {
    value.to_string()
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    let value = args
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))?;
    Ok(value.to_string())
}

fn optional_non_empty_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.trim().to_string())),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn optional_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn optional_u64(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a positive integer")),
    }
}

fn optional_usize(args: &Value, key: &str) -> Result<Option<usize>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(|value| value as usize)
            .map(Some)
            .ok_or_else(|| format!("{key} must be a positive integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    #[test]
    fn foreground_terminal_runs_and_returns_output() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_terminal(
            &json!({
                "command": "printf 'hi from shell\\n'",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["exit_code"], json!(0));
        assert_eq!(parsed["output"], json!("hi from shell"));
    }

    #[test]
    fn background_terminal_process_can_be_polled_and_killed() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let start = handle_terminal(
            &json!({
                "command": "printf 'ready\\n'; sleep 30",
                "background": true,
            }),
            &runtime,
        );
        let start_json: Value = serde_json::from_str(&start).unwrap();
        let session_id = start_json["session_id"].as_str().unwrap().to_string();

        let poll = handle_process(
            &json!({
                "action": "poll",
                "session_id": session_id,
            }),
            &runtime,
        );
        let poll_json: Value = serde_json::from_str(&poll).unwrap();
        assert!(poll_json["running"].is_boolean());

        let kill = handle_process(
            &json!({
                "action": "kill",
                "session_id": start_json["session_id"],
            }),
            &runtime,
        );
        let kill_json: Value = serde_json::from_str(&kill).unwrap();
        assert!(kill_json["exit_code"].is_number());
    }

    #[test]
    fn process_log_returns_recent_lines() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let start = handle_terminal(
            &json!({
                "command": "printf 'alpha\\nbeta\\n'",
                "background": true,
            }),
            &runtime,
        );
        let start_json: Value = serde_json::from_str(&start).unwrap();
        let session_id = start_json["session_id"].as_str().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            let log = handle_process(
                &json!({
                    "action": "log",
                    "session_id": session_id,
                    "limit": 10,
                }),
                &runtime,
            );
            let log_json: Value = serde_json::from_str(&log).unwrap();
            if log_json["lines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|line| line == "alpha")
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "background process log did not contain expected output in time"
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
}
