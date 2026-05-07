use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::tools::{ToolRuntime, dispatch_tool, tool_error};

const DEFAULT_TIMEOUT_SECS: u64 = 300;
const MAX_TOOL_CALLS: usize = 50;
const MAX_STDOUT_BYTES: usize = 50_000;
const MAX_STDERR_BYTES: usize = 10_000;
const SANDBOX_ALLOWED_TOOLS: &[&str] = &[
    "web_search",
    "web_extract",
    "read_file",
    "write_file",
    "search_files",
    "patch",
    "terminal",
];

pub fn execute_code_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn execute_code_schema() -> Value {
    json!({
        "name": "execute_code",
        "description": "Run a Python script that can call a limited subset of Hermes tools programmatically. Use it for multi-step file, web, and terminal workflows where normal tool chaining would be noisy or repetitive. Available imports from `hermes_tools`: `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal`. Scripts run locally with `python3` in the session working directory. Print the final result to stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Import tools with `from hermes_tools import ...` and print the final result to stdout."
                }
            },
            "required": ["code"]
        }
    })
}

pub fn handle_execute_code(args: &Value, runtime: &ToolRuntime) -> String {
    if !execute_code_available() {
        return tool_error("execute_code requires python3 in PATH");
    }

    let Some(code) = args.get("code").and_then(Value::as_str).map(str::trim) else {
        return tool_error("code must be a string");
    };
    if code.is_empty() {
        return tool_error("No code provided.");
    }

    let allowed_tools = sandbox_tools(runtime);
    if allowed_tools.is_empty() {
        return tool_error("execute_code has no available tools in the current session");
    }

    let sandbox_id = format!("{:x}", unix_ts_nanos());
    let temp_root = std::env::temp_dir().join(format!("hermes_exec_{sandbox_id}"));
    let socket_path = std::env::temp_dir().join(format!("hermes_rpc_{sandbox_id}.sock"));
    let listener = match prepare_sandbox(&temp_root, &socket_path, code, &allowed_tools) {
        Ok(listener) => listener,
        Err(error) => return tool_error(error),
    };

    let runtime_clone = runtime.clone();
    let allowed_clone = allowed_tools.clone();
    let rpc_thread = thread::spawn(move || rpc_server_loop(listener, runtime_clone, allowed_clone));

    let result = run_python_script(&temp_root, &socket_path, runtime);
    let _ = std::os::unix::net::UnixStream::connect(&socket_path);
    let tool_calls_made = rpc_thread.join().unwrap_or_default();
    let result = attach_tool_call_count(result, tool_calls_made);
    let _ = fs::remove_dir_all(&temp_root);
    let _ = fs::remove_file(&socket_path);
    result
}

fn sandbox_tools(runtime: &ToolRuntime) -> BTreeSet<String> {
    let allowed = SANDBOX_ALLOWED_TOOLS
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    match runtime.available_tool_names() {
        Some(session_tools) => allowed
            .intersection(session_tools)
            .cloned()
            .collect::<BTreeSet<_>>(),
        None => allowed,
    }
}

fn prepare_sandbox(
    temp_root: &Path,
    socket_path: &Path,
    code: &str,
    allowed_tools: &BTreeSet<String>,
) -> Result<UnixListener, String> {
    fs::create_dir_all(temp_root)
        .map_err(|error| format!("creating sandbox {} failed: {error}", temp_root.display()))?;
    fs::write(
        temp_root.join("hermes_tools.py"),
        generate_hermes_tools_module(allowed_tools),
    )
    .map_err(|error| format!("writing hermes_tools.py failed: {error}"))?;
    fs::write(temp_root.join("script.py"), code.as_bytes())
        .map_err(|error| format!("writing script.py failed: {error}"))?;
    let _ = fs::remove_file(socket_path);
    UnixListener::bind(socket_path).map_err(|error| {
        format!(
            "binding rpc socket {} failed: {error}",
            socket_path.display()
        )
    })
}

fn generate_hermes_tools_module(enabled_tools: &BTreeSet<String>) -> String {
    let mut stubs = String::new();
    for name in enabled_tools {
        let stub = match name.as_str() {
            "web_search" => {
                "def web_search(query: str, limit: int = 5):\n    return _call('web_search', {'query': query, 'limit': limit})\n"
            }
            "web_extract" => {
                "def web_extract(urls):\n    return _call('web_extract', {'urls': urls})\n"
            }
            "read_file" => {
                "def read_file(path: str, offset: int = 1, limit: int = 500):\n    return _call('read_file', {'path': path, 'offset': offset, 'limit': limit})\n"
            }
            "write_file" => {
                "def write_file(path: str, content: str):\n    return _call('write_file', {'path': path, 'content': content})\n"
            }
            "search_files" => {
                "def search_files(pattern: str, target: str = 'content', path: str = '.', file_glob = None, limit: int = 50, offset: int = 0, output_mode: str = 'content', context: int = 0):\n    return _call('search_files', {'pattern': pattern, 'target': target, 'path': path, 'file_glob': file_glob, 'limit': limit, 'offset': offset, 'output_mode': output_mode, 'context': context})\n"
            }
            "patch" => {
                "def patch(path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = 'replace', patch: str = None):\n    return _call('patch', {'path': path, 'old_string': old_string, 'new_string': new_string, 'replace_all': replace_all, 'mode': mode, 'patch': patch})\n"
            }
            "terminal" => {
                "def terminal(command: str, timeout: int = None, workdir: str = None):\n    return _call('terminal', {'command': command, 'timeout': timeout, 'workdir': workdir})\n"
            }
            _ => continue,
        };
        stubs.push_str(stub);
        stubs.push('\n');
    }

    let header = r#"import json, os, socket, shlex, threading, time
_sock = None
_call_lock = threading.Lock()

def json_parse(text: str):
    return json.loads(text, strict=False)

def shell_quote(s: str) -> str:
    return shlex.quote(s)

def retry(fn, max_attempts=3, delay=2):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_err

def _connect():
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.connect(os.environ['HERMES_RPC_SOCKET'])
        _sock.settimeout(__TIMEOUT__)
    return _sock

def _call(tool_name, args):
    request = json.dumps({'tool': tool_name, 'args': args}) + '\n'
    with _call_lock:
        conn = _connect()
        conn.sendall(request.encode())
        buf = b''
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                raise RuntimeError('Agent process disconnected')
            buf += chunk
            if buf.endswith(b'\n'):
                break
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return result
    return result

"#;
    let header = header.replace("__TIMEOUT__", &DEFAULT_TIMEOUT_SECS.to_string());
    format!("{header}{stubs}")
}

fn rpc_server_loop(
    listener: UnixListener,
    runtime: ToolRuntime,
    allowed_tools: BTreeSet<String>,
) -> usize {
    let Ok((stream, _)) = listener.accept() else {
        return 0;
    };
    let cloned = match stream.try_clone() {
        Ok(stream) => stream,
        Err(_) => return 0,
    };
    let mut reader = BufReader::new(cloned);
    let mut writer = stream;
    let mut tool_calls = 0usize;
    loop {
        let mut line = String::new();
        let Ok(read) = reader.read_line(&mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => value,
            Err(error) => {
                let _ = writer.write_all(
                    format!("{}\n", tool_error(format!("Invalid RPC request: {error}"))).as_bytes(),
                );
                continue;
            }
        };
        let tool_name = request
            .get("tool")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let mut tool_args = request.get("args").cloned().unwrap_or_else(|| json!({}));

        let response = if !allowed_tools.contains(&tool_name) {
            tool_error(format!(
                "Tool '{tool_name}' is not available in execute_code. Available: {}",
                allowed_tools.iter().cloned().collect::<Vec<_>>().join(", ")
            ))
        } else if tool_calls >= MAX_TOOL_CALLS {
            tool_error(format!(
                "Tool call limit reached ({MAX_TOOL_CALLS}). No more tool calls allowed in this execution."
            ))
        } else {
            if tool_name == "terminal"
                && let Some(object) = tool_args.as_object_mut()
            {
                object.remove("background");
            }
            tool_calls += 1;
            dispatch_tool(&tool_name, tool_args, &runtime)
        };

        if writer
            .write_all(format!("{response}\n").as_bytes())
            .is_err()
        {
            break;
        }
        let _ = writer.flush();
    }
    tool_calls
}

fn run_python_script(temp_root: &Path, socket_path: &Path, runtime: &ToolRuntime) -> String {
    let child_env = scrubbed_child_env(temp_root, socket_path);
    let script_path = temp_root.join("script.py");
    let mut command = Command::new("python3");
    command
        .arg(script_path)
        .current_dir(runtime.cwd())
        .env_clear()
        .envs(child_env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return tool_error(format!("starting python3 failed: {error}"));
        }
    };

    let pid = child.id() as i32;
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => return tool_error("execute_code could not capture stdout"),
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => return tool_error("execute_code could not capture stderr"),
    };

    let stdout_reader = thread::spawn(move || read_stream_capped(stdout, MAX_STDOUT_BYTES * 4));
    let stderr_reader = thread::spawn(move || read_stream_capped(stderr, MAX_STDERR_BYTES * 4));
    let deadline = Instant::now() + Duration::from_secs(DEFAULT_TIMEOUT_SECS);

    let (status, exit_code) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break ("success", status.code().unwrap_or_default()),
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_process_group(pid, true);
                    let _ = child.wait();
                    break ("timeout", 124);
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                kill_process_group(pid, false);
                let _ = child.wait();
                return tool_error(format!("waiting for execute_code failed: {error}"));
            }
        }
    };

    let stdout_bytes = stdout_reader.join().unwrap_or_default();
    let stderr_bytes = stderr_reader.join().unwrap_or_default();
    let stdout_text = truncate_head_tail(
        &strip_ansi(&String::from_utf8_lossy(&stdout_bytes)),
        MAX_STDOUT_BYTES,
    );
    let stderr_text = truncate_head_tail(
        &strip_ansi(&String::from_utf8_lossy(&stderr_bytes)),
        MAX_STDERR_BYTES,
    );
    let duration = Instant::now()
        .saturating_duration_since(deadline - Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .as_secs_f64();

    let mut result = json!({
        "status": status,
        "output": stdout_text,
        "tool_calls_made": Value::Null,
        "duration_seconds": duration,
    });

    if status == "timeout" {
        let timeout_msg = format!(
            "Script timed out after {}s and was killed.",
            DEFAULT_TIMEOUT_SECS
        );
        result["error"] = Value::String(timeout_msg.clone());
        result["output"] = Value::String(if stdout_text.is_empty() {
            format!("⏰ {timeout_msg}")
        } else {
            format!("{stdout_text}\n\n⏰ {timeout_msg}")
        });
        return result.to_string();
    }

    if exit_code != 0 {
        result["status"] = Value::String("error".to_string());
        result["error"] = Value::String(if stderr_text.is_empty() {
            format!("Script exited with code {exit_code}")
        } else {
            stderr_text.clone()
        });
        if !stderr_text.is_empty() {
            result["output"] = Value::String(if stdout_text.is_empty() {
                format!("--- stderr ---\n{stderr_text}")
            } else {
                format!("{stdout_text}\n--- stderr ---\n{stderr_text}")
            });
        }
    } else {
        result["error"] = Value::Null;
    }

    result.to_string()
}

fn attach_tool_call_count(result: String, tool_calls_made: usize) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(&result) else {
        return result;
    };
    if let Some(object) = value.as_object_mut() {
        object.insert("tool_calls_made".to_string(), json!(tool_calls_made));
    }
    value.to_string()
}

fn scrubbed_child_env(temp_root: &Path, socket_path: &Path) -> Vec<(String, String)> {
    let safe_prefixes = [
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SHELL",
        "LOGNAME",
        "XDG_",
        "VIRTUAL_ENV",
        "CONDA",
        "HERMES_",
    ];
    let secret_needles = [
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "PASSWD",
        "AUTH",
    ];
    let mut envs = std::env::vars()
        .filter(|(key, _)| {
            let upper = key.to_ascii_uppercase();
            if secret_needles.iter().any(|needle| upper.contains(needle)) {
                return false;
            }
            safe_prefixes.iter().any(|prefix| key.starts_with(prefix))
        })
        .collect::<Vec<_>>();

    let mut python_path = temp_root.display().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH")
        && !existing.trim().is_empty()
    {
        python_path.push(':');
        python_path.push_str(existing.trim());
    }
    envs.retain(|(key, _)| key != "PYTHONPATH" && key != "HERMES_RPC_SOCKET");
    envs.push((
        "HERMES_RPC_SOCKET".to_string(),
        socket_path.display().to_string(),
    ));
    envs.push(("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()));
    envs.push(("PYTHONPATH".to_string(), python_path));
    envs
}

fn read_stream_capped<R: std::io::Read>(mut reader: R, max_bytes: usize) -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                if buffer.len() < max_bytes {
                    let keep = (max_bytes - buffer.len()).min(read);
                    buffer.extend_from_slice(&chunk[..keep]);
                }
            }
            Err(_) => break,
        }
    }
    buffer
}

fn truncate_head_tail(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    let head_len = ((max_chars as f64) * 0.4) as usize;
    let tail_len = max_chars.saturating_sub(head_len);
    let head = chars[..head_len].iter().collect::<String>();
    let tail = chars[chars.len().saturating_sub(tail_len)..]
        .iter()
        .collect::<String>();
    let omitted = chars.len().saturating_sub(head_len + tail_len);
    format!(
        "{head}\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted out of {} total] ...\n\n{tail}",
        chars.len()
    )
}

fn strip_ansi(input: &str) -> String {
    let mut output = String::new();
    let bytes = input.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == 0x1b {
            index += 1;
            if index < bytes.len() && bytes[index] == b'[' {
                index += 1;
                while index < bytes.len() {
                    let ch = bytes[index];
                    index += 1;
                    if (0x40..=0x7e).contains(&ch) {
                        break;
                    }
                }
                continue;
            }
        }
        output.push(bytes[index] as char);
        index += 1;
    }
    output
}

fn kill_process_group(pid: i32, escalate: bool) {
    let _ = signal_group(pid, libc::SIGTERM);
    thread::sleep(Duration::from_millis(1_000));
    if process_alive(pid) && escalate {
        let _ = signal_group(pid, libc::SIGKILL);
    }
}

fn process_alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

fn signal_group(pid: i32, signal: i32) -> Result<(), String> {
    let result = unsafe { libc::killpg(pid, signal) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().to_string())
    }
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}
