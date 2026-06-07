use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

fn temp_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "hermes-agent-{name}-{nonce}-{}",
        std::process::id()
    ))
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> ExitStatus {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            return child.wait().unwrap();
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn fake_python(path: &PathBuf, log_path: &PathBuf) {
    fs::write(
        path,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{}'\nexit 42\n",
            log_path.display()
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }
}

#[test]
fn native_mode_does_not_spawn_python_worker_and_errors_unmigrated_methods() {
    let bin = env!("CARGO_BIN_EXE_hermes-agent");
    let home = temp_path("native-home");
    let cwd = temp_path("native-cwd");
    let fake_python_path = temp_path("fake-python");
    let python_log = temp_path("python-log");
    fs::create_dir_all(&home).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    fake_python(&fake_python_path, &python_log);

    let mut child = Command::new(bin)
        .arg("tui-gateway")
        .env("HERMES_HOME", &home)
        .env("HERMES_CWD", &cwd)
        .env("HERMES_TUI_NATIVE", "1")
        .env("HERMES_PYTHON", &fake_python_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc":"2.0","id":"create","method":"session.create","params":{"cols":80}})
        )
        .unwrap();
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc":"2.0","id":"tools","method":"tools.list","params":{}})
        )
        .unwrap();
    }
    drop(child.stdin.take());

    let status = wait_with_timeout(&mut child, Duration::from_secs(5));
    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();

    assert!(
        status.success(),
        "gateway exited with {status}; stderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        !python_log.exists(),
        "native mode invoked HERMES_PYTHON unexpectedly: {}",
        fs::read_to_string(&python_log).unwrap_or_default()
    );

    let frames: Vec<Value> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert!(
        frames.iter().any(|frame| {
            frame.get("method").and_then(Value::as_str) == Some("event")
                && frame
                    .get("params")
                    .and_then(|params| params.get("type"))
                    .and_then(Value::as_str)
                    == Some("gateway.ready")
        }),
        "missing gateway.ready event in stdout:\n{stdout}"
    );
    assert!(
        frames.iter().any(|frame| {
            frame.get("id").and_then(Value::as_str) == Some("create")
                && frame
                    .get("result")
                    .and_then(|result| result.get("session_id"))
                    .and_then(Value::as_str)
                    .is_some()
        }),
        "missing native session.create response in stdout:\n{stdout}"
    );
    let tools = frames
        .iter()
        .find(|frame| frame.get("id").and_then(Value::as_str) == Some("tools"))
        .expect("missing tools.list response");
    assert_eq!(tools["error"]["code"], json!(5019));
    assert!(
        tools["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("tui_gateway.worker is disabled"),
        "unexpected tools.list error: {tools}"
    );
}
