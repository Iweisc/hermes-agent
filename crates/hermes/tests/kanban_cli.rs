use std::fs;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use rusqlite::Connection;
use serde_json::Value;
use tempfile::TempDir;

fn hermes(home: &Path, args: &[&str]) -> Output {
    hermes_with_env(home, args, &[])
}

fn hermes_with_env(home: &Path, args: &[&str], envs: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hermes"))
        .env("HERMES_HOME", home)
        .envs(envs.iter().copied())
        .args(args)
        .output()
        .expect("run hermes")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn assert_ok(output: &Output) {
    assert!(
        output.status.success(),
        "status={:?}\nstdout=\n{}\nstderr=\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn parse_json_output(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("json output")
}

fn open_db(home: &Path) -> Connection {
    Connection::open(home.join("kanban.db")).expect("open default kanban db")
}

#[test]
fn boards_create_switch_and_isolate_tasks() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(&home).unwrap();

    let output = hermes(&home, &["kanban", "boards", "create", "alpha", "--switch"]);
    assert_ok(&output);

    let output = hermes(
        &home,
        &["kanban", "create", "Task A", "--assignee", "dev", "--json"],
    );
    assert_ok(&output);
    let alpha_task = parse_json_output(&output);

    let output = hermes(&home, &["kanban", "boards", "create", "beta"]);
    assert_ok(&output);
    let output = hermes(
        &home,
        &[
            "kanban",
            "--board",
            "beta",
            "create",
            "Task B",
            "--assignee",
            "dev",
            "--json",
        ],
    );
    assert_ok(&output);
    let beta_task = parse_json_output(&output);

    let output = hermes(&home, &["kanban", "boards", "list", "--json"]);
    assert_ok(&output);
    let boards = parse_json_output(&output).as_array().unwrap().clone();
    assert!(boards.iter().any(|board| board["slug"] == "alpha"));
    assert!(boards.iter().any(|board| board["slug"] == "beta"));
    assert!(
        boards
            .iter()
            .any(|board| board["slug"] == "alpha" && board["current"] == true)
    );

    let output = hermes(&home, &["kanban", "--board", "alpha", "list", "--json"]);
    assert_ok(&output);
    let alpha_list = parse_json_output(&output).as_array().unwrap().clone();
    assert_eq!(alpha_list.len(), 1);
    assert_eq!(alpha_list[0]["id"], alpha_task["id"]);

    let output = hermes(&home, &["kanban", "--board", "beta", "list", "--json"]);
    assert_ok(&output);
    let beta_list = parse_json_output(&output).as_array().unwrap().clone();
    assert_eq!(beta_list.len(), 1);
    assert_eq!(beta_list[0]["id"], beta_task["id"]);
}

#[test]
fn task_workflow_promotes_children_and_builds_context() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(&home).unwrap();

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Parent",
            "--assignee",
            "alice",
            "--json",
        ],
    );
    assert_ok(&output);
    let parent_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Child",
            "--assignee",
            "bob",
            "--parent",
            &parent_id,
            "--json",
        ],
    );
    assert_ok(&output);
    let child = parse_json_output(&output);
    let child_id = child["id"].as_str().unwrap().to_string();
    assert_eq!(child["status"], "todo");

    let output = hermes(
        &home,
        &["kanban", "comment", &child_id, "remember", "the", "handoff"],
    );
    assert_ok(&output);
    let output = hermes(&home, &["kanban", "claim", &parent_id]);
    assert_ok(&output);
    let output = hermes(
        &home,
        &[
            "kanban",
            "complete",
            &parent_id,
            "--summary",
            "Parent finished cleanly",
        ],
    );
    assert_ok(&output);

    let output = hermes(&home, &["kanban", "show", &child_id, "--json"]);
    assert_ok(&output);
    let show = parse_json_output(&output);
    assert_eq!(show["task"]["status"], "ready");
    assert_eq!(show["comments"].as_array().unwrap().len(), 1);

    let output = hermes(&home, &["kanban", "context", &child_id]);
    assert_ok(&output);
    let context = stdout(&output);
    assert!(context.contains("Parent finished cleanly"));
    assert!(context.contains("remember the handoff"));
}

#[test]
fn reclaim_reassign_notify_log_gc_and_stats_round_trip() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(&home).unwrap();

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Recover Me",
            "--assignee",
            "orig",
            "--json",
        ],
    );
    assert_ok(&output);
    let task_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();

    let output = hermes(&home, &["kanban", "claim", &task_id]);
    assert_ok(&output);
    let output = hermes(
        &home,
        &[
            "kanban",
            "reassign",
            &task_id,
            "newbie",
            "--reclaim",
            "--reason",
            "switch model",
        ],
    );
    assert_ok(&output);

    let output = hermes(&home, &["kanban", "runs", &task_id, "--json"]);
    assert_ok(&output);
    let runs = parse_json_output(&output).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["outcome"], "reclaimed");

    let output = hermes(
        &home,
        &[
            "kanban",
            "notify-subscribe",
            &task_id,
            "--platform",
            "telegram",
            "--chat-id",
            "999",
        ],
    );
    assert_ok(&output);
    let output = hermes(&home, &["kanban", "notify-list", "--json"]);
    assert_ok(&output);
    let subs = parse_json_output(&output).as_array().unwrap().clone();
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0]["task_id"], task_id);

    let output = hermes(&home, &["kanban", "stats", "--json"]);
    assert_ok(&output);
    let stats = parse_json_output(&output);
    assert_eq!(stats["by_status"]["ready"], 1);
    assert_eq!(stats["by_assignee"]["newbie"]["ready"], 1);

    let log_dir = home.join("kanban").join("logs");
    fs::create_dir_all(&log_dir).unwrap();
    let log_path = log_dir.join(format!("{task_id}.log"));
    fs::write(&log_path, "line one\nline two\n").unwrap();
    let output = hermes(&home, &["kanban", "log", &task_id]);
    assert_ok(&output);
    assert!(stdout(&output).contains("line two"));

    std::thread::sleep(Duration::from_secs(1));

    let output = hermes(
        &home,
        &[
            "kanban",
            "notify-unsubscribe",
            &task_id,
            "--platform",
            "telegram",
            "--chat-id",
            "999",
        ],
    );
    assert_ok(&output);

    let output = hermes(
        &home,
        &[
            "kanban",
            "gc",
            "--event-retention-days",
            "0",
            "--log-retention-days",
            "0",
        ],
    );
    assert_ok(&output);
    assert!(!log_path.exists());

    let conn = open_db(&home);
    let assignee: String = conn
        .query_row(
            "SELECT assignee FROM tasks WHERE id = ?",
            [&task_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(assignee, "newbie");
    let subs_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM kanban_notify_subs", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(subs_count, 0);
}

#[test]
fn archive_heartbeat_assignees_and_diagnostics_work() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(home.join("profiles").join("writer")).unwrap();

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Heartbeat Me",
            "--assignee",
            "worker",
            "--json",
        ],
    );
    assert_ok(&output);
    let heartbeat_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ok(&hermes(&home, &["kanban", "claim", &heartbeat_id]));
    let output = hermes(
        &home,
        &["kanban", "heartbeat", &heartbeat_id, "--note", "step 42"],
    );
    assert_ok(&output);
    assert!(stdout(&output).contains("Heartbeat recorded"));

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Archive Me",
            "--assignee",
            "board-only",
            "--json",
        ],
    );
    assert_ok(&output);
    let archived_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();
    let output = hermes(&home, &["kanban", "archive", &archived_id]);
    assert_ok(&output);
    assert!(stdout(&output).contains("Archived"));

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Board Only",
            "--assignee",
            "board-only",
            "--json",
        ],
    );
    assert_ok(&output);

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Blocked Task",
            "--assignee",
            "writer",
            "--json",
        ],
    );
    assert_ok(&output);
    let blocked_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();
    let conn = open_db(&home);
    conn.execute(
        "UPDATE tasks SET status = 'blocked' WHERE id = ?",
        [&blocked_id],
    )
    .unwrap();
    let stale_ts = chrono::Local::now().timestamp() - 48 * 3600;
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'blocked', '{\"reason\":\"need input\"}', ?)",
        rusqlite::params![blocked_id, stale_ts],
    )
    .unwrap();
    let hb_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'heartbeat'",
            [&heartbeat_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(hb_count, 1);
    drop(conn);

    let output = hermes(&home, &["kanban", "assignees", "--json"]);
    assert_ok(&output);
    let assignees = parse_json_output(&output).as_array().unwrap().clone();
    assert!(
        assignees
            .iter()
            .any(|row| row["name"] == "writer" && row["on_disk"] == true)
    );
    assert!(
        assignees
            .iter()
            .any(|row| row["name"] == "board-only" && row["on_disk"] == false)
    );

    let output = hermes(&home, &["kanban", "diagnostics", "--json"]);
    assert_ok(&output);
    let diagnostics = parse_json_output(&output).as_array().unwrap().clone();
    assert!(diagnostics.iter().any(|entry| {
        entry["task_id"] == blocked_id
            && entry["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["kind"] == "stuck_in_blocked")
    }));
}

#[test]
fn watch_streams_new_events() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(&home).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_hermes"))
        .env("HERMES_HOME", &home)
        .args(["kanban", "watch", "--interval", "0.1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn watch");

    std::thread::sleep(Duration::from_millis(300));
    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Watched Task",
            "--assignee",
            "dev",
            "--json",
        ],
    );
    assert_ok(&output);

    std::thread::sleep(Duration::from_millis(700));
    let _ = child.kill();
    let output = child.wait_with_output().expect("wait for watch");
    let watched = String::from_utf8_lossy(&output.stdout);
    assert!(watched.contains("Watching kanban events"));
    assert!(watched.contains("created"), "watch output:\n{watched}");
    assert!(
        watched.contains("Watched Task") || watched.contains("t_"),
        "watch output:\n{watched}"
    );
}

#[test]
fn python_surface_verbs_and_aliases_work() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    fs::create_dir_all(home.join("profiles").join("researcher")).unwrap();

    let output = hermes(&home, &["kanban", "init"]);
    assert_ok(&output);
    let init_text = stdout(&output);
    assert!(init_text.contains("Kanban DB initialized"));
    assert!(init_text.contains("researcher"));

    let output = hermes(&home, &["kanban", "boards", "new", "alpha", "--switch"]);
    assert_ok(&output);
    let output = hermes(&home, &["kanban", "boards", "ls", "--json"]);
    assert_ok(&output);
    let boards = parse_json_output(&output).as_array().unwrap().clone();
    assert!(boards.iter().any(|board| board["slug"] == "alpha"));
    let output = hermes(&home, &["kanban", "boards", "current"]);
    assert_ok(&output);
    assert_eq!(stdout(&output), "alpha");

    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Alpha Parent",
            "--assignee",
            "researcher",
            "--json",
        ],
    );
    assert_ok(&output);
    let parent_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();
    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Alpha Child",
            "--assignee",
            "researcher",
            "--parent",
            &parent_id,
            "--json",
        ],
    );
    assert_ok(&output);
    let child_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();
    let output = hermes(
        &home,
        &[
            "kanban",
            "create",
            "Alpha Extra",
            "--assignee",
            "researcher",
            "--json",
        ],
    );
    assert_ok(&output);
    let extra_id = parse_json_output(&output)["id"]
        .as_str()
        .unwrap()
        .to_string();

    let output = hermes(&home, &["kanban", "ls"]);
    assert_ok(&output);
    assert!(stdout(&output).contains(&parent_id));

    let output = hermes_with_env(
        &home,
        &["kanban", "list", "--mine", "--json"],
        &[("HERMES_PROFILE", "researcher")],
    );
    assert_ok(&output);
    let mine = parse_json_output(&output).as_array().unwrap().clone();
    assert_eq!(mine.len(), 3);

    let output = hermes(
        &home,
        &[
            "kanban",
            "complete",
            &parent_id,
            "--result",
            "done",
            "--summary",
            "handoff",
            "--metadata",
            "{\"changed_files\":[\"src/lib.rs\"]}",
        ],
    );
    assert_ok(&output);
    let output = hermes(
        &home,
        &[
            "kanban",
            "edit",
            &parent_id,
            "--result",
            "done better",
            "--summary",
            "better handoff",
            "--metadata",
            "{\"tests_run\":3}",
        ],
    );
    assert_ok(&output);
    let output = hermes(&home, &["kanban", "show", &parent_id, "--json"]);
    assert_ok(&output);
    let show = parse_json_output(&output);
    assert_eq!(show["task"]["result"], "done better");
    assert_eq!(show["latest_summary"], "better handoff");
    assert!(
        show["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["kind"] == "edited")
    );

    let output = hermes(
        &home,
        &[
            "kanban", "block", &child_id, "need", "input", "--ids", &extra_id,
        ],
    );
    assert_ok(&output);
    let output = hermes(&home, &["kanban", "unblock", &child_id, &extra_id]);
    assert_ok(&output);

    let mut tail = Command::new(env!("CARGO_BIN_EXE_hermes"))
        .env("HERMES_HOME", &home)
        .args(["kanban", "tail", &parent_id, "--interval", "0.1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn tail");
    std::thread::sleep(Duration::from_millis(300));
    let output = hermes(&home, &["kanban", "comment", &parent_id, "tail", "event"]);
    assert_ok(&output);
    std::thread::sleep(Duration::from_millis(500));
    let _ = tail.kill();
    let output = tail.wait_with_output().expect("wait for tail");
    let tailed = String::from_utf8_lossy(&output.stdout);
    assert!(tailed.contains("Tailing events"));
    assert!(tailed.contains("commented"), "tail output:\n{tailed}");

    let output = hermes(&home, &["kanban", "dispatch", "--dry-run", "--json"]);
    assert_ok(&output);
    let dispatch = parse_json_output(&output);
    assert!(dispatch.get("promoted").is_some());
    assert!(dispatch.get("spawned").is_some());

    let output = hermes(&home, &["kanban", "diag"]);
    assert_ok(&output);
    assert!(!stdout(&output).is_empty());

    let log_dir = home
        .join("kanban")
        .join("boards")
        .join("alpha")
        .join("logs");
    fs::create_dir_all(&log_dir).unwrap();
    let log_path = log_dir.join(format!("{parent_id}.log"));
    fs::write(&log_path, "abcdefghij").unwrap();
    let output = hermes(&home, &["kanban", "log", &parent_id, "--tail", "4"]);
    assert_ok(&output);
    assert_eq!(stdout(&output), "ghij");

    let output = hermes(&home, &["kanban", "daemon"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("deprecated") || stderr.contains("gateway start"));
}
