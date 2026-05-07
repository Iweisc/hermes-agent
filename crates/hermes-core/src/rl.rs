use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use serde_yaml::Value as YamlValue;

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_WANDB_PROJECT: &str = "atropos-tinker";
const DEFAULT_STATUS_INTERVAL_SECS: u64 = 30 * 60;
const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 600;
const DEFAULT_NUM_STEPS: u64 = 3;
const DEFAULT_GROUP_SIZE: u64 = 16;
const MAX_NUM_STEPS: u64 = 50;
const MAX_GROUP_SIZE: u64 = 128;
const WAIT_AFTER_KILL_MILLIS: u64 = 1_000;
const LOG_TAIL_LINES: usize = 40;
const LOCKED_ENV_FIELDS: &[(&str, ValueKind)] = &[
    ("tokenizer_name", ValueKind::String),
    ("rollout_server_url", ValueKind::String),
    ("use_wandb", ValueKind::Bool),
    ("max_token_length", ValueKind::Number),
    ("max_num_workers", ValueKind::Number),
    ("worker_timeout", ValueKind::Number),
    ("total_steps", ValueKind::Number),
    ("steps_per_eval", ValueKind::Number),
    ("max_batches_offpolicy", ValueKind::Number),
    ("inference_weight", ValueKind::Number),
    ("eval_limit_ratio", ValueKind::Number),
];
const TEST_MODELS: &[(&str, &str, &str)] = &[
    ("qwen/qwen3-8b", "Qwen3 8B", "small"),
    ("z-ai/glm-4.7-flash", "GLM-4.7 Flash", "medium"),
    ("minimax/minimax-m2.7", "MiniMax M2.7", "large"),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnvironmentInfo {
    name: String,
    class_name: String,
    file_path: PathBuf,
    description: String,
    default_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ConfigField {
    name: String,
    default: Value,
    locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SelectionState {
    current_env: Option<String>,
    current_config: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedProcess {
    pid: i32,
    pgid: i32,
    command: String,
    cwd: String,
    log_path: PathBuf,
    status_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunState {
    run_id: String,
    mode: String,
    environment: String,
    config: BTreeMap<String, Value>,
    status: String,
    error_message: String,
    wandb_project: String,
    wandb_run_name: String,
    created_at: f64,
    started_at: Option<f64>,
    config_path: Option<PathBuf>,
    output_file: Option<PathBuf>,
    api_process: Option<ManagedProcess>,
    trainer_process: Option<ManagedProcess>,
    env_process: Option<ManagedProcess>,
    last_status_check: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    String,
    Bool,
    Number,
}

#[derive(Debug)]
struct ForegroundOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

#[derive(Debug)]
struct InferenceSummary {
    steps: Value,
    steps_tested: u64,
    total_completions: u64,
    correct_completions: u64,
    accuracy: f64,
    steps_with_correct: u64,
    step_success_rate: f64,
}

pub fn rl_available() -> bool {
    python3_available()
        && env_var_present("TINKER_API_KEY")
        && env_var_present("WANDB_API_KEY")
        && !discover_environments().is_empty()
}

pub fn rl_list_environments_schema() -> Value {
    json!({
        "name": "rl_list_environments",
        "description": "List available RL environments discovered from the repo's environments tree. Returns environment names, file paths, descriptions, and any default config path.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn rl_select_environment_schema() -> Value {
    json!({
        "name": "rl_select_environment",
        "description": "Select an RL environment for later config edits, training runs, or process-mode inference tests.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Environment name from rl_list_environments."
                }
            },
            "required": ["name"]
        }
    })
}

pub fn rl_get_current_config_schema() -> Value {
    json!({
        "name": "rl_get_current_config",
        "description": "Show the selected RL environment's configurable and locked fields, along with their current values.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn rl_edit_config_schema() -> Value {
    json!({
        "name": "rl_edit_config",
        "description": "Update one configurable RL environment field after selecting an environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "Field name from rl_get_current_config."
                },
                "value": {
                    "description": "New value for the selected field."
                }
            },
            "required": ["field", "value"]
        }
    })
}

pub fn rl_start_training_schema() -> Value {
    json!({
        "name": "rl_start_training",
        "description": "Start an RL training run for the selected environment. Writes a merged config and launches the local run-api, trainer, and environment processes when those commands are available.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn rl_check_status_schema() -> Value {
    json!({
        "name": "rl_check_status",
        "description": "Inspect one RL run's process state, logs, and high-level status. Like the Python tool, this is rate limited by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run id returned by rl_start_training."
                }
            },
            "required": ["run_id"]
        }
    })
}

pub fn rl_stop_training_schema() -> Value {
    json!({
        "name": "rl_stop_training",
        "description": "Stop a running RL training run and terminate any tracked child processes.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run id returned by rl_start_training."
                }
            },
            "required": ["run_id"]
        }
    })
}

pub fn rl_get_results_schema() -> Value {
    json!({
        "name": "rl_get_results",
        "description": "Return persisted metadata, log tails, and final status for one RL run.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run id returned by rl_start_training."
                }
            },
            "required": ["run_id"]
        }
    })
}

pub fn rl_list_runs_schema() -> Value {
    json!({
        "name": "rl_list_runs",
        "description": "List known RL runs from Hermes home with their status and wandb run names.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn rl_test_inference_schema() -> Value {
    json!({
        "name": "rl_test_inference",
        "description": "Run a small process-mode inference sanity check for the selected RL environment. This validates environment loading and parses saved JSONL scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "num_steps": {
                    "type": "integer",
                    "description": "Number of process-mode steps to run.",
                    "default": DEFAULT_NUM_STEPS
                },
                "group_size": {
                    "type": "integer",
                    "description": "Number of completions per step.",
                    "default": DEFAULT_GROUP_SIZE
                },
                "models": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Optional OpenRouter model ids. Defaults to a small built-in test set."
                }
            },
            "required": []
        }
    })
}

pub fn handle_rl_list_environments(_args: &Value, _runtime: &ToolRuntime) -> String {
    let environments = discover_environments();
    tool_result(json!({
        "success": true,
        "environments": environments
            .iter()
            .map(environment_json)
            .collect::<Vec<_>>(),
        "count": environments.len(),
        "tips": [
            "Use rl_select_environment(name) to select an environment.",
            "Read the file_path with file tools to inspect rewards, datasets, and verifiers.",
            "If default_config_path is present, read it for baseline settings."
        ]
    }))
}

pub fn handle_rl_select_environment(args: &Value, runtime: &ToolRuntime) -> String {
    let name = match required_non_empty_string(args, "name") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let environments = discover_environments();
    let Some(environment) = environments.into_iter().find(|item| item.name == name) else {
        return tool_result(json!({
            "success": false,
            "error": format!("Environment '{name}' not found"),
            "available": discover_environments().into_iter().map(|item| item.name).collect::<Vec<_>>(),
        }));
    };

    let fields = config_fields_for_environment(&environment);
    let mut current_config = BTreeMap::new();
    for field in fields.values() {
        if !field.locked {
            current_config.insert(field.name.clone(), field.default.clone());
        }
    }
    current_config.insert(
        "wandb_name".to_string(),
        Value::String(format!("{}-{}", environment.name, timestamp_slug())),
    );
    current_config
        .entry("wandb_project".to_string())
        .or_insert_with(|| Value::String(DEFAULT_WANDB_PROJECT.to_string()));

    let state = SelectionState {
        current_env: Some(environment.name.clone()),
        current_config,
    };
    if let Err(error) = save_selection_state(runtime, &state) {
        return tool_result(json!({
            "success": false,
            "error": error,
        }));
    }

    tool_result(json!({
        "success": true,
        "message": format!("Selected environment: {}", environment.name),
        "environment": environment.name,
        "file_path": environment.file_path.display().to_string(),
        "default_config_path": environment.default_config_path.map(|path| path.display().to_string()),
        "configurable_field_count": state.current_config.len(),
    }))
}

pub fn handle_rl_get_current_config(_args: &Value, runtime: &ToolRuntime) -> String {
    let state = match load_selection_state(runtime) {
        Ok(state) => state,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    let Some(current_env) = state.current_env else {
        return tool_result(json!({
            "success": false,
            "error": "No environment selected. Use rl_select_environment(name) first.",
        }));
    };
    let environments = discover_environments();
    let Some(environment) = environments
        .into_iter()
        .find(|item| item.name == current_env)
    else {
        return tool_result(json!({
            "success": false,
            "error": format!("Selected environment '{}' is no longer available", current_env),
        }));
    };
    let fields = config_fields_for_environment(&environment);
    let mut configurable = Vec::new();
    let mut locked = Vec::new();
    for field in fields.values() {
        let current_value = state
            .current_config
            .get(&field.name)
            .cloned()
            .unwrap_or_else(|| field.default.clone());
        let mut entry = json!({
            "name": field.name,
            "default": field.default,
            "current_value": current_value,
        });
        if field.locked {
            entry["locked_value"] = locked_env_value(&field.name);
            locked.push(entry);
        } else {
            configurable.push(entry);
        }
    }
    tool_result(json!({
        "success": true,
        "environment": environment.name,
        "configurable_fields": configurable,
        "locked_fields": locked,
        "tip": "Use rl_edit_config(field, value) to change a configurable field.",
    }))
}

pub fn handle_rl_edit_config(args: &Value, runtime: &ToolRuntime) -> String {
    let field = match required_non_empty_string(args, "field") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let Some(raw_value) = args.get("value") else {
        return tool_error("value is required");
    };
    let mut state = match load_selection_state(runtime) {
        Ok(state) => state,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    let Some(current_env) = state.current_env.clone() else {
        return tool_result(json!({
            "success": false,
            "error": "No environment selected. Use rl_select_environment(name) first.",
        }));
    };
    let environments = discover_environments();
    let Some(environment) = environments
        .into_iter()
        .find(|item| item.name == current_env)
    else {
        return tool_result(json!({
            "success": false,
            "error": format!("Selected environment '{}' is no longer available", current_env),
        }));
    };
    let fields = config_fields_for_environment(&environment);
    let Some(config_field) = fields.get(&field) else {
        return tool_result(json!({
            "success": false,
            "error": format!("Unknown field '{}'", field),
            "available_fields": fields.keys().cloned().collect::<Vec<_>>(),
        }));
    };
    if config_field.locked {
        return tool_result(json!({
            "success": false,
            "error": format!("Field '{}' is locked and cannot be changed", field),
            "locked_value": locked_env_value(&field),
        }));
    }
    let coerced = match coerce_value(raw_value.clone(), &config_field.default) {
        Ok(value) => value,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    state.current_config.insert(field.clone(), coerced.clone());
    if let Err(error) = save_selection_state(runtime, &state) {
        return tool_result(json!({ "success": false, "error": error }));
    }
    tool_result(json!({
        "success": true,
        "message": format!("Updated {} = {}", field, value_preview(&coerced)),
        "field": field,
        "value": coerced,
        "config": state.current_config,
    }))
}

pub fn handle_rl_start_training(_args: &Value, runtime: &ToolRuntime) -> String {
    let state = match load_selection_state(runtime) {
        Ok(state) => state,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    let Some(current_env) = state.current_env.clone() else {
        return tool_result(json!({
            "success": false,
            "error": "No environment selected. Use rl_select_environment(name) first.",
        }));
    };
    if !env_var_present("TINKER_API_KEY") {
        return tool_result(json!({
            "success": false,
            "error": "TINKER_API_KEY not set. Add it to your Hermes environment before starting training.",
        }));
    }
    if !env_var_present("WANDB_API_KEY") {
        return tool_result(json!({
            "success": false,
            "error": "WANDB_API_KEY not set. Add it to your Hermes environment before starting training.",
        }));
    }
    let environments = discover_environments();
    let Some(environment) = environments
        .into_iter()
        .find(|item| item.name == current_env)
    else {
        return tool_result(json!({
            "success": false,
            "error": format!("Selected environment '{}' is no longer available", current_env),
        }));
    };
    if let Err(error) = ensure_rl_dirs(runtime) {
        return tool_result(json!({ "success": false, "error": error }));
    }

    let run_api_command = match find_run_api_command() {
        Ok(command) => command,
        Err(error) => {
            return tool_result(json!({
                "success": false,
                "error": error,
            }));
        }
    };
    let launch_training = match find_launch_training_script() {
        Ok(path) => path,
        Err(error) => {
            return tool_result(json!({
                "success": false,
                "error": error,
            }));
        }
    };

    let run_id = format!("rl_{:x}", unix_ts_nanos());
    let config_path = rl_configs_root(runtime).join(format!("run_{run_id}.yaml"));
    let logs_dir = rl_logs_root(runtime);
    if let Err(error) = fs::create_dir_all(&logs_dir) {
        return tool_result(json!({
            "success": false,
            "error": format!("creating RL log directory {} failed: {error}", logs_dir.display()),
        }));
    }
    let merged_yaml = build_training_config_yaml(&state.current_config);
    if let Err(error) = write_yaml_file(&config_path, &merged_yaml) {
        return tool_result(json!({ "success": false, "error": error }));
    }

    let training_cwd = training_root();
    let api_log = logs_dir.join(format!("api_{run_id}.log"));
    let trainer_log = logs_dir.join(format!("trainer_{run_id}.log"));
    let env_log = logs_dir.join(format!("env_{run_id}.log"));
    let api_status = logs_dir.join(format!("api_{run_id}.status"));
    let trainer_status = logs_dir.join(format!("trainer_{run_id}.status"));
    let env_status = logs_dir.join(format!("env_{run_id}.status"));
    let extra_env = vec![
        (
            "TINKER_API_KEY",
            env::var("TINKER_API_KEY").unwrap_or_default(),
        ),
        (
            "WANDB_API_KEY",
            env::var("WANDB_API_KEY").unwrap_or_default(),
        ),
    ];

    let api_process = match spawn_background_program(
        &run_api_command[0],
        &run_api_command[1..],
        &training_cwd,
        &api_log,
        &api_status,
        &extra_env,
    ) {
        Ok(process) => process,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    let trainer_process = match spawn_background_program(
        "python3",
        &[
            launch_training.display().to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
        ],
        &training_cwd,
        &trainer_log,
        &trainer_status,
        &extra_env,
    ) {
        Ok(process) => process,
        Err(error) => {
            let _ = terminate_process(&api_process);
            return tool_result(json!({ "success": false, "error": error }));
        }
    };
    let env_process = match spawn_background_program(
        "python3",
        &[
            environment.file_path.display().to_string(),
            "serve".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
        ],
        &training_cwd,
        &env_log,
        &env_status,
        &extra_env,
    ) {
        Ok(process) => process,
        Err(error) => {
            let _ = terminate_process(&trainer_process);
            let _ = terminate_process(&api_process);
            return tool_result(json!({ "success": false, "error": error }));
        }
    };

    let wandb_project = state
        .current_config
        .get("wandb_project")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_WANDB_PROJECT)
        .to_string();
    let wandb_run_name = format!("{}-{}", current_env, run_id.trim_start_matches("rl_"));
    let run = RunState {
        run_id: run_id.clone(),
        mode: "training".to_string(),
        environment: current_env.clone(),
        config: state.current_config.clone(),
        status: "starting".to_string(),
        error_message: String::new(),
        wandb_project: wandb_project.clone(),
        wandb_run_name: wandb_run_name.clone(),
        created_at: unix_ts_secs(),
        started_at: Some(unix_ts_secs()),
        config_path: Some(config_path.clone()),
        output_file: None,
        api_process: Some(api_process),
        trainer_process: Some(trainer_process),
        env_process: Some(env_process),
        last_status_check: None,
    };
    if let Err(error) = save_run_state(runtime, &run) {
        return tool_result(json!({ "success": false, "error": error }));
    }

    tool_result(json!({
        "success": true,
        "run_id": run_id,
        "status": "starting",
        "environment": current_env,
        "config": state.current_config,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "config_path": config_path.display().to_string(),
        "logs": {
            "api": api_log.display().to_string(),
            "trainer": trainer_log.display().to_string(),
            "env": env_log.display().to_string(),
        },
        "message": "Training starting. Use rl_check_status(run_id) to monitor progress.",
    }))
}

pub fn handle_rl_check_status(args: &Value, runtime: &ToolRuntime) -> String {
    let run_id = match required_non_empty_string(args, "run_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mut run = match load_run_state(runtime, &run_id) {
        Ok(run) => run,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };

    let now = unix_ts_secs();
    let min_interval = min_status_interval_secs() as f64;
    if let Some(last_check) = run.last_status_check {
        let elapsed = now - last_check;
        if min_interval > 0.0 && elapsed < min_interval {
            let remaining = min_interval - elapsed;
            return tool_result(json!({
                "success": true,
                "rate_limited": true,
                "run_id": run.run_id,
                "message": format!("Rate limited. Next check available in {:.0} minutes.", remaining / 60.0),
                "next_check_in_seconds": remaining,
            }));
        }
    }
    run.last_status_check = Some(now);
    refresh_run_status(&mut run);
    if let Err(error) = save_run_state(runtime, &run) {
        return tool_result(json!({ "success": false, "error": error }));
    }
    tool_result(status_payload(&run, true))
}

pub fn handle_rl_stop_training(args: &Value, runtime: &ToolRuntime) -> String {
    let run_id = match required_non_empty_string(args, "run_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mut run = match load_run_state(runtime, &run_id) {
        Ok(run) => run,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    if run.status != "running" && run.status != "starting" {
        return tool_result(json!({
            "success": true,
            "message": format!("Run '{}' is not running (status: {})", run_id, run.status),
            "run_id": run_id,
            "status": run.status,
        }));
    }
    if let Some(process) = run.env_process.as_ref() {
        let _ = terminate_process(process);
        ensure_status_marker(process, 143);
    }
    if let Some(process) = run.trainer_process.as_ref() {
        let _ = terminate_process(process);
        ensure_status_marker(process, 143);
    }
    if let Some(process) = run.api_process.as_ref() {
        let _ = terminate_process(process);
        ensure_status_marker(process, 143);
    }
    run.status = "stopped".to_string();
    run.error_message.clear();
    if let Err(error) = save_run_state(runtime, &run) {
        return tool_result(json!({ "success": false, "error": error }));
    }
    tool_result(json!({
        "success": true,
        "message": format!("Stopped training run '{}'", run_id),
        "run_id": run_id,
        "status": run.status,
    }))
}

pub fn handle_rl_get_results(args: &Value, runtime: &ToolRuntime) -> String {
    let run_id = match required_non_empty_string(args, "run_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mut run = match load_run_state(runtime, &run_id) {
        Ok(run) => run,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    refresh_run_status(&mut run);
    if let Err(error) = save_run_state(runtime, &run) {
        return tool_result(json!({ "success": false, "error": error }));
    }
    let mut payload = status_payload(&run, false);
    payload["result_logs"] = json!({
        "api_tail": run.api_process.as_ref().map(|process| read_last_lines(&process.log_path, LOG_TAIL_LINES)).unwrap_or_default(),
        "trainer_tail": run.trainer_process.as_ref().map(|process| read_last_lines(&process.log_path, LOG_TAIL_LINES)).unwrap_or_default(),
        "env_tail": run.env_process.as_ref().map(|process| read_last_lines(&process.log_path, LOG_TAIL_LINES)).unwrap_or_default(),
    });
    tool_result(payload)
}

pub fn handle_rl_list_runs(_args: &Value, runtime: &ToolRuntime) -> String {
    let mut runs = load_all_runs(runtime);
    runs.sort_by(|left, right| right.created_at.total_cmp(&left.created_at));
    tool_result(json!({
        "success": true,
        "runs": runs.iter().map(|run| {
            json!({
                "run_id": run.run_id,
                "mode": run.mode,
                "environment": run.environment,
                "status": run.status,
                "wandb_run_name": run.wandb_run_name,
                "created_at": run.created_at,
            })
        }).collect::<Vec<_>>(),
        "count": runs.len(),
    }))
}

pub fn handle_rl_test_inference(args: &Value, runtime: &ToolRuntime) -> String {
    let state = match load_selection_state(runtime) {
        Ok(state) => state,
        Err(error) => return tool_result(json!({ "success": false, "error": error })),
    };
    let Some(current_env) = state.current_env.clone() else {
        return tool_result(json!({
            "success": false,
            "error": "No environment selected. Use rl_select_environment(name) first.",
        }));
    };
    let openrouter_key = match env::var("OPENROUTER_API_KEY") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            return tool_result(json!({
                "success": false,
                "error": "OPENROUTER_API_KEY not set. Required for inference testing.",
            }));
        }
    };
    let num_steps = match optional_u64(args, "num_steps") {
        Ok(value) => value.unwrap_or(DEFAULT_NUM_STEPS),
        Err(error) => return tool_error(error),
    };
    if num_steps == 0 || num_steps > MAX_NUM_STEPS {
        return tool_result(json!({
            "success": false,
            "error": format!("num_steps must be between 1 and {}", MAX_NUM_STEPS),
        }));
    }
    let group_size = match optional_u64(args, "group_size") {
        Ok(value) => value.unwrap_or(DEFAULT_GROUP_SIZE),
        Err(error) => return tool_error(error),
    };
    if group_size == 0 || group_size > MAX_GROUP_SIZE {
        return tool_result(json!({
            "success": false,
            "error": format!("group_size must be between 1 and {}", MAX_GROUP_SIZE),
        }));
    }
    let models = match optional_string_array(args, "models") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let environments = discover_environments();
    let Some(environment) = environments
        .into_iter()
        .find(|item| item.name == current_env)
    else {
        return tool_result(json!({
            "success": false,
            "error": format!("Selected environment '{}' is no longer available", current_env),
        }));
    };
    let model_specs = if models.is_empty() {
        TEST_MODELS
            .iter()
            .map(|(id, name, scale)| ((*id).to_string(), (*name).to_string(), (*scale).to_string()))
            .collect::<Vec<_>>()
    } else {
        models
            .into_iter()
            .map(|id| (id.clone(), id, "custom".to_string()))
            .collect::<Vec<_>>()
    };

    let output_root = rl_logs_root(runtime).join("inference_tests");
    if let Err(error) = fs::create_dir_all(&output_root) {
        return tool_result(json!({
            "success": false,
            "error": format!("creating inference test directory {} failed: {error}", output_root.display()),
        }));
    }

    let mut model_results = Vec::new();
    for (model_id, model_name, scale) in model_specs {
        let model_safe = model_id.replace('/', "_");
        let output_file = output_root.join(format!(
            "test_{}_{}_{}.jsonl",
            current_env,
            model_safe,
            unix_ts_nanos()
        ));
        let log_file = output_root.join(format!(
            "test_{}_{}_{}.log",
            current_env,
            model_safe,
            unix_ts_nanos()
        ));
        let wandb_run_name = format!("test_inference_RSIAgent_{}_{}", current_env, short_id());
        let command = vec![
            environment.file_path.display().to_string(),
            "process".to_string(),
            "--env.total_steps".to_string(),
            num_steps.to_string(),
            "--env.group_size".to_string(),
            group_size.to_string(),
            "--env.use_wandb".to_string(),
            "true".to_string(),
            "--env.wandb_name".to_string(),
            wandb_run_name.clone(),
            "--env.data_path_to_save_groups".to_string(),
            output_file.display().to_string(),
            "--env.tokenizer_name".to_string(),
            "Qwen/Qwen3-8B".to_string(),
            "--env.max_token_length".to_string(),
            "8192".to_string(),
            "--env.max_num_workers".to_string(),
            "2048".to_string(),
            "--env.max_batches_offpolicy".to_string(),
            "3".to_string(),
            "--openai.base_url".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            "--openai.api_key".to_string(),
            openrouter_key.clone(),
            "--openai.model_name".to_string(),
            model_id.clone(),
            "--openai.server_type".to_string(),
            "openai".to_string(),
            "--openai.health_check".to_string(),
            "false".to_string(),
        ];
        let run = run_foreground_program(
            "python3",
            &command,
            &repo_root(),
            &[],
            inference_timeout_secs(),
        );
        let mut result = json!({
            "model": model_id,
            "name": model_name,
            "scale": scale,
            "wandb_run": wandb_run_name,
            "output_file": output_file.display().to_string(),
            "log_file": log_file.display().to_string(),
            "steps": Vec::<Value>::new(),
            "steps_tested": 0,
            "total_completions": 0,
            "correct_completions": 0,
        });

        match run {
            Ok(output) => {
                let log_text = format!(
                    "stdout:\n{}\n\nstderr:\n{}\n",
                    output.stdout.trim(),
                    output.stderr.trim()
                );
                let _ = fs::write(&log_file, log_text);
                if output.exit_code != 0 {
                    result["error"] =
                        Value::String(format!("Process exited with code {}", output.exit_code));
                    result["stderr"] = Value::String(output.stderr);
                    result["stdout"] = Value::String(output.stdout);
                } else if output_file.is_file() {
                    match parse_inference_output(&output_file) {
                        Ok(parsed) => {
                            result["steps"] = parsed.steps;
                            result["steps_tested"] = json!(parsed.steps_tested);
                            result["total_completions"] = json!(parsed.total_completions);
                            result["correct_completions"] = json!(parsed.correct_completions);
                            result["accuracy"] = json!(parsed.accuracy);
                            result["steps_with_correct"] = json!(parsed.steps_with_correct);
                            result["step_success_rate"] = json!(parsed.step_success_rate);
                        }
                        Err(error) => {
                            result["error"] = Value::String(error);
                        }
                    }
                } else {
                    result["error"] = Value::String(format!(
                        "Output file not created: {}",
                        output_file.display()
                    ));
                }
            }
            Err(error) => {
                result["error"] = Value::String(error);
            }
        }

        if result.get("accuracy").is_none() {
            result["accuracy"] = json!(0.0);
        }
        if result.get("step_success_rate").is_none() {
            result["step_success_rate"] = json!(0.0);
        }
        if result.get("steps_with_correct").is_none() {
            result["steps_with_correct"] = json!(0);
        }
        model_results.push(result);
    }

    let succeeded = model_results
        .iter()
        .filter(|item| {
            item.get("steps_tested")
                .and_then(Value::as_u64)
                .unwrap_or_default()
                > 0
        })
        .count();
    let avg_accuracy = {
        let accuracies = model_results
            .iter()
            .filter_map(|item| item.get("accuracy").and_then(Value::as_f64))
            .collect::<Vec<_>>();
        if accuracies.is_empty() {
            0.0
        } else {
            accuracies.iter().sum::<f64>() / accuracies.len() as f64
        }
    };
    tool_result(json!({
        "success": true,
        "environment": current_env,
        "environment_file": environment.file_path.display().to_string(),
        "test_config": {
            "num_steps": num_steps,
            "group_size": group_size,
            "rollouts_per_model": num_steps * group_size,
            "total_rollouts": num_steps * group_size * model_results.len() as u64,
        },
        "models_tested": model_results,
        "summary": {
            "steps_requested": num_steps,
            "models_tested": model_results.len(),
            "models_succeeded": succeeded,
            "avg_accuracy": avg_accuracy,
            "environment_working": succeeded > 0,
            "output_directory": output_root.display().to_string(),
        }
    }))
}

fn parse_inference_output(path: &Path) -> Result<InferenceSummary, String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "opening inference output {} failed: {error}",
            path.display()
        )
    })?;
    let reader = BufReader::new(file);
    let mut steps = Vec::new();
    let mut steps_tested = 0u64;
    let mut total_completions = 0u64;
    let mut correct_completions = 0u64;
    let mut steps_with_correct = 0u64;

    for line in reader.lines() {
        let line = line.map_err(|error| format!("reading inference output failed: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| format!("parsing inference output line failed: {error}"))?;
        let scores = value
            .get("scores")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let correct = scores
            .iter()
            .filter(|score| score.as_f64().unwrap_or_default() > 0.0)
            .count() as u64;
        steps_tested += 1;
        total_completions += scores.len() as u64;
        correct_completions += correct;
        if correct > 0 {
            steps_with_correct += 1;
        }
        steps.push(json!({
            "step": steps_tested,
            "completions": scores.len(),
            "correct": correct,
            "scores": scores,
        }));
    }

    let accuracy = if total_completions == 0 {
        0.0
    } else {
        correct_completions as f64 / total_completions as f64
    };
    let step_success_rate = if steps_tested == 0 {
        0.0
    } else {
        steps_with_correct as f64 / steps_tested as f64
    };
    Ok(InferenceSummary {
        steps: Value::Array(steps),
        steps_tested,
        total_completions,
        correct_completions,
        accuracy,
        steps_with_correct,
        step_success_rate,
    })
}

fn environment_json(environment: &EnvironmentInfo) -> Value {
    json!({
        "name": environment.name,
        "class_name": environment.class_name,
        "file_path": environment.file_path.display().to_string(),
        "description": environment.description,
        "default_config_path": environment
            .default_config_path
            .as_ref()
            .map(|path| path.display().to_string()),
    })
}

fn status_payload(run: &RunState, include_rate_limit_fields: bool) -> Value {
    let running_time_minutes = run
        .started_at
        .map(|started| (unix_ts_secs() - started) / 60.0)
        .unwrap_or_default();
    let mut payload = json!({
        "success": true,
        "run_id": run.run_id,
        "mode": run.mode,
        "status": run.status,
        "environment": run.environment,
        "running_time_minutes": running_time_minutes,
        "wandb_project": run.wandb_project,
        "wandb_run_name": run.wandb_run_name,
        "config_path": run.config_path.as_ref().map(|path| path.display().to_string()),
        "output_file": run.output_file.as_ref().map(|path| path.display().to_string()),
        "processes": {
            "api": process_json(run.api_process.as_ref()),
            "trainer": process_json(run.trainer_process.as_ref()),
            "env": process_json(run.env_process.as_ref()),
        },
    });
    if !run.error_message.is_empty() {
        payload["error"] = Value::String(run.error_message.clone());
    }
    if include_rate_limit_fields {
        payload["rate_limited"] = Value::Bool(false);
    }
    payload
}

fn process_json(process: Option<&ManagedProcess>) -> Value {
    let Some(process) = process else {
        return json!({ "state": "not_started" });
    };
    let exit_code = read_exit_code(&process.status_path);
    let running = exit_code.is_none() && process_alive(process.pid);
    json!({
        "state": if running {
            "running".to_string()
        } else if let Some(code) = exit_code {
            format!("exited ({code})")
        } else {
            "unknown".to_string()
        },
        "pid": process.pid,
        "log_path": process.log_path.display().to_string(),
        "status_path": process.status_path.display().to_string(),
        "exit_code": exit_code,
    })
}

fn refresh_run_status(run: &mut RunState) {
    if run.status == "stopped" || run.status == "completed" || run.status == "failed" {
        return;
    }
    let process_states = [
        run.api_process.as_ref(),
        run.trainer_process.as_ref(),
        run.env_process.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|process| {
        (
            process_alive(process.pid),
            read_exit_code(&process.status_path),
            process.command.clone(),
        )
    })
    .collect::<Vec<_>>();

    if let Some((_, Some(exit_code), command)) = process_states
        .iter()
        .find(|(_, exit_code, _)| exit_code.unwrap_or_default() != 0)
    {
        run.status = "failed".to_string();
        run.error_message = format!("Process '{}' exited with code {}", command, *exit_code);
        return;
    }
    if !process_states.is_empty()
        && process_states
            .iter()
            .all(|(_, exit_code, _)| exit_code == &Some(0))
    {
        run.status = "completed".to_string();
        run.error_message.clear();
        return;
    }
    if process_states.iter().any(|(alive, _, _)| *alive) {
        run.status = "running".to_string();
        run.error_message.clear();
        return;
    }
    run.status = "failed".to_string();
    if run.error_message.is_empty() {
        run.error_message = "All tracked RL processes exited unexpectedly.".to_string();
    }
}

fn config_fields_for_environment(environment: &EnvironmentInfo) -> BTreeMap<String, ConfigField> {
    let defaults = load_environment_defaults(environment);
    let mut fields = defaults
        .into_iter()
        .map(|(name, default)| {
            let locked = LOCKED_ENV_FIELDS
                .iter()
                .any(|(field_name, _)| *field_name == name);
            (
                name.clone(),
                ConfigField {
                    name,
                    default,
                    locked,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    fields
        .entry("wandb_project".to_string())
        .or_insert_with(|| ConfigField {
            name: "wandb_project".to_string(),
            default: Value::String(DEFAULT_WANDB_PROJECT.to_string()),
            locked: false,
        });
    fields
        .entry("wandb_name".to_string())
        .or_insert_with(|| ConfigField {
            name: "wandb_name".to_string(),
            default: Value::String(environment.name.clone()),
            locked: false,
        });
    fields
}

fn load_environment_defaults(environment: &EnvironmentInfo) -> BTreeMap<String, Value> {
    if let Some(path) = environment.default_config_path.as_ref()
        && path.is_file()
        && let Ok(defaults) = load_defaults_from_yaml(path)
    {
        return defaults;
    }
    if let Ok(text) = fs::read_to_string(&environment.file_path) {
        let parsed = parse_env_config_init_defaults(&text);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    BTreeMap::new()
}

fn load_defaults_from_yaml(path: &Path) -> Result<BTreeMap<String, Value>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("reading default config {} failed: {error}", path.display()))?;
    let yaml: YamlValue = serde_yaml::from_str(&text)
        .map_err(|error| format!("parsing default config {} failed: {error}", path.display()))?;
    let Some(env_map) = yaml
        .as_mapping()
        .and_then(|mapping| mapping.get(YamlValue::String("env".to_string())))
        .and_then(YamlValue::as_mapping)
    else {
        return Ok(BTreeMap::new());
    };
    let mut out = BTreeMap::new();
    for (key, value) in env_map {
        if let Some(key) = key.as_str() {
            out.insert(key.to_string(), yaml_to_json(value));
        }
    }
    Ok(out)
}

fn parse_env_config_init_defaults(text: &str) -> BTreeMap<String, Value> {
    let Some(start) = text.find("env_config =") else {
        return BTreeMap::new();
    };
    let Some(open_paren) = text[start..].find('(').map(|offset| start + offset) else {
        return BTreeMap::new();
    };
    let Some(inner) = extract_parenthesized_block(text, open_paren) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for piece in split_top_level(&inner, ',') {
        let trimmed = strip_comment(piece.trim());
        if trimmed.is_empty() {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            continue;
        }
        if let Some(value) = parse_python_literal(raw_value.trim()) {
            out.insert(key.to_string(), value);
        }
    }
    out
}

fn parse_python_literal(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "None" {
        return Some(Value::Null);
    }
    if trimmed == "True" {
        return Some(Value::Bool(true));
    }
    if trimmed == "False" {
        return Some(Value::Bool(false));
    }
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return Some(Value::String(unquote_python_string(trimmed)));
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = &trimmed[1..trimmed.len() - 1];
        let items = split_top_level(inner, ',')
            .into_iter()
            .filter_map(|item| parse_python_literal(item.trim()))
            .collect::<Vec<_>>();
        return Some(Value::Array(items));
    }
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut map = Map::new();
        for entry in split_top_level(inner, ',') {
            let Some((key, value)) = entry.split_once(':') else {
                continue;
            };
            let key = unquote_python_string(key.trim());
            if let Some(value) = parse_python_literal(value.trim()) {
                map.insert(key, value);
            }
        }
        return Some(Value::Object(map));
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return Some(json!(value));
    }
    if let Ok(value) = trimmed.parse::<u64>() {
        return Some(json!(value));
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return Some(json!(value));
    }
    None
}

fn yaml_to_json(value: &YamlValue) -> Value {
    match value {
        YamlValue::Null => Value::Null,
        YamlValue::Bool(value) => Value::Bool(*value),
        YamlValue::Number(value) => {
            if let Some(integer) = value.as_i64() {
                json!(integer)
            } else if let Some(integer) = value.as_u64() {
                json!(integer)
            } else if let Some(float) = value.as_f64() {
                json!(float)
            } else {
                Value::Null
            }
        }
        YamlValue::String(value) => Value::String(value.clone()),
        YamlValue::Sequence(values) => {
            Value::Array(values.iter().map(yaml_to_json).collect::<Vec<_>>())
        }
        YamlValue::Mapping(mapping) => {
            let mut object = Map::new();
            for (key, value) in mapping {
                if let Some(key) = key.as_str() {
                    object.insert(key.to_string(), yaml_to_json(value));
                }
            }
            Value::Object(object)
        }
        YamlValue::Tagged(tagged) => yaml_to_json(&tagged.value),
    }
}

fn build_training_config_yaml(current_config: &BTreeMap<String, Value>) -> YamlValue {
    let mut root = serde_yaml::Mapping::new();
    let mut env_map = serde_yaml::Mapping::new();
    for (field, kind) in LOCKED_ENV_FIELDS {
        env_map.insert(
            YamlValue::String((*field).to_string()),
            default_locked_yaml_value(field, *kind),
        );
    }
    for (field, value) in current_config {
        if field == "wandb_project" {
            continue;
        }
        if !value.is_null() {
            env_map.insert(YamlValue::String(field.clone()), json_to_yaml(value));
        }
    }
    root.insert(
        YamlValue::String("env".to_string()),
        YamlValue::Mapping(env_map),
    );
    root.insert(
        YamlValue::String("openai".to_string()),
        YamlValue::Sequence(vec![YamlValue::Mapping({
            let mut mapping = serde_yaml::Mapping::new();
            mapping.insert(
                YamlValue::String("model_name".to_string()),
                YamlValue::String("Qwen/Qwen3-8B".to_string()),
            );
            mapping.insert(
                YamlValue::String("base_url".to_string()),
                YamlValue::String("http://localhost:8001/v1".to_string()),
            );
            mapping.insert(
                YamlValue::String("api_key".to_string()),
                YamlValue::String("x".to_string()),
            );
            mapping.insert(
                YamlValue::String("weight".to_string()),
                YamlValue::Number(serde_yaml::Number::from(1)),
            );
            mapping.insert(
                YamlValue::String("num_requests_for_eval".to_string()),
                YamlValue::Number(serde_yaml::Number::from(256)),
            );
            mapping.insert(
                YamlValue::String("timeout".to_string()),
                YamlValue::Number(serde_yaml::Number::from(3600)),
            );
            mapping.insert(
                YamlValue::String("server_type".to_string()),
                YamlValue::String("sglang".to_string()),
            );
            mapping
        })]),
    );
    let mut tinker = serde_yaml::Mapping::new();
    tinker.insert(
        YamlValue::String("lora_rank".to_string()),
        YamlValue::Number(serde_yaml::Number::from(32)),
    );
    tinker.insert(
        YamlValue::String("learning_rate".to_string()),
        YamlValue::String("0.00004".to_string()),
    );
    tinker.insert(
        YamlValue::String("max_token_trainer_length".to_string()),
        YamlValue::Number(serde_yaml::Number::from(9000)),
    );
    tinker.insert(
        YamlValue::String("checkpoint_dir".to_string()),
        YamlValue::String("./temp/".to_string()),
    );
    tinker.insert(
        YamlValue::String("save_checkpoint_interval".to_string()),
        YamlValue::Number(serde_yaml::Number::from(25)),
    );
    tinker.insert(
        YamlValue::String("wandb_project".to_string()),
        YamlValue::String(
            current_config
                .get("wandb_project")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_WANDB_PROJECT)
                .to_string(),
        ),
    );
    root.insert(
        YamlValue::String("tinker".to_string()),
        YamlValue::Mapping(tinker),
    );
    root.insert(
        YamlValue::String("slurm".to_string()),
        YamlValue::Bool(false),
    );
    root.insert(
        YamlValue::String("testing".to_string()),
        YamlValue::Bool(false),
    );
    YamlValue::Mapping(root)
}

fn default_locked_yaml_value(field: &str, kind: ValueKind) -> YamlValue {
    match (field, kind) {
        ("tokenizer_name", _) => YamlValue::String("Qwen/Qwen3-8B".to_string()),
        ("rollout_server_url", _) => YamlValue::String("http://localhost:8000".to_string()),
        ("use_wandb", _) => YamlValue::Bool(true),
        ("max_token_length", _) => YamlValue::Number(serde_yaml::Number::from(8192)),
        ("max_num_workers", _) => YamlValue::Number(serde_yaml::Number::from(2048)),
        ("worker_timeout", _) => YamlValue::Number(serde_yaml::Number::from(3600)),
        ("total_steps", _) => YamlValue::Number(serde_yaml::Number::from(2500)),
        ("steps_per_eval", _) => YamlValue::Number(serde_yaml::Number::from(25)),
        ("max_batches_offpolicy", _) => YamlValue::Number(serde_yaml::Number::from(3)),
        ("inference_weight", _) => YamlValue::String("1.0".to_string()),
        ("eval_limit_ratio", _) => YamlValue::String("0.1".to_string()),
        _ => match kind {
            ValueKind::String => YamlValue::String(String::new()),
            ValueKind::Bool => YamlValue::Bool(false),
            ValueKind::Number => YamlValue::Number(serde_yaml::Number::from(0)),
        },
    }
}

fn json_to_yaml(value: &Value) -> YamlValue {
    match value {
        Value::Null => YamlValue::Null,
        Value::Bool(value) => YamlValue::Bool(*value),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                YamlValue::Number(serde_yaml::Number::from(integer))
            } else if let Some(integer) = value.as_u64() {
                YamlValue::Number(serde_yaml::Number::from(integer))
            } else if let Some(float) = value.as_f64() {
                YamlValue::String(float.to_string())
            } else {
                YamlValue::Null
            }
        }
        Value::String(value) => YamlValue::String(value.clone()),
        Value::Array(values) => YamlValue::Sequence(values.iter().map(json_to_yaml).collect()),
        Value::Object(values) => {
            let mut mapping = serde_yaml::Mapping::new();
            for (key, value) in values {
                mapping.insert(YamlValue::String(key.clone()), json_to_yaml(value));
            }
            YamlValue::Mapping(mapping)
        }
    }
}

fn locked_env_value(field: &str) -> Value {
    match field {
        "tokenizer_name" => Value::String("Qwen/Qwen3-8B".to_string()),
        "rollout_server_url" => Value::String("http://localhost:8000".to_string()),
        "use_wandb" => Value::Bool(true),
        "max_token_length" => json!(8192),
        "max_num_workers" => json!(2048),
        "worker_timeout" => json!(3600),
        "total_steps" => json!(2500),
        "steps_per_eval" => json!(25),
        "max_batches_offpolicy" => json!(3),
        "inference_weight" => json!(1.0),
        "eval_limit_ratio" => json!(0.1),
        _ => Value::Null,
    }
}

fn coerce_value(value: Value, default: &Value) -> Result<Value, String> {
    match default {
        Value::Bool(_) => match value {
            Value::Bool(_) => Ok(value),
            Value::String(text) => parse_bool(&text)
                .map(Value::Bool)
                .ok_or_else(|| "value must be a boolean".to_string()),
            _ => Err("value must be a boolean".to_string()),
        },
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                match value {
                    Value::Number(_) => Ok(value),
                    Value::String(text) => text
                        .trim()
                        .parse::<i64>()
                        .map(|parsed| json!(parsed))
                        .map_err(|_| "value must be an integer".to_string()),
                    _ => Err("value must be an integer".to_string()),
                }
            } else {
                match value {
                    Value::Number(_) => Ok(value),
                    Value::String(text) => text
                        .trim()
                        .parse::<f64>()
                        .map(|parsed| json!(parsed))
                        .map_err(|_| "value must be a number".to_string()),
                    _ => Err("value must be a number".to_string()),
                }
            }
        }
        Value::String(_) => match value {
            Value::String(text) => Ok(Value::String(text.trim().to_string())),
            _ => Err("value must be a string".to_string()),
        },
        Value::Array(_) => match value {
            Value::Array(_) => Ok(value),
            _ => Err("value must be an array".to_string()),
        },
        Value::Object(_) => match value {
            Value::Object(_) => Ok(value),
            _ => Err("value must be an object".to_string()),
        },
        Value::Null => Ok(value),
    }
}

fn save_selection_state(runtime: &ToolRuntime, state: &SelectionState) -> Result<(), String> {
    ensure_rl_dirs(runtime)?;
    let path = selection_state_path(runtime);
    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|error| format!("serializing RL selection state failed: {error}"))?;
    fs::write(&path, bytes).map_err(|error| {
        format!(
            "writing RL selection state {} failed: {error}",
            path.display()
        )
    })
}

fn load_selection_state(runtime: &ToolRuntime) -> Result<SelectionState, String> {
    let path = selection_state_path(runtime);
    if !path.is_file() {
        return Ok(SelectionState::default());
    }
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "reading RL selection state {} failed: {error}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "parsing RL selection state {} failed: {error}",
            path.display()
        )
    })
}

fn save_run_state(runtime: &ToolRuntime, run: &RunState) -> Result<(), String> {
    ensure_rl_dirs(runtime)?;
    let path = run_state_path(runtime, &run.run_id);
    let bytes = serde_json::to_vec_pretty(run)
        .map_err(|error| format!("serializing RL run state failed: {error}"))?;
    fs::write(&path, bytes)
        .map_err(|error| format!("writing RL run state {} failed: {error}", path.display()))
}

fn load_run_state(runtime: &ToolRuntime, run_id: &str) -> Result<RunState, String> {
    let path = run_state_path(runtime, run_id);
    if !path.is_file() {
        return Err(format!("Run '{}' not found", run_id));
    }
    let bytes = fs::read(&path)
        .map_err(|error| format!("reading RL run state {} failed: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parsing RL run state {} failed: {error}", path.display()))
}

fn load_all_runs(runtime: &ToolRuntime) -> Vec<RunState> {
    let root = rl_runs_root(runtime);
    let Ok(entries) = fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut runs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if let Ok(bytes) = fs::read(&path)
            && let Ok(run) = serde_json::from_slice::<RunState>(&bytes)
        {
            runs.push(run);
        }
    }
    runs
}

fn ensure_rl_dirs(runtime: &ToolRuntime) -> Result<(), String> {
    for path in [
        rl_root(runtime),
        rl_runs_root(runtime),
        rl_configs_root(runtime),
        rl_logs_root(runtime),
    ] {
        fs::create_dir_all(&path)
            .map_err(|error| format!("creating {} failed: {error}", path.display()))?;
    }
    Ok(())
}

fn selection_state_path(runtime: &ToolRuntime) -> PathBuf {
    rl_root(runtime).join("current_selection.json")
}

fn run_state_path(runtime: &ToolRuntime, run_id: &str) -> PathBuf {
    rl_runs_root(runtime).join(format!("{run_id}.json"))
}

fn rl_root(runtime: &ToolRuntime) -> PathBuf {
    runtime.hermes_home().join("rl")
}

fn rl_runs_root(runtime: &ToolRuntime) -> PathBuf {
    rl_root(runtime).join("runs")
}

fn rl_configs_root(runtime: &ToolRuntime) -> PathBuf {
    rl_root(runtime).join("configs")
}

fn rl_logs_root(runtime: &ToolRuntime) -> PathBuf {
    runtime.hermes_home().join("logs").join("rl_training")
}

fn discover_environments() -> Vec<EnvironmentInfo> {
    let mut environments = Vec::new();
    for root in environment_roots() {
        collect_environment_files(&root, &mut environments);
    }
    environments.sort_by(|left, right| left.name.cmp(&right.name));
    environments
        .dedup_by(|left, right| left.name == right.name && left.file_path == right.file_path);
    environments
}

fn collect_environment_files(root: &Path, out: &mut Vec<EnvironmentInfo>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_environment_files(&path, out);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("py") {
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some("__init__.py") {
            continue;
        }
        if let Some(environment) = parse_environment_info(&path) {
            out.push(environment);
        }
    }
}

fn parse_environment_info(path: &Path) -> Option<EnvironmentInfo> {
    let text = fs::read_to_string(path).ok()?;
    let class_name = extract_env_class_name(&text)?;
    let name = extract_env_name(&text).unwrap_or_else(|| {
        path.file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("environment")
            .to_string()
    });
    let description = extract_description(&text)
        .unwrap_or_else(|| format!("Environment from {}", path.display()));
    let default_config_path = {
        let candidate = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("default.yaml");
        if candidate.is_file() {
            Some(candidate)
        } else {
            None
        }
    };
    Some(EnvironmentInfo {
        name,
        class_name,
        file_path: path.to_path_buf(),
        description,
        default_config_path,
    })
}

fn extract_env_class_name(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("class ") {
            return None;
        }
        if !(trimmed.contains("BaseEnv") || trimmed.contains("HermesAgentBaseEnv")) {
            return None;
        }
        let rest = trimmed.trim_start_matches("class ").trim();
        let end = rest.find(['(', ':']).unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    })
}

fn extract_env_name(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let trimmed = line.trim();
        if !trimmed.starts_with("name") || !trimmed.contains('=') {
            return None;
        }
        let (_, raw_value) = trimmed.split_once('=')?;
        let parsed = parse_python_literal(raw_value.trim())?;
        parsed.as_str().map(|value| value.to_string())
    })
}

fn extract_description(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    for marker in ["\"\"\"", "'''"] {
        if !trimmed.starts_with(marker) {
            continue;
        }
        let rest = &trimmed[marker.len()..];
        let end = rest.find(marker)?;
        let doc = &rest[..end];
        for line in doc.lines() {
            let candidate = line.trim();
            if !candidate.is_empty() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

fn environment_roots() -> Vec<PathBuf> {
    if let Some(custom) = env::var_os("HERMES_RL_ENVIRONMENTS_ROOT").map(PathBuf::from)
        && custom.is_dir()
    {
        return vec![custom];
    }
    let repo = repo_root();
    let training = training_root();
    let mut roots = Vec::new();
    for candidate in [
        training
            .join("tinker-atropos")
            .join("tinker_atropos")
            .join("environments"),
        training.join("tinker_atropos").join("environments"),
        training.join("environments"),
        repo.join("environments"),
    ] {
        if candidate.is_dir() && !roots.contains(&candidate) {
            roots.push(candidate);
        }
    }
    roots
}

fn repo_root() -> PathBuf {
    if let Some(path) = env::var_os("HERMES_RL_REPO_ROOT").map(PathBuf::from) {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn training_root() -> PathBuf {
    env::var_os("HERMES_RL_TRAINING_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(repo_root)
}

fn find_run_api_command() -> Result<Vec<String>, String> {
    if let Some(path) = env::var_os("HERMES_RL_RUN_API").map(PathBuf::from) {
        return Ok(vec![path.display().to_string()]);
    }
    if let Some(path) = find_in_path("run-api") {
        return Ok(vec![path.display().to_string()]);
    }
    let local = training_root().join("run-api");
    if local.is_file() {
        return Ok(vec![local.display().to_string()]);
    }
    Err("run-api command not found. Install or expose the Atropos runtime before starting RL training.".to_string())
}

fn find_launch_training_script() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("HERMES_RL_LAUNCH_TRAINING").map(PathBuf::from)
        && path.is_file()
    {
        return Ok(path);
    }
    for candidate in [
        training_root().join("launch_training.py"),
        training_root()
            .join("tinker-atropos")
            .join("launch_training.py"),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(
        "launch_training.py not found. The RL training runtime is not installed in this checkout."
            .to_string(),
    )
}

fn write_yaml_file(path: &Path, value: &YamlValue) -> Result<(), String> {
    let text = serde_yaml::to_string(value)
        .map_err(|error| format!("serializing training config failed: {error}"))?;
    fs::write(path, text)
        .map_err(|error| format!("writing training config {} failed: {error}", path.display()))
}

fn spawn_background_program(
    program: &str,
    args: &[String],
    cwd: &Path,
    log_path: &Path,
    status_path: &Path,
    extra_env: &[(&str, String)],
) -> Result<ManagedProcess, String> {
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| format!("opening {} failed: {error}", log_path.display()))?;
    let stderr_file = log_file
        .try_clone()
        .map_err(|error| format!("cloning log handle failed: {error}"))?;
    let command_line = shell_join(
        std::iter::once(program.to_string())
            .chain(args.iter().cloned())
            .collect::<Vec<_>>()
            .iter()
            .map(String::as_str),
    );
    let wrapped = format!(
        "{command_line}\ncode=$?\nprintf '%s' \"$code\" > {status}\nexit \"$code\"",
        status = shell_quote(&status_path.display().to_string()),
    );
    let mut command = shell_command(&wrapped, cwd);
    command.stdin(Stdio::null());
    command.stdout(Stdio::from(log_file));
    command.stderr(Stdio::from(stderr_file));
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let child = command
        .spawn()
        .map_err(|error| format!("starting background program failed: {error}"))?;
    let pid = child.id() as i32;
    drop(child);
    Ok(ManagedProcess {
        pid,
        pgid: pid,
        command: command_line,
        cwd: cwd.display().to_string(),
        log_path: log_path.to_path_buf(),
        status_path: status_path.to_path_buf(),
    })
}

fn run_foreground_program(
    program: &str,
    args: &[String],
    cwd: &Path,
    extra_env: &[(&str, String)],
    timeout_secs: u64,
) -> Result<ForegroundOutput, String> {
    let mut command = Command::new(program);
    command.args(args).current_dir(cwd).stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    for (key, value) in extra_env {
        command.env(key, value);
    }
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .map_err(|error| format!("starting foreground program failed: {error}"))?;
    let pid = child.id() as i32;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let output = match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(result) => {
            result.map_err(|error| format!("waiting for foreground program failed: {error}"))?
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = signal_group(pid, libc::SIGTERM);
            thread::sleep(Duration::from_millis(WAIT_AFTER_KILL_MILLIS));
            if process_alive(pid) {
                let _ = signal_group(pid, libc::SIGKILL);
            }
            return Err(format!(
                "foreground program timed out after {} seconds",
                timeout_secs
            ));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err("foreground program worker disconnected unexpectedly".to_string());
        }
    };
    Ok(ForegroundOutput {
        exit_code: output.status.code().unwrap_or_default(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn terminate_process(process: &ManagedProcess) -> Result<(), String> {
    signal_group(process.pgid, libc::SIGTERM)
        .map_err(|error| format!("terminating process {} failed: {error}", process.pid))?;
    thread::sleep(Duration::from_millis(WAIT_AFTER_KILL_MILLIS));
    if process_alive(process.pid) {
        signal_group(process.pgid, libc::SIGKILL)
            .map_err(|error| format!("killing process {} failed: {error}", process.pid))?;
    }
    Ok(())
}

fn ensure_status_marker(process: &ManagedProcess, code: i32) {
    if !process.status_path.is_file() {
        let _ = fs::write(&process.status_path, code.to_string());
    }
}

fn read_exit_code(path: &Path) -> Option<i32> {
    let text = fs::read_to_string(path).ok()?;
    text.trim().parse::<i32>().ok()
}

fn read_last_lines(path: &Path, limit: usize) -> Vec<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines = text.lines().map(str::to_string).collect::<Vec<_>>();
    let start = lines.len().saturating_sub(limit);
    lines.into_iter().skip(start).collect()
}

fn signal_group(pgid: i32, signal: i32) -> io::Result<()> {
    let rc = unsafe { libc::kill(-pgid, signal) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn process_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn shell_command(command: &str, cwd: &Path) -> Command {
    let mut builder = Command::new("bash");
    builder.arg("-lc").arg(command).current_dir(cwd);
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

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|base| base.join(name))
        .find(|candidate| candidate.is_file())
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    let Some(value) = args.get(key) else {
        return Err(format!("{key} is required"));
    };
    let Some(text) = value.as_str() else {
        return Err(format!("{key} must be a string"));
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(trimmed.to_string())
}

fn optional_u64(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a non-negative integer")),
        Some(Value::String(text)) => text
            .trim()
            .parse::<u64>()
            .map(Some)
            .map_err(|_| format!("{key} must be a non-negative integer")),
        _ => Err(format!("{key} must be a non-negative integer")),
    }
}

fn optional_string_array(args: &Value, key: &str) -> Result<Vec<String>, String> {
    let Some(value) = args.get(key) else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Err(format!("{key} must be an array of strings"));
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(format!("{key} must be an array of strings"));
        };
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    Ok(out)
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn strip_comment(raw: &str) -> &str {
    raw.split('#').next().unwrap_or(raw).trim()
}

fn extract_parenthesized_block(text: &str, open_paren_index: usize) -> Option<String> {
    let mut depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut start = None;
    for (offset, ch) in text[open_paren_index..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && (in_single || in_double) {
            escaped = true;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if in_single || in_double {
            continue;
        }
        if ch == '(' {
            depth += 1;
            if depth == 1 {
                start = Some(open_paren_index + offset + 1);
            }
        } else if ch == ')' {
            depth -= 1;
            if depth == 0 {
                let begin = start?;
                let end = open_paren_index + offset;
                return Some(text[begin..end].to_string());
            }
        }
    }
    None
}

fn split_top_level(input: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for (index, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && (in_single || in_double) {
            escaped = true;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if in_single || in_double {
            continue;
        }
        match ch {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        if ch == delimiter && paren == 0 && bracket == 0 && brace == 0 {
            parts.push(&input[start..index]);
            start = index + ch.len_utf8();
        }
    }
    if start <= input.len() {
        parts.push(&input[start..]);
    }
    parts
}

fn unquote_python_string(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() < 2 {
        return trimmed.to_string();
    }
    let quote = trimmed.as_bytes()[0] as char;
    if (quote == '"' || quote == '\'') && trimmed.ends_with(quote) {
        trimmed[1..trimmed.len() - 1]
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\'", "'")
    } else {
        trimmed.to_string()
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn shell_join<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    items
        .into_iter()
        .map(shell_quote)
        .collect::<Vec<_>>()
        .join(" ")
}

fn env_var_present(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn min_status_interval_secs() -> u64 {
    env::var("HERMES_RL_MIN_STATUS_INTERVAL")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_STATUS_INTERVAL_SECS)
}

fn inference_timeout_secs() -> u64 {
    env::var("HERMES_RL_INFERENCE_TIMEOUT")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_INFERENCE_TIMEOUT_SECS)
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

fn short_id() -> String {
    format!("{:x}", unix_ts_nanos()).chars().take(8).collect()
}

fn timestamp_slug() -> String {
    format!("{:x}", unix_ts_nanos())
}

fn value_preview(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Mutex, OnceLock};

    use tempfile::TempDir;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<str>) -> Self {
            let previous = env::var(key).ok();
            unsafe { env::set_var(key, value.as_ref()) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                unsafe { env::set_var(self.key, previous) };
            } else {
                unsafe { env::remove_var(self.key) };
            }
        }
    }

    fn write_file(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o755));
    }

    fn test_runtime(home: &Path, cwd: &Path) -> ToolRuntime {
        ToolRuntime::new(cwd).with_hermes_home(home)
    }

    #[test]
    fn list_environments_discovers_local_tree() {
        let _guard = env_lock().lock().unwrap_or_else(|error| error.into_inner());
        let repo = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write_file(
            &repo.path().join("environments/demo_env/demo_env.py"),
            r#"
"""Demo environment."""

class DemoEnv(HermesAgentBaseEnv):
    name = "demo-env"
    def config_init(cls):
        env_config = DemoEnvConfig(
            group_size=4,
            total_steps=12,
            steps_per_eval=3,
            wandb_name="demo-env",
            enabled_toolsets=["terminal", "file"],
        )
"#,
        );
        write_file(
            &repo.path().join("environments/demo_env/default.yaml"),
            r#"
env:
  group_size: 4
  total_steps: 12
  steps_per_eval: 3
  wandb_name: "demo-env"
"#,
        );
        let _repo = EnvGuard::set("HERMES_RL_REPO_ROOT", repo.path().display().to_string());
        let runtime = test_runtime(home.path(), repo.path());
        let output = handle_rl_list_environments(&json!({}), &runtime);
        let value: Value = serde_json::from_str(&output).unwrap();
        let environments = value.get("environments").and_then(Value::as_array).unwrap();
        assert_eq!(environments.len(), 1);
        assert_eq!(
            environments[0].get("name").and_then(Value::as_str),
            Some("demo-env")
        );
    }

    #[test]
    fn select_and_edit_config_persist() {
        let _guard = env_lock().lock().unwrap_or_else(|error| error.into_inner());
        let repo = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write_file(
            &repo.path().join("environments/demo_env/demo_env.py"),
            r#"
"""Demo environment."""
class DemoEnv(HermesAgentBaseEnv):
    name = "demo-env"
"#,
        );
        write_file(
            &repo.path().join("environments/demo_env/default.yaml"),
            r#"
env:
  group_size: 4
  total_steps: 12
  steps_per_eval: 3
  wandb_name: "demo-env"
"#,
        );
        let _repo = EnvGuard::set("HERMES_RL_REPO_ROOT", repo.path().display().to_string());
        let runtime = test_runtime(home.path(), repo.path());

        let selected = handle_rl_select_environment(&json!({ "name": "demo-env" }), &runtime);
        let selected: Value = serde_json::from_str(&selected).unwrap();
        assert_eq!(selected.get("success").and_then(Value::as_bool), Some(true));

        let edited = handle_rl_edit_config(&json!({ "field": "group_size", "value": 7 }), &runtime);
        let edited: Value = serde_json::from_str(&edited).unwrap();
        assert_eq!(edited.get("success").and_then(Value::as_bool), Some(true));

        let config = handle_rl_get_current_config(&json!({}), &runtime);
        let config: Value = serde_json::from_str(&config).unwrap();
        let fields = config
            .get("configurable_fields")
            .and_then(Value::as_array)
            .unwrap();
        let group_size = fields
            .iter()
            .find(|field| field.get("name").and_then(Value::as_str) == Some("group_size"))
            .unwrap();
        assert_eq!(
            group_size.get("current_value").and_then(Value::as_i64),
            Some(7)
        );
    }

    #[test]
    fn start_status_stop_and_list_runs_work_with_dummy_scripts() {
        let _guard = env_lock().lock().unwrap_or_else(|error| error.into_inner());
        let repo = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write_file(
            &repo.path().join("environments/demo_env/demo_env.py"),
            r#"
#!/usr/bin/env python3
"""Demo environment."""
class HermesAgentBaseEnv: pass
class DemoEnv(HermesAgentBaseEnv):
    name = "demo-env"

import sys, time
if len(sys.argv) > 1 and sys.argv[1] == "serve":
    time.sleep(60)
elif len(sys.argv) > 1 and sys.argv[1] == "process":
    time.sleep(1)
"#,
        );
        write_file(
            &repo.path().join("environments/demo_env/default.yaml"),
            r#"
env:
  group_size: 2
  total_steps: 8
  steps_per_eval: 2
  wandb_name: "demo-env"
"#,
        );
        write_file(
            &repo.path().join("launch_training.py"),
            r#"
#!/usr/bin/env python3
import time
time.sleep(60)
"#,
        );
        write_file(
            &repo.path().join("run-api"),
            "#!/usr/bin/env bash\nsleep 60\n",
        );
        let _repo = EnvGuard::set("HERMES_RL_REPO_ROOT", repo.path().display().to_string());
        let _path = EnvGuard::set(
            "PATH",
            format!(
                "{}:{}",
                repo.path().display(),
                env::var("PATH").unwrap_or_default()
            ),
        );
        let _tinker = EnvGuard::set("TINKER_API_KEY", "test-key");
        let _wandb = EnvGuard::set("WANDB_API_KEY", "test-key");
        let _interval = EnvGuard::set("HERMES_RL_MIN_STATUS_INTERVAL", "0");
        let runtime = test_runtime(home.path(), repo.path());

        let _ = handle_rl_select_environment(&json!({ "name": "demo-env" }), &runtime);
        let started = handle_rl_start_training(&json!({}), &runtime);
        let started: Value = serde_json::from_str(&started).unwrap();
        assert_eq!(started.get("success").and_then(Value::as_bool), Some(true));
        let run_id = started
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();

        let status = handle_rl_check_status(&json!({ "run_id": run_id }), &runtime);
        let status: Value = serde_json::from_str(&status).unwrap();
        assert_eq!(status.get("success").and_then(Value::as_bool), Some(true));

        let runs = handle_rl_list_runs(&json!({}), &runtime);
        let runs: Value = serde_json::from_str(&runs).unwrap();
        assert_eq!(runs.get("count").and_then(Value::as_u64), Some(1));

        let stopped = handle_rl_stop_training(
            &json!({ "run_id": status.get("run_id").cloned().unwrap() }),
            &runtime,
        );
        let stopped: Value = serde_json::from_str(&stopped).unwrap();
        assert_eq!(stopped.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(
            stopped.get("status").and_then(Value::as_str),
            Some("stopped")
        );
    }

    #[test]
    fn test_inference_parses_jsonl_output() {
        let _guard = env_lock().lock().unwrap_or_else(|error| error.into_inner());
        let repo = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write_file(
            &repo.path().join("environments/demo_env/demo_env.py"),
            r#"
#!/usr/bin/env python3
"""Demo environment."""
class HermesAgentBaseEnv: pass
class DemoEnv(HermesAgentBaseEnv):
    name = "demo-env"

import json, sys
output_path = None
for idx, arg in enumerate(sys.argv):
    if arg == "--env.data_path_to_save_groups":
        output_path = sys.argv[idx + 1]
if output_path:
    with open(output_path, "w") as handle:
        handle.write(json.dumps({"scores": [1, 0, 1, 1]}) + "\n")
        handle.write(json.dumps({"scores": [0, 0, 1, 0]}) + "\n")
"#,
        );
        write_file(
            &repo.path().join("environments/demo_env/default.yaml"),
            r#"
env:
  group_size: 4
  total_steps: 2
  steps_per_eval: 1
  wandb_name: "demo-env"
"#,
        );
        let _repo = EnvGuard::set("HERMES_RL_REPO_ROOT", repo.path().display().to_string());
        let _openrouter = EnvGuard::set("OPENROUTER_API_KEY", "test-key");
        let runtime = test_runtime(home.path(), repo.path());
        let _ = handle_rl_select_environment(&json!({ "name": "demo-env" }), &runtime);

        let output = handle_rl_test_inference(
            &json!({ "num_steps": 2, "group_size": 4, "models": ["demo/model"] }),
            &runtime,
        );
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value.get("success").and_then(Value::as_bool), Some(true));
        let models = value
            .get("models_tested")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].get("steps_tested").and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            models[0].get("correct_completions").and_then(Value::as_u64),
            Some(4)
        );
    }
}
