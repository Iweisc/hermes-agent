//! RL Training CLI Runner — native Rust port of `rl_cli.py`.
//!
//! Dedicated CLI runner for RL training workflows with:
//! - Extended timeouts / iteration budgets for long-running training
//! - RL-focused system prompts
//! - Full RL toolset (terminal, web, rl)
//! - Special handling for 30-minute check intervals
//!
//! The original Python file is a `fire.Fire(main)` entry point. This port keeps
//! the *deterministic, testable* surface (config loading, requirement checks,
//! tinker-atropos discovery, environment listing, run-config assembly, and the
//! human-facing banner text) as pure functions, and exposes a `run()` driver
//! that mirrors `main()`'s control flow.
//!
//! The actual agent loop (`AIAgent.run_conversation`) is represented through the
//! [`AgentRunner`] trait so callers can plug in the ported agent without this
//! module taking a hard dependency on the full agent runtime.
//!
//! Environment variables (read at runtime, matching Python):
//! - `OPENROUTER_API_KEY` — required for the agent
//! - `TINKER_API_KEY`, `WANDB_API_KEY` — required for RL training
//! - `TERMINAL_CWD`, `HERMES_QUIET` — set by [`configure_terminal_cwd`]

use std::path::{Path, PathBuf};

use serde_json::Value;

// ============================================================================
// Constants (mirror rl_cli.py module-level constants)
// ============================================================================

/// Default agent model when none is configured.
pub const DEFAULT_MODEL: &str = "anthropic/claude-opus-4.5";

/// Default base URL — OpenRouter. Mirrors `hermes_constants.OPENROUTER_BASE_URL`.
/// Kept as a local const so this module does not hard-fail if the constants
/// module is not yet wired; prefer `crate::mod_hermes_constants::OPENROUTER_BASE_URL`
/// at the integration boundary.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Extended iteration budget for long-running RL workflows.
pub const RL_MAX_ITERATIONS: u32 = 200;

/// Toolsets enabled for RL workflows.
pub const RL_TOOLSETS: [&str; 3] = ["terminal", "web", "rl"];

/// RL API keys required for training (mirrors `tools.rl_training_tool._rl_env`).
pub const RL_REQUIRED_KEYS: [&str; 2] = ["TINKER_API_KEY", "WANDB_API_KEY"];

/// RL-focused system prompt handed to the agent.
pub const RL_SYSTEM_PROMPT: &str = r#"You are an automated post-training engineer specializing in reinforcement learning for language models.

## Your Capabilities

You have access to RL training tools for running reinforcement learning on models through Tinker-Atropos:

1. **DISCOVER**: Use `rl_list_environments` to see available RL environments
2. **INSPECT**: Read environment files to understand how they work (verifiers, data loading, rewards)
3. **INSPECT DATA**: Use terminal to explore HuggingFace datasets and understand their format
4. **CREATE**: Copy existing environments as templates, modify for your needs
5. **CONFIGURE**: Use `rl_select_environment` and `rl_edit_config` to set up training
6. **TEST**: Always use `rl_test_inference` before full training to validate your setup
7. **TRAIN**: Use `rl_start_training` to begin, `rl_check_status` to monitor
8. **EVALUATE**: Use `rl_get_results` and analyze WandB metrics to assess performance

## Environment Files

Environment files are located in: `tinker-atropos/tinker_atropos/environments/`

Study existing environments to learn patterns. Look for:
- `load_dataset()` calls - how data is loaded
- `score_answer()` / `score()` - verification logic
- `get_next_item()` - prompt formatting
- `system_prompt` - instruction format
- `config_init()` - default configuration

## Creating New Environments

To create a new environment:
1. Read an existing environment file (e.g., gsm8k_tinker.py)
2. Use terminal to explore the target dataset format
3. Copy the environment file as a template
4. Modify the dataset loading, prompt formatting, and verifier logic
5. Test with `rl_test_inference` before training

## Important Guidelines

- **Always test before training**: Training runs take hours - verify everything works first
- **Monitor metrics**: Check WandB for reward/mean and percent_correct
- **Status check intervals**: Wait at least 30 minutes between status checks
- **Early stopping**: Stop training early if metrics look bad or stagnant
- **Iterate quickly**: Start with small total_steps to validate, then scale up

## Available Toolsets

You have access to:
- **RL tools**: Environment discovery, config management, training, testing
- **Terminal**: Run commands, inspect files, explore datasets
- **Web**: Search for information, documentation, papers
- **File tools**: Read and modify code files

When asked to train a model, follow this workflow:
1. List available environments
2. Select and configure the appropriate environment
3. Test with sample prompts
4. Start training with conservative settings
5. Monitor progress and adjust as needed
"#;

// ============================================================================
// Config Loading
// ============================================================================

/// Resolved Hermes config used to seed the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermesConfig {
    pub model: String,
    pub base_url: String,
}

impl Default for HermesConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }
}

/// Load configuration from `<hermes_home>/config.yaml`.
///
/// Mirrors `load_hermes_config()`:
/// - `model` may be a string, or a mapping with a `default` key.
/// - `base_url` is taken verbatim when present.
/// - On any parse failure, the defaults are kept and a warning string is
///   returned (the Python code prints it; here we surface it to the caller).
pub fn load_hermes_config(hermes_home: &Path) -> (HermesConfig, Option<String>) {
    let mut config = HermesConfig::default();
    let config_path = hermes_home.join("config.yaml");

    if !config_path.exists() {
        return (config, None);
    }

    let text = match std::fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(e) => {
            return (
                config,
                Some(format!("⚠️  Warning: Failed to load config.yaml: {e}")),
            );
        }
    };

    let parsed: Value = match serde_yaml::from_str::<Value>(&text) {
        // `yaml.safe_load(f) or {}` — null/empty becomes an empty mapping.
        Ok(Value::Null) => Value::Object(Default::default()),
        Ok(v) => v,
        Err(e) => {
            return (
                config,
                Some(format!("⚠️  Warning: Failed to load config.yaml: {e}")),
            );
        }
    };

    apply_config_value(&mut config, &parsed);
    (config, None)
}

/// Apply a parsed YAML mapping onto a [`HermesConfig`] following Python's rules.
fn apply_config_value(config: &mut HermesConfig, file_config: &Value) {
    if let Some(model) = file_config.get("model") {
        match model {
            Value::String(s) => config.model = s.clone(),
            Value::Object(_) => {
                config.model = model
                    .get("default")
                    .and_then(Value::as_str)
                    .unwrap_or(DEFAULT_MODEL)
                    .to_string();
            }
            _ => {}
        }
    }

    if let Some(base_url) = file_config.get("base_url").and_then(Value::as_str) {
        config.base_url = base_url.to_string();
    }
}

// ============================================================================
// Terminal CWD configuration
// ============================================================================

/// Result of resolving the terminal working directory for RL work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalCwd {
    /// Directory written to `TERMINAL_CWD`.
    pub cwd: PathBuf,
    /// Whether the `tinker-atropos` submodule was found.
    pub submodule_found: bool,
}

/// Resolve the terminal working directory the way `rl_cli.py` does at import:
/// prefer `<project_root>/tinker-atropos`, else fall back to `<project_root>`.
///
/// Pure: does not touch the environment. Use [`apply_terminal_cwd`] to set the
/// process env vars.
pub fn resolve_terminal_cwd(project_root: &Path) -> TerminalCwd {
    let tinker = project_root.join("tinker-atropos");
    if tinker.exists() {
        TerminalCwd {
            cwd: tinker,
            submodule_found: true,
        }
    } else {
        TerminalCwd {
            cwd: project_root.to_path_buf(),
            submodule_found: false,
        }
    }
}

/// Set `TERMINAL_CWD` and `HERMES_QUIET=1` to match the Python import side
/// effects. Returns a human-facing status line.
pub fn apply_terminal_cwd(project_root: &Path) -> (TerminalCwd, String) {
    let resolved = resolve_terminal_cwd(project_root);
    // Edition 2024 requires `unsafe` around env mutation.
    unsafe {
        std::env::set_var("TERMINAL_CWD", &resolved.cwd);
        std::env::set_var("HERMES_QUIET", "1");
    }
    let msg = if resolved.submodule_found {
        format!("📂 Terminal working directory: {}", resolved.cwd.display())
    } else {
        format!(
            "⚠️  tinker-atropos submodule not found, using: {}",
            resolved.cwd.display()
        )
    };
    (resolved, msg)
}

// ============================================================================
// Requirement / setup checks
// ============================================================================

/// Return the missing RL API keys (mirrors `tools.rl_training_tool.get_missing_keys`).
pub fn get_missing_keys() -> Vec<String> {
    RL_REQUIRED_KEYS
        .iter()
        .filter(|key| match std::env::var(key) {
            Ok(v) => v.is_empty(),
            Err(_) => true,
        })
        .map(|key| key.to_string())
        .collect()
}

/// Check that required environment variables are present.
///
/// Mirrors `check_requirements()`: requires `OPENROUTER_API_KEY` plus the RL
/// keys. Returns `Ok(())` when satisfied, or `Err(errors)` listing each problem
/// in the same order/format as the Python implementation.
pub fn check_requirements() -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    let has_openrouter = std::env::var("OPENROUTER_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if !has_openrouter {
        errors.push("OPENROUTER_API_KEY not set - required for agent".to_string());
    }

    let missing = get_missing_keys();
    if !missing.is_empty() {
        errors.push(format!("Missing RL API keys: {}", missing.join(", ")));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Outcome of [`check_tinker_atropos`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TinkerStatus {
    /// Submodule + environments dir present. Carries path and env file count.
    Ready { path: PathBuf, environments_count: usize },
    /// Something is missing; carries a human-facing reason.
    Missing(String),
}

/// Check whether the `tinker-atropos` submodule is properly set up.
///
/// Mirrors `check_tinker_atropos()`:
/// - submodule dir must exist,
/// - `tinker_atropos/environments` must exist,
/// - counts `*.py` files not starting with `_`.
pub fn check_tinker_atropos(project_root: &Path) -> TinkerStatus {
    let tinker_path = project_root.join("tinker-atropos");
    if !tinker_path.exists() {
        return TinkerStatus::Missing(
            "tinker-atropos submodule not found. Run: git submodule update --init".to_string(),
        );
    }

    let envs_path = tinker_path.join("tinker_atropos").join("environments");
    if !envs_path.exists() {
        return TinkerStatus::Missing(format!(
            "environments directory not found at {}",
            envs_path.display()
        ));
    }

    let count = count_environment_files(&envs_path);
    TinkerStatus::Ready {
        path: tinker_path,
        environments_count: count,
    }
}

/// Count `*.py` files in `dir` whose name does not start with `_`.
fn count_environment_files(dir: &Path) -> usize {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".py") && !name.starts_with('_')
        })
        .count()
}

// ============================================================================
// Environment listing
// ============================================================================

/// One discovered RL environment, as surfaced by `rl_list_environments`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvSummary {
    pub name: String,
    pub class_name: String,
    pub file_path: String,
    pub description: Option<String>,
}

/// Parse the JSON returned by the `rl_list_environments` tool into typed
/// summaries. Mirrors the consumption in `main()` for `--list-environments`.
///
/// Returns `Err(message)` if the payload carries an `"error"` field; otherwise
/// returns the (possibly empty) list of environments.
pub fn parse_environments(data: &Value) -> Result<Vec<EnvSummary>, String> {
    if let Some(err) = data.get("error").and_then(Value::as_str) {
        return Err(err.to_string());
    }

    let envs = match data.get("environments").and_then(Value::as_array) {
        Some(arr) => arr,
        None => return Ok(Vec::new()),
    };

    Ok(envs
        .iter()
        .map(|env| EnvSummary {
            name: env
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            class_name: env
                .get("class_name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            file_path: env
                .get("file_path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            description: env
                .get("description")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        })
        .collect())
}

/// Truncate a description for display the way `main()` does: descriptions longer
/// than 100 chars are cut to 100 chars plus an ellipsis.
pub fn truncate_description(desc: &str) -> String {
    if desc.chars().count() > 100 {
        // Python slices by codepoint; replicate with `char` boundaries.
        let truncated: String = desc.chars().take(100).collect();
        format!("{truncated}...")
    } else {
        desc.to_string()
    }
}

/// Format the human-facing environment listing block (mirrors the `print`
/// statements under `--list-environments`).
pub fn format_environment_listing(envs: &[EnvSummary]) -> String {
    let mut out = String::new();
    for env in envs {
        out.push_str(&format!("\n  📦 {}\n", env.name));
        out.push_str(&format!("     Class: {}\n", env.class_name));
        out.push_str(&format!("     Path: {}\n", env.file_path));
        if let Some(desc) = &env.description {
            out.push_str(&format!("     Description: {}\n", truncate_description(desc)));
        }
    }
    out.push_str(&format!("\n📊 Total: {} environments\n", envs.len()));
    out.push_str("\nUse `rl_select_environment(name)` to select an environment for training.");
    out
}

// ============================================================================
// Run configuration assembly
// ============================================================================

/// The fully-resolved configuration for an agent run, mirroring the keyword
/// arguments passed to `AIAgent(...)` in `main()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRunConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_iterations: u32,
    pub enabled_toolsets: Vec<String>,
    pub save_trajectories: bool,
    pub verbose_logging: bool,
    pub quiet_mode: bool,
    pub ephemeral_system_prompt: String,
}

/// CLI inputs, mirroring `main()`'s signature.
#[derive(Debug, Clone, Default)]
pub struct CliArgs {
    pub task: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_iterations: Option<u32>,
    pub interactive: bool,
    pub list_environments: bool,
    pub check_server: bool,
    pub verbose: bool,
    /// `None` means "use the RL default" (True). Python defaults to True.
    pub save_trajectories: Option<bool>,
}

/// Resolve `model`/`base_url` from CLI args, falling back to the loaded config
/// (mirrors `if model is None: model = config["model"]`).
pub fn resolve_model_and_base_url(args: &CliArgs, config: &HermesConfig) -> (String, String) {
    let model = args.model.clone().unwrap_or_else(|| config.model.clone());
    let base_url = args
        .base_url
        .clone()
        .unwrap_or_else(|| config.base_url.clone());
    (model, base_url)
}

/// Resolve the API key: explicit `--api-key` wins, else `OPENROUTER_API_KEY`.
/// Mirrors `api_key = api_key or os.getenv("OPENROUTER_API_KEY")`.
pub fn resolve_api_key(args: &CliArgs) -> Option<String> {
    if let Some(key) = args.api_key.as_ref().filter(|s| !s.is_empty()) {
        return Some(key.clone());
    }
    std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Build the [`AgentRunConfig`] from resolved inputs. `quiet_mode` is always
/// false (Python passes `quiet_mode=False`).
pub fn build_run_config(
    args: &CliArgs,
    model: String,
    base_url: String,
    api_key: String,
) -> AgentRunConfig {
    AgentRunConfig {
        base_url,
        api_key,
        model,
        max_iterations: args.max_iterations.unwrap_or(RL_MAX_ITERATIONS),
        enabled_toolsets: RL_TOOLSETS.iter().map(|s| s.to_string()).collect(),
        save_trajectories: args.save_trajectories.unwrap_or(true),
        verbose_logging: args.verbose,
        quiet_mode: false,
        ephemeral_system_prompt: RL_SYSTEM_PROMPT.to_string(),
    }
}

// ============================================================================
// Agent abstraction (stands in for AIAgent)
// ============================================================================

/// Abstraction over the agent loop so this module does not hard-depend on the
/// full ported agent runtime. The integration layer supplies a concrete
/// implementation wrapping the real `AIAgent` / `crate::mod_run_agent` driver.
pub trait AgentRunner {
    /// Run one conversation turn with the given user input. Mirrors
    /// `AIAgent.run_conversation`. Returns `Err` on agent failure.
    fn run_conversation(&mut self, user_input: &str) -> Result<(), String>;
}

/// Source of interactive lines (stdin abstraction for testability), and a sink
/// for active-run status queries.
pub trait InteractiveIo {
    /// Read one line of input. `None` signals EOF / interrupt (loop exits).
    fn read_line(&mut self, prompt: &str) -> Option<String>;
    /// Query active runs (mirrors `rl_list_runs()` JSON output). Default: none.
    fn list_runs(&mut self) -> Value {
        Value::Array(Vec::new())
    }
    /// Emit a line of output.
    fn output(&mut self, line: &str);
}

/// Format the active-runs block from a `rl_list_runs` JSON payload.
/// Mirrors the `status` branch of interactive mode.
pub fn format_active_runs(runs: &Value) -> String {
    match runs.as_array() {
        Some(arr) if !arr.is_empty() => {
            let mut out = String::from("\n📊 Active Runs:");
            for run in arr {
                let run_id = run.get("run_id").and_then(Value::as_str).unwrap_or("");
                let environment = run.get("environment").and_then(Value::as_str).unwrap_or("");
                let status = run.get("status").and_then(Value::as_str).unwrap_or("");
                out.push_str(&format!("\n  - {run_id}: {environment} ({status})"));
            }
            out
        }
        _ => "\nNo active runs.".to_string(),
    }
}

/// Classify an interactive line into a control action. Mirrors the `quit`/
/// `exit`/`q`, `status`, empty, and task dispatch logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractiveAction {
    /// Empty input — continue without action.
    Skip,
    /// Quit the loop.
    Quit,
    /// Show active runs.
    Status,
    /// Run the agent with the trimmed task text.
    Task(String),
}

/// Classify a raw interactive input line.
pub fn classify_interactive(raw: &str) -> InteractiveAction {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return InteractiveAction::Skip;
    }
    let lower = trimmed.to_lowercase();
    if lower == "quit" || lower == "exit" || lower == "q" {
        return InteractiveAction::Quit;
    }
    if lower == "status" {
        return InteractiveAction::Status;
    }
    InteractiveAction::Task(trimmed.to_string())
}

// ============================================================================
// Banner / message helpers
// ============================================================================

const RULE_60: &str = "============================================================";

/// Header banner emitted at the start of `main()`.
pub fn header_banner() -> String {
    format!("🎯 RL Training Agent\n{RULE_60}")
}

/// Pre-run summary block (model / iterations / toolsets).
pub fn run_summary(model: &str, max_iterations: u32) -> String {
    format!(
        "\n🤖 Model: {model}\n🔧 Max iterations: {max_iterations}\n📁 Toolsets: {}\n{RULE_60}",
        RL_TOOLSETS.join(", ")
    )
}

/// "No task" guidance block, shown when neither a task nor interactive mode is
/// given.
pub fn no_task_help() -> String {
    let mut out = String::new();
    out.push_str(
        "\n⚠️  No task provided. Use --interactive for interactive mode or provide a task.\n",
    );
    out.push_str("\nExamples:\n");
    out.push_str("  python rl_cli.py \"Train a model on GSM8k math problems\"\n");
    out.push_str("  python rl_cli.py \"Create an RL environment for code generation\"\n");
    out.push_str("  python rl_cli.py --interactive");
    out
}

// ============================================================================
// run() — control-flow driver mirroring main()
// ============================================================================

/// What the high-level driver decided to do — useful for callers/tests that do
/// not want to actually invoke the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// `--check-server` handled; nothing further.
    CheckedServer(TinkerStatus),
    /// `--list-environments` handled.
    ListedEnvironments,
    /// Requirements failed; caller should exit(1).
    RequirementsFailed(Vec<String>),
    /// No task and not interactive — help printed.
    NoTask,
    /// No API key resolved; caller should exit(1).
    MissingApiKey,
    /// A run config was assembled and (optionally) executed.
    Ran(AgentRunConfig),
}

/// Drive the CLI flow up to (and including) agent construction, mirroring
/// `main()`'s ordering. `project_root` and `hermes_home` are injected for
/// testability. When `runner` is `Some` and a run config is produced, the
/// single-task path invokes `run_conversation`.
///
/// This intentionally returns the decision rather than calling `std::process::exit`,
/// so it is safe to unit-test; the binary wrapper maps `RequirementsFailed` /
/// `MissingApiKey` to `exit(1)`.
pub fn run(
    args: &CliArgs,
    project_root: &Path,
    hermes_home: &Path,
    runner: Option<&mut dyn AgentRunner>,
) -> RunOutcome {
    let (config, _warn) = load_hermes_config(hermes_home);
    let (model, base_url) = resolve_model_and_base_url(args, &config);

    // check_server short-circuits.
    if args.check_server {
        return RunOutcome::CheckedServer(check_tinker_atropos(project_root));
    }

    // list_environments short-circuits (the actual tool call is the caller's
    // job; this driver reports the branch).
    if args.list_environments {
        return RunOutcome::ListedEnvironments;
    }

    if let Err(errors) = check_requirements() {
        return RunOutcome::RequirementsFailed(errors);
    }

    if args.task.as_deref().unwrap_or("").is_empty() && !args.interactive {
        return RunOutcome::NoTask;
    }

    let api_key = match resolve_api_key(args) {
        Some(k) => k,
        None => return RunOutcome::MissingApiKey,
    };

    let run_config = build_run_config(args, model, base_url, api_key);

    // Single-task mode actually runs the agent when a runner is supplied.
    if !args.interactive {
        if let (Some(runner), Some(task)) = (runner, args.task.as_deref()) {
            let _ = runner.run_conversation(task);
        }
    }

    RunOutcome::Ran(run_config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // Env tests must not race: serialize them.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rl_cli_test_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn config_defaults_when_missing() {
        let dir = tmp_dir("cfg_missing");
        let (cfg, warn) = load_hermes_config(&dir);
        assert_eq!(cfg, HermesConfig::default());
        assert!(warn.is_none());
    }

    #[test]
    fn config_string_model() {
        let dir = tmp_dir("cfg_str");
        std::fs::write(
            dir.join("config.yaml"),
            "model: openai/gpt-5\nbase_url: https://example.test/v1\n",
        )
        .unwrap();
        let (cfg, warn) = load_hermes_config(&dir);
        assert!(warn.is_none());
        assert_eq!(cfg.model, "openai/gpt-5");
        assert_eq!(cfg.base_url, "https://example.test/v1");
    }

    #[test]
    fn config_dict_model_default() {
        let dir = tmp_dir("cfg_dict");
        std::fs::write(
            dir.join("config.yaml"),
            "model:\n  default: anthropic/claude-x\n  fallback: y\n",
        )
        .unwrap();
        let (cfg, _) = load_hermes_config(&dir);
        assert_eq!(cfg.model, "anthropic/claude-x");
        // base_url untouched → default.
        assert_eq!(cfg.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn config_dict_model_without_default_falls_back() {
        let dir = tmp_dir("cfg_dict_nodef");
        std::fs::write(dir.join("config.yaml"), "model:\n  other: z\n").unwrap();
        let (cfg, _) = load_hermes_config(&dir);
        assert_eq!(cfg.model, DEFAULT_MODEL);
    }

    #[test]
    fn config_empty_yaml_is_defaults() {
        let dir = tmp_dir("cfg_empty");
        std::fs::write(dir.join("config.yaml"), "").unwrap();
        let (cfg, warn) = load_hermes_config(&dir);
        assert_eq!(cfg, HermesConfig::default());
        assert!(warn.is_none());
    }

    #[test]
    fn missing_keys_reports_both_when_unset() {
        let _g = env_lock();
        unsafe {
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
        let missing = get_missing_keys();
        assert_eq!(missing, vec!["TINKER_API_KEY", "WANDB_API_KEY"]);
    }

    #[test]
    fn missing_keys_empty_when_set() {
        let _g = env_lock();
        unsafe {
            std::env::set_var("TINKER_API_KEY", "t");
            std::env::set_var("WANDB_API_KEY", "w");
        }
        assert!(get_missing_keys().is_empty());
        unsafe {
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
    }

    #[test]
    fn check_requirements_collects_all() {
        let _g = env_lock();
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
        let err = check_requirements().unwrap_err();
        assert_eq!(err.len(), 2);
        assert!(err[0].contains("OPENROUTER_API_KEY"));
        assert!(err[1].contains("TINKER_API_KEY") && err[1].contains("WANDB_API_KEY"));
    }

    #[test]
    fn check_requirements_ok_when_all_present() {
        let _g = env_lock();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "o");
            std::env::set_var("TINKER_API_KEY", "t");
            std::env::set_var("WANDB_API_KEY", "w");
        }
        assert!(check_requirements().is_ok());
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
    }

    #[test]
    fn tinker_missing_submodule() {
        let dir = tmp_dir("tinker_none");
        match check_tinker_atropos(&dir) {
            TinkerStatus::Missing(m) => assert!(m.contains("submodule not found")),
            _ => panic!("expected Missing"),
        }
    }

    #[test]
    fn tinker_missing_envs_dir() {
        let dir = tmp_dir("tinker_partial");
        std::fs::create_dir_all(dir.join("tinker-atropos")).unwrap();
        match check_tinker_atropos(&dir) {
            TinkerStatus::Missing(m) => assert!(m.contains("environments directory not found")),
            _ => panic!("expected Missing"),
        }
    }

    #[test]
    fn tinker_ready_counts_env_files() {
        let dir = tmp_dir("tinker_ready");
        let envs = dir.join("tinker-atropos").join("tinker_atropos").join("environments");
        std::fs::create_dir_all(&envs).unwrap();
        std::fs::write(envs.join("gsm8k_tinker.py"), "").unwrap();
        std::fs::write(envs.join("code_env.py"), "").unwrap();
        std::fs::write(envs.join("__init__.py"), "").unwrap(); // excluded (leading _)
        std::fs::write(envs.join("notes.txt"), "").unwrap(); // excluded (not .py)
        match check_tinker_atropos(&dir) {
            TinkerStatus::Ready { environments_count, .. } => assert_eq!(environments_count, 2),
            _ => panic!("expected Ready"),
        }
    }

    #[test]
    fn resolve_terminal_cwd_prefers_submodule() {
        let dir = tmp_dir("tcwd");
        let tinker = dir.join("tinker-atropos");
        std::fs::create_dir_all(&tinker).unwrap();
        let r = resolve_terminal_cwd(&dir);
        assert!(r.submodule_found);
        assert_eq!(r.cwd, tinker);
    }

    #[test]
    fn resolve_terminal_cwd_falls_back() {
        let dir = tmp_dir("tcwd_fb");
        let r = resolve_terminal_cwd(&dir);
        assert!(!r.submodule_found);
        assert_eq!(r.cwd, dir);
    }

    #[test]
    fn parse_environments_error() {
        let data = json!({"error": "boom"});
        assert_eq!(parse_environments(&data), Err("boom".to_string()));
    }

    #[test]
    fn parse_environments_empty() {
        assert!(parse_environments(&json!({})).unwrap().is_empty());
        assert!(parse_environments(&json!({"environments": []})).unwrap().is_empty());
    }

    #[test]
    fn parse_environments_full() {
        let data = json!({
            "environments": [
                {"name": "gsm8k", "class_name": "Gsm8kEnv", "file_path": "/x/gsm8k.py", "description": "math"},
                {"name": "code", "class_name": "CodeEnv", "file_path": "/x/code.py"}
            ]
        });
        let envs = parse_environments(&data).unwrap();
        assert_eq!(envs.len(), 2);
        assert_eq!(envs[0].name, "gsm8k");
        assert_eq!(envs[0].description.as_deref(), Some("math"));
        assert_eq!(envs[1].description, None);
    }

    #[test]
    fn truncate_long_description() {
        let long = "x".repeat(150);
        let t = truncate_description(&long);
        assert_eq!(t.chars().count(), 103); // 100 + "..."
        assert!(t.ends_with("..."));
        let short = "short";
        assert_eq!(truncate_description(short), "short");
    }

    #[test]
    fn classify_interactive_variants() {
        assert_eq!(classify_interactive("   "), InteractiveAction::Skip);
        assert_eq!(classify_interactive("Quit"), InteractiveAction::Quit);
        assert_eq!(classify_interactive("EXIT"), InteractiveAction::Quit);
        assert_eq!(classify_interactive("q"), InteractiveAction::Quit);
        assert_eq!(classify_interactive("Status"), InteractiveAction::Status);
        assert_eq!(
            classify_interactive("  train gsm8k  "),
            InteractiveAction::Task("train gsm8k".to_string())
        );
    }

    #[test]
    fn format_active_runs_variants() {
        assert!(format_active_runs(&json!([])).contains("No active runs"));
        let runs = json!([{"run_id": "r1", "environment": "gsm8k", "status": "running"}]);
        let s = format_active_runs(&runs);
        assert!(s.contains("r1: gsm8k (running)"));
    }

    #[test]
    fn resolve_inputs_prefer_args() {
        let cfg = HermesConfig {
            model: "cfg-model".into(),
            base_url: "cfg-url".into(),
        };
        let args = CliArgs {
            model: Some("arg-model".into()),
            base_url: None,
            ..Default::default()
        };
        let (m, b) = resolve_model_and_base_url(&args, &cfg);
        assert_eq!(m, "arg-model");
        assert_eq!(b, "cfg-url");
    }

    #[test]
    fn build_run_config_defaults() {
        let args = CliArgs::default();
        let cfg = build_run_config(
            &args,
            "m".into(),
            "b".into(),
            "k".into(),
        );
        assert_eq!(cfg.max_iterations, RL_MAX_ITERATIONS);
        assert!(cfg.save_trajectories);
        assert!(!cfg.quiet_mode);
        assert_eq!(cfg.enabled_toolsets, vec!["terminal", "web", "rl"]);
        assert_eq!(cfg.ephemeral_system_prompt, RL_SYSTEM_PROMPT);
    }

    #[test]
    fn run_check_server_branch() {
        let _g = env_lock();
        let dir = tmp_dir("run_check");
        let hh = tmp_dir("run_check_hh");
        let args = CliArgs {
            check_server: true,
            ..Default::default()
        };
        match run(&args, &dir, &hh, None) {
            RunOutcome::CheckedServer(TinkerStatus::Missing(_)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn run_requirements_failed() {
        let _g = env_lock();
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
        let dir = tmp_dir("run_req");
        let hh = tmp_dir("run_req_hh");
        let args = CliArgs {
            task: Some("do it".into()),
            ..Default::default()
        };
        match run(&args, &dir, &hh, None) {
            RunOutcome::RequirementsFailed(e) => assert_eq!(e.len(), 2),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn run_no_task_after_requirements() {
        let _g = env_lock();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "o");
            std::env::set_var("TINKER_API_KEY", "t");
            std::env::set_var("WANDB_API_KEY", "w");
        }
        let dir = tmp_dir("run_notask");
        let hh = tmp_dir("run_notask_hh");
        let args = CliArgs::default();
        let outcome = run(&args, &dir, &hh, None);
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
        assert_eq!(outcome, RunOutcome::NoTask);
    }

    struct CountingRunner {
        calls: u32,
        last: Option<String>,
    }
    impl AgentRunner for CountingRunner {
        fn run_conversation(&mut self, user_input: &str) -> Result<(), String> {
            self.calls += 1;
            self.last = Some(user_input.to_string());
            Ok(())
        }
    }

    #[test]
    fn run_single_task_invokes_runner() {
        let _g = env_lock();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "o");
            std::env::set_var("TINKER_API_KEY", "t");
            std::env::set_var("WANDB_API_KEY", "w");
        }
        let dir = tmp_dir("run_task");
        let hh = tmp_dir("run_task_hh");
        let args = CliArgs {
            task: Some("train gsm8k".into()),
            ..Default::default()
        };
        let mut runner = CountingRunner { calls: 0, last: None };
        let outcome = run(&args, &dir, &hh, Some(&mut runner));
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("TINKER_API_KEY");
            std::env::remove_var("WANDB_API_KEY");
        }
        assert!(matches!(outcome, RunOutcome::Ran(_)));
        assert_eq!(runner.calls, 1);
        assert_eq!(runner.last.as_deref(), Some("train gsm8k"));
    }

    #[test]
    fn banners_contain_expected_text() {
        assert!(header_banner().contains("RL Training Agent"));
        let s = run_summary("m", 200);
        assert!(s.contains("Model: m"));
        assert!(s.contains("Max iterations: 200"));
        assert!(s.contains("terminal, web, rl"));
        assert!(no_task_help().contains("--interactive"));
    }
}
