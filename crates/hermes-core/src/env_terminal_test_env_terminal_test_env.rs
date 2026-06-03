//! TerminalTestEnv -- simple test environment for validating the stack (native Rust port).
//!
//! Port of `environments/terminal_test_env/terminal_test_env.py`.
//!
//! A self-contained environment with inline tasks (no external dataset needed).
//! Each task asks the model to create a file at a known path with specific
//! content. The reward verifier `cat`s the file and checks whether the content
//! matches.
//!
//! The Python class subclasses `HermesAgentBaseEnv` (atroposlib-bound async
//! orchestration: agent loop, server manager, wandb plumbing). That framework
//! surface is not part of hermes-core. This port reproduces the deterministic,
//! portable logic faithfully:
//!   * inline `TRAIN_TASKS` / `EVAL_TASKS` definitions
//!   * default configuration (`TerminalTestEnvConfig`, server config)
//!   * `get_next_item` task cycling
//!   * `format_prompt`
//!   * the reward verifier scoring (`compute_reward_from_result`)
//!   * the wandb-metrics aggregation over the reward buffer
//!
//! The terminal access used by the verifier is modelled as the `TerminalResult`
//! struct (mirroring the `{"exit_code", "output"}` dict returned by
//! `ToolContext.terminal(...)`) so the scoring logic is testable without a live
//! terminal backend.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// =============================================================================
// Inline task definitions -- no external dataset needed
// =============================================================================

/// A single inline file-creation task.
///
/// Mirrors the dicts in `TRAIN_TASKS` / `EVAL_TASKS`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalTestTask {
    /// The instruction shown to the model.
    pub prompt: String,
    /// Path the verifier `cat`s to check the result.
    pub verify_path: String,
    /// Exact expected file content.
    pub expected_content: String,
}

impl TerminalTestTask {
    fn new(prompt: &str, verify_path: &str, expected_content: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            verify_path: verify_path.to_string(),
            expected_content: expected_content.to_string(),
        }
    }
}

/// The three inline training tasks (mirrors `TRAIN_TASKS`).
pub fn train_tasks() -> Vec<TerminalTestTask> {
    vec![
        TerminalTestTask::new(
            "Create a file at ~/greeting.txt containing exactly the text: Hello from Hermes Agent",
            "~/greeting.txt",
            "Hello from Hermes Agent",
        ),
        TerminalTestTask::new(
            "Create a file at ~/count.txt containing the numbers 1 through 5, one per line",
            "~/count.txt",
            "1\n2\n3\n4\n5",
        ),
        TerminalTestTask::new(
            "Create a file at ~/answer.txt containing the result of 123 + 456",
            "~/answer.txt",
            "579",
        ),
    ]
}

/// The single inline eval task (mirrors `EVAL_TASKS`).
pub fn eval_tasks() -> Vec<TerminalTestTask> {
    vec![TerminalTestTask::new(
        "Create a file at ~/result.txt containing the result of 6 * 7",
        "~/result.txt",
        "42",
    )]
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the terminal test environment.
///
/// Mirrors the defaults set in `TerminalTestEnv.config_init()` (the subset of
/// `HermesAgentEnvConfig` actually populated there). The broader agent-loop /
/// atroposlib config fields are framework concerns and not modelled here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalTestEnvConfig {
    pub enabled_toolsets: Vec<String>,
    pub disabled_toolsets: Option<Vec<String>>,
    pub distribution: Option<String>,
    pub max_agent_turns: u32,
    pub max_token_length: u32,
    pub agent_temperature: f64,
    pub system_prompt: String,
    pub terminal_backend: String,
    pub group_size: u32,
    pub tokenizer_name: String,
    pub tool_call_parser: String,
    pub steps_per_eval: u32,
    pub total_steps: u32,
    pub use_wandb: bool,
    pub wandb_name: String,
    pub ensure_scores_are_not_same: bool,
    pub dataset_name: Option<String>,
}

impl Default for TerminalTestEnvConfig {
    fn default() -> Self {
        // Matches the values in `config_init()`.
        Self {
            enabled_toolsets: vec!["terminal".to_string(), "file".to_string()],
            disabled_toolsets: None,
            distribution: None,
            max_agent_turns: 10,
            max_token_length: 16000,
            agent_temperature: 1.0,
            system_prompt:
                "You are a helpful assistant with access to a terminal and file tools. \
                 Complete the user's request by using the available tools. \
                 Be precise and follow instructions exactly."
                    .to_string(),
            terminal_backend: "modal".to_string(),
            group_size: 3,
            tokenizer_name: "NousResearch/q-30b-t-h45-e1".to_string(),
            tool_call_parser: "hermes".to_string(),
            steps_per_eval: 3,
            total_steps: 3,
            use_wandb: true,
            wandb_name: "terminal-test".to_string(),
            ensure_scores_are_not_same: false,
            dataset_name: None,
        }
    }
}

/// Minimal mirror of the single `APIServerConfig` built in `config_init()`.
///
/// OpenRouter with Claude; API key loaded from the `OPENROUTER_API_KEY` env var
/// (empty string if unset, matching `os.getenv("OPENROUTER_API_KEY", "")`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub base_url: String,
    pub model_name: String,
    pub server_type: String,
    pub api_key: String,
    pub health_check: bool,
}

impl ServerConfig {
    /// Build the default OpenRouter/Claude server config, reading the API key
    /// from the environment exactly as the Python code does.
    pub fn default_openrouter() -> Self {
        Self {
            base_url: "https://openrouter.ai/api/v1".to_string(),
            model_name: "anthropic/claude-opus-4.6".to_string(),
            server_type: "openai".to_string(),
            api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            health_check: false,
        }
    }
}

/// Mirror of `TerminalTestEnv.config_init()` -- returns the env config plus the
/// list of server configs.
pub fn config_init() -> (TerminalTestEnvConfig, Vec<ServerConfig>) {
    (
        TerminalTestEnvConfig::default(),
        vec![ServerConfig::default_openrouter()],
    )
}

// =============================================================================
// Environment state
// =============================================================================

/// Result of running a terminal command, mirroring the dict returned by
/// `ToolContext.terminal(cmd)` -> `{"exit_code": int, "output": str, ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalResult {
    pub exit_code: i32,
    #[serde(default)]
    pub output: String,
}

/// The portable state of `TerminalTestEnv`: task lists, the cycling iterator,
/// and the reward buffer used for wandb logging.
#[derive(Debug, Clone)]
pub struct TerminalTestEnv {
    pub config: TerminalTestEnvConfig,
    pub train_tasks: Vec<TerminalTestTask>,
    pub eval_tasks: Vec<TerminalTestTask>,
    /// Cursor used by `get_next_item` (Python's `self.iter`).
    pub iter: usize,
    /// Per-rollout reward history; drained by `wandb_log`.
    pub reward_buffer: Vec<f64>,
}

impl Default for TerminalTestEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalTestEnv {
    /// Build a fresh environment (mirrors `setup()` plus the default config).
    pub fn new() -> Self {
        Self::with_config(TerminalTestEnvConfig::default())
    }

    /// Build with an explicit config; initialises task lists and buffers,
    /// mirroring `setup()`.
    pub fn with_config(config: TerminalTestEnvConfig) -> Self {
        Self {
            config,
            train_tasks: train_tasks(),
            eval_tasks: eval_tasks(),
            iter: 0,
            reward_buffer: Vec::new(),
        }
    }

    /// Cycle through training tasks (mirrors `get_next_item`).
    ///
    /// Returns a clone of the next task and advances the iterator.
    pub fn get_next_item(&mut self) -> TerminalTestTask {
        let n = self.train_tasks.len();
        let item = self.train_tasks[self.iter % n].clone();
        self.iter += 1;
        item
    }

    /// The prompt is directly in the task item (mirrors `format_prompt`).
    pub fn format_prompt(item: &TerminalTestTask) -> String {
        item.prompt.clone()
    }

    /// Verify by checking the `cat` result against the expected content,
    /// recording the score in `reward_buffer` (mirrors `compute_reward`).
    ///
    /// Scoring:
    ///   * 1.0 = exact match
    ///   * 0.5 = expected content is present but with extra stuff
    ///   * 0.0 = file doesn't exist (non-zero exit) or content doesn't match
    ///
    /// `verify_result` is the value `ToolContext.terminal("cat <verify_path>")`
    /// would return.
    pub fn compute_reward(
        &mut self,
        item: &TerminalTestTask,
        verify_result: &TerminalResult,
    ) -> f64 {
        let reward = score_reward(item, verify_result);
        self.reward_buffer.push(reward);
        reward
    }

    /// Compute the wandb metrics from the current reward buffer, then drain it
    /// (mirrors the buffer-handling logic in `wandb_log`).
    ///
    /// Returns the metrics that would be merged into `wandb_metrics`. If the
    /// buffer is empty, returns an empty map and leaves the buffer untouched
    /// (matching the Python `if self.reward_buffer:` guard).
    pub fn wandb_log(&mut self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        if self.reward_buffer.is_empty() {
            return metrics;
        }

        let total = self.reward_buffer.len();
        let correct = self.reward_buffer.iter().filter(|&&r| r == 1.0).count();
        let partial = self.reward_buffer.iter().filter(|&&r| r == 0.5).count();
        let sum: f64 = self.reward_buffer.iter().sum();

        metrics.insert("train/avg_reward".to_string(), sum / total as f64);
        metrics.insert("train/accuracy".to_string(), correct as f64 / total as f64);
        metrics.insert(
            "train/partial_match_rate".to_string(),
            partial as f64 / total as f64,
        );
        metrics.insert("train/total_rollouts".to_string(), total as f64);

        self.reward_buffer.clear();
        metrics
    }
}

/// Pure scoring helper backing `compute_reward` (no side effects).
///
/// Mirrors the body of `compute_reward`: non-zero exit -> 0.0; exact (after
/// `.strip()`) -> 1.0; expected-substring-present -> 0.5; else 0.0.
pub fn score_reward(item: &TerminalTestTask, verify_result: &TerminalResult) -> f64 {
    // File doesn't exist or can't be read.
    if verify_result.exit_code != 0 {
        return 0.0;
    }

    let actual = verify_result.output.trim();
    let expected = item.expected_content.trim();

    if actual == expected {
        return 1.0;
    }

    // Partial credit: expected content present but with extra stuff.
    if actual.contains(expected) {
        return 0.5;
    }

    0.0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(output: &str) -> TerminalResult {
        TerminalResult {
            exit_code: 0,
            output: output.to_string(),
        }
    }

    #[test]
    fn test_task_lists() {
        let train = train_tasks();
        assert_eq!(train.len(), 3);
        assert_eq!(train[0].verify_path, "~/greeting.txt");
        assert_eq!(train[1].expected_content, "1\n2\n3\n4\n5");
        assert_eq!(train[2].expected_content, "579");

        let eval = eval_tasks();
        assert_eq!(eval.len(), 1);
        assert_eq!(eval[0].verify_path, "~/result.txt");
        assert_eq!(eval[0].expected_content, "42");
    }

    #[test]
    fn test_get_next_item_cycles() {
        let mut env = TerminalTestEnv::new();
        let a = env.get_next_item();
        let b = env.get_next_item();
        let c = env.get_next_item();
        let d = env.get_next_item(); // wraps to first
        assert_eq!(a.verify_path, "~/greeting.txt");
        assert_eq!(b.verify_path, "~/count.txt");
        assert_eq!(c.verify_path, "~/answer.txt");
        assert_eq!(d.verify_path, "~/greeting.txt");
        assert_eq!(env.iter, 4);
    }

    #[test]
    fn test_format_prompt() {
        let item = &train_tasks()[0];
        assert_eq!(TerminalTestEnv::format_prompt(item), item.prompt);
    }

    #[test]
    fn test_score_exact_match() {
        let item = &train_tasks()[0];
        assert_eq!(score_reward(item, &ok("Hello from Hermes Agent")), 1.0);
        // Trailing whitespace is stripped on both sides.
        assert_eq!(score_reward(item, &ok("  Hello from Hermes Agent\n")), 1.0);
    }

    #[test]
    fn test_score_partial_match() {
        let item = &train_tasks()[0];
        let r = ok("Hello from Hermes Agent\nand some extra trailing line");
        assert_eq!(score_reward(item, &r), 0.5);
    }

    #[test]
    fn test_score_no_match() {
        let item = &train_tasks()[2]; // expected "579"
        assert_eq!(score_reward(item, &ok("wrong answer")), 0.0);
    }

    #[test]
    fn test_score_nonzero_exit() {
        let item = &train_tasks()[0];
        let r = TerminalResult {
            exit_code: 1,
            output: "cat: no such file".to_string(),
        };
        assert_eq!(score_reward(item, &r), 0.0);
    }

    #[test]
    fn test_compute_reward_records_buffer() {
        let mut env = TerminalTestEnv::new();
        let item = env.train_tasks[0].clone();
        env.compute_reward(&item, &ok("Hello from Hermes Agent"));
        env.compute_reward(&item, &ok("Hello from Hermes Agent extra"));
        env.compute_reward(
            &item,
            &TerminalResult {
                exit_code: 2,
                output: String::new(),
            },
        );
        assert_eq!(env.reward_buffer, vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn test_wandb_log_metrics_and_drain() {
        let mut env = TerminalTestEnv::new();
        env.reward_buffer = vec![1.0, 1.0, 0.5, 0.0];
        let m = env.wandb_log();
        assert_eq!(m["train/total_rollouts"], 4.0);
        assert!((m["train/avg_reward"] - 2.5 / 4.0).abs() < 1e-12);
        assert!((m["train/accuracy"] - 2.0 / 4.0).abs() < 1e-12);
        assert!((m["train/partial_match_rate"] - 1.0 / 4.0).abs() < 1e-12);
        // Buffer drained.
        assert!(env.reward_buffer.is_empty());
    }

    #[test]
    fn test_wandb_log_empty_buffer() {
        let mut env = TerminalTestEnv::new();
        let m = env.wandb_log();
        assert!(m.is_empty());
    }

    #[test]
    fn test_default_config() {
        let cfg = TerminalTestEnvConfig::default();
        assert_eq!(cfg.enabled_toolsets, vec!["terminal", "file"]);
        assert_eq!(cfg.max_agent_turns, 10);
        assert_eq!(cfg.group_size, 3);
        assert_eq!(cfg.terminal_backend, "modal");
        assert!(!cfg.ensure_scores_are_not_same);
        assert!(cfg.dataset_name.is_none());
    }

    #[test]
    fn test_config_init_server() {
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
        let (_cfg, servers) = config_init();
        assert_eq!(servers.len(), 1);
        let s = &servers[0];
        assert_eq!(s.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(s.model_name, "anthropic/claude-opus-4.6");
        assert_eq!(s.server_type, "openai");
        assert!(!s.health_check);
        assert_eq!(s.api_key, "");
    }

    #[test]
    fn test_server_reads_api_key_env() {
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-123");
        }
        let s = ServerConfig::default_openrouter();
        assert_eq!(s.api_key, "test-key-123");
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
    }
}
