//! HermesSweEnv -- SWE-Bench Style Environment with Modal Sandboxes (native Rust port).
//!
//! Port of `environments/hermes_swe_env/hermes_swe_env.py`.
//!
//! A concrete environment for software-engineering tasks where the model writes
//! code and the reward function runs tests to verify correctness. The Python
//! flow drives a `HermesAgentBaseEnv` (atroposlib-bound async agent loop) against
//! a Modal-backed terminal sandbox; the framework plumbing (async base env,
//! atroposlib server manager, wandb logging, `datasets.load_dataset`) is not part
//! of hermes-core. This port reproduces the deterministic, portable logic
//! faithfully:
//!   * default configuration (`HermesSweEnvConfig` / `config_init` defaults)
//!   * dataset cycling (`get_next_item`)
//!   * prompt formatting (`format_prompt`)
//!   * the reward function (`compute_reward`), parameterised over a terminal
//!     callback so callers can wire in their Modal sandbox `ToolContext.terminal`
//!   * a reward buffer with the wandb-metric aggregation (`wandb_log`)
//!
//! The async agent driving, dataset loading, and atroposlib/wandb plumbing are
//! left as integration concerns for the caller.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the SWE environment.
///
/// Mirrors `HermesSweEnvConfig` together with the defaults supplied by
/// `config_init()`. Only the fields actually consumed by the portable logic
/// (dataset cycling, prompt formatting, reward computation) are modelled with
/// behavior; the broader agent-loop knobs are carried so the shape matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesSweEnvConfig {
    pub enabled_toolsets: Vec<String>,
    pub disabled_toolsets: Option<Vec<String>>,
    pub distribution: Option<String>,
    pub max_agent_turns: u32,
    pub max_token_length: u32,
    pub agent_temperature: f64,
    pub system_prompt: String,
    pub terminal_backend: String,
    pub dataset_name: String,
    pub dataset_split: String,
    /// Field of the dataset item used as the base prompt text.
    pub prompt_field: String,
    pub group_size: u32,
    pub tokenizer_name: String,
    pub tool_call_parser: String,
    pub steps_per_eval: u32,
    pub total_steps: u32,
    pub use_wandb: bool,
    pub wandb_name: String,
}

/// System prompt presented to the SWE agent (from `config_init`).
pub const SWE_SYSTEM_PROMPT: &str = "You are a skilled software engineer. You have access to a terminal, \
file tools, and web search. Use these tools to complete the coding task. \
Write clean, working code and verify it runs correctly before finishing.";

impl Default for HermesSweEnvConfig {
    fn default() -> Self {
        // Matches the values used in `config_init()` of the Python source.
        Self {
            enabled_toolsets: vec![
                "terminal".to_string(),
                "file".to_string(),
                "web".to_string(),
            ],
            disabled_toolsets: None,
            distribution: None,
            max_agent_turns: 30,
            max_token_length: 4096,
            agent_temperature: 1.0,
            system_prompt: SWE_SYSTEM_PROMPT.to_string(),
            terminal_backend: "modal".to_string(),
            dataset_name: "bigcode/humanevalpack".to_string(),
            dataset_split: "test".to_string(),
            prompt_field: "prompt".to_string(),
            group_size: 4,
            tokenizer_name: "NousResearch/DeepHermes-3-Llama-3-3B-Preview".to_string(),
            tool_call_parser: "hermes".to_string(),
            steps_per_eval: 50,
            total_steps: 500,
            use_wandb: true,
            wandb_name: "hermes-swe".to_string(),
        }
    }
}

/// Default API server config (mirrors the single `APIServerConfig` in
/// `config_init`'s `server_configs`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiServerConfig {
    pub base_url: String,
    pub model_name: String,
    pub server_type: String,
    pub api_key: String,
}

impl Default for ApiServerConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000/v1".to_string(),
            model_name: "NousResearch/DeepHermes-3-Llama-3-3B-Preview".to_string(),
            server_type: "openai".to_string(),
            api_key: String::new(),
        }
    }
}

/// Mirror of `config_init()` -- returns the default env config and server
/// configs.
pub fn config_init() -> (HermesSweEnvConfig, Vec<ApiServerConfig>) {
    (HermesSweEnvConfig::default(), vec![ApiServerConfig::default()])
}

// =============================================================================
// Dataset item helpers
// =============================================================================

/// A dataset item is a free-form JSON object (mirrors `Dict[str, Any]`).
pub type Item = Map<String, Value>;

/// Extract a string field from an item, returning `""` when absent or non-string.
///
/// Mirrors Python's `item.get(field, "")` followed by string interpolation:
/// only string values yield text; missing keys yield the empty string. (Numeric
/// or other JSON types are stringified to match `f"...{value}..."` semantics.)
fn item_str(item: &Item, field: &str) -> String {
    match item.get(field) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Resolve the test code from an item.
///
/// Mirrors `item.get("test", item.get("test_code", item.get("tests", "")))`:
/// precedence is `test` > `test_code` > `tests` > `""`. Note Python's nested
/// `.get` only falls through when a key is *absent*; a present-but-empty string
/// short-circuits. We replicate that: the first key that is *present* wins,
/// even if its value is an empty string.
fn resolve_test_code(item: &Item) -> String {
    if item.contains_key("test") {
        item_str(item, "test")
    } else if item.contains_key("test_code") {
        item_str(item, "test_code")
    } else if item.contains_key("tests") {
        item_str(item, "tests")
    } else {
        String::new()
    }
}

// =============================================================================
// Dataset cycling
// =============================================================================

/// Minimal stand-in for the env's dataset + cursor state.
///
/// Mirrors `self.dataset`, `self.iter`, and `self.reward_buffer` set up in
/// `setup()`.
#[derive(Debug, Clone, Default)]
pub struct HermesSweEnv {
    pub config: HermesSweEnvConfig,
    pub dataset: Vec<Item>,
    pub iter: usize,
    pub reward_buffer: Vec<f64>,
}

impl HermesSweEnv {
    /// Construct with a config and dataset (mirrors `setup()` post-conditions).
    pub fn new(config: HermesSweEnvConfig, dataset: Vec<Item>) -> Self {
        Self {
            config,
            dataset,
            iter: 0,
            reward_buffer: Vec::new(),
        }
    }

    /// Cycle through the SWE dataset.
    ///
    /// Mirrors `get_next_item`: raises (here: returns `Err`) when no dataset is
    /// loaded, otherwise returns `dataset[iter % len]` and advances the cursor.
    pub fn get_next_item(&mut self) -> Result<Item, String> {
        if self.dataset.is_empty() {
            return Err("No dataset loaded. Set dataset_name in config.".to_string());
        }
        let idx = self.iter % self.dataset.len();
        let item = self.dataset[idx].clone();
        self.iter += 1;
        Ok(item)
    }

    /// Format the SWE task prompt for an item.
    ///
    /// Delegates to the free function `format_prompt`, using this env's
    /// `prompt_field`.
    pub fn format_prompt(&self, item: &Item) -> String {
        format_prompt(item, &self.config.prompt_field)
    }

    /// Score by running tests in the model's sandbox.
    ///
    /// Mirrors `compute_reward`. `terminal` is a callback equivalent to
    /// `ToolContext.terminal(cmd, timeout)`; it returns a `TerminalResult`.
    /// Appends the score to `reward_buffer` (matching the Python side-effect) and
    /// returns it.
    pub fn compute_reward<F>(&mut self, item: &Item, mut terminal: F) -> f64
    where
        F: FnMut(&str, Option<u64>) -> TerminalResult,
    {
        let score = compute_reward(item, &mut terminal);
        self.reward_buffer.push(score);
        score
    }

    /// Drain the reward buffer into wandb metrics.
    ///
    /// Mirrors `wandb_log`: when the buffer is non-empty, inserts
    /// `train/avg_reward` and `train/pass_rate` into the metrics map and clears
    /// the buffer. Returns the (possibly augmented) metrics.
    pub fn wandb_log(&mut self, wandb_metrics: Option<Map<String, Value>>) -> Map<String, Value> {
        let mut metrics = wandb_metrics.unwrap_or_default();
        if !self.reward_buffer.is_empty() {
            let n = self.reward_buffer.len() as f64;
            let avg = self.reward_buffer.iter().sum::<f64>() / n;
            // pass_rate counts only exact-1.0 rewards, matching `r == 1.0`.
            let passes = self
                .reward_buffer
                .iter()
                .filter(|&&r| r == 1.0)
                .count() as f64;
            let pass_rate = passes / n;
            metrics.insert(
                "train/avg_reward".to_string(),
                Value::from(avg),
            );
            metrics.insert(
                "train/pass_rate".to_string(),
                Value::from(pass_rate),
            );
            self.reward_buffer.clear();
        }
        metrics
    }
}

// =============================================================================
// Prompt formatting (free function)
// =============================================================================

/// Format the SWE task prompt.
///
/// Mirrors `format_prompt`: takes the base prompt from `prompt_field`, then -- if
/// the item carries test information under `test` / `test_code` / `tests` -- appends
/// a `"\n\nTests to pass:\n{test_info}"` block. The append only happens when the
/// resolved test string is non-empty (Python truthiness).
pub fn format_prompt(item: &Item, prompt_field: &str) -> String {
    let mut prompt = item_str(item, prompt_field);
    let test_info = resolve_test_code(item);
    if !test_info.is_empty() {
        prompt.push_str(&format!("\n\nTests to pass:\n{test_info}"));
    }
    prompt
}

// =============================================================================
// Reward computation
// =============================================================================

/// Result of executing a command in the sandbox terminal.
///
/// Mirrors the dict returned by `ToolContext.terminal(...)`, which carries at
/// least `exit_code` and (optionally) `output`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TerminalResult {
    pub exit_code: i64,
    pub output: String,
}

impl TerminalResult {
    pub fn new(exit_code: i64, output: impl Into<String>) -> Self {
        Self {
            exit_code,
            output: output.into(),
        }
    }
}

/// Build the test command run inside the sandbox.
///
/// Mirrors `f'cd /workspace && python3 -c "{test_code}"'`.
pub fn build_test_command(test_code: &str) -> String {
    format!("cd /workspace && python3 -c \"{test_code}\"")
}

/// The exact `find` command used for the partial-credit file check.
pub const FILE_CHECK_COMMAND: &str =
    "find /workspace -name '*.py' -newer /tmp/.start_marker 2>/dev/null | head -5";

/// Compute the reward for a SWE rollout.
///
/// Mirrors `compute_reward`:
///   * If the item carries test code, run `python3 -c "<test>"` in `/workspace`
///     with a 60s timeout; exit code 0 => reward 1.0.
///   * Otherwise (or on non-zero exit), check for newly created `*.py` files;
///     exit code 0 with non-empty trimmed output => reward 0.1.
///   * Else => reward 0.0.
///
/// `terminal` mirrors `ToolContext.terminal(cmd, timeout)`.
pub fn compute_reward<F>(item: &Item, terminal: &mut F) -> f64
where
    F: FnMut(&str, Option<u64>) -> TerminalResult,
{
    let test_code = resolve_test_code(item);

    if !test_code.is_empty() {
        let cmd = build_test_command(&test_code);
        let test_result = terminal(&cmd, Some(60));
        if test_result.exit_code == 0 {
            return 1.0;
        }
    }

    // Partial credit: check whether the model created any new Python files.
    let file_check = terminal(FILE_CHECK_COMMAND, None);
    if file_check.exit_code == 0 && !file_check.output.trim().is_empty() {
        return 0.1;
    }

    0.0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item_from(v: Value) -> Item {
        match v {
            Value::Object(m) => m,
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn test_config_defaults() {
        let (cfg, servers) = config_init();
        assert_eq!(cfg.enabled_toolsets, vec!["terminal", "file", "web"]);
        assert_eq!(cfg.max_agent_turns, 30);
        assert_eq!(cfg.terminal_backend, "modal");
        assert_eq!(cfg.dataset_name, "bigcode/humanevalpack");
        assert_eq!(cfg.prompt_field, "prompt");
        assert_eq!(cfg.group_size, 4);
        assert!(cfg.use_wandb);
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].server_type, "openai");
        assert_eq!(servers[0].base_url, "http://localhost:8000/v1");
    }

    #[test]
    fn test_get_next_item_empty_errors() {
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), vec![]);
        assert!(env.get_next_item().is_err());
    }

    #[test]
    fn test_get_next_item_cycles() {
        let ds = vec![
            item_from(json!({"prompt": "a"})),
            item_from(json!({"prompt": "b"})),
        ];
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), ds);
        assert_eq!(item_str(&env.get_next_item().unwrap(), "prompt"), "a");
        assert_eq!(item_str(&env.get_next_item().unwrap(), "prompt"), "b");
        // wraps around
        assert_eq!(item_str(&env.get_next_item().unwrap(), "prompt"), "a");
        assert_eq!(env.iter, 3);
    }

    #[test]
    fn test_format_prompt_no_tests() {
        let item = item_from(json!({"prompt": "write a function"}));
        assert_eq!(format_prompt(&item, "prompt"), "write a function");
    }

    #[test]
    fn test_format_prompt_with_test() {
        let item = item_from(json!({"prompt": "p", "test": "assert f() == 1"}));
        assert_eq!(
            format_prompt(&item, "prompt"),
            "p\n\nTests to pass:\nassert f() == 1"
        );
    }

    #[test]
    fn test_format_prompt_test_precedence() {
        // `test` wins over `test_code` and `tests`.
        let item = item_from(json!({
            "prompt": "p",
            "test": "T1",
            "test_code": "T2",
            "tests": "T3"
        }));
        assert!(format_prompt(&item, "prompt").ends_with("Tests to pass:\nT1"));
        // falls through to test_code when `test` absent
        let item2 = item_from(json!({"prompt": "p", "test_code": "T2", "tests": "T3"}));
        assert!(format_prompt(&item2, "prompt").ends_with("Tests to pass:\nT2"));
        // then tests
        let item3 = item_from(json!({"prompt": "p", "tests": "T3"}));
        assert!(format_prompt(&item3, "prompt").ends_with("Tests to pass:\nT3"));
    }

    #[test]
    fn test_format_prompt_empty_test_present_no_append() {
        // present-but-empty test => no append (Python truthiness short-circuit).
        let item = item_from(json!({"prompt": "p", "test": ""}));
        assert_eq!(format_prompt(&item, "prompt"), "p");
    }

    #[test]
    fn test_format_prompt_missing_prompt_field() {
        let item = item_from(json!({"other": "x"}));
        assert_eq!(format_prompt(&item, "prompt"), "");
    }

    #[test]
    fn test_build_test_command() {
        assert_eq!(
            build_test_command("print(1)"),
            "cd /workspace && python3 -c \"print(1)\""
        );
    }

    #[test]
    fn test_compute_reward_test_pass() {
        let item = item_from(json!({"prompt": "p", "test": "assert True"}));
        let mut calls: Vec<String> = Vec::new();
        let mut term = |cmd: &str, _t: Option<u64>| {
            calls.push(cmd.to_string());
            TerminalResult::new(0, "")
        };
        let r = compute_reward(&item, &mut term);
        assert_eq!(r, 1.0);
        // only the test command should have run
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains("assert True"));
    }

    #[test]
    fn test_compute_reward_test_fail_then_partial() {
        let item = item_from(json!({"prompt": "p", "test": "assert False"}));
        let mut step = 0;
        let mut term = |cmd: &str, _t: Option<u64>| {
            step += 1;
            if cmd.contains("find /workspace") {
                TerminalResult::new(0, "solution.py\n")
            } else {
                TerminalResult::new(1, "AssertionError")
            }
        };
        let r = compute_reward(&item, &mut term);
        assert_eq!(r, 0.1);
        assert_eq!(step, 2);
    }

    #[test]
    fn test_compute_reward_no_test_partial() {
        let item = item_from(json!({"prompt": "p"}));
        let mut term = |_cmd: &str, _t: Option<u64>| TerminalResult::new(0, "new.py");
        assert_eq!(compute_reward(&item, &mut term), 0.1);
    }

    #[test]
    fn test_compute_reward_zero() {
        let item = item_from(json!({"prompt": "p", "test": "assert False"}));
        let mut term = |cmd: &str, _t: Option<u64>| {
            if cmd.contains("find /workspace") {
                // exit 0 but empty output => not partial credit
                TerminalResult::new(0, "   \n  ")
            } else {
                TerminalResult::new(1, "fail")
            }
        };
        assert_eq!(compute_reward(&item, &mut term), 0.0);
    }

    #[test]
    fn test_compute_reward_file_check_nonzero_exit() {
        let item = item_from(json!({"prompt": "p"}));
        let mut term = |_cmd: &str, _t: Option<u64>| TerminalResult::new(2, "has output");
        // non-zero exit on file check => 0.0 even with output
        assert_eq!(compute_reward(&item, &mut term), 0.0);
    }

    #[test]
    fn test_env_compute_reward_buffers() {
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), vec![]);
        let item = item_from(json!({"prompt": "p", "test": "assert True"}));
        let r = env.compute_reward(&item, |_c, _t| TerminalResult::new(0, ""));
        assert_eq!(r, 1.0);
        assert_eq!(env.reward_buffer, vec![1.0]);
    }

    #[test]
    fn test_wandb_log_aggregation() {
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), vec![]);
        env.reward_buffer = vec![1.0, 0.0, 1.0, 0.1];
        let m = env.wandb_log(None);
        let avg = m["train/avg_reward"].as_f64().unwrap();
        assert!((avg - (2.1 / 4.0)).abs() < 1e-12);
        let pr = m["train/pass_rate"].as_f64().unwrap();
        assert!((pr - 0.5).abs() < 1e-12);
        // buffer cleared
        assert!(env.reward_buffer.is_empty());
    }

    #[test]
    fn test_wandb_log_empty_buffer_no_keys() {
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), vec![]);
        let m = env.wandb_log(None);
        assert!(!m.contains_key("train/avg_reward"));
        assert!(!m.contains_key("train/pass_rate"));
    }

    #[test]
    fn test_wandb_log_preserves_existing_metrics() {
        let mut env = HermesSweEnv::new(HermesSweEnvConfig::default(), vec![]);
        env.reward_buffer = vec![1.0];
        let mut existing = Map::new();
        existing.insert("foo".to_string(), Value::from(42));
        let m = env.wandb_log(Some(existing));
        assert_eq!(m["foo"], Value::from(42));
        assert_eq!(m["train/pass_rate"].as_f64().unwrap(), 1.0);
    }
}
