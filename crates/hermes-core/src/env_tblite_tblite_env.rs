//! OpenThoughts-TBLite evaluation environment (native Rust port).
//!
//! Port of `environments/benchmarks/tblite/tblite_env.py`.
//!
//! TBLite is a lighter, faster alternative to Terminal-Bench 2.0. It reuses
//! *all* of the TB2 evaluation logic (agent loop, test verification, Docker
//! image resolution, metrics, wandb logging) and only differs in its default
//! configuration:
//!
//!   * `dataset_name` defaults to `"NousResearch/openthoughts-tblite"`
//!     (100 difficulty-calibrated tasks) instead of TB2's
//!     `"NousResearch/terminal-bench-2"`.
//!   * `task_timeout` defaults to `1200` (20 minutes) instead of TB2's `1800`,
//!     since TBLite tasks are generally faster.
//!
//! In Python, `TBLiteEvalConfig` subclasses `TerminalBench2EvalConfig` and
//! `TBLiteEvalEnv` subclasses `TerminalBench2EvalEnv`, overriding only the two
//! field defaults and the run defaults assembled in `config_init`. Here we mirror
//! that by reusing [`crate::env_terminalbench_2_terminalbench2_env`] types and
//! producing a config whose values match the TBLite `config_init`.

use crate::env_terminalbench_2_terminalbench2_env::{ApiServerConfig, TerminalBench2EvalConfig};

/// HuggingFace dataset containing TBLite tasks (the TBLite `dataset_name`
/// Pydantic field default).
pub const TBLITE_DATASET_NAME: &str = "NousResearch/openthoughts-tblite";

/// Maximum wall-clock seconds per task (the TBLite `task_timeout` Pydantic field
/// default). TBLite tasks are generally faster than TB2, so 20 minutes is
/// usually sufficient.
pub const TBLITE_TASK_TIMEOUT: i64 = 1200;

/// The environment's `name` class attribute.
pub const TBLITE_ENV_NAME: &str = "openthoughts-tblite";

/// Marker type mirroring `TBLiteEvalConfig`.
///
/// The Python `TBLiteEvalConfig` is a `TerminalBench2EvalConfig` subclass that
/// only changes the `dataset_name` and `task_timeout` field-level defaults. We
/// represent it as a thin wrapper over [`TerminalBench2EvalConfig`] that knows
/// how to produce TBLite's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct TBLiteEvalConfig;

impl TBLiteEvalConfig {
    /// Build a [`TerminalBench2EvalConfig`] with TBLite's *field-level* defaults
    /// applied on top of the TB2 base defaults.
    ///
    /// This mirrors instantiating `TBLiteEvalConfig()` with no run overrides:
    /// every TB2 field default is kept except `dataset_name` and `task_timeout`,
    /// which take TBLite's values.
    pub fn field_defaults() -> TerminalBench2EvalConfig {
        let mut cfg = TerminalBench2EvalConfig::default();
        cfg.dataset_name = TBLITE_DATASET_NAME.to_string();
        cfg.task_timeout = TBLITE_TASK_TIMEOUT;
        cfg
    }

    /// Build the default TBLite eval configuration produced by
    /// `TBLiteEvalEnv.config_init`.
    ///
    /// Replicates the explicit keyword arguments passed in the Python
    /// `config_init`. Note that `config_init` does *not* override
    /// `dataset_name` or `task_timeout`, so those keep the TBLite field-level
    /// defaults set in [`Self::field_defaults`].
    pub fn config_init() -> TerminalBench2EvalConfig {
        let mut cfg = Self::field_defaults();

        // --- Explicit overrides from the Python `config_init` ---
        cfg.enabled_toolsets = vec!["terminal".to_string(), "file".to_string()];
        // disabled_toolsets=None / distribution=None are not modelled fields.

        cfg.max_agent_turns = 60;
        cfg.max_token_length = 16000;
        cfg.agent_temperature = 0.6;
        cfg.system_prompt = None;

        cfg.terminal_backend = "modal".to_string();
        cfg.terminal_timeout = 300;

        cfg.test_timeout = 180;

        // 100 tasks in parallel.
        cfg.tool_pool_size = 128;

        // eval_handling=EvalHandlingEnum.STOP_TRAIN is a base-env concern not
        // represented in TerminalBench2EvalConfig.
        cfg.group_size = 1;
        cfg.steps_per_eval = 1;
        cfg.total_steps = 1;

        cfg.tokenizer_name = "NousResearch/Hermes-3-Llama-3.1-8B".to_string();
        cfg.use_wandb = true;
        cfg.wandb_name = "openthoughts-tblite".to_string();
        cfg.ensure_scores_are_not_same = false;

        cfg
    }
}

/// Default OpenRouter/Claude server config produced by TBLite's `config_init`.
///
/// Identical to the TB2 server config: `base_url` is the OpenRouter API, the
/// model is `anthropic/claude-sonnet-4`, the server type is `openai`, the API
/// key is read from `OPENROUTER_API_KEY` (empty string if unset), and health
/// checks are disabled.
pub fn tblite_server_config() -> ApiServerConfig {
    // The Python `config_init` server block is byte-for-byte the same as TB2's,
    // so we reuse the TB2 default directly.
    ApiServerConfig::tb2_default()
}

/// Mirror of `TBLiteEvalEnv.config_init` returning both the env config and the
/// server configs, matching the Python `Tuple[TBLiteEvalConfig, List[APIServerConfig]]`.
pub fn config_init() -> (TerminalBench2EvalConfig, Vec<ApiServerConfig>) {
    (TBLiteEvalConfig::config_init(), vec![tblite_server_config()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_defaults_override_only_dataset_and_timeout() {
        let tb2 = TerminalBench2EvalConfig::default();
        let lite = TBLiteEvalConfig::field_defaults();

        // Overridden TBLite fields.
        assert_eq!(lite.dataset_name, "NousResearch/openthoughts-tblite");
        assert_eq!(lite.task_timeout, 1200);

        // TB2 had different values for exactly these two fields.
        assert_eq!(tb2.dataset_name, "NousResearch/terminal-bench-2");
        assert_eq!(tb2.task_timeout, 1800);

        // Everything else is inherited unchanged from TB2 base defaults.
        let mut tb2_as_lite = tb2.clone();
        tb2_as_lite.dataset_name = lite.dataset_name.clone();
        tb2_as_lite.task_timeout = lite.task_timeout;
        assert_eq!(tb2_as_lite, lite);
    }

    #[test]
    fn config_init_matches_python_defaults() {
        let c = TBLiteEvalConfig::config_init();

        // Field-level TBLite defaults (not touched by config_init).
        assert_eq!(c.dataset_name, "NousResearch/openthoughts-tblite");
        assert_eq!(c.task_timeout, 1200);

        // Explicit config_init overrides.
        assert_eq!(c.enabled_toolsets, vec!["terminal", "file"]);
        assert_eq!(c.max_agent_turns, 60);
        assert_eq!(c.max_token_length, 16000);
        assert_eq!(c.agent_temperature, 0.6);
        assert_eq!(c.system_prompt, None);
        assert_eq!(c.terminal_backend, "modal");
        assert_eq!(c.terminal_timeout, 300);
        assert_eq!(c.test_timeout, 180);
        assert_eq!(c.tool_pool_size, 128);
        assert_eq!(c.group_size, 1);
        assert_eq!(c.steps_per_eval, 1);
        assert_eq!(c.total_steps, 1);
        assert_eq!(c.tokenizer_name, "NousResearch/Hermes-3-Llama-3.1-8B");
        assert!(c.use_wandb);
        assert_eq!(c.wandb_name, "openthoughts-tblite");
        assert!(!c.ensure_scores_are_not_same);
    }

    #[test]
    fn env_name_is_openthoughts_tblite() {
        assert_eq!(TBLITE_ENV_NAME, "openthoughts-tblite");
    }

    #[test]
    fn server_config_reads_openrouter_key() {
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "tblite-test-key");
        }
        let s = tblite_server_config();
        assert_eq!(s.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(s.model_name, "anthropic/claude-sonnet-4");
        assert_eq!(s.server_type, "openai");
        assert_eq!(s.api_key, "tblite-test-key");
        assert!(!s.health_check);
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
    }

    #[test]
    fn config_init_tuple_has_one_server() {
        let (cfg, servers) = config_init();
        assert_eq!(cfg.dataset_name, "NousResearch/openthoughts-tblite");
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].base_url, "https://openrouter.ai/api/v1");
    }
}
