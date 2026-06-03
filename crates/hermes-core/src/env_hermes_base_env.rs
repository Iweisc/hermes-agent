//! Native Rust port of `environments/hermes_base_env.py`.
//!
//! `HermesAgentBaseEnv` is the abstract base environment that all hermes-agent
//! Atropos environments share. It provides:
//!
//! - Two-mode operation (OpenAI server for Phase 1, VLLM ManagedServer for Phase 2)
//! - Per-group toolset / distribution resolution
//! - Agent loop orchestration via `HermesAgentLoop`
//! - `ToolContext` creation for reward functions
//! - `ScoredDataGroup` construction from `ManagedServer` state
//!
//! Because the live Atropos runtime, the agent loop, the LLM server, and the
//! tokenizer are all asynchronous Python objects that are not yet ported, this
//! module ports the *pure, deterministic* logic that is fully reproducible in
//! Rust:
//!
//! - [`HermesAgentEnvConfig`] -- the full pydantic config with defaults.
//! - [`HermesAgentEnvConfig::build_budget_config`] -- BudgetConfig construction.
//! - [`HermesAgentBaseEnv::resolve_tools_for_group`] -- toolset resolution.
//! - [`HermesAgentBaseEnv::use_managed_server`] -- Phase 1/Phase 2 detection.
//! - [`format_trajectory_for_display`] -- wandb trajectory formatting.
//! - [`HermesAgentBaseEnv::record_tool_errors`] / [`drain_tool_error_metrics`]
//!   -- the wandb tool-error buffer.
//! - [`build_scored_item`] -- ScoredDataItem construction from managed state.
//!
//! The pieces that genuinely require the async server / tokenizer / agent loop
//! are expressed via small traits ([`ServerLike`], [`TokenizerLike`]) and helper
//! data types ([`AgentResult`], [`SequenceNode`], [`ToolError`]) so the logic can
//! be exercised and unit-tested without those runtimes.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::tool_budget_config::{
    BudgetConfig, DEFAULT_PREVIEW_SIZE_CHARS, DEFAULT_RESULT_SIZE_CHARS, DEFAULT_TURN_BUDGET_CHARS,
};

// ============================================================================
// Config -- HermesAgentEnvConfig (extends BaseEnvConfig)
// ============================================================================

/// The subset of `BaseEnvConfig` fields that `HermesAgentBaseEnv` reads.
///
/// The full Atropos `BaseEnvConfig` has many more fields; only the ones used by
/// `hermes_base_env.py` are reproduced here so the port is self-contained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseEnvConfigFields {
    /// Number of rollouts collected per group (`group_size`).
    #[serde(default = "default_group_size")]
    pub group_size: i64,
    /// Maximum token length for generation (`max_token_length`).
    #[serde(default)]
    pub max_token_length: Option<i64>,
    /// Whether thinking-mode / think-block preservation is enabled.
    #[serde(default)]
    pub thinking_mode: bool,
    /// How many rollouts per group to keep for wandb logging. `-1` means "all".
    #[serde(default = "default_num_rollouts_per_group_for_logging")]
    pub num_rollouts_per_group_for_logging: i64,
    /// How many groups of rollouts to retain in the wandb buffer.
    #[serde(default = "default_num_rollouts_to_keep")]
    pub num_rollouts_to_keep: usize,
}

fn default_group_size() -> i64 {
    2
}
fn default_num_rollouts_per_group_for_logging() -> i64 {
    1
}
fn default_num_rollouts_to_keep() -> usize {
    32
}

impl Default for BaseEnvConfigFields {
    fn default() -> Self {
        Self {
            group_size: default_group_size(),
            max_token_length: None,
            thinking_mode: false,
            num_rollouts_per_group_for_logging: default_num_rollouts_per_group_for_logging(),
            num_rollouts_to_keep: default_num_rollouts_to_keep(),
        }
    }
}

/// Configuration for hermes-agent Atropos environments.
///
/// Faithful port of `HermesAgentEnvConfig(BaseEnvConfig)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesAgentEnvConfig {
    /// Inherited `BaseEnvConfig` fields used by this environment.
    #[serde(flatten, default)]
    pub base: BaseEnvConfigFields,

    // --- Toolset configuration ---
    /// Explicit list of hermes toolsets to enable. `None` + `distribution` None
    /// means "all available toolsets".
    #[serde(default)]
    pub enabled_toolsets: Option<Vec<String>>,
    /// Toolsets to disable. Applied as a filter on top of `enabled_toolsets`/`distribution`.
    #[serde(default)]
    pub disabled_toolsets: Option<Vec<String>>,
    /// Name of a toolset distribution. Sampled once per group. Mutually exclusive
    /// with `enabled_toolsets`.
    #[serde(default)]
    pub distribution: Option<String>,

    // --- Agent loop configuration ---
    /// Maximum number of LLM calls (tool-calling iterations) per rollout.
    #[serde(default = "default_max_agent_turns")]
    pub max_agent_turns: i64,
    /// System prompt for the agent.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Sampling temperature for agent generation during rollouts.
    #[serde(default = "default_agent_temperature")]
    pub agent_temperature: f64,

    // --- Terminal backend ---
    /// Terminal backend: 'local', 'docker', 'modal', 'daytona', 'ssh', 'singularity'.
    #[serde(default = "default_terminal_backend")]
    pub terminal_backend: String,
    /// Per-command timeout in seconds for terminal tool calls.
    #[serde(default = "default_terminal_timeout")]
    pub terminal_timeout: i64,
    /// Sandbox inactivity lifetime in seconds.
    #[serde(default = "default_terminal_lifetime")]
    pub terminal_lifetime: i64,

    // --- Dataset ---
    /// HuggingFace dataset name. Optional if tasks are defined inline.
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// Dataset split to use.
    #[serde(default = "default_dataset_split")]
    pub dataset_split: String,
    /// Which field in the dataset contains the prompt.
    #[serde(default = "default_prompt_field")]
    pub prompt_field: String,

    // --- Thread pool ---
    /// Thread pool size for tool execution.
    #[serde(default = "default_tool_pool_size")]
    pub tool_pool_size: i64,

    // --- Phase 2: Tool call parsing ---
    /// Tool call parser name for Phase 2 (VLLM server type).
    #[serde(default = "default_tool_call_parser")]
    pub tool_call_parser: String,

    // --- Tool result budget ---
    /// Default per-tool threshold (chars) for persisting large results.
    #[serde(default = "default_result_size_chars")]
    pub default_result_size_chars: i64,
    /// Aggregate char budget per assistant turn.
    #[serde(default = "default_turn_budget_chars")]
    pub turn_budget_chars: i64,
    /// Size of the inline preview shown after a tool result is persisted.
    #[serde(default = "default_preview_size_chars")]
    pub preview_size_chars: i64,
    /// Per-tool threshold overrides (chars).
    #[serde(default)]
    pub tool_result_overrides: Option<HashMap<String, i64>>,

    // --- Provider-specific parameters ---
    /// Extra body parameters passed to the OpenAI client's chat.completions.create().
    #[serde(default)]
    pub extra_body: Option<Value>,
}

fn default_max_agent_turns() -> i64 {
    30
}
fn default_agent_temperature() -> f64 {
    1.0
}
fn default_terminal_backend() -> String {
    "local".to_string()
}
fn default_terminal_timeout() -> i64 {
    120
}
fn default_terminal_lifetime() -> i64 {
    3600
}
fn default_dataset_split() -> String {
    "train".to_string()
}
fn default_prompt_field() -> String {
    "prompt".to_string()
}
fn default_tool_pool_size() -> i64 {
    128
}
fn default_tool_call_parser() -> String {
    "hermes".to_string()
}
fn default_result_size_chars() -> i64 {
    DEFAULT_RESULT_SIZE_CHARS
}
fn default_turn_budget_chars() -> i64 {
    DEFAULT_TURN_BUDGET_CHARS
}
fn default_preview_size_chars() -> i64 {
    DEFAULT_PREVIEW_SIZE_CHARS
}

impl Default for HermesAgentEnvConfig {
    fn default() -> Self {
        Self {
            base: BaseEnvConfigFields::default(),
            enabled_toolsets: None,
            disabled_toolsets: None,
            distribution: None,
            max_agent_turns: default_max_agent_turns(),
            system_prompt: None,
            agent_temperature: default_agent_temperature(),
            terminal_backend: default_terminal_backend(),
            terminal_timeout: default_terminal_timeout(),
            terminal_lifetime: default_terminal_lifetime(),
            dataset_name: None,
            dataset_split: default_dataset_split(),
            prompt_field: default_prompt_field(),
            tool_pool_size: default_tool_pool_size(),
            tool_call_parser: default_tool_call_parser(),
            default_result_size_chars: default_result_size_chars(),
            turn_budget_chars: default_turn_budget_chars(),
            preview_size_chars: default_preview_size_chars(),
            tool_result_overrides: None,
            extra_body: None,
        }
    }
}

impl HermesAgentEnvConfig {
    /// Build a [`BudgetConfig`] from env config fields.
    ///
    /// Faithful port of `HermesAgentEnvConfig.build_budget_config`.
    pub fn build_budget_config(&self) -> BudgetConfig {
        BudgetConfig {
            default_result_size: self.default_result_size_chars,
            turn_budget: self.turn_budget_chars,
            preview_size: self.preview_size_chars,
            tool_overrides: self
                .tool_result_overrides
                .clone()
                .unwrap_or_default(),
        }
    }

    /// Apply the terminal-related environment variables that hermes tools read.
    ///
    /// Mirrors the `os.environ[...] = ...` block in `__init__`. The Python code
    /// only sets `TERMINAL_ENV` when `terminal_backend` is truthy (non-empty).
    ///
    /// # Safety
    /// Mutating process environment variables is not thread-safe; callers must
    /// ensure no other thread is reading/writing the environment concurrently.
    pub unsafe fn apply_terminal_env(&self) {
        if !self.terminal_backend.is_empty() {
            unsafe {
                std::env::set_var("TERMINAL_ENV", &self.terminal_backend);
            }
        }
        unsafe {
            std::env::set_var("TERMINAL_TIMEOUT", self.terminal_timeout.to_string());
            std::env::set_var(
                "TERMINAL_LIFETIME_SECONDS",
                self.terminal_lifetime.to_string(),
            );
        }
    }

    /// The startup banner printed by `__init__`.
    pub fn terminal_banner(&self) -> String {
        format!(
            "🖥️  Terminal: backend={}, timeout={}s, lifetime={}s",
            self.terminal_backend, self.terminal_timeout, self.terminal_lifetime
        )
    }
}

// ============================================================================
// Agent-loop result types (mirrors environments/agent_loop.py dataclasses)
// ============================================================================

/// Record of a tool execution error during the agent loop.
///
/// Port of the `ToolError` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolError {
    /// Which turn the error occurred on.
    pub turn: i64,
    /// Which tool was called.
    pub tool_name: String,
    /// The arguments passed (truncated upstream).
    pub arguments: String,
    /// The error message.
    pub error: String,
    /// The raw result returned to the model.
    pub tool_result: String,
}

/// A single sequence node from a ManagedServer state.
///
/// Only the fields read by `collect_trajectory` are modelled.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SequenceNode {
    /// Token ids for this node.
    pub tokens: Vec<i64>,
    /// Mask values (`-100` for masked / prompt tokens).
    pub masked_tokens: Vec<i64>,
    /// Per-token logprobs, if available (Phase 2 only).
    #[serde(default)]
    pub logprobs: Option<Vec<f64>>,
}

/// Result of running the agent loop.
///
/// Port of the `AgentResult` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentResult {
    /// Full conversation history in OpenAI message format.
    pub messages: Vec<Value>,
    /// `ManagedServer.get_state()` if available (Phase 2), `None` otherwise.
    #[serde(default)]
    pub managed_state: Option<ManagedState>,
    /// How many LLM calls were made.
    #[serde(default)]
    pub turns_used: i64,
    /// True if the model stopped calling tools naturally (vs hitting max_turns).
    #[serde(default)]
    pub finished_naturally: bool,
    /// Extracted reasoning content per turn.
    #[serde(default)]
    pub reasoning_per_turn: Vec<Option<String>>,
    /// Tool errors encountered during the loop.
    #[serde(default)]
    pub tool_errors: Vec<ToolError>,
}

/// Minimal model of a ManagedServer state dict, carrying the sequence nodes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ManagedState {
    /// Sequence nodes; the final one is the full trajectory.
    #[serde(default)]
    pub nodes: Vec<SequenceNode>,
}

// ============================================================================
// Abstraction traits for the parts that need a live runtime
// ============================================================================

/// A handle on the LLM server, mirroring the slice of `ServerManager` /
/// individual servers used by `hermes_base_env.py`.
pub trait ServerLike {
    /// Whether the underlying server list is non-empty
    /// (`self.server.servers` truthiness).
    fn has_servers(&self) -> bool;
    /// Whether the first underlying server is an `OpenAIServer`. Phase 1 mode
    /// is used when this is `true`.
    fn first_server_is_openai(&self) -> bool;
    /// Whether the server exposes a `tool_parser` attribute that we should set.
    fn supports_tool_parser(&self) -> bool;
    /// Set the tool parser name on the server (no-op if unsupported).
    fn set_tool_parser(&mut self, parser: &str);
}

/// A minimal tokenizer abstraction used for Phase 1 placeholder token
/// generation.
pub trait TokenizerLike {
    /// Encode text into token ids, optionally adding special tokens.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<i64>;
}

// ============================================================================
// Toolset resolution
// ============================================================================

/// Outcome of toolset resolution for a group.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTools {
    /// Tool JSON schemas (OpenAI `tools=` format).
    pub tools: Vec<Value>,
    /// The set of valid tool names extracted from the schemas.
    pub valid_tool_names: BTreeSet<String>,
}

/// Extract the `function.name` field of a tool schema, if present.
fn tool_schema_name(schema: &Value) -> Option<String> {
    schema
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .map(|s| s.to_string())
}

/// Build the `valid_tool_names` set from a list of tool schemas.
///
/// Mirrors `{t["function"]["name"] for t in tools}`.
pub fn valid_tool_names_from_schemas(tools: &[Value]) -> BTreeSet<String> {
    tools.iter().filter_map(tool_schema_name).collect()
}

// ============================================================================
// Trajectory formatting for wandb
// ============================================================================

/// Truncate a string to `max` characters (by char count), appending `...` if
/// it was truncated. Mirrors Python's `s[:max] + "..."` slice semantics.
fn truncate_with_ellipsis(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let prefix: String = s.chars().take(max).collect();
        format!("{prefix}...")
    } else {
        s.to_string()
    }
}

/// Get a message's `content` field as a string, defaulting to "".
fn msg_content_str(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Format a conversation's messages into a readable trajectory string for wandb
/// rollout tables.
///
/// Faithful port of `HermesAgentBaseEnv._format_trajectory_for_display`.
pub fn format_trajectory_for_display(messages: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        let content = msg_content_str(msg);

        match role {
            "system" => {
                parts.push(format!("[SYSTEM]\n{content}"));
            }
            "user" => {
                parts.push(format!("[USER]\n{content}"));
            }
            "assistant" => {
                // Reasoning (truncate at 300 chars).
                let reasoning = msg
                    .get("reasoning_content")
                    .and_then(|r| r.as_str())
                    .unwrap_or("");
                if !reasoning.is_empty() {
                    let reasoning = truncate_with_ellipsis(reasoning, 300);
                    parts.push(format!("[ASSISTANT thinking]\n{reasoning}"));
                }

                if !content.is_empty() {
                    parts.push(format!("[ASSISTANT]\n{content}"));
                }

                // Tool calls.
                if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let func = tc.get("function");
                        let name = func
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("?");
                        let args_raw = func
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}");
                        let args = truncate_with_ellipsis(args_raw, 200);
                        parts.push(format!("[TOOL CALL] {name}({args})"));
                    }
                }
            }
            "tool" => {
                let result = truncate_with_ellipsis(&content, 500);
                parts.push(format!("[TOOL RESULT] {result}"));
            }
            _ => {}
        }
    }

    parts.join("\n\n")
}

// ============================================================================
// ScoredDataItem construction
// ============================================================================

/// A built scored-data item, mirroring the dict assembled in
/// `collect_trajectory`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDataItem {
    /// Token ids.
    pub tokens: Vec<i64>,
    /// Mask values.
    pub masks: Vec<i64>,
    /// The reward score.
    pub scores: f64,
    /// Present (always `None`) only when logprobs were available (Phase 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantages: Option<Value>,
    /// Present (always `None`) only when logprobs were available (Phase 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_logprobs: Option<Value>,
    /// Always included: the full conversation for wandb display / data logging.
    pub messages: Vec<Value>,
}

/// Build a [`ScoredDataItem`] from an [`AgentResult`] and computed reward.
///
/// Faithful port of the scored-item assembly in `collect_trajectory`:
///
/// - When managed-state nodes exist (Phase 2 / DummyManagedServer), uses the
///   final node's tokens & masks; when that node has logprobs, sets
///   `advantages`/`ref_logprobs` to `None` (computed by the trainer).
/// - Otherwise (Phase 1 with no managed state), tokenizes the joined message
///   contents into placeholder tokens, masking the first token with `-100`.
///   When no tokenizer is supplied, falls back to `range(min(len//4, 128))`.
pub fn build_scored_item<T: TokenizerLike>(
    result: &AgentResult,
    reward: f64,
    tokenizer: Option<&T>,
) -> ScoredDataItem {
    let nodes: &[SequenceNode] = result
        .managed_state
        .as_ref()
        .map(|s| s.nodes.as_slice())
        .unwrap_or(&[]);

    let (tokens, masks, advantages, ref_logprobs) = if let Some(node) = nodes.last() {
        // Phase 2 (or DummyManagedServer): use actual node data.
        let has_logprobs = node
            .logprobs
            .as_ref()
            .map(|lp| !lp.is_empty())
            .unwrap_or(false);
        let (adv, refl) = if has_logprobs {
            (Some(Value::Null), Some(Value::Null))
        } else {
            (None, None)
        };
        (node.tokens.clone(), node.masked_tokens.clone(), adv, refl)
    } else {
        // Phase 1 with no managed state: create placeholder tokens.
        let full_text = result
            .messages
            .iter()
            .filter_map(|m| {
                let c = msg_content_str(m);
                if c.is_empty() {
                    None
                } else {
                    Some(c)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let tokens = match tokenizer {
            Some(tk) => tk.encode(&full_text, true),
            None => {
                let n = std::cmp::min(full_text.chars().count() / 4, 128);
                (0..n as i64).collect()
            }
        };

        // masks = [-100] + tokens[1:]
        let mut masks: Vec<i64> = Vec::with_capacity(tokens.len());
        masks.push(-100);
        if tokens.len() > 1 {
            masks.extend_from_slice(&tokens[1..]);
        }

        (tokens, masks, None, None)
    };

    ScoredDataItem {
        tokens,
        masks,
        scores: reward,
        advantages,
        ref_logprobs,
        messages: result.messages.clone(),
    }
}

/// Determine whether reward computation should be skipped for a result.
///
/// Mirrors the guard in `collect_trajectory`: skip when `turns_used == 0` or
/// every message has role `system`/`user` only.
pub fn should_skip_reward(result: &AgentResult) -> bool {
    let only_system_and_user = result.messages.iter().all(|m| {
        matches!(
            m.get("role").and_then(|r| r.as_str()),
            Some("system") | Some("user")
        )
    });
    result.turns_used == 0 || only_system_and_user
}

// ============================================================================
// HermesAgentBaseEnv -- the (non-async) state container + ported methods
// ============================================================================

/// State for the abstract hermes-agent base environment.
///
/// This carries the parts of `HermesAgentBaseEnv` that are independent of the
/// live Atropos async runtime: the config, the per-group resolved tools, the
/// tool-error buffer, and the wandb rollout buffer.
pub struct HermesAgentBaseEnv {
    /// The environment configuration.
    pub config: HermesAgentEnvConfig,
    /// Current group's resolved tools (set in `collect_trajectories`).
    pub current_group_tools: Option<ResolvedTools>,
    /// Tool error records buffered for wandb logging.
    pub tool_error_buffer: Vec<Value>,
    /// Wandb rollout buffer: list of groups, each a list of (text, score).
    pub rollouts_for_wandb: Vec<Vec<(String, f64)>>,
}

impl HermesAgentBaseEnv {
    /// The environment name (`name = "hermes-agent"`).
    pub const NAME: &'static str = "hermes-agent";

    /// Create a new base environment, applying the terminal env vars and (if
    /// supported) the tool parser on the supplied server.
    ///
    /// Mirrors `HermesAgentBaseEnv.__init__`. The thread-pool resize
    /// (`resize_tool_pool`) is a runtime concern handled elsewhere.
    ///
    /// # Safety
    /// Calls [`HermesAgentEnvConfig::apply_terminal_env`], which mutates process
    /// environment variables; see its safety note.
    pub unsafe fn new<S: ServerLike>(config: HermesAgentEnvConfig, server: &mut S) -> Self {
        unsafe {
            config.apply_terminal_env();
        }

        if server.supports_tool_parser() {
            server.set_tool_parser(&config.tool_call_parser);
        }

        Self {
            config,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        }
    }

    /// Resolve toolsets for a group.
    ///
    /// Faithful port of `_resolve_tools_for_group`. The actual tool-definition
    /// lookup is delegated through `resolve` closures so the caller can wire in
    /// `crate::mod_model_tools::get_tool_definitions` and
    /// `crate::mod_toolset_distributions::sample_toolsets_from_distribution`
    /// with a live `ToolRegistry`.
    ///
    /// - If `distribution` is set, samples toolsets via `sample`.
    /// - Otherwise uses `enabled_toolsets` (`None` => all available).
    /// - `disabled_toolsets` is applied as a filter inside `get_defs`.
    pub fn resolve_tools_for_group<FSample, FDefs>(
        &self,
        sample: FSample,
        get_defs: FDefs,
    ) -> ResolvedTools
    where
        FSample: Fn(&str) -> Vec<String>,
        FDefs: Fn(Option<&[String]>, &[String]) -> Vec<Value>,
    {
        let group_toolsets: Option<Vec<String>> = if let Some(dist) = &self.config.distribution {
            let sampled = sample(dist);
            log::info!("Sampled toolsets from '{dist}': {sampled:?}");
            Some(sampled)
        } else {
            let ts = self.config.enabled_toolsets.clone();
            if ts.is_none() {
                log::warn!(
                    "enabled_toolsets is None -- loading ALL tools including messaging. \
                     Set explicit enabled_toolsets for RL training."
                );
            }
            ts
        };

        let disabled = self
            .config
            .disabled_toolsets
            .clone()
            .unwrap_or_default();

        let tools = get_defs(group_toolsets.as_deref(), &disabled);
        let valid_names = valid_tool_names_from_schemas(&tools);
        log::info!(
            "Resolved {} tools for group: {:?}",
            valid_names.len(),
            valid_names
        );

        ResolvedTools {
            tools,
            valid_tool_names: valid_names,
        }
    }

    /// Determine whether ManagedServer (Phase 2) should be used.
    ///
    /// Faithful port of `_use_managed_server`: returns `false` if there are no
    /// servers, otherwise `true` when the first server is *not* an OpenAIServer.
    pub fn use_managed_server<S: ServerLike>(&self, server: &S) -> bool {
        if !server.has_servers() {
            return false;
        }
        !server.first_server_is_openai()
    }

    /// Record any tool errors from a result into the wandb error buffer.
    ///
    /// Faithful port of the tool-error tracking block in `collect_trajectory`:
    /// truncates `args` to 150 chars, `error`/`result` to 300 chars.
    pub fn record_tool_errors(&mut self, result: &AgentResult) {
        for err in &result.tool_errors {
            self.tool_error_buffer.push(json!({
                "turn": err.turn,
                "tool": err.tool_name,
                "args": truncate_chars(&err.arguments, 150),
                "error": truncate_chars(&err.error, 300),
                "result": truncate_chars(&err.tool_result, 300),
            }));
        }
    }

    /// Compute the wandb metrics for tool errors and drain the buffer.
    ///
    /// Faithful port of the tool-error logic in `wandb_log`. Returns the
    /// `(metrics, stdout_lines)` pair: `metrics` is the dict to merge into
    /// `wandb_metrics`, and `stdout_lines` are the lines the Python code prints
    /// for immediate visibility.
    pub fn drain_tool_error_metrics(&mut self) -> (Map<String, Value>, Vec<String>) {
        let mut metrics = Map::new();
        let mut stdout_lines = Vec::new();

        if !self.tool_error_buffer.is_empty() {
            metrics.insert(
                "train/tool_errors_count".to_string(),
                json!(self.tool_error_buffer.len()),
            );

            let mut error_summaries: Vec<String> = Vec::new();
            for err in &self.tool_error_buffer {
                let turn = err.get("turn").cloned().unwrap_or(Value::Null);
                let tool = err
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let args = err.get("args").and_then(|v| v.as_str()).unwrap_or_default();
                let error = err
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let args = truncate_chars(args, 80);
                let error = truncate_chars(error, 150);
                let turn_str = match &turn {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                error_summaries.push(format!("[turn {turn_str}] {tool}({args}) -> {error}"));
            }

            metrics.insert(
                "train/tool_error_details".to_string(),
                json!(error_summaries.join("\n")),
            );

            for summary in &error_summaries {
                stdout_lines.push(format!("  Tool Error: {summary}"));
            }

            self.tool_error_buffer.clear();
        } else {
            metrics.insert("train/tool_errors_count".to_string(), json!(0));
        }

        (metrics, stdout_lines)
    }

    /// Append a formatted group to the wandb rollout buffer.
    ///
    /// Faithful port of `add_rollouts_for_wandb`. `scored_data` is expected to be
    /// the dict-shaped value with `scores`, optionally `messages` and `tokens`.
    /// `decode_tokens` is used as the fallback when only tokens are available.
    pub fn add_rollouts_for_wandb<F>(&mut self, scored_data: &Value, decode_tokens: F)
    where
        F: Fn(&[i64]) -> String,
    {
        let group_size = self.config.base.group_size;
        let num_keep = if self.config.base.num_rollouts_per_group_for_logging == -1 {
            group_size
        } else {
            self.config.base.num_rollouts_per_group_for_logging
        };

        let scores = scored_data
            .get("scores")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let messages_list = scored_data.get("messages").and_then(|v| v.as_array());
        let tokens_list = scored_data.get("tokens").and_then(|v| v.as_array());

        let limit = std::cmp::min(num_keep.max(0) as usize, scores.len());

        let mut group: Vec<(String, f64)> = Vec::new();
        for i in 0..limit {
            let score = scores[i].as_f64().unwrap_or(0.0);

            let text = if let Some(msgs) = messages_list {
                if i < msgs.len() {
                    if let Some(arr) = msgs[i].as_array() {
                        format_trajectory_for_display(arr)
                    } else {
                        fallback_text(tokens_list, i, &decode_tokens)
                    }
                } else {
                    fallback_text(tokens_list, i, &decode_tokens)
                }
            } else {
                fallback_text(tokens_list, i, &decode_tokens)
            };

            group.push((text, score));
        }

        self.rollouts_for_wandb.push(group);
        if self.rollouts_for_wandb.len() > self.config.base.num_rollouts_to_keep {
            self.rollouts_for_wandb.remove(0);
        }
    }

    /// Build the initial OpenAI-format messages for a rollout.
    ///
    /// Mirrors the message construction at the top of `collect_trajectory`:
    /// optional system prompt followed by the formatted user prompt.
    pub fn build_initial_messages(&self, formatted_prompt: &str) -> Vec<Value> {
        let mut messages = Vec::new();
        if let Some(sp) = &self.config.system_prompt {
            messages.push(json!({"role": "system", "content": sp}));
        }
        messages.push(json!({"role": "user", "content": formatted_prompt}));
        messages
    }
}

/// Fallback display text for `add_rollouts_for_wandb`: decode tokens if present,
/// else "(no data)".
fn fallback_text<F>(tokens_list: Option<&Vec<Value>>, i: usize, decode_tokens: &F) -> String
where
    F: Fn(&[i64]) -> String,
{
    if let Some(toks) = tokens_list {
        if i < toks.len() {
            if let Some(arr) = toks[i].as_array() {
                let ids: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                return decode_tokens(&ids);
            }
        }
    }
    "(no data)".to_string()
}

/// Truncate a string to `max` characters (Python `s[:max]` semantics: no
/// ellipsis -- this matches `err['args'][:80]` style slicing).
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeServer {
        has: bool,
        openai: bool,
        supports_parser: bool,
        parser: Option<String>,
    }

    impl ServerLike for FakeServer {
        fn has_servers(&self) -> bool {
            self.has
        }
        fn first_server_is_openai(&self) -> bool {
            self.openai
        }
        fn supports_tool_parser(&self) -> bool {
            self.supports_parser
        }
        fn set_tool_parser(&mut self, parser: &str) {
            self.parser = Some(parser.to_string());
        }
    }

    struct FakeTokenizer;
    impl TokenizerLike for FakeTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<i64> {
            // Deterministic: one token per word, plus a leading special token.
            let mut out = Vec::new();
            if add_special_tokens {
                out.push(1);
            }
            for (i, _) in text.split_whitespace().enumerate() {
                out.push(100 + i as i64);
            }
            out
        }
    }

    #[test]
    fn config_defaults_match_python() {
        let c = HermesAgentEnvConfig::default();
        assert_eq!(c.max_agent_turns, 30);
        assert_eq!(c.agent_temperature, 1.0);
        assert_eq!(c.terminal_backend, "local");
        assert_eq!(c.terminal_timeout, 120);
        assert_eq!(c.terminal_lifetime, 3600);
        assert_eq!(c.dataset_split, "train");
        assert_eq!(c.prompt_field, "prompt");
        assert_eq!(c.tool_pool_size, 128);
        assert_eq!(c.tool_call_parser, "hermes");
        assert_eq!(c.default_result_size_chars, DEFAULT_RESULT_SIZE_CHARS);
        assert_eq!(c.turn_budget_chars, DEFAULT_TURN_BUDGET_CHARS);
        assert_eq!(c.preview_size_chars, DEFAULT_PREVIEW_SIZE_CHARS);
        assert!(c.enabled_toolsets.is_none());
        assert!(c.distribution.is_none());
    }

    #[test]
    fn build_budget_config_maps_fields() {
        let mut c = HermesAgentEnvConfig::default();
        c.default_result_size_chars = 10_000;
        c.turn_budget_chars = 20_000;
        c.preview_size_chars = 500;
        let mut ov = HashMap::new();
        ov.insert("terminal".to_string(), 10_000);
        c.tool_result_overrides = Some(ov);

        let b = c.build_budget_config();
        assert_eq!(b.default_result_size, 10_000);
        assert_eq!(b.turn_budget, 20_000);
        assert_eq!(b.preview_size, 500);
        assert_eq!(b.tool_overrides.get("terminal"), Some(&10_000));
    }

    #[test]
    fn build_budget_config_empty_overrides() {
        let c = HermesAgentEnvConfig::default();
        let b = c.build_budget_config();
        assert!(b.tool_overrides.is_empty());
    }

    #[test]
    fn use_managed_server_logic() {
        let env = HermesAgentBaseEnv {
            config: HermesAgentEnvConfig::default(),
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        // No servers => false.
        let s = FakeServer {
            has: false,
            openai: false,
            supports_parser: false,
            parser: None,
        };
        assert!(!env.use_managed_server(&s));
        // OpenAI server => Phase 1 => false.
        let s = FakeServer {
            has: true,
            openai: true,
            supports_parser: false,
            parser: None,
        };
        assert!(!env.use_managed_server(&s));
        // VLLM-style server => Phase 2 => true.
        let s = FakeServer {
            has: true,
            openai: false,
            supports_parser: false,
            parser: None,
        };
        assert!(env.use_managed_server(&s));
    }

    #[test]
    fn new_sets_tool_parser_when_supported() {
        let cfg = HermesAgentEnvConfig::default();
        let mut s = FakeServer {
            has: true,
            openai: false,
            supports_parser: true,
            parser: None,
        };
        let _env = unsafe { HermesAgentBaseEnv::new(cfg, &mut s) };
        assert_eq!(s.parser.as_deref(), Some("hermes"));
    }

    #[test]
    fn valid_tool_names_extraction() {
        let tools = vec![
            json!({"type": "function", "function": {"name": "terminal"}}),
            json!({"type": "function", "function": {"name": "read_file"}}),
            json!({"type": "function", "function": {}}), // no name -> skipped
        ];
        let names = valid_tool_names_from_schemas(&tools);
        assert_eq!(names.len(), 2);
        assert!(names.contains("terminal"));
        assert!(names.contains("read_file"));
    }

    #[test]
    fn resolve_tools_uses_enabled_when_no_distribution() {
        let mut cfg = HermesAgentEnvConfig::default();
        cfg.enabled_toolsets = Some(vec!["terminal".to_string()]);
        cfg.disabled_toolsets = Some(vec!["web".to_string()]);
        let env = HermesAgentBaseEnv {
            config: cfg,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let resolved = env.resolve_tools_for_group(
            |_d| panic!("sample should not be called without distribution"),
            |enabled, disabled| {
                assert_eq!(enabled, Some(["terminal".to_string()].as_slice()));
                assert_eq!(disabled, ["web".to_string()].as_slice());
                vec![json!({"function": {"name": "terminal"}})]
            },
        );
        assert!(resolved.valid_tool_names.contains("terminal"));
    }

    #[test]
    fn resolve_tools_samples_distribution() {
        let mut cfg = HermesAgentEnvConfig::default();
        cfg.distribution = Some("development".to_string());
        let env = HermesAgentBaseEnv {
            config: cfg,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let resolved = env.resolve_tools_for_group(
            |d| {
                assert_eq!(d, "development");
                vec!["terminal".to_string(), "file".to_string()]
            },
            |enabled, _disabled| {
                assert_eq!(
                    enabled,
                    Some(["terminal".to_string(), "file".to_string()].as_slice())
                );
                vec![
                    json!({"function": {"name": "terminal"}}),
                    json!({"function": {"name": "read_file"}}),
                ]
            },
        );
        assert_eq!(resolved.valid_tool_names.len(), 2);
    }

    #[test]
    fn format_trajectory_full() {
        let long_reasoning = "r".repeat(400);
        let long_args = format!("{{\"a\":\"{}\"}}", "x".repeat(300));
        let messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "hi"}),
            json!({
                "role": "assistant",
                "reasoning_content": long_reasoning,
                "content": "thinking done",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": long_args}}
                ]
            }),
            json!({"role": "tool", "tool_call_id": "x", "content": "y".repeat(600)}),
        ];
        let out = format_trajectory_for_display(&messages);
        assert!(out.contains("[SYSTEM]\nsys"));
        assert!(out.contains("[USER]\nhi"));
        assert!(out.contains("[ASSISTANT thinking]"));
        assert!(out.contains("[ASSISTANT]\nthinking done"));
        assert!(out.contains("[TOOL CALL] terminal("));
        // Reasoning truncated to 300 + ellipsis.
        assert!(out.contains(&format!("{}...", "r".repeat(300))));
        // Tool result truncated to 500 + ellipsis.
        assert!(out.contains(&format!("{}...", "y".repeat(500))));
    }

    #[test]
    fn build_scored_item_phase2_with_logprobs() {
        let result = AgentResult {
            messages: vec![json!({"role": "user", "content": "hi"})],
            managed_state: Some(ManagedState {
                nodes: vec![
                    SequenceNode {
                        tokens: vec![1, 2],
                        masked_tokens: vec![-100, 2],
                        logprobs: None,
                    },
                    SequenceNode {
                        tokens: vec![1, 2, 3],
                        masked_tokens: vec![-100, -100, 3],
                        logprobs: Some(vec![-0.1, -0.2, -0.3]),
                    },
                ],
            }),
            turns_used: 2,
            ..Default::default()
        };
        let item = build_scored_item::<FakeTokenizer>(&result, 0.75, None);
        // Uses the LAST node.
        assert_eq!(item.tokens, vec![1, 2, 3]);
        assert_eq!(item.masks, vec![-100, -100, 3]);
        assert_eq!(item.scores, 0.75);
        // logprobs present => advantages/ref_logprobs set to null.
        assert_eq!(item.advantages, Some(Value::Null));
        assert_eq!(item.ref_logprobs, Some(Value::Null));
    }

    #[test]
    fn build_scored_item_phase2_no_logprobs() {
        let result = AgentResult {
            messages: vec![],
            managed_state: Some(ManagedState {
                nodes: vec![SequenceNode {
                    tokens: vec![5, 6],
                    masked_tokens: vec![-100, 6],
                    logprobs: None,
                }],
            }),
            turns_used: 1,
            ..Default::default()
        };
        let item = build_scored_item::<FakeTokenizer>(&result, 1.0, None);
        assert_eq!(item.tokens, vec![5, 6]);
        assert!(item.advantages.is_none());
        assert!(item.ref_logprobs.is_none());
    }

    #[test]
    fn build_scored_item_phase1_with_tokenizer() {
        let result = AgentResult {
            messages: vec![
                json!({"role": "user", "content": "hello world"}),
                json!({"role": "assistant", "content": "hi there"}),
                json!({"role": "assistant", "content": Value::Null}), // dropped
            ],
            managed_state: None,
            turns_used: 2,
            ..Default::default()
        };
        let tk = FakeTokenizer;
        let item = build_scored_item(&result, 0.5, Some(&tk));
        // "hello world\nhi there" => special(1) + 4 words.
        assert_eq!(item.tokens, vec![1, 100, 101, 102, 103]);
        // masks = [-100] + tokens[1:].
        assert_eq!(item.masks, vec![-100, 100, 101, 102, 103]);
        assert_eq!(item.scores, 0.5);
    }

    #[test]
    fn build_scored_item_phase1_no_tokenizer() {
        // 40-char content => 40/4 = 10 placeholder tokens.
        let content = "x".repeat(40);
        let result = AgentResult {
            messages: vec![json!({"role": "user", "content": content})],
            managed_state: None,
            turns_used: 1,
            ..Default::default()
        };
        let item = build_scored_item::<FakeTokenizer>(&result, 0.0, None);
        assert_eq!(item.tokens, (0..10).collect::<Vec<_>>());
        assert_eq!(item.masks[0], -100);
    }

    #[test]
    fn should_skip_reward_logic() {
        // turns_used == 0 => skip.
        let r = AgentResult {
            messages: vec![json!({"role": "assistant", "content": "x"})],
            turns_used: 0,
            ..Default::default()
        };
        assert!(should_skip_reward(&r));

        // only system+user => skip.
        let r = AgentResult {
            messages: vec![
                json!({"role": "system", "content": "s"}),
                json!({"role": "user", "content": "u"}),
            ],
            turns_used: 1,
            ..Default::default()
        };
        assert!(should_skip_reward(&r));

        // has assistant turn => don't skip.
        let r = AgentResult {
            messages: vec![
                json!({"role": "user", "content": "u"}),
                json!({"role": "assistant", "content": "a"}),
            ],
            turns_used: 1,
            ..Default::default()
        };
        assert!(!should_skip_reward(&r));
    }

    #[test]
    fn tool_error_buffer_and_drain() {
        let mut env = HermesAgentBaseEnv {
            config: HermesAgentEnvConfig::default(),
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let result = AgentResult {
            tool_errors: vec![ToolError {
                turn: 3,
                tool_name: "terminal".to_string(),
                arguments: "a".repeat(200),
                error: "boom".to_string(),
                tool_result: "res".to_string(),
            }],
            ..Default::default()
        };
        env.record_tool_errors(&result);
        assert_eq!(env.tool_error_buffer.len(), 1);
        // args truncated to 150.
        let stored = &env.tool_error_buffer[0];
        assert_eq!(stored.get("args").unwrap().as_str().unwrap().len(), 150);

        let (metrics, lines) = env.drain_tool_error_metrics();
        assert_eq!(metrics.get("train/tool_errors_count").unwrap(), &json!(1));
        let details = metrics
            .get("train/tool_error_details")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(details.contains("[turn 3] terminal("));
        assert!(details.contains("-> boom"));
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("  Tool Error:"));
        // Buffer drained.
        assert!(env.tool_error_buffer.is_empty());

        // Empty drain => count 0, no detail key.
        let (metrics, lines) = env.drain_tool_error_metrics();
        assert_eq!(metrics.get("train/tool_errors_count").unwrap(), &json!(0));
        assert!(!metrics.contains_key("train/tool_error_details"));
        assert!(lines.is_empty());
    }

    #[test]
    fn add_rollouts_uses_messages_then_truncates_buffer() {
        let mut cfg = HermesAgentEnvConfig::default();
        cfg.base.num_rollouts_per_group_for_logging = -1; // => group_size
        cfg.base.group_size = 2;
        cfg.base.num_rollouts_to_keep = 1;
        let mut env = HermesAgentBaseEnv {
            config: cfg,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };

        let scored = json!({
            "scores": [1.0, 0.0],
            "messages": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}]
            ]
        });
        env.add_rollouts_for_wandb(&scored, |_t| "decoded".to_string());
        env.add_rollouts_for_wandb(&scored, |_t| "decoded".to_string());

        // num_rollouts_to_keep = 1 => only the latest group retained.
        assert_eq!(env.rollouts_for_wandb.len(), 1);
        let group = &env.rollouts_for_wandb[0];
        assert_eq!(group.len(), 2);
        assert!(group[0].0.contains("[USER]\nfirst"));
        assert_eq!(group[0].1, 1.0);
    }

    #[test]
    fn add_rollouts_falls_back_to_tokens() {
        let mut cfg = HermesAgentEnvConfig::default();
        cfg.base.num_rollouts_per_group_for_logging = 1;
        let mut env = HermesAgentBaseEnv {
            config: cfg,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let scored = json!({
            "scores": [0.3],
            "tokens": [[1, 2, 3]]
        });
        env.add_rollouts_for_wandb(&scored, |t| format!("decoded:{}", t.len()));
        assert_eq!(env.rollouts_for_wandb[0][0].0, "decoded:3");
    }

    #[test]
    fn build_initial_messages_with_and_without_system() {
        let mut cfg = HermesAgentEnvConfig::default();
        let env_no_sys = HermesAgentBaseEnv {
            config: cfg.clone(),
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let msgs = env_no_sys.build_initial_messages("hello");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");

        cfg.system_prompt = Some("be good".to_string());
        let env_sys = HermesAgentBaseEnv {
            config: cfg,
            current_group_tools: None,
            tool_error_buffer: Vec::new(),
            rollouts_for_wandb: Vec::new(),
        };
        let msgs = env_sys.build_initial_messages("hello");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be good");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn config_roundtrips_through_yaml() {
        let yaml = r#"
enabled_toolsets: ["terminal", "file"]
distribution: null
max_agent_turns: 10
terminal_backend: "modal"
group_size: 4
"#;
        let cfg: HermesAgentEnvConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.max_agent_turns, 10);
        assert_eq!(cfg.terminal_backend, "modal");
        assert_eq!(cfg.base.group_size, 4);
        assert_eq!(
            cfg.enabled_toolsets,
            Some(vec!["terminal".to_string(), "file".to_string()])
        );
    }
}
