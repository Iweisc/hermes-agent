//! Native Rust port of `batch_runner.py`.
//!
//! Provides parallel batch processing capabilities for running the agent across
//! multiple prompts from a dataset. This port faithfully reproduces the pure
//! data-processing logic of the Python module:
//!
//! - Dataset loading and batching
//! - Checkpointing for fault tolerance and resumption
//! - Trajectory statistics extraction (tool usage + reasoning coverage)
//! - Tool-stats / error-count normalization to a consistent schema
//! - Content-based resume (scanning existing batch files)
//! - Combining all batch files into a single `trajectories.jsonl`, filtering
//!   corrupted entries
//!
//! The Python module also drives the agent loop (`AIAgent.run_conversation`)
//! through a `multiprocessing.Pool`. That side-effecting orchestration depends
//! on the (very large) `run_agent` surface and on `multiprocessing`. Here the
//! agent execution is abstracted behind the [`PromptProcessor`] trait so the
//! pure batch-orchestration logic can be exercised and tested independently,
//! and so the caller can plug in the real agent runner once it is ported.
//!
//! Cross-references:
//! - [`crate::mod_utils::atomic_json_write`] for checkpoint persistence.
//! - Toolset distribution sampling (`toolset_distributions.py`) is taken as a
//!   parameter via [`DistributionSampler`] because it is not yet ported.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Default per-tool stats for tools that were never used.
///
/// Mirrors Python's `DEFAULT_TOOL_STATS = {'count': 0, 'success': 0, 'failure': 0}`.
pub fn default_tool_stats() -> ToolStat {
    ToolStat::default()
}

/// Per-tool usage statistics (`{count, success, failure}` plus optional rates).
///
/// The success/failure-rate fields are only populated by [`finalize_success_rates`]
/// for the aggregated total-stats table, matching the Python behaviour where
/// `success_rate`/`failure_rate` are added at the end of `run()`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolStat {
    pub count: i64,
    pub success: i64,
    pub failure: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_rate: Option<f64>,
}

impl ToolStat {
    /// A zeroed stat with no rates set (the `count/success/failure` triple).
    pub fn zeroed() -> Self {
        ToolStat {
            count: 0,
            success: 0,
            failure: 0,
            success_rate: None,
            failure_rate: None,
        }
    }
}

/// Reasoning coverage counters across assistant turns.
///
/// Mirrors the dict returned by `_extract_reasoning_stats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningStats {
    pub total_assistant_turns: i64,
    pub turns_with_reasoning: i64,
    pub turns_without_reasoning: i64,
    pub has_any_reasoning: bool,
}

/// Trait providing the set of all known/valid tool names.
///
/// In Python this is `ALL_POSSIBLE_TOOLS = set(TOOL_TO_TOOLSET_MAP.keys())`,
/// auto-derived from `model_tools.TOOL_TO_TOOLSET_MAP`. The Rust port takes it
/// as an injected dependency so callers can wire in the real map once
/// `model_tools` is ported, without this module hard-depending on it.
pub trait ToolCatalog {
    /// Every tool name that is considered valid for schema normalization and
    /// corrupted-entry filtering.
    fn all_possible_tools(&self) -> HashSet<String>;
}

/// Simple [`ToolCatalog`] backed by an explicit set of tool names.
#[derive(Debug, Clone, Default)]
pub struct StaticToolCatalog {
    pub tools: HashSet<String>,
}

impl StaticToolCatalog {
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        StaticToolCatalog {
            tools: tools.into_iter().map(Into::into).collect(),
        }
    }
}

impl ToolCatalog for StaticToolCatalog {
    fn all_possible_tools(&self) -> HashSet<String> {
        self.tools.clone()
    }
}

// ---------------------------------------------------------------------------
// Stats extraction
// ---------------------------------------------------------------------------

/// Extract tool usage statistics from a message history.
///
/// Faithful port of `_extract_tool_stats`. Walks the message list, counting
/// tool calls issued by `assistant` messages and matching them to subsequent
/// `tool` responses to determine success/failure.
///
/// Success determination mirrors the Python heuristics exactly:
/// - If the response content parses as a JSON object:
///   - a non-null `error` field marks failure;
///   - a nested `content` object with a non-null `error` marks failure;
///   - `success == false` marks failure.
/// - If the content is not JSON: empty content is a failure, and content whose
///   trimmed lowercase form starts with `error:` is a failure.
pub fn extract_tool_stats(messages: &[Value]) -> BTreeMap<String, ToolStat> {
    let mut tool_stats: BTreeMap<String, ToolStat> = BTreeMap::new();
    // Map tool_call_id -> tool name.
    let mut tool_calls_map: HashMap<String, String> = HashMap::new();

    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");

        if role == "assistant" {
            // Track tool calls from assistant messages (only if present + truthy).
            if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
                if tool_calls.is_empty() {
                    continue;
                }
                for tool_call in tool_calls {
                    // `if not tool_call or not isinstance(tool_call, dict): continue`
                    let obj = match tool_call {
                        Value::Object(o) if !o.is_empty() => o,
                        _ => continue,
                    };
                    let tool_name = match obj
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                    {
                        Some(n) => n.to_string(),
                        // Python would KeyError; we skip malformed entries.
                        None => continue,
                    };
                    let tool_call_id = match obj.get("id").and_then(Value::as_str) {
                        Some(i) => i.to_string(),
                        None => continue,
                    };

                    let entry = tool_stats.entry(tool_name.clone()).or_insert_with(ToolStat::zeroed);
                    entry.count += 1;
                    tool_calls_map.insert(tool_call_id, tool_name);
                }
            }
        } else if role == "tool" {
            let tool_call_id = msg
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let content_val = msg.get("content").cloned().unwrap_or(Value::String(String::new()));

            let is_success = tool_response_is_success(&content_val);

            if let Some(tool_name) = tool_calls_map.get(&tool_call_id) {
                if let Some(stat) = tool_stats.get_mut(tool_name) {
                    if is_success {
                        stat.success += 1;
                    } else {
                        stat.failure += 1;
                    }
                }
            }
        }
    }

    tool_stats
}

/// Determine whether a `tool` message's content represents a successful call.
///
/// Encapsulates the success heuristics of `_extract_tool_stats`.
fn tool_response_is_success(content: &Value) -> bool {
    // Replicate Python: `content_json = json.loads(content) if isinstance(content, str) else content`.
    // The whole try/except guards JSON parsing of string content.
    let parsed: Option<Value> = match content {
        Value::String(s) => serde_json::from_str::<Value>(s).ok().map(Some).unwrap_or(None),
        other => Some(other.clone()),
    };

    match content {
        Value::String(s) => {
            match parsed {
                Some(content_json) => json_object_is_success(&content_json),
                None => {
                    // Not JSON: empty => failure; "error:" prefix => failure.
                    if s.is_empty() {
                        false
                    } else {
                        !s.trim().to_lowercase().starts_with("error:")
                    }
                }
            }
        }
        // Non-string content: treated directly as the parsed value in Python.
        other => json_object_is_success(other),
    }
}

/// Apply the dict-shape success checks to a parsed JSON value.
fn json_object_is_success(content_json: &Value) -> bool {
    let obj = match content_json {
        Value::Object(o) => o,
        _ => return true,
    };

    let mut is_success = true;

    // error field present AND non-null => failure
    if let Some(err) = obj.get("error") {
        if !err.is_null() {
            is_success = false;
        }
    }

    // nested content object with non-null error => failure
    if let Some(Value::Object(inner)) = obj.get("content") {
        if let Some(inner_err) = inner.get("error") {
            if !inner_err.is_null() {
                is_success = false;
            }
        }
    }

    // success == false => failure
    if let Some(Value::Bool(false)) = obj.get("success") {
        is_success = false;
    }

    is_success
}

/// Count assistant turns that contain reasoning vs none.
///
/// Faithful port of `_extract_reasoning_stats`. A turn counts as reasoning if it
/// contains `<REASONING_SCRATCHPAD>` in its content, or if a non-empty
/// `reasoning` field (native thinking tokens) is present.
pub fn extract_reasoning_stats(messages: &[Value]) -> ReasoningStats {
    let mut total: i64 = 0;
    let mut with_reasoning: i64 = 0;

    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        total += 1;

        // content = msg.get("content", "") or ""
        let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        let has_scratchpad = content.contains("<REASONING_SCRATCHPAD>");

        // has_native_reasoning = bool(msg.get("reasoning", "").strip()) if msg.get("reasoning") else False
        let has_native_reasoning = match msg.get("reasoning") {
            Some(Value::String(s)) => !s.trim().is_empty(),
            Some(v) if !v.is_null() => {
                // Truthy non-string value: Python's `.strip()` would error, but a
                // truthy `reasoning` that isn't a string won't reach `.strip()`
                // cleanly; the guard `if msg.get("reasoning")` is truthy, then
                // `msg.get("reasoning", "").strip()` runs. For non-strings this
                // would raise in Python; in practice reasoning is a string. We
                // treat truthy non-empty values as having reasoning.
                !matches!(v, Value::String(_)) // already handled string above
            }
            _ => false,
        };

        if has_scratchpad || has_native_reasoning {
            with_reasoning += 1;
        }
    }

    ReasoningStats {
        total_assistant_turns: total,
        turns_with_reasoning: with_reasoning,
        turns_without_reasoning: total - with_reasoning,
        has_any_reasoning: with_reasoning > 0,
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize tool_stats to include all possible tools with consistent schema.
///
/// Faithful port of `_normalize_tool_stats`. Every tool in the catalog is
/// present (zeroed if unused), and any unexpected tools from the raw stats are
/// also preserved.
pub fn normalize_tool_stats(
    tool_stats: &BTreeMap<String, ToolStat>,
    all_possible_tools: &HashSet<String>,
) -> BTreeMap<String, ToolStat> {
    let mut normalized: BTreeMap<String, ToolStat> = BTreeMap::new();

    for tool in all_possible_tools {
        if let Some(stats) = tool_stats.get(tool) {
            normalized.insert(tool.clone(), stats.clone());
        } else {
            normalized.insert(tool.clone(), default_tool_stats());
        }
    }

    for (tool, stats) in tool_stats {
        normalized.entry(tool.clone()).or_insert_with(|| stats.clone());
    }

    normalized
}

/// Normalize tool_error_counts to include all possible tools.
///
/// Faithful port of `_normalize_tool_error_counts`.
pub fn normalize_tool_error_counts(
    tool_error_counts: &BTreeMap<String, i64>,
    all_possible_tools: &HashSet<String>,
) -> BTreeMap<String, i64> {
    let mut normalized: BTreeMap<String, i64> = BTreeMap::new();

    for tool in all_possible_tools {
        normalized.insert(tool.clone(), *tool_error_counts.get(tool).unwrap_or(&0));
    }

    for (tool, count) in tool_error_counts {
        normalized.entry(tool.clone()).or_insert(*count);
    }

    normalized
}

// ---------------------------------------------------------------------------
// Prompt processing abstraction
// ---------------------------------------------------------------------------

/// Worker configuration shared with the prompt processor.
///
/// Mirrors the `config` dict built in `BatchRunner.run()` and consumed by
/// `_process_single_prompt`. Optional fields use `None` to mirror Python's
/// `config.get(...)` returning `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub distribution: String,
    pub model: String,
    pub max_iterations: i64,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub verbose: bool,
    pub ephemeral_system_prompt: Option<String>,
    pub log_prefix_chars: i64,
    pub providers_allowed: Option<Vec<String>>,
    pub providers_ignored: Option<Vec<String>>,
    pub providers_order: Option<Vec<String>>,
    pub provider_sort: Option<String>,
    pub max_tokens: Option<i64>,
    pub reasoning_config: Option<Value>,
    pub prefill_messages: Option<Vec<Value>>,
}

/// Result of processing a single prompt.
///
/// Mirrors the dict returned by `_process_single_prompt`. On failure,
/// `trajectory` is `None` and stats are empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptResult {
    pub success: bool,
    pub prompt_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Trajectory in from/value conversation format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<Value>,
    pub tool_stats: BTreeMap<String, ToolStat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_stats: Option<ReasoningStats>,
    pub completed: bool,
    pub partial: bool,
    pub api_calls: i64,
    pub toolsets_used: Vec<String>,
    pub metadata: Value,
}

/// Abstraction over running the agent on a single prompt.
///
/// In Python this is `_process_single_prompt`, which constructs an `AIAgent`,
/// runs the conversation, and extracts stats. Because `AIAgent` is a large
/// not-yet-ported surface, the orchestration here delegates to this trait so
/// the batch logic remains testable and decoupled.
pub trait PromptProcessor {
    fn process(
        &self,
        prompt_index: usize,
        prompt_data: &Value,
        batch_num: usize,
        config: &WorkerConfig,
    ) -> PromptResult;
}

/// Abstraction over toolset-distribution sampling (`toolset_distributions.py`).
///
/// `sample_toolsets_from_distribution` is not yet ported; callers inject it.
pub trait DistributionSampler {
    /// Returns the sampled toolset names for the named distribution.
    fn sample(&self, distribution: &str) -> Vec<String>;
    /// Whether the named distribution exists (`validate_distribution`).
    fn validate(&self, distribution: &str) -> bool;
    /// Available distribution names (`list_distributions().keys()`).
    fn list(&self) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// Batch worker
// ---------------------------------------------------------------------------

/// Aggregated result of processing one batch.
///
/// Mirrors the dict returned by `_process_batch_worker`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch_num: usize,
    pub processed: usize,
    pub skipped: usize,
    pub tool_stats: BTreeMap<String, ToolStat>,
    pub reasoning_stats: ReasoningStats,
    pub discarded_no_reasoning: i64,
    pub completed_prompts: Vec<usize>,
}

/// Process a single batch of prompts.
///
/// Faithful port of `_process_batch_worker`. Filters out already-completed
/// prompts (by index), processes each remaining prompt via `processor`, appends
/// successful (and reasoning-bearing) trajectories to `batch_<n>.jsonl`, and
/// aggregates tool/reasoning stats.
///
/// `batch_data` is the list of `(index, entry)` pairs for the batch.
#[allow(clippy::too_many_arguments)]
pub fn process_batch_worker<P: PromptProcessor>(
    batch_num: usize,
    batch_data: &[(usize, Value)],
    output_dir: &Path,
    completed_prompts_set: &HashSet<usize>,
    config: &WorkerConfig,
    processor: &P,
    all_possible_tools: &HashSet<String>,
) -> std::io::Result<BatchResult> {
    eprintln!("\n🔄 Batch {}: Starting ({} prompts)", batch_num, batch_data.len());

    let batch_output_file = output_dir.join(format!("batch_{}.jsonl", batch_num));

    // Filter out already-completed prompts.
    let prompts_to_process: Vec<&(usize, Value)> = batch_data
        .iter()
        .filter(|(idx, _)| !completed_prompts_set.contains(idx))
        .collect();

    if prompts_to_process.is_empty() {
        eprintln!("✅ Batch {}: Already completed (skipping)", batch_num);
        return Ok(BatchResult {
            batch_num,
            processed: 0,
            skipped: batch_data.len(),
            ..Default::default()
        });
    }

    eprintln!(
        "   Processing {} prompts (skipping {} already completed)",
        prompts_to_process.len(),
        batch_data.len() - prompts_to_process.len()
    );

    let mut batch_tool_stats: BTreeMap<String, ToolStat> = BTreeMap::new();
    let mut batch_reasoning_stats = ReasoningStats::default();
    let mut completed_in_batch: Vec<usize> = Vec::new();
    let mut discarded_no_reasoning: i64 = 0;

    for (prompt_index, prompt_data) in &prompts_to_process {
        let result = processor.process(*prompt_index, prompt_data, batch_num, config);

        // Save trajectory if successful.
        if result.success && result.trajectory.is_some() {
            // Discard samples with zero reasoning across all turns.
            let has_any_reasoning = result
                .reasoning_stats
                .as_ref()
                .map(|r| r.has_any_reasoning)
                // Python: reasoning.get("has_any_reasoning", True)
                .unwrap_or(true);

            if !has_any_reasoning {
                eprintln!(
                    "   🚫 Prompt {} discarded (no reasoning in any turn)",
                    prompt_index
                );
                discarded_no_reasoning += 1;
                completed_in_batch.push(*prompt_index);
                continue;
            }

            let raw_tool_stats = &result.tool_stats;
            let tool_stats = normalize_tool_stats(raw_tool_stats, all_possible_tools);

            let raw_error_counts: BTreeMap<String, i64> = raw_tool_stats
                .iter()
                .map(|(name, stats)| (name.clone(), stats.failure))
                .collect();
            let tool_error_counts = normalize_tool_error_counts(&raw_error_counts, all_possible_tools);

            let trajectory_entry = json!({
                "prompt_index": prompt_index,
                "conversations": result.trajectory,
                "metadata": result.metadata,
                "completed": result.completed,
                "partial": result.partial,
                "api_calls": result.api_calls,
                "toolsets_used": result.toolsets_used,
                "tool_stats": tool_stats,
                "tool_error_counts": tool_error_counts,
            });

            // Append to batch output file (ensure_ascii=False => write raw UTF-8).
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&batch_output_file)?;
            f.write_all(serde_json::to_string(&trajectory_entry)?.as_bytes())?;
            f.write_all(b"\n")?;
        }

        // Aggregate tool statistics.
        for (tool_name, stats) in &result.tool_stats {
            let entry = batch_tool_stats.entry(tool_name.clone()).or_insert_with(ToolStat::zeroed);
            entry.count += stats.count;
            entry.success += stats.success;
            entry.failure += stats.failure;
        }

        // Aggregate reasoning stats (only the three counter fields, per Python).
        if let Some(r) = &result.reasoning_stats {
            batch_reasoning_stats.total_assistant_turns += r.total_assistant_turns;
            batch_reasoning_stats.turns_with_reasoning += r.turns_with_reasoning;
            batch_reasoning_stats.turns_without_reasoning += r.turns_without_reasoning;
        }

        // Only mark as completed if successfully saved.
        if result.success && result.trajectory.is_some() {
            completed_in_batch.push(*prompt_index);
            let status = if result.partial { "⚠️  partial" } else { "✅" };
            eprintln!("   {} Prompt {} completed", status, prompt_index);
        } else {
            eprintln!("   ❌ Prompt {} failed (will retry on resume)", prompt_index);
        }
    }

    eprintln!(
        "✅ Batch {}: Completed ({} prompts processed)",
        batch_num,
        prompts_to_process.len()
    );

    Ok(BatchResult {
        batch_num,
        processed: prompts_to_process.len(),
        skipped: batch_data.len() - prompts_to_process.len(),
        tool_stats: batch_tool_stats,
        reasoning_stats: batch_reasoning_stats,
        discarded_no_reasoning,
        completed_prompts: completed_in_batch,
    })
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Checkpoint data persisted between runs.
///
/// Mirrors the checkpoint dict in the Python module. `completed_prompts` is the
/// sorted list of indices; `batch_stats` maps stringified batch numbers to
/// `{processed, skipped, discarded_no_reasoning}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub run_name: String,
    pub completed_prompts: Vec<i64>,
    pub batch_stats: Map<String, Value>,
    pub last_updated: Option<String>,
}

impl CheckpointData {
    /// Fresh empty checkpoint for the given run.
    pub fn empty(run_name: &str) -> Self {
        CheckpointData {
            run_name: run_name.to_string(),
            completed_prompts: Vec::new(),
            batch_stats: Map::new(),
            last_updated: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchRunner
// ---------------------------------------------------------------------------

/// Errors raised by [`BatchRunner`] construction / execution.
#[derive(Debug)]
pub enum BatchError {
    UnknownDistribution { name: String, available: Vec<String> },
    DatasetNotFound(PathBuf),
    NoValidEntries(PathBuf),
    Io(std::io::Error),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::UnknownDistribution { name, available } => write!(
                f,
                "Unknown distribution: {}. Available: {:?}",
                name, available
            ),
            BatchError::DatasetNotFound(p) => write!(f, "Dataset file not found: {}", p.display()),
            BatchError::NoValidEntries(p) => {
                write!(f, "No valid entries found in dataset file: {}", p.display())
            }
            BatchError::Io(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<std::io::Error> for BatchError {
    fn from(e: std::io::Error) -> Self {
        BatchError::Io(e)
    }
}

/// Configuration for constructing a [`BatchRunner`].
///
/// Mirrors the keyword arguments of `BatchRunner.__init__`.
#[derive(Debug, Clone)]
pub struct BatchRunnerConfig {
    pub dataset_file: PathBuf,
    pub batch_size: usize,
    pub run_name: String,
    pub distribution: String,
    pub max_iterations: i64,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: String,
    pub num_workers: usize,
    pub verbose: bool,
    pub ephemeral_system_prompt: Option<String>,
    pub log_prefix_chars: i64,
    pub providers_allowed: Option<Vec<String>>,
    pub providers_ignored: Option<Vec<String>>,
    pub providers_order: Option<Vec<String>>,
    pub provider_sort: Option<String>,
    pub max_tokens: Option<i64>,
    pub reasoning_config: Option<Value>,
    pub prefill_messages: Option<Vec<Value>>,
    pub max_samples: Option<usize>,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        BatchRunnerConfig {
            dataset_file: PathBuf::new(),
            batch_size: 1,
            run_name: String::new(),
            distribution: "default".to_string(),
            max_iterations: 10,
            base_url: None,
            api_key: None,
            model: "claude-opus-4-20250514".to_string(),
            num_workers: 4,
            verbose: false,
            ephemeral_system_prompt: None,
            log_prefix_chars: 100,
            providers_allowed: None,
            providers_ignored: None,
            providers_order: None,
            provider_sort: None,
            max_tokens: None,
            reasoning_config: None,
            prefill_messages: None,
            max_samples: None,
        }
    }
}

/// Manages batch processing of agent prompts with checkpointing and statistics.
///
/// Port of the Python `BatchRunner` class. Construction loads + truncates the
/// dataset and creates batches; [`BatchRunner::run`] drives the pipeline.
pub struct BatchRunner {
    pub cfg: BatchRunnerConfig,
    pub output_dir: PathBuf,
    pub checkpoint_file: PathBuf,
    pub stats_file: PathBuf,
    pub dataset: Vec<Value>,
    /// Each batch is a list of `(original_index, entry)` pairs.
    pub batches: Vec<Vec<(usize, Value)>>,
}

impl BatchRunner {
    /// Construct a runner. Validates the distribution, sets up the output dir,
    /// loads the dataset (optionally truncated to `max_samples`), and batches it.
    ///
    /// Faithful port of `BatchRunner.__init__`.
    pub fn new<D: DistributionSampler>(
        cfg: BatchRunnerConfig,
        sampler: &D,
    ) -> Result<Self, BatchError> {
        if !sampler.validate(&cfg.distribution) {
            return Err(BatchError::UnknownDistribution {
                name: cfg.distribution.clone(),
                available: sampler.list(),
            });
        }

        let output_dir = Path::new("data").join(&cfg.run_name);
        std::fs::create_dir_all(&output_dir)?;

        let checkpoint_file = output_dir.join("checkpoint.json");
        let stats_file = output_dir.join("statistics.json");

        let mut dataset = load_dataset(&cfg.dataset_file)?;
        if let Some(max) = cfg.max_samples {
            if max < dataset.len() {
                let full_count = dataset.len();
                dataset.truncate(max);
                eprintln!(
                    "✂️  Truncated dataset from {} to {} samples (--max_samples)",
                    full_count, max
                );
            }
        }

        let batches = create_batches(&dataset, cfg.batch_size);

        eprintln!("📊 Batch Runner Initialized");
        eprintln!("   Dataset: {} ({} prompts)", cfg.dataset_file.display(), dataset.len());
        eprintln!("   Batch size: {}", cfg.batch_size);
        eprintln!("   Total batches: {}", batches.len());
        eprintln!("   Run name: {}", cfg.run_name);
        eprintln!("   Distribution: {}", cfg.distribution);
        eprintln!("   Output directory: {}", output_dir.display());
        eprintln!("   Workers: {}", cfg.num_workers);
        if let Some(prompt) = &cfg.ephemeral_system_prompt {
            let preview = if prompt.chars().count() > 60 {
                let truncated: String = prompt.chars().take(60).collect();
                format!("{}...", truncated)
            } else {
                prompt.clone()
            };
            eprintln!("   🔒 Ephemeral system prompt: '{}'", preview);
        }

        Ok(BatchRunner {
            cfg,
            output_dir,
            checkpoint_file,
            stats_file,
            dataset,
            batches,
        })
    }

    /// Load checkpoint data if it exists; otherwise return an empty checkpoint.
    ///
    /// Faithful port of `_load_checkpoint` (including the warn-and-default
    /// behaviour on read/parse failure).
    pub fn load_checkpoint(&self) -> CheckpointData {
        if !self.checkpoint_file.exists() {
            return CheckpointData::empty(&self.cfg.run_name);
        }
        match std::fs::read_to_string(&self.checkpoint_file)
            .ok()
            .and_then(|s| serde_json::from_str::<CheckpointData>(&s).ok())
        {
            Some(c) => c,
            None => {
                eprintln!("⚠️  Warning: Failed to load checkpoint");
                CheckpointData::empty(&self.cfg.run_name)
            }
        }
    }

    /// Save checkpoint data atomically (stamping `last_updated` first).
    ///
    /// Faithful port of `_save_checkpoint`. Uses
    /// [`crate::mod_utils::atomic_json_write`].
    pub fn save_checkpoint(&self, checkpoint_data: &mut CheckpointData) -> std::io::Result<()> {
        checkpoint_data.last_updated = Some(now_iso());
        let value = serde_json::to_value(&*checkpoint_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        crate::mod_utils::atomic_json_write(&self.checkpoint_file, &value, 2)
    }

    /// Scan all batch files and extract completed prompts by their content.
    ///
    /// Faithful port of `_scan_completed_prompts_by_content`. Returns the set of
    /// human/user prompt texts that have been successfully processed.
    pub fn scan_completed_prompts_by_content(&self) -> HashSet<String> {
        let mut completed_prompts: HashSet<String> = HashSet::new();
        let batch_files = sorted_batch_files(&self.output_dir);

        if batch_files.is_empty() {
            return completed_prompts;
        }

        eprintln!(
            "📂 Scanning {} batch files for completed prompts...",
            batch_files.len()
        );

        for batch_file in &batch_files {
            let file = match File::open(batch_file) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!(
                        "  ⚠️  Warning: Error reading {}: {}",
                        batch_file.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                        e
                    );
                    continue;
                }
            };
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let entry: Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Skip failed entries.
                if entry.get("failed").and_then(Value::as_bool).unwrap_or(false) {
                    continue;
                }

                if let Some(Value::Array(conversations)) = entry.get("conversations") {
                    for msg in conversations {
                        if msg.get("from").and_then(Value::as_str) == Some("human") {
                            let prompt_text = msg
                                .get("value")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .trim()
                                .to_string();
                            if !prompt_text.is_empty() {
                                completed_prompts.insert(prompt_text);
                            }
                            break; // only the first human message
                        }
                    }
                }
            }
        }

        completed_prompts
    }

    /// Filter the dataset to exclude prompts that have already been completed.
    ///
    /// Faithful port of `_filter_dataset_by_completed`. Returns
    /// `(filtered_entries, skipped_indices)` where `filtered_entries` keeps the
    /// original dataset index for tracking.
    pub fn filter_dataset_by_completed(
        &self,
        completed_prompts: &HashSet<String>,
    ) -> (Vec<(usize, Value)>, Vec<usize>) {
        let mut filtered: Vec<(usize, Value)> = Vec::new();
        let mut skipped: Vec<usize> = Vec::new();

        for (idx, entry) in self.dataset.iter().enumerate() {
            let mut prompt_text = entry
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if prompt_text.is_empty() {
                if let Some(Value::Array(conversations)) = entry.get("conversations") {
                    for msg in conversations {
                        let role = msg
                            .get("role")
                            .and_then(Value::as_str)
                            .or_else(|| msg.get("from").and_then(Value::as_str));
                        if matches!(role, Some("user") | Some("human")) {
                            let content = msg
                                .get("content")
                                .and_then(Value::as_str)
                                .or_else(|| msg.get("value").and_then(Value::as_str))
                                .unwrap_or("");
                            prompt_text = content.trim().to_string();
                            break;
                        }
                    }
                }
            }

            if completed_prompts.contains(&prompt_text) {
                skipped.push(idx);
            } else {
                filtered.push((idx, entry.clone()));
            }
        }

        (filtered, skipped)
    }

    /// Run the batch processing pipeline.
    ///
    /// Faithful port of `BatchRunner.run`. Drives batches sequentially (the
    /// Python module uses a process pool; this port runs each batch in order,
    /// which is observationally equivalent for the produced files and stats),
    /// writing incremental checkpoints, combining batch files into
    /// `trajectories.jsonl`, and writing `statistics.json`.
    ///
    /// `processor` runs a single prompt; `sampler` is accepted for parity but
    /// distribution sampling happens inside the processor.
    pub fn run<P: PromptProcessor, D: DistributionSampler>(
        &mut self,
        resume: bool,
        processor: &P,
        _sampler: &D,
        catalog: &dyn ToolCatalog,
    ) -> Result<(), BatchError> {
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("🚀 Starting Batch Processing");
        eprintln!("{}", "=".repeat(70));

        let all_possible_tools = catalog.all_possible_tools();
        let start_time = std::time::Instant::now();

        // Smart resume: scan batch files by content.
        let mut completed_prompt_texts: HashSet<String> = HashSet::new();
        if resume {
            completed_prompt_texts = self.scan_completed_prompts_by_content();
            if !completed_prompt_texts.is_empty() {
                eprintln!(
                    "   Found {} already-completed prompts by content matching",
                    completed_prompt_texts.len()
                );
            }
        }

        // Filter dataset to only unprocessed prompts.
        if resume && !completed_prompt_texts.is_empty() {
            let (filtered_entries, skipped_indices) =
                self.filter_dataset_by_completed(&completed_prompt_texts);

            if filtered_entries.is_empty() {
                eprintln!("\n✅ All prompts have already been processed!");
                return Ok(());
            }

            let mut batches_to_process: Vec<Vec<(usize, Value)>> = Vec::new();
            let mut i = 0;
            while i < filtered_entries.len() {
                let end = (i + self.cfg.batch_size).min(filtered_entries.len());
                batches_to_process.push(filtered_entries[i..end].to_vec());
                i += self.cfg.batch_size;
            }
            let new_batch_count = batches_to_process.len();
            self.batches = batches_to_process;

            eprintln!("\n{}", "=".repeat(70));
            eprintln!("📊 RESUME SUMMARY");
            eprintln!("{}", "=".repeat(70));
            eprintln!("   Original dataset size:     {} prompts", self.dataset.len());
            eprintln!("   Already completed:         {} prompts", skipped_indices.len());
            eprintln!("   ─────────────────────────────────────────");
            eprintln!("   🎯 RESUMING WITH:          {} prompts", filtered_entries.len());
            eprintln!("   New batches created:       {}", new_batch_count);
            eprintln!("{}\n", "=".repeat(70));
        }

        // Load existing checkpoint.
        let mut checkpoint_data = self.load_checkpoint();
        if checkpoint_data.run_name != self.cfg.run_name {
            checkpoint_data = CheckpointData::empty(&self.cfg.run_name);
        }

        let config = WorkerConfig {
            distribution: self.cfg.distribution.clone(),
            model: self.cfg.model.clone(),
            max_iterations: self.cfg.max_iterations,
            base_url: self.cfg.base_url.clone(),
            api_key: self.cfg.api_key.clone(),
            verbose: self.cfg.verbose,
            ephemeral_system_prompt: self.cfg.ephemeral_system_prompt.clone(),
            log_prefix_chars: self.cfg.log_prefix_chars,
            providers_allowed: self.cfg.providers_allowed.clone(),
            providers_ignored: self.cfg.providers_ignored.clone(),
            providers_order: self.cfg.providers_order.clone(),
            provider_sort: self.cfg.provider_sort.clone(),
            max_tokens: self.cfg.max_tokens,
            reasoning_config: self.cfg.reasoning_config.clone(),
            prefill_messages: self.cfg.prefill_messages.clone(),
        };

        // Track completed indices (backward compat, secondary to content matching).
        let mut completed_prompts_set: HashSet<usize> = checkpoint_data
            .completed_prompts
            .iter()
            .filter_map(|n| usize::try_from(*n).ok())
            .collect();

        let mut total_tool_stats: BTreeMap<String, ToolStat> = BTreeMap::new();

        eprintln!("\n🔧 Initializing {} worker processes...", self.cfg.num_workers);
        eprintln!("✅ Created {} batch tasks", self.batches.len());
        eprintln!("🚀 Starting parallel batch processing...\n");

        let mut results: Vec<BatchResult> = Vec::new();
        let batches = self.batches.clone();
        for (batch_num, batch_data) in batches.iter().enumerate() {
            let result = process_batch_worker(
                batch_num,
                batch_data,
                &self.output_dir,
                &completed_prompts_set,
                &config,
                processor,
                &all_possible_tools,
            )?;

            // Incremental checkpoint update.
            for idx in &result.completed_prompts {
                completed_prompts_set.insert(*idx);
            }
            checkpoint_data.batch_stats.insert(
                result.batch_num.to_string(),
                json!({
                    "processed": result.processed,
                    "skipped": result.skipped,
                    "discarded_no_reasoning": result.discarded_no_reasoning,
                }),
            );
            let mut sorted: Vec<i64> = completed_prompts_set.iter().map(|n| *n as i64).collect();
            sorted.sort_unstable();
            checkpoint_data.completed_prompts = sorted;
            if let Err(e) = self.save_checkpoint(&mut checkpoint_data) {
                eprintln!("⚠️  Warning: Failed to save incremental checkpoint: {}", e);
            }

            results.push(result);
        }

        // Aggregate all batch statistics.
        let mut total_reasoning_stats = ReasoningStats::default();
        for batch_result in &results {
            for (tool_name, stats) in &batch_result.tool_stats {
                let entry = total_tool_stats.entry(tool_name.clone()).or_insert_with(ToolStat::zeroed);
                entry.count += stats.count;
                entry.success += stats.success;
                entry.failure += stats.failure;
            }
            total_reasoning_stats.total_assistant_turns += batch_result.reasoning_stats.total_assistant_turns;
            total_reasoning_stats.turns_with_reasoning += batch_result.reasoning_stats.turns_with_reasoning;
            total_reasoning_stats.turns_without_reasoning += batch_result.reasoning_stats.turns_without_reasoning;
        }

        // Save final checkpoint (best-effort).
        let mut sorted: Vec<i64> = completed_prompts_set.iter().map(|n| *n as i64).collect();
        sorted.sort_unstable();
        checkpoint_data.completed_prompts = sorted;
        if let Err(e) = self.save_checkpoint(&mut checkpoint_data) {
            eprintln!("⚠️  Warning: Failed to save final checkpoint: {}", e);
        }

        // Calculate success rates.
        finalize_success_rates(&mut total_tool_stats);

        // Combine ALL batch files into trajectories.jsonl.
        let combined_file = self.output_dir.join("trajectories.jsonl");
        eprintln!("\n📦 Combining ALL batch files into trajectories.jsonl...");

        let valid_tools = &all_possible_tools;
        let (total_entries, filtered_entries, batch_files_found) =
            combine_batch_files(&self.output_dir, &combined_file, valid_tools)?;

        if filtered_entries > 0 {
            eprintln!(
                "⚠️  Filtered {} corrupted entries out of {} total",
                filtered_entries, total_entries
            );
        }
        eprintln!(
            "✅ Combined {} batch files into trajectories.jsonl ({} entries)",
            batch_files_found,
            total_entries - filtered_entries
        );

        // Save final statistics.
        let final_stats = json!({
            "run_name": self.cfg.run_name,
            "distribution": self.cfg.distribution,
            "total_prompts": self.dataset.len(),
            "total_batches": self.batches.len(),
            "batch_size": self.cfg.batch_size,
            "model": self.cfg.model,
            "completed_at": now_iso(),
            "duration_seconds": round2(start_time.elapsed().as_secs_f64()),
            "tool_statistics": total_tool_stats,
            "reasoning_statistics": total_reasoning_stats,
        });
        let mut stats_f = File::create(&self.stats_file)?;
        stats_f.write_all(serde_json::to_string_pretty(&final_stats)?.as_bytes())?;

        // Summary.
        let processed_this_run: usize = results.iter().map(|r| r.processed).sum();
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("📊 BATCH PROCESSING COMPLETE");
        eprintln!("{}", "=".repeat(70));
        eprintln!("✅ Prompts processed this run: {}", processed_this_run);
        eprintln!("✅ Total trajectories in merged file: {}", total_entries - filtered_entries);
        eprintln!("✅ Total batch files merged: {}", batch_files_found);
        eprintln!("⏱️  Total duration: {}s", round2(start_time.elapsed().as_secs_f64()));

        eprintln!("\n📈 Tool Usage Statistics:");
        eprintln!("{}", "-".repeat(70));
        if !total_tool_stats.is_empty() {
            let mut sorted_tools: Vec<(&String, &ToolStat)> = total_tool_stats.iter().collect();
            sorted_tools.sort_by(|a, b| b.1.count.cmp(&a.1.count));
            eprintln!(
                "{:<25} {:<10} {:<10} {:<10} {:<12}",
                "Tool Name", "Count", "Success", "Failure", "Success Rate"
            );
            eprintln!("{}", "-".repeat(70));
            for (tool_name, stats) in sorted_tools {
                eprintln!(
                    "{:<25} {:<10} {:<10} {:<10} {:.1}%",
                    tool_name,
                    stats.count,
                    stats.success,
                    stats.failure,
                    stats.success_rate.unwrap_or(0.0)
                );
            }
        } else {
            eprintln!("No tool calls were made during this run.");
        }

        let total_discarded: i64 = results.iter().map(|r| r.discarded_no_reasoning).sum();
        eprintln!("\n🧠 Reasoning Coverage:");
        eprintln!("{}", "-".repeat(70));
        let total_turns = total_reasoning_stats.total_assistant_turns;
        let with_reasoning = total_reasoning_stats.turns_with_reasoning;
        let without_reasoning = total_reasoning_stats.turns_without_reasoning;
        if total_turns > 0 {
            let pct_with = round1(with_reasoning as f64 / total_turns as f64 * 100.0);
            let pct_without = round1(without_reasoning as f64 / total_turns as f64 * 100.0);
            eprintln!("   Total assistant turns:    {}", total_turns);
            eprintln!("   With reasoning:           {} ({}%)", with_reasoning, pct_with);
            eprintln!("   Without reasoning:        {} ({}%)", without_reasoning, pct_without);
        } else {
            eprintln!("   No assistant turns recorded.");
        }
        if total_discarded > 0 {
            eprintln!("   🚫 Samples discarded (zero reasoning): {}", total_discarded);
        }

        eprintln!("\n💾 Results saved to: {}", self.output_dir.display());
        eprintln!("   - Trajectories: trajectories.jsonl (combined)");
        eprintln!("   - Individual batches: batch_*.jsonl (for debugging)");
        eprintln!("   - Statistics: statistics.json");
        eprintln!("   - Checkpoint: checkpoint.json");

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Load a dataset from a JSONL file.
///
/// Faithful port of `_load_dataset`: skips blank lines, warns on entries missing
/// a `prompt` field or invalid JSON, and errors if no valid entries remain.
pub fn load_dataset(dataset_file: &Path) -> Result<Vec<Value>, BatchError> {
    if !dataset_file.exists() {
        return Err(BatchError::DatasetNotFound(dataset_file.to_path_buf()));
    }

    let file = File::open(dataset_file)?;
    let reader = BufReader::new(file);
    let mut dataset: Vec<Value> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line_num = i + 1;
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(entry) => {
                if entry.get("prompt").is_none() {
                    eprintln!(
                        "⚠️  Warning: Line {} missing 'prompt' field, skipping",
                        line_num
                    );
                    continue;
                }
                dataset.push(entry);
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Invalid JSON on line {}: {}", line_num, e);
                continue;
            }
        }
    }

    if dataset.is_empty() {
        return Err(BatchError::NoValidEntries(dataset_file.to_path_buf()));
    }

    Ok(dataset)
}

/// Split a dataset into batches of `(index, entry)` pairs.
///
/// Faithful port of `_create_batches`. Indices are the original dataset indices.
pub fn create_batches(dataset: &[Value], batch_size: usize) -> Vec<Vec<(usize, Value)>> {
    let mut batches: Vec<Vec<(usize, Value)>> = Vec::new();
    if batch_size == 0 {
        return batches;
    }
    let mut i = 0;
    while i < dataset.len() {
        let end = (i + batch_size).min(dataset.len());
        let batch: Vec<(usize, Value)> = (i..end).map(|idx| (idx, dataset[idx].clone())).collect();
        batches.push(batch);
        i += batch_size;
    }
    batches
}

/// Add `success_rate`/`failure_rate` to each tool stat, rounded to 2 decimals.
///
/// Faithful port of the success-rate calculation at the end of `run()`.
pub fn finalize_success_rates(total_tool_stats: &mut BTreeMap<String, ToolStat>) {
    for stats in total_tool_stats.values_mut() {
        let total_calls = stats.success + stats.failure;
        if total_calls > 0 {
            stats.success_rate = Some(round2(stats.success as f64 / total_calls as f64 * 100.0));
            stats.failure_rate = Some(round2(stats.failure as f64 / total_calls as f64 * 100.0));
        } else {
            stats.success_rate = Some(0.0);
            stats.failure_rate = Some(0.0);
        }
    }
}

/// Combine all `batch_*.jsonl` files into a single `trajectories.jsonl`,
/// filtering corrupted entries (invalid tool names or invalid JSON).
///
/// Faithful port of the combination loop in `run()`. Returns
/// `(total_entries, filtered_entries, batch_files_found)`.
pub fn combine_batch_files(
    output_dir: &Path,
    combined_file: &Path,
    valid_tools: &HashSet<String>,
) -> std::io::Result<(usize, usize, usize)> {
    let all_batch_files = sorted_batch_files(output_dir);

    let mut total_entries = 0usize;
    let mut filtered_entries = 0usize;
    let mut batch_files_found = 0usize;

    let mut outfile = File::create(combined_file)?;

    for batch_file in &all_batch_files {
        batch_files_found += 1;
        // Extract batch number for logging: stem.split("_")[1].
        let stem = batch_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let batch_num = stem.split('_').nth(1).unwrap_or("");

        let infile = File::open(batch_file)?;
        let reader = BufReader::new(infile);
        for line in reader.lines() {
            let line = line?;
            total_entries += 1;
            let data: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => {
                    filtered_entries += 1;
                    eprintln!("   ⚠️  Filtering invalid JSON entry (batch {})", batch_num);
                    continue;
                }
            };

            let invalid_tools: Vec<&String> = match data.get("tool_stats") {
                Some(Value::Object(ts)) => ts.keys().filter(|k| !valid_tools.contains(*k)).collect(),
                _ => Vec::new(),
            };

            if !invalid_tools.is_empty() {
                filtered_entries += 1;
                let first = invalid_tools[0];
                let preview = if first.chars().count() > 50 {
                    let truncated: String = first.chars().take(50).collect();
                    format!("{}...", truncated)
                } else {
                    first.clone()
                };
                eprintln!(
                    "   ⚠️  Filtering corrupted entry (batch {}): invalid tool '{}'",
                    batch_num, preview
                );
                continue;
            }

            outfile.write_all(line.as_bytes())?;
            outfile.write_all(b"\n")?;
        }
    }

    Ok((total_entries, filtered_entries, batch_files_found))
}

/// Return `batch_*.jsonl` files in the directory, sorted by name.
///
/// Mirrors `sorted(self.output_dir.glob("batch_*.jsonl"))`.
fn sorted_batch_files(output_dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(output_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.starts_with("batch_") && name.ends_with(".jsonl") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

/// ISO-8601 timestamp matching Python's `datetime.now().isoformat()` shape
/// (local time, microsecond precision, no timezone suffix).
pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

/// Round to two decimal places (Python `round(x, 2)`, banker's-rounding aside).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Round to one decimal place (Python `round(x, 1)`).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Parse a comma-separated provider list into `Some(Vec<String>)`, trimming
/// each element, or `None` if the input is `None`/empty.
///
/// Mirrors the `[p.strip() for p in providers.split(",")] if providers else None`
/// logic in `main`.
pub fn parse_provider_list(value: Option<&str>) -> Option<Vec<String>> {
    match value {
        Some(s) if !s.is_empty() => Some(s.split(',').map(|p| p.trim().to_string()).collect()),
        _ => None,
    }
}

/// Build a `reasoning_config` value from the CLI flags.
///
/// Mirrors the `--reasoning_disabled` / `--reasoning_effort` precedence in
/// `main`. Returns `Ok(None)` when neither is set, `Err(message)` for an invalid
/// effort level.
pub fn build_reasoning_config(
    reasoning_disabled: bool,
    reasoning_effort: Option<&str>,
) -> Result<Option<Value>, String> {
    if reasoning_disabled {
        return Ok(Some(json!({"effort": "none"})));
    }
    if let Some(effort) = reasoning_effort {
        let valid = ["none", "minimal", "low", "medium", "high", "xhigh"];
        if !valid.contains(&effort) {
            return Err(format!(
                "--reasoning_effort must be one of: {}",
                valid.join(", ")
            ));
        }
        return Ok(Some(json!({"enabled": true, "effort": effort})));
    }
    Ok(None)
}

/// Convenience: build the catalog from an iterator of tool names.
///
/// Mirrors `ALL_POSSIBLE_TOOLS = set(TOOL_TO_TOOLSET_MAP.keys())`.
pub fn all_possible_tools_from<I, S>(tools: I) -> HashSet<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    tools.into_iter().map(Into::into).collect()
}

/// Build the set of human prompt texts present in a list of trajectory entries.
/// Internal helper kept for potential reuse / testing of the resume path.
#[allow(dead_code)]
fn human_prompts_in_entries(entries: &[Value]) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for entry in entries {
        if let Some(Value::Array(convs)) = entry.get("conversations") {
            for msg in convs {
                if msg.get("from").and_then(Value::as_str) == Some("human") {
                    if let Some(v) = msg.get("value").and_then(Value::as_str) {
                        let t = v.trim();
                        if !t.is_empty() {
                            set.insert(t.to_string());
                        }
                    }
                    break;
                }
            }
        }
    }
    set
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant_with_calls(calls: Value) -> Value {
        json!({"role": "assistant", "content": "", "tool_calls": calls})
    }

    fn tool_response(id: &str, content: Value) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": content})
    }

    #[test]
    fn extract_tool_stats_counts_success_and_failure() {
        let messages = vec![
            assistant_with_calls(json!([
                {"id": "c1", "function": {"name": "bash"}},
                {"id": "c2", "function": {"name": "bash"}},
            ])),
            tool_response("c1", Value::String("ok output".into())),
            tool_response("c2", Value::String(json!({"error": "boom"}).to_string())),
        ];
        let stats = extract_tool_stats(&messages);
        let bash = stats.get("bash").unwrap();
        assert_eq!(bash.count, 2);
        assert_eq!(bash.success, 1);
        assert_eq!(bash.failure, 1);
    }

    #[test]
    fn null_error_field_is_success() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "t"}}])),
            tool_response("c1", json!({"error": null, "result": "fine"})),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("t").unwrap().success, 1);
        assert_eq!(stats.get("t").unwrap().failure, 0);
    }

    #[test]
    fn nested_content_error_is_failure() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "terminal"}}])),
            tool_response("c1", json!({"content": {"error": "nonzero", "exit_code": 1}})),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("terminal").unwrap().failure, 1);
    }

    #[test]
    fn success_false_pattern_is_failure() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "x"}}])),
            tool_response("c1", json!({"success": false})),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("x").unwrap().failure, 1);
    }

    #[test]
    fn empty_string_content_is_failure() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "y"}}])),
            tool_response("c1", Value::String(String::new())),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("y").unwrap().failure, 1);
    }

    #[test]
    fn error_prefix_string_is_failure() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "z"}}])),
            tool_response("c1", Value::String("Error: something went wrong".into())),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("z").unwrap().failure, 1);
    }

    #[test]
    fn non_error_plain_string_is_success() {
        let messages = vec![
            assistant_with_calls(json!([{"id": "c1", "function": {"name": "w"}}])),
            // Not JSON, not empty, doesn't start with "error:".
            tool_response("c1", Value::String("the error happened later".into())),
        ];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("w").unwrap().success, 1);
    }

    #[test]
    fn skips_null_and_non_dict_tool_calls() {
        let messages = vec![assistant_with_calls(json!([
            Value::Null,
            {"id": "c1", "function": {"name": "good"}},
        ]))];
        let stats = extract_tool_stats(&messages);
        assert_eq!(stats.get("good").unwrap().count, 1);
        assert_eq!(stats.len(), 1);
    }

    #[test]
    fn reasoning_stats_scratchpad_and_native() {
        let messages = vec![
            json!({"role": "assistant", "content": "<REASONING_SCRATCHPAD>think</REASONING_SCRATCHPAD>"}),
            json!({"role": "assistant", "content": "no reasoning here"}),
            json!({"role": "assistant", "content": "x", "reasoning": "native thoughts"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let stats = extract_reasoning_stats(&messages);
        assert_eq!(stats.total_assistant_turns, 3);
        assert_eq!(stats.turns_with_reasoning, 2);
        assert_eq!(stats.turns_without_reasoning, 1);
        assert!(stats.has_any_reasoning);
    }

    #[test]
    fn reasoning_empty_native_string_does_not_count() {
        let messages = vec![json!({"role": "assistant", "content": "x", "reasoning": "   "})];
        let stats = extract_reasoning_stats(&messages);
        assert_eq!(stats.turns_with_reasoning, 0);
    }

    #[test]
    fn normalize_tool_stats_fills_all_and_keeps_extra() {
        let mut raw = BTreeMap::new();
        raw.insert(
            "bash".to_string(),
            ToolStat { count: 3, success: 2, failure: 1, ..Default::default() },
        );
        raw.insert(
            "weird_unknown".to_string(),
            ToolStat { count: 1, success: 1, failure: 0, ..Default::default() },
        );
        let all: HashSet<String> = ["bash", "python"].iter().map(|s| s.to_string()).collect();
        let norm = normalize_tool_stats(&raw, &all);
        assert_eq!(norm.get("bash").unwrap().count, 3);
        assert_eq!(norm.get("python").unwrap(), &ToolStat::zeroed());
        assert_eq!(norm.get("weird_unknown").unwrap().count, 1);
    }

    #[test]
    fn normalize_error_counts_fills_all() {
        let mut raw = BTreeMap::new();
        raw.insert("bash".to_string(), 2i64);
        let all: HashSet<String> = ["bash", "python"].iter().map(|s| s.to_string()).collect();
        let norm = normalize_tool_error_counts(&raw, &all);
        assert_eq!(*norm.get("bash").unwrap(), 2);
        assert_eq!(*norm.get("python").unwrap(), 0);
    }

    #[test]
    fn create_batches_uses_original_indices() {
        let dataset: Vec<Value> = (0..5).map(|i| json!({"prompt": format!("p{i}")})).collect();
        let batches = create_batches(&dataset, 2);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].0, 0);
        assert_eq!(batches[1][0].0, 2);
        assert_eq!(batches[2][0].0, 4);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn finalize_success_rates_computes_percentages() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "t".to_string(),
            ToolStat { count: 4, success: 3, failure: 1, ..Default::default() },
        );
        stats.insert(
            "unused".to_string(),
            ToolStat::zeroed(),
        );
        finalize_success_rates(&mut stats);
        assert_eq!(stats.get("t").unwrap().success_rate, Some(75.0));
        assert_eq!(stats.get("t").unwrap().failure_rate, Some(25.0));
        assert_eq!(stats.get("unused").unwrap().success_rate, Some(0.0));
    }

    #[test]
    fn build_reasoning_config_precedence() {
        assert_eq!(
            build_reasoning_config(true, Some("high")).unwrap(),
            Some(json!({"effort": "none"}))
        );
        assert_eq!(
            build_reasoning_config(false, Some("high")).unwrap(),
            Some(json!({"enabled": true, "effort": "high"}))
        );
        assert_eq!(build_reasoning_config(false, None).unwrap(), None);
        assert!(build_reasoning_config(false, Some("bogus")).is_err());
    }

    #[test]
    fn parse_provider_list_trims_and_handles_none() {
        assert_eq!(
            parse_provider_list(Some("anthropic, openai , google")),
            Some(vec![
                "anthropic".to_string(),
                "openai".to_string(),
                "google".to_string()
            ])
        );
        assert_eq!(parse_provider_list(None), None);
        assert_eq!(parse_provider_list(Some("")), None);
    }

    #[test]
    fn load_dataset_skips_blank_invalid_and_missing_prompt() {
        let dir = std::env::temp_dir().join(format!("br_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.jsonl");
        let contents = "\n{\"prompt\": \"a\"}\n{bad json}\n{\"no_prompt\": 1}\n{\"prompt\": \"b\"}\n";
        std::fs::write(&path, contents).unwrap();
        let ds = load_dataset(&path).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds[0]["prompt"], json!("a"));
        assert_eq!(ds[1]["prompt"], json!("b"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn combine_batch_files_filters_invalid_tools_and_json() {
        let dir = std::env::temp_dir().join(format!("br_combine_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let good = json!({"tool_stats": {"bash": {"count": 1, "success": 1, "failure": 0}}});
        let bad_tool = json!({"tool_stats": {"hallucinated_tool": {"count": 1}}});
        let batch_path = dir.join("batch_0.jsonl");
        let mut content = String::new();
        content.push_str(&serde_json::to_string(&good).unwrap());
        content.push('\n');
        content.push_str(&serde_json::to_string(&bad_tool).unwrap());
        content.push('\n');
        content.push_str("{not valid json}\n");
        std::fs::write(&batch_path, content).unwrap();

        let valid: HashSet<String> = ["bash"].iter().map(|s| s.to_string()).collect();
        let combined = dir.join("trajectories.jsonl");
        let (total, filtered, found) = combine_batch_files(&dir, &combined, &valid).unwrap();
        assert_eq!(total, 3);
        assert_eq!(filtered, 2);
        assert_eq!(found, 1);

        let out = std::fs::read_to_string(&combined).unwrap();
        assert_eq!(out.lines().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
