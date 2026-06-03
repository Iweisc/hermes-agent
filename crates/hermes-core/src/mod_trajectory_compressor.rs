//! Trajectory Compressor — native Rust port of `trajectory_compressor.py`.
//!
//! Post-processes completed agent trajectories to compress them within a target
//! token budget while preserving training-signal quality.
//!
//! Compression strategy (faithful to the Python original):
//!   1. Protect first turns (system, human, first gpt, first tool).
//!   2. Protect last N turns (final actions and conclusions).
//!   3. Compress MIDDLE turns only, starting from the 2nd tool response region.
//!   4. Compress only as much as needed to fit under target.
//!   5. Replace the compressed region with a single `human` summary message.
//!   6. Keep remaining tool calls intact (model continues after the summary).
//!
//! Notable porting decisions:
//!   * The Python uses a HuggingFace `AutoTokenizer` for token counting. There
//!     is no native HF tokenizer here, so [`TokenCounter`] is a trait. The
//!     default [`HeuristicTokenCounter`] reproduces the Python's documented
//!     `len(text) // 4` fallback path. A caller that wants exact counts can
//!     supply its own implementation.
//!   * Summarization uses [`crate::ag_auxiliary_client::ChatCompletionsClient`]
//!     (reqwest blocking) for the custom-endpoint path. Provider-routed
//!     `call_llm` is represented via a pluggable [`SummaryClient`] trait so the
//!     module does not hard-depend on the (not-yet-ported) provider router.
//!   * Async variants in the Python collapse to the sync path here; concurrency
//!     is the caller's concern (the Rust gateway is already multi-threaded).

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::ag_auxiliary_client::{
    fixed_temperature_for_model, to_openai_base_url, ChatCompletionsClient, TemperatureDirective,
};
use crate::agent_retry::jittered_backoff_with;
use crate::mod_hermes_constants::OPENROUTER_BASE_URL;
use crate::mod_utils::{base_url_host_matches, base_url_hostname};

// ── Effective temperature ────────────────────────────────────────────────────

/// Apply fixed model temperature contracts to direct client calls.
///
/// Returns `None` when the model manages temperature server-side (Kimi);
/// callers must omit the `temperature` kwarg entirely in that case. Otherwise
/// returns the (possibly overridden) temperature to use.
///
/// Mirrors `_effective_temperature_for_model`.
pub fn effective_temperature_for_model(
    model: &str,
    requested_temperature: f64,
    base_url: Option<&str>,
) -> Option<f64> {
    match fixed_temperature_for_model(Some(model), base_url) {
        TemperatureDirective::Omit => None,
        TemperatureDirective::Fixed(v) => Some(v),
        TemperatureDirective::None => Some(requested_temperature),
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for trajectory compression. Mirrors `CompressionConfig`.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    // Tokenizer
    pub tokenizer_name: String,
    pub trust_remote_code: bool,

    // Compression targets
    pub target_max_tokens: i64,
    pub summary_target_tokens: i64,

    // Protected turns
    pub protect_first_system: bool,
    pub protect_first_human: bool,
    pub protect_first_gpt: bool,
    pub protect_first_tool: bool,
    pub protect_last_n_turns: i64,

    // Summarization
    pub summarization_model: String,
    pub base_url: String,
    pub api_key_env: String,
    pub temperature: f64,
    pub max_retries: i64,
    pub retry_delay: i64,

    // Output
    pub add_summary_notice: bool,
    pub summary_notice_text: String,
    pub output_suffix: String,

    // Processing
    pub num_workers: i64,
    pub max_concurrent_requests: i64,
    pub skip_under_target: bool,
    pub save_over_limit: bool,
    pub per_trajectory_timeout: i64,

    // Metrics
    pub metrics_enabled: bool,
    pub metrics_per_trajectory: bool,
    pub metrics_output_file: String,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            tokenizer_name: "moonshotai/Kimi-K2-Thinking".to_string(),
            trust_remote_code: true,

            target_max_tokens: 15250,
            summary_target_tokens: 750,

            protect_first_system: true,
            protect_first_human: true,
            protect_first_gpt: true,
            protect_first_tool: true,
            protect_last_n_turns: 4,

            summarization_model: "google/gemini-3-flash-preview".to_string(),
            base_url: OPENROUTER_BASE_URL.to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            temperature: 0.3,
            max_retries: 3,
            retry_delay: 2,

            add_summary_notice: true,
            summary_notice_text:
                "\n\nSome of your previous tool responses may be summarized to preserve context."
                    .to_string(),
            output_suffix: "_compressed".to_string(),

            num_workers: 4,
            max_concurrent_requests: 50,
            skip_under_target: true,
            save_over_limit: true,
            per_trajectory_timeout: 300,

            metrics_enabled: true,
            metrics_per_trajectory: true,
            metrics_output_file: "compression_metrics.json".to_string(),
        }
    }
}

impl CompressionConfig {
    /// Load configuration from a parsed YAML value. Mirrors `from_yaml` field
    /// mapping exactly: any missing section/key falls back to the default.
    pub fn from_yaml_value(data: &Value) -> Self {
        let mut config = Self::default();

        // Tokenizer
        if let Some(t) = data.get("tokenizer") {
            if let Some(v) = t.get("name").and_then(|v| v.as_str()) {
                config.tokenizer_name = v.to_string();
            }
            if let Some(v) = t.get("trust_remote_code").and_then(|v| v.as_bool()) {
                config.trust_remote_code = v;
            }
        }

        // Compression
        if let Some(c) = data.get("compression") {
            if let Some(v) = c.get("target_max_tokens").and_then(as_i64) {
                config.target_max_tokens = v;
            }
            if let Some(v) = c.get("summary_target_tokens").and_then(as_i64) {
                config.summary_target_tokens = v;
            }
        }

        // Protected turns
        if let Some(p) = data.get("protected_turns") {
            if let Some(v) = p.get("first_system").and_then(|v| v.as_bool()) {
                config.protect_first_system = v;
            }
            if let Some(v) = p.get("first_human").and_then(|v| v.as_bool()) {
                config.protect_first_human = v;
            }
            if let Some(v) = p.get("first_gpt").and_then(|v| v.as_bool()) {
                config.protect_first_gpt = v;
            }
            if let Some(v) = p.get("first_tool").and_then(|v| v.as_bool()) {
                config.protect_first_tool = v;
            }
            if let Some(v) = p.get("last_n_turns").and_then(as_i64) {
                config.protect_last_n_turns = v;
            }
        }

        // Summarization
        if let Some(s) = data.get("summarization") {
            if let Some(v) = s.get("model").and_then(|v| v.as_str()) {
                config.summarization_model = v.to_string();
            }
            // `base_url` uses `or` semantics: only override when truthy.
            if let Some(v) = s.get("base_url").and_then(|v| v.as_str()) {
                if !v.is_empty() {
                    config.base_url = v.to_string();
                }
            }
            if let Some(v) = s.get("api_key_env").and_then(|v| v.as_str()) {
                config.api_key_env = v.to_string();
            }
            if let Some(v) = s.get("temperature").and_then(|v| v.as_f64()) {
                config.temperature = v;
            }
            if let Some(v) = s.get("max_retries").and_then(as_i64) {
                config.max_retries = v;
            }
            if let Some(v) = s.get("retry_delay").and_then(as_i64) {
                config.retry_delay = v;
            }
        }

        // Output
        if let Some(o) = data.get("output") {
            if let Some(v) = o.get("add_summary_notice").and_then(|v| v.as_bool()) {
                config.add_summary_notice = v;
            }
            if let Some(v) = o.get("summary_notice_text").and_then(|v| v.as_str()) {
                config.summary_notice_text = v.to_string();
            }
            if let Some(v) = o.get("output_suffix").and_then(|v| v.as_str()) {
                config.output_suffix = v.to_string();
            }
        }

        // Processing
        if let Some(p) = data.get("processing") {
            if let Some(v) = p.get("num_workers").and_then(as_i64) {
                config.num_workers = v;
            }
            if let Some(v) = p.get("max_concurrent_requests").and_then(as_i64) {
                config.max_concurrent_requests = v;
            }
            if let Some(v) = p.get("skip_under_target").and_then(|v| v.as_bool()) {
                config.skip_under_target = v;
            }
            if let Some(v) = p.get("save_over_limit").and_then(|v| v.as_bool()) {
                config.save_over_limit = v;
            }
        }

        // Metrics
        if let Some(m) = data.get("metrics") {
            if let Some(v) = m.get("enabled").and_then(|v| v.as_bool()) {
                config.metrics_enabled = v;
            }
            if let Some(v) = m.get("per_trajectory").and_then(|v| v.as_bool()) {
                config.metrics_per_trajectory = v;
            }
            if let Some(v) = m.get("output_file").and_then(|v| v.as_str()) {
                config.metrics_output_file = v.to_string();
            }
        }

        config
    }

    /// Load configuration from a YAML file at `yaml_path`. Mirrors `from_yaml`.
    pub fn from_yaml_file(yaml_path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(yaml_path).map_err(|e| format!("read {yaml_path:?}: {e}"))?;
        let data: Value =
            serde_yaml::from_str(&text).map_err(|e| format!("parse yaml {yaml_path:?}: {e}"))?;
        Ok(Self::from_yaml_value(&data))
    }
}

/// Coerce a JSON/YAML number to i64, accepting integer-valued floats too (YAML
/// can deserialize `15250` as either depending on quoting).
fn as_i64(v: &Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        return Some(i);
    }
    v.as_f64().map(|f| f as i64)
}

// ── Per-trajectory metrics ─────────────────────────────────────────────────────

/// Metrics for a single trajectory compression. Mirrors `TrajectoryMetrics`.
#[derive(Debug, Clone)]
pub struct TrajectoryMetrics {
    pub original_tokens: i64,
    pub compressed_tokens: i64,
    pub tokens_saved: i64,
    pub compression_ratio: f64,

    pub original_turns: i64,
    pub compressed_turns: i64,
    pub turns_removed: i64,

    pub turns_compressed_start_idx: i64,
    pub turns_compressed_end_idx: i64,
    pub turns_in_compressed_region: i64,

    pub was_compressed: bool,
    pub still_over_limit: bool,
    pub skipped_under_target: bool,

    pub summarization_api_calls: i64,
    pub summarization_errors: i64,
}

impl Default for TrajectoryMetrics {
    fn default() -> Self {
        Self {
            original_tokens: 0,
            compressed_tokens: 0,
            tokens_saved: 0,
            compression_ratio: 1.0,
            original_turns: 0,
            compressed_turns: 0,
            turns_removed: 0,
            turns_compressed_start_idx: -1,
            turns_compressed_end_idx: -1,
            turns_in_compressed_region: 0,
            was_compressed: false,
            still_over_limit: false,
            skipped_under_target: false,
            summarization_api_calls: 0,
            summarization_errors: 0,
        }
    }
}

impl TrajectoryMetrics {
    /// Serialize to the same JSON shape as Python's `to_dict`.
    pub fn to_dict(&self) -> Value {
        json!({
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round_to(self.compression_ratio, 4),
            "original_turns": self.original_turns,
            "compressed_turns": self.compressed_turns,
            "turns_removed": self.turns_removed,
            "compression_region": {
                "start_idx": self.turns_compressed_start_idx,
                "end_idx": self.turns_compressed_end_idx,
                "turns_count": self.turns_in_compressed_region,
            },
            "was_compressed": self.was_compressed,
            "still_over_limit": self.still_over_limit,
            "skipped_under_target": self.skipped_under_target,
            "summarization_api_calls": self.summarization_api_calls,
            "summarization_errors": self.summarization_errors,
        })
    }
}

// ── Aggregate metrics ──────────────────────────────────────────────────────────

/// Aggregate metrics across all trajectories. Mirrors `AggregateMetrics`.
#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    pub total_trajectories: i64,
    pub trajectories_compressed: i64,
    pub trajectories_skipped_under_target: i64,
    pub trajectories_still_over_limit: i64,
    pub trajectories_failed: i64,

    pub total_tokens_before: i64,
    pub total_tokens_after: i64,
    pub total_tokens_saved: i64,

    pub total_turns_before: i64,
    pub total_turns_after: i64,
    pub total_turns_removed: i64,

    pub total_summarization_calls: i64,
    pub total_summarization_errors: i64,

    pub compression_ratios: Vec<f64>,
    pub tokens_saved_list: Vec<i64>,
    pub turns_removed_list: Vec<i64>,

    pub processing_start_time: String,
    pub processing_end_time: String,
    pub processing_duration_seconds: f64,
}

impl AggregateMetrics {
    /// Add a single trajectory's metrics to the aggregate. Mirrors
    /// `add_trajectory_metrics`.
    pub fn add_trajectory_metrics(&mut self, metrics: &TrajectoryMetrics) {
        self.total_trajectories += 1;
        self.total_tokens_before += metrics.original_tokens;
        self.total_tokens_after += metrics.compressed_tokens;
        self.total_tokens_saved += metrics.tokens_saved;
        self.total_turns_before += metrics.original_turns;
        self.total_turns_after += metrics.compressed_turns;
        self.total_turns_removed += metrics.turns_removed;
        self.total_summarization_calls += metrics.summarization_api_calls;
        self.total_summarization_errors += metrics.summarization_errors;

        if metrics.was_compressed {
            self.trajectories_compressed += 1;
            self.compression_ratios.push(metrics.compression_ratio);
            self.tokens_saved_list.push(metrics.tokens_saved);
            self.turns_removed_list.push(metrics.turns_removed);
        }

        if metrics.skipped_under_target {
            self.trajectories_skipped_under_target += 1;
        }

        if metrics.still_over_limit {
            self.trajectories_still_over_limit += 1;
        }
    }

    /// Serialize to the same nested JSON shape as Python's `to_dict`.
    pub fn to_dict(&self) -> Value {
        let avg_compression_ratio = if self.compression_ratios.is_empty() {
            1.0
        } else {
            self.compression_ratios.iter().sum::<f64>() / self.compression_ratios.len() as f64
        };
        let avg_tokens_saved = if self.tokens_saved_list.is_empty() {
            0.0
        } else {
            self.tokens_saved_list.iter().sum::<i64>() as f64 / self.tokens_saved_list.len() as f64
        };
        let avg_turns_removed = if self.turns_removed_list.is_empty() {
            0.0
        } else {
            self.turns_removed_list.iter().sum::<i64>() as f64
                / self.turns_removed_list.len() as f64
        };

        json!({
            "summary": {
                "total_trajectories": self.total_trajectories,
                "trajectories_compressed": self.trajectories_compressed,
                "trajectories_skipped_under_target": self.trajectories_skipped_under_target,
                "trajectories_still_over_limit": self.trajectories_still_over_limit,
                "trajectories_failed": self.trajectories_failed,
                "compression_rate": round_to(
                    self.trajectories_compressed as f64 / self.total_trajectories.max(1) as f64,
                    4,
                ),
            },
            "tokens": {
                "total_before": self.total_tokens_before,
                "total_after": self.total_tokens_after,
                "total_saved": self.total_tokens_saved,
                "overall_compression_ratio": round_to(
                    self.total_tokens_after as f64 / self.total_tokens_before.max(1) as f64,
                    4,
                ),
            },
            "turns": {
                "total_before": self.total_turns_before,
                "total_after": self.total_turns_after,
                "total_removed": self.total_turns_removed,
            },
            "averages": {
                "avg_compression_ratio": round_to(avg_compression_ratio, 4),
                "avg_tokens_saved_per_compressed": round_to(avg_tokens_saved, 1),
                "avg_turns_removed_per_compressed": round_to(avg_turns_removed, 2),
            },
            "summarization": {
                "total_api_calls": self.total_summarization_calls,
                "total_errors": self.total_summarization_errors,
                "success_rate": round_to(
                    1.0 - (self.total_summarization_errors as f64
                        / self.total_summarization_calls.max(1) as f64),
                    4,
                ),
            },
            "processing": {
                "start_time": self.processing_start_time,
                "end_time": self.processing_end_time,
                "duration_seconds": round_to(self.processing_duration_seconds, 2),
            },
        })
    }
}

/// Round a float to `ndigits` decimal places (banker's-rounding-free, matching
/// Python's `round()` closely enough for metric reporting).
fn round_to(v: f64, ndigits: u32) -> f64 {
    let factor = 10f64.powi(ndigits as i32);
    (v * factor).round() / factor
}

// ── Token counting ─────────────────────────────────────────────────────────────

/// Abstraction over token counting. The Python original uses a HuggingFace
/// `AutoTokenizer`; callers may inject an exact implementation. The default
/// reproduces the Python's documented fallback (`len(text) // 4`).
pub trait TokenCounter: Send + Sync {
    /// Count tokens in `text`. Empty text counts as 0 (matching Python).
    fn count(&self, text: &str) -> i64;
}

/// Character-heuristic token counter: `chars / 4`, with empty → 0.
///
/// This is exactly the fallback path the Python uses when the tokenizer raises.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, text: &str) -> i64 {
        if text.is_empty() {
            return 0;
        }
        // Python `len(text) // 4` counts Unicode codepoints, not bytes.
        (text.chars().count() / 4) as i64
    }
}

// ── Summarization client ────────────────────────────────────────────────────────

/// Pluggable summary backend. The default [`OpenAICompatSummaryClient`] posts to
/// an OpenAI-compatible `/chat/completions` endpoint via reqwest blocking.
pub trait SummaryClient: Send + Sync {
    /// Run one summarization request. `prompt` is the user message. `temperature`
    /// is `None` when the model manages temperature server-side (it must then be
    /// omitted from the request body). Returns the assistant message content.
    fn complete(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f64>,
        max_tokens: i64,
    ) -> Result<String, String>;
}

/// Summary client targeting an OpenAI-compatible chat-completions endpoint.
pub struct OpenAICompatSummaryClient {
    client: ChatCompletionsClient,
    timeout: Duration,
}

impl OpenAICompatSummaryClient {
    /// Construct from a base URL + API key. The base URL is normalised through
    /// [`to_openai_base_url`] to mirror the Python `_to_openai_base_url`.
    pub fn new(base_url: &str, api_key: &str, timeout: Duration) -> Self {
        Self {
            client: ChatCompletionsClient::new(to_openai_base_url(base_url), api_key),
            timeout,
        }
    }

    /// Attach extra default headers (e.g. OpenRouter referer/title headers).
    pub fn with_headers(mut self, headers: BTreeMap<String, String>) -> Self {
        self.client = self.client.with_headers(headers);
        self
    }
}

impl SummaryClient for OpenAICompatSummaryClient {
    fn complete(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f64>,
        max_tokens: i64,
    ) -> Result<String, String> {
        let mut kwargs = Map::new();
        kwargs.insert("model".to_string(), Value::String(model.to_string()));
        kwargs.insert(
            "messages".to_string(),
            json!([{ "role": "user", "content": prompt }]),
        );
        kwargs.insert("max_tokens".to_string(), json!(max_tokens));
        if let Some(t) = temperature {
            kwargs.insert("temperature".to_string(), json!(t));
        }
        let response = self
            .client
            .create(&Value::Object(kwargs), self.timeout)
            .map_err(|e| e.message)?;
        Ok(extract_message_content(&response))
    }
}

/// Extract `choices[0].message.content` from a chat-completions response,
/// returning an empty string when absent (Python coerces None → "").
fn extract_message_content(response: &Value) -> String {
    response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// ── Summary text normalisation ───────────────────────────────────────────────

/// Normalize summary-model output to a safe trimmed string. Mirrors
/// `_coerce_summary_content`.
pub fn coerce_summary_content(content: &Value) -> String {
    let s = match content {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    s.trim().to_string()
}

/// Normalize summary text to include the `[CONTEXT SUMMARY]:` prefix exactly
/// once. Mirrors `_ensure_summary_prefix`.
pub fn ensure_summary_prefix(summary: &str) -> String {
    let text = summary.trim();
    if text.starts_with("[CONTEXT SUMMARY]:") {
        return text.to_string();
    }
    if text.is_empty() {
        "[CONTEXT SUMMARY]:".to_string()
    } else {
        format!("[CONTEXT SUMMARY]: {text}")
    }
}

/// Fallback summary used when all summarization attempts fail.
pub const FALLBACK_SUMMARY: &str = "[CONTEXT SUMMARY]: [Summary generation failed - previous turns contained tool calls and responses that have been compressed to save context space.]";

// ── Provider detection ───────────────────────────────────────────────────────

/// Detect the provider name from a configured base URL. Mirrors
/// `_detect_provider` (returns empty string for unknown URLs).
pub fn detect_provider(base_url: &str) -> String {
    let url = base_url;
    if base_url_host_matches(url, "openrouter.ai") {
        return "openrouter".to_string();
    }
    if base_url_host_matches(url, "nousresearch.com") {
        return "nous".to_string();
    }
    if base_url_hostname(url) == "chatgpt.com" && url.to_lowercase().contains("/backend-api/codex") {
        return "codex".to_string();
    }
    if base_url_host_matches(url, "z.ai") {
        return "zai".to_string();
    }
    if base_url_host_matches(url, "moonshot.ai")
        || base_url_host_matches(url, "moonshot.cn")
        || base_url_host_matches(url, "api.kimi.com")
    {
        return "kimi-coding".to_string();
    }
    if base_url_host_matches(url, "arcee.ai") {
        return "arcee".to_string();
    }
    if base_url_host_matches(url, "minimaxi.com") {
        return "minimax-cn".to_string();
    }
    if base_url_host_matches(url, "minimax.io") {
        return "minimax".to_string();
    }
    String::new()
}

// ── Turn helpers ─────────────────────────────────────────────────────────────

/// Read a turn's string field (`from` / `value`), defaulting to `default`.
fn turn_str<'a>(turn: &'a Value, key: &str, default: &'a str) -> &'a str {
    turn.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

// ── The compressor ──────────────────────────────────────────────────────────────

/// Compresses agent trajectories to fit within a target token budget.
///
/// Holds borrowed references to a token counter and an optional summary client.
/// The Python `TrajectoryCompressor` owns these; here they are injected so the
/// module can be exercised without network access or a HF tokenizer.
pub struct TrajectoryCompressor<'a> {
    pub config: CompressionConfig,
    pub aggregate_metrics: AggregateMetrics,
    counter: &'a dyn TokenCounter,
    summary_client: Option<&'a dyn SummaryClient>,
}

impl<'a> TrajectoryCompressor<'a> {
    /// Construct a compressor. `summary_client` may be `None`, in which case
    /// summary generation always returns [`FALLBACK_SUMMARY`] after exhausting
    /// retries (matching the Python fallback when the API keeps failing).
    pub fn new(
        config: CompressionConfig,
        counter: &'a dyn TokenCounter,
        summary_client: Option<&'a dyn SummaryClient>,
    ) -> Self {
        Self {
            config,
            aggregate_metrics: AggregateMetrics::default(),
            counter,
            summary_client,
        }
    }

    /// Count tokens in text. Mirrors `count_tokens`.
    pub fn count_tokens(&self, text: &str) -> i64 {
        if text.is_empty() {
            return 0;
        }
        self.counter.count(text)
    }

    /// Count total tokens in a trajectory. Mirrors `count_trajectory_tokens`.
    pub fn count_trajectory_tokens(&self, trajectory: &[Value]) -> i64 {
        trajectory
            .iter()
            .map(|turn| self.count_tokens(turn_str(turn, "value", "")))
            .sum()
    }

    /// Count tokens for each turn. Mirrors `count_turn_tokens`.
    pub fn count_turn_tokens(&self, trajectory: &[Value]) -> Vec<i64> {
        trajectory
            .iter()
            .map(|turn| self.count_tokens(turn_str(turn, "value", "")))
            .collect()
    }

    /// Find protected turn indices and the compressible region. Mirrors
    /// `_find_protected_indices`. Returns `(protected, compress_start,
    /// compress_end)`.
    pub fn find_protected_indices(&self, trajectory: &[Value]) -> (HashSet<usize>, usize, usize) {
        let n = trajectory.len();
        let mut protected: HashSet<usize> = HashSet::new();

        let mut first_system: Option<usize> = None;
        let mut first_human: Option<usize> = None;
        let mut first_gpt: Option<usize> = None;
        let mut first_tool: Option<usize> = None;

        for (i, turn) in trajectory.iter().enumerate() {
            let role = turn_str(turn, "from", "");
            match role {
                "system" if first_system.is_none() => first_system = Some(i),
                "human" if first_human.is_none() => first_human = Some(i),
                "gpt" if first_gpt.is_none() => first_gpt = Some(i),
                "tool" if first_tool.is_none() => first_tool = Some(i),
                _ => {}
            }
        }

        if self.config.protect_first_system {
            if let Some(i) = first_system {
                protected.insert(i);
            }
        }
        if self.config.protect_first_human {
            if let Some(i) = first_human {
                protected.insert(i);
            }
        }
        if self.config.protect_first_gpt {
            if let Some(i) = first_gpt {
                protected.insert(i);
            }
        }
        if self.config.protect_first_tool {
            if let Some(i) = first_tool {
                protected.insert(i);
            }
        }

        // Protect last N turns: range(max(0, n - last_n), n).
        let last_n = self.config.protect_last_n_turns.max(0) as usize;
        let start = n.saturating_sub(last_n);
        for i in start..n {
            protected.insert(i);
        }

        // Split protected into head/tail around the midpoint (n // 2).
        let mid = n / 2;
        let head_protected: Vec<usize> = protected.iter().copied().filter(|&i| i < mid).collect();
        let tail_protected: Vec<usize> = protected.iter().copied().filter(|&i| i >= mid).collect();

        let compressible_start = head_protected.iter().copied().max().map_or(0, |m| m + 1);
        let compressible_end = tail_protected.iter().copied().min().unwrap_or(n);

        (protected, compressible_start, compressible_end)
    }

    /// Format turn contents for the summary prompt. Mirrors
    /// `_extract_turn_content_for_summary`. `end` is exclusive.
    pub fn extract_turn_content_for_summary(
        &self,
        trajectory: &[Value],
        start: usize,
        end: usize,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();
        for i in start..end {
            let turn = &trajectory[i];
            let role = turn_str(turn, "from", "unknown");
            let mut value = turn_str(turn, "value", "").to_string();

            // Truncate very long values for the summary prompt. The Python uses
            // `value[:1500]` and `value[-500:]` on a Python str (codepoints).
            if value.chars().count() > 3000 {
                let head: String = value.chars().take(1500).collect();
                let total = value.chars().count();
                let tail: String = value.chars().skip(total - 500).collect();
                value = format!("{head}\n...[truncated]...\n{tail}");
            }

            parts.push(format!("[Turn {i} - {}]:\n{value}", role.to_uppercase()));
        }
        parts.join("\n\n")
    }

    /// Build the summarization prompt for `content`. Mirrors the prompt string
    /// used in both `_generate_summary` and `_generate_summary_async`.
    fn build_summary_prompt(&self, content: &str) -> String {
        format!(
            "Summarize the following agent conversation turns concisely. This summary will replace these turns in the conversation history.\n\nWrite the summary from a neutral perspective describing what the assistant did and learned. Include:\n1. What actions the assistant took (tool calls, searches, file operations)\n2. Key information or results obtained\n3. Any important decisions or findings\n4. Relevant data, file names, values, or outputs\n\nKeep the summary factual and informative. Target approximately {} tokens.\n\n---\nTURNS TO SUMMARIZE:\n{content}\n---\n\nWrite only the summary, starting with \"[CONTEXT SUMMARY]:\" prefix.",
            self.config.summary_target_tokens
        )
    }

    /// Generate a summary of the compressed turns. Mirrors `_generate_summary`
    /// including retry/backoff and the failure fallback. Sleeps between retries
    /// using the same `jittered_backoff` tuning (base_delay = retry_delay,
    /// max_delay = 30.0).
    pub fn generate_summary(&self, content: &str, metrics: &mut TrajectoryMetrics) -> String {
        let prompt = self.build_summary_prompt(content);
        let max_retries = self.config.max_retries.max(0);

        for attempt in 0..max_retries {
            metrics.summarization_api_calls += 1;
            let summary_temperature = effective_temperature_for_model(
                &self.config.summarization_model,
                self.config.temperature,
                Some(&self.config.base_url),
            );

            let result = match self.summary_client {
                Some(client) => client.complete(
                    &self.config.summarization_model,
                    &prompt,
                    summary_temperature,
                    self.config.summary_target_tokens * 2,
                ),
                None => Err("no summary client configured".to_string()),
            };

            match result {
                Ok(content_str) => {
                    let summary = coerce_summary_content(&Value::String(content_str));
                    return ensure_summary_prefix(&summary);
                }
                Err(e) => {
                    metrics.summarization_errors += 1;
                    log::warn!("Summarization attempt {} failed: {e}", attempt + 1);
                    if attempt < max_retries - 1 {
                        let delay = jittered_backoff_with(
                            attempt + 1,
                            self.config.retry_delay as f64,
                            30.0,
                            crate::agent_retry::DEFAULT_JITTER_RATIO,
                        );
                        std::thread::sleep(Duration::from_secs_f64(delay));
                    } else {
                        return FALLBACK_SUMMARY.to_string();
                    }
                }
            }
        }

        // max_retries <= 0: nothing attempted, return fallback.
        FALLBACK_SUMMARY.to_string()
    }

    /// Compress a single trajectory to fit within the target token budget.
    /// Mirrors `compress_trajectory`. Returns `(compressed, metrics)`.
    pub fn compress_trajectory(&self, trajectory: &[Value]) -> (Vec<Value>, TrajectoryMetrics) {
        let mut metrics = TrajectoryMetrics::default();
        metrics.original_turns = trajectory.len() as i64;

        let turn_tokens = self.count_turn_tokens(trajectory);
        let total_tokens: i64 = turn_tokens.iter().sum();
        metrics.original_tokens = total_tokens;

        // Under target → skip.
        if total_tokens <= self.config.target_max_tokens {
            metrics.skipped_under_target = true;
            metrics.compressed_tokens = total_tokens;
            metrics.compressed_turns = trajectory.len() as i64;
            metrics.compression_ratio = 1.0;
            return (trajectory.to_vec(), metrics);
        }

        let (_protected, compress_start, compress_end) = self.find_protected_indices(trajectory);

        // Nothing to compress.
        if compress_start >= compress_end {
            metrics.compressed_tokens = total_tokens;
            metrics.compressed_turns = trajectory.len() as i64;
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens;
            return (trajectory.to_vec(), metrics);
        }

        let tokens_to_save = total_tokens - self.config.target_max_tokens;
        let target_tokens_to_compress = tokens_to_save + self.config.summary_target_tokens;

        // Accumulate turns from compress_start until enough savings.
        let mut accumulated_tokens: i64 = 0;
        let mut compress_until = compress_start;

        for i in compress_start..compress_end {
            accumulated_tokens += turn_tokens[i];
            compress_until = i + 1; // exclusive end
            if accumulated_tokens >= target_tokens_to_compress {
                break;
            }
        }

        // If still not enough, compress the entire compressible region.
        if accumulated_tokens < target_tokens_to_compress && compress_until < compress_end {
            compress_until = compress_end;
            accumulated_tokens = turn_tokens[compress_start..compress_end].iter().sum();
        }
        let _ = accumulated_tokens;

        metrics.turns_compressed_start_idx = compress_start as i64;
        metrics.turns_compressed_end_idx = compress_until as i64;
        metrics.turns_in_compressed_region = (compress_until - compress_start) as i64;

        let content_to_summarize =
            self.extract_turn_content_for_summary(trajectory, compress_start, compress_until);

        let summary = self.generate_summary(&content_to_summarize, &mut metrics);

        let compressed = self.assemble_compressed(trajectory, compress_start, compress_until, summary);

        metrics.compressed_turns = compressed.len() as i64;
        metrics.compressed_tokens = self.count_trajectory_tokens(&compressed);
        metrics.turns_removed = metrics.original_turns - metrics.compressed_turns;
        metrics.tokens_saved = metrics.original_tokens - metrics.compressed_tokens;
        metrics.compression_ratio =
            metrics.compressed_tokens as f64 / metrics.original_tokens.max(1) as f64;
        metrics.was_compressed = true;
        metrics.still_over_limit = metrics.compressed_tokens > self.config.target_max_tokens;

        (compressed, metrics)
    }

    /// Build the compressed trajectory: head (with optional system notice),
    /// the summary as a `human` turn, then the untouched tail. Shared by the
    /// sync/async Python paths.
    fn assemble_compressed(
        &self,
        trajectory: &[Value],
        compress_start: usize,
        compress_until: usize,
        summary: String,
    ) -> Vec<Value> {
        let mut compressed: Vec<Value> = Vec::new();

        // Head turns (before compression region).
        for turn in trajectory.iter().take(compress_start) {
            let mut turn = turn.clone();
            if turn_str(&turn, "from", "") == "system" && self.config.add_summary_notice {
                let new_val = format!(
                    "{}{}",
                    turn_str(&turn, "value", ""),
                    self.config.summary_notice_text
                );
                if let Some(obj) = turn.as_object_mut() {
                    obj.insert("value".to_string(), Value::String(new_val));
                }
            }
            compressed.push(turn);
        }

        // Summary as a human message.
        compressed.push(json!({ "from": "human", "value": summary }));

        // Tail turns (after compression region).
        for turn in trajectory.iter().skip(compress_until) {
            compressed.push(turn.clone());
        }

        compressed
    }

    /// Process a single JSONL entry. Mirrors `process_entry`. Returns
    /// `(result_entry, metrics)`. Entries without `conversations` pass through
    /// unchanged with default metrics.
    pub fn process_entry(&self, entry: &Value) -> (Value, TrajectoryMetrics) {
        let conversations = match entry.get("conversations").and_then(|v| v.as_array()) {
            Some(c) => c,
            None => return (entry.clone(), TrajectoryMetrics::default()),
        };

        let (compressed, metrics) = self.compress_trajectory(conversations);

        let mut result = entry.clone();
        if let Some(obj) = result.as_object_mut() {
            obj.insert(
                "conversations".to_string(),
                Value::Array(compressed),
            );
            if self.config.metrics_per_trajectory && metrics.was_compressed {
                obj.insert("compression_metrics".to_string(), metrics.to_dict());
            }
        }

        (result, metrics)
    }
}

// ── JSONL helpers ──────────────────────────────────────────────────────────────

/// Parse a JSONL string into entries, skipping blank lines and (with a warning)
/// invalid JSON lines. Mirrors the loop in `_process_directory_async`/`main`.
pub fn parse_jsonl(content: &str) -> Vec<Value> {
    let mut entries = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(v) => entries.push(v),
            Err(e) => {
                log::warn!("Skipping invalid JSON at line {line_num}: {e}");
            }
        }
    }
    entries
}

/// Serialize entries to a JSONL string (one compact object per line, trailing
/// newline after each). Equivalent to `json.dumps(entry, ensure_ascii=False)`
/// per line.
pub fn entries_to_jsonl(entries: &[Value]) -> String {
    let mut out = String::new();
    for entry in entries {
        out.push_str(&serde_json::to_string(entry).unwrap_or_else(|_| "{}".to_string()));
        out.push('\n');
    }
    out
}

/// Compute the default output path for single-file input. Mirrors the Python:
/// `input.parent / (input.stem + output_suffix + ".jsonl")`.
pub fn default_file_output_path(input_path: &Path, output_suffix: &str) -> PathBuf {
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    parent.join(format!("{stem}{output_suffix}.jsonl"))
}

/// Compute the default output path for directory input. Mirrors the Python:
/// `input.parent / (input.name + output_suffix)`.
pub fn default_dir_output_path(input_path: &Path, output_suffix: &str) -> PathBuf {
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let name = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    parent.join(format!("{name}{output_suffix}"))
}

// ── Serde mirrors for config (optional convenience) ──────────────────────────

/// Optional serde-friendly snapshot of [`CompressionConfig`] for callers that
/// want to persist or transmit it as JSON. Not used by the core logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfigSnapshot {
    pub target_max_tokens: i64,
    pub summary_target_tokens: i64,
    pub summarization_model: String,
    pub base_url: String,
    pub protect_last_n_turns: i64,
}

impl From<&CompressionConfig> for CompressionConfigSnapshot {
    fn from(c: &CompressionConfig) -> Self {
        Self {
            target_max_tokens: c.target_max_tokens,
            summary_target_tokens: c.summary_target_tokens,
            summarization_model: c.summarization_model.clone(),
            base_url: c.base_url.clone(),
            protect_last_n_turns: c.protect_last_n_turns,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A token counter that returns one token per whitespace-delimited word,
    /// letting tests control token budgets precisely.
    struct WordCounter;
    impl TokenCounter for WordCounter {
        fn count(&self, text: &str) -> i64 {
            if text.is_empty() {
                return 0;
            }
            text.split_whitespace().count() as i64
        }
    }

    /// A summary client that returns a fixed short summary (one "word" per the
    /// WordCounter), so compressed totals are deterministic.
    struct StubSummary;
    impl SummaryClient for StubSummary {
        fn complete(
            &self,
            _model: &str,
            _prompt: &str,
            _temperature: Option<f64>,
            _max_tokens: i64,
        ) -> Result<String, String> {
            Ok("summarized".to_string())
        }
    }

    fn turn(from: &str, value: &str) -> Value {
        json!({ "from": from, "value": value })
    }

    fn words(n: usize) -> String {
        vec!["w"; n].join(" ")
    }

    #[test]
    fn heuristic_counter_matches_python_fallback() {
        let c = HeuristicTokenCounter;
        assert_eq!(c.count(""), 0);
        assert_eq!(c.count("abcd"), 1);
        assert_eq!(c.count("abcdefgh"), 2);
        assert_eq!(c.count("abcde"), 1); // 5 // 4
    }

    #[test]
    fn ensure_prefix_idempotent() {
        assert_eq!(ensure_summary_prefix(""), "[CONTEXT SUMMARY]:");
        assert_eq!(ensure_summary_prefix("  "), "[CONTEXT SUMMARY]:");
        assert_eq!(ensure_summary_prefix("hello"), "[CONTEXT SUMMARY]: hello");
        assert_eq!(
            ensure_summary_prefix("[CONTEXT SUMMARY]: hi"),
            "[CONTEXT SUMMARY]: hi"
        );
        assert_eq!(
            ensure_summary_prefix("  [CONTEXT SUMMARY]: hi  "),
            "[CONTEXT SUMMARY]: hi"
        );
    }

    #[test]
    fn coerce_summary_handles_non_strings() {
        assert_eq!(coerce_summary_content(&Value::Null), "");
        assert_eq!(coerce_summary_content(&json!("  hi  ")), "hi");
        assert_eq!(coerce_summary_content(&json!(42)), "42");
    }

    #[test]
    fn detect_provider_known_and_unknown() {
        assert_eq!(detect_provider("https://openrouter.ai/api/v1"), "openrouter");
        assert_eq!(detect_provider("https://z.ai/api"), "zai");
        assert_eq!(detect_provider("https://api.moonshot.ai/v1"), "kimi-coding");
        assert_eq!(detect_provider("https://example.com/v1"), "");
        assert_eq!(
            detect_provider("https://chatgpt.com/backend-api/codex"),
            "codex"
        );
    }

    #[test]
    fn effective_temperature_kimi_omits() {
        // Kimi → omit temperature entirely.
        assert_eq!(
            effective_temperature_for_model("moonshotai/kimi-k2", 0.3, None),
            None
        );
        // Ordinary model → requested temperature passes through.
        assert_eq!(
            effective_temperature_for_model("google/gemini-3-flash-preview", 0.3, None),
            Some(0.3)
        );
    }

    #[test]
    fn skip_under_target_returns_unchanged() {
        let counter = WordCounter;
        let summary = StubSummary;
        let mut config = CompressionConfig::default();
        config.target_max_tokens = 1000;
        let comp = TrajectoryCompressor::new(config, &counter, Some(&summary));

        let traj = vec![turn("system", &words(5)), turn("human", &words(5))];
        let (out, m) = comp.compress_trajectory(&traj);
        assert!(m.skipped_under_target);
        assert!(!m.was_compressed);
        assert_eq!(out.len(), 2);
        assert_eq!(m.compression_ratio, 1.0);
        assert_eq!(m.original_tokens, 10);
    }

    #[test]
    fn protected_indices_head_and_tail() {
        let counter = WordCounter;
        let summary = StubSummary;
        let mut config = CompressionConfig::default();
        config.protect_last_n_turns = 2;
        let comp = TrajectoryCompressor::new(config, &counter, Some(&summary));

        // 8 turns: system, human, gpt, tool, gpt, tool, gpt, tool
        let traj = vec![
            turn("system", "s"),
            turn("human", "h"),
            turn("gpt", "g"),
            turn("tool", "t"),
            turn("gpt", "g"),
            turn("tool", "t"),
            turn("gpt", "g"),
            turn("tool", "t"),
        ];
        let (protected, start, end) = comp.find_protected_indices(&traj);
        // First system(0), human(1), gpt(2), tool(3) protected; last 2 → 6,7.
        assert!(protected.contains(&0));
        assert!(protected.contains(&1));
        assert!(protected.contains(&2));
        assert!(protected.contains(&3));
        assert!(protected.contains(&6));
        assert!(protected.contains(&7));
        // head protected max = 3 → start = 4; tail min = 6 → end = 6.
        assert_eq!(start, 4);
        assert_eq!(end, 6);
    }

    #[test]
    fn compress_replaces_middle_with_summary() {
        let counter = WordCounter;
        let summary = StubSummary;
        let mut config = CompressionConfig::default();
        config.target_max_tokens = 20;
        config.summary_target_tokens = 1;
        config.protect_last_n_turns = 2;
        config.add_summary_notice = false;
        let comp = TrajectoryCompressor::new(config, &counter, Some(&summary));

        // 8 turns each big enough to need compression.
        let traj = vec![
            turn("system", &words(10)),
            turn("human", &words(10)),
            turn("gpt", &words(10)),
            turn("tool", &words(10)),
            turn("gpt", &words(30)),
            turn("tool", &words(30)),
            turn("gpt", &words(10)),
            turn("tool", &words(10)),
        ];
        let (out, m) = comp.compress_trajectory(&traj);
        assert!(m.was_compressed);
        // One of the middle turns becomes a single human summary turn.
        let summary_turn = out
            .iter()
            .find(|t| turn_str(t, "value", "").starts_with("[CONTEXT SUMMARY]:"));
        assert!(summary_turn.is_some());
        assert_eq!(turn_str(summary_turn.unwrap(), "from", ""), "human");
        assert!(m.compressed_turns < m.original_turns);
        assert!(m.tokens_saved > 0);
    }

    #[test]
    fn summary_notice_appended_to_system() {
        let counter = WordCounter;
        let summary = StubSummary;
        let mut config = CompressionConfig::default();
        config.target_max_tokens = 20;
        config.summary_target_tokens = 1;
        config.protect_last_n_turns = 2;
        config.add_summary_notice = true;
        config.protect_first_system = false; // so system can be in head but...
        // Keep system in head region by structure.
        let comp = TrajectoryCompressor::new(config, &counter, Some(&summary));

        let traj = vec![
            turn("system", "SYS"),
            turn("human", &words(10)),
            turn("gpt", &words(30)),
            turn("tool", &words(30)),
            turn("gpt", &words(10)),
            turn("tool", &words(10)),
        ];
        let (out, m) = comp.compress_trajectory(&traj);
        assert!(m.was_compressed);
        // The head's system turn (index 0) should carry the notice.
        let sys = &out[0];
        if turn_str(sys, "from", "") == "system" {
            assert!(turn_str(sys, "value", "").contains("summarized to preserve context"));
        }
    }

    #[test]
    fn generate_summary_fallback_without_client() {
        let counter = WordCounter;
        let mut config = CompressionConfig::default();
        config.max_retries = 1;
        let comp = TrajectoryCompressor::new(config, &counter, None);
        let mut m = TrajectoryMetrics::default();
        let s = comp.generate_summary("content", &mut m);
        assert_eq!(s, FALLBACK_SUMMARY);
        assert_eq!(m.summarization_api_calls, 1);
        assert_eq!(m.summarization_errors, 1);
    }

    #[test]
    fn process_entry_without_conversations_passthrough() {
        let counter = WordCounter;
        let summary = StubSummary;
        let comp = TrajectoryCompressor::new(CompressionConfig::default(), &counter, Some(&summary));
        let entry = json!({ "id": 1, "foo": "bar" });
        let (out, m) = comp.process_entry(&entry);
        assert_eq!(out, entry);
        assert!(!m.was_compressed);
    }

    #[test]
    fn process_entry_adds_metrics_when_compressed() {
        let counter = WordCounter;
        let summary = StubSummary;
        let mut config = CompressionConfig::default();
        config.target_max_tokens = 20;
        config.summary_target_tokens = 1;
        config.protect_last_n_turns = 2;
        let comp = TrajectoryCompressor::new(config, &counter, Some(&summary));

        let entry = json!({
            "conversations": [
                turn("system", &words(10)),
                turn("human", &words(10)),
                turn("gpt", &words(30)),
                turn("tool", &words(30)),
                turn("gpt", &words(10)),
                turn("tool", &words(10)),
            ]
        });
        let (out, m) = comp.process_entry(&entry);
        assert!(m.was_compressed);
        assert!(out.get("compression_metrics").is_some());
    }

    #[test]
    fn aggregate_metrics_to_dict_shape() {
        let mut agg = AggregateMetrics::default();
        let mut tm = TrajectoryMetrics::default();
        tm.was_compressed = true;
        tm.original_tokens = 100;
        tm.compressed_tokens = 40;
        tm.tokens_saved = 60;
        tm.compression_ratio = 0.4;
        tm.summarization_api_calls = 1;
        agg.add_trajectory_metrics(&tm);

        let d = agg.to_dict();
        assert_eq!(d["summary"]["total_trajectories"], json!(1));
        assert_eq!(d["summary"]["trajectories_compressed"], json!(1));
        assert_eq!(d["tokens"]["total_saved"], json!(60));
        assert_eq!(d["summarization"]["total_api_calls"], json!(1));
        assert_eq!(d["summarization"]["success_rate"], json!(1.0));
    }

    #[test]
    fn config_from_yaml_value_maps_fields() {
        let data = json!({
            "compression": { "target_max_tokens": 8000, "summary_target_tokens": 500 },
            "protected_turns": { "last_n_turns": 6, "first_gpt": false },
            "summarization": { "model": "x/y", "base_url": "", "temperature": 0.7 },
            "output": { "output_suffix": "_z" },
            "metrics": { "enabled": false },
        });
        let c = CompressionConfig::from_yaml_value(&data);
        assert_eq!(c.target_max_tokens, 8000);
        assert_eq!(c.summary_target_tokens, 500);
        assert_eq!(c.protect_last_n_turns, 6);
        assert!(!c.protect_first_gpt);
        assert_eq!(c.summarization_model, "x/y");
        // base_url empty → keep default (or-semantics).
        assert_eq!(c.base_url, OPENROUTER_BASE_URL);
        assert_eq!(c.temperature, 0.7);
        assert_eq!(c.output_suffix, "_z");
        assert!(!c.metrics_enabled);
    }

    #[test]
    fn jsonl_roundtrip() {
        let content = "{\"a\":1}\n\n  \nnot json\n{\"b\":2}\n";
        let entries = parse_jsonl(content);
        assert_eq!(entries.len(), 2);
        let out = entries_to_jsonl(&entries);
        assert!(out.contains("\"a\":1"));
        assert!(out.ends_with('\n'));
    }

    #[test]
    fn default_output_paths() {
        let f = default_file_output_path(Path::new("/data/traj.jsonl"), "_compressed");
        assert_eq!(f, PathBuf::from("/data/traj_compressed.jsonl"));
        let d = default_dir_output_path(Path::new("/data/my_run"), "_compressed");
        assert_eq!(d, PathBuf::from("/data/my_run_compressed"));
    }

    #[test]
    fn truncation_of_long_turn_values() {
        let counter = WordCounter;
        let summary = StubSummary;
        let comp = TrajectoryCompressor::new(CompressionConfig::default(), &counter, Some(&summary));
        let long = "x".repeat(5000);
        let traj = vec![turn("tool", &long)];
        let s = comp.extract_turn_content_for_summary(&traj, 0, 1);
        assert!(s.contains("...[truncated]..."));
        assert!(s.starts_with("[Turn 0 - TOOL]:"));
    }
}
