//! YC-Bench Long-Horizon Agent Benchmark Environment (native Rust port).
//!
//! Port of `environments/benchmarks/yc_bench/yc_bench_env.py`.
//!
//! Evaluates agentic LLMs on YC-Bench: a deterministic, long-horizon benchmark
//! where the agent acts as CEO of an AI startup over a simulated 1-3 year run.
//! The agent manages cash flow, employees, tasks, and prestige across 4 domains,
//! interacting exclusively via CLI subprocess calls against a SQLite-backed
//! discrete-event simulation.
//!
//! This is an eval-only environment. The Python flow drives a `HermesAgentLoop`
//! against the `yc-bench` CLI, then reads the resulting SQLite DB to extract a
//! score. The framework-bound async agent loop (`HermesAgentLoop`, atroposlib
//! base env) is not part of hermes-core; this port reproduces the deterministic
//! portable logic faithfully:
//!   * configuration (`YcBenchEvalConfig`)
//!   * preset-horizon resolution
//!   * eval-matrix construction (preset x seed)
//!   * prompt formatting
//!   * CLI command construction (`sim init`, CLI availability check)
//!   * SQLite final-score reading (`read_final_score`)
//!   * composite scoring (`compute_composite_score`)
//!   * per-preset / overall metrics aggregation
//!   * streaming JSONL result persistence
//!
//! The async agent driving and atroposlib server/wandb plumbing are left as
//! integration concerns for the caller.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;
use std::process::Command;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// =============================================================================
// System prompt
// =============================================================================

/// System prompt presented to the YC-Bench CEO agent.
pub const YC_BENCH_SYSTEM_PROMPT: &str = "You are the autonomous CEO of an early-stage AI startup in a deterministic
business simulation. You manage the company exclusively through the `yc-bench`
CLI tool. Your primary goal is to **survive** until the simulation horizon ends
without going bankrupt, while **maximising final funds**.

## Simulation Mechanics

- **Funds**: You start with $250,000 seed capital. Revenue comes from completing
  tasks. Rewards scale with your prestige: `base × (1 + scale × (prestige − 1))`.
- **Domains**: There are 4 skill domains: **research**, **inference**,
  **data_environment**, and **training**. Each has its own prestige level
  (1.0-10.0). Higher prestige unlocks better-paying tasks.
- **Employees**: You have employees (Junior/Mid/Senior) with domain-specific
  skill rates. **Throughput splits**: `effective_rate = base_rate / N` where N
  is the number of active tasks assigned to that employee. Focus beats breadth.
- **Payroll**: Deducted automatically on the first business day of each month.
  Running out of funds = bankruptcy = game over.
- **Time**: The simulation runs on business days (Mon-Fri), 09:00-18:00.
  Time only advances when you call `yc-bench sim resume`.

## Task Lifecycle

1. Browse market tasks with `market browse`
2. Accept a task with `task accept` (this sets its deadline)
3. Assign employees with `task assign`
4. Dispatch with `task dispatch` to start work
5. Call `sim resume` to advance time and let employees make progress
6. Tasks complete when all domain requirements are fulfilled

**Penalties for failure vary by difficulty preset.** Completing a task on time
earns full reward + prestige gain. Missing a deadline or cancelling a task
incurs prestige penalties -- cancelling is always more costly than letting a
task fail, so cancel only as a last resort.

## CLI Commands

### Observe
- `yc-bench company status`                                         -- funds, prestige, runway
- `yc-bench employee list`                                          -- skills, salary, active tasks
- `yc-bench market browse [--domain D] [--required-prestige-lte N]` -- available tasks
- `yc-bench task list [--status active|planned]`                    -- your tasks
- `yc-bench task inspect --task-id UUID`                            -- progress, deadline, assignments
- `yc-bench finance ledger [--category monthly_payroll|task_reward]` -- transaction history
- `yc-bench report monthly`                                         -- monthly P&L

### Act
- `yc-bench task accept --task-id UUID`                              -- accept from market
- `yc-bench task assign --task-id UUID --employee-id UUID`           -- assign employee
- `yc-bench task dispatch --task-id UUID`                            -- start work (needs >=1 assignment)
- `yc-bench task cancel --task-id UUID --reason \"text\"`              -- cancel (prestige penalty)
- `yc-bench sim resume`                                              -- advance simulation clock

### Memory (persists across context truncation)
- `yc-bench scratchpad read`            -- read your persistent notes
- `yc-bench scratchpad write --content \"text\"`  -- overwrite notes
- `yc-bench scratchpad append --content \"text\"` -- append to notes
- `yc-bench scratchpad clear`           -- clear notes

## Strategy Guidelines

1. **Specialise in 2-3 domains** to climb the prestige ladder faster and unlock
   high-reward tasks. Don't spread thin across all 4 domains early on.
2. **Focus employees** -- assigning one employee to many tasks halves their
   throughput per additional task. Keep assignments concentrated.
3. **Use the scratchpad** to track your strategy, upcoming deadlines, and
   employee assignments. This persists even if conversation context is truncated.
4. **Monitor runway** -- always know how many months of payroll you can cover.
   Accept high-reward tasks before payroll dates.
5. **Don't over-accept** -- taking too many tasks and missing deadlines cascades
   into prestige loss, locking you out of profitable contracts.
6. Use `finance ledger` and `report monthly` to track revenue trends.

## Your Turn

Each turn:
1. Call `yc-bench company status` and `yc-bench task list` to orient yourself.
2. Check for completed tasks and pending deadlines.
3. Browse market for profitable tasks within your prestige level.
4. Accept, assign, and dispatch tasks strategically.
5. Call `yc-bench sim resume` to advance time.
6. Repeat until the simulation ends.

Think step by step before acting.";

/// Starting funds in cents ($250,000).
pub const INITIAL_FUNDS_CENTS: i64 = 25_000_000;

/// Default horizon (in years) per preset name.
///
/// Mirrors `_PRESET_HORIZONS` in the Python source. Falls back to 1 when the
/// preset is unknown.
pub fn preset_horizon(preset: &str) -> i64 {
    match preset {
        "tutorial" => 1,
        "easy" => 1,
        "medium" => 1,
        "hard" => 1,
        "nightmare" => 1,
        "fast_test" => 1,
        "default" => 3,
        "high_reward" => 1,
        _ => 1,
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the YC-Bench evaluation environment.
///
/// Mirrors `YCBenchEvalConfig` (the YC-Bench-specific fields layered on top of
/// `HermesAgentEnvConfig`). Only the fields actually consumed by the portable
/// logic are modelled; the broader agent-loop config lives in the framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YcBenchEvalConfig {
    /// YC-Bench preset names to evaluate.
    pub presets: Vec<String>,
    /// Random seeds -- each preset x seed = one run.
    pub seeds: Vec<i64>,
    /// Maximum wall-clock seconds per run. Default 60 minutes.
    pub run_timeout: u64,
    /// Weight of survival (0/1) in composite score.
    pub survival_weight: f64,
    /// Weight of normalised final funds in composite score.
    pub funds_weight: f64,
    /// Directory for per-run SQLite databases.
    pub db_dir: String,
    /// Simulation horizon in years. If `None`, inferred from preset name.
    pub horizon_years: Option<i64>,
    /// Name of the simulated company.
    pub company_name: String,
    /// Simulation start date in MM/DD/YYYY format (yc-bench convention).
    pub start_date: String,
    /// Max agent turns per run.
    pub max_agent_turns: u32,
    /// Max token length.
    pub max_token_length: u32,
    /// Agent sampling temperature.
    pub agent_temperature: f64,
}

impl Default for YcBenchEvalConfig {
    fn default() -> Self {
        // Matches the values used in `config_init()` of the Python source.
        Self {
            presets: vec![
                "fast_test".to_string(),
                "medium".to_string(),
                "hard".to_string(),
            ],
            seeds: vec![1, 2, 3],
            run_timeout: 3600,
            survival_weight: 0.5,
            funds_weight: 0.5,
            db_dir: "/tmp/yc_bench_dbs".to_string(),
            horizon_years: None,
            company_name: "BenchCo".to_string(),
            start_date: "01/01/2025".to_string(),
            max_agent_turns: 200,
            max_token_length: 32000,
            agent_temperature: 0.0,
        }
    }
}

impl YcBenchEvalConfig {
    /// Resolve the simulation horizon for a preset.
    ///
    /// Precedence: explicit config override > preset lookup > default 1.
    /// Mirrors `self.config.horizon_years or _PRESET_HORIZONS.get(preset, 1)`.
    pub fn resolve_horizon(&self, preset: &str) -> i64 {
        // Python treats `horizon_years == 0` as falsy and falls through to the
        // preset lookup; replicate that truthiness behavior.
        match self.horizon_years {
            Some(h) if h != 0 => h,
            _ => preset_horizon(preset),
        }
    }
}

/// A single eval-matrix item: one (preset, seed) pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalItem {
    pub preset: String,
    pub seed: i64,
}

/// Build the eval matrix as the cartesian product of presets x seeds.
///
/// Iteration order matches Python: preset-outer, seed-inner.
pub fn build_eval_matrix(presets: &[String], seeds: &[i64]) -> Vec<EvalItem> {
    let mut out = Vec::with_capacity(presets.len() * seeds.len());
    for preset in presets {
        for &seed in seeds {
            out.push(EvalItem {
                preset: preset.clone(),
                seed,
            });
        }
    }
    out
}

// =============================================================================
// Prompt formatting
// =============================================================================

/// Format the per-run user prompt for an eval item.
///
/// Mirrors `YCBenchEvalEnv.format_prompt`.
pub fn format_prompt(item: &EvalItem, company_name: &str) -> String {
    format!(
        "A new YC-Bench simulation has been initialized \
(preset='{preset}', seed={seed}).\n\
Your company '{company}' is ready.\n\n\
Begin by calling:\n\
1. `yc-bench company status` -- see your starting funds and prestige\n\
2. `yc-bench employee list` -- see your team and their skills\n\
3. `yc-bench market browse --required-prestige-lte 1` -- find tasks \
you can take\n\n\
Then accept 2-3 tasks, assign employees, dispatch them, and call \
`yc-bench sim resume` to advance time. Repeat this loop until the \
simulation ends (horizon reached or bankruptcy).",
        preset = item.preset,
        seed = item.seed,
        company = company_name,
    )
}

// =============================================================================
// CLI command construction
// =============================================================================

/// Build the `yc-bench sim init` argument vector for a run.
///
/// IMPORTANT (preserved from Python): we use `sim init`, NOT `yc-bench run`.
/// `yc-bench run` starts yc-bench's own LLM agent loop, which would compete
/// with the driving agent. `sim init` just sets up the world and returns.
pub fn sim_init_args(seed: i64, start_date: &str, company_name: &str, horizon_years: i64) -> Vec<String> {
    vec![
        "sim".to_string(),
        "init".to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--start-date".to_string(),
        start_date.to_string(),
        "--company-name".to_string(),
        company_name.to_string(),
        "--horizon-years".to_string(),
        horizon_years.to_string(),
    ]
}

/// Build the SQLite `DATABASE_URL` value for a run DB path.
///
/// Mirrors `os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"`.
pub fn database_url(db_path: &str) -> String {
    format!("sqlite:///{db_path}")
}

/// Build the per-run isolated DB path inside `db_dir`.
///
/// Mirrors `os.path.join(self.config.db_dir, f"yc_bench_{run_key}.db")`.
pub fn run_db_path(db_dir: &str, run_key: &str) -> String {
    Path::new(db_dir)
        .join(format!("yc_bench_{run_key}.db"))
        .to_string_lossy()
        .into_owned()
}

/// Verify the `yc-bench` CLI is installed and runnable.
///
/// Mirrors `setup()`'s `subprocess.run(["yc-bench", "--help"], timeout=10)`
/// check. Returns `Ok(())` if `yc-bench --help` exits 0, otherwise an
/// installation-hint error string (matching the Python `RuntimeError` text).
pub fn verify_cli_available() -> Result<(), String> {
    let install_hint = "yc-bench CLI not found. Install with:\n  \
        pip install \"hermes-agent[yc-bench]\"\n\
        Or: git clone https://github.com/collinear-ai/yc-bench \
        && cd yc-bench && pip install -e .";

    match Command::new("yc-bench").arg("--help").output() {
        Ok(out) if out.status.success() => Ok(()),
        _ => Err(install_hint.to_string()),
    }
}

/// Outcome of running `yc-bench sim init`.
#[derive(Debug, Clone)]
pub struct SimInitResult {
    pub success: bool,
    pub error_message: String,
}

/// Run `yc-bench sim init` for a run, with `DATABASE_URL`/`YC_BENCH_EXPERIMENT`
/// set in the child process environment (rather than mutating the parent's
/// process env as the Python code does -- this is the cleaner equivalent).
///
/// Mirrors the `subprocess.run(init_cmd, ...)` invocation and its error
/// extraction `(stderr or stdout).strip()`.
pub fn run_sim_init(
    db_path: &str,
    preset: &str,
    seed: i64,
    start_date: &str,
    company_name: &str,
    horizon_years: i64,
) -> SimInitResult {
    let args = sim_init_args(seed, start_date, company_name, horizon_years);
    let output = Command::new("yc-bench")
        .args(&args)
        .env("DATABASE_URL", database_url(db_path))
        .env("YC_BENCH_EXPERIMENT", preset)
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                SimInitResult {
                    success: true,
                    error_message: String::new(),
                }
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                // Python: (stderr or stdout).strip()
                let msg = if !stderr.trim().is_empty() {
                    stderr.trim().to_string()
                } else {
                    stdout.trim().to_string()
                };
                SimInitResult {
                    success: false,
                    error_message: format!("yc-bench sim init failed: {msg}"),
                }
            }
        }
        Err(e) => SimInitResult {
            success: false,
            error_message: format!("yc-bench sim init failed: {e}"),
        },
    }
}

// =============================================================================
// Scoring helpers
// =============================================================================

/// Final game state read from a YC-Bench SQLite database.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalScore {
    pub final_funds_cents: i64,
    pub survived: bool,
    pub terminal_reason: String,
}

/// Read final game state from a YC-Bench SQLite database.
///
/// Mirrors `_read_final_score`. Returns `final_funds_cents`, `survived`, and
/// `terminal_reason`.
///
/// Note: yc-bench table names are plural -- `companies` not `company`,
/// `sim_events` not `simulation_log`.
pub fn read_final_score(db_path: &str) -> FinalScore {
    if !Path::new(db_path).exists() {
        log::warn!("DB not found at {db_path}");
        return FinalScore {
            final_funds_cents: 0,
            survived: false,
            terminal_reason: "db_missing".to_string(),
        };
    }

    let conn = match Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to read DB {db_path}: {e}");
            return FinalScore {
                final_funds_cents: 0,
                survived: false,
                terminal_reason: format!("db_error: {e}"),
            };
        }
    };

    // Read final funds from the 'companies' table.
    let funds: i64 = match conn.query_row("SELECT funds_cents FROM companies LIMIT 1", [], |row| {
        row.get::<_, i64>(0)
    }) {
        Ok(v) => v,
        Err(rusqlite::Error::QueryReturnedNoRows) => 0,
        Err(e) => {
            // Any other read failure mirrors the broad `except Exception`.
            log::error!("Failed to read DB {db_path}: {e}");
            return FinalScore {
                final_funds_cents: 0,
                survived: false,
                terminal_reason: format!("db_error: {e}"),
            };
        }
    };

    // Determine terminal reason from 'sim_events' table.
    // The Python code swallows `sqlite3.OperationalError` (e.g. missing table)
    // and keeps `terminal_reason = "unknown"`.
    let mut terminal_reason = "unknown".to_string();
    let event_query = "SELECT event_type FROM sim_events \
        WHERE event_type IN ('bankruptcy', 'horizon_end') \
        ORDER BY scheduled_at DESC LIMIT 1";
    match conn.query_row(event_query, [], |row| row.get::<_, String>(0)) {
        Ok(et) => terminal_reason = et,
        Err(rusqlite::Error::QueryReturnedNoRows) => {}
        Err(_) => {
            // Table may not exist if simulation didn't progress -> leave "unknown".
        }
    }

    let survived = funds >= 0 && terminal_reason != "bankruptcy";
    FinalScore {
        final_funds_cents: funds,
        survived,
        terminal_reason,
    }
}

/// Compute composite score from survival and final funds.
///
/// Mirrors `_compute_composite_score`.
///
/// ```text
/// Score = survival_weight * survival_score
///       + funds_weight * normalised_funds_score
/// ```
///
/// Normalised funds uses log-scale relative to initial capital:
/// - funds <= 0:          0.0
/// - funds == initial:   ~0.15
/// - funds == 10x:       ~0.52
/// - funds == 100x:       1.0
pub fn compute_composite_score(
    final_funds_cents: i64,
    survived: bool,
    survival_weight: f64,
    funds_weight: f64,
    initial_funds_cents: i64,
) -> f64 {
    let survival_score = if survived { 1.0 } else { 0.0 };

    let funds_score = if final_funds_cents <= 0 {
        0.0
    } else {
        let max_ratio = 100.0_f64;
        let denom = std::cmp::max(initial_funds_cents, 1) as f64;
        let ratio = final_funds_cents as f64 / denom;
        (ratio.ln_1p() / max_ratio.ln_1p()).min(1.0)
    };

    survival_weight * survival_score + funds_weight * funds_score
}

/// Convenience wrapper using the default initial funds.
pub fn compute_composite_score_default(
    final_funds_cents: i64,
    survived: bool,
    survival_weight: f64,
    funds_weight: f64,
) -> f64 {
    compute_composite_score(
        final_funds_cents,
        survived,
        survival_weight,
        funds_weight,
        INITIAL_FUNDS_CENTS,
    )
}

// =============================================================================
// Run results & metrics aggregation
// =============================================================================

/// Result of a single (preset, seed) run.
///
/// Mirrors the `out` dict assembled in `rollout_and_score_eval` (and its
/// error/timeout variants). `messages` is intentionally a free-form JSON value
/// so callers can stash the agent transcript when available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub preset: String,
    pub seed: i64,
    pub survived: bool,
    pub final_funds_cents: i64,
    pub final_funds_usd: f64,
    pub terminal_reason: String,
    pub composite_score: f64,
    pub turns_used: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_naturally: Option<bool>,
    pub elapsed_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl RunResult {
    /// Build the success-path result (mirrors the main `out` dict).
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        preset: &str,
        seed: i64,
        score: &FinalScore,
        composite: f64,
        turns_used: u32,
        finished_naturally: bool,
        elapsed_seconds: f64,
        db_path: &str,
    ) -> Self {
        Self {
            preset: preset.to_string(),
            seed,
            survived: score.survived,
            final_funds_cents: score.final_funds_cents,
            final_funds_usd: score.final_funds_cents as f64 / 100.0,
            terminal_reason: score.terminal_reason.clone(),
            composite_score: composite,
            turns_used,
            finished_naturally: Some(finished_naturally),
            elapsed_seconds,
            db_path: Some(db_path.to_string()),
            error: None,
        }
    }

    /// Build an error-path result (mirrors the `except` branch `out` dict).
    pub fn error(preset: &str, seed: i64, error: &str, elapsed_seconds: f64) -> Self {
        Self {
            preset: preset.to_string(),
            seed,
            survived: false,
            final_funds_cents: 0,
            final_funds_usd: 0.0,
            terminal_reason: format!("error: {error}"),
            composite_score: 0.0,
            turns_used: 0,
            finished_naturally: None,
            elapsed_seconds,
            db_path: None,
            error: Some(error.to_string()),
        }
    }

    /// Build a timeout-path result (mirrors `_run_with_timeout`'s `out` dict).
    pub fn timeout(preset: &str, seed: i64, run_timeout: u64) -> Self {
        Self {
            preset: preset.to_string(),
            seed,
            survived: false,
            final_funds_cents: 0,
            final_funds_usd: 0.0,
            terminal_reason: format!("timeout ({run_timeout}s)"),
            composite_score: 0.0,
            turns_used: 0,
            finished_naturally: None,
            elapsed_seconds: 0.0,
            db_path: None,
            error: Some("timeout".to_string()),
        }
    }
}

/// Aggregated evaluation metrics over all runs.
///
/// Mirrors the `eval_metrics` dict assembled in `evaluate()`, including the
/// per-preset breakdown keys. Stored in a `BTreeMap` to keep deterministic
/// (sorted) key ordering, matching the `sorted(preset_results.items())` loops.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub metrics: BTreeMap<String, f64>,
}

impl EvalMetrics {
    /// Compute aggregate metrics from a slice of run results.
    ///
    /// Mirrors the metric computation in `evaluate()`:
    ///   * `eval/survival_rate`, `eval/avg_composite_score`,
    ///     `eval/total_runs`, `eval/survived_runs`,
    ///     `eval/evaluation_time_seconds`
    ///   * per-preset `eval/survival_rate_<key>` and `eval/avg_score_<key>`,
    ///     where `<key>` is the preset with `-` replaced by `_`.
    pub fn compute(results: &[RunResult], evaluation_time_seconds: f64) -> Self {
        let mut metrics = BTreeMap::new();
        let total = results.len();
        if total == 0 {
            return Self { metrics };
        }

        let survived_total = results.iter().filter(|r| r.survived).count();
        let survival_rate = survived_total as f64 / total as f64;
        let avg_score =
            results.iter().map(|r| r.composite_score).sum::<f64>() / total as f64;

        metrics.insert("eval/survival_rate".to_string(), survival_rate);
        metrics.insert("eval/avg_composite_score".to_string(), avg_score);
        metrics.insert("eval/total_runs".to_string(), total as f64);
        metrics.insert("eval/survived_runs".to_string(), survived_total as f64);
        metrics.insert(
            "eval/evaluation_time_seconds".to_string(),
            evaluation_time_seconds,
        );

        // Per-preset breakdown (BTreeMap keeps presets sorted, like Python's
        // `sorted(preset_results.items())`).
        let mut preset_results: BTreeMap<&str, Vec<&RunResult>> = BTreeMap::new();
        for r in results {
            preset_results.entry(r.preset.as_str()).or_default().push(r);
        }

        for (preset, items) in &preset_results {
            let ps = items.iter().filter(|r| r.survived).count();
            let pt = items.len();
            let pa = if pt > 0 {
                items.iter().map(|r| r.composite_score).sum::<f64>() / pt as f64
            } else {
                0.0
            };
            let key = preset.replace('-', "_");
            let sr = if pt > 0 { ps as f64 / pt as f64 } else { 0.0 };
            metrics.insert(format!("eval/survival_rate_{key}"), sr);
            metrics.insert(format!("eval/avg_score_{key}"), pa);
        }

        Self { metrics }
    }
}

// =============================================================================
// Streaming JSONL result persistence
// =============================================================================

/// Crash-safe streaming JSONL logger for run results.
///
/// Mirrors the `_streaming_file` / `_save_result` machinery: each result is
/// serialised as a single JSON line and flushed immediately.
pub struct StreamingResultLog {
    path: String,
    file: Option<std::fs::File>,
}

impl StreamingResultLog {
    /// Open (truncating) a streaming JSONL log at `path`.
    pub fn open(path: &str) -> std::io::Result<Self> {
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self {
            path: path.to_string(),
            file: Some(file),
        })
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Write a single run result (mirrors `_save_result`).
    ///
    /// `ensure_ascii=False` -> serde_json already emits UTF-8 directly.
    pub fn save(&mut self, result: &RunResult) -> std::io::Result<()> {
        if let Some(file) = self.file.as_mut() {
            let line = serde_json::to_string(result)
                .unwrap_or_else(|e| json!({ "serialize_error": e.to_string() }).to_string());
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
        }
        Ok(())
    }

    /// Write an arbitrary JSON value as one line (used when a richer record,
    /// e.g. one carrying `messages`, must be persisted verbatim).
    pub fn save_value(&mut self, value: &Value) -> std::io::Result<()> {
        if let Some(file) = self.file.as_mut() {
            let line = serde_json::to_string(value)
                .unwrap_or_else(|e| json!({ "serialize_error": e.to_string() }).to_string());
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
        }
        Ok(())
    }

    /// Close the underlying file (idempotent).
    pub fn close(&mut self) {
        self.file = None;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_horizon() {
        assert_eq!(preset_horizon("default"), 3);
        assert_eq!(preset_horizon("medium"), 1);
        assert_eq!(preset_horizon("fast_test"), 1);
        // Unknown preset falls back to 1.
        assert_eq!(preset_horizon("does_not_exist"), 1);
    }

    #[test]
    fn test_resolve_horizon_precedence() {
        let mut cfg = YcBenchEvalConfig::default();
        // No override -> preset lookup.
        assert_eq!(cfg.resolve_horizon("default"), 3);
        assert_eq!(cfg.resolve_horizon("medium"), 1);
        // Explicit non-zero override wins.
        cfg.horizon_years = Some(5);
        assert_eq!(cfg.resolve_horizon("default"), 5);
        // Zero is falsy in Python -> falls through to preset.
        cfg.horizon_years = Some(0);
        assert_eq!(cfg.resolve_horizon("default"), 3);
    }

    #[test]
    fn test_build_eval_matrix_order() {
        let presets = vec!["a".to_string(), "b".to_string()];
        let seeds = vec![1, 2];
        let matrix = build_eval_matrix(&presets, &seeds);
        // preset-outer, seed-inner.
        assert_eq!(
            matrix,
            vec![
                EvalItem { preset: "a".into(), seed: 1 },
                EvalItem { preset: "a".into(), seed: 2 },
                EvalItem { preset: "b".into(), seed: 1 },
                EvalItem { preset: "b".into(), seed: 2 },
            ]
        );
    }

    #[test]
    fn test_format_prompt() {
        let item = EvalItem { preset: "hard".into(), seed: 7 };
        let p = format_prompt(&item, "BenchCo");
        assert!(p.contains("preset='hard', seed=7"));
        assert!(p.contains("Your company 'BenchCo' is ready."));
        assert!(p.contains("yc-bench company status"));
    }

    #[test]
    fn test_sim_init_args() {
        let args = sim_init_args(3, "01/01/2025", "BenchCo", 1);
        assert_eq!(
            args,
            vec![
                "sim", "init",
                "--seed", "3",
                "--start-date", "01/01/2025",
                "--company-name", "BenchCo",
                "--horizon-years", "1",
            ]
        );
    }

    #[test]
    fn test_database_url_and_path() {
        assert_eq!(database_url("/tmp/x.db"), "sqlite:////tmp/x.db");
        let p = run_db_path("/tmp/yc_bench_dbs", "medium_seed1_abcd1234");
        assert_eq!(p, "/tmp/yc_bench_dbs/yc_bench_medium_seed1_abcd1234.db");
    }

    #[test]
    fn test_compute_composite_score_zero_funds() {
        // funds <= 0 -> funds_score 0.0; survived false -> 0 overall.
        let s = compute_composite_score_default(0, false, 0.5, 0.5);
        assert!((s - 0.0).abs() < 1e-12);
        // negative funds survived (edge): survival 0.5, funds 0.
        let s2 = compute_composite_score_default(-100, true, 0.5, 0.5);
        assert!((s2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_composite_score_reference_points() {
        // funds == initial -> ~0.15 funds_score component (per docstring).
        let funds_only = compute_composite_score_default(INITIAL_FUNDS_CENTS, false, 0.0, 1.0);
        assert!((funds_only - 0.15).abs() < 0.02, "got {funds_only}");

        // 100x -> funds_score 1.0.
        let max = compute_composite_score_default(INITIAL_FUNDS_CENTS * 100, false, 0.0, 1.0);
        assert!((max - 1.0).abs() < 1e-9, "got {max}");

        // 10x -> ~0.52.
        let ten = compute_composite_score_default(INITIAL_FUNDS_CENTS * 10, false, 0.0, 1.0);
        assert!((ten - 0.52).abs() < 0.02, "got {ten}");

        // survived + initial funds with default weights.
        let composite = compute_composite_score_default(INITIAL_FUNDS_CENTS, true, 0.5, 0.5);
        // 0.5 * 1.0 + 0.5 * ~0.15
        assert!(composite > 0.55 && composite < 0.60, "got {composite}");
    }

    #[test]
    fn test_compute_composite_score_clamps_at_one() {
        let s = compute_composite_score_default(INITIAL_FUNDS_CENTS * 100_000, false, 0.0, 1.0);
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_read_final_score_missing_db() {
        let fs = read_final_score("/nonexistent/path/to/db_xyz.db");
        assert_eq!(fs.final_funds_cents, 0);
        assert!(!fs.survived);
        assert_eq!(fs.terminal_reason, "db_missing");
    }

    #[test]
    fn test_read_final_score_survived() {
        let dir = std::env::temp_dir();
        let db = dir.join(format!("yc_bench_test_survived_{}.db", std::process::id()));
        let db_path = db.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&db_path);
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE companies (funds_cents INTEGER);
                 INSERT INTO companies (funds_cents) VALUES (500000);
                 CREATE TABLE sim_events (event_type TEXT, scheduled_at INTEGER);
                 INSERT INTO sim_events (event_type, scheduled_at) VALUES ('horizon_end', 100);",
            )
            .unwrap();
        }
        let fs = read_final_score(&db_path);
        assert_eq!(fs.final_funds_cents, 500000);
        assert!(fs.survived);
        assert_eq!(fs.terminal_reason, "horizon_end");
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_read_final_score_bankruptcy() {
        let dir = std::env::temp_dir();
        let db = dir.join(format!("yc_bench_test_bankrupt_{}.db", std::process::id()));
        let db_path = db.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&db_path);
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE companies (funds_cents INTEGER);
                 INSERT INTO companies (funds_cents) VALUES (-1000);
                 CREATE TABLE sim_events (event_type TEXT, scheduled_at INTEGER);
                 INSERT INTO sim_events (event_type, scheduled_at) VALUES ('bankruptcy', 200);",
            )
            .unwrap();
        }
        let fs = read_final_score(&db_path);
        assert_eq!(fs.final_funds_cents, -1000);
        assert!(!fs.survived);
        assert_eq!(fs.terminal_reason, "bankruptcy");
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_read_final_score_missing_sim_events_table() {
        // sim_events table absent -> terminal_reason stays "unknown".
        let dir = std::env::temp_dir();
        let db = dir.join(format!("yc_bench_test_noevents_{}.db", std::process::id()));
        let db_path = db.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&db_path);
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE companies (funds_cents INTEGER);
                 INSERT INTO companies (funds_cents) VALUES (12345);",
            )
            .unwrap();
        }
        let fs = read_final_score(&db_path);
        assert_eq!(fs.final_funds_cents, 12345);
        assert!(fs.survived); // funds >= 0 and reason != bankruptcy
        assert_eq!(fs.terminal_reason, "unknown");
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_eval_metrics_aggregation() {
        let results = vec![
            {
                let mut r = RunResult::error("medium", 1, "boom", 1.0);
                r.composite_score = 0.0;
                r
            },
            {
                let score = FinalScore {
                    final_funds_cents: INITIAL_FUNDS_CENTS,
                    survived: true,
                    terminal_reason: "horizon_end".into(),
                };
                let comp = compute_composite_score_default(score.final_funds_cents, true, 0.5, 0.5);
                RunResult::success("medium", 2, &score, comp, 50, true, 12.0, "/tmp/x.db")
            },
            {
                let score = FinalScore {
                    final_funds_cents: 1000,
                    survived: true,
                    terminal_reason: "horizon_end".into(),
                };
                RunResult::success("hard", 1, &score, 0.5, 30, true, 9.0, "/tmp/y.db")
            },
        ];
        let m = EvalMetrics::compute(&results, 99.5).metrics;
        assert_eq!(m["eval/total_runs"], 3.0);
        assert_eq!(m["eval/survived_runs"], 2.0);
        assert!((m["eval/survival_rate"] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(m["eval/evaluation_time_seconds"], 99.5);
        // Per-preset keys present.
        assert!((m["eval/survival_rate_medium"] - 0.5).abs() < 1e-12);
        assert!((m["eval/survival_rate_hard"] - 1.0).abs() < 1e-12);
        assert!(m.contains_key("eval/avg_score_medium"));
        assert!(m.contains_key("eval/avg_score_hard"));
    }

    #[test]
    fn test_eval_metrics_empty() {
        let m = EvalMetrics::compute(&[], 1.0).metrics;
        assert!(m.is_empty());
    }

    #[test]
    fn test_preset_key_dash_replacement() {
        let score = FinalScore {
            final_funds_cents: 100,
            survived: true,
            terminal_reason: "horizon_end".into(),
        };
        let r = RunResult::success("high-reward", 1, &score, 0.7, 10, true, 1.0, "/tmp/z.db");
        let m = EvalMetrics::compute(&[r], 1.0).metrics;
        assert!(m.contains_key("eval/survival_rate_high_reward"));
        assert!(m.contains_key("eval/avg_score_high_reward"));
    }

    #[test]
    fn test_streaming_result_log_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir
            .join(format!("yc_bench_stream_{}.jsonl", std::process::id()))
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_file(&path);
        let mut logf = StreamingResultLog::open(&path).unwrap();
        let r = RunResult::error("medium", 1, "x", 1.0);
        logf.save(&r).unwrap();
        logf.save(&r).unwrap();
        logf.close();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 2);
        let first: RunResult = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        assert_eq!(first.preset, "medium");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_run_result_usd_conversion() {
        let score = FinalScore {
            final_funds_cents: 25_000_000,
            survived: true,
            terminal_reason: "horizon_end".into(),
        };
        let r = RunResult::success("medium", 1, &score, 0.9, 5, true, 1.0, "/tmp/a.db");
        assert!((r.final_funds_usd - 250_000.0).abs() < 1e-9);
    }
}
