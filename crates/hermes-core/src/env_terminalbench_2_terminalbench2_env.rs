//! TerminalBench2Env -- Terminal-Bench 2.0 Evaluation Environment (native Rust port)
//!
//! Port of `environments/benchmarks/terminalbench_2/terminalbench2_env.py`.
//!
//! The original Python module is an eval-only environment that:
//!   1. loads the Terminal-Bench 2.0 dataset from HuggingFace,
//!   2. for each task resolves a Docker image, runs an agent loop inside a
//!      per-task Modal sandbox, uploads a test suite, runs `test.sh`, and reads
//!      `/logs/verifier/reward.txt` to decide pass/fail,
//!   3. aggregates per-task / per-category / overall pass rates.
//!
//! The async orchestration (asyncio.gather, Modal sandbox creation, the
//! HermesAgentLoop, HuggingFace `load_dataset`, wandb logging) lives in other
//! subsystems and is out of scope for a faithful self-contained port. This
//! module reproduces the *deterministic, testable* logic that does not require
//! network or a live sandbox:
//!
//!   * safe tar member-path normalisation (`normalize_tar_member_parts`)
//!   * safe tar extraction without traversal / link entries (`safe_extract_tar`)
//!   * base64 `.tar.gz` extraction (`extract_base64_tar`)
//!   * the default eval configuration (`TerminalBench2EvalConfig`)
//!   * the Modal-incompatible task skip list (`modal_incompatible_tasks`)
//!   * task filtering / skipping (`filter_tasks`)
//!   * Docker image resolution (`resolve_task_image`)
//!   * reward-file parsing (`parse_reward`)
//!   * per-category / overall metric aggregation (`aggregate_metrics`)
//!
//! Behaviour is matched byte-for-byte where it is observable (e.g. the
//! `reward.txt` parsing fallbacks, the category-key normalisation, the unsafe
//! archive member rules).

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};

use base64::Engine as _;
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Terminal-Bench 2.0 evaluation environment.
///
/// Mirrors the TB2-specific fields added on top of `HermesAgentEnvConfig`. The
/// inherited base-env fields that the eval flow actually reads are included
/// here (with their TB2 defaults from `config_init`) so the config is usable
/// standalone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TerminalBench2EvalConfig {
    // --- Dataset ---
    pub dataset_name: String,

    // --- Test execution ---
    pub test_timeout: i64,

    // --- Image strategy ---
    pub force_build: bool,

    // --- Task filtering (comma-separated from CLI) ---
    pub task_filter: Option<String>,
    pub skip_tasks: Option<String>,

    // --- Per-task wall-clock timeout ---
    pub task_timeout: i64,

    // --- Concurrency control ---
    pub max_concurrent_tasks: i64,

    // --- Eval concurrency ---
    pub eval_concurrency: i64,

    // --- Inherited base-env fields used by the eval flow (TB2 defaults) ---
    pub enabled_toolsets: Vec<String>,
    pub max_agent_turns: i64,
    pub max_token_length: i64,
    pub agent_temperature: f64,
    pub system_prompt: Option<String>,
    pub terminal_backend: String,
    pub terminal_timeout: i64,
    pub tool_pool_size: i64,
    pub group_size: i64,
    pub steps_per_eval: i64,
    pub total_steps: i64,
    pub tokenizer_name: String,
    pub use_wandb: bool,
    pub wandb_name: String,
    pub ensure_scores_are_not_same: bool,
    /// Auto-set during `setup()` to `task_timeout + 120`.
    pub terminal_lifetime: i64,
}

impl Default for TerminalBench2EvalConfig {
    fn default() -> Self {
        // Field-level Pydantic defaults from `TerminalBench2EvalConfig`.
        Self {
            dataset_name: "NousResearch/terminal-bench-2".to_string(),
            test_timeout: 180,
            force_build: false,
            task_filter: None,
            skip_tasks: None,
            task_timeout: 1800,
            max_concurrent_tasks: 8,
            eval_concurrency: 0,
            // Base-env defaults are unset here; `config_init` is the
            // authoritative source for the TB2 run defaults below.
            enabled_toolsets: vec!["terminal".to_string(), "file".to_string()],
            max_agent_turns: 60,
            max_token_length: 16000,
            agent_temperature: 0.6,
            system_prompt: None,
            terminal_backend: "modal".to_string(),
            terminal_timeout: 300,
            tool_pool_size: 128,
            group_size: 1,
            steps_per_eval: 1,
            total_steps: 1,
            tokenizer_name: "NousResearch/Hermes-3-Llama-3.1-8B".to_string(),
            use_wandb: true,
            wandb_name: "terminal-bench-2".to_string(),
            ensure_scores_are_not_same: false,
            terminal_lifetime: 0,
        }
    }
}

impl TerminalBench2EvalConfig {
    /// Build the default TB2 eval configuration (matches `config_init`).
    pub fn config_init() -> Self {
        Self::default()
    }

    /// Replicates `setup()`'s auto-set of `terminal_lifetime` to
    /// `task_timeout + 120`. Returns the lifetime (also stored on `self`), so
    /// callers can mirror the Python `os.environ["TERMINAL_LIFETIME_SECONDS"]`
    /// side-effect.
    pub fn auto_set_terminal_lifetime(&mut self) -> i64 {
        let lifetime = self.task_timeout + 120;
        self.terminal_lifetime = lifetime;
        lifetime
    }
}

/// Default OpenRouter/Claude server config produced by `config_init`.
///
/// `api_key` is read from `OPENROUTER_API_KEY` (empty string if unset), exactly
/// like the Python `os.getenv("OPENROUTER_API_KEY", "")`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiServerConfig {
    pub base_url: String,
    pub model_name: String,
    pub server_type: String,
    pub api_key: String,
    pub health_check: bool,
}

impl ApiServerConfig {
    pub fn tb2_default() -> Self {
        Self {
            base_url: "https://openrouter.ai/api/v1".to_string(),
            model_name: "anthropic/claude-sonnet-4".to_string(),
            server_type: "openai".to_string(),
            api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            health_check: false,
        }
    }
}

/// Tasks that cannot run properly on Modal and are excluded from scoring.
pub fn modal_incompatible_tasks() -> HashSet<String> {
    [
        "qemu-startup",    // Needs KVM/hardware virtualization
        "qemu-alpine-ssh", // Needs KVM/hardware virtualization
        "crack-7z-hash",   // Password brute-force -- too slow for cloud timeouts
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// =============================================================================
// Tar extraction helper
// =============================================================================

/// Returns true if a Windows-style path component looks like a drive letter
/// (e.g. `C:`), matching `PureWindowsPath(...).drive`.
fn has_windows_drive(name: &str) -> bool {
    let bytes = name.as_bytes();
    // Drive prefix like "C:" or "C:\..." / "C:/...".
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return true;
    }
    // UNC prefix like "\\server\share".
    name.starts_with("\\\\") || name.starts_with("//")
}

fn is_windows_absolute(name: &str) -> bool {
    if has_windows_drive(name) {
        return true;
    }
    // A leading separator is "absolute" on Windows too.
    name.starts_with('\\') || name.starts_with('/')
}

/// Return safe path components for a tar member or an error.
///
/// Faithful port of `_normalize_tar_member_parts`. Rejects empty names,
/// absolute POSIX or Windows paths, drive-qualified Windows paths, and any
/// path containing a `..` component.
pub fn normalize_tar_member_parts(member_name: &str) -> Result<Vec<String>, String> {
    let normalized_name = member_name.replace('\\', "/");

    let posix_is_absolute = normalized_name.starts_with('/');

    if normalized_name.is_empty()
        || posix_is_absolute
        || is_windows_absolute(member_name)
        || has_windows_drive(member_name)
    {
        return Err(format!("Unsafe archive member path: {member_name}"));
    }

    // PurePosixPath.parts, dropping "" and ".".
    let parts: Vec<String> = normalized_name
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .map(|p| p.to_string())
        .collect();

    if parts.is_empty() || parts.iter().any(|p| p == "..") {
        return Err(format!("Unsafe archive member path: {member_name}"));
    }

    Ok(parts)
}

/// Resolve `target_dir.join(parts)` lexically and confirm it stays within
/// `target_root`. Used to defend against symlink/`..` traversal the same way
/// the Python code uses `Path.resolve().relative_to(root)`.
fn join_within(target_dir: &Path, parts: &[String]) -> PathBuf {
    let mut p = target_dir.to_path_buf();
    for part in parts {
        p.push(part);
    }
    p
}

/// Lexical containment check: is `path` inside `root` after normalising `.`
/// and `..` components without touching the filesystem.
fn is_within(root: &Path, path: &Path) -> bool {
    let clean = |p: &Path| -> PathBuf {
        let mut out = PathBuf::new();
        for comp in p.components() {
            match comp {
                Component::ParentDir => {
                    out.pop();
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        out
    };

    let root_c = clean(root);
    let path_c = clean(path);
    path_c == root_c || path_c.starts_with(&root_c)
}

/// Extract a `tar` archive without allowing traversal or link entries.
///
/// Faithful port of `_safe_extract_tar`. Directories are created, regular
/// files are written with their mode masked to `0o777`; any other member type
/// (symlink/hardlink/device/fifo) is rejected with an error.
pub fn safe_extract_tar<R: Read>(
    archive: &mut tar::Archive<R>,
    target_dir: &Path,
) -> Result<(), String> {
    fs::create_dir_all(target_dir).map_err(|e| e.to_string())?;
    // The Python code resolves both the root and each target before comparing.
    // We compare lexically (no symlink resolution) against the *same*
    // `target_dir` we build member paths from, so the two sides are consistent
    // regardless of OS symlink quirks (e.g. macOS `/var` -> `/private/var`).
    let target_root = target_dir;

    let entries = archive.entries().map_err(|e| e.to_string())?;
    for entry in entries {
        let mut entry = entry.map_err(|e| e.to_string())?;
        // Read the raw member name from the header rather than `entry.path()`,
        // which the `tar` crate would reject for `..` segments before we get a
        // chance to apply our own (faithful) safety checks.
        let name = String::from_utf8_lossy(&entry.path_bytes()).to_string();

        let parts = normalize_tar_member_parts(&name)?;
        let target = join_within(target_dir, &parts);

        if !is_within(target_root, &target) {
            return Err(format!("Unsafe archive member path: {name}"));
        }

        let etype = entry.header().entry_type();
        if etype.is_dir() {
            fs::create_dir_all(&target).map_err(|e| e.to_string())?;
            continue;
        }

        if !etype.is_file() {
            return Err(format!("Unsupported archive member type: {name}"));
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        fs::write(&target, &buf).map_err(|e| e.to_string())?;

        // Best-effort chmod(member.mode & 0o777); ignore failures (OSError).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(mode) = entry.header().mode() {
                let _ = fs::set_permissions(&target, fs::Permissions::from_mode(mode & 0o777));
            }
        }
    }

    Ok(())
}

/// Extract a base64-encoded `tar.gz` archive into `target_dir`.
///
/// Faithful port of `_extract_base64_tar`: empty input is a no-op.
pub fn extract_base64_tar(b64_data: &str, target_dir: &Path) -> Result<(), String> {
    if b64_data.is_empty() {
        return Ok(());
    }
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64_data)
        .map_err(|e| format!("base64 decode failed: {e}"))?;
    let gz = GzDecoder::new(Cursor::new(raw));
    let mut archive = tar::Archive::new(gz);
    safe_extract_tar(&mut archive, target_dir)
}

// =============================================================================
// Task model + filtering
// =============================================================================

/// A single Terminal-Bench 2.0 task row from the dataset.
///
/// Only the fields the eval flow reads are modelled; everything else is ignored
/// on deserialisation. Missing string fields default to `""` and missing
/// `category` / `task_name` default to `"unknown"`, matching the Python
/// `item.get(..., default)` calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tb2Task {
    #[serde(default = "default_unknown")]
    pub task_name: String,
    #[serde(default = "default_unknown")]
    pub category: String,
    #[serde(default)]
    pub instruction: String,
    #[serde(default)]
    pub docker_image: String,
    #[serde(default)]
    pub environment_tar: String,
    #[serde(default)]
    pub tests_tar: String,
    #[serde(default)]
    pub test_sh: String,
}

fn default_unknown() -> String {
    "unknown".to_string()
}

/// Parse a comma-separated CLI list into a trimmed set (matches
/// `{name.strip() for name in value.split(",")}`).
pub fn parse_comma_set(value: &str) -> HashSet<String> {
    value.split(',').map(|s| s.trim().to_string()).collect()
}

/// Apply the task_filter / skip logic from `setup()`.
///
/// 1. If `task_filter` is set, keep only tasks whose name is in the filter set.
/// 2. Build the skip set: `MODAL_INCOMPATIBLE_TASKS` when backend == "modal",
///    unioned with `skip_tasks` if present, then drop any matching tasks.
pub fn filter_tasks(tasks: Vec<Tb2Task>, config: &TerminalBench2EvalConfig) -> Vec<Tb2Task> {
    let mut out = tasks;

    if let Some(ref filt) = config.task_filter {
        let allowed = parse_comma_set(filt);
        out.retain(|t| allowed.contains(&t.task_name));
    }

    let mut skip: HashSet<String> = if config.terminal_backend == "modal" {
        modal_incompatible_tasks()
    } else {
        HashSet::new()
    };
    if let Some(ref sk) = config.skip_tasks {
        skip.extend(parse_comma_set(sk));
    }
    if !skip.is_empty() {
        out.retain(|t| !skip.contains(&t.task_name));
    }

    out
}

/// Build the per-category index `{category -> [indices]}` from `setup()`,
/// using `category` defaulting to `"unknown"`. Returned sorted by category
/// (BTreeMap) to match the sorted iteration in the Python summary output.
pub fn build_category_index(tasks: &[Tb2Task]) -> BTreeMap<String, Vec<usize>> {
    let mut idx: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, t) in tasks.iter().enumerate() {
        idx.entry(t.category.clone()).or_default().push(i);
    }
    idx
}

// =============================================================================
// Docker image resolution
// =============================================================================

/// Outcome of image resolution: a Docker Hub image name or a Dockerfile path,
/// plus an optional temp dir that needs later cleanup.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedImage {
    /// Docker Hub image name, or an absolute path to an extracted Dockerfile.
    /// Empty string means "no image available".
    pub modal_image: String,
    /// Set when we extracted files that must be cleaned up.
    pub temp_dir: Option<PathBuf>,
}

/// Resolve the Docker image for a task, with fallback to Dockerfile.
///
/// Faithful port of `_resolve_task_image`:
/// 1. If `force_build` is false and `docker_image` is set, use the pre-built
///    Hub image (fast path).
/// 2. Otherwise, if `environment_tar` is present, extract it into a fresh temp
///    dir; if a `Dockerfile` exists there, return its path + the temp dir.
/// 3. Fall back to `docker_image` if `force_build` was set but no tar existed.
/// 4. Otherwise return an empty image with no temp dir.
///
/// `temp_root` is the directory under which the extraction temp dir is created
/// (Python uses `tempfile.mkdtemp`; pass `std::env::temp_dir()` to match).
pub fn resolve_task_image(
    item: &Tb2Task,
    task_name: &str,
    config: &TerminalBench2EvalConfig,
    temp_root: &Path,
) -> Result<ResolvedImage, String> {
    let docker_image = &item.docker_image;
    let environment_tar = &item.environment_tar;

    // Fast path: pre-built Docker Hub image.
    if !docker_image.is_empty() && !config.force_build {
        log::info!("Task {task_name}: using pre-built image {docker_image}");
        return Ok(ResolvedImage {
            modal_image: docker_image.clone(),
            temp_dir: None,
        });
    }

    // Slow path: extract Dockerfile from environment_tar and build.
    if !environment_tar.is_empty() {
        let task_dir = make_temp_dir(temp_root, &format!("tb2-{task_name}-"))?;
        extract_base64_tar(environment_tar, &task_dir)?;
        let dockerfile_path = task_dir.join("Dockerfile");
        if dockerfile_path.exists() {
            log::info!(
                "Task {task_name}: building from Dockerfile (force_build={}, docker_image={})",
                config.force_build,
                !docker_image.is_empty()
            );
            return Ok(ResolvedImage {
                modal_image: dockerfile_path.to_string_lossy().to_string(),
                temp_dir: Some(task_dir),
            });
        }
        // No Dockerfile found; the temp dir is still on disk. Python leaves it
        // for the caller's `finally` cleanup, but since we don't return it here
        // when falling through, remove it to avoid a leak.
        let _ = fs::remove_dir_all(&task_dir);
    }

    // Neither available -- fall back to Hub image if force_build was True.
    if !docker_image.is_empty() {
        log::warn!(
            "Task {task_name}: force_build=True but no environment_tar, \
             falling back to docker_image {docker_image}"
        );
        return Ok(ResolvedImage {
            modal_image: docker_image.clone(),
            temp_dir: None,
        });
    }

    Ok(ResolvedImage {
        modal_image: String::new(),
        temp_dir: None,
    })
}

/// Create a unique temp directory under `root` with the given prefix, mirroring
/// `tempfile.mkdtemp(prefix=...)`.
fn make_temp_dir(root: &Path, prefix: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    for _ in 0..1000 {
        let suffix: u64 = {
            // Cheap unique-ish suffix without pulling in a new dep.
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            nanos ^ (std::process::id() as u64).rotate_left(17)
        };
        let candidate = root.join(format!("{prefix}{suffix:016x}"));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("could not create unique temp dir".to_string())
}

// =============================================================================
// Reward parsing
// =============================================================================

/// Parse the contents of `reward.txt` exactly like `_run_tests`.
///
/// * `"1"` -> 1.0
/// * `"0"` -> 0.0
/// * any other value: try parsing as f64; on failure fall back to
///   `exit_code == 0 ? 1.0 : 0.0`.
///
/// `content` should already be stripped of surrounding whitespace (the Python
/// code calls `.strip()`); this function strips again for safety.
pub fn parse_reward_content(content: &str, exit_code: i64) -> f64 {
    let c = content.trim();
    if c == "1" {
        1.0
    } else if c == "0" {
        0.0
    } else {
        match c.parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                if exit_code == 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Decide the reward from a (possibly missing/empty) `reward.txt` and the test
/// `exit_code`, matching the full `_run_tests` branch logic.
///
/// `reward_txt`:
///   * `Some(content)` with non-empty content -> `parse_reward_content`
///   * `Some("")` / `None` (file missing or empty) -> exit-code fallback.
pub fn parse_reward(reward_txt: Option<&str>, exit_code: i64) -> f64 {
    match reward_txt {
        Some(content) if !content.trim_end_matches(['\n', '\r']).is_empty() => {
            parse_reward_content(content, exit_code)
        }
        _ => {
            if exit_code == 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

// =============================================================================
// Metrics aggregation
// =============================================================================

/// A per-task evaluation result, mirroring the dicts returned by
/// `rollout_and_score_eval` / `_eval_with_timeout`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskResult {
    pub passed: bool,
    pub reward: f64,
    pub task_name: String,
    pub category: String,
    #[serde(default)]
    pub turns_used: Option<i64>,
    #[serde(default)]
    pub finished_naturally: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Aggregated evaluation metrics (the `eval_metrics` dict from `evaluate`).
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateMetrics {
    pub pass_rate: f64,
    pub total_tasks: usize,
    pub passed_tasks: usize,
    pub evaluation_time_seconds: f64,
    /// `eval/pass_rate_<cat_key>` -> rate, with normalised category keys.
    pub per_category_pass_rate: BTreeMap<String, f64>,
}

/// Normalise a category name into a metric key:
/// `replace(" ", "_").replace("-", "_").lower()`.
pub fn normalize_category_key(category: &str) -> String {
    category.replace(' ', "_").replace('-', "_").to_lowercase()
}

/// Compute overall + per-category metrics from valid results, matching the
/// metric construction in `evaluate`.
///
/// `evaluation_time_seconds` is `end_time - start_time`, passed by the caller.
pub fn aggregate_metrics(results: &[TaskResult], evaluation_time_seconds: f64) -> AggregateMetrics {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let overall_pass_rate = if total > 0 {
        passed as f64 / total as f64
    } else {
        0.0
    };

    // Per-category breakdown (sorted by category via BTreeMap).
    let mut cat_results: BTreeMap<String, Vec<&TaskResult>> = BTreeMap::new();
    for r in results {
        cat_results.entry(r.category.clone()).or_default().push(r);
    }

    let mut per_category_pass_rate = BTreeMap::new();
    for (category, items) in &cat_results {
        let cat_passed = items.iter().filter(|r| r.passed).count();
        let cat_total = items.len();
        let cat_rate = if cat_total > 0 {
            cat_passed as f64 / cat_total as f64
        } else {
            0.0
        };
        per_category_pass_rate.insert(normalize_category_key(category), cat_rate);
    }

    AggregateMetrics {
        pass_rate: overall_pass_rate,
        total_tasks: total,
        passed_tasks: passed,
        evaluation_time_seconds,
        per_category_pass_rate,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn config_defaults_match_python() {
        let c = TerminalBench2EvalConfig::config_init();
        assert_eq!(c.dataset_name, "NousResearch/terminal-bench-2");
        assert_eq!(c.test_timeout, 180);
        assert!(!c.force_build);
        assert_eq!(c.task_timeout, 1800);
        assert_eq!(c.max_concurrent_tasks, 8);
        assert_eq!(c.eval_concurrency, 0);
        assert_eq!(c.max_agent_turns, 60);
        assert_eq!(c.max_token_length, 16000);
        assert!((c.agent_temperature - 0.6).abs() < 1e-9);
        assert_eq!(c.terminal_backend, "modal");
        assert_eq!(c.terminal_timeout, 300);
        assert_eq!(c.tool_pool_size, 128);
        assert_eq!(c.group_size, 1);
        assert!(c.use_wandb);
        assert_eq!(c.wandb_name, "terminal-bench-2");
        assert!(!c.ensure_scores_are_not_same);
    }

    #[test]
    fn terminal_lifetime_is_task_timeout_plus_120() {
        let mut c = TerminalBench2EvalConfig::config_init();
        let life = c.auto_set_terminal_lifetime();
        assert_eq!(life, 1800 + 120);
        assert_eq!(c.terminal_lifetime, 1920);
    }

    #[test]
    fn server_config_reads_env() {
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-123");
        }
        let s = ApiServerConfig::tb2_default();
        assert_eq!(s.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(s.model_name, "anthropic/claude-sonnet-4");
        assert_eq!(s.server_type, "openai");
        assert_eq!(s.api_key, "test-key-123");
        assert!(!s.health_check);
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
        let s2 = ApiServerConfig::tb2_default();
        assert_eq!(s2.api_key, "");
    }

    #[test]
    fn normalize_rejects_unsafe_paths() {
        assert!(normalize_tar_member_parts("").is_err());
        assert!(normalize_tar_member_parts("/etc/passwd").is_err());
        assert!(normalize_tar_member_parts("../escape").is_err());
        assert!(normalize_tar_member_parts("a/../b").is_err());
        assert!(normalize_tar_member_parts("C:\\Windows").is_err());
        assert!(normalize_tar_member_parts("\\\\server\\share").is_err());
        assert!(normalize_tar_member_parts("\\abs").is_err());
    }

    #[test]
    fn normalize_accepts_and_cleans_safe_paths() {
        assert_eq!(
            normalize_tar_member_parts("foo/bar.txt").unwrap(),
            vec!["foo".to_string(), "bar.txt".to_string()]
        );
        assert_eq!(
            normalize_tar_member_parts("./a/./b").unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        // backslash treated as separator
        assert_eq!(
            normalize_tar_member_parts("a\\b").unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn reward_parsing_branches() {
        assert_eq!(parse_reward_content("1", 0), 1.0);
        assert_eq!(parse_reward_content("0", 0), 0.0);
        assert_eq!(parse_reward_content("  1  ", 1), 1.0);
        // float-parseable other value
        assert_eq!(parse_reward_content("0.5", 1), 0.5);
        // non-numeric -> exit code fallback
        assert_eq!(parse_reward_content("garbage", 0), 1.0);
        assert_eq!(parse_reward_content("garbage", 3), 0.0);
    }

    #[test]
    fn reward_missing_or_empty_uses_exit_code() {
        assert_eq!(parse_reward(None, 0), 1.0);
        assert_eq!(parse_reward(None, 5), 0.0);
        assert_eq!(parse_reward(Some(""), 0), 1.0);
        assert_eq!(parse_reward(Some("\n"), 7), 0.0);
        assert_eq!(parse_reward(Some("1"), 9), 1.0);
        assert_eq!(parse_reward(Some("0"), 0), 0.0);
    }

    #[test]
    fn category_key_normalisation() {
        assert_eq!(normalize_category_key("System Admin"), "system_admin");
        assert_eq!(normalize_category_key("data-science"), "data_science");
        assert_eq!(normalize_category_key("Mixed Case-Thing"), "mixed_case_thing");
    }

    #[test]
    fn filter_and_skip_tasks() {
        let tasks = vec![
            Tb2Task {
                task_name: "fix-git".into(),
                category: "git".into(),
                ..mk_task()
            },
            Tb2Task {
                task_name: "qemu-startup".into(),
                category: "vm".into(),
                ..mk_task()
            },
            Tb2Task {
                task_name: "git-multibranch".into(),
                category: "git".into(),
                ..mk_task()
            },
        ];

        // Modal backend skips qemu-startup automatically.
        let mut cfg = TerminalBench2EvalConfig::config_init();
        let out = filter_tasks(tasks.clone(), &cfg);
        let names: Vec<_> = out.iter().map(|t| t.task_name.clone()).collect();
        assert_eq!(names, vec!["fix-git", "git-multibranch"]);

        // task_filter restricts the set first.
        cfg.task_filter = Some("fix-git".into());
        let out2 = filter_tasks(tasks.clone(), &cfg);
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].task_name, "fix-git");

        // skip_tasks unions with the modal set.
        cfg.task_filter = None;
        cfg.skip_tasks = Some("git-multibranch".into());
        let out3 = filter_tasks(tasks, &cfg);
        let names3: Vec<_> = out3.iter().map(|t| t.task_name.clone()).collect();
        assert_eq!(names3, vec!["fix-git"]);
    }

    #[test]
    fn category_index_groups_and_defaults_unknown() {
        let tasks = vec![
            Tb2Task {
                task_name: "a".into(),
                category: "x".into(),
                ..mk_task()
            },
            Tb2Task {
                task_name: "b".into(),
                category: "unknown".into(),
                ..mk_task()
            },
            Tb2Task {
                task_name: "c".into(),
                category: "x".into(),
                ..mk_task()
            },
        ];
        let idx = build_category_index(&tasks);
        assert_eq!(idx.get("x").unwrap(), &vec![0usize, 2]);
        assert_eq!(idx.get("unknown").unwrap(), &vec![1usize]);
    }

    #[test]
    fn resolve_image_fast_path() {
        let item = Tb2Task {
            docker_image: "hub/img:1".into(),
            ..mk_task()
        };
        let cfg = TerminalBench2EvalConfig::config_init();
        let r = resolve_task_image(&item, "t", &cfg, &std::env::temp_dir()).unwrap();
        assert_eq!(r.modal_image, "hub/img:1");
        assert!(r.temp_dir.is_none());
    }

    #[test]
    fn resolve_image_none_available() {
        let item = mk_task();
        let cfg = TerminalBench2EvalConfig::config_init();
        let r = resolve_task_image(&item, "t", &cfg, &std::env::temp_dir()).unwrap();
        assert_eq!(r.modal_image, "");
        assert!(r.temp_dir.is_none());
    }

    #[test]
    fn resolve_image_force_build_falls_back_without_tar() {
        let item = Tb2Task {
            docker_image: "hub/img:2".into(),
            ..mk_task()
        };
        let mut cfg = TerminalBench2EvalConfig::config_init();
        cfg.force_build = true;
        let r = resolve_task_image(&item, "t", &cfg, &std::env::temp_dir()).unwrap();
        assert_eq!(r.modal_image, "hub/img:2");
    }

    #[test]
    fn resolve_image_builds_from_extracted_dockerfile() {
        // Build a base64 tar.gz containing a Dockerfile.
        let mut tar_buf = Vec::new();
        {
            let enc = flate2::write::GzEncoder::new(&mut tar_buf, flate2::Compression::default());
            let mut builder = tar::Builder::new(enc);
            let data = b"FROM scratch\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "Dockerfile", &data[..])
                .unwrap();
            builder.into_inner().unwrap().finish().unwrap();
        }
        let b64 = base64::engine::general_purpose::STANDARD.encode(&tar_buf);

        let item = Tb2Task {
            environment_tar: b64,
            ..mk_task()
        };
        let cfg = TerminalBench2EvalConfig::config_init();
        let r = resolve_task_image(&item, "build-test", &cfg, &std::env::temp_dir()).unwrap();
        assert!(r.modal_image.ends_with("Dockerfile"));
        assert!(r.temp_dir.is_some());
        // Cleanup
        if let Some(d) = r.temp_dir {
            let _ = fs::remove_dir_all(d);
        }
    }

    #[test]
    fn extract_base64_tar_roundtrip_and_safety() {
        // Good archive.
        let mut tar_buf = Vec::new();
        {
            let enc = flate2::write::GzEncoder::new(&mut tar_buf, flate2::Compression::default());
            let mut builder = tar::Builder::new(enc);
            let data = b"hello";
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o600);
            header.set_cksum();
            builder
                .append_data(&mut header, "sub/file.txt", &data[..])
                .unwrap();
            builder.into_inner().unwrap().finish().unwrap();
        }
        let b64 = base64::engine::general_purpose::STANDARD.encode(&tar_buf);

        let dir = make_temp_dir(&std::env::temp_dir(), "tb2-test-").unwrap();
        extract_base64_tar(&b64, &dir).unwrap();
        let mut f = fs::File::open(dir.join("sub/file.txt")).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        assert_eq!(s, "hello");
        let _ = fs::remove_dir_all(&dir);

        // Empty input is a no-op.
        let dir2 = make_temp_dir(&std::env::temp_dir(), "tb2-empty-").unwrap();
        extract_base64_tar("", &dir2).unwrap();
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn extract_rejects_traversal_member() {
        // The `tar` crate's `append_data` refuses to *build* a `..` member, so
        // we hand-craft a single 512-byte ustar header whose name field is
        // "../evil.txt" to exercise our own (faithful) rejection path.
        let mut header = [0u8; 512];
        let name = b"../evil.txt";
        header[..name.len()].copy_from_slice(name);
        // mode "0000644\0"
        header[100..108].copy_from_slice(b"0000644\0");
        // uid / gid
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");
        // size = 0 (octal)
        header[124..136].copy_from_slice(b"00000000000\0");
        // mtime = 0
        header[136..148].copy_from_slice(b"00000000000\0");
        // typeflag '0' (regular file)
        header[156] = b'0';
        // ustar magic
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        // checksum: fill field with spaces, sum all bytes, write octal.
        for b in &mut header[148..156] {
            *b = b' ';
        }
        let sum: u32 = header.iter().map(|&b| b as u32).sum();
        let chk = format!("{sum:06o}\0 ");
        header[148..148 + chk.len()].copy_from_slice(chk.as_bytes());

        let mut tar_buf = header.to_vec();
        // Two zero blocks mark end of archive.
        tar_buf.extend(std::iter::repeat(0u8).take(1024));

        let dir = make_temp_dir(&std::env::temp_dir(), "tb2-evil-").unwrap();
        let mut archive = tar::Archive::new(Cursor::new(tar_buf));
        let res = safe_extract_tar(&mut archive, &dir);
        assert!(res.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn aggregate_metrics_overall_and_categories() {
        let results = vec![
            TaskResult {
                passed: true,
                reward: 1.0,
                task_name: "a".into(),
                category: "Git Stuff".into(),
                turns_used: Some(3),
                finished_naturally: Some(true),
                error: None,
            },
            TaskResult {
                passed: false,
                reward: 0.0,
                task_name: "b".into(),
                category: "Git Stuff".into(),
                turns_used: Some(5),
                finished_naturally: Some(false),
                error: None,
            },
            TaskResult {
                passed: true,
                reward: 1.0,
                task_name: "c".into(),
                category: "data-proc".into(),
                turns_used: Some(2),
                finished_naturally: Some(true),
                error: None,
            },
        ];
        let m = aggregate_metrics(&results, 12.5);
        assert_eq!(m.total_tasks, 3);
        assert_eq!(m.passed_tasks, 2);
        assert!((m.pass_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((m.evaluation_time_seconds - 12.5).abs() < 1e-9);
        assert!((m.per_category_pass_rate["git_stuff"] - 0.5).abs() < 1e-9);
        assert!((m.per_category_pass_rate["data_proc"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_empty_results() {
        let m = aggregate_metrics(&[], 0.0);
        assert_eq!(m.total_tasks, 0);
        assert_eq!(m.passed_tasks, 0);
        assert_eq!(m.pass_rate, 0.0);
        assert!(m.per_category_pass_rate.is_empty());
    }

    #[test]
    fn modal_skip_set_contents() {
        let s = modal_incompatible_tasks();
        assert!(s.contains("qemu-startup"));
        assert!(s.contains("qemu-alpine-ssh"));
        assert!(s.contains("crack-7z-hash"));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn task_deserializes_with_defaults() {
        let json = r#"{"instruction": "do it"}"#;
        let t: Tb2Task = serde_json::from_str(json).unwrap();
        assert_eq!(t.task_name, "unknown");
        assert_eq!(t.category, "unknown");
        assert_eq!(t.instruction, "do it");
        assert_eq!(t.docker_image, "");
    }

    // Helper to build a minimal task in tests.
    fn mk_task() -> Tb2Task {
        Tb2Task {
            task_name: "unknown".into(),
            category: "unknown".into(),
            instruction: String::new(),
            docker_image: String::new(),
            environment_tar: String::new(),
            tests_tar: String::new(),
            test_sh: String::new(),
        }
    }

    // Silence unused-import warning for Write in environments where it isn't
    // otherwise referenced.
    #[allow(dead_code)]
    fn _use_write() {
        let mut v = Vec::new();
        let _ = v.write_all(b"");
    }
}
