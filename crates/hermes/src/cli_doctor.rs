//! Native Rust port of `hermes_cli/doctor.py`.
//!
//! The `doctor` command diagnoses issues with a Hermes Agent setup: Python
//! environment, configuration files, auth providers, directory structure,
//! external tools, API connectivity, tool availability, the skills hub, the
//! active memory provider, and named profiles. It can optionally auto-fix a
//! handful of problems (`--fix`).
//!
//! ## Porting notes
//!
//! The original module is deeply entangled with the Python runtime (its own
//! interpreter version, `importlib`, lazy imports of many sibling modules) and
//! with runtime-only state (live auth status, the in-process tool registry,
//! memory-plugin clients). Those Python-specific or not-yet-ported surfaces are
//! modeled as *inputs* via [`DoctorInputs`]: the caller (or a future native
//! integration) supplies the resolved facts, and this module reproduces the
//! exact diagnostic rendering and issue-collection logic. Pure helpers
//! (provider-list construction, env-hint detection, install-command selection,
//! the Termux browser steps, runtime-gated tool overrides, the API-key provider
//! health-check URL/header derivation) are ported faithfully and unit-tested.
//!
//! Network checks are performed with `reqwest::blocking`, preserving the exact
//! request shapes (URLs, headers, status-code branches) from the Python source.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::cli_colors::{color, Colors};
use crate::cli_vercel_auth::{describe_vercel_auth, VercelAuthStatus};

// ---------------------------------------------------------------------------
// Constants ported verbatim from doctor.py
// ---------------------------------------------------------------------------

/// `_PROVIDER_ENV_HINTS` — env keys whose presence in `.env` proves provider
/// auth or a custom base URL is configured.
pub const PROVIDER_ENV_HINTS: &[&str] = &[
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
    "NOUS_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "GMI_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
    "AI_GATEWAY_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
    "XIAOMI_API_KEY",
    "TOKENHUB_API_KEY",
];

/// `https://openrouter.ai/api/v1/models` — mirrors `hermes_constants.OPENROUTER_MODELS_URL`.
pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// User-Agent string used for provider health checks (`hermes_cli.models._HERMES_USER_AGENT`).
///
/// The Python value is `f"hermes-cli/{_HERMES_VERSION}"`. The version is sourced
/// at runtime; callers may override via [`DoctorInputs::hermes_user_agent`].
pub const DEFAULT_HERMES_USER_AGENT: &str = "hermes-cli/0";

/// Anthropic adapter beta headers (`agent.anthropic_adapter`), kept in sync with
/// `crate::ag_anthropic_adapter`.
pub const COMMON_BETAS: &[&str] = &[
    "fine-grained-tool-streaming-2025-05-14",
    "context-1m-2025-08-07",
];
pub const CONTEXT_1M_BETA: &str = "context-1m-2025-08-07";
pub const OAUTH_ONLY_BETAS: &[&str] = &["claude-code-20250219", "oauth-2025-04-20"];

// ---------------------------------------------------------------------------
// Small pure helpers
// ---------------------------------------------------------------------------

/// `_python_install_cmd()` — the install command shown for missing Python deps.
pub fn python_install_cmd(is_termux: bool) -> &'static str {
    if is_termux {
        "python -m pip install"
    } else {
        "uv pip install"
    }
}

/// `_system_package_install_cmd(pkg)` — platform-appropriate system-package
/// install command. `platform` mirrors Python's `sys.platform`.
pub fn system_package_install_cmd(pkg: &str, is_termux: bool, platform: &str) -> String {
    if is_termux {
        return format!("pkg install {pkg}");
    }
    if platform == "darwin" {
        return format!("brew install {pkg}");
    }
    format!("sudo apt install {pkg}")
}

/// `sys.platform` analogue for the running host.
pub fn current_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    }
}

/// `_termux_browser_setup_steps(node_installed)` — numbered browser setup steps.
pub fn termux_browser_setup_steps(node_installed: bool) -> Vec<String> {
    let mut steps: Vec<String> = Vec::new();
    let mut step = 1;
    if !node_installed {
        steps.push(format!("{step}) pkg install nodejs"));
        step += 1;
    }
    steps.push(format!("{step}) npm install -g agent-browser"));
    steps.push(format!("{}) agent-browser install", step + 1));
    steps
}

/// `_has_provider_env_config(content)` — True when `.env` text mentions any of
/// the provider env hints.
pub fn has_provider_env_config(content: &str) -> bool {
    PROVIDER_ENV_HINTS.iter().any(|key| content.contains(key))
}

/// `_safe_which(cmd)` — resolve a command on PATH, returning the full path.
///
/// Mirrors `shutil.which`: checks each PATH entry for an existing file (plus
/// `.exe` on Windows). Returns `None` on any failure.
pub fn safe_which(cmd: &str) -> Option<PathBuf> {
    // Absolute / relative path with a separator: check directly.
    if cmd.contains(std::path::MAIN_SEPARATOR) {
        let p = PathBuf::from(cmd);
        return if p.is_file() { Some(p) } else { None };
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let full = dir.join(cmd);
        if full.is_file() {
            return Some(full);
        }
        if cfg!(windows) {
            let exe = dir.join(format!("{cmd}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tool-availability override logic (pure)
// ---------------------------------------------------------------------------

/// A toolset entry the runtime reports as unavailable.
///
/// Mirrors the dict shape used by `model_tools.check_tool_availability`:
/// `{name, tools?, missing_vars?, env_vars?}`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnavailableTool {
    pub name: String,
    pub tools: Vec<String>,
    pub missing_vars: Vec<String>,
    pub env_vars: Vec<String>,
}

impl UnavailableTool {
    /// The effective env-var list — `missing_vars` if present, else `env_vars`.
    /// Mirrors `item.get("missing_vars") or item.get("env_vars") or []`.
    pub fn effective_vars(&self) -> &[String] {
        if !self.missing_vars.is_empty() {
            &self.missing_vars
        } else {
            &self.env_vars
        }
    }
}

/// `_is_kanban_worker_env_gate(item)` — True when kanban is unavailable solely
/// because `HERMES_KANBAN_TASK` is unset and every tool is a `kanban_` tool.
pub fn is_kanban_worker_env_gate(item: &UnavailableTool, kanban_task_set: bool) -> bool {
    if item.name != "kanban" {
        return false;
    }
    if kanban_task_set {
        return false;
    }
    let tools = &item.tools;
    !tools.is_empty() && tools.iter().all(|t| t.starts_with("kanban_"))
}

/// `_doctor_tool_availability_detail(toolset)` — optional explanatory suffix.
pub fn doctor_tool_availability_detail(toolset: &str, kanban_task_set: bool) -> &'static str {
    if toolset == "kanban" && !kanban_task_set {
        return "(runtime-gated; loaded only for dispatcher-spawned workers)";
    }
    ""
}

/// `_apply_doctor_tool_availability_overrides(available, unavailable)` —
/// promote runtime-gated kanban/honcho entries into the available list for
/// doctor diagnostics.
pub fn apply_doctor_tool_availability_overrides(
    available: &[String],
    unavailable: &[UnavailableTool],
    kanban_task_set: bool,
    honcho_configured: bool,
) -> (Vec<String>, Vec<UnavailableTool>) {
    let mut updated_available: Vec<String> = available.to_vec();
    let mut updated_unavailable: Vec<UnavailableTool> = Vec::new();
    for item in unavailable {
        if is_kanban_worker_env_gate(item, kanban_task_set) {
            if !updated_available.iter().any(|x| x == "kanban") {
                updated_available.push("kanban".to_string());
            }
            continue;
        }
        if item.name == "honcho" && honcho_configured {
            if !updated_available.iter().any(|x| x == "honcho") {
                updated_available.push("honcho".to_string());
            }
            continue;
        }
        updated_unavailable.push(item.clone());
    }
    (updated_available, updated_unavailable)
}

// ---------------------------------------------------------------------------
// API-key provider list (pure)
// ---------------------------------------------------------------------------

/// One API-key provider health-check entry.
///
/// Tuple format from Python:
/// `(name, env_vars, default_url, base_env, supports_models_endpoint)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKeyProvider {
    pub name: String,
    pub env_vars: Vec<String>,
    pub default_url: Option<String>,
    pub base_env: Option<String>,
    pub supports_health_check: bool,
}

impl ApiKeyProvider {
    fn new(
        name: &str,
        env_vars: &[&str],
        default_url: Option<&str>,
        base_env: Option<&str>,
        supports_health_check: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            env_vars: env_vars.iter().map(|s| s.to_string()).collect(),
            default_url: default_url.map(|s| s.to_string()),
            base_env: base_env.map(|s| s.to_string()),
            supports_health_check,
        }
    }
}

/// `_build_apikey_providers_list()` static portion. The dynamic
/// profile-augmentation branch (`from providers import list_providers`) reads
/// the in-process provider registry; callers may append extra entries via
/// [`DoctorInputs::extra_apikey_providers`].
pub fn build_apikey_providers_static() -> Vec<ApiKeyProvider> {
    vec![
        ApiKeyProvider::new(
            "Z.AI / GLM",
            &["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"],
            Some("https://api.z.ai/api/paas/v4/models"),
            Some("GLM_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "Kimi / Moonshot",
            &["KIMI_API_KEY"],
            Some("https://api.moonshot.ai/v1/models"),
            Some("KIMI_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "StepFun Step Plan",
            &["STEPFUN_API_KEY"],
            Some("https://api.stepfun.ai/step_plan/v1/models"),
            Some("STEPFUN_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "Kimi / Moonshot (China)",
            &["KIMI_CN_API_KEY"],
            Some("https://api.moonshot.cn/v1/models"),
            None,
            true,
        ),
        ApiKeyProvider::new(
            "Arcee AI",
            &["ARCEEAI_API_KEY"],
            Some("https://api.arcee.ai/api/v1/models"),
            Some("ARCEE_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "GMI Cloud",
            &["GMI_API_KEY"],
            Some("https://api.gmi-serving.com/v1/models"),
            Some("GMI_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "DeepSeek",
            &["DEEPSEEK_API_KEY"],
            Some("https://api.deepseek.com/v1/models"),
            Some("DEEPSEEK_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "Hugging Face",
            &["HF_TOKEN"],
            Some("https://router.huggingface.co/v1/models"),
            Some("HF_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "NVIDIA NIM",
            &["NVIDIA_API_KEY"],
            Some("https://integrate.api.nvidia.com/v1/models"),
            Some("NVIDIA_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "Alibaba/DashScope",
            &["DASHSCOPE_API_KEY"],
            Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"),
            Some("DASHSCOPE_BASE_URL"),
            true,
        ),
        // MiniMax global: /v1 endpoint supports /models.
        ApiKeyProvider::new(
            "MiniMax",
            &["MINIMAX_API_KEY"],
            Some("https://api.minimax.io/v1/models"),
            Some("MINIMAX_BASE_URL"),
            true,
        ),
        // MiniMax CN: /v1 endpoint does NOT support /models (returns 404).
        ApiKeyProvider::new(
            "MiniMax (China)",
            &["MINIMAX_CN_API_KEY"],
            Some("https://api.minimaxi.com/v1/models"),
            Some("MINIMAX_CN_BASE_URL"),
            false,
        ),
        ApiKeyProvider::new(
            "Vercel AI Gateway",
            &["AI_GATEWAY_API_KEY"],
            Some("https://ai-gateway.vercel.sh/v1/models"),
            Some("AI_GATEWAY_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "Kilo Code",
            &["KILOCODE_API_KEY"],
            Some("https://api.kilo.ai/api/gateway/models"),
            Some("KILOCODE_BASE_URL"),
            true,
        ),
        ApiKeyProvider::new(
            "OpenCode Zen",
            &["OPENCODE_ZEN_API_KEY"],
            Some("https://opencode.ai/zen/v1/models"),
            Some("OPENCODE_ZEN_BASE_URL"),
            true,
        ),
        // OpenCode Go has no shared /models endpoint; skip the health check.
        ApiKeyProvider::new(
            "OpenCode Go",
            &["OPENCODE_GO_API_KEY"],
            None,
            Some("OPENCODE_GO_BASE_URL"),
            false,
        ),
    ]
}

// ---------------------------------------------------------------------------
// API-key provider health-check URL/header derivation (pure)
// ---------------------------------------------------------------------------

/// Mirror of `utils.base_url_host_matches` / `crate::ag_anthropic_adapter::base_url_host_matches`.
///
/// The host of `base_url` must equal `domain` exactly (not a prefix/suffix).
pub fn base_url_host_matches(base_url: &str, domain: &str) -> bool {
    if base_url.is_empty() {
        return false;
    }
    let candidate = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("https://{base_url}")
    };
    match url::Url::parse(&candidate) {
        Ok(u) => u.host_str().map(|h| h == domain).unwrap_or(false),
        Err(_) => false,
    }
}

/// `agent.auxiliary_client._to_openai_base_url` — rewrite an Anthropic-compat
/// base URL ending in `/anthropic` to the corresponding OpenAI-compat `/v1`
/// surface.
pub fn to_openai_base_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if let Some(prefix) = trimmed.strip_suffix("/anthropic") {
        format!("{prefix}/v1")
    } else {
        base.to_string()
    }
}

/// Result of deriving the health-check request for an API-key provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthCheckRequest {
    pub url: String,
    pub user_agent: String,
}

/// Reproduce the URL + User-Agent derivation inside the API-key provider loop.
///
/// `key` is the resolved key value; `base` is the value of `base_env`
/// (already resolved from the environment, empty when unset/absent).
pub fn derive_health_check_request(
    key: &str,
    base_in: &str,
    default_url: Option<&str>,
    hermes_user_agent: &str,
) -> Option<HealthCheckRequest> {
    let mut base = base_in.to_string();
    // Auto-detect Kimi Code keys (sk-kimi-) → api.kimi.com/coding/v1.
    if base.is_empty() && key.starts_with("sk-kimi-") {
        base = "https://api.kimi.com/coding/v1".to_string();
    }
    // Anthropic-compat endpoints don't support /models — rewrite to /v1.
    if !base.is_empty() && base.trim_end_matches('/').ends_with("/anthropic") {
        base = to_openai_base_url(&base);
    }
    if base_url_host_matches(&base, "api.kimi.com") && base.trim_end_matches('/').ends_with("/coding")
    {
        base = format!("{}/v1", base.trim_end_matches('/'));
    }
    let url = if !base.is_empty() {
        format!("{}/models", base.trim_end_matches('/'))
    } else {
        default_url?.to_string()
    };
    let mut user_agent = hermes_user_agent.to_string();
    if base_url_host_matches(&base, "api.kimi.com") {
        user_agent = "claude-code/0.1.0".to_string();
    }
    Some(HealthCheckRequest { url, user_agent })
}

// ---------------------------------------------------------------------------
// Print helpers (status lines), faithful to colors.py output shape
// ---------------------------------------------------------------------------

fn fmt_check_ok(text: &str, detail: &str) -> String {
    let mark = color("\u{2713}", &[Colors::GREEN]);
    if detail.is_empty() {
        format!("  {mark} {text}")
    } else {
        format!("  {mark} {text} {}", color(detail, &[Colors::DIM]))
    }
}

fn fmt_check_warn(text: &str, detail: &str) -> String {
    let mark = color("\u{26a0}", &[Colors::YELLOW]);
    if detail.is_empty() {
        format!("  {mark} {text}")
    } else {
        format!("  {mark} {text} {}", color(detail, &[Colors::DIM]))
    }
}

fn fmt_check_fail(text: &str, detail: &str) -> String {
    let mark = color("\u{2717}", &[Colors::RED]);
    if detail.is_empty() {
        format!("  {mark} {text}")
    } else {
        format!("  {mark} {text} {}", color(detail, &[Colors::DIM]))
    }
}

fn fmt_check_info(text: &str) -> String {
    format!("    {} {text}", color("\u{2192}", &[Colors::CYAN]))
}

fn fmt_section(title: &str) -> String {
    color(title, &[Colors::CYAN, Colors::BOLD])
}

// ---------------------------------------------------------------------------
// Doctor inputs (runtime facts supplied by the caller / native integration)
// ---------------------------------------------------------------------------

/// A single OAuth/auth provider status line, mirroring the dicts returned by
/// `hermes_cli.auth.get_*_auth_status()`.
#[derive(Debug, Clone, Default)]
pub struct AuthStatus {
    pub logged_in: bool,
    pub error: Option<String>,
    pub email: Option<String>,
    pub project_id: Option<String>,
    pub region: Option<String>,
}

/// Runtime facts the doctor needs but cannot derive purely from the filesystem
/// or environment in native Rust. Defaults read what they can from `env`.
#[derive(Debug, Clone)]
pub struct DoctorInputs {
    pub should_fix: bool,
    pub is_termux: bool,
    pub platform: String,
    pub hermes_home: PathBuf,
    pub project_root: PathBuf,
    pub display_hermes_home: String,
    pub hermes_user_agent: String,

    /// `os.getenv("OPENROUTER_API_KEY")`.
    pub openrouter_api_key: Option<String>,
    /// `hermes_cli.auth.get_anthropic_key()`.
    pub anthropic_key: Option<String>,

    /// Auth-provider statuses (nous, codex, gemini, minimax).
    pub auth_statuses: Option<AuthProviderStatuses>,

    /// Extra API-key providers contributed by the in-process provider registry.
    pub extra_apikey_providers: Vec<ApiKeyProvider>,

    /// Tool availability (available toolset ids + unavailable entries).
    pub tool_availability: Option<(Vec<String>, Vec<UnavailableTool>)>,

    /// Whether to actually perform network requests for API connectivity.
    pub network_checks: bool,
}

/// Bundle of OAuth/auth provider statuses.
#[derive(Debug, Clone, Default)]
pub struct AuthProviderStatuses {
    pub nous: AuthStatus,
    pub codex: AuthStatus,
    pub gemini: AuthStatus,
    pub minimax: AuthStatus,
}

impl Default for DoctorInputs {
    fn default() -> Self {
        let is_termux = crate_is_termux();
        Self {
            should_fix: false,
            is_termux,
            platform: current_platform().to_string(),
            hermes_home: crate_hermes_home(),
            project_root: env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            display_hermes_home: crate_display_hermes_home(),
            hermes_user_agent: DEFAULT_HERMES_USER_AGENT.to_string(),
            openrouter_api_key: nonempty_env("OPENROUTER_API_KEY"),
            anthropic_key: anthropic_key_from_env(),
            auth_statuses: None,
            extra_apikey_providers: Vec::new(),
            tool_availability: None,
            network_checks: true,
        }
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.is_empty())
}

/// Best-effort `get_anthropic_key()`: prefer `ANTHROPIC_API_KEY`, fall back to
/// `ANTHROPIC_TOKEN`. The full Python helper also consults stored OAuth
/// credentials; native integration may override `DoctorInputs::anthropic_key`.
fn anthropic_key_from_env() -> Option<String> {
    nonempty_env("ANTHROPIC_API_KEY").or_else(|| nonempty_env("ANTHROPIC_TOKEN"))
}

fn crate_is_termux() -> bool {
    // Mirror hermes_constants.is_termux(): TERMUX_VERSION set, or PREFIX path.
    if env::var_os("TERMUX_VERSION").is_some() {
        return true;
    }
    env::var("PREFIX")
        .map(|p| p.contains("com.termux/files/usr"))
        .unwrap_or(false)
}

fn crate_hermes_home() -> PathBuf {
    if let Some(custom) = env::var_os("HERMES_HOME") {
        return PathBuf::from(custom);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
}

fn crate_display_hermes_home() -> String {
    let home = crate_hermes_home();
    if let Some(h) = dirs::home_dir() {
        if let Ok(rel) = home.strip_prefix(&h) {
            if rel.as_os_str().is_empty() {
                return "~".to_string();
            }
            return format!("~/{}", rel.display());
        }
    }
    home.display().to_string()
}

/// Determine whether `is_oauth_token` (mirrors `agent.anthropic_adapter._is_oauth_token`).
pub fn is_oauth_token(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    key.starts_with("sk-ant-oat")
        || key.starts_with("eyJ")
        || key.starts_with("cc-")
        || key.starts_with("oauth")
}

// ---------------------------------------------------------------------------
// HTTP status helpers
// ---------------------------------------------------------------------------

fn blocking_client() -> Option<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the diagnostic checks, printing to stdout. Faithful port of
/// `run_doctor(args)`. Returns the collected (non-fatal) issue strings so the
/// caller can inspect them; the human-readable summary is also printed.
pub fn run_doctor(inputs: &DoctorInputs) -> Vec<String> {
    // Doctor runs from the interactive CLI.
    if env::var_os("HERMES_INTERACTIVE").is_none() {
        unsafe { env::set_var("HERMES_INTERACTIVE", "1") };
    }

    let mut issues: Vec<String> = Vec::new();
    let mut manual_issues: Vec<String> = Vec::new();
    let mut fixed_count: usize = 0;

    let should_fix = inputs.should_fix;
    let dhh = &inputs.display_hermes_home;
    let hermes_home = &inputs.hermes_home;
    let project_root = &inputs.project_root;

    println!();
    println!(
        "{}",
        color(
            "\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}",
            &[Colors::CYAN]
        )
    );
    println!(
        "{}",
        color(
            "\u{2502}                 \u{1fa7a} Hermes Doctor                        \u{2502}",
            &[Colors::CYAN]
        )
    );
    println!(
        "{}",
        color(
            "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}",
            &[Colors::CYAN]
        )
    );

    // --- Python Environment ------------------------------------------------
    // Native build: report the host runtime instead of the interpreter version.
    println!();
    println!("{}", fmt_section("\u{25c6} Runtime Environment"));
    println!("{}", fmt_check_ok(&format!("hermes (native rust)"), ""));

    // --- Required Packages -------------------------------------------------
    // Skipped in native build: these check importability of the Python deps of
    // the legacy interpreter, which do not apply to the Rust binary.

    // --- Configuration Files ----------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Configuration Files"));

    let env_path = hermes_home.join(".env");
    if env_path.exists() {
        println!("{}", fmt_check_ok(&format!("{dhh}/.env file exists"), ""));
        let content = fs::read_to_string(&env_path).unwrap_or_default();
        if has_provider_env_config(&content) {
            println!(
                "{}",
                fmt_check_ok("API key or custom endpoint configured", "")
            );
        } else {
            println!(
                "{}",
                fmt_check_warn(&format!("No API key found in {dhh}/.env"), "")
            );
            issues.push("Run 'hermes setup' to configure API keys".to_string());
        }
    } else {
        let fallback_env = project_root.join(".env");
        if fallback_env.exists() {
            println!(
                "{}",
                fmt_check_ok(".env file exists (in project directory)", "")
            );
        } else {
            println!("{}", fmt_check_fail(&format!("{dhh}/.env file missing"), ""));
            if should_fix {
                if let Some(parent) = env_path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::write(&env_path, b"");
                println!("{}", fmt_check_ok(&format!("Created empty {dhh}/.env"), ""));
                println!("{}", fmt_check_info("Run 'hermes setup' to configure API keys"));
                fixed_count += 1;
            } else {
                println!("{}", fmt_check_info("Run 'hermes setup' to create one"));
                issues.push("Run 'hermes setup' to create .env".to_string());
            }
        }
    }

    let config_path = hermes_home.join("config.yaml");
    if config_path.exists() {
        println!("{}", fmt_check_ok(&format!("{dhh}/config.yaml exists"), ""));
        validate_model_provider_config(&config_path, dhh, &mut issues);
    } else {
        let fallback_config = project_root.join("cli-config.yaml");
        if fallback_config.exists() {
            println!(
                "{}",
                fmt_check_ok("cli-config.yaml exists (in project directory)", "")
            );
        } else {
            let example_config = project_root.join("cli-config.yaml.example");
            if should_fix && example_config.exists() {
                if let Some(parent) = config_path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::copy(&example_config, &config_path);
                println!(
                    "{}",
                    fmt_check_ok(
                        &format!("Created {dhh}/config.yaml from cli-config.yaml.example"),
                        ""
                    )
                );
                fixed_count += 1;
            } else if should_fix {
                println!(
                    "{}",
                    fmt_check_warn("config.yaml not found and no example to copy from", "")
                );
                manual_issues.push(format!("Create {dhh}/config.yaml manually"));
            } else {
                println!(
                    "{}",
                    fmt_check_warn("config.yaml not found", "(using defaults)")
                );
            }
        }
    }

    // Stale root-level config keys.
    if config_path.exists() {
        check_stale_root_keys(&config_path, should_fix, &mut issues, &mut fixed_count);
    }

    // --- Auth Providers ----------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Auth Providers"));
    if let Some(statuses) = &inputs.auth_statuses {
        print_auth_status_ok_warn("Nous Portal auth", statuses.nous.logged_in, "(logged in)", "(not logged in)");
        if statuses.codex.logged_in {
            println!("{}", fmt_check_ok("OpenAI Codex auth", "(logged in)"));
        } else {
            println!("{}", fmt_check_warn("OpenAI Codex auth", "(not logged in)"));
            if let Some(err) = &statuses.codex.error {
                println!("{}", fmt_check_info(err));
            }
        }
        if statuses.gemini.logged_in {
            let mut pieces: Vec<String> = Vec::new();
            if let Some(email) = &statuses.gemini.email {
                if !email.is_empty() {
                    pieces.push(email.clone());
                }
            }
            if let Some(project) = &statuses.gemini.project_id {
                if !project.is_empty() {
                    pieces.push(format!("project={project}"));
                }
            }
            let suffix = if pieces.is_empty() {
                String::new()
            } else {
                format!(" ({})", pieces.join(", "))
            };
            println!(
                "{}",
                fmt_check_ok("Google Gemini OAuth", &format!("(logged in{suffix})"))
            );
        } else {
            println!("{}", fmt_check_warn("Google Gemini OAuth", "(not logged in)"));
        }
        if statuses.minimax.logged_in {
            let region = statuses
                .minimax
                .region
                .clone()
                .unwrap_or_else(|| "global".to_string());
            println!(
                "{}",
                fmt_check_ok("MiniMax OAuth", &format!("(logged in, region={region})"))
            );
        } else {
            println!("{}", fmt_check_warn("MiniMax OAuth", "(not logged in)"));
        }
    } else {
        println!(
            "{}",
            fmt_check_warn("Auth provider status", "(not available in native build)")
        );
    }

    if safe_which("codex").is_some() {
        println!("{}", fmt_check_ok("codex CLI", ""));
    } else {
        println!(
            "{}",
            fmt_check_info(
                "codex CLI not installed (optional — only required to import tokens from an existing Codex CLI login)"
            )
        );
    }

    // --- Directory Structure ----------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Directory Structure"));

    if hermes_home.exists() {
        println!("{}", fmt_check_ok(&format!("{dhh} directory exists"), ""));
    } else if should_fix {
        let _ = fs::create_dir_all(hermes_home);
        println!("{}", fmt_check_ok(&format!("Created {dhh} directory"), ""));
        fixed_count += 1;
    } else {
        println!(
            "{}",
            fmt_check_warn(&format!("{dhh} not found"), "(will be created on first use)")
        );
    }

    let expected_subdirs = ["cron", "sessions", "logs", "skills", "memories"];
    for subdir_name in expected_subdirs {
        let subdir_path = hermes_home.join(subdir_name);
        if subdir_path.exists() {
            println!("{}", fmt_check_ok(&format!("{dhh}/{subdir_name}/ exists"), ""));
        } else if should_fix {
            let _ = fs::create_dir_all(&subdir_path);
            println!("{}", fmt_check_ok(&format!("Created {dhh}/{subdir_name}/"), ""));
            fixed_count += 1;
        } else {
            println!(
                "{}",
                fmt_check_warn(
                    &format!("{dhh}/{subdir_name}/ not found"),
                    "(will be created on first use)"
                )
            );
        }
    }

    // SOUL.md persona.
    let soul_path = hermes_home.join("SOUL.md");
    if soul_path.exists() {
        let content = fs::read_to_string(&soul_path).unwrap_or_default();
        let content = content.trim();
        let lines: Vec<&str> = content
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with("<!--")
                    && !t.starts_with("-->")
                    && !t.starts_with('#')
            })
            .collect();
        if !lines.is_empty() {
            println!(
                "{}",
                fmt_check_ok(&format!("{dhh}/SOUL.md exists (persona configured)"), "")
            );
        } else {
            println!(
                "{}",
                fmt_check_info(&format!(
                    "{dhh}/SOUL.md exists but is empty — edit it to customize personality"
                ))
            );
        }
    } else {
        println!(
            "{}",
            fmt_check_warn(
                &format!("{dhh}/SOUL.md not found"),
                "(create it to give Hermes a custom personality)"
            )
        );
        if should_fix {
            if let Some(parent) = soul_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(
                &soul_path,
                "# Hermes Agent Persona\n\n<!-- Edit this file to customize how Hermes communicates. -->\n\nYou are Hermes, a helpful AI assistant.\n",
            );
            println!(
                "{}",
                fmt_check_ok(&format!("Created {dhh}/SOUL.md with basic template"), "")
            );
            fixed_count += 1;
        }
    }

    // memories directory.
    let memories_dir = hermes_home.join("memories");
    if memories_dir.exists() {
        println!("{}", fmt_check_ok(&format!("{dhh}/memories/ directory exists"), ""));
        let memory_file = memories_dir.join("MEMORY.md");
        let user_file = memories_dir.join("USER.md");
        if memory_file.exists() {
            let size = fs::read_to_string(&memory_file)
                .map(|c| c.trim().chars().count())
                .unwrap_or(0);
            println!("{}", fmt_check_ok(&format!("MEMORY.md exists ({size} chars)"), ""));
        } else {
            println!(
                "{}",
                fmt_check_info("MEMORY.md not created yet (will be created when the agent first writes a memory)")
            );
        }
        if user_file.exists() {
            let size = fs::read_to_string(&user_file)
                .map(|c| c.trim().chars().count())
                .unwrap_or(0);
            println!("{}", fmt_check_ok(&format!("USER.md exists ({size} chars)"), ""));
        } else {
            println!(
                "{}",
                fmt_check_info("USER.md not created yet (will be created when the agent first writes a memory)")
            );
        }
    } else {
        println!(
            "{}",
            fmt_check_warn(&format!("{dhh}/memories/ not found"), "(will be created on first use)")
        );
        if should_fix {
            let _ = fs::create_dir_all(&memories_dir);
            println!("{}", fmt_check_ok(&format!("Created {dhh}/memories/"), ""));
            fixed_count += 1;
        }
    }

    // SQLite session store.
    let state_db_path = hermes_home.join("state.db");
    if state_db_path.exists() {
        match sqlite_session_count(&state_db_path) {
            Ok(count) => {
                println!(
                    "{}",
                    fmt_check_ok(&format!("{dhh}/state.db exists ({count} sessions)"), "")
                );
            }
            Err(e) => {
                println!(
                    "{}",
                    fmt_check_warn(&format!("{dhh}/state.db exists but has issues: {e}"), "")
                );
            }
        }
    } else {
        println!(
            "{}",
            fmt_check_info(&format!("{dhh}/state.db not created yet (will be created on first session)"))
        );
    }

    // WAL file size.
    let wal_path = hermes_home.join("state.db-wal");
    if wal_path.exists() {
        if let Ok(meta) = fs::metadata(&wal_path) {
            let wal_size = meta.len();
            if wal_size > 50 * 1024 * 1024 {
                println!(
                    "{}",
                    fmt_check_warn(
                        &format!("WAL file is large ({} MB)", wal_size / (1024 * 1024)),
                        "(may indicate missed checkpoints)"
                    )
                );
                if should_fix {
                    let _ = sqlite_wal_checkpoint(&state_db_path);
                    let new_size = fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
                    println!(
                        "{}",
                        fmt_check_ok(
                            &format!(
                                "WAL checkpoint performed ({}K → {}K)",
                                wal_size / 1024,
                                new_size / 1024
                            ),
                            ""
                        )
                    );
                    fixed_count += 1;
                } else {
                    issues.push("Large WAL file — run 'hermes doctor --fix' to checkpoint".to_string());
                }
            } else if wal_size > 10 * 1024 * 1024 {
                println!(
                    "{}",
                    fmt_check_info(&format!(
                        "WAL file is {} MB (normal for active sessions)",
                        wal_size / (1024 * 1024)
                    ))
                );
            }
        }
    }

    // --- Command Installation (non-Windows) -------------------------------
    if inputs.platform != "win32" {
        check_command_installation(inputs, &mut issues, &mut manual_issues, &mut fixed_count);
    }

    // --- External Tools ----------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} External Tools"));

    if safe_which("git").is_some() {
        println!("{}", fmt_check_ok("git", ""));
    } else {
        println!("{}", fmt_check_warn("git not found", "(optional)"));
    }

    if safe_which("rg").is_some() {
        println!("{}", fmt_check_ok("ripgrep (rg)", "(faster file search)"));
    } else {
        println!(
            "{}",
            fmt_check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        );
        println!(
            "{}",
            fmt_check_info(&format!(
                "Install for faster search: {}",
                system_package_install_cmd("ripgrep", inputs.is_termux, &inputs.platform)
            ))
        );
    }

    let terminal_env = env::var("TERMINAL_ENV").unwrap_or_else(|_| "local".to_string());

    // Docker.
    if terminal_env == "docker" {
        if safe_which("docker").is_some() {
            let running = docker_daemon_running();
            if running {
                println!("{}", fmt_check_ok("docker", "(daemon running)"));
            } else {
                println!("{}", fmt_check_fail("docker daemon not running", ""));
                issues.push("Start Docker daemon".to_string());
            }
        } else {
            println!(
                "{}",
                fmt_check_fail("docker not found", "(required for TERMINAL_ENV=docker)")
            );
            issues.push("Install Docker or change TERMINAL_ENV".to_string());
        }
    } else if safe_which("docker").is_some() {
        println!("{}", fmt_check_ok("docker", "(optional)"));
    } else if inputs.is_termux {
        println!(
            "{}",
            fmt_check_info("Docker backend is not available inside Termux (expected on Android)")
        );
    } else {
        println!("{}", fmt_check_warn("docker not found", "(optional)"));
    }

    // SSH backend.
    if terminal_env == "ssh" {
        let ssh_host = env::var("TERMINAL_SSH_HOST").unwrap_or_default();
        if !ssh_host.is_empty() {
            let ok = ssh_connect_ok(&ssh_host);
            if ok {
                println!("{}", fmt_check_ok(&format!("SSH connection to {ssh_host}"), ""));
            } else {
                println!("{}", fmt_check_fail(&format!("SSH connection to {ssh_host}"), ""));
                issues.push(format!("Check SSH configuration for {ssh_host}"));
            }
        } else {
            println!(
                "{}",
                fmt_check_fail("TERMINAL_SSH_HOST not set", "(required for TERMINAL_ENV=ssh)")
            );
            issues.push("Set TERMINAL_SSH_HOST in .env".to_string());
        }
    }

    // Daytona backend.
    if terminal_env == "daytona" {
        let daytona_key = env::var("DAYTONA_API_KEY").unwrap_or_default();
        if !daytona_key.is_empty() {
            println!("{}", fmt_check_ok("Daytona API key", "(configured)"));
        } else {
            println!(
                "{}",
                fmt_check_fail("DAYTONA_API_KEY not set", "(required for TERMINAL_ENV=daytona)")
            );
            issues.push("Set DAYTONA_API_KEY environment variable".to_string());
        }
        // The Python `daytona` SDK import-presence check has no native analogue.
    }

    // Vercel Sandbox backend.
    if terminal_env == "vercel_sandbox" {
        check_vercel_sandbox(&mut issues);
    }

    // Node.js + agent-browser.
    if safe_which("node").is_some() {
        println!("{}", fmt_check_ok("Node.js", ""));
        let agent_browser_path = project_root.join("node_modules").join("agent-browser");
        if agent_browser_path.exists() {
            println!("{}", fmt_check_ok("agent-browser (Node.js)", "(browser automation)"));
        } else if safe_which("agent-browser").is_some() {
            println!("{}", fmt_check_ok("agent-browser", "(browser automation)"));
        } else if inputs.is_termux {
            println!(
                "{}",
                fmt_check_info("agent-browser is not installed (expected in the tested Termux path)")
            );
            println!(
                "{}",
                fmt_check_info("Install it manually later with: npm install -g agent-browser && agent-browser install")
            );
            println!("{}", fmt_check_info("Termux browser setup:"));
            for step in termux_browser_setup_steps(true) {
                println!("{}", fmt_check_info(&step));
            }
        } else {
            println!("{}", fmt_check_warn("agent-browser not installed", "(run: npm install)"));
        }
    } else if inputs.is_termux {
        println!(
            "{}",
            fmt_check_info("Node.js not found (browser tools are optional in the tested Termux path)")
        );
        println!("{}", fmt_check_info("Install Node.js on Termux with: pkg install nodejs"));
        println!("{}", fmt_check_info("Termux browser setup:"));
        for step in termux_browser_setup_steps(false) {
            println!("{}", fmt_check_info(&step));
        }
    } else {
        println!(
            "{}",
            fmt_check_warn("Node.js not found", "(optional, needed for browser tools)")
        );
    }

    // npm audit.
    if safe_which("npm").is_some() {
        let npm_dirs = [
            (project_root.clone(), "Browser tools (agent-browser)"),
            (
                project_root.join("scripts").join("whatsapp-bridge"),
                "WhatsApp bridge",
            ),
        ];
        for (npm_dir, label) in npm_dirs {
            if !npm_dir.join("node_modules").exists() {
                continue;
            }
            if let Some((critical, high, moderate)) = npm_audit(&npm_dir) {
                let total = critical + high + moderate;
                if total == 0 {
                    println!("{}", fmt_check_ok(&format!("{label} deps"), "(no known vulnerabilities)"));
                } else if critical > 0 || high > 0 {
                    println!(
                        "{}",
                        fmt_check_warn(
                            &format!("{label} deps"),
                            &format!(
                                "({critical} critical, {high} high, {moderate} moderate — run: cd {} && npm audit fix)",
                                npm_dir.display()
                            )
                        )
                    );
                    issues.push(format!("{label} has {total} npm vulnerability(ies)"));
                } else {
                    println!(
                        "{}",
                        fmt_check_ok(
                            &format!("{label} deps"),
                            &format!("({moderate} moderate vulnerability(ies))")
                        )
                    );
                }
            }
        }
    }

    // --- API Connectivity --------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} API Connectivity"));

    check_openrouter(inputs, &mut issues);
    check_anthropic(inputs);
    check_apikey_providers(inputs, &mut issues);

    // --- Submodules --------------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Submodules"));
    let tinker_dir = project_root.join("tinker-atropos");
    if tinker_dir.exists() && tinker_dir.join("pyproject.toml").exists() {
        // Python-version gating / import check has no native analogue; report presence.
        println!(
            "{}",
            fmt_check_warn(
                "tinker-atropos found",
                "(Python RL backend; install with the Python toolchain)"
            )
        );
    } else {
        println!(
            "{}",
            fmt_check_warn("tinker-atropos not found", "(run: git submodule update --init --recursive)")
        );
    }

    // --- Tool Availability -------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Tool Availability"));
    let kanban_task_set = env::var_os("HERMES_KANBAN_TASK").is_some();
    if let Some((available, unavailable)) = &inputs.tool_availability {
        // honcho_configured is not derivable natively; treat as the runtime fact
        // that an unavailable "honcho" entry should not be promoted unless told.
        let (available, unavailable) =
            apply_doctor_tool_availability_overrides(available, unavailable, kanban_task_set, false);
        for tid in &available {
            let detail = doctor_tool_availability_detail(tid, kanban_task_set);
            println!("{}", fmt_check_ok(tid, detail));
        }
        let mut api_disabled = 0usize;
        for item in &unavailable {
            let env_vars = item.effective_vars();
            if !env_vars.is_empty() {
                api_disabled += 1;
                println!(
                    "{}",
                    fmt_check_warn(&item.name, &format!("(missing {})", env_vars.join(", ")))
                );
            } else {
                println!("{}", fmt_check_warn(&item.name, "(system dependency not met)"));
            }
        }
        if api_disabled > 0 {
            issues.push("Run 'hermes setup' to configure missing API keys for full tool access".to_string());
        }
    } else {
        println!(
            "{}",
            fmt_check_warn("Could not check tool availability", "(not available in native build)")
        );
    }

    // --- Skills Hub --------------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Skills Hub"));
    let hub_dir = hermes_home.join("skills").join(".hub");
    if hub_dir.exists() {
        println!("{}", fmt_check_ok("Skills Hub directory exists", ""));
        let lock_file = hub_dir.join("lock.json");
        if lock_file.exists() {
            match fs::read_to_string(&lock_file)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            {
                Some(v) => {
                    let count = v
                        .get("installed")
                        .and_then(|i| i.as_object())
                        .map(|o| o.len())
                        .unwrap_or(0);
                    println!(
                        "{}",
                        fmt_check_ok(&format!("Lock file OK ({count} hub-installed skill(s))"), "")
                    );
                }
                None => {
                    println!("{}", fmt_check_warn("Lock file", "(corrupted or unreadable)"));
                }
            }
        }
        let quarantine = hub_dir.join("quarantine");
        let q_count = if quarantine.exists() {
            fs::read_dir(&quarantine)
                .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).count())
                .unwrap_or(0)
        } else {
            0
        };
        if q_count > 0 {
            println!(
                "{}",
                fmt_check_warn(&format!("{q_count} skill(s) in quarantine"), "(pending review)")
            );
        }
    } else {
        println!(
            "{}",
            fmt_check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")
        );
    }

    // GitHub token.
    let github_token = nonempty_env("GITHUB_TOKEN").or_else(|| nonempty_env("GH_TOKEN"));
    if github_token.is_some() {
        println!(
            "{}",
            fmt_check_ok("GitHub token configured (authenticated API access)", "")
        );
    } else if gh_authenticated() {
        println!(
            "{}",
            fmt_check_ok("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)")
        );
    } else {
        println!(
            "{}",
            fmt_check_warn(
                "No GITHUB_TOKEN",
                &format!("(60 req/hr rate limit — set in {dhh}/.env for better rates)")
            )
        );
    }

    // --- Memory Provider ---------------------------------------------------
    println!();
    println!("{}", fmt_section("\u{25c6} Memory Provider"));
    let active_memory_provider = read_active_memory_provider(&hermes_home.join("config.yaml"));
    if active_memory_provider.is_empty() {
        println!(
            "{}",
            fmt_check_ok("Built-in memory active", "(no external provider configured — this is fine)")
        );
    } else {
        // Native build cannot load Python memory plugins; report the configured
        // provider name and defer detailed connectivity to the Python runtime.
        println!(
            "{}",
            fmt_check_warn(
                &format!("{active_memory_provider} provider configured"),
                "(connectivity check requires the Python runtime)"
            )
        );
    }

    // --- Summary -----------------------------------------------------------
    println!();
    let mut remaining_issues: Vec<String> = issues.clone();
    remaining_issues.extend(manual_issues.iter().cloned());

    if should_fix && fixed_count > 0 {
        println!("{}", color(&"\u{2500}".repeat(60), &[Colors::GREEN]));
        print!(
            "{}",
            color(&format!("  Fixed {fixed_count} issue(s)."), &[Colors::GREEN, Colors::BOLD])
        );
        if !remaining_issues.is_empty() {
            println!(
                "{}",
                color(
                    &format!(" {} issue(s) require manual intervention.", remaining_issues.len()),
                    &[Colors::YELLOW, Colors::BOLD]
                )
            );
        } else {
            println!();
        }
        println!();
        if !remaining_issues.is_empty() {
            for (i, issue) in remaining_issues.iter().enumerate() {
                println!("  {}. {issue}", i + 1);
            }
            println!();
        }
    } else if !remaining_issues.is_empty() {
        println!("{}", color(&"\u{2500}".repeat(60), &[Colors::YELLOW]));
        println!(
            "{}",
            color(
                &format!("  Found {} issue(s) to address:", remaining_issues.len()),
                &[Colors::YELLOW, Colors::BOLD]
            )
        );
        println!();
        for (i, issue) in remaining_issues.iter().enumerate() {
            println!("  {}. {issue}", i + 1);
        }
        println!();
        if !should_fix {
            println!(
                "{}",
                color("  Tip: run 'hermes doctor --fix' to auto-fix what's possible.", &[Colors::DIM])
            );
        }
    } else {
        println!("{}", color(&"\u{2500}".repeat(60), &[Colors::GREEN]));
        println!(
            "{}",
            color("  All checks passed! \u{1f389}", &[Colors::GREEN, Colors::BOLD])
        );
    }
    println!();

    remaining_issues
}

// ---------------------------------------------------------------------------
// Sub-check helpers
// ---------------------------------------------------------------------------

fn print_auth_status_ok_warn(label: &str, ok: bool, ok_detail: &str, warn_detail: &str) {
    if ok {
        println!("{}", fmt_check_ok(label, ok_detail));
    } else {
        println!("{}", fmt_check_warn(label, warn_detail));
    }
}

/// Validate `model.provider` / `model.default` and detect provider-prefixed
/// slugs on providers that don't accept them. The full Python version consults
/// several provider registries; the native port performs the structural checks
/// it can do from the YAML alone, preserving the warning shapes.
fn validate_model_provider_config(config_path: &Path, dhh: &str, issues: &mut Vec<String>) {
    let raw = match fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "{}",
                fmt_check_warn("Could not validate model/provider config", &format!("({e})"))
            );
            return;
        }
    };
    let cfg: serde_yaml::Value = match serde_yaml::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            println!(
                "{}",
                fmt_check_warn("Could not validate model/provider config", &format!("({e})"))
            );
            return;
        }
    };
    let model_section = cfg.get("model");
    let provider_raw = model_section
        .and_then(|m| m.get("provider"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let provider = provider_raw.to_lowercase();
    let default_model = model_section
        .and_then(|m| m.get("default").or_else(|| m.get("model")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    // Warn if model is a vendor/model slug on a provider that doesn't use them.
    let providers_accepting_vendor_slugs: BTreeSet<&str> = [
        "openrouter",
        "custom",
        "auto",
        "ai-gateway",
        "kilocode",
        "opencode-zen",
        "huggingface",
        "lmstudio",
        "nous",
    ]
    .into_iter()
    .collect();

    if !default_model.is_empty()
        && default_model.contains('/')
        && !provider.is_empty()
        && !providers_accepting_vendor_slugs.contains(provider.as_str())
    {
        println!(
            "{}",
            fmt_check_warn(
                &format!(
                    "model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'"
                ),
                "(vendor-prefixed slugs belong to aggregators like openrouter)"
            )
        );
        issues.push(format!(
            "model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. Either set model.provider to 'openrouter', or drop the vendor prefix."
        ));
    }

    let _ = dhh; // dhh used by callers for credential messages handled in the Python runtime.
}

/// Detect stale root-level `provider`/`base_url` keys (PR #4329 bug source).
fn check_stale_root_keys(
    config_path: &Path,
    should_fix: bool,
    issues: &mut Vec<String>,
    fixed_count: &mut usize,
) {
    let raw = match fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut value: serde_yaml::Value = match serde_yaml::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mapping = match value.as_mapping() {
        Some(m) => m,
        None => return,
    };
    let stale: Vec<&str> = ["provider", "base_url"]
        .into_iter()
        .filter(|k| {
            mapping
                .get(serde_yaml::Value::String(k.to_string()))
                .map(|v| v.is_string())
                .unwrap_or(false)
        })
        .collect();
    if stale.is_empty() {
        return;
    }
    println!(
        "{}",
        fmt_check_warn(
            &format!("Stale root-level config keys: {}", stale.join(", ")),
            "(should be under 'model:' section)"
        )
    );
    if should_fix {
        if let Some(map) = value.as_mapping_mut() {
            // Ensure a model mapping exists.
            let model_key = serde_yaml::Value::String("model".to_string());
            if !map.contains_key(&model_key) {
                map.insert(model_key.clone(), serde_yaml::Value::Mapping(Default::default()));
            }
            for k in &stale {
                let key = serde_yaml::Value::String(k.to_string());
                let popped = map.remove(&key);
                if let Some(popped) = popped {
                    if let Some(serde_yaml::Value::Mapping(model_map)) = map.get_mut(&model_key) {
                        let mk = serde_yaml::Value::String(k.to_string());
                        let existing_empty = model_map
                            .get(&mk)
                            .map(|v| v.as_str().map(|s| s.is_empty()).unwrap_or(false) || v.is_null())
                            .unwrap_or(true);
                        if existing_empty {
                            model_map.insert(mk, popped);
                        }
                    }
                }
            }
        }
        if let Ok(serialized) = serde_yaml::to_string(&value) {
            if atomic_yaml_write(config_path, &serialized).is_ok() {
                println!("{}", fmt_check_ok("Migrated stale root-level keys into model section", ""));
                *fixed_count += 1;
                return;
            }
        }
    }
    issues.push("Stale root-level provider/base_url in config.yaml — run 'hermes doctor --fix'".to_string());
}

fn atomic_yaml_write(path: &Path, contents: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp-doctor");
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn check_command_installation(
    inputs: &DoctorInputs,
    issues: &mut Vec<String>,
    manual_issues: &mut Vec<String>,
    fixed_count: &mut usize,
) {
    let project_root = &inputs.project_root;
    let should_fix = inputs.should_fix;

    println!();
    println!("{}", fmt_section("\u{25c6} Command Installation"));

    let mut venv_bin: Option<PathBuf> = None;
    for venv_name in ["venv", ".venv"] {
        let candidate = project_root.join(venv_name).join("bin").join("hermes");
        if candidate.exists() {
            venv_bin = Some(candidate);
            break;
        }
    }

    let prefix = env::var("PREFIX").unwrap_or_default();
    let is_termux_env =
        env::var_os("TERMUX_VERSION").is_some() || prefix.contains("com.termux/files/usr");
    let (cmd_link_dir, cmd_link_display) = if is_termux_env && !prefix.is_empty() {
        (PathBuf::from(&prefix).join("bin"), "$PREFIX/bin".to_string())
    } else {
        (
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local")
                .join("bin"),
            "~/.local/bin".to_string(),
        )
    };
    let cmd_link = cmd_link_dir.join("hermes");

    let Some(venv_bin) = venv_bin else {
        println!(
            "{}",
            fmt_check_warn(
                "Venv entry point not found",
                "(hermes not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')"
            )
        );
        manual_issues.push(format!(
            "Reinstall entry point: cd {} && source venv/bin/activate && pip install -e '.[all]'",
            project_root.display()
        ));
        return;
    };

    let rel = venv_bin
        .strip_prefix(project_root)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| venv_bin.display().to_string());
    println!("{}", fmt_check_ok(&format!("Venv entry point exists ({rel})"), ""));

    let is_symlink = fs::symlink_metadata(&cmd_link)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if is_symlink {
        let target = fs::canonicalize(&cmd_link).unwrap_or_else(|_| cmd_link.clone());
        let expected = fs::canonicalize(&venv_bin).unwrap_or_else(|_| venv_bin.clone());
        if target == expected {
            println!("{}", fmt_check_ok(&format!("{cmd_link_display}/hermes → correct target"), ""));
        } else {
            println!(
                "{}",
                fmt_check_warn(
                    &format!("{cmd_link_display}/hermes points to wrong target"),
                    &format!("(→ {}, expected → {})", target.display(), expected.display())
                )
            );
            if should_fix {
                let _ = fs::remove_file(&cmd_link);
                let _ = symlink(&venv_bin, &cmd_link);
                println!(
                    "{}",
                    fmt_check_ok(
                        &format!("Fixed symlink: {cmd_link_display}/hermes → {}", venv_bin.display()),
                        ""
                    )
                );
                *fixed_count += 1;
            } else {
                issues.push(format!(
                    "Broken symlink at {cmd_link_display}/hermes — run 'hermes doctor --fix'"
                ));
            }
        }
    } else if cmd_link.exists() {
        println!(
            "{}",
            fmt_check_ok(&format!("{cmd_link_display}/hermes exists (non-symlink)"), "")
        );
    } else {
        println!(
            "{}",
            fmt_check_fail(
                &format!("{cmd_link_display}/hermes not found"),
                "(hermes command may not work outside the venv)"
            )
        );
        if should_fix {
            let _ = fs::create_dir_all(&cmd_link_dir);
            let _ = symlink(&venv_bin, &cmd_link);
            println!(
                "{}",
                fmt_check_ok(
                    &format!("Created symlink: {cmd_link_display}/hermes → {}", venv_bin.display()),
                    ""
                )
            );
            *fixed_count += 1;

            let path_var = env::var("PATH").unwrap_or_default();
            let on_path = env::split_paths(&path_var).any(|d| d == cmd_link_dir);
            if !on_path {
                println!(
                    "{}",
                    fmt_check_warn(
                        &format!("{cmd_link_display} is not on your PATH"),
                        "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")"
                    )
                );
                manual_issues.push(format!("Add {cmd_link_display} to your PATH"));
            }
        } else {
            issues.push(format!(
                "Missing {cmd_link_display}/hermes symlink — run 'hermes doctor --fix'"
            ));
        }
    }
}

#[cfg(unix)]
fn symlink(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dst)
}

#[cfg(not(unix))]
fn symlink(_src: &Path, _dst: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "symlinks unsupported on this platform",
    ))
}

fn check_vercel_sandbox(issues: &mut Vec<String>) {
    let supported_runtimes = ["node24", "node22", "node20", "python3.13", "python3.12"];
    let runtime = env::var("TERMINAL_VERCEL_RUNTIME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node24".to_string());
    if supported_runtimes.contains(&runtime.as_str()) {
        println!("{}", fmt_check_ok("Vercel runtime", &format!("({runtime})")));
    } else {
        let supported = supported_runtimes.join(", ");
        println!(
            "{}",
            fmt_check_fail("Vercel runtime unsupported", &format!("({runtime}; use {supported})"))
        );
        issues.push(format!("Set TERMINAL_VERCEL_RUNTIME to one of: {supported}"));
    }

    let disk = env::var("TERMINAL_CONTAINER_DISK")
        .unwrap_or_else(|_| "51200".to_string())
        .trim()
        .to_string();
    if disk.is_empty() || disk == "0" || disk == "51200" {
        println!("{}", fmt_check_ok("Vercel disk setting", "(uses platform default)"));
    } else {
        println!(
            "{}",
            fmt_check_fail("Vercel custom disk unsupported", "(reset terminal.container_disk to 51200)")
        );
        issues.push(
            "Vercel Sandbox does not support custom container_disk; use the shared default 51200".to_string(),
        );
    }

    // Python `vercel` SDK presence check has no native analogue; skip it.

    let auth_status: VercelAuthStatus = describe_vercel_auth();
    if auth_status.ok {
        println!("{}", fmt_check_ok("Vercel auth", &format!("({})", auth_status.label)));
    } else if auth_status.label.starts_with("partial") {
        println!("{}", fmt_check_fail("Vercel auth incomplete", &format!("({})", auth_status.label)));
        issues.push("Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together".to_string());
    } else {
        println!(
            "{}",
            fmt_check_fail("Vercel auth not configured", &format!("({})", auth_status.label))
        );
        issues.push(
            "Configure Vercel Sandbox auth with VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID".to_string(),
        );
    }
    for line in &auth_status.detail_lines {
        println!("{}", fmt_check_info(&format!("Vercel auth {line}")));
    }

    let persistent = env::var("TERMINAL_CONTAINER_PERSISTENT")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();
    let is_persistent = matches!(persistent.as_str(), "1" | "true" | "yes" | "on");
    if is_persistent {
        println!(
            "{}",
            fmt_check_info(
                "Vercel persistence: snapshot filesystem only; live processes do not survive sandbox recreation"
            )
        );
    } else {
        println!("{}", fmt_check_info("Vercel persistence: ephemeral filesystem"));
    }
}

fn check_openrouter(inputs: &DoctorInputs, issues: &mut Vec<String>) {
    let Some(key) = &inputs.openrouter_api_key else {
        println!("{}", fmt_check_warn("OpenRouter API", "(not configured)"));
        return;
    };
    if !inputs.network_checks {
        println!("{}", fmt_check_info("OpenRouter API check skipped (network disabled)"));
        return;
    }
    let Some(client) = blocking_client() else {
        return;
    };
    match client
        .get(OPENROUTER_MODELS_URL)
        .header("Authorization", format!("Bearer {key}"))
        .send()
    {
        Ok(resp) => {
            let code = resp.status().as_u16();
            match code {
                200 => println!("{}", fmt_check_ok("OpenRouter API", "")),
                401 => {
                    println!("{}", fmt_check_fail("OpenRouter API", "(invalid API key)"));
                    issues.push("Check OPENROUTER_API_KEY in .env".to_string());
                }
                402 => {
                    println!("{}", fmt_check_fail("OpenRouter API", "(out of credits — payment required)"));
                    issues.push(
                        "OpenRouter account has insufficient credits. Fix: run 'hermes config set model.provider <provider>' to switch providers, or fund your OpenRouter account at https://openrouter.ai/settings/credits".to_string(),
                    );
                }
                429 => {
                    println!("{}", fmt_check_fail("OpenRouter API", "(rate limited)"));
                    issues.push(
                        "OpenRouter rate limit hit — consider switching to a different provider or waiting".to_string(),
                    );
                }
                other => {
                    println!("{}", fmt_check_fail("OpenRouter API", &format!("(HTTP {other})")));
                }
            }
        }
        Err(e) => {
            println!("{}", fmt_check_fail("OpenRouter API", &format!("({e})")));
            issues.push("Check network connectivity".to_string());
        }
    }
}

fn check_anthropic(inputs: &DoctorInputs) {
    let Some(key) = &inputs.anthropic_key else {
        return;
    };
    if !inputs.network_checks {
        println!("{}", fmt_check_info("Anthropic API check skipped (network disabled)"));
        return;
    }
    let Some(client) = blocking_client() else {
        return;
    };

    let is_oauth = is_oauth_token(key);
    let send = |beta: Option<String>| -> Result<reqwest::blocking::Response, reqwest::Error> {
        let mut req = client
            .get("https://api.anthropic.com/v1/models")
            .header("anthropic-version", "2023-06-01");
        if is_oauth {
            req = req.header("Authorization", format!("Bearer {key}"));
            let betas = beta.unwrap_or_else(|| {
                COMMON_BETAS
                    .iter()
                    .chain(OAUTH_ONLY_BETAS.iter())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(",")
            });
            req = req.header("anthropic-beta", betas);
        } else {
            req = req.header("x-api-key", key.as_str());
        }
        req.send()
    };

    let resp = send(None);
    let resp = match resp {
        Ok(r) => {
            if is_oauth && r.status().as_u16() == 400 {
                let text = r.text().unwrap_or_default().to_lowercase();
                if text.contains("long context beta") && text.contains("not yet available") {
                    // Retry with the 1M context beta stripped.
                    let betas: Vec<&str> = COMMON_BETAS
                        .iter()
                        .copied()
                        .filter(|b| *b != CONTEXT_1M_BETA)
                        .chain(OAUTH_ONLY_BETAS.iter().copied())
                        .collect();
                    send(Some(betas.join(",")))
                } else {
                    // Re-issue to inspect the original status (response consumed).
                    send(None)
                }
            } else {
                Ok(r)
            }
        }
        Err(e) => Err(e),
    };

    match resp {
        Ok(r) => {
            let code = r.status().as_u16();
            match code {
                200 => println!("{}", fmt_check_ok("Anthropic API", "")),
                401 => println!("{}", fmt_check_fail("Anthropic API", "(invalid API key)")),
                _ => println!("{}", fmt_check_warn("Anthropic API", "(couldn't verify)")),
            }
        }
        Err(e) => {
            println!("{}", fmt_check_warn("Anthropic API", &format!("({e})")));
        }
    }
}

fn check_apikey_providers(inputs: &DoctorInputs, issues: &mut Vec<String>) {
    let mut providers = build_apikey_providers_static();
    providers.extend(inputs.extra_apikey_providers.iter().cloned());

    for provider in &providers {
        let mut key = String::new();
        for ev in &provider.env_vars {
            key = env::var(ev).unwrap_or_default();
            if !key.is_empty() {
                break;
            }
        }
        if key.is_empty() {
            continue;
        }
        let label = pad_right(&provider.name, 20);
        if !provider.supports_health_check {
            println!(
                "  {} {} {}",
                color("\u{2713}", &[Colors::GREEN]),
                label,
                color("(key configured)", &[Colors::DIM])
            );
            continue;
        }
        if !inputs.network_checks {
            println!(
                "  {} {} {}",
                color("\u{2192}", &[Colors::CYAN]),
                label,
                color("(check skipped: network disabled)", &[Colors::DIM])
            );
            continue;
        }
        let base = provider
            .base_env
            .as_ref()
            .map(|e| env::var(e).unwrap_or_default())
            .unwrap_or_default();
        let Some(req) = derive_health_check_request(
            &key,
            &base,
            provider.default_url.as_deref(),
            &inputs.hermes_user_agent,
        ) else {
            continue;
        };
        let Some(client) = blocking_client() else {
            continue;
        };
        match client
            .get(&req.url)
            .header("Authorization", format!("Bearer {key}"))
            .header("User-Agent", req.user_agent.as_str())
            .send()
        {
            Ok(resp) => {
                let code = resp.status().as_u16();
                match code {
                    200 => println!(
                        "  {} {}",
                        color("\u{2713}", &[Colors::GREEN]),
                        label
                    ),
                    401 => {
                        println!(
                            "  {} {} {}",
                            color("\u{2717}", &[Colors::RED]),
                            label,
                            color("(invalid API key)", &[Colors::DIM])
                        );
                        issues.push(format!("Check {} in .env", provider.env_vars[0]));
                    }
                    other => println!(
                        "  {} {} {}",
                        color("\u{26a0}", &[Colors::YELLOW]),
                        label,
                        color(&format!("(HTTP {other})"), &[Colors::DIM])
                    ),
                }
            }
            Err(e) => println!(
                "  {} {} {}",
                color("\u{26a0}", &[Colors::YELLOW]),
                label,
                color(&format!("({e})"), &[Colors::DIM])
            ),
        }
    }
}

fn pad_right(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - len))
    }
}

fn read_active_memory_provider(config_path: &Path) -> String {
    let raw = match fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let cfg: serde_yaml::Value = match serde_yaml::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    cfg.get("memory")
        .and_then(|m| m.get("provider"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn sqlite_session_count(db_path: &Path) -> Result<i64, String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get::<_, i64>(0))
        .map_err(|e| e.to_string())
}

fn sqlite_wal_checkpoint(db_path: &Path) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)")
        .map_err(|e| e.to_string())
}

fn docker_daemon_running() -> bool {
    Command::new("docker")
        .arg("info")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn ssh_connect_ok(host: &str) -> bool {
    Command::new("ssh")
        .args([
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            host,
            "echo ok",
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn gh_authenticated() -> bool {
    Command::new("gh")
        .args(["auth", "status", "--json", "authenticated"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `npm audit --json` in `dir`, returning (critical, high, moderate) counts.
fn npm_audit(dir: &Path) -> Option<(u64, u64, u64)> {
    let output = Command::new("npm")
        .args(["audit", "--json"])
        .current_dir(dir)
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Some((0, 0, 0));
    }
    let data: serde_json::Value = serde_json::from_str(&stdout).ok()?;
    let vulns = data
        .get("metadata")
        .and_then(|m| m.get("vulnerabilities"));
    let get = |k: &str| -> u64 {
        vulns.and_then(|v| v.get(k)).and_then(|v| v.as_u64()).unwrap_or(0)
    };
    Some((get("critical"), get("high"), get("moderate")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_install_cmd_branches() {
        assert_eq!(python_install_cmd(true), "python -m pip install");
        assert_eq!(python_install_cmd(false), "uv pip install");
    }

    #[test]
    fn system_package_install_cmd_branches() {
        assert_eq!(system_package_install_cmd("rg", true, "linux"), "pkg install rg");
        assert_eq!(system_package_install_cmd("rg", false, "darwin"), "brew install rg");
        assert_eq!(system_package_install_cmd("rg", false, "linux"), "sudo apt install rg");
    }

    #[test]
    fn termux_steps_with_node() {
        assert_eq!(
            termux_browser_setup_steps(true),
            vec![
                "1) npm install -g agent-browser".to_string(),
                "2) agent-browser install".to_string(),
            ]
        );
    }

    #[test]
    fn termux_steps_without_node() {
        assert_eq!(
            termux_browser_setup_steps(false),
            vec![
                "1) pkg install nodejs".to_string(),
                "2) npm install -g agent-browser".to_string(),
                "3) agent-browser install".to_string(),
            ]
        );
    }

    #[test]
    fn provider_env_hint_detection() {
        assert!(has_provider_env_config("OPENROUTER_API_KEY=sk-xxx\n"));
        assert!(has_provider_env_config("OPENAI_BASE_URL=https://x\n"));
        assert!(!has_provider_env_config("FOO=bar\n"));
        assert!(!has_provider_env_config(""));
    }

    #[test]
    fn kanban_env_gate() {
        let item = UnavailableTool {
            name: "kanban".to_string(),
            tools: vec!["kanban_create".to_string(), "kanban_list".to_string()],
            ..Default::default()
        };
        assert!(is_kanban_worker_env_gate(&item, false));
        // HERMES_KANBAN_TASK set → not gated.
        assert!(!is_kanban_worker_env_gate(&item, true));
        // Non-kanban tool present → not gated.
        let mixed = UnavailableTool {
            name: "kanban".to_string(),
            tools: vec!["kanban_create".to_string(), "other".to_string()],
            ..Default::default()
        };
        assert!(!is_kanban_worker_env_gate(&mixed, false));
        // Empty tools → not gated.
        let empty = UnavailableTool {
            name: "kanban".to_string(),
            tools: vec![],
            ..Default::default()
        };
        assert!(!is_kanban_worker_env_gate(&empty, false));
    }

    #[test]
    fn overrides_promote_kanban_and_honcho() {
        let available = vec!["files".to_string()];
        let unavailable = vec![
            UnavailableTool {
                name: "kanban".to_string(),
                tools: vec!["kanban_a".to_string()],
                ..Default::default()
            },
            UnavailableTool {
                name: "honcho".to_string(),
                ..Default::default()
            },
            UnavailableTool {
                name: "weather".to_string(),
                env_vars: vec!["WEATHER_KEY".to_string()],
                ..Default::default()
            },
        ];
        let (avail, unavail) =
            apply_doctor_tool_availability_overrides(&available, &unavailable, false, true);
        assert!(avail.contains(&"kanban".to_string()));
        assert!(avail.contains(&"honcho".to_string()));
        assert_eq!(unavail.len(), 1);
        assert_eq!(unavail[0].name, "weather");
    }

    #[test]
    fn overrides_keep_honcho_unavailable_when_not_configured() {
        let available: Vec<String> = vec![];
        let unavailable = vec![UnavailableTool {
            name: "honcho".to_string(),
            ..Default::default()
        }];
        let (avail, unavail) =
            apply_doctor_tool_availability_overrides(&available, &unavailable, false, false);
        assert!(!avail.contains(&"honcho".to_string()));
        assert_eq!(unavail.len(), 1);
    }

    #[test]
    fn effective_vars_prefers_missing() {
        let item = UnavailableTool {
            name: "x".to_string(),
            missing_vars: vec!["A".to_string()],
            env_vars: vec!["B".to_string()],
            ..Default::default()
        };
        assert_eq!(item.effective_vars(), &["A".to_string()]);
        let item2 = UnavailableTool {
            name: "x".to_string(),
            env_vars: vec!["B".to_string()],
            ..Default::default()
        };
        assert_eq!(item2.effective_vars(), &["B".to_string()]);
    }

    #[test]
    fn static_provider_list_shapes() {
        let list = build_apikey_providers_static();
        // 16 static providers, matching the Python list length.
        assert_eq!(list.len(), 16);
        let minimax_cn = list.iter().find(|p| p.name == "MiniMax (China)").unwrap();
        assert!(!minimax_cn.supports_health_check);
        let opencode_go = list.iter().find(|p| p.name == "OpenCode Go").unwrap();
        assert!(!opencode_go.supports_health_check);
        assert!(opencode_go.default_url.is_none());
        let zai = list.iter().find(|p| p.name == "Z.AI / GLM").unwrap();
        assert_eq!(zai.env_vars, vec!["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"]);
    }

    #[test]
    fn health_check_default_url() {
        let req = derive_health_check_request(
            "sk-foo",
            "",
            Some("https://api.z.ai/api/paas/v4/models"),
            "hermes-cli/1",
        )
        .unwrap();
        assert_eq!(req.url, "https://api.z.ai/api/paas/v4/models");
        assert_eq!(req.user_agent, "hermes-cli/1");
    }

    #[test]
    fn health_check_base_override_appends_models() {
        let req = derive_health_check_request("k", "https://example.com/v1", None, "ua").unwrap();
        assert_eq!(req.url, "https://example.com/v1/models");
    }

    #[test]
    fn health_check_kimi_key_autodetect() {
        let req = derive_health_check_request("sk-kimi-abc", "", None, "ua").unwrap();
        assert_eq!(req.url, "https://api.kimi.com/coding/v1/models");
        // api.kimi.com host forces the claude-code user agent.
        assert_eq!(req.user_agent, "claude-code/0.1.0");
    }

    #[test]
    fn health_check_anthropic_compat_rewrite() {
        let req =
            derive_health_check_request("k", "https://host.example/anthropic", None, "ua").unwrap();
        assert_eq!(req.url, "https://host.example/v1/models");
    }

    #[test]
    fn health_check_kimi_coding_appends_v1() {
        let req =
            derive_health_check_request("k", "https://api.kimi.com/coding", None, "ua").unwrap();
        assert_eq!(req.url, "https://api.kimi.com/coding/v1/models");
        assert_eq!(req.user_agent, "claude-code/0.1.0");
    }

    #[test]
    fn health_check_no_url_returns_none() {
        assert!(derive_health_check_request("k", "", None, "ua").is_none());
    }

    #[test]
    fn base_url_host_matches_exact() {
        assert!(base_url_host_matches("https://api.kimi.com/coding", "api.kimi.com"));
        assert!(base_url_host_matches("api.kimi.com", "api.kimi.com"));
        assert!(!base_url_host_matches("https://evil.com/api.kimi.com", "api.kimi.com"));
        assert!(!base_url_host_matches("", "api.kimi.com"));
    }

    #[test]
    fn to_openai_base_url_rewrites_anthropic() {
        assert_eq!(to_openai_base_url("https://x/anthropic"), "https://x/v1");
        assert_eq!(to_openai_base_url("https://x/anthropic/"), "https://x/v1");
        assert_eq!(to_openai_base_url("https://x/v1"), "https://x/v1");
    }

    #[test]
    fn oauth_token_detection() {
        assert!(!is_oauth_token(""));
        assert!(!is_oauth_token("sk-ant-api03-xxx"));
        assert!(is_oauth_token("sk-ant-oat01-xxx"));
        assert!(is_oauth_token("eyJabc"));
        assert!(is_oauth_token("cc-token"));
    }

    #[test]
    fn pad_right_pads_and_truncates_correctly() {
        assert_eq!(pad_right("Z.AI", 8), "Z.AI    ");
        assert_eq!(pad_right("exactly-twenty-charss", 5), "exactly-twenty-charss");
    }

    #[test]
    fn safe_which_resolves_known_tool() {
        // `sh` exists on every unix CI host; on others this just asserts None-safety.
        if cfg!(unix) {
            assert!(safe_which("sh").is_some());
        }
        assert!(safe_which("definitely-not-a-real-binary-xyz123").is_none());
    }

    #[test]
    fn current_platform_is_known() {
        let p = current_platform();
        assert!(["darwin", "win32", "linux", "unknown"].contains(&p));
    }
}
