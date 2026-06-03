use std::env;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

pub mod agent;
mod agent_file_safety;
mod agent_redact;
mod agent_retry;
mod agent_trajectory;
mod anthropic_caching;
mod approvals;
mod auth;
mod browser;
mod checkpoints;
mod clarify;
pub mod clipboard;
mod code_execution;
pub mod commands;
mod commands_registry_data;
mod compression_feedback;
pub mod config;
mod context_engine;
mod cronjob;
pub mod delegate;
mod discord;
mod error_classifier;
pub mod env_loader;
mod feishu;
mod gemini_schema;
mod gateway_events;
pub mod gateway;
pub mod gateway_display_config;
pub mod gateway_footer;
pub mod gateway_mirror;
pub mod gateway_sticker_cache;
pub mod gateway_whatsapp_identity;
mod homeassistant;
mod i18n;
mod image_gen;
mod kanban;
mod lmstudio_reasoning;
pub mod logging;
mod memory;
mod moa;
mod moonshot_schema;
mod plugin_runtime;
pub mod providers;
pub mod pyhost;
mod rl;
mod send_message;
mod shell_hooks;
mod skills;
pub mod skins;
mod skins_data;
pub mod spawn_tree;
pub mod tui_config;
mod spotify;
pub mod state;
mod terminal;
mod think_scrubber;
pub mod tools;
mod tts;
mod turn_runner;
mod video;
mod vision;
mod web;
mod yuanbao;

pub use agent::{
    AgentInterruptController, AgentInterruptPhase, AgentTurnEvent, AgentTurnOptions,
    AgentTurnResult,
};
pub use approvals::{
    ApprovalCheckResult, ApprovalManager, ApprovalRequest, shell_command_block_reason,
};
pub use auth::{
    AuthStatusSummary, CopilotAcpRuntimeCredentials, CopilotRuntimeCredentials,
    GoogleGeminiRuntimeCredentials, MinimaxOAuthRuntimeCredentials, NousRuntimeCredentials,
    QwenRuntimeCredentials, anthropic_base_url_supports_oauth, anthropic_oauth_default_headers,
    anthropic_token_is_oauth, clear_provider_auth_state, clear_provider_runtime_cache,
    codex_cloudflare_headers, force_refresh_anthropic_token,
    force_refresh_codex_access_token, force_refresh_codex_credential_pool_entry,
    force_refresh_google_gemini_runtime_credentials,
    force_refresh_minimax_oauth_runtime_credentials, force_refresh_nous_credential_pool_entry,
    force_refresh_nous_runtime_credentials, force_refresh_qwen_runtime_credentials,
    get_active_auth_provider, get_auth_status_summary, resolve_anthropic_token,
    resolve_codex_access_token, resolve_copilot_acp_runtime_credentials,
    resolve_copilot_runtime_credentials, resolve_google_gemini_runtime_credentials,
    resolve_nous_access_token,
    resolve_minimax_oauth_runtime_credentials, resolve_nous_runtime_credentials,
    resolve_qwen_runtime_credentials,
};
pub use config::{
    AgentConfig, CompressionConfig, DelegationConfig, DisplayConfig, FallbackProviderConfig,
    HermesConfig, LoadedConfig, LoggingConfig, MemoryConfig, ModelOverrides, ModelRuntimeConfig,
    NetworkConfig, SecurityConfig, TerminalConfig,
};
pub use clipboard::{
    has_clipboard_image, image_dimensions, image_token_estimate, save_clipboard_image,
};
pub use commands::{
    CommandDef, DispatchOutcome, HERMES_RELEASE_DATE, command_dispatch, command_registry,
    commands_catalog, complete_slash, initial_session_info, read_quick_commands, resolve_command,
    resolve_command_json, resolve_tui_model,
};
pub use context_engine::{ContextEngine, ContextEngineSessionStart};
pub use cronjob::{
    CronRunResult, CronTickResult, SILENT_MARKER, handle_cronjob, run_cron_job_now,
    run_due_cron_jobs,
};
pub use delegate::DelegateExecutor;
pub use env_loader::EnvLoadReport;
pub use gateway_events::{
    GatewayApprovalPrompt, GatewayClarifyPrompt, GatewayEventBridge, GatewayEventEnvelope,
    GatewaySessionPoll, GatewayTurnBridge, GatewayTurnOutcome, GatewayTurnSession,
    attach_gateway_event_callbacks,
};
pub use kanban::{
    Comment as KanbanComment, CreateTaskInput as KanbanCreateTaskInput, Event as KanbanEvent,
    KanbanAssigneeRecord, KanbanBoardRecord, KanbanBoardRemoval, KanbanBoardStats,
    KanbanDispatchOptions, KanbanDispatchResult, KanbanNotifySubscription, KanbanRunResult,
    KanbanTaskDetail, KanbanTaskQuery, Run as KanbanRun, Task as KanbanTask, VALID_KANBAN_STATUSES,
    VALID_WORKSPACE_KINDS, add_comment, add_notify_sub, archive_task, assign_task, block_task,
    board_stats, build_worker_context, child_ids, claim_task, clear_current_kanban_board,
    complete_task, create_kanban_board, create_task, current_kanban_board, dispatch_kanban_once,
    edit_completed_task_result, gc_events, gc_worker_logs, get_task, heartbeat_worker,
    kanban_board_exists, kanban_db_path_for_home, kanban_has_spawnable_ready, kanban_task_detail,
    known_assignees, latest_run, latest_summary, link_tasks, list_comments, list_events,
    list_kanban_boards, list_notify_subs, list_runs, list_tasks, open_kanban_db, parent_ids,
    read_worker_log, reassign_task, reclaim_task, recompute_ready, release_stale_claims,
    remove_kanban_board, remove_notify_sub, rename_kanban_board, run_kanban_task,
    set_current_kanban_board, unblock_task, unlink_tasks, worker_log_path_for_task,
};
pub use logging::{
    LoggingMode, LoggingSetup, clear_session_context, enable_verbose_logging, set_session_context,
};
pub use memory::build_memory_context_block;
pub use plugin_runtime::{
    DashboardSurface, DiscoveredPlugin, ImageGenPluginProvider, ImageGenProviderEnvVar,
    ImageGenProviderModel, PlatformSurface, PluginCliCommand, PluginCliDispatchResult, PluginKind,
    PluginSource, attach_python_plugin_callbacks, attach_python_plugin_runtime,
    discover_context_engine_plugins, discover_dashboard_surfaces, discover_enabled_general_plugins,
    discover_enabled_plugin_cli_commands, discover_enabled_plugin_platforms,
    discover_general_plugins, discover_hook_registrations_from_source,
    discover_memory_provider_plugins, discover_platform_surfaces_from_source,
    discover_plugin_cli_commands_from_source, discover_plugin_image_gen_providers,
    discover_scanned_plugins, discover_tool_definitions_from_source,
    dispatch_python_plugin_cli_command, dispatch_python_plugin_image_generate,
    is_effectively_enabled, plugins_list, PluginListing, python_plugin_image_gen_available,
    run_python_plugin_platform_setup,
};
pub use providers::{
    ProviderProfile, auto_provider_candidates, get_provider_profile, infer_api_mode_from_base_url,
    infer_provider_from_base_url, list_provider_profiles, normalize_model_for_provider,
    normalize_provider_alias, resolve_provider_api_mode,
};
pub use skills::load_skill_prompt_content;
pub use skills::build_skill_invocation_message;
pub use skills::{SkillCommand, scan_skill_commands};
pub use skins::{SkinConfig, load_skin, resolve_skin};
pub use state::{
    ExportedSession, MessageAppend, MessageRecord, MessageSearchRow, SearchContextMessage,
    SessionCreate, SessionRecord, SessionSearchRow, SessionStore, SessionSummary,
    SessionTruncateResult, SessionUsageDelta, SessionUsageRecord,
};
pub use tools::{
    ClarifyRequest, StepToolRecord, StepUpdate, ToolDefinition, ToolProgressUpdate, ToolRuntime,
    ToolsetInfo, coerce_tool_args, dispatch_tool, get_all_tool_names, get_all_toolsets,
    get_tool_definitions, get_tool_definitions_for_runtime, get_tool_definitions_with_runtime,
    get_toolset_for_tool, get_toolset_info, get_toolset_names, resolve_toolset, tool_error,
    tool_result, validate_toolset,
};
pub use turn_runner::{
    InteractiveTurnEvent, InteractiveTurnOptions, InteractiveTurnRequest,
    spawn_chat_turn_with_events,
};

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
pub const AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";
pub const VALID_REASONING_EFFORTS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];
pub const PROFILE_DIRS: [&str; 9] = [
    "memories",
    "sessions",
    "skills",
    "skins",
    "logs",
    "plans",
    "workspace",
    "cron",
    "home",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningConfig {
    Disabled,
    Enabled(ReasoningEffort),
}

#[derive(Debug)]
pub enum HermesError {
    InvalidProfileName(String),
    MissingProfile(String),
    ProfileAlreadyExists {
        name: String,
        path: PathBuf,
    },
    CannotCreateDefaultProfile,
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    State {
        action: &'static str,
        detail: String,
    },
}

impl Display for HermesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfileName(name) => write!(
                f,
                "Invalid profile name {name:?}. Must match [a-z0-9][a-z0-9_-]{{0,63}}"
            ),
            Self::MissingProfile(name) => write!(
                f,
                "Profile '{name}' does not exist. Create it with: hermes profile create {name}"
            ),
            Self::ProfileAlreadyExists { name, path } => {
                write!(f, "Profile '{name}' already exists at {}", path.display())
            }
            Self::CannotCreateDefaultProfile => write!(
                f,
                "Cannot create a profile named 'default' - it is the built-in profile (~/.hermes)."
            ),
            Self::Io {
                action,
                path,
                source,
            } => write!(f, "{action} {} failed: {source}", path.display()),
            Self::State { action, detail } => write!(f, "{action}: {detail}"),
        }
    }
}

impl Error for HermesError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
pub(crate) fn test_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileOverride {
    pub args: Vec<String>,
    pub profile_name: Option<String>,
    pub hermes_home: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermesContext {
    home_dir: PathBuf,
    hermes_home_env: Option<PathBuf>,
    optional_skills_env: Option<PathBuf>,
    prefix: Option<String>,
    termux_version: Option<String>,
}

impl HermesContext {
    pub fn detect() -> Self {
        Self {
            home_dir: dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")),
            hermes_home_env: env::var_os("HERMES_HOME").and_then(path_from_os),
            optional_skills_env: env::var_os("HERMES_OPTIONAL_SKILLS").and_then(path_from_os),
            prefix: env::var("PREFIX").ok().and_then(non_empty_string),
            termux_version: env::var("TERMUX_VERSION").ok().and_then(non_empty_string),
        }
    }

    pub fn new(home_dir: impl Into<PathBuf>) -> Self {
        Self {
            home_dir: home_dir.into(),
            hermes_home_env: None,
            optional_skills_env: None,
            prefix: None,
            termux_version: None,
        }
    }

    pub fn with_hermes_home_env(mut self, value: Option<PathBuf>) -> Self {
        self.hermes_home_env = value;
        self
    }

    pub fn with_optional_skills_env(mut self, value: Option<PathBuf>) -> Self {
        self.optional_skills_env = value;
        self
    }

    pub fn with_prefix(mut self, value: Option<String>) -> Self {
        self.prefix = value.and_then(non_empty_string);
        self
    }

    pub fn with_termux_version(mut self, value: Option<String>) -> Self {
        self.termux_version = value.and_then(non_empty_string);
        self
    }

    pub fn home_dir(&self) -> &Path {
        &self.home_dir
    }

    pub fn hermes_home(&self) -> PathBuf {
        self.hermes_home_env
            .clone()
            .unwrap_or_else(|| self.home_dir.join(".hermes"))
    }

    pub fn default_hermes_root(&self) -> PathBuf {
        let native_home = self.home_dir.join(".hermes");
        let Some(env_home) = self.hermes_home_env.as_ref() else {
            return native_home;
        };

        if env_home.starts_with(&native_home) {
            return native_home;
        }

        if env_home
            .parent()
            .and_then(|parent| parent.file_name())
            .is_some_and(|name| name == "profiles")
        {
            if let Some(root) = env_home.parent().and_then(Path::parent) {
                return root.to_path_buf();
            }
        }

        env_home.clone()
    }

    pub fn active_profile_path(&self) -> PathBuf {
        self.default_hermes_root().join("active_profile")
    }

    pub fn active_profile(&self) -> String {
        read_trimmed(&self.active_profile_path()).unwrap_or_else(|| "default".to_string())
    }

    pub fn profile_fallback_warning(&self) -> Option<String> {
        if self.hermes_home_env.is_some() {
            return None;
        }

        let active = self.active_profile();
        if active.is_empty() || active == "default" {
            return None;
        }

        Some(format!(
            "[HERMES_HOME fallback] HERMES_HOME is unset but active profile is {active:?}. Falling back to ~/.hermes, which is the DEFAULT profile - not {active:?}. Any data this process writes will land in the wrong profile. The subprocess spawner should pass HERMES_HOME explicitly (see issue #18594)."
        ))
    }

    pub fn profiles_root(&self) -> PathBuf {
        self.default_hermes_root().join("profiles")
    }

    pub fn profile_dir(&self, name: &str) -> Result<PathBuf, HermesError> {
        let canon = normalize_profile_name(name)?;
        if canon == "default" {
            return Ok(self.default_hermes_root());
        }
        Ok(self.profiles_root().join(canon))
    }

    pub fn profile_exists(&self, name: &str) -> bool {
        match normalize_profile_name(name) {
            Ok(canon) if canon == "default" => true,
            Ok(canon) => self.profiles_root().join(canon).is_dir(),
            Err(_) => false,
        }
    }

    pub fn resolve_profile_env(&self, profile_name: &str) -> Result<PathBuf, HermesError> {
        let canon = normalize_profile_name(profile_name)?;
        let profile_dir = self.profile_dir(&canon)?;

        if canon != "default" && !profile_dir.is_dir() {
            return Err(HermesError::MissingProfile(canon));
        }

        Ok(profile_dir)
    }

    pub fn create_profile(&self, profile_name: &str) -> Result<PathBuf, HermesError> {
        let canon = normalize_profile_name(profile_name)?;
        validate_profile_name(&canon)?;

        if canon == "default" {
            return Err(HermesError::CannotCreateDefaultProfile);
        }

        let profile_dir = self.profile_dir(&canon)?;
        if profile_dir.exists() {
            return Err(HermesError::ProfileAlreadyExists {
                name: canon,
                path: profile_dir,
            });
        }

        self.create_dir_all(&profile_dir)?;
        for subdir in PROFILE_DIRS {
            self.create_dir_all(&profile_dir.join(subdir))?;
        }
        Ok(profile_dir)
    }

    pub fn set_active_profile(&self, profile_name: &str) -> Result<(), HermesError> {
        let canon = normalize_profile_name(profile_name)?;
        validate_profile_name(&canon)?;

        if canon != "default" && !self.profile_exists(&canon) {
            return Err(HermesError::MissingProfile(canon));
        }

        let path = self.active_profile_path();
        if let Some(parent) = path.parent() {
            self.create_dir_all(parent)?;
        }

        if canon == "default" {
            if let Err(source) = fs::remove_file(&path) {
                if source.kind() != io::ErrorKind::NotFound {
                    return Err(HermesError::Io {
                        action: "removing",
                        path,
                        source,
                    });
                }
            }
            return Ok(());
        }

        let tmp = path.with_extension("tmp");
        fs::write(&tmp, format!("{canon}\n")).map_err(|source| HermesError::Io {
            action: "writing",
            path: tmp.clone(),
            source,
        })?;
        fs::rename(&tmp, &path).map_err(|source| HermesError::Io {
            action: "renaming",
            path: path.clone(),
            source,
        })?;
        Ok(())
    }

    pub fn current_profile_name(&self) -> String {
        let hermes_home = self.hermes_home();
        let default_root = self.default_hermes_root();
        if hermes_home == default_root {
            return "default".to_string();
        }

        let profiles_root = self.profiles_root();
        if let Ok(relative) = hermes_home.strip_prefix(&profiles_root) {
            let mut parts = relative.components();
            if let Some(first) = parts.next() {
                let candidate = first.as_os_str().to_string_lossy();
                if parts.next().is_none() && is_valid_profile_id(&candidate) {
                    return candidate.to_string();
                }
            }
        }

        "custom".to_string()
    }

    pub fn optional_skills_dir(&self, default: Option<&Path>) -> PathBuf {
        if let Some(override_dir) = self.optional_skills_env.as_ref() {
            return override_dir.clone();
        }
        if let Some(default_dir) = default {
            return default_dir.to_path_buf();
        }
        self.hermes_home().join("optional-skills")
    }

    pub fn hermes_dir(&self, new_subpath: &str, old_name: &str) -> PathBuf {
        let home = self.hermes_home();
        let old_path = home.join(old_name);
        if old_path.exists() {
            return old_path;
        }
        home.join(new_subpath)
    }

    pub fn display_hermes_home(&self) -> String {
        match self.hermes_home().strip_prefix(&self.home_dir) {
            Ok(relative) => format!("~/{}", relative.display()),
            Err(_) => self.hermes_home().display().to_string(),
        }
    }

    pub fn subprocess_home(&self) -> Option<PathBuf> {
        let env_home = self.hermes_home_env.as_ref()?;
        let profile_home = env_home.join("home");
        profile_home.is_dir().then_some(profile_home)
    }

    pub fn config_path(&self) -> PathBuf {
        self.hermes_home().join("config.yaml")
    }

    pub fn skills_dir(&self) -> PathBuf {
        self.hermes_home().join("skills")
    }

    pub fn env_path(&self) -> PathBuf {
        self.hermes_home().join(".env")
    }

    pub fn is_termux(&self) -> bool {
        self.termux_version.is_some()
            || self
                .prefix
                .as_deref()
                .is_some_and(|prefix| prefix.contains("com.termux/files/usr"))
    }

    pub fn apply_profile_override(&self, args: &[String]) -> Result<ProfileOverride, HermesError> {
        let explicit_flag = extract_profile_flag(args);

        if let Some(flag) = explicit_flag.as_ref() {
            if flag.separate_value && !looks_like_profile_flag_value(&flag.value) {
                return self.apply_sticky_profile(args);
            }
        }

        if let Some(flag) = explicit_flag {
            let canon = normalize_profile_name(&flag.value)?;
            let hermes_home = self.resolve_profile_env(&canon)?;
            let mut stripped = args.to_vec();
            stripped.drain(flag.start..flag.start + flag.len);
            return Ok(ProfileOverride {
                args: stripped,
                profile_name: Some(canon),
                hermes_home: Some(hermes_home),
            });
        }

        if self.hermes_home_env.is_some() {
            return Ok(ProfileOverride {
                args: args.to_vec(),
                profile_name: None,
                hermes_home: None,
            });
        }

        self.apply_sticky_profile(args)
    }

    fn apply_sticky_profile(&self, args: &[String]) -> Result<ProfileOverride, HermesError> {
        let active = self.active_profile();
        if active.is_empty() || active == "default" {
            return Ok(ProfileOverride {
                args: args.to_vec(),
                profile_name: None,
                hermes_home: None,
            });
        }

        let canon = normalize_profile_name(&active)?;
        let hermes_home = self.resolve_profile_env(&canon)?;
        Ok(ProfileOverride {
            args: args.to_vec(),
            profile_name: Some(canon),
            hermes_home: Some(hermes_home),
        })
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), HermesError> {
        fs::create_dir_all(path).map_err(|source| HermesError::Io {
            action: "creating",
            path: path.to_path_buf(),
            source,
        })
    }
}

/// Return true when at least one inference provider is usable. Port of
/// `hermes_cli.main._has_any_provider_configured` (and parity with the
/// `hermes` crate's `has_any_chat_provider_configured`): checks configured
/// model api_key/base_url, provider env vars (process + `.env`), the active
/// auth provider, and non-api-key provider auth status.
///
/// Used by the TUI gateway `setup.status` RPC. Best-effort: any error loading
/// config/auth resolves to `false` rather than propagating.
pub fn provider_configured(context: &HermesContext) -> bool {
    // Whether Hermes itself has been explicitly configured (a non-empty model
    // name). The Python default model is "", so any configured model counts.
    // Gates the external-tool (Claude Code) credential fallback below.
    let mut has_hermes_config = false;

    if let Ok(loaded) = context.load_config_document() {
        if loaded
            .configured_model_name()
            .is_some_and(|name| !name.trim().is_empty())
        {
            has_hermes_config = true;
        }
        if loaded.configured_model_api_key().is_some()
            || loaded.configured_model_base_url().is_some()
        {
            return true;
        }
    }

    for key in provider_env_keys() {
        if env_value_for_context(context, key).is_some() {
            return true;
        }
    }

    if let Ok(Some(active)) = get_active_auth_provider(context.hermes_home().as_path()) {
        if is_inference_auth_provider(&active) {
            return true;
        }
    }

    for profile in list_provider_profiles() {
        if profile.auth_type == "api_key" {
            continue;
        }
        if let Ok(status) = get_auth_status_summary(context.hermes_home().as_path(), profile.name) {
            if status.configured || status.logged_in {
                return true;
            }
        }
    }

    // Claude Code OAuth credentials only count when Hermes has been explicitly
    // configured (mirrors the _has_hermes_config gate in the Python check) —
    // having Claude Code installed doesn't mean the user wants Hermes to use it.
    if has_hermes_config && auth::claude_code_credentials_present() {
        return true;
    }

    false
}

fn provider_env_keys() -> Vec<&'static str> {
    let mut keys = vec![
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    ];
    for profile in list_provider_profiles() {
        for key in profile.env_vars {
            if !keys.contains(key) {
                keys.push(key);
            }
        }
    }
    keys
}

fn is_inference_auth_provider(provider: &str) -> bool {
    let provider = provider.trim();
    !provider.is_empty()
        && list_provider_profiles()
            .iter()
            .any(|profile| profile.name == provider)
}

fn env_value_for_context(context: &HermesContext, key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| read_env_file_value(context.env_path().as_path(), key))
}

fn read_env_file_value(path: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((entry_key, entry_value)) = trimmed.split_once('=') else {
            continue;
        };
        if entry_key.trim() != key {
            continue;
        }
        let value = entry_value.trim().trim_matches(['"', '\'']);
        if value.is_empty() {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProfileFlag {
    value: String,
    start: usize,
    len: usize,
    separate_value: bool,
}

pub fn normalize_profile_name(name: &str) -> Result<String, HermesError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(HermesError::InvalidProfileName(trimmed.to_string()));
    }
    if trimmed.eq_ignore_ascii_case("default") {
        return Ok("default".to_string());
    }
    Ok(trimmed.to_ascii_lowercase())
}

pub fn validate_profile_name(name: &str) -> Result<(), HermesError> {
    if name == "default" {
        return Ok(());
    }
    if is_valid_profile_id(name) {
        return Ok(());
    }
    Err(HermesError::InvalidProfileName(name.to_string()))
}

pub fn parse_reasoning_effort(effort: &str) -> Option<ReasoningConfig> {
    let normalized = effort.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized == "none" {
        return Some(ReasoningConfig::Disabled);
    }
    let config = match normalized.as_str() {
        "minimal" => ReasoningConfig::Enabled(ReasoningEffort::Minimal),
        "low" => ReasoningConfig::Enabled(ReasoningEffort::Low),
        "medium" => ReasoningConfig::Enabled(ReasoningEffort::Medium),
        "high" => ReasoningConfig::Enabled(ReasoningEffort::High),
        "xhigh" => ReasoningConfig::Enabled(ReasoningEffort::XHigh),
        _ => return None,
    };
    Some(config)
}

pub fn is_wsl() -> bool {
    fs::read_to_string("/proc/version")
        .map(|content| content.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

pub fn is_container() -> bool {
    if Path::new("/.dockerenv").exists() || Path::new("/run/.containerenv").exists() {
        return true;
    }

    fs::read_to_string("/proc/1/cgroup")
        .map(|content| {
            let lowered = content.to_ascii_lowercase();
            lowered.contains("docker") || lowered.contains("podman") || lowered.contains("/lxc/")
        })
        .unwrap_or(false)
}

fn extract_profile_flag(args: &[String]) -> Option<ProfileFlag> {
    for (index, arg) in args.iter().enumerate() {
        if (arg == "--profile" || arg == "-p") && index + 1 < args.len() {
            return Some(ProfileFlag {
                value: args[index + 1].clone(),
                start: index,
                len: 2,
                separate_value: true,
            });
        }
        if let Some(value) = arg.strip_prefix("--profile=") {
            return Some(ProfileFlag {
                value: value.to_string(),
                start: index,
                len: 1,
                separate_value: false,
            });
        }
    }
    None
}

fn looks_like_profile_flag_value(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("default") {
        return true;
    }
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn is_valid_profile_id(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }

    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }

    bytes.iter().copied().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
    })
}

fn path_from_os(value: std::ffi::OsString) -> Option<PathBuf> {
    let path = PathBuf::from(value);
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(path)
}

fn non_empty_string(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().and_then(|content| {
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().expect("tempdir");
        let home = temp.path().join("home");
        fs::create_dir_all(&home).expect("home dir");
        (temp, HermesContext::new(home))
    }

    #[test]
    fn hermes_home_defaults_to_home_dot_hermes() {
        let (_temp, ctx) = test_context();
        assert_eq!(ctx.hermes_home(), ctx.home_dir().join(".hermes"));
    }

    #[test]
    fn default_root_uses_profile_parent_for_custom_profiles() {
        let (_temp, ctx) = test_context();
        let ctx = ctx.with_hermes_home_env(Some(PathBuf::from("/opt/hermes/profiles/coder")));
        assert_eq!(ctx.default_hermes_root(), PathBuf::from("/opt/hermes"));
    }

    #[test]
    fn fallback_warning_appears_when_active_profile_is_sticky_non_default() {
        let (_temp, ctx) = test_context();
        let active_path = ctx.home_dir().join(".hermes").join("active_profile");
        fs::create_dir_all(active_path.parent().expect("parent")).expect("mkdir");
        fs::write(&active_path, "coder\n").expect("write");
        assert!(
            ctx.profile_fallback_warning()
                .expect("warning")
                .contains("active profile is \"coder\"")
        );
    }

    #[test]
    fn resolve_profile_env_requires_existing_named_profile() {
        let (_temp, ctx) = test_context();
        let err = ctx
            .resolve_profile_env("coder")
            .expect_err("missing profile");
        assert_eq!(
            err.to_string(),
            "Profile 'coder' does not exist. Create it with: hermes profile create coder"
        );
    }

    #[test]
    fn apply_profile_override_strips_explicit_flag() {
        let (_temp, ctx) = test_context();
        let profile_dir = ctx
            .home_dir()
            .join(".hermes")
            .join("profiles")
            .join("coder");
        fs::create_dir_all(&profile_dir).expect("profile dir");
        let args = vec!["-p".to_string(), "coder".to_string(), "chat".to_string()];
        let resolved = ctx.apply_profile_override(&args).expect("override");
        assert_eq!(resolved.profile_name.as_deref(), Some("coder"));
        assert_eq!(resolved.hermes_home.as_deref(), Some(profile_dir.as_path()));
        assert_eq!(resolved.args, vec!["chat".to_string()]);
    }

    #[test]
    fn apply_profile_override_ignores_non_profile_short_flag_values() {
        let (_temp, ctx) = test_context();
        let args = vec!["-p".to_string(), "no:xdist".to_string(), "chat".to_string()];
        let resolved = ctx.apply_profile_override(&args).expect("override");
        assert_eq!(resolved.profile_name, None);
        assert_eq!(resolved.hermes_home, None);
        assert_eq!(resolved.args, args);
    }

    #[test]
    fn current_profile_name_detects_named_profile() {
        let (_temp, ctx) = test_context();
        let profile_dir = ctx
            .home_dir()
            .join(".hermes")
            .join("profiles")
            .join("coder");
        fs::create_dir_all(&profile_dir).expect("profile dir");
        let ctx = ctx.with_hermes_home_env(Some(profile_dir));
        assert_eq!(ctx.current_profile_name(), "coder");
    }

    #[test]
    fn display_hermes_home_uses_tilde_prefix() {
        let (_temp, ctx) = test_context();
        assert_eq!(ctx.display_hermes_home(), "~/.hermes");
    }

    #[test]
    fn parse_reasoning_effort_matches_python_variants() {
        assert_eq!(
            parse_reasoning_effort("high"),
            Some(ReasoningConfig::Enabled(ReasoningEffort::High))
        );
        assert_eq!(
            parse_reasoning_effort("none"),
            Some(ReasoningConfig::Disabled)
        );
        assert_eq!(parse_reasoning_effort(""), None);
        assert_eq!(parse_reasoning_effort("unknown"), None);
    }

    #[test]
    fn termux_detection_checks_both_signals() {
        let (_temp, ctx) = test_context();
        assert!(!ctx.is_termux());
        let ctx = ctx.with_prefix(Some("/data/data/com.termux/files/usr".to_string()));
        assert!(ctx.is_termux());
    }

    #[test]
    fn create_profile_bootstraps_expected_directories() {
        let (_temp, ctx) = test_context();
        let profile_dir = ctx.create_profile("coder").expect("create profile");
        for subdir in PROFILE_DIRS {
            assert!(profile_dir.join(subdir).is_dir(), "missing {subdir}");
        }
    }

    #[test]
    fn set_active_profile_writes_and_clears_sticky_selection() {
        let (_temp, ctx) = test_context();
        ctx.create_profile("coder").expect("create profile");
        ctx.set_active_profile("coder").expect("set profile");
        assert_eq!(ctx.active_profile(), "coder");
        ctx.set_active_profile("default").expect("clear profile");
        assert_eq!(ctx.active_profile(), "default");
    }
}
