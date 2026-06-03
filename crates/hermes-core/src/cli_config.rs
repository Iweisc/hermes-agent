//! Configuration management for Hermes Agent (native Rust port of
//! `hermes_cli/config.py`).
//!
//! Config files are stored in `~/.hermes/` for easy access:
//! - `~/.hermes/config.yaml`  - All settings (model, toolsets, terminal, etc.)
//! - `~/.hermes/.env`         - API keys and secrets
//!
//! This module provides the config loading/saving/migration pipeline, the
//! managed-mode (NixOS / Homebrew) detection, the container-exec metadata
//! reader, the `.env` read/write/sanitise helpers, the env-var metadata
//! tables (`REQUIRED_ENV_VARS`, `OPTIONAL_ENV_VARS`, `ENV_VARS_BY_VERSION`),
//! the config-structure validator, and the dotted-key setter.
//!
//! The dynamic config tree is represented with `serde_yaml::Value` (matching
//! the Python dict that comes out of `yaml.safe_load`), and the env-var
//! metadata uses small owned structs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_yaml::{Mapping, Value};

use crate::mod_hermes_constants::{
    get_config_path, get_env_path, get_hermes_home, is_container as constants_is_container,
};

// =============================================================================
// Globals / regexes
// =============================================================================

fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

/// `^[A-Za-z_][A-Za-z0-9_]*$`
pub fn is_valid_env_var_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Env var names written to .env that aren't in OPTIONAL_ENV_VARS
/// (managed by setup/provider flows directly).
pub const EXTRA_ENV_KEYS: &[&str] = &[
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "DISCORD_HOME_CHANNEL",
    "DISCORD_HOME_CHANNEL_NAME",
    "TELEGRAM_HOME_CHANNEL",
    "TELEGRAM_HOME_CHANNEL_NAME",
    "SLACK_HOME_CHANNEL",
    "SLACK_HOME_CHANNEL_NAME",
    "SIGNAL_ACCOUNT",
    "SIGNAL_HTTP_URL",
    "SIGNAL_ALLOWED_USERS",
    "SIGNAL_GROUP_ALLOWED_USERS",
    "SIGNAL_HOME_CHANNEL",
    "SIGNAL_HOME_CHANNEL_NAME",
    "SMS_HOME_CHANNEL",
    "SMS_HOME_CHANNEL_NAME",
    "DINGTALK_CLIENT_ID",
    "DINGTALK_CLIENT_SECRET",
    "DINGTALK_HOME_CHANNEL",
    "DINGTALK_HOME_CHANNEL_NAME",
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_ENCRYPT_KEY",
    "FEISHU_VERIFICATION_TOKEN",
    "FEISHU_HOME_CHANNEL",
    "FEISHU_HOME_CHANNEL_NAME",
    "YUANBAO_HOME_CHANNEL",
    "YUANBAO_HOME_CHANNEL_NAME",
    "WECOM_BOT_ID",
    "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_ID",
    "WECOM_CALLBACK_CORP_SECRET",
    "WECOM_CALLBACK_AGENT_ID",
    "WECOM_CALLBACK_TOKEN",
    "WECOM_CALLBACK_ENCODING_AES_KEY",
    "WECOM_CALLBACK_HOST",
    "WECOM_CALLBACK_PORT",
    "WECOM_HOME_CHANNEL",
    "WECOM_HOME_CHANNEL_NAME",
    "WEIXIN_ACCOUNT_ID",
    "WEIXIN_TOKEN",
    "WEIXIN_BASE_URL",
    "WEIXIN_CDN_BASE_URL",
    "WEIXIN_HOME_CHANNEL",
    "WEIXIN_HOME_CHANNEL_NAME",
    "WEIXIN_DM_POLICY",
    "WEIXIN_GROUP_POLICY",
    "WEIXIN_ALLOWED_USERS",
    "WEIXIN_GROUP_ALLOWED_USERS",
    "WEIXIN_ALLOW_ALL_USERS",
    "BLUEBUBBLES_SERVER_URL",
    "BLUEBUBBLES_PASSWORD",
    "BLUEBUBBLES_HOME_CHANNEL",
    "BLUEBUBBLES_HOME_CHANNEL_NAME",
    "QQ_APP_ID",
    "QQ_CLIENT_SECRET",
    "QQBOT_HOME_CHANNEL",
    "QQBOT_HOME_CHANNEL_NAME",
    "QQ_HOME_CHANNEL",
    "QQ_HOME_CHANNEL_NAME",
    "QQ_ALLOWED_USERS",
    "QQ_GROUP_ALLOWED_USERS",
    "QQ_ALLOW_ALL_USERS",
    "QQ_MARKDOWN_SUPPORT",
    "QQ_STT_API_KEY",
    "QQ_STT_BASE_URL",
    "QQ_STT_MODEL",
    "IRC_SERVER",
    "IRC_PORT",
    "IRC_NICKNAME",
    "IRC_CHANNEL",
    "IRC_USE_TLS",
    "IRC_SERVER_PASSWORD",
    "IRC_NICKSERV_PASSWORD",
    "TERMINAL_ENV",
    "TERMINAL_SSH_KEY",
    "TERMINAL_SSH_PORT",
    "WHATSAPP_MODE",
    "WHATSAPP_ENABLED",
    "MATTERMOST_HOME_CHANNEL",
    "MATTERMOST_HOME_CHANNEL_NAME",
    "MATTERMOST_REPLY_MODE",
    "MATRIX_PASSWORD",
    "MATRIX_ENCRYPTION",
    "MATRIX_DEVICE_ID",
    "MATRIX_HOME_ROOM",
    "MATRIX_REQUIRE_MENTION",
    "MATRIX_FREE_RESPONSE_ROOMS",
    "MATRIX_AUTO_THREAD",
    "MATRIX_DM_AUTO_THREAD",
    "MATRIX_RECOVERY_KEY",
    "HERMES_LANGFUSE_ENV",
    "HERMES_LANGFUSE_RELEASE",
    "HERMES_LANGFUSE_SAMPLE_RATE",
    "HERMES_LANGFUSE_MAX_CHARS",
    "HERMES_LANGFUSE_DEBUG",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
];

// load_config / read_raw_config caches keyed on str(config_path) -> (mtime_ns, size, value).
type CacheEntry = (i128, u64, Value);

fn load_config_cache() -> &'static Mutex<HashMap<String, CacheEntry>> {
    static REAL: std::sync::OnceLock<Mutex<HashMap<String, CacheEntry>>> =
        std::sync::OnceLock::new();
    REAL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn raw_config_cache() -> &'static Mutex<HashMap<String, CacheEntry>> {
    static REAL: std::sync::OnceLock<Mutex<HashMap<String, CacheEntry>>> =
        std::sync::OnceLock::new();
    REAL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn last_expanded_config_by_path() -> &'static Mutex<HashMap<String, Value>> {
    static REAL: std::sync::OnceLock<Mutex<HashMap<String, Value>>> = std::sync::OnceLock::new();
    REAL.get_or_init(|| Mutex::new(HashMap::new()))
}

// =============================================================================
// Managed mode (NixOS declarative config)
// =============================================================================

const MANAGED_TRUE_VALUES: &[&str] = &["true", "1", "yes"];

fn managed_system_name(normalized: &str) -> Option<&'static str> {
    match normalized {
        "brew" | "homebrew" => Some("Homebrew"),
        "nix" | "nixos" => Some("NixOS"),
        _ => None,
    }
}

/// Return the package manager owning this install, if any.
pub fn get_managed_system() -> Option<String> {
    let raw = std::env::var("HERMES_MANAGED")
        .unwrap_or_default()
        .trim()
        .to_string();
    if !raw.is_empty() {
        let normalized = raw.to_lowercase();
        if MANAGED_TRUE_VALUES.contains(&normalized.as_str()) {
            return Some("NixOS".to_string());
        }
        return Some(
            managed_system_name(&normalized)
                .map(|s| s.to_string())
                .unwrap_or(raw),
        );
    }

    let managed_marker = get_hermes_home().join(".managed");
    if managed_marker.exists() {
        return Some("NixOS".to_string());
    }
    None
}

/// Check if Hermes is running in package-manager-managed mode.
pub fn is_managed() -> bool {
    get_managed_system().is_some()
}

/// Return the preferred upgrade command for a managed install.
pub fn get_managed_update_command() -> Option<String> {
    match get_managed_system().as_deref() {
        Some("Homebrew") => Some("brew upgrade hermes-agent".to_string()),
        Some("NixOS") => Some("sudo nixos-rebuild switch".to_string()),
        _ => None,
    }
}

/// Return the best update command for the current installation.
pub fn recommended_update_command() -> String {
    get_managed_update_command().unwrap_or_else(|| "hermes update".to_string())
}

/// Build a user-facing error for managed installs.
pub fn format_managed_message(action: &str) -> String {
    let managed_system = get_managed_system().unwrap_or_else(|| "a package manager".to_string());
    let raw = std::env::var("HERMES_MANAGED")
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    if managed_system == "NixOS" {
        let env_hint = if MANAGED_TRUE_VALUES.contains(&raw.as_str()) {
            "true".to_string()
        } else if raw.is_empty() {
            "true".to_string()
        } else {
            raw
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by NixOS \
(HERMES_MANAGED={env_hint}).\n\
Edit services.hermes-agent.settings in your configuration.nix and run:\n\
  sudo nixos-rebuild switch"
        );
    }

    if managed_system == "Homebrew" {
        let env_hint = if raw.is_empty() {
            "homebrew".to_string()
        } else {
            raw
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by Homebrew \
(HERMES_MANAGED={env_hint}).\n\
Use:\n\
  brew upgrade hermes-agent"
        );
    }

    format!(
        "Cannot {action}: this Hermes installation is managed by {managed_system}.\n\
Use your package manager to upgrade or reinstall Hermes."
    )
}

/// Print user-friendly error for managed mode (to stderr).
pub fn managed_error(action: &str) {
    eprintln!("{}", format_managed_message(action));
}

// =============================================================================
// Container-aware CLI (NixOS container mode)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerExecInfo {
    pub backend: String,
    pub container_name: String,
    pub exec_user: String,
    pub hermes_bin: String,
}

/// Read container mode metadata from `HERMES_HOME/.container-mode`.
///
/// Returns `None` if container mode is not active, we're already inside the
/// container, or `HERMES_DEV=1` is set.
pub fn get_container_exec_info() -> Option<ContainerExecInfo> {
    if std::env::var("HERMES_DEV").as_deref() == Ok("1") {
        return None;
    }
    if constants_is_container() {
        return None;
    }

    let container_mode_file = get_hermes_home().join(".container-mode");
    let content = match fs::read_to_string(&container_mode_file) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        // All other errors propagate in Python; here we treat as None to be
        // permissive, but preserve "not found => None" semantics explicitly.
        Err(_) => return None,
    };

    let mut info: HashMap<String, String> = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.contains('=') && !line.starts_with('#') {
            if let Some((key, value)) = line.split_once('=') {
                info.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
    }

    Some(ContainerExecInfo {
        backend: info
            .get("backend")
            .cloned()
            .unwrap_or_else(|| "docker".to_string()),
        container_name: info
            .get("container_name")
            .cloned()
            .unwrap_or_else(|| "hermes-agent".to_string()),
        exec_user: info
            .get("exec_user")
            .cloned()
            .unwrap_or_else(|| "hermes".to_string()),
        hermes_bin: info
            .get("hermes_bin")
            .cloned()
            .unwrap_or_else(|| "/data/current-package/bin/hermes".to_string()),
    })
}

// =============================================================================
// Config paths
// =============================================================================

/// Get the project installation directory.
///
/// The Python uses `Path(__file__).parent.parent.resolve()`. There's no exact
/// analogue for a compiled binary; we approximate with the current executable's
/// grandparent directory, falling back to the current dir.
pub fn get_project_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent().and_then(|p| p.parent()) {
            return parent.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Set directory to owner-only access (0700 by default). No-op on Windows.
/// Skipped in managed mode. Mode overridable via `HERMES_HOME_MODE`.
fn secure_dir(path: &Path) {
    if is_managed() {
        return;
    }
    if is_windows() {
        return;
    }
    #[cfg(unix)]
    {
        let mode_str = std::env::var("HERMES_HOME_MODE")
            .unwrap_or_default()
            .trim()
            .to_string();
        let mode = if mode_str.is_empty() {
            0o700
        } else {
            u32::from_str_radix(&mode_str, 8).unwrap_or(0o700)
        };
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
}

/// Detect if we're running inside a Docker/Podman/LXC container.
fn is_container_secure() -> bool {
    if std::env::var("HERMES_CONTAINER").map(|v| !v.is_empty()).unwrap_or(false)
        || std::env::var("HERMES_SKIP_CHMOD").map(|v| !v.is_empty()).unwrap_or(false)
    {
        return true;
    }
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    if let Ok(content) = fs::read_to_string("/proc/1/cgroup") {
        if content.contains("docker") || content.contains("lxc") || content.contains("kubepods") {
            return true;
        }
    }
    false
}

/// Set file to owner-only read/write (0600). No-op on Windows / managed /
/// container.
fn secure_file(path: &Path) {
    if is_managed() || is_container_secure() {
        return;
    }
    if is_windows() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if path.exists() {
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        }
    }
}

/// Seed a default SOUL.md into HERMES_HOME if the user doesn't have one yet.
fn ensure_default_soul_md(home: &Path) {
    let soul_path = home.join("SOUL.md");
    if soul_path.exists() {
        return;
    }
    if fs::write(&soul_path, default_soul_md()).is_ok() {
        secure_file(&soul_path);
    }
}

/// Default SOUL.md content. The Python imports this from
/// `hermes_cli.default_soul`; the canonical text is the same string the ported
/// `config.rs` seeds.
pub const DEFAULT_SOUL_MD: &str = "You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.";

fn default_soul_md() -> &'static str {
    DEFAULT_SOUL_MD
}

/// Ensure ~/.hermes directory structure exists with secure permissions.
pub fn ensure_hermes_home() -> Result<(), String> {
    let home = get_hermes_home();
    if is_managed() {
        // Python sets umask(0o007) around the managed variant. We can't portably
        // scope umask in safe Rust; the managed activation script owns dir
        // creation anyway, so we just verify + seed.
        ensure_hermes_home_managed(&home)
    } else {
        fs::create_dir_all(&home).map_err(|e| e.to_string())?;
        secure_dir(&home);
        for subdir in ["cron", "sessions", "logs", "logs/curator", "memories"] {
            let d = home.join(subdir);
            fs::create_dir_all(&d).map_err(|e| e.to_string())?;
            secure_dir(&d);
        }
        ensure_default_soul_md(&home);
        Ok(())
    }
}

/// Managed-mode variant: verify dirs exist (activation creates them), seed
/// SOUL.md.
fn ensure_hermes_home_managed(home: &Path) -> Result<(), String> {
    if !home.is_dir() {
        return Err(format!(
            "HERMES_HOME {} does not exist. Run 'sudo nixos-rebuild switch' first.",
            home.display()
        ));
    }
    for subdir in ["cron", "sessions", "logs", "memories"] {
        let d = home.join(subdir);
        if !d.is_dir() {
            return Err(format!(
                "{} does not exist. Run 'sudo nixos-rebuild switch' first.",
                d.display()
            ));
        }
    }
    let _ = fs::create_dir_all(home.join("logs").join("curator"));
    ensure_default_soul_md(home);
    Ok(())
}

// =============================================================================
// DEFAULT_CONFIG
// =============================================================================

/// The current config schema version.
pub const CONFIG_VERSION: i64 = 23;

const DEFAULT_CONFIG_YAML: &str = r#"
model: ""
providers: {}
fallback_providers: []
credential_pool_strategies: {}
toolsets: ["hermes-cli"]
agent:
  max_turns: 90
  gateway_timeout: 1800
  restart_drain_timeout: 180
  api_max_retries: 3
  service_tier: ""
  tool_use_enforcement: "auto"
  gateway_timeout_warning: 900
  gateway_notify_interval: 180
  gateway_auto_continue_freshness: 3600
  image_input_mode: "auto"
  disabled_toolsets: []
terminal:
  backend: "local"
  modal_mode: "auto"
  cwd: "."
  timeout: 180
  env_passthrough: []
  shell_init_files: []
  auto_source_bashrc: true
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []
  docker_env: {}
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  vercel_runtime: "node24"
  container_cpu: 1
  container_memory: 5120
  container_disk: 51200
  container_persistent: true
  docker_volumes: []
  docker_mount_cwd_to_workspace: false
  docker_run_as_host_user: false
  persistent_shell: true
web:
  backend: ""
  search_backend: ""
  extract_backend: ""
browser:
  inactivity_timeout: 120
  command_timeout: 30
  record_sessions: false
  allow_private_urls: false
  engine: "auto"
  auto_local_for_private_urls: true
  cdp_url: ""
  dialog_policy: "must_respond"
  dialog_timeout_s: 300
  camofox:
    managed_persistence: false
checkpoints:
  enabled: false
  max_snapshots: 20
  max_total_size_mb: 500
  max_file_size_mb: 10
  auto_prune: true
  retention_days: 7
  delete_orphans: true
  min_interval_hours: 24
file_read_max_chars: 100000
tool_output:
  max_bytes: 50000
  max_lines: 2000
  max_line_length: 2000
tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: false
  warn_after:
    exact_failure: 2
    same_tool_failure: 3
    idempotent_no_progress: 2
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20
  protect_last_n: 20
  hygiene_hard_message_limit: 400
prompt_caching:
  cache_ttl: "5m"
openrouter:
  response_cache: true
  response_cache_ttl: 300
bedrock:
  region: ""
  discovery:
    enabled: true
    provider_filter: []
    refresh_interval: 3600
  guardrail:
    guardrail_identifier: ""
    guardrail_version: ""
    stream_processing_mode: "async"
    trace: "disabled"
auxiliary:
  vision:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
    extra_body: {}
    download_timeout: 30
  web_extract:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 360
    extra_body: {}
  compression:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
    extra_body: {}
  session_search:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
    max_concurrency: 3
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
  title_generation:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    extra_body: {}
  curator:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 600
    extra_body: {}
display:
  compact: false
  personality: "kawaii"
  resume_display: "full"
  busy_input_mode: "interrupt"
  tui_auto_resume_recent: false
  bell_on_complete: false
  show_reasoning: false
  streaming: false
  final_response_markdown: "strip"
  persistent_output: true
  persistent_output_max_lines: 200
  inline_diffs: true
  show_cost: false
  skin: "default"
  language: "en"
  tui_status_indicator: "kaomoji"
  user_message_preview:
    first_lines: 2
    last_lines: 2
  interim_assistant_messages: true
  tool_progress_command: false
  tool_progress_overrides: {}
  tool_preview_length: 0
  ephemeral_system_ttl: 0
  platforms: {}
  runtime_footer:
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  copy_shortcut: "auto"
dashboard:
  theme: "default"
privacy:
  redact_pii: false
tts:
  provider: "edge"
  edge:
    voice: "en-US-AriaNeural"
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"
  xai:
    voice_id: "eve"
    language: "en"
    sample_rate: 24000
    bit_rate: 128000
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"
  neutts:
    ref_audio: ""
    ref_text: ""
    model: "neuphonic/neutts-air-q4-gguf"
    device: "cpu"
  piper:
    voice: "en_US-lessac-medium"
stt:
  enabled: true
  provider: "local"
  local:
    model: "base"
    language: ""
  openai:
    model: "whisper-1"
  mistral:
    model: "voxtral-mini-latest"
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0
human_delay:
  mode: "off"
  min_ms: 800
  max_ms: 2500
context:
  engine: "compressor"
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: ""
delegation:
  model: ""
  provider: ""
  base_url: ""
  api_key: ""
  inherit_mcp_toolsets: true
  max_iterations: 50
  child_timeout_seconds: 600
  reasoning_effort: ""
  max_concurrent_children: 3
  max_spawn_depth: 1
  orchestrator_enabled: true
  subagent_auto_approve: false
prefill_messages_file: ""
goals:
  max_turns: 20
skills:
  external_dirs: []
  template_vars: true
  inline_shell: false
  inline_shell_timeout: 10
  guard_agent_created: false
curator:
  enabled: true
  interval_hours: 168
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
  backup:
    enabled: true
    keep: 5
honcho: {}
timezone: ""
discord:
  require_mention: true
  free_response_channels: ""
  allowed_channels: ""
  auto_thread: true
  reactions: true
  channel_prompts: {}
  server_actions: ""
whatsapp: {}
telegram:
  reactions: false
  channel_prompts: {}
slack:
  channel_prompts: {}
mattermost:
  channel_prompts: {}
approvals:
  mode: "manual"
  timeout: 60
  cron_mode: "deny"
  mcp_reload_confirm: true
command_allowlist: []
quick_commands: {}
hooks: {}
hooks_auto_accept: false
personalities: {}
security:
  allow_private_urls: false
  redact_secrets: false
  tirith_enabled: true
  tirith_path: "tirith"
  tirith_timeout: 5
  tirith_fail_open: true
  website_blocklist:
    enabled: false
    domains: []
    shared_files: []
cron:
  wrap_response: true
  max_parallel_jobs: null
kanban:
  dispatch_in_gateway: true
  dispatch_interval_seconds: 60
code_execution:
  mode: "project"
logging:
  level: "INFO"
  max_size_mb: 5
  backup_count: 3
model_catalog:
  enabled: true
  url: "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json"
  ttl_hours: 24
  providers: {}
network:
  force_ipv4: false
sessions:
  auto_prune: false
  retention_days: 90
  vacuum_after_prune: true
  min_interval_hours: 24
onboarding:
  seen: {}
updates:
  pre_update_backup: false
  backup_keep: 5
_config_version: 23
"#;

/// Build a fresh deep copy of the default config tree.
pub fn default_config() -> Value {
    // Parsed once from the embedded YAML; cloned per call.
    static PARSED: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    PARSED
        .get_or_init(|| {
            serde_yaml::from_str::<Value>(DEFAULT_CONFIG_YAML)
                .expect("embedded default config YAML must parse")
        })
        .clone()
}

// =============================================================================
// Env-var metadata tables
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct EnvVarInfo {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub url: Option<String>,
    pub password: bool,
    pub tools: Vec<String>,
    pub category: String,
    pub advanced: bool,
    pub is_required: bool,
}

/// Versions in which each batch of optional env vars was introduced.
pub fn env_vars_by_version() -> BTreeMap<i64, Vec<&'static str>> {
    let mut m = BTreeMap::new();
    m.insert(
        3,
        vec![
            "FIRECRAWL_API_KEY",
            "BROWSERBASE_API_KEY",
            "BROWSERBASE_PROJECT_ID",
            "FAL_KEY",
        ],
    );
    m.insert(4, vec!["VOICE_TOOLS_OPENAI_KEY", "ELEVENLABS_API_KEY"]);
    m.insert(
        5,
        vec![
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_ALLOWED_USERS",
            "SLACK_BOT_TOKEN",
            "SLACK_APP_TOKEN",
            "SLACK_ALLOWED_USERS",
        ],
    );
    m.insert(10, vec!["TAVILY_API_KEY"]);
    m.insert(11, vec!["TERMINAL_MODAL_MODE"]);
    m
}

/// Required env vars (intentionally empty — provider selection happens in the
/// setup wizard).
pub fn required_env_vars() -> Vec<EnvVarInfo> {
    Vec::new()
}

/// Helper used to build the static OPTIONAL_ENV_VARS table compactly.
struct EnvSpec {
    name: &'static str,
    description: &'static str,
    prompt: &'static str,
    url: Option<&'static str>,
    password: bool,
    tools: &'static [&'static str],
    category: &'static str,
    advanced: bool,
}

impl EnvSpec {
    fn into_info(self) -> EnvVarInfo {
        EnvVarInfo {
            name: self.name.to_string(),
            description: self.description.to_string(),
            prompt: self.prompt.to_string(),
            url: self.url.map(|s| s.to_string()),
            password: self.password,
            tools: self.tools.iter().map(|s| s.to_string()).collect(),
            category: self.category.to_string(),
            advanced: self.advanced,
            is_required: false,
        }
    }
}

macro_rules! spec {
    ($name:expr, $desc:expr, $prompt:expr, $url:expr, $pw:expr, $tools:expr, $cat:expr, $adv:expr) => {
        EnvSpec {
            name: $name,
            description: $desc,
            prompt: $prompt,
            url: $url,
            password: $pw,
            tools: $tools,
            category: $cat,
            advanced: $adv,
        }
    };
}

/// The built-in OPTIONAL_ENV_VARS table (insertion order preserved).
///
/// Profile-driven injection (the Python `_inject_profile_env_vars`) is a
/// runtime concern in the Rust provider layer; this returns the statically
/// declared set, which the caller may extend.
pub fn optional_env_vars() -> Vec<EnvVarInfo> {
    let specs: Vec<EnvSpec> = vec![
        // ── Provider ──
        spec!("NOUS_BASE_URL", "Nous Portal base URL override", "Nous Portal base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("OPENROUTER_API_KEY", "OpenRouter API key (for vision, web scraping helpers, and MoA)", "OpenRouter API key", Some("https://openrouter.ai/keys"), true, &["vision_analyze", "mixture_of_agents"], "provider", true),
        spec!("GOOGLE_API_KEY", "Google AI Studio API key (also recognized as GEMINI_API_KEY)", "Google AI Studio API key", Some("https://aistudio.google.com/app/apikey"), true, &[], "provider", true),
        spec!("GEMINI_API_KEY", "Google AI Studio API key (alias for GOOGLE_API_KEY)", "Gemini API key", Some("https://aistudio.google.com/app/apikey"), true, &[], "provider", true),
        spec!("GEMINI_BASE_URL", "Google AI Studio base URL override", "Gemini base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("XAI_API_KEY", "xAI API key", "xAI API key", Some("https://console.x.ai/"), true, &[], "provider", true),
        spec!("XAI_BASE_URL", "xAI base URL override", "xAI base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("NVIDIA_API_KEY", "NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)", "NVIDIA NIM API key", Some("https://build.nvidia.com/"), true, &[], "provider", true),
        spec!("NVIDIA_BASE_URL", "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)", "NVIDIA NIM base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("LM_API_KEY", "LM Studio bearer token for auth-enabled local servers", "LM Studio API key / bearer token", None, true, &[], "provider", true),
        spec!("LM_BASE_URL", "LM Studio base URL override", "LM Studio base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("GLM_API_KEY", "Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)", "Z.AI / GLM API key", Some("https://z.ai/"), true, &[], "provider", true),
        spec!("ZAI_API_KEY", "Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", Some("https://z.ai/"), true, &[], "provider", true),
        spec!("Z_AI_API_KEY", "Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", Some("https://z.ai/"), true, &[], "provider", true),
        spec!("GLM_BASE_URL", "Z.AI / GLM base URL override", "Z.AI / GLM base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("KIMI_API_KEY", "Kimi / Moonshot API key", "Kimi API key", Some("https://platform.moonshot.cn/"), true, &[], "provider", true),
        spec!("KIMI_BASE_URL", "Kimi / Moonshot base URL override", "Kimi base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("KIMI_CN_API_KEY", "Kimi / Moonshot China API key", "Kimi (China) API key", Some("https://platform.moonshot.cn/"), true, &[], "provider", true),
        spec!("STEPFUN_API_KEY", "StepFun Step Plan API key", "StepFun Step Plan API key", Some("https://platform.stepfun.com/"), true, &[], "provider", true),
        spec!("STEPFUN_BASE_URL", "StepFun Step Plan base URL override", "StepFun Step Plan base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("ARCEEAI_API_KEY", "Arcee AI API key", "Arcee AI API key", Some("https://chat.arcee.ai/"), true, &[], "provider", true),
        spec!("ARCEE_BASE_URL", "Arcee AI base URL override", "Arcee base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("GMI_API_KEY", "GMI Cloud API key", "GMI Cloud API key", Some("https://www.gmicloud.ai/"), true, &[], "provider", true),
        spec!("GMI_BASE_URL", "GMI Cloud base URL override", "GMI Cloud base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("MINIMAX_API_KEY", "MiniMax API key (international)", "MiniMax API key", Some("https://www.minimax.io/"), true, &[], "provider", true),
        spec!("MINIMAX_BASE_URL", "MiniMax base URL override", "MiniMax base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("MINIMAX_CN_API_KEY", "MiniMax API key (China endpoint)", "MiniMax (China) API key", Some("https://www.minimaxi.com/"), true, &[], "provider", true),
        spec!("MINIMAX_CN_BASE_URL", "MiniMax (China) base URL override", "MiniMax (China) base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("DEEPSEEK_API_KEY", "DeepSeek API key for direct DeepSeek access", "DeepSeek API Key", Some("https://platform.deepseek.com/api_keys"), true, &[], "provider", false),
        spec!("DEEPSEEK_BASE_URL", "Custom DeepSeek API base URL (advanced)", "DeepSeek Base URL", Some(""), false, &[], "provider", false),
        spec!("DASHSCOPE_API_KEY", "Alibaba Cloud DashScope API key (Qwen + multi-provider models)", "DashScope API Key", Some("https://modelstudio.console.alibabacloud.com/"), true, &[], "provider", false),
        spec!("DASHSCOPE_BASE_URL", "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)", "DashScope Base URL", Some(""), false, &[], "provider", true),
        spec!("HERMES_QWEN_BASE_URL", "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)", "Qwen Portal base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("HERMES_GEMINI_CLIENT_ID", "Google OAuth client ID for google-gemini-cli (optional; defaults to Google's public gemini-cli client)", "Google OAuth client ID (optional — leave empty to use the public default)", Some("https://console.cloud.google.com/apis/credentials"), false, &[], "provider", true),
        spec!("HERMES_GEMINI_CLIENT_SECRET", "Google OAuth client secret for google-gemini-cli (optional)", "Google OAuth client secret (optional)", Some("https://console.cloud.google.com/apis/credentials"), true, &[], "provider", true),
        spec!("HERMES_GEMINI_PROJECT_ID", "GCP project ID for paid Gemini tiers (free tier auto-provisions)", "GCP project ID for Gemini OAuth (leave empty for free tier)", None, false, &[], "provider", true),
        spec!("OPENCODE_ZEN_API_KEY", "OpenCode Zen API key (pay-as-you-go access to curated models)", "OpenCode Zen API key", Some("https://opencode.ai/auth"), true, &[], "provider", true),
        spec!("OPENCODE_ZEN_BASE_URL", "OpenCode Zen base URL override", "OpenCode Zen base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("OPENCODE_GO_API_KEY", "OpenCode Go API key ($10/month subscription for open models)", "OpenCode Go API key", Some("https://opencode.ai/auth"), true, &[], "provider", true),
        spec!("OPENCODE_GO_BASE_URL", "OpenCode Go base URL override", "OpenCode Go base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("HF_TOKEN", "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)", "Hugging Face Token", Some("https://huggingface.co/settings/tokens"), true, &[], "provider", false),
        spec!("HF_BASE_URL", "Hugging Face Inference Providers base URL override", "HF base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("OLLAMA_API_KEY", "Ollama Cloud API key (ollama.com — cloud-hosted open models)", "Ollama Cloud API key", Some("https://ollama.com/settings"), true, &[], "provider", true),
        spec!("OLLAMA_BASE_URL", "Ollama Cloud base URL override (default: https://ollama.com/v1)", "Ollama base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("XIAOMI_API_KEY", "Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, mimo-v2-omni, mimo-v2-flash)", "Xiaomi MiMo API Key", Some("https://platform.xiaomimimo.com"), true, &[], "provider", false),
        spec!("XIAOMI_BASE_URL", "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)", "Xiaomi base URL (leave empty for default)", None, false, &[], "provider", true),
        spec!("AWS_REGION", "AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)", "AWS Region", Some("https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html"), false, &[], "provider", true),
        spec!("AWS_PROFILE", "AWS named profile for Bedrock authentication (from ~/.aws/credentials)", "AWS Profile", None, false, &[], "provider", true),
        spec!("AZURE_FOUNDRY_API_KEY", "Azure Foundry API key for custom Azure endpoints", "Azure Foundry API Key", Some("https://ai.azure.com/"), true, &[], "provider", false),
        spec!("AZURE_FOUNDRY_BASE_URL", "Azure Foundry base URL (set via 'hermes model' for endpoint-specific config)", "Azure Foundry base URL", None, false, &[], "provider", true),
        // ── Tool API keys ──
        spec!("EXA_API_KEY", "Exa API key for AI-native web search and contents", "Exa API key", Some("https://exa.ai/"), true, &["web_search", "web_extract"], "tool", false),
        spec!("PARALLEL_API_KEY", "Parallel API key for AI-native web search and extract", "Parallel API key", Some("https://parallel.ai/"), true, &["web_search", "web_extract"], "tool", false),
        spec!("FIRECRAWL_API_KEY", "Firecrawl API key for web search and scraping", "Firecrawl API key", Some("https://firecrawl.dev/"), true, &["web_search", "web_extract"], "tool", false),
        spec!("FIRECRAWL_API_URL", "Firecrawl API URL for self-hosted instances (optional)", "Firecrawl API URL (leave empty for cloud)", None, false, &[], "tool", true),
        spec!("FIRECRAWL_GATEWAY_URL", "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)", "Firecrawl gateway URL (leave empty to derive from domain)", None, false, &[], "tool", true),
        spec!("TOOL_GATEWAY_DOMAIN", "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com", "Tool-gateway domain suffix", None, false, &[], "tool", true),
        spec!("TOOL_GATEWAY_SCHEME", "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts (`https` by default, set `http` for local gateway testing)", "Tool-gateway URL scheme", None, false, &[], "tool", true),
        spec!("TOOL_GATEWAY_USER_TOKEN", "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise read from the Hermes auth store)", "Tool-gateway user token", None, true, &[], "tool", true),
        spec!("TAVILY_API_KEY", "Tavily API key for AI-native web search, extract, and crawl", "Tavily API key", Some("https://app.tavily.com/home"), true, &["web_search", "web_extract", "web_crawl"], "tool", false),
        spec!("SEARXNG_URL", "URL of your SearXNG instance for free self-hosted web search", "SearXNG URL (e.g. http://localhost:8080)", Some("https://searxng.github.io/searxng/"), false, &["web_search"], "tool", false),
        spec!("BROWSERBASE_API_KEY", "Browserbase API key for cloud browser (optional — local browser works without this)", "Browserbase API key", Some("https://browserbase.com/"), true, &["browser_navigate", "browser_click"], "tool", false),
        spec!("BROWSERBASE_PROJECT_ID", "Browserbase project ID (optional — only needed for cloud browser)", "Browserbase project ID", Some("https://browserbase.com/"), false, &["browser_navigate", "browser_click"], "tool", false),
        spec!("BROWSER_USE_API_KEY", "Browser Use API key for cloud browser (optional — local browser works without this)", "Browser Use API key", Some("https://browser-use.com/"), true, &["browser_navigate", "browser_click"], "tool", false),
        spec!("FIRECRAWL_BROWSER_TTL", "Firecrawl browser session TTL in seconds (optional, default 300)", "Browser session TTL (seconds)", None, false, &["browser_navigate", "browser_click"], "tool", false),
        spec!("AGENT_BROWSER_ENGINE", "Browser engine for local mode: auto (default Chrome), lightpanda (faster, no screenshots), chrome", "Browser engine (auto/lightpanda/chrome)", Some("https://github.com/vercel-labs/agent-browser"), false, &["browser_navigate", "browser_snapshot", "browser_click", "browser_vision"], "tool", true),
        spec!("CAMOFOX_URL", "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)", "Camofox server URL", Some("https://github.com/jo-inc/camofox-browser"), false, &["browser_navigate", "browser_click"], "tool", false),
        spec!("FAL_KEY", "FAL API key for image generation", "FAL API key", Some("https://fal.ai/"), true, &["image_generate"], "tool", false),
        spec!("TINKER_API_KEY", "Tinker API key for RL training", "Tinker API key", Some("https://tinker-console.thinkingmachines.ai/keys"), true, &["rl_start_training", "rl_check_status", "rl_stop_training"], "tool", false),
        spec!("WANDB_API_KEY", "Weights & Biases API key for experiment tracking", "WandB API key", Some("https://wandb.ai/authorize"), true, &["rl_get_results", "rl_check_status"], "tool", false),
        spec!("VOICE_TOOLS_OPENAI_KEY", "OpenAI API key for voice transcription (Whisper) and OpenAI TTS", "OpenAI API Key (for Whisper STT + TTS)", Some("https://platform.openai.com/api-keys"), true, &["voice_transcription", "openai_tts"], "tool", false),
        spec!("ELEVENLABS_API_KEY", "ElevenLabs API key for premium text-to-speech voices", "ElevenLabs API key", Some("https://elevenlabs.io/"), true, &[], "tool", false),
        spec!("MISTRAL_API_KEY", "Mistral API key for Voxtral TTS and transcription (STT)", "Mistral API key", Some("https://console.mistral.ai/"), true, &[], "tool", false),
        spec!("GITHUB_TOKEN", "GitHub token for Skills Hub (higher API rate limits, skill publish)", "GitHub Token", Some("https://github.com/settings/tokens"), true, &[], "tool", false),
        // ── Bundled skills ──
        spec!("NOTION_API_KEY", "Notion integration token (used by the `notion` skill)", "Notion API key", Some("https://www.notion.so/my-integrations"), true, &[], "skill", true),
        spec!("LINEAR_API_KEY", "Linear personal API key (used by the `linear` skill)", "Linear API key", Some("https://linear.app/settings/account/security"), true, &[], "skill", true),
        spec!("AIRTABLE_API_KEY", "Airtable personal access token (used by the `airtable` skill)", "Airtable API key", Some("https://airtable.com/create/tokens"), true, &[], "skill", true),
        spec!("TENOR_API_KEY", "Tenor API key for GIF search (used by the `gif-search` skill)", "Tenor API key", Some("https://developers.google.com/tenor/guides/quickstart"), true, &[], "skill", true),
        // ── Honcho ──
        spec!("HONCHO_API_KEY", "Honcho API key for AI-native persistent memory", "Honcho API key", Some("https://app.honcho.dev"), true, &["honcho_context"], "tool", false),
        spec!("HONCHO_BASE_URL", "Base URL for self-hosted Honcho instances (no API key needed)", "Honcho base URL (e.g. http://localhost:8000)", None, false, &[], "tool", false),
        // ── Langfuse ──
        spec!("HERMES_LANGFUSE_PUBLIC_KEY", "Langfuse project public key (pk-lf-...)", "Langfuse public key", Some("https://cloud.langfuse.com"), false, &[], "tool", false),
        spec!("HERMES_LANGFUSE_SECRET_KEY", "Langfuse project secret key (sk-lf-...)", "Langfuse secret key", Some("https://cloud.langfuse.com"), true, &[], "tool", false),
        spec!("HERMES_LANGFUSE_BASE_URL", "Langfuse server URL (default: https://cloud.langfuse.com)", "Langfuse server URL (leave empty for cloud.langfuse.com)", None, false, &[], "tool", true),
        // ── Messaging platforms ──
        spec!("TELEGRAM_BOT_TOKEN", "Telegram bot token from @BotFather", "Telegram bot token", Some("https://t.me/BotFather"), true, &[], "messaging", false),
        spec!("TELEGRAM_ALLOWED_USERS", "Comma-separated Telegram user IDs allowed to use the bot (get ID from @userinfobot)", "Allowed Telegram user IDs (comma-separated)", Some("https://t.me/userinfobot"), false, &[], "messaging", false),
        spec!("TELEGRAM_PROXY", "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, https://, socks5://", "Telegram proxy URL (optional)", None, false, &[], "messaging", false),
        spec!("DISCORD_BOT_TOKEN", "Discord bot token from Developer Portal", "Discord bot token", Some("https://discord.com/developers/applications"), true, &[], "messaging", false),
        spec!("DISCORD_ALLOWED_USERS", "Comma-separated Discord user IDs allowed to use the bot", "Allowed Discord user IDs (comma-separated)", None, false, &[], "messaging", false),
        spec!("DISCORD_REPLY_TO_MODE", "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first message only, default), 'all' (reply on every chunk)", "Discord reply mode (off/first/all)", None, false, &[], "messaging", false),
        spec!("SLACK_BOT_TOKEN", "Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. Required scopes: chat:write, app_mentions:read, channels:history, groups:history, im:history, im:read, im:write, users:read, files:read, files:write", "Slack Bot Token (xoxb-...)", Some("https://api.slack.com/apps"), true, &[], "messaging", false),
        spec!("SLACK_APP_TOKEN", "Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → App-Level Tokens. Also ensure Event Subscriptions include: message.im, message.channels, message.groups, app_mention", "Slack App Token (xapp-...)", Some("https://api.slack.com/apps"), true, &[], "messaging", false),
        spec!("MATTERMOST_URL", "Mattermost server URL (e.g. https://mm.example.com)", "Mattermost server URL", Some("https://mattermost.com/deploy/"), false, &[], "messaging", false),
        spec!("MATTERMOST_TOKEN", "Mattermost bot token or personal access token", "Mattermost bot token", None, true, &[], "messaging", false),
        spec!("MATTERMOST_ALLOWED_USERS", "Comma-separated Mattermost user IDs allowed to use the bot", "Allowed Mattermost user IDs (comma-separated)", None, false, &[], "messaging", false),
        spec!("MATTERMOST_REQUIRE_MENTION", "Require @mention in Mattermost channels (default: true). Set to false to respond to all messages.", "Require @mention in channels", None, false, &[], "messaging", false),
        spec!("MATTERMOST_FREE_RESPONSE_CHANNELS", "Comma-separated Mattermost channel IDs where bot responds without @mention", "Free-response channel IDs (comma-separated)", None, false, &[], "messaging", false),
        spec!("MATRIX_HOMESERVER", "Matrix homeserver URL (e.g. https://matrix.example.org)", "Matrix homeserver URL", Some("https://matrix.org/ecosystem/servers/"), false, &[], "messaging", false),
        spec!("MATRIX_ACCESS_TOKEN", "Matrix access token (preferred over password login)", "Matrix access token", None, true, &[], "messaging", false),
        spec!("MATRIX_USER_ID", "Matrix user ID (e.g. @hermes:example.org)", "Matrix user ID (@user:server)", None, false, &[], "messaging", false),
        spec!("MATRIX_ALLOWED_USERS", "Comma-separated Matrix user IDs allowed to use the bot (@user:server format)", "Allowed Matrix user IDs (comma-separated)", None, false, &[], "messaging", false),
        spec!("MATRIX_REQUIRE_MENTION", "Require @mention in Matrix rooms (default: true). Set to false to respond to all messages.", "Require @mention in rooms (true/false)", None, false, &[], "messaging", true),
        spec!("MATRIX_FREE_RESPONSE_ROOMS", "Comma-separated Matrix room IDs where bot responds without @mention", "Free-response room IDs (comma-separated)", None, false, &[], "messaging", true),
        spec!("MATRIX_AUTO_THREAD", "Auto-create threads for messages in Matrix rooms (default: true)", "Auto-create threads in rooms (true/false)", None, false, &[], "messaging", true),
        spec!("MATRIX_DM_AUTO_THREAD", "Auto-create threads for DM messages in Matrix (default: false)", "Auto-create threads in DMs (true/false)", None, false, &[], "messaging", true),
        spec!("MATRIX_DEVICE_ID", "Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)", "Matrix device ID (stable across restarts)", None, false, &[], "messaging", true),
        spec!("MATRIX_RECOVERY_KEY", "Matrix recovery key for cross-signing verification after device key rotation (from Element: Settings → Security → Recovery Key)", "Matrix recovery key", None, true, &[], "messaging", true),
        spec!("BLUEBUBBLES_SERVER_URL", "BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)", "BlueBubbles server URL", Some("https://bluebubbles.app/"), false, &[], "messaging", false),
        spec!("BLUEBUBBLES_PASSWORD", "BlueBubbles server password (from BlueBubbles Server → Settings → API)", "BlueBubbles server password", None, true, &[], "messaging", false),
        spec!("BLUEBUBBLES_ALLOWED_USERS", "Comma-separated iMessage addresses (email or phone) allowed to use the bot", "Allowed iMessage addresses (comma-separated)", None, false, &[], "messaging", false),
        spec!("BLUEBUBBLES_ALLOW_ALL_USERS", "Allow all BlueBubbles users without allowlist", "Allow All BlueBubbles Users", None, false, &[], "messaging", false),
        spec!("QQ_APP_ID", "QQ Bot App ID from QQ Open Platform (q.qq.com)", "QQ App ID", Some("https://q.qq.com"), false, &[], "messaging", false),
        spec!("QQ_CLIENT_SECRET", "QQ Bot Client Secret from QQ Open Platform", "QQ Client Secret", None, true, &[], "messaging", false),
        spec!("QQ_ALLOWED_USERS", "Comma-separated QQ user IDs allowed to use the bot", "QQ Allowed Users", None, false, &[], "messaging", false),
        spec!("QQ_GROUP_ALLOWED_USERS", "Comma-separated QQ group IDs allowed to interact with the bot", "QQ Group Allowed Users", None, false, &[], "messaging", false),
        spec!("QQ_ALLOW_ALL_USERS", "Allow all QQ users without an allowlist (true/false)", "Allow All QQ Users", None, false, &[], "messaging", false),
        spec!("QQBOT_HOME_CHANNEL", "Default QQ channel/group for cron delivery and notifications", "QQ Home Channel", None, false, &[], "messaging", false),
        spec!("QQBOT_HOME_CHANNEL_NAME", "Display name for the QQ home channel", "QQ Home Channel Name", None, false, &[], "messaging", false),
        spec!("QQ_SANDBOX", "Enable QQ sandbox mode for development testing (true/false)", "QQ Sandbox Mode", None, false, &[], "messaging", false),
        spec!("IRC_SERVER", "IRC server hostname (e.g. irc.libera.chat)", "IRC server", None, false, &[], "messaging", false),
        spec!("IRC_CHANNEL", "IRC channel to join (e.g. #hermes)", "IRC channel", None, false, &[], "messaging", false),
        spec!("IRC_NICKNAME", "Bot nickname on IRC (default: hermes-bot)", "IRC nickname", None, false, &[], "messaging", false),
        spec!("IRC_SERVER_PASSWORD", "IRC server password (if required)", "IRC server password", None, true, &[], "messaging", true),
        spec!("IRC_NICKSERV_PASSWORD", "NickServ password for nick identification", "NickServ password", None, true, &[], "messaging", true),
        spec!("GATEWAY_ALLOW_ALL_USERS", "Allow all users to interact with messaging bots (true/false). Default: false.", "Allow all users (true/false)", None, false, &[], "messaging", true),
        spec!("API_SERVER_ENABLED", "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open WebUI, LobeChat, etc. to connect.", "Enable API server (true/false)", None, false, &[], "messaging", true),
        spec!("API_SERVER_KEY", "Bearer token for API server authentication. Required for non-loopback binding; server refuses to start without it. On loopback (127.0.0.1), all requests are allowed if empty.", "API server auth key (required for network access)", None, true, &[], "messaging", true),
        spec!("API_SERVER_PORT", "Port for the API server (default: 8642).", "API server port", None, false, &[], "messaging", true),
        spec!("API_SERVER_HOST", "Host/bind address for the API server (default: 127.0.0.1). Use 0.0.0.0 for network access — server refuses to start without API_SERVER_KEY.", "API server host", None, false, &[], "messaging", true),
        spec!("API_SERVER_MODEL_NAME", "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' for the default profile). Useful for multi-user setups with OpenWebUI.", "API server model name", None, false, &[], "messaging", true),
        spec!("GATEWAY_PROXY_URL", "URL of a remote Hermes API server to forward messages to (proxy mode). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Use for Docker E2EE containers that relay to a host agent. Also configurable via gateway.proxy_url in config.yaml.", "Remote Hermes API server URL (e.g. http://192.168.1.100:8642)", None, false, &[], "messaging", true),
        spec!("GATEWAY_PROXY_KEY", "Bearer token for authenticating with the remote Hermes API server (proxy mode). Must match the API_SERVER_KEY on the remote host.", "Remote API server auth key", None, true, &[], "messaging", true),
        spec!("WEBHOOK_ENABLED", "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.", "Enable webhooks (true/false)", None, false, &[], "messaging", false),
        spec!("WEBHOOK_PORT", "Port for the webhook HTTP server (default: 8644).", "Webhook port", None, false, &[], "messaging", false),
        spec!("WEBHOOK_SECRET", "Global HMAC secret for webhook signature validation (overridable per route in config.yaml).", "Webhook secret", None, true, &[], "messaging", false),
        // ── Agent settings ──
        spec!("SUDO_PASSWORD", "Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting", "Sudo password", None, true, &[], "setting", false),
        spec!("HERMES_MAX_ITERATIONS", "Maximum tool-calling iterations per conversation (default: 90)", "Max iterations", None, false, &[], "setting", false),
        spec!("HERMES_TOOL_PROGRESS", "(deprecated) Use display.tool_progress in config.yaml instead", "Tool progress (deprecated — use config.yaml)", None, false, &[], "setting", false),
        spec!("HERMES_TOOL_PROGRESS_MODE", "(deprecated) Use display.tool_progress in config.yaml instead", "Progress mode (deprecated — use config.yaml)", None, false, &[], "setting", false),
        spec!("HERMES_PREFILL_MESSAGES_FILE", "Path to JSON file with ephemeral prefill messages for few-shot priming", "Prefill messages file path", None, false, &[], "setting", false),
        spec!("HERMES_EPHEMERAL_SYSTEM_PROMPT", "Ephemeral system prompt injected at API-call time (never persisted to sessions)", "Ephemeral system prompt", None, false, &[], "setting", false),
    ];
    specs.into_iter().map(|s| s.into_info()).collect()
}

/// Check which environment variables are missing. Returns the info records of
/// any missing variables (required first, then optional unless `required_only`).
pub fn get_missing_env_vars(required_only: bool) -> Vec<EnvVarInfo> {
    let mut missing = Vec::new();
    for mut info in required_env_vars() {
        if get_env_value(&info.name).map(|v| v.is_empty()).unwrap_or(true) {
            info.is_required = true;
            missing.push(info);
        }
    }
    if !required_only {
        for info in optional_env_vars() {
            if get_env_value(&info.name).map(|v| v.is_empty()).unwrap_or(true) {
                missing.push(info);
            }
        }
    }
    missing
}

// =============================================================================
// Nested set / dotted-key navigation
// =============================================================================

/// Set a value at an arbitrarily nested dotted key path, supporting both dict
/// and list (numeric-index) navigation. Faithful port of `_set_nested`.
pub fn set_nested(config: &mut Value, dotted_key: &str, value: Value) -> Result<(), String> {
    let parts: Vec<&str> = dotted_key.split('.').collect();
    set_nested_inner(config, &parts, dotted_key, value)
}

fn set_nested_inner(
    mut current: &mut Value,
    parts: &[&str],
    dotted_key: &str,
    value: Value,
) -> Result<(), String> {
    for part in &parts[..parts.len() - 1] {
        match current {
            Value::Sequence(seq) => {
                let idx: usize = part.parse().map_err(|_| {
                    format!(
                        "Cannot navigate into list at key {dotted_key:?}: segment {part:?} is not a numeric index"
                    )
                })?;
                current = seq
                    .get_mut(idx)
                    .ok_or_else(|| format!("list index {idx} out of range for {dotted_key:?}"))?;
            }
            Value::Mapping(map) => {
                let key = Value::String((*part).to_string());
                let needs_fresh = match map.get(&key) {
                    None => true,
                    Some(Value::Mapping(_)) | Some(Value::Sequence(_)) => false,
                    Some(_) => true,
                };
                if needs_fresh {
                    map.insert(key.clone(), Value::Mapping(Mapping::new()));
                }
                current = map.get_mut(&key).unwrap();
            }
            other => {
                return Err(format!(
                    "Cannot navigate into {} at key {dotted_key:?}",
                    type_name(other)
                ));
            }
        }
    }
    let last = parts[parts.len() - 1];
    match current {
        Value::Sequence(seq) => {
            let idx: usize = last
                .parse()
                .map_err(|_| format!("invalid list index {last:?} in {dotted_key:?}"))?;
            if idx >= seq.len() {
                return Err(format!("list index {idx} out of range for {dotted_key:?}"));
            }
            seq[idx] = value;
        }
        Value::Mapping(map) => {
            map.insert(Value::String(last.to_string()), value);
        }
        other => {
            // mirror python: `current[last] = value` would fail; treat as a
            // mapping replacement when the leaf isn't a container.
            return Err(format!(
                "Cannot assign into {} at key {dotted_key:?}",
                type_name(other)
            ));
        }
    }
    Ok(())
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Sequence(_) => "list",
        Value::Mapping(_) => "dict",
        Value::Tagged(_) => "tagged",
    }
}

/// Check which config fields are missing (recursive). Returns (dotted_key,
/// default_value) tuples for any keys present in DEFAULT_CONFIG but absent
/// from the loaded config.
pub fn get_missing_config_fields() -> Vec<(String, Value)> {
    let config = load_config();
    let defaults = default_config();
    let mut missing = Vec::new();
    check_missing(&defaults, &config, "", &mut missing);
    missing
}

fn check_missing(defaults: &Value, current: &Value, prefix: &str, missing: &mut Vec<(String, Value)>) {
    let (Value::Mapping(def_map), cur_map) = (defaults, current) else {
        return;
    };
    let cur_map_ref = cur_map.as_mapping();
    for (key_v, default_value) in def_map {
        let Value::String(key) = key_v else { continue };
        if key.starts_with('_') {
            continue;
        }
        let full_key = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        let cur_val = cur_map_ref.and_then(|m| m.get(key_v));
        match cur_val {
            None => {
                missing.push((full_key, default_value.clone()));
            }
            Some(cv) => {
                if matches!(default_value, Value::Mapping(_)) && matches!(cv, Value::Mapping(_)) {
                    check_missing(default_value, cv, &full_key, missing);
                }
            }
        }
    }
}

// =============================================================================
// Custom provider normalisation
// =============================================================================

fn str_trimmed(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    }
}

/// Return a runtime-compatible custom provider entry or `None`.
/// Faithful port of `_normalize_custom_provider_entry`.
pub fn normalize_custom_provider_entry(entry: &Value, provider_key: &str) -> Option<Value> {
    let map = entry.as_mapping()?;
    let mut entry = map.clone();

    // api_key_env -> key_env alias.
    let key_env_k = Value::String("key_env".into());
    let api_key_env_k = Value::String("api_key_env".into());
    if entry.contains_key(&api_key_env_k) && !entry.contains_key(&key_env_k) {
        if let Some(v) = entry.get(&api_key_env_k).cloned() {
            entry.insert(key_env_k.clone(), v);
        }
    }

    // camelCase aliases.
    let camel_aliases: &[(&str, &str)] = &[
        ("apiKey", "api_key"),
        ("baseUrl", "base_url"),
        ("apiMode", "api_mode"),
        ("keyEnv", "key_env"),
        ("apiKeyEnv", "key_env"),
        ("defaultModel", "default_model"),
        ("contextLength", "context_length"),
        ("rateLimitDelay", "rate_limit_delay"),
    ];
    for (camel, snake) in camel_aliases {
        let camel_k = Value::String((*camel).to_string());
        let snake_k = Value::String((*snake).to_string());
        if entry.contains_key(&camel_k) && !entry.contains_key(&snake_k) {
            log::warn!(
                "providers.{}: camelCase key '{}' auto-mapped to '{}' (use snake_case to avoid this warning)",
                if provider_key.is_empty() { "?" } else { provider_key },
                camel,
                snake
            );
            if let Some(v) = entry.get(&camel_k).cloned() {
                entry.insert(snake_k, v);
            }
        }
    }

    // url resolution.
    let mut base_url = String::new();
    for url_key in ["base_url", "url", "api"] {
        if let Some(candidate) = str_trimmed(entry.get(Value::String(url_key.to_string()))) {
            if let Ok(parsed) = url::Url::parse(&candidate) {
                if !parsed.scheme().is_empty() && parsed.host_str().is_some() {
                    base_url = candidate;
                    break;
                }
            }
            log::warn!(
                "providers.{}: '{}' value '{}' is not a valid URL (no scheme or host) — skipped",
                if provider_key.is_empty() { "?" } else { provider_key },
                url_key,
                candidate
            );
        }
    }
    if base_url.is_empty() {
        return None;
    }

    let name = str_trimmed(entry.get(Value::String("name".into())))
        .or_else(|| {
            let pk = provider_key.trim();
            if pk.is_empty() {
                None
            } else {
                Some(pk.to_string())
            }
        });
    let name = name?;

    let mut normalized = Mapping::new();
    normalized.insert(Value::String("name".into()), Value::String(name));
    normalized.insert(Value::String("base_url".into()), Value::String(base_url));

    let pk = provider_key.trim();
    if !pk.is_empty() {
        normalized.insert(Value::String("provider_key".into()), Value::String(pk.to_string()));
    }

    if let Some(api_key) = str_trimmed(entry.get(Value::String("api_key".into()))) {
        normalized.insert(Value::String("api_key".into()), Value::String(api_key));
    }
    if let Some(key_env) = str_trimmed(entry.get(Value::String("key_env".into()))) {
        normalized.insert(Value::String("key_env".into()), Value::String(key_env));
    }

    let api_mode = str_trimmed(entry.get(Value::String("api_mode".into())))
        .or_else(|| str_trimmed(entry.get(Value::String("transport".into()))));
    if let Some(api_mode) = api_mode {
        normalized.insert(Value::String("api_mode".into()), Value::String(api_mode));
    }

    let model_name = str_trimmed(entry.get(Value::String("model".into())))
        .or_else(|| str_trimmed(entry.get(Value::String("default_model".into()))));
    if let Some(model_name) = model_name {
        normalized.insert(Value::String("model".into()), Value::String(model_name));
    }

    match entry.get(Value::String("models".into())) {
        Some(Value::Mapping(m)) if !m.is_empty() => {
            normalized.insert(Value::String("models".into()), Value::Mapping(m.clone()));
        }
        Some(Value::Sequence(seq)) if !seq.is_empty() => {
            let mut models = Mapping::new();
            for m in seq {
                if let Value::String(s) = m {
                    if !s.trim().is_empty() {
                        models.insert(Value::String(s.clone()), Value::Mapping(Mapping::new()));
                    }
                }
            }
            normalized.insert(Value::String("models".into()), Value::Mapping(models));
        }
        _ => {}
    }

    if let Some(Value::Number(n)) = entry.get(Value::String("context_length".into())) {
        if let Some(i) = n.as_i64() {
            if i > 0 {
                normalized.insert(
                    Value::String("context_length".into()),
                    Value::Number(i.into()),
                );
            }
        }
    }

    if let Some(Value::Number(n)) = entry.get(Value::String("rate_limit_delay".into())) {
        if let Some(f) = n.as_f64() {
            if f >= 0.0 {
                normalized.insert(
                    Value::String("rate_limit_delay".into()),
                    Value::Number(n.clone()),
                );
            }
        }
    }

    Some(Value::Mapping(normalized))
}

/// Normalize `providers` config entries into the legacy custom-provider shape.
pub fn providers_dict_to_custom_providers(providers_dict: &Value) -> Vec<Value> {
    let Some(map) = providers_dict.as_mapping() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (key, entry) in map {
        let key_str = match key {
            Value::String(s) => s.clone(),
            other => yaml_scalar_string(other),
        };
        if let Some(n) = normalize_custom_provider_entry(entry, &key_str) {
            out.push(n);
        }
    }
    out
}

fn yaml_scalar_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        _ => String::new(),
    }
}

/// Return a deduplicated custom-provider view across legacy and v12+ config.
pub fn get_compatible_custom_providers(config: Option<&Value>) -> Vec<Value> {
    let owned;
    let config = match config {
        Some(c) => c,
        None => {
            owned = load_config();
            &owned
        }
    };

    let mut compatible: Vec<Value> = Vec::new();
    let mut seen_provider_keys: BTreeSet<String> = BTreeSet::new();
    let mut seen_name_url_pairs: BTreeSet<(String, String, String)> = BTreeSet::new();

    let mut append_if_new = |entry: Option<Value>,
                             compatible: &mut Vec<Value>,
                             seen_pk: &mut BTreeSet<String>,
                             seen_pairs: &mut BTreeSet<(String, String, String)>| {
        let Some(entry) = entry else { return };
        let m = match entry.as_mapping() {
            Some(m) => m,
            None => return,
        };
        let get_s = |k: &str| -> String {
            m.get(Value::String(k.to_string()))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        };
        let provider_key = get_s("provider_key").to_lowercase();
        let name = get_s("name").to_lowercase();
        let base_url = get_s("base_url")
            .trim_end_matches('/')
            .to_lowercase();
        let model = get_s("model").to_lowercase();
        let pair = (name.clone(), base_url.clone(), model);

        if !provider_key.is_empty() && seen_pk.contains(&provider_key) {
            return;
        }
        if !name.is_empty() && !base_url.is_empty() && seen_pairs.contains(&pair) {
            return;
        }

        compatible.push(entry);
        if !provider_key.is_empty() {
            seen_pk.insert(provider_key);
        }
        if !name.is_empty() && !base_url.is_empty() {
            seen_pairs.insert(pair);
        }
    };

    if let Some(cp) = config.as_mapping().and_then(|m| m.get(Value::String("custom_providers".into()))) {
        match cp {
            Value::Sequence(seq) => {
                for entry in seq {
                    append_if_new(
                        normalize_custom_provider_entry(entry, ""),
                        &mut compatible,
                        &mut seen_provider_keys,
                        &mut seen_name_url_pairs,
                    );
                }
            }
            // not a list — Python returns [] in this case.
            _ => return Vec::new(),
        }
    }

    if let Some(providers) = config.as_mapping().and_then(|m| m.get(Value::String("providers".into()))) {
        for entry in providers_dict_to_custom_providers(providers) {
            append_if_new(
                Some(entry),
                &mut compatible,
                &mut seen_provider_keys,
                &mut seen_name_url_pairs,
            );
        }
    }

    compatible
}

/// Look up a per-model `context_length` override from `custom_providers`.
pub fn get_custom_provider_context_length(
    model: &str,
    base_url: &str,
    custom_providers: Option<&[Value]>,
    config: Option<&Value>,
) -> Option<i64> {
    if model.is_empty() || base_url.is_empty() {
        return None;
    }
    let resolved;
    let providers: &[Value] = match custom_providers {
        Some(cp) => cp,
        None => {
            resolved = get_compatible_custom_providers(config);
            &resolved
        }
    };

    let target_url = base_url.trim_end_matches('/');
    if target_url.is_empty() {
        return None;
    }

    for entry in providers {
        let Some(m) = entry.as_mapping() else { continue };
        let entry_url = m
            .get(Value::String("base_url".into()))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim_end_matches('/');
        if entry_url.is_empty() || entry_url != target_url {
            continue;
        }
        let Some(models) = m
            .get(Value::String("models".into()))
            .and_then(|v| v.as_mapping())
        else {
            continue;
        };
        let Some(model_cfg) = models
            .get(Value::String(model.to_string()))
            .and_then(|v| v.as_mapping())
        else {
            continue;
        };
        let raw_ctx = model_cfg.get(Value::String("context_length".into()));
        let ctx = match raw_ctx {
            Some(Value::Number(n)) => n.as_i64(),
            Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
            _ => None,
        };
        if let Some(ctx) = ctx {
            if ctx > 0 {
                return Some(ctx);
            }
        }
    }
    None
}

/// Check config version. Returns (current_version, latest_version).
pub fn check_config_version() -> (i64, i64) {
    let config = load_config();
    let current = config
        .as_mapping()
        .and_then(|m| m.get(Value::String("_config_version".into())))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    (current, CONFIG_VERSION)
}

// =============================================================================
// Config structure validation
// =============================================================================

pub const KNOWN_ROOT_KEYS: &[&str] = &[
    "_config_version",
    "model",
    "providers",
    "fallback_model",
    "fallback_providers",
    "credential_pool_strategies",
    "toolsets",
    "agent",
    "terminal",
    "display",
    "compression",
    "delegation",
    "auxiliary",
    "custom_providers",
    "context",
    "memory",
    "gateway",
    "sessions",
];

pub const CUSTOM_PROVIDER_LIKE_FIELDS: &[&str] =
    &["base_url", "api_key", "rate_limit_delay", "api_mode"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigIssue {
    pub severity: String, // "error" | "warning"
    pub message: String,
    pub hint: String,
}

impl ConfigIssue {
    fn new(severity: &str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        ConfigIssue {
            severity: severity.to_string(),
            message: message.into(),
            hint: hint.into(),
        }
    }
}

/// Validate config.yaml structure and return a list of detected issues.
pub fn validate_config_structure(config: Option<&Value>) -> Vec<ConfigIssue> {
    let owned;
    let config = match config {
        Some(c) => c,
        None => {
            owned = load_config();
            &owned
        }
    };
    let Some(root) = config.as_mapping() else {
        return vec![ConfigIssue::new(
            "error",
            "Could not load config.yaml",
            "Run 'hermes setup' to create a valid config",
        )];
    };

    let mut issues = Vec::new();

    let cp = root.get(Value::String("custom_providers".into()));

    // custom_providers must be a list, not a dict.
    if let Some(cp_val) = cp {
        match cp_val {
            Value::Mapping(cp_map) => {
                issues.push(ConfigIssue::new(
                    "error",
                    "custom_providers is a dict — it must be a YAML list (items prefixed with '-')",
                    "Change to:\n  custom_providers:\n    - name: my-provider\n      base_url: https://...\n      api_key: ...",
                ));
                let mut suspicious: Vec<String> = Vec::new();
                for k in CUSTOM_PROVIDER_LIKE_FIELDS {
                    if cp_map.contains_key(&Value::String((*k).to_string())) {
                        suspicious.push((*k).to_string());
                    }
                }
                if !suspicious.is_empty() {
                    suspicious.sort();
                    issues.push(ConfigIssue::new(
                        "warning",
                        format!(
                            "Root-level keys {} look like custom_providers entry fields",
                            fmt_str_list(&suspicious)
                        ),
                        "These should be indented under a '- name: ...' list entry, not at root level",
                    ));
                }
            }
            Value::Sequence(seq) => {
                for (i, entry) in seq.iter().enumerate() {
                    let Some(m) = entry.as_mapping() else {
                        issues.push(ConfigIssue::new(
                            "warning",
                            format!(
                                "custom_providers[{i}] is not a dict (got {})",
                                type_name(entry)
                            ),
                            "Each entry should have at minimum: name, base_url",
                        ));
                        continue;
                    };
                    if !truthy(m.get(Value::String("name".into()))) {
                        issues.push(ConfigIssue::new(
                            "warning",
                            format!("custom_providers[{i}] is missing 'name' field"),
                            "Add a name, e.g.: name: my-provider",
                        ));
                    }
                    if !truthy(m.get(Value::String("base_url".into()))) {
                        issues.push(ConfigIssue::new(
                            "warning",
                            format!("custom_providers[{i}] is missing 'base_url' field"),
                            "Add the API endpoint URL, e.g.: base_url: https://api.example.com/v1",
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    // fallback_model: single dict OR list of dicts (chain).
    let fb = root.get(Value::String("fallback_model".into()));
    if let Some(fb_val) = fb {
        match fb_val {
            Value::Sequence(seq) => {
                for (i, entry) in seq.iter().enumerate() {
                    match entry.as_mapping() {
                        None => issues.push(ConfigIssue::new(
                            "error",
                            format!(
                                "fallback_model[{i}] should be a dict, got {}",
                                type_name(entry)
                            ),
                            "Each entry needs provider + model",
                        )),
                        Some(m) => {
                            if !truthy(m.get(Value::String("provider".into()))) {
                                issues.push(ConfigIssue::new(
                                    "warning",
                                    format!("fallback_model[{i}] is missing 'provider' field"),
                                    "Add: provider: openrouter (or another provider)",
                                ));
                            }
                            if !truthy(m.get(Value::String("model".into()))) {
                                issues.push(ConfigIssue::new(
                                    "warning",
                                    format!("fallback_model[{i}] is missing 'model' field"),
                                    "Add: model: <model-name>",
                                ));
                            }
                        }
                    }
                }
            }
            Value::Mapping(m) => {
                if !m.is_empty() {
                    if !truthy(m.get(Value::String("provider".into()))) {
                        issues.push(ConfigIssue::new(
                            "warning",
                            "fallback_model is missing 'provider' field — fallback will be disabled",
                            "Add: provider: openrouter (or another provider)",
                        ));
                    }
                    if !truthy(m.get(Value::String("model".into()))) {
                        issues.push(ConfigIssue::new(
                            "warning",
                            "fallback_model is missing 'model' field — fallback will be disabled",
                            "Add: model: anthropic/claude-sonnet-4 (or another model)",
                        ));
                    }
                }
            }
            other => issues.push(ConfigIssue::new(
                "error",
                format!(
                    "fallback_model should be a dict with 'provider' and 'model', got {}",
                    type_name(other)
                ),
                "Change to:\n  fallback_model:\n    provider: openrouter\n    model: anthropic/claude-sonnet-4",
            )),
        }
    }

    // fallback_model accidentally nested inside custom_providers.
    if let Some(Value::Mapping(cp_map)) = cp {
        if !root.contains_key(&Value::String("fallback_model".into()))
            && cp_map.contains_key(&Value::String("fallback_model".into()))
        {
            issues.push(ConfigIssue::new(
                "error",
                "fallback_model appears inside custom_providers instead of at root level",
                "Move fallback_model to the top level of config.yaml (no indentation)",
            ));
        }
    }

    // model section should exist when custom_providers is configured.
    let cp_truthy = matches!(cp, Some(v) if truthy(Some(v)));
    let model_truthy = truthy(root.get(Value::String("model".into())));
    if cp_truthy && !model_truthy {
        issues.push(ConfigIssue::new(
            "warning",
            "custom_providers defined but no 'model' section — Hermes won't know which provider to use",
            "Add a model section:\n  model:\n    provider: custom\n    default: your-model-name\n    base_url: https://...",
        ));
    }

    // Root-level keys that look misplaced.
    for (key_v, _) in root {
        let Value::String(key) = key_v else { continue };
        if key.starts_with('_') {
            continue;
        }
        if !KNOWN_ROOT_KEYS.contains(&key.as_str())
            && CUSTOM_PROVIDER_LIKE_FIELDS.contains(&key.as_str())
        {
            issues.push(ConfigIssue::new(
                "warning",
                format!(
                    "Root-level key '{key}' looks misplaced — should it be under 'model:' or inside a 'custom_providers' entry?"
                ),
                format!("Move '{key}' under the appropriate section"),
            ));
        }
    }

    issues
}

fn fmt_str_list(items: &[String]) -> String {
    // Mimic python sorted-list repr: ['a', 'b']
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// `bool(value)` semantics for YAML values (used in validation).
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::Sequence(s)) => !s.is_empty(),
        Some(Value::Mapping(m)) => !m.is_empty(),
        Some(Value::Tagged(_)) => true,
    }
}

/// Print config structure warnings to stderr at startup.
pub fn print_config_warnings(config: Option<&Value>) {
    let issues = validate_config_structure(config);
    if issues.is_empty() {
        return;
    }
    let mut lines = vec!["\x1b[33m⚠ Config issues detected in config.yaml:\x1b[0m".to_string()];
    for ci in &issues {
        let marker = if ci.severity == "error" {
            "\x1b[31m✗\x1b[0m"
        } else {
            "\x1b[33m⚠\x1b[0m"
        };
        lines.push(format!("  {marker} {}", ci.message));
    }
    lines.push("  \x1b[2mRun 'hermes doctor' for fix suggestions.\x1b[0m".to_string());
    eprint!("{}\n\n", lines.join("\n"));
}

/// Warn if MESSAGING_CWD or TERMINAL_CWD is set in env instead of config.yaml.
pub fn warn_deprecated_cwd_env_vars(config: Option<&Value>) {
    let messaging_cwd = std::env::var("MESSAGING_CWD").ok();
    let terminal_cwd_env = std::env::var("TERMINAL_CWD").ok();

    let owned;
    let config = match config {
        Some(c) => c,
        None => {
            owned = load_config();
            &owned
        }
    };

    let config_cwd = config
        .as_mapping()
        .and_then(|m| m.get(Value::String("terminal".into())))
        .and_then(|t| t.as_mapping())
        .and_then(|t| t.get(Value::String("cwd".into())))
        .and_then(|v| v.as_str())
        .unwrap_or(".")
        .to_string();
    let config_has_explicit_cwd = !matches!(config_cwd.as_str(), "." | "auto" | "cwd" | "");

    let mut lines: Vec<String> = Vec::new();
    if let Some(v) = messaging_cwd.as_deref() {
        if !v.is_empty() {
            lines.push(format!(
                "  \x1b[33m⚠\x1b[0m MESSAGING_CWD={v} found in .env — this is deprecated."
            ));
        }
    }
    if let Some(v) = terminal_cwd_env.as_deref() {
        if !v.is_empty() && !config_has_explicit_cwd {
            lines.push(format!(
                "  \x1b[33m⚠\x1b[0m TERMINAL_CWD={v} found in .env — this is deprecated."
            ));
        }
    }
    if !lines.is_empty() {
        let hint_path = std::env::var("HERMES_HOME").unwrap_or_else(|_| "~/.hermes".to_string());
        lines.insert(0, "\x1b[33m⚠ Deprecated .env settings detected:\x1b[0m".to_string());
        lines.push(
            "  \x1b[2mMove to config.yaml instead:  terminal:\\n    cwd: /your/project/path\x1b[0m"
                .to_string(),
        );
        lines.push(format!(
            "  \x1b[2mThen remove the old entries from {hint_path}/.env\x1b[0m"
        ));
        eprint!("{}\n\n", lines.join("\n"));
    }
}

// =============================================================================
// Deep merge / env expansion / normalisation
// =============================================================================

/// Recursively merge `override_map` into `base`, preserving nested defaults.
pub fn deep_merge(base: &Value, override_val: &Value) -> Value {
    let (Some(base_map), Some(over_map)) = (base.as_mapping(), override_val.as_mapping()) else {
        return override_val.clone();
    };
    let mut result = base_map.clone();
    for (key, value) in over_map {
        if let Some(existing) = result.get(key) {
            if existing.as_mapping().is_some() && value.as_mapping().is_some() {
                let merged = deep_merge(existing, value);
                result.insert(key.clone(), merged);
                continue;
            }
        }
        result.insert(key.clone(), value.clone());
    }
    Value::Mapping(result)
}

/// Recursively expand `${VAR}` references in config string values.
pub fn expand_env_vars(obj: &Value) -> Value {
    match obj {
        Value::String(s) => Value::String(expand_env_in_str(s)),
        Value::Mapping(m) => {
            let mut out = Mapping::new();
            for (k, v) in m {
                out.insert(k.clone(), expand_env_vars(v));
            }
            Value::Mapping(out)
        }
        Value::Sequence(seq) => Value::Sequence(seq.iter().map(expand_env_vars).collect()),
        other => other.clone(),
    }
}

fn expand_env_in_str(s: &str) -> String {
    // Replace `${...}` where ... is any run of non-`}` chars. Unresolved refs
    // are left verbatim (full match including braces).
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(close) = s[i + 2..].find('}') {
                let name = &s[i + 2..i + 2 + close];
                let whole = &s[i..i + 2 + close + 1];
                match std::env::var(name) {
                    Ok(val) => out.push_str(&val),
                    Err(_) => out.push_str(whole),
                }
                i = i + 2 + close + 1;
                continue;
            }
        }
        // push current char (handle multibyte safely)
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Return a name-indexed map only when all items have unique string names.
fn items_by_unique_name(items: &Value) -> Option<HashMap<String, Value>> {
    let seq = items.as_sequence()?;
    let mut indexed = HashMap::new();
    for item in seq {
        let m = item.as_mapping()?;
        let name = match m.get(Value::String("name".into())) {
            Some(Value::String(s)) => s.clone(),
            _ => return None,
        };
        if indexed.contains_key(&name) {
            return None;
        }
        indexed.insert(name, item.clone());
    }
    Some(indexed)
}

/// Restore raw `${VAR}` templates when a value is otherwise unchanged.
pub fn preserve_env_ref_templates(
    current: &Value,
    raw: Option<&Value>,
    loaded_expanded: Option<&Value>,
) -> Value {
    static ENV_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = ENV_RE.get_or_init(|| regex::Regex::new(r"\$\{[^}]+\}").unwrap());

    if let (Value::String(cur), Some(Value::String(raw_s))) = (current, raw) {
        if re.is_match(raw_s) {
            if cur == raw_s {
                return Value::String(raw_s.clone());
            }
            if let Some(Value::String(le)) = loaded_expanded {
                if cur == le {
                    return Value::String(raw_s.clone());
                }
            }
            if expand_env_in_str(raw_s) == *cur {
                return Value::String(raw_s.clone());
            }
            return Value::String(cur.clone());
        }
    }

    if let (Value::Mapping(cur_map), Some(Value::Mapping(raw_map))) = (current, raw) {
        let le_map = loaded_expanded.and_then(|v| v.as_mapping());
        let mut out = Mapping::new();
        for (key, value) in cur_map {
            let raw_v = raw_map.get(key);
            let le_v = le_map.and_then(|m| m.get(key));
            out.insert(key.clone(), preserve_env_ref_templates(value, raw_v, le_v));
        }
        return Value::Mapping(out);
    }

    if let (Value::Sequence(cur_seq), Some(Value::Sequence(raw_seq))) = (current, raw) {
        let current_by_name = items_by_unique_name(current);
        let raw_by_name = items_by_unique_name(raw.unwrap());
        let loaded_by_name = loaded_expanded.and_then(items_by_unique_name);
        if let (Some(_), Some(raw_bn)) = (&current_by_name, &raw_by_name) {
            let out: Vec<Value> = cur_seq
                .iter()
                .map(|item| {
                    let name = item
                        .as_mapping()
                        .and_then(|m| m.get(Value::String("name".into())))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let raw_v = name.as_ref().and_then(|n| raw_bn.get(n));
                    let le_v = name
                        .as_ref()
                        .and_then(|n| loaded_by_name.as_ref().and_then(|lb| lb.get(n)));
                    preserve_env_ref_templates(item, raw_v, le_v)
                })
                .collect();
            return Value::Sequence(out);
        }
        let le_seq = loaded_expanded.and_then(|v| v.as_sequence());
        let out: Vec<Value> = cur_seq
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let raw_v = raw_seq.get(idx);
                let le_v = le_seq.and_then(|s| s.get(idx));
                preserve_env_ref_templates(item, raw_v, le_v)
            })
            .collect();
        return Value::Sequence(out);
    }

    current.clone()
}

/// Move stale root-level provider/base_url/context_length into model section.
pub fn normalize_root_model_keys(config: &Value) -> Value {
    let Some(map) = config.as_mapping() else {
        return config.clone();
    };
    let has_root = ["provider", "base_url", "context_length"]
        .iter()
        .any(|k| truthy(map.get(Value::String((*k).to_string()))));
    if !has_root {
        return config.clone();
    }

    let mut config = map.clone();
    let model_key = Value::String("model".into());
    let mut model_map = match config.get(&model_key) {
        Some(Value::Mapping(m)) => m.clone(),
        Some(other) if truthy(Some(other)) => {
            let mut m = Mapping::new();
            m.insert(Value::String("default".into()), other.clone());
            m
        }
        _ => Mapping::new(),
    };

    for key in ["provider", "base_url", "context_length"] {
        let key_v = Value::String(key.to_string());
        let root_val = config.get(&key_v).cloned();
        if let Some(rv) = root_val {
            if truthy(Some(&rv)) && !truthy(model_map.get(&key_v)) {
                model_map.insert(key_v.clone(), rv);
            }
        }
        config.remove(&key_v);
    }
    config.insert(model_key, Value::Mapping(model_map));
    Value::Mapping(config)
}

/// Normalize legacy root-level max_turns into agent.max_turns.
pub fn normalize_max_turns_config(config: &Value) -> Value {
    let mut map = config.as_mapping().cloned().unwrap_or_default();
    let agent_key = Value::String("agent".into());
    let max_turns_key = Value::String("max_turns".into());

    let mut agent_map = match map.get(&agent_key) {
        Some(Value::Mapping(m)) => m.clone(),
        _ => Mapping::new(),
    };

    if map.contains_key(&max_turns_key) && !agent_map.contains_key(&max_turns_key) {
        if let Some(v) = map.get(&max_turns_key).cloned() {
            agent_map.insert(max_turns_key.clone(), v);
        }
    }

    if !agent_map.contains_key(&max_turns_key) {
        let default_mt = default_config()
            .as_mapping()
            .and_then(|m| m.get(Value::String("agent".into())))
            .and_then(|a| a.as_mapping())
            .and_then(|a| a.get(Value::String("max_turns".into())))
            .cloned()
            .unwrap_or(Value::Number(90.into()));
        agent_map.insert(max_turns_key.clone(), default_mt);
    }

    map.insert(agent_key, Value::Mapping(agent_map));
    map.remove(&max_turns_key);
    Value::Mapping(map)
}

/// Traverse nested dict keys safely, returning `default` on any miss.
pub fn cfg_get<'a>(cfg: Option<&'a Value>, keys: &[&str], default: Option<&'a Value>) -> Option<&'a Value> {
    let mut node = match cfg {
        Some(c) if c.as_mapping().is_some() => c,
        _ => return default,
    };
    for key in keys {
        let Some(map) = node.as_mapping() else {
            return default;
        };
        let key_v = Value::String((*key).to_string());
        match map.get(&key_v) {
            Some(v) => node = v,
            None => return default,
        }
    }
    Some(node)
}

// =============================================================================
// Config loading / saving
// =============================================================================

fn stat_key(path: &Path) -> Option<(i128, u64)> {
    let meta = fs::metadata(path).ok()?;
    let mtime_ns = mtime_ns(&meta);
    Some((mtime_ns, meta.len()))
}

#[cfg(unix)]
fn mtime_ns(meta: &fs::Metadata) -> i128 {
    use std::os::unix::fs::MetadataExt;
    (meta.mtime() as i128) * 1_000_000_000 + (meta.mtime_nsec() as i128)
}

#[cfg(not(unix))]
fn mtime_ns(meta: &fs::Metadata) -> i128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
}

/// Read ~/.hermes/config.yaml as-is, without merging defaults or migrating.
pub fn read_raw_config() -> Value {
    let config_path = get_config_path();
    let cache_key = match stat_key(&config_path) {
        Some(k) => k,
        None => return Value::Mapping(Mapping::new()),
    };
    let path_key = config_path.to_string_lossy().to_string();

    {
        let cache = raw_config_cache().lock().unwrap();
        if let Some((mtime, size, val)) = cache.get(&path_key) {
            if (*mtime, *size) == cache_key {
                return val.clone();
            }
        }
    }

    let data = match fs::read_to_string(&config_path) {
        Ok(text) => match serde_yaml::from_str::<Value>(&text) {
            Ok(v) if v.as_mapping().is_some() => v,
            _ => Value::Mapping(Mapping::new()),
        },
        Err(_) => return Value::Mapping(Mapping::new()),
    };
    let data = if data.as_mapping().is_some() {
        data
    } else {
        Value::Mapping(Mapping::new())
    };

    let mut cache = raw_config_cache().lock().unwrap();
    cache.insert(path_key, (cache_key.0, cache_key.1, data.clone()));
    data
}

/// Load configuration from ~/.hermes/config.yaml (deep-merged with defaults,
/// normalised, env-expanded). Caches on file (mtime_ns, size).
pub fn load_config() -> Value {
    let _ = ensure_hermes_home();
    let config_path = get_config_path();
    let path_key = config_path.to_string_lossy().to_string();
    let cache_key = stat_key(&config_path);

    {
        let cache = load_config_cache().lock().unwrap();
        if let (Some(ck), Some((mtime, size, val))) = (cache_key, cache.get(&path_key)) {
            if (*mtime, *size) == ck {
                return val.clone();
            }
        }
    }

    let mut config = default_config();

    if cache_key.is_some() {
        match fs::read_to_string(&config_path) {
            Ok(text) => match serde_yaml::from_str::<Value>(&text) {
                Ok(mut user_config) => {
                    if user_config.as_mapping().is_none() {
                        user_config = Value::Mapping(Mapping::new());
                    }
                    // pull root max_turns into agent.max_turns when agent.max_turns is None
                    if let Some(uc) = user_config.as_mapping_mut() {
                        let mt_key = Value::String("max_turns".into());
                        if uc.contains_key(&mt_key) {
                            let agent_key = Value::String("agent".into());
                            let mut agent_map = match uc.get(&agent_key) {
                                Some(Value::Mapping(m)) => m.clone(),
                                _ => Mapping::new(),
                            };
                            let agent_mt = agent_map.get(&mt_key);
                            if agent_mt.is_none() || matches!(agent_mt, Some(Value::Null)) {
                                if let Some(v) = uc.get(&mt_key).cloned() {
                                    agent_map.insert(mt_key.clone(), v);
                                }
                            }
                            uc.insert(agent_key, Value::Mapping(agent_map));
                            uc.remove(&mt_key);
                        }
                    }
                    config = deep_merge(&config, &user_config);
                }
                Err(e) => {
                    println!("Warning: Failed to load config: {e}");
                }
            },
            Err(e) => {
                println!("Warning: Failed to load config: {e}");
            }
        }
    }

    let normalized = normalize_root_model_keys(&normalize_max_turns_config(&config));
    let expanded = expand_env_vars(&normalized);

    {
        let mut le = last_expanded_config_by_path().lock().unwrap();
        le.insert(path_key.clone(), expanded.clone());
    }

    {
        let mut cache = load_config_cache().lock().unwrap();
        if let Some(ck) = cache_key {
            cache.insert(path_key, (ck.0, ck.1, expanded.clone()));
        } else {
            cache.remove(&path_key);
        }
    }

    expanded
}

const SECURITY_COMMENT: &str = "
# ── Security ──────────────────────────────────────────────────────────
# Secret redaction is OFF by default — tool output (terminal stdout,
# read_file results, web content) passes through unmodified. Set
# redact_secrets to true to mask strings that look like API keys, tokens,
# and passwords before they enter the model context and logs.
# tirith pre-exec scanning is enabled by default when the tirith binary
# is available. Configure via security.tirith_* keys or env vars
# (TIRITH_ENABLED, TIRITH_BIN, TIRITH_TIMEOUT, TIRITH_FAIL_OPEN).
#
# security:
#   redact_secrets: true
#   tirith_enabled: true
#   tirith_path: \"tirith\"
#   tirith_timeout: 5
#   tirith_fail_open: true
";

const FALLBACK_COMMENT: &str = "
# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)
#
# For custom OpenAI-compatible endpoints, add base_url and key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
";

/// Save configuration to ~/.hermes/config.yaml.
pub fn save_config(config: &Value) -> Result<(), String> {
    if is_managed() {
        managed_error("save configuration");
        return Ok(());
    }
    ensure_hermes_home()?;
    let config_path = get_config_path();
    let current_normalized = normalize_root_model_keys(&normalize_max_turns_config(config));
    let mut normalized = current_normalized.clone();
    let raw_existing =
        normalize_root_model_keys(&normalize_max_turns_config(&read_raw_config()));
    if truthy(Some(&raw_existing)) {
        let le = {
            let map = last_expanded_config_by_path().lock().unwrap();
            map.get(&config_path.to_string_lossy().to_string()).cloned()
        };
        normalized = preserve_env_ref_templates(&normalized, Some(&raw_existing), le.as_ref());
    }

    let mut parts: Vec<&str> = Vec::new();
    // security comment
    let sec = normalized
        .as_mapping()
        .and_then(|m| m.get(Value::String("security".into())));
    let sec_redact_is_none = match sec {
        None | Some(Value::Null) => true,
        Some(Value::Mapping(m)) => {
            !m.contains_key(&Value::String("redact_secrets".into()))
                || m.is_empty()
        }
        _ => false,
    };
    // `not sec or sec.get("redact_secrets") is None`
    let sec_falsy = !truthy(sec);
    if sec_falsy || sec_redact_is_none {
        parts.push(SECURITY_COMMENT);
    }

    // fallback comment
    let fb = normalized
        .as_mapping()
        .and_then(|m| m.get(Value::String("fallback_model".into())));
    let fb_is_valid = match fb {
        Some(Value::Sequence(seq)) => seq.iter().any(|e| {
            e.as_mapping()
                .map(|m| {
                    truthy(m.get(Value::String("provider".into())))
                        && truthy(m.get(Value::String("model".into())))
                })
                .unwrap_or(false)
        }),
        Some(Value::Mapping(m)) => {
            truthy(m.get(Value::String("provider".into())))
                && truthy(m.get(Value::String("model".into())))
        }
        _ => false,
    };
    if !fb_is_valid {
        parts.push(FALLBACK_COMMENT);
    }

    let extra = if parts.is_empty() {
        None
    } else {
        Some(parts.concat())
    };

    crate::mod_utils::atomic_yaml_write(&config_path, &normalized, extra.as_deref())
        .map_err(|e| e.to_string())?;
    secure_file(&config_path);

    {
        let mut le = last_expanded_config_by_path().lock().unwrap();
        le.insert(config_path.to_string_lossy().to_string(), current_normalized);
    }
    Ok(())
}

// =============================================================================
// .env handling
// =============================================================================

/// Load environment variables from ~/.hermes/.env (sanitised).
pub fn load_env() -> BTreeMap<String, String> {
    let env_path = get_env_path();
    let mut env_vars = BTreeMap::new();

    if env_path.exists() {
        let raw_lines = read_lines_lossy(&env_path);
        let lines = sanitize_env_lines(&raw_lines);
        for line in lines {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') && line.contains('=') {
                if let Some((key, value)) = line.split_once('=') {
                    let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
                    env_vars.insert(key.trim().to_string(), value.to_string());
                }
            }
        }
    }
    env_vars
}

fn read_lines_lossy(path: &Path) -> Vec<String> {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    // Preserve trailing-newline-ness similar to Python readlines (keep \n).
    let mut out = Vec::new();
    for line in text.split_inclusive('\n') {
        out.push(line.to_string());
    }
    out
}

/// Fix corrupted .env lines before reading or writing.
///
/// Handles concatenated KEY=VALUE pairs and (implicitly via re-emit) stale
/// placeholder entries. Returns sanitised lines, each terminated with `\n`.
pub fn sanitize_env_lines(lines: &[String]) -> Vec<String> {
    let mut known_keys: BTreeSet<String> = BTreeSet::new();
    for info in optional_env_vars() {
        known_keys.insert(info.name);
    }
    for k in EXTRA_ENV_KEYS {
        known_keys.insert((*k).to_string());
    }

    let mut sanitized: Vec<String> = Vec::new();
    for line in lines {
        let raw = line.trim_end_matches(['\r', '\n']);
        let stripped = raw.trim();

        if stripped.is_empty() || stripped.starts_with('#') {
            sanitized.push(format!("{raw}\n"));
            continue;
        }

        // Collect full needle ranges (byte offsets).
        let mut match_ranges: Vec<(usize, usize)> = Vec::new();
        for key_name in &known_keys {
            let needle = format!("{key_name}=");
            let mut start = 0usize;
            while let Some(rel) = stripped[start..].find(&needle) {
                let idx = start + rel;
                match_ranges.push((idx, idx + needle.len()));
                start = idx + needle.len();
            }
        }

        // split positions: those not fully contained within a longer needle.
        let mut split_positions: BTreeSet<usize> = BTreeSet::new();
        for (s, e) in &match_ranges {
            let contained = match_ranges.iter().any(|(s2, e2)| {
                *s2 <= *s && *e2 >= *e && (*s2, *e2) != (*s, *e)
            });
            if !contained {
                split_positions.insert(*s);
            }
        }
        let split_positions: Vec<usize> = split_positions.into_iter().collect();

        if split_positions.len() > 1 {
            for i in 0..split_positions.len() {
                let pos = split_positions[i];
                let end = if i + 1 < split_positions.len() {
                    split_positions[i + 1]
                } else {
                    stripped.len()
                };
                let part = stripped[pos..end].trim();
                if !part.is_empty() {
                    sanitized.push(format!("{part}\n"));
                }
            }
        } else {
            sanitized.push(format!("{stripped}\n"));
        }
    }

    sanitized
}

/// Read, sanitize, and rewrite ~/.hermes/.env in place. Returns the number of
/// lines that were fixed (0 when no changes are needed).
pub fn sanitize_env_file() -> Result<usize, String> {
    let env_path = get_env_path();
    if !env_path.exists() {
        return Ok(0);
    }

    let original_lines = read_lines_lossy(&env_path);
    let sanitized = sanitize_env_lines(&original_lines);

    if sanitized == original_lines {
        return Ok(0);
    }

    let mut fixes = sanitized.len().abs_diff(original_lines.len());
    if fixes == 0 {
        let zipped = original_lines
            .iter()
            .zip(sanitized.iter())
            .filter(|(a, b)| a != b)
            .count();
        fixes = zipped + sanitized.len().abs_diff(original_lines.len());
    }

    write_env_atomic(&env_path, &sanitized, None)?;
    secure_file(&env_path);
    Ok(fixes)
}

/// Warn and strip non-ASCII characters from credential values.
///
/// Returns the sanitised (ASCII-only) value and prints a warning to stderr if
/// any non-ASCII characters were found.
pub fn check_non_ascii_credential(key: &str, value: &str) -> String {
    if value.is_ascii() {
        return value.to_string();
    }
    let mut bad_chars: Vec<String> = Vec::new();
    for (i, ch) in value.char_indices() {
        if (ch as u32) > 127 {
            bad_chars.push(format!("  position {i}: {ch:?} (U+{:04X})", ch as u32));
        }
    }
    let sanitized: String = value.chars().filter(|c| c.is_ascii()).collect();

    let mut msg = format!(
        "\n  Warning: {key} contains non-ASCII characters that will break API requests.\n  This usually happens when copy-pasting from a PDF, rich-text editor,\n  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n\n"
    );
    let shown: Vec<String> = bad_chars.iter().take(5).map(|l| format!("  {l}")).collect();
    msg.push_str(&shown.join("\n"));
    if bad_chars.len() > 5 {
        msg.push_str("\n  ... and more");
    }
    msg.push_str(
        "\n\n  The non-ASCII characters have been stripped automatically.\n  If authentication fails, re-copy the key from the provider's dashboard.\n",
    );
    eprintln!("{msg}");
    sanitized
}

/// Atomically write the given lines to `path`, optionally restoring an explicit
/// permission mode after rename (used to preserve existing file perms).
fn write_env_atomic(path: &Path, lines: &[String], restore_mode: Option<u32>) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    // Build a unique temp file in the same dir.
    let tmp_path = parent.join(format!(".env_{}.tmp", std::process::id()));
    {
        let mut f = fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        for line in lines {
            f.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        }
        f.flush().map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
    }
    if let Err(e) = crate::mod_utils::atomic_replace(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.to_string());
    }
    #[cfg(unix)]
    if let Some(mode) = restore_mode {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    let _ = restore_mode;
    Ok(())
}

#[cfg(unix)]
fn file_mode(path: &Path) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).ok().map(|m| m.permissions().mode() & 0o7777)
}

#[cfg(not(unix))]
fn file_mode(_path: &Path) -> Option<u32> {
    None
}

/// Save or update a value in ~/.hermes/.env.
pub fn save_env_value(key: &str, value: &str) -> Result<(), String> {
    if is_managed() {
        managed_error(&format!("set {key}"));
        return Ok(());
    }
    if !is_valid_env_var_name(key) {
        return Err(format!("Invalid environment variable name: {key:?}"));
    }
    let value = value.replace('\n', "").replace('\r', "");
    let value = check_non_ascii_credential(key, &value);
    ensure_hermes_home()?;
    let env_path = get_env_path();

    let mut lines: Vec<String> = Vec::new();
    if env_path.exists() {
        lines = sanitize_env_lines(&read_lines_lossy(&env_path));
    }

    let prefix = format!("{key}=");
    let mut found = false;
    for line in lines.iter_mut() {
        if line.trim_start().starts_with(&prefix) {
            *line = format!("{key}={value}\n");
            found = true;
            break;
        }
    }
    if !found {
        if let Some(last) = lines.last_mut() {
            if !last.ends_with('\n') {
                last.push('\n');
            }
        }
        lines.push(format!("{key}={value}\n"));
    }

    let original_mode = if env_path.exists() {
        file_mode(&env_path)
    } else {
        None
    };
    write_env_atomic(&env_path, &lines, original_mode)?;
    secure_file(&env_path);

    unsafe {
        std::env::set_var(key, &value);
    }
    Ok(())
}

/// Remove a key from ~/.hermes/.env and the process environment.
/// Returns true if the key was found and removed.
pub fn remove_env_value(key: &str) -> Result<bool, String> {
    if is_managed() {
        managed_error(&format!("remove {key}"));
        return Ok(false);
    }
    if !is_valid_env_var_name(key) {
        return Err(format!("Invalid environment variable name: {key:?}"));
    }
    let env_path = get_env_path();
    if !env_path.exists() {
        unsafe {
            std::env::remove_var(key);
        }
        return Ok(false);
    }

    let lines = sanitize_env_lines(&read_lines_lossy(&env_path));
    let prefix = format!("{key}=");
    let new_lines: Vec<String> = lines
        .iter()
        .filter(|line| !line.trim_start().starts_with(&prefix))
        .cloned()
        .collect();
    let found = new_lines.len() < lines.len();

    if found {
        let original_mode = file_mode(&env_path);
        write_env_atomic(&env_path, &new_lines, original_mode)?;
        secure_file(&env_path);
    }

    unsafe {
        std::env::remove_var(key);
    }
    Ok(found)
}

/// Persist an Anthropic OAuth/setup token and clear the API-key slot.
pub fn save_anthropic_oauth_token(value: &str) -> Result<(), String> {
    save_env_value("ANTHROPIC_TOKEN", value)?;
    save_env_value("ANTHROPIC_API_KEY", "")
}

/// Use Claude Code's own credential files instead of persisting env tokens.
pub fn use_anthropic_claude_code_credentials() -> Result<(), String> {
    save_env_value("ANTHROPIC_TOKEN", "")?;
    save_env_value("ANTHROPIC_API_KEY", "")
}

/// Persist an Anthropic API key and clear the OAuth/setup-token slot.
pub fn save_anthropic_api_key(value: &str) -> Result<(), String> {
    save_env_value("ANTHROPIC_API_KEY", value)?;
    save_env_value("ANTHROPIC_TOKEN", "")
}

#[derive(Debug, Clone)]
pub struct SaveEnvSecureResult {
    pub success: bool,
    pub stored_as: String,
    pub validated: bool,
}

pub fn save_env_value_secure(key: &str, value: &str) -> Result<SaveEnvSecureResult, String> {
    save_env_value(key, value)?;
    Ok(SaveEnvSecureResult {
        success: true,
        stored_as: key.to_string(),
        validated: false,
    })
}

/// Re-read ~/.hermes/.env into the process environment. Returns count of vars
/// updated (added/changed/removed).
pub fn reload_env() -> usize {
    let env_vars = load_env();
    let mut known_keys: BTreeSet<String> = BTreeSet::new();
    for info in optional_env_vars() {
        known_keys.insert(info.name);
    }
    for k in EXTRA_ENV_KEYS {
        known_keys.insert((*k).to_string());
    }

    let mut count = 0usize;
    for (key, value) in &env_vars {
        if std::env::var(key).ok().as_deref() != Some(value.as_str()) {
            unsafe {
                std::env::set_var(key, value);
            }
            count += 1;
        }
    }
    for key in &known_keys {
        if !env_vars.contains_key(key) && std::env::var(key).is_ok() {
            unsafe {
                std::env::remove_var(key);
            }
            count += 1;
        }
    }
    count
}

/// Get a value from the process environment or the ~/.hermes/.env file.
pub fn get_env_value(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        return Some(v);
    }
    load_env().get(key).cloned()
}

// =============================================================================
// set_config_value (config.yaml + .env sync)
// =============================================================================

const SET_VALUE_API_KEYS: &[&str] = &[
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
    "EXA_API_KEY",
    "PARALLEL_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_GATEWAY_URL",
    "TOOL_GATEWAY_DOMAIN",
    "TOOL_GATEWAY_SCHEME",
    "TOOL_GATEWAY_USER_TOKEN",
    "TAVILY_API_KEY",
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FAL_KEY",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "TERMINAL_SSH_HOST",
    "TERMINAL_SSH_USER",
    "TERMINAL_SSH_KEY",
    "SUDO_PASSWORD",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "GITHUB_TOKEN",
    "HONCHO_API_KEY",
    "WANDB_API_KEY",
    "TINKER_API_KEY",
];

/// config.yaml dotted-key -> .env var sync map (terminal_tool reads these).
fn config_to_env_sync(key: &str) -> Option<&'static str> {
    Some(match key {
        "terminal.backend" => "TERMINAL_ENV",
        "terminal.modal_mode" => "TERMINAL_MODAL_MODE",
        "terminal.docker_image" => "TERMINAL_DOCKER_IMAGE",
        "terminal.singularity_image" => "TERMINAL_SINGULARITY_IMAGE",
        "terminal.modal_image" => "TERMINAL_MODAL_IMAGE",
        "terminal.daytona_image" => "TERMINAL_DAYTONA_IMAGE",
        "terminal.vercel_runtime" => "TERMINAL_VERCEL_RUNTIME",
        "terminal.docker_mount_cwd_to_workspace" => "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "terminal.docker_run_as_host_user" => "TERMINAL_DOCKER_RUN_AS_HOST_USER",
        "terminal.timeout" => "TERMINAL_TIMEOUT",
        "terminal.sandbox_dir" => "TERMINAL_SANDBOX_DIR",
        "terminal.persistent_shell" => "TERMINAL_PERSISTENT_SHELL",
        "terminal.container_cpu" => "TERMINAL_CONTAINER_CPU",
        "terminal.container_memory" => "TERMINAL_CONTAINER_MEMORY",
        "terminal.container_disk" => "TERMINAL_CONTAINER_DISK",
        "terminal.container_persistent" => "TERMINAL_CONTAINER_PERSISTENT",
        _ => return None,
    })
}

/// Coerce a string value the way Python's set_config_value does, returning the
/// typed YAML value and a display string.
pub fn coerce_set_value(value: &str) -> (Value, String) {
    let lower = value.to_lowercase();
    if lower == "true" || lower == "yes" || lower == "on" {
        return (Value::Bool(true), "true".to_string());
    }
    if lower == "false" || lower == "no" || lower == "off" {
        return (Value::Bool(false), "false".to_string());
    }
    // isdigit() — all chars ASCII digits and non-empty.
    if !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(i) = value.parse::<i64>() {
            return (Value::Number(i.into()), i.to_string());
        }
    }
    // value.replace('.', '', 1).isdigit() — float check (one dot removed).
    let without_one_dot = {
        let mut s = String::with_capacity(value.len());
        let mut removed = false;
        for c in value.chars() {
            if c == '.' && !removed {
                removed = true;
                continue;
            }
            s.push(c);
        }
        s
    };
    if !without_one_dot.is_empty() && without_one_dot.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(f) = value.parse::<f64>() {
            return (Value::Number(serde_yaml::Number::from(f)), value.to_string());
        }
    }
    (Value::String(value.to_string()), value.to_string())
}

/// Set a configuration value (routes API keys to .env, otherwise config.yaml).
/// Returns a human-readable confirmation string, or an error.
pub fn set_config_value(key: &str, value: &str) -> Result<String, String> {
    if is_managed() {
        managed_error("set configuration values");
        return Ok(String::new());
    }

    let upper = key.to_uppercase();
    if SET_VALUE_API_KEYS.contains(&upper.as_str())
        || upper.ends_with("_API_KEY")
        || upper.ends_with("_TOKEN")
        || upper.starts_with("TERMINAL_SSH")
    {
        save_env_value(&upper, value)?;
        return Ok(format!("✓ Set {key} in {}", get_env_path().display()));
    }

    let config_path = get_config_path();
    let mut user_config = if config_path.exists() {
        match fs::read_to_string(&config_path) {
            Ok(text) => serde_yaml::from_str::<Value>(&text)
                .ok()
                .filter(|v| v.as_mapping().is_some())
                .unwrap_or_else(|| Value::Mapping(Mapping::new())),
            Err(_) => Value::Mapping(Mapping::new()),
        }
    } else {
        Value::Mapping(Mapping::new())
    };

    let (typed_value, display) = coerce_set_value(value);
    set_nested(&mut user_config, key, typed_value)?;

    ensure_hermes_home()?;
    crate::mod_utils::atomic_yaml_write(&config_path, &user_config, None)
        .map_err(|e| e.to_string())?;

    if let Some(env_key) = config_to_env_sync(key) {
        save_env_value(env_key, &display)?;
    }

    Ok(format!(
        "✓ Set {key} = {display} in {}",
        config_path.display()
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_home<F: FnOnce(&Path)>(f: F) {
        let dir = std::env::temp_dir().join(format!("hermes_cfg_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        f(&dir);
        unsafe {
            match prev {
                Some(p) => std::env::set_var("HERMES_HOME", p),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_valid_env_var_name() {
        assert!(is_valid_env_var_name("OPENAI_API_KEY"));
        assert!(is_valid_env_var_name("_x"));
        assert!(!is_valid_env_var_name("1ABC"));
        assert!(!is_valid_env_var_name("A-B"));
        assert!(!is_valid_env_var_name(""));
    }

    #[test]
    fn test_default_config_parses_and_version() {
        let cfg = default_config();
        let v = cfg
            .as_mapping()
            .unwrap()
            .get(Value::String("_config_version".into()))
            .unwrap()
            .as_i64()
            .unwrap();
        assert_eq!(v, CONFIG_VERSION);
        // a couple of nested defaults
        assert_eq!(
            cfg.as_mapping()
                .unwrap()
                .get(Value::String("agent".into()))
                .unwrap()
                .as_mapping()
                .unwrap()
                .get(Value::String("max_turns".into()))
                .unwrap()
                .as_i64(),
            Some(90)
        );
    }

    #[test]
    fn test_deep_merge_preserves_nested_defaults() {
        let base: Value = serde_yaml::from_str("tts:\n  elevenlabs:\n    voice_id: a\n    model_id: b\n").unwrap();
        let over: Value = serde_yaml::from_str("tts:\n  elevenlabs:\n    voice_id: c\n").unwrap();
        let merged = deep_merge(&base, &over);
        let el = merged
            .as_mapping()
            .unwrap()
            .get(Value::String("tts".into()))
            .unwrap()
            .as_mapping()
            .unwrap()
            .get(Value::String("elevenlabs".into()))
            .unwrap();
        assert_eq!(
            el.as_mapping()
                .unwrap()
                .get(Value::String("voice_id".into()))
                .unwrap()
                .as_str(),
            Some("c")
        );
        assert_eq!(
            el.as_mapping()
                .unwrap()
                .get(Value::String("model_id".into()))
                .unwrap()
                .as_str(),
            Some("b")
        );
    }

    #[test]
    fn test_expand_env_vars() {
        unsafe {
            std::env::set_var("HERMES_TEST_EXPAND", "expanded");
        }
        let v = Value::String("pre-${HERMES_TEST_EXPAND}-post-${HERMES_TEST_MISSING_XYZ}".into());
        let out = expand_env_vars(&v);
        assert_eq!(
            out.as_str(),
            Some("pre-expanded-post-${HERMES_TEST_MISSING_XYZ}")
        );
        unsafe {
            std::env::remove_var("HERMES_TEST_EXPAND");
        }
    }

    #[test]
    fn test_set_nested_dict_and_list() {
        let mut cfg: Value = serde_yaml::from_str("a:\n  b: 1\nlst:\n  - x: 1\n").unwrap();
        set_nested(&mut cfg, "a.b.c", Value::Number(5.into())).unwrap();
        // a.b was scalar -> replaced with dict, then c set
        let c = cfg
            .as_mapping()
            .unwrap()
            .get(Value::String("a".into()))
            .unwrap()
            .as_mapping()
            .unwrap()
            .get(Value::String("b".into()))
            .unwrap()
            .as_mapping()
            .unwrap()
            .get(Value::String("c".into()))
            .unwrap()
            .as_i64();
        assert_eq!(c, Some(5));

        set_nested(&mut cfg, "lst.0.x", Value::String("y".into())).unwrap();
        let x = cfg
            .as_mapping()
            .unwrap()
            .get(Value::String("lst".into()))
            .unwrap()
            .as_sequence()
            .unwrap()[0]
            .as_mapping()
            .unwrap()
            .get(Value::String("x".into()))
            .unwrap()
            .as_str();
        assert_eq!(x, Some("y"));
    }

    #[test]
    fn test_set_nested_preserves_list() {
        // #17876: an indexed path should not clobber a list-typed node.
        let mut cfg: Value =
            serde_yaml::from_str("custom_providers:\n  - name: a\n    api_key: old\n").unwrap();
        set_nested(&mut cfg, "custom_providers.0.api_key", Value::String("new".into())).unwrap();
        let cp = cfg
            .as_mapping()
            .unwrap()
            .get(Value::String("custom_providers".into()))
            .unwrap();
        assert!(cp.is_sequence());
        assert_eq!(
            cp.as_sequence().unwrap()[0]
                .as_mapping()
                .unwrap()
                .get(Value::String("api_key".into()))
                .unwrap()
                .as_str(),
            Some("new")
        );
    }

    #[test]
    fn test_normalize_root_model_keys() {
        let cfg: Value =
            serde_yaml::from_str("provider: openrouter\nbase_url: https://x\nmodel:\n  default: foo\n")
                .unwrap();
        let out = normalize_root_model_keys(&cfg);
        let m = out.as_mapping().unwrap();
        assert!(!m.contains_key(&Value::String("provider".into())));
        let model = m.get(Value::String("model".into())).unwrap().as_mapping().unwrap();
        assert_eq!(
            model.get(Value::String("provider".into())).unwrap().as_str(),
            Some("openrouter")
        );
        assert_eq!(
            model.get(Value::String("default".into())).unwrap().as_str(),
            Some("foo")
        );
    }

    #[test]
    fn test_normalize_max_turns() {
        let cfg: Value = serde_yaml::from_str("max_turns: 42\n").unwrap();
        let out = normalize_max_turns_config(&cfg);
        let m = out.as_mapping().unwrap();
        assert!(!m.contains_key(&Value::String("max_turns".into())));
        assert_eq!(
            m.get(Value::String("agent".into()))
                .unwrap()
                .as_mapping()
                .unwrap()
                .get(Value::String("max_turns".into()))
                .unwrap()
                .as_i64(),
            Some(42)
        );
    }

    #[test]
    fn test_cfg_get() {
        let cfg: Value = serde_yaml::from_str("agent:\n  reasoning_effort: high\n").unwrap();
        let got = cfg_get(Some(&cfg), &["agent", "reasoning_effort"], None);
        assert_eq!(got.unwrap().as_str(), Some("high"));
        let def = Value::String("medium".into());
        let miss = cfg_get(Some(&cfg), &["agent", "nope"], Some(&def));
        assert_eq!(miss.unwrap().as_str(), Some("medium"));
        // intermediate not a dict
        let cfg2: Value = serde_yaml::from_str("agent: oops\n").unwrap();
        let def2 = Value::String("low".into());
        assert_eq!(
            cfg_get(Some(&cfg2), &["agent", "x"], Some(&def2))
                .unwrap()
                .as_str(),
            Some("low")
        );
    }

    #[test]
    fn test_normalize_custom_provider_entry() {
        let entry: Value = serde_yaml::from_str(
            "apiKey: sk-123\nbaseUrl: https://api.example.com/v1\nname: My Provider\n",
        )
        .unwrap();
        let n = normalize_custom_provider_entry(&entry, "my-key").unwrap();
        let m = n.as_mapping().unwrap();
        assert_eq!(
            m.get(Value::String("api_key".into())).unwrap().as_str(),
            Some("sk-123")
        );
        assert_eq!(
            m.get(Value::String("base_url".into())).unwrap().as_str(),
            Some("https://api.example.com/v1")
        );
        assert_eq!(
            m.get(Value::String("provider_key".into())).unwrap().as_str(),
            Some("my-key")
        );
    }

    #[test]
    fn test_normalize_custom_provider_models_list() {
        let entry: Value = serde_yaml::from_str(
            "base_url: https://api.example.com/v1\nname: p\nmodels:\n  - foo\n  - bar\n",
        )
        .unwrap();
        let n = normalize_custom_provider_entry(&entry, "").unwrap();
        let models = n
            .as_mapping()
            .unwrap()
            .get(Value::String("models".into()))
            .unwrap()
            .as_mapping()
            .unwrap();
        assert!(models.contains_key(&Value::String("foo".into())));
        assert!(models.contains_key(&Value::String("bar".into())));
    }

    #[test]
    fn test_get_custom_provider_context_length() {
        let cp: Vec<Value> = vec![
            serde_yaml::from_str(
                "base_url: https://api.example.com/v1/\nmodels:\n  big-model:\n    context_length: 200000\n",
            )
            .unwrap(),
        ];
        let got = get_custom_provider_context_length(
            "big-model",
            "https://api.example.com/v1",
            Some(&cp),
            None,
        );
        assert_eq!(got, Some(200000));
        let miss = get_custom_provider_context_length(
            "other",
            "https://api.example.com/v1",
            Some(&cp),
            None,
        );
        assert_eq!(miss, None);
    }

    #[test]
    fn test_validate_config_structure_dict_custom_providers() {
        let cfg: Value =
            serde_yaml::from_str("custom_providers:\n  base_url: https://x\n  api_key: y\n").unwrap();
        let issues = validate_config_structure(Some(&cfg));
        assert!(issues.iter().any(|i| i.severity == "error"
            && i.message.contains("custom_providers is a dict")));
        assert!(issues
            .iter()
            .any(|i| i.severity == "warning" && i.message.contains("look like custom_providers entry fields")));
    }

    #[test]
    fn test_validate_config_structure_fallback_string() {
        let cfg: Value = serde_yaml::from_str("fallback_model: just-a-string\n").unwrap();
        let issues = validate_config_structure(Some(&cfg));
        assert!(issues
            .iter()
            .any(|i| i.severity == "error" && i.message.contains("fallback_model should be a dict")));
    }

    #[test]
    fn test_coerce_set_value() {
        assert_eq!(coerce_set_value("true").0, Value::Bool(true));
        assert_eq!(coerce_set_value("OFF").0, Value::Bool(false));
        assert_eq!(coerce_set_value("42").0.as_i64(), Some(42));
        assert!((coerce_set_value("3.14").0.as_f64().unwrap() - 3.14).abs() < 1e-9);
        assert_eq!(coerce_set_value("hello").0.as_str(), Some("hello"));
        // "1.2.3" — replace one dot -> "12.3" not all digits -> string
        assert_eq!(coerce_set_value("1.2.3").0.as_str(), Some("1.2.3"));
    }

    #[test]
    fn test_sanitize_env_lines_concatenated() {
        // Two known keys concatenated on one line.
        let lines = vec!["OPENROUTER_API_KEY=sk-aaaTAVILY_API_KEY=tvly-bbb\n".to_string()];
        let out = sanitize_env_lines(&lines);
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|l| l.trim() == "OPENROUTER_API_KEY=sk-aaa"));
        assert!(out.iter().any(|l| l.trim() == "TAVILY_API_KEY=tvly-bbb"));
    }

    #[test]
    fn test_sanitize_env_lines_overlap_suffix() {
        // GLM_API_KEY contains LM_API_KEY as a suffix; must NOT split inside.
        let lines = vec!["GLM_API_KEY=glm-secret\n".to_string()];
        let out = sanitize_env_lines(&lines);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].trim(), "GLM_API_KEY=glm-secret");
    }

    #[test]
    fn test_check_non_ascii_credential() {
        assert_eq!(check_non_ascii_credential("K", "abc123"), "abc123");
        // Cyrillic 'а' (U+0430) lookalike stripped.
        let v = "sk-\u{0430}bc";
        assert_eq!(check_non_ascii_credential("K", v), "sk-bc");
    }

    #[test]
    fn test_save_and_load_env() {
        with_temp_home(|_| {
            // ensure clean process env
            unsafe {
                std::env::remove_var("EXA_API_KEY");
            }
            save_env_value("EXA_API_KEY", "exa-123").unwrap();
            assert_eq!(get_env_value("EXA_API_KEY").as_deref(), Some("exa-123"));
            // update
            save_env_value("EXA_API_KEY", "exa-456").unwrap();
            let env = load_env();
            assert_eq!(env.get("EXA_API_KEY").map(|s| s.as_str()), Some("exa-456"));
            // remove
            let removed = remove_env_value("EXA_API_KEY").unwrap();
            assert!(removed);
            assert!(load_env().get("EXA_API_KEY").is_none());
            unsafe {
                std::env::remove_var("EXA_API_KEY");
            }
        });
    }

    #[test]
    fn test_save_and_load_config_roundtrip() {
        with_temp_home(|_| {
            let mut cfg = default_config();
            set_nested(&mut cfg, "model", Value::String("test/model".into())).unwrap();
            save_config(&cfg).unwrap();
            let loaded = load_config();
            assert_eq!(
                loaded
                    .as_mapping()
                    .unwrap()
                    .get(Value::String("model".into()))
                    .unwrap()
                    .as_str(),
                Some("test/model")
            );
        });
    }

    #[test]
    fn test_get_missing_config_fields_after_partial() {
        with_temp_home(|home| {
            // Write a minimal config missing most defaults.
            fs::write(home.join("config.yaml"), "model: foo\n_config_version: 23\n").unwrap();
            let missing = get_missing_config_fields();
            // Should report several top-level defaults as missing (e.g. terminal).
            assert!(missing.iter().any(|(k, _)| k == "terminal"));
        });
    }

    #[test]
    fn test_managed_message() {
        let prev = std::env::var("HERMES_MANAGED").ok();
        unsafe {
            std::env::set_var("HERMES_MANAGED", "nixos");
        }
        assert_eq!(get_managed_system().as_deref(), Some("NixOS"));
        let msg = format_managed_message("edit configuration");
        assert!(msg.contains("nixos-rebuild switch"));
        unsafe {
            match prev {
                Some(p) => std::env::set_var("HERMES_MANAGED", p),
                None => std::env::remove_var("HERMES_MANAGED"),
            }
        }
    }
}
