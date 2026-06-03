//! Transcription Tools — native Rust port of `tools/transcription_tools.py`.
//!
//! Provides speech-to-text transcription with several providers:
//!
//!   - **local** — faster-whisper running locally. The Rust port does **not**
//!     embed the faster-whisper Python runtime; the in-process model path is
//!     therefore reported as unavailable (`_HAS_FASTER_WHISPER == false`) and
//!     the provider resolution falls through to `local_command` / cloud
//!     providers exactly as the Python code does when the package is absent.
//!   - **local_command** — runs a configured CLI whisper command and reads back
//!     the produced `.txt` transcript.
//!   - **groq** — Groq Whisper API (OpenAI-compatible), requires `GROQ_API_KEY`.
//!   - **openai** — OpenAI Whisper API, requires `VOICE_TOOLS_OPENAI_KEY` /
//!     `OPENAI_API_KEY` (or `stt.openai.api_key`, or the managed gateway).
//!   - **mistral** — Mistral Voxtral Transcribe API, requires `MISTRAL_API_KEY`.
//!   - **xai** — xAI Grok STT API, requires `XAI_API_KEY`.
//!
//! Network providers are implemented with `reqwest::blocking`, keeping the HTTP
//! request shapes identical to the Python SDK calls (multipart form uploads to
//! the OpenAI-compatible `/audio/transcriptions` endpoint and the xAI `/stt`
//! endpoint).
//!
//! Cross-references (other ported modules):
//!   - `hermes_core::mod_utils::is_truthy_value` / `TruthyInput`
//!   - `hermes_core::tool_tool_backend_helpers::resolve_openai_audio_api_key`
//!   - `hermes_core::tool_managed_tool_gateway::resolve_managed_tool_gateway`
//!   - `hermes_core::tool_xai_http::hermes_xai_user_agent`
//!   - `hermes_core::cli_config::{load_config, get_env_value}`

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Local truthy coercion
// ---------------------------------------------------------------------------
//
// The cross-referenced helpers `hermes_core::mod_utils::is_truthy_value` and
// `TruthyInput` are not re-exported publicly from hermes-core, so a faithful
// local copy is kept here to avoid depending on private modules. Behaviour
// matches the shared `utils.is_truthy_value(value, default)`.

/// Shared set of strings treated as boolean-true (`{"1","true","yes","on"}`).
const TRUTHY_STRINGS: [&str; 4] = ["1", "true", "yes", "on"];

/// Loosely-typed value mirroring Python's `is_truthy_value` argument: `None`,
/// `bool`, `str`, or any other object (carried as its Python truthiness).
#[derive(Debug, Clone, PartialEq)]
pub enum TruthyInput {
    None,
    Bool(bool),
    Str(String),
    Other(bool),
}

fn is_truthy_str(s: &str) -> bool {
    let normalized = s.trim().to_lowercase();
    TRUTHY_STRINGS.contains(&normalized.as_str())
}

/// Faithful port of `is_truthy_value(value, default)`.
pub fn is_truthy_value(value: &TruthyInput, default: bool) -> bool {
    match value {
        TruthyInput::None => default,
        TruthyInput::Bool(b) => *b,
        TruthyInput::Str(s) => is_truthy_str(s),
        TruthyInput::Other(b) => *b,
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_PROVIDER: &str = "local";
pub const DEFAULT_LOCAL_MODEL: &str = "base";
pub const DEFAULT_LOCAL_STT_LANGUAGE: &str = "en";
pub const LOCAL_STT_COMMAND_ENV: &str = "HERMES_LOCAL_STT_COMMAND";
pub const LOCAL_STT_LANGUAGE_ENV: &str = "HERMES_LOCAL_STT_LANGUAGE";
pub const COMMON_LOCAL_BIN_DIRS: [&str; 2] = ["/opt/homebrew/bin", "/usr/local/bin"];

pub const MAX_FILE_SIZE: u64 = 25 * 1024 * 1024; // 25 MB

/// Supported input formats (lower-cased extensions including the leading dot).
pub const SUPPORTED_FORMATS: [&str; 10] = [
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".aac", ".flac",
];

/// Formats the local CLI path can consume directly without an ffmpeg pass.
pub const LOCAL_NATIVE_AUDIO_FORMATS: [&str; 3] = [".wav", ".aiff", ".aif"];

/// Known cloud-only model names used for auto-correction.
pub const OPENAI_MODELS: [&str; 3] =
    ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"];
pub const GROQ_MODELS: [&str; 3] = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
];

/// Substrings that identify a missing/unloadable CUDA runtime library.
const CUDA_LIB_ERROR_MARKERS: [&str; 8] = [
    "libcublas",
    "libcudnn",
    "libcudart",
    "cannot be loaded",
    "cannot open shared object",
    "no kernel image is available",
    "no CUDA-capable device",
    "CUDA driver version is insufficient",
];

/// The Rust port has no embedded faster-whisper runtime.
const HAS_FASTER_WHISPER: bool = false;
/// The Rust port has no embedded OpenAI Python SDK; HTTP is used directly, so
/// the OpenAI-compatible providers are always considered "available" at the
/// package level (matching `_HAS_OPENAI` being importable in deployments).
const HAS_OPENAI: bool = true;
/// The Rust port has no embedded mistralai SDK, but the HTTP path is used
/// directly. Treated as available (matches deployments that ship the SDK).
const HAS_MISTRAL: bool = true;

// ---------------------------------------------------------------------------
// Default model resolution (env-backed, matching module-load defaults)
// ---------------------------------------------------------------------------

pub fn default_stt_model() -> String {
    std::env::var("STT_OPENAI_MODEL").unwrap_or_else(|_| "whisper-1".to_string())
}

pub fn default_groq_stt_model() -> String {
    std::env::var("STT_GROQ_MODEL").unwrap_or_else(|_| "whisper-large-v3-turbo".to_string())
}

pub fn default_mistral_stt_model() -> String {
    std::env::var("STT_MISTRAL_MODEL").unwrap_or_else(|_| "voxtral-mini-latest".to_string())
}

pub fn groq_base_url() -> String {
    std::env::var("GROQ_BASE_URL").unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string())
}

pub fn openai_base_url() -> String {
    std::env::var("STT_OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string())
}

pub fn xai_stt_base_url() -> String {
    std::env::var("XAI_STT_BASE_URL").unwrap_or_else(|_| "https://api.x.ai/v1".to_string())
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a transcription attempt — mirrors the Python result dict with keys
/// `success`, `transcript`, optional `error`, and optional `provider`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    pub success: bool,
    pub transcript: String,
    pub error: Option<String>,
    pub provider: Option<String>,
}

impl TranscriptionResult {
    pub fn failure(error: impl Into<String>) -> Self {
        TranscriptionResult {
            success: false,
            transcript: String::new(),
            error: Some(error.into()),
            provider: None,
        }
    }

    pub fn ok(transcript: impl Into<String>, provider: impl Into<String>) -> Self {
        TranscriptionResult {
            success: true,
            transcript: transcript.into(),
            error: None,
            provider: Some(provider.into()),
        }
    }

    /// Serialise to the JSON object shape the Python dict produced. `error` and
    /// `provider` keys are only included when present.
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("success".into(), Value::Bool(self.success));
        map.insert("transcript".into(), Value::String(self.transcript.clone()));
        if let Some(err) = &self.error {
            map.insert("error".into(), Value::String(err.clone()));
        }
        if let Some(prov) = &self.provider {
            map.insert("provider".into(), Value::String(prov.clone()));
        }
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// Environment / config indirection — overridable hooks for testing
// ---------------------------------------------------------------------------

/// Hooks that abstract the live config / environment lookups the Python module
/// performs through `hermes_cli.config`. Defaults read the real environment and
/// the ported `hermes_core::cli_config` loaders, but tests (and callers without
/// a fully-wired config layer) can supply pure substitutes.
pub struct SttHooks<'a> {
    /// Read an env value through the live config module
    /// (`hermes_cli.config.get_env_value` with `os.getenv` fallback).
    pub get_env_value: Box<dyn Fn(&str) -> Option<String> + 'a>,
    /// Load the `stt` section of user config as a JSON object.
    pub load_stt_config: Box<dyn Fn() -> Value + 'a>,
    /// Mirrors `tools.tool_backend_helpers.managed_nous_tools_enabled`.
    pub managed_nous_tools_enabled: Box<dyn Fn() -> bool + 'a>,
    /// Resolve a managed gateway `(token, origin)` for the given vendor, or
    /// `None` when unavailable.
    pub resolve_managed_gateway: Box<dyn Fn(&str) -> Option<(String, String)> + 'a>,
}

impl<'a> Default for SttHooks<'a> {
    fn default() -> Self {
        SttHooks {
            // Faithful to Python get_env_value: read the live environment. The
            // `.env`-file fallback layer (`hermes_cli.config.get_env_value`) is
            // not re-exported publicly from hermes-core; callers that need it
            // can override this hook. `os.getenv` is the documented fallback.
            get_env_value: Box::new(|name| std::env::var(name).ok()),
            load_stt_config: Box::new(default_load_stt_config),
            // Mirrors `tools.tool_backend_helpers.managed_nous_tools_enabled`;
            // defaults to false (never blocks). Override when wired up.
            managed_nous_tools_enabled: Box::new(|| false),
            // Managed-gateway resolution lives in a private hermes-core module;
            // default to "unavailable" and let callers inject a real resolver.
            resolve_managed_gateway: Box::new(|_vendor| None),
        }
    }
}

/// Load the `stt` section from user config, falling back to an empty object.
///
/// Faithful port of `_load_stt_config`: returns `load_config().get("stt", {})`,
/// swallowing any error into an empty object. The full config loader
/// (`hermes_cli.config.load_config`) is not re-exported from hermes-core, so the
/// default returns an empty section; callers with config access should override
/// the `load_stt_config` hook.
pub fn default_load_stt_config() -> Value {
    Value::Object(serde_json::Map::new())
}

// ---------------------------------------------------------------------------
// Small config helpers
// ---------------------------------------------------------------------------

/// Borrow `obj.get(key)` as an object section, returning `None` when missing or
/// not an object (mirrors Python `dict.get(key, {})` where callers then `.get`
/// further keys).
fn section<'v>(obj: &'v Value, key: &str) -> Option<&'v serde_json::Map<String, Value>> {
    obj.get(key).and_then(|v| v.as_object())
}

/// Read a nested string value `obj[key]` when present and a non-null string.
fn get_str<'v>(obj: &'v serde_json::Map<String, Value>, key: &str) -> Option<&'v str> {
    obj.get(key).and_then(|v| v.as_str())
}

/// Return whether STT is enabled in config (`stt.enabled`, default `true`).
pub fn is_stt_enabled(stt_config: &Value) -> bool {
    let enabled = stt_config.get("enabled");
    let truthy = json_to_truthy(enabled);
    is_truthy_value(&truthy, true)
}

/// Convert an optional JSON value to a [`TruthyInput`], matching Python's
/// handling of `dict.get(...)`: missing/null -> None, bool -> Bool,
/// string -> Str, any other value -> Other(<python truthiness>).
fn json_to_truthy(value: Option<&Value>) -> TruthyInput {
    match value {
        None | Some(Value::Null) => TruthyInput::None,
        Some(Value::Bool(b)) => TruthyInput::Bool(*b),
        Some(Value::String(s)) => TruthyInput::Str(s.clone()),
        Some(Value::Number(n)) => {
            // Python bool(number): 0 -> False, anything else -> True.
            let truthy = n.as_f64().map(|f| f != 0.0).unwrap_or(true);
            TruthyInput::Other(truthy)
        }
        Some(Value::Array(a)) => TruthyInput::Other(!a.is_empty()),
        Some(Value::Object(o)) => TruthyInput::Other(!o.is_empty()),
    }
}

// ---------------------------------------------------------------------------
// Binary discovery
// ---------------------------------------------------------------------------

/// Find a local binary, checking common Homebrew/local prefixes as well as PATH.
pub fn find_binary(binary_name: &str) -> Option<String> {
    for directory in COMMON_LOCAL_BIN_DIRS {
        let candidate = Path::new(directory).join(binary_name);
        if candidate.exists() && is_executable(&candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    which(binary_name)
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111) != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Minimal `shutil.which` analogue scanning `PATH`.
fn which(binary_name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(binary_name);
        if candidate.exists() && is_executable(&candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

pub fn find_ffmpeg_binary() -> Option<String> {
    find_binary("ffmpeg")
}

pub fn find_whisper_binary() -> Option<String> {
    find_binary("whisper")
}

/// Resolve the local STT command template.
///
/// Honors `HERMES_LOCAL_STT_COMMAND` if set (non-empty after trim); otherwise
/// synthesises a template around a discovered `whisper` binary.
pub fn get_local_command_template() -> Option<String> {
    let configured = std::env::var(LOCAL_STT_COMMAND_ENV)
        .unwrap_or_default()
        .trim()
        .to_string();
    if !configured.is_empty() {
        return Some(configured);
    }

    let whisper_binary = find_whisper_binary()?;
    let quoted_binary = shell_quote(&whisper_binary);
    Some(format!(
        "{quoted_binary} {{input_path}} --model {{model}} --output_format txt \
         --output_dir {{output_dir}} --language {{language}}"
    ))
}

pub fn has_local_command() -> bool {
    get_local_command_template().is_some()
}

/// POSIX-style shell quoting, equivalent to `shlex.quote`.
fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // Safe if it contains only shell-safe characters.
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-' | '_'));
    if safe {
        return s.to_string();
    }
    // Wrap in single quotes, escaping embedded single quotes.
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

// ---------------------------------------------------------------------------
// Model normalisation
// ---------------------------------------------------------------------------

/// Return a valid faster-whisper model size, mapping cloud-only names to the
/// default. Mirrors `_normalize_local_model`.
pub fn normalize_local_model(model_name: Option<&str>) -> String {
    let is_cloud = match model_name {
        Some(m) => OPENAI_MODELS.contains(&m) || GROQ_MODELS.contains(&m),
        None => false,
    };
    match model_name {
        Some(m) if !is_cloud && !m.is_empty() => m.to_string(),
        _ => {
            if let Some(m) = model_name {
                if is_cloud {
                    log::warn!(
                        "STT model '{m}' is a cloud-only name and cannot be used with the local \
                         provider. Falling back to '{DEFAULT_LOCAL_MODEL}'. Set stt.local.model to \
                         a valid faster-whisper size (tiny, base, small, medium, large-v3)."
                    );
                }
            }
            DEFAULT_LOCAL_MODEL.to_string()
        }
    }
}

pub fn normalize_local_command_model(model_name: Option<&str>) -> String {
    normalize_local_model(model_name)
}

// ---------------------------------------------------------------------------
// CUDA error heuristic (kept for parity / potential local backend wiring)
// ---------------------------------------------------------------------------

/// Heuristic: is this error message a missing/broken CUDA runtime library?
pub fn looks_like_cuda_lib_error(msg: &str) -> bool {
    CUDA_LIB_ERROR_MARKERS.iter().any(|marker| msg.contains(marker))
}

// ---------------------------------------------------------------------------
// Provider resolution
// ---------------------------------------------------------------------------

fn env_truthy(hooks: &SttHooks<'_>, name: &str) -> bool {
    (hooks.get_env_value)(name)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

fn has_openai_audio_backend(hooks: &SttHooks<'_>, stt_config: &Value) -> bool {
    resolve_openai_audio_client_config(hooks, stt_config).is_ok()
}

/// Determine which STT provider to use.
///
/// Faithful port of `_get_provider`. Returns the provider name to dispatch on,
/// or `"none"` / an unknown passthrough string.
pub fn get_provider(hooks: &SttHooks<'_>, stt_config: &Value) -> String {
    if !is_stt_enabled(stt_config) {
        return "none".to_string();
    }

    let explicit = stt_config
        .as_object()
        .map(|m| m.contains_key("provider"))
        .unwrap_or(false);
    let provider = stt_config
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or(DEFAULT_PROVIDER)
        .to_string();

    if explicit {
        match provider.as_str() {
            "local" => {
                if HAS_FASTER_WHISPER {
                    return "local".to_string();
                }
                if has_local_command() {
                    return "local_command".to_string();
                }
                log::warn!(
                    "STT provider 'local' configured but unavailable \
                     (install faster-whisper or set HERMES_LOCAL_STT_COMMAND)"
                );
                return "none".to_string();
            }
            "local_command" => {
                if has_local_command() {
                    return "local_command".to_string();
                }
                if HAS_FASTER_WHISPER {
                    log::info!("Local STT command unavailable, using local faster-whisper");
                    return "local".to_string();
                }
                log::warn!("STT provider 'local_command' configured but unavailable");
                return "none".to_string();
            }
            "groq" => {
                if HAS_OPENAI && env_truthy(hooks, "GROQ_API_KEY") {
                    return "groq".to_string();
                }
                log::warn!("STT provider 'groq' configured but GROQ_API_KEY not set");
                return "none".to_string();
            }
            "openai" => {
                if HAS_OPENAI && has_openai_audio_backend(hooks, stt_config) {
                    return "openai".to_string();
                }
                log::warn!("STT provider 'openai' configured but no API key available");
                return "none".to_string();
            }
            "mistral" => {
                if HAS_MISTRAL && env_truthy(hooks, "MISTRAL_API_KEY") {
                    return "mistral".to_string();
                }
                log::warn!(
                    "STT provider 'mistral' configured but mistralai package \
                     not installed or MISTRAL_API_KEY not set"
                );
                return "none".to_string();
            }
            "xai" => {
                if env_truthy(hooks, "XAI_API_KEY") {
                    return "xai".to_string();
                }
                log::warn!("STT provider 'xai' configured but XAI_API_KEY not set");
                return "none".to_string();
            }
            _ => return provider, // Unknown — let it fail downstream
        }
    }

    // Auto-detect: local > local_command > groq > openai > mistral > xai
    if HAS_FASTER_WHISPER {
        return "local".to_string();
    }
    if has_local_command() {
        return "local_command".to_string();
    }
    if HAS_OPENAI && env_truthy(hooks, "GROQ_API_KEY") {
        log::info!("No local STT available, using Groq Whisper API");
        return "groq".to_string();
    }
    if HAS_OPENAI && has_openai_audio_backend(hooks, stt_config) {
        log::info!("No local STT available, using OpenAI Whisper API");
        return "openai".to_string();
    }
    if HAS_MISTRAL && env_truthy(hooks, "MISTRAL_API_KEY") {
        log::info!("No local STT available, using Mistral Voxtral Transcribe API");
        return "mistral".to_string();
    }
    if env_truthy(hooks, "XAI_API_KEY") {
        log::info!("No local STT available, using xAI Grok STT API");
        return "xai".to_string();
    }
    "none".to_string()
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

/// Validate the audio file. Returns `Some(error_result)` or `None` if OK.
pub fn validate_audio_file(file_path: &str) -> Option<TranscriptionResult> {
    let audio_path = Path::new(file_path);

    if !audio_path.exists() {
        return Some(TranscriptionResult::failure(format!(
            "Audio file not found: {file_path}"
        )));
    }
    if !audio_path.is_file() {
        return Some(TranscriptionResult::failure(format!(
            "Path is not a file: {file_path}"
        )));
    }

    let suffix = file_suffix_lower(audio_path);
    if !SUPPORTED_FORMATS.contains(&suffix.as_str()) {
        let mut sorted: Vec<&str> = SUPPORTED_FORMATS.to_vec();
        sorted.sort_unstable();
        return Some(TranscriptionResult::failure(format!(
            "Unsupported format: {}. Supported: {}",
            suffix_for_error(audio_path),
            sorted.join(", ")
        )));
    }

    match std::fs::metadata(audio_path) {
        Ok(meta) => {
            let file_size = meta.len();
            if file_size > MAX_FILE_SIZE {
                return Some(TranscriptionResult::failure(format!(
                    "File too large: {:.1}MB (max {:.0}MB)",
                    file_size as f64 / (1024.0 * 1024.0),
                    MAX_FILE_SIZE as f64 / (1024.0 * 1024.0)
                )));
            }
        }
        Err(e) => {
            return Some(TranscriptionResult::failure(format!(
                "Failed to access file: {e}"
            )));
        }
    }

    None
}

/// Lower-cased extension including the leading dot, or empty when none.
fn file_suffix_lower(path: &Path) -> String {
    match path.extension() {
        Some(ext) => format!(".{}", ext.to_string_lossy().to_lowercase()),
        None => String::new(),
    }
}

/// The original-cased suffix (with leading dot) as Python's `Path.suffix`
/// reports it for the error message.
fn suffix_for_error(path: &Path) -> String {
    match path.extension() {
        Some(ext) => format!(".{}", ext.to_string_lossy()),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Provider: local (faster-whisper) — not available in the Rust port
// ---------------------------------------------------------------------------

/// Transcribe using faster-whisper (local). The Rust port has no embedded
/// faster-whisper runtime, so this always reports unavailability, matching the
/// `not _HAS_FASTER_WHISPER` branch of the Python implementation.
pub fn transcribe_local(_file_path: &str, _model_name: &str) -> TranscriptionResult {
    if !HAS_FASTER_WHISPER {
        return TranscriptionResult::failure("faster-whisper not installed");
    }
    // Unreachable while HAS_FASTER_WHISPER is false; kept for parity.
    TranscriptionResult::failure("faster-whisper not installed")
}

// ---------------------------------------------------------------------------
// Provider: local_command
// ---------------------------------------------------------------------------

/// Normalize audio for local CLI STT when needed. Returns `(prepared_path,
/// error)`; on success `error` is `None`.
fn prepare_local_audio(file_path: &str, work_dir: &Path) -> (Option<String>, Option<String>) {
    let audio_path = Path::new(file_path);
    let suffix = file_suffix_lower(audio_path);
    if LOCAL_NATIVE_AUDIO_FORMATS.contains(&suffix.as_str()) {
        return (Some(file_path.to_string()), None);
    }

    let ffmpeg = match find_ffmpeg_binary() {
        Some(f) => f,
        None => {
            return (
                None,
                Some(
                    "Local STT fallback requires ffmpeg for non-WAV inputs, but ffmpeg was not found"
                        .to_string(),
                ),
            );
        }
    };

    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let converted_path = work_dir.join(format!("{stem}.wav"));
    let converted_str = converted_path.to_string_lossy().into_owned();

    let output = Command::new(&ffmpeg)
        .args(["-y", "-i", file_path, &converted_str])
        .output();

    match output {
        Ok(out) if out.status.success() => (Some(converted_str), None),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let details = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("ffmpeg exited with status {}", out.status)
            };
            log::error!("ffmpeg conversion failed for {file_path}: {details}");
            (None, Some(format!("Failed to convert audio for local STT: {details}")))
        }
        Err(e) => {
            let details = e.to_string();
            log::error!("ffmpeg conversion failed for {file_path}: {details}");
            (None, Some(format!("Failed to convert audio for local STT: {details}")))
        }
    }
}

/// Run the configured local STT command template and read back a `.txt`
/// transcript. Faithful port of `_transcribe_local_command`.
pub fn transcribe_local_command(
    hooks: &SttHooks<'_>,
    file_path: &str,
    model_name: &str,
) -> TranscriptionResult {
    let command_template = match get_local_command_template() {
        Some(t) => t,
        None => {
            return TranscriptionResult::failure(format!(
                "{LOCAL_STT_COMMAND_ENV} not configured and no local whisper binary was found"
            ));
        }
    };

    // Language: config.yaml (stt.local.language) > env var > "en" default.
    let stt_config = (hooks.load_stt_config)();
    let language = section(&stt_config, "local")
        .and_then(|m| get_str(m, "language"))
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var(LOCAL_STT_LANGUAGE_ENV).ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| DEFAULT_LOCAL_STT_LANGUAGE.to_string());
    let normalized_model = normalize_local_command_model(Some(model_name));

    let output_dir = match tempfile::Builder::new()
        .prefix("hermes-local-stt-")
        .tempdir()
    {
        Ok(d) => d,
        Err(e) => {
            return TranscriptionResult::failure(format!("Local transcription failed: {e}"));
        }
    };
    let output_path = output_dir.path();

    let (prepared_input, prep_error) = prepare_local_audio(file_path, output_path);
    if let Some(err) = prep_error {
        return TranscriptionResult::failure(err);
    }
    let prepared_input = prepared_input.unwrap_or_default();

    // Fill the template placeholders. A missing required placeholder mirrors the
    // Python KeyError branch.
    let filled = match fill_command_template(
        &command_template,
        &shell_quote(&prepared_input),
        &shell_quote(&output_path.to_string_lossy()),
        &shell_quote(&language),
        &shell_quote(&normalized_model),
    ) {
        Ok(s) => s,
        Err(missing) => {
            return TranscriptionResult::failure(format!(
                "Invalid {LOCAL_STT_COMMAND_ENV} template, missing placeholder: {missing}"
            ));
        }
    };

    // Run via the shell (Python used shell=True).
    let result = run_shell(&filled);
    match result {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let details = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("command exited with status {}", out.status)
            };
            log::error!("Local STT command failed for {file_path}: {details}");
            return TranscriptionResult::failure(format!("Local STT failed: {details}"));
        }
        Err(e) => {
            log::error!("Unexpected error during local command transcription: {e}");
            return TranscriptionResult::failure(format!("Local transcription failed: {e}"));
        }
    }

    // Collect *.txt files, sorted by path (matching sorted(Path.glob)).
    let mut txt_files: Vec<PathBuf> = match std::fs::read_dir(output_path) {
        Ok(entries) => entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("txt"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(e) => {
            return TranscriptionResult::failure(format!("Local transcription failed: {e}"));
        }
    };
    txt_files.sort();

    if txt_files.is_empty() {
        return TranscriptionResult::failure(
            "Local STT command completed but did not produce a .txt transcript",
        );
    }

    let transcript_text = match std::fs::read_to_string(&txt_files[0]) {
        Ok(t) => t.trim().to_string(),
        Err(e) => {
            return TranscriptionResult::failure(format!("Local transcription failed: {e}"));
        }
    };

    let name = Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    log::info!(
        "Transcribed {name} via local STT command ({normalized_model}, {} chars)",
        transcript_text.chars().count()
    );
    TranscriptionResult::ok(transcript_text, "local_command")
}

/// Fill `{input_path}`, `{output_dir}`, `{language}`, `{model}` placeholders.
/// Returns `Err(name)` for the first unknown `{placeholder}` encountered,
/// mirroring Python's `str.format` raising `KeyError`.
fn fill_command_template(
    template: &str,
    input_path: &str,
    output_dir: &str,
    language: &str,
    model: &str,
) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    out.push('{');
                    continue;
                }
                let mut name = String::new();
                let mut closed = false;
                for nc in chars.by_ref() {
                    if nc == '}' {
                        closed = true;
                        break;
                    }
                    name.push(nc);
                }
                if !closed {
                    // Unterminated brace — treat literally to avoid panicking.
                    out.push('{');
                    out.push_str(&name);
                    continue;
                }
                match name.as_str() {
                    "input_path" => out.push_str(input_path),
                    "output_dir" => out.push_str(output_dir),
                    "language" => out.push_str(language),
                    "model" => out.push_str(model),
                    other => return Err(format!("'{other}'")),
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                }
                out.push('}');
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

#[cfg(unix)]
fn run_shell(command: &str) -> std::io::Result<std::process::Output> {
    Command::new("/bin/sh").arg("-c").arg(command).output()
}

#[cfg(not(unix))]
fn run_shell(command: &str) -> std::io::Result<std::process::Output> {
    Command::new("cmd").args(["/C", command]).output()
}

// ---------------------------------------------------------------------------
// OpenAI-compatible HTTP transcription (groq + openai)
// ---------------------------------------------------------------------------

/// MIME type guess for the multipart upload based on the file extension.
fn guess_audio_mime(file_path: &str) -> &'static str {
    let suffix = file_suffix_lower(Path::new(file_path));
    match suffix.as_str() {
        ".mp3" | ".mpga" => "audio/mpeg",
        ".mp4" | ".m4a" => "audio/mp4",
        ".mpeg" => "video/mpeg",
        ".wav" => "audio/wav",
        ".webm" => "audio/webm",
        ".ogg" => "audio/ogg",
        ".aac" => "audio/aac",
        ".flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

/// Post an audio file to an OpenAI-compatible `/audio/transcriptions` endpoint.
///
/// `response_format` is the value passed to the API ("text" or "json").
/// Returns the raw response body string on HTTP 200, or an error string.
fn post_openai_transcription(
    base_url: &str,
    api_key: &str,
    model_name: &str,
    file_path: &str,
    response_format: &str,
) -> Result<String, String> {
    let bytes = std::fs::read(file_path).map_err(|e| {
        if is_permission_error(&e) {
            format!("Permission denied: {file_path}")
        } else {
            format!("Transcription failed: {e}")
        }
    })?;

    let file_name = Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".to_string());

    let part = reqwest::blocking::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str(guess_audio_mime(file_path))
        .map_err(|e| format!("Transcription failed: {e}"))?;

    let form = reqwest::blocking::multipart::Form::new()
        .text("model", model_name.to_string())
        .text("response_format", response_format.to_string())
        .part("file", part);

    let url = format!("{}/audio/transcriptions", base_url.trim_end_matches('/'));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Connection error: {e}"))?;

    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                format!("Request timeout: {e}")
            } else if e.is_connect() {
                format!("Connection error: {e}")
            } else {
                format!("Transcription failed: {e}")
            }
        })?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(format!("API error: HTTP {} {body}", status.as_u16()));
    }
    Ok(body)
}

fn is_permission_error(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::PermissionDenied
}

/// Transcribe using Groq Whisper API (OpenAI-compatible). Faithful port of
/// `_transcribe_groq`.
pub fn transcribe_groq(
    hooks: &SttHooks<'_>,
    file_path: &str,
    model_name: &str,
) -> TranscriptionResult {
    let api_key = match (hooks.get_env_value)("GROQ_API_KEY").filter(|k| !k.is_empty()) {
        Some(k) => k,
        None => return TranscriptionResult::failure("GROQ_API_KEY not set"),
    };

    if !HAS_OPENAI {
        return TranscriptionResult::failure("openai package not installed");
    }

    let mut model_name = model_name.to_string();
    if OPENAI_MODELS.contains(&model_name.as_str()) {
        let default = default_groq_stt_model();
        log::info!("Model {model_name} not available on Groq, using {default}");
        model_name = default;
    }

    match post_openai_transcription(
        &groq_base_url(),
        &api_key,
        &model_name,
        file_path,
        "text",
    ) {
        Ok(body) => {
            let transcript_text = body.trim().to_string();
            let name = file_name_of(file_path);
            log::info!(
                "Transcribed {name} via Groq API ({model_name}, {} chars)",
                transcript_text.chars().count()
            );
            TranscriptionResult::ok(transcript_text, "groq")
        }
        Err(e) => TranscriptionResult::failure(e),
    }
}

/// Transcribe using OpenAI Whisper API. Faithful port of `_transcribe_openai`.
pub fn transcribe_openai(
    hooks: &SttHooks<'_>,
    file_path: &str,
    model_name: &str,
) -> TranscriptionResult {
    let stt_config = (hooks.load_stt_config)();
    let (api_key, base_url) = match resolve_openai_audio_client_config(hooks, &stt_config) {
        Ok(v) => v,
        Err(msg) => return TranscriptionResult::failure(msg),
    };

    if !HAS_OPENAI {
        return TranscriptionResult::failure("openai package not installed");
    }

    let mut model_name = model_name.to_string();
    if GROQ_MODELS.contains(&model_name.as_str()) {
        let default = default_stt_model();
        log::info!("Model {model_name} not available on OpenAI, using {default}");
        model_name = default;
    }

    let response_format = if model_name == "whisper-1" { "text" } else { "json" };

    match post_openai_transcription(
        &base_url,
        &api_key,
        &model_name,
        file_path,
        response_format,
    ) {
        Ok(body) => {
            let transcript_text = extract_transcript_text_from_body(&body, response_format);
            let name = file_name_of(file_path);
            log::info!(
                "Transcribed {name} via OpenAI API ({model_name}, {} chars)",
                transcript_text.chars().count()
            );
            TranscriptionResult::ok(transcript_text, "openai")
        }
        Err(e) => TranscriptionResult::failure(e),
    }
}

/// Extract transcript text from an OpenAI-compatible response body.
///
/// For `response_format == "text"` the body *is* the transcript. For "json"
/// the body is `{"text": "..."}`. Falls back to the trimmed raw body.
fn extract_transcript_text_from_body(body: &str, response_format: &str) -> String {
    if response_format == "text" {
        return body.trim().to_string();
    }
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        return extract_transcript_text(&value);
    }
    body.trim().to_string()
}

/// Normalize text and JSON transcription responses to a plain string.
/// Faithful port of `_extract_transcript_text`.
pub fn extract_transcript_text(transcription: &Value) -> String {
    match transcription {
        Value::String(s) => s.trim().to_string(),
        Value::Object(map) => {
            if let Some(Value::String(s)) = map.get("text") {
                s.trim().to_string()
            } else {
                value_str(transcription).trim().to_string()
            }
        }
        other => value_str(other).trim().to_string(),
    }
}

/// Stringify a JSON value the way Python's `str(obj)` roughly would for the
/// fallback path. Strings are bare; other values use their JSON encoding.
fn value_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Provider: mistral (Voxtral Transcribe API)
// ---------------------------------------------------------------------------

/// Transcribe using Mistral Voxtral Transcribe API. The Python original uses
/// the `mistralai` SDK against `/v1/audio/transcriptions`; this port performs
/// the equivalent multipart upload directly.
///
/// Note: on any error the Python code returns only the exception *type name*
/// (`type(e).__name__`) — this port mirrors that opaque behaviour.
pub fn transcribe_mistral(
    hooks: &SttHooks<'_>,
    file_path: &str,
    model_name: &str,
) -> TranscriptionResult {
    let api_key = match (hooks.get_env_value)("MISTRAL_API_KEY").filter(|k| !k.is_empty()) {
        Some(k) => k,
        None => return TranscriptionResult::failure("MISTRAL_API_KEY not set"),
    };

    let base_url =
        std::env::var("MISTRAL_BASE_URL").unwrap_or_else(|_| "https://api.mistral.ai/v1".to_string());

    let bytes = match std::fs::read(file_path) {
        Ok(b) => b,
        Err(e) if is_permission_error(&e) => {
            return TranscriptionResult::failure(format!("Permission denied: {file_path}"));
        }
        Err(e) => {
            log::error!("Mistral transcription failed: {e}");
            // Python returns type(e).__name__; for a missing/permission file
            // that is approximately the OS error category.
            return TranscriptionResult::failure(format!(
                "Mistral transcription failed: {}",
                io_error_type_name(&e)
            ));
        }
    };

    let file_name = file_name_of(file_path);
    let part = match reqwest::blocking::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str(guess_audio_mime(file_path))
    {
        Ok(p) => p,
        Err(_) => {
            log::error!("Mistral transcription failed: invalid mime");
            return TranscriptionResult::failure("Mistral transcription failed: ValueError");
        }
    };

    let form = reqwest::blocking::multipart::Form::new()
        .text("model", model_name.to_string())
        .part("file", part);

    let url = format!("{}/audio/transcriptions", base_url.trim_end_matches('/'));
    let client = match reqwest::blocking::Client::builder().build() {
        Ok(c) => c,
        Err(_) => {
            return TranscriptionResult::failure("Mistral transcription failed: SDKError")
        }
    };

    let raw = match client.post(&url).bearer_auth(&api_key).multipart(form).send() {
        Ok(r) => r,
        Err(_) => {
            log::error!("Mistral transcription failed: request error");
            return TranscriptionResult::failure("Mistral transcription failed: SDKError");
        }
    };
    let resp = SentResponse::from_reqwest(raw);

    if !resp.status.is_success() {
        log::error!("Mistral transcription failed: HTTP {}", resp.status.as_u16());
        return TranscriptionResult::failure("Mistral transcription failed: SDKError");
    }

    let body = resp.body.unwrap_or_default();
    let value: Value = serde_json::from_str(&body).unwrap_or(Value::String(body.clone()));
    let transcript_text = extract_transcript_text(&value);
    let name = file_name_of(file_path);
    log::info!(
        "Transcribed {name} via Mistral API ({model_name}, {} chars)",
        transcript_text.chars().count()
    );
    TranscriptionResult::ok(transcript_text, "mistral")
}

fn io_error_type_name(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "FileNotFoundError".to_string(),
        std::io::ErrorKind::PermissionDenied => "PermissionError".to_string(),
        _ => "OSError".to_string(),
    }
}

// Tiny adapter so the mistral function reads status/body uniformly after the
// request is sent.
struct SentResponse {
    status: reqwest::StatusCode,
    body: Option<String>,
}

impl SentResponse {
    fn from_reqwest(resp: reqwest::blocking::Response) -> Self {
        let status = resp.status();
        let body = resp.text().ok();
        SentResponse { status, body }
    }
}

// ---------------------------------------------------------------------------
// Provider: xAI (Grok STT API)
// ---------------------------------------------------------------------------

/// Transcribe using xAI Grok STT API (`POST /v1/stt`, multipart/form-data).
/// Faithful port of `_transcribe_xai`.
pub fn transcribe_xai(
    hooks: &SttHooks<'_>,
    file_path: &str,
    _model_name: &str,
) -> TranscriptionResult {
    let api_key = match (hooks.get_env_value)("XAI_API_KEY").filter(|k| !k.is_empty()) {
        Some(k) => k,
        None => return TranscriptionResult::failure("XAI_API_KEY not set"),
    };

    let stt_config = (hooks.load_stt_config)();
    let empty = serde_json::Map::new();
    let xai_config = section(&stt_config, "xai").unwrap_or(&empty);

    let base_url = {
        let from_cfg = get_str(xai_config, "base_url").map(|s| s.to_string());
        let from_env = (hooks.get_env_value)("XAI_STT_BASE_URL");
        let chosen = from_cfg
            .filter(|s| !s.is_empty())
            .or(from_env.filter(|s| !s.is_empty()))
            .unwrap_or_else(xai_stt_base_url);
        chosen.trim().trim_end_matches('/').to_string()
    };

    let language = {
        let from_cfg = get_str(xai_config, "language").map(|s| s.to_string());
        let from_env = std::env::var("HERMES_LOCAL_STT_LANGUAGE").ok();
        let chosen = from_cfg
            .filter(|s| !s.is_empty())
            .or(from_env.filter(|s| !s.is_empty()))
            .unwrap_or_else(|| DEFAULT_LOCAL_STT_LANGUAGE.to_string());
        chosen.trim().to_string()
    };

    let use_format = is_truthy_value(&json_to_truthy_default_true(xai_config.get("format")), false);
    let use_diarize = is_truthy_value(&json_to_truthy(xai_config.get("diarize")), false);

    let bytes = match std::fs::read(file_path) {
        Ok(b) => b,
        Err(e) if is_permission_error(&e) => {
            return TranscriptionResult::failure(format!("Permission denied: {file_path}"));
        }
        Err(e) => {
            log::error!("xAI STT transcription failed: {e}");
            return TranscriptionResult::failure(format!("xAI STT transcription failed: {e}"));
        }
    };

    let file_name = file_name_of(file_path);
    let part = match reqwest::blocking::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str(guess_audio_mime(file_path))
    {
        Ok(p) => p,
        Err(e) => {
            log::error!("xAI STT transcription failed: {e}");
            return TranscriptionResult::failure(format!("xAI STT transcription failed: {e}"));
        }
    };

    let mut form = reqwest::blocking::multipart::Form::new().part("file", part);
    if !language.is_empty() {
        form = form.text("language", language.clone());
    }
    if use_format {
        form = form.text("format", "true");
    }
    if use_diarize {
        form = form.text("diarize", "true");
    }

    let url = format!("{base_url}/stt");
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(c) => c,
        Err(e) => return TranscriptionResult::failure(format!("xAI STT transcription failed: {e}")),
    };

    let resp = match client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("User-Agent", hermes_xai_user_agent())
        .multipart(form)
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            log::error!("xAI STT transcription failed: {e}");
            return TranscriptionResult::failure(format!("xAI STT transcription failed: {e}"));
        }
    };

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if status.as_u16() != 200 {
        let detail = parse_xai_error_detail(&body);
        return TranscriptionResult::failure(format!(
            "xAI STT API error (HTTP {}): {detail}",
            status.as_u16()
        ));
    }

    let value: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            log::error!("xAI STT transcription failed: {e}");
            return TranscriptionResult::failure(format!("xAI STT transcription failed: {e}"));
        }
    };

    let transcript_text = value
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if transcript_text.is_empty() {
        return TranscriptionResult::failure("xAI STT returned empty transcript");
    }

    let name = file_name_of(file_path);
    let resp_lang = value
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or(&language);
    let duration = value.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
    log::info!(
        "Transcribed {name} via xAI Grok STT (lang={resp_lang}, {duration:.1}s audio, {} chars)",
        transcript_text.chars().count()
    );

    TranscriptionResult::ok(transcript_text, "xai")
}

/// `xai_config.get("format", True)` defaults to true when the key is absent.
fn json_to_truthy_default_true(value: Option<&Value>) -> TruthyInput {
    match value {
        None => TruthyInput::Bool(true),
        other => json_to_truthy(other),
    }
}

/// Extract `error.message` (or the first 300 chars of the body) from an xAI
/// error response. Mirrors the Python `err_body.get("error", {}).get("message")`
/// then `response.text[:300]` fallback.
fn parse_xai_error_detail(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            if !msg.is_empty() {
                return msg.to_string();
            }
        }
    }
    truncate_chars(body, 300)
}

fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Return a stable Hermes-specific `User-Agent` for xAI HTTP calls.
/// Equivalent to `hermes_xai_user_agent()`: `Hermes-Agent/<version>`.
fn hermes_xai_user_agent() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let version = if version.is_empty() { "unknown" } else { version };
    format!("Hermes-Agent/{version}")
}

fn file_name_of(file_path: &str) -> String {
    Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// OpenAI audio client config resolution
// ---------------------------------------------------------------------------

/// Return direct OpenAI audio config `(api_key, base_url)` or a managed gateway
/// fallback. Faithful port of `_resolve_openai_audio_client_config`, raising a
/// `ValueError`-equivalent `Err(String)` when nothing is configured.
pub fn resolve_openai_audio_client_config(
    hooks: &SttHooks<'_>,
    stt_config: &Value,
) -> Result<(String, String), String> {
    let empty = serde_json::Map::new();
    let openai_cfg = section(stt_config, "openai").unwrap_or(&empty);
    let cfg_api_key = get_str(openai_cfg, "api_key").unwrap_or("");
    let cfg_base_url = get_str(openai_cfg, "base_url").unwrap_or("");

    if !cfg_api_key.is_empty() {
        let base = if cfg_base_url.is_empty() {
            openai_base_url()
        } else {
            cfg_base_url.to_string()
        };
        return Ok((cfg_api_key.to_string(), base));
    }

    let direct_api_key = resolve_openai_audio_api_key(hooks);
    if !direct_api_key.is_empty() {
        return Ok((direct_api_key, openai_base_url()));
    }

    match (hooks.resolve_managed_gateway)("openai-audio") {
        Some((token, origin)) => {
            let base = join_v1(&origin);
            Ok((token, base))
        }
        None => {
            let mut message =
                "Neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
                    .to_string();
            if (hooks.managed_nous_tools_enabled)() {
                message.push_str(", and the managed OpenAI audio gateway is unavailable");
            }
            Err(message)
        }
    }
}

/// `urljoin(f"{origin.rstrip('/')}/", "v1")`.
fn join_v1(origin: &str) -> String {
    format!("{}/v1", origin.trim_end_matches('/'))
}

/// Resolve a direct OpenAI audio API key. Faithful port of
/// `tools.tool_backend_helpers.resolve_openai_audio_api_key`:
/// `VOICE_TOOLS_OPENAI_KEY` or `OPENAI_API_KEY`, trimmed.
fn resolve_openai_audio_api_key(hooks: &SttHooks<'_>) -> String {
    let voice = (hooks.get_env_value)("VOICE_TOOLS_OPENAI_KEY").unwrap_or_default();
    let chosen = if !voice.is_empty() {
        voice
    } else {
        (hooks.get_env_value)("OPENAI_API_KEY").unwrap_or_default()
    };
    chosen.trim().to_string()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Transcribe an audio file using the configured STT provider.
///
/// Faithful port of `transcribe_audio`. `model` overrides the provider model
/// when `Some`.
pub fn transcribe_audio(file_path: &str, model: Option<&str>) -> TranscriptionResult {
    transcribe_audio_with(&SttHooks::default(), file_path, model)
}

/// As [`transcribe_audio`] but with injectable [`SttHooks`].
pub fn transcribe_audio_with(
    hooks: &SttHooks<'_>,
    file_path: &str,
    model: Option<&str>,
) -> TranscriptionResult {
    if let Some(err) = validate_audio_file(file_path) {
        return err;
    }

    let stt_config = (hooks.load_stt_config)();
    if !is_stt_enabled(&stt_config) {
        return TranscriptionResult::failure(
            "STT is disabled in config.yaml (stt.enabled: false).",
        );
    }

    let provider = get_provider(hooks, &stt_config);
    let model_opt = model.filter(|m| !m.is_empty());

    match provider.as_str() {
        "local" => {
            let local_cfg = section(&stt_config, "local");
            let cfg_model = local_cfg.and_then(|m| get_str(m, "model"));
            let chosen = model_opt.or(cfg_model).unwrap_or(DEFAULT_LOCAL_MODEL);
            let model_name = normalize_local_model(Some(chosen));
            transcribe_local(file_path, &model_name)
        }
        "local_command" => {
            let local_cfg = section(&stt_config, "local");
            let cfg_model = local_cfg.and_then(|m| get_str(m, "model"));
            let chosen = model_opt.or(cfg_model).unwrap_or(DEFAULT_LOCAL_MODEL);
            let model_name = normalize_local_command_model(Some(chosen));
            transcribe_local_command(hooks, file_path, &model_name)
        }
        "groq" => {
            let default = default_groq_stt_model();
            let model_name = model_opt.unwrap_or(&default);
            transcribe_groq(hooks, file_path, model_name)
        }
        "openai" => {
            let openai_cfg = section(&stt_config, "openai");
            let cfg_model = openai_cfg.and_then(|m| get_str(m, "model"));
            let default = default_stt_model();
            let model_name = model_opt.or(cfg_model).unwrap_or(&default);
            transcribe_openai(hooks, file_path, model_name)
        }
        "mistral" => {
            let mistral_cfg = section(&stt_config, "mistral");
            let cfg_model = mistral_cfg.and_then(|m| get_str(m, "model"));
            let default = default_mistral_stt_model();
            let model_name = model_opt.or(cfg_model).unwrap_or(&default);
            transcribe_mistral(hooks, file_path, model_name)
        }
        "xai" => {
            let model_name = model_opt.unwrap_or("grok-stt");
            transcribe_xai(hooks, file_path, model_name)
        }
        _ => TranscriptionResult::failure(format!(
            "No STT provider available. Install faster-whisper for free local \
             transcription, configure {LOCAL_STT_COMMAND_ENV} or install a local whisper CLI, \
             set GROQ_API_KEY for free Groq Whisper, set MISTRAL_API_KEY for Mistral \
             Voxtral Transcribe, set XAI_API_KEY for xAI Grok STT, or set VOICE_TOOLS_OPENAI_KEY \
             or OPENAI_API_KEY for the OpenAI Whisper API."
        )),
    }
}

// Make the BTreeSet import used (sorted supported formats helper alternative).
#[allow(dead_code)]
fn supported_formats_sorted() -> BTreeSet<&'static str> {
    SUPPORTED_FORMATS.iter().copied().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn empty_hooks<'a>(
        env: std::collections::HashMap<String, String>,
        config: Value,
    ) -> SttHooks<'a> {
        SttHooks {
            get_env_value: Box::new(move |name| env.get(name).cloned()),
            load_stt_config: Box::new(move || config.clone()),
            managed_nous_tools_enabled: Box::new(|| false),
            resolve_managed_gateway: Box::new(|_| None),
        }
    }

    #[test]
    fn is_stt_enabled_defaults_true() {
        assert!(is_stt_enabled(&json!({})));
        assert!(is_stt_enabled(&json!({"enabled": true})));
        assert!(is_stt_enabled(&json!({"enabled": "yes"})));
        assert!(!is_stt_enabled(&json!({"enabled": false})));
        assert!(!is_stt_enabled(&json!({"enabled": "no"})));
        assert!(!is_stt_enabled(&json!({"enabled": "0"})));
    }

    #[test]
    fn normalize_local_model_maps_cloud_names() {
        assert_eq!(normalize_local_model(Some("base")), "base");
        assert_eq!(normalize_local_model(Some("small")), "small");
        assert_eq!(normalize_local_model(Some("whisper-1")), DEFAULT_LOCAL_MODEL);
        assert_eq!(
            normalize_local_model(Some("whisper-large-v3-turbo")),
            DEFAULT_LOCAL_MODEL
        );
        assert_eq!(normalize_local_model(None), DEFAULT_LOCAL_MODEL);
        assert_eq!(normalize_local_model(Some("")), DEFAULT_LOCAL_MODEL);
    }

    #[test]
    fn validate_missing_file() {
        let r = validate_audio_file("/no/such/file/here.ogg").unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("Audio file not found"));
    }

    #[test]
    fn validate_unsupported_format() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("note.txt");
        std::fs::write(&p, b"hello").unwrap();
        let r = validate_audio_file(p.to_str().unwrap()).unwrap();
        assert!(!r.success);
        let err = r.error.unwrap();
        assert!(err.contains("Unsupported format"));
        assert!(err.contains(".txt"));
    }

    #[test]
    fn validate_supported_ok() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("clip.ogg");
        std::fs::write(&p, b"audio-bytes").unwrap();
        assert!(validate_audio_file(p.to_str().unwrap()).is_none());
    }

    #[test]
    fn validate_too_large() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.wav");
        // Sparse-ish file just over the limit.
        let f = std::fs::File::create(&p).unwrap();
        f.set_len(MAX_FILE_SIZE + 1).unwrap();
        let r = validate_audio_file(p.to_str().unwrap()).unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("File too large"));
    }

    #[test]
    fn extract_transcript_variants() {
        assert_eq!(extract_transcript_text(&json!("  hi  ")), "hi");
        assert_eq!(extract_transcript_text(&json!({"text": "  yo "})), "yo");
        // No text key falls back to the JSON encoding.
        let other = extract_transcript_text(&json!({"foo": 1}));
        assert!(other.contains("foo"));
    }

    #[test]
    fn extract_from_body_text_and_json() {
        assert_eq!(extract_transcript_text_from_body("  hello ", "text"), "hello");
        assert_eq!(
            extract_transcript_text_from_body("{\"text\": \" world \"}", "json"),
            "world"
        );
    }

    #[test]
    fn fill_template_basic() {
        let t = "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}";
        let s = fill_command_template(t, "IN", "OUT", "en", "base").unwrap();
        assert_eq!(s, "whisper IN --model base --output_dir OUT --language en");
    }

    #[test]
    fn fill_template_unknown_placeholder() {
        let err = fill_command_template("x {bogus}", "i", "o", "l", "m").unwrap_err();
        assert_eq!(err, "'bogus'");
    }

    #[test]
    fn fill_template_escaped_braces() {
        let s = fill_command_template("{{literal}} {model}", "i", "o", "l", "m").unwrap();
        assert_eq!(s, "{literal} m");
    }

    #[test]
    fn shell_quote_cases() {
        assert_eq!(shell_quote("simple"), "simple");
        assert_eq!(shell_quote("/usr/bin/whisper"), "/usr/bin/whisper");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn xai_format_defaults_true() {
        let t = json_to_truthy_default_true(None);
        assert!(is_truthy_value(&t, false));
        let f = json_to_truthy_default_true(Some(&json!(false)));
        assert!(!is_truthy_value(&f, false));
    }

    #[test]
    fn provider_disabled_returns_none() {
        let hooks = empty_hooks(Default::default(), json!({"enabled": false}));
        assert_eq!(get_provider(&hooks, &json!({"enabled": false})), "none");
    }

    #[test]
    fn provider_explicit_groq_requires_key() {
        let cfg = json!({"provider": "groq"});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "none");

        let mut env = std::collections::HashMap::new();
        env.insert("GROQ_API_KEY".to_string(), "k".to_string());
        let hooks = empty_hooks(env, cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "groq");
    }

    #[test]
    fn provider_explicit_xai_requires_key() {
        let cfg = json!({"provider": "xai"});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "none");

        let mut env = std::collections::HashMap::new();
        env.insert("XAI_API_KEY".to_string(), "k".to_string());
        let hooks = empty_hooks(env, cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "xai");
    }

    #[test]
    fn provider_unknown_passthrough() {
        let cfg = json!({"provider": "weird"});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "weird");
    }

    #[test]
    fn provider_explicit_openai_no_key() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("VOICE_TOOLS_OPENAI_KEY");
            std::env::remove_var("OPENAI_API_KEY");
        }
        let cfg = json!({"provider": "openai"});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "none");
    }

    #[test]
    fn provider_openai_with_config_key() {
        let cfg = json!({"provider": "openai", "openai": {"api_key": "sk-test"}});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        assert_eq!(get_provider(&hooks, &cfg), "openai");
    }

    #[test]
    fn resolve_openai_config_from_section() {
        let cfg = json!({"openai": {"api_key": "sk-1", "base_url": "https://x/v1"}});
        let hooks = empty_hooks(Default::default(), cfg.clone());
        let (k, b) = resolve_openai_audio_client_config(&hooks, &cfg).unwrap();
        assert_eq!(k, "sk-1");
        assert_eq!(b, "https://x/v1");
    }

    #[test]
    fn resolve_openai_config_managed_gateway() {
        let cfg = json!({});
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("VOICE_TOOLS_OPENAI_KEY");
            std::env::remove_var("OPENAI_API_KEY");
        }
        let hooks = SttHooks {
            get_env_value: Box::new(|_| None),
            load_stt_config: Box::new(move || json!({})),
            managed_nous_tools_enabled: Box::new(|| true),
            resolve_managed_gateway: Box::new(|_| {
                Some(("tok".to_string(), "https://gw.example.com/".to_string()))
            }),
        };
        let (k, b) = resolve_openai_audio_client_config(&hooks, &cfg).unwrap();
        assert_eq!(k, "tok");
        assert_eq!(b, "https://gw.example.com/v1");
    }

    #[test]
    fn resolve_openai_config_error_message() {
        let cfg = json!({});
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("VOICE_TOOLS_OPENAI_KEY");
            std::env::remove_var("OPENAI_API_KEY");
        }
        let hooks = SttHooks {
            get_env_value: Box::new(|_| None),
            load_stt_config: Box::new(move || json!({})),
            managed_nous_tools_enabled: Box::new(|| true),
            resolve_managed_gateway: Box::new(|_| None),
        };
        let err = resolve_openai_audio_client_config(&hooks, &cfg).unwrap_err();
        assert!(err.contains("managed OpenAI audio gateway is unavailable"));
    }

    #[test]
    fn join_v1_strips_trailing_slash() {
        assert_eq!(join_v1("https://h/"), "https://h/v1");
        assert_eq!(join_v1("https://h"), "https://h/v1");
    }

    #[test]
    fn parse_xai_error_uses_message_then_body() {
        assert_eq!(
            parse_xai_error_detail("{\"error\": {\"message\": \"bad\"}}"),
            "bad"
        );
        assert_eq!(parse_xai_error_detail("plain text body"), "plain text body");
    }

    #[test]
    fn missing_key_providers_fail_fast() {
        let hooks = empty_hooks(Default::default(), json!({}));
        let r = transcribe_groq(&hooks, "x.ogg", "whisper-large-v3");
        assert!(!r.success);
        assert_eq!(r.error.unwrap(), "GROQ_API_KEY not set");

        let r = transcribe_mistral(&hooks, "x.ogg", "voxtral-mini-latest");
        assert!(!r.success);
        assert_eq!(r.error.unwrap(), "MISTRAL_API_KEY not set");

        let r = transcribe_xai(&hooks, "x.ogg", "grok-stt");
        assert!(!r.success);
        assert_eq!(r.error.unwrap(), "XAI_API_KEY not set");
    }
}
