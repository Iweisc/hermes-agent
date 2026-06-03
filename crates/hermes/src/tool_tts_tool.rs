//! Text-to-Speech tool — native Rust port of `tools/tts_tool.py`.
//!
//! Built-in TTS providers:
//! - Edge TTS (default, free, no API key) — handled externally / not synthesised
//!   here (no native edge-tts client in Rust), but config + dispatch is modeled.
//! - ElevenLabs (premium): needs `ELEVENLABS_API_KEY`.
//! - OpenAI TTS: needs `OPENAI_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`.
//! - MiniMax TTS: needs `MINIMAX_API_KEY`.
//! - Mistral (Voxtral TTS): needs `MISTRAL_API_KEY`.
//! - Google Gemini TTS: needs `GEMINI_API_KEY` / `GOOGLE_API_KEY`.
//! - xAI TTS: needs `XAI_API_KEY`.
//! - NeuTTS / KittenTTS / Piper (local engines).
//!
//! Custom command providers: users declare named providers with `type: command`
//! under `tts.providers.<name>` in `~/.hermes/config.yaml`.
//!
//! This port reproduces the Python behavior for:
//!  * per-provider max-text-length resolution (incl. ElevenLabs model-aware table),
//!  * command-provider config resolution, shell-quote-aware template rendering,
//!    timeout / output-format handling and command execution,
//!  * the network providers (request construction + response parsing) via
//!    `reqwest::blocking`,
//!  * WAV RIFF header wrapping for Gemini PCM,
//!  * markdown stripping and the streaming sentence-buffer state machine,
//!  * the top-level `text_to_speech_tool` dispatch + JSON result shape.
//!
//! Where the Python relied on Python-only client libraries (edge-tts, the
//! ElevenLabs/OpenAI/Mistral SDKs, sounddevice, piper, kittentts, neutts), the
//! corresponding synth is implemented directly via HTTP where the provider
//! exposes a documented HTTP shape (ElevenLabs/OpenAI/Mistral), and is surfaced
//! as a structured "engine unavailable" error otherwise — matching the Python
//! "package not installed" guard rails.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

// Cross-refs into already-ported hermes-core modules.
use crate::tool_tool_backend_helpers::{prefers_gateway, resolve_openai_audio_api_key};
use crate::tool_xai_http::hermes_xai_user_agent;

// ===========================================================================
// Defaults
// ===========================================================================
pub const DEFAULT_PROVIDER: &str = "edge";
pub const DEFAULT_EDGE_VOICE: &str = "en-US-AriaNeural";
pub const DEFAULT_ELEVENLABS_VOICE_ID: &str = "pNInz6obpgDQGcFmaJgB"; // Adam
pub const DEFAULT_ELEVENLABS_MODEL_ID: &str = "eleven_multilingual_v2";
pub const DEFAULT_ELEVENLABS_STREAMING_MODEL_ID: &str = "eleven_flash_v2_5";
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini-tts";
pub const DEFAULT_KITTENTTS_MODEL: &str = "KittenML/kitten-tts-nano-0.8-int8";
pub const DEFAULT_KITTENTTS_VOICE: &str = "Jasper";
pub const DEFAULT_PIPER_VOICE: &str = "en_US-lessac-medium";
pub const DEFAULT_OPENAI_VOICE: &str = "alloy";
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MINIMAX_MODEL: &str = "speech-01";
pub const DEFAULT_MINIMAX_VOICE_ID: &str = "female-shaonv";
pub const DEFAULT_MINIMAX_BASE_URL: &str = "https://api.minimax.chat/v1/text_to_speech";
pub const DEFAULT_MISTRAL_TTS_MODEL: &str = "voxtral-mini-tts-2603";
pub const DEFAULT_MISTRAL_TTS_VOICE_ID: &str = "c69964a6-ab8b-4f8a-9465-ec0925096ec8";
pub const DEFAULT_XAI_VOICE_ID: &str = "eve";
pub const DEFAULT_XAI_LANGUAGE: &str = "en";
pub const DEFAULT_XAI_SAMPLE_RATE: i64 = 24000;
pub const DEFAULT_XAI_BIT_RATE: i64 = 128000;
pub const DEFAULT_XAI_BASE_URL: &str = "https://api.x.ai/v1";
pub const DEFAULT_GEMINI_TTS_MODEL: &str = "gemini-2.5-flash-preview-tts";
pub const DEFAULT_GEMINI_TTS_VOICE: &str = "Kore";
pub const DEFAULT_GEMINI_TTS_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// PCM output specs for Gemini TTS (fixed by the API)
pub const GEMINI_TTS_SAMPLE_RATE: u32 = 24000;
pub const GEMINI_TTS_CHANNELS: u16 = 1;
pub const GEMINI_TTS_SAMPLE_WIDTH: u16 = 2; // 16-bit PCM (L16)

pub const FALLBACK_MAX_TEXT_LENGTH: i64 = 4000;
/// Back-compat alias.
pub const MAX_TEXT_LENGTH: i64 = FALLBACK_MAX_TEXT_LENGTH;

pub const DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS: f64 = 120.0;
pub const DEFAULT_COMMAND_TTS_OUTPUT_FORMAT: &str = "mp3";
pub const DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH: i64 = 5000;

/// Built-in provider names. Any `tts.provider` value NOT in this set is
/// interpreted as a reference to `tts.providers.<name>`.
pub fn builtin_tts_providers() -> &'static [&'static str] {
    &[
        "edge",
        "elevenlabs",
        "openai",
        "minimax",
        "xai",
        "mistral",
        "gemini",
        "neutts",
        "kittentts",
        "piper",
    ]
}

pub fn is_builtin_provider(name: &str) -> bool {
    builtin_tts_providers().contains(&name)
}

pub fn command_tts_output_formats() -> &'static [&'static str] {
    &["mp3", "wav", "ogg", "flac"]
}

/// Per-provider input-character limits (from official provider docs).
pub fn provider_max_text_length(provider: &str) -> Option<i64> {
    match provider {
        "edge" => Some(5000),
        "openai" => Some(4096),
        "xai" => Some(15000),
        "minimax" => Some(10000),
        "mistral" => Some(4000),
        "gemini" => Some(5000),
        "elevenlabs" => Some(10000),
        "neutts" => Some(2000),
        "kittentts" => Some(2000),
        "piper" => Some(5000),
        _ => None,
    }
}

/// ElevenLabs caps vary by model_id.
pub fn elevenlabs_model_max_text_length(model_id: &str) -> Option<i64> {
    match model_id {
        "eleven_v3" => Some(5000),
        "eleven_ttv_v3" => Some(5000),
        "eleven_multilingual_v2" => Some(10000),
        "eleven_multilingual_v1" => Some(10000),
        "eleven_english_sts_v2" => Some(10000),
        "eleven_english_sts_v1" => Some(10000),
        "eleven_flash_v2" => Some(30000),
        "eleven_flash_v2_5" => Some(40000),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Small JSON access helpers mirroring Python's dict.get semantics.
// ---------------------------------------------------------------------------

/// `value.get(key)` returning the inner object when value is a mapping.
fn obj_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|m| m.get(key))
}

/// Return a string value, treating absent / null / non-string as `None`,
/// matching `str(cfg.get(...))` only when the value is a string.
fn opt_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    obj_get(value, key).and_then(|v| v.as_str())
}

/// Python `cfg.get(key)` returning the sub-dict if it's a dict, else `{}`.
fn dict_section<'a>(value: &'a Value, key: &str) -> Value {
    match obj_get(value, key) {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Default::default()),
    }
}

/// Replicate `str(config.get(k, default))` where the value may be any JSON.
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Read an integer override the same way Python does: bool is *not* treated as
/// an int, only genuine integers count. Returns `Some(v)` only for `v > 0`.
fn positive_int_override(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Bool(_)) => None,
        Some(Value::Number(n)) => {
            // Python's isinstance(override, int) excludes floats.
            if n.is_i64() || n.is_u64() {
                let iv = n.as_i64()?;
                if iv > 0 {
                    Some(iv)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

// ===========================================================================
// Config loader — reads `tts:` section from `~/.hermes/config.yaml`.
// ===========================================================================

/// Load the `tts:` section from `~/.hermes/config.yaml`.
///
/// Mirrors `_load_tts_config`: on any failure returns an empty object.
pub fn load_tts_config() -> Value {
    let path = match config_yaml_path() {
        Some(p) => p,
        None => return Value::Object(Default::default()),
    };
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Value::Object(Default::default()),
    };
    let parsed: serde_yaml::Value = match serde_yaml::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Value::Object(Default::default()),
    };
    let as_json: Value = match serde_json::to_value(parsed) {
        Ok(v) => v,
        Err(_) => return Value::Object(Default::default()),
    };
    match obj_get(&as_json, "tts") {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Default::default()),
    }
}

fn config_yaml_path() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HERMES_HOME") {
        if !home.trim().is_empty() {
            return Some(PathBuf::from(home).join("config.yaml"));
        }
    }
    dirs::home_dir().map(|h| h.join(".hermes").join("config.yaml"))
}

/// Get the configured TTS provider name. Mirrors `_get_provider`.
pub fn get_provider(tts_config: &Value) -> String {
    let raw = opt_str(tts_config, "provider")
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_PROVIDER);
    raw.to_lowercase().trim().to_string()
}

// ===========================================================================
// max_text_length resolution
// ===========================================================================

/// Return the input-character cap for *provider*. Mirrors `_resolve_max_text_length`.
pub fn resolve_max_text_length(provider: Option<&str>, tts_config: &Value) -> i64 {
    let provider = match provider {
        Some(p) if !p.is_empty() => p,
        _ => return FALLBACK_MAX_TEXT_LENGTH,
    };
    let key = provider.to_lowercase().trim().to_string();

    let prov_cfg = dict_section(tts_config, &key);
    if let Some(v) = positive_int_override(obj_get(&prov_cfg, "max_text_length")) {
        return v;
    }

    if key == "elevenlabs" {
        let model_id = opt_str(&prov_cfg, "model_id")
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_ELEVENLABS_MODEL_ID);
        if let Some(mapped) = elevenlabs_model_max_text_length(model_id.trim()) {
            return mapped;
        }
    }

    if let Some(v) = provider_max_text_length(&key) {
        return v;
    }

    if !is_builtin_provider(&key) {
        let named = get_named_provider_config(tts_config, &key);
        if is_command_provider_config(&named) {
            if let Some(v) = positive_int_override(obj_get(&named, "max_text_length")) {
                return v;
            }
            return DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH;
        }
    }

    FALLBACK_MAX_TEXT_LENGTH
}

// ===========================================================================
// Custom command providers
// ===========================================================================

/// Return a provider config block if it's a dict, else an empty dict.
fn get_provider_section(tts_config: &Value, name: &str) -> Value {
    if !tts_config.is_object() {
        return Value::Object(Default::default());
    }
    match obj_get(tts_config, name) {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Default::default()),
    }
}

/// Return the config dict for a user-declared provider. Mirrors
/// `_get_named_provider_config`.
pub fn get_named_provider_config(tts_config: &Value, name: &str) -> Value {
    let providers = get_provider_section(tts_config, "providers");
    if let Some(section) = obj_get(&providers, name) {
        if section.is_object() {
            return section.clone();
        }
    }
    if !is_builtin_provider(&name.to_lowercase()) {
        let legacy = get_provider_section(tts_config, name);
        if legacy.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            return legacy;
        }
    }
    Value::Object(Default::default())
}

/// Return True when *config* declares a command-type provider.
pub fn is_command_provider_config(config: &Value) -> bool {
    if !config.is_object() {
        return false;
    }
    let ptype = opt_str(config, "type").unwrap_or("").trim().to_lowercase();
    if !ptype.is_empty() && ptype != "command" {
        return false;
    }
    match opt_str(config, "command") {
        Some(c) => !c.trim().is_empty(),
        None => false,
    }
}

/// Return the provider config if *provider* resolves to a command type.
/// Mirrors `_resolve_command_provider_config`.
pub fn resolve_command_provider_config(provider: &str, tts_config: &Value) -> Option<Value> {
    if provider.is_empty() {
        return None;
    }
    let key = provider.to_lowercase().trim().to_string();
    if is_builtin_provider(&key) {
        return None;
    }
    let config = get_named_provider_config(tts_config, &key);
    if is_command_provider_config(&config) {
        Some(config)
    } else {
        None
    }
}

/// Yield (name, config) for every declared command-type provider.
pub fn iter_command_providers(tts_config: &Value) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    if !tts_config.is_object() {
        return out;
    }
    let providers = get_provider_section(tts_config, "providers");
    if let Some(map) = providers.as_object() {
        for (name, cfg) in map {
            if !is_builtin_provider(&name.to_lowercase()) && is_command_provider_config(cfg) {
                out.push((name.clone(), cfg.clone()));
            }
        }
    }
    out
}

/// Return timeout in seconds, falling back when invalid. Mirrors
/// `_get_command_tts_timeout`.
pub fn get_command_tts_timeout(config: &Value) -> f64 {
    let raw = obj_get(config, "timeout")
        .or_else(|| obj_get(config, "timeout_seconds"))
        .cloned();
    let value: Option<f64> = match raw {
        None | Some(Value::Null) => Some(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS),
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        Some(Value::Bool(b)) => Some(if b { 1.0 } else { 0.0 }), // float(True)=1.0
        _ => None,
    };
    match value {
        Some(v) if v > 0.0 => v,
        _ => DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS,
    }
}

/// Return the validated output format (mp3/wav/ogg/flac). Mirrors
/// `_get_command_tts_output_format`.
pub fn get_command_tts_output_format(config: &Value, output_path: Option<&str>) -> String {
    if let Some(p) = output_path {
        if let Some(suffix) = Path::new(p)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
        {
            let suffix = suffix.trim().trim_start_matches('.').to_string();
            if command_tts_output_formats().contains(&suffix.as_str()) {
                return suffix;
            }
        }
    }
    let raw = opt_str(config, "format")
        .filter(|s| !s.is_empty())
        .or_else(|| opt_str(config, "output_format").filter(|s| !s.is_empty()))
        .unwrap_or(DEFAULT_COMMAND_TTS_OUTPUT_FORMAT);
    let fmt = raw.to_lowercase().trim().trim_start_matches('.').to_string();
    if command_tts_output_formats().contains(&fmt.as_str()) {
        fmt
    } else {
        DEFAULT_COMMAND_TTS_OUTPUT_FORMAT.to_string()
    }
}

/// Return True only when the user explicitly opted in to voice delivery.
/// Mirrors `_is_command_tts_voice_compatible`.
pub fn is_command_tts_voice_compatible(config: &Value) -> bool {
    match obj_get(config, "voice_compatible") {
        Some(Value::String(s)) => {
            matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
        }
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::Null) | None => false,
        Some(other) => match other {
            // Python bool() of a non-empty container / value -> True.
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
            _ => false,
        },
    }
}

// ---------------------------------------------------------------------------
// Shell-quote-aware template rendering.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum QuoteCtx {
    None,
    Single,
    Double,
}

/// Return the shell quote character active right before *position*.
/// Mirrors `_shell_quote_context`.
fn shell_quote_context(template: &str, position: usize) -> QuoteCtx {
    let chars: Vec<char> = template.chars().collect();
    let mut quote = QuoteCtx::None;
    let mut escaped = false;
    let mut i = 0;
    while i < position && i < chars.len() {
        let ch = chars[i];
        match quote {
            QuoteCtx::Single => {
                if ch == '\'' {
                    quote = QuoteCtx::None;
                }
            }
            QuoteCtx::Double => {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    quote = QuoteCtx::None;
                }
            }
            QuoteCtx::None => {
                if ch == '\'' {
                    quote = QuoteCtx::Single;
                } else if ch == '"' {
                    quote = QuoteCtx::Double;
                } else if ch == '\\' {
                    i += 1;
                }
            }
        }
        i += 1;
    }
    quote
}

/// Quote a placeholder value for its position in a shell command template.
/// Mirrors `_quote_command_tts_placeholder` (POSIX path; Windows list2cmdline
/// is not modeled — non-Windows behavior only, matching the test platform).
fn quote_command_tts_placeholder(value: &str, quote_ctx: QuoteCtx) -> String {
    match quote_ctx {
        QuoteCtx::Single => value.replace('\'', "'\\''"),
        QuoteCtx::Double => value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('$', "\\$")
            .replace('`', "\\`"),
        QuoteCtx::None => shlex_quote(value),
    }
}

/// POSIX shell quoting equivalent to Python's `shlex.quote`.
fn shlex_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    let safe = value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '%' | '_' | '-' | '+' | '=' | ':' | ',' | '.' | '/'));
    if safe {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

/// Replace supported placeholders while preserving `{{` / `}}`.
/// Mirrors `_render_command_tts_template`.
pub fn render_command_tts_template(
    command_template: &str,
    placeholders: &BTreeMap<String, String>,
) -> String {
    // Build a regex `(?<!\$)(?:\{\{(name)\}\}|\{(name)\})`. Rust regex lacks
    // look-behind, so we emulate `(?<!\$)` by checking the preceding char.
    let names: Vec<&str> = placeholders.keys().map(|s| s.as_str()).collect();
    let names_alt = names
        .iter()
        .map(|n| regex::escape(n))
        .collect::<Vec<_>>()
        .join("|");
    let pattern = format!(
        r"(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})",
        names = names_alt
    );
    let re = match regex::Regex::new(&pattern) {
        Ok(r) => r,
        Err(_) => return command_template.to_string(),
    };

    let mut replacements: Vec<(String, String)> = Vec::new();
    let mut rendered = String::new();
    let mut last = 0usize;
    for caps in re.captures_iter(command_template) {
        let m = caps.get(0).unwrap();
        // Emulate the `(?<!\$)` negative look-behind.
        if m.start() > 0 && command_template.as_bytes()[m.start() - 1] == b'$' {
            // Skip this match: copy through it verbatim.
            continue;
        }
        rendered.push_str(&command_template[last..m.start()]);
        let name = caps
            .name("double")
            .or_else(|| caps.name("single"))
            .map(|x| x.as_str())
            .unwrap_or("");
        let token = format!("__HERMES_TTS_PLACEHOLDER_{}__", replacements.len());
        let val = placeholders.get(name).cloned().unwrap_or_default();
        let quoted = quote_command_tts_placeholder(
            &val,
            shell_quote_context(command_template, m.start()),
        );
        replacements.push((token.clone(), quoted));
        rendered.push_str(&token);
        last = m.end();
    }
    rendered.push_str(&command_template[last..]);

    rendered = rendered.replace("{{", "{").replace("}}", "}");
    for (token, value) in &replacements {
        rendered = rendered.replace(token, value);
    }
    rendered
}

/// Result of running a command-provider shell command.
pub struct CommandRun {
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// Run a command-provider shell command with a timeout. Mirrors `_run_command_tts`
/// + `_terminate_command_tts_process_tree`. POSIX-oriented (uses `sh -c`).
pub fn run_command_tts(command: &str, timeout: f64) -> std::io::Result<CommandRun> {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let deadline = std::time::Instant::now() + Duration::from_secs_f64(timeout.max(0.0));
    loop {
        match child.try_wait()? {
            Some(_status) => break,
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let out = child.wait_with_output().ok();
                    let (so, se) = out
                        .map(|o| {
                            (
                                String::from_utf8_lossy(&o.stdout).into_owned(),
                                String::from_utf8_lossy(&o.stderr).into_owned(),
                            )
                        })
                        .unwrap_or_default();
                    return Ok(CommandRun {
                        returncode: -1,
                        stdout: so,
                        stderr: se,
                        timed_out: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
    let output = child.wait_with_output()?;
    Ok(CommandRun {
        returncode: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        timed_out: false,
    })
}

/// Return an output path whose extension matches the provider's output_format.
/// Mirrors `_configured_command_tts_output_path`.
pub fn configured_command_tts_output_path(path: &Path, config: &Value) -> PathBuf {
    let fmt = get_command_tts_output_format(config, None);
    path.with_extension(fmt)
}

/// Generate speech by running a user-configured shell command. Mirrors
/// `_generate_command_tts`. Returns the absolute path written, or an error.
pub fn generate_command_tts(
    text: &str,
    output_path: &str,
    provider_name: &str,
    config: &Value,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let command_template = opt_str(config, "command").unwrap_or("").trim().to_string();
    if command_template.is_empty() {
        return Err(TtsError::Value(format!(
            "tts.providers.{}.command is not configured",
            provider_name
        )));
    }

    let output = expand_user(output_path);
    if let Some(parent) = output.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if output.exists() {
        let _ = fs::remove_file(&output);
    }

    let timeout = get_command_tts_timeout(config);
    let output_format = get_command_tts_output_format(config, Some(&output.to_string_lossy()));
    // speed: config.get("speed", tts_config.get("speed", ""))
    let speed = obj_get(config, "speed")
        .cloned()
        .or_else(|| obj_get(tts_config, "speed").cloned())
        .map(|v| value_to_str(&v))
        .unwrap_or_default();

    let tmpdir = std::env::temp_dir().join(format!("hermes_tts_{}", std::process::id()));
    fs::create_dir_all(&tmpdir).map_err(|e| TtsError::Runtime(e.to_string()))?;
    let text_path = tmpdir.join("input.txt");
    fs::write(&text_path, text.as_bytes()).map_err(|e| TtsError::Runtime(e.to_string()))?;

    let mut placeholders: BTreeMap<String, String> = BTreeMap::new();
    placeholders.insert("input_path".into(), text_path.to_string_lossy().into_owned());
    placeholders.insert("text_path".into(), text_path.to_string_lossy().into_owned());
    placeholders.insert("output_path".into(), output.to_string_lossy().into_owned());
    placeholders.insert("format".into(), output_format.clone());
    placeholders.insert("voice".into(), opt_str(config, "voice").unwrap_or("").to_string());
    placeholders.insert("model".into(), opt_str(config, "model").unwrap_or("").to_string());
    placeholders.insert("speed".into(), speed);

    let command = render_command_tts_template(&command_template, &placeholders);

    let run = run_command_tts(&command, timeout);
    let _ = fs::remove_dir_all(&tmpdir);

    match run {
        Ok(r) if r.timed_out => {
            return Err(TtsError::Runtime(format!(
                "TTS provider '{}' timed out after {}s",
                provider_name,
                format_g(timeout)
            )));
        }
        Ok(r) if r.returncode != 0 => {
            let mut parts = Vec::new();
            if !r.stderr.trim().is_empty() {
                parts.push(format!("stderr: {}", r.stderr.trim()));
            }
            if !r.stdout.trim().is_empty() {
                parts.push(format!("stdout: {}", r.stdout.trim()));
            }
            let detail = if parts.is_empty() {
                "no command output".to_string()
            } else {
                parts.join("; ")
            };
            return Err(TtsError::Runtime(format!(
                "TTS provider '{}' exited with code {}: {}",
                provider_name, r.returncode, detail
            )));
        }
        Ok(_) => {}
        Err(e) => {
            return Err(TtsError::Runtime(format!(
                "TTS provider '{}' failed to start: {}",
                provider_name, e
            )));
        }
    }

    let size = output.metadata().map(|m| m.len()).unwrap_or(0);
    if !output.exists() || size == 0 {
        return Err(TtsError::Runtime(format!(
            "TTS provider '{}' produced no output at {}",
            provider_name,
            output.display()
        )));
    }
    Ok(output.to_string_lossy().into_owned())
}

/// Return True when any command-type TTS provider is configured.
pub fn has_any_command_tts_provider(tts_config: Option<&Value>) -> bool {
    match tts_config {
        Some(cfg) => !iter_command_providers(cfg).is_empty(),
        None => !iter_command_providers(&load_tts_config()).is_empty(),
    }
}

/// Format a float like Python's `%g`.
fn format_g(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e16 {
        format!("{}", value as i64)
    } else {
        let s = format!("{}", value);
        s
    }
}

fn expand_user(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~") {
        if let Some(home) = dirs::home_dir() {
            let rest = stripped.trim_start_matches('/');
            if rest.is_empty() {
                return home;
            }
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

// ===========================================================================
// ffmpeg helpers
// ===========================================================================

/// Check if ffmpeg is available on the system. Mirrors `_has_ffmpeg`.
pub fn has_ffmpeg() -> bool {
    which("ffmpeg").is_some()
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Convert an MP3 file to OGG Opus for Telegram voice bubbles. Mirrors
/// `_convert_to_opus`. Returns the .ogg path on success.
pub fn convert_to_opus(mp3_path: &str) -> Option<String> {
    if !has_ffmpeg() {
        return None;
    }
    let stem = match mp3_path.rsplit_once('.') {
        Some((s, _)) => s.to_string(),
        None => mp3_path.to_string(),
    };
    let ogg_path = format!("{}.ogg", stem);
    let result = Command::new("ffmpeg")
        .args([
            "-i", mp3_path, "-acodec", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off",
            &ogg_path, "-y",
        ])
        .output();
    match result {
        Ok(out) => {
            if !out.status.success() {
                return None;
            }
            let ok = fs::metadata(&ogg_path).map(|m| m.len() > 0).unwrap_or(false);
            if ok {
                Some(ogg_path)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

// ===========================================================================
// WAV wrapping for Gemini PCM
// ===========================================================================

/// Wrap raw signed-little-endian PCM with a standard WAV RIFF header.
/// Mirrors `_wrap_pcm_as_wav`.
pub fn wrap_pcm_as_wav(
    pcm_bytes: &[u8],
    sample_rate: u32,
    channels: u16,
    sample_width: u16,
) -> Vec<u8> {
    let byte_rate = sample_rate * (channels as u32) * (sample_width as u32);
    let block_align = channels * sample_width;
    let data_size = pcm_bytes.len() as u32;

    let mut fmt_chunk = Vec::new();
    fmt_chunk.extend_from_slice(b"fmt ");
    fmt_chunk.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size (PCM)
    fmt_chunk.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    fmt_chunk.extend_from_slice(&channels.to_le_bytes());
    fmt_chunk.extend_from_slice(&sample_rate.to_le_bytes());
    fmt_chunk.extend_from_slice(&byte_rate.to_le_bytes());
    fmt_chunk.extend_from_slice(&block_align.to_le_bytes());
    fmt_chunk.extend_from_slice(&(sample_width * 8).to_le_bytes());

    let mut data_header = Vec::new();
    data_header.extend_from_slice(b"data");
    data_header.extend_from_slice(&data_size.to_le_bytes());

    let riff_size = 4 + fmt_chunk.len() as u32 + data_header.len() as u32 + data_size;

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(&fmt_chunk);
    out.extend_from_slice(&data_header);
    out.extend_from_slice(pcm_bytes);
    out
}

// ===========================================================================
// Errors
// ===========================================================================

#[derive(Debug)]
pub enum TtsError {
    /// Configuration errors (missing API keys, invalid command config).
    Value(String),
    /// Missing dependency / file.
    FileNotFound(String),
    /// Everything else (HTTP failures, ffmpeg failures, empty output).
    Runtime(String),
}

impl std::fmt::Display for TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsError::Value(m) | TtsError::FileNotFound(m) | TtsError::Runtime(m) => {
                write!(f, "{}", m)
            }
        }
    }
}

impl std::error::Error for TtsError {}

// ===========================================================================
// Network providers (request construction + response parsing)
// ===========================================================================

fn env_value(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

fn blocking_client(timeout_secs: u64) -> Result<reqwest::blocking::Client, TtsError> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| TtsError::Runtime(format!("HTTP client init failed: {}", e)))
}

/// xAI TTS via dedicated `/v1/tts`. Mirrors `_generate_xai_tts`.
pub fn generate_xai_tts(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let api_key = env_value("XAI_API_KEY");
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(TtsError::Value(
            "XAI_API_KEY not set. Get one at https://console.x.ai/".into(),
        ));
    }

    let xai_config = dict_section(tts_config, "xai");
    let voice_id = {
        let v = opt_str(&xai_config, "voice_id")
            .map(value_to_str_or)
            .unwrap_or_default();
        let v = v.trim().to_string();
        if v.is_empty() {
            DEFAULT_XAI_VOICE_ID.to_string()
        } else {
            v
        }
    };
    let language = {
        let v = opt_str(&xai_config, "language")
            .map(value_to_str_or)
            .unwrap_or_default();
        let v = v.trim().to_string();
        if v.is_empty() {
            DEFAULT_XAI_LANGUAGE.to_string()
        } else {
            v
        }
    };
    let sample_rate = int_field(&xai_config, "sample_rate", DEFAULT_XAI_SAMPLE_RATE);
    let bit_rate = int_field(&xai_config, "bit_rate", DEFAULT_XAI_BIT_RATE);
    let base_url = {
        let v = opt_str(&xai_config, "base_url")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| {
                let e = env_value("XAI_BASE_URL");
                if e.is_empty() {
                    None
                } else {
                    Some(e)
                }
            })
            .unwrap_or_else(|| DEFAULT_XAI_BASE_URL.to_string());
        v.trim().trim_end_matches('/').to_string()
    };

    let codec = if output_path.ends_with(".wav") { "wav" } else { "mp3" };
    let mut payload = json!({
        "text": text,
        "voice_id": voice_id,
        "language": language,
    });
    if codec != "mp3"
        || sample_rate != DEFAULT_XAI_SAMPLE_RATE
        || (codec == "mp3" && bit_rate != DEFAULT_XAI_BIT_RATE)
    {
        let mut output_format = serde_json::Map::new();
        output_format.insert("codec".into(), json!(codec));
        if sample_rate != 0 {
            output_format.insert("sample_rate".into(), json!(sample_rate));
        }
        if codec == "mp3" && bit_rate != 0 {
            output_format.insert("bit_rate".into(), json!(bit_rate));
        }
        payload["output_format"] = Value::Object(output_format);
    }

    let client = blocking_client(60)?;
    let resp = client
        .post(format!("{}/tts", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .header("User-Agent", hermes_xai_user_agent())
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("xAI TTS request failed: {}", e)))?;

    let resp = resp
        .error_for_status()
        .map_err(|e| TtsError::Runtime(format!("xAI TTS HTTP error: {}", e)))?;
    let bytes = resp
        .bytes()
        .map_err(|e| TtsError::Runtime(e.to_string()))?;
    write_bytes(output_path, &bytes)?;
    Ok(output_path.to_string())
}

/// MiniMax TTS. Mirrors `_generate_minimax_tts`.
pub fn generate_minimax_tts(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let api_key = env_value("MINIMAX_API_KEY");
    if api_key.is_empty() {
        return Err(TtsError::Value(
            "MINIMAX_API_KEY not set. Get one at https://platform.minimax.io/".into(),
        ));
    }
    let mm = dict_section(tts_config, "minimax");
    let model = opt_str(&mm, "model").unwrap_or(DEFAULT_MINIMAX_MODEL);
    let voice_id = opt_str(&mm, "voice_id").unwrap_or(DEFAULT_MINIMAX_VOICE_ID);
    let base_url = opt_str(&mm, "base_url").unwrap_or(DEFAULT_MINIMAX_BASE_URL);

    let payload = json!({ "model": model, "text": text, "voice_id": voice_id });

    let client = blocking_client(60)?;
    let resp = client
        .post(base_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("MiniMax TTS request failed: {}", e)))?;

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let status = resp.status();
    let bytes = resp
        .bytes()
        .map_err(|e| TtsError::Runtime(e.to_string()))?;

    if content_type.contains("audio/") {
        write_bytes(output_path, &bytes)?;
        return Ok(output_path.to_string());
    }

    // Legacy / fallback: parse JSON with hex-encoded audio.
    let result: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            if !status.is_success() {
                return Err(TtsError::Runtime(format!("MiniMax TTS HTTP {}", status)));
            }
            return Err(TtsError::Runtime(format!(
                "MiniMax TTS returned unexpected Content-Type '{}' ({} bytes)",
                content_type,
                bytes.len()
            )));
        }
    };

    let base_resp = dict_section(&result, "base_resp");
    let status_code = obj_get(&base_resp, "status_code")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    if status_code != 0 {
        let status_msg = opt_str(&base_resp, "status_msg").unwrap_or("unknown error");
        return Err(TtsError::Runtime(format!(
            "MiniMax TTS API error (code {}): {}",
            status_code, status_msg
        )));
    }
    let data = dict_section(&result, "data");
    let hex_audio = opt_str(&data, "audio").unwrap_or("");
    if hex_audio.is_empty() {
        return Err(TtsError::Runtime("MiniMax TTS returned empty audio data".into()));
    }
    let audio_bytes = hex_decode(hex_audio)
        .ok_or_else(|| TtsError::Runtime("MiniMax TTS returned invalid hex audio".into()))?;
    write_bytes(output_path, &audio_bytes)?;
    Ok(output_path.to_string())
}

/// Gemini TTS. Mirrors `_generate_gemini_tts` (request + parse + WAV wrap +
/// optional ffmpeg conversion).
pub fn generate_gemini_tts(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let api_key = {
        let g = env_value("GEMINI_API_KEY");
        let chosen = if !g.is_empty() {
            g
        } else {
            env_value("GOOGLE_API_KEY")
        };
        chosen.trim().to_string()
    };
    if api_key.is_empty() {
        return Err(TtsError::Value(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey".into(),
        ));
    }
    let gemini = dict_section(tts_config, "gemini");
    let model = {
        let v = opt_str(&gemini, "model").unwrap_or(DEFAULT_GEMINI_TTS_MODEL).trim().to_string();
        if v.is_empty() {
            DEFAULT_GEMINI_TTS_MODEL.to_string()
        } else {
            v
        }
    };
    let voice = {
        let v = opt_str(&gemini, "voice").unwrap_or(DEFAULT_GEMINI_TTS_VOICE).trim().to_string();
        if v.is_empty() {
            DEFAULT_GEMINI_TTS_VOICE.to_string()
        } else {
            v
        }
    };
    let base_url = {
        let v = opt_str(&gemini, "base_url")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| {
                let e = env_value("GEMINI_BASE_URL");
                if e.is_empty() {
                    None
                } else {
                    Some(e)
                }
            })
            .unwrap_or_else(|| DEFAULT_GEMINI_TTS_BASE_URL.to_string());
        v.trim().trim_end_matches('/').to_string()
    };

    let payload = json!({
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice},
                },
            },
        },
    });

    let endpoint = format!("{}/models/{}:generateContent", base_url, model);
    let client = blocking_client(60)?;
    let resp = client
        .post(&endpoint)
        .query(&[("key", api_key.as_str())])
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("Gemini TTS request failed: {}", e)))?;

    let status = resp.status();
    let text_body = resp.text().map_err(|e| TtsError::Runtime(e.to_string()))?;
    if status.as_u16() != 200 {
        let detail = serde_json::from_str::<Value>(&text_body)
            .ok()
            .and_then(|v| {
                dict_section(&v, "error")
                    .get("message")
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| truncate(&text_body, 300));
        return Err(TtsError::Runtime(format!(
            "Gemini TTS API error (HTTP {}): {}",
            status.as_u16(),
            detail
        )));
    }

    let data: Value = serde_json::from_str(&text_body)
        .map_err(|e| TtsError::Runtime(format!("Gemini TTS response was malformed: {}", e)))?;
    let parts = data
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .ok_or_else(|| TtsError::Runtime("Gemini TTS response was malformed".into()))?;
    let audio_part = parts
        .iter()
        .find(|p| p.get("inlineData").is_some() || p.get("inline_data").is_some());
    let audio_part = match audio_part {
        Some(p) => p,
        None => {
            return Err(TtsError::Runtime(
                "Gemini TTS response contained no audio data".into(),
            ))
        }
    };
    let inline = audio_part
        .get("inlineData")
        .or_else(|| audio_part.get("inline_data"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let audio_b64 = inline.get("data").and_then(|d| d.as_str()).unwrap_or("");
    if audio_b64.is_empty() {
        return Err(TtsError::Runtime("Gemini TTS returned empty audio data".into()));
    }

    use base64::Engine as _;
    let pcm_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64)
        .map_err(|e| TtsError::Runtime(format!("Gemini TTS base64 decode failed: {}", e)))?;
    let wav_bytes = wrap_pcm_as_wav(
        &pcm_bytes,
        GEMINI_TTS_SAMPLE_RATE,
        GEMINI_TTS_CHANNELS,
        GEMINI_TTS_SAMPLE_WIDTH,
    );

    if output_path.to_lowercase().ends_with(".wav") {
        write_bytes(output_path, &wav_bytes)?;
        return Ok(output_path.to_string());
    }

    // Write WAV temp file and ffmpeg-convert.
    let wav_path = std::env::temp_dir().join(format!("hermes_gemini_{}.wav", std::process::id()));
    write_bytes(&wav_path.to_string_lossy(), &wav_bytes)?;

    let result = (|| -> Result<(), TtsError> {
        if let Some(ffmpeg) = which("ffmpeg") {
            let ffmpeg = ffmpeg.to_string_lossy().into_owned();
            let cmd_out = if output_path.to_lowercase().ends_with(".ogg") {
                Command::new(&ffmpeg)
                    .args([
                        "-i",
                        &wav_path.to_string_lossy(),
                        "-acodec",
                        "libopus",
                        "-ac",
                        "1",
                        "-b:a",
                        "64k",
                        "-vbr",
                        "off",
                        "-y",
                        "-loglevel",
                        "error",
                        output_path,
                    ])
                    .output()
            } else {
                Command::new(&ffmpeg)
                    .args(["-i", &wav_path.to_string_lossy(), "-y", "-loglevel", "error", output_path])
                    .output()
            };
            match cmd_out {
                Ok(out) if !out.status.success() => {
                    let stderr = truncate(&String::from_utf8_lossy(&out.stderr), 300);
                    Err(TtsError::Runtime(format!("ffmpeg conversion failed: {}", stderr)))
                }
                Ok(_) => Ok(()),
                Err(e) => Err(TtsError::Runtime(format!("ffmpeg conversion failed: {}", e))),
            }
        } else {
            fs::copy(&wav_path, output_path)
                .map(|_| ())
                .map_err(|e| TtsError::Runtime(e.to_string()))
        }
    })();
    let _ = fs::remove_file(&wav_path);
    result?;
    Ok(output_path.to_string())
}

/// ElevenLabs TTS via HTTP. Mirrors the SDK call in `_generate_elevenlabs`.
/// (The Python uses the `elevenlabs` SDK; we hit the documented REST endpoint.)
pub fn generate_elevenlabs(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let api_key = env_value("ELEVENLABS_API_KEY");
    if api_key.is_empty() {
        return Err(TtsError::Value(
            "ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/".into(),
        ));
    }
    let el = dict_section(tts_config, "elevenlabs");
    let voice_id = opt_str(&el, "voice_id").unwrap_or(DEFAULT_ELEVENLABS_VOICE_ID);
    let model_id = opt_str(&el, "model_id").unwrap_or(DEFAULT_ELEVENLABS_MODEL_ID);
    let output_format = if output_path.ends_with(".ogg") {
        "opus_48000_64"
    } else {
        "mp3_44100_128"
    };

    let payload = json!({ "text": text, "model_id": model_id });
    let url = format!(
        "https://api.elevenlabs.io/v1/text-to-speech/{}?output_format={}",
        voice_id, output_format
    );
    let client = blocking_client(60)?;
    let resp = client
        .post(&url)
        .header("xi-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("ElevenLabs request failed: {}", e)))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| TtsError::Runtime(format!("ElevenLabs HTTP error: {}", e)))?;
    let bytes = resp.bytes().map_err(|e| TtsError::Runtime(e.to_string()))?;
    write_bytes(output_path, &bytes)?;
    Ok(output_path.to_string())
}

/// OpenAI TTS via HTTP. Mirrors `_generate_openai_tts` request construction.
pub fn generate_openai_tts(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let (api_key, mut base_url) = resolve_openai_audio_client_config(tts_config)?;
    let oai = dict_section(tts_config, "openai");
    let model = opt_str(&oai, "model").unwrap_or(DEFAULT_OPENAI_MODEL).to_string();
    let voice = opt_str(&oai, "voice").unwrap_or(DEFAULT_OPENAI_VOICE).to_string();
    if let Some(b) = opt_str(&oai, "base_url") {
        base_url = b.to_string();
    }
    let speed = float_field(&oai, "speed").or_else(|| float_field(tts_config, "speed")).unwrap_or(1.0);

    let response_format = if output_path.ends_with(".ogg") { "opus" } else { "mp3" };

    let mut payload = json!({
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    });
    if speed != 1.0 {
        let clamped = speed.clamp(0.25, 4.0);
        payload["speed"] = json!(clamped);
    }

    let client = blocking_client(120)?;
    let resp = client
        .post(format!("{}/audio/speech", base_url.trim_end_matches('/')))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .header("x-idempotency-key", new_uuid_v4())
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("OpenAI TTS request failed: {}", e)))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| TtsError::Runtime(format!("OpenAI TTS HTTP error: {}", e)))?;
    let bytes = resp.bytes().map_err(|e| TtsError::Runtime(e.to_string()))?;
    write_bytes(output_path, &bytes)?;
    Ok(output_path.to_string())
}

/// Mistral Voxtral TTS via HTTP. Mirrors `_generate_mistral_tts`: the API
/// returns base64-encoded audio under `audio` (the SDK's `audio_data`).
pub fn generate_mistral_tts(
    text: &str,
    output_path: &str,
    tts_config: &Value,
) -> Result<String, TtsError> {
    let api_key = env_value("MISTRAL_API_KEY");
    if api_key.is_empty() {
        return Err(TtsError::Value(
            "MISTRAL_API_KEY not set. Get one at https://console.mistral.ai/".into(),
        ));
    }
    let mi = dict_section(tts_config, "mistral");
    let model = opt_str(&mi, "model").unwrap_or(DEFAULT_MISTRAL_TTS_MODEL);
    let voice_id = opt_str(&mi, "voice_id")
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_MISTRAL_TTS_VOICE_ID);
    let response_format = if output_path.ends_with(".ogg") {
        "opus"
    } else if output_path.ends_with(".wav") {
        "wav"
    } else if output_path.ends_with(".flac") {
        "flac"
    } else {
        "mp3"
    };

    let payload = json!({
        "model": model,
        "input": text,
        "voice_id": voice_id,
        "response_format": response_format,
    });

    let client = blocking_client(60)?;
    let resp = client
        .post("https://api.mistral.ai/v1/audio/speech")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| TtsError::Runtime(format!("Mistral TTS failed: {}", e)))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| TtsError::Runtime(format!("Mistral TTS failed: {}", e)))?;
    let body: Value = resp
        .json()
        .map_err(|e| TtsError::Runtime(format!("Mistral TTS failed: {}", e)))?;
    let audio_b64 = body
        .get("audio_data")
        .or_else(|| body.get("audio"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| TtsError::Runtime("Mistral TTS returned no audio".into()))?;
    use base64::Engine as _;
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64)
        .map_err(|e| TtsError::Runtime(format!("Mistral TTS base64 decode failed: {}", e)))?;
    write_bytes(output_path, &audio_bytes)?;
    Ok(output_path.to_string())
}

/// Return direct OpenAI audio config or a managed gateway fallback. Mirrors
/// `_resolve_openai_audio_client_config`. The managed-gateway branch is
/// represented by a runtime error here (the gateway resolution needs hooks not
/// available in this module); direct credentials are the common case.
pub fn resolve_openai_audio_client_config(tts_config: &Value) -> Result<(String, String), TtsError> {
    let direct = resolve_openai_audio_api_key();
    let tts_section = obj_get(tts_config, "use_gateway").map(|_| tts_config.clone());
    let prefers = prefers_gateway(tts_section.as_ref());
    if !direct.is_empty() && !prefers {
        return Ok((direct, DEFAULT_OPENAI_BASE_URL.to_string()));
    }
    Err(TtsError::Value(
        "Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set".into(),
    ))
}

fn has_openai_audio_backend() -> bool {
    !resolve_openai_audio_api_key().is_empty()
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn value_to_str_or(s: &str) -> String {
    s.to_string()
}

fn int_field(obj: &Value, key: &str, default: i64) -> i64 {
    match obj_get(obj, key) {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)).unwrap_or(default),
        _ => default,
    }
}

fn float_field(obj: &Value, key: &str) -> Option<f64> {
    match obj_get(obj, key) {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn write_bytes(path: &str, bytes: &[u8]) -> Result<(), TtsError> {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::File::create(path).map_err(|e| TtsError::Runtime(e.to_string()))?;
    f.write_all(bytes).map_err(|e| TtsError::Runtime(e.to_string()))?;
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

/// Generate a random UUIDv4 string without external deps.
fn new_uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mut seed = nanos ^ (pid << 64) ^ 0x9E37_79B9_7F4A_7C15_9E37_79B9_7F4A_7C15;
    let mut bytes = [0u8; 16];
    for b in bytes.iter_mut() {
        // xorshift-style mixing
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *b = (seed & 0xff) as u8;
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

// ===========================================================================
// Markdown stripping + streaming sentence buffer
// ===========================================================================

/// Remove markdown formatting that shouldn't be spoken aloud. Mirrors
/// `_strip_markdown_for_tts`.
pub fn strip_markdown_for_tts(text: &str) -> String {
    use std::sync::OnceLock;
    struct Pats {
        code_block: regex::Regex,
        link: regex::Regex,
        url: regex::Regex,
        bold: regex::Regex,
        italic: regex::Regex,
        inline_code: regex::Regex,
        header: regex::Regex,
        list_item: regex::Regex,
        hr: regex::Regex,
        excess_nl: regex::Regex,
    }
    static PATS: OnceLock<Pats> = OnceLock::new();
    let p = PATS.get_or_init(|| Pats {
        code_block: regex::Regex::new(r"(?s)```.*?```").unwrap(),
        link: regex::Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap(),
        url: regex::Regex::new(r"https?://\S+").unwrap(),
        bold: regex::Regex::new(r"\*\*(.+?)\*\*").unwrap(),
        italic: regex::Regex::new(r"\*(.+?)\*").unwrap(),
        inline_code: regex::Regex::new(r"`(.+?)`").unwrap(),
        header: regex::Regex::new(r"(?m)^#+\s*").unwrap(),
        list_item: regex::Regex::new(r"(?m)^\s*[-*]\s+").unwrap(),
        hr: regex::Regex::new(r"---+").unwrap(),
        excess_nl: regex::Regex::new(r"\n{3,}").unwrap(),
    });
    let t = p.code_block.replace_all(text, " ").into_owned();
    let t = p.link.replace_all(&t, "$1").into_owned();
    let t = p.url.replace_all(&t, "").into_owned();
    let t = p.bold.replace_all(&t, "$1").into_owned();
    let t = p.italic.replace_all(&t, "$1").into_owned();
    let t = p.inline_code.replace_all(&t, "$1").into_owned();
    let t = p.header.replace_all(&t, "").into_owned();
    let t = p.list_item.replace_all(&t, "").into_owned();
    let t = p.hr.replace_all(&t, "").into_owned();
    let t = p.excess_nl.replace_all(&t, "\n\n").into_owned();
    t.trim().to_string()
}

/// Find the first sentence boundary in `buf`, returning the byte end position
/// of the match (Python `_SENTENCE_BOUNDARY_RE.search(...).end()`).
///
/// Pattern: `(?<=[.!?])(?:\s|\n)|(?:\n\n)`.
pub fn find_sentence_boundary(buf: &str) -> Option<usize> {
    let bytes = buf.as_bytes();
    let mut i = 0;
    // The two alternatives, scanned left-to-right; regex picks the leftmost
    // match, preferring the first alternative at the same position.
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Alternative 1: char preceded by [.!?] that is whitespace.
        if (c.is_whitespace() && (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\x0b' || c == '\x0c'))
            && i > 0
        {
            let prev = bytes[i - 1] as char;
            if prev == '.' || prev == '!' || prev == '?' {
                return Some(i + 1);
            }
        }
        // Alternative 2: literal `\n\n`.
        if c == '\n' && i + 1 < bytes.len() && bytes[i + 1] as char == '\n' {
            return Some(i + 2);
        }
        i += 1;
    }
    None
}

/// State for the streaming sentence buffer, mirroring the loop in
/// `stream_tts_to_speaker`. This exposes the buffering / sentence-extraction
/// logic so it can be driven and tested without audio devices.
pub struct StreamBuffer {
    pub buf: String,
    pub spoken: Vec<String>,
    pub min_sentence_len: usize,
    think_re: regex::Regex,
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamBuffer {
    pub fn new() -> Self {
        StreamBuffer {
            buf: String::new(),
            spoken: Vec::new(),
            min_sentence_len: 20,
            think_re: regex::Regex::new(r"(?s)<think[\s>].*?</think>").unwrap(),
        }
    }

    fn strip_think(&self, s: &str) -> String {
        self.think_re.replace_all(s, "").into_owned()
    }

    /// Push a text delta; returns any sentences that should be spoken now.
    pub fn push_delta(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        self.buf = self.strip_think(&self.buf);
        // Wait for closing tag if an incomplete <think is at the end.
        if self.buf.contains("<think") && !self.buf.contains("</think>") {
            return Vec::new();
        }
        let mut out = Vec::new();
        loop {
            let end = match find_sentence_boundary(&self.buf) {
                Some(e) => e,
                None => break,
            };
            let sentence = self.buf[..end].to_string();
            let rest = self.buf[end..].to_string();
            if sentence.trim().chars().count() < self.min_sentence_len {
                // Merge short fragment into the next sentence.
                self.buf = format!("{}{}", sentence, rest);
                break;
            }
            self.buf = rest;
            if let Some(s) = self.consider_sentence(&sentence) {
                out.push(s);
            }
        }
        out
    }

    /// Flush remaining buffer on end-of-text sentinel.
    pub fn flush(&mut self) -> Option<String> {
        self.buf = self.strip_think(&self.buf);
        if self.buf.trim().is_empty() {
            return None;
        }
        let sentence = std::mem::take(&mut self.buf);
        self.consider_sentence(&sentence)
    }

    /// Apply markdown stripping + duplicate suppression to a sentence,
    /// returning the *raw* sentence when it should be spoken (Python passes the
    /// raw sentence to display, the cleaned text to TTS — we surface the raw).
    fn consider_sentence(&mut self, sentence: &str) -> Option<String> {
        let cleaned = strip_markdown_for_tts(sentence);
        let cleaned = cleaned.trim();
        if cleaned.is_empty() {
            return None;
        }
        let cleaned_lower = cleaned.to_lowercase().trim_end_matches(['.', '!', ',']).to_string();
        for prev in &self.spoken {
            if prev.to_lowercase().trim_end_matches(['.', '!', ',']) == cleaned_lower {
                return None;
            }
        }
        self.spoken.push(cleaned.to_string());
        Some(sentence.to_string())
    }
}

// ===========================================================================
// Top-level tool
// ===========================================================================

/// Convert text to speech audio. Mirrors `text_to_speech_tool`.
///
/// Returns a JSON string with `success`, `file_path`, `media_tag`, `provider`,
/// and `voice_compatible` (success) or `success: false` + `error`.
///
/// `session_platform` is the value of `HERMES_SESSION_PLATFORM` from the
/// gateway session env (passed in because the session-context lookup lives in a
/// separate module). Pass `""` for CLI / no platform.
pub fn text_to_speech_tool(
    text: &str,
    output_path: Option<&str>,
    session_platform: &str,
) -> String {
    if text.trim().is_empty() {
        return tool_error_json("Text is required");
    }

    let tts_config = load_tts_config();
    let mut provider = get_provider(&tts_config);

    let command_provider_config = resolve_command_provider_config(&provider, &tts_config);

    // Truncate very long text.
    let max_len = resolve_max_text_length(Some(&provider), &tts_config) as usize;
    let mut text_owned = text.to_string();
    if text_owned.chars().count() > max_len {
        text_owned = text_owned.chars().take(max_len).collect();
    }
    let text = text_owned.as_str();

    let platform = session_platform.to_lowercase();
    let want_opus = platform == "telegram";

    // Determine output path.
    let default_output_dir = default_output_dir();
    let mut file_path: PathBuf = if let Some(op) = output_path {
        let mut fp = expand_user(op);
        if let Some(cfg) = &command_provider_config {
            fp = configured_command_tts_output_path(&fp, cfg);
        }
        fp
    } else {
        let timestamp = local_timestamp();
        let _ = fs::create_dir_all(&default_output_dir);
        if let Some(cfg) = &command_provider_config {
            let fmt = get_command_tts_output_format(cfg, None);
            default_output_dir.join(format!("tts_{}.{}", timestamp, fmt))
        } else if want_opus
            && matches!(provider.as_str(), "openai" | "elevenlabs" | "mistral" | "gemini")
        {
            default_output_dir.join(format!("tts_{}.ogg", timestamp))
        } else {
            default_output_dir.join(format!("tts_{}.mp3", timestamp))
        }
    };

    if let Some(parent) = file_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut file_str = file_path.to_string_lossy().into_owned();

    // Generate.
    let gen_result: Result<(), TtsError> = (|| {
        if let Some(cfg) = &command_provider_config {
            file_str = generate_command_tts(text, &file_str, &provider, cfg, &tts_config)?;
            return Ok(());
        }
        match provider.as_str() {
            "elevenlabs" => generate_elevenlabs(text, &file_str, &tts_config).map(|_| ()),
            "openai" => generate_openai_tts(text, &file_str, &tts_config).map(|_| ()),
            "minimax" => generate_minimax_tts(text, &file_str, &tts_config).map(|_| ()),
            "xai" => generate_xai_tts(text, &file_str, &tts_config).map(|_| ()),
            "mistral" => generate_mistral_tts(text, &file_str, &tts_config).map(|_| ()),
            "gemini" => generate_gemini_tts(text, &file_str, &tts_config).map(|_| ()),
            "neutts" => Err(TtsError::Value(
                "NeuTTS provider selected but neutts is not installed (no native engine).".into(),
            )),
            "kittentts" => Err(TtsError::Value(
                "KittenTTS provider selected but 'kittentts' package not installed.".into(),
            )),
            "piper" => Err(TtsError::Value(
                "Piper provider selected but 'piper-tts' package not installed.".into(),
            )),
            _ => Err(TtsError::Value(
                "Edge TTS provider has no native Rust engine; configure a network provider or command provider.".into(),
            )),
        }
    })();

    if let Err(e) = gen_result {
        let (kind, msg) = match &e {
            TtsError::Value(m) => ("configuration error", m.clone()),
            TtsError::FileNotFound(m) => ("dependency missing", m.clone()),
            TtsError::Runtime(m) => ("generation failed", m.clone()),
        };
        return tool_error_json(&format!("TTS {} ({}): {}", kind, provider, msg));
    }

    // Verify file exists & is non-empty.
    let exists_nonempty = fs::metadata(&file_str).map(|m| m.len() > 0).unwrap_or(false);
    if !exists_nonempty {
        return json!({
            "success": false,
            "error": format!("TTS generation produced no output (provider: {})", provider),
        })
        .to_string();
    }

    // Opus conversion for Telegram compatibility.
    let mut voice_compatible = false;
    if let Some(cfg) = &command_provider_config {
        if is_command_tts_voice_compatible(cfg) {
            if !file_str.ends_with(".ogg") {
                if let Some(opus_path) = convert_to_opus(&file_str) {
                    file_str = opus_path;
                }
            }
            voice_compatible = file_str.ends_with(".ogg");
        }
    } else if matches!(
        provider.as_str(),
        "edge" | "neutts" | "minimax" | "xai" | "kittentts" | "piper"
    ) && !file_str.ends_with(".ogg")
    {
        if let Some(opus_path) = convert_to_opus(&file_str) {
            file_str = opus_path;
            voice_compatible = true;
        }
    } else if matches!(provider.as_str(), "elevenlabs" | "openai" | "mistral" | "gemini") {
        voice_compatible = file_str.ends_with(".ogg");
    }

    let mut media_tag = format!("MEDIA:{}", file_str);
    if voice_compatible {
        media_tag = format!("[[audio_as_voice]]\n{}", media_tag);
    }

    // keep `provider` borrow happy for editors; value already final
    let _ = &mut provider;
    let _ = &mut file_path;

    json!({
        "success": true,
        "file_path": file_str,
        "media_tag": media_tag,
        "provider": provider,
        "voice_compatible": voice_compatible,
    })
    .to_string()
}

/// JSON `{"success": false, "error": ...}` matching `tool_error(..., success=False)`.
fn tool_error_json(message: &str) -> String {
    json!({ "success": false, "error": message }).to_string()
}

fn default_output_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HERMES_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join("cache").join("audio");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
        .join("cache")
        .join("audio")
}

fn local_timestamp() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

/// Check if at least one TTS provider is available. Mirrors
/// `check_tts_requirements`, scoped to what's resolvable natively (command
/// providers + API keys for the network providers).
pub fn check_tts_requirements() -> bool {
    if has_any_command_tts_provider(None) {
        return true;
    }
    if !env_value("ELEVENLABS_API_KEY").is_empty() {
        return true;
    }
    if has_openai_audio_backend() {
        return true;
    }
    if !env_value("MINIMAX_API_KEY").is_empty() {
        return true;
    }
    if !env_value("XAI_API_KEY").is_empty() {
        return true;
    }
    if !env_value("GEMINI_API_KEY").is_empty() || !env_value("GOOGLE_API_KEY").is_empty() {
        return true;
    }
    if !env_value("MISTRAL_API_KEY").is_empty() {
        return true;
    }
    false
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn max_len_default_and_fallback() {
        let empty = json!({});
        assert_eq!(resolve_max_text_length(None, &empty), FALLBACK_MAX_TEXT_LENGTH);
        assert_eq!(resolve_max_text_length(Some(""), &empty), FALLBACK_MAX_TEXT_LENGTH);
        assert_eq!(resolve_max_text_length(Some("openai"), &empty), 4096);
        assert_eq!(resolve_max_text_length(Some("xai"), &empty), 15000);
        assert_eq!(resolve_max_text_length(Some("minimax"), &empty), 10000);
    }

    #[test]
    fn max_len_user_override_wins() {
        let cfg = json!({ "openai": { "max_text_length": 123 } });
        assert_eq!(resolve_max_text_length(Some("openai"), &cfg), 123);
        // bool is not an int override
        let cfg2 = json!({ "openai": { "max_text_length": true } });
        assert_eq!(resolve_max_text_length(Some("openai"), &cfg2), 4096);
        // non-positive falls through
        let cfg3 = json!({ "openai": { "max_text_length": 0 } });
        assert_eq!(resolve_max_text_length(Some("openai"), &cfg3), 4096);
    }

    #[test]
    fn max_len_elevenlabs_model_aware() {
        let cfg = json!({ "elevenlabs": { "model_id": "eleven_flash_v2_5" } });
        assert_eq!(resolve_max_text_length(Some("elevenlabs"), &cfg), 40000);
        let cfg2 = json!({ "elevenlabs": { "model_id": "unknown_model" } });
        assert_eq!(resolve_max_text_length(Some("elevenlabs"), &cfg2), 10000);
        let cfg3 = json!({});
        // default model_id -> eleven_multilingual_v2 -> 10000
        assert_eq!(resolve_max_text_length(Some("elevenlabs"), &cfg3), 10000);
    }

    #[test]
    fn max_len_command_provider() {
        let cfg = json!({
            "providers": { "myp": { "type": "command", "command": "x {output_path}" } }
        });
        assert_eq!(
            resolve_max_text_length(Some("myp"), &cfg),
            DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH
        );
        let cfg2 = json!({
            "providers": { "myp": { "type": "command", "command": "x", "max_text_length": 99 } }
        });
        assert_eq!(resolve_max_text_length(Some("myp"), &cfg2), 99);
    }

    #[test]
    fn command_provider_detection() {
        let cfg = json!({
            "providers": {
                "piper-en": { "type": "command", "command": "piper -f {output_path}" },
                "bad": { "type": "command", "command": "   " },
                "edge": { "type": "command", "command": "should be ignored" },
            }
        });
        assert!(resolve_command_provider_config("piper-en", &cfg).is_some());
        // built-in name short-circuits
        assert!(resolve_command_provider_config("edge", &cfg).is_none());
        // blank command -> not a command provider
        assert!(resolve_command_provider_config("bad", &cfg).is_none());
        let names: Vec<String> = iter_command_providers(&cfg).into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"piper-en".to_string()));
        assert!(!names.contains(&"edge".to_string()));
    }

    #[test]
    fn output_format_resolution() {
        let cfg = json!({ "output_format": "wav" });
        assert_eq!(get_command_tts_output_format(&cfg, None), "wav");
        // extension wins
        assert_eq!(get_command_tts_output_format(&cfg, Some("/x/y.ogg")), "ogg");
        // invalid -> default
        let cfg2 = json!({ "format": "aac" });
        assert_eq!(get_command_tts_output_format(&cfg2, None), "mp3");
    }

    #[test]
    fn timeout_resolution() {
        assert_eq!(get_command_tts_timeout(&json!({})), 120.0);
        assert_eq!(get_command_tts_timeout(&json!({ "timeout": 30 })), 30.0);
        assert_eq!(get_command_tts_timeout(&json!({ "timeout_seconds": 45.5 })), 45.5);
        // invalid / non-positive -> default
        assert_eq!(get_command_tts_timeout(&json!({ "timeout": -5 })), 120.0);
        assert_eq!(get_command_tts_timeout(&json!({ "timeout": "bad" })), 120.0);
    }

    #[test]
    fn voice_compatible_parsing() {
        assert!(is_command_tts_voice_compatible(&json!({ "voice_compatible": true })));
        assert!(is_command_tts_voice_compatible(&json!({ "voice_compatible": "yes" })));
        assert!(is_command_tts_voice_compatible(&json!({ "voice_compatible": "ON" })));
        assert!(!is_command_tts_voice_compatible(&json!({ "voice_compatible": "no" })));
        assert!(!is_command_tts_voice_compatible(&json!({})));
    }

    #[test]
    fn template_rendering_bare_and_quoted() {
        let mut ph = BTreeMap::new();
        ph.insert("output_path".to_string(), "/tmp/with space.mp3".to_string());
        ph.insert("voice".to_string(), "alloy".to_string());
        // bare context uses shlex.quote -> single-quoted because of the space
        let r = render_command_tts_template("synth -o {output_path} -v {voice}", &ph);
        assert_eq!(r, "synth -o '/tmp/with space.mp3' -v alloy");
        // literal braces preserved
        let r2 = render_command_tts_template("echo {{literal}} {voice}", &ph);
        assert_eq!(r2, "echo {literal} alloy");
    }

    #[test]
    fn template_rendering_inside_single_quotes() {
        let mut ph = BTreeMap::new();
        ph.insert("voice".to_string(), "o'brien".to_string());
        let r = render_command_tts_template("synth -v '{voice}'", &ph);
        // inside single quotes, ' becomes '\'' -> the value embedded
        assert_eq!(r, "synth -v 'o'\\''brien'");
    }

    #[test]
    fn template_skips_dollar_prefixed() {
        let mut ph = BTreeMap::new();
        ph.insert("voice".to_string(), "x".to_string());
        // `${voice}` should NOT be substituted (negative look-behind on $)
        let r = render_command_tts_template("echo ${voice}", &ph);
        assert_eq!(r, "echo ${voice}");
    }

    #[test]
    fn wav_header_is_well_formed() {
        let pcm = vec![1u8, 2, 3, 4];
        let wav = wrap_pcm_as_wav(&pcm, 24000, 1, 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // data chunk header at offset 36
        assert_eq!(&wav[36..40], b"data");
        // total = 44 byte header + 4 data bytes
        assert_eq!(wav.len(), 48);
        // riff size little-endian = 36 + 4 = 40
        let riff_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(riff_size, 40);
    }

    #[test]
    fn strip_markdown_basics() {
        assert_eq!(strip_markdown_for_tts("**bold** text"), "bold text");
        assert_eq!(strip_markdown_for_tts("# Header\nbody"), "Header\nbody");
        assert_eq!(strip_markdown_for_tts("see [link](http://x.com)"), "see link");
        assert_eq!(strip_markdown_for_tts("visit https://x.com now"), "visit  now");
    }

    #[test]
    fn sentence_boundary_detection() {
        assert_eq!(find_sentence_boundary("Hi there. More"), Some(10));
        assert_eq!(find_sentence_boundary("para one\n\npara two"), Some(10));
        assert_eq!(find_sentence_boundary("no boundary here"), None);
    }

    #[test]
    fn stream_buffer_extracts_sentences() {
        let mut sb = StreamBuffer::new();
        // short fragment merges, long one flushes
        let out = sb.push_delta("This is a long enough sentence to flush. ");
        assert_eq!(out.len(), 1);
        assert!(out[0].starts_with("This is a long enough"));
    }

    #[test]
    fn stream_buffer_strips_think_blocks() {
        let mut sb = StreamBuffer::new();
        // incomplete think block holds output
        let out = sb.push_delta("<think>internal");
        assert!(out.is_empty());
        let out2 = sb.push_delta(" reasoning</think>Hello there friend everyone. ");
        assert_eq!(out2.len(), 1);
        assert!(out2[0].contains("Hello there friend"));
    }

    #[test]
    fn stream_buffer_dedupes() {
        let mut sb = StreamBuffer::new();
        let _ = sb.push_delta("This is a repeated long sentence here. ");
        let again = sb.push_delta("This is a repeated long sentence here. ");
        assert!(again.is_empty());
    }

    #[test]
    fn tool_empty_text_errors() {
        let out = text_to_speech_tool("   ", None, "");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
        assert_eq!(v["error"], json!("Text is required"));
    }

    #[test]
    fn shlex_quote_matches_python() {
        assert_eq!(shlex_quote("simple"), "simple");
        assert_eq!(shlex_quote("with space"), "'with space'");
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("a/b-c.d"), "a/b-c.d");
    }

    #[test]
    fn run_command_tts_executes() {
        let r = run_command_tts("printf out; printf err 1>&2; exit 3", 5.0).unwrap();
        assert_eq!(r.returncode, 3);
        assert_eq!(r.stdout, "out");
        assert_eq!(r.stderr, "err");
        assert!(!r.timed_out);
    }

    #[test]
    fn run_command_tts_times_out() {
        let r = run_command_tts("sleep 5", 0.2).unwrap();
        assert!(r.timed_out);
    }

    #[test]
    fn check_requirements_with_env_key() {
        unsafe {
            std::env::set_var("XAI_API_KEY", "test-key");
        }
        assert!(check_tts_requirements());
        unsafe {
            std::env::remove_var("XAI_API_KEY");
        }
    }
}
