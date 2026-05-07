use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use base64::Engine;
use chrono::Local;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;

use crate::tools::{ToolRuntime, tool_error, tool_result};
use crate::{HermesContext, HermesError};

const DEFAULT_PROVIDER: &str = "edge";
const DEFAULT_EDGE_VOICE: &str = "en-US-AriaNeural";
const DEFAULT_ELEVENLABS_VOICE_ID: &str = "pNInz6obpgDQGcFmaJgB";
const DEFAULT_ELEVENLABS_MODEL_ID: &str = "eleven_multilingual_v2";
const DEFAULT_ELEVENLABS_BASE_URL: &str = "https://api.elevenlabs.io/v1";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini-tts";
const DEFAULT_OPENAI_VOICE: &str = "alloy";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_PIPER_VOICE: &str = "en_US-lessac-medium";
const DEFAULT_MINIMAX_MODEL: &str = "speech-01";
const DEFAULT_MINIMAX_VOICE_ID: &str = "female-shaonv";
const DEFAULT_MINIMAX_BASE_URL: &str = "https://api.minimax.chat/v1/text_to_speech";
const DEFAULT_GEMINI_TTS_MODEL: &str = "gemini-2.5-flash-preview-tts";
const DEFAULT_GEMINI_TTS_VOICE: &str = "Kore";
const DEFAULT_GEMINI_TTS_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MISTRAL_TTS_MODEL: &str = "voxtral-mini-tts-2603";
const DEFAULT_MISTRAL_TTS_VOICE_ID: &str = "c69964a6-ab8b-4f8a-9465-ec0925096ec8";
const DEFAULT_MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_XAI_VOICE_ID: &str = "eve";
const DEFAULT_XAI_LANGUAGE: &str = "en";
const DEFAULT_XAI_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_XAI_BIT_RATE: u32 = 128_000;
const DEFAULT_XAI_BASE_URL: &str = "https://api.x.ai/v1";
const DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS: u64 = 120;
const DEFAULT_COMMAND_TTS_OUTPUT_FORMAT: &str = "mp3";
const DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH: usize = 5_000;
const EDGE_MAX_TEXT_LENGTH: usize = 5_000;
const OPENAI_MAX_TEXT_LENGTH: usize = 4_096;
const MINIMAX_MAX_TEXT_LENGTH: usize = 10_000;
const GEMINI_MAX_TEXT_LENGTH: usize = 5_000;
const MISTRAL_MAX_TEXT_LENGTH: usize = 4_000;
const XAI_MAX_TEXT_LENGTH: usize = 15_000;
const FALLBACK_MAX_TEXT_LENGTH: usize = 4_000;
const GEMINI_TTS_SAMPLE_RATE: u32 = 24_000;
const GEMINI_TTS_CHANNELS: u16 = 1;
const GEMINI_TTS_SAMPLE_WIDTH: u16 = 2;

#[derive(Debug, Clone)]
struct CommandTtsProvider {
    command: String,
    timeout_seconds: f64,
    output_format: String,
    voice_compatible: bool,
    voice: String,
    model: String,
    speed: String,
    max_text_length: Option<usize>,
}

#[derive(Debug, Clone)]
struct TtsSettings {
    provider: String,
    output_dir: PathBuf,
    edge_voice: String,
    elevenlabs_voice_id: String,
    elevenlabs_model_id: String,
    elevenlabs_base_url: String,
    elevenlabs_api_key: String,
    openai_model: String,
    openai_voice: String,
    openai_base_url: String,
    openai_api_key: String,
    minimax_model: String,
    minimax_voice_id: String,
    minimax_base_url: String,
    minimax_api_key: String,
    piper_voice: String,
    piper_voices_dir: PathBuf,
    piper_use_cuda: bool,
    piper_pythonpath: Option<String>,
    piper_length_scale: Option<f64>,
    piper_noise_scale: Option<f64>,
    piper_noise_w_scale: Option<f64>,
    piper_volume: Option<f64>,
    piper_normalize_audio: Option<bool>,
    gemini_model: String,
    gemini_voice: String,
    gemini_base_url: String,
    gemini_api_key: String,
    mistral_model: String,
    mistral_voice_id: String,
    mistral_base_url: String,
    mistral_api_key: String,
    xai_voice_id: String,
    xai_language: String,
    xai_sample_rate: u32,
    xai_bit_rate: u32,
    xai_base_url: String,
    xai_api_key: String,
    command_provider: Option<CommandTtsProvider>,
}

pub fn text_to_speech_schema() -> Value {
    json!({
        "name": "text_to_speech",
        "description": "Convert text to speech audio. Supports built-in providers and configured tts.providers.<name> command backends. Returns a MEDIA path tag that compatible delivery surfaces can send as native audio.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to speech."
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional custom output path. Defaults to HERMES_HOME/cache/audio/tts_<timestamp>.<ext>."
                }
            },
            "required": ["text"]
        }
    })
}

pub fn handle_text_to_speech(args: &Value, runtime: &ToolRuntime) -> String {
    let Some(text) = args.get("text").and_then(Value::as_str).map(str::trim) else {
        return tool_error("Text is required");
    };
    if text.is_empty() {
        return tool_error("Text is required");
    }
    let output_path = match args.get("output_path") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if value.trim().is_empty() => None,
        Some(Value::String(value)) => Some(value.trim().to_string()),
        Some(_) => return tool_error("output_path must be a string"),
    };

    let settings = match load_tts_settings(runtime.hermes_home()) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let output_path = match resolve_output_path(runtime, &settings, output_path.as_deref()) {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };
    if let Some(parent) = output_path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        return tool_error(format!("creating {} failed: {error}", parent.display()));
    }

    let provider = settings.provider.as_str();
    let max_len = provider_max_text_length(&settings);
    let truncated = if text.chars().count() > max_len {
        text.chars().take(max_len).collect::<String>()
    } else {
        text.to_string()
    };

    let mut effective_output_path = output_path.clone();
    let result = match provider {
        other if settings.command_provider.is_some() && !is_builtin_tts_provider(other) => {
            synthesize_command_provider(
                settings.command_provider.as_ref().expect("checked is_some"),
                &truncated,
                &effective_output_path,
            )
        }
        "edge" => synthesize_edge(&settings, &truncated, &output_path),
        "elevenlabs" => synthesize_elevenlabs(&settings, &truncated, &output_path),
        "openai" => synthesize_openai(&settings, &truncated, &output_path),
        "piper" => synthesize_piper(&settings, &truncated, &output_path),
        "minimax" => synthesize_minimax(&settings, &truncated, &output_path),
        "gemini" => synthesize_gemini(&settings, &truncated, &output_path),
        "mistral" => synthesize_mistral(&settings, &truncated, &output_path),
        "xai" => synthesize_xai(&settings, &truncated, &output_path),
        other => Err(format!(
            "TTS provider '{other}' is not ported in the Rust runtime yet. Supported providers: edge, elevenlabs, openai, piper, minimax, gemini, mistral, xai, and configured tts.providers.<name> command backends."
        )),
    };
    if let Err(error) = result {
        return tool_error(error);
    }

    let voice_compatible = if let Some(config) = settings.command_provider.as_ref() {
        if config.voice_compatible {
            if !path_is_voice_compatible(&effective_output_path)
                && let Some(converted) = convert_audio_to_opus(&effective_output_path)
            {
                effective_output_path = converted;
            }
            path_is_voice_compatible(&effective_output_path)
        } else {
            false
        }
    } else {
        matches!(provider, "openai" | "mistral" | "elevenlabs" | "gemini")
            && path_is_voice_compatible(&effective_output_path)
    };

    let file_path = effective_output_path.display().to_string();
    let media_tag = if voice_compatible {
        format!("[[audio_as_voice]]\nMEDIA:{file_path}")
    } else {
        format!("MEDIA:{file_path}")
    };
    tool_result(json!({
        "success": true,
        "file_path": file_path,
        "media_tag": media_tag,
        "provider": provider,
        "voice_compatible": voice_compatible,
    }))
}

fn load_tts_settings(hermes_home: &Path) -> Result<TtsSettings, String> {
    let context =
        HermesContext::new(hermes_home).with_hermes_home_env(Some(hermes_home.to_path_buf()));
    let loaded = context
        .load_config_document()
        .map_err(|error| tts_config_error("loading config", error))?;
    let root = loaded.cfg_get(&["tts"]);

    let provider = yaml_mapping_value(root, &["provider"])
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string())
        .to_ascii_lowercase();
    let output_dir = yaml_mapping_value(root, &["output_dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| hermes_home.join("cache/audio"));
    let edge_voice = yaml_mapping_value(root, &["edge", "voice"])
        .unwrap_or_else(|| DEFAULT_EDGE_VOICE.to_string());
    let elevenlabs_voice_id = yaml_mapping_value(root, &["elevenlabs", "voice_id"])
        .unwrap_or_else(|| DEFAULT_ELEVENLABS_VOICE_ID.to_string());
    let elevenlabs_model_id = yaml_mapping_value(root, &["elevenlabs", "model_id"])
        .unwrap_or_else(|| DEFAULT_ELEVENLABS_MODEL_ID.to_string());
    let elevenlabs_base_url = yaml_mapping_value(root, &["elevenlabs", "base_url"])
        .or_else(|| std::env::var("ELEVENLABS_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_ELEVENLABS_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let elevenlabs_api_key = yaml_mapping_value(root, &["elevenlabs", "api_key"])
        .or_else(|| std::env::var("ELEVENLABS_API_KEY").ok())
        .unwrap_or_default();
    let openai_model = yaml_mapping_value(root, &["openai", "model"])
        .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string());
    let openai_voice = yaml_mapping_value(root, &["openai", "voice"])
        .unwrap_or_else(|| DEFAULT_OPENAI_VOICE.to_string());
    let openai_base_url = yaml_mapping_value(root, &["openai", "base_url"])
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let openai_api_key = yaml_mapping_value(root, &["openai", "api_key"])
        .or_else(|| std::env::var("VOICE_TOOLS_OPENAI_KEY").ok())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_default();
    let minimax_model = yaml_mapping_value(root, &["minimax", "model"])
        .unwrap_or_else(|| DEFAULT_MINIMAX_MODEL.to_string());
    let minimax_voice_id = yaml_mapping_value(root, &["minimax", "voice_id"])
        .unwrap_or_else(|| DEFAULT_MINIMAX_VOICE_ID.to_string());
    let minimax_base_url = yaml_mapping_value(root, &["minimax", "base_url"])
        .or_else(|| std::env::var("MINIMAX_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_MINIMAX_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let minimax_api_key = yaml_mapping_value(root, &["minimax", "api_key"])
        .or_else(|| std::env::var("MINIMAX_API_KEY").ok())
        .unwrap_or_default();
    let piper_voice = yaml_mapping_value(root, &["piper", "voice"])
        .unwrap_or_else(|| DEFAULT_PIPER_VOICE.to_string());
    let piper_voices_dir = yaml_mapping_value(root, &["piper", "voices_dir"])
        .map(|value| expand_user_path(&value))
        .unwrap_or_else(|| hermes_home.join("cache/piper-voices"));
    let piper_use_cuda = yaml_mapping_bool(root, &["piper", "use_cuda"]).unwrap_or(false);
    let piper_pythonpath = yaml_mapping_value(root, &["piper", "pythonpath"]);
    let piper_length_scale = yaml_mapping_f64(root, &["piper", "length_scale"]);
    let piper_noise_scale = yaml_mapping_f64(root, &["piper", "noise_scale"]);
    let piper_noise_w_scale = yaml_mapping_f64(root, &["piper", "noise_w_scale"]);
    let piper_volume = yaml_mapping_f64(root, &["piper", "volume"]);
    let piper_normalize_audio = yaml_mapping_bool(root, &["piper", "normalize_audio"]);
    let gemini_model = yaml_mapping_value(root, &["gemini", "model"])
        .unwrap_or_else(|| DEFAULT_GEMINI_TTS_MODEL.to_string());
    let gemini_voice = yaml_mapping_value(root, &["gemini", "voice"])
        .unwrap_or_else(|| DEFAULT_GEMINI_TTS_VOICE.to_string());
    let gemini_base_url = yaml_mapping_value(root, &["gemini", "base_url"])
        .or_else(|| std::env::var("GEMINI_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_GEMINI_TTS_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let gemini_api_key = yaml_mapping_value(root, &["gemini", "api_key"])
        .or_else(|| std::env::var("GEMINI_API_KEY").ok())
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .unwrap_or_default();
    let mistral_model = yaml_mapping_value(root, &["mistral", "model"])
        .unwrap_or_else(|| DEFAULT_MISTRAL_TTS_MODEL.to_string());
    let mistral_voice_id = yaml_mapping_value(root, &["mistral", "voice_id"])
        .unwrap_or_else(|| DEFAULT_MISTRAL_TTS_VOICE_ID.to_string());
    let mistral_base_url = yaml_mapping_value(root, &["mistral", "base_url"])
        .unwrap_or_else(|| DEFAULT_MISTRAL_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let mistral_api_key = yaml_mapping_value(root, &["mistral", "api_key"])
        .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
        .unwrap_or_default();
    let xai_voice_id = yaml_mapping_value(root, &["xai", "voice_id"])
        .unwrap_or_else(|| DEFAULT_XAI_VOICE_ID.to_string());
    let xai_language = yaml_mapping_value(root, &["xai", "language"])
        .unwrap_or_else(|| DEFAULT_XAI_LANGUAGE.to_string());
    let xai_sample_rate =
        yaml_mapping_u32(root, &["xai", "sample_rate"]).unwrap_or(DEFAULT_XAI_SAMPLE_RATE);
    let xai_bit_rate = yaml_mapping_u32(root, &["xai", "bit_rate"]).unwrap_or(DEFAULT_XAI_BIT_RATE);
    let xai_base_url = yaml_mapping_value(root, &["xai", "base_url"])
        .or_else(|| std::env::var("XAI_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_XAI_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let xai_api_key = yaml_mapping_value(root, &["xai", "api_key"])
        .or_else(|| std::env::var("XAI_API_KEY").ok())
        .unwrap_or_default();
    let command_provider = load_command_provider(root, &provider);

    Ok(TtsSettings {
        provider,
        output_dir,
        edge_voice,
        elevenlabs_voice_id,
        elevenlabs_model_id,
        elevenlabs_base_url,
        elevenlabs_api_key,
        openai_model,
        openai_voice,
        openai_base_url,
        openai_api_key,
        minimax_model,
        minimax_voice_id,
        minimax_base_url,
        minimax_api_key,
        piper_voice,
        piper_voices_dir,
        piper_use_cuda,
        piper_pythonpath,
        piper_length_scale,
        piper_noise_scale,
        piper_noise_w_scale,
        piper_volume,
        piper_normalize_audio,
        gemini_model,
        gemini_voice,
        gemini_base_url,
        gemini_api_key,
        mistral_model,
        mistral_voice_id,
        mistral_base_url,
        mistral_api_key,
        xai_voice_id,
        xai_language,
        xai_sample_rate,
        xai_bit_rate,
        xai_base_url,
        xai_api_key,
        command_provider,
    })
}

fn load_command_provider(root: Option<&YamlValue>, provider: &str) -> Option<CommandTtsProvider> {
    if provider.is_empty() || is_builtin_tts_provider(provider) {
        return None;
    }
    let config = get_named_provider_config(root, provider)?;
    if !is_command_provider_config(config) {
        return None;
    }
    let output_format = get_command_tts_output_format(config, None);
    let max_text_length = yaml_value_usize(mapping_get_case_insensitive(config, "max_text_length"))
        .filter(|value| *value > 0);
    Some(CommandTtsProvider {
        command: mapping_get_case_insensitive(config, "command")
            .and_then(yaml_value_string)
            .expect("validated by is_command_provider_config"),
        timeout_seconds: get_command_tts_timeout(config),
        output_format,
        voice_compatible: yaml_value_bool(mapping_get_case_insensitive(config, "voice_compatible")),
        voice: mapping_get_case_insensitive(config, "voice")
            .and_then(yaml_value_string)
            .unwrap_or_default(),
        model: mapping_get_case_insensitive(config, "model")
            .and_then(yaml_value_string)
            .unwrap_or_default(),
        speed: mapping_get_case_insensitive(config, "speed")
            .and_then(yaml_value_string)
            .or_else(|| yaml_mapping_scalar_string(root, &["speed"]))
            .unwrap_or_default(),
        max_text_length,
    })
}

fn synthesize_edge(settings: &TtsSettings, text: &str, output_path: &Path) -> Result<(), String> {
    let status = Command::new("edge-tts")
        .arg("--voice")
        .arg(&settings.edge_voice)
        .arg("--text")
        .arg(text)
        .arg("--write-media")
        .arg(output_path)
        .status()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                "edge-tts is not installed or not on PATH.".to_string()
            } else {
                format!("edge-tts invocation failed: {error}")
            }
        })?;
    if !status.success() {
        return Err(format!(
            "edge-tts exited with code {}",
            status.code().unwrap_or(-1)
        ));
    }
    ensure_audio_file(output_path)
}

fn synthesize_elevenlabs(
    settings: &TtsSettings,
    text: &str,
    output_path: &Path,
) -> Result<(), String> {
    if settings.elevenlabs_api_key.trim().is_empty() {
        return Err(
            "ElevenLabs TTS requires ELEVENLABS_API_KEY or tts.elevenlabs.api_key.".to_string(),
        );
    }
    let output_format = match output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("ogg") | Some("opus") => "opus_48000_64",
        _ => "mp3_44100_128",
    };
    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let url = format!(
        "{}/text-to-speech/{}",
        settings.elevenlabs_base_url, settings.elevenlabs_voice_id
    );
    let response = client
        .post(&url)
        .header("xi-api-key", &settings.elevenlabs_api_key)
        .header("Content-Type", "application/json")
        .query(&[("output_format", output_format)])
        .json(&json!({
            "text": text,
            "model_id": settings.elevenlabs_model_id,
        }))
        .send()
        .map_err(|error| format!("ElevenLabs TTS request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!(
            "ElevenLabs TTS request failed: {} {}",
            status, body
        ));
    }
    let bytes = response
        .bytes()
        .map_err(|error| format!("reading ElevenLabs TTS response failed: {error}"))?;
    fs::write(output_path, &bytes)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    ensure_audio_file(output_path)
}

fn synthesize_openai(settings: &TtsSettings, text: &str, output_path: &Path) -> Result<(), String> {
    if settings.openai_api_key.trim().is_empty() {
        return Err(
            "OpenAI TTS requires VOICE_TOOLS_OPENAI_KEY, OPENAI_API_KEY, or tts.openai.api_key."
                .to_string(),
        );
    }
    let response_format = match output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("ogg") | Some("opus") => "opus",
        Some("wav") => "wav",
        _ => "mp3",
    };
    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let url = format!("{}/audio/speech", settings.openai_base_url);
    let response = client
        .post(&url)
        .bearer_auth(&settings.openai_api_key)
        .json(&json!({
            "model": settings.openai_model,
            "voice": settings.openai_voice,
            "input": text,
            "response_format": response_format,
        }))
        .send()
        .map_err(|error| format!("OpenAI TTS request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("OpenAI TTS request failed: {} {}", status, body));
    }
    let bytes = response
        .bytes()
        .map_err(|error| format!("reading OpenAI TTS response failed: {error}"))?;
    fs::write(output_path, &bytes)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    ensure_audio_file(output_path)
}

fn synthesize_xai(settings: &TtsSettings, text: &str, output_path: &Path) -> Result<(), String> {
    if settings.xai_api_key.trim().is_empty() {
        return Err("xAI TTS requires XAI_API_KEY or tts.xai.api_key.".to_string());
    }
    let codec = match output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("wav") => "wav",
        _ => "mp3",
    };
    let mut payload = json!({
        "text": text,
        "voice_id": settings.xai_voice_id,
        "language": settings.xai_language,
    });
    let needs_output_format = codec != "mp3"
        || settings.xai_sample_rate != DEFAULT_XAI_SAMPLE_RATE
        || settings.xai_bit_rate != DEFAULT_XAI_BIT_RATE;
    if needs_output_format {
        let mut output_format = json!({
            "codec": codec,
            "sample_rate": settings.xai_sample_rate,
        });
        if codec == "mp3" {
            output_format["bit_rate"] = json!(settings.xai_bit_rate);
        }
        payload["output_format"] = output_format;
    }

    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let url = format!("{}/tts", settings.xai_base_url);
    let response = client
        .post(&url)
        .bearer_auth(&settings.xai_api_key)
        .header("Content-Type", "application/json")
        .header("User-Agent", "hermes-agent/1.0")
        .json(&payload)
        .send()
        .map_err(|error| format!("xAI TTS request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("xAI TTS request failed: {} {}", status, body));
    }
    let bytes = response
        .bytes()
        .map_err(|error| format!("reading xAI TTS response failed: {error}"))?;
    fs::write(output_path, &bytes)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    ensure_audio_file(output_path)
}

fn synthesize_piper(settings: &TtsSettings, text: &str, output_path: &Path) -> Result<(), String> {
    let model_path = resolve_piper_voice_path(settings)?;
    let wav_path = if output_path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        output_path.to_path_buf()
    } else {
        output_path.with_extension("wav")
    };
    let synth_code = r#"
import json
import sys
import wave
from piper import PiperVoice

text, model_path, wav_path, use_cuda_raw, knobs_raw = sys.argv[1:6]
use_cuda = use_cuda_raw == "1"
voice = PiperVoice.load(model_path, use_cuda=use_cuda)
knobs = json.loads(knobs_raw)
syn_config = None
if knobs:
    try:
        from piper import SynthesisConfig
        syn_config = SynthesisConfig(**knobs)
    except Exception:
        syn_config = None
with wave.open(wav_path, "wb") as wav_file:
    if syn_config is not None:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)
    else:
        voice.synthesize_wav(text, wav_file)
"#;
    let knobs = piper_synthesis_knobs(settings);
    let interpreter = resolve_python_interpreter();
    let mut command = Command::new(&interpreter);
    command
        .arg("-c")
        .arg(synth_code)
        .arg(text)
        .arg(&model_path)
        .arg(&wav_path)
        .arg(if settings.piper_use_cuda { "1" } else { "0" })
        .arg(serde_json::to_string(&knobs).unwrap_or_else(|_| "{}".to_string()));
    apply_pythonpath_override(&mut command, settings.piper_pythonpath.as_deref());
    let output = command.output().map_err(|error| {
        format!(
            "starting Piper synthesis with {} failed: {error}",
            interpreter.display()
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'piper'")
            || stderr.contains("No module named \"piper\"")
        {
            return Err(
                "Piper provider selected but 'piper-tts' is not installed. Install it with: pip install piper-tts"
                    .to_string(),
            );
        }
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            format!(
                "Piper synthesis exited with code {}",
                output.status.code().unwrap_or(-1)
            )
        } else {
            format!(
                "Piper synthesis exited with code {}: {}",
                output.status.code().unwrap_or(-1),
                detail
            )
        });
    }
    ensure_audio_file(&wav_path)?;
    finalize_wav_output(&wav_path, output_path)
}

fn synthesize_minimax(
    settings: &TtsSettings,
    text: &str,
    output_path: &Path,
) -> Result<(), String> {
    if settings.minimax_api_key.trim().is_empty() {
        return Err("MiniMax TTS requires MINIMAX_API_KEY or tts.minimax.api_key.".to_string());
    }
    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let response = client
        .post(&settings.minimax_base_url)
        .bearer_auth(&settings.minimax_api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": settings.minimax_model,
            "text": text,
            "voice_id": settings.minimax_voice_id,
        }))
        .send()
        .map_err(|error| format!("MiniMax TTS request failed: {error}"))?;
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let bytes = response
        .bytes()
        .map_err(|error| format!("reading MiniMax TTS response failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "MiniMax TTS request failed: {} {}",
            status,
            String::from_utf8_lossy(&bytes)
        ));
    }
    let audio = if content_type.contains("audio/") {
        bytes.to_vec()
    } else {
        let body: Value = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "MiniMax TTS returned unexpected content-type '{}': {}",
                content_type, error
            )
        })?;
        let status_code = body
            .pointer("/base_resp/status_code")
            .and_then(Value::as_i64)
            .unwrap_or(-1);
        if status_code != 0 {
            let status_msg = body
                .pointer("/base_resp/status_msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(format!(
                "MiniMax TTS API error (code {status_code}): {status_msg}"
            ));
        }
        let hex_audio = body
            .pointer("/data/audio")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "MiniMax TTS returned empty audio data".to_string())?;
        decode_hex_bytes(hex_audio)?
    };
    fs::write(output_path, &audio)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    ensure_audio_file(output_path)
}

fn synthesize_gemini(settings: &TtsSettings, text: &str, output_path: &Path) -> Result<(), String> {
    if settings.gemini_api_key.trim().is_empty() {
        return Err(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey".to_string(),
        );
    }
    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let url = format!(
        "{}/models/{}:generateContent",
        settings.gemini_base_url, settings.gemini_model
    );
    let response = client
        .post(&url)
        .query(&[("key", settings.gemini_api_key.as_str())])
        .header("Content-Type", "application/json")
        .json(&json!({
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": settings.gemini_voice,
                        }
                    }
                }
            }
        }))
        .send()
        .map_err(|error| format!("Gemini TTS request failed: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|error| format!("reading Gemini TTS response failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "Gemini TTS API error (HTTP {}): {}",
            status.as_u16(),
            extract_tts_api_error_message(&body)
        ));
    }
    let parsed = serde_json::from_str::<Value>(&body)
        .map_err(|error| format!("Gemini TTS response was malformed: {error}"))?;
    let audio_b64 = parsed
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .and_then(|parts| {
            parts.iter().find_map(|part| {
                part.get("inlineData")
                    .or_else(|| part.get("inline_data"))
                    .and_then(Value::as_object)
                    .and_then(|inline| inline.get("data"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
            })
        })
        .ok_or_else(|| "Gemini TTS response contained no audio data".to_string())?;
    let pcm_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64)
        .map_err(|error| format!("decoding Gemini TTS audio failed: {error}"))?;
    if pcm_bytes.is_empty() {
        return Err("Gemini TTS returned empty audio data".to_string());
    }
    let wav_bytes = wrap_pcm_as_wav(
        &pcm_bytes,
        GEMINI_TTS_SAMPLE_RATE,
        GEMINI_TTS_CHANNELS,
        GEMINI_TTS_SAMPLE_WIDTH,
    );
    write_wav_or_transcode(&wav_bytes, output_path)
}

fn synthesize_mistral(
    settings: &TtsSettings,
    text: &str,
    output_path: &Path,
) -> Result<(), String> {
    if settings.mistral_api_key.trim().is_empty() {
        return Err("Mistral TTS requires MISTRAL_API_KEY or tts.mistral.api_key.".to_string());
    }
    let response_format = match output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("ogg") | Some("opus") => "opus",
        Some("wav") => "wav",
        Some("flac") => "flac",
        _ => "mp3",
    };
    let client = Client::builder()
        .build()
        .map_err(|error| format!("building HTTP client failed: {error}"))?;
    let url = format!("{}/audio/speech", settings.mistral_base_url);
    let response = client
        .post(&url)
        .bearer_auth(&settings.mistral_api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": settings.mistral_model,
            "input": text,
            "voice_id": settings.mistral_voice_id,
            "response_format": response_format,
        }))
        .send()
        .map_err(|error| format!("Mistral TTS request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("Mistral TTS request failed: {} {}", status, body));
    }
    let body: Value = response
        .json()
        .map_err(|error| format!("reading Mistral TTS response failed: {error}"))?;
    let audio_data = body
        .get("audio_data")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Mistral TTS response did not contain audio_data".to_string())?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_data)
        .map_err(|error| format!("decoding Mistral TTS audio failed: {error}"))?;
    fs::write(output_path, &bytes)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    ensure_audio_file(output_path)
}

fn resolve_output_path(
    runtime: &ToolRuntime,
    settings: &TtsSettings,
    explicit: Option<&str>,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        let resolved = runtime.resolve_path(path)?;
        if let Some(config) = settings.command_provider.as_ref() {
            return Ok(configured_command_tts_output_path(&resolved, config));
        }
        return Ok(resolved);
    }
    let platform = std::env::var("HERMES_SESSION_PLATFORM")
        .ok()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if let Some(config) = settings.command_provider.as_ref() {
        return Ok(settings.output_dir.join(format!(
            "tts_{}_{:x}.{}",
            Local::now().format("%Y%m%d_%H%M%S"),
            unix_ts_nanos(),
            config.output_format
        )));
    }
    let extension = if matches!(
        settings.provider.as_str(),
        "openai" | "mistral" | "elevenlabs" | "gemini"
    ) && platform == "telegram"
    {
        "ogg"
    } else {
        "mp3"
    };
    Ok(settings.output_dir.join(format!(
        "tts_{}_{:x}.{}",
        Local::now().format("%Y%m%d_%H%M%S"),
        unix_ts_nanos(),
        extension
    )))
}

fn ensure_audio_file(path: &Path) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    if metadata.len() == 0 {
        return Err(format!(
            "TTS generation produced no output at {}",
            path.display()
        ));
    }
    Ok(())
}

fn yaml_mapping_value(root: Option<&YamlValue>, path: &[&str]) -> Option<String> {
    let mut current = root?;
    for key in path {
        current = current
            .as_mapping()?
            .get(YamlValue::String((*key).to_string()))?;
    }
    current
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn yaml_mapping_scalar_string(root: Option<&YamlValue>, path: &[&str]) -> Option<String> {
    yaml_lookup(root, path).and_then(yaml_value_string)
}

fn yaml_mapping_u32(root: Option<&YamlValue>, path: &[&str]) -> Option<u32> {
    match yaml_lookup(root, path)? {
        YamlValue::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        YamlValue::String(value) => value.trim().parse::<u32>().ok(),
        _ => None,
    }
}

fn yaml_mapping_f64(root: Option<&YamlValue>, path: &[&str]) -> Option<f64> {
    match yaml_lookup(root, path)? {
        YamlValue::Number(value) => value.as_f64(),
        YamlValue::String(value) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn yaml_mapping_bool(root: Option<&YamlValue>, path: &[&str]) -> Option<bool> {
    yaml_lookup(root, path).map(|value| yaml_value_bool(Some(value)))
}

fn yaml_lookup<'a>(root: Option<&'a YamlValue>, path: &[&str]) -> Option<&'a YamlValue> {
    let mut current = root?;
    for key in path {
        current = current
            .as_mapping()?
            .get(YamlValue::String((*key).to_string()))?;
    }
    Some(current)
}

fn yaml_value_string(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        YamlValue::Number(value) => Some(value.to_string()),
        YamlValue::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn yaml_value_bool(value: Option<&YamlValue>) -> bool {
    match value {
        Some(YamlValue::Bool(value)) => *value,
        Some(YamlValue::String(value)) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Some(YamlValue::Number(value)) => value.as_i64().is_some_and(|value| value != 0),
        _ => false,
    }
}

fn yaml_value_usize(value: Option<&YamlValue>) -> Option<usize> {
    match value? {
        YamlValue::Number(value) => value.as_u64().and_then(|value| usize::try_from(value).ok()),
        YamlValue::String(value) => value.trim().parse::<usize>().ok(),
        _ => None,
    }
}

fn provider_root_mapping<'a>(root: Option<&'a YamlValue>) -> Option<&'a serde_yaml::Mapping> {
    root?.as_mapping()
}

fn mapping_get_case_insensitive<'a>(
    mapping: &'a serde_yaml::Mapping,
    key: &str,
) -> Option<&'a YamlValue> {
    mapping.iter().find_map(|(candidate, value)| {
        candidate
            .as_str()
            .filter(|candidate| candidate.eq_ignore_ascii_case(key))
            .map(|_| value)
    })
}

fn get_named_provider_config<'a>(
    root: Option<&'a YamlValue>,
    provider: &str,
) -> Option<&'a serde_yaml::Mapping> {
    let root_mapping = provider_root_mapping(root)?;
    if let Some(section) = mapping_get_case_insensitive(root_mapping, "providers")
        .and_then(YamlValue::as_mapping)
        .and_then(|providers| mapping_get_case_insensitive(providers, provider))
        .and_then(YamlValue::as_mapping)
    {
        return Some(section);
    }
    if is_builtin_tts_provider(provider) {
        return None;
    }
    mapping_get_case_insensitive(root_mapping, provider).and_then(YamlValue::as_mapping)
}

fn is_command_provider_config(config: &serde_yaml::Mapping) -> bool {
    if let Some(kind) = mapping_get_case_insensitive(config, "type")
        .and_then(yaml_value_string)
        .map(|value| value.to_ascii_lowercase())
        && kind != "command"
    {
        return false;
    }
    mapping_get_case_insensitive(config, "command")
        .and_then(yaml_value_string)
        .is_some()
}

fn is_builtin_tts_provider(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "edge" | "elevenlabs" | "openai" | "piper" | "minimax" | "gemini" | "mistral" | "xai"
    )
}

fn get_command_tts_timeout(config: &serde_yaml::Mapping) -> f64 {
    let value = mapping_get_case_insensitive(config, "timeout")
        .or_else(|| mapping_get_case_insensitive(config, "timeout_seconds"));
    let Some(value) = value else {
        return DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS as f64;
    };
    let parsed = match value {
        YamlValue::Number(number) => number.as_f64(),
        YamlValue::String(raw) => raw.trim().parse::<f64>().ok(),
        _ => None,
    };
    match parsed {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS as f64,
    }
}

fn get_command_tts_output_format(
    config: &serde_yaml::Mapping,
    output_path: Option<&Path>,
) -> String {
    if let Some(path) = output_path
        && let Some(extension) = path.extension().and_then(|value| value.to_str())
    {
        let normalized = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        if is_command_tts_output_format(&normalized) {
            return normalized;
        }
    }
    let configured = mapping_get_case_insensitive(config, "format")
        .or_else(|| mapping_get_case_insensitive(config, "output_format"))
        .and_then(yaml_value_string)
        .unwrap_or_else(|| DEFAULT_COMMAND_TTS_OUTPUT_FORMAT.to_string());
    let normalized = configured
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    if is_command_tts_output_format(&normalized) {
        normalized
    } else {
        DEFAULT_COMMAND_TTS_OUTPUT_FORMAT.to_string()
    }
}

fn is_command_tts_output_format(value: &str) -> bool {
    matches!(value, "mp3" | "wav" | "ogg" | "flac")
}

fn configured_command_tts_output_path(path: &Path, config: &CommandTtsProvider) -> PathBuf {
    let format = get_command_tts_output_format_from_provider(config, Some(path));
    path.with_extension(format)
}

fn get_command_tts_output_format_from_provider(
    config: &CommandTtsProvider,
    output_path: Option<&Path>,
) -> String {
    if let Some(path) = output_path
        && let Some(extension) = path.extension().and_then(|value| value.to_str())
    {
        let normalized = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        if is_command_tts_output_format(&normalized) {
            return normalized;
        }
    }
    config.output_format.clone()
}

fn provider_max_text_length(settings: &TtsSettings) -> usize {
    match settings.provider.as_str() {
        "edge" => EDGE_MAX_TEXT_LENGTH,
        "openai" => OPENAI_MAX_TEXT_LENGTH,
        "piper" => DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH,
        "minimax" => MINIMAX_MAX_TEXT_LENGTH,
        "gemini" => GEMINI_MAX_TEXT_LENGTH,
        "mistral" => MISTRAL_MAX_TEXT_LENGTH,
        "xai" => XAI_MAX_TEXT_LENGTH,
        "elevenlabs" => match settings.elevenlabs_model_id.trim() {
            "eleven_v3" | "eleven_ttv_v3" => 5_000,
            "eleven_multilingual_v2"
            | "eleven_multilingual_v1"
            | "eleven_english_sts_v2"
            | "eleven_english_sts_v1" => 10_000,
            "eleven_flash_v2" => 30_000,
            "eleven_flash_v2_5" => 40_000,
            _ => 10_000,
        },
        other if !is_builtin_tts_provider(other) => settings
            .command_provider
            .as_ref()
            .and_then(|config| config.max_text_length)
            .unwrap_or(DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH),
        _ => FALLBACK_MAX_TEXT_LENGTH,
    }
}

fn synthesize_command_provider(
    config: &CommandTtsProvider,
    text: &str,
    output_path: &Path,
) -> Result<(), String> {
    if config.command.trim().is_empty() {
        return Err("TTS command provider command is not configured.".to_string());
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("creating {} failed: {error}", parent.display()))?;
    }
    if output_path.exists() {
        let _ = fs::remove_file(output_path);
    }
    let temp_dir = std::env::temp_dir().join(format!("hermes_tts_{:x}", unix_ts_nanos()));
    fs::create_dir_all(&temp_dir)
        .map_err(|error| format!("creating {} failed: {error}", temp_dir.display()))?;
    let input_path = temp_dir.join("input.txt");
    fs::write(&input_path, text)
        .map_err(|error| format!("writing {} failed: {error}", input_path.display()))?;

    let placeholders = [
        ("input_path", input_path.display().to_string()),
        ("text_path", input_path.display().to_string()),
        ("output_path", output_path.display().to_string()),
        (
            "format",
            get_command_tts_output_format_from_provider(config, Some(output_path)),
        ),
        ("voice", config.voice.clone()),
        ("model", config.model.clone()),
        ("speed", config.speed.clone()),
    ];
    let command = render_command_tts_template(&config.command, &placeholders);
    let result = run_command_tts(&command, Duration::from_secs_f64(config.timeout_seconds));
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_dir_all(&temp_dir);

    match result {
        Ok(_) => ensure_audio_file(output_path),
        Err(CommandTtsError::TimedOut) => Err(format!(
            "TTS provider timed out after {}s",
            trim_decimal(config.timeout_seconds)
        )),
        Err(CommandTtsError::Exited {
            code,
            stdout,
            stderr,
        }) => {
            let mut detail_parts = Vec::new();
            let stderr = stderr.trim();
            if !stderr.is_empty() {
                detail_parts.push(format!("stderr: {stderr}"));
            }
            let stdout = stdout.trim();
            if !stdout.is_empty() {
                detail_parts.push(format!("stdout: {stdout}"));
            }
            let detail = if detail_parts.is_empty() {
                "no command output".to_string()
            } else {
                detail_parts.join("; ")
            };
            Err(format!(
                "TTS provider exited with code {}: {}",
                code.unwrap_or(-1),
                detail
            ))
        }
        Err(CommandTtsError::Spawn(error)) => Err(error),
    }
}

#[derive(Debug)]
enum CommandTtsError {
    TimedOut,
    Exited {
        code: Option<i32>,
        stdout: String,
        stderr: String,
    },
    Spawn(String),
}

fn run_command_tts(command: &str, timeout: Duration) -> Result<(), CommandTtsError> {
    let mut builder = if cfg!(windows) {
        let mut builder = Command::new("cmd");
        builder.arg("/C").arg(command);
        builder
    } else {
        let mut builder = Command::new("sh");
        builder.arg("-c").arg(command);
        builder
    };
    builder
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        // SAFETY: setsid is called in the child immediately before exec so the
        // shell runs in its own process group and can be terminated as a unit.
        unsafe {
            builder.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let mut child = builder
        .spawn()
        .map_err(|error| CommandTtsError::Spawn(format!("spawning TTS command failed: {error}")))?;
    let pid = child.id() as i32;
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let output = child.wait_with_output().map_err(|error| {
                    CommandTtsError::Spawn(format!("collecting TTS command output failed: {error}"))
                })?;
                if output.status.success() {
                    return Ok(());
                }
                return Err(CommandTtsError::Exited {
                    code: output.status.code(),
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                });
            }
            Ok(None) if started.elapsed() >= timeout => {
                terminate_command_tts_process_group(pid, &mut child);
                return Err(CommandTtsError::TimedOut);
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(error) => {
                terminate_command_tts_process_group(pid, &mut child);
                return Err(CommandTtsError::Spawn(format!(
                    "waiting for TTS command failed: {error}"
                )));
            }
        }
    }
}

fn terminate_command_tts_process_group(pid: i32, child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let _ = signal_process_group(pid, libc::SIGTERM);
        thread::sleep(Duration::from_millis(250));
        if child.try_wait().ok().flatten().is_none() {
            let _ = signal_process_group(pid, libc::SIGKILL);
        }
        let _ = child.wait();
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(unix)]
fn signal_process_group(pid: i32, signal: i32) -> Result<(), String> {
    let result = unsafe { libc::killpg(pid, signal) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().to_string())
    }
}

fn render_command_tts_template(template: &str, placeholders: &[(&str, String)]) -> String {
    let mut rendered = String::with_capacity(template.len() + 64);
    let bytes = template.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'{' {
            if bytes.get(index + 1) == Some(&b'{') {
                rendered.push('{');
                index += 2;
                continue;
            }
            if let Some(end) = bytes[index + 1..].iter().position(|byte| *byte == b'}') {
                let end = index + 1 + end;
                let name = &template[index + 1..end];
                if bytes.get(index.wrapping_sub(1)) != Some(&b'$')
                    && let Some((_, value)) = placeholders
                        .iter()
                        .find(|(candidate, _)| *candidate == name)
                {
                    rendered.push_str(&quote_command_tts_placeholder(
                        value,
                        shell_quote_context(template, index),
                    ));
                    index = end + 1;
                    continue;
                }
            }
        }
        if bytes[index] == b'}' && bytes.get(index + 1) == Some(&b'}') {
            rendered.push('}');
            index += 2;
            continue;
        }
        rendered.push(bytes[index] as char);
        index += 1;
    }
    rendered
}

fn shell_quote_context(command_template: &str, position: usize) -> Option<char> {
    let bytes = command_template.as_bytes();
    let mut quote = None;
    let mut escaped = false;
    let mut index = 0usize;
    while index < position {
        let byte = bytes[index];
        match quote {
            Some('\'') => {
                if byte == b'\'' {
                    quote = None;
                }
            }
            Some('"') => {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quote = None;
                }
            }
            _ => {
                if byte == b'\'' {
                    quote = Some('\'');
                } else if byte == b'"' {
                    quote = Some('"');
                } else if byte == b'\\' {
                    index += 1;
                }
            }
        }
        index += 1;
    }
    quote
}

fn quote_command_tts_placeholder(value: &str, quote_context: Option<char>) -> String {
    match quote_context {
        Some('\'') => value.replace('\'', r"'\''"),
        Some('"') => value
            .replace('\\', r"\\")
            .replace('"', r#"\""#)
            .replace('$', r"\$")
            .replace('`', r"\`"),
        _ => shell_quote(value),
    }
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if cfg!(windows) {
        let escaped = value.replace('"', r#"\""#);
        return format!(r#""{escaped}""#);
    }
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&byte))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', r#"'"'"'"#))
}

fn convert_audio_to_opus(path: &Path) -> Option<PathBuf> {
    if path_is_voice_compatible(path) {
        return Some(path.to_path_buf());
    }
    let output_path = path.with_extension("ogg");
    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(path)
        .arg("-acodec")
        .arg("libopus")
        .arg("-ac")
        .arg("1")
        .arg("-b:a")
        .arg("64k")
        .arg("-vbr")
        .arg("off")
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg(&output_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    if ensure_audio_file(&output_path).is_ok() {
        Some(output_path)
    } else {
        None
    }
}

fn finalize_wav_output(wav_path: &Path, output_path: &Path) -> Result<(), String> {
    if wav_path == output_path {
        return ensure_audio_file(output_path);
    }
    if ffmpeg_available() {
        let status = Command::new("ffmpeg")
            .arg("-i")
            .arg(wav_path)
            .arg("-y")
            .arg("-loglevel")
            .arg("error")
            .arg(output_path)
            .status()
            .map_err(|error| format!("ffmpeg conversion failed: {error}"))?;
        let _ = fs::remove_file(wav_path);
        if !status.success() {
            return Err(format!(
                "ffmpeg conversion failed with code {}",
                status.code().unwrap_or(-1)
            ));
        }
        return ensure_audio_file(output_path);
    }
    fs::rename(wav_path, output_path).map_err(|error| {
        format!(
            "moving {} to {} failed: {error}",
            wav_path.display(),
            output_path.display()
        )
    })?;
    ensure_audio_file(output_path)
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn resolve_python_interpreter() -> PathBuf {
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
        let candidate = PathBuf::from(venv).join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        });
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(if python_command_available("python3") {
        "python3"
    } else {
        "python"
    })
}

fn python_command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn resolve_piper_voice_path(settings: &TtsSettings) -> Result<PathBuf, String> {
    let voice = settings.piper_voice.trim();
    let voice = if voice.is_empty() {
        DEFAULT_PIPER_VOICE
    } else {
        voice
    };
    let direct = expand_user_path(voice);
    if direct
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        && direct.exists()
    {
        return Ok(direct);
    }

    let download_dir = &settings.piper_voices_dir;
    fs::create_dir_all(download_dir)
        .map_err(|error| format!("creating {} failed: {error}", download_dir.display()))?;
    let cached = download_dir.join(format!("{voice}.onnx"));
    let cached_meta = download_dir.join(format!("{voice}.onnx.json"));
    if cached.exists() && cached_meta.exists() {
        return Ok(cached);
    }

    let interpreter = resolve_python_interpreter();
    let mut command = Command::new(&interpreter);
    command
        .arg("-m")
        .arg("piper.download_voices")
        .arg(voice)
        .arg("--download-dir")
        .arg(download_dir);
    apply_pythonpath_override(&mut command, settings.piper_pythonpath.as_deref());
    let output = command.output().map_err(|error| {
        format!(
            "starting Piper voice download with {} failed: {error}",
            interpreter.display()
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'piper'")
            || stderr.contains("No module named \"piper\"")
        {
            return Err(
                "Piper provider selected but 'piper-tts' is not installed. Install it with: pip install piper-tts"
                    .to_string(),
            );
        }
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            format!("Piper voice download failed for '{voice}'")
        } else {
            format!("Piper voice download failed for '{voice}': {detail}")
        });
    }
    if !cached.exists() {
        return Err(format!(
            "Piper voice download completed but {} is missing",
            cached.display()
        ));
    }
    Ok(cached)
}

fn piper_synthesis_knobs(settings: &TtsSettings) -> serde_json::Map<String, Value> {
    let mut knobs = serde_json::Map::new();
    if let Some(value) = settings.piper_length_scale {
        knobs.insert("length_scale".to_string(), json!(value));
    }
    if let Some(value) = settings.piper_noise_scale {
        knobs.insert("noise_scale".to_string(), json!(value));
    }
    if let Some(value) = settings.piper_noise_w_scale {
        knobs.insert("noise_w_scale".to_string(), json!(value));
    }
    if let Some(value) = settings.piper_volume {
        knobs.insert("volume".to_string(), json!(value));
    }
    if let Some(value) = settings.piper_normalize_audio {
        knobs.insert("normalize_audio".to_string(), json!(value));
    }
    knobs
}

fn expand_user_path(raw: &str) -> PathBuf {
    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join(rest);
    }
    PathBuf::from(raw)
}

fn apply_pythonpath_override(command: &mut Command, override_value: Option<&str>) {
    let Some(override_value) = override_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return;
    };
    let merged = if let Some(existing) = std::env::var_os("PYTHONPATH") {
        let separator = if cfg!(windows) { ";" } else { ":" };
        format!(
            "{}{}{}",
            override_value,
            separator,
            existing.to_string_lossy()
        )
    } else {
        override_value.to_string()
    };
    command.env("PYTHONPATH", merged);
}

fn path_is_voice_compatible(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "ogg" | "opus"))
}

fn trim_decimal(value: f64) -> String {
    let mut rendered = format!("{value}");
    if rendered.contains('.') {
        while rendered.ends_with('0') {
            rendered.pop();
        }
        if rendered.ends_with('.') {
            rendered.pop();
        }
    }
    rendered
}

fn wrap_pcm_as_wav(
    pcm_bytes: &[u8],
    sample_rate: u32,
    channels: u16,
    sample_width: u16,
) -> Vec<u8> {
    let byte_rate = sample_rate * channels as u32 * sample_width as u32;
    let block_align = channels * sample_width;
    let data_size = pcm_bytes.len() as u32;
    let riff_size = 4 + 24 + 8 + data_size;

    let mut wav = Vec::with_capacity(44 + pcm_bytes.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&(sample_width * 8).to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm_bytes);
    wav
}

fn write_wav_or_transcode(wav_bytes: &[u8], output_path: &Path) -> Result<(), String> {
    let extension = output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| "mp3".to_string());
    if extension == "wav" {
        fs::write(output_path, wav_bytes)
            .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
        return ensure_audio_file(output_path);
    }

    let temp_wav = output_path.with_extension("wav.tmp");
    fs::write(&temp_wav, wav_bytes)
        .map_err(|error| format!("writing {} failed: {error}", temp_wav.display()))?;

    let ffmpeg = Command::new("ffmpeg").arg("-version").output();
    if ffmpeg.is_ok() {
        let mut command = Command::new("ffmpeg");
        command.arg("-i").arg(&temp_wav);
        if extension == "ogg" {
            command
                .arg("-acodec")
                .arg("libopus")
                .arg("-ac")
                .arg("1")
                .arg("-b:a")
                .arg("64k")
                .arg("-vbr")
                .arg("off");
        }
        let status = command
            .arg("-y")
            .arg("-loglevel")
            .arg("error")
            .arg(output_path)
            .status()
            .map_err(|error| format!("ffmpeg conversion failed: {error}"))?;
        let _ = fs::remove_file(&temp_wav);
        if !status.success() {
            return Err(format!(
                "ffmpeg conversion failed with code {}",
                status.code().unwrap_or(-1)
            ));
        }
        return ensure_audio_file(output_path);
    }

    fs::write(output_path, wav_bytes)
        .map_err(|error| format!("writing {} failed: {error}", output_path.display()))?;
    let _ = fs::remove_file(&temp_wav);
    ensure_audio_file(output_path)
}

fn extract_tts_api_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    value
                        .get("detail")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .or_else(|| {
                    value
                        .get("error")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
        })
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| body.to_string())
}

fn decode_hex_bytes(raw: &str) -> Result<Vec<u8>, String> {
    let trimmed = raw.trim();
    if trimmed.len() % 2 != 0 {
        return Err("MiniMax TTS returned invalid hex audio data".to_string());
    }
    let mut bytes = Vec::with_capacity(trimmed.len() / 2);
    let mut index = 0usize;
    while index < trimmed.len() {
        let chunk = &trimmed[index..index + 2];
        let value = u8::from_str_radix(chunk, 16)
            .map_err(|_| "MiniMax TTS returned invalid hex audio data".to_string())?;
        bytes.push(value);
        index += 2;
    }
    Ok(bytes)
}

fn tts_config_error(action: &'static str, error: HermesError) -> String {
    format!("TTS config {action} failed: {error}")
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::env;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use tempfile::TempDir;

    fn serve_audio_once(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = Arc::new(body);
        let counter = Arc::new(AtomicUsize::new(0));

        thread::spawn({
            let body = Arc::clone(&body);
            let counter = Arc::clone(&counter);
            move || {
                for stream in listener.incoming().take(1) {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let mut content_length = 0usize;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or_default() == 0 {
                            break;
                        }
                        let trimmed = line.trim_end();
                        if trimmed.is_empty() {
                            break;
                        }
                        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                            content_length = value.trim().parse::<usize>().unwrap_or_default();
                        }
                    }
                    let mut payload = vec![0_u8; content_length];
                    let _ = reader.read_exact(&mut payload);
                    counter.fetch_add(1, Ordering::SeqCst);
                    let http = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(http.as_bytes());
                    let _ = stream.write_all(&body);
                }
            }
        });

        format!("http://{}", addr)
    }

    fn command_copy_command() -> &'static str {
        "cp {input_path} {output_path}"
    }

    fn install_fake_piper_package(root: &Path, broken: bool) {
        let package = root.join("piper");
        fs::create_dir_all(&package).unwrap();
        if broken {
            fs::write(
                package.join("__init__.py"),
                "raise ModuleNotFoundError(\"No module named 'piper'\")\n",
            )
            .unwrap();
            return;
        }
        fs::write(
            package.join("__init__.py"),
            r#"
class SynthesisConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class PiperVoice:
    @classmethod
    def load(cls, model_path, use_cuda=False):
        instance = cls()
        instance.model_path = model_path
        instance.use_cuda = use_cuda
        return instance

    def synthesize_wav(self, text, wav_file, syn_config=None):
        payload = text.encode("utf-8")
        if len(payload) % 2:
            payload += b" "
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(payload)
"#,
        )
        .unwrap();
        fs::write(
            package.join("download_voices.py"),
            r#"
import pathlib
import sys

voice = sys.argv[1]
download_dir = pathlib.Path(sys.argv[sys.argv.index("--download-dir") + 1])
download_dir.mkdir(parents=True, exist_ok=True)
(download_dir / f"{voice}.onnx").write_bytes(b"model")
(download_dir / f"{voice}.onnx.json").write_text("{}")
"#,
        )
        .unwrap();
    }

    #[test]
    fn openai_tts_writes_audio_file_and_media_tag() {
        let temp = TempDir::new().unwrap();
        let base_url = serve_audio_once(b"fake-audio".to_vec());
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: openai\n  openai:\n    base_url: {}\n    api_key: test-key\n    model: test-model\n    voice: test-voice\n",
                base_url
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("openai.mp3");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello world",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("openai"));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert_eq!(file_path, output_path.display().to_string());
        assert_eq!(fs::read(file_path).unwrap(), b"fake-audio");
        assert_eq!(parsed["media_tag"], json!(format!("MEDIA:{file_path}")));
    }

    #[test]
    fn elevenlabs_tts_writes_audio_file_and_request_payload() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(
                request_line.trim_end(),
                "POST /v1/text-to-speech/voice-test?output_format=mp3_44100_128 HTTP/1.1"
            );
            let mut headers = Vec::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
                headers.push(trimmed);
            }
            assert!(
                headers
                    .iter()
                    .any(|line| line.eq_ignore_ascii_case("xi-api-key: eleven-test-key"))
            );
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["text"], json!("hello eleven"));
            assert_eq!(body["model_id"], json!("eleven_multilingual_v2"));

            let audio = b"eleven-audio";
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                audio.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.write_all(audio);
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: elevenlabs\n  elevenlabs:\n    base_url: http://{}/v1\n    api_key: eleven-test-key\n    voice_id: voice-test\n    model_id: eleven_multilingual_v2\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("eleven.mp3");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello eleven",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("elevenlabs"));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert_eq!(file_path, output_path.display().to_string());
        assert_eq!(fs::read(&output_path).unwrap(), b"eleven-audio");
        assert_eq!(parsed["media_tag"], json!(format!("MEDIA:{file_path}")));
        join.join().unwrap();
    }

    #[test]
    fn xai_tts_writes_audio_file_and_request_payload() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(request_line.trim_end(), "POST /v1/tts HTTP/1.1");
            let mut headers = Vec::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
                headers.push(trimmed);
            }
            assert!(
                headers
                    .iter()
                    .any(|line| line.eq_ignore_ascii_case("Authorization: Bearer xai-test-key"))
            );
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["text"], json!("hello xai"));
            assert_eq!(body["voice_id"], json!("custom-eve"));
            assert_eq!(body["language"], json!("fr"));
            let output_format = body["output_format"].as_object().unwrap();
            assert_eq!(output_format["codec"], json!("wav"));
            assert_eq!(output_format["sample_rate"], json!(16000));
            assert!(output_format.get("bit_rate").is_none());

            let audio = b"xai-audio";
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                audio.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.write_all(audio);
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: xai\n  xai:\n    base_url: http://{}/v1\n    api_key: xai-test-key\n    voice_id: custom-eve\n    language: fr\n    sample_rate: 16000\n    bit_rate: 96000\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("speech.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello xai",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("xai"));
        assert_eq!(
            parsed["file_path"],
            json!(output_path.display().to_string())
        );
        assert_eq!(fs::read(&output_path).unwrap(), b"xai-audio");
        join.join().unwrap();
    }

    #[test]
    fn gemini_tts_writes_wav_and_request_payload() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let pcm_bytes = vec![0_u8; 4800];
        let pcm_b64 = base64::engine::general_purpose::STANDARD.encode(&pcm_bytes);
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(
                request_line.trim_end(),
                "POST /v1beta/models/gemini-2.5-pro-preview-tts:generateContent?key=gemini-test-key HTTP/1.1"
            );
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(
                body["contents"][0]["parts"][0]["text"],
                json!("hello gemini")
            );
            assert_eq!(
                body["generationConfig"]["responseModalities"],
                json!(["AUDIO"])
            );
            assert_eq!(
                body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
                json!("Puck")
            );

            let response_body = json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": pcm_b64,
                            }
                        }]
                    }
                }]
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: gemini\n  gemini:\n    base_url: http://{}/v1beta\n    api_key: gemini-test-key\n    model: gemini-2.5-pro-preview-tts\n    voice: Puck\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("speech.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello gemini",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("gemini"));
        let wav = fs::read(&output_path).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[44..], pcm_bytes.as_slice());
        join.join().unwrap();
    }

    #[test]
    fn minimax_tts_writes_audio_file_and_request_payload() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(request_line.trim_end(), "POST /v1/text_to_speech HTTP/1.1");
            let mut headers = Vec::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
                headers.push(trimmed);
            }
            assert!(
                headers
                    .iter()
                    .any(|line| line.eq_ignore_ascii_case("Authorization: Bearer minimax-test-key"))
            );
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["model"], json!("speech-test"));
            assert_eq!(body["text"], json!("hello minimax"));
            assert_eq!(body["voice_id"], json!("voice-test"));

            let audio = b"minimax-audio";
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                audio.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.write_all(audio);
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: minimax\n  minimax:\n    base_url: http://{}/v1/text_to_speech\n    api_key: minimax-test-key\n    model: speech-test\n    voice_id: voice-test\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"hello minimax"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("minimax"));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert_eq!(fs::read(file_path).unwrap(), b"minimax-audio");
        join.join().unwrap();
    }

    #[test]
    fn minimax_tts_decodes_legacy_hex_audio_response() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(
                request_line.trim_end(),
                "POST /legacy/text_to_speech HTTP/1.1"
            );
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["text"], json!("legacy hello"));

            let response_body = json!({
                "base_resp": { "status_code": 0, "status_msg": "ok" },
                "data": { "audio": "68656c6c6f" }
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: minimax\n  minimax:\n    base_url: http://{}/legacy/text_to_speech\n    api_key: minimax-test-key\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("legacy.mp3");
        let result = handle_text_to_speech(
            &json!({
                "text":"legacy hello",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("minimax"));
        assert_eq!(fs::read(&output_path).unwrap(), b"hello");
        join.join().unwrap();
    }

    #[test]
    fn mistral_tts_writes_audio_file_and_request_payload() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(request_line.trim_end(), "POST /v1/audio/speech HTTP/1.1");
            let mut headers = Vec::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
                headers.push(trimmed);
            }
            assert!(
                headers
                    .iter()
                    .any(|line| line.eq_ignore_ascii_case("Authorization: Bearer mistral-test-key"))
            );
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["model"], json!("voxtral-test"));
            assert_eq!(body["input"], json!("hello mistral"));
            assert_eq!(body["voice_id"], json!("voice-mistral"));
            assert_eq!(body["response_format"], json!("wav"));

            let response_body = json!({
                "audio_data": base64::engine::general_purpose::STANDARD.encode(b"mistral-audio")
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: mistral\n  mistral:\n    base_url: http://{}/v1\n    api_key: mistral-test-key\n    model: voxtral-test\n    voice_id: voice-mistral\n",
                addr
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("speech.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello mistral",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("mistral"));
        assert_eq!(fs::read(&output_path).unwrap(), b"mistral-audio");
        join.join().unwrap();
    }

    #[test]
    fn mistral_tts_telegram_defaults_to_ogg_voice_media() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(request_line.trim_end(), "POST /v1/audio/speech HTTP/1.1");
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["response_format"], json!("opus"));

            let response_body = json!({
                "audio_data": base64::engine::general_purpose::STANDARD.encode(b"opus-audio")
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: mistral\n  mistral:\n    base_url: http://{}/v1\n    api_key: mistral-test-key\n",
                addr
            ),
        )
        .unwrap();
        unsafe {
            env::set_var("HERMES_SESSION_PLATFORM", "telegram");
        }
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"voice me"}), &runtime);
        unsafe {
            env::remove_var("HERMES_SESSION_PLATFORM");
        }
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("mistral"));
        assert_eq!(parsed["voice_compatible"], json!(true));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert!(file_path.ends_with(".ogg"));
        assert_eq!(
            parsed["media_tag"],
            json!(format!("[[audio_as_voice]]\nMEDIA:{file_path}"))
        );
        assert_eq!(fs::read(file_path).unwrap(), b"opus-audio");
        join.join().unwrap();
    }

    #[test]
    fn gemini_tts_telegram_defaults_to_ogg_voice_media() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let pcm_bytes = vec![1_u8; 4800];
        let pcm_b64 = base64::engine::general_purpose::STANDARD.encode(&pcm_bytes);
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(
                request_line.trim_end(),
                "POST /v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=gemini-test-key HTTP/1.1"
            );
            let response_body = json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inline_data": {
                                "data": pcm_b64,
                            }
                        }]
                    }
                }]
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: gemini\n  gemini:\n    base_url: http://{}/v1beta\n    api_key: gemini-test-key\n",
                addr
            ),
        )
        .unwrap();
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let script = bin_dir.join("ffmpeg");
        fs::write(
            &script,
            "#!/usr/bin/env bash\nin=''\nout=''\nwhile [ $# -gt 0 ]; do\n  case \"$1\" in\n    -i)\n      shift\n      in=\"$1\"\n      ;;\n    -acodec|-ac|-b:a|-vbr|-loglevel)\n      shift\n      ;;\n    -y)\n      ;;\n    *)\n      if [ \"${1#-}\" = \"$1\" ]; then out=\"$1\"; fi\n      ;;\n  esac\n  shift\ndone\ncp \"$in\" \"$out\"\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }

        let original_path = env::var("PATH").unwrap_or_default();
        unsafe {
            env::set_var("PATH", format!("{}:{}", bin_dir.display(), original_path));
            env::set_var("HERMES_SESSION_PLATFORM", "telegram");
        }
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"voice me gemini"}), &runtime);
        unsafe {
            env::set_var("PATH", original_path);
            env::remove_var("HERMES_SESSION_PLATFORM");
        }
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("gemini"));
        assert_eq!(parsed["voice_compatible"], json!(true));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert!(file_path.ends_with(".ogg"));
        assert_eq!(
            parsed["media_tag"],
            json!(format!("[[audio_as_voice]]\nMEDIA:{file_path}"))
        );
        let output = fs::read(file_path).unwrap();
        assert!(output.starts_with(b"RIFF") || output.starts_with(b"OggS"));
        join.join().unwrap();
    }

    #[test]
    fn elevenlabs_tts_telegram_defaults_to_ogg_voice_media() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(
                request_line.trim_end(),
                "POST /v1/text-to-speech/voice-telegram?output_format=opus_48000_64 HTTP/1.1"
            );
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut payload = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut payload);
            let body: Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(body["text"], json!("voice me eleven"));
            assert_eq!(body["model_id"], json!("eleven_multilingual_v2"));

            let audio = b"eleven-opus";
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: audio/ogg\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                audio.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.write_all(audio);
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: elevenlabs\n  elevenlabs:\n    base_url: http://{}/v1\n    api_key: eleven-test-key\n    voice_id: voice-telegram\n",
                addr
            ),
        )
        .unwrap();
        unsafe {
            env::set_var("HERMES_SESSION_PLATFORM", "telegram");
        }
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"voice me eleven"}), &runtime);
        unsafe {
            env::remove_var("HERMES_SESSION_PLATFORM");
        }
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("elevenlabs"));
        assert_eq!(parsed["voice_compatible"], json!(true));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert!(file_path.ends_with(".ogg"));
        assert_eq!(
            parsed["media_tag"],
            json!(format!("[[audio_as_voice]]\nMEDIA:{file_path}"))
        );
        assert_eq!(fs::read(file_path).unwrap(), b"eleven-opus");
        join.join().unwrap();
    }

    #[test]
    fn edge_tts_invokes_cli_when_configured() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "tts:\n  provider: edge\n  edge:\n    voice: custom-voice\n",
        )
        .unwrap();
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let script = bin_dir.join("edge-tts");
        fs::write(
            &script,
            "#!/usr/bin/env bash\nout=''\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"--write-media\" ]; then\n    shift\n    out=\"$1\"\n  fi\n  shift\ndone\nprintf 'edge-audio' > \"$out\"\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }

        let original_path = env::var("PATH").unwrap_or_default();
        unsafe {
            env::set_var("PATH", format!("{}:{}", bin_dir.display(), original_path));
        }
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"hello edge"}), &runtime);
        unsafe {
            env::set_var("PATH", original_path);
        }
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("edge"));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert_eq!(fs::read(file_path).unwrap(), b"edge-audio");
    }

    #[test]
    fn command_tts_provider_dispatches_end_to_end() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: py-copy\n  providers:\n    py-copy:\n      type: command\n      command: \"{}\"\n      output_format: mp3\n",
                command_copy_command()
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("clip.mp3");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello command provider",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("py-copy"));
        assert_eq!(parsed["voice_compatible"], json!(false));
        assert_eq!(
            fs::read_to_string(&output_path).unwrap(),
            "hello command provider"
        );
    }

    #[test]
    fn command_tts_provider_legacy_block_resolves_and_truncates() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: legacy-copy\n  speed: 1.25\n  legacy-copy:\n    type: command\n    command: \"{}\"\n    output_format: wav\n    max_text_length: 2\n",
                command_copy_command()
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("clip.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("legacy-copy"));
        assert_eq!(fs::read_to_string(&output_path).unwrap(), "he");
    }

    #[test]
    fn command_tts_provider_voice_opt_in_marks_ogg_as_voice_media() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: voice-copy\n  providers:\n    voice-copy:\n      type: command\n      command: \"{}\"\n      output_format: ogg\n      voice_compatible: true\n",
                command_copy_command()
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("voice.ogg");
        let result = handle_text_to_speech(
            &json!({
                "text":"voice me",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["voice_compatible"], json!(true));
        let file_path = parsed["file_path"].as_str().unwrap();
        assert!(file_path.ends_with(".ogg"));
        assert_eq!(
            parsed["media_tag"],
            json!(format!("[[audio_as_voice]]\nMEDIA:{file_path}"))
        );
    }

    #[test]
    fn command_tts_provider_explicit_extension_overrides_config_format() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: ext-copy\n  providers:\n    ext-copy:\n      type: command\n      command: \"{}\"\n      output_format: mp3\n",
                command_copy_command()
            ),
        )
        .unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let explicit = temp.path().join("clip.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello wav",
                "output_path": explicit.display().to_string(),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["file_path"], json!(explicit.display().to_string()));
        assert_eq!(fs::read_to_string(explicit).unwrap(), "hello wav");
    }

    #[test]
    fn render_command_tts_template_quotes_spacey_paths() {
        let rendered = render_command_tts_template(
            "tts --in {input_path} --out {output_path}",
            &[
                ("input_path", "/tmp/Jane Doe/input.txt".to_string()),
                ("output_path", "/tmp/out file.mp3".to_string()),
            ],
        );
        assert!(rendered.contains("'/tmp/Jane Doe/input.txt'"));
        assert!(rendered.contains("'/tmp/out file.mp3'"));
    }

    #[test]
    fn piper_tts_downloads_voice_and_writes_audio_file() {
        let temp = TempDir::new().unwrap();
        let pyroot = temp.path().join("pyroot");
        install_fake_piper_package(&pyroot, false);
        let voices_dir = temp.path().join("voices");
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: piper\n  piper:\n    voice: en_US-lessac-medium\n    voices_dir: {}\n    pythonpath: {}\n",
                voices_dir.display(),
                pyroot.display()
            ),
        )
        .unwrap();

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let output_path = temp.path().join("piper.wav");
        let result = handle_text_to_speech(
            &json!({
                "text":"hello piper",
                "output_path": output_path.display().to_string(),
            }),
            &runtime,
        );

        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("piper"));
        let wav = fs::read(&output_path).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert!(
            wav.windows(b"hello piper".len())
                .any(|chunk| chunk == b"hello piper")
        );
        assert!(voices_dir.join("en_US-lessac-medium.onnx").exists());
        assert!(voices_dir.join("en_US-lessac-medium.onnx.json").exists());
    }

    #[test]
    fn piper_tts_missing_package_returns_helpful_error() {
        let temp = TempDir::new().unwrap();
        let pyroot = temp.path().join("pyroot");
        install_fake_piper_package(&pyroot, true);
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "tts:\n  provider: piper\n  piper:\n    pythonpath: {}\n",
                pyroot.display()
            ),
        )
        .unwrap();

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = handle_text_to_speech(&json!({"text":"hello piper"}), &runtime);

        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["error"].as_str().unwrap().contains("piper-tts"));
    }
}
