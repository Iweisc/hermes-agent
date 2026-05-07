use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
const EDGE_MAX_TEXT_LENGTH: usize = 5_000;
const OPENAI_MAX_TEXT_LENGTH: usize = 4_096;
const MINIMAX_MAX_TEXT_LENGTH: usize = 10_000;
const GEMINI_MAX_TEXT_LENGTH: usize = 5_000;
const MISTRAL_MAX_TEXT_LENGTH: usize = 4_000;
const XAI_MAX_TEXT_LENGTH: usize = 15_000;
const GEMINI_TTS_SAMPLE_RATE: u32 = 24_000;
const GEMINI_TTS_CHANNELS: u16 = 1;
const GEMINI_TTS_SAMPLE_WIDTH: u16 = 2;

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
}

pub fn text_to_speech_schema() -> Value {
    json!({
        "name": "text_to_speech",
        "description": "Convert text to speech audio. Returns a MEDIA path tag that compatible delivery surfaces can send as native audio.",
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

    let voice_compatible = matches!(provider, "openai" | "mistral" | "elevenlabs" | "gemini")
        && output_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "ogg" | "opus"));

    let result = match provider {
        "edge" => synthesize_edge(&settings, &truncated, &output_path),
        "elevenlabs" => synthesize_elevenlabs(&settings, &truncated, &output_path),
        "openai" => synthesize_openai(&settings, &truncated, &output_path),
        "minimax" => synthesize_minimax(&settings, &truncated, &output_path),
        "gemini" => synthesize_gemini(&settings, &truncated, &output_path),
        "mistral" => synthesize_mistral(&settings, &truncated, &output_path),
        "xai" => synthesize_xai(&settings, &truncated, &output_path),
        other => Err(format!(
            "TTS provider '{other}' is not ported in the Rust runtime yet. Supported providers: edge, elevenlabs, openai, minimax, gemini, mistral, xai."
        )),
    };
    if let Err(error) = result {
        return tool_error(error);
    }

    let file_path = output_path.display().to_string();
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
        return runtime.resolve_path(path);
    }
    let platform = std::env::var("HERMES_SESSION_PLATFORM")
        .ok()
        .unwrap_or_default()
        .to_ascii_lowercase();
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
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0),
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

fn yaml_mapping_u32(root: Option<&YamlValue>, path: &[&str]) -> Option<u32> {
    let mut current = root?;
    for key in path {
        current = current
            .as_mapping()?
            .get(YamlValue::String((*key).to_string()))?;
    }
    match current {
        YamlValue::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        YamlValue::String(value) => value.trim().parse::<u32>().ok(),
        _ => None,
    }
}

fn provider_max_text_length(settings: &TtsSettings) -> usize {
    match settings.provider.as_str() {
        "edge" => EDGE_MAX_TEXT_LENGTH,
        "openai" => OPENAI_MAX_TEXT_LENGTH,
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
        _ => EDGE_MAX_TEXT_LENGTH,
    }
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
        let result = handle_text_to_speech(&json!({"text":"hello world"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["provider"], json!("openai"));
        let file_path = parsed["file_path"].as_str().unwrap();
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
}
