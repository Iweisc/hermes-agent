use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;
use chrono::Local;
use reqwest::blocking::{Client, RequestBuilder};
use serde_json::{Map, Value, json};
use serde_yaml::Value as YamlValue;

use crate::tools::{tool_error, tool_result};
use crate::{codex_cloudflare_headers, resolve_codex_access_token};

const DEFAULT_MODEL: &str = "fal-ai/flux-2/klein/9b";
const DEFAULT_ASPECT_RATIO: &str = "landscape";
const VALID_ASPECT_RATIOS: &[&str] = &["landscape", "square", "portrait"];
const DEFAULT_TIMEOUT_SECS: u64 = 180;
const POLL_INTERVAL_MS: u64 = 250;
const OPENAI_IMAGE_API_MODEL: &str = "gpt-image-2";
const OPENAI_IMAGE_DEFAULT_MODEL: &str = "gpt-image-2-medium";
const OPENAI_IMAGE_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_CODEX_IMAGE_DEFAULT_MODEL: &str = "gpt-image-2-medium";
const OPENAI_CODEX_CHAT_MODEL: &str = "gpt-5.4";
const OPENAI_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_CODEX_INSTRUCTIONS: &str = "You are an assistant that must fulfill image generation requests by using the image_generation tool when provided.";
const XAI_IMAGE_API_MODEL: &str = "grok-imagine-image";
const XAI_IMAGE_DEFAULT_MODEL: &str = "grok-imagine-image";
const XAI_IMAGE_BASE_URL: &str = "https://api.x.ai/v1";
const XAI_IMAGE_DEFAULT_RESOLUTION: &str = "1k";

#[derive(Clone, Copy)]
enum SizeStyle {
    ImageSizePreset,
    AspectRatio,
    GptLiteral,
}

#[derive(Clone, Copy)]
enum DefaultValue {
    Str(&'static str),
    Bool(bool),
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy)]
struct FalModelMeta {
    size_style: SizeStyle,
    landscape: &'static str,
    square: &'static str,
    portrait: &'static str,
    defaults: &'static [(&'static str, DefaultValue)],
    supports: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImageGenConfig {
    provider: Option<String>,
    model: Option<String>,
    use_gateway: bool,
    openai: OpenAiImageGenConfig,
    openai_codex: OpenAiCodexImageGenConfig,
    xai: XaiImageGenConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct OpenAiImageGenConfig {
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct OpenAiCodexImageGenConfig {
    model: Option<String>,
    base_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct XaiImageGenConfig {
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    resolution: Option<String>,
}

const KLEIN_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("num_inference_steps", DefaultValue::Int(4)),
    ("output_format", DefaultValue::Str("png")),
    ("enable_safety_checker", DefaultValue::Bool(false)),
];
const FLUX2_PRO_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("num_inference_steps", DefaultValue::Int(50)),
    ("guidance_scale", DefaultValue::Float(4.5)),
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
    ("enable_safety_checker", DefaultValue::Bool(false)),
    ("safety_tolerance", DefaultValue::Str("5")),
    ("sync_mode", DefaultValue::Bool(true)),
];
const Z_IMAGE_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("num_inference_steps", DefaultValue::Int(8)),
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
    ("enable_safety_checker", DefaultValue::Bool(false)),
    ("enable_prompt_expansion", DefaultValue::Bool(false)),
];
const NANO_BANANA_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
    ("safety_tolerance", DefaultValue::Str("5")),
    ("resolution", DefaultValue::Str("1K")),
];
const GPT_IMAGE_15_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("quality", DefaultValue::Str("medium")),
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
];
const GPT_IMAGE_2_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("quality", DefaultValue::Str("medium")),
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
];
const IDEOGRAM_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("rendering_speed", DefaultValue::Str("BALANCED")),
    ("expand_prompt", DefaultValue::Bool(true)),
    ("style", DefaultValue::Str("AUTO")),
];
const RECRAFT_DEFAULTS: &[(&str, DefaultValue)] =
    &[("enable_safety_checker", DefaultValue::Bool(false))];
const QWEN_DEFAULTS: &[(&str, DefaultValue)] = &[
    ("num_inference_steps", DefaultValue::Int(30)),
    ("guidance_scale", DefaultValue::Float(2.5)),
    ("num_images", DefaultValue::Int(1)),
    ("output_format", DefaultValue::Str("png")),
    ("acceleration", DefaultValue::Str("regular")),
];

const KLEIN_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "num_inference_steps",
    "seed",
    "output_format",
    "enable_safety_checker",
];
const FLUX2_PRO_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "num_inference_steps",
    "guidance_scale",
    "num_images",
    "output_format",
    "enable_safety_checker",
    "safety_tolerance",
    "sync_mode",
    "seed",
];
const Z_IMAGE_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "num_inference_steps",
    "num_images",
    "seed",
    "output_format",
    "enable_safety_checker",
    "enable_prompt_expansion",
];
const NANO_BANANA_SUPPORTS: &[&str] = &[
    "prompt",
    "aspect_ratio",
    "num_images",
    "output_format",
    "safety_tolerance",
    "seed",
    "sync_mode",
    "resolution",
    "enable_web_search",
    "limit_generations",
];
const GPT_IMAGE_15_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "quality",
    "num_images",
    "output_format",
    "background",
    "sync_mode",
];
const GPT_IMAGE_2_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "quality",
    "num_images",
    "output_format",
    "sync_mode",
];
const IDEOGRAM_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "rendering_speed",
    "expand_prompt",
    "style",
    "seed",
];
const RECRAFT_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "enable_safety_checker",
    "colors",
    "background_color",
];
const QWEN_SUPPORTS: &[&str] = &[
    "prompt",
    "image_size",
    "num_inference_steps",
    "guidance_scale",
    "num_images",
    "output_format",
    "acceleration",
    "seed",
    "sync_mode",
];

const KLEIN_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: KLEIN_DEFAULTS,
    supports: KLEIN_SUPPORTS,
};
const FLUX2_PRO_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: FLUX2_PRO_DEFAULTS,
    supports: FLUX2_PRO_SUPPORTS,
};
const Z_IMAGE_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: Z_IMAGE_DEFAULTS,
    supports: Z_IMAGE_SUPPORTS,
};
const NANO_BANANA_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::AspectRatio,
    landscape: "16:9",
    square: "1:1",
    portrait: "9:16",
    defaults: NANO_BANANA_DEFAULTS,
    supports: NANO_BANANA_SUPPORTS,
};
const GPT_IMAGE_15_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::GptLiteral,
    landscape: "1536x1024",
    square: "1024x1024",
    portrait: "1024x1536",
    defaults: GPT_IMAGE_15_DEFAULTS,
    supports: GPT_IMAGE_15_SUPPORTS,
};
const GPT_IMAGE_2_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_4_3",
    square: "square_hd",
    portrait: "portrait_4_3",
    defaults: GPT_IMAGE_2_DEFAULTS,
    supports: GPT_IMAGE_2_SUPPORTS,
};
const IDEOGRAM_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: IDEOGRAM_DEFAULTS,
    supports: IDEOGRAM_SUPPORTS,
};
const RECRAFT_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: RECRAFT_DEFAULTS,
    supports: RECRAFT_SUPPORTS,
};
const QWEN_META: FalModelMeta = FalModelMeta {
    size_style: SizeStyle::ImageSizePreset,
    landscape: "landscape_16_9",
    square: "square_hd",
    portrait: "portrait_16_9",
    defaults: QWEN_DEFAULTS,
    supports: QWEN_SUPPORTS,
};

pub fn image_generate_available() -> bool {
    let config = read_image_gen_config(&hermes_home());
    match config.provider.as_deref() {
        Some("openai") => openai_api_key(&config).is_some(),
        Some("openai-codex") => resolve_codex_access_token(&hermes_home()).is_ok(),
        Some("xai") => xai_api_key(&config).is_some(),
        Some(provider) if provider != "fal" => false,
        _ => has_direct_fal_key() || managed_gateway_ready(),
    }
}

pub fn image_generate_schema() -> Value {
    json!({
        "name": "image_generate",
        "description": "Generate high-quality images from text prompts. The underlying backend and model are user-configured and not selectable by the agent. Returns either a URL or an absolute file path in the `image` field.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt describing the desired image. Be detailed and descriptive."
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": VALID_ASPECT_RATIOS,
                    "description": "The aspect ratio of the generated image.",
                    "default": DEFAULT_ASPECT_RATIO
                }
            },
            "required": ["prompt"]
        }
    })
}

pub fn handle_image_generate(args: &Value, runtime: &crate::tools::ToolRuntime) -> String {
    let prompt = match required_non_empty_string(args, "prompt") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let aspect_ratio = args
        .get("aspect_ratio")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_ASPECT_RATIO);
    let config = read_image_gen_config(runtime.hermes_home());
    if let Some(provider) = config.provider.as_deref() {
        return match provider {
            "fal" => match generate_fal_image(&prompt, aspect_ratio, &config) {
                Ok(result) => tool_result(result),
                Err(error) => tool_result(json!({
                    "success": false,
                    "image": Value::Null,
                    "error": error,
                    "error_type": "api_error",
                })),
            },
            "openai" => tool_result(generate_openai_image(
                &prompt,
                aspect_ratio,
                &config,
                runtime.hermes_home(),
            )),
            "openai-codex" => tool_result(generate_openai_codex_image(
                &prompt,
                aspect_ratio,
                &config,
                runtime.hermes_home(),
            )),
            "xai" => tool_result(generate_xai_image(
                &prompt,
                aspect_ratio,
                &config,
                runtime.hermes_home(),
            )),
            other => tool_result(json!({
                "success": false,
                "image": Value::Null,
                "error": format!(
                    "image_gen.provider='{}' is set but that backend is not ported in the Rust runtime yet.",
                    other
                ),
                "error_type": "provider_not_registered",
            })),
        };
    }

    match generate_fal_image(&prompt, aspect_ratio, &config) {
        Ok(result) => tool_result(result),
        Err(error) => tool_result(json!({
            "success": false,
            "image": Value::Null,
            "error": error,
            "error_type": "api_error",
        })),
    }
}

fn generate_openai_image(
    prompt: &str,
    aspect_ratio: &str,
    config: &ImageGenConfig,
    hermes_home: &Path,
) -> Value {
    let prompt = prompt.trim();
    let aspect = normalize_aspect_ratio(aspect_ratio);
    if prompt.is_empty() {
        return provider_error_response(
            "Prompt is required and must be a non-empty string".to_string(),
            "invalid_argument",
            "openai",
            Some(OPENAI_IMAGE_DEFAULT_MODEL),
            prompt,
            &aspect,
        );
    }

    let Some(api_key) = openai_api_key(config) else {
        return provider_error_response(
            "OPENAI_API_KEY not set. Run `hermes tools` -> Image Generation -> OpenAI to configure, or `hermes setup` to add the key.".to_string(),
            "auth_required",
            "openai",
            None,
            prompt,
            &aspect,
        );
    };
    let (tier_id, quality) = resolve_openai_model(config);
    let size = openai_size_for_aspect(&aspect);
    let base_url = config
        .openai
        .base_url
        .clone()
        .or_else(|| env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| OPENAI_IMAGE_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();

    let client = match Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return provider_error_response(
                format!("OpenAI image generation failed: building HTTP client failed: {error}"),
                "api_error",
                "openai",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let response = match client
        .post(format!("{base_url}/images/generations"))
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": OPENAI_IMAGE_API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": quality,
        }))
        .send()
    {
        Ok(response) => response,
        Err(error) => {
            return provider_error_response(
                format!("OpenAI image generation failed: {error}"),
                "api_error",
                "openai",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let status = response.status();
    let body = match response.text() {
        Ok(body) => body,
        Err(error) => {
            return provider_error_response(
                format!("OpenAI image generation failed: reading response failed: {error}"),
                "api_error",
                "openai",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    if !status.is_success() {
        return provider_error_response(
            format!(
                "OpenAI image generation failed: {} {}",
                status.as_u16(),
                body
            ),
            "api_error",
            "openai",
            Some(&tier_id),
            prompt,
            &aspect,
        );
    }

    let parsed = match serde_json::from_str::<Value>(&body) {
        Ok(parsed) => parsed,
        Err(error) => {
            return provider_error_response(
                format!("OpenAI returned invalid JSON: {error}"),
                "invalid_response",
                "openai",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let Some(first) = parsed
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first())
    else {
        return provider_error_response(
            "OpenAI returned no image data".to_string(),
            "empty_response",
            "openai",
            Some(&tier_id),
            prompt,
            &aspect,
        );
    };

    let image_ref = if let Some(b64) = first.get("b64_json").and_then(Value::as_str) {
        match save_b64_image(hermes_home, b64, &format!("openai_{tier_id}")) {
            Ok(path) => path,
            Err(error) => {
                return provider_error_response(
                    format!("Could not save image to cache: {error}"),
                    "io_error",
                    "openai",
                    Some(&tier_id),
                    prompt,
                    &aspect,
                );
            }
        }
    } else if let Some(url) = first.get("url").and_then(Value::as_str) {
        url.to_string()
    } else {
        return provider_error_response(
            "OpenAI response contained neither b64_json nor URL".to_string(),
            "empty_response",
            "openai",
            Some(&tier_id),
            prompt,
            &aspect,
        );
    };

    let mut extra = Map::new();
    extra.insert("size".to_string(), Value::String(size.to_string()));
    extra.insert("quality".to_string(), Value::String(quality.to_string()));
    if let Some(revised_prompt) = first
        .get("revised_prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        extra.insert(
            "revised_prompt".to_string(),
            Value::String(revised_prompt.to_string()),
        );
    }
    provider_success_response(&image_ref, &tier_id, prompt, &aspect, "openai", Some(extra))
}

fn generate_xai_image(
    prompt: &str,
    aspect_ratio: &str,
    config: &ImageGenConfig,
    hermes_home: &Path,
) -> Value {
    let prompt = prompt.trim();
    let aspect = normalize_aspect_ratio(aspect_ratio);
    if prompt.is_empty() {
        return provider_error_response(
            "Prompt is required and must be a non-empty string".to_string(),
            "invalid_argument",
            "xai",
            Some(XAI_IMAGE_DEFAULT_MODEL),
            prompt,
            &aspect,
        );
    }

    let Some(api_key) = xai_api_key(config) else {
        return provider_error_response(
            "XAI_API_KEY not set. Get one at https://console.x.ai/".to_string(),
            "missing_api_key",
            "xai",
            None,
            prompt,
            &aspect,
        );
    };
    let model_id = resolve_xai_model(config);
    let xai_ar = xai_aspect_ratio_for_aspect(&aspect);
    let (resolution_key, resolution_value) = resolve_xai_resolution(config);
    let base_url = config
        .xai
        .base_url
        .clone()
        .or_else(|| env::var("XAI_BASE_URL").ok())
        .unwrap_or_else(|| XAI_IMAGE_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();

    let client = match Client::builder().timeout(Duration::from_secs(120)).build() {
        Ok(client) => client,
        Err(error) => {
            return provider_error_response(
                format!("xAI image generation failed: building HTTP client failed: {error}"),
                "api_error",
                "xai",
                Some(&model_id),
                prompt,
                &aspect,
            );
        }
    };
    let response = match client
        .post(format!("{base_url}/images/generations"))
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("User-Agent", "hermes-agent/1.0")
        .json(&json!({
            "model": XAI_IMAGE_API_MODEL,
            "prompt": prompt,
            "aspect_ratio": xai_ar,
            "resolution": resolution_value,
        }))
        .send()
    {
        Ok(response) => response,
        Err(error) => {
            return provider_error_response(
                format!("xAI image generation failed: {error}"),
                "api_error",
                "xai",
                Some(&model_id),
                prompt,
                &aspect,
            );
        }
    };
    let status = response.status();
    let body = match response.text() {
        Ok(body) => body,
        Err(error) => {
            return provider_error_response(
                format!("xAI image generation failed: reading response failed: {error}"),
                "api_error",
                "xai",
                Some(&model_id),
                prompt,
                &aspect,
            );
        }
    };
    if !status.is_success() {
        return provider_error_response(
            format!(
                "xAI image generation failed ({}): {}",
                status.as_u16(),
                extract_image_error_message(&body)
            ),
            "api_error",
            "xai",
            Some(&model_id),
            prompt,
            &aspect,
        );
    }

    let parsed = match serde_json::from_str::<Value>(&body) {
        Ok(parsed) => parsed,
        Err(error) => {
            return provider_error_response(
                format!("xAI returned invalid JSON: {error}"),
                "invalid_response",
                "xai",
                Some(&model_id),
                prompt,
                &aspect,
            );
        }
    };
    let Some(first) = parsed
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first())
    else {
        return provider_error_response(
            "xAI returned no image data".to_string(),
            "empty_response",
            "xai",
            Some(&model_id),
            prompt,
            &aspect,
        );
    };

    let image_ref = if let Some(b64) = first.get("b64_json").and_then(Value::as_str) {
        match save_b64_image(hermes_home, b64, &format!("xai_{model_id}")) {
            Ok(path) => path,
            Err(error) => {
                return provider_error_response(
                    format!("Could not save image to cache: {error}"),
                    "io_error",
                    "xai",
                    Some(&model_id),
                    prompt,
                    &aspect,
                );
            }
        }
    } else if let Some(url) = first.get("url").and_then(Value::as_str) {
        url.to_string()
    } else {
        return provider_error_response(
            "xAI response contained neither b64_json nor URL".to_string(),
            "empty_response",
            "xai",
            Some(&model_id),
            prompt,
            &aspect,
        );
    };

    let mut extra = Map::new();
    extra.insert(
        "resolution".to_string(),
        Value::String(resolution_value.to_string()),
    );
    extra.insert(
        "resolution_key".to_string(),
        Value::String(resolution_key.to_string()),
    );
    provider_success_response(&image_ref, &model_id, prompt, &aspect, "xai", Some(extra))
}

fn generate_openai_codex_image(
    prompt: &str,
    aspect_ratio: &str,
    config: &ImageGenConfig,
    hermes_home: &Path,
) -> Value {
    let prompt = prompt.trim();
    let aspect = normalize_aspect_ratio(aspect_ratio);
    if prompt.is_empty() {
        return provider_error_response(
            "Prompt is required and must be a non-empty string".to_string(),
            "invalid_argument",
            "openai-codex",
            Some(OPENAI_CODEX_IMAGE_DEFAULT_MODEL),
            prompt,
            &aspect,
        );
    }

    let access_token = match resolve_codex_access_token(hermes_home) {
        Ok(token) => token,
        Err(_) => {
            return provider_error_response(
                "No Codex/ChatGPT OAuth credentials available. Run `hermes auth codex` to sign in."
                    .to_string(),
                "auth_required",
                "openai-codex",
                None,
                prompt,
                &aspect,
            );
        }
    };
    let (tier_id, quality) = resolve_openai_codex_model(config);
    let size = openai_size_for_aspect(&aspect);
    let base_url = config
        .openai_codex
        .base_url
        .clone()
        .unwrap_or_else(|| OPENAI_CODEX_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();

    let client = match Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return provider_error_response(
                format!("Could not initialize Codex image client: {error}"),
                "auth_required",
                "openai-codex",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let mut request = client
        .post(format!("{base_url}/responses"))
        .bearer_auth(&access_token)
        .header("Content-Type", "application/json");
    for (name, value) in codex_cloudflare_headers(&access_token) {
        request = request.header(name, value);
    }
    let response = match request
        .json(&json!({
            "model": OPENAI_CODEX_CHAT_MODEL,
            "store": false,
            "instructions": OPENAI_CODEX_INSTRUCTIONS,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": prompt,
                }],
            }],
            "tools": [{
                "type": "image_generation",
                "model": OPENAI_IMAGE_API_MODEL,
                "size": size,
                "quality": quality,
                "output_format": "png",
                "background": "opaque",
                "partial_images": 1,
            }],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "image_generation"}],
            }
        }))
        .send()
    {
        Ok(response) => response,
        Err(error) => {
            return provider_error_response(
                format!("OpenAI image generation via Codex auth failed: {error}"),
                "api_error",
                "openai-codex",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let status = response.status();
    let body = match response.text() {
        Ok(body) => body,
        Err(error) => {
            return provider_error_response(
                format!(
                    "OpenAI image generation via Codex auth failed: reading response failed: {error}"
                ),
                "api_error",
                "openai-codex",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    if !status.is_success() {
        return provider_error_response(
            format!(
                "OpenAI image generation via Codex auth failed: {} {}",
                status.as_u16(),
                body
            ),
            "api_error",
            "openai-codex",
            Some(&tier_id),
            prompt,
            &aspect,
        );
    }
    let parsed = match serde_json::from_str::<Value>(&body) {
        Ok(parsed) => parsed,
        Err(error) => {
            return provider_error_response(
                format!("Codex image generation returned invalid JSON: {error}"),
                "invalid_response",
                "openai-codex",
                Some(&tier_id),
                prompt,
                &aspect,
            );
        }
    };
    let Some(image_b64) = parsed
        .get("output")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                if item.get("type").and_then(Value::as_str) == Some("image_generation_call") {
                    item.get("result")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToOwned::to_owned)
                } else {
                    None
                }
            })
        })
    else {
        return provider_error_response(
            "Codex response contained no image_generation_call result".to_string(),
            "empty_response",
            "openai-codex",
            Some(&tier_id),
            prompt,
            &aspect,
        );
    };
    let image_ref =
        match save_b64_image(hermes_home, &image_b64, &format!("openai_codex_{tier_id}")) {
            Ok(path) => path,
            Err(error) => {
                return provider_error_response(
                    format!("Could not save image to cache: {error}"),
                    "io_error",
                    "openai-codex",
                    Some(&tier_id),
                    prompt,
                    &aspect,
                );
            }
        };

    let mut extra = Map::new();
    extra.insert("size".to_string(), Value::String(size.to_string()));
    extra.insert("quality".to_string(), Value::String(quality.to_string()));
    provider_success_response(
        &image_ref,
        &tier_id,
        prompt,
        &aspect,
        "openai-codex",
        Some(extra),
    )
}

fn generate_fal_image(
    prompt: &str,
    aspect_ratio: &str,
    config: &ImageGenConfig,
) -> Result<Value, String> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err("Prompt is required and must be a non-empty string".to_string());
    }
    let aspect_ratio = normalize_aspect_ratio(aspect_ratio);
    let (model_id, _) = resolve_fal_model(config);
    let payload = build_fal_payload(&model_id, prompt, &aspect_ratio, None, None)?;
    let (base_url, api_key) = resolve_fal_backend(config)?;
    let result = submit_fal_request(&base_url, &api_key, &model_id, &payload)?;

    let first_image = result
        .get("images")
        .and_then(Value::as_array)
        .and_then(|images| images.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "Invalid response from FAL API: no images returned".to_string())?;
    let image_ref = first_image
        .get("url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Invalid response from FAL API: image URL missing".to_string())?;

    Ok(json!({
        "success": true,
        "image": image_ref,
        "model": model_id,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": "fal",
    }))
}

fn resolve_fal_backend(config: &ImageGenConfig) -> Result<(String, String), String> {
    let direct_key = env::var("FAL_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if let Some(key) = direct_key
        && !config.use_gateway
    {
        return Ok(("https://queue.fal.run".to_string(), key));
    }

    if let Some((gateway_url, gateway_token)) = managed_gateway_config() {
        return Ok((gateway_url, gateway_token));
    }

    if let Some(key) = env::var("FAL_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(("https://queue.fal.run".to_string(), key));
    }

    Err("FAL_KEY environment variable not set and managed FAL gateway is unavailable".to_string())
}

fn submit_fal_request(
    base_url: &str,
    api_key: &str,
    model_id: &str,
    payload: &Value,
) -> Result<Value, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
        .map_err(|error| format!("building FAL client failed: {error}"))?;
    let submit_url = format!("{}/{}", base_url.trim_end_matches('/'), model_id);
    let initial = send_json(
        authorized_request(
            client
                .post(&submit_url)
                .header("x-idempotency-key", format!("{:x}", unix_ts_nanos()))
                .json(payload),
            api_key,
        ),
        "submitting FAL request",
    )?;

    let request_id = initial
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "FAL response missing request_id".to_string())?;
    let status_url = initial
        .get("status_url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("{}/{}/status", submit_url, request_id));
    let response_url = initial
        .get("response_url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("{}/{}/response", submit_url, request_id));

    let deadline = Instant::now() + Duration::from_secs(DEFAULT_TIMEOUT_SECS);
    loop {
        if Instant::now() > deadline {
            return Err(format!(
                "FAL image generation timed out after {}s",
                DEFAULT_TIMEOUT_SECS
            ));
        }
        let status = send_json(
            authorized_request(client.get(&status_url), api_key),
            "polling FAL request status",
        )?;
        let state = status
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_uppercase();
        match state.as_str() {
            "COMPLETED" => {
                return send_json(
                    authorized_request(client.get(&response_url), api_key),
                    "fetching FAL response",
                );
            }
            "IN_QUEUE" | "IN_PROGRESS" | "QUEUED" => {
                thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
            }
            "FAILED" | "CANCELED" | "CANCELLED" => {
                let detail = status
                    .get("error")
                    .and_then(Value::as_str)
                    .or_else(|| status.get("detail").and_then(Value::as_str))
                    .unwrap_or("request failed");
                return Err(format!("FAL request {request_id} {state}: {detail}"));
            }
            _ => {
                thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
            }
        }
    }
}

fn authorized_request(request: RequestBuilder, api_key: &str) -> RequestBuilder {
    request.header("Authorization", format!("Key {api_key}"))
}

fn send_json(request: RequestBuilder, action: &str) -> Result<Value, String> {
    let response = request
        .send()
        .map_err(|error| format!("{action} failed: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|error| format!("reading FAL response failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "{action} failed with HTTP {}: {}",
            status.as_u16(),
            body
        ));
    }
    serde_json::from_str::<Value>(&body)
        .map_err(|error| format!("decoding FAL JSON failed: {error}"))
}

fn resolve_fal_model(config: &ImageGenConfig) -> (String, FalModelMeta) {
    let env_model = env::var("FAL_IMAGE_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let requested = config.model.as_deref().or(env_model.as_deref());
    if let Some(model_id) = requested
        && let Some(meta) = model_meta(model_id)
    {
        return (model_id.to_string(), meta);
    }
    (
        DEFAULT_MODEL.to_string(),
        model_meta(DEFAULT_MODEL).expect("default model exists"),
    )
}

fn build_fal_payload(
    model_id: &str,
    prompt: &str,
    aspect_ratio: &str,
    seed: Option<i64>,
    overrides: Option<&Map<String, Value>>,
) -> Result<Value, String> {
    let Some(meta) = model_meta(model_id) else {
        return Err(format!("Unknown FAL model '{model_id}'"));
    };
    let mut object = Map::new();
    for (key, value) in meta.defaults {
        object.insert((*key).to_string(), default_value_to_json(*value));
    }
    object.insert(
        "prompt".to_string(),
        Value::String(prompt.trim().to_string()),
    );
    let size_value = match normalize_aspect_ratio(aspect_ratio).as_str() {
        "square" => meta.square,
        "portrait" => meta.portrait,
        _ => meta.landscape,
    };
    match meta.size_style {
        SizeStyle::ImageSizePreset | SizeStyle::GptLiteral => {
            object.insert(
                "image_size".to_string(),
                Value::String(size_value.to_string()),
            );
        }
        SizeStyle::AspectRatio => {
            object.insert(
                "aspect_ratio".to_string(),
                Value::String(size_value.to_string()),
            );
        }
    }
    if let Some(seed) = seed {
        object.insert("seed".to_string(), Value::from(seed));
    }
    if let Some(overrides) = overrides {
        for (key, value) in overrides {
            if !value.is_null() {
                object.insert(key.clone(), value.clone());
            }
        }
    }

    let supports = meta.supports.iter().copied().collect::<BTreeSet<_>>();
    object.retain(|key, _| supports.contains(key.as_str()));
    Ok(Value::Object(object))
}

fn model_meta(model_id: &str) -> Option<FalModelMeta> {
    match model_id {
        "fal-ai/flux-2/klein/9b" => Some(KLEIN_META),
        "fal-ai/flux-2-pro" => Some(FLUX2_PRO_META),
        "fal-ai/z-image/turbo" => Some(Z_IMAGE_META),
        "fal-ai/nano-banana-pro" => Some(NANO_BANANA_META),
        "fal-ai/gpt-image-1.5" => Some(GPT_IMAGE_15_META),
        "fal-ai/gpt-image-2" => Some(GPT_IMAGE_2_META),
        "fal-ai/ideogram/v3" => Some(IDEOGRAM_META),
        "fal-ai/recraft/v4/pro/text-to-image" => Some(RECRAFT_META),
        "fal-ai/qwen-image" => Some(QWEN_META),
        _ => None,
    }
}

fn default_value_to_json(value: DefaultValue) -> Value {
    match value {
        DefaultValue::Str(value) => Value::String(value.to_string()),
        DefaultValue::Bool(value) => Value::Bool(value),
        DefaultValue::Int(value) => Value::from(value),
        DefaultValue::Float(value) => json!(value),
    }
}

fn normalize_aspect_ratio(value: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase();
    if VALID_ASPECT_RATIOS.contains(&normalized.as_str()) {
        normalized
    } else {
        DEFAULT_ASPECT_RATIO.to_string()
    }
}

fn openai_api_key(config: &ImageGenConfig) -> Option<String> {
    env::var("OPENAI_API_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| config.openai.api_key.clone())
}

fn xai_api_key(config: &ImageGenConfig) -> Option<String> {
    env::var("XAI_API_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| config.xai.api_key.clone())
}

fn resolve_openai_model(config: &ImageGenConfig) -> (String, &'static str) {
    let candidate = env::var("OPENAI_IMAGE_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| config.openai.model.clone())
        .or_else(|| config.model.clone());
    match candidate.as_deref() {
        Some("gpt-image-2-low") => ("gpt-image-2-low".to_string(), "low"),
        Some("gpt-image-2-high") => ("gpt-image-2-high".to_string(), "high"),
        Some("gpt-image-2-medium") => ("gpt-image-2-medium".to_string(), "medium"),
        _ => (OPENAI_IMAGE_DEFAULT_MODEL.to_string(), "medium"),
    }
}

fn resolve_openai_codex_model(config: &ImageGenConfig) -> (String, &'static str) {
    let candidate = env::var("OPENAI_IMAGE_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| config.openai_codex.model.clone())
        .or_else(|| config.model.clone());
    match candidate.as_deref() {
        Some("gpt-image-2-low") => ("gpt-image-2-low".to_string(), "low"),
        Some("gpt-image-2-high") => ("gpt-image-2-high".to_string(), "high"),
        Some("gpt-image-2-medium") => ("gpt-image-2-medium".to_string(), "medium"),
        _ => (OPENAI_CODEX_IMAGE_DEFAULT_MODEL.to_string(), "medium"),
    }
}

fn resolve_xai_model(config: &ImageGenConfig) -> String {
    let candidate = env::var("XAI_IMAGE_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| config.xai.model.clone());
    match candidate.as_deref() {
        Some(XAI_IMAGE_API_MODEL) => XAI_IMAGE_API_MODEL.to_string(),
        _ => XAI_IMAGE_DEFAULT_MODEL.to_string(),
    }
}

fn openai_size_for_aspect(aspect_ratio: &str) -> &'static str {
    match aspect_ratio {
        "landscape" => "1536x1024",
        "portrait" => "1024x1536",
        _ => "1024x1024",
    }
}

fn xai_aspect_ratio_for_aspect(aspect_ratio: &str) -> &'static str {
    match aspect_ratio {
        "landscape" => "16:9",
        "portrait" => "9:16",
        _ => "1:1",
    }
}

fn resolve_xai_resolution(config: &ImageGenConfig) -> (&'static str, &'static str) {
    match config.xai.resolution.as_deref() {
        Some("2k") => ("2k", "2048"),
        _ => (XAI_IMAGE_DEFAULT_RESOLUTION, "1024"),
    }
}

fn provider_success_response(
    image: &str,
    model: &str,
    prompt: &str,
    aspect_ratio: &str,
    provider: &str,
    extra: Option<Map<String, Value>>,
) -> Value {
    let mut payload = Map::new();
    payload.insert("success".to_string(), Value::Bool(true));
    payload.insert("image".to_string(), Value::String(image.to_string()));
    payload.insert("model".to_string(), Value::String(model.to_string()));
    payload.insert("prompt".to_string(), Value::String(prompt.to_string()));
    payload.insert(
        "aspect_ratio".to_string(),
        Value::String(aspect_ratio.to_string()),
    );
    payload.insert("provider".to_string(), Value::String(provider.to_string()));
    if let Some(extra) = extra {
        for (key, value) in extra {
            payload.entry(key).or_insert(value);
        }
    }
    Value::Object(payload)
}

fn provider_error_response(
    error: String,
    error_type: &str,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    aspect_ratio: &str,
) -> Value {
    json!({
        "success": false,
        "image": Value::Null,
        "error": error,
        "error_type": error_type,
        "model": model.unwrap_or_default(),
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
    })
}

fn save_b64_image(hermes_home: &Path, b64_data: &str, prefix: &str) -> Result<String, String> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64_data.trim())
        .map_err(|error| format!("invalid base64 image data: {error}"))?;
    let dir = hermes_home.join("cache/images");
    fs::create_dir_all(&dir)
        .map_err(|error| format!("creating {} failed: {error}", dir.display()))?;
    let path = dir.join(format!(
        "{}_{}_{:x}.png",
        prefix,
        Local::now().format("%Y%m%d_%H%M%S"),
        unix_ts_nanos(),
    ));
    fs::write(&path, raw).map_err(|error| format!("writing {} failed: {error}", path.display()))?;
    Ok(path.display().to_string())
}

fn extract_image_error_message(body: &str) -> String {
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

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn read_image_gen_config(hermes_home: &Path) -> ImageGenConfig {
    let path = config_path(hermes_home);
    let Ok(contents) = fs::read_to_string(path) else {
        return ImageGenConfig {
            provider: None,
            model: None,
            use_gateway: false,
            openai: OpenAiImageGenConfig::default(),
            openai_codex: OpenAiCodexImageGenConfig::default(),
            xai: XaiImageGenConfig::default(),
        };
    };
    let Ok(root) = serde_yaml::from_str::<YamlValue>(&contents) else {
        return ImageGenConfig {
            provider: None,
            model: None,
            use_gateway: false,
            openai: OpenAiImageGenConfig::default(),
            openai_codex: OpenAiCodexImageGenConfig::default(),
            xai: XaiImageGenConfig::default(),
        };
    };
    let image_gen = root
        .as_mapping()
        .and_then(|mapping| mapping.get(YamlValue::String("image_gen".to_string())))
        .and_then(YamlValue::as_mapping);
    let openai = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("openai".to_string())))
        .and_then(YamlValue::as_mapping);
    let openai_codex = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("openai-codex".to_string())))
        .and_then(YamlValue::as_mapping);
    let xai = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("xai".to_string())))
        .and_then(YamlValue::as_mapping);

    let provider = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("provider".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let model = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("model".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let use_gateway = image_gen
        .and_then(|mapping| mapping.get(YamlValue::String("use_gateway".to_string())))
        .and_then(YamlValue::as_bool)
        .unwrap_or(false);
    let openai_model = openai
        .and_then(|mapping| mapping.get(YamlValue::String("model".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let openai_base_url = openai
        .and_then(|mapping| mapping.get(YamlValue::String("base_url".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let openai_api_key = openai
        .and_then(|mapping| mapping.get(YamlValue::String("api_key".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let openai_codex_model = openai_codex
        .and_then(|mapping| mapping.get(YamlValue::String("model".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let openai_codex_base_url = openai_codex
        .and_then(|mapping| mapping.get(YamlValue::String("base_url".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let xai_model = xai
        .and_then(|mapping| mapping.get(YamlValue::String("model".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let xai_base_url = xai
        .and_then(|mapping| mapping.get(YamlValue::String("base_url".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let xai_api_key = xai
        .and_then(|mapping| mapping.get(YamlValue::String("api_key".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let xai_resolution = xai
        .and_then(|mapping| mapping.get(YamlValue::String("resolution".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| matches!(*value, "1k" | "2k"))
        .map(ToOwned::to_owned);

    ImageGenConfig {
        provider,
        model,
        use_gateway,
        openai: OpenAiImageGenConfig {
            model: openai_model,
            base_url: openai_base_url,
            api_key: openai_api_key,
        },
        openai_codex: OpenAiCodexImageGenConfig {
            model: openai_codex_model,
            base_url: openai_codex_base_url,
        },
        xai: XaiImageGenConfig {
            model: xai_model,
            base_url: xai_base_url,
            api_key: xai_api_key,
            resolution: xai_resolution,
        },
    }
}

fn config_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join("config.yaml")
}

fn hermes_home() -> PathBuf {
    env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

fn has_direct_fal_key() -> bool {
    env::var("FAL_KEY")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn managed_gateway_ready() -> bool {
    managed_gateway_config().is_some()
}

fn managed_gateway_config() -> Option<(String, String)> {
    let token = env::var("TOOL_GATEWAY_USER_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;
    let url = env::var("FAL_QUEUE_GATEWAY_URL")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            let domain = env::var("TOOL_GATEWAY_DOMAIN").ok()?;
            let domain = domain.trim().trim_matches('/').to_string();
            if domain.is_empty() {
                return None;
            }
            let scheme = env::var("TOOL_GATEWAY_SCHEME")
                .ok()
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| matches!(value.as_str(), "http" | "https"))
                .unwrap_or_else(|| "https".to_string());
            Some(format!("{scheme}://fal-queue-gateway.{domain}"))
        })?;
    Some((url, token))
}

fn unix_ts_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use super::*;

    use tempfile::TempDir;

    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    #[test]
    fn image_size_preset_family_uses_expected_sizes() {
        let payload =
            build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "landscape", None, None).unwrap();
        assert_eq!(payload["image_size"], "landscape_16_9");
        assert!(payload.get("aspect_ratio").is_none());
    }

    #[test]
    fn aspect_ratio_family_uses_expected_sizes() {
        let payload =
            build_fal_payload("fal-ai/nano-banana-pro", "hello", "portrait", None, None).unwrap();
        assert_eq!(payload["aspect_ratio"], "9:16");
        assert!(payload.get("image_size").is_none());
    }

    #[test]
    fn gpt_literal_family_uses_expected_sizes() {
        let payload =
            build_fal_payload("fal-ai/gpt-image-1.5", "hello", "square", None, None).unwrap();
        assert_eq!(payload["image_size"], "1024x1024");
    }

    #[test]
    fn supports_filter_strips_unsupported_keys() {
        let mut overrides = Map::new();
        overrides.insert("guidance_scale".to_string(), json!(7.5));
        overrides.insert("num_inference_steps".to_string(), json!(50));
        overrides.insert("openai_api_key".to_string(), json!("sk-123"));
        let payload = build_fal_payload(
            "fal-ai/gpt-image-2",
            "hi",
            "square",
            Some(42),
            Some(&overrides),
        )
        .unwrap();
        assert_eq!(payload["quality"], "medium");
        assert!(payload.get("guidance_scale").is_none());
        assert!(payload.get("num_inference_steps").is_none());
        assert!(payload.get("openai_api_key").is_none());
        assert!(payload.get("seed").is_none());
    }

    #[test]
    fn resolve_fal_model_reads_config_and_env_fallbacks() {
        let temp = TempDir::new().unwrap();
        let previous = env::var_os("HERMES_HOME");
        set_env_var("HERMES_HOME", temp.path());
        fs::write(
            temp.path().join("config.yaml"),
            "image_gen:\n  model: fal-ai/flux-2-pro\n",
        )
        .unwrap();

        let config = read_image_gen_config(temp.path());
        let (model, _) = resolve_fal_model(&config);
        assert_eq!(model, "fal-ai/flux-2-pro");

        fs::write(
            temp.path().join("config.yaml"),
            "image_gen:\n  model: fal-ai/nonexistent\n",
        )
        .unwrap();
        set_env_var("FAL_IMAGE_MODEL", "fal-ai/z-image/turbo");
        let config = read_image_gen_config(temp.path());
        let (model, _) = resolve_fal_model(&config);
        assert_eq!(model, DEFAULT_MODEL);

        remove_env_var("FAL_IMAGE_MODEL");
        match previous {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
    }

    fn serve_fal_queue() -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let handle = std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                loop {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .unwrap()
                    + 4;
                let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let mut body_bytes = request[header_end..].to_vec();
                while body_bytes.len() < content_length {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    body_bytes.extend_from_slice(&buffer[..read]);
                }
                captured_clone.lock().unwrap().push(format!(
                    "{}\n{}",
                    headers.lines().next().unwrap_or_default(),
                    String::from_utf8_lossy(&body_bytes)
                ));
                let first_line = headers.lines().next().unwrap_or_default().to_string();
                let response_body = if first_line.starts_with("POST /fal-ai/flux-2/klein/9b ") {
                    json!({
                        "request_id": "req-123",
                        "response_url": format!("http://{}/requests/req-123/response", addr),
                        "status_url": format!("http://{}/requests/req-123/status", addr),
                        "cancel_url": format!("http://{}/requests/req-123/cancel", addr),
                        "queue_position": 1,
                    })
                    .to_string()
                } else if first_line.starts_with("GET /requests/req-123/status ") {
                    json!({"status": "COMPLETED"}).to_string()
                } else {
                    json!({
                        "images": [{
                            "url": "https://v3.fal.media/files/demo/test.png",
                            "width": 1024,
                            "height": 1024
                        }]
                    })
                    .to_string()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (format!("http://{addr}"), captured, handle)
    }

    #[test]
    fn image_generate_runs_against_managed_gateway_contract() {
        let temp = TempDir::new().unwrap();
        let previous_home = env::var_os("HERMES_HOME");
        let previous_key = env::var_os("FAL_KEY");
        let previous_gateway = env::var_os("FAL_QUEUE_GATEWAY_URL");
        let previous_token = env::var_os("TOOL_GATEWAY_USER_TOKEN");
        let (gateway_url, captured, server) = serve_fal_queue();

        set_env_var("HERMES_HOME", temp.path());
        remove_env_var("FAL_KEY");
        set_env_var("FAL_QUEUE_GATEWAY_URL", &gateway_url);
        set_env_var("TOOL_GATEWAY_USER_TOKEN", "nous-token");

        let result = serde_json::from_str::<Value>(&handle_image_generate(
            &json!({"prompt": "draw cat", "aspect_ratio": "square"}),
            &crate::tools::ToolRuntime::new(temp.path()),
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result["success"], true);
        assert_eq!(result["provider"], "fal");
        assert_eq!(result["image"], "https://v3.fal.media/files/demo/test.png");

        let requests = captured.lock().unwrap();
        assert!(requests[0].starts_with("POST /fal-ai/flux-2/klein/9b "));
        assert!(requests[0].contains("\"image_size\":\"square_hd\""));
        assert!(requests[0].contains("\"num_inference_steps\":4"));

        match previous_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
        match previous_key {
            Some(value) => set_env_var("FAL_KEY", value),
            None => remove_env_var("FAL_KEY"),
        }
        match previous_gateway {
            Some(value) => set_env_var("FAL_QUEUE_GATEWAY_URL", value),
            None => remove_env_var("FAL_QUEUE_GATEWAY_URL"),
        }
        match previous_token {
            Some(value) => set_env_var("TOOL_GATEWAY_USER_TOKEN", value),
            None => remove_env_var("TOOL_GATEWAY_USER_TOKEN"),
        }
    }

    #[test]
    fn image_generate_runs_with_openai_provider() {
        let temp = TempDir::new().unwrap();
        let previous_home = env::var_os("HERMES_HOME");
        set_env_var("HERMES_HOME", temp.path());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            captured_clone.lock().unwrap().push(format!(
                "{}\n{}",
                headers.lines().next().unwrap_or_default(),
                String::from_utf8_lossy(&body_bytes)
            ));

            let response_body = json!({
                "data": [{
                    "b64_json": base64::engine::general_purpose::STANDARD.encode(b"openai-image"),
                    "revised_prompt": "revised cat"
                }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "image_gen:\n  provider: openai\n  openai:\n    base_url: http://{}\n    api_key: openai-test-key\n    model: gpt-image-2-high\n",
                addr
            ),
        )
        .unwrap();
        let result = serde_json::from_str::<Value>(&handle_image_generate(
            &json!({"prompt": "draw cat", "aspect_ratio": "portrait"}),
            &crate::tools::ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result["success"], true);
        assert_eq!(result["provider"], "openai");
        assert_eq!(result["model"], "gpt-image-2-high");
        assert_eq!(result["quality"], "high");
        assert_eq!(result["size"], "1024x1536");
        assert_eq!(result["revised_prompt"], "revised cat");
        let image_path = result["image"].as_str().unwrap();
        assert!(image_path.contains("/cache/images/"));
        assert_eq!(fs::read(image_path).unwrap(), b"openai-image");

        let requests = captured.lock().unwrap();
        assert!(requests[0].starts_with("POST /images/generations "));
        assert!(requests[0].contains("\"model\":\"gpt-image-2\""));
        assert!(requests[0].contains("\"size\":\"1024x1536\""));
        assert!(requests[0].contains("\"quality\":\"high\""));

        match previous_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
    }

    #[test]
    fn image_generate_runs_with_openai_codex_provider() {
        let temp = TempDir::new().unwrap();
        let previous_home = env::var_os("HERMES_HOME");
        set_env_var("HERMES_HOME", temp.path());

        let token = {
            let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(r#"{"alg":"none","typ":"JWT"}"#);
            let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
                json!({
                    "exp": i64::MAX / 2,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct-image",
                    }
                })
                .to_string(),
            );
            format!("{header}.{payload}.sig")
        };
        fs::write(
            temp.path().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": token,
                            "refresh_token": "refresh-image",
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            captured_clone.lock().unwrap().push(format!(
                "{}\n{}{}",
                headers,
                if headers.ends_with("\n") { "" } else { "\n" },
                String::from_utf8_lossy(&body_bytes)
            ));

            let response_body = json!({
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig_test",
                    "result": base64::engine::general_purpose::STANDARD.encode(b"codex-image")
                }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "image_gen:\n  provider: openai-codex\n  openai-codex:\n    base_url: http://{}\n    model: gpt-image-2-low\n",
                addr
            ),
        )
        .unwrap();
        let result = serde_json::from_str::<Value>(&handle_image_generate(
            &json!({"prompt": "draw owl", "aspect_ratio": "square"}),
            &crate::tools::ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result["success"], true);
        assert_eq!(result["provider"], "openai-codex");
        assert_eq!(result["model"], "gpt-image-2-low");
        assert_eq!(result["quality"], "low");
        assert_eq!(result["size"], "1024x1024");
        let image_path = result["image"].as_str().unwrap();
        assert!(image_path.contains("/cache/images/"));
        assert_eq!(fs::read(image_path).unwrap(), b"codex-image");

        let requests = captured.lock().unwrap();
        let request = &requests[0];
        let request_lower = request.to_ascii_lowercase();
        assert!(request.contains("POST /responses HTTP/1.1"));
        assert!(request_lower.contains("authorization: bearer "));
        assert!(request_lower.contains("originator: codex_cli_rs"));
        assert!(request_lower.contains("user-agent: codex_cli_rs/0.0.0 (hermes agent)"));
        assert!(request_lower.contains("chatgpt-account-id: acct-image"));
        assert!(request.contains("\"model\":\"gpt-5.4\""));
        assert!(request.contains("\"type\":\"image_generation\""));
        assert!(request.contains("\"quality\":\"low\""));
        assert!(request.contains("\"size\":\"1024x1024\""));
        assert!(request.contains("\"tool_choice\":"));
        assert!(request.contains("\"allowed_tools\""));
        assert!(request.contains("\"mode\":\"required\""));

        match previous_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
    }

    #[test]
    fn image_generate_runs_with_xai_provider() {
        let temp = TempDir::new().unwrap();
        let previous_home = env::var_os("HERMES_HOME");
        set_env_var("HERMES_HOME", temp.path());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            captured_clone.lock().unwrap().push(format!(
                "{}\n{}",
                headers.lines().next().unwrap_or_default(),
                String::from_utf8_lossy(&body_bytes)
            ));

            let response_body = json!({
                "data": [{
                    "b64_json": base64::engine::general_purpose::STANDARD.encode(b"xai-image")
                }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "image_gen:\n  provider: xai\n  xai:\n    base_url: http://{}\n    api_key: xai-test-key\n    resolution: 2k\n",
                addr
            ),
        )
        .unwrap();
        let result = serde_json::from_str::<Value>(&handle_image_generate(
            &json!({"prompt": "draw fox", "aspect_ratio": "landscape"}),
            &crate::tools::ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result["success"], true);
        assert_eq!(result["provider"], "xai");
        assert_eq!(result["model"], "grok-imagine-image");
        assert_eq!(result["resolution"], "2048");
        assert_eq!(result["resolution_key"], "2k");
        let image_path = result["image"].as_str().unwrap();
        assert!(image_path.contains("/cache/images/"));
        assert_eq!(fs::read(image_path).unwrap(), b"xai-image");

        let requests = captured.lock().unwrap();
        assert!(requests[0].starts_with("POST /images/generations "));
        assert!(requests[0].contains("\"model\":\"grok-imagine-image\""));
        assert!(requests[0].contains("\"aspect_ratio\":\"16:9\""));
        assert!(requests[0].contains("\"resolution\":\"2048\""));

        match previous_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
    }

    #[test]
    fn unsupported_provider_returns_registry_style_error() {
        let temp = TempDir::new().unwrap();
        let previous_home = env::var_os("HERMES_HOME");
        set_env_var("HERMES_HOME", temp.path());
        fs::write(
            temp.path().join("config.yaml"),
            "image_gen:\n  provider: replicate\n",
        )
        .unwrap();
        let result = serde_json::from_str::<Value>(&handle_image_generate(
            &json!({"prompt": "draw cat"}),
            &crate::tools::ToolRuntime::new(temp.path()),
        ))
        .unwrap();
        assert_eq!(result["success"], false);
        assert_eq!(result["error_type"], "provider_not_registered");
        match previous_home {
            Some(value) => set_env_var("HERMES_HOME", value),
            None => remove_env_var("HERMES_HOME"),
        }
    }
}
