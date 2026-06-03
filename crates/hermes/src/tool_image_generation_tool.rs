//! Image generation tool — native Rust port of `tools/image_generation_tool.py`.
//!
//! Provides image generation via FAL.ai. Multiple FAL models are supported and
//! selectable via `hermes tools` -> Image Generation; the active model is
//! persisted to `image_gen.model` in `config.yaml`.
//!
//! Architecture (mirrors the Python original):
//! - [`fal_models`] is a catalog of supported models with per-model metadata
//!   (size-style family, defaults, `supports` whitelist, upscaler flag).
//! - [`build_fal_payload`] translates the agent's unified inputs (prompt +
//!   aspect_ratio) into the model-specific payload and filters to the
//!   `supports` whitelist so models never receive rejected keys.
//! - Upscaling via FAL's Clarity Upscaler is gated per-model via the `upscale`
//!   flag — on for FLUX 2 Pro (backward-compat), off for all other models.
//!
//! Network specifics: the original delegates to the `fal_client` Python SDK,
//! which submits to FAL's queue API (`POST https://queue.fal.run/<model>`),
//! then polls the status URL until `COMPLETED` and fetches the response URL.
//! This port reproduces that request-construction + response-parsing flow with
//! `reqwest::blocking`. The managed Nous gateway path overrides the queue
//! origin with the gateway origin and authenticates with the Nous user token.

use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

// Cross-module references. These live as flat files in the `hermes-core` crate
// (accessed as `hermes_core::<mod>` from this binary crate). `config_image_model`
// and the backend helpers degrade gracefully when config/credentials are absent.
//
// Relevant cross-refs (resolved at integration time):
//   * hermes_core::cli_config::load_config — config.yaml loader
//   * hermes_core::tool_tool_backend_helpers::{fal_key_is_configured, prefers_gateway}
//   * hermes_core::tool_managed_tool_gateway::resolve_managed_tool_gateway

// ---------------------------------------------------------------------------
// FAL model catalog
// ---------------------------------------------------------------------------

/// Size-spec family for a model. Mirrors the Python `size_style` strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeStyle {
    /// preset enum ("square_hd", "landscape_16_9", ...) -> `image_size`.
    ImageSizePreset,
    /// aspect ratio enum ("16:9", "1:1", ...) -> `aspect_ratio`.
    AspectRatio,
    /// literal dimension strings ("1024x1024", etc.) -> `image_size`.
    GptLiteral,
}

/// Per-model metadata describing how to translate unified inputs into the
/// model's native FAL payload shape.
#[derive(Debug, Clone)]
pub struct FalModel {
    pub display: &'static str,
    pub speed: &'static str,
    pub strengths: &'static str,
    pub price: &'static str,
    pub size_style: SizeStyle,
    /// landscape / square / portrait -> native size token.
    pub sizes: HashMap<&'static str, &'static str>,
    /// Default payload keys merged before overrides/filtering.
    pub defaults: Map<String, Value>,
    /// Whitelist of keys allowed in the outgoing payload.
    pub supports: HashSet<&'static str>,
    /// Whether to chain Clarity Upscaler after generation.
    pub upscale: bool,
}

/// Default model — the fastest reasonable option. Kept cheap and sub-1s.
pub const DEFAULT_MODEL: &str = "fal-ai/flux-2/klein/9b";

pub const DEFAULT_ASPECT_RATIO: &str = "landscape";
pub const VALID_ASPECT_RATIOS: [&str; 3] = ["landscape", "square", "portrait"];

// Upscaler (Clarity Upscaler) constants.
pub const UPSCALER_MODEL: &str = "fal-ai/clarity-upscaler";
pub const UPSCALER_FACTOR: i64 = 2;
pub const UPSCALER_SAFETY_CHECKER: bool = false;
pub const UPSCALER_DEFAULT_PROMPT: &str = "masterpiece, best quality, highres";
pub const UPSCALER_NEGATIVE_PROMPT: &str = "(worst quality, low quality, normal quality:2)";
pub const UPSCALER_CREATIVITY: f64 = 0.35;
pub const UPSCALER_RESEMBLANCE: f64 = 0.6;
pub const UPSCALER_GUIDANCE_SCALE: i64 = 4;
pub const UPSCALER_NUM_INFERENCE_STEPS: i64 = 18;

fn sizes(landscape: &'static str, square: &'static str, portrait: &'static str) -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("landscape", landscape);
    m.insert("square", square);
    m.insert("portrait", portrait);
    m
}

fn supports(keys: &[&'static str]) -> HashSet<&'static str> {
    keys.iter().copied().collect()
}

fn defaults(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), v.clone());
    }
    m
}

/// Build the full FAL model catalog. Faithful reproduction of `FAL_MODELS`.
pub fn fal_models() -> HashMap<&'static str, FalModel> {
    let mut models: HashMap<&'static str, FalModel> = HashMap::new();

    models.insert(
        "fal-ai/flux-2/klein/9b",
        FalModel {
            display: "FLUX 2 Klein 9B",
            speed: "<1s",
            strengths: "Fast, crisp text",
            price: "$0.006/MP",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("num_inference_steps", json!(4)),
                ("output_format", json!("png")),
                ("enable_safety_checker", json!(false)),
            ]),
            supports: supports(&[
                "prompt", "image_size", "num_inference_steps", "seed",
                "output_format", "enable_safety_checker",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/flux-2-pro",
        FalModel {
            display: "FLUX 2 Pro",
            speed: "~6s",
            strengths: "Studio photorealism",
            price: "$0.03/MP",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("num_inference_steps", json!(50)),
                ("guidance_scale", json!(4.5)),
                ("num_images", json!(1)),
                ("output_format", json!("png")),
                ("enable_safety_checker", json!(false)),
                ("safety_tolerance", json!("5")),
                ("sync_mode", json!(true)),
            ]),
            supports: supports(&[
                "prompt", "image_size", "num_inference_steps", "guidance_scale",
                "num_images", "output_format", "enable_safety_checker",
                "safety_tolerance", "sync_mode", "seed",
            ]),
            upscale: true, // Backward-compat: current default behavior.
        },
    );

    models.insert(
        "fal-ai/z-image/turbo",
        FalModel {
            display: "Z-Image Turbo",
            speed: "~2s",
            strengths: "Bilingual EN/CN, 6B",
            price: "$0.005/MP",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("num_inference_steps", json!(8)),
                ("num_images", json!(1)),
                ("output_format", json!("png")),
                ("enable_safety_checker", json!(false)),
                ("enable_prompt_expansion", json!(false)),
            ]),
            supports: supports(&[
                "prompt", "image_size", "num_inference_steps", "num_images",
                "seed", "output_format", "enable_safety_checker",
                "enable_prompt_expansion",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/nano-banana-pro",
        FalModel {
            display: "Nano Banana Pro (Gemini 3 Pro Image)",
            speed: "~8s",
            strengths: "Gemini 3 Pro, reasoning depth, text rendering",
            price: "$0.15/image (1K)",
            size_style: SizeStyle::AspectRatio,
            sizes: sizes("16:9", "1:1", "9:16"),
            defaults: defaults(&[
                ("num_images", json!(1)),
                ("output_format", json!("png")),
                ("safety_tolerance", json!("5")),
                ("resolution", json!("1K")),
            ]),
            supports: supports(&[
                "prompt", "aspect_ratio", "num_images", "output_format",
                "safety_tolerance", "seed", "sync_mode", "resolution",
                "enable_web_search", "limit_generations",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/gpt-image-1.5",
        FalModel {
            display: "GPT Image 1.5",
            speed: "~15s",
            strengths: "Prompt adherence",
            price: "$0.034/image",
            size_style: SizeStyle::GptLiteral,
            sizes: sizes("1536x1024", "1024x1024", "1024x1536"),
            defaults: defaults(&[
                ("quality", json!("medium")),
                ("num_images", json!(1)),
                ("output_format", json!("png")),
            ]),
            supports: supports(&[
                "prompt", "image_size", "quality", "num_images", "output_format",
                "background", "sync_mode",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/gpt-image-2",
        FalModel {
            display: "GPT Image 2",
            speed: "~20s",
            strengths: "SOTA text rendering + CJK, world-aware photorealism",
            price: "$0.04–0.06/image",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_4_3", "square_hd", "portrait_4_3"),
            defaults: defaults(&[
                ("quality", json!("medium")),
                ("num_images", json!(1)),
                ("output_format", json!("png")),
            ]),
            supports: supports(&[
                "prompt", "image_size", "quality", "num_images", "output_format",
                "sync_mode",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/ideogram/v3",
        FalModel {
            display: "Ideogram V3",
            speed: "~5s",
            strengths: "Best typography",
            price: "$0.03-0.09/image",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("rendering_speed", json!("BALANCED")),
                ("expand_prompt", json!(true)),
                ("style", json!("AUTO")),
            ]),
            supports: supports(&[
                "prompt", "image_size", "rendering_speed", "expand_prompt",
                "style", "seed",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/recraft/v4/pro/text-to-image",
        FalModel {
            display: "Recraft V4 Pro",
            speed: "~8s",
            strengths: "Design, brand systems, production-ready",
            price: "$0.25/image",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("enable_safety_checker", json!(false)),
            ]),
            supports: supports(&[
                "prompt", "image_size", "enable_safety_checker",
                "colors", "background_color",
            ]),
            upscale: false,
        },
    );

    models.insert(
        "fal-ai/qwen-image",
        FalModel {
            display: "Qwen Image",
            speed: "~12s",
            strengths: "LLM-based, complex text",
            price: "$0.02/MP",
            size_style: SizeStyle::ImageSizePreset,
            sizes: sizes("landscape_16_9", "square_hd", "portrait_16_9"),
            defaults: defaults(&[
                ("num_inference_steps", json!(30)),
                ("guidance_scale", json!(2.5)),
                ("num_images", json!(1)),
                ("output_format", json!("png")),
                ("acceleration", json!("regular")),
            ]),
            supports: supports(&[
                "prompt", "image_size", "num_inference_steps", "guidance_scale",
                "num_images", "output_format", "acceleration", "seed", "sync_mode",
            ]),
            upscale: false,
        },
    );

    models
}

// ---------------------------------------------------------------------------
// Model resolution + payload construction
// ---------------------------------------------------------------------------

/// Resolve the active FAL model id from config.yaml (primary), the
/// `FAL_IMAGE_MODEL` env-var escape hatch, or [`DEFAULT_MODEL`].
///
/// `config_model` should be the value of `image_gen.model` from config.yaml
/// (pass `None` if unset/unavailable). This keeps the function pure and
/// testable; the caller wires config loading. Falls back to [`DEFAULT_MODEL`]
/// if the configured model is unknown.
///
/// Returns the resolved model id (always a key present in [`fal_models`]).
pub fn resolve_fal_model_id(config_model: Option<&str>) -> String {
    let mut model_id = config_model.map(|s| s.trim().to_string()).unwrap_or_default();

    // Env var escape hatch (undocumented; backward-compat for tests/scripts).
    if model_id.is_empty() {
        model_id = std::env::var("FAL_IMAGE_MODEL")
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
    }

    if model_id.is_empty() {
        return DEFAULT_MODEL.to_string();
    }

    let catalog = fal_models();
    if !catalog.contains_key(model_id.as_str()) {
        log::warn!(
            "Unknown FAL model '{}' in config; falling back to {}",
            model_id, DEFAULT_MODEL
        );
        return DEFAULT_MODEL.to_string();
    }

    model_id
}

/// Read `image_gen.model` from config.yaml via the ported config loader.
///
/// Mirrors the config-load branch of `_resolve_fal_model`. Returns `None` when
/// the value is absent or not a string. Never panics.
pub fn config_image_model() -> Option<String> {
    let cfg = hermes_core::cli_config::load_config();
    let section = cfg.get("image_gen")?;
    let raw = section.get("model")?;
    raw.as_str().map(|s| s.trim().to_string())
}

/// Build a FAL request payload for `model_id` from unified inputs.
///
/// Translates `aspect_ratio` into the model's native size spec, merges model
/// defaults, applies caller overrides (skipping null values), then filters to
/// the model's `supports` whitelist. Faithful port of `_build_fal_payload`.
///
/// Panics only if `model_id` is not in the catalog — callers always resolve
/// via [`resolve_fal_model_id`] first, matching the Python `FAL_MODELS[id]`
/// indexing.
pub fn build_fal_payload(
    model_id: &str,
    prompt: &str,
    aspect_ratio: &str,
    seed: Option<i64>,
    overrides: Option<&Map<String, Value>>,
) -> Map<String, Value> {
    let catalog = fal_models();
    let meta = catalog
        .get(model_id)
        .unwrap_or_else(|| panic!("Unknown FAL model: {model_id}"));

    let mut aspect = if aspect_ratio.is_empty() {
        DEFAULT_ASPECT_RATIO.to_string()
    } else {
        aspect_ratio.to_lowercase().trim().to_string()
    };
    if !meta.sizes.contains_key(aspect.as_str()) {
        aspect = DEFAULT_ASPECT_RATIO.to_string();
    }

    let mut payload: Map<String, Value> = meta.defaults.clone();
    payload.insert("prompt".to_string(), json!(prompt.trim()));

    match meta.size_style {
        SizeStyle::ImageSizePreset | SizeStyle::GptLiteral => {
            payload.insert(
                "image_size".to_string(),
                json!(meta.sizes.get(aspect.as_str()).copied().unwrap_or("")),
            );
        }
        SizeStyle::AspectRatio => {
            payload.insert(
                "aspect_ratio".to_string(),
                json!(meta.sizes.get(aspect.as_str()).copied().unwrap_or("")),
            );
        }
    }

    if let Some(s) = seed {
        payload.insert("seed".to_string(), json!(s));
    }

    if let Some(ov) = overrides {
        for (k, v) in ov {
            if !v.is_null() {
                payload.insert(k.clone(), v.clone());
            }
        }
    }

    // Filter to the supports whitelist.
    payload
        .into_iter()
        .filter(|(k, _)| meta.supports.contains(k.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// FAL queue submission (reqwest::blocking)
// ---------------------------------------------------------------------------

/// Where to submit a FAL request: directly to `queue.fal.run` with the FAL key,
/// or through the managed Nous gateway with the Nous user token.
#[derive(Debug, Clone)]
pub struct FalEndpoint {
    /// Queue origin with a trailing slash, e.g. `https://queue.fal.run/` or
    /// the managed gateway origin.
    pub queue_origin: String,
    /// API key used for the `Authorization: Key <key>` header.
    pub key: String,
}

impl FalEndpoint {
    /// Direct FAL endpoint using `FAL_KEY`.
    pub fn direct(key: impl Into<String>) -> Self {
        Self {
            queue_origin: "https://queue.fal.run/".to_string(),
            key: key.into(),
        }
    }

    /// Managed gateway endpoint. Normalizes the origin to a trailing slash,
    /// mirroring `_normalize_fal_queue_url_format`.
    pub fn managed(origin: &str, token: impl Into<String>) -> Result<Self, String> {
        let normalized = normalize_fal_queue_url_format(origin)?;
        Ok(Self {
            queue_origin: normalized,
            key: token.into(),
        })
    }
}

/// Normalize a managed FAL queue origin to a trailing-slash form.
/// Faithful port of `_normalize_fal_queue_url_format`.
pub fn normalize_fal_queue_url_format(queue_run_origin: &str) -> Result<String, String> {
    let normalized = queue_run_origin.trim().trim_end_matches('/');
    if normalized.is_empty() {
        return Err("Managed FAL queue origin is required".to_string());
    }
    Ok(format!("{normalized}/"))
}

/// A submitted FAL queue request handle (parsed from the submit response).
#[derive(Debug, Clone)]
pub struct FalRequestHandle {
    pub request_id: String,
    pub response_url: String,
    pub status_url: String,
    pub cancel_url: String,
}

/// Submit a FAL queue request and return the request handle.
///
/// Reproduces `fal_client.submit` / `_ManagedFalSyncClient.submit`:
/// `POST <queue_origin><model>` with the JSON arguments, an idempotency key
/// header, and `Authorization: Key <key>`. The response carries the request
/// id + the polling/response URLs.
pub fn submit_fal_request(
    endpoint: &FalEndpoint,
    model: &str,
    arguments: &Value,
    idempotency_key: &str,
    timeout: Duration,
) -> Result<FalRequestHandle, String> {
    let url = format!("{}{}", endpoint.queue_origin, model);
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let resp = client
        .post(&url)
        .header("Authorization", format!("Key {}", endpoint.key))
        .header("x-idempotency-key", idempotency_key)
        .json(arguments)
        .send()
        .map_err(|e| format!("FAL submit request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        // 4xx from the managed gateway typically means the portal doesn't
        // currently proxy this model — surface a clearer message. We can't
        // distinguish gateway vs direct here, so mirror the Python message
        // shape for 4xx and a generic message otherwise.
        if (400..500).contains(&code) {
            return Err(format!(
                "Nous Subscription gateway rejected model '{model}' (HTTP {code}). \
                 This model may not yet be enabled on the Nous Portal's FAL proxy. Either:\n\
                 \u{2022} Set FAL_KEY in your environment to use FAL.ai directly, or\n\
                 \u{2022} Pick a different model via `hermes tools` \u{2192} Image Generation."
            ));
        }
        return Err(format!("FAL submit returned HTTP {code}"));
    }

    let data: Value = resp
        .json()
        .map_err(|e| format!("Failed to parse FAL submit response: {e}"))?;

    Ok(FalRequestHandle {
        request_id: data
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        response_url: data
            .get("response_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        status_url: data
            .get("status_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        cancel_url: data
            .get("cancel_url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
    })
}

/// Poll the status URL until the request completes, then fetch the response.
///
/// Reproduces `fal_client` `handle.get()`: poll `status_url` until the JSON
/// `status` is `COMPLETED`, then GET `response_url` for the final result.
pub fn fal_get_result(
    endpoint: &FalEndpoint,
    handle: &FalRequestHandle,
    poll_interval: Duration,
    overall_timeout: Duration,
) -> Result<Value, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(poll_interval.max(Duration::from_secs(30)))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > overall_timeout {
            return Err("Timed out waiting for FAL result".to_string());
        }
        let status_resp = client
            .get(&handle.status_url)
            .header("Authorization", format!("Key {}", endpoint.key))
            .send()
            .map_err(|e| format!("FAL status poll failed: {e}"))?;
        if !status_resp.status().is_success() {
            return Err(format!(
                "FAL status poll returned HTTP {}",
                status_resp.status().as_u16()
            ));
        }
        let status_json: Value = status_resp
            .json()
            .map_err(|e| format!("Failed to parse FAL status: {e}"))?;
        let status = status_json
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if status == "COMPLETED" {
            break;
        }
        std::thread::sleep(poll_interval);
    }

    let resp = client
        .get(&handle.response_url)
        .header("Authorization", format!("Key {}", endpoint.key))
        .send()
        .map_err(|e| format!("FAL response fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "FAL response fetch returned HTTP {}",
            resp.status().as_u16()
        ));
    }
    resp.json()
        .map_err(|e| format!("Failed to parse FAL result: {e}"))
}

// ---------------------------------------------------------------------------
// Upscaler
// ---------------------------------------------------------------------------

/// A formatted output image entry, matching the dict shape the Python tool
/// builds for each image.
#[derive(Debug, Clone, PartialEq)]
pub struct FormattedImage {
    pub url: String,
    pub width: i64,
    pub height: i64,
    pub upscaled: bool,
    /// Present only for upscaled images (Python sets `upscale_factor`).
    pub upscale_factor: Option<i64>,
}

/// Build the Clarity Upscaler arguments payload for a given source image URL
/// and original prompt. Faithful port of `_upscale_image`'s argument map.
pub fn build_upscaler_arguments(image_url: &str, original_prompt: &str) -> Value {
    json!({
        "image_url": image_url,
        "prompt": format!("{}, {}", UPSCALER_DEFAULT_PROMPT, original_prompt),
        "upscale_factor": UPSCALER_FACTOR,
        "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
        "creativity": UPSCALER_CREATIVITY,
        "resemblance": UPSCALER_RESEMBLANCE,
        "guidance_scale": UPSCALER_GUIDANCE_SCALE,
        "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
        "enable_safety_checker": UPSCALER_SAFETY_CHECKER,
    })
}

/// Parse a Clarity Upscaler result into a [`FormattedImage`].
///
/// Returns `None` when the response lacks the `image.url` field (caller falls
/// back to the original image), mirroring `_upscale_image`'s validation.
pub fn parse_upscaler_result(result: &Value) -> Option<FormattedImage> {
    let image = result.get("image")?;
    let url = image.get("url")?.as_str()?.to_string();
    Some(FormattedImage {
        url,
        width: image.get("width").and_then(|v| v.as_i64()).unwrap_or(0),
        height: image.get("height").and_then(|v| v.as_i64()).unwrap_or(0),
        upscaled: true,
        upscale_factor: Some(UPSCALER_FACTOR),
    })
}

// ---------------------------------------------------------------------------
// Response parsing for the main generation result
// ---------------------------------------------------------------------------

/// Parse a FAL generation result's `images` array into raw (non-upscaled)
/// formatted images. Mirrors the loop in `image_generate_tool` before
/// upscaling — entries without a `url` are skipped.
///
/// Returns an error string matching the Python `ValueError` messages for the
/// invalid-response cases so the caller can surface them verbatim.
pub fn parse_generation_images(result: &Value) -> Result<Vec<FormattedImage>, String> {
    let obj = match result {
        Value::Object(_) => result,
        _ => return Err("Invalid response from FAL.ai API — no images returned".to_string()),
    };
    let images = match obj.get("images") {
        Some(Value::Array(arr)) => arr,
        Some(_) | None => {
            return Err("Invalid response from FAL.ai API — no images returned".to_string())
        }
    };
    if images.is_empty() {
        return Err("No images were generated".to_string());
    }

    let mut out = Vec::new();
    for img in images {
        let url = match img.get("url").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => continue,
        };
        out.push(FormattedImage {
            url,
            width: img.get("width").and_then(|v| v.as_i64()).unwrap_or(0),
            height: img.get("height").and_then(|v| v.as_i64()).unwrap_or(0),
            upscaled: false,
            upscale_factor: None,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tool result construction
// ---------------------------------------------------------------------------

/// Serialize a successful response as the Python tool does: a JSON string with
/// `{"success": true, "image": <url|null>}` pretty-printed (indent 2).
pub fn success_response_json(image_url: Option<&str>) -> String {
    let val = json!({
        "success": true,
        "image": image_url,
    });
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_string())
}

/// Serialize an error response: `{"success": false, "image": null,
/// "error": <msg>, "error_type": <type>}`. `error_type` mirrors the Python
/// `type(e).__name__`; callers pass the equivalent label (e.g. "ValueError").
pub fn error_response_json(error: &str, error_type: &str) -> String {
    let val = json!({
        "success": false,
        "image": Value::Null,
        "error": error,
        "error_type": error_type,
    });
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_string())
}

/// `tool_error`-style JSON used by the registry handler for missing prompt.
pub fn tool_error_json(message: &str) -> String {
    json!({ "error": message }).to_string()
}

// ---------------------------------------------------------------------------
// Aspect-ratio validation helper
// ---------------------------------------------------------------------------

/// Normalize and validate an aspect ratio, defaulting to [`DEFAULT_ASPECT_RATIO`]
/// when empty or not one of [`VALID_ASPECT_RATIOS`]. Mirrors the validation in
/// `image_generate_tool`.
pub fn normalize_aspect_ratio(aspect_ratio: &str) -> String {
    let lc = if aspect_ratio.is_empty() {
        DEFAULT_ASPECT_RATIO.to_string()
    } else {
        aspect_ratio.to_lowercase().trim().to_string()
    };
    if VALID_ASPECT_RATIOS.contains(&lc.as_str()) {
        lc
    } else {
        log::warn!(
            "Invalid aspect_ratio '{}', defaulting to '{}'",
            aspect_ratio, DEFAULT_ASPECT_RATIO
        );
        DEFAULT_ASPECT_RATIO.to_string()
    }
}

/// Build the overrides map from the optional direct-caller kwargs. Mirrors the
/// `overrides` assembly in `image_generate_tool` (only non-`None` values).
pub fn build_overrides(
    num_inference_steps: Option<i64>,
    guidance_scale: Option<f64>,
    num_images: Option<i64>,
    output_format: Option<&str>,
) -> Map<String, Value> {
    let mut m = Map::new();
    if let Some(v) = num_inference_steps {
        m.insert("num_inference_steps".to_string(), json!(v));
    }
    if let Some(v) = guidance_scale {
        m.insert("guidance_scale".to_string(), json!(v));
    }
    if let Some(v) = num_images {
        m.insert("num_images".to_string(), json!(v));
    }
    if let Some(v) = output_format {
        m.insert("output_format".to_string(), json!(v));
    }
    m
}

// ---------------------------------------------------------------------------
// Tool schema (registry)
// ---------------------------------------------------------------------------

/// The agent-facing JSON schema for the `image_generate` tool. Faithful port
/// of `IMAGE_GENERATE_SCHEMA`.
pub fn image_generate_schema() -> Value {
    json!({
        "name": "image_generate",
        "description": "Generate high-quality images from text prompts. The underlying \
            backend (FAL, OpenAI, etc.) and model are user-configured and not \
            selectable by the agent. Returns either a URL or an absolute file \
            path in the `image` field; display it with markdown \
            ![description](url-or-path) and the gateway will deliver it.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt describing the desired image. Be detailed and descriptive.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": VALID_ASPECT_RATIOS.to_vec(),
                    "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                    "default": DEFAULT_ASPECT_RATIO,
                },
            },
            "required": ["prompt"],
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_all_models_and_default_present() {
        let m = fal_models();
        assert_eq!(m.len(), 9);
        assert!(m.contains_key(DEFAULT_MODEL));
        assert!(m.contains_key("fal-ai/gpt-image-2"));
    }

    #[test]
    fn build_payload_preset_filters_to_supports() {
        let p = build_fal_payload("fal-ai/flux-2/klein/9b", "  a cat  ", "square", None, None);
        // prompt is trimmed.
        assert_eq!(p.get("prompt").unwrap(), "a cat");
        // square -> square_hd for this preset model.
        assert_eq!(p.get("image_size").unwrap(), "square_hd");
        // num_inference_steps default kept (in supports).
        assert_eq!(p.get("num_inference_steps").unwrap(), 4);
        // guidance_scale not in supports for klein -> absent.
        assert!(p.get("guidance_scale").is_none());
        // klein has no num_images in supports.
        assert!(p.get("num_images").is_none());
    }

    #[test]
    fn build_payload_aspect_ratio_style() {
        let p = build_fal_payload("fal-ai/nano-banana-pro", "x", "portrait", None, None);
        assert_eq!(p.get("aspect_ratio").unwrap(), "9:16");
        assert!(p.get("image_size").is_none());
        assert_eq!(p.get("resolution").unwrap(), "1K");
    }

    #[test]
    fn build_payload_gpt_literal() {
        let p = build_fal_payload("fal-ai/gpt-image-1.5", "x", "landscape", None, None);
        assert_eq!(p.get("image_size").unwrap(), "1536x1024");
        assert_eq!(p.get("quality").unwrap(), "medium");
    }

    #[test]
    fn invalid_aspect_defaults_to_landscape() {
        let p = build_fal_payload("fal-ai/flux-2/klein/9b", "x", "diagonal", None, None);
        // landscape preset for klein.
        assert_eq!(p.get("image_size").unwrap(), "landscape_16_9");
    }

    #[test]
    fn seed_only_kept_when_supported() {
        // klein supports seed.
        let p = build_fal_payload("fal-ai/flux-2/klein/9b", "x", "square", Some(42), None);
        assert_eq!(p.get("seed").unwrap(), 42);
        // recraft does not support seed.
        let p2 = build_fal_payload(
            "fal-ai/recraft/v4/pro/text-to-image",
            "x",
            "square",
            Some(42),
            None,
        );
        assert!(p2.get("seed").is_none());
    }

    #[test]
    fn overrides_skip_nulls_and_unsupported() {
        let mut ov = Map::new();
        ov.insert("num_images".to_string(), Value::Null); // skipped (null)
        ov.insert("num_inference_steps".to_string(), json!(8));
        ov.insert("guidance_scale".to_string(), json!(3.0)); // unsupported for klein
        let p = build_fal_payload("fal-ai/flux-2/klein/9b", "x", "square", None, Some(&ov));
        assert_eq!(p.get("num_inference_steps").unwrap(), 8);
        assert!(p.get("guidance_scale").is_none());
        assert!(p.get("num_images").is_none());
    }

    #[test]
    fn upscale_flag_only_for_flux2_pro() {
        let m = fal_models();
        assert!(m.get("fal-ai/flux-2-pro").unwrap().upscale);
        assert!(!m.get("fal-ai/flux-2/klein/9b").unwrap().upscale);
        assert!(!m.get("fal-ai/qwen-image").unwrap().upscale);
    }

    #[test]
    fn resolve_model_config_then_env_then_default() {
        // Unknown config -> default.
        unsafe {
            std::env::remove_var("FAL_IMAGE_MODEL");
        }
        assert_eq!(resolve_fal_model_id(Some("nope/unknown")), DEFAULT_MODEL);
        // Known config wins.
        assert_eq!(
            resolve_fal_model_id(Some("fal-ai/qwen-image")),
            "fal-ai/qwen-image"
        );
        // Empty config falls to env.
        unsafe {
            std::env::set_var("FAL_IMAGE_MODEL", "fal-ai/ideogram/v3");
        }
        assert_eq!(resolve_fal_model_id(None), "fal-ai/ideogram/v3");
        unsafe {
            std::env::set_var("FAL_IMAGE_MODEL", "fal-ai/unknown");
        }
        assert_eq!(resolve_fal_model_id(None), DEFAULT_MODEL);
        unsafe {
            std::env::remove_var("FAL_IMAGE_MODEL");
        }
        // Nothing set -> default.
        assert_eq!(resolve_fal_model_id(None), DEFAULT_MODEL);
    }

    #[test]
    fn normalize_queue_origin() {
        assert_eq!(
            normalize_fal_queue_url_format("https://x.example/q/").unwrap(),
            "https://x.example/q/"
        );
        assert_eq!(
            normalize_fal_queue_url_format("  https://x.example/q  ").unwrap(),
            "https://x.example/q/"
        );
        assert!(normalize_fal_queue_url_format("   ").is_err());
    }

    #[test]
    fn upscaler_arguments_shape() {
        let a = build_upscaler_arguments("https://img/1.png", "a cat");
        assert_eq!(a.get("image_url").unwrap(), "https://img/1.png");
        assert_eq!(
            a.get("prompt").unwrap(),
            "masterpiece, best quality, highres, a cat"
        );
        assert_eq!(a.get("upscale_factor").unwrap(), 2);
        assert_eq!(a.get("enable_safety_checker").unwrap(), false);
    }

    #[test]
    fn parse_upscaler_result_ok_and_missing() {
        let ok = json!({"image": {"url": "u", "width": 100, "height": 200}});
        let f = parse_upscaler_result(&ok).unwrap();
        assert_eq!(f.url, "u");
        assert_eq!(f.width, 100);
        assert_eq!(f.height, 200);
        assert!(f.upscaled);
        assert_eq!(f.upscale_factor, Some(2));

        // missing image -> None.
        assert!(parse_upscaler_result(&json!({})).is_none());
        // missing url -> None.
        assert!(parse_upscaler_result(&json!({"image": {"width": 1}})).is_none());
    }

    #[test]
    fn parse_generation_images_cases() {
        // No images key.
        assert!(parse_generation_images(&json!({"foo": 1})).is_err());
        // Empty images.
        let e = parse_generation_images(&json!({"images": []})).unwrap_err();
        assert_eq!(e, "No images were generated");
        // Non-object response.
        assert!(parse_generation_images(&json!([1, 2])).is_err());
        // Valid + one without url skipped.
        let imgs = parse_generation_images(&json!({
            "images": [
                {"url": "a", "width": 10, "height": 20},
                {"width": 5},
                {"url": "b"}
            ]
        }))
        .unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].url, "a");
        assert_eq!(imgs[0].width, 10);
        assert!(!imgs[0].upscaled);
        assert_eq!(imgs[1].url, "b");
        assert_eq!(imgs[1].width, 0);
    }

    #[test]
    fn success_and_error_json() {
        let s = success_response_json(Some("https://img/1.png"));
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v.get("success").unwrap(), true);
        assert_eq!(v.get("image").unwrap(), "https://img/1.png");

        let s2 = success_response_json(None);
        let v2: Value = serde_json::from_str(&s2).unwrap();
        assert_eq!(v2.get("image").unwrap(), &Value::Null);

        let e = error_response_json("boom", "ValueError");
        let ev: Value = serde_json::from_str(&e).unwrap();
        assert_eq!(ev.get("success").unwrap(), false);
        assert_eq!(ev.get("error").unwrap(), "boom");
        assert_eq!(ev.get("error_type").unwrap(), "ValueError");
        assert_eq!(ev.get("image").unwrap(), &Value::Null);
    }

    #[test]
    fn aspect_ratio_normalization() {
        assert_eq!(normalize_aspect_ratio("SQUARE"), "square");
        assert_eq!(normalize_aspect_ratio(""), "landscape");
        assert_eq!(normalize_aspect_ratio("weird"), "landscape");
        assert_eq!(normalize_aspect_ratio("portrait"), "portrait");
    }

    #[test]
    fn overrides_builder_only_present() {
        let m = build_overrides(Some(8), None, Some(2), Some("jpeg"));
        assert_eq!(m.get("num_inference_steps").unwrap(), 8);
        assert!(m.get("guidance_scale").is_none());
        assert_eq!(m.get("num_images").unwrap(), 2);
        assert_eq!(m.get("output_format").unwrap(), "jpeg");
    }

    #[test]
    fn schema_has_required_prompt_and_enum() {
        let s = image_generate_schema();
        assert_eq!(s.get("name").unwrap(), "image_generate");
        let req = s["parameters"]["required"].as_array().unwrap();
        assert_eq!(req.len(), 1);
        assert_eq!(req[0], "prompt");
        let en = s["parameters"]["properties"]["aspect_ratio"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(en.len(), 3);
    }

    #[test]
    fn tool_error_shape() {
        let e = tool_error_json("prompt is required for image generation");
        let v: Value = serde_json::from_str(&e).unwrap();
        assert_eq!(v.get("error").unwrap(), "prompt is required for image generation");
    }
}
