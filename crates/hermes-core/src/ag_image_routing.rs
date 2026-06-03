//! Routing helpers for inbound user-attached images.
//!
//! Two modes:
//!
//!   native  — attach images as OpenAI-style `image_url` content parts on the
//!             user turn. Provider adapters (Anthropic, Gemini, Bedrock, Codex,
//!             OpenAI chat.completions) already translate these into their
//!             vendor-specific multimodal formats.
//!
//!   text    — run `vision_analyze` on each image up-front and prepend the
//!             description to the user's text. The model never sees the pixels;
//!             it only sees a lossy text summary. This is the pre-existing
//!             behaviour and still the right choice for non-vision models.
//!
//! The decision is made once per message turn by [`decide_image_input_mode`].
//! It reads `agent.image_input_mode` from config.yaml (`auto` | `native`
//! | `text`, default `auto`) and the active model's capability metadata.
//!
//! In `auto` mode:
//!   - If the user has explicitly configured `auxiliary.vision.provider`
//!     (i.e. not `auto` and not empty), we assume they want the text pipeline
//!     regardless of the main model — they've opted in to a specific vision
//!     backend for a reason (cost, quality, local-only, etc.).
//!   - Otherwise, if the active model reports `supports_vision=True` in its
//!     models.dev metadata, we attach natively.
//!   - Otherwise (non-vision model, no explicit override), we fall back to text.
//!
//! This keeps `vision_analyze` surfaced as a tool in every session — skills
//! and agent flows that chain it (browser screenshots, deeper inspection of
//! URL-referenced images, style-gating loops) keep working. The routing only
//! affects *how user-attached images on the current turn* are presented to the
//! main model.

use std::path::Path;

use serde_json::{json, Value};

/// Valid `agent.image_input_mode` values.
const VALID_MODES: &[&str] = &["auto", "native", "text"];

/// Normalize a config value into one of the valid modes.
///
/// Anything that isn't a recognised string falls back to `"auto"`.
fn coerce_mode(raw: Option<&Value>) -> String {
    match raw {
        Some(Value::String(s)) => {
            let val = s.trim().to_lowercase();
            if VALID_MODES.contains(&val.as_str()) {
                val
            } else {
                "auto".to_string()
            }
        }
        _ => "auto".to_string(),
    }
}

/// Coerce a JSON value into a trimmed string (mirrors Python's `str(x or "")`).
///
/// `null`/missing/empty become `""`. Numbers and bools are stringified the way
/// Python's `str()` would surface them for the relevant config fields (here we
/// only ever feed strings, but be defensive).
fn json_to_trimmed_string(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        // Arrays/objects: Python `str(x or "")` would stringify the container;
        // for these config fields that's never expected, so treat as empty.
        Some(_) => String::new(),
    }
}

/// True when the user configured a specific auxiliary vision backend.
///
/// An explicit override means the user *wants* the text pipeline (they're
/// paying for a dedicated vision model), so we don't silently bypass it.
pub fn explicit_aux_vision_override(cfg: Option<&Value>) -> bool {
    let cfg = match cfg {
        Some(Value::Object(_)) => cfg.unwrap(),
        _ => return false,
    };

    let aux = match cfg.get("auxiliary") {
        Some(v @ Value::Object(_)) => v,
        // `cfg.get("auxiliary") or {}` → if present-but-not-dict, Python keeps
        // the truthy non-dict and then the isinstance check returns False.
        Some(Value::Null) | None => return false,
        Some(_) => return false,
    };

    let vision = match aux.get("vision") {
        Some(v @ Value::Object(_)) => v,
        Some(Value::Null) | None => return false,
        Some(_) => return false,
    };

    let provider = json_to_trimmed_string(vision.get("provider")).to_lowercase();
    let model = json_to_trimmed_string(vision.get("model"));
    let base_url = json_to_trimmed_string(vision.get("base_url"));

    // "auto" / "" / blank = not explicit
    if (provider.is_empty() || provider == "auto") && model.is_empty() && base_url.is_empty() {
        return false;
    }
    true
}

/// Resolve whether the active model supports vision.
///
/// Returns `Some(true)`/`Some(false)` if caps resolve, `None` if unknown.
pub fn lookup_supports_vision(provider: &str, model: &str) -> Option<bool> {
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    let caps = crate::ag_models_dev::get_model_capabilities(provider, model)?;
    Some(caps.supports_vision)
}

/// Return `"native"` or `"text"` for the given turn.
///
/// Args:
///   - `provider`: active inference provider ID (e.g. `"anthropic"`, `"openrouter"`).
///   - `model`:    active model slug as it would be sent to the provider.
///   - `cfg`:      loaded config.yaml value, or `None`. When `None`, behaves as auto.
pub fn decide_image_input_mode(provider: &str, model: &str, cfg: Option<&Value>) -> String {
    let mut mode_cfg = "auto".to_string();
    if let Some(Value::Object(_)) = cfg {
        let cfg = cfg.unwrap();
        if let Some(agent_cfg @ Value::Object(_)) = cfg.get("agent") {
            mode_cfg = coerce_mode(agent_cfg.get("image_input_mode"));
        }
        // `cfg.get("agent") or {}` then isinstance(dict) — non-dict agent_cfg
        // leaves mode_cfg at "auto".
    }

    if mode_cfg == "native" {
        return "native".to_string();
    }
    if mode_cfg == "text" {
        return "text".to_string();
    }

    // auto
    if explicit_aux_vision_override(cfg) {
        return "text".to_string();
    }

    match lookup_supports_vision(provider, model) {
        Some(true) => "native".to_string(),
        _ => "text".to_string(),
    }
}

// Image size handling is REACTIVE rather than proactive: we attempt native
// attachment at full size regardless of provider, and rely on
// `run_agent._try_shrink_image_parts_in_messages` to shrink + retry if
// the provider rejects the request (e.g. Anthropic's hard 5 MB per-image
// ceiling returned as HTTP 400 "image exceeds 5 MB maximum").
//
// Why reactive: our knowledge of provider ceilings is partial and evolving
// (OpenAI accepts 49 MB+, Anthropic 5 MB, Gemini 100 MB, others unknown).
// A proactive per-provider table would be stale the moment a provider raises
// or lowers its limit, and silently degrading quality for users on providers
// that would have accepted the full image is the worse failure mode.

/// Guess the MIME type for a local image path.
///
/// Tries to infer `image/*` from the file extension; defaults to `image/jpeg`
/// when the suffix is unknown (mirrors the Python fallback table).
pub fn guess_mime(path: &Path) -> String {
    let suffix = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match suffix.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "heic" => "image/heic",
        "heif" => "image/heif",
        "avif" => "image/avif",
        _ => "image/jpeg",
    }
    .to_string()
}

/// Encode a local image as a base64 data URL at its native size.
///
/// Size limits are NOT enforced here — the agent retry loop shrinks on the
/// provider's first rejection. Keeping this simple means providers that accept
/// large images (OpenAI 49 MB+, Gemini 100 MB) don't pay a silent quality tax
/// just because one other provider is stricter.
///
/// Returns `None` only if the file can't be read (missing, permission denied,
/// etc.); the caller reports those paths in `skipped`.
pub fn file_to_data_url(path: &Path) -> Option<String> {
    let raw = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(exc) => {
            log::warn!("image_routing: failed to read {} — {}", path.display(), exc);
            return None;
        }
    };
    let mime = guess_mime(path);
    let b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&raw)
    };
    Some(format!("data:{};base64,{}", mime, b64))
}

/// Build an OpenAI-style `content` list for a user turn.
///
/// Shape:
/// ```text
/// [{"type": "text", "text": "..."},
///  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
///  ...]
/// ```
///
/// Images are attached at their native size. If a provider rejects the request
/// because an image is too large (e.g. Anthropic's 5 MB per-image ceiling), the
/// agent's retry loop transparently shrinks and retries once.
///
/// Returns `(content_parts, skipped_paths)`. Skipped paths are files that
/// couldn't be read from disk.
pub fn build_native_content_parts(
    user_text: &str,
    image_paths: &[String],
) -> (Vec<Value>, Vec<String>) {
    let mut parts: Vec<Value> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    let text = user_text.trim();
    if !text.is_empty() {
        parts.push(json!({"type": "text", "text": text}));
    }

    for raw_path in image_paths {
        let p = Path::new(raw_path);
        if !p.is_file() {
            skipped.push(raw_path.clone());
            continue;
        }
        match file_to_data_url(p) {
            Some(data_url) if !data_url.is_empty() => {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }));
            }
            _ => {
                skipped.push(raw_path.clone());
            }
        }
    }

    // If the text was empty, add a neutral prompt so the turn isn't just images.
    let has_image = parts
        .iter()
        .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"));
    if text.is_empty() && has_image {
        parts.insert(0, json!({"type": "text", "text": "What do you see in this image?"}));
    }

    (parts, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_coerce_mode() {
        assert_eq!(coerce_mode(Some(&json!("native"))), "native");
        assert_eq!(coerce_mode(Some(&json!("  TEXT  "))), "text");
        assert_eq!(coerce_mode(Some(&json!("AUTO"))), "auto");
        assert_eq!(coerce_mode(Some(&json!("bogus"))), "auto");
        assert_eq!(coerce_mode(Some(&json!(123))), "auto");
        assert_eq!(coerce_mode(None), "auto");
    }

    #[test]
    fn test_explicit_aux_vision_override_none() {
        assert!(!explicit_aux_vision_override(None));
        assert!(!explicit_aux_vision_override(Some(&json!("not a dict"))));
        assert!(!explicit_aux_vision_override(Some(&json!({}))));
    }

    #[test]
    fn test_explicit_aux_vision_override_auto_blank() {
        let cfg = json!({"auxiliary": {"vision": {"provider": "auto"}}});
        assert!(!explicit_aux_vision_override(Some(&cfg)));

        let cfg2 = json!({"auxiliary": {"vision": {"provider": "  "}}});
        assert!(!explicit_aux_vision_override(Some(&cfg2)));

        let cfg3 = json!({"auxiliary": {"vision": {"provider": "", "model": "", "base_url": ""}}});
        assert!(!explicit_aux_vision_override(Some(&cfg3)));
    }

    #[test]
    fn test_explicit_aux_vision_override_explicit() {
        // explicit provider
        let cfg = json!({"auxiliary": {"vision": {"provider": "openai"}}});
        assert!(explicit_aux_vision_override(Some(&cfg)));

        // model set even if provider auto
        let cfg2 = json!({"auxiliary": {"vision": {"provider": "auto", "model": "gpt-4o"}}});
        assert!(explicit_aux_vision_override(Some(&cfg2)));

        // base_url set
        let cfg3 = json!({"auxiliary": {"vision": {"base_url": "http://localhost:1234"}}});
        assert!(explicit_aux_vision_override(Some(&cfg3)));
    }

    #[test]
    fn test_explicit_aux_vision_non_dict_nested() {
        let cfg = json!({"auxiliary": "string"});
        assert!(!explicit_aux_vision_override(Some(&cfg)));
        let cfg2 = json!({"auxiliary": {"vision": "string"}});
        assert!(!explicit_aux_vision_override(Some(&cfg2)));
    }

    #[test]
    fn test_decide_explicit_modes() {
        let native = json!({"agent": {"image_input_mode": "native"}});
        assert_eq!(decide_image_input_mode("p", "m", Some(&native)), "native");

        let text = json!({"agent": {"image_input_mode": "text"}});
        assert_eq!(decide_image_input_mode("p", "m", Some(&text)), "text");
    }

    #[test]
    fn test_decide_auto_with_aux_override() {
        // auto mode + explicit aux vision → text
        let cfg = json!({
            "agent": {"image_input_mode": "auto"},
            "auxiliary": {"vision": {"provider": "openai"}}
        });
        assert_eq!(decide_image_input_mode("anthropic", "claude", Some(&cfg)), "text");
    }

    #[test]
    fn test_decide_auto_unknown_model_falls_back_text() {
        // No cfg, unknown model → lookup returns None → text
        assert_eq!(
            decide_image_input_mode("nonexistent_provider", "nonexistent_model", None),
            "text"
        );
        // empty provider/model → None → text
        assert_eq!(decide_image_input_mode("", "", None), "text");
    }

    #[test]
    fn test_guess_mime() {
        assert_eq!(guess_mime(Path::new("a.png")), "image/png");
        assert_eq!(guess_mime(Path::new("a.JPG")), "image/jpeg");
        assert_eq!(guess_mime(Path::new("a.jpeg")), "image/jpeg");
        assert_eq!(guess_mime(Path::new("a.gif")), "image/gif");
        assert_eq!(guess_mime(Path::new("a.webp")), "image/webp");
        assert_eq!(guess_mime(Path::new("a.bmp")), "image/bmp");
        assert_eq!(guess_mime(Path::new("a.unknown")), "image/jpeg");
        assert_eq!(guess_mime(Path::new("noext")), "image/jpeg");
    }

    #[test]
    fn test_build_native_content_parts_text_only() {
        let (parts, skipped) = build_native_content_parts("hello", &[]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "hello");
        assert!(skipped.is_empty());
    }

    #[test]
    fn test_build_native_content_parts_missing_image() {
        let (parts, skipped) =
            build_native_content_parts("hi", &["/no/such/file.png".to_string()]);
        assert_eq!(parts.len(), 1);
        assert_eq!(skipped, vec!["/no/such/file.png".to_string()]);
    }

    #[test]
    fn test_build_native_content_parts_empty_text_no_images() {
        let (parts, skipped) = build_native_content_parts("   ", &[]);
        // no text part, no images, no neutral prompt
        assert!(parts.is_empty());
        assert!(skipped.is_empty());
    }

    #[test]
    fn test_build_native_content_parts_real_image_neutral_prompt() {
        // Write a tiny temp file, attach with empty text → neutral prompt prepended.
        let dir = std::env::temp_dir();
        let path = dir.join("ag_image_routing_test.png");
        std::fs::write(&path, b"\x89PNG\r\n\x1a\nfakecontent").unwrap();
        let p = path.to_string_lossy().to_string();

        let (parts, skipped) = build_native_content_parts("", &[p.clone()]);
        assert!(skipped.is_empty());
        // neutral text prompt + image_url
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "What do you see in this image?");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_build_native_content_parts_text_and_image() {
        let dir = std::env::temp_dir();
        let path = dir.join("ag_image_routing_test2.jpg");
        std::fs::write(&path, b"jpegbytes").unwrap();
        let p = path.to_string_lossy().to_string();

        let (parts, skipped) = build_native_content_parts("look here", &[p.clone()]);
        assert!(skipped.is_empty());
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "look here");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_to_data_url_missing() {
        assert!(file_to_data_url(Path::new("/no/such/path/xyz.png")).is_none());
    }
}
