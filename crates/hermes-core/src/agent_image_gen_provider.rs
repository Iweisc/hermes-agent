//! Image Generation Provider ABC - faithful port of agent/image_gen_provider.py.
//! Abstract base class plus helper layer. Concrete backends live in image_gen.rs.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use chrono::Local;
use serde_json::{Map, Value};
use sha1::{Digest, Sha1};

/// The set of aspect-ratio values a provider must accept.
pub const VALID_ASPECT_RATIOS: &[&str] = &["landscape", "square", "portrait"];

/// The aspect ratio used when none is supplied or the value is invalid.
pub const DEFAULT_ASPECT_RATIO: &str = "landscape";

/// Abstract interface for an image-generation backend. Implementors must
/// provide `name` and `generate`; everything else has defaults matching the
/// Python base class.
pub trait ImageGenProvider {
    /// Stable short identifier used in `image_gen.provider` config.
    fn name(&self) -> String;

    /// Human-readable label; defaults to the title-cased name.
    fn display_name(&self) -> String {
        title_case(&self.name())
    }

    /// Whether this provider can currently service calls. Default: true.
    fn is_available(&self) -> bool {
        true
    }

    /// Catalog entries for the model picker. Default: empty.
    fn list_models(&self) -> Vec<Value> {
        Vec::new()
    }

    /// Provider metadata for the picker. Default: minimal entry.
    fn get_setup_schema(&self) -> Value {
        let mut map = Map::new();
        map.insert("name".to_string(), Value::String(self.display_name()));
        map.insert("badge".to_string(), Value::String(String::new()));
        map.insert("tag".to_string(), Value::String(String::new()));
        map.insert("env_vars".to_string(), Value::Array(Vec::new()));
        Value::Object(map)
    }

    /// The default model id, or None. Defaults to the first model's `id`.
    fn default_model(&self) -> Option<String> {
        self.list_models().first().and_then(|model| {
            model
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
    }

    /// Generate an image. Returns a success_response / error_response. `kwargs`
    /// carries forward-compatible parameters; ignore unknown keys.
    fn generate(&self, prompt: &str, aspect_ratio: &str, kwargs: &Map<String, Value>) -> Value;
}

/// Clamp an aspect-ratio value to the valid set, defaulting to landscape.
pub fn resolve_aspect_ratio(value: Option<&str>) -> String {
    match value {
        Some(value) => {
            let normalized = value.trim().to_lowercase();
            if VALID_ASPECT_RATIOS.contains(&normalized.as_str()) {
                normalized
            } else {
                DEFAULT_ASPECT_RATIO.to_string()
            }
        }
        None => DEFAULT_ASPECT_RATIO.to_string(),
    }
}

/// Decode base64 image data and write under `$HERMES_HOME/cache/images/`.
pub fn save_b64_image(b64_data: &str, prefix: &str, extension: &str) -> Result<PathBuf, String> {
    let dir = images_cache_dir()?;
    save_b64_image_in(&dir, b64_data, prefix, extension)
}

/// Like `save_b64_image` but writes into an explicit directory.
pub fn save_b64_image_in(
    dir: &Path,
    b64_data: &str,
    prefix: &str,
    extension: &str,
) -> Result<PathBuf, String> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64_data.trim())
        .map_err(|error| format!("invalid base64 image data: {error}"))?;
    fs::create_dir_all(dir)
        .map_err(|error| format!("creating {} failed: {error}", dir.display()))?;
    let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let short = short_token(prefix, raw.len());
    let path = dir.join(format!("{prefix}_{ts}_{short}.{extension}"));
    fs::write(&path, &raw)
        .map_err(|error| format!("writing {} failed: {error}", path.display()))?;
    Ok(path)
}

/// Build a uniform success response object. `extra` uses setdefault semantics.
pub fn success_response(
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

/// Build a uniform error response object.
pub fn error_response(
    error: &str,
    error_type: &str,
    provider: &str,
    model: &str,
    prompt: &str,
    aspect_ratio: &str,
) -> Value {
    let mut payload = Map::new();
    payload.insert("success".to_string(), Value::Bool(false));
    payload.insert("image".to_string(), Value::Null);
    payload.insert("error".to_string(), Value::String(error.to_string()));
    payload.insert(
        "error_type".to_string(),
        Value::String(error_type.to_string()),
    );
    payload.insert("model".to_string(), Value::String(model.to_string()));
    payload.insert("prompt".to_string(), Value::String(prompt.to_string()));
    payload.insert(
        "aspect_ratio".to_string(),
        Value::String(aspect_ratio.to_string()),
    );
    payload.insert("provider".to_string(), Value::String(provider.to_string()));
    Value::Object(payload)
}

/// Title-case like Python's str.title(): first alphabetic char of each run
/// uppercased, rest lowercased.
fn title_case(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut prev_alpha = false;
    for ch in value.chars() {
        if ch.is_alphabetic() {
            if prev_alpha {
                result.extend(ch.to_lowercase());
            } else {
                result.extend(ch.to_uppercase());
            }
            prev_alpha = true;
        } else {
            result.push(ch);
            prev_alpha = false;
        }
    }
    result
}

/// Resolve `$HERMES_HOME/cache/images`, creating parents as needed.
fn images_cache_dir() -> Result<PathBuf, String> {
    let home = env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"));
    let dir = home.join("cache").join("images");
    fs::create_dir_all(&dir)
        .map_err(|error| format!("creating {} failed: {error}", dir.display()))?;
    Ok(dir)
}

/// Produce an 8-hex-char token replacing Python's uuid4().hex[:8].
fn short_token(prefix: &str, payload_len: usize) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let mut hasher = Sha1::new();
    hasher.update(nanos.to_le_bytes());
    hasher.update(pid.to_le_bytes());
    hasher.update(prefix.as_bytes());
    hasher.update((payload_len as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut token = String::with_capacity(8);
    for byte in digest.iter().take(4) {
        token.push_str(&format!("{byte:02x}"));
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    struct StubProvider;
    impl ImageGenProvider for StubProvider {
        fn name(&self) -> String {
            "openai-codex".to_string()
        }
        fn generate(&self, prompt: &str, aspect_ratio: &str, _k: &Map<String, Value>) -> Value {
            success_response(
                "https://example.com/x.png",
                "stub-model",
                prompt,
                aspect_ratio,
                &self.name(),
                None,
            )
        }
    }

    struct ModelProvider;
    impl ImageGenProvider for ModelProvider {
        fn name(&self) -> String {
            "fal".to_string()
        }
        fn list_models(&self) -> Vec<Value> {
            vec![json!({"id": "flux", "display": "Flux"}), json!({"id": "klein"})]
        }
        fn generate(&self, prompt: &str, aspect_ratio: &str, _k: &Map<String, Value>) -> Value {
            error_response("nope", "provider_error", &self.name(), "", prompt, aspect_ratio)
        }
    }

    #[test]
    fn resolve_aspect_ratio_accepts_valid_values() {
        assert_eq!(resolve_aspect_ratio(Some("square")), "square");
        assert_eq!(resolve_aspect_ratio(Some("portrait")), "portrait");
        assert_eq!(resolve_aspect_ratio(Some("landscape")), "landscape");
    }

    #[test]
    fn resolve_aspect_ratio_normalizes_case_and_whitespace() {
        assert_eq!(resolve_aspect_ratio(Some("  SQUARE ")), "square");
        assert_eq!(resolve_aspect_ratio(Some("Portrait")), "portrait");
    }

    #[test]
    fn resolve_aspect_ratio_coerces_invalid_and_none_to_default() {
        assert_eq!(resolve_aspect_ratio(Some("wide")), DEFAULT_ASPECT_RATIO);
        assert_eq!(resolve_aspect_ratio(Some("")), DEFAULT_ASPECT_RATIO);
        assert_eq!(resolve_aspect_ratio(None), DEFAULT_ASPECT_RATIO);
    }

    #[test]
    fn display_name_title_cases_like_python() {
        assert_eq!(StubProvider.display_name(), "Openai-Codex");
        assert_eq!(ModelProvider.display_name(), "Fal");
    }

    #[test]
    fn default_provider_defaults_match_python() {
        let stub = StubProvider;
        assert!(stub.is_available());
        assert!(stub.list_models().is_empty());
        assert_eq!(stub.default_model(), None);
        let schema = stub.get_setup_schema();
        assert_eq!(schema["name"], "Openai-Codex");
        assert_eq!(schema["badge"], "");
        assert_eq!(schema["tag"], "");
        assert_eq!(schema["env_vars"], json!([]));
    }

    #[test]
    fn default_model_reads_first_model_id() {
        assert_eq!(ModelProvider.default_model(), Some("flux".to_string()));
    }

    #[test]
    fn success_response_has_expected_shape() {
        let value = success_response("img", "m", "p", "square", "fal", None);
        assert_eq!(value["success"], true);
        assert_eq!(value["image"], "img");
        assert_eq!(value["model"], "m");
        assert_eq!(value["prompt"], "p");
        assert_eq!(value["aspect_ratio"], "square");
        assert_eq!(value["provider"], "fal");
        assert!(value.get("error").is_none());
    }

    #[test]
    fn success_response_extra_uses_setdefault_semantics() {
        let mut extra = Map::new();
        extra.insert("provider".to_string(), json!("evil"));
        extra.insert("revised_prompt".to_string(), json!("rewritten"));
        let value = success_response("img", "m", "p", "square", "fal", Some(extra));
        assert_eq!(value["provider"], "fal");
        assert_eq!(value["revised_prompt"], "rewritten");
    }

    #[test]
    fn error_response_has_expected_shape() {
        let value = error_response("boom", "provider_error", "fal", "", "p", "landscape");
        assert_eq!(value["success"], false);
        assert_eq!(value["image"], Value::Null);
        assert_eq!(value["error"], "boom");
        assert_eq!(value["error_type"], "provider_error");
        assert_eq!(value["model"], "");
        assert_eq!(value["prompt"], "p");
        assert_eq!(value["aspect_ratio"], "landscape");
        assert_eq!(value["provider"], "fal");
    }

    #[test]
    fn provider_generate_round_trips_through_helpers() {
        let value = StubProvider.generate("a cat", "square", &Map::new());
        assert_eq!(value["success"], true);
        assert_eq!(value["provider"], "openai-codex");
        let value = ModelProvider.generate("a cat", "portrait", &Map::new());
        assert_eq!(value["success"], false);
        assert_eq!(value["aspect_ratio"], "portrait");
    }

    #[test]
    fn save_b64_image_in_writes_decoded_bytes_with_expected_filename() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("cache").join("images");
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"hello");
        let path = save_b64_image_in(&dir, &b64, "openai", "png").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        assert!(path.starts_with(&dir));
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("openai_"));
        assert!(name.ends_with(".png"));
        let stem = name.strip_suffix(".png").unwrap();
        let parts: Vec<&str> = stem.split('_').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "openai");
        assert_eq!(parts[1].len(), 8);
        assert_eq!(parts[2].len(), 6);
        assert_eq!(parts[3].len(), 8);
        assert!(parts[3].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn save_b64_image_in_rejects_invalid_base64() {
        let temp = TempDir::new().unwrap();
        let result = save_b64_image_in(temp.path(), "not!!base64", "x", "png");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid base64"));
    }
}
