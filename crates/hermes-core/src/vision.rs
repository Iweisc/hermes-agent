use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use serde_json::{Value, json};

use crate::agent::{build_http_client_with_timeout, request_model_text};
use crate::tools::{ToolRuntime, tool_error, tool_result};
use crate::web::{
    BLOCKED_URL_SECRET_ERROR, PRIVATE_URL_ERROR, check_website_access, contains_embedded_secret,
    is_safe_url, parse_http_url,
};
use crate::{HermesContext, HermesError, LoadedConfig, ModelOverrides};

const DEFAULT_VISION_TIMEOUT_SECS: f64 = 120.0;
const DEFAULT_DOWNLOAD_TIMEOUT_SECS: f64 = 30.0;
const DEFAULT_VISION_TEMPERATURE: f64 = 0.1;
const MAX_BASE64_BYTES: usize = 20 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const MAX_REMOTE_BYTES: usize = 24 * 1024 * 1024;

#[derive(Debug, Clone)]
struct VisionSettings {
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    timeout_secs: f64,
    download_timeout_secs: f64,
    _temperature: f64,
}

impl VisionSettings {
    fn from_loaded(loaded: &LoadedConfig) -> Self {
        let provider = yaml_string(loaded, &["auxiliary", "vision", "provider"]);
        let model = yaml_string(loaded, &["auxiliary", "vision", "model"]);
        let base_url = yaml_string(loaded, &["auxiliary", "vision", "base_url"]);
        let api_key = yaml_string(loaded, &["auxiliary", "vision", "api_key"]);
        let timeout_secs = yaml_number(loaded, &["auxiliary", "vision", "timeout"])
            .unwrap_or(DEFAULT_VISION_TIMEOUT_SECS)
            .max(1.0);
        let download_timeout_secs =
            yaml_number(loaded, &["auxiliary", "vision", "download_timeout"])
                .unwrap_or(DEFAULT_DOWNLOAD_TIMEOUT_SECS)
                .max(1.0);
        let temperature = yaml_number(loaded, &["auxiliary", "vision", "temperature"])
            .unwrap_or(DEFAULT_VISION_TEMPERATURE);
        Self {
            provider,
            model,
            base_url,
            api_key,
            timeout_secs,
            download_timeout_secs,
            _temperature: temperature,
        }
    }

    fn to_overrides(&self) -> ModelOverrides {
        ModelOverrides {
            model: self.model.clone(),
            provider: self.provider.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            api_mode: None,
        }
    }
}

pub fn vision_analyze_schema() -> Value {
    json!({
        "name": "vision_analyze",
        "description": "Inspect an image from a URL, file path, or file:// URI when you need closer detail than what's visible in the conversation. Use it for local screenshots, tool-generated images, browser captures, or remote images that need deeper analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Image URL (http/https), file:// URI, or local file path to analyze."
                },
                "question": {
                    "type": "string",
                    "description": "Your specific question about the image. The tool will also request a complete image description."
                }
            },
            "required": ["image_url", "question"]
        }
    })
}

pub fn handle_vision_analyze(args: &Value, runtime: &ToolRuntime) -> String {
    let image_url = match required_non_empty_string(args, "image_url") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let question = match required_non_empty_string(args, "question") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match run_vision_analysis(&image_url, &question, runtime) {
        Ok(result) => tool_result(json!({
            "success": true,
            "analysis": result,
        })),
        Err(error) => tool_result(classify_vision_error(error)),
    }
}

pub(crate) fn run_vision_analysis(
    image_url: &str,
    question: &str,
    runtime: &ToolRuntime,
) -> Result<String, HermesError> {
    let context =
        HermesContext::detect().with_hermes_home_env(Some(runtime.hermes_home().to_path_buf()));
    let loaded = context.load_config_document()?;
    let settings = VisionSettings::from_loaded(&loaded);
    let (image_path, should_cleanup) =
        resolve_image_source(image_url, runtime, settings.download_timeout_secs)?;

    let data_url = match image_path_to_data_url(&image_path) {
        Ok(value) => value,
        Err(error) => {
            if should_cleanup {
                let _ = fs::remove_file(&image_path);
            }
            return Err(HermesError::State {
                action: "analyzing image",
                detail: error,
            });
        }
    };
    let runtime_model = context.resolve_model_runtime(&loaded, &settings.to_overrides())?;

    let prompt = format!(
        "Fully describe and explain everything about this image, then answer the following question:\n\n{}",
        question.trim()
    );
    let messages = vec![json!({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                }
            }
        ]
    })];

    let analysis_result = (|| {
        let client = build_http_client_with_timeout(settings.timeout_secs.ceil() as u64)?;
        let first = request_model_text(&client, &runtime_model, &messages)?;
        if let Some(text) = first {
            return Ok(text);
        }
        request_model_text(&client, &runtime_model, &messages)?.ok_or_else(|| HermesError::State {
            action: "analyzing image",
            detail: "Vision model returned no text content.".to_string(),
        })
    })();

    if should_cleanup {
        let _ = fs::remove_file(&image_path);
    }
    analysis_result
}

fn resolve_image_source(
    image_url: &str,
    runtime: &ToolRuntime,
    download_timeout_secs: f64,
) -> Result<(PathBuf, bool), HermesError> {
    let trimmed = image_url.trim();
    if trimmed.is_empty() {
        return Err(HermesError::State {
            action: "analyzing image",
            detail: "image_url must not be empty.".to_string(),
        });
    }

    let local_hint = trimmed
        .strip_prefix("file://")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| trimmed.to_string());
    let local_path = runtime
        .resolve_path(&local_hint)
        .map_err(|detail| HermesError::State {
            action: "analyzing image",
            detail,
        })?;
    if local_path.is_file() {
        return Ok((local_path, false));
    }

    if contains_embedded_secret(trimmed) {
        return Err(HermesError::State {
            action: "analyzing image",
            detail: BLOCKED_URL_SECRET_ERROR.to_string(),
        });
    }

    let parsed = parse_http_url(trimmed).map_err(|detail| HermesError::State {
        action: "analyzing image",
        detail: detail.to_string(),
    })?;
    if !is_safe_url(&parsed, runtime) {
        return Err(HermesError::State {
            action: "analyzing image",
            detail: PRIVATE_URL_ERROR.to_string(),
        });
    }
    if let Some(blocked) = check_website_access(&parsed, runtime) {
        return Err(HermesError::State {
            action: "analyzing image",
            detail: format!(
                "Blocked by website policy for host '{}' (rule: {})",
                blocked.host, blocked.rule
            ),
        });
    }

    let cache_dir = runtime.hermes_home().join("cache/vision");
    fs::create_dir_all(&cache_dir).map_err(|source| HermesError::Io {
        action: "creating",
        path: cache_dir.clone(),
        source,
    })?;
    let temp_path = cache_dir.join(format!("vision_{:x}.img", unix_ts_nanos()));
    download_image(&parsed, &temp_path, runtime, download_timeout_secs)?;
    Ok((temp_path, true))
}

fn download_image(
    start_url: &Url,
    destination: &Path,
    runtime: &ToolRuntime,
    timeout_secs: f64,
) -> Result<(), HermesError> {
    let client = Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs.max(1.0)))
        .redirect(Policy::none())
        .build()
        .map_err(|error| HermesError::State {
            action: "downloading image",
            detail: error.to_string(),
        })?;

    let mut current = start_url.clone();
    for _ in 0..=MAX_REDIRECTS {
        let response = client
            .get(current.clone())
            .send()
            .map_err(|error| HermesError::State {
                action: "downloading image",
                detail: error.to_string(),
            })?;

        if response.status().is_redirection() {
            let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
                return Err(HermesError::State {
                    action: "downloading image",
                    detail: "Redirect response was missing a Location header.".to_string(),
                });
            };
            let location = location.to_str().map_err(|_| HermesError::State {
                action: "downloading image",
                detail: "Redirect Location was not valid UTF-8.".to_string(),
            })?;
            let next = current.join(location).map_err(|error| HermesError::State {
                action: "downloading image",
                detail: format!("Invalid redirect target: {error}"),
            })?;
            if contains_embedded_secret(next.as_str()) {
                return Err(HermesError::State {
                    action: "downloading image",
                    detail: BLOCKED_URL_SECRET_ERROR.to_string(),
                });
            }
            if !is_safe_url(&next, runtime) {
                return Err(HermesError::State {
                    action: "downloading image",
                    detail: PRIVATE_URL_ERROR.to_string(),
                });
            }
            if let Some(blocked) = check_website_access(&next, runtime) {
                return Err(HermesError::State {
                    action: "downloading image",
                    detail: format!(
                        "Blocked by website policy for host '{}' (rule: {})",
                        blocked.host, blocked.rule
                    ),
                });
            }
            current = next;
            continue;
        }

        if !response.status().is_success() {
            return Err(HermesError::State {
                action: "downloading image",
                detail: format!("HTTP {} while fetching image.", response.status().as_u16()),
            });
        }

        if let Some(final_blocked) = check_website_access(response.url(), runtime) {
            return Err(HermesError::State {
                action: "downloading image",
                detail: format!(
                    "Blocked by website policy for host '{}' (rule: {})",
                    final_blocked.host, final_blocked.rule
                ),
            });
        }
        if !is_safe_url(response.url(), runtime) {
            return Err(HermesError::State {
                action: "downloading image",
                detail: PRIVATE_URL_ERROR.to_string(),
            });
        }

        let bytes = response.bytes().map_err(|error| HermesError::State {
            action: "downloading image",
            detail: error.to_string(),
        })?;
        if bytes.len() > MAX_REMOTE_BYTES {
            return Err(HermesError::State {
                action: "downloading image",
                detail: format!(
                    "Downloaded image exceeded the size limit ({} bytes).",
                    bytes.len()
                ),
            });
        }

        let mut file = fs::File::create(destination).map_err(|source| HermesError::Io {
            action: "creating",
            path: destination.to_path_buf(),
            source,
        })?;
        file.write_all(&bytes).map_err(|source| HermesError::Io {
            action: "writing",
            path: destination.to_path_buf(),
            source,
        })?;
        return Ok(());
    }

    Err(HermesError::State {
        action: "downloading image",
        detail: "Too many redirects while fetching image.".to_string(),
    })
}

fn image_path_to_data_url(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    let Some(mime_type) = detect_image_mime(&bytes, path) else {
        return Err("Only real image files are supported for vision analysis.".to_string());
    };
    let encoded = BASE64.encode(&bytes);
    let data_url = format!("data:{mime_type};base64,{encoded}");
    if data_url.len() > MAX_BASE64_BYTES {
        return Err(format!(
            "Image too large for vision API: base64 payload is {:.1} MB (limit {:.0} MB).",
            data_url.len() as f64 / (1024.0 * 1024.0),
            MAX_BASE64_BYTES as f64 / (1024.0 * 1024.0),
        ));
    }
    Ok(data_url)
}

fn detect_image_mime(bytes: &[u8], path: &Path) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }

    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Some("image/png"),
        Some("jpg") | Some("jpeg") => Some("image/jpeg"),
        Some("gif") => Some("image/gif"),
        Some("bmp") => Some("image/bmp"),
        Some("webp") => Some("image/webp"),
        _ => None,
    }
}

pub(crate) fn classify_vision_error(error: HermesError) -> Value {
    let detail = error.to_string();
    let lower = detail.to_ascii_lowercase();
    let analysis = if lower.contains("payment required")
        || lower.contains("insufficient")
        || lower.contains("credits")
        || lower.contains("billing")
    {
        format!(
            "Insufficient credits or payment required. Please top up your API provider account and try again. Error: {detail}"
        )
    } else if lower.contains("does not support")
        || lower.contains("not support image")
        || lower.contains("multimodal")
    {
        format!("The configured runtime does not appear to support vision input. Error: {detail}")
    } else if lower.contains("invalid_request")
        || lower.contains("image_url")
        || lower.contains("rejected the image")
    {
        format!(
            "The vision API rejected the image. This can happen when the image format is unsupported or the payload is too large. Error: {detail}"
        )
    } else {
        format!(
            "There was a problem with the request and the image could not be analyzed. Error: {detail}"
        )
    };

    json!({
        "success": false,
        "error": detail,
        "analysis": analysis,
    })
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn yaml_string(loaded: &LoadedConfig, keys: &[&str]) -> Option<String> {
    loaded
        .cfg_get(keys)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn yaml_number(loaded: &LoadedConfig, keys: &[&str]) -> Option<f64> {
    match loaded.cfg_get(keys) {
        Some(serde_yaml::Value::Number(value)) => value.as_f64(),
        Some(serde_yaml::Value::String(value)) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
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
    use std::thread;

    use super::*;

    use tempfile::TempDir;

    fn runtime_for(home: &Path, cwd: &Path) -> ToolRuntime {
        ToolRuntime::new(cwd.to_path_buf()).with_hermes_home(home.to_path_buf())
    }

    fn write_config(home: &Path, body: &str) {
        fs::write(home.join("config.yaml"), body).unwrap();
    }

    fn serve_chat_completion(
        body: Value,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let handle = thread::spawn(move || {
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
            let headers = String::from_utf8_lossy(&request[..header_end]);
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
            captured_clone
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&body_bytes).to_string());

            let response_body = body.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (format!("http://{addr}/v1"), captured, handle)
    }

    #[test]
    fn local_non_image_file_is_rejected_before_llm_call() {
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(temp.path(), temp.path());
        let text_path = temp.path().join("secret.txt");
        fs::write(&text_path, "TOP-SECRET=1\n").unwrap();
        let result = serde_json::from_str::<Value>(&handle_vision_analyze(
            &json!({"image_url": text_path.display().to_string(), "question": "extract text"}),
            &runtime,
        ))
        .unwrap();
        assert_eq!(result.get("success").and_then(Value::as_bool), Some(false));
        assert!(
            result
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("Only real image files are supported")
        );
    }

    #[test]
    fn blocked_remote_url_short_circuits_before_download() {
        let temp = TempDir::new().unwrap();
        write_config(
            temp.path(),
            "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: http://127.0.0.1:1/v1\n  api_key: test-key\nsecurity:\n  website_blocklist:\n    enabled: true\n    domains:\n      - blocked.test\n",
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let result = serde_json::from_str::<Value>(&handle_vision_analyze(
            &json!({"image_url": "https://blocked.test/cat.png", "question": "describe"}),
            &runtime,
        ))
        .unwrap();
        assert_eq!(result.get("success").and_then(Value::as_bool), Some(false));
        assert!(
            result
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("Blocked by website policy")
        );
    }

    #[test]
    fn local_image_file_calls_multimodal_chat_completion() {
        let temp = TempDir::new().unwrap();
        let (base_url, captured, server) = serve_chat_completion(json!({
            "choices": [{
                "message": {
                    "content": "A tiny PNG test image."
                },
                "finish_reason": "stop"
            }]
        }));
        write_config(
            temp.path(),
            &format!(
                "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: {base_url}\n  api_key: test-key\nauxiliary:\n  vision:\n    timeout: 77\n    temperature: 1.0\n"
            ),
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let image_path = temp.path().join("test.png");
        fs::write(
            &image_path,
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();

        let result = serde_json::from_str::<Value>(&handle_vision_analyze(
            &json!({"image_url": image_path.display().to_string(), "question": "What is shown?"}),
            &runtime,
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(
            result.get("analysis").and_then(Value::as_str),
            Some("A tiny PNG test image.")
        );

        let requests = captured.lock().unwrap();
        let payload = serde_json::from_str::<Value>(&requests[0]).unwrap();
        assert_eq!(
            payload.get("model").and_then(Value::as_str),
            Some("gpt-4.1-mini")
        );
        let content = payload["messages"][0]["content"].as_array().unwrap();
        assert!(
            content[0]["text"]
                .as_str()
                .unwrap()
                .contains("Fully describe and explain everything about this image")
        );
        assert!(
            content[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn file_uri_is_treated_as_local_path() {
        let temp = TempDir::new().unwrap();
        let (base_url, _, server) = serve_chat_completion(json!({
            "choices": [{
                "message": {
                    "content": "A local file URI image."
                },
                "finish_reason": "stop"
            }]
        }));
        write_config(
            temp.path(),
            &format!(
                "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: {base_url}\n  api_key: test-key\n"
            ),
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let image_path = temp.path().join("photo.png");
        fs::write(
            &image_path,
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();

        let result = serde_json::from_str::<Value>(&handle_vision_analyze(
            &json!({"image_url": format!("file://{}", image_path.display()), "question": "describe this"}),
            &runtime,
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(
            result.get("analysis").and_then(Value::as_str),
            Some("A local file URI image.")
        );
    }
}
