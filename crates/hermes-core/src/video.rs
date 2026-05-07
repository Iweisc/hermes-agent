use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use serde_json::{Value, json};

use crate::agent::{build_http_client_with_timeout, request_model_text};
use crate::tools::{ToolRuntime, tool_result};
use crate::web::{
    BLOCKED_URL_SECRET_ERROR, PRIVATE_URL_ERROR, check_website_access, contains_embedded_secret,
    is_safe_url, parse_http_url,
};
use crate::{HermesContext, HermesError, LoadedConfig, ModelOverrides};

const DEFAULT_VIDEO_TIMEOUT_SECS: f64 = 180.0;
const DEFAULT_DOWNLOAD_TIMEOUT_SECS: f64 = 60.0;
const MAX_VIDEO_BASE64_BYTES: usize = 50 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const MAX_REMOTE_BYTES: usize = 50 * 1024 * 1024;

const VIDEO_MIME_TYPES: &[(&str, &str)] = &[
    (".mp4", "video/mp4"),
    (".m4v", "video/mp4"),
    (".avi", "video/mp4"),
    (".mkv", "video/mp4"),
    (".webm", "video/webm"),
    (".mov", "video/mov"),
    (".mpeg", "video/mpeg"),
    (".mpg", "video/mpeg"),
];

#[derive(Debug, Clone)]
struct VideoSettings {
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    timeout_secs: f64,
    download_timeout_secs: f64,
}

impl VideoSettings {
    fn from_loaded(loaded: &LoadedConfig) -> Self {
        let env_video_model = env::var("AUXILIARY_VIDEO_MODEL")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let env_vision_model = env::var("AUXILIARY_VISION_MODEL")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let provider = yaml_string(loaded, &["auxiliary", "video", "provider"])
            .or_else(|| yaml_string(loaded, &["auxiliary", "vision", "provider"]));
        let model = env_video_model
            .or(env_vision_model)
            .or_else(|| yaml_string(loaded, &["auxiliary", "video", "model"]))
            .or_else(|| yaml_string(loaded, &["auxiliary", "vision", "model"]));
        let base_url = yaml_string(loaded, &["auxiliary", "video", "base_url"])
            .or_else(|| yaml_string(loaded, &["auxiliary", "vision", "base_url"]));
        let api_key = yaml_string(loaded, &["auxiliary", "video", "api_key"])
            .or_else(|| yaml_string(loaded, &["auxiliary", "vision", "api_key"]));
        let timeout_secs = yaml_number(loaded, &["auxiliary", "video", "timeout"])
            .or_else(|| yaml_number(loaded, &["auxiliary", "vision", "timeout"]))
            .unwrap_or(DEFAULT_VIDEO_TIMEOUT_SECS)
            .max(1.0);
        let download_timeout_secs =
            yaml_number(loaded, &["auxiliary", "video", "download_timeout"])
                .or_else(|| yaml_number(loaded, &["auxiliary", "vision", "download_timeout"]))
                .unwrap_or(DEFAULT_DOWNLOAD_TIMEOUT_SECS)
                .max(1.0);
        Self {
            provider,
            model,
            base_url,
            api_key,
            timeout_secs,
            download_timeout_secs,
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

pub fn video_analyze_schema() -> Value {
    json!({
        "name": "video_analyze",
        "description": "Analyze a video from an HTTP/HTTPS URL, local file path, or file:// URI using a video-capable multimodal model. Use this for video files; for still images use vision_analyze instead. Supports mp4, webm, mov, avi, mkv, mpeg, and mpg inputs. Large videos may be slow or rejected; the Rust runtime enforces a roughly 50 MB payload cap.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "Video URL (http/https), file:// URI, or local file path to analyze."
                },
                "question": {
                    "type": "string",
                    "description": "Your specific question about the video. The tool will first ask the model for a full description of the clip."
                }
            },
            "required": ["video_url", "question"]
        }
    })
}

pub fn handle_video_analyze(args: &Value, runtime: &ToolRuntime) -> String {
    let video_url = match required_non_empty_string(args, "video_url") {
        Ok(value) => value,
        Err(error) => return tool_result(classify_video_error(error)),
    };
    let question = match required_non_empty_string(args, "question") {
        Ok(value) => value,
        Err(error) => return tool_result(classify_video_error(error)),
    };
    let prompt = format!(
        "Fully describe and explain everything happening in this video, including visual content, motion, audio cues, text overlays, and scene transitions. Then answer the following question:\n\n{}",
        question.trim()
    );

    match run_video_analysis(&video_url, &prompt, runtime) {
        Ok(result) => tool_result(json!({
            "success": true,
            "analysis": result,
        })),
        Err(error) => tool_result(classify_video_error(error.to_string())),
    }
}

fn run_video_analysis(
    video_url: &str,
    prompt: &str,
    runtime: &ToolRuntime,
) -> Result<String, HermesError> {
    let context =
        HermesContext::detect().with_hermes_home_env(Some(runtime.hermes_home().to_path_buf()));
    let loaded = context.load_config_document()?;
    let settings = VideoSettings::from_loaded(&loaded);
    let (video_path, should_cleanup) =
        resolve_video_source(video_url, runtime, settings.download_timeout_secs)?;

    let mime_type = detect_video_mime_type(&video_path).ok_or_else(|| HermesError::State {
        action: "analyzing video",
        detail: format!(
            "Unsupported video format: '{}'. Supported: mp4, webm, mov, avi, mkv, mpeg, mpg.",
            video_path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
        ),
    })?;

    let data_url = match video_path_to_data_url(&video_path, mime_type) {
        Ok(value) => value,
        Err(error) => {
            if should_cleanup {
                let _ = fs::remove_file(&video_path);
            }
            return Err(HermesError::State {
                action: "analyzing video",
                detail: error,
            });
        }
    };

    let runtime_model = context.resolve_model_runtime(&loaded, &settings.to_overrides())?;
    if runtime_model.api_mode != "chat_completions" {
        if should_cleanup {
            let _ = fs::remove_file(&video_path);
        }
        return Err(HermesError::State {
            action: "analyzing video",
            detail: format!(
                "The configured runtime uses api_mode='{}'. Rust video_analyze currently requires chat_completions.",
                runtime_model.api_mode
            ),
        });
    }

    let messages = vec![json!({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt.trim(),
            },
            {
                "type": "video_url",
                "video_url": {
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
            action: "analyzing video",
            detail: "Video model returned no text content.".to_string(),
        })
    })();

    if should_cleanup {
        let _ = fs::remove_file(&video_path);
    }

    analysis_result
}

fn resolve_video_source(
    video_url: &str,
    runtime: &ToolRuntime,
    download_timeout_secs: f64,
) -> Result<(PathBuf, bool), HermesError> {
    let trimmed = video_url.trim();
    if trimmed.is_empty() {
        return Err(HermesError::State {
            action: "analyzing video",
            detail: "video_url must not be empty.".to_string(),
        });
    }

    let local_hint = trimmed
        .strip_prefix("file://")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| trimmed.to_string());
    let local_path = runtime
        .resolve_path(&local_hint)
        .map_err(|detail| HermesError::State {
            action: "analyzing video",
            detail,
        })?;
    if local_path.is_file() {
        return Ok((local_path, false));
    }

    if contains_embedded_secret(trimmed) {
        return Err(HermesError::State {
            action: "analyzing video",
            detail: BLOCKED_URL_SECRET_ERROR.to_string(),
        });
    }

    let parsed = parse_http_url(trimmed).map_err(|detail| HermesError::State {
        action: "analyzing video",
        detail: detail.to_string(),
    })?;
    if !is_safe_url(&parsed, runtime) {
        return Err(HermesError::State {
            action: "analyzing video",
            detail: PRIVATE_URL_ERROR.to_string(),
        });
    }
    if let Some(blocked) = check_website_access(&parsed, runtime) {
        return Err(HermesError::State {
            action: "analyzing video",
            detail: format!(
                "Blocked by website policy for host '{}' (rule: {})",
                blocked.host, blocked.rule
            ),
        });
    }

    let cache_dir = runtime.hermes_home().join("cache/video");
    fs::create_dir_all(&cache_dir).map_err(|source| HermesError::Io {
        action: "creating",
        path: cache_dir.clone(),
        source,
    })?;
    let temp_path = cache_dir.join(format!("video_{:x}.mp4", unix_ts_nanos()));
    download_video(&parsed, &temp_path, runtime, download_timeout_secs)?;
    Ok((temp_path, true))
}

fn download_video(
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
            action: "downloading video",
            detail: error.to_string(),
        })?;

    let mut current = start_url.clone();
    for _ in 0..MAX_REDIRECTS {
        let response = client
            .get(current.clone())
            .header(
                "User-Agent",
                "Mozilla/5.0 (X11; Linux x86_64) Hermes-Agent-Rust/0.12",
            )
            .send()
            .map_err(|error| HermesError::State {
                action: "downloading video",
                detail: error.to_string(),
            })?;

        let status = response.status();
        if status.is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| HermesError::State {
                    action: "downloading video",
                    detail: "Redirect response missing Location header.".to_string(),
                })?;
            let next = current.join(location).map_err(|error| HermesError::State {
                action: "downloading video",
                detail: error.to_string(),
            })?;
            if !is_safe_url(&next, runtime) {
                return Err(HermesError::State {
                    action: "downloading video",
                    detail: PRIVATE_URL_ERROR.to_string(),
                });
            }
            if let Some(blocked) = check_website_access(&next, runtime) {
                return Err(HermesError::State {
                    action: "downloading video",
                    detail: format!(
                        "Blocked by website policy for host '{}' (rule: {})",
                        blocked.host, blocked.rule
                    ),
                });
            }
            current = next;
            continue;
        }

        if !status.is_success() {
            return Err(HermesError::State {
                action: "downloading video",
                detail: format!("HTTP {} while fetching {}", status.as_u16(), current),
            });
        }

        if let Some(length) = response.content_length()
            && length > MAX_REMOTE_BYTES as u64
        {
            return Err(HermesError::State {
                action: "analyzing video",
                detail: format!("Video too large ({length} bytes, max {MAX_REMOTE_BYTES} bytes)."),
            });
        }

        let bytes = response.bytes().map_err(|error| HermesError::State {
            action: "downloading video",
            detail: error.to_string(),
        })?;
        if bytes.len() > MAX_REMOTE_BYTES {
            return Err(HermesError::State {
                action: "analyzing video",
                detail: format!(
                    "Video too large ({} bytes, max {} bytes).",
                    bytes.len(),
                    MAX_REMOTE_BYTES
                ),
            });
        }
        fs::write(destination, &bytes).map_err(|source| HermesError::Io {
            action: "writing",
            path: destination.to_path_buf(),
            source,
        })?;
        return Ok(());
    }

    Err(HermesError::State {
        action: "downloading video",
        detail: format!("Too many redirects while fetching {}", start_url),
    })
}

fn video_path_to_data_url(video_path: &Path, mime_type: &str) -> Result<String, String> {
    let data = fs::read(video_path)
        .map_err(|error| format!("reading {} failed: {error}", video_path.display()))?;
    let encoded = BASE64.encode(data);
    let result = format!("data:{mime_type};base64,{encoded}");
    if result.len() > MAX_VIDEO_BASE64_BYTES {
        return Err(format!(
            "Video too large for API: base64 payload is {:.1} MB (limit {:.0} MB). Compress or trim the video and retry.",
            result.len() as f64 / (1024.0 * 1024.0),
            MAX_VIDEO_BASE64_BYTES as f64 / (1024.0 * 1024.0)
        ));
    }
    Ok(result)
}

fn detect_video_mime_type(video_path: &Path) -> Option<&'static str> {
    let extension = video_path.extension()?.to_str()?.to_ascii_lowercase();
    let dotted = format!(".{extension}");
    VIDEO_MIME_TYPES
        .iter()
        .find_map(|(suffix, mime)| (*suffix == dotted).then_some(*mime))
}

fn classify_video_error(detail: impl Into<String>) -> Value {
    let detail = detail.into();
    let lower = detail.to_ascii_lowercase();
    let analysis = if lower.contains("too large")
        || lower.contains("request too large")
        || lower.contains("413")
        || lower.contains("payload")
        || lower.contains("size limit")
    {
        format!(
            "The video is too large for the current runtime or provider. Compress or trim the clip and retry. Error: {}",
            detail
        )
    } else if lower.contains("unsupported video format") {
        format!(
            "The provided video format is not supported. Error: {}",
            detail
        )
    } else if lower.contains("invalid video source")
        || lower.contains("must not be empty")
        || lower.contains("no such file")
        || lower.contains("not found")
    {
        format!(
            "Invalid video source. Provide an HTTP/HTTPS URL, file:// URI, or valid local file path. Error: {}",
            detail
        )
    } else if lower.contains("does not support")
        || lower.contains("video_url")
        || lower.contains("multimodal")
        || lower.contains("chat_completions")
    {
        format!(
            "The configured model or provider does not support native video analysis in the Rust runtime. Use a video-capable chat-completions model such as a Gemini/OpenRouter setup. Error: {}",
            detail
        )
    } else {
        format!(
            "There was a problem with the request and the video could not be analyzed. Error: {}",
            detail
        )
    };
    json!({
        "success": false,
        "error": detail,
        "analysis": analysis,
    })
}

fn yaml_string(loaded: &LoadedConfig, path: &[&str]) -> Option<String> {
    let mut cursor = &loaded.raw;
    for segment in path {
        cursor = cursor.get(*segment)?;
    }
    cursor.as_str().map(str::trim).and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

fn yaml_number(loaded: &LoadedConfig, path: &[&str]) -> Option<f64> {
    let mut cursor = &loaded.raw;
    for segment in path {
        cursor = cursor.get(*segment)?;
    }
    cursor.as_f64()
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn unix_ts_nanos() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
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
    fn local_unsupported_video_file_is_rejected_before_llm_call() {
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(temp.path(), temp.path());
        let video_path = temp.path().join("clip.flv");
        fs::write(&video_path, b"not-a-real-video").unwrap();

        let result = serde_json::from_str::<Value>(&handle_video_analyze(
            &json!({"video_url": video_path.display().to_string(), "question": "what is this?"}),
            &runtime,
        ))
        .unwrap();
        assert_eq!(result.get("success").and_then(Value::as_bool), Some(false));
        assert!(
            result
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("Unsupported video format")
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
        let result = serde_json::from_str::<Value>(&handle_video_analyze(
            &json!({"video_url": "https://blocked.test/cat.mp4", "question": "describe"}),
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
    fn local_video_file_calls_multimodal_chat_completion() {
        let temp = TempDir::new().unwrap();
        let (base_url, captured, server) = serve_chat_completion(json!({
            "choices": [{
                "message": {
                    "content": "A short test clip."
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
        let video_path = temp.path().join("test.mp4");
        fs::write(&video_path, b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42").unwrap();

        let result = serde_json::from_str::<Value>(&handle_video_analyze(
            &json!({"video_url": video_path.display().to_string(), "question": "What happens?"}),
            &runtime,
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(
            result.get("analysis").and_then(Value::as_str),
            Some("A short test clip.")
        );

        let requests = captured.lock().unwrap();
        let payload = serde_json::from_str::<Value>(&requests[0]).unwrap();
        assert_eq!(
            payload.get("model").and_then(Value::as_str),
            Some("gpt-4.1-mini")
        );
        let content = payload["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert!(
            content[0]["text"]
                .as_str()
                .unwrap()
                .contains("Fully describe and explain everything happening in this video")
        );
        assert_eq!(content[1]["type"].as_str(), Some("video_url"));
        assert!(
            content[1]["video_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:video/mp4;base64,")
        );
    }

    #[test]
    fn file_uri_is_treated_as_local_path() {
        let temp = TempDir::new().unwrap();
        let (base_url, _, server) = serve_chat_completion(json!({
            "choices": [{
                "message": {
                    "content": "A local file URI video."
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
        let video_path = temp.path().join("clip.mp4");
        fs::write(&video_path, b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42").unwrap();

        let result = serde_json::from_str::<Value>(&handle_video_analyze(
            &json!({"video_url": format!("file://{}", video_path.display()), "question": "describe this"}),
            &runtime,
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(
            result.get("analysis").and_then(Value::as_str),
            Some("A local file URI video.")
        );
    }

    #[test]
    fn anthropic_mode_is_rejected_with_clear_error() {
        let temp = TempDir::new().unwrap();
        write_config(
            temp.path(),
            "model:\n  default: claude-3-7-sonnet\n  provider: anthropic\n  api_key: test-key\n",
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let video_path = temp.path().join("clip.mp4");
        fs::write(&video_path, b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42").unwrap();

        let result = serde_json::from_str::<Value>(&handle_video_analyze(
            &json!({"video_url": video_path.display().to_string(), "question": "what is this?"}),
            &runtime,
        ))
        .unwrap();
        assert_eq!(result.get("success").and_then(Value::as_bool), Some(false));
        assert!(
            result
                .get("analysis")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("does not support native video analysis")
        );
    }
}
