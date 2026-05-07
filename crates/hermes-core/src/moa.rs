use std::env;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::agent::{build_http_client_with_timeout, request_model_text};
use crate::tools::{ToolRuntime, tool_error, tool_result};
use crate::{HermesContext, LoadedConfig, ModelOverrides};

const REFERENCE_MODELS: &[&str] = &[
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-pro",
    "openai/gpt-5.4-pro",
    "deepseek/deepseek-v3.2",
];
const AGGREGATOR_MODEL: &str = "anthropic/claude-opus-4.6";
const MIN_SUCCESSFUL_REFERENCES: usize = 1;
const MAX_RETRIES: usize = 3;
const REQUEST_TIMEOUT_SECS: u64 = 300;
const RETRY_BACKOFF_MS: u64 = 250;
const AGGREGATOR_SYSTEM_PROMPT: &str = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Critically evaluate the information in those responses because some of it may be biased or incorrect. Do not simply replicate the given answers. Produce a refined, accurate, comprehensive reply that is well-structured and reliable.\n\nResponses from models:";

pub fn mixture_of_agents_schema() -> Value {
    json!({
        "name": "mixture_of_agents",
        "description": "Route a genuinely hard problem through multiple frontier models collaboratively. This makes multiple high-latency model calls and should be used sparingly for tasks that benefit from diverse reasoning, such as complex math, advanced algorithms, or multi-step analytical problems.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The difficult problem or query to solve with multiple collaborating models."
                }
            },
            "required": ["user_prompt"]
        }
    })
}

pub fn mixture_of_agents_available() -> bool {
    env::var("OPENROUTER_API_KEY")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

pub fn handle_mixture_of_agents(args: &Value, runtime: &ToolRuntime) -> String {
    let user_prompt = match required_non_empty_string(args, "user_prompt") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match run_mixture_of_agents(&user_prompt, REFERENCE_MODELS, AGGREGATOR_MODEL, runtime) {
        Ok(response) => tool_result(json!({
            "success": true,
            "response": response,
            "models_used": {
                "reference_models": REFERENCE_MODELS,
                "aggregator_model": AGGREGATOR_MODEL,
            }
        })),
        Err(error) => tool_result(json!({
            "success": false,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": REFERENCE_MODELS,
                "aggregator_model": AGGREGATOR_MODEL,
            },
            "error": error,
        })),
    }
}

fn run_mixture_of_agents(
    user_prompt: &str,
    reference_models: &[&str],
    aggregator_model: &str,
    runtime: &ToolRuntime,
) -> Result<String, String> {
    if user_prompt.trim().is_empty() {
        return Err("user_prompt must be a non-empty string".to_string());
    }
    if reference_models.is_empty() {
        return Err("mixture_of_agents requires at least one reference model".to_string());
    }

    let started = Instant::now();
    let context =
        HermesContext::detect().with_hermes_home_env(Some(runtime.hermes_home().to_path_buf()));
    let loaded = context
        .load_config_document()
        .map_err(|error| error.to_string())?;

    let mut handles = Vec::new();
    for model in reference_models {
        let model = (*model).to_string();
        let context = context.clone();
        let loaded = loaded.clone();
        let prompt = user_prompt.to_string();
        handles.push(thread::spawn(move || {
            run_reference_model_safe(&context, &loaded, &model, &prompt)
        }));
    }

    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Ok(text)) => successful.push(text),
            Ok(Err(error)) => failed.push(error),
            Err(_) => failed.push("reference model worker panicked".to_string()),
        }
    }

    if successful.len() < MIN_SUCCESSFUL_REFERENCES {
        return Err(format!(
            "Insufficient successful reference models ({}/{}). Failures: {}",
            successful.len(),
            reference_models.len(),
            failed.join("; ")
        ));
    }

    let aggregated = run_aggregator_model(
        &context,
        &loaded,
        aggregator_model,
        user_prompt,
        &successful,
    )?;
    log::info!(
        target: "moa_tool",
        "completed refs_ok={} refs_failed={} elapsed_ms={}",
        successful.len(),
        failed.len(),
        started.elapsed().as_millis()
    );
    Ok(aggregated)
}

fn run_reference_model_safe(
    context: &HermesContext,
    loaded: &LoadedConfig,
    model: &str,
    user_prompt: &str,
) -> Result<String, String> {
    let overrides = ModelOverrides {
        model: Some(model.to_string()),
        provider: Some("openrouter".to_string()),
        ..ModelOverrides::default()
    };
    let runtime_model = context
        .resolve_model_runtime(loaded, &overrides)
        .map_err(|error| error.to_string())?;
    let messages = vec![json!({
        "role": "user",
        "content": user_prompt,
    })];
    let client =
        build_http_client_with_timeout(REQUEST_TIMEOUT_SECS).map_err(|error| error.to_string())?;

    let mut last_error = None;
    for attempt in 0..MAX_RETRIES {
        match request_model_text(&client, &runtime_model, &messages) {
            Ok(Some(text)) if !text.trim().is_empty() => return Ok(text),
            Ok(_) => {
                last_error = Some(format!("{model} returned empty content"));
            }
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }
        if attempt + 1 < MAX_RETRIES {
            let backoff = RETRY_BACKOFF_MS.saturating_mul(1_u64 << attempt);
            thread::sleep(Duration::from_millis(backoff));
        }
    }

    Err(format!(
        "{model} failed after {MAX_RETRIES} attempts: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    ))
}

fn run_aggregator_model(
    context: &HermesContext,
    loaded: &LoadedConfig,
    aggregator_model: &str,
    user_prompt: &str,
    responses: &[String],
) -> Result<String, String> {
    let overrides = ModelOverrides {
        model: Some(aggregator_model.to_string()),
        provider: Some("openrouter".to_string()),
        ..ModelOverrides::default()
    };
    let runtime_model = context
        .resolve_model_runtime(loaded, &overrides)
        .map_err(|error| error.to_string())?;
    let system_prompt = construct_aggregator_prompt(responses);
    let messages = vec![
        json!({
            "role": "system",
            "content": system_prompt,
        }),
        json!({
            "role": "user",
            "content": user_prompt,
        }),
    ];
    let client =
        build_http_client_with_timeout(REQUEST_TIMEOUT_SECS).map_err(|error| error.to_string())?;
    request_model_text(&client, &runtime_model, &messages)
        .map_err(|error| error.to_string())?
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Aggregator model returned no text content.".to_string())
}

fn construct_aggregator_prompt(responses: &[String]) -> String {
    let numbered = responses
        .iter()
        .enumerate()
        .map(|(index, response)| format!("{}. {}", index + 1, response))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{AGGREGATOR_SYSTEM_PROMPT}\n\n{numbered}")
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use super::*;

    use tempfile::TempDir;

    fn runtime_for(home: &Path, cwd: &Path) -> ToolRuntime {
        ToolRuntime::new(cwd.to_path_buf()).with_hermes_home(home.to_path_buf())
    }

    fn write_config(home: &Path, body: &str) {
        fs::write(home.join("config.yaml"), body).unwrap();
    }

    fn serve_sequence(
        responses: Vec<(u16, Value)>,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let handle = thread::spawn(move || {
            for (status, body) in responses {
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
                let status_text = if status == 200 { "OK" } else { "ERROR" };
                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (format!("http://{addr}/v1"), captured, handle)
    }

    #[test]
    fn schema_requires_user_prompt() {
        let schema = mixture_of_agents_schema();
        assert_eq!(schema["name"], "mixture_of_agents");
        assert_eq!(schema["parameters"]["required"], json!(["user_prompt"]));
    }

    #[test]
    fn successful_run_aggregates_reference_responses() {
        let temp = TempDir::new().unwrap();
        let (base_url, captured, server) = serve_sequence(vec![
            (
                200,
                json!({"choices":[{"message":{"content":"Reference answer A"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"Reference answer B"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"Final aggregated answer"},"finish_reason":"stop"}]}),
            ),
        ]);
        write_config(
            temp.path(),
            &format!(
                "model:\n  default: test-default\n  provider: openrouter\n  base_url: {base_url}\n  api_key: test-key\n"
            ),
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let result = run_mixture_of_agents(
            "Solve this",
            &["model-a", "model-b"],
            "aggregator",
            &runtime,
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(result, "Final aggregated answer");
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 3);
        let payload = serde_json::from_str::<Value>(&requests[2]).unwrap();
        assert_eq!(payload["messages"][0]["role"], "system");
        let system_prompt = payload["messages"][0]["content"].as_str().unwrap();
        assert!(system_prompt.contains("1. Reference answer"));
        assert!(system_prompt.contains("2. Reference answer"));
    }

    #[test]
    fn transient_reference_failure_retries_and_succeeds() {
        let temp = TempDir::new().unwrap();
        let (base_url, _, server) = serve_sequence(vec![
            (500, json!({"error":"rate limited"})),
            (
                200,
                json!({"choices":[{"message":{"content":"Reference answer ok"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"Recovered aggregate"},"finish_reason":"stop"}]}),
            ),
        ]);
        write_config(
            temp.path(),
            &format!(
                "model:\n  default: test-default\n  provider: openrouter\n  base_url: {base_url}\n  api_key: test-key\n"
            ),
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let result =
            run_mixture_of_agents("Recover this", &["model-a"], "aggregator", &runtime).unwrap();
        server.join().unwrap();

        assert_eq!(result, "Recovered aggregate");
    }

    #[test]
    fn handle_returns_structured_error_when_every_reference_fails() {
        let temp = TempDir::new().unwrap();
        let (base_url, _, server) = serve_sequence(vec![
            (500, json!({"error":"fail-1"})),
            (500, json!({"error":"fail-2"})),
            (500, json!({"error":"fail-3"})),
            (500, json!({"error":"fail-4"})),
            (500, json!({"error":"fail-5"})),
            (500, json!({"error":"fail-6"})),
            (500, json!({"error":"fail-7"})),
            (500, json!({"error":"fail-8"})),
            (500, json!({"error":"fail-9"})),
            (500, json!({"error":"fail-10"})),
            (500, json!({"error":"fail-11"})),
            (500, json!({"error":"fail-12"})),
        ]);
        write_config(
            temp.path(),
            &format!(
                "model:\n  default: test-default\n  provider: openrouter\n  base_url: {base_url}\n  api_key: test-key\n"
            ),
        );
        let runtime = runtime_for(temp.path(), temp.path());
        let result = serde_json::from_str::<Value>(&handle_mixture_of_agents(
            &json!({"user_prompt":"hard problem"}),
            &runtime,
        ))
        .unwrap();
        server.join().unwrap();

        assert_eq!(result["success"], Value::Bool(false));
        assert!(
            result["error"]
                .as_str()
                .unwrap_or_default()
                .contains("Insufficient successful reference models")
        );
    }
}
