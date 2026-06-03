//! Mixture-of-Agents (MoA) tool — native Rust port of
//! `tools/mixture_of_agents_tool.py`.
//!
//! Implements the Mixture-of-Agents methodology: multiple frontier reference
//! models generate diverse responses in parallel (layer 1), and an aggregator
//! model synthesises them into a single high-quality answer (layer 2).
//!
//! Based on "Mixture-of-Agents Enhances Large Language Model Capabilities"
//! (Junlin Wang et al., arXiv:2406.04692v1).
//!
//! Architecture:
//!   1. Reference models generate diverse initial responses in parallel.
//!   2. Aggregator model synthesises responses into a high-quality output.
//!
//! This port keeps the OpenRouter request/response wire shapes identical to the
//! Python implementation:
//!   * `POST {base_url}/chat/completions`
//!   * body: `model`, `messages`, `max_tokens`, `reasoning`, optional `temperature`
//!   * response parsing mirrors `extract_content_or_reasoning` (content first,
//!     then `reasoning`/`reasoning_content`, then `reasoning_details`).
//!
//! Unlike the async Python version (which uses `asyncio.gather`), the reference
//! models here run concurrently on OS threads via `reqwest::blocking`.

use std::env;
use std::thread;
use std::time::{Duration, Instant};

use regex::Regex;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Reference models — these generate diverse initial responses in parallel.
pub const REFERENCE_MODELS: &[&str] = &[
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-pro",
    "openai/gpt-5.4-pro",
    "deepseek/deepseek-v3.2",
];

/// Aggregator model — synthesises reference responses into the final output.
pub const AGGREGATOR_MODEL: &str = "anthropic/claude-opus-4.6";

/// Balanced creativity for diverse reference perspectives.
pub const REFERENCE_TEMPERATURE: f64 = 0.6;
/// Focused synthesis for consistent aggregation.
pub const AGGREGATOR_TEMPERATURE: f64 = 0.4;

/// Minimum successful reference models needed to proceed to aggregation.
pub const MIN_SUCCESSFUL_REFERENCES: usize = 1;

/// Maximum response tokens requested from each reference model.
pub const REFERENCE_MAX_TOKENS: u64 = 32_000;

/// Retry attempts per reference model (matches Python `max_retries=6`).
pub const MAX_RETRIES: usize = 6;

/// Request timeout for a single model call.
pub const REQUEST_TIMEOUT_SECS: u64 = 300;

/// OpenRouter chat-completions base URL (mirrors `OPENROUTER_BASE_URL`).
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// System prompt for the aggregator model (verbatim from the research paper).
pub const AGGREGATOR_SYSTEM_PROMPT: &str = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:";

// ---------------------------------------------------------------------------
// Tool schema / registration helpers
// ---------------------------------------------------------------------------

/// JSON schema for the `mixture_of_agents` tool (mirrors `MOA_SCHEMA`).
pub fn moa_schema() -> Value {
    json!({
        "name": "mixture_of_agents",
        "description": "Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The complex query or problem to solve using multiple AI models. Should be a challenging problem that benefits from diverse perspectives and collaborative reasoning."
                }
            },
            "required": ["user_prompt"]
        }
    })
}

/// Equivalent of `check_moa_requirements()` / `check_openrouter_api_key()`.
pub fn check_moa_requirements() -> bool {
    env::var("OPENROUTER_API_KEY")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

/// Equivalent of `get_moa_configuration()`.
pub fn get_moa_configuration() -> Value {
    let total = REFERENCE_MODELS.len();
    json!({
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": total,
        "failure_tolerance": format!(
            "{}/{} models can fail",
            total - MIN_SUCCESSFUL_REFERENCES,
            total
        ),
    })
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Result of a reference-model run: `(model, content_or_error, success)`.
pub type ReferenceResult = (String, String, bool);

/// Process a complex query using the Mixture-of-Agents methodology.
///
/// Returns a pretty-printed JSON string (matching the Python tool's contract):
/// ```text
/// {
///   "success": bool,
///   "response": str,
///   "models_used": { "reference_models": [...], "aggregator_model": "..." },
///   "error": str   // only present on failure
/// }
/// ```
///
/// The OpenRouter base URL is taken from `OPENROUTER_BASE_URL` (env override
/// supported via `OPENROUTER_BASE_URL` env var) and the API key from
/// `OPENROUTER_API_KEY`.
pub fn mixture_of_agents_tool(
    user_prompt: &str,
    reference_models: Option<&[String]>,
    aggregator_model: Option<&str>,
) -> String {
    let ref_models: Vec<String> = match reference_models {
        Some(models) if !models.is_empty() => models.to_vec(),
        _ => REFERENCE_MODELS.iter().map(|s| s.to_string()).collect(),
    };
    let agg_model = aggregator_model.unwrap_or(AGGREGATOR_MODEL).to_string();

    match run_mixture_of_agents(user_prompt, &ref_models, &agg_model) {
        Ok(final_response) => {
            let result = json!({
                "success": true,
                "response": final_response,
                "models_used": {
                    "reference_models": ref_models,
                    "aggregator_model": agg_model,
                }
            });
            to_pretty_json(&result)
        }
        Err(error) => {
            let error_msg = format!("Error in MoA processing: {error}");
            log::error!(target: "moa_tools", "{error_msg}");
            let result = json!({
                "success": false,
                "response": "MoA processing failed. Please try again or use a single model for this query.",
                "models_used": {
                    "reference_models": ref_models,
                    "aggregator_model": agg_model,
                },
                "error": error_msg,
            });
            to_pretty_json(&result)
        }
    }
}

/// Core orchestration: layer 1 (parallel references) + layer 2 (aggregation).
///
/// Returns the aggregated text on success, or a human-readable error string.
pub fn run_mixture_of_agents(
    user_prompt: &str,
    reference_models: &[String],
    aggregator_model: &str,
) -> Result<String, String> {
    let started = Instant::now();

    log::info!(target: "moa_tools", "Starting Mixture-of-Agents processing...");

    // Validate API key availability (mirrors the explicit ValueError in Python).
    let api_key = env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "OPENROUTER_API_KEY environment variable not set".to_string())?;

    if reference_models.is_empty() {
        return Err("mixture_of_agents requires at least one reference model".to_string());
    }

    let base_url = resolve_base_url();

    // ── Layer 1: generate diverse responses from reference models in parallel.
    log::info!(target: "moa_tools", "Layer 1: Generating reference responses...");
    let mut handles = Vec::with_capacity(reference_models.len());
    for model in reference_models {
        let model = model.clone();
        let prompt = user_prompt.to_string();
        let api_key = api_key.clone();
        let base_url = base_url.clone();
        handles.push(thread::spawn(move || {
            run_reference_model_safe(
                &base_url,
                &api_key,
                &model,
                &prompt,
                REFERENCE_TEMPERATURE,
                REFERENCE_MAX_TOKENS,
                MAX_RETRIES,
            )
        }));
    }

    let mut successful_responses: Vec<String> = Vec::new();
    let mut failed_models: Vec<String> = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok((model, content, true)) => {
                let _ = model;
                successful_responses.push(content);
            }
            Ok((model, _error, false)) => failed_models.push(model),
            Err(_) => failed_models.push("reference model worker panicked".to_string()),
        }
    }

    let successful_count = successful_responses.len();
    let failed_count = failed_models.len();
    log::info!(
        target: "moa_tools",
        "Reference model results: {successful_count} successful, {failed_count} failed"
    );
    if !failed_models.is_empty() {
        log::warn!(target: "moa_tools", "Failed models: {}", failed_models.join(", "));
    }

    if successful_count < MIN_SUCCESSFUL_REFERENCES {
        return Err(format!(
            "Insufficient successful reference models ({}/{}). Need at least {} successful responses.",
            successful_count,
            reference_models.len(),
            MIN_SUCCESSFUL_REFERENCES
        ));
    }

    // ── Layer 2: aggregate responses using the aggregator model.
    log::info!(target: "moa_tools", "Layer 2: Synthesizing final response...");
    let aggregator_system_prompt =
        construct_aggregator_prompt(AGGREGATOR_SYSTEM_PROMPT, &successful_responses);

    let final_response = run_aggregator_model(
        &base_url,
        &api_key,
        aggregator_model,
        &aggregator_system_prompt,
        user_prompt,
        AGGREGATOR_TEMPERATURE,
    )?;

    log::info!(
        target: "moa_tools",
        "MoA processing completed in {:.2} seconds",
        started.elapsed().as_secs_f64()
    );

    Ok(final_response)
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// Construct the aggregator system prompt with enumerated reference responses.
///
/// Mirrors `_construct_aggregator_prompt`: 1-indexed numbering joined by `\n`,
/// appended to the base system prompt after a blank line.
pub fn construct_aggregator_prompt(system_prompt: &str, responses: &[String]) -> String {
    let response_text = responses
        .iter()
        .enumerate()
        .map(|(index, response)| format!("{}. {}", index + 1, response))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{system_prompt}\n\n{response_text}")
}

// ---------------------------------------------------------------------------
// Model invocation
// ---------------------------------------------------------------------------

/// Run a single reference model with retry + graceful failure handling.
///
/// Returns `(model, content_or_error, success)`, matching the Python tuple.
/// On a retryable failure or empty (reasoning-only) content, it sleeps with
/// exponential backoff (`2^(attempt+1)` capped at 60s) before retrying.
pub fn run_reference_model_safe(
    base_url: &str,
    api_key: &str,
    model: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u64,
    max_retries: usize,
) -> ReferenceResult {
    let messages = json!([{ "role": "user", "content": user_prompt }]);

    let mut last_error: Option<String> = None;
    for attempt in 0..max_retries {
        log::info!(
            target: "moa_tools",
            "Querying {model} (attempt {}/{})",
            attempt + 1,
            max_retries
        );

        match request_chat_completion(
            base_url,
            api_key,
            model,
            &messages,
            Some(max_tokens),
            Some(temperature),
        ) {
            Ok(content) if !content.trim().is_empty() => {
                log::info!(
                    target: "moa_tools",
                    "{model} responded ({} characters)",
                    content.chars().count()
                );
                return (model.to_string(), content, true);
            }
            Ok(_) => {
                // Reasoning-only / empty response — let the retry loop handle it.
                log::warn!(
                    target: "moa_tools",
                    "{model} returned empty content (attempt {}/{}), retrying",
                    attempt + 1,
                    max_retries
                );
                last_error = Some(format!("{model} returned empty content"));
            }
            Err(error) => {
                let lower = error.to_lowercase();
                if lower.contains("invalid") {
                    log::warn!(
                        target: "moa_tools",
                        "{model} invalid request error (attempt {}): {error}",
                        attempt + 1
                    );
                } else if lower.contains("rate") || lower.contains("limit") {
                    log::warn!(
                        target: "moa_tools",
                        "{model} rate limit error (attempt {}): {error}",
                        attempt + 1
                    );
                } else {
                    log::warn!(
                        target: "moa_tools",
                        "{model} unknown error (attempt {}): {error}",
                        attempt + 1
                    );
                }
                last_error = Some(error);
            }
        }

        if attempt + 1 < max_retries {
            // Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s.
            let sleep_secs = (1_u64 << (attempt + 1)).min(60);
            log::info!(target: "moa_tools", "Retrying in {sleep_secs}s...");
            thread::sleep(Duration::from_secs(sleep_secs));
        }
    }

    let error_msg = format!(
        "{model} failed after {max_retries} attempts: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
    log::error!(target: "moa_tools", "{error_msg}");
    (model.to_string(), error_msg, false)
}

/// Run the aggregator model to synthesise the final response.
///
/// Mirrors `_run_aggregator_model`: builds a system+user message pair, and
/// retries once if the first response yields empty (reasoning-only) content.
pub fn run_aggregator_model(
    base_url: &str,
    api_key: &str,
    aggregator_model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
) -> Result<String, String> {
    log::info!(target: "moa_tools", "Running aggregator model: {aggregator_model}");

    let messages = json!([
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": user_prompt },
    ]);

    let mut content = request_chat_completion(
        base_url,
        api_key,
        aggregator_model,
        &messages,
        None,
        Some(temperature),
    )?;

    if content.trim().is_empty() {
        log::warn!(target: "moa_tools", "Aggregator returned empty content, retrying once");
        content = request_chat_completion(
            base_url,
            api_key,
            aggregator_model,
            &messages,
            None,
            Some(temperature),
        )?;
    }

    log::info!(
        target: "moa_tools",
        "Aggregation complete ({} characters)",
        content.chars().count()
    );
    Ok(content)
}

// ---------------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------------

/// Build the chat-completions request body, matching the Python `api_params`.
///
/// The Python OpenAI SDK merges `extra_body` into the top-level JSON body, so
/// `reasoning` appears at top level here. Temperature is omitted for GPT models
/// (matching `if not model.lower().startswith('gpt-')`).
pub fn build_request_body(
    model: &str,
    messages: &Value,
    max_tokens: Option<u64>,
    temperature: Option<f64>,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {
            "enabled": true,
            "effort": "xhigh"
        }
    });

    // GPT models don't support custom temperature values.
    if !model.to_lowercase().starts_with("gpt-") {
        if let Some(temp) = temperature {
            body["temperature"] = json!(temp);
        }
    }

    body
}

/// OpenRouter app-attribution headers (mirror `_OR_HEADERS_BASE`).
fn openrouter_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("HTTP-Referer", "https://hermes-agent.nousresearch.com"),
        ("X-Title", "Hermes Agent"),
        ("X-OpenRouter-Categories", "productivity,cli-agent"),
    ]
}

/// Resolve the OpenRouter base URL, honouring an `OPENROUTER_BASE_URL` override.
fn resolve_base_url() -> String {
    env::var("OPENROUTER_BASE_URL")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OPENROUTER_BASE_URL.to_string())
}

/// Perform a single blocking chat-completion request and extract the text.
///
/// Returns the extracted content (possibly empty for reasoning-only replies),
/// or an `Err` describing a transport/HTTP/parse failure.
pub fn request_chat_completion(
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &Value,
    max_tokens: Option<u64>,
    temperature: Option<f64>,
) -> Result<String, String> {
    let body = build_request_body(model, messages, max_tokens, temperature);
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .map_err(|error| format!("failed to build HTTP client: {error}"))?;

    let mut request = client
        .post(&url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json");
    for (name, value) in openrouter_headers() {
        request = request.header(name, value);
    }

    let response = request
        .json(&body)
        .send()
        .map_err(|error| format!("request failed: {error}"))?;

    let status = response.status();
    let text = response
        .text()
        .map_err(|error| format!("failed to read response body: {error}"))?;

    if !status.is_success() {
        return Err(format!("HTTP {}: {}", status.as_u16(), text));
    }

    let parsed: Value =
        serde_json::from_str(&text).map_err(|error| format!("invalid JSON response: {error}"))?;

    // Surface API-level error objects the same way the SDK would raise.
    if let Some(error_obj) = parsed.get("error") {
        if !error_obj.is_null() {
            let message = error_obj
                .get("message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| error_obj.to_string());
            return Err(message);
        }
    }

    Ok(extract_content_or_reasoning(&parsed))
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// Extract content from a chat-completion response, falling back to reasoning.
///
/// Faithful port of `extract_content_or_reasoning`:
///   1. `choices[0].message.content` with inline think/reasoning blocks stripped.
///   2. `message.reasoning` / `message.reasoning_content`.
///   3. `message.reasoning_details[].{summary|content|text}`.
///
/// Returns `""` when nothing usable is found.
pub fn extract_content_or_reasoning(response: &Value) -> String {
    let message = match response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
    {
        Some(message) => message,
        None => return String::new(),
    };

    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();

    if !content.is_empty() {
        let cleaned = strip_think_blocks(content);
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }

    // Content empty or reasoning-only — try structured reasoning fields.
    let mut reasoning_parts: Vec<String> = Vec::new();
    for field in ["reasoning", "reasoning_content"] {
        if let Some(val) = message.get(field).and_then(Value::as_str) {
            let trimmed = val.trim();
            if !trimmed.is_empty() && !reasoning_parts.iter().any(|p| p == trimmed) {
                reasoning_parts.push(trimmed.to_string());
            }
        }
    }

    if let Some(details) = message.get("reasoning_details").and_then(Value::as_array) {
        for detail in details {
            if !detail.is_object() {
                continue;
            }
            let summary = ["summary", "content", "text"]
                .iter()
                .find_map(|key| detail.get(*key))
                .filter(|value| !value.is_null());
            if let Some(summary) = summary {
                let rendered = match summary.as_str() {
                    Some(text) => text.trim().to_string(),
                    None => summary.to_string(),
                };
                if !rendered.is_empty() && !reasoning_parts.iter().any(|p| p == &rendered) {
                    reasoning_parts.push(rendered);
                }
            }
        }
    }

    if !reasoning_parts.is_empty() {
        return reasoning_parts.join("\n\n");
    }

    String::new()
}

/// Strip inline `<think>`/`<thinking>`/`<reasoning>`/`<thought>`/
/// `<REASONING_SCRATCHPAD>` blocks (case-insensitive, dot-all), mirroring the
/// Python regex in `extract_content_or_reasoning`.
fn strip_think_blocks(content: &str) -> String {
    let pattern = r"(?si)<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>.*?</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>";
    match Regex::new(pattern) {
        Ok(regex) => regex.replace_all(content, "").into_owned(),
        Err(_) => content.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_requires_user_prompt() {
        let schema = moa_schema();
        assert_eq!(schema["name"], "mixture_of_agents");
        assert_eq!(schema["parameters"]["required"], json!(["user_prompt"]));
    }

    #[test]
    fn configuration_reports_failure_tolerance() {
        let config = get_moa_configuration();
        assert_eq!(config["aggregator_model"], AGGREGATOR_MODEL);
        assert_eq!(config["total_reference_models"], REFERENCE_MODELS.len());
        assert_eq!(config["min_successful_references"], MIN_SUCCESSFUL_REFERENCES);
        assert_eq!(config["failure_tolerance"], "3/4 models can fail");
    }

    #[test]
    fn aggregator_prompt_is_enumerated() {
        let responses = vec!["alpha".to_string(), "beta".to_string()];
        let prompt = construct_aggregator_prompt("BASE", &responses);
        assert!(prompt.starts_with("BASE\n\n"));
        assert!(prompt.contains("1. alpha"));
        assert!(prompt.contains("2. beta"));
    }

    #[test]
    fn request_body_includes_reasoning_and_temperature() {
        let messages = json!([{ "role": "user", "content": "hi" }]);
        let body = build_request_body("anthropic/claude-opus-4.6", &messages, Some(32_000), Some(0.6));
        assert_eq!(body["model"], "anthropic/claude-opus-4.6");
        assert_eq!(body["max_tokens"], 32_000);
        assert_eq!(body["reasoning"]["enabled"], true);
        assert_eq!(body["reasoning"]["effort"], "xhigh");
        assert_eq!(body["temperature"], 0.6);
    }

    #[test]
    fn request_body_omits_temperature_for_gpt_models() {
        let messages = json!([{ "role": "user", "content": "hi" }]);
        let body = build_request_body("gpt-5.4-pro", &messages, None, Some(0.6));
        assert!(body.get("temperature").is_none());
        // max_tokens=None should serialise as JSON null, matching Python.
        assert!(body["max_tokens"].is_null());
    }

    #[test]
    fn extract_prefers_plain_content() {
        let response = json!({
            "choices": [{ "message": { "content": "  the answer  " } }]
        });
        assert_eq!(extract_content_or_reasoning(&response), "the answer");
    }

    #[test]
    fn extract_strips_think_blocks() {
        let response = json!({
            "choices": [{
                "message": { "content": "<think>secret plan</think>visible answer" }
            }]
        });
        assert_eq!(extract_content_or_reasoning(&response), "visible answer");
    }

    #[test]
    fn extract_falls_back_to_reasoning_when_content_empty() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "<think>only reasoning here</think>",
                    "reasoning": "deep thought"
                }
            }]
        });
        assert_eq!(extract_content_or_reasoning(&response), "deep thought");
    }

    #[test]
    fn extract_uses_reasoning_details_array() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_details": [
                        { "summary": "first" },
                        { "text": "second" }
                    ]
                }
            }]
        });
        assert_eq!(extract_content_or_reasoning(&response), "first\n\nsecond");
    }

    #[test]
    fn extract_handles_missing_choices() {
        assert_eq!(extract_content_or_reasoning(&json!({})), "");
    }

    #[test]
    fn tool_errors_when_api_key_missing() {
        // Ensure the key is absent for this test.
        let saved = env::var("OPENROUTER_API_KEY").ok();
        unsafe { env::remove_var("OPENROUTER_API_KEY"); }

        let output = mixture_of_agents_tool("hard problem", None, None);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["success"], false);
        assert!(parsed["error"]
            .as_str()
            .unwrap()
            .contains("OPENROUTER_API_KEY"));
        assert_eq!(parsed["models_used"]["aggregator_model"], AGGREGATOR_MODEL);

        if let Some(value) = saved {
            unsafe { env::set_var("OPENROUTER_API_KEY", value); }
        }
    }

    #[test]
    fn check_requirements_reflects_env() {
        let saved = env::var("OPENROUTER_API_KEY").ok();
        unsafe { env::remove_var("OPENROUTER_API_KEY"); }
        assert!(!check_moa_requirements());
        unsafe { env::set_var("OPENROUTER_API_KEY", "test-key"); }
        assert!(check_moa_requirements());
        match saved {
            Some(value) => env::set_var("OPENROUTER_API_KEY", value),
            None => env::remove_var("OPENROUTER_API_KEY"),
        }
    }
}
