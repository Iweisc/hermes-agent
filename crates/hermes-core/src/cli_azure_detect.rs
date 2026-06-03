//! Azure Foundry endpoint auto-detection.
//!
//! Inspect an Azure AI Foundry / Azure OpenAI endpoint to determine:
//!   - API transport (OpenAI-style `chat_completions` vs Anthropic-style
//!     `anthropic_messages`)
//!   - Available models (best effort — Azure does not expose a deployment
//!     listing via the inference API key, but Azure OpenAI v1 endpoints
//!     return the resource's model catalog via `GET /models`)
//!   - Context length for each discovered/entered model, via the existing
//!     [`crate::ag_model_metadata::get_model_context_length`] resolver.
//!
//! Rationale:
//!
//! Azure has no pure-API-key deployment-listing endpoint — per Microsoft,
//! deployment enumeration requires ARM management-plane auth. Azure OpenAI v1
//! endpoints `{resource}.openai.azure.com/openai/v1` do return a `/models`
//! list, but it reflects the resource's *available* models rather than the
//! user's *deployed* deployment names. In practice it is still a useful hint —
//! the user picks a familiar model name and we look up its context length from
//! the catalog.
//!
//! The detector never crashes on errors (every HTTP call is wrapped in a broad
//! error path). Callers get a [`DetectionResult`] with whatever information
//! could be gathered, and fall back to manual entry for the rest.
//!
//! This is a faithful native Rust port of `hermes_cli/azure_detect.py`. Network
//! calls use `reqwest::blocking` (the Python original used `urllib`).

use std::time::Duration;

use regex::Regex;
use serde_json::Value;
use url::Url;

/// Default Azure OpenAI `api-version` values to probe with. The v1 GA endpoint
/// accepts requests without `api-version` entirely, so these are only used as a
/// fallback for pre-v1 resources that still require it.
const AZURE_OPENAI_PROBE_API_VERSIONS: &[&str] = &[
    "2025-04-01-preview",
    "2024-10-21", // oldest GA that supports /models
];

/// Default Azure Anthropic `api-version`. Matches the value used by
/// `agent/anthropic_adapter.py` when building the Anthropic client.
const AZURE_ANTHROPIC_API_VERSION: &str = "2025-04-15";

const USER_AGENT: &str = "hermes-agent/azure-detect";
const PROBE_TIMEOUT_SECS: u64 = 6;

/// Everything auto-detection could gather from a base URL + API key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetectionResult {
    /// Detected API transport: `"chat_completions"`, `"anthropic_messages"`,
    /// or `None` when detection failed.
    pub api_mode: Option<String>,

    /// Deployment / model IDs returned by `/models` (best effort). Empty when
    /// the endpoint doesn't expose the list with an API key.
    pub models: Vec<String>,

    /// Lowercased host from the base URL (used for display messages).
    pub hostname: String,

    /// Human-readable reason the detector chose `api_mode`. Useful for
    /// explaining auto-detection to the user in the wizard.
    pub reason: String,

    /// `true` when `/models` returned a valid OpenAI-shaped payload.
    pub models_probe_ok: bool,

    /// `true` when the URL was determined to be an Anthropic-style endpoint
    /// (from path suffix or live probe).
    pub is_anthropic: bool,
}

fn build_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(PROBE_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

/// GET a URL with `api-key` + `Authorization` headers. Returns
/// `(status_code, parsed_json_or_None)`. Never raises.
///
/// On transport errors (connection refused, timeout, DNS) the status code is
/// `0`, matching the Python original's `URLError`/`OSError` handling.
fn http_get_json(client: &reqwest::blocking::Client, url: &str, api_key: &str) -> (u16, Option<Value>) {
    // Azure OpenAI uses `api-key`. Some Azure deployments (and Anthropic-style
    // routes) use `Authorization: Bearer`. Send both so we probe once per URL
    // rather than twice.
    let resp = client
        .get(url)
        .header("api-key", api_key)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("User-Agent", USER_AGENT)
        .send();

    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            // Read the body and attempt to parse JSON; non-JSON => None.
            match r.text() {
                Ok(body) => (status, serde_json::from_str::<Value>(&body).ok()),
                Err(_) => (status, None),
            }
        }
        Err(e) => {
            log::debug!("azure_detect: GET {url} failed: {e}");
            (0, None)
        }
    }
}

/// Strip trailing `/v1` or `/v1/` so we can construct sub-paths.
fn strip_trailing_v1(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    // Regex `/v1/?$` against the already right-stripped string.
    let re = Regex::new(r"/v1/?$").expect("static regex");
    re.replace(trimmed, "").into_owned()
}

/// Return `true` when the URL's path ends in `/anthropic` or contains an
/// `/anthropic/` segment. Used by Azure Foundry resources that route Claude
/// traffic through a dedicated path.
fn looks_like_anthropic_path(url: &str) -> bool {
    match Url::parse(url) {
        Ok(parsed) => {
            let path = parsed.path().to_lowercase();
            let path = path.trim_end_matches('/').to_string();
            path.ends_with("/anthropic") || format!("{path}/").contains("/anthropic/")
        }
        Err(_) => false,
    }
}

/// Extract a list of model IDs from an OpenAI-shaped `/models` response.
/// Returns `[]` on any shape mismatch.
fn extract_model_ids(payload: &Value) -> Vec<String> {
    let data = match payload.get("data") {
        Some(d) => d,
        None => return Vec::new(),
    };
    let arr = match data.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut ids: Vec<String> = Vec::new();
    for item in arr {
        if !item.is_object() {
            continue;
        }
        // OpenAI shape: {"id": "gpt-5.4", "object": "model", ...}
        let mid = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| item.get("model").and_then(Value::as_str).filter(|s| !s.is_empty()))
            .or_else(|| item.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()));
        if let Some(s) = mid {
            ids.push(s.to_string());
        }
    }
    ids
}

/// Probe `<base>/models` for an OpenAI-shaped response.
///
/// Returns `(ok, models)`. `ok` is `true` iff the endpoint accepted us as an
/// OpenAI-style caller (200 OK + OpenAI-shaped JSON body).
fn probe_openai_models(client: &reqwest::blocking::Client, base_url: &str, api_key: &str) -> (bool, Vec<String>) {
    let base_url = base_url.trim_end_matches('/');

    // Azure OpenAI v1: {resource}.openai.azure.com/openai/v1 — no api-version
    // required for GA paths, so probe without first.
    let mut candidates = vec![format!("{base_url}/models")];
    // Fallback: explicit api-version for pre-v1 resources.
    for v in AZURE_OPENAI_PROBE_API_VERSIONS {
        candidates.push(format!("{base_url}/models?api-version={v}"));
    }

    for url in &candidates {
        let (status, body) = http_get_json(client, url, api_key);
        if status == 200 {
            if let Some(body) = body {
                let ids = extract_model_ids(&body);
                if !ids.is_empty() {
                    log::info!(
                        "azure_detect: /models probe OK at {url} ({} models)",
                        ids.len()
                    );
                    return (true, ids);
                }
                // 200 + empty list still counts as "OpenAI shape, no models
                // listed" — let the user proceed with manual entry.
                if body.is_object() && body.get("data").is_some() {
                    return (true, Vec::new());
                }
            }
        }
    }
    (false, Vec::new())
}

/// Send a zero-token request to `<base>/v1/messages` and check whether the
/// endpoint at least *recognises* the Anthropic Messages shape (any 4xx that
/// mentions `messages` or `model`, or a 400 `invalid_request` with an Anthropic
/// error shape). Never completes a real chat.
fn probe_anthropic_messages(client: &reqwest::blocking::Client, base_url: &str, api_key: &str) -> bool {
    let base = strip_trailing_v1(base_url);
    let url = format!("{base}/v1/messages?api-version={AZURE_ANTHROPIC_API_VERSION}");
    let payload = serde_json::json!({
        "model": "probe",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    });

    let resp = client
        .post(&url)
        .header("api-key", api_key)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .header("User-Agent", USER_AGENT)
        .body(serde_json::to_vec(&payload).unwrap_or_default())
        .send();

    let resp = match resp {
        Ok(r) => r,
        // URLError / TimeoutError / OSError equivalent.
        Err(_) => return false,
    };

    let status = resp.status().as_u16();

    if status < 400 {
        // Should never 2xx/3xx — "probe" isn't a real deployment. But if it
        // does, the endpoint definitely speaks Anthropic (status < 500).
        return status < 500;
    }

    // 4xx/5xx: inspect the error body. A 4xx with an Anthropic-shaped error
    // body = Anthropic endpoint.
    let body = match resp.text() {
        Ok(b) => b,
        Err(_) => return false,
    };
    classify_anthropic_error(status, &body)
}

/// Decide whether an error response body indicates an Anthropic Messages
/// endpoint. Split out for unit testing.
///
/// Mirrors the Python precedence exactly:
/// `"anthropic" in lowered or '"type"' in lowered and '"error"' in lowered`
/// where Python `and` binds tighter than `or`.
fn classify_anthropic_error(status: u16, body: &str) -> bool {
    let lowered = body.to_lowercase();
    if lowered.contains("anthropic") || (lowered.contains("\"type\"") && lowered.contains("\"error\"")) {
        return true;
    }
    // Pre-Azure-v1 Azure Foundry returns a plain 404 for Anthropic-style calls
    // on non-Anthropic deployments. A 400 "model not found" IS Anthropic though.
    if status == 400 && (lowered.contains("messages") || lowered.contains("model")) {
        return true;
    }
    false
}

/// Inspect an Azure endpoint and describe its transport + models.
///
/// Call this from the wizard before asking the user to pick an API mode
/// manually. The caller should treat the returned [`DetectionResult`] as
/// *advisory* — if `api_mode` is `None`, fall back to asking the user.
pub fn detect(base_url: &str, api_key: &str) -> DetectionResult {
    let client = build_client();
    let mut result = DetectionResult::default();

    result.hostname = Url::parse(base_url)
        .ok()
        .and_then(|p| p.host_str().map(|h| h.to_lowercase()))
        .unwrap_or_default();

    // 1. Path sniff. Azure Foundry exposes Anthropic-style deployments under a
    //    dedicated `/anthropic` path.
    if looks_like_anthropic_path(base_url) {
        result.is_anthropic = true;
        result.api_mode = Some("anthropic_messages".to_string());
        result.reason = "URL path ends in /anthropic → Anthropic Messages API".to_string();
        return result;
    }

    // 2. Try the OpenAI-style /models probe. If this works, the endpoint
    //    definitely speaks OpenAI wire.
    let (ok, models) = probe_openai_models(&client, base_url, api_key);
    if ok {
        result.models_probe_ok = true;
        result.reason = if models.is_empty() {
            "GET /models returned an OpenAI-shaped empty list — OpenAI-style endpoint".to_string()
        } else {
            format!("GET /models returned {} model(s) — OpenAI-style endpoint", models.len())
        };
        result.models = models;
        result.api_mode = Some("chat_completions".to_string());
        return result;
    }

    // 3. Fallback: probe the Anthropic Messages shape. Slower and more
    //    intrusive than /models, so only run it when the OpenAI probe failed.
    if probe_anthropic_messages(&client, base_url, api_key) {
        result.is_anthropic = true;
        result.api_mode = Some("anthropic_messages".to_string());
        result.reason = "Endpoint accepts Anthropic Messages shape".to_string();
        return result;
    }

    // Nothing matched. Caller falls back to manual selection.
    result.reason = "Could not probe endpoint (private network, missing model list, or \
         non-standard path) — falling back to manual API-mode selection"
        .to_string();
    result
}

/// Thin wrapper around [`crate::ag_model_metadata::get_model_context_length`]
/// that returns `None` when only the fallback default (128k) would fire, so the
/// wizard can distinguish "we actually know this" from "we guessed."
///
/// Note: the native `get_model_context_length` takes a few extra arguments not
/// present in the Python signature (`config_context_length`, `provider`,
/// `custom_providers_present`). We pass the neutral defaults the wizard uses:
/// no config override, the `"azure"` provider, and no custom providers.
pub fn lookup_context_length(model: &str, base_url: &str, api_key: &str) -> Option<i64> {
    let n = crate::ag_model_metadata::get_model_context_length(
        model, base_url, api_key, None, "azure", false,
    );

    if n > 0 && n != crate::ag_model_metadata::DEFAULT_FALLBACK_CONTEXT {
        Some(n)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_trailing_v1_variants() {
        assert_eq!(strip_trailing_v1("https://x.com/openai/v1"), "https://x.com/openai");
        assert_eq!(strip_trailing_v1("https://x.com/openai/v1/"), "https://x.com/openai");
        assert_eq!(strip_trailing_v1("https://x.com/openai/"), "https://x.com/openai");
        // No /v1 suffix: only trailing slash removed.
        assert_eq!(strip_trailing_v1("https://x.com/foo"), "https://x.com/foo");
        // v1 only when it's the final segment.
        assert_eq!(strip_trailing_v1("https://x.com/v1/models"), "https://x.com/v1/models");
    }

    #[test]
    fn anthropic_path_detection() {
        assert!(looks_like_anthropic_path("https://r.services.ai.azure.com/anthropic"));
        assert!(looks_like_anthropic_path("https://r.services.ai.azure.com/anthropic/"));
        assert!(looks_like_anthropic_path(
            "https://r.services.ai.azure.com/models/anthropic/v1"
        ));
        // Case insensitive.
        assert!(looks_like_anthropic_path("https://r.azure.com/Anthropic"));
        // Not anthropic.
        assert!(!looks_like_anthropic_path("https://r.openai.azure.com/openai/v1"));
        assert!(!looks_like_anthropic_path("not a url"));
        // "anthropic" as a substring of a longer segment should NOT match.
        assert!(!looks_like_anthropic_path("https://r.azure.com/anthropicx"));
    }

    #[test]
    fn extract_ids_openai_shape() {
        let payload = json!({
            "object": "list",
            "data": [
                {"id": "gpt-5.4", "object": "model"},
                {"id": "o3-mini", "object": "model"},
            ]
        });
        assert_eq!(extract_model_ids(&payload), vec!["gpt-5.4", "o3-mini"]);
    }

    #[test]
    fn extract_ids_fallback_keys_and_skips() {
        let payload = json!({
            "data": [
                {"model": "claude-x"},
                {"name": "named-model"},
                {"object": "model"},          // no id/model/name -> skipped
                "not-an-object",                // skipped
                {"id": ""},                      // empty id -> skipped
            ]
        });
        assert_eq!(extract_model_ids(&payload), vec!["claude-x", "named-model"]);
    }

    #[test]
    fn extract_ids_shape_mismatch() {
        assert!(extract_model_ids(&json!({"foo": "bar"})).is_empty());
        assert!(extract_model_ids(&json!({"data": "scalar"})).is_empty());
        assert!(extract_model_ids(&json!([1, 2, 3])).is_empty());
    }

    #[test]
    fn anthropic_error_classification() {
        // Body mentioning anthropic.
        assert!(classify_anthropic_error(404, "the anthropic route failed"));
        // Anthropic error shape: both "type" and "error" present.
        assert!(classify_anthropic_error(400, r#"{"type":"error","error":{"x":1}}"#));
        // "type" alone without "error" should not trip the and-clause.
        assert!(!classify_anthropic_error(404, r#"{"type":"list"}"#));
        // 400 mentioning model.
        assert!(classify_anthropic_error(400, "deployment model not found"));
        // 400 mentioning messages.
        assert!(classify_anthropic_error(400, "invalid messages array"));
        // 404 plain, no anthropic hints -> not anthropic.
        assert!(!classify_anthropic_error(404, "not found"));
        // 400 hint only matters at 400, not 404.
        assert!(!classify_anthropic_error(404, "model missing"));
    }

    #[test]
    fn detection_result_default() {
        let r = DetectionResult::default();
        assert_eq!(r.api_mode, None);
        assert!(r.models.is_empty());
        assert_eq!(r.hostname, "");
        assert!(!r.models_probe_ok);
        assert!(!r.is_anthropic);
    }

    #[test]
    fn detect_anthropic_path_shortcircuits() {
        // No network involved: the path sniff returns before any HTTP call.
        let r = detect("https://res.services.ai.azure.com/anthropic", "key");
        assert_eq!(r.api_mode.as_deref(), Some("anthropic_messages"));
        assert!(r.is_anthropic);
        assert_eq!(r.hostname, "res.services.ai.azure.com");
        assert!(r.reason.contains("/anthropic"));
    }
}
