//! Google Code Assist API client — project discovery, onboarding, quota.
//!
//! The Code Assist API powers Google's official gemini-cli. It sits at
//! `cloudcode-pa.googleapis.com` and provides:
//!
//! - Free tier access (generous daily quota) for personal Google accounts
//! - Paid tier access via GCP projects with billing / Workspace / Standard / Enterprise
//!
//! This module handles the **request/response shaping** for the control-plane
//! dance needed before inference:
//!
//! 1. `loadCodeAssist` — probe the user's account to learn what tier they're on
//!    and whether a `cloudaicompanionProject` is already assigned.
//! 2. `onboardUser` — provision a user on a tier (LRO polling done by caller).
//! 3. `retrieveUserQuota` — fetch the `buckets[]` array showing remaining quota.
//!
//! VPC-SC handling: enterprise accounts under a VPC Service Controls perimeter
//! get `SECURITY_POLICY_VIOLATED` on `loadCodeAssist`. Callers catch the
//! corresponding [`CodeAssistError`] (`is_vpc_sc()`) and force the account to
//! `standard-tier` so the call chain still succeeds.
//!
//! Scope note: per the porting instruction, this module is **request/response
//! shaping only** — there is no live OAuth and no live HTTP here. The actual
//! `urllib`-equivalent POST loop, the onboarding LRO sleep-poll loop, and the
//! networked `resolve_project_context` live in the later `google_oauth` batch.
//! The pure URL builders and shaping functions below give that future module
//! everything it needs to drive the calls.

use serde_json::{json, Value};
use std::collections::BTreeMap;

// =============================================================================
// Constants
// =============================================================================

pub const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";

/// Fallback endpoints tried when prod returns an error during project discovery.
pub const FALLBACK_ENDPOINTS: [&str; 2] = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
];

// Tier identifiers that Google's API uses.
pub const FREE_TIER_ID: &str = "free-tier";
pub const LEGACY_TIER_ID: &str = "legacy-tier";
pub const STANDARD_TIER_ID: &str = "standard-tier";

// Default HTTP fingerprint matching gemini-cli. Google may reject unrecognized
// User-Agents on these internal endpoints.
const GEMINI_CLI_USER_AGENT: &str = "google-api-nodejs-client/9.15.1 (gzip)";
const X_GOOG_API_CLIENT: &str = "gl-node/24.0.0";

/// Default per-request timeout, seconds. (Used by the networked caller.)
pub const DEFAULT_REQUEST_TIMEOUT_SECS: f64 = 30.0;
/// Number of onboarding LRO poll attempts. (Used by the networked caller.)
pub const ONBOARDING_POLL_ATTEMPTS: u32 = 12;
/// Delay between onboarding LRO polls, seconds. (Used by the networked caller.)
pub const ONBOARDING_POLL_INTERVAL_SECONDS: f64 = 5.0;

// =============================================================================
// Error type
// =============================================================================

/// Error raised by the Code Assist (`cloudcode-pa`) integration.
///
/// Carries HTTP status / response / retry-after metadata so the agent's error
/// classifier and the main loop's `Retry-After` handling pick up the right
/// signals. Mirrors the Python `CodeAssistError` (and its `ProjectIdRequiredError`
/// subclass, collapsed here into a constructor + `code` string).
#[derive(Debug, Clone, Default)]
pub struct CodeAssistError {
    pub message: String,
    /// Stable machine code, e.g. `code_assist_error`, `code_assist_vpc_sc`,
    /// `code_assist_http_429`, `code_assist_project_id_required`.
    pub code: String,
    /// HTTP status, picked up by the error classifier (e.g. 429 -> rate_limit).
    pub status_code: Option<u16>,
    /// Raw underlying response body, if any.
    pub response: Option<String>,
    /// Parsed `Retry-After` seconds (header or `google.rpc.RetryInfo`).
    pub retry_after: Option<f64>,
    /// Parsed structured error details from the Google error envelope, e.g.
    /// `{"reason": "MODEL_CAPACITY_EXHAUSTED", "status": "RESOURCE_EXHAUSTED"}`.
    pub details: BTreeMap<String, Value>,
}

impl CodeAssistError {
    /// General-purpose constructor with the default `code_assist_error` code.
    pub fn new(message: impl Into<String>) -> Self {
        CodeAssistError {
            message: message.into(),
            code: "code_assist_error".to_string(),
            status_code: None,
            response: None,
            retry_after: None,
            details: BTreeMap::new(),
        }
    }

    /// Constructor with an explicit machine code.
    pub fn with_code(message: impl Into<String>, code: impl Into<String>) -> Self {
        CodeAssistError {
            code: code.into(),
            ..CodeAssistError::new(message)
        }
    }

    /// VPC Service Controls violation — callers default the account to
    /// `standard-tier`. Equivalent to the Python `code="code_assist_vpc_sc"`.
    pub fn vpc_sc(detail: impl AsRef<str>) -> Self {
        CodeAssistError::with_code(
            format!("VPC-SC policy violation: {}", detail.as_ref()),
            "code_assist_vpc_sc",
        )
    }

    /// HTTP error mapping helper, mirroring `f"code_assist_http_{code}"`.
    pub fn http(status: u16, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        CodeAssistError {
            status_code: Some(status),
            response: Some(detail.clone()),
            code: format!("code_assist_http_{status}"),
            message: format!("Code Assist HTTP {status}: {detail}"),
            retry_after: None,
            details: BTreeMap::new(),
        }
    }

    /// Network error mapping helper.
    pub fn network(message: impl Into<String>) -> Self {
        CodeAssistError::with_code(
            format!("Code Assist request failed: {}", message.into()),
            "code_assist_network_error",
        )
    }

    /// Equivalent of the Python `ProjectIdRequiredError`.
    pub fn project_id_required(message: Option<String>) -> Self {
        CodeAssistError::with_code(
            message.unwrap_or_else(|| "GCP project id required for this tier".to_string()),
            "code_assist_project_id_required",
        )
    }

    pub fn is_vpc_sc(&self) -> bool {
        self.code == "code_assist_vpc_sc"
    }

    pub fn is_project_id_required(&self) -> bool {
        self.code == "code_assist_project_id_required"
    }
}

impl std::fmt::Display for CodeAssistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CodeAssistError {}

// =============================================================================
// HTTP primitive shaping (no live network here)
// =============================================================================

/// Build the default headers matching gemini-cli's fingerprint, including the
/// optional `model/<x>` User-Agent suffix and a unique `x-activity-request-id`.
///
/// Returned as an ordered list of `(name, value)` pairs so callers can apply
/// them to whatever HTTP client they use. The Python original used a
/// `uuid.uuid4()` request id; the `uuid` crate is not a dependency here, so we
/// generate a unique id the same way `agent.rs`'s Google path does.
pub fn build_headers(access_token: &str, user_agent_model: &str) -> Vec<(String, String)> {
    let ua = if user_agent_model.is_empty() {
        GEMINI_CLI_USER_AGENT.to_string()
    } else {
        format!("{GEMINI_CLI_USER_AGENT} model/{user_agent_model}")
    };
    vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Accept".to_string(), "application/json".to_string()),
        ("Authorization".to_string(), format!("Bearer {access_token}")),
        ("User-Agent".to_string(), ua),
        ("X-Goog-Api-Client".to_string(), X_GOOG_API_CLIENT.to_string()),
        ("x-activity-request-id".to_string(), activity_request_id()),
    ]
}

fn activity_request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("hermes-{nanos:x}")
}

/// Match Google's gemini-cli exactly — unrecognized metadata may be rejected.
pub fn client_metadata() -> Value {
    json!({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    })
}

/// Detect a VPC Service Controls violation from a response body.
///
/// Faithful port of `_is_vpc_sc_violation`: parse the JSON envelope and walk
/// `error.details[].reason == "SECURITY_POLICY_VIOLATED"` and `error.message`;
/// on a JSON parse failure, fall back to a raw substring match.
pub fn is_vpc_sc_violation(body: &str) -> bool {
    if body.is_empty() {
        return false;
    }
    let parsed: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return body.contains("SECURITY_POLICY_VIOLATED"),
    };
    let error = match parsed.get("error") {
        Some(Value::Object(map)) => map,
        _ => return false,
    };
    if let Some(Value::Array(details)) = error.get("details") {
        for item in details {
            if let Some(reason) = item.get("reason").and_then(Value::as_str) {
                if reason == "SECURITY_POLICY_VIOLATED" {
                    return true;
                }
            }
        }
    }
    let msg = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    msg.contains("SECURITY_POLICY_VIOLATED")
}

// =============================================================================
// URL builders + endpoint list (for the networked caller in google_oauth)
// =============================================================================

pub fn load_code_assist_url(endpoint: &str) -> String {
    format!("{endpoint}/v1internal:loadCodeAssist")
}

pub fn onboard_user_url(endpoint: &str) -> String {
    format!("{endpoint}/v1internal:onboardUser")
}

pub fn onboard_poll_url(endpoint: &str, op_name: &str) -> String {
    format!("{endpoint}/v1internal/{op_name}")
}

pub fn retrieve_user_quota_url(endpoint: &str) -> String {
    format!("{endpoint}/v1internal:retrieveUserQuota")
}

/// Prod endpoint first, then the sandbox fallbacks — the order
/// `load_code_assist` tries them in.
pub fn discovery_endpoints() -> Vec<String> {
    let mut v = vec![CODE_ASSIST_ENDPOINT.to_string()];
    v.extend(FALLBACK_ENDPOINTS.iter().map(|s| s.to_string()));
    v
}

// =============================================================================
// loadCodeAssist — discovers current tier + assigned project
// =============================================================================

/// Result from `loadCodeAssist`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CodeAssistProjectInfo {
    pub current_tier_id: String,
    /// Google-managed project (free tier).
    pub cloudaicompanion_project: String,
    pub allowed_tiers: Vec<String>,
    pub raw: Value,
}

/// Build the `loadCodeAssist` request body.
pub fn build_load_code_assist_body(project_id: &str) -> Value {
    let mut metadata = json!({ "duetProject": project_id });
    if let (Value::Object(meta), Value::Object(client)) = (&mut metadata, client_metadata()) {
        for (k, v) in client {
            meta.insert(k, v);
        }
    }
    let mut body = json!({ "metadata": metadata });
    if !project_id.is_empty() {
        body["cloudaicompanionProject"] = Value::String(project_id.to_string());
    }
    body
}

/// Parse a `loadCodeAssist` response. Port of `_parse_load_response`.
pub fn parse_load_response(resp: &Value) -> CodeAssistProjectInfo {
    let tier_id = resp
        .get("currentTier")
        .and_then(Value::as_object)
        .and_then(|t| t.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let project = resp
        .get("cloudaicompanionProject")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut allowed_ids = Vec::new();
    if let Some(Value::Array(allowed)) = resp.get("allowedTiers") {
        for t in allowed {
            if let Some(tid) = t.get("id").and_then(Value::as_str) {
                if !tid.is_empty() {
                    allowed_ids.push(tid.to_string());
                }
            }
        }
    }

    CodeAssistProjectInfo {
        current_tier_id: tier_id,
        cloudaicompanion_project: project,
        allowed_tiers: allowed_ids,
        raw: resp.clone(),
    }
}

// =============================================================================
// onboardUser — provisions a new user on a tier
// =============================================================================

/// Build the `onboardUser` request body.
///
/// For paid tiers, `project_id` is REQUIRED (returns
/// [`CodeAssistError::project_id_required`]). For free/legacy tiers it is
/// optional — Google assigns one.
pub fn build_onboard_user_body(
    tier_id: &str,
    project_id: &str,
) -> Result<Value, CodeAssistError> {
    if tier_id != FREE_TIER_ID && tier_id != LEGACY_TIER_ID && project_id.is_empty() {
        return Err(CodeAssistError::project_id_required(Some(format!(
            "Tier '{tier_id}' requires a GCP project id. \
             Set HERMES_GEMINI_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
        ))));
    }

    let mut body = json!({
        "tierId": tier_id,
        "metadata": client_metadata(),
    });
    if !project_id.is_empty() {
        body["cloudaicompanionProject"] = Value::String(project_id.to_string());
    }
    Ok(body)
}

// =============================================================================
// retrieveUserQuota — for /gquota
// =============================================================================

#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuotaBucket {
    pub model_id: String,
    pub token_type: String,
    pub remaining_fraction: f64,
    pub reset_time_iso: String,
    pub raw: Value,
}

/// Build the `retrieveUserQuota` request body.
pub fn build_retrieve_quota_body(project_id: &str) -> Value {
    if project_id.is_empty() {
        json!({})
    } else {
        json!({ "project": project_id })
    }
}

/// Parse a `retrieveUserQuota` response into `buckets[]`.
pub fn parse_quota_response(resp: &Value) -> Vec<QuotaBucket> {
    let raw_buckets = match resp.get("buckets") {
        Some(Value::Array(b)) => b,
        _ => return Vec::new(),
    };
    let mut buckets = Vec::new();
    for b in raw_buckets {
        if !b.is_object() {
            continue;
        }
        buckets.push(QuotaBucket {
            model_id: b
                .get("modelId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            token_type: b
                .get("tokenType")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            // Python does `float(b.get("remainingFraction") or 0.0)`: a missing
            // / null / 0 value all collapse to 0.0.
            remaining_fraction: b
                .get("remainingFraction")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            reset_time_iso: b
                .get("resetTime")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            raw: b.clone(),
        });
    }
    buckets
}

// =============================================================================
// Project context resolution
// =============================================================================

/// Resolved state for a given OAuth session.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectContext {
    /// Effective project id sent on requests.
    pub project_id: String,
    /// Google-assigned project (free tier).
    pub managed_project_id: String,
    pub tier_id: String,
    /// `"env"`, `"config"`, `"discovered"`, or `"onboarded"`.
    pub source: String,
}

/// Pure resolution logic for [`resolve_project_context`], split out so it can be
/// unit-tested and reused by the networked `google_oauth` module.
///
/// Priority:
///   1. `configured_project_id` -> `source="config"`, `standard-tier`.
///   2. `env_project_id` -> `source="env"`, `standard-tier`.
///   3. Otherwise use `info` (from `loadCodeAssist`). If a tier is already
///      assigned, `source="discovered"`. If not, the caller must have run
///      `onboardUser` (free tier) and pass its parsed response in
///      `onboard_response`; `source="onboarded"` and the effective project is
///      taken from `info` or, failing that, the onboard response's
///      `response.cloudaicompanionProject`.
pub fn resolve_project_context_from_discovery(
    info: &CodeAssistProjectInfo,
    onboard_response: Option<&Value>,
    configured_project_id: &str,
    env_project_id: &str,
) -> ProjectContext {
    // Short-circuit: caller provided a project id.
    if !configured_project_id.is_empty() {
        return ProjectContext {
            project_id: configured_project_id.to_string(),
            managed_project_id: String::new(),
            tier_id: STANDARD_TIER_ID.to_string(), // assume paid since specified
            source: "config".to_string(),
        };
    }
    if !env_project_id.is_empty() {
        return ProjectContext {
            project_id: env_project_id.to_string(),
            managed_project_id: String::new(),
            tier_id: STANDARD_TIER_ID.to_string(),
            source: "env".to_string(),
        };
    }

    let mut effective_project = info.cloudaicompanion_project.clone();
    let tier;
    let source;

    if info.current_tier_id.is_empty() {
        // User hasn't been onboarded — caller provisioned them on free tier.
        if let Some(resp) = onboard_response {
            if let Some(response_body) = resp.get("response").and_then(Value::as_object) {
                if effective_project.is_empty() {
                    effective_project = response_body
                        .get("cloudaicompanionProject")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                }
            }
        }
        tier = FREE_TIER_ID.to_string();
        source = "onboarded".to_string();
    } else {
        tier = info.current_tier_id.clone();
        source = "discovered".to_string();
    }

    let managed = if tier == FREE_TIER_ID {
        effective_project.clone()
    } else {
        String::new()
    };

    ProjectContext {
        project_id: effective_project,
        managed_project_id: managed,
        tier_id: tier,
        source,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_default_user_agent() {
        let h = build_headers("tok", "");
        let map: BTreeMap<_, _> = h.iter().cloned().collect();
        assert_eq!(map["Authorization"], "Bearer tok");
        assert_eq!(map["User-Agent"], GEMINI_CLI_USER_AGENT);
        assert_eq!(map["X-Goog-Api-Client"], X_GOOG_API_CLIENT);
        assert!(map.contains_key("x-activity-request-id"));
    }

    #[test]
    fn headers_with_model_suffix() {
        let h = build_headers("tok", "gemini-2.5-pro");
        let map: BTreeMap<_, _> = h.into_iter().collect();
        assert_eq!(
            map["User-Agent"],
            format!("{GEMINI_CLI_USER_AGENT} model/gemini-2.5-pro")
        );
    }

    #[test]
    fn client_metadata_shape() {
        assert_eq!(
            client_metadata(),
            json!({
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            })
        );
    }

    #[test]
    fn vpc_sc_detection_via_details() {
        let body = json!({
            "error": {
                "message": "denied",
                "details": [{"reason": "SECURITY_POLICY_VIOLATED"}],
            }
        })
        .to_string();
        assert!(is_vpc_sc_violation(&body));
    }

    #[test]
    fn vpc_sc_detection_via_message() {
        let body = json!({
            "error": { "message": "SECURITY_POLICY_VIOLATED here" }
        })
        .to_string();
        assert!(is_vpc_sc_violation(&body));
    }

    #[test]
    fn vpc_sc_detection_non_json_fallback() {
        assert!(is_vpc_sc_violation("garbage SECURITY_POLICY_VIOLATED garbage"));
        assert!(!is_vpc_sc_violation("garbage not it"));
        assert!(!is_vpc_sc_violation(""));
    }

    #[test]
    fn vpc_sc_detection_clean_envelope_is_false() {
        let body = json!({ "error": { "message": "permission denied" } }).to_string();
        assert!(!is_vpc_sc_violation(&body));
    }

    #[test]
    fn load_body_no_project() {
        let body = build_load_code_assist_body("");
        assert_eq!(body["metadata"]["duetProject"], "");
        assert_eq!(body["metadata"]["pluginType"], "GEMINI");
        assert!(body.get("cloudaicompanionProject").is_none());
    }

    #[test]
    fn load_body_with_project() {
        let body = build_load_code_assist_body("my-proj");
        assert_eq!(body["metadata"]["duetProject"], "my-proj");
        assert_eq!(body["cloudaicompanionProject"], "my-proj");
    }

    #[test]
    fn parse_load_full() {
        let resp = json!({
            "currentTier": {"id": "free-tier"},
            "cloudaicompanionProject": "managed-123",
            "allowedTiers": [{"id": "free-tier"}, {"id": "standard-tier"}, {"name": "no-id"}],
        });
        let info = parse_load_response(&resp);
        assert_eq!(info.current_tier_id, "free-tier");
        assert_eq!(info.cloudaicompanion_project, "managed-123");
        assert_eq!(info.allowed_tiers, vec!["free-tier", "standard-tier"]);
        assert_eq!(info.raw, resp);
    }

    #[test]
    fn parse_load_empty() {
        let info = parse_load_response(&json!({}));
        assert_eq!(info.current_tier_id, "");
        assert_eq!(info.cloudaicompanion_project, "");
        assert!(info.allowed_tiers.is_empty());
    }

    #[test]
    fn onboard_body_free_tier_no_project_ok() {
        let body = build_onboard_user_body(FREE_TIER_ID, "").unwrap();
        assert_eq!(body["tierId"], FREE_TIER_ID);
        assert_eq!(body["metadata"]["pluginType"], "GEMINI");
        assert!(body.get("cloudaicompanionProject").is_none());
    }

    #[test]
    fn onboard_body_legacy_tier_no_project_ok() {
        assert!(build_onboard_user_body(LEGACY_TIER_ID, "").is_ok());
    }

    #[test]
    fn onboard_body_paid_tier_requires_project() {
        let err = build_onboard_user_body(STANDARD_TIER_ID, "").unwrap_err();
        assert!(err.is_project_id_required());
        assert_eq!(err.code, "code_assist_project_id_required");
    }

    #[test]
    fn onboard_body_paid_tier_with_project_ok() {
        let body = build_onboard_user_body(STANDARD_TIER_ID, "proj-9").unwrap();
        assert_eq!(body["cloudaicompanionProject"], "proj-9");
    }

    #[test]
    fn quota_parse() {
        let resp = json!({
            "buckets": [
                {"modelId": "gemini-2.5-pro", "tokenType": "INPUT", "remainingFraction": 0.5, "resetTime": "2026-06-04T00:00:00Z"},
                {"modelId": "gemini-2.5-flash"},
                "not-a-dict",
            ]
        });
        let buckets = parse_quota_response(&resp);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].model_id, "gemini-2.5-pro");
        assert_eq!(buckets[0].token_type, "INPUT");
        assert!((buckets[0].remaining_fraction - 0.5).abs() < 1e-9);
        assert_eq!(buckets[0].reset_time_iso, "2026-06-04T00:00:00Z");
        assert_eq!(buckets[1].model_id, "gemini-2.5-flash");
        assert_eq!(buckets[1].remaining_fraction, 0.0);
    }

    #[test]
    fn quota_parse_missing_buckets() {
        assert!(parse_quota_response(&json!({})).is_empty());
        assert!(parse_quota_response(&json!({"buckets": "nope"})).is_empty());
    }

    #[test]
    fn quota_body() {
        assert_eq!(build_retrieve_quota_body(""), json!({}));
        assert_eq!(build_retrieve_quota_body("p"), json!({"project": "p"}));
    }

    #[test]
    fn url_builders() {
        assert_eq!(
            load_code_assist_url(CODE_ASSIST_ENDPOINT),
            "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
        );
        assert_eq!(
            onboard_user_url(CODE_ASSIST_ENDPOINT),
            "https://cloudcode-pa.googleapis.com/v1internal:onboardUser"
        );
        assert_eq!(
            onboard_poll_url(CODE_ASSIST_ENDPOINT, "operations/abc"),
            "https://cloudcode-pa.googleapis.com/v1internal/operations/abc"
        );
        assert_eq!(
            retrieve_user_quota_url(CODE_ASSIST_ENDPOINT),
            "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
        );
    }

    #[test]
    fn discovery_endpoints_order() {
        let eps = discovery_endpoints();
        assert_eq!(eps[0], CODE_ASSIST_ENDPOINT);
        assert_eq!(eps.len(), 3);
    }

    #[test]
    fn resolve_config_short_circuit() {
        let ctx = resolve_project_context_from_discovery(
            &CodeAssistProjectInfo::default(),
            None,
            "configured-proj",
            "env-proj",
        );
        assert_eq!(ctx.project_id, "configured-proj");
        assert_eq!(ctx.tier_id, STANDARD_TIER_ID);
        assert_eq!(ctx.source, "config");
        assert_eq!(ctx.managed_project_id, "");
    }

    #[test]
    fn resolve_env_short_circuit() {
        let ctx = resolve_project_context_from_discovery(
            &CodeAssistProjectInfo::default(),
            None,
            "",
            "env-proj",
        );
        assert_eq!(ctx.project_id, "env-proj");
        assert_eq!(ctx.tier_id, STANDARD_TIER_ID);
        assert_eq!(ctx.source, "env");
    }

    #[test]
    fn resolve_discovered_free_tier() {
        let info = CodeAssistProjectInfo {
            current_tier_id: FREE_TIER_ID.to_string(),
            cloudaicompanion_project: "managed-abc".to_string(),
            ..Default::default()
        };
        let ctx = resolve_project_context_from_discovery(&info, None, "", "");
        assert_eq!(ctx.project_id, "managed-abc");
        assert_eq!(ctx.managed_project_id, "managed-abc");
        assert_eq!(ctx.tier_id, FREE_TIER_ID);
        assert_eq!(ctx.source, "discovered");
    }

    #[test]
    fn resolve_discovered_paid_tier_no_managed() {
        let info = CodeAssistProjectInfo {
            current_tier_id: STANDARD_TIER_ID.to_string(),
            cloudaicompanion_project: "p".to_string(),
            ..Default::default()
        };
        let ctx = resolve_project_context_from_discovery(&info, None, "", "");
        assert_eq!(ctx.tier_id, STANDARD_TIER_ID);
        assert_eq!(ctx.source, "discovered");
        assert_eq!(ctx.managed_project_id, "");
    }

    #[test]
    fn resolve_onboarded_pulls_project_from_response() {
        let info = CodeAssistProjectInfo::default(); // no tier yet
        let onboard = json!({
            "done": true,
            "response": {"cloudaicompanionProject": "newly-assigned"}
        });
        let ctx = resolve_project_context_from_discovery(&info, Some(&onboard), "", "");
        assert_eq!(ctx.project_id, "newly-assigned");
        assert_eq!(ctx.managed_project_id, "newly-assigned");
        assert_eq!(ctx.tier_id, FREE_TIER_ID);
        assert_eq!(ctx.source, "onboarded");
    }

    #[test]
    fn resolve_onboarded_keeps_existing_managed_project() {
        let info = CodeAssistProjectInfo {
            cloudaicompanion_project: "from-load".to_string(),
            ..Default::default()
        };
        let onboard = json!({"response": {"cloudaicompanionProject": "ignored"}});
        let ctx = resolve_project_context_from_discovery(&info, Some(&onboard), "", "");
        // existing project from load wins (Python: `effective_project or ...`)
        assert_eq!(ctx.project_id, "from-load");
        assert_eq!(ctx.source, "onboarded");
    }

    #[test]
    fn error_constructors() {
        assert!(CodeAssistError::vpc_sc("x").is_vpc_sc());
        let http = CodeAssistError::http(429, "rate limited");
        assert_eq!(http.status_code, Some(429));
        assert_eq!(http.code, "code_assist_http_429");
        assert_eq!(CodeAssistError::new("m").code, "code_assist_error");
    }
}
