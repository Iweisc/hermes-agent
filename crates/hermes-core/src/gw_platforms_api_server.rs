//! OpenAI-compatible API server platform adapter — native Rust port.
//!
//! Ported from `gateway/platforms/api_server.py`.  The original module runs an
//! aiohttp web server exposing OpenAI Chat Completions / Responses / Runs
//! endpoints that route through hermes-agent's `AIAgent`.
//!
//! This port reproduces the platform-independent *business logic* that backs
//! those endpoints — request validation, content normalization, error
//! envelopes, the SQLite-backed `ResponseStore`, the idempotency cache, CORS
//! resolution, bearer-token auth, session-id derivation, request
//! fingerprinting, and output-item extraction — as synchronous, idiomatic Rust
//! that can be driven by any HTTP layer.
//!
//! The async aiohttp request handlers (`_handle_*`) and the SSE writers are not
//! reproduced verbatim — they are thin wrappers in Python around the helpers
//! ported here.  Where useful, this module exposes pure helpers (e.g.
//! [`build_models_response`], [`build_capabilities_response`],
//! [`extract_output_items`]) that those handlers can call, plus typed results
//! ([`AuthOutcome`], [`SessionKeyOutcome`]) describing the decisions the Python
//! handlers make.

use std::collections::HashMap;
use std::sync::Mutex;

use rusqlite::Connection;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Default settings (mirror the module-level constants)
// ---------------------------------------------------------------------------

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8642;
pub const MAX_STORED_RESPONSES: i64 = 100;
/// 10 MB — accommodates long agent conversations with tool calls.
pub const MAX_REQUEST_BYTES: u64 = 10_000_000;
pub const CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS: f64 = 30.0;
/// 64 KB cap for normalized content parts.
pub const MAX_NORMALIZED_TEXT_LENGTH: usize = 65_536;
/// Max items when content is an array.
pub const MAX_CONTENT_LIST_SIZE: usize = 1_000;

/// Soft length cap for session identifiers (X-Hermes-Session-Id / -Key).
pub const MAX_SESSION_HEADER_LEN: usize = 256;

pub const MAX_CONCURRENT_RUNS: usize = 10;
pub const RUN_STREAM_TTL: f64 = 300.0;
pub const RUN_STATUS_TTL: f64 = 3600.0;

pub const MAX_NAME_LENGTH: usize = 200;
pub const MAX_PROMPT_LENGTH: usize = 5000;

/// Content part type aliases used by the OpenAI Chat Completions and Responses
/// APIs.  Both spellings are accepted on input.
const TEXT_PART_TYPES: &[&str] = &["text", "input_text", "output_text"];
const IMAGE_PART_TYPES: &[&str] = &["image_url", "input_image"];
const FILE_PART_TYPES: &[&str] = &["file", "input_file"];

/// Fields whitelisted for cron job updates.
pub const UPDATE_ALLOWED_FIELDS: &[&str] = &[
    "name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled",
];

/// CORS base headers shared by every allowed origin.
fn cors_base_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"),
        (
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, Idempotency-Key",
        ),
    ]
}

/// Security headers added to all responses.
pub fn security_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
    ]
}

// ---------------------------------------------------------------------------
// Port coercion
// ---------------------------------------------------------------------------

/// Parse a listen port without letting malformed env/config values crash
/// startup.  Mirrors `_coerce_port`.
pub fn coerce_port(value: &Value, default: u16) -> u16 {
    match value {
        Value::Number(n) => n.as_i64().and_then(|i| u16::try_from(i).ok()).unwrap_or(default),
        Value::String(s) => s.trim().parse::<u16>().unwrap_or(default),
        Value::Bool(b) => {
            // Python int(True) == 1, int(False) == 0
            if *b { 1 } else { 0 }
        }
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// Content normalization
// ---------------------------------------------------------------------------

fn truncate_text(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        s.chars().take(max).collect()
    } else {
        s.to_string()
    }
}

/// Normalize OpenAI chat message content into a plain text string.
///
/// Mirrors `_normalize_chat_content`: flattens typed content-part arrays into a
/// single newline-joined string with defensive recursion/size/length bounds.
pub fn normalize_chat_content(content: &Value) -> String {
    normalize_chat_content_inner(content, 10, 0)
}

fn normalize_chat_content_inner(content: &Value, max_depth: usize, depth: usize) -> String {
    if depth > max_depth {
        return String::new();
    }
    match content {
        Value::Null => String::new(),
        Value::String(s) => truncate_text(s, MAX_NORMALIZED_TEXT_LENGTH),
        Value::Array(items) => {
            let mut parts: Vec<String> = Vec::new();
            let slice: &[Value] = if items.len() > MAX_CONTENT_LIST_SIZE {
                &items[..MAX_CONTENT_LIST_SIZE]
            } else {
                items
            };
            for item in slice {
                match item {
                    Value::String(s) => {
                        if !s.is_empty() {
                            parts.push(truncate_text(s, MAX_NORMALIZED_TEXT_LENGTH));
                        }
                    }
                    Value::Object(map) => {
                        let item_type = map
                            .get("type")
                            .and_then(value_to_optional_str)
                            .unwrap_or_default()
                            .trim()
                            .to_lowercase();
                        if TEXT_PART_TYPES.contains(&item_type.as_str()) {
                            if let Some(text) = map.get("text") {
                                let text_str = value_as_str(text);
                                if !text_str.is_empty() {
                                    parts.push(truncate_text(&text_str, MAX_NORMALIZED_TEXT_LENGTH));
                                }
                            }
                        }
                        // Silently skip image_url / other non-text parts.
                    }
                    Value::Array(_) => {
                        let nested = normalize_chat_content_inner(item, max_depth, depth + 1);
                        if !nested.is_empty() {
                            parts.push(nested);
                        }
                    }
                    _ => {}
                }
                if parts.iter().map(|p| p.len()).sum::<usize>() >= MAX_NORMALIZED_TEXT_LENGTH {
                    break;
                }
            }
            truncate_text(&parts.join("\n"), MAX_NORMALIZED_TEXT_LENGTH)
        }
        // Fallback for unexpected scalar types (int, float, bool).
        other => truncate_text(&value_as_str(other), MAX_NORMALIZED_TEXT_LENGTH),
    }
}

/// Render a JSON value the way Python's `str()` would for the simple scalar
/// cases this module cares about (numbers, bools).
fn value_as_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn value_to_optional_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(value_as_str(other)),
    }
}

/// Error returned by [`normalize_multimodal_content`].  The `code` maps to an
/// OpenAI-style error code; `message` is the human-readable detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalError {
    pub code: String,
    pub message: String,
}

impl MultimodalError {
    fn new(code: &str, message: &str) -> Self {
        MultimodalError {
            code: code.to_string(),
            message: message.to_string(),
        }
    }
}

/// Validate and normalize multimodal content for the API server.
///
/// Mirrors `_normalize_multimodal_content`.  Returns a plain JSON string when
/// the content is text-only, or a JSON array of `{"type": "text"|"image_url",
/// ...}` parts when images are present.  Returns [`MultimodalError`] on invalid
/// input (callers translate it into a 400 via [`multimodal_validation_error`]).
pub fn normalize_multimodal_content(content: &Value) -> Result<Value, MultimodalError> {
    match content {
        Value::Null => return Ok(Value::String(String::new())),
        Value::String(s) => {
            return Ok(Value::String(truncate_text(s, MAX_NORMALIZED_TEXT_LENGTH)))
        }
        Value::Array(_) => {}
        other => {
            // Mirror the legacy text-normalizer fallback for non-list scalars.
            return Ok(Value::String(normalize_chat_content(other)));
        }
    }

    let items = content.as_array().unwrap();
    let slice: &[Value] = if items.len() > MAX_CONTENT_LIST_SIZE {
        &items[..MAX_CONTENT_LIST_SIZE]
    } else {
        items
    };

    let mut normalized_parts: Vec<Value> = Vec::new();

    for part in slice {
        match part {
            Value::String(s) => {
                if !s.is_empty() {
                    let trimmed = truncate_text(s, MAX_NORMALIZED_TEXT_LENGTH);
                    normalized_parts.push(json!({"type": "text", "text": trimmed}));
                }
                continue;
            }
            Value::Object(map) => {
                let raw_type = map.get("type").cloned().unwrap_or(Value::Null);
                let part_type = value_to_optional_str(&raw_type)
                    .unwrap_or_default()
                    .trim()
                    .to_lowercase();

                if TEXT_PART_TYPES.contains(&part_type.as_str()) {
                    let text_val = map.get("text");
                    let text = match text_val {
                        None | Some(Value::Null) => continue,
                        Some(Value::String(s)) => s.clone(),
                        Some(other) => value_as_str(other),
                    };
                    if !text.is_empty() {
                        let trimmed = truncate_text(&text, MAX_NORMALIZED_TEXT_LENGTH);
                        normalized_parts.push(json!({"type": "text", "text": trimmed}));
                    }
                    continue;
                }

                if IMAGE_PART_TYPES.contains(&part_type.as_str()) {
                    let mut detail = map.get("detail").cloned();
                    let image_ref = map.get("image_url").cloned().unwrap_or(Value::Null);
                    let url_value: Option<String> = match &image_ref {
                        Value::Object(img) => {
                            if let Some(d) = img.get("detail") {
                                detail = Some(d.clone());
                            }
                            img.get("url").and_then(|u| match u {
                                Value::String(s) => Some(s.clone()),
                                _ => None,
                            })
                        }
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    };
                    let url_value = match url_value {
                        Some(u) if !u.trim().is_empty() => u.trim().to_string(),
                        _ => {
                            return Err(MultimodalError::new(
                                "invalid_image_url",
                                "Image parts must include a non-empty image URL.",
                            ))
                        }
                    };
                    let lowered = url_value.to_lowercase();
                    if lowered.starts_with("data:") {
                        if !lowered.starts_with("data:image/") || !url_value.contains(',') {
                            return Err(MultimodalError::new(
                                "unsupported_content_type",
                                "Only image data URLs are supported. Non-image data payloads are not supported.",
                            ));
                        }
                    } else if !(lowered.starts_with("http://") || lowered.starts_with("https://")) {
                        return Err(MultimodalError::new(
                            "invalid_image_url",
                            "Image inputs must use http(s) URLs or data:image/... URLs.",
                        ));
                    }
                    let mut image_url_obj = Map::new();
                    image_url_obj.insert("url".to_string(), Value::String(url_value));
                    if let Some(d) = detail {
                        if !matches!(d, Value::Null) {
                            let d_str = match &d {
                                Value::String(s) => s.clone(),
                                _ => String::new(),
                            };
                            if !matches!(d, Value::String(_)) || d_str.trim().is_empty() {
                                return Err(MultimodalError::new(
                                    "invalid_content_part",
                                    "Image detail must be a non-empty string when provided.",
                                ));
                            }
                            image_url_obj
                                .insert("detail".to_string(), Value::String(d_str.trim().to_string()));
                        }
                    }
                    normalized_parts.push(json!({
                        "type": "image_url",
                        "image_url": Value::Object(image_url_obj),
                    }));
                    continue;
                }

                if FILE_PART_TYPES.contains(&part_type.as_str()) {
                    return Err(MultimodalError::new(
                        "unsupported_content_type",
                        "Inline image inputs are supported, but uploaded files and document inputs are not supported on this endpoint.",
                    ));
                }

                // Unknown part type — reject explicitly.
                let raw_repr = python_repr(&raw_type);
                return Err(MultimodalError::new(
                    "unsupported_content_type",
                    &format!(
                        "Unsupported content part type {raw_repr}. Only text and image_url/input_image parts are supported."
                    ),
                ));
            }
            // Ignore unknown scalars for forward compatibility.
            _ => continue,
        }
    }

    if normalized_parts.is_empty() {
        return Ok(Value::String(String::new()));
    }

    // Text-only: collapse to a plain string.
    if normalized_parts
        .iter()
        .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
    {
        let joined = normalized_parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(Value::String(joined));
    }

    Ok(Value::Array(normalized_parts))
}

/// Render a value the way Python's `repr()` would for error messages.
fn python_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

/// True when content has any text or image attachment.  Mirrors
/// `_content_has_visible_payload` — used to reject empty turns.
pub fn content_has_visible_payload(content: &Value) -> bool {
    match content {
        Value::String(s) => !s.trim().is_empty(),
        Value::Array(items) => {
            for part in items {
                if let Value::Object(map) = part {
                    let ptype = map
                        .get("type")
                        .and_then(value_to_optional_str)
                        .unwrap_or_default()
                        .trim()
                        .to_lowercase();
                    if TEXT_PART_TYPES.contains(&ptype.as_str()) {
                        let text = map
                            .get("text")
                            .and_then(value_to_optional_str)
                            .unwrap_or_default();
                        if !text.trim().is_empty() {
                            return true;
                        }
                    }
                    if IMAGE_PART_TYPES.contains(&ptype.as_str()) {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Error envelopes
// ---------------------------------------------------------------------------

/// OpenAI-style error envelope.  Mirrors `_openai_error`.
pub fn openai_error(
    message: &str,
    err_type: Option<&str>,
    param: Option<&str>,
    code: Option<&str>,
) -> Value {
    json!({
        "error": {
            "message": message,
            "type": err_type.unwrap_or("invalid_request_error"),
            "param": param,
            "code": code,
        }
    })
}

/// A pending HTTP response: a status code, JSON body, and any extra headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonResponse {
    pub status: u16,
    pub body: Value,
    pub headers: Vec<(String, String)>,
}

impl JsonResponse {
    pub fn new(status: u16, body: Value) -> Self {
        JsonResponse {
            status,
            body,
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

/// Translate a [`MultimodalError`] into a 400 JsonResponse.  Mirrors
/// `_multimodal_validation_error`.
pub fn multimodal_validation_error(exc: &MultimodalError, param: &str) -> JsonResponse {
    JsonResponse::new(
        400,
        openai_error(&exc.message, None, Some(param), Some(&exc.code)),
    )
}

// ---------------------------------------------------------------------------
// Session-id derivation + fingerprinting
// ---------------------------------------------------------------------------

/// Derive a stable session ID from the conversation's first user message.
/// Mirrors `_derive_chat_session_id`.
pub fn derive_chat_session_id(system_prompt: Option<&str>, first_user_message: &str) -> String {
    use sha2::{Digest, Sha256};
    let seed = format!("{}\n{}", system_prompt.unwrap_or(""), first_user_message);
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("api-{}", &hex[..16])
}

/// Build a deterministic fingerprint of the selected request keys.
///
/// Mirrors `_make_request_fingerprint`, which hashes `repr({k: body.get(k)})`.
/// Python's `dict` preserves insertion order, and the subset is built by
/// iterating `keys` in order, so we reproduce that ordering here.
pub fn make_request_fingerprint(body: &Map<String, Value>, keys: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    let subset: Vec<(&str, Value)> = keys
        .iter()
        .map(|k| (*k, body.get(*k).cloned().unwrap_or(Value::Null)))
        .collect();
    let repr = python_dict_repr(&subset);
    let mut hasher = Sha256::new();
    hasher.update(repr.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Render a (key -> value) sequence as Python's `repr(dict)` would.
fn python_dict_repr(items: &[(&str, Value)]) -> String {
    let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| format!("'{}': {}", k, python_value_repr(v)))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

fn python_value_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{s}'"),
        Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(python_value_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, val)| format!("'{}': {}", k, python_value_repr(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// CORS / auth helpers
// ---------------------------------------------------------------------------

/// Normalize configured CORS origins into a stable vec.  Mirrors
/// `_parse_cors_origins`.
pub fn parse_cors_origins(value: &Value) -> Vec<String> {
    let items: Vec<String> = match value {
        Value::Null => return Vec::new(),
        Value::String(s) => {
            if s.is_empty() {
                return Vec::new();
            }
            s.split(',').map(|p| p.to_string()).collect()
        }
        Value::Array(arr) => arr.iter().map(value_as_str).collect(),
        other => vec![value_as_str(other)],
    };
    items
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Return CORS headers for an allowed browser origin.  Mirrors
/// `_cors_headers_for_origin`.  Returns `None` when no headers should be sent.
pub fn cors_headers_for_origin(
    origin: &str,
    cors_origins: &[String],
) -> Option<Vec<(String, String)>> {
    if origin.is_empty() || cors_origins.is_empty() {
        return None;
    }

    let mut headers: Vec<(String, String)> = cors_base_headers()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    if cors_origins.iter().any(|o| o == "*") {
        headers.push(("Access-Control-Allow-Origin".to_string(), "*".to_string()));
        headers.push(("Access-Control-Max-Age".to_string(), "600".to_string()));
        return Some(headers);
    }

    if !cors_origins.iter().any(|o| o == origin) {
        return None;
    }

    headers.push((
        "Access-Control-Allow-Origin".to_string(),
        origin.to_string(),
    ));
    headers.push(("Vary".to_string(), "Origin".to_string()));
    headers.push(("Access-Control-Max-Age".to_string(), "600".to_string()));
    Some(headers)
}

/// Allow non-browser clients and explicitly configured browser origins.
/// Mirrors `_origin_allowed`.
pub fn origin_allowed(origin: &str, cors_origins: &[String]) -> bool {
    if origin.is_empty() {
        return true;
    }
    if cors_origins.is_empty() {
        return false;
    }
    cors_origins.iter().any(|o| o == "*" || o == origin)
}

/// Constant-time comparison of two byte slices (mirrors `hmac.compare_digest`).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Outcome of bearer-token auth validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthOutcome {
    /// Auth OK (no key configured, or token matched).
    Ok,
    /// Auth failed — caller should return this 401 response.
    Unauthorized(JsonResponse),
}

/// Validate a Bearer token against the configured API key.  Mirrors
/// `_check_auth`.  When no key is configured, all requests are allowed.
pub fn check_auth(api_key: &str, authorization_header: &str) -> AuthOutcome {
    if api_key.is_empty() {
        return AuthOutcome::Ok;
    }
    if let Some(token) = authorization_header.strip_prefix("Bearer ") {
        let token = token.trim();
        if constant_time_eq(token.as_bytes(), api_key.as_bytes()) {
            return AuthOutcome::Ok;
        }
    }
    AuthOutcome::Unauthorized(JsonResponse::new(
        401,
        json!({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }),
    ))
}

/// Outcome of parsing the `X-Hermes-Session-Key` header.  Mirrors
/// `_parse_session_key_header`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionKeyOutcome {
    /// Header absent/empty — no key.
    None,
    /// Validated session key.
    Key(String),
    /// Validation failed — caller should return this response.
    Error(JsonResponse),
}

/// Returns true if the string contains a control char that could enable header
/// injection (CR, LF, or NUL).  Mirrors `re.search(r'[\r\n\x00]', raw)`.
fn has_header_injection_char(s: &str) -> bool {
    s.contains('\r') || s.contains('\n') || s.contains('\u{0}')
}

/// Extract and validate the `X-Hermes-Session-Key` header.  Mirrors
/// `_parse_session_key_header`.  Requires API-key authentication to accept a
/// caller-supplied memory scope.
pub fn parse_session_key_header(api_key: &str, raw_header: &str) -> SessionKeyOutcome {
    let raw = raw_header.trim();
    if raw.is_empty() {
        return SessionKeyOutcome::None;
    }

    if api_key.is_empty() {
        return SessionKeyOutcome::Error(JsonResponse::new(
            403,
            openai_error(
                "X-Hermes-Session-Key requires API key authentication. Configure API_SERVER_KEY to enable this feature.",
                None,
                None,
                None,
            ),
        ));
    }

    if has_header_injection_char(raw) {
        return SessionKeyOutcome::Error(JsonResponse::new(
            400,
            json!({"error": {"message": "Invalid session key", "type": "invalid_request_error"}}),
        ));
    }

    if raw.chars().count() > MAX_SESSION_HEADER_LEN {
        return SessionKeyOutcome::Error(JsonResponse::new(
            400,
            json!({"error": {"message": "Session key too long", "type": "invalid_request_error"}}),
        ));
    }

    SessionKeyOutcome::Key(raw.to_string())
}

/// Outcome of validating the `X-Hermes-Session-Id` header for session
/// continuation.  Mirrors the inline logic in `_handle_chat_completions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionIdOutcome {
    /// No header provided — caller derives an id from the conversation.
    None,
    /// Validated continuation session id.
    SessionId(String),
    /// Validation failed — caller should return this response.
    Error(JsonResponse),
}

/// Validate the optional `X-Hermes-Session-Id` continuation header.
pub fn parse_session_id_header(api_key: &str, raw_header: &str) -> SessionIdOutcome {
    let provided = raw_header.trim();
    if provided.is_empty() {
        return SessionIdOutcome::None;
    }
    if api_key.is_empty() {
        return SessionIdOutcome::Error(JsonResponse::new(
            403,
            openai_error(
                "Session continuation requires API key authentication. Configure API_SERVER_KEY to enable this feature.",
                None,
                None,
                None,
            ),
        ));
    }
    if has_header_injection_char(provided) {
        return SessionIdOutcome::Error(JsonResponse::new(
            400,
            json!({"error": {"message": "Invalid session ID", "type": "invalid_request_error"}}),
        ));
    }
    SessionIdOutcome::SessionId(provided.to_string())
}

// ---------------------------------------------------------------------------
// Model-name / id helpers
// ---------------------------------------------------------------------------

/// Derive the advertised model name for /v1/models.  Mirrors
/// `_resolve_model_name` priority: explicit override, else active profile (when
/// not default/custom), else `hermes-agent`.
pub fn resolve_model_name(explicit: &str, active_profile: Option<&str>) -> String {
    let trimmed = explicit.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    if let Some(profile) = active_profile {
        if !profile.is_empty() && profile != "default" && profile != "custom" {
            return profile.to_string();
        }
    }
    "hermes-agent".to_string()
}

/// Generate a random lowercase-hex string of `n` chars (mirrors
/// `uuid.uuid4().hex[:n]`).
pub fn random_hex(n: usize) -> String {
    let nbytes = n.div_ceil(2);
    let mut buf = vec![0u8; nbytes];
    if getrandom::fill(&mut buf).is_err() {
        // Fallback: time-seeded — extremely unlikely to be reached.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let bytes = nanos.to_le_bytes();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = bytes[i % bytes.len()];
        }
    }
    let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    hex.chars().take(n).collect()
}

/// `chatcmpl-<29 hex chars>`.
pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", random_hex(29))
}

/// `resp_<28 hex chars>`.
pub fn new_response_id() -> String {
    format!("resp_{}", random_hex(28))
}

/// `run_<32 hex chars>` (full uuid4 hex).
pub fn new_run_id() -> String {
    format!("run_{}", random_hex(32))
}

/// `msg_<24 hex chars>`.
pub fn new_message_item_id() -> String {
    format!("msg_{}", random_hex(24))
}

// ---------------------------------------------------------------------------
// Endpoint payload builders
// ---------------------------------------------------------------------------

/// Build the `/v1/models` response body.  Mirrors `_handle_models`.
pub fn build_models_response(model_name: &str, created: i64) -> Value {
    json!({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": created,
                "owned_by": "hermes",
                "permission": [],
                "root": model_name,
                "parent": null,
            }
        ],
    })
}

/// Build the `/v1/capabilities` response body.  Mirrors `_handle_capabilities`.
pub fn build_capabilities_response(
    model_name: &str,
    api_key_configured: bool,
    cors_configured: bool,
) -> Value {
    json!({
        "object": "hermes.api_server.capabilities",
        "platform": "hermes-agent",
        "model": model_name,
        "auth": {
            "type": "bearer",
            "required": api_key_configured,
        },
        "features": {
            "chat_completions": true,
            "chat_completions_streaming": true,
            "responses_api": true,
            "responses_streaming": true,
            "run_submission": true,
            "run_status": true,
            "run_events_sse": true,
            "run_stop": true,
            "tool_progress_events": true,
            "session_continuity_header": "X-Hermes-Session-Id",
            "session_key_header": "X-Hermes-Session-Key",
            "cors": cors_configured,
        },
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "health_detailed": {"method": "GET", "path": "/health/detailed"},
            "models": {"method": "GET", "path": "/v1/models"},
            "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
            "responses": {"method": "POST", "path": "/v1/responses"},
            "runs": {"method": "POST", "path": "/v1/runs"},
            "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
            "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
            "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
        },
    })
}

/// Build a non-streaming chat.completion response body.  Mirrors the
/// `response_data` dict in `_handle_chat_completions`.
pub fn build_chat_completion_response(
    completion_id: &str,
    model_name: &str,
    created: i64,
    final_response: &str,
    input_tokens: i64,
    output_tokens: i64,
    total_tokens: i64,
) -> Value {
    json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    })
}

// ---------------------------------------------------------------------------
// Output item extraction (Responses API)
// ---------------------------------------------------------------------------

/// Build the full output item array from the agent's messages.  Mirrors
/// `_extract_output_items`.  `result` is the agent's result dict.
pub fn extract_output_items(result: &Value) -> Vec<Value> {
    let mut items: Vec<Value> = Vec::new();
    let empty: Vec<Value> = Vec::new();
    let messages = result
        .get("messages")
        .and_then(|m| m.as_array())
        .unwrap_or(&empty);

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str());
        match role {
            Some("assistant") => {
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tool_calls {
                        let func = tc.get("function");
                        let name = func
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("");
                        let arguments = func
                            .and_then(|f| f.get("arguments"))
                            .cloned()
                            .unwrap_or(Value::String(String::new()));
                        let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        items.push(json!({
                            "type": "function_call",
                            "name": name,
                            "arguments": arguments,
                            "call_id": id,
                        }));
                    }
                }
            }
            Some("tool") => {
                let call_id = msg
                    .get("tool_call_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("");
                let content = msg.get("content").cloned().unwrap_or(Value::String(String::new()));
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                }));
            }
            _ => {}
        }
    }

    // Final assistant message.
    let final_text = result
        .get("final_response")
        .and_then(|f| f.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| result.get("error").and_then(|e| e.as_str()))
        .unwrap_or("(No response generated)");

    items.push(json!({
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": final_text,
            }
        ],
    }));
    items
}

// ---------------------------------------------------------------------------
// Cron job-id validation
// ---------------------------------------------------------------------------

/// Validate a cron job id (`[a-f0-9]{12}`, full match).  Mirrors `_JOB_ID_RE`
/// usage in `_check_job_id`.
pub fn is_valid_job_id(job_id: &str) -> bool {
    job_id.len() == 12 && job_id.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

// ---------------------------------------------------------------------------
// Idempotency cache
// ---------------------------------------------------------------------------

struct IdemEntry {
    resp: Value,
    fp: String,
    ts: f64,
}

/// In-memory idempotency cache with TTL and basic LRU semantics.  Mirrors
/// `_IdempotencyCache` (without the asyncio in-flight de-duplication, which is
/// a coroutine concern; the cache + purge + fingerprint match is reproduced).
pub struct IdempotencyCache {
    store: Mutex<Vec<(String, IdemEntry)>>,
    ttl: f64,
    max: usize,
}

impl IdempotencyCache {
    pub fn new(max_items: usize, ttl_seconds: f64) -> Self {
        IdempotencyCache {
            store: Mutex::new(Vec::new()),
            ttl: ttl_seconds,
            max: max_items,
        }
    }

    fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    fn purge_locked(&self, store: &mut Vec<(String, IdemEntry)>) {
        let now = Self::now();
        store.retain(|(_, v)| now - v.ts <= self.ttl);
        while store.len() > self.max {
            store.remove(0);
        }
    }

    /// Return a cached response if `key` is present with a matching
    /// fingerprint; otherwise `None`.  Mirrors the lookup half of
    /// `get_or_set`.
    pub fn get(&self, key: &str, fingerprint: &str) -> Option<Value> {
        let mut store = self.store.lock().unwrap();
        self.purge_locked(&mut store);
        store
            .iter()
            .find(|(k, v)| k == key && v.fp == fingerprint)
            .map(|(_, v)| v.resp.clone())
    }

    /// Store a computed response under `key`/`fingerprint`.  Mirrors the store
    /// half of `_compute_and_store`.
    pub fn set(&self, key: &str, fingerprint: &str, resp: Value) {
        let mut store = self.store.lock().unwrap();
        store.retain(|(k, _)| k != key);
        store.push((
            key.to_string(),
            IdemEntry {
                resp,
                fp: fingerprint.to_string(),
                ts: Self::now(),
            },
        ));
        self.purge_locked(&mut store);
    }
}

impl Default for IdempotencyCache {
    fn default() -> Self {
        IdempotencyCache::new(1000, 300.0)
    }
}

// ---------------------------------------------------------------------------
// ResponseStore — SQLite-backed LRU store for Responses API state
// ---------------------------------------------------------------------------

/// SQLite-backed LRU store for Responses API state.  Mirrors `ResponseStore`.
///
/// Each stored response includes the full internal conversation history so it
/// can be reconstructed via `previous_response_id`.  Persists across restarts;
/// falls back to in-memory SQLite if the on-disk path is unavailable.
pub struct ResponseStore {
    conn: Connection,
    max_size: i64,
}

impl ResponseStore {
    /// Open (or create) the store at `db_path`.  Pass `":memory:"` for a
    /// transient store.  Falls back to in-memory on open failure.
    pub fn new(max_size: i64, db_path: &str) -> Self {
        let conn = Connection::open(db_path)
            .unwrap_or_else(|_| Connection::open_in_memory().expect("in-memory sqlite"));
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            );",
        )
        .expect("create response_store tables");
        ResponseStore { conn, max_size }
    }

    fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Retrieve a stored response by ID (updates access time for LRU).
    pub fn get(&self, response_id: &str) -> Option<Value> {
        let data: Option<String> = self
            .conn
            .query_row(
                "SELECT data FROM responses WHERE response_id = ?",
                [response_id],
                |row| row.get(0),
            )
            .ok();
        let data = data?;
        let _ = self.conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            rusqlite::params![Self::now(), response_id],
        );
        serde_json::from_str(&data).ok()
    }

    /// Store a response, evicting the oldest if at capacity.
    pub fn put(&self, response_id: &str, data: &Value) {
        let serialized = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            rusqlite::params![response_id, serialized, Self::now()],
        );
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM responses", [], |row| row.get(0))
            .unwrap_or(0);
        if count > self.max_size {
            let _ = self.conn.execute(
                "DELETE FROM responses WHERE response_id IN \
                 (SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?)",
                [count - self.max_size],
            );
        }
    }

    /// Remove a response.  Returns true if found and deleted.
    pub fn delete(&self, response_id: &str) -> bool {
        let affected = self
            .conn
            .execute(
                "DELETE FROM responses WHERE response_id = ?",
                [response_id],
            )
            .unwrap_or(0);
        affected > 0
    }

    /// Get the latest response_id for a conversation name.
    pub fn get_conversation(&self, name: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT response_id FROM conversations WHERE name = ?",
                [name],
                |row| row.get(0),
            )
            .ok()
    }

    /// Map a conversation name to its latest response_id.
    pub fn set_conversation(&self, name: &str, response_id: &str) {
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            rusqlite::params![name, response_id],
        );
    }

    /// Number of stored responses.
    pub fn len(&self) -> i64 {
        self.conn
            .query_row("SELECT COUNT(*) FROM responses", [], |row| row.get(0))
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Run status tracking
// ---------------------------------------------------------------------------

/// Pollable run-status registry for dashboards/control-plane UIs.  Mirrors the
/// `_run_statuses` dict + `_set_run_status` on the adapter.
#[derive(Default)]
pub struct RunStatusRegistry {
    statuses: Mutex<HashMap<String, Map<String, Value>>>,
}

impl RunStatusRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Update pollable run status.  Mirrors `_set_run_status`.  `fields` are
    /// merged in after the standard envelope; if `fields` contains a
    /// `created_at`, it is used only when none is already set.
    pub fn set(&self, run_id: &str, status: &str, mut fields: Map<String, Value>) -> Value {
        let now = Self::now();
        let mut map = self.statuses.lock().unwrap();
        let entry = map.entry(run_id.to_string()).or_default();
        entry.insert("object".to_string(), json!("hermes.run"));
        entry.insert("run_id".to_string(), json!(run_id));
        entry.insert("status".to_string(), json!(status));
        entry.insert("updated_at".to_string(), json!(now));
        // setdefault("created_at", fields.pop("created_at", now))
        let created_default = fields.remove("created_at").unwrap_or(json!(now));
        entry
            .entry("created_at".to_string())
            .or_insert(created_default);
        for (k, v) in fields {
            entry.insert(k, v);
        }
        Value::Object(entry.clone())
    }

    /// Get a run's current status, if any.  Mirrors `_handle_get_run` lookup.
    pub fn get(&self, run_id: &str) -> Option<Value> {
        self.statuses
            .lock()
            .unwrap()
            .get(run_id)
            .map(|m| Value::Object(m.clone()))
    }

    /// Remove a run's status.
    pub fn remove(&self, run_id: &str) {
        self.statuses.lock().unwrap().remove(run_id);
    }

    /// IDs of terminal statuses older than the TTL.  Mirrors the
    /// `stale_statuses` sweep in `_sweep_orphaned_runs`.
    pub fn stale_terminal(&self, now: f64, ttl: f64) -> Vec<String> {
        let map = self.statuses.lock().unwrap();
        map.iter()
            .filter(|(_, status)| {
                let s = status.get("status").and_then(|v| v.as_str()).unwrap_or("");
                let terminal = matches!(s, "completed" | "failed" | "cancelled");
                let updated = status
                    .get("updated_at")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                terminal && now - updated > ttl
            })
            .map(|(k, _)| k.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Run-event payload builders (structured event streaming)
// ---------------------------------------------------------------------------

/// Build a `message.delta` run event.  Mirrors the `_text_cb` payload.
pub fn run_message_delta_event(run_id: &str, timestamp: f64, delta: &str) -> Value {
    json!({
        "event": "message.delta",
        "run_id": run_id,
        "timestamp": timestamp,
        "delta": delta,
    })
}

/// Build a `tool.started` run event.
pub fn run_tool_started_event(
    run_id: &str,
    timestamp: f64,
    tool: Option<&str>,
    preview: Option<&str>,
) -> Value {
    json!({
        "event": "tool.started",
        "run_id": run_id,
        "timestamp": timestamp,
        "tool": tool,
        "preview": preview,
    })
}

/// Build a `tool.completed` run event.  `duration` is rounded to 3 dp.
pub fn run_tool_completed_event(
    run_id: &str,
    timestamp: f64,
    tool: Option<&str>,
    duration: f64,
    is_error: bool,
) -> Value {
    json!({
        "event": "tool.completed",
        "run_id": run_id,
        "timestamp": timestamp,
        "tool": tool,
        "duration": round3(duration),
        "error": is_error,
    })
}

/// Build a `reasoning.available` run event.
pub fn run_reasoning_event(run_id: &str, timestamp: f64, text: &str) -> Value {
    json!({
        "event": "reasoning.available",
        "run_id": run_id,
        "timestamp": timestamp,
        "text": text,
    })
}

/// Build a terminal `run.completed` event.
pub fn run_completed_event(run_id: &str, timestamp: f64, output: &str, usage: &Value) -> Value {
    json!({
        "event": "run.completed",
        "run_id": run_id,
        "timestamp": timestamp,
        "output": output,
        "usage": usage,
    })
}

/// Build a terminal `run.failed` event.
pub fn run_failed_event(run_id: &str, timestamp: f64, error: &str) -> Value {
    json!({
        "event": "run.failed",
        "run_id": run_id,
        "timestamp": timestamp,
        "error": error,
    })
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coerce_port() {
        assert_eq!(coerce_port(&json!(9000), DEFAULT_PORT), 9000);
        assert_eq!(coerce_port(&json!("9001"), DEFAULT_PORT), 9001);
        assert_eq!(coerce_port(&json!("not-a-port"), DEFAULT_PORT), DEFAULT_PORT);
        assert_eq!(coerce_port(&json!(null), 4242), 4242);
    }

    #[test]
    fn test_normalize_chat_content_string() {
        assert_eq!(normalize_chat_content(&json!("hello")), "hello");
        assert_eq!(normalize_chat_content(&json!(null)), "");
    }

    #[test]
    fn test_normalize_chat_content_parts() {
        let content = json!([
            {"type": "text", "text": "hello"},
            {"type": "input_text", "text": "world"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ]);
        assert_eq!(normalize_chat_content(&content), "hello\nworld");
    }

    #[test]
    fn test_normalize_chat_content_truncation() {
        let big = "a".repeat(MAX_NORMALIZED_TEXT_LENGTH + 100);
        let out = normalize_chat_content(&json!(big));
        assert_eq!(out.chars().count(), MAX_NORMALIZED_TEXT_LENGTH);
    }

    #[test]
    fn test_normalize_multimodal_text_only_collapses() {
        let content = json!([
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]);
        let out = normalize_multimodal_content(&content).unwrap();
        assert_eq!(out, json!("a\nb"));
    }

    #[test]
    fn test_normalize_multimodal_image_passthrough() {
        let content = json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://img/x.png", "detail": "high"}},
        ]);
        let out = normalize_multimodal_content(&content).unwrap();
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(arr[1]["image_url"]["url"], "https://img/x.png");
        assert_eq!(arr[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn test_normalize_multimodal_data_url_validation() {
        let bad = json!([{"type": "input_image", "image_url": "data:text/plain,hello"}]);
        let err = normalize_multimodal_content(&bad).unwrap_err();
        assert_eq!(err.code, "unsupported_content_type");

        let good = json!([{"type": "input_image", "image_url": "data:image/png,abc"}]);
        assert!(normalize_multimodal_content(&good).is_ok());
    }

    #[test]
    fn test_normalize_multimodal_missing_url() {
        let bad = json!([{"type": "image_url", "image_url": {"url": "  "}}]);
        let err = normalize_multimodal_content(&bad).unwrap_err();
        assert_eq!(err.code, "invalid_image_url");
    }

    #[test]
    fn test_normalize_multimodal_file_rejected() {
        let bad = json!([{"type": "input_file", "file_id": "f"}]);
        let err = normalize_multimodal_content(&bad).unwrap_err();
        assert_eq!(err.code, "unsupported_content_type");
    }

    #[test]
    fn test_normalize_multimodal_unknown_type() {
        let bad = json!([{"type": "refusal_xyz", "text": "x"}]);
        let err = normalize_multimodal_content(&bad).unwrap_err();
        assert_eq!(err.code, "unsupported_content_type");
        assert!(err.message.contains("'refusal_xyz'"));
    }

    #[test]
    fn test_content_has_visible_payload() {
        assert!(content_has_visible_payload(&json!("hi")));
        assert!(!content_has_visible_payload(&json!("   ")));
        assert!(content_has_visible_payload(&json!([
            {"type": "image_url", "image_url": {"url": "http://x"}}
        ])));
        assert!(content_has_visible_payload(&json!([
            {"type": "text", "text": "ok"}
        ])));
        assert!(!content_has_visible_payload(&json!([
            {"type": "text", "text": "  "}
        ])));
    }

    #[test]
    fn test_openai_error_shape() {
        let e = openai_error("msg", None, Some("field"), Some("c"));
        assert_eq!(e["error"]["message"], "msg");
        assert_eq!(e["error"]["type"], "invalid_request_error");
        assert_eq!(e["error"]["param"], "field");
        assert_eq!(e["error"]["code"], "c");
    }

    #[test]
    fn test_derive_chat_session_id_stable() {
        let a = derive_chat_session_id(Some("sys"), "hi");
        let b = derive_chat_session_id(Some("sys"), "hi");
        assert_eq!(a, b);
        assert!(a.starts_with("api-"));
        assert_eq!(a.len(), 4 + 16);
        let c = derive_chat_session_id(None, "hi");
        assert_ne!(a, c);
    }

    #[test]
    fn test_make_request_fingerprint_deterministic() {
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        body.insert("stream".to_string(), json!(true));
        let f1 = make_request_fingerprint(&body, &["model", "stream", "missing"]);
        let f2 = make_request_fingerprint(&body, &["model", "stream", "missing"]);
        assert_eq!(f1, f2);
        assert_eq!(f1.len(), 64);
        // Different key order produces a different fingerprint (order-sensitive
        // like the Python dict).
        let f3 = make_request_fingerprint(&body, &["stream", "model", "missing"]);
        assert_ne!(f1, f3);
    }

    #[test]
    fn test_parse_cors_origins() {
        assert_eq!(parse_cors_origins(&json!("")), Vec::<String>::new());
        assert_eq!(
            parse_cors_origins(&json!("a.com, b.com ,")),
            vec!["a.com".to_string(), "b.com".to_string()]
        );
        assert_eq!(
            parse_cors_origins(&json!(["x", " y "])),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_cors_headers_for_origin() {
        let origins = vec!["https://app.com".to_string()];
        assert!(cors_headers_for_origin("", &origins).is_none());
        assert!(cors_headers_for_origin("https://other.com", &origins).is_none());
        let h = cors_headers_for_origin("https://app.com", &origins).unwrap();
        assert!(h
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "https://app.com"));
        assert!(h.iter().any(|(k, _)| k == "Vary"));

        let wildcard = vec!["*".to_string()];
        let hw = cors_headers_for_origin("https://anything.com", &wildcard).unwrap();
        assert!(hw
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "*"));
    }

    #[test]
    fn test_origin_allowed() {
        assert!(origin_allowed("", &[]));
        assert!(!origin_allowed("https://x.com", &[]));
        assert!(origin_allowed("https://x.com", &["*".to_string()]));
        assert!(origin_allowed("https://x.com", &["https://x.com".to_string()]));
        assert!(!origin_allowed("https://y.com", &["https://x.com".to_string()]));
    }

    #[test]
    fn test_check_auth() {
        assert_eq!(check_auth("", "anything"), AuthOutcome::Ok);
        assert_eq!(check_auth("secret", "Bearer secret"), AuthOutcome::Ok);
        assert_eq!(check_auth("secret", "Bearer  secret "), AuthOutcome::Ok);
        match check_auth("secret", "Bearer wrong") {
            AuthOutcome::Unauthorized(r) => assert_eq!(r.status, 401),
            _ => panic!("expected unauthorized"),
        }
        match check_auth("secret", "") {
            AuthOutcome::Unauthorized(r) => assert_eq!(r.status, 401),
            _ => panic!("expected unauthorized"),
        }
    }

    #[test]
    fn test_parse_session_key_header() {
        assert_eq!(parse_session_key_header("k", ""), SessionKeyOutcome::None);
        assert_eq!(
            parse_session_key_header("k", "scope-1"),
            SessionKeyOutcome::Key("scope-1".to_string())
        );
        match parse_session_key_header("", "scope") {
            SessionKeyOutcome::Error(r) => assert_eq!(r.status, 403),
            _ => panic!("expected error"),
        }
        match parse_session_key_header("k", "bad\nkey") {
            SessionKeyOutcome::Error(r) => assert_eq!(r.status, 400),
            _ => panic!("expected error"),
        }
        let long = "x".repeat(MAX_SESSION_HEADER_LEN + 1);
        match parse_session_key_header("k", &long) {
            SessionKeyOutcome::Error(r) => assert_eq!(r.status, 400),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_parse_session_id_header() {
        assert_eq!(parse_session_id_header("k", "  "), SessionIdOutcome::None);
        assert_eq!(
            parse_session_id_header("k", "sess-1"),
            SessionIdOutcome::SessionId("sess-1".to_string())
        );
        match parse_session_id_header("", "sess") {
            SessionIdOutcome::Error(r) => assert_eq!(r.status, 403),
            _ => panic!("expected error"),
        }
        match parse_session_id_header("k", "x\r\ny") {
            SessionIdOutcome::Error(r) => assert_eq!(r.status, 400),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_resolve_model_name() {
        assert_eq!(resolve_model_name("  custom-x ", None), "custom-x");
        assert_eq!(resolve_model_name("", Some("prod")), "prod");
        assert_eq!(resolve_model_name("", Some("default")), "hermes-agent");
        assert_eq!(resolve_model_name("", Some("custom")), "hermes-agent");
        assert_eq!(resolve_model_name("", None), "hermes-agent");
    }

    #[test]
    fn test_id_generators() {
        assert!(new_completion_id().starts_with("chatcmpl-"));
        assert_eq!(new_completion_id().len(), "chatcmpl-".len() + 29);
        assert!(new_response_id().starts_with("resp_"));
        assert_eq!(new_response_id().len(), "resp_".len() + 28);
        assert!(new_run_id().starts_with("run_"));
        assert_eq!(new_run_id().len(), "run_".len() + 32);
        assert!(new_message_item_id().starts_with("msg_"));
        // Hex only
        assert!(random_hex(16).chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_is_valid_job_id() {
        assert!(is_valid_job_id("abcdef012345"));
        assert!(!is_valid_job_id("ABCDEF012345"));
        assert!(!is_valid_job_id("abcdef01234")); // 11 chars
        assert!(!is_valid_job_id("abcdef0123456")); // 13 chars
        assert!(!is_valid_job_id("ghijklmnopqr"));
    }

    #[test]
    fn test_extract_output_items() {
        let result = json!({
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "search", "arguments": "{\"q\":1}"}}
                    ]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
            ],
            "final_response": "All done",
        });
        let items = extract_output_items(&result);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["name"], "search");
        assert_eq!(items[0]["call_id"], "call_1");
        assert_eq!(items[1]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call_1");
        assert_eq!(items[1]["output"], "result text");
        assert_eq!(items[2]["type"], "message");
        assert_eq!(items[2]["content"][0]["text"], "All done");
    }

    #[test]
    fn test_extract_output_items_error_fallback() {
        let result = json!({"messages": [], "error": "boom"});
        let items = extract_output_items(&result);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["content"][0]["text"], "boom");

        let empty = json!({});
        let items2 = extract_output_items(&empty);
        assert_eq!(items2[0]["content"][0]["text"], "(No response generated)");
    }

    #[test]
    fn test_build_models_response() {
        let v = build_models_response("hermes-agent", 1234);
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "hermes-agent");
        assert_eq!(v["data"][0]["created"], 1234);
        assert_eq!(v["data"][0]["parent"], Value::Null);
    }

    #[test]
    fn test_build_capabilities_response() {
        let v = build_capabilities_response("m", true, false);
        assert_eq!(v["model"], "m");
        assert_eq!(v["auth"]["required"], true);
        assert_eq!(v["features"]["cors"], false);
        assert_eq!(v["features"]["responses_api"], true);
        assert_eq!(
            v["endpoints"]["chat_completions"]["path"],
            "/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_chat_completion_response() {
        let v = build_chat_completion_response("id1", "m", 5, "hi", 1, 2, 3);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 1);
        assert_eq!(v["usage"]["completion_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 3);
    }

    #[test]
    fn test_idempotency_cache() {
        let cache = IdempotencyCache::new(2, 300.0);
        assert!(cache.get("k1", "fp1").is_none());
        cache.set("k1", "fp1", json!({"r": 1}));
        assert_eq!(cache.get("k1", "fp1"), Some(json!({"r": 1})));
        // Fingerprint mismatch — no hit.
        assert!(cache.get("k1", "fp2").is_none());
        // LRU eviction beyond max.
        cache.set("k2", "fp2", json!(2));
        cache.set("k3", "fp3", json!(3));
        assert!(cache.get("k1", "fp1").is_none());
        assert!(cache.get("k3", "fp3").is_some());
    }

    #[test]
    fn test_response_store_roundtrip() {
        let store = ResponseStore::new(MAX_STORED_RESPONSES, ":memory:");
        assert!(store.is_empty());
        store.put("resp_1", &json!({"response": {"x": 1}}));
        assert_eq!(store.len(), 1);
        let got = store.get("resp_1").unwrap();
        assert_eq!(got["response"]["x"], 1);
        assert!(store.delete("resp_1"));
        assert!(!store.delete("resp_1"));
        assert!(store.get("resp_1").is_none());
    }

    #[test]
    fn test_response_store_conversation_mapping() {
        let store = ResponseStore::new(MAX_STORED_RESPONSES, ":memory:");
        assert!(store.get_conversation("chat-a").is_none());
        store.set_conversation("chat-a", "resp_9");
        assert_eq!(store.get_conversation("chat-a").as_deref(), Some("resp_9"));
        store.set_conversation("chat-a", "resp_10");
        assert_eq!(store.get_conversation("chat-a").as_deref(), Some("resp_10"));
    }

    #[test]
    fn test_response_store_eviction() {
        let store = ResponseStore::new(2, ":memory:");
        store.put("a", &json!({"n": 1}));
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.put("b", &json!({"n": 2}));
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.put("c", &json!({"n": 3}));
        assert_eq!(store.len(), 2);
        // Oldest ("a") evicted.
        assert!(store.get("a").is_none());
        assert!(store.get("c").is_some());
    }

    #[test]
    fn test_run_status_registry() {
        let reg = RunStatusRegistry::new();
        let mut fields = Map::new();
        fields.insert("model".to_string(), json!("m"));
        let v = reg.set("run_1", "queued", fields);
        assert_eq!(v["status"], "queued");
        assert_eq!(v["object"], "hermes.run");
        assert_eq!(v["model"], "m");
        assert!(v.get("created_at").is_some());

        // Status update preserves created_at.
        let created = v["created_at"].clone();
        let v2 = reg.set("run_1", "running", Map::new());
        assert_eq!(v2["status"], "running");
        assert_eq!(v2["created_at"], created);

        assert!(reg.get("run_1").is_some());
        assert!(reg.get("missing").is_none());
        reg.remove("run_1");
        assert!(reg.get("run_1").is_none());
    }

    #[test]
    fn test_run_status_stale_terminal() {
        let reg = RunStatusRegistry::new();
        let mut old = Map::new();
        old.insert("updated_at".to_string(), json!(0.0));
        reg.set("done", "completed", old);
        reg.set("active", "running", Map::new());
        let now = 100000.0;
        let stale = reg.stale_terminal(now, RUN_STATUS_TTL);
        assert_eq!(stale, vec!["done".to_string()]);
    }

    #[test]
    fn test_run_event_builders() {
        let d = run_message_delta_event("r", 1.0, "hi");
        assert_eq!(d["event"], "message.delta");
        assert_eq!(d["delta"], "hi");

        let c = run_tool_completed_event("r", 2.0, Some("bash"), 1.23456, true);
        assert_eq!(c["event"], "tool.completed");
        assert_eq!(c["duration"], 1.235);
        assert_eq!(c["error"], true);

        let comp = run_completed_event("r", 3.0, "out", &json!({"total_tokens": 5}));
        assert_eq!(comp["event"], "run.completed");
        assert_eq!(comp["output"], "out");
        assert_eq!(comp["usage"]["total_tokens"], 5);

        let f = run_failed_event("r", 4.0, "boom");
        assert_eq!(f["event"], "run.failed");
        assert_eq!(f["error"], "boom");

        let reason = run_reasoning_event("r", 5.0, "thinking");
        assert_eq!(reason["event"], "reasoning.available");
        assert_eq!(reason["text"], "thinking");

        let started = run_tool_started_event("r", 6.0, Some("ls"), Some("preview"));
        assert_eq!(started["event"], "tool.started");
        assert_eq!(started["tool"], "ls");
        assert_eq!(started["preview"], "preview");
    }

    #[test]
    fn test_multimodal_validation_error() {
        let err = MultimodalError::new("invalid_image_url", "bad url");
        let resp = multimodal_validation_error(&err, "messages[0].content");
        assert_eq!(resp.status, 400);
        assert_eq!(resp.body["error"]["code"], "invalid_image_url");
        assert_eq!(resp.body["error"]["param"], "messages[0].content");
    }
}
