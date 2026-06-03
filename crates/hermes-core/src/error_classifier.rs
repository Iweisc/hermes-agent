//! API error classification for smart failover and recovery.
//!
//! Provides a structured taxonomy of API errors and a priority-ordered
//! classification pipeline that determines the correct recovery action
//! (retry, rotate credential, fallback to another provider, compress
//! context, or abort).
//!
//! This is a native Rust port of `agent/error_classifier.py`. Python's
//! classifier inspects a live exception object, walking its attributes
//! (`status_code`, `body`, `response.json()`, `__cause__` chain) and its
//! runtime type name. Rust has no equivalent live-exception introspection,
//! so callers extract those fields up front and pass them in via
//! [`ApiError`]. The classification *logic* (priority ordering, pattern
//! sets, thresholds) is reproduced faithfully.

use serde_json::Value;

// ── Error taxonomy ──────────────────────────────────────────────────────

/// Why an API call failed — determines recovery strategy.
///
/// Mirrors the Python `FailoverReason` enum. The string forms returned by
/// [`FailoverReason::as_str`] match the Python enum *values* exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailoverReason {
    /// Transient auth (401/403) — refresh/rotate.
    Auth,
    /// Auth failed after refresh — abort.
    AuthPermanent,
    /// 402 or confirmed credit exhaustion — rotate immediately.
    Billing,
    /// 429 or quota-based throttling — backoff then rotate.
    RateLimit,
    /// 503/529 — provider overloaded, backoff.
    Overloaded,
    /// 500/502 — internal server error, retry.
    ServerError,
    /// Connection/read timeout — rebuild client + retry.
    Timeout,
    /// Context too large — compress, not failover.
    ContextOverflow,
    /// 413 — compress payload.
    PayloadTooLarge,
    /// Native image part exceeds provider's per-image limit — shrink and retry.
    ImageTooLarge,
    /// 404 or invalid model — fallback to different model.
    ModelNotFound,
    /// Aggregator (e.g. OpenRouter) blocked the only endpoint due to account
    /// data/privacy policy.
    ProviderPolicyBlocked,
    /// 400 bad request — abort or strip + retry.
    FormatError,
    /// Anthropic thinking block sig invalid.
    ThinkingSignature,
    /// Anthropic "extra usage" tier gate.
    LongContextTier,
    /// Anthropic OAuth subscription rejects 1M context beta — disable beta and retry.
    OauthLongContextBetaForbidden,
    /// llama.cpp json-schema-to-grammar rejects regex escapes in `pattern` /
    /// `format` — strip from tools and retry.
    LlamaCppGrammarPattern,
    /// Unclassifiable — retry with backoff.
    Unknown,
}

impl FailoverReason {
    /// The canonical string value (matches the Python enum value).
    pub fn as_str(self) -> &'static str {
        match self {
            FailoverReason::Auth => "auth",
            FailoverReason::AuthPermanent => "auth_permanent",
            FailoverReason::Billing => "billing",
            FailoverReason::RateLimit => "rate_limit",
            FailoverReason::Overloaded => "overloaded",
            FailoverReason::ServerError => "server_error",
            FailoverReason::Timeout => "timeout",
            FailoverReason::ContextOverflow => "context_overflow",
            FailoverReason::PayloadTooLarge => "payload_too_large",
            FailoverReason::ImageTooLarge => "image_too_large",
            FailoverReason::ModelNotFound => "model_not_found",
            FailoverReason::ProviderPolicyBlocked => "provider_policy_blocked",
            FailoverReason::FormatError => "format_error",
            FailoverReason::ThinkingSignature => "thinking_signature",
            FailoverReason::LongContextTier => "long_context_tier",
            FailoverReason::OauthLongContextBetaForbidden => "oauth_long_context_beta_forbidden",
            FailoverReason::LlamaCppGrammarPattern => "llama_cpp_grammar_pattern",
            FailoverReason::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for FailoverReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Classification input ─────────────────────────────────────────────────

/// The extracted facts about an API failure that the classifier needs.
///
/// In Python the classifier walked a live exception. Here the caller is
/// responsible for the equivalent extraction:
///   - `status_code`: from `error.status_code` / `error.status` (walking the
///     `__cause__`/`__context__` chain up to 5 deep).
///   - `error_type`: `type(error).__name__`.
///   - `body`: `error.body` if a dict, else `error.response.json()` if a dict.
///   - `message`: `str(error)` (the raw text representation).
#[derive(Debug, Clone, Default)]
pub struct ApiError {
    /// HTTP status code, if one was found on the error or its cause chain.
    pub status_code: Option<i64>,
    /// Runtime type name of the error (`type(error).__name__` in Python).
    pub error_type: String,
    /// Structured error body (JSON object), if available.
    pub body: Option<Value>,
    /// `str(error)` — the raw message representation.
    pub message: String,
}

impl ApiError {
    /// Convenience constructor for a bare message with no status/body.
    pub fn from_message(message: impl Into<String>) -> Self {
        ApiError {
            status_code: None,
            error_type: String::new(),
            body: None,
            message: message.into(),
        }
    }
}

/// Optional classification context (provider / model / token sizing).
///
/// Defaults mirror the Python keyword-argument defaults: empty provider and
/// model, `approx_tokens = 0`, `context_length = 200_000`, `num_messages = 0`.
#[derive(Debug, Clone)]
pub struct ClassifyContext {
    pub provider: String,
    pub model: String,
    pub approx_tokens: i64,
    pub context_length: i64,
    pub num_messages: i64,
}

impl Default for ClassifyContext {
    fn default() -> Self {
        ClassifyContext {
            provider: String::new(),
            model: String::new(),
            approx_tokens: 0,
            context_length: 200_000,
            num_messages: 0,
        }
    }
}

// ── Classification result ───────────────────────────────────────────────

/// Structured classification of an API error with recovery hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedError {
    pub reason: FailoverReason,
    pub status_code: Option<i64>,
    pub provider: String,
    pub model: String,
    pub message: String,

    // Recovery action hints — the retry loop checks these instead of
    // re-classifying the error itself.
    pub retryable: bool,
    pub should_compress: bool,
    pub should_rotate_credential: bool,
    pub should_fallback: bool,
}

impl ClassifiedError {
    /// True if the reason is one of the auth variants.
    pub fn is_auth(&self) -> bool {
        matches!(
            self.reason,
            FailoverReason::Auth | FailoverReason::AuthPermanent
        )
    }
}

/// Mutable override set applied on top of the per-call defaults, mirroring
/// the Python `_result(reason, **overrides)` helper. Any field left `None`
/// keeps the dataclass default.
#[derive(Default)]
struct Overrides {
    retryable: Option<bool>,
    should_compress: Option<bool>,
    should_rotate_credential: Option<bool>,
    should_fallback: Option<bool>,
}

// ── Provider-specific patterns ──────────────────────────────────────────

/// Patterns that indicate billing exhaustion (not transient rate limit).
const BILLING_PATTERNS: &[&str] = &[
    "insufficient credits",
    "insufficient_quota",
    "insufficient balance",
    "credit balance",
    "credits have been exhausted",
    "top up your credits",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
];

/// Patterns that indicate rate limiting (transient, will resolve).
const RATE_LIMIT_PATTERNS: &[&str] = &[
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "rate increased too quickly", // Alibaba/DashScope throttling
    // AWS Bedrock throttling
    "throttlingexception",
    "too many concurrent requests",
    "servicequotaexceededexception",
];

/// Usage-limit patterns that need disambiguation (could be billing OR rate_limit).
const USAGE_LIMIT_PATTERNS: &[&str] = &[
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
];

/// Patterns confirming usage limit is transient (not billing).
const USAGE_LIMIT_TRANSIENT_SIGNALS: &[&str] = &[
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
];

/// Payload-too-large patterns detected from message text (no status_code).
const PAYLOAD_TOO_LARGE_PATTERNS: &[&str] = &[
    "request entity too large",
    "payload too large",
    "error code: 413",
];

/// Image-size patterns. Matched against 400 bodies (not 413).
const IMAGE_TOO_LARGE_PATTERNS: &[&str] = &[
    "image exceeds",      // Anthropic: "image exceeds 5 MB maximum"
    "image too large",    // generic
    "image_too_large",    // error_code variant
    "image size exceeds", // variant
];

/// Context overflow patterns.
const CONTEXT_OVERFLOW_PATTERNS: &[&str] = &[
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "max_tokens",
    "maximum number of tokens",
    // vLLM / local inference server patterns
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length", // "engine prompt length X exceeds"
    "input is too long",
    "maximum model length",
    // Ollama patterns
    "context length exceeded",
    "truncating input",
    // llama.cpp / llama-server patterns
    "slot context", // "slot context: N tokens, prompt N tokens"
    "n_ctx_slot",
    // Chinese error messages (some providers return these)
    "超过最大长度",
    "上下文长度",
    // AWS Bedrock Converse API error patterns
    "input is too long",
    "max input token",
    "input token",
    "exceeds the maximum number of input tokens",
];

/// Model not found patterns.
const MODEL_NOT_FOUND_PATTERNS: &[&str] = &[
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
];

/// OpenRouter aggregator policy-block patterns.
const PROVIDER_POLICY_BLOCKED_PATTERNS: &[&str] = &[
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
];

/// Auth patterns (non-status-code signals).
const AUTH_PATTERNS: &[&str] = &[
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
];

/// Transport error type names.
const TRANSPORT_ERROR_TYPES: &[&str] = &[
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "ConnectError",
    "RemoteProtocolError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
    "TimeoutError",
    "ReadError",
    "ServerDisconnectedError",
    // SSL/TLS transport errors
    "SSLError",
    "SSLZeroReturnError",
    "SSLWantReadError",
    "SSLWantWriteError",
    "SSLEOFError",
    "SSLSyscallError",
    // OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
];

/// Server disconnect patterns (no status code, but transport-level).
const SERVER_DISCONNECT_PATTERNS: &[&str] = &[
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
];

/// SSL/TLS transient failure patterns — distinct from disconnect patterns.
const SSL_TRANSIENT_PATTERNS: &[&str] = &[
    // Space-separated (human-readable form)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    // Underscore-separated (OpenSSL error code tokens)
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    // Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
];

/// `any(p in haystack for p in patterns)`.
fn any_in(haystack: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| haystack.contains(p))
}

// ── Classification pipeline ─────────────────────────────────────────────

/// Classify an API error into a structured recovery recommendation.
///
/// Priority-ordered pipeline:
///   1. Special-case provider-specific patterns (thinking sigs, tier gates)
///   2. HTTP status code + message-aware refinement
///   3. Error code classification (from body)
///   4. Message pattern matching (billing vs rate_limit vs context vs auth)
///   5. SSL/TLS transient alert patterns → retry as timeout
///   6. Server disconnect + large session → context overflow
///   7. Transport error heuristics
///   8. Fallback: unknown (retryable with backoff)
pub fn classify_api_error(error: &ApiError, ctx: &ClassifyContext) -> ClassifiedError {
    let mut status_code = extract_status_code(error);
    let error_type = error.error_type.as_str();
    // Copilot/GitHub Models RateLimitError may not set .status_code; force 429.
    if status_code.is_none() && error_type == "RateLimitError" {
        status_code = Some(429);
    }
    let body = extract_error_body(error);
    let error_code = extract_error_code(&body);

    // Build a comprehensive lowercased error message for pattern matching,
    // combining str(error), body.error.message, and metadata.raw inner message.
    let raw_msg = error.message.to_lowercase();
    let mut body_msg = String::new();
    let mut metadata_msg = String::new();
    if let Value::Object(map) = &body {
        if let Some(Value::Object(err_obj)) = map.get("error") {
            body_msg = str_field_lower(err_obj.get("message"));
            // Parse metadata.raw for wrapped provider errors.
            if let Some(Value::Object(metadata)) = err_obj.get("metadata") {
                if let Some(Value::String(raw_json)) = metadata.get("raw") {
                    if !raw_json.trim().is_empty() {
                        if let Ok(Value::Object(inner)) = serde_json::from_str::<Value>(raw_json) {
                            if let Some(Value::Object(inner_err)) = inner.get("error") {
                                metadata_msg = str_field_lower(inner_err.get("message"));
                            }
                        }
                    }
                }
            }
        }
        if body_msg.is_empty() {
            body_msg = str_field_lower(map.get("message"));
        }
    }

    let mut parts: Vec<&str> = vec![raw_msg.as_str()];
    if !body_msg.is_empty() && !raw_msg.contains(&body_msg) {
        parts.push(body_msg.as_str());
    }
    if !metadata_msg.is_empty()
        && !raw_msg.contains(&metadata_msg)
        && !body_msg.contains(&metadata_msg)
    {
        parts.push(metadata_msg.as_str());
    }
    let error_msg = parts.join(" ");

    let provider_lower = ctx.provider.trim().to_lowercase();
    let model_lower = ctx.model.trim().to_lowercase();

    let extracted_message = extract_message(error, &body);
    let result = |reason: FailoverReason, ov: Overrides| -> ClassifiedError {
        ClassifiedError {
            reason,
            status_code,
            provider: ctx.provider.clone(),
            model: ctx.model.clone(),
            message: extracted_message.clone(),
            retryable: ov.retryable.unwrap_or(true),
            should_compress: ov.should_compress.unwrap_or(false),
            should_rotate_credential: ov.should_rotate_credential.unwrap_or(false),
            should_fallback: ov.should_fallback.unwrap_or(false),
        }
    };

    // ── 1. Provider-specific patterns (highest priority) ────────────

    // Anthropic thinking block signature invalid (400).
    if status_code == Some(400)
        && error_msg.contains("signature")
        && error_msg.contains("thinking")
    {
        return result(
            FailoverReason::ThinkingSignature,
            Overrides {
                retryable: Some(true),
                should_compress: Some(false),
                ..Default::default()
            },
        );
    }

    // Anthropic long-context tier gate (429 "extra usage" + "long context").
    if status_code == Some(429)
        && error_msg.contains("extra usage")
        && error_msg.contains("long context")
    {
        return result(
            FailoverReason::LongContextTier,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        );
    }

    // Anthropic OAuth subscription rejects the 1M-context beta header.
    if status_code == Some(400)
        && error_msg.contains("long context beta")
        && error_msg.contains("not yet available")
    {
        return result(
            FailoverReason::OauthLongContextBetaForbidden,
            Overrides {
                retryable: Some(true),
                should_compress: Some(false),
                ..Default::default()
            },
        );
    }

    // llama.cpp json-schema-to-grammar rejects regex escapes (400).
    if status_code == Some(400)
        && (error_msg.contains("error parsing grammar")
            || error_msg.contains("json-schema-to-grammar")
            || (error_msg.contains("unable to generate parser")
                && error_msg.contains("template")))
    {
        return result(
            FailoverReason::LlamaCppGrammarPattern,
            Overrides {
                retryable: Some(true),
                should_compress: Some(false),
                ..Default::default()
            },
        );
    }

    // ── 2. HTTP status code classification ──────────────────────────

    if let Some(code) = status_code {
        if let Some(classified) = classify_by_status(
            code,
            &error_msg,
            &error_code,
            &body,
            &provider_lower,
            &model_lower,
            ctx,
            &result,
        ) {
            return classified;
        }
    }

    // ── 3. Error code classification ────────────────────────────────

    if !error_code.is_empty() {
        if let Some(classified) = classify_by_error_code(&error_code, &result) {
            return classified;
        }
    }

    // ── 4. Message pattern matching (no status code) ────────────────

    if let Some(classified) = classify_by_message(&error_msg, &result) {
        return classified;
    }

    // ── 5. SSL/TLS transient errors → retry as timeout ──────────────
    if any_in(&error_msg, SSL_TRANSIENT_PATTERNS) {
        return result(
            FailoverReason::Timeout,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        );
    }

    // ── 6. Server disconnect + large session → context overflow ─────
    let is_disconnect = any_in(&error_msg, SERVER_DISCONNECT_PATTERNS);
    // `not status_code` in Python is true for None and for 0.
    let no_status = matches!(status_code, None | Some(0));
    if is_disconnect && no_status {
        let is_large = (ctx.approx_tokens as f64) > (ctx.context_length as f64) * 0.6
            || (ctx.context_length <= 256_000
                && (ctx.approx_tokens > 120_000 || ctx.num_messages > 200));
        if is_large {
            return result(
                FailoverReason::ContextOverflow,
                Overrides {
                    retryable: Some(true),
                    should_compress: Some(true),
                    ..Default::default()
                },
            );
        }
        return result(
            FailoverReason::Timeout,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        );
    }

    // ── 7. Transport / timeout heuristics ───────────────────────────
    if TRANSPORT_ERROR_TYPES.contains(&error_type)
        || matches!(error_type, "TimeoutError" | "ConnectionError" | "OSError")
    {
        return result(
            FailoverReason::Timeout,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        );
    }

    // ── 8. Fallback: unknown ────────────────────────────────────────
    result(
        FailoverReason::Unknown,
        Overrides {
            retryable: Some(true),
            ..Default::default()
        },
    )
}

// ── Status code classification ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn classify_by_status<F>(
    status_code: i64,
    error_msg: &str,
    error_code: &str,
    body: &Value,
    _provider: &str,
    _model: &str,
    ctx: &ClassifyContext,
    result_fn: &F,
) -> Option<ClassifiedError>
where
    F: Fn(FailoverReason, Overrides) -> ClassifiedError,
{
    if status_code == 401 {
        return Some(result_fn(
            FailoverReason::Auth,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 403 {
        // OpenRouter 403 "key limit exceeded" is actually billing.
        if error_msg.contains("key limit exceeded") || error_msg.contains("spending limit") {
            return Some(result_fn(
                FailoverReason::Billing,
                Overrides {
                    retryable: Some(false),
                    should_rotate_credential: Some(true),
                    should_fallback: Some(true),
                    ..Default::default()
                },
            ));
        }
        return Some(result_fn(
            FailoverReason::Auth,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 402 {
        return Some(classify_402(error_msg, result_fn));
    }

    if status_code == 404 {
        if any_in(error_msg, PROVIDER_POLICY_BLOCKED_PATTERNS) {
            return Some(result_fn(
                FailoverReason::ProviderPolicyBlocked,
                Overrides {
                    retryable: Some(false),
                    should_fallback: Some(false),
                    ..Default::default()
                },
            ));
        }
        if any_in(error_msg, MODEL_NOT_FOUND_PATTERNS) {
            return Some(result_fn(
                FailoverReason::ModelNotFound,
                Overrides {
                    retryable: Some(false),
                    should_fallback: Some(true),
                    ..Default::default()
                },
            ));
        }
        // Generic 404 with no "model not found" signal — treat as unknown.
        return Some(result_fn(
            FailoverReason::Unknown,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 413 {
        return Some(result_fn(
            FailoverReason::PayloadTooLarge,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 429 {
        return Some(result_fn(
            FailoverReason::RateLimit,
            Overrides {
                retryable: Some(true),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 400 {
        return Some(classify_400(error_msg, error_code, body, ctx, result_fn));
    }

    if status_code == 500 || status_code == 502 {
        return Some(result_fn(
            FailoverReason::ServerError,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        ));
    }

    if status_code == 503 || status_code == 529 {
        return Some(result_fn(
            FailoverReason::Overloaded,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        ));
    }

    // Other 4xx — non-retryable.
    if (400..500).contains(&status_code) {
        return Some(result_fn(
            FailoverReason::FormatError,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    // Other 5xx — retryable.
    if (500..600).contains(&status_code) {
        return Some(result_fn(
            FailoverReason::ServerError,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        ));
    }

    None
}

/// Disambiguate 402: billing exhaustion vs transient usage limit.
fn classify_402<F>(error_msg: &str, result_fn: &F) -> ClassifiedError
where
    F: Fn(FailoverReason, Overrides) -> ClassifiedError,
{
    let has_usage_limit = any_in(error_msg, USAGE_LIMIT_PATTERNS);
    let has_transient_signal = any_in(error_msg, USAGE_LIMIT_TRANSIENT_SIGNALS);

    if has_usage_limit && has_transient_signal {
        return result_fn(
            FailoverReason::RateLimit,
            Overrides {
                retryable: Some(true),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        );
    }

    result_fn(
        FailoverReason::Billing,
        Overrides {
            retryable: Some(false),
            should_rotate_credential: Some(true),
            should_fallback: Some(true),
            ..Default::default()
        },
    )
}

/// Classify 400 Bad Request — context overflow, format error, or generic.
fn classify_400<F>(
    error_msg: &str,
    _error_code: &str,
    body: &Value,
    ctx: &ClassifyContext,
    result_fn: &F,
) -> ClassifiedError
where
    F: Fn(FailoverReason, Overrides) -> ClassifiedError,
{
    // Image-too-large from 400 (checked before context_overflow).
    if any_in(error_msg, IMAGE_TOO_LARGE_PATTERNS) {
        return result_fn(
            FailoverReason::ImageTooLarge,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        );
    }

    // Context overflow from 400.
    if any_in(error_msg, CONTEXT_OVERFLOW_PATTERNS) {
        return result_fn(
            FailoverReason::ContextOverflow,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        );
    }

    // Some providers return model-not-found as 400 instead of 404.
    if any_in(error_msg, PROVIDER_POLICY_BLOCKED_PATTERNS) {
        return result_fn(
            FailoverReason::ProviderPolicyBlocked,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(false),
                ..Default::default()
            },
        );
    }
    if any_in(error_msg, MODEL_NOT_FOUND_PATTERNS) {
        return result_fn(
            FailoverReason::ModelNotFound,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(true),
                ..Default::default()
            },
        );
    }

    // Some providers return rate limit / billing as 400 instead of 429/402.
    if any_in(error_msg, RATE_LIMIT_PATTERNS) {
        return result_fn(
            FailoverReason::RateLimit,
            Overrides {
                retryable: Some(true),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        );
    }
    if any_in(error_msg, BILLING_PATTERNS) {
        return result_fn(
            FailoverReason::Billing,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        );
    }

    // Generic 400 + large session → probable context overflow.
    let mut err_body_msg = String::new();
    if let Value::Object(map) = body {
        if let Some(Value::Object(err_obj)) = map.get("error") {
            err_body_msg = str_field_trim_lower(err_obj.get("message"));
        }
        if err_body_msg.is_empty() {
            err_body_msg = str_field_trim_lower(map.get("message"));
        }
    }
    let is_generic = err_body_msg.len() < 30 || err_body_msg == "error" || err_body_msg.is_empty();
    let is_large = (ctx.approx_tokens as f64) > (ctx.context_length as f64) * 0.4
        || (ctx.context_length <= 256_000
            && (ctx.approx_tokens > 80_000 || ctx.num_messages > 80));

    if is_generic && is_large {
        return result_fn(
            FailoverReason::ContextOverflow,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        );
    }

    // Non-retryable format error.
    result_fn(
        FailoverReason::FormatError,
        Overrides {
            retryable: Some(false),
            should_fallback: Some(true),
            ..Default::default()
        },
    )
}

// ── Error code classification ───────────────────────────────────────────

/// Classify by structured error codes from the response body.
fn classify_by_error_code<F>(error_code: &str, result_fn: &F) -> Option<ClassifiedError>
where
    F: Fn(FailoverReason, Overrides) -> ClassifiedError,
{
    let code_lower = error_code.to_lowercase();

    match code_lower.as_str() {
        "resource_exhausted" | "throttled" | "rate_limit_exceeded" => Some(result_fn(
            FailoverReason::RateLimit,
            Overrides {
                retryable: Some(true),
                should_rotate_credential: Some(true),
                ..Default::default()
            },
        )),
        "insufficient_quota" | "billing_not_active" | "payment_required" => Some(result_fn(
            FailoverReason::Billing,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        )),
        "model_not_found" | "model_not_available" | "invalid_model" => Some(result_fn(
            FailoverReason::ModelNotFound,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(true),
                ..Default::default()
            },
        )),
        "context_length_exceeded" | "max_tokens_exceeded" => Some(result_fn(
            FailoverReason::ContextOverflow,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        )),
        _ => None,
    }
}

// ── Message pattern classification ──────────────────────────────────────

/// Classify based on error message patterns when no status code is available.
fn classify_by_message<F>(error_msg: &str, result_fn: &F) -> Option<ClassifiedError>
where
    F: Fn(FailoverReason, Overrides) -> ClassifiedError,
{
    // Payload-too-large patterns.
    if any_in(error_msg, PAYLOAD_TOO_LARGE_PATTERNS) {
        return Some(result_fn(
            FailoverReason::PayloadTooLarge,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        ));
    }

    // Image-too-large patterns.
    if any_in(error_msg, IMAGE_TOO_LARGE_PATTERNS) {
        return Some(result_fn(
            FailoverReason::ImageTooLarge,
            Overrides {
                retryable: Some(true),
                ..Default::default()
            },
        ));
    }

    // Usage-limit patterns need the same disambiguation as 402.
    if any_in(error_msg, USAGE_LIMIT_PATTERNS) {
        if any_in(error_msg, USAGE_LIMIT_TRANSIENT_SIGNALS) {
            return Some(result_fn(
                FailoverReason::RateLimit,
                Overrides {
                    retryable: Some(true),
                    should_rotate_credential: Some(true),
                    should_fallback: Some(true),
                    ..Default::default()
                },
            ));
        }
        return Some(result_fn(
            FailoverReason::Billing,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    // Billing patterns.
    if any_in(error_msg, BILLING_PATTERNS) {
        return Some(result_fn(
            FailoverReason::Billing,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    // Rate limit patterns.
    if any_in(error_msg, RATE_LIMIT_PATTERNS) {
        return Some(result_fn(
            FailoverReason::RateLimit,
            Overrides {
                retryable: Some(true),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    // Context overflow patterns.
    if any_in(error_msg, CONTEXT_OVERFLOW_PATTERNS) {
        return Some(result_fn(
            FailoverReason::ContextOverflow,
            Overrides {
                retryable: Some(true),
                should_compress: Some(true),
                ..Default::default()
            },
        ));
    }

    // Auth patterns.
    if any_in(error_msg, AUTH_PATTERNS) {
        return Some(result_fn(
            FailoverReason::Auth,
            Overrides {
                retryable: Some(false),
                should_rotate_credential: Some(true),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    // Provider policy-block — check before model_not_found.
    if any_in(error_msg, PROVIDER_POLICY_BLOCKED_PATTERNS) {
        return Some(result_fn(
            FailoverReason::ProviderPolicyBlocked,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(false),
                ..Default::default()
            },
        ));
    }

    // Model not found patterns.
    if any_in(error_msg, MODEL_NOT_FOUND_PATTERNS) {
        return Some(result_fn(
            FailoverReason::ModelNotFound,
            Overrides {
                retryable: Some(false),
                should_fallback: Some(true),
                ..Default::default()
            },
        ));
    }

    None
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// In Python this walks `error.status_code` / `error.status` and the
/// `__cause__`/`__context__` chain. The caller has already done that walk and
/// stuffed the result into [`ApiError::status_code`]; we just validate the
/// `.status` constraint (`100 <= code < 600`) is not re-applied here because
/// `status_code` is already the resolved value. Returned verbatim.
fn extract_status_code(error: &ApiError) -> Option<i64> {
    error.status_code
}

/// Extract the structured error body. The caller supplies a JSON body if one
/// was found on `error.body` or `error.response.json()`; otherwise we treat it
/// as an empty object (matching Python's `{}` sentinel).
fn extract_error_body(error: &ApiError) -> Value {
    match &error.body {
        Some(v @ Value::Object(_)) => v.clone(),
        _ => Value::Object(serde_json::Map::new()),
    }
}

/// Extract an error code string from the response body.
fn extract_error_code(body: &Value) -> String {
    let map = match body {
        Value::Object(m) if !m.is_empty() => m,
        _ => return String::new(),
    };

    if let Some(Value::Object(error_obj)) = map.get("error") {
        // code or type, first truthy string.
        let code = error_obj
            .get("code")
            .and_then(truthy_string)
            .or_else(|| error_obj.get("type").and_then(truthy_string));
        if let Some(c) = code {
            let trimmed = c.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    // Top-level code or error_code (str or int).
    for key in ["code", "error_code"] {
        match map.get(key) {
            Some(Value::String(s)) => {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
            Some(Value::Number(n)) => {
                // Python `body.get("code") or body.get("error_code")` skips a
                // falsy 0, but a non-zero int is stringified.
                if n.as_f64() != Some(0.0) {
                    return n.to_string();
                }
            }
            _ => {}
        }
    }
    String::new()
}

/// Extract the most informative error message (truncated to 500 chars).
fn extract_message(error: &ApiError, body: &Value) -> String {
    if let Value::Object(map) = body {
        if !map.is_empty() {
            if let Some(Value::Object(error_obj)) = map.get("error") {
                if let Some(Value::String(msg)) = error_obj.get("message") {
                    let trimmed = msg.trim();
                    if !trimmed.is_empty() {
                        return truncate_chars(trimmed, 500);
                    }
                }
            }
            if let Some(Value::String(msg)) = map.get("message") {
                let trimmed = msg.trim();
                if !trimmed.is_empty() {
                    return truncate_chars(trimmed, 500);
                }
            }
        }
    }
    truncate_chars(&error.message, 500)
}

/// `str(value or "").lower()` for an optional JSON value treated as a string.
/// Only `Value::String` contributes text; everything else (including null and
/// missing) yields an empty string — matching `str(... or "")` where the
/// Python code reads a `.get("message")` it expects to be a string.
fn str_field_lower(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.to_lowercase(),
        _ => String::new(),
    }
}

/// Like [`str_field_lower`] but trims before lowercasing (`.strip().lower()`).
fn str_field_trim_lower(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_lowercase(),
        _ => String::new(),
    }
}

/// Returns the string if it is a non-empty (truthy) JSON string, used to mimic
/// Python's `a or b` short-circuit on string fields.
fn truthy_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Truncate to at most `n` Unicode scalar values (Python `[:500]` on a str
/// slices by code points, not bytes).
fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn classify(err: ApiError) -> ClassifiedError {
        classify_api_error(&err, &ClassifyContext::default())
    }

    fn classify_ctx(err: ApiError, ctx: ClassifyContext) -> ClassifiedError {
        classify_api_error(&err, &ctx)
    }

    fn status(code: i64, msg: &str) -> ApiError {
        ApiError {
            status_code: Some(code),
            error_type: String::new(),
            body: None,
            message: msg.to_string(),
        }
    }

    #[test]
    fn enum_string_values_match_python() {
        assert_eq!(FailoverReason::Auth.as_str(), "auth");
        assert_eq!(FailoverReason::RateLimit.as_str(), "rate_limit");
        assert_eq!(FailoverReason::ContextOverflow.as_str(), "context_overflow");
        assert_eq!(
            FailoverReason::OauthLongContextBetaForbidden.as_str(),
            "oauth_long_context_beta_forbidden"
        );
        assert_eq!(FailoverReason::Unknown.as_str(), "unknown");
    }

    #[test]
    fn status_401_is_auth_nonretryable_rotate_fallback() {
        let c = classify(status(401, "Unauthorized"));
        assert_eq!(c.reason, FailoverReason::Auth);
        assert!(!c.retryable);
        assert!(c.should_rotate_credential);
        assert!(c.should_fallback);
        assert!(c.is_auth());
    }

    #[test]
    fn status_403_key_limit_is_billing() {
        let c = classify(status(403, "key limit exceeded"));
        assert_eq!(c.reason, FailoverReason::Billing);
        assert!(!c.retryable);
        assert!(c.should_rotate_credential);
    }

    #[test]
    fn status_403_generic_is_auth() {
        let c = classify(status(403, "Forbidden"));
        assert_eq!(c.reason, FailoverReason::Auth);
        assert!(!c.retryable);
        assert!(c.should_fallback);
        assert!(!c.should_rotate_credential);
    }

    #[test]
    fn status_402_billing_vs_transient_usage_limit() {
        let billing = classify(status(402, "Payment required: insufficient credits"));
        assert_eq!(billing.reason, FailoverReason::Billing);
        assert!(!billing.retryable);

        let transient = classify(status(402, "Usage limit reached, try again in 5 minutes"));
        assert_eq!(transient.reason, FailoverReason::RateLimit);
        assert!(transient.retryable);
    }

    #[test]
    fn status_404_policy_block_vs_model_not_found_vs_unknown() {
        let policy = classify(status(
            404,
            "No endpoints available matching your data policy. Configure: ...",
        ));
        assert_eq!(policy.reason, FailoverReason::ProviderPolicyBlocked);
        assert!(!policy.retryable);
        assert!(!policy.should_fallback);

        let nf = classify(status(404, "model not found"));
        assert_eq!(nf.reason, FailoverReason::ModelNotFound);
        assert!(!nf.retryable);
        assert!(nf.should_fallback);

        let generic = classify(status(404, "Not Found"));
        assert_eq!(generic.reason, FailoverReason::Unknown);
        assert!(generic.retryable);
    }

    #[test]
    fn status_413_payload_too_large() {
        let c = classify(status(413, "Request Entity Too Large"));
        assert_eq!(c.reason, FailoverReason::PayloadTooLarge);
        assert!(c.retryable);
        assert!(c.should_compress);
    }

    #[test]
    fn status_429_rate_limit() {
        let c = classify(status(429, "Too Many Requests"));
        assert_eq!(c.reason, FailoverReason::RateLimit);
        assert!(c.retryable);
        assert!(c.should_rotate_credential);
        assert!(c.should_fallback);
    }

    #[test]
    fn status_500_502_server_error_503_529_overloaded() {
        assert_eq!(
            classify(status(500, "boom")).reason,
            FailoverReason::ServerError
        );
        assert_eq!(
            classify(status(502, "bad gateway")).reason,
            FailoverReason::ServerError
        );
        assert_eq!(
            classify(status(503, "unavailable")).reason,
            FailoverReason::Overloaded
        );
        assert_eq!(
            classify(status(529, "overloaded")).reason,
            FailoverReason::Overloaded
        );
    }

    #[test]
    fn other_4xx_is_format_error_other_5xx_server_error() {
        let c = classify(status(418, "I'm a teapot"));
        assert_eq!(c.reason, FailoverReason::FormatError);
        assert!(!c.retryable);
        assert!(c.should_fallback);

        let s = classify(status(504, "gateway timeout"));
        assert_eq!(s.reason, FailoverReason::ServerError);
        assert!(s.retryable);
    }

    #[test]
    fn thinking_signature_400() {
        let c = classify(status(
            400,
            "Invalid thinking block: signature verification failed",
        ));
        assert_eq!(c.reason, FailoverReason::ThinkingSignature);
        assert!(c.retryable);
        assert!(!c.should_compress);
    }

    #[test]
    fn long_context_tier_429() {
        let c = classify(status(
            429,
            "extra usage required for long context requests",
        ));
        assert_eq!(c.reason, FailoverReason::LongContextTier);
        assert!(c.should_compress);
    }

    #[test]
    fn oauth_long_context_beta_forbidden_400() {
        let c = classify(status(
            400,
            "The long context beta is not yet available for this subscription.",
        ));
        assert_eq!(c.reason, FailoverReason::OauthLongContextBetaForbidden);
        assert!(c.retryable);
    }

    #[test]
    fn llama_cpp_grammar_pattern_400() {
        let c = classify(status(400, "error parsing grammar: unexpected escape"));
        assert_eq!(c.reason, FailoverReason::LlamaCppGrammarPattern);

        let c2 = classify(status(
            400,
            "Unable to generate parser for the provided template",
        ));
        assert_eq!(c2.reason, FailoverReason::LlamaCppGrammarPattern);
    }

    #[test]
    fn image_too_large_checked_before_context_overflow_on_400() {
        // Message trips both "exceeds" (context) and image patterns; image wins.
        let c = classify(status(400, "messages.0.content.1.image: image exceeds 5 MB maximum"));
        assert_eq!(c.reason, FailoverReason::ImageTooLarge);
        assert!(c.retryable);
    }

    #[test]
    fn context_overflow_400() {
        let c = classify(status(400, "This model's maximum context length is 8192 tokens"));
        assert_eq!(c.reason, FailoverReason::ContextOverflow);
        assert!(c.should_compress);
    }

    #[test]
    fn generic_400_large_session_is_context_overflow() {
        let ctx = ClassifyContext {
            approx_tokens: 90_000,
            context_length: 200_000,
            ..Default::default()
        };
        let c = classify_ctx(status(400, "Error"), ctx);
        assert_eq!(c.reason, FailoverReason::ContextOverflow);
        assert!(c.should_compress);
    }

    #[test]
    fn generic_400_small_session_is_format_error() {
        let c = classify(status(400, "Error"));
        assert_eq!(c.reason, FailoverReason::FormatError);
        assert!(!c.retryable);
    }

    #[test]
    fn rate_limit_and_billing_on_400_body() {
        let c = classify(status(400, "You have hit the rate limit, slow down"));
        assert_eq!(c.reason, FailoverReason::RateLimit);

        let b = classify(status(400, "Your credit balance is too low"));
        assert_eq!(b.reason, FailoverReason::Billing);
    }

    #[test]
    fn error_code_from_body_classification() {
        let err = ApiError {
            status_code: None,
            error_type: String::new(),
            body: Some(json!({"error": {"code": "insufficient_quota"}})),
            message: "spend more money".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.reason, FailoverReason::Billing);
        assert!(!c.retryable);
    }

    #[test]
    fn error_code_uses_type_when_no_code() {
        let err = ApiError {
            status_code: None,
            error_type: String::new(),
            body: Some(json!({"error": {"type": "context_length_exceeded"}})),
            message: "".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.reason, FailoverReason::ContextOverflow);
        assert!(c.should_compress);
    }

    #[test]
    fn ratelimiterror_type_forces_429() {
        let err = ApiError {
            status_code: None,
            error_type: "RateLimitError".to_string(),
            body: None,
            message: "no status here".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.reason, FailoverReason::RateLimit);
        assert_eq!(c.status_code, Some(429));
    }

    #[test]
    fn message_only_billing_vs_rate_limit() {
        let c = classify(ApiError::from_message("Insufficient credits remaining"));
        assert_eq!(c.reason, FailoverReason::Billing);

        let r = classify(ApiError::from_message("Rate limit exceeded, try again"));
        assert_eq!(r.reason, FailoverReason::RateLimit);
    }

    #[test]
    fn message_only_auth_nonretryable() {
        let c = classify(ApiError::from_message("Invalid API key provided"));
        assert_eq!(c.reason, FailoverReason::Auth);
        assert!(!c.retryable);
        assert!(c.should_rotate_credential);
        assert!(c.should_fallback);
    }

    #[test]
    fn message_only_payload_too_large() {
        let c = classify(ApiError::from_message("Request Entity Too Large (error code: 413)"));
        assert_eq!(c.reason, FailoverReason::PayloadTooLarge);
        assert!(c.should_compress);
    }

    #[test]
    fn ssl_transient_is_timeout_not_compression() {
        let ctx = ClassifyContext {
            approx_tokens: 500_000,
            context_length: 200_000,
            ..Default::default()
        };
        // Even with a huge session, an SSL alert is a timeout (no compression).
        let c = classify_ctx(
            ApiError::from_message("SSLError: [SSL: BAD_RECORD_MAC] bad record mac"),
            ctx,
        );
        assert_eq!(c.reason, FailoverReason::Timeout);
        assert!(!c.should_compress);
    }

    #[test]
    fn server_disconnect_large_session_is_context_overflow() {
        let ctx = ClassifyContext {
            approx_tokens: 130_000,
            context_length: 200_000,
            ..Default::default()
        };
        let c = classify_ctx(
            ApiError::from_message("Server disconnected without sending a response"),
            ctx,
        );
        assert_eq!(c.reason, FailoverReason::ContextOverflow);
        assert!(c.should_compress);
    }

    #[test]
    fn server_disconnect_small_session_is_timeout() {
        let c = classify(ApiError::from_message("peer closed connection unexpectedly"));
        assert_eq!(c.reason, FailoverReason::Timeout);
        assert!(!c.should_compress);
    }

    #[test]
    fn server_disconnect_with_status_does_not_take_disconnect_path() {
        // A disconnect pattern but with a status code falls through to status
        // handling instead. 500 → server_error.
        let c = classify(status(500, "connection reset by peer"));
        assert_eq!(c.reason, FailoverReason::ServerError);
    }

    #[test]
    fn transport_error_type_is_timeout() {
        let err = ApiError {
            status_code: None,
            error_type: "APIConnectionError".to_string(),
            body: None,
            message: "connection failed".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.reason, FailoverReason::Timeout);
        assert!(c.retryable);
    }

    #[test]
    fn unknown_fallback() {
        let c = classify(ApiError::from_message("something totally unrecognizable zzz"));
        assert_eq!(c.reason, FailoverReason::Unknown);
        assert!(c.retryable);
    }

    #[test]
    fn metadata_raw_inner_message_is_parsed() {
        // OpenRouter wraps the real error inside error.metadata.raw.
        let body = json!({
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": "{\"error\": {\"message\": \"This model's maximum context length is exceeded\"}}"
                }
            }
        });
        let err = ApiError {
            status_code: Some(400),
            error_type: String::new(),
            body: Some(body),
            message: "Provider returned error".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.reason, FailoverReason::ContextOverflow);
        assert!(c.should_compress);
    }

    #[test]
    fn extract_message_prefers_body_error_message() {
        let body = json!({"error": {"message": "  Detailed problem here  "}});
        let err = ApiError {
            status_code: Some(400),
            error_type: String::new(),
            body: Some(body),
            message: "short".to_string(),
        };
        let c = classify(err);
        assert_eq!(c.message, "Detailed problem here");
    }

    #[test]
    fn extract_message_truncates_to_500_chars() {
        let long = "x".repeat(800);
        let c = classify(ApiError::from_message(&long));
        assert_eq!(c.message.chars().count(), 500);
    }

    #[test]
    fn provider_and_model_preserved() {
        let ctx = ClassifyContext {
            provider: "OpenRouter".to_string(),
            model: "anthropic/claude".to_string(),
            ..Default::default()
        };
        let c = classify_ctx(status(401, "nope"), ctx);
        assert_eq!(c.provider, "OpenRouter");
        assert_eq!(c.model, "anthropic/claude");
    }

    #[test]
    fn usage_limit_message_without_status_disambiguates() {
        let transient = classify(ApiError::from_message(
            "Usage limit reached. Resets at midnight.",
        ));
        assert_eq!(transient.reason, FailoverReason::RateLimit);

        let permanent = classify(ApiError::from_message("Monthly usage limit exhausted"));
        assert_eq!(permanent.reason, FailoverReason::Billing);
    }
}
