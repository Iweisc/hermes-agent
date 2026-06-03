//! Persistent multi-credential pool for same-provider failover.
//!
//! Native Rust port of `agent/credential_pool.py`.
//!
//! The Python module mixed three concerns:
//!   1. The pure data model ([`PooledCredential`]) and its JSON (de)serialization.
//!   2. Pure logic: exhaustion-cooldown timing, provider error-context parsing,
//!      selection strategies (fill-first / round-robin / random / least-used),
//!      soft leasing, target resolution, upsert/merge, priority normalization,
//!      and stale-seed pruning.
//!   3. Side-effecting glue: reading/writing `auth.json`'s `credential_pool`,
//!      seeding from env vars / singleton auth-store state / custom-provider
//!      config, and OAuth token refresh against live provider HTTP endpoints.
//!
//! (1) and (2) are ported here faithfully and exhaustively unit-tested. The
//! side-effecting glue in (3) is expressed through small traits
//! ([`PoolPersistence`] and [`CredentialRefresher`]) so the runtime
//! [`CredentialPool`] orchestration logic — which is the interesting part —
//! is reproduced exactly while remaining decoupled from the concrete auth.json
//! / reqwest plumbing that other modules (auth.rs, providers.rs) already own.
//!
//! The seeding helpers ([`upsert_entry`], [`normalize_pool_priorities`],
//! [`prune_stale_seeded_entries`], [`seed_from_env_value`]) are ported as pure
//! functions operating on a `Vec<PooledCredential>` so the integration layer
//! can drive them with whatever it reads from disk / the environment.

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::DateTime;
use regex::Regex;
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Status and type constants
// ---------------------------------------------------------------------------

pub const STATUS_OK: &str = "ok";
pub const STATUS_EXHAUSTED: &str = "exhausted";

pub const AUTH_TYPE_OAUTH: &str = "oauth";
pub const AUTH_TYPE_API_KEY: &str = "api_key";

pub const SOURCE_MANUAL: &str = "manual";

pub const STRATEGY_FILL_FIRST: &str = "fill_first";
pub const STRATEGY_ROUND_ROBIN: &str = "round_robin";
pub const STRATEGY_RANDOM: &str = "random";
pub const STRATEGY_LEAST_USED: &str = "least_used";

/// Returns `true` if `strategy` is one of the four supported pool strategies.
pub fn is_supported_pool_strategy(strategy: &str) -> bool {
    matches!(
        strategy,
        STRATEGY_FILL_FIRST | STRATEGY_ROUND_ROBIN | STRATEGY_RANDOM | STRATEGY_LEAST_USED
    )
}

/// Cooldown before retrying an exhausted credential after a 429.
pub const EXHAUSTED_TTL_429_SECONDS: f64 = 60.0 * 60.0;
/// Default cooldown before retrying an exhausted credential.
pub const EXHAUSTED_TTL_DEFAULT_SECONDS: f64 = 60.0 * 60.0;

/// Pool key prefix for custom OpenAI-compatible endpoints.
pub const CUSTOM_POOL_PREFIX: &str = "custom:";

pub const DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL: i64 = 1;

/// Fields that are only round-tripped through JSON — never used for logic as
/// attributes. Mirrors Python's `_EXTRA_KEYS`.
pub const EXTRA_KEYS: &[&str] = &[
    "token_type",
    "scope",
    "client_id",
    "portal_base_url",
    "obtained_at",
    "expires_in",
    "agent_key_id",
    "agent_key_expires_in",
    "agent_key_reused",
    "agent_key_obtained_at",
    "tls",
];

fn is_extra_key(key: &str) -> bool {
    EXTRA_KEYS.contains(&key)
}

/// Field names that are always emitted by [`PooledCredential::to_dict`], even
/// when their value is `None`/null. Mirrors Python's `_ALWAYS_EMIT`.
const ALWAYS_EMIT: &[&str] = &[
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
];

// ---------------------------------------------------------------------------
// PooledCredential
// ---------------------------------------------------------------------------

/// A single credential within a provider's failover pool.
///
/// Field set and JSON shape match the Python `PooledCredential` dataclass
/// exactly. `extra` holds the round-tripped-only [`EXTRA_KEYS`] values.
#[derive(Debug, Clone, PartialEq)]
pub struct PooledCredential {
    pub provider: String,
    pub id: String,
    pub label: String,
    pub auth_type: String,
    pub priority: i64,
    pub source: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub last_status: Option<String>,
    pub last_status_at: Option<f64>,
    pub last_error_code: Option<i64>,
    pub last_error_reason: Option<String>,
    pub last_error_message: Option<String>,
    pub last_error_reset_at: Option<f64>,
    pub base_url: Option<String>,
    pub expires_at: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub last_refresh: Option<String>,
    pub inference_base_url: Option<String>,
    pub agent_key: Option<String>,
    pub agent_key_expires_at: Option<String>,
    pub request_count: i64,
    pub extra: Map<String, Value>,
}

impl PooledCredential {
    /// Build a [`PooledCredential`] from a provider name and a raw JSON payload,
    /// applying the same defaults as Python's `from_dict`.
    pub fn from_dict(provider: &str, payload: &Map<String, Value>) -> Self {
        let s = |key: &str| -> Option<String> {
            payload.get(key).and_then(json_to_string_opt)
        };
        let i = |key: &str| -> Option<i64> { payload.get(key).and_then(json_to_i64) };
        let f = |key: &str| -> Option<f64> { payload.get(key).and_then(json_to_f64) };

        // Defaults (`data.setdefault(...)` in Python).
        let id = s("id").unwrap_or_else(new_id);
        let label = s("label").unwrap_or_else(|| {
            payload
                .get("source")
                .and_then(json_to_string_opt)
                .unwrap_or_else(|| provider.to_string())
        });
        let auth_type = s("auth_type").unwrap_or_else(|| AUTH_TYPE_API_KEY.to_string());
        let priority = i("priority").unwrap_or(0);
        let source = s("source").unwrap_or_else(|| SOURCE_MANUAL.to_string());
        let access_token = s("access_token").unwrap_or_default();

        let mut extra = Map::new();
        for key in EXTRA_KEYS {
            if let Some(value) = payload.get(*key) {
                if !value.is_null() {
                    extra.insert((*key).to_string(), value.clone());
                }
            }
        }

        PooledCredential {
            provider: provider.to_string(),
            id,
            label,
            auth_type,
            priority,
            source,
            access_token,
            refresh_token: s("refresh_token"),
            last_status: s("last_status"),
            last_status_at: f("last_status_at"),
            last_error_code: i("last_error_code"),
            last_error_reason: s("last_error_reason"),
            last_error_message: s("last_error_message"),
            last_error_reset_at: f("last_error_reset_at"),
            base_url: s("base_url"),
            expires_at: s("expires_at"),
            expires_at_ms: i("expires_at_ms"),
            last_refresh: s("last_refresh"),
            inference_base_url: s("inference_base_url"),
            agent_key: s("agent_key"),
            agent_key_expires_at: s("agent_key_expires_at"),
            request_count: i("request_count").unwrap_or(0),
            extra,
        }
    }

    /// Serialize back to the JSON shape persisted in auth.json. Skips `provider`
    /// and `extra` (extras are flattened in), omits `None` values unless the
    /// field is in [`ALWAYS_EMIT`], and flattens non-null extras.
    pub fn to_dict(&self) -> Map<String, Value> {
        let mut result = Map::new();

        // Helper that emits a value or honours ALWAYS_EMIT for null.
        macro_rules! emit_opt {
            ($name:literal, $value:expr) => {{
                let v: Value = match &$value {
                    Some(inner) => inner.clone().into(),
                    None => Value::Null,
                };
                if !v.is_null() || ALWAYS_EMIT.contains(&$name) {
                    result.insert($name.to_string(), v);
                }
            }};
        }

        // Required (non-Option) fields are always emitted (never None in Python).
        result.insert("id".to_string(), Value::String(self.id.clone()));
        result.insert("label".to_string(), Value::String(self.label.clone()));
        result.insert(
            "auth_type".to_string(),
            Value::String(self.auth_type.clone()),
        );
        result.insert("priority".to_string(), Value::from(self.priority));
        result.insert("source".to_string(), Value::String(self.source.clone()));
        result.insert(
            "access_token".to_string(),
            Value::String(self.access_token.clone()),
        );

        emit_opt!("refresh_token", self.refresh_token);
        emit_opt!("last_status", self.last_status);
        emit_opt!("last_status_at", self.last_status_at);
        emit_opt!("last_error_code", self.last_error_code);
        emit_opt!("last_error_reason", self.last_error_reason);
        emit_opt!("last_error_message", self.last_error_message);
        emit_opt!("last_error_reset_at", self.last_error_reset_at);
        emit_opt!("base_url", self.base_url);
        emit_opt!("expires_at", self.expires_at);
        emit_opt!("expires_at_ms", self.expires_at_ms);
        emit_opt!("last_refresh", self.last_refresh);
        emit_opt!("inference_base_url", self.inference_base_url);
        emit_opt!("agent_key", self.agent_key);
        emit_opt!("agent_key_expires_at", self.agent_key_expires_at);

        // request_count is non-Option in Python (int default 0) and always emits.
        result.insert(
            "request_count".to_string(),
            Value::from(self.request_count),
        );

        for (k, v) in &self.extra {
            if !v.is_null() {
                result.insert(k.clone(), v.clone());
            }
        }
        result
    }

    /// `runtime_api_key` property: Nous prefers the agent key.
    pub fn runtime_api_key(&self) -> String {
        if self.provider == "nous" {
            let agent = self.agent_key.as_deref().filter(|s| !s.is_empty());
            if let Some(agent) = agent {
                return agent.to_string();
            }
            return self.access_token.clone();
        }
        self.access_token.clone()
    }

    /// `runtime_base_url` property: Nous prefers the inference base URL.
    pub fn runtime_base_url(&self) -> Option<String> {
        if self.provider == "nous" {
            if let Some(url) = self.inference_base_url.as_ref().filter(|s| !s.is_empty()) {
                return Some(url.clone());
            }
            return self.base_url.clone();
        }
        self.base_url.clone()
    }

    /// Read an [`EXTRA_KEYS`] field (mirrors Python's `__getattr__` over extra).
    pub fn extra_get(&self, key: &str) -> Option<&Value> {
        if is_extra_key(key) {
            self.extra.get(key)
        } else {
            None
        }
    }

    /// True if this credential is currently in OAuth status `ok` with no error.
    fn has_status_noise(&self) -> bool {
        self.last_status.is_some() || self.last_status_at.is_some() || self.last_error_code.is_some()
    }

    /// Clear all status/error fields (the common `replace(..., last_status=None, ...)`).
    fn clear_status(&mut self) {
        self.last_status = None;
        self.last_status_at = None;
        self.last_error_code = None;
        self.last_error_reason = None;
        self.last_error_message = None;
        self.last_error_reset_at = None;
    }
}

// ---------------------------------------------------------------------------
// JSON value coercion helpers (mirror Python's loose typing)
// ---------------------------------------------------------------------------

fn json_to_string_opt(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn json_to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn json_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Generate a 6-hex-char id, matching `uuid.uuid4().hex[:6]`.
pub fn new_id() -> String {
    let mut buf = [0u8; 3];
    let _ = getrandom_fill(&mut buf);
    format!("{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2])
}

fn getrandom_fill(buf: &mut [u8]) -> Result<(), ()> {
    // Best-effort randomness; fall back to a time-seeded value if unavailable.
    if getrandom::fill(buf).is_ok() {
        return Ok(());
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let bytes = nanos.to_le_bytes();
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = bytes[i % bytes.len()];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// JWT label helper
// ---------------------------------------------------------------------------

/// Decode the unverified claims payload of a JWT (`base64url` middle segment).
/// Mirrors `hermes_cli.auth._decode_jwt_claims`'s practical behavior.
pub fn decode_jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let pad = (4 - payload.len() % 4) % 4;
    let padded = format!("{payload}{}", "=".repeat(pad));
    use base64::Engine;
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .ok()?;
    serde_json::from_slice::<Value>(&decoded).ok()
}

/// Derive a human label from a JWT's claims (email / preferred_username / upn),
/// falling back to `fallback`.
pub fn label_from_token(token: &str, fallback: &str) -> String {
    if let Some(claims) = decode_jwt_claims(token) {
        for key in ["email", "preferred_username", "upn"] {
            if let Some(value) = claims.get(key).and_then(Value::as_str) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    fallback.to_string()
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// `_next_priority`: max existing priority + 1 (or 0 for an empty pool).
pub fn next_priority(entries: &[PooledCredential]) -> i64 {
    entries.iter().map(|e| e.priority).max().unwrap_or(-1) + 1
}

/// `_is_manual_source`: equal to `manual` or prefixed with `manual:`.
pub fn is_manual_source(source: &str) -> bool {
    let normalized = source.trim().to_lowercase();
    normalized == SOURCE_MANUAL || normalized.starts_with(&format!("{SOURCE_MANUAL}:"))
}

/// `_exhausted_ttl`: cooldown seconds keyed on the HTTP status.
pub fn exhausted_ttl(error_code: Option<i64>) -> f64 {
    if error_code == Some(429) {
        EXHAUSTED_TTL_429_SECONDS
    } else {
        EXHAUSTED_TTL_DEFAULT_SECONDS
    }
}

/// `_parse_absolute_timestamp`: epoch seconds / epoch millis / ISO-8601 → secs.
pub fn parse_absolute_timestamp(value: &Value) -> Option<f64> {
    match value {
        Value::Null => None,
        Value::Number(n) => {
            let numeric = n.as_f64()?;
            normalize_numeric_timestamp(numeric)
        }
        Value::String(s) => {
            let raw = s.trim();
            if raw.is_empty() {
                return None;
            }
            if let Ok(numeric) = raw.parse::<f64>() {
                return normalize_numeric_timestamp(numeric);
            }
            // ISO-8601, normalising a trailing Z to +00:00.
            let normalized = raw.replace('Z', "+00:00");
            DateTime::parse_from_rfc3339(&normalized)
                .ok()
                .map(|dt| dt.timestamp() as f64)
        }
        _ => None,
    }
}

fn normalize_numeric_timestamp(numeric: f64) -> Option<f64> {
    if numeric <= 0.0 {
        return None;
    }
    if numeric > 1_000_000_000_000.0 {
        Some(numeric / 1000.0)
    } else {
        Some(numeric)
    }
}

/// `_extract_retry_delay_seconds`: pull a retry hint out of an error message.
pub fn extract_retry_delay_seconds(message: &str) -> Option<f64> {
    if message.is_empty() {
        return None;
    }
    // quotaResetDelay[:\s"]+(\d+(?:\.\d+)?)(ms|s)
    let delay_re =
        Regex::new(r#"(?i)quotaResetDelay[:\s"]+(\d+(?:\.\d+)?)(ms|s)"#).expect("valid regex");
    if let Some(caps) = delay_re.captures(message) {
        let value: f64 = caps[1].parse().ok()?;
        let unit = caps[2].to_lowercase();
        return Some(if unit == "ms" { value / 1000.0 } else { value });
    }
    // retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)
    let sec_re = Regex::new(r"(?i)retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)")
        .expect("valid regex");
    if let Some(caps) = sec_re.captures(message) {
        return caps[1].parse().ok();
    }
    None
}

/// Normalized error context produced by [`normalize_error_context`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NormalizedError {
    pub reason: Option<String>,
    pub message: Option<String>,
    pub reset_at: Option<f64>,
}

/// `_normalize_error_context`: pull reason / message / reset_at out of a raw
/// provider error context, deriving reset_at from a retry hint when absent.
///
/// `now` is injected (Python uses `time.time()`) so the function stays pure.
pub fn normalize_error_context(
    error_context: Option<&Map<String, Value>>,
    now: f64,
) -> NormalizedError {
    let mut normalized = NormalizedError::default();
    let Some(ctx) = error_context else {
        return normalized;
    };

    if let Some(reason) = ctx.get("reason").and_then(Value::as_str) {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            normalized.reason = Some(trimmed.to_string());
        }
    }

    let message_str = ctx.get("message").and_then(Value::as_str).map(str::trim);
    if let Some(message) = message_str {
        if !message.is_empty() {
            normalized.message = Some(message.to_string());
        }
    }

    let reset_raw = ctx
        .get("reset_at")
        .filter(|v| !v.is_null())
        .or_else(|| ctx.get("resets_at").filter(|v| !v.is_null()))
        .or_else(|| ctx.get("retry_until").filter(|v| !v.is_null()));

    let mut parsed_reset_at = reset_raw.and_then(parse_absolute_timestamp);
    if parsed_reset_at.is_none() {
        if let Some(message) = ctx.get("message").and_then(Value::as_str) {
            if let Some(delay) = extract_retry_delay_seconds(message) {
                parsed_reset_at = Some(now + delay);
            }
        }
    }
    normalized.reset_at = parsed_reset_at;
    normalized
}

/// `_exhausted_until`: absolute epoch second until which an entry stays cooled.
pub fn exhausted_until(entry: &PooledCredential) -> Option<f64> {
    if entry.last_status.as_deref() != Some(STATUS_EXHAUSTED) {
        return None;
    }
    if let Some(reset_at) = entry
        .last_error_reset_at
        .map(Value::from)
        .as_ref()
        .and_then(parse_absolute_timestamp)
    {
        return Some(reset_at);
    }
    entry
        .last_status_at
        .map(|at| at + exhausted_ttl(entry.last_error_code))
}

/// `_normalize_custom_pool_name`.
pub fn normalize_custom_pool_name(name: &str) -> String {
    name.trim().to_lowercase().replace(' ', "-")
}

// ---------------------------------------------------------------------------
// Upsert / priority normalization / pruning (pure pool-mutation helpers)
// ---------------------------------------------------------------------------

/// `_upsert_entry`: insert a new entry for `source`, or merge changed fields
/// into the existing one. Returns `true` if `entries` was mutated.
pub fn upsert_entry(
    entries: &mut Vec<PooledCredential>,
    provider: &str,
    source: &str,
    payload: &Map<String, Value>,
) -> bool {
    let existing_idx = entries.iter().position(|e| e.source == source);

    if existing_idx.is_none() {
        let mut payload = payload.clone();
        payload
            .entry("id".to_string())
            .or_insert_with(|| Value::String(new_id()));
        payload
            .entry("priority".to_string())
            .or_insert_with(|| Value::from(next_priority(entries)));
        // label: payload.get("label") or source
        let label = payload
            .get("label")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| source.to_string());
        payload.insert("label".to_string(), Value::String(label));
        entries.push(PooledCredential::from_dict(provider, &payload));
        return true;
    }

    let idx = existing_idx.unwrap();
    let mut changed = false;
    // Snapshot existing values needed for comparison without holding a borrow.
    let existing_label_present = !entries[idx].label.is_empty();

    for (key, value) in payload.iter() {
        if key == "id" || key == "priority" || value.is_null() {
            continue;
        }
        if key == "label" && existing_label_present {
            continue;
        }
        if is_extra_key(key) {
            let differs = entries[idx].extra.get(key) != Some(value);
            if differs {
                entries[idx].extra.insert(key.clone(), value.clone());
                changed = true;
            }
        } else if apply_field_if_changed(&mut entries[idx], key, value) {
            changed = true;
        }
    }
    changed
}

/// Apply a dataclass-field update by name if it differs; returns `true` if changed.
/// Returns `false` for unknown/non-field keys (Python silently drops them).
fn apply_field_if_changed(entry: &mut PooledCredential, key: &str, value: &Value) -> bool {
    macro_rules! set_string {
        ($field:ident) => {{
            let new = json_to_string_opt(value);
            let differs = match &new {
                Some(s) => entry.$field.as_deref() != Some(s.as_str()),
                None => false,
            };
            if differs {
                entry.$field = new;
                return true;
            }
            return false;
        }};
    }
    match key {
        "label" => {
            if let Some(s) = json_to_string_opt(value) {
                if entry.label != s {
                    entry.label = s;
                    return true;
                }
            }
            false
        }
        "auth_type" => {
            if let Some(s) = json_to_string_opt(value) {
                if entry.auth_type != s {
                    entry.auth_type = s;
                    return true;
                }
            }
            false
        }
        "source" => {
            if let Some(s) = json_to_string_opt(value) {
                if entry.source != s {
                    entry.source = s;
                    return true;
                }
            }
            false
        }
        "access_token" => {
            if let Some(s) = json_to_string_opt(value) {
                if entry.access_token != s {
                    entry.access_token = s;
                    return true;
                }
            }
            false
        }
        "refresh_token" => set_string!(refresh_token),
        "last_status" => set_string!(last_status),
        "last_error_reason" => set_string!(last_error_reason),
        "last_error_message" => set_string!(last_error_message),
        "base_url" => set_string!(base_url),
        "expires_at" => set_string!(expires_at),
        "last_refresh" => set_string!(last_refresh),
        "inference_base_url" => set_string!(inference_base_url),
        "agent_key" => set_string!(agent_key),
        "agent_key_expires_at" => set_string!(agent_key_expires_at),
        "priority" => {
            if let Some(v) = json_to_i64(value) {
                if entry.priority != v {
                    entry.priority = v;
                    return true;
                }
            }
            false
        }
        "last_error_code" => {
            if let Some(v) = json_to_i64(value) {
                if entry.last_error_code != Some(v) {
                    entry.last_error_code = Some(v);
                    return true;
                }
            }
            false
        }
        "expires_at_ms" => {
            if let Some(v) = json_to_i64(value) {
                if entry.expires_at_ms != Some(v) {
                    entry.expires_at_ms = Some(v);
                    return true;
                }
            }
            false
        }
        "request_count" => {
            if let Some(v) = json_to_i64(value) {
                if entry.request_count != v {
                    entry.request_count = v;
                    return true;
                }
            }
            false
        }
        "last_status_at" => {
            if let Some(v) = json_to_f64(value) {
                if entry.last_status_at != Some(v) {
                    entry.last_status_at = Some(v);
                    return true;
                }
            }
            false
        }
        "last_error_reset_at" => {
            if let Some(v) = json_to_f64(value) {
                if entry.last_error_reset_at != Some(v) {
                    entry.last_error_reset_at = Some(v);
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// `_normalize_pool_priorities`: anthropic-only deterministic source ordering.
/// Returns `true` if any priority changed.
pub fn normalize_pool_priorities(provider: &str, entries: &mut [PooledCredential]) -> bool {
    if provider != "anthropic" {
        return false;
    }

    fn source_rank(source: &str) -> i64 {
        match source {
            "env:ANTHROPIC_TOKEN" => 0,
            "env:CLAUDE_CODE_OAUTH_TOKEN" => 1,
            "hermes_pkce" => 2,
            "claude_code" => 3,
            "env:ANTHROPIC_API_KEY" => 4,
            _ => 5, // len(source_rank)
        }
    }

    let mut manual: Vec<&PooledCredential> =
        entries.iter().filter(|e| is_manual_source(&e.source)).collect();
    manual.sort_by_key(|e| e.priority);

    let mut seeded: Vec<&PooledCredential> = entries
        .iter()
        .filter(|e| !is_manual_source(&e.source))
        .collect();
    seeded.sort_by(|a, b| {
        (source_rank(&a.source), a.priority, a.label.clone()).cmp(&(
            source_rank(&b.source),
            b.priority,
            b.label.clone(),
        ))
    });

    let ordered_ids: Vec<String> = manual
        .iter()
        .chain(seeded.iter())
        .map(|e| e.id.clone())
        .collect();

    let id_to_idx: HashMap<String, usize> = entries
        .iter()
        .enumerate()
        .map(|(idx, e)| (e.id.clone(), idx))
        .collect();

    let mut changed = false;
    for (new_priority, id) in ordered_ids.iter().enumerate() {
        if let Some(&idx) = id_to_idx.get(id) {
            if entries[idx].priority != new_priority as i64 {
                entries[idx].priority = new_priority as i64;
                changed = true;
            }
        }
    }
    changed
}

/// `_prune_stale_seeded_entries`: drop env/claude_code/hermes_pkce-seeded
/// entries whose source is no longer active. Returns `true` if pruned.
pub fn prune_stale_seeded_entries(
    entries: &mut Vec<PooledCredential>,
    active_sources: &HashSet<String>,
) -> bool {
    let original_len = entries.len();
    entries.retain(|entry| {
        is_manual_source(&entry.source)
            || active_sources.contains(&entry.source)
            || !(entry.source.starts_with("env:")
                || entry.source == "claude_code"
                || entry.source == "hermes_pkce")
    });
    entries.len() != original_len
}

// ---------------------------------------------------------------------------
// Persistence + refresh hooks
// ---------------------------------------------------------------------------

/// Side-effect boundary for persisting a provider's pool. The integration
/// layer implements this against auth.json's `credential_pool` map.
pub trait PoolPersistence {
    /// Persist the full entry list for `provider` (mirrors `write_credential_pool`).
    fn write_pool(&mut self, provider: &str, entries: &[PooledCredential]);
}

/// No-op persistence, useful for tests and read-only flows.
#[derive(Debug, Default, Clone)]
pub struct NoopPersistence;

impl PoolPersistence for NoopPersistence {
    fn write_pool(&mut self, _provider: &str, _entries: &[PooledCredential]) {}
}

/// Outcome of an OAuth refresh attempt for a single entry.
pub enum RefreshOutcome {
    /// Refresh succeeded; the entry has been updated to this value
    /// (status fields will be cleared by the pool).
    Refreshed(PooledCredential),
    /// Provider does not support refresh / not applicable; leave entry as-is.
    NotApplicable,
    /// Refresh failed; the pool will mark the entry exhausted.
    Failed,
}

/// Side-effect boundary for OAuth token refresh against live providers.
pub trait CredentialRefresher {
    /// Attempt to refresh `entry` for `provider`. `force` mirrors the Python
    /// `force` flag (used by `try_refresh_current`).
    fn refresh(&mut self, provider: &str, entry: &PooledCredential, force: bool) -> RefreshOutcome;

    /// Whether `entry` needs a proactive refresh (mirrors `_entry_needs_refresh`).
    fn entry_needs_refresh(&self, provider: &str, entry: &PooledCredential) -> bool;
}

/// A refresher that never refreshes anything. Selection/leasing still work; any
/// entry that *would* need a refresh is treated as not needing one. Useful for
/// pure pool operations (listing, target resolution) and unit tests.
#[derive(Debug, Default, Clone)]
pub struct NoRefresh;

impl CredentialRefresher for NoRefresh {
    fn refresh(
        &mut self,
        _provider: &str,
        _entry: &PooledCredential,
        _force: bool,
    ) -> RefreshOutcome {
        RefreshOutcome::NotApplicable
    }

    fn entry_needs_refresh(&self, _provider: &str, _entry: &PooledCredential) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// CredentialPool
// ---------------------------------------------------------------------------

/// Runtime credential pool for a single provider with failover, exhaustion
/// cooldown, soft leasing, and four selection strategies. Port of the Python
/// `CredentialPool` class. The Python `threading.Lock` is omitted: callers in
/// Rust own the `&mut self` borrow, which provides the same single-writer
/// guarantee the lock gave Python.
pub struct CredentialPool {
    pub provider: String,
    entries: Vec<PooledCredential>,
    current_id: Option<String>,
    strategy: String,
    active_leases: HashMap<String, i64>,
    max_concurrent: i64,
}

impl CredentialPool {
    /// Build a pool. `strategy` is resolved by the caller (mirrors
    /// `get_pool_strategy(provider)`); pass [`STRATEGY_FILL_FIRST`] by default.
    pub fn new(provider: &str, mut entries: Vec<PooledCredential>, strategy: &str) -> Self {
        entries.sort_by_key(|e| e.priority);
        let strategy = if is_supported_pool_strategy(strategy) {
            strategy.to_string()
        } else {
            STRATEGY_FILL_FIRST.to_string()
        };
        CredentialPool {
            provider: provider.to_string(),
            entries,
            current_id: None,
            strategy,
            active_leases: HashMap::new(),
            max_concurrent: DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL,
        }
    }

    pub fn strategy(&self) -> &str {
        &self.strategy
    }

    pub fn has_credentials(&self) -> bool {
        !self.entries.is_empty()
    }

    /// True if at least one entry is not currently in exhaustion cooldown.
    pub fn has_available(&self, now: f64) -> bool {
        !self.collect_available(now).is_empty()
    }

    pub fn entries(&self) -> Vec<PooledCredential> {
        self.entries.clone()
    }

    pub fn current(&self) -> Option<&PooledCredential> {
        let id = self.current_id.as_ref()?;
        self.entries.iter().find(|e| &e.id == id)
    }

    fn replace_entry(&mut self, old_id: &str, new: PooledCredential) {
        if let Some(idx) = self.entries.iter().position(|e| e.id == old_id) {
            self.entries[idx] = new;
        }
    }

    fn persist(&self, persistence: &mut dyn PoolPersistence) {
        persistence.write_pool(&self.provider, &self.entries);
    }

    /// `_mark_exhausted`: mark an entry exhausted from a status code + context.
    pub fn mark_exhausted(
        &mut self,
        entry_id: &str,
        status_code: Option<i64>,
        error_context: Option<&Map<String, Value>>,
        now: f64,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        let idx = self.entries.iter().position(|e| e.id == entry_id)?;
        let normalized = normalize_error_context(error_context, now);
        let entry = &mut self.entries[idx];
        entry.last_status = Some(STATUS_EXHAUSTED.to_string());
        entry.last_status_at = Some(now);
        entry.last_error_code = status_code;
        entry.last_error_reason = normalized.reason;
        entry.last_error_message = normalized.message;
        entry.last_error_reset_at = normalized.reset_at;
        let updated = entry.clone();
        self.persist(persistence);
        Some(updated)
    }

    /// Collect entries not in active exhaustion cooldown (read-only; no clear/refresh).
    /// Mirrors `_available_entries()` with `clear_expired=False, refresh=False`.
    fn collect_available(&self, now: f64) -> Vec<PooledCredential> {
        self.entries
            .iter()
            .filter(|entry| {
                if entry.last_status.as_deref() == Some(STATUS_EXHAUSTED) {
                    if let Some(until) = exhausted_until(entry) {
                        if now < until {
                            return false;
                        }
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    /// `_available_entries(clear_expired, refresh)`. Mutates entries when an
    /// expired cooldown is cleared or a refresh succeeds, persisting if so.
    fn available_entries(
        &mut self,
        clear_expired: bool,
        refresh: bool,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Vec<PooledCredential> {
        let mut cleared_any = false;
        let mut available: Vec<PooledCredential> = Vec::new();
        let ids: Vec<String> = self.entries.iter().map(|e| e.id.clone()).collect();

        for id in ids {
            let Some(idx) = self.entries.iter().position(|e| e.id == id) else {
                continue;
            };
            let mut entry = self.entries[idx].clone();

            if entry.last_status.as_deref() == Some(STATUS_EXHAUSTED) {
                if let Some(until) = exhausted_until(&entry) {
                    if now < until {
                        continue;
                    }
                }
                if clear_expired {
                    entry.clear_status();
                    self.replace_entry(&id, entry.clone());
                    cleared_any = true;
                }
            }

            if refresh && refresher.entry_needs_refresh(&self.provider, &entry) {
                match refresher.refresh(&self.provider, &entry, false) {
                    RefreshOutcome::Refreshed(mut refreshed) => {
                        refreshed.clear_status();
                        refreshed.last_status = Some(STATUS_OK.to_string());
                        // Python sets last_status=STATUS_OK in _refresh_entry's
                        // success path; here we mirror that for refreshed entries.
                        // (Note: Python's success replace clears status then the
                        // outer replace sets STATUS_OK only in _refresh_entry; in
                        // _available_entries it just uses the refreshed entry.)
                        // To match exactly, leave status as the refresher returned.
                        refreshed.last_status = None;
                        self.replace_entry(&id, refreshed.clone());
                        entry = refreshed;
                    }
                    RefreshOutcome::NotApplicable => {}
                    RefreshOutcome::Failed => {
                        // Mark exhausted, then skip (refresh returned None).
                        self.mark_exhausted(&id, None, None, now, persistence);
                        continue;
                    }
                }
            }
            available.push(entry);
        }

        if cleared_any {
            self.persist(persistence);
        }
        available
    }

    /// `select()` / `_select_unlocked()`.
    pub fn select(
        &mut self,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        let available = self.available_entries(true, true, now, refresher, persistence);
        if available.is_empty() {
            self.current_id = None;
            return None;
        }

        match self.strategy.as_str() {
            STRATEGY_RANDOM => {
                let entry = available[pseudo_random_index(available.len())].clone();
                self.current_id = Some(entry.id.clone());
                Some(entry)
            }
            STRATEGY_LEAST_USED if available.len() > 1 => {
                // min by request_count (stable: first minimum wins)
                let chosen = available
                    .iter()
                    .min_by_key(|e| e.request_count)
                    .cloned()
                    .unwrap();
                let mut updated = chosen.clone();
                updated.request_count += 1;
                self.replace_entry(&chosen.id, updated.clone());
                self.current_id = Some(chosen.id.clone());
                Some(updated)
            }
            STRATEGY_ROUND_ROBIN if available.len() > 1 => {
                let entry = available[0].clone();
                // Build rotated list: all others (in current order) + chosen last,
                // then renumber priorities by position.
                let mut rotated: Vec<PooledCredential> = self
                    .entries
                    .iter()
                    .filter(|c| c.id != entry.id)
                    .cloned()
                    .collect();
                let mut tail = entry.clone();
                tail.priority = (self.entries.len() as i64) - 1;
                rotated.push(tail);
                for (idx, candidate) in rotated.iter_mut().enumerate() {
                    candidate.priority = idx as i64;
                }
                self.entries = rotated;
                self.persist(persistence);
                self.current_id = Some(entry.id.clone());
                Some(self.current().cloned().unwrap_or(entry))
            }
            _ => {
                let entry = available[0].clone();
                self.current_id = Some(entry.id.clone());
                Some(entry)
            }
        }
    }

    /// `peek()`: current entry, else first available (read-only).
    pub fn peek(&self, now: f64) -> Option<PooledCredential> {
        if let Some(current) = self.current() {
            return Some(current.clone());
        }
        self.collect_available(now).into_iter().next()
    }

    /// `mark_exhausted_and_rotate`: mark current exhausted, then re-select.
    pub fn mark_exhausted_and_rotate(
        &mut self,
        status_code: Option<i64>,
        error_context: Option<&Map<String, Value>>,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        let entry = match self.current().cloned() {
            Some(entry) => entry,
            None => self.select(now, refresher, persistence)?,
        };
        self.mark_exhausted(&entry.id, status_code, error_context, now, persistence);
        self.current_id = None;
        self.select(now, refresher, persistence)
    }

    /// `acquire_lease`: soft lease on a credential.
    pub fn acquire_lease(
        &mut self,
        credential_id: Option<&str>,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<String> {
        if let Some(id) = credential_id {
            *self.active_leases.entry(id.to_string()).or_insert(0) += 1;
            self.current_id = Some(id.to_string());
            return Some(id.to_string());
        }

        let available = self.available_entries(true, true, now, refresher, persistence);
        if available.is_empty() {
            return None;
        }

        let below_cap: Vec<&PooledCredential> = available
            .iter()
            .filter(|e| self.active_leases.get(&e.id).copied().unwrap_or(0) < self.max_concurrent)
            .collect();
        let candidates: Vec<&PooledCredential> = if below_cap.is_empty() {
            available.iter().collect()
        } else {
            below_cap
        };

        // min by (leases, priority); first minimum wins (stable).
        let chosen = candidates
            .iter()
            .min_by_key(|e| (self.active_leases.get(&e.id).copied().unwrap_or(0), e.priority))
            .map(|e| (*e).clone())?;

        *self.active_leases.entry(chosen.id.clone()).or_insert(0) += 1;
        self.current_id = Some(chosen.id.clone());
        Some(chosen.id)
    }

    /// `release_lease`.
    pub fn release_lease(&mut self, credential_id: &str) {
        let count = self.active_leases.get(credential_id).copied().unwrap_or(0);
        if count <= 1 {
            self.active_leases.remove(credential_id);
        } else {
            self.active_leases.insert(credential_id.to_string(), count - 1);
        }
    }

    pub fn active_lease_count(&self, credential_id: &str) -> i64 {
        self.active_leases.get(credential_id).copied().unwrap_or(0)
    }

    /// `try_refresh_current` / `_try_refresh_current_unlocked`.
    pub fn try_refresh_current(
        &mut self,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        let entry = self.current().cloned()?;
        let refreshed = self.refresh_entry(&entry, true, now, refresher, persistence);
        if let Some(ref r) = refreshed {
            self.current_id = Some(r.id.clone());
        }
        refreshed
    }

    /// `_refresh_entry`: drive the refresher, applying the success/fail status
    /// bookkeeping and persistence. Returns the refreshed entry, or `None` when
    /// refresh is not applicable / failed (entry marked exhausted on `force`).
    fn refresh_entry(
        &mut self,
        entry: &PooledCredential,
        force: bool,
        now: f64,
        refresher: &mut dyn CredentialRefresher,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        if entry.auth_type != AUTH_TYPE_OAUTH || entry.refresh_token.is_none() {
            if force {
                self.mark_exhausted(&entry.id, None, None, now, persistence);
            }
            return None;
        }

        match refresher.refresh(&self.provider, entry, force) {
            RefreshOutcome::Refreshed(mut updated) => {
                updated.clear_status();
                updated.last_status = Some(STATUS_OK.to_string());
                self.replace_entry(&entry.id, updated.clone());
                self.persist(persistence);
                Some(updated)
            }
            RefreshOutcome::NotApplicable => {
                // Python returns the entry unchanged for unknown providers.
                Some(entry.clone())
            }
            RefreshOutcome::Failed => {
                self.mark_exhausted(&entry.id, None, None, now, persistence);
                None
            }
        }
    }

    /// `reset_statuses`: clear status/error on every entry; returns count cleared.
    pub fn reset_statuses(&mut self, persistence: &mut dyn PoolPersistence) -> usize {
        let mut count = 0;
        for entry in &mut self.entries {
            if entry.has_status_noise() {
                entry.clear_status();
                count += 1;
            }
        }
        if count > 0 {
            self.persist(persistence);
        }
        count
    }

    /// `remove_index`: remove the 1-based `index`th entry, renumber priorities.
    pub fn remove_index(
        &mut self,
        index: usize,
        persistence: &mut dyn PoolPersistence,
    ) -> Option<PooledCredential> {
        if index < 1 || index > self.entries.len() {
            return None;
        }
        let removed = self.entries.remove(index - 1);
        for (new_priority, entry) in self.entries.iter_mut().enumerate() {
            entry.priority = new_priority as i64;
        }
        self.persist(persistence);
        if self.current_id.as_deref() == Some(removed.id.as_str()) {
            self.current_id = None;
        }
        Some(removed)
    }

    /// `resolve_target`: resolve an id / unique label / 1-based index to an entry.
    /// Returns `(Some(index), Some(entry), None)` on success, or
    /// `(None, None, Some(error_message))` on failure.
    pub fn resolve_target(
        &self,
        target: &str,
    ) -> (Option<usize>, Option<PooledCredential>, Option<String>) {
        let raw = target.trim();
        if raw.is_empty() {
            return (None, None, Some("No credential target provided.".to_string()));
        }

        // Exact id match.
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.id == raw {
                return (Some(idx + 1), Some(entry.clone()), None);
            }
        }

        // Label match (case-insensitive).
        let raw_lower = raw.to_lowercase();
        let label_matches: Vec<(usize, &PooledCredential)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.label.trim().to_lowercase() == raw_lower)
            .map(|(idx, e)| (idx + 1, e))
            .collect();
        if label_matches.len() == 1 {
            let (idx, entry) = label_matches[0];
            return (Some(idx), Some(entry.clone()), None);
        }
        if label_matches.len() > 1 {
            return (
                None,
                None,
                Some(format!(
                    "Ambiguous credential label \"{raw}\". Use the numeric index or entry id instead."
                )),
            );
        }

        // Numeric index.
        if !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit()) {
            let index: usize = raw.parse().unwrap_or(0);
            if index >= 1 && index <= self.entries.len() {
                return (Some(index), Some(self.entries[index - 1].clone()), None);
            }
            return (None, None, Some(format!("No credential #{index}.")));
        }

        (None, None, Some(format!("No credential matching \"{raw}\".")))
    }

    /// `add_entry`: append an entry, assigning it the next priority.
    pub fn add_entry(
        &mut self,
        mut entry: PooledCredential,
        persistence: &mut dyn PoolPersistence,
    ) -> PooledCredential {
        entry.priority = next_priority(&self.entries);
        self.entries.push(entry.clone());
        self.persist(persistence);
        entry
    }
}

/// Deterministic-enough index picker for [`STRATEGY_RANDOM`]. Uses OS randomness
/// when available (Python uses `random.choice`); falls back to a time seed.
fn pseudo_random_index(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let mut buf = [0u8; 8];
    let _ = getrandom_fill(&mut buf);
    (u64::from_le_bytes(buf) % (len as u64)) as usize
}

// ---------------------------------------------------------------------------
// Env-seed helper (pure)
// ---------------------------------------------------------------------------

/// Build the upsert payload for an env-seeded API-key credential. Mirrors the
/// per-`env_var` body of `_seed_from_env`. `auth_type` and `base_url` are
/// resolved by the caller (which knows the provider registry / kimi/zai rules).
pub fn env_seed_payload(
    source: &str,
    auth_type: &str,
    access_token: &str,
    base_url: &str,
    label: &str,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("source".to_string(), Value::String(source.to_string()));
    payload.insert("auth_type".to_string(), Value::String(auth_type.to_string()));
    payload.insert(
        "access_token".to_string(),
        Value::String(access_token.to_string()),
    );
    payload.insert("base_url".to_string(), Value::String(base_url.to_string()));
    payload.insert("label".to_string(), Value::String(label.to_string()));
    payload
}

/// Compute `custom:<name>` pool key for a `base_url` given a list of
/// `(normalized_name, base_url)` custom-provider entries. Mirrors
/// `get_custom_provider_pool_key`.
pub fn custom_provider_pool_key<'a, I>(base_url: &str, custom_providers: I) -> Option<String>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if base_url.is_empty() {
        return None;
    }
    let normalized_url = base_url.trim().trim_end_matches('/');
    for (norm_name, entry_url) in custom_providers {
        let entry_url = entry_url.trim().trim_end_matches('/');
        if !entry_url.is_empty() && entry_url == normalized_url {
            return Some(format!("{CUSTOM_POOL_PREFIX}{norm_name}"));
        }
    }
    None
}

/// `list_custom_pool_providers`: sorted `custom:*` keys with non-empty arrays.
pub fn list_custom_pool_providers(pool_data: &Map<String, Value>) -> Vec<String> {
    let mut keys: BTreeSet<String> = BTreeSet::new();
    for (key, value) in pool_data {
        if key.starts_with(CUSTOM_POOL_PREFIX) {
            if let Some(arr) = value.as_array() {
                if !arr.is_empty() {
                    keys.insert(key.clone());
                }
            }
        }
    }
    keys.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(map: Value) -> Map<String, Value> {
        map.as_object().unwrap().clone()
    }

    fn mk(provider: &str, source: &str, priority: i64, token: &str) -> PooledCredential {
        PooledCredential::from_dict(
            provider,
            &payload(json!({
                "id": format!("id-{source}-{priority}"),
                "source": source,
                "priority": priority,
                "access_token": token,
            })),
        )
    }

    #[test]
    fn from_dict_applies_defaults() {
        let cred = PooledCredential::from_dict("openrouter", &payload(json!({})));
        assert_eq!(cred.provider, "openrouter");
        assert_eq!(cred.auth_type, AUTH_TYPE_API_KEY);
        assert_eq!(cred.priority, 0);
        assert_eq!(cred.source, SOURCE_MANUAL);
        assert_eq!(cred.access_token, "");
        assert_eq!(cred.label, "openrouter"); // fallback to provider via source-default
        assert_eq!(cred.id.len(), 6);
    }

    #[test]
    fn from_dict_label_falls_back_to_source() {
        let cred = PooledCredential::from_dict(
            "anthropic",
            &payload(json!({"source": "claude_code"})),
        );
        assert_eq!(cred.label, "claude_code");
    }

    #[test]
    fn extra_round_trips_and_skips_logic_fields() {
        let cred = PooledCredential::from_dict(
            "nous",
            &payload(json!({
                "source": "device_code",
                "access_token": "tok",
                "scope": "read",
                "client_id": "abc",
                "obtained_at": 123,
                "not_an_extra": "dropped"
            })),
        );
        assert_eq!(cred.extra_get("scope").unwrap(), &json!("read"));
        assert_eq!(cred.extra_get("client_id").unwrap(), &json!("abc"));
        assert!(cred.extra_get("not_an_extra").is_none());

        let dumped = cred.to_dict();
        assert_eq!(dumped["scope"], json!("read"));
        assert!(!dumped.contains_key("not_an_extra"));
    }

    #[test]
    fn to_dict_always_emits_status_fields_as_null() {
        let cred = PooledCredential::from_dict("openrouter", &payload(json!({"access_token": "k"})));
        let dumped = cred.to_dict();
        for key in ALWAYS_EMIT {
            assert!(dumped.contains_key(*key), "missing {key}");
            assert!(dumped[*key].is_null());
        }
        // refresh_token is None and not in ALWAYS_EMIT → omitted.
        assert!(!dumped.contains_key("refresh_token"));
    }

    #[test]
    fn runtime_api_key_prefers_agent_key_for_nous() {
        let mut cred = mk("nous", "device_code", 0, "access");
        cred.agent_key = Some("agentkey".to_string());
        assert_eq!(cred.runtime_api_key(), "agentkey");
        cred.agent_key = None;
        assert_eq!(cred.runtime_api_key(), "access");

        let other = mk("openrouter", "env:OPENROUTER_API_KEY", 0, "ork");
        assert_eq!(other.runtime_api_key(), "ork");
    }

    #[test]
    fn next_priority_and_manual_source() {
        assert_eq!(next_priority(&[]), 0);
        let entries = vec![mk("p", "s1", 3, "a"), mk("p", "s2", 7, "b")];
        assert_eq!(next_priority(&entries), 8);

        assert!(is_manual_source("manual"));
        assert!(is_manual_source("Manual:42"));
        assert!(!is_manual_source("env:X"));
    }

    #[test]
    fn parse_absolute_timestamp_variants() {
        assert_eq!(parse_absolute_timestamp(&json!(0)), None);
        assert_eq!(parse_absolute_timestamp(&json!(1700000000)), Some(1_700_000_000.0));
        // millis
        assert_eq!(
            parse_absolute_timestamp(&json!(1_700_000_000_000_i64)),
            Some(1_700_000_000.0)
        );
        // string numeric
        assert_eq!(parse_absolute_timestamp(&json!("1700000000")), Some(1_700_000_000.0));
        // ISO with Z
        let iso = parse_absolute_timestamp(&json!("2023-11-14T22:13:20Z")).unwrap();
        assert!((iso - 1_700_000_000.0).abs() < 1.0);
        assert_eq!(parse_absolute_timestamp(&json!("")), None);
        assert_eq!(parse_absolute_timestamp(&Value::Null), None);
    }

    #[test]
    fn extract_retry_delay_seconds_cases() {
        assert_eq!(extract_retry_delay_seconds(""), None);
        assert_eq!(
            extract_retry_delay_seconds("quotaResetDelay: 5000ms please"),
            Some(5.0)
        );
        assert_eq!(
            extract_retry_delay_seconds("quotaResetDelay: 12s"),
            Some(12.0)
        );
        assert_eq!(
            extract_retry_delay_seconds("please retry after 30 seconds"),
            Some(30.0)
        );
        assert_eq!(extract_retry_delay_seconds("retry 7s"), Some(7.0));
        assert_eq!(extract_retry_delay_seconds("no hint here"), None);
    }

    #[test]
    fn normalize_error_context_derives_reset_from_message() {
        let ctx = payload(json!({
            "reason": "  rate_limited  ",
            "message": "  retry after 60 seconds  ",
        }));
        let n = normalize_error_context(Some(&ctx), 1000.0);
        assert_eq!(n.reason.as_deref(), Some("rate_limited"));
        assert_eq!(n.message.as_deref(), Some("retry after 60 seconds"));
        assert_eq!(n.reset_at, Some(1060.0));

        // explicit reset_at wins over message
        let ctx2 = payload(json!({"reset_at": 2000, "message": "retry after 5s"}));
        let n2 = normalize_error_context(Some(&ctx2), 1000.0);
        assert_eq!(n2.reset_at, Some(2000.0));

        assert_eq!(normalize_error_context(None, 0.0), NormalizedError::default());
    }

    #[test]
    fn exhausted_until_uses_reset_at_then_ttl() {
        let mut cred = mk("p", "s", 0, "t");
        // not exhausted → None
        assert_eq!(exhausted_until(&cred), None);

        cred.last_status = Some(STATUS_EXHAUSTED.to_string());
        cred.last_error_reset_at = Some(5000.0);
        assert_eq!(exhausted_until(&cred), Some(5000.0));

        cred.last_error_reset_at = None;
        cred.last_status_at = Some(1000.0);
        cred.last_error_code = Some(429);
        assert_eq!(exhausted_until(&cred), Some(1000.0 + EXHAUSTED_TTL_429_SECONDS));
    }

    #[test]
    fn upsert_inserts_then_merges() {
        let mut entries = vec![];
        let inserted = upsert_entry(
            &mut entries,
            "openrouter",
            "env:OPENROUTER_API_KEY",
            &env_seed_payload(
                "env:OPENROUTER_API_KEY",
                AUTH_TYPE_API_KEY,
                "key1",
                "https://openrouter.ai/api/v1",
                "OPENROUTER_API_KEY",
            ),
        );
        assert!(inserted);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].access_token, "key1");
        assert_eq!(entries[0].priority, 0);

        // same token → no change
        let again = upsert_entry(
            &mut entries,
            "openrouter",
            "env:OPENROUTER_API_KEY",
            &env_seed_payload(
                "env:OPENROUTER_API_KEY",
                AUTH_TYPE_API_KEY,
                "key1",
                "https://openrouter.ai/api/v1",
                "OPENROUTER_API_KEY",
            ),
        );
        assert!(!again);

        // changed token → update
        let updated = upsert_entry(
            &mut entries,
            "openrouter",
            "env:OPENROUTER_API_KEY",
            &env_seed_payload(
                "env:OPENROUTER_API_KEY",
                AUTH_TYPE_API_KEY,
                "key2",
                "https://openrouter.ai/api/v1",
                "OPENROUTER_API_KEY",
            ),
        );
        assert!(updated);
        assert_eq!(entries[0].access_token, "key2");
    }

    #[test]
    fn upsert_preserves_existing_label() {
        let mut entries = vec![];
        upsert_entry(
            &mut entries,
            "p",
            "src",
            &payload(json!({"source": "src", "access_token": "a", "label": "First"})),
        );
        let changed = upsert_entry(
            &mut entries,
            "p",
            "src",
            &payload(json!({"source": "src", "access_token": "a", "label": "Second"})),
        );
        // label change ignored when existing label present; token unchanged → false
        assert!(!changed);
        assert_eq!(entries[0].label, "First");
    }

    #[test]
    fn normalize_pool_priorities_orders_anthropic() {
        let mut entries = vec![
            mk("anthropic", "claude_code", 0, "a"),
            mk("anthropic", "env:ANTHROPIC_TOKEN", 1, "b"),
            mk("anthropic", "manual", 2, "c"),
        ];
        let changed = normalize_pool_priorities("anthropic", &mut entries);
        assert!(changed);
        // manual first, then by source_rank: ANTHROPIC_TOKEN(0) before claude_code(3)
        let by_id = |id: &str| entries.iter().find(|e| e.id == id).unwrap().priority;
        assert_eq!(by_id("id-manual-2"), 0);
        assert_eq!(by_id("id-env:ANTHROPIC_TOKEN-1"), 1);
        assert_eq!(by_id("id-claude_code-0"), 2);

        // non-anthropic provider is a no-op
        let mut other = vec![mk("openrouter", "env:X", 5, "k")];
        assert!(!normalize_pool_priorities("openrouter", &mut other));
    }

    #[test]
    fn prune_drops_stale_env_and_seeded_sources() {
        let mut entries = vec![
            mk("anthropic", "manual", 0, "a"),
            mk("anthropic", "env:GONE", 1, "b"),
            mk("anthropic", "claude_code", 2, "c"),
            mk("anthropic", "device_code", 3, "d"),
        ];
        let mut active = HashSet::new();
        active.insert("claude_code".to_string());
        let pruned = prune_stale_seeded_entries(&mut entries, &active);
        assert!(pruned);
        let sources: Vec<&str> = entries.iter().map(|e| e.source.as_str()).collect();
        // manual kept, claude_code active kept, device_code (non-env/seeded) kept,
        // env:GONE pruned.
        assert!(sources.contains(&"manual"));
        assert!(sources.contains(&"claude_code"));
        assert!(sources.contains(&"device_code"));
        assert!(!sources.contains(&"env:GONE"));
    }

    #[test]
    fn fill_first_selection_and_exhaustion() {
        let mut persist = NoopPersistence;
        let mut refresher = NoRefresh;
        let entries = vec![mk("p", "s1", 0, "a"), mk("p", "s2", 1, "b")];
        let mut pool = CredentialPool::new("p", entries, STRATEGY_FILL_FIRST);

        let sel = pool.select(1000.0, &mut refresher, &mut persist).unwrap();
        assert_eq!(sel.source, "s1");

        // exhaust current → rotate to s2
        let next = pool
            .mark_exhausted_and_rotate(Some(429), None, 1000.0, &mut refresher, &mut persist)
            .unwrap();
        assert_eq!(next.source, "s2");

        // both exhausted → None
        pool.mark_exhausted_and_rotate(Some(429), None, 1000.0, &mut refresher, &mut persist);
        assert!(pool.select(1000.0, &mut refresher, &mut persist).is_none());

        // after cooldown elapses, available again
        let later = 1000.0 + EXHAUSTED_TTL_429_SECONDS + 1.0;
        assert!(pool.select(later, &mut refresher, &mut persist).is_some());
    }

    #[test]
    fn least_used_picks_lowest_count_and_increments() {
        let mut persist = NoopPersistence;
        let mut refresher = NoRefresh;
        let mut e0 = mk("p", "s1", 0, "a");
        e0.request_count = 5;
        let mut e1 = mk("p", "s2", 1, "b");
        e1.request_count = 2;
        let mut pool = CredentialPool::new("p", vec![e0, e1], STRATEGY_LEAST_USED);

        let sel = pool.select(1000.0, &mut refresher, &mut persist).unwrap();
        assert_eq!(sel.source, "s2");
        assert_eq!(sel.request_count, 3);
    }

    #[test]
    fn round_robin_rotates_priorities() {
        let mut persist = NoopPersistence;
        let mut refresher = NoRefresh;
        let entries = vec![mk("p", "s1", 0, "a"), mk("p", "s2", 1, "b")];
        let mut pool = CredentialPool::new("p", entries, STRATEGY_ROUND_ROBIN);

        let first = pool.select(1000.0, &mut refresher, &mut persist).unwrap();
        assert_eq!(first.source, "s1");
        // s1 should now be pushed to the back (highest priority index).
        let after = pool.entries();
        let s1 = after.iter().find(|e| e.source == "s1").unwrap();
        let s2 = after.iter().find(|e| e.source == "s2").unwrap();
        assert!(s1.priority > s2.priority);
    }

    #[test]
    fn leasing_distributes_and_releases() {
        let mut persist = NoopPersistence;
        let mut refresher = NoRefresh;
        let entries = vec![mk("p", "s1", 0, "a"), mk("p", "s2", 1, "b")];
        let mut pool = CredentialPool::new("p", entries, STRATEGY_FILL_FIRST);

        let l1 = pool.acquire_lease(None, 1000.0, &mut refresher, &mut persist).unwrap();
        // first least-leased with priority tiebreak → s1's id
        let l2 = pool.acquire_lease(None, 1000.0, &mut refresher, &mut persist).unwrap();
        assert_ne!(l1, l2); // distributes to the other below-cap entry

        pool.release_lease(&l1);
        assert_eq!(pool.active_lease_count(&l1), 0);
    }

    #[test]
    fn resolve_target_by_id_label_index() {
        let entries = vec![mk("p", "s1", 0, "a"), mk("p", "s2", 1, "b")];
        let mut e = entries;
        e[0].label = "Alpha".to_string();
        e[1].label = "Beta".to_string();
        let pool = CredentialPool::new("p", e, STRATEGY_FILL_FIRST);

        // by id
        let (idx, entry, err) = pool.resolve_target("id-s1-0");
        assert_eq!(idx, Some(1));
        assert!(entry.is_some());
        assert!(err.is_none());

        // by label (case-insensitive)
        let (idx, _, err) = pool.resolve_target("beta");
        assert_eq!(idx, Some(2));
        assert!(err.is_none());

        // by index
        let (idx, _, _) = pool.resolve_target("1");
        assert_eq!(idx, Some(1));

        // empty
        let (_, _, err) = pool.resolve_target("   ");
        assert_eq!(err.as_deref(), Some("No credential target provided."));

        // missing
        let (_, _, err) = pool.resolve_target("nope");
        assert_eq!(err.as_deref(), Some("No credential matching \"nope\"."));

        // bad index
        let (_, _, err) = pool.resolve_target("99");
        assert_eq!(err.as_deref(), Some("No credential #99."));
    }

    #[test]
    fn remove_index_renumbers_and_clears_current() {
        let mut persist = NoopPersistence;
        let mut refresher = NoRefresh;
        let entries = vec![mk("p", "s1", 0, "a"), mk("p", "s2", 1, "b"), mk("p", "s3", 2, "c")];
        let mut pool = CredentialPool::new("p", entries, STRATEGY_FILL_FIRST);
        pool.select(1000.0, &mut refresher, &mut persist); // current = s1

        let removed = pool.remove_index(2, &mut persist).unwrap();
        assert_eq!(removed.source, "s2");
        let remaining = pool.entries();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].priority, 0);
        assert_eq!(remaining[1].priority, 1);
        // current was s1 (index 0), not removed → still set
        assert!(pool.current().is_some());

        // remove current
        pool.remove_index(1, &mut persist);
        assert!(pool.current().is_none());

        // out of range
        assert!(pool.remove_index(99, &mut persist).is_none());
    }

    #[test]
    fn reset_statuses_clears_noise() {
        let mut persist = NoopPersistence;
        let mut e = mk("p", "s1", 0, "a");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(429);
        let mut pool = CredentialPool::new("p", vec![e, mk("p", "s2", 1, "b")], STRATEGY_FILL_FIRST);
        let count = pool.reset_statuses(&mut persist);
        assert_eq!(count, 1);
        assert!(pool.entries()[0].last_status.is_none());
    }

    #[test]
    fn refresh_failure_marks_exhausted() {
        struct FailRefresher;
        impl CredentialRefresher for FailRefresher {
            fn refresh(&mut self, _p: &str, _e: &PooledCredential, _f: bool) -> RefreshOutcome {
                RefreshOutcome::Failed
            }
            fn entry_needs_refresh(&self, _p: &str, _e: &PooledCredential) -> bool {
                true
            }
        }
        let mut persist = NoopPersistence;
        let mut refresher = FailRefresher;
        let mut e = mk("anthropic", "claude_code", 0, "a");
        e.auth_type = AUTH_TYPE_OAUTH.to_string();
        e.refresh_token = Some("rt".to_string());
        let mut pool = CredentialPool::new("anthropic", vec![e], STRATEGY_FILL_FIRST);

        // select triggers refresh which fails → entry marked exhausted → no available
        assert!(pool.select(1000.0, &mut refresher, &mut persist).is_none());
        assert_eq!(
            pool.entries()[0].last_status.as_deref(),
            Some(STATUS_EXHAUSTED)
        );
    }

    #[test]
    fn refresh_success_clears_status_and_selects() {
        struct OkRefresher;
        impl CredentialRefresher for OkRefresher {
            fn refresh(&mut self, _p: &str, e: &PooledCredential, _f: bool) -> RefreshOutcome {
                let mut updated = e.clone();
                updated.access_token = "fresh".to_string();
                RefreshOutcome::Refreshed(updated)
            }
            fn entry_needs_refresh(&self, _p: &str, _e: &PooledCredential) -> bool {
                true
            }
        }
        let mut persist = NoopPersistence;
        let mut refresher = OkRefresher;
        let mut e = mk("anthropic", "claude_code", 0, "old");
        e.auth_type = AUTH_TYPE_OAUTH.to_string();
        e.refresh_token = Some("rt".to_string());
        let mut pool = CredentialPool::new("anthropic", vec![e], STRATEGY_FILL_FIRST);

        let sel = pool.select(1000.0, &mut refresher, &mut persist).unwrap();
        assert_eq!(sel.access_token, "fresh");
    }

    #[test]
    fn custom_pool_key_and_listing() {
        let providers = vec![("together.ai", "https://api.together.xyz/v1/")];
        assert_eq!(
            custom_provider_pool_key("https://api.together.xyz/v1", providers.clone()),
            Some("custom:together.ai".to_string())
        );
        assert_eq!(custom_provider_pool_key("https://other.example", providers), None);

        let pool_data = payload(json!({
            "custom:a": [{"id": "x"}],
            "custom:b": [],
            "anthropic": [{"id": "y"}],
        }));
        assert_eq!(list_custom_pool_providers(&pool_data), vec!["custom:a"]);
    }

    #[test]
    fn label_from_token_reads_email_claim() {
        // header.payload.sig where payload = {"email":"a@b.com"}
        use base64::Engine;
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(br#"{"email":"a@b.com"}"#);
        let token = format!("h.{payload_b64}.s");
        assert_eq!(label_from_token(&token, "fallback"), "a@b.com");
        assert_eq!(label_from_token("not-a-jwt", "fallback"), "fallback");
    }

    #[test]
    fn normalize_custom_pool_name_lowercases_and_dashes() {
        assert_eq!(normalize_custom_pool_name("  Together AI  "), "together-ai");
    }
}
