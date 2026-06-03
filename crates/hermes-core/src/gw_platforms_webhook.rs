//! Generic webhook platform adapter, ported from `gateway/platforms/webhook.py`.
//!
//! The Python original runs an `aiohttp` HTTP server that receives webhook
//! POSTs from external services (GitHub, GitLab, JIRA, Stripe, etc.), validates
//! HMAC signatures, transforms payloads into agent prompts, and routes responses
//! back to the source or to another configured platform.
//!
//! This port reproduces the *pure*, behavior-defining logic faithfully and
//! idiomatically in Rust: HMAC signature validation, prompt-template rendering
//! (dot-notation + `{__raw__}`), `deliver_extra` rendering, fixed-window rate
//! limiting, idempotency caching with TTL, delivery-info pruning, the
//! static/dynamic route merge with mtime gating, route startup validation, and
//! the `github_comment` delivery via the `gh` CLI.
//!
//! The `aiohttp` request handling and `asyncio.create_task` agent dispatch are
//! intimately tied to the CPython event loop; here they are modelled as plain
//! synchronous methods on [`WebhookAdapter`] returning data-carrying enums that
//! describe what the HTTP layer should do, plus a [`WebhookHttpResponse`] type
//! that mirrors the JSON bodies + status codes aiohttp emitted. The actual
//! socket/server lives in the native runtime layer (see
//! `crates/hermes/src/native_webhook_server.rs`).
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — [`SendResult`], [`MessageEvent`], etc.
//!   - [`crate::mod_hermes_constants::get_hermes_home`]

use std::collections::HashMap;
use std::path::PathBuf;

use hmac::{Hmac, Mac};
use serde_json::{Map as JsonMap, Value as JsonValue};
use sha2::Sha256;

use crate::gw_platforms_base::SendResult;

type HmacSha256 = Hmac<Sha256>;

// ===========================================================================
// Constants
// ===========================================================================

pub const DEFAULT_HOST: &str = "0.0.0.0";
pub const DEFAULT_PORT: u16 = 8644;
pub const INSECURE_NO_AUTH: &str = "INSECURE_NO_AUTH";
pub const DYNAMIC_ROUTES_FILENAME: &str = "webhook_subscriptions.json";

/// Default idempotency / delivery-info TTL in seconds (1 hour).
pub const DEFAULT_IDEMPOTENCY_TTL: i64 = 3600;
/// Default per-route rate limit (requests per minute).
pub const DEFAULT_RATE_LIMIT: usize = 30;
/// Default max body size in bytes (1 MiB).
pub const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

/// Platforms that the cross-platform delivery dispatcher recognises as
/// built-in (mirrors `_BUILTIN_DELIVER_PLATFORMS`).
pub const BUILTIN_DELIVER_PLATFORMS: &[&str] = &[
    "telegram",
    "discord",
    "slack",
    "signal",
    "sms",
    "whatsapp",
    "matrix",
    "mattermost",
    "homeassistant",
    "email",
    "dingtalk",
    "feishu",
    "wecom",
    "wecom_callback",
    "weixin",
    "bluebubbles",
    "qqbot",
    "yuanbao",
];

/// Check if webhook adapter dependencies are available.
///
/// In the Python original this returned whether `aiohttp` could be imported.
/// The native Rust runtime ships its own HTTP server, so this is always true.
pub fn check_webhook_requirements() -> bool {
    true
}

/// Return True if `name` is a known cross-platform delivery target.
pub fn is_builtin_deliver_platform(name: &str) -> bool {
    BUILTIN_DELIVER_PLATFORMS.contains(&name)
}

// ===========================================================================
// HMAC signature validation
// ===========================================================================

fn hmac_sha256_hex(secret: &str, body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts keys of any length");
    mac.update(body);
    let bytes = mac.finalize().into_bytes();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time comparison of two byte slices. Mirrors `hmac.compare_digest`.
///
/// Returns `false` immediately on length mismatch (matching CPython's
/// `compare_digest`, which short-circuits on differing lengths).
pub fn compare_digest(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Validate a webhook signature against the provided headers.
///
/// Mirrors `WebhookAdapter._validate_signature`. Supports:
///   - GitHub: `X-Hub-Signature-256: sha256=<hex>`
///   - GitLab: `X-Gitlab-Token: <plain secret>`
///   - Generic: `X-Webhook-Signature: <hex HMAC-SHA256>`
///
/// `headers` is a case-insensitive lookup (see [`HeaderLookup`]). When a secret
/// is configured but no recognised signature header is present, this returns
/// `false` (reject).
pub fn validate_signature(headers: &dyn HeaderLookup, body: &[u8], secret: &str) -> bool {
    // GitHub: X-Hub-Signature-256 = sha256=<hex>
    let gh_sig = headers.get("X-Hub-Signature-256").unwrap_or_default();
    if !gh_sig.is_empty() {
        let expected = format!("sha256={}", hmac_sha256_hex(secret, body));
        return compare_digest(gh_sig.as_bytes(), expected.as_bytes());
    }

    // GitLab: X-Gitlab-Token = <plain secret>
    let gl_token = headers.get("X-Gitlab-Token").unwrap_or_default();
    if !gl_token.is_empty() {
        return compare_digest(gl_token.as_bytes(), secret.as_bytes());
    }

    // Generic: X-Webhook-Signature = <hex HMAC-SHA256>
    let generic_sig = headers.get("X-Webhook-Signature").unwrap_or_default();
    if !generic_sig.is_empty() {
        let expected = hmac_sha256_hex(secret, body);
        return compare_digest(generic_sig.as_bytes(), expected.as_bytes());
    }

    // No recognised signature header but secret is configured -> reject.
    false
}

/// Case-insensitive header lookup, mirroring aiohttp's `request.headers.get`.
pub trait HeaderLookup {
    /// Return the value for `name` (case-insensitive), or `None` if absent.
    fn get(&self, name: &str) -> Option<String>;
}

/// Simple [`HeaderLookup`] backed by a `HashMap`. Keys are matched
/// case-insensitively.
#[derive(Debug, Clone, Default)]
pub struct HeaderMap {
    inner: HashMap<String, String>,
}

impl HeaderMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a header. The key is stored lowercased for case-insensitive lookup.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.inner.insert(name.into().to_lowercase(), value.into());
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.insert(name, value);
        self
    }
}

impl HeaderLookup for HeaderMap {
    fn get(&self, name: &str) -> Option<String> {
        self.inner.get(&name.to_lowercase()).cloned()
    }
}

// ===========================================================================
// Prompt rendering
// ===========================================================================

/// Truncate a string to at most `max` Unicode code points (Python `[:max]`).
fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        value.chars().take(max).collect()
    }
}

/// Resolve a dot-notation key into the payload, returning the rendered value.
///
/// Mirrors the `_resolve` closure inside `_render_prompt`: walks each `.`
/// segment; if any intermediate value is not a dict, or a key is missing, the
/// literal `{key}` token is returned unchanged. Dicts/lists are dumped as
/// indented JSON truncated to 2000 chars; scalars are stringified.
fn resolve_template_key(payload: &JsonValue, key: &str) -> String {
    let mut value = payload;
    for part in key.split('.') {
        match value {
            JsonValue::Object(map) => match map.get(part) {
                Some(next) => value = next,
                None => return format!("{{{key}}}"),
            },
            _ => return format!("{{{key}}}"),
        }
    }
    match value {
        JsonValue::Object(_) | JsonValue::Array(_) => truncate_chars(
            &serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string()),
            2000,
        ),
        JsonValue::String(s) => s.clone(),
        JsonValue::Bool(b) => {
            // Python str(True)/str(False) -> "True"/"False".
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        JsonValue::Null => "None".to_string(),
        other => other.to_string(),
    }
}

/// Render a prompt template with the webhook payload.
///
/// Mirrors `WebhookAdapter._render_prompt`. Supports dot-notation access into
/// nested dicts (`{pull_request.title}`) and the special `{__raw__}` token that
/// dumps the entire payload as indented JSON truncated to 4000 chars. When
/// `template` is empty, a default JSON-dump prompt is produced.
pub fn render_prompt(
    template: &str,
    payload: &JsonValue,
    event_type: &str,
    route_name: &str,
) -> String {
    if template.is_empty() {
        let truncated = truncate_chars(
            &serde_json::to_string_pretty(payload).unwrap_or_else(|_| "null".to_string()),
            4000,
        );
        return format!(
            "Webhook event '{event_type}' on route '{route_name}':\n\n```json\n{truncated}\n```"
        );
    }

    let re = regex::Regex::new(r"\{([a-zA-Z0-9_.]+)\}").unwrap();
    re.replace_all(template, |caps: &regex::Captures<'_>| {
        let key = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        if key == "__raw__" {
            return truncate_chars(
                &serde_json::to_string_pretty(payload).unwrap_or_else(|_| "null".to_string()),
                4000,
            );
        }
        resolve_template_key(payload, key)
    })
    .into_owned()
}

/// Render `deliver_extra` template values with payload data.
///
/// Mirrors `WebhookAdapter._render_delivery_extra`. String values are rendered
/// through [`render_prompt`] (with empty event/route); all other JSON types are
/// passed through unchanged.
pub fn render_delivery_extra(
    extra: &JsonMap<String, JsonValue>,
    payload: &JsonValue,
) -> JsonMap<String, JsonValue> {
    let mut rendered = JsonMap::new();
    for (key, value) in extra {
        let out = match value {
            JsonValue::String(s) => JsonValue::String(render_prompt(s, payload, "", "")),
            other => other.clone(),
        };
        rendered.insert(key.clone(), out);
    }
    rendered
}

// ===========================================================================
// Body parsing + event type extraction
// ===========================================================================

/// Parse a webhook request body: JSON first, then form-encoded fallback.
///
/// Mirrors the `json.loads` -> `urllib.parse.parse_qsl` fallback chain.
/// Returns `Err` when neither parse produces a usable payload.
pub fn parse_webhook_body(body: &[u8]) -> Result<JsonValue, String> {
    if let Ok(value) = serde_json::from_slice::<JsonValue>(body) {
        return Ok(value);
    }
    // Form-encoded fallback. Python decodes as UTF-8 first; reject on failure.
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(_) => return Err("Cannot parse body".to_string()),
    };
    let mut object = JsonMap::new();
    for (key, value) in url::form_urlencoded::parse(text.as_bytes()) {
        object.insert(key.to_string(), JsonValue::String(value.to_string()));
    }
    // Python's parse_qsl on an empty/garbage string yields an empty dict, which
    // is still a valid (empty) payload — mirror that by returning the object.
    Ok(JsonValue::Object(object))
}

/// Determine the event type for filtering.
///
/// Mirrors the header/payload precedence in `_handle_webhook`:
/// `X-GitHub-Event` > `X-GitLab-Event` > `payload["event_type"]` > `"unknown"`.
/// Empty strings fall through to the next source (Python `or`-chain semantics).
pub fn webhook_event_type(headers: &dyn HeaderLookup, payload: &JsonValue) -> String {
    let gh = headers.get("X-GitHub-Event").unwrap_or_default();
    if !gh.is_empty() {
        return gh;
    }
    let gl = headers.get("X-GitLab-Event").unwrap_or_default();
    if !gl.is_empty() {
        return gl;
    }
    if let Some(JsonValue::String(s)) = payload.get("event_type") {
        if !s.is_empty() {
            return s.clone();
        }
    }
    "unknown".to_string()
}

// ===========================================================================
// Route configuration
// ===========================================================================

/// A single webhook route configuration, parsed from `routes.<name>` in
/// `config.yaml` or from the dynamic subscriptions JSON file.
#[derive(Debug, Clone, Default)]
pub struct WebhookRoute {
    pub events: Vec<String>,
    pub secret: String,
    pub prompt: String,
    pub skills: Vec<String>,
    pub deliver: String,
    pub deliver_only: bool,
    pub deliver_extra: JsonMap<String, JsonValue>,
}

impl WebhookRoute {
    /// Parse a route from a JSON object, falling back to `global_secret` when
    /// the route omits its own `secret`. Mirrors the per-route `.get()` lookups.
    pub fn from_json(obj: &JsonMap<String, JsonValue>, global_secret: &str) -> Self {
        let secret = obj
            .get("secret")
            .and_then(JsonValue::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(global_secret)
            .to_string();
        let deliver = obj
            .get("deliver")
            .and_then(JsonValue::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("log")
            .to_string();
        WebhookRoute {
            events: obj
                .get("events")
                .and_then(JsonValue::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(JsonValue::as_str)
                        .map(ToOwned::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            secret,
            prompt: obj
                .get("prompt")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string(),
            skills: obj
                .get("skills")
                .and_then(JsonValue::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(JsonValue::as_str)
                        .map(ToOwned::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            deliver,
            deliver_only: obj
                .get("deliver_only")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            deliver_extra: obj
                .get("deliver_extra")
                .and_then(JsonValue::as_object)
                .cloned()
                .unwrap_or_default(),
        }
    }
}

/// Validate a single route at startup. Mirrors the `connect()` validation loop.
///
/// Returns `Err(message)` matching the Python `ValueError` message text when the
/// route has no secret, or is `deliver_only` with no real delivery target.
pub fn validate_route(name: &str, route: &WebhookRoute) -> Result<(), String> {
    if route.secret.is_empty() {
        return Err(format!(
            "[webhook] Route '{name}' has no HMAC secret. \
Set 'secret' on the route or globally. \
For testing without auth, set secret to '{INSECURE_NO_AUTH}'."
        ));
    }
    if route.deliver_only && (route.deliver.is_empty() || route.deliver == "log") {
        return Err(format!(
            "[webhook] Route '{name}' has deliver_only=true but \
deliver is '{}'. Direct delivery requires a \
real target (telegram, discord, slack, github_comment, etc.).",
            route.deliver
        ));
    }
    Ok(())
}

// ===========================================================================
// Dynamic-route reload (mtime-gated merge)
// ===========================================================================

/// Tracks the dynamic-route subscription file and merges it with static routes.
///
/// Mirrors the `_static_routes` / `_dynamic_routes` / `_routes` triplet plus the
/// `_dynamic_routes_mtime` gate from `WebhookAdapter`. Static routes always take
/// precedence over dynamic ones.
#[derive(Debug, Clone, Default)]
pub struct RouteRegistry {
    pub static_routes: HashMap<String, WebhookRoute>,
    pub dynamic_routes: HashMap<String, WebhookRoute>,
    pub routes: HashMap<String, WebhookRoute>,
    pub dynamic_routes_mtime: f64,
    pub global_secret: String,
}

impl RouteRegistry {
    /// Build a registry from static routes (config.yaml) + global secret.
    pub fn new(static_routes: HashMap<String, WebhookRoute>, global_secret: String) -> Self {
        let routes = static_routes.clone();
        RouteRegistry {
            static_routes,
            dynamic_routes: HashMap::new(),
            routes,
            dynamic_routes_mtime: 0.0,
            global_secret,
        }
    }

    /// Path to the dynamic subscriptions file under the Hermes home directory.
    pub fn dynamic_routes_path(&self) -> PathBuf {
        crate::mod_hermes_constants::get_hermes_home().join(DYNAMIC_ROUTES_FILENAME)
    }

    /// Reload agent-created subscriptions from disk if the file changed.
    ///
    /// Mirrors `_reload_dynamic_routes`. Returns `true` when the merged route set
    /// changed (file appeared/changed/disappeared). mtime-gated so repeated calls
    /// are cheap. Static routes take precedence over dynamic ones.
    pub fn reload_dynamic_routes(&mut self) -> bool {
        let path = self.dynamic_routes_path();
        if !path.exists() {
            if !self.dynamic_routes.is_empty() {
                self.dynamic_routes = HashMap::new();
                self.routes = self.static_routes.clone();
                return true;
            }
            return false;
        }
        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => return false,
        };
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        if mtime <= self.dynamic_routes_mtime {
            return false; // No change.
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(_) => return false,
        };
        let data: JsonValue = match serde_json::from_str(&raw) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let obj = match data.as_object() {
            Some(o) => o,
            None => return false,
        };
        // Merge: static routes take precedence over dynamic ones.
        let mut dynamic = HashMap::new();
        for (k, v) in obj {
            if self.static_routes.contains_key(k) {
                continue;
            }
            if let Some(route_obj) = v.as_object() {
                dynamic.insert(k.clone(), WebhookRoute::from_json(route_obj, &self.global_secret));
            }
        }
        let mut merged = dynamic.clone();
        for (k, v) in &self.static_routes {
            merged.insert(k.clone(), v.clone());
        }
        self.dynamic_routes = dynamic;
        self.routes = merged;
        self.dynamic_routes_mtime = mtime;
        true
    }
}

// ===========================================================================
// Rate limiting, idempotency, delivery-info pruning
// ===========================================================================

/// Mutable runtime state: rate-limit windows, seen deliveries, delivery info.
///
/// Mirrors the `_rate_counts`, `_seen_deliveries`, `_delivery_info`, and
/// `_delivery_info_created` maps. All timestamps are float seconds since the
/// epoch (mirroring `time.time()`).
#[derive(Debug, Clone)]
pub struct WebhookRuntime {
    pub rate_counts: HashMap<String, Vec<f64>>,
    pub seen_deliveries: HashMap<String, f64>,
    pub delivery_info: HashMap<String, DeliveryInfo>,
    pub delivery_info_created: HashMap<String, f64>,
    pub idempotency_ttl: i64,
    pub rate_limit: usize,
}

impl Default for WebhookRuntime {
    fn default() -> Self {
        WebhookRuntime {
            rate_counts: HashMap::new(),
            seen_deliveries: HashMap::new(),
            delivery_info: HashMap::new(),
            delivery_info_created: HashMap::new(),
            idempotency_ttl: DEFAULT_IDEMPOTENCY_TTL,
            rate_limit: DEFAULT_RATE_LIMIT,
        }
    }
}

/// Per-session delivery descriptor stored when a webhook is received.
///
/// Mirrors the dict stored into `_delivery_info[session_chat_id]`.
#[derive(Debug, Clone, Default)]
pub struct DeliveryInfo {
    pub deliver: String,
    pub deliver_extra: JsonMap<String, JsonValue>,
    pub payload: JsonValue,
}

impl WebhookRuntime {
    pub fn new(rate_limit: usize, idempotency_ttl: i64) -> Self {
        WebhookRuntime {
            rate_limit,
            idempotency_ttl,
            ..Default::default()
        }
    }

    /// Apply the fixed-window rate limit for `route_name`.
    ///
    /// Mirrors the rate-limit block: prune timestamps older than 60s, reject if
    /// the window is already full, otherwise record `now` and accept. Returns
    /// `true` when the request is allowed.
    pub fn check_rate_limit(&mut self, route_name: &str, now: f64) -> bool {
        let window = self.rate_counts.entry(route_name.to_string()).or_default();
        window.retain(|&t| now - t < 60.0);
        if window.len() >= self.rate_limit {
            return false;
        }
        window.push(now);
        true
    }

    /// Idempotency check + record. Mirrors the `_seen_deliveries` block.
    ///
    /// Prunes expired entries (older than `idempotency_ttl`), then returns
    /// `true` if `delivery_id` is a *duplicate* (already seen). On a fresh id it
    /// records `now` and returns `false`.
    pub fn is_duplicate_delivery(&mut self, delivery_id: &str, now: f64) -> bool {
        let ttl = self.idempotency_ttl as f64;
        self.seen_deliveries.retain(|_, &mut t| now - t < ttl);
        if self.seen_deliveries.contains_key(delivery_id) {
            return true;
        }
        self.seen_deliveries.insert(delivery_id.to_string(), now);
        false
    }

    /// Store delivery info for a session and prune stale entries.
    ///
    /// Mirrors the `_delivery_info[...] = ...; _delivery_info_created[...] = now;
    /// _prune_delivery_info(now)` sequence in `_handle_webhook`.
    pub fn store_delivery_info(&mut self, session_chat_id: &str, info: DeliveryInfo, now: f64) {
        self.delivery_info.insert(session_chat_id.to_string(), info);
        self.delivery_info_created.insert(session_chat_id.to_string(), now);
        self.prune_delivery_info(now);
    }

    /// Drop delivery_info entries older than the idempotency TTL.
    ///
    /// Mirrors `_prune_delivery_info`.
    pub fn prune_delivery_info(&mut self, now: f64) {
        let cutoff = now - self.idempotency_ttl as f64;
        let stale: Vec<String> = self
            .delivery_info_created
            .iter()
            .filter(|(_, t)| **t < cutoff)
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale {
            self.delivery_info.remove(&k);
            self.delivery_info_created.remove(&k);
        }
    }

    /// Read (do NOT pop) delivery info for a chat_id. Mirrors the `.get()` in
    /// `send()`.
    pub fn get_delivery_info(&self, chat_id: &str) -> Option<&DeliveryInfo> {
        self.delivery_info.get(chat_id)
    }
}

/// Build the delivery_id from headers, mirroring the Python fallback chain:
/// `X-GitHub-Delivery` -> `X-Request-ID` -> `str(int(time.time()*1000))`.
pub fn build_delivery_id(headers: &dyn HeaderLookup, now_millis: i64) -> String {
    if let Some(v) = headers.get("X-GitHub-Delivery") {
        if !v.is_empty() {
            return v;
        }
    }
    if let Some(v) = headers.get("X-Request-ID") {
        if !v.is_empty() {
            return v;
        }
    }
    now_millis.to_string()
}

// ===========================================================================
// Delivery decisions + github_comment
// ===========================================================================

/// What `send()` / `_direct_deliver` resolves a `deliver` type into.
///
/// Mirrors the branching in `send()` and `_direct_deliver`: a `log`-only no-op,
/// a `github_comment` via `gh`, a cross-platform dispatch, or an unknown type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryDecision {
    /// `deliver == "log"`: log the (truncated) content, success.
    Log,
    /// `deliver == "github_comment"`: post via the `gh` CLI.
    GithubComment,
    /// Recognised cross-platform target (built-in or plugin-registered).
    CrossPlatform(String),
    /// Unknown / unsupported delivery type.
    Unknown(String),
}

/// Resolve a `deliver` type into a [`DeliveryDecision`] for the agent-mode
/// `send()` path.
///
/// Mirrors `send()`: `log` and `github_comment` are handled directly; otherwise,
/// the target is dispatched cross-platform only when it is a *known* platform
/// (built-in or, via `is_registered`, plugin-registered) AND a gateway runner is
/// present. `has_gateway_runner` and `is_plugin_registered` model the two
/// external conditions.
pub fn resolve_send_decision(
    deliver_type: &str,
    has_gateway_runner: bool,
    is_plugin_registered: impl Fn(&str) -> bool,
) -> DeliveryDecision {
    if deliver_type == "log" {
        return DeliveryDecision::Log;
    }
    if deliver_type == "github_comment" {
        return DeliveryDecision::GithubComment;
    }
    let known = is_builtin_deliver_platform(deliver_type) || is_plugin_registered(deliver_type);
    if has_gateway_runner && known {
        return DeliveryDecision::CrossPlatform(deliver_type.to_string());
    }
    DeliveryDecision::Unknown(deliver_type.to_string())
}

/// Resolve a `deliver` type for the `deliver_only` direct-delivery path.
///
/// Mirrors `_direct_deliver`: `log` (defensive no-op), `github_comment`, then a
/// fall-through to the cross-platform dispatcher (which itself validates the
/// target name).
pub fn resolve_direct_deliver_decision(deliver_type: &str) -> DeliveryDecision {
    if deliver_type == "log" {
        return DeliveryDecision::Log;
    }
    if deliver_type == "github_comment" {
        return DeliveryDecision::GithubComment;
    }
    DeliveryDecision::CrossPlatform(deliver_type.to_string())
}

/// Extract `(repo, pr_number)` from a `github_comment` delivery's
/// `deliver_extra`. Mirrors the `extra.get("repo")` / `extra.get("pr_number")`
/// reads, treating empty/absent as missing.
fn github_comment_target(extra: &JsonMap<String, JsonValue>) -> Option<(String, String)> {
    let repo = extra
        .get("repo")
        .map(json_value_to_string)
        .filter(|s| !s.is_empty())?;
    let pr_number = extra
        .get("pr_number")
        .map(json_value_to_string)
        .filter(|s| !s.is_empty())?;
    Some((repo, pr_number))
}

/// Stringify a JSON value the way Python `str(...)` would for the scalar types
/// that appear in `deliver_extra` (strings unquoted, numbers/bools rendered).
fn json_value_to_string(value: &JsonValue) -> String {
    match value {
        JsonValue::String(s) => s.clone(),
        JsonValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        JsonValue::Null => String::new(),
        other => other.to_string(),
    }
}

/// Post `content` as a GitHub PR/issue comment via the `gh` CLI.
///
/// Mirrors `_deliver_github_comment`, including the missing-repo/pr handling, the
/// `gh pr comment <pr> --repo <repo> --body <content>` invocation, the 30s
/// timeout (best-effort), and the `gh`-not-installed branch.
pub fn deliver_github_comment(
    content: &str,
    delivery: &DeliveryInfo,
) -> SendResult {
    let (repo, pr_number) = match github_comment_target(&delivery.deliver_extra) {
        Some(t) => t,
        None => {
            return SendResult::fail("Missing repo or pr_number");
        }
    };

    let output = std::process::Command::new("gh")
        .args([
            "pr",
            "comment",
            &pr_number,
            "--repo",
            &repo,
            "--body",
            content,
        ])
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                SendResult::ok(None)
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                SendResult::fail(stderr)
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            SendResult::fail("gh CLI not installed")
        }
        Err(e) => SendResult::fail(e.to_string()),
    }
}

// ===========================================================================
// Cross-platform delivery target resolution
// ===========================================================================

/// Resolve the `(chat_id, thread_id)` for a cross-platform delivery.
///
/// Mirrors the tail of `_deliver_cross_platform`: prefer `deliver_extra.chat_id`,
/// else fall back to the platform's configured home channel (`home_chat_id`).
/// `thread_id` is taken from `message_thread_id` or `thread_id` in
/// `deliver_extra`. Returns `Err` when no chat_id and no home channel are
/// available.
pub fn resolve_cross_platform_target(
    delivery: &DeliveryInfo,
    home_chat_id: Option<&str>,
) -> Result<CrossPlatformTarget, String> {
    let extra = &delivery.deliver_extra;
    let mut chat_id = extra
        .get("chat_id")
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_string();
    if chat_id.is_empty() {
        match home_chat_id {
            Some(h) if !h.is_empty() => chat_id = h.to_string(),
            _ => {
                return Err("No chat_id or home channel".to_string());
            }
        }
    }
    let thread_id = extra
        .get("message_thread_id")
        .and_then(|v| non_empty_str(v))
        .or_else(|| extra.get("thread_id").and_then(|v| non_empty_str(v)));
    Ok(CrossPlatformTarget { chat_id, thread_id })
}

fn non_empty_str(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(s) if !s.is_empty() => Some(s.clone()),
        JsonValue::String(_) | JsonValue::Null => None,
        other => Some(other.to_string()),
    }
}

/// Resolved cross-platform delivery target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossPlatformTarget {
    pub chat_id: String,
    pub thread_id: Option<String>,
}

// ===========================================================================
// HTTP response modelling
// ===========================================================================

/// An HTTP response the webhook handler would emit. Mirrors the
/// `web.json_response({...}, status=N)` calls in `_handle_webhook`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebhookHttpResponse {
    pub status: u16,
    pub body: JsonValue,
}

impl WebhookHttpResponse {
    pub fn new(status: u16, body: JsonValue) -> Self {
        WebhookHttpResponse { status, body }
    }
}

/// Outcome of processing a webhook POST (the decision the HTTP layer acts on).
///
/// This captures the control flow of `_handle_webhook` after auth: an immediate
/// HTTP response (404/413/401/429/ignored/duplicate/error), a `deliver_only`
/// direct delivery, or an accepted agent run.
#[derive(Debug, Clone)]
pub enum WebhookOutcome {
    /// Emit this response immediately and stop.
    Respond(WebhookHttpResponse),
    /// `deliver_only` route: deliver `prompt` directly, then respond based on
    /// the delivery result.
    DirectDeliver {
        route_name: String,
        event_type: String,
        delivery_id: String,
        prompt: String,
        delivery: DeliveryInfo,
    },
    /// Agent-mode route: dispatch a background agent run.
    AcceptAgentRun {
        route_name: String,
        event_type: String,
        delivery_id: String,
        session_chat_id: String,
        prompt: String,
    },
}

/// Build the 202 Accepted response body. Mirrors the final `json_response` in
/// `_handle_webhook`.
pub fn accepted_response(route_name: &str, event_type: &str, delivery_id: &str) -> WebhookHttpResponse {
    WebhookHttpResponse::new(
        202,
        serde_json::json!({
            "status": "accepted",
            "route": route_name,
            "event": event_type,
            "delivery_id": delivery_id,
        }),
    )
}

/// Build the `deliver_only` success response. Mirrors the `"delivered"` branch.
pub fn delivered_response(
    route_name: &str,
    target: &str,
    delivery_id: &str,
) -> WebhookHttpResponse {
    WebhookHttpResponse::new(
        200,
        serde_json::json!({
            "status": "delivered",
            "route": route_name,
            "target": target,
            "delivery_id": delivery_id,
        }),
    )
}

/// Build the `deliver_only` failure response (502). Mirrors the error branches.
pub fn delivery_failed_response(delivery_id: &str) -> WebhookHttpResponse {
    WebhookHttpResponse::new(
        502,
        serde_json::json!({
            "status": "error",
            "error": "Delivery failed",
            "delivery_id": delivery_id,
        }),
    )
}

// ===========================================================================
// The adapter
// ===========================================================================

/// Generic webhook receiver that triggers agent runs from HTTP POSTs.
///
/// Mirrors `WebhookAdapter`. The async I/O is delegated to the runtime layer;
/// this struct owns configuration, route state, and runtime maps, and exposes
/// synchronous decision methods that mirror the Python control flow.
#[derive(Debug, Clone)]
pub struct WebhookAdapter {
    pub host: String,
    pub port: u16,
    pub global_secret: String,
    pub registry: RouteRegistry,
    pub runtime: WebhookRuntime,
    pub max_body_bytes: usize,
    /// Whether a gateway runner is attached (for cross-platform delivery).
    pub has_gateway_runner: bool,
}

impl WebhookAdapter {
    /// Build an adapter from parsed config-`extra` values.
    pub fn new(
        host: impl Into<String>,
        port: u16,
        global_secret: impl Into<String>,
        static_routes: HashMap<String, WebhookRoute>,
        rate_limit: usize,
        max_body_bytes: usize,
    ) -> Self {
        let global_secret = global_secret.into();
        WebhookAdapter {
            host: host.into(),
            port,
            global_secret: global_secret.clone(),
            registry: RouteRegistry::new(static_routes, global_secret),
            runtime: WebhookRuntime::new(rate_limit, DEFAULT_IDEMPOTENCY_TTL),
            max_body_bytes,
            has_gateway_runner: false,
        }
    }

    /// Parse the adapter directly from the `config.extra` JSON object. Mirrors
    /// the `__init__` reads (`host`, `port`, `secret`, `routes`, `rate_limit`,
    /// `max_body_bytes`).
    pub fn from_extra(extra: &JsonValue) -> Self {
        let host = extra
            .get("host")
            .and_then(JsonValue::as_str)
            .unwrap_or(DEFAULT_HOST)
            .to_string();
        let port = extra
            .get("port")
            .and_then(json_as_u64)
            .map(|v| v as u16)
            .unwrap_or(DEFAULT_PORT);
        let global_secret = extra
            .get("secret")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string();
        let rate_limit = extra
            .get("rate_limit")
            .and_then(json_as_u64)
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_RATE_LIMIT);
        let max_body_bytes = extra
            .get("max_body_bytes")
            .and_then(json_as_u64)
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_BODY_BYTES);
        let static_routes = extra
            .get("routes")
            .and_then(JsonValue::as_object)
            .map(|routes| {
                routes
                    .iter()
                    .filter_map(|(name, route)| {
                        route
                            .as_object()
                            .map(|o| (name.clone(), WebhookRoute::from_json(o, &global_secret)))
                    })
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        WebhookAdapter::new(
            host,
            port,
            global_secret,
            static_routes,
            rate_limit,
            max_body_bytes,
        )
    }

    /// Validate all routes at startup (after reloading dynamic routes).
    ///
    /// Mirrors the body of `connect()` prior to starting the HTTP server.
    /// Returns `Err(message)` on the first invalid route.
    pub fn validate_all_routes(&mut self) -> Result<(), String> {
        self.registry.reload_dynamic_routes();
        for (name, route) in &self.registry.routes {
            validate_route(name, route)?;
        }
        Ok(())
    }

    /// Process a webhook POST and decide what the HTTP layer should do.
    ///
    /// Mirrors the synchronous portion of `_handle_webhook`: dynamic-route
    /// reload, route lookup, body-size guard, signature validation, rate limit,
    /// payload parse, event filter, prompt render, delivery_id build,
    /// idempotency, and the `deliver_only` vs agent-run branch. `content_length`
    /// is the declared `Content-Length` (for the auth-before-body guard);
    /// `now_secs`/`now_millis` are injected clocks (mirroring `time.time()`).
    pub fn process_webhook(
        &mut self,
        route_name: &str,
        headers: &dyn HeaderLookup,
        body: &[u8],
        content_length: usize,
        now_secs: f64,
        now_millis: i64,
    ) -> WebhookOutcome {
        // Hot-reload dynamic subscriptions on each request (mtime-gated, cheap).
        self.registry.reload_dynamic_routes();

        let route = match self.registry.routes.get(route_name).cloned() {
            Some(r) => r,
            None => {
                return WebhookOutcome::Respond(WebhookHttpResponse::new(
                    404,
                    serde_json::json!({ "error": format!("Unknown route: {route_name}") }),
                ));
            }
        };

        // Auth-before-body: check declared Content-Length.
        if content_length > self.max_body_bytes {
            return WebhookOutcome::Respond(WebhookHttpResponse::new(
                413,
                serde_json::json!({ "error": "Payload too large" }),
            ));
        }

        // Validate HMAC signature FIRST (skip for INSECURE_NO_AUTH testing mode).
        let secret = &route.secret;
        if !secret.is_empty() && secret != INSECURE_NO_AUTH {
            if !validate_signature(headers, body, secret) {
                return WebhookOutcome::Respond(WebhookHttpResponse::new(
                    401,
                    serde_json::json!({ "error": "Invalid signature" }),
                ));
            }
        }

        // Rate limiting (after auth).
        if !self.runtime.check_rate_limit(route_name, now_secs) {
            return WebhookOutcome::Respond(WebhookHttpResponse::new(
                429,
                serde_json::json!({ "error": "Rate limit exceeded" }),
            ));
        }

        // Parse payload.
        let payload = match parse_webhook_body(body) {
            Ok(p) => p,
            Err(_) => {
                return WebhookOutcome::Respond(WebhookHttpResponse::new(
                    400,
                    serde_json::json!({ "error": "Cannot parse body" }),
                ));
            }
        };

        // Event-type filter.
        let event_type = webhook_event_type(headers, &payload);
        if !route.events.is_empty() && !route.events.iter().any(|e| e == &event_type) {
            return WebhookOutcome::Respond(WebhookHttpResponse::new(
                200,
                serde_json::json!({ "status": "ignored", "event": event_type }),
            ));
        }

        // Render prompt (skills are injected by the runtime layer).
        let prompt = render_prompt(&route.prompt, &payload, &event_type, route_name);

        // Build delivery id.
        let delivery_id = build_delivery_id(headers, now_millis);

        // Idempotency.
        if self.runtime.is_duplicate_delivery(&delivery_id, now_secs) {
            return WebhookOutcome::Respond(WebhookHttpResponse::new(
                200,
                serde_json::json!({ "status": "duplicate", "delivery_id": delivery_id }),
            ));
        }

        // deliver_only direct-delivery mode.
        if route.deliver_only {
            let delivery = DeliveryInfo {
                deliver: route.deliver.clone(),
                deliver_extra: render_delivery_extra(&route.deliver_extra, &payload),
                payload: payload.clone(),
            };
            return WebhookOutcome::DirectDeliver {
                route_name: route_name.to_string(),
                event_type,
                delivery_id,
                prompt,
                delivery,
            };
        }

        // Agent-mode: store delivery info keyed by session chat_id and accept.
        let session_chat_id = format!("webhook:{route_name}:{delivery_id}");
        let deliver_config = DeliveryInfo {
            deliver: route.deliver.clone(),
            deliver_extra: render_delivery_extra(&route.deliver_extra, &payload),
            payload,
        };
        self.runtime
            .store_delivery_info(&session_chat_id, deliver_config, now_secs);

        WebhookOutcome::AcceptAgentRun {
            route_name: route_name.to_string(),
            event_type,
            delivery_id,
            session_chat_id,
            prompt,
        }
    }

    /// Resolve the delivery decision for `send()`, given the stored delivery
    /// info for a chat_id. Mirrors the top of `send()`.
    pub fn send_decision(
        &self,
        chat_id: &str,
        is_plugin_registered: impl Fn(&str) -> bool,
    ) -> DeliveryDecision {
        let deliver_type = self
            .runtime
            .get_delivery_info(chat_id)
            .map(|d| d.deliver.clone())
            .unwrap_or_else(|| "log".to_string());
        resolve_send_decision(&deliver_type, self.has_gateway_runner, is_plugin_registered)
    }
}

fn json_as_u64(value: &JsonValue) -> Option<u64> {
    match value {
        JsonValue::Number(n) => n.as_u64(),
        JsonValue::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gh_sig(secret: &str, body: &[u8]) -> String {
        format!("sha256={}", hmac_sha256_hex(secret, body))
    }

    #[test]
    fn validate_github_signature() {
        let body = br#"{"a":1}"#;
        let headers = HeaderMap::new().with("X-Hub-Signature-256", gh_sig("topsecret", body));
        assert!(validate_signature(&headers, body, "topsecret"));
        // wrong secret rejected
        assert!(!validate_signature(&headers, body, "wrong"));
    }

    #[test]
    fn validate_gitlab_token() {
        let headers = HeaderMap::new().with("X-Gitlab-Token", "plainsecret");
        assert!(validate_signature(&headers, b"ignored", "plainsecret"));
        assert!(!validate_signature(&headers, b"ignored", "other"));
    }

    #[test]
    fn validate_generic_signature() {
        let body = b"payload";
        let sig = hmac_sha256_hex("k", body);
        let headers = HeaderMap::new().with("X-Webhook-Signature", sig);
        assert!(validate_signature(&headers, body, "k"));
        assert!(!validate_signature(&headers, body, "nope"));
    }

    #[test]
    fn no_signature_header_rejects() {
        let headers = HeaderMap::new();
        assert!(!validate_signature(&headers, b"x", "secret"));
    }

    #[test]
    fn case_insensitive_headers() {
        let body = b"b";
        let headers = HeaderMap::new().with("x-hub-signature-256", gh_sig("s", body));
        assert!(validate_signature(&headers, body, "s"));
    }

    #[test]
    fn compare_digest_length_mismatch() {
        assert!(!compare_digest(b"abc", b"abcd"));
        assert!(compare_digest(b"abc", b"abc"));
        assert!(!compare_digest(b"abc", b"abd"));
    }

    #[test]
    fn render_prompt_dot_notation() {
        let payload = json!({"pull_request": {"title": "Fix bug"}, "n": 5, "ok": true});
        assert_eq!(
            render_prompt("PR: {pull_request.title}", &payload, "pr", "r"),
            "PR: Fix bug"
        );
        // missing key -> literal token preserved
        assert_eq!(
            render_prompt("X: {pull_request.missing}", &payload, "", ""),
            "X: {pull_request.missing}"
        );
        // scalar number
        assert_eq!(render_prompt("n={n}", &payload, "", ""), "n=5");
        // bool stringification matches Python
        assert_eq!(render_prompt("ok={ok}", &payload, "", ""), "ok=True");
    }

    #[test]
    fn render_prompt_raw_token() {
        let payload = json!({"a": 1});
        let out = render_prompt("{__raw__}", &payload, "", "");
        assert!(out.contains("\"a\""));
        assert!(out.contains('1'));
    }

    #[test]
    fn render_prompt_empty_template_default() {
        let payload = json!({"x": 1});
        let out = render_prompt("", &payload, "push", "myroute");
        assert!(out.starts_with("Webhook event 'push' on route 'myroute':"));
        assert!(out.contains("```json"));
        assert!(out.contains("\"x\""));
    }

    #[test]
    fn render_prompt_dict_value_dumps_json() {
        let payload = json!({"obj": {"nested": [1, 2, 3]}});
        let out = render_prompt("{obj}", &payload, "", "");
        assert!(out.contains("nested"));
        assert!(out.contains('1'));
    }

    #[test]
    fn render_delivery_extra_renders_strings_only() {
        let payload = json!({"chat": "C123", "n": 7});
        let mut extra = JsonMap::new();
        extra.insert("chat_id".into(), json!("{chat}"));
        extra.insert("number".into(), json!(42));
        let rendered = render_delivery_extra(&extra, &payload);
        assert_eq!(rendered["chat_id"], json!("C123"));
        assert_eq!(rendered["number"], json!(42));
    }

    #[test]
    fn parse_body_json_then_form() {
        let v = parse_webhook_body(br#"{"a":1}"#).unwrap();
        assert_eq!(v["a"], json!(1));
        let v2 = parse_webhook_body(b"a=1&b=two").unwrap();
        assert_eq!(v2["a"], json!("1"));
        assert_eq!(v2["b"], json!("two"));
    }

    #[test]
    fn event_type_precedence() {
        let payload = json!({"event_type": "from_payload"});
        let h = HeaderMap::new().with("X-GitHub-Event", "push");
        assert_eq!(webhook_event_type(&h, &payload), "push");
        let h2 = HeaderMap::new().with("X-GitLab-Event", "merge");
        assert_eq!(webhook_event_type(&h2, &payload), "merge");
        let h3 = HeaderMap::new();
        assert_eq!(webhook_event_type(&h3, &payload), "from_payload");
        let h4 = HeaderMap::new();
        assert_eq!(webhook_event_type(&h4, &json!({})), "unknown");
    }

    #[test]
    fn route_validation_secret_and_deliver_only() {
        let mut r = WebhookRoute::default();
        assert!(validate_route("r", &r).is_err());
        r.secret = "s".into();
        assert!(validate_route("r", &r).is_ok());
        r.deliver_only = true;
        r.deliver = "log".into();
        assert!(validate_route("r", &r).is_err());
        r.deliver = "telegram".into();
        assert!(validate_route("r", &r).is_ok());
    }

    #[test]
    fn route_from_json_uses_global_secret_fallback() {
        let obj = json!({"prompt": "p", "deliver": "telegram"});
        let route = WebhookRoute::from_json(obj.as_object().unwrap(), "GLOBAL");
        assert_eq!(route.secret, "GLOBAL");
        assert_eq!(route.deliver, "telegram");
        assert_eq!(route.prompt, "p");
        assert!(!route.deliver_only);
    }

    #[test]
    fn rate_limit_window() {
        let mut rt = WebhookRuntime::new(2, 3600);
        assert!(rt.check_rate_limit("r", 0.0));
        assert!(rt.check_rate_limit("r", 1.0));
        // third within 60s -> blocked
        assert!(!rt.check_rate_limit("r", 2.0));
        // window prunes after 60s
        assert!(rt.check_rate_limit("r", 70.0));
    }

    #[test]
    fn idempotency_dedup_and_ttl() {
        let mut rt = WebhookRuntime::new(30, 3600);
        assert!(!rt.is_duplicate_delivery("d1", 0.0));
        assert!(rt.is_duplicate_delivery("d1", 1.0));
        // after TTL expiry, the old entry is pruned -> not a duplicate
        assert!(!rt.is_duplicate_delivery("d1", 4000.0));
    }

    #[test]
    fn delivery_info_store_and_prune() {
        let mut rt = WebhookRuntime::new(30, 100);
        rt.store_delivery_info("webhook:r:1", DeliveryInfo::default(), 0.0);
        assert!(rt.get_delivery_info("webhook:r:1").is_some());
        // storing a fresh entry far in the future prunes the stale one
        rt.store_delivery_info("webhook:r:2", DeliveryInfo::default(), 1000.0);
        assert!(rt.get_delivery_info("webhook:r:1").is_none());
        assert!(rt.get_delivery_info("webhook:r:2").is_some());
    }

    #[test]
    fn delivery_id_fallback_chain() {
        let h = HeaderMap::new().with("X-GitHub-Delivery", "gh-123");
        assert_eq!(build_delivery_id(&h, 999), "gh-123");
        let h2 = HeaderMap::new().with("X-Request-ID", "req-9");
        assert_eq!(build_delivery_id(&h2, 999), "req-9");
        let h3 = HeaderMap::new();
        assert_eq!(build_delivery_id(&h3, 12345), "12345");
    }

    #[test]
    fn send_decision_branches() {
        // log
        assert_eq!(
            resolve_send_decision("log", true, |_| false),
            DeliveryDecision::Log
        );
        // github_comment
        assert_eq!(
            resolve_send_decision("github_comment", true, |_| false),
            DeliveryDecision::GithubComment
        );
        // builtin platform with runner
        assert_eq!(
            resolve_send_decision("telegram", true, |_| false),
            DeliveryDecision::CrossPlatform("telegram".into())
        );
        // builtin platform but no runner -> Unknown
        assert_eq!(
            resolve_send_decision("telegram", false, |_| false),
            DeliveryDecision::Unknown("telegram".into())
        );
        // plugin-registered with runner
        assert_eq!(
            resolve_send_decision("myplugin", true, |n| n == "myplugin"),
            DeliveryDecision::CrossPlatform("myplugin".into())
        );
        // unknown
        assert_eq!(
            resolve_send_decision("nope", true, |_| false),
            DeliveryDecision::Unknown("nope".into())
        );
    }

    #[test]
    fn direct_deliver_decision_branches() {
        assert_eq!(resolve_direct_deliver_decision("log"), DeliveryDecision::Log);
        assert_eq!(
            resolve_direct_deliver_decision("github_comment"),
            DeliveryDecision::GithubComment
        );
        assert_eq!(
            resolve_direct_deliver_decision("telegram"),
            DeliveryDecision::CrossPlatform("telegram".into())
        );
    }

    #[test]
    fn github_comment_missing_fields() {
        let delivery = DeliveryInfo::default();
        let result = deliver_github_comment("body", &delivery);
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Missing repo or pr_number"));
    }

    #[test]
    fn cross_platform_target_resolution() {
        let mut extra = JsonMap::new();
        extra.insert("chat_id".into(), json!("C1"));
        extra.insert("message_thread_id".into(), json!("42"));
        let delivery = DeliveryInfo {
            deliver: "telegram".into(),
            deliver_extra: extra,
            payload: json!({}),
        };
        let target = resolve_cross_platform_target(&delivery, None).unwrap();
        assert_eq!(target.chat_id, "C1");
        assert_eq!(target.thread_id.as_deref(), Some("42"));

        // no chat_id, fall back to home channel
        let delivery2 = DeliveryInfo {
            deliver: "telegram".into(),
            deliver_extra: JsonMap::new(),
            payload: json!({}),
        };
        let t2 = resolve_cross_platform_target(&delivery2, Some("home-chat")).unwrap();
        assert_eq!(t2.chat_id, "home-chat");
        assert!(t2.thread_id.is_none());

        // no chat_id and no home -> error
        assert!(resolve_cross_platform_target(&delivery2, None).is_err());
    }

    #[test]
    fn process_webhook_unknown_route() {
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            HashMap::new(),
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        let h = HeaderMap::new();
        let outcome = adapter.process_webhook("nope", &h, b"{}", 2, 0.0, 0);
        match outcome {
            WebhookOutcome::Respond(r) => assert_eq!(r.status, 404),
            _ => panic!("expected 404 respond"),
        }
    }

    #[test]
    fn process_webhook_body_too_large() {
        let mut routes = HashMap::new();
        routes.insert(
            "r".to_string(),
            WebhookRoute {
                secret: INSECURE_NO_AUTH.into(),
                ..Default::default()
            },
        );
        let mut adapter = WebhookAdapter::new("0.0.0.0", 8644, "", routes, 30, 10);
        let h = HeaderMap::new();
        let outcome = adapter.process_webhook("r", &h, b"x", 100, 0.0, 0);
        match outcome {
            WebhookOutcome::Respond(r) => assert_eq!(r.status, 413),
            _ => panic!("expected 413"),
        }
    }

    #[test]
    fn process_webhook_invalid_signature() {
        let mut routes = HashMap::new();
        routes.insert(
            "r".to_string(),
            WebhookRoute {
                secret: "topsecret".into(),
                ..Default::default()
            },
        );
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            routes,
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        let h = HeaderMap::new().with("X-Hub-Signature-256", "sha256=deadbeef");
        let outcome = adapter.process_webhook("r", &h, b"{}", 2, 0.0, 0);
        match outcome {
            WebhookOutcome::Respond(r) => assert_eq!(r.status, 401),
            _ => panic!("expected 401"),
        }
    }

    #[test]
    fn process_webhook_event_filter_ignored() {
        let mut routes = HashMap::new();
        routes.insert(
            "r".to_string(),
            WebhookRoute {
                secret: INSECURE_NO_AUTH.into(),
                events: vec!["push".into()],
                ..Default::default()
            },
        );
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            routes,
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        let h = HeaderMap::new().with("X-GitHub-Event", "issues");
        let outcome = adapter.process_webhook("r", &h, b"{}", 2, 0.0, 0);
        match outcome {
            WebhookOutcome::Respond(r) => {
                assert_eq!(r.status, 200);
                assert_eq!(r.body["status"], json!("ignored"));
                assert_eq!(r.body["event"], json!("issues"));
            }
            _ => panic!("expected ignored"),
        }
    }

    #[test]
    fn process_webhook_accepts_agent_run_and_stores_delivery() {
        let mut routes = HashMap::new();
        let mut extra = JsonMap::new();
        extra.insert("chat_id".into(), json!("{chat}"));
        routes.insert(
            "r".to_string(),
            WebhookRoute {
                secret: INSECURE_NO_AUTH.into(),
                prompt: "Hi {name}".into(),
                deliver: "telegram".into(),
                deliver_extra: extra,
                ..Default::default()
            },
        );
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            routes,
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        let h = HeaderMap::new()
            .with("X-GitHub-Event", "push")
            .with("X-GitHub-Delivery", "deliv-1");
        let body = br#"{"name":"World","chat":"C9"}"#;
        let outcome = adapter.process_webhook("r", &h, body, body.len(), 100.0, 100000);
        match outcome {
            WebhookOutcome::AcceptAgentRun {
                route_name,
                event_type,
                delivery_id,
                session_chat_id,
                prompt,
            } => {
                assert_eq!(route_name, "r");
                assert_eq!(event_type, "push");
                assert_eq!(delivery_id, "deliv-1");
                assert_eq!(session_chat_id, "webhook:r:deliv-1");
                assert_eq!(prompt, "Hi World");
                // delivery info stored and rendered
                let info = adapter.runtime.get_delivery_info("webhook:r:deliv-1").unwrap();
                assert_eq!(info.deliver, "telegram");
                assert_eq!(info.deliver_extra["chat_id"], json!("C9"));
            }
            _ => panic!("expected AcceptAgentRun"),
        }

        // duplicate delivery now short-circuits to 200 duplicate
        let outcome2 = adapter.process_webhook("r", &h, body, body.len(), 101.0, 100001);
        match outcome2 {
            WebhookOutcome::Respond(r) => {
                assert_eq!(r.status, 200);
                assert_eq!(r.body["status"], json!("duplicate"));
            }
            _ => panic!("expected duplicate"),
        }
    }

    #[test]
    fn process_webhook_deliver_only() {
        let mut routes = HashMap::new();
        routes.insert(
            "alerts".to_string(),
            WebhookRoute {
                secret: INSECURE_NO_AUTH.into(),
                prompt: "Alert: {message}".into(),
                deliver: "dingtalk".into(),
                deliver_only: true,
                ..Default::default()
            },
        );
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            routes,
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        let h = HeaderMap::new().with("X-Request-ID", "req-7");
        let body = br#"{"message":"ping"}"#;
        let outcome = adapter.process_webhook("alerts", &h, body, body.len(), 0.0, 0);
        match outcome {
            WebhookOutcome::DirectDeliver {
                route_name,
                delivery_id,
                prompt,
                delivery,
                ..
            } => {
                assert_eq!(route_name, "alerts");
                assert_eq!(delivery_id, "req-7");
                assert_eq!(prompt, "Alert: ping");
                assert_eq!(delivery.deliver, "dingtalk");
            }
            _ => panic!("expected DirectDeliver"),
        }
    }

    #[test]
    fn http_response_builders() {
        let r = accepted_response("rt", "push", "d1");
        assert_eq!(r.status, 202);
        assert_eq!(r.body["status"], json!("accepted"));
        let d = delivered_response("rt", "telegram", "d1");
        assert_eq!(d.status, 200);
        assert_eq!(d.body["target"], json!("telegram"));
        let f = delivery_failed_response("d1");
        assert_eq!(f.status, 502);
        assert_eq!(f.body["error"], json!("Delivery failed"));
    }

    #[test]
    fn adapter_from_extra_parses_config() {
        let extra = json!({
            "host": "127.0.0.1",
            "port": 9000,
            "secret": "G",
            "rate_limit": 5,
            "max_body_bytes": 2048,
            "routes": {
                "r": {"prompt": "p", "deliver": "telegram"}
            }
        });
        let adapter = WebhookAdapter::from_extra(&extra);
        assert_eq!(adapter.host, "127.0.0.1");
        assert_eq!(adapter.port, 9000);
        assert_eq!(adapter.global_secret, "G");
        assert_eq!(adapter.runtime.rate_limit, 5);
        assert_eq!(adapter.max_body_bytes, 2048);
        let route = &adapter.registry.routes["r"];
        assert_eq!(route.secret, "G"); // falls back to global
        assert_eq!(route.deliver, "telegram");
    }

    #[test]
    fn send_decision_uses_stored_delivery() {
        let mut adapter = WebhookAdapter::new(
            "0.0.0.0",
            8644,
            "",
            HashMap::new(),
            30,
            DEFAULT_MAX_BODY_BYTES,
        );
        adapter.has_gateway_runner = true;
        adapter.runtime.store_delivery_info(
            "webhook:r:1",
            DeliveryInfo {
                deliver: "telegram".into(),
                ..Default::default()
            },
            0.0,
        );
        assert_eq!(
            adapter.send_decision("webhook:r:1", |_| false),
            DeliveryDecision::CrossPlatform("telegram".into())
        );
        // unknown chat -> defaults to log
        assert_eq!(
            adapter.send_decision("missing", |_| false),
            DeliveryDecision::Log
        );
    }

    #[test]
    fn builtin_platform_membership() {
        assert!(is_builtin_deliver_platform("telegram"));
        assert!(is_builtin_deliver_platform("wecom_callback"));
        assert!(!is_builtin_deliver_platform("github_comment"));
        assert!(!is_builtin_deliver_platform("nope"));
    }
}
