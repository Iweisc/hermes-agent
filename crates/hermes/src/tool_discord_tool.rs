//! Discord server introspection and management tool.
//!
//! Native Rust port of `tools/discord_tool.py`.
//!
//! Provides the agent with the ability to interact with Discord servers when
//! running on the Discord gateway. Uses the Discord REST API directly with the
//! bot token -- no dependency on the gateway adapter's client.
//!
//! The schema exposed to the model is filtered by two gates:
//!
//! 1. Privileged intents detected from `GET /applications/@me` at schema build
//!    time. Actions that require an intent the bot doesn't have
//!    (`search_members` / `member_info` -> GUILD_MEMBERS intent) are hidden.
//!    `fetch_messages` is kept regardless of MESSAGE_CONTENT intent, but its
//!    description is annotated when the intent is missing.
//!
//! 2. User config allowlist at `discord.server_actions`. If the user sets a
//!    comma-separated list (or YAML list) of action names, only those appear in
//!    the schema. Empty/unset means all intent-available actions are exposed.
//!
//! Per-guild permissions (MANAGE_ROLES etc.) are NOT pre-checked -- Discord
//! returns a 403 at call time and [`enrich_403`] maps it to actionable guidance
//! the model can relay to the user.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

/// Base URL for the Discord REST API (v10).
pub const DISCORD_API_BASE: &str = "https://discord.com/api/v10";

/// User-Agent string sent with every request (matches the Python original).
pub const DISCORD_USER_AGENT: &str =
    "Hermes-Agent (https://github.com/NousResearch/hermes-agent)";

// Application flag bits (from GET /applications/@me -> "flags").
// Source: https://discord.com/developers/docs/resources/application#application-object-application-flags
const FLAG_GATEWAY_GUILD_MEMBERS: i64 = 1 << 14;
const FLAG_GATEWAY_GUILD_MEMBERS_LIMITED: i64 = 1 << 15;
const FLAG_GATEWAY_MESSAGE_CONTENT: i64 = 1 << 18;
const FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED: i64 = 1 << 19;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Raised when a Discord API call fails.
#[derive(Debug, Clone)]
pub struct DiscordApiError {
    /// HTTP status code returned by Discord.
    pub status: u16,
    /// Raw response body (best-effort decoded as UTF-8).
    pub body: String,
}

impl DiscordApiError {
    /// Construct a new error from a status code and body.
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
        }
    }
}

impl std::fmt::Display for DiscordApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Discord API error {}: {}", self.status, self.body)
    }
}

impl std::error::Error for DiscordApiError {}

/// Error type covering both HTTP transport failures and Discord API errors.
#[derive(Debug)]
pub enum DiscordError {
    /// A structured Discord API failure (HTTP status >= 400 with a body).
    Api(DiscordApiError),
    /// A transport/parsing failure (network error, bad JSON, etc.).
    Transport(String),
}

impl std::fmt::Display for DiscordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscordError::Api(e) => write!(f, "{e}"),
            DiscordError::Transport(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for DiscordError {}

// ---------------------------------------------------------------------------
// Token resolution
// ---------------------------------------------------------------------------

/// Resolve the Discord bot token from the environment.
///
/// Mirrors `_get_bot_token`: reads `DISCORD_BOT_TOKEN`, strips whitespace, and
/// returns `None` when empty/unset.
pub fn get_bot_token() -> Option<String> {
    match std::env::var("DISCORD_BOT_TOKEN") {
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// HTTP transport abstraction
// ---------------------------------------------------------------------------

/// Trait abstracting the Discord REST transport, so unit tests can inject a
/// fake client without performing real network I/O.
///
/// `params` are query-string parameters; `body` is an optional JSON body. A
/// `204 No Content` response is represented as `Ok(Value::Null)`.
pub trait DiscordTransport {
    /// Perform a request against the Discord REST API.
    fn request(
        &self,
        method: &str,
        path: &str,
        token: &str,
        params: Option<&[(String, String)]>,
        body: Option<&Value>,
        timeout_secs: u64,
    ) -> Result<Value, DiscordError>;
}

/// Build the full URL for a request, appending an url-encoded query string.
fn build_url(path: &str, params: Option<&[(String, String)]>) -> String {
    let mut url = format!("{DISCORD_API_BASE}{path}");
    if let Some(p) = params {
        if !p.is_empty() {
            let qs: Vec<String> = p
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}={}",
                        urlencode(k),
                        urlencode(v)
                    )
                })
                .collect();
            url.push('?');
            url.push_str(&qs.join("&"));
        }
    }
    url
}

/// Minimal application/x-www-form-urlencoded percent-encoding matching
/// `urllib.parse.urlencode` (quote_via=quote_plus) closely enough for the
/// query values used here.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Real `reqwest::blocking` transport implementing [`DiscordTransport`].
#[derive(Debug, Default, Clone)]
pub struct ReqwestDiscordTransport;

impl DiscordTransport for ReqwestDiscordTransport {
    fn request(
        &self,
        method: &str,
        path: &str,
        token: &str,
        params: Option<&[(String, String)]>,
        body: Option<&Value>,
        timeout_secs: u64,
    ) -> Result<Value, DiscordError> {
        let url = build_url(path, params);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| DiscordError::Transport(e.to_string()))?;

        let m = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| DiscordError::Transport(e.to_string()))?;

        let mut req = client
            .request(m, &url)
            .header("Authorization", format!("Bot {token}"))
            .header("Content-Type", "application/json")
            .header("User-Agent", DISCORD_USER_AGENT);

        if let Some(b) = body {
            // Serialize explicitly to match `json.dumps(body).encode("utf-8")`.
            let data = serde_json::to_vec(b)
                .map_err(|e| DiscordError::Transport(e.to_string()))?;
            req = req.body(data);
        }

        let resp = req
            .send()
            .map_err(|e| DiscordError::Transport(e.to_string()))?;

        let status = resp.status().as_u16();

        if status == 204 {
            return Ok(Value::Null);
        }

        if status >= 400 {
            let body = resp.text().unwrap_or_default();
            return Err(DiscordError::Api(DiscordApiError::new(status, body)));
        }

        // 2xx (other than 204): parse JSON body.
        let text = resp
            .text()
            .map_err(|e| DiscordError::Transport(e.to_string()))?;
        if text.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| DiscordError::Transport(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Channel type mapping
// ---------------------------------------------------------------------------

/// Map a Discord channel type id to a human-readable name.
///
/// Mirrors `_channel_type_name` (`_CHANNEL_TYPE_NAMES` table + unknown fallback).
pub fn channel_type_name(type_id: i64) -> String {
    match type_id {
        0 => "text".to_string(),
        2 => "voice".to_string(),
        4 => "category".to_string(),
        5 => "announcement".to_string(),
        10 => "announcement_thread".to_string(),
        11 => "public_thread".to_string(),
        12 => "private_thread".to_string(),
        13 => "stage".to_string(),
        15 => "forum".to_string(),
        16 => "media".to_string(),
        other => format!("unknown({other})"),
    }
}

// ---------------------------------------------------------------------------
// Capability detection (application intents)
// ---------------------------------------------------------------------------

/// Bot app-wide capabilities, mirroring the dict returned by
/// `_detect_capabilities`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// GUILD_MEMBERS intent is enabled.
    pub has_members_intent: bool,
    /// MESSAGE_CONTENT intent is enabled.
    pub has_message_content: bool,
    /// Detection actually succeeded (false => expose everything and let
    /// runtime errors handle it).
    pub detected: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            has_members_intent: true,
            has_message_content: true,
            detected: false,
        }
    }
}

// Module-level cache so the app/me endpoint is hit at most once per token.
static CAPABILITY_CACHE: Mutex<Option<HashMap<String, Capabilities>>> = Mutex::new(None);

/// Detect the bot's app-wide capabilities via `GET /applications/@me`.
///
/// Cached per token in a module-global. Pass `force=true` to re-fetch.
/// Detection is best-effort: any error yields the default (all-true,
/// `detected=false`) capabilities while still caching the result.
pub fn detect_capabilities<T: DiscordTransport>(
    transport: &T,
    token: &str,
    force: bool,
) -> Capabilities {
    {
        let guard = CAPABILITY_CACHE.lock().unwrap();
        if !force {
            if let Some(map) = guard.as_ref() {
                if let Some(c) = map.get(token) {
                    return c.clone();
                }
            }
        }
    }

    let mut caps = Capabilities::default();

    match transport.request("GET", "/applications/@me", token, None, None, 5) {
        Ok(app) => {
            let flags = app
                .get("flags")
                .and_then(value_to_i64)
                .unwrap_or(0);
            caps.has_members_intent =
                (flags & (FLAG_GATEWAY_GUILD_MEMBERS | FLAG_GATEWAY_GUILD_MEMBERS_LIMITED)) != 0;
            caps.has_message_content = (flags
                & (FLAG_GATEWAY_MESSAGE_CONTENT | FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED))
                != 0;
            caps.detected = true;
        }
        Err(exc) => {
            log::info!(
                "Discord capability detection failed ({exc}); exposing all actions."
            );
        }
    }

    {
        let mut guard = CAPABILITY_CACHE.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        map.insert(token.to_string(), caps.clone());
    }
    caps
}

/// Test hook: clear the detection cache (mirrors `_reset_capability_cache`).
pub fn reset_capability_cache() {
    let mut guard = CAPABILITY_CACHE.lock().unwrap();
    *guard = None;
}

/// Interpret a JSON value as an i64, tolerating ints, floats, and numeric
/// strings (matching Python's `int(app.get("flags", 0) or 0)`).
fn value_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(*b as i64),
        Value::Null => Some(0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Small JSON access helpers
// ---------------------------------------------------------------------------

fn get_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

/// Return `obj[key]` as an owned `Value`, or `Value::Null` if absent.
fn opt(v: &Value, key: &str) -> Value {
    v.get(key).cloned().unwrap_or(Value::Null)
}

/// Return `obj[key]` as a `Value`, falling back to `default` if absent/null.
fn opt_or(v: &Value, key: &str, default: Value) -> Value {
    match v.get(key) {
        Some(Value::Null) | None => default,
        Some(other) => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Action implementations
// ---------------------------------------------------------------------------

/// Parameters bundle for action implementations, mirroring the keyword
/// arguments passed to the Python action functions.
#[derive(Debug, Clone, Default)]
pub struct ActionParams {
    pub guild_id: String,
    pub channel_id: String,
    pub user_id: String,
    pub role_id: String,
    pub message_id: String,
    pub query: String,
    pub name: String,
    pub limit: i64,
    pub before: String,
    pub after: String,
    pub auto_archive_duration: i64,
}

impl ActionParams {
    /// Defaults matching `_HANDLER_DEFAULTS` (limit=50, auto_archive=1440).
    pub fn new() -> Self {
        Self {
            limit: 50,
            auto_archive_duration: 1440,
            ..Default::default()
        }
    }
}

/// `_list_guilds`: list all guilds the bot is a member of.
fn list_guilds<T: DiscordTransport>(t: &T, token: &str, _p: &ActionParams) -> Result<String, DiscordError> {
    let guilds = t.request("GET", "/users/@me/guilds", token, None, None, 15)?;
    let mut result: Vec<Value> = Vec::new();
    if let Some(arr) = guilds.as_array() {
        for g in arr {
            result.push(json!({
                "id": opt(g, "id"),
                "name": opt(g, "name"),
                "icon": opt(g, "icon"),
                "owner": opt_or(g, "owner", json!(false)),
                "permissions": opt(g, "permissions"),
            }));
        }
    }
    let count = result.len();
    Ok(json!({"guilds": result, "count": count}).to_string())
}

/// `_server_info`: detailed information about a guild.
fn server_info<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/guilds/{}", p.guild_id);
    let params = vec![("with_counts".to_string(), "true".to_string())];
    let g = t.request("GET", &path, token, Some(&params), None, 15)?;
    Ok(json!({
        "id": opt(&g, "id"),
        "name": opt(&g, "name"),
        "description": opt(&g, "description"),
        "icon": opt(&g, "icon"),
        "owner_id": opt(&g, "owner_id"),
        "member_count": opt(&g, "approximate_member_count"),
        "online_count": opt(&g, "approximate_presence_count"),
        "features": opt_or(&g, "features", json!([])),
        "premium_tier": opt(&g, "premium_tier"),
        "premium_subscription_count": opt(&g, "premium_subscription_count"),
        "verification_level": opt(&g, "verification_level"),
    })
    .to_string())
}

/// `_list_channels`: all channels in a guild, organized by category.
fn list_channels<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/guilds/{}/channels", p.guild_id);
    let channels = t.request("GET", &path, token, None, None, 15)?;
    let empty: Vec<Value> = Vec::new();
    let channels = channels.as_array().unwrap_or(&empty);

    // category id -> (name, position, channels)
    struct Cat {
        id: String,
        name: String,
        position: i64,
        channels: Vec<Value>,
    }
    let mut categories: HashMap<String, Cat> = HashMap::new();
    let mut cat_order: Vec<String> = Vec::new();
    let mut uncategorized: Vec<Value> = Vec::new();

    // First pass: collect categories.
    for ch in channels {
        if ch.get("type").and_then(value_to_i64) == Some(4) {
            let id = get_str(ch, "id").unwrap_or_default().to_string();
            let name = get_str(ch, "name").unwrap_or_default().to_string();
            let position = ch.get("position").and_then(value_to_i64).unwrap_or(0);
            cat_order.push(id.clone());
            categories.insert(
                id.clone(),
                Cat {
                    id,
                    name,
                    position,
                    channels: Vec::new(),
                },
            );
        }
    }

    // Second pass: assign channels to categories.
    for ch in channels {
        let ty = ch.get("type").and_then(value_to_i64).unwrap_or(-1);
        if ty == 4 {
            continue;
        }
        let position = ch.get("position").and_then(value_to_i64).unwrap_or(0);
        let entry = json!({
            "id": opt(ch, "id"),
            "name": opt_or(ch, "name", json!("")),
            "type": channel_type_name(ty),
            "position": position,
            "topic": opt(ch, "topic"),
            "nsfw": opt_or(ch, "nsfw", json!(false)),
        });
        let parent = get_str(ch, "parent_id");
        match parent {
            Some(pid) if categories.contains_key(pid) => {
                categories.get_mut(pid).unwrap().channels.push(entry);
            }
            _ => uncategorized.push(entry),
        }
    }

    // Sort categories by position; sort each category's channels by position.
    let mut sorted_cats: Vec<&mut Cat> = categories.values_mut().collect();
    sorted_cats.sort_by_key(|c| c.position);
    for cat in sorted_cats.iter_mut() {
        cat.channels.sort_by_key(channel_position);
    }
    uncategorized.sort_by_key(channel_position);

    let mut result: Vec<Value> = Vec::new();
    if !uncategorized.is_empty() {
        result.push(json!({"category": Value::Null, "channels": uncategorized}));
    }
    for cat in sorted_cats {
        result.push(json!({
            "category": {"id": cat.id, "name": cat.name},
            "channels": cat.channels,
        }));
    }

    let total: usize = result
        .iter()
        .map(|g| g.get("channels").and_then(|c| c.as_array()).map_or(0, |a| a.len()))
        .sum();

    Ok(json!({"channel_groups": result, "total_channels": total}).to_string())
}

fn channel_position(v: &Value) -> i64 {
    v.get("position").and_then(value_to_i64).unwrap_or(0)
}

/// `_channel_info`: detailed info about a specific channel.
fn channel_info<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/channels/{}", p.channel_id);
    let ch = t.request("GET", &path, token, None, None, 15)?;
    let ty = ch.get("type").and_then(value_to_i64).unwrap_or(-1);
    Ok(json!({
        "id": opt(&ch, "id"),
        "name": opt(&ch, "name"),
        "type": channel_type_name(ty),
        "guild_id": opt(&ch, "guild_id"),
        "topic": opt(&ch, "topic"),
        "nsfw": opt_or(&ch, "nsfw", json!(false)),
        "position": opt(&ch, "position"),
        "parent_id": opt(&ch, "parent_id"),
        "rate_limit_per_user": opt_or(&ch, "rate_limit_per_user", json!(0)),
        "last_message_id": opt(&ch, "last_message_id"),
    })
    .to_string())
}

/// `_list_roles`: all roles in a guild, sorted by position descending.
fn list_roles<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/guilds/{}/roles", p.guild_id);
    let roles = t.request("GET", &path, token, None, None, 15)?;
    let empty: Vec<Value> = Vec::new();
    let mut roles: Vec<&Value> = roles.as_array().unwrap_or(&empty).iter().collect();
    // sorted(..., key=position, reverse=True). Python sort is stable.
    roles.sort_by(|a, b| {
        let pa = a.get("position").and_then(value_to_i64).unwrap_or(0);
        let pb = b.get("position").and_then(value_to_i64).unwrap_or(0);
        pb.cmp(&pa)
    });

    let mut result: Vec<Value> = Vec::new();
    for r in roles {
        // color: f"#{color:06x}" if color (truthy) else None
        let color_val = r.get("color").and_then(value_to_i64).unwrap_or(0);
        let color = if color_val != 0 {
            json!(format!("#{color_val:06x}"))
        } else {
            Value::Null
        };
        result.push(json!({
            "id": opt(r, "id"),
            "name": opt(r, "name"),
            "color": color,
            "position": opt_or(r, "position", json!(0)),
            "mentionable": opt_or(r, "mentionable", json!(false)),
            "managed": opt_or(r, "managed", json!(false)),
            "member_count": opt(r, "member_count"),
            "hoist": opt_or(r, "hoist", json!(false)),
        }));
    }
    let count = result.len();
    Ok(json!({"roles": result, "count": count}).to_string())
}

/// `_member_info`: info about a specific guild member.
fn member_info<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/guilds/{}/members/{}", p.guild_id, p.user_id);
    let m = t.request("GET", &path, token, None, None, 15)?;
    let user = opt_or(&m, "user", json!({}));
    Ok(json!({
        "user_id": opt(&user, "id"),
        "username": opt(&user, "username"),
        "display_name": opt(&user, "global_name"),
        "nickname": opt(&m, "nick"),
        "avatar": opt(&user, "avatar"),
        "bot": opt_or(&user, "bot", json!(false)),
        "roles": opt_or(&m, "roles", json!([])),
        "joined_at": opt(&m, "joined_at"),
        "premium_since": opt(&m, "premium_since"),
    })
    .to_string())
}

/// `_search_members`: search for guild members by name prefix.
fn search_members<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    // limit already coerced to int upstream; clamp to <= 100.
    let limit = std::cmp::min(p.limit, 100);
    let params = vec![
        ("query".to_string(), p.query.clone()),
        ("limit".to_string(), limit.to_string()),
    ];
    let path = format!("/guilds/{}/members/search", p.guild_id);
    let members = t.request("GET", &path, token, Some(&params), None, 15)?;
    let mut result: Vec<Value> = Vec::new();
    if let Some(arr) = members.as_array() {
        for m in arr {
            let user = opt_or(m, "user", json!({}));
            result.push(json!({
                "user_id": opt(&user, "id"),
                "username": opt(&user, "username"),
                "display_name": opt(&user, "global_name"),
                "nickname": opt(m, "nick"),
                "bot": opt_or(&user, "bot", json!(false)),
                "roles": opt_or(m, "roles", json!([])),
            }));
        }
    }
    let count = result.len();
    Ok(json!({"members": result, "count": count}).to_string())
}

/// `_fetch_messages`: fetch recent messages from a channel.
fn fetch_messages<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let limit = std::cmp::min(p.limit, 100);
    let mut params = vec![("limit".to_string(), limit.to_string())];
    if !p.before.is_empty() {
        params.push(("before".to_string(), p.before.clone()));
    }
    if !p.after.is_empty() {
        params.push(("after".to_string(), p.after.clone()));
    }
    let path = format!("/channels/{}/messages", p.channel_id);
    let messages = t.request("GET", &path, token, Some(&params), None, 15)?;
    let mut result: Vec<Value> = Vec::new();
    if let Some(arr) = messages.as_array() {
        for msg in arr {
            let author = opt_or(msg, "author", json!({}));
            let attachments: Vec<Value> = msg
                .get("attachments")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|a| {
                            json!({
                                "filename": opt(a, "filename"),
                                "url": opt(a, "url"),
                                "size": opt(a, "size"),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            // reactions: [...] if msg.get("reactions") else []
            let reactions: Vec<Value> = match msg.get("reactions") {
                Some(r) if is_truthy(r) => r
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|rr| {
                                let emoji_name = rr
                                    .get("emoji")
                                    .map(|e| opt(e, "name"))
                                    .unwrap_or(Value::Null);
                                json!({
                                    "emoji": emoji_name,
                                    "count": opt_or(rr, "count", json!(0)),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            result.push(json!({
                "id": opt(msg, "id"),
                "content": opt_or(msg, "content", json!("")),
                "author": {
                    "id": opt(&author, "id"),
                    "username": opt(&author, "username"),
                    "display_name": opt(&author, "global_name"),
                    "bot": opt_or(&author, "bot", json!(false)),
                },
                "timestamp": opt(msg, "timestamp"),
                "edited_timestamp": opt(msg, "edited_timestamp"),
                "attachments": attachments,
                "reactions": reactions,
                "pinned": opt_or(msg, "pinned", json!(false)),
            }));
        }
    }
    let count = result.len();
    Ok(json!({"messages": result, "count": count}).to_string())
}

/// Python truthiness for the subset of JSON values relevant here.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `_list_pins`: pinned messages in a channel (content truncated to 200 chars).
fn list_pins<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/channels/{}/pins", p.channel_id);
    let messages = t.request("GET", &path, token, None, None, 15)?;
    let mut result: Vec<Value> = Vec::new();
    if let Some(arr) = messages.as_array() {
        for msg in arr {
            let author = opt_or(msg, "author", json!({}));
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            // Python slices by chars ([:200]).
            let truncated: String = content.chars().take(200).collect();
            result.push(json!({
                "id": opt(msg, "id"),
                "content": truncated,
                "author": opt(&author, "username"),
                "timestamp": opt(msg, "timestamp"),
            }));
        }
    }
    let count = result.len();
    Ok(json!({"pinned_messages": result, "count": count}).to_string())
}

/// `_pin_message`: pin a message in a channel.
fn pin_message<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/channels/{}/pins/{}", p.channel_id, p.message_id);
    t.request("PUT", &path, token, None, None, 15)?;
    Ok(json!({
        "success": true,
        "message": format!("Message {} pinned.", p.message_id),
    })
    .to_string())
}

/// `_unpin_message`: unpin a message from a channel.
fn unpin_message<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!("/channels/{}/pins/{}", p.channel_id, p.message_id);
    t.request("DELETE", &path, token, None, None, 15)?;
    Ok(json!({
        "success": true,
        "message": format!("Message {} unpinned.", p.message_id),
    })
    .to_string())
}

/// `_create_thread`: create a thread in a channel.
fn create_thread<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let (path, body) = if !p.message_id.is_empty() {
        // Create thread from an existing message.
        (
            format!("/channels/{}/messages/{}/threads", p.channel_id, p.message_id),
            json!({
                "name": p.name,
                "auto_archive_duration": p.auto_archive_duration,
            }),
        )
    } else {
        // Create a standalone thread (PUBLIC_THREAD = 11).
        (
            format!("/channels/{}/threads", p.channel_id),
            json!({
                "name": p.name,
                "auto_archive_duration": p.auto_archive_duration,
                "type": 11,
            }),
        )
    };
    let thread = t.request("POST", &path, token, None, Some(&body), 15)?;
    Ok(json!({
        "success": true,
        "thread_id": opt(&thread, "id"),
        "name": opt(&thread, "name"),
    })
    .to_string())
}

/// `_add_role`: add a role to a guild member.
fn add_role<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!(
        "/guilds/{}/members/{}/roles/{}",
        p.guild_id, p.user_id, p.role_id
    );
    t.request("PUT", &path, token, None, None, 15)?;
    Ok(json!({
        "success": true,
        "message": format!("Role {} added to user {}.", p.role_id, p.user_id),
    })
    .to_string())
}

/// `_remove_role`: remove a role from a guild member.
fn remove_role<T: DiscordTransport>(t: &T, token: &str, p: &ActionParams) -> Result<String, DiscordError> {
    let path = format!(
        "/guilds/{}/members/{}/roles/{}",
        p.guild_id, p.user_id, p.role_id
    );
    t.request("DELETE", &path, token, None, None, 15)?;
    Ok(json!({
        "success": true,
        "message": format!("Role {} removed from user {}.", p.role_id, p.user_id),
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// Action dispatch + metadata
// ---------------------------------------------------------------------------

/// Function pointer type for action implementations parameterised over a
/// concrete transport.
type ActionFn<T> = fn(&T, &str, &ActionParams) -> Result<String, DiscordError>;

/// Canonical ordered list of all action names (matches the insertion order of
/// the Python `_ACTIONS` dict).
pub const ALL_ACTIONS: &[&str] = &[
    "list_guilds",
    "server_info",
    "list_channels",
    "channel_info",
    "list_roles",
    "member_info",
    "search_members",
    "fetch_messages",
    "list_pins",
    "pin_message",
    "unpin_message",
    "create_thread",
    "add_role",
    "remove_role",
];

/// Core action names (exposed via the `discord` tool).
pub const CORE_ACTION_NAMES: &[&str] = &["fetch_messages", "search_members", "create_thread"];

/// Resolve an action implementation by name for a given transport.
fn action_fn<T: DiscordTransport>(name: &str) -> Option<ActionFn<T>> {
    let f: ActionFn<T> = match name {
        "list_guilds" => list_guilds,
        "server_info" => server_info,
        "list_channels" => list_channels,
        "channel_info" => channel_info,
        "list_roles" => list_roles,
        "member_info" => member_info,
        "search_members" => search_members,
        "fetch_messages" => fetch_messages,
        "list_pins" => list_pins,
        "pin_message" => pin_message,
        "unpin_message" => unpin_message,
        "create_thread" => create_thread,
        "add_role" => add_role,
        "remove_role" => remove_role,
        _ => return None,
    };
    Some(f)
}

/// True if `name` is a known action.
pub fn is_known_action(name: &str) -> bool {
    ALL_ACTIONS.contains(&name)
}

/// Core action set (mirrors `_CORE_ACTIONS.keys()`), preserving canonical order.
pub fn core_action_names() -> Vec<&'static str> {
    ALL_ACTIONS
        .iter()
        .copied()
        .filter(|n| CORE_ACTION_NAMES.contains(n))
        .collect()
}

/// Admin action set (mirrors `_ADMIN_ACTIONS.keys()`), preserving canonical order.
pub fn admin_action_names() -> Vec<&'static str> {
    ALL_ACTIONS
        .iter()
        .copied()
        .filter(|n| !CORE_ACTION_NAMES.contains(n))
        .collect()
}

/// Single-source-of-truth manifest: (action, signature, one-line description).
/// Mirrors `_ACTION_MANIFEST`.
pub const ACTION_MANIFEST: &[(&str, &str, &str)] = &[
    ("list_guilds", "()", "list servers the bot is in"),
    ("server_info", "(guild_id)", "server details + member counts"),
    ("list_channels", "(guild_id)", "all channels grouped by category"),
    ("channel_info", "(channel_id)", "single channel details"),
    ("list_roles", "(guild_id)", "roles sorted by position"),
    ("member_info", "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", "(guild_id, query)", "find members by name prefix"),
    (
        "fetch_messages",
        "(channel_id)",
        "recent messages; optional before/after snowflakes",
    ),
    ("list_pins", "(channel_id)", "pinned messages in a channel"),
    ("pin_message", "(channel_id, message_id)", "pin a message"),
    ("unpin_message", "(channel_id, message_id)", "unpin a message"),
    (
        "create_thread",
        "(channel_id, name)",
        "create a public thread; optional message_id anchor",
    ),
    ("add_role", "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", "(guild_id, user_id, role_id)", "remove a role"),
];

/// Actions that require the GUILD_MEMBERS privileged intent.
pub const INTENT_GATED_MEMBERS: &[&str] = &["member_info", "search_members"];

/// Per-action required params for runtime validation (mirrors `_REQUIRED_PARAMS`).
pub fn required_params(action: &str) -> &'static [&'static str] {
    match action {
        "server_info" => &["guild_id"],
        "list_channels" => &["guild_id"],
        "list_roles" => &["guild_id"],
        "member_info" => &["guild_id", "user_id"],
        "search_members" => &["guild_id", "query"],
        "channel_info" => &["channel_id"],
        "fetch_messages" => &["channel_id"],
        "list_pins" => &["channel_id"],
        "pin_message" => &["channel_id", "message_id"],
        "unpin_message" => &["channel_id", "message_id"],
        "create_thread" => &["channel_id", "name"],
        "add_role" => &["guild_id", "user_id", "role_id"],
        "remove_role" => &["guild_id", "user_id", "role_id"],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Config-based action allowlist
// ---------------------------------------------------------------------------

/// Read `discord.server_actions` from a pre-loaded config value.
///
/// The Python original loads config via `hermes_cli.config.load_config`; that
/// loader is not necessarily ported here, so this function operates on an
/// already-resolved config `Value` (the parent passes in the parsed config, or
/// `None` if it could not be loaded -- both map to "allow all").
///
/// Returns:
///   * `None` when the user hasn't restricted the set (default: all allowed),
///   * `Some(vec)` of valid, known action names otherwise.
///
/// Accepts either a comma-separated string or a YAML/JSON list. Unknown action
/// names are dropped with a log warning.
pub fn parse_allowed_actions_config(cfg: Option<&Value>) -> Option<Vec<String>> {
    let cfg = cfg?;
    let raw = cfg.get("discord").and_then(|d| d.get("server_actions"));

    let raw = match raw {
        None | Some(Value::Null) => return None,
        Some(Value::String(s)) if s.is_empty() => return None,
        Some(v) => v,
    };

    let names: Vec<String> = match raw {
        Value::String(s) => s
            .split(',')
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .collect(),
        Value::Array(arr) => arr
            .iter()
            .map(value_to_plain_string)
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .collect(),
        other => {
            let tyname = json_type_name(other);
            log::warn!("discord.server_actions: unexpected type {tyname}; ignoring.");
            return None;
        }
    };

    let valid: Vec<String> = names
        .iter()
        .filter(|n| is_known_action(n))
        .cloned()
        .collect();
    let invalid: Vec<&String> = names.iter().filter(|n| !is_known_action(n)).collect();
    if !invalid.is_empty() {
        let inv = invalid
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let known = ALL_ACTIONS.join(", ");
        log::warn!(
            "discord.server_actions: unknown action(s) ignored: {inv}. Known: {known}"
        );
    }
    Some(valid)
}

/// Stringify a JSON value the way `str(n)` would for list elements.
fn value_to_plain_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Compute the visible action list from intents + config allowlist.
///
/// Preserves the canonical order from [`ALL_ACTIONS`]. Mirrors
/// `_available_actions`.
pub fn available_actions(caps: &Capabilities, allowlist: Option<&[String]>) -> Vec<String> {
    let mut actions: Vec<String> = Vec::new();
    for name in ALL_ACTIONS {
        // Intent filter.
        if !caps.has_members_intent && INTENT_GATED_MEMBERS.contains(name) {
            continue;
        }
        // Config allowlist filter.
        if let Some(allow) = allowlist {
            if !allow.iter().any(|a| a == name) {
                continue;
            }
        }
        actions.push((*name).to_string());
    }
    actions
}

// ---------------------------------------------------------------------------
// Schema construction
// ---------------------------------------------------------------------------

/// Build the tool schema for the given filtered action list.
///
/// Returns `None` when `actions` is empty -- callers should drop the tool from
/// registration in that case. Mirrors `_build_schema`.
pub fn build_schema(actions: &[String], caps: &Capabilities, tool_name: &str) -> Option<Value> {
    if actions.is_empty() {
        return None;
    }

    // Action manifest lines (action-first, parameter-scoped).
    let manifest_lines: Vec<String> = ACTION_MANIFEST
        .iter()
        .filter(|(name, _, _)| actions.iter().any(|a| a == name))
        .map(|(name, sig, desc)| format!("  {name}{sig}  \u{2014} {desc}"))
        .collect();
    let manifest_block = manifest_lines.join("\n");

    // content_note: affected = {"fetch_messages","list_pins"} & set(actions)
    let mut affected: Vec<&str> = ["fetch_messages", "list_pins"]
        .iter()
        .copied()
        .filter(|n| actions.iter().any(|a| a == n))
        .collect();
    // sorted(affected_actions) joined with " and "
    affected.sort_unstable();

    let content_note = if !affected.is_empty()
        && caps.detected
        && !caps.has_message_content
    {
        let names = affected.join(" and ");
        format!(
            "\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. \
             {names} will return message metadata (author, timestamps, attachments, \
             reactions, pin state) but `content` will be empty for messages not sent \
             as a direct mention to the bot or in DMs. Enable the intent in the \
             Discord Developer Portal to see all content."
        )
    } else {
        String::new()
    };

    let description = if tool_name == "discord_admin" {
        format!(
            "Manage a Discord server via the REST API.\n\n\
             Available actions:\n\
             {manifest_block}\n\n\
             Call list_guilds first to discover guild_ids, then list_channels for \
             channel_ids. Runtime errors will tell you if the bot lacks a specific \
             per-guild permission (e.g. MANAGE_ROLES for add_role).{content_note}"
        )
    } else {
        format!(
            "Read and participate in a Discord server.\n\n\
             Available actions:\n\
             {manifest_block}\n\n\
             Use the channel_id from the current conversation context. \
             Use search_members to look up user IDs by name prefix.{content_note}"
        )
    };

    let mut properties = Map::new();
    properties.insert(
        "action".to_string(),
        json!({"type": "string", "enum": actions}),
    );
    properties.insert(
        "guild_id".to_string(),
        json!({"type": "string", "description": "Discord server (guild) ID."}),
    );
    properties.insert(
        "channel_id".to_string(),
        json!({"type": "string", "description": "Discord channel ID."}),
    );
    properties.insert(
        "user_id".to_string(),
        json!({"type": "string", "description": "Discord user ID."}),
    );
    properties.insert(
        "role_id".to_string(),
        json!({"type": "string", "description": "Discord role ID."}),
    );
    properties.insert(
        "message_id".to_string(),
        json!({"type": "string", "description": "Discord message ID."}),
    );
    properties.insert(
        "query".to_string(),
        json!({"type": "string", "description": "Member name prefix to search for (search_members)."}),
    );
    properties.insert(
        "name".to_string(),
        json!({"type": "string", "description": "New thread name (create_thread)."}),
    );
    properties.insert(
        "limit".to_string(),
        json!({
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max results (default 50). Applies to fetch_messages, search_members.",
        }),
    );
    properties.insert(
        "before".to_string(),
        json!({"type": "string", "description": "Snowflake ID for reverse pagination (fetch_messages)."}),
    );
    properties.insert(
        "after".to_string(),
        json!({"type": "string", "description": "Snowflake ID for forward pagination (fetch_messages)."}),
    );
    properties.insert(
        "auto_archive_duration".to_string(),
        json!({
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "Thread archive duration in minutes (create_thread, default 1440).",
        }),
    );

    Some(json!({
        "name": tool_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": Value::Object(properties),
            "required": ["action"],
        },
    }))
}

/// Build a dynamic schema for a given action subset, filtered by intents + config.
///
/// Mirrors `_get_dynamic_schema`. `cfg` is the (optionally loaded) user config.
pub fn get_dynamic_schema<T: DiscordTransport>(
    transport: &T,
    action_subset: &[&str],
    tool_name: &str,
    cfg: Option<&Value>,
) -> Option<Value> {
    let token = get_bot_token()?;
    let caps = detect_capabilities(transport, &token, false);
    let allowlist = parse_allowed_actions_config(cfg);
    let actions: Vec<String> = available_actions(&caps, allowlist.as_deref())
        .into_iter()
        .filter(|a| action_subset.contains(&a.as_str()))
        .collect();
    if actions.is_empty() {
        return None;
    }
    build_schema(&actions, &caps, tool_name)
}

/// Dynamic core schema (`discord` tool). Mirrors `get_dynamic_schema_core`.
pub fn get_dynamic_schema_core<T: DiscordTransport>(
    transport: &T,
    cfg: Option<&Value>,
) -> Option<Value> {
    get_dynamic_schema(transport, CORE_ACTION_NAMES, "discord", cfg)
}

/// Dynamic admin schema (`discord_admin` tool). Mirrors `get_dynamic_schema_admin`.
pub fn get_dynamic_schema_admin<T: DiscordTransport>(
    transport: &T,
    cfg: Option<&Value>,
) -> Option<Value> {
    let admin: Vec<&str> = admin_action_names();
    get_dynamic_schema(transport, &admin, "discord_admin", cfg)
}

/// Backward-compat wrapper -- returns the core schema. Mirrors `get_dynamic_schema`.
pub fn get_dynamic_schema_default<T: DiscordTransport>(
    transport: &T,
    cfg: Option<&Value>,
) -> Option<Value> {
    get_dynamic_schema_core(transport, cfg)
}

/// Static core schema with `detected=false` capabilities (mirrors
/// `_STATIC_CORE_SCHEMA`).
pub fn static_core_schema() -> Option<Value> {
    let actions: Vec<String> = core_action_names().iter().map(|s| s.to_string()).collect();
    build_schema(
        &actions,
        &Capabilities {
            detected: false,
            ..Default::default()
        },
        "discord",
    )
}

/// Static admin schema with `detected=false` capabilities (mirrors
/// `_STATIC_ADMIN_SCHEMA`).
pub fn static_admin_schema() -> Option<Value> {
    let actions: Vec<String> = admin_action_names().iter().map(|s| s.to_string()).collect();
    build_schema(
        &actions,
        &Capabilities {
            detected: false,
            ..Default::default()
        },
        "discord_admin",
    )
}

// ---------------------------------------------------------------------------
// 403 error enrichment
// ---------------------------------------------------------------------------

/// Per-action 403 guidance string (mirrors `_ACTION_403_HINT`).
fn action_403_hint(action: &str) -> Option<&'static str> {
    match action {
        "pin_message" => Some(
            "Bot lacks MANAGE_MESSAGES permission in this channel. \
             Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, \
             or a per-channel overwrite.",
        ),
        "unpin_message" => Some("Bot lacks MANAGE_MESSAGES permission in this channel."),
        "create_thread" => Some("Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it."),
        "add_role" => Some(
            "Either the bot lacks MANAGE_ROLES, or the target role sits higher \
             than the bot's highest role. Roles can only be assigned below the \
             bot's own position in the role hierarchy.",
        ),
        "remove_role" => Some(
            "Either the bot lacks MANAGE_ROLES, or the target role sits higher \
             than the bot's highest role.",
        ),
        "fetch_messages" => Some(
            "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY).",
        ),
        "list_pins" => Some(
            "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY).",
        ),
        "channel_info" => Some("Bot cannot view this channel (missing VIEW_CHANNEL)."),
        "search_members" => Some(
            "Likely missing the Server Members privileged intent -- enable it in the \
             Discord Developer Portal under your bot's settings.",
        ),
        "member_info" => Some(
            "Bot cannot see this guild member (missing Server Members intent or \
             insufficient permissions).",
        ),
        _ => None,
    }
}

/// Return a user-friendly guidance string for a 403 on `action`.
/// Mirrors `_enrich_403`.
pub fn enrich_403(action: &str, body: &str) -> String {
    let base = format!("Discord API 403 (forbidden) on '{action}'.");
    match action_403_hint(action) {
        Some(hint) => format!("{base} {hint} (Raw: {body})"),
        None => format!("{base} (Raw: {body})"),
    }
}

// ---------------------------------------------------------------------------
// Check function
// ---------------------------------------------------------------------------

/// Tool is available only when a Discord bot token is configured.
/// Mirrors `check_discord_tool_requirements`.
pub fn check_discord_tool_requirements() -> bool {
    get_bot_token().is_some()
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Which tool a [`run_discord_action`] call belongs to. Determines the visible
/// action set and the log label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// `discord` (core actions).
    Core,
    /// `discord_admin` (server management actions).
    Admin,
}

impl ToolKind {
    fn label(self) -> &'static str {
        match self {
            ToolKind::Core => "discord",
            ToolKind::Admin => "discord_admin",
        }
    }

    fn valid_actions(self) -> Vec<&'static str> {
        match self {
            ToolKind::Core => core_action_names(),
            ToolKind::Admin => admin_action_names(),
        }
    }
}

/// Shared handler logic for both discord tools. Mirrors `_run_discord_action`.
///
/// `cfg` is the (optionally loaded) user config used for the defense-in-depth
/// allowlist gate.
pub fn run_discord_action<T: DiscordTransport>(
    transport: &T,
    action: &str,
    kind: ToolKind,
    params: &ActionParams,
    cfg: Option<&Value>,
) -> String {
    let token = match get_bot_token() {
        Some(t) => t,
        None => return json!({"error": "DISCORD_BOT_TOKEN not configured."}).to_string(),
    };

    let valid = kind.valid_actions();
    if !valid.contains(&action) {
        return json!({
            "error": format!("Unknown action: {action}"),
            "available_actions": valid,
        })
        .to_string();
    }

    // Config-level allowlist gate (defense in depth).
    let allowlist = parse_allowed_actions_config(cfg);
    if let Some(allow) = allowlist.as_ref() {
        if !allow.iter().any(|a| a == action) {
            let allowed = if allow.is_empty() {
                "<none>".to_string()
            } else {
                allow.join(", ")
            };
            return json!({
                "error": format!(
                    "Action '{action}' is disabled by config (discord.server_actions). Allowed: {allowed}"
                ),
            })
            .to_string();
        }
    }

    // Required-parameter validation.
    let local = |p: &str| -> &str {
        match p {
            "guild_id" => &params.guild_id,
            "channel_id" => &params.channel_id,
            "user_id" => &params.user_id,
            "role_id" => &params.role_id,
            "message_id" => &params.message_id,
            "query" => &params.query,
            "name" => &params.name,
            _ => "",
        }
    };
    let missing: Vec<&str> = required_params(action)
        .iter()
        .copied()
        .filter(|p| local(p).is_empty())
        .collect();
    if !missing.is_empty() {
        return json!({
            "error": format!(
                "Missing required parameters for '{action}': {}",
                missing.join(", ")
            ),
        })
        .to_string();
    }

    let f = match action_fn::<T>(action) {
        Some(f) => f,
        None => {
            return json!({
                "error": format!("Unknown action: {action}"),
                "available_actions": valid,
            })
            .to_string()
        }
    };

    match f(transport, &token, params) {
        Ok(s) => s,
        Err(DiscordError::Api(e)) => {
            log::warn!(
                "Discord API error in {} action '{}': {}",
                kind.label(),
                action,
                e
            );
            if e.status == 403 {
                json!({"error": enrich_403(action, &e.body)}).to_string()
            } else {
                json!({"error": e.to_string()}).to_string()
            }
        }
        Err(DiscordError::Transport(s)) => {
            log::error!("Unexpected error in {} action '{}'", kind.label(), action);
            json!({"error": format!("Unexpected error: {s}")}).to_string()
        }
    }
}

/// Execute a core Discord action. Mirrors `discord_core`.
pub fn discord_core<T: DiscordTransport>(
    transport: &T,
    action: &str,
    params: &ActionParams,
    cfg: Option<&Value>,
) -> String {
    run_discord_action(transport, action, ToolKind::Core, params, cfg)
}

/// Execute a Discord admin action. Mirrors `discord_admin_handler`.
pub fn discord_admin_handler<T: DiscordTransport>(
    transport: &T,
    action: &str,
    params: &ActionParams,
    cfg: Option<&Value>,
) -> String {
    run_discord_action(transport, action, ToolKind::Admin, params, cfg)
}

// ---------------------------------------------------------------------------
// Argument parsing (registry handler bridge)
// ---------------------------------------------------------------------------

/// Parse a registry-style JSON args object into [`ActionParams`] plus the
/// `action` string, applying the same defaults as `_HANDLER_DEFAULTS`.
///
/// String defaults are empty; `limit` defaults to 50 and
/// `auto_archive_duration` to 1440. `limit` tolerates int/float/string inputs
/// (matching Python's `int(limit)` coercion, falling back to 50 on failure).
pub fn parse_handler_args(args: &Value) -> (String, ActionParams) {
    let s = |k: &str| -> String {
        args.get(k)
            .and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Null => None,
                other => Some(value_to_plain_string(other)),
            })
            .unwrap_or_default()
    };

    let action = s("action");
    let limit = args
        .get("limit")
        .and_then(coerce_int)
        .unwrap_or(50);
    let auto = args
        .get("auto_archive_duration")
        .and_then(coerce_int)
        .unwrap_or(1440);

    let params = ActionParams {
        guild_id: s("guild_id"),
        channel_id: s("channel_id"),
        user_id: s("user_id"),
        role_id: s("role_id"),
        message_id: s("message_id"),
        query: s("query"),
        name: s("name"),
        limit,
        before: s("before"),
        after: s("after"),
        auto_archive_duration: auto,
    };
    (action, params)
}

/// Coerce a JSON value to an i64 like Python's `int(...)`, returning `None` on
/// failure (so callers can apply their default).
fn coerce_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serializes tests that mutate the DISCORD_BOT_TOKEN env var or the global
    // capability cache.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// A scripted fake transport. Each call pops the next queued response.
    struct FakeTransport {
        responses: StdMutex<Vec<Result<Value, DiscordError>>>,
        calls: StdMutex<Vec<(String, String)>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<Result<Value, DiscordError>>) -> Self {
            Self {
                responses: StdMutex::new(responses),
                calls: StdMutex::new(Vec::new()),
            }
        }
    }

    impl DiscordTransport for FakeTransport {
        fn request(
            &self,
            method: &str,
            path: &str,
            _token: &str,
            _params: Option<&[(String, String)]>,
            _body: Option<&Value>,
            _timeout: u64,
        ) -> Result<Value, DiscordError> {
            self.calls
                .lock()
                .unwrap()
                .push((method.to_string(), path.to_string()));
            let mut r = self.responses.lock().unwrap();
            if r.is_empty() {
                return Err(DiscordError::Transport("no scripted response".into()));
            }
            r.remove(0)
        }
    }

    #[test]
    fn test_channel_type_name() {
        assert_eq!(channel_type_name(0), "text");
        assert_eq!(channel_type_name(15), "forum");
        assert_eq!(channel_type_name(99), "unknown(99)");
    }

    #[test]
    fn test_action_sets() {
        assert_eq!(core_action_names(), vec!["search_members", "fetch_messages", "create_thread"]);
        let admin = admin_action_names();
        assert!(admin.contains(&"list_guilds"));
        assert!(!admin.contains(&"fetch_messages"));
        assert_eq!(admin.len(), 11);
    }

    #[test]
    fn test_available_actions_intent_filter() {
        let caps = Capabilities {
            has_members_intent: false,
            has_message_content: true,
            detected: true,
        };
        let actions = available_actions(&caps, None);
        assert!(!actions.iter().any(|a| a == "member_info"));
        assert!(!actions.iter().any(|a| a == "search_members"));
        assert!(actions.iter().any(|a| a == "fetch_messages"));
    }

    #[test]
    fn test_available_actions_allowlist() {
        let caps = Capabilities::default();
        let allow = vec!["list_guilds".to_string(), "fetch_messages".to_string()];
        let actions = available_actions(&caps, Some(&allow));
        assert_eq!(actions, vec!["list_guilds", "fetch_messages"]);
    }

    #[test]
    fn test_parse_allowed_actions_string() {
        let cfg = json!({"discord": {"server_actions": "list_guilds, fetch_messages, bogus"}});
        let got = parse_allowed_actions_config(Some(&cfg));
        assert_eq!(
            got,
            Some(vec!["list_guilds".to_string(), "fetch_messages".to_string()])
        );
    }

    #[test]
    fn test_parse_allowed_actions_list() {
        let cfg = json!({"discord": {"server_actions": ["pin_message", "add_role"]}});
        let got = parse_allowed_actions_config(Some(&cfg));
        assert_eq!(
            got,
            Some(vec!["pin_message".to_string(), "add_role".to_string()])
        );
    }

    #[test]
    fn test_parse_allowed_actions_empty_means_none() {
        let cfg = json!({"discord": {"server_actions": ""}});
        assert_eq!(parse_allowed_actions_config(Some(&cfg)), None);
        let cfg2 = json!({"discord": {}});
        assert_eq!(parse_allowed_actions_config(Some(&cfg2)), None);
        assert_eq!(parse_allowed_actions_config(None), None);
    }

    #[test]
    fn test_build_schema_empty_returns_none() {
        let caps = Capabilities::default();
        assert!(build_schema(&[], &caps, "discord").is_none());
    }

    #[test]
    fn test_build_schema_content_note() {
        let actions: Vec<String> = vec!["fetch_messages".to_string()];
        let caps = Capabilities {
            has_members_intent: true,
            has_message_content: false,
            detected: true,
        };
        let schema = build_schema(&actions, &caps, "discord").unwrap();
        let desc = schema["description"].as_str().unwrap();
        assert!(desc.contains("MESSAGE_CONTENT privileged intent"));
        assert!(desc.contains("fetch_messages"));
    }

    #[test]
    fn test_build_schema_no_content_note_when_undetected() {
        let actions: Vec<String> = vec!["fetch_messages".to_string()];
        let caps = Capabilities {
            has_members_intent: true,
            has_message_content: false,
            detected: false,
        };
        let schema = build_schema(&actions, &caps, "discord").unwrap();
        let desc = schema["description"].as_str().unwrap();
        assert!(!desc.contains("MESSAGE_CONTENT privileged intent"));
    }

    #[test]
    fn test_static_schemas() {
        let core = static_core_schema().unwrap();
        assert_eq!(core["name"], "discord");
        let admin = static_admin_schema().unwrap();
        assert_eq!(admin["name"], "discord_admin");
        // action enum must match the filtered set
        let core_enum = core["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(core_enum.len(), 3);
    }

    #[test]
    fn test_enrich_403_with_hint() {
        let msg = enrich_403("add_role", "{\"code\":50013}");
        assert!(msg.starts_with("Discord API 403 (forbidden) on 'add_role'."));
        assert!(msg.contains("MANAGE_ROLES"));
        assert!(msg.contains("Raw: {\"code\":50013}"));
    }

    #[test]
    fn test_enrich_403_without_hint() {
        let msg = enrich_403("list_guilds", "boom");
        assert_eq!(
            msg,
            "Discord API 403 (forbidden) on 'list_guilds'. (Raw: boom)"
        );
    }

    #[test]
    fn test_detect_capabilities_flags() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_capability_cache();
        // Set both members + message content flags.
        let flags = FLAG_GATEWAY_GUILD_MEMBERS | FLAG_GATEWAY_MESSAGE_CONTENT;
        let t = FakeTransport::new(vec![Ok(json!({"flags": flags}))]);
        let caps = detect_capabilities(&t, "tok1", false);
        assert!(caps.has_members_intent);
        assert!(caps.has_message_content);
        assert!(caps.detected);
        reset_capability_cache();
    }

    #[test]
    fn test_detect_capabilities_no_intents() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_capability_cache();
        let t = FakeTransport::new(vec![Ok(json!({"flags": 0}))]);
        let caps = detect_capabilities(&t, "tok2", false);
        assert!(!caps.has_members_intent);
        assert!(!caps.has_message_content);
        assert!(caps.detected);
        reset_capability_cache();
    }

    #[test]
    fn test_detect_capabilities_failure_defaults() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_capability_cache();
        let t = FakeTransport::new(vec![Err(DiscordError::Transport("network".into()))]);
        let caps = detect_capabilities(&t, "tok3", false);
        assert!(caps.has_members_intent);
        assert!(caps.has_message_content);
        assert!(!caps.detected);
        reset_capability_cache();
    }

    #[test]
    fn test_list_guilds_parsing() {
        let _g = ENV_LOCK.lock().unwrap();
        let t = FakeTransport::new(vec![Ok(json!([
            {"id": "1", "name": "A", "owner": true, "permissions": "8"},
            {"id": "2", "name": "B"}
        ]))]);
        let out = list_guilds(&t, "tok", &ActionParams::new()).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["count"], 2);
        assert_eq!(v["guilds"][0]["owner"], true);
        assert_eq!(v["guilds"][1]["owner"], false);
        assert_eq!(v["guilds"][1]["permissions"], Value::Null);
    }

    #[test]
    fn test_list_roles_sort_and_color() {
        let _g = ENV_LOCK.lock().unwrap();
        let t = FakeTransport::new(vec![Ok(json!([
            {"id": "a", "name": "low", "position": 1, "color": 0},
            {"id": "b", "name": "high", "position": 5, "color": 16711680}
        ]))]);
        let out = list_roles(&t, "tok", &ActionParams::new()).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        // highest position first
        assert_eq!(v["roles"][0]["name"], "high");
        assert_eq!(v["roles"][0]["color"], "#ff0000");
        // color 0 -> null
        assert_eq!(v["roles"][1]["color"], Value::Null);
    }

    #[test]
    fn test_list_channels_grouping() {
        let _g = ENV_LOCK.lock().unwrap();
        let t = FakeTransport::new(vec![Ok(json!([
            {"id": "cat1", "name": "Cat", "type": 4, "position": 0},
            {"id": "c1", "name": "general", "type": 0, "position": 1, "parent_id": "cat1"},
            {"id": "c2", "name": "loose", "type": 0, "position": 0}
        ]))]);
        let out = list_channels(&t, "tok", &ActionParams::new()).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["total_channels"], 2);
        // uncategorized group comes first
        assert_eq!(v["channel_groups"][0]["category"], Value::Null);
        assert_eq!(v["channel_groups"][0]["channels"][0]["name"], "loose");
        assert_eq!(v["channel_groups"][1]["category"]["name"], "Cat");
        assert_eq!(v["channel_groups"][1]["channels"][0]["name"], "general");
    }

    #[test]
    fn test_fetch_messages_reactions_and_truncation() {
        let _g = ENV_LOCK.lock().unwrap();
        let t = FakeTransport::new(vec![Ok(json!([
            {
                "id": "m1",
                "content": "hi",
                "author": {"id": "u1", "username": "bob", "global_name": "Bob"},
                "reactions": [{"emoji": {"name": "thumbsup"}, "count": 3}],
                "attachments": [{"filename": "f.png", "url": "http://x", "size": 10}]
            },
            {"id": "m2"}
        ]))]);
        let out = fetch_messages(&t, "tok", &ActionParams::new()).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["count"], 2);
        assert_eq!(v["messages"][0]["reactions"][0]["emoji"], "thumbsup");
        assert_eq!(v["messages"][0]["reactions"][0]["count"], 3);
        assert_eq!(v["messages"][0]["attachments"][0]["filename"], "f.png");
        // m2 has no reactions -> empty array, content defaults to ""
        assert_eq!(v["messages"][1]["reactions"], json!([]));
        assert_eq!(v["messages"][1]["content"], "");
    }

    #[test]
    fn test_run_action_missing_token() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
        let t = FakeTransport::new(vec![]);
        let out = run_discord_action(&t, "list_guilds", ToolKind::Admin, &ActionParams::new(), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], "DISCORD_BOT_TOKEN not configured.");
    }

    #[test]
    fn test_run_action_unknown_action() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "secret");
        }
        let t = FakeTransport::new(vec![]);
        let out = run_discord_action(&t, "nonsense", ToolKind::Admin, &ActionParams::new(), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("Unknown action: nonsense"));
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
    }

    #[test]
    fn test_run_action_missing_required_params() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "secret");
        }
        let t = FakeTransport::new(vec![]);
        // server_info requires guild_id; params default empty.
        let out = run_discord_action(&t, "server_info", ToolKind::Admin, &ActionParams::new(), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("Missing required parameters for 'server_info': guild_id"));
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
    }

    #[test]
    fn test_run_action_allowlist_denied() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "secret");
        }
        let t = FakeTransport::new(vec![]);
        let cfg = json!({"discord": {"server_actions": "list_guilds"}});
        // pin_message not in allowlist -> denied (it's an admin action with required params,
        // but allowlist check happens before required-param check).
        let out = run_discord_action(
            &t,
            "pin_message",
            ToolKind::Admin,
            &ActionParams::new(),
            Some(&cfg),
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("disabled by config"));
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
    }

    #[test]
    fn test_run_action_403_enrichment() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "secret");
        }
        let t = FakeTransport::new(vec![Err(DiscordError::Api(DiscordApiError::new(
            403,
            "forbidden-body".into(),
        )))]);
        let mut params = ActionParams::new();
        params.guild_id = "g".into();
        params.user_id = "u".into();
        params.role_id = "r".into();
        let out = run_discord_action(&t, "add_role", ToolKind::Admin, &params, None);
        let v: Value = serde_json::from_str(&out).unwrap();
        let err = v["error"].as_str().unwrap();
        assert!(err.contains("403 (forbidden) on 'add_role'"));
        assert!(err.contains("MANAGE_ROLES"));
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
    }

    #[test]
    fn test_get_bot_token_trims() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "  abc  ");
        }
        assert_eq!(get_bot_token(), Some("abc".to_string()));
        unsafe {
            std::env::set_var("DISCORD_BOT_TOKEN", "   ");
        }
        assert_eq!(get_bot_token(), None);
        unsafe {
            std::env::remove_var("DISCORD_BOT_TOKEN");
        }
        assert_eq!(get_bot_token(), None);
    }

    #[test]
    fn test_parse_handler_args() {
        let args = json!({
            "action": "fetch_messages",
            "channel_id": "123",
            "limit": "25",
            "auto_archive_duration": 4320
        });
        let (action, p) = parse_handler_args(&args);
        assert_eq!(action, "fetch_messages");
        assert_eq!(p.channel_id, "123");
        assert_eq!(p.limit, 25);
        assert_eq!(p.auto_archive_duration, 4320);
        assert_eq!(p.guild_id, "");
        // missing limit -> default 50
        let (_, p2) = parse_handler_args(&json!({"action": "x"}));
        assert_eq!(p2.limit, 50);
        assert_eq!(p2.auto_archive_duration, 1440);
    }

    #[test]
    fn test_urlencode() {
        assert_eq!(urlencode("hello world"), "hello+world");
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        assert_eq!(urlencode("plain.txt"), "plain.txt");
    }

    #[test]
    fn test_build_url_with_params() {
        let params = vec![
            ("query".to_string(), "jo".to_string()),
            ("limit".to_string(), "20".to_string()),
        ];
        let url = build_url("/guilds/1/members/search", Some(&params));
        assert_eq!(
            url,
            "https://discord.com/api/v10/guilds/1/members/search?query=jo&limit=20"
        );
    }
}
