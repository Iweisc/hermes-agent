//! Discord platform adapter — native Rust port of
//! `gateway/platforms/discord.py`.
//!
//! The original Python module is built on the `discord.py` library, whose
//! gateway websocket lifecycle, async event loop, voice client, and `discord.ui`
//! component framework have no faithful equivalent in the crates available to
//! this port. This module therefore ports the **deterministic, side-effect-free
//! logic** that other Hermes code (and tests) depend on, reproducing the
//! Python behavior exactly:
//!
//! - [`clean_discord_id`] — strip mention/`user:` prefixes from an allowlist
//!   entry (Python `_clean_discord_id`).
//! - [`AllowedMentions`] / [`build_allowed_mentions`] — env-driven safe mention
//!   defaults (Python `_build_allowed_mentions`).
//! - [`VoiceReceiver`] RTP parsing helpers — header sizing, padding stripping,
//!   silence detection thresholds (Python `VoiceReceiver`).
//! - Slash-command sync diffing: [`canonicalize_app_command_payload`],
//!   [`canonicalize_app_command_option`], [`patchable_app_command_payload`],
//!   [`normalize_permissions`], and the [`reconcile_slash_commands`] diff
//!   algorithm (Python `_safe_sync_*`).
//! - Authorization gates: [`is_allowed_user`], [`component_check_auth`],
//!   [`evaluate_slash_authorization`] (Python `_is_allowed_user`,
//!   `_component_check_auth`, `_evaluate_slash_authorization`).
//! - Env / config helpers: [`command_sync_policy`], [`require_mention`],
//!   [`free_response_channels`], [`reactions_enabled`], [`allow_bots_mode`],
//!   [`ignore_no_mention`], [`auto_thread_enabled`], [`hide_slash_commands`].
//! - Message shaping: [`derive_auto_thread_name`], [`format_thread_chat_name`],
//!   [`is_forum_parent_type`], [`effective_topic`], [`document_ext_for`].
//!
//! Network sends, voice playback, and the websocket loop are out of scope —
//! they require the live `discord.py` client. Callers wire those through the
//! Python bridge; this module supplies the pure logic those paths call into.

use std::collections::{HashMap, HashSet};
use std::env;

// ─── Constants (mirror module-level Python constants) ───────────────────────

/// Valid `auto_archive_duration` values for Discord threads (minutes).
pub const VALID_THREAD_AUTO_ARCHIVE_MINUTES: &[i64] = &[60, 1440, 4320, 10080];

/// Recognized values for `DISCORD_COMMAND_SYNC_POLICY`.
pub const DISCORD_COMMAND_SYNC_POLICIES: &[&str] = &["safe", "bulk", "off"];

/// Discord single-message character limit.
pub const MAX_MESSAGE_LENGTH: usize = 2000;

/// Near the 2000-char split point — used by the text-batch flush delay.
pub const SPLIT_THRESHOLD: usize = 1900;

/// Auto-disconnect from a voice channel after this many seconds of inactivity.
pub const VOICE_TIMEOUT_SECS: u64 = 300;

/// Maximum seconds to wait for voice playback before giving up.
pub const PLAYBACK_TIMEOUT_SECS: u64 = 120;

/// UDP keepalive interval (seconds) for the voice listen loop.
pub const VOICE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

// ─── ID cleaning ────────────────────────────────────────────────────────────

/// Strip common prefixes from a Discord user ID or username entry.
///
/// Mirrors Python `_clean_discord_id`: strips `<@123>` / `<@!123>` mention
/// syntax and a leading `user:` prefix (case-insensitive), then trims.
pub fn clean_discord_id(entry: &str) -> String {
    let mut entry = entry.trim().to_string();
    // Strip Discord mention syntax: <@123> or <@!123>
    if entry.starts_with("<@") && entry.ends_with('>') {
        // Python: entry.lstrip("<@!").rstrip(">")
        let start = entry.trim_start_matches(['<', '@', '!']);
        entry = start.trim_end_matches('>').to_string();
    }
    // Strip "user:" prefix (case-insensitive on the prefix).
    if entry.to_lowercase().starts_with("user:") {
        entry = entry[5..].to_string();
    }
    entry.trim().to_string()
}

// ─── Allowed mentions ───────────────────────────────────────────────────────

/// Safe default mention permissions for the Discord client.
///
/// Mirrors Python `_build_allowed_mentions`: deny `@everyone`/`@here` and role
/// pings by default; allow user and replied-user pings. Overridable via the
/// `DISCORD_ALLOW_MENTION_*` env vars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowedMentions {
    pub everyone: bool,
    pub roles: bool,
    pub users: bool,
    pub replied_user: bool,
}

impl Default for AllowedMentions {
    fn default() -> Self {
        Self {
            everyone: false,
            roles: false,
            users: true,
            replied_user: true,
        }
    }
}

/// Parse a boolean-ish env var the way Python `_b` does: empty → default,
/// otherwise true iff the lowercased value is one of `true/1/yes/on`.
fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(raw) => {
            let raw = raw.trim().to_lowercase();
            if raw.is_empty() {
                default
            } else {
                matches!(raw.as_str(), "true" | "1" | "yes" | "on")
            }
        }
        Err(_) => default,
    }
}

/// Build [`AllowedMentions`] from the `DISCORD_ALLOW_MENTION_*` env vars.
pub fn build_allowed_mentions() -> AllowedMentions {
    AllowedMentions {
        everyone: env_bool("DISCORD_ALLOW_MENTION_EVERYONE", false),
        roles: env_bool("DISCORD_ALLOW_MENTION_ROLES", false),
        users: env_bool("DISCORD_ALLOW_MENTION_USERS", true),
        replied_user: env_bool("DISCORD_ALLOW_MENTION_REPLIED_USER", true),
    }
}

// ─── Reply / batching config ────────────────────────────────────────────────

/// Reply-threading mode: how reply references are attached to chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyToMode {
    /// Never attach a reply reference.
    Off,
    /// Attach the reference on the first chunk only (Python default).
    First,
    /// Attach the reference on every chunk.
    All,
}

impl ReplyToMode {
    pub fn from_str_or_default(value: Option<&str>) -> Self {
        match value.map(|s| s.trim().to_lowercase()).as_deref() {
            Some("off") => ReplyToMode::Off,
            Some("all") => ReplyToMode::All,
            // "first" or anything unrecognized / empty / None → First (default).
            _ => ReplyToMode::First,
        }
    }

    /// Whether a given chunk index should carry the reply reference.
    pub fn reference_for_chunk(self, index: usize) -> bool {
        match self {
            ReplyToMode::Off => false,
            ReplyToMode::All => true,
            ReplyToMode::First => index == 0,
        }
    }
}

/// Text-batch delays, read from the same env vars as the Python adapter.
#[derive(Debug, Clone, Copy)]
pub struct TextBatchConfig {
    pub delay_seconds: f64,
    pub split_delay_seconds: f64,
}

impl TextBatchConfig {
    pub fn from_env() -> Self {
        fn parse(name: &str, default: f64) -> f64 {
            env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or(default)
        }
        Self {
            delay_seconds: parse("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", 0.6),
            split_delay_seconds: parse("HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0),
        }
    }

    /// Pick the flush delay for a batch whose latest chunk had `last_chunk_len`
    /// chars: the longer split delay when near the 2000-char split point.
    pub fn delay_for(&self, last_chunk_len: usize) -> f64 {
        if last_chunk_len >= SPLIT_THRESHOLD {
            self.split_delay_seconds
        } else {
            self.delay_seconds
        }
    }

    /// Batching is active only when the base delay is positive (Python
    /// `if msg_type == TEXT and self._text_batch_delay_seconds > 0`).
    pub fn enabled(&self) -> bool {
        self.delay_seconds > 0.0
    }
}

// ─── Env-driven feature flags ───────────────────────────────────────────────

/// Resolve `DISCORD_COMMAND_SYNC_POLICY` to one of `safe` / `bulk` / `off`,
/// falling back to `safe` for unset or invalid values.
pub fn command_sync_policy() -> String {
    let raw = env::var("DISCORD_COMMAND_SYNC_POLICY")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if DISCORD_COMMAND_SYNC_POLICIES.contains(&raw.as_str()) {
        raw
    } else {
        "safe".to_string()
    }
}

/// Whether message reactions are enabled (`DISCORD_REACTIONS`, default true).
///
/// Mirrors Python: disabled iff lowercased value is `false`/`0`/`no`.
pub fn reactions_enabled() -> bool {
    let v = env::var("DISCORD_REACTIONS")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();
    !matches!(v.as_str(), "false" | "0" | "no")
}

/// How foreign-bot messages are treated: parsed from `DISCORD_ALLOW_BOTS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowBotsMode {
    /// Ignore all other bots (default).
    None,
    /// Accept bot messages only when they @mention us.
    Mentions,
    /// Accept all bot messages.
    All,
}

/// Parse `DISCORD_ALLOW_BOTS` (default `none`). Unknown values map to `None`
/// because the Python code treats anything that isn't `mentions`/`all` as the
/// `none` early-return path (only `none`/`mentions`/`all` are handled, with the
/// implicit else for `"all"` falling through — here we follow the explicit
/// branches and default everything else to `None`).
pub fn allow_bots_mode() -> AllowBotsMode {
    let v = env::var("DISCORD_ALLOW_BOTS")
        .unwrap_or_else(|_| "none".to_string())
        .to_lowercase();
    let v = v.trim();
    match v {
        "mentions" => AllowBotsMode::Mentions,
        "all" => AllowBotsMode::All,
        _ => AllowBotsMode::None,
    }
}

/// Whether messages with no self-mention should be ignored in channels.
///
/// `DISCORD_IGNORE_NO_MENTION` (default true): true iff lowercased value is in
/// `true`/`1`/`yes`.
pub fn ignore_no_mention() -> bool {
    let v = env::var("DISCORD_IGNORE_NO_MENTION")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();
    matches!(v.as_str(), "true" | "1" | "yes")
}

/// Whether auto-threading is enabled (`DISCORD_AUTO_THREAD`, default true).
pub fn auto_thread_enabled() -> bool {
    let v = env::var("DISCORD_AUTO_THREAD")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();
    matches!(v.as_str(), "true" | "1" | "yes")
}

/// Whether to hide slash commands from non-admins (`DISCORD_HIDE_SLASH_COMMANDS`,
/// default false): true iff lowercased trimmed value is `true`/`1`/`yes`/`on`.
pub fn hide_slash_commands() -> bool {
    let v = env::var("DISCORD_HIDE_SLASH_COMMANDS")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    matches!(v.as_str(), "true" | "1" | "yes" | "on")
}

/// Resolve whether channel messages require a bot mention.
///
/// Mirrors Python `_discord_require_mention`: a configured `require_mention`
/// value (passed in via `configured`) wins; a string is truthy unless it is
/// `false`/`0`/`no`/`off`; otherwise consult `DISCORD_REQUIRE_MENTION`
/// (default true; false iff `false`/`0`/`no`/`off`).
pub fn require_mention(configured: Option<&ConfiguredFlag>) -> bool {
    if let Some(flag) = configured {
        return match flag {
            ConfiguredFlag::Bool(b) => *b,
            ConfiguredFlag::Str(s) => {
                !matches!(s.trim().to_lowercase().as_str(), "false" | "0" | "no" | "off")
            }
        };
    }
    let v = env::var("DISCORD_REQUIRE_MENTION")
        .unwrap_or_else(|_| "true".to_string())
        .to_lowercase();
    !matches!(v.as_str(), "false" | "0" | "no" | "off")
}

/// A config value that may be a bool or a string (mirrors how YAML scalars
/// arrive from `config.extra` in Python).
#[derive(Debug, Clone)]
pub enum ConfiguredFlag {
    Bool(bool),
    Str(String),
}

/// Return Discord channel IDs where no bot mention is required.
///
/// Mirrors Python `_discord_free_response_channels`: a list is normalized to a
/// trimmed string set; a scalar is coerced to a string and CSV-split; falls
/// back to `DISCORD_FREE_RESPONSE_CHANNELS` when no config value is provided.
/// A `"*"` wildcard entry is preserved.
pub fn free_response_channels(configured: Option<&FreeResponseConfig>) -> HashSet<String> {
    match configured {
        Some(FreeResponseConfig::List(items)) => items
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(FreeResponseConfig::Scalar(s)) => csv_set(s),
        None => {
            let raw = env::var("DISCORD_FREE_RESPONSE_CHANNELS").unwrap_or_default();
            csv_set(&raw)
        }
    }
}

/// Config value for free-response channels.
#[derive(Debug, Clone)]
pub enum FreeResponseConfig {
    List(Vec<String>),
    Scalar(String),
}

/// Split a comma-separated string into a trimmed, non-empty set.
pub fn csv_set(raw: &str) -> HashSet<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ─── Channel / topic / forum helpers ────────────────────────────────────────

/// Whether a Discord channel type value denotes a forum channel (type 15).
///
/// Mirrors the `_is_forum_parent` type-value branch (the `isinstance`
/// fast-path requires a live discord object, so callers pass the numeric type).
pub fn is_forum_parent_type(channel_type: Option<i64>) -> bool {
    channel_type == Some(15)
}

/// Resolve the effective topic, falling back to a forum parent's topic for
/// forum threads. Mirrors Python `_get_effective_topic`.
pub fn effective_topic(
    channel_topic: Option<&str>,
    is_thread: bool,
    parent_is_forum: bool,
    parent_topic: Option<&str>,
) -> Option<String> {
    let topic = channel_topic.filter(|t| !t.is_empty());
    if topic.is_none() && is_thread && parent_is_forum {
        return parent_topic.filter(|t| !t.is_empty()).map(|t| t.to_string());
    }
    topic.map(|t| t.to_string())
}

/// Build a readable chat name for a thread-like channel.
///
/// Mirrors Python `_format_thread_chat_name`.
pub fn format_thread_chat_name(
    thread_name: Option<&str>,
    thread_id: Option<&str>,
    parent_name: Option<&str>,
    parent_is_forum: bool,
    guild_name: Option<&str>,
) -> String {
    let thread_name = thread_name
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| thread_id.unwrap_or("thread").to_string());

    let parent_name = parent_name.filter(|s| !s.is_empty());
    let guild_name = guild_name.filter(|s| !s.is_empty());

    match (parent_is_forum, guild_name, parent_name) {
        (true, Some(g), Some(p)) => format!("{g} / {p} / {thread_name}"),
        (_, Some(g), Some(p)) => format!("{g} / #{p} / {thread_name}"),
        (_, _, Some(p)) => format!("{p} / {thread_name}"),
        _ => thread_name,
    }
}

/// Derive a short thread name for auto-threading from a message body.
///
/// Mirrors Python `_auto_create_thread`'s name derivation: strip mention/role/
/// channel mention syntax, collapse whitespace, clamp to 80 chars (with a
/// `...` suffix when truncated), default to "Hermes" when empty.
pub fn derive_auto_thread_name(content: &str) -> String {
    let mut content = content.trim().to_string();
    // <@123>, <@!123>, <@&123>
    let re_user = regex::Regex::new(r"<@[!&]?\d+>").unwrap();
    content = re_user.replace_all(&content, "").to_string();
    // <#123>
    let re_chan = regex::Regex::new(r"<#\d+>").unwrap();
    content = re_chan.replace_all(&content, "").to_string();
    // collapse whitespace
    let re_ws = regex::Regex::new(r"\s+").unwrap();
    content = re_ws.replace_all(&content, " ").trim().to_string();

    if content.is_empty() {
        return "Hermes".to_string();
    }
    // Python uses character (codepoint) slicing.
    let chars: Vec<char> = content.chars().collect();
    if chars.len() > 80 {
        let truncated: String = chars[..77].iter().collect();
        format!("{truncated}...")
    } else {
        chars.into_iter().collect()
    }
}

// ─── Document type detection ────────────────────────────────────────────────

/// Determine the lowercased file extension (with leading dot) for a document
/// attachment, mirroring the Python logic that prefers the filename extension
/// then falls back to a MIME→ext mapping.
pub fn document_ext_for(
    filename: Option<&str>,
    content_type: Option<&str>,
    mime_to_ext: &HashMap<String, String>,
) -> String {
    if let Some(name) = filename {
        if let Some(idx) = name.rfind('.') {
            let ext = name[idx..].to_lowercase();
            if !ext.is_empty() && ext != "." {
                return ext;
            }
        }
    }
    if let Some(ct) = content_type {
        if let Some(ext) = mime_to_ext.get(ct) {
            return ext.clone();
        }
    }
    String::new()
}

// ─── Authorization ──────────────────────────────────────────────────────────

/// A minimal Discord member/user view for authorization checks.
#[derive(Debug, Clone, Default)]
pub struct AuthorView {
    pub id: Option<i64>,
    /// Role IDs available on a Member object. `None` means the role list is not
    /// resolvable (e.g. a raw User payload / DM context) — fail-closed for the
    /// component check.
    pub roles: Option<Vec<i64>>,
}

/// Core user/role allowlist check (Python `_is_allowed_user`).
///
/// OR semantics: a user is allowed if they match *either* allowlist. Both
/// empty → everyone allowed. `mutual_guild_roles` supplies the fallback role
/// lookup (the Python code scans the bot's mutual guilds when the author has no
/// direct `.roles`).
pub fn is_allowed_user(
    user_id: &str,
    allowed_users: &HashSet<String>,
    allowed_roles: &HashSet<i64>,
    author: Option<&AuthorView>,
    mutual_guild_roles: &[i64],
) -> bool {
    let has_users = !allowed_users.is_empty();
    let has_roles = !allowed_roles.is_empty();
    if !has_users && !has_roles {
        return true;
    }
    if has_users && allowed_users.contains(user_id) {
        return true;
    }
    if has_roles {
        // Direct role check from the Member object.
        if let Some(view) = author {
            if let Some(roles) = &view.roles {
                if roles.iter().any(|r| allowed_roles.contains(r)) {
                    return true;
                }
            }
        }
        // Fallback: mutual-guild role scan (already resolved by the caller).
        if mutual_guild_roles.iter().any(|r| allowed_roles.contains(r)) {
            return true;
        }
    }
    false
}

/// Component view auth check (Python `_component_check_auth`).
///
/// - both allowlists empty → allow
/// - user in user allowlist → allow
/// - role allowlist set and user has a matching role → allow
/// - role allowlist set but user has no resolvable role list → fail closed
/// - missing user → reject
pub fn component_check_auth(
    user: Option<&AuthorView>,
    allowed_users: &HashSet<String>,
    allowed_roles: &HashSet<i64>,
) -> bool {
    let has_users = !allowed_users.is_empty();
    let has_roles = !allowed_roles.is_empty();
    if !has_users && !has_roles {
        return true;
    }
    let user = match user {
        Some(u) => u,
        None => return false,
    };
    if has_users {
        if let Some(id) = user.id {
            if allowed_users.contains(&id.to_string()) {
                return true;
            }
        }
    }
    if has_roles {
        match &user.roles {
            // Role policy active but no role data → fail closed.
            None => return false,
            Some(roles) => {
                if roles.iter().any(|r| allowed_roles.contains(r)) {
                    return true;
                }
            }
        }
    }
    false
}

/// A view of a slash interaction for authorization (Python
/// `_evaluate_slash_authorization` inputs).
#[derive(Debug, Clone, Default)]
pub struct SlashInteractionView {
    pub in_dm: bool,
    /// The interaction's channel id (`channel_id` or `channel.id`).
    pub channel_id: Option<String>,
    /// If the channel is a thread, its parent channel id.
    pub thread_parent_id: Option<String>,
    pub user: Option<AuthorView>,
}

/// The outcome of a slash authorization evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashAuthResult {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl SlashAuthResult {
    fn allow() -> Self {
        Self { allowed: true, reason: None }
    }
    fn reject(reason: &str) -> Self {
        Self { allowed: false, reason: Some(reason.to_string()) }
    }
}

/// Evaluate slash authorization without producing a response.
///
/// Faithful port of `_evaluate_slash_authorization`, including the fail-closed
/// branches for malformed payloads. `allowed_channels_raw` / `ignored_channels_raw`
/// come from `DISCORD_ALLOWED_CHANNELS` / `DISCORD_IGNORED_CHANNELS`.
pub fn evaluate_slash_authorization(
    interaction: &SlashInteractionView,
    allowed_users: &HashSet<String>,
    allowed_roles: &HashSet<i64>,
    mutual_guild_roles: &[i64],
) -> SlashAuthResult {
    // ── Channel scope (DMs are not channel-gated) ──
    if !interaction.in_dm {
        let mut channel_ids: HashSet<String> = HashSet::new();
        if let Some(cid) = &interaction.channel_id {
            channel_ids.insert(cid.clone());
            if let Some(pid) = &interaction.thread_parent_id {
                channel_ids.insert(pid.clone());
            }
        }

        let allowed_raw = env::var("DISCORD_ALLOWED_CHANNELS").unwrap_or_default();
        if !allowed_raw.is_empty() {
            let allowed = csv_set(&allowed_raw);
            if !allowed.contains("*") {
                if channel_ids.is_empty() {
                    return SlashAuthResult::reject(
                        "channel id missing with DISCORD_ALLOWED_CHANNELS configured",
                    );
                }
                if channel_ids.intersection(&allowed).next().is_none() {
                    return SlashAuthResult::reject("channel not in DISCORD_ALLOWED_CHANNELS");
                }
            }
        }

        // Ignored beats allowed.
        let ignored_raw = env::var("DISCORD_IGNORED_CHANNELS").unwrap_or_default();
        if !ignored_raw.is_empty() && !channel_ids.is_empty() {
            let ignored = csv_set(&ignored_raw);
            if ignored.contains("*") || channel_ids.intersection(&ignored).next().is_some() {
                return SlashAuthResult::reject("channel in DISCORD_IGNORED_CHANNELS");
            }
        }
    }

    // ── User / role allowlist ──
    let user = interaction.user.as_ref();
    let has_user_id = user.map(|u| u.id.is_some()).unwrap_or(false);
    if !has_user_id {
        if !allowed_users.is_empty() || !allowed_roles.is_empty() {
            return SlashAuthResult::reject("missing interaction.user with allowlist configured");
        }
        return SlashAuthResult::allow();
    }

    let view = user.unwrap();
    let user_id = view.id.unwrap().to_string();
    if !is_allowed_user(&user_id, allowed_users, allowed_roles, Some(view), mutual_guild_roles) {
        return SlashAuthResult::reject(
            "user not in DISCORD_ALLOWED_USERS / DISCORD_ALLOWED_ROLES",
        );
    }

    SlashAuthResult::allow()
}

// ─── Slash-command canonicalization & diffing ───────────────────────────────

/// Normalize `default_member_permissions` to a stable str-or-None
/// (Python `_normalize_permissions`).
pub fn normalize_permissions(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(other) => Some(json_scalar_to_string(other)),
    }
}

/// Stringify a JSON scalar the way Python `str()` would for ints/bools/floats.
fn json_scalar_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => {
            if *b { "True".to_string() } else { "False".to_string() }
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn as_i64(v: Option<&serde_json::Value>, default: i64) -> i64 {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(default),
        Some(serde_json::Value::String(s)) => s.parse().unwrap_or(default),
        _ => default,
    }
}

fn as_bool(v: Option<&serde_json::Value>, default: bool) -> bool {
    match v {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Null) | None => default,
        Some(serde_json::Value::Number(n)) => n.as_i64().map(|i| i != 0).unwrap_or(default),
        Some(serde_json::Value::String(s)) => !s.is_empty(),
        _ => default,
    }
}

fn as_str(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Null) | None => String::new(),
        Some(other) => json_scalar_to_string(other),
    }
}

/// Sorted-list-of-ints helper for `contexts` / `integration_types`.
fn sorted_int_list(v: Option<&serde_json::Value>) -> Option<Vec<i64>> {
    match v {
        Some(serde_json::Value::Array(arr)) if !arr.is_empty() => {
            let mut out: Vec<i64> = arr
                .iter()
                .filter_map(|x| match x {
                    serde_json::Value::Number(n) => n.as_i64(),
                    serde_json::Value::String(s) => s.parse().ok(),
                    _ => None,
                })
                .collect();
            out.sort_unstable();
            Some(out)
        }
        _ => None,
    }
}

/// Canonicalize an app-command option payload (recursive).
/// Mirrors Python `_canonicalize_app_command_option`.
pub fn canonicalize_app_command_option(payload: &serde_json::Value) -> serde_json::Value {
    let obj = payload.as_object();
    let get = |k: &str| obj.and_then(|o| o.get(k));

    let choices: Vec<serde_json::Value> = get("choices")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|c| c.is_object())
                .map(|c| {
                    let co = c.as_object().unwrap();
                    serde_json::json!({
                        "name": as_str(co.get("name")),
                        "value": co.get("value").cloned().unwrap_or(serde_json::Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let channel_types: Vec<serde_json::Value> = get("channel_types")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    let options: Vec<serde_json::Value> = get("options")
        .and_then(|o| o.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|i| i.is_object())
                .map(canonicalize_app_command_option)
                .collect()
        })
        .unwrap_or_default();

    serde_json::json!({
        "type": as_i64(get("type"), 0),
        "name": as_str(get("name")),
        "description": as_str(get("description")),
        "required": as_bool(get("required"), false),
        "autocomplete": as_bool(get("autocomplete"), false),
        "choices": choices,
        "channel_types": channel_types,
        "min_value": get("min_value").cloned().unwrap_or(serde_json::Value::Null),
        "max_value": get("max_value").cloned().unwrap_or(serde_json::Value::Null),
        "min_length": get("min_length").cloned().unwrap_or(serde_json::Value::Null),
        "max_length": get("max_length").cloned().unwrap_or(serde_json::Value::Null),
        "options": options,
    })
}

/// Canonicalize a top-level app-command payload to the fields Hermes manages.
/// Mirrors Python `_canonicalize_app_command_payload`.
pub fn canonicalize_app_command_payload(payload: &serde_json::Value) -> serde_json::Value {
    let obj = payload.as_object();
    let get = |k: &str| obj.and_then(|o| o.get(k));

    let cmd_type = {
        // Python: int(payload.get("type", 1) or 1)
        let raw = as_i64(get("type"), 1);
        if raw == 0 { 1 } else { raw }
    };

    let options: Vec<serde_json::Value> = get("options")
        .and_then(|o| o.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|i| i.is_object())
                .map(canonicalize_app_command_option)
                .collect()
        })
        .unwrap_or_default();

    let mut map = serde_json::Map::new();
    map.insert("type".into(), serde_json::json!(cmd_type));
    map.insert("name".into(), serde_json::json!(as_str(get("name"))));
    map.insert("description".into(), serde_json::json!(as_str(get("description"))));
    map.insert(
        "default_member_permissions".into(),
        match normalize_permissions(get("default_member_permissions")) {
            Some(s) => serde_json::Value::String(s),
            None => serde_json::Value::Null,
        },
    );
    map.insert("dm_permission".into(), serde_json::json!(as_bool(get("dm_permission"), true)));
    map.insert("nsfw".into(), serde_json::json!(as_bool(get("nsfw"), false)));
    map.insert(
        "contexts".into(),
        match sorted_int_list(get("contexts")) {
            Some(list) => serde_json::json!(list),
            None => serde_json::Value::Null,
        },
    );
    map.insert(
        "integration_types".into(),
        match sorted_int_list(get("integration_types")) {
            Some(list) => serde_json::json!(list),
            None => serde_json::Value::Null,
        },
    );
    map.insert("options".into(), serde_json::Value::Array(options));

    serde_json::Value::Object(map)
}

/// Reduce a payload to the fields supported by `edit_global_command`.
/// Mirrors Python `_patchable_app_command_payload`.
pub fn patchable_app_command_payload(payload: &serde_json::Value) -> serde_json::Value {
    let canonical = canonicalize_app_command_payload(payload);
    serde_json::json!({
        "name": canonical.get("name").cloned().unwrap_or(serde_json::Value::Null),
        "description": canonical.get("description").cloned().unwrap_or(serde_json::Value::Null),
        "options": canonical.get("options").cloned().unwrap_or(serde_json::Value::Array(vec![])),
    })
}

/// Summary of a slash-command reconcile diff. Mirrors the dict returned by
/// Python `_safe_sync_slash_commands`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SyncSummary {
    pub total: usize,
    pub unchanged: usize,
    pub updated: usize,
    pub recreated: usize,
    pub created: usize,
    pub deleted: usize,
}

/// The action computed for one command during reconciliation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncAction {
    /// No change needed.
    Unchanged,
    /// `upsert_global_command(desired)`.
    Create,
    /// `delete(existing_id)` then `upsert(desired)`.
    Recreate,
    /// `edit_global_command(existing_id, desired)`.
    Update,
    /// `delete_global_command(existing_id)` for orphaned commands.
    Delete,
}

/// A planned mutation produced by [`reconcile_slash_commands`].
#[derive(Debug, Clone)]
pub struct SyncPlanItem {
    pub key: (i64, String),
    pub action: SyncAction,
    /// The desired payload (None for `Delete`).
    pub desired: Option<serde_json::Value>,
    /// The existing command id, when one exists.
    pub existing_id: Option<String>,
}

/// Build the command key `(type, lowercased name)` for a desired payload.
fn desired_key(payload: &serde_json::Value) -> (i64, String) {
    let raw_type = as_i64(payload.get("type"), 1);
    let t = if raw_type == 0 { 1 } else { raw_type };
    let name = as_str(payload.get("name")).to_lowercase();
    (t, name)
}

/// Build the command key from an existing command payload (already merged with
/// the attribute fields by the caller, as Python's
/// `_existing_command_to_payload` does).
fn existing_key(payload: &serde_json::Value) -> (i64, String) {
    let raw_type = as_i64(payload.get("type"), 1);
    let t = if raw_type == 0 { 1 } else { raw_type };
    let name = as_str(payload.get("name")).to_lowercase();
    (t, name)
}

/// Diff desired vs. existing slash commands and produce a plan + summary.
///
/// Faithful port of the `_safe_sync_slash_commands` reconcile loop. Each
/// existing command is a `(id, payload)` pair where the payload has already had
/// the attribute-only fields (`nsfw`, `dm_permission`,
/// `default_member_permissions`) merged in (Python `_existing_command_to_payload`).
///
/// Note: like the Python original, this preserves first-seen ordering for
/// desired commands and de-duplicates by key (last write wins for the desired
/// map, matching the dict-comprehension semantics).
pub fn reconcile_slash_commands(
    desired_payloads: &[serde_json::Value],
    existing: &[(String, serde_json::Value)],
) -> (Vec<SyncPlanItem>, SyncSummary) {
    // Build desired-by-key (dict comprehension: last write wins).
    let mut desired_by_key: HashMap<(i64, String), serde_json::Value> = HashMap::new();
    let mut desired_order: Vec<(i64, String)> = Vec::new();
    for payload in desired_payloads {
        let key = desired_key(payload);
        if !desired_by_key.contains_key(&key) {
            desired_order.push(key.clone());
        }
        desired_by_key.insert(key, payload.clone());
    }

    let mut existing_by_key: HashMap<(i64, String), (String, serde_json::Value)> = HashMap::new();
    for (id, payload) in existing {
        let key = existing_key(payload);
        existing_by_key.insert(key, (id.clone(), payload.clone()));
    }

    let mut plan: Vec<SyncPlanItem> = Vec::new();
    let mut summary = SyncSummary {
        total: desired_payloads.len(),
        ..Default::default()
    };

    for key in &desired_order {
        let desired = desired_by_key.get(key).unwrap();
        match existing_by_key.remove(key) {
            None => {
                summary.created += 1;
                plan.push(SyncPlanItem {
                    key: key.clone(),
                    action: SyncAction::Create,
                    desired: Some(desired.clone()),
                    existing_id: None,
                });
            }
            Some((id, existing_payload)) => {
                let current_canon = canonicalize_app_command_payload(&existing_payload);
                let desired_canon = canonicalize_app_command_payload(desired);
                if current_canon == desired_canon {
                    summary.unchanged += 1;
                    plan.push(SyncPlanItem {
                        key: key.clone(),
                        action: SyncAction::Unchanged,
                        desired: Some(desired.clone()),
                        existing_id: Some(id),
                    });
                    continue;
                }
                if patchable_app_command_payload(&existing_payload)
                    == patchable_app_command_payload(desired)
                {
                    summary.recreated += 1;
                    plan.push(SyncPlanItem {
                        key: key.clone(),
                        action: SyncAction::Recreate,
                        desired: Some(desired.clone()),
                        existing_id: Some(id),
                    });
                    continue;
                }
                summary.updated += 1;
                plan.push(SyncPlanItem {
                    key: key.clone(),
                    action: SyncAction::Update,
                    desired: Some(desired.clone()),
                    existing_id: Some(id),
                });
            }
        }
    }

    // Remaining existing commands are orphans → delete.
    for (key, (id, _payload)) in existing_by_key.into_iter() {
        summary.deleted += 1;
        plan.push(SyncPlanItem {
            key,
            action: SyncAction::Delete,
            desired: None,
            existing_id: Some(id),
        });
    }

    (plan, summary)
}

// ─── Voice message helpers ──────────────────────────────────────────────────

/// Compute the fallback voice-message duration (seconds) the way the Python
/// `send_voice` path does when mutagen is unavailable:
/// `max(1.0, len(file_data) / 2000.0)`.
pub fn fallback_voice_duration_secs(file_len: usize) -> f64 {
    (file_len as f64 / 2000.0).max(1.0)
}

/// Build the flat 256-byte (all 0x80) waveform used by the native voice-message
/// upload path.
pub fn voice_message_waveform() -> Vec<u8> {
    vec![0x80u8; 256]
}

// ─── VoiceReceiver RTP parsing ──────────────────────────────────────────────

/// Outcome of attempting to parse a raw inbound UDP packet as RTP voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtpParse {
    /// Packet too short, not RTP, or the bot's own audio — discard.
    Skip,
    /// A parsed RTP voice packet.
    Packet(RtpPacket),
}

/// Parsed RTP voice packet header info (subset used by [`VoiceReceiver`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpPacket {
    pub seq: u16,
    pub timestamp: u32,
    pub ssrc: u32,
    pub header_size: usize,
    pub ext_data_len: usize,
    pub has_padding: bool,
    /// The bytes `[header_size..]` (payload including the 4-byte nonce suffix).
    pub payload_with_nonce_len: usize,
}

/// Parse a raw UDP packet's RTP header, returning [`RtpParse::Skip`] for any of
/// the cases the Python `_on_packet` early-returns on (before NaCl decrypt).
///
/// `bot_ssrc` lets us skip the bot's own audio. This mirrors the header-sizing
/// logic exactly: version-2 check, payload-type 0x78, dynamic header size from
/// CSRC count + extension bit, and extension data length read from the
/// preamble.
pub fn parse_rtp_header(data: &[u8], bot_ssrc: u32) -> RtpParse {
    if data.len() < 16 {
        return RtpParse::Skip;
    }
    // RTP version: top 2 bits of byte 0 must be 2. Payload type (byte1 & 0x7F)
    // must be 0x78 (120) for voice.
    if (data[0] >> 6) != 2 || (data[1] & 0x7F) != 0x78 {
        return RtpParse::Skip;
    }

    let first_byte = data[0];
    // struct.unpack_from(">BBHII", data, 0): byte, byte, u16, u32, u32
    let seq = u16::from_be_bytes([data[2], data[3]]);
    let timestamp = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let ssrc = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

    if ssrc == bot_ssrc {
        return RtpParse::Skip;
    }

    let cc = (first_byte & 0x0F) as usize; // CSRC count
    let has_extension = (first_byte & 0x10) != 0;
    let has_padding = (first_byte & 0x20) != 0;
    let header_size = 12 + (4 * cc) + if has_extension { 4 } else { 0 };

    if data.len() < header_size + 4 {
        return RtpParse::Skip;
    }

    let mut ext_data_len = 0usize;
    if has_extension {
        let ext_preamble_offset = 12 + (4 * cc);
        // struct.unpack_from(">H", data, ext_preamble_offset + 2)
        let idx = ext_preamble_offset + 2;
        if idx + 2 > data.len() {
            return RtpParse::Skip;
        }
        let ext_words = u16::from_be_bytes([data[idx], data[idx + 1]]) as usize;
        ext_data_len = ext_words * 4;
    }

    let payload_with_nonce_len = data.len() - header_size;
    // Python: if len(payload_with_nonce) < 4: return
    if payload_with_nonce_len < 4 {
        return RtpParse::Skip;
    }

    RtpParse::Packet(RtpPacket {
        seq,
        timestamp,
        ssrc,
        header_size,
        ext_data_len,
        has_padding,
        payload_with_nonce_len,
    })
}

/// Build the 24-byte NaCl nonce from a payload's trailing 4 bytes, and return
/// the encrypted body. Mirrors the Python nonce construction:
/// `nonce[:4] = payload_with_nonce[-4:]`, `encrypted = payload_with_nonce[:-4]`.
pub fn build_nacl_nonce(payload_with_nonce: &[u8]) -> Option<([u8; 24], Vec<u8>)> {
    if payload_with_nonce.len() < 4 {
        return None;
    }
    let mut nonce = [0u8; 24];
    let tail = &payload_with_nonce[payload_with_nonce.len() - 4..];
    nonce[..4].copy_from_slice(tail);
    let encrypted = payload_with_nonce[..payload_with_nonce.len() - 4].to_vec();
    Some((nonce, encrypted))
}

/// Strip RTP padding from a decrypted payload (RFC 3550 §5.1).
///
/// Returns `Some(stripped)` on success or `None` for any of the invalid cases
/// Python early-returns on (empty payload, zero / over-long pad length, or the
/// padding consuming the entire payload).
pub fn strip_rtp_padding(decrypted: &[u8], has_padding: bool) -> Option<Vec<u8>> {
    if !has_padding {
        return Some(decrypted.to_vec());
    }
    if decrypted.is_empty() {
        return None;
    }
    let pad_len = *decrypted.last().unwrap() as usize;
    if pad_len == 0 || pad_len > decrypted.len() {
        return None;
    }
    let stripped = &decrypted[..decrypted.len() - pad_len];
    if stripped.is_empty() {
        return None;
    }
    Some(stripped.to_vec())
}

/// Skip encrypted extension data ahead of the opus payload, matching the Python
/// `if ext_data_len and len(decrypted) > ext_data_len:` guard.
pub fn skip_extension_data(decrypted: &[u8], ext_data_len: usize) -> Vec<u8> {
    if ext_data_len != 0 && decrypted.len() > ext_data_len {
        decrypted[ext_data_len..].to_vec()
    } else {
        decrypted.to_vec()
    }
}

/// Per-user voice buffering + silence detection (deterministic core of
/// `VoiceReceiver`).
///
/// The decrypt/decode and discord.py socket-listener wiring are omitted (they
/// need NaCl, DAVE, opus, and a live voice client); the buffering and silence
/// thresholds are ported exactly so the listen loop's `check_silence` semantics
/// can be exercised and reused.
#[derive(Debug)]
pub struct VoiceReceiver {
    pub bot_ssrc: u32,
    pub allowed_user_ids: HashSet<String>,
    pub running: bool,
    pub paused: bool,
    ssrc_to_user: HashMap<u32, i64>,
    buffers: HashMap<u32, Vec<u8>>,
    /// Per-SSRC last packet time (monotonic seconds).
    last_packet_time: HashMap<u32, f64>,
}

impl VoiceReceiver {
    /// Seconds of silence that mark the end of an utterance.
    pub const SILENCE_THRESHOLD: f64 = 1.5;
    /// Minimum buffered duration (seconds) to process; below this is noise.
    pub const MIN_SPEECH_DURATION: f64 = 0.5;
    /// Discord native sample rate.
    pub const SAMPLE_RATE: usize = 48000;
    /// Discord sends stereo.
    pub const CHANNELS: usize = 2;

    pub fn new(bot_ssrc: u32, allowed_user_ids: HashSet<String>) -> Self {
        Self {
            bot_ssrc,
            allowed_user_ids,
            running: false,
            paused: false,
            ssrc_to_user: HashMap::new(),
            buffers: HashMap::new(),
            last_packet_time: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        self.running = true;
    }

    pub fn stop(&mut self) {
        self.running = false;
        self.buffers.clear();
        self.last_packet_time.clear();
        self.ssrc_to_user.clear();
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Map an SSRC to a user id (from a SPEAKING event).
    pub fn map_ssrc(&mut self, ssrc: u32, user_id: i64) {
        self.ssrc_to_user.insert(ssrc, user_id);
    }

    /// Append decoded PCM for an SSRC and record the packet time.
    pub fn append_pcm(&mut self, ssrc: u32, pcm: &[u8], now: f64) {
        self.buffers.entry(ssrc).or_default().extend_from_slice(pcm);
        self.last_packet_time.insert(ssrc, now);
    }

    /// Bytes-per-second for 48kHz, 16-bit, stereo PCM.
    pub fn bytes_per_second() -> usize {
        Self::SAMPLE_RATE * Self::CHANNELS * 2
    }

    /// Return completed utterances `(user_id, pcm)` and flush/discard buffers,
    /// faithfully porting `check_silence`. `infer_user` resolves an unmapped
    /// SSRC the way `_infer_user_for_ssrc` would (returns 0 when no inference
    /// is possible); when it returns nonzero the mapping is recorded.
    pub fn check_silence<F>(&mut self, now: f64, mut infer_user: F) -> Vec<(i64, Vec<u8>)>
    where
        F: FnMut(u32) -> i64,
    {
        let mut completed: Vec<(i64, Vec<u8>)> = Vec::new();
        let bps = Self::bytes_per_second() as f64;

        let ssrc_list: Vec<u32> = self.buffers.keys().copied().collect();
        for ssrc in ssrc_list {
            let last_time = *self.last_packet_time.get(&ssrc).unwrap_or(&now);
            let silence = now - last_time;
            let buf_len = self.buffers.get(&ssrc).map(|b| b.len()).unwrap_or(0);
            let buf_duration = buf_len as f64 / bps;

            if silence >= Self::SILENCE_THRESHOLD && buf_duration >= Self::MIN_SPEECH_DURATION {
                let mut user_id = self.ssrc_to_user.get(&ssrc).copied().unwrap_or(0);
                if user_id == 0 {
                    user_id = infer_user(ssrc);
                    if user_id != 0 {
                        self.ssrc_to_user.insert(ssrc, user_id);
                    }
                }
                if user_id != 0 {
                    if let Some(buf) = self.buffers.get(&ssrc) {
                        completed.push((user_id, buf.clone()));
                    }
                }
                // Reset buffer + drop the packet time (mirrors Python).
                self.buffers.insert(ssrc, Vec::new());
                self.last_packet_time.remove(&ssrc);
            } else if silence >= Self::SILENCE_THRESHOLD * 2.0 {
                // Stale buffer with no valid user — discard.
                self.buffers.remove(&ssrc);
                self.last_packet_time.remove(&ssrc);
            }
        }

        completed
    }

    /// SSRCs considered "speaking" — audio received within the last 2 seconds
    /// (used by `get_voice_channel_info`). Returns the mapped user ids.
    pub fn speaking_user_ids(&self, now: f64) -> HashSet<i64> {
        let mut out = HashSet::new();
        for (ssrc, last_t) in &self.last_packet_time {
            if now - *last_t < 2.0 {
                if let Some(uid) = self.ssrc_to_user.get(ssrc) {
                    out.insert(*uid);
                }
            }
        }
        out
    }
}

// ─── Voice channel context formatting ───────────────────────────────────────

/// A member view for voice-channel context rendering.
#[derive(Debug, Clone)]
pub struct VoiceMember {
    pub user_id: i64,
    pub display_name: String,
    pub is_bot: bool,
    pub is_speaking: bool,
}

/// Render the human-readable voice channel context string (Python
/// `get_voice_channel_context`). Returns an empty string when there is no
/// channel info.
pub fn voice_channel_context(channel_name: Option<&str>, members: &[VoiceMember]) -> String {
    let channel_name = match channel_name {
        Some(n) => n,
        None => return String::new(),
    };
    let mut parts = vec![format!(
        "[Voice channel: #{} — {} participant(s)]",
        channel_name,
        members.len()
    )];
    for m in members {
        let status = if m.is_speaking { " (speaking)" } else { "" };
        parts.push(format!("  - {}{}", m.display_name, status));
    }
    parts.join("\n")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-mutating tests so they don't race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn clean_id_strips_mention_and_prefix() {
        assert_eq!(clean_discord_id("  <@123>  "), "123");
        assert_eq!(clean_discord_id("<@!456>"), "456");
        assert_eq!(clean_discord_id("user:789"), "789");
        assert_eq!(clean_discord_id("USER:abc"), "abc");
        assert_eq!(clean_discord_id("plainname"), "plainname");
        // Python's lstrip("<@!") does NOT strip '&', so a role mention keeps it.
        assert_eq!(clean_discord_id("<@&999>"), "&999");
    }

    #[test]
    fn allowed_mentions_defaults() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::remove_var("DISCORD_ALLOW_MENTION_EVERYONE");
            env::remove_var("DISCORD_ALLOW_MENTION_ROLES");
            env::remove_var("DISCORD_ALLOW_MENTION_USERS");
            env::remove_var("DISCORD_ALLOW_MENTION_REPLIED_USER");
        }
        let m = build_allowed_mentions();
        assert_eq!(m, AllowedMentions::default());
        assert!(!m.everyone);
        assert!(m.users);
    }

    #[test]
    fn allowed_mentions_override() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::set_var("DISCORD_ALLOW_MENTION_EVERYONE", "yes");
            env::set_var("DISCORD_ALLOW_MENTION_USERS", "false");
        }
        let m = build_allowed_mentions();
        assert!(m.everyone);
        assert!(!m.users);
        unsafe {
            env::remove_var("DISCORD_ALLOW_MENTION_EVERYONE");
            env::remove_var("DISCORD_ALLOW_MENTION_USERS");
        }
    }

    #[test]
    fn reply_mode_chunk_logic() {
        assert_eq!(ReplyToMode::from_str_or_default(None), ReplyToMode::First);
        assert_eq!(ReplyToMode::from_str_or_default(Some("ALL")), ReplyToMode::All);
        assert_eq!(ReplyToMode::from_str_or_default(Some("off")), ReplyToMode::Off);
        assert_eq!(ReplyToMode::from_str_or_default(Some("weird")), ReplyToMode::First);

        assert!(ReplyToMode::First.reference_for_chunk(0));
        assert!(!ReplyToMode::First.reference_for_chunk(1));
        assert!(ReplyToMode::All.reference_for_chunk(3));
        assert!(!ReplyToMode::Off.reference_for_chunk(0));
    }

    #[test]
    fn command_sync_policy_fallback() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("DISCORD_COMMAND_SYNC_POLICY", "BULK"); }
        assert_eq!(command_sync_policy(), "bulk");
        unsafe { env::set_var("DISCORD_COMMAND_SYNC_POLICY", "garbage"); }
        assert_eq!(command_sync_policy(), "safe");
        unsafe { env::remove_var("DISCORD_COMMAND_SYNC_POLICY"); }
        assert_eq!(command_sync_policy(), "safe");
    }

    #[test]
    fn require_mention_config_and_env() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("DISCORD_REQUIRE_MENTION"); }
        // Config string "false" → not required.
        assert!(!require_mention(Some(&ConfiguredFlag::Str("false".into()))));
        assert!(require_mention(Some(&ConfiguredFlag::Str("yes".into()))));
        assert!(!require_mention(Some(&ConfiguredFlag::Bool(false))));
        // No config → env default true.
        assert!(require_mention(None));
        unsafe { env::set_var("DISCORD_REQUIRE_MENTION", "off"); }
        assert!(!require_mention(None));
        unsafe { env::remove_var("DISCORD_REQUIRE_MENTION"); }
    }

    #[test]
    fn free_response_channels_forms() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("DISCORD_FREE_RESPONSE_CHANNELS"); }
        let list = FreeResponseConfig::List(vec!["1".into(), " 2 ".into(), "".into()]);
        let s = free_response_channels(Some(&list));
        assert!(s.contains("1") && s.contains("2") && s.len() == 2);

        let scalar = FreeResponseConfig::Scalar("10,20, 30".into());
        let s = free_response_channels(Some(&scalar));
        assert!(s.contains("30") && s.len() == 3);

        unsafe { env::set_var("DISCORD_FREE_RESPONSE_CHANNELS", "*"); }
        let s = free_response_channels(None);
        assert!(s.contains("*"));
        unsafe { env::remove_var("DISCORD_FREE_RESPONSE_CHANNELS"); }
    }

    #[test]
    fn allow_bots_parsing() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("DISCORD_ALLOW_BOTS"); }
        assert_eq!(allow_bots_mode(), AllowBotsMode::None);
        unsafe { env::set_var("DISCORD_ALLOW_BOTS", "Mentions"); }
        assert_eq!(allow_bots_mode(), AllowBotsMode::Mentions);
        unsafe { env::set_var("DISCORD_ALLOW_BOTS", "all"); }
        assert_eq!(allow_bots_mode(), AllowBotsMode::All);
        unsafe { env::remove_var("DISCORD_ALLOW_BOTS"); }
    }

    #[test]
    fn auto_thread_name_derivation() {
        assert_eq!(derive_auto_thread_name("<@123> hello   there"), "hello there");
        assert_eq!(derive_auto_thread_name("<@!1> <#2> <@&3>"), "Hermes");
        assert_eq!(derive_auto_thread_name(""), "Hermes");
        let long = "a".repeat(100);
        let name = derive_auto_thread_name(&long);
        assert_eq!(name.chars().count(), 80);
        assert!(name.ends_with("..."));
    }

    #[test]
    fn forum_and_topic() {
        assert!(is_forum_parent_type(Some(15)));
        assert!(!is_forum_parent_type(Some(0)));
        assert!(!is_forum_parent_type(None));

        // Thread with no topic inherits forum parent topic.
        assert_eq!(
            effective_topic(None, true, true, Some("forum desc")),
            Some("forum desc".to_string())
        );
        // Direct topic wins.
        assert_eq!(
            effective_topic(Some("chan topic"), true, true, Some("forum desc")),
            Some("chan topic".to_string())
        );
        // Non-thread with no topic → None.
        assert_eq!(effective_topic(None, false, false, None), None);
    }

    #[test]
    fn thread_chat_name_variants() {
        assert_eq!(
            format_thread_chat_name(Some("T"), None, Some("P"), true, Some("G")),
            "G / P / T"
        );
        assert_eq!(
            format_thread_chat_name(Some("T"), None, Some("P"), false, Some("G")),
            "G / #P / T"
        );
        assert_eq!(
            format_thread_chat_name(Some("T"), None, Some("P"), false, None),
            "P / T"
        );
        assert_eq!(
            format_thread_chat_name(None, Some("99"), None, false, None),
            "99"
        );
    }

    #[test]
    fn is_allowed_user_semantics() {
        let empty_u: HashSet<String> = HashSet::new();
        let empty_r: HashSet<i64> = HashSet::new();
        // No allowlists → everyone allowed.
        assert!(is_allowed_user("1", &empty_u, &empty_r, None, &[]));

        let users: HashSet<String> = ["42".to_string()].into_iter().collect();
        assert!(is_allowed_user("42", &users, &empty_r, None, &[]));
        assert!(!is_allowed_user("99", &users, &empty_r, None, &[]));

        let roles: HashSet<i64> = [7i64].into_iter().collect();
        let author = AuthorView { id: Some(5), roles: Some(vec![7]) };
        assert!(is_allowed_user("5", &empty_u, &roles, Some(&author), &[]));
        // Role via mutual-guild scan.
        assert!(is_allowed_user("5", &empty_u, &roles, None, &[7]));
        assert!(!is_allowed_user("5", &empty_u, &roles, None, &[8]));
    }

    #[test]
    fn component_auth_fail_closed_on_dm_role_policy() {
        let empty_u: HashSet<String> = HashSet::new();
        let roles: HashSet<i64> = [3i64].into_iter().collect();
        // Role policy active but user has no role list → reject.
        let user = AuthorView { id: Some(1), roles: None };
        assert!(!component_check_auth(Some(&user), &empty_u, &roles));
        // With role → allow.
        let user2 = AuthorView { id: Some(1), roles: Some(vec![3]) };
        assert!(component_check_auth(Some(&user2), &empty_u, &roles));
        // No allowlists → allow.
        assert!(component_check_auth(None, &empty_u, &HashSet::new()));
        // Missing user with allowlist → reject.
        let users: HashSet<String> = ["1".into()].into_iter().collect();
        assert!(!component_check_auth(None, &users, &HashSet::new()));
    }

    #[test]
    fn slash_auth_channel_and_user_gates() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::set_var("DISCORD_ALLOWED_CHANNELS", "100,200");
            env::remove_var("DISCORD_IGNORED_CHANNELS");
        }
        let users: HashSet<String> = ["5".into()].into_iter().collect();
        let roles: HashSet<i64> = HashSet::new();

        // Allowed channel + allowed user → allow.
        let ix = SlashInteractionView {
            in_dm: false,
            channel_id: Some("100".into()),
            thread_parent_id: None,
            user: Some(AuthorView { id: Some(5), roles: None }),
        };
        assert!(evaluate_slash_authorization(&ix, &users, &roles, &[]).allowed);

        // Channel not allowed → reject.
        let ix2 = SlashInteractionView {
            channel_id: Some("999".into()),
            ..ix.clone()
        };
        let r = evaluate_slash_authorization(&ix2, &users, &roles, &[]);
        assert!(!r.allowed);
        assert_eq!(r.reason.as_deref(), Some("channel not in DISCORD_ALLOWED_CHANNELS"));

        // Allowed channel but wrong user → reject.
        let ix3 = SlashInteractionView {
            user: Some(AuthorView { id: Some(6), roles: None }),
            ..ix.clone()
        };
        assert!(!evaluate_slash_authorization(&ix3, &users, &roles, &[]).allowed);

        // Missing channel id with channel policy → fail closed.
        let ix4 = SlashInteractionView {
            channel_id: None,
            ..ix.clone()
        };
        let r = evaluate_slash_authorization(&ix4, &users, &roles, &[]);
        assert!(!r.allowed);
        assert!(r.reason.unwrap().contains("channel id missing"));

        unsafe { env::remove_var("DISCORD_ALLOWED_CHANNELS"); }
    }

    #[test]
    fn slash_auth_ignored_beats_allowed() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::set_var("DISCORD_ALLOWED_CHANNELS", "100");
            env::set_var("DISCORD_IGNORED_CHANNELS", "100");
        }
        let ix = SlashInteractionView {
            in_dm: false,
            channel_id: Some("100".into()),
            thread_parent_id: None,
            user: Some(AuthorView { id: Some(5), roles: None }),
        };
        let r = evaluate_slash_authorization(&ix, &HashSet::new(), &HashSet::new(), &[]);
        assert!(!r.allowed);
        assert_eq!(r.reason.as_deref(), Some("channel in DISCORD_IGNORED_CHANNELS"));
        unsafe {
            env::remove_var("DISCORD_ALLOWED_CHANNELS");
            env::remove_var("DISCORD_IGNORED_CHANNELS");
        }
    }

    #[test]
    fn canonicalize_payload_normalizes() {
        let payload = serde_json::json!({
            "type": 1,
            "name": "Ask",
            "description": "do it",
            "default_member_permissions": 8,
            "contexts": [2, 0, 1],
            "options": [
                {"type": 3, "name": "q", "description": "query", "required": true}
            ],
        });
        let canon = canonicalize_app_command_payload(&payload);
        assert_eq!(canon["type"], serde_json::json!(1));
        assert_eq!(canon["default_member_permissions"], serde_json::json!("8"));
        assert_eq!(canon["contexts"], serde_json::json!([0, 1, 2]));
        assert_eq!(canon["dm_permission"], serde_json::json!(true));
        assert_eq!(canon["options"][0]["required"], serde_json::json!(true));
        assert_eq!(canon["integration_types"], serde_json::Value::Null);
    }

    #[test]
    fn normalize_permissions_str_vs_int() {
        assert_eq!(normalize_permissions(None), None);
        assert_eq!(
            normalize_permissions(Some(&serde_json::json!(8))),
            Some("8".to_string())
        );
        assert_eq!(
            normalize_permissions(Some(&serde_json::json!("8"))),
            Some("8".to_string())
        );
    }

    #[test]
    fn reconcile_create_update_recreate_delete_unchanged() {
        let desired = vec![
            serde_json::json!({"type": 1, "name": "new", "description": "n"}),
            serde_json::json!({"type": 1, "name": "same", "description": "d"}),
            serde_json::json!({"type": 1, "name": "edited", "description": "new desc",
                               "nsfw": true}),
        ];
        let existing = vec![
            // unchanged
            ("id-same".to_string(),
             serde_json::json!({"type": 1, "name": "same", "description": "d"})),
            // patchable diff (only desc/options differ, perms identical) → recreate?
            // Here description differs but so do other canonical fields (nsfw),
            // so this is an update.
            ("id-edited".to_string(),
             serde_json::json!({"type": 1, "name": "edited", "description": "old desc",
                                "nsfw": false})),
            // orphan → delete
            ("id-old".to_string(),
             serde_json::json!({"type": 1, "name": "orphan", "description": "x"})),
        ];

        let (plan, summary) = reconcile_slash_commands(&desired, &existing);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.created, 1);
        assert_eq!(summary.unchanged, 1);
        assert_eq!(summary.updated, 1);
        assert_eq!(summary.deleted, 1);

        let created: Vec<_> = plan.iter().filter(|p| p.action == SyncAction::Create).collect();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].key.1, "new");

        let deleted: Vec<_> = plan.iter().filter(|p| p.action == SyncAction::Delete).collect();
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0].existing_id.as_deref(), Some("id-old"));
    }

    #[test]
    fn reconcile_recreate_path() {
        // Same patchable payload (name/description/options) but a non-patchable
        // field (nsfw) is identical so canonical equality holds → unchanged.
        // To force recreate, the patchable fields must match while a
        // non-patchable canonical field differs.
        let desired = vec![serde_json::json!({
            "type": 1, "name": "cmd", "description": "d", "nsfw": true,
        })];
        let existing = vec![(
            "id-1".to_string(),
            serde_json::json!({"type": 1, "name": "cmd", "description": "d", "nsfw": false}),
        )];
        let (plan, summary) = reconcile_slash_commands(&desired, &existing);
        assert_eq!(summary.recreated, 1);
        assert_eq!(plan[0].action, SyncAction::Recreate);
    }

    #[test]
    fn rtp_parse_basic() {
        // Version 2 (0x80), payload type 0x78.
        let mut data = vec![0u8; 20];
        data[0] = 0x80; // version 2, no cc/ext/padding
        data[1] = 0x78;
        // seq=1, ts=2, ssrc=3
        data[2..4].copy_from_slice(&1u16.to_be_bytes());
        data[4..8].copy_from_slice(&2u32.to_be_bytes());
        data[8..12].copy_from_slice(&3u32.to_be_bytes());
        match parse_rtp_header(&data, 999) {
            RtpParse::Packet(p) => {
                assert_eq!(p.ssrc, 3);
                assert_eq!(p.seq, 1);
                assert_eq!(p.header_size, 12);
                assert_eq!(p.ext_data_len, 0);
                assert!(!p.has_padding);
            }
            RtpParse::Skip => panic!("expected packet"),
        }

        // Bot's own SSRC → skip.
        assert_eq!(parse_rtp_header(&data, 3), RtpParse::Skip);

        // Wrong payload type → skip.
        let mut bad = data.clone();
        bad[1] = 0x00;
        assert_eq!(parse_rtp_header(&bad, 999), RtpParse::Skip);

        // Too short → skip.
        assert_eq!(parse_rtp_header(&data[..10], 999), RtpParse::Skip);
    }

    #[test]
    fn rtp_extension_header_size() {
        let mut data = vec![0u8; 24];
        data[0] = 0x90; // version 2, extension bit set, cc=0
        data[1] = 0x78;
        // ssrc must be nonzero so it isn't mistaken for the bot's own audio.
        data[8..12].copy_from_slice(&5u32.to_be_bytes());
        // extension preamble at offset 12: profile (2 bytes) + length (2 bytes)
        // ext_words at offset 14
        data[14..16].copy_from_slice(&1u16.to_be_bytes()); // 1 word = 4 bytes
        match parse_rtp_header(&data, 0) {
            RtpParse::Packet(p) => {
                assert_eq!(p.header_size, 16); // 12 + 4 (ext)
                assert_eq!(p.ext_data_len, 4);
            }
            RtpParse::Skip => panic!("expected packet"),
        }
    }

    #[test]
    fn nacl_nonce_and_padding() {
        let payload = vec![1u8, 2, 3, 4, 0xAA, 0xBB, 0xCC, 0xDD];
        let (nonce, enc) = build_nacl_nonce(&payload).unwrap();
        assert_eq!(&nonce[..4], &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(nonce[4..], [0u8; 20]);
        assert_eq!(enc, vec![1, 2, 3, 4]);

        // Padding strip: last byte is pad length.
        let decrypted = vec![1u8, 2, 3, 2]; // pad_len=2 → keep [1,2]
        assert_eq!(strip_rtp_padding(&decrypted, true), Some(vec![1, 2]));
        // No padding flag → unchanged.
        assert_eq!(strip_rtp_padding(&decrypted, false), Some(decrypted.clone()));
        // Invalid pad length.
        assert_eq!(strip_rtp_padding(&[5u8], true), None);
        // Padding consumes entire payload.
        assert_eq!(strip_rtp_padding(&[1u8], true), None);
    }

    #[test]
    fn skip_extension_data_logic() {
        let d = vec![1u8, 2, 3, 4, 5];
        assert_eq!(skip_extension_data(&d, 0), d);
        assert_eq!(skip_extension_data(&d, 2), vec![3, 4, 5]);
        // ext_data_len >= len → unchanged.
        assert_eq!(skip_extension_data(&d, 5), d);
    }

    #[test]
    fn voice_receiver_silence_detection() {
        let mut vr = VoiceReceiver::new(0, HashSet::new());
        vr.start();
        vr.map_ssrc(1, 42);
        // Fill enough PCM for > MIN_SPEECH_DURATION (0.5s).
        let bps = VoiceReceiver::bytes_per_second();
        let pcm = vec![0u8; bps]; // 1 second of audio
        vr.append_pcm(1, &pcm, 0.0);

        // Not silent yet.
        let done = vr.check_silence(1.0, |_| 0);
        assert!(done.is_empty());

        // After SILENCE_THRESHOLD → completed.
        let done = vr.check_silence(0.0 + 2.0, |_| 0);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].0, 42);
        assert_eq!(done[0].1.len(), bps);
    }

    #[test]
    fn voice_receiver_infers_unmapped_ssrc() {
        let mut vr = VoiceReceiver::new(0, HashSet::new());
        vr.start();
        // No SSRC mapping; provide enough audio.
        let bps = VoiceReceiver::bytes_per_second();
        vr.append_pcm(7, &vec![0u8; bps], 0.0);
        let done = vr.check_silence(2.0, |ssrc| if ssrc == 7 { 88 } else { 0 });
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].0, 88);
    }

    #[test]
    fn voice_context_rendering() {
        let members = vec![
            VoiceMember { user_id: 1, display_name: "Alice".into(), is_bot: false, is_speaking: true },
            VoiceMember { user_id: 2, display_name: "Bob".into(), is_bot: false, is_speaking: false },
        ];
        let ctx = voice_channel_context(Some("general"), &members);
        assert!(ctx.contains("[Voice channel: #general — 2 participant(s)]"));
        assert!(ctx.contains("- Alice (speaking)"));
        assert!(ctx.contains("- Bob\n") || ctx.ends_with("- Bob"));
        assert_eq!(voice_channel_context(None, &members), "");
    }

    #[test]
    fn voice_duration_and_waveform() {
        assert_eq!(fallback_voice_duration_secs(1000), 1.0); // max(1.0, 0.5)
        assert_eq!(fallback_voice_duration_secs(4000), 2.0);
        assert_eq!(voice_message_waveform().len(), 256);
        assert!(voice_message_waveform().iter().all(|&b| b == 0x80));
    }

    #[test]
    fn document_ext_resolution() {
        let mut mime: HashMap<String, String> = HashMap::new();
        mime.insert("application/pdf".into(), ".pdf".into());
        assert_eq!(document_ext_for(Some("file.TXT"), None, &mime), ".txt");
        assert_eq!(document_ext_for(None, Some("application/pdf"), &mime), ".pdf");
        assert_eq!(document_ext_for(Some("noext"), None, &mime), "");
        assert_eq!(document_ext_for(None, Some("unknown/x"), &mime), "");
    }

    #[test]
    fn text_batch_delay_selection() {
        let cfg = TextBatchConfig { delay_seconds: 0.6, split_delay_seconds: 2.0 };
        assert_eq!(cfg.delay_for(10), 0.6);
        assert_eq!(cfg.delay_for(SPLIT_THRESHOLD), 2.0);
        assert!(cfg.enabled());
        let off = TextBatchConfig { delay_seconds: 0.0, split_delay_seconds: 2.0 };
        assert!(!off.enabled());
    }
}
