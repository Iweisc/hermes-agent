//! Channel directory -- cached map of reachable channels/contacts per platform.
//!
//! Built on gateway startup, refreshed periodically (every 5 min), and saved to
//! `~/.hermes/channel_directory.json`. The `send_message` tool reads this file
//! for `action="list"` and for resolving human-friendly channel names to numeric
//! IDs.
//!
//! Native Rust port of `gateway/channel_directory.py`.
//!
//! ## Adapter integration notes
//!
//! The Python original enumerates Discord channels via the live `discord.py`
//! client object and Slack channels via async web-client pagination
//! (`users.conversations`). Those depend on live SDK client objects that do not
//! exist in the native data layer. To keep this module self-contained and
//! faithful, the adapter-specific builders here accept already-fetched data:
//!
//! - [`build_discord`] takes the enumerated guild channels/forums.
//! - [`build_slack`] takes the enumerated workspace channels.
//!
//! Both still merge in session-derived entries exactly like the Python code.
//! The pure data-shaping helpers ([`channel_target_name`],
//! [`session_entry_id`], [`session_entry_name`], [`resolve_channel_name`],
//! [`format_directory_for_display`], etc.) are ported one-to-one.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::atomic_json_write;

/// Platform names skipped for session-based discovery: infrastructure entries
/// that aren't messaging platforms.
pub const SKIP_SESSION_DISCOVERY: &[&str] = &["local", "api_server", "webhook"];

/// Absolute path to the on-disk channel directory cache.
pub fn directory_path() -> PathBuf {
    get_hermes_home().join("channel_directory.json")
}

/// Path to the sessions database read by [`build_from_sessions`].
fn sessions_path() -> PathBuf {
    get_hermes_home().join("sessions").join("sessions.json")
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// `value.lstrip("#").strip().lower()` — Python `str.lstrip("#")` removes *all*
/// leading `#` characters, then surrounding whitespace, then lowercases.
pub fn normalize_channel_query(value: &str) -> String {
    value
        .trim_start_matches('#')
        .trim()
        .to_lowercase()
}

/// Borrow a string field from a JSON object, treating non-strings / missing as
/// absent.
fn str_field<'a>(obj: &'a Value, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(Value::as_str)
}

/// Return the human-facing target label shown to users for a channel entry.
///
/// Mirrors `_channel_target_name`. The Python code does `channel["name"]`
/// (raising `KeyError` if absent); here a missing/non-string name degrades to
/// the empty string.
pub fn channel_target_name(platform_name: &str, channel: &Value) -> String {
    let name = str_field(channel, "name").unwrap_or("");

    let guild = str_field(channel, "guild");
    if platform_name == "discord" {
        // Python: `channel.get("guild")` — truthy means non-empty string.
        if guild.map(|g| !g.is_empty()).unwrap_or(false) {
            return format!("#{name}");
        }
    } else {
        // `channel.get("type")` truthy => append it.
        if let Some(ty) = str_field(channel, "type") {
            if !ty.is_empty() {
                return format!("{name} ({ty})");
            }
        }
    }
    name.to_string()
}

/// Mirror Python `_session_entry_id`.
///
/// `chat_id` is taken from origin; falsy (missing/empty/zero) => `None`.
/// If a truthy `thread_id` is present, returns `"{chat_id}:{thread_id}"`,
/// otherwise `str(chat_id)`.
pub fn session_entry_id(origin: &Value) -> Option<String> {
    let chat_id = origin.get("chat_id");
    if !is_truthy(chat_id) {
        return None;
    }
    let chat_id_str = py_str_scalar(chat_id.unwrap());

    let thread_id = origin.get("thread_id");
    if is_truthy(thread_id) {
        let thread_str = py_str_scalar(thread_id.unwrap());
        return Some(format!("{chat_id_str}:{thread_str}"));
    }
    Some(chat_id_str)
}

/// Mirror Python `_session_entry_name`.
///
/// `base_name = chat_name or user_name or str(chat_id)`. If no truthy
/// `thread_id`, returns base. Otherwise appends `" / {topic}"` where topic is
/// `chat_topic or f"topic {thread_id}"`.
pub fn session_entry_name(origin: &Value) -> String {
    let base_name = first_truthy_str(origin, &["chat_name", "user_name"])
        .unwrap_or_else(|| py_str_scalar_opt(origin.get("chat_id")));

    let thread_id = origin.get("thread_id");
    if !is_truthy(thread_id) {
        return base_name;
    }

    let topic_label = match str_field(origin, "chat_topic") {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => format!("topic {}", py_str_scalar(thread_id.unwrap())),
    };
    format!("{base_name} / {topic_label}")
}

/// Python truthiness for a JSON value as fetched via `dict.get`.
///
/// `None`/missing, empty string, `0`/`0.0`, `false`, empty array/object are
/// falsy; everything else is truthy.
fn is_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                true
            }
        }
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Render a scalar JSON value the way Python's `str()` would for the values we
/// expect here (strings, ints). Used to mimic `str(chat_id)` / interpolation.
fn py_str_scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn py_str_scalar_opt(v: Option<&Value>) -> String {
    match v {
        Some(val) => py_str_scalar(val),
        None => "None".to_string(),
    }
}

/// Return the first key whose value is a truthy string.
fn first_truthy_str(obj: &Value, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(s) = str_field(obj, k) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Build / refresh
// ---------------------------------------------------------------------------

/// Build a channel directory from pre-enumerated platform data and session
/// history, then persist it to [`directory_path`].
///
/// Faithful port of `build_channel_directory`. Because live adapter SDK client
/// objects aren't available natively, the per-platform channel lists are passed
/// in via `prebuilt` (keyed by platform value string, e.g. `"discord"` after
/// having already merged any adapter-side enumeration). Use [`build_discord`] /
/// [`build_slack`] to construct those lists.
///
/// `builtin_platform_values` is the ordered list of built-in platform value
/// strings (matching Python's `for plat in Platform`). `plugin_platform_names`
/// supplies dynamic plugin-registered platform names (Python's
/// `platform_registry.plugin_entries()`).
///
/// Returns the directory `Value` and the result of the write attempt (write
/// failures are logged-and-swallowed exactly like the Python `try/except`).
pub fn build_channel_directory(
    prebuilt: &BTreeMap<String, Vec<Value>>,
    builtin_platform_values: &[String],
    plugin_platform_names: &[String],
) -> Value {
    let mut platforms: Map<String, Value> = Map::new();

    // Seed with whatever the caller pre-built from live adapters.
    for (name, list) in prebuilt {
        platforms.insert(name.clone(), Value::Array(list.clone()));
    }

    let skip: HashSet<&str> = SKIP_SESSION_DISCOVERY.iter().copied().collect();

    // Platforms that don't support direct channel enumeration get session-based
    // discovery automatically.
    for plat_name in builtin_platform_values {
        if skip.contains(plat_name.as_str()) || platforms.contains_key(plat_name) {
            continue;
        }
        let entries = build_from_sessions(plat_name);
        platforms.insert(plat_name.clone(), Value::Array(entries));
    }

    // Include plugin-registered platforms.
    for name in plugin_platform_names {
        if !skip.contains(name.as_str()) && !platforms.contains_key(name) {
            let entries = build_from_sessions(name);
            platforms.insert(name.clone(), Value::Array(entries));
        }
    }

    let directory = json!({
        "updated_at": now_isoformat(),
        "platforms": Value::Object(platforms),
    });

    // Python uses atomic_json_write(...) with the default indent=2.
    if let Err(e) = atomic_json_write(&directory_path(), &directory, 2) {
        log::warn!("Channel directory: failed to write: {e}");
    }

    directory
}

/// Local ISO-8601 timestamp matching Python's `datetime.now().isoformat()`
/// (no timezone offset, microsecond precision).
fn now_isoformat() -> String {
    chrono::Local::now()
        .naive_local()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

/// Build the Discord channel list from pre-enumerated guild data, then merge in
/// DMs from session history.
///
/// Faithful to `_build_discord`: each `(guild_name, text_channels,
/// forum_channels)` tuple contributes `{"id","name","guild","type"}` entries
/// with `type="channel"` for text channels and `type="forum"` for forums.
/// `text_channels` / `forum_channels` are `(id, name)` pairs.
pub fn build_discord(guilds: &[DiscordGuild]) -> Vec<Value> {
    let mut channels: Vec<Value> = Vec::new();

    for guild in guilds {
        for (id, name) in &guild.text_channels {
            channels.push(json!({
                "id": id,
                "name": name,
                "guild": guild.name,
                "type": "channel",
            }));
        }
        for (id, name) in &guild.forum_channels {
            channels.push(json!({
                "id": id,
                "name": name,
                "guild": guild.name,
                "type": "forum",
            }));
        }
    }

    channels.extend(build_from_sessions("discord"));
    channels
}

/// A single Discord guild's enumerable channels.
#[derive(Debug, Clone)]
pub struct DiscordGuild {
    pub name: String,
    /// `(id, name)` pairs for text channels.
    pub text_channels: Vec<(String, String)>,
    /// `(id, name)` pairs for forum channels (type 15).
    pub forum_channels: Vec<(String, String)>,
}

/// Build the Slack channel list from pre-paginated workspace channels, then
/// merge in DM/group entries from session history.
///
/// Faithful to `_build_slack`: dedups by id, marks `type="private"` when
/// `is_private` else `"channel"`, skips entries missing id or name. If no
/// channels were enumerated at all (empty input), falls back to session
/// discovery only — matching the `if not team_clients` short-circuit.
///
/// `enumerated` is the flattened list of channel objects pulled from
/// `users.conversations` across all teams; each should carry `id`, `name`, and
/// optional `is_private`. `had_team_clients` indicates whether any workspace
/// web clients existed (Python's `team_clients` truthiness).
pub fn build_slack(enumerated: &[Value], had_team_clients: bool) -> Vec<Value> {
    if !had_team_clients {
        return build_from_sessions("slack");
    }

    let mut channels: Vec<Value> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for ch in enumerated {
        let cid = str_field(ch, "id");
        let name = str_field(ch, "name");
        // Python: `if not cid or not name or cid in seen_ids: continue`
        let (cid, name) = match (cid, name) {
            (Some(c), Some(n)) if !c.is_empty() && !n.is_empty() => (c, n),
            _ => continue,
        };
        if seen_ids.contains(cid) {
            continue;
        }
        seen_ids.insert(cid.to_string());
        let is_private = ch.get("is_private").and_then(Value::as_bool).unwrap_or(false);
        channels.push(json!({
            "id": cid,
            "name": name,
            "type": if is_private { "private" } else { "channel" },
        }));
    }

    for entry in build_from_sessions("slack") {
        let id = str_field(&entry, "id").map(|s| s.to_string());
        match &id {
            Some(i) if !seen_ids.contains(i) => {
                channels.push(entry);
                seen_ids.insert(i.clone());
            }
            None => {
                // Python: `entry.get("id")` is None → not in seen_ids set
                // (which holds strings) → appended, then None added to set.
                channels.push(entry);
            }
            _ => {}
        }
    }

    channels
}

/// Pull known channels/contacts from `sessions.json` origin data.
///
/// Faithful port of `_build_from_sessions`. Reads the sessions file, filters by
/// `origin.platform == platform_name`, dedups by [`session_entry_id`], and
/// emits `{"id","name","type","thread_id"}` entries. Any error reading/parsing
/// is logged at debug and yields `[]` (matching the broad `except`).
pub fn build_from_sessions(platform_name: &str) -> Vec<Value> {
    let path = sessions_path();
    if !path.exists() {
        return Vec::new();
    }

    let mut entries: Vec<Value> = Vec::new();

    let result: Result<(), String> = (|| {
        let raw = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let data: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let obj = match data.as_object() {
            Some(o) => o,
            // Python iterates `data.items()`; a non-dict would raise and be
            // caught, yielding [].
            None => return Err("sessions.json is not an object".to_string()),
        };

        let mut seen_ids: HashSet<String> = HashSet::new();
        for (_key, session) in obj {
            let origin = match session.get("origin") {
                Some(o) if o.is_object() => o.clone(),
                // `session.get("origin") or {}`
                _ => Value::Object(Map::new()),
            };
            if str_field(&origin, "platform") != Some(platform_name) {
                continue;
            }
            let entry_id = match session_entry_id(&origin) {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };
            if seen_ids.contains(&entry_id) {
                continue;
            }
            seen_ids.insert(entry_id.clone());

            let chat_type = match session.get("chat_type") {
                Some(v) if !v.is_null() => v.clone(),
                _ => Value::String("dm".to_string()),
            };
            let thread_id = origin.get("thread_id").cloned().unwrap_or(Value::Null);

            entries.push(json!({
                "id": entry_id,
                "name": session_entry_name(&origin),
                "type": chat_type,
                "thread_id": thread_id,
            }));
        }
        Ok(())
    })();

    if let Err(e) = result {
        log::debug!("Channel directory: failed to read sessions for {platform_name}: {e}");
        return Vec::new();
    }

    entries
}

// ---------------------------------------------------------------------------
// Read / resolve
// ---------------------------------------------------------------------------

/// Load the cached channel directory from disk, returning the default empty
/// shape on any error. Mirrors `load_directory`.
pub fn load_directory() -> Value {
    let path = directory_path();
    if !path.exists() {
        return empty_directory();
    }
    match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
        Some(v) => v,
        None => empty_directory(),
    }
}

fn empty_directory() -> Value {
    json!({"updated_at": Value::Null, "platforms": {}})
}

/// Return the channels list for a platform from a loaded directory `Value`.
fn platform_channels<'a>(directory: &'a Value, platform_name: &str) -> Vec<&'a Value> {
    directory
        .get("platforms")
        .and_then(Value::as_object)
        .and_then(|p| p.get(platform_name))
        .and_then(Value::as_array)
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

/// Return the channel `type` string (e.g. `"channel"`, `"forum"`) for
/// *chat_id*, or `None` if unknown. Mirrors `lookup_channel_type`.
pub fn lookup_channel_type(platform_name: &str, chat_id: &str) -> Option<String> {
    let directory = load_directory();
    for ch in platform_channels(&directory, platform_name) {
        if str_field(ch, "id") == Some(chat_id) {
            return str_field(ch, "type").map(|s| s.to_string());
        }
    }
    None
}

/// Resolve a human-friendly channel name to a numeric ID.
///
/// Faithful port of `resolve_channel_name`. Matching strategy
/// (case-insensitive, first match wins):
/// 0. Exact raw-id match (case-sensitive, trimmed only).
/// 1. Exact normalized name / display-label match.
/// 2. Guild-qualified `"Guild/channel"` match (Discord).
/// 3. Unambiguous prefix match.
pub fn resolve_channel_name(platform_name: &str, name: &str) -> Option<String> {
    let directory = load_directory();
    let channels = platform_channels(&directory, platform_name);
    if channels.is_empty() {
        return None;
    }

    // 0. Exact ID match — case-sensitive, trimmed.
    let raw = name.trim();
    for ch in &channels {
        if str_field(ch, "id") == Some(raw) {
            // Python returns ch["id"]; we know it's a string here.
            if let Some(id) = str_field(ch, "id") {
                return Some(id.to_string());
            }
        }
    }

    let query = normalize_channel_query(name);

    // 1. Exact name match, including display labels.
    for ch in &channels {
        let ch_name = str_field(ch, "name").unwrap_or("");
        if normalize_channel_query(ch_name) == query {
            return ch_id_string(ch);
        }
        if normalize_channel_query(&channel_target_name(platform_name, ch)) == query {
            return ch_id_string(ch);
        }
    }

    // 2. Guild-qualified match for Discord ("GuildName/channel").
    if query.contains('/') {
        // Python rsplit("/", 1)
        if let Some(idx) = query.rfind('/') {
            let guild_part = &query[..idx];
            let ch_part = &query[idx + 1..];
            for ch in &channels {
                let guild = str_field(ch, "guild")
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                let ch_name = str_field(ch, "name").unwrap_or("");
                if guild == guild_part && normalize_channel_query(ch_name) == ch_part {
                    return ch_id_string(ch);
                }
            }
        }
    }

    // 3. Partial prefix match (only if unambiguous).
    let matches: Vec<&&Value> = channels
        .iter()
        .filter(|ch| {
            let ch_name = str_field(ch, "name").unwrap_or("");
            normalize_channel_query(ch_name).starts_with(&query)
        })
        .collect();
    if matches.len() == 1 {
        return ch_id_string(matches[0]);
    }

    None
}

/// Extract a channel entry's `id` as a string (matching `ch["id"]` returns).
fn ch_id_string(ch: &Value) -> Option<String> {
    str_field(ch, "id").map(|s| s.to_string())
}

/// Format the channel directory as a human-readable list for the model.
/// Faithful port of `format_directory_for_display`.
pub fn format_directory_for_display() -> String {
    let directory = load_directory();
    let platforms_obj = directory
        .get("platforms")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    // `if not any(platforms.values())` — true (i.e. show "none") when every
    // value is falsy (empty list / empty object / null / etc.).
    let any_truthy = platforms_obj.values().any(|v| is_truthy(Some(v)));
    if !any_truthy {
        return "No messaging platforms connected or no channels discovered yet.".to_string();
    }

    let mut lines: Vec<String> = vec!["Available messaging targets:\n".to_string()];

    // `sorted(platforms.items())` — sort by platform name.
    let mut sorted_platforms: Vec<(&String, &Value)> = platforms_obj.iter().collect();
    sorted_platforms.sort_by(|a, b| a.0.cmp(b.0));

    for (plat_name, channels_val) in sorted_platforms {
        let channels = match channels_val.as_array() {
            Some(a) if !a.is_empty() => a,
            // `if not channels: continue` — empty / non-list skipped.
            _ => continue,
        };

        if plat_name == "discord" {
            // Group Discord channels by guild.
            let mut guilds: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
            let mut dms: Vec<&Value> = Vec::new();
            for ch in channels {
                match str_field(ch, "guild") {
                    Some(g) if !g.is_empty() => {
                        guilds.entry(g.to_string()).or_default().push(ch);
                    }
                    _ => dms.push(ch),
                }
            }

            // `sorted(guilds.items())` — BTreeMap already sorts by guild name.
            for (guild_name, guild_channels) in &guilds {
                lines.push(format!("Discord ({guild_name}):"));
                let mut sorted_chs: Vec<&Value> = guild_channels.clone();
                sorted_chs.sort_by(|a, b| {
                    str_field(a, "name").unwrap_or("").cmp(str_field(b, "name").unwrap_or(""))
                });
                for ch in sorted_chs {
                    lines.push(format!("  discord:{}", channel_target_name(plat_name, ch)));
                }
            }
            if !dms.is_empty() {
                lines.push("Discord (DMs):".to_string());
                for ch in &dms {
                    lines.push(format!("  discord:{}", channel_target_name(plat_name, ch)));
                }
            }
            lines.push(String::new());
        } else {
            lines.push(format!("{}:", title_case(plat_name)));
            for ch in channels {
                lines.push(format!("  {}:{}", plat_name, channel_target_name(plat_name, ch)));
            }
            lines.push(String::new());
        }
    }

    lines.push("Use these as the \"target\" parameter when sending.".to_string());
    lines.push("Bare platform name (e.g. \"telegram\") sends to home channel.".to_string());

    lines.join("\n")
}

/// Python `str.title()`: uppercase the first cased character of each
/// alphabetic run, lowercase the rest. Word boundaries are non-alphabetic
/// characters.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_channel_query() {
        assert_eq!(normalize_channel_query("##Bot-Home  "), "bot-home");
        assert_eq!(normalize_channel_query("  Engineering"), "engineering");
        assert_eq!(normalize_channel_query("#a#b"), "a#b"); // only leading # stripped
    }

    #[test]
    fn test_channel_target_name_discord_guild() {
        let ch = json!({"name": "general", "guild": "MyGuild", "type": "channel"});
        assert_eq!(channel_target_name("discord", &ch), "#general");
    }

    #[test]
    fn test_channel_target_name_discord_dm_no_guild() {
        let ch = json!({"name": "Alice", "type": "dm"});
        // discord without guild => bare name
        assert_eq!(channel_target_name("discord", &ch), "Alice");
    }

    #[test]
    fn test_channel_target_name_other_with_type() {
        let ch = json!({"name": "engineering", "type": "private"});
        assert_eq!(channel_target_name("slack", &ch), "engineering (private)");
    }

    #[test]
    fn test_channel_target_name_other_no_type() {
        let ch = json!({"name": "engineering"});
        assert_eq!(channel_target_name("slack", &ch), "engineering");
    }

    #[test]
    fn test_session_entry_id() {
        assert_eq!(session_entry_id(&json!({})), None);
        assert_eq!(session_entry_id(&json!({"chat_id": ""})), None);
        assert_eq!(
            session_entry_id(&json!({"chat_id": "123"})),
            Some("123".to_string())
        );
        assert_eq!(
            session_entry_id(&json!({"chat_id": 123})),
            Some("123".to_string())
        );
        assert_eq!(
            session_entry_id(&json!({"chat_id": "123", "thread_id": "9"})),
            Some("123:9".to_string())
        );
        // thread_id falsy (0) => ignored
        assert_eq!(
            session_entry_id(&json!({"chat_id": "123", "thread_id": 0})),
            Some("123".to_string())
        );
    }

    #[test]
    fn test_session_entry_name() {
        assert_eq!(
            session_entry_name(&json!({"chat_name": "General"})),
            "General"
        );
        assert_eq!(
            session_entry_name(&json!({"user_name": "Bob"})),
            "Bob"
        );
        assert_eq!(
            session_entry_name(&json!({"chat_id": 42})),
            "42"
        );
        assert_eq!(
            session_entry_name(&json!({"chat_name": "General", "thread_id": "7"})),
            "General / topic 7"
        );
        assert_eq!(
            session_entry_name(
                &json!({"chat_name": "General", "thread_id": "7", "chat_topic": "Bugs"})
            ),
            "General / Bugs"
        );
    }

    #[test]
    fn test_build_discord_shapes() {
        let guilds = vec![DiscordGuild {
            name: "MyGuild".to_string(),
            text_channels: vec![("100".to_string(), "general".to_string())],
            forum_channels: vec![("200".to_string(), "help".to_string())],
        }];
        // Note: build_from_sessions will read real path; in CI HERMES_HOME
        // likely points nowhere, so it returns []. We only assert prefix.
        let result = build_discord(&guilds);
        assert!(result.len() >= 2);
        assert_eq!(result[0]["type"], "channel");
        assert_eq!(result[0]["guild"], "MyGuild");
        assert_eq!(result[1]["type"], "forum");
    }

    #[test]
    fn test_build_slack_dedup_and_private() {
        let enumerated = vec![
            json!({"id": "C1", "name": "general", "is_private": false}),
            json!({"id": "C1", "name": "dup"}), // dup id skipped
            json!({"id": "C2", "name": "secret", "is_private": true}),
            json!({"id": "", "name": "bad"}),   // empty id skipped
            json!({"name": "noid"}),            // missing id skipped
        ];
        let result = build_slack(&enumerated, true);
        // At least the two real channels (sessions may add more, but those have
        // distinct ids).
        let general = result.iter().find(|c| c["id"] == "C1").unwrap();
        assert_eq!(general["type"], "channel");
        let secret = result.iter().find(|c| c["id"] == "C2").unwrap();
        assert_eq!(secret["type"], "private");
        assert!(result.iter().all(|c| c["name"] != "dup"));
        assert!(result.iter().all(|c| c["name"] != "bad"));
    }

    #[test]
    fn test_resolve_channel_name_with_temp_directory() {
        // Build an isolated HERMES_HOME with a directory file.
        let tmp = std::env::temp_dir().join(format!(
            "hermes_chandir_test_{}_{}",
            std::process::id(),
            "resolve"
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let dir_value = json!({
            "updated_at": "2026-06-03T00:00:00",
            "platforms": {
                "discord": [
                    {"id": "100", "name": "general", "guild": "MyGuild", "type": "channel"},
                    {"id": "200", "name": "random", "guild": "MyGuild", "type": "channel"},
                    {"id": "C0B0QV5434G", "name": "weird", "guild": "MyGuild", "type": "channel"}
                ],
                "slack": [
                    {"id": "S1", "name": "engineering", "type": "channel"}
                ]
            }
        });
        std::fs::write(
            tmp.join("channel_directory.json"),
            serde_json::to_string(&dir_value).unwrap(),
        )
        .unwrap();

        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }

        // 0. exact id (case-sensitive)
        assert_eq!(
            resolve_channel_name("discord", "C0B0QV5434G"),
            Some("C0B0QV5434G".to_string())
        );
        // 1. exact name
        assert_eq!(
            resolve_channel_name("discord", "general"),
            Some("100".to_string())
        );
        // 1. display-label match (#general)
        assert_eq!(
            resolve_channel_name("discord", "#general"),
            Some("100".to_string())
        );
        // 2. guild-qualified
        assert_eq!(
            resolve_channel_name("discord", "MyGuild/random"),
            Some("200".to_string())
        );
        // 3. unambiguous prefix
        assert_eq!(
            resolve_channel_name("slack", "engin"),
            Some("S1".to_string())
        );
        // ambiguous prefix on discord ("r" matches only random => still 1)
        assert_eq!(
            resolve_channel_name("discord", "ran"),
            Some("200".to_string())
        );
        // unknown
        assert_eq!(resolve_channel_name("discord", "nope"), None);

        // lookup_channel_type
        assert_eq!(
            lookup_channel_type("discord", "100"),
            Some("channel".to_string())
        );
        assert_eq!(lookup_channel_type("discord", "999"), None);

        // restore
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_format_directory_for_display_empty() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes_chandir_test_{}_{}",
            std::process::id(),
            "fmt_empty"
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("channel_directory.json"),
            serde_json::to_string(&json!({"updated_at": null, "platforms": {}})).unwrap(),
        )
        .unwrap();

        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }

        assert_eq!(
            format_directory_for_display(),
            "No messaging platforms connected or no channels discovered yet."
        );

        unsafe {
            match prev {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_format_directory_for_display_grouped() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes_chandir_test_{}_{}",
            std::process::id(),
            "fmt_grouped"
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let dir_value = json!({
            "updated_at": "2026-06-03T00:00:00",
            "platforms": {
                "discord": [
                    {"id": "100", "name": "general", "guild": "MyGuild", "type": "channel"},
                    {"id": "300", "name": "Alice", "type": "dm"}
                ],
                "slack": [
                    {"id": "S1", "name": "engineering", "type": "channel"}
                ]
            }
        });
        std::fs::write(
            tmp.join("channel_directory.json"),
            serde_json::to_string(&dir_value).unwrap(),
        )
        .unwrap();

        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }

        let out = format_directory_for_display();
        assert!(out.contains("Available messaging targets:"));
        assert!(out.contains("Discord (MyGuild):"));
        assert!(out.contains("  discord:#general"));
        assert!(out.contains("Discord (DMs):"));
        assert!(out.contains("  discord:Alice"));
        assert!(out.contains("Slack:"));
        assert!(out.contains("  slack:engineering (channel)"));
        assert!(out.contains("Bare platform name"));

        unsafe {
            match prev {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("telegram"), "Telegram");
        assert_eq!(title_case("api_server"), "Api_Server");
        assert_eq!(title_case("MATRIX"), "Matrix");
    }
}
