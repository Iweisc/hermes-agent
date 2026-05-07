use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt::{self, Display, Formatter};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use reqwest::Method;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;

use crate::HermesContext;
use crate::tools::{ToolRuntime, tool_error, tool_result};

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";
const DISCORD_TIMEOUT_SECS: u64 = 15;
const DISCORD_USER_AGENT: &str = "Hermes-Agent-Rust";
const FLAG_GATEWAY_GUILD_MEMBERS: u64 = 1 << 14;
const FLAG_GATEWAY_GUILD_MEMBERS_LIMITED: u64 = 1 << 15;
const FLAG_GATEWAY_MESSAGE_CONTENT: u64 = 1 << 18;
const FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED: u64 = 1 << 19;
const VALID_ARCHIVE_DURATIONS: &[i64] = &[60, 1440, 4320, 10080];

#[derive(Debug, Clone, Copy)]
struct ActionSpec {
    name: &'static str,
    signature: &'static str,
    summary: &'static str,
    members_intent: bool,
}

#[derive(Debug, Clone, Copy)]
struct DiscordCapabilities {
    has_members_intent: bool,
    has_message_content: bool,
    detected: bool,
}

impl DiscordCapabilities {
    fn permissive() -> Self {
        Self {
            has_members_intent: true,
            has_message_content: true,
            detected: false,
        }
    }
}

#[derive(Debug, Clone)]
struct ActionRequest {
    guild_id: String,
    channel_id: String,
    user_id: String,
    role_id: String,
    message_id: String,
    query: String,
    name: String,
    limit: i64,
    before: String,
    after: String,
    auto_archive_duration: i64,
}

#[derive(Debug, Clone)]
struct DiscordApiError {
    status: u16,
    body: String,
}

impl Display for DiscordApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Discord API error {}: {}", self.status, self.body)
    }
}

#[derive(Debug, Clone)]
enum DiscordError {
    Api(DiscordApiError),
    Message(String),
}

impl Display for DiscordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api(error) => Display::fmt(error, f),
            Self::Message(message) => f.write_str(message),
        }
    }
}

const ACTIONS: &[ActionSpec] = &[
    ActionSpec {
        name: "list_guilds",
        signature: "()",
        summary: "list servers the bot is in",
        members_intent: false,
    },
    ActionSpec {
        name: "server_info",
        signature: "(guild_id)",
        summary: "server details plus member counts",
        members_intent: false,
    },
    ActionSpec {
        name: "list_channels",
        signature: "(guild_id)",
        summary: "all channels grouped by category",
        members_intent: false,
    },
    ActionSpec {
        name: "channel_info",
        signature: "(channel_id)",
        summary: "single channel details",
        members_intent: false,
    },
    ActionSpec {
        name: "list_roles",
        signature: "(guild_id)",
        summary: "roles sorted by position",
        members_intent: false,
    },
    ActionSpec {
        name: "member_info",
        signature: "(guild_id, user_id)",
        summary: "lookup a specific member",
        members_intent: true,
    },
    ActionSpec {
        name: "search_members",
        signature: "(guild_id, query)",
        summary: "find members by name prefix",
        members_intent: true,
    },
    ActionSpec {
        name: "fetch_messages",
        signature: "(channel_id)",
        summary: "recent messages with optional before/after pagination",
        members_intent: false,
    },
    ActionSpec {
        name: "list_pins",
        signature: "(channel_id)",
        summary: "pinned messages in a channel",
        members_intent: false,
    },
    ActionSpec {
        name: "pin_message",
        signature: "(channel_id, message_id)",
        summary: "pin a message",
        members_intent: false,
    },
    ActionSpec {
        name: "unpin_message",
        signature: "(channel_id, message_id)",
        summary: "unpin a message",
        members_intent: false,
    },
    ActionSpec {
        name: "create_thread",
        signature: "(channel_id, name)",
        summary: "create a public thread with optional message anchor",
        members_intent: false,
    },
    ActionSpec {
        name: "add_role",
        signature: "(guild_id, user_id, role_id)",
        summary: "assign a role",
        members_intent: false,
    },
    ActionSpec {
        name: "remove_role",
        signature: "(guild_id, user_id, role_id)",
        summary: "remove a role",
        members_intent: false,
    },
];

const CORE_ACTIONS: &[&str] = &["fetch_messages", "search_members", "create_thread"];
const ADMIN_ACTIONS: &[&str] = &[
    "list_guilds",
    "server_info",
    "list_channels",
    "channel_info",
    "list_roles",
    "member_info",
    "list_pins",
    "pin_message",
    "unpin_message",
    "add_role",
    "remove_role",
];

static CAPABILITY_CACHE: OnceLock<Mutex<HashMap<String, DiscordCapabilities>>> = OnceLock::new();

pub fn discord_available() -> bool {
    tool_actions_available(CORE_ACTIONS)
}

pub fn discord_admin_available() -> bool {
    tool_actions_available(ADMIN_ACTIONS)
}

pub fn discord_core_schema() -> Value {
    build_schema_for_tool(CORE_ACTIONS, "discord")
}

pub fn discord_admin_schema() -> Value {
    build_schema_for_tool(ADMIN_ACTIONS, "discord_admin")
}

pub fn handle_discord(args: &Value, runtime: &ToolRuntime) -> String {
    handle_discord_tool(args, runtime, CORE_ACTIONS, "discord")
}

pub fn handle_discord_admin(args: &Value, runtime: &ToolRuntime) -> String {
    handle_discord_tool(args, runtime, ADMIN_ACTIONS, "discord_admin")
}

fn capability_cache() -> &'static Mutex<HashMap<String, DiscordCapabilities>> {
    CAPABILITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tool_actions_available(subset: &[&str]) -> bool {
    let Some(token) = discord_bot_token() else {
        return false;
    };
    let context = HermesContext::detect();
    let caps = detect_capabilities(&token);
    !available_actions(subset, caps, load_allowed_actions(&context)).is_empty()
}

fn build_schema_for_tool(subset: &[&str], tool_name: &str) -> Value {
    let context = HermesContext::detect();
    let (actions, caps) = match discord_bot_token() {
        Some(token) => {
            let caps = detect_capabilities(&token);
            let actions = available_actions(subset, caps, load_allowed_actions(&context));
            let actions = if actions.is_empty() {
                subset.to_vec()
            } else {
                actions
            };
            (actions, caps)
        }
        None => (subset.to_vec(), DiscordCapabilities::permissive()),
    };

    let manifest = ACTIONS
        .iter()
        .filter(|spec| actions.contains(&spec.name))
        .map(|spec| format!("  {}{}  - {}", spec.name, spec.signature, spec.summary))
        .collect::<Vec<_>>()
        .join("\n");

    let mut description = if tool_name == "discord_admin" {
        format!(
            "Manage a Discord server via the REST API.\n\nAvailable actions:\n{}\n\nCall list_guilds first to discover guild IDs, then list_channels for channel IDs. Runtime errors will explain missing per-server permissions such as MANAGE_ROLES.",
            manifest
        )
    } else {
        format!(
            "Read and participate in a Discord server.\n\nAvailable actions:\n{}\n\nUse the channel_id from the current conversation context. Use search_members to look up user IDs by name prefix.",
            manifest
        )
    };

    if caps.detected
        && !caps.has_message_content
        && actions
            .iter()
            .any(|name| matches!(*name, "fetch_messages" | "list_pins"))
    {
        description.push_str(
            "\n\nNOTE: The bot does not have the MESSAGE_CONTENT privileged intent. fetch_messages and list_pins can still return metadata, attachments, reactions, and pin state, but message content may be empty outside mentions or DMs.",
        );
    }

    json!({
        "name": tool_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": actions,
                },
                "guild_id": {
                    "type": "string",
                    "description": "Discord server ID."
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID."
                },
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID."
                },
                "role_id": {
                    "type": "string",
                    "description": "Discord role ID."
                },
                "message_id": {
                    "type": "string",
                    "description": "Discord message ID."
                },
                "query": {
                    "type": "string",
                    "description": "Member name prefix to search for."
                },
                "name": {
                    "type": "string",
                    "description": "New thread name."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum result count. Applies to fetch_messages and search_members."
                },
                "before": {
                    "type": "string",
                    "description": "Snowflake ID for reverse pagination in fetch_messages."
                },
                "after": {
                    "type": "string",
                    "description": "Snowflake ID for forward pagination in fetch_messages."
                },
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": VALID_ARCHIVE_DURATIONS,
                    "description": "Thread archive duration in minutes for create_thread."
                }
            },
            "required": ["action"]
        }
    })
}

fn handle_discord_tool(
    args: &Value,
    runtime: &ToolRuntime,
    valid_actions: &[&str],
    tool_name: &str,
) -> String {
    let Some(token) = discord_bot_token() else {
        return tool_error("DISCORD_BOT_TOKEN not configured.");
    };

    let action = match required_string(args, "action") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if !valid_actions.contains(&action.as_str()) {
        return tool_result(json!({
            "error": format!("Unknown action: {action}"),
            "available_actions": valid_actions,
        }));
    }

    let context =
        HermesContext::detect().with_hermes_home_env(Some(runtime.hermes_home().to_path_buf()));
    if let Some(allowlist) = load_allowed_actions(&context) {
        if !allowlist.iter().any(|value| value == &action) {
            let allowed = if allowlist.is_empty() {
                "<none>".to_string()
            } else {
                allowlist.join(", ")
            };
            return tool_error(format!(
                "Action '{action}' is disabled by config (discord.server_actions). Allowed: {allowed}"
            ));
        }
    }

    let request = match parse_action_request(args, &action) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match run_action(&action, &token, &request) {
        Ok(result) => tool_result(result),
        Err(DiscordError::Api(error)) => {
            log::warn!(target: "hermes_discord", "{} {} failed: {}", tool_name, action, error);
            if error.status == 403 {
                return tool_error(enrich_403(&action, &error.body));
            }
            tool_error(error.to_string())
        }
        Err(DiscordError::Message(error)) => {
            log::warn!(target: "hermes_discord", "{} {} failed: {}", tool_name, action, error);
            tool_error(error)
        }
    }
}

fn parse_action_request(args: &Value, action: &str) -> Result<ActionRequest, String> {
    let guild_id = optional_string(args, "guild_id")?;
    let channel_id = optional_string(args, "channel_id")?;
    let user_id = optional_string(args, "user_id")?;
    let role_id = optional_string(args, "role_id")?;
    let message_id = optional_string(args, "message_id")?;
    let query = optional_string(args, "query")?;
    let name = optional_string(args, "name")?;

    let missing = required_params(action)
        .iter()
        .copied()
        .filter(|field| match *field {
            "guild_id" => guild_id.is_empty(),
            "channel_id" => channel_id.is_empty(),
            "user_id" => user_id.is_empty(),
            "role_id" => role_id.is_empty(),
            "message_id" => message_id.is_empty(),
            "query" => query.is_empty(),
            "name" => name.is_empty(),
            _ => false,
        })
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Missing required parameters for '{action}': {}",
            missing.join(", ")
        ));
    }

    for (label, value) in [
        ("guild_id", &guild_id),
        ("channel_id", &channel_id),
        ("user_id", &user_id),
        ("role_id", &role_id),
        ("message_id", &message_id),
    ] {
        if !value.is_empty() && !is_snowflake(value) {
            return Err(format!(
                "Invalid {label}: expected a Discord snowflake string."
            ));
        }
    }

    let limit = optional_integer(args, "limit").unwrap_or(50).clamp(1, 100);
    let auto_archive_duration = optional_integer(args, "auto_archive_duration").unwrap_or(1440);
    if action == "create_thread" && !VALID_ARCHIVE_DURATIONS.contains(&auto_archive_duration) {
        return Err(format!(
            "Invalid auto_archive_duration: {auto_archive_duration}. Allowed: 60, 1440, 4320, 10080."
        ));
    }

    Ok(ActionRequest {
        guild_id,
        channel_id,
        user_id,
        role_id,
        message_id,
        query,
        name,
        limit,
        before: optional_string(args, "before")?,
        after: optional_string(args, "after")?,
        auto_archive_duration,
    })
}

fn run_action(action: &str, token: &str, request: &ActionRequest) -> Result<Value, DiscordError> {
    match action {
        "list_guilds" => list_guilds(token),
        "server_info" => server_info(token, &request.guild_id),
        "list_channels" => list_channels(token, &request.guild_id),
        "channel_info" => channel_info(token, &request.channel_id),
        "list_roles" => list_roles(token, &request.guild_id),
        "member_info" => member_info(token, &request.guild_id, &request.user_id),
        "search_members" => search_members(token, &request.guild_id, &request.query, request.limit),
        "fetch_messages" => fetch_messages(
            token,
            &request.channel_id,
            request.limit,
            empty_to_none(&request.before),
            empty_to_none(&request.after),
        ),
        "list_pins" => list_pins(token, &request.channel_id),
        "pin_message" => pin_message(token, &request.channel_id, &request.message_id),
        "unpin_message" => unpin_message(token, &request.channel_id, &request.message_id),
        "create_thread" => create_thread(
            token,
            &request.channel_id,
            &request.name,
            empty_to_none(&request.message_id),
            request.auto_archive_duration,
        ),
        "add_role" => add_role(token, &request.guild_id, &request.user_id, &request.role_id),
        "remove_role" => remove_role(token, &request.guild_id, &request.user_id, &request.role_id),
        _ => Err(DiscordError::Message(format!("Unknown action: {action}"))),
    }
}

fn list_guilds(token: &str) -> Result<Value, DiscordError> {
    let payload = request_json(Method::GET, "/users/@me/guilds", token, &[], None)?;
    let guilds = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected guild list.".to_string())
    })?;
    let guilds = guilds
        .iter()
        .map(|guild| {
            json!({
                "id": guild.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": guild.get("name").and_then(Value::as_str).unwrap_or_default(),
                "icon": guild.get("icon").cloned().unwrap_or(Value::Null),
                "owner": guild.get("owner").and_then(Value::as_bool).unwrap_or(false),
                "permissions": guild.get("permissions").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "guilds": guilds,
        "count": guilds.len(),
    }))
}

fn server_info(token: &str, guild_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/guilds/{guild_id}"),
        token,
        &[("with_counts", "true".to_string())],
        None,
    )?;
    Ok(json!({
        "id": payload.get("id").and_then(Value::as_str).unwrap_or_default(),
        "name": payload.get("name").and_then(Value::as_str).unwrap_or_default(),
        "description": payload.get("description").cloned().unwrap_or(Value::Null),
        "icon": payload.get("icon").cloned().unwrap_or(Value::Null),
        "owner_id": payload.get("owner_id").cloned().unwrap_or(Value::Null),
        "member_count": payload.get("approximate_member_count").cloned().unwrap_or(Value::Null),
        "online_count": payload.get("approximate_presence_count").cloned().unwrap_or(Value::Null),
        "features": payload.get("features").cloned().unwrap_or_else(|| json!([])),
        "premium_tier": payload.get("premium_tier").cloned().unwrap_or(Value::Null),
        "premium_subscription_count": payload.get("premium_subscription_count").cloned().unwrap_or(Value::Null),
        "verification_level": payload.get("verification_level").cloned().unwrap_or(Value::Null),
    }))
}

fn list_channels(token: &str, guild_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/guilds/{guild_id}/channels"),
        token,
        &[],
        None,
    )?;
    let channels = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected channel list.".to_string())
    })?;

    #[derive(Clone)]
    struct CategoryGroup {
        id: String,
        name: String,
        position: i64,
        channels: Vec<Value>,
    }

    let mut categories = HashMap::<String, CategoryGroup>::new();
    let mut uncategorized = Vec::new();

    for channel in channels {
        if channel.get("type").and_then(Value::as_i64) == Some(4) {
            let id = channel
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            categories.insert(
                id.clone(),
                CategoryGroup {
                    id,
                    name: channel
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    position: channel.get("position").and_then(Value::as_i64).unwrap_or(0),
                    channels: Vec::new(),
                },
            );
        }
    }

    for channel in channels {
        if channel.get("type").and_then(Value::as_i64) == Some(4) {
            continue;
        }
        let entry = json!({
            "id": channel.get("id").and_then(Value::as_str).unwrap_or_default(),
            "name": channel.get("name").and_then(Value::as_str).unwrap_or_default(),
            "type": channel_type_name(channel.get("type").and_then(Value::as_i64).unwrap_or(-1)),
            "position": channel.get("position").and_then(Value::as_i64).unwrap_or(0),
            "topic": channel.get("topic").cloned().unwrap_or(Value::Null),
            "nsfw": channel.get("nsfw").and_then(Value::as_bool).unwrap_or(false),
        });
        match channel.get("parent_id").and_then(Value::as_str) {
            Some(parent) if categories.contains_key(parent) => {
                if let Some(group) = categories.get_mut(parent) {
                    group.channels.push(entry);
                }
            }
            _ => uncategorized.push(entry),
        }
    }

    uncategorized.sort_by_key(|entry| entry.get("position").and_then(Value::as_i64).unwrap_or(0));
    let mut category_values = categories.into_values().collect::<Vec<_>>();
    category_values.sort_by_key(|group| group.position);
    for group in &mut category_values {
        group
            .channels
            .sort_by_key(|entry| entry.get("position").and_then(Value::as_i64).unwrap_or(0));
    }

    let mut result = Vec::new();
    if !uncategorized.is_empty() {
        result.push(json!({
            "category": Value::Null,
            "channels": uncategorized,
        }));
    }
    for group in category_values {
        result.push(json!({
            "category": {
                "id": group.id,
                "name": group.name,
            },
            "channels": group.channels,
        }));
    }

    let total_channels = result
        .iter()
        .map(|group| {
            group
                .get("channels")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        })
        .sum::<usize>();

    Ok(json!({
        "channel_groups": result,
        "total_channels": total_channels,
    }))
}

fn channel_info(token: &str, channel_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/channels/{channel_id}"),
        token,
        &[],
        None,
    )?;
    Ok(json!({
        "id": payload.get("id").and_then(Value::as_str).unwrap_or_default(),
        "name": payload.get("name").cloned().unwrap_or(Value::Null),
        "type": channel_type_name(payload.get("type").and_then(Value::as_i64).unwrap_or(-1)),
        "guild_id": payload.get("guild_id").cloned().unwrap_or(Value::Null),
        "topic": payload.get("topic").cloned().unwrap_or(Value::Null),
        "nsfw": payload.get("nsfw").and_then(Value::as_bool).unwrap_or(false),
        "position": payload.get("position").cloned().unwrap_or(Value::Null),
        "parent_id": payload.get("parent_id").cloned().unwrap_or(Value::Null),
        "rate_limit_per_user": payload.get("rate_limit_per_user").cloned().unwrap_or_else(|| json!(0)),
        "last_message_id": payload.get("last_message_id").cloned().unwrap_or(Value::Null),
    }))
}

fn list_roles(token: &str, guild_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/guilds/{guild_id}/roles"),
        token,
        &[],
        None,
    )?;
    let roles = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected role list.".to_string())
    })?;
    let mut roles = roles
        .iter()
        .map(|role| {
            let color = role.get("color").and_then(Value::as_i64).unwrap_or(0);
            json!({
                "id": role.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": role.get("name").and_then(Value::as_str).unwrap_or_default(),
                "color": if color > 0 { Value::String(format!("#{color:06x}")) } else { Value::Null },
                "position": role.get("position").and_then(Value::as_i64).unwrap_or(0),
                "mentionable": role.get("mentionable").and_then(Value::as_bool).unwrap_or(false),
                "managed": role.get("managed").and_then(Value::as_bool).unwrap_or(false),
                "member_count": role.get("member_count").cloned().unwrap_or(Value::Null),
                "hoist": role.get("hoist").and_then(Value::as_bool).unwrap_or(false),
            })
        })
        .collect::<Vec<_>>();
    roles.sort_by_key(|role| -(role.get("position").and_then(Value::as_i64).unwrap_or(0)));
    Ok(json!({
        "roles": roles,
        "count": roles.len(),
    }))
}

fn member_info(token: &str, guild_id: &str, user_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/guilds/{guild_id}/members/{user_id}"),
        token,
        &[],
        None,
    )?;
    let user = payload.get("user").cloned().unwrap_or_else(|| json!({}));
    Ok(json!({
        "user_id": user.get("id").and_then(Value::as_str).unwrap_or_default(),
        "username": user.get("username").cloned().unwrap_or(Value::Null),
        "display_name": user.get("global_name").cloned().unwrap_or(Value::Null),
        "nickname": payload.get("nick").cloned().unwrap_or(Value::Null),
        "avatar": user.get("avatar").cloned().unwrap_or(Value::Null),
        "bot": user.get("bot").and_then(Value::as_bool).unwrap_or(false),
        "roles": payload.get("roles").cloned().unwrap_or_else(|| json!([])),
        "joined_at": payload.get("joined_at").cloned().unwrap_or(Value::Null),
        "premium_since": payload.get("premium_since").cloned().unwrap_or(Value::Null),
    }))
}

fn search_members(
    token: &str,
    guild_id: &str,
    query: &str,
    limit: i64,
) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/guilds/{guild_id}/members/search"),
        token,
        &[
            ("query", query.to_string()),
            ("limit", limit.clamp(1, 100).to_string()),
        ],
        None,
    )?;
    let members = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected member search result.".to_string())
    })?;
    let members = members
        .iter()
        .map(|member| {
            let user = member.get("user").cloned().unwrap_or_else(|| json!({}));
            json!({
                "user_id": user.get("id").and_then(Value::as_str).unwrap_or_default(),
                "username": user.get("username").cloned().unwrap_or(Value::Null),
                "display_name": user.get("global_name").cloned().unwrap_or(Value::Null),
                "nickname": member.get("nick").cloned().unwrap_or(Value::Null),
                "bot": user.get("bot").and_then(Value::as_bool).unwrap_or(false),
                "roles": member.get("roles").cloned().unwrap_or_else(|| json!([])),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "members": members,
        "count": members.len(),
    }))
}

fn fetch_messages(
    token: &str,
    channel_id: &str,
    limit: i64,
    before: Option<&str>,
    after: Option<&str>,
) -> Result<Value, DiscordError> {
    let mut params = vec![("limit", limit.clamp(1, 100).to_string())];
    if let Some(before) = before {
        params.push(("before", before.to_string()));
    }
    if let Some(after) = after {
        params.push(("after", after.to_string()));
    }
    let payload = request_json(
        Method::GET,
        &format!("/channels/{channel_id}/messages"),
        token,
        &params,
        None,
    )?;
    let messages = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected message list.".to_string())
    })?;
    let messages = messages
        .iter()
        .map(|message| {
            let author = message.get("author").cloned().unwrap_or_else(|| json!({}));
            let attachments = message
                .get("attachments")
                .and_then(Value::as_array)
                .map(|items| {
                    items.iter()
                        .map(|attachment| {
                            json!({
                                "filename": attachment.get("filename").cloned().unwrap_or(Value::Null),
                                "url": attachment.get("url").cloned().unwrap_or(Value::Null),
                                "size": attachment.get("size").cloned().unwrap_or(Value::Null),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let reactions = message
                .get("reactions")
                .and_then(Value::as_array)
                .map(|items| {
                    items.iter()
                        .map(|reaction| {
                            json!({
                                "emoji": reaction.get("emoji").and_then(|value| value.get("name")).cloned().unwrap_or(Value::Null),
                                "count": reaction.get("count").and_then(Value::as_i64).unwrap_or(0),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            json!({
                "id": message.get("id").and_then(Value::as_str).unwrap_or_default(),
                "content": message.get("content").and_then(Value::as_str).unwrap_or_default(),
                "author": {
                    "id": author.get("id").and_then(Value::as_str).unwrap_or_default(),
                    "username": author.get("username").cloned().unwrap_or(Value::Null),
                    "display_name": author.get("global_name").cloned().unwrap_or(Value::Null),
                    "bot": author.get("bot").and_then(Value::as_bool).unwrap_or(false),
                },
                "timestamp": message.get("timestamp").cloned().unwrap_or(Value::Null),
                "edited_timestamp": message.get("edited_timestamp").cloned().unwrap_or(Value::Null),
                "attachments": attachments,
                "reactions": reactions,
                "pinned": message.get("pinned").and_then(Value::as_bool).unwrap_or(false),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "messages": messages,
        "count": messages.len(),
    }))
}

fn list_pins(token: &str, channel_id: &str) -> Result<Value, DiscordError> {
    let payload = request_json(
        Method::GET,
        &format!("/channels/{channel_id}/pins"),
        token,
        &[],
        None,
    )?;
    let messages = payload.as_array().ok_or_else(|| {
        DiscordError::Message("Discord returned an unexpected pin list.".to_string())
    })?;
    let pinned_messages = messages
        .iter()
        .map(|message| {
            let author = message.get("author").cloned().unwrap_or_else(|| json!({}));
            json!({
                "id": message.get("id").and_then(Value::as_str).unwrap_or_default(),
                "content": truncate_chars(message.get("content").and_then(Value::as_str).unwrap_or_default(), 200),
                "author": author.get("username").cloned().unwrap_or(Value::Null),
                "timestamp": message.get("timestamp").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "pinned_messages": pinned_messages,
        "count": pinned_messages.len(),
    }))
}

fn pin_message(token: &str, channel_id: &str, message_id: &str) -> Result<Value, DiscordError> {
    let _ = discord_request(
        Method::PUT,
        &format!("/channels/{channel_id}/pins/{message_id}"),
        token,
        &[],
        None,
    )?;
    Ok(json!({
        "success": true,
        "message": format!("Message {message_id} pinned."),
    }))
}

fn unpin_message(token: &str, channel_id: &str, message_id: &str) -> Result<Value, DiscordError> {
    let _ = discord_request(
        Method::DELETE,
        &format!("/channels/{channel_id}/pins/{message_id}"),
        token,
        &[],
        None,
    )?;
    Ok(json!({
        "success": true,
        "message": format!("Message {message_id} unpinned."),
    }))
}

fn create_thread(
    token: &str,
    channel_id: &str,
    name: &str,
    message_id: Option<&str>,
    auto_archive_duration: i64,
) -> Result<Value, DiscordError> {
    let (path, body) = match message_id {
        Some(message_id) => (
            format!("/channels/{channel_id}/messages/{message_id}/threads"),
            json!({
                "name": name,
                "auto_archive_duration": auto_archive_duration,
            }),
        ),
        None => (
            format!("/channels/{channel_id}/threads"),
            json!({
                "name": name,
                "auto_archive_duration": auto_archive_duration,
                "type": 11,
            }),
        ),
    };
    let payload = request_json(Method::POST, &path, token, &[], Some(body))?;
    Ok(json!({
        "success": true,
        "thread_id": payload.get("id").and_then(Value::as_str).unwrap_or_default(),
        "name": payload.get("name").cloned().unwrap_or(Value::Null),
    }))
}

fn add_role(
    token: &str,
    guild_id: &str,
    user_id: &str,
    role_id: &str,
) -> Result<Value, DiscordError> {
    let _ = discord_request(
        Method::PUT,
        &format!("/guilds/{guild_id}/members/{user_id}/roles/{role_id}"),
        token,
        &[],
        None,
    )?;
    Ok(json!({
        "success": true,
        "message": format!("Role {role_id} added to user {user_id}."),
    }))
}

fn remove_role(
    token: &str,
    guild_id: &str,
    user_id: &str,
    role_id: &str,
) -> Result<Value, DiscordError> {
    let _ = discord_request(
        Method::DELETE,
        &format!("/guilds/{guild_id}/members/{user_id}/roles/{role_id}"),
        token,
        &[],
        None,
    )?;
    Ok(json!({
        "success": true,
        "message": format!("Role {role_id} removed from user {user_id}."),
    }))
}

fn request_json(
    method: Method,
    path: &str,
    token: &str,
    params: &[(&str, String)],
    body: Option<Value>,
) -> Result<Value, DiscordError> {
    match discord_request(method, path, token, params, body)? {
        Some(value) => Ok(value),
        None => Ok(Value::Null),
    }
}

fn discord_request(
    method: Method,
    path: &str,
    token: &str,
    params: &[(&str, String)],
    body: Option<Value>,
) -> Result<Option<Value>, DiscordError> {
    let url = build_url(path, params)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(DISCORD_TIMEOUT_SECS))
        .build()
        .map_err(|error| {
            DiscordError::Message(format!("Failed to build Discord client: {error}"))
        })?;

    let mut request = client
        .request(method, url)
        .header("Authorization", format!("Bot {token}"))
        .header("Content-Type", "application/json")
        .header("User-Agent", DISCORD_USER_AGENT);
    if let Some(body) = body {
        request = request.json(&body);
    }

    let response = request
        .send()
        .map_err(|error| DiscordError::Message(format!("Discord request failed: {error}")))?;
    let status = response.status();
    if status.as_u16() == 204 {
        return Ok(None);
    }
    let text = response.text().map_err(|error| {
        DiscordError::Message(format!("Failed to read Discord response: {error}"))
    })?;
    if !status.is_success() {
        return Err(DiscordError::Api(DiscordApiError {
            status: status.as_u16(),
            body: text,
        }));
    }
    if text.trim().is_empty() {
        return Ok(Some(Value::Null));
    }
    serde_json::from_str::<Value>(&text)
        .map(Some)
        .map_err(|error| {
            DiscordError::Message(format!("Failed to parse Discord response JSON: {error}"))
        })
}

fn build_url(path: &str, params: &[(&str, String)]) -> Result<String, DiscordError> {
    let base = env::var("DISCORD_API_BASE_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DISCORD_API_BASE.to_string());
    let mut url = reqwest::Url::parse(&format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    ))
    .map_err(|error| DiscordError::Message(format!("Invalid Discord API base URL: {error}")))?;
    if !params.is_empty() {
        let mut query = url.query_pairs_mut();
        for (key, value) in params {
            query.append_pair(key, value);
        }
    }
    Ok(url.to_string())
}

fn detect_capabilities(token: &str) -> DiscordCapabilities {
    if let Ok(cache) = capability_cache().lock() {
        if let Some(cached) = cache.get(token).copied() {
            return cached;
        }
    }

    let mut capabilities = DiscordCapabilities::permissive();
    match request_json(Method::GET, "/applications/@me", token, &[], None) {
        Ok(payload) => {
            let flags = payload
                .get("flags")
                .and_then(|value| {
                    value
                        .as_u64()
                        .or_else(|| value.as_str()?.parse::<u64>().ok())
                })
                .unwrap_or(0);
            capabilities.has_members_intent =
                flags & (FLAG_GATEWAY_GUILD_MEMBERS | FLAG_GATEWAY_GUILD_MEMBERS_LIMITED) != 0;
            capabilities.has_message_content =
                flags & (FLAG_GATEWAY_MESSAGE_CONTENT | FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED) != 0;
            capabilities.detected = true;
        }
        Err(error) => {
            log::info!(target: "hermes_discord", "discord capability detection failed: {}", error);
        }
    }

    if let Ok(mut cache) = capability_cache().lock() {
        cache.insert(token.to_string(), capabilities);
    }
    capabilities
}

fn load_allowed_actions(context: &HermesContext) -> Option<Vec<String>> {
    let loaded = match context.load_config_document() {
        Ok(value) => value,
        Err(error) => {
            log::debug!(target: "hermes_discord", "discord config load failed: {}", error);
            return None;
        }
    };
    let Some(raw) = loaded.cfg_get(&["discord", "server_actions"]) else {
        return None;
    };
    let names = match raw {
        YamlValue::Null => return None,
        YamlValue::String(text) => {
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            text.split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        }
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(|item| match item {
                YamlValue::String(text) => {
                    let value = text.trim();
                    if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    }
                }
                YamlValue::Number(number) => Some(number.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        other => {
            log::warn!(
                target: "hermes_discord",
                "discord.server_actions unexpected type {}; ignoring",
                yaml_type_name(other)
            );
            return None;
        }
    };

    let known = ACTIONS.iter().map(|spec| spec.name).collect::<HashSet<_>>();
    let valid = names
        .iter()
        .filter(|name| known.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let invalid = names
        .iter()
        .filter(|name| !known.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !invalid.is_empty() {
        log::warn!(
            target: "hermes_discord",
            "discord.server_actions ignored unknown actions: {}",
            invalid.join(", ")
        );
    }
    Some(valid)
}

fn available_actions(
    subset: &[&str],
    capabilities: DiscordCapabilities,
    allowlist: Option<Vec<String>>,
) -> Vec<&'static str> {
    let subset = subset.iter().copied().collect::<HashSet<_>>();
    let allowlist = allowlist.map(|items| items.into_iter().collect::<HashSet<_>>());

    ACTIONS
        .iter()
        .filter(|spec| subset.contains(spec.name))
        .filter(|spec| capabilities.has_members_intent || !spec.members_intent)
        .filter(|spec| {
            allowlist
                .as_ref()
                .is_none_or(|allowed| allowed.contains(spec.name))
        })
        .map(|spec| spec.name)
        .collect()
}

fn required_params(action: &str) -> &'static [&'static str] {
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

fn channel_type_name(type_id: i64) -> String {
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

fn discord_bot_token() -> Option<String> {
    env::var("DISCORD_BOT_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    let value = optional_string(args, key)?;
    if value.is_empty() {
        return Err(format!("Missing required parameter: {key}"));
    }
    Ok(value)
}

fn optional_string(args: &Value, key: &str) -> Result<String, String> {
    let Some(value) = args.get(key) else {
        return Ok(String::new());
    };
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.trim().to_string()),
        Value::Number(number) => Ok(number.to_string()),
        Value::Bool(boolean) => Ok(boolean.to_string()),
        _ => Err(format!("Parameter '{key}' must be a string.")),
    }
}

fn optional_integer(args: &Value, key: &str) -> Option<i64> {
    let value = args.get(key)?;
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn is_snowflake(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn empty_to_none(value: &str) -> Option<&str> {
    if value.is_empty() { None } else { Some(value) }
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}

fn yaml_type_name(value: &YamlValue) -> &'static str {
    match value {
        YamlValue::Null => "null",
        YamlValue::Bool(_) => "bool",
        YamlValue::Number(_) => "number",
        YamlValue::String(_) => "string",
        YamlValue::Sequence(_) => "sequence",
        YamlValue::Mapping(_) => "mapping",
        YamlValue::Tagged(_) => "tagged",
    }
}

fn enrich_403(action: &str, body: &str) -> String {
    let hint = match action {
        "pin_message" => Some(
            "Bot lacks MANAGE_MESSAGES permission in this channel. Ask the server admin to grant MANAGE_MESSAGES or a per-channel overwrite.",
        ),
        "unpin_message" => Some("Bot lacks MANAGE_MESSAGES permission in this channel."),
        "create_thread" => {
            Some("Bot lacks CREATE_PUBLIC_THREADS in this channel or cannot view it.")
        }
        "add_role" => Some(
            "Either the bot lacks MANAGE_ROLES, or the target role sits above the bot's highest role in the hierarchy.",
        ),
        "remove_role" => Some(
            "Either the bot lacks MANAGE_ROLES, or the target role sits above the bot's highest role.",
        ),
        "fetch_messages" => Some("Bot cannot view this channel or read its history."),
        "list_pins" => Some("Bot cannot view this channel or read its history."),
        "channel_info" => Some("Bot cannot view this channel."),
        "search_members" => Some(
            "The bot likely lacks the Server Members privileged intent in the Discord Developer Portal.",
        ),
        "member_info" => Some(
            "Bot cannot see this guild member because it lacks the Server Members intent or sufficient permissions.",
        ),
        _ => None,
    };
    match hint {
        Some(hint) => format!("Discord API 403 (forbidden) on '{action}'. {hint} (Raw: {body})"),
        None => format!("Discord API 403 (forbidden) on '{action}'. (Raw: {body})"),
    }
}

#[cfg(test)]
fn reset_capability_cache() {
    if let Ok(mut cache) = capability_cache().lock() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::thread;

    use tempfile::TempDir;

    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_var(key: &str, value: Option<&str>) {
        match value {
            Some(value) => unsafe { env::set_var(key, value) },
            None => unsafe { env::remove_var(key) },
        }
    }

    fn test_env_lock() -> &'static Mutex<()> {
        TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn mock_discord_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(String) -> (u16, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .unwrap_or(request.len());
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            let request = format!(
                "{}{}",
                String::from_utf8_lossy(&request[..header_end]),
                String::from_utf8_lossy(&body_bytes)
            );
            let (status, body) = handler(request);
            let response = format!(
                "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status,
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (address, join)
    }

    fn prepare_home() -> TempDir {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        temp
    }

    fn runtime_for(home: &TempDir) -> ToolRuntime {
        ToolRuntime::new(home.path()).with_hermes_home(home.path())
    }

    #[test]
    fn discord_toolsets_are_opt_in() {
        assert_eq!(
            crate::resolve_toolset("discord"),
            vec!["discord".to_string()]
        );
        assert_eq!(
            crate::resolve_toolset("discord_admin"),
            vec!["discord_admin".to_string()]
        );
        assert!(!crate::resolve_toolset("hermes-cli").contains(&"discord".to_string()));
        assert!(!crate::resolve_toolset("hermes-cli").contains(&"discord_admin".to_string()));
    }

    #[test]
    fn dynamic_admin_schema_respects_config_allowlist() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        fs::write(
            home.path().join("config.yaml"),
            "discord:\n  server_actions: list_guilds,list_channels\n",
        )
        .unwrap();

        let old_home = env::var("HERMES_HOME").ok();
        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        let old_base = env::var("DISCORD_API_BASE_URL").ok();
        reset_capability_cache();
        with_env_var("HERMES_HOME", Some(home.path().to_str().unwrap()));
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let (base_url, join) = mock_discord_server(|request| {
            assert!(request.starts_with("GET /applications/@me "));
            (
                200,
                json!({ "flags": FLAG_GATEWAY_GUILD_MEMBERS | FLAG_GATEWAY_MESSAGE_CONTENT })
                    .to_string(),
            )
        });
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));

        let schema = discord_admin_schema();
        let actions = schema["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert_eq!(actions, vec!["list_guilds", "list_channels"]);
        assert!(
            schema["description"]
                .as_str()
                .unwrap()
                .contains("list_guilds()")
        );

        join.join().unwrap();
        reset_capability_cache();
        with_env_var("DISCORD_API_BASE_URL", old_base.as_deref());
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
        with_env_var("HERMES_HOME", old_home.as_deref());
    }

    #[test]
    fn load_allowed_actions_reads_config_values() {
        let home = prepare_home();
        fs::write(
            home.path().join("config.yaml"),
            "discord:\n  server_actions: list_guilds,list_channels\n",
        )
        .unwrap();

        let context =
            HermesContext::new(home.path()).with_hermes_home_env(Some(home.path().into()));
        let allowlist = load_allowed_actions(&context).unwrap();
        assert_eq!(allowlist, vec!["list_guilds", "list_channels"]);
    }

    #[test]
    fn dynamic_core_schema_hides_member_actions_without_intent() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        fs::write(
            home.path().join("config.yaml"),
            "discord:\n  server_actions: ''\n",
        )
        .unwrap();

        let old_home = env::var("HERMES_HOME").ok();
        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        let old_base = env::var("DISCORD_API_BASE_URL").ok();
        reset_capability_cache();
        with_env_var("HERMES_HOME", Some(home.path().to_str().unwrap()));
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let (base_url, join) = mock_discord_server(|request| {
            assert!(request.starts_with("GET /applications/@me "));
            (
                200,
                json!({ "flags": FLAG_GATEWAY_MESSAGE_CONTENT }).to_string(),
            )
        });
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));

        let schema = discord_core_schema();
        let actions = schema["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert_eq!(actions, vec!["fetch_messages", "create_thread"]);

        join.join().unwrap();
        reset_capability_cache();
        with_env_var("DISCORD_API_BASE_URL", old_base.as_deref());
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
        with_env_var("HERMES_HOME", old_home.as_deref());
    }

    #[test]
    fn discord_admin_tool_drops_when_allowlist_is_empty() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        fs::write(
            home.path().join("config.yaml"),
            "discord:\n  server_actions: typo_one,typo_two\n",
        )
        .unwrap();

        let old_home = env::var("HERMES_HOME").ok();
        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        let old_base = env::var("DISCORD_API_BASE_URL").ok();
        reset_capability_cache();
        with_env_var("HERMES_HOME", Some(home.path().to_str().unwrap()));
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let (base_url, join) = mock_discord_server(|_| {
            (
                200,
                json!({ "flags": FLAG_GATEWAY_GUILD_MEMBERS | FLAG_GATEWAY_MESSAGE_CONTENT })
                    .to_string(),
            )
        });
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));

        let enabled = vec!["discord_admin".to_string()];
        let tools = crate::get_tool_definitions(Some(&enabled), None);
        assert!(tools.is_empty());

        join.join().unwrap();
        reset_capability_cache();
        with_env_var("DISCORD_API_BASE_URL", old_base.as_deref());
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
        with_env_var("HERMES_HOME", old_home.as_deref());
    }

    #[test]
    fn discord_runtime_allowlist_blocks_denied_action() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        fs::write(
            home.path().join("config.yaml"),
            "discord:\n  server_actions: list_guilds\n",
        )
        .unwrap();

        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let result = serde_json::from_str::<Value>(&handle_discord_admin(
            &json!({
                "action": "add_role",
                "guild_id": "1",
                "user_id": "2",
                "role_id": "3",
            }),
            &runtime_for(&home),
        ))
        .unwrap();
        assert!(
            result["error"]
                .as_str()
                .unwrap()
                .contains("disabled by config")
        );
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
    }

    #[test]
    fn fetch_messages_returns_python_compatible_shape() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        let old_base = env::var("DISCORD_API_BASE_URL").ok();
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let (base_url, join) = mock_discord_server(|request| {
            assert!(request.starts_with("GET /channels/11/messages?"));
            assert!(request.contains("limit=10"));
            assert!(request.contains("before=999"));
            (
                200,
                json!([
                    {
                        "id": "500",
                        "content": "hello",
                        "author": {
                            "id": "42",
                            "username": "tester",
                            "global_name": "Tester",
                            "bot": false
                        },
                        "timestamp": "2026-05-07T00:00:00.000000+00:00",
                        "edited_timestamp": null,
                        "attachments": [{"filename": "a.txt", "url": "https://x", "size": 1}],
                        "reactions": [{"emoji": {"name": "👍"}, "count": 2}],
                        "pinned": true
                    }
                ])
                .to_string(),
            )
        });
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));

        let result = serde_json::from_str::<Value>(&handle_discord(
            &json!({
                "action": "fetch_messages",
                "channel_id": "11",
                "limit": 10,
                "before": "999",
            }),
            &runtime_for(&home),
        ))
        .unwrap();
        assert_eq!(result["count"], json!(1));
        assert_eq!(result["messages"][0]["author"]["username"], json!("tester"));
        assert_eq!(result["messages"][0]["reactions"][0]["emoji"], json!("👍"));

        join.join().unwrap();
        with_env_var("DISCORD_API_BASE_URL", old_base.as_deref());
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
    }

    #[test]
    fn add_role_403_is_enriched() {
        let _guard = test_env_lock().lock().unwrap();
        let home = prepare_home();
        let old_token = env::var("DISCORD_BOT_TOKEN").ok();
        let old_base = env::var("DISCORD_API_BASE_URL").ok();
        with_env_var("DISCORD_BOT_TOKEN", Some("tok"));
        let (base_url, join) = mock_discord_server(|request| {
            assert!(request.starts_with("PUT /guilds/1/members/2/roles/3 "));
            (403, json!({ "message": "Missing Permissions" }).to_string())
        });
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));

        let result = serde_json::from_str::<Value>(&handle_discord_admin(
            &json!({
                "action": "add_role",
                "guild_id": "1",
                "user_id": "2",
                "role_id": "3",
            }),
            &runtime_for(&home),
        ))
        .unwrap();
        assert!(result["error"].as_str().unwrap().contains("MANAGE_ROLES"));
        assert!(
            result["error"]
                .as_str()
                .unwrap()
                .contains("Missing Permissions")
        );

        join.join().unwrap();
        with_env_var("DISCORD_API_BASE_URL", old_base.as_deref());
        with_env_var("DISCORD_BOT_TOKEN", old_token.as_deref());
    }
}
