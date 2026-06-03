//! Toolsets module — native Rust port of `toolsets.py`.
//!
//! Provides a flexible system for defining and managing tool aliases /
//! toolsets. Toolsets group tools together for specific scenarios and can be
//! composed from individual tools or other toolsets.
//!
//! Features:
//! - Define custom toolsets with specific tools
//! - Compose toolsets from other toolsets
//! - Built-in common toolsets for typical use cases
//! - Easy extension for new toolsets
//! - Support for dynamic toolset resolution
//!
//! ## Registry / platform integration
//!
//! The original Python module reaches into `tools.registry` and
//! `gateway.platform_registry` at runtime to merge in plugin- and
//! platform-registered toolsets. Those integrations are dynamic Python hooks
//! with no native equivalent here, so this port models them behind the
//! [`ToolsetRegistry`] trait. By default a [`NullRegistry`] is used (matching
//! the Python behaviour when those imports fail — i.e. the `except Exception`
//! branches return empty / static data), but callers may pass a custom
//! registry to reproduce the full dynamic merge behaviour.
//!
//! The free functions ([`get_toolset`], [`resolve_toolset`], etc.) operate
//! against the static [`base_toolsets`] table plus a thread-safe runtime
//! overlay populated by [`create_custom_toolset`], using a [`NullRegistry`].
//! For registry-aware behaviour use the equivalent methods on a
//! [`Toolsets`] instance.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

/// A single toolset definition: description + direct tools + included toolsets.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolsetDef {
    pub description: String,
    pub tools: Vec<String>,
    pub includes: Vec<String>,
}

impl ToolsetDef {
    pub fn new<S: Into<String>>(description: S, tools: Vec<String>, includes: Vec<String>) -> Self {
        Self {
            description: description.into(),
            tools,
            includes,
        }
    }
}

/// Detailed information about a toolset including resolved tools.
/// Mirrors the dict returned by Python's `get_toolset_info`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolsetInfo {
    pub name: String,
    pub description: String,
    pub direct_tools: Vec<String>,
    pub includes: Vec<String>,
    pub resolved_tools: Vec<String>,
    pub tool_count: usize,
    pub is_composite: bool,
}

/// Pluggable hook for plugin- and platform-registered toolsets.
///
/// All methods have empty defaults that reproduce the Python `except
/// Exception: return …` fallbacks (i.e. when the registry / platform modules
/// are unavailable). Implement this to wire in dynamic data.
pub trait ToolsetRegistry {
    /// Tools registered (in the live registry) for a given toolset name.
    /// Mirrors `registry.get_tool_names_for_toolset(name)`.
    fn tool_names_for_toolset(&self, _name: &str) -> Vec<String> {
        Vec::new()
    }

    /// All toolset names registered in the live registry.
    /// Mirrors `registry.get_registered_toolset_names()`.
    fn registered_toolset_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Explicit toolset aliases registered in the live registry
    /// (`alias -> canonical`). Mirrors
    /// `registry.get_registered_toolset_aliases()`.
    fn registered_toolset_aliases(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Resolve an alias name to its canonical toolset, if any.
    /// Mirrors `registry.get_toolset_alias_target(name)`.
    fn toolset_alias_target(&self, name: &str) -> Option<String> {
        self.registered_toolset_aliases().get(name).cloned()
    }

    /// Registered plugin platform entries as `(name, label)` pairs.
    /// Mirrors `platform_registry.plugin_entries()` / `.get(name)`.
    fn plugin_platform_entries(&self) -> Vec<PlatformEntry> {
        Vec::new()
    }
}

/// A registered plugin platform entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformEntry {
    pub name: String,
    pub label: String,
}

/// A registry that provides no dynamic data — reproduces the Python behaviour
/// when `tools.registry` / `gateway.platform_registry` are unavailable.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullRegistry;

impl ToolsetRegistry for NullRegistry {}

/// Shared tool list for CLI and all messaging platform toolsets.
/// Edit this once to update all platforms simultaneously.
/// Mirrors `_HERMES_CORE_TOOLS`.
pub const HERMES_CORE_TOOLS: &[&str] = &[
    // Web
    "web_search",
    "web_extract",
    // Terminal + process management
    "terminal",
    "process",
    // File manipulation
    "read_file",
    "write_file",
    "patch",
    "search_files",
    // Vision + image generation
    "vision_analyze",
    "image_generate",
    // Skills
    "skills_list",
    "skill_view",
    "skill_manage",
    // Browser automation
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_get_images",
    "browser_vision",
    "browser_console",
    "browser_cdp",
    "browser_dialog",
    // Text-to-speech
    "text_to_speech",
    // Planning & memory
    "todo",
    "memory",
    // Session history search
    "session_search",
    // Clarifying questions
    "clarify",
    // Code execution + delegation
    "execute_code",
    "delegate_task",
    // Cronjob management
    "cronjob",
    // Cross-platform messaging (gated on gateway running via check_fn)
    "send_message",
    // Home Assistant smart home control (gated on HASS_TOKEN via check_fn)
    "ha_list_entities",
    "ha_get_state",
    "ha_list_services",
    "ha_call_service",
    // Kanban multi-agent coordination
    "kanban_show",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
    "kanban_create",
    "kanban_link",
];

fn core_tools_vec() -> Vec<String> {
    HERMES_CORE_TOOLS.iter().map(|s| s.to_string()).collect()
}

fn vec_of(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// Core tools plus the given extra tools (preserving order, appending extras).
/// Mirrors `_HERMES_CORE_TOOLS + [...]` list concatenation.
fn core_plus(extra: &[&str]) -> Vec<String> {
    let mut v = core_tools_vec();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

/// Build the static base toolset table — a faithful reproduction of the
/// Python `TOOLSETS` dict (insertion order preserved via `Vec`).
pub fn base_toolsets() -> Vec<(String, ToolsetDef)> {
    let mut t: Vec<(String, ToolsetDef)> = Vec::new();
    let mut push = |name: &str, def: ToolsetDef| t.push((name.to_string(), def));

    // Basic toolsets - individual tool categories
    push(
        "web",
        ToolsetDef::new(
            "Web research and content extraction tools",
            vec_of(&["web_search", "web_extract"]),
            vec![],
        ),
    );
    push(
        "search",
        ToolsetDef::new(
            "Web search only (no content extraction/scraping)",
            vec_of(&["web_search"]),
            vec![],
        ),
    );
    push(
        "vision",
        ToolsetDef::new(
            "Image analysis and vision tools",
            vec_of(&["vision_analyze"]),
            vec![],
        ),
    );
    push(
        "video",
        ToolsetDef::new(
            "Video analysis and understanding tools (opt-in, not in default toolset)",
            vec_of(&["video_analyze"]),
            vec![],
        ),
    );
    push(
        "image_gen",
        ToolsetDef::new(
            "Creative generation tools (images)",
            vec_of(&["image_generate"]),
            vec![],
        ),
    );
    push(
        "terminal",
        ToolsetDef::new(
            "Terminal/command execution and process management tools",
            vec_of(&["terminal", "process"]),
            vec![],
        ),
    );
    push(
        "moa",
        ToolsetDef::new(
            "Advanced reasoning and problem-solving tools",
            vec_of(&["mixture_of_agents"]),
            vec![],
        ),
    );
    push(
        "skills",
        ToolsetDef::new(
            "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
            vec_of(&["skills_list", "skill_view", "skill_manage"]),
            vec![],
        ),
    );
    push(
        "browser",
        ToolsetDef::new(
            "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
            vec_of(&[
                "browser_navigate", "browser_snapshot", "browser_click",
                "browser_type", "browser_scroll", "browser_back",
                "browser_press", "browser_get_images",
                "browser_vision", "browser_console", "browser_cdp",
                "browser_dialog", "web_search",
            ]),
            vec![],
        ),
    );
    push(
        "cronjob",
        ToolsetDef::new(
            "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
            vec_of(&["cronjob"]),
            vec![],
        ),
    );
    push(
        "messaging",
        ToolsetDef::new(
            "Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.",
            vec_of(&["send_message"]),
            vec![],
        ),
    );
    push(
        "rl",
        ToolsetDef::new(
            "RL training tools for running reinforcement learning on Tinker-Atropos",
            vec_of(&[
                "rl_list_environments", "rl_select_environment",
                "rl_get_current_config", "rl_edit_config",
                "rl_start_training", "rl_check_status",
                "rl_stop_training", "rl_get_results",
                "rl_list_runs", "rl_test_inference",
            ]),
            vec![],
        ),
    );
    push(
        "file",
        ToolsetDef::new(
            "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
            vec_of(&["read_file", "write_file", "patch", "search_files"]),
            vec![],
        ),
    );
    push(
        "tts",
        ToolsetDef::new(
            "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI",
            vec_of(&["text_to_speech"]),
            vec![],
        ),
    );
    push(
        "todo",
        ToolsetDef::new(
            "Task planning and tracking for multi-step work",
            vec_of(&["todo"]),
            vec![],
        ),
    );
    push(
        "memory",
        ToolsetDef::new(
            "Persistent memory across sessions (personal notes + user profile)",
            vec_of(&["memory"]),
            vec![],
        ),
    );
    push(
        "session_search",
        ToolsetDef::new(
            "Search and recall past conversations with summarization",
            vec_of(&["session_search"]),
            vec![],
        ),
    );
    push(
        "clarify",
        ToolsetDef::new(
            "Ask the user clarifying questions (multiple-choice or open-ended)",
            vec_of(&["clarify"]),
            vec![],
        ),
    );
    push(
        "code_execution",
        ToolsetDef::new(
            "Run Python scripts that call tools programmatically (reduces LLM round trips)",
            vec_of(&["execute_code"]),
            vec![],
        ),
    );
    push(
        "delegation",
        ToolsetDef::new(
            "Spawn subagents with isolated context for complex subtasks",
            vec_of(&["delegate_task"]),
            vec![],
        ),
    );
    push(
        "homeassistant",
        ToolsetDef::new(
            "Home Assistant smart home control and monitoring",
            vec_of(&["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]),
            vec![],
        ),
    );
    push(
        "kanban",
        ToolsetDef::new(
            "Kanban multi-agent coordination — only active when the agent \
             is spawned by the kanban dispatcher (HERMES_KANBAN_TASK env \
             set). The dispatcher runs inside the gateway by default; see \
             `kanban.dispatch_in_gateway` in config.yaml. Lets workers mark \
             tasks done with structured handoffs, block for human input, \
             heartbeat during long ops, comment on threads, and (for \
             orchestrators) fan out into child tasks.",
            vec_of(&[
                "kanban_show", "kanban_complete", "kanban_block",
                "kanban_heartbeat", "kanban_comment",
                "kanban_create", "kanban_link",
            ]),
            vec![],
        ),
    );
    push(
        "discord",
        ToolsetDef::new(
            "Discord read and participate tools (fetch messages, search members, create threads)",
            vec_of(&["discord"]),
            vec![],
        ),
    );
    push(
        "discord_admin",
        ToolsetDef::new(
            "Discord server management (list channels/roles, pin messages, assign roles)",
            vec_of(&["discord_admin"]),
            vec![],
        ),
    );
    push(
        "yuanbao",
        ToolsetDef::new(
            "Yuanbao platform tools - group info, member queries, DM, stickers",
            vec_of(&[
                "yb_query_group_info", "yb_query_group_members", "yb_send_dm",
                "yb_search_sticker", "yb_send_sticker",
            ]),
            vec![],
        ),
    );
    push(
        "feishu_doc",
        ToolsetDef::new(
            "Read Feishu/Lark document content",
            vec_of(&["feishu_doc_read"]),
            vec![],
        ),
    );
    push(
        "feishu_drive",
        ToolsetDef::new(
            "Feishu/Lark document comment operations (list, reply, add)",
            vec_of(&[
                "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
                "feishu_drive_reply_comment", "feishu_drive_add_comment",
            ]),
            vec![],
        ),
    );
    push(
        "spotify",
        ToolsetDef::new(
            "Native Spotify playback, search, playlist, album, and library tools",
            vec_of(&[
                "spotify_playback", "spotify_devices", "spotify_queue", "spotify_search",
                "spotify_playlists", "spotify_albums", "spotify_library",
            ]),
            vec![],
        ),
    );

    // Scenario-specific toolsets
    push(
        "debugging",
        ToolsetDef::new(
            "Debugging and troubleshooting toolkit",
            vec_of(&["terminal", "process"]),
            vec_of(&["web", "file"]),
        ),
    );
    push(
        "safe",
        ToolsetDef::new(
            "Safe toolkit without terminal access",
            vec![],
            vec_of(&["web", "vision", "image_gen"]),
        ),
    );

    // Full Hermes toolsets (CLI + messaging platforms)
    push(
        "hermes-acp",
        ToolsetDef::new(
            "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
            vec_of(&[
                "web_search", "web_extract",
                "terminal", "process",
                "read_file", "write_file", "patch", "search_files",
                "vision_analyze",
                "skills_list", "skill_view", "skill_manage",
                "browser_navigate", "browser_snapshot", "browser_click",
                "browser_type", "browser_scroll", "browser_back",
                "browser_press", "browser_get_images",
                "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
                "todo", "memory",
                "session_search",
                "execute_code", "delegate_task",
            ]),
            vec![],
        ),
    );
    push(
        "hermes-api-server",
        ToolsetDef::new(
            "OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)",
            vec_of(&[
                "web_search", "web_extract",
                "terminal", "process",
                "read_file", "write_file", "patch", "search_files",
                "vision_analyze", "image_generate",
                "skills_list", "skill_view", "skill_manage",
                "browser_navigate", "browser_snapshot", "browser_click",
                "browser_type", "browser_scroll", "browser_back",
                "browser_press", "browser_get_images",
                "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
                "todo", "memory",
                "session_search",
                "execute_code", "delegate_task",
                "cronjob",
                "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
            ]),
            vec![],
        ),
    );
    push(
        "hermes-cli",
        ToolsetDef::new(
            "Full interactive CLI toolset - all default tools plus cronjob management",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-cron",
        ToolsetDef::new(
            "Default cron toolset - same core tools as hermes-cli; gated by `hermes tools`",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-telegram",
        ToolsetDef::new(
            "Telegram bot toolset - full access for personal use (terminal has safety checks)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-discord",
        ToolsetDef::new(
            "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
            core_plus(&["discord", "discord_admin"]),
            vec![],
        ),
    );
    push(
        "hermes-whatsapp",
        ToolsetDef::new(
            "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-slack",
        ToolsetDef::new(
            "Slack bot toolset - full access for workspace use (terminal has safety checks)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-signal",
        ToolsetDef::new(
            "Signal bot toolset - encrypted messaging platform (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-bluebubbles",
        ToolsetDef::new(
            "BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-homeassistant",
        ToolsetDef::new(
            "Home Assistant bot toolset - smart home event monitoring and control",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-email",
        ToolsetDef::new(
            "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-mattermost",
        ToolsetDef::new(
            "Mattermost bot toolset - self-hosted team messaging (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-matrix",
        ToolsetDef::new(
            "Matrix bot toolset - decentralized encrypted messaging (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-dingtalk",
        ToolsetDef::new(
            "DingTalk bot toolset - enterprise messaging platform (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-feishu",
        ToolsetDef::new(
            "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
            core_plus(&[
                "feishu_doc_read",
                "feishu_drive_list_comments",
                "feishu_drive_list_comment_replies",
                "feishu_drive_reply_comment",
                "feishu_drive_add_comment",
            ]),
            vec![],
        ),
    );
    push(
        "hermes-weixin",
        ToolsetDef::new(
            "Weixin bot toolset - personal WeChat messaging via iLink (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-qqbot",
        ToolsetDef::new(
            "QQBot toolset - QQ messaging via Official Bot API v2 (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-wecom",
        ToolsetDef::new(
            "WeCom bot toolset - enterprise WeChat messaging (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-wecom-callback",
        ToolsetDef::new(
            "WeCom callback toolset - enterprise self-built app messaging (full access)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-yuanbao",
        ToolsetDef::new(
            "Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情",
            core_plus(&[
                "yb_query_group_info",
                "yb_query_group_members",
                "yb_send_dm",
                "yb_search_sticker",
                "yb_send_sticker",
            ]),
            vec![],
        ),
    );
    push(
        "hermes-sms",
        ToolsetDef::new(
            "SMS bot toolset - interact with Hermes via SMS (Twilio)",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-webhook",
        ToolsetDef::new(
            "Webhook toolset - receive and process external webhook events",
            core_tools_vec(),
            vec![],
        ),
    );
    push(
        "hermes-gateway",
        ToolsetDef::new(
            "Gateway toolset - union of all messaging platform tools",
            vec![],
            vec_of(&[
                "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack",
                "hermes-signal", "hermes-bluebubbles", "hermes-homeassistant", "hermes-email",
                "hermes-sms", "hermes-mattermost", "hermes-matrix", "hermes-dingtalk",
                "hermes-feishu", "hermes-wecom", "hermes-wecom-callback", "hermes-weixin",
                "hermes-qqbot", "hermes-webhook", "hermes-yuanbao",
            ]),
        ),
    );

    t
}

// ---------------------------------------------------------------------------
// Runtime state: static table + custom toolsets overlay
// ---------------------------------------------------------------------------

fn static_table() -> &'static BTreeMap<String, ToolsetDef> {
    static TABLE: OnceLock<BTreeMap<String, ToolsetDef>> = OnceLock::new();
    TABLE.get_or_init(|| base_toolsets().into_iter().collect())
}

/// Insertion-ordered list of static toolset names (matches Python dict order).
fn static_order() -> &'static Vec<String> {
    static ORDER: OnceLock<Vec<String>> = OnceLock::new();
    ORDER.get_or_init(|| base_toolsets().into_iter().map(|(n, _)| n).collect())
}

/// Runtime-created custom toolsets (via `create_custom_toolset`). Mirrors
/// mutating the module-level `TOOLSETS` dict in Python.
fn custom_table() -> &'static Mutex<Vec<(String, ToolsetDef)>> {
    static CUSTOM: OnceLock<Mutex<Vec<(String, ToolsetDef)>>> = OnceLock::new();
    CUSTOM.get_or_init(|| Mutex::new(Vec::new()))
}

/// Look up a name in the static table, then the custom overlay.
fn lookup_static_or_custom(name: &str) -> Option<ToolsetDef> {
    if let Some(def) = static_table().get(name) {
        return Some(def.clone());
    }
    let custom = custom_table().lock().unwrap();
    custom
        .iter()
        .rev()
        .find(|(n, _)| n == name)
        .map(|(_, d)| d.clone())
}

/// Whether `name` exists in the static or custom table (the combined Python
/// `TOOLSETS` dict).
fn in_toolsets(name: &str) -> bool {
    static_table().contains_key(name)
        || custom_table().lock().unwrap().iter().any(|(n, _)| n == name)
}

/// All names in the combined `TOOLSETS` table (static + custom), in a stable
/// order (static insertion order, then custom insertion order).
fn all_local_names() -> Vec<String> {
    let mut names = static_order().clone();
    let custom = custom_table().lock().unwrap();
    for (n, _) in custom.iter() {
        if !names.iter().any(|existing| existing == n) {
            names.push(n.clone());
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Toolsets handle (registry-aware API)
// ---------------------------------------------------------------------------

/// A registry-aware view over the toolset table. Wraps a [`ToolsetRegistry`]
/// and exposes the full Python API surface (`get_toolset`, `resolve_toolset`,
/// etc.) with dynamic plugin / platform merging.
pub struct Toolsets<'r, R: ToolsetRegistry + ?Sized> {
    registry: &'r R,
}

impl<'r, R: ToolsetRegistry + ?Sized> Toolsets<'r, R> {
    pub fn new(registry: &'r R) -> Self {
        Self { registry }
    }

    /// Toolset names registered by plugins (in the registry but not in
    /// `TOOLSETS`). Mirrors `_get_plugin_toolset_names`.
    fn plugin_toolset_names(&self) -> BTreeSet<String> {
        self.registry
            .registered_toolset_names()
            .into_iter()
            .filter(|n| !in_toolsets(n))
            .collect()
    }

    /// Synthetic `hermes-<platform>` names for registered plugin platforms.
    /// Mirrors `_get_dynamic_platform_toolset_names`.
    fn dynamic_platform_toolset_names(&self) -> BTreeSet<String> {
        self.registry
            .plugin_platform_entries()
            .into_iter()
            .map(|e| format!("hermes-{}", e.name))
            .filter(|n| !in_toolsets(n))
            .collect()
    }

    /// Explicit toolset aliases registered in the registry.
    /// Mirrors `_get_registry_toolset_aliases`.
    fn registry_toolset_aliases(&self) -> HashMap<String, String> {
        self.registry.registered_toolset_aliases()
    }

    /// Build a synthetic toolset for a registered plugin platform.
    /// Mirrors `_get_dynamic_platform_toolset`.
    fn dynamic_platform_toolset(&self, name: &str) -> Option<ToolsetDef> {
        let platform_name = name.strip_prefix("hermes-")?;
        let entry = self
            .registry
            .plugin_platform_entries()
            .into_iter()
            .find(|e| e.name == platform_name)?;

        let extra_tools = self.registry.tool_names_for_toolset(platform_name);
        let tools = sorted_unique(core_tools_vec().into_iter().chain(extra_tools));

        Some(ToolsetDef::new(
            format!(
                "{} platform toolset - full Hermes core tools plus platform-specific plugin tools",
                entry.label
            ),
            tools,
            vec![],
        ))
    }

    /// Get a toolset definition by name. Mirrors `get_toolset`.
    pub fn get_toolset(&self, name: &str) -> Option<ToolsetDef> {
        let toolset = lookup_static_or_custom(name);

        // Python: when the registry import fails, return the static toolset
        // (or None). With our trait, the registry is always present; a
        // NullRegistry naturally yields the merged-with-empty result, which is
        // equivalent for the static case.
        if let Some(ts) = toolset {
            let merged = sorted_unique(
                ts.tools
                    .iter()
                    .cloned()
                    .chain(self.registry.tool_names_for_toolset(name)),
            );
            return Some(ToolsetDef {
                description: ts.description,
                tools: merged,
                includes: ts.includes,
            });
        }

        if let Some(dynamic) = self.dynamic_platform_toolset(name) {
            return Some(dynamic);
        }

        let mut registry_toolset = name.to_string();
        let mut description = format!("Plugin toolset: {name}");
        let alias_target = self.registry.toolset_alias_target(name);

        if !self.plugin_toolset_names().contains(name) {
            match alias_target {
                Some(target) if !target.is_empty() => {
                    registry_toolset = target;
                    description = format!("MCP server '{name}' tools");
                }
                _ => return None,
            }
        } else {
            // reverse_aliases: canonical -> alias, only for aliases not in TOOLSETS
            let reverse: HashMap<String, String> = self
                .registry_toolset_aliases()
                .into_iter()
                .filter(|(alias, _)| !in_toolsets(alias))
                .map(|(alias, canonical)| (canonical, alias))
                .collect();
            if let Some(alias) = reverse.get(name) {
                description = format!("MCP server '{alias}' tools");
            }
        }

        Some(ToolsetDef {
            description,
            tools: self.registry.tool_names_for_toolset(&registry_toolset),
            includes: vec![],
        })
    }

    /// Recursively resolve a toolset to all its tool names.
    /// Mirrors `resolve_toolset`.
    pub fn resolve_toolset(&self, name: &str) -> Vec<String> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        self.resolve_inner(name, &mut visited)
    }

    fn resolve_inner(&self, name: &str, visited: &mut BTreeSet<String>) -> Vec<String> {
        // Special aliases: all tools across every toolset.
        if name == "all" || name == "*" {
            let mut all_tools: BTreeSet<String> = BTreeSet::new();
            for toolset_name in self.get_toolset_names() {
                // Fresh visited set per branch to avoid cross-branch contamination.
                let mut branch = visited.clone();
                for t in self.resolve_inner(&toolset_name, &mut branch) {
                    all_tools.insert(t);
                }
            }
            return all_tools.into_iter().collect();
        }

        if visited.contains(name) {
            return Vec::new();
        }
        visited.insert(name.to_string());

        let toolset = match self.get_toolset(name) {
            Some(ts) => ts,
            None => return Vec::new(),
        };

        let mut tools: BTreeSet<String> = toolset.tools.into_iter().collect();
        for included in &toolset.includes {
            for t in self.resolve_inner(included, visited) {
                tools.insert(t);
            }
        }
        tools.into_iter().collect()
    }

    /// Resolve multiple toolsets and combine (deduplicated, sorted).
    /// Mirrors `resolve_multiple_toolsets`.
    pub fn resolve_multiple_toolsets<I, S>(&self, names: I) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut all: BTreeSet<String> = BTreeSet::new();
        for name in names {
            for t in self.resolve_toolset(name.as_ref()) {
                all.insert(t);
            }
        }
        all.into_iter().collect()
    }

    /// All available toolsets (static + custom + plugin + dynamic platform).
    /// Mirrors `get_all_toolsets`.
    pub fn get_all_toolsets(&self) -> BTreeMap<String, ToolsetDef> {
        // Start with the local table, but resolved through get_toolset so
        // registry tools are merged in (matching Python `dict(TOOLSETS)` which
        // holds the raw defs — but Python's get_all_toolsets seeds with raw
        // TOOLSETS then only calls get_toolset for plugin/dynamic names).
        let mut result: BTreeMap<String, ToolsetDef> = BTreeMap::new();
        for name in all_local_names() {
            if let Some(def) = lookup_static_or_custom(&name) {
                result.insert(name, def);
            }
        }

        let aliases = self.registry_toolset_aliases();
        for ts_name in self.plugin_toolset_names() {
            // for-else: pick alias display name if one maps to this canonical
            // and isn't itself in TOOLSETS; otherwise keep ts_name.
            let mut display_name = ts_name.clone();
            for (alias, canonical) in aliases.iter() {
                if canonical == &ts_name && !in_toolsets(alias) {
                    display_name = alias.clone();
                    break;
                }
            }
            if result.contains_key(&display_name) {
                continue;
            }
            if let Some(toolset) = self.get_toolset(&display_name) {
                result.insert(display_name, toolset);
            }
        }

        for ts_name in self.dynamic_platform_toolset_names() {
            if result.contains_key(&ts_name) {
                continue;
            }
            if let Some(toolset) = self.get_toolset(&ts_name) {
                result.insert(ts_name, toolset);
            }
        }

        result
    }

    /// Names of all available toolsets (excluding aliases that ARE plain
    /// plugin names). Mirrors `get_toolset_names`.
    pub fn get_toolset_names(&self) -> Vec<String> {
        let mut names: BTreeSet<String> = all_local_names().into_iter().collect();
        let aliases = self.registry_toolset_aliases();
        for ts_name in self.plugin_toolset_names() {
            // for-else semantics: if an alias maps to this canonical (and the
            // alias is not itself in TOOLSETS), add the alias and break;
            // otherwise add ts_name.
            let mut matched = false;
            for (alias, canonical) in aliases.iter() {
                if canonical == &ts_name && !in_toolsets(alias) {
                    names.insert(alias.clone());
                    matched = true;
                    break;
                }
            }
            if !matched {
                names.insert(ts_name);
            }
        }
        for n in self.dynamic_platform_toolset_names() {
            names.insert(n);
        }
        names.into_iter().collect()
    }

    /// Whether a toolset name is valid. Mirrors `validate_toolset`.
    pub fn validate_toolset(&self, name: &str) -> bool {
        if name == "all" || name == "*" {
            return true;
        }
        if in_toolsets(name) {
            return true;
        }
        if self.plugin_toolset_names().contains(name) {
            return true;
        }
        if self.dynamic_platform_toolset_names().contains(name) {
            return true;
        }
        self.registry_toolset_aliases().contains_key(name)
    }

    /// Detailed info about a toolset including resolved tools.
    /// Mirrors `get_toolset_info`.
    pub fn get_toolset_info(&self, name: &str) -> Option<ToolsetInfo> {
        let toolset = self.get_toolset(name)?;
        let resolved = self.resolve_toolset(name);
        Some(ToolsetInfo {
            name: name.to_string(),
            description: toolset.description,
            direct_tools: toolset.tools,
            includes: toolset.includes.clone(),
            tool_count: resolved.len(),
            is_composite: !toolset.includes.is_empty(),
            resolved_tools: resolved,
        })
    }
}

/// Sort + dedup an iterator of strings (Python `sorted(set(...))`).
fn sorted_unique<I: IntoIterator<Item = String>>(items: I) -> Vec<String> {
    let set: BTreeSet<String> = items.into_iter().collect();
    set.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Module-level free functions (NullRegistry — static + custom only)
// ---------------------------------------------------------------------------

fn null_handle() -> Toolsets<'static, NullRegistry> {
    static NULL: NullRegistry = NullRegistry;
    Toolsets::new(&NULL)
}

/// Get a toolset definition by name (static + custom; no registry).
pub fn get_toolset(name: &str) -> Option<ToolsetDef> {
    null_handle().get_toolset(name)
}

/// Recursively resolve a toolset to all its tool names.
pub fn resolve_toolset(name: &str) -> Vec<String> {
    null_handle().resolve_toolset(name)
}

/// Resolve multiple toolsets and combine their tools (deduplicated, sorted).
pub fn resolve_multiple_toolsets<I, S>(names: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    null_handle().resolve_multiple_toolsets(names)
}

/// Get all available toolsets with their definitions.
pub fn get_all_toolsets() -> BTreeMap<String, ToolsetDef> {
    null_handle().get_all_toolsets()
}

/// Get names of all available toolsets.
pub fn get_toolset_names() -> Vec<String> {
    null_handle().get_toolset_names()
}

/// Check if a toolset name is valid.
pub fn validate_toolset(name: &str) -> bool {
    null_handle().validate_toolset(name)
}

/// Get detailed information about a toolset including resolved tools.
pub fn get_toolset_info(name: &str) -> Option<ToolsetInfo> {
    null_handle().get_toolset_info(name)
}

/// Create a custom toolset at runtime. Mirrors `create_custom_toolset`.
/// A `None` for tools/includes is treated as an empty list (Python `or []`).
pub fn create_custom_toolset(
    name: &str,
    description: &str,
    tools: Option<Vec<String>>,
    includes: Option<Vec<String>>,
) {
    let def = ToolsetDef::new(
        description,
        tools.unwrap_or_default(),
        includes.unwrap_or_default(),
    );
    let mut custom = custom_table().lock().unwrap();
    // Mirror dict assignment: replace any existing entry with the same name.
    if let Some(slot) = custom.iter_mut().find(|(n, _)| n == name) {
        slot.1 = def;
    } else {
        custom.push((name.to_string(), def));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_resolves_to_two_tools() {
        let mut tools = resolve_toolset("web");
        tools.sort();
        assert_eq!(tools, vec!["web_extract".to_string(), "web_search".to_string()]);
    }

    #[test]
    fn terminal_resolves() {
        let tools = resolve_toolset("terminal");
        assert_eq!(tools, vec!["process".to_string(), "terminal".to_string()]);
    }

    #[test]
    fn safe_composes_includes_only() {
        // safe includes web, vision, image_gen and has no direct tools.
        let tools = resolve_toolset("safe");
        assert!(tools.contains(&"web_search".to_string()));
        assert!(tools.contains(&"web_extract".to_string()));
        assert!(tools.contains(&"vision_analyze".to_string()));
        assert!(tools.contains(&"image_generate".to_string()));
        assert!(!tools.contains(&"terminal".to_string()));
        // sorted
        let mut sorted = tools.clone();
        sorted.sort();
        assert_eq!(tools, sorted);
    }

    #[test]
    fn debugging_merges_direct_and_includes() {
        let tools = resolve_toolset("debugging");
        // direct: terminal, process
        assert!(tools.contains(&"terminal".to_string()));
        assert!(tools.contains(&"process".to_string()));
        // from web include
        assert!(tools.contains(&"web_search".to_string()));
        // from file include
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"patch".to_string()));
    }

    #[test]
    fn unknown_toolset_resolves_empty() {
        assert!(resolve_toolset("does-not-exist").is_empty());
        assert!(get_toolset("does-not-exist").is_none());
    }

    #[test]
    fn hermes_cli_has_core_tools() {
        let ts = get_toolset("hermes-cli").unwrap();
        assert!(ts.tools.contains(&"send_message".to_string()));
        assert!(ts.tools.contains(&"kanban_link".to_string()));
        // hermes-cli has no includes -> leaf
        assert!(ts.includes.is_empty());
    }

    #[test]
    fn hermes_discord_adds_discord_tools() {
        let ts = get_toolset("hermes-discord").unwrap();
        assert!(ts.tools.contains(&"discord".to_string()));
        assert!(ts.tools.contains(&"discord_admin".to_string()));
        assert!(ts.tools.contains(&"web_search".to_string()));
    }

    #[test]
    fn hermes_gateway_unions_platforms() {
        let resolved = resolve_toolset("hermes-gateway");
        // From included platform toolsets: core tools plus discord/feishu/yuanbao extras.
        assert!(resolved.contains(&"send_message".to_string()));
        assert!(resolved.contains(&"discord".to_string()));
        assert!(resolved.contains(&"feishu_doc_read".to_string()));
        assert!(resolved.contains(&"yb_send_dm".to_string()));
        // sorted output
        let mut sorted = resolved.clone();
        sorted.sort();
        assert_eq!(resolved, sorted);
    }

    #[test]
    fn resolve_multiple_dedups() {
        let combined = resolve_multiple_toolsets(["web", "vision", "terminal"]);
        assert!(combined.contains(&"web_search".to_string()));
        assert!(combined.contains(&"vision_analyze".to_string()));
        assert!(combined.contains(&"terminal".to_string()));
        // dedup: web_search appears once
        let count = combined.iter().filter(|t| *t == "web_search").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn validate_special_aliases() {
        assert!(validate_toolset("all"));
        assert!(validate_toolset("*"));
        assert!(validate_toolset("web"));
        assert!(!validate_toolset("nope"));
    }

    #[test]
    fn toolset_info_composite_flag() {
        let safe = get_toolset_info("safe").unwrap();
        assert!(safe.is_composite);
        assert!(safe.direct_tools.is_empty());
        assert!(safe.tool_count > 0);
        assert_eq!(safe.tool_count, safe.resolved_tools.len());

        let web = get_toolset_info("web").unwrap();
        assert!(!web.is_composite);
    }

    #[test]
    fn all_alias_includes_everything() {
        let all = resolve_toolset("all");
        // Should be a union; contains tools from many toolsets.
        assert!(all.contains(&"web_search".to_string()));
        assert!(all.contains(&"mixture_of_agents".to_string()));
        assert!(all.contains(&"video_analyze".to_string()));
        assert!(all.contains(&"spotify_playback".to_string()));
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(all, sorted);
    }

    #[test]
    fn get_toolset_names_includes_statics() {
        let names = get_toolset_names();
        assert!(names.contains(&"web".to_string()));
        assert!(names.contains(&"hermes-gateway".to_string()));
        // sorted
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn custom_toolset_creation() {
        create_custom_toolset(
            "my_custom_test",
            "My custom toolset",
            Some(vec!["web_search".to_string()]),
            Some(vec!["terminal".to_string(), "vision".to_string()]),
        );
        let info = get_toolset_info("my_custom_test").unwrap();
        assert_eq!(info.description, "My custom toolset");
        assert!(info.is_composite);
        assert!(info.resolved_tools.contains(&"web_search".to_string()));
        assert!(info.resolved_tools.contains(&"terminal".to_string()));
        assert!(info.resolved_tools.contains(&"vision_analyze".to_string()));
        assert!(validate_toolset("my_custom_test"));
    }

    #[test]
    fn registry_merges_extra_tools() {
        struct R;
        impl ToolsetRegistry for R {
            fn tool_names_for_toolset(&self, name: &str) -> Vec<String> {
                if name == "web" {
                    vec!["plugin_web_tool".to_string()]
                } else {
                    Vec::new()
                }
            }
            fn registered_toolset_names(&self) -> Vec<String> {
                vec!["my_plugin_ts".to_string()]
            }
            fn plugin_platform_entries(&self) -> Vec<PlatformEntry> {
                vec![PlatformEntry {
                    name: "myplat".to_string(),
                    label: "MyPlatform".to_string(),
                }]
            }
        }
        let r = R;
        let h = Toolsets::new(&r);

        // Static toolset merges registry tools.
        let web = h.get_toolset("web").unwrap();
        assert!(web.tools.contains(&"plugin_web_tool".to_string()));
        assert!(web.tools.contains(&"web_search".to_string()));

        // Plugin toolset name is valid and discoverable.
        assert!(h.validate_toolset("my_plugin_ts"));
        assert!(h.get_toolset_names().contains(&"my_plugin_ts".to_string()));

        // Dynamic platform toolset synthesized.
        let plat = h.get_toolset("hermes-myplat").unwrap();
        assert!(plat.description.contains("MyPlatform"));
        assert!(plat.tools.contains(&"send_message".to_string()));
        assert!(h.validate_toolset("hermes-myplat"));
    }

    #[test]
    fn cycle_detection_in_includes() {
        create_custom_toolset(
            "cycle_a",
            "a",
            None,
            Some(vec!["cycle_b".to_string()]),
        );
        create_custom_toolset(
            "cycle_b",
            "b",
            Some(vec!["t1".to_string()]),
            Some(vec!["cycle_a".to_string()]),
        );
        let resolved = resolve_toolset("cycle_a");
        // Should terminate and include t1 from cycle_b.
        assert!(resolved.contains(&"t1".to_string()));
    }
}
