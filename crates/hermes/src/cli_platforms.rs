//! Shared platform registry for Hermes Agent.
//!
//! Single source of truth for platform metadata consumed by both
//! `skills_config` (label display) and `tools_config` (default toolset
//! resolution). Use [`PLATFORMS`] / [`platforms`] here instead of
//! maintaining duplicate maps in each module.
//!
//! Native Rust port of `hermes_cli/platforms.py`.
//!
//! The Python module also consults a dynamic `gateway.platform_registry`
//! (wrapped in a `try/except` so it degrades gracefully when unavailable).
//! That registry is generic in the Rust codebase, so this port exposes the
//! plugin merge points via [`PluginEntry`] and the
//! [`platform_label_with_plugins`] / [`get_all_platforms_with_plugins`]
//! helpers. The bare [`platform_label`] / [`get_all_platforms`] functions
//! mirror the Python behavior when the plugin registry lookup fails (i.e.
//! they only consider the static builtins).

/// Metadata for a single platform entry.
///
/// Mirrors the Python `PlatformInfo` `NamedTuple`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    /// Human-readable display label (may contain a leading emoji).
    pub label: String,
    /// Default toolset name resolved for this platform.
    pub default_toolset: String,
}

impl PlatformInfo {
    /// Construct a [`PlatformInfo`] from string-like parts.
    pub fn new(label: impl Into<String>, default_toolset: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            default_toolset: default_toolset.into(),
        }
    }
}

/// Raw, ordered builtin platform table.
///
/// `(key, label, default_toolset)` in the exact order defined by the Python
/// `OrderedDict`, so TUI menus are deterministic.
pub const PLATFORMS: &[(&str, &str, &str)] = &[
    ("cli", "🖥️  CLI", "hermes-cli"),
    ("telegram", "📱 Telegram", "hermes-telegram"),
    ("discord", "💬 Discord", "hermes-discord"),
    ("slack", "💼 Slack", "hermes-slack"),
    ("whatsapp", "📱 WhatsApp", "hermes-whatsapp"),
    ("signal", "📡 Signal", "hermes-signal"),
    ("bluebubbles", "💙 BlueBubbles", "hermes-bluebubbles"),
    ("email", "📧 Email", "hermes-email"),
    ("homeassistant", "🏠 Home Assistant", "hermes-homeassistant"),
    ("mattermost", "💬 Mattermost", "hermes-mattermost"),
    ("matrix", "💬 Matrix", "hermes-matrix"),
    ("dingtalk", "💬 DingTalk", "hermes-dingtalk"),
    ("feishu", "🪽 Feishu", "hermes-feishu"),
    ("wecom", "💬 WeCom", "hermes-wecom"),
    ("wecom_callback", "💬 WeCom Callback", "hermes-wecom-callback"),
    ("weixin", "💬 Weixin", "hermes-weixin"),
    ("qqbot", "💬 QQBot", "hermes-qqbot"),
    ("yuanbao", "🤖 Yuanbao", "hermes-yuanbao"),
    ("webhook", "🔗 Webhook", "hermes-webhook"),
    ("api_server", "🌐 API Server", "hermes-api-server"),
    ("cron", "⏰ Cron", "hermes-cron"),
];

/// Return the ordered builtin platform table as `(key, PlatformInfo)` pairs.
///
/// Order matches [`PLATFORMS`] (and the Python `OrderedDict`).
pub fn platforms() -> Vec<(String, PlatformInfo)> {
    PLATFORMS
        .iter()
        .map(|(key, label, toolset)| (key.to_string(), PlatformInfo::new(*label, *toolset)))
        .collect()
}

/// Look up the [`PlatformInfo`] for a builtin platform key.
///
/// Returns `None` for keys not present in the static [`PLATFORMS`] table.
pub fn platform_info(key: &str) -> Option<PlatformInfo> {
    PLATFORMS
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, label, toolset)| PlatformInfo::new(*label, *toolset))
}

/// A dynamically-registered plugin platform entry.
///
/// Equivalent to an entry from `gateway.platform_registry` on the Python side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginEntry {
    /// Registry name / key (e.g. `"irc"`).
    pub name: String,
    /// Human-readable label (e.g. `"IRC"`).
    pub label: String,
    /// Optional leading emoji; empty string means "no emoji".
    pub emoji: String,
}

impl PluginEntry {
    /// Build the composed display label for this plugin entry.
    ///
    /// Mirrors the Python expression:
    /// `f"{entry.emoji}  {entry.label}" if entry.emoji else entry.label`.
    pub fn display_label(&self) -> String {
        if self.emoji.is_empty() {
            self.label.clone()
        } else {
            format!("{}  {}", self.emoji, self.label)
        }
    }
}

/// Return the display label for a platform `key`, or `default`.
///
/// Mirrors Python `platform_label` when the dynamic plugin registry is
/// unavailable: only the static [`PLATFORMS`] table is consulted. Use
/// [`platform_label_with_plugins`] to also consult a plugin lookup.
pub fn platform_label(key: &str, default: &str) -> String {
    match platform_info(key) {
        Some(info) => info.label,
        None => default.to_string(),
    }
}

/// Return the display label for a platform `key`, consulting builtins first
/// and then a plugin registry lookup, falling back to `default`.
///
/// `plugin_lookup` plays the role of `platform_registry.get(key)` in Python:
/// it returns the matching [`PluginEntry`] for a key, or `None`.
pub fn platform_label_with_plugins<F>(key: &str, default: &str, plugin_lookup: F) -> String
where
    F: FnOnce(&str) -> Option<PluginEntry>,
{
    if let Some(info) = platform_info(key) {
        return info.label;
    }
    if let Some(entry) = plugin_lookup(key) {
        return entry.display_label();
    }
    default.to_string()
}

/// Return the builtin [`PLATFORMS`] as an ordered list.
///
/// Mirrors Python `get_all_platforms` when the plugin registry is
/// unavailable. Use [`get_all_platforms_with_plugins`] to append
/// plugin-registered platforms.
pub fn get_all_platforms() -> Vec<(String, PlatformInfo)> {
    platforms()
}

/// Return the builtins merged with any plugin-registered platforms.
///
/// Plugin platforms are appended after builtins, and only if their `name`
/// is not already present (matching Python's `if entry.name not in merged`).
/// This is the function that `tools_config` / `skills_config` should use for
/// platform menus.
///
/// `plugin_entries` plays the role of `platform_registry.plugin_entries()`.
pub fn get_all_platforms_with_plugins(plugin_entries: Vec<PluginEntry>) -> Vec<(String, PlatformInfo)> {
    let mut merged = platforms();
    for entry in plugin_entries {
        if merged.iter().any(|(k, _)| *k == entry.name) {
            continue;
        }
        let label = entry.display_label();
        let default_toolset = format!("hermes-{}", entry.name);
        merged.push((entry.name.clone(), PlatformInfo::new(label, default_toolset)));
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_count_and_order() {
        let all = platforms();
        assert_eq!(all.len(), 21);
        assert_eq!(all[0].0, "cli");
        assert_eq!(all[1].0, "telegram");
        assert_eq!(all.last().unwrap().0, "cron");
    }

    #[test]
    fn platform_info_lookup() {
        let info = platform_info("slack").unwrap();
        assert_eq!(info.label, "💼 Slack");
        assert_eq!(info.default_toolset, "hermes-slack");
        assert!(platform_info("nope").is_none());
    }

    #[test]
    fn label_known_and_default() {
        assert_eq!(platform_label("telegram", ""), "📱 Telegram");
        assert_eq!(platform_label("cron", "x"), "⏰ Cron");
        assert_eq!(platform_label("unknown", "fallback"), "fallback");
        assert_eq!(platform_label("unknown", ""), "");
    }

    #[test]
    fn label_with_plugins_prefers_builtin() {
        // Builtin wins even if a plugin lookup would return something.
        let lbl = platform_label_with_plugins("slack", "def", |_| {
            Some(PluginEntry {
                name: "slack".into(),
                label: "OtherSlack".into(),
                emoji: "X".into(),
            })
        });
        assert_eq!(lbl, "💼 Slack");
    }

    #[test]
    fn label_with_plugins_uses_registry() {
        let lbl = platform_label_with_plugins("irc", "def", |k| {
            assert_eq!(k, "irc");
            Some(PluginEntry {
                name: "irc".into(),
                label: "IRC".into(),
                emoji: "🛰".into(),
            })
        });
        assert_eq!(lbl, "🛰  IRC");

        let no_emoji = platform_label_with_plugins("viber", "def", |_| {
            Some(PluginEntry {
                name: "viber".into(),
                label: "Viber".into(),
                emoji: "".into(),
            })
        });
        assert_eq!(no_emoji, "Viber");
    }

    #[test]
    fn label_with_plugins_falls_back_to_default() {
        let lbl = platform_label_with_plugins("ghost", "fallback", |_| None);
        assert_eq!(lbl, "fallback");
    }

    #[test]
    fn display_label_emoji_rules() {
        let with = PluginEntry {
            name: "irc".into(),
            label: "IRC".into(),
            emoji: "🛰".into(),
        };
        assert_eq!(with.display_label(), "🛰  IRC");
        let without = PluginEntry {
            name: "irc".into(),
            label: "IRC".into(),
            emoji: "".into(),
        };
        assert_eq!(without.display_label(), "IRC");
    }

    #[test]
    fn get_all_platforms_is_builtins() {
        assert_eq!(get_all_platforms(), platforms());
    }

    #[test]
    fn merge_appends_new_plugins() {
        let merged = get_all_platforms_with_plugins(vec![
            PluginEntry {
                name: "irc".into(),
                label: "IRC".into(),
                emoji: "🛰".into(),
            },
            PluginEntry {
                name: "viber".into(),
                label: "Viber".into(),
                emoji: "".into(),
            },
        ]);
        assert_eq!(merged.len(), 23);
        let irc = &merged[21];
        assert_eq!(irc.0, "irc");
        assert_eq!(irc.1.label, "🛰  IRC");
        assert_eq!(irc.1.default_toolset, "hermes-irc");
        let viber = &merged[22];
        assert_eq!(viber.0, "viber");
        assert_eq!(viber.1.label, "Viber");
        assert_eq!(viber.1.default_toolset, "hermes-viber");
    }

    #[test]
    fn merge_skips_existing_keys() {
        let merged = get_all_platforms_with_plugins(vec![PluginEntry {
            name: "slack".into(),
            label: "Shadow".into(),
            emoji: "X".into(),
        }]);
        // No new entry added; builtin slack untouched.
        assert_eq!(merged.len(), 21);
        let slack = merged.iter().find(|(k, _)| k == "slack").unwrap();
        assert_eq!(slack.1.label, "💼 Slack");
    }
}
