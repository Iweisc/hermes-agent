//! Per-platform display/verbosity configuration resolver.
//!
//! Faithful port of `gateway/display_config.py`. Resolves a display setting
//! with platform-specific overrides and built-in tiered defaults.
//!
//! Resolution order (first non-None wins):
//!   1. `display.platforms.<platform>.<key>` — explicit per-platform override
//!   2. `display.<key>` — global user setting (skipped for `streaming`)
//!   3. `_PLATFORM_DEFAULTS[<platform>][<key>]` — built-in platform default
//!   4. `_GLOBAL_DEFAULTS[<key>]` — built-in global default
//!   5. caller fallback
//!
//! `display.streaming` is CLI-only; gateway streaming follows the top-level
//! `streaming` config unless a per-platform override sets it. Legacy
//! `display.tool_progress_overrides.<platform>` is read as a `tool_progress`
//! fallback.

use serde_yaml::Value as YamlValue;

/// A resolved display setting value. Mirrors the Python return which can be a
/// string (`tool_progress`), bool (`show_reasoning`/`streaming`), int
/// (`tool_preview_length`), or absent.
#[derive(Debug, Clone, PartialEq)]
pub enum DisplaySetting {
    Str(String),
    Bool(bool),
    Int(i64),
    /// `streaming` global default is "follow top-level config" (Python None).
    None,
}

impl DisplaySetting {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DisplaySetting::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DisplaySetting::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            DisplaySetting::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn is_none(&self) -> bool {
        matches!(self, DisplaySetting::None)
    }
}

/// Per-platform overrideable keys + their global defaults (Python `_GLOBAL_DEFAULTS`).
fn global_default(setting: &str) -> Option<DisplaySetting> {
    match setting {
        "tool_progress" => Some(DisplaySetting::Str("all".to_string())),
        "show_reasoning" => Some(DisplaySetting::Bool(false)),
        "tool_preview_length" => Some(DisplaySetting::Int(0)),
        "streaming" => Some(DisplaySetting::None), // follow top-level
        _ => None,
    }
}

/// The set of keys that participate in per-platform resolution.
pub fn overrideable_keys() -> [&'static str; 4] {
    ["tool_progress", "show_reasoning", "tool_preview_length", "streaming"]
}

/// Capability tiers (Python `_TIER_*`): (tool_progress, show_reasoning,
/// tool_preview_length, streaming). `streaming` is `None` -> follow-global,
/// `Some(false)` -> forced off.
struct Tier {
    tool_progress: &'static str,
    show_reasoning: bool,
    tool_preview_length: i64,
    streaming: Option<bool>,
}

const TIER_HIGH: Tier = Tier { tool_progress: "all", show_reasoning: false, tool_preview_length: 40, streaming: None };
const TIER_MEDIUM: Tier = Tier { tool_progress: "new", show_reasoning: false, tool_preview_length: 40, streaming: None };
const TIER_LOW: Tier = Tier { tool_progress: "off", show_reasoning: false, tool_preview_length: 40, streaming: Some(false) };
const TIER_MINIMAL: Tier = Tier { tool_progress: "off", show_reasoning: false, tool_preview_length: 0, streaming: Some(false) };

/// Look up a built-in platform default for `setting`, returning `None` when the
/// platform is unknown OR the resolved value is "None-like" (matching Python's
/// `plat_defaults.get(setting)` returning None and falling through).
fn platform_default(platform_key: &str, setting: &str) -> Option<DisplaySetting> {
    // Platform -> tier, with the two documented per-platform overrides
    // (slack forces tool_progress off; api_server forces tool_preview_length 0).
    let (tier, tp_override, tpl_override): (&Tier, Option<&str>, Option<i64>) = match platform_key {
        "telegram" | "discord" => (&TIER_HIGH, None, None),
        "slack" => (&TIER_MEDIUM, Some("off"), None),
        "mattermost" | "matrix" | "feishu" => (&TIER_MEDIUM, None, None),
        "signal" | "bluebubbles" | "weixin" | "wecom" | "wecom_callback" | "dingtalk" => {
            (&TIER_LOW, None, None)
        }
        "whatsapp" => (&TIER_MEDIUM, None, None),
        "email" | "sms" | "webhook" | "homeassistant" => (&TIER_MINIMAL, None, None),
        "api_server" => (&TIER_HIGH, None, Some(0)),
        _ => return None,
    };

    match setting {
        "tool_progress" => Some(DisplaySetting::Str(
            tp_override.unwrap_or(tier.tool_progress).to_string(),
        )),
        "show_reasoning" => Some(DisplaySetting::Bool(tier.show_reasoning)),
        "tool_preview_length" => {
            Some(DisplaySetting::Int(tpl_override.unwrap_or(tier.tool_preview_length)))
        }
        // streaming: None means "not set" at this layer (Python `None` value is
        // skipped by the `if val is not None` guard, falling through to global).
        "streaming" => tier.streaming.map(DisplaySetting::Bool),
        _ => None,
    }
}

fn yaml_get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

/// Normalize a raw YAML value for `setting`, mirroring Python `_normalise`.
fn normalise(setting: &str, value: &YamlValue) -> DisplaySetting {
    match setting {
        "tool_progress" => match value {
            YamlValue::Bool(false) => DisplaySetting::Str("off".to_string()),
            YamlValue::Bool(true) => DisplaySetting::Str("all".to_string()),
            YamlValue::String(s) => DisplaySetting::Str(s.to_lowercase()),
            YamlValue::Number(n) => DisplaySetting::Str(n.to_string().to_lowercase()),
            other => DisplaySetting::Str(yaml_scalar_str(other).to_lowercase()),
        },
        "show_reasoning" | "streaming" => {
            let b = match value {
                YamlValue::String(s) => {
                    matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "on")
                }
                YamlValue::Bool(b) => *b,
                YamlValue::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                YamlValue::Null => false,
                _ => true, // non-empty seq/map are truthy in Python bool()
            };
            DisplaySetting::Bool(b)
        }
        "tool_preview_length" => {
            let i = match value {
                YamlValue::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
                YamlValue::String(s) => s.trim().parse::<i64>().ok(),
                YamlValue::Bool(b) => Some(*b as i64),
                _ => None,
            };
            DisplaySetting::Int(i.unwrap_or(0))
        }
        _ => match value {
            YamlValue::String(s) => DisplaySetting::Str(s.clone()),
            YamlValue::Bool(b) => DisplaySetting::Bool(*b),
            YamlValue::Number(n) => DisplaySetting::Int(n.as_i64().unwrap_or(0)),
            _ => DisplaySetting::None,
        },
    }
}

fn yaml_scalar_str(value: &YamlValue) -> String {
    match value {
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => {
            // Python str(True) == "True"; lower() applied by caller.
            if *b { "True".to_string() } else { "False".to_string() }
        }
        YamlValue::Number(n) => n.to_string(),
        YamlValue::Null => "None".to_string(),
        _ => String::new(),
    }
}

/// A YAML value is "present" (Python `is not None`) when it is anything other
/// than an explicit null.
fn is_present(value: &YamlValue) -> bool {
    !matches!(value, YamlValue::Null)
}

/// Resolve a display setting with per-platform override support. Port of
/// `resolve_display_setting`. `user_config` is the full parsed config.yaml.
pub fn resolve_display_setting(
    user_config: &YamlValue,
    platform_key: &str,
    setting: &str,
    fallback: Option<DisplaySetting>,
) -> Option<DisplaySetting> {
    let display = user_config
        .as_mapping()
        .and_then(|root| yaml_get(root, "display"))
        .and_then(YamlValue::as_mapping);

    if let Some(display) = display {
        // 1. Explicit per-platform override.
        if let Some(YamlValue::Mapping(platforms)) = yaml_get(display, "platforms") {
            if let Some(YamlValue::Mapping(plat)) = yaml_get(platforms, platform_key) {
                if let Some(val) = yaml_get(plat, setting) {
                    if is_present(val) {
                        return Some(normalise(setting, val));
                    }
                }
            }
        }

        // 1b. Backward compat: display.tool_progress_overrides.<platform>.
        if setting == "tool_progress" {
            if let Some(YamlValue::Mapping(legacy)) = yaml_get(display, "tool_progress_overrides") {
                if let Some(val) = yaml_get(legacy, platform_key) {
                    if is_present(val) {
                        return Some(normalise(setting, val));
                    }
                }
            }
        }

        // 2. Global user setting (skip streaming — CLI-only).
        if setting != "streaming" {
            if let Some(val) = yaml_get(display, setting) {
                if is_present(val) {
                    return Some(normalise(setting, val));
                }
            }
        }
    }

    // 3. Built-in platform default.
    if let Some(val) = platform_default(platform_key, setting) {
        return Some(val);
    }

    // 4. Built-in global default.
    if let Some(val) = global_default(setting) {
        // The global `streaming` default is None (follow top-level), which
        // Python returns as-is (it's the value, not absence).
        return Some(val);
    }

    // 5. Caller fallback.
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(text: &str) -> YamlValue {
        serde_yaml::from_str(text).unwrap()
    }

    #[test]
    fn platform_default_tiers() {
        let empty = yaml("{}");
        // telegram is TIER_HIGH
        assert_eq!(
            resolve_display_setting(&empty, "telegram", "tool_progress", None).unwrap(),
            DisplaySetting::Str("all".to_string())
        );
        assert_eq!(
            resolve_display_setting(&empty, "telegram", "tool_preview_length", None).unwrap(),
            DisplaySetting::Int(40)
        );
        // slack overrides tool_progress to off
        assert_eq!(
            resolve_display_setting(&empty, "slack", "tool_progress", None).unwrap(),
            DisplaySetting::Str("off".to_string())
        );
        // signal is TIER_LOW -> streaming forced off
        assert_eq!(
            resolve_display_setting(&empty, "signal", "streaming", None).unwrap(),
            DisplaySetting::Bool(false)
        );
        // api_server forces tool_preview_length 0
        assert_eq!(
            resolve_display_setting(&empty, "api_server", "tool_preview_length", None).unwrap(),
            DisplaySetting::Int(0)
        );
    }

    #[test]
    fn unknown_platform_falls_back_to_global_defaults() {
        let empty = yaml("{}");
        assert_eq!(
            resolve_display_setting(&empty, "mystery", "tool_progress", None).unwrap(),
            DisplaySetting::Str("all".to_string())
        );
        // streaming global default is None (follow top-level)
        assert!(
            resolve_display_setting(&empty, "mystery", "streaming", None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn per_platform_override_and_normalise() {
        let cfg = yaml(
            "display:\n  platforms:\n    telegram:\n      tool_progress: off\n      show_reasoning: \"yes\"\n      tool_preview_length: \"25\"\n",
        );
        // YAML bare `off` -> false -> normalised "off"
        assert_eq!(
            resolve_display_setting(&cfg, "telegram", "tool_progress", None).unwrap(),
            DisplaySetting::Str("off".to_string())
        );
        // string "yes" -> bool true
        assert_eq!(
            resolve_display_setting(&cfg, "telegram", "show_reasoning", None).unwrap(),
            DisplaySetting::Bool(true)
        );
        // string "25" -> int 25
        assert_eq!(
            resolve_display_setting(&cfg, "telegram", "tool_preview_length", None).unwrap(),
            DisplaySetting::Int(25)
        );
    }

    #[test]
    fn global_user_setting_applies_but_not_for_streaming() {
        let cfg = yaml("display:\n  tool_progress: new\n  streaming: true\n");
        // global tool_progress overrides platform default
        assert_eq!(
            resolve_display_setting(&cfg, "telegram", "tool_progress", None).unwrap(),
            DisplaySetting::Str("new".to_string())
        );
        // streaming global is skipped -> telegram TIER_HIGH streaming None
        assert!(
            resolve_display_setting(&cfg, "telegram", "streaming", None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn legacy_tool_progress_overrides() {
        let cfg = yaml("display:\n  tool_progress_overrides:\n    discord: off\n");
        assert_eq!(
            resolve_display_setting(&cfg, "discord", "tool_progress", None).unwrap(),
            DisplaySetting::Str("off".to_string())
        );
    }
}
