//! Per-platform display/verbosity configuration resolver.
//!
//! Provides [`resolve_display_setting`] — the single entry-point for reading
//! display settings with platform-specific overrides and sensible defaults.
//!
//! Resolution order (first non-None wins):
//!   1. `display.platforms.<platform>.<key>`  — explicit per-platform user override
//!   2. `display.<key>`                        — global user setting
//!   3. `_PLATFORM_DEFAULTS[<platform>][<key>]`  — built-in sensible default
//!   4. `_GLOBAL_DEFAULTS[<key>]`               — built-in global default
//!
//! Exception: `display.streaming` is CLI-only. Gateway streaming follows the
//! top-level `streaming` config unless `display.platforms.<platform>.streaming`
//! sets an explicit per-platform override.
//!
//! Backward compatibility: `display.tool_progress_overrides` is still read as a
//! fallback for `tool_progress` when no `display.platforms` entry exists.
//!
//! This is a faithful port of `gateway/display_config.py`. Because the Python
//! version operates on the parsed `config.yaml` dict, this port operates on
//! [`serde_yaml::Value`]. The resolved values are returned as a
//! [`DisplayValue`] enum so that callers can recover the dynamically-typed
//! result the Python function produced (`str`, `bool`, `int`, or `None`).

use serde_yaml::Value;

/// A resolved display setting value.
///
/// Mirrors the dynamically-typed return of the Python `resolve_display_setting`:
/// after normalisation a value can be a string (`tool_progress`), a bool
/// (`show_reasoning`, `streaming`), an integer (`tool_preview_length`), or
/// `None` (only `streaming`'s "follow global" default, or an unmatched
/// fallback).
#[derive(Debug, Clone, PartialEq)]
pub enum DisplayValue {
    /// No configured value (Python `None`) — e.g. `streaming` follow-global.
    None,
    Str(String),
    Bool(bool),
    Int(i64),
}

impl DisplayValue {
    /// Whether this value is "None" (Python falsy-for-resolution sentinel).
    pub fn is_none(&self) -> bool {
        matches!(self, DisplayValue::None)
    }

    /// Return the string value, if this is a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DisplayValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the bool value, if this is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DisplayValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Return the integer value, if this is an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            DisplayValue::Int(i) => Some(*i),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Overrideable display settings and their global defaults
// ---------------------------------------------------------------------------
// These are the settings that can be configured per-platform.
// Other display settings (compact, personality, skin, etc.) are CLI-only and
// don't participate in per-platform resolution.

/// The canonical list of per-platform overrideable setting keys (in insertion
/// order matching `_GLOBAL_DEFAULTS` in Python).
pub const OVERRIDEABLE_KEYS: &[&str] = &[
    "tool_progress",
    "show_reasoning",
    "tool_preview_length",
    "streaming",
];

/// Whether `key` is a per-platform overrideable display setting.
pub fn is_overrideable_key(key: &str) -> bool {
    OVERRIDEABLE_KEYS.contains(&key)
}

/// Built-in global default for a setting, or `None` if the key is unknown.
fn global_default(setting: &str) -> Option<DisplayValue> {
    match setting {
        "tool_progress" => Some(DisplayValue::Str("all".to_string())),
        "show_reasoning" => Some(DisplayValue::Bool(false)),
        "tool_preview_length" => Some(DisplayValue::Int(0)),
        // None = follow top-level streaming config
        "streaming" => Some(DisplayValue::None),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Sensible per-platform defaults — tiered by platform capability
// ---------------------------------------------------------------------------
// Tier 1 (high):    Supports message editing, typically personal/team use
// Tier 2 (medium):  Supports editing but often workspace/customer-facing
// Tier 3 (low):     No edit support — each progress msg is permanent
// Tier 4 (minimal): Batch/non-interactive delivery

/// A platform default tier. Returns the default `DisplayValue` for a setting,
/// or `None` if the setting key is not part of this tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    High,
    Medium,
    Low,
    Minimal,
}

impl Tier {
    fn lookup(self, setting: &str) -> Option<DisplayValue> {
        // For each tier, return (tool_progress, tool_preview_length, streaming).
        // show_reasoning is always false across all tiers.
        match setting {
            "show_reasoning" => Some(DisplayValue::Bool(false)),
            "tool_progress" => Some(DisplayValue::Str(
                match self {
                    Tier::High => "all",
                    Tier::Medium => "new",
                    Tier::Low => "off",
                    Tier::Minimal => "off",
                }
                .to_string(),
            )),
            "tool_preview_length" => Some(DisplayValue::Int(match self {
                Tier::High => 40,
                Tier::Medium => 40,
                Tier::Low => 40,
                Tier::Minimal => 0,
            })),
            "streaming" => Some(match self {
                // follow global
                Tier::High => DisplayValue::None,
                Tier::Medium => DisplayValue::None,
                Tier::Low => DisplayValue::Bool(false),
                Tier::Minimal => DisplayValue::Bool(false),
            }),
            _ => None,
        }
    }
}

/// Built-in per-platform default for a setting, or `None` if the platform is
/// unknown or doesn't define that setting.
fn platform_default(platform_key: &str, setting: &str) -> Option<DisplayValue> {
    // Some platforms layer overrides on top of a base tier. We replicate the
    // Python `{**_TIER_*, "key": val}` spread merges here.
    match platform_key {
        // Tier 1 — full edit support, personal/team use
        "telegram" | "discord" => Tier::High.lookup(setting),

        // Tier 2 — edit support, often customer/workspace channels
        // Slack: tool_progress off by default — Bolt posts cannot be edited
        // like CLI; "new"/"all" spam permanent lines in channels.
        "slack" => {
            if setting == "tool_progress" {
                Some(DisplayValue::Str("off".to_string()))
            } else {
                Tier::Medium.lookup(setting)
            }
        }
        "mattermost" | "matrix" | "feishu" => Tier::Medium.lookup(setting),

        // Tier 3 — no edit support, progress messages are permanent
        "signal" | "bluebubbles" | "weixin" | "wecom" | "wecom_callback"
        | "dingtalk" => Tier::Low.lookup(setting),
        // whatsapp: Baileys bridge supports /edit
        "whatsapp" => Tier::Medium.lookup(setting),

        // Tier 4 — batch or non-interactive delivery
        "email" | "sms" | "webhook" | "homeassistant" => Tier::Minimal.lookup(setting),

        "api_server" => {
            if setting == "tool_preview_length" {
                Some(DisplayValue::Int(0))
            } else {
                Tier::High.lookup(setting)
            }
        }

        _ => None,
    }
}

/// Whether `platform_key` has any built-in per-platform defaults.
///
/// Mirrors the `if plat_defaults:` truthiness check in Python: every entry in
/// `_PLATFORM_DEFAULTS` is a non-empty dict, so this is true for any known
/// platform key.
fn has_platform_defaults(platform_key: &str) -> bool {
    matches!(
        platform_key,
        "telegram"
            | "discord"
            | "slack"
            | "mattermost"
            | "matrix"
            | "feishu"
            | "signal"
            | "whatsapp"
            | "bluebubbles"
            | "weixin"
            | "wecom"
            | "wecom_callback"
            | "dingtalk"
            | "email"
            | "sms"
            | "webhook"
            | "homeassistant"
            | "api_server"
    )
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Look up `key` within a YAML mapping, returning the value if present and not
/// null. Returns `None` if `value` is not a mapping, the key is absent, or the
/// stored value is YAML null (mirrors Python's `dict.get(key)` returning
/// `None`, combined with the `if val is not None` guards).
fn map_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    let mapping = value.as_mapping()?;
    let v = mapping.get(Value::String(key.to_string()))?;
    if v.is_null() {
        None
    } else {
        Some(v)
    }
}

/// Resolve a display setting with per-platform override support.
///
/// # Parameters
/// - `user_config`: the full parsed `config.yaml` value.
/// - `platform_key`: platform config key (e.g. `"telegram"`, `"slack"`).
/// - `setting`: display setting name (e.g. `"tool_progress"`).
/// - `fallback`: fallback value when the setting isn't found anywhere.
///
/// Returns the resolved value, or `fallback` if nothing is configured.
pub fn resolve_display_setting(
    user_config: &Value,
    platform_key: &str,
    setting: &str,
    fallback: DisplayValue,
) -> DisplayValue {
    // display_cfg = user_config.get("display") or {}
    // `or {}` means a null/missing/falsy display becomes an empty mapping.
    let empty = Value::Mapping(serde_yaml::Mapping::new());
    let display_cfg = match user_config.as_mapping() {
        Some(m) => match m.get(Value::String("display".to_string())) {
            Some(v) if !v.is_null() && !is_falsy(v) => v,
            _ => &empty,
        },
        None => &empty,
    };

    // 1. Explicit per-platform override (display.platforms.<platform>.<key>)
    if let Some(platforms) = display_cfg
        .as_mapping()
        .and_then(|m| m.get(Value::String("platforms".to_string())))
    {
        if !is_falsy(platforms) {
            if let Some(plat_overrides) = map_get(platforms, platform_key) {
                if plat_overrides.is_mapping() {
                    if let Some(val) = map_get(plat_overrides, setting) {
                        return normalise(setting, val);
                    }
                }
            }
        }
    }

    // 1b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress" {
        if let Some(legacy) = map_get(display_cfg, "tool_progress_overrides") {
            if legacy.is_mapping() {
                if let Some(val) = map_get(legacy, platform_key) {
                    return normalise(setting, val);
                }
            }
        }
    }

    // 2. Global user setting (display.<key>). Skip display.streaming because
    // that key controls only CLI terminal streaming; gateway token streaming
    // is governed by the top-level streaming config plus per-platform
    // overrides.
    if setting != "streaming" {
        if let Some(val) = map_get(display_cfg, setting) {
            return normalise(setting, val);
        }
    }

    // 3. Built-in platform default
    if has_platform_defaults(platform_key) {
        if let Some(val) = platform_default(platform_key, setting) {
            if !val.is_none() {
                return val;
            }
        }
    }

    // 4. Built-in global default
    if let Some(val) = global_default(setting) {
        if !val.is_none() {
            return val;
        }
    }

    fallback
}

/// Mirror Python truthiness for the dict-or-empty (`... or {}`) and
/// `if platforms` style guards: an empty mapping, empty sequence, false, 0, or
/// empty string is falsy.
fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !*b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i == 0
            } else if let Some(f) = n.as_f64() {
                f == 0.0
            } else {
                false
            }
        }
        Value::String(s) => s.is_empty(),
        Value::Sequence(s) => s.is_empty(),
        Value::Mapping(m) => m.is_empty(),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stringify a YAML scalar the way Python's `str(value).lower()` would for the
/// values that reach `_normalise`. Booleans become `"true"`/`"false"`, numbers
/// their decimal form, strings themselves, null `"none"`.
fn yaml_to_string(value: &Value) -> String {
    match value {
        Value::Null => "none".to_string(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        // Fallback for sequences/mappings — unlikely for display settings.
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

/// Normalise YAML quirks (bare `off` → False in YAML 1.1).
///
/// Faithful port of Python `_normalise`. Note that YAML deserialisers vary on
/// whether bare `off`/`on`/`yes`/`no` parse as bools; this function handles
/// both the bool and string representations the same way Python's logic did.
pub fn normalise(setting: &str, value: &Value) -> DisplayValue {
    match setting {
        "tool_progress" => {
            // In Python: False -> "off", True -> "all", else str(value).lower()
            if let Value::Bool(false) = value {
                return DisplayValue::Str("off".to_string());
            }
            if let Value::Bool(true) = value {
                return DisplayValue::Str("all".to_string());
            }
            DisplayValue::Str(yaml_to_string(value).to_lowercase())
        }
        "show_reasoning" | "streaming" => {
            // If string: lower in ("true","1","yes","on"); else bool(value)
            match value {
                Value::String(s) => {
                    let lc = s.to_lowercase();
                    let b = matches!(lc.as_str(), "true" | "1" | "yes" | "on");
                    DisplayValue::Bool(b)
                }
                other => DisplayValue::Bool(!is_falsy(other)),
            }
        }
        "tool_preview_length" => {
            // try int(value) except (TypeError, ValueError): 0
            DisplayValue::Int(coerce_int(value))
        }
        _ => yaml_to_display(value),
    }
}

/// Best-effort `int(value)` mirroring Python: ints pass through, floats
/// truncate toward zero, numeric strings parse, everything else -> 0.
fn coerce_int(value: &Value) -> i64 {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                f.trunc() as i64
            } else {
                0
            }
        }
        Value::Bool(b) => {
            // Python int(True) == 1, int(False) == 0
            if *b {
                1
            } else {
                0
            }
        }
        Value::String(s) => {
            // Python int("12") works; int("12.5") raises -> 0.
            s.trim().parse::<i64>().unwrap_or(0)
        }
        _ => 0,
    }
}

/// Convert an arbitrary YAML scalar into a `DisplayValue` for the default
/// (non-normalised) branch — used only when `setting` is not one of the four
/// known keys, matching Python's `return value`.
fn yaml_to_display(value: &Value) -> DisplayValue {
    match value {
        Value::Null => DisplayValue::None,
        Value::Bool(b) => DisplayValue::Bool(*b),
        Value::String(s) => DisplayValue::Str(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                DisplayValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                DisplayValue::Int(f.trunc() as i64)
            } else {
                DisplayValue::None
            }
        }
        other => DisplayValue::Str(yaml_to_string(other)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(yaml: &str) -> Value {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn empty_config_uses_platform_default() {
        let c = cfg("{}");
        // telegram is Tier HIGH: tool_progress = "all"
        let v = resolve_display_setting(&c, "telegram", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("all".to_string()));
        // slack overrides tool_progress to "off"
        let v = resolve_display_setting(&c, "slack", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("off".to_string()));
    }

    #[test]
    fn unknown_platform_falls_to_global_default() {
        let c = cfg("{}");
        let v = resolve_display_setting(&c, "unknown_plat", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("all".to_string()));
        let v =
            resolve_display_setting(&c, "unknown_plat", "tool_preview_length", DisplayValue::None);
        assert_eq!(v, DisplayValue::Int(0));
    }

    #[test]
    fn per_platform_override_wins() {
        let c = cfg(
            "display:\n  platforms:\n    telegram:\n      tool_progress: new\n",
        );
        let v = resolve_display_setting(&c, "telegram", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("new".to_string()));
    }

    #[test]
    fn global_user_setting_wins_over_platform_default() {
        let c = cfg("display:\n  tool_progress: new\n");
        // discord default is "all"; global user setting "new" wins.
        let v = resolve_display_setting(&c, "discord", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("new".to_string()));
    }

    #[test]
    fn streaming_skips_global_user_setting() {
        // display.streaming is CLI-only and must NOT be used for gateway.
        let c = cfg("display:\n  streaming: true\n");
        // signal is Tier LOW -> streaming false default.
        let v = resolve_display_setting(&c, "signal", "streaming", DisplayValue::None);
        assert_eq!(v, DisplayValue::Bool(false));
        // telegram Tier HIGH streaming is None -> falls to global default None.
        let v = resolve_display_setting(&c, "telegram", "streaming", DisplayValue::None);
        assert_eq!(v, DisplayValue::None);
    }

    #[test]
    fn streaming_per_platform_override_is_honoured() {
        let c = cfg(
            "display:\n  platforms:\n    telegram:\n      streaming: true\n",
        );
        let v = resolve_display_setting(&c, "telegram", "streaming", DisplayValue::None);
        assert_eq!(v, DisplayValue::Bool(true));
    }

    #[test]
    fn legacy_tool_progress_overrides_fallback() {
        let c = cfg(
            "display:\n  tool_progress_overrides:\n    discord: off\n",
        );
        // YAML 1.1 bare `off` may parse as bool false -> normalise to "off".
        let v = resolve_display_setting(&c, "discord", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("off".to_string()));
    }

    #[test]
    fn normalise_tool_progress_bools() {
        assert_eq!(
            normalise("tool_progress", &Value::Bool(false)),
            DisplayValue::Str("off".to_string())
        );
        assert_eq!(
            normalise("tool_progress", &Value::Bool(true)),
            DisplayValue::Str("all".to_string())
        );
        assert_eq!(
            normalise("tool_progress", &Value::String("NEW".to_string())),
            DisplayValue::Str("new".to_string())
        );
    }

    #[test]
    fn normalise_bool_settings_from_strings() {
        for s in ["true", "1", "yes", "on", "TRUE", "On"] {
            assert_eq!(
                normalise("show_reasoning", &Value::String(s.to_string())),
                DisplayValue::Bool(true)
            );
        }
        for s in ["false", "0", "no", "off", "nope"] {
            assert_eq!(
                normalise("show_reasoning", &Value::String(s.to_string())),
                DisplayValue::Bool(false)
            );
        }
    }

    #[test]
    fn normalise_preview_length() {
        assert_eq!(
            normalise("tool_preview_length", &serde_yaml::from_str("80").unwrap()),
            DisplayValue::Int(80)
        );
        assert_eq!(
            normalise(
                "tool_preview_length",
                &Value::String("not-a-number".to_string())
            ),
            DisplayValue::Int(0)
        );
        assert_eq!(
            normalise(
                "tool_preview_length",
                &Value::String("42".to_string())
            ),
            DisplayValue::Int(42)
        );
    }

    #[test]
    fn fallback_returned_for_unknown_setting() {
        let c = cfg("{}");
        let v = resolve_display_setting(
            &c,
            "telegram",
            "nonexistent_key",
            DisplayValue::Str("fb".to_string()),
        );
        assert_eq!(v, DisplayValue::Str("fb".to_string()));
    }

    #[test]
    fn null_display_treated_as_empty() {
        let c = cfg("display: null\n");
        let v = resolve_display_setting(&c, "telegram", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("all".to_string()));
    }

    #[test]
    fn api_server_preview_length_override() {
        let c = cfg("{}");
        // api_server is Tier HIGH but tool_preview_length overridden to 0.
        let v =
            resolve_display_setting(&c, "api_server", "tool_preview_length", DisplayValue::None);
        assert_eq!(v, DisplayValue::Int(0));
        // tool_progress stays "all" from the HIGH tier.
        let v = resolve_display_setting(&c, "api_server", "tool_progress", DisplayValue::None);
        assert_eq!(v, DisplayValue::Str("all".to_string()));
    }

    #[test]
    fn overrideable_keys_set() {
        assert!(is_overrideable_key("tool_progress"));
        assert!(is_overrideable_key("streaming"));
        assert!(!is_overrideable_key("personality"));
        assert_eq!(OVERRIDEABLE_KEYS.len(), 4);
    }
}
