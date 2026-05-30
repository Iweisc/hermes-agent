//! Native TUI gateway `config.get` resolution.
//!
//! Port of the self-contained branches of `tui_gateway.server`'s `config.get`
//! handler. The Python helper ran in a fresh `python3 -c` process with an empty
//! `_sessions` dict, so any session-coupled branch (e.g. `fast` inspecting a
//! live agent) was dead code and always fell through to the config-file value —
//! this native port reproduces that effective behavior from `config.yaml` alone.
//!
//! Returns the JSON-RPC `result` object on success, or `(code, message)` for the
//! caller to turn into an error response.

use std::path::Path;

use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;

use crate::commands::resolve_tui_model;

const STATUSBAR_MODES: &[&str] = &["off", "top", "bottom"];
const INDICATOR_STYLES: &[&str] = &["ascii", "emoji", "kaomoji", "unicode"];
const INDICATOR_DEFAULT: &str = "kaomoji";
const DETAILS_MODES: &[&str] = &["hidden", "collapsed", "expanded"];
const THINKING_MODES: &[&str] = &["collapsed", "truncated", "full"];
const BUSY_MODES: &[&str] = &["queue", "steer", "interrupt"];

fn read_config(hermes_home: &Path) -> serde_yaml::Mapping {
    std::fs::read_to_string(hermes_home.join("config.yaml"))
        .ok()
        .and_then(|text| serde_yaml::from_str::<YamlValue>(&text).ok())
        .and_then(|value| match value {
            YamlValue::Mapping(mapping) => Some(mapping),
            _ => None,
        })
        .unwrap_or_default()
}

fn get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn section<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a serde_yaml::Mapping> {
    match get(mapping, key) {
        Some(YamlValue::Mapping(inner)) => Some(inner),
        _ => None,
    }
}

fn str_field(mapping: Option<&serde_yaml::Mapping>, key: &str) -> Option<String> {
    mapping
        .and_then(|m| get(m, key))
        .and_then(YamlValue::as_str)
        .map(str::to_string)
}

fn bool_field(mapping: Option<&serde_yaml::Mapping>, key: &str, default: bool) -> bool {
    match mapping.and_then(|m| get(m, key)) {
        Some(YamlValue::Bool(b)) => *b,
        _ => default,
    }
}

/// Port of `_coerce_statusbar`: `False`->"off", a known string mode, else "top".
fn coerce_statusbar(raw: Option<&YamlValue>) -> &'static str {
    match raw {
        Some(YamlValue::Bool(false)) => "off",
        Some(YamlValue::String(s)) => {
            let s = s.trim().to_lowercase();
            STATUSBAR_MODES
                .iter()
                .copied()
                .find(|mode| *mode == s)
                .unwrap_or("top")
        }
        _ => "top",
    }
}

/// Port of `_display_mouse_tracking`: canonical `mouse_tracking` with legacy
/// `tui_mouse` fallback; truthy unless explicitly disabled.
fn display_mouse_tracking(display: Option<&serde_yaml::Mapping>) -> bool {
    let Some(display) = display else {
        return true;
    };
    let raw = if get(display, "mouse_tracking").is_some() {
        get(display, "mouse_tracking")
    } else {
        get(display, "tui_mouse")
    };
    match raw {
        Some(YamlValue::Bool(b)) => *b,
        Some(YamlValue::Number(n)) => n.as_i64() != Some(0),
        Some(YamlValue::String(s)) => {
            !matches!(s.trim().to_lowercase().as_str(), "0" | "false" | "no" | "off")
        }
        None => true,
        _ => true,
    }
}

/// Port of `_load_service_tier`: map config `agent.service_tier` to
/// `Some("priority")` or `None`.
fn service_tier_is_priority(config: &serde_yaml::Mapping) -> bool {
    let raw = str_field(section(config, "agent"), "service_tier")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    matches!(raw.as_str(), "fast" | "priority" | "on")
}

/// Resolve a `config.get` key against `config.yaml`, returning the JSON-RPC
/// `result` object.
///
/// Returns `Ok(None)` for keys this native path deliberately does NOT handle
/// (currently only `provider`, which depends on credential-aware provider
/// enumeration in `hermes_cli.models.list_available_providers`); the caller
/// should fall back to the Python helper for those. Returns
/// `Err((code, message))` for genuinely unknown keys (matching the Python
/// `4002` error).
///
/// `home_display` is the display string for the hermes home dir (the caller
/// supplies it to match `hermes_constants.display_hermes_home`).
pub fn config_get(
    hermes_home: &Path,
    key: &str,
    home_display: &str,
) -> Result<Option<Value>, (i64, String)> {
    let config = read_config(hermes_home);
    let display = section(&config, "display");

    let value = match key {
        // `provider` requires credential-aware provider enumeration; leave it
        // to the Python helper to avoid diverging from list_available_providers.
        "provider" => return Ok(None),
        "profile" => json!({
            "home": hermes_home.display().to_string(),
            "display": home_display,
        }),
        "full" => json!({ "config": yaml_mapping_to_json(&config) }),
        "prompt" => json!({
            "prompt": get(&config, "custom_prompt")
                .and_then(YamlValue::as_str)
                .unwrap_or_default(),
        }),
        "skin" => json!({
            "value": str_field(display, "skin").unwrap_or_else(|| "default".to_string()),
        }),
        "indicator" => {
            let raw = str_field(display, "tui_status_indicator").unwrap_or_default();
            let norm = raw.trim().to_lowercase();
            let value = if INDICATOR_STYLES.contains(&norm.as_str()) {
                norm
            } else {
                INDICATOR_DEFAULT.to_string()
            };
            json!({ "value": value })
        }
        "personality" => json!({
            "value": str_field(display, "personality").unwrap_or_else(|| "default".to_string()),
        }),
        "reasoning" => {
            let effort = str_field(section(&config, "agent"), "reasoning_effort")
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "medium".to_string());
            let display_mode = if bool_field(display, "show_reasoning", false) {
                "show"
            } else {
                "hide"
            };
            json!({ "value": effort, "display": display_mode })
        }
        "fast" => {
            // Session-coupled branch is dead in the fresh-process model; always
            // resolve from config service tier.
            let value = if service_tier_is_priority(&config) {
                "fast"
            } else {
                "normal"
            };
            json!({ "value": value })
        }
        "busy" => {
            let raw = str_field(display, "busy_input_mode")
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            let value = if BUSY_MODES.contains(&raw.as_str()) {
                raw
            } else {
                "interrupt".to_string()
            };
            json!({ "value": value })
        }
        "details_mode" => {
            let raw = str_field(display, "details_mode")
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "collapsed".to_string())
                .trim()
                .to_lowercase();
            let value = if DETAILS_MODES.contains(&raw.as_str()) {
                raw
            } else {
                "collapsed".to_string()
            };
            json!({ "value": value })
        }
        "thinking_mode" => {
            let raw = str_field(display, "thinking_mode")
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            let value = if THINKING_MODES.contains(&raw.as_str()) {
                raw
            } else {
                let details = str_field(display, "details_mode")
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "collapsed".to_string())
                    .trim()
                    .to_lowercase();
                if details == "expanded" {
                    "full".to_string()
                } else {
                    "collapsed".to_string()
                }
            };
            json!({ "value": value })
        }
        "compact" => {
            let on = bool_field(display, "tui_compact", false);
            json!({ "value": if on { "on" } else { "off" } })
        }
        "statusbar" => {
            let raw = display.and_then(|d| get(d, "tui_statusbar"));
            json!({ "value": coerce_statusbar(raw) })
        }
        "mouse" => {
            let on = display_mouse_tracking(display);
            json!({ "value": if on { "on" } else { "off" } })
        }
        "mtime" => {
            let path = hermes_home.join("config.yaml");
            let mtime = std::fs::metadata(&path)
                .ok()
                .and_then(|meta| meta.modified().ok())
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|dur| dur.as_secs_f64())
                .unwrap_or(0.0);
            json!({ "mtime": mtime })
        }
        other => {
            return Err((4002, format!("unknown config key: {other}")));
        }
    };

    Ok(Some(value))
}

fn yaml_mapping_to_json(mapping: &serde_yaml::Mapping) -> Value {
    yaml_to_json(&YamlValue::Mapping(mapping.clone()))
}

fn yaml_to_json(value: &YamlValue) -> Value {
    match value {
        YamlValue::Null => Value::Null,
        YamlValue::Bool(b) => Value::Bool(*b),
        YamlValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                json!(i)
            } else if let Some(u) = n.as_u64() {
                json!(u)
            } else if let Some(f) = n.as_f64() {
                json!(f)
            } else {
                Value::Null
            }
        }
        YamlValue::String(s) => Value::String(s.clone()),
        YamlValue::Sequence(seq) => Value::Array(seq.iter().map(yaml_to_json).collect()),
        YamlValue::Mapping(mapping) => {
            let mut object = serde_json::Map::new();
            for (k, v) in mapping {
                if let Some(k) = k.as_str() {
                    object.insert(k.to_string(), yaml_to_json(v));
                }
            }
            Value::Object(object)
        }
        YamlValue::Tagged(tagged) => yaml_to_json(&tagged.value),
    }
}

/// Port of `_cfg_max_turns`: `HERMES_TUI_MAX_TURNS` env (>0), else
/// `agent.max_turns`, else top-level `max_turns`, else `default`.
fn cfg_max_turns(config: &serde_yaml::Mapping, default: i64) -> i64 {
    if let Ok(env) = std::env::var("HERMES_TUI_MAX_TURNS") {
        if let Ok(value) = env.trim().parse::<i64>() {
            if value > 0 {
                return value;
            }
        }
    }
    if let Some(value) = section(config, "agent")
        .and_then(|agent| get(agent, "max_turns"))
        .and_then(YamlValue::as_i64)
        .filter(|v| *v != 0)
    {
        return value;
    }
    get(config, "max_turns")
        .and_then(YamlValue::as_i64)
        .filter(|v| *v != 0)
        .unwrap_or(default)
}

/// Build the `config.show` payload (port of the Python handler): a `sections`
/// array of Model / Agent / Environment rows. `cwd` and the absolute config
/// path are supplied by the caller (which has the live working dir).
pub fn config_show(hermes_home: &Path, cwd: &str) -> Value {
    let config = read_config(hermes_home);

    let model = resolve_tui_model(hermes_home);
    let api_key = std::env::var("HERMES_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| get(&config, "api_key").and_then(YamlValue::as_str).map(str::to_string))
        .unwrap_or_default();
    let masked = if api_key.chars().count() > 4 {
        let tail: String = api_key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        format!("****{tail}")
    } else {
        "(not set)".to_string()
    };
    let base_url = std::env::var("HERMES_BASE_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| get(&config, "base_url").and_then(YamlValue::as_str).map(str::to_string))
        .unwrap_or_default();

    let max_turns = cfg_max_turns(&config, 90);
    let toolsets = match get(&config, "enabled_toolsets") {
        Some(YamlValue::Sequence(seq)) => {
            let names: Vec<String> = seq.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
            if names.is_empty() { "all".to_string() } else { names.join(", ") }
        }
        _ => "all".to_string(),
    };
    let verbose = match get(&config, "verbose") {
        Some(YamlValue::Bool(b)) => *b,
        _ => false,
    };
    // Python renders Python bool repr: True/False.
    let verbose_str = if verbose { "True" } else { "False" };

    let config_file = hermes_home.join("config.yaml").display().to_string();

    json!({
        "sections": [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", if base_url.is_empty() { "(default)".to_string() } else { base_url }],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", max_turns.to_string()],
                    ["Toolsets", toolsets],
                    ["Verbose", verbose_str],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", cwd],
                    ["Config File", config_file],
                ],
            },
        ]
    })
}

// ===========================================================================
// config.set
// ===========================================================================

/// Persist the full config mapping to `<hermes_home>/config.yaml` (port of
/// `_save_cfg`: a full `yaml.safe_dump` rewrite).
fn save_config(hermes_home: &Path, config: &serde_yaml::Mapping) -> Result<(), (i64, String)> {
    let text = serde_yaml::to_string(&YamlValue::Mapping(config.clone()))
        .map_err(|e| (5001, format!("serializing config failed: {e}")))?;
    std::fs::write(hermes_home.join("config.yaml"), text)
        .map_err(|e| (5001, format!("writing config.yaml failed: {e}")))
}

/// Set a dotted `key_path` to `value`, creating intermediate mappings, then
/// persist. Port of `_write_config_key` + `_save_cfg`.
fn write_config_key(
    hermes_home: &Path,
    key_path: &str,
    value: YamlValue,
) -> Result<(), (i64, String)> {
    let mut config = read_config(hermes_home);
    let keys: Vec<&str> = key_path.split('.').collect();
    set_nested(&mut config, &keys, value);
    save_config(hermes_home, &config)
}

fn set_nested(mapping: &mut serde_yaml::Mapping, keys: &[&str], value: YamlValue) {
    let head = keys[0];
    let head_key = YamlValue::String(head.to_string());
    if keys.len() == 1 {
        mapping.insert(head_key, value);
        return;
    }
    // Ensure intermediate mapping exists.
    let needs_replace = !matches!(mapping.get(&head_key), Some(YamlValue::Mapping(_)));
    if needs_replace {
        mapping.insert(head_key.clone(), YamlValue::Mapping(serde_yaml::Mapping::new()));
    }
    if let Some(YamlValue::Mapping(inner)) = mapping.get_mut(&head_key) {
        set_nested(inner, &keys[1..], value);
    }
}

/// Result of a native `config.set`: either a ready JSON-RPC `result` object, an
/// error to surface, or `NotHandled` to fall back to the Python helper.
pub enum ConfigSetOutcome {
    Ok(Value),
    Err(i64, String),
    NotHandled,
}

/// Apply a `config.set` for the self-contained, pure config-write keys, writing
/// `config.yaml` directly. Port of the corresponding branches of
/// `tui_gateway.server`'s `config.set`.
///
/// Keys that mutate live agent/session state (`model`, `fast`, `verbose`,
/// `yolo`, `reasoning`, `personality`) — or that need provider/fast-mode/yolo
/// machinery — return [`ConfigSetOutcome::NotHandled`] so the caller falls back
/// to the Python helper. Note the caller already routes session-bound instances
/// of those keys to the live child worker before reaching here.
pub fn config_set(hermes_home: &Path, key: &str, value: &Value) -> ConfigSetOutcome {
    // String form of the value, mirroring Python's `str(value or "").strip().lower()`.
    let raw = value_to_string(value).trim().to_lowercase();

    match key {
        "compact" => {
            let config = read_config(hermes_home);
            let cur = bool_field(section(&config, "display"), "tui_compact", false);
            let nv = match raw.as_str() {
                "" | "toggle" => !cur,
                "on" => true,
                "off" => false,
                _ => return ConfigSetOutcome::Err(4002, format!("unknown compact value: {}", value_to_string(value))),
            };
            try_set(hermes_home, "display.tui_compact", YamlValue::Bool(nv), key, if nv { "on" } else { "off" })
        }
        "statusbar" => {
            let config = read_config(hermes_home);
            let current = coerce_statusbar(section(&config, "display").and_then(|d| get(d, "tui_statusbar")));
            let nv = match raw.as_str() {
                "" | "toggle" => if current == "off" { "top" } else { "off" },
                "on" => "top",
                other if STATUSBAR_MODES.contains(&other) => other,
                _ => return ConfigSetOutcome::Err(4002, format!("unknown statusbar value: {}", value_to_string(value))),
            };
            try_set(hermes_home, "display.tui_statusbar", YamlValue::String(nv.to_string()), key, nv)
        }
        "mouse" => {
            let config = read_config(hermes_home);
            let current = display_mouse_tracking(section(&config, "display"));
            let nv = match raw.as_str() {
                "" | "toggle" => !current,
                "on" => true,
                "off" => false,
                _ => return ConfigSetOutcome::Err(4002, format!("unknown mouse value: {}", value_to_string(value))),
            };
            try_set(hermes_home, "display.mouse_tracking", YamlValue::Bool(nv), key, if nv { "on" } else { "off" })
        }
        "indicator" => {
            // Explicit None check (mirrors Python) so falsy non-strings surface as themselves.
            let raw_disp = if value.is_null() { String::new() } else { value_to_string(value) };
            let normalized = raw_disp.trim().to_lowercase();
            if !INDICATOR_STYLES.contains(&normalized.as_str()) {
                return ConfigSetOutcome::Err(
                    4002,
                    format!("unknown indicator: {normalized:?}; pick one of {}", INDICATOR_STYLES.join("|")),
                );
            }
            try_set(hermes_home, "display.tui_status_indicator", YamlValue::String(normalized.clone()), key, &normalized)
        }
        "thinking_mode" => {
            if !THINKING_MODES.contains(&raw.as_str()) {
                return ConfigSetOutcome::Err(4002, format!("unknown thinking_mode: {}", value_to_string(value)));
            }
            let mut config = read_config(hermes_home);
            set_nested(&mut config, &["display", "thinking_mode"], YamlValue::String(raw.clone()));
            let details = if raw == "full" { "expanded" } else { "collapsed" };
            set_nested(&mut config, &["display", "details_mode"], YamlValue::String(details.to_string()));
            match save_config(hermes_home, &config) {
                Ok(()) => ConfigSetOutcome::Ok(json!({ "key": key, "value": raw })),
                Err((c, m)) => ConfigSetOutcome::Err(c, m),
            }
        }
        "details_mode" => {
            if !DETAILS_MODES.contains(&raw.as_str()) {
                return ConfigSetOutcome::Err(4002, format!("unknown details_mode: {}", value_to_string(value)));
            }
            let mut config = read_config(hermes_home);
            set_nested(&mut config, &["display", "details_mode"], YamlValue::String(raw.clone()));
            for sect in DETAIL_SECTION_NAMES {
                set_nested(&mut config, &["display", "sections", sect], YamlValue::String(raw.clone()));
            }
            match save_config(hermes_home, &config) {
                Ok(()) => ConfigSetOutcome::Ok(json!({ "key": key, "value": raw })),
                Err((c, m)) => ConfigSetOutcome::Err(c, m),
            }
        }
        _ if key.starts_with("details_mode.") => {
            let sect = &key["details_mode.".len()..];
            if !DETAIL_SECTION_NAMES.contains(&sect) {
                return ConfigSetOutcome::Err(4002, format!("unknown section: {sect}"));
            }
            let mut config = read_config(hermes_home);
            if raw.is_empty() {
                // Clear the explicit override.
                if let Some(YamlValue::Mapping(display)) =
                    config.get_mut(YamlValue::String("display".to_string()))
                {
                    if let Some(YamlValue::Mapping(sections)) =
                        display.get_mut(YamlValue::String("sections".to_string()))
                    {
                        sections.remove(YamlValue::String(sect.to_string()));
                    }
                }
                return match save_config(hermes_home, &config) {
                    Ok(()) => ConfigSetOutcome::Ok(json!({ "key": key, "value": "" })),
                    Err((c, m)) => ConfigSetOutcome::Err(c, m),
                };
            }
            if !DETAILS_MODES.contains(&raw.as_str()) {
                return ConfigSetOutcome::Err(4002, format!("unknown details_mode: {}", value_to_string(value)));
            }
            set_nested(&mut config, &["display", "sections", sect], YamlValue::String(raw.clone()));
            match save_config(hermes_home, &config) {
                Ok(()) => ConfigSetOutcome::Ok(json!({ "key": key, "value": raw })),
                Err((c, m)) => ConfigSetOutcome::Err(c, m),
            }
        }
        "busy" => {
            if raw.is_empty() || raw == "status" {
                let config = read_config(hermes_home);
                let current = str_field(section(&config, "display"), "busy_input_mode")
                    .unwrap_or_default()
                    .trim()
                    .to_lowercase();
                let value = if BUSY_MODES.contains(&current.as_str()) { current } else { "interrupt".to_string() };
                return ConfigSetOutcome::Ok(json!({ "key": key, "value": value }));
            }
            if !BUSY_MODES.contains(&raw.as_str()) {
                return ConfigSetOutcome::Err(4002, format!("unknown busy mode: {}", value_to_string(value)));
            }
            try_set(hermes_home, "display.busy_input_mode", YamlValue::String(raw.clone()), key, &raw)
        }
        "prompt" => {
            let mut config = read_config(hermes_home);
            let nv = if value_to_string(value) == "clear" {
                config.remove(YamlValue::String("custom_prompt".to_string()));
                String::new()
            } else {
                let v = value_to_string(value);
                config.insert(YamlValue::String("custom_prompt".to_string()), value_to_yaml(value));
                v
            };
            match save_config(hermes_home, &config) {
                Ok(()) => ConfigSetOutcome::Ok(json!({ "key": key, "value": nv })),
                Err((c, m)) => ConfigSetOutcome::Err(c, m),
            }
        }
        // model/fast/verbose/yolo/reasoning/personality/skin and any other key:
        // defer to the Python helper (session-coupled or needs extra machinery).
        _ => ConfigSetOutcome::NotHandled,
    }
}

const DETAIL_SECTION_NAMES: &[&str] = &["thinking", "tools", "subagents", "activity"];

fn try_set(
    hermes_home: &Path,
    key_path: &str,
    value: YamlValue,
    resp_key: &str,
    resp_value: &str,
) -> ConfigSetOutcome {
    match write_config_key(hermes_home, key_path, value) {
        Ok(()) => ConfigSetOutcome::Ok(json!({ "key": resp_key, "value": resp_value })),
        Err((code, message)) => ConfigSetOutcome::Err(code, message),
    }
}

/// Coerce a JSON value to a string the way Python's `str(value or "")` would for
/// the scalar cases used by config.set (strings as-is; null/false/empty -> "").
fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn value_to_yaml(value: &Value) -> YamlValue {
    match value {
        Value::Null => YamlValue::Null,
        Value::Bool(b) => YamlValue::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                YamlValue::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                YamlValue::Number(u.into())
            } else if let Some(f) = n.as_f64() {
                YamlValue::Number(f.into())
            } else {
                YamlValue::Null
            }
        }
        Value::String(s) => YamlValue::String(s.clone()),
        Value::Array(arr) => YamlValue::Sequence(arr.iter().map(value_to_yaml).collect()),
        Value::Object(obj) => {
            let mut mapping = serde_yaml::Mapping::new();
            for (k, v) in obj {
                mapping.insert(YamlValue::String(k.clone()), value_to_yaml(v));
            }
            YamlValue::Mapping(mapping)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn home_with(config: &str) -> TempDir {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("config.yaml"), config).unwrap();
        temp
    }

    #[test]
    fn unknown_key_errors() {
        let temp = TempDir::new().unwrap();
        let err = config_get(temp.path(), "nope", "~/.hermes").unwrap_err();
        assert_eq!(err.0, 4002);
    }

    #[test]
    fn skin_and_indicator_defaults_and_normalization() {
        let temp = home_with("display:\n  tui_status_indicator: WEIRD\n");
        assert_eq!(
            config_get(temp.path(), "skin", "~").unwrap().unwrap()["value"],
            "default"
        );
        // unknown indicator normalizes to kaomoji
        assert_eq!(
            config_get(temp.path(), "indicator", "~").unwrap().unwrap()["value"],
            "kaomoji"
        );
        let temp2 = home_with("display:\n  tui_status_indicator: Emoji\n");
        assert_eq!(
            config_get(temp2.path(), "indicator", "~").unwrap().unwrap()["value"],
            "emoji"
        );
    }

    #[test]
    fn statusbar_and_mouse_and_compact() {
        let temp = home_with("display:\n  tui_statusbar: bottom\n  tui_compact: true\n  mouse_tracking: false\n");
        assert_eq!(
            config_get(temp.path(), "statusbar", "~").unwrap().unwrap()["value"],
            "bottom"
        );
        assert_eq!(
            config_get(temp.path(), "compact", "~").unwrap().unwrap()["value"],
            "on"
        );
        assert_eq!(
            config_get(temp.path(), "mouse", "~").unwrap().unwrap()["value"],
            "off"
        );
    }

    #[test]
    fn reasoning_thinking_and_fast() {
        let temp = home_with(
            "agent:\n  reasoning_effort: high\n  service_tier: priority\ndisplay:\n  show_reasoning: true\n  details_mode: expanded\n",
        );
        let reasoning = config_get(temp.path(), "reasoning", "~").unwrap().unwrap();
        assert_eq!(reasoning["value"], "high");
        assert_eq!(reasoning["display"], "show");
        // thinking_mode falls back from details_mode=expanded -> full
        assert_eq!(
            config_get(temp.path(), "thinking_mode", "~").unwrap().unwrap()["value"],
            "full"
        );
        assert_eq!(
            config_get(temp.path(), "fast", "~").unwrap().unwrap()["value"],
            "fast"
        );
    }

    #[test]
    fn busy_defaults_to_interrupt() {
        let temp = TempDir::new().unwrap();
        assert_eq!(
            config_get(temp.path(), "busy", "~").unwrap().unwrap()["value"],
            "interrupt"
        );
    }

    #[test]
    fn profile_reports_home() {
        let temp = TempDir::new().unwrap();
        let result = config_get(temp.path(), "profile", "~/.hermes").unwrap().unwrap();
        assert_eq!(result["display"], "~/.hermes");
        assert_eq!(result["home"], temp.path().display().to_string());
    }

    fn set_ok(outcome: ConfigSetOutcome) -> Value {
        match outcome {
            ConfigSetOutcome::Ok(v) => v,
            ConfigSetOutcome::Err(c, m) => panic!("expected Ok, got Err({c}, {m})"),
            ConfigSetOutcome::NotHandled => panic!("expected Ok, got NotHandled"),
        }
    }

    #[test]
    fn set_compact_toggle_and_persist() {
        let temp = TempDir::new().unwrap();
        // toggle from default false -> on
        let r = set_ok(config_set(temp.path(), "compact", &json!("toggle")));
        assert_eq!(r["value"], "on");
        // round-trips through config.get
        assert_eq!(
            config_get(temp.path(), "compact", "~").unwrap().unwrap()["value"],
            "on"
        );
        // explicit off
        let r = set_ok(config_set(temp.path(), "compact", &json!("off")));
        assert_eq!(r["value"], "off");
    }

    #[test]
    fn set_statusbar_indicator_mouse() {
        let temp = TempDir::new().unwrap();
        assert_eq!(set_ok(config_set(temp.path(), "statusbar", &json!("bottom")))["value"], "bottom");
        assert_eq!(set_ok(config_set(temp.path(), "indicator", &json!("Emoji")))["value"], "emoji");
        assert_eq!(set_ok(config_set(temp.path(), "mouse", &json!("off")))["value"], "off");
        // verify persisted + readable
        assert_eq!(config_get(temp.path(), "statusbar", "~").unwrap().unwrap()["value"], "bottom");
        assert_eq!(config_get(temp.path(), "indicator", "~").unwrap().unwrap()["value"], "emoji");
        assert_eq!(config_get(temp.path(), "mouse", "~").unwrap().unwrap()["value"], "off");
    }

    #[test]
    fn set_thinking_mode_bridges_details() {
        let temp = TempDir::new().unwrap();
        assert_eq!(set_ok(config_set(temp.path(), "thinking_mode", &json!("full")))["value"], "full");
        // details_mode bridged to expanded
        assert_eq!(config_get(temp.path(), "details_mode", "~").unwrap().unwrap()["value"], "expanded");
    }

    #[test]
    fn set_details_mode_section_override_and_clear() {
        let temp = TempDir::new().unwrap();
        // per-section override
        let r = set_ok(config_set(temp.path(), "details_mode.tools", &json!("expanded")));
        assert_eq!(r["value"], "expanded");
        // clearing with empty value
        let r = set_ok(config_set(temp.path(), "details_mode.tools", &json!("")));
        assert_eq!(r["value"], "");
        // unknown section errors
        match config_set(temp.path(), "details_mode.bogus", &json!("expanded")) {
            ConfigSetOutcome::Err(code, _) => assert_eq!(code, 4002),
            _ => panic!("expected error for unknown section"),
        }
    }

    #[test]
    fn set_prompt_set_and_clear() {
        let temp = TempDir::new().unwrap();
        assert_eq!(set_ok(config_set(temp.path(), "prompt", &json!("be terse")))["value"], "be terse");
        assert_eq!(config_get(temp.path(), "prompt", "~").unwrap().unwrap()["prompt"], "be terse");
        // clear
        let r = set_ok(config_set(temp.path(), "prompt", &json!("clear")));
        assert_eq!(r["value"], "");
        assert_eq!(config_get(temp.path(), "prompt", "~").unwrap().unwrap()["prompt"], "");
    }

    #[test]
    fn set_defers_session_coupled_keys() {
        let temp = TempDir::new().unwrap();
        for key in ["model", "fast", "verbose", "yolo", "reasoning", "personality", "skin"] {
            assert!(
                matches!(config_set(temp.path(), key, &json!("x")), ConfigSetOutcome::NotHandled),
                "{key} should defer to python helper"
            );
        }
    }

    #[test]
    fn config_show_sections_and_masking() {
        let temp = home_with(
            "api_key: sk-abcdEFGH\nenabled_toolsets:\n  - core\n  - web\nverbose: true\nagent:\n  max_turns: 42\n",
        );
        // Avoid env interference from a real HERMES_API_KEY/model in the test env.
        if std::env::var_os("HERMES_API_KEY").is_some()
            || std::env::var_os("HERMES_MODEL").is_some()
            || std::env::var_os("HERMES_TUI_MAX_TURNS").is_some()
        {
            return;
        }
        let show = config_show(temp.path(), "/tmp/work");
        let sections = show["sections"].as_array().unwrap();
        let model_rows = sections[0]["rows"].as_array().unwrap();
        // API Key masked to ****last4
        assert_eq!(model_rows[2][0], "API Key");
        assert_eq!(model_rows[2][1], "****EFGH");
        let agent_rows = sections[1]["rows"].as_array().unwrap();
        assert_eq!(agent_rows[0][1], "42");
        assert_eq!(agent_rows[1][1], "core, web");
        assert_eq!(agent_rows[2][1], "True");
        let env_rows = sections[2]["rows"].as_array().unwrap();
        assert_eq!(env_rows[0][1], "/tmp/work");
        assert!(env_rows[1][1].as_str().unwrap().ends_with("config.yaml"));
    }

    #[test]
    fn config_show_defaults() {
        let temp = TempDir::new().unwrap();
        if std::env::var_os("HERMES_TUI_MAX_TURNS").is_some() {
            return;
        }
        let show = config_show(temp.path(), "/w");
        let sections = show["sections"].as_array().unwrap();
        // default max turns 90, toolsets "all", verbose False, API key (not set)
        assert_eq!(sections[1]["rows"][0][1], "90");
        assert_eq!(sections[1]["rows"][1][1], "all");
        assert_eq!(sections[1]["rows"][2][1], "False");
        assert_eq!(sections[0]["rows"][2][1], "(not set)");
        assert_eq!(sections[0]["rows"][1][1], "(default)");
    }

    #[test]
    fn set_preserves_unrelated_config() {
        let temp = home_with("agent:\n  service_tier: priority\ndisplay:\n  skin: ares\n");
        set_ok(config_set(temp.path(), "compact", &json!("on")));
        // unrelated keys survive the round-trip
        assert_eq!(config_get(temp.path(), "fast", "~").unwrap().unwrap()["value"], "fast");
        assert_eq!(config_get(temp.path(), "skin", "~").unwrap().unwrap()["value"], "ares");
    }
}
