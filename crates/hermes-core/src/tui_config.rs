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
}
