//! Native skin/theming engine.
//!
//! Mirrors `hermes_cli/skin_engine.py`: the built-in skin table is generated
//! into `skins_data.rs` by `scripts/gen_skins.py`, and the resolution logic
//! below (merge-over-default, user YAML skins, the `resolve_skin` payload) is a
//! faithful port. This replaces the `GATEWAY_READY_HELPER` and
//! `SKIN_PAYLOAD_HELPER` `python3 -c` helpers the TUI gateway spawned.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value, json};
use serde_yaml::Value as YamlValue;

use crate::skins_data::BUILTIN_SKINS;

/// A built-in skin definition (generated data). Field set matches what the
/// `resolve_skin` payload needs; spinner/tool_emojis are intentionally omitted.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinSkin {
    pub key: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub colors: &'static [(&'static str, &'static str)],
    pub branding: &'static [(&'static str, &'static str)],
    pub tool_prefix: &'static str,
    pub banner_logo: &'static str,
    pub banner_hero: &'static str,
}

/// A fully-resolved skin (built-in or user YAML), with colors/branding already
/// merged over the `default` skin. Port of the Python `SkinConfig`.
#[derive(Debug, Clone)]
pub struct SkinConfig {
    pub name: String,
    pub description: String,
    pub colors: BTreeMap<String, String>,
    pub branding: BTreeMap<String, String>,
    pub tool_prefix: String,
    pub banner_logo: String,
    pub banner_hero: String,
}

fn builtin(key: &str) -> Option<&'static BuiltinSkin> {
    BUILTIN_SKINS.iter().find(|skin| skin.key == key)
}

fn default_builtin() -> &'static BuiltinSkin {
    builtin("default").unwrap_or(&BUILTIN_SKINS[0])
}

fn pairs_to_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Build a `SkinConfig` from raw data (built-in or YAML), layering colors,
/// branding, and tool_prefix over the `default` skin. Port of `_build_skin_config`.
fn build_skin_config(
    name: &str,
    description: &str,
    colors: BTreeMap<String, String>,
    branding: BTreeMap<String, String>,
    tool_prefix: Option<String>,
    banner_logo: String,
    banner_hero: String,
) -> SkinConfig {
    let default = default_builtin();
    let mut merged_colors = pairs_to_map(default.colors);
    merged_colors.extend(colors);
    let mut merged_branding = pairs_to_map(default.branding);
    merged_branding.extend(branding);
    SkinConfig {
        name: name.to_string(),
        description: description.to_string(),
        colors: merged_colors,
        branding: merged_branding,
        tool_prefix: tool_prefix.unwrap_or_else(|| default.tool_prefix.to_string()),
        banner_logo,
        banner_hero,
    }
}

fn config_from_builtin(skin: &BuiltinSkin) -> SkinConfig {
    build_skin_config(
        skin.name,
        skin.description,
        pairs_to_map(skin.colors),
        pairs_to_map(skin.branding),
        Some(skin.tool_prefix.to_string()),
        skin.banner_logo.to_string(),
        skin.banner_hero.to_string(),
    )
}

/// Convert a YAML mapping value to a `BTreeMap<String,String>` (string-coerced),
/// dropping non-scalar values — mirrors how the Python skin dicts are string maps.
fn yaml_string_map(value: Option<&YamlValue>) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    if let Some(YamlValue::Mapping(mapping)) = value {
        for (k, v) in mapping {
            if let Some(k) = k.as_str() {
                if let Some(s) = yaml_scalar_string(v) {
                    map.insert(k.to_string(), s);
                }
            }
        }
    }
    map
}

fn yaml_scalar_string(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(s) => Some(s.clone()),
        YamlValue::Bool(b) => Some(b.to_string()),
        YamlValue::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn yaml_get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

/// Load a user skin YAML from `<hermes_home>/skins/<name>.yaml` if present and
/// valid (a mapping containing a `name` key). Port of `_load_skin_from_yaml`.
fn load_user_skin(hermes_home: &Path, name: &str) -> Option<SkinConfig> {
    let path = hermes_home.join("skins").join(format!("{name}.yaml"));
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let YamlValue::Mapping(data) = serde_yaml::from_str::<YamlValue>(&text).ok()? else {
        return None;
    };
    // Must contain a "name" key to be a valid skin (mirrors the Python guard).
    yaml_get(&data, "name")?;
    let skin_name = yaml_get(&data, "name")
        .and_then(YamlValue::as_str)
        .unwrap_or(name)
        .to_string();
    Some(build_skin_config(
        &skin_name,
        yaml_get(&data, "description")
            .and_then(YamlValue::as_str)
            .unwrap_or(""),
        yaml_string_map(yaml_get(&data, "colors")),
        yaml_string_map(yaml_get(&data, "branding")),
        yaml_get(&data, "tool_prefix")
            .and_then(YamlValue::as_str)
            .map(str::to_string),
        yaml_get(&data, "banner_logo")
            .and_then(YamlValue::as_str)
            .unwrap_or("")
            .to_string(),
        yaml_get(&data, "banner_hero")
            .and_then(YamlValue::as_str)
            .unwrap_or("")
            .to_string(),
    ))
}

/// Load a skin by name: user skins dir first, then built-in, then default.
/// Port of `load_skin`.
pub fn load_skin(hermes_home: &Path, name: &str) -> SkinConfig {
    if let Some(user) = load_user_skin(hermes_home, name) {
        return user;
    }
    if let Some(skin) = builtin(name) {
        return config_from_builtin(skin);
    }
    config_from_builtin(default_builtin())
}

/// Resolve the active skin name from a config mapping's `display.skin`.
/// Port of `init_skin_from_config` (defaults to "default").
fn active_skin_name(config: &serde_yaml::Mapping) -> String {
    if let Some(YamlValue::Mapping(display)) = yaml_get(config, "display") {
        if let Some(name) = yaml_get(display, "skin")
            .and_then(YamlValue::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return name.to_string();
        }
    }
    "default".to_string()
}

fn read_config_mapping(hermes_home: &Path) -> Option<serde_yaml::Mapping> {
    let text = std::fs::read_to_string(hermes_home.join("config.yaml")).ok()?;
    match serde_yaml::from_str::<YamlValue>(&text).ok()? {
        YamlValue::Mapping(mapping) => Some(mapping),
        _ => None,
    }
}

/// Build the `resolve_skin()` payload consumed by the TUI gateway. Port of
/// `tui_gateway.server.resolve_skin`: reads `display.skin` from config, resolves
/// the skin (user YAML or built-in, merged over default), and returns
/// `{name, colors, branding, banner_logo, banner_hero, tool_prefix, help_header}`.
pub fn resolve_skin(hermes_home: &Path) -> Value {
    let config = read_config_mapping(hermes_home);
    let name = config
        .as_ref()
        .map(active_skin_name)
        .unwrap_or_else(|| "default".to_string());
    let skin = load_skin(hermes_home, &name);

    let colors: Map<String, Value> = skin
        .colors
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let branding: Map<String, Value> = skin
        .branding
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let help_header = skin
        .branding
        .get("help_header")
        .cloned()
        .unwrap_or_default();

    json!({
        "name": skin.name,
        "colors": colors,
        "branding": branding,
        "banner_logo": skin.banner_logo,
        "banner_hero": skin.banner_hero,
        "tool_prefix": skin.tool_prefix,
        "help_header": help_header,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_skin_resolves_with_merged_branding() {
        let temp = TempDir::new().unwrap();
        let payload = resolve_skin(temp.path());
        assert_eq!(payload["name"], "default");
        assert_eq!(payload["colors"]["banner_title"], "#FFD700");
        assert_eq!(payload["tool_prefix"], "┊");
        assert_eq!(payload["help_header"], "(^_^)? Available Commands");
        // banner_logo empty for default
        assert_eq!(payload["banner_logo"], "");
    }

    #[test]
    fn config_selects_builtin_and_merges_over_default() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("config.yaml"),
            "display:\n  skin: ares\n",
        )
        .unwrap();
        let payload = resolve_skin(temp.path());
        assert_eq!(payload["name"], "ares");
        // ares overrides
        assert_eq!(payload["colors"]["banner_border"], "#9F1C1C");
        assert_eq!(payload["tool_prefix"], "╎");
        assert!(payload["banner_logo"].as_str().unwrap().contains("█"));
        // ui_ok is shared with default and present via merge
        assert_eq!(payload["colors"]["ui_ok"], "#4caf50");
        // ares help_header
        assert_eq!(payload["help_header"], "(⚔) Available Commands");
    }

    #[test]
    fn unknown_skin_falls_back_to_default() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("config.yaml"),
            "display:\n  skin: does-not-exist\n",
        )
        .unwrap();
        let payload = resolve_skin(temp.path());
        assert_eq!(payload["name"], "default");
    }

    #[test]
    fn user_yaml_skin_overrides_and_merges() {
        let temp = TempDir::new().unwrap();
        let skins = temp.path().join("skins");
        std::fs::create_dir_all(&skins).unwrap();
        std::fs::write(
            skins.join("custom.yaml"),
            "name: custom\ndescription: My skin\ncolors:\n  banner_title: \"#123456\"\nbranding:\n  help_header: Custom Help\n",
        )
        .unwrap();
        std::fs::write(
            temp.path().join("config.yaml"),
            "display:\n  skin: custom\n",
        )
        .unwrap();
        let payload = resolve_skin(temp.path());
        assert_eq!(payload["name"], "custom");
        // overridden color
        assert_eq!(payload["colors"]["banner_title"], "#123456");
        // inherited-from-default color still present
        assert_eq!(payload["colors"]["ui_ok"], "#4caf50");
        // overridden branding
        assert_eq!(payload["help_header"], "Custom Help");
        // tool_prefix falls back to default when absent
        assert_eq!(payload["tool_prefix"], "┊");
    }
}
