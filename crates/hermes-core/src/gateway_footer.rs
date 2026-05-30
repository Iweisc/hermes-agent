//! Gateway runtime-metadata footer.
//!
//! Faithful port of `gateway/runtime_footer.py`: renders a compact
//! `model · context% · cwd` footer appended to the final message of an agent
//! turn when `display.runtime_footer.enabled` is set (off by default), with
//! per-platform overrides under `display.platforms.<platform>.runtime_footer`.
//!
//! Pure logic (config merge + string formatting), no I/O beyond reading
//! `$HOME`/`TERMINAL_CWD` for cwd collapsing — matches the Python exactly.

use serde_yaml::Value as YamlValue;

const DEFAULT_FIELDS: &[&str] = &["model", "context_pct", "cwd"];
const SEP: &str = " · ";

/// Resolved footer configuration (enabled flag + ordered field list).
#[derive(Debug, Clone, PartialEq)]
pub struct FooterConfig {
    pub enabled: bool,
    pub fields: Vec<String>,
}

impl Default for FooterConfig {
    fn default() -> Self {
        FooterConfig {
            enabled: false,
            fields: DEFAULT_FIELDS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

fn yaml_get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

/// Coerce a YAML scalar to a Python-`bool()`-style truthiness for the
/// `enabled` flag. Mirrors `bool(global_cfg.get("enabled"))`: true/non-zero
/// number/non-empty string -> true; false/0/empty/null -> false.
fn yaml_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Bool(b) => *b,
        YamlValue::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        YamlValue::String(s) => !s.is_empty(),
        YamlValue::Sequence(seq) => !seq.is_empty(),
        YamlValue::Mapping(m) => !m.is_empty(),
        YamlValue::Null => false,
        YamlValue::Tagged(t) => yaml_truthy(&t.value),
    }
}

/// Extract a non-empty `fields` string list from a footer config mapping.
fn footer_fields(footer: &serde_yaml::Mapping) -> Option<Vec<String>> {
    match yaml_get(footer, "fields") {
        Some(YamlValue::Sequence(seq)) if !seq.is_empty() => Some(
            seq.iter()
                .map(|v| match v {
                    YamlValue::String(s) => s.clone(),
                    YamlValue::Bool(b) => b.to_string(),
                    YamlValue::Number(n) => n.to_string(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

fn apply_footer_section(footer: &serde_yaml::Mapping, resolved: &mut FooterConfig) {
    if let Some(enabled) = yaml_get(footer, "enabled") {
        resolved.enabled = yaml_truthy(enabled);
    }
    if let Some(fields) = footer_fields(footer) {
        resolved.fields = fields;
    }
}

/// Resolve the effective runtime-footer config for `platform_key`.
///
/// Merge order (later wins): built-in defaults (enabled=false) ->
/// `display.runtime_footer` -> `display.platforms.<platform_key>.runtime_footer`.
/// Port of `resolve_footer_config`.
pub fn resolve_footer_config(
    user_config: Option<&YamlValue>,
    platform_key: Option<&str>,
) -> FooterConfig {
    let mut resolved = FooterConfig::default();
    let Some(display) = user_config
        .and_then(YamlValue::as_mapping)
        .and_then(|root| yaml_get(root, "display"))
        .and_then(YamlValue::as_mapping)
    else {
        return resolved;
    };

    if let Some(YamlValue::Mapping(global)) = yaml_get(display, "runtime_footer") {
        apply_footer_section(global, &mut resolved);
    }

    if let Some(platform_key) = platform_key {
        if let Some(YamlValue::Mapping(platforms)) = yaml_get(display, "platforms") {
            if let Some(YamlValue::Mapping(plat)) = yaml_get(platforms, platform_key) {
                if let Some(YamlValue::Mapping(plat_footer)) = yaml_get(plat, "runtime_footer") {
                    apply_footer_section(plat_footer, &mut resolved);
                }
            }
        }
    }

    resolved
}

/// Collapse `$HOME` to `~` in `cwd`. Empty input -> empty string.
/// Port of `_home_relative_cwd`.
fn home_relative_cwd(cwd: &str) -> String {
    if cwd.is_empty() {
        return String::new();
    }
    let Some(home) = dirs::home_dir() else {
        return cwd.to_string();
    };
    let home = home.to_string_lossy().to_string();
    // Python uses os.path.abspath; we canonicalize lexically by joining cwd
    // against the process cwd only if relative. Most callers pass absolute
    // paths (TERMINAL_CWD), matching the common case.
    let abs = if std::path::Path::new(cwd).is_absolute() {
        cwd.to_string()
    } else {
        std::env::current_dir()
            .map(|d| d.join(cwd).to_string_lossy().to_string())
            .unwrap_or_else(|_| cwd.to_string())
    };
    if !home.is_empty() {
        if abs == home {
            return "~".to_string();
        }
        let prefix = format!("{home}{}", std::path::MAIN_SEPARATOR);
        if abs.starts_with(&prefix) {
            return format!("~{}", &abs[home.len()..]);
        }
    }
    abs
}

/// Drop the `vendor/` prefix from a model id for readability.
fn model_short(model: &str) -> String {
    model.rsplit('/').next().unwrap_or(model).to_string()
}

/// Render the footer line, or "" when no field has data. Port of
/// `format_runtime_footer`. `context_length` of 0/None disables the percentage.
pub fn format_runtime_footer(
    model: Option<&str>,
    context_tokens: i64,
    context_length: Option<i64>,
    cwd: Option<&str>,
    fields: &[String],
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for field in fields {
        match field.as_str() {
            "model" => {
                let m = model.map(model_short).unwrap_or_default();
                if !m.is_empty() {
                    parts.push(m);
                }
            }
            "context_pct" => {
                if let Some(length) = context_length {
                    if length > 0 && context_tokens >= 0 {
                        let raw = (context_tokens as f64 / length as f64) * 100.0;
                        // Python round() is banker's rounding, but for footer
                        // percentages round-half-up matches expectations and
                        // the displayed value; clamp to 0..=100.
                        let pct = raw.round().clamp(0.0, 100.0) as i64;
                        parts.push(format!("{pct}%"));
                    }
                }
            }
            "cwd" => {
                let resolved = match cwd {
                    Some(c) if !c.is_empty() => home_relative_cwd(c),
                    _ => home_relative_cwd(
                        &std::env::var("TERMINAL_CWD").unwrap_or_default(),
                    ),
                };
                if !resolved.is_empty() {
                    parts.push(resolved);
                }
            }
            _ => {} // Unknown field names ignored, matching Python.
        }
    }
    if parts.is_empty() {
        return String::new();
    }
    parts.join(SEP)
}

/// Top-level entry point (port of `build_footer_line`): resolve config for the
/// platform, return "" when disabled, else the rendered footer.
pub fn build_footer_line(
    user_config: Option<&YamlValue>,
    platform_key: Option<&str>,
    model: Option<&str>,
    context_tokens: i64,
    context_length: Option<i64>,
    cwd: Option<&str>,
) -> String {
    let cfg = resolve_footer_config(user_config, platform_key);
    if !cfg.enabled {
        return String::new();
    }
    format_runtime_footer(model, context_tokens, context_length, cwd, &cfg.fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(text: &str) -> YamlValue {
        serde_yaml::from_str(text).unwrap()
    }

    #[test]
    fn disabled_by_default() {
        assert!(!resolve_footer_config(None, None).enabled);
        assert_eq!(
            build_footer_line(None, None, Some("openai/gpt-5"), 10, Some(100), Some("/x")),
            ""
        );
    }

    #[test]
    fn global_enable_and_fields() {
        let cfg = yaml("display:\n  runtime_footer:\n    enabled: true\n    fields: [model, context_pct]\n");
        let resolved = resolve_footer_config(Some(&cfg), None);
        assert!(resolved.enabled);
        assert_eq!(resolved.fields, vec!["model", "context_pct"]);
        let line = build_footer_line(
            Some(&cfg),
            None,
            Some("anthropic/claude-sonnet-4"),
            25,
            Some(100),
            Some("/tmp"),
        );
        assert_eq!(line, "claude-sonnet-4 · 25%");
    }

    #[test]
    fn platform_override_wins() {
        let cfg = yaml(
            "display:\n  runtime_footer:\n    enabled: true\n  platforms:\n    telegram:\n      runtime_footer:\n        enabled: false\n",
        );
        // global enabled, telegram override disables
        assert!(resolve_footer_config(Some(&cfg), None).enabled);
        assert!(!resolve_footer_config(Some(&cfg), Some("telegram")).enabled);
    }

    #[test]
    fn context_pct_skipped_without_length() {
        let line = format_runtime_footer(
            Some("m"),
            50,
            None,
            Some("/x"),
            &["context_pct".to_string()],
        );
        assert_eq!(line, "");
    }

    #[test]
    fn model_short_drops_vendor() {
        assert_eq!(model_short("openai/gpt-5.4"), "gpt-5.4");
        assert_eq!(model_short("plain"), "plain");
    }

    #[test]
    fn unknown_fields_ignored_and_empty_yields_blank() {
        let line = format_runtime_footer(None, 0, Some(0), None, &["bogus".to_string()]);
        assert_eq!(line, "");
    }
}
