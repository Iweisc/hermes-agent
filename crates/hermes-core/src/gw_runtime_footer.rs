//! Gateway runtime-metadata footer.
//!
//! Faithful port of `gateway/runtime_footer.py`.
//!
//! Renders a compact footer showing runtime state (`model · context% · cwd`)
//! and appends it to the FINAL message of an agent turn when enabled. Off by
//! default to keep replies minimal.
//!
//! Config (`~/.hermes/config.yaml`):
//!
//! ```yaml
//! display:
//!   runtime_footer:
//!     enabled: true                       # off by default
//!     fields: [model, context_pct, cwd]   # order shown; drop any to hide
//! ```
//!
//! Per-platform overrides live under
//! `display.platforms.<platform>.runtime_footer`. Users can toggle the global
//! setting with `/footer on|off` from both the CLI and any gateway platform.
//!
//! Pure logic (config merge + string formatting). The only I/O is reading
//! `$HOME` (for collapsing cwd to `~`) and the `TERMINAL_CWD` environment
//! variable as a cwd fallback — matching the Python exactly.

use serde_yaml::Value as YamlValue;

/// Default footer fields, in display order.
pub const DEFAULT_FIELDS: &[&str] = &["model", "context_pct", "cwd"];

/// Separator between footer parts.
pub const SEP: &str = " · ";

/// Resolved runtime-footer configuration (enabled flag + ordered field list).
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

/// Fetch a string-keyed value from a YAML mapping.
fn yaml_get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

/// Coerce a YAML scalar to Python `bool()`-style truthiness.
///
/// Mirrors `bool(global_cfg.get("enabled"))`: `true`/non-zero number/non-empty
/// string/non-empty collection -> `true`; `false`/`0`/empty/null -> `false`.
fn yaml_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Null => false,
        YamlValue::Bool(b) => *b,
        YamlValue::Number(n) => {
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
        YamlValue::String(s) => !s.is_empty(),
        YamlValue::Sequence(seq) => !seq.is_empty(),
        YamlValue::Mapping(m) => !m.is_empty(),
        // serde_yaml tagged values: treat as truthy (non-null presence).
        _ => true,
    }
}

/// Extract a `fields` list from a footer mapping, applying Python's
/// `isinstance(..., list) and <list>` guard: only a non-empty sequence is used,
/// and each element is stringified via `str(f)`.
fn fields_from_mapping(footer: &serde_yaml::Mapping) -> Option<Vec<String>> {
    match yaml_get(footer, "fields") {
        Some(YamlValue::Sequence(seq)) if !seq.is_empty() => {
            Some(seq.iter().map(yaml_to_py_str).collect())
        }
        _ => None,
    }
}

/// Stringify a YAML scalar the way Python's `str(f)` would for the common
/// scalar cases (strings, bools, ints, floats, null).
fn yaml_to_py_str(value: &YamlValue) -> String {
    match value {
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => {
            // Python str(True) -> "True"
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        YamlValue::Null => "None".to_string(),
        YamlValue::Number(n) => n.to_string(),
        other => serde_yaml::to_string(other)
            .map(|s| s.trim_end().to_string())
            .unwrap_or_default(),
    }
}

/// Apply an `enabled`/`fields` override from a footer mapping onto `resolved`.
fn apply_footer_override(resolved: &mut FooterConfig, footer: &serde_yaml::Mapping) {
    if let Some(enabled) = yaml_get(footer, "enabled") {
        resolved.enabled = yaml_truthy(enabled);
    }
    if let Some(fields) = fields_from_mapping(footer) {
        resolved.fields = fields;
    }
}

/// Resolve the effective runtime-footer config for `platform_key`.
///
/// Merge order (later wins):
///   1. Built-in defaults (`enabled=false`)
///   2. `display.runtime_footer`
///   3. `display.platforms.<platform_key>.runtime_footer`
///
/// `user_config` is the parsed `config.yaml` mapping (or `None`).
pub fn resolve_footer_config(
    user_config: Option<&YamlValue>,
    platform_key: Option<&str>,
) -> FooterConfig {
    let mut resolved = FooterConfig::default();

    // `cfg = (user_config or {}).get("display") or {}`
    let display = user_config
        .and_then(|v| v.as_mapping())
        .and_then(|m| yaml_get(m, "display"))
        .and_then(|v| v.as_mapping());

    let display = match display {
        Some(d) => d,
        None => return resolved,
    };

    // Global runtime_footer override.
    if let Some(global_cfg) = yaml_get(display, "runtime_footer").and_then(|v| v.as_mapping()) {
        apply_footer_override(&mut resolved, global_cfg);
    }

    // Per-platform override.
    if let Some(platform_key) = platform_key {
        if !platform_key.is_empty() {
            if let Some(platforms) = yaml_get(display, "platforms").and_then(|v| v.as_mapping()) {
                if let Some(plat_cfg) = yaml_get(platforms, platform_key).and_then(|v| v.as_mapping())
                {
                    if let Some(plat_footer) =
                        yaml_get(plat_cfg, "runtime_footer").and_then(|v| v.as_mapping())
                    {
                        apply_footer_override(&mut resolved, plat_footer);
                    }
                }
            }
        }
    }

    resolved
}

/// Return `cwd` with `$HOME` collapsed to `~`. Empty string if unset.
fn home_relative_cwd(cwd: &str) -> String {
    if cwd.is_empty() {
        return String::new();
    }
    // os.path.expanduser("~")
    let home = home_dir();
    // os.path.abspath(cwd)
    let p = abspath(cwd);
    if let Some(home) = home {
        if !home.is_empty() {
            let sep = std::path::MAIN_SEPARATOR;
            let home_prefix = format!("{home}{sep}");
            if p == home {
                return "~".to_string();
            }
            if p.starts_with(&home_prefix) {
                return format!("~{}", &p[home.len()..]);
            }
        }
    }
    p
}

/// Equivalent of `os.path.expanduser("~")`: resolve the home directory.
fn home_dir() -> Option<String> {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return Some(h);
        }
    }
    dirs::home_dir().and_then(|p| p.to_str().map(|s| s.to_string()))
}

/// Equivalent of `os.path.abspath(cwd)`: make absolute relative to CWD and
/// normalise without touching the filesystem (no symlink resolution, matching
/// Python's `abspath`).
fn abspath(path: &str) -> String {
    let p = std::path::Path::new(path);
    let absolute = if p.is_absolute() {
        p.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(p),
            Err(_) => p.to_path_buf(),
        }
    };
    normalize_path(&absolute)
}

/// Lexically normalise a path (collapse `.` and `..`) without filesystem
/// access, like Python's `os.path.normpath` (invoked inside `abspath`).
fn normalize_path(path: &std::path::Path) -> String {
    use std::path::Component;
    let mut out: Vec<std::ffi::OsString> = Vec::new();
    let mut root = std::path::PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => root.push(p.as_os_str()),
            Component::RootDir => root.push(std::path::MAIN_SEPARATOR_STR),
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop a normal component if present; otherwise keep `..`
                // only when there is no absolute root.
                if matches!(out.last().map(|s| s.as_os_str()), Some(_))
                    && out.last().map(|s| s.as_os_str()) != Some(std::ffi::OsStr::new(".."))
                {
                    out.pop();
                } else if root.as_os_str().is_empty() {
                    out.push(std::ffi::OsString::from(".."));
                }
            }
            Component::Normal(c) => out.push(c.to_os_string()),
        }
    }
    let mut result = root;
    for c in out {
        result.push(c);
    }
    result.to_string_lossy().to_string()
}

/// Drop the `vendor/` prefix from a model id (`openai/gpt-5.4` -> `gpt-5.4`).
fn model_short(model: Option<&str>) -> String {
    match model {
        None => String::new(),
        Some(m) if m.is_empty() => String::new(),
        Some(m) => match m.rsplit_once('/') {
            Some((_, tail)) => tail.to_string(),
            None => m.to_string(),
        },
    }
}

/// Render the footer line, or return `""` if no fields have data.
///
/// Fields are skipped silently when their underlying data is missing — a
/// partially-populated footer is better than a line with `?%` or empty slots.
///
/// * `model` — full model id (may be `None`/empty).
/// * `context_tokens` — tokens used so far.
/// * `context_length` — context window size (`None`/`<=0` hides the percentage).
/// * `cwd` — working directory; falls back to `$TERMINAL_CWD` when `None`/empty.
/// * `fields` — ordered field names to render.
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
                let m = model_short(model);
                if !m.is_empty() {
                    parts.push(m);
                }
            }
            "context_pct" => {
                if let Some(len) = context_length {
                    if len > 0 && context_tokens >= 0 {
                        let raw = (context_tokens as f64 / len as f64) * 100.0;
                        let pct = py_round(raw).clamp(0, 100);
                        parts.push(format!("{pct}%"));
                    }
                }
            }
            "cwd" => {
                let chosen = match cwd {
                    Some(c) if !c.is_empty() => c.to_string(),
                    _ => std::env::var("TERMINAL_CWD").unwrap_or_default(),
                };
                let rel = home_relative_cwd(&chosen);
                if !rel.is_empty() {
                    parts.push(rel);
                }
            }
            // Unknown field names are silently ignored.
            _ => {}
        }
    }

    if parts.is_empty() {
        return String::new();
    }
    parts.join(SEP)
}

/// Python 3 `round()` — banker's rounding (round-half-to-even) to an integer.
fn py_round(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor as i64
    } else if diff > 0.5 {
        (floor as i64) + 1
    } else {
        // Exactly halfway: round to even.
        let f = floor as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    }
}

/// Top-level entry point used by `gateway/run.py`.
///
/// Returns the footer text (empty string when disabled or no data). Callers
/// append this to the final response themselves, preserving a single blank
/// line of separation.
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
    let fields = if cfg.fields.is_empty() {
        DEFAULT_FIELDS.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.fields
    };
    format_runtime_footer(model, context_tokens, context_length, cwd, &fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(s: &str) -> YamlValue {
        serde_yaml::from_str(s).unwrap()
    }

    fn default_fields() -> Vec<String> {
        DEFAULT_FIELDS.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_disabled() {
        let cfg = resolve_footer_config(None, None);
        assert!(!cfg.enabled);
        assert_eq!(cfg.fields, default_fields());
    }

    #[test]
    fn global_enable_and_fields() {
        let c = yaml(
            "display:\n  runtime_footer:\n    enabled: true\n    fields: [model, cwd]\n",
        );
        let cfg = resolve_footer_config(Some(&c), None);
        assert!(cfg.enabled);
        assert_eq!(cfg.fields, vec!["model".to_string(), "cwd".to_string()]);
    }

    #[test]
    fn empty_fields_list_ignored() {
        let c = yaml("display:\n  runtime_footer:\n    enabled: true\n    fields: []\n");
        let cfg = resolve_footer_config(Some(&c), None);
        assert!(cfg.enabled);
        assert_eq!(cfg.fields, default_fields());
    }

    #[test]
    fn platform_override_wins() {
        let c = yaml(
            "display:\n  runtime_footer:\n    enabled: true\n  platforms:\n    slack:\n      runtime_footer:\n        enabled: false\n",
        );
        let global = resolve_footer_config(Some(&c), None);
        assert!(global.enabled);
        let slack = resolve_footer_config(Some(&c), Some("slack"));
        assert!(!slack.enabled);
    }

    #[test]
    fn platform_fields_override() {
        let c = yaml(
            "display:\n  platforms:\n    discord:\n      runtime_footer:\n        enabled: true\n        fields: [context_pct]\n",
        );
        let cfg = resolve_footer_config(Some(&c), Some("discord"));
        assert!(cfg.enabled);
        assert_eq!(cfg.fields, vec!["context_pct".to_string()]);
    }

    #[test]
    fn enabled_truthiness_string() {
        // bool("false") is True in Python — non-empty string is truthy.
        let c = yaml("display:\n  runtime_footer:\n    enabled: \"false\"\n");
        let cfg = resolve_footer_config(Some(&c), None);
        assert!(cfg.enabled);
    }

    #[test]
    fn model_short_drops_vendor() {
        assert_eq!(model_short(Some("openai/gpt-5.4")), "gpt-5.4");
        assert_eq!(model_short(Some("gpt-5.4")), "gpt-5.4");
        assert_eq!(model_short(Some("a/b/c")), "c");
        assert_eq!(model_short(None), "");
        assert_eq!(model_short(Some("")), "");
    }

    #[test]
    fn format_model_and_pct() {
        let out = format_runtime_footer(
            Some("anthropic/claude"),
            5000,
            Some(10000),
            Some("/nonexistent/path/xyz"),
            &default_fields(),
        );
        // model · 50% · /nonexistent/path/xyz
        assert!(out.starts_with("claude · 50% · "));
        assert!(out.contains(SEP));
    }

    #[test]
    fn format_pct_skipped_when_no_length() {
        let out = format_runtime_footer(
            Some("m"),
            100,
            None,
            Some("/tmp/zzz_no_such"),
            &["model".to_string(), "context_pct".to_string()],
        );
        assert_eq!(out, "m");
    }

    #[test]
    fn format_pct_clamped() {
        let out = format_runtime_footer(
            None,
            999999,
            Some(10),
            None,
            &["context_pct".to_string()],
        );
        assert_eq!(out, "100%");
    }

    #[test]
    fn format_empty_when_no_data() {
        let out = format_runtime_footer(None, 0, None, Some(""), &default_fields());
        // With TERMINAL_CWD unset and no real cwd data, model/pct empty.
        // cwd may resolve to current dir though; force a clean check on model+pct only.
        let out2 = format_runtime_footer(
            None,
            0,
            None,
            None,
            &["model".to_string(), "context_pct".to_string()],
        );
        assert_eq!(out2, "");
        let _ = out;
    }

    #[test]
    fn home_relative_collapses() {
        unsafe {
            std::env::set_var("HOME", "/Users/test");
        }
        assert_eq!(home_relative_cwd("/Users/test/proj"), "~/proj");
        assert_eq!(home_relative_cwd("/Users/test"), "~");
        assert_eq!(home_relative_cwd("/other/place"), "/other/place");
        assert_eq!(home_relative_cwd(""), "");
    }

    #[test]
    fn cwd_falls_back_to_terminal_cwd() {
        unsafe {
            std::env::set_var("HOME", "/Users/test");
            std::env::set_var("TERMINAL_CWD", "/Users/test/work");
        }
        let out = format_runtime_footer(None, 0, None, None, &["cwd".to_string()]);
        assert_eq!(out, "~/work");
        unsafe {
            std::env::remove_var("TERMINAL_CWD");
        }
    }

    #[test]
    fn build_disabled_returns_empty() {
        let c = yaml("display:\n  runtime_footer:\n    enabled: false\n");
        let out = build_footer_line(Some(&c), None, Some("m"), 1, Some(2), Some("/x"));
        assert_eq!(out, "");
    }

    #[test]
    fn build_enabled_renders() {
        let c = yaml(
            "display:\n  runtime_footer:\n    enabled: true\n    fields: [model, context_pct]\n",
        );
        let out = build_footer_line(Some(&c), None, Some("v/m"), 1, Some(4), None);
        assert_eq!(out, "m · 25%");
    }

    #[test]
    fn py_round_half_to_even() {
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(2.4), 2);
        assert_eq!(py_round(2.6), 3);
    }
}
