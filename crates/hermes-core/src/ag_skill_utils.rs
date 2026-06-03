//! Lightweight skill metadata utilities shared by prompt_builder and skills_tool.
//!
//! Native Rust port of `agent/skill_utils.py`. This module intentionally avoids
//! heavy dependency chains. It provides frontmatter parsing, platform matching,
//! disabled-skill resolution, external-skill-dir discovery, skill condition /
//! config-var extraction, description truncation, skill-index iteration, and
//! namespace helpers.
//!
//! Where the Python relied on `hermes_constants.get_config_path()` /
//! `get_skills_dir()` / `get_hermes_home()` (methods on a runtime config in the
//! Rust tree), the porting style here accepts the relevant paths as explicit
//! arguments so the module stays self-contained and free of construction
//! coupling. Convenience env-based resolvers (`default_hermes_home`,
//! `default_config_path`, `default_skills_dir`) reproduce the fallback layout
//! (`~/.hermes`, honoring `HERMES_HOME`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_yaml::Value as YamlValue;

// ── Platform mapping ──────────────────────────────────────────────────────

/// Maps human-friendly platform names to `sys.platform`-style prefixes.
pub const PLATFORM_MAP: &[(&str, &str)] = &[
    ("macos", "darwin"),
    ("linux", "linux"),
    ("windows", "win32"),
];

/// Directory names skipped while walking skill trees.
pub const EXCLUDED_SKILL_DIRS: &[&str] = &[".git", ".github", ".hub", ".archive"];

/// Storage prefix: all skill config vars are stored under `skills.config.*`
/// in config.yaml. Skill authors declare logical keys (e.g. `wiki.path`); the
/// system adds this prefix for storage and strips it for display.
pub const SKILL_CONFIG_PREFIX: &str = "skills.config";

fn map_platform(normalized: &str) -> &str {
    for (name, mapped) in PLATFORM_MAP {
        if *name == normalized {
            return mapped;
        }
    }
    normalized
}

/// The current platform as a `sys.platform`-style string.
///
/// Mirrors Python's `sys.platform`: `"darwin"` on macOS, `"linux"` on Linux,
/// `"win32"` on Windows.
pub fn current_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        std::env::consts::OS
    }
}

// ── Default path resolution ────────────────────────────────────────────────

/// Resolve the Hermes home directory, honoring `HERMES_HOME`, defaulting to
/// `~/.hermes`.
pub fn default_hermes_home() -> PathBuf {
    if let Ok(value) = std::env::var("HERMES_HOME") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes")
}

/// Resolve the config.yaml path under the Hermes home.
pub fn default_config_path() -> PathBuf {
    default_hermes_home().join("config.yaml")
}

/// Resolve the local skills directory under the Hermes home.
pub fn default_skills_dir() -> PathBuf {
    default_hermes_home().join("skills")
}

// ── YAML helpers ────────────────────────────────────────────────────────────

/// Parse a YAML document into a `serde_yaml::Value`. Returns `None` on error.
pub fn yaml_load(content: &str) -> Option<YamlValue> {
    serde_yaml::from_str::<YamlValue>(content).ok()
}

fn yaml_get<'a>(map: &'a YamlValue, key: &str) -> Option<&'a YamlValue> {
    map.as_mapping()
        .and_then(|m| m.get(&YamlValue::String(key.to_string())))
}

fn yaml_to_string(value: &YamlValue) -> String {
    match value {
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => b.to_string(),
        YamlValue::Number(n) => n.to_string(),
        YamlValue::Null => "".to_string(),
        other => serde_yaml::to_string(other)
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
    }
}

// ── Frontmatter parsing ──────────────────────────────────────────────────

/// Parse YAML frontmatter from a markdown string.
///
/// Returns `(frontmatter, remaining_body)`. When the content has no leading
/// `---` delimiter or no closing delimiter, the frontmatter is empty and the
/// body is the original content. On malformed YAML, falls back to simple
/// `key: value` splitting.
pub fn parse_frontmatter(content: &str) -> (YamlValue, String) {
    let empty = YamlValue::Mapping(Default::default());

    if !content.starts_with("---") {
        return (empty, content.to_string());
    }

    // Search within content[3:] for the closing `\n---\s*\n`.
    let rest = &content[3..];
    let end_re = Regex::new(r"\n---[ \t]*\n").expect("valid regex");
    let Some(m) = end_re.find(rest) else {
        return (empty, content.to_string());
    };

    // Python: yaml_content = content[3 : end_match.start() + 3]
    //         body = content[end_match.end() + 3 :]
    let yaml_content = &content[3..m.start() + 3];
    let body = content[m.end() + 3..].to_string();

    match yaml_load(yaml_content) {
        Some(parsed) if parsed.is_mapping() => (parsed, body),
        _ => {
            // Fallback: simple key:value parsing for malformed YAML.
            let mut mapping = serde_yaml::Mapping::new();
            for line in yaml_content.trim().split('\n') {
                if let Some(idx) = line.find(':') {
                    let key = line[..idx].trim().to_string();
                    let value = line[idx + 1..].trim().to_string();
                    mapping.insert(YamlValue::String(key), YamlValue::String(value));
                }
            }
            (YamlValue::Mapping(mapping), body)
        }
    }
}

// ── Platform matching ─────────────────────────────────────────────────────

/// Return `true` when the skill is compatible with the current OS.
///
/// Skills declare platform requirements via a top-level `platforms` list. When
/// the field is absent or empty, the skill is compatible with all platforms.
pub fn skill_matches_platform(frontmatter: &YamlValue) -> bool {
    let Some(platforms) = yaml_get(frontmatter, "platforms") else {
        return true;
    };

    let list: Vec<YamlValue> = match platforms {
        YamlValue::Null => return true,
        YamlValue::Sequence(seq) => {
            if seq.is_empty() {
                return true;
            }
            seq.clone()
        }
        // Non-list, truthy value: wrap as single-element list.
        YamlValue::String(s) if s.is_empty() => return true,
        other => vec![other.clone()],
    };

    let current = current_platform();
    for platform in &list {
        let normalized = yaml_to_string(platform).to_lowercase().trim().to_string();
        let mapped = map_platform(&normalized);
        if current.starts_with(mapped) {
            return true;
        }
    }
    false
}

// ── Disabled skills ───────────────────────────────────────────────────────

fn normalize_string_set(value: Option<&YamlValue>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(value) = value else {
        return out;
    };
    match value {
        YamlValue::Null => {}
        YamlValue::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                out.insert(t.to_string());
            }
        }
        YamlValue::Sequence(seq) => {
            for v in seq {
                let s = yaml_to_string(v);
                let t = s.trim();
                if !t.is_empty() {
                    out.insert(t.to_string());
                }
            }
        }
        other => {
            let s = yaml_to_string(other);
            let t = s.trim();
            if !t.is_empty() {
                out.insert(t.to_string());
            }
        }
    }
    out
}

/// Read disabled skill names from config.yaml at `config_path`.
///
/// `platform` is the explicit platform name (e.g. `"telegram"`). When `None`,
/// the caller is expected to have already resolved it from the environment
/// (`HERMES_PLATFORM` / `HERMES_SESSION_PLATFORM`); this port reproduces that
/// env fallback via [`resolve_platform`]. Falls back to the global `disabled`
/// list when no platform is determined.
pub fn get_disabled_skill_names(config_path: &Path, platform: Option<&str>) -> BTreeSet<String> {
    if !config_path.exists() {
        return BTreeSet::new();
    }
    let Ok(text) = std::fs::read_to_string(config_path) else {
        return BTreeSet::new();
    };
    let Some(parsed) = yaml_load(&text) else {
        log::debug!("Could not read skill config {}", config_path.display());
        return BTreeSet::new();
    };
    if !parsed.is_mapping() {
        return BTreeSet::new();
    }

    let Some(skills_cfg) = yaml_get(&parsed, "skills") else {
        return BTreeSet::new();
    };
    if !skills_cfg.is_mapping() {
        return BTreeSet::new();
    }

    let resolved_platform = resolve_platform(platform);
    if let Some(plat) = resolved_platform.as_deref() {
        if let Some(platform_disabled) = yaml_get(skills_cfg, "platform_disabled") {
            if platform_disabled.is_mapping() {
                if let Some(value) = yaml_get(platform_disabled, plat) {
                    // Python: `if platform_disabled is not None: return ...`
                    // A present (even empty/null) entry short-circuits.
                    return normalize_string_set(Some(value));
                }
            }
        }
    }
    normalize_string_set(yaml_get(skills_cfg, "disabled"))
}

/// Resolve the effective platform: explicit argument, else `HERMES_PLATFORM`,
/// else `HERMES_SESSION_PLATFORM` from the environment.
pub fn resolve_platform(platform: Option<&str>) -> Option<String> {
    if let Some(p) = platform {
        if !p.is_empty() {
            return Some(p.to_string());
        }
    }
    for var in ["HERMES_PLATFORM", "HERMES_SESSION_PLATFORM"] {
        if let Ok(value) = std::env::var(var) {
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

// ── Path expansion (~ and ${VAR}) ────────────────────────────────────────────

/// Expand a leading `~` (and `~/...`) to the user's home directory.
pub fn expand_user(input: &str) -> String {
    if input == "~" {
        return dirs::home_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| input.to_string());
    }
    if let Some(rest) = input.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    input.to_string()
}

/// Expand `${VAR}` and `$VAR` environment-variable references.
pub fn expand_vars(input: &str) -> String {
    let re = Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
        .expect("valid regex");
    re.replace_all(input, |caps: &regex::Captures| {
        let name = caps
            .get(1)
            .or_else(|| caps.get(2))
            .map(|m| m.as_str())
            .unwrap_or("");
        match std::env::var(name) {
            Ok(v) => v,
            // Mirror os.path.expandvars: leave unmatched references untouched.
            Err(_) => caps.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
        }
    })
    .into_owned()
}

/// Apply `expandvars` then `expanduser`, matching the Python call ordering.
pub fn expand_path(input: &str) -> String {
    expand_user(&expand_vars(input))
}

// ── External skills directories ──────────────────────────────────────────

/// Read `skills.external_dirs` from config.yaml and return validated paths.
///
/// Each entry is expanded (`~` and `${VAR}`) and resolved to an absolute path.
/// Relative paths resolve against `hermes_home`. Only existing directories are
/// returned. Duplicates and paths equal to `local_skills` are skipped.
pub fn get_external_skills_dirs(
    config_path: &Path,
    hermes_home: &Path,
    local_skills: &Path,
) -> Vec<PathBuf> {
    if !config_path.exists() {
        return Vec::new();
    }
    let Ok(text) = std::fs::read_to_string(config_path) else {
        return Vec::new();
    };
    let Some(parsed) = yaml_load(&text) else {
        return Vec::new();
    };
    if !parsed.is_mapping() {
        return Vec::new();
    }
    let Some(skills_cfg) = yaml_get(&parsed, "skills") else {
        return Vec::new();
    };
    if !skills_cfg.is_mapping() {
        return Vec::new();
    }
    let Some(raw_dirs) = yaml_get(skills_cfg, "external_dirs") else {
        return Vec::new();
    };

    let entries: Vec<YamlValue> = match raw_dirs {
        YamlValue::Null => return Vec::new(),
        YamlValue::String(s) if s.is_empty() => return Vec::new(),
        YamlValue::String(_) => vec![raw_dirs.clone()],
        YamlValue::Sequence(seq) => {
            if seq.is_empty() {
                return Vec::new();
            }
            seq.clone()
        }
        _ => return Vec::new(),
    };

    let local_skills = resolve_path(local_skills);
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let mut result: Vec<PathBuf> = Vec::new();

    for entry in &entries {
        let entry = yaml_to_string(entry);
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let expanded = expand_path(entry);
        let p = PathBuf::from(&expanded);
        let p = if !p.is_absolute() {
            resolve_path(&hermes_home.join(&p))
        } else {
            resolve_path(&p)
        };
        if p == local_skills {
            continue;
        }
        if seen.contains(&p) {
            continue;
        }
        if p.is_dir() {
            seen.insert(p.clone());
            result.push(p);
        } else {
            log::debug!("External skills dir does not exist, skipping: {}", p.display());
        }
    }

    result
}

/// Best-effort path canonicalization. Falls back to the input path when the
/// target does not yet exist (mirrors `Path.resolve()` which does not require
/// existence).
fn resolve_path(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Return all skill directories: local skills dir first, then external dirs in
/// config order. The local dir is always included even if it does not exist.
pub fn get_all_skills_dirs(config_path: &Path, hermes_home: &Path, skills_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![skills_dir.to_path_buf()];
    dirs.extend(get_external_skills_dirs(config_path, hermes_home, skills_dir));
    dirs
}

// ── Condition extraction ──────────────────────────────────────────────────

/// Conditional activation fields extracted from a skill's frontmatter
/// `metadata.hermes` block. Each field defaults to an empty list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkillConditions {
    pub fallback_for_toolsets: Vec<YamlValue>,
    pub requires_toolsets: Vec<YamlValue>,
    pub fallback_for_tools: Vec<YamlValue>,
    pub requires_tools: Vec<YamlValue>,
}

fn list_field(map: &YamlValue, key: &str) -> Vec<YamlValue> {
    match yaml_get(map, key) {
        Some(YamlValue::Sequence(seq)) => seq.clone(),
        Some(YamlValue::Null) | None => Vec::new(),
        // Python returns the raw value as default `[]` only when absent; when
        // present-but-not-a-list the dict.get returns the value as-is. To stay
        // faithful, wrap a present scalar in a single-element list.
        Some(other) => vec![other.clone()],
    }
}

/// Extract conditional activation fields from parsed frontmatter.
pub fn extract_skill_conditions(frontmatter: &YamlValue) -> SkillConditions {
    let metadata = yaml_get(frontmatter, "metadata");
    let hermes = match metadata {
        Some(m) if m.is_mapping() => match yaml_get(m, "hermes") {
            Some(h) if h.is_mapping() => Some(h.clone()),
            _ => None,
        },
        _ => None,
    };
    let hermes = hermes.unwrap_or_else(|| YamlValue::Mapping(Default::default()));
    SkillConditions {
        fallback_for_toolsets: list_field(&hermes, "fallback_for_toolsets"),
        requires_toolsets: list_field(&hermes, "requires_toolsets"),
        fallback_for_tools: list_field(&hermes, "fallback_for_tools"),
        requires_tools: list_field(&hermes, "requires_tools"),
    }
}

// ── Skill config extraction ───────────────────────────────────────────────

/// A single skill-declared config variable.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillConfigVar {
    pub key: String,
    pub description: String,
    /// Declared default, if any (preserved as a YAML value).
    pub default: Option<YamlValue>,
    /// Prompt text shown when collecting the value; defaults to `description`.
    pub prompt: String,
    /// Owning skill name (populated by [`discover_all_skill_config_vars`]).
    pub skill: Option<String>,
}

/// Extract config variable declarations from `metadata.hermes.config`.
///
/// Invalid or incomplete entries (missing key or description, duplicate keys)
/// are silently skipped.
pub fn extract_skill_config_vars(frontmatter: &YamlValue) -> Vec<SkillConfigVar> {
    let metadata = match yaml_get(frontmatter, "metadata") {
        Some(m) if m.is_mapping() => m,
        _ => return Vec::new(),
    };
    let hermes = match yaml_get(metadata, "hermes") {
        Some(h) if h.is_mapping() => h,
        _ => return Vec::new(),
    };
    let raw = match yaml_get(hermes, "config") {
        Some(YamlValue::Null) | None => return Vec::new(),
        Some(YamlValue::String(s)) if s.is_empty() => return Vec::new(),
        Some(v) => v,
    };

    let items: Vec<YamlValue> = match raw {
        YamlValue::Mapping(_) => vec![raw.clone()],
        YamlValue::Sequence(seq) => seq.clone(),
        _ => return Vec::new(),
    };

    let mut result: Vec<SkillConfigVar> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for item in &items {
        if !item.is_mapping() {
            continue;
        }
        let key = yaml_get(item, "key")
            .map(yaml_to_string)
            .unwrap_or_default()
            .trim()
            .to_string();
        if key.is_empty() || seen.contains(&key) {
            continue;
        }
        let desc = yaml_get(item, "description")
            .map(yaml_to_string)
            .unwrap_or_default()
            .trim()
            .to_string();
        if desc.is_empty() {
            continue;
        }
        let default = match yaml_get(item, "default") {
            Some(YamlValue::Null) | None => None,
            Some(v) => Some(v.clone()),
        };
        let prompt = match yaml_get(item, "prompt") {
            Some(YamlValue::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
            _ => desc.clone(),
        };
        seen.insert(key.clone());
        result.push(SkillConfigVar {
            key,
            description: desc,
            default,
            prompt,
            skill: None,
        });
    }
    result
}

/// Scan all enabled skills and collect deduplicated config-var declarations.
///
/// Walks every skills directory, parses each `SKILL.md` frontmatter, and
/// attributes each var to its owning skill. Disabled and platform-incompatible
/// skills are excluded.
pub fn discover_all_skill_config_vars(
    config_path: &Path,
    hermes_home: &Path,
    skills_dir: &Path,
) -> Vec<SkillConfigVar> {
    let mut all_vars: Vec<SkillConfigVar> = Vec::new();
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();

    let disabled = get_disabled_skill_names(config_path, None);
    for dir in get_all_skills_dirs(config_path, hermes_home, skills_dir) {
        if !dir.is_dir() {
            continue;
        }
        for skill_file in iter_skill_index_files(&dir, "SKILL.md") {
            let Ok(raw) = std::fs::read_to_string(&skill_file) else {
                continue;
            };
            let (frontmatter, _) = parse_frontmatter(&raw);

            let skill_name = yaml_get(&frontmatter, "name")
                .map(yaml_to_string)
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    skill_file
                        .parent()
                        .and_then(|p| p.file_name())
                        .map(|n| n.to_string_lossy().into_owned())
                })
                .unwrap_or_default();

            if disabled.contains(&skill_name) {
                continue;
            }
            if !skill_matches_platform(&frontmatter) {
                continue;
            }

            for mut var in extract_skill_config_vars(&frontmatter) {
                if !seen_keys.contains(&var.key) {
                    seen_keys.insert(var.key.clone());
                    var.skill = Some(skill_name.clone());
                    all_vars.push(var);
                }
            }
        }
    }

    all_vars
}

// ── Skill config value resolution ────────────────────────────────────────────

fn resolve_dotpath<'a>(config: &'a YamlValue, dotted_key: &str) -> Option<&'a YamlValue> {
    let mut current = config;
    for part in dotted_key.split('.') {
        match current.as_mapping() {
            Some(m) => match m.get(&YamlValue::String(part.to_string())) {
                Some(next) => current = next,
                None => return None,
            },
            None => return None,
        }
    }
    Some(current)
}

/// Resolve current values for skill config vars from config.yaml.
///
/// Skill config is stored under `skills.config.<key>`. Returns a list of
/// `(logical_key, value)` pairs (preserving declaration order) where each value
/// is the stored value, the declared default, or `""`. String path values
/// containing `~` or `${` are expanded.
pub fn resolve_skill_config_values(
    config_path: &Path,
    config_vars: &[SkillConfigVar],
) -> Vec<(String, YamlValue)> {
    let mut config = YamlValue::Mapping(Default::default());
    if config_path.exists() {
        if let Ok(text) = std::fs::read_to_string(config_path) {
            if let Some(parsed) = yaml_load(&text) {
                if parsed.is_mapping() {
                    config = parsed;
                }
            }
        }
    }

    let mut resolved: Vec<(String, YamlValue)> = Vec::new();
    for var in config_vars {
        let logical_key = &var.key;
        let storage_key = format!("{SKILL_CONFIG_PREFIX}.{logical_key}");
        let stored = resolve_dotpath(&config, &storage_key);

        let mut value: YamlValue = match stored {
            Some(YamlValue::Null) | None => var
                .default
                .clone()
                .unwrap_or_else(|| YamlValue::String(String::new())),
            Some(YamlValue::String(s)) if s.trim().is_empty() => var
                .default
                .clone()
                .unwrap_or_else(|| YamlValue::String(String::new())),
            Some(v) => v.clone(),
        };

        if let YamlValue::String(s) = &value {
            if s.contains('~') || s.contains("${") {
                value = YamlValue::String(expand_path(s));
            }
        }

        resolved.push((logical_key.clone(), value));
    }

    resolved
}

// ── Description extraction ────────────────────────────────────────────────

/// Extract a truncated description from parsed frontmatter.
///
/// The raw value is trimmed and stripped of surrounding quotes, then truncated
/// to 60 characters (with a trailing `...` after 57 chars).
pub fn extract_skill_description(frontmatter: &YamlValue) -> String {
    let raw = match yaml_get(frontmatter, "description") {
        Some(YamlValue::Null) | None => return String::new(),
        Some(YamlValue::String(s)) if s.is_empty() => return String::new(),
        Some(v) => yaml_to_string(v),
    };
    if raw.is_empty() {
        return String::new();
    }
    let desc = raw.trim().trim_matches(|c| c == '\'' || c == '"');
    let count = desc.chars().count();
    if count > 60 {
        let truncated: String = desc.chars().take(57).collect();
        format!("{truncated}...")
    } else {
        desc.to_string()
    }
}

// ── File iteration ────────────────────────────────────────────────────────

/// Walk `skills_dir` returning sorted paths matching `filename`.
///
/// Excludes `.git`, `.github`, `.hub`, `.archive` directories. Results are
/// sorted by their path relative to `skills_dir`, matching the Python ordering.
pub fn iter_skill_index_files(skills_dir: &Path, filename: &str) -> Vec<PathBuf> {
    let mut matches: Vec<PathBuf> = Vec::new();
    walk_skill_dir(skills_dir, filename, &mut matches);

    matches.sort_by(|a, b| {
        let ra = a.strip_prefix(skills_dir).unwrap_or(a);
        let rb = b.strip_prefix(skills_dir).unwrap_or(b);
        ra.to_string_lossy().cmp(&rb.to_string_lossy())
    });
    matches
}

fn walk_skill_dir(dir: &Path, filename: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs: Vec<PathBuf> = Vec::new();
    let mut has_file = false;

    for entry in entries.flatten() {
        let path = entry.path();
        // Follow symlinks: use metadata() (not symlink_metadata) to mirror
        // os.walk(followlinks=True).
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !EXCLUDED_SKILL_DIRS.contains(&name.as_ref()) {
                subdirs.push(path);
            }
        } else if meta.is_file() {
            if entry.file_name().to_string_lossy() == filename {
                has_file = true;
            }
        }
    }

    if has_file {
        out.push(dir.join(filename));
    }
    for sub in subdirs {
        walk_skill_dir(&sub, filename, out);
    }
}

// ── Namespace helpers for plugin-provided skills ───────────────────────────

/// Split `"namespace:skill-name"` into `(Some(namespace), bare_name)`.
///
/// Returns `(None, name)` when there is no `':'`.
pub fn parse_qualified_name(name: &str) -> (Option<String>, String) {
    match name.split_once(':') {
        Some((ns, bare)) => (Some(ns.to_string()), bare.to_string()),
        None => (None, name.to_string()),
    }
}

/// Check whether `candidate` is a valid namespace (`[a-zA-Z0-9_-]+`).
pub fn is_valid_namespace(candidate: Option<&str>) -> bool {
    let Some(candidate) = candidate else {
        return false;
    };
    if candidate.is_empty() {
        return false;
    }
    let re = Regex::new(r"^[a-zA-Z0-9_-]+$").expect("valid regex");
    re.is_match(candidate)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn yaml(s: &str) -> YamlValue {
        yaml_load(s).unwrap()
    }

    #[test]
    fn frontmatter_basic() {
        let content = "---\nname: foo\ndescription: hi\n---\nbody text";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(yaml_to_string(yaml_get(&fm, "name").unwrap()), "foo");
        assert_eq!(yaml_to_string(yaml_get(&fm, "description").unwrap()), "hi");
        assert_eq!(body, "body text");
    }

    #[test]
    fn frontmatter_no_leading_delim() {
        let content = "no frontmatter here";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.as_mapping().unwrap().is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn frontmatter_no_closing_delim() {
        let content = "---\nname: foo\nstill going";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.as_mapping().unwrap().is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn frontmatter_malformed_fallback() {
        // Tab in a value makes serde_yaml fail to parse → fallback key:value.
        let content = "---\nkey1: val1\n: noop\nkey2: val2\n---\nbody";
        // Make YAML actually invalid: a bare colon line + duplicate-ish.
        let content2 = "---\n\tbad: [unterminated\nkey2: val2\n---\nbody";
        let (_fm, body) = parse_frontmatter(content);
        // content parses fine as yaml; just ensure body correct.
        assert_eq!(body, "body");

        let (fm2, body2) = parse_frontmatter(content2);
        assert_eq!(body2, "body");
        // Fallback splitting captures key2.
        assert_eq!(
            yaml_get(&fm2, "key2").map(yaml_to_string),
            Some("val2".to_string())
        );
    }

    #[test]
    fn platform_absent_matches() {
        let fm = yaml("name: x");
        assert!(skill_matches_platform(&fm));
    }

    #[test]
    fn platform_empty_list_matches() {
        let fm = yaml("platforms: []");
        assert!(skill_matches_platform(&fm));
    }

    #[test]
    fn platform_matching_current() {
        let cur = current_platform();
        let human = match cur {
            "darwin" => "macos",
            "win32" => "windows",
            _ => "linux",
        };
        let fm = yaml(&format!("platforms: [{human}]"));
        assert!(skill_matches_platform(&fm));
    }

    #[test]
    fn platform_non_matching() {
        let cur = current_platform();
        let other = if cur == "darwin" { "windows" } else { "macos" };
        let fm = yaml(&format!("platforms: [{other}]"));
        // On linux, "macos"->darwin won't match, "windows"->win32 won't match.
        if cur == "linux" {
            assert!(!skill_matches_platform(&fm));
        }
    }

    #[test]
    fn platform_scalar_value() {
        let cur = current_platform();
        let human = match cur {
            "darwin" => "macos",
            "win32" => "windows",
            _ => "linux",
        };
        let fm = yaml(&format!("platforms: {human}"));
        assert!(skill_matches_platform(&fm));
    }

    #[test]
    fn conditions_extraction() {
        let fm = yaml(
            "metadata:\n  hermes:\n    requires_tools: [a, b]\n    fallback_for_toolsets: [x]\n",
        );
        let c = extract_skill_conditions(&fm);
        assert_eq!(c.requires_tools.len(), 2);
        assert_eq!(c.fallback_for_toolsets.len(), 1);
        assert!(c.requires_toolsets.is_empty());
        assert!(c.fallback_for_tools.is_empty());
    }

    #[test]
    fn conditions_metadata_not_dict() {
        let fm = yaml("metadata: just-a-string");
        let c = extract_skill_conditions(&fm);
        assert_eq!(c, SkillConditions::default());
    }

    #[test]
    fn config_vars_basic() {
        let fm = yaml(
            "metadata:\n  hermes:\n    config:\n      - key: wiki.path\n        description: Path to wiki\n        default: \"~/wiki\"\n        prompt: Wiki dir\n",
        );
        let vars = extract_skill_config_vars(&fm);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].key, "wiki.path");
        assert_eq!(vars[0].description, "Path to wiki");
        assert_eq!(vars[0].prompt, "Wiki dir");
        assert!(vars[0].default.is_some());
    }

    #[test]
    fn config_vars_prompt_defaults_to_desc() {
        let fm = yaml(
            "metadata:\n  hermes:\n    config:\n      - key: k\n        description: D\n",
        );
        let vars = extract_skill_config_vars(&fm);
        assert_eq!(vars[0].prompt, "D");
        assert!(vars[0].default.is_none());
    }

    #[test]
    fn config_vars_skips_missing_desc_and_dupes() {
        let fm = yaml(
            "metadata:\n  hermes:\n    config:\n      - key: k\n        description: D\n      - key: k\n        description: D2\n      - key: nodesc\n",
        );
        let vars = extract_skill_config_vars(&fm);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].description, "D");
    }

    #[test]
    fn config_vars_single_dict_wrapped() {
        let fm = yaml(
            "metadata:\n  hermes:\n    config:\n      key: k\n      description: D\n",
        );
        let vars = extract_skill_config_vars(&fm);
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn description_truncation() {
        let fm = yaml(&format!("description: \"{}\"", "a".repeat(100)));
        let d = extract_skill_description(&fm);
        assert_eq!(d.chars().count(), 60);
        assert!(d.ends_with("..."));
    }

    #[test]
    fn description_quote_strip() {
        let fm = yaml("description: \"'quoted'\"");
        let d = extract_skill_description(&fm);
        assert_eq!(d, "quoted");
    }

    #[test]
    fn description_absent() {
        let fm = yaml("name: x");
        assert_eq!(extract_skill_description(&fm), "");
    }

    #[test]
    fn namespace_parsing() {
        assert_eq!(
            parse_qualified_name("ns:skill"),
            (Some("ns".to_string()), "skill".to_string())
        );
        assert_eq!(
            parse_qualified_name("plain"),
            (None, "plain".to_string())
        );
        assert_eq!(
            parse_qualified_name("a:b:c"),
            (Some("a".to_string()), "b:c".to_string())
        );
    }

    #[test]
    fn namespace_validity() {
        assert!(is_valid_namespace(Some("foo-bar_1")));
        assert!(!is_valid_namespace(Some("foo bar")));
        assert!(!is_valid_namespace(Some("foo:bar")));
        assert!(!is_valid_namespace(Some("")));
        assert!(!is_valid_namespace(None));
    }

    #[test]
    fn normalize_set_variants() {
        assert!(normalize_string_set(None).is_empty());
        assert!(normalize_string_set(Some(&YamlValue::Null)).is_empty());
        let single = YamlValue::String(" a ".to_string());
        assert_eq!(
            normalize_string_set(Some(&single)),
            BTreeSet::from(["a".to_string()])
        );
        let list = yaml("[a, ' b ', '']");
        assert_eq!(
            normalize_string_set(Some(&list)),
            BTreeSet::from(["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn dotpath_resolution() {
        let cfg = yaml("skills:\n  config:\n    wiki.path: /x\n");
        // Note: "wiki.path" is a literal key here, not nested.
        let v = resolve_dotpath(&cfg, "skills.config.wiki.path");
        // skills.config has key "wiki.path" not "wiki"->"path", so lookup fails.
        assert!(v.is_none());

        let cfg2 = yaml("a:\n  b:\n    c: 5\n");
        let v2 = resolve_dotpath(&cfg2, "a.b.c");
        assert_eq!(v2, Some(&YamlValue::Number(5.into())));
        assert!(resolve_dotpath(&cfg2, "a.b.missing").is_none());
    }

    #[test]
    fn iter_skill_files_sorted_and_excludes() {
        let tmp = std::env::temp_dir().join(format!("skutil_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("beta")).unwrap();
        std::fs::create_dir_all(tmp.join("alpha")).unwrap();
        std::fs::create_dir_all(tmp.join(".git")).unwrap();
        for sub in ["alpha", "beta", ".git"] {
            let mut f = std::fs::File::create(tmp.join(sub).join("SKILL.md")).unwrap();
            writeln!(f, "name: {sub}").unwrap();
        }
        let files = iter_skill_index_files(&tmp, "SKILL.md");
        let rels: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(&tmp).unwrap().to_string_lossy().into_owned())
            .collect();
        // .git excluded; alpha before beta.
        assert_eq!(
            rels,
            vec![
                format!("alpha{}SKILL.md", std::path::MAIN_SEPARATOR),
                format!("beta{}SKILL.md", std::path::MAIN_SEPARATOR),
            ]
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn disabled_global_and_platform() {
        let tmp = std::env::temp_dir().join(format!("skutil_cfg_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let cfg = tmp.join("config.yaml");
        std::fs::write(
            &cfg,
            "skills:\n  disabled: [foo, bar]\n  platform_disabled:\n    telegram: [baz]\n",
        )
        .unwrap();

        let global = get_disabled_skill_names(&cfg, None);
        assert_eq!(global, BTreeSet::from(["foo".to_string(), "bar".to_string()]));

        let plat = get_disabled_skill_names(&cfg, Some("telegram"));
        assert_eq!(plat, BTreeSet::from(["baz".to_string()]));

        // Unknown platform → falls back to global disabled.
        let unknown = get_disabled_skill_names(&cfg, Some("nope"));
        assert_eq!(unknown, BTreeSet::from(["foo".to_string(), "bar".to_string()]));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resolve_config_values_default_and_stored() {
        let tmp = std::env::temp_dir().join(format!("skutil_rv_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let cfg = tmp.join("config.yaml");
        // Storage key "skills.config.stored.key" walks nested mappings, so the
        // config must be nested skills -> config -> stored -> key (matching the
        // Python `_resolve_dotpath` dotted-walk semantics).
        std::fs::write(
            &cfg,
            "skills:\n  config:\n    stored:\n      key: /real/path\n",
        )
        .unwrap();

        let vars = vec![
            SkillConfigVar {
                key: "stored.key".to_string(),
                description: "d".to_string(),
                default: Some(YamlValue::String("/default".to_string())),
                prompt: "p".to_string(),
                skill: None,
            },
            SkillConfigVar {
                key: "missing.key".to_string(),
                description: "d".to_string(),
                default: Some(YamlValue::String("/fallback".to_string())),
                prompt: "p".to_string(),
                skill: None,
            },
        ];
        let resolved = resolve_skill_config_values(&cfg, &vars);
        assert_eq!(resolved[0].0, "stored.key");
        assert_eq!(resolved[0].1, YamlValue::String("/real/path".to_string()));
        assert_eq!(resolved[1].1, YamlValue::String("/fallback".to_string()));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn expand_path_tilde() {
        if let Some(home) = dirs::home_dir() {
            let got = expand_path("~/foo");
            assert_eq!(got, home.join("foo").to_string_lossy());
        }
    }

    #[test]
    fn expand_vars_basic() {
        unsafe { std::env::set_var("SKUTIL_TEST_VAR", "xyz") };
        assert_eq!(expand_vars("${SKUTIL_TEST_VAR}/p"), "xyz/p");
        assert_eq!(expand_vars("$SKUTIL_TEST_VAR"), "xyz");
        // Unknown var left intact.
        assert_eq!(expand_vars("${NOPE_SKUTIL}"), "${NOPE_SKUTIL}");
    }
}
