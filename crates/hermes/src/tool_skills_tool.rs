//! Skills Tool Module — native Rust port of `tools/skills_tool.py`.
//!
//! Provides tools for listing and viewing skill documents. Skills are
//! organized as directories containing a `SKILL.md` file (the main
//! instructions) plus optional supporting files (references, templates,
//! assets, scripts).
//!
//! Inspired by Anthropic's Claude Skills progressive-disclosure model:
//! - Metadata (name <= 64 chars, description <= 1024 chars) shown in
//!   [`skills_list`].
//! - Full instructions loaded via [`skill_view`] when needed.
//! - Linked files (references, templates, ...) loaded on demand.
//!
//! This is a faithful port of the local-skill code paths. Surfaces that
//! depend on as-yet-unported subsystems (plugin manager, gateway session
//! context, the interactive secret-capture callback, skill preprocessing,
//! credential-file mounting, usage telemetry) are modelled with minimal
//! local types / pluggable callbacks so this module is self-contained.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{json, Map, Value as JsonValue};
use serde_yaml::Value as YamlValue;

use hermes_core::ag_skill_utils::{
    get_external_skills_dirs, is_valid_namespace, iter_skill_index_files, parse_frontmatter,
    parse_qualified_name, skill_matches_platform,
};
use hermes_core::mod_hermes_constants::{display_hermes_home, get_config_path, get_hermes_home};
use hermes_core::tool_path_security::{has_traversal_component, validate_within_dir};
use hermes_core::tool_registry::tool_error;

// Anthropic-recommended limits for progressive disclosure efficiency.
pub const MAX_NAME_LENGTH: usize = 64;
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// Directories skipped when scanning for skills (mirrors `_EXCLUDED_SKILL_DIRS`).
pub const EXCLUDED_SKILL_DIRS: &[&str] = &[".git", ".github", ".hub", ".archive"];

/// Sandbox/remote backends that need requirements available inside the remote
/// environment as well (mirrors `_REMOTE_ENV_BACKENDS`).
pub const REMOTE_ENV_BACKENDS: &[&str] =
    &["docker", "singularity", "modal", "ssh", "daytona", "vercel_sandbox"];

/// Prompt-injection markers — shared by local-skill and plugin-skill paths.
pub const INJECTION_PATTERNS: &[&str] = &[
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "<system>",
    "]]>",
];

/// Skill readiness state surfaced to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillReadinessStatus {
    Available,
    SetupNeeded,
    Unsupported,
}

impl SkillReadinessStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SkillReadinessStatus::Available => "available",
            SkillReadinessStatus::SetupNeeded => "setup_needed",
            SkillReadinessStatus::Unsupported => "unsupported",
        }
    }
}

// ── env-var name validation ────────────────────────────────────────────────

/// Returns `true` when `name` is a valid shell env var identifier
/// (`^[A-Za-z_][A-Za-z0-9_]*$`).
fn is_valid_env_var_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ── module paths ───────────────────────────────────────────────────────────

/// `~/.hermes/skills` — single source of truth for skills.
pub fn skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}

// ── .env loading ───────────────────────────────────────────────────────────

/// Load profile-scoped environment variables from `HERMES_HOME/.env`.
///
/// Each non-comment `KEY=VALUE` line is parsed; surrounding quotes on the
/// value are stripped, mirroring the Python implementation.
pub fn load_env() -> std::collections::HashMap<String, String> {
    let env_path = get_hermes_home().join(".env");
    let mut env_vars = std::collections::HashMap::new();
    let Ok(content) = std::fs::read_to_string(&env_path) else {
        return env_vars;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let value = value.trim();
            let value = value.trim_matches(|c| c == '"' || c == '\'');
            env_vars.insert(key.trim().to_string(), value.to_string());
        }
    }
    env_vars
}

// ── pluggable secret-capture callback ──────────────────────────────────────

/// Result of an interactive secret-capture attempt.
#[derive(Debug, Clone, Default)]
pub struct SecretCaptureResult {
    pub success: bool,
    pub skipped: bool,
}

/// Signature: `(env_name, prompt, skill_name, help, required_for) -> result`.
pub type SecretCaptureCallback =
    Box<dyn Fn(&str, &str, &str, Option<&str>, Option<&str>) -> SecretCaptureResult + Send + Sync>;

static SECRET_CAPTURE_CALLBACK: Mutex<Option<SecretCaptureCallback>> = Mutex::new(None);

/// Install (or clear with `None`) the interactive secret-capture callback used
/// when a skill requires environment variables that are not yet set.
pub fn set_secret_capture_callback(callback: Option<SecretCaptureCallback>) {
    *SECRET_CAPTURE_CALLBACK.lock().unwrap() = callback;
}

fn has_secret_capture_callback() -> bool {
    SECRET_CAPTURE_CALLBACK.lock().unwrap().is_some()
}

// ── YAML helpers (frontmatter access) ──────────────────────────────────────

fn ym_get<'a>(map: &'a YamlValue, key: &str) -> Option<&'a YamlValue> {
    map.as_mapping()
        .and_then(|m| m.get(&YamlValue::String(key.to_string())))
}

/// Mirror Python `str(value)` over a YAML scalar for prompt/help text.
fn ym_to_string(value: &YamlValue) -> String {
    match value {
        YamlValue::Null => "None".to_string(),
        YamlValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        YamlValue::String(s) => s.clone(),
        YamlValue::Number(n) => n.to_string(),
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim_end()
            .to_string(),
    }
}

/// Python truthiness for a YAML value (used by `bool(item.get(...))`).
fn ym_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Null => false,
        YamlValue::Bool(b) => *b,
        YamlValue::String(s) => !s.is_empty(),
        YamlValue::Number(n) => {
            n.as_f64().map(|f| f != 0.0).unwrap_or(false)
                || n.as_i64().map(|i| i != 0).unwrap_or(false)
        }
        YamlValue::Sequence(s) => !s.is_empty(),
        YamlValue::Mapping(m) => !m.is_empty(),
        _ => true,
    }
}

fn ym_string_or_none(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Convert a YAML value to a JSON value for inclusion in the result payload.
fn yaml_to_json(value: &YamlValue) -> JsonValue {
    match value {
        YamlValue::Null => JsonValue::Null,
        YamlValue::Bool(b) => JsonValue::Bool(*b),
        YamlValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                json!(i)
            } else if let Some(u) = n.as_u64() {
                json!(u)
            } else if let Some(f) = n.as_f64() {
                json!(f)
            } else {
                JsonValue::Null
            }
        }
        YamlValue::String(s) => JsonValue::String(s.clone()),
        YamlValue::Sequence(seq) => JsonValue::Array(seq.iter().map(yaml_to_json).collect()),
        YamlValue::Mapping(map) => {
            let mut obj = Map::new();
            for (k, v) in map {
                let key = ym_to_string(k);
                obj.insert(key, yaml_to_json(v));
            }
            JsonValue::Object(obj)
        }
        YamlValue::Tagged(tagged) => yaml_to_json(&tagged.value),
    }
}

// ── prerequisite / setup metadata ──────────────────────────────────────────

fn normalize_prerequisite_values(value: Option<&YamlValue>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        YamlValue::Null => Vec::new(),
        YamlValue::String(s) => {
            if s.trim().is_empty() {
                Vec::new()
            } else {
                vec![s.clone()]
            }
        }
        YamlValue::Sequence(seq) => seq
            .iter()
            .map(ym_to_string)
            .filter(|s| !s.trim().is_empty())
            .collect(),
        other => {
            let s = ym_to_string(other);
            if s.trim().is_empty() {
                Vec::new()
            } else {
                vec![s]
            }
        }
    }
}

/// Returns `(env_vars, commands)` from the `prerequisites` block.
fn collect_prerequisite_values(frontmatter: &YamlValue) -> (Vec<String>, Vec<String>) {
    let Some(prereqs) = ym_get(frontmatter, "prerequisites") else {
        return (Vec::new(), Vec::new());
    };
    if prereqs.as_mapping().is_none() {
        return (Vec::new(), Vec::new());
    }
    (
        normalize_prerequisite_values(ym_get(prereqs, "env_vars")),
        normalize_prerequisite_values(ym_get(prereqs, "commands")),
    )
}

/// A normalized `setup.collect_secrets` entry.
#[derive(Debug, Clone)]
pub struct CollectSecret {
    pub env_var: String,
    pub prompt: String,
    pub secret: bool,
    pub provider_url: Option<String>,
}

/// Normalized `setup` block.
#[derive(Debug, Clone, Default)]
pub struct SetupMetadata {
    pub help: Option<String>,
    pub collect_secrets: Vec<CollectSecret>,
}

fn normalize_setup_metadata(frontmatter: &YamlValue) -> SetupMetadata {
    let Some(setup) = ym_get(frontmatter, "setup") else {
        return SetupMetadata::default();
    };
    if setup.as_mapping().is_none() {
        return SetupMetadata::default();
    }

    let normalized_help = ym_get(setup, "help")
        .and_then(ym_string_or_none)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let collect_secrets_raw: Vec<YamlValue> = match ym_get(setup, "collect_secrets") {
        Some(YamlValue::Mapping(_)) => vec![ym_get(setup, "collect_secrets").unwrap().clone()],
        Some(YamlValue::Sequence(seq)) => seq.clone(),
        _ => Vec::new(),
    };

    let mut collect_secrets = Vec::new();
    for item in &collect_secrets_raw {
        if item.as_mapping().is_none() {
            continue;
        }
        let env_var = ym_get(item, "env_var")
            .map(ym_to_string)
            .unwrap_or_default()
            .trim()
            .to_string();
        if env_var.is_empty() {
            continue;
        }
        let prompt = ym_get(item, "prompt")
            .map(ym_to_string)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("Enter value for {env_var}"))
            .trim()
            .to_string();
        let provider_url = ym_get(item, "provider_url")
            .or_else(|| ym_get(item, "url"))
            .map(ym_to_string)
            .unwrap_or_default()
            .trim()
            .to_string();
        let secret = ym_get(item, "secret").map(ym_truthy).unwrap_or(true);
        collect_secrets.push(CollectSecret {
            env_var,
            prompt,
            secret,
            provider_url: if provider_url.is_empty() {
                None
            } else {
                Some(provider_url)
            },
        });
    }

    SetupMetadata {
        help: normalized_help,
        collect_secrets,
    }
}

/// A normalized required-environment-variable entry.
#[derive(Debug, Clone)]
pub struct RequiredEnvVar {
    pub name: String,
    pub prompt: String,
    pub help: Option<String>,
    pub required_for: Option<String>,
    pub optional: bool,
}

impl RequiredEnvVar {
    fn to_json(&self) -> JsonValue {
        let mut obj = Map::new();
        obj.insert("name".into(), json!(self.name));
        obj.insert("prompt".into(), json!(self.prompt));
        if let Some(help) = &self.help {
            obj.insert("help".into(), json!(help));
        }
        if let Some(rf) = &self.required_for {
            obj.insert("required_for".into(), json!(rf));
        }
        if self.optional {
            obj.insert("optional".into(), json!(true));
        }
        JsonValue::Object(obj)
    }
}

/// Build the deduplicated list of required environment variables from
/// `required_environment_variables`, `setup.collect_secrets`, and legacy
/// `prerequisites.env_vars`.
pub fn get_required_environment_variables(
    frontmatter: &YamlValue,
    legacy_env_vars: Option<Vec<String>>,
) -> Vec<RequiredEnvVar> {
    let setup = normalize_setup_metadata(frontmatter);

    let required_raw: Vec<YamlValue> = match ym_get(frontmatter, "required_environment_variables") {
        Some(v @ YamlValue::Mapping(_)) => vec![v.clone()],
        Some(YamlValue::Sequence(seq)) => seq.clone(),
        _ => Vec::new(),
    };

    let mut required: Vec<RequiredEnvVar> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let mut append_required =
        |env_name: String,
         prompt: Option<String>,
         help: Option<String>,
         required_for: Option<String>,
         optional: bool,
         required: &mut Vec<RequiredEnvVar>,
         seen: &mut HashSet<String>| {
            let env_name = env_name.trim().to_string();
            if env_name.is_empty() || seen.contains(&env_name) {
                return;
            }
            if !is_valid_env_var_name(&env_name) {
                return;
            }
            let prompt = prompt
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("Enter value for {env_name}"))
                .trim()
                .to_string();
            let help = help.map(|h| h.trim().to_string()).filter(|h| !h.is_empty());
            let required_for = required_for
                .map(|r| r.trim().to_string())
                .filter(|r| !r.is_empty());
            seen.insert(env_name.clone());
            required.push(RequiredEnvVar {
                name: env_name,
                prompt,
                help,
                required_for,
                optional,
            });
        };

    for item in &required_raw {
        match item {
            YamlValue::String(s) => {
                append_required(
                    s.clone(),
                    None,
                    setup.help.clone(),
                    None,
                    false,
                    &mut required,
                    &mut seen,
                );
            }
            YamlValue::Mapping(_) => {
                let env_name = ym_get(item, "name")
                    .map(ym_to_string)
                    .filter(|s| !s.trim().is_empty())
                    .or_else(|| {
                        ym_get(item, "env_var")
                            .map(ym_to_string)
                            .filter(|s| !s.trim().is_empty())
                    })
                    .unwrap_or_default();
                let prompt = ym_get(item, "prompt").map(ym_to_string);
                // help precedence: help -> provider_url -> url -> setup.help
                let help = ym_get(item, "help")
                    .and_then(ym_string_or_none)
                    .filter(|s| !s.is_empty())
                    .or_else(|| {
                        ym_get(item, "provider_url")
                            .and_then(ym_string_or_none)
                            .filter(|s| !s.is_empty())
                    })
                    .or_else(|| {
                        ym_get(item, "url")
                            .and_then(ym_string_or_none)
                            .filter(|s| !s.is_empty())
                    })
                    .or_else(|| setup.help.clone());
                let required_for = ym_get(item, "required_for").and_then(ym_string_or_none);
                let optional = ym_get(item, "optional").map(ym_truthy).unwrap_or(false);
                append_required(
                    env_name,
                    prompt,
                    help,
                    required_for,
                    optional,
                    &mut required,
                    &mut seen,
                );
            }
            _ => {}
        }
    }

    for item in &setup.collect_secrets {
        let help = item.provider_url.clone().or_else(|| setup.help.clone());
        append_required(
            item.env_var.clone(),
            Some(item.prompt.clone()),
            help,
            None,
            false,
            &mut required,
            &mut seen,
        );
    }

    let legacy = match legacy_env_vars {
        Some(v) => v,
        None => collect_prerequisite_values(frontmatter).0,
    };
    for env_var in legacy {
        append_required(
            env_var,
            None,
            setup.help.clone(),
            None,
            false,
            &mut required,
            &mut seen,
        );
    }

    required
}

// ── gateway / backend probes ───────────────────────────────────────────────

/// Whether we are running inside a gateway surface (interactive secret entry
/// unavailable). Honours `HERMES_GATEWAY_SESSION`; the gateway session-context
/// lookup is not ported, so it degrades to the env-var check.
fn is_gateway_surface() -> bool {
    std::env::var("HERMES_GATEWAY_SESSION")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

fn terminal_backend_name() -> String {
    let raw = std::env::var("TERMINAL_ENV").unwrap_or_default();
    let trimmed = raw.trim().to_lowercase();
    if trimmed.is_empty() {
        "local".to_string()
    } else {
        trimmed
    }
}

fn gateway_setup_hint() -> String {
    format!(
        "Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to {}/.env manually.",
        display_hermes_home()
    )
}

fn is_env_var_persisted(
    var_name: &str,
    env_snapshot: &std::collections::HashMap<String, String>,
) -> bool {
    if let Some(v) = env_snapshot.get(var_name) {
        return !v.is_empty();
    }
    std::env::var(var_name).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Outcome of attempting to capture missing required env vars.
#[derive(Debug, Clone, Default)]
pub struct CaptureResult {
    pub missing_names: Vec<String>,
    pub setup_skipped: bool,
    pub gateway_setup_hint: Option<String>,
}

fn capture_required_environment_variables(
    skill_name: &str,
    missing_entries: &[RequiredEnvVar],
) -> CaptureResult {
    if missing_entries.is_empty() {
        return CaptureResult::default();
    }

    let missing_names: Vec<String> = missing_entries.iter().map(|e| e.name.clone()).collect();

    if is_gateway_surface() {
        return CaptureResult {
            missing_names,
            setup_skipped: false,
            gateway_setup_hint: Some(gateway_setup_hint()),
        };
    }

    if !has_secret_capture_callback() {
        return CaptureResult {
            missing_names,
            setup_skipped: false,
            gateway_setup_hint: None,
        };
    }

    let mut setup_skipped = false;
    let mut remaining_names: Vec<String> = Vec::new();

    let guard = SECRET_CAPTURE_CALLBACK.lock().unwrap();
    let callback = guard.as_ref();

    for entry in missing_entries {
        let result = match callback {
            Some(cb) => cb(
                &entry.name,
                &entry.prompt,
                skill_name,
                entry.help.as_deref(),
                entry.required_for.as_deref(),
            ),
            None => SecretCaptureResult {
                success: false,
                skipped: true,
            },
        };
        if result.success && !result.skipped {
            continue;
        }
        setup_skipped = true;
        remaining_names.push(entry.name.clone());
    }

    CaptureResult {
        missing_names: remaining_names,
        setup_skipped,
        gateway_setup_hint: None,
    }
}

fn remaining_required_environment_names(
    required_env_vars: &[RequiredEnvVar],
    capture_result: &CaptureResult,
    env_snapshot: &std::collections::HashMap<String, String>,
) -> Vec<String> {
    let missing_names: HashSet<&String> = capture_result.missing_names.iter().collect();
    let mut remaining = Vec::new();
    for entry in required_env_vars {
        if entry.optional {
            continue;
        }
        if missing_names.contains(&entry.name) || !is_env_var_persisted(&entry.name, env_snapshot) {
            remaining.push(entry.name.clone());
        }
    }
    remaining
}

fn build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: &[String],
    setup_help: Option<&str>,
) -> Option<String> {
    if readiness_status == SkillReadinessStatus::SetupNeeded {
        let missing_str = if missing.is_empty() {
            "required prerequisites".to_string()
        } else {
            missing.join(", ")
        };
        let note = format!("Setup needed before using this skill: missing {missing_str}.");
        if let Some(help) = setup_help {
            return Some(format!("{note} {help}"));
        }
        return Some(note);
    }
    None
}

/// Skills are always available — the directory is created on first use.
pub fn check_skills_requirements() -> bool {
    true
}

// ── tags parsing ───────────────────────────────────────────────────────────

/// Parse tags from a frontmatter value. Handles already-parsed lists,
/// bracket-wrapped strings, and comma-separated strings.
pub fn parse_tags(tags_value: Option<&YamlValue>) -> Vec<String> {
    let Some(tags_value) = tags_value else {
        return Vec::new();
    };
    match tags_value {
        YamlValue::Null => Vec::new(),
        YamlValue::Sequence(seq) => seq
            .iter()
            .filter(|t| ym_truthy(t))
            .map(|t| ym_to_string(t).trim().to_string())
            .collect(),
        YamlValue::String(s) if s.is_empty() => Vec::new(),
        other => {
            let mut s = ym_to_string(other).trim().to_string();
            if s.starts_with('[') && s.ends_with(']') {
                s = s[1..s.len() - 1].to_string();
            }
            s.split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| {
                    t.trim()
                        .trim_matches(|c| c == '"' || c == '\'')
                        .to_string()
                })
                .collect()
        }
    }
}

// ── category resolution ────────────────────────────────────────────────────

fn external_dirs() -> Vec<PathBuf> {
    let config_path = get_config_path();
    let hermes_home = get_hermes_home();
    let local = skills_dir();
    get_external_skills_dirs(&config_path, &hermes_home, &local)
}

fn all_skill_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let local = skills_dir();
    if local.exists() {
        dirs.push(local);
    }
    dirs.extend(external_dirs());
    dirs
}

/// Extract a category from the skill path: `<dir>/category/skill/SKILL.md` ->
/// `Some("category")`. Returns `None` for top-level skills.
fn get_category_from_path(skill_path: &Path) -> Option<String> {
    let mut dirs_to_check = vec![skills_dir()];
    dirs_to_check.extend(external_dirs());

    for skills_root in &dirs_to_check {
        if let Ok(rel) = skill_path.strip_prefix(skills_root) {
            let parts: Vec<_> = rel.components().collect();
            if parts.len() >= 3 {
                return Some(parts[0].as_os_str().to_string_lossy().to_string());
            }
        }
    }
    None
}

// ── disabled-skill resolution ──────────────────────────────────────────────

fn get_session_platform() -> String {
    // Gateway session context is not ported; only honour env precedence.
    String::new()
}

/// Load disabled skill names from config (platform-aware).
fn get_disabled_skill_names() -> BTreeSet<String> {
    let platform =
        std::env::var("HERMES_PLATFORM")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| {
                let p = get_session_platform();
                if p.is_empty() {
                    None
                } else {
                    Some(p)
                }
            });
    hermes_core::ag_skill_utils::get_disabled_skill_names(&get_config_path(), platform.as_deref())
}

fn is_skill_disabled(name: &str) -> bool {
    get_disabled_skill_names().contains(name)
}

// ── skill discovery ────────────────────────────────────────────────────────

/// Minimal skill listing entry.
#[derive(Debug, Clone)]
pub struct SkillListEntry {
    pub name: String,
    pub description: String,
    pub category: Option<String>,
}

/// Recursively find all skills in `~/.hermes/skills/` and external dirs.
///
/// When `skip_disabled` is `true`, all skills are returned regardless of
/// disabled state (used by the `hermes skills` config UI).
pub fn find_all_skills(skip_disabled: bool) -> Vec<SkillListEntry> {
    let mut skills: Vec<SkillListEntry> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    let disabled = if skip_disabled {
        BTreeSet::new()
    } else {
        get_disabled_skill_names()
    };

    let local = skills_dir();
    let mut dirs_to_scan: Vec<PathBuf> = Vec::new();
    if local.exists() {
        dirs_to_scan.push(local);
    }
    dirs_to_scan.extend(external_dirs());

    for scan_dir in &dirs_to_scan {
        for skill_md in iter_skill_index_files(scan_dir, "SKILL.md") {
            if skill_md
                .components()
                .any(|c| EXCLUDED_SKILL_DIRS.contains(&c.as_os_str().to_string_lossy().as_ref()))
            {
                continue;
            }
            let skill_dir = match skill_md.parent() {
                Some(p) => p.to_path_buf(),
                None => continue,
            };

            let Ok(raw) = std::fs::read_to_string(&skill_md) else {
                log::debug!("Failed to read skill file {}", skill_md.display());
                continue;
            };
            // Python reads only the first 4000 chars (by code points).
            let content: String = raw.chars().take(4000).collect();
            let (frontmatter, body) = parse_frontmatter(&content);

            if !skill_matches_platform(&frontmatter) {
                continue;
            }

            let dir_name = skill_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let name = ym_get(&frontmatter, "name")
                .and_then(ym_string_or_none)
                .unwrap_or(dir_name);
            let name: String = name.chars().take(MAX_NAME_LENGTH).collect();

            if seen_names.contains(&name) || disabled.contains(&name) {
                continue;
            }

            let mut description = ym_get(&frontmatter, "description")
                .and_then(ym_string_or_none)
                .unwrap_or_default();
            if description.is_empty() {
                for line in body.trim().split('\n') {
                    let line = line.trim();
                    if !line.is_empty() && !line.starts_with('#') {
                        description = line.to_string();
                        break;
                    }
                }
            }
            description = truncate_description(&description);

            let category = get_category_from_path(&skill_md);

            seen_names.insert(name.clone());
            skills.push(SkillListEntry {
                name,
                description,
                category,
            });
        }
    }

    skills
}

fn truncate_description(description: &str) -> String {
    let count = description.chars().count();
    if count > MAX_DESCRIPTION_LENGTH {
        let head: String = description
            .chars()
            .take(MAX_DESCRIPTION_LENGTH - 3)
            .collect();
        format!("{head}...")
    } else {
        description.to_string()
    }
}

/// Sort skills by `(category, name)`, mirroring `_sort_skills`.
pub fn sort_skills(mut skills: Vec<SkillListEntry>) -> Vec<SkillListEntry> {
    skills.sort_by(|a, b| {
        let ca = a.category.clone().unwrap_or_default();
        let cb = b.category.clone().unwrap_or_default();
        ca.cmp(&cb).then_with(|| a.name.cmp(&b.name))
    });
    skills
}

// ── skills_list ────────────────────────────────────────────────────────────

/// List all available skills (progressive-disclosure tier 1 — minimal
/// metadata). Returns a JSON string with `name`, `description`, `category`.
pub fn skills_list(category: Option<&str>) -> String {
    let result = std::panic::catch_unwind(|| skills_list_inner(category));
    match result {
        Ok(s) => s,
        Err(e) => {
            let msg = panic_message(&e);
            tool_error(msg, Some(json!({ "success": false })))
        }
    }
}

fn skills_list_inner(category: Option<&str>) -> String {
    let dir = skills_dir();
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
        return json!({
            "success": true,
            "skills": [],
            "categories": [],
            "message": format!(
                "No skills found. Skills directory created at {}/skills/",
                display_hermes_home()
            ),
        })
        .to_string();
    }

    let mut all_skills = find_all_skills(false);

    if all_skills.is_empty() {
        return json!({
            "success": true,
            "skills": [],
            "categories": [],
            "message": "No skills found in skills/ directory.",
        })
        .to_string();
    }

    if let Some(cat) = category {
        all_skills.retain(|s| s.category.as_deref() == Some(cat));
    }

    let all_skills = sort_skills(all_skills);

    let mut categories: BTreeSet<String> = BTreeSet::new();
    for s in &all_skills {
        if let Some(c) = &s.category {
            if !c.is_empty() {
                categories.insert(c.clone());
            }
        }
    }

    let skills_json: Vec<JsonValue> = all_skills
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "description": s.description,
                "category": s.category,
            })
        })
        .collect();

    json!({
        "success": true,
        "skills": skills_json,
        "categories": categories.into_iter().collect::<Vec<_>>(),
        "count": all_skills.len(),
        "hint": "Use skill_view(name) to see full content, tags, and linked files",
    })
    .to_string()
}

// ── linked-file gathering helpers ──────────────────────────────────────────

fn rel_str(path: &Path, base: &Path) -> Option<String> {
    path.strip_prefix(base)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Glob `dir` for direct children matching any of `exts` (e.g. "md", "py").
fn glob_dir_exts(dir: &Path, base: &Path, exts: &[&str], out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut found: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if exts.contains(&ext) {
                if let Some(rel) = rel_str(&path, base) {
                    found.push(rel);
                }
            }
        }
    }
    found.sort();
    out.extend(found);
}

/// Recursively walk `dir` collecting relative paths of files matching `exts`
/// (or all files when `exts` is `None`).
fn rglob_collect(dir: &Path, base: &Path, exts: Option<&[&str]>, out: &mut Vec<String>) {
    let mut found: Vec<String> = Vec::new();
    rglob_walk(dir, base, exts, &mut found);
    found.sort();
    out.extend(found);
}

fn rglob_walk(dir: &Path, base: &Path, exts: Option<&[&str]>, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            rglob_walk(&path, base, exts, out);
        } else if path.is_file() {
            match exts {
                None => {
                    if let Some(rel) = rel_str(&path, base) {
                        out.push(rel);
                    }
                }
                Some(exts) => {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if exts.contains(&ext) {
                            if let Some(rel) = rel_str(&path, base) {
                                out.push(rel);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── skill_view ─────────────────────────────────────────────────────────────

/// View the content of a skill or a specific file within a skill directory.
///
/// `name` may be a bare name, a categorized path (`category/skill`), or a
/// qualified `plugin:skill` form. The plugin registry is not ported, so
/// qualified names fall through to the local flat-tree scan (translating
/// `plugin:skill` to a categorized `plugin/skill` lookup).
pub fn skill_view(name: &str, file_path: Option<&str>) -> String {
    let result = std::panic::catch_unwind(|| skill_view_inner(name, file_path));
    match result {
        Ok(s) => s,
        Err(e) => {
            let msg = panic_message(&e);
            tool_error(msg, Some(json!({ "success": false })))
        }
    }
}

fn skill_view_inner(name: &str, file_path: Option<&str>) -> String {
    let mut local_category_name: Option<String> = None;

    // ── Qualified name dispatch ──
    if name.contains(':') {
        let (namespace, bare) = parse_qualified_name(name);
        if !is_valid_namespace(namespace.as_deref()) {
            return json!({
                "success": false,
                "error": format!(
                    "Invalid namespace '{}' in '{}'. Namespaces must match [a-zA-Z0-9_-]+.",
                    namespace.clone().unwrap_or_default(),
                    name
                ),
            })
            .to_string();
        }
        // Plugin registry unported: treat the qualified form as a categorized
        // local lookup (`namespace/bare`).
        if !bare.is_empty() {
            local_category_name = Some(format!("{}/{}", namespace.unwrap_or_default(), bare));
        }
    }

    let all_dirs = all_skill_dirs();
    if all_dirs.is_empty() {
        return json!({
            "success": false,
            "error": "Skills directory does not exist yet. It will be created on first install.",
        })
        .to_string();
    }

    let mut skill_dir: Option<PathBuf> = None;
    let mut skill_md: Option<PathBuf> = None;

    // Direct path / categorized path resolution.
    for search_dir in &all_dirs {
        let direct_path = search_dir.join(name);
        if direct_path.is_dir() && direct_path.join("SKILL.md").exists() {
            skill_dir = Some(direct_path.clone());
            skill_md = Some(direct_path.join("SKILL.md"));
            break;
        } else if with_md_suffix(&direct_path).exists() {
            skill_md = Some(with_md_suffix(&direct_path));
            break;
        }
        if let Some(cat_name) = &local_category_name {
            let categorized = search_dir.join(cat_name);
            if categorized.is_dir() && categorized.join("SKILL.md").exists() {
                skill_dir = Some(categorized.clone());
                skill_md = Some(categorized.join("SKILL.md"));
                break;
            } else if with_md_suffix(&categorized).exists() {
                skill_md = Some(with_md_suffix(&categorized));
                break;
            }
        }
    }

    // Search by directory name across all dirs.
    if skill_md.is_none() {
        'outer: for search_dir in &all_dirs {
            for found in iter_skill_index_files(search_dir, "SKILL.md") {
                if found.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy())
                    == Some(std::borrow::Cow::Borrowed(name))
                {
                    skill_dir = found.parent().map(|p| p.to_path_buf());
                    skill_md = Some(found);
                    break 'outer;
                }
            }
        }
    }

    // Legacy: flat `<name>.md` files (recursive).
    if skill_md.is_none() {
        let target = format!("{name}.md");
        'outer: for search_dir in &all_dirs {
            let mut found_files: Vec<PathBuf> = Vec::new();
            rglob_by_name(search_dir, &target, &mut found_files);
            found_files.sort();
            for found in found_files {
                if found.file_name().map(|n| n.to_string_lossy()) != Some(std::borrow::Cow::Borrowed("SKILL.md")) {
                    skill_md = Some(found);
                    break 'outer;
                }
            }
        }
    }

    let skill_md = match &skill_md {
        Some(p) if p.exists() => p.clone(),
        _ => {
            let available: Vec<String> = sort_skills(find_all_skills(false))
                .into_iter()
                .take(20)
                .map(|s| s.name)
                .collect();
            return json!({
                "success": false,
                "error": format!("Skill '{name}' not found."),
                "available_skills": available,
                "hint": "Use skills_list to see all available skills",
            })
            .to_string();
        }
    };

    // Read the file once.
    let content = match std::fs::read_to_string(&skill_md) {
        Ok(c) => c,
        Err(e) => {
            return json!({
                "success": false,
                "error": format!("Failed to read skill '{name}': {e}"),
            })
            .to_string();
        }
    };

    // Security: warn if outside trusted dirs.
    let mut trusted_dirs: Vec<PathBuf> = vec![resolve(&skills_dir())];
    for d in all_dirs.iter().skip(1) {
        trusted_dirs.push(resolve(d));
    }
    let resolved_md = resolve(&skill_md);
    let mut outside_skills_dir = true;
    for td in &trusted_dirs {
        if resolved_md.starts_with(td) {
            outside_skills_dir = false;
            break;
        }
    }

    let content_lower = content.to_lowercase();
    let injection_detected = INJECTION_PATTERNS.iter().any(|p| content_lower.contains(p));

    if outside_skills_dir || injection_detected {
        let mut warnings: Vec<String> = Vec::new();
        if outside_skills_dir {
            warnings.push(format!(
                "skill file is outside the trusted skills directory (~/.hermes/skills/): {}",
                skill_md.display()
            ));
        }
        if injection_detected {
            warnings.push(
                "skill content contains patterns that may indicate prompt injection".to_string(),
            );
        }
        log::warn!(
            "Skill security warning for '{}': {}",
            name,
            warnings.join("; ")
        );
    }

    let (parsed_frontmatter, _body) = parse_frontmatter(&content);

    if !skill_matches_platform(&parsed_frontmatter) {
        return json!({
            "success": false,
            "error": format!("Skill '{name}' is not supported on this platform."),
            "readiness_status": SkillReadinessStatus::Unsupported.as_str(),
        })
        .to_string();
    }

    // Disabled check.
    let parent_name = skill_md
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let resolved_name = ym_get(&parsed_frontmatter, "name")
        .and_then(ym_string_or_none)
        .unwrap_or_else(|| parent_name.clone());
    if is_skill_disabled(&resolved_name) {
        return json!({
            "success": false,
            "error": format!(
                "Skill '{resolved_name}' is disabled. Enable it with `hermes skills` or inspect the files directly on disk."
            ),
        })
        .to_string();
    }

    // ── Specific-file request ──
    if let (Some(fp), Some(sdir)) = (file_path, &skill_dir) {
        if has_traversal_component(fp) {
            return json!({
                "success": false,
                "error": "Path traversal ('..') is not allowed.",
                "hint": "Use a relative path within the skill directory",
            })
            .to_string();
        }

        let target_file = sdir.join(fp);
        if let Some(err) = validate_within_dir(&target_file, sdir) {
            return json!({
                "success": false,
                "error": err,
                "hint": "Use a relative path within the skill directory",
            })
            .to_string();
        }

        if !target_file.exists() {
            let available_files = list_available_files(sdir);
            return json!({
                "success": false,
                "error": format!("File '{fp}' not found in skill '{name}'."),
                "available_files": available_files,
                "hint": "Use one of the available file paths listed above",
            })
            .to_string();
        }

        // Try reading as UTF-8 text; binary -> info payload.
        match std::fs::read(&target_file) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => {
                    let suffix = file_suffix(&target_file);
                    return json!({
                        "success": true,
                        "name": name,
                        "file": fp,
                        "content": text,
                        "file_type": suffix,
                    })
                    .to_string();
                }
                Err(e) => {
                    let size = e.as_bytes().len();
                    let fname = target_file
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    return json!({
                        "success": true,
                        "name": name,
                        "file": fp,
                        "content": format!("[Binary file: {fname}, size: {size} bytes]"),
                        "is_binary": true,
                    })
                    .to_string();
                }
            },
            Err(e) => {
                return json!({
                    "success": false,
                    "error": format!("Failed to read skill '{name}': {e}"),
                })
                .to_string();
            }
        }
    }

    let frontmatter = parsed_frontmatter;

    // Gather linked files.
    let mut reference_files: Vec<String> = Vec::new();
    let mut template_files: Vec<String> = Vec::new();
    let mut asset_files: Vec<String> = Vec::new();
    let mut script_files: Vec<String> = Vec::new();

    if let Some(sdir) = &skill_dir {
        let references_dir = sdir.join("references");
        if references_dir.exists() {
            glob_dir_exts(&references_dir, sdir, &["md"], &mut reference_files);
        }

        let templates_dir = sdir.join("templates");
        if templates_dir.exists() {
            let exts = ["md", "py", "yaml", "yml", "json", "tex", "sh"];
            rglob_collect(&templates_dir, sdir, Some(&exts), &mut template_files);
        }

        let assets_dir = sdir.join("assets");
        if assets_dir.exists() {
            rglob_collect(&assets_dir, sdir, None, &mut asset_files);
        }

        let scripts_dir = sdir.join("scripts");
        if scripts_dir.exists() {
            let exts = ["py", "sh", "bash", "js", "ts", "rb"];
            glob_dir_exts(&scripts_dir, sdir, &exts, &mut script_files);
        }
    }

    // tags / related_skills: metadata.hermes.* first, fall back to top-level.
    let metadata = ym_get(&frontmatter, "metadata");
    let hermes_meta = metadata
        .filter(|m| m.as_mapping().is_some())
        .and_then(|m| ym_get(m, "hermes"))
        .filter(|m| m.as_mapping().is_some());

    let tags_value = hermes_meta
        .and_then(|m| ym_get(m, "tags"))
        .filter(|v| ym_truthy(v))
        .or_else(|| ym_get(&frontmatter, "tags"));
    let tags = parse_tags(tags_value);

    let related_value = hermes_meta
        .and_then(|m| ym_get(m, "related_skills"))
        .filter(|v| ym_truthy(v))
        .or_else(|| ym_get(&frontmatter, "related_skills"));
    let related_skills = parse_tags(related_value);

    let mut linked_files = Map::new();
    if !reference_files.is_empty() {
        linked_files.insert("references".into(), json!(reference_files));
    }
    if !template_files.is_empty() {
        linked_files.insert("templates".into(), json!(template_files));
    }
    if !asset_files.is_empty() {
        linked_files.insert("assets".into(), json!(asset_files));
    }
    if !script_files.is_empty() {
        linked_files.insert("scripts".into(), json!(script_files));
    }
    let has_linked_files = !linked_files.is_empty();

    // Relative path.
    let local = skills_dir();
    let rel_path = match skill_md.strip_prefix(&local) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => skill_md
            .parent()
            .and_then(|p| p.parent())
            .and_then(|gp| skill_md.strip_prefix(gp).ok())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| {
                skill_md
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            }),
    };

    let skill_name = ym_get(&frontmatter, "name")
        .and_then(ym_string_or_none)
        .unwrap_or_else(|| {
            if skill_dir.is_some() {
                parent_name.clone()
            } else {
                // skill_md.stem
                skill_md
                    .file_stem()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            }
        });

    let (legacy_env_vars, _) = collect_prerequisite_values(&frontmatter);
    let required_env_vars =
        get_required_environment_variables(&frontmatter, Some(legacy_env_vars));
    let backend = terminal_backend_name();
    let mut env_snapshot = load_env();
    let missing_required_env_vars: Vec<RequiredEnvVar> = required_env_vars
        .iter()
        .filter(|e| !e.optional && !is_env_var_persisted(&e.name, &env_snapshot))
        .cloned()
        .collect();
    let capture_result =
        capture_required_environment_variables(&skill_name, &missing_required_env_vars);
    if !missing_required_env_vars.is_empty() {
        env_snapshot = load_env();
    }
    let remaining_missing_required_envs =
        remaining_required_environment_names(&required_env_vars, &capture_result, &env_snapshot);
    let mut setup_needed = !remaining_missing_required_envs.is_empty();

    // Register available skill env vars for sandbox passthrough.
    let available_env_names: Vec<String> = required_env_vars
        .iter()
        .filter(|e| !remaining_missing_required_envs.contains(&e.name))
        .map(|e| e.name.clone())
        .collect();
    if !available_env_names.is_empty() {
        hermes_core::tool_env_passthrough::register_env_passthrough(available_env_names.iter());
    }

    // Required credential files: registration subsystem is not ported, so
    // detect missing files directly (mirrors register_credential_files'
    // missing-file reporting).
    let required_cred_files_raw: Vec<String> = match ym_get(&frontmatter, "required_credential_files") {
        Some(YamlValue::Sequence(seq)) => seq.iter().map(ym_to_string).collect(),
        _ => Vec::new(),
    };
    let mut missing_cred_files: Vec<String> = Vec::new();
    for raw in &required_cred_files_raw {
        let expanded = hermes_core::ag_skill_utils::expand_path(raw);
        if !Path::new(&expanded).exists() {
            missing_cred_files.push(raw.clone());
        }
    }
    if !missing_cred_files.is_empty() {
        setup_needed = true;
    }

    // Preprocessing subsystem unported — content is returned verbatim.
    let rendered_content = content;

    let description = ym_get(&frontmatter, "description")
        .and_then(ym_string_or_none)
        .unwrap_or_default();

    let mut result = Map::new();
    result.insert("success".into(), json!(true));
    result.insert("name".into(), json!(skill_name));
    result.insert("description".into(), json!(description));
    result.insert("tags".into(), json!(tags));
    result.insert("related_skills".into(), json!(related_skills));
    result.insert("content".into(), json!(rendered_content));
    result.insert("path".into(), json!(rel_path));
    result.insert(
        "skill_dir".into(),
        match &skill_dir {
            Some(d) => json!(d.to_string_lossy().to_string()),
            None => JsonValue::Null,
        },
    );
    result.insert(
        "linked_files".into(),
        if has_linked_files {
            JsonValue::Object(linked_files)
        } else {
            JsonValue::Null
        },
    );
    result.insert(
        "usage_hint".into(),
        if has_linked_files {
            json!("To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'")
        } else {
            JsonValue::Null
        },
    );
    result.insert(
        "required_environment_variables".into(),
        JsonValue::Array(required_env_vars.iter().map(|e| e.to_json()).collect()),
    );
    result.insert("required_commands".into(), json!([] as [JsonValue; 0]));
    result.insert(
        "missing_required_environment_variables".into(),
        json!(remaining_missing_required_envs),
    );
    result.insert("missing_credential_files".into(), json!(missing_cred_files));
    result.insert("missing_required_commands".into(), json!([] as [JsonValue; 0]));
    result.insert("setup_needed".into(), json!(setup_needed));
    result.insert("setup_skipped".into(), json!(capture_result.setup_skipped));
    result.insert(
        "readiness_status".into(),
        json!(if setup_needed {
            SkillReadinessStatus::SetupNeeded.as_str()
        } else {
            SkillReadinessStatus::Available.as_str()
        }),
    );

    let setup_help = required_env_vars.iter().find_map(|e| e.help.clone());
    if let Some(help) = &setup_help {
        result.insert("setup_help".into(), json!(help));
    }

    if let Some(hint) = &capture_result.gateway_setup_hint {
        result.insert("gateway_setup_hint".into(), json!(hint));
    }

    if setup_needed {
        let mut missing_items: Vec<String> = remaining_missing_required_envs
            .iter()
            .map(|n| format!("env ${n}"))
            .collect();
        missing_items.extend(missing_cred_files.iter().map(|p| format!("file {p}")));
        let mut setup_note = build_setup_note(
            SkillReadinessStatus::SetupNeeded,
            &missing_items,
            setup_help.as_deref(),
        );
        if REMOTE_ENV_BACKENDS.contains(&backend.as_str()) {
            if let Some(note) = setup_note.take() {
                setup_note = Some(format!(
                    "{note} {}-backed skills need these requirements available inside the remote environment as well.",
                    backend.to_uppercase()
                ));
            }
        }
        if let Some(note) = setup_note {
            result.insert("setup_note".into(), json!(note));
        }
    }

    // agentskills.io optional fields.
    if let Some(compat) = ym_get(&frontmatter, "compatibility") {
        if ym_truthy(compat) {
            result.insert("compatibility".into(), yaml_to_json(compat));
        }
    }
    if let Some(meta) = metadata {
        if meta.as_mapping().is_some() {
            result.insert("metadata".into(), yaml_to_json(meta));
        }
    }

    JsonValue::Object(result).to_string()
}

/// List available files in a skill directory, organized by type, mirroring the
/// Python `available_files` payload (empty categories removed).
fn list_available_files(skill_dir: &Path) -> JsonValue {
    let mut references: Vec<String> = Vec::new();
    let mut templates: Vec<String> = Vec::new();
    let mut assets: Vec<String> = Vec::new();
    let mut scripts: Vec<String> = Vec::new();
    let mut other: Vec<String> = Vec::new();

    let other_exts = ["md", "py", "yaml", "yml", "json", "tex", "sh"];

    let mut all_files: Vec<PathBuf> = Vec::new();
    rglob_walk_all(skill_dir, &mut all_files);
    all_files.sort();

    for f in &all_files {
        let fname = f
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if fname == "SKILL.md" {
            continue;
        }
        let Some(rel) = rel_str(f, skill_dir) else {
            continue;
        };
        if rel.starts_with("references/") {
            references.push(rel);
        } else if rel.starts_with("templates/") {
            templates.push(rel);
        } else if rel.starts_with("assets/") {
            assets.push(rel);
        } else if rel.starts_with("scripts/") {
            scripts.push(rel);
        } else if f
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| other_exts.contains(&e))
            .unwrap_or(false)
        {
            other.push(rel);
        }
    }

    let mut obj = Map::new();
    if !references.is_empty() {
        obj.insert("references".into(), json!(references));
    }
    if !templates.is_empty() {
        obj.insert("templates".into(), json!(templates));
    }
    if !assets.is_empty() {
        obj.insert("assets".into(), json!(assets));
    }
    if !scripts.is_empty() {
        obj.insert("scripts".into(), json!(scripts));
    }
    if !other.is_empty() {
        obj.insert("other".into(), json!(other));
    }
    JsonValue::Object(obj)
}

fn rglob_walk_all(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rglob_walk_all(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn rglob_by_name(dir: &Path, target: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rglob_by_name(&path, target, out);
        } else if path.is_file()
            && path.file_name().map(|n| n.to_string_lossy()) == Some(std::borrow::Cow::Borrowed(target))
        {
            out.push(path);
        }
    }
}

// ── small path utilities ───────────────────────────────────────────────────

/// Replace the file's extension with `.md` (Python `Path.with_suffix(".md")`).
fn with_md_suffix(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    p.set_extension("md");
    p
}

/// Python `Path.suffix` — the last extension including the leading dot, or "".
fn file_suffix(path: &Path) -> String {
    match path.extension() {
        Some(ext) => format!(".{}", ext.to_string_lossy()),
        None => String::new(),
    }
}

fn resolve(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_string()
    }
}

// ── tool schemas ───────────────────────────────────────────────────────────

/// JSON schema for the `skills_list` tool.
pub fn skills_list_schema() -> JsonValue {
    json!({
        "name": "skills_list",
        "description": "List available skills (name + description). Use skill_view(name) to load full content.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter to narrow results"
                }
            },
            "required": []
        }
    })
}

/// JSON schema for the `skill_view` tool.
pub fn skill_view_schema() -> JsonValue {
    json!({
        "name": "skill_view",
        "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans')."
                },
                "file_path": {
                    "type": "string",
                    "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content."
                }
            },
            "required": ["name"]
        }
    })
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests that mutate HERMES_HOME.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn yaml(s: &str) -> YamlValue {
        let (fm, _) = parse_frontmatter(s);
        fm
    }

    #[test]
    fn readiness_status_strings() {
        assert_eq!(SkillReadinessStatus::Available.as_str(), "available");
        assert_eq!(SkillReadinessStatus::SetupNeeded.as_str(), "setup_needed");
        assert_eq!(SkillReadinessStatus::Unsupported.as_str(), "unsupported");
    }

    #[test]
    fn env_var_name_validation() {
        assert!(is_valid_env_var_name("API_KEY"));
        assert!(is_valid_env_var_name("_x"));
        assert!(!is_valid_env_var_name("1ABC"));
        assert!(!is_valid_env_var_name("AB-CD"));
        assert!(!is_valid_env_var_name(""));
    }

    #[test]
    fn parse_tags_handles_list_bracket_and_csv() {
        let fm = yaml("---\ntags:\n  - alpha\n  - beta\n---\nbody");
        assert_eq!(parse_tags(ym_get(&fm, "tags")), vec!["alpha", "beta"]);

        let fm2 = yaml("---\ntags: \"[one, two]\"\n---\nbody");
        assert_eq!(parse_tags(ym_get(&fm2, "tags")), vec!["one", "two"]);

        let fm3 = yaml("---\ntags: \"x, y, z\"\n---\nbody");
        assert_eq!(parse_tags(ym_get(&fm3, "tags")), vec!["x", "y", "z"]);

        let fm4 = yaml("---\nname: s\n---\nbody");
        assert!(parse_tags(ym_get(&fm4, "tags")).is_empty());
    }

    #[test]
    fn required_env_vars_dedup_and_prompt() {
        let fm = yaml(
            "---\nname: s\nrequired_environment_variables:\n  - API_KEY\n  - name: TOKEN\n    prompt: Give token\n    help: https://x\n---\nbody",
        );
        let req = get_required_environment_variables(&fm, None);
        assert_eq!(req.len(), 2);
        assert_eq!(req[0].name, "API_KEY");
        assert_eq!(req[0].prompt, "Enter value for API_KEY");
        assert_eq!(req[1].name, "TOKEN");
        assert_eq!(req[1].prompt, "Give token");
        assert_eq!(req[1].help.as_deref(), Some("https://x"));
    }

    #[test]
    fn required_env_vars_from_collect_secrets_and_legacy() {
        let fm = yaml(
            "---\nname: s\nsetup:\n  help: see docs\n  collect_secrets:\n    - env_var: SECRET_ONE\n      provider_url: https://prov\nprerequisites:\n  env_vars:\n    - LEGACY_VAR\n---\nbody",
        );
        let req = get_required_environment_variables(&fm, None);
        let names: Vec<&str> = req.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"SECRET_ONE"));
        assert!(names.contains(&"LEGACY_VAR"));
        let secret = req.iter().find(|e| e.name == "SECRET_ONE").unwrap();
        assert_eq!(secret.help.as_deref(), Some("https://prov"));
    }

    #[test]
    fn invalid_env_var_names_skipped() {
        let fm = yaml(
            "---\nrequired_environment_variables:\n  - \"1BAD\"\n  - GOOD_VAR\n---\nbody",
        );
        let req = get_required_environment_variables(&fm, None);
        let names: Vec<&str> = req.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["GOOD_VAR"]);
    }

    #[test]
    fn build_setup_note_variants() {
        let note = build_setup_note(
            SkillReadinessStatus::SetupNeeded,
            &["env $API_KEY".to_string()],
            None,
        );
        assert_eq!(
            note.unwrap(),
            "Setup needed before using this skill: missing env $API_KEY."
        );

        let note2 = build_setup_note(
            SkillReadinessStatus::SetupNeeded,
            &[],
            Some("Add the key."),
        );
        assert_eq!(
            note2.unwrap(),
            "Setup needed before using this skill: missing required prerequisites. Add the key."
        );

        assert!(build_setup_note(SkillReadinessStatus::Available, &[], None).is_none());
    }

    #[test]
    fn truncate_description_long() {
        let long = "a".repeat(MAX_DESCRIPTION_LENGTH + 50);
        let t = truncate_description(&long);
        assert_eq!(t.chars().count(), MAX_DESCRIPTION_LENGTH);
        assert!(t.ends_with("..."));
    }

    #[test]
    fn with_md_suffix_and_file_suffix() {
        assert_eq!(with_md_suffix(Path::new("a/b/c")), PathBuf::from("a/b/c.md"));
        assert_eq!(file_suffix(Path::new("x/y.yaml")), ".yaml");
        assert_eq!(file_suffix(Path::new("x/y")), "");
    }

    #[test]
    fn skills_list_empty_dir_creates_and_reports() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!("hermes_skills_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let out = skills_list(None);
        let v: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(true));
        // Directory was created, so the "No skills found in skills/" path runs.
        assert!(v["message"].as_str().is_some());
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }

    #[test]
    fn skill_view_qualified_invalid_namespace() {
        // A namespace with an invalid char triggers the early error.
        let out = skill_view("bad ns:thing", None);
        let v: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("Invalid namespace"));
    }

    #[test]
    fn skill_view_not_found_lists_available() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!("hermes_skills_nf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("skills")).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let out = skill_view("does-not-exist", None);
        let v: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("not found"));
        assert!(v["available_skills"].is_array());
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }

    #[test]
    fn skill_view_reads_skill_and_linked_files() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!("hermes_skills_view_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let sdir = tmp.join("skills").join("demo");
        std::fs::create_dir_all(sdir.join("references")).unwrap();
        std::fs::write(
            sdir.join("SKILL.md"),
            "---\nname: demo\ndescription: A demo skill\ntags: [t1, t2]\n---\n# Demo\nHello.",
        )
        .unwrap();
        std::fs::write(sdir.join("references").join("api.md"), "ref body").unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
            std::env::remove_var("HERMES_PLATFORM");
            std::env::remove_var("HERMES_GATEWAY_SESSION");
            std::env::remove_var("TERMINAL_ENV");
        }

        let out = skill_view("demo", None);
        let v: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(true), "payload: {out}");
        assert_eq!(v["name"], json!("demo"));
        assert_eq!(v["description"], json!("A demo skill"));
        assert_eq!(v["tags"], json!(["t1", "t2"]));
        assert!(v["content"].as_str().unwrap().contains("Hello."));
        assert_eq!(v["linked_files"]["references"], json!(["references/api.md"]));
        assert_eq!(v["readiness_status"], json!("available"));

        // Now view the reference file.
        let out2 = skill_view("demo", Some("references/api.md"));
        let v2: JsonValue = serde_json::from_str(&out2).unwrap();
        assert_eq!(v2["success"], json!(true), "payload: {out2}");
        assert_eq!(v2["content"], json!("ref body"));
        assert_eq!(v2["file_type"], json!(".md"));

        // Traversal is rejected.
        let out3 = skill_view("demo", Some("../escape.md"));
        let v3: JsonValue = serde_json::from_str(&out3).unwrap();
        assert_eq!(v3["success"], json!(false));
        assert!(v3["error"].as_str().unwrap().contains("traversal"));

        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }

    #[test]
    fn skill_view_setup_needed_when_env_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!("hermes_skills_setup_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let sdir = tmp.join("skills").join("needs-env");
        std::fs::create_dir_all(&sdir).unwrap();
        std::fs::write(
            sdir.join("SKILL.md"),
            "---\nname: needs-env\ndescription: needs a key\nrequired_environment_variables:\n  - SOME_UNIQUE_TEST_KEY_XYZ\n---\nbody",
        )
        .unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
            std::env::remove_var("SOME_UNIQUE_TEST_KEY_XYZ");
            std::env::remove_var("HERMES_GATEWAY_SESSION");
            std::env::remove_var("TERMINAL_ENV");
        }
        set_secret_capture_callback(None);

        let out = skill_view("needs-env", None);
        let v: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(true), "payload: {out}");
        assert_eq!(v["setup_needed"], json!(true));
        assert_eq!(v["readiness_status"], json!("setup_needed"));
        assert_eq!(
            v["missing_required_environment_variables"],
            json!(["SOME_UNIQUE_TEST_KEY_XYZ"])
        );
        assert!(v["setup_note"].as_str().is_some());

        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }
}
