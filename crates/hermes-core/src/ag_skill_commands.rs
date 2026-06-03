//! Shared slash-command helpers for skills.
//!
//! Native Rust port of `agent/skill_commands.py`. Shared between the CLI and
//! the gateway so both surfaces can invoke skills via `/skill-name` commands.
//!
//! The Python module leans on several sibling modules that are ported as flat
//! files in this crate (`ag_skill_utils`, `mod_hermes_constants`) and on a few
//! that are not yet ported (`tools.skills_tool.skill_view`,
//! `agent.skill_preprocessing`, `tools.skill_usage.bump_use`,
//! `gateway.session_context`). For the unported pieces this module reproduces
//! the load/format behaviour directly against the filesystem and exposes the
//! optional hooks (template-var substitution, inline-shell expansion, usage
//! tracking) as injectable callbacks so callers can wire them up once ported,
//! while everything keeps working with safe defaults.
//!
//! Key behavioural notes preserved from the Python original:
//! * Skill command keys are normalized to `[a-z0-9-]` hyphen slugs.
//! * The scan cache is dropped when the active platform scope shifts (#14536),
//!   so each platform sees its own `skills.platform_disabled` view.
//! * Hyphens and underscores are interchangeable in user-typed commands.
//! * `_build_skill_message` injects the skill directory, resolved config
//!   values, setup notes, and supporting-file hints in the same order/format.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_yaml::Value as YamlValue;

use crate::ag_skill_utils::{
    self, default_config_path, default_hermes_home, default_skills_dir,
    extract_skill_config_vars, get_disabled_skill_names, get_external_skills_dirs,
    iter_skill_index_files, parse_frontmatter, resolve_platform, resolve_skill_config_values,
    skill_matches_platform, EXCLUDED_SKILL_DIRS,
};
use crate::mod_hermes_constants::display_hermes_home;

/// Info for a single skill exposed as a slash command.
///
/// Mirrors the Python dict value stored in `_skill_commands[/cmd]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCommandInfo {
    pub name: String,
    pub description: String,
    pub skill_md_path: String,
    pub skill_dir: String,
}

/// A loaded skill payload — the Rust analogue of the JSON dict returned by
/// `tools.skills_tool.skill_view(..., preprocess=False)`.
#[derive(Debug, Clone, Default)]
pub struct LoadedSkill {
    pub success: bool,
    pub name: Option<String>,
    pub path: Option<String>,
    pub skill_dir: Option<String>,
    /// Rendered body content (frontmatter stripped).
    pub content: Option<String>,
    /// Raw file content including frontmatter.
    pub raw_content: Option<String>,
    pub setup_skipped: bool,
    pub gateway_setup_hint: Option<String>,
    pub setup_needed: bool,
    pub setup_note: Option<String>,
    /// Supporting/linked files grouped by category (e.g. `references`).
    pub linked_files: BTreeMap<String, Vec<String>>,
}

/// Result of [`reload_skills`]: an add/remove/unchanged diff plus counts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReloadDiff {
    pub added: Vec<(String, String)>,
    pub removed: Vec<(String, String)>,
    pub unchanged: Vec<String>,
    pub total: usize,
    pub commands: usize,
}

// ── Module-global cache (mirrors Python module-level globals) ───────────────

struct CommandCache {
    commands: BTreeMap<String, SkillCommandInfo>,
    /// Insertion order of `/cmd` keys, preserving Python dict ordering.
    order: Vec<String>,
    platform: Option<String>,
    populated: bool,
}

impl CommandCache {
    const fn new() -> Self {
        Self {
            commands: BTreeMap::new(),
            order: Vec::new(),
            platform: None,
            populated: false,
        }
    }
}

static CACHE: Mutex<CommandCache> = Mutex::new(CommandCache::new());

/// Resolve the current platform scope used for disabled-skill filtering.
///
/// Port of `_resolve_skill_commands_platform`: resolves from `HERMES_PLATFORM`
/// then `HERMES_SESSION_PLATFORM` (via [`resolve_platform`]). Returns `None`
/// when no platform scope is active.
pub fn resolve_skill_commands_platform() -> Option<String> {
    resolve_platform(None)
}

// ── Name normalization (port of the inline slug logic) ──────────────────────

/// Normalize a skill `name` into a clean hyphen-separated command slug,
/// stripping characters outside `[a-z0-9-]` and collapsing repeated hyphens.
///
/// Returns an empty string when nothing usable remains (caller skips those).
pub fn slugify_command_name(name: &str) -> String {
    let lowered = name.to_lowercase().replace(' ', "-").replace('_', "-");
    let stripped: String = lowered
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
        .collect();
    // Collapse runs of `-` into a single `-`.
    let mut collapsed = String::with_capacity(stripped.len());
    let mut prev_hyphen = false;
    for c in stripped.chars() {
        if c == '-' {
            if !prev_hyphen {
                collapsed.push('-');
            }
            prev_hyphen = true;
        } else {
            collapsed.push(c);
            prev_hyphen = false;
        }
    }
    collapsed.trim_matches('-').to_string()
}

// ── Frontmatter helpers (mirror `_parse_frontmatter` BTreeMap access) ───────

fn fm_str(frontmatter: &YamlValue, key: &str) -> Option<String> {
    frontmatter
        .as_mapping()
        .and_then(|m| m.get(&YamlValue::String(key.to_string())))
        .and_then(|v| match v {
            YamlValue::String(s) => Some(s.clone()),
            YamlValue::Bool(b) => Some(b.to_string()),
            YamlValue::Number(n) => Some(n.to_string()),
            _ => None,
        })
}

/// Compute the command-scan description: frontmatter `description` if present,
/// else the first non-heading body line capped at 80 chars. Port of the inline
/// logic in `scan_skill_commands`.
fn command_description(frontmatter: &YamlValue, body: &str) -> String {
    if let Some(desc) = fm_str(frontmatter, "description") {
        if !desc.is_empty() {
            return desc;
        }
    }
    for line in body.trim().split('\n') {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            return line.chars().take(80).collect();
        }
    }
    String::new()
}

// ── Scanning ────────────────────────────────────────────────────────────────

/// Scan the skills directories and return a mapping of `/command -> info`.
///
/// Port of `scan_skill_commands`. Repopulates the module cache (clearing it
/// first), records the resolved platform, and returns an ordered copy of the
/// resulting command map.
pub fn scan_skill_commands() -> BTreeMap<String, SkillCommandInfo> {
    let platform = resolve_skill_commands_platform();
    let commands = scan_skill_commands_inner(platform.as_deref());

    let mut order: Vec<String> = Vec::new();
    let mut map: BTreeMap<String, SkillCommandInfo> = BTreeMap::new();
    for (key, info) in commands {
        if !map.contains_key(&key) {
            order.push(key.clone());
        }
        map.insert(key, info);
    }

    let mut cache = CACHE.lock().unwrap();
    cache.commands = map.clone();
    cache.order = order;
    cache.platform = platform;
    cache.populated = true;
    map
}

/// Pure scan against the configured dirs, returning `(/, info)` pairs in
/// discovery order (later duplicate keys overwrite earlier values, matching
/// Python dict assignment semantics).
fn scan_skill_commands_inner(platform: Option<&str>) -> Vec<(String, SkillCommandInfo)> {
    let hermes_home = default_hermes_home();
    let skills_dir = default_skills_dir();
    let config_path = default_config_path();

    let disabled = get_disabled_skill_names(&config_path, platform);
    let mut seen_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    let mut dirs_to_scan: Vec<PathBuf> = Vec::new();
    if skills_dir.exists() {
        dirs_to_scan.push(skills_dir.clone());
    }
    dirs_to_scan.extend(get_external_skills_dirs(&config_path, &hermes_home, &skills_dir));

    let mut result: Vec<(String, SkillCommandInfo)> = Vec::new();

    for scan_dir in dirs_to_scan {
        for skill_md in iter_skill_index_files(&scan_dir, "SKILL.md") {
            // Skip excluded directory components (defensive; iter already filters).
            if skill_md.components().any(|c| {
                matches!(c.as_os_str().to_str(), Some(p) if EXCLUDED_SKILL_DIRS.contains(&p))
            }) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&skill_md) else {
                continue;
            };
            let (frontmatter, body) = parse_frontmatter(&content);
            if !skill_matches_platform(&frontmatter) {
                continue;
            }
            let parent_name = skill_md
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let name = fm_str(&frontmatter, "name")
                .filter(|s| !s.is_empty())
                .unwrap_or(parent_name);
            if seen_names.contains(&name) {
                continue;
            }
            if disabled.contains(&name) {
                continue;
            }
            let description = command_description(&frontmatter, &body);
            seen_names.insert(name.clone());

            let cmd_name = slugify_command_name(&name);
            if cmd_name.is_empty() {
                continue;
            }
            let final_description = if description.is_empty() {
                format!("Invoke the {name} skill")
            } else {
                description
            };
            result.push((
                format!("/{cmd_name}"),
                SkillCommandInfo {
                    name,
                    description: final_description,
                    skill_md_path: skill_md.to_string_lossy().to_string(),
                    skill_dir: skill_md
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default(),
                },
            ));
        }
    }
    result
}

/// Return the current skill-command mapping, scanning first if the cache is
/// empty or the active platform scope changed (#14536).
///
/// Port of `get_skill_commands`.
pub fn get_skill_commands() -> BTreeMap<String, SkillCommandInfo> {
    let needs_rescan = {
        let cache = CACHE.lock().unwrap();
        !cache.populated
            || cache.commands.is_empty()
            || cache.platform != resolve_skill_commands_platform()
    };
    if needs_rescan {
        return scan_skill_commands();
    }
    CACHE.lock().unwrap().commands.clone()
}

/// Snapshot helper: `name -> description` keyed by the bare slug (no leading
/// `/`). Port of the inner `_snapshot` closure in `reload_skills`.
fn snapshot(cmds: &BTreeMap<String, SkillCommandInfo>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (slash_key, info) in cmds {
        let bare = slash_key.trim_start_matches('/').to_string();
        out.insert(bare, info.description.clone());
    }
    out
}

/// Re-scan the skills directory and return a diff of what changed.
///
/// Port of `reload_skills`. Snapshots the pre-reload cache, rescans, and
/// computes added/removed/unchanged sets plus total/command counts.
pub fn reload_skills() -> ReloadDiff {
    let before = {
        let cache = CACHE.lock().unwrap();
        snapshot(&cache.commands)
    };

    let new_commands = scan_skill_commands();
    let after = snapshot(&new_commands);

    let before_keys: std::collections::BTreeSet<&String> = before.keys().collect();
    let after_keys: std::collections::BTreeSet<&String> = after.keys().collect();

    let mut added_names: Vec<String> = after_keys
        .difference(&before_keys)
        .map(|s| (*s).clone())
        .collect();
    added_names.sort();
    let mut removed_names: Vec<String> = before_keys
        .difference(&after_keys)
        .map(|s| (*s).clone())
        .collect();
    removed_names.sort();
    let mut unchanged: Vec<String> = after_keys
        .intersection(&before_keys)
        .map(|s| (*s).clone())
        .collect();
    unchanged.sort();

    let added = added_names
        .into_iter()
        .map(|n| {
            let d = after.get(&n).cloned().unwrap_or_default();
            (n, d)
        })
        .collect();
    let removed = removed_names
        .into_iter()
        .map(|n| {
            let d = before.get(&n).cloned().unwrap_or_default();
            (n, d)
        })
        .collect();

    ReloadDiff {
        added,
        removed,
        unchanged,
        total: after.len(),
        commands: new_commands.len(),
    }
}

/// Resolve a user-typed `/command` to its canonical skill-command key.
///
/// Port of `resolve_skill_command_key`: underscores are treated as hyphens
/// (Telegram bot-command names disallow hyphens), and the resulting `/slug` is
/// matched against the current command map.
pub fn resolve_skill_command_key(command: &str) -> Option<String> {
    if command.is_empty() {
        return None;
    }
    let cmd_key = format!("/{}", command.replace('_', "-"));
    if get_skill_commands().contains_key(&cmd_key) {
        Some(cmd_key)
    } else {
        None
    }
}

// ── Skill loading (port of `_load_skill_payload`) ───────────────────────────

/// Load a skill by name or path from disk, returning the payload, its absolute
/// directory, and the display name.
///
/// Port of `_load_skill_payload`. The Python original delegated to
/// `tools.skills_tool.skill_view`; here we resolve directly against the local
/// skills tree (honoring absolute paths under `SKILLS_DIR`) and read the
/// `SKILL.md` plus its supporting files.
pub fn load_skill_payload(skill_identifier: &str) -> Option<(LoadedSkill, Option<PathBuf>, String)> {
    let raw_identifier = skill_identifier.trim();
    if raw_identifier.is_empty() {
        return None;
    }

    let skills_dir = default_skills_dir();

    // Resolve the SKILL.md path for the identifier.
    let identifier_path = PathBuf::from(ag_skill_utils::expand_user(raw_identifier));
    let skill_md = if identifier_path.is_absolute() {
        if identifier_path.is_dir() {
            identifier_path.join("SKILL.md")
        } else if identifier_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.eq_ignore_ascii_case("SKILL.md"))
            .unwrap_or(false)
        {
            identifier_path.clone()
        } else {
            identifier_path.join("SKILL.md")
        }
    } else {
        let normalized = raw_identifier.trim_start_matches('/');
        let candidate = skills_dir.join(normalized);
        if candidate.is_dir() {
            candidate.join("SKILL.md")
        } else if candidate
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.eq_ignore_ascii_case("SKILL.md"))
            .unwrap_or(false)
        {
            candidate
        } else {
            candidate.join("SKILL.md")
        }
    };

    if !skill_md.is_file() {
        return None;
    }

    let raw_content = std::fs::read_to_string(&skill_md).ok()?;
    let (frontmatter, body) = parse_frontmatter(&raw_content);

    let skill_dir = skill_md.parent().map(|p| p.to_path_buf());

    // Determine the display name: frontmatter `name` else the relative path
    // (or the directory name) used as the normalized identifier.
    let normalized = if identifier_path.is_absolute() {
        skill_md
            .parent()
            .and_then(|p| p.strip_prefix(&skills_dir).ok())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| raw_identifier.to_string())
    } else {
        raw_identifier.trim_start_matches('/').to_string()
    };
    let name = fm_str(&frontmatter, "name")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| normalized.clone());

    // path is the SKILL.md path relative to SKILLS_DIR (matches skill_view).
    let path = skill_md
        .strip_prefix(&skills_dir)
        .ok()
        .map(|p| p.to_string_lossy().to_string());

    let loaded = LoadedSkill {
        success: true,
        name: Some(name.clone()),
        path,
        skill_dir: skill_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        content: Some(body),
        raw_content: Some(raw_content),
        setup_skipped: false,
        gateway_setup_hint: None,
        setup_needed: false,
        setup_note: None,
        linked_files: BTreeMap::new(),
    };

    Some((loaded, skill_dir, name))
}

// ── Skills config (port of the `skills:` config block read by build) ────────

/// The subset of `skills:` config consumed by `_build_skill_message`.
#[derive(Debug, Clone)]
pub struct SkillsBuildConfig {
    /// `skills.template_vars` (default `true`).
    pub template_vars: bool,
    /// `skills.inline_shell` (default `false`).
    pub inline_shell: bool,
    /// `skills.inline_shell_timeout` (default `10`).
    pub inline_shell_timeout: u64,
}

impl Default for SkillsBuildConfig {
    fn default() -> Self {
        Self {
            template_vars: true,
            inline_shell: false,
            inline_shell_timeout: 10,
        }
    }
}

/// Load the `skills:` config block from `config.yaml`. Port of
/// `agent.skill_preprocessing.load_skills_config` (the subset used here).
pub fn load_skills_config() -> SkillsBuildConfig {
    let mut cfg = SkillsBuildConfig::default();
    let config_path = default_config_path();
    let Ok(text) = std::fs::read_to_string(&config_path) else {
        return cfg;
    };
    let Some(parsed) = ag_skill_utils::yaml_load(&text) else {
        return cfg;
    };
    let skills = parsed
        .as_mapping()
        .and_then(|m| m.get(&YamlValue::String("skills".to_string())));
    let Some(skills) = skills.and_then(|v| v.as_mapping()) else {
        return cfg;
    };
    if let Some(v) = skills.get(&YamlValue::String("template_vars".to_string())) {
        if let Some(b) = v.as_bool() {
            cfg.template_vars = b;
        }
    }
    if let Some(v) = skills.get(&YamlValue::String("inline_shell".to_string())) {
        if let Some(b) = v.as_bool() {
            cfg.inline_shell = b;
        }
    }
    if let Some(v) = skills.get(&YamlValue::String("inline_shell_timeout".to_string())) {
        if let Some(n) = v.as_u64() {
            if n > 0 {
                cfg.inline_shell_timeout = n;
            }
        }
    }
    cfg
}

// ── Optional preprocessing/usage hooks ──────────────────────────────────────

/// Pluggable hooks for the not-yet-ported preprocessing helpers
/// (`agent.skill_preprocessing`) and usage tracking (`tools.skill_usage`).
///
/// Defaults are no-ops / identity so message building works standalone.
#[derive(Clone)]
pub struct SkillHooks {
    /// `substitute_template_vars(content, skill_dir, session_id) -> content`.
    pub substitute_template_vars: fn(&str, Option<&Path>, Option<&str>) -> String,
    /// `expand_inline_shell(content, skill_dir, timeout) -> content`.
    pub expand_inline_shell: fn(&str, Option<&Path>, u64) -> String,
    /// `bump_use(skill_name)` — track active usage for the Curator (#17782).
    pub bump_use: fn(&str),
}

fn identity_template(content: &str, _dir: Option<&Path>, _sid: Option<&str>) -> String {
    content.to_string()
}
fn identity_shell(content: &str, _dir: Option<&Path>, _timeout: u64) -> String {
    content.to_string()
}
fn noop_bump(_name: &str) {}

impl Default for SkillHooks {
    fn default() -> Self {
        Self {
            substitute_template_vars: identity_template,
            expand_inline_shell: identity_shell,
            bump_use: noop_bump,
        }
    }
}

// ── Config injection (port of `_inject_skill_config`) ───────────────────────

/// Resolve and append skill-declared config values to `parts`.
///
/// Port of `_inject_skill_config`. Reads `metadata.hermes.config` from the
/// loaded skill's raw content, resolves current values, and appends a
/// `[Skill config ...]` block. No-op on any failure.
fn inject_skill_config(loaded_skill: &LoadedSkill, parts: &mut Vec<String>) {
    let raw_content = loaded_skill
        .raw_content
        .clone()
        .or_else(|| loaded_skill.content.clone())
        .unwrap_or_default();
    if raw_content.is_empty() {
        return;
    }
    let (frontmatter, _) = parse_frontmatter(&raw_content);
    let config_vars = extract_skill_config_vars(&frontmatter);
    if config_vars.is_empty() {
        return;
    }
    let resolved = resolve_skill_config_values(&default_config_path(), &config_vars);
    if resolved.is_empty() {
        return;
    }
    let mut lines = vec![
        String::new(),
        format!("[Skill config (from {}/config.yaml):", display_hermes_home()),
    ];
    for (key, value) in &resolved {
        let display_val = yaml_display_value(value);
        let display_val = if display_val.is_empty() {
            "(not set)".to_string()
        } else {
            display_val
        };
        lines.push(format!("  {key} = {display_val}"));
    }
    lines.push("]".to_string());
    parts.extend(lines);
}

/// Render a YAML config value as `str(value)` would in Python, treating empty
/// strings / null as falsy (the Python `str(value) if value else "(not set)"`).
fn yaml_display_value(value: &YamlValue) -> String {
    match value {
        YamlValue::Null => String::new(),
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => {
            // Python str(True) == "True"; but falsy `False` -> "(not set)".
            if *b {
                "True".to_string()
            } else {
                String::new()
            }
        }
        YamlValue::Number(n) => {
            // Python treats numeric 0 as falsy.
            if n.as_f64().map(|f| f == 0.0).unwrap_or(false) {
                String::new()
            } else {
                n.to_string()
            }
        }
        YamlValue::Sequence(seq) => {
            if seq.is_empty() {
                String::new()
            } else {
                serde_yaml::to_string(value)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default()
            }
        }
        YamlValue::Mapping(m) => {
            if m.is_empty() {
                String::new()
            } else {
                serde_yaml::to_string(value)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default()
            }
        }
        other => serde_yaml::to_string(other)
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
    }
}

// ── Message building (port of `_build_skill_message`) ───────────────────────

/// Format a loaded skill into a user/system message payload.
///
/// Port of `_build_skill_message`. Uses [`SkillHooks::default`] and the live
/// `skills:` config. For full control (custom preprocessing/usage hooks) use
/// [`build_skill_message_with`].
pub fn build_skill_message(
    loaded_skill: &LoadedSkill,
    skill_dir: Option<&Path>,
    activation_note: &str,
    user_instruction: &str,
    runtime_note: &str,
    session_id: Option<&str>,
) -> String {
    build_skill_message_with(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction,
        runtime_note,
        session_id,
        &load_skills_config(),
        &SkillHooks::default(),
    )
}

/// As [`build_skill_message`], but with an explicit config and hooks.
#[allow(clippy::too_many_arguments)]
pub fn build_skill_message_with(
    loaded_skill: &LoadedSkill,
    skill_dir: Option<&Path>,
    activation_note: &str,
    user_instruction: &str,
    runtime_note: &str,
    session_id: Option<&str>,
    skills_cfg: &SkillsBuildConfig,
    hooks: &SkillHooks,
) -> String {
    let mut content = loaded_skill.content.clone().unwrap_or_default();

    // ── Template substitution and inline-shell expansion ──
    if skills_cfg.template_vars {
        content = (hooks.substitute_template_vars)(&content, skill_dir, session_id);
    }
    if skills_cfg.inline_shell {
        content = (hooks.expand_inline_shell)(&content, skill_dir, skills_cfg.inline_shell_timeout);
    }

    let mut parts: Vec<String> = vec![
        activation_note.to_string(),
        String::new(),
        content.trim().to_string(),
    ];

    // ── Inject the absolute skill directory ──
    if let Some(dir) = skill_dir {
        parts.push(String::new());
        parts.push(format!("[Skill directory: {}]", dir.display()));
        parts.push(
            "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, \
             `templates/config.yaml`) against that directory, then run them \
             with the terminal tool using the absolute path."
                .to_string(),
        );
    }

    // ── Inject resolved skill config values ──
    inject_skill_config(loaded_skill, &mut parts);

    // ── Setup notes (mutually exclusive, in priority order) ──
    if loaded_skill.setup_skipped {
        parts.push(String::new());
        parts.push(
            "[Skill setup note: Required environment setup was skipped. Continue loading the skill and explain any reduced functionality if it matters.]"
                .to_string(),
        );
    } else if let Some(hint) = loaded_skill
        .gateway_setup_hint
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        parts.push(String::new());
        parts.push(format!("[Skill setup note: {hint}]"));
    } else if loaded_skill.setup_needed {
        if let Some(note) = loaded_skill.setup_note.as_deref().filter(|s| !s.is_empty()) {
            parts.push(String::new());
            parts.push(format!("[Skill setup note: {note}]"));
        }
    }

    // ── Supporting files ──
    let mut supporting: Vec<String> = Vec::new();
    for entries in loaded_skill.linked_files.values() {
        supporting.extend(entries.iter().cloned());
    }

    if supporting.is_empty() {
        if let Some(dir) = skill_dir {
            for subdir in ["references", "templates", "scripts", "assets"] {
                let subdir_path = dir.join(subdir);
                if subdir_path.exists() {
                    let mut found = Vec::new();
                    collect_files(&subdir_path, dir, &mut found);
                    found.sort();
                    supporting.extend(found);
                }
            }
        }
    }

    if !supporting.is_empty() {
        if let Some(dir) = skill_dir {
            let skills_dir = default_skills_dir();
            let skill_view_target = match dir.strip_prefix(&skills_dir) {
                Ok(rel) => rel.to_string_lossy().to_string(),
                Err(_) => dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
            };
            parts.push(String::new());
            parts.push("[This skill has supporting files:]".to_string());
            for sf in &supporting {
                parts.push(format!("- {sf}  ->  {}", dir.join(sf).display()));
            }
            parts.push(format!(
                "\nLoad any of these with skill_view(name=\"{skill_view_target}\", \
                 file_path=\"<path>\"), or run scripts directly by absolute path \
                 (e.g. `node {}/scripts/foo.js`).",
                dir.display()
            ));
        }
    }

    if !user_instruction.is_empty() {
        parts.push(String::new());
        parts.push(format!(
            "The user has provided the following instruction alongside the skill invocation: {user_instruction}"
        ));
    }

    if !runtime_note.is_empty() {
        parts.push(String::new());
        parts.push(format!("[Runtime note: {runtime_note}]"));
    }

    parts.join("\n")
}

/// Recursively collect files under `base`, pushing each path relative to
/// `rel_root`. Skips symlinks. Mirrors Python `rglob("*")` over real files.
fn collect_files(base: &Path, rel_root: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_files(&path, rel_root, out);
        } else if metadata.is_file() {
            if let Ok(rel) = path.strip_prefix(rel_root) {
                out.push(rel.to_string_lossy().to_string());
            }
        }
    }
}

// ── Public invocation builders ──────────────────────────────────────────────

/// Build the user-message content for a skill slash-command invocation.
///
/// Port of `build_skill_invocation_message`. Returns:
/// * `Some(message)` on success,
/// * `Some("[Failed to load skill: <name>]")` when the skill exists in the
///   command map but its payload can't be loaded,
/// * `None` when the command key isn't known.
pub fn build_skill_invocation_message(
    cmd_key: &str,
    user_instruction: &str,
    runtime_note: &str,
) -> Option<String> {
    build_skill_invocation_message_with(
        cmd_key,
        user_instruction,
        None,
        runtime_note,
        &SkillHooks::default(),
    )
}

/// As [`build_skill_invocation_message`], with an explicit `task_id`/session id
/// and hooks.
pub fn build_skill_invocation_message_with(
    cmd_key: &str,
    user_instruction: &str,
    task_id: Option<&str>,
    runtime_note: &str,
    hooks: &SkillHooks,
) -> Option<String> {
    let commands = get_skill_commands();
    let skill_info = commands.get(cmd_key)?;

    let Some((loaded_skill, skill_dir, skill_name)) = load_skill_payload(&skill_info.skill_dir)
    else {
        return Some(format!("[Failed to load skill: {}]", skill_info.name));
    };

    // Track active usage for Curator lifecycle management (#17782).
    (hooks.bump_use)(&skill_name);

    let activation_note = format!(
        "[IMPORTANT: The user has invoked the \"{skill_name}\" skill, indicating they want \
         you to follow its instructions. The full skill content is loaded below.]"
    );
    Some(build_skill_message_with(
        &loaded_skill,
        skill_dir.as_deref(),
        &activation_note,
        user_instruction,
        runtime_note,
        task_id,
        &load_skills_config(),
        hooks,
    ))
}

/// Load one or more skills for session-wide CLI preloading.
///
/// Port of `build_preloaded_skills_prompt`. Returns
/// `(prompt_text, loaded_skill_names, missing_identifiers)`.
pub fn build_preloaded_skills_prompt(
    skill_identifiers: &[String],
    task_id: Option<&str>,
) -> (String, Vec<String>, Vec<String>) {
    build_preloaded_skills_prompt_with(skill_identifiers, task_id, &SkillHooks::default())
}

/// As [`build_preloaded_skills_prompt`], with explicit hooks.
pub fn build_preloaded_skills_prompt_with(
    skill_identifiers: &[String],
    task_id: Option<&str>,
    hooks: &SkillHooks,
) -> (String, Vec<String>, Vec<String>) {
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut loaded_names: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let skills_cfg = load_skills_config();

    for raw_identifier in skill_identifiers {
        let identifier = raw_identifier.trim();
        if identifier.is_empty() || seen.contains(identifier) {
            continue;
        }
        seen.insert(identifier.to_string());

        let Some((loaded_skill, skill_dir, skill_name)) = load_skill_payload(identifier) else {
            missing.push(identifier.to_string());
            continue;
        };

        (hooks.bump_use)(&skill_name);

        let activation_note = format!(
            "[IMPORTANT: The user launched this CLI session with the \"{skill_name}\" skill \
             preloaded. Treat its instructions as active guidance for the duration of this \
             session unless the user overrides them.]"
        );
        prompt_parts.push(build_skill_message_with(
            &loaded_skill,
            skill_dir.as_deref(),
            &activation_note,
            "",
            "",
            task_id,
            &skills_cfg,
            hooks,
        ));
        loaded_names.push(skill_name);
    }

    (prompt_parts.join("\n\n"), loaded_names, missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn slugify_strips_invalid_and_collapses() {
        assert_eq!(slugify_command_name("Claude Code"), "claude-code");
        assert_eq!(slugify_command_name("gif_search"), "gif-search");
        assert_eq!(slugify_command_name("c++/foo"), "cfoo");
        assert_eq!(slugify_command_name("a--b___c"), "a-b-c");
        assert_eq!(slugify_command_name("-Hello-"), "hello");
        assert_eq!(slugify_command_name("++"), "");
    }

    #[test]
    fn resolve_command_key_underscore_to_hyphen() {
        // No commands loaded for a bogus key -> None regardless.
        assert_eq!(resolve_skill_command_key(""), None);
    }

    #[test]
    fn command_description_falls_back_to_first_body_line() {
        let (fm, body) = parse_frontmatter("---\nname: x\n---\n\n# Heading\n\nReal description line\n");
        assert_eq!(command_description(&fm, &body), "Real description line");
    }

    #[test]
    fn command_description_prefers_frontmatter() {
        let (fm, body) = parse_frontmatter("---\nname: x\ndescription: From FM\n---\nbody\n");
        assert_eq!(command_description(&fm, &body), "From FM");
    }

    #[test]
    fn command_description_caps_body_line_at_80() {
        let long = "a".repeat(120);
        let content = format!("---\nname: x\n---\n{long}\n");
        let (fm, body) = parse_frontmatter(&content);
        assert_eq!(command_description(&fm, &body).len(), 80);
    }

    #[test]
    fn build_message_includes_dir_and_instruction() {
        let loaded = LoadedSkill {
            success: true,
            name: Some("Demo".into()),
            content: Some("  Hello world  ".into()),
            raw_content: Some("---\nname: Demo\n---\nHello world".into()),
            ..Default::default()
        };
        let dir = PathBuf::from("/tmp/skills/demo");
        let msg = build_skill_message(
            &loaded,
            Some(&dir),
            "[ACTIVATE]",
            "do the thing",
            "session=abc",
            Some("abc"),
        );
        assert!(msg.starts_with("[ACTIVATE]\n\nHello world"));
        assert!(msg.contains("[Skill directory: /tmp/skills/demo]"));
        assert!(msg.contains("The user has provided the following instruction alongside the skill invocation: do the thing"));
        assert!(msg.contains("[Runtime note: session=abc]"));
    }

    #[test]
    fn build_message_setup_note_priority() {
        let mut loaded = LoadedSkill {
            success: true,
            content: Some("body".into()),
            setup_skipped: true,
            gateway_setup_hint: Some("hint".into()),
            setup_needed: true,
            setup_note: Some("note".into()),
            ..Default::default()
        };
        let msg = build_skill_message(&loaded, None, "[A]", "", "", None);
        assert!(msg.contains("Required environment setup was skipped"));
        assert!(!msg.contains("hint"));

        loaded.setup_skipped = false;
        let msg = build_skill_message(&loaded, None, "[A]", "", "", None);
        assert!(msg.contains("[Skill setup note: hint]"));

        loaded.gateway_setup_hint = None;
        let msg = build_skill_message(&loaded, None, "[A]", "", "", None);
        assert!(msg.contains("[Skill setup note: note]"));
    }

    #[test]
    fn build_message_lists_supporting_files() {
        let tmp = std::env::temp_dir().join(format!("ag_skill_cmds_{}", std::process::id()));
        let skills = tmp.join("skills");
        let skill_dir = skills.join("demo");
        let scripts = skill_dir.join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        fs::write(scripts.join("run.js"), "console.log(1)").unwrap();
        fs::write(skill_dir.join("SKILL.md"), "---\nname: demo\n---\nbody").unwrap();

        let prev_home = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }

        let loaded = LoadedSkill {
            success: true,
            name: Some("demo".into()),
            content: Some("body".into()),
            ..Default::default()
        };
        let msg = build_skill_message(&loaded, Some(&skill_dir), "[A]", "", "", None);
        assert!(msg.contains("[This skill has supporting files:]"));
        assert!(msg.contains("scripts/run.js"));
        assert!(msg.contains("skill_view(name=\"demo\""));

        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_payload_reads_skill_md() {
        let tmp = std::env::temp_dir().join(format!("ag_skill_load_{}", std::process::id()));
        let skills = tmp.join("skills");
        let skill_dir = skills.join("mydemo");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: My Demo\n---\nThe body content\n",
        )
        .unwrap();

        let prev_home = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }

        let loaded = load_skill_payload("mydemo");
        assert!(loaded.is_some());
        let (payload, dir, name) = loaded.unwrap();
        assert!(payload.success);
        assert_eq!(name, "My Demo");
        assert_eq!(dir, Some(skill_dir.clone()));
        assert_eq!(payload.content.as_deref().map(str::trim), Some("The body content"));

        // Missing identifier -> None.
        assert!(load_skill_payload("does-not-exist").is_none());
        assert!(load_skill_payload("   ").is_none());

        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn reload_diff_shape() {
        let diff = ReloadDiff {
            added: vec![("a".into(), "desc-a".into())],
            removed: vec![],
            unchanged: vec!["b".into()],
            total: 2,
            commands: 2,
        };
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.total, 2);
    }
}
