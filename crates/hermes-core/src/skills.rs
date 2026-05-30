use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map as JsonMap, Value, json};
use serde_yaml::Value as YamlValue;

use crate::tools::{ToolRuntime, tool_error, tool_result};

const MAX_NAME_LENGTH: usize = 64;
const MAX_DESCRIPTION_LENGTH: usize = 1024;
const EXCLUDED_SKILL_DIRS: &[&str] = &[".git", ".github", ".hub", ".archive"];
const ALLOWED_SUBDIRS: &[&str] = &["references", "templates", "scripts", "assets"];

#[derive(Debug, Clone)]
struct SkillEntry {
    name: String,
    description: String,
    category: Option<String>,
    skill_dir: PathBuf,
    skill_md: PathBuf,
}

pub fn skills_available() -> bool {
    true
}

pub fn skills_list_schema() -> Value {
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

pub fn skill_view_schema() -> Value {
    json!({
        "name": "skill_view",
        "description": "Load a skill's full content or access its linked files (references, templates, scripts, assets). First call returns SKILL.md content plus a linked_files dict showing available references/templates/scripts/assets. Call again with file_path to read a linked file.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name (use skills_list to see available skills)"
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional path to a linked file within the skill, such as references/api.md or templates/config.yaml"
                }
            },
            "required": ["name"]
        }
    })
}

pub fn skill_manage_schema() -> Value {
    json!({
        "name": "skill_manage",
        "description": "Manage local skills: create, patch, edit, delete, write_file, or remove_file. Skills are stored under HERMES_HOME/skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
                    "description": "The action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Skill name using lowercase letters, numbers, dots, hyphens, and underscores"
                },
                "content": {
                    "type": "string",
                    "description": "Full SKILL.md content for create or edit"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category used only for create"
                },
                "file_path": {
                    "type": "string",
                    "description": "Relative path within the skill directory. For write_file and remove_file this must live under references/, templates/, scripts/, or assets/. For patch it defaults to SKILL.md."
                },
                "file_content": {
                    "type": "string",
                    "description": "Content to write for write_file"
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace for patch"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text for patch. Use an empty string to delete matched text."
                },
                "replace_all": {
                    "type": "boolean",
                    "default": false,
                    "description": "Replace all matches instead of requiring a unique match"
                }
            },
            "required": ["action", "name"]
        }
    })
}

pub fn handle_skills_list(args: &Value, runtime: &ToolRuntime) -> String {
    let category = match optional_simple_string(args, "category") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let skills_root = runtime.hermes_home().join("skills");
    if !skills_root.exists() {
        if let Err(error) = fs::create_dir_all(&skills_root) {
            return tool_error(format!(
                "creating skills directory {} failed: {error}",
                skills_root.display()
            ));
        }
        return tool_result(json!({
            "success": true,
            "skills": [],
            "categories": [],
            "message": format!("No skills found. Skills directory created at {}/skills/", runtime.hermes_home().display()),
        }));
    }

    let mut skills = match discover_skills(&skills_root) {
        Ok(skills) => skills,
        Err(error) => return tool_error(error),
    };
    if let Some(category) = category.as_deref() {
        skills.retain(|skill| skill.category.as_deref() == Some(category));
    }
    if skills.is_empty() {
        return tool_result(json!({
            "success": true,
            "skills": [],
            "categories": [],
            "message": "No skills found in skills/ directory.",
        }));
    }

    skills.sort_by(|left, right| {
        let left_key = (
            left.category.as_deref().unwrap_or_default(),
            left.name.as_str(),
        );
        let right_key = (
            right.category.as_deref().unwrap_or_default(),
            right.name.as_str(),
        );
        left_key.cmp(&right_key)
    });

    let categories = skills
        .iter()
        .filter_map(|skill| skill.category.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let items = skills
        .into_iter()
        .map(|skill| {
            json!({
                "name": skill.name,
                "description": skill.description,
                "category": skill.category,
            })
        })
        .collect::<Vec<_>>();

    tool_result(json!({
        "success": true,
        "skills": items,
        "categories": categories,
        "count": items.len(),
        "hint": "Use skill_view(name) to see full content, tags, and linked files",
    }))
}

pub fn handle_skill_view(args: &Value, runtime: &ToolRuntime) -> String {
    let name = match required_non_empty_string(args, "name") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let file_path = match optional_non_empty_string(args, "file_path") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let skills_root = runtime.hermes_home().join("skills");
    if !skills_root.exists() {
        return tool_result(json!({
            "success": false,
            "error": "Skills directory does not exist yet. It will be created on first install.",
        }));
    }

    let skills = match discover_skills(&skills_root) {
        Ok(skills) => skills,
        Err(error) => return tool_error(error),
    };
    let Some(skill) = find_skill(&skills_root, &skills, &name) else {
        let available = skills
            .iter()
            .map(|skill| skill.name.clone())
            .take(20)
            .collect::<Vec<_>>();
        return tool_result(json!({
            "success": false,
            "error": format!("Skill '{name}' not found."),
            "available_skills": available,
            "hint": "Use skills_list to see all available skills",
        }));
    };

    if let Some(file_path) = file_path.as_deref() {
        let relative = match validated_relative_path(file_path) {
            Ok(value) => value,
            Err(error) => {
                return tool_result(json!({
                    "success": false,
                    "error": error,
                    "hint": "Use a relative path within the skill directory",
                }));
            }
        };
        let target = skill.skill_dir.join(&relative);
        if !target.exists() {
            return tool_result(json!({
                "success": false,
                "error": format!("File '{file_path}' not found in skill '{}'.", skill.name),
                "available_files": available_files(&skill.skill_dir),
                "hint": "Use one of the available file paths listed above",
            }));
        }
        if !target.is_file() {
            return tool_result(json!({
                "success": false,
                "error": format!("'{}' is not a file within skill '{}'.", file_path, skill.name),
            }));
        }
        let bytes = match fs::read(&target) {
            Ok(bytes) => bytes,
            Err(error) => {
                return tool_result(json!({
                    "success": false,
                    "error": format!("Failed to read '{}' from skill '{}': {error}", file_path, skill.name),
                }));
            }
        };
        let relative_text = relative.to_string_lossy().to_string();
        return match String::from_utf8(bytes) {
            Ok(content) => tool_result(json!({
                "success": true,
                "name": skill.name,
                "file": relative_text,
                "content": content,
                "file_type": target.extension().and_then(|value| value.to_str()).map(|value| format!(".{value}")).unwrap_or_default(),
            })),
            Err(error) => tool_result(json!({
                "success": true,
                "name": skill.name,
                "file": relative_text,
                "content": format!("[Binary file: {}, size: {} bytes]", target.file_name().and_then(|value| value.to_str()).unwrap_or("file"), error.as_bytes().len()),
                "is_binary": true,
            })),
        };
    }

    let content = match fs::read_to_string(&skill.skill_md) {
        Ok(content) => content,
        Err(error) => {
            return tool_result(json!({
                "success": false,
                "error": format!("Failed to read skill '{}': {error}", skill.name),
            }));
        }
    };
    let (frontmatter, _) = parse_frontmatter(&content);
    let description = resolved_description(&frontmatter, &content, &skill.name);
    let tags = parsed_tags(frontmatter_path(
        &frontmatter,
        &["metadata", "hermes", "tags"],
    ))
    .or_else(|| parsed_tags(frontmatter.get("tags")))
    .unwrap_or_default();
    let related_skills = parsed_tags(frontmatter_path(
        &frontmatter,
        &["metadata", "hermes", "related_skills"],
    ))
    .or_else(|| parsed_tags(frontmatter.get("related_skills")))
    .unwrap_or_default();
    let linked_files = linked_files(&skill.skill_dir);
    let path = skill
        .skill_md
        .strip_prefix(&skills_root)
        .unwrap_or(&skill.skill_md)
        .to_string_lossy()
        .to_string();

    let mut result = JsonMap::new();
    result.insert("success".to_string(), Value::Bool(true));
    result.insert("name".to_string(), Value::String(skill.name));
    result.insert("description".to_string(), Value::String(description));
    result.insert("tags".to_string(), json!(tags));
    result.insert("related_skills".to_string(), json!(related_skills));
    result.insert("content".to_string(), Value::String(content));
    result.insert("path".to_string(), Value::String(path));
    result.insert(
        "skill_dir".to_string(),
        Value::String(skill.skill_dir.display().to_string()),
    );
    result.insert(
        "readiness_status".to_string(),
        Value::String("available".to_string()),
    );
    result.insert("setup_needed".to_string(), Value::Bool(false));
    if let Some(linked_files) = linked_files {
        result.insert("linked_files".to_string(), Value::Object(linked_files));
        result.insert(
            "usage_hint".to_string(),
            Value::String("To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'".to_string()),
        );
    }
    tool_result(Value::Object(result))
}

pub fn build_skill_invocation_message(
    hermes_home: &Path,
    command_name: &str,
    user_instruction: &str,
    session_id: Option<&str>,
) -> Result<Option<(String, String)>, String> {
    let runtime = ToolRuntime::new(".").with_hermes_home(hermes_home);
    let skills_root = runtime.hermes_home().join("skills");
    if !skills_root.exists() {
        return Ok(None);
    }
    let skills = discover_skills(&skills_root)?;
    let normalized = normalize_skill_command_name(command_name);
    let Some(skill) = skills
        .into_iter()
        .find(|skill| normalize_skill_command_name(&skill.name) == normalized)
    else {
        return Ok(None);
    };

    let content = fs::read_to_string(&skill.skill_md)
        .map_err(|error| format!("Failed to read skill '{}': {error}", skill.name))?;
    let mut parts = vec![
        format!(
            "[IMPORTANT: The user has invoked the \"{}\" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]",
            skill.name
        ),
        String::new(),
        content.trim().to_string(),
    ];
    parts.push(String::new());
    parts.push(format!("[Skill directory: {}]", skill.skill_dir.display()));
    parts.push(
        "Resolve relative paths in this skill against that directory before loading files or running scripts."
            .to_string(),
    );
    if let Some(linked) = linked_files(&skill.skill_dir) {
        let mut supporting = Vec::new();
        for value in linked.values() {
            if let Some(items) = value.as_array() {
                for item in items {
                    if let Some(text) = item.as_str() {
                        supporting.push(text.to_string());
                    }
                }
            }
        }
        if !supporting.is_empty() {
            supporting.sort();
            parts.push(String::new());
            parts.push("[This skill has supporting files:]".to_string());
            for file in supporting {
                parts.push(format!(
                    "- {file}  ->  {}",
                    skill.skill_dir.join(&file).display()
                ));
            }
            parts.push(
                "Load supporting files with skill_view(name, file_path) or run scripts directly by absolute path."
                    .to_string(),
            );
        }
    }
    let trimmed_instruction = user_instruction.trim();
    if !trimmed_instruction.is_empty() {
        parts.push(String::new());
        parts.push(format!(
            "The user has provided the following instruction alongside the skill invocation: {trimmed_instruction}"
        ));
    }
    if let Some(session_id) = session_id.filter(|value| !value.trim().is_empty()) {
        parts.push(String::new());
        parts.push(format!("[Runtime note: session_id={session_id}]"));
    }
    Ok(Some((skill.name, parts.join("\n"))))
}

pub fn handle_skill_manage(args: &Value, runtime: &ToolRuntime) -> String {
    let action = match required_non_empty_string(args, "action") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let name = match required_non_empty_string(args, "name") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Err(error) = validate_skill_name(&name) {
        return tool_error(error);
    }

    match action.as_str() {
        "create" => handle_skill_create(args, runtime, &name),
        "edit" => handle_skill_edit(args, runtime, &name),
        "patch" => handle_skill_patch(args, runtime, &name),
        "delete" => handle_skill_delete(runtime, &name),
        "write_file" => handle_skill_write_file(args, runtime, &name),
        "remove_file" => handle_skill_remove_file(args, runtime, &name),
        _ => tool_result(json!({
            "success": false,
            "error": format!("Unknown action '{action}'. Use: create, patch, edit, delete, write_file, remove_file"),
        })),
    }
}

pub fn load_skill_prompt_content(hermes_home: &Path, name: &str) -> Result<String, String> {
    let runtime = ToolRuntime::new(".").with_hermes_home(hermes_home);
    let skill = resolved_skill(&runtime, name)?;
    fs::read_to_string(&skill.skill_md)
        .map_err(|error| format!("Failed to read skill '{}': {error}", name))
}

/// A skill exposed as a slash command: `/cmd` -> name + description.
#[derive(Debug, Clone)]
pub struct SkillCommand {
    pub command: String,
    pub name: String,
    pub description: String,
    pub skill_md_path: PathBuf,
    pub skill_dir: PathBuf,
}

/// Map `sys.platform`-style targets for the skill `platforms:` frontmatter.
/// Mirrors `agent.skill_utils.PLATFORM_MAP`.
fn platform_target(value: &str) -> String {
    match value {
        "macos" => "darwin".to_string(),
        "linux" => "linux".to_string(),
        "windows" => "win32".to_string(),
        other => other.to_string(),
    }
}

/// Current `sys.platform` value for the host OS (the subset Python reports).
fn current_sys_platform() -> &'static str {
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

/// Port of `agent.skill_utils.skill_matches_platform`: a skill with no
/// `platforms` field matches everything; otherwise the current `sys.platform`
/// must start with one of the mapped platform targets.
fn skill_matches_platform(frontmatter: &BTreeMap<String, YamlValue>) -> bool {
    let Some(platforms) = frontmatter.get("platforms") else {
        return true;
    };
    let entries: Vec<String> = match platforms {
        YamlValue::Null => return true,
        YamlValue::Sequence(seq) => {
            if seq.is_empty() {
                return true;
            }
            seq.iter().map(yaml_scalar_to_string).collect()
        }
        YamlValue::String(s) if s.is_empty() => return true,
        other => vec![yaml_scalar_to_string(other)],
    };
    let current = current_sys_platform();
    entries.iter().any(|platform| {
        let normalized = platform.to_lowercase();
        let normalized = normalized.trim();
        current.starts_with(&platform_target(normalized))
    })
}

fn yaml_scalar_to_string(value: &YamlValue) -> String {
    match value {
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => b.to_string(),
        YamlValue::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

/// Read `config.yaml` under `hermes_home` as a YAML mapping (best-effort).
fn read_config_mapping(hermes_home: &Path) -> Option<serde_yaml::Mapping> {
    let path = hermes_home.join("config.yaml");
    let text = fs::read_to_string(path).ok()?;
    match serde_yaml::from_str::<YamlValue>(&text).ok()? {
        YamlValue::Mapping(mapping) => Some(mapping),
        _ => None,
    }
}

fn mapping_get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn normalize_string_set(value: Option<&YamlValue>) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    match value {
        Some(YamlValue::Sequence(seq)) => {
            for item in seq {
                let s = yaml_scalar_to_string(item);
                let s = s.trim();
                if !s.is_empty() {
                    set.insert(s.to_string());
                }
            }
        }
        Some(YamlValue::String(s)) => {
            let s = s.trim();
            if !s.is_empty() {
                set.insert(s.to_string());
            }
        }
        _ => {}
    }
    set
}

/// Port of `agent.skill_utils.get_disabled_skill_names`. Reads
/// `skills.disabled` (or `skills.platform_disabled.<platform>` when a platform
/// is active) from `config.yaml`. `platform` resolves from the explicit arg
/// then `HERMES_PLATFORM` then `HERMES_SESSION_PLATFORM`.
fn disabled_skill_names(hermes_home: &Path, platform: Option<&str>) -> BTreeSet<String> {
    let Some(config) = read_config_mapping(hermes_home) else {
        return BTreeSet::new();
    };
    let Some(YamlValue::Mapping(skills_cfg)) = mapping_get(&config, "skills") else {
        return BTreeSet::new();
    };
    let resolved_platform = platform
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HERMES_PLATFORM").ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            std::env::var("HERMES_SESSION_PLATFORM")
                .ok()
                .filter(|s| !s.is_empty())
        });
    if let Some(platform) = resolved_platform {
        if let Some(YamlValue::Mapping(platform_disabled)) =
            mapping_get(skills_cfg, "platform_disabled")
        {
            if let Some(entry) = platform_disabled.get(YamlValue::String(platform)) {
                return normalize_string_set(Some(entry));
            }
        }
    }
    normalize_string_set(mapping_get(skills_cfg, "disabled"))
}

/// Port of `agent.skill_utils.get_external_skills_dirs`. Reads
/// `skills.external_dirs` from `config.yaml`, expands `~`/`$VAR`, resolves
/// relative entries against `hermes_home`, drops the local skills dir,
/// dedupes, and keeps only existing directories.
fn external_skills_dirs(hermes_home: &Path, local_skills: &Path) -> Vec<PathBuf> {
    let Some(config) = read_config_mapping(hermes_home) else {
        return Vec::new();
    };
    let Some(YamlValue::Mapping(skills_cfg)) = mapping_get(&config, "skills") else {
        return Vec::new();
    };
    let raw = match mapping_get(skills_cfg, "external_dirs") {
        Some(YamlValue::Sequence(seq)) => seq
            .iter()
            .map(yaml_scalar_to_string)
            .collect::<Vec<_>>(),
        Some(YamlValue::String(s)) => vec![s.clone()],
        _ => return Vec::new(),
    };
    let local = local_skills
        .canonicalize()
        .unwrap_or_else(|_| local_skills.to_path_buf());
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let mut result = Vec::new();
    for entry in raw {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let expanded = expand_user_and_vars(entry);
        let path = Path::new(&expanded);
        let resolved = if path.is_absolute() {
            path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
        } else {
            let joined = hermes_home.join(path);
            joined.canonicalize().unwrap_or(joined)
        };
        if resolved == local || seen.contains(&resolved) {
            continue;
        }
        if resolved.is_dir() {
            seen.insert(resolved.clone());
            result.push(resolved);
        }
    }
    result
}

/// Expand a leading `~` and `$VAR`/`${VAR}` references using the environment.
fn expand_user_and_vars(input: &str) -> String {
    let mut expanded = input.to_string();
    if expanded == "~" || expanded.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            expanded = expanded.replacen('~', &home.to_string_lossy(), 1);
        }
    }
    expand_env_vars(&expanded)
}

fn expand_env_vars(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'{' {
                if let Some(end) = input[i + 2..].find('}') {
                    let name = &input[i + 2..i + 2 + end];
                    out.push_str(&std::env::var(name).unwrap_or_default());
                    i = i + 2 + end + 1;
                    continue;
                }
            } else {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
                {
                    j += 1;
                }
                if j > start {
                    let name = &input[start..j];
                    out.push_str(&std::env::var(name).unwrap_or_default());
                    i = j;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Description fallback used by skill-command scanning: the frontmatter
/// `description` (untruncated) or the first non-heading body line capped at 80
/// chars. Mirrors the inline logic in `agent.skill_commands.scan_skill_commands`
/// (distinct from `resolved_description`, which truncates differently).
fn skill_command_description(
    frontmatter: &BTreeMap<String, YamlValue>,
    body: &str,
) -> String {
    if let Some(description) = frontmatter
        .get("description")
        .and_then(YamlValue::as_str)
        .filter(|value| !value.is_empty())
    {
        return description.to_string();
    }
    for line in body.trim().split('\n') {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            return line.chars().take(80).collect();
        }
    }
    String::new()
}

/// Scan `~/.hermes/skills/` (and configured external dirs) for skills exposed
/// as slash commands. Faithful port of `agent.skill_commands.scan_skill_commands`:
/// platform-incompatible and disabled skills are skipped, the first occurrence
/// of each skill name wins, and command slugs are normalized to `[a-z0-9-]`.
pub fn scan_skill_commands(hermes_home: &Path, platform: Option<&str>) -> Vec<SkillCommand> {
    let local_skills = hermes_home.join("skills");
    let disabled = disabled_skill_names(hermes_home, platform);

    let mut dirs_to_scan = Vec::new();
    if local_skills.exists() {
        dirs_to_scan.push(local_skills.clone());
    }
    dirs_to_scan.extend(external_skills_dirs(hermes_home, &local_skills));

    let mut seen_names: BTreeSet<String> = BTreeSet::new();
    // Preserve first-insert order while allowing later duplicate /cmd keys to
    // overwrite the stored value, matching Python's dict semantics.
    let mut order: Vec<String> = Vec::new();
    let mut by_command: BTreeMap<String, SkillCommand> = BTreeMap::new();

    for scan_dir in dirs_to_scan {
        let mut skill_files = Vec::new();
        if collect_skill_files(&scan_dir, &mut skill_files).is_err() {
            continue;
        }
        // collect_skill_files yields filesystem order; sort by path relative to
        // the scan dir to match Python's iter_skill_index_files ordering.
        skill_files.sort_by(|a, b| {
            let ra = a.strip_prefix(&scan_dir).unwrap_or(a);
            let rb = b.strip_prefix(&scan_dir).unwrap_or(b);
            ra.cmp(rb)
        });
        for skill_md in skill_files {
            if skill_md
                .components()
                .any(|c| matches!(c.as_os_str().to_str(), Some(p) if EXCLUDED_SKILL_DIRS.contains(&p)))
            {
                continue;
            }
            let Ok(content) = fs::read_to_string(&skill_md) else {
                continue;
            };
            let (frontmatter, body) = parse_frontmatter(&content);
            if !skill_matches_platform(&frontmatter) {
                continue;
            }
            let name = frontmatter
                .get("name")
                .and_then(YamlValue::as_str)
                .map(str::to_string)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| {
                    skill_md
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("skill")
                        .to_string()
                });
            if seen_names.contains(&name) {
                continue;
            }
            if disabled.contains(&name) {
                continue;
            }
            let description = skill_command_description(&frontmatter, &body);
            seen_names.insert(name.clone());
            let cmd_name = normalize_skill_command_name(&name);
            if cmd_name.is_empty() {
                continue;
            }
            let command = format!("/{cmd_name}");
            let description = if description.is_empty() {
                format!("Invoke the {name} skill")
            } else {
                description
            };
            let skill_dir = skill_md
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| skill_md.clone());
            if !by_command.contains_key(&command) {
                order.push(command.clone());
            }
            by_command.insert(
                command.clone(),
                SkillCommand {
                    command,
                    name,
                    description,
                    skill_md_path: skill_md,
                    skill_dir,
                },
            );
        }
    }

    order
        .into_iter()
        .filter_map(|key| by_command.remove(&key))
        .collect()
}

fn discover_skills(skills_root: &Path) -> Result<Vec<SkillEntry>, String> {
    let mut skill_files = Vec::new();
    collect_skill_files(skills_root, &mut skill_files)?;
    let mut skills = Vec::new();
    let mut seen = BTreeSet::new();
    for skill_md in skill_files {
        let content = match fs::read_to_string(&skill_md) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let (frontmatter, _) = parse_frontmatter(&content);
        let Some(skill_dir) = skill_md.parent().map(Path::to_path_buf) else {
            continue;
        };
        let name = frontmatter
            .get("name")
            .and_then(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| truncate(value, MAX_NAME_LENGTH))
            .unwrap_or_else(|| {
                truncate(
                    &skill_dir
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("skill"),
                    MAX_NAME_LENGTH,
                )
            });
        if !seen.insert(name.clone()) {
            continue;
        }
        skills.push(SkillEntry {
            description: resolved_description(&frontmatter, &content, &name),
            category: category_from_path(skills_root, &skill_md),
            skill_dir,
            skill_md,
            name,
        });
    }
    Ok(skills)
}

fn handle_skill_create(args: &Value, runtime: &ToolRuntime, name: &str) -> String {
    let content = match required_non_empty_string(args, "content") {
        Ok(value) => value,
        Err(_) => {
            return tool_result(json!({
                "success": false,
                "error": "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).",
            }));
        }
    };
    if let Err(error) = validate_skill_frontmatter(&content, name) {
        return tool_error(error);
    }
    let category = match optional_simple_string(args, "category") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let skills_root = runtime.hermes_home().join("skills");
    if let Err(error) = fs::create_dir_all(&skills_root) {
        return tool_error(format!(
            "creating skills directory {} failed: {error}",
            skills_root.display()
        ));
    }
    let skill_dir = category
        .as_deref()
        .map(|category| skills_root.join(category).join(name))
        .unwrap_or_else(|| skills_root.join(name));
    if skill_dir.exists() {
        return tool_result(json!({
            "success": false,
            "error": format!("Skill '{name}' already exists."),
        }));
    }
    for dir in [
        skill_dir.as_path(),
        &skill_dir.join("references"),
        &skill_dir.join("templates"),
        &skill_dir.join("scripts"),
        &skill_dir.join("assets"),
    ] {
        if let Err(error) = fs::create_dir_all(dir) {
            return tool_error(format!("creating {} failed: {error}", dir.display()));
        }
    }
    let skill_md = skill_dir.join("SKILL.md");
    if let Err(error) = fs::write(&skill_md, content) {
        return tool_error(format!("writing {} failed: {error}", skill_md.display()));
    }
    tool_result(json!({
        "success": true,
        "action": "create",
        "name": name,
        "path": skill_md.display().to_string(),
    }))
}

fn handle_skill_edit(args: &Value, runtime: &ToolRuntime, name: &str) -> String {
    let content = match required_non_empty_string(args, "content") {
        Ok(value) => value,
        Err(_) => {
            return tool_result(json!({
                "success": false,
                "error": "content is required for 'edit'. Provide the full updated SKILL.md text.",
            }));
        }
    };
    if let Err(error) = validate_skill_frontmatter(&content, name) {
        return tool_error(error);
    }
    let skill = match resolved_skill(runtime, name) {
        Ok(skill) => skill,
        Err(error) => return tool_result(json!({"success": false, "error": error})),
    };
    if let Err(error) = fs::write(&skill.skill_md, content) {
        return tool_error(format!(
            "writing {} failed: {error}",
            skill.skill_md.display()
        ));
    }
    tool_result(json!({
        "success": true,
        "action": "edit",
        "name": name,
        "path": skill.skill_md.display().to_string(),
    }))
}

fn handle_skill_patch(args: &Value, runtime: &ToolRuntime, name: &str) -> String {
    let old_string = match args.get("old_string") {
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        _ => {
            return tool_result(json!({
                "success": false,
                "error": "old_string is required for 'patch'. Provide the text to find.",
            }));
        }
    };
    let new_string = match args.get("new_string") {
        Some(Value::String(value)) => value.clone(),
        Some(_) => return tool_error("new_string must be a string"),
        None => {
            return tool_result(json!({
                "success": false,
                "error": "new_string is required for 'patch'. Use empty string to delete matched text.",
            }));
        }
    };
    let replace_all = args
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let skill = match resolved_skill(runtime, name) {
        Ok(skill) => skill,
        Err(error) => return tool_result(json!({"success": false, "error": error})),
    };
    let target = match optional_non_empty_string(args, "file_path") {
        Ok(Some(path)) => match validated_relative_path(&path) {
            Ok(relative) => skill.skill_dir.join(relative),
            Err(error) => return tool_error(error),
        },
        Ok(None) => skill.skill_md.clone(),
        Err(error) => return tool_error(error),
    };
    if !target.exists() || !target.is_file() {
        return tool_result(json!({
            "success": false,
            "error": format!("Target file '{}' does not exist for skill '{}'.", target.display(), name),
        }));
    }
    let original = match fs::read_to_string(&target) {
        Ok(content) => content,
        Err(error) => return tool_error(format!("reading {} failed: {error}", target.display())),
    };
    let matches = original.matches(&old_string).count();
    if matches == 0 {
        return tool_result(json!({
            "success": false,
            "error": "old_string was not found in the target file.",
        }));
    }
    if !replace_all && matches != 1 {
        return tool_result(json!({
            "success": false,
            "error": format!("old_string matched {matches} locations. Set replace_all=true or provide more context."),
        }));
    }
    let updated = if replace_all {
        original.replace(&old_string, &new_string)
    } else {
        original.replacen(&old_string, &new_string, 1)
    };
    if target == skill.skill_md
        && let Err(error) = validate_skill_frontmatter(&updated, name)
    {
        return tool_error(error);
    }
    if let Err(error) = fs::write(&target, updated) {
        return tool_error(format!("writing {} failed: {error}", target.display()));
    }
    tool_result(json!({
        "success": true,
        "action": "patch",
        "name": name,
        "path": target.display().to_string(),
        "matches_replaced": if replace_all { matches } else { 1 },
    }))
}

fn handle_skill_delete(runtime: &ToolRuntime, name: &str) -> String {
    let skill = match resolved_skill(runtime, name) {
        Ok(skill) => skill,
        Err(error) => return tool_result(json!({"success": false, "error": error})),
    };
    if let Err(error) = fs::remove_dir_all(&skill.skill_dir) {
        return tool_error(format!(
            "deleting {} failed: {error}",
            skill.skill_dir.display()
        ));
    }
    tool_result(json!({
        "success": true,
        "action": "delete",
        "name": name,
    }))
}

fn handle_skill_write_file(args: &Value, runtime: &ToolRuntime, name: &str) -> String {
    let file_path = match required_non_empty_string(args, "file_path") {
        Ok(value) => value,
        Err(_) => {
            return tool_result(json!({
                "success": false,
                "error": "file_path is required for 'write_file'. Example: 'references/api-guide.md'",
            }));
        }
    };
    let file_content = match args.get("file_content") {
        Some(Value::String(value)) => value.clone(),
        Some(_) => return tool_error("file_content must be a string"),
        None => {
            return tool_result(json!({
                "success": false,
                "error": "file_content is required for 'write_file'.",
            }));
        }
    };
    let skill = match resolved_skill(runtime, name) {
        Ok(skill) => skill,
        Err(error) => return tool_result(json!({"success": false, "error": error})),
    };
    let relative = match validated_supporting_file_path(&file_path) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let target = skill.skill_dir.join(&relative);
    if let Some(parent) = target.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        return tool_error(format!("creating {} failed: {error}", parent.display()));
    }
    if let Err(error) = fs::write(&target, file_content) {
        return tool_error(format!("writing {} failed: {error}", target.display()));
    }
    tool_result(json!({
        "success": true,
        "action": "write_file",
        "name": name,
        "path": target.display().to_string(),
    }))
}

fn handle_skill_remove_file(args: &Value, runtime: &ToolRuntime, name: &str) -> String {
    let file_path = match required_non_empty_string(args, "file_path") {
        Ok(value) => value,
        Err(_) => {
            return tool_result(json!({
                "success": false,
                "error": "file_path is required for 'remove_file'.",
            }));
        }
    };
    let skill = match resolved_skill(runtime, name) {
        Ok(skill) => skill,
        Err(error) => return tool_result(json!({"success": false, "error": error})),
    };
    let relative = match validated_supporting_file_path(&file_path) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let target = skill.skill_dir.join(&relative);
    if !target.exists() || !target.is_file() {
        return tool_result(json!({
            "success": false,
            "error": format!("Supporting file '{}' does not exist for skill '{}'.", file_path, name),
        }));
    }
    if let Err(error) = fs::remove_file(&target) {
        return tool_error(format!("removing {} failed: {error}", target.display()));
    }
    tool_result(json!({
        "success": true,
        "action": "remove_file",
        "name": name,
        "path": target.display().to_string(),
    }))
}

fn resolved_skill(runtime: &ToolRuntime, name: &str) -> Result<SkillEntry, String> {
    let skills_root = runtime.hermes_home().join("skills");
    let skills = discover_skills(&skills_root)?;
    find_skill(&skills_root, &skills, name).ok_or_else(|| format!("Skill '{name}' not found."))
}

fn collect_skill_files(dir: &Path, results: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|error| format!("reading skills directory {} failed: {error}", dir.display()))?;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if file_type.is_dir() {
            if EXCLUDED_SKILL_DIRS.contains(&name) {
                continue;
            }
            collect_skill_files(&path, results)?;
        } else if file_type.is_file() && name == "SKILL.md" {
            results.push(path);
        }
    }
    Ok(())
}

fn find_skill(skills_root: &Path, skills: &[SkillEntry], name: &str) -> Option<SkillEntry> {
    let direct_path = skills_root.join(name);
    if direct_path.is_dir() && direct_path.join("SKILL.md").exists() {
        return skills
            .iter()
            .find(|skill| skill.skill_dir == direct_path)
            .cloned();
    }
    skills
        .iter()
        .find(|skill| {
            skill.name == name
                || skill
                    .skill_dir
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value == name)
        })
        .cloned()
}

fn normalize_skill_command_name(name: &str) -> String {
    let mut normalized = name.to_ascii_lowercase().replace([' ', '_'], "-");
    normalized.retain(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-');
    while normalized.contains("--") {
        normalized = normalized.replace("--", "-");
    }
    normalized.trim_matches('-').to_string()
}

fn parse_frontmatter(content: &str) -> (BTreeMap<String, YamlValue>, String) {
    let trimmed = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"));
    let Some(rest) = trimmed else {
        return (BTreeMap::new(), content.to_string());
    };
    let separator = if let Some(index) = rest.find("\n---\n") {
        (index, 5)
    } else if let Some(index) = rest.find("\n---\r\n") {
        (index, 6)
    } else {
        return (BTreeMap::new(), content.to_string());
    };
    let yaml_text = &rest[..separator.0];
    let body = rest[separator.0 + separator.1..].to_string();
    let frontmatter = serde_yaml::from_str::<YamlValue>(yaml_text)
        .ok()
        .and_then(|value| match value {
            YamlValue::Mapping(mapping) => Some(mapping),
            _ => None,
        })
        .map(|mapping| {
            mapping
                .into_iter()
                .filter_map(|(key, value)| key.as_str().map(|key| (key.to_string(), value)))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    (frontmatter, body)
}

fn resolved_description(
    frontmatter: &BTreeMap<String, YamlValue>,
    content: &str,
    skill_name: &str,
) -> String {
    if let Some(description) = frontmatter
        .get("description")
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return truncate(description, MAX_DESCRIPTION_LENGTH);
    }
    let (_, body) = parse_frontmatter(content);
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return truncate(trimmed, MAX_DESCRIPTION_LENGTH);
    }
    truncate(skill_name, MAX_DESCRIPTION_LENGTH)
}

fn category_from_path(skills_root: &Path, skill_md: &Path) -> Option<String> {
    let relative = skill_md.strip_prefix(skills_root).ok()?;
    let mut parts = relative.components();
    let first = parts.next()?.as_os_str().to_str()?.trim();
    let second = parts.next()?;
    if second.as_os_str() == "SKILL.md" || first.is_empty() {
        return None;
    }
    Some(first.to_string())
}

fn linked_files(skill_dir: &Path) -> Option<JsonMap<String, Value>> {
    let mut result = JsonMap::new();
    for group in ["references", "templates", "assets", "scripts"] {
        let dir = skill_dir.join(group);
        if !dir.exists() || !dir.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        if collect_relative_files(skill_dir, &dir, &mut files).is_ok() && !files.is_empty() {
            files.sort();
            result.insert(group.to_string(), json!(files));
        }
    }
    (!result.is_empty()).then_some(result)
}

fn collect_relative_files(
    root: &Path,
    dir: &Path,
    results: &mut Vec<String>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("reading skill directory {} failed: {error}", dir.display()))?
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            collect_relative_files(root, &path, results)?;
        } else if file_type.is_file()
            && let Ok(relative) = path.strip_prefix(root)
        {
            results.push(relative.to_string_lossy().to_string());
        }
    }
    Ok(())
}

fn available_files(skill_dir: &Path) -> Value {
    let mut result = JsonMap::new();
    if let Some(linked) = linked_files(skill_dir) {
        for (key, value) in linked {
            result.insert(key, value);
        }
    }
    Value::Object(result)
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn optional_non_empty_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn optional_simple_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    let value = optional_non_empty_string(args, key)?;
    if let Some(value) = value.as_deref()
        && (value.contains('/') || value.contains('\\'))
    {
        return Err(format!("{key} must be a simple category name"));
    }
    Ok(value)
}

fn validated_relative_path(input: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(input);
    if path.is_absolute() {
        return Err("Absolute paths are not allowed.".to_string());
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            Component::ParentDir => return Err("Path traversal ('..') is not allowed.".to_string()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("Absolute paths are not allowed.".to_string());
            }
        }
    }
    Ok(path)
}

fn validated_supporting_file_path(input: &str) -> Result<PathBuf, String> {
    let path = validated_relative_path(input)?;
    let Some(first) = path.components().next() else {
        return Err("file_path must not be empty".to_string());
    };
    let Component::Normal(first) = first else {
        return Err("file_path must be relative".to_string());
    };
    let Some(first) = first.to_str() else {
        return Err("file_path must be valid UTF-8".to_string());
    };
    if !ALLOWED_SUBDIRS.contains(&first) {
        return Err(
            "file_path must live under references/, templates/, scripts/, or assets/.".to_string(),
        );
    }
    Ok(path)
}

fn validate_skill_name(name: &str) -> Result<(), String> {
    if name.len() > MAX_NAME_LENGTH {
        return Err(format!("Skill name exceeds {MAX_NAME_LENGTH} characters."));
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err("Skill name is required.".to_string());
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(format!(
            "Invalid skill name '{name}'. Use lowercase letters, numbers, hyphens, dots, and underscores. Must start with a letter or digit."
        ));
    }
    if chars.any(|ch| {
        !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_' | '.'))
    }) {
        return Err(format!(
            "Invalid skill name '{name}'. Use lowercase letters, numbers, hyphens, dots, and underscores. Must start with a letter or digit."
        ));
    }
    Ok(())
}

fn validate_skill_frontmatter(content: &str, expected_name: &str) -> Result<(), String> {
    let (frontmatter, _) = parse_frontmatter(content);
    let name = frontmatter
        .get("name")
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            "SKILL.md content must include YAML frontmatter with a non-empty 'name' field."
                .to_string()
        })?;
    let description = frontmatter
        .get("description")
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            "SKILL.md content must include YAML frontmatter with a non-empty 'description' field."
                .to_string()
        })?;
    validate_skill_name(name)?;
    if name != expected_name {
        return Err(format!(
            "Frontmatter name '{name}' must match the tool argument '{expected_name}'."
        ));
    }
    if description.chars().count() > MAX_DESCRIPTION_LENGTH {
        return Err(format!(
            "Skill description exceeds {MAX_DESCRIPTION_LENGTH} characters."
        ));
    }
    Ok(())
}

fn frontmatter_path<'a>(
    frontmatter: &'a BTreeMap<String, YamlValue>,
    path: &[&str],
) -> Option<&'a YamlValue> {
    let mut current = frontmatter.get(*path.first()?)?;
    for segment in &path[1..] {
        current = match current {
            YamlValue::Mapping(mapping) => {
                mapping.get(YamlValue::String((*segment).to_string()))?
            }
            _ => return None,
        };
    }
    Some(current)
}

fn parsed_tags(value: Option<&YamlValue>) -> Option<Vec<String>> {
    let value = value?;
    match value {
        YamlValue::Sequence(items) => Some(
            items
                .iter()
                .filter_map(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect(),
        ),
        YamlValue::String(text) => Some(
            text.trim_matches(|ch| ch == '[' || ch == ']')
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.trim_matches(|ch| ch == '"' || ch == '\'').to_string())
                .collect(),
        ),
        _ => None,
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let mut truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() && max_chars >= 3 {
        truncated.truncate(max_chars.saturating_sub(3));
        truncated.push_str("...");
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    fn write_skill(home: &Path) {
        let skill_dir = home.join("skills/mlops/axolotl");
        fs::create_dir_all(skill_dir.join("references")).unwrap();
        fs::create_dir_all(skill_dir.join("templates")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: axolotl
description: Fine-tuning workflow
metadata:
  hermes:
    tags: [mlops, finetuning]
    related_skills: [peft]
---

# Axolotl

Use this workflow.
"#,
        )
        .unwrap();
        fs::write(skill_dir.join("references/api.md"), "ref").unwrap();
        fs::write(skill_dir.join("templates/config.yaml"), "cfg").unwrap();
    }

    #[test]
    fn skills_list_discovers_local_skills() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path());
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let result = handle_skills_list(&json!({}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));
        assert_eq!(parsed["count"], json!(1));
        assert_eq!(parsed["categories"], json!(["mlops"]));
        assert_eq!(parsed["skills"][0]["name"], json!("axolotl"));
    }

    #[test]
    fn skill_view_returns_main_content_and_linked_files() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path());
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let result = handle_skill_view(&json!({"name": "axolotl"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));
        assert_eq!(parsed["name"], json!("axolotl"));
        assert_eq!(parsed["tags"], json!(["mlops", "finetuning"]));
        assert_eq!(parsed["related_skills"], json!(["peft"]));
        assert_eq!(
            parsed["linked_files"]["references"][0],
            json!("references/api.md")
        );
    }

    #[test]
    fn build_skill_invocation_message_loads_skill_and_user_instruction() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path());

        let built = build_skill_invocation_message(
            temp.path(),
            "axolotl",
            "focus on staging",
            Some("sess-1"),
        )
        .unwrap()
        .unwrap();

        assert_eq!(built.0, "axolotl");
        assert!(
            built
                .1
                .contains("The user has invoked the \"axolotl\" skill")
        );
        assert!(built.1.contains("focus on staging"));
        assert!(built.1.contains("references/api.md"));
        assert!(built.1.contains("session_id=sess-1"));
    }

    #[test]
    fn skill_view_reads_linked_file_and_rejects_traversal() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path());
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let file_result = handle_skill_view(
            &json!({"name": "axolotl", "file_path": "references/api.md"}),
            &runtime,
        );
        let file_json: Value = serde_json::from_str(&file_result).unwrap();
        assert_eq!(file_json["success"], Value::Bool(true));
        assert_eq!(file_json["content"], json!("ref"));

        let bad_result = handle_skill_view(
            &json!({"name": "axolotl", "file_path": "../secret.txt"}),
            &runtime,
        );
        let bad_json: Value = serde_json::from_str(&bad_result).unwrap();
        assert_eq!(bad_json["success"], Value::Bool(false));
        assert!(
            bad_json["error"]
                .as_str()
                .unwrap()
                .contains("Path traversal")
        );
    }

    #[test]
    fn skill_manage_can_create_patch_and_delete_skill() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let create = handle_skill_manage(
            &json!({
                "action": "create",
                "name": "demo-skill",
                "category": "devops",
                "content": "---\nname: demo-skill\ndescription: Demo skill\n---\n\n# Demo\n\nUse this.\n",
            }),
            &runtime,
        );
        let create_json: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create_json["success"], Value::Bool(true));

        let patch = handle_skill_manage(
            &json!({
                "action": "patch",
                "name": "demo-skill",
                "old_string": "Use this.",
                "new_string": "Use that.",
            }),
            &runtime,
        );
        let patch_json: Value = serde_json::from_str(&patch).unwrap();
        assert_eq!(patch_json["success"], Value::Bool(true));

        let view = handle_skill_view(&json!({"name": "demo-skill"}), &runtime);
        let view_json: Value = serde_json::from_str(&view).unwrap();
        assert!(view_json["content"].as_str().unwrap().contains("Use that."));

        let delete = handle_skill_manage(
            &json!({
                "action": "delete",
                "name": "demo-skill",
            }),
            &runtime,
        );
        let delete_json: Value = serde_json::from_str(&delete).unwrap();
        assert_eq!(delete_json["success"], Value::Bool(true));
    }

    #[test]
    fn skill_manage_can_write_and_remove_supporting_files() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path());
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let write = handle_skill_manage(
            &json!({
                "action": "write_file",
                "name": "axolotl",
                "file_path": "references/extra.md",
                "file_content": "extra",
            }),
            &runtime,
        );
        let write_json: Value = serde_json::from_str(&write).unwrap();
        assert_eq!(write_json["success"], Value::Bool(true));

        let read = handle_skill_view(
            &json!({"name": "axolotl", "file_path": "references/extra.md"}),
            &runtime,
        );
        let read_json: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(read_json["content"], json!("extra"));

        let remove = handle_skill_manage(
            &json!({
                "action": "remove_file",
                "name": "axolotl",
                "file_path": "references/extra.md",
            }),
            &runtime,
        );
        let remove_json: Value = serde_json::from_str(&remove).unwrap();
        assert_eq!(remove_json["success"], Value::Bool(true));
    }

    #[test]
    fn scan_skill_commands_normalizes_and_filters() {
        let temp = TempDir::new().unwrap();
        let home = temp.path();
        // A normal skill with a frontmatter description.
        let a = home.join("skills/group/Code Review");
        fs::create_dir_all(&a).unwrap();
        fs::write(
            a.join("SKILL.md"),
            "---\nname: Code Review\ndescription: Review a diff\n---\n\n# body\n",
        )
        .unwrap();
        // A skill with no description -> falls back to first body line (<=80).
        let b = home.join("skills/no_desc");
        fs::create_dir_all(&b).unwrap();
        fs::write(
            b.join("SKILL.md"),
            "---\nname: deploy_helper\n---\n\n# Heading\nHelps you deploy things\n",
        )
        .unwrap();
        // A disabled skill (config.yaml skills.disabled).
        let c = home.join("skills/secret");
        fs::create_dir_all(&c).unwrap();
        fs::write(c.join("SKILL.md"), "---\nname: secret\n---\n\nhidden\n").unwrap();
        fs::write(
            home.join("config.yaml"),
            "skills:\n  disabled:\n    - secret\n",
        )
        .unwrap();

        let commands = scan_skill_commands(home, None);
        let by_cmd: std::collections::HashMap<_, _> = commands
            .iter()
            .map(|s| (s.command.as_str(), s))
            .collect();

        // "Code Review" -> "/code-review" (spaces -> hyphens, lowercased)
        assert_eq!(by_cmd["/code-review"].name, "Code Review");
        assert_eq!(by_cmd["/code-review"].description, "Review a diff");
        // "deploy_helper" -> "/deploy-helper", body-line fallback description
        assert_eq!(by_cmd["/deploy-helper"].description, "Helps you deploy things");
        // disabled skill is absent
        assert!(!by_cmd.contains_key("/secret"));
    }
}
