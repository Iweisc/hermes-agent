//! Skill Manager Tool — native Rust port of `tools/skill_manager_tool.py`.
//!
//! Allows the agent to create, update, and delete skills, turning successful
//! approaches into reusable procedural knowledge. New skills are created in
//! `~/.hermes/skills/`. Existing skills (bundled, hub-installed, or
//! user-created) can be modified or deleted wherever they live.
//!
//! Actions:
//!   - `create`      — Create a new skill (SKILL.md + directory structure)
//!   - `edit`        — Replace the SKILL.md content of a skill (full rewrite)
//!   - `patch`       — Targeted find-and-replace within SKILL.md or a file
//!   - `delete`      — Remove a skill entirely
//!   - `write_file`  — Add/overwrite a supporting file
//!   - `remove_file` — Remove a supporting file from a skill
//!
//! Directory layout for user skills:
//! ```text
//!     ~/.hermes/skills/
//!     ├── my-skill/
//!     │   ├── SKILL.md
//!     │   ├── references/
//!     │   ├── templates/
//!     │   ├── scripts/
//!     │   └── assets/
//!     └── category-name/
//!         └── another-skill/
//!             └── SKILL.md
//! ```
//!
//! ## Ports / dependencies
//!
//! Core validation, file I/O, dispatch, and the find/create/edit/patch/delete
//! logic are fully native. Several Python collaborators are not yet ported to
//! Rust; rather than block, they are modelled as pluggable hooks with
//! best-effort no-op defaults (matching the Python `try/except` semantics):
//!
//! - `tools.skills_guard` security scanning → native via
//!   [`crate::skills_guard`] is not used directly here; the scan is wired
//!   through [`SkillManagerHooks::security_scan`] (default no-op, since the
//!   gate is disabled by default).
//! - `tools.skill_usage` telemetry (`pinned`, `bump_patch`, `forget`,
//!   `mark_agent_created`) → [`SkillManagerHooks`] callbacks (default no-op).
//! - `agent.prompt_builder.clear_skills_system_prompt_cache` → hook (no-op).
//! - `tools.skill_provenance.is_background_review` → hook (default false).
//!
//! The fuzzy matcher, path-security helpers, atomic-write, skill-dir discovery,
//! and `hermes_home` resolution all reuse already-ported core modules.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{json, Value as JsonValue};

use hermes_core::ag_skill_utils::get_all_skills_dirs;
use hermes_core::mod_hermes_constants::{
    display_hermes_home, get_config_path, get_hermes_home,
};
use hermes_core::tool_fuzzy_match::{format_no_match_hint, fuzzy_find_and_replace};
use hermes_core::tool_path_security::{has_traversal_component, validate_within_dir};

// =============================================================================
// Constants
// =============================================================================

pub const MAX_NAME_LENGTH: usize = 64;
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// ~36k tokens at 2.75 chars/token.
pub const MAX_SKILL_CONTENT_CHARS: usize = 100_000;
/// 1 MiB per supporting file.
pub const MAX_SKILL_FILE_BYTES: usize = 1_048_576;

/// Subdirectories allowed for write_file/remove_file.
pub const ALLOWED_SUBDIRS: &[&str] = &["references", "templates", "scripts", "assets"];

/// Return the local skills root (`~/.hermes/skills`).
pub fn skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}

// =============================================================================
// Hooks for not-yet-ported collaborators (best-effort, no-op by default)
// =============================================================================

/// Pluggable hooks mirroring the Python module's `try/except`-wrapped
/// collaborators (security scan, usage telemetry, prompt-cache clearing,
/// provenance). All default to the conservative/no-op behaviour the Python
/// code falls back to when the optional imports are unavailable.
pub struct SkillManagerHooks {
    /// Scan a freshly-written skill directory. Returns `Some(error)` to block
    /// (and trigger rollback), or `None` to allow. Default: never blocks.
    pub security_scan: Box<dyn Fn(&Path) -> Option<String> + Send + Sync>,
    /// Return a refusal message if a skill name is pinned, else `None`.
    /// Default: nothing is pinned.
    pub pinned_guard: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
    /// Clear the cached skills system prompt after a successful mutation.
    pub clear_prompt_cache: Box<dyn Fn() + Send + Sync>,
    /// Telemetry: bump patch_count for `name`.
    pub bump_patch: Box<dyn Fn(&str) + Send + Sync>,
    /// Telemetry: drop the record for `name` (on delete).
    pub forget: Box<dyn Fn(&str) + Send + Sync>,
    /// Telemetry: mark `name` as agent-created.
    pub mark_agent_created: Box<dyn Fn(&str) + Send + Sync>,
    /// Whether the current write is a background self-improvement review.
    pub is_background_review: Box<dyn Fn() -> bool + Send + Sync>,
}

impl Default for SkillManagerHooks {
    fn default() -> Self {
        SkillManagerHooks {
            security_scan: Box::new(|_| None),
            pinned_guard: Box::new(|_| None),
            clear_prompt_cache: Box::new(|| {}),
            bump_patch: Box::new(|_| {}),
            forget: Box::new(|_| {}),
            mark_agent_created: Box::new(|_| {}),
            is_background_review: Box::new(|| false),
        }
    }
}

impl std::fmt::Debug for SkillManagerHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SkillManagerHooks { .. }")
    }
}

// =============================================================================
// Discovery helpers
// =============================================================================

/// All skill roots (local `~/.hermes/skills` first, then external dirs).
fn all_skills_dirs() -> Vec<PathBuf> {
    get_all_skills_dirs(&get_config_path(), &get_hermes_home(), &skills_dir())
}

/// A located skill on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundSkill {
    pub path: PathBuf,
}

/// Find a skill by name across all skill directories.
///
/// Searches the local skills dir first, then any external dirs configured via
/// `skills.external_dirs`. Returns the directory containing `SKILL.md` whose
/// parent directory name matches `name`.
pub fn find_skill(name: &str) -> Option<FoundSkill> {
    for sdir in all_skills_dirs() {
        if !sdir.exists() {
            continue;
        }
        for skill_md in rglob_skill_md(&sdir) {
            if let Some(parent) = skill_md.parent() {
                if parent.file_name().and_then(|n| n.to_str()) == Some(name) {
                    return Some(FoundSkill {
                        path: parent.to_path_buf(),
                    });
                }
            }
        }
    }
    None
}

/// Recursively collect all `SKILL.md` paths beneath `root` (mirrors
/// `Path.rglob("SKILL.md")`).
fn rglob_skill_md(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let ftype = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ftype.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                out.push(path);
            }
        }
    }
    out
}

/// Return the skills root directory (local or external_dirs entry) that
/// contains `skill_path`. Falls back to the local skills dir if no match.
fn containing_skills_root(skill_path: &Path) -> PathBuf {
    let resolved = fs::canonicalize(skill_path).unwrap_or_else(|_| skill_path.to_path_buf());
    for root in all_skills_dirs() {
        let root_resolved = fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        if resolved.starts_with(&root_resolved) {
            return root;
        }
    }
    skills_dir()
}

// =============================================================================
// Validation helpers
// =============================================================================

/// Is `name` filesystem-safe / URL-friendly?
/// Mirrors `VALID_NAME_RE = ^[a-z0-9][a-z0-9._-]*$`.
fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-')
}

/// Validate a skill name. Returns an error message or `None` if valid.
pub fn validate_name(name: &str) -> Option<String> {
    if name.is_empty() {
        return Some("Skill name is required.".to_string());
    }
    if name.chars().count() > MAX_NAME_LENGTH {
        return Some(format!("Skill name exceeds {MAX_NAME_LENGTH} characters."));
    }
    if !valid_name(name) {
        return Some(format!(
            "Invalid skill name '{name}'. Use lowercase letters, numbers, \
             hyphens, dots, and underscores. Must start with a letter or digit."
        ));
    }
    None
}

/// Validate an optional category name used as a single directory segment.
pub fn validate_category(category: Option<&str>) -> Option<String> {
    let category = match category {
        None => return None,
        Some(c) => c.trim(),
    };
    if category.is_empty() {
        return None;
    }
    if category.contains('/') || category.contains('\\') {
        return Some(format!(
            "Invalid category '{category}'. Use lowercase letters, numbers, \
             hyphens, dots, and underscores. Categories must be a single directory name."
        ));
    }
    if category.chars().count() > MAX_NAME_LENGTH {
        return Some(format!("Category exceeds {MAX_NAME_LENGTH} characters."));
    }
    if !valid_name(category) {
        return Some(format!(
            "Invalid category '{category}'. Use lowercase letters, numbers, \
             hyphens, dots, and underscores. Categories must be a single directory name."
        ));
    }
    None
}

/// Find the closing `---` of the YAML frontmatter inside `content[3:]`.
///
/// Mirrors `re.search(r'\n---\s*\n', content[3:])`, returning the (start, end)
/// char offsets *relative to the slice `content[3:]`* of the match, matching
/// Python's `end_match.start()`/`end_match.end()`.
fn find_frontmatter_close(after: &str) -> Option<(usize, usize)> {
    // Pattern: '\n' '---' then \s* (greedy) then a required '\n'.
    // `re.search` returns the leftmost match; emulate by scanning left→right.
    let chars: Vec<char> = after.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if chars[i] == '\n' && chars[i + 1] == '-' && chars[i + 2] == '-' && chars[i + 3] == '-' {
            let start = i;
            let body_start = i + 4;
            // \s* is greedy: consume all whitespace, then backtrack so the
            // last matched char is the mandatory trailing '\n'.
            let mut k = body_start;
            while k < n && chars[k].is_whitespace() {
                k += 1;
            }
            let mut end = None;
            let mut t = k;
            while t > body_start {
                if chars[t - 1] == '\n' {
                    end = Some(t);
                    break;
                }
                t -= 1;
            }
            // \s* may match zero chars, with the trailing '\n' immediately next.
            if end.is_none() && body_start < n && chars[body_start] == '\n' {
                end = Some(body_start + 1);
            }
            if let Some(e) = end {
                return Some((start, e));
            }
        }
        i += 1;
    }
    None
}

/// Validate that SKILL.md content has proper frontmatter with required fields.
/// Returns an error message or `None` if valid.
pub fn validate_frontmatter(content: &str) -> Option<String> {
    if content.trim().is_empty() {
        return Some("Content cannot be empty.".to_string());
    }
    if !content.starts_with("---") {
        return Some(
            "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
                .to_string(),
        );
    }

    // content[3:] in char terms.
    let after: String = content.chars().skip(3).collect();
    let close = match find_frontmatter_close(&after) {
        Some(c) => c,
        None => {
            return Some(
                "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
                    .to_string(),
            )
        }
    };

    // yaml_content = content[3 : end_match.start() + 3] (char slice).
    let after_chars: Vec<char> = after.chars().collect();
    let yaml_content: String = after_chars[..close.0].iter().collect();

    let parsed: serde_yaml::Value = match serde_yaml::from_str(&yaml_content) {
        Ok(v) => v,
        Err(e) => return Some(format!("YAML frontmatter parse error: {e}")),
    };

    let map = match parsed.as_mapping() {
        Some(m) => m,
        None => return Some("Frontmatter must be a YAML mapping (key: value pairs).".to_string()),
    };

    let name_key = serde_yaml::Value::String("name".to_string());
    let desc_key = serde_yaml::Value::String("description".to_string());
    if !map.contains_key(&name_key) {
        return Some("Frontmatter must include 'name' field.".to_string());
    }
    let desc = match map.get(&desc_key) {
        Some(d) => d,
        None => return Some("Frontmatter must include 'description' field.".to_string()),
    };
    if yaml_scalar_str(desc).chars().count() > MAX_DESCRIPTION_LENGTH {
        return Some(format!(
            "Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
        ));
    }

    // body = content[end_match.end() + 3:].strip()
    // end_match.end() is relative to `after`; the +3 puts it back in `content`
    // coordinates, i.e. after.chars()[close.1..].
    let body: String = after_chars[close.1..].iter().collect();
    if body.trim().is_empty() {
        return Some(
            "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
                .to_string(),
        );
    }

    None
}

/// Render a YAML scalar the way Python's `str(parsed["description"])` would
/// for the common scalar cases (string/number/bool). Falls back to the YAML
/// serialization for complex values (matching length-check intent).
fn yaml_scalar_str(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Bool(b) => {
            // Python str(True) == "True"
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Null => "None".to_string(),
        other => serde_yaml::to_string(other).unwrap_or_default().trim().to_string(),
    }
}

/// Check that content doesn't exceed the character limit for agent writes.
pub fn validate_content_size(content: &str, label: &str) -> Option<String> {
    let len = content.chars().count();
    if len > MAX_SKILL_CONTENT_CHARS {
        return Some(format!(
            "{label} content is {} characters (limit: {}). \
             Consider splitting into a smaller SKILL.md with supporting files \
             in references/ or templates/.",
            comma_sep(len),
            comma_sep(MAX_SKILL_CONTENT_CHARS),
        ));
    }
    None
}

/// Format an integer with thousands separators (Python `{:,}`).
fn comma_sep(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Build the directory path for a new skill, optionally under a category.
fn resolve_skill_dir(name: &str, category: Option<&str>) -> PathBuf {
    match category {
        Some(c) if !c.trim().is_empty() => skills_dir().join(c).join(name),
        _ => skills_dir().join(name),
    }
}

/// Validate a file path for write_file/remove_file. Must be under an allowed
/// subdirectory and not escape the skill dir.
pub fn validate_file_path(file_path: &str) -> Option<String> {
    if file_path.is_empty() {
        return Some("file_path is required.".to_string());
    }
    if has_traversal_component(file_path) {
        return Some("Path traversal ('..') is not allowed.".to_string());
    }

    let parts: Vec<String> = path_parts(file_path);
    let first = parts.first().map(|s| s.as_str());
    match first {
        Some(p) if ALLOWED_SUBDIRS.contains(&p) => {}
        _ => {
            let mut allowed: Vec<&str> = ALLOWED_SUBDIRS.to_vec();
            allowed.sort_unstable();
            return Some(format!(
                "File must be under one of: {}. Got: '{file_path}'",
                allowed.join(", ")
            ));
        }
    }

    if parts.len() < 2 {
        return Some(format!(
            "Provide a file path, not just a directory. Example: '{}/myfile.md'",
            parts[0]
        ));
    }

    None
}

/// Return path components mirroring `pathlib.Path(p).parts` for the relative
/// paths this tool handles (drops empty/`.` segments).
fn path_parts(p: &str) -> Vec<String> {
    Path::new(p)
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => s.to_str().map(|x| x.to_string()),
            std::path::Component::ParentDir => Some("..".to_string()),
            _ => None,
        })
        .collect()
}

/// Resolve a supporting-file path and ensure it stays within the skill dir.
fn resolve_skill_target(skill_dir: &Path, file_path: &str) -> Result<PathBuf, String> {
    let target = skill_dir.join(file_path);
    if let Some(err) = validate_within_dir(&target, skill_dir) {
        return Err(err);
    }
    Ok(target)
}

/// Atomically write text content to a file (temp file in same dir + rename).
fn atomic_write_text(file_path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let parent = file_path.parent().unwrap_or_else(|| Path::new("."));
    let fname = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("skillfile");
    // Create a unique temp file in the same directory.
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{fname}.tmp.{pid}_{nanos}"));

    let write_res = (|| -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.sync_all().ok();
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    match hermes_core::mod_utils::atomic_replace(&tmp, file_path) {
        Ok(_) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

// =============================================================================
// Core actions
// =============================================================================

fn err_result(msg: impl Into<String>) -> JsonValue {
    json!({"success": false, "error": msg.into()})
}

/// Create a new user skill with SKILL.md content.
pub fn create_skill(
    name: &str,
    content: &str,
    category: Option<&str>,
    hooks: &SkillManagerHooks,
) -> JsonValue {
    if let Some(e) = validate_name(name) {
        return err_result(e);
    }
    if let Some(e) = validate_category(category) {
        return err_result(e);
    }
    if let Some(e) = validate_frontmatter(content) {
        return err_result(e);
    }
    if let Some(e) = validate_content_size(content, "SKILL.md") {
        return err_result(e);
    }

    if let Some(existing) = find_skill(name) {
        return err_result(format!(
            "A skill named '{name}' already exists at {}.",
            existing.path.display()
        ));
    }

    let skill_dir = resolve_skill_dir(name, category);
    if let Err(e) = fs::create_dir_all(&skill_dir) {
        return err_result(format!("Failed to create skill directory: {e}"));
    }

    let skill_md = skill_dir.join("SKILL.md");
    if let Err(e) = atomic_write_text(&skill_md, content) {
        return err_result(format!("Failed to write SKILL.md: {e}"));
    }

    if let Some(scan_error) = (hooks.security_scan)(&skill_dir) {
        let _ = fs::remove_dir_all(&skill_dir);
        return err_result(scan_error);
    }

    let rel = skill_dir
        .strip_prefix(skills_dir())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| skill_dir.clone());

    let mut result = json!({
        "success": true,
        "message": format!("Skill '{name}' created."),
        "path": rel.to_string_lossy(),
        "skill_md": skill_md.to_string_lossy(),
    });
    if let Some(c) = category {
        if !c.trim().is_empty() {
            result["category"] = json!(c);
        }
    }
    result["hint"] = json!(format!(
        "To add reference files, templates, or scripts, use \
         skill_manage(action='write_file', name='{name}', \
         file_path='references/example.md', file_content='...')"
    ));
    result
}

/// Replace the SKILL.md of any existing skill (full rewrite).
pub fn edit_skill(name: &str, content: &str, hooks: &SkillManagerHooks) -> JsonValue {
    if let Some(e) = validate_frontmatter(content) {
        return err_result(e);
    }
    if let Some(e) = validate_content_size(content, "SKILL.md") {
        return err_result(e);
    }

    let existing = match find_skill(name) {
        Some(s) => s,
        None => {
            return err_result(format!(
                "Skill '{name}' not found. Use skills_list() to see available skills."
            ))
        }
    };

    let skill_md = existing.path.join("SKILL.md");
    let original_content = fs::read_to_string(&skill_md).ok();
    if let Err(e) = atomic_write_text(&skill_md, content) {
        return err_result(format!("Failed to write SKILL.md: {e}"));
    }

    if let Some(scan_error) = (hooks.security_scan)(&existing.path) {
        if let Some(orig) = original_content {
            let _ = atomic_write_text(&skill_md, &orig);
        }
        return err_result(scan_error);
    }

    json!({
        "success": true,
        "message": format!("Skill '{name}' updated."),
        "path": existing.path.to_string_lossy(),
    })
}

/// Targeted find-and-replace within a skill file. Defaults to SKILL.md.
pub fn patch_skill(
    name: &str,
    old_string: &str,
    new_string: Option<&str>,
    file_path: Option<&str>,
    replace_all: bool,
    hooks: &SkillManagerHooks,
) -> JsonValue {
    if old_string.is_empty() {
        return err_result("old_string is required for 'patch'.");
    }
    let new_string = match new_string {
        Some(s) => s,
        None => {
            return err_result(
                "new_string is required for 'patch'. Use an empty string to delete matched text.",
            )
        }
    };

    let existing = match find_skill(name) {
        Some(s) => s,
        None => return err_result(format!("Skill '{name}' not found.")),
    };
    let skill_dir = existing.path.clone();

    let target: PathBuf = match file_path {
        Some(fp) if !fp.is_empty() => {
            if let Some(e) = validate_file_path(fp) {
                return err_result(e);
            }
            match resolve_skill_target(&skill_dir, fp) {
                Ok(t) => t,
                Err(e) => return err_result(e),
            }
        }
        _ => skill_dir.join("SKILL.md"),
    };
    let has_file_path = matches!(file_path, Some(fp) if !fp.is_empty());

    if !target.exists() {
        let rel = target
            .strip_prefix(&skill_dir)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| target.clone());
        return err_result(format!("File not found: {}", rel.to_string_lossy()));
    }

    let content = match fs::read_to_string(&target) {
        Ok(c) => c,
        Err(e) => return err_result(format!("Failed to read file: {e}")),
    };

    let res = fuzzy_find_and_replace(&content, old_string, new_string, replace_all);
    if let Some(match_error) = res.error {
        let preview: String = {
            let chars: Vec<char> = content.chars().collect();
            let head: String = chars.iter().take(500).collect();
            if chars.len() > 500 {
                format!("{head}...")
            } else {
                head
            }
        };
        let mut err_msg = match_error.clone();
        err_msg.push_str(&format_no_match_hint(
            Some(&match_error),
            res.match_count,
            old_string,
            &content,
        ));
        return json!({
            "success": false,
            "error": err_msg,
            "file_preview": preview,
        });
    }

    let new_content = res.content;
    let match_count = res.match_count;

    let target_label = match file_path {
        Some(fp) if !fp.is_empty() => fp.to_string(),
        _ => "SKILL.md".to_string(),
    };
    if let Some(e) = validate_content_size(&new_content, &target_label) {
        return err_result(e);
    }

    if !has_file_path {
        if let Some(e) = validate_frontmatter(&new_content) {
            return err_result(format!("Patch would break SKILL.md structure: {e}"));
        }
    }

    let original_content = content;
    if let Err(e) = atomic_write_text(&target, &new_content) {
        return err_result(format!("Failed to write file: {e}"));
    }

    if let Some(scan_error) = (hooks.security_scan)(&skill_dir) {
        let _ = atomic_write_text(&target, &original_content);
        return err_result(scan_error);
    }

    let what = if has_file_path {
        file_path.unwrap().to_string()
    } else {
        "SKILL.md".to_string()
    };
    let plural = if match_count > 1 { "s" } else { "" };
    json!({
        "success": true,
        "message": format!("Patched {what} in skill '{name}' ({match_count} replacement{plural})."),
    })
}

/// Delete a skill.
///
/// `absorbed_into` declares intent: `None` → undeclared (legacy); `Some("")`
/// → explicitly pruned with no forwarding target; `Some("<name>")` → content
/// absorbed into that umbrella (which must exist on disk).
pub fn delete_skill(
    name: &str,
    absorbed_into: Option<&str>,
    hooks: &SkillManagerHooks,
) -> JsonValue {
    let existing = match find_skill(name) {
        Some(s) => s,
        None => return err_result(format!("Skill '{name}' not found.")),
    };

    if let Some(pinned_err) = (hooks.pinned_guard)(name) {
        return err_result(pinned_err);
    }

    let declared_target: Option<String> = match absorbed_into {
        Some(a) if !a.trim().is_empty() => Some(a.trim().to_string()),
        _ => None,
    };

    if let Some(target_name) = &declared_target {
        if target_name == name {
            return err_result(format!(
                "absorbed_into='{target_name}' cannot equal the skill being deleted."
            ));
        }
        if find_skill(target_name).is_none() {
            return err_result(format!(
                "absorbed_into='{target_name}' does not exist. \
                 Create or patch the umbrella skill first, then retry the delete."
            ));
        }
    }

    let skill_dir = existing.path.clone();
    let skills_root = containing_skills_root(&skill_dir);
    if let Err(e) = fs::remove_dir_all(&skill_dir) {
        return err_result(format!("Failed to delete skill directory: {e}"));
    }

    // Clean up empty category directories (don't remove the skills root).
    if let Some(parent) = skill_dir.parent() {
        if parent != skills_root && parent.exists() && dir_is_empty(parent) {
            let _ = fs::remove_dir(parent);
        }
    }

    let mut message = format!("Skill '{name}' deleted.");
    if let Some(target) = &declared_target {
        message.push_str(&format!(" Content absorbed into '{target}'."));
    }

    json!({"success": true, "message": message})
}

/// Add or overwrite a supporting file within any skill directory.
pub fn write_file(
    name: &str,
    file_path: &str,
    file_content: &str,
    hooks: &SkillManagerHooks,
) -> JsonValue {
    if let Some(e) = validate_file_path(file_path) {
        return err_result(e);
    }

    // (In Python: `if not file_content and file_content != ""` — only triggers
    // for None, which the dispatcher already guards. Empty string is allowed.)

    let content_bytes = file_content.as_bytes().len();
    if content_bytes > MAX_SKILL_FILE_BYTES {
        return err_result(format!(
            "File content is {} bytes (limit: {} bytes / 1 MiB). \
             Consider splitting into smaller files.",
            comma_sep(content_bytes),
            comma_sep(MAX_SKILL_FILE_BYTES),
        ));
    }
    if let Some(e) = validate_content_size(file_content, file_path) {
        return err_result(e);
    }

    let existing = match find_skill(name) {
        Some(s) => s,
        None => {
            return err_result(format!(
                "Skill '{name}' not found. Create it first with action='create'."
            ))
        }
    };

    let target = match resolve_skill_target(&existing.path, file_path) {
        Ok(t) => t,
        Err(e) => return err_result(e),
    };
    if let Some(parent) = target.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let original_content = fs::read_to_string(&target).ok();
    if let Err(e) = atomic_write_text(&target, file_content) {
        return err_result(format!("Failed to write file: {e}"));
    }

    if let Some(scan_error) = (hooks.security_scan)(&existing.path) {
        match original_content {
            Some(orig) => {
                let _ = atomic_write_text(&target, &orig);
            }
            None => {
                let _ = fs::remove_file(&target);
            }
        }
        return err_result(scan_error);
    }

    json!({
        "success": true,
        "message": format!("File '{file_path}' written to skill '{name}'."),
        "path": target.to_string_lossy(),
    })
}

/// Remove a supporting file from any skill directory.
pub fn remove_file(name: &str, file_path: &str) -> JsonValue {
    if let Some(e) = validate_file_path(file_path) {
        return err_result(e);
    }

    let existing = match find_skill(name) {
        Some(s) => s,
        None => return err_result(format!("Skill '{name}' not found.")),
    };
    let skill_dir = existing.path.clone();

    let target = match resolve_skill_target(&skill_dir, file_path) {
        Ok(t) => t,
        Err(e) => return err_result(e),
    };

    if !target.exists() {
        let mut available: Vec<String> = Vec::new();
        for subdir in ALLOWED_SUBDIRS {
            let d = skill_dir.join(subdir);
            if d.exists() {
                for f in rglob_all_files(&d) {
                    if let Ok(rel) = f.strip_prefix(&skill_dir) {
                        available.push(rel.to_string_lossy().to_string());
                    }
                }
            }
        }
        return json!({
            "success": false,
            "error": format!("File '{file_path}' not found in skill '{name}'."),
            "available_files": if available.is_empty() { JsonValue::Null } else { json!(available) },
        });
    }

    if let Err(e) = fs::remove_file(&target) {
        return err_result(format!("Failed to remove file: {e}"));
    }

    // Clean up empty subdirectories.
    if let Some(parent) = target.parent() {
        if parent != skill_dir && parent.exists() && dir_is_empty(parent) {
            let _ = fs::remove_dir(parent);
        }
    }

    json!({
        "success": true,
        "message": format!("File '{file_path}' removed from skill '{name}'."),
    })
}

fn dir_is_empty(dir: &Path) -> bool {
    match fs::read_dir(dir) {
        Ok(mut it) => it.next().is_none(),
        Err(_) => false,
    }
}

/// Recursively collect all regular files beneath `root`.
fn rglob_all_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(path),
                Ok(t) if t.is_file() => out.push(path),
                _ => {}
            }
        }
    }
    out.sort();
    out
}

// =============================================================================
// Main entry point
// =============================================================================

/// Parameters for [`skill_manage`], mirroring the Python keyword arguments.
#[derive(Debug, Default, Clone)]
pub struct SkillManageArgs {
    pub action: String,
    pub name: String,
    pub content: Option<String>,
    pub category: Option<String>,
    pub file_path: Option<String>,
    pub file_content: Option<String>,
    pub old_string: Option<String>,
    pub new_string: Option<String>,
    pub replace_all: bool,
    pub absorbed_into: Option<String>,
}

impl SkillManageArgs {
    /// Build args from a JSON object the way the Python registry handler does
    /// (`args.get(...)` with defaults).
    pub fn from_json(args: &JsonValue) -> Self {
        let s = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|x| x.to_string());
        SkillManageArgs {
            action: s("action").unwrap_or_default(),
            name: s("name").unwrap_or_default(),
            content: s("content"),
            category: s("category"),
            file_path: s("file_path"),
            file_content: s("file_content"),
            old_string: s("old_string"),
            new_string: s("new_string"),
            replace_all: args
                .get("replace_all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            absorbed_into: s("absorbed_into"),
        }
    }
}

/// `tool_error` helper mirroring `tools.registry.tool_error(message, success=False)`.
fn tool_error_json(message: &str) -> String {
    serde_json::to_string(&json!({"success": false, "error": message}))
        .unwrap_or_else(|_| "{\"success\": false, \"error\": \"serialization failure\"}".to_string())
}

/// Manage user-created skills. Dispatches to the appropriate action handler.
/// Returns a JSON string with results (matching the Python contract).
pub fn skill_manage(args: &SkillManageArgs, hooks: &SkillManagerHooks) -> String {
    let action = args.action.as_str();
    let name = args.name.as_str();

    let result: JsonValue = match action {
        "create" => {
            let content = match non_empty(&args.content) {
                Some(c) => c,
                None => {
                    return tool_error_json(
                        "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).",
                    )
                }
            };
            create_skill(name, content, args.category.as_deref(), hooks)
        }
        "edit" => {
            let content = match non_empty(&args.content) {
                Some(c) => c,
                None => {
                    return tool_error_json(
                        "content is required for 'edit'. Provide the full updated SKILL.md text.",
                    )
                }
            };
            edit_skill(name, content, hooks)
        }
        "patch" => {
            let old_string = match non_empty(&args.old_string) {
                Some(s) => s,
                None => {
                    return tool_error_json(
                        "old_string is required for 'patch'. Provide the text to find.",
                    )
                }
            };
            if args.new_string.is_none() {
                return tool_error_json(
                    "new_string is required for 'patch'. Use empty string to delete matched text.",
                );
            }
            patch_skill(
                name,
                old_string,
                args.new_string.as_deref(),
                args.file_path.as_deref(),
                args.replace_all,
                hooks,
            )
        }
        "delete" => delete_skill(name, args.absorbed_into.as_deref(), hooks),
        "write_file" => {
            let file_path = match non_empty(&args.file_path) {
                Some(s) => s,
                None => {
                    return tool_error_json(
                        "file_path is required for 'write_file'. Example: 'references/api-guide.md'",
                    )
                }
            };
            let file_content = match &args.file_content {
                Some(c) => c.as_str(),
                None => return tool_error_json("file_content is required for 'write_file'."),
            };
            write_file(name, file_path, file_content, hooks)
        }
        "remove_file" => {
            let file_path = match non_empty(&args.file_path) {
                Some(s) => s,
                None => return tool_error_json("file_path is required for 'remove_file'."),
            };
            remove_file(name, file_path)
        }
        _ => json!({
            "success": false,
            "error": format!(
                "Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"
            ),
        }),
    };

    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        (hooks.clear_prompt_cache)();
        // Curator telemetry (best-effort). Mirrors the Python dispatcher.
        match action {
            "create" => {
                if (hooks.is_background_review)() {
                    (hooks.mark_agent_created)(name);
                }
            }
            "patch" | "edit" | "write_file" | "remove_file" => (hooks.bump_patch)(name),
            "delete" => (hooks.forget)(name),
            _ => {}
        }
    }

    serde_json::to_string(&result).unwrap_or_else(|_| {
        "{\"success\": false, \"error\": \"result serialization failure\"}".to_string()
    })
}

/// Python truthiness for an `Option<String>`: `Some(non-empty)` is truthy.
fn non_empty(opt: &Option<String>) -> Option<&str> {
    match opt {
        Some(s) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

// =============================================================================
// Function-calling schema
// =============================================================================

/// Build the OpenAI function-calling schema for `skill_manage`.
pub fn skill_manage_schema() -> JsonValue {
    let home = display_hermes_home();
    json!({
        "name": "skill_manage",
        "description": format!(
            "Manage skills (create, update, delete). Skills are your procedural \
             memory — reusable approaches for recurring task types. \
             New skills go to {home}/skills/; existing skills can be modified wherever they live.\n\n\
             Actions: create (full SKILL.md + optional category), \
             patch (old_string/new_string — preferred for fixes), \
             edit (full SKILL.md rewrite — major overhauls only), \
             delete, write_file, remove_file.\n\n\
             On delete, pass `absorbed_into=<umbrella>` when you're merging this \
             skill's content into another one, or `absorbed_into=\"\"` when you're \
             pruning it with no forwarding target. This lets the curator tell \
             consolidation from pruning without guessing, so downstream consumers \
             (cron jobs that reference the old skill name, etc.) get updated \
             correctly. The target you name in `absorbed_into` must already \
             exist — create/patch the umbrella first, then delete.\n\n\
             Create when: complex task succeeded (5+ calls), errors overcome, \
             user-corrected approach worked, non-trivial workflow discovered, \
             or user asks you to remember a procedure.\n\
             Update when: instructions stale/wrong, OS-specific failures, \
             missing steps or pitfalls found during use. \
             If you used a skill and hit issues not covered by it, patch it immediately.\n\n\
             After difficult/iterative tasks, offer to save as a skill. \
             Skip for simple one-offs. Confirm with user before creating/deleting.\n\n\
             Good skills: trigger conditions, numbered steps with exact commands, \
             pitfalls section, verification steps. Use skill_view() to see format examples.\n\n\
             Pinned skills are protected from deletion only — skill_manage(action='delete') \
             will refuse with a message pointing the user to `hermes curator unpin <name>`. \
             Patches and edits go through on pinned skills so you can still improve them as \
             pitfalls come up; pin only guards against irrecoverable loss."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
                    "description": "The action to perform."
                },
                "name": {
                    "type": "string",
                    "description":
                        "Skill name (lowercase, hyphens/underscores, max 64 chars). \
                         Must match an existing skill for patch/edit/delete/write_file/remove_file."
                },
                "content": {
                    "type": "string",
                    "description":
                        "Full SKILL.md content (YAML frontmatter + markdown body). \
                         Required for 'create' and 'edit'. For 'edit', read the skill \
                         first with skill_view() and provide the complete updated text."
                },
                "old_string": {
                    "type": "string",
                    "description":
                        "Text to find in the file (required for 'patch'). Must be unique \
                         unless replace_all=true. Include enough surrounding context to \
                         ensure uniqueness."
                },
                "new_string": {
                    "type": "string",
                    "description":
                        "Replacement text (required for 'patch'). Can be empty string \
                         to delete the matched text."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "For 'patch': replace all occurrences instead of requiring a unique match (default: false)."
                },
                "category": {
                    "type": "string",
                    "description":
                        "Optional category/domain for organizing the skill (e.g., 'devops', \
                         'data-science', 'mlops'). Creates a subdirectory grouping. \
                         Only used with 'create'."
                },
                "file_path": {
                    "type": "string",
                    "description":
                        "Path to a supporting file within the skill directory. \
                         For 'write_file'/'remove_file': required, must be under references/, \
                         templates/, scripts/, or assets/. \
                         For 'patch': optional, defaults to SKILL.md if omitted."
                },
                "file_content": {
                    "type": "string",
                    "description": "Content for the file. Required for 'write_file'."
                },
                "absorbed_into": {
                    "type": "string",
                    "description":
                        "For 'delete' only — declares intent so the curator can \
                         tell consolidation from pruning without guessing. \
                         Pass the umbrella skill name when this skill's content \
                         was merged into another (the target must already exist). \
                         Pass an empty string when the skill is truly stale and \
                         being pruned with no forwarding target. Omitting the arg \
                         on delete is supported for backward compatibility but \
                         downstream tooling (e.g. cron-job skill reference \
                         rewriting) will have to guess at intent."
                }
            },
            "required": ["action", "name"]
        }
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate HERMES_HOME env / filesystem state.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_home() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "hermes_skillmgr_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }

    fn good_skill_content() -> &'static str {
        "---\nname: my-skill\ndescription: A test skill\n---\n\nDo the thing.\n"
    }

    #[test]
    fn name_validation() {
        assert!(validate_name("").is_some());
        assert!(validate_name("Good-Name").is_some()); // uppercase rejected
        assert!(validate_name("-bad").is_some()); // must start alnum
        assert!(validate_name("good-name_1.2").is_none());
        assert!(validate_name(&"a".repeat(65)).is_some());
    }

    #[test]
    fn category_validation() {
        assert!(validate_category(None).is_none());
        assert!(validate_category(Some("")).is_none());
        assert!(validate_category(Some("   ")).is_none());
        assert!(validate_category(Some("a/b")).is_some());
        assert!(validate_category(Some("a\\b")).is_some());
        assert!(validate_category(Some("devops")).is_none());
        assert!(validate_category(Some("Bad")).is_some());
    }

    #[test]
    fn frontmatter_validation() {
        assert_eq!(validate_frontmatter(good_skill_content()), None);
        assert!(validate_frontmatter("").is_some());
        assert!(validate_frontmatter("no frontmatter").is_some());
        // Unclosed frontmatter.
        assert!(validate_frontmatter("---\nname: x\ndescription: y\n").is_some());
        // Missing name.
        assert!(validate_frontmatter("---\ndescription: y\n---\n\nbody\n").is_some());
        // Missing description.
        assert!(validate_frontmatter("---\nname: x\n---\n\nbody\n").is_some());
        // No body.
        assert!(validate_frontmatter("---\nname: x\ndescription: y\n---\n   \n").is_some());
        // Over-long description.
        let long = format!(
            "---\nname: x\ndescription: {}\n---\n\nbody\n",
            "z".repeat(MAX_DESCRIPTION_LENGTH + 1)
        );
        assert!(validate_frontmatter(&long).is_some());
    }

    #[test]
    fn content_size_validation() {
        assert!(validate_content_size("small", "SKILL.md").is_none());
        let big = "x".repeat(MAX_SKILL_CONTENT_CHARS + 1);
        assert!(validate_content_size(&big, "SKILL.md").is_some());
    }

    #[test]
    fn file_path_validation() {
        assert!(validate_file_path("").is_some());
        assert!(validate_file_path("../escape").is_some());
        assert!(validate_file_path("references/../x").is_some());
        assert!(validate_file_path("bogus/x.md").is_some());
        assert!(validate_file_path("references").is_some()); // dir only
        assert!(validate_file_path("references/api.md").is_none());
        assert!(validate_file_path("scripts/run.sh").is_none());
    }

    #[test]
    fn comma_sep_format() {
        assert_eq!(comma_sep(0), "0");
        assert_eq!(comma_sep(1000), "1,000");
        assert_eq!(comma_sep(100000), "100,000");
        assert_eq!(comma_sep(1048576), "1,048,576");
    }

    #[test]
    fn create_edit_patch_delete_roundtrip() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = temp_home();
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
        }

        let hooks = SkillManagerHooks::default();

        // Create.
        let r = create_skill("my-skill", good_skill_content(), None, &hooks);
        assert_eq!(r["success"], json!(true), "create failed: {r}");
        assert!(home.join("skills/my-skill/SKILL.md").exists());

        // Duplicate create blocked.
        let dup = create_skill("my-skill", good_skill_content(), None, &hooks);
        assert_eq!(dup["success"], json!(false));

        // Edit (full rewrite).
        let new_content = "---\nname: my-skill\ndescription: Updated\n---\n\nNew body.\n";
        let e = edit_skill("my-skill", new_content, &hooks);
        assert_eq!(e["success"], json!(true), "edit failed: {e}");
        let on_disk = fs::read_to_string(home.join("skills/my-skill/SKILL.md")).unwrap();
        assert!(on_disk.contains("New body."));

        // Patch the body.
        let p = patch_skill(
            "my-skill",
            "New body.",
            Some("Patched body."),
            None,
            false,
            &hooks,
        );
        assert_eq!(p["success"], json!(true), "patch failed: {p}");
        let on_disk = fs::read_to_string(home.join("skills/my-skill/SKILL.md")).unwrap();
        assert!(on_disk.contains("Patched body."));

        // write_file then remove_file.
        let w = write_file("my-skill", "references/a.md", "hello", &hooks);
        assert_eq!(w["success"], json!(true), "write_file failed: {w}");
        assert!(home.join("skills/my-skill/references/a.md").exists());

        let rm = remove_file("my-skill", "references/a.md");
        assert_eq!(rm["success"], json!(true), "remove_file failed: {rm}");
        assert!(!home.join("skills/my-skill/references/a.md").exists());
        // Empty subdir cleaned up.
        assert!(!home.join("skills/my-skill/references").exists());

        // Delete.
        let d = delete_skill("my-skill", Some(""), &hooks);
        assert_eq!(d["success"], json!(true), "delete failed: {d}");
        assert!(!home.join("skills/my-skill").exists());

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn create_with_category_and_cleanup_on_delete() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = temp_home();
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
        }
        let hooks = SkillManagerHooks::default();

        let r = create_skill("cat-skill", good_skill_content(), Some("devops"), &hooks);
        assert_eq!(r["success"], json!(true), "create failed: {r}");
        assert_eq!(r["category"], json!("devops"));
        assert!(home.join("skills/devops/cat-skill/SKILL.md").exists());

        let d = delete_skill("cat-skill", None, &hooks);
        assert_eq!(d["success"], json!(true));
        // Empty category dir removed.
        assert!(!home.join("skills/devops").exists());

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn pinned_guard_blocks_delete() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = temp_home();
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
        }
        let mut hooks = SkillManagerHooks::default();
        hooks.pinned_guard = Box::new(|n| Some(format!("Skill '{n}' is pinned")));

        let _ = create_skill("pinned-skill", good_skill_content(), None, &SkillManagerHooks::default());
        let d = delete_skill("pinned-skill", None, &hooks);
        assert_eq!(d["success"], json!(false));
        assert!(d["error"].as_str().unwrap().contains("pinned"));
        assert!(home.join("skills/pinned-skill").exists());

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn dispatch_missing_args() {
        let hooks = SkillManagerHooks::default();
        let args = SkillManageArgs {
            action: "create".to_string(),
            name: "x".to_string(),
            ..Default::default()
        };
        let out = skill_manage(&args, &hooks);
        assert!(out.contains("content is required"));

        let args = SkillManageArgs {
            action: "bogus".to_string(),
            name: "x".to_string(),
            ..Default::default()
        };
        let out = skill_manage(&args, &hooks);
        assert!(out.contains("Unknown action"));
    }

    #[test]
    fn delete_absorbed_into_self_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = temp_home();
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
        }
        let hooks = SkillManagerHooks::default();
        let _ = create_skill("solo", good_skill_content(), None, &hooks);
        let d = delete_skill("solo", Some("solo"), &hooks);
        assert_eq!(d["success"], json!(false));
        assert!(d["error"].as_str().unwrap().contains("cannot equal"));

        let d2 = delete_skill("solo", Some("does-not-exist"), &hooks);
        assert_eq!(d2["success"], json!(false));
        assert!(d2["error"].as_str().unwrap().contains("does not exist"));

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn schema_shape() {
        let s = skill_manage_schema();
        assert_eq!(s["name"], json!("skill_manage"));
        assert_eq!(s["parameters"]["required"], json!(["action", "name"]));
        assert!(s["parameters"]["properties"]["action"]["enum"].is_array());
    }

    #[test]
    fn args_from_json() {
        let a = SkillManageArgs::from_json(&json!({
            "action": "patch",
            "name": "foo",
            "old_string": "a",
            "new_string": "b",
            "replace_all": true,
        }));
        assert_eq!(a.action, "patch");
        assert_eq!(a.name, "foo");
        assert_eq!(a.old_string.as_deref(), Some("a"));
        assert!(a.replace_all);
        assert!(a.content.is_none());
    }
}
