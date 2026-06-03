//! Skills Sync -- Manifest-based seeding and updating of bundled skills.
//!
//! Faithful Rust port of `tools/skills_sync.py`.
//!
//! Copies bundled skills from the repo's `skills/` directory into
//! `~/.hermes/skills/` and uses a manifest to track which skills have been
//! synced and their origin hash.
//!
//! Manifest format (v2): each line is `skill_name:origin_hash` where
//! `origin_hash` is the MD5 of the bundled skill at the time it was last synced
//! to the user dir. Old v1 manifests (plain names without hashes) are
//! auto-migrated.
//!
//! Update logic:
//!   - NEW skills (not in manifest): copied to user dir, origin hash recorded.
//!   - EXISTING skills (in manifest, present in user dir):
//!       * If user copy matches origin hash: user hasn't modified it -> safe to
//!         update from bundled if bundled changed. New origin hash recorded.
//!       * If user copy differs from origin hash: user customized it -> SKIP.
//!   - DELETED by user (in manifest, absent from user dir): respected, not re-added.
//!   - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.
//!
//! The manifest lives at `~/.hermes/skills/.bundled_manifest`.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::atomic_replace;

/// Returns the user skills directory: `{HERMES_HOME}/skills`.
pub fn skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}

/// Returns the manifest file path: `{HERMES_HOME}/skills/.bundled_manifest`.
pub fn manifest_file() -> PathBuf {
    skills_dir().join(".bundled_manifest")
}

/// Result of a [`sync_skills`] run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncResult {
    pub copied: Vec<String>,
    pub updated: Vec<String>,
    pub skipped: u64,
    pub user_modified: Vec<String>,
    pub cleaned: Vec<String>,
    pub total_bundled: usize,
}

/// Result of a [`reset_bundled_skill`] call.
#[derive(Debug, Clone)]
pub struct ResetResult {
    pub ok: bool,
    /// One of: "manifest_cleared", "restored", "not_in_manifest", "bundled_missing".
    pub action: String,
    pub message: String,
    pub synced: Option<SyncResult>,
}

/// Locate the bundled `skills/` directory.
///
/// Checks `HERMES_BUNDLED_SKILLS` env var first (set by Nix wrapper), then
/// falls back to a relative path. Since there is no `__file__` in Rust, the
/// fallback resolves relative to the current executable's directory parent's
/// parent, mirroring `Path(__file__).parent.parent / "skills"`.
pub fn get_bundled_dir() -> PathBuf {
    if let Ok(env_override) = std::env::var("HERMES_BUNDLED_SKILLS") {
        if !env_override.is_empty() {
            return PathBuf::from(env_override);
        }
    }
    // Fallback: mimic `Path(__file__).parent.parent / "skills"`.
    // In Python __file__ is tools/skills_sync.py, so parent.parent is the repo
    // root, then / "skills". We approximate from the executable location.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent().and_then(|p| p.parent()) {
            return dir.join("skills");
        }
    }
    PathBuf::from("skills")
}

/// Read the manifest as a map of `{skill_name: origin_hash}`.
///
/// Handles both v1 (plain names) and v2 (`name:hash`) formats. v1 entries get an
/// empty hash string which triggers migration on next sync.
pub fn read_manifest() -> BTreeMap<String, String> {
    read_manifest_at(&manifest_file())
}

fn read_manifest_at(manifest: &Path) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    if !manifest.exists() {
        return result;
    }
    let content = match fs::read_to_string(manifest) {
        Ok(c) => c,
        Err(_) => return BTreeMap::new(),
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(idx) = line.find(':') {
            // v2 format: name:hash
            let name = line[..idx].trim();
            let hash_val = line[idx + 1..].trim();
            result.insert(name.to_string(), hash_val.to_string());
        } else {
            // v1 format: plain name -- empty hash triggers migration
            result.insert(line.to_string(), String::new());
        }
    }
    result
}

/// Write the manifest file atomically in v2 format (`name:hash`).
///
/// Uses a temp file + atomic replace to avoid corruption if the process crashes
/// or is interrupted mid-write. Errors are swallowed (logged at debug), matching
/// the Python behaviour.
pub fn write_manifest(entries: &BTreeMap<String, String>) {
    write_manifest_at(&manifest_file(), entries);
}

fn write_manifest_at(manifest: &Path, entries: &BTreeMap<String, String>) {
    let parent = match manifest.parent() {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from("."),
    };
    if let Err(e) = fs::create_dir_all(&parent) {
        log::debug!("Failed to create manifest dir {}: {e}", parent.display());
        return;
    }

    // entries is a BTreeMap so iteration is already sorted by key, matching
    // Python's `sorted(entries.items())`.
    let mut data = String::new();
    for (name, hash_val) in entries {
        data.push_str(name);
        data.push(':');
        data.push_str(hash_val);
        data.push('\n');
    }
    // Python joins with "\n" then appends a trailing "\n". For an empty map this
    // yields just "\n"; replicate that.
    if entries.is_empty() {
        data.push('\n');
    }

    if let Err(e) = atomic_write(manifest, &parent, data.as_bytes()) {
        log::debug!(
            "Failed to write skills manifest {}: {e}",
            manifest.display()
        );
    }
}

/// Write `data` to `target` atomically via a temp file in `dir`.
fn atomic_write(target: &Path, dir: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp_path = make_temp_path(dir, ".bundled_manifest_", ".tmp");
    let write_then_replace = || -> std::io::Result<()> {
        {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(data)?;
            f.flush()?;
            f.sync_all()?;
        }
        atomic_replace(&tmp_path, target)?;
        Ok(())
    };
    match write_then_replace() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Generate a non-colliding temp path inside `dir`. Mirrors `tempfile.mkstemp`
/// closely enough for our purposes.
fn make_temp_path(dir: &Path, prefix: &str, suffix: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let mut counter: u128 = 0;
    loop {
        let name = format!("{prefix}{pid}_{}_{counter}{suffix}", nanos.wrapping_add(counter));
        let candidate = dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

/// Read the `name` field from `SKILL.md` YAML frontmatter, falling back to
/// `fallback`.
pub fn read_skill_name(skill_md: &Path, fallback: &str) -> String {
    let raw = match fs::read(skill_md) {
        Ok(bytes) => bytes,
        Err(_) => return fallback.to_string(),
    };
    // Python: read_text(errors="replace")[:4000] -- decode lossily, then take
    // the first 4000 chars.
    let content_full = String::from_utf8_lossy(&raw);
    let content: String = content_full.chars().take(4000).collect();

    let mut in_frontmatter = false;
    for line in content.split('\n') {
        let stripped = line.trim();
        if stripped == "---" {
            if in_frontmatter {
                break;
            }
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter {
            if let Some(rest) = stripped.strip_prefix("name:") {
                let value = rest.trim().trim_matches(|c| c == '"' || c == '\'');
                if !value.is_empty() {
                    return value.to_string();
                }
            }
        }
    }
    fallback.to_string()
}

/// A discovered bundled skill: its name and the directory containing `SKILL.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundledSkill {
    pub name: String,
    pub dir: PathBuf,
}

/// Find all `SKILL.md` files in the bundled directory.
///
/// Returns list of `(skill_name, skill_directory_path)` entries. Skips paths
/// containing `/.git/`, `/.github/` or `/.hub/`.
pub fn discover_bundled_skills(bundled_dir: &Path) -> Vec<BundledSkill> {
    let mut skills = Vec::new();
    if !bundled_dir.exists() {
        return skills;
    }
    let mut found: Vec<PathBuf> = Vec::new();
    rglob_skill_md(bundled_dir, &mut found);
    // Sorting is not required for parity (Python rglob order is arbitrary) but
    // keeps results deterministic.
    found.sort();
    for skill_md in found {
        let path_str = skill_md.to_string_lossy().replace('\\', "/");
        if path_str.contains("/.git/")
            || path_str.contains("/.github/")
            || path_str.contains("/.hub/")
        {
            continue;
        }
        let skill_dir = match skill_md.parent() {
            Some(p) => p.to_path_buf(),
            None => continue,
        };
        let fallback = skill_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let skill_name = read_skill_name(&skill_md, &fallback);
        skills.push(BundledSkill {
            name: skill_name,
            dir: skill_dir,
        });
    }
    skills
}

/// Recursively collect all files named `SKILL.md` under `root`.
fn rglob_skill_md(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if ft.is_dir() {
            rglob_skill_md(&path, out);
        } else if ft.is_file() {
            if path.file_name().map(|n| n == "SKILL.md").unwrap_or(false) {
                out.push(path);
            }
        }
    }
}

/// Compute the destination path in [`skills_dir`] preserving the category
/// structure. e.g. `bundled/skills/mlops/axolotl` -> `~/.hermes/skills/mlops/axolotl`.
pub fn compute_relative_dest(skill_dir: &Path, bundled_dir: &Path) -> PathBuf {
    match skill_dir.strip_prefix(bundled_dir) {
        Ok(rel) => skills_dir().join(rel),
        // Python's relative_to raises if not a prefix; we mirror that loosely by
        // falling back to joining the full dir name. This should not happen in
        // practice since skill_dir always lives under bundled_dir.
        Err(_) => skills_dir().join(skill_dir.file_name().unwrap_or_default()),
    }
}

/// Compute an MD5 hash of all file contents in a directory for change detection.
///
/// Iterates files in sorted order, hashing the relative path bytes followed by
/// the file bytes -- byte-identical to the Python implementation.
pub fn dir_hash(directory: &Path) -> String {
    let mut hasher = md5::Context::new();
    let mut files: Vec<PathBuf> = Vec::new();
    rglob_all(directory, &mut files);
    files.sort();
    for fpath in files {
        // Only hash regular files (Python checks fpath.is_file()).
        match fs::metadata(&fpath) {
            Ok(m) if m.is_file() => {}
            _ => continue,
        }
        let rel = match fpath.strip_prefix(directory) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Python uses str(rel) -- platform path separator. On unix this is '/'.
        let rel_str = rel.to_string_lossy().to_string();
        hasher.consume(rel_str.as_bytes());
        if let Ok(bytes) = fs::read(&fpath) {
            hasher.consume(&bytes);
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Recursively collect every path under `root` (files and dirs), like
/// `rglob("*")`.
fn rglob_all(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        out.push(path.clone());
        if ft.is_dir() {
            rglob_all(&path, out);
        }
    }
}

/// Recursively copy a directory tree (like `shutil.copytree` where dest must not
/// pre-exist).
fn copytree(src: &Path, dest: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        let target = dest.join(entry.file_name());
        if ft.is_dir() {
            copytree(&path, &target)?;
        } else if ft.is_symlink() {
            // Preserve symlink semantics loosely: copy the link target's bytes.
            // copytree by default copies symlinks as the files they point to.
            fs::copy(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Path with `.bak` suffix appended (like `Path.with_suffix(".bak")`).
fn with_bak_suffix(p: &Path) -> PathBuf {
    let mut owned = p.to_path_buf();
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    owned.set_file_name(format!("{stem}.bak"));
    owned
}

/// Sync bundled skills into `~/.hermes/skills/` using the manifest.
///
/// When `quiet` is false, progress lines are printed to stdout (matching the
/// Python CLI behaviour).
pub fn sync_skills(quiet: bool) -> SyncResult {
    let bundled_dir = get_bundled_dir();
    if !bundled_dir.exists() {
        return SyncResult::default();
    }

    let s_dir = skills_dir();
    let _ = fs::create_dir_all(&s_dir);
    let mut manifest = read_manifest();
    let bundled_skills = discover_bundled_skills(&bundled_dir);
    let bundled_names: BTreeSet<String> =
        bundled_skills.iter().map(|s| s.name.clone()).collect();

    let mut copied: Vec<String> = Vec::new();
    let mut updated: Vec<String> = Vec::new();
    let mut user_modified: Vec<String> = Vec::new();
    let mut skipped: u64 = 0;

    for skill in &bundled_skills {
        let skill_name = &skill.name;
        let skill_src = &skill.dir;
        let dest = compute_relative_dest(skill_src, &bundled_dir);
        let bundled_hash = dir_hash(skill_src);

        if !manifest.contains_key(skill_name) {
            // New skill -- never offered before.
            if dest.exists() {
                // User already has a skill with the same name -- don't overwrite.
                skipped += 1;
                if dir_hash(&dest) == bundled_hash {
                    manifest.insert(skill_name.clone(), bundled_hash.clone());
                } else if !quiet {
                    println!(
                        "  \u{26a0} {skill_name}: bundled version shipped but you \
                         already have a local skill by this name -- yours \
                         was kept. Run `hermes skills reset {skill_name}` \
                         to replace it with the bundled version."
                    );
                }
            } else {
                let copy_result = (|| -> std::io::Result<()> {
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    copytree(skill_src, &dest)
                })();
                match copy_result {
                    Ok(()) => {
                        copied.push(skill_name.clone());
                        manifest.insert(skill_name.clone(), bundled_hash.clone());
                        if !quiet {
                            println!("  + {skill_name}");
                        }
                    }
                    Err(e) => {
                        if !quiet {
                            println!("  ! Failed to copy {skill_name}: {e}");
                        }
                        // Do NOT add to manifest -- next sync should retry.
                    }
                }
            }
        } else if dest.exists() {
            // Existing skill -- in manifest AND on disk.
            let origin_hash = manifest.get(skill_name).cloned().unwrap_or_default();
            let user_hash = dir_hash(&dest);

            if origin_hash.is_empty() {
                // v1 migration: no origin hash recorded. Set baseline from the
                // user's current copy so future syncs can detect modifications.
                manifest.insert(skill_name.clone(), user_hash.clone());
                // Both branches in Python increment skipped.
                skipped += 1;
                continue;
            }

            if user_hash != origin_hash {
                // User modified this skill -- don't overwrite their changes.
                user_modified.push(skill_name.clone());
                if !quiet {
                    println!("  ~ {skill_name} (user-modified, skipping)");
                }
                continue;
            }

            // User copy matches origin -- check if bundled has a newer version.
            if bundled_hash != origin_hash {
                let backup = with_bak_suffix(&dest);
                // Move old copy to a backup so we can restore on failure.
                match fs::rename(&dest, &backup) {
                    Ok(()) => {
                        match copytree(skill_src, &dest) {
                            Ok(()) => {
                                manifest
                                    .insert(skill_name.clone(), bundled_hash.clone());
                                updated.push(skill_name.clone());
                                if !quiet {
                                    println!("  \u{2191} {skill_name} (updated)");
                                }
                                // Remove backup after successful copy.
                                let _ = remove_dir_all_ignore(&backup);
                            }
                            Err(e) => {
                                // Restore from backup.
                                if backup.exists() && !dest.exists() {
                                    let _ = fs::rename(&backup, &dest);
                                }
                                if !quiet {
                                    println!("  ! Failed to update {skill_name}: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if !quiet {
                            println!("  ! Failed to update {skill_name}: {e}");
                        }
                    }
                }
            } else {
                // bundled unchanged, user unchanged.
                skipped += 1;
            }
        } else {
            // In manifest but not on disk -- user deleted it.
            skipped += 1;
        }
    }

    // Clean stale manifest entries (skills removed from bundled dir).
    let manifest_keys: BTreeSet<String> = manifest.keys().cloned().collect();
    let cleaned: Vec<String> = manifest_keys
        .difference(&bundled_names)
        .cloned()
        .collect();
    for name in &cleaned {
        manifest.remove(name);
    }

    // Also copy DESCRIPTION.md files for categories (if not already present).
    let mut desc_files: Vec<PathBuf> = Vec::new();
    rglob_description_md(&bundled_dir, &mut desc_files);
    desc_files.sort();
    for desc_md in desc_files {
        let rel = match desc_md.strip_prefix(&bundled_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let dest_desc = s_dir.join(rel);
        if !dest_desc.exists() {
            let r = (|| -> std::io::Result<()> {
                if let Some(parent) = dest_desc.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&desc_md, &dest_desc)?;
                Ok(())
            })();
            if let Err(e) = r {
                log::debug!("Could not copy {}: {e}", desc_md.display());
            }
        }
    }

    write_manifest(&manifest);

    SyncResult {
        copied,
        updated,
        skipped,
        user_modified,
        cleaned,
        total_bundled: bundled_skills.len(),
    }
}

/// Recursively collect every `DESCRIPTION.md` under `root`.
fn rglob_description_md(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if ft.is_dir() {
            rglob_description_md(&path, out);
        } else if ft.is_file()
            && path
                .file_name()
                .map(|n| n == "DESCRIPTION.md")
                .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Recursively remove a directory tree, ignoring errors (like
/// `shutil.rmtree(..., ignore_errors=True)`).
fn remove_dir_all_ignore(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)
    } else if path.exists() {
        fs::remove_file(path)
    } else {
        Ok(())
    }
}

/// Reset a bundled skill's manifest tracking so future syncs work normally.
///
/// When `restore` is true, also deletes the user's copy in [`skills_dir`] and
/// lets the next sync re-copy the current bundled version. When false (default),
/// only clears the manifest entry -- the user's current copy is preserved but
/// future updates work again.
pub fn reset_bundled_skill(name: &str, restore: bool) -> ResetResult {
    let mut manifest = read_manifest();
    let bundled_dir = get_bundled_dir();
    let bundled_skills = discover_bundled_skills(&bundled_dir);
    let bundled_by_name: BTreeMap<String, PathBuf> = bundled_skills
        .iter()
        .map(|s| (s.name.clone(), s.dir.clone()))
        .collect();

    let in_manifest = manifest.contains_key(name);
    let is_bundled = bundled_by_name.contains_key(name);

    if !in_manifest && !is_bundled {
        return ResetResult {
            ok: false,
            action: "not_in_manifest".to_string(),
            message: format!(
                "'{name}' is not a tracked bundled skill. Nothing to reset. \
                 (Hub-installed skills use `hermes skills uninstall`.)"
            ),
            synced: None,
        };
    }

    // Step 1: drop the manifest entry so next sync treats it as new.
    if in_manifest {
        manifest.remove(name);
        write_manifest(&manifest);
    }

    // Step 2 (optional): delete the user's copy so next sync re-copies bundled.
    let mut deleted_user_copy = false;
    if restore {
        if !is_bundled {
            return ResetResult {
                ok: false,
                action: "bundled_missing".to_string(),
                message: format!(
                    "'{name}' has no bundled source -- manifest entry cleared \
                     but cannot restore from bundled (skill was removed upstream)."
                ),
                synced: None,
            };
        }
        let dest = compute_relative_dest(&bundled_by_name[name], &bundled_dir);
        if dest.exists() {
            match fs::remove_dir_all(&dest) {
                Ok(()) => deleted_user_copy = true,
                Err(e) => {
                    return ResetResult {
                        ok: false,
                        action: "manifest_cleared".to_string(),
                        message: format!(
                            "Cleared manifest entry for '{name}' but could not \
                             delete user copy at {}: {e}",
                            dest.display()
                        ),
                        synced: None,
                    };
                }
            }
        }
    }

    // Step 3: run sync to re-baseline (or re-copy if we deleted).
    let synced = sync_skills(true);

    let (action, message) = if restore && deleted_user_copy {
        (
            "restored".to_string(),
            format!("Restored '{name}' from bundled source."),
        )
    } else if restore {
        (
            "restored".to_string(),
            format!("Restored '{name}' (no prior user copy, re-copied from bundled)."),
        )
    } else {
        (
            "manifest_cleared".to_string(),
            format!(
                "Cleared manifest entry for '{name}'. Future `hermes update` runs \
                 will re-baseline against your current copy and accept upstream changes."
            ),
        )
    };

    ResetResult {
        ok: true,
        action,
        message,
        synced: Some(synced),
    }
}

/// CLI-style entrypoint mirroring `if __name__ == "__main__"`.
///
/// Runs a non-quiet sync and prints a summary line. Returns the result.
pub fn run_main() -> SyncResult {
    println!("Syncing bundled skills into ~/.hermes/skills/ ...");
    let result = sync_skills(false);
    let mut parts = vec![
        format!("{} new", result.copied.len()),
        format!("{} updated", result.updated.len()),
        format!("{} unchanged", result.skipped),
    ];
    if !result.user_modified.is_empty() {
        parts.push(format!("{} user-modified (kept)", result.user_modified.len()));
    }
    if !result.cleaned.is_empty() {
        parts.push(format!("{} cleaned from manifest", result.cleaned.len()));
    }
    println!(
        "\nDone: {}. {} total bundled.",
        parts.join(", "),
        result.total_bundled
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise tests that mutate HERMES_HOME / HERMES_BUNDLED_SKILLS.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("hermes_skills_sync_{tag}_{}_{nanos}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            TempDir { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(p: &Path, content: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    fn make_bundled_skill(bundled: &Path, rel: &str, name: &str, body: &str) {
        let skill_dir = bundled.join(rel);
        fs::create_dir_all(&skill_dir).unwrap();
        let md = format!("---\nname: {name}\n---\n{body}\n");
        write_file(&skill_dir.join("SKILL.md"), &md);
    }

    #[test]
    fn test_read_manifest_v1_and_v2() {
        let td = TempDir::new("manifest");
        let mf = td.path.join(".bundled_manifest");
        write_file(&mf, "alpha:abc123\nbeta\n  \ngamma:deadbeef\n");
        let m = read_manifest_at(&mf);
        assert_eq!(m.get("alpha").unwrap(), "abc123");
        assert_eq!(m.get("beta").unwrap(), ""); // v1 -> empty
        assert_eq!(m.get("gamma").unwrap(), "deadbeef");
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_read_manifest_missing() {
        let td = TempDir::new("manifest_missing");
        let mf = td.path.join("nope");
        assert!(read_manifest_at(&mf).is_empty());
    }

    #[test]
    fn test_write_then_read_roundtrip_sorted() {
        let td = TempDir::new("roundtrip");
        let mf = td.path.join(".bundled_manifest");
        let mut entries = BTreeMap::new();
        entries.insert("zeta".to_string(), "h1".to_string());
        entries.insert("alpha".to_string(), "h2".to_string());
        write_manifest_at(&mf, &entries);
        let content = fs::read_to_string(&mf).unwrap();
        // Sorted: alpha before zeta.
        assert_eq!(content, "alpha:h2\nzeta:h1\n");
        let back = read_manifest_at(&mf);
        assert_eq!(back, entries);
    }

    #[test]
    fn test_write_empty_manifest() {
        let td = TempDir::new("empty");
        let mf = td.path.join(".bundled_manifest");
        let entries = BTreeMap::new();
        write_manifest_at(&mf, &entries);
        let content = fs::read_to_string(&mf).unwrap();
        assert_eq!(content, "\n");
    }

    #[test]
    fn test_read_skill_name_frontmatter() {
        let td = TempDir::new("skillname");
        let md = td.path.join("SKILL.md");
        write_file(&md, "---\nname: \"My Skill\"\ndescription: x\n---\nbody");
        assert_eq!(read_skill_name(&md, "fallback"), "My Skill");
    }

    #[test]
    fn test_read_skill_name_no_name_uses_fallback() {
        let td = TempDir::new("skillname2");
        let md = td.path.join("SKILL.md");
        write_file(&md, "---\ndescription: x\n---\nbody");
        assert_eq!(read_skill_name(&md, "fallback"), "fallback");
    }

    #[test]
    fn test_read_skill_name_single_quotes() {
        let td = TempDir::new("skillname3");
        let md = td.path.join("SKILL.md");
        write_file(&md, "---\nname: 'quoted'\n---\n");
        assert_eq!(read_skill_name(&md, "fb"), "quoted");
    }

    #[test]
    fn test_dir_hash_deterministic_and_sensitive() {
        let td = TempDir::new("dirhash");
        let d1 = td.path.join("a");
        fs::create_dir_all(&d1).unwrap();
        write_file(&d1.join("x.txt"), "hello");
        write_file(&d1.join("sub/y.txt"), "world");
        let h1 = dir_hash(&d1);

        let d2 = td.path.join("b");
        fs::create_dir_all(&d2).unwrap();
        write_file(&d2.join("x.txt"), "hello");
        write_file(&d2.join("sub/y.txt"), "world");
        let h2 = dir_hash(&d2);
        assert_eq!(h1, h2);

        write_file(&d2.join("x.txt"), "changed");
        let h3 = dir_hash(&d2);
        assert_ne!(h1, h3);
        // MD5 hex is 32 chars.
        assert_eq!(h1.len(), 32);
    }

    #[test]
    fn test_discover_bundled_skills_skips_dotdirs() {
        let td = TempDir::new("discover");
        let bundled = td.path.join("skills");
        make_bundled_skill(&bundled, "cat/alpha", "alpha", "a");
        make_bundled_skill(&bundled, "cat/beta", "beta", "b");
        // Should be skipped:
        make_bundled_skill(&bundled, ".git/ghost", "ghost", "g");
        make_bundled_skill(&bundled, ".hub/hubskill", "hubskill", "h");

        let found = discover_bundled_skills(&bundled);
        let names: BTreeSet<String> = found.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains("alpha"));
        assert!(names.contains("beta"));
        assert!(!names.contains("ghost"));
        assert!(!names.contains("hubskill"));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_full_sync_lifecycle() {
        let _guard = ENV_LOCK.lock().unwrap();
        let td = TempDir::new("lifecycle");
        let home = td.path.join("home");
        let bundled = td.path.join("bundled_skills");
        make_bundled_skill(&bundled, "cat/alpha", "alpha", "v1");
        make_bundled_skill(&bundled, "cat/beta", "beta", "v1");
        // a category description
        write_file(&bundled.join("cat/DESCRIPTION.md"), "category desc");

        unsafe {
            std::env::set_var("HERMES_HOME", &home);
            std::env::set_var("HERMES_BUNDLED_SKILLS", &bundled);
        }

        // First sync: both skills are new -> copied.
        let r1 = sync_skills(true);
        assert_eq!(r1.copied.len(), 2);
        assert_eq!(r1.total_bundled, 2);
        assert!(home.join("skills/cat/alpha/SKILL.md").exists());
        assert!(home.join("skills/cat/DESCRIPTION.md").exists());

        // Second sync: nothing changed -> all skipped.
        let r2 = sync_skills(true);
        assert!(r2.copied.is_empty());
        assert!(r2.updated.is_empty());
        assert_eq!(r2.skipped, 2);

        // Modify bundled alpha -> should update on next sync.
        make_bundled_skill(&bundled, "cat/alpha", "alpha", "v2-bundled");
        let r3 = sync_skills(true);
        assert_eq!(r3.updated, vec!["alpha".to_string()]);
        // beta unchanged.
        assert_eq!(r3.skipped, 1);

        // User modifies beta -> detected as user_modified, not overwritten.
        write_file(&home.join("skills/cat/beta/SKILL.md"), "user hacked beta");
        make_bundled_skill(&bundled, "cat/beta", "beta", "v2-bundled");
        let r4 = sync_skills(true);
        assert_eq!(r4.user_modified, vec!["beta".to_string()]);
        assert_eq!(
            fs::read_to_string(home.join("skills/cat/beta/SKILL.md")).unwrap(),
            "user hacked beta"
        );

        // Remove alpha from bundled -> cleaned from manifest.
        fs::remove_dir_all(bundled.join("cat/alpha")).unwrap();
        let r5 = sync_skills(true);
        assert!(r5.cleaned.contains(&"alpha".to_string()));

        unsafe {
            std::env::remove_var("HERMES_HOME");
            std::env::remove_var("HERMES_BUNDLED_SKILLS");
        }
    }

    #[test]
    fn test_reset_not_tracked() {
        let _guard = ENV_LOCK.lock().unwrap();
        let td = TempDir::new("reset_nt");
        let home = td.path.join("home");
        let bundled = td.path.join("bundled_skills");
        fs::create_dir_all(&bundled).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
            std::env::set_var("HERMES_BUNDLED_SKILLS", &bundled);
        }
        let r = reset_bundled_skill("ghost", false);
        assert!(!r.ok);
        assert_eq!(r.action, "not_in_manifest");
        unsafe {
            std::env::remove_var("HERMES_HOME");
            std::env::remove_var("HERMES_BUNDLED_SKILLS");
        }
    }

    #[test]
    fn test_reset_restore() {
        let _guard = ENV_LOCK.lock().unwrap();
        let td = TempDir::new("reset_restore");
        let home = td.path.join("home");
        let bundled = td.path.join("bundled_skills");
        make_bundled_skill(&bundled, "cat/alpha", "alpha", "v1");
        unsafe {
            std::env::set_var("HERMES_HOME", &home);
            std::env::set_var("HERMES_BUNDLED_SKILLS", &bundled);
        }
        // Initial sync copies alpha.
        sync_skills(true);
        // User edits it.
        write_file(&home.join("skills/cat/alpha/SKILL.md"), "edited");
        // Reset with restore -> delete user copy and re-sync from bundled.
        let r = reset_bundled_skill("alpha", true);
        assert!(r.ok);
        assert_eq!(r.action, "restored");
        let content = fs::read_to_string(home.join("skills/cat/alpha/SKILL.md")).unwrap();
        assert!(content.contains("name: alpha"));
        assert!(!content.contains("edited"));
        unsafe {
            std::env::remove_var("HERMES_HOME");
            std::env::remove_var("HERMES_BUNDLED_SKILLS");
        }
    }

    #[test]
    fn test_with_bak_suffix() {
        let p = Path::new("/tmp/foo/alpha");
        assert_eq!(with_bak_suffix(p), PathBuf::from("/tmp/foo/alpha.bak"));
    }
}
