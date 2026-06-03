//! Skill usage telemetry + provenance tracking for the Curator feature.
//!
//! Native Rust port of `tools/skill_usage.py`.
//!
//! Tracks per-skill usage metadata in a sidecar JSON file
//! (`~/.hermes/skills/.usage.json`) keyed by skill name. Counters are bumped by
//! the existing skill tools (skill_view, skill_manage); the curator orchestrator
//! reads the derived activity timestamp to decide lifecycle transitions.
//!
//! Design notes:
//!   - Sidecar, not frontmatter. Keeps operational telemetry out of
//!     user-authored SKILL.md content.
//!   - Atomic writes via tempfile + rename (same pattern as `.bundled_manifest`).
//!   - All counter bumps are best-effort: failures log at debug and return
//!     silently. A broken sidecar never breaks the underlying tool call.
//!   - Provenance filter: curator-managed skills are explicitly marked when
//!     created through skill_manage. Bundled / hub-installed skills stay
//!     off-limits, and manually authored skills are not inferred from location.
//!
//! Lifecycle states:
//!     active    -> default
//!     stale     -> unused > stale_after_days (config)
//!     archived  -> unused > archive_after_days (config); moved to .archive/
//!     pinned    -> opt-out from auto transitions (boolean flag, orthogonal)

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset, Utc};
use serde_json::{Map, Value};

use crate::mod_hermes_constants::get_hermes_home;

pub const STATE_ACTIVE: &str = "active";
pub const STATE_STALE: &str = "stale";
pub const STATE_ARCHIVED: &str = "archived";

fn valid_states() -> HashSet<&'static str> {
    [STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED].into_iter().collect()
}

/// A single skill's usage record (`Map<String, Value>` mirrors the Python dict).
pub type Record = Map<String, Value>;
/// The full sidecar map: skill name -> record.
pub type UsageMap = BTreeMap<String, Record>;

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

pub fn skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}

fn usage_file() -> PathBuf {
    skills_dir().join(".usage.json")
}

fn archive_dir() -> PathBuf {
    skills_dir().join(".archive")
}

fn now_iso() -> String {
    // datetime.now(timezone.utc).isoformat() -> e.g. 2026-06-03T12:34:56.789012+00:00
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

/// Parse an ISO timestamp defensively for activity comparisons.
fn parse_iso_timestamp(value: &Value) -> Option<DateTime<FixedOffset>> {
    let s = match value {
        Value::String(s) if !s.is_empty() => s.clone(),
        Value::Null => return None,
        Value::String(_) => return None,
        // Mirror Python's `str(value)` coercion for non-strings, but `not value`
        // semantics: numeric 0 / false are falsy.
        Value::Number(n) => {
            if n.as_f64() == Some(0.0) {
                return None;
            }
            n.to_string()
        }
        Value::Bool(false) => return None,
        Value::Bool(true) => "true".to_string(),
        Value::Array(a) if a.is_empty() => return None,
        Value::Object(o) if o.is_empty() => return None,
        other => other.to_string(),
    };
    // Try RFC3339 (handles explicit offsets and 'Z').
    if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
        return Some(dt);
    }
    // Python's datetime.fromisoformat accepts a space separator and naive
    // timestamps (assumed UTC). Try a few common shapes.
    let normalized = s.replacen(' ', "T", 1);
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Some(dt);
    }
    // Naive: no timezone -> assume UTC (matches tzinfo backfill in Python).
    let utc = FixedOffset::east_opt(0).unwrap();
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(&normalized, fmt) {
            return Some(DateTime::<FixedOffset>::from_naive_utc_and_offset(naive, utc));
        }
    }
    None
}

/// Return the newest actual activity timestamp for a usage record.
///
/// "Activity" means a skill was used, viewed, or patched. Creation time is
/// intentionally excluded so callers can still distinguish never-active skills.
pub fn latest_activity_at(record: &Record) -> Option<String> {
    let mut latest_dt: Option<DateTime<FixedOffset>> = None;
    let mut latest_raw: Option<String> = None;
    for key in ["last_used_at", "last_viewed_at", "last_patched_at"] {
        let raw = match record.get(key) {
            Some(v) => v,
            None => continue,
        };
        let dt = match parse_iso_timestamp(raw) {
            Some(d) => d,
            None => continue,
        };
        if latest_dt.is_none() || dt > latest_dt.unwrap() {
            latest_dt = Some(dt);
            latest_raw = Some(value_to_str(raw));
        }
    }
    latest_raw
}

/// Mirror Python `str(value)` for the raw stored timestamp.
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Coerce a JSON value to int the way `int(record.get(key) or 0)` would behave.
fn value_to_int(v: Option<&Value>) -> Option<i64> {
    match v {
        None | Some(Value::Null) => Some(0),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                // float -> int truncates toward zero (Python int(float)).
                n.as_f64().map(|f| f.trunc() as i64)
            }
        }
        Some(Value::Bool(b)) => Some(if *b { 1 } else { 0 }),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0)
            } else {
                t.parse::<i64>().ok().or_else(|| {
                    t.parse::<f64>().ok().map(|f| f.trunc() as i64)
                })
            }
        }
        _ => None,
    }
}

/// Return the total observed activity count across use/view/patch events.
pub fn activity_count(record: &Record) -> i64 {
    let mut total = 0i64;
    for key in ["use_count", "view_count", "patch_count"] {
        // `or 0` makes falsy values (None, 0, "", false) become 0; only on a
        // genuine parse failure (TypeError/ValueError) do we skip.
        let raw = record.get(key);
        // Replicate `int(record.get(key) or 0)`: if the value is falsy, use 0.
        let coerced = match raw {
            None | Some(Value::Null) | Some(Value::Bool(false)) => Some(0),
            Some(Value::Number(n)) if n.as_f64() == Some(0.0) => Some(0),
            Some(Value::String(s)) if s.is_empty() => Some(0),
            Some(Value::Array(a)) if a.is_empty() => Some(0),
            Some(Value::Object(o)) if o.is_empty() => Some(0),
            other => value_to_int(other),
        };
        if let Some(n) = coerced {
            total += n;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Provenance — which skills are agent-created (and thus eligible for curation)
// ---------------------------------------------------------------------------

/// Return the set of skill names that were seeded from the bundled repo.
///
/// Reads `~/.hermes/skills/.bundled_manifest` (format: "name:hash" per line).
fn read_bundled_manifest_names() -> HashSet<String> {
    let manifest = skills_dir().join(".bundled_manifest");
    if !manifest.exists() {
        return HashSet::new();
    }
    let mut names = HashSet::new();
    match fs::read_to_string(&manifest) {
        Ok(text) => {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let name = line.split(':').next().unwrap_or("").trim();
                if !name.is_empty() {
                    names.insert(name.to_string());
                }
            }
        }
        Err(e) => {
            log::debug!("Failed to read bundled manifest: {e}");
        }
    }
    names
}

/// Return the set of skill names installed via the Skills Hub.
///
/// Reads `~/.hermes/skills/.hub/lock.json`.
fn read_hub_installed_names() -> HashSet<String> {
    let lock_path = skills_dir().join(".hub").join("lock.json");
    if !lock_path.exists() {
        return HashSet::new();
    }
    let text = match fs::read_to_string(&lock_path) {
        Ok(t) => t,
        Err(e) => {
            log::debug!("Failed to read hub lock file: {e}");
            return HashSet::new();
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(e) => {
            log::debug!("Failed to read hub lock file: {e}");
            return HashSet::new();
        }
    };
    let mut names: HashSet<String> = HashSet::new();
    if let Value::Object(obj) = &data {
        let installed = match obj.get("installed") {
            Some(Value::Object(m)) => m,
            _ => return names,
        };
        let skills_dir = skills_dir();
        for k in installed.keys() {
            names.insert(k.clone());
        }
        for entry in installed.values() {
            let entry = match entry {
                Value::Object(m) => m,
                _ => continue,
            };
            let install_path = match entry.get("install_path") {
                Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
                _ => continue,
            };
            let mut skill_dir = PathBuf::from(&install_path);
            if !skill_dir.is_absolute() {
                skill_dir = skills_dir.join(&skill_dir);
            }
            // resolve() + relative_to(skills_dir.resolve()) — must stay inside.
            let resolved = match fs::canonicalize(&skill_dir) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let base_resolved = match fs::canonicalize(&skills_dir) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if !resolved.starts_with(&base_resolved) {
                continue;
            }
            let skill_md = resolved.join("SKILL.md");
            if skill_md.exists() {
                let fallback = resolved
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                names.insert(read_skill_name(&skill_md, &fallback));
            }
        }
    }
    names
}

/// Enumerate skills explicitly authored by the agent.
///
/// The curator operates exclusively on this set. Skills are only eligible after
/// `skill_manage(action="create")` marks them in `.usage.json`; manually
/// authored skills must not be inferred from filesystem location.
pub fn list_agent_created_skill_names() -> Vec<String> {
    let base = skills_dir();
    if !base.exists() {
        return Vec::new();
    }
    let bundled = read_bundled_manifest_names();
    let hub = read_hub_installed_names();
    let mut off_limits = bundled;
    off_limits.extend(hub);
    let usage = load_usage();

    let mut names: HashSet<String> = HashSet::new();
    for skill_md in rglob_skill_md(&base) {
        let rel = match skill_md.strip_prefix(&base) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let parts: Vec<&str> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        if let Some(first) = parts.first() {
            if first.starts_with('.') || *first == "node_modules" {
                continue;
            }
        }
        let fallback = skill_md
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = read_skill_name(&skill_md, &fallback);
        if off_limits.contains(&name) {
            continue;
        }
        if !is_curator_managed_record(usage.get(&name).map(|r| r as &Record)) {
            continue;
        }
        names.push_unique(name);
    }
    let mut out: Vec<String> = names.into_iter().collect();
    out.sort();
    out
}

trait PushUnique {
    fn push_unique(&mut self, v: String);
}
impl PushUnique for HashSet<String> {
    fn push_unique(&mut self, v: String) {
        self.insert(v);
    }
}

/// Recursively collect all `SKILL.md` paths under `base` (mirrors `rglob`).
fn rglob_skill_md(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                walk(&path, out);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                out.push(path);
            }
        }
    }
    walk(base, &mut out);
    out
}

/// Parse the `name:` field from a SKILL.md YAML frontmatter.
fn read_skill_name(skill_md: &Path, fallback: &str) -> String {
    let text = match fs::read_to_string(skill_md) {
        Ok(t) => t,
        Err(_) => {
            // Python uses errors="replace"; a true read failure returns fallback.
            // read_to_string fails on invalid utf-8, so fall back to lossy read.
            match fs::read(skill_md) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(_) => return fallback.to_string(),
            }
        }
    };
    let text: String = text.chars().take(4000).collect();
    let mut in_frontmatter = false;
    for line in text.split('\n') {
        let stripped = line.trim();
        if stripped == "---" {
            if in_frontmatter {
                break;
            }
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter && stripped.starts_with("name:") {
            let value = stripped.splitn(2, ':').nth(1).unwrap_or("").trim();
            let value = value.trim_matches(|c| c == '"' || c == '\'');
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    fallback.to_string()
}

/// Whether *skill_name* is neither bundled nor hub-installed.
pub fn is_agent_created(skill_name: &str) -> bool {
    let mut off_limits = read_bundled_manifest_names();
    off_limits.extend(read_hub_installed_names());
    !off_limits.contains(skill_name)
}

/// Return True when a usage record opts a skill into curator management.
fn is_curator_managed_record(record: Option<&Record>) -> bool {
    let rec = match record {
        Some(r) => r,
        None => return false,
    };
    rec.get("created_by") == Some(&Value::String("agent".to_string()))
        || rec.get("agent_created") == Some(&Value::Bool(true))
}

// ---------------------------------------------------------------------------
// Sidecar I/O
// ---------------------------------------------------------------------------

fn empty_record() -> Record {
    let mut m = Map::new();
    m.insert("created_by".into(), Value::Null);
    m.insert("use_count".into(), Value::from(0));
    m.insert("view_count".into(), Value::from(0));
    m.insert("last_used_at".into(), Value::Null);
    m.insert("last_viewed_at".into(), Value::Null);
    m.insert("patch_count".into(), Value::from(0));
    m.insert("last_patched_at".into(), Value::Null);
    m.insert("created_at".into(), Value::String(now_iso()));
    m.insert("state".into(), Value::String(STATE_ACTIVE.to_string()));
    m.insert("pinned".into(), Value::Bool(false));
    m.insert("archived_at".into(), Value::Null);
    m
}

/// Read the entire `.usage.json` map. Returns empty map on missing/corrupt.
pub fn load_usage() -> UsageMap {
    let path = usage_file();
    if !path.exists() {
        return UsageMap::new();
    }
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            log::debug!("Failed to read {}: {e}", path.display());
            return UsageMap::new();
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(e) => {
            log::debug!("Failed to read {}: {e}", path.display());
            return UsageMap::new();
        }
    };
    let obj = match data {
        Value::Object(o) => o,
        _ => return UsageMap::new(),
    };
    // Defensive: keep only dict values.
    let mut clean = UsageMap::new();
    for (k, v) in obj {
        if let Value::Object(rec) = v {
            clean.insert(k, rec);
        }
    }
    clean
}

/// Write the usage map atomically. Best-effort — errors are logged, not raised.
pub fn save_usage(data: &UsageMap) {
    let path = usage_file();
    if let Err(e) = save_usage_inner(&path, data) {
        log::debug!("Failed to write {}: {e}", path.display());
    }
}

fn save_usage_inner(path: &Path, data: &UsageMap) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    // json.dump(..., indent=2, sort_keys=True, ensure_ascii=False)
    // BTreeMap already gives sorted keys; serde_json sorts object keys when the
    // `preserve_order` feature is off. To guarantee sort_keys behaviour for the
    // nested records too, serialize via a sorted value.
    let value = usage_to_sorted_value(data);
    let serialized = serde_json::to_string_pretty(&value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // tempfile in the same dir, then atomic rename.
    let pid = std::process::id();
    let nanos = Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let tmp_path = parent.join(format!(".usage_{pid}_{nanos}.tmp"));

    let write_res = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(serialized.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        Ok(())
    })();

    match write_res {
        Ok(()) => match fs::rename(&tmp_path, path) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&tmp_path);
                Err(e)
            }
        },
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Build a Value with all object keys sorted (recursively), to match Python's
/// `sort_keys=True`.
fn usage_to_sorted_value(data: &UsageMap) -> Value {
    let mut top = Map::new();
    for (k, rec) in data {
        top.insert(k.clone(), sort_value(&Value::Object(rec.clone())));
    }
    Value::Object(top)
}

fn sort_value(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            // BTreeMap gives sorted iteration; rebuild a Map (serde_json::Map
            // without preserve_order is itself a BTreeMap so order is sorted).
            let sorted: BTreeMap<String, Value> =
                m.iter().map(|(k, val)| (k.clone(), sort_value(val))).collect();
            let mut out = Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

/// Return the record for *skill_name*, creating a fresh one if missing.
/// Backfills any missing keys.
pub fn get_record(skill_name: &str) -> Record {
    let data = load_usage();
    let mut rec = match data.get(skill_name) {
        Some(r) => r.clone(),
        None => return empty_record(),
    };
    let base = empty_record();
    for (k, v) in base {
        rec.entry(k).or_insert(v);
    }
    rec
}

/// Load, apply *mutator(record)* in place, save. Best-effort.
///
/// Bundled and hub-installed skills are NEVER recorded in the sidecar. Local
/// manual skills may still accrue usage telemetry, but they only become
/// curator-managed when `created_by` is explicitly marked.
fn mutate<F: FnOnce(&mut Record)>(skill_name: &str, mutator: F) {
    if skill_name.is_empty() {
        return;
    }
    if !is_agent_created(skill_name) {
        return;
    }
    let mut data = load_usage();
    let mut rec = match data.get(skill_name) {
        Some(r) => r.clone(),
        None => empty_record(),
    };
    mutator(&mut rec);
    data.insert(skill_name.to_string(), rec);
    save_usage(&data);
}

// ---------------------------------------------------------------------------
// Public counter-bump helpers
// ---------------------------------------------------------------------------

/// Bump view_count and last_viewed_at. Called from skill_view().
pub fn bump_view(skill_name: &str) {
    mutate(skill_name, |rec| {
        let n = value_to_int(rec.get("view_count")).unwrap_or(0) + 1;
        rec.insert("view_count".into(), Value::from(n));
        rec.insert("last_viewed_at".into(), Value::String(now_iso()));
    });
}

/// Bump use_count and last_used_at. Called when a skill is actively used.
pub fn bump_use(skill_name: &str) {
    mutate(skill_name, |rec| {
        let n = value_to_int(rec.get("use_count")).unwrap_or(0) + 1;
        rec.insert("use_count".into(), Value::from(n));
        rec.insert("last_used_at".into(), Value::String(now_iso()));
    });
}

/// Bump patch_count and last_patched_at. Called from skill_manage (patch/edit).
pub fn bump_patch(skill_name: &str) {
    mutate(skill_name, |rec| {
        let n = value_to_int(rec.get("patch_count")).unwrap_or(0) + 1;
        rec.insert("patch_count".into(), Value::from(n));
        rec.insert("last_patched_at".into(), Value::String(now_iso()));
    });
}

/// Opt a skill created by skill_manage into curator management.
pub fn mark_agent_created(skill_name: &str) {
    mutate(skill_name, |rec| {
        rec.insert("created_by".into(), Value::String("agent".to_string()));
    });
}

/// Set lifecycle state. No-op if *state* is invalid.
pub fn set_state(skill_name: &str, state: &str) {
    if !valid_states().contains(state) {
        log::debug!("set_state: invalid state {state:?} for {skill_name}");
        return;
    }
    let state = state.to_string();
    mutate(skill_name, |rec| {
        rec.insert("state".into(), Value::String(state.clone()));
        if state == STATE_ARCHIVED {
            rec.insert("archived_at".into(), Value::String(now_iso()));
        } else if state == STATE_ACTIVE {
            rec.insert("archived_at".into(), Value::Null);
        }
    });
}

pub fn set_pinned(skill_name: &str, pinned: bool) {
    mutate(skill_name, |rec| {
        rec.insert("pinned".into(), Value::Bool(pinned));
    });
}

/// Drop a skill's usage entry entirely. Called when the skill is deleted.
pub fn forget(skill_name: &str) {
    if skill_name.is_empty() {
        return;
    }
    let mut data = load_usage();
    if data.remove(skill_name).is_some() {
        save_usage(&data);
    }
}

// ---------------------------------------------------------------------------
// Archive / restore
// ---------------------------------------------------------------------------

/// Move an agent-created skill directory to `~/.hermes/skills/.archive/`.
///
/// Returns (ok, message). Never archives bundled or hub skills.
pub fn archive_skill(skill_name: &str) -> (bool, String) {
    if !is_agent_created(skill_name) {
        return (
            false,
            format!("skill '{skill_name}' is bundled or hub-installed; never archive"),
        );
    }

    let skill_dir = match find_skill_dir(skill_name) {
        Some(d) => d,
        None => return (false, format!("skill '{skill_name}' not found")),
    };

    let archive_root = archive_dir();
    if let Err(e) = fs::create_dir_all(&archive_root) {
        return (false, format!("failed to create archive dir: {e}"));
    }

    let dir_name = skill_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut dest = archive_root.join(&dir_name);
    if dest.exists() {
        let ts = Utc::now().format("%Y%m%d%H%M%S").to_string();
        dest = archive_root.join(format!("{dir_name}-{ts}"));
    }

    if let Err(e) = fs::rename(&skill_dir, &dest) {
        // Cross-device — fall back to recursive copy + remove.
        if let Err(e2) = move_dir(&skill_dir, &dest) {
            let _ = e;
            return (false, format!("failed to archive: {e2}"));
        }
    }

    set_state(skill_name, STATE_ARCHIVED);
    (true, format!("archived to {}", dest.display()))
}

/// Move an archived skill back to `~/.hermes/skills/`. Restores to the flat
/// top-level layout; original category nesting is NOT reconstructed.
pub fn restore_skill(skill_name: &str) -> (bool, String) {
    if !is_agent_created(skill_name) {
        return (
            false,
            format!(
                "skill '{skill_name}' is now bundled or hub-installed; \
                 restore would shadow the upstream version"
            ),
        );
    }
    let archive_root = archive_dir();
    if !archive_root.exists() {
        return (false, "no archive directory".to_string());
    }

    // Exact name match first, then prefix match (timestamped dupes), reverse sorted.
    let all_dirs: Vec<PathBuf> = rglob_dirs(&archive_root);
    let mut candidates: Vec<PathBuf> = all_dirs
        .iter()
        .filter(|p| p.file_name().and_then(|n| n.to_str()) == Some(skill_name))
        .cloned()
        .collect();
    if candidates.is_empty() {
        let prefix = format!("{skill_name}-");
        let mut prefixed: Vec<PathBuf> = all_dirs
            .iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(&prefix))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        // sorted(..., reverse=True) on the Path objects: sort by full path string.
        prefixed.sort();
        prefixed.reverse();
        candidates = prefixed;
    }
    if candidates.is_empty() {
        return (false, format!("skill '{skill_name}' not found in archive"));
    }

    let src = &candidates[0];
    let dest = skills_dir().join(skill_name);
    if dest.exists() {
        return (false, format!("destination already exists: {}", dest.display()));
    }

    if fs::rename(src, &dest).is_err() {
        if let Err(e) = move_dir(src, &dest) {
            return (false, format!("failed to restore: {e}"));
        }
    }

    set_state(skill_name, STATE_ACTIVE);
    (true, format!("restored to {}", dest.display()))
}

/// Recursively copy a directory tree then remove the source (shutil.move fallback).
fn move_dir(src: &Path, dest: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        copy_dir_all(src, dest)?;
        fs::remove_dir_all(src)?;
    } else {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dest)?;
        fs::remove_file(src)?;
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dest: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Recursively collect all directories under `base` (mirrors `rglob("*")` dir filter).
fn rglob_dirs(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                out.push(path.clone());
                walk(&path, out);
            }
        }
    }
    walk(base, &mut out);
    out
}

/// Locate the directory for a skill by its frontmatter `name:` field.
///
/// Handles both flat and category-nested layouts.
fn find_skill_dir(skill_name: &str) -> Option<PathBuf> {
    let base = skills_dir();
    if !base.exists() {
        return None;
    }
    for skill_md in rglob_skill_md(&base) {
        let rel = match skill_md.strip_prefix(&base) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let first_part = rel
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str());
        if let Some(p) = first_part {
            if p.starts_with('.') {
                continue;
            }
        }
        let fallback = skill_md
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if read_skill_name(&skill_md, &fallback) == skill_name {
            return skill_md.parent().map(|p| p.to_path_buf());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Reporting — for the curator CLI / slash command
// ---------------------------------------------------------------------------

/// Return a list of {name, state, pinned, last_activity_at, ...} records for
/// every agent-created skill. Missing usage records are backfilled with
/// defaults so callers can always index fields.
pub fn agent_created_report() -> Vec<Record> {
    let data = load_usage();
    let mut rows: Vec<Record> = Vec::new();
    for name in list_agent_created_skill_names() {
        let mut rec = match data.get(&name) {
            Some(r) => r.clone(),
            None => empty_record(),
        };
        let base = empty_record();
        for (k, v) in base {
            rec.entry(k).or_insert(v);
        }
        let mut row = Map::new();
        row.insert("name".into(), Value::String(name.clone()));
        for (k, v) in rec {
            row.insert(k, v);
        }
        let last = latest_activity_at(&row);
        row.insert(
            "last_activity_at".into(),
            last.map(Value::String).unwrap_or(Value::Null),
        );
        row.insert("activity_count".into(), Value::from(activity_count(&row)));
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate HERMES_HOME / the global sidecar.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TempHome {
        dir: PathBuf,
    }
    impl TempHome {
        fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "hermes_usage_test_{}_{}",
                std::process::id(),
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ));
            fs::create_dir_all(base.join("skills")).unwrap();
            unsafe {
                std::env::set_var("HERMES_HOME", &base);
            }
            TempHome { dir: base }
        }
    }
    impl Drop for TempHome {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
            unsafe {
                std::env::remove_var("HERMES_HOME");
            }
        }
    }

    fn write_skill(home: &Path, rel_dir: &str, name: &str) {
        let dir = home.join("skills").join(rel_dir);
        fs::create_dir_all(&dir).unwrap();
        let content = format!("---\nname: {name}\ndescription: x\n---\nbody\n");
        fs::write(dir.join("SKILL.md"), content).unwrap();
    }

    #[test]
    fn test_activity_count_and_latest() {
        let mut rec = empty_record();
        rec.insert("use_count".into(), Value::from(3));
        rec.insert("view_count".into(), Value::from(2));
        rec.insert("patch_count".into(), Value::Null);
        assert_eq!(activity_count(&rec), 5);

        rec.insert(
            "last_used_at".into(),
            Value::String("2026-01-01T00:00:00+00:00".into()),
        );
        rec.insert(
            "last_viewed_at".into(),
            Value::String("2026-02-01T00:00:00+00:00".into()),
        );
        rec.insert("last_patched_at".into(), Value::Null);
        assert_eq!(
            latest_activity_at(&rec).as_deref(),
            Some("2026-02-01T00:00:00+00:00")
        );
    }

    #[test]
    fn test_activity_count_bad_values() {
        let mut rec = empty_record();
        rec.insert("use_count".into(), Value::String("notanumber".into()));
        rec.insert("view_count".into(), Value::from(4));
        rec.insert("patch_count".into(), Value::from(1));
        // "notanumber" -> parse failure -> skipped; total = 5
        assert_eq!(activity_count(&rec), 5);
    }

    #[test]
    fn test_bump_and_provenance() {
        let _g = ENV_LOCK.lock().unwrap();
        let home = TempHome::new();
        write_skill(&home.dir, "demo", "demo");

        // Not marked agent-created yet -> not in report, but telemetry still
        // accrues because skill is agent-created (no bundled/hub manifest).
        bump_view("demo");
        bump_use("demo");
        bump_use("demo");
        let rec = get_record("demo");
        assert_eq!(value_to_int(rec.get("view_count")), Some(1));
        assert_eq!(value_to_int(rec.get("use_count")), Some(2));

        // Report empty until marked.
        assert!(agent_created_report().is_empty());
        mark_agent_created("demo");
        let report = agent_created_report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].get("name"), Some(&Value::String("demo".into())));
        assert_eq!(report[0].get("activity_count"), Some(&Value::from(3)));
    }

    #[test]
    fn test_bundled_off_limits() {
        let _g = ENV_LOCK.lock().unwrap();
        let home = TempHome::new();
        write_skill(&home.dir, "bundled_one", "bundled_one");
        fs::write(
            home.dir.join("skills").join(".bundled_manifest"),
            "bundled_one:abc123\n",
        )
        .unwrap();

        assert!(!is_agent_created("bundled_one"));
        // mutate is a no-op for off-limits skills.
        bump_use("bundled_one");
        let usage = load_usage();
        assert!(usage.get("bundled_one").is_none());
    }

    #[test]
    fn test_set_state_invalid_and_pin() {
        let _g = ENV_LOCK.lock().unwrap();
        let _home = TempHome::new();
        mark_agent_created("s");
        set_state("s", "bogus"); // no-op
        let rec = get_record("s");
        assert_eq!(rec.get("state"), Some(&Value::String(STATE_ACTIVE.into())));

        set_state("s", STATE_ARCHIVED);
        let rec = get_record("s");
        assert_eq!(rec.get("state"), Some(&Value::String(STATE_ARCHIVED.into())));
        assert!(rec.get("archived_at").map(|v| !v.is_null()).unwrap_or(false));

        set_state("s", STATE_ACTIVE);
        let rec = get_record("s");
        assert_eq!(rec.get("archived_at"), Some(&Value::Null));

        set_pinned("s", true);
        assert_eq!(get_record("s").get("pinned"), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_forget() {
        let _g = ENV_LOCK.lock().unwrap();
        let _home = TempHome::new();
        mark_agent_created("gone");
        assert!(load_usage().contains_key("gone"));
        forget("gone");
        assert!(!load_usage().contains_key("gone"));
    }

    #[test]
    fn test_archive_and_restore() {
        let _g = ENV_LOCK.lock().unwrap();
        let home = TempHome::new();
        write_skill(&home.dir, "cat/movable", "movable");
        mark_agent_created("movable");

        let (ok, msg) = archive_skill("movable");
        assert!(ok, "archive failed: {msg}");
        assert!(home.dir.join("skills").join(".archive").join("movable").exists());
        assert_eq!(get_record("movable").get("state"), Some(&Value::String(STATE_ARCHIVED.into())));

        let (ok, msg) = restore_skill("movable");
        assert!(ok, "restore failed: {msg}");
        assert!(home.dir.join("skills").join("movable").join("SKILL.md").exists());
        assert_eq!(get_record("movable").get("state"), Some(&Value::String(STATE_ACTIVE.into())));
    }

    #[test]
    fn test_save_load_roundtrip_sorted() {
        let _g = ENV_LOCK.lock().unwrap();
        let _home = TempHome::new();
        let mut data = UsageMap::new();
        data.insert("zeta".into(), empty_record());
        data.insert("alpha".into(), empty_record());
        save_usage(&data);
        let loaded = load_usage();
        assert!(loaded.contains_key("zeta"));
        assert!(loaded.contains_key("alpha"));
    }
}
