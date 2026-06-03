//! Curator snapshot + rollback.
//!
//! A pre-run snapshot of `~/.hermes/skills/` (excluding `.curator_backups/`
//! itself) is taken before any mutating curator pass. Snapshots are tar.gz
//! files under `~/.hermes/skills/.curator_backups/<utc-iso>/` with a
//! companion `manifest.json` describing the snapshot (reason, time, size,
//! counted skill files). Rollback picks a snapshot, moves the current
//! `skills/` tree aside into a staging dir so even the rollback itself is
//! undoable (the safety snapshot is the user-facing undo handle), then
//! extracts the chosen snapshot into place.
//!
//! The snapshot does NOT include:
//!   - `.curator_backups/` (would recurse)
//!   - `.hub/` (hub-installed skills — managed by the hub, not us)
//!
//! It DOES include everything else under `skills/`: SKILL.md trees,
//! `.usage.json`, `.archive/`, `.curator_state`, `.bundled_manifest`, etc.
//!
//! Alongside the skills tarball, each snapshot also captures a copy of
//! `~/.hermes/cron/jobs.json` as `cron-jobs.json` when it exists. Rollback
//! only touches the `skills`/`skill` fields of matching jobs (by id), leaving
//! the rest of each job (schedule, next_run_at, enabled, prompt, etc.) alone.
//!
//! This is a faithful Rust port of the Python `agent/curator_backup.py`.
//! Unlike the Python original which relied on global `get_hermes_home()` and
//! a `cron.jobs` module, the public entry points here take the hermes home
//! directory as an explicit parameter (matching the rest of hermes-core), and
//! the cron skill-link reconciliation operates directly on `cron/jobs.json`.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use chrono::{SecondsFormat, Utc};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use regex::Regex;
use serde_json::{json, Value};

/// Default number of regular snapshots to keep when pruning.
pub const DEFAULT_KEEP: i64 = 5;

/// Filename used for the captured copy of `cron/jobs.json` inside a snapshot.
pub const CRON_JOBS_FILENAME: &str = "cron-jobs.json";

/// Top-level entries under `skills/` that must never be rolled into a snapshot.
/// `.hub/` is managed by the skills hub; `.curator_backups` is the backup dir
/// itself (recursion bomb).
const EXCLUDE_TOP_LEVEL: &[&str] = &[".curator_backups", ".hub"];

fn id_regex() -> Regex {
    // UTC ISO with colons replaced by dashes, optional `-NN` collision suffix.
    Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$").unwrap()
}

/// Whether `name` matches the snapshot-id shape.
pub fn is_snapshot_id(name: &str) -> bool {
    id_regex().is_match(name)
}

fn backups_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("skills").join(".curator_backups")
}

fn skills_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("skills")
}

fn cron_jobs_file(hermes_home: &Path) -> PathBuf {
    hermes_home.join("cron").join("jobs.json")
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Backup configuration extracted from `curator.backup` in hermes config.
#[derive(Debug, Clone)]
pub struct BackupConfig {
    pub enabled: bool,
    pub keep: i64,
}

impl Default for BackupConfig {
    fn default() -> Self {
        BackupConfig {
            enabled: true,
            keep: DEFAULT_KEEP,
        }
    }
}

impl BackupConfig {
    /// Parse from a top-level config JSON value, reading `curator.backup`.
    ///
    /// Mirrors the Python `_load_config` + `is_enabled` + `get_keep` logic:
    /// default ON, keep coerced to int and clamped to a minimum of 1.
    pub fn from_config_value(cfg: &Value) -> Self {
        let backup = cfg
            .get("curator")
            .and_then(|c| c.as_object())
            .and_then(|c| c.get("backup"))
            .and_then(|b| b.as_object());

        let mut out = BackupConfig::default();
        if let Some(bk) = backup {
            // enabled: bool(...) — truthy default True.
            out.enabled = match bk.get("enabled") {
                Some(v) => json_truthy(v),
                None => true,
            };
            // keep: int(cfg.get("keep", DEFAULT_KEEP)); on TypeError/ValueError
            // fall back to DEFAULT_KEEP, then max(1, n).
            out.keep = match bk.get("keep") {
                Some(v) => coerce_int(v).unwrap_or(DEFAULT_KEEP),
                None => DEFAULT_KEEP,
            };
        }
        out.keep = out.keep.max(1);
        out
    }
}

/// Python `bool(x)` truthiness for the JSON values we care about.
fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `int(x)`-ish coercion for config values; ints, floats (truncated),
/// and numeric strings succeed, everything else fails.
fn coerce_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f.trunc() as i64)
            }
        }
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Timestamp id
// ---------------------------------------------------------------------------

/// UTC ISO-ish filesystem-safe timestamp: `2026-05-01T13-05-42Z`.
pub fn utc_id() -> String {
    utc_id_at(Utc::now())
}

/// Same as [`utc_id`] but at an explicit instant (handy for tests).
pub fn utc_id_at(now: chrono::DateTime<Utc>) -> String {
    // isoformat without subseconds, no `+00:00` tz, colons → dashes, + "Z".
    let s = now.to_rfc3339_opts(SecondsFormat::Secs, true);
    // rfc3339 with use_z=true gives e.g. "2026-05-01T13:05:42Z" — strip the
    // trailing Z, dash the colons, re-add Z (matching the Python output).
    let s = s.strip_suffix('Z').unwrap_or(&s);
    format!("{}Z", s.replace(':', "-"))
}

// ---------------------------------------------------------------------------
// Cron backup helper
// ---------------------------------------------------------------------------

/// Result of attempting to capture `cron/jobs.json` into a snapshot dir.
#[derive(Debug, Clone, Default)]
pub struct CronBackupInfo {
    pub backed_up: bool,
    pub jobs_count: usize,
    pub reason: Option<String>,
    pub parse_warning: Option<String>,
}

/// Copy the live `cron/jobs.json` into `dest` as `cron-jobs.json`.
///
/// Never returns an error — failures are folded into the returned info so the
/// snapshot can proceed (the skills side is the core guarantee; cron is
/// additive).
pub fn backup_cron_jobs_into(hermes_home: &Path, dest: &Path) -> CronBackupInfo {
    let mut info = CronBackupInfo::default();
    let src = cron_jobs_file(hermes_home);
    if !src.exists() {
        info.reason = Some("no cron/jobs.json present".to_string());
        return info;
    }
    let raw = match fs::read_to_string(&src) {
        Ok(r) => r,
        Err(e) => {
            info.reason = Some(format!("read error: {}", e));
            return info;
        }
    };
    // Count jobs as a diagnostic but never fail on bad JSON.
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Object(obj)) => {
            if let Some(Value::Array(inner)) = obj.get("jobs") {
                info.jobs_count = inner.len();
            }
        }
        Ok(Value::Array(arr)) => {
            info.jobs_count = arr.len();
        }
        Ok(_) => {}
        Err(_) => {
            info.jobs_count = 0;
            info.parse_warning =
                Some("jobs.json was not valid JSON at snapshot time".to_string());
        }
    }
    if let Err(e) = fs::write(dest.join(CRON_JOBS_FILENAME), raw.as_bytes()) {
        info.reason = Some(format!("write error: {}", e));
        return info;
    }
    info.backed_up = true;
    info
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// Count `SKILL.md` files under `base`, recursively. Returns 0 on error.
fn count_skill_files(base: &Path) -> i64 {
    fn walk(dir: &Path, count: &mut i64) {
        let entries = match fs::read_dir(dir) {
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
                walk(&path, count);
            } else if entry.file_name() == "SKILL.md" {
                *count += 1;
            }
        }
    }
    let mut count = 0;
    walk(base, &mut count);
    count
}

/// Write `manifest.json` describing the snapshot at `dest`.
fn write_manifest(
    dest: &Path,
    reason: &str,
    archive_path: &Path,
    skills_counted: i64,
    cron_info: Option<&CronBackupInfo>,
) -> io::Result<()> {
    let archive_bytes = fs::metadata(archive_path).map(|m| m.len()).unwrap_or(0);
    let id = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let archive_name = archive_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut manifest = serde_json::Map::new();
    manifest.insert("id".to_string(), json!(id));
    manifest.insert("reason".to_string(), json!(reason));
    manifest.insert(
        "created_at".to_string(),
        json!(Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false)),
    );
    manifest.insert("archive".to_string(), json!(archive_name));
    manifest.insert("archive_bytes".to_string(), json!(archive_bytes));
    manifest.insert("skill_files".to_string(), json!(skills_counted));

    if let Some(ci) = cron_info {
        let mut cron = serde_json::Map::new();
        cron.insert("backed_up".to_string(), json!(ci.backed_up));
        cron.insert("jobs_count".to_string(), json!(ci.jobs_count));
        if !ci.backed_up {
            cron.insert(
                "reason".to_string(),
                json!(ci.reason.clone().unwrap_or_else(|| "not captured".to_string())),
            );
        }
        if let Some(pw) = &ci.parse_warning {
            cron.insert("parse_warning".to_string(), json!(pw));
        }
        manifest.insert("cron_jobs".to_string(), Value::Object(cron));
    }

    // json.dumps(..., indent=2, sort_keys=True): serde_json::Map preserves
    // insertion order, so re-pack into a BTreeMap to emit sorted keys.
    let sorted: BTreeMap<String, Value> = manifest.into_iter().collect();
    let text = serde_json::to_string_pretty(&sorted)
        .unwrap_or_else(|_| "{}".to_string());
    fs::write(dest.join("manifest.json"), text.as_bytes())
}

/// Create a tar.gz snapshot of `~/.hermes/skills/` and prune old ones.
///
/// Returns `Ok(Some(dir))` with the snapshot directory, or `Ok(None)` if the
/// snapshot was skipped (backup disabled, skills dir missing, or an IO error
/// occurred — matching the Python contract that a backup failure never aborts
/// a curator pass).
pub fn snapshot_skills(
    hermes_home: &Path,
    config: &BackupConfig,
    reason: &str,
) -> Option<PathBuf> {
    if !config.enabled {
        return None;
    }

    let skills = skills_dir(hermes_home);
    if !skills.exists() {
        return None;
    }

    let backups = backups_dir(hermes_home);
    if fs::create_dir_all(&backups).is_err() {
        return None;
    }

    // Uniquify against same-second collisions.
    let base_id = utc_id();
    let mut snap_id = base_id.clone();
    let mut counter = 1u32;
    while backups.join(&snap_id).exists() {
        snap_id = format!("{}-{:02}", base_id, counter);
        counter += 1;
    }

    let dest = backups.join(&snap_id);
    // mkdir exist_ok=False
    if fs::create_dir(&dest).is_err() {
        return None;
    }

    let archive = dest.join("skills.tar.gz");
    let snapshot_result = (|| -> io::Result<()> {
        write_skills_tarball(&skills, &archive)?;
        let cron_info = backup_cron_jobs_into(hermes_home, &dest);
        write_manifest(
            &dest,
            reason,
            &archive,
            count_skill_files(&skills),
            Some(&cron_info),
        )?;
        Ok(())
    })();

    if snapshot_result.is_err() {
        let _ = fs::remove_dir_all(&dest);
        return None;
    }

    prune_old(hermes_home, config.keep);
    Some(dest)
}

/// Build the `skills.tar.gz` tarball, storing each non-excluded top-level entry
/// under its own name (relative to `skills/`) so extraction drops cleanly back.
fn write_skills_tarball(skills: &Path, archive: &Path) -> io::Result<()> {
    let file = fs::File::create(archive)?;
    let enc = GzEncoder::new(file, Compression::new(6));
    let mut builder = tar::Builder::new(enc);
    builder.follow_symlinks(false);

    let mut entries: Vec<PathBuf> = fs::read_dir(skills)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();

    for entry in entries {
        let name = match entry.file_name() {
            Some(n) => n.to_string_lossy().into_owned(),
            None => continue,
        };
        if EXCLUDE_TOP_LEVEL.contains(&name.as_str()) {
            continue;
        }
        let meta = fs::symlink_metadata(&entry)?;
        if meta.is_dir() {
            builder.append_dir_all(&name, &entry)?;
        } else {
            let mut f = fs::File::open(&entry)?;
            builder.append_file(&name, &mut f)?;
        }
    }

    let enc = builder.into_inner()?;
    enc.finish()?;
    Ok(())
}

/// Delete regular snapshots beyond the newest `keep`. Returns deleted ids.
/// Stale `.rollback-staging-*` dirs are cleaned up on every call.
pub fn prune_old(hermes_home: &Path, keep: i64) -> Vec<String> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return Vec::new();
    }
    let re = id_regex();
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    let mut stale_staging: Vec<PathBuf> = Vec::new();

    let read = match fs::read_dir(&backups) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    for child in read.flatten() {
        let path = child.path();
        if !path.is_dir() {
            continue;
        }
        let name = child.file_name().to_string_lossy().into_owned();
        if name.starts_with(".rollback-staging-") {
            stale_staging.push(path);
            continue;
        }
        if re.is_match(&name) {
            entries.push((name, path));
        }
    }
    // Newest first (lexicographic works for UTC ISO ids).
    entries.sort_by(|a, b| b.0.cmp(&a.0));

    let keep = keep.max(0) as usize;
    let mut deleted = Vec::new();
    for (name, path) in entries.into_iter().skip(keep) {
        if fs::remove_dir_all(&path).is_ok() {
            deleted.push(name);
        }
    }
    for path in stale_staging {
        let _ = fs::remove_dir_all(&path);
    }
    deleted
}

// ---------------------------------------------------------------------------
// List + rollback
// ---------------------------------------------------------------------------

fn read_manifest(snap_dir: &Path) -> serde_json::Map<String, Value> {
    let mf = snap_dir.join("manifest.json");
    if !mf.exists() {
        return serde_json::Map::new();
    }
    match fs::read_to_string(&mf) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(obj)) => obj,
            _ => serde_json::Map::new(),
        },
        Err(_) => serde_json::Map::new(),
    }
}

/// Return all restorable snapshots, newest first. Only entries with a real
/// `skills.tar.gz` tarball are listed; transient `.rollback-staging-*` dirs are
/// implementation detail and not shown.
pub fn list_backups(hermes_home: &Path) -> Vec<Value> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return Vec::new();
    }
    let re = id_regex();
    let mut children: Vec<PathBuf> = match fs::read_dir(&backups) {
        Ok(r) => r.flatten().map(|e| e.path()).collect(),
        Err(_) => return Vec::new(),
    };
    // sorted(..., reverse=True) on the paths' names.
    children.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    let mut out = Vec::new();
    for child in children {
        if !child.is_dir() {
            continue;
        }
        let name = match child.file_name() {
            Some(n) => n.to_string_lossy().into_owned(),
            None => continue,
        };
        if !re.is_match(&name) {
            continue;
        }
        let archive = child.join("skills.tar.gz");
        if !archive.exists() {
            continue;
        }
        let mut mf = read_manifest(&child);
        mf.entry("id".to_string()).or_insert_with(|| json!(name));
        mf.entry("path".to_string())
            .or_insert_with(|| json!(child.to_string_lossy().into_owned()));
        if !mf.contains_key("archive_bytes") {
            let size = fs::metadata(&archive).map(|m| m.len()).unwrap_or(0);
            mf.insert("archive_bytes".to_string(), json!(size));
        }
        out.push(Value::Object(mf));
    }
    out
}

/// Return the path of the requested backup, or the newest one if `backup_id`
/// is `None`. Returns `None` if no match.
pub fn resolve_backup(hermes_home: &Path, backup_id: Option<&str>) -> Option<PathBuf> {
    let backups = backups_dir(hermes_home);
    if !backups.exists() {
        return None;
    }
    let re = id_regex();
    if let Some(bid) = backup_id {
        if !bid.is_empty() {
            let target = backups.join(bid);
            if target.is_dir() && re.is_match(bid) && target.join("skills.tar.gz").exists()
            {
                return Some(target);
            }
            return None;
        }
    }
    let mut candidates: Vec<PathBuf> = match fs::read_dir(&backups) {
        Ok(r) => r.flatten().map(|e| e.path()).collect(),
        Err(_) => return None,
    };
    candidates.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    candidates.into_iter().find(|c| {
        c.is_dir()
            && c.file_name()
                .map(|n| re.is_match(&n.to_string_lossy()))
                .unwrap_or(false)
            && c.join("skills.tar.gz").exists()
    })
}

/// Report from reconciling cron skill-links during rollback.
#[derive(Debug, Clone, Default)]
pub struct CronRestoreReport {
    pub attempted: bool,
    pub restored: Vec<Value>,
    pub skipped_missing: Vec<Value>,
    pub unchanged: usize,
    pub error: Option<String>,
}

/// Reconcile backed-up cron skill links into the live `cron/jobs.json`.
///
/// Only the `skills` and `skill` fields are restored, and only on jobs that
/// still exist in the current file (matched by `id`). Everything else about a
/// job is live state and left untouched. Never returns an error — failures are
/// captured in the returned report.
pub fn restore_cron_skill_links(hermes_home: &Path, snapshot_dir: &Path) -> CronRestoreReport {
    let mut report = CronRestoreReport::default();
    let backup_file = snapshot_dir.join(CRON_JOBS_FILENAME);
    if !backup_file.exists() {
        report.error = Some(format!("snapshot has no {}", CRON_JOBS_FILENAME));
        return report;
    }

    let backup_text = match fs::read_to_string(&backup_file) {
        Ok(t) => t,
        Err(e) => {
            report.error = Some(format!("failed to load backed-up jobs: {}", e));
            return report;
        }
    };
    let backup_parsed: Value = match serde_json::from_str(&backup_text) {
        Ok(v) => v,
        Err(e) => {
            report.error = Some(format!("failed to load backed-up jobs: {}", e));
            return report;
        }
    };

    let backup_jobs: Option<&Vec<Value>> = match &backup_parsed {
        Value::Object(obj) => obj.get("jobs").and_then(|j| j.as_array()),
        Value::Array(arr) => Some(arr),
        _ => None,
    };
    let backup_jobs = match backup_jobs {
        Some(j) => j,
        None => {
            report.error = Some("backed-up cron-jobs.json has no jobs list".to_string());
            return report;
        }
    };

    // Lookup of backed-up skill state keyed by job id.
    let mut backup_by_id: BTreeMap<String, BackupSkillState> = BTreeMap::new();
    for job in backup_jobs {
        let obj = match job.as_object() {
            Some(o) => o,
            None => continue,
        };
        let jid = match obj.get("id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| jid.clone());
        backup_by_id.insert(
            jid,
            BackupSkillState {
                skills: obj.get("skills").cloned(),
                skill: obj.get("skill").cloned(),
                name,
            },
        );
    }

    if backup_by_id.is_empty() {
        report.attempted = true; // tried, nothing to do
        return report;
    }

    // Load the live jobs file. We operate directly on cron/jobs.json (the
    // Python original went through cron.jobs for locking; here we read +
    // atomically rewrite in place, preserving every field except skills/skill
    // on matched jobs).
    report.attempted = true;
    let result = reconcile_live_jobs(hermes_home, &backup_by_id, &mut report);
    if let Err(e) = result {
        report.error = Some(format!("restore failed mid-flight: {}", e));
    }

    report
}

#[derive(Debug, Clone)]
struct BackupSkillState {
    skills: Option<Value>,
    skill: Option<Value>,
    name: String,
}

/// Load `cron/jobs.json`, apply skill-link restores, and atomically write back
/// if anything changed. Returns the set of live job ids seen so the caller can
/// flag backed-up jobs that no longer exist.
fn reconcile_live_jobs(
    hermes_home: &Path,
    backup_by_id: &BTreeMap<String, BackupSkillState>,
    report: &mut CronRestoreReport,
) -> io::Result<()> {
    let path = cron_jobs_file(hermes_home);
    let raw = match fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    let mut parsed: Value = serde_json::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // jobs.json is `{"jobs": [...], "updated_at": ...}` or a bare list.
    let mut changed = false;
    let mut live_ids: HashSet<String> = HashSet::new();

    {
        let live_jobs: &mut Vec<Value> = match &mut parsed {
            Value::Object(obj) => match obj.get_mut("jobs").and_then(|j| j.as_array_mut()) {
                Some(arr) => arr,
                None => {
                    // No jobs list — nothing live to reconcile.
                    flag_missing(backup_by_id, &live_ids, report);
                    return Ok(());
                }
            },
            Value::Array(arr) => arr,
            _ => {
                flag_missing(backup_by_id, &live_ids, report);
                return Ok(());
            }
        };

        for live in live_jobs.iter_mut() {
            let obj = match live.as_object_mut() {
                Some(o) => o,
                None => continue,
            };
            let jid = match obj.get("id").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            live_ids.insert(jid.clone());

            let backup = match backup_by_id.get(&jid) {
                Some(b) => b,
                None => continue, // live job didn't exist at snapshot time
            };

            let cur_skills = obj.get("skills").cloned();
            let cur_skill = obj.get("skill").cloned();
            let bkp_skills = backup.skills.clone();
            let bkp_skill = backup.skill.clone();

            if cur_skills == bkp_skills && cur_skill == bkp_skill {
                report.unchanged += 1;
                continue;
            }

            // Restore, preserving absence.
            match &bkp_skills {
                None => {
                    obj.remove("skills");
                }
                Some(v) => {
                    obj.insert("skills".to_string(), v.clone());
                }
            }
            match &bkp_skill {
                None => {
                    obj.remove("skill");
                }
                Some(v) => {
                    obj.insert("skill".to_string(), v.clone());
                }
            }

            report.restored.push(json!({
                "job_id": jid,
                "job_name": backup.name,
                "from": {
                    "skills": cur_skills.unwrap_or(Value::Null),
                    "skill": cur_skill.unwrap_or(Value::Null),
                },
                "to": {
                    "skills": bkp_skills.unwrap_or(Value::Null),
                    "skill": bkp_skill.unwrap_or(Value::Null),
                },
            }));
            changed = true;
        }
    }

    flag_missing(backup_by_id, &live_ids, report);

    if changed {
        let text = serde_json::to_string_pretty(&parsed)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        atomic_write(&path, text.as_bytes())?;
    }
    Ok(())
}

/// Record backed-up jobs that are absent from the live file.
fn flag_missing(
    backup_by_id: &BTreeMap<String, BackupSkillState>,
    live_ids: &HashSet<String>,
    report: &mut CronRestoreReport,
) {
    for (jid, backup) in backup_by_id {
        if !live_ids.contains(jid) {
            report.skipped_missing.push(json!({
                "job_id": jid,
                "job_name": backup.name,
            }));
        }
    }
}

/// Atomic write via temp file + rename in the same directory.
fn atomic_write(path: &Path, data: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = dir.join(format!(
        ".{}.tmp.{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "jobs.json".to_string()),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.flush()?;
    }
    fs::rename(&tmp, path)
}

/// Outcome of a rollback attempt.
#[derive(Debug, Clone)]
pub struct RollbackResult {
    pub ok: bool,
    pub message: String,
    pub snapshot_path: Option<PathBuf>,
}

/// Restore `~/.hermes/skills/` from a snapshot.
///
/// Strategy:
///   1. Resolve the target snapshot (explicit id or newest regular).
///   2. Take a safety snapshot of the CURRENT skills tree (undo handle).
///   3. Move all current top-level entries (except `.curator_backups`/`.hub`)
///      into a staging dir so extraction lands in an empty tree.
///   4. Extract the chosen snapshot into `skills/`.
///   5. On failure during 4, move staged contents back (best-effort).
///   6. Reconcile cron skill-links (failures don't fail the rollback).
pub fn rollback(
    hermes_home: &Path,
    config: &BackupConfig,
    backup_id: Option<&str>,
) -> RollbackResult {
    let target = match resolve_backup(hermes_home, backup_id) {
        Some(t) => t,
        None => {
            let mut msg = "no matching backup found".to_string();
            if let Some(bid) = backup_id {
                if !bid.is_empty() {
                    msg.push_str(&format!(" for id '{}'", bid));
                }
            }
            msg.push_str(
                " (use `hermes curator rollback --list` to see available snapshots)",
            );
            return RollbackResult {
                ok: false,
                message: msg,
                snapshot_path: None,
            };
        }
    };

    let target_name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let archive = target.join("skills.tar.gz");
    if !archive.exists() {
        return RollbackResult {
            ok: false,
            message: format!("snapshot {} has no skills.tar.gz — corrupted?", target_name),
            snapshot_path: None,
        };
    }

    let skills = skills_dir(hermes_home);
    if fs::create_dir_all(&skills).is_err() {
        return RollbackResult {
            ok: false,
            message: "failed to create skills dir".to_string(),
            snapshot_path: None,
        };
    }
    let backups = backups_dir(hermes_home);
    let _ = fs::create_dir_all(&backups);

    // Step 2: safety snapshot FIRST.
    snapshot_skills(hermes_home, config, &format!("pre-rollback to {}", target_name));

    // Step 3: stage current entries into an internal staging dir.
    let staged = backups.join(format!(".rollback-staging-{}", utc_id()));
    if fs::create_dir(&staged).is_err() {
        return RollbackResult {
            ok: false,
            message: "failed to create staging dir".to_string(),
            snapshot_path: None,
        };
    }

    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let stage_result = (|| -> io::Result<()> {
        let entries: Vec<PathBuf> = fs::read_dir(&skills)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        for entry in entries {
            let name = match entry.file_name() {
                Some(n) => n.to_string_lossy().into_owned(),
                None => continue,
            };
            if EXCLUDE_TOP_LEVEL.contains(&name.as_str()) {
                continue;
            }
            let dest = staged.join(&name);
            move_path(&entry, &dest)?;
            moved.push((entry, dest));
        }
        Ok(())
    })();

    if let Err(e) = stage_result {
        for (orig, dest) in &moved {
            let _ = move_path(dest, orig);
        }
        let _ = fs::remove_dir_all(&staged);
        return RollbackResult {
            ok: false,
            message: format!("failed to stage current skills: {}", e),
            snapshot_path: None,
        };
    }

    // Step 4: extract the snapshot into skills/.
    if let Err(e) = extract_tarball_safely(&archive, &skills) {
        for (orig, dest) in &moved {
            let _ = move_path(dest, orig);
        }
        let _ = fs::remove_dir_all(&staged);
        return RollbackResult {
            ok: false,
            message: format!("snapshot extract failed (state restored): {}", e),
            snapshot_path: None,
        };
    }

    // Extract succeeded — drop the staging dir.
    let _ = fs::remove_dir_all(&staged);

    // Reconcile cron skill-links (surgical; failures don't fail rollback).
    let cron_report = restore_cron_skill_links(hermes_home, &target);

    let mut summary_bits = vec![format!("restored from snapshot {}", target_name)];
    if cron_report.attempted {
        let restored_n = cron_report.restored.len();
        let skipped_n = cron_report.skipped_missing.len();
        if let Some(err) = &cron_report.error {
            summary_bits.push(format!("cron links: error — {}", err));
        } else if restored_n == 0 && skipped_n == 0 && cron_report.unchanged == 0 {
            // Attempted but nothing matched — no-op.
        } else {
            let mut parts = Vec::new();
            if restored_n > 0 {
                parts.push(format!("{} job(s) had skill links restored", restored_n));
            }
            if skipped_n > 0 {
                parts.push(format!(
                    "{} backed-up job(s) no longer exist (skipped)",
                    skipped_n
                ));
            }
            if cron_report.unchanged > 0 {
                parts.push(format!("{} already matched", cron_report.unchanged));
            }
            summary_bits.push(format!("cron links: {}", parts.join(", ")));
        }
    }

    RollbackResult {
        ok: true,
        message: summary_bits.join("; "),
        snapshot_path: Some(target),
    }
}

/// Move `src` to `dst`, falling back to recursive copy + delete across devices.
fn move_path(src: &Path, dst: &Path) -> io::Result<()> {
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            let meta = fs::symlink_metadata(src)?;
            if meta.is_dir() {
                copy_dir_all(src, dst)?;
                fs::remove_dir_all(src)?;
            } else {
                fs::copy(src, dst)?;
                fs::remove_file(src)?;
            }
            Ok(())
        }
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Extract a tar.gz into `dest`, rejecting absolute paths and `..` components
/// (mirrors the Python defensive checks + `filter="data"` intent).
fn extract_tarball_safely(archive: &Path, dest: &Path) -> io::Result<()> {
    // First pass: validate member paths.
    {
        let file = fs::File::open(archive)?;
        let dec = GzDecoder::new(file);
        let mut ar = tar::Archive::new(dec);
        for entry in ar.entries()? {
            let entry = entry?;
            let path = entry.path()?;
            let s = path.to_string_lossy();
            if s.starts_with('/')
                || path
                    .components()
                    .any(|c| matches!(c, Component::ParentDir))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("refusing to extract unsafe path: {:?}", s),
                ));
            }
        }
    }
    // Second pass: actually extract.
    let file = fs::File::open(archive)?;
    let dec = GzDecoder::new(file);
    let mut ar = tar::Archive::new(dec);
    ar.set_preserve_permissions(true);
    ar.set_overwrite(true);
    ar.unpack(dest)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Human-readable summary for CLI
// ---------------------------------------------------------------------------

/// Format a byte count like the Python `format_size`: bytes as integer,
/// KB/MB with one decimal, GB as the catch-all.
pub fn format_size(n: i64) -> String {
    let mut value = n as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if value < 1024.0 || unit == "GB" {
            if unit == "B" {
                return format!("{} B", n);
            }
            return format!("{:.1} {}", value, unit);
        }
        value /= 1024.0;
    }
    format!("{:.1} GB", value)
}

/// Produce the human-readable table of snapshots used by the CLI.
pub fn summarize_backups(hermes_home: &Path) -> String {
    let rows = list_backups(hermes_home);
    if rows.is_empty() {
        return "No curator snapshots yet.".to_string();
    }
    let header = format!("{:<24}  {:<40}  {:>6}  {:>8}", "id", "reason", "skills", "size");
    let mut lines = vec![header.clone()];
    // Python uses the box-drawing char repeated to header char-length.
    lines.push("─".repeat(header.chars().count()));
    for r in &rows {
        let obj = match r.as_object() {
            Some(o) => o,
            None => continue,
        };
        let id = obj.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let reason = obj
            .get("reason")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("?");
        let reason_trunc: String = reason.chars().take(40).collect();
        let skill_files = obj
            .get("skill_files")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let archive_bytes = obj
            .get("archive_bytes")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        lines.push(format!(
            "{:<24}  {:<40}  {:>6}  {:>8}",
            id,
            reason_trunc,
            skill_files,
            format_size(archive_bytes)
        ));
    }
    lines.join("\n")
}

/// Convenience: read the whole bytes of a path (used in tests).
#[allow(dead_code)]
fn read_all(path: &Path) -> io::Result<Vec<u8>> {
    let mut f = fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_home() -> PathBuf {
        let mut p = env::temp_dir();
        let unique = format!(
            "hermes-curator-test-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        );
        p.push(unique);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_skill(home: &Path, name: &str) {
        let dir = skills_dir(home).join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("SKILL.md"), format!("# {}\n", name)).unwrap();
    }

    #[test]
    fn utc_id_shape() {
        let dt = chrono::DateTime::parse_from_rfc3339("2026-05-01T13:05:42.123456+00:00")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(utc_id_at(dt), "2026-05-01T13-05-42Z");
        assert!(is_snapshot_id("2026-05-01T13-05-42Z"));
        assert!(is_snapshot_id("2026-05-01T13-05-42Z-01"));
        assert!(!is_snapshot_id("not-an-id"));
        assert!(!is_snapshot_id("2026-05-01T13:05:42Z"));
    }

    #[test]
    fn config_defaults_and_parsing() {
        let cfg = BackupConfig::from_config_value(&json!({}));
        assert!(cfg.enabled);
        assert_eq!(cfg.keep, DEFAULT_KEEP);

        let cfg = BackupConfig::from_config_value(&json!({
            "curator": {"backup": {"enabled": false, "keep": 3}}
        }));
        assert!(!cfg.enabled);
        assert_eq!(cfg.keep, 3);

        // keep clamped to >= 1
        let cfg = BackupConfig::from_config_value(&json!({
            "curator": {"backup": {"keep": 0}}
        }));
        assert_eq!(cfg.keep, 1);

        // string keep coerces
        let cfg = BackupConfig::from_config_value(&json!({
            "curator": {"backup": {"keep": "7"}}
        }));
        assert_eq!(cfg.keep, 7);

        // bad keep falls back to default
        let cfg = BackupConfig::from_config_value(&json!({
            "curator": {"backup": {"keep": "abc"}}
        }));
        assert_eq!(cfg.keep, DEFAULT_KEEP);
    }

    #[test]
    fn format_size_matches_python() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn snapshot_and_list_and_summary() {
        let home = tmp_home();
        write_skill(&home, "alpha");
        write_skill(&home, "beta");
        // Excluded entries
        fs::create_dir_all(skills_dir(&home).join(".hub")).unwrap();
        fs::write(skills_dir(&home).join(".hub").join("x"), b"hub").unwrap();

        let cfg = BackupConfig::default();
        let snap = snapshot_skills(&home, &cfg, "manual").expect("snapshot created");
        assert!(snap.join("skills.tar.gz").exists());
        assert!(snap.join("manifest.json").exists());

        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(snap.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["reason"], json!("manual"));
        assert_eq!(manifest["skill_files"], json!(2));

        let listed = list_backups(&home);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["reason"], json!("manual"));

        let summary = summarize_backups(&home);
        assert!(summary.contains("manual"));

        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn snapshot_disabled_returns_none() {
        let home = tmp_home();
        write_skill(&home, "alpha");
        let cfg = BackupConfig {
            enabled: false,
            keep: 5,
        };
        assert!(snapshot_skills(&home, &cfg, "manual").is_none());
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn rollback_restores_skills() {
        let home = tmp_home();
        write_skill(&home, "alpha");
        write_skill(&home, "beta");

        let cfg = BackupConfig::default();
        let snap = snapshot_skills(&home, &cfg, "before-edit").unwrap();
        let snap_id = snap.file_name().unwrap().to_string_lossy().into_owned();

        // Mutate: delete beta, add gamma.
        fs::remove_dir_all(skills_dir(&home).join("beta")).unwrap();
        write_skill(&home, "gamma");
        assert!(!skills_dir(&home).join("beta").exists());
        assert!(skills_dir(&home).join("gamma").exists());

        let res = rollback(&home, &cfg, Some(&snap_id));
        assert!(res.ok, "rollback failed: {}", res.message);

        // beta restored; gamma was staged away (not in the snapshot).
        assert!(skills_dir(&home).join("beta").join("SKILL.md").exists());
        assert!(skills_dir(&home).join("alpha").join("SKILL.md").exists());
        assert!(!skills_dir(&home).join("gamma").exists());

        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn rollback_no_backup_found() {
        let home = tmp_home();
        let cfg = BackupConfig::default();
        let res = rollback(&home, &cfg, Some("2099-01-01T00-00-00Z"));
        assert!(!res.ok);
        assert!(res.message.contains("no matching backup found"));
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn cron_jobs_captured_and_restored() {
        let home = tmp_home();
        write_skill(&home, "alpha");
        // Live cron jobs.json with two jobs.
        let cron_dir = home.join("cron");
        fs::create_dir_all(&cron_dir).unwrap();
        let jobs = json!({
            "jobs": [
                {"id": "j1", "name": "Job One", "skill": "narrow-skill", "schedule": "* * * * *"},
                {"id": "j2", "name": "Job Two", "skills": ["a", "b"]}
            ],
            "updated_at": "2026-01-01T00:00:00Z"
        });
        fs::write(
            cron_dir.join("jobs.json"),
            serde_json::to_string_pretty(&jobs).unwrap(),
        )
        .unwrap();

        let cfg = BackupConfig::default();
        let snap = snapshot_skills(&home, &cfg, "before-consolidate").unwrap();
        assert!(snap.join(CRON_JOBS_FILENAME).exists());
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(snap.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["cron_jobs"]["backed_up"], json!(true));
        assert_eq!(manifest["cron_jobs"]["jobs_count"], json!(2));

        // Consolidation rewrites skill links + deletes j2.
        let mutated = json!({
            "jobs": [
                {"id": "j1", "name": "Job One", "skill": "umbrella-skill", "schedule": "* * * * *"},
                {"id": "j3", "name": "New Job", "skill": "fresh"}
            ],
            "updated_at": "2026-02-02T00:00:00Z"
        });
        fs::write(
            cron_dir.join("jobs.json"),
            serde_json::to_string_pretty(&mutated).unwrap(),
        )
        .unwrap();

        let report = restore_cron_skill_links(&home, &snap);
        assert!(report.attempted);
        assert_eq!(report.restored.len(), 1); // j1 skill restored
        assert_eq!(report.skipped_missing.len(), 1); // j2 gone
        assert!(report.error.is_none());

        // Verify j1 skill restored, schedule untouched, j3 untouched.
        let after: Value =
            serde_json::from_str(&fs::read_to_string(cron_dir.join("jobs.json")).unwrap())
                .unwrap();
        let jobs_arr = after["jobs"].as_array().unwrap();
        let j1 = jobs_arr.iter().find(|j| j["id"] == json!("j1")).unwrap();
        assert_eq!(j1["skill"], json!("narrow-skill"));
        assert_eq!(j1["schedule"], json!("* * * * *"));
        let j3 = jobs_arr.iter().find(|j| j["id"] == json!("j3")).unwrap();
        assert_eq!(j3["skill"], json!("fresh"));
        // updated_at preserved (we only touch skills/skill).
        assert_eq!(after["updated_at"], json!("2026-02-02T00:00:00Z"));

        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn prune_keeps_newest() {
        let home = tmp_home();
        let backups = backups_dir(&home);
        fs::create_dir_all(&backups).unwrap();
        let ids = [
            "2026-01-01T00-00-00Z",
            "2026-01-02T00-00-00Z",
            "2026-01-03T00-00-00Z",
            "2026-01-04T00-00-00Z",
        ];
        for id in ids {
            let d = backups.join(id);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("skills.tar.gz"), b"x").unwrap();
        }
        // stale staging dir should be cleaned up
        let stale = backups.join(".rollback-staging-2026-01-01T00-00-00Z");
        fs::create_dir_all(&stale).unwrap();

        let deleted = prune_old(&home, 2);
        assert_eq!(deleted.len(), 2);
        assert!(deleted.contains(&"2026-01-01T00-00-00Z".to_string()));
        assert!(deleted.contains(&"2026-01-02T00-00-00Z".to_string()));
        assert!(backups.join("2026-01-03T00-00-00Z").exists());
        assert!(backups.join("2026-01-04T00-00-00Z").exists());
        assert!(!stale.exists());

        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn resolve_backup_newest_when_none() {
        let home = tmp_home();
        let backups = backups_dir(&home);
        fs::create_dir_all(&backups).unwrap();
        for id in ["2026-01-01T00-00-00Z", "2026-01-05T00-00-00Z"] {
            let d = backups.join(id);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("skills.tar.gz"), b"x").unwrap();
        }
        let resolved = resolve_backup(&home, None).unwrap();
        assert_eq!(
            resolved.file_name().unwrap().to_string_lossy(),
            "2026-01-05T00-00-00Z"
        );
        fs::remove_dir_all(&home).ok();
    }
}
