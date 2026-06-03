//! Cron job storage and management.
//!
//! Faithful native-Rust port of `cron/jobs.py`.
//!
//! Jobs are stored in `~/.hermes/cron/jobs.json`.
//! Output is saved to `~/.hermes/cron/output/{job_id}/{timestamp}.md`.
//!
//! Jobs are modelled as [`serde_json::Value`] objects (JSON objects), matching
//! the dynamic dict semantics of the Python source: extra fields are preserved
//! across load/save, partial updates merge into existing dicts, and missing
//! keys default the same way Python's `dict.get(..., default)` does.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Duration, Local, NaiveDateTime, TimeZone};
use cronexpr::{FallbackTimezoneOption, ParseOptions, parse_crontab_with};
use serde_json::{Map, Value, json};

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::atomic_replace;

// =============================================================================
// Configuration
// =============================================================================

pub const ONESHOT_GRACE_SECONDS: i64 = 120;

/// In-process lock protecting load_jobs->modify->save_jobs cycles.
///
/// Required when tick() runs jobs in parallel threads — without this,
/// concurrent mark_job_run / advance_next_run calls can clobber each other.
static JOBS_FILE_LOCK: Mutex<()> = Mutex::new(());

/// Resolved `~/.hermes` directory.
pub fn hermes_dir() -> PathBuf {
    let home = get_hermes_home();
    fs::canonicalize(&home).unwrap_or(home)
}

/// `~/.hermes/cron`.
pub fn cron_dir() -> PathBuf {
    hermes_dir().join("cron")
}

/// `~/.hermes/cron/jobs.json`.
pub fn jobs_file() -> PathBuf {
    cron_dir().join("jobs.json")
}

/// `~/.hermes/cron/output`.
pub fn output_dir() -> PathBuf {
    cron_dir().join("output")
}

// =============================================================================
// Time helpers
// =============================================================================

/// Equivalent of `hermes_time.now()`: the current time in the configured
/// (here: system local) timezone.
fn hermes_now() -> DateTime<Local> {
    Local::now()
}

/// Parse an ISO-8601-ish timestamp into a local `DateTime`.
///
/// Mirrors `_ensure_aware(datetime.fromisoformat(...))`: aware timestamps are
/// converted to local time; naive timestamps are interpreted as local wall
/// time (the same timezone `datetime.now()` produced when they were written).
fn ensure_aware(value: &str) -> Result<DateTime<Local>, String> {
    let trimmed = value.trim();
    // Try an offset-aware RFC3339 first.
    if let Ok(parsed) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(parsed.with_timezone(&Local));
    }
    // Handle a trailing Z that some producers emit without colon-offset.
    if let Some(stripped) = trimmed.strip_suffix('Z') {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(&format!("{stripped}+00:00")) {
            return Ok(parsed.with_timezone(&Local));
        }
    }
    // Naive forms: interpret as local wall time.
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, fmt) {
            if let Some(local) = Local
                .from_local_datetime(&naive)
                .single()
                .or_else(|| Local.from_local_datetime(&naive).earliest())
            {
                return Ok(local);
            }
        }
    }
    Err(format!("Invalid timestamp '{value}'"))
}

/// `datetime.fromisoformat(s).isoformat()` round-trip in local tz used by
/// `parse_schedule` for ISO timestamps.
fn parse_isoish_timestamp(value: &str) -> Result<Option<DateTime<Local>>, String> {
    let trimmed = value.trim();
    let looks_dateish = trimmed.contains('T') || iso_date_prefix(trimmed);
    if !looks_dateish {
        return Ok(None);
    }
    let normalized = trimmed.replace('Z', "+00:00");
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&normalized) {
        return Ok(Some(parsed.with_timezone(&Local)));
    }
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&normalized, fmt) {
            if let Some(local) = Local
                .from_local_datetime(&naive)
                .single()
                .or_else(|| Local.from_local_datetime(&naive).earliest())
            {
                return Ok(Some(local));
            }
        }
        // Date-only via NaiveDate.
        if fmt == "%Y-%m-%d" {
            if let Ok(date) = chrono::NaiveDate::parse_from_str(&normalized, fmt) {
                let naive = date.and_hms_opt(0, 0, 0).unwrap();
                if let Some(local) = Local
                    .from_local_datetime(&naive)
                    .single()
                    .or_else(|| Local.from_local_datetime(&naive).earliest())
                {
                    return Ok(Some(local));
                }
            }
        }
    }
    Err(format!("Invalid timestamp '{trimmed}'"))
}

/// True when `s` begins with `YYYY-MM-DD` (the `re.match(r'^\d{4}-\d{2}-\d{2}')`
/// check from the Python source).
fn iso_date_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 10
        && b[0].is_ascii_digit()
        && b[1].is_ascii_digit()
        && b[2].is_ascii_digit()
        && b[3].is_ascii_digit()
        && b[4] == b'-'
        && b[5].is_ascii_digit()
        && b[6].is_ascii_digit()
        && b[7] == b'-'
        && b[8].is_ascii_digit()
        && b[9].is_ascii_digit()
}

// =============================================================================
// Filesystem permissions
// =============================================================================

#[cfg(unix)]
fn secure_dir(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn secure_dir(_path: &Path) {}

#[cfg(unix)]
fn secure_file(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if path.exists() {
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
}

#[cfg(not(unix))]
fn secure_file(_path: &Path) {}

/// Ensure cron directories exist with secure permissions.
pub fn ensure_dirs() -> std::io::Result<()> {
    let cron = cron_dir();
    let output = output_dir();
    fs::create_dir_all(&cron)?;
    fs::create_dir_all(&output)?;
    secure_dir(&cron);
    secure_dir(&output);
    Ok(())
}

// =============================================================================
// Skill list normalisation
// =============================================================================

/// Normalize legacy/single-skill and multi-skill inputs into a unique ordered
/// list. `skill` is the legacy single name; `skills` is a JSON value that may
/// be a string, an array, or null.
pub fn normalize_skill_list(skill: Option<&str>, skills: Option<&Value>) -> Vec<String> {
    let raw_items: Vec<String> = match skills {
        None | Some(Value::Null) => match skill {
            Some(s) if !s.is_empty() => vec![s.to_string()],
            _ => Vec::new(),
        },
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => value_to_str(other),
            })
            .collect(),
        Some(other) => vec![value_to_str(other)],
    };

    let mut normalized: Vec<String> = Vec::new();
    for item in raw_items {
        let text = item.trim().to_string();
        if !text.is_empty() && !normalized.contains(&text) {
            normalized.push(text);
        }
    }
    normalized
}

/// `str(item)` equivalent for non-string JSON scalars inside a skill list.
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => {
            // Python str(True) -> "True"
            if *b { "True".to_string() } else { "False".to_string() }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Return a job with canonical `skills` and legacy `skill` fields aligned.
pub fn apply_skill_fields(job: &Value) -> Value {
    let mut normalized = job.clone();
    let obj = match normalized.as_object_mut() {
        Some(o) => o,
        None => return normalized,
    };
    let skill = obj.get("skill").and_then(|v| v.as_str()).map(str::to_string);
    let skills_val = obj.get("skills").cloned();
    let skills = normalize_skill_list(skill.as_deref(), skills_val.as_ref());
    obj.insert(
        "skills".to_string(),
        Value::Array(skills.iter().map(|s| Value::String(s.clone())).collect()),
    );
    obj.insert(
        "skill".to_string(),
        skills.first().map(|s| Value::String(s.clone())).unwrap_or(Value::Null),
    );
    normalized
}

// =============================================================================
// Schedule parsing
// =============================================================================

/// Parse duration string into minutes. Examples: `"30m"` -> 30, `"2h"` -> 120,
/// `"1d"` -> 1440.
pub fn parse_duration(s: &str) -> Result<i64, String> {
    let lowered = s.trim().to_ascii_lowercase();
    let re = regex::Regex::new(
        r"^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
    )
    .unwrap();
    let caps = re.captures(&lowered).ok_or_else(|| {
        format!("Invalid duration: '{lowered}'. Use format like '30m', '2h', or '1d'")
    })?;
    let value: i64 = caps
        .get(1)
        .unwrap()
        .as_str()
        .parse()
        .map_err(|_| format!("Invalid duration number in '{lowered}'"))?;
    let unit = caps.get(2).unwrap().as_str().chars().next().unwrap();
    let mult = match unit {
        'm' => 1,
        'h' => 60,
        'd' => 1440,
        _ => unreachable!(),
    };
    Ok(value * mult)
}

/// True when `value` parses as a 5+ field cron expression (per the Python
/// `re.match(r'^[\d\*\-,/]+$', p)` test on the first five fields).
fn is_cron_expression(value: &str) -> bool {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() < 5 {
        return false;
    }
    parts[..5].iter().all(|p| {
        !p.is_empty()
            && p.chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '*' | '-' | ',' | '/'))
    })
}

/// Parse a schedule string into a structured JSON object. The result has a
/// `kind` of `"once" | "interval" | "cron"` plus the per-kind fields and a
/// `display` string, exactly mirroring the Python return shape.
pub fn parse_schedule(schedule: &str) -> Result<Value, String> {
    let schedule = schedule.trim();
    let original = schedule.to_string();
    let schedule_lower = schedule.to_ascii_lowercase();

    // "every X" pattern -> recurring interval. Mirror Python's slice on the
    // original string (duration parsing is case-insensitive anyway).
    if schedule_lower.starts_with("every ") {
        let duration_str = schedule[6..].trim();
        let minutes = parse_duration(duration_str)?;
        return Ok(json!({
            "kind": "interval",
            "minutes": minutes,
            "display": format!("every {minutes}m"),
        }));
    }

    // Cron expression (5 or 6 space-separated fields, first 5 are cron-shaped).
    if is_cron_expression(schedule) {
        let mut options = ParseOptions::default();
        options.fallback_timezone_option = FallbackTimezoneOption::System;
        parse_crontab_with(schedule, options)
            .map_err(|e| format!("Invalid cron expression '{schedule}': {e}"))?;
        return Ok(json!({
            "kind": "cron",
            "expr": schedule,
            "display": schedule,
        }));
    }

    // ISO timestamp.
    if schedule.contains('T') || iso_date_prefix(schedule) {
        match parse_isoish_timestamp(schedule) {
            Ok(Some(dt)) => {
                return Ok(json!({
                    "kind": "once",
                    "run_at": dt.to_rfc3339(),
                    "display": format!("once at {}", dt.format("%Y-%m-%d %H:%M")),
                }));
            }
            Ok(None) => {}
            Err(e) => return Err(format!("Invalid timestamp '{schedule}': {e}")),
        }
    }

    // Duration like "30m", "2h", "1d" -> one-shot from now.
    if let Ok(minutes) = parse_duration(schedule) {
        let run_at = hermes_now() + Duration::minutes(minutes);
        return Ok(json!({
            "kind": "once",
            "run_at": run_at.to_rfc3339(),
            "display": format!("once in {original}"),
        }));
    }

    Err(format!(
        "Invalid schedule '{original}'. Use:\n  - Duration: '30m', '2h', '1d' (one-shot)\n  - Interval: 'every 30m', 'every 2h' (recurring)\n  - Cron: '0 9 * * *' (cron expression)\n  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    ))
}

/// Return a one-shot run time if it is still eligible to fire.
fn recoverable_oneshot_run_at(
    schedule: &Value,
    now: DateTime<Local>,
    last_run_at: Option<&str>,
) -> Option<String> {
    if schedule.get("kind").and_then(|v| v.as_str()) != Some("once") {
        return None;
    }
    if last_run_at.map(|s| !s.is_empty()).unwrap_or(false) {
        return None;
    }
    let run_at = schedule.get("run_at").and_then(|v| v.as_str())?;
    if run_at.is_empty() {
        return None;
    }
    let run_at_dt = ensure_aware(run_at).ok()?;
    if run_at_dt >= now - Duration::seconds(ONESHOT_GRACE_SECONDS) {
        Some(run_at.to_string())
    } else {
        None
    }
}

/// Compute how late a job can be and still catch up instead of fast-forwarding.
fn compute_grace_seconds(schedule: &Value) -> i64 {
    const MIN_GRACE: i64 = 120;
    const MAX_GRACE: i64 = 7200;

    let kind = schedule.get("kind").and_then(|v| v.as_str());
    match kind {
        Some("interval") => {
            let minutes = schedule.get("minutes").and_then(|v| v.as_i64()).unwrap_or(1);
            let grace = (minutes * 60) / 2;
            grace.clamp(MIN_GRACE, MAX_GRACE)
        }
        Some("cron") => {
            if let Some(expr) = schedule.get("expr").and_then(|v| v.as_str()) {
                if let Ok(g) = cron_period_grace(expr) {
                    return g;
                }
            }
            MIN_GRACE
        }
        _ => MIN_GRACE,
    }
}

fn cron_period_grace(expr: &str) -> Result<i64, String> {
    let mut options = ParseOptions::default();
    options.fallback_timezone_option = FallbackTimezoneOption::System;
    let crontab =
        parse_crontab_with(expr, options).map_err(|e| format!("Invalid cron expression: {e}"))?;
    let now = hermes_now();
    let first = crontab
        .find_next(now.to_rfc3339().as_str())
        .map_err(|e| format!("cron grace: {e}"))?;
    let second = crontab
        .find_next(first.to_string().as_str())
        .map_err(|e| format!("cron grace: {e}"))?;
    let first_dt = ensure_aware(&first.to_string())?;
    let second_dt = ensure_aware(&second.to_string())?;
    let period = (second_dt - first_dt).num_seconds();
    Ok((period / 2).clamp(120, 7200))
}

/// Compute the next run time for a schedule. Returns an ISO timestamp string,
/// or `None` if no more runs.
pub fn compute_next_run(schedule: &Value, last_run_at: Option<&str>) -> Option<String> {
    let now = hermes_now();
    let kind = schedule.get("kind").and_then(|v| v.as_str())?;

    match kind {
        "once" => recoverable_oneshot_run_at(schedule, now, last_run_at),
        "interval" => {
            let minutes = schedule.get("minutes").and_then(|v| v.as_i64()).unwrap_or(0);
            let next_run = match last_run_at {
                Some(lr) if !lr.is_empty() => {
                    let last = ensure_aware(lr).ok()?;
                    last + Duration::minutes(minutes)
                }
                _ => now + Duration::minutes(minutes),
            };
            Some(next_run.to_rfc3339())
        }
        "cron" => {
            let expr = schedule.get("expr").and_then(|v| v.as_str())?;
            let base = match last_run_at {
                Some(lr) if !lr.is_empty() => ensure_aware(lr).ok()?,
                _ => now,
            };
            let mut options = ParseOptions::default();
            options.fallback_timezone_option = FallbackTimezoneOption::System;
            let crontab = parse_crontab_with(expr, options).ok()?;
            let next = crontab.find_next(base.to_rfc3339().as_str()).ok()?;
            Some(next.to_string())
        }
        _ => None,
    }
}

// =============================================================================
// Job storage CRUD
// =============================================================================

/// Load all jobs from storage.
pub fn load_jobs() -> Result<Vec<Value>, String> {
    ensure_dirs().map_err(|e| format!("Failed to ensure cron dirs: {e}"))?;
    let path = jobs_file();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path).map_err(|e| {
        log::error!("IOError reading jobs.json: {e}");
        format!("Failed to read cron database: {e}")
    })?;

    match serde_json::from_str::<Value>(&text) {
        Ok(data) => Ok(extract_jobs(&data)),
        Err(_) => {
            // serde_json already accepts bare control chars in some cases; on a
            // genuine parse failure mirror the Python "auto-repair" path: if we
            // can recover a jobs array, rewrite it cleanly.
            match serde_json::from_str::<Value>(&text) {
                Ok(data) => {
                    let jobs = extract_jobs(&data);
                    if !jobs.is_empty() {
                        save_jobs(&jobs)?;
                        log::warn!("Auto-repaired jobs.json (had invalid control characters)");
                    }
                    Ok(jobs)
                }
                Err(e) => {
                    log::error!("Failed to auto-repair jobs.json: {e}");
                    Err(format!("Cron database corrupted and unrepairable: {e}"))
                }
            }
        }
    }
}

fn extract_jobs(data: &Value) -> Vec<Value> {
    data.get("jobs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

/// Save all jobs to storage atomically.
pub fn save_jobs(jobs: &[Value]) -> Result<(), String> {
    ensure_dirs().map_err(|e| format!("Failed to ensure cron dirs: {e}"))?;
    let path = jobs_file();
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_else(|| cron_dir());

    let payload = json!({
        "jobs": jobs,
        "updated_at": hermes_now().to_rfc3339(),
    });
    let serialized =
        serde_json::to_string_pretty(&payload).map_err(|e| format!("serialize jobs: {e}"))?;

    let tmp_path = mkstemp(&parent, ".jobs_", ".tmp")
        .map_err(|e| format!("Failed to create temp file: {e}"))?;
    let result = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(serialized.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        atomic_replace(&tmp_path, &path)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            secure_file(&path);
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(format!("Failed to save jobs: {e}"))
        }
    }
}

/// Create a uniquely-named temp file in `dir` with the given prefix/suffix
/// (mirrors `tempfile.mkstemp`). On Unix it is created with `0o600`.
fn mkstemp(dir: &Path, prefix: &str, suffix: &str) -> std::io::Result<PathBuf> {
    use std::fs::OpenOptions;
    for _ in 0..10_000u64 {
        let mut buf = [0u8; 8];
        getrandom_bytes(&mut buf);
        let token = u64::from_le_bytes(buf);
        let candidate = dir.join(format!("{prefix}{token:016x}{suffix}"));
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&candidate) {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other("mkstemp: exhausted attempts"))
}

fn getrandom_bytes(buf: &mut [u8]) {
    if getrandom::fill(buf).is_err() {
        // Fallback: nanosecond-seeded fill (never expected in practice).
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((nanos >> (i % 16 * 8)) & 0xff) as u8;
        }
    }
}

/// Generate a 12-char lowercase-hex job id (`uuid.uuid4().hex[:12]`).
fn new_job_id() -> String {
    let mut buf = [0u8; 6];
    getrandom_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// Normalize and validate a cron job workdir. Empty/None -> None. `~` is
/// expanded; relative paths are rejected; the path must exist and be a dir.
pub fn normalize_workdir(workdir: Option<&str>) -> Result<Option<String>, String> {
    let raw = match workdir {
        None => return Ok(None),
        Some(s) => s.trim(),
    };
    if raw.is_empty() {
        return Ok(None);
    }
    let expanded = expanduser(raw);
    if !expanded.is_absolute() {
        return Err(format!(
            "Cron workdir must be an absolute path (got {raw:?}). Cron jobs run detached from any shell cwd, so relative paths are ambiguous."
        ));
    }
    let resolved = fs::canonicalize(&expanded).unwrap_or(expanded.clone());
    if !resolved.exists() {
        return Err(format!("Cron workdir does not exist: {}", resolved.display()));
    }
    if !resolved.is_dir() {
        return Err(format!(
            "Cron workdir is not a directory: {}",
            resolved.display()
        ));
    }
    Ok(Some(resolved.to_string_lossy().to_string()))
}

fn expanduser(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(path)
}

/// Options for [`create_job`]. All fields mirror the Python keyword arguments.
#[derive(Debug, Default, Clone)]
pub struct CreateJobParams {
    pub prompt: Option<String>,
    pub schedule: String,
    pub name: Option<String>,
    pub repeat: Option<i64>,
    pub deliver: Option<String>,
    pub origin: Option<Value>,
    pub skill: Option<String>,
    pub skills: Option<Vec<String>>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub script: Option<String>,
    /// Single id or list of ids; pass a JSON string or array.
    pub context_from: Option<Value>,
    pub enabled_toolsets: Option<Vec<String>>,
    pub workdir: Option<String>,
    pub no_agent: bool,
}

/// Create a new cron job and persist it. Returns the created job dict.
pub fn create_job(params: CreateJobParams) -> Result<Value, String> {
    let parsed_schedule = parse_schedule(&params.schedule)?;

    // Normalize repeat: treat 0 or negative as None (infinite).
    let mut repeat = params.repeat.filter(|&r| r > 0);

    // Auto-set repeat=1 for one-shot schedules if not specified.
    if parsed_schedule.get("kind").and_then(|v| v.as_str()) == Some("once") && repeat.is_none() {
        repeat = Some(1);
    }

    // Default delivery to origin if available, otherwise local.
    let deliver = params.deliver.clone().unwrap_or_else(|| {
        if params.origin.is_some() {
            "origin".to_string()
        } else {
            "local".to_string()
        }
    });

    let job_id = new_job_id();
    let now = hermes_now().to_rfc3339();

    let skills_val = params
        .skills
        .as_ref()
        .map(|v| Value::Array(v.iter().map(|s| Value::String(s.clone())).collect()));
    let normalized_skills = normalize_skill_list(params.skill.as_deref(), skills_val.as_ref());

    let normalized_model = opt_trimmed(params.model.as_deref());
    let normalized_provider = opt_trimmed(params.provider.as_deref());
    let normalized_base_url =
        opt_trimmed(params.base_url.as_deref()).map(|s| s.trim_end_matches('/').to_string());
    let normalized_script = opt_trimmed(params.script.as_deref());

    let normalized_toolsets: Option<Vec<String>> = params.enabled_toolsets.as_ref().and_then(|v| {
        let filtered: Vec<String> = v
            .iter()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    });

    let normalized_workdir = normalize_workdir(params.workdir.as_deref())?;
    let normalized_no_agent = params.no_agent;

    if normalized_no_agent && normalized_script.is_none() {
        return Err(
            "no_agent=True requires a script — with no agent and no script there is nothing for the job to run."
                .to_string(),
        );
    }

    // Normalize context_from: str or list of str -> list or None.
    let context_from: Value = match &params.context_from {
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                Value::Null
            } else {
                Value::Array(vec![Value::String(t.to_string())])
            }
        }
        Some(Value::Array(arr)) => {
            let items: Vec<Value> = arr
                .iter()
                .map(value_to_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(Value::String)
                .collect();
            if items.is_empty() {
                Value::Null
            } else {
                Value::Array(items)
            }
        }
        _ => Value::Null,
    };

    // label_source = prompt or first skill or (script if no_agent) or "cron job"
    let label_source = params
        .prompt
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| normalized_skills.first().cloned())
        .or_else(|| {
            if normalized_no_agent {
                normalized_script.clone()
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "cron job".to_string());

    let name = params.name.clone().unwrap_or_else(|| {
        // label_source[:50].strip()
        let truncated: String = label_source.chars().take(50).collect();
        truncated.trim().to_string()
    });

    let next_run_at = compute_next_run(&parsed_schedule, None)
        .map(Value::String)
        .unwrap_or(Value::Null);

    let schedule_display = parsed_schedule
        .get("display")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| params.schedule.clone());

    let mut job = Map::new();
    job.insert("id".into(), Value::String(job_id));
    job.insert("name".into(), Value::String(name));
    job.insert(
        "prompt".into(),
        params
            .prompt
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    job.insert(
        "skills".into(),
        Value::Array(normalized_skills.iter().map(|s| Value::String(s.clone())).collect()),
    );
    job.insert(
        "skill".into(),
        normalized_skills.first().map(|s| Value::String(s.clone())).unwrap_or(Value::Null),
    );
    job.insert("model".into(), opt_string(normalized_model));
    job.insert("provider".into(), opt_string(normalized_provider));
    job.insert("base_url".into(), opt_string(normalized_base_url));
    job.insert("script".into(), opt_string(normalized_script));
    job.insert("no_agent".into(), Value::Bool(normalized_no_agent));
    job.insert("context_from".into(), context_from);
    job.insert("schedule".into(), parsed_schedule.clone());
    job.insert("schedule_display".into(), Value::String(schedule_display));
    job.insert(
        "repeat".into(),
        json!({
            "times": repeat,
            "completed": 0,
        }),
    );
    job.insert("enabled".into(), Value::Bool(true));
    job.insert("state".into(), Value::String("scheduled".into()));
    job.insert("paused_at".into(), Value::Null);
    job.insert("paused_reason".into(), Value::Null);
    job.insert("created_at".into(), Value::String(now));
    job.insert("next_run_at".into(), next_run_at);
    job.insert("last_run_at".into(), Value::Null);
    job.insert("last_status".into(), Value::Null);
    job.insert("last_error".into(), Value::Null);
    job.insert("last_delivery_error".into(), Value::Null);
    job.insert("deliver".into(), Value::String(deliver));
    job.insert(
        "origin".into(),
        params.origin.clone().unwrap_or(Value::Null),
    );
    job.insert(
        "enabled_toolsets".into(),
        normalized_toolsets
            .map(|v| Value::Array(v.into_iter().map(Value::String).collect()))
            .unwrap_or(Value::Null),
    );
    job.insert("workdir".into(), opt_string(normalized_workdir));

    let job = Value::Object(job);

    let mut jobs = load_jobs()?;
    jobs.push(job.clone());
    save_jobs(&jobs)?;

    Ok(job)
}

fn opt_trimmed(s: Option<&str>) -> Option<String> {
    s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn opt_string(s: Option<String>) -> Value {
    s.map(Value::String).unwrap_or(Value::Null)
}

/// Get a job by id (with skill fields aligned).
pub fn get_job(job_id: &str) -> Result<Option<Value>, String> {
    let jobs = load_jobs()?;
    for job in &jobs {
        if job.get("id").and_then(|v| v.as_str()) == Some(job_id) {
            return Ok(Some(apply_skill_fields(job)));
        }
    }
    Ok(None)
}

/// List all jobs, optionally including disabled ones.
pub fn list_jobs(include_disabled: bool) -> Result<Vec<Value>, String> {
    let jobs: Vec<Value> = load_jobs()?.iter().map(apply_skill_fields).collect();
    if include_disabled {
        Ok(jobs)
    } else {
        Ok(jobs
            .into_iter()
            .filter(|j| j.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true))
            .collect())
    }
}

/// Merge `b` into `a` (shallow), like Python `{**a, **b}`.
fn merge_dicts(a: &Value, b: &Map<String, Value>) -> Value {
    let mut out = a.as_object().cloned().unwrap_or_default();
    for (k, v) in b {
        out.insert(k.clone(), v.clone());
    }
    Value::Object(out)
}

/// Update a job by id, refreshing derived schedule fields when needed.
/// `updates` is a JSON object of fields to merge.
pub fn update_job(job_id: &str, updates: &Value) -> Result<Option<Value>, String> {
    let mut updates_map = updates
        .as_object()
        .cloned()
        .ok_or_else(|| "updates must be an object".to_string())?;

    let mut jobs = load_jobs()?;
    for i in 0..jobs.len() {
        if jobs[i].get("id").and_then(|v| v.as_str()) != Some(job_id) {
            continue;
        }

        // Validate / normalize workdir if present in updates.
        if updates_map.contains_key("workdir") {
            let wd = updates_map.get("workdir").cloned().unwrap_or(Value::Null);
            let cleared = matches!(&wd, Value::Null)
                || matches!(&wd, Value::String(s) if s.is_empty())
                || matches!(&wd, Value::Bool(false));
            if cleared {
                updates_map.insert("workdir".into(), Value::Null);
            } else {
                let s = wd.as_str().map(str::to_string).unwrap_or_else(|| value_to_str(&wd));
                let normalized = normalize_workdir(Some(&s))?;
                updates_map.insert("workdir".into(), opt_string(normalized));
            }
        }

        let mut updated = apply_skill_fields(&merge_dicts(&jobs[i], &updates_map));
        let schedule_changed = updates_map.contains_key("schedule");

        if updates_map.contains_key("skills") || updates_map.contains_key("skill") {
            let skill = updated.get("skill").and_then(|v| v.as_str()).map(str::to_string);
            let skills_val = updated.get("skills").cloned();
            let normalized = normalize_skill_list(skill.as_deref(), skills_val.as_ref());
            let obj = updated.as_object_mut().unwrap();
            obj.insert(
                "skills".into(),
                Value::Array(normalized.iter().map(|s| Value::String(s.clone())).collect()),
            );
            obj.insert(
                "skill".into(),
                normalized.first().map(|s| Value::String(s.clone())).unwrap_or(Value::Null),
            );
        }

        if schedule_changed {
            // Schedule may arrive as a raw string; normalize like create_job.
            let sched = updated.get("schedule").cloned().unwrap_or(Value::Null);
            let updated_schedule = if let Value::String(s) = &sched {
                let parsed = parse_schedule(s)?;
                updated
                    .as_object_mut()
                    .unwrap()
                    .insert("schedule".into(), parsed.clone());
                parsed
            } else {
                sched
            };

            let display = match updates_map.get("schedule_display") {
                Some(v) => v.clone(),
                None => updated_schedule
                    .get("display")
                    .cloned()
                    .or_else(|| updated.get("schedule_display").cloned())
                    .unwrap_or(Value::Null),
            };
            updated
                .as_object_mut()
                .unwrap()
                .insert("schedule_display".into(), display);

            if updated.get("state").and_then(|v| v.as_str()) != Some("paused") {
                let nr = compute_next_run(&updated_schedule, None)
                    .map(Value::String)
                    .unwrap_or(Value::Null);
                updated.as_object_mut().unwrap().insert("next_run_at".into(), nr);
            }
        }

        let enabled = updated.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
        let not_paused = updated.get("state").and_then(|v| v.as_str()) != Some("paused");
        let no_next = updated
            .get("next_run_at")
            .map(|v| v.is_null() || matches!(v, Value::String(s) if s.is_empty()))
            .unwrap_or(true);
        if enabled && not_paused && no_next {
            let sched = updated.get("schedule").cloned().unwrap_or(Value::Null);
            let nr = compute_next_run(&sched, None)
                .map(Value::String)
                .unwrap_or(Value::Null);
            updated.as_object_mut().unwrap().insert("next_run_at".into(), nr);
        }

        jobs[i] = updated;
        save_jobs(&jobs)?;
        return Ok(Some(apply_skill_fields(&jobs[i])));
    }
    Ok(None)
}

/// Pause a job without deleting it.
pub fn pause_job(job_id: &str, reason: Option<&str>) -> Result<Option<Value>, String> {
    update_job(
        job_id,
        &json!({
            "enabled": false,
            "state": "paused",
            "paused_at": hermes_now().to_rfc3339(),
            "paused_reason": reason,
        }),
    )
}

/// Resume a paused job and compute the next future run from now.
pub fn resume_job(job_id: &str) -> Result<Option<Value>, String> {
    let job = match get_job(job_id)? {
        Some(j) => j,
        None => return Ok(None),
    };
    let schedule = job.get("schedule").cloned().unwrap_or(Value::Null);
    let next_run_at = compute_next_run(&schedule, None)
        .map(Value::String)
        .unwrap_or(Value::Null);
    update_job(
        job_id,
        &json!({
            "enabled": true,
            "state": "scheduled",
            "paused_at": Value::Null,
            "paused_reason": Value::Null,
            "next_run_at": next_run_at,
        }),
    )
}

/// Schedule a job to run on the next scheduler tick.
pub fn trigger_job(job_id: &str) -> Result<Option<Value>, String> {
    if get_job(job_id)?.is_none() {
        return Ok(None);
    }
    update_job(
        job_id,
        &json!({
            "enabled": true,
            "state": "scheduled",
            "paused_at": Value::Null,
            "paused_reason": Value::Null,
            "next_run_at": hermes_now().to_rfc3339(),
        }),
    )
}

/// Remove a job by id. Returns true if a job was removed.
pub fn remove_job(job_id: &str) -> Result<bool, String> {
    let mut jobs = load_jobs()?;
    let original_len = jobs.len();
    jobs.retain(|j| j.get("id").and_then(|v| v.as_str()) != Some(job_id));
    if jobs.len() < original_len {
        save_jobs(&jobs)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Mark a job as having been run, updating status/repeat/next_run and possibly
/// auto-deleting it when the repeat limit is reached.
pub fn mark_job_run(
    job_id: &str,
    success: bool,
    error: Option<&str>,
    delivery_error: Option<&str>,
) -> Result<(), String> {
    let _guard = JOBS_FILE_LOCK.lock().unwrap();
    let mut jobs = load_jobs()?;
    for i in 0..jobs.len() {
        if jobs[i].get("id").and_then(|v| v.as_str()) != Some(job_id) {
            continue;
        }
        let now = hermes_now().to_rfc3339();
        {
            let job = jobs[i].as_object_mut().unwrap();
            job.insert("last_run_at".into(), Value::String(now.clone()));
            job.insert(
                "last_status".into(),
                Value::String(if success { "ok" } else { "error" }.into()),
            );
            job.insert(
                "last_error".into(),
                if !success {
                    error.map(|e| Value::String(e.to_string())).unwrap_or(Value::Null)
                } else {
                    Value::Null
                },
            );
            job.insert(
                "last_delivery_error".into(),
                delivery_error
                    .map(|e| Value::String(e.to_string()))
                    .unwrap_or(Value::Null),
            );
        }

        // Increment completed count.
        let repeat_is_truthy = jobs[i]
            .get("repeat")
            .map(|r| !matches!(r, Value::Null) && !matches!(r, Value::Bool(false)))
            .unwrap_or(false);
        if repeat_is_truthy {
            let (completed, times) = {
                let repeat = jobs[i]
                    .get_mut("repeat")
                    .and_then(|v| v.as_object_mut())
                    .unwrap();
                let completed = repeat.get("completed").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                repeat.insert("completed".into(), json!(completed));
                let times = repeat.get("times").and_then(|v| v.as_i64());
                (completed, times)
            };
            if let Some(t) = times {
                if t > 0 && completed >= t {
                    jobs.remove(i);
                    save_jobs(&jobs)?;
                    return Ok(());
                }
            }
        }

        // Compute next run.
        let schedule = jobs[i].get("schedule").cloned().unwrap_or(Value::Null);
        let next = compute_next_run(&schedule, Some(&now));
        jobs[i].as_object_mut().unwrap().insert(
            "next_run_at".into(),
            next.clone().map(Value::String).unwrap_or(Value::Null),
        );

        if next.is_none() {
            let kind = jobs[i]
                .get("schedule")
                .and_then(|s| s.get("kind"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            if matches!(kind.as_deref(), Some("cron") | Some("interval")) {
                let job = jobs[i].as_object_mut().unwrap();
                job.insert("state".into(), Value::String("error".into()));
                let has_error = job
                    .get("last_error")
                    .map(|v| !v.is_null() && !matches!(v, Value::String(s) if s.is_empty()))
                    .unwrap_or(false);
                if !has_error {
                    job.insert(
                        "last_error".into(),
                        Value::String(
                            "Failed to compute next run for recurring schedule (is the 'croniter' package installed in the gateway's Python env?)"
                                .into(),
                        ),
                    );
                }
                let name = jobs[i]
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| job_id.to_string());
                log::error!(
                    "Job '{}' ({}) could not compute next_run_at; leaving enabled and marking state=error so the job is not silently disabled.",
                    name,
                    kind.as_deref().unwrap_or("")
                );
            } else {
                let job = jobs[i].as_object_mut().unwrap();
                job.insert("enabled".into(), Value::Bool(false));
                job.insert("state".into(), Value::String("completed".into()));
            }
        } else if jobs[i].get("state").and_then(|v| v.as_str()) != Some("paused") {
            jobs[i]
                .as_object_mut()
                .unwrap()
                .insert("state".into(), Value::String("scheduled".into()));
        }

        save_jobs(&jobs)?;
        return Ok(());
    }

    log::warn!("mark_job_run: job_id {job_id} not found, skipping save");
    Ok(())
}

/// Preemptively advance `next_run_at` for a recurring job before execution.
/// Returns true if `next_run_at` was advanced.
pub fn advance_next_run(job_id: &str) -> Result<bool, String> {
    let _guard = JOBS_FILE_LOCK.lock().unwrap();
    let mut jobs = load_jobs()?;
    for i in 0..jobs.len() {
        if jobs[i].get("id").and_then(|v| v.as_str()) != Some(job_id) {
            continue;
        }
        let kind = jobs[i]
            .get("schedule")
            .and_then(|s| s.get("kind"))
            .and_then(|v| v.as_str());
        if !matches!(kind, Some("cron") | Some("interval")) {
            return Ok(false);
        }
        let now = hermes_now().to_rfc3339();
        let schedule = jobs[i].get("schedule").cloned().unwrap_or(Value::Null);
        let new_next = compute_next_run(&schedule, Some(&now));
        let current = jobs[i].get("next_run_at").and_then(|v| v.as_str()).map(str::to_string);
        if let Some(nn) = new_next {
            if Some(&nn) != current.as_ref() {
                jobs[i]
                    .as_object_mut()
                    .unwrap()
                    .insert("next_run_at".into(), Value::String(nn));
                save_jobs(&jobs)?;
                return Ok(true);
            }
        }
        return Ok(false);
    }
    Ok(false)
}

/// Get all jobs that are due to run now (with stale-run fast-forwarding for
/// recurring jobs).
pub fn get_due_jobs() -> Result<Vec<Value>, String> {
    let _guard = JOBS_FILE_LOCK.lock().unwrap();
    get_due_jobs_locked()
}

fn get_due_jobs_locked() -> Result<Vec<Value>, String> {
    let now = hermes_now();
    let mut raw_jobs = load_jobs()?;
    let jobs: Vec<Value> = raw_jobs.iter().map(apply_skill_fields).collect();
    let mut due: Vec<Value> = Vec::new();
    let mut needs_save = false;

    for mut job in jobs {
        let enabled = job.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
        if !enabled {
            continue;
        }

        let job_id = job.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let mut next_run = job
            .get("next_run_at")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        if next_run.is_none() {
            let schedule = job.get("schedule").cloned().unwrap_or(Value::Null);
            let kind = schedule.get("kind").and_then(|v| v.as_str()).map(str::to_string);
            let last_run_at = job.get("last_run_at").and_then(|v| v.as_str());

            let mut recovered_next =
                recoverable_oneshot_run_at(&schedule, now, last_run_at);

            if recovered_next.is_none()
                && matches!(kind.as_deref(), Some("cron") | Some("interval"))
            {
                recovered_next = compute_next_run(&schedule, Some(&now.to_rfc3339()));
            }

            let recovered_next = match recovered_next {
                Some(r) => r,
                None => continue,
            };

            job.as_object_mut()
                .unwrap()
                .insert("next_run_at".into(), Value::String(recovered_next.clone()));
            next_run = Some(recovered_next.clone());
            log::info!(
                "Job '{}' had no next_run_at; recovering run at {}",
                job.get("name").and_then(|v| v.as_str()).unwrap_or(&job_id),
                recovered_next
            );
            for rj in raw_jobs.iter_mut() {
                if rj.get("id").and_then(|v| v.as_str()) == Some(job_id.as_str()) {
                    rj.as_object_mut()
                        .unwrap()
                        .insert("next_run_at".into(), Value::String(recovered_next.clone()));
                    needs_save = true;
                    break;
                }
            }
        }

        let next_run = next_run.unwrap();
        let next_run_dt = match ensure_aware(&next_run) {
            Ok(dt) => dt,
            Err(_) => continue,
        };

        if next_run_dt <= now {
            let schedule = job.get("schedule").cloned().unwrap_or(Value::Null);
            let kind = schedule.get("kind").and_then(|v| v.as_str()).map(str::to_string);

            let grace = compute_grace_seconds(&schedule);
            let late = (now - next_run_dt).num_seconds();
            if matches!(kind.as_deref(), Some("cron") | Some("interval")) && late > grace {
                let new_next = compute_next_run(&schedule, Some(&now.to_rfc3339()));
                if let Some(nn) = new_next {
                    log::info!(
                        "Job '{}' missed its scheduled time ({}, grace={}s). Fast-forwarding to next run: {}",
                        job.get("name").and_then(|v| v.as_str()).unwrap_or(&job_id),
                        next_run,
                        grace,
                        nn
                    );
                    for rj in raw_jobs.iter_mut() {
                        if rj.get("id").and_then(|v| v.as_str()) == Some(job_id.as_str()) {
                            rj.as_object_mut()
                                .unwrap()
                                .insert("next_run_at".into(), Value::String(nn.clone()));
                            needs_save = true;
                            break;
                        }
                    }
                    continue;
                }
            }

            due.push(job);
        }
    }

    if needs_save {
        save_jobs(&raw_jobs)?;
    }

    Ok(due)
}

/// Save job output to a timestamped markdown file. Returns the path written.
pub fn save_job_output(job_id: &str, output: &str) -> Result<PathBuf, String> {
    ensure_dirs().map_err(|e| format!("Failed to ensure cron dirs: {e}"))?;
    let job_output_dir = output_dir().join(job_id);
    fs::create_dir_all(&job_output_dir).map_err(|e| format!("create output dir: {e}"))?;
    secure_dir(&job_output_dir);

    let timestamp = hermes_now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_file = job_output_dir.join(format!("{timestamp}.md"));

    let tmp_path = mkstemp(&job_output_dir, ".output_", ".tmp")
        .map_err(|e| format!("create temp output: {e}"))?;
    let result = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(output.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        atomic_replace(&tmp_path, &output_file)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            secure_file(&output_file);
            Ok(output_file)
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(format!("Failed to save job output: {e}"))
        }
    }
}

// =============================================================================
// Skill reference rewriting (curator integration)
// =============================================================================

/// Rewrite cron job skill references after a curator consolidation pass.
///
/// `consolidated` maps old skill name -> umbrella target. `pruned` lists skill
/// names that were archived with no forwarding target. Returns a report JSON
/// object matching the Python shape (`rewrites`, `jobs_updated`,
/// `jobs_scanned`).
pub fn rewrite_skill_refs(
    consolidated: Option<&Map<String, Value>>,
    pruned: Option<&[String]>,
) -> Result<Value, String> {
    let consolidated: Map<String, Value> = consolidated.cloned().unwrap_or_default();
    let mut pruned_set: std::collections::BTreeSet<String> =
        pruned.map(|p| p.iter().cloned().collect()).unwrap_or_default();
    // A skill in both wins as "consolidated".
    for k in consolidated.keys() {
        pruned_set.remove(k);
    }

    if consolidated.is_empty() && pruned_set.is_empty() {
        return Ok(json!({"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}));
    }

    let _guard = JOBS_FILE_LOCK.lock().unwrap();
    let mut jobs = load_jobs()?;
    let mut rewrites: Vec<Value> = Vec::new();
    let mut changed = false;

    let jobs_scanned = jobs.len();

    for job in jobs.iter_mut() {
        let skill = job.get("skill").and_then(|v| v.as_str()).map(str::to_string);
        let skills_val = job.get("skills").cloned();
        let skills_before = normalize_skill_list(skill.as_deref(), skills_val.as_ref());
        if skills_before.is_empty() {
            continue;
        }

        let mut mapped: Map<String, Value> = Map::new();
        let mut dropped: Vec<String> = Vec::new();
        let mut new_skills: Vec<String> = Vec::new();

        for name in &skills_before {
            if let Some(target_v) = consolidated.get(name) {
                let target = target_v.as_str().unwrap_or("").to_string();
                mapped.insert(name.clone(), Value::String(target.clone()));
                if !target.is_empty() && !new_skills.contains(&target) {
                    new_skills.push(target);
                }
            } else if pruned_set.contains(name) {
                dropped.push(name.clone());
            } else if !new_skills.contains(name) {
                new_skills.push(name.clone());
            }
        }

        if mapped.is_empty() && dropped.is_empty() {
            continue;
        }

        let obj = job.as_object_mut().unwrap();
        obj.insert(
            "skills".into(),
            Value::Array(new_skills.iter().map(|s| Value::String(s.clone())).collect()),
        );
        obj.insert(
            "skill".into(),
            new_skills.first().map(|s| Value::String(s.clone())).unwrap_or(Value::Null),
        );
        changed = true;

        let job_id = obj.get("id").cloned().unwrap_or(Value::Null);
        let job_name = obj
            .get("name")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or_else(|| job_id.clone());

        rewrites.push(json!({
            "job_id": job_id,
            "job_name": job_name,
            "before": skills_before,
            "after": new_skills,
            "mapped": Value::Object(mapped),
            "dropped": dropped,
        }));
    }

    if changed {
        save_jobs(&jobs)?;
        log::info!("Curator rewrote skill references in {} cron job(s)", rewrites.len());
    }

    let jobs_updated = rewrites.len();
    Ok(json!({
        "rewrites": rewrites,
        "jobs_updated": jobs_updated,
        "jobs_scanned": jobs_scanned,
    }))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialise tests that mutate HERMES_HOME / the on-disk store.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn with_temp_home<F: FnOnce()>(f: F) {
        let _g = TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("hermes_cron_test_{}", new_job_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        f();
        match prev {
            Some(v) => unsafe { std::env::set_var("HERMES_HOME", v) },
            None => unsafe { std::env::remove_var("HERMES_HOME") },
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_duration_units() {
        assert_eq!(parse_duration("30m").unwrap(), 30);
        assert_eq!(parse_duration("2h").unwrap(), 120);
        assert_eq!(parse_duration("1d").unwrap(), 1440);
        assert_eq!(parse_duration("  15 MINUTES ").unwrap(), 15);
        assert!(parse_duration("bogus").is_err());
    }

    #[test]
    fn parse_schedule_interval() {
        let s = parse_schedule("every 30m").unwrap();
        assert_eq!(s["kind"], "interval");
        assert_eq!(s["minutes"], 30);
        assert_eq!(s["display"], "every 30m");
    }

    #[test]
    fn parse_schedule_cron() {
        let s = parse_schedule("0 9 * * *").unwrap();
        assert_eq!(s["kind"], "cron");
        assert_eq!(s["expr"], "0 9 * * *");
    }

    #[test]
    fn parse_schedule_oneshot_duration() {
        let s = parse_schedule("45m").unwrap();
        assert_eq!(s["kind"], "once");
        assert_eq!(s["display"], "once in 45m");
        assert!(s["run_at"].as_str().unwrap().len() > 0);
    }

    #[test]
    fn parse_schedule_timestamp() {
        let s = parse_schedule("2030-02-03T14:00:00").unwrap();
        assert_eq!(s["kind"], "once");
        assert!(s["display"].as_str().unwrap().starts_with("once at 2030-02-03"));
    }

    #[test]
    fn normalize_skill_list_dedup() {
        let skills = json!(["a", "a", " b ", "", "c"]);
        let out = normalize_skill_list(None, Some(&skills));
        assert_eq!(out, vec!["a", "b", "c"]);

        // legacy single
        let out2 = normalize_skill_list(Some("x"), None);
        assert_eq!(out2, vec!["x"]);

        // string form
        let out3 = normalize_skill_list(None, Some(&json!("solo")));
        assert_eq!(out3, vec!["solo"]);
    }

    #[test]
    fn apply_skill_fields_aligns() {
        let job = json!({"skills": ["foo", "bar"]});
        let out = apply_skill_fields(&job);
        assert_eq!(out["skill"], "foo");
        assert_eq!(out["skills"], json!(["foo", "bar"]));

        let empty = apply_skill_fields(&json!({}));
        assert_eq!(empty["skill"], Value::Null);
        assert_eq!(empty["skills"], json!([]));
    }

    #[test]
    fn create_get_list_remove_roundtrip() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("say hi".into()),
                schedule: "every 10m".into(),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            assert_eq!(id.len(), 12);
            assert_eq!(job["deliver"], "local");
            assert_eq!(job["enabled"], true);
            assert_eq!(job["repeat"]["completed"], 0);

            let fetched = get_job(&id).unwrap().unwrap();
            assert_eq!(fetched["prompt"], "say hi");

            let listed = list_jobs(false).unwrap();
            assert_eq!(listed.len(), 1);

            assert!(remove_job(&id).unwrap());
            assert!(get_job(&id).unwrap().is_none());
            assert!(!remove_job(&id).unwrap());
        });
    }

    #[test]
    fn oneshot_auto_repeat_one() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("once".into()),
                schedule: "5m".into(),
                ..Default::default()
            })
            .unwrap();
            assert_eq!(job["repeat"]["times"], 1);
            assert_eq!(job["schedule"]["kind"], "once");
        });
    }

    #[test]
    fn no_agent_requires_script() {
        with_temp_home(|| {
            let err = create_job(CreateJobParams {
                prompt: Some("x".into()),
                schedule: "every 5m".into(),
                no_agent: true,
                ..Default::default()
            })
            .unwrap_err();
            assert!(err.contains("no_agent"));
        });
    }

    #[test]
    fn mark_job_run_repeat_limit_deletes() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                repeat: Some(1),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            mark_job_run(&id, true, None, None).unwrap();
            // repeat limit 1 reached -> job removed.
            assert!(get_job(&id).unwrap().is_none());
        });
    }

    #[test]
    fn mark_job_run_recurring_advances() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                repeat: Some(3),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            mark_job_run(&id, true, None, None).unwrap();
            let updated = get_job(&id).unwrap().unwrap();
            assert_eq!(updated["last_status"], "ok");
            assert_eq!(updated["repeat"]["completed"], 1);
            assert_eq!(updated["state"], "scheduled");
            assert!(!updated["next_run_at"].is_null());
        });
    }

    #[test]
    fn pause_resume_cycle() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            let paused = pause_job(&id, Some("nap")).unwrap().unwrap();
            assert_eq!(paused["state"], "paused");
            assert_eq!(paused["enabled"], false);
            assert_eq!(paused["paused_reason"], "nap");

            let resumed = resume_job(&id).unwrap().unwrap();
            assert_eq!(resumed["state"], "scheduled");
            assert_eq!(resumed["enabled"], true);
            assert!(resumed["paused_at"].is_null());
        });
    }

    #[test]
    fn update_job_changes_schedule() {
        with_temp_home(|| {
            let job = create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            let updated = update_job(&id, &json!({"schedule": "every 30m"}))
                .unwrap()
                .unwrap();
            assert_eq!(updated["schedule"]["minutes"], 30);
            assert_eq!(updated["schedule_display"], "every 30m");
        });
    }

    #[test]
    fn rewrite_skill_refs_consolidate_and_prune() {
        with_temp_home(|| {
            create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                skills: Some(vec!["old".into(), "keep".into(), "gone".into()]),
                ..Default::default()
            })
            .unwrap();

            let mut consolidated = Map::new();
            consolidated.insert("old".into(), Value::String("umbrella".into()));
            let pruned = vec!["gone".to_string()];
            let report = rewrite_skill_refs(Some(&consolidated), Some(&pruned)).unwrap();
            assert_eq!(report["jobs_updated"], 1);
            assert_eq!(report["jobs_scanned"], 1);
            let after = &report["rewrites"][0]["after"];
            assert_eq!(*after, json!(["umbrella", "keep"]));
        });
    }

    #[test]
    fn get_due_jobs_returns_due() {
        with_temp_home(|| {
            // trigger_job sets next_run_at to now -> due immediately.
            let job = create_job(CreateJobParams {
                prompt: Some("p".into()),
                schedule: "every 10m".into(),
                ..Default::default()
            })
            .unwrap();
            let id = job["id"].as_str().unwrap().to_string();
            trigger_job(&id).unwrap();
            let due = get_due_jobs().unwrap();
            assert!(due.iter().any(|j| j["id"] == Value::String(id.clone())));
        });
    }

    #[test]
    fn save_job_output_writes_file() {
        with_temp_home(|| {
            let path = save_job_output("job123", "hello output").unwrap();
            assert!(path.exists());
            let content = std::fs::read_to_string(&path).unwrap();
            assert_eq!(content, "hello output");
        });
    }
}
