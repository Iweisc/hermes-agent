//! Cron subcommand for the hermes CLI.
//!
//! Faithful native-Rust port of `hermes_cli/cron.py`.
//!
//! Handles standalone cron management commands: list, create, edit,
//! pause/resume/run/remove, status, and tick.
//!
//! The Python module delegates to:
//!   - `cron.jobs.{list_jobs,get_job}` (raw stored job dicts), and
//!   - `tools.cronjob_tools.cronjob(**kwargs)` (the action-oriented tool,
//!     returning a JSON string), via the `_cron_api` helper.
//!
//! Here we map those onto the already-ported native cron storage layer in
//! [`hermes_core::cron_jobs`]. `_cron_api` is reimplemented directly on top of the
//! storage CRUD functions, reproducing the exact response shapes the CLI
//! rendering code reads (`success`, `job_id`, `name`, `schedule`, `skills`,
//! `next_run_at`, `job`, `removed_job`, `error`).
//!
//! `tick` (used by `cron tick`) lives in the scheduler, which requires full
//! agent context that this CLI module does not own; it is therefore injected
//! as a callback into [`cron_command`] / [`cron_tick`].

use serde_json::{json, Map, Value};

use crate::cli_colors::{color, Colors};
use crate::cli_uninstall::find_gateway_pids;
use hermes_core::cron_jobs;

// =============================================================================
// Argument structs (mirror argparse `args`)
// =============================================================================

/// Parsed arguments for the cron subcommand, mirroring the attributes the
/// Python code reads off the argparse `Namespace` via `getattr(args, ...)`.
#[derive(Debug, Default, Clone)]
pub struct CronArgs {
    /// Selected subcommand (`args.cron_command`). `None` => default to `list`.
    pub cron_command: Option<String>,
    /// `--all` flag for `list`.
    pub all: bool,

    // create / edit fields
    pub job_id: Option<String>,
    pub schedule: Option<String>,
    pub prompt: Option<String>,
    pub name: Option<String>,
    pub deliver: Option<String>,
    pub repeat: Option<i64>,
    pub skill: Option<String>,
    pub skills: Option<Vec<String>>,
    pub script: Option<String>,
    pub workdir: Option<String>,
    /// `--no-agent`; tri-state on edit (None = leave unchanged).
    pub no_agent: Option<bool>,

    // edit-only skill mutators
    pub add_skills: Option<Vec<String>>,
    pub remove_skills: Option<Vec<String>>,
    pub clear_skills: bool,
}

// =============================================================================
// Skill normalisation (`_normalize_skills`)
// =============================================================================

/// Port of `_normalize_skills(single_skill=None, skills=None)`.
///
/// When `skills` is `None`: if `single_skill` is also `None`, return `None`;
/// otherwise treat `[single_skill]` as the raw items. When `skills` is `Some`,
/// use it as the raw items. Each item is stringified, trimmed, and appended to
/// the result if non-empty and not already present (order-preserving dedupe).
pub fn normalize_skills(
    single_skill: Option<&str>,
    skills: Option<&[String]>,
) -> Option<Vec<String>> {
    let raw_items: Vec<String> = match skills {
        None => match single_skill {
            None => return None,
            Some(s) => vec![s.to_string()],
        },
        Some(items) => items.to_vec(),
    };

    let mut normalized: Vec<String> = Vec::new();
    for item in raw_items {
        let text = item.trim().to_string();
        if !text.is_empty() && !normalized.contains(&text) {
            normalized.push(text);
        }
    }
    Some(normalized)
}

// =============================================================================
// `_cron_api` — native equivalent of `cronjob(**kwargs)` returning parsed JSON
// =============================================================================

/// Keyword arguments accepted by [`cron_api`], mirroring the Python
/// `cronjob(**kwargs)` call sites. Only the fields the CLI uses are modelled.
#[derive(Debug, Default, Clone)]
pub struct CronApiArgs {
    pub action: String,
    pub job_id: Option<String>,
    pub schedule: Option<String>,
    pub prompt: Option<String>,
    pub name: Option<String>,
    pub deliver: Option<String>,
    pub repeat: Option<i64>,
    pub skill: Option<String>,
    /// `None` => omit; `Some(_)` => explicit (possibly empty to clear).
    pub skills: Option<Vec<String>>,
    pub script: Option<String>,
    pub workdir: Option<String>,
    /// Tri-state: `None` omits the field entirely (matching Python `or None`).
    pub no_agent: Option<bool>,
}

/// Native equivalent of `_cron_api(**kwargs)`: dispatch a cron action against
/// the native storage layer and return the parsed JSON response object whose
/// shape matches the Python `tools.cronjob_tools.cronjob` tool.
pub fn cron_api(args: &CronApiArgs) -> Value {
    match args.action.as_str() {
        "create" => api_create(args),
        "update" => api_update(args),
        "pause" => api_simple("pause", args),
        "resume" => api_simple("resume", args),
        "run" | "run_now" | "trigger" => api_simple("run", args),
        "remove" => api_remove(args),
        other => json!({
            "success": false,
            "error": format!(
                "Unknown cron action '{}'. Use create, list, update, pause, resume, remove, or run.",
                other
            ),
        }),
    }
}

fn api_error(msg: impl Into<String>) -> Value {
    json!({ "success": false, "error": msg.into() })
}

fn job_not_found(job_id: &str) -> Value {
    api_error(format!(
        "Job with ID '{}' not found. Use cronjob(action='list') to inspect jobs.",
        job_id
    ))
}

fn api_create(args: &CronApiArgs) -> Value {
    let schedule = match args.schedule.as_deref() {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => return api_error("schedule is required for action='create'."),
    };

    // _normalize_skills(skill, skills) is computed by the caller in cron_create;
    // here we mirror the merge of legacy `skill` + explicit `skills`.
    let normalized_skills = normalize_skills(args.skill.as_deref(), args.skills.as_deref());

    let params = cron_jobs::CreateJobParams {
        prompt: args.prompt.clone(),
        schedule,
        name: args.name.clone(),
        repeat: args.repeat,
        deliver: args.deliver.clone(),
        origin: None,
        skill: args.skill.clone(),
        skills: normalized_skills,
        model: None,
        provider: None,
        base_url: None,
        script: args.script.clone(),
        context_from: None,
        enabled_toolsets: None,
        workdir: args.workdir.clone(),
        no_agent: args.no_agent.unwrap_or(false),
    };

    match cron_jobs::create_job(params) {
        Ok(job) => {
            let skills = job
                .get("skills")
                .cloned()
                .unwrap_or_else(|| Value::Array(Vec::new()));
            json!({
                "success": true,
                "job_id": job.get("id").cloned().unwrap_or(Value::Null),
                "name": job.get("name").cloned().unwrap_or(Value::Null),
                "skill": job.get("skill").cloned().unwrap_or(Value::Null),
                "skills": skills,
                "schedule": job.get("schedule_display").cloned().unwrap_or(Value::Null),
                "deliver": job.get("deliver").cloned().unwrap_or(Value::String("local".into())),
                "next_run_at": job.get("next_run_at").cloned().unwrap_or(Value::Null),
                "job": format_job(&job),
            })
        }
        Err(e) => api_error(e),
    }
}

fn api_update(args: &CronApiArgs) -> Value {
    let job_id = match args.job_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return api_error("job_id is required for action='update'."),
    };

    // Build the partial updates dict, only including provided (non-None) fields,
    // matching how the Python tool reads its kwargs.
    let mut updates = Map::new();
    if let Some(s) = &args.schedule {
        updates.insert("schedule".into(), Value::String(s.clone()));
    }
    if let Some(p) = &args.prompt {
        updates.insert("prompt".into(), Value::String(p.clone()));
    }
    if let Some(n) = &args.name {
        updates.insert("name".into(), Value::String(n.clone()));
    }
    if let Some(d) = &args.deliver {
        updates.insert("deliver".into(), Value::String(d.clone()));
    }
    if let Some(r) = args.repeat {
        updates.insert("repeat".into(), json!(r));
    }
    if let Some(skills) = &args.skills {
        updates.insert(
            "skills".into(),
            Value::Array(skills.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(s) = &args.script {
        updates.insert("script".into(), Value::String(s.clone()));
    }
    if let Some(w) = &args.workdir {
        updates.insert("workdir".into(), Value::String(w.clone()));
    }
    if let Some(na) = args.no_agent {
        updates.insert("no_agent".into(), Value::Bool(na));
    }

    match cron_jobs::update_job(&job_id, &Value::Object(updates)) {
        Ok(Some(job)) => json!({ "success": true, "job": format_job(&job) }),
        Ok(None) => job_not_found(&job_id),
        Err(e) => api_error(e),
    }
}

fn api_simple(action: &str, args: &CronApiArgs) -> Value {
    let job_id = match args.job_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return api_error(format!("job_id is required for action='{action}'.")),
    };
    let result = match action {
        "pause" => cron_jobs::pause_job(&job_id, None),
        "resume" => cron_jobs::resume_job(&job_id),
        "run" => cron_jobs::trigger_job(&job_id),
        _ => unreachable!(),
    };
    match result {
        Ok(Some(job)) => json!({ "success": true, "job": format_job(&job) }),
        Ok(None) => job_not_found(&job_id),
        Err(e) => api_error(e),
    }
}

fn api_remove(args: &CronApiArgs) -> Value {
    let job_id = match args.job_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return api_error("job_id is required for action='remove'."),
    };
    // Fetch first so we can echo name/schedule in removed_job.
    let job = match cron_jobs::get_job(&job_id) {
        Ok(Some(j)) => j,
        Ok(None) => return job_not_found(&job_id),
        Err(e) => return api_error(e),
    };
    match cron_jobs::remove_job(&job_id) {
        Ok(true) => json!({
            "success": true,
            "removed_job": {
                "id": job_id,
                "name": job.get("name").cloned().unwrap_or(Value::Null),
                "schedule": job.get("schedule_display").cloned().unwrap_or(Value::Null),
            },
        }),
        Ok(false) => job_not_found(&job_id),
        Err(e) => api_error(e),
    }
}

/// Build the `job` payload used in tool responses, matching the Python
/// `format_job` shape the CLI reads (`job_id`, `name`, `schedule` display
/// string, `skills`, and the optional `script`/`no_agent`/`workdir` fields).
fn format_job(job: &Value) -> Value {
    let mut out = Map::new();
    out.insert(
        "job_id".into(),
        job.get("id").cloned().unwrap_or(Value::Null),
    );
    out.insert("name".into(), job.get("name").cloned().unwrap_or(Value::Null));
    out.insert(
        "skill".into(),
        job.get("skill").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "skills".into(),
        job.get("skills")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    );
    out.insert(
        "schedule".into(),
        job.get("schedule_display").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "next_run_at".into(),
        job.get("next_run_at").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "deliver".into(),
        job.get("deliver")
            .cloned()
            .unwrap_or(Value::String("local".into())),
    );

    // script: only present when set.
    if let Some(script) = job.get("script").and_then(|v| v.as_str()) {
        if !script.is_empty() {
            out.insert("script".into(), Value::String(script.to_string()));
        }
    }
    // no_agent: only present when true.
    if job.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false) {
        out.insert("no_agent".into(), Value::Bool(true));
    }
    // workdir: only present when set.
    if let Some(wd) = job.get("workdir").and_then(|v| v.as_str()) {
        if !wd.is_empty() {
            out.insert("workdir".into(), Value::String(wd.to_string()));
        }
    }

    Value::Object(out)
}

// =============================================================================
// Output sink
// =============================================================================

/// Trait abstracting line output so the rendering can be unit-tested without
/// touching stdout. The default [`StdoutSink`] writes via `println!`.
pub trait OutputSink {
    fn line(&mut self, s: &str);
}

/// Writes to real stdout.
pub struct StdoutSink;

impl OutputSink for StdoutSink {
    fn line(&mut self, s: &str) {
        println!("{s}");
    }
}

// =============================================================================
// cron list
// =============================================================================

fn job_str(job: &Value, key: &str, default: &str) -> String {
    job.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| default.to_string())
}

/// Port of `cron_list(show_all)`.
pub fn cron_list(show_all: bool, out: &mut dyn OutputSink) {
    let jobs = match cron_jobs::list_jobs(show_all) {
        Ok(j) => j,
        Err(_) => Vec::new(),
    };

    if jobs.is_empty() {
        out.line(&color("No scheduled jobs.", &[Colors::DIM]));
        out.line(&color(
            "Create one with 'hermes cron create ...' or the /cron command in chat.",
            &[Colors::DIM],
        ));
        return;
    }

    out.line("");
    out.line(&color(
        "┌─────────────────────────────────────────────────────────────────────────┐",
        &[Colors::CYAN],
    ));
    out.line(&color(
        "│                         Scheduled Jobs                                  │",
        &[Colors::CYAN],
    ));
    out.line(&color(
        "└─────────────────────────────────────────────────────────────────────────┘",
        &[Colors::CYAN],
    ));
    out.line("");

    for job in &jobs {
        let job_id = job_str(job, "id", "?");
        let name = job_str(job, "name", "(unnamed)");

        // schedule = schedule_display, fallback to schedule.value, fallback "?"
        let schedule = job
            .get("schedule_display")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                job.get("schedule")
                    .and_then(|s| s.get("value"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| "?".to_string())
            });

        // state = job.state, default "scheduled" if enabled else "paused"
        let enabled = job.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
        let state = job
            .get("state")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if enabled { "scheduled".to_string() } else { "paused".to_string() }
            });
        let next_run = job_str(job, "next_run_at", "?");

        // repeat
        let repeat_times = job
            .get("repeat")
            .and_then(|r| r.get("times"))
            .and_then(|v| v.as_i64());
        let repeat_completed = job
            .get("repeat")
            .and_then(|r| r.get("completed"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let repeat_str = match repeat_times {
            Some(t) if t != 0 => format!("{repeat_completed}/{t}"),
            _ => "∞".to_string(),
        };

        // deliver: str or list
        let deliver_str = match job.get("deliver") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => arr
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
                .join(", "),
            _ => "local".to_string(),
        };

        // skills = job.skills or ([job.skill] if job.skill else [])
        let skills: Vec<String> = match job.get("skills") {
            Some(Value::Array(arr)) if !arr.is_empty() => arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => match job.get("skill").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => vec![s.to_string()],
                _ => Vec::new(),
            },
        };

        let status = match state.as_str() {
            "paused" => color("[paused]", &[Colors::YELLOW]),
            "completed" => color("[completed]", &[Colors::BLUE]),
            _ if enabled => color("[active]", &[Colors::GREEN]),
            _ => color("[disabled]", &[Colors::RED]),
        };

        out.line(&format!("  {} {}", color(&job_id, &[Colors::YELLOW]), status));
        out.line(&format!("    Name:      {name}"));
        out.line(&format!("    Schedule:  {schedule}"));
        out.line(&format!("    Repeat:    {repeat_str}"));
        out.line(&format!("    Next run:  {next_run}"));
        out.line(&format!("    Deliver:   {deliver_str}"));
        if !skills.is_empty() {
            out.line(&format!("    Skills:    {}", skills.join(", ")));
        }
        if let Some(script) = job.get("script").and_then(|v| v.as_str()) {
            if !script.is_empty() {
                out.line(&format!("    Script:    {script}"));
            }
        }
        if job.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false) {
            out.line(&format!(
                "    Mode:      {} (script stdout delivered directly)",
                color("no-agent", &[Colors::DIM])
            ));
        }
        if let Some(workdir) = job.get("workdir").and_then(|v| v.as_str()) {
            if !workdir.is_empty() {
                out.line(&format!("    Workdir:   {workdir}"));
            }
        }

        // Execution history.
        if let Some(last_status) = job.get("last_status").and_then(|v| v.as_str()) {
            if !last_status.is_empty() {
                let last_run = job_str(job, "last_run_at", "?");
                let status_display = if last_status == "ok" {
                    color("ok", &[Colors::GREEN])
                } else {
                    let last_error = job_str(job, "last_error", "?");
                    color(&format!("{last_status}: {last_error}"), &[Colors::RED])
                };
                out.line(&format!("    Last run:  {last_run}  {status_display}"));
            }
        }

        if let Some(delivery_err) = job.get("last_delivery_error").and_then(|v| v.as_str()) {
            if !delivery_err.is_empty() {
                out.line(&format!(
                    "    {} {delivery_err}",
                    color("⚠ Delivery failed:", &[Colors::YELLOW])
                ));
            }
        }

        out.line("");
    }

    if find_gateway_pids().is_empty() {
        out.line(&color(
            "  ⚠  Gateway is not running — jobs won't fire automatically.",
            &[Colors::YELLOW],
        ));
        out.line(&color(
            "     Start it with: hermes gateway install",
            &[Colors::DIM],
        ));
        out.line(&color(
            "                    sudo hermes gateway install --system  # Linux servers",
            &[Colors::DIM],
        ));
        out.line("");
    }
}

// =============================================================================
// cron tick
// =============================================================================

/// Port of `cron_tick()`. The scheduler `tick(verbose=True)` requires agent
/// context this module does not own, so it is supplied as a callback.
pub fn cron_tick(tick_fn: impl FnOnce(bool)) {
    tick_fn(true);
}

// =============================================================================
// cron status
// =============================================================================

/// Port of `cron_status()`.
pub fn cron_status(out: &mut dyn OutputSink) {
    out.line("");

    let pids = find_gateway_pids();
    if !pids.is_empty() {
        out.line(&color(
            "✓ Gateway is running — cron jobs will fire automatically",
            &[Colors::GREEN],
        ));
        let joined = pids
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.line(&format!("  PID: {joined}"));
    } else {
        out.line(&color(
            "✗ Gateway is not running — cron jobs will NOT fire",
            &[Colors::RED],
        ));
        out.line("");
        out.line("  To enable automatic execution:");
        out.line("    hermes gateway install    # Install as a user service");
        out.line("    sudo hermes gateway install --system  # Linux servers: boot-time system service");
        out.line("    hermes gateway            # Or run in foreground");
    }

    out.line("");

    let jobs = cron_jobs::list_jobs(false).unwrap_or_default();
    if !jobs.is_empty() {
        let next_runs: Vec<String> = jobs
            .iter()
            .filter_map(|j| {
                j.get("next_run_at")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            })
            .collect();
        out.line(&format!("  {} active job(s)", jobs.len()));
        if !next_runs.is_empty() {
            // min() over ISO-8601 strings == chronological min (lexicographic).
            let min = next_runs.iter().min().unwrap();
            out.line(&format!("  Next run: {min}"));
        }
    } else {
        out.line("  No active jobs");
    }

    out.line("");
}

// =============================================================================
// cron create
// =============================================================================

/// Port of `cron_create(args)`. Returns the process exit code.
pub fn cron_create(args: &CronArgs, out: &mut dyn OutputSink) -> i32 {
    // no_agent=False -> None (Python `getattr(args, "no_agent", False) or None`)
    let no_agent = match args.no_agent {
        Some(true) => Some(true),
        _ => None,
    };

    let api_args = CronApiArgs {
        action: "create".into(),
        job_id: None,
        schedule: args.schedule.clone(),
        prompt: args.prompt.clone(),
        name: args.name.clone(),
        deliver: args.deliver.clone(),
        repeat: args.repeat,
        skill: args.skill.clone(),
        skills: normalize_skills(args.skill.as_deref(), args.skills.as_deref()),
        script: args.script.clone(),
        workdir: args.workdir.clone(),
        no_agent,
    };

    let result = cron_api(&api_args);

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        out.line(&color(&format!("Failed to create job: {err}"), &[Colors::RED]));
        return 1;
    }

    let job_id = result.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
    out.line(&color(&format!("Created job: {job_id}"), &[Colors::GREEN]));
    out.line(&format!(
        "  Name: {}",
        result.get("name").and_then(|v| v.as_str()).unwrap_or("")
    ));
    out.line(&format!(
        "  Schedule: {}",
        result.get("schedule").and_then(|v| v.as_str()).unwrap_or("")
    ));
    if let Some(Value::Array(skills)) = result.get("skills") {
        if !skills.is_empty() {
            let joined = skills
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            out.line(&format!("  Skills: {joined}"));
        }
    }
    let job_data = result.get("job").cloned().unwrap_or(Value::Null);
    if let Some(script) = job_data.get("script").and_then(|v| v.as_str()) {
        if !script.is_empty() {
            out.line(&format!("  Script: {script}"));
        }
    }
    if job_data.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false) {
        out.line("  Mode: no-agent (script stdout delivered directly)");
    }
    if let Some(workdir) = job_data.get("workdir").and_then(|v| v.as_str()) {
        if !workdir.is_empty() {
            out.line(&format!("  Workdir: {workdir}"));
        }
    }
    out.line(&format!(
        "  Next run: {}",
        result.get("next_run_at").and_then(|v| v.as_str()).unwrap_or("")
    ));
    0
}

// =============================================================================
// cron edit
// =============================================================================

/// Port of `cron_edit(args)`. Returns the process exit code.
pub fn cron_edit(args: &CronArgs, out: &mut dyn OutputSink) -> i32 {
    let job_id = args.job_id.clone().unwrap_or_default();

    let job = match cron_jobs::get_job(&job_id) {
        Ok(Some(j)) => j,
        Ok(None) => {
            out.line(&color(&format!("Job not found: {job_id}"), &[Colors::RED]));
            return 1;
        }
        Err(e) => {
            out.line(&color(&format!("Job not found: {job_id} ({e})"), &[Colors::RED]));
            return 1;
        }
    };

    // existing_skills = job.skills or ([] if not job.skill else [job.skill])
    let existing_skills: Vec<String> = match job.get("skills") {
        Some(Value::Array(arr)) if !arr.is_empty() => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => match job.get("skill").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => vec![s.to_string()],
            _ => Vec::new(),
        },
    };

    let replacement_skills = normalize_skills(args.skill.as_deref(), args.skills.as_deref());
    let add_skills = normalize_skills(None, args.add_skills.as_deref()).unwrap_or_default();
    let remove_skills: std::collections::BTreeSet<String> =
        normalize_skills(None, args.remove_skills.as_deref())
            .unwrap_or_default()
            .into_iter()
            .collect();

    let final_skills: Option<Vec<String>> = if args.clear_skills {
        Some(Vec::new())
    } else if let Some(repl) = replacement_skills {
        Some(repl)
    } else if !add_skills.is_empty() || !remove_skills.is_empty() {
        let mut fs: Vec<String> = existing_skills
            .iter()
            .filter(|s| !remove_skills.contains(*s))
            .cloned()
            .collect();
        for s in &add_skills {
            if !fs.contains(s) {
                fs.push(s.clone());
            }
        }
        Some(fs)
    } else {
        None
    };

    let api_args = CronApiArgs {
        action: "update".into(),
        job_id: Some(job_id),
        schedule: args.schedule.clone(),
        prompt: args.prompt.clone(),
        name: args.name.clone(),
        deliver: args.deliver.clone(),
        repeat: args.repeat,
        skill: None,
        skills: final_skills,
        script: args.script.clone(),
        workdir: args.workdir.clone(),
        no_agent: args.no_agent,
    };

    let result = cron_api(&api_args);

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        out.line(&color(&format!("Failed to update job: {err}"), &[Colors::RED]));
        return 1;
    }

    let updated = result.get("job").cloned().unwrap_or(Value::Null);
    out.line(&color(
        &format!(
            "Updated job: {}",
            updated.get("job_id").and_then(|v| v.as_str()).unwrap_or("")
        ),
        &[Colors::GREEN],
    ));
    out.line(&format!(
        "  Name: {}",
        updated.get("name").and_then(|v| v.as_str()).unwrap_or("")
    ));
    out.line(&format!(
        "  Schedule: {}",
        updated.get("schedule").and_then(|v| v.as_str()).unwrap_or("")
    ));
    match updated.get("skills") {
        Some(Value::Array(skills)) if !skills.is_empty() => {
            let joined = skills
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            out.line(&format!("  Skills: {joined}"));
        }
        _ => out.line("  Skills: none"),
    }
    if let Some(script) = updated.get("script").and_then(|v| v.as_str()) {
        if !script.is_empty() {
            out.line(&format!("  Script: {script}"));
        }
    }
    if updated.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false) {
        out.line("  Mode: no-agent (script stdout delivered directly)");
    }
    if let Some(workdir) = updated.get("workdir").and_then(|v| v.as_str()) {
        if !workdir.is_empty() {
            out.line(&format!("  Workdir: {workdir}"));
        }
    }
    0
}

// =============================================================================
// job action (pause/resume/run/remove)
// =============================================================================

/// Port of `_job_action(action, job_id, success_verb)`. Returns the exit code.
pub fn job_action(action: &str, job_id: &str, success_verb: &str, out: &mut dyn OutputSink) -> i32 {
    let api_args = CronApiArgs {
        action: action.to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };
    let result = cron_api(&api_args);

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        out.line(&color(
            &format!("Failed to {action} job: {err}"),
            &[Colors::RED],
        ));
        return 1;
    }

    // job = result.job or result.removed_job or {}
    let job = result
        .get("job")
        .cloned()
        .filter(|v| !v.is_null())
        .or_else(|| result.get("removed_job").cloned())
        .unwrap_or(Value::Null);
    let job_name = job
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(job_id);
    out.line(&color(
        &format!("{success_verb} job: {job_name} ({job_id})"),
        &[Colors::GREEN],
    ));

    if matches!(action, "resume" | "run") {
        if let Some(nr) = result
            .get("job")
            .and_then(|j| j.get("next_run_at"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            out.line(&format!("  Next run: {nr}"));
        }
    }
    if action == "run" {
        out.line("  It will run on the next scheduler tick.");
    }
    0
}

// =============================================================================
// dispatch
// =============================================================================

/// Outcome of [`cron_command`]: an exit code, with a distinguished
/// `Exit(code)` variant for the Python `sys.exit(1)` on an unknown subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronOutcome {
    /// Return this exit code from the subcommand handler (Python `return N`).
    Code(i32),
    /// The unknown-subcommand path which Python handles via `sys.exit(1)`.
    Exit(i32),
}

/// Port of `cron_command(args)`.
///
/// `tick_fn` is invoked for the `tick` subcommand (Python `tick(verbose=True)`).
pub fn cron_command(
    args: &CronArgs,
    out: &mut dyn OutputSink,
    tick_fn: impl FnOnce(bool),
) -> CronOutcome {
    let subcmd = args.cron_command.as_deref();

    match subcmd {
        None | Some("list") => {
            cron_list(args.all, out);
            CronOutcome::Code(0)
        }
        Some("status") => {
            cron_status(out);
            CronOutcome::Code(0)
        }
        Some("tick") => {
            cron_tick(tick_fn);
            CronOutcome::Code(0)
        }
        Some("create") | Some("add") => CronOutcome::Code(cron_create(args, out)),
        Some("edit") => CronOutcome::Code(cron_edit(args, out)),
        Some("pause") => CronOutcome::Code(job_action(
            "pause",
            args.job_id.as_deref().unwrap_or(""),
            "Paused",
            out,
        )),
        Some("resume") => CronOutcome::Code(job_action(
            "resume",
            args.job_id.as_deref().unwrap_or(""),
            "Resumed",
            out,
        )),
        Some("run") => CronOutcome::Code(job_action(
            "run",
            args.job_id.as_deref().unwrap_or(""),
            "Triggered",
            out,
        )),
        Some("remove") | Some("rm") | Some("delete") => CronOutcome::Code(job_action(
            "remove",
            args.job_id.as_deref().unwrap_or(""),
            "Removed",
            out,
        )),
        Some(other) => {
            out.line(&format!("Unknown cron command: {other}"));
            out.line("Usage: hermes cron [list|create|edit|pause|resume|run|remove|status|tick]");
            CronOutcome::Exit(1)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Captures output lines for assertions.
    struct VecSink(Vec<String>);
    impl OutputSink for VecSink {
        fn line(&mut self, s: &str) {
            self.0.push(s.to_string());
        }
    }

    fn with_temp_home<F: FnOnce()>(f: F) {
        let _g = TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!(
            "hermes_cli_cron_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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
    fn normalize_skills_none_when_both_none() {
        assert_eq!(normalize_skills(None, None), None);
    }

    #[test]
    fn normalize_skills_single_legacy() {
        assert_eq!(
            normalize_skills(Some("alpha"), None),
            Some(vec!["alpha".to_string()])
        );
    }

    #[test]
    fn normalize_skills_dedupes_and_trims() {
        let input = vec![
            "a".to_string(),
            " a ".to_string(),
            "".to_string(),
            "b".to_string(),
        ];
        assert_eq!(
            normalize_skills(None, Some(&input)),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn normalize_skills_empty_list_stays_empty() {
        // skills=[] is "explicit empty", distinct from None.
        assert_eq!(normalize_skills(None, Some(&[])), Some(Vec::new()));
    }

    #[test]
    fn cron_api_unknown_action() {
        let r = cron_api(&CronApiArgs {
            action: "bogus".into(),
            ..Default::default()
        });
        assert_eq!(r["success"], false);
        assert!(r["error"].as_str().unwrap().contains("Unknown cron action"));
    }

    #[test]
    fn create_then_list_then_remove() {
        with_temp_home(|| {
            let r = cron_api(&CronApiArgs {
                action: "create".into(),
                schedule: Some("every 10m".into()),
                prompt: Some("hello".into()),
                ..Default::default()
            });
            assert_eq!(r["success"], true);
            let job_id = r["job_id"].as_str().unwrap().to_string();
            assert_eq!(r["job"]["job_id"], r["job_id"]);

            let mut sink = VecSink(Vec::new());
            cron_list(false, &mut sink);
            let joined = sink.0.join("\n");
            assert!(joined.contains(&job_id));
            assert!(joined.contains("Scheduled Jobs"));

            // remove via job_action
            let mut s2 = VecSink(Vec::new());
            let code = job_action("remove", &job_id, "Removed", &mut s2);
            assert_eq!(code, 0);
            assert!(s2.0.join("\n").contains("Removed job"));
        });
    }

    #[test]
    fn list_empty_shows_hint() {
        with_temp_home(|| {
            let mut sink = VecSink(Vec::new());
            cron_list(false, &mut sink);
            assert!(sink.0.iter().any(|l| l.contains("No scheduled jobs.")));
        });
    }

    #[test]
    fn create_missing_schedule_fails() {
        with_temp_home(|| {
            let args = CronArgs {
                cron_command: Some("create".into()),
                prompt: Some("x".into()),
                ..Default::default()
            };
            let mut sink = VecSink(Vec::new());
            let code = cron_create(&args, &mut sink);
            assert_eq!(code, 1);
            assert!(sink.0.iter().any(|l| l.contains("Failed to create job")));
        });
    }

    #[test]
    fn edit_not_found() {
        with_temp_home(|| {
            let args = CronArgs {
                cron_command: Some("edit".into()),
                job_id: Some("nope".into()),
                ..Default::default()
            };
            let mut sink = VecSink(Vec::new());
            let code = cron_edit(&args, &mut sink);
            assert_eq!(code, 1);
            assert!(sink.0.iter().any(|l| l.contains("Job not found")));
        });
    }

    #[test]
    fn edit_clear_skills_overrides() {
        with_temp_home(|| {
            let created = cron_api(&CronApiArgs {
                action: "create".into(),
                schedule: Some("every 10m".into()),
                prompt: Some("p".into()),
                skills: Some(vec!["a".into(), "b".into()]),
                ..Default::default()
            });
            let id = created["job_id"].as_str().unwrap().to_string();

            let args = CronArgs {
                cron_command: Some("edit".into()),
                job_id: Some(id.clone()),
                clear_skills: true,
                ..Default::default()
            };
            let mut sink = VecSink(Vec::new());
            let code = cron_edit(&args, &mut sink);
            assert_eq!(code, 0);
            assert!(sink.0.iter().any(|l| l.contains("Skills: none")));
        });
    }

    #[test]
    fn edit_add_remove_skills() {
        with_temp_home(|| {
            let created = cron_api(&CronApiArgs {
                action: "create".into(),
                schedule: Some("every 10m".into()),
                prompt: Some("p".into()),
                skills: Some(vec!["keep".into(), "drop".into()]),
                ..Default::default()
            });
            let id = created["job_id"].as_str().unwrap().to_string();

            let args = CronArgs {
                cron_command: Some("edit".into()),
                job_id: Some(id.clone()),
                add_skills: Some(vec!["new".into()]),
                remove_skills: Some(vec!["drop".into()]),
                ..Default::default()
            };
            let mut sink = VecSink(Vec::new());
            assert_eq!(cron_edit(&args, &mut sink), 0);

            let job = cron_jobs::get_job(&id).unwrap().unwrap();
            let skills: Vec<String> = job["skills"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert_eq!(skills, vec!["keep".to_string(), "new".to_string()]);
        });
    }

    #[test]
    fn pause_resume_run_actions() {
        with_temp_home(|| {
            let created = cron_api(&CronApiArgs {
                action: "create".into(),
                schedule: Some("every 10m".into()),
                prompt: Some("p".into()),
                ..Default::default()
            });
            let id = created["job_id"].as_str().unwrap().to_string();

            let mut s = VecSink(Vec::new());
            assert_eq!(job_action("pause", &id, "Paused", &mut s), 0);
            assert!(s.0.join("\n").contains("Paused job"));

            let mut s = VecSink(Vec::new());
            assert_eq!(job_action("resume", &id, "Resumed", &mut s), 0);
            assert!(s.0.join("\n").contains("Resumed job"));

            let mut s = VecSink(Vec::new());
            assert_eq!(job_action("run", &id, "Triggered", &mut s), 0);
            let joined = s.0.join("\n");
            assert!(joined.contains("Triggered job"));
            assert!(joined.contains("next scheduler tick"));
        });
    }

    #[test]
    fn job_action_missing_returns_1() {
        with_temp_home(|| {
            let mut s = VecSink(Vec::new());
            assert_eq!(job_action("pause", "missing", "Paused", &mut s), 1);
            assert!(s.0.join("\n").contains("Failed to pause job"));
        });
    }

    #[test]
    fn command_unknown_exits() {
        let args = CronArgs {
            cron_command: Some("frobnicate".into()),
            ..Default::default()
        };
        let mut s = VecSink(Vec::new());
        let outcome = cron_command(&args, &mut s, |_| {});
        assert_eq!(outcome, CronOutcome::Exit(1));
        assert!(s.0.iter().any(|l| l.contains("Unknown cron command")));
    }

    #[test]
    fn command_tick_invokes_callback() {
        let args = CronArgs {
            cron_command: Some("tick".into()),
            ..Default::default()
        };
        let mut s = VecSink(Vec::new());
        let mut called = false;
        let outcome = cron_command(&args, &mut s, |verbose| {
            assert!(verbose);
            called = true;
        });
        assert_eq!(outcome, CronOutcome::Code(0));
        assert!(called);
    }

    #[test]
    fn command_default_is_list() {
        with_temp_home(|| {
            let args = CronArgs::default();
            let mut s = VecSink(Vec::new());
            let outcome = cron_command(&args, &mut s, |_| {});
            assert_eq!(outcome, CronOutcome::Code(0));
        });
    }
}
