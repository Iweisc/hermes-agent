use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Datelike, Local, TimeZone, Timelike};
use clap::Args;
use hermes_core::HermesContext;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::Value;

#[derive(Args, Debug, Clone)]
pub struct InsightsArgs {
    #[arg(long, default_value_t = 30)]
    pub days: i64,
    #[arg(long)]
    pub source: Option<String>,
}

#[derive(Debug, Clone)]
struct SessionRow {
    id: String,
    source: String,
    model: String,
    started_at: f64,
    ended_at: Option<f64>,
    message_count: i64,
    tool_call_count: i64,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_tokens: i64,
    cache_write_tokens: i64,
    estimated_cost_usd: f64,
    actual_cost_usd: f64,
}

#[derive(Debug, Clone, Default)]
struct MessageStats {
    total_messages: i64,
    user_messages: i64,
    assistant_messages: i64,
    tool_messages: i64,
}

#[derive(Debug, Clone)]
struct ToolUsage {
    tool: String,
    count: i64,
}

#[derive(Debug, Clone)]
struct SkillUsage {
    skill: String,
    view_count: i64,
    manage_count: i64,
    last_used_at: Option<f64>,
}

#[derive(Debug, Clone)]
struct Overview {
    total_sessions: usize,
    total_messages: i64,
    total_tool_calls: i64,
    total_input_tokens: i64,
    total_output_tokens: i64,
    total_cache_read_tokens: i64,
    total_cache_write_tokens: i64,
    total_tokens: i64,
    recorded_estimated_cost: f64,
    recorded_actual_cost: f64,
    total_hours: f64,
    avg_session_duration_seconds: f64,
    avg_messages_per_session: f64,
    avg_tokens_per_session: f64,
    user_messages: i64,
    assistant_messages: i64,
    tool_messages: i64,
    date_range_start: Option<f64>,
    date_range_end: Option<f64>,
    sessions_with_recorded_cost: usize,
}

#[derive(Debug, Clone)]
struct ModelBreakdown {
    model: String,
    sessions: usize,
    total_tokens: i64,
}

#[derive(Debug, Clone)]
struct PlatformBreakdown {
    platform: String,
    sessions: usize,
    messages: i64,
    total_tokens: i64,
}

#[derive(Debug, Clone)]
struct ActivityBreakdown {
    by_day: Vec<(String, i64)>,
    by_hour: Vec<(u32, i64)>,
    busiest_day: Option<(String, i64)>,
    busiest_hour: Option<(u32, i64)>,
    active_days: usize,
    max_streak: usize,
}

#[derive(Debug, Clone)]
struct TopSession {
    label: &'static str,
    session_id: String,
    value: String,
    date: String,
}

pub fn print_insights(context: &HermesContext, args: InsightsArgs) -> Result<(), Box<dyn Error>> {
    let days = validate_days(args.days)?;
    let source = normalize_source(args.source);
    let path = context.state_db_path();
    if !path.is_file() {
        println!("{}", empty_message(days, source.as_deref()));
        return Ok(());
    }

    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let cutoff = now_ts() - (days as f64 * 86_400.0);
    let sessions = fetch_sessions(&conn, cutoff, source.as_deref())?;
    if sessions.is_empty() {
        println!("{}", empty_message(days, source.as_deref()));
        return Ok(());
    }

    let message_stats = fetch_message_stats(&conn, cutoff, source.as_deref())?;
    let tool_usage = fetch_tool_usage(&conn, cutoff, source.as_deref())?;
    let skill_usage = fetch_skill_usage(&conn, cutoff, source.as_deref())?;

    let overview = compute_overview(&sessions, &message_stats);
    let model_breakdown = compute_model_breakdown(&sessions);
    let platform_breakdown = compute_platform_breakdown(&sessions);
    let activity = compute_activity(&sessions);
    let top_sessions = compute_top_sessions(&sessions);

    print_report(
        days,
        source.as_deref(),
        &overview,
        &model_breakdown,
        &platform_breakdown,
        &tool_usage,
        &skill_usage,
        &activity,
        &top_sessions,
    );
    Ok(())
}

fn validate_days(days: i64) -> Result<i64, Box<dyn Error>> {
    if days <= 0 {
        return Err("days must be greater than 0".into());
    }
    Ok(days)
}

fn normalize_source(source: Option<String>) -> Option<String> {
    source.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn empty_message(days: i64, source: Option<&str>) -> String {
    match source {
        Some(source) => format!("No sessions found in the last {days} days (source: {source})."),
        None => format!("No sessions found in the last {days} days."),
    }
}

fn fetch_sessions(
    conn: &Connection,
    cutoff: f64,
    source: Option<&str>,
) -> Result<Vec<SessionRow>, Box<dyn Error>> {
    let sql = if source.is_some() {
        "SELECT id, source, COALESCE(model, ''), started_at, ended_at, message_count, tool_call_count,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                COALESCE(estimated_cost_usd, 0), COALESCE(actual_cost_usd, 0)
         FROM sessions
         WHERE started_at >= ? AND source = ?
         ORDER BY started_at DESC"
    } else {
        "SELECT id, source, COALESCE(model, ''), started_at, ended_at, message_count, tool_call_count,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                COALESCE(estimated_cost_usd, 0), COALESCE(actual_cost_usd, 0)
         FROM sessions
         WHERE started_at >= ?
         ORDER BY started_at DESC"
    };
    let mut statement = conn.prepare(sql)?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SessionRow> {
        Ok(SessionRow {
            id: row.get(0)?,
            source: row.get(1)?,
            model: row.get(2)?,
            started_at: row.get(3)?,
            ended_at: row.get(4)?,
            message_count: row.get(5)?,
            tool_call_count: row.get(6)?,
            input_tokens: row.get(7)?,
            output_tokens: row.get(8)?,
            cache_read_tokens: row.get(9)?,
            cache_write_tokens: row.get(10)?,
            estimated_cost_usd: row.get(11)?,
            actual_cost_usd: row.get(12)?,
        })
    };
    let rows = if let Some(source) = source {
        statement.query_map(params![cutoff, source], mapper)?
    } else {
        statement.query_map(params![cutoff], mapper)?
    };
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

fn fetch_message_stats(
    conn: &Connection,
    cutoff: f64,
    source: Option<&str>,
) -> Result<MessageStats, Box<dyn Error>> {
    let sql = if source.is_some() {
        "SELECT
             COUNT(*) as total_messages,
             SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
             SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
             SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ? AND s.source = ?"
    } else {
        "SELECT
             COUNT(*) as total_messages,
             SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
             SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
             SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ?"
    };
    let mut statement = conn.prepare(sql)?;
    let stats = if let Some(source) = source {
        statement.query_row(params![cutoff, source], |row| {
            Ok(MessageStats {
                total_messages: row.get::<_, Option<i64>>(0)?.unwrap_or(0),
                user_messages: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                assistant_messages: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                tool_messages: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
            })
        })?
    } else {
        statement.query_row(params![cutoff], |row| {
            Ok(MessageStats {
                total_messages: row.get::<_, Option<i64>>(0)?.unwrap_or(0),
                user_messages: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                assistant_messages: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                tool_messages: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
            })
        })?
    };
    Ok(stats)
}

fn fetch_tool_usage(
    conn: &Connection,
    cutoff: f64,
    source: Option<&str>,
) -> Result<Vec<ToolUsage>, Box<dyn Error>> {
    let sql = if source.is_some() {
        "SELECT m.tool_name, COUNT(*) as count
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ? AND s.source = ?
           AND m.role = 'tool' AND m.tool_name IS NOT NULL
         GROUP BY m.tool_name
         ORDER BY count DESC"
    } else {
        "SELECT m.tool_name, COUNT(*) as count
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ?
           AND m.role = 'tool' AND m.tool_name IS NOT NULL
         GROUP BY m.tool_name
         ORDER BY count DESC"
    };
    let mut direct_counts = BTreeMap::new();
    let rows: Vec<(String, i64)> = if let Some(source) = source {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff, source], |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    row.get::<_, i64>(1)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff], |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    row.get::<_, i64>(1)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (tool, count) in rows {
        if !tool.is_empty() {
            direct_counts.insert(tool, count);
        }
    }

    let sql = if source.is_some() {
        "SELECT m.tool_calls
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ? AND s.source = ?
           AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
    } else {
        "SELECT m.tool_calls
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ?
           AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
    };
    let mut parsed_counts: HashMap<String, i64> = HashMap::new();
    let rows: Vec<String> = if let Some(source) = source {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff, source], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?
    };
    for row in rows {
        for name in extract_tool_call_names(&row) {
            *parsed_counts.entry(name).or_default() += 1;
        }
    }

    let final_counts: HashMap<String, i64> =
        if direct_counts.is_empty() && !parsed_counts.is_empty() {
            parsed_counts
        } else if !direct_counts.is_empty() && !parsed_counts.is_empty() {
            let mut merged = HashMap::new();
            for key in direct_counts.keys().chain(parsed_counts.keys()) {
                let direct = direct_counts.get(key).copied().unwrap_or(0);
                let parsed = parsed_counts.get(key).copied().unwrap_or(0);
                merged.insert(key.clone(), direct.max(parsed));
            }
            merged
        } else {
            direct_counts.into_iter().collect()
        };

    let mut tools = final_counts
        .into_iter()
        .map(|(tool, count)| ToolUsage { tool, count })
        .collect::<Vec<_>>();
    tools.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.tool.cmp(&right.tool))
    });
    Ok(tools)
}

fn fetch_skill_usage(
    conn: &Connection,
    cutoff: f64,
    source: Option<&str>,
) -> Result<Vec<SkillUsage>, Box<dyn Error>> {
    let sql = if source.is_some() {
        "SELECT m.tool_calls, m.timestamp
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ? AND s.source = ?
           AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
    } else {
        "SELECT m.tool_calls, m.timestamp
         FROM messages m
         JOIN sessions s ON s.id = m.session_id
         WHERE s.started_at >= ?
           AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
    };
    let mut usage: BTreeMap<String, SkillUsage> = BTreeMap::new();
    let rows: Vec<(String, f64)> = if let Some(source) = source {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff, source], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map(params![cutoff], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (tool_calls, timestamp) in rows {
        for (tool_name, skill_name) in extract_skill_calls(&tool_calls) {
            let entry = usage
                .entry(skill_name.clone())
                .or_insert_with(|| SkillUsage {
                    skill: skill_name.clone(),
                    view_count: 0,
                    manage_count: 0,
                    last_used_at: None,
                });
            match tool_name.as_str() {
                "skill_view" => entry.view_count += 1,
                "skill_manage" => entry.manage_count += 1,
                _ => {}
            }
            if entry
                .last_used_at
                .map(|existing| timestamp > existing)
                .unwrap_or(true)
            {
                entry.last_used_at = Some(timestamp);
            }
        }
    }

    let mut skills = usage.into_values().collect::<Vec<_>>();
    skills.sort_by(|left, right| {
        let left_total = left.view_count + left.manage_count;
        let right_total = right.view_count + right.manage_count;
        right_total
            .cmp(&left_total)
            .then_with(|| right.view_count.cmp(&left.view_count))
            .then_with(|| right.manage_count.cmp(&left.manage_count))
            .then_with(|| {
                right
                    .last_used_at
                    .partial_cmp(&left.last_used_at)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.skill.cmp(&right.skill))
    });
    Ok(skills)
}

fn extract_tool_call_names(raw: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return Vec::new();
    };
    let Some(calls) = value.as_array() else {
        return Vec::new();
    };
    calls.iter().filter_map(tool_call_name).collect()
}

fn extract_skill_calls(raw: &str) -> Vec<(String, String)> {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return Vec::new();
    };
    let Some(calls) = value.as_array() else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for call in calls {
        let Some(tool_name) = tool_call_name(call) else {
            continue;
        };
        if tool_name != "skill_view" && tool_name != "skill_manage" {
            continue;
        }
        let Some(arguments) = tool_call_arguments(call) else {
            continue;
        };
        let Some(skill_name) = arguments.get("name").and_then(Value::as_str) else {
            continue;
        };
        let skill_name = skill_name.trim();
        if skill_name.is_empty() {
            continue;
        }
        result.push((tool_name, skill_name.to_string()));
    }
    result
}

fn tool_call_name(call: &Value) -> Option<String> {
    call.get("function")
        .and_then(|value| value.get("name"))
        .and_then(Value::as_str)
        .or_else(|| call.get("name").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn tool_call_arguments(call: &Value) -> Option<Value> {
    let value = call
        .get("function")
        .and_then(|value| value.get("arguments"))
        .or_else(|| call.get("arguments"))?;
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).ok()
    } else if value.is_object() {
        Some(value.clone())
    } else {
        None
    }
}

fn compute_overview(sessions: &[SessionRow], message_stats: &MessageStats) -> Overview {
    let total_input_tokens = sessions.iter().map(|session| session.input_tokens).sum();
    let total_output_tokens = sessions.iter().map(|session| session.output_tokens).sum();
    let total_cache_read_tokens = sessions
        .iter()
        .map(|session| session.cache_read_tokens)
        .sum();
    let total_cache_write_tokens = sessions
        .iter()
        .map(|session| session.cache_write_tokens)
        .sum();
    let total_tokens = total_input_tokens
        + total_output_tokens
        + total_cache_read_tokens
        + total_cache_write_tokens;
    let total_tool_calls = sessions.iter().map(|session| session.tool_call_count).sum();
    let total_messages = if message_stats.total_messages > 0 {
        message_stats.total_messages
    } else {
        sessions.iter().map(|session| session.message_count).sum()
    };
    let recorded_estimated_cost = sessions
        .iter()
        .map(|session| session.estimated_cost_usd)
        .sum::<f64>();
    let recorded_actual_cost = sessions
        .iter()
        .map(|session| session.actual_cost_usd)
        .sum::<f64>();
    let sessions_with_recorded_cost = sessions
        .iter()
        .filter(|session| session.estimated_cost_usd > 0.0 || session.actual_cost_usd > 0.0)
        .count();

    let durations = sessions
        .iter()
        .filter_map(|session| {
            let end = session.ended_at?;
            (end > session.started_at).then_some(end - session.started_at)
        })
        .collect::<Vec<_>>();
    let total_duration = durations.iter().sum::<f64>();
    let total_hours = total_duration / 3600.0;
    let avg_session_duration_seconds = if durations.is_empty() {
        0.0
    } else {
        total_duration / durations.len() as f64
    };

    let started = sessions
        .iter()
        .map(|session| session.started_at)
        .collect::<Vec<_>>();
    let date_range_start = started.iter().copied().reduce(f64::min);
    let date_range_end = started.iter().copied().reduce(f64::max);

    Overview {
        total_sessions: sessions.len(),
        total_messages,
        total_tool_calls,
        total_input_tokens,
        total_output_tokens,
        total_cache_read_tokens,
        total_cache_write_tokens,
        total_tokens,
        recorded_estimated_cost,
        recorded_actual_cost,
        total_hours,
        avg_session_duration_seconds,
        avg_messages_per_session: total_messages as f64 / sessions.len() as f64,
        avg_tokens_per_session: total_tokens as f64 / sessions.len() as f64,
        user_messages: message_stats.user_messages,
        assistant_messages: message_stats.assistant_messages,
        tool_messages: message_stats.tool_messages,
        date_range_start,
        date_range_end,
        sessions_with_recorded_cost,
    }
}

fn compute_model_breakdown(sessions: &[SessionRow]) -> Vec<ModelBreakdown> {
    let mut map: BTreeMap<String, ModelBreakdown> = BTreeMap::new();
    for session in sessions {
        let model = display_model(&session.model);
        let entry = map.entry(model.clone()).or_insert(ModelBreakdown {
            model,
            sessions: 0,
            total_tokens: 0,
        });
        entry.sessions += 1;
        entry.total_tokens += session.input_tokens
            + session.output_tokens
            + session.cache_read_tokens
            + session.cache_write_tokens;
    }
    let mut values = map.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .total_tokens
            .cmp(&left.total_tokens)
            .then_with(|| right.sessions.cmp(&left.sessions))
            .then_with(|| left.model.cmp(&right.model))
    });
    values
}

fn compute_platform_breakdown(sessions: &[SessionRow]) -> Vec<PlatformBreakdown> {
    let mut map: BTreeMap<String, PlatformBreakdown> = BTreeMap::new();
    for session in sessions {
        let entry = map
            .entry(session.source.clone())
            .or_insert(PlatformBreakdown {
                platform: session.source.clone(),
                sessions: 0,
                messages: 0,
                total_tokens: 0,
            });
        entry.sessions += 1;
        entry.messages += session.message_count;
        entry.total_tokens += session.input_tokens
            + session.output_tokens
            + session.cache_read_tokens
            + session.cache_write_tokens;
    }
    let mut values = map.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .sessions
            .cmp(&left.sessions)
            .then_with(|| right.messages.cmp(&left.messages))
            .then_with(|| left.platform.cmp(&right.platform))
    });
    values
}

fn compute_activity(sessions: &[SessionRow]) -> ActivityBreakdown {
    let mut day_counts = [0_i64; 7];
    let mut hour_counts = [0_i64; 24];
    let mut active_days = BTreeSet::new();

    for session in sessions {
        let Some(dt) = to_local_datetime(session.started_at) else {
            continue;
        };
        let day_index = dt.weekday().num_days_from_monday() as usize;
        let hour = dt.hour() as usize;
        day_counts[day_index] += 1;
        hour_counts[hour] += 1;
        active_days.insert(dt.date_naive());
    }

    let day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    let by_day = day_counts
        .into_iter()
        .enumerate()
        .map(|(index, count)| (day_names[index].to_string(), count))
        .collect::<Vec<_>>();
    let by_hour = hour_counts
        .into_iter()
        .enumerate()
        .map(|(hour, count)| (hour as u32, count))
        .collect::<Vec<_>>();
    let busiest_day = by_day.iter().max_by_key(|(_, count)| *count).cloned();
    let busiest_hour = by_hour.iter().max_by_key(|(_, count)| *count).cloned();
    let max_streak = longest_date_streak(&active_days);

    ActivityBreakdown {
        by_day,
        by_hour,
        busiest_day,
        busiest_hour,
        active_days: active_days.len(),
        max_streak,
    }
}

fn compute_top_sessions(sessions: &[SessionRow]) -> Vec<TopSession> {
    let mut result = Vec::new();

    if let Some(longest) = sessions
        .iter()
        .filter_map(|session| {
            let end = session.ended_at?;
            (end > session.started_at).then_some((session, end - session.started_at))
        })
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        result.push(TopSession {
            label: "Longest session",
            session_id: abbreviate_session_id(&longest.0.id),
            value: format_duration(longest.1),
            date: format_date(longest.0.started_at),
        });
    }

    if let Some(session) = sessions.iter().max_by_key(|session| session.message_count)
        && session.message_count > 0
    {
        result.push(TopSession {
            label: "Most messages",
            session_id: abbreviate_session_id(&session.id),
            value: format!("{} msgs", session.message_count),
            date: format_date(session.started_at),
        });
    }

    if let Some(session) = sessions.iter().max_by_key(|session| {
        session.input_tokens
            + session.output_tokens
            + session.cache_read_tokens
            + session.cache_write_tokens
    }) {
        let total_tokens = session.input_tokens
            + session.output_tokens
            + session.cache_read_tokens
            + session.cache_write_tokens;
        if total_tokens > 0 {
            result.push(TopSession {
                label: "Most tokens",
                session_id: abbreviate_session_id(&session.id),
                value: format!("{} tokens", format_number(total_tokens)),
                date: format_date(session.started_at),
            });
        }
    }

    if let Some(session) = sessions
        .iter()
        .max_by_key(|session| session.tool_call_count)
        && session.tool_call_count > 0
    {
        result.push(TopSession {
            label: "Most tool calls",
            session_id: abbreviate_session_id(&session.id),
            value: format!("{} calls", session.tool_call_count),
            date: format_date(session.started_at),
        });
    }

    result
}

fn longest_date_streak(days: &BTreeSet<chrono::NaiveDate>) -> usize {
    let mut previous: Option<chrono::NaiveDate> = None;
    let mut current = 0_usize;
    let mut best = 0_usize;
    for day in days {
        if let Some(prev) = previous {
            if (*day - prev).num_days() == 1 {
                current += 1;
            } else {
                current = 1;
            }
        } else {
            current = 1;
        }
        best = best.max(current);
        previous = Some(*day);
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    days: i64,
    source: Option<&str>,
    overview: &Overview,
    models: &[ModelBreakdown],
    platforms: &[PlatformBreakdown],
    tools: &[ToolUsage],
    skills: &[SkillUsage],
    activity: &ActivityBreakdown,
    top_sessions: &[TopSession],
) {
    println!();
    println!("Hermes Insights");
    println!("Period: last {days} days{}", source_label(source));
    if let (Some(start), Some(end)) = (overview.date_range_start, overview.date_range_end) {
        println!(
            "Range:  {} - {}",
            format_full_date(start),
            format_full_date(end)
        );
    }
    println!();

    println!("Overview");
    println!("  Sessions:         {}", overview.total_sessions);
    println!(
        "  Messages:         {}",
        format_number(overview.total_messages)
    );
    println!(
        "  Tool calls:       {}",
        format_number(overview.total_tool_calls)
    );
    println!(
        "  Input tokens:     {}",
        format_number(overview.total_input_tokens)
    );
    println!(
        "  Output tokens:    {}",
        format_number(overview.total_output_tokens)
    );
    if overview.total_cache_read_tokens > 0 || overview.total_cache_write_tokens > 0 {
        println!(
            "  Cache tokens:     read {} / write {}",
            format_number(overview.total_cache_read_tokens),
            format_number(overview.total_cache_write_tokens)
        );
    }
    println!(
        "  Total tokens:     {}",
        format_number(overview.total_tokens)
    );
    if overview.recorded_estimated_cost > 0.0 || overview.recorded_actual_cost > 0.0 {
        println!(
            "  Recorded cost:    est ${:.4} / actual ${:.4}  ({} sessions)",
            overview.recorded_estimated_cost,
            overview.recorded_actual_cost,
            overview.sessions_with_recorded_cost
        );
    }
    if overview.total_hours > 0.0 {
        println!(
            "  Active time:      ~{}",
            format_duration(overview.total_hours * 3600.0)
        );
        println!(
            "  Avg session:      ~{}",
            format_duration(overview.avg_session_duration_seconds)
        );
    }
    println!(
        "  Avg msgs/session: {:.1}",
        overview.avg_messages_per_session
    );
    println!("  Avg toks/session: {:.1}", overview.avg_tokens_per_session);
    println!(
        "  Roles:            user {} / assistant {} / tool {}",
        format_number(overview.user_messages),
        format_number(overview.assistant_messages),
        format_number(overview.tool_messages)
    );

    if !models.is_empty() {
        println!();
        println!("Models Used");
        for model in models.iter().take(10) {
            println!(
                "  {:<30} {:>8} sessions {:>14} tokens",
                truncate_text(&model.model, 30),
                model.sessions,
                format_number(model.total_tokens)
            );
        }
    }

    if platforms.len() > 1
        || platforms
            .first()
            .is_some_and(|platform| platform.platform != "cli")
    {
        println!();
        println!("Platforms");
        for platform in platforms {
            println!(
                "  {:<14} {:>8} sessions {:>10} msgs {:>14} tokens",
                truncate_text(&platform.platform, 14),
                platform.sessions,
                format_number(platform.messages),
                format_number(platform.total_tokens)
            );
        }
    }

    if !tools.is_empty() {
        let total_calls = tools.iter().map(|tool| tool.count).sum::<i64>().max(1);
        println!();
        println!("Top Tools");
        for tool in tools.iter().take(15) {
            println!(
                "  {:<28} {:>8} calls {:>6.1}%",
                truncate_text(&tool.tool, 28),
                format_number(tool.count),
                tool.count as f64 / total_calls as f64 * 100.0
            );
        }
    }

    if !skills.is_empty() {
        println!();
        println!("Top Skills");
        for skill in skills.iter().take(10) {
            let last_used = skill
                .last_used_at
                .map(format_date)
                .unwrap_or_else(|| "—".to_string());
            println!(
                "  {:<28} {:>6} loads {:>6} edits {:>10}",
                truncate_text(&skill.skill, 28),
                format_number(skill.view_count),
                format_number(skill.manage_count),
                last_used
            );
        }
    }

    println!();
    println!("Activity");
    for (day, count) in &activity.by_day {
        println!(
            "  {:<3} {}",
            day,
            activity_bar(*count, busiest_count_by_day(activity))
        );
    }
    let peak_hours = activity
        .by_hour
        .iter()
        .filter(|(_, count)| *count > 0)
        .cloned()
        .collect::<Vec<_>>();
    if !peak_hours.is_empty() {
        let mut top_hours = peak_hours;
        top_hours.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        let summary = top_hours
            .into_iter()
            .take(5)
            .map(|(hour, count)| format!("{} ({count})", format_hour(hour)))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  Peak hours: {summary}");
    }
    if let Some((hour, count)) = activity.busiest_hour {
        println!("  Busiest hour: {} ({count})", format_hour(hour));
    }
    println!("  Active days: {}", activity.active_days);
    if activity.max_streak > 1 {
        println!("  Best streak: {} consecutive days", activity.max_streak);
    }

    if !top_sessions.is_empty() {
        println!();
        println!("Notable Sessions");
        for session in top_sessions {
            println!(
                "  {:<18} {:<18} ({}, {})",
                session.label, session.value, session.date, session.session_id
            );
        }
    }
}

fn source_label(source: Option<&str>) -> String {
    match source {
        Some(source) => format!(" ({source})"),
        None => String::new(),
    }
}

fn busiest_count_by_day(activity: &ActivityBreakdown) -> i64 {
    activity
        .busiest_day
        .as_ref()
        .map(|(_, count)| *count)
        .unwrap_or(0)
}

fn activity_bar(count: i64, peak: i64) -> String {
    if count <= 0 || peak <= 0 {
        return String::new();
    }
    let width = ((count as f64 / peak as f64) * 15.0).round().max(1.0) as usize;
    format!("{} {}", "█".repeat(width), count)
}

fn display_model(model: &str) -> String {
    if model.trim().is_empty() {
        return "unknown".to_string();
    }
    model
        .rsplit('/')
        .next()
        .map(str::to_string)
        .unwrap_or_else(|| model.to_string())
}

fn format_number(value: i64) -> String {
    let negative = value < 0;
    let digits = value.abs().to_string();
    let mut out = String::new();
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    let mut value = out.chars().rev().collect::<String>();
    if negative {
        value.insert(0, '-');
    }
    value
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    text.chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>()
        + "…"
}

fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{:.0}s", seconds.max(0.0));
    }
    if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor() as i64;
        let remainder = (seconds as i64) % 60;
        return format!("{minutes}m {remainder}s");
    }
    if seconds < 86_400.0 {
        let hours = (seconds / 3600.0).floor() as i64;
        let minutes = ((seconds as i64) % 3600) / 60;
        return format!("{hours}h {minutes}m");
    }
    let days = (seconds / 86_400.0).floor() as i64;
    let hours = ((seconds as i64) % 86_400) / 3600;
    format!("{days}d {hours}h")
}

fn format_date(timestamp: f64) -> String {
    to_local_datetime(timestamp)
        .map(|value| value.format("%b %d").to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn format_full_date(timestamp: f64) -> String {
    to_local_datetime(timestamp)
        .map(|value| value.format("%b %d, %Y").to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn format_hour(hour: u32) -> String {
    let suffix = if hour < 12 { "AM" } else { "PM" };
    let display = match hour % 12 {
        0 => 12,
        value => value,
    };
    format!("{display}{suffix}")
}

fn abbreviate_session_id(session_id: &str) -> String {
    session_id.chars().take(16).collect()
}

fn to_local_datetime(timestamp: f64) -> Option<chrono::DateTime<Local>> {
    if !timestamp.is_finite() {
        return None;
    }
    Local.timestamp_opt(timestamp as i64, 0).single()
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::{MessageAppend, SessionCreate};
    use rusqlite::Connection as SqlConnection;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn tool_call_name_supports_openai_and_flat_shapes() {
        let nested = serde_json::json!({"function": {"name": "search_files"}});
        let flat = serde_json::json!({"name": "terminal"});
        assert_eq!(tool_call_name(&nested).as_deref(), Some("search_files"));
        assert_eq!(tool_call_name(&flat).as_deref(), Some("terminal"));
    }

    #[test]
    fn extract_skill_calls_reads_json_arguments() {
        let raw = serde_json::json!([
            {
                "function": {
                    "name": "skill_view",
                    "arguments": "{\"name\":\"deploy-checklist\"}"
                }
            },
            {
                "function": {
                    "name": "skill_manage",
                    "arguments": {"name": "ops-runbook"}
                }
            }
        ])
        .to_string();
        let calls = extract_skill_calls(&raw);
        assert_eq!(
            calls,
            vec![
                ("skill_view".to_string(), "deploy-checklist".to_string()),
                ("skill_manage".to_string(), "ops-runbook".to_string())
            ]
        );
    }

    #[test]
    fn print_insights_reports_seeded_usage() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        context.ensure_hermes_home().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: "sess-1".to_string(),
                source: "cli".to_string(),
                user_id: None,
                model: Some("openrouter/openai/gpt-4.1-mini".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "sess-1",
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String("working".to_string())),
                    tool_call_id: None,
                    tool_calls: Some(serde_json::json!([
                        {
                            "function": {
                                "name": "skill_view",
                                "arguments": "{\"name\":\"deploy-checklist\"}"
                            }
                        },
                        {
                            "function": {
                                "name": "search_files",
                                "arguments": "{}"
                            }
                        }
                    ])),
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store
            .append_message(
                "sess-1",
                &MessageAppend {
                    role: "tool".to_string(),
                    content: Some(Value::String("tool output".to_string())),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: Some("search_files".to_string()),
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();
        store.end_session("sess-1", "completed").unwrap();

        let conn = SqlConnection::open(context.state_db_path()).unwrap();
        conn.execute(
            "UPDATE sessions
             SET input_tokens = 1200,
                 output_tokens = 450,
                 cache_read_tokens = 50,
                 cache_write_tokens = 25,
                 estimated_cost_usd = 0.1234
             WHERE id = 'sess-1'",
            [],
        )
        .unwrap();

        let cutoff = now_ts() - 86_400.0;
        let sessions = fetch_sessions(&conn, cutoff, Some("cli")).unwrap();
        let stats = fetch_message_stats(&conn, cutoff, Some("cli")).unwrap();
        let tools = fetch_tool_usage(&conn, cutoff, Some("cli")).unwrap();
        let skills = fetch_skill_usage(&conn, cutoff, Some("cli")).unwrap();
        let overview = compute_overview(&sessions, &stats);

        assert_eq!(sessions.len(), 1);
        assert_eq!(overview.total_tokens, 1725);
        assert_eq!(tools[0].tool, "search_files");
        assert_eq!(tools[0].count, 1);
        assert_eq!(skills[0].skill, "deploy-checklist");
        assert_eq!(skills[0].view_count, 1);
    }

    #[test]
    fn print_insights_handles_missing_state_db() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        let message = empty_message(7, Some("cli"));
        assert_eq!(
            message,
            "No sessions found in the last 7 days (source: cli)."
        );
        assert!(!Path::new(&context.state_db_path()).exists());
    }
}
