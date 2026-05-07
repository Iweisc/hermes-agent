use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::sleep;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Duration as ChronoDuration, Local, NaiveDateTime};
use clap::Args;
use hermes_core::HermesContext;

const DEFAULT_LOG_NAME: &str = "agent";
const POLL_INTERVAL_MS: u64 = 300;
const SMALL_FILE_BYTES: u64 = 1_048_576;
const VALID_LEVELS: &[&str] = &["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
const COMPONENT_NAMES: &[&str] = &["gateway", "agent", "tools", "cli", "cron"];
const GATEWAY_PREFIXES: &[&str] = &["gateway"];
const AGENT_PREFIXES: &[&str] = &["agent", "run_agent", "model_tools", "batch_runner"];
const TOOLS_PREFIXES: &[&str] = &["tools"];
const CLI_PREFIXES: &[&str] = &["hermes_cli", "cli"];
const CRON_PREFIXES: &[&str] = &["cron"];

#[derive(Args, Debug)]
pub struct LogsArgs {
    #[arg(value_name = "LOG")]
    pub log: Option<String>,
    #[arg(short = 'n', long = "lines", default_value_t = 50)]
    pub lines: usize,
    #[arg(short = 'f', long)]
    pub follow: bool,
    #[arg(long)]
    pub level: Option<String>,
    #[arg(long)]
    pub session: Option<String>,
    #[arg(long)]
    pub component: Option<String>,
    #[arg(long)]
    pub since: Option<String>,
}

pub fn print_logs(context: &HermesContext, args: LogsArgs) -> Result<(), Box<dyn Error>> {
    if args.lines == 0 {
        return Err("--lines must be positive".into());
    }

    let selected = args
        .log
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_LOG_NAME)
        .to_ascii_lowercase();
    if selected == "list" {
        list_logs(context)?;
        return Ok(());
    }

    let filename = match selected.as_str() {
        "agent" => "agent.log",
        "errors" => "errors.log",
        "gateway" => "gateway.log",
        other => {
            return Err(
                format!("Unknown log '{other}'. Available: agent, errors, gateway, list.").into(),
            );
        }
    };

    let log_path = context.hermes_home().join("logs").join(filename);
    if !log_path.is_file() {
        return Err(format!(
            "Log file not found: {}. Run Hermes first to generate it.",
            log_path.display()
        )
        .into());
    }

    let min_level = normalize_level(args.level.as_deref())?;
    let session_filter = normalize_non_empty(args.session.as_deref());
    let component_prefixes = resolve_component_prefixes(args.component.as_deref())?;
    let since_dt = parse_since(args.since.as_deref())?;
    let has_filters = min_level.is_some()
        || session_filter.is_some()
        || component_prefixes.is_some()
        || since_dt.is_some();

    let lines = read_tail(
        &log_path,
        args.lines,
        has_filters,
        min_level.as_deref(),
        session_filter.as_deref(),
        since_dt,
        component_prefixes,
    )?;

    let mut filters = Vec::new();
    if let Some(level) = min_level.as_deref() {
        filters.push(format!("level>={level}"));
    }
    if let Some(session) = session_filter.as_deref() {
        filters.push(format!("session={session}"));
    }
    if let Some(component) = normalize_non_empty(args.component.as_deref()) {
        filters.push(format!("component={component}"));
    }
    if let Some(since) = normalize_non_empty(args.since.as_deref()) {
        filters.push(format!("since={since}"));
    }
    let filter_desc = if filters.is_empty() {
        String::new()
    } else {
        format!(" [{}]", filters.join(", "))
    };

    if args.follow {
        println!(
            "--- {}/logs/{}{} (Ctrl+C to stop) ---",
            context.display_hermes_home(),
            filename,
            filter_desc
        );
    } else {
        println!(
            "--- {}/logs/{}{} (last {}) ---",
            context.display_hermes_home(),
            filename,
            filter_desc,
            args.lines
        );
    }
    for line in lines {
        print!("{line}");
    }
    if !args.follow {
        return Ok(());
    }

    follow_log(
        &log_path,
        min_level.as_deref(),
        session_filter.as_deref(),
        since_dt,
        component_prefixes,
    )?;
    println!("\n--- stopped ---");
    Ok(())
}

fn list_logs(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let log_dir = context.hermes_home().join("logs");
    if !log_dir.is_dir() {
        println!(
            "No logs directory at {}/logs/",
            context.display_hermes_home()
        );
        return Ok(());
    }

    println!("Log files in {}/logs/:\n", context.display_hermes_home());
    let mut entries = fs::read_dir(&log_dir)?
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".log"))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());

    if entries.is_empty() {
        println!("  (no log files yet — run `hermes chat` to generate logs)");
        return Ok(());
    }

    for entry in entries {
        let path = entry.path();
        let metadata = entry.metadata()?;
        let size = format_bytes(metadata.len());
        let age = metadata
            .modified()
            .ok()
            .map(format_age)
            .unwrap_or_else(|| "?".to_string());
        println!(
            "  {:<25} {:>8}   {}",
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
            size,
            age
        );
    }
    Ok(())
}

fn read_tail(
    path: &Path,
    lines: usize,
    has_filters: bool,
    min_level: Option<&str>,
    session_filter: Option<&str>,
    since: Option<NaiveDateTime>,
    component_prefixes: Option<&[&str]>,
) -> io::Result<Vec<String>> {
    if has_filters {
        let raw_lines = read_last_n_lines(path, (lines.saturating_mul(20)).max(2000))?;
        let mut filtered = raw_lines
            .into_iter()
            .filter(|line| {
                matches_filters(line, min_level, session_filter, since, component_prefixes)
            })
            .collect::<Vec<_>>();
        if filtered.len() > lines {
            filtered = filtered.split_off(filtered.len() - lines);
        }
        Ok(filtered)
    } else {
        read_last_n_lines(path, lines)
    }
}

fn read_last_n_lines(path: &Path, lines: usize) -> io::Result<Vec<String>> {
    if lines == 0 {
        return Ok(Vec::new());
    }
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    if size == 0 {
        return Ok(Vec::new());
    }

    if size <= SMALL_FILE_BYTES {
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        return Ok(last_lines_from_text(&text, lines));
    }

    let mut pos = size;
    let mut chunk_size = 8192_usize;
    let mut buffer = Vec::new();
    let mut newline_count = 0_usize;

    while pos > 0 && newline_count <= lines + 1 {
        let read_size = chunk_size.min(pos as usize);
        pos -= read_size as u64;
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk = vec![0_u8; read_size];
        file.read_exact(&mut chunk)?;
        chunk.extend_from_slice(&buffer);
        newline_count = chunk.iter().filter(|byte| **byte == b'\n').count();
        buffer = chunk;
        chunk_size = (chunk_size * 2).min(65_536);
    }

    let text = String::from_utf8_lossy(&buffer);
    Ok(last_lines_from_text(&text, lines))
}

fn last_lines_from_text(text: &str, lines: usize) -> Vec<String> {
    let mut collected = text
        .split_inclusive('\n')
        .filter(|line| !line.trim().is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if collected.len() > lines {
        collected = collected.split_off(collected.len() - lines);
    }
    collected
}

fn follow_log(
    path: &Path,
    min_level: Option<&str>,
    session_filter: Option<&str>,
    since: Option<NaiveDateTime>,
    component_prefixes: Option<&[&str]>,
) -> Result<(), Box<dyn Error>> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
    })?;

    let mut position = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    while !stop.load(Ordering::SeqCst) {
        let current_len = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        if current_len < position {
            position = 0;
        }
        if current_len > position {
            let mut file = File::open(path)?;
            file.seek(SeekFrom::Start(position))?;
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                line.clear();
                let read = reader.read_line(&mut line)?;
                if read == 0 {
                    break;
                }
                position += read as u64;
                if matches_filters(&line, min_level, session_filter, since, component_prefixes) {
                    print!("{line}");
                }
            }
        }
        sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
    Ok(())
}

fn parse_since(raw: Option<&str>) -> Result<Option<NaiveDateTime>, Box<dyn Error>> {
    let Some(raw) = normalize_non_empty(raw) else {
        return Ok(None);
    };
    if raw.len() < 2 {
        return Err(
            format!("Invalid --since value '{raw}'. Use formats like 30m, 1h, or 2d.").into(),
        );
    }
    let (value, unit) = raw.split_at(raw.len() - 1);
    let quantity: i64 = value
        .trim()
        .parse()
        .map_err(|_| format!("Invalid --since value '{raw}'. Use formats like 30m, 1h, or 2d."))?;
    let delta = match unit {
        "s" => ChronoDuration::seconds(quantity),
        "m" => ChronoDuration::minutes(quantity),
        "h" => ChronoDuration::hours(quantity),
        "d" => ChronoDuration::days(quantity),
        _ => {
            return Err(
                format!("Invalid --since value '{raw}'. Use formats like 30m, 1h, or 2d.").into(),
            );
        }
    };
    Ok(Some((Local::now() - delta).naive_local()))
}

fn parse_line_timestamp(line: &str) -> Option<NaiveDateTime> {
    let text = line.get(0..19)?;
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").ok()
}

fn normalize_level(raw: Option<&str>) -> Result<Option<String>, Box<dyn Error>> {
    let Some(raw) = normalize_non_empty(raw) else {
        return Ok(None);
    };
    let normalized = raw.to_ascii_uppercase();
    if !VALID_LEVELS.contains(&normalized.as_str()) {
        return Err(format!(
            "Invalid --level '{raw}'. Use one of: {}.",
            VALID_LEVELS.join(", ")
        )
        .into());
    }
    Ok(Some(normalized))
}

fn resolve_component_prefixes(
    raw: Option<&str>,
) -> Result<Option<&'static [&'static str]>, Box<dyn Error>> {
    let Some(raw) = normalize_non_empty(raw) else {
        return Ok(None);
    };
    let normalized = raw.to_ascii_lowercase();
    let prefixes = match normalized.as_str() {
        "gateway" => GATEWAY_PREFIXES,
        "agent" => AGENT_PREFIXES,
        "tools" => TOOLS_PREFIXES,
        "cli" => CLI_PREFIXES,
        "cron" => CRON_PREFIXES,
        _ => {
            return Err(format!(
                "Unknown --component '{raw}'. Available: {}.",
                COMPONENT_NAMES.join(", ")
            )
            .into());
        }
    };
    Ok(Some(prefixes))
}

fn matches_filters(
    line: &str,
    min_level: Option<&str>,
    session_filter: Option<&str>,
    since: Option<NaiveDateTime>,
    component_prefixes: Option<&[&str]>,
) -> bool {
    if let Some(cutoff) = since
        && let Some(timestamp) = parse_line_timestamp(line)
        && timestamp < cutoff
    {
        return false;
    }

    if let Some(min_level) = min_level
        && let Some(level) = extract_level(line)
        && level_order(level) < level_order(min_level)
    {
        return false;
    }

    if let Some(session_filter) = session_filter
        && !line.contains(session_filter)
    {
        return false;
    }

    if let Some(prefixes) = component_prefixes {
        let Some(logger_name) = extract_logger_name(line) else {
            return false;
        };
        if !prefixes
            .iter()
            .any(|prefix| logger_name.starts_with(prefix))
        {
            return false;
        }
    }

    true
}

fn extract_level(line: &str) -> Option<&str> {
    let token = line.split_whitespace().nth(2)?;
    VALID_LEVELS.contains(&token).then_some(token)
}

fn extract_logger_name(line: &str) -> Option<&str> {
    let mut parts = line.split_whitespace().skip(3);
    let token = parts.next()?;
    let logger = if token.starts_with('[') && token.ends_with(']') {
        parts.next()?
    } else {
        token
    };
    logger.strip_suffix(':')
}

fn level_order(level: &str) -> usize {
    match level {
        "TRACE" => 0,
        "DEBUG" => 1,
        "INFO" => 2,
        "WARNING" => 3,
        "ERROR" => 4,
        "CRITICAL" => 5,
        _ => 0,
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn format_age(modified: SystemTime) -> String {
    let now: DateTime<Local> = Local::now();
    let modified: DateTime<Local> = modified.into();
    let age = now.signed_duration_since(modified);
    if age < ChronoDuration::minutes(1) {
        "just now".to_string()
    } else if age < ChronoDuration::hours(1) {
        format!("{}m ago", age.num_minutes())
    } else if age < ChronoDuration::days(1) {
        format!("{}h ago", age.num_hours())
    } else {
        modified.format("%Y-%m-%d").to_string()
    }
}

fn normalize_non_empty(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file_path(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-logs-{name}-{unique}.log"))
    }

    #[test]
    fn parses_since_strings() {
        assert!(parse_since(Some("30m")).unwrap().is_some());
        assert!(parse_since(Some("2d")).unwrap().is_some());
        assert!(parse_since(Some("bogus")).is_err());
    }

    #[test]
    fn extracts_level_and_logger_name() {
        let line = "2026-05-07 12:45:08 INFO [sess_1] hermes_cli: startup\n";
        assert_eq!(extract_level(line), Some("INFO"));
        assert_eq!(extract_logger_name(line), Some("hermes_cli"));
    }

    #[test]
    fn filters_tail_lines() {
        let path = temp_file_path("tail");
        fs::write(
            &path,
            concat!(
                "2026-05-07 12:45:08 INFO hermes_cli: startup\n",
                "2026-05-07 12:45:09 WARNING tools.exec: warning\n",
                "2026-05-07 12:45:10 ERROR gateway.run: failed\n"
            ),
        )
        .unwrap();

        let lines = read_tail(
            &path,
            10,
            true,
            Some("WARNING"),
            None,
            None,
            Some(GATEWAY_PREFIXES),
        )
        .unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("gateway.run"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reads_last_lines_from_large_text() {
        let mut text = String::new();
        for index in 0..5000 {
            text.push_str(&format!("line-{index}\n"));
        }
        let path = temp_file_path("large");
        fs::write(&path, text).unwrap();

        let lines = read_last_n_lines(&path, 3).unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line-4997\n");
        assert_eq!(lines[2], "line-4999\n");

        let _ = fs::remove_file(path);
    }
}
