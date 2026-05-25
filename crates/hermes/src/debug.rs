use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Args, Subcommand};
#[cfg(test)]
use hermes_core::HermesConfig;
use hermes_core::{HermesContext, LoadedConfig};
use reqwest::blocking::{Client, multipart::Form};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
#[cfg(test)]
use serde_yaml::Value as YamlValue;

use crate::dump;

const REDACTION_BANNER: &str =
    "[hermes debug share: log content redacted at upload time. run with --no-redact to disable]\n";
const PRIVACY_NOTICE: &str = "\
WARNING: This uploads the following to a public paste service:
  - System info, provider, and API key presence (not the raw keys)
  - Recent log tails (may contain conversation fragments and file paths)
  - Full agent.log and gateway.log snapshots when present

Pastes auto-delete after 6 hours.
";
const PASTE_RS_URL: &str = "https://paste.rs/";
const DPASTE_COM_URL: &str = "https://dpaste.com/api/";
const MAX_LOG_BYTES: usize = 512_000;
const AUTO_DELETE_SECONDS: u64 = 21_600;
const LOG_SUFFIXES: &[(&str, &str)] = &[
    ("agent", "agent.log"),
    ("errors", "errors.log"),
    ("gateway", "gateway.log"),
];
const SECRET_ENV_MARKERS: &[&str] = &["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"];
const SECRET_JSON_MARKERS: &[&str] = &["key", "token", "secret", "password", "credential"];

#[derive(Subcommand, Debug)]
pub enum DebugCommand {
    Share(DebugShareArgs),
    Delete { urls: Vec<String> },
}

#[derive(Args, Debug)]
pub struct DebugShareArgs {
    #[arg(long, default_value_t = 200)]
    pub lines: usize,
    #[arg(long, default_value_t = 7)]
    pub expire: u64,
    #[arg(long)]
    pub local: bool,
    #[arg(long = "no-redact")]
    pub no_redact: bool,
}

#[derive(Debug, Clone)]
struct PasteEndpoints {
    paste_rs_url: String,
    dpaste_url: String,
}

#[derive(Debug, Clone)]
struct LogSnapshot {
    tail_text: String,
    full_text: Option<String>,
}

#[derive(Debug, Clone)]
struct DebugLogSnapshots {
    agent: LogSnapshot,
    errors: LogSnapshot,
    gateway: LogSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PendingPasteEntry {
    url: String,
    expire_at: f64,
}

impl Default for PasteEndpoints {
    fn default() -> Self {
        Self {
            paste_rs_url: PASTE_RS_URL.to_string(),
            dpaste_url: DPASTE_COM_URL.to_string(),
        }
    }
}

pub fn print_debug(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<DebugCommand>,
) -> Result<(), Box<dyn Error>> {
    best_effort_sweep_expired_pastes(context, &PasteEndpoints::default());
    match command {
        Some(DebugCommand::Share(args)) => run_debug_share(context, loaded, args),
        Some(DebugCommand::Delete { urls }) => run_debug_delete(context, urls),
        None => {
            print_debug_usage();
            Ok(())
        }
    }
}

fn run_debug_share(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: DebugShareArgs,
) -> Result<(), Box<dyn Error>> {
    if args.lines == 0 {
        return Err("--lines must be positive".into());
    }
    if args.expire == 0 {
        return Err("--expire must be positive".into());
    }

    let endpoints = PasteEndpoints::default();
    let redact = !args.no_redact;

    if !args.local {
        println!("{PRIVACY_NOTICE}");
    }

    println!("Collecting debug report...");
    let dump_text = dump::render_dump(context, loaded, false)?;
    let log_snapshots = capture_default_log_snapshots(context, args.lines, redact)?;
    let mut report = collect_debug_report(args.lines, &dump_text, &log_snapshots);
    let mut agent_log = build_full_log_body(
        &dump_text,
        "agent.log",
        log_snapshots.agent.full_text.as_deref(),
    );
    let mut gateway_log = build_full_log_body(
        &dump_text,
        "gateway.log",
        log_snapshots.gateway.full_text.as_deref(),
    );

    if redact {
        report = format!("{REDACTION_BANNER}{report}");
        if let Some(text) = agent_log.as_mut() {
            *text = format!("{REDACTION_BANNER}{text}");
        }
        if let Some(text) = gateway_log.as_mut() {
            *text = format!("{REDACTION_BANNER}{text}");
        }
    }

    if args.local {
        print!("{report}");
        if let Some(text) = agent_log {
            println!("\n\n============================================================");
            println!("FULL agent.log");
            println!("============================================================\n");
            print!("{text}");
        }
        if let Some(text) = gateway_log {
            println!("\n\n============================================================");
            println!("FULL gateway.log");
            println!("============================================================\n");
            print!("{text}");
        }
        return Ok(());
    }

    println!("Uploading...");
    let client = build_http_client()?;
    let mut urls = BTreeMap::new();
    let mut failures = Vec::new();

    match upload_to_paste_services(&client, &report, args.expire, &endpoints) {
        Ok(url) => {
            urls.insert("Report".to_string(), url);
        }
        Err(error) => {
            eprintln!("\nUpload failed: {error}");
            println!("\nFull report printed below -- copy it manually:\n");
            print!("{report}");
            return Err("debug report upload failed".into());
        }
    }

    if let Some(text) = agent_log.as_deref() {
        match upload_to_paste_services(&client, text, args.expire, &endpoints) {
            Ok(url) => {
                urls.insert("agent.log".to_string(), url);
            }
            Err(error) => failures.push(format!("agent.log: {error}")),
        }
    }

    if let Some(text) = gateway_log.as_deref() {
        match upload_to_paste_services(&client, text, args.expire, &endpoints) {
            Ok(url) => {
                urls.insert("gateway.log".to_string(), url);
            }
            Err(error) => failures.push(format!("gateway.log: {error}")),
        }
    }

    println!("\nDebug report uploaded:");
    let width = urls.keys().map(String::len).max().unwrap_or(6);
    for (label, url) in &urls {
        println!("  {label:<width$}  {url}", width = width);
    }
    if !failures.is_empty() {
        println!("\n  failed to upload: {}", failures.join(", "));
    }

    let uploaded = urls.values().cloned().collect::<Vec<_>>();
    record_pending(context, &uploaded, AUTO_DELETE_SECONDS, &endpoints)?;
    println!("\nPastes will auto-delete in 6 hours.");
    println!("To delete now: hermes debug delete <url>");
    println!("\nShare these links with the Hermes team for support.");
    Ok(())
}

fn run_debug_delete(context: &HermesContext, urls: Vec<String>) -> Result<(), Box<dyn Error>> {
    let endpoints = PasteEndpoints::default();
    if urls.is_empty() {
        print_debug_usage();
        return Ok(());
    }
    let client = build_http_client()?;
    for url in urls {
        match delete_paste(&client, &url, &endpoints) {
            Ok(true) => println!("deleted: {url}"),
            Ok(false) => println!("failed: {url} (unexpected response)"),
            Err(error) => println!("failed: {url} ({error})"),
        }
    }
    best_effort_sweep_expired_pastes(context, &endpoints);
    Ok(())
}

fn print_debug_usage() {
    println!("Usage: hermes debug <command>");
    println!();
    println!("Commands:");
    println!("  share    Upload a debug report and print URLs");
    println!("  delete   Delete a previously uploaded paste");
    println!();
    println!("Options (share):");
    println!("  --lines N     Number of log lines to include (default: 200)");
    println!("  --expire N    Paste expiry in days (default: 7)");
    println!("  --local       Print the report locally instead of uploading");
    println!("  --no-redact   Disable upload-time secret redaction");
    println!();
    println!("Options (delete):");
    println!("  <url> ...     One or more paste URLs to delete");
}

fn collect_debug_report(
    log_lines: usize,
    dump_text: &str,
    snapshots: &DebugLogSnapshots,
) -> String {
    let errors_lines = log_lines.min(100);
    [
        dump_text.trim_end_matches('\n').to_string(),
        format!(
            "\n\n--- agent.log (last {log_lines} lines) ---\n{}",
            snapshots.agent.tail_text
        ),
        format!(
            "\n\n--- errors.log (last {errors_lines} lines) ---\n{}",
            snapshots.errors.tail_text
        ),
        format!(
            "\n\n--- gateway.log (last {errors_lines} lines) ---\n{}",
            snapshots.gateway.tail_text
        ),
        "\n".to_string(),
    ]
    .join("")
}

fn build_full_log_body(dump_text: &str, label: &str, text: Option<&str>) -> Option<String> {
    text.map(|body| {
        format!(
            "{}\n\n--- full {label} ---\n{}",
            dump_text.trim_end_matches('\n'),
            body
        )
    })
}

fn capture_default_log_snapshots(
    context: &HermesContext,
    log_lines: usize,
    redact: bool,
) -> Result<DebugLogSnapshots, Box<dyn Error>> {
    Ok(DebugLogSnapshots {
        agent: capture_log_snapshot(context, "agent", log_lines, MAX_LOG_BYTES, redact)?,
        errors: capture_log_snapshot(context, "errors", log_lines.min(100), MAX_LOG_BYTES, redact)?,
        gateway: capture_log_snapshot(
            context,
            "gateway",
            log_lines.min(100),
            MAX_LOG_BYTES,
            redact,
        )?,
    })
}

fn capture_log_snapshot(
    context: &HermesContext,
    log_name: &str,
    tail_lines: usize,
    max_bytes: usize,
    redact: bool,
) -> Result<LogSnapshot, Box<dyn Error>> {
    let Some(primary) = primary_log_path(context, log_name) else {
        return Ok(LogSnapshot {
            tail_text: "(file not found)".to_string(),
            full_text: None,
        });
    };
    let Some(log_path) = resolve_log_path(&primary) else {
        let tail_text = if primary.exists() {
            "(file empty)".to_string()
        } else {
            "(file not found)".to_string()
        };
        return Ok(LogSnapshot {
            tail_text,
            full_text: None,
        });
    };

    let size = fs::metadata(&log_path)?.len() as usize;
    if size == 0 {
        return Ok(LogSnapshot {
            tail_text: "(file empty)".to_string(),
            full_text: None,
        });
    }

    let mut file = File::open(&log_path)?;
    let (raw, truncated) = if size <= max_bytes {
        let mut raw = Vec::with_capacity(size);
        file.read_to_end(&mut raw)?;
        (raw, false)
    } else {
        read_log_tail_bytes(&mut file, size, tail_lines, max_bytes)?
    };

    let mut full_raw = raw.clone();
    if truncated && full_raw.len() > max_bytes {
        let cut = full_raw.len() - max_bytes;
        let on_boundary = cut > 0 && full_raw[cut - 1] == b'\n';
        full_raw = full_raw.split_off(cut);
        if !on_boundary && let Some(position) = full_raw.iter().position(|byte| *byte == b'\n') {
            full_raw = full_raw.split_off(position + 1);
        }
    }

    let all_text = String::from_utf8_lossy(&raw).to_string();
    let mut tail_text = last_lines(&all_text, tail_lines).join("");
    if tail_text.trim().is_empty() {
        tail_text = "(file empty)".to_string();
    }

    let mut full_text = String::from_utf8_lossy(&full_raw).to_string();
    if truncated {
        full_text = format!(
            "[... truncated -- showing last ~{}KB ...]\n{}",
            max_bytes / 1024,
            full_text
        );
    }

    if redact {
        tail_text = redact_upload_text(context, &tail_text);
        full_text = redact_upload_text(context, &full_text);
    }

    Ok(LogSnapshot {
        tail_text,
        full_text: Some(full_text),
    })
}

fn read_log_tail_bytes(
    file: &mut File,
    size: usize,
    tail_lines: usize,
    max_bytes: usize,
) -> Result<(Vec<u8>, bool), Box<dyn Error>> {
    let mut position = size;
    let mut chunk_size = 8192_usize;
    let mut chunks = Vec::new();
    let mut total = 0_usize;
    let mut newline_count = 0_usize;

    while position > 0
        && (total < max_bytes || newline_count <= tail_lines + 1)
        && total < max_bytes.saturating_mul(2)
    {
        let read_size = chunk_size.min(position);
        position -= read_size;
        file.seek(SeekFrom::Start(position as u64))?;
        let mut chunk = vec![0_u8; read_size];
        file.read_exact(&mut chunk)?;
        newline_count += chunk.iter().filter(|byte| **byte == b'\n').count();
        total += chunk.len();
        chunks.push(chunk);
        chunk_size = (chunk_size * 2).min(65_536);
    }

    let truncated = position > 0;
    chunks.reverse();
    let mut joined = Vec::with_capacity(total);
    for chunk in chunks {
        joined.extend_from_slice(&chunk);
    }
    Ok((joined, truncated))
}

fn primary_log_path(context: &HermesContext, log_name: &str) -> Option<PathBuf> {
    LOG_SUFFIXES.iter().find_map(|(name, suffix)| {
        (*name == log_name).then(|| context.hermes_home().join("logs").join(suffix))
    })
}

fn resolve_log_path(primary: &Path) -> Option<PathBuf> {
    if primary.exists()
        && fs::metadata(primary)
            .ok()
            .is_some_and(|meta| meta.len() > 0)
    {
        return Some(primary.to_path_buf());
    }
    let rotated = primary.with_file_name(format!("{}.1", primary.file_name()?.to_string_lossy()));
    if rotated.exists()
        && fs::metadata(&rotated)
            .ok()
            .is_some_and(|meta| meta.len() > 0)
    {
        return Some(rotated);
    }
    None
}

fn last_lines(text: &str, lines: usize) -> Vec<String> {
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

fn redact_upload_text(context: &HermesContext, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut redacted = redact_bearer_tokens(text.to_string());
    let mut secrets = collect_sensitive_values(context);
    secrets.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    secrets.dedup();
    for secret in secrets {
        redacted = redacted.replace(&secret, "[REDACTED]");
    }
    redacted
}

fn redact_bearer_tokens(text: String) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(index) = rest.find("Bearer ") {
        output.push_str(&rest[..index + 7]);
        let token_start = index + 7;
        let token_end = rest[token_start..]
            .find(|ch: char| {
                ch.is_whitespace() || ch == '"' || ch == '\'' || ch == ',' || ch == ']'
            })
            .map(|offset| token_start + offset)
            .unwrap_or(rest.len());
        if token_end > token_start {
            output.push_str("[REDACTED]");
        }
        rest = &rest[token_end..];
    }
    output.push_str(rest);
    output
}

fn collect_sensitive_values(context: &HermesContext) -> Vec<String> {
    let mut values = std::env::vars()
        .filter_map(|(key, value)| {
            let key = key.to_ascii_uppercase();
            let trimmed = value.trim();
            (trimmed.len() >= 8 && SECRET_ENV_MARKERS.iter().any(|marker| key.contains(marker)))
                .then(|| trimmed.to_string())
        })
        .collect::<Vec<_>>();

    let auth_path = context.hermes_home().join("auth.json");
    if let Ok(raw) = fs::read_to_string(auth_path)
        && let Ok(json) = serde_json::from_str::<JsonValue>(&raw)
    {
        collect_sensitive_from_json(&json, None, &mut values);
    }

    values
}

fn collect_sensitive_from_json(value: &JsonValue, key: Option<&str>, out: &mut Vec<String>) {
    match value {
        JsonValue::Object(map) => {
            for (child_key, child_value) in map {
                collect_sensitive_from_json(child_value, Some(child_key), out);
            }
        }
        JsonValue::Array(items) => {
            for item in items {
                collect_sensitive_from_json(item, key, out);
            }
        }
        JsonValue::String(text) => {
            let trimmed = text.trim();
            if trimmed.len() < 8 {
                return;
            }
            if key.is_some_and(|name| {
                let lowered = name.to_ascii_lowercase();
                SECRET_JSON_MARKERS
                    .iter()
                    .any(|marker| lowered.contains(marker))
            }) {
                out.push(trimmed.to_string());
            }
        }
        _ => {}
    }
}

fn upload_to_paste_services(
    client: &Client,
    content: &str,
    expiry_days: u64,
    endpoints: &PasteEndpoints,
) -> Result<String, String> {
    let mut errors = Vec::new();

    match upload_paste_rs(client, &endpoints.paste_rs_url, content) {
        Ok(url) => return Ok(url),
        Err(error) => errors.push(format!("paste.rs: {error}")),
    }

    match upload_dpaste(client, &endpoints.dpaste_url, content, expiry_days) {
        Ok(url) => return Ok(url),
        Err(error) => errors.push(format!("dpaste.com: {error}")),
    }

    Err(format!(
        "Failed to upload to any paste service:\n  {}",
        errors.join("\n  ")
    ))
}

fn upload_paste_rs(client: &Client, url: &str, content: &str) -> Result<String, Box<dyn Error>> {
    let response = client
        .post(url.trim_end_matches('/'))
        .header("Content-Type", "text/plain; charset=utf-8")
        .header("User-Agent", "hermes-agent/debug-share")
        .body(content.to_string())
        .send()?
        .error_for_status()?;
    let body = response.text()?.trim().to_string();
    if !body.starts_with("http") {
        return Err(format!(
            "Unexpected response from paste.rs: {}",
            truncate_response(&body)
        )
        .into());
    }
    Ok(body)
}

fn upload_dpaste(
    client: &Client,
    url: &str,
    content: &str,
    expiry_days: u64,
) -> Result<String, Box<dyn Error>> {
    let form = Form::new()
        .text("content", content.to_string())
        .text("syntax", "text".to_string())
        .text("expiry_days", expiry_days.to_string());
    let response = client
        .post(url)
        .header("User-Agent", "hermes-agent/debug-share")
        .multipart(form)
        .send()?
        .error_for_status()?;
    let body = response.text()?.trim().to_string();
    if !body.starts_with("http") {
        return Err(format!(
            "Unexpected response from dpaste.com: {}",
            truncate_response(&body)
        )
        .into());
    }
    Ok(body)
}

fn delete_paste(
    client: &Client,
    url: &str,
    endpoints: &PasteEndpoints,
) -> Result<bool, Box<dyn Error>> {
    let Some(paste_id) = extract_paste_id(url, &endpoints.paste_rs_url) else {
        return Err(format!("Cannot delete: only paste.rs URLs are supported. Got: {url}").into());
    };
    let target = format!(
        "{}/{}",
        endpoints.paste_rs_url.trim_end_matches('/'),
        paste_id
    );
    let response = client
        .delete(target)
        .header("User-Agent", "hermes-agent/debug-share")
        .send()?;
    Ok(response.status().is_success())
}

fn record_pending(
    context: &HermesContext,
    urls: &[String],
    delay_seconds: u64,
    endpoints: &PasteEndpoints,
) -> Result<(), Box<dyn Error>> {
    let mut entries = load_pending(context);
    let expire_at = now_unix_seconds() + delay_seconds as f64;
    for url in urls {
        if extract_paste_id(url, &endpoints.paste_rs_url).is_none() {
            continue;
        }
        if let Some(existing) = entries.iter_mut().find(|entry| entry.url == *url) {
            existing.expire_at = existing.expire_at.max(expire_at);
        } else {
            entries.push(PendingPasteEntry {
                url: url.clone(),
                expire_at,
            });
        }
    }
    if !entries.is_empty() {
        save_pending(context, &entries)?;
    }
    Ok(())
}

fn load_pending(context: &HermesContext) -> Vec<PendingPasteEntry> {
    let path = pending_file(context);
    let Ok(raw) = fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<PendingPasteEntry>>(&raw).unwrap_or_default()
}

fn save_pending(
    context: &HermesContext,
    entries: &[PendingPasteEntry],
) -> Result<(), Box<dyn Error>> {
    let path = pending_file(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write(&path, serde_json::to_string_pretty(entries)?.as_bytes())
}

fn pending_file(context: &HermesContext) -> PathBuf {
    context.hermes_home().join("pastes").join("pending.json")
}

fn sweep_expired_pastes<F>(
    context: &HermesContext,
    endpoints: &PasteEndpoints,
    now: f64,
    mut deleter: F,
) -> Result<(usize, usize), Box<dyn Error>>
where
    F: FnMut(&str, &PasteEndpoints) -> Result<bool, String>,
{
    let entries = load_pending(context);
    if entries.is_empty() {
        return Ok((0, 0));
    }

    let mut deleted = 0_usize;
    let mut remaining = Vec::new();
    for entry in entries {
        if entry.expire_at > now {
            remaining.push(entry);
            continue;
        }

        match deleter(&entry.url, endpoints) {
            Ok(true) => {
                deleted += 1;
            }
            Ok(false) | Err(_) => {
                if entry.expire_at + 86_400.0 > now {
                    remaining.push(entry);
                } else {
                    deleted += 1;
                }
            }
        }
    }

    save_pending(context, &remaining)?;
    Ok((deleted, remaining.len()))
}

fn best_effort_sweep_expired_pastes(context: &HermesContext, endpoints: &PasteEndpoints) {
    let Ok(client) = build_http_client() else {
        return;
    };
    let _ = sweep_expired_pastes(context, endpoints, now_unix_seconds(), |url, endpoints| {
        delete_paste(&client, url, endpoints).map_err(|error| error.to_string())
    });
}

fn extract_paste_id(url: &str, paste_rs_base: &str) -> Option<String> {
    let normalized_url = url.trim().trim_end_matches('/');
    let candidates = if paste_rs_base.trim().trim_end_matches('/') == "https://paste.rs" {
        vec![
            "https://paste.rs".to_string(),
            "http://paste.rs".to_string(),
        ]
    } else {
        vec![paste_rs_base.trim().trim_end_matches('/').to_string()]
    };
    for base in candidates {
        if let Some(rest) = normalized_url.strip_prefix(&base) {
            let id = rest.trim_start_matches('/').trim();
            if !id.is_empty() {
                return Some(id.to_string());
            }
        }
    }
    None
}

fn truncate_response(text: &str) -> String {
    if text.chars().count() <= 200 {
        text.to_string()
    } else {
        text.chars().take(200).collect::<String>()
    }
}

fn build_http_client() -> Result<Client, Box<dyn Error>> {
    Ok(Client::builder().timeout(Duration::from_secs(30)).build()?)
}

fn now_unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;
    use tempfile::TempDir;

    fn temp_context() -> (TempDir, HermesContext, LoadedConfig) {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        let loaded = LoadedConfig {
            path: context.config_path(),
            raw: serde_yaml::from_str::<YamlValue>("{}").unwrap(),
            config: HermesConfig::default(),
            warnings: Vec::new(),
        };
        (home, context, loaded)
    }

    fn write_log(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];
        let mut header_end = None;
        let mut content_length = 0usize;

        loop {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);

            if header_end.is_none()
                && let Some(pos) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
            {
                let end = pos + 4;
                header_end = Some(end);
                let headers = String::from_utf8_lossy(&buffer[..end]);
                for line in headers.lines() {
                    if let Some(value) = line
                        .strip_prefix("Content-Length:")
                        .or_else(|| line.strip_prefix("content-length:"))
                    {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
            }

            if let Some(end) = header_end
                && buffer.len() >= end + content_length
            {
                break;
            }
        }

        String::from_utf8_lossy(&buffer).to_string()
    }

    #[test]
    fn extract_paste_id_supports_paste_rs_urls() {
        assert_eq!(
            extract_paste_id("https://paste.rs/abc123", PASTE_RS_URL).as_deref(),
            Some("abc123")
        );
        assert_eq!(
            extract_paste_id("http://paste.rs/xyz/", PASTE_RS_URL).as_deref(),
            Some("xyz")
        );
        assert!(extract_paste_id("https://dpaste.com/abc", PASTE_RS_URL).is_none());
    }

    #[test]
    fn collect_debug_report_includes_dump_and_log_sections() {
        let snapshots = DebugLogSnapshots {
            agent: LogSnapshot {
                tail_text: "agent tail\n".to_string(),
                full_text: None,
            },
            errors: LogSnapshot {
                tail_text: "errors tail\n".to_string(),
                full_text: None,
            },
            gateway: LogSnapshot {
                tail_text: "gateway tail\n".to_string(),
                full_text: None,
            },
        };
        let report = collect_debug_report(50, "--- hermes dump ---\n", &snapshots);
        assert!(report.contains("--- hermes dump ---"));
        assert!(report.contains("--- agent.log (last 50 lines) ---"));
        assert!(report.contains("errors tail"));
        assert!(report.contains("gateway tail"));
    }

    #[test]
    fn record_pending_only_tracks_paste_rs_urls() {
        let (_home, context, _) = temp_context();
        let endpoints = PasteEndpoints::default();
        record_pending(
            &context,
            &[
                "https://paste.rs/first".to_string(),
                "https://dpaste.com/second".to_string(),
            ],
            10,
            &endpoints,
        )
        .unwrap();
        let entries = load_pending(&context);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].url, "https://paste.rs/first");
    }

    #[test]
    fn sweep_expired_pastes_deletes_old_entries_and_keeps_future() {
        let (_home, context, _) = temp_context();
        let endpoints = PasteEndpoints::default();
        save_pending(
            &context,
            &[
                PendingPasteEntry {
                    url: "https://paste.rs/expired".to_string(),
                    expire_at: 10.0,
                },
                PendingPasteEntry {
                    url: "https://paste.rs/future".to_string(),
                    expire_at: 10_000.0,
                },
            ],
        )
        .unwrap();

        let mut deleted_urls = Vec::new();
        let (deleted, remaining) = sweep_expired_pastes(&context, &endpoints, 100.0, |url, _| {
            deleted_urls.push(url.to_string());
            Ok(true)
        })
        .unwrap();

        assert_eq!(deleted, 1);
        assert_eq!(remaining, 1);
        assert_eq!(deleted_urls, vec!["https://paste.rs/expired"]);
        let entries = load_pending(&context);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].url, "https://paste.rs/future");
    }

    #[test]
    fn log_snapshot_redacts_auth_tokens() {
        let (_home, context, _loaded) = temp_context();
        fs::create_dir_all(context.hermes_home().join("logs")).unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            r#"{"providers":{"openai-codex":{"tokens":{"access_token":"secret-token-value"}}}}"#,
        )
        .unwrap();
        write_log(
            &context.hermes_home().join("logs/agent.log"),
            "Authorization: Bearer secret-token-value\n",
        );

        let snapshots = capture_default_log_snapshots(&context, 20, true).unwrap();
        assert!(snapshots.agent.tail_text.contains("[REDACTED]"));
        assert!(!snapshots.agent.tail_text.contains("secret-token-value"));
        assert!(
            snapshots
                .agent
                .full_text
                .as_deref()
                .unwrap_or_default()
                .contains("[REDACTED]")
        );
    }

    #[test]
    fn upload_falls_back_to_dpaste_when_paste_rs_fails() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut requests = Vec::new();
            for status in [500_u16, 200_u16] {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                let mut reader = BufReader::new(request.as_bytes());
                let mut first_line = String::new();
                reader.read_line(&mut first_line).unwrap();
                requests.push(first_line.trim().to_string());
                let body = if status == 500 {
                    "server error"
                } else {
                    "http://127.0.0.1/fallback"
                };
                let response = format!(
                    "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    if status == 200 { "OK" } else { "ERR" },
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
            requests
        });

        let client = build_http_client().unwrap();
        let endpoints = PasteEndpoints {
            paste_rs_url: format!("http://{addr}/paste"),
            dpaste_url: format!("http://{addr}/dpaste"),
        };
        let url = upload_to_paste_services(&client, "hello", 7, &endpoints).unwrap();
        let requests = handle.join().unwrap();
        assert_eq!(url, "http://127.0.0.1/fallback");
        assert!(requests[0].starts_with("POST /paste"));
        assert!(requests[1].starts_with("POST /dpaste"));
    }

    #[test]
    fn delete_paste_rejects_non_paste_rs_urls() {
        let client = build_http_client().unwrap();
        let endpoints = PasteEndpoints::default();
        let error = delete_paste(&client, "https://dpaste.com/abc", &endpoints)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("only paste.rs URLs are supported"));
    }
}
