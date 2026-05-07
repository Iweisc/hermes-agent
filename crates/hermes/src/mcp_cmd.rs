use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
#[cfg(not(windows))]
use std::os::fd::AsRawFd;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use regex::Regex;
use serde_json::{Value as JsonValue, json};
use serde_yaml::{Mapping, Value};

use crate::compat_cmd::CompatArgs;
use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::python_bridge::{project_root, resolve_repo_python};

#[cfg(test)]
use clap::Parser;

#[derive(Subcommand, Debug)]
pub enum McpCommand {
    #[command(alias = "ls")]
    List,
    #[command(alias = "rm")]
    Remove(RemoveArgs),
    Serve(CompatArgs),
    Add(AddArgs),
    Test(TestArgs),
    #[command(alias = "config")]
    Configure(ConfigureArgs),
    Login(LoginArgs),
}

#[derive(Args, Debug, Clone)]
pub struct RemoveArgs {
    pub name: String,
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct AddArgs {
    pub name: String,
    #[arg(long, conflicts_with = "command")]
    pub url: Option<String>,
    #[arg(long, conflicts_with = "url")]
    pub command: Option<String>,
    #[arg(long = "args", num_args = 1..)]
    pub command_args: Vec<String>,
    #[arg(long, value_parser = ["oauth", "header"])]
    pub auth: Option<String>,
    #[arg(long)]
    pub preset: Option<String>,
    #[arg(long = "env", num_args = 1..)]
    pub env: Vec<String>,
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct TestArgs {
    pub name: String,
}

#[derive(Args, Debug, Clone)]
pub struct ConfigureArgs {
    pub name: String,
}

#[derive(Args, Debug, Clone)]
pub struct LoginArgs {
    pub name: String,
}

#[derive(Debug, Clone)]
struct McpServerEntry {
    config: Mapping,
}

pub fn print_mcp(
    context: &HermesContext,
    command: Option<McpCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => {
            print_list(context)?;
            print_help_summary();
            Ok(())
        }
        Some(McpCommand::List) => print_list(context),
        Some(McpCommand::Remove(args)) => remove_server(context, args),
        Some(McpCommand::Add(args)) => add_server(context, args),
        Some(McpCommand::Serve(args)) => bridge_mcp("serve", &args.args),
        Some(McpCommand::Test(args)) => test_server(context, args),
        Some(McpCommand::Configure(args)) => configure_server(context, args),
        Some(McpCommand::Login(args)) => login_server(context, args),
    }
}

const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-03-26";
const DEFAULT_MCP_CONNECT_TIMEOUT_SECS: u64 = 30;

fn test_server(context: &HermesContext, args: TestArgs) -> Result<(), Box<dyn Error>> {
    let name = normalize_server_lookup_name(&args.name)?;
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let servers = collect_mcp_servers(&root);
    let Some(entry) = servers.get(name) else {
        let suffix = if servers.is_empty() {
            String::new()
        } else {
            format!(
                " Available: {}",
                servers.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        };
        return Err(format!("Server '{name}' not found in config.{suffix}").into());
    };

    println!();
    println!("  Testing '{name}'...");
    if let Some(url) = config_string(&entry.config, "url") {
        println!("  Transport: HTTP -> {url}");
    } else if let Some(command) = config_string(&entry.config, "command") {
        println!("  Transport: stdio -> {command}");
    } else {
        return Err(format!("Server '{name}' is missing transport configuration").into());
    }
    print_auth_info(&entry.config)?;

    if config_string(&entry.config, "auth")
        .map(|value| value.eq_ignore_ascii_case("oauth"))
        .unwrap_or(false)
    {
        println!("  OAuth-configured server detected; using compatibility probe.");
        return bridge_mcp("test", &[name.to_string()]);
    }

    let start = Instant::now();
    let tools = probe_server_tools(name, &entry.config)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("  Connected ({elapsed_ms:.0}ms)");
    println!("  Tools discovered: {}", tools.len());
    if !tools.is_empty() {
        println!();
        for (tool_name, description) in tools {
            let short = truncate_description(&description, 55);
            println!("    {:36} {}", tool_name, short);
        }
    }
    println!();
    Ok(())
}

fn configure_server(context: &HermesContext, args: ConfigureArgs) -> Result<(), Box<dyn Error>> {
    if !stdin_is_terminal() {
        return Err("'hermes mcp configure' requires an interactive terminal.".into());
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    configure_server_io(context, &args, stdin.lock(), stdout.lock())
}

fn login_server(context: &HermesContext, args: LoginArgs) -> Result<(), Box<dyn Error>> {
    let name = normalize_server_lookup_name(&args.name)?;
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let servers = collect_mcp_servers(&root);
    let Some(entry) = servers.get(name) else {
        let suffix = if servers.is_empty() {
            String::new()
        } else {
            format!(
                " Available servers: {}",
                servers.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        };
        return Err(format!("Server '{name}' not found in config.{suffix}").into());
    };

    if config_string(&entry.config, "url").is_none() {
        return Err(format!("Server '{name}' has no URL — not an OAuth-capable server").into());
    }
    if !config_string(&entry.config, "auth")
        .map(|value| value.eq_ignore_ascii_case("oauth"))
        .unwrap_or(false)
    {
        return Err(format!(
            "Server '{name}' is not configured for OAuth (auth={})\nUse `hermes mcp remove` + `hermes mcp add` to reconfigure auth.",
            config_string(&entry.config, "auth").unwrap_or_else(|| String::from("none"))
        )
        .into());
    }

    let cleared = cleanup_oauth_tokens(context, name)?;
    println!();
    if cleared {
        println!("  Cleared cached OAuth tokens for '{name}'.");
    } else {
        println!("  No cached OAuth tokens found for '{name}'.");
    }
    println!("  Starting OAuth flow for '{name}'...");
    bridge_mcp("test", &[name.to_string()])
}

fn configure_server_io<R: BufRead, W: Write>(
    context: &HermesContext,
    args: &ConfigureArgs,
    mut input: R,
    mut output: W,
) -> Result<(), Box<dyn Error>> {
    let name = normalize_server_lookup_name(&args.name)?;
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let servers = collect_mcp_servers(&root);
    let Some(entry) = servers.get(name) else {
        let suffix = if servers.is_empty() {
            String::new()
        } else {
            format!(
                " Available: {}",
                servers.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        };
        return Err(format!("Server '{name}' not found in config.{suffix}").into());
    };

    if config_string(&entry.config, "auth")
        .map(|value| value.eq_ignore_ascii_case("oauth"))
        .unwrap_or(false)
    {
        return bridge_mcp("configure", &[name.to_string()]);
    }

    writeln!(output)?;
    writeln!(output, "  Connecting to '{name}' to discover tools...")?;
    let all_tools = probe_server_tools(name, &entry.config)?;
    if all_tools.is_empty() {
        writeln!(output, "  Server reports no tools.")?;
        return Ok(());
    }

    let preselected = preselected_tool_indices(&entry.config, &all_tools);
    writeln!(
        output,
        "  Currently {}/{} tools enabled for '{name}'.",
        preselected.len(),
        all_tools.len()
    )?;

    let labels = all_tools
        .iter()
        .map(|(tool_name, description)| format!("{tool_name} — {description}"))
        .collect::<Vec<_>>();
    let chosen =
        prompt_enabled_indices_io("Tools", &labels, &preselected, &mut input, &mut output)?;
    if chosen == preselected {
        writeln!(output, "  No changes made.")?;
        return Ok(());
    }

    let mut updated = entry.config.clone();
    if chosen.len() == all_tools.len() {
        updated.remove(yaml_key("tools"));
    } else {
        let chosen_names = all_tools
            .iter()
            .enumerate()
            .filter_map(|(index, (tool_name, _))| {
                chosen.contains(&index).then_some(tool_name.clone())
            })
            .collect::<Vec<_>>();
        let mut tools = updated
            .get(yaml_key("tools"))
            .and_then(Value::as_mapping)
            .cloned()
            .unwrap_or_else(Mapping::new);
        tools.insert(
            yaml_key("include"),
            Value::Sequence(chosen_names.into_iter().map(yaml_string).collect()),
        );
        tools.remove(yaml_key("exclude"));
        updated.insert(yaml_key("tools"), Value::Mapping(tools));
    }

    let _ = remove_mcp_server(&mut root, name);
    upsert_top_level_mcp_server(&mut root, name, Value::Mapping(updated));
    write_yaml_mapping(&context.config_path(), &root)?;

    writeln!(
        output,
        "  Updated config: {}/{} tools enabled",
        chosen.len(),
        all_tools.len()
    )?;
    writeln!(output, "  Start a new session for changes to take effect.")?;
    Ok(())
}

fn probe_server_tools(
    name: &str,
    config: &Mapping,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    if config.contains_key(yaml_key("url")) {
        return probe_http_server(name, config);
    }
    if config.contains_key(yaml_key("command")) {
        return probe_stdio_server(name, config);
    }
    Err(format!("Server '{name}' has no url or command transport configured").into())
}

fn probe_stdio_server(
    name: &str,
    config: &Mapping,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let command = config_string(config, "command")
        .ok_or_else(|| format!("Server '{name}' has no command configured"))?;
    let command_args = config_sequence_strings(config, "args");
    let timeout = config_timeout(config);
    let user_env = config_mapping_strings(config, "env");
    let safe_env = build_safe_env(&user_env);
    let resolved_command = resolve_stdio_command(&command, &safe_env);

    let mut child = Command::new(&resolved_command)
        .args(&command_args)
        .env_clear()
        .envs(&safe_env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start MCP stdio server '{name}': {error}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("MCP stdio server '{name}' did not expose stdout"))?;
    let stderr = child.stderr.take();
    let lines = spawn_jsonrpc_line_reader(stdout);
    let stderr_handle = stderr.map(spawn_stderr_reader);

    let result = (|| -> Result<Vec<(String, String)>, Box<dyn Error>> {
        write_jsonrpc_line(
            child.stdin.as_mut(),
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "hermes-rs-cli",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                }
            }),
        )?;
        let init = wait_for_jsonrpc_response(&lines, 1, timeout)?;
        let protocol_version = extract_protocol_version(&init)
            .unwrap_or_else(|| DEFAULT_MCP_PROTOCOL_VERSION.to_string());

        write_jsonrpc_line(
            child.stdin.as_mut(),
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {
                    "protocolVersion": protocol_version,
                }
            }),
        )?;

        write_jsonrpc_line(
            child.stdin.as_mut(),
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }),
        )?;
        let tools = wait_for_jsonrpc_response(&lines, 2, timeout)?;
        extract_tools_from_payload(&tools)
    })();

    if let Some(stdin) = child.stdin.take() {
        drop(stdin);
    }
    let _ = child.kill();
    let _ = child.wait();

    let stderr_output = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    match result {
        Ok(tools) => Ok(tools),
        Err(error) => {
            let stderr_tail = stderr_output.trim();
            if stderr_tail.is_empty() {
                Err(error)
            } else {
                Err(format!(
                    "{error}; stderr: {}",
                    truncate_description(stderr_tail, 160)
                )
                .into())
            }
        }
    }
}

fn probe_http_server(
    name: &str,
    config: &Mapping,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let url = config_string(config, "url")
        .ok_or_else(|| format!("Server '{name}' has no URL configured"))?;
    let timeout = config_timeout(config);
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()?;

    let default_headers = build_http_headers(config, None, Some(DEFAULT_MCP_PROTOCOL_VERSION))?;
    let init = send_http_jsonrpc(
        &client,
        &url,
        &default_headers,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "hermes-rs-cli",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }
        }),
        false,
    )?;
    let protocol_version = extract_protocol_version(&init.body)
        .unwrap_or_else(|| DEFAULT_MCP_PROTOCOL_VERSION.to_string());

    let initialized_headers =
        build_http_headers(config, init.session_id.as_deref(), Some(&protocol_version))?;
    let _ = send_http_jsonrpc(
        &client,
        &url,
        &initialized_headers,
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {
                "protocolVersion": protocol_version,
            }
        }),
        true,
    )?;

    let list_headers =
        build_http_headers(config, init.session_id.as_deref(), Some(&protocol_version))?;
    let tools = send_http_jsonrpc(
        &client,
        &url,
        &list_headers,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
        false,
    )?;
    extract_tools_from_payload(&tools.body)
}

struct HttpProbeResponse {
    body: JsonValue,
    session_id: Option<String>,
}

fn send_http_jsonrpc(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: &JsonValue,
    allow_empty: bool,
) -> Result<HttpProbeResponse, Box<dyn Error>> {
    let mut request = client.post(url).json(payload);
    for (key, value) in headers {
        request = request.header(key, value);
    }
    let mut response = request.send()?;
    let status = response.status();
    if allow_empty && status == reqwest::StatusCode::ACCEPTED {
        return Ok(HttpProbeResponse {
            body: JsonValue::Null,
            session_id: response
                .headers()
                .get("mcp-session-id")
                .and_then(|value| value.to_str().ok())
                .map(|value| value.to_string()),
        });
    }
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(format!(
            "HTTP MCP request failed ({}): {}",
            status,
            truncate_description(body.trim(), 160)
        )
        .into());
    }

    let session_id = response
        .headers()
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let body = if content_type.starts_with("text/event-stream") {
        read_sse_jsonrpc_message(&mut response)?
    } else {
        response.json()?
    };
    Ok(HttpProbeResponse { body, session_id })
}

fn build_http_headers(
    config: &Mapping,
    session_id: Option<&str>,
    protocol_version: Option<&str>,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut headers = BTreeMap::new();
    headers.insert(
        String::from("accept"),
        String::from("application/json, text/event-stream"),
    );
    headers.insert(
        String::from("content-type"),
        String::from("application/json"),
    );
    if let Some(version) = protocol_version {
        headers.insert(String::from("mcp-protocol-version"), version.to_string());
    }
    if let Some(value) = session_id.filter(|value| !value.trim().is_empty()) {
        headers.insert(String::from("mcp-session-id"), value.to_string());
    }

    if let Some(mapping) = config.get(yaml_key("headers")).and_then(Value::as_mapping) {
        for (key, value) in mapping {
            let Some(header_name) = key.as_str().map(str::trim).filter(|text| !text.is_empty())
            else {
                continue;
            };
            let Some(header_value) = value.as_str() else {
                return Err(format!("header '{header_name}' must be a string").into());
            };
            headers.insert(
                header_name.to_string(),
                interpolate_env_placeholders(header_value),
            );
        }
    }

    if !headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("mcp-protocol-version"))
    {
        headers.insert(
            String::from("mcp-protocol-version"),
            DEFAULT_MCP_PROTOCOL_VERSION.to_string(),
        );
    }
    Ok(headers)
}

fn print_auth_info(config: &Mapping) -> Result<(), Box<dyn Error>> {
    let auth_type = config_string(config, "auth").unwrap_or_default();
    if auth_type.eq_ignore_ascii_case("oauth") {
        println!("  Auth: OAuth 2.1 PKCE");
        return Ok(());
    }

    let Some(headers) = config.get(yaml_key("headers")).and_then(Value::as_mapping) else {
        println!("  Auth: none");
        return Ok(());
    };

    let mut printed = false;
    for (key, value) in headers {
        let Some(header_name) = key.as_str().map(str::trim).filter(|text| !text.is_empty()) else {
            continue;
        };
        let Some(raw_value) = value.as_str() else {
            return Err(format!("header '{header_name}' must be a string").into());
        };
        let resolved = interpolate_env_placeholders(raw_value);
        if header_name.eq_ignore_ascii_case("authorization")
            || header_name.to_ascii_lowercase().contains("key")
            || header_name.to_ascii_lowercase().contains("token")
        {
            println!("    {}: {}", header_name, mask_secret(&resolved));
            printed = true;
        }
    }

    if !printed {
        println!("  Auth: none");
    }
    Ok(())
}

fn mask_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() <= 8 {
        return String::from("***");
    }
    format!("{}***{}", &trimmed[..4], &trimmed[trimmed.len() - 4..])
}

fn interpolate_env_placeholders(value: &str) -> String {
    let pattern = Regex::new(r"\$\{(\w+)\}").expect("valid env interpolation regex");
    pattern
        .replace_all(value, |captures: &regex::Captures<'_>| {
            std::env::var(&captures[1]).unwrap_or_default()
        })
        .into_owned()
}

fn build_safe_env(user_env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for (key, value) in std::env::vars() {
        if matches!(
            key.as_str(),
            "PATH" | "HOME" | "USER" | "LANG" | "LC_ALL" | "TERM" | "SHELL" | "TMPDIR"
        ) || key.starts_with("XDG_")
        {
            env_map.insert(key, value);
        }
    }
    env_map.extend(user_env.clone());
    env_map
}

fn resolve_stdio_command(command: &str, env_map: &BTreeMap<String, String>) -> String {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains(std::path::MAIN_SEPARATOR) {
        return trimmed.to_string();
    }
    if let Some(path) = find_executable_on_path(trimmed, env_map.get("PATH").map(String::as_str)) {
        return path;
    }
    if matches!(trimmed, "npx" | "npm" | "node") {
        let mut candidates = Vec::new();
        if let Some(home) = std::env::var_os("HERMES_HOME") {
            candidates.push(
                std::path::PathBuf::from(home)
                    .join("node")
                    .join("bin")
                    .join(trimmed),
            );
        }
        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(
                std::path::PathBuf::from(home)
                    .join(".local")
                    .join("bin")
                    .join(trimmed),
            );
        }
        for candidate in candidates {
            if is_executable(&candidate) {
                return candidate.display().to_string();
            }
        }
    }
    trimmed.to_string()
}

fn find_executable_on_path(command: &str, path_value: Option<&str>) -> Option<String> {
    let path_value = path_value?;
    for dir in std::env::split_paths(path_value) {
        let candidate = dir.join(command);
        if is_executable(&candidate) {
            return Some(candidate.display().to_string());
        }
    }
    None
}

fn is_executable(path: &std::path::Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        return fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn spawn_jsonrpc_line_reader(
    stdout: std::process::ChildStdout,
) -> mpsc::Receiver<Result<String, String>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let message = line.map_err(|error| error.to_string());
            if sender.send(message).is_err() {
                break;
            }
        }
    });
    receiver
}

fn spawn_stderr_reader(stderr: std::process::ChildStderr) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut output = String::new();
        let _ = reader.read_to_string(&mut output);
        output
    })
}

fn write_jsonrpc_line(
    stdin: Option<&mut std::process::ChildStdin>,
    payload: &JsonValue,
) -> Result<(), Box<dyn Error>> {
    let stdin = stdin.ok_or("MCP stdio server stdin is unavailable")?;
    serde_json::to_writer(&mut *stdin, payload)?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    Ok(())
}

fn wait_for_jsonrpc_response(
    receiver: &mpsc::Receiver<Result<String, String>>,
    expected_id: i64,
    timeout: Duration,
) -> Result<JsonValue, Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!("timed out waiting for MCP response id {expected_id}").into());
        }
        let remaining = deadline.saturating_duration_since(now);
        match receiver.recv_timeout(remaining) {
            Ok(Ok(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let payload: JsonValue = serde_json::from_str(trimmed)
                    .map_err(|error| format!("invalid JSON-RPC message: {error}"))?;
                if payload
                    .get("id")
                    .and_then(JsonValue::as_i64)
                    .is_some_and(|value| value == expected_id)
                {
                    if let Some(error) = payload.get("error") {
                        let message = error
                            .get("message")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("unknown MCP error");
                        return Err(message.to_string().into());
                    }
                    return Ok(payload);
                }
            }
            Ok(Err(error)) => return Err(error.into()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(format!("timed out waiting for MCP response id {expected_id}").into());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(
                    format!("MCP response stream closed before response id {expected_id}").into(),
                );
            }
        }
    }
}

fn extract_protocol_version(payload: &JsonValue) -> Option<String> {
    payload
        .get("result")
        .and_then(JsonValue::as_object)
        .and_then(|result| result.get("protocolVersion"))
        .and_then(JsonValue::as_str)
        .map(|value| value.to_string())
}

fn extract_tools_from_payload(
    payload: &JsonValue,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let Some(result) = payload.get("result").and_then(JsonValue::as_object) else {
        return Err("MCP response is missing a result object".into());
    };
    let Some(tools) = result.get("tools").and_then(JsonValue::as_array) else {
        return Err("MCP response is missing result.tools".into());
    };
    Ok(tools
        .iter()
        .filter_map(|tool| {
            let object = tool.as_object()?;
            let name = object.get("name")?.as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            let description = object
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            Some((name.to_string(), description))
        })
        .collect())
}

fn read_sse_jsonrpc_message(
    response: &mut reqwest::blocking::Response,
) -> Result<JsonValue, Box<dyn Error>> {
    let mut reader = BufReader::new(response);
    let mut event_name = String::new();
    let mut data_lines = Vec::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            if !data_lines.is_empty() && (event_name.is_empty() || event_name == "message") {
                return Ok(serde_json::from_str(&data_lines.join("\n"))?);
            }
            event_name.clear();
            data_lines.clear();
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("event:") {
            event_name = value.trim().to_string();
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_string());
        }
    }
    Err("HTTP MCP SSE response did not contain a JSON-RPC message".into())
}

fn config_string(config: &Mapping, key: &str) -> Option<String> {
    config
        .get(yaml_key(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn config_sequence_strings(config: &Mapping, key: &str) -> Vec<String> {
    config
        .get(yaml_key(key))
        .and_then(Value::as_sequence)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn config_mapping_strings(config: &Mapping, key: &str) -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    let Some(mapping) = config.get(yaml_key(key)).and_then(Value::as_mapping) else {
        return values;
    };
    for (entry_key, entry_value) in mapping {
        let Some(name) = entry_key
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let Some(value) = entry_value.as_str() else {
            continue;
        };
        values.insert(name.to_string(), value.to_string());
    }
    values
}

fn config_timeout(config: &Mapping) -> Duration {
    let seconds = config
        .get(yaml_key("connect_timeout"))
        .and_then(value_as_u64)
        .unwrap_or(DEFAULT_MCP_CONNECT_TIMEOUT_SECS);
    Duration::from_secs(seconds.max(1))
}

fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn truncate_description(value: &str, max_len: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_len {
        return value.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let prefix = chars[..max_len - 3].iter().collect::<String>();
    format!("{prefix}...")
}

fn normalize_server_lookup_name(raw: &str) -> Result<&str, Box<dyn Error>> {
    let name = raw.trim();
    if name.is_empty() {
        return Err("server name cannot be empty".into());
    }
    Ok(name)
}

fn preselected_tool_indices(
    config: &Mapping,
    all_tools: &[(String, String)],
) -> std::collections::BTreeSet<usize> {
    let tool_names = all_tools
        .iter()
        .map(|(tool_name, _)| tool_name.as_str())
        .collect::<Vec<_>>();
    let tools_cfg = config.get(yaml_key("tools")).and_then(Value::as_mapping);
    let include = tools_cfg
        .and_then(|mapping| mapping.get(yaml_key("include")))
        .and_then(Value::as_sequence);
    let exclude = tools_cfg
        .and_then(|mapping| mapping.get(yaml_key("exclude")))
        .and_then(Value::as_sequence);
    if let Some(include) = include {
        let include_set = include
            .iter()
            .filter_map(Value::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        return tool_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| include_set.contains(name).then_some(index))
            .collect();
    }
    if let Some(exclude) = exclude {
        let exclude_set = exclude
            .iter()
            .filter_map(Value::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        return tool_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| (!exclude_set.contains(name)).then_some(index))
            .collect();
    }
    (0..all_tools.len()).collect()
}

fn prompt_enabled_indices_io<R: BufRead, W: Write>(
    title: &str,
    labels: &[String],
    preselected: &std::collections::BTreeSet<usize>,
    input: &mut R,
    output: &mut W,
) -> Result<std::collections::BTreeSet<usize>, Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "{title}:")?;
    for (index, label) in labels.iter().enumerate() {
        let marker = if preselected.contains(&index) {
            'x'
        } else {
            ' '
        };
        writeln!(output, "  {:>2}. [{}] {}", index + 1, marker, label)?;
    }
    writeln!(
        output,
        "Enter enabled numbers like 1,3-5, 'all', 'none', or press Enter to keep current."
    )?;
    write!(output, "Enabled [keep]: ")?;
    output.flush()?;

    let mut line = String::new();
    input.read_line(&mut line)?;
    let raw = line.trim();
    if raw.is_empty() {
        return Ok(preselected.clone());
    }
    parse_enabled_indices(raw, labels.len())
}

fn parse_enabled_indices(
    raw: &str,
    total: usize,
) -> Result<std::collections::BTreeSet<usize>, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok((0..total).collect());
    }
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(std::collections::BTreeSet::new());
    }

    let mut selected = std::collections::BTreeSet::new();
    for segment in trimmed.split(',') {
        let piece = segment.trim();
        if piece.is_empty() {
            return Err("selection contains an empty item".into());
        }
        if let Some((start_raw, end_raw)) = piece.split_once('-') {
            let start = parse_selection_index(start_raw, total)?;
            let end = parse_selection_index(end_raw, total)?;
            if start > end {
                return Err("selection range must be ascending".into());
            }
            for index in start..=end {
                selected.insert(index);
            }
        } else {
            selected.insert(parse_selection_index(piece, total)?);
        }
    }
    Ok(selected)
}

fn parse_selection_index(raw: &str, total: usize) -> Result<usize, Box<dyn Error>> {
    let selection = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| "selection must use numeric entries")?;
    if selection == 0 || selection > total {
        return Err("selection is out of range".into());
    }
    Ok(selection - 1)
}

fn stdin_is_terminal() -> bool {
    #[cfg(windows)]
    {
        true
    }
    #[cfg(not(windows))]
    {
        unsafe { libc::isatty(io::stdin().as_raw_fd()) == 1 }
    }
}

fn bridge_mcp(subcommand: &str, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_MCP_PYTHON"))
        .ok_or("could not find a Python interpreter for mcp")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_MCP_SUBCOMMAND", subcommand)
        .arg("-c")
        .arg(MCP_BOOTSTRAP)
        .args(passthrough);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("mcp", status).into())
}

const MCP_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "import sys\n",
    "from hermes_cli.mcp_config import mcp_command\n",
    "subcommand = (os.environ.get('HERMES_MCP_SUBCOMMAND') or '').strip()\n",
    "parser = argparse.ArgumentParser(prog=f'hermes mcp {subcommand}')\n",
    "parser.set_defaults(mcp_action=subcommand)\n",
    "if subcommand == 'serve':\n",
    "    parser.add_argument('-v', '--verbose', action='store_true')\n",
    "    parser.add_argument('--accept-hooks', action='store_true', default=False)\n",
    "elif subcommand == 'add':\n",
    "    parser.add_argument('name')\n",
    "    parser.add_argument('--url')\n",
    "    parser.add_argument('--command')\n",
    "    parser.add_argument('--args', nargs='*', default=[])\n",
    "    parser.add_argument('--auth', choices=['oauth', 'header'])\n",
    "    parser.add_argument('--preset')\n",
    "    parser.add_argument('--env', nargs='*', default=[])\n",
    "elif subcommand == 'test':\n",
    "    parser.add_argument('name')\n",
    "elif subcommand == 'configure':\n",
    "    parser.add_argument('name')\n",
    "elif subcommand == 'login':\n",
    "    parser.add_argument('name')\n",
    "else:\n",
    "    raise SystemExit(f'unsupported mcp subcommand: {subcommand}')\n",
    "mcp_command(parser.parse_args(sys.argv[1:]))\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn print_list(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let servers = collect_mcp_servers(&root);

    if servers.is_empty() {
        println!("No MCP servers configured.");
        println!();
        println!("Add one with:");
        println!("  hermes mcp add <name> --url <endpoint>");
        println!("  hermes mcp add <name> --command <cmd> --args <args...>");
        println!();
        return Ok(());
    }

    println!("MCP Servers:");
    println!();
    println!("{:<16} {:<30} {:<12} Status", "Name", "Transport", "Tools");
    println!(
        "{:<16} {:<30} {:<12} ------",
        "----------------", "------------------------------", "------------"
    );
    for (name, entry) in servers {
        println!(
            "{:<16} {:<30} {:<12} {}",
            name,
            describe_transport(&entry.config),
            describe_tools(&entry.config),
            describe_enabled(&entry.config)
        );
    }
    println!();
    Ok(())
}

fn print_help_summary() {
    println!("Commands:");
    println!("  hermes mcp serve                              Run as MCP server");
    println!("  hermes mcp add <name> --url <endpoint>        Add an MCP server");
    println!("  hermes mcp add <name> --command <cmd>         Add a stdio server");
    println!("  hermes mcp remove <name>                      Remove a server");
    println!("  hermes mcp list                               List servers");
    println!("  hermes mcp test <name>                        Test connection");
    println!("  hermes mcp configure <name>                   Toggle tools");
    println!("  hermes mcp login <name>                       Re-authenticate OAuth");
    println!();
}

fn add_server(context: &HermesContext, args: AddArgs) -> Result<(), Box<dyn Error>> {
    let name = validate_server_name(&args.name)?.to_string();
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let existing = collect_mcp_servers(&root);
    if existing.contains_key(&name)
        && !args.yes
        && !confirm_prompt(&format!(
            "Server '{name}' already exists. Overwrite? [y/N] "
        ))?
    {
        println!("Cancelled.");
        return Ok(());
    }

    let mut url = args.url.as_deref().map(validate_http_url).transpose()?;
    let mut command = args
        .command
        .as_deref()
        .map(validate_command_value)
        .transpose()?;
    let mut command_args = args.command_args.clone();
    apply_mcp_preset(
        args.preset.as_deref(),
        &mut url,
        &mut command,
        &mut command_args,
    )?;

    if url.is_none() && command.is_none() {
        return Err("Must specify --url <endpoint>, --command <cmd>, or --preset <name>".into());
    }

    let env_assignments = parse_env_assignments(&args.env)?;
    if url.is_some() && !env_assignments.is_empty() {
        return Err(
            "--env is only supported for stdio MCP servers (--command or stdio presets)".into(),
        );
    }

    let mut server = Mapping::new();
    if let Some(url) = url {
        server.insert(yaml_key("url"), yaml_string(url));
    } else if let Some(command) = command {
        server.insert(yaml_key("command"), yaml_string(command));
        if !command_args.is_empty() {
            server.insert(
                yaml_key("args"),
                Value::Sequence(command_args.into_iter().map(yaml_string).collect()),
            );
        }
        if !env_assignments.is_empty() {
            server.insert(yaml_key("env"), env_mapping(&env_assignments));
        }
    }

    if let Some(auth) = args.auth.as_deref() {
        match auth {
            "oauth" => {
                if !server.contains_key(yaml_key("url")) {
                    return Err("--auth oauth is only supported for HTTP MCP servers".into());
                }
                server.insert(yaml_key("auth"), yaml_string("oauth"));
            }
            "header" => {
                if !server.contains_key(yaml_key("url")) {
                    return Err("--auth header is only supported for HTTP MCP servers".into());
                }
                let env_key = env_key_for_server(&name);
                let api_key = match std::env::var(&env_key)
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                {
                    Some(value) => value,
                    None => {
                        let value = prompt_value("API key / Bearer token: ")?;
                        if value.trim().is_empty() {
                            return Err("API key / Bearer token cannot be empty".into());
                        }
                        save_env_value(context.hermes_home().join(".env"), &env_key, &value)?;
                        println!(
                            "Saved {} to {}",
                            env_key,
                            context.hermes_home().join(".env").display()
                        );
                        value
                    }
                };
                let _ = api_key;
                let mut headers = Mapping::new();
                headers.insert(
                    yaml_string("Authorization"),
                    yaml_string(format!("Bearer ${{{env_key}}}")),
                );
                server.insert(yaml_key("headers"), Value::Mapping(headers));
            }
            _ => return Err(format!("unsupported auth type: {auth}").into()),
        }
    }

    server.insert(yaml_key("enabled"), Value::Bool(true));
    upsert_top_level_mcp_server(&mut root, &name, Value::Mapping(server));
    write_yaml_mapping(&context.config_path(), &root)?;

    println!("Saved '{name}' to {}", context.config_path().display());
    println!("Use `hermes mcp test {name}` to verify the connection.");
    println!(
        "Use `hermes mcp configure {name}` after discovery if you want to narrow enabled tools."
    );
    Ok(())
}

fn remove_server(context: &HermesContext, args: RemoveArgs) -> Result<(), Box<dyn Error>> {
    let name = validate_server_name(&args.name)?;
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let available = collect_mcp_servers(&root).into_keys().collect::<Vec<_>>();
    if !available.iter().any(|candidate| candidate == name) {
        let suffix = if available.is_empty() {
            String::from(" No MCP servers are configured.")
        } else {
            format!(" Available servers: {}", available.join(", "))
        };
        return Err(format!("MCP server '{name}' not found.{suffix}").into());
    }

    if !args.yes && !confirm_prompt(&format!("Remove server '{name}'? [Y/n] "))? {
        println!("Cancelled.");
        return Ok(());
    }

    let removed = remove_mcp_server(&mut root, name);
    if !removed {
        return Err(format!("MCP server '{name}' could not be removed").into());
    }
    write_yaml_mapping(&context.config_path(), &root)?;
    println!("Removed '{name}' from config");

    if cleanup_oauth_tokens(context, name)? {
        println!("Cleaned up OAuth tokens");
    }
    Ok(())
}

fn collect_mcp_servers(root: &Mapping) -> BTreeMap<String, McpServerEntry> {
    let mut merged = BTreeMap::new();
    for servers in [top_level_servers(root), nested_servers(root)] {
        let Some(servers) = servers else {
            continue;
        };
        for (key, value) in servers {
            let Some(name) = key.as_str() else {
                continue;
            };
            let Some(config) = value.as_mapping() else {
                continue;
            };
            merged
                .entry(name.to_string())
                .or_insert_with(|| McpServerEntry {
                    config: config.clone(),
                });
        }
    }
    merged
}

fn top_level_servers(root: &Mapping) -> Option<&Mapping> {
    root.get(yaml_key("mcp_servers"))
        .and_then(Value::as_mapping)
}

fn nested_servers(root: &Mapping) -> Option<&Mapping> {
    root.get(yaml_key("mcp"))
        .and_then(Value::as_mapping)
        .and_then(|mapping| mapping.get(yaml_key("servers")))
        .and_then(Value::as_mapping)
}

fn describe_transport(config: &Mapping) -> String {
    if let Some(url) = config
        .get(yaml_key("url"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return truncate_column(url, 28);
    }

    if let Some(command) = config
        .get(yaml_key("command"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let mut rendered = command.to_string();
        if let Some(Value::Sequence(items)) = config.get(yaml_key("args")) {
            let args = items
                .iter()
                .filter_map(Value::as_str)
                .take(2)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            if !args.is_empty() {
                rendered.push(' ');
                rendered.push_str(&args.join(" "));
            }
        }
        return truncate_column(&rendered, 28);
    }

    String::from("?")
}

fn describe_tools(config: &Mapping) -> String {
    let Some(tools) = config.get(yaml_key("tools")).and_then(Value::as_mapping) else {
        return String::from("all");
    };
    if let Some(include) = tools.get(yaml_key("include")).and_then(Value::as_sequence) {
        if !include.is_empty() {
            return format!("{} selected", include.len());
        }
    }
    if let Some(exclude) = tools.get(yaml_key("exclude")).and_then(Value::as_sequence) {
        if !exclude.is_empty() {
            return format!("-{} excluded", exclude.len());
        }
    }
    String::from("all")
}

fn describe_enabled(config: &Mapping) -> &'static str {
    if value_as_bool(config.get(yaml_key("enabled"))).unwrap_or(true) {
        "enabled"
    } else {
        "disabled"
    }
}

fn value_as_bool(value: Option<&Value>) -> Option<bool> {
    match value {
        Some(Value::Bool(value)) => Some(*value),
        Some(Value::String(value)) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn truncate_column(value: &str, max_len: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_len {
        return value.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let prefix = chars[..max_len - 3].iter().collect::<String>();
    format!("{prefix}...")
}

fn validate_server_name(raw: &str) -> Result<&str, Box<dyn Error>> {
    let name = raw.trim();
    if name.is_empty() {
        return Err("server name cannot be empty".into());
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Err("server name may only contain letters, numbers, '.', '_' and '-'".into());
    }
    Ok(name)
}

fn validate_http_url(raw: &str) -> Result<String, Box<dyn Error>> {
    let url = raw.trim();
    if url.is_empty() {
        return Err("URL cannot be empty".into());
    }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("URL must start with http:// or https://".into());
    }
    Ok(url.to_string())
}

fn validate_command_value(raw: &str) -> Result<String, Box<dyn Error>> {
    let command = raw.trim();
    if command.is_empty() {
        return Err("command cannot be empty".into());
    }
    Ok(command.to_string())
}

fn parse_env_assignments(raw_env: &[String]) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut parsed = BTreeMap::new();
    for item in raw_env {
        let text = item.trim();
        if text.is_empty() {
            continue;
        }
        let Some((key, value)) = text.split_once('=') else {
            return Err(format!("Invalid --env value '{text}' (expected KEY=VALUE)").into());
        };
        let key = key.trim();
        if !is_env_key(key) {
            return Err(format!("Invalid --env variable name '{key}'").into());
        }
        parsed.insert(key.to_string(), value.to_string());
    }
    Ok(parsed)
}

fn is_env_key(key: &str) -> bool {
    let mut chars = key.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn env_mapping(assignments: &BTreeMap<String, String>) -> Value {
    Value::Mapping(
        assignments
            .iter()
            .map(|(key, value)| (yaml_string(key), yaml_string(value)))
            .collect(),
    )
}

fn env_key_for_server(name: &str) -> String {
    format!(
        "MCP_{}_API_KEY",
        name.to_ascii_uppercase().replace('-', "_")
    )
}

fn apply_mcp_preset(
    preset_name: Option<&str>,
    url: &mut Option<String>,
    command: &mut Option<String>,
    command_args: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let Some(preset_name) = preset_name.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(());
    };
    if url.is_some() || command.is_some() {
        return Ok(());
    }
    let _ = command_args;
    Err(format!("Unknown MCP preset: {preset_name}").into())
}

fn prompt_value(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn remove_mcp_server(root: &mut Mapping, name: &str) -> bool {
    let mut removed = false;
    let server_key = yaml_key(name);

    let top_key = yaml_key("mcp_servers");
    if let Some(Value::Mapping(servers)) = root.get_mut(&top_key) {
        removed |= servers.remove(&server_key).is_some();
        if servers.is_empty() {
            root.remove(&top_key);
        }
    }

    let mcp_key = yaml_key("mcp");
    let servers_key = yaml_key("servers");
    if let Some(Value::Mapping(mcp)) = root.get_mut(&mcp_key) {
        if let Some(Value::Mapping(servers)) = mcp.get_mut(&servers_key) {
            removed |= servers.remove(&server_key).is_some();
            if servers.is_empty() {
                mcp.remove(&servers_key);
            }
        }
        if mcp.is_empty() {
            root.remove(&mcp_key);
        }
    }

    removed
}

fn upsert_top_level_mcp_server(root: &mut Mapping, name: &str, value: Value) {
    let top_key = yaml_key("mcp_servers");
    if !matches!(root.get(&top_key), Some(Value::Mapping(_))) {
        root.insert(top_key.clone(), Value::Mapping(Mapping::new()));
    }
    if let Some(Value::Mapping(servers)) = root.get_mut(&top_key) {
        servers.insert(yaml_key(name), value);
    }
}

fn cleanup_oauth_tokens(
    context: &HermesContext,
    server_name: &str,
) -> Result<bool, Box<dyn Error>> {
    let safe = safe_token_filename(server_name);
    let token_dir = context.hermes_home().join("mcp-tokens");
    let mut removed = false;

    for suffix in [".json", ".client.json"] {
        let path = token_dir.join(format!("{safe}{suffix}"));
        if path.exists() {
            fs::remove_file(path)?;
            removed = true;
        }
    }

    Ok(removed)
}

fn safe_token_filename(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim_matches('_');
    let limited = trimmed.chars().take(128).collect::<String>();
    if limited.is_empty() {
        String::from("default")
    } else {
        limited
    }
}

fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(true);
    }
    Ok(matches!(trimmed.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn yaml_key(key: &str) -> Value {
    Value::String(key.to_string())
}

fn yaml_string(value: impl Into<String>) -> Value {
    Value::String(value.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::Arc;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::TempDir;

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-mcp-{label}-{unique}"))
    }

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(test)]
    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    #[cfg(test)]
    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    #[cfg(unix)]
    fn write_stdio_test_server(path: &std::path::Path) {
        fs::write(
            path,
            "#!/bin/sh\n\
read line1\n\
printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"2025-03-26\",\"serverInfo\":{\"name\":\"demo\",\"version\":\"1.0\"},\"capabilities\":{}}}'\n\
read line2\n\
read line3\n\
printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"alpha\",\"description\":\"First tool\"},{\"name\":\"beta\",\"description\":\"Second tool\"}]}}'\n",
        )
        .unwrap();
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

    fn read_http_request(
        stream: &mut std::net::TcpStream,
    ) -> (String, BTreeMap<String, String>, String) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let mut headers = BTreeMap::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" || line == "\n" || line.is_empty() {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                headers.insert(
                    key.trim().to_ascii_lowercase(),
                    value.trim().trim_end_matches('\r').to_string(),
                );
            }
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).unwrap();
        (
            request_line.trim().to_string(),
            headers,
            String::from_utf8(body).unwrap(),
        )
    }

    fn spawn_http_test_server(
        requests: Arc<Mutex<Vec<String>>>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for index in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let (request_line, headers, body) = read_http_request(&mut stream);
                requests
                    .lock()
                    .unwrap()
                    .push(format!("{request_line}\n{:?}\n{body}", headers));
                match index {
                    0 => {
                        assert!(body.contains("\"method\":\"initialize\""));
                        let response_body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"2025-03-26\",\"serverInfo\":{\"name\":\"demo\",\"version\":\"1.0\"},\"capabilities\":{}}}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMCP-Session-Id: session-123\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                            response_body.len(),
                            response_body
                        )
                        .unwrap();
                    }
                    1 => {
                        assert!(body.contains("\"method\":\"notifications/initialized\""));
                        write!(
                            stream,
                            "HTTP/1.1 202 Accepted\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                        )
                        .unwrap();
                    }
                    2 => {
                        assert_eq!(
                            headers.get("mcp-session-id").map(String::as_str),
                            Some("session-123")
                        );
                        assert_eq!(
                            headers.get("authorization").map(String::as_str),
                            Some("Bearer secret-token")
                        );
                        assert!(body.contains("\"method\":\"tools/list\""));
                        let response_body = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"http-tool\",\"description\":\"HTTP tool\"}]}}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                            response_body.len(),
                            response_body
                        )
                        .unwrap();
                    }
                    _ => unreachable!(),
                }
            }
        });
        (format!("http://{addr}/mcp"), handle)
    }

    #[derive(Parser, Debug)]
    struct McpHarness {
        #[command(subcommand)]
        command: McpCommand,
    }

    #[test]
    fn add_subcommand_parses_structured_args() {
        let parsed = McpHarness::try_parse_from([
            "mcp",
            "add",
            "alpha",
            "--url",
            "https://example.com/mcp",
            "--auth",
            "oauth",
            "--env",
            "API_KEY=test",
            "OTHER=value",
        ])
        .unwrap();

        match parsed.command {
            McpCommand::Add(args) => {
                assert_eq!(args.name, "alpha");
                assert_eq!(args.url.as_deref(), Some("https://example.com/mcp"));
                assert_eq!(args.auth.as_deref(), Some("oauth"));
                assert_eq!(
                    args.env,
                    vec![String::from("API_KEY=test"), String::from("OTHER=value")]
                );
            }
            _ => panic!("expected add args"),
        }
    }

    #[test]
    fn test_subcommand_parses_structured_args() {
        let parsed = McpHarness::try_parse_from(["mcp", "test", "alpha"]).unwrap();
        match parsed.command {
            McpCommand::Test(args) => assert_eq!(args.name, "alpha"),
            _ => panic!("expected test args"),
        }
    }

    #[test]
    fn configure_subcommand_parses_structured_args() {
        let parsed = McpHarness::try_parse_from(["mcp", "configure", "alpha"]).unwrap();
        match parsed.command {
            McpCommand::Configure(args) => assert_eq!(args.name, "alpha"),
            _ => panic!("expected configure args"),
        }
    }

    #[test]
    fn login_subcommand_parses_structured_args() {
        let parsed = McpHarness::try_parse_from(["mcp", "login", "alpha"]).unwrap();
        match parsed.command {
            McpCommand::Login(args) => assert_eq!(args.name, "alpha"),
            _ => panic!("expected login args"),
        }
    }

    #[test]
    fn add_http_server_writes_native_config() {
        let home = temp_path("add-http");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_mcp(
            &context,
            Some(McpCommand::Add(AddArgs {
                name: String::from("alpha"),
                url: Some(String::from("https://example.com/mcp")),
                command: None,
                command_args: Vec::new(),
                auth: None,
                preset: None,
                env: Vec::new(),
                yes: true,
            })),
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("mcp_servers:"));
        assert!(saved.contains("alpha:"));
        assert!(saved.contains("url: https://example.com/mcp"));
        assert!(saved.contains("enabled: true"));

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn add_stdio_server_writes_native_config_with_env() {
        let home = temp_path("add-stdio");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        print_mcp(
            &context,
            Some(McpCommand::Add(AddArgs {
                name: String::from("github"),
                url: None,
                command: Some(String::from("npx")),
                command_args: vec![String::from("@modelcontextprotocol/server-github")],
                auth: None,
                preset: None,
                env: vec![String::from("GITHUB_PERSONAL_ACCESS_TOKEN=test-token")],
                yes: true,
            })),
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("github:"));
        assert!(saved.contains("command: npx"));
        assert!(saved.contains("- '@modelcontextprotocol/server-github'"));
        assert!(saved.contains("GITHUB_PERSONAL_ACCESS_TOKEN: test-token"));

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn add_http_header_auth_uses_existing_env_key() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("add-http-auth");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let env_key = env_key_for_server("remote-api");
        let old = env::var_os(&env_key);
        set_env_var(&env_key, "secret-token");

        print_mcp(
            &context,
            Some(McpCommand::Add(AddArgs {
                name: String::from("remote-api"),
                url: Some(String::from("https://example.com/mcp")),
                command: None,
                command_args: Vec::new(),
                auth: Some(String::from("header")),
                preset: None,
                env: Vec::new(),
                yes: true,
            })),
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("remote-api:"));
        assert!(saved.contains("Authorization: Bearer ${MCP_REMOTE_API_API_KEY}"));

        match old {
            Some(value) => set_env_var(&env_key, value),
            None => remove_env_var(&env_key),
        }
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn native_stdio_test_discovers_tools() {
        let home = temp_path("native-stdio-test");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(&home).unwrap();
        let temp = TempDir::new().unwrap();
        let script = temp.path().join("server.sh");
        write_stdio_test_server(&script);
        fs::write(
            context.config_path(),
            format!(
                "mcp_servers:\n  alpha:\n    command: {}\n",
                serde_yaml::to_string(&script.display().to_string())
                    .unwrap()
                    .trim()
            ),
        )
        .unwrap();

        print_mcp(
            &context,
            Some(McpCommand::Test(TestArgs {
                name: String::from("alpha"),
            })),
        )
        .unwrap();

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn native_http_test_resolves_env_headers_and_discovers_tools() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("native-http-test");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(&home).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (url, handle) = spawn_http_test_server(requests.clone());
        let old = env::var_os("MCP_REMOTE_API_KEY");
        set_env_var("MCP_REMOTE_API_KEY", "secret-token");
        fs::write(
            context.config_path(),
            format!(
                "mcp_servers:\n  remote:\n    url: {url}\n    headers:\n      Authorization: \"Bearer ${{MCP_REMOTE_API_KEY}}\"\n"
            ),
        )
        .unwrap();

        print_mcp(
            &context,
            Some(McpCommand::Test(TestArgs {
                name: String::from("remote"),
            })),
        )
        .unwrap();

        handle.join().unwrap();
        let logged = requests.lock().unwrap().clone();
        assert_eq!(logged.len(), 3);
        assert!(logged[0].contains("\"method\":\"initialize\""));
        assert!(logged[1].contains("\"method\":\"notifications/initialized\""));
        assert!(logged[2].contains("\"method\":\"tools/list\""));
        assert!(logged[2].contains("authorization"));

        match old {
            Some(value) => set_env_var("MCP_REMOTE_API_KEY", value),
            None => remove_env_var("MCP_REMOTE_API_KEY"),
        }
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn native_configure_updates_tool_include_list() {
        let home = temp_path("native-configure");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(&home).unwrap();
        let temp = TempDir::new().unwrap();
        let script = temp.path().join("server.sh");
        write_stdio_test_server(&script);
        fs::write(
            context.config_path(),
            format!(
                "mcp_servers:\n  alpha:\n    command: {}\n",
                serde_yaml::to_string(&script.display().to_string())
                    .unwrap()
                    .trim()
            ),
        )
        .unwrap();

        let input = std::io::Cursor::new("2\n");
        let mut output = Vec::new();
        configure_server_io(
            &context,
            &ConfigureArgs {
                name: String::from("alpha"),
            },
            input,
            &mut output,
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("tools:"));
        assert!(saved.contains("include:"));
        assert!(saved.contains("- beta"));
        assert!(!saved.contains("- alpha"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Currently 2/2 tools enabled"));
        assert!(rendered.contains("Updated config: 1/2 tools enabled"));

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn native_login_clears_tokens_and_bridges_oauth_probe() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("native-login");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("mcp-tokens")).unwrap();
        fs::write(
            context.config_path(),
            "mcp_servers:\n  alpha:\n    url: https://example.com/mcp\n    auth: oauth\n",
        )
        .unwrap();
        fs::write(home.join("mcp-tokens").join("alpha.json"), "{}").unwrap();
        fs::write(home.join("mcp-tokens").join("alpha.client.json"), "{}").unwrap();

        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  shift 2\n\
  printf 'subcommand=%s argv=%s\\n' \"$HERMES_MCP_SUBCOMMAND\" \"$*\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_MCP_PYTHON", &fake_python);

        print_mcp(
            &context,
            Some(McpCommand::Login(LoginArgs {
                name: String::from("alpha"),
            })),
        )
        .unwrap();

        assert!(!home.join("mcp-tokens").join("alpha.json").exists());
        assert!(!home.join("mcp-tokens").join("alpha.client.json").exists());
        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("subcommand=test"));
        assert!(output.contains("argv=alpha"));

        remove_env_var("HERMES_MCP_PYTHON");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn bridge_uses_python_override_and_passes_subcommand_and_args() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  shift 2\n\
  printf 'subcommand=%s argv=%s\\n' \"$HERMES_MCP_SUBCOMMAND\" \"$*\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_MCP_PYTHON", &fake_python);
        bridge_mcp("login", &[String::from("alpha")]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("subcommand=login"));
        assert!(output.contains("argv=alpha"));

        remove_env_var("HERMES_MCP_PYTHON");
    }

    #[test]
    fn collects_servers_from_both_config_shapes() {
        let parsed = serde_yaml::from_str::<Value>(
            r#"
mcp:
  servers:
    nested:
      command: npx
mcp_servers:
  alpha:
    url: https://example.com/a
"#,
        )
        .unwrap();
        let servers = collect_mcp_servers(parsed.as_mapping().unwrap());
        assert_eq!(servers.len(), 2);
        assert!(servers.contains_key("alpha"));
        assert!(servers.contains_key("nested"));
    }

    #[test]
    fn remove_cleans_config_and_token_files() {
        let home = temp_path("remove");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("mcp-tokens")).unwrap();
        fs::write(
            context.config_path(),
            r#"
mcp:
  servers:
    nested:
      command: npx
mcp_servers:
  alpha:
    url: https://example.com/a
"#,
        )
        .unwrap();
        fs::write(home.join("mcp-tokens").join("alpha.json"), "{}").unwrap();
        fs::write(home.join("mcp-tokens").join("alpha.client.json"), "{}").unwrap();

        print_mcp(
            &context,
            Some(McpCommand::Remove(RemoveArgs {
                name: String::from("alpha"),
                yes: true,
            })),
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(!saved.contains("alpha:"));
        assert!(saved.contains("nested:"));
        assert!(!home.join("mcp-tokens").join("alpha.json").exists());
        assert!(!home.join("mcp-tokens").join("alpha.client.json").exists());

        let _ = fs::remove_dir_all(home);
    }
}
