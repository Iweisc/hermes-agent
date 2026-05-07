use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
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
    Test(CompatArgs),
    #[command(alias = "config")]
    Configure(CompatArgs),
    Login(CompatArgs),
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
        Some(McpCommand::Test(args)) => bridge_mcp("test", &args.args),
        Some(McpCommand::Configure(args)) => bridge_mcp("configure", &args.args),
        Some(McpCommand::Login(args)) => bridge_mcp("login", &args.args),
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
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
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
        bridge_mcp("test", &[String::from("alpha")]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("subcommand=test"));
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
