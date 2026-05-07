use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use serde_yaml::{Mapping, Value};

use crate::compat_cmd::CompatArgs;
use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::python_bridge::launch_python_main_command;

#[cfg(test)]
use clap::Parser;

#[derive(Subcommand, Debug)]
pub enum McpCommand {
    #[command(alias = "ls")]
    List,
    #[command(alias = "rm")]
    Remove(RemoveArgs),
    Serve(CompatArgs),
    Add(CompatArgs),
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
        Some(McpCommand::Serve(args)) => bridge_mcp("serve", &args.args),
        Some(McpCommand::Add(args)) => bridge_mcp("add", &args.args),
        Some(McpCommand::Test(args)) => bridge_mcp("test", &args.args),
        Some(McpCommand::Configure(args)) => bridge_mcp("configure", &args.args),
        Some(McpCommand::Login(args)) => bridge_mcp("login", &args.args),
    }
}

fn bridge_mcp(subcommand: &str, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    let mut argv = Vec::with_capacity(1 + passthrough.len());
    argv.push(subcommand.to_string());
    argv.extend(passthrough.iter().cloned());
    launch_python_main_command("mcp", &argv, Some("HERMES_MCP_PYTHON"), &[])
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
    Ok(name)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-mcp-{label}-{unique}"))
    }

    #[derive(Parser, Debug)]
    struct McpHarness {
        #[command(subcommand)]
        command: McpCommand,
    }

    #[test]
    fn bridge_subcommand_preserves_remaining_args() {
        let parsed = McpHarness::try_parse_from([
            "mcp",
            "add",
            "alpha",
            "--url",
            "https://example.com/mcp",
            "--env",
            "API_KEY=test",
        ])
        .unwrap();

        match parsed.command {
            McpCommand::Add(args) => {
                assert_eq!(
                    args.args,
                    vec![
                        String::from("alpha"),
                        String::from("--url"),
                        String::from("https://example.com/mcp"),
                        String::from("--env"),
                        String::from("API_KEY=test"),
                    ]
                );
            }
            _ => panic!("expected add args"),
        }
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
