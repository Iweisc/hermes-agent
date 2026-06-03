//! MCP Server Management CLI — `hermes mcp` subcommand.
//!
//! Native Rust port of `hermes_cli/mcp_config.py`.
//!
//! Implements `hermes mcp add/remove/list/test/configure/login` for
//! interactive MCP server lifecycle management. Configuration is kept in
//! `~/.hermes/config.yaml` under the `mcp_servers` key.
//!
//! ## Design notes vs the Python original
//!
//! * Server discovery (the temporary "connect, list tools, disconnect" probe)
//!   relies on the MCP SDK in Python (`tools/mcp_tool.py`). To avoid blocking on
//!   that not-yet-ported surface, discovery is abstracted behind the
//!   [`ToolProber`] trait. Callers inject a prober; a [`UnavailableProber`]
//!   default is provided that surfaces a clear error mirroring the Python
//!   "MCP SDK not available" path.
//! * Interactive input/output is abstracted behind the [`Io`] trait so the
//!   command flows can be unit-tested deterministically. A [`StdIo`] default
//!   wraps stdin/stdout.
//! * OAuth provider management (`tools/mcp_oauth_manager.py`) is abstracted
//!   behind the [`OAuthManager`] trait; a no-op default is provided.

use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};

use regex::Regex;
use serde_yaml::{Mapping, Value};

use crate::cli_colors::{color, Colors};

// ─── Regex / constants ──────────────────────────────────────────────────────

/// Validate an environment variable name: `^[A-Za-z_][A-Za-z0-9_]*$`.
fn env_var_name_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap())
}

/// Matches `${ENV_VAR}` interpolation references.
fn interpolate_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\$\{(\w+)\}").unwrap())
}

/// Known MCP presets. Empty by default, mirroring the Python `_MCP_PRESETS`.
///
/// Returned as a `Mapping` so a preset entry can carry `url`, `command`, and
/// `args` exactly as the YAML config would.
pub fn mcp_presets() -> Mapping {
    Mapping::new()
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Error raised by argument parsing / preset application (analogue of the
/// Python `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Error raised when a probe (temporary connection) fails.
#[derive(Debug, Clone)]
pub struct ProbeError(pub String);

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProbeError {}

// ─── Pluggable collaborators ─────────────────────────────────────────────────

/// A discovered tool: `(name, description)`.
pub type DiscoveredTool = (String, String);

/// Temporarily connect to an MCP server, list its tools, disconnect.
///
/// Mirrors `_probe_single_server` from the Python module.
pub trait ToolProber {
    /// Probe `name` using `config`, returning `(tool_name, description)` pairs.
    fn probe(&self, name: &str, config: &Mapping) -> Result<Vec<DiscoveredTool>, ProbeError>;
}

/// Default prober that always fails — used until the MCP SDK is ported.
pub struct UnavailableProber;

impl ToolProber for UnavailableProber {
    fn probe(&self, _name: &str, _config: &Mapping) -> Result<Vec<DiscoveredTool>, ProbeError> {
        Err(ProbeError(
            "MCP discovery unavailable (MCP SDK not yet ported)".to_string(),
        ))
    }
}

/// Manage OAuth providers/tokens for MCP servers.
///
/// Mirrors the subset of `tools/mcp_oauth_manager.MCPOAuthManager` used here.
pub trait OAuthManager {
    /// Acquire or build an OAuth provider for `name` at `url`.
    ///
    /// Returns `Ok(true)` when OAuth was configured, `Ok(false)` when the
    /// server does not support it / the SDK auth module is unavailable.
    fn get_or_build_provider(&self, name: &str, url: &str) -> Result<bool, String>;

    /// Remove any cached tokens/provider for `name`. Returns whether removal
    /// succeeded (errors are swallowed by callers, as in Python).
    fn remove(&self, name: &str) -> Result<(), String>;
}

/// No-op OAuth manager: never configures OAuth, removal is a no-op success.
pub struct NoopOAuthManager;

impl OAuthManager for NoopOAuthManager {
    fn get_or_build_provider(&self, _name: &str, _url: &str) -> Result<bool, String> {
        Ok(false)
    }
    fn remove(&self, _name: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Interactive checklist used during tool selection (`curses_checklist`).
///
/// Returns the set of selected indices, or `None` if the user cancelled.
pub trait Checklist {
    fn select(
        &mut self,
        title: &str,
        labels: &[String],
        preselected: &BTreeSet<usize>,
    ) -> Option<BTreeSet<usize>>;
}

/// Abstraction over interactive I/O so commands can be exercised in tests.
pub trait Io {
    /// Print a line to the output sink.
    fn println(&mut self, text: &str);
    /// Prompt the user with `question` and read a single line.
    ///
    /// Returns `None` on EOF / interrupt (matching the Python
    /// `KeyboardInterrupt`/`EOFError` handling).
    fn input(&mut self, question: &str) -> Option<String>;
    /// Prompt for a (possibly password) value, returning the entered string.
    fn prompt(&mut self, question: &str, password: bool, default: &str) -> String;
    /// Whether stdin is an interactive terminal.
    fn is_tty(&self) -> bool {
        false
    }
}

/// Standard stdin/stdout-backed [`Io`].
pub struct StdIo;

impl Io for StdIo {
    fn println(&mut self, text: &str) {
        println!("{text}");
    }

    fn input(&mut self, question: &str) -> Option<String> {
        print!("{question}");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match io::stdin().lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                None
            }
            Ok(_) => Some(line),
            Err(_) => {
                println!();
                None
            }
        }
    }

    fn prompt(&mut self, question: &str, _password: bool, default: &str) -> String {
        // Faithful enough: the shared cli_output.prompt handles password
        // masking; here we read a line and fall back to `default` if empty.
        let prefix = color(&format!("  {question}: "), &[Colors::YELLOW]);
        print!("{prefix}");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match io::stdin().lock().read_line(&mut line) {
            Ok(_) => {
                let val = line.trim();
                if val.is_empty() {
                    default.to_string()
                } else {
                    val.to_string()
                }
            }
            Err(_) => default.to_string(),
        }
    }

    fn is_tty(&self) -> bool {
        // SAFETY: isatty merely inspects a file descriptor.
        unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
    }
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────

fn info(io: &mut dyn Io, text: &str) {
    io.println(&color(&format!("  {text}"), &[Colors::DIM]));
}

fn success(io: &mut dyn Io, text: &str) {
    io.println(&color(&format!("  \u{2713} {text}"), &[Colors::GREEN]));
}

fn warning(io: &mut dyn Io, text: &str) {
    io.println(&color(&format!("  \u{26a0} {text}"), &[Colors::YELLOW]));
}

fn error(io: &mut dyn Io, text: &str) {
    io.println(&color(&format!("  \u{2717} {text}"), &[Colors::RED]));
}

/// Ask a yes/no question, returning `default` on empty input / interrupt.
fn confirm(io: &mut dyn Io, question: &str, default: bool) -> bool {
    let default_str = if default { "Y/n" } else { "y/N" };
    let prompt = color(&format!("  {question} [{default_str}]: "), &[Colors::YELLOW]);
    match io.input(&prompt) {
        None => default,
        Some(raw) => {
            let val = raw.trim().to_lowercase();
            if val.is_empty() {
                default
            } else {
                val == "y" || val == "yes"
            }
        }
    }
}

// ─── Config persistence ───────────────────────────────────────────────────────

/// Path to the hermes config file (`~/.hermes/config.yaml`), honoring
/// `HERMES_HOME`.
pub fn config_path() -> std::path::PathBuf {
    hermes_home().join("config.yaml")
}

/// Resolve the hermes home directory, honoring `HERMES_HOME`.
fn hermes_home() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HERMES_HOME") {
        return std::path::PathBuf::from(home);
    }
    let base = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join(".hermes")
}

/// User-friendly display string for the current HERMES_HOME (uses `~/`).
///
/// Local equivalent of `hermes_constants.display_hermes_home()` so the module
/// stays self-contained.
fn display_hermes_home() -> String {
    let home = hermes_home();
    let user_home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    match home.strip_prefix(&user_home) {
        Ok(rel) => format!("~/{}", rel.to_string_lossy()),
        Err(_) => home.to_string_lossy().into_owned(),
    }
}

/// Load `config.yaml` into a YAML mapping (empty mapping if missing/invalid).
pub fn load_config() -> Mapping {
    let path = config_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Mapping::new();
    };
    match serde_yaml::from_str::<Value>(&text) {
        Ok(Value::Mapping(map)) => map,
        _ => Mapping::new(),
    }
}

/// Persist `config` back to `config.yaml`, creating parent dirs.
pub fn save_config(config: &Mapping) -> io::Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_yaml::to_string(&Value::Mapping(config.clone()))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    std::fs::write(&path, text)
}

fn ykey(key: &str) -> Value {
    Value::String(key.to_string())
}

/// Read an env value from `~/.hermes/.env`, falling back to process env.
pub fn get_env_value(key: &str) -> Option<String> {
    if let Ok(val) = std::env::var(key) {
        if !val.is_empty() {
            return Some(val);
        }
    }
    let path = hermes_home().join(".env");
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == key {
                let v = v.trim().trim_matches('"').trim_matches('\'');
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Save (or update) `key=value` in `~/.hermes/.env`.
pub fn save_env_value(key: &str, value: &str) -> io::Result<()> {
    let path = hermes_home().join(".env");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default();
    let mut replaced = false;
    for line in lines.iter_mut() {
        if let Some((k, _)) = line.split_once('=') {
            if k.trim() == key {
                *line = format!("{key}={value}");
                replaced = true;
                break;
            }
        }
    }
    if !replaced {
        lines.push(format!("{key}={value}"));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    std::fs::write(&path, out)
}

// ─── Config Helpers ───────────────────────────────────────────────────────────

/// Return the `mcp_servers` mapping from config, or an empty mapping.
pub fn get_mcp_servers(config: Option<&Mapping>) -> Mapping {
    let owned;
    let cfg = match config {
        Some(c) => c,
        None => {
            owned = load_config();
            &owned
        }
    };
    match cfg.get(ykey("mcp_servers")) {
        Some(Value::Mapping(m)) if !m.is_empty() => m.clone(),
        _ => Mapping::new(),
    }
}

/// Add or update a server entry in `config.yaml`.
pub fn save_mcp_server(name: &str, server_config: &Mapping) -> io::Result<()> {
    let mut config = load_config();
    let servers = config
        .entry(ykey("mcp_servers"))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if let Value::Mapping(map) = servers {
        map.insert(ykey(name), Value::Mapping(server_config.clone()));
    }
    save_config(&config)
}

/// Remove a server from `config.yaml`. Returns `true` if it existed.
pub fn remove_mcp_server(name: &str) -> io::Result<bool> {
    let mut config = load_config();
    let Some(Value::Mapping(servers)) = config.get_mut(ykey("mcp_servers")) else {
        return Ok(false);
    };
    if servers.remove(ykey(name)).is_none() {
        return Ok(false);
    }
    if servers.is_empty() {
        config.remove(ykey("mcp_servers"));
    }
    save_config(&config)?;
    Ok(true)
}

/// Convert server name to an env-var key like `MCP_MYSERVER_API_KEY`.
pub fn env_key_for_server(name: &str) -> String {
    format!("MCP_{}_API_KEY", name.to_uppercase().replace('-', "_"))
}

/// Parse `KEY=VALUE` strings from CLI args into an env mapping.
pub fn parse_env_assignments(raw_env: &[String]) -> Result<Mapping, ConfigError> {
    let mut parsed = Mapping::new();
    for item in raw_env {
        let text = item.trim();
        if text.is_empty() {
            continue;
        }
        if !text.contains('=') {
            return Err(ConfigError(format!(
                "Invalid --env value '{text}' (expected KEY=VALUE)"
            )));
        }
        let (key, value) = text.split_once('=').unwrap();
        let key = key.trim();
        if key.is_empty() {
            return Err(ConfigError(format!(
                "Invalid --env value '{text}' (missing variable name)"
            )));
        }
        if !env_var_name_re().is_match(key) {
            return Err(ConfigError(format!(
                "Invalid --env variable name '{key}'"
            )));
        }
        parsed.insert(ykey(key), Value::String(value.to_string()));
    }
    Ok(parsed)
}

/// Outcome of [`apply_mcp_preset`]: resolved transport details + applied flag.
#[derive(Debug, Clone, PartialEq)]
pub struct PresetOutcome {
    pub url: Option<String>,
    pub command: Option<String>,
    pub cmd_args: Vec<String>,
    pub applied: bool,
}

/// Apply a known MCP preset when transport details were omitted.
///
/// Mirrors `_apply_mcp_preset`. Mutates `server_config` with the preset's
/// `url`/`command`/`args` when applied.
pub fn apply_mcp_preset(
    presets: &Mapping,
    preset_name: Option<&str>,
    url: Option<String>,
    command: Option<String>,
    cmd_args: Vec<String>,
    server_config: &mut Mapping,
) -> Result<PresetOutcome, ConfigError> {
    let Some(preset_name) = preset_name else {
        return Ok(PresetOutcome {
            url,
            command,
            cmd_args,
            applied: false,
        });
    };

    let Some(Value::Mapping(preset)) = presets.get(ykey(preset_name)) else {
        return Err(ConfigError(format!("Unknown MCP preset: {preset_name}")));
    };

    if url.is_some() || command.is_some() {
        return Ok(PresetOutcome {
            url,
            command,
            cmd_args,
            applied: false,
        });
    }

    let preset_url = preset
        .get(ykey("url"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let preset_command = preset
        .get(ykey("command"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let preset_args: Vec<String> = preset
        .get(ykey("args"))
        .and_then(Value::as_sequence)
        .map(|seq| {
            seq.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    if let Some(u) = &preset_url {
        server_config.insert(ykey("url"), Value::String(u.clone()));
    }
    if let Some(c) = &preset_command {
        server_config.insert(ykey("command"), Value::String(c.clone()));
    }
    if !preset_args.is_empty() {
        server_config.insert(
            ykey("args"),
            Value::Sequence(preset_args.iter().map(|a| Value::String(a.clone())).collect()),
        );
    }

    Ok(PresetOutcome {
        url: preset_url,
        command: preset_command,
        cmd_args: preset_args,
        applied: true,
    })
}

/// Resolve `${ENV_VAR}` references in a string against the process env.
pub fn interpolate_value(value: &str) -> String {
    interpolate_re()
        .replace_all(value, |caps: &regex::Captures| {
            std::env::var(&caps[1]).unwrap_or_default()
        })
        .into_owned()
}

/// Truncate `value` (by character count) to `max_len`, with a `...` suffix when
/// over the limit — mirroring Python slicing `desc[:max_len-3] + "..."`.
fn truncate(value: &str, max_len: usize) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= max_len {
        return value.to_string();
    }
    if max_len <= 3 {
        return chars[..max_len].iter().collect();
    }
    let prefix: String = chars[..max_len - 3].iter().collect();
    format!("{prefix}...")
}

// ─── Arguments ──────────────────────────────────────────────────────────────

/// Arguments for `hermes mcp add`.
#[derive(Debug, Clone, Default)]
pub struct AddArgs {
    pub name: String,
    pub url: Option<String>,
    pub command: Option<String>,
    pub args: Vec<String>,
    /// One of `Some("oauth")`, `Some("header")`, or `None`.
    pub auth: Option<String>,
    pub preset: Option<String>,
    pub env: Vec<String>,
}

// ─── hermes mcp add ──────────────────────────────────────────────────────────

/// Add a new MCP server with discovery-first tool selection.
pub fn cmd_mcp_add(
    args: &AddArgs,
    io: &mut dyn Io,
    prober: &dyn ToolProber,
    oauth: &dyn OAuthManager,
    checklist: &mut dyn Checklist,
) {
    let name = args.name.clone();
    let presets = mcp_presets();

    let mut server_config = Mapping::new();

    let explicit_env = match parse_env_assignments(&args.env) {
        Ok(env) => env,
        Err(e) => {
            error(io, &e.0);
            return;
        }
    };
    let preset_outcome = match apply_mcp_preset(
        &presets,
        args.preset.as_deref(),
        args.url.clone(),
        args.command.clone(),
        args.args.clone(),
        &mut server_config,
    ) {
        Ok(o) => o,
        Err(e) => {
            error(io, &e.0);
            return;
        }
    };
    let url = preset_outcome.url;
    let command = preset_outcome.command;
    let cmd_args = preset_outcome.cmd_args;

    if url.is_some() && !explicit_env.is_empty() {
        error(
            io,
            "--env is only supported for stdio MCP servers (--command or stdio presets)",
        );
        return;
    }

    // Validate transport
    if url.is_none() && command.is_none() {
        error(
            io,
            "Must specify --url <endpoint>, --command <cmd>, or --preset <name>",
        );
        info(io, "Examples:");
        info(io, r#"  hermes mcp add ink --url "https://mcp.ml.ink/mcp""#);
        info(
            io,
            "  hermes mcp add github --command npx --args @modelcontextprotocol/server-github",
        );
        info(io, "  hermes mcp add myserver --preset mypreset");
        return;
    }

    // Check if server already exists
    let existing = get_mcp_servers(None);
    if existing.contains_key(ykey(&name)) {
        if !confirm(
            io,
            &format!("Server '{name}' already exists. Overwrite?"),
            false,
        ) {
            info(io, "Cancelled.");
            return;
        }
    }

    // Build initial config
    if let Some(u) = &url {
        server_config.insert(ykey("url"), Value::String(u.clone()));
    } else {
        server_config.insert(
            ykey("command"),
            Value::String(command.clone().unwrap_or_default()),
        );
        if !cmd_args.is_empty() {
            server_config.insert(
                ykey("args"),
                Value::Sequence(cmd_args.iter().map(|a| Value::String(a.clone())).collect()),
            );
        }
        if !explicit_env.is_empty() {
            server_config.insert(ykey("env"), Value::Mapping(explicit_env.clone()));
        }
    }

    // ── Authentication ────────────────────────────────────────────────
    let auth_type = args.auth.as_deref();

    if let Some(u) = &url {
        if auth_type == Some("oauth") {
            io.println("");
            info(io, &format!("Starting OAuth flow for '{name}'..."));
            let mut oauth_ok = false;
            match oauth.get_or_build_provider(&name, u) {
                Ok(true) => {
                    server_config.insert(ykey("auth"), Value::String("oauth".to_string()));
                    success(
                        io,
                        "OAuth configured (tokens will be acquired on first connection)",
                    );
                    oauth_ok = true;
                }
                Ok(false) => {
                    warning(io, "OAuth setup failed — MCP SDK auth module not available");
                }
                Err(e) => {
                    warning(io, &format!("OAuth error: {e}"));
                }
            }

            if !oauth_ok {
                info(io, "This server may not support OAuth.");
                if confirm(io, "Continue without authentication?", true) {
                    // Don't store auth: oauth — server doesn't support it
                } else {
                    info(io, "Cancelled.");
                    return;
                }
            }
        } else {
            // Prompt for API key / Bearer token for HTTP servers
            io.println("");
            info(io, &format!("Connecting to {u}"));
            let needs_auth = confirm(io, "Does this server require authentication?", true);
            if needs_auth && (auth_type == Some("header") || auth_type.is_none()) {
                let env_key = env_key_for_server(&name);
                let existing_key = get_env_value(&env_key);
                let api_key = if let Some(existing) = &existing_key {
                    success(io, &format!("{env_key}: already configured"));
                    existing.clone()
                } else {
                    let key = io.prompt("API key / Bearer token", true, "");
                    if !key.is_empty() {
                        let _ = save_env_value(&env_key, &key);
                        success(
                            io,
                            &format!("Saved to {}/.env as {env_key}", display_hermes_home()),
                        );
                    }
                    key
                };

                if !api_key.is_empty() || existing_key.is_some() {
                    let mut headers = Mapping::new();
                    headers.insert(
                        ykey("Authorization"),
                        Value::String(format!("Bearer ${{{env_key}}}")),
                    );
                    server_config.insert(ykey("headers"), Value::Mapping(headers));
                }
            }
        }
    }

    // ── Discovery: connect and list tools ─────────────────────────────
    io.println("");
    io.println(&color(&format!("  Connecting to '{name}'..."), &[Colors::CYAN]));

    let tools = match prober.probe(&name, &server_config) {
        Ok(t) => t,
        Err(exc) => {
            error(io, &format!("Failed to connect: {exc}"));
            if confirm(io, "Save config anyway (you can test later)?", false) {
                server_config.insert(ykey("enabled"), Value::Bool(false));
                let _ = save_mcp_server(&name, &server_config);
                success(io, &format!("Saved '{name}' to config (disabled)"));
                info(io, &format!("Fix the issue, then: hermes mcp test {name}"));
            }
            return;
        }
    };

    if tools.is_empty() {
        warning(io, "Server connected but reported no tools.");
        if confirm(io, "Save config anyway?", true) {
            let _ = save_mcp_server(&name, &server_config);
            success(io, &format!("Saved '{name}' to config"));
        }
        return;
    }

    // ── Tool selection ────────────────────────────────────────────────
    io.println("");
    success(io, &format!("Connected! Found {} tool(s) from '{name}':", tools.len()));
    io.println("");
    for (tool_name, desc) in &tools {
        let short = truncate(desc, 60);
        let colored = color(tool_name, &[Colors::GREEN]);
        io.println(&format!("    {colored:40} {short}"));
    }
    io.println("");

    // Ask: enable all, select, or cancel
    let prompt = color(
        &format!("  Enable all {} tools? [Y/n/select]: ", tools.len()),
        &[Colors::YELLOW],
    );
    let choice = match io.input(&prompt) {
        None => {
            info(io, "Cancelled.");
            return;
        }
        Some(raw) => raw.trim().to_lowercase(),
    };

    if choice == "n" || choice == "no" {
        info(io, "Cancelled — server not saved.");
        return;
    }

    let (tool_count, total): (usize, usize);
    if choice == "s" || choice == "select" {
        let labels: Vec<String> = tools
            .iter()
            .map(|(n, d)| format!("{n}  —  {d}"))
            .collect();
        let pre_selected: BTreeSet<usize> = (0..tools.len()).collect();

        let chosen = checklist.select(&format!("Select tools for '{name}'"), &labels, &pre_selected);

        let Some(chosen) = chosen else {
            info(io, "No tools selected — server not saved.");
            return;
        };
        if chosen.is_empty() {
            info(io, "No tools selected — server not saved.");
            return;
        }

        let mut sorted: Vec<usize> = chosen.into_iter().collect();
        sorted.sort_unstable();
        let chosen_names: Vec<Value> = sorted
            .iter()
            .map(|&i| Value::String(tools[i].0.clone()))
            .collect();

        let tools_entry = server_config
            .entry(ykey("tools"))
            .or_insert_with(|| Value::Mapping(Mapping::new()));
        if let Value::Mapping(m) = tools_entry {
            m.insert(ykey("include"), Value::Sequence(chosen_names.clone()));
        }

        tool_count = chosen_names.len();
        total = tools.len();
    } else {
        // Enable all (no filter needed — default behaviour)
        tool_count = tools.len();
        total = tools.len();
    }

    // ── Save ──────────────────────────────────────────────────────────
    server_config.insert(ykey("enabled"), Value::Bool(true));
    let _ = save_mcp_server(&name, &server_config);

    io.println("");
    success(
        io,
        &format!(
            "Saved '{name}' to {}/config.yaml ({tool_count}/{total} tools enabled)",
            display_hermes_home()
        ),
    );
    info(io, "Start a new session to use these tools.");
}

// ─── hermes mcp remove ───────────────────────────────────────────────────────

/// Remove an MCP server from config.
pub fn cmd_mcp_remove(name: &str, io: &mut dyn Io, oauth: &dyn OAuthManager) {
    let existing = get_mcp_servers(None);

    if !existing.contains_key(ykey(name)) {
        error(io, &format!("Server '{name}' not found in config."));
        let servers: Vec<String> = existing
            .keys()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect();
        if !servers.is_empty() {
            info(io, &format!("Available servers: {}", servers.join(", ")));
        }
        return;
    }

    if !confirm(io, &format!("Remove server '{name}'?"), true) {
        info(io, "Cancelled.");
        return;
    }

    let _ = remove_mcp_server(name);
    success(io, &format!("Removed '{name}' from config"));

    // Clean up OAuth tokens if they exist.
    if oauth.remove(name).is_ok() {
        success(io, "Cleaned up OAuth tokens");
    }
}

// ─── hermes mcp list ──────────────────────────────────────────────────────────

/// List all configured MCP servers.
pub fn cmd_mcp_list(io: &mut dyn Io) {
    let servers = get_mcp_servers(None);

    if servers.is_empty() {
        io.println("");
        info(io, "No MCP servers configured.");
        io.println("");
        info(io, "Add one with:");
        info(io, "  hermes mcp add <name> --url <endpoint>");
        info(io, "  hermes mcp add <name> --command <cmd> --args <args...>");
        io.println("");
        return;
    }

    io.println("");
    io.println(&color("  MCP Servers:", &[Colors::CYAN, Colors::BOLD]));
    io.println("");

    // Table header
    io.println(&format!(
        "  {:<16} {:<30} {:<12} {:<10}",
        "Name", "Transport", "Tools", "Status"
    ));
    io.println(&format!(
        "  {} {} {} {}",
        "\u{2500}".repeat(16),
        "\u{2500}".repeat(30),
        "\u{2500}".repeat(12),
        "\u{2500}".repeat(10),
    ));

    for (name_v, cfg_v) in &servers {
        let name = name_v.as_str().unwrap_or("");
        let cfg = match cfg_v {
            Value::Mapping(m) => m.clone(),
            _ => Mapping::new(),
        };

        // Transport info
        let transport = if let Some(url) = cfg.get(ykey("url")).and_then(Value::as_str) {
            truncate(url, 28)
        } else if let Some(cmd) = cfg.get(ykey("command")).and_then(Value::as_str) {
            let cmd_args = cfg.get(ykey("args")).and_then(Value::as_sequence);
            let mut transport = match cmd_args {
                Some(seq) if !seq.is_empty() => {
                    let joined: Vec<String> = seq
                        .iter()
                        .take(2)
                        .map(|a| value_to_string(a))
                        .collect();
                    format!("{cmd} {}", joined.join(" "))
                }
                _ => cmd.to_string(),
            };
            transport = truncate(&transport, 28);
            transport
        } else {
            "?".to_string()
        };

        // Tool count
        let tools_str = match cfg.get(ykey("tools")) {
            Some(Value::Mapping(tools_cfg)) => {
                let include = tools_cfg.get(ykey("include")).and_then(Value::as_sequence);
                let exclude = tools_cfg.get(ykey("exclude")).and_then(Value::as_sequence);
                if let Some(inc) = include.filter(|s| !s.is_empty()) {
                    format!("{} selected", inc.len())
                } else if let Some(exc) = exclude.filter(|s| !s.is_empty()) {
                    format!("-{} excluded", exc.len())
                } else {
                    "all".to_string()
                }
            }
            _ => "all".to_string(),
        };

        // Enabled status
        let enabled = match cfg.get(ykey("enabled")) {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => {
                let low = s.to_lowercase();
                low == "true" || low == "1" || low == "yes"
            }
            None => true,
            _ => true,
        };
        let status = if enabled {
            color("\u{2713} enabled", &[Colors::GREEN])
        } else {
            color("\u{2717} disabled", &[Colors::DIM])
        };

        io.println(&format!(
            "  {name:<16} {transport:<30} {tools_str:<12} {status}"
        ));
    }

    io.println("");
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => serde_yaml::to_string(other)
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
    }
}

// ─── hermes mcp test ──────────────────────────────────────────────────────────

/// Test connection to an MCP server.
pub fn cmd_mcp_test(name: &str, io: &mut dyn Io, prober: &dyn ToolProber) {
    let servers = get_mcp_servers(None);

    if !servers.contains_key(ykey(name)) {
        error(io, &format!("Server '{name}' not found in config."));
        let available: Vec<String> = servers
            .keys()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect();
        if !available.is_empty() {
            info(io, &format!("Available: {}", available.join(", ")));
        }
        return;
    }
    let cfg = match servers.get(ykey(name)) {
        Some(Value::Mapping(m)) => m.clone(),
        _ => Mapping::new(),
    };

    io.println("");
    io.println(&color(&format!("  Testing '{name}'..."), &[Colors::CYAN]));

    // Show transport info
    if let Some(url) = cfg.get(ykey("url")).and_then(Value::as_str) {
        info(io, &format!("Transport: HTTP → {url}"));
    } else {
        let cmd = cfg
            .get(ykey("command"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        info(io, &format!("Transport: stdio → {cmd}"));
    }

    // Show auth info (masked)
    let auth_type = cfg
        .get(ykey("auth"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let headers = cfg.get(ykey("headers")).and_then(Value::as_mapping);
    if auth_type == "oauth" {
        info(io, "Auth: OAuth 2.1 PKCE");
    } else if let Some(headers) = headers.filter(|m| !m.is_empty()) {
        for (k, v) in headers {
            let key = k.as_str().unwrap_or("");
            if let Value::String(val) = v {
                let lk = key.to_lowercase();
                if lk.contains("key") || lk.contains("auth") {
                    let resolved = interpolate_value(val);
                    let masked = mask_secret(&resolved);
                    io.println(&format!("    {key}: {masked}"));
                }
            }
        }
    } else {
        info(io, "Auth: none");
    }

    // Attempt connection
    let start = std::time::Instant::now();
    let tools = match prober.probe(name, &cfg) {
        Ok(t) => t,
        Err(exc) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            error(io, &format!("Connection failed ({elapsed_ms:.0}ms): {exc}"));
            return;
        }
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    success(io, &format!("Connected ({elapsed_ms:.0}ms)"));
    success(io, &format!("Tools discovered: {}", tools.len()));

    if !tools.is_empty() {
        io.println("");
        for (tool_name, desc) in &tools {
            let short = truncate(desc, 55);
            let colored = color(tool_name, &[Colors::GREEN]);
            io.println(&format!("    {colored:36} {short}"));
        }
    }
    io.println("");
}

/// Mask a secret: keep 4 chars each side when long enough, else `***`.
fn mask_secret(resolved: &str) -> String {
    let chars: Vec<char> = resolved.chars().collect();
    if chars.len() > 8 {
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}***{tail}")
    } else {
        "***".to_string()
    }
}

// ─── hermes mcp login ────────────────────────────────────────────────────────

/// Force re-authentication for an OAuth-based MCP server.
pub fn cmd_mcp_login(name: &str, io: &mut dyn Io, prober: &dyn ToolProber, oauth: &dyn OAuthManager) {
    let servers = get_mcp_servers(None);

    let Some(Value::Mapping(server_config)) = servers.get(ykey(name)).cloned() else {
        error(io, &format!("Server '{name}' not found in config."));
        if !servers.is_empty() {
            let names: Vec<String> = servers
                .keys()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect();
            info(io, &format!("Available servers: {}", names.join(", ")));
        }
        return;
    };

    let url = server_config.get(ykey("url")).and_then(Value::as_str);
    if url.is_none() {
        error(
            io,
            &format!("Server '{name}' has no URL — not an OAuth-capable server"),
        );
        return;
    }
    let auth = server_config
        .get(ykey("auth"))
        .and_then(Value::as_str);
    if auth != Some("oauth") {
        error(
            io,
            &format!(
                "Server '{name}' is not configured for OAuth (auth={})",
                auth.unwrap_or("None")
            ),
        );
        info(
            io,
            "Use `hermes mcp remove` + `hermes mcp add` to reconfigure auth.",
        );
        return;
    }

    // Wipe both disk and in-memory cache so the next probe forces a fresh flow.
    if let Err(exc) = oauth.remove(name) {
        warning(io, &format!("Could not clear existing OAuth state: {exc}"));
    }

    io.println("");
    info(io, &format!("Starting OAuth flow for '{name}'..."));

    match prober.probe(name, &server_config) {
        Ok(tools) => {
            if !tools.is_empty() {
                success(io, &format!("Authenticated — {} tool(s) available", tools.len()));
            } else {
                success(io, "Authenticated (server reported no tools)");
            }
        }
        Err(exc) => {
            error(io, &format!("Authentication failed: {exc}"));
        }
    }
}

// ─── hermes mcp configure ────────────────────────────────────────────────────

/// Result of `cmd_mcp_configure`, distinguishing the non-tty early exit.
#[derive(Debug, PartialEq, Eq)]
pub enum ConfigureOutcome {
    /// Ran (whether or not changes were saved).
    Ran,
    /// stdin was not a TTY; the command requires an interactive terminal.
    NotInteractive,
}

/// Reconfigure which tools are enabled for an existing MCP server.
///
/// Returns [`ConfigureOutcome::NotInteractive`] (mirroring the Python
/// `sys.exit(1)` on a non-tty) so the caller can set the exit status.
pub fn cmd_mcp_configure(
    name: &str,
    io: &mut dyn Io,
    prober: &dyn ToolProber,
    checklist: &mut dyn Checklist,
) -> ConfigureOutcome {
    if !io.is_tty() {
        return ConfigureOutcome::NotInteractive;
    }

    let servers = get_mcp_servers(None);

    let Some(Value::Mapping(cfg)) = servers.get(ykey(name)).cloned() else {
        error(io, &format!("Server '{name}' not found in config."));
        let available: Vec<String> = servers
            .keys()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect();
        if !available.is_empty() {
            info(io, &format!("Available: {}", available.join(", ")));
        }
        return ConfigureOutcome::Ran;
    };

    // Discover all available tools
    io.println("");
    io.println(&color(
        &format!("  Connecting to '{name}' to discover tools..."),
        &[Colors::CYAN],
    ));

    let all_tools = match prober.probe(name, &cfg) {
        Ok(t) => t,
        Err(exc) => {
            error(io, &format!("Failed to connect: {exc}"));
            return ConfigureOutcome::Ran;
        }
    };

    if all_tools.is_empty() {
        warning(io, "Server reports no tools.");
        return ConfigureOutcome::Ran;
    }

    // Determine which are currently enabled
    let tools_cfg = cfg.get(ykey("tools")).and_then(Value::as_mapping);
    let include = tools_cfg
        .and_then(|m| m.get(ykey("include")))
        .and_then(Value::as_sequence);
    let exclude = tools_cfg
        .and_then(|m| m.get(ykey("exclude")))
        .and_then(Value::as_sequence);

    let tool_names: Vec<String> = all_tools.iter().map(|t| t.0.clone()).collect();

    let pre_selected: BTreeSet<usize> = if let Some(inc) = include.filter(|s| !s.is_empty()) {
        let include_set: BTreeSet<&str> = inc.iter().filter_map(Value::as_str).collect();
        tool_names
            .iter()
            .enumerate()
            .filter_map(|(i, tn)| include_set.contains(tn.as_str()).then_some(i))
            .collect()
    } else if let Some(exc) = exclude.filter(|s| !s.is_empty()) {
        let exclude_set: BTreeSet<&str> = exc.iter().filter_map(Value::as_str).collect();
        tool_names
            .iter()
            .enumerate()
            .filter_map(|(i, tn)| (!exclude_set.contains(tn.as_str())).then_some(i))
            .collect()
    } else {
        (0..all_tools.len()).collect()
    };

    let currently = pre_selected.len();
    let total = all_tools.len();
    info(io, &format!("Currently {currently}/{total} tools enabled for '{name}'."));
    io.println("");

    // Interactive checklist
    let labels: Vec<String> = all_tools
        .iter()
        .map(|(n, d)| format!("{n}  —  {d}"))
        .collect();

    let chosen = match checklist.select(&format!("Select tools for '{name}'"), &labels, &pre_selected)
    {
        Some(c) => c,
        None => pre_selected.clone(),
    };

    if chosen == pre_selected {
        info(io, "No changes made.");
        return ConfigureOutcome::Ran;
    }

    // Update config
    let mut config = load_config();
    let mut server_entry = match config
        .get(ykey("mcp_servers"))
        .and_then(Value::as_mapping)
        .and_then(|m| m.get(ykey(name)))
    {
        Some(Value::Mapping(m)) => m.clone(),
        _ => Mapping::new(),
    };

    if chosen.len() == total {
        // All selected → remove include/exclude (register all)
        server_entry.remove(ykey("tools"));
    } else {
        let mut sorted: Vec<usize> = chosen.iter().copied().collect();
        sorted.sort_unstable();
        let chosen_names: Vec<Value> = sorted
            .iter()
            .map(|&i| Value::String(tool_names[i].clone()))
            .collect();
        let tools_entry = server_entry
            .entry(ykey("tools"))
            .or_insert_with(|| Value::Mapping(Mapping::new()));
        if let Value::Mapping(m) = tools_entry {
            m.insert(ykey("include"), Value::Sequence(chosen_names));
            m.remove(ykey("exclude"));
        }
    }

    let servers_entry = config
        .entry(ykey("mcp_servers"))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if let Value::Mapping(m) = servers_entry {
        m.insert(ykey(name), Value::Mapping(server_entry));
    }
    let _ = save_config(&config);

    let new_count = chosen.len();
    success(io, &format!("Updated config: {new_count}/{total} tools enabled"));
    info(io, "Start a new session for changes to take effect.");
    ConfigureOutcome::Ran
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────

/// Help footer printed when no subcommand is given.
pub fn print_help_footer(io: &mut dyn Io) {
    io.println(&color("  Commands:", &[Colors::CYAN]));
    info(io, "hermes mcp serve                              Run as MCP server");
    info(io, "hermes mcp add <name> --url <endpoint>        Add an MCP server");
    info(io, "hermes mcp add <name> --command <cmd>         Add a stdio server");
    info(io, "hermes mcp add <name> --preset <preset>       Add from a known preset");
    info(io, "hermes mcp remove <name>                      Remove a server");
    info(io, "hermes mcp list                               List servers");
    info(io, "hermes mcp test <name>                        Test connection");
    info(io, "hermes mcp configure <name>                   Toggle tools");
    info(io, "hermes mcp login <name>                       Re-authenticate OAuth");
    io.println("");
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate HERMES_HOME / process env.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Scriptable Io: queued inputs, captured output.
    struct FakeIo {
        inputs: Vec<String>,
        idx: usize,
        out: Vec<String>,
        prompts: Vec<String>,
        tty: bool,
    }

    impl FakeIo {
        fn new(inputs: &[&str]) -> Self {
            FakeIo {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
                idx: 0,
                out: Vec::new(),
                prompts: Vec::new(),
                tty: true,
            }
        }
        fn joined(&self) -> String {
            self.out.join("\n")
        }
    }

    impl Io for FakeIo {
        fn println(&mut self, text: &str) {
            self.out.push(text.to_string());
        }
        fn input(&mut self, _question: &str) -> Option<String> {
            if self.idx < self.inputs.len() {
                let v = self.inputs[self.idx].clone();
                self.idx += 1;
                Some(v)
            } else {
                None
            }
        }
        fn prompt(&mut self, _question: &str, _password: bool, default: &str) -> String {
            if self.idx < self.prompts.len() {
                let v = self.prompts[self.idx].clone();
                self.idx += 1;
                return v;
            }
            // Reuse the input queue for prompt answers in tests.
            if self.idx < self.inputs.len() {
                let v = self.inputs[self.idx].clone();
                self.idx += 1;
                return v;
            }
            default.to_string()
        }
        fn is_tty(&self) -> bool {
            self.tty
        }
    }

    struct OkProber(Vec<DiscoveredTool>);
    impl ToolProber for OkProber {
        fn probe(&self, _n: &str, _c: &Mapping) -> Result<Vec<DiscoveredTool>, ProbeError> {
            Ok(self.0.clone())
        }
    }

    struct FailProber;
    impl ToolProber for FailProber {
        fn probe(&self, _n: &str, _c: &Mapping) -> Result<Vec<DiscoveredTool>, ProbeError> {
            Err(ProbeError("401 Unauthorized".to_string()))
        }
    }

    struct AllChecklist;
    impl Checklist for AllChecklist {
        fn select(
            &mut self,
            _t: &str,
            labels: &[String],
            _p: &BTreeSet<usize>,
        ) -> Option<BTreeSet<usize>> {
            Some((0..labels.len()).collect())
        }
    }

    struct SubsetChecklist(Vec<usize>);
    impl Checklist for SubsetChecklist {
        fn select(
            &mut self,
            _t: &str,
            _l: &[String],
            _p: &BTreeSet<usize>,
        ) -> Option<BTreeSet<usize>> {
            Some(self.0.iter().copied().collect())
        }
    }

    fn with_tmp_home<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!(
            "hermes_mcp_cfg_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: edition 2024 requires unsafe for env mutation; serialized via ENV_LOCK.
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        f();
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn env_key_for_server_uppercases_and_replaces_dashes() {
        assert_eq!(env_key_for_server("my-server"), "MCP_MY_SERVER_API_KEY");
        assert_eq!(env_key_for_server("ink"), "MCP_INK_API_KEY");
    }

    #[test]
    fn parse_env_assignments_valid_and_invalid() {
        let parsed = parse_env_assignments(&["A=1".to_string(), "B=x=y".to_string()]).unwrap();
        assert_eq!(parsed.get(ykey("A")).unwrap().as_str(), Some("1"));
        assert_eq!(parsed.get(ykey("B")).unwrap().as_str(), Some("x=y"));

        assert!(parse_env_assignments(&["NOEQ".to_string()]).is_err());
        assert!(parse_env_assignments(&["=val".to_string()]).is_err());
        assert!(parse_env_assignments(&["1BAD=x".to_string()]).is_err());

        // Blank entries are skipped.
        let parsed = parse_env_assignments(&["   ".to_string(), "C=3".to_string()]).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn apply_preset_unknown_errors() {
        let presets = Mapping::new();
        let mut sc = Mapping::new();
        let err = apply_mcp_preset(&presets, Some("nope"), None, None, vec![], &mut sc)
            .unwrap_err();
        assert_eq!(err.0, "Unknown MCP preset: nope");
    }

    #[test]
    fn apply_preset_fills_transport() {
        let mut presets = Mapping::new();
        let mut p = Mapping::new();
        p.insert(ykey("command"), Value::String("npx".into()));
        p.insert(
            ykey("args"),
            Value::Sequence(vec![Value::String("server-x".into())]),
        );
        presets.insert(ykey("demo"), Value::Mapping(p));

        let mut sc = Mapping::new();
        let out = apply_mcp_preset(&presets, Some("demo"), None, None, vec![], &mut sc).unwrap();
        assert!(out.applied);
        assert_eq!(out.command.as_deref(), Some("npx"));
        assert_eq!(out.cmd_args, vec!["server-x".to_string()]);
        assert_eq!(sc.get(ykey("command")).unwrap().as_str(), Some("npx"));
    }

    #[test]
    fn apply_preset_skips_when_explicit_transport_given() {
        let mut presets = Mapping::new();
        let mut p = Mapping::new();
        p.insert(ykey("url"), Value::String("https://preset".into()));
        presets.insert(ykey("demo"), Value::Mapping(p));

        let mut sc = Mapping::new();
        let out = apply_mcp_preset(
            &presets,
            Some("demo"),
            Some("https://explicit".into()),
            None,
            vec![],
            &mut sc,
        )
        .unwrap();
        assert!(!out.applied);
        assert_eq!(out.url.as_deref(), Some("https://explicit"));
    }

    #[test]
    fn interpolate_replaces_env_refs() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized via ENV_LOCK; edition 2024 unsafe env mutation.
        unsafe {
            std::env::set_var("MCP_TEST_TOKEN", "secret123");
        }
        assert_eq!(interpolate_value("Bearer ${MCP_TEST_TOKEN}"), "Bearer secret123");
        assert_eq!(interpolate_value("Bearer ${MISSING_XYZ}"), "Bearer ");
        unsafe {
            std::env::remove_var("MCP_TEST_TOKEN");
        }
    }

    #[test]
    fn mask_secret_logic() {
        assert_eq!(mask_secret("123456789"), "1234***6789");
        assert_eq!(mask_secret("short"), "***");
        assert_eq!(mask_secret("12345678"), "***");
    }

    #[test]
    fn truncate_matches_python_slicing() {
        assert_eq!(truncate("short", 60), "short");
        let long = "a".repeat(100);
        let t = truncate(&long, 60);
        assert_eq!(t.chars().count(), 60);
        assert!(t.ends_with("..."));
    }

    #[test]
    fn save_load_remove_roundtrip() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://x".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let servers = get_mcp_servers(None);
            assert!(servers.contains_key(ykey("alpha")));

            assert!(remove_mcp_server("alpha").unwrap());
            assert!(!remove_mcp_server("alpha").unwrap());

            // mcp_servers key dropped when last server removed.
            let cfg = load_config();
            assert!(!cfg.contains_key(ykey("mcp_servers")));
        });
    }

    #[test]
    fn env_value_roundtrip() {
        with_tmp_home(|| {
            save_env_value("MCP_FOO_API_KEY", "tok").unwrap();
            assert_eq!(get_env_value("MCP_FOO_API_KEY").as_deref(), Some("tok"));
            // Overwrite existing key.
            save_env_value("MCP_FOO_API_KEY", "tok2").unwrap();
            assert_eq!(get_env_value("MCP_FOO_API_KEY").as_deref(), Some("tok2"));
        });
    }

    #[test]
    fn add_requires_transport() {
        with_tmp_home(|| {
            let mut io = FakeIo::new(&[]);
            let args = AddArgs {
                name: "x".into(),
                ..Default::default()
            };
            cmd_mcp_add(
                &args,
                &mut io,
                &UnavailableProber,
                &NoopOAuthManager,
                &mut AllChecklist,
            );
            assert!(io.joined().contains("Must specify --url"));
        });
    }

    #[test]
    fn add_stdio_enable_all_saves() {
        with_tmp_home(|| {
            // Inputs: skip overwrite check (server is new) -> enable-all prompt "y".
            let mut io = FakeIo::new(&["y"]);
            let args = AddArgs {
                name: "alpha".into(),
                command: Some("npx".into()),
                args: vec!["server-x".into()],
                ..Default::default()
            };
            let prober = OkProber(vec![
                ("tool_a".into(), "desc a".into()),
                ("tool_b".into(), "desc b".into()),
            ]);
            cmd_mcp_add(&args, &mut io, &prober, &NoopOAuthManager, &mut AllChecklist);

            let servers = get_mcp_servers(None);
            let entry = servers.get(ykey("alpha")).unwrap().as_mapping().unwrap();
            assert_eq!(entry.get(ykey("command")).unwrap().as_str(), Some("npx"));
            assert_eq!(entry.get(ykey("enabled")).unwrap().as_bool(), Some(true));
            assert!(io.joined().contains("2/2 tools enabled"));
        });
    }

    #[test]
    fn add_select_subset_writes_include() {
        with_tmp_home(|| {
            let mut io = FakeIo::new(&["select"]);
            let args = AddArgs {
                name: "beta".into(),
                command: Some("npx".into()),
                ..Default::default()
            };
            let prober = OkProber(vec![
                ("t0".into(), "d0".into()),
                ("t1".into(), "d1".into()),
                ("t2".into(), "d2".into()),
            ]);
            cmd_mcp_add(
                &args,
                &mut io,
                &prober,
                &NoopOAuthManager,
                &mut SubsetChecklist(vec![0, 2]),
            );

            let servers = get_mcp_servers(None);
            let entry = servers.get(ykey("beta")).unwrap().as_mapping().unwrap();
            let include = entry
                .get(ykey("tools"))
                .unwrap()
                .as_mapping()
                .unwrap()
                .get(ykey("include"))
                .unwrap()
                .as_sequence()
                .unwrap();
            let names: Vec<&str> = include.iter().filter_map(Value::as_str).collect();
            assert_eq!(names, vec!["t0", "t2"]);
            assert!(io.joined().contains("2/3 tools enabled"));
        });
    }

    #[test]
    fn add_probe_failure_saves_disabled_on_confirm() {
        with_tmp_home(|| {
            let args = AddArgs {
                name: "gamma".into(),
                url: Some("https://x".into()),
                ..Default::default()
            };
            // Inputs: "does this server require auth?" -> n,
            // then "Save config anyway (you can test later)?" -> y.
            let mut io = FakeIo::new(&["n", "y"]);
            cmd_mcp_add(&args, &mut io, &FailProber, &NoopOAuthManager, &mut AllChecklist);
            let servers = get_mcp_servers(None);
            let entry = servers.get(ykey("gamma")).unwrap().as_mapping().unwrap();
            assert_eq!(entry.get(ykey("enabled")).unwrap().as_bool(), Some(false));
        });
    }

    #[test]
    fn list_empty_and_populated() {
        with_tmp_home(|| {
            let mut io = FakeIo::new(&[]);
            cmd_mcp_list(&mut io);
            assert!(io.joined().contains("No MCP servers configured."));

            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://mcp.example/mcp".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let mut io = FakeIo::new(&[]);
            cmd_mcp_list(&mut io);
            let out = io.joined();
            assert!(out.contains("MCP Servers:"));
            assert!(out.contains("alpha"));
            assert!(out.contains("enabled"));
        });
    }

    #[test]
    fn remove_missing_lists_available() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://x".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let mut io = FakeIo::new(&[]);
            cmd_mcp_remove("missing", &mut io, &NoopOAuthManager);
            let out = io.joined();
            assert!(out.contains("not found"));
            assert!(out.contains("alpha"));
        });
    }

    #[test]
    fn remove_confirmed_deletes() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://x".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let mut io = FakeIo::new(&["y"]);
            cmd_mcp_remove("alpha", &mut io, &NoopOAuthManager);
            assert!(get_mcp_servers(None).is_empty());
        });
    }

    #[test]
    fn test_command_reports_not_found() {
        with_tmp_home(|| {
            let mut io = FakeIo::new(&[]);
            cmd_mcp_test("nope", &mut io, &UnavailableProber);
            assert!(io.joined().contains("not found"));
        });
    }

    #[test]
    fn test_command_masks_header_secret() {
        with_tmp_home(|| {
            // SAFETY: env mutation serialized via ENV_LOCK held by with_tmp_home.
            unsafe {
                std::env::set_var("MCP_TOK", "abcdefghij");
            }
            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://x".into()));
            let mut headers = Mapping::new();
            headers.insert(
                ykey("Authorization"),
                Value::String("Bearer ${MCP_TOK}".into()),
            );
            sc.insert(ykey("headers"), Value::Mapping(headers));
            save_mcp_server("alpha", &sc).unwrap();

            let mut io = FakeIo::new(&[]);
            cmd_mcp_test("alpha", &mut io, &OkProber(vec![]));
            let out = io.joined();
            // "Bearer abcdefghij" -> masked Bear***ghij
            assert!(out.contains("Bear***ghij"), "out: {out}");
            unsafe {
                std::env::remove_var("MCP_TOK");
            }
        });
    }

    #[test]
    fn login_requires_oauth() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("url"), Value::String("https://x".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let mut io = FakeIo::new(&[]);
            cmd_mcp_login("alpha", &mut io, &UnavailableProber, &NoopOAuthManager);
            assert!(io.joined().contains("not configured for OAuth"));
        });
    }

    #[test]
    fn configure_not_interactive() {
        with_tmp_home(|| {
            let mut io = FakeIo::new(&[]);
            io.tty = false;
            let outcome = cmd_mcp_configure("alpha", &mut io, &UnavailableProber, &mut AllChecklist);
            assert_eq!(outcome, ConfigureOutcome::NotInteractive);
        });
    }

    #[test]
    fn configure_updates_include_list() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("command"), Value::String("npx".into()));
            save_mcp_server("alpha", &sc).unwrap();

            let prober = OkProber(vec![
                ("t0".into(), "d0".into()),
                ("t1".into(), "d1".into()),
                ("t2".into(), "d2".into()),
            ]);
            let mut io = FakeIo::new(&[]);
            // pre_selected = all (no tools filter), choose subset {1}
            let outcome =
                cmd_mcp_configure("alpha", &mut io, &prober, &mut SubsetChecklist(vec![1]));
            assert_eq!(outcome, ConfigureOutcome::Ran);

            let servers = get_mcp_servers(None);
            let entry = servers.get(ykey("alpha")).unwrap().as_mapping().unwrap();
            let include = entry
                .get(ykey("tools"))
                .unwrap()
                .as_mapping()
                .unwrap()
                .get(ykey("include"))
                .unwrap()
                .as_sequence()
                .unwrap();
            let names: Vec<&str> = include.iter().filter_map(Value::as_str).collect();
            assert_eq!(names, vec!["t1"]);
        });
    }

    #[test]
    fn configure_all_selected_removes_filter() {
        with_tmp_home(|| {
            let mut sc = Mapping::new();
            sc.insert(ykey("command"), Value::String("npx".into()));
            let mut tools = Mapping::new();
            tools.insert(
                ykey("include"),
                Value::Sequence(vec![Value::String("t0".into())]),
            );
            sc.insert(ykey("tools"), Value::Mapping(tools));
            save_mcp_server("alpha", &sc).unwrap();

            let prober = OkProber(vec![
                ("t0".into(), "d0".into()),
                ("t1".into(), "d1".into()),
            ]);
            let mut io = FakeIo::new(&[]);
            // currently include=[t0] -> pre_selected {0}; choose all {0,1}
            let outcome =
                cmd_mcp_configure("alpha", &mut io, &prober, &mut AllChecklist);
            assert_eq!(outcome, ConfigureOutcome::Ran);

            let servers = get_mcp_servers(None);
            let entry = servers.get(ykey("alpha")).unwrap().as_mapping().unwrap();
            assert!(entry.get(ykey("tools")).is_none());
        });
    }
}
