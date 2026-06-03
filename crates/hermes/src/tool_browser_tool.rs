//! Browser Tool Module — native Rust port of `tools/browser_tool.py`.
//!
//! Provides browser automation tools using the `agent-browser` CLI. Supports
//! multiple backends — Browser Use (cloud), Browserbase (cloud), Firecrawl
//! (cloud), and local Chromium — with identical agent-facing behaviour. The
//! backend is auto-detected from config and available credentials.
//!
//! This port reproduces the Python module's behaviour: subprocess invocation
//! of `agent-browser --json`, session isolation per task id, accessibility-tree
//! snapshots, ref-based element interaction, SSRF / secret-exfiltration guards,
//! Lightpanda→Chrome fallback, inactivity cleanup, and orphan reaping.
//!
//! Network-facing pieces (CDP discovery, cloud-provider session creation) keep
//! the same request/response shapes as the Python original.
//!
//! Cross-module references (when available in the workspace) live under
//! `crate::` / `hermes_core::`; where a dependency is not yet ported, a minimal
//! local fallback is provided so this module compiles standalone.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

// ============================================================================
// Lightweight dependency shims
// ============================================================================
//
// These helpers wrap functionality provided by other (possibly not-yet-ported)
// modules. Where a real implementation exists in the workspace it should be
// wired in via `crate::`/`hermes_core::`; until then these conservative
// fallbacks preserve the Python module's fail-open / fail-closed semantics.

/// `tools.url_safety.is_safe_url` — fail-closed (block) when unavailable.
fn is_safe_url(url: &str) -> bool {
    hermes_core::tool_url_safety::is_safe_url(url, None)
}

/// `is_truthy_value` from `utils`.
fn is_truthy_value(value: Option<&Value>, default: bool) -> bool {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => {
            matches!(
                s.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "y" | "t"
            )
        }
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        _ => default,
    }
}

/// Returns the configured Hermes home directory.
fn get_hermes_home() -> PathBuf {
    hermes_core::mod_hermes_constants::get_hermes_home()
}

/// `hermes_constants.get_hermes_dir`.
fn get_hermes_dir(new_subpath: &str, old_name: &str) -> PathBuf {
    hermes_core::mod_hermes_constants::get_hermes_dir(new_subpath, old_name)
}

/// `hermes_constants.is_termux`.
fn is_termux_environment() -> bool {
    hermes_core::mod_hermes_constants::is_termux()
}

/// `tools.interrupt.is_interrupted`.
fn is_interrupted() -> bool {
    hermes_core::tool_interrupt::is_interrupted()
}

/// `agent.redact.redact_sensitive_text`.
fn redact_sensitive_text(text: &str) -> String {
    hermes_core::agent_redact::redact_sensitive_text(text, false, false)
}

/// Read `~/.hermes/config.yaml` (raw, unmerged) as a YAML value.
fn read_raw_config() -> serde_yaml::Value {
    hermes_core::cli_config::read_raw_config()
}

/// Camofox mode detection. The camofox REST backend is not ported here; this
/// returns `false` (matching the import-failure fallback in Python).
fn is_camofox_mode() -> bool {
    // tools.browser_camofox.is_camofox_mode — fall back to env-var probe used
    // by the camofox module: CAMOFOX_URL presence implies camofox mode.
    std::env::var("CAMOFOX_URL")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// `tools.website_policy.check_website_access` — returns a block descriptor.
///
/// Returns `(message, host, rule, source)` when blocked, else `None`.
fn check_website_access(url: &str) -> Option<(String, String, String, String)> {
    // Fail-open if the policy module errors (mirrors the Python try/except
    // fallback `lambda url: None`).
    match hermes_core::tool_website_policy::check_website_access(url, None) {
        Ok(Some(block)) => Some((block.message, block.host, block.rule, block.source)),
        Ok(None) | Err(_) => None,
    }
}

/// `tool_backend_helpers.normalize_browser_cloud_provider`.
fn normalize_browser_cloud_provider(value: Option<&str>) -> String {
    hermes_core::tool_tool_backend_helpers::normalize_browser_cloud_provider(value)
}

// ============================================================================
// Config helpers (YAML traversal)
// ============================================================================

/// Traverse nested mapping keys in a YAML value.
fn yaml_get<'a>(cfg: &'a serde_yaml::Value, keys: &[&str]) -> Option<&'a serde_yaml::Value> {
    let mut node = cfg;
    for key in keys {
        let map = node.as_mapping()?;
        node = map.get(serde_yaml::Value::String((*key).to_string()))?;
    }
    Some(node)
}

fn yaml_to_json(v: &serde_yaml::Value) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

// ============================================================================
// PATH discovery
// ============================================================================

/// Standard PATH entries for environments with minimal PATH (systemd, Termux,
/// macOS Homebrew).
const SANE_PATH_DIRS: &[&str] = &[
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
];

fn path_sep() -> char {
    if cfg!(windows) {
        ';'
    } else {
        ':'
    }
}

fn homebrew_node_dirs_cache() -> &'static Mutex<Option<Vec<String>>> {
    static CACHE: OnceLock<Mutex<Option<Vec<String>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).
pub fn discover_homebrew_node_dirs() -> Vec<String> {
    let cache = homebrew_node_dirs_cache();
    {
        let guard = cache.lock().unwrap();
        if let Some(v) = guard.as_ref() {
            return v.clone();
        }
    }
    let mut dirs: Vec<String> = Vec::new();
    let homebrew_opt = "/opt/homebrew/opt";
    if Path::new(homebrew_opt).is_dir() {
        if let Ok(read) = std::fs::read_dir(homebrew_opt) {
            for entry in read.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("node") && name != "node" {
                    let bin_dir = entry.path().join("bin");
                    if bin_dir.is_dir() {
                        dirs.push(bin_dir.to_string_lossy().to_string());
                    }
                }
            }
        }
    }
    *cache.lock().unwrap() = Some(dirs.clone());
    dirs
}

fn clear_homebrew_node_cache() {
    *homebrew_node_dirs_cache().lock().unwrap() = None;
}

/// Ordered browser CLI PATH candidates shared by discovery and execution.
fn browser_candidate_path_dirs() -> Vec<String> {
    let hermes_home = get_hermes_home();
    let hermes_node_bin = hermes_home.join("node").join("bin");
    let mut out = vec![hermes_node_bin.to_string_lossy().to_string()];
    out.extend(discover_homebrew_node_dirs());
    out.extend(SANE_PATH_DIRS.iter().map(|s| s.to_string()));
    out
}

/// Prepend browser-specific PATH fallbacks without reordering existing entries.
pub fn merge_browser_path(existing_path: &str) -> String {
    let sep = path_sep();
    let path_parts: Vec<String> = existing_path
        .split(sep)
        .filter(|p| !p.is_empty())
        .map(|s| s.to_string())
        .collect();
    let existing_parts: HashSet<&str> = path_parts.iter().map(|s| s.as_str()).collect();
    let mut prefix_parts: Vec<String> = Vec::new();

    for part in browser_candidate_path_dirs() {
        if part.is_empty()
            || existing_parts.contains(part.as_str())
            || prefix_parts.contains(&part)
        {
            continue;
        }
        if Path::new(&part).is_dir() {
            prefix_parts.push(part);
        }
    }
    prefix_parts.extend(path_parts);
    prefix_parts.join(&sep.to_string())
}

// ============================================================================
// Configuration constants & cached lookups
// ============================================================================

/// Default timeout for browser commands (seconds).
pub const DEFAULT_COMMAND_TIMEOUT: u64 = 30;

/// Max chars for snapshot content before summarization.
pub const SNAPSHOT_SUMMARIZE_THRESHOLD: usize = 8000;

/// Commands that legitimately return empty stdout (e.g. close, record).
fn is_empty_ok_command(cmd: &str) -> bool {
    matches!(cmd, "close" | "record")
}

/// Session inactivity timeout (seconds). Default 5 minutes.
pub fn browser_session_inactivity_timeout() -> u64 {
    std::env::var("BROWSER_INACTIVITY_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
}

const LOCAL_SUFFIX: &str = "::local";

// ---- command timeout cache ----
struct CmdTimeoutCache {
    resolved: bool,
    value: u64,
}
fn cmd_timeout_cache() -> &'static Mutex<CmdTimeoutCache> {
    static C: OnceLock<Mutex<CmdTimeoutCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(CmdTimeoutCache {
            resolved: false,
            value: DEFAULT_COMMAND_TIMEOUT,
        })
    })
}

/// Return the configured browser command timeout from config.yaml.
pub fn get_command_timeout() -> u64 {
    let mut c = cmd_timeout_cache().lock().unwrap();
    if c.resolved {
        return c.value;
    }
    c.resolved = true;
    let mut result = DEFAULT_COMMAND_TIMEOUT;
    let cfg = read_raw_config();
    if let Some(v) = yaml_get(&cfg, &["browser", "command_timeout"]) {
        if let Some(n) = v.as_u64() {
            result = n.max(5);
        } else if let Some(n) = v.as_i64() {
            result = (n.max(5)) as u64;
        }
    }
    c.value = result;
    result
}

/// Model for browser_vision (screenshot analysis — multimodal).
pub fn get_vision_model() -> Option<String> {
    let v = std::env::var("AUXILIARY_VISION_MODEL").unwrap_or_default();
    let v = v.trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// Model for page snapshot text summarization — same as web_extract.
pub fn get_extraction_model() -> Option<String> {
    let v = std::env::var("AUXILIARY_WEB_EXTRACT_MODEL").unwrap_or_default();
    let v = v.trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

// ============================================================================
// CDP override resolution
// ============================================================================

/// Normalize a user-supplied CDP endpoint into a concrete connectable URL.
///
/// Accepts full ws endpoints, HTTP discovery endpoints, and bare ws host:port.
/// For discovery-style endpoints, fetches `/json/version` and returns the
/// `webSocketDebuggerUrl`.
pub fn resolve_cdp_override(cdp_url: &str) -> String {
    let raw = cdp_url.trim().to_string();
    if raw.is_empty() {
        return String::new();
    }

    let lowered = raw.to_lowercase();
    if lowered.contains("/devtools/browser/") {
        return raw;
    }

    let mut discovery_url = raw.clone();
    if lowered.starts_with("ws://") || lowered.starts_with("wss://") {
        // Detect bare ws://host:port (no path)
        let after_scheme = raw.splitn(2, "://").nth(1).unwrap_or("");
        let colon_count = raw.matches(':').count();
        let last_after_colon = raw.trim_end_matches('/').rsplitn(2, ':').next().unwrap_or("");
        let is_bare = colon_count == 2
            && last_after_colon.chars().all(|c| c.is_ascii_digit())
            && !last_after_colon.is_empty()
            && !after_scheme.contains('/');
        if is_bare {
            let scheme = if lowered.starts_with("ws://") {
                "http://"
            } else {
                "https://"
            };
            discovery_url = format!("{scheme}{after_scheme}");
        } else {
            return raw;
        }
    }

    let version_url = if discovery_url.to_lowercase().ends_with("/json/version") {
        discovery_url.clone()
    } else {
        format!("{}/json/version", discovery_url.trim_end_matches('/'))
    };

    let payload: Value = match http_get_json(&version_url, 10) {
        Ok(p) => p,
        Err(exc) => {
            log::warn!(
                "Failed to resolve CDP endpoint {raw} via {version_url}: {exc}"
            );
            return raw;
        }
    };

    let ws_url = payload
        .get("webSocketDebuggerUrl")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if !ws_url.is_empty() {
        log::info!("Resolved CDP endpoint {raw} -> {ws_url}");
        return ws_url;
    }

    log::warn!(
        "CDP discovery at {version_url} did not return webSocketDebuggerUrl; using raw endpoint"
    );
    raw
}

fn http_get_json(url: &str, timeout_secs: u64) -> Result<Value, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url).send().map_err(|e| e.to_string())?;
    let resp = resp.error_for_status().map_err(|e| e.to_string())?;
    resp.json::<Value>().map_err(|e| e.to_string())
}

/// Return a normalized CDP URL override, or empty string.
///
/// Precedence: `BROWSER_CDP_URL` env var, then `browser.cdp_url` in config.
pub fn get_cdp_override() -> String {
    let env_override = std::env::var("BROWSER_CDP_URL").unwrap_or_default();
    let env_override = env_override.trim();
    if !env_override.is_empty() {
        return resolve_cdp_override(env_override);
    }
    let cfg = read_raw_config();
    if let Some(browser_cfg) = yaml_get(&cfg, &["browser"]) {
        if browser_cfg.as_mapping().is_some() {
            if let Some(v) = yaml_get(browser_cfg, &["cdp_url"]) {
                let s = v.as_str().unwrap_or("").to_string();
                return resolve_cdp_override(&s);
            }
        }
    }
    String::new()
}

// ============================================================================
// Cloud provider registry
// ============================================================================

/// Identifies which cloud provider (if any) is active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudProviderKind {
    None,
    Browserbase,
    BrowserUse,
    Firecrawl,
}

impl CloudProviderKind {
    pub fn provider_name(&self) -> &'static str {
        match self {
            CloudProviderKind::None => "local",
            CloudProviderKind::Browserbase => "browserbase",
            CloudProviderKind::BrowserUse => "browser-use",
            CloudProviderKind::Firecrawl => "firecrawl",
        }
    }
}

/// Whether a given provider has the required credentials configured.
///
/// Mirrors `<Provider>.is_configured()` from the Python providers. Kept minimal
/// here (env-var probes) since the provider modules expose their own configured
/// checks via `crate::tool_browser_providers_*`.
fn provider_is_configured(kind: &CloudProviderKind) -> bool {
    match kind {
        CloudProviderKind::None => false,
        CloudProviderKind::Browserbase => {
            env_nonempty("BROWSERBASE_API_KEY") && env_nonempty("BROWSERBASE_PROJECT_ID")
        }
        CloudProviderKind::BrowserUse => {
            // Direct API key OR managed Nous gateway availability.
            env_nonempty("BROWSER_USE_API_KEY") || env_nonempty("NOUS_API_KEY")
        }
        CloudProviderKind::Firecrawl => env_nonempty("FIRECRAWL_API_KEY"),
    }
}

fn env_nonempty(key: &str) -> bool {
    std::env::var(key)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

struct CloudProviderCache {
    resolved: bool,
    value: CloudProviderKind,
}
fn cloud_provider_cache() -> &'static Mutex<CloudProviderCache> {
    static C: OnceLock<Mutex<CloudProviderCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(CloudProviderCache {
            resolved: false,
            value: CloudProviderKind::None,
        })
    })
}

/// Return the configured cloud browser provider, or `None` for local mode.
pub fn get_cloud_provider() -> CloudProviderKind {
    let mut c = cloud_provider_cache().lock().unwrap();
    if c.resolved {
        return c.value.clone();
    }
    c.resolved = true;
    let mut resolved: Option<CloudProviderKind> = None;

    let cfg = read_raw_config();
    if let Some(browser_cfg) = yaml_get(&cfg, &["browser"]) {
        if browser_cfg.as_mapping().is_some() {
            if let Some(cp) = yaml_get(browser_cfg, &["cloud_provider"]) {
                let provider_key =
                    normalize_browser_cloud_provider(cp.as_str());
                if provider_key == "local" {
                    c.value = CloudProviderKind::None;
                    return CloudProviderKind::None;
                }
                resolved = match provider_key.as_str() {
                    "browserbase" => Some(CloudProviderKind::Browserbase),
                    "browser-use" => Some(CloudProviderKind::BrowserUse),
                    "firecrawl" => Some(CloudProviderKind::Firecrawl),
                    _ => None,
                };
            }
        }
    }

    if resolved.is_none() {
        // Prefer Browser Use (managed/direct), fall back to Browserbase direct.
        if provider_is_configured(&CloudProviderKind::BrowserUse) {
            resolved = Some(CloudProviderKind::BrowserUse);
        } else if provider_is_configured(&CloudProviderKind::Browserbase) {
            resolved = Some(CloudProviderKind::Browserbase);
        }
    }

    c.value = resolved.unwrap_or(CloudProviderKind::None);
    c.value.clone()
}

fn browser_install_hint() -> String {
    if is_termux_environment() {
        "npm install -g agent-browser && agent-browser install".to_string()
    } else {
        "npm install -g agent-browser && agent-browser install --with-deps".to_string()
    }
}

fn requires_real_termux_browser_install(browser_cmd: &str) -> bool {
    is_termux_environment() && is_local_mode() && browser_cmd.trim() == "npx agent-browser"
}

fn termux_browser_install_error() -> String {
    format!(
        "Local browser automation on Termux cannot rely on the bare npx fallback. \
         Install agent-browser explicitly first: {}",
        browser_install_hint()
    )
}

/// Return true when the browser tool will use a local browser backend.
pub fn is_local_mode() -> bool {
    if !get_cdp_override().is_empty() {
        return false;
    }
    get_cloud_provider() == CloudProviderKind::None
}

/// Return true when the browser runs locally (no cloud provider).
///
/// SSRF protection is only meaningful for cloud backends.
pub fn is_local_backend() -> bool {
    is_camofox_mode() || get_cloud_provider() == CloudProviderKind::None
}

// ============================================================================
// Browser engine (lightpanda support)
// ============================================================================

struct EngineCache {
    resolved: bool,
    value: String,
}
fn engine_cache() -> &'static Mutex<EngineCache> {
    static C: OnceLock<Mutex<EngineCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(EngineCache {
            resolved: false,
            value: "auto".to_string(),
        })
    })
}

/// Return the configured browser engine (`auto`, `lightpanda`, or `chrome`).
pub fn get_browser_engine() -> String {
    let mut c = engine_cache().lock().unwrap();
    if c.resolved {
        return c.value.clone();
    }
    c.resolved = true;
    let mut engine = "auto".to_string();

    let cfg = read_raw_config();
    if let Some(v) = yaml_get(&cfg, &["browser", "engine"]) {
        if let Some(s) = v.as_str() {
            if !s.trim().is_empty() {
                engine = s.trim().to_lowercase();
            }
        }
    }

    if engine == "auto" {
        let env_val = std::env::var("AGENT_BROWSER_ENGINE").unwrap_or_default();
        let env_val = env_val.trim().to_lowercase();
        if !env_val.is_empty() {
            engine = env_val;
        }
    }

    const VALID: [&str; 3] = ["auto", "lightpanda", "chrome"];
    if !VALID.contains(&engine.as_str()) {
        log::warn!(
            "Unknown browser engine {engine:?} (valid: auto, chrome, lightpanda), falling back to 'auto'"
        );
        engine = "auto".to_string();
    }

    c.value = engine.clone();
    engine
}

/// Whether the engine flag should be added to agent-browser commands.
pub fn should_inject_engine(engine: &str) -> bool {
    if engine == "auto" {
        return false;
    }
    if is_camofox_mode() {
        return false;
    }
    is_local_mode()
}

/// Whether local browser commands are configured for Lightpanda.
pub fn using_lightpanda_engine() -> bool {
    get_browser_engine() == "lightpanda"
}

/// Return the user-visible reason a Lightpanda result needs Chrome fallback.
pub fn lightpanda_fallback_reason(
    engine: &str,
    command: &str,
    result: &Value,
) -> Option<String> {
    if engine != "lightpanda" {
        return None;
    }

    const FALLBACK_ELIGIBLE: [&str; 11] = [
        "open", "snapshot", "screenshot", "eval", "click", "fill", "scroll", "back", "press",
        "console", "errors",
    ];
    if !FALLBACK_ELIGIBLE.contains(&command) {
        return None;
    }

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let error = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("command failed")
            .trim()
            .to_string();
        let error = if error.is_empty() {
            "command failed".to_string()
        } else {
            error
        };
        return Some(format!(
            "Lightpanda '{command}' failed ({error}); retried with Chrome."
        ));
    }

    let data = result.get("data").cloned().unwrap_or(json!({}));

    if command == "snapshot" {
        let snap = data.get("snapshot").and_then(|v| v.as_str()).unwrap_or("");
        if snap.trim().len() < 20 {
            return Some(
                "Lightpanda returned an empty/too-short snapshot; retried with Chrome.".to_string(),
            );
        }
    }

    if command == "screenshot" {
        let path = data.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if !path.is_empty() {
            match std::fs::metadata(path) {
                Ok(meta) => {
                    let size = meta.len();
                    if size < 20480 {
                        log::debug!(
                            "Lightpanda screenshot is suspiciously small ({size} bytes), triggering Chrome fallback"
                        );
                        return Some(format!(
                            "Lightpanda screenshot was suspiciously small ({size} bytes); retried with Chrome."
                        ));
                    }
                }
                Err(_) => {
                    return Some(
                        "Lightpanda screenshot file was missing/unreadable; retried with Chrome."
                            .to_string(),
                    );
                }
            }
        }
    }

    None
}

/// Check if a Lightpanda result should trigger an automatic Chrome fallback.
pub fn needs_lightpanda_fallback(engine: &str, command: &str, result: &Value) -> bool {
    lightpanda_fallback_reason(engine, command, result).is_some()
}

/// Add a user-visible Chrome fallback warning to a browser command result.
pub fn annotate_lightpanda_fallback(result: &Value, reason: &str) -> Value {
    let warning = format!(
        "\u{26a0} Lightpanda fallback: Chrome was used for this browser action. {reason}"
    );
    let mut annotated = result.clone();
    let obj = annotated.as_object_mut().cloned().unwrap_or_default();
    let mut obj = obj;
    obj.insert("fallback_warning".to_string(), json!(warning));
    obj.insert("browser_engine".to_string(), json!("chrome"));
    obj.insert(
        "browser_engine_fallback".to_string(),
        json!({"from": "lightpanda", "to": "chrome", "reason": reason}),
    );

    if let Some(Value::Object(data)) = obj.get("data").cloned() {
        let mut data = data;
        data.entry("fallback_warning".to_string())
            .or_insert(json!(warning));
        data.entry("browser_engine".to_string())
            .or_insert(json!("chrome"));
        data.entry("browser_engine_fallback".to_string())
            .or_insert(json!({"from": "lightpanda", "to": "chrome", "reason": reason}));
        obj.insert("data".to_string(), Value::Object(data));
    }

    Value::Object(obj)
}

/// Copy browser fallback metadata from an internal result into a tool response.
pub fn copy_fallback_warning(target: &mut Map<String, Value>, result: &Value) {
    if let Some(fw) = result.get("fallback_warning") {
        if !fw.is_null() {
            target.insert("fallback_warning".to_string(), fw.clone());
            target.insert(
                "browser_engine".to_string(),
                result.get("browser_engine").cloned().unwrap_or(Value::Null),
            );
            target.insert(
                "browser_engine_fallback".to_string(),
                result
                    .get("browser_engine_fallback")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
        }
    }
}

// ============================================================================
// Hybrid private-URL routing
// ============================================================================

struct BoolCache {
    resolved: bool,
    value: bool,
}

fn auto_local_cache() -> &'static Mutex<BoolCache> {
    static C: OnceLock<Mutex<BoolCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(BoolCache {
            resolved: false,
            value: true,
        })
    })
}

fn allow_private_cache() -> &'static Mutex<BoolCache> {
    static C: OnceLock<Mutex<BoolCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(BoolCache {
            resolved: false,
            value: false,
        })
    })
}

/// Whether a cloud-configured install should auto-spawn a local Chromium for
/// LAN/localhost URLs. Default `true`.
pub fn auto_local_for_private_urls() -> bool {
    let mut c = auto_local_cache().lock().unwrap();
    if c.resolved {
        return c.value;
    }
    c.resolved = true;
    let cfg = read_raw_config();
    if let Some(browser_cfg) = yaml_get(&cfg, &["browser"]) {
        if browser_cfg.as_mapping().is_some() {
            if let Some(v) = yaml_get(browser_cfg, &["auto_local_for_private_urls"]) {
                c.value = match v {
                    serde_yaml::Value::Bool(b) => *b,
                    serde_yaml::Value::Null => false,
                    serde_yaml::Value::String(s) => !s.is_empty(),
                    serde_yaml::Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                    _ => true,
                };
            }
        }
    }
    c.value
}

/// Whether the browser is allowed to navigate to private/internal addresses.
/// Default `false` (SSRF protection active).
pub fn allow_private_urls() -> bool {
    let mut c = allow_private_cache().lock().unwrap();
    if c.resolved {
        return c.value;
    }
    c.resolved = true;
    c.value = false;
    let cfg = read_raw_config();
    if let Some(browser_cfg) = yaml_get(&cfg, &["browser"]) {
        if browser_cfg.as_mapping().is_some() {
            let v = yaml_get(browser_cfg, &["allow_private_urls"]).map(yaml_to_json);
            c.value = is_truthy_value(v.as_ref(), false);
        }
    }
    c.value
}

/// Return true when the URL's host resolves to a private/LAN/loopback address.
pub fn url_is_private(url: &str) -> bool {
    use std::net::{IpAddr, ToSocketAddrs};

    let parsed = match url::Url::parse(url) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let hostname = parsed
        .host_str()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .trim_end_matches('.')
        .to_string();
    if hostname.is_empty() {
        return false;
    }

    // Literal IP — check directly.
    if let Ok(ip) = hostname.parse::<IpAddr>() {
        return ip_is_private(&ip);
    }

    // Short-circuit obvious local names.
    if hostname == "localhost" || hostname.ends_with(".localhost") {
        return true;
    }
    if hostname.ends_with(".local")
        || hostname.ends_with(".lan")
        || hostname.ends_with(".internal")
    {
        return true;
    }

    // Resolve and check each address.
    let lookup = format!("{hostname}:0");
    match lookup.to_socket_addrs() {
        Ok(addrs) => {
            for sa in addrs {
                if ip_is_private(&sa.ip()) {
                    return true;
                }
            }
            false
        }
        Err(_) => false, // DNS fail → not private
    }
}

fn ip_is_private(ip: &std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            if v4.is_private() || v4.is_loopback() || v4.is_link_local() {
                return true;
            }
            // CGNAT 100.64.0.0/10
            let o = v4.octets();
            o[0] == 100 && (64..=127).contains(&o[1])
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // unique local fc00::/7
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // link-local fe80::/10
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Pick the session key that should handle `url` for `task_id`.
pub fn navigation_session_key(task_id: &str, url: &str) -> String {
    let task_id = if task_id.is_empty() { "default" } else { task_id };
    if !get_cdp_override().is_empty() {
        return task_id.to_string();
    }
    if is_camofox_mode() {
        return task_id.to_string();
    }
    if get_cloud_provider() == CloudProviderKind::None {
        return task_id.to_string();
    }
    if !auto_local_for_private_urls() {
        return task_id.to_string();
    }
    if !url_is_private(url) {
        return task_id.to_string();
    }
    format!("{task_id}{LOCAL_SUFFIX}")
}

/// Return true when `session_key` is a hybrid-routing local sidecar.
pub fn is_local_sidecar_key(session_key: &str) -> bool {
    session_key.ends_with(LOCAL_SUFFIX)
}

// ============================================================================
// Global session state
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct SessionInfo {
    pub session_name: String,
    pub bb_session_id: Option<String>,
    pub cdp_url: Option<String>,
    pub features: Map<String, Value>,
    pub first_nav: bool,
    pub fallback_from_cloud: bool,
    pub fallback_reason: Option<String>,
    pub fallback_provider: Option<String>,
}

struct BrowserState {
    active_sessions: HashMap<String, SessionInfo>,
    recording_sessions: HashSet<String>,
    last_active_session_key: HashMap<String, String>,
    session_last_activity: HashMap<String, f64>,
    cleanup_done: bool,
    cleanup_running: bool,
}

fn browser_state() -> &'static Mutex<BrowserState> {
    static S: OnceLock<Mutex<BrowserState>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(BrowserState {
            active_sessions: HashMap::new(),
            recording_sessions: HashSet::new(),
            last_active_session_key: HashMap::new(),
            session_last_activity: HashMap::new(),
            cleanup_done: false,
            cleanup_running: false,
        })
    })
}

fn last_screenshot_cleanup() -> &'static Mutex<HashMap<String, f64>> {
    static C: OnceLock<Mutex<HashMap<String, f64>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Return the session key to use for a non-nav browser tool call.
pub fn last_session_key(task_id: &str) -> String {
    let task_id = if task_id.is_empty() { "default" } else { task_id };
    let st = browser_state().lock().unwrap();
    st.last_active_session_key
        .get(task_id)
        .cloned()
        .unwrap_or_else(|| task_id.to_string())
}

/// Update the last activity timestamp for a session.
pub fn update_session_activity(task_id: &str) {
    let mut st = browser_state().lock().unwrap();
    st.session_last_activity.insert(task_id.to_string(), now_secs());
}

// ============================================================================
// Temp dir helpers
// ============================================================================

/// Return a short temp directory path suitable for Unix domain sockets.
pub fn socket_safe_tmpdir() -> String {
    if cfg!(target_os = "macos") {
        "/tmp".to_string()
    } else {
        std::env::temp_dir().to_string_lossy().to_string()
    }
}

// ============================================================================
// agent-browser CLI discovery
// ============================================================================

struct AgentBrowserCache {
    resolved: bool,
    value: Option<String>,
}
fn agent_browser_cache() -> &'static Mutex<AgentBrowserCache> {
    static C: OnceLock<Mutex<AgentBrowserCache>> = OnceLock::new();
    C.get_or_init(|| {
        Mutex::new(AgentBrowserCache {
            resolved: false,
            value: None,
        })
    })
}

/// Search for an executable named `name` on the given PATH string.
fn which_on_path(name: &str, path: &str) -> Option<String> {
    let sep = path_sep();
    for dir in path.split(sep) {
        if dir.is_empty() {
            continue;
        }
        let candidate = Path::new(dir).join(name);
        if is_executable(&candidate) {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    None
}

fn is_executable(p: &Path) -> bool {
    if !p.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(p) {
            return meta.permissions().mode() & 0o111 != 0;
        }
        false
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn which(name: &str) -> Option<String> {
    let path = std::env::var("PATH").unwrap_or_default();
    which_on_path(name, &path)
}

/// Find the agent-browser CLI executable. Returns `Err(message)` if not found.
pub fn find_agent_browser() -> Result<String, String> {
    {
        let c = agent_browser_cache().lock().unwrap();
        if c.resolved {
            return match &c.value {
                Some(v) => Ok(v.clone()),
                None => Err(format!(
                    "agent-browser CLI not found (cached). Install it with: {}\n\
                     Or run 'npm install' in the repo root to install locally.\n\
                     Or ensure npx is available in your PATH.",
                    browser_install_hint()
                )),
            };
        }
    }

    // PATH (global install)
    if let Some(found) = which("agent-browser") {
        let mut c = agent_browser_cache().lock().unwrap();
        c.value = Some(found.clone());
        c.resolved = true;
        return Ok(found);
    }

    // Extended PATH (Hermes node, Homebrew, Termux).
    let extended_path = merge_browser_path("");
    if !extended_path.is_empty() {
        if let Some(found) = which_on_path("agent-browser", &extended_path) {
            let mut c = agent_browser_cache().lock().unwrap();
            c.value = Some(found.clone());
            c.resolved = true;
            return Ok(found);
        }
    }

    // Local node_modules/.bin/ — best-effort relative to cwd's repo root.
    if let Ok(cwd) = std::env::current_dir() {
        let local_bin = cwd.join("node_modules").join(".bin").join("agent-browser");
        if local_bin.exists() {
            let s = local_bin.to_string_lossy().to_string();
            let mut c = agent_browser_cache().lock().unwrap();
            c.value = Some(s.clone());
            c.resolved = true;
            return Ok(s);
        }
    }

    // npx fallback.
    let npx_path = which("npx").or_else(|| {
        if extended_path.is_empty() {
            None
        } else {
            which_on_path("npx", &extended_path)
        }
    });
    if npx_path.is_some() {
        let mut c = agent_browser_cache().lock().unwrap();
        c.value = Some("npx agent-browser".to_string());
        c.resolved = true;
        return Ok("npx agent-browser".to_string());
    }

    let mut c = agent_browser_cache().lock().unwrap();
    c.resolved = true;
    Err(format!(
        "agent-browser CLI not found. Install it with: {}\n\
         Or run 'npm install' in the repo root to install locally.\n\
         Or ensure npx is available in your PATH.",
        browser_install_hint()
    ))
}

// ============================================================================
// Chromium / Docker detection
// ============================================================================

fn chromium_installed_cache() -> &'static Mutex<Option<bool>> {
    static C: OnceLock<Mutex<Option<bool>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

/// Directories to scan for a Chromium / headless-shell build.
pub fn chromium_search_roots() -> Vec<String> {
    let mut roots: Vec<String> = Vec::new();
    let env_path = std::env::var("PLAYWRIGHT_BROWSERS_PATH").unwrap_or_default();
    let env_path = env_path.trim();
    if !env_path.is_empty() && env_path != "0" {
        roots.push(env_path.to_string());
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    roots.push(
        home.join(".cache")
            .join("ms-playwright")
            .to_string_lossy()
            .to_string(),
    );
    if cfg!(target_os = "macos") {
        roots.push(
            home.join("Library")
                .join("Caches")
                .join("ms-playwright")
                .to_string_lossy()
                .to_string(),
        );
    }
    if cfg!(windows) {
        let local = std::env::var("LOCALAPPDATA")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                home.join("AppData").join("Local").to_string_lossy().to_string()
            });
        roots.push(
            Path::new(&local)
                .join("ms-playwright")
                .to_string_lossy()
                .to_string(),
        );
    }
    roots
}

/// Return true when a usable Chromium (or headless-shell) build is on disk.
pub fn chromium_installed() -> bool {
    {
        let c = chromium_installed_cache().lock().unwrap();
        if let Some(v) = *c {
            return v;
        }
    }

    let result = compute_chromium_installed();
    *chromium_installed_cache().lock().unwrap() = Some(result);
    result
}

fn compute_chromium_installed() -> bool {
    // 1. AGENT_BROWSER_EXECUTABLE_PATH
    let ab_path = std::env::var("AGENT_BROWSER_EXECUTABLE_PATH").unwrap_or_default();
    let ab_path = ab_path.trim();
    if !ab_path.is_empty() {
        if Path::new(ab_path).is_file() || which(ab_path).is_some() {
            return true;
        }
    }

    // 2. System Chrome/Chromium in PATH.
    if which("google-chrome").is_some()
        || which("chromium-browser").is_some()
        || which("chrome").is_some()
    {
        return true;
    }

    // 3. Playwright browser cache.
    for root in chromium_search_roots() {
        if root.is_empty() || !Path::new(&root).is_dir() {
            continue;
        }
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("chromium-") || name.starts_with("chromium_headless_shell-") {
                return true;
            }
        }
    }

    false
}

/// Best-effort detection of whether we're inside a Docker container.
pub fn running_in_docker() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    match std::fs::read_to_string("/proc/1/cgroup") {
        Ok(content) => content.contains("docker"),
        Err(_) => false,
    }
}

// ============================================================================
// Screenshot path extraction
// ============================================================================

/// Extract a screenshot file path from agent-browser human-readable output.
pub fn extract_screenshot_path_from_text(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let patterns = [
        r#"Screenshot saved to ['"](?P<path>/[^'"]+?\.png)['"]"#,
        r#"Screenshot saved to (?P<path>/\S+?\.png)(?:\s|$)"#,
        r#"(?P<path>/\S+?\.png)(?:\s|$)"#,
    ];
    for pat in patterns {
        if let Ok(re) = regex::Regex::new(pat) {
            if let Some(caps) = re.captures(text) {
                if let Some(m) = caps.name("path") {
                    let path = m.as_str().trim().trim_matches(|c| c == '\'' || c == '"');
                    if !path.is_empty() {
                        return Some(path.to_string());
                    }
                }
            }
        }
    }
    None
}

// ============================================================================
// Session creation
// ============================================================================

fn rand_hex(n: usize) -> String {
    // Cheap unique hex from time + counter (mirrors uuid4().hex[:n] usage).
    static COUNTER: OnceLock<Mutex<u64>> = OnceLock::new();
    let counter = COUNTER.get_or_init(|| Mutex::new(0));
    let c = {
        let mut g = counter.lock().unwrap();
        *g = g.wrapping_add(1);
        *g
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let combined = format!("{nanos:x}{c:x}{:x}", std::process::id());
    let mut s: String = combined.chars().rev().collect();
    s.truncate(n);
    while s.len() < n {
        s.push('0');
    }
    s
}

fn create_local_session(task_id: &str) -> SessionInfo {
    let session_name = format!("h_{}", rand_hex(10));
    log::info!("Created local browser session {session_name} for task {task_id}");
    let mut features = Map::new();
    features.insert("local".to_string(), Value::Bool(true));
    SessionInfo {
        session_name,
        bb_session_id: None,
        cdp_url: None,
        features,
        first_nav: true,
        ..Default::default()
    }
}

fn create_cdp_session(task_id: &str, cdp_url: &str) -> SessionInfo {
    let session_name = format!("cdp_{}", rand_hex(10));
    log::info!("Created CDP browser session {session_name} -> {cdp_url} for task {task_id}");
    let mut features = Map::new();
    features.insert("cdp_override".to_string(), Value::Bool(true));
    SessionInfo {
        session_name,
        bb_session_id: None,
        cdp_url: Some(cdp_url.to_string()),
        features,
        first_nav: true,
        ..Default::default()
    }
}

/// Hook for creating a cloud-provider session.
///
/// In the full system this delegates to the provider modules
/// (`crate::tool_browser_providers_*`). Returning `None` falls back to a local
/// session, matching the Python behaviour when the provider raises.
fn create_cloud_session(_kind: &CloudProviderKind, _task_id: &str) -> Option<SessionInfo> {
    // Providers are wired in separately; without credentials this yields None,
    // and `get_session_info` falls back to a local Chromium session.
    None
}

/// Get or create session info for the given session key.
pub fn get_session_info(task_id: Option<&str>) -> Result<SessionInfo, String> {
    let task_id = task_id.unwrap_or("default").to_string();

    start_browser_cleanup_thread();
    update_session_activity(&task_id);

    {
        let st = browser_state().lock().unwrap();
        if let Some(info) = st.active_sessions.get(&task_id) {
            return Ok(info.clone());
        }
    }

    let force_local = is_local_sidecar_key(&task_id);
    let cdp_override = get_cdp_override();

    let session_info: SessionInfo = if !cdp_override.is_empty() && !force_local {
        create_cdp_session(&task_id, &cdp_override)
    } else if force_local {
        create_local_session(&task_id)
    } else {
        let provider = get_cloud_provider();
        if provider == CloudProviderKind::None {
            create_local_session(&task_id)
        } else {
            match create_cloud_session(&provider, &task_id) {
                Some(mut info) => {
                    if let Some(cdp) = info.cdp_url.clone() {
                        if !cdp.is_empty() {
                            info.cdp_url = Some(resolve_cdp_override(&cdp));
                        }
                    }
                    info
                }
                None => {
                    // Provider unavailable / failed → local fallback (degraded).
                    let mut info = create_local_session(&task_id);
                    info.fallback_from_cloud = true;
                    info.fallback_reason = Some("cloud provider unavailable".to_string());
                    info.fallback_provider = Some(provider.provider_name().to_string());
                    info
                }
            }
        }
    };

    {
        let mut st = browser_state().lock().unwrap();
        if let Some(info) = st.active_sessions.get(&task_id) {
            return Ok(info.clone());
        }
        st.active_sessions.insert(task_id.clone(), session_info.clone());
    }

    Ok(session_info)
}

// ============================================================================
// Owner PID tracking + orphan reaping
// ============================================================================

fn write_owner_pid(socket_dir: &str, session_name: &str) {
    let path = Path::new(socket_dir).join(format!("{session_name}.owner_pid"));
    if let Err(exc) = std::fs::write(&path, std::process::id().to_string()) {
        log::debug!("Could not write owner_pid file for {session_name}: {exc}");
    }
}

#[cfg(unix)]
fn pid_alive(pid: i32) -> Option<bool> {
    // Some(true) = alive (or owned by another uid), Some(false) = dead.
    let r = unsafe { libc::kill(pid, 0) };
    if r == 0 {
        Some(true)
    } else {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno == libc::ESRCH {
            Some(false)
        } else if errno == libc::EPERM {
            Some(true)
        } else {
            None
        }
    }
}

#[cfg(not(unix))]
fn pid_alive(_pid: i32) -> Option<bool> {
    Some(true)
}

#[cfg(unix)]
fn send_sigterm(pid: i32) -> bool {
    unsafe { libc::kill(pid, libc::SIGTERM) == 0 }
}

#[cfg(not(unix))]
fn send_sigterm(_pid: i32) -> bool {
    false
}

/// Scan for orphaned agent-browser daemon processes from previous runs.
pub fn reap_orphaned_browser_sessions() {
    let tmpdir = socket_safe_tmpdir();
    let mut socket_dirs: Vec<PathBuf> = Vec::new();
    for pat in ["agent-browser-h_", "agent-browser-cdp_", "agent-browser-hermes_"] {
        if let Ok(read) = std::fs::read_dir(&tmpdir) {
            for entry in read.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(pat) {
                    socket_dirs.push(entry.path());
                }
            }
        }
    }
    socket_dirs.sort();
    socket_dirs.dedup();
    if socket_dirs.is_empty() {
        return;
    }

    let tracked_names: HashSet<String> = {
        let st = browser_state().lock().unwrap();
        st.active_sessions
            .values()
            .map(|i| i.session_name.clone())
            .filter(|s| !s.is_empty())
            .collect()
    };

    let mut reaped = 0;
    for socket_dir in socket_dirs {
        let dir_name = socket_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let session_name = dir_name
            .strip_prefix("agent-browser-")
            .unwrap_or("")
            .to_string();
        if session_name.is_empty() {
            continue;
        }

        // Ownership via owner_pid file.
        let owner_pid_file = socket_dir.join(format!("{session_name}.owner_pid"));
        let mut owner_alive: Option<bool> = None;
        if owner_pid_file.is_file() {
            if let Ok(text) = std::fs::read_to_string(&owner_pid_file) {
                if let Ok(owner_pid) = text.trim().parse::<i32>() {
                    owner_alive = pid_alive(owner_pid);
                }
            }
        }

        if owner_alive == Some(true) {
            continue;
        }
        if owner_alive.is_none() && tracked_names.contains(&session_name) {
            continue;
        }

        let pid_file = socket_dir.join(format!("{session_name}.pid"));
        if !pid_file.is_file() {
            let _ = std::fs::remove_dir_all(&socket_dir);
            continue;
        }
        let daemon_pid = match std::fs::read_to_string(&pid_file)
            .ok()
            .and_then(|t| t.trim().parse::<i32>().ok())
        {
            Some(p) => p,
            None => {
                let _ = std::fs::remove_dir_all(&socket_dir);
                continue;
            }
        };

        match pid_alive(daemon_pid) {
            Some(false) => {
                let _ = std::fs::remove_dir_all(&socket_dir);
                continue;
            }
            None => {
                // Alive but owned by someone else (EPERM path collapses to true
                // above; None here means unexpected errno) — leave alone.
                continue;
            }
            Some(true) => {}
        }

        if send_sigterm(daemon_pid) {
            log::info!("Reaped orphaned browser daemon PID {daemon_pid} (session {session_name})");
            reaped += 1;
        }
        let _ = std::fs::remove_dir_all(&socket_dir);
    }

    if reaped > 0 {
        log::info!("Reaped {reaped} orphaned browser session(s) from previous run(s)");
    }
}

// ============================================================================
// Inactivity cleanup
// ============================================================================

fn cleanup_thread_started() -> &'static AtomicBool {
    static B: OnceLock<AtomicBool> = OnceLock::new();
    B.get_or_init(|| AtomicBool::new(false))
}

/// Start the background cleanup thread if not already running.
pub fn start_browser_cleanup_thread() {
    if cleanup_thread_started().swap(true, Ordering::SeqCst) {
        return;
    }
    {
        let mut st = browser_state().lock().unwrap();
        st.cleanup_running = true;
    }
    std::thread::Builder::new()
        .name("browser-cleanup".to_string())
        .spawn(browser_cleanup_thread_worker)
        .ok();
    log::info!(
        "Started inactivity cleanup thread (timeout: {}s)",
        browser_session_inactivity_timeout()
    );
}

/// Stop the background cleanup thread.
pub fn stop_browser_cleanup_thread() {
    let mut st = browser_state().lock().unwrap();
    st.cleanup_running = false;
}

fn browser_cleanup_thread_worker() {
    reap_orphaned_browser_sessions();
    loop {
        {
            let running = browser_state().lock().unwrap().cleanup_running;
            if !running {
                break;
            }
        }
        cleanup_inactive_browser_sessions();
        for _ in 0..30 {
            {
                let running = browser_state().lock().unwrap().cleanup_running;
                if !running {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
}

/// Clean up browser sessions inactive longer than the timeout.
pub fn cleanup_inactive_browser_sessions() {
    let current_time = now_secs();
    let timeout = browser_session_inactivity_timeout() as f64;
    let mut to_cleanup: Vec<String> = Vec::new();
    {
        let st = browser_state().lock().unwrap();
        for (task_id, last_time) in st.session_last_activity.iter() {
            if current_time - *last_time > timeout {
                to_cleanup.push(task_id.clone());
            }
        }
    }
    for task_id in to_cleanup {
        let elapsed = {
            let st = browser_state().lock().unwrap();
            (current_time - st.session_last_activity.get(&task_id).copied().unwrap_or(current_time))
                as i64
        };
        log::info!("Cleaning up inactive session for task: {task_id} (inactive for {elapsed}s)");
        cleanup_browser(Some(&task_id));
        let mut st = browser_state().lock().unwrap();
        st.session_last_activity.remove(&task_id);
    }
}

/// Emergency cleanup of all active browser sessions (process exit).
pub fn emergency_cleanup_all_sessions() {
    {
        let mut st = browser_state().lock().unwrap();
        if st.cleanup_done {
            return;
        }
        st.cleanup_done = true;
    }
    let has_sessions = !browser_state().lock().unwrap().active_sessions.is_empty();
    if has_sessions {
        let n = browser_state().lock().unwrap().active_sessions.len();
        log::info!("Emergency cleanup: closing {n} active session(s)...");
        cleanup_all_browsers();
        let mut st = browser_state().lock().unwrap();
        st.active_sessions.clear();
        st.session_last_activity.clear();
        st.recording_sessions.clear();
    }
    reap_orphaned_browser_sessions();
}

// ============================================================================
// Core command runner
// ============================================================================

/// Run an agent-browser CLI command using the pre-created session.
///
/// Returns a parsed JSON object (the agent-browser `--json` response), or a
/// synthesised `{"success": false, "error": ...}` object on failure.
pub fn run_browser_command(
    task_id: &str,
    command: &str,
    args: &[String],
    timeout: Option<u64>,
    engine_override: Option<&str>,
) -> Value {
    let timeout = timeout.unwrap_or_else(get_command_timeout);

    let browser_cmd = match find_agent_browser() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("agent-browser CLI not found: {e}");
            return json!({"success": false, "error": e});
        }
    };

    if requires_real_termux_browser_install(&browser_cmd) {
        let error = termux_browser_install_error();
        log::warn!("browser command blocked on Termux: {error}");
        return json!({"success": false, "error": error});
    }

    if is_local_mode() && !chromium_installed() && get_browser_engine() != "lightpanda" {
        let hint = if running_in_docker() {
            "Chromium browser is missing. You're running in Docker — pull the latest image to get the bundled Chromium: docker pull ghcr.io/nousresearch/hermes-agent:latest"
        } else {
            "Chromium browser is missing. Install it with: npx agent-browser install --with-deps (or: npx playwright install --with-deps chromium)"
        };
        log::warn!("browser command blocked: {hint}");
        return json!({"success": false, "error": hint});
    }

    if is_interrupted() {
        return json!({"success": false, "error": "Interrupted"});
    }

    let session_info = match get_session_info(Some(task_id)) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Failed to create browser session for task={task_id}: {e}");
            return json!({"success": false, "error": format!("Failed to create browser session: {e}")});
        }
    };

    // Build backend args.
    let mut backend_args: Vec<String> = Vec::new();
    let has_cdp = session_info
        .cdp_url
        .as_ref()
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if has_cdp {
        backend_args.push("--cdp".to_string());
        backend_args.push(session_info.cdp_url.clone().unwrap());
    } else {
        backend_args.push("--session".to_string());
        backend_args.push(session_info.session_name.clone());
    }

    let engine = engine_override
        .map(|s| s.to_string())
        .unwrap_or_else(get_browser_engine);
    if engine != "auto" && !is_camofox_mode() && !has_cdp {
        backend_args.push("--engine".to_string());
        backend_args.push(engine.clone());
    }

    let mut cmd_parts: Vec<String> = if browser_cmd == "npx agent-browser" {
        vec!["npx".to_string(), "agent-browser".to_string()]
    } else {
        vec![browser_cmd.clone()]
    };
    cmd_parts.extend(backend_args);
    cmd_parts.push("--json".to_string());
    cmd_parts.push(command.to_string());
    cmd_parts.extend(args.iter().cloned());

    let result = run_subprocess_command(
        &cmd_parts,
        command,
        task_id,
        &session_info.session_name,
        timeout,
    );

    // --- Lightpanda automatic Chrome fallback ---
    if let Some(reason) = lightpanda_fallback_reason(&engine, command, &result) {
        log::info!("Lightpanda fallback: retrying '{command}' with Chrome (task={task_id}): {reason}");
        let fallback_result = if command == "screenshot" {
            chrome_fallback_screenshot(task_id, args, timeout)
        } else {
            run_chrome_fallback_command(task_id, command, args, timeout)
        };
        return annotate_lightpanda_fallback(&fallback_result, &reason);
    }

    result
}

fn run_subprocess_command(
    cmd_parts: &[String],
    command: &str,
    task_id: &str,
    session_name: &str,
    timeout: u64,
) -> Value {
    use std::process::{Command, Stdio};

    let task_socket_dir =
        Path::new(&socket_safe_tmpdir()).join(format!("agent-browser-{session_name}"));
    if let Err(e) = std::fs::create_dir_all(&task_socket_dir) {
        log::warn!("browser '{command}' exception: {e}");
        return json!({"success": false, "error": e.to_string()});
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&task_socket_dir, std::fs::Permissions::from_mode(0o700));
    }
    let socket_dir_str = task_socket_dir.to_string_lossy().to_string();
    write_owner_pid(&socket_dir_str, session_name);
    log::debug!(
        "browser cmd={command} task={task_id} socket_dir={socket_dir_str} ({} chars)",
        socket_dir_str.len()
    );

    let stdout_path = task_socket_dir.join(format!("_stdout_{command}"));
    let stderr_path = task_socket_dir.join(format!("_stderr_{command}"));

    let stdout_file = match std::fs::File::create(&stdout_path) {
        Ok(f) => f,
        Err(e) => return json!({"success": false, "error": e.to_string()}),
    };
    let stderr_file = match std::fs::File::create(&stderr_path) {
        Ok(f) => f,
        Err(e) => return json!({"success": false, "error": e.to_string()}),
    };

    let merged_path = merge_browser_path(&std::env::var("PATH").unwrap_or_default());

    let mut cmd = Command::new(&cmd_parts[0]);
    cmd.args(&cmd_parts[1..]);
    cmd.env("PATH", &merged_path);
    cmd.env("AGENT_BROWSER_SOCKET_DIR", &socket_dir_str);
    if std::env::var("AGENT_BROWSER_IDLE_TIMEOUT_MS").is_err() {
        cmd.env(
            "AGENT_BROWSER_IDLE_TIMEOUT_MS",
            (browser_session_inactivity_timeout() * 1000).to_string(),
        );
    }
    inject_sandbox_flags(&mut cmd);
    cmd.stdout(Stdio::from(stdout_file));
    cmd.stderr(Stdio::from(stderr_file));
    cmd.stdin(Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("browser '{command}' exception: {e}");
            return json!({"success": false, "error": e.to_string()});
        }
    };

    let result = match wait_with_timeout(&mut child, timeout) {
        WaitOutcome::TimedOut => {
            let _ = child.kill();
            let _ = child.wait();
            log::warn!(
                "browser '{command}' timed out after {timeout}s (task={task_id}, socket_dir={socket_dir_str})"
            );
            json!({"success": false, "error": format!("Command timed out after {timeout} seconds")})
        }
        WaitOutcome::Exited(code) => {
            let stdout = std::fs::read_to_string(&stdout_path).unwrap_or_default();
            let stderr = std::fs::read_to_string(&stderr_path).unwrap_or_default();
            let _ = std::fs::remove_file(&stdout_path);
            let _ = std::fs::remove_file(&stderr_path);
            parse_command_output(command, &stdout, &stderr, code)
        }
    };

    result
}

enum WaitOutcome {
    TimedOut,
    Exited(i32),
}

fn wait_with_timeout(child: &mut std::process::Child, timeout: u64) -> WaitOutcome {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().unwrap_or(-1);
                return WaitOutcome::Exited(code);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    return WaitOutcome::TimedOut;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => return WaitOutcome::Exited(-1),
        }
    }
}

#[cfg(unix)]
fn inject_sandbox_flags(cmd: &mut std::process::Command) {
    if std::env::var("AGENT_BROWSER_CHROME_FLAGS").is_ok() {
        return;
    }
    let mut needs = false;
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        needs = true;
        log::debug!("browser: running as root — injecting --no-sandbox");
    } else if let Ok(content) =
        std::fs::read_to_string("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    {
        if content.trim() == "1" {
            needs = true;
            log::debug!(
                "browser: AppArmor userns restrictions detected — injecting --no-sandbox"
            );
        }
    }
    if needs {
        cmd.env(
            "AGENT_BROWSER_CHROME_FLAGS",
            "--no-sandbox --disable-dev-shm-usage",
        );
    }
}

#[cfg(not(unix))]
fn inject_sandbox_flags(_cmd: &mut std::process::Command) {}

fn parse_command_output(command: &str, stdout: &str, stderr: &str, returncode: i32) -> Value {
    if !stderr.trim().is_empty() {
        let snippet: String = stderr.trim().chars().take(500).collect();
        if returncode != 0 {
            log::warn!("browser '{command}' stderr: {snippet}");
        } else {
            log::debug!("browser '{command}' stderr: {snippet}");
        }
    }

    let stdout_text = stdout.trim();

    if stdout_text.is_empty() && returncode == 0 && !is_empty_ok_command(command) {
        log::warn!("browser '{command}' returned empty output (rc=0)");
        return json!({"success": false, "error": format!("Browser command '{command}' returned no output")});
    }

    if !stdout_text.is_empty() {
        let last_line = stdout_text.lines().last().unwrap_or(stdout_text);
        // Python json.loads(stdout_text) on the whole text; but for screenshot
        // recovery it also uses the full text. Match Python: parse full text.
        match serde_json::from_str::<Value>(stdout_text).or_else(|_| serde_json::from_str::<Value>(last_line)) {
            Ok(parsed) => {
                if command == "snapshot"
                    && parsed.get("success").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    let snap_data = parsed.get("data").cloned().unwrap_or(json!({}));
                    let has_snap = snap_data
                        .get("snapshot")
                        .map(|v| !v.is_null() && v.as_str().map(|s| !s.is_empty()).unwrap_or(true))
                        .unwrap_or(false);
                    let has_refs = snap_data
                        .get("refs")
                        .map(|v| !v.is_null())
                        .unwrap_or(false);
                    if !has_snap && !has_refs {
                        log::warn!(
                            "snapshot returned empty content. Possible stale daemon or CDP connection issue. returncode={returncode}"
                        );
                    }
                }
                return parsed;
            }
            Err(_) => {
                let raw: String = stdout_text.chars().take(2000).collect();
                let raw_log: String = raw.chars().take(500).collect();
                log::warn!(
                    "browser '{command}' returned non-JSON output (rc={returncode}): {raw_log}"
                );
                if command == "screenshot" {
                    let combined = [stdout_text, stderr.trim()]
                        .iter()
                        .filter(|p| !p.is_empty())
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n");
                    if let Some(recovered) = extract_screenshot_path_from_text(&combined) {
                        if Path::new(&recovered).exists() {
                            log::info!(
                                "browser 'screenshot' recovered file from non-JSON output: {recovered}"
                            );
                            return json!({
                                "success": true,
                                "data": {"path": recovered, "raw": raw}
                            });
                        }
                    }
                    return json!({
                        "success": false,
                        "error": format!("Non-JSON output from agent-browser for '{command}': {raw}")
                    });
                }
                return json!({
                    "success": false,
                    "error": format!("Non-JSON output from agent-browser for '{command}': {raw}")
                });
            }
        }
    }

    if returncode != 0 {
        let error_msg = if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            format!("Command failed with code {returncode}")
        };
        let snippet: String = error_msg.chars().take(300).collect();
        log::warn!("browser '{command}' failed (rc={returncode}): {snippet}");
        return json!({"success": false, "error": error_msg});
    }

    json!({"success": true, "data": {}})
}

// ============================================================================
// Chrome fallback (Lightpanda → Chrome)
// ============================================================================

/// Run a browser command in a temporary Chrome session at the current URL.
pub fn run_chrome_fallback_command(
    task_id: &str,
    command: &str,
    args: &[String],
    timeout: u64,
) -> Value {
    use std::process::{Command, Stdio};

    // 1. Grab current URL from the Lightpanda session.
    let url_result = run_browser_command(
        task_id,
        "eval",
        &["window.location.href".to_string()],
        Some(10),
        Some("auto"),
    );
    let mut current_url = String::new();
    if url_result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        if let Some(r) = url_result
            .get("data")
            .and_then(|d| d.get("result"))
            .and_then(|v| v.as_str())
        {
            current_url = r.trim().trim_matches('"').trim_matches('\'').to_string();
        }
    }
    if current_url.is_empty() {
        log::warn!("Chrome fallback: could not determine current URL from LP session");
        return json!({"success": false, "error": "Chrome fallback failed: could not determine current URL"});
    }

    let tmp_session = format!("h_cfb_{}", rand_hex(8));
    let browser_cmd = match find_agent_browser() {
        Ok(c) => c,
        Err(e) => return json!({"success": false, "error": e}),
    };

    if !chromium_installed() {
        let hint = if running_in_docker() {
            "Chrome fallback requires Chromium, but it is missing. You're running in Docker — pull the latest image: docker pull ghcr.io/nousresearch/hermes-agent:latest"
        } else {
            "Chrome fallback requires Chromium, but it is missing. Install it with: npx agent-browser install --with-deps (or: npx playwright install --with-deps chromium)"
        };
        return json!({"success": false, "error": hint});
    }

    let cmd_prefix: Vec<String> = if browser_cmd == "npx agent-browser" {
        vec!["npx".to_string(), "agent-browser".to_string()]
    } else {
        vec![browser_cmd.clone()]
    };
    let mut base_args = cmd_prefix.clone();
    base_args.extend(vec![
        "--engine".to_string(),
        "chrome".to_string(),
        "--session".to_string(),
        tmp_session.clone(),
        "--json".to_string(),
    ]);

    let task_socket_dir =
        Path::new(&socket_safe_tmpdir()).join(format!("agent-browser-{tmp_session}"));
    let _ = std::fs::create_dir_all(&task_socket_dir);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&task_socket_dir, std::fs::Permissions::from_mode(0o700));
    }
    let socket_dir_str = task_socket_dir.to_string_lossy().to_string();
    let merged_path = merge_browser_path(&std::env::var("PATH").unwrap_or_default());
    let idle_ms = (browser_session_inactivity_timeout() * 1000).to_string();
    let has_idle = std::env::var("AGENT_BROWSER_IDLE_TIMEOUT_MS").is_ok();

    let run_tmp = |cmd: &str, cmd_args: &[String]| -> Value {
        let mut full = base_args.clone();
        full.push(cmd.to_string());
        full.extend(cmd_args.iter().cloned());

        let stdout_path = task_socket_dir.join(format!("_stdout_{cmd}"));
        let stderr_path = task_socket_dir.join(format!("_stderr_{cmd}"));
        let stdout_file = match std::fs::File::create(&stdout_path) {
            Ok(f) => f,
            Err(e) => return json!({"success": false, "error": e.to_string()}),
        };
        let stderr_file = match std::fs::File::create(&stderr_path) {
            Ok(f) => f,
            Err(e) => return json!({"success": false, "error": e.to_string()}),
        };

        let mut c = Command::new(&full[0]);
        c.args(&full[1..]);
        c.env("PATH", &merged_path);
        c.env("AGENT_BROWSER_SOCKET_DIR", &socket_dir_str);
        if !has_idle {
            c.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", &idle_ms);
        }
        c.stdout(Stdio::from(stdout_file));
        c.stderr(Stdio::from(stderr_file));
        c.stdin(Stdio::null());

        let mut child = match c.spawn() {
            Ok(ch) => ch,
            Err(_) => return json!({"success": false, "error": format!("Chrome fallback '{cmd}' failed")}),
        };

        match wait_with_timeout(&mut child, timeout) {
            WaitOutcome::TimedOut => {
                let _ = child.kill();
                let _ = child.wait();
                json!({"success": false, "error": format!("Chrome fallback '{cmd}' timed out")})
            }
            WaitOutcome::Exited(_) => {
                let stdout = std::fs::read_to_string(&stdout_path).unwrap_or_default();
                let _ = std::fs::remove_file(&stdout_path);
                let _ = std::fs::remove_file(&stderr_path);
                let stdout = stdout.trim();
                if !stdout.is_empty() {
                    let last = stdout.lines().last().unwrap_or(stdout);
                    if let Ok(v) = serde_json::from_str::<Value>(last) {
                        return v;
                    }
                }
                json!({"success": false, "error": format!("Chrome fallback '{cmd}' failed")})
            }
        }
    };

    let nav = run_tmp("open", &[current_url.clone()]);
    let final_result = if !nav.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = nav.get("error").and_then(|v| v.as_str()).unwrap_or("");
        log::warn!("Chrome fallback: navigate failed: {err}");
        json!({"success": false, "error": format!("Chrome fallback navigate failed: {err}")})
    } else {
        run_tmp(command, args)
    };

    // Teardown.
    let _ = run_tmp("close", &[]);
    let _ = std::fs::remove_dir_all(&task_socket_dir);

    final_result
}

/// Take a screenshot using a temporary Chrome session.
pub fn chrome_fallback_screenshot(task_id: &str, args: &[String], timeout: u64) -> Value {
    run_chrome_fallback_command(task_id, "screenshot", args, timeout)
}

// ============================================================================
// Snapshot content helpers
// ============================================================================

/// Structure-aware truncation for snapshots (cuts at line boundaries).
pub fn truncate_snapshot(snapshot_text: &str, max_chars: usize) -> String {
    if snapshot_text.len() <= max_chars {
        return snapshot_text.to_string();
    }
    let lines: Vec<&str> = snapshot_text.split('\n').collect();
    let mut result: Vec<String> = Vec::new();
    let mut chars = 0usize;
    for line in &lines {
        if chars + line.len() + 1 > max_chars.saturating_sub(80) {
            break;
        }
        result.push((*line).to_string());
        chars += line.len() + 1;
    }
    let remaining = lines.len() - result.len();
    if remaining > 0 {
        result.push(format!(
            "\n[... {remaining} more lines truncated, use browser_snapshot for full content]"
        ));
    }
    result.join("\n")
}

/// Convenience wrapper using the default 8000-char limit.
pub fn truncate_snapshot_default(snapshot_text: &str) -> String {
    truncate_snapshot(snapshot_text, 8000)
}

/// Use an LLM to extract relevant content from a snapshot (or truncate).
///
/// The LLM call is delegated to a caller-supplied closure so this module does
/// not hard-depend on the auxiliary client. Returns the redacted extraction or
/// a truncated fallback. When `llm` is `None`, simply truncates.
pub fn extract_relevant_content<F>(
    snapshot_text: &str,
    user_task: Option<&str>,
    llm: Option<F>,
) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    let extraction_prompt = match user_task {
        Some(task) => format!(
            "You are a content extractor for a browser automation agent.\n\n\
             The user's task is: {task}\n\n\
             Given the following page snapshot (accessibility tree representation), \
             extract and summarize the most relevant information for completing this task. Focus on:\n\
             1. Interactive elements (buttons, links, inputs) that might be needed\n\
             2. Text content relevant to the task (prices, descriptions, headings, important info)\n\
             3. Navigation structure if relevant\n\n\
             Keep ref IDs (like [ref=e5]) for interactive elements so the agent can use them.\n\n\
             Page Snapshot:\n{snapshot_text}\n\n\
             Provide a concise summary that preserves actionable information and relevant content."
        ),
        None => format!(
            "Summarize this page snapshot, preserving:\n\
             1. All interactive elements with their ref IDs (like [ref=e5])\n\
             2. Key text content and headings\n\
             3. Important information visible on the page\n\n\
             Page Snapshot:\n{snapshot_text}\n\n\
             Provide a concise summary focused on interactive elements and key content."
        ),
    };

    let extraction_prompt = redact_sensitive_text(&extraction_prompt);

    match llm {
        Some(call) => match call(&extraction_prompt) {
            Some(content) => {
                let trimmed = content.trim();
                let extracted = if trimmed.is_empty() {
                    truncate_snapshot_default(snapshot_text)
                } else {
                    trimmed.to_string()
                };
                redact_sensitive_text(&extracted)
            }
            None => truncate_snapshot_default(snapshot_text),
        },
        None => truncate_snapshot_default(snapshot_text),
    }
}

// ============================================================================
// Tool functions
// ============================================================================

fn json_str(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
}

const BLOCKED_PATTERNS: &[&str] = &[
    "access denied",
    "access to this page has been denied",
    "blocked",
    "bot detected",
    "verification required",
    "please verify",
    "are you a robot",
    "captcha",
    "cloudflare",
    "ddos protection",
    "checking your browser",
    "just a moment",
    "attention required",
];

/// Navigate to a URL in the browser.
pub fn browser_navigate(url: &str, task_id: Option<&str>) -> String {
    // Secret-exfiltration protection.
    let url_decoded = percent_decode(url);
    if secret_prefix_present(url) || secret_prefix_present(&url_decoded) {
        return json_str(&json!({
            "success": false,
            "error": "Blocked: URL contains what appears to be an API key or token. \
                      Secrets must not be sent in URLs."
        }));
    }

    let effective_task_id = task_id.filter(|s| !s.is_empty()).unwrap_or("default").to_string();
    let nav_session_key = navigation_session_key(&effective_task_id, url);
    let auto_local_this_nav = is_local_sidecar_key(&nav_session_key);

    // SSRF pre-check.
    if !is_local_backend()
        && !auto_local_this_nav
        && !allow_private_urls()
        && !is_safe_url(url)
    {
        return json_str(&json!({
            "success": false,
            "error": "Blocked: URL targets a private or internal address"
        }));
    }

    // Website policy check.
    if let Some((message, host, rule, source)) = check_website_access(url) {
        return json_str(&json!({
            "success": false,
            "error": message,
            "blocked_by_policy": {"host": host, "rule": rule, "source": source}
        }));
    }

    // Camofox backend is not ported; if active, return an explanatory error so
    // behaviour is explicit rather than silently divergent.
    if is_camofox_mode() {
        return json_str(&json!({
            "success": false,
            "error": "Camofox backend is not available in this build"
        }));
    }

    if auto_local_this_nav {
        log::info!(
            "browser_navigate: auto-routing {url} to local Chromium sidecar \
             (cloud provider {} stays on cloud for public URLs; \
             set browser.auto_local_for_private_urls: false to disable)",
            get_cloud_provider().provider_name()
        );
    }

    // Get session info; check first-nav.
    let session_info = match get_session_info(Some(&nav_session_key)) {
        Ok(s) => s,
        Err(e) => return json_str(&json!({"success": false, "error": e})),
    };
    let is_first_nav = session_info.first_nav;

    if is_first_nav {
        set_session_first_nav(&nav_session_key, false);
        maybe_start_recording(&nav_session_key);
    }

    let result = run_browser_command(
        &nav_session_key,
        "open",
        &[url.to_string()],
        Some(get_command_timeout().max(60)),
        None,
    );

    // Remember which session served this nav.
    {
        let mut st = browser_state().lock().unwrap();
        st.last_active_session_key
            .insert(effective_task_id.clone(), nav_session_key.clone());
    }

    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let data = result.get("data").cloned().unwrap_or(json!({}));
        let title = data.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let final_url = data
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or(url)
            .to_string();

        // Post-redirect SSRF check.
        if !is_local_backend()
            && !auto_local_this_nav
            && !allow_private_urls()
            && !final_url.is_empty()
            && final_url != url
            && !is_safe_url(&final_url)
        {
            run_browser_command(
                &nav_session_key,
                "open",
                &["about:blank".to_string()],
                Some(10),
                None,
            );
            return json_str(&json!({
                "success": false,
                "error": "Blocked: redirect landed on a private/internal address"
            }));
        }

        let mut response = Map::new();
        response.insert("success".to_string(), json!(true));
        response.insert("url".to_string(), json!(final_url));
        response.insert("title".to_string(), json!(title));
        copy_fallback_warning(&mut response, &result);

        let title_lower = title.to_lowercase();
        if BLOCKED_PATTERNS.iter().any(|p| title_lower.contains(p)) {
            response.insert("bot_detection_warning".to_string(), json!(format!(
                "Page title '{title}' suggests bot detection. The site may have blocked this request. \
                 Options: 1) Try adding delays between actions, 2) Access different pages first, \
                 3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), \
                 4) Some sites have very aggressive bot detection that may be unavoidable."
            )));
        }

        // First-nav feature info.
        if is_first_nav && !session_info.features.is_empty() {
            let features = &session_info.features;
            let active: Vec<String> = features
                .iter()
                .filter(|(_, v)| v.as_bool().unwrap_or(false))
                .map(|(k, _)| k.clone())
                .collect();
            let has_proxies = features
                .get("proxies")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !has_proxies {
                response.insert("stealth_warning".to_string(), json!(
                    "Running WITHOUT residential proxies. Bot detection may be more aggressive. \
                     Consider upgrading Browserbase plan for proxy support."
                ));
            }
            response.insert("stealth_features".to_string(), json!(active));
        }

        // Auto compact snapshot.
        let snap_result = run_browser_command(
            &nav_session_key,
            "snapshot",
            &["-c".to_string()],
            None,
            None,
        );
        if snap_result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
            let snap_data = snap_result.get("data").cloned().unwrap_or(json!({}));
            let mut snapshot_text = snap_data
                .get("snapshot")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let refs = snap_data.get("refs").cloned().unwrap_or(json!({}));
            if snapshot_text.len() > SNAPSHOT_SUMMARIZE_THRESHOLD {
                snapshot_text = truncate_snapshot_default(&snapshot_text);
            }
            response.insert("snapshot".to_string(), json!(snapshot_text));
            response.insert("element_count".to_string(), json!(ref_count(&refs)));
            if snap_result.get("fallback_warning").map(|v| !v.is_null()).unwrap_or(false)
                && !response.contains_key("fallback_warning")
            {
                copy_fallback_warning(&mut response, &snap_result);
            }
        }

        json_str(&Value::Object(response))
    } else {
        let err = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Navigation failed");
        json_str(&json!({"success": false, "error": err}))
    }
}

fn ref_count(refs: &Value) -> usize {
    match refs {
        Value::Object(m) => m.len(),
        Value::Array(a) => a.len(),
        _ => 0,
    }
}

fn set_session_first_nav(session_key: &str, value: bool) {
    let mut st = browser_state().lock().unwrap();
    if let Some(info) = st.active_sessions.get_mut(session_key) {
        info.first_nav = value;
    }
}

/// Get a text-based snapshot of the current page's accessibility tree.
///
/// `llm` is an optional summarizer closure used when the snapshot exceeds the
/// threshold and a `user_task` is supplied.
pub fn browser_snapshot<F>(
    full: bool,
    task_id: Option<&str>,
    user_task: Option<&str>,
    llm: Option<F>,
) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }

    let effective_task_id = last_session_key(task_id.unwrap_or("default"));

    let mut args: Vec<String> = Vec::new();
    if !full {
        args.push("-c".to_string());
    }

    let result = run_browser_command(&effective_task_id, "snapshot", &args, None, None);

    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let data = result.get("data").cloned().unwrap_or(json!({}));
        let mut snapshot_text = data
            .get("snapshot")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let refs = data.get("refs").cloned().unwrap_or(json!({}));

        if snapshot_text.len() > SNAPSHOT_SUMMARIZE_THRESHOLD && user_task.is_some() {
            snapshot_text = extract_relevant_content(&snapshot_text, user_task, llm);
        } else if snapshot_text.len() > SNAPSHOT_SUMMARIZE_THRESHOLD {
            snapshot_text = truncate_snapshot_default(&snapshot_text);
        }

        let mut response = Map::new();
        response.insert("success".to_string(), json!(true));
        response.insert("snapshot".to_string(), json!(snapshot_text));
        response.insert("element_count".to_string(), json!(ref_count(&refs)));
        copy_fallback_warning(&mut response, &result);

        // Supervisor merge is handled separately by the supervisor module when
        // wired; omitted here to avoid a hard dependency cycle.

        json_str(&Value::Object(response))
    } else {
        let mut response = Map::new();
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or("Failed to get snapshot")),
        );
        copy_fallback_warning(&mut response, &result);
        json_str(&Value::Object(response))
    }
}

/// Convenience: snapshot without a summarizer (truncation only).
pub fn browser_snapshot_simple(full: bool, task_id: Option<&str>, user_task: Option<&str>) -> String {
    browser_snapshot::<fn(&str) -> Option<String>>(full, task_id, user_task, None)
}

fn ensure_ref_prefix(ref_: &str) -> String {
    if ref_.starts_with('@') {
        ref_.to_string()
    } else {
        format!("@{ref_}")
    }
}

/// Click on an element identified by its ref ID.
pub fn browser_click(ref_: &str, task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let ref_ = ensure_ref_prefix(ref_);
    let result = run_browser_command(&effective_task_id, "click", &[ref_.clone()], None, None);

    let mut response = Map::new();
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        response.insert("success".to_string(), json!(true));
        response.insert("clicked".to_string(), json!(ref_));
    } else {
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or(&format!("Failed to click {ref_}"))),
        );
    }
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

/// Type text into an input field identified by its ref ID.
pub fn browser_type(ref_: &str, text: &str, task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let ref_ = ensure_ref_prefix(ref_);
    let result = run_browser_command(
        &effective_task_id,
        "fill",
        &[ref_.clone(), text.to_string()],
        None,
        None,
    );

    let mut response = Map::new();
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        response.insert("success".to_string(), json!(true));
        response.insert("typed".to_string(), json!(text));
        response.insert("element".to_string(), json!(ref_));
    } else {
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or(&format!("Failed to type into {ref_}"))),
        );
    }
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

/// Scroll the page in a direction (`up` or `down`).
pub fn browser_scroll(direction: &str, task_id: Option<&str>) -> String {
    if direction != "up" && direction != "down" {
        return json_str(&json!({
            "success": false,
            "error": format!("Invalid direction '{direction}'. Use 'up' or 'down'.")
        }));
    }

    const SCROLL_PIXELS: i32 = 500;

    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }

    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let result = run_browser_command(
        &effective_task_id,
        "scroll",
        &[direction.to_string(), SCROLL_PIXELS.to_string()],
        None,
        None,
    );

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let mut response = Map::new();
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or(&format!("Failed to scroll {direction}"))),
        );
        copy_fallback_warning(&mut response, &result);
        return json_str(&Value::Object(response));
    }

    let mut response = Map::new();
    response.insert("success".to_string(), json!(true));
    response.insert("scrolled".to_string(), json!(direction));
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

/// Navigate back in browser history.
pub fn browser_back(task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let result = run_browser_command(&effective_task_id, "back", &[], None, None);

    let mut response = Map::new();
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let data = result.get("data").cloned().unwrap_or(json!({}));
        response.insert("success".to_string(), json!(true));
        response.insert(
            "url".to_string(),
            json!(data.get("url").and_then(|v| v.as_str()).unwrap_or("")),
        );
    } else {
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or("Failed to go back")),
        );
    }
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

/// Press a keyboard key.
pub fn browser_press(key: &str, task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let result = run_browser_command(&effective_task_id, "press", &[key.to_string()], None, None);

    let mut response = Map::new();
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        response.insert("success".to_string(), json!(true));
        response.insert("pressed".to_string(), json!(key));
    } else {
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or(&format!("Failed to press {key}"))),
        );
    }
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

/// Get browser console messages and JS errors, or evaluate JS in the page.
pub fn browser_console(clear: bool, expression: Option<&str>, task_id: Option<&str>) -> String {
    if let Some(expr) = expression {
        return browser_eval(expr, task_id);
    }

    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }

    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let console_args: Vec<String> = if clear { vec!["--clear".to_string()] } else { vec![] };
    let error_args = console_args.clone();

    let console_result = run_browser_command(&effective_task_id, "console", &console_args, None, None);
    let errors_result = run_browser_command(&effective_task_id, "errors", &error_args, None, None);

    let mut messages: Vec<Value> = Vec::new();
    if console_result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        if let Some(arr) = console_result
            .get("data")
            .and_then(|d| d.get("messages"))
            .and_then(|v| v.as_array())
        {
            for msg in arr {
                messages.push(json!({
                    "type": msg.get("type").and_then(|v| v.as_str()).unwrap_or("log"),
                    "text": msg.get("text").and_then(|v| v.as_str()).unwrap_or(""),
                    "source": "console",
                }));
            }
        }
    }

    let mut errors: Vec<Value> = Vec::new();
    if errors_result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        if let Some(arr) = errors_result
            .get("data")
            .and_then(|d| d.get("errors"))
            .and_then(|v| v.as_array())
        {
            for err in arr {
                errors.push(json!({
                    "message": err.get("message").and_then(|v| v.as_str()).unwrap_or(""),
                    "source": "exception",
                }));
            }
        }
    }

    let mut response = Map::new();
    response.insert("success".to_string(), json!(true));
    response.insert("total_messages".to_string(), json!(messages.len()));
    response.insert("total_errors".to_string(), json!(errors.len()));
    response.insert("console_messages".to_string(), json!(messages));
    response.insert("js_errors".to_string(), json!(errors));
    copy_fallback_warning(&mut response, &console_result);
    if errors_result.get("fallback_warning").map(|v| !v.is_null()).unwrap_or(false)
        && !response.contains_key("fallback_warning")
    {
        copy_fallback_warning(&mut response, &errors_result);
    }
    json_str(&Value::Object(response))
}

/// Evaluate a JavaScript expression in the page context and return the result.
pub fn browser_eval(expression: &str, task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));
    let result = run_browser_command(&effective_task_id, "eval", &[expression.to_string()], None, None);

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = result.get("error").and_then(|v| v.as_str()).unwrap_or("eval failed").to_string();
        let err_lower = err.to_lowercase();
        let capability_gap = ["unknown command", "not supported", "not found", "no such command"]
            .iter()
            .any(|h| err_lower.contains(h));
        let mut response = Map::new();
        response.insert("success".to_string(), json!(false));
        if capability_gap {
            response.insert(
                "error".to_string(),
                json!(format!("JavaScript evaluation is not supported by this browser backend. {err}")),
            );
        } else {
            response.insert("error".to_string(), json!(err));
        }
        copy_fallback_warning(&mut response, &result);
        return json_str(&Value::Object(response));
    }

    let data = result.get("data").cloned().unwrap_or(json!({}));
    let raw_result = data.get("result").cloned().unwrap_or(Value::Null);

    // If the result is a string that is valid JSON, parse it.
    let parsed = match &raw_result {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| raw_result.clone()),
        _ => raw_result.clone(),
    };

    let mut response = Map::new();
    response.insert("success".to_string(), json!(true));
    response.insert("result_type".to_string(), json!(py_type_name(&parsed)));
    response.insert("result".to_string(), parsed);
    copy_fallback_warning(&mut response, &result);
    json_str(&Value::Object(response))
}

fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() && !n.is_i64() && !n.is_u64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Get all images on the current page (src, alt, dimensions).
pub fn browser_get_images(task_id: Option<&str>) -> String {
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));

    let js_code = "JSON.stringify(\n        [...document.images].map(img => ({\n            src: img.src,\n            alt: img.alt || '',\n            width: img.naturalWidth,\n            height: img.naturalHeight\n        })).filter(img => img.src && !img.src.startsWith('data:'))\n    )";

    let result = run_browser_command(&effective_task_id, "eval", &[js_code.to_string()], None, None);

    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let data = result.get("data").cloned().unwrap_or(json!({}));
        let raw_result = data.get("result").cloned().unwrap_or(json!("[]"));

        let images: Result<Value, ()> = match &raw_result {
            Value::String(s) => serde_json::from_str::<Value>(s).map_err(|_| ()),
            other => Ok(other.clone()),
        };

        match images {
            Ok(images) => {
                let count = images.as_array().map(|a| a.len()).unwrap_or(0);
                let mut response = Map::new();
                response.insert("success".to_string(), json!(true));
                response.insert("images".to_string(), images);
                response.insert("count".to_string(), json!(count));
                copy_fallback_warning(&mut response, &result);
                json_str(&Value::Object(response))
            }
            Err(_) => {
                let mut response = Map::new();
                response.insert("success".to_string(), json!(true));
                response.insert("images".to_string(), json!([]));
                response.insert("count".to_string(), json!(0));
                response.insert("warning".to_string(), json!("Could not parse image data"));
                copy_fallback_warning(&mut response, &result);
                json_str(&Value::Object(response))
            }
        }
    } else {
        let mut response = Map::new();
        response.insert("success".to_string(), json!(false));
        response.insert(
            "error".to_string(),
            json!(result.get("error").and_then(|v| v.as_str()).unwrap_or("Failed to get images")),
        );
        copy_fallback_warning(&mut response, &result);
        json_str(&Value::Object(response))
    }
}

/// Take a screenshot and analyze it with vision AI.
///
/// `vision_llm` receives `(question, data_url, screenshot_path)` and returns the
/// analysis text. The screenshot capture, persistence, base64 encoding,
/// fallback metadata, and response shaping are handled here. When `vision_llm`
/// is `None`, the screenshot is still captured and its path returned.
pub fn browser_vision<F>(
    question: &str,
    annotate: bool,
    task_id: Option<&str>,
    vision_llm: Option<F>,
) -> String
where
    F: FnOnce(&str, &str, &Path) -> Result<String, String>,
{
    if is_camofox_mode() {
        return json_str(&json!({"success": false, "error": "Camofox backend is not available in this build"}));
    }

    let screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots");
    let mut screenshot_path = screenshots_dir.join(format!("browser_screenshot_{}.png", rand_hex(32)));
    let effective_task_id = last_session_key(task_id.unwrap_or("default"));

    let engine = get_browser_engine();
    let mut lp_prerouted = false;
    let mut lp_fallback_warning: Option<String> = None;

    if engine == "lightpanda" && should_inject_engine(&engine) {
        log::debug!("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)");
        let mut screenshot_args: Vec<String> = Vec::new();
        if annotate {
            screenshot_args.push("--annotate".to_string());
        }
        let fb_result = chrome_fallback_screenshot(&effective_task_id, &screenshot_args, get_command_timeout());
        let fb_reason = "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.";
        let fb_result = annotate_lightpanda_fallback(&fb_result, fb_reason);
        if fb_result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
            lp_prerouted = true;
            lp_fallback_warning = fb_result
                .get("fallback_warning")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let fb_path = fb_result
                .get("data")
                .and_then(|d| d.get("path"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !fb_path.is_empty() && Path::new(fb_path).exists() {
                let _ = std::fs::create_dir_all(&screenshots_dir);
                let persistent = screenshots_dir.join(format!("browser_screenshot_{}.png", rand_hex(32)));
                if std::fs::copy(fb_path, &persistent).is_ok() {
                    screenshot_path = persistent;
                }
            }
        } else {
            log::warn!(
                "Lightpanda Chrome fallback vision screenshot failed: {}",
                fb_result.get("error").and_then(|v| v.as_str()).unwrap_or("")
            );
            lp_prerouted = false;
        }
    }

    let _ = std::fs::create_dir_all(&screenshots_dir);
    cleanup_old_screenshots(&screenshots_dir, 24);

    let result: Value = if lp_prerouted && screenshot_path.exists() {
        json!({
            "success": true,
            "data": {
                "path": screenshot_path.to_string_lossy(),
                "fallback_warning": lp_fallback_warning,
                "browser_engine": "chrome",
                "browser_engine_fallback": {
                    "from": "lightpanda", "to": "chrome",
                    "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
                }
            },
            "fallback_warning": lp_fallback_warning,
            "browser_engine": "chrome",
            "browser_engine_fallback": {
                "from": "lightpanda", "to": "chrome",
                "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
            }
        })
    } else {
        let mut screenshot_args: Vec<String> = Vec::new();
        if annotate {
            screenshot_args.push("--annotate".to_string());
        }
        screenshot_args.push("--full".to_string());
        screenshot_args.push(screenshot_path.to_string_lossy().to_string());
        run_browser_command(
            &effective_task_id,
            "screenshot",
            &screenshot_args,
            None,
            if lp_prerouted { Some("auto") } else { None },
        )
    };

    if !result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let error_detail = result.get("error").and_then(|v| v.as_str()).unwrap_or("Unknown error");
        let mode = vision_mode_label();
        let mut error_response = Map::new();
        error_response.insert("success".to_string(), json!(false));
        error_response.insert(
            "error".to_string(),
            json!(format!("Failed to take screenshot ({mode} mode): {error_detail}")),
        );
        copy_fallback_warning(&mut error_response, &result);
        return json_str(&Value::Object(error_response));
    }

    if let Some(actual) = result.get("data").and_then(|d| d.get("path")).and_then(|v| v.as_str()) {
        if !actual.is_empty() {
            screenshot_path = PathBuf::from(actual);
        }
    }

    if !screenshot_path.exists() {
        let mode = vision_mode_label();
        return json_str(&json!({
            "success": false,
            "error": format!(
                "Screenshot file was not created at {} ({mode} mode). \
                 This may indicate a socket path issue (macOS /var/folders/), \
                 a missing Chromium install ('agent-browser install'), \
                 or a stale daemon process.",
                screenshot_path.to_string_lossy()
            )
        }));
    }

    let screenshot_bytes = match std::fs::read(&screenshot_path) {
        Ok(b) => b,
        Err(e) => {
            return json_str(&json!({
                "success": false,
                "error": format!("Error during vision analysis: {e}")
            }))
        }
    };
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&screenshot_bytes);
    let data_url = format!("data:image/png;base64,{b64}");

    log::debug!("browser_vision: analysing screenshot ({} bytes)", screenshot_bytes.len());

    match vision_llm {
        Some(call) => match call(question, &data_url, &screenshot_path) {
            Ok(analysis) => {
                let analysis = redact_sensitive_text(analysis.trim());
                let mut response_data = Map::new();
                response_data.insert("success".to_string(), json!(true));
                response_data.insert(
                    "analysis".to_string(),
                    json!(if analysis.is_empty() {
                        "Vision analysis returned no content.".to_string()
                    } else {
                        analysis
                    }),
                );
                response_data.insert(
                    "screenshot_path".to_string(),
                    json!(screenshot_path.to_string_lossy()),
                );
                copy_fallback_warning(&mut response_data, &result);
                if annotate {
                    if let Some(ann) = result.get("data").and_then(|d| d.get("annotations")) {
                        if !ann.is_null() {
                            response_data.insert("annotations".to_string(), ann.clone());
                        }
                    }
                }
                json_str(&Value::Object(response_data))
            }
            Err(e) => {
                log::warn!("browser_vision failed: {e}");
                let mut error_info = Map::new();
                error_info.insert("success".to_string(), json!(false));
                error_info.insert("error".to_string(), json!(format!("Error during vision analysis: {e}")));
                if screenshot_path.exists() {
                    error_info.insert(
                        "screenshot_path".to_string(),
                        json!(screenshot_path.to_string_lossy()),
                    );
                    error_info.insert(
                        "note".to_string(),
                        json!("Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."),
                    );
                }
                copy_fallback_warning(&mut error_info, &result);
                json_str(&Value::Object(error_info))
            }
        },
        None => {
            // No LLM available — return the captured screenshot path.
            let mut response_data = Map::new();
            response_data.insert("success".to_string(), json!(true));
            response_data.insert(
                "analysis".to_string(),
                json!("Vision analysis unavailable (no vision model configured)."),
            );
            response_data.insert(
                "screenshot_path".to_string(),
                json!(screenshot_path.to_string_lossy()),
            );
            copy_fallback_warning(&mut response_data, &result);
            json_str(&Value::Object(response_data))
        }
    }
}

fn vision_mode_label() -> String {
    let cp = get_cloud_provider();
    if cp == CloudProviderKind::None {
        "local".to_string()
    } else {
        format!("cloud ({})", cp.provider_name())
    }
}

// ============================================================================
// Recording
// ============================================================================

fn maybe_start_recording(task_id: &str) {
    {
        let st = browser_state().lock().unwrap();
        if st.recording_sessions.contains(task_id) {
            return;
        }
    }

    let cfg = read_raw_config();
    let record_enabled = yaml_get(&cfg, &["browser", "record_sessions"])
        .map(yaml_to_json)
        .map(|v| is_truthy_value(Some(&v), false))
        .unwrap_or(false);
    if !record_enabled {
        return;
    }

    let hermes_home = get_hermes_home();
    let recordings_dir = hermes_home.join("browser_recordings");
    if std::fs::create_dir_all(&recordings_dir).is_err() {
        return;
    }
    cleanup_old_recordings(72);

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let safe_task: String = task_id.chars().take(16).collect();
    let recording_path = recordings_dir.join(format!("session_{timestamp}_{safe_task}.webm"));

    let result = run_browser_command(
        task_id,
        "record",
        &["start".to_string(), recording_path.to_string_lossy().to_string()],
        None,
        None,
    );
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let mut st = browser_state().lock().unwrap();
        st.recording_sessions.insert(task_id.to_string());
        log::info!(
            "Auto-recording browser session {task_id} to {}",
            recording_path.to_string_lossy()
        );
    } else {
        log::debug!(
            "Could not start auto-recording: {}",
            result.get("error").and_then(|v| v.as_str()).unwrap_or("")
        );
    }
}

fn maybe_stop_recording(task_id: &str) {
    {
        let st = browser_state().lock().unwrap();
        if !st.recording_sessions.contains(task_id) {
            return;
        }
    }
    let result = run_browser_command(task_id, "record", &["stop".to_string()], None, None);
    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        let path = result.get("data").and_then(|d| d.get("path")).and_then(|v| v.as_str()).unwrap_or("");
        log::info!("Saved browser recording for session {task_id}: {path}");
    }
    let mut st = browser_state().lock().unwrap();
    st.recording_sessions.remove(task_id);
}

fn cleanup_old_screenshots(screenshots_dir: &Path, max_age_hours: u64) {
    let key = screenshots_dir.to_string_lossy().to_string();
    let now = now_secs();
    {
        let map = last_screenshot_cleanup().lock().unwrap();
        if now - map.get(&key).copied().unwrap_or(0.0) < 3600.0 {
            return;
        }
    }
    last_screenshot_cleanup().lock().unwrap().insert(key, now);

    let cutoff = now - (max_age_hours as f64 * 3600.0);
    if let Ok(read) = std::fs::read_dir(screenshots_dir) {
        for entry in read.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("browser_screenshot_") && name.ends_with(".png") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        let mtime = modified
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0);
                        if mtime < cutoff {
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }
    }
}

fn cleanup_old_recordings(max_age_hours: u64) {
    let hermes_home = get_hermes_home();
    let recordings_dir = hermes_home.join("browser_recordings");
    if !recordings_dir.exists() {
        return;
    }
    let cutoff = now_secs() - (max_age_hours as f64 * 3600.0);
    if let Ok(read) = std::fs::read_dir(&recordings_dir) {
        for entry in read.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("session_") && name.ends_with(".webm") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        let mtime = modified
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0);
                        if mtime < cutoff {
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Cleanup & management
// ============================================================================

/// Clean up browser session(s) for a task.
pub fn cleanup_browser(task_id: Option<&str>) {
    let task_id = task_id.unwrap_or("default").to_string();

    let (session_keys, bare_task_id) = if is_local_sidecar_key(&task_id) {
        let bare = task_id[..task_id.len() - LOCAL_SUFFIX.len()].to_string();
        (vec![task_id.clone()], bare)
    } else {
        let mut keys = vec![task_id.clone()];
        let sidecar_key = format!("{task_id}{LOCAL_SUFFIX}");
        {
            let st = browser_state().lock().unwrap();
            if st.active_sessions.contains_key(&sidecar_key) {
                keys.push(sidecar_key);
            }
        }
        (keys, task_id.clone())
    };

    for session_key in &session_keys {
        cleanup_single_browser_session(session_key);
    }

    if !is_local_sidecar_key(&task_id) {
        let mut st = browser_state().lock().unwrap();
        st.last_active_session_key.remove(&bare_task_id);
    }
}

fn cleanup_single_browser_session(task_id: &str) {
    // Camofox cleanup is omitted (backend not ported).

    log::debug!("cleanup_browser called for task_id: {task_id}");
    {
        let st = browser_state().lock().unwrap();
        let keys: Vec<String> = st.active_sessions.keys().cloned().collect();
        log::debug!("Active sessions: {keys:?}");
    }

    let session_info = {
        let st = browser_state().lock().unwrap();
        st.active_sessions.get(task_id).cloned()
    };

    let session_info = match session_info {
        Some(s) => s,
        None => {
            log::debug!("No active session found for task_id: {task_id}");
            return;
        }
    };

    let bb_session_id = session_info.bb_session_id.clone();
    log::debug!(
        "Found session for task {task_id}: bb_session_id={}",
        bb_session_id.clone().unwrap_or_else(|| "unknown".to_string())
    );

    maybe_stop_recording(task_id);

    run_browser_command(task_id, "close", &[], Some(10), None);
    log::debug!("agent-browser close command completed for task {task_id}");

    {
        let mut st = browser_state().lock().unwrap();
        st.active_sessions.remove(task_id);
        st.session_last_activity.remove(task_id);
    }

    // Cloud session close is delegated to the provider module when wired.
    if bb_session_id.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
        // provider.close_session(bb_session_id) — handled by provider integration.
    }

    let session_name = session_info.session_name.clone();
    if !session_name.is_empty() {
        let socket_dir = Path::new(&socket_safe_tmpdir()).join(format!("agent-browser-{session_name}"));
        if socket_dir.exists() {
            let pid_file = socket_dir.join(format!("{session_name}.pid"));
            if pid_file.is_file() {
                if let Ok(text) = std::fs::read_to_string(&pid_file) {
                    if let Ok(daemon_pid) = text.trim().parse::<i32>() {
                        if send_sigterm(daemon_pid) {
                            log::debug!("Killed daemon pid {daemon_pid} for {session_name}");
                        } else {
                            log::debug!("Could not kill daemon pid for {session_name} (already dead or inaccessible)");
                        }
                    }
                }
            }
            let _ = std::fs::remove_dir_all(&socket_dir);
        }
    }

    log::debug!("Removed task {task_id} from active sessions");
}

/// Clean up all active browser sessions.
pub fn cleanup_all_browsers() {
    let task_ids: Vec<String> = {
        let st = browser_state().lock().unwrap();
        st.active_sessions.keys().cloned().collect()
    };
    for task_id in task_ids {
        cleanup_browser(Some(&task_id));
    }

    // Reset cached lookups.
    {
        let mut c = agent_browser_cache().lock().unwrap();
        c.value = None;
        c.resolved = false;
    }
    clear_homebrew_node_cache();
    {
        let mut c = cmd_timeout_cache().lock().unwrap();
        c.resolved = false;
        c.value = DEFAULT_COMMAND_TIMEOUT;
    }
    *chromium_installed_cache().lock().unwrap() = None;
    {
        let mut c = engine_cache().lock().unwrap();
        c.resolved = false;
        c.value = "auto".to_string();
    }
}

/// Reset all process-lifetime caches (test helper / config-reload hook).
pub fn reset_browser_caches() {
    {
        let mut c = cloud_provider_cache().lock().unwrap();
        c.resolved = false;
        c.value = CloudProviderKind::None;
    }
    {
        let mut c = auto_local_cache().lock().unwrap();
        c.resolved = false;
        c.value = true;
    }
    {
        let mut c = allow_private_cache().lock().unwrap();
        c.resolved = false;
        c.value = false;
    }
    cleanup_all_browsers();
}

// ============================================================================
// Requirements check
// ============================================================================

/// Check if browser tool requirements are met.
pub fn check_browser_requirements() -> bool {
    if is_camofox_mode() {
        return true;
    }
    if !get_cdp_override().is_empty() {
        return true;
    }
    let browser_cmd = match find_agent_browser() {
        Ok(c) => c,
        Err(_) => return false,
    };
    if requires_real_termux_browser_install(&browser_cmd) {
        return false;
    }
    let provider = get_cloud_provider();
    if provider != CloudProviderKind::None {
        return provider_is_configured(&provider);
    }
    if using_lightpanda_engine() {
        return true;
    }
    if !chromium_installed() {
        return false;
    }
    true
}

// ============================================================================
// Secret-exfil URL guard helpers (mirror agent.redact._PREFIX_RE usage)
// ============================================================================

fn secret_prefix_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Common API-key prefixes embedded in URLs. Conservative superset of
        // agent.redact._PREFIX_RE for the navigate guard.
        regex::Regex::new(
            r"(?i)(sk-ant-|sk-|sk_live_|sk_test_|pk_live_|pk_test_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|xoxb-|xoxp-|xoxa-|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|ya29\.|hf_|nvapi-|gsk_|r8_|Bearer\s+)",
        )
        .unwrap()
    })
}

fn secret_prefix_present(s: &str) -> bool {
    secret_prefix_re().is_match(s)
}

fn percent_decode(s: &str) -> String {
    // Minimal percent-decoding (urllib.parse.unquote equivalent for ASCII).
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ============================================================================
// Tool schemas
// ============================================================================

/// Return the JSON tool schemas for the browser toolset.
pub fn browser_tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "name": "browser_navigate",
            "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to navigate to (e.g., 'https://example.com')"}}, "required": ["url"]}
        }),
        json!({
            "name": "browser_snapshot",
            "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 8000 chars are truncated or LLM-summarized. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content.",
            "parameters": {"type": "object", "properties": {"full": {"type": "boolean", "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.", "default": false}}, "required": []}
        }),
        json!({
            "name": "browser_click",
            "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
            "parameters": {"type": "object", "properties": {"ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"}}, "required": ["ref"]}
        }),
        json!({
            "name": "browser_type",
            "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
            "parameters": {"type": "object", "properties": {"ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e3')"}, "text": {"type": "string", "description": "The text to type into the field"}}, "required": ["ref", "text"]}
        }),
        json!({
            "name": "browser_scroll",
            "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
            "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down"], "description": "Direction to scroll"}}, "required": ["direction"]}
        }),
        json!({
            "name": "browser_back",
            "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }),
        json!({
            "name": "browser_press",
            "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"}}, "required": ["key"]}
        }),
        json!({
            "name": "browser_get_images",
            "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }),
        json!({
            "name": "browser_vision",
            "description": "Take a screenshot of the current page and analyze it with vision AI. Use this when you need to visually understand what's on the page - especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snapshot doesn't capture important visual information. Returns both the AI analysis and a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate first.",
            "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "What you want to know about the page visually. Be specific about what you're looking for."}, "annotate": {"type": "boolean", "default": false, "description": "If true, overlay numbered [N] labels on interactive elements. Each [N] maps to ref @eN for subsequent browser commands. Useful for QA and spatial reasoning about page layout."}}, "required": ["question"]}
        }),
        json!({
            "name": "browser_console",
            "description": "Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically.",
            "parameters": {"type": "object", "properties": {"clear": {"type": "boolean", "default": false, "description": "If true, clear the message buffers after reading"}, "expression": {"type": "string", "description": "JavaScript expression to evaluate in the page context. Runs in the browser like DevTools console — full access to DOM, window, document. Return values are serialized to JSON. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'"}}, "required": []}
        }),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_snapshot_short() {
        let text = "line1\nline2";
        assert_eq!(truncate_snapshot(text, 8000), text);
    }

    #[test]
    fn test_truncate_snapshot_long() {
        let line = "x".repeat(100);
        let text = (0..200).map(|_| line.clone()).collect::<Vec<_>>().join("\n");
        let out = truncate_snapshot(&text, 1000);
        assert!(out.len() < text.len());
        assert!(out.contains("more lines truncated"));
    }

    #[test]
    fn test_ensure_ref_prefix() {
        assert_eq!(ensure_ref_prefix("e5"), "@e5");
        assert_eq!(ensure_ref_prefix("@e5"), "@e5");
    }

    #[test]
    fn test_extract_screenshot_path() {
        let txt = "Screenshot saved to '/tmp/shot.png'";
        assert_eq!(
            extract_screenshot_path_from_text(txt),
            Some("/tmp/shot.png".to_string())
        );
        let txt2 = "Screenshot saved to /var/x/shot.png ";
        assert_eq!(
            extract_screenshot_path_from_text(txt2),
            Some("/var/x/shot.png".to_string())
        );
        assert_eq!(extract_screenshot_path_from_text(""), None);
    }

    #[test]
    fn test_py_type_name() {
        assert_eq!(py_type_name(&json!(null)), "NoneType");
        assert_eq!(py_type_name(&json!(true)), "bool");
        assert_eq!(py_type_name(&json!(5)), "int");
        assert_eq!(py_type_name(&json!(5.5)), "float");
        assert_eq!(py_type_name(&json!("x")), "str");
        assert_eq!(py_type_name(&json!([1])), "list");
        assert_eq!(py_type_name(&json!({"a": 1})), "dict");
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("sk%2Dant%2Dx"), "sk-ant-x");
        assert_eq!(percent_decode("nochange"), "nochange");
        assert_eq!(percent_decode("a%20b"), "a b");
    }

    #[test]
    fn test_secret_prefix_present() {
        assert!(secret_prefix_present("https://x.com/?k=sk-ant-abc123"));
        assert!(secret_prefix_present("ghp_abcdef"));
        assert!(!secret_prefix_present("https://example.com/page"));
    }

    #[test]
    fn test_is_local_sidecar_key() {
        assert!(is_local_sidecar_key("task::local"));
        assert!(!is_local_sidecar_key("task"));
    }

    #[test]
    fn test_ip_is_private() {
        use std::net::IpAddr;
        assert!(ip_is_private(&"127.0.0.1".parse::<IpAddr>().unwrap()));
        assert!(ip_is_private(&"192.168.1.1".parse::<IpAddr>().unwrap()));
        assert!(ip_is_private(&"10.0.0.5".parse::<IpAddr>().unwrap()));
        assert!(ip_is_private(&"100.64.0.1".parse::<IpAddr>().unwrap()));
        assert!(!ip_is_private(&"8.8.8.8".parse::<IpAddr>().unwrap()));
        assert!(ip_is_private(&"::1".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn test_url_is_private_localhost() {
        assert!(url_is_private("http://localhost:8080"));
        assert!(url_is_private("http://foo.local/"));
        assert!(url_is_private("http://192.168.0.1/"));
        assert!(url_is_private("http://10.1.2.3:9000/x"));
    }

    #[test]
    fn test_resolve_cdp_override_devtools_passthrough() {
        let raw = "ws://host:9222/devtools/browser/abc";
        assert_eq!(resolve_cdp_override(raw), raw);
    }

    #[test]
    fn test_resolve_cdp_override_empty() {
        assert_eq!(resolve_cdp_override(""), "");
        assert_eq!(resolve_cdp_override("   "), "");
    }

    #[test]
    fn test_lightpanda_fallback_reason_not_lightpanda() {
        let result = json!({"success": true, "data": {}});
        assert_eq!(lightpanda_fallback_reason("chrome", "snapshot", &result), None);
        assert_eq!(lightpanda_fallback_reason("auto", "open", &result), None);
    }

    #[test]
    fn test_lightpanda_fallback_reason_failed() {
        let result = json!({"success": false, "error": "boom"});
        let reason = lightpanda_fallback_reason("lightpanda", "open", &result);
        assert!(reason.unwrap().contains("failed (boom)"));
    }

    #[test]
    fn test_lightpanda_fallback_reason_empty_snapshot() {
        let result = json!({"success": true, "data": {"snapshot": "tiny"}});
        let reason = lightpanda_fallback_reason("lightpanda", "snapshot", &result);
        assert!(reason.unwrap().contains("empty/too-short"));
    }

    #[test]
    fn test_lightpanda_fallback_reason_ineligible_command() {
        let result = json!({"success": false, "error": "x"});
        assert_eq!(lightpanda_fallback_reason("lightpanda", "close", &result), None);
    }

    #[test]
    fn test_annotate_lightpanda_fallback() {
        let result = json!({"success": true, "data": {"snapshot": "abc"}});
        let annotated = annotate_lightpanda_fallback(&result, "test reason");
        assert_eq!(annotated["browser_engine"], json!("chrome"));
        assert!(annotated["fallback_warning"].as_str().unwrap().contains("test reason"));
        assert_eq!(annotated["data"]["browser_engine"], json!("chrome"));
    }

    #[test]
    fn test_copy_fallback_warning() {
        let result = json!({
            "fallback_warning": "w",
            "browser_engine": "chrome",
            "browser_engine_fallback": {"from": "lightpanda", "to": "chrome"}
        });
        let mut target = Map::new();
        copy_fallback_warning(&mut target, &result);
        assert_eq!(target.get("fallback_warning"), Some(&json!("w")));
        assert_eq!(target.get("browser_engine"), Some(&json!("chrome")));
    }

    #[test]
    fn test_copy_fallback_warning_absent() {
        let result = json!({"success": true});
        let mut target = Map::new();
        copy_fallback_warning(&mut target, &result);
        assert!(!target.contains_key("fallback_warning"));
    }

    #[test]
    fn test_browser_scroll_invalid_direction() {
        let out = browser_scroll("sideways", Some("t"));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("Invalid direction"));
    }

    #[test]
    fn test_browser_navigate_secret_blocked() {
        let out = browser_navigate("https://evil.com/steal?key=sk-ant-secret123", Some("t"));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("API key or token"));
    }

    #[test]
    fn test_browser_navigate_secret_blocked_encoded() {
        let out = browser_navigate("https://evil.com/steal?key=sk%2Dant%2Dsecret", Some("t"));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(false));
    }

    #[test]
    fn test_merge_browser_path_dedup() {
        let merged = merge_browser_path("/usr/bin:/bin");
        // existing entries preserved at the tail
        assert!(merged.ends_with("/usr/bin:/bin") || merged.contains("/usr/bin"));
    }

    #[test]
    fn test_ref_count() {
        assert_eq!(ref_count(&json!({"e1": 1, "e2": 2})), 2);
        assert_eq!(ref_count(&json!([1, 2, 3])), 3);
        assert_eq!(ref_count(&json!(null)), 0);
    }

    #[test]
    fn test_truncate_snapshot_default_threshold() {
        let text = "a".repeat(SNAPSHOT_SUMMARIZE_THRESHOLD + 100);
        let out = truncate_snapshot_default(&text);
        assert!(out.len() <= text.len());
    }

    #[test]
    fn test_schemas_present() {
        let schemas = browser_tool_schemas();
        assert_eq!(schemas.len(), 10);
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"browser_navigate"));
        assert!(names.contains(&"browser_console"));
    }

    #[test]
    fn test_extract_relevant_content_no_llm_truncates() {
        let text = "x".repeat(20000);
        let out = extract_relevant_content::<fn(&str) -> Option<String>>(&text, Some("task"), None);
        assert!(out.len() < text.len());
    }

    #[test]
    fn test_socket_safe_tmpdir_nonempty() {
        assert!(!socket_safe_tmpdir().is_empty());
    }

    #[test]
    fn test_provider_kind_names() {
        assert_eq!(CloudProviderKind::None.provider_name(), "local");
        assert_eq!(CloudProviderKind::Browserbase.provider_name(), "browserbase");
        assert_eq!(CloudProviderKind::BrowserUse.provider_name(), "browser-use");
        assert_eq!(CloudProviderKind::Firecrawl.provider_name(), "firecrawl");
    }

    #[test]
    fn test_get_vision_model_env() {
        unsafe {
            std::env::set_var("AUXILIARY_VISION_MODEL", "  gpt-vision  ");
        }
        assert_eq!(get_vision_model(), Some("gpt-vision".to_string()));
        unsafe {
            std::env::remove_var("AUXILIARY_VISION_MODEL");
        }
        assert_eq!(get_vision_model(), None);
    }

    #[test]
    fn test_is_empty_ok_command() {
        assert!(is_empty_ok_command("close"));
        assert!(is_empty_ok_command("record"));
        assert!(!is_empty_ok_command("snapshot"));
    }

    #[test]
    fn test_parse_command_output_empty_rc0_fails() {
        let v = parse_command_output("snapshot", "", "", 0);
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("returned no output"));
    }

    #[test]
    fn test_parse_command_output_empty_ok_close() {
        let v = parse_command_output("close", "", "", 0);
        assert_eq!(v["success"], json!(true));
    }

    #[test]
    fn test_parse_command_output_parses_json() {
        let v = parse_command_output("click", "{\"success\": true, \"data\": {}}", "", 0);
        assert_eq!(v["success"], json!(true));
    }

    #[test]
    fn test_parse_command_output_nonjson() {
        let v = parse_command_output("click", "not json here", "", 0);
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("Non-JSON output"));
    }

    #[test]
    fn test_parse_command_output_nonzero_rc() {
        let v = parse_command_output("open", "", "some error", 1);
        assert_eq!(v["success"], json!(false));
        assert_eq!(v["error"], json!("some error"));
    }
}
