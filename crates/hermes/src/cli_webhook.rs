//! hermes webhook — manage dynamic webhook subscriptions from the CLI.
//!
//! Usage:
//!     hermes webhook subscribe <name> [options]
//!     hermes webhook list
//!     hermes webhook remove <name>
//!     hermes webhook test <name> [--payload '{"key": "value"}']
//!
//! Subscriptions persist to ~/.hermes/webhook_subscriptions.json and are
//! hot-reloaded by the webhook adapter without a gateway restart.
//!
//! Native Rust port of `hermes_cli/webhook.py`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde_json::{Map, Value};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const SUBSCRIPTIONS_FILENAME: &str = "webhook_subscriptions.json";

// ---------------------------------------------------------------------------
// Argument bag
// ---------------------------------------------------------------------------

/// Parsed arguments for the `hermes webhook` subcommand.
///
/// The CLI front-end is responsible for populating this from clap/argparse;
/// the orchestration of behaviour lives in [`webhook_command`].
#[derive(Debug, Clone, Default)]
pub struct WebhookArgs {
    /// Subcommand action: subscribe|add|list|ls|remove|rm|test (or None).
    pub webhook_action: Option<String>,
    /// Subscription name (used by subscribe/remove/test).
    pub name: String,
    /// Optional HMAC secret (subscribe); generated if empty.
    pub secret: Option<String>,
    /// Comma-separated event list (subscribe).
    pub events: Option<String>,
    /// Description (subscribe).
    pub description: Option<String>,
    /// Prompt / message body (subscribe).
    pub prompt: Option<String>,
    /// Comma-separated skills list (subscribe).
    pub skills: Option<String>,
    /// Delivery target (subscribe); defaults to "log".
    pub deliver: Option<String>,
    /// Whether direct-only delivery is requested (subscribe).
    pub deliver_only: bool,
    /// Delivery chat id (subscribe).
    pub deliver_chat_id: Option<String>,
    /// Test payload (test); defaults to a canned JSON body.
    pub payload: Option<String>,
}

// ---------------------------------------------------------------------------
// Pluggable dependencies
//
// In the Python original these are module-level imports from hermes_constants,
// utils and hermes_cli.config. To keep this module decoupled (and unit
// testable without a real ~/.hermes), the side-effecting bits are routed
// through a `WebhookEnv` trait. Production code uses `RealEnv`.
// ---------------------------------------------------------------------------

/// Side-effecting environment dependencies for the webhook CLI.
pub trait WebhookEnv {
    /// `~/.hermes` (HERMES_HOME).
    fn hermes_home(&self) -> PathBuf;
    /// Display form of the hermes home directory (e.g. `~/.hermes`).
    fn display_hermes_home(&self) -> String;
    /// Loaded webhook platform config block (`platforms.webhook`), or `{}`.
    fn webhook_config(&self) -> Value;
    /// Emit a line of output (stdout in production).
    fn emit(&mut self, line: &str);
}

/// Production environment, wired to the real hermes-core helpers.
pub struct RealEnv;

impl WebhookEnv for RealEnv {
    fn hermes_home(&self) -> PathBuf {
        get_hermes_home()
    }

    fn display_hermes_home(&self) -> String {
        display_hermes_home()
    }

    fn webhook_config(&self) -> Value {
        get_webhook_config_real()
    }

    fn emit(&mut self, line: &str) {
        println!("{line}");
    }
}

/// User home directory, mirroring `hermes_constants` (falls back to `.`).
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Return the Hermes home directory (default: `~/.hermes`).
///
/// Reads the `HERMES_HOME` env var, falls back to `~/.hermes`. Local mirror of
/// `hermes_core::mod_hermes_constants::get_hermes_home` (a private module).
fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    home_dir().join(".hermes")
}

/// User-friendly display string for the current HERMES_HOME (uses `~/`).
fn display_hermes_home() -> String {
    let home = get_hermes_home();
    let user_home = home_dir();
    match home.strip_prefix(&user_home) {
        Ok(rel) => format!("~/{}", rel.to_string_lossy()),
        Err(_) => home.to_string_lossy().into_owned(),
    }
}

/// Atomic-replace `tmp_path` onto `target`, resolving a symlinked target first.
///
/// Local mirror of `hermes_core::mod_utils::atomic_replace` (a private module).
fn atomic_replace(tmp_path: &Path, target: &Path) -> std::io::Result<PathBuf> {
    let is_link = std::fs::symlink_metadata(target)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let real_path: PathBuf = if is_link {
        std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf())
    } else {
        target.to_path_buf()
    };
    std::fs::rename(tmp_path, &real_path)?;
    Ok(real_path)
}

/// Load `platforms.webhook` from `~/.hermes/config.yaml`, converted to a
/// serde_json::Value mapping. Returns `{}` (an empty object) if unavailable.
fn get_webhook_config_real() -> Value {
    let config_path = get_hermes_home().join("config.yaml");
    let text = match std::fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(_) => return Value::Object(Map::new()),
    };
    let cfg: serde_yaml::Value = match serde_yaml::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Value::Object(Map::new()),
    };
    let block = cfg
        .get("platforms")
        .and_then(|p| p.get("webhook"));
    match block {
        Some(v) => yaml_to_json(v),
        None => Value::Object(Map::new()),
    }
}

/// Convert a serde_yaml::Value into a serde_json::Value (best-effort, lossy on
/// non-string keys, matching how the Python code only ever indexes by string).
fn yaml_to_json(v: &serde_yaml::Value) -> Value {
    match v {
        serde_yaml::Value::Null => Value::Null,
        serde_yaml::Value::Bool(b) => Value::Bool(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(u) = n.as_u64() {
                Value::from(u)
            } else if let Some(f) = n.as_f64() {
                Value::from(f)
            } else {
                Value::Null
            }
        }
        serde_yaml::Value::String(s) => Value::String(s.clone()),
        serde_yaml::Value::Sequence(seq) => {
            Value::Array(seq.iter().map(yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let mut out = Map::new();
            for (k, val) in map {
                let key = match k {
                    serde_yaml::Value::String(s) => s.clone(),
                    serde_yaml::Value::Number(n) => n.to_string(),
                    serde_yaml::Value::Bool(b) => b.to_string(),
                    _ => continue,
                };
                out.insert(key, yaml_to_json(val));
            }
            Value::Object(out)
        }
        serde_yaml::Value::Tagged(t) => yaml_to_json(&t.value),
    }
}

// ---------------------------------------------------------------------------
// Subscription store helpers
// ---------------------------------------------------------------------------

fn subscriptions_path(env: &dyn WebhookEnv) -> PathBuf {
    env.hermes_home().join(SUBSCRIPTIONS_FILENAME)
}

/// Load subscriptions from disk. Returns an empty map on any error (mirrors the
/// Python `try/except -> {}` and the `isinstance(data, dict)` guard).
///
/// Uses a BTreeMap purely so unit tests have deterministic ordering; the Python
/// version preserves insertion order, but the on-disk JSON is what matters and
/// callers iterate for display only.
pub fn load_subscriptions(env: &dyn WebhookEnv) -> Map<String, Value> {
    let path = subscriptions_path(env);
    if !path.exists() {
        return Map::new();
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        },
        Err(_) => Map::new(),
    }
}

/// Persist subscriptions atomically (write to `.tmp`, then atomic_replace).
pub fn save_subscriptions(
    env: &dyn WebhookEnv,
    subs: &Map<String, Value>,
) -> std::io::Result<()> {
    let path = subscriptions_path(env);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("tmp");
    let body = serde_json::to_string_pretty(&Value::Object(subs.clone()))
        .unwrap_or_else(|_| "{}".to_string());
    std::fs::write(&tmp_path, body)?;
    atomic_replace(&tmp_path, &path)?;
    Ok(())
}

fn is_webhook_enabled(env: &dyn WebhookEnv) -> bool {
    let cfg = env.webhook_config();
    matches!(cfg.get("enabled"), Some(v) if truthy(v))
}

/// Python `bool(...)` truthiness for the values that can land in `enabled`.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn webhook_base_url(env: &dyn WebhookEnv) -> String {
    let cfg = env.webhook_config();
    let extra = cfg.get("extra").and_then(|v| v.as_object());
    let host = extra
        .and_then(|e| e.get("host"))
        .and_then(|v| v.as_str())
        .unwrap_or("0.0.0.0");
    let port: i64 = extra
        .and_then(|e| e.get("port"))
        .and_then(json_to_i64)
        .unwrap_or(8644);
    let display_host = if host == "0.0.0.0" { "localhost" } else { host };
    format!("http://{display_host}:{port}")
}

fn json_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn setup_hint(env: &dyn WebhookEnv) -> String {
    let dhh = env.display_hermes_home();
    format!(
        "
  Webhook platform is not enabled. To set it up:

  1. Run the gateway setup wizard:
     hermes gateway setup

  2. Or manually add to {dhh}/config.yaml:
     platforms:
       webhook:
         enabled: true
         extra:
           host: \"0.0.0.0\"
           port: 8644
           secret: \"your-global-hmac-secret\"

  3. Or set environment variables in {dhh}/.env:
     WEBHOOK_ENABLED=true
     WEBHOOK_PORT=8644
     WEBHOOK_SECRET=your-global-secret

  Then start the gateway: hermes gateway run
"
    )
}

/// Check webhook is enabled. Print setup guide and return false if not.
pub fn require_webhook_enabled(env: &mut dyn WebhookEnv) -> bool {
    if is_webhook_enabled(env) {
        return true;
    }
    let hint = setup_hint(env);
    env.emit(&hint);
    false
}

// ---------------------------------------------------------------------------
// secrets.token_urlsafe(32) equivalent
// ---------------------------------------------------------------------------

/// Mirror of Python `secrets.token_urlsafe(32)`: 32 random bytes, base64
/// url-safe encoded with padding stripped.
pub fn token_urlsafe(nbytes: usize) -> String {
    let mut buf = vec![0u8; nbytes];
    if getrandom::fill(&mut buf).is_err() {
        // Extremely unlikely fallback: seed from time + address entropy.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut x = seed as u64 ^ (&buf as *const _ as u64);
        for b in buf.iter_mut() {
            // xorshift64
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
    }
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&buf)
}

// ---------------------------------------------------------------------------
// HMAC signature
// ---------------------------------------------------------------------------

/// `sha256=<hex>` GitHub-style signature of `payload` with `secret`.
pub fn sign_payload(secret: &str, payload: &[u8]) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(payload);
    let bytes = mac.finalize().into_bytes();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256={hex}")
}

// ---------------------------------------------------------------------------
// Command entry point
// ---------------------------------------------------------------------------

/// Entry point for the `hermes webhook` subcommand.
pub fn webhook_command(env: &mut dyn WebhookEnv, args: &WebhookArgs) {
    let sub = match &args.webhook_action {
        Some(s) if !s.is_empty() => s.clone(),
        _ => {
            env.emit("Usage: hermes webhook {subscribe|list|remove|test}");
            env.emit("Run 'hermes webhook --help' for details.");
            return;
        }
    };

    if !require_webhook_enabled(env) {
        return;
    }

    match sub.as_str() {
        "subscribe" | "add" => cmd_subscribe(env, args),
        "list" | "ls" => cmd_list(env, args),
        "remove" | "rm" => cmd_remove(env, args),
        "test" => cmd_test(env, args),
        _ => {}
    }
}

/// Validate `^[a-z0-9][a-z0-9_-]*$` without pulling in the regex engine — the
/// shape is fixed and trivial to match by hand.
fn is_valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

fn split_csv(raw: &Option<String>) -> Vec<String> {
    match raw {
        Some(s) if !s.is_empty() => s.split(',').map(|e| e.trim().to_string()).collect(),
        _ => Vec::new(),
    }
}

fn opt_or_empty(v: &Option<String>) -> String {
    v.clone().unwrap_or_default()
}

fn cmd_subscribe(env: &mut dyn WebhookEnv, args: &WebhookArgs) {
    let name = args.name.trim().to_lowercase().replace(' ', "-");
    if !is_valid_name(&name) {
        env.emit(&format!(
            "Error: Invalid name '{name}'. Use lowercase alphanumeric with hyphens/underscores."
        ));
        return;
    }

    let mut subs = load_subscriptions(env);
    let is_update = subs.contains_key(&name);

    let secret = match &args.secret {
        Some(s) if !s.is_empty() => s.clone(),
        _ => token_urlsafe(32),
    };
    let events = split_csv(&args.events);

    let description = match &args.description {
        Some(s) if !s.is_empty() => s.clone(),
        _ => format!("Agent-created subscription: {name}"),
    };
    let deliver = match &args.deliver {
        Some(s) if !s.is_empty() => s.clone(),
        _ => "log".to_string(),
    };

    let mut route = Map::new();
    route.insert("description".into(), Value::String(description));
    route.insert(
        "events".into(),
        Value::Array(events.iter().cloned().map(Value::String).collect()),
    );
    route.insert("secret".into(), Value::String(secret.clone()));
    route.insert("prompt".into(), Value::String(opt_or_empty(&args.prompt)));
    route.insert(
        "skills".into(),
        Value::Array(split_csv(&args.skills).into_iter().map(Value::String).collect()),
    );
    route.insert("deliver".into(), Value::String(deliver.clone()));
    route.insert("created_at".into(), Value::String(utc_now_iso()));

    if args.deliver_only {
        if deliver == "log" {
            env.emit(
                "Error: --deliver-only requires --deliver to be a real target \
(telegram, discord, slack, github_comment, etc.) — not 'log'.",
            );
            return;
        }
        route.insert("deliver_only".into(), Value::Bool(true));
    }

    if let Some(chat_id) = &args.deliver_chat_id {
        if !chat_id.is_empty() {
            let mut extra = Map::new();
            extra.insert("chat_id".into(), Value::String(chat_id.clone()));
            route.insert("deliver_extra".into(), Value::Object(extra));
        }
    }

    subs.insert(name.clone(), Value::Object(route.clone()));
    if let Err(e) = save_subscriptions(env, &subs) {
        env.emit(&format!("  Error saving subscription: {e}"));
        return;
    }

    let base_url = webhook_base_url(env);
    let status = if is_update { "Updated" } else { "Created" };

    env.emit(&format!("\n  {status} webhook subscription: {name}"));
    env.emit(&format!("  URL:    {base_url}/webhooks/{name}"));
    env.emit(&format!("  Secret: {secret}"));
    if events.is_empty() {
        env.emit("  Events: (all)");
    } else {
        env.emit(&format!("  Events: {}", events.join(", ")));
    }
    env.emit(&format!("  Deliver: {deliver}"));

    let deliver_only = route
        .get("deliver_only")
        .map(truthy)
        .unwrap_or(false);
    if deliver_only {
        env.emit("  Mode: direct delivery (no agent, zero LLM cost)");
    }

    let prompt = route.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    if !prompt.is_empty() {
        let preview = prompt_preview(prompt);
        let label = if deliver_only { "Message" } else { "Prompt" };
        env.emit(&format!("  {label}: {preview}"));
    }
    env.emit("\n  Configure your service to POST to the URL above.");
    env.emit("  Use the secret for HMAC-SHA256 signature validation.");
    env.emit("  The gateway must be running to receive events (hermes gateway run).\n");
}

/// Python `prompt[:80] + ("..." if len(prompt) > 80 else "")` using char
/// (Unicode codepoint) slicing semantics.
fn prompt_preview(prompt: &str) -> String {
    let chars: Vec<char> = prompt.chars().collect();
    if chars.len() > 80 {
        let head: String = chars[..80].iter().collect();
        format!("{head}...")
    } else {
        prompt.to_string()
    }
}

fn utc_now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn cmd_list(env: &mut dyn WebhookEnv, _args: &WebhookArgs) {
    let subs = load_subscriptions(env);
    if subs.is_empty() {
        env.emit("  No dynamic webhook subscriptions.");
        env.emit("  Create one with: hermes webhook subscribe <name>");
        return;
    }

    let base_url = webhook_base_url(env);
    env.emit(&format!("\n  {} webhook subscription(s):\n", subs.len()));
    for (name, route) in &subs {
        let events_vec: Vec<String> = route
            .get("events")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let events = if events_vec.is_empty() {
            "(all)".to_string()
        } else {
            events_vec.join(", ")
        };
        let mut deliver = route
            .get("deliver")
            .and_then(|v| v.as_str())
            .unwrap_or("log")
            .to_string();
        if route.get("deliver_only").map(truthy).unwrap_or(false) {
            deliver = format!("{deliver} (direct — no agent)");
        }
        let desc = route
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        env.emit(&format!("  ◆ {name}"));
        if !desc.is_empty() {
            env.emit(&format!("    {desc}"));
        }
        env.emit(&format!("    URL:     {base_url}/webhooks/{name}"));
        env.emit(&format!("    Events:  {events}"));
        env.emit(&format!("    Deliver: {deliver}"));
        env.emit("");
    }
}

fn cmd_remove(env: &mut dyn WebhookEnv, args: &WebhookArgs) {
    let name = args.name.trim().to_lowercase();
    let mut subs = load_subscriptions(env);

    if !subs.contains_key(&name) {
        env.emit(&format!("  No subscription named '{name}'."));
        env.emit("  Note: Static routes from config.yaml cannot be removed here.");
        return;
    }

    subs.remove(&name);
    if let Err(e) = save_subscriptions(env, &subs) {
        env.emit(&format!("  Error saving subscriptions: {e}"));
        return;
    }
    env.emit(&format!("  Removed webhook subscription: {name}"));
}

fn cmd_test(env: &mut dyn WebhookEnv, args: &WebhookArgs) {
    let name = args.name.trim().to_lowercase();
    let subs = load_subscriptions(env);

    let route = match subs.get(&name) {
        Some(r) => r,
        None => {
            env.emit(&format!("  No subscription named '{name}'."));
            return;
        }
    };

    let secret = route
        .get("secret")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let base_url = webhook_base_url(env);
    let url = format!("{base_url}/webhooks/{name}");

    let payload = match &args.payload {
        Some(p) if !p.is_empty() => p.clone(),
        _ => "{\"test\": true, \"event_type\": \"test\", \"message\": \"Hello from hermes webhook test\"}"
            .to_string(),
    };

    let sig = sign_payload(&secret, payload.as_bytes());

    env.emit(&format!("  Sending test POST to {url}"));

    match send_test_request(&url, &payload, &sig) {
        Ok((status, body)) => {
            env.emit(&format!("  Response ({status}): {body}"));
        }
        Err(e) => {
            env.emit(&format!("  Error: {e}"));
            env.emit("  Is the gateway running? (hermes gateway run)");
        }
    }
}

/// POST the test payload. Returns (status_code, body) on success.
fn send_test_request(url: &str, payload: &str, sig: &str) -> Result<(u16, String), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("X-Hub-Signature-256", sig)
        .header("X-GitHub-Event", "test")
        .body(payload.to_string())
        .send()
        .map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let body = resp.text().map_err(|e| e.to_string())?;
    Ok((status, body))
}

// Keep BTreeMap referenced even if compiled without it elsewhere.
#[allow(dead_code)]
type _UnusedBTreeMap = BTreeMap<String, Value>;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::path::Path;

    /// Test env that uses a temp dir and captures emitted output.
    struct TestEnv {
        home: PathBuf,
        cfg: Value,
        out: RefCell<Vec<String>>,
    }

    impl TestEnv {
        fn new(cfg: Value) -> Self {
            let mut dir = std::env::temp_dir();
            let unique = format!(
                "hermes_webhook_test_{}_{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            );
            dir.push(unique);
            std::fs::create_dir_all(&dir).unwrap();
            TestEnv {
                home: dir,
                cfg,
                out: RefCell::new(Vec::new()),
            }
        }

        fn output(&self) -> String {
            self.out.borrow().join("\n")
        }
    }

    impl Drop for TestEnv {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.home);
        }
    }

    impl WebhookEnv for TestEnv {
        fn hermes_home(&self) -> PathBuf {
            self.home.clone()
        }
        fn display_hermes_home(&self) -> String {
            "~/.hermes".to_string()
        }
        fn webhook_config(&self) -> Value {
            self.cfg.clone()
        }
        fn emit(&mut self, line: &str) {
            self.out.borrow_mut().push(line.to_string());
        }
    }

    fn enabled_cfg() -> Value {
        serde_json::json!({
            "enabled": true,
            "extra": { "host": "0.0.0.0", "port": 8644 }
        })
    }

    #[test]
    fn name_validation() {
        assert!(is_valid_name("foo"));
        assert!(is_valid_name("a1_b-c"));
        assert!(is_valid_name("9start"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("-leading"));
        assert!(!is_valid_name("_leading"));
        assert!(!is_valid_name("Has Caps")); // space + caps
        assert!(!is_valid_name("has.dot"));
    }

    #[test]
    fn base_url_maps_any_host_to_localhost() {
        let env = TestEnv::new(enabled_cfg());
        assert_eq!(webhook_base_url(&env), "http://localhost:8644");

        let env2 = TestEnv::new(serde_json::json!({
            "enabled": true,
            "extra": { "host": "example.com", "port": 9000 }
        }));
        assert_eq!(webhook_base_url(&env2), "http://example.com:9000");

        let env3 = TestEnv::new(serde_json::json!({ "enabled": true }));
        assert_eq!(webhook_base_url(&env3), "http://localhost:8644");
    }

    #[test]
    fn require_enabled_prints_hint_when_disabled() {
        let mut env = TestEnv::new(serde_json::json!({ "enabled": false }));
        assert!(!require_webhook_enabled(&mut env));
        assert!(env.output().contains("Webhook platform is not enabled"));
    }

    #[test]
    fn require_enabled_true_when_enabled() {
        let mut env = TestEnv::new(enabled_cfg());
        assert!(require_webhook_enabled(&mut env));
        assert!(env.output().is_empty());
    }

    #[test]
    fn signature_matches_known_vector() {
        // openssl: echo -n 'hello' | openssl dgst -sha256 -hmac 'secret'
        let sig = sign_payload("secret", b"hello");
        assert_eq!(
            sig,
            "sha256=88aab3ede8d3adf94d26ab90d3bafd4a2083070c3bcce9c014ee04a443847c0b"
        );
    }

    #[test]
    fn token_urlsafe_length_and_charset() {
        let t = token_urlsafe(32);
        // 32 bytes base64-no-pad => 43 chars.
        assert_eq!(t.len(), 43);
        assert!(t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        // Two calls should differ (randomness).
        assert_ne!(token_urlsafe(32), token_urlsafe(32));
    }

    #[test]
    fn subscribe_persists_and_lists_roundtrip() {
        let mut env = TestEnv::new(enabled_cfg());
        let args = WebhookArgs {
            webhook_action: Some("subscribe".into()),
            name: "My Hook".into(), // -> "my-hook"
            secret: Some("topsecret".into()),
            events: Some("push, pull_request".into()),
            description: None,
            prompt: Some("do the thing".into()),
            skills: None,
            deliver: Some("log".into()),
            deliver_only: false,
            deliver_chat_id: None,
            payload: None,
        };
        webhook_command(&mut env, &args);

        let out = env.output();
        assert!(out.contains("Created webhook subscription: my-hook"));
        assert!(out.contains("http://localhost:8644/webhooks/my-hook"));
        assert!(out.contains("Secret: topsecret"));
        assert!(out.contains("Events: push, pull_request"));

        // Verify persisted JSON.
        let path = subscriptions_path(&env);
        let data: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let route = &data["my-hook"];
        assert_eq!(route["secret"], "topsecret");
        assert_eq!(route["deliver"], "log");
        assert_eq!(route["description"], "Agent-created subscription: my-hook");
        assert_eq!(route["events"][0], "push");
        assert!(route.get("deliver_only").is_none());

        // List shows it.
        let mut env_list = TestEnv::new(enabled_cfg());
        // reuse same home so the file is found
        env_list.home = env.home.clone();
        let list_args = WebhookArgs {
            webhook_action: Some("list".into()),
            ..Default::default()
        };
        cmd_list(&mut env_list, &list_args);
        let list_out = env_list.output();
        assert!(list_out.contains("1 webhook subscription(s)"));
        assert!(list_out.contains("◆ my-hook"));
        // prevent double-cleanup of same dir
        env_list.home = std::env::temp_dir().join("nonexistent_for_test");
    }

    #[test]
    fn subscribe_rejects_deliver_only_with_log() {
        let mut env = TestEnv::new(enabled_cfg());
        let args = WebhookArgs {
            webhook_action: Some("subscribe".into()),
            name: "hook".into(),
            deliver: Some("log".into()),
            deliver_only: true,
            ..Default::default()
        };
        webhook_command(&mut env, &args);
        assert!(env.output().contains("--deliver-only requires --deliver"));
        // Nothing persisted.
        assert!(!subscriptions_path(&env).exists());
    }

    #[test]
    fn subscribe_invalid_name_rejected() {
        let mut env = TestEnv::new(enabled_cfg());
        let args = WebhookArgs {
            webhook_action: Some("subscribe".into()),
            name: "bad.name".into(),
            ..Default::default()
        };
        webhook_command(&mut env, &args);
        assert!(env.output().contains("Invalid name"));
    }

    #[test]
    fn remove_missing_and_present() {
        let mut env = TestEnv::new(enabled_cfg());
        // missing
        let rm_args = WebhookArgs {
            webhook_action: Some("remove".into()),
            name: "ghost".into(),
            ..Default::default()
        };
        cmd_remove(&mut env, &rm_args);
        assert!(env.output().contains("No subscription named 'ghost'"));

        // create then remove
        let mut subs = Map::new();
        subs.insert("real".into(), serde_json::json!({ "secret": "x" }));
        save_subscriptions(&env, &subs).unwrap();

        let mut env2 = TestEnv::new(enabled_cfg());
        env2.home = env.home.clone();
        cmd_remove(&mut env2, &rm_args);
        // ghost still missing after first run; now remove "real"
        let rm_real = WebhookArgs {
            webhook_action: Some("remove".into()),
            name: "real".into(),
            ..Default::default()
        };
        let mut env3 = TestEnv::new(enabled_cfg());
        env3.home = env.home.clone();
        cmd_remove(&mut env3, &rm_real);
        assert!(env3.output().contains("Removed webhook subscription: real"));
        env2.home = std::env::temp_dir().join("nonexistent_for_test_a");
        env3.home = std::env::temp_dir().join("nonexistent_for_test_b");
    }

    #[test]
    fn no_action_prints_usage() {
        let mut env = TestEnv::new(enabled_cfg());
        let args = WebhookArgs::default();
        webhook_command(&mut env, &args);
        assert!(env.output().contains("Usage: hermes webhook"));
    }

    #[test]
    fn prompt_preview_truncates_at_80_chars() {
        let long = "x".repeat(100);
        let p = prompt_preview(&long);
        assert_eq!(p.chars().count(), 83); // 80 + "..."
        assert!(p.ends_with("..."));
        let short = "hi";
        assert_eq!(prompt_preview(short), "hi");
    }

    #[test]
    fn load_subscriptions_handles_garbage() {
        let env = TestEnv::new(enabled_cfg());
        let path = subscriptions_path(&env);
        std::fs::write(&path, "not json").unwrap();
        assert!(load_subscriptions(&env).is_empty());
        std::fs::write(&path, "[1,2,3]").unwrap(); // not a dict
        assert!(load_subscriptions(&env).is_empty());
    }

    #[test]
    fn yaml_to_json_basic() {
        let y: serde_yaml::Value =
            serde_yaml::from_str("enabled: true\nextra:\n  host: a\n  port: 80").unwrap();
        let j = yaml_to_json(&y);
        assert_eq!(j["enabled"], serde_json::json!(true));
        assert_eq!(j["extra"]["port"], serde_json::json!(80));
    }

    fn _assert_path(_p: &Path) {}
}
