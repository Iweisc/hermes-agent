//! Credential-pool auth subcommands.
//!
//! Native Rust port of `hermes_cli/auth_commands.py`.
//!
//! The Python module is a thin, heavily interactive CLI layer on top of the
//! credential-pool data model (`agent.credential_pool`) and the OAuth / auth
//! plumbing (`hermes_cli.auth`). It mixes three concerns:
//!
//!   1. **Pure helpers** — provider-name normalization, default-label
//!      generation, source-display trimming, exhausted-status classification and
//!      human-readable formatting. These are ported here faithfully and unit
//!      tested; they have no side effects beyond reading clock time.
//!
//!   2. **Pool/command orchestration** — `auth_add`, `auth_list`, `auth_remove`,
//!      `auth_reset`, `auth_status`, the interactive menu, etc. These call into
//!      the runtime pool (`load_pool`), persistence, the provider registry, the
//!      removal-step dispatch, and the assorted OAuth login flows. None of those
//!      runtime surfaces are fully ported to native Rust yet, so the
//!      orchestration is reproduced against a small set of traits
//!      ([`AuthEnvironment`], [`AuthIo`], [`OauthFlows`]). This keeps the exact
//!      control flow of the Python commands while remaining decoupled from the
//!      concrete auth.json / reqwest plumbing other modules own.
//!
//!   3. **OAuth login flows** — `run_hermes_oauth_login_pure`,
//!      `_nous_device_code_login`, `_codex_device_code_login`,
//!      `run_gemini_oauth_login_pure`, `resolve_qwen_runtime_credentials`,
//!      `resolve_minimax_oauth_runtime_credentials`. These already live (or will
//!      live) in [`crate`]-level modules (`auth_cmd`, `crate::ag_anthropic_adapter`,
//!      `crate::ag_google_oauth`, ...). They are surfaced through the
//!      [`OauthFlows`] trait so the dispatch in `auth_add_command` matches Python
//!      exactly without hard-coding a particular implementation here.
//!
//! The pure constants, the [`PooledCredential`] data model, and the small
//! pool helpers used here mirror the ported `agent/credential_pool.py`
//! (`hermes_core::ag_credential_pool`); they are reproduced locally because
//! that module's symbols are not currently re-exported from `hermes_core`.

use std::collections::BTreeSet;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Credential-pool constants and the minimal `PooledCredential` data model.
//
// The full pool (`agent/credential_pool.py`) is ported in
// `hermes-core/src/ag_credential_pool.rs`, but that module is private to the
// `hermes_core` crate and its symbols are not re-exported at the crate root.
// To avoid coupling this CLI module to a not-yet-public path (and to keep it
// compiling standalone per the port rules), the constants, the credential
// record, and the small pure helpers this module needs (`new_id`,
// `label_from_token`, `exhausted_until`, `normalize_custom_pool_name`) are
// reproduced here faithfully. When `ag_credential_pool` is exported these
// definitions can be swapped for re-uses without changing call sites.
// ---------------------------------------------------------------------------

/// `OPENROUTER_BASE_URL` (mirrors `hermes_constants.OPENROUTER_BASE_URL`).
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub const STATUS_EXHAUSTED: &str = "exhausted";
pub const AUTH_TYPE_OAUTH: &str = "oauth";
pub const AUTH_TYPE_API_KEY: &str = "api_key";
pub const SOURCE_MANUAL: &str = "manual";
pub const CUSTOM_POOL_PREFIX: &str = "custom:";

pub const STRATEGY_FILL_FIRST: &str = "fill_first";
pub const STRATEGY_ROUND_ROBIN: &str = "round_robin";
pub const STRATEGY_RANDOM: &str = "random";
pub const STRATEGY_LEAST_USED: &str = "least_used";

const EXHAUSTED_TTL_429_SECONDS: f64 = 60.0 * 60.0;
const EXHAUSTED_TTL_DEFAULT_SECONDS: f64 = 60.0 * 60.0;

/// A single credential within a provider's failover pool. The field set mirrors
/// the Python `PooledCredential` dataclass / the ported
/// `hermes_core::ag_credential_pool::PooledCredential`.
#[derive(Debug, Clone, PartialEq)]
pub struct PooledCredential {
    pub provider: String,
    pub id: String,
    pub label: String,
    pub auth_type: String,
    pub priority: i64,
    pub source: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub last_status: Option<String>,
    pub last_status_at: Option<f64>,
    pub last_error_code: Option<i64>,
    pub last_error_reason: Option<String>,
    pub last_error_message: Option<String>,
    pub last_error_reset_at: Option<f64>,
    pub base_url: Option<String>,
    pub expires_at: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub last_refresh: Option<String>,
    pub inference_base_url: Option<String>,
    pub agent_key: Option<String>,
    pub agent_key_expires_at: Option<String>,
    pub request_count: i64,
    pub extra: serde_json::Map<String, Value>,
}

/// Generate a 6-hex-char id, matching `uuid.uuid4().hex[:6]`.
pub fn new_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let mut buf = [0u8; 3];
    let bytes = nanos.to_le_bytes();
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = bytes[i % bytes.len()].wrapping_add((i as u8).wrapping_mul(37));
    }
    format!("{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2])
}

/// Decode the unverified claims payload of a JWT (`base64url` middle segment).
fn decode_jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let pad = (4 - payload.len() % 4) % 4;
    let padded = format!("{payload}{}", "=".repeat(pad));
    use base64::Engine;
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .ok()?;
    serde_json::from_slice::<Value>(&decoded).ok()
}

/// `label_from_token`: derive a human label from a JWT's claims
/// (email / preferred_username / upn), falling back to `fallback`.
pub fn label_from_token(token: &str, fallback: &str) -> String {
    if let Some(claims) = decode_jwt_claims(token) {
        for key in ["email", "preferred_username", "upn"] {
            if let Some(value) = claims.get(key).and_then(Value::as_str) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    fallback.to_string()
}

/// `_normalize_custom_pool_name`.
pub fn normalize_custom_pool_name(name: &str) -> String {
    name.trim().to_lowercase().replace(' ', "-")
}

fn exhausted_ttl(error_code: Option<i64>) -> f64 {
    if error_code == Some(429) {
        EXHAUSTED_TTL_429_SECONDS
    } else {
        EXHAUSTED_TTL_DEFAULT_SECONDS
    }
}

fn normalize_numeric_timestamp(numeric: f64) -> Option<f64> {
    if numeric <= 0.0 {
        return None;
    }
    if numeric > 1_000_000_000_000.0 {
        Some(numeric / 1000.0)
    } else {
        Some(numeric)
    }
}

fn parse_absolute_timestamp(value: &Value) -> Option<f64> {
    match value {
        Value::Null => None,
        Value::Number(n) => normalize_numeric_timestamp(n.as_f64()?),
        Value::String(s) => {
            let raw = s.trim();
            if raw.is_empty() {
                return None;
            }
            if let Ok(numeric) = raw.parse::<f64>() {
                return normalize_numeric_timestamp(numeric);
            }
            let normalized = raw.replace('Z', "+00:00");
            chrono::DateTime::parse_from_rfc3339(&normalized)
                .ok()
                .map(|dt| dt.timestamp() as f64)
        }
        _ => None,
    }
}

/// `_exhausted_until`: absolute epoch second until which an entry stays cooled.
pub fn exhausted_until(entry: &PooledCredential) -> Option<f64> {
    if entry.last_status.as_deref() != Some(STATUS_EXHAUSTED) {
        return None;
    }
    if let Some(reset_at) = entry
        .last_error_reset_at
        .map(Value::from)
        .as_ref()
        .and_then(parse_absolute_timestamp)
    {
        return Some(reset_at);
    }
    entry
        .last_status_at
        .map(|at| at + exhausted_ttl(entry.last_error_code))
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Providers that support OAuth login in addition to API keys.
/// Mirrors Python's `_OAUTH_CAPABLE_PROVIDERS`.
pub const OAUTH_CAPABLE_PROVIDERS: &[&str] = &[
    "anthropic",
    "nous",
    "openai-codex",
    "qwen-oauth",
    "google-gemini-cli",
    "minimax-oauth",
];

/// True if `provider` supports OAuth login (membership in
/// [`OAUTH_CAPABLE_PROVIDERS`]).
pub fn is_oauth_capable(provider: &str) -> bool {
    OAUTH_CAPABLE_PROVIDERS.contains(&provider)
}

// ---------------------------------------------------------------------------
// A custom provider entry (display_name, pool_key, provider_key) tuple.
// ---------------------------------------------------------------------------

/// `(display_name, pool_key, provider_key)` triple, mirroring the tuple
/// returned by Python's `_get_custom_provider_names`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomProviderName {
    pub display_name: String,
    pub pool_key: String,
    pub provider_key: String,
}

/// `_get_custom_provider_names` (pure portion).
///
/// Given the raw `custom_providers` list (each a `name` + optional
/// `provider_key`), produce the normalized `(display, pool_key, provider_key)`
/// triples. Entries with a missing/blank `name` are skipped. The Python helper
/// loads the config and swallows any error; the config load is the caller's job
/// here — pass the already-loaded list of `(name, provider_key)` pairs.
pub fn custom_provider_names<'a, I>(entries: I) -> Vec<CustomProviderName>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut result = Vec::new();
    for (name, provider_key) in entries {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            continue;
        }
        let pool_key = format!("{CUSTOM_POOL_PREFIX}{}", normalize_custom_pool_name(trimmed));
        result.push(CustomProviderName {
            display_name: trimmed.to_string(),
            pool_key,
            provider_key: provider_key.trim().to_string(),
        });
    }
    result
}

/// `_resolve_custom_provider_input`.
///
/// If `raw` matches a custom_providers entry (by normalized display name or by
/// provider_key, case-insensitive), return its `custom:<name>` pool key. A raw
/// value already beginning with `custom:` is returned normalized as-is.
pub fn resolve_custom_provider_input(
    raw: &str,
    custom_names: &[CustomProviderName],
) -> Option<String> {
    let normalized = raw.trim().to_lowercase().replace(' ', "-");
    if normalized.is_empty() {
        return None;
    }
    if normalized.starts_with(CUSTOM_POOL_PREFIX) {
        return Some(normalized);
    }
    for cp in custom_names {
        if normalize_custom_pool_name(&cp.display_name) == normalized {
            return Some(cp.pool_key.clone());
        }
        if !cp.provider_key.is_empty() && cp.provider_key.trim().to_lowercase() == normalized {
            return Some(cp.pool_key.clone());
        }
    }
    None
}

/// `_normalize_provider`.
///
/// Lowercases/trims `provider`, maps `or` / `open-router` to `openrouter`, and
/// resolves custom-provider names/keys to their `custom:<name>` pool key.
pub fn normalize_provider(provider: &str, custom_names: &[CustomProviderName]) -> String {
    let normalized = provider.trim().to_lowercase();
    if normalized == "or" || normalized == "open-router" {
        return "openrouter".to_string();
    }
    if let Some(custom_key) = resolve_custom_provider_input(&normalized, custom_names) {
        return custom_key;
    }
    normalized
}

/// `_oauth_default_label`.
pub fn oauth_default_label(provider: &str, count: usize) -> String {
    format!("{provider}-oauth-{count}")
}

/// `_api_key_default_label`.
pub fn api_key_default_label(count: usize) -> String {
    format!("api-key-{count}")
}

/// `_display_source`: strip a leading `manual:` prefix.
pub fn display_source(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("manual:") {
        rest.to_string()
    } else {
        source.to_string()
    }
}

/// `_provider_base_url`.
///
/// For `openrouter` returns the OpenRouter base URL. For `custom:*` keys the
/// caller must supply the resolved custom-provider base URL via `custom_base_url`
/// (Python looks this up via `_get_custom_provider_config`). Otherwise looks up
/// the provider registry's `inference_base_url` via `registry_base_url`.
pub fn provider_base_url(
    provider: &str,
    registry_base_url: Option<&str>,
    custom_base_url: Option<&str>,
) -> String {
    if provider == "openrouter" {
        return OPENROUTER_BASE_URL.to_string();
    }
    if provider.starts_with(CUSTOM_POOL_PREFIX) {
        return custom_base_url.unwrap_or("").trim().to_string();
    }
    registry_base_url.unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Exhausted-status classification / formatting
// ---------------------------------------------------------------------------

/// `_classify_exhausted_status`: returns `(label, show_retry_window)`.
pub fn classify_exhausted_status(entry: &PooledCredential) -> (&'static str, bool) {
    let code = entry.last_error_code;
    let reason = entry
        .last_error_reason
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let message = entry
        .last_error_message
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();

    let reason_rate = ["rate_limit", "usage_limit", "quota", "exhausted"]
        .iter()
        .any(|t| reason.contains(t));
    let message_rate = ["rate limit", "usage limit", "quota", "too many requests"]
        .iter()
        .any(|t| message.contains(t));
    if code == Some(429) || reason_rate || message_rate {
        return ("rate-limited", true);
    }

    let reason_auth = [
        "invalid_token",
        "invalid_grant",
        "unauthorized",
        "forbidden",
        "auth",
    ]
    .iter()
    .any(|t| reason.contains(t));
    let message_auth = [
        "unauthorized",
        "forbidden",
        "expired",
        "revoked",
        "invalid token",
        "authentication",
    ]
    .iter()
    .any(|t| message.contains(t));
    if code == Some(401) || code == Some(403) || reason_auth || message_auth {
        return ("auth failed", false);
    }

    ("exhausted", true)
}

/// `_format_exhausted_status`. `now` is the current epoch time in seconds
/// (Python uses `time.time()`); injected for testability.
pub fn format_exhausted_status(entry: &PooledCredential, now: f64) -> String {
    if entry.last_status.as_deref() != Some(STATUS_EXHAUSTED) {
        return String::new();
    }
    let (label, show_retry_window) = classify_exhausted_status(entry);
    let reason_text = match entry.last_error_reason.as_deref() {
        Some(reason) if !reason.trim().is_empty() => format!(" {reason}"),
        _ => String::new(),
    };
    let code = match entry.last_error_code {
        Some(code) if code != 0 => format!(" ({code})"),
        _ => String::new(),
    };

    if !show_retry_window {
        return format!(" {label}{reason_text}{code} (re-auth may be required)");
    }

    let exhausted = match exhausted_until(entry) {
        Some(value) => value,
        None => return format!(" {label}{reason_text}{code}"),
    };

    let remaining = (exhausted - now).ceil() as i64;
    let remaining = remaining.max(0);
    if remaining <= 0 {
        return format!(" {label}{reason_text}{code} (ready to retry)");
    }

    let (minutes, seconds) = (remaining / 60, remaining % 60);
    let (hours, minutes) = (minutes / 60, minutes % 60);
    let (days, hours) = (hours / 24, hours % 24);
    let wait = if days != 0 {
        format!("{days}d {hours}h")
    } else if hours != 0 {
        format!("{hours}h {minutes}m")
    } else if minutes != 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    };
    format!(" {label}{reason_text}{code} ({wait} left)")
}

// ---------------------------------------------------------------------------
// Strategy descriptions (for interactive strategy menu)
// ---------------------------------------------------------------------------

/// The four strategies in display order, mirroring the interactive menu list.
pub fn strategy_order() -> [&'static str; 4] {
    [
        STRATEGY_FILL_FIRST,
        STRATEGY_ROUND_ROBIN,
        STRATEGY_LEAST_USED,
        STRATEGY_RANDOM,
    ]
}

/// One-line description per strategy, mirroring Python's `descriptions` dict.
pub fn strategy_description(strategy: &str) -> &'static str {
    match strategy {
        STRATEGY_FILL_FIRST => "Use first key until exhausted, then next",
        STRATEGY_ROUND_ROBIN => "Cycle through keys evenly",
        STRATEGY_LEAST_USED => "Always pick the least-used key",
        STRATEGY_RANDOM => "Random selection",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// I/O + environment abstractions
// ---------------------------------------------------------------------------

/// User-facing prompt / print operations. Mirrors `print(...)`,
/// `input(...)`, `getpass(...)` and `sys.stdin.isatty()` from the Python
/// module. `prompt` / `getpass` return `None` to signal EOF / KeyboardInterrupt
/// (Python catches `(EOFError, KeyboardInterrupt)`).
pub trait AuthIo {
    /// `print(line)`.
    fn print(&mut self, line: &str);
    /// `input(prompt)`. Returns `None` on EOF / interrupt.
    fn prompt(&mut self, prompt: &str) -> Option<String>;
    /// `getpass(prompt)`. Returns `None` on EOF / interrupt.
    fn getpass(&mut self, prompt: &str) -> Option<String>;
    /// `sys.stdin.isatty()`.
    fn is_tty(&self) -> bool;
}

/// Credentials returned by an OAuth login flow. Field names mirror the dict keys
/// the Python flows return; absent fields are `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OauthCredentials {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub base_url: Option<String>,
    pub last_refresh: Option<String>,
    /// e.g. Gemini's `email`, used as a default label.
    pub email: Option<String>,
}

/// The set of OAuth login flows dispatched by `auth_add_command`. Each method
/// mirrors a Python call and may perform network I/O. An error string aborts the
/// command (Python raises `SystemExit`).
pub trait OauthFlows {
    /// `anthropic_adapter.run_hermes_oauth_login_pure()`.
    fn anthropic_login(&mut self) -> Result<OauthCredentials, String>;
    /// `auth_mod._codex_device_code_login()`.
    fn codex_device_code_login(&mut self) -> Result<OauthCredentials, String>;
    /// `agent.google_oauth.run_gemini_oauth_login_pure()`.
    fn gemini_login(&mut self) -> Result<OauthCredentials, String>;
    /// `auth_mod.resolve_qwen_runtime_credentials(refresh_if_expiring=False)`.
    fn qwen_runtime_credentials(&mut self) -> Result<OauthCredentials, String>;
    /// `resolve_minimax_oauth_runtime_credentials()`.
    fn minimax_runtime_credentials(&mut self) -> Result<OauthCredentials, String>;
    /// The Nous device-code / shared-import flow. Returns the label the command
    /// should print (already persisted). Returning `None` is not expected; on
    /// error return `Err`. `import_shared` indicates the user accepted the
    /// shared-credential import prompt; the implementation decides fallback.
    fn nous_login(&mut self, label: Option<&str>) -> Result<String, String>;
}

/// Side-effecting environment: pool loading/persistence, provider registry,
/// custom-provider resolution, removal-step dispatch, auth status, config
/// strategy storage. Mirrors the runtime modules the Python CLI imports.
pub trait AuthEnvironment {
    /// `load_pool(provider).entries()`.
    fn pool_entries(&mut self, provider: &str) -> Vec<PooledCredential>;
    /// `load_pool(provider).peek()` — currently selected credential, if any.
    fn pool_peek(&mut self, provider: &str) -> Option<PooledCredential>;
    /// `pool.add_entry(entry)` — append and persist; returns the persisted entry.
    fn pool_add_entry(&mut self, provider: &str, entry: PooledCredential) -> PooledCredential;
    /// `pool.resolve_target(target)` → `(Some(index), Some(entry), None)` or
    /// `(None, None, Some(error))`.
    fn pool_resolve_target(
        &mut self,
        provider: &str,
        target: &str,
    ) -> (Option<usize>, Option<PooledCredential>, Option<String>);
    /// `pool.remove_index(index)`.
    fn pool_remove_index(&mut self, provider: &str, index: usize) -> Option<PooledCredential>;
    /// `pool.reset_statuses()` → number reset.
    fn pool_reset_statuses(&mut self, provider: &str) -> usize;
    /// `pool.has_credentials()`.
    fn pool_has_credentials(&mut self, provider: &str) -> bool;

    /// `provider in PROVIDER_REGISTRY`.
    fn is_registered_provider(&self, provider: &str) -> bool;
    /// All known provider registry keys (for listing / picker hints).
    fn registry_providers(&self) -> Vec<String>;
    /// `list_custom_pool_providers()`.
    fn list_custom_pool_providers(&mut self) -> Vec<String>;
    /// `PROVIDER_REGISTRY.get(provider).inference_base_url`.
    fn registry_base_url(&self, provider: &str) -> Option<String>;
    /// Resolved base URL for a `custom:*` provider key (Python's
    /// `_get_custom_provider_config(...).base_url`).
    fn custom_provider_base_url(&mut self, provider: &str) -> Option<String>;
    /// The list of `(display_name, provider_key)` custom-provider config entries.
    fn custom_provider_entries(&mut self) -> Vec<(String, String)>;

    /// Clear all suppressed sources for `provider` (the re-engagement reset in
    /// `auth_add_command`). Errors are swallowed in Python; ignore failures.
    fn clear_suppressions(&mut self, provider: &str);
    /// `unsuppress_credential_source(provider, source)`.
    fn unsuppress_credential_source(&mut self, provider: &str, source: &str);

    /// Removal dispatch: run the registered `RemovalStep` for `(provider,
    /// source)` against the removed entry. Returns `(cleaned, hints)` lines to
    /// print, having already applied any source suppression. Returns `None` when
    /// no step is registered (e.g. plain `manual`).
    fn run_removal_step(
        &mut self,
        provider: &str,
        removed: &PooledCredential,
    ) -> Option<RemovalOutput>;

    /// `auth_mod.get_auth_status(provider)`.
    fn get_auth_status(&mut self, provider: &str) -> AuthStatus;
    /// `auth_mod.logout_command(...)`.
    fn logout(&mut self, provider: Option<&str>);
    /// `auth_mod.login_spotify_command(args)`.
    fn login_spotify(&mut self);

    /// `get_pool_strategy(provider)`.
    fn get_pool_strategy(&mut self, provider: &str) -> String;
    /// Persist a strategy choice into config
    /// (`credential_pool_strategies[provider] = strategy` + `save_config`).
    fn set_pool_strategy(&mut self, provider: &str, strategy: &str);
}

/// Output of a removal step: the lines to print after a successful removal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemovalOutput {
    pub cleaned: Vec<String>,
    pub hints: Vec<String>,
}

/// Mirrors the dict returned by `auth_mod.get_auth_status(provider)` for the
/// fields the command reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthStatus {
    pub logged_in: bool,
    pub error: Option<String>,
    pub auth_type: Option<String>,
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    pub scope: Option<String>,
    pub expires_at: Option<String>,
    pub api_base_url: Option<String>,
}

/// Aborting error, matching Python's `raise SystemExit(message)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthExit(pub String);

impl std::fmt::Display for AuthExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AuthExit {}

impl From<String> for AuthExit {
    fn from(message: String) -> Self {
        AuthExit(message)
    }
}

type AuthResult = Result<(), AuthExit>;

// ---------------------------------------------------------------------------
// Argument structs (mirroring the argparse namespaces used by each command)
// ---------------------------------------------------------------------------

/// Args for `auth add`. Mirrors the attributes accessed via `getattr(args, …)`.
#[derive(Debug, Clone, Default)]
pub struct AuthAddArgs {
    pub provider: String,
    pub auth_type: Option<String>,
    pub api_key: Option<String>,
    pub label: Option<String>,
    // Nous-specific knobs (passed through to the Nous flow).
    pub portal_url: Option<String>,
    pub inference_url: Option<String>,
    pub client_id: Option<String>,
    pub scope: Option<String>,
    pub no_browser: bool,
    pub timeout: Option<f64>,
    pub insecure: bool,
    pub ca_bundle: Option<String>,
    pub min_key_ttl_seconds: Option<i64>,
}

/// Args for `auth list` / `reset` / `status` (provider filter only).
#[derive(Debug, Clone, Default)]
pub struct AuthProviderArgs {
    pub provider: Option<String>,
}

/// Args for `auth remove`.
#[derive(Debug, Clone, Default)]
pub struct AuthRemoveArgs {
    pub provider: String,
    /// `target`, falling back to `index` in Python.
    pub target: Option<String>,
    pub index: Option<String>,
}

/// Top-level `auth <action>` dispatch input.
#[derive(Debug, Clone, Default)]
pub struct AuthArgs {
    pub auth_action: Option<String>,
    pub add: AuthAddArgs,
    pub provider_args: AuthProviderArgs,
    pub remove: AuthRemoveArgs,
    pub spotify_action: Option<String>,
    pub logout_provider: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper: normalize provider via environment-loaded custom names
// ---------------------------------------------------------------------------

fn env_custom_names(env: &mut dyn AuthEnvironment) -> Vec<CustomProviderName> {
    let entries = env.custom_provider_entries();
    let refs: Vec<(&str, &str)> = entries
        .iter()
        .map(|(name, key)| (name.as_str(), key.as_str()))
        .collect();
    custom_provider_names(refs)
}

fn env_normalize_provider(env: &mut dyn AuthEnvironment, provider: &str) -> String {
    let custom_names = env_custom_names(env);
    normalize_provider(provider, &custom_names)
}

fn env_provider_base_url(env: &mut dyn AuthEnvironment, provider: &str) -> String {
    let registry = env.registry_base_url(provider);
    let custom = if provider.starts_with(CUSTOM_POOL_PREFIX) {
        env.custom_provider_base_url(provider)
    } else {
        None
    };
    provider_base_url(provider, registry.as_deref(), custom.as_deref())
}

// ---------------------------------------------------------------------------
// auth add
// ---------------------------------------------------------------------------

/// `auth_add_command`.
pub fn auth_add_command(
    args: &AuthAddArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
    flows: &mut dyn OauthFlows,
) -> AuthResult {
    let provider = env_normalize_provider(env, &args.provider);
    if !env.is_registered_provider(&provider)
        && provider != "openrouter"
        && !provider.starts_with(CUSTOM_POOL_PREFIX)
    {
        return Err(AuthExit(format!("Unknown provider: {provider}")));
    }

    let mut requested_type = args
        .auth_type
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if requested_type == AUTH_TYPE_API_KEY || requested_type == "api-key" {
        requested_type = AUTH_TYPE_API_KEY.to_string();
    }
    if requested_type.is_empty() {
        requested_type = if provider.starts_with(CUSTOM_POOL_PREFIX) {
            AUTH_TYPE_API_KEY.to_string()
        } else if is_oauth_capable(&provider) {
            AUTH_TYPE_OAUTH.to_string()
        } else {
            AUTH_TYPE_API_KEY.to_string()
        };
    }

    // Clear ALL suppressions for this provider — re-adding a credential is a
    // strong signal the user wants auth re-enabled.
    if !provider.starts_with(CUSTOM_POOL_PREFIX) {
        env.clear_suppressions(&provider);
    }

    let entries_len = env.pool_entries(&provider).len();

    if requested_type == AUTH_TYPE_API_KEY {
        let mut token = args.api_key.as_deref().unwrap_or("").trim().to_string();
        if token.is_empty() {
            token = io
                .getpass("Paste your API key: ")
                .unwrap_or_default()
                .trim()
                .to_string();
        }
        if token.is_empty() {
            return Err(AuthExit("No API key provided.".to_string()));
        }
        let default_label = api_key_default_label(entries_len + 1);
        let mut label = args.label.as_deref().unwrap_or("").trim().to_string();
        if label.is_empty() {
            if io.is_tty() {
                let typed = io
                    .prompt(&format!("Label (optional, default: {default_label}): "))
                    .unwrap_or_default();
                let typed = typed.trim();
                label = if typed.is_empty() {
                    default_label.clone()
                } else {
                    typed.to_string()
                };
            } else {
                label = default_label.clone();
            }
        }
        let base_url = env_provider_base_url(env, &provider);
        let entry = PooledCredential {
            provider: provider.clone(),
            id: new_id(),
            label: label.clone(),
            auth_type: AUTH_TYPE_API_KEY.to_string(),
            priority: 0,
            source: SOURCE_MANUAL.to_string(),
            access_token: token,
            refresh_token: None,
            last_status: None,
            last_status_at: None,
            last_error_code: None,
            last_error_reason: None,
            last_error_message: None,
            last_error_reset_at: None,
            base_url: Some(base_url),
            expires_at: None,
            expires_at_ms: None,
            last_refresh: None,
            inference_base_url: None,
            agent_key: None,
            agent_key_expires_at: None,
            request_count: 0,
            extra: Default::default(),
        };
        env.pool_add_entry(&provider, entry);
        let new_count = env.pool_entries(&provider).len();
        io.print(&format!(
            "Added {provider} credential #{new_count}: \"{label}\""
        ));
        return Ok(());
    }

    match provider.as_str() {
        "anthropic" => {
            let creds = flows.anthropic_login()?;
            if creds.access_token.is_empty() {
                return Err(AuthExit(
                    "Anthropic OAuth login did not return credentials.".to_string(),
                ));
            }
            let label = non_empty_label(args.label.as_deref()).unwrap_or_else(|| {
                label_from_token(
                    &creds.access_token,
                    &oauth_default_label(&provider, entries_len + 1),
                )
            });
            let base_url = env_provider_base_url(env, &provider);
            let entry = oauth_entry(
                &provider,
                &label,
                &format!("{SOURCE_MANUAL}:hermes_pkce"),
                &creds.access_token,
                creds.refresh_token.clone(),
                |e| {
                    e.expires_at_ms = creds.expires_at_ms;
                    e.base_url = Some(base_url.clone());
                },
            );
            let entry = env.pool_add_entry(&provider, entry);
            let new_count = env.pool_entries(&provider).len();
            io.print(&format!(
                "Added {provider} OAuth credential #{new_count}: \"{}\"",
                entry.label
            ));
            Ok(())
        }
        "nous" => {
            let custom_label = non_empty_label(args.label.as_deref());
            let shown_label = flows.nous_login(custom_label.as_deref())?;
            io.print(&format!(
                "Saved {provider} OAuth device-code credentials: \"{shown_label}\""
            ));
            Ok(())
        }
        "openai-codex" => {
            env.unsuppress_credential_source(&provider, "device_code");
            let creds = flows.codex_device_code_login()?;
            let label = non_empty_label(args.label.as_deref()).unwrap_or_else(|| {
                label_from_token(
                    &creds.access_token,
                    &oauth_default_label(&provider, entries_len + 1),
                )
            });
            let entry = oauth_entry(
                &provider,
                &label,
                &format!("{SOURCE_MANUAL}:device_code"),
                &creds.access_token,
                creds.refresh_token.clone(),
                |e| {
                    e.base_url = creds.base_url.clone();
                    e.last_refresh = creds.last_refresh.clone();
                },
            );
            let entry = env.pool_add_entry(&provider, entry);
            let new_count = env.pool_entries(&provider).len();
            io.print(&format!(
                "Added {provider} OAuth credential #{new_count}: \"{}\"",
                entry.label
            ));
            Ok(())
        }
        "google-gemini-cli" => {
            let creds = flows.gemini_login()?;
            let label = non_empty_label(args.label.as_deref()).unwrap_or_else(|| {
                creds
                    .email
                    .clone()
                    .filter(|e| !e.is_empty())
                    .unwrap_or_else(|| oauth_default_label(&provider, entries_len + 1))
            });
            let entry = oauth_entry(
                &provider,
                &label,
                &format!("{SOURCE_MANUAL}:google_pkce"),
                &creds.access_token,
                creds.refresh_token.clone(),
                |_| {},
            );
            let entry = env.pool_add_entry(&provider, entry);
            let new_count = env.pool_entries(&provider).len();
            io.print(&format!(
                "Added {provider} OAuth credential #{new_count}: \"{}\"",
                entry.label
            ));
            Ok(())
        }
        "qwen-oauth" => {
            let creds = flows.qwen_runtime_credentials()?;
            let label = non_empty_label(args.label.as_deref()).unwrap_or_else(|| {
                label_from_token(
                    &creds.access_token,
                    &oauth_default_label(&provider, entries_len + 1),
                )
            });
            let entry = oauth_entry(
                &provider,
                &label,
                &format!("{SOURCE_MANUAL}:qwen_cli"),
                &creds.access_token,
                None,
                |e| {
                    e.base_url = creds.base_url.clone();
                },
            );
            let entry = env.pool_add_entry(&provider, entry);
            let new_count = env.pool_entries(&provider).len();
            io.print(&format!(
                "Added {provider} OAuth credential #{new_count}: \"{}\"",
                entry.label
            ));
            Ok(())
        }
        "minimax-oauth" => {
            let creds = flows.minimax_runtime_credentials()?;
            let label = non_empty_label(args.label.as_deref()).unwrap_or_else(|| {
                label_from_token(
                    &creds.access_token,
                    &oauth_default_label(&provider, entries_len + 1),
                )
            });
            let entry = oauth_entry(
                &provider,
                &label,
                &format!("{SOURCE_MANUAL}:minimax_oauth"),
                &creds.access_token,
                None,
                |e| {
                    e.base_url = creds.base_url.clone();
                },
            );
            let entry = env.pool_add_entry(&provider, entry);
            let new_count = env.pool_entries(&provider).len();
            io.print(&format!(
                "Added {provider} OAuth credential #{new_count}: \"{}\"",
                entry.label
            ));
            Ok(())
        }
        _ => Err(AuthExit(format!(
            "`hermes auth add {provider}` is not implemented for auth type {requested_type} yet."
        ))),
    }
}

fn non_empty_label(label: Option<&str>) -> Option<String> {
    label
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Build an OAuth `PooledCredential` with common defaults, then apply
/// `customize` for provider-specific fields.
fn oauth_entry<F: FnOnce(&mut PooledCredential)>(
    provider: &str,
    label: &str,
    source: &str,
    access_token: &str,
    refresh_token: Option<String>,
    customize: F,
) -> PooledCredential {
    let mut entry = PooledCredential {
        provider: provider.to_string(),
        id: new_id(),
        label: label.to_string(),
        auth_type: AUTH_TYPE_OAUTH.to_string(),
        priority: 0,
        source: source.to_string(),
        access_token: access_token.to_string(),
        refresh_token,
        last_status: None,
        last_status_at: None,
        last_error_code: None,
        last_error_reason: None,
        last_error_message: None,
        last_error_reset_at: None,
        base_url: None,
        expires_at: None,
        expires_at_ms: None,
        last_refresh: None,
        inference_base_url: None,
        agent_key: None,
        agent_key_expires_at: None,
        request_count: 0,
        extra: Default::default(),
    };
    customize(&mut entry);
    entry
}

// ---------------------------------------------------------------------------
// auth list
// ---------------------------------------------------------------------------

/// `auth_list_command`. `now` is the current epoch time in seconds.
pub fn auth_list_command(
    args: &AuthProviderArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
    now: f64,
) {
    let provider_filter = env_normalize_provider(env, args.provider.as_deref().unwrap_or(""));
    let providers: Vec<String> = if !provider_filter.is_empty() {
        vec![provider_filter]
    } else {
        let mut set: BTreeSet<String> = BTreeSet::new();
        for p in env.registry_providers() {
            set.insert(p);
        }
        set.insert("openrouter".to_string());
        for p in env.list_custom_pool_providers() {
            set.insert(p);
        }
        set.into_iter().collect()
    };

    for provider in providers {
        let entries = env.pool_entries(&provider);
        if entries.is_empty() {
            continue;
        }
        let current = env.pool_peek(&provider);
        io.print(&format!("{provider} ({} credentials):", entries.len()));
        for (idx0, entry) in entries.iter().enumerate() {
            let idx = idx0 + 1;
            let mut marker = "  ";
            if let Some(cur) = &current {
                if entry.id == cur.id {
                    marker = "← ";
                }
            }
            let status = format_exhausted_status(entry, now);
            let source = display_source(&entry.source);
            let line = format!(
                "  #{idx}  {:<20} {:<7} {source}{status} {marker}",
                entry.label, entry.auth_type
            );
            io.print(line.trim_end());
        }
        io.print("");
    }
}

// ---------------------------------------------------------------------------
// auth remove
// ---------------------------------------------------------------------------

/// `auth_remove_command`.
pub fn auth_remove_command(
    args: &AuthRemoveArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
) -> AuthResult {
    let provider = env_normalize_provider(env, &args.provider);
    let target = args.target.clone().or_else(|| args.index.clone());
    let target_str = target.clone().unwrap_or_default();

    let (index, matched, error) = env.pool_resolve_target(&provider, &target_str);
    if matched.is_none() || index.is_none() {
        let error = error.unwrap_or_default();
        return Err(AuthExit(format!("{error} Provider: {provider}.")));
    }
    let index = index.unwrap();
    let removed = match env.pool_remove_index(&provider, index) {
        Some(removed) => removed,
        None => {
            return Err(AuthExit(format!(
                "No credential matching \"{target_str}\" for provider {provider}."
            )));
        }
    };
    io.print(&format!(
        "Removed {provider} credential #{index} ({})",
        removed.label
    ));

    // Unified removal dispatch. The env applies suppression internally and
    // returns the cleaned + hint lines, or None when no step is registered.
    if let Some(output) = env.run_removal_step(&provider, &removed) {
        for line in &output.cleaned {
            io.print(line);
        }
        for line in &output.hints {
            io.print(line);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// auth reset
// ---------------------------------------------------------------------------

/// `auth_reset_command`.
pub fn auth_reset_command(
    args: &AuthProviderArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
) {
    let provider = env_normalize_provider(env, args.provider.as_deref().unwrap_or(""));
    let count = env.pool_reset_statuses(&provider);
    io.print(&format!("Reset status on {count} {provider} credentials"));
}

// ---------------------------------------------------------------------------
// auth status
// ---------------------------------------------------------------------------

/// `auth_status_command`.
pub fn auth_status_command(
    args: &AuthProviderArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
) -> AuthResult {
    let provider = env_normalize_provider(env, args.provider.as_deref().unwrap_or(""));
    if provider.is_empty() {
        return Err(AuthExit(
            "Provider is required. Example: `hermes auth status spotify`.".to_string(),
        ));
    }
    let status = env.get_auth_status(&provider);
    if !status.logged_in {
        match status.error.as_deref().filter(|e| !e.is_empty()) {
            Some(reason) => io.print(&format!("{provider}: logged out ({reason})")),
            None => io.print(&format!("{provider}: logged out")),
        }
        return Ok(());
    }

    io.print(&format!("{provider}: logged in"));
    let pairs: [(&str, &Option<String>); 6] = [
        ("auth_type", &status.auth_type),
        ("client_id", &status.client_id),
        ("redirect_uri", &status.redirect_uri),
        ("scope", &status.scope),
        ("expires_at", &status.expires_at),
        ("api_base_url", &status.api_base_url),
    ];
    for (key, value) in pairs {
        if let Some(value) = value.as_deref().filter(|v| !v.is_empty()) {
            io.print(&format!("  {key}: {value}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// auth logout / spotify
// ---------------------------------------------------------------------------

/// `auth_logout_command`.
pub fn auth_logout_command(provider: Option<&str>, env: &mut dyn AuthEnvironment) {
    env.logout(provider);
}

/// `auth_spotify_command`.
pub fn auth_spotify_command(
    spotify_action: Option<&str>,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
) -> AuthResult {
    let action = spotify_action.unwrap_or("login").trim().to_lowercase();
    match action.as_str() {
        "" | "login" => {
            env.login_spotify();
            Ok(())
        }
        "status" => auth_status_command(
            &AuthProviderArgs {
                provider: Some("spotify".to_string()),
            },
            env,
            io,
        ),
        "logout" => {
            auth_logout_command(Some("spotify"), env);
            Ok(())
        }
        other => Err(AuthExit(format!("Unknown Spotify auth action: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Interactive provider picker + menus
// ---------------------------------------------------------------------------

/// `_pick_provider`: print hints, prompt, normalize. Returns `None` on EOF /
/// interrupt (Python raises bare `SystemExit()`); callers treat `None` as abort.
pub fn pick_provider(
    prompt: &str,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
) -> Option<String> {
    let mut known: BTreeSet<String> = env.registry_providers().into_iter().collect();
    known.insert("openrouter".to_string());
    let known_list: Vec<String> = known.into_iter().collect();

    let custom_names = env_custom_names(env);
    if !custom_names.is_empty() {
        let custom_display: Vec<String> = custom_names
            .iter()
            .map(|c| c.display_name.clone())
            .collect();
        io.print(&format!("\nKnown providers: {}", known_list.join(", ")));
        io.print(&format!("Custom endpoints: {}", custom_display.join(", ")));
    } else {
        io.print(&format!("\nKnown providers: {}", known_list.join(", ")));
    }
    let raw = io.prompt(&format!("{prompt}: "))?;
    let raw = raw.trim();
    Some(env_normalize_provider(env, raw))
}

/// `_interactive_add`.
pub fn interactive_add(
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
    flows: &mut dyn OauthFlows,
) -> AuthResult {
    let provider = match pick_provider("Provider to add credential for", env, io) {
        Some(p) => p,
        None => return Ok(()),
    };
    if !env.is_registered_provider(&provider)
        && provider != "openrouter"
        && !provider.starts_with(CUSTOM_POOL_PREFIX)
    {
        return Err(AuthExit(format!("Unknown provider: {provider}")));
    }

    let auth_type = if is_oauth_capable(&provider) {
        io.print(&format!(
            "\n{provider} supports both API keys and OAuth login."
        ));
        io.print("  1. API key (paste a key from the provider dashboard)");
        io.print("  2. OAuth login (authenticate via browser)");
        match io.prompt("Type [1/2]: ") {
            None => return Ok(()),
            Some(choice) => {
                if choice.trim() == "2" {
                    "oauth".to_string()
                } else {
                    "api_key".to_string()
                }
            }
        }
    } else {
        "api_key".to_string()
    };

    let label = match io.prompt("Label / account name (optional): ") {
        None => return Ok(()),
        Some(typed) => {
            let typed = typed.trim();
            if typed.is_empty() {
                None
            } else {
                Some(typed.to_string())
            }
        }
    };

    let add_args = AuthAddArgs {
        provider,
        auth_type: Some(auth_type),
        label,
        ..Default::default()
    };
    auth_add_command(&add_args, env, io, flows)
}

/// `_interactive_remove`. `now` for status formatting.
pub fn interactive_remove(env: &mut dyn AuthEnvironment, io: &mut dyn AuthIo, now: f64) -> AuthResult {
    let provider = match pick_provider("Provider to remove credential from", env, io) {
        Some(p) => p,
        None => return Ok(()),
    };
    if !env.pool_has_credentials(&provider) {
        io.print(&format!("No credentials for {provider}."));
        return Ok(());
    }

    for (idx0, entry) in env.pool_entries(&provider).iter().enumerate() {
        let i = idx0 + 1;
        let exhausted = format_exhausted_status(entry, now);
        io.print(&format!(
            "  #{i}  {:25} {:10} {}{exhausted} [id:{}]",
            entry.label, entry.auth_type, entry.source, entry.id
        ));
    }

    let raw = match io.prompt("Remove #, id, or label (blank to cancel): ") {
        None => return Ok(()),
        Some(raw) => raw.trim().to_string(),
    };
    if raw.is_empty() {
        return Ok(());
    }

    let remove_args = AuthRemoveArgs {
        provider,
        target: Some(raw),
        index: None,
    };
    auth_remove_command(&remove_args, env, io)
}

/// `_interactive_reset`.
pub fn interactive_reset(env: &mut dyn AuthEnvironment, io: &mut dyn AuthIo) -> AuthResult {
    let provider = match pick_provider("Provider to reset cooldowns for", env, io) {
        Some(p) => p,
        None => return Ok(()),
    };
    auth_reset_command(
        &AuthProviderArgs {
            provider: Some(provider),
        },
        env,
        io,
    );
    Ok(())
}

/// `_interactive_strategy`.
pub fn interactive_strategy(env: &mut dyn AuthEnvironment, io: &mut dyn AuthIo) -> AuthResult {
    let provider = match pick_provider("Provider to set strategy for", env, io) {
        Some(p) => p,
        None => return Ok(()),
    };
    let current = env.get_pool_strategy(&provider);
    let strategies = strategy_order();

    io.print(&format!("\nCurrent strategy for {provider}: {current}"));
    io.print("");
    for (idx0, s) in strategies.iter().enumerate() {
        let i = idx0 + 1;
        let marker = if *s == current { " ←" } else { "" };
        io.print(&format!(
            "  {i}. {:15} — {}{marker}",
            s,
            strategy_description(s)
        ));
    }

    let raw = match io.prompt("\nStrategy [1-4]: ") {
        None => return Ok(()),
        Some(raw) => raw.trim().to_string(),
    };
    if raw.is_empty() {
        return Ok(());
    }

    let strategy = match raw.parse::<i64>() {
        Ok(n) if n >= 1 && (n as usize) <= strategies.len() => strategies[(n - 1) as usize],
        _ => {
            io.print("Invalid choice.");
            return Ok(());
        }
    };

    env.set_pool_strategy(&provider, strategy);
    io.print(&format!("Set {provider} strategy to: {strategy}"));
    Ok(())
}

/// `_interactive_auth`: the bare `hermes auth` menu. `now` for status
/// formatting; `bedrock` carries optional AWS Bedrock status to render
/// (Python's boto3 block); pass `None` when unavailable.
pub fn interactive_auth(
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
    flows: &mut dyn OauthFlows,
    now: f64,
    bedrock: Option<&BedrockStatus>,
) -> AuthResult {
    io.print("Credential Pool Status");
    io.print(&"=".repeat(50));

    auth_list_command(&AuthProviderArgs { provider: None }, env, io, now);

    if let Some(b) = bedrock {
        io.print("bedrock (AWS SDK credential chain):");
        io.print(&format!("  Auth: {}", b.auth_source));
        io.print(&format!("  Region: {}", b.region));
        match &b.identity {
            Some(arn) => io.print(&format!("  Identity: {arn}")),
            None => io.print("  Identity: (could not resolve — boto3 STS call failed)"),
        }
        io.print("");
    }
    io.print("");

    let choices = [
        "Add a credential",
        "Remove a credential",
        "Reset cooldowns for a provider",
        "Set rotation strategy for a provider",
        "Exit",
    ];
    io.print("What would you like to do?");
    for (idx0, choice) in choices.iter().enumerate() {
        io.print(&format!("  {}. {choice}", idx0 + 1));
    }

    let raw = match io.prompt("\nChoice: ") {
        None => return Ok(()),
        Some(raw) => raw.trim().to_string(),
    };

    if raw.is_empty() || raw == choices.len().to_string() {
        return Ok(());
    }

    match raw.as_str() {
        "1" => interactive_add(env, io, flows),
        "2" => interactive_remove(env, io, now),
        "3" => interactive_reset(env, io),
        "4" => interactive_strategy(env, io),
        _ => Ok(()),
    }
}

/// Optional AWS Bedrock status block rendered by [`interactive_auth`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BedrockStatus {
    pub auth_source: String,
    pub region: String,
    pub identity: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level dispatch
// ---------------------------------------------------------------------------

/// `auth_command`: dispatch on `auth_action`. `now` is the current epoch time;
/// `bedrock` is the optional Bedrock status used only by the interactive menu.
pub fn auth_command(
    args: &AuthArgs,
    env: &mut dyn AuthEnvironment,
    io: &mut dyn AuthIo,
    flows: &mut dyn OauthFlows,
    now: f64,
    bedrock: Option<&BedrockStatus>,
) -> AuthResult {
    match args.auth_action.as_deref().unwrap_or("") {
        "add" => auth_add_command(&args.add, env, io, flows),
        "list" => {
            auth_list_command(&args.provider_args, env, io, now);
            Ok(())
        }
        "remove" => auth_remove_command(&args.remove, env, io),
        "reset" => {
            auth_reset_command(&args.provider_args, env, io);
            Ok(())
        }
        "status" => auth_status_command(&args.provider_args, env, io),
        "logout" => {
            auth_logout_command(args.logout_provider.as_deref(), env);
            Ok(())
        }
        "spotify" => auth_spotify_command(args.spotify_action.as_deref(), env, io),
        _ => interactive_auth(env, io, flows, now, bedrock),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cred(provider: &str, label: &str, auth_type: &str, source: &str) -> PooledCredential {
        PooledCredential {
            provider: provider.to_string(),
            id: "abc123".to_string(),
            label: label.to_string(),
            auth_type: auth_type.to_string(),
            priority: 0,
            source: source.to_string(),
            access_token: "tok".to_string(),
            refresh_token: None,
            last_status: None,
            last_status_at: None,
            last_error_code: None,
            last_error_reason: None,
            last_error_message: None,
            last_error_reset_at: None,
            base_url: None,
            expires_at: None,
            expires_at_ms: None,
            last_refresh: None,
            inference_base_url: None,
            agent_key: None,
            agent_key_expires_at: None,
            request_count: 0,
            extra: Default::default(),
        }
    }

    #[test]
    fn normalize_provider_aliases() {
        assert_eq!(normalize_provider("OR", &[]), "openrouter");
        assert_eq!(normalize_provider(" Open-Router ", &[]), "openrouter");
        assert_eq!(normalize_provider("Anthropic", &[]), "anthropic");
    }

    #[test]
    fn normalize_provider_custom_match() {
        let names = custom_provider_names([("My Endpoint", "mykey")]);
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].pool_key, "custom:my-endpoint");
        // By display name (normalized).
        assert_eq!(
            normalize_provider("my endpoint", &names),
            "custom:my-endpoint"
        );
        // By provider_key.
        assert_eq!(normalize_provider("mykey", &names), "custom:my-endpoint");
        // Direct custom: passthrough.
        assert_eq!(
            normalize_provider("custom:other", &names),
            "custom:other"
        );
        // No match returns normalized input.
        assert_eq!(normalize_provider("unknown", &names), "unknown");
    }

    #[test]
    fn custom_names_skip_blank() {
        let names = custom_provider_names([("  ", "k"), ("Good", "")]);
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].display_name, "Good");
        assert_eq!(names[0].provider_key, "");
    }

    #[test]
    fn default_labels() {
        assert_eq!(oauth_default_label("anthropic", 3), "anthropic-oauth-3");
        assert_eq!(api_key_default_label(2), "api-key-2");
    }

    #[test]
    fn display_source_trims_manual() {
        assert_eq!(display_source("manual:device_code"), "device_code");
        assert_eq!(display_source("env:OPENAI_API_KEY"), "env:OPENAI_API_KEY");
        assert_eq!(display_source("manual"), "manual");
    }

    #[test]
    fn provider_base_url_resolution() {
        assert_eq!(
            provider_base_url("openrouter", None, None),
            OPENROUTER_BASE_URL
        );
        assert_eq!(
            provider_base_url("custom:x", None, Some(" https://api.x/v1 ")),
            "https://api.x/v1"
        );
        assert_eq!(provider_base_url("custom:x", None, None), "");
        assert_eq!(
            provider_base_url("anthropic", Some("https://api.anthropic.com"), None),
            "https://api.anthropic.com"
        );
        assert_eq!(provider_base_url("anthropic", None, None), "");
    }

    #[test]
    fn classify_rate_limited_429() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(429);
        assert_eq!(classify_exhausted_status(&e), ("rate-limited", true));
    }

    #[test]
    fn classify_rate_limited_by_reason() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_error_reason = Some("USAGE_LIMIT_reached".to_string());
        assert_eq!(classify_exhausted_status(&e), ("rate-limited", true));
    }

    #[test]
    fn classify_auth_failed_401() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_error_code = Some(401);
        assert_eq!(classify_exhausted_status(&e), ("auth failed", false));
    }

    #[test]
    fn classify_auth_failed_by_message() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_error_message = Some("Token has expired".to_string());
        assert_eq!(classify_exhausted_status(&e), ("auth failed", false));
    }

    #[test]
    fn classify_generic_exhausted() {
        let e = cred("anthropic", "l", "oauth", "manual");
        assert_eq!(classify_exhausted_status(&e), ("exhausted", true));
    }

    #[test]
    fn format_status_empty_when_not_exhausted() {
        let e = cred("anthropic", "l", "oauth", "manual");
        assert_eq!(format_exhausted_status(&e, 0.0), "");
    }

    #[test]
    fn format_status_auth_failed() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(401);
        e.last_error_reason = Some("unauthorized".to_string());
        let s = format_exhausted_status(&e, 0.0);
        assert_eq!(s, " auth failed unauthorized (401) (re-auth may be required)");
    }

    #[test]
    fn format_status_rate_limited_wait_window() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(429);
        // reset 90s in the future from now=0.
        e.last_error_reset_at = Some(90.0);
        let s = format_exhausted_status(&e, 0.0);
        // 90s => 1m 30s.
        assert_eq!(s, " rate-limited (429) (1m 30s left)");
    }

    #[test]
    fn format_status_ready_to_retry() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(429);
        e.last_error_reset_at = Some(10.0);
        // now far past the reset.
        let s = format_exhausted_status(&e, 100.0);
        assert_eq!(s, " rate-limited (429) (ready to retry)");
    }

    #[test]
    fn format_status_days_window() {
        let mut e = cred("anthropic", "l", "oauth", "manual");
        e.last_status = Some(STATUS_EXHAUSTED.to_string());
        e.last_error_code = Some(429);
        // 2 days + 3 hours = 183600s.
        e.last_error_reset_at = Some(2.0 * 86400.0 + 3.0 * 3600.0);
        let s = format_exhausted_status(&e, 0.0);
        assert_eq!(s, " rate-limited (429) (2d 3h left)");
    }

    #[test]
    fn strategy_order_and_descriptions() {
        let order = strategy_order();
        assert_eq!(
            order,
            [
                STRATEGY_FILL_FIRST,
                STRATEGY_ROUND_ROBIN,
                STRATEGY_LEAST_USED,
                STRATEGY_RANDOM
            ]
        );
        assert_eq!(
            strategy_description(STRATEGY_ROUND_ROBIN),
            "Cycle through keys evenly"
        );
        assert_eq!(strategy_description("nonsense"), "");
    }

    // ----- Command-level tests against a mock environment ------------------

    #[derive(Default)]
    struct MockIo {
        prompts: Vec<String>,
        getpass_inputs: Vec<String>,
        is_tty: bool,
        out: Vec<String>,
    }

    impl AuthIo for MockIo {
        fn print(&mut self, line: &str) {
            self.out.push(line.to_string());
        }
        fn prompt(&mut self, _prompt: &str) -> Option<String> {
            if self.prompts.is_empty() {
                None
            } else {
                Some(self.prompts.remove(0))
            }
        }
        fn getpass(&mut self, _prompt: &str) -> Option<String> {
            if self.getpass_inputs.is_empty() {
                None
            } else {
                Some(self.getpass_inputs.remove(0))
            }
        }
        fn is_tty(&self) -> bool {
            self.is_tty
        }
    }

    #[derive(Default)]
    struct MockEnv {
        pools: HashMap<String, Vec<PooledCredential>>,
        statuses: HashMap<String, AuthStatus>,
        strategies: HashMap<String, String>,
        registry: Vec<String>,
        unsuppressed: Vec<(String, String)>,
    }

    impl AuthEnvironment for MockEnv {
        fn pool_entries(&mut self, provider: &str) -> Vec<PooledCredential> {
            self.pools.get(provider).cloned().unwrap_or_default()
        }
        fn pool_peek(&mut self, provider: &str) -> Option<PooledCredential> {
            self.pools.get(provider).and_then(|v| v.first().cloned())
        }
        fn pool_add_entry(&mut self, provider: &str, mut entry: PooledCredential) -> PooledCredential {
            let v = self.pools.entry(provider.to_string()).or_default();
            entry.priority = v.len() as i64;
            v.push(entry.clone());
            entry
        }
        fn pool_resolve_target(
            &mut self,
            provider: &str,
            target: &str,
        ) -> (Option<usize>, Option<PooledCredential>, Option<String>) {
            let v = self.pools.get(provider).cloned().unwrap_or_default();
            if let Ok(n) = target.parse::<usize>() {
                if n >= 1 && n <= v.len() {
                    return (Some(n), Some(v[n - 1].clone()), None);
                }
            }
            (None, None, Some(format!("No credential matching \"{target}\".")))
        }
        fn pool_remove_index(&mut self, provider: &str, index: usize) -> Option<PooledCredential> {
            let v = self.pools.get_mut(provider)?;
            if index >= 1 && index <= v.len() {
                Some(v.remove(index - 1))
            } else {
                None
            }
        }
        fn pool_reset_statuses(&mut self, provider: &str) -> usize {
            let v = self.pools.get_mut(provider);
            v.map(|v| v.len()).unwrap_or(0)
        }
        fn pool_has_credentials(&mut self, provider: &str) -> bool {
            self.pools.get(provider).map(|v| !v.is_empty()).unwrap_or(false)
        }
        fn is_registered_provider(&self, provider: &str) -> bool {
            self.registry.iter().any(|p| p == provider)
        }
        fn registry_providers(&self) -> Vec<String> {
            self.registry.clone()
        }
        fn list_custom_pool_providers(&mut self) -> Vec<String> {
            Vec::new()
        }
        fn registry_base_url(&self, _provider: &str) -> Option<String> {
            Some("https://api.example/v1".to_string())
        }
        fn custom_provider_base_url(&mut self, _provider: &str) -> Option<String> {
            None
        }
        fn custom_provider_entries(&mut self) -> Vec<(String, String)> {
            Vec::new()
        }
        fn clear_suppressions(&mut self, _provider: &str) {}
        fn unsuppress_credential_source(&mut self, provider: &str, source: &str) {
            self.unsuppressed.push((provider.to_string(), source.to_string()));
        }
        fn run_removal_step(
            &mut self,
            _provider: &str,
            _removed: &PooledCredential,
        ) -> Option<RemovalOutput> {
            None
        }
        fn get_auth_status(&mut self, provider: &str) -> AuthStatus {
            self.statuses.get(provider).cloned().unwrap_or_default()
        }
        fn logout(&mut self, _provider: Option<&str>) {}
        fn login_spotify(&mut self) {}
        fn get_pool_strategy(&mut self, provider: &str) -> String {
            self.strategies
                .get(provider)
                .cloned()
                .unwrap_or_else(|| STRATEGY_FILL_FIRST.to_string())
        }
        fn set_pool_strategy(&mut self, provider: &str, strategy: &str) {
            self.strategies
                .insert(provider.to_string(), strategy.to_string());
        }
    }

    struct NoFlows;
    impl OauthFlows for NoFlows {
        fn anthropic_login(&mut self) -> Result<OauthCredentials, String> {
            Err("not used".to_string())
        }
        fn codex_device_code_login(&mut self) -> Result<OauthCredentials, String> {
            Err("not used".to_string())
        }
        fn gemini_login(&mut self) -> Result<OauthCredentials, String> {
            Err("not used".to_string())
        }
        fn qwen_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
            Err("not used".to_string())
        }
        fn minimax_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
            Err("not used".to_string())
        }
        fn nous_login(&mut self, _label: Option<&str>) -> Result<String, String> {
            Err("not used".to_string())
        }
    }

    #[test]
    fn add_api_key_unknown_provider_errors() {
        let mut env = MockEnv::default();
        let mut io = MockIo::default();
        let mut flows = NoFlows;
        let args = AuthAddArgs {
            provider: "bogus".to_string(),
            ..Default::default()
        };
        let err = auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap_err();
        assert_eq!(err.0, "Unknown provider: bogus");
    }

    #[test]
    fn add_api_key_with_explicit_key_and_label() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        let mut flows = NoFlows;
        let args = AuthAddArgs {
            provider: "groq".to_string(),
            auth_type: Some("api_key".to_string()),
            api_key: Some("  sk-test  ".to_string()),
            label: Some(" my-key ".to_string()),
            ..Default::default()
        };
        auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap();
        let entries = env.pool_entries("groq");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].access_token, "sk-test");
        assert_eq!(entries[0].label, "my-key");
        assert_eq!(entries[0].auth_type, AUTH_TYPE_API_KEY);
        assert_eq!(entries[0].base_url.as_deref(), Some("https://api.example/v1"));
        assert_eq!(io.out, vec!["Added groq credential #1: \"my-key\""]);
    }

    #[test]
    fn add_api_key_default_label_non_tty() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        io.is_tty = false;
        let mut flows = NoFlows;
        let args = AuthAddArgs {
            provider: "groq".to_string(),
            auth_type: Some("api-key".to_string()),
            api_key: Some("sk".to_string()),
            ..Default::default()
        };
        auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap();
        let entries = env.pool_entries("groq");
        assert_eq!(entries[0].label, "api-key-1");
    }

    #[test]
    fn add_api_key_prompts_when_missing() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        io.getpass_inputs.push("  pasted  ".to_string());
        io.is_tty = false;
        let mut flows = NoFlows;
        let args = AuthAddArgs {
            provider: "groq".to_string(),
            auth_type: Some("api_key".to_string()),
            ..Default::default()
        };
        auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap();
        assert_eq!(env.pool_entries("groq")[0].access_token, "pasted");
    }

    #[test]
    fn add_api_key_no_key_errors() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        io.getpass_inputs.push("   ".to_string());
        let mut flows = NoFlows;
        let args = AuthAddArgs {
            provider: "groq".to_string(),
            auth_type: Some("api_key".to_string()),
            ..Default::default()
        };
        let err = auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap_err();
        assert_eq!(err.0, "No API key provided.");
    }

    #[test]
    fn add_anthropic_oauth_uses_flow() {
        struct AnthFlow;
        impl OauthFlows for AnthFlow {
            fn anthropic_login(&mut self) -> Result<OauthCredentials, String> {
                Ok(OauthCredentials {
                    access_token: "oauth-tok".to_string(),
                    refresh_token: Some("ref".to_string()),
                    expires_at_ms: Some(123),
                    ..Default::default()
                })
            }
            fn codex_device_code_login(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn gemini_login(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn qwen_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn minimax_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn nous_login(&mut self, _label: Option<&str>) -> Result<String, String> {
                unreachable!()
            }
        }
        let mut env = MockEnv::default();
        env.registry.push("anthropic".to_string());
        let mut io = MockIo::default();
        let mut flows = AnthFlow;
        let args = AuthAddArgs {
            provider: "anthropic".to_string(),
            label: Some("work".to_string()),
            ..Default::default()
        };
        auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap();
        let entries = env.pool_entries("anthropic");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].auth_type, AUTH_TYPE_OAUTH);
        assert_eq!(entries[0].source, "manual:hermes_pkce");
        assert_eq!(entries[0].refresh_token.as_deref(), Some("ref"));
        assert_eq!(entries[0].expires_at_ms, Some(123));
        assert_eq!(
            io.out,
            vec!["Added anthropic OAuth credential #1: \"work\""]
        );
    }

    #[test]
    fn codex_oauth_unsuppresses_device_code() {
        struct CodexFlow;
        impl OauthFlows for CodexFlow {
            fn anthropic_login(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn codex_device_code_login(&mut self) -> Result<OauthCredentials, String> {
                Ok(OauthCredentials {
                    access_token: "ctok".to_string(),
                    refresh_token: Some("cref".to_string()),
                    base_url: Some("https://codex/v1".to_string()),
                    last_refresh: Some("2026-01-01".to_string()),
                    ..Default::default()
                })
            }
            fn gemini_login(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn qwen_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn minimax_runtime_credentials(&mut self) -> Result<OauthCredentials, String> {
                unreachable!()
            }
            fn nous_login(&mut self, _label: Option<&str>) -> Result<String, String> {
                unreachable!()
            }
        }
        let mut env = MockEnv::default();
        env.registry.push("openai-codex".to_string());
        let mut io = MockIo::default();
        let mut flows = CodexFlow;
        let args = AuthAddArgs {
            provider: "openai-codex".to_string(),
            label: Some("codex".to_string()),
            ..Default::default()
        };
        auth_add_command(&args, &mut env, &mut io, &mut flows).unwrap();
        assert_eq!(
            env.unsuppressed,
            vec![("openai-codex".to_string(), "device_code".to_string())]
        );
        let entries = env.pool_entries("openai-codex");
        assert_eq!(entries[0].base_url.as_deref(), Some("https://codex/v1"));
        assert_eq!(entries[0].last_refresh.as_deref(), Some("2026-01-01"));
    }

    #[test]
    fn remove_command_happy_path() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        env.pools.insert(
            "groq".to_string(),
            vec![cred("groq", "k1", "api_key", "manual")],
        );
        let mut io = MockIo::default();
        let args = AuthRemoveArgs {
            provider: "groq".to_string(),
            target: Some("1".to_string()),
            index: None,
        };
        auth_remove_command(&args, &mut env, &mut io).unwrap();
        assert!(env.pool_entries("groq").is_empty());
        assert_eq!(io.out, vec!["Removed groq credential #1 (k1)"]);
    }

    #[test]
    fn remove_command_not_found_errors() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        let args = AuthRemoveArgs {
            provider: "groq".to_string(),
            target: Some("9".to_string()),
            index: None,
        };
        let err = auth_remove_command(&args, &mut env, &mut io).unwrap_err();
        assert!(err.0.contains("Provider: groq."));
    }

    #[test]
    fn reset_command_prints_count() {
        let mut env = MockEnv::default();
        env.pools.insert(
            "groq".to_string(),
            vec![
                cred("groq", "a", "api_key", "manual"),
                cred("groq", "b", "api_key", "manual"),
            ],
        );
        let mut io = MockIo::default();
        auth_reset_command(
            &AuthProviderArgs {
                provider: Some("groq".to_string()),
            },
            &mut env,
            &mut io,
        );
        assert_eq!(io.out, vec!["Reset status on 2 groq credentials"]);
    }

    #[test]
    fn status_logged_out_with_reason() {
        let mut env = MockEnv::default();
        env.statuses.insert(
            "spotify".to_string(),
            AuthStatus {
                logged_in: false,
                error: Some("token expired".to_string()),
                ..Default::default()
            },
        );
        let mut io = MockIo::default();
        auth_status_command(
            &AuthProviderArgs {
                provider: Some("spotify".to_string()),
            },
            &mut env,
            &mut io,
        )
        .unwrap();
        assert_eq!(io.out, vec!["spotify: logged out (token expired)"]);
    }

    #[test]
    fn status_logged_in_prints_fields() {
        let mut env = MockEnv::default();
        env.statuses.insert(
            "spotify".to_string(),
            AuthStatus {
                logged_in: true,
                auth_type: Some("oauth".to_string()),
                scope: Some("user-read".to_string()),
                ..Default::default()
            },
        );
        let mut io = MockIo::default();
        auth_status_command(
            &AuthProviderArgs {
                provider: Some("spotify".to_string()),
            },
            &mut env,
            &mut io,
        )
        .unwrap();
        assert_eq!(
            io.out,
            vec![
                "spotify: logged in",
                "  auth_type: oauth",
                "  scope: user-read",
            ]
        );
    }

    #[test]
    fn status_requires_provider() {
        let mut env = MockEnv::default();
        let mut io = MockIo::default();
        let err = auth_status_command(
            &AuthProviderArgs { provider: None },
            &mut env,
            &mut io,
        )
        .unwrap_err();
        assert!(err.0.contains("Provider is required"));
    }

    #[test]
    fn spotify_unknown_action_errors() {
        let mut env = MockEnv::default();
        let mut io = MockIo::default();
        let err = auth_spotify_command(Some("frobnicate"), &mut env, &mut io).unwrap_err();
        assert_eq!(err.0, "Unknown Spotify auth action: frobnicate");
    }

    #[test]
    fn interactive_strategy_sets_choice() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        // pick_provider prompt, then strategy choice.
        io.prompts.push("groq".to_string());
        io.prompts.push("2".to_string());
        interactive_strategy(&mut env, &mut io).unwrap();
        assert_eq!(env.strategies.get("groq").map(String::as_str), Some(STRATEGY_ROUND_ROBIN));
        assert!(io.out.iter().any(|l| l == "Set groq strategy to: round_robin"));
    }

    #[test]
    fn interactive_strategy_invalid_choice() {
        let mut env = MockEnv::default();
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        io.prompts.push("groq".to_string());
        io.prompts.push("99".to_string());
        interactive_strategy(&mut env, &mut io).unwrap();
        assert!(io.out.iter().any(|l| l == "Invalid choice."));
        assert!(env.strategies.is_empty());
    }

    #[test]
    fn auth_command_dispatch_list() {
        let mut env = MockEnv::default();
        env.pools.insert(
            "groq".to_string(),
            vec![cred("groq", "k1", "api_key", "manual:device")],
        );
        env.registry.push("groq".to_string());
        let mut io = MockIo::default();
        let mut flows = NoFlows;
        let args = AuthArgs {
            auth_action: Some("list".to_string()),
            provider_args: AuthProviderArgs {
                provider: Some("groq".to_string()),
            },
            ..Default::default()
        };
        auth_command(&args, &mut env, &mut io, &mut flows, 0.0, None).unwrap();
        assert!(io.out.iter().any(|l| l.starts_with("groq (1 credentials):")));
        assert!(io.out.iter().any(|l| l.contains("device")));
    }
}
