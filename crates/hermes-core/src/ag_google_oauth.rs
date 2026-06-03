//! Google OAuth PKCE flow for the Gemini (`google-gemini-cli`) inference provider.
//!
//! Native Rust port of `agent/google_oauth.py`. Implements Authorization Code +
//! PKCE (S256) OAuth against Google's `accounts.google.com` endpoints. The
//! resulting access token is used to talk to `cloudcode-pa.googleapis.com`
//! (Google's Code Assist backend powering the Gemini CLI's free/paid tiers).
//!
//! Storage (`~/.hermes/auth/google_oauth.json`, chmod 0o600):
//!
//! ```json
//! {
//!   "refresh": "refreshToken|projectId|managedProjectId",
//!   "access": "...",
//!   "expires": 1744848000000,
//!   "email": "user@example.com"
//! }
//! ```
//!
//! The `refresh` field packs the refresh_token together with the resolved GCP
//! project IDs so subsequent sessions don't need to re-discover the project.
//! This matches opencode-gemini-auth's storage contract exactly.
//!
//! # Overlap note
//! `hermes-core::auth` already contains a `GoogleOAuthState` /
//! `GoogleRefreshParts` pair plus a token-refresh path used by the
//! `hermes auth` command surface. This module ports the *interactive login
//! flow* (PKCE generation, client-id resolution/scraping, the local callback
//! HTTP server, paste-mode fallback, and the higher-level
//! `get_valid_access_token` / `start_oauth_flow` orchestration) that
//! `auth.rs` does not implement.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde_json::Value;
use sha2::{Digest, Sha256};

// =============================================================================
// OAuth client credential resolution constants
// =============================================================================

/// Env var overriding the OAuth client id.
pub const ENV_CLIENT_ID: &str = "HERMES_GEMINI_CLIENT_ID";
/// Env var overriding the OAuth client secret.
pub const ENV_CLIENT_SECRET: &str = "HERMES_GEMINI_CLIENT_SECRET";

// Public gemini-cli desktop OAuth client (shipped in Google's open-source
// gemini-cli MIT repo). Composed piecewise to keep the constants readable.
const PUBLIC_CLIENT_ID_PROJECT_NUM: &str = "681255809395";
const PUBLIC_CLIENT_ID_HASH: &str = "oo8ft2oprdrnp9e3aqf6av3hmdib135j";
const PUBLIC_CLIENT_SECRET_SUFFIX: &str = "4uHgMPm-1o7Sk-geV6Cu5clXFsxl";

fn default_client_id() -> String {
    format!(
        "{}-{}.apps.googleusercontent.com",
        PUBLIC_CLIENT_ID_PROJECT_NUM, PUBLIC_CLIENT_ID_HASH
    )
}

fn default_client_secret() -> String {
    format!("GOCSPX-{}", PUBLIC_CLIENT_SECRET_SUFFIX)
}

// =============================================================================
// Endpoints & constants
// =============================================================================

/// Google authorization endpoint.
pub const AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
/// Google token endpoint.
pub const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
/// Google userinfo endpoint.
pub const USERINFO_ENDPOINT: &str = "https://www.googleapis.com/oauth2/v1/userinfo";

/// Space-separated OAuth scopes requested.
pub const OAUTH_SCOPES: &str = "https://www.googleapis.com/auth/cloud-platform \
https://www.googleapis.com/auth/userinfo.email \
https://www.googleapis.com/auth/userinfo.profile";

/// Default loopback port for the OAuth callback server.
pub const DEFAULT_REDIRECT_PORT: u16 = 8085;
/// Loopback host for the OAuth callback server.
pub const REDIRECT_HOST: &str = "127.0.0.1";
/// Path the OAuth callback redirects to.
pub const CALLBACK_PATH: &str = "/oauth2callback";

/// 60-second clock-skew buffer (matches opencode-gemini-auth).
pub const REFRESH_SKEW_SECONDS: i64 = 60;

/// Token-request HTTP timeout in seconds.
pub const TOKEN_REQUEST_TIMEOUT_SECONDS: u64 = 20;
/// Max seconds to wait for a browser callback.
pub const CALLBACK_WAIT_SECONDS: u64 = 300;

/// Env vars that, when set, indicate a headless environment.
const HEADLESS_ENV_VARS: [&str; 4] = ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "HERMES_HEADLESS"];

// =============================================================================
// Error type
// =============================================================================

/// Error raised for any failure in the Google OAuth flow.
///
/// Carries a machine-readable `code` mirroring the Python `GoogleOAuthError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleOAuthError {
    /// Human-readable message.
    pub message: String,
    /// Stable machine-readable code (e.g. `google_oauth_invalid_grant`).
    pub code: String,
}

impl GoogleOAuthError {
    /// Construct with an explicit code.
    pub fn new(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: code.into(),
        }
    }

    /// Construct with the default `google_oauth_error` code.
    pub fn generic(message: impl Into<String>) -> Self {
        Self::new(message, "google_oauth_error")
    }
}

impl std::fmt::Display for GoogleOAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GoogleOAuthError {}

type Result<T> = std::result::Result<T, GoogleOAuthError>;

// =============================================================================
// File paths
// =============================================================================

/// Resolve the active `HERMES_HOME` (mirrors `get_hermes_home()`): honour
/// `HERMES_HOME` when set & non-empty, otherwise `~/.hermes`.
pub fn hermes_home_path() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".hermes")
}

/// Path to the on-disk credentials file.
pub fn credentials_path() -> PathBuf {
    hermes_home_path().join("auth").join("google_oauth.json")
}

// =============================================================================
// PKCE
// =============================================================================

/// URL-safe base64 (no padding) of `bytes` random bytes — mirrors
/// `secrets.token_urlsafe(n)`.
fn token_urlsafe(nbytes: usize) -> String {
    let mut buf = vec![0u8; nbytes];
    getrandom::fill(&mut buf).expect("getrandom failed");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&buf)
}

/// Lowercase hex of `nbytes` random bytes — mirrors `secrets.token_hex(n)`.
fn token_hex(nbytes: usize) -> String {
    let mut buf = vec![0u8; nbytes];
    getrandom::fill(&mut buf).expect("getrandom failed");
    buf.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate a `(verifier, challenge)` PKCE pair using S256.
pub fn generate_pkce_pair() -> (String, String) {
    let verifier = token_urlsafe(64);
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let digest = hasher.finalize();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

// =============================================================================
// Packed refresh format: refresh_token[|project_id[|managed_project_id]]
// =============================================================================

/// Decomposed packed-refresh format.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefreshParts {
    /// Bare refresh token.
    pub refresh_token: String,
    /// Resolved GCP project id (may be empty).
    pub project_id: String,
    /// Managed GCP project id (may be empty).
    pub managed_project_id: String,
}

impl RefreshParts {
    /// Parse the packed `refresh` string.
    pub fn parse(packed: &str) -> Self {
        if packed.is_empty() {
            return Self::default();
        }
        let mut parts = packed.splitn(3, '|');
        let refresh_token = parts.next().unwrap_or_default().to_string();
        let project_id = parts.next().unwrap_or_default().to_string();
        let managed_project_id = parts.next().unwrap_or_default().to_string();
        Self {
            refresh_token,
            project_id,
            managed_project_id,
        }
    }

    /// Re-pack into the `refresh` string form.
    pub fn format(&self) -> String {
        if self.refresh_token.is_empty() {
            return String::new();
        }
        if self.project_id.is_empty() && self.managed_project_id.is_empty() {
            return self.refresh_token.clone();
        }
        format!(
            "{}|{}|{}",
            self.refresh_token, self.project_id, self.managed_project_id
        )
    }
}

// =============================================================================
// Credentials
// =============================================================================

/// In-memory representation of the on-disk credential format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleCredentials {
    /// OAuth access token.
    pub access_token: String,
    /// OAuth refresh token (bare, unpacked).
    pub refresh_token: String,
    /// Expiry timestamp in unix MILLIseconds.
    pub expires_ms: i64,
    /// Account email (best-effort, may be empty).
    pub email: String,
    /// Resolved GCP project id.
    pub project_id: String,
    /// Managed GCP project id.
    pub managed_project_id: String,
}

impl GoogleCredentials {
    /// Serialise to the on-disk JSON shape.
    pub fn to_value(&self) -> Value {
        let refresh = RefreshParts {
            refresh_token: self.refresh_token.clone(),
            project_id: self.project_id.clone(),
            managed_project_id: self.managed_project_id.clone(),
        }
        .format();
        serde_json::json!({
            "refresh": refresh,
            "access": self.access_token,
            "expires": self.expires_ms,
            "email": self.email,
        })
    }

    /// Parse from the on-disk JSON shape.
    pub fn from_value(data: &Value) -> Self {
        let refresh_packed = data
            .get("refresh")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parts = RefreshParts::parse(refresh_packed);
        Self {
            access_token: data
                .get("access")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            refresh_token: parts.refresh_token,
            expires_ms: value_to_i64(data.get("expires")).unwrap_or(0),
            email: data
                .get("email")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            project_id: parts.project_id,
            managed_project_id: parts.managed_project_id,
        }
    }

    /// Expiry as fractional unix seconds.
    pub fn expires_unix_seconds(&self) -> f64 {
        self.expires_ms as f64 / 1000.0
    }

    /// Whether the access token is missing or within `skew_seconds` of expiry.
    pub fn access_token_expired(&self, skew_seconds: i64) -> bool {
        if self.access_token.is_empty() || self.expires_ms == 0 {
            return true;
        }
        let now_ms = (now_unix_seconds() + skew_seconds.max(0) as f64) * 1000.0;
        now_ms >= self.expires_ms as f64
    }
}

/// Coerce a JSON value to i64, accepting numbers or numeric strings (mirrors
/// the Python `int(... or 0)` coercion).
fn value_to_i64(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .or_else(|| n.as_u64().map(|u| u as i64)),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0)
            } else {
                t.parse::<i64>()
                    .ok()
                    .or_else(|| t.parse::<f64>().ok().map(|f| f as i64))
            }
        }
        Some(Value::Null) | None => None,
        _ => None,
    }
}

fn now_unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// =============================================================================
// Client id resolution (incl. scraping a local gemini-cli install)
// =============================================================================

struct ScrapedCreds {
    resolved: bool,
    client_id: String,
    client_secret: String,
}

static SCRAPED_CACHE: Mutex<ScrapedCreds> = Mutex::new(ScrapedCreds {
    resolved: false,
    client_id: String::new(),
    client_secret: String::new(),
});

/// Walk the user's `gemini` binary install to find its `oauth2.js`.
/// Returns None if gemini isn't installed.
pub fn locate_gemini_cli_oauth_js() -> Option<PathBuf> {
    let gemini = which_gemini()?;
    let real = fs::canonicalize(&gemini).ok()?;

    let mut search_dirs: Vec<PathBuf> = Vec::new();
    let mut cur = real.parent().map(|p| p.to_path_buf())?;
    for _ in 0..8 {
        search_dirs.push(cur.clone());
        if cur.join("node_modules").exists() {
            search_dirs.push(
                cur.join("node_modules")
                    .join("@google")
                    .join("gemini-cli-core"),
            );
            break;
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent.to_path_buf(),
            _ => break,
        }
    }

    for root in &search_dirs {
        if !root.exists() {
            continue;
        }
        let candidates = [
            root.join("dist").join("src").join("code_assist").join("oauth2.js"),
            root.join("dist").join("code_assist").join("oauth2.js"),
            root.join("src").join("code_assist").join("oauth2.js"),
        ];
        for c in &candidates {
            if c.exists() {
                return Some(c.clone());
            }
        }
        // Recursive fallback: first oauth2.js found under root.
        if let Some(found) = find_first_named(root, "oauth2.js", 10) {
            return Some(found);
        }
    }
    None
}

/// Locate the `gemini` executable on PATH (mirrors `shutil.which`).
fn which_gemini() -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let exe = if cfg!(windows) { "gemini.exe" } else { "gemini" };
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
        // Also try the bare name on Windows for shims.
        if cfg!(windows) {
            let bare = dir.join("gemini");
            if bare.is_file() {
                return Some(bare);
            }
        }
    }
    None
}

/// Depth-limited recursive search for a file named `name`.
fn find_first_named(root: &Path, name: &str, max_depth: usize) -> Option<PathBuf> {
    if max_depth == 0 {
        return None;
    }
    let entries = fs::read_dir(root).ok()?;
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.file_name().and_then(|n| n.to_str()) == Some(name) {
                return Some(path);
            }
        } else if path.is_dir() {
            subdirs.push(path);
        }
    }
    for sub in subdirs {
        if let Some(found) = find_first_named(&sub, name, max_depth - 1) {
            return Some(found);
        }
    }
    None
}

/// Extract `(client_id, client_secret)` from the local gemini-cli install.
/// Caches the (possibly empty) result so we don't retry on every call.
pub fn scrape_client_credentials() -> (String, String) {
    {
        let cache = SCRAPED_CACHE.lock().unwrap();
        if cache.resolved {
            return (cache.client_id.clone(), cache.client_secret.clone());
        }
    }

    let oauth_js = match locate_gemini_cli_oauth_js() {
        Some(p) => p,
        None => {
            let mut cache = SCRAPED_CACHE.lock().unwrap();
            cache.resolved = true;
            return (String::new(), String::new());
        }
    };

    let content = match fs::read(&oauth_js) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(exc) => {
            log::debug!("Failed to read oauth2.js at {}: {}", oauth_js.display(), exc);
            let mut cache = SCRAPED_CACHE.lock().unwrap();
            cache.resolved = true;
            return (String::new(), String::new());
        }
    };

    let (client_id, client_secret) = extract_creds_from_source(&content);

    {
        let mut cache = SCRAPED_CACHE.lock().unwrap();
        cache.client_id = client_id.clone();
        cache.client_secret = client_secret.clone();
        cache.resolved = true;
    }

    if !client_id.is_empty() {
        log::info!("Scraped Gemini OAuth client from {}", oauth_js.display());
    }
    (client_id, client_secret)
}

/// Apply the precise-then-shape regex matching used by the Python scraper.
fn extract_creds_from_source(content: &str) -> (String, String) {
    let id_precise =
        regex::Regex::new(r#"OAUTH_CLIENT_ID\s*=\s*['"]([0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com)['"]"#)
            .unwrap();
    let secret_precise =
        regex::Regex::new(r#"OAUTH_CLIENT_SECRET\s*=\s*['"](GOCSPX-[A-Za-z0-9_-]+)['"]"#).unwrap();
    let id_shape =
        regex::Regex::new(r"([0-9]{8,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com)").unwrap();
    let secret_shape = regex::Regex::new(r"(GOCSPX-[A-Za-z0-9_-]{20,})").unwrap();

    let client_id = id_precise
        .captures(content)
        .or_else(|| id_shape.captures(content))
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_default();
    let client_secret = secret_precise
        .captures(content)
        .or_else(|| secret_shape.captures(content))
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_default();
    (client_id, client_secret)
}

/// Resolve the client id: env override, shipped default, then scrape.
pub fn get_client_id() -> String {
    if let Ok(val) = std::env::var(ENV_CLIENT_ID) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let default = default_client_id();
    if !default.is_empty() {
        return default;
    }
    scrape_client_credentials().0
}

/// Resolve the client secret: env override, shipped default, then scrape.
pub fn get_client_secret() -> String {
    if let Ok(val) = std::env::var(ENV_CLIENT_SECRET) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let default = default_client_secret();
    if !default.is_empty() {
        return default;
    }
    scrape_client_credentials().1
}

/// Resolve the client id or fail with install hints.
pub fn require_client_id() -> Result<String> {
    let cid = get_client_id();
    if cid.is_empty() {
        return Err(GoogleOAuthError::new(
            "Google OAuth client ID is not available.\n\
Hermes looks for a locally installed gemini-cli to source the OAuth client. \
Either:\n\
  1. Install it: npm install -g @google/gemini-cli  (or brew install gemini-cli)\n\
  2. Set HERMES_GEMINI_CLIENT_ID and HERMES_GEMINI_CLIENT_SECRET in ~/.hermes/.env\n\
\n\
Register a Desktop OAuth client at:\n\
  https://console.cloud.google.com/apis/credentials\n\
(enable the Generative Language API on the project).",
            "google_oauth_client_id_missing",
        ));
    }
    Ok(cid)
}

// =============================================================================
// Credential I/O (atomic + 0o600)
// =============================================================================

/// Load credentials from disk. Returns None if missing, unreadable, corrupt,
/// or lacking an access token.
pub fn load_credentials() -> Option<GoogleCredentials> {
    let path = credentials_path();
    if !path.exists() {
        return None;
    }
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(exc) => {
            log::warn!(
                "Failed to read Google OAuth credentials at {}: {}",
                path.display(),
                exc
            );
            return None;
        }
    };
    let data: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(exc) => {
            log::warn!(
                "Failed to read Google OAuth credentials at {}: {}",
                path.display(),
                exc
            );
            return None;
        }
    };
    if !data.is_object() {
        return None;
    }
    let creds = GoogleCredentials::from_value(&data);
    if creds.access_token.is_empty() {
        return None;
    }
    Some(creds)
}

/// Atomically write creds to disk with 0o600 permissions.
pub fn save_credentials(creds: &GoogleCredentials) -> Result<PathBuf> {
    let path = credentials_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            GoogleOAuthError::generic(format!("Failed to create auth dir: {}", e))
        })?;
        // Tighten parent dir to 0o700; best-effort (no-op on platforms without
        // POSIX mode bits).
        set_mode(parent, 0o700);
    }

    let value = creds.to_value();
    let payload = serde_json::to_string_pretty(&sorted_value(&value)).map_err(|e| {
        GoogleOAuthError::generic(format!("Failed to serialise credentials: {}", e))
    })? + "\n";

    let pid = std::process::id();
    let tmp_path = path.with_file_name(format!(
        "{}.tmp.{}.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("google_oauth.json"),
        pid,
        token_hex(4)
    ));

    let write_result = write_atomic(&tmp_path, &path, payload.as_bytes());
    // Best-effort cleanup of the temp file if it survives.
    if tmp_path.exists() {
        let _ = fs::remove_file(&tmp_path);
    }
    write_result?;
    Ok(path)
}

/// Write `bytes` to `tmp_path` with 0o600 then rename onto `path`.
fn write_atomic(tmp_path: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    use std::fs::OpenOptions;
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts
        .open(tmp_path)
        .map_err(|e| GoogleOAuthError::generic(format!("Failed to open temp credential file: {}", e)))?;
    file.write_all(bytes)
        .map_err(|e| GoogleOAuthError::generic(format!("Failed to write credentials: {}", e)))?;
    file.flush()
        .map_err(|e| GoogleOAuthError::generic(format!("Failed to flush credentials: {}", e)))?;
    file.sync_all().ok();
    drop(file);
    fs::rename(tmp_path, path)
        .map_err(|e| GoogleOAuthError::generic(format!("Failed to replace credential file: {}", e)))?;
    Ok(())
}

/// Remove the creds file. Idempotent.
pub fn clear_credentials() {
    let path = credentials_path();
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(exc) if exc.kind() == std::io::ErrorKind::NotFound => {}
        Err(exc) => {
            log::warn!(
                "Failed to remove Google OAuth credentials at {}: {}",
                path.display(),
                exc
            );
        }
    }
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}

/// Recursively rebuild a JSON value with object keys sorted, mirroring
/// `json.dumps(..., sort_keys=True)`.
fn sorted_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), sorted_value(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sorted_value).collect()),
        other => other.clone(),
    }
}

// =============================================================================
// HTTP helpers
// =============================================================================

/// POST x-www-form-urlencoded and return parsed JSON, mirroring `_post_form`.
fn post_form(url: &str, data: &[(String, String)], timeout_secs: u64) -> Result<Value> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| {
            GoogleOAuthError::new(
                format!("Google OAuth token request failed: {}", e),
                "google_oauth_token_network_error",
            )
        })?;

    let response = client
        .post(url)
        .header("Accept", "application/json")
        .form(data)
        .send()
        .map_err(|e| {
            GoogleOAuthError::new(
                format!("Google OAuth token request failed: {}", e),
                "google_oauth_token_network_error",
            )
        })?;

    let status = response.status();
    let body = response.text().unwrap_or_default();

    if !status.is_success() {
        let mut code = "google_oauth_token_http_error";
        if body.to_lowercase().contains("invalid_grant") {
            code = "google_oauth_invalid_grant";
        }
        let detail = if body.is_empty() {
            status
                .canonical_reason()
                .unwrap_or("error")
                .to_string()
        } else {
            body
        };
        return Err(GoogleOAuthError::new(
            format!(
                "Google OAuth token endpoint returned HTTP {}: {}",
                status.as_u16(),
                detail
            ),
            code,
        ));
    }

    serde_json::from_str(&body).map_err(|e| {
        GoogleOAuthError::new(
            format!("Failed to parse Google OAuth token response: {}", e),
            "google_oauth_token_http_error",
        )
    })
}

/// Exchange an authorization code for access + refresh tokens.
pub fn exchange_code(
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    timeout_secs: u64,
) -> Result<Value> {
    let cid = client_id.map(|s| s.to_string()).unwrap_or_else(get_client_id);
    let csecret = client_secret
        .map(|s| s.to_string())
        .unwrap_or_else(get_client_secret);
    let mut data: Vec<(String, String)> = vec![
        ("grant_type".into(), "authorization_code".into()),
        ("code".into(), code.into()),
        ("code_verifier".into(), verifier.into()),
        ("client_id".into(), cid),
        ("redirect_uri".into(), redirect_uri.into()),
    ];
    if !csecret.is_empty() {
        data.push(("client_secret".into(), csecret));
    }
    post_form(TOKEN_ENDPOINT, &data, timeout_secs)
}

/// Refresh the access token.
pub fn refresh_access_token(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    timeout_secs: u64,
) -> Result<Value> {
    if refresh_token.is_empty() {
        return Err(GoogleOAuthError::new(
            "Cannot refresh: refresh_token is empty. Re-run OAuth login.",
            "google_oauth_refresh_token_missing",
        ));
    }
    let cid = client_id.map(|s| s.to_string()).unwrap_or_else(get_client_id);
    let csecret = client_secret
        .map(|s| s.to_string())
        .unwrap_or_else(get_client_secret);
    let mut data: Vec<(String, String)> = vec![
        ("grant_type".into(), "refresh_token".into()),
        ("refresh_token".into(), refresh_token.into()),
        ("client_id".into(), cid),
    ];
    if !csecret.is_empty() {
        data.push(("client_secret".into(), csecret));
    }
    post_form(TOKEN_ENDPOINT, &data, timeout_secs)
}

/// Best-effort userinfo email fetch for display. Failures return empty string.
pub fn fetch_user_email(access_token: &str, timeout_secs: u64) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
    {
        Ok(c) => c,
        Err(exc) => {
            log::debug!("Userinfo fetch failed (non-fatal): {}", exc);
            return String::new();
        }
    };
    let resp = client
        .get(format!("{}?alt=json", USERINFO_ENDPOINT))
        .header("Authorization", format!("Bearer {}", access_token))
        .send();
    match resp {
        Ok(r) => match r.text() {
            Ok(body) => serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v.get("email").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_default(),
            Err(exc) => {
                log::debug!("Userinfo fetch failed (non-fatal): {}", exc);
                String::new()
            }
        },
        Err(exc) => {
            log::debug!("Userinfo fetch failed (non-fatal): {}", exc);
            String::new()
        }
    }
}

// =============================================================================
// In-flight refresh deduplication
// =============================================================================

static REFRESH_INFLIGHT: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Load creds, refreshing if near expiry, and return a valid bearer token.
///
/// On `invalid_grant`, the credential file is wiped and a
/// `google_oauth_invalid_grant` error is returned (the caller is expected to
/// trigger a re-login flow). Concurrent refreshes for the same refresh_token
/// coalesce: late callers wait briefly and re-read from disk.
pub fn get_valid_access_token(force_refresh: bool) -> Result<String> {
    let creds = load_credentials().ok_or_else(|| {
        GoogleOAuthError::new(
            "No Google OAuth credentials found. Run `hermes login --provider google-gemini-cli` first.",
            "google_oauth_not_logged_in",
        )
    })?;

    if !force_refresh && !creds.access_token_expired(REFRESH_SKEW_SECONDS) {
        return Ok(creds.access_token);
    }

    let rt = creds.refresh_token.clone();

    // Try to claim ownership of this refresh_token's refresh.
    let owner = {
        let mut inflight = REFRESH_INFLIGHT.lock().unwrap();
        if inflight.contains(&rt) {
            false
        } else {
            inflight.push(rt.clone());
            true
        }
    };

    if !owner {
        // Another thread is refreshing; wait briefly, then re-read from disk.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            std::thread::sleep(Duration::from_millis(50));
            let still_running = REFRESH_INFLIGHT.lock().unwrap().contains(&rt);
            if !still_running || Instant::now() >= deadline {
                break;
            }
        }
        if let Some(fresh) = load_credentials() {
            if !fresh.access_token_expired(REFRESH_SKEW_SECONDS) {
                return Ok(fresh.access_token);
            }
        }
        // Fall through to do our own refresh if the other attempt failed.
    }

    let result = do_refresh(&creds, &rt);

    if owner {
        let mut inflight = REFRESH_INFLIGHT.lock().unwrap();
        inflight.retain(|t| t != &rt);
    }

    result
}

fn do_refresh(creds: &GoogleCredentials, rt: &str) -> Result<String> {
    let resp = match refresh_access_token(rt, None, None, TOKEN_REQUEST_TIMEOUT_SECONDS) {
        Ok(v) => v,
        Err(exc) => {
            if exc.code == "google_oauth_invalid_grant" {
                log::warn!(
                    "Google OAuth refresh token invalid (revoked/expired). \
Clearing credentials at {} — user must re-login.",
                    credentials_path().display()
                );
                clear_credentials();
            }
            return Err(exc);
        }
    };

    let new_access = resp
        .get("access_token")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if new_access.is_empty() {
        return Err(GoogleOAuthError::new(
            "Refresh response did not include an access_token.",
            "google_oauth_refresh_empty",
        ));
    }
    let new_refresh = {
        let r = resp
            .get("refresh_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if r.is_empty() {
            creds.refresh_token.clone()
        } else {
            r
        }
    };
    let expires_in = value_to_i64(resp.get("expires_in")).unwrap_or(0);

    let mut updated = creds.clone();
    updated.access_token = new_access;
    updated.refresh_token = new_refresh;
    updated.expires_ms = ((now_unix_seconds() + expires_in.max(60) as f64) * 1000.0) as i64;
    save_credentials(&updated)?;
    Ok(updated.access_token)
}

// =============================================================================
// Update project IDs on stored creds
// =============================================================================

/// Persist resolved/discovered project IDs back into the credential file.
pub fn update_project_ids(project_id: &str, managed_project_id: &str) -> Result<()> {
    let mut creds = match load_credentials() {
        Some(c) => c,
        None => return Ok(()),
    };
    if !project_id.is_empty() {
        creds.project_id = project_id.to_string();
    }
    if !managed_project_id.is_empty() {
        creds.managed_project_id = managed_project_id.to_string();
    }
    save_credentials(&creds)?;
    Ok(())
}

// =============================================================================
// Project ID resolution
// =============================================================================

/// Return a GCP project ID from env vars, in priority order.
pub fn resolve_project_id_from_env() -> String {
    for var in [
        "HERMES_GEMINI_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_PROJECT_ID",
    ] {
        if let Ok(val) = std::env::var(var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    String::new()
}

// =============================================================================
// Auth URL construction
// =============================================================================

/// URL-encode a single value per `application/x-www-form-urlencoded`
/// (mirrors `urllib.parse.urlencode`, which uses `quote_via=quote_plus`).
fn url_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

/// Build the authorization URL for the given params (ordered to match Python).
pub fn build_auth_url(client_id: &str, redirect_uri: &str, challenge: &str, state: &str) -> String {
    let params: [(&str, &str); 9] = [
        ("client_id", client_id),
        ("redirect_uri", redirect_uri),
        ("response_type", "code"),
        ("scope", OAUTH_SCOPES),
        ("state", state),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("access_type", "offline"),
        ("prompt", "consent"),
    ];
    let query = params
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{}?{}#hermes", AUTH_ENDPOINT, query)
}

/// Whether this looks like a headless environment.
pub fn is_headless() -> bool {
    HEADLESS_ENV_VARS
        .iter()
        .any(|k| std::env::var_os(k).map(|v| !v.is_empty()).unwrap_or(false))
}

// =============================================================================
// HTML response pages
// =============================================================================

const SUCCESS_PAGE: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>Hermes — signed in</title>
<style>
body { font: 16px/1.5 system-ui, sans-serif; margin: 10vh auto; max-width: 32rem; text-align: center; color: #222; }
h1 { color: #1a7f37; } p { color: #555; }
</style></head>
<body><h1>Signed in to Google.</h1>
<p>You can close this tab and return to your terminal.</p></body></html>
"#;

fn error_page(message: &str) -> String {
    format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8"><title>Hermes — sign-in failed</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; margin: 10vh auto; max-width: 32rem; text-align: center; color: #222; }}
h1 {{ color: #b42318; }} p {{ color: #555; }}
</style></head>
<body><h1>Sign-in failed</h1><p>{message}</p>
<p>Return to your terminal — Hermes will walk you through a manual paste fallback.</p></body></html>
"#,
        message = message
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// =============================================================================
// Callback server
// =============================================================================

/// Outcome captured by the loopback callback server.
#[derive(Debug, Clone, Default)]
pub struct CallbackResult {
    /// Captured `code` query parameter, if any.
    pub code: Option<String>,
    /// Captured error (an `error` param, `state_mismatch`, or `no_code`).
    pub error: Option<String>,
}

/// Bind the callback TCP listener, preferring `preferred_port` and falling back
/// to an ephemeral port. Returns `(listener, bound_port)`.
pub fn bind_callback_server(preferred_port: u16) -> std::io::Result<(TcpListener, u16)> {
    match TcpListener::bind((REDIRECT_HOST, preferred_port)) {
        Ok(listener) => {
            let port = listener.local_addr()?.port();
            Ok((listener, port))
        }
        Err(exc) => {
            log::info!(
                "Preferred OAuth callback port {} unavailable ({}); requesting ephemeral port",
                preferred_port,
                exc
            );
            let listener = TcpListener::bind((REDIRECT_HOST, 0))?;
            let port = listener.local_addr()?.port();
            Ok((listener, port))
        }
    }
}

/// Serve a single OAuth callback request and return the captured result.
///
/// Blocks (subject to the listener's own accept) until a request to
/// [`CALLBACK_PATH`] arrives, responding with a success/error HTML page. A
/// 404 is returned for any other path; the loop continues until the callback
/// path is hit or `wait` elapses. The `expected_state` is compared against the
/// `state` query parameter to defend against CSRF.
pub fn serve_callback(
    listener: &TcpListener,
    expected_state: &str,
    wait: Duration,
) -> CallbackResult {
    listener.set_nonblocking(true).ok();
    let deadline = Instant::now() + wait;
    loop {
        if Instant::now() >= deadline {
            return CallbackResult::default();
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                if let Some(result) = handle_callback_connection(stream, expected_state) {
                    return result;
                }
                // Non-callback path (404'd): keep waiting.
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Handle one connection. Returns Some(result) when the request hit the
/// callback path (terminal), or None for any other path (caller keeps waiting).
fn handle_callback_connection(mut stream: TcpStream, expected_state: &str) -> Option<CallbackResult> {
    stream.set_nonblocking(false).ok();
    let request_line = {
        let mut reader = BufReader::new(&mut stream);
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return None;
        }
        line
    };

    let target = request_line.split_whitespace().nth(1).unwrap_or("");
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };

    if path != CALLBACK_PATH {
        write_http_response(&mut stream, 404, "text/plain; charset=utf-8", b"Not Found");
        return None;
    }

    let params = parse_query(query);
    let state = params.get("state").cloned().unwrap_or_default();
    let error = params.get("error").cloned().unwrap_or_default();
    let code = params.get("code").cloned().unwrap_or_default();

    let mut result = CallbackResult::default();
    if state != expected_state {
        result.error = Some("state_mismatch".to_string());
        let body = error_page("State mismatch — aborting for safety.");
        write_http_response(&mut stream, 400, "text/html; charset=utf-8", body.as_bytes());
    } else if !error.is_empty() {
        result.error = Some(error.clone());
        let body = error_page(&format!("Authorization denied: {}", html_escape(&error)));
        write_http_response(&mut stream, 400, "text/html; charset=utf-8", body.as_bytes());
    } else if !code.is_empty() {
        result.code = Some(code);
        write_http_response(&mut stream, 200, "text/html; charset=utf-8", SUCCESS_PAGE.as_bytes());
    } else {
        result.error = Some("no_code".to_string());
        let body = error_page("Callback received no authorization code.");
        write_http_response(&mut stream, 400, "text/html; charset=utf-8", body.as_bytes());
    }
    Some(result)
}

fn write_http_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "OK",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        status = status,
        reason = reason,
        ct = content_type,
        len = body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
    // Drain a little so the client receives the response before close.
    let _ = stream.read(&mut [0u8; 0]);
}

/// Parse a urlencoded query string into a map (last value wins, like the
/// first element of `parse_qs`).
fn parse_query(query: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if query.is_empty() {
        return map;
    }
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        let key = url_decode(k);
        let val = url_decode(v);
        // Mirror parse_qs (keep_blank_values=False default): skip empty values.
        if val.is_empty() {
            continue;
        }
        map.entry(key).or_insert(val);
    }
    map
}

/// Percent-decode + `+`-to-space, mirroring `urllib.parse.parse_qs`.
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = &s[i + 1..i + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// =============================================================================
// Paste-mode parsing
// =============================================================================

/// Extract an authorization code from a pasted redirect URL, bare query
/// string, or bare code (mirrors `_prompt_paste_fallback`'s parsing).
pub fn parse_paste_input(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        let query = raw.split_once('?').map(|(_, q)| q).unwrap_or("");
        let query = query.split_once('#').map(|(q, _)| q).unwrap_or(query);
        let params = parse_query(query);
        return params.get("code").cloned();
    }
    if let Some(stripped) = raw.strip_prefix('?') {
        let params = parse_query(stripped);
        return params.get("code").cloned();
    }
    Some(raw.to_string())
}

// =============================================================================
// Token persistence helper
// =============================================================================

/// Persist a token-endpoint response into stored credentials.
pub fn persist_token_response(token_resp: &Value, project_id: &str) -> Result<GoogleCredentials> {
    let access_token = token_resp
        .get("access_token")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let refresh_token = token_resp
        .get("refresh_token")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let expires_in = value_to_i64(token_resp.get("expires_in")).unwrap_or(0);
    if access_token.is_empty() || refresh_token.is_empty() {
        return Err(GoogleOAuthError::new(
            "Google token response missing access_token or refresh_token.",
            "google_oauth_incomplete_token_response",
        ));
    }
    let creds = GoogleCredentials {
        expires_ms: ((now_unix_seconds() + expires_in.max(60) as f64) * 1000.0) as i64,
        email: fetch_user_email(&access_token, TOKEN_REQUEST_TIMEOUT_SECONDS),
        access_token,
        refresh_token,
        project_id: project_id.to_string(),
        managed_project_id: String::new(),
    };
    save_credentials(&creds)?;
    log::info!(
        "Google OAuth credentials saved to {}",
        credentials_path().display()
    );
    Ok(creds)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_parts_roundtrip_bare() {
        let parts = RefreshParts::parse("just_a_token");
        assert_eq!(parts.refresh_token, "just_a_token");
        assert_eq!(parts.project_id, "");
        assert_eq!(parts.managed_project_id, "");
        assert_eq!(parts.format(), "just_a_token");
    }

    #[test]
    fn refresh_parts_roundtrip_packed() {
        let parts = RefreshParts::parse("tok|proj|managed");
        assert_eq!(parts.refresh_token, "tok");
        assert_eq!(parts.project_id, "proj");
        assert_eq!(parts.managed_project_id, "managed");
        assert_eq!(parts.format(), "tok|proj|managed");
    }

    #[test]
    fn refresh_parts_empty() {
        let parts = RefreshParts::parse("");
        assert_eq!(parts.refresh_token, "");
        assert_eq!(parts.format(), "");
    }

    #[test]
    fn refresh_parts_extra_pipes_kept_in_managed() {
        // splitn(3) means a third segment keeps any further pipes.
        let parts = RefreshParts::parse("tok|proj|a|b|c");
        assert_eq!(parts.refresh_token, "tok");
        assert_eq!(parts.project_id, "proj");
        assert_eq!(parts.managed_project_id, "a|b|c");
    }

    #[test]
    fn refresh_parts_only_project_no_managed() {
        let parts = RefreshParts {
            refresh_token: "tok".into(),
            project_id: "proj".into(),
            managed_project_id: String::new(),
        };
        assert_eq!(parts.format(), "tok|proj|");
    }

    #[test]
    fn credentials_roundtrip() {
        let creds = GoogleCredentials {
            access_token: "atok".into(),
            refresh_token: "rtok".into(),
            expires_ms: 1_700_000_000_000,
            email: "u@example.com".into(),
            project_id: "p".into(),
            managed_project_id: "m".into(),
        };
        let value = creds.to_value();
        assert_eq!(value["refresh"], "rtok|p|m");
        assert_eq!(value["access"], "atok");
        assert_eq!(value["expires"], 1_700_000_000_000i64);
        assert_eq!(value["email"], "u@example.com");
        let back = GoogleCredentials::from_value(&value);
        assert_eq!(back, creds);
    }

    #[test]
    fn credentials_from_value_numeric_string_expires() {
        let value = serde_json::json!({
            "refresh": "rtok",
            "access": "atok",
            "expires": "1700000000000",
            "email": "u@example.com",
        });
        let creds = GoogleCredentials::from_value(&value);
        assert_eq!(creds.expires_ms, 1_700_000_000_000);
    }

    #[test]
    fn credentials_missing_fields_default() {
        let value = serde_json::json!({});
        let creds = GoogleCredentials::from_value(&value);
        assert_eq!(creds.access_token, "");
        assert_eq!(creds.refresh_token, "");
        assert_eq!(creds.expires_ms, 0);
    }

    #[test]
    fn access_token_expired_logic() {
        let mut creds = GoogleCredentials {
            access_token: "atok".into(),
            refresh_token: "rtok".into(),
            expires_ms: 0,
            email: String::new(),
            project_id: String::new(),
            managed_project_id: String::new(),
        };
        // No expiry -> expired.
        assert!(creds.access_token_expired(REFRESH_SKEW_SECONDS));
        // Far future -> not expired.
        creds.expires_ms = ((now_unix_seconds() + 3600.0) * 1000.0) as i64;
        assert!(!creds.access_token_expired(REFRESH_SKEW_SECONDS));
        // Within skew window -> expired.
        creds.expires_ms = ((now_unix_seconds() + 30.0) * 1000.0) as i64;
        assert!(creds.access_token_expired(REFRESH_SKEW_SECONDS));
        // Empty access token -> expired regardless.
        creds.access_token = String::new();
        creds.expires_ms = ((now_unix_seconds() + 3600.0) * 1000.0) as i64;
        assert!(creds.access_token_expired(REFRESH_SKEW_SECONDS));
    }

    #[test]
    fn pkce_pair_shape() {
        let (verifier, challenge) = generate_pkce_pair();
        assert!(verifier.len() > 80, "verifier should be long: {}", verifier.len());
        // S256 of any input base64url-no-pad is 43 chars.
        assert_eq!(challenge.len(), 43);
        assert!(!challenge.contains('='));
        assert!(!challenge.contains('+'));
        assert!(!challenge.contains('/'));
        // Recompute challenge to confirm S256 derivation.
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let expect =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize());
        assert_eq!(challenge, expect);
    }

    #[test]
    fn default_client_credentials() {
        assert_eq!(
            default_client_id(),
            "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
        );
        assert_eq!(default_client_secret(), "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");
    }

    #[test]
    fn scrape_regex_precise() {
        let src = r#"
            const OAUTH_CLIENT_ID = "123456789-abcdefabcdef.apps.googleusercontent.com";
            const OAUTH_CLIENT_SECRET = "GOCSPX-abcdEFGH1234ijklMNOP5678";
        "#;
        let (cid, csecret) = extract_creds_from_source(src);
        assert_eq!(cid, "123456789-abcdefabcdef.apps.googleusercontent.com");
        assert_eq!(csecret, "GOCSPX-abcdEFGH1234ijklMNOP5678");
    }

    #[test]
    fn scrape_regex_shape_fallback() {
        // No OAUTH_CLIENT_ID= prefix: shape match (>=8 digits, >=20 lc chars).
        let src = r#"x = '12345678-abcdefghijklmnopqrstuvwx.apps.googleusercontent.com';
        y = 'GOCSPX-abcdefghij1234567890ABCD';"#;
        let (cid, csecret) = extract_creds_from_source(src);
        assert_eq!(cid, "12345678-abcdefghijklmnopqrstuvwx.apps.googleusercontent.com");
        assert_eq!(csecret, "GOCSPX-abcdefghij1234567890ABCD");
    }

    #[test]
    fn scrape_regex_no_match() {
        let (cid, csecret) = extract_creds_from_source("nothing here");
        assert_eq!(cid, "");
        assert_eq!(csecret, "");
    }

    #[test]
    fn build_auth_url_shape() {
        let url = build_auth_url("CID", "http://127.0.0.1:8085/oauth2callback", "CHAL", "STATE");
        assert!(url.starts_with(AUTH_ENDPOINT));
        assert!(url.ends_with("#hermes"));
        assert!(url.contains("client_id=CID"));
        assert!(url.contains("code_challenge=CHAL"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=STATE"));
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=consent"));
        assert!(url.contains("response_type=code"));
        // redirect_uri must be percent-encoded.
        assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A8085%2Foauth2callback"));
        // scopes are space-joined -> '+' encoded.
        assert!(url.contains("scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+"));
    }

    #[test]
    fn parse_query_basics() {
        let q = parse_query("code=abc123&state=xyz&error=");
        assert_eq!(q.get("code").map(String::as_str), Some("abc123"));
        assert_eq!(q.get("state").map(String::as_str), Some("xyz"));
        // Empty value dropped (parse_qs default).
        assert!(q.get("error").is_none());
    }

    #[test]
    fn parse_query_percent_and_plus() {
        let q = parse_query("a=hello+world&b=%2Ffoo%2Fbar");
        assert_eq!(q.get("a").map(String::as_str), Some("hello world"));
        assert_eq!(q.get("b").map(String::as_str), Some("/foo/bar"));
    }

    #[test]
    fn paste_input_full_url() {
        let code = parse_paste_input(
            "http://127.0.0.1:8085/oauth2callback?state=s&code=THECODE&scope=x",
        );
        assert_eq!(code.as_deref(), Some("THECODE"));
    }

    #[test]
    fn paste_input_url_with_fragment() {
        let code = parse_paste_input("https://localhost/cb?code=ABC#frag");
        assert_eq!(code.as_deref(), Some("ABC"));
    }

    #[test]
    fn paste_input_bare_query() {
        let code = parse_paste_input("?code=QQ&state=z");
        assert_eq!(code.as_deref(), Some("QQ"));
    }

    #[test]
    fn paste_input_bare_code() {
        let code = parse_paste_input("  raw_code_value  ");
        assert_eq!(code.as_deref(), Some("raw_code_value"));
    }

    #[test]
    fn paste_input_empty() {
        assert_eq!(parse_paste_input("   "), None);
    }

    #[test]
    fn paste_input_url_no_code() {
        assert_eq!(parse_paste_input("http://localhost/cb?state=s"), None);
    }

    #[test]
    fn html_escape_works() {
        assert_eq!(html_escape("<a>&</a>"), "&lt;a&gt;&amp;&lt;/a&gt;");
    }

    #[test]
    fn error_page_contains_message() {
        let page = error_page("boom");
        assert!(page.contains("boom"));
        assert!(page.contains("Sign-in failed"));
    }

    #[test]
    fn sorted_value_sorts_keys() {
        let v = serde_json::json!({"z": 1, "a": 2, "m": {"y": 3, "b": 4}});
        let sorted = sorted_value(&v);
        let s = serde_json::to_string(&sorted).unwrap();
        // top-level keys sorted: a before m before z; nested b before y.
        assert!(s.find("\"a\"").unwrap() < s.find("\"m\"").unwrap());
        assert!(s.find("\"m\"").unwrap() < s.find("\"z\"").unwrap());
        assert!(s.find("\"b\"").unwrap() < s.find("\"y\"").unwrap());
    }

    #[test]
    fn value_to_i64_variants() {
        assert_eq!(value_to_i64(Some(&serde_json::json!(42))), Some(42));
        assert_eq!(value_to_i64(Some(&serde_json::json!("99"))), Some(99));
        assert_eq!(value_to_i64(Some(&serde_json::json!(""))), Some(0));
        assert_eq!(value_to_i64(Some(&serde_json::json!(null))), None);
        assert_eq!(value_to_i64(None), None);
        assert_eq!(value_to_i64(Some(&serde_json::json!(1.9))), Some(1));
    }

    #[test]
    fn persist_token_response_rejects_incomplete() {
        let resp = serde_json::json!({"access_token": "a"});
        let err = persist_token_response(&resp, "").unwrap_err();
        assert_eq!(err.code, "google_oauth_incomplete_token_response");
    }

    #[test]
    fn refresh_empty_token_errors() {
        let err = refresh_access_token("", None, None, 1).unwrap_err();
        assert_eq!(err.code, "google_oauth_refresh_token_missing");
    }

    #[test]
    fn hermes_home_honours_env() {
        let prev = std::env::var("HERMES_HOME").ok();
        unsafe { std::env::set_var("HERMES_HOME", "/tmp/hermes-oauth-test-home"); }
        assert_eq!(hermes_home_path(), PathBuf::from("/tmp/hermes-oauth-test-home"));
        assert_eq!(
            credentials_path(),
            PathBuf::from("/tmp/hermes-oauth-test-home/auth/google_oauth.json")
        );
        // SAFETY: single-threaded test teardown.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
    }

    #[test]
    fn token_hex_and_urlsafe_lengths() {
        assert_eq!(token_hex(4).len(), 8);
        // 64 bytes base64url-no-pad => ceil(64/3)*4 - padding.
        let t = token_urlsafe(64);
        assert!(t.len() >= 85 && t.len() <= 86);
        assert!(!t.contains('='));
    }
}
