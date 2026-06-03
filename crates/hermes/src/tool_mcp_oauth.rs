//! MCP OAuth 2.1 Client Support (native Rust port of `tools/mcp_oauth.py`).
//!
//! Implements the persistence, configuration, and local-callback-server glue
//! for the browser-based OAuth 2.1 authorization code flow (with PKCE) used by
//! MCP servers requiring OAuth instead of static bearer tokens.
//!
//! The original Python module delegated the actual discovery / dynamic client
//! registration / PKCE / token exchange to the MCP Python SDK's
//! `OAuthClientProvider` (an `httpx.Auth` subclass). That SDK object has no
//! Rust equivalent here, so this port reproduces the *self-contained* glue:
//!
//!   - [`HermesTokenStorage`]: persists tokens / client-info to disk so they
//!     survive process restarts, including the wall-clock `expires_at` Fix-A
//!     logic.
//!   - The ephemeral localhost HTTP callback server that captures the OAuth
//!     redirect with the authorization code ([`run_callback_server`]).
//!   - [`build_oauth_config`]: the entry point that resolves the callback
//!     port, builds client metadata, and pre-registers a configured client_id.
//!     It returns a fully-populated [`OAuthProviderConfig`] (the analogue of
//!     the Python `OAuthClientProvider` constructor arguments) rather than a
//!     live auth object.
//!
//! Configuration in config.yaml::
//!
//!     mcp_servers:
//!       my_server:
//!         url: "https://mcp.example.com/mcp"
//!         auth: oauth
//!         oauth:                                  # all fields optional
//!           client_id: "pre-registered-id"        # skip dynamic registration
//!           client_secret: "secret"               # confidential clients only
//!           scope: "read write"                   # default: server-provided
//!           redirect_port: 0                      # 0 = auto-pick free port
//!           client_name: "My Custom Client"       # default: "Hermes Agent"

use std::collections::HashMap;
use std::io::{Read, Write as IoWrite};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

/// Default callback / overall flow timeout in seconds.
pub const DEFAULT_TIMEOUT: f64 = 300.0;

/// Port used by the most recent [`build_oauth_config`] call.
///
/// Mirrors the Python module-level `_oauth_port`. Stored so that the callback
/// server and the redirect_uri share a port, and so tests can verify it.
/// `-1` means "unset".
static OAUTH_PORT: AtomicI64 = AtomicI64::new(-1);

/// Error raised when OAuth requires browser interaction in a non-interactive
/// environment, or the browser callback could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthError {
    /// OAuth requires browser interaction in a non-interactive env, or the
    /// callback timed out with no authorization code received.
    NonInteractive(String),
    /// The authorization server returned an explicit `error` query parameter.
    AuthorizationFailed(String),
    /// A generic runtime error (e.g. callback port not set).
    Runtime(String),
}

impl std::fmt::Display for OAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OAuthError::NonInteractive(m) => write!(f, "{m}"),
            OAuthError::AuthorizationFailed(m) => write!(f, "OAuth authorization failed: {m}"),
            OAuthError::Runtime(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for OAuthError {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Return the directory for MCP OAuth token files: `HERMES_HOME/mcp-tokens/`.
///
/// Uses `HERMES_HOME` so each profile gets its own OAuth tokens. Falls back to
/// `~/.hermes` when the env var is blank/unset (matching the Python fallback).
pub fn get_token_dir() -> PathBuf {
    let base = match std::env::var("HERMES_HOME") {
        Ok(v) if !v.trim().is_empty() => PathBuf::from(v),
        _ => match dirs::home_dir() {
            Some(h) => h.join(".hermes"),
            None => PathBuf::from(".hermes"),
        },
    };
    base.join("mcp-tokens")
}

/// Sanitize a server name for use as a filename (no path separators).
///
/// Replaces any char that is not a word char or `-` with `_`, strips leading
/// and trailing `_`, truncates to 128 chars, and falls back to `"default"`
/// when the result is empty.
pub fn safe_filename(name: &str) -> String {
    let replaced: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = replaced.trim_matches('_');
    let truncated: String = trimmed.chars().take(128).collect();
    if truncated.is_empty() {
        "default".to_string()
    } else {
        truncated
    }
}

/// Find an available TCP port on localhost.
pub fn find_free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Return true if we can reasonably expect to interact with a user
/// (stdin is a TTY).
pub fn is_interactive() -> bool {
    // SAFETY: isatty on a valid fd has no preconditions.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Return true if opening a browser is likely to work.
pub fn can_open_browser() -> bool {
    fn env_set(key: &str) -> bool {
        std::env::var_os(key).is_some_and(|v| !v.is_empty())
    }
    // Explicit SSH session -> no local display.
    if env_set("SSH_CLIENT") || env_set("SSH_TTY") {
        return false;
    }
    // Windows usually has a display.
    if cfg!(target_os = "windows") {
        return true;
    }
    // macOS usually has a display.
    if cfg!(target_os = "macos") {
        return true;
    }
    // Linux / other posix: need DISPLAY or WAYLAND_DISPLAY.
    env_set("DISPLAY") || env_set("WAYLAND_DISPLAY")
}

/// Read a JSON object file, returning `None` if it doesn't exist or is invalid.
pub fn read_json(path: &std::path::Path) -> Option<Map<String, Value>> {
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => Some(map),
            Ok(_) => {
                log::warn!("Failed to read {}: not a JSON object", path.display());
                None
            }
            Err(exc) => {
                log::warn!("Failed to read {}: {exc}", path.display());
                None
            }
        },
        Err(exc) => {
            log::warn!("Failed to read {}: {exc}", path.display());
            None
        }
    }
}

/// Write a map as pretty JSON with restricted permissions (0o600), atomically
/// via a `.tmp` sibling + rename.
pub fn write_json(path: &std::path::Path, data: &Map<String, Value>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let body = serde_json::to_string_pretty(&Value::Object(data.clone()))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let write_result = (|| -> std::io::Result<()> {
        std::fs::write(&tmp, body.as_bytes())?;
        set_file_mode_600(&tmp)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    write_result
}

#[cfg(unix)]
fn set_file_mode_600(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_file_mode_600(_path: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}

fn file_mtime_secs(path: &std::path::Path) -> Option<f64> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    mtime
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs_f64())
}

// ---------------------------------------------------------------------------
// HermesTokenStorage -- persistent token/client-info on disk
// ---------------------------------------------------------------------------

/// Persist OAuth tokens and client registration to JSON files.
///
/// File layout::
///
///     HERMES_HOME/mcp-tokens/<server_name>.json          -- tokens
///     HERMES_HOME/mcp-tokens/<server_name>.client.json   -- client info
pub struct HermesTokenStorage {
    server_name: String,
}

impl HermesTokenStorage {
    /// Construct storage for `server_name`. The name is sanitized into a safe
    /// filename component.
    pub fn new(server_name: &str) -> Self {
        Self {
            server_name: safe_filename(server_name),
        }
    }

    /// The sanitized server-name component used in filenames.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Path to the tokens file.
    pub fn tokens_path(&self) -> PathBuf {
        get_token_dir().join(format!("{}.json", self.server_name))
    }

    /// Path to the client-info file.
    pub fn client_info_path(&self) -> PathBuf {
        get_token_dir().join(format!("{}.client.json", self.server_name))
    }

    // -- tokens ------------------------------------------------------------

    /// Read tokens from disk, applying the Fix-A absolute-expiry rewrite.
    ///
    /// Hermes records an absolute wall-clock `expires_at` alongside the
    /// serialized token (see [`set_tokens`](Self::set_tokens)). On read we
    /// rewrite `expires_in` to the remaining seconds so a downstream consumer
    /// computes the correct absolute expiry and correctly reports expiry for
    /// tokens that lapsed while the process was down.
    ///
    /// Legacy token files (pre-Fix-A) have `expires_in` but no `expires_at`.
    /// We fall back to the file's mtime as a best-effort wall-clock proxy: if
    /// `(mtime + expires_in)` is in the past, clamp `expires_in` to zero so the
    /// token is refreshed before first use. The stored `expires_at` is
    /// stripped from the returned map (it is not part of the SDK token schema).
    pub fn get_tokens(&self) -> Option<Map<String, Value>> {
        let path = self.tokens_path();
        let mut data = read_json(&path)?;

        let absolute_expiry = data.remove("expires_at").as_ref().and_then(value_as_f64);
        if let Some(abs) = absolute_expiry {
            let remaining = (abs - now_secs()).max(0.0) as i64;
            data.insert("expires_in".to_string(), Value::from(remaining));
        } else if let Some(expires_in) = data.get("expires_in").and_then(value_as_f64) {
            if let Some(mtime) = file_mtime_secs(&path) {
                let implied_expiry = mtime + expires_in;
                let remaining = (implied_expiry - now_secs()).max(0.0) as i64;
                data.insert("expires_in".to_string(), Value::from(remaining));
            }
        }
        Some(data)
    }

    /// Persist tokens, adding an absolute `expires_at` derived from the
    /// current wall clock plus `expires_in`.
    ///
    /// `tokens` should be a JSON object with `None`-valued fields already
    /// omitted (matching the Python `exclude_none=True` model dump).
    pub fn set_tokens(&self, tokens: &Map<String, Value>) -> std::io::Result<()> {
        let mut payload = tokens.clone();
        if let Some(expires_in) = payload.get("expires_in").and_then(value_as_f64) {
            payload.insert("expires_at".to_string(), Value::from(now_secs() + expires_in));
        }
        write_json(&self.tokens_path(), &payload)?;
        log::debug!("OAuth tokens saved for {}", self.server_name);
        Ok(())
    }

    // -- client info -------------------------------------------------------

    /// Read client info from disk, or `None` when absent / corrupt.
    pub fn get_client_info(&self) -> Option<Map<String, Value>> {
        read_json(&self.client_info_path())
    }

    /// Persist client info.
    pub fn set_client_info(&self, client_info: &Map<String, Value>) -> std::io::Result<()> {
        write_json(&self.client_info_path(), client_info)?;
        log::debug!("OAuth client info saved for {}", self.server_name);
        Ok(())
    }

    // -- cleanup -----------------------------------------------------------

    /// Delete all stored OAuth state for this server.
    pub fn remove(&self) {
        for p in [self.tokens_path(), self.client_info_path()] {
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    /// Return true if we have tokens on disk (they may be expired).
    pub fn has_cached_tokens(&self) -> bool {
        self.tokens_path().exists()
    }
}

/// Coerce a JSON value into an f64 (accepting numbers and numeric strings).
fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Callback server -- ephemeral localhost HTTP server capturing the redirect
// ---------------------------------------------------------------------------

/// Result captured from the OAuth redirect callback.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CallbackResult {
    /// The authorization `code` query parameter, if present.
    pub auth_code: Option<String>,
    /// The `state` query parameter, if present.
    pub state: Option<String>,
    /// The `error` query parameter, if present.
    pub error: Option<String>,
}

/// Parse the query portion of a request path into a key -> first-value map.
///
/// Mirrors `urllib.parse.parse_qs` + `[0]` indexing: only the first value of
/// each key is kept, and percent-encoding / `+` are decoded.
pub fn parse_query(path: &str) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    let query = match path.split_once('?') {
        Some((_, q)) => q,
        None => return out,
    };
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        let key = url_decode(k);
        if key.is_empty() {
            continue;
        }
        // Keep only the first occurrence (parse_qs returns lists; Python reads [0]).
        out.entry(key).or_insert_with(|| url_decode(v));
    }
    out
}

/// Decode an application/x-www-form-urlencoded component (`+` -> space, `%XX`).
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
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Build the HTML body returned to the browser after the redirect.
pub fn callback_body(result: &CallbackResult) -> String {
    if result.auth_code.is_some() {
        "<html><body><h2>Authorization Successful</h2>\
         <p>You can close this tab and return to Hermes.</p></body></html>"
            .to_string()
    } else {
        let err = result.error.as_deref().unwrap_or("unknown");
        format!(
            "<html><body><h2>Authorization Failed</h2>\
             <p>Error: {err}</p></body></html>"
        )
    }
}

/// Handle a single incoming HTTP connection: parse the GET request line,
/// extract the OAuth parameters, write the HTML response, and return the
/// captured [`CallbackResult`].
fn handle_one_connection(mut stream: TcpStream) -> std::io::Result<CallbackResult> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    // Read up to the end of the request line (we only need the first line).
    let mut buf = [0u8; 8192];
    let mut data: Vec<u8> = Vec::new();
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        if data.windows(4).any(|w| w == b"\r\n\r\n") || data.contains(&b'\n') {
            break;
        }
        if data.len() > 65536 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&data);
    let request_line = text.lines().next().unwrap_or("");
    // "GET /callback?code=...&state=... HTTP/1.1"
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/");

    let params = parse_query(path);
    let result = CallbackResult {
        auth_code: params.get("code").cloned(),
        state: params.get("state").cloned(),
        error: params.get("error").cloned(),
    };

    let body = callback_body(&result);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.as_bytes().len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush().ok();

    log::debug!("OAuth callback: {request_line}");
    Ok(result)
}

/// Run a single-request callback server on `127.0.0.1:port`, blocking until a
/// redirect arrives or `timeout` elapses.
///
/// Returns the captured [`CallbackResult`]. On timeout the result has all
/// fields `None`. Returns an [`OAuthError::Runtime`] if the port cannot be
/// bound.
pub fn run_callback_server(port: u16, timeout: Duration) -> Result<CallbackResult, OAuthError> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| {
        OAuthError::Runtime(format!("could not bind callback port {port}: {e}"))
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| OAuthError::Runtime(e.to_string()))?;

    let deadline = SystemTime::now() + timeout;
    let poll_interval = Duration::from_millis(100);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).ok();
                return handle_one_connection(stream)
                    .map_err(|e| OAuthError::Runtime(e.to_string()));
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if SystemTime::now() >= deadline {
                    return Ok(CallbackResult::default());
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => return Err(OAuthError::Runtime(e.to_string())),
        }
    }
}

/// Wait for the OAuth callback to arrive on the local callback server.
///
/// Uses the module-level [`OAUTH_PORT`] (set by [`build_oauth_config`]). Blocks
/// for up to `timeout`. Returns `(auth_code, state)`.
///
/// # Errors
/// - [`OAuthError::Runtime`] if the port has not been set.
/// - [`OAuthError::AuthorizationFailed`] if the redirect carried an `error`.
/// - [`OAuthError::NonInteractive`] if it timed out with no code.
pub fn wait_for_callback(timeout: Duration) -> Result<(String, Option<String>), OAuthError> {
    let port = OAUTH_PORT.load(Ordering::SeqCst);
    if port < 0 {
        return Err(OAuthError::Runtime(
            "OAuth callback port not set — build_oauth_config must be called \
             before wait_for_callback"
                .to_string(),
        ));
    }
    let result = run_callback_server(port as u16, timeout)?;
    if let Some(err) = result.error.filter(|e| !e.is_empty()) {
        return Err(OAuthError::AuthorizationFailed(err));
    }
    match result.auth_code {
        Some(code) => Ok((code, result.state)),
        None => Err(OAuthError::NonInteractive(
            "OAuth callback timed out — no authorization code received. \
             Ensure you completed the browser authorization flow."
                .to_string(),
        )),
    }
}

/// Show the authorization URL to the user (stderr), opening the browser when
/// possible. Returns true if a browser was opened.
pub fn redirect_handler(authorization_url: &str) -> bool {
    eprintln!(
        "\n  MCP OAuth: authorization required.\n  \
         Open this URL in your browser:\n\n    {authorization_url}\n"
    );
    if can_open_browser() {
        match open_browser(authorization_url) {
            true => {
                eprintln!("  (Browser opened automatically.)\n");
                true
            }
            false => {
                eprintln!("  (Could not open browser — please open the URL manually.)\n");
                false
            }
        }
    } else {
        eprintln!("  (Headless environment detected — open the URL manually.)\n");
        false
    }
}

/// Best-effort cross-platform browser launch (analogue of `webbrowser.open`).
fn open_browser(url: &str) -> bool {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    result.is_ok()
}

// ---------------------------------------------------------------------------
// Client metadata + provider config
// ---------------------------------------------------------------------------

/// Parsed `oauth:` config block from config.yaml.
#[derive(Debug, Clone, Default)]
pub struct OAuthConfig {
    /// Pre-registered client id (skips dynamic registration).
    pub client_id: Option<String>,
    /// Client secret for confidential clients.
    pub client_secret: Option<String>,
    /// Requested scope.
    pub scope: Option<String>,
    /// Requested redirect port (0 = auto-pick).
    pub redirect_port: u16,
    /// Client name advertised during registration.
    pub client_name: Option<String>,
    /// Flow timeout in seconds.
    pub timeout: Option<f64>,
}

impl OAuthConfig {
    /// Parse from a serde_json object (the `oauth:` block). Unknown / missing
    /// fields use defaults, matching the Python `cfg.get(...)` access.
    pub fn from_map(map: &Map<String, Value>) -> Self {
        OAuthConfig {
            client_id: map.get("client_id").and_then(value_as_string),
            client_secret: map.get("client_secret").and_then(value_as_string),
            scope: map.get("scope").and_then(value_as_string),
            redirect_port: map
                .get("redirect_port")
                .and_then(value_as_f64)
                .map(|v| v as u16)
                .unwrap_or(0),
            client_name: map.get("client_name").and_then(value_as_string),
            timeout: map.get("timeout").and_then(value_as_f64),
        }
    }
}

fn value_as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::String(_) => None,
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// OAuth client metadata sent during dynamic client registration. Mirrors the
/// MCP SDK's `OAuthClientMetadata`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthClientMetadata {
    /// Advertised client name.
    pub client_name: String,
    /// Redirect URIs (always a single localhost callback here).
    pub redirect_uris: Vec<String>,
    /// Supported grant types.
    pub grant_types: Vec<String>,
    /// Supported response types.
    pub response_types: Vec<String>,
    /// Token endpoint auth method (`none` for public clients,
    /// `client_secret_post` when a secret is configured).
    pub token_endpoint_auth_method: String,
    /// Requested scope, if any.
    pub scope: Option<String>,
}

impl OAuthClientMetadata {
    /// Serialize to a JSON object with `None`-valued fields omitted.
    pub fn to_map(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("client_name".into(), Value::from(self.client_name.clone()));
        m.insert(
            "redirect_uris".into(),
            Value::from(self.redirect_uris.clone()),
        );
        m.insert("grant_types".into(), Value::from(self.grant_types.clone()));
        m.insert(
            "response_types".into(),
            Value::from(self.response_types.clone()),
        );
        m.insert(
            "token_endpoint_auth_method".into(),
            Value::from(self.token_endpoint_auth_method.clone()),
        );
        if let Some(scope) = &self.scope {
            m.insert("scope".into(), Value::from(scope.clone()));
        }
        m
    }
}

/// Fully-resolved provider configuration — the Rust analogue of the arguments
/// passed to the Python `OAuthClientProvider(...)` constructor.
#[derive(Debug, Clone)]
pub struct OAuthProviderConfig {
    /// MCP server endpoint URL.
    pub server_url: String,
    /// Sanitized server name (storage key).
    pub server_name: String,
    /// Resolved callback port.
    pub port: u16,
    /// Built client metadata.
    pub client_metadata: OAuthClientMetadata,
    /// Flow timeout in seconds.
    pub timeout: f64,
}

/// Pick or validate the OAuth callback port and stash it in [`OAUTH_PORT`].
///
/// Returns the resolved port. A `requested` of 0 auto-picks a free port.
pub fn configure_callback_port(requested: u16) -> std::io::Result<u16> {
    let port = if requested == 0 {
        find_free_port()?
    } else {
        requested
    };
    OAUTH_PORT.store(port as i64, Ordering::SeqCst);
    Ok(port)
}

/// Build [`OAuthClientMetadata`] from the oauth config and resolved port.
pub fn build_client_metadata(cfg: &OAuthConfig, port: u16) -> OAuthClientMetadata {
    let client_name = cfg
        .client_name
        .clone()
        .unwrap_or_else(|| "Hermes Agent".to_string());
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let token_endpoint_auth_method = if cfg.client_secret.is_some() {
        "client_secret_post".to_string()
    } else {
        "none".to_string()
    };

    OAuthClientMetadata {
        client_name,
        redirect_uris: vec![redirect_uri],
        grant_types: vec![
            "authorization_code".to_string(),
            "refresh_token".to_string(),
        ],
        response_types: vec!["code".to_string()],
        token_endpoint_auth_method,
        scope: cfg.scope.clone(),
    }
}

/// If `cfg` carries a pre-registered `client_id`, persist a client-info file to
/// `storage` so dynamic registration is skipped.
pub fn maybe_preregister_client(
    storage: &HermesTokenStorage,
    cfg: &OAuthConfig,
    client_metadata: &OAuthClientMetadata,
    port: u16,
) -> std::io::Result<()> {
    let client_id = match &cfg.client_id {
        Some(id) if !id.is_empty() => id.clone(),
        _ => return Ok(()),
    };
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let mut info = Map::new();
    info.insert("client_id".into(), Value::from(client_id.clone()));
    info.insert("redirect_uris".into(), Value::from(vec![redirect_uri]));
    info.insert(
        "grant_types".into(),
        Value::from(client_metadata.grant_types.clone()),
    );
    info.insert(
        "response_types".into(),
        Value::from(client_metadata.response_types.clone()),
    );
    info.insert(
        "token_endpoint_auth_method".into(),
        Value::from(client_metadata.token_endpoint_auth_method.clone()),
    );
    if let Some(secret) = &cfg.client_secret {
        info.insert("client_secret".into(), Value::from(secret.clone()));
    }
    if let Some(name) = &cfg.client_name {
        info.insert("client_name".into(), Value::from(name.clone()));
    }
    if let Some(scope) = &cfg.scope {
        info.insert("scope".into(), Value::from(scope.clone()));
    }

    write_json(&storage.client_info_path(), &info)?;
    log::debug!(
        "Pre-registered client_id={client_id} for '{}'",
        storage.server_name()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Delete stored OAuth tokens and client info for a server.
pub fn remove_oauth_tokens(server_name: &str) {
    let storage = HermesTokenStorage::new(server_name);
    storage.remove();
    log::info!("OAuth tokens removed for '{server_name}'");
}

/// Build the resolved OAuth provider configuration for an MCP server.
///
/// Public entry point (analogue of the Python `build_oauth_auth`). Resolves the
/// callback port (and stashes it in [`OAUTH_PORT`]), builds client metadata,
/// and pre-registers a configured client_id into on-disk storage.
///
/// Emits a warning when running non-interactively with no cached tokens (the
/// browser flow cannot complete unattended).
pub fn build_oauth_config(
    server_name: &str,
    server_url: &str,
    oauth_config: Option<&Map<String, Value>>,
) -> std::io::Result<OAuthProviderConfig> {
    let empty = Map::new();
    let cfg = OAuthConfig::from_map(oauth_config.unwrap_or(&empty));
    let storage = HermesTokenStorage::new(server_name);

    if !is_interactive() && !storage.has_cached_tokens() {
        log::warn!(
            "MCP OAuth for '{server_name}': non-interactive environment and no cached \
             tokens found. The OAuth flow requires browser authorization. Run \
             interactively first to complete the initial authorization, then cached \
             tokens will be reused."
        );
    }

    let port = configure_callback_port(cfg.redirect_port)?;
    let client_metadata = build_client_metadata(&cfg, port);
    maybe_preregister_client(&storage, &cfg, &client_metadata, port)?;

    Ok(OAuthProviderConfig {
        server_url: server_url.to_string(),
        server_name: storage.server_name().to_string(),
        port,
        client_metadata,
        timeout: cfg.timeout.unwrap_or(DEFAULT_TIMEOUT),
    })
}

/// Return the port used by the most recent [`build_oauth_config`] call, or
/// `None` if it has not been called.
pub fn current_oauth_port() -> Option<u16> {
    let p = OAUTH_PORT.load(Ordering::SeqCst);
    if p < 0 {
        None
    } else {
        Some(p as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn test_safe_filename_basic() {
        assert_eq!(safe_filename("my_server"), "my_server");
        assert_eq!(safe_filename("my-server"), "my-server");
    }

    #[test]
    fn test_safe_filename_sanitizes_separators() {
        assert_eq!(safe_filename("a/b\\c"), "a_b_c");
        assert_eq!(safe_filename("foo.bar"), "foo_bar");
    }

    #[test]
    fn test_safe_filename_strips_underscores() {
        assert_eq!(safe_filename("__name__"), "name");
        assert_eq!(safe_filename("/leading"), "leading");
    }

    #[test]
    fn test_safe_filename_empty_falls_back() {
        assert_eq!(safe_filename(""), "default");
        assert_eq!(safe_filename("///"), "default");
        assert_eq!(safe_filename("___"), "default");
    }

    #[test]
    fn test_safe_filename_truncates_to_128() {
        let long = "a".repeat(200);
        assert_eq!(safe_filename(&long).len(), 128);
    }

    #[test]
    fn test_find_free_port_nonzero() {
        let p = find_free_port().unwrap();
        assert!(p > 0);
    }

    #[test]
    fn test_parse_query_basic() {
        let q = parse_query("/callback?code=abc&state=xyz");
        assert_eq!(q.get("code").map(String::as_str), Some("abc"));
        assert_eq!(q.get("state").map(String::as_str), Some("xyz"));
    }

    #[test]
    fn test_parse_query_first_value_wins() {
        let q = parse_query("/cb?code=first&code=second");
        assert_eq!(q.get("code").map(String::as_str), Some("first"));
    }

    #[test]
    fn test_parse_query_url_decoding() {
        let q = parse_query("/cb?error=access%20denied&x=a+b");
        assert_eq!(q.get("error").map(String::as_str), Some("access denied"));
        assert_eq!(q.get("x").map(String::as_str), Some("a b"));
    }

    #[test]
    fn test_parse_query_no_query() {
        let q = parse_query("/callback");
        assert!(q.is_empty());
    }

    #[test]
    fn test_callback_body_success_vs_failure() {
        let ok = CallbackResult {
            auth_code: Some("c".into()),
            ..Default::default()
        };
        assert!(callback_body(&ok).contains("Authorization Successful"));

        let bad = CallbackResult {
            error: Some("denied".into()),
            ..Default::default()
        };
        let body = callback_body(&bad);
        assert!(body.contains("Authorization Failed"));
        assert!(body.contains("denied"));

        let unknown = CallbackResult::default();
        assert!(callback_body(&unknown).contains("unknown"));
    }

    #[test]
    fn test_build_client_metadata_public_client() {
        let cfg = OAuthConfig::default();
        let md = build_client_metadata(&cfg, 12345);
        assert_eq!(md.client_name, "Hermes Agent");
        assert_eq!(md.token_endpoint_auth_method, "none");
        assert_eq!(md.redirect_uris, vec!["http://127.0.0.1:12345/callback"]);
        assert_eq!(
            md.grant_types,
            vec!["authorization_code".to_string(), "refresh_token".to_string()]
        );
        assert_eq!(md.response_types, vec!["code".to_string()]);
        assert!(md.scope.is_none());
    }

    #[test]
    fn test_build_client_metadata_confidential_client() {
        let cfg = OAuthConfig {
            client_secret: Some("s3cr3t".into()),
            scope: Some("read write".into()),
            client_name: Some("Custom".into()),
            ..Default::default()
        };
        let md = build_client_metadata(&cfg, 1);
        assert_eq!(md.token_endpoint_auth_method, "client_secret_post");
        assert_eq!(md.scope.as_deref(), Some("read write"));
        assert_eq!(md.client_name, "Custom");
    }

    #[test]
    fn test_metadata_to_map_omits_none_scope() {
        let cfg = OAuthConfig::default();
        let md = build_client_metadata(&cfg, 9);
        let map = md.to_map();
        assert!(!map.contains_key("scope"));
        assert_eq!(map.get("token_endpoint_auth_method").unwrap(), "none");
    }

    #[test]
    fn test_oauth_config_from_map() {
        let map = obj(&[
            ("client_id", Value::from("cid")),
            ("redirect_port", Value::from(8080)),
            ("scope", Value::from("a b")),
            ("timeout", Value::from(120.0)),
        ]);
        let cfg = OAuthConfig::from_map(&map);
        assert_eq!(cfg.client_id.as_deref(), Some("cid"));
        assert_eq!(cfg.redirect_port, 8080);
        assert_eq!(cfg.scope.as_deref(), Some("a b"));
        assert_eq!(cfg.timeout, Some(120.0));
        assert!(cfg.client_secret.is_none());
    }

    #[test]
    fn test_configure_callback_port_explicit_and_global() {
        let port = configure_callback_port(54321).unwrap();
        assert_eq!(port, 54321);
        assert_eq!(current_oauth_port(), Some(54321));
    }

    #[test]
    fn test_configure_callback_port_auto() {
        let port = configure_callback_port(0).unwrap();
        assert!(port > 0);
        assert_eq!(current_oauth_port(), Some(port));
    }

    #[test]
    fn test_token_storage_roundtrip_with_expires_at() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes-oauth-test-{}",
            std::process::id()
        ));
        unsafe { std::env::set_var("HERMES_HOME", &tmp); }

        let storage = HermesTokenStorage::new("test/server!");
        // sanitized name
        assert_eq!(storage.server_name(), "test_server");

        let tokens = obj(&[
            ("access_token", Value::from("tok")),
            ("token_type", Value::from("Bearer")),
            ("expires_in", Value::from(3600)),
        ]);
        storage.set_tokens(&tokens).unwrap();
        assert!(storage.has_cached_tokens());

        // On disk, expires_at must be present.
        let raw = read_json(&storage.tokens_path()).unwrap();
        assert!(raw.contains_key("expires_at"));

        // On read, expires_at is stripped and expires_in is recomputed (~3600).
        let loaded = storage.get_tokens().unwrap();
        assert!(!loaded.contains_key("expires_at"));
        let remaining = loaded.get("expires_in").and_then(value_as_f64).unwrap();
        assert!((0.0..=3600.0).contains(&remaining));
        assert!(remaining > 3000.0);

        storage.remove();
        assert!(!storage.has_cached_tokens());
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_get_tokens_clamps_expired_absolute() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes-oauth-expired-{}",
            std::process::id()
        ));
        unsafe { std::env::set_var("HERMES_HOME", &tmp); }

        let storage = HermesTokenStorage::new("expsrv");
        // Write a token whose expires_at is already in the past.
        let mut payload = obj(&[
            ("access_token", Value::from("tok")),
            ("expires_in", Value::from(3600)),
        ]);
        payload.insert("expires_at".into(), Value::from(now_secs() - 100.0));
        write_json(&storage.tokens_path(), &payload).unwrap();

        let loaded = storage.get_tokens().unwrap();
        assert_eq!(
            loaded.get("expires_in").and_then(value_as_f64),
            Some(0.0)
        );

        storage.remove();
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_maybe_preregister_writes_client_info() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes-oauth-prereg-{}",
            std::process::id()
        ));
        unsafe { std::env::set_var("HERMES_HOME", &tmp); }

        let storage = HermesTokenStorage::new("pre");
        let cfg = OAuthConfig {
            client_id: Some("my-id".into()),
            client_secret: Some("sec".into()),
            scope: Some("read".into()),
            client_name: Some("App".into()),
            ..Default::default()
        };
        let md = build_client_metadata(&cfg, 7000);
        maybe_preregister_client(&storage, &cfg, &md, 7000).unwrap();

        let info = storage.get_client_info().unwrap();
        assert_eq!(info.get("client_id").unwrap(), "my-id");
        assert_eq!(info.get("client_secret").unwrap(), "sec");
        assert_eq!(info.get("token_endpoint_auth_method").unwrap(), "client_secret_post");
        assert!(info.get("redirect_uris").unwrap().is_array());

        storage.remove();
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_maybe_preregister_noop_without_client_id() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes-oauth-noprereg-{}",
            std::process::id()
        ));
        unsafe { std::env::set_var("HERMES_HOME", &tmp); }

        let storage = HermesTokenStorage::new("nopre");
        let cfg = OAuthConfig::default();
        let md = build_client_metadata(&cfg, 7001);
        maybe_preregister_client(&storage, &cfg, &md, 7001).unwrap();
        assert!(storage.get_client_info().is_none());

        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_wait_for_callback_no_port_errors() {
        OAUTH_PORT.store(-1, Ordering::SeqCst);
        let err = wait_for_callback(Duration::from_millis(10)).unwrap_err();
        match err {
            OAuthError::Runtime(m) => assert!(m.contains("callback port not set")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_run_callback_server_timeout_returns_empty() {
        let port = find_free_port().unwrap();
        let result = run_callback_server(port, Duration::from_millis(150)).unwrap();
        assert_eq!(result, CallbackResult::default());
    }

    #[test]
    fn test_run_callback_server_captures_code() {
        let port = find_free_port().unwrap();
        let handle = std::thread::spawn(move || {
            run_callback_server(port, Duration::from_secs(5))
        });
        // Give the server a moment to bind.
        std::thread::sleep(Duration::from_millis(200));
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .write_all(b"GET /callback?code=THECODE&state=ST HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        // Read the response so the server can finish.
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf);
        let result = handle.join().unwrap().unwrap();
        assert_eq!(result.auth_code.as_deref(), Some("THECODE"));
        assert_eq!(result.state.as_deref(), Some("ST"));
        assert!(String::from_utf8_lossy(&buf).contains("Authorization Successful"));
    }
}
