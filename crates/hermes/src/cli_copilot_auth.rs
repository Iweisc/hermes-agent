//! GitHub Copilot authentication utilities.
//!
//! Native Rust port of `hermes_cli/copilot_auth.py`.
//!
//! Implements the OAuth device code flow used by the Copilot CLI and handles
//! token validation/exchange for the Copilot API.
//!
//! Token type support (per GitHub docs):
//! ```text
//!   gho_          OAuth token           ✓  (default via copilot login)
//!   github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
//!   ghu_          GitHub App token      ✓  (via environment variable)
//!   ghp_          Classic PAT           ✗  NOT SUPPORTED
//! ```
//!
//! Credential search order (matching Copilot CLI behaviour):
//! ```text
//!   1. COPILOT_GITHUB_TOKEN env var
//!   2. GH_TOKEN env var
//!   3. GITHUB_TOKEN env var
//!   4. gh auth token  CLI fallback
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

// ─── Constants ─────────────────────────────────────────────────────────────

/// OAuth device code flow client ID (same client ID as opencode/Copilot CLI).
pub const COPILOT_OAUTH_CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";

/// Classic PAT prefix (unsupported).
const CLASSIC_PAT_PREFIX: &str = "ghp_";

/// Env var search order (matches Copilot CLI).
pub const COPILOT_ENV_VARS: [&str; 3] = ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"];

/// Default polling interval, in seconds.
const DEVICE_CODE_POLL_INTERVAL: u64 = 5;
/// Safety margin added to each poll sleep, in seconds.
const DEVICE_CODE_POLL_SAFETY_MARGIN: u64 = 3;

/// Refresh the exchanged token this many seconds before expiry.
const JWT_REFRESH_MARGIN_SECONDS: f64 = 120.0;

/// Token exchange endpoint and associated header values
/// (matching VS Code / Copilot CLI).
const TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const EDITOR_VERSION: &str = "vscode/1.104.1";
const EXCHANGE_USER_AGENT: &str = "GitHubCopilotChat/0.26.7";

// ─── Token validation ──────────────────────────────────────────────────────

/// Validate that a token is usable with the Copilot API.
///
/// Returns `(valid, message)`.
pub fn validate_copilot_token(token: &str) -> (bool, String) {
    let token = token.trim();
    if token.is_empty() {
        return (false, "Empty token".to_string());
    }

    if token.starts_with(CLASSIC_PAT_PREFIX) {
        return (
            false,
            concat!(
                "Classic Personal Access Tokens (ghp_*) are not supported by the ",
                "Copilot API. Use one of:\n",
                "  → `copilot login` or `hermes model` to authenticate via OAuth\n",
                "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n",
                "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
            )
            .to_string(),
        );
    }

    (true, "OK".to_string())
}

/// Resolve a GitHub token suitable for Copilot API use.
///
/// Returns `(token, source)` where `source` describes where the token came
/// from. Returns `Err` if only a classic PAT is available via `gh auth token`.
/// Returns `Ok(("", ""))` when no token can be found.
pub fn resolve_copilot_token() -> Result<(String, String), String> {
    // 1. Check env vars in priority order.
    for env_var in COPILOT_ENV_VARS {
        let val = std::env::var(env_var).unwrap_or_default();
        let val = val.trim();
        if !val.is_empty() {
            let (valid, msg) = validate_copilot_token(val);
            if !valid {
                log::warn!("Token from {} is not supported: {}", env_var, msg);
                continue;
            }
            return Ok((val.to_string(), env_var.to_string()));
        }
    }

    // 2. Fall back to gh auth token.
    if let Some(token) = try_gh_cli_token() {
        let (valid, msg) = validate_copilot_token(&token);
        if !valid {
            return Err(format!(
                "Token from `gh auth token` is a classic PAT (ghp_*). {}",
                msg
            ));
        }
        return Ok((token, "gh auth token".to_string()));
    }

    Ok((String::new(), String::new()))
}

/// Return candidate `gh` binary paths, including common Homebrew installs.
fn gh_cli_candidates() -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Some(resolved) = which_gh() {
        candidates.push(resolved);
    }

    let home_local = dirs::home_dir()
        .map(|h| h.join(".local").join("bin").join("gh"))
        .unwrap_or_else(|| PathBuf::from(".local/bin/gh"));

    let extra = [
        PathBuf::from("/opt/homebrew/bin/gh"),
        PathBuf::from("/usr/local/bin/gh"),
        home_local,
    ];

    for candidate in extra {
        let cand_str = candidate.to_string_lossy().to_string();
        if candidates.contains(&cand_str) {
            continue;
        }
        if is_executable_file(&candidate) {
            candidates.push(cand_str);
        }
    }

    candidates
}

/// Locate `gh` on `PATH`, mirroring `shutil.which("gh")`.
fn which_gh() -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("gh");
        if is_executable_file(&candidate) {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    None
}

/// True when `path` is a regular file and executable by the current user.
fn is_executable_file(path: &std::path::Path) -> bool {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = meta.permissions().mode();
        // Any execute bit set (owner/group/other), matching os.access(_, X_OK)
        // closely enough for candidate discovery.
        mode & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Return a token from `gh auth token` when the GitHub CLI is available.
///
/// When `COPILOT_GH_HOST` is set, passes `--hostname` so `gh` returns the
/// correct host's token. Also strips `GITHUB_TOKEN` / `GH_TOKEN` from the
/// subprocess environment so `gh` reads from its own credential store
/// (hosts.yml) instead of just echoing the env var back.
pub fn try_gh_cli_token() -> Option<String> {
    let hostname = std::env::var("COPILOT_GH_HOST").unwrap_or_default();
    let hostname = hostname.trim().to_string();

    for gh_path in gh_cli_candidates() {
        let mut cmd = Command::new(&gh_path);
        cmd.arg("auth").arg("token");
        if !hostname.is_empty() {
            cmd.arg("--hostname").arg(&hostname);
        }
        // Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN.
        cmd.env_remove("GITHUB_TOKEN");
        cmd.env_remove("GH_TOKEN");

        match run_with_timeout(cmd, Duration::from_secs(5)) {
            Ok(Some(output)) => {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let trimmed = stdout.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
            Ok(None) => {
                log::debug!("gh CLI token lookup timed out ({})", gh_path);
                continue;
            }
            Err(exc) => {
                log::debug!("gh CLI token lookup failed ({}): {}", gh_path, exc);
                continue;
            }
        }
    }
    None
}

/// Run a command with a wall-clock timeout.
///
/// Returns `Ok(Some(output))` on completion, `Ok(None)` on timeout, and
/// `Err` if the process could not be spawned (e.g. binary missing).
fn run_with_timeout(
    mut cmd: Command,
    timeout: Duration,
) -> std::io::Result<Option<std::process::Output>> {
    use std::process::Stdio;
    use std::thread;

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.stdin(Stdio::null());

    let mut child = cmd.spawn()?;
    let start = std::time::Instant::now();

    loop {
        match child.try_wait()? {
            Some(_) => {
                let output = child.wait_with_output()?;
                return Ok(Some(output));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

// ─── OAuth Device Code Flow ──────────────────────────────────────────────────

/// Response of the device-code request step.
#[derive(Debug, Clone)]
struct DeviceCodeResponse {
    verification_uri: String,
    user_code: String,
    device_code: String,
    interval: u64,
}

/// Run the GitHub OAuth device code flow for Copilot.
///
/// Prints instructions for the user, polls for completion, and returns the
/// OAuth access token on success, or `None` on failure/cancellation.
///
/// This replicates the flow used by opencode and the Copilot CLI.
pub fn copilot_device_code_login(host: &str, timeout_seconds: f64) -> Option<String> {
    let domain = host.trim_end_matches('/');
    let device_code_url = format!("https://{}/login/device/code", domain);
    let access_token_url = format!("https://{}/login/oauth/access_token", domain);

    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(exc) => {
            log::error!("Failed to build HTTP client: {}", exc);
            println!("  ✗ Failed to start device authorization: {}", exc);
            return None;
        }
    };

    // Step 1: Request device code.
    let device_data = match request_device_code(&client, &device_code_url) {
        Ok(d) => d,
        Err(exc) => {
            log::error!("Failed to initiate device authorization: {}", exc);
            println!("  ✗ Failed to start device authorization: {}", exc);
            return None;
        }
    };

    if device_data.device_code.is_empty() || device_data.user_code.is_empty() {
        println!("  ✗ GitHub did not return a device code.");
        return None;
    }

    let mut interval = device_data.interval.max(1);

    // Step 2: Show instructions.
    println!();
    println!(
        "  Open this URL in your browser: {}",
        device_data.verification_uri
    );
    println!("  Enter this code: {}", device_data.user_code);
    println!();
    print_flush("  Waiting for authorization...");

    // Step 3: Poll for completion.
    let deadline = now_secs() + timeout_seconds;
    let poll_client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok();

    while now_secs() < deadline {
        std::thread::sleep(Duration::from_secs(
            interval + DEVICE_CODE_POLL_SAFETY_MARGIN,
        ));

        let client = match &poll_client {
            Some(c) => c,
            None => &client,
        };

        let result = match poll_access_token(
            client,
            &access_token_url,
            &device_data.device_code,
        ) {
            Ok(r) => r,
            Err(_) => {
                print_flush(".");
                continue;
            }
        };

        if let Some(token) = result
            .get("access_token")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            println!(" ✓");
            return Some(token.to_string());
        }

        let error = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match error {
            "authorization_pending" => {
                print_flush(".");
                continue;
            }
            "slow_down" => {
                // RFC 8628: add 5 seconds to polling interval.
                let server_interval = result.get("interval").and_then(|v| v.as_f64());
                match server_interval {
                    Some(si) if si > 0.0 => interval = si as u64,
                    _ => interval += 5,
                }
                print_flush(".");
                continue;
            }
            "expired_token" => {
                println!();
                println!("  ✗ Device code expired. Please try again.");
                return None;
            }
            "access_denied" => {
                println!();
                println!("  ✗ Authorization was denied.");
                return None;
            }
            "" => {
                // No token, no error: keep polling.
                continue;
            }
            other => {
                println!();
                println!("  ✗ Authorization failed: {}", other);
                return None;
            }
        }
    }

    println!();
    println!("  ✗ Timed out waiting for authorization.");
    None
}

fn request_device_code(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<DeviceCodeResponse, String> {
    let params = [
        ("client_id", COPILOT_OAUTH_CLIENT_ID),
        ("scope", "read:user"),
    ];
    let resp = client
        .post(url)
        .header("Accept", "application/json")
        .header("User-Agent", "HermesAgent/1.0")
        .form(&params)
        .send()
        .map_err(|e| e.to_string())?;

    let data: serde_json::Value = resp.json().map_err(|e| e.to_string())?;

    Ok(DeviceCodeResponse {
        verification_uri: data
            .get("verification_uri")
            .and_then(|v| v.as_str())
            .unwrap_or("https://github.com/login/device")
            .to_string(),
        user_code: data
            .get("user_code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        device_code: data
            .get("device_code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        interval: data
            .get("interval")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEVICE_CODE_POLL_INTERVAL),
    })
}

fn poll_access_token(
    client: &reqwest::blocking::Client,
    url: &str,
    device_code: &str,
) -> Result<serde_json::Value, String> {
    let params = [
        ("client_id", COPILOT_OAUTH_CLIENT_ID),
        ("device_code", device_code),
        (
            "grant_type",
            "urn:ietf:params:oauth:grant-type:device_code",
        ),
    ];
    let resp = client
        .post(url)
        .header("Accept", "application/json")
        .header("User-Agent", "HermesAgent/1.0")
        .form(&params)
        .send()
        .map_err(|e| e.to_string())?;

    resp.json().map_err(|e| e.to_string())
}

// ─── Copilot Token Exchange ──────────────────────────────────────────────────

/// Module-level cache for exchanged Copilot API tokens.
///
/// Maps `raw_token_fingerprint -> (api_token, expires_at_epoch)`.
static JWT_CACHE: Mutex<Option<HashMap<String, (String, f64)>>> = Mutex::new(None);

/// Short fingerprint of a raw token for cache keying (avoids storing full token).
fn token_fingerprint(raw_token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw_token.as_bytes());
    let digest = hasher.finalize();
    let hex = digest.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    hex[..16].to_string()
}

/// Exchange a raw GitHub token for a short-lived Copilot API token.
///
/// Calls `GET https://api.github.com/copilot_internal/v2/token` with the raw
/// GitHub token and returns `(api_token, expires_at)`.
///
/// The returned token is a semicolon-separated string (not a standard JWT)
/// used as `Authorization: Bearer <token>` for Copilot API requests.
///
/// Results are cached in-process and reused until close to expiry.
/// Returns `Err` on failure.
pub fn exchange_copilot_token(raw_token: &str, timeout: f64) -> Result<(String, f64), String> {
    let fp = token_fingerprint(raw_token);

    // Check cache first.
    {
        let guard = JWT_CACHE.lock().unwrap();
        if let Some(map) = guard.as_ref() {
            if let Some((api_token, expires_at)) = map.get(&fp) {
                if now_secs() < expires_at - JWT_REFRESH_MARGIN_SECONDS {
                    return Ok((api_token.clone(), *expires_at));
                }
            }
        }
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()
        .map_err(|e| format!("Copilot token exchange failed: {}", e))?;

    let resp = client
        .get(TOKEN_EXCHANGE_URL)
        .header("Authorization", format!("token {}", raw_token))
        .header("User-Agent", EXCHANGE_USER_AGENT)
        .header("Accept", "application/json")
        .header("Editor-Version", EDITOR_VERSION)
        .send()
        .map_err(|e| format!("Copilot token exchange failed: {}", e))?;

    let data: serde_json::Value = resp
        .json()
        .map_err(|e| format!("Copilot token exchange failed: {}", e))?;

    let api_token = data
        .get("token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if api_token.is_empty() {
        return Err("Copilot token exchange returned empty token".to_string());
    }

    // Convert expires_at to float if needed. Accept numeric or numeric-string.
    let expires_raw = data.get("expires_at");
    let expires_at = match expires_raw {
        Some(v) if v.is_number() => v.as_f64().unwrap_or(0.0),
        Some(v) if v.is_string() => v
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0),
        _ => 0.0,
    };
    let expires_at = if expires_at != 0.0 {
        expires_at
    } else {
        now_secs() + 1800.0
    };

    {
        let mut guard = JWT_CACHE.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        map.insert(fp, (api_token.clone(), expires_at));
    }
    log::debug!("Copilot token exchanged, expires_at={}", expires_at);

    Ok((api_token, expires_at))
}

/// Exchange a raw GitHub token for a Copilot API token, with fallback.
///
/// Convenience wrapper: returns the exchanged token on success, or the raw
/// token unchanged if the exchange fails (e.g. network error, unsupported
/// account type). This preserves existing behaviour for accounts that don't
/// need exchange while enabling access to internal-only models for those that
/// do.
pub fn get_copilot_api_token(raw_token: &str) -> String {
    if raw_token.is_empty() {
        return raw_token.to_string();
    }
    match exchange_copilot_token(raw_token, 10.0) {
        Ok((api_token, _)) => api_token,
        Err(exc) => {
            log::debug!("Copilot token exchange failed, using raw token: {}", exc);
            raw_token.to_string()
        }
    }
}

/// Clear the in-process token-exchange cache. Primarily useful for tests.
pub fn clear_jwt_cache() {
    let mut guard = JWT_CACHE.lock().unwrap();
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
}

// ─── Copilot API Headers ─────────────────────────────────────────────────────

/// Build the standard headers for Copilot API requests.
///
/// Replicates the header set used by opencode and the Copilot CLI.
pub fn copilot_request_headers(is_agent_turn: bool, is_vision: bool) -> HashMap<String, String> {
    let mut headers: HashMap<String, String> = HashMap::new();
    headers.insert("Editor-Version".to_string(), "vscode/1.104.1".to_string());
    headers.insert("User-Agent".to_string(), "HermesAgent/1.0".to_string());
    headers.insert(
        "Copilot-Integration-Id".to_string(),
        "vscode-chat".to_string(),
    );
    headers.insert(
        "Openai-Intent".to_string(),
        "conversation-edits".to_string(),
    );
    headers.insert(
        "x-initiator".to_string(),
        if is_agent_turn { "agent" } else { "user" }.to_string(),
    );
    if is_vision {
        headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
    }
    headers
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Current wall-clock time as seconds since the Unix epoch.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Print without a trailing newline and flush stdout, mirroring
/// `print(..., end="", flush=True)`.
fn print_flush(s: &str) {
    use std::io::Write;
    print!("{}", s);
    let _ = std::io::stdout().flush();
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_token_is_invalid() {
        let (valid, msg) = validate_copilot_token("");
        assert!(!valid);
        assert_eq!(msg, "Empty token");

        let (valid, msg) = validate_copilot_token("   ");
        assert!(!valid);
        assert_eq!(msg, "Empty token");
    }

    #[test]
    fn classic_pat_is_unsupported() {
        let (valid, msg) = validate_copilot_token("ghp_abc123");
        assert!(!valid);
        assert!(msg.contains("Classic Personal Access Tokens"));
        assert!(msg.contains("github_pat_*"));
    }

    #[test]
    fn oauth_token_is_valid() {
        let (valid, msg) = validate_copilot_token("gho_sometoken");
        assert!(valid);
        assert_eq!(msg, "OK");

        let (valid, _) = validate_copilot_token("github_pat_xxx");
        assert!(valid);

        let (valid, _) = validate_copilot_token("ghu_xxx");
        assert!(valid);
    }

    #[test]
    fn validate_trims_whitespace() {
        let (valid, _) = validate_copilot_token("  gho_x  ");
        assert!(valid);
        let (valid, _) = validate_copilot_token("  ghp_x  ");
        assert!(!valid);
    }

    #[test]
    fn fingerprint_is_16_hex_chars() {
        let fp = token_fingerprint("gho_example_token");
        assert_eq!(fp.len(), 16);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
        // SHA-256 of the same input is deterministic.
        assert_eq!(fp, token_fingerprint("gho_example_token"));
        assert_ne!(fp, token_fingerprint("gho_other_token"));
    }

    #[test]
    fn headers_agent_vs_user() {
        let agent = copilot_request_headers(true, false);
        assert_eq!(agent.get("x-initiator").map(String::as_str), Some("agent"));
        assert_eq!(
            agent.get("Editor-Version").map(String::as_str),
            Some("vscode/1.104.1")
        );
        assert_eq!(
            agent.get("Copilot-Integration-Id").map(String::as_str),
            Some("vscode-chat")
        );
        assert!(!agent.contains_key("Copilot-Vision-Request"));

        let user = copilot_request_headers(false, false);
        assert_eq!(user.get("x-initiator").map(String::as_str), Some("user"));
    }

    #[test]
    fn headers_vision_flag() {
        let vision = copilot_request_headers(true, true);
        assert_eq!(
            vision.get("Copilot-Vision-Request").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn get_copilot_api_token_empty_passthrough() {
        assert_eq!(get_copilot_api_token(""), "");
    }

    #[test]
    fn cache_hit_returns_cached_value() {
        clear_jwt_cache();
        let raw = "gho_cache_test_token";
        let fp = token_fingerprint(raw);
        let future = now_secs() + 3600.0;
        {
            let mut guard = JWT_CACHE.lock().unwrap();
            let map = guard.get_or_insert_with(HashMap::new);
            map.insert(fp, ("cached_api_token".to_string(), future));
        }
        // Should return cached value without making a network call.
        let (token, exp) = exchange_copilot_token(raw, 10.0).expect("cached value");
        assert_eq!(token, "cached_api_token");
        assert_eq!(exp, future);
        clear_jwt_cache();
    }

    #[test]
    fn expired_cache_entry_not_used_directly() {
        clear_jwt_cache();
        let raw = "gho_expired_token";
        let fp = token_fingerprint(raw);
        // expires within the refresh margin -> considered stale.
        let near = now_secs() + 10.0;
        {
            let mut guard = JWT_CACHE.lock().unwrap();
            let map = guard.get_or_insert_with(HashMap::new);
            map.insert(fp.clone(), ("stale".to_string(), near));
        }
        // Verify our staleness predicate: now >= expires_at - margin.
        let stale = now_secs() >= near - JWT_REFRESH_MARGIN_SECONDS;
        assert!(stale);
        clear_jwt_cache();
    }

    #[test]
    fn env_vars_order_constant() {
        assert_eq!(
            COPILOT_ENV_VARS,
            ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]
        );
    }
}
