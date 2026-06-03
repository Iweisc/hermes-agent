//! Generic managed-tool gateway helpers for Nous-hosted vendor passthroughs.
//!
//! Native Rust port of `tools/managed_tool_gateway.py`.
//!
//! Behavioural notes vs. the Python original:
//! * `get_hermes_home()` is reused from [`crate::mod_hermes_constants`], so the
//!   `HERMES_HOME` override is honoured identically.
//! * `managed_nous_tools_enabled()` is *not* yet ported to native Rust (it lives
//!   in `tools/tool_backend_helpers.py` and depends on the Nous auth/model
//!   subsystem). It is exposed here as a pluggable hook on
//!   [`resolve_managed_tool_gateway`] / [`is_managed_tool_gateway_ready`]. The
//!   default implementation mirrors the Python "never block startup" contract:
//!   it returns `false` unless a caller-supplied `managed_enabled` closure says
//!   otherwise. Once `tool_backend_helpers` lands in Rust, the default can call
//!   `crate::tool_backend_helpers::managed_nous_tools_enabled` instead.
//! * The Nous OAuth refresh path (`hermes_cli.auth.resolve_nous_access_token`)
//!   is likewise not ported; the cached token from `auth.json` is honoured, and
//!   a pluggable `refresh_reader` hook stands in for the refresh call. The
//!   default refresh hook is a no-op (returns `None`), matching a world where
//!   the refresh import fails and the cached token is returned.

use std::path::PathBuf;

use chrono::{DateTime, FixedOffset, NaiveDateTime, TimeZone, Utc};

use crate::mod_hermes_constants::get_hermes_home;

const DEFAULT_TOOL_GATEWAY_DOMAIN: &str = "nousresearch.com";
const DEFAULT_TOOL_GATEWAY_SCHEME: &str = "https";
const NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;

/// Resolved shared managed-tool gateway configuration for a vendor.
///
/// Mirrors the frozen Python dataclass `ManagedToolGatewayConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedToolGatewayConfig {
    pub vendor: String,
    pub gateway_origin: String,
    pub nous_user_token: String,
    pub managed_mode: bool,
}

/// Return the Hermes auth store path, respecting `HERMES_HOME` overrides.
pub fn auth_json_path() -> PathBuf {
    get_hermes_home().join("auth.json")
}

/// Read the `nous` provider object from the auth store, if present.
///
/// Returns `None` on any I/O / parse error or if the structure does not match,
/// mirroring the Python `try/except` that swallows all failures.
fn read_nous_provider_state() -> Option<serde_json::Map<String, serde_json::Value>> {
    let path = auth_json_path();
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    let providers = data.get("providers")?;
    let providers = providers.as_object()?;
    let nous_provider = providers.get("nous")?;
    nous_provider.as_object().cloned()
}

/// Parse an ISO-8601 timestamp into a UTC datetime.
///
/// Faithful port of the Python `_parse_timestamp`:
/// * Non-string / blank inputs yield `None`.
/// * A trailing `Z` is normalised to `+00:00`.
/// * Naive (no-offset) timestamps are treated as UTC.
/// * Aware timestamps are converted to UTC.
fn parse_timestamp(value: Option<&serde_json::Value>) -> Option<DateTime<Utc>> {
    let raw = value?.as_str()?;
    if raw.trim().is_empty() {
        return None;
    }
    let mut normalized = raw.trim().to_string();
    if let Some(stripped) = normalized.strip_suffix('Z') {
        normalized = format!("{stripped}+00:00");
    }

    // Try offset-aware parse first (datetime.fromisoformat with tzinfo).
    if let Ok(dt) = DateTime::<FixedOffset>::parse_from_rfc3339(&normalized) {
        return Some(dt.with_timezone(&Utc));
    }
    // Some ISO strings use a space separator or omit seconds; try a couple of
    // common naive forms and treat them as UTC.
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&normalized, fmt) {
            return Some(Utc.from_utc_datetime(&naive));
        }
        // Date-only needs a separate path (NaiveDate).
        if fmt == "%Y-%m-%d" {
            if let Ok(date) = chrono::NaiveDate::parse_from_str(&normalized, fmt) {
                let naive = date.and_hms_opt(0, 0, 0)?;
                return Some(Utc.from_utc_datetime(&naive));
            }
        }
    }
    None
}

/// Return `true` when the access token is expiring within `skew_seconds`.
///
/// An unparseable / missing `expires_at` is treated as expiring (`true`),
/// matching the Python behaviour.
fn access_token_is_expiring(
    expires_at: Option<&serde_json::Value>,
    skew_seconds: i64,
) -> bool {
    let expires = match parse_timestamp(expires_at) {
        Some(dt) => dt,
        None => return true,
    };
    let remaining = (expires - Utc::now()).num_seconds();
    remaining <= skew_seconds.max(0)
}

/// Read a Nous Subscriber OAuth access token from the auth store or env override.
///
/// Resolution order (faithful to Python `read_nous_access_token`):
/// 1. `TOOL_GATEWAY_USER_TOKEN` env override (trimmed, if non-empty).
/// 2. Cached `access_token` from the auth store, *if* not expiring.
/// 3. A refreshed token from `refresh_reader` (the OAuth refresh hook).
/// 4. Otherwise the (possibly expiring) cached token.
///
/// `refresh_reader` stands in for `hermes_cli.auth.resolve_nous_access_token`;
/// pass `None` to use the default no-op (which mirrors the refresh import
/// failing and the cached token being returned).
pub fn read_nous_access_token_with(
    refresh_reader: Option<&dyn Fn(i64) -> Option<String>>,
) -> Option<String> {
    if let Ok(explicit) = std::env::var("TOOL_GATEWAY_USER_TOKEN") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let nous_provider = read_nous_provider_state().unwrap_or_default();

    let cached_token = nous_provider
        .get("access_token")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    if let Some(ref token) = cached_token {
        if !access_token_is_expiring(
            nous_provider.get("expires_at"),
            NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
        ) {
            return Some(token.clone());
        }
    }

    if let Some(refresh) = refresh_reader {
        if let Some(refreshed) = refresh(NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
            let trimmed = refreshed.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    cached_token
}

/// Convenience wrapper for [`read_nous_access_token_with`] with no refresh hook.
pub fn read_nous_access_token() -> Option<String> {
    read_nous_access_token_with(None)
}

/// Return the configured shared gateway URL scheme.
///
/// Reads `TOOL_GATEWAY_SCHEME` (case-insensitive). Empty / unset yields the
/// default (`https`). Any value other than `http`/`https` is an error.
pub fn get_tool_gateway_scheme() -> Result<String, String> {
    let scheme = std::env::var("TOOL_GATEWAY_SCHEME")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if scheme.is_empty() {
        return Ok(DEFAULT_TOOL_GATEWAY_SCHEME.to_string());
    }
    if scheme == "http" || scheme == "https" {
        return Ok(scheme);
    }
    Err("TOOL_GATEWAY_SCHEME must be 'http' or 'https'".to_string())
}

/// Return the gateway origin for a specific vendor.
///
/// Mirrors `build_vendor_gateway_url`:
/// 1. A vendor-specific `<VENDOR>_GATEWAY_URL` override (trailing `/` stripped).
/// 2. Otherwise `<scheme>://<vendor>-gateway.<TOOL_GATEWAY_DOMAIN>`.
/// 3. Otherwise `<scheme>://<vendor>-gateway.<DEFAULT_TOOL_GATEWAY_DOMAIN>`.
///
/// Returns the same error as [`get_tool_gateway_scheme`] on an invalid scheme.
pub fn build_vendor_gateway_url(vendor: &str) -> Result<String, String> {
    let vendor_key = format!("{}_GATEWAY_URL", vendor.to_uppercase().replace('-', "_"));
    let explicit_vendor_url = std::env::var(&vendor_key)
        .unwrap_or_default()
        .trim()
        .trim_end_matches('/')
        .to_string();
    if !explicit_vendor_url.is_empty() {
        return Ok(explicit_vendor_url);
    }

    let shared_scheme = get_tool_gateway_scheme()?;
    let shared_domain = std::env::var("TOOL_GATEWAY_DOMAIN")
        .unwrap_or_default()
        .trim()
        .trim_matches('/')
        .to_string();
    if !shared_domain.is_empty() {
        return Ok(format!("{shared_scheme}://{vendor}-gateway.{shared_domain}"));
    }

    Ok(format!(
        "{shared_scheme}://{vendor}-gateway.{DEFAULT_TOOL_GATEWAY_DOMAIN}"
    ))
}

/// Pluggable hooks for [`resolve_managed_tool_gateway`].
///
/// Each field is optional; `None` falls back to the native default. This
/// captures the three Python parameters (`managed_nous_tools_enabled`, which is
/// implicit in Python, plus `gateway_builder` and `token_reader`).
pub struct GatewayResolveHooks<'a> {
    /// Stand-in for `tools.tool_backend_helpers.managed_nous_tools_enabled`.
    /// Default: `false` (never block startup; gateway disabled until wired up).
    pub managed_enabled: Option<&'a dyn Fn() -> bool>,
    /// Stand-in for `gateway_builder`. Default: [`build_vendor_gateway_url`].
    pub gateway_builder: Option<&'a dyn Fn(&str) -> Result<String, String>>,
    /// Stand-in for `token_reader`. Default: [`read_nous_access_token`].
    pub token_reader: Option<&'a dyn Fn() -> Option<String>>,
}

impl<'a> Default for GatewayResolveHooks<'a> {
    fn default() -> Self {
        Self {
            managed_enabled: None,
            gateway_builder: None,
            token_reader: None,
        }
    }
}

/// Resolve the shared managed-tool gateway config for a vendor.
///
/// Faithful port of `resolve_managed_tool_gateway`. Returns `None` when managed
/// Nous tools are disabled, or when either the gateway origin or the Nous token
/// resolves to an empty value.
pub fn resolve_managed_tool_gateway(
    vendor: &str,
    hooks: &GatewayResolveHooks<'_>,
) -> Option<ManagedToolGatewayConfig> {
    let enabled = match hooks.managed_enabled {
        Some(f) => f(),
        None => false,
    };
    if !enabled {
        return None;
    }

    let gateway_origin = match hooks.gateway_builder {
        Some(f) => f(vendor).ok()?,
        None => build_vendor_gateway_url(vendor).ok()?,
    };

    let nous_user_token = match hooks.token_reader {
        Some(f) => f(),
        None => read_nous_access_token(),
    };

    let nous_user_token = nous_user_token.filter(|t| !t.is_empty());

    if gateway_origin.is_empty() {
        return None;
    }
    let nous_user_token = nous_user_token?;

    Some(ManagedToolGatewayConfig {
        vendor: vendor.to_string(),
        gateway_origin,
        nous_user_token,
        managed_mode: true,
    })
}

/// Return `true` when the gateway URL and Nous access token are both available.
///
/// Faithful port of `is_managed_tool_gateway_ready`.
pub fn is_managed_tool_gateway_ready(vendor: &str, hooks: &GatewayResolveHooks<'_>) -> bool {
    resolve_managed_tool_gateway(vendor, hooks).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env vars are process-global; serialise tests that touch them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        unsafe {
            std::env::remove_var("TOOL_GATEWAY_SCHEME");
            std::env::remove_var("TOOL_GATEWAY_DOMAIN");
            std::env::remove_var("TOOL_GATEWAY_USER_TOKEN");
            std::env::remove_var("OPENAI_GATEWAY_URL");
            std::env::remove_var("ELEVEN_LABS_GATEWAY_URL");
        }
    }

    #[test]
    fn scheme_defaults_to_https() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(get_tool_gateway_scheme().unwrap(), "https");
    }

    #[test]
    fn scheme_honours_http_override_case_insensitive() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("TOOL_GATEWAY_SCHEME", "  HTTP ");
        }
        assert_eq!(get_tool_gateway_scheme().unwrap(), "http");
        clear_env();
    }

    #[test]
    fn scheme_rejects_invalid() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("TOOL_GATEWAY_SCHEME", "ftp");
        }
        assert!(get_tool_gateway_scheme().is_err());
        clear_env();
    }

    #[test]
    fn vendor_url_default_domain() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(
            build_vendor_gateway_url("openai").unwrap(),
            "https://openai-gateway.nousresearch.com"
        );
    }

    #[test]
    fn vendor_url_shared_domain_strips_slashes() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("TOOL_GATEWAY_DOMAIN", "/example.com/");
        }
        assert_eq!(
            build_vendor_gateway_url("openai").unwrap(),
            "https://openai-gateway.example.com"
        );
        clear_env();
    }

    #[test]
    fn vendor_url_explicit_override_strips_trailing_slash() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("OPENAI_GATEWAY_URL", "https://custom.example.com/");
        }
        assert_eq!(
            build_vendor_gateway_url("openai").unwrap(),
            "https://custom.example.com"
        );
        clear_env();
    }

    #[test]
    fn vendor_key_uppercases_and_replaces_hyphens() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("ELEVEN_LABS_GATEWAY_URL", "https://el.example.com");
        }
        assert_eq!(
            build_vendor_gateway_url("eleven-labs").unwrap(),
            "https://el.example.com"
        );
        clear_env();
    }

    #[test]
    fn token_env_override_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("TOOL_GATEWAY_USER_TOKEN", "  tok-abc  ");
        }
        assert_eq!(read_nous_access_token().as_deref(), Some("tok-abc"));
        clear_env();
    }

    #[test]
    fn parse_timestamp_handles_z_suffix() {
        let v = serde_json::Value::String("2030-01-01T00:00:00Z".to_string());
        let parsed = parse_timestamp(Some(&v)).unwrap();
        assert_eq!(parsed.timezone(), Utc);
    }

    #[test]
    fn parse_timestamp_rejects_blank_and_nonstring() {
        assert!(parse_timestamp(Some(&serde_json::Value::String("   ".into()))).is_none());
        assert!(parse_timestamp(Some(&serde_json::json!(123))).is_none());
        assert!(parse_timestamp(None).is_none());
    }

    #[test]
    fn expiring_missing_is_true() {
        assert!(access_token_is_expiring(None, 120));
    }

    #[test]
    fn expiring_far_future_is_false() {
        let v = serde_json::Value::String("2999-01-01T00:00:00Z".to_string());
        assert!(!access_token_is_expiring(Some(&v), 120));
    }

    #[test]
    fn expiring_past_is_true() {
        let v = serde_json::Value::String("2000-01-01T00:00:00Z".to_string());
        assert!(access_token_is_expiring(Some(&v), 120));
    }

    #[test]
    fn resolve_disabled_returns_none() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let hooks = GatewayResolveHooks::default();
        assert!(resolve_managed_tool_gateway("openai", &hooks).is_none());
        clear_env();
    }

    #[test]
    fn resolve_with_hooks_builds_config() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let enabled = || true;
        let token = || Some("tok-xyz".to_string());
        let hooks = GatewayResolveHooks {
            managed_enabled: Some(&enabled),
            gateway_builder: None,
            token_reader: Some(&token),
        };
        let cfg = resolve_managed_tool_gateway("openai", &hooks).unwrap();
        assert_eq!(cfg.vendor, "openai");
        assert_eq!(cfg.gateway_origin, "https://openai-gateway.nousresearch.com");
        assert_eq!(cfg.nous_user_token, "tok-xyz");
        assert!(cfg.managed_mode);
        assert!(is_managed_tool_gateway_ready("openai", &hooks));
        clear_env();
    }

    #[test]
    fn resolve_empty_token_returns_none() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let enabled = || true;
        let token = || Some(String::new());
        let hooks = GatewayResolveHooks {
            managed_enabled: Some(&enabled),
            gateway_builder: None,
            token_reader: Some(&token),
        };
        assert!(resolve_managed_tool_gateway("openai", &hooks).is_none());
        clear_env();
    }
}
