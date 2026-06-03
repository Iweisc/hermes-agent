//! Shared HTTP client tuning for long-lived platform adapters.
//!
//! Gateway messaging platforms (QQ Bot, Feishu, WeCom, DingTalk, Signal,
//! BlueBubbles, WeCom-callback) keep a persistent HTTP client alive for the
//! adapter's lifetime. That amortises TLS/connection setup across many API
//! calls, but it also means the process's file-descriptor pressure is
//! sensitive to how aggressively the pool recycles idle keep-alive
//! connections.
//!
//! The upstream Python uses httpx whose default `keepalive_expiry` is 5
//! seconds. On macOS behind Cloudflare Warp (and other transparent proxies),
//! peer-initiated FIN can sit in `CLOSE_WAIT` longer than that before the
//! local socket actually drains — which, multiplied across 7 long-lived
//! adapters plus the LLM client and MCP clients, walks straight into the
//! default 256 fd limit. See #18451.
//!
//! [`platform_httpx_limits`] returns a tighter set of limits the adapter
//! factories use instead of the httpx default. The values chosen:
//!
//! * `max_keepalive_connections = 10` — plenty for any single adapter;
//!   platform APIs rarely parallelise beyond this.
//! * `keepalive_expiry = 2.0` — close idle sockets aggressively so a proxy's
//!   lingering CLOSE_WAIT window can't starve the process.
//!
//! Override via `HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY` /
//! `HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE` env vars when tuning under load.

use std::env;

/// Default idle keep-alive expiry, in seconds.
pub const DEFAULT_KEEPALIVE_EXPIRY_S: f64 = 2.0;

/// Default maximum number of idle keep-alive connections to retain.
pub const DEFAULT_MAX_KEEPALIVE: u32 = 10;

/// HTTP connection-pool limits tuned for persistent platform-adapter clients.
///
/// Mirrors the subset of `httpx.Limits` the upstream Python helper produces.
/// `max_connections` is intentionally left at the httpx default (100) and is
/// represented here as `None` to signal "use the client default".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlatformHttpxLimits {
    /// Maximum number of idle keep-alive connections to retain in the pool.
    pub max_keepalive_connections: u32,
    /// Maximum number of concurrent connections; `None` means use the default.
    pub max_connections: Option<u32>,
    /// How long, in seconds, an idle keep-alive connection may live.
    pub keepalive_expiry: f64,
}

/// Read an environment variable as a positive `f64`.
///
/// Returns `default` when the variable is unset, blank, unparsable, or not
/// strictly greater than zero — matching the Python `_env_float` semantics.
fn env_float(name: &str, default: f64) -> f64 {
    let raw = env::var(name).unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return default;
    }
    match raw.parse::<f64>() {
        Ok(val) if val > 0.0 => val,
        _ => default,
    }
}

/// Read an environment variable as a positive `u32`.
///
/// Returns `default` when the variable is unset, blank, unparsable, or not
/// strictly greater than zero — matching the Python `_env_int` semantics.
fn env_int(name: &str, default: u32) -> u32 {
    let raw = env::var(name).unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return default;
    }
    // Python `int()` rejects floats like "10.5" and negative values produce a
    // valid parse that the `> 0` guard then accepts/rejects. We parse a signed
    // integer first so negatives parse (and are then rejected by the guard),
    // mirroring CPython's behaviour.
    match raw.parse::<i64>() {
        Ok(val) if val > 0 => val as u32,
        _ => default,
    }
}

/// Return connection-pool limits tuned for persistent platform-adapter clients.
///
/// Always returns `Some` in the Rust port: unlike the Python helper, which
/// returns `None` when httpx is not importable, reqwest is a hard dependency
/// here so the limits can always be computed. The `Option` is retained so the
/// call site keeps the same "fall back to client default" shape.
pub fn platform_httpx_limits() -> Option<PlatformHttpxLimits> {
    let keepalive_expiry = env_float(
        "HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY",
        DEFAULT_KEEPALIVE_EXPIRY_S,
    );
    let max_keepalive = env_int("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", DEFAULT_MAX_KEEPALIVE);

    Some(PlatformHttpxLimits {
        max_keepalive_connections: max_keepalive,
        // Leave max_connections at the client default (100) — plenty of headroom.
        max_connections: None,
        keepalive_expiry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env mutation is process-global; serialise the tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const EXPIRY: &str = "HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY";
    const MAX_KA: &str = "HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE";

    fn clear_env() {
        unsafe { env::remove_var(EXPIRY); }
        unsafe { env::remove_var(MAX_KA); }
    }

    #[test]
    fn defaults_when_env_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.max_keepalive_connections, DEFAULT_MAX_KEEPALIVE);
        assert_eq!(limits.keepalive_expiry, DEFAULT_KEEPALIVE_EXPIRY_S);
        assert_eq!(limits.max_connections, None);
    }

    #[test]
    fn env_overrides_applied() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(EXPIRY, "7.5"); }
        unsafe { env::set_var(MAX_KA, "42"); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.keepalive_expiry, 7.5);
        assert_eq!(limits.max_keepalive_connections, 42);
        clear_env();
    }

    #[test]
    fn blank_and_whitespace_fall_back_to_default() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(EXPIRY, "   "); }
        unsafe { env::set_var(MAX_KA, ""); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.keepalive_expiry, DEFAULT_KEEPALIVE_EXPIRY_S);
        assert_eq!(limits.max_keepalive_connections, DEFAULT_MAX_KEEPALIVE);
        clear_env();
    }

    #[test]
    fn whitespace_padded_values_are_trimmed() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(EXPIRY, "  3.0  "); }
        unsafe { env::set_var(MAX_KA, "  5  "); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.keepalive_expiry, 3.0);
        assert_eq!(limits.max_keepalive_connections, 5);
        clear_env();
    }

    #[test]
    fn unparsable_falls_back_to_default() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(EXPIRY, "not-a-number"); }
        unsafe { env::set_var(MAX_KA, "abc"); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.keepalive_expiry, DEFAULT_KEEPALIVE_EXPIRY_S);
        assert_eq!(limits.max_keepalive_connections, DEFAULT_MAX_KEEPALIVE);
        clear_env();
    }

    #[test]
    fn non_positive_falls_back_to_default() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(EXPIRY, "0"); }
        unsafe { env::set_var(MAX_KA, "-3"); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.keepalive_expiry, DEFAULT_KEEPALIVE_EXPIRY_S);
        assert_eq!(limits.max_keepalive_connections, DEFAULT_MAX_KEEPALIVE);
        clear_env();
    }

    #[test]
    fn float_for_int_env_falls_back() {
        // Python int("10.5") raises ValueError -> default. Match that.
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { env::set_var(MAX_KA, "10.5"); }
        let limits = platform_httpx_limits().expect("always Some");
        assert_eq!(limits.max_keepalive_connections, DEFAULT_MAX_KEEPALIVE);
        clear_env();
    }

    #[test]
    fn env_float_helper_direct() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(env_float("HERMES_NONEXISTENT_FLOAT_VAR", 1.5), 1.5);
    }

    #[test]
    fn env_int_helper_direct() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(env_int("HERMES_NONEXISTENT_INT_VAR", 9), 9);
    }
}
