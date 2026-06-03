//! Shared gateway restart constants and parsing helpers.
//!
//! Native Rust port of `gateway/restart.py`. Provides the service-manager
//! restart exit code and the drain-timeout parsing helper used along the
//! graceful drain/reload path.

use serde_json::Value;

/// `EX_TEMPFAIL` from `sysexits.h` — used to ask the service manager to
/// restart the gateway after a graceful drain/reload path completes.
pub const GATEWAY_SERVICE_RESTART_EXIT_CODE: i32 = 75;

/// Hardcoded fallback for `DEFAULT_CONFIG["agent"]["restart_drain_timeout"]`
/// in case the shared config cannot be loaded/parsed. Mirrors the Python
/// default of `180` (seconds).
const RESTART_DRAIN_TIMEOUT_FALLBACK: f64 = 180.0;

/// Default gateway restart drain timeout (seconds).
///
/// Equivalent of Python's
/// `float(DEFAULT_CONFIG["agent"]["restart_drain_timeout"])`. Reads from the
/// shared default config when available, falling back to
/// [`RESTART_DRAIN_TIMEOUT_FALLBACK`] otherwise.
pub fn default_gateway_restart_drain_timeout() -> f64 {
    default_drain_timeout_from_config().unwrap_or(RESTART_DRAIN_TIMEOUT_FALLBACK)
}

/// Look up `agent.restart_drain_timeout` from the shared default config and
/// coerce it to an `f64`, mirroring Python's `float(...)`. Returns `None` if
/// the key is missing or not coercible to a finite float.
fn default_drain_timeout_from_config() -> Option<f64> {
    // cli_config returns a serde_yaml::Value; bridge to serde_json::Value so the
    // coercion helpers (written against serde_json) apply unchanged.
    let cfg_yaml = crate::cli_config::default_config();
    let cfg: Value = serde_json::to_value(&cfg_yaml).ok()?;
    let raw = cfg.get("agent")?.get("restart_drain_timeout")?;
    coerce_float(raw).filter(|v| v.is_finite())
}

/// Coerce a JSON value to `f64` the way Python's `float()` would for the
/// inputs this module accepts: numbers parse directly, strings are parsed
/// after trimming, everything else yields `None`.
fn coerce_float(raw: &Value) -> Option<f64> {
    match raw {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse::<f64>().ok()
            }
        }
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Parse a configured drain timeout, falling back to the shared default.
///
/// Port of `gateway.restart.parse_restart_drain_timeout`. Behavior:
///   * `None` / JSON null / empty-or-whitespace string -> shared default
///   * a value coercible to a finite float -> `max(0.0, value)`
///   * anything else (non-numeric string, list, object, NaN, ...) -> default
pub fn parse_restart_drain_timeout(raw: Option<&Value>) -> f64 {
    let default = default_gateway_restart_drain_timeout();

    // Python first computes `str(raw or "").strip()`; an empty result means
    // "fall back to the default" rather than attempting a float conversion.
    // This covers: None, null, the empty string, whitespace, and falsy 0/""
    // -> we treat None/null/empty-string as the default below.
    let value = match raw {
        None | Some(Value::Null) => return default,
        Some(Value::String(s)) if s.trim().is_empty() => return default,
        Some(v) => v,
    };

    match coerce_float(value) {
        Some(parsed) if parsed.is_finite() => parsed.max(0.0),
        // Non-coercible or non-finite (e.g. NaN/inf-like) -> default, matching
        // the Python try/except returning the default on ValueError/TypeError.
        _ => default,
    }
}

/// Convenience overload accepting an owned [`Value`].
pub fn parse_restart_drain_timeout_value(raw: &Value) -> f64 {
    parse_restart_drain_timeout(Some(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn restart_exit_code_is_ex_tempfail() {
        assert_eq!(GATEWAY_SERVICE_RESTART_EXIT_CODE, 75);
    }

    #[test]
    fn default_matches_python_config() {
        // DEFAULT_CONFIG["agent"]["restart_drain_timeout"] == 180
        assert_eq!(default_gateway_restart_drain_timeout(), 180.0);
    }

    #[test]
    fn none_returns_default() {
        assert_eq!(parse_restart_drain_timeout(None), 180.0);
    }

    #[test]
    fn null_returns_default() {
        assert_eq!(parse_restart_drain_timeout(Some(&Value::Null)), 180.0);
    }

    #[test]
    fn empty_or_whitespace_string_returns_default() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!(""))), 180.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!("   "))), 180.0);
    }

    #[test]
    fn numeric_value_passes_through() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!(30))), 30.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!(12.5))), 12.5);
        assert_eq!(parse_restart_drain_timeout(Some(&json!(0))), 0.0);
    }

    #[test]
    fn numeric_string_parses() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!("45"))), 45.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!("  7.5  "))), 7.5);
    }

    #[test]
    fn negative_clamped_to_zero() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!(-5))), 0.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!("-10.2"))), 0.0);
    }

    #[test]
    fn non_numeric_string_returns_default() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!("abc"))), 180.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!("12abc"))), 180.0);
    }

    #[test]
    fn non_scalar_returns_default() {
        assert_eq!(parse_restart_drain_timeout(Some(&json!([1, 2]))), 180.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!({"x": 1}))), 180.0);
    }

    #[test]
    fn nan_returns_default() {
        // A NaN string parses to a non-finite float -> default.
        assert_eq!(parse_restart_drain_timeout(Some(&json!("nan"))), 180.0);
        assert_eq!(parse_restart_drain_timeout(Some(&json!("inf"))), 180.0);
    }

    #[test]
    fn value_overload_matches() {
        assert_eq!(parse_restart_drain_timeout_value(&json!(30)), 30.0);
    }
}
