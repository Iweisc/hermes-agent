//! Native Rust port of `hermes_cli/timeouts.py`.
//!
//! Resolves configured provider request/stale timeouts from the loaded
//! configuration document. The Python original looks up the config via
//! `hermes_cli.config.load_config`; here we delegate to
//! [`crate::cli_config::load_config`], which returns a `serde_json::Value`.
//!
//! Behaviour notes:
//! - Timeouts are coerced to `f64`; non-numeric, missing, or non-positive
//!   values yield `None` (mirroring Python's `float()` + `<= 0` guard).
//! - Python coerces strings/ints/floats to float. We replicate that by
//!   accepting JSON numbers and numeric strings.
//! - A model-level override takes precedence over the provider-level value.

use serde_json::Value;

/// Coerce an arbitrary JSON value into a positive timeout in seconds.
///
/// Mirrors Python's `_coerce_timeout`: anything that cannot be turned into a
/// `float`, or that is `<= 0`, returns `None`.
fn coerce_timeout(raw: Option<&Value>) -> Option<f64> {
    let value = raw?;
    let timeout = value_to_f64(value)?;
    if timeout <= 0.0 || timeout.is_nan() {
        return None;
    }
    Some(timeout)
}

/// Replicate Python's permissive `float(raw)` coercion for the value kinds we
/// can encounter in a parsed JSON/YAML config: numbers and numeric strings.
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => {
            // Python `float(True)` == 1.0, `float(False)` == 0.0.
            Some(if *b { 1.0 } else { 0.0 })
        }
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<f64>().ok()
        }
        _ => None,
    }
}

/// Return the model-specific config sub-document, if present and a mapping.
///
/// Mirrors Python's `_get_model_config`.
fn get_model_config<'a>(provider_config: &'a Value, model: Option<&str>) -> Option<&'a Value> {
    let model = model?;
    if model.is_empty() {
        return None;
    }

    let models = provider_config.get("models");
    let model_config = match models {
        Some(Value::Object(map)) => map.get(model),
        _ => None,
    }?;

    if model_config.is_object() {
        Some(model_config)
    } else {
        None
    }
}

/// Extract the provider config mapping from a loaded config document.
///
/// Returns `None` when the provider id is empty or the corresponding config is
/// not a mapping, matching the Python guards.
fn provider_config_for<'a>(config: &'a Value, provider_id: &str) -> Option<&'a Value> {
    if provider_id.is_empty() {
        return None;
    }

    let providers = match config {
        Value::Object(map) => map.get("providers"),
        _ => return None,
    };

    let provider_config = match providers {
        Some(Value::Object(map)) => map.get(provider_id),
        _ => None,
    }?;

    if provider_config.is_object() {
        Some(provider_config)
    } else {
        None
    }
}

/// Resolve a configured provider request timeout in seconds, if any.
///
/// Port of `get_provider_request_timeout`. A model-level `timeout_seconds`
/// override takes precedence over the provider-level `request_timeout_seconds`.
/// Loads the configuration via [`crate::cli_config::load_config`]; any failure
/// there results in `None`.
pub fn get_provider_request_timeout(provider_id: &str, model: Option<&str>) -> Option<f64> {
    let config = load_config_safe()?;
    resolve_timeout(&config, provider_id, model, "timeout_seconds", "request_timeout_seconds")
}

/// Resolve a configured non-stream stale timeout in seconds, if any.
///
/// Port of `get_provider_stale_timeout`. A model-level `stale_timeout_seconds`
/// override takes precedence over the provider-level `stale_timeout_seconds`.
pub fn get_provider_stale_timeout(provider_id: &str, model: Option<&str>) -> Option<f64> {
    let config = load_config_safe()?;
    resolve_timeout(&config, provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds")
}

/// Shared resolution logic over an already-loaded config document. Exposed so
/// callers (and tests) can resolve against an in-memory config without going
/// through `load_config`.
pub fn resolve_timeout(
    config: &Value,
    provider_id: &str,
    model: Option<&str>,
    model_key: &str,
    provider_key: &str,
) -> Option<f64> {
    let provider_config = provider_config_for(config, provider_id)?;

    if let Some(model_config) = get_model_config(provider_config, model) {
        if let Some(timeout) = coerce_timeout(model_config.get(model_key)) {
            return Some(timeout);
        }
    }

    coerce_timeout(provider_config.get(provider_key))
}

/// Load the config document, swallowing any error into `None`, matching the
/// Python `try/except Exception` around `load_config()`.
///
/// `crate::cli_config::load_config` returns a `Value` directly (no Result), so
/// the only "failure" mode we model is a non-mapping document, which the
/// downstream helpers already treat as absent.
fn load_config_safe() -> Option<Value> {
    Some(crate::cli_config::load_config())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coerce_rejects_non_positive_and_invalid() {
        assert_eq!(coerce_timeout(Some(&json!(0))), None);
        assert_eq!(coerce_timeout(Some(&json!(-5))), None);
        assert_eq!(coerce_timeout(Some(&json!("abc"))), None);
        assert_eq!(coerce_timeout(Some(&json!(null))), None);
        assert_eq!(coerce_timeout(None), None);
        assert_eq!(coerce_timeout(Some(&json!([]))), None);
    }

    #[test]
    fn coerce_accepts_numbers_and_numeric_strings() {
        assert_eq!(coerce_timeout(Some(&json!(30))), Some(30.0));
        assert_eq!(coerce_timeout(Some(&json!(12.5))), Some(12.5));
        assert_eq!(coerce_timeout(Some(&json!("45"))), Some(45.0));
        assert_eq!(coerce_timeout(Some(&json!("  7.5 "))), Some(7.5));
    }

    fn sample_config() -> Value {
        json!({
            "providers": {
                "openai": {
                    "request_timeout_seconds": 60,
                    "stale_timeout_seconds": 20,
                    "models": {
                        "gpt-4": {
                            "timeout_seconds": 120,
                            "stale_timeout_seconds": 15
                        },
                        "gpt-3.5": {
                            "timeout_seconds": "0"
                        }
                    }
                },
                "bad": "not-a-mapping"
            }
        })
    }

    #[test]
    fn empty_provider_id_returns_none() {
        let cfg = sample_config();
        assert_eq!(resolve_timeout(&cfg, "", None, "timeout_seconds", "request_timeout_seconds"), None);
    }

    #[test]
    fn provider_level_request_timeout() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "openai", None, "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, Some(60.0));
    }

    #[test]
    fn model_override_takes_precedence() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "openai", Some("gpt-4"), "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, Some(120.0));
    }

    #[test]
    fn model_invalid_falls_back_to_provider() {
        // gpt-3.5 has timeout_seconds "0" -> coerces to non-positive -> None,
        // so falls back to provider request_timeout_seconds.
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "openai", Some("gpt-3.5"), "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, Some(60.0));
    }

    #[test]
    fn stale_timeout_model_override() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "openai", Some("gpt-4"), "stale_timeout_seconds", "stale_timeout_seconds");
        assert_eq!(v, Some(15.0));
    }

    #[test]
    fn stale_timeout_provider_level() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "openai", None, "stale_timeout_seconds", "stale_timeout_seconds");
        assert_eq!(v, Some(20.0));
    }

    #[test]
    fn non_mapping_provider_returns_none() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "bad", None, "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, None);
    }

    #[test]
    fn missing_provider_returns_none() {
        let cfg = sample_config();
        let v = resolve_timeout(&cfg, "nope", None, "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, None);
    }

    #[test]
    fn non_mapping_config_returns_none() {
        let cfg = json!("just a string");
        let v = resolve_timeout(&cfg, "openai", None, "timeout_seconds", "request_timeout_seconds");
        assert_eq!(v, None);
    }
}
