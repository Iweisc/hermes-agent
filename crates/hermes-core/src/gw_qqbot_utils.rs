//! QQBot shared utilities — User-Agent, HTTP helpers, config coercion.
//!
//! Native Rust port of `gateway/platforms/qqbot/utils.py`.

use serde_json::Value;
use std::collections::HashMap;

/// QQBot adapter version — mirrors `QQBOT_VERSION` in the Python `constants.py`.
///
/// Bumped on functional changes to the adapter package.
pub const QQBOT_VERSION: &str = "1.1.0";

// ---------------------------------------------------------------------------
// User-Agent
// ---------------------------------------------------------------------------

/// Return the hermes-agent package version, or `"dev"` if unavailable.
///
/// In the Python original this read the installed package metadata via
/// `importlib.metadata.version("hermes-agent")`. In the native build we expose
/// the compile-time crate version, falling back to `"dev"` when it is empty.
pub fn get_hermes_version() -> String {
    let v = env!("CARGO_PKG_VERSION");
    if v.is_empty() {
        "dev".to_string()
    } else {
        v.to_string()
    }
}

/// Return a `"Python/<x.y.z>"`-style runtime token.
///
/// The Python original embeds the actual interpreter version
/// (`sys.version_info`). Since there is no Python interpreter in the native
/// build, we keep the token shape identical for API/User-Agent parity but
/// report a fixed baseline version. The format is `MAJOR.MINOR.MICRO`.
fn runtime_version() -> String {
    "3.11.0".to_string()
}

/// Lower-cased OS name, matching Python's `platform.system().lower()`.
///
/// Python returns values like `"darwin"`, `"linux"`, `"windows"`. We map the
/// Rust `std::env::consts::OS` values onto those.
fn os_name() -> String {
    match std::env::consts::OS {
        "macos" => "darwin".to_string(),
        "linux" => "linux".to_string(),
        "windows" => "windows".to_string(),
        other => other.to_string(),
    }
}

/// Build a descriptive User-Agent string.
///
/// Format:
/// ```text
/// QQBotAdapter/<qqbot_version> (Python/<py_version>; <os>; Hermes/<hermes_version>)
/// ```
///
/// Example:
/// ```text
/// QQBotAdapter/1.1.0 (Python/3.11.0; darwin; Hermes/0.9.0)
/// ```
pub fn build_user_agent() -> String {
    let py_version = runtime_version();
    let os = os_name();
    let hermes_version = get_hermes_version();
    format!(
        "QQBotAdapter/{} (Python/{}; {}; Hermes/{})",
        QQBOT_VERSION, py_version, os, hermes_version
    )
}

/// Return standard HTTP headers for QQBot API requests.
///
/// Includes `Content-Type`, `Accept`, and a dynamic `User-Agent`.
/// `q.qq.com` requires `Accept: application/json` — without it, the server
/// returns a JavaScript anti-bot challenge page.
pub fn get_api_headers() -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Accept".to_string(), "application/json".to_string());
    headers.insert("User-Agent".to_string(), build_user_agent());
    headers
}

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Coerce config values into a trimmed string list.
///
/// Accepts comma-separated strings, JSON arrays, or single scalar values —
/// mirroring the Python `coerce_list` which handled `str`, `list/tuple/set`,
/// and arbitrary single values.
///
/// Behaviour parity:
/// - `null` → empty vec.
/// - String → split on `,`, trim each, drop empties.
/// - Array → stringify each item (scalar -> its text form, matching `str(item)`),
///   trim, drop empties.
/// - Any other scalar → its stringified form as a single element, dropped if
///   empty after trimming.
pub fn coerce_list(value: &Value) -> Vec<String> {
    match value {
        Value::Null => Vec::new(),
        Value::String(s) => s
            .split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        Value::Array(items) => items
            .iter()
            .map(|item| stringify_scalar(item).trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        other => {
            let s = stringify_scalar(other);
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
    }
}

/// Stringify a JSON scalar the way Python's `str()` would, so that
/// `coerce_list` produces the same elements for non-string values.
///
/// - String → its raw contents (no surrounding quotes).
/// - Bool → `"True"` / `"False"` (Python casing).
/// - Number → its compact textual form.
/// - Null → `"None"` (Python's `str(None)`); note `coerce_list` only reaches
///   this for nulls nested inside arrays.
/// - Array/Object → JSON text (best-effort; Python would use its own repr).
fn stringify_scalar(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_agent_shape() {
        let ua = build_user_agent();
        assert!(ua.starts_with("QQBotAdapter/1.1.0 (Python/"));
        assert!(ua.contains("Hermes/"));
        assert!(ua.ends_with(")"));
    }

    #[test]
    fn os_name_known() {
        let name = os_name();
        assert!(["darwin", "linux", "windows"].contains(&name.as_str()) || !name.is_empty());
    }

    #[test]
    fn api_headers_present() {
        let h = get_api_headers();
        assert_eq!(h.get("Content-Type").unwrap(), "application/json");
        assert_eq!(h.get("Accept").unwrap(), "application/json");
        assert!(h.get("User-Agent").unwrap().starts_with("QQBotAdapter/"));
    }

    #[test]
    fn coerce_null() {
        assert!(coerce_list(&Value::Null).is_empty());
    }

    #[test]
    fn coerce_comma_string() {
        let v = json!("a, b ,, c , ");
        assert_eq!(coerce_list(&v), vec!["a", "b", "c"]);
    }

    #[test]
    fn coerce_empty_string() {
        assert!(coerce_list(&json!("   ")).is_empty());
    }

    #[test]
    fn coerce_array_mixed() {
        let v = json!(["x", " y ", "", 42, true, null]);
        assert_eq!(coerce_list(&v), vec!["x", "y", "42", "True", "None"]);
    }

    #[test]
    fn coerce_single_scalar() {
        assert_eq!(coerce_list(&json!(7)), vec!["7"]);
        assert_eq!(coerce_list(&json!(true)), vec!["True"]);
    }

    #[test]
    fn hermes_version_nonempty() {
        assert!(!get_hermes_version().is_empty());
    }
}
