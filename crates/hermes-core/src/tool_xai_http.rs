//! Shared helpers for direct xAI HTTP integrations.
//!
//! Native Rust port of `tools/xai_http.py`.
//!
//! The Python original looks up `hermes_cli.__version__`, falling back to the
//! string `"unknown"` when that import fails, and returns a stable
//! `User-Agent` of the form `Hermes-Agent/<version>`.
//!
//! In the Rust workspace the equivalent of `hermes_cli.__version__` is the
//! crate version exposed via the `CARGO_PKG_VERSION` environment variable at
//! compile time. We treat an empty version string as the `"unknown"` fallback
//! to faithfully mirror the Python behaviour.

/// The fallback version string used when no concrete version is available.
///
/// Mirrors the Python `__version__ = "unknown"` branch.
pub const HERMES_XAI_UNKNOWN_VERSION: &str = "unknown";

/// Resolve the Hermes version used in the xAI `User-Agent` string.
///
/// Returns the compile-time crate version, or [`HERMES_XAI_UNKNOWN_VERSION`]
/// when that version is empty (the analogue of the Python import failing).
pub fn hermes_xai_version() -> &'static str {
    let version = env!("CARGO_PKG_VERSION");
    if version.is_empty() {
        HERMES_XAI_UNKNOWN_VERSION
    } else {
        version
    }
}

/// Return a stable Hermes-specific `User-Agent` for xAI HTTP calls.
///
/// Equivalent to the Python `hermes_xai_user_agent()`:
/// `f"Hermes-Agent/{__version__}"`.
pub fn hermes_xai_user_agent() -> String {
    format!("Hermes-Agent/{}", hermes_xai_version())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_agent_has_expected_prefix() {
        let ua = hermes_xai_user_agent();
        assert!(
            ua.starts_with("Hermes-Agent/"),
            "unexpected user agent: {ua}"
        );
    }

    #[test]
    fn user_agent_embeds_resolved_version() {
        let ua = hermes_xai_user_agent();
        let version = hermes_xai_version();
        assert_eq!(ua, format!("Hermes-Agent/{version}"));
    }

    #[test]
    fn version_is_never_empty() {
        // Either the real crate version or the "unknown" fallback, but never
        // an empty string.
        assert!(!hermes_xai_version().is_empty());
    }

    #[test]
    fn user_agent_has_no_whitespace_in_token() {
        // A User-Agent product token should not contain spaces.
        let ua = hermes_xai_user_agent();
        assert!(!ua.contains(' '), "user agent should be a single token: {ua}");
    }
}
