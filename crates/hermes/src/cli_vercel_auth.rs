//! Helpers for reporting Vercel Sandbox authentication state.
//!
//! Native Rust port of `hermes_cli/vercel_auth.py`. Inspects environment
//! variables to describe how (or whether) Vercel auth is configured, without
//! ever exposing secret values.

use std::env;

/// Environment variables that together constitute access-token auth.
const TOKEN_TUPLE_VARS: [&str; 3] = ["VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID"];

/// Snapshot of the current Vercel authentication state.
///
/// Mirrors the frozen `VercelAuthStatus` dataclass from the Python source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VercelAuthStatus {
    /// Whether auth is considered usable (`True`/`False` in Python).
    pub ok: bool,
    /// Short human-readable summary label.
    pub label: String,
    /// Ordered detail lines describing the state (never contains secrets).
    pub detail_lines: Vec<String>,
}

impl VercelAuthStatus {
    fn new(ok: bool, label: impl Into<String>, detail_lines: Vec<String>) -> Self {
        Self {
            ok,
            label: label.into(),
            detail_lines,
        }
    }
}

/// Returns `true` when the named env var is set to a non-empty value.
///
/// Matches Python's `bool(os.getenv(name))`: missing or empty string is falsey.
fn present(name: &str) -> bool {
    matches!(env::var(name), Ok(value) if !value.is_empty())
}

/// Return Vercel auth status without exposing secret values.
///
/// Reads the process environment directly.
pub fn describe_vercel_auth() -> VercelAuthStatus {
    describe_with(|name| present(name))
}

/// Core logic parameterised over a presence-checking closure so it can be
/// unit-tested deterministically without mutating the real environment.
fn describe_with<F: Fn(&str) -> bool>(is_present: F) -> VercelAuthStatus {
    let has_oidc = is_present("VERCEL_OIDC_TOKEN");

    let present_token_vars: Vec<&str> = TOKEN_TUPLE_VARS
        .iter()
        .copied()
        .filter(|name| is_present(name))
        .collect();
    let missing_token_vars: Vec<&str> = TOKEN_TUPLE_VARS
        .iter()
        .copied()
        .filter(|name| !is_present(name))
        .collect();

    if has_oidc {
        let mut details = vec![
            "mode: OIDC".to_string(),
            "active env: VERCEL_OIDC_TOKEN".to_string(),
            "note: OIDC tokens are development-only; use access-token auth for deployments and long-running processes".to_string(),
        ];
        if !present_token_vars.is_empty() {
            details.push(format!("also present: {}", present_token_vars.join(", ")));
        }
        return VercelAuthStatus::new(true, "OIDC token via VERCEL_OIDC_TOKEN", details);
    }

    if missing_token_vars.is_empty() {
        return VercelAuthStatus::new(
            true,
            "access token + project/team via VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID",
            vec![
                "mode: access token".to_string(),
                "active env: VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID".to_string(),
            ],
        );
    }

    if !present_token_vars.is_empty() {
        return VercelAuthStatus::new(
            false,
            format!(
                "partial access-token auth (missing {})",
                missing_token_vars.join(", ")
            ),
            vec![
                "mode: incomplete access token".to_string(),
                format!("present env: {}", present_token_vars.join(", ")),
                format!("missing env: {}", missing_token_vars.join(", ")),
                "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together"
                    .to_string(),
            ],
        );
    }

    VercelAuthStatus::new(
        false,
        "not configured",
        vec![
            "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID".to_string(),
            "development-only alternative: set VERCEL_OIDC_TOKEN".to_string(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Build a presence checker over a fixed set of "present" variable names.
    fn checker(present_names: &[&str]) -> impl Fn(&str) -> bool {
        let set: HashSet<String> = present_names.iter().map(|s| s.to_string()).collect();
        move |name: &str| set.contains(name)
    }

    #[test]
    fn oidc_only() {
        let status = describe_with(checker(&["VERCEL_OIDC_TOKEN"]));
        assert!(status.ok);
        assert_eq!(status.label, "OIDC token via VERCEL_OIDC_TOKEN");
        assert_eq!(
            status.detail_lines,
            vec![
                "mode: OIDC".to_string(),
                "active env: VERCEL_OIDC_TOKEN".to_string(),
                "note: OIDC tokens are development-only; use access-token auth for deployments and long-running processes".to_string(),
            ]
        );
    }

    #[test]
    fn oidc_with_some_tokens_present() {
        let status =
            describe_with(checker(&["VERCEL_OIDC_TOKEN", "VERCEL_TOKEN", "VERCEL_TEAM_ID"]));
        assert!(status.ok);
        assert_eq!(status.label, "OIDC token via VERCEL_OIDC_TOKEN");
        // present_token_vars preserve TOKEN_TUPLE_VARS ordering.
        assert_eq!(
            status.detail_lines.last().unwrap(),
            "also present: VERCEL_TOKEN, VERCEL_TEAM_ID"
        );
    }

    #[test]
    fn full_access_token() {
        let status = describe_with(checker(&[
            "VERCEL_TOKEN",
            "VERCEL_PROJECT_ID",
            "VERCEL_TEAM_ID",
        ]));
        assert!(status.ok);
        assert_eq!(
            status.label,
            "access token + project/team via VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID"
        );
        assert_eq!(
            status.detail_lines,
            vec![
                "mode: access token".to_string(),
                "active env: VERCEL_TOKEN, VERCEL_PROJECT_ID, VERCEL_TEAM_ID".to_string(),
            ]
        );
    }

    #[test]
    fn partial_access_token() {
        let status = describe_with(checker(&["VERCEL_TOKEN"]));
        assert!(!status.ok);
        assert_eq!(
            status.label,
            "partial access-token auth (missing VERCEL_PROJECT_ID, VERCEL_TEAM_ID)"
        );
        assert_eq!(
            status.detail_lines,
            vec![
                "mode: incomplete access token".to_string(),
                "present env: VERCEL_TOKEN".to_string(),
                "missing env: VERCEL_PROJECT_ID, VERCEL_TEAM_ID".to_string(),
                "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn not_configured() {
        let status = describe_with(checker(&[]));
        assert!(!status.ok);
        assert_eq!(status.label, "not configured");
        assert_eq!(
            status.detail_lines,
            vec![
                "recommended: set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID".to_string(),
                "development-only alternative: set VERCEL_OIDC_TOKEN".to_string(),
            ]
        );
    }

    #[test]
    fn present_empty_string_is_falsey() {
        // Mirrors Python bool(os.getenv(...)): empty value counts as absent.
        let var = "HERMES_TEST_VERCEL_EMPTY_VAR";
        unsafe { env::set_var(var, ""); }
        assert!(!present(var));
        unsafe { env::set_var(var, "x"); }
        assert!(present(var));
        unsafe { env::remove_var(var); }
        assert!(!present(var));
    }
}
