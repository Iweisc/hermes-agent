//! OSV malware check for MCP extension packages.
//!
//! Before launching an MCP server via `npx`/`uvx`, queries the OSV (Open Source
//! Vulnerabilities) API to check if the package has any known malware advisories
//! (`MAL-*` IDs). Regular CVEs are ignored — only confirmed malware is blocked.
//!
//! The API is free, public, and maintained by Google. Typical latency is ~300ms.
//! Fail-open: network errors allow the package to proceed.
//!
//! Inspired by Block/goose's extension malware check.
//!
//! This is a native Rust port of `tools/osv_check.py`.

use std::time::Duration;

use serde_json::{json, Value};

/// Default OSV query endpoint. Overridable via the `OSV_ENDPOINT` env var.
pub const DEFAULT_OSV_ENDPOINT: &str = "https://api.osv.dev/v1/query";

/// Request timeout for the OSV query.
pub const OSV_TIMEOUT: Duration = Duration::from_secs(10);

/// Resolve the OSV endpoint, honouring the `OSV_ENDPOINT` env var.
pub fn osv_endpoint() -> String {
    std::env::var("OSV_ENDPOINT").unwrap_or_else(|_| DEFAULT_OSV_ENDPOINT.to_string())
}

/// Check if an MCP server package has known malware advisories.
///
/// Inspects the `command` (e.g. `npx`, `uvx`) and `args` to infer the package
/// name and ecosystem. Queries the OSV API for `MAL-*` advisories.
///
/// Returns an error message string if malware is found, or `None` if the
/// package is clean/unknown. Returns `None` (allow) on network errors or
/// unrecognized commands (fail-open).
pub fn check_package_for_malware<S: AsRef<str>>(command: &str, args: &[S]) -> Option<String> {
    let ecosystem = match infer_ecosystem(command) {
        Some(e) => e,
        None => return None, // not npx/uvx — skip
    };

    let (package, version) = parse_package_from_args(args, ecosystem);
    let package = match package {
        Some(p) => p,
        None => return None,
    };

    let malware = match query_osv(&package, ecosystem, version.as_deref()) {
        Ok(m) => m,
        Err(exc) => {
            // Fail-open: network errors, timeouts, parse failures → allow
            log::debug!(
                "OSV check failed for {}/{} (allowing): {}",
                ecosystem.as_str(),
                package,
                exc
            );
            return None;
        }
    };

    if !malware.is_empty() {
        let ids = malware
            .iter()
            .take(3)
            .map(|m| m.id.clone())
            .collect::<Vec<_>>()
            .join(", ");
        let summaries = malware
            .iter()
            .take(3)
            .map(|m| {
                let s = m.summary.as_deref().unwrap_or(&m.id);
                truncate_chars(s, 100)
            })
            .collect::<Vec<_>>()
            .join("; ");
        return Some(format!(
            "BLOCKED: Package '{}' ({}) has known malware advisories: {}. Details: {}",
            package,
            ecosystem.as_str(),
            ids,
            summaries
        ));
    }
    None
}

/// Package ecosystem inferred from the launch command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ecosystem {
    Npm,
    PyPI,
}

impl Ecosystem {
    /// The OSV ecosystem string (`"npm"` or `"PyPI"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Ecosystem::Npm => "npm",
            Ecosystem::PyPI => "PyPI",
        }
    }
}

/// Infer the package ecosystem from the command name.
pub fn infer_ecosystem(command: &str) -> Option<Ecosystem> {
    let base = basename(command).to_lowercase();
    match base.as_str() {
        "npx" | "npx.cmd" => Some(Ecosystem::Npm),
        "uvx" | "uvx.cmd" | "pipx" => Some(Ecosystem::PyPI),
        _ => None,
    }
}

/// Return the final path component of `command` (handles `/` and `\`).
fn basename(command: &str) -> &str {
    let after_slash = command.rsplit('/').next().unwrap_or(command);
    after_slash.rsplit('\\').next().unwrap_or(after_slash)
}

/// Extract the package name and optional version from command args.
///
/// Returns `(package_name, version)` or `(None, None)` if not parseable.
pub fn parse_package_from_args<S: AsRef<str>>(
    args: &[S],
    ecosystem: Ecosystem,
) -> (Option<String>, Option<String>) {
    if args.is_empty() {
        return (None, None);
    }

    // Skip flags to find the package token.
    let mut package_token: Option<&str> = None;
    for arg in args {
        let arg = arg.as_ref();
        if arg.starts_with('-') {
            continue;
        }
        package_token = Some(arg);
        break;
    }

    let token = match package_token {
        Some(t) => t,
        None => return (None, None),
    };

    match ecosystem {
        Ecosystem::Npm => parse_npm_package(token),
        Ecosystem::PyPI => parse_pypi_package(token),
    }
}

/// Parse an npm package token: `@scope/name@version` or `name@version`.
pub fn parse_npm_package(token: &str) -> (Option<String>, Option<String>) {
    if token.starts_with('@') {
        // Scoped: @scope/name@version
        let re = regex::Regex::new(r"^(@[^/]+/[^@]+)(?:@(.+))?$").expect("valid regex");
        if let Some(caps) = re.captures(token) {
            let name = caps.get(1).map(|m| m.as_str().to_string());
            let version = caps.get(2).map(|m| m.as_str().to_string());
            return (name, version);
        }
        return (Some(token.to_string()), None);
    }
    // Unscoped: name@version
    if token.contains('@') {
        // rsplit on the last '@'
        if let Some(idx) = token.rfind('@') {
            let name = &token[..idx];
            let ver = &token[idx + 1..];
            let version = if ver != "latest" {
                Some(ver.to_string())
            } else {
                None
            };
            return (Some(name.to_string()), version);
        }
    }
    (Some(token.to_string()), None)
}

/// Parse a PyPI package token: `name==version` or `name[extras]==version`.
pub fn parse_pypi_package(token: &str) -> (Option<String>, Option<String>) {
    let re = regex::Regex::new(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$")
        .expect("valid regex");
    if let Some(caps) = re.captures(token) {
        let name = caps.get(1).map(|m| m.as_str().to_string());
        let version = caps.get(2).map(|m| m.as_str().to_string());
        return (name, version);
    }
    (Some(token.to_string()), None)
}

/// A malware advisory returned by OSV (only `MAL-*` IDs are kept).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalwareVuln {
    pub id: String,
    pub summary: Option<String>,
}

/// Query the OSV API for `MAL-*` advisories. Returns the list of malware vulns.
///
/// Mirrors the Python `_query_osv`: POSTs a JSON body with the package name and
/// ecosystem (plus version when known), then filters the `vulns` array for IDs
/// starting with `MAL-`.
pub fn query_osv(
    package: &str,
    ecosystem: Ecosystem,
    version: Option<&str>,
) -> Result<Vec<MalwareVuln>, String> {
    let mut payload = json!({
        "package": {
            "name": package,
            "ecosystem": ecosystem.as_str(),
        }
    });
    if let Some(v) = version {
        payload["version"] = Value::String(v.to_string());
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(OSV_TIMEOUT)
        .build()
        .map_err(|e| format!("client build error: {e}"))?;

    let resp = client
        .post(osv_endpoint())
        .header("Content-Type", "application/json")
        .header("User-Agent", "hermes-agent-osv-check/1.0")
        .body(serde_json::to_vec(&payload).map_err(|e| format!("serialize error: {e}"))?)
        .send()
        .map_err(|e| format!("request error: {e}"))?;

    let text = resp.text().map_err(|e| format!("read error: {e}"))?;
    let result: Value = serde_json::from_str(&text).map_err(|e| format!("parse error: {e}"))?;

    Ok(parse_malware_vulns(&result))
}

/// Extract `MAL-*` malware advisories from an OSV API response value.
pub fn parse_malware_vulns(result: &Value) -> Vec<MalwareVuln> {
    let vulns = match result.get("vulns").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };

    vulns
        .iter()
        .filter_map(|v| {
            let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("");
            if !id.starts_with("MAL-") {
                return None;
            }
            let summary = v
                .get("summary")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            Some(MalwareVuln {
                id: id.to_string(),
                summary,
            })
        })
        .collect()
}

/// Truncate a string to at most `max` characters (by Unicode scalar value).
///
/// Matches Python's `s[:100]` slice semantics closely enough for advisory text.
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn infer_ecosystem_npx() {
        assert_eq!(infer_ecosystem("npx"), Some(Ecosystem::Npm));
        assert_eq!(infer_ecosystem("/usr/local/bin/npx"), Some(Ecosystem::Npm));
        assert_eq!(infer_ecosystem("NPX"), Some(Ecosystem::Npm));
        assert_eq!(
            infer_ecosystem("C:\\tools\\npx.cmd"),
            Some(Ecosystem::Npm)
        );
    }

    #[test]
    fn infer_ecosystem_uvx_pipx() {
        assert_eq!(infer_ecosystem("uvx"), Some(Ecosystem::PyPI));
        assert_eq!(infer_ecosystem("uvx.cmd"), Some(Ecosystem::PyPI));
        assert_eq!(infer_ecosystem("pipx"), Some(Ecosystem::PyPI));
        assert_eq!(infer_ecosystem("/opt/bin/pipx"), Some(Ecosystem::PyPI));
    }

    #[test]
    fn infer_ecosystem_unknown() {
        assert_eq!(infer_ecosystem("node"), None);
        assert_eq!(infer_ecosystem("python"), None);
        assert_eq!(infer_ecosystem(""), None);
    }

    #[test]
    fn npm_unscoped() {
        assert_eq!(
            parse_npm_package("left-pad"),
            (Some("left-pad".to_string()), None)
        );
        assert_eq!(
            parse_npm_package("left-pad@1.3.0"),
            (Some("left-pad".to_string()), Some("1.3.0".to_string()))
        );
    }

    #[test]
    fn npm_latest_dropped() {
        assert_eq!(
            parse_npm_package("foo@latest"),
            (Some("foo".to_string()), None)
        );
    }

    #[test]
    fn npm_scoped() {
        assert_eq!(
            parse_npm_package("@scope/name"),
            (Some("@scope/name".to_string()), None)
        );
        assert_eq!(
            parse_npm_package("@scope/name@2.0.0"),
            (Some("@scope/name".to_string()), Some("2.0.0".to_string()))
        );
        assert_eq!(
            parse_npm_package("@modelcontextprotocol/server-filesystem@0.1.0"),
            (
                Some("@modelcontextprotocol/server-filesystem".to_string()),
                Some("0.1.0".to_string())
            )
        );
    }

    #[test]
    fn pypi_plain() {
        assert_eq!(
            parse_pypi_package("requests"),
            (Some("requests".to_string()), None)
        );
    }

    #[test]
    fn pypi_versioned() {
        assert_eq!(
            parse_pypi_package("requests==2.31.0"),
            (Some("requests".to_string()), Some("2.31.0".to_string()))
        );
    }

    #[test]
    fn pypi_extras() {
        assert_eq!(
            parse_pypi_package("package[extra1,extra2]==1.0.0"),
            (Some("package".to_string()), Some("1.0.0".to_string()))
        );
        assert_eq!(
            parse_pypi_package("package[extra]"),
            (Some("package".to_string()), None)
        );
    }

    #[test]
    fn parse_args_skips_flags() {
        let args = vec!["-y", "--quiet", "express@4"];
        let (pkg, ver) = parse_package_from_args(&args, Ecosystem::Npm);
        assert_eq!(pkg, Some("express".to_string()));
        assert_eq!(ver, Some("4".to_string()));
    }

    #[test]
    fn parse_args_empty() {
        let args: Vec<&str> = vec![];
        assert_eq!(
            parse_package_from_args(&args, Ecosystem::Npm),
            (None, None)
        );
    }

    #[test]
    fn parse_args_only_flags() {
        let args = vec!["-y", "--help"];
        assert_eq!(
            parse_package_from_args(&args, Ecosystem::Npm),
            (None, None)
        );
    }

    #[test]
    fn malware_filter_keeps_only_mal_ids() {
        let resp = json!({
            "vulns": [
                {"id": "CVE-2021-1234", "summary": "a regular cve"},
                {"id": "MAL-2024-0001", "summary": "evil package"},
                {"id": "GHSA-xxxx", "summary": "advisory"},
                {"id": "MAL-2024-0002"}
            ]
        });
        let vulns = parse_malware_vulns(&resp);
        assert_eq!(vulns.len(), 2);
        assert_eq!(vulns[0].id, "MAL-2024-0001");
        assert_eq!(vulns[0].summary.as_deref(), Some("evil package"));
        assert_eq!(vulns[1].id, "MAL-2024-0002");
        assert_eq!(vulns[1].summary, None);
    }

    #[test]
    fn malware_filter_no_vulns_field() {
        let resp = json!({});
        assert!(parse_malware_vulns(&resp).is_empty());
        let resp = json!({"vulns": []});
        assert!(parse_malware_vulns(&resp).is_empty());
    }

    #[test]
    fn unknown_command_allows() {
        let args = vec!["something"];
        assert_eq!(check_package_for_malware("node", &args), None);
    }

    #[test]
    fn truncate_summary() {
        let long = "x".repeat(200);
        assert_eq!(truncate_chars(&long, 100).chars().count(), 100);
        assert_eq!(truncate_chars("short", 100), "short");
    }

    #[test]
    fn endpoint_default() {
        // Without the env var set this should be the default. We don't mutate
        // process env here to avoid cross-test interference, just check default.
        assert_eq!(DEFAULT_OSV_ENDPOINT, "https://api.osv.dev/v1/query");
    }
}
