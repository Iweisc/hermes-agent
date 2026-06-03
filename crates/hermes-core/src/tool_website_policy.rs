//! Website access policy helpers for URL-capable tools.
//!
//! This module loads a user-managed website blocklist from
//! `~/.hermes/config.yaml` and optional shared list files. It is intentionally
//! lightweight so web/browser tools can enforce URL policy without pulling in
//! the heavier CLI config stack.
//!
//! Policy is cached in memory with a short TTL so config changes take effect
//! quickly without re-reading the file on every URL check.
//!
//! Faithful port of `tools/website_policy.py`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Map, Value};
use url::Url;

use crate::mod_hermes_constants::get_hermes_home;

/// Cache TTL: avoids re-reading config.yaml on every URL check (a web_crawl
/// with 50 pages would otherwise mean 51 YAML parses).
const CACHE_TTL_SECONDS: f64 = 30.0;

/// Default website blocklist policy values (`enabled`, `domains`,
/// `shared_files`). Mirrors `_DEFAULT_WEBSITE_BLOCKLIST` in Python.
fn default_website_blocklist() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("enabled".to_string(), Value::Bool(false));
    m.insert("domains".to_string(), Value::Array(Vec::new()));
    m.insert("shared_files".to_string(), Value::Array(Vec::new()));
    m
}

/// Raised when a website policy file is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebsitePolicyError(pub String);

impl std::fmt::Display for WebsitePolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for WebsitePolicyError {}

/// A single normalized blocklist rule and its source ("config" or a file path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRule {
    pub pattern: String,
    pub source: String,
}

/// Parsed website blocklist policy: whether enforcement is enabled plus the
/// list of normalized rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebsitePolicy {
    pub enabled: bool,
    pub rules: Vec<PolicyRule>,
}

impl WebsitePolicy {
    /// JSON view matching the Python dict shape `{"enabled": ..., "rules": [...]}`.
    pub fn to_json(&self) -> Value {
        let rules: Vec<Value> = self
            .rules
            .iter()
            .map(|r| json!({"pattern": r.pattern, "source": r.source}))
            .collect();
        json!({"enabled": self.enabled, "rules": rules})
    }
}

/// Block metadata returned when a URL is denied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockResult {
    pub url: String,
    pub host: String,
    pub rule: String,
    pub source: String,
    pub message: String,
}

impl BlockResult {
    /// JSON view matching the Python dict returned by `check_website_access`.
    pub fn to_json(&self) -> Value {
        json!({
            "url": self.url,
            "host": self.host,
            "rule": self.rule,
            "source": self.source,
            "message": self.message,
        })
    }
}

struct CacheState {
    policy: Option<WebsitePolicy>,
    path: Option<String>,
    time: Option<Instant>,
}

static CACHE: Mutex<CacheState> = Mutex::new(CacheState {
    policy: None,
    path: None,
    time: None,
});

fn get_default_config_path() -> PathBuf {
    get_hermes_home().join("config.yaml")
}

fn normalize_host(host: &str) -> String {
    host.trim().to_lowercase().trim_end_matches('.').to_string()
}

/// Normalize a single rule string into a bare host pattern, or `None` if the
/// input is empty, a comment, or not a usable rule.
fn normalize_rule(rule: &Value) -> Option<String> {
    let s = rule.as_str()?;
    let value = s.trim().to_lowercase();
    if value.is_empty() || value.starts_with('#') {
        return None;
    }

    let mut value = if value.contains("://") {
        // urlparse(value): netloc or path.
        match Url::parse(&value) {
            Ok(parsed) => {
                let netloc = url_netloc(&parsed);
                if !netloc.is_empty() {
                    netloc
                } else {
                    // path component
                    parsed.path().to_string()
                }
            }
            Err(_) => value.clone(),
        }
    } else {
        value.clone()
    };

    // value.split("/", 1)[0]
    if let Some(idx) = value.find('/') {
        value = value[..idx].to_string();
    }
    let mut value = value.trim().trim_end_matches('.').to_string();

    if let Some(stripped) = value.strip_prefix("www.") {
        value = stripped.to_string();
    }

    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Reconstruct the Python `urlparse(...).netloc` form (host[:port], with
/// optional userinfo) from a parsed `Url`.
fn url_netloc(parsed: &Url) -> String {
    let host = match parsed.host_str() {
        Some(h) => h,
        None => return String::new(),
    };
    let mut netloc = String::new();
    let user = parsed.username();
    if !user.is_empty() || parsed.password().is_some() {
        netloc.push_str(user);
        if let Some(pw) = parsed.password() {
            netloc.push(':');
            netloc.push_str(pw);
        }
        netloc.push('@');
    }
    netloc.push_str(host);
    if let Some(port) = parsed.port() {
        netloc.push(':');
        netloc.push_str(&port.to_string());
    }
    netloc
}

/// Load rules from a shared blocklist file.
///
/// Missing or unreadable files log a warning and return an empty list rather
/// than raising — a bad file path should not disable all web tools.
fn iter_blocklist_file_rules(path: &Path) -> Vec<String> {
    let raw = match std::fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(exc) => {
                log::warn!(
                    "Failed to read shared blocklist file {} (skipping): {}",
                    path.display(),
                    exc
                );
                return Vec::new();
            }
        },
        Err(exc) => {
            if exc.kind() == std::io::ErrorKind::NotFound {
                log::warn!(
                    "Shared blocklist file not found (skipping): {}",
                    path.display()
                );
            } else {
                log::warn!(
                    "Failed to read shared blocklist file {} (skipping): {}",
                    path.display(),
                    exc
                );
            }
            return Vec::new();
        }
    };

    let mut rules: Vec<String> = Vec::new();
    for line in raw.lines() {
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }
        if let Some(normalized) = normalize_rule(&Value::String(stripped.to_string())) {
            rules.push(normalized);
        }
    }
    rules
}

/// Load the raw `security.website_blocklist` mapping merged over defaults.
fn load_policy_config(config_path: &Path) -> Result<Map<String, Value>, WebsitePolicyError> {
    if !config_path.exists() {
        return Ok(default_website_blocklist());
    }

    let text = match std::fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(exc) => {
            return Err(WebsitePolicyError(format!(
                "Failed to read config file {}: {}",
                config_path.display(),
                exc
            )));
        }
    };

    let config: Value = match serde_yaml::from_str(&text) {
        Ok(Value::Null) => Value::Object(Map::new()),
        Ok(v) => v,
        Err(exc) => {
            return Err(WebsitePolicyError(format!(
                "Invalid config YAML at {}: {}",
                config_path.display(),
                exc
            )));
        }
    };

    let config = match config {
        Value::Object(m) => m,
        _ => {
            return Err(WebsitePolicyError("config root must be a mapping".to_string()));
        }
    };

    let security = match config.get("security") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return Err(WebsitePolicyError("security must be a mapping".to_string()));
        }
    };

    let website_blocklist = match security.get("website_blocklist") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return Err(WebsitePolicyError(
                "security.website_blocklist must be a mapping".to_string(),
            ));
        }
    };

    let mut policy = default_website_blocklist();
    for (k, v) in website_blocklist {
        policy.insert(k, v);
    }
    Ok(policy)
}

/// Load and return the parsed website blocklist policy.
///
/// Results are cached for [`CACHE_TTL_SECONDS`] to avoid re-reading config.yaml
/// on every URL check. Pass an explicit `config_path` to bypass the cache (used
/// by tests).
pub fn load_website_blocklist(
    config_path: Option<&Path>,
) -> Result<WebsitePolicy, WebsitePolicyError> {
    let resolved_path = match config_path {
        Some(p) => p.to_string_lossy().to_string(),
        None => "__default__".to_string(),
    };
    let now = Instant::now();

    if config_path.is_none() {
        let cache = CACHE.lock().unwrap();
        if let (Some(policy), Some(path), Some(time)) =
            (&cache.policy, &cache.path, &cache.time)
        {
            if path == &resolved_path && now.duration_since(*time).as_secs_f64() < CACHE_TTL_SECONDS
            {
                return Ok(policy.clone());
            }
        }
    }

    let default_path = get_default_config_path();
    let effective_path: PathBuf = match config_path {
        Some(p) => p.to_path_buf(),
        None => default_path.clone(),
    };

    let policy = load_policy_config(&effective_path)?;

    let raw_domains: Vec<Value> = match policy.get("domains") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => {
            return Err(WebsitePolicyError(
                "security.website_blocklist.domains must be a list".to_string(),
            ));
        }
    };

    let raw_shared_files: Vec<Value> = match policy.get("shared_files") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => {
            return Err(WebsitePolicyError(
                "security.website_blocklist.shared_files must be a list".to_string(),
            ));
        }
    };

    let enabled = match policy.get("enabled") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => {
            return Err(WebsitePolicyError(
                "security.website_blocklist.enabled must be a boolean".to_string(),
            ));
        }
    };

    let mut rules: Vec<PolicyRule> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    for raw_rule in &raw_domains {
        if let Some(normalized) = normalize_rule(raw_rule) {
            let key = ("config".to_string(), normalized.clone());
            if !seen.contains(&key) {
                rules.push(PolicyRule {
                    pattern: normalized,
                    source: "config".to_string(),
                });
                seen.insert(key);
            }
        }
    }

    for shared_file in &raw_shared_files {
        let s = match shared_file.as_str() {
            Some(s) if !s.trim().is_empty() => s,
            _ => continue,
        };
        let mut path = expanduser(s);
        if !path.is_absolute() {
            // (get_hermes_home() / path).resolve()
            let joined = get_hermes_home().join(&path);
            path = std::fs::canonicalize(&joined).unwrap_or(joined);
        }
        let path_str = path.to_string_lossy().to_string();
        for normalized in iter_blocklist_file_rules(&path) {
            let key = (path_str.clone(), normalized.clone());
            if seen.contains(&key) {
                continue;
            }
            rules.push(PolicyRule {
                pattern: normalized,
                source: path_str.clone(),
            });
            seen.insert(key);
        }
    }

    let result = WebsitePolicy { enabled, rules };

    // Cache the result (only for the default path — explicit paths are tests).
    if effective_path == default_path {
        let mut cache = CACHE.lock().unwrap();
        cache.policy = Some(result.clone());
        cache.path = Some("__default__".to_string());
        cache.time = Some(now);
    }

    Ok(result)
}

/// Expand a leading `~` to the user's home directory, mirroring
/// `Path(...).expanduser()`.
fn expanduser(s: &str) -> PathBuf {
    if s == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    } else if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(s)
}

/// Force the next [`check_website_access`] call to re-read config.
pub fn invalidate_cache() {
    let mut cache = CACHE.lock().unwrap();
    cache.policy = None;
}

fn match_host_against_rule(host: &str, pattern: &str) -> bool {
    if host.is_empty() || pattern.is_empty() {
        return false;
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        // fnmatch.fnmatch(host, pattern) where pattern == "*.<suffix>".
        return fnmatch(host, pattern, suffix);
    }
    host == pattern || host.ends_with(&format!(".{pattern}"))
}

/// Minimal fnmatch supporting `*` and `?` wildcards, case-insensitive on the
/// already-lowercased inputs. Implemented specifically for `*.suffix` patterns
/// but kept general to mirror Python's `fnmatch.fnmatch`.
fn fnmatch(name: &str, pattern: &str, _suffix: &str) -> bool {
    glob_match(pattern.as_bytes(), name.as_bytes())
}

/// Recursive glob matcher for `*` (any sequence) and `?` (single char).
fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    // Iterative with backtracking to avoid pathological recursion.
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star_p, mut star_t): (Option<usize>, usize) = (None, 0);

    while t < text.len() {
        if p < pattern.len() && (pattern[p] == b'?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star_p = Some(p);
            star_t = t;
            p += 1;
        } else if let Some(sp) = star_p {
            p = sp + 1;
            star_t += 1;
            t = star_t;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

/// Extract a normalized host from a URL-ish string. Falls back to parsing a
/// schemeless input as `//<url>` to recover a netloc, mirroring the Python.
fn extract_host_from_urlish(url: &str) -> String {
    if let Ok(parsed) = Url::parse(url) {
        let host = normalize_host(
            &parsed
                .host_str()
                .map(|h| h.to_string())
                .unwrap_or_else(|| url_netloc(&parsed)),
        );
        if !host.is_empty() {
            return host;
        }
    }

    if !url.contains("://") {
        let schemeless = format!("//{url}");
        // Python urlparse("//host/...") yields a netloc. url::Url needs a base
        // or scheme, so emulate by extracting the netloc segment manually.
        if let Some(host) = parse_netloc_only(&schemeless) {
            let host = normalize_host(&host);
            if !host.is_empty() {
                return host;
            }
        }
    }

    String::new()
}

/// Emulate `urlparse("//netloc/path").hostname / .netloc` extraction.
fn parse_netloc_only(s: &str) -> Option<String> {
    let rest = s.strip_prefix("//")?;
    // netloc ends at the first '/', '?' or '#'.
    let end = rest
        .find(|c| c == '/' || c == '?' || c == '#')
        .unwrap_or(rest.len());
    let netloc = &rest[..end];
    if netloc.is_empty() {
        return None;
    }
    // hostname: strip userinfo and port.
    let after_user = match netloc.rfind('@') {
        Some(idx) => &netloc[idx + 1..],
        None => netloc,
    };
    // Handle IPv6 bracketed host.
    let host = if let Some(stripped) = after_user.strip_prefix('[') {
        match stripped.find(']') {
            Some(idx) => &stripped[..idx],
            None => after_user,
        }
    } else {
        match after_user.find(':') {
            Some(idx) => &after_user[..idx],
            None => after_user,
        }
    };
    Some(host.to_string())
}

/// Check whether a URL is allowed by the website blocklist policy.
///
/// Returns `None` if access is allowed, or a [`BlockResult`] with block metadata
/// if blocked.
///
/// Never raises on policy errors when `config_path` is `None` — logs a warning
/// and returns `None` (fail-open) so a config typo doesn't break all web tools.
/// Pass `config_path` explicitly (tests) to get strict error propagation.
pub fn check_website_access(
    url: &str,
    config_path: Option<&Path>,
) -> Result<Option<BlockResult>, WebsitePolicyError> {
    // Fast path: if no explicit config_path and the cached policy is disabled
    // or empty, skip all work.
    if config_path.is_none() {
        let cache = CACHE.lock().unwrap();
        if let Some(policy) = &cache.policy {
            if !policy.enabled {
                return Ok(None);
            }
        }
    }

    let host = extract_host_from_urlish(url);
    if host.is_empty() {
        return Ok(None);
    }

    let policy = match load_website_blocklist(config_path) {
        Ok(p) => p,
        Err(exc) => {
            if config_path.is_some() {
                return Err(exc); // Tests pass explicit paths — propagate.
            }
            log::warn!("Website policy config error (failing open): {}", exc);
            return Ok(None);
        }
    };

    if !policy.enabled {
        return Ok(None);
    }

    for rule in &policy.rules {
        let pattern = &rule.pattern;
        if match_host_against_rule(&host, pattern) {
            log::info!(
                "Blocked URL {} — matched rule '{}' from {}",
                url,
                pattern,
                rule.source
            );
            let message = format!(
                "Blocked by website policy: '{host}' matched rule '{pattern}' from {}",
                rule.source
            );
            return Ok(Some(BlockResult {
                url: url.to_string(),
                host,
                rule: pattern.clone(),
                source: rule.source.clone(),
                message,
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_config(body: &str) -> tempfilePath {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "hermes_wp_{}_{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        tempfilePath(path)
    }

    struct tempfilePath(PathBuf);
    impl Drop for tempfilePath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn test_normalize_rule_basic() {
        assert_eq!(
            normalize_rule(&Value::String("Example.COM".to_string())),
            Some("example.com".to_string())
        );
        assert_eq!(
            normalize_rule(&Value::String("www.example.com".to_string())),
            Some("example.com".to_string())
        );
        assert_eq!(
            normalize_rule(&Value::String("https://www.example.com/path".to_string())),
            Some("example.com".to_string())
        );
        assert_eq!(
            normalize_rule(&Value::String("  # comment".to_string())),
            None
        );
        assert_eq!(normalize_rule(&Value::String("".to_string())), None);
        assert_eq!(normalize_rule(&Value::Bool(true)), None);
        assert_eq!(
            normalize_rule(&Value::String("foo.com.".to_string())),
            Some("foo.com".to_string())
        );
    }

    #[test]
    fn test_extract_host() {
        assert_eq!(extract_host_from_urlish("https://Example.com/x"), "example.com");
        assert_eq!(extract_host_from_urlish("example.com/path"), "example.com");
        assert_eq!(extract_host_from_urlish("example.com"), "example.com");
        assert_eq!(extract_host_from_urlish("user:pw@example.com:8080/p"), "example.com");
        assert_eq!(extract_host_from_urlish(""), "");
    }

    #[test]
    fn test_match_host_against_rule() {
        assert!(match_host_against_rule("example.com", "example.com"));
        assert!(match_host_against_rule("sub.example.com", "example.com"));
        assert!(!match_host_against_rule("notexample.com", "example.com"));
        assert!(match_host_against_rule("sub.example.com", "*.example.com"));
        assert!(!match_host_against_rule("example.com", "*.example.com"));
        assert!(!match_host_against_rule("", "example.com"));
        assert!(!match_host_against_rule("example.com", ""));
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match(b"*.example.com", b"sub.example.com"));
        assert!(glob_match(b"*.example.com", b"a.b.example.com"));
        assert!(!glob_match(b"*.example.com", b"example.com"));
        assert!(glob_match(b"a?c", b"abc"));
        assert!(!glob_match(b"a?c", b"ac"));
    }

    #[test]
    fn test_load_disabled_when_missing() {
        let p = PathBuf::from("/nonexistent/path/hermes_test_xyz.yaml");
        let policy = load_website_blocklist(Some(&p)).unwrap();
        assert!(!policy.enabled);
        assert!(policy.rules.is_empty());
    }

    #[test]
    fn test_load_with_domains() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - Example.com\n      - www.foo.com\n      - example.com\n",
        );
        let policy = load_website_blocklist(Some(&cfg.0)).unwrap();
        assert!(policy.enabled);
        // dedupe: example.com appears once, foo.com once
        let patterns: Vec<&str> = policy.rules.iter().map(|r| r.pattern.as_str()).collect();
        assert_eq!(patterns, vec!["example.com", "foo.com"]);
        for r in &policy.rules {
            assert_eq!(r.source, "config");
        }
    }

    #[test]
    fn test_check_blocks_and_allows() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - blocked.com\n",
        );
        let blocked = check_website_access("https://sub.blocked.com/path", Some(&cfg.0)).unwrap();
        assert!(blocked.is_some());
        let b = blocked.unwrap();
        assert_eq!(b.host, "sub.blocked.com");
        assert_eq!(b.rule, "blocked.com");
        assert_eq!(b.source, "config");
        assert!(b.message.contains("Blocked by website policy"));

        let allowed = check_website_access("https://allowed.com", Some(&cfg.0)).unwrap();
        assert!(allowed.is_none());
    }

    #[test]
    fn test_check_disabled_allows_all() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    enabled: false\n    domains:\n      - blocked.com\n",
        );
        let res = check_website_access("https://blocked.com", Some(&cfg.0)).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn test_invalid_root_raises_with_explicit_path() {
        let cfg = write_config("- just\n- a\n- list\n");
        let err = load_website_blocklist(Some(&cfg.0));
        assert!(err.is_err());
        assert_eq!(err.unwrap_err().0, "config root must be a mapping");
    }

    #[test]
    fn test_domains_not_list_raises() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    enabled: true\n    domains: nope\n",
        );
        let err = load_website_blocklist(Some(&cfg.0));
        assert!(err.is_err());
        assert!(err.unwrap_err().0.contains("domains must be a list"));
    }

    #[test]
    fn test_enabled_not_bool_raises() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    enabled: yesplease\n",
        );
        // yaml 'yesplease' is a string -> error
        let err = load_website_blocklist(Some(&cfg.0));
        assert!(err.is_err());
        assert!(err.unwrap_err().0.contains("enabled must be a boolean"));
    }

    #[test]
    fn test_default_enabled_true_when_omitted() {
        let cfg = write_config(
            "security:\n  website_blocklist:\n    domains:\n      - x.com\n",
        );
        let policy = load_website_blocklist(Some(&cfg.0)).unwrap();
        assert!(policy.enabled);
    }

    #[test]
    fn test_invalidate_cache() {
        invalidate_cache();
        let cache = CACHE.lock().unwrap();
        assert!(cache.policy.is_none());
    }

    #[test]
    fn test_to_json_shape() {
        let policy = WebsitePolicy {
            enabled: true,
            rules: vec![PolicyRule {
                pattern: "x.com".to_string(),
                source: "config".to_string(),
            }],
        };
        let v = policy.to_json();
        assert_eq!(v["enabled"], json!(true));
        assert_eq!(v["rules"][0]["pattern"], json!("x.com"));
        assert_eq!(v["rules"][0]["source"], json!("config"));
    }
}
