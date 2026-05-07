use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use reqwest::Url;
use serde_json::{Value, json};
use serde_yaml::{Mapping, Value as YamlValue};

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_SEARCH_LIMIT: i64 = 5;
const MAX_SEARCH_LIMIT: i64 = 100;
const MAX_EXTRACT_URLS: usize = 5;
pub(crate) const BLOCKED_URL_SECRET_ERROR: &str = "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.";
pub(crate) const PRIVATE_URL_ERROR: &str =
    "Blocked: URL targets a private or internal network address";
pub(crate) const INVALID_URL_ERROR: &str = "Blocked: URL must be a valid http or https URL";
const BLOCKED_HOSTNAMES: &[&str] = &["metadata.google.internal", "metadata.goog"];
const SENSITIVE_QUERY_PARAMS: &[&str] = &[
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "password",
    "auth",
    "jwt",
    "session",
    "secret",
    "key",
    "code",
    "signature",
    "x-amz-signature",
];
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "ghp_",
    "github_pat_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "xox",
    "AIza",
    "pplx-",
    "fal_",
    "fc-",
    "bb_live_",
    "AKIA",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "SG.",
    "hf_",
    "r8_",
    "npm_",
    "pypi-",
    "dop_v1_",
    "doo_v1_",
    "am_",
    "sk_",
    "tvly-",
    "exa_",
    "gsk_",
    "syt_",
    "retaindb_",
    "hsk-",
    "mem0_",
    "brv_",
];

#[derive(Debug, Clone, Default)]
struct WebsitePolicy {
    enabled: bool,
    rules: Vec<WebsiteRule>,
}

#[derive(Debug, Clone)]
struct WebsiteRule {
    pattern: String,
    source: String,
}

#[derive(Debug, Clone)]
pub(crate) struct WebsiteBlock {
    pub(crate) host: String,
    pub(crate) rule: String,
    pub(crate) source: String,
}

pub fn web_tools_available() -> bool {
    Command::new("firecrawl")
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn web_search_schema() -> Value {
    json!({
        "name": "web_search",
        "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to Firecrawl, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SEARCH_LIMIT,
                    "default": DEFAULT_SEARCH_LIMIT,
                    "description": "Maximum number of search results to return"
                }
            },
            "required": ["query"]
        }
    })
}

pub fn web_extract_schema() -> Value {
    json!({
        "name": "web_extract",
        "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs by passing the PDF link directly. If a URL fails or is blocked, the result includes a per-URL error.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": { "type": "string" },
                    "maxItems": MAX_EXTRACT_URLS,
                    "description": "List of URLs to extract content from"
                }
            },
            "required": ["urls"]
        }
    })
}

pub fn handle_web_search(args: &Value, runtime: &ToolRuntime) -> String {
    let query = match required_non_empty_string(args, "query") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let limit = match bounded_integer(args, "limit", DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    let payload = match run_firecrawl(
        &[
            "search".to_string(),
            "--json".to_string(),
            "--limit".to_string(),
            limit.to_string(),
            query,
        ],
        runtime.cwd(),
    ) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    let results = normalize_search_results(&payload);
    tool_result(json!({
        "success": true,
        "data": {
            "web": results,
        }
    }))
}

pub fn handle_web_extract(args: &Value, runtime: &ToolRuntime) -> String {
    let urls = match required_string_array(args, "urls", MAX_EXTRACT_URLS) {
        Ok(values) => values,
        Err(error) => return tool_error(error),
    };
    if urls.is_empty() {
        return tool_error("urls must contain at least one URL");
    }

    if urls.iter().any(|url| contains_embedded_secret(url)) {
        return tool_result(json!({
            "success": false,
            "error": BLOCKED_URL_SECRET_ERROR,
        }));
    }

    let mut results = Vec::new();
    for url in &urls {
        let parsed = match parse_http_url(url) {
            Ok(parsed) => parsed,
            Err(error) => {
                results.push(blocked_result(url, error, None));
                continue;
            }
        };

        if !is_safe_url(&parsed, runtime) {
            results.push(blocked_result(url, PRIVATE_URL_ERROR, None));
            continue;
        }

        if let Some(blocked) = check_website_access(&parsed, runtime) {
            results.push(blocked_result(
                url,
                format!(
                    "Blocked by website policy for host '{}' (rule: {})",
                    blocked.host, blocked.rule
                ),
                Some(json!({
                    "host": blocked.host,
                    "rule": blocked.rule,
                    "source": blocked.source,
                })),
            ));
            continue;
        }

        let payload = match run_firecrawl(
            &[
                "scrape".to_string(),
                "--json".to_string(),
                "--format".to_string(),
                "markdown".to_string(),
                url.clone(),
            ],
            runtime.cwd(),
        ) {
            Ok(value) => value,
            Err(error) => {
                results.push(blocked_result(url, error, None));
                continue;
            }
        };

        let normalized = normalize_extract_results(&payload, std::slice::from_ref(url));
        if normalized.is_empty() {
            results.push(blocked_result(url, "No content returned", None));
        } else {
            results.extend(normalized);
        }
    }

    if results.is_empty() {
        return tool_error("Content was inaccessible or not found");
    }
    tool_result(json!({ "results": results }))
}

fn run_firecrawl(args: &[String], cwd: &Path) -> Result<Value, String> {
    let output = Command::new("firecrawl")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("starting firecrawl failed: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("firecrawl failed: {detail}"));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| "firecrawl returned non-UTF-8 output".to_string())?;
    serde_json::from_str::<Value>(stdout.trim())
        .map_err(|error| format!("decoding firecrawl JSON failed: {error}"))
}

fn normalize_search_results(payload: &Value) -> Vec<Value> {
    let items = payload
        .get("data")
        .and_then(|value| value.get("web"))
        .and_then(Value::as_array)
        .or_else(|| payload.get("web").and_then(Value::as_array))
        .or_else(|| payload.get("results").and_then(Value::as_array))
        .or_else(|| payload.as_array())
        .cloned()
        .unwrap_or_default();

    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let title = value_string(&item, &["title", "name"]);
            let url = value_string(&item, &["url", "sourceURL", "link"]);
            let description = value_string(&item, &["description", "snippet", "content"]);
            let position = item
                .get("position")
                .and_then(Value::as_i64)
                .unwrap_or((index + 1) as i64);
            json!({
                "title": title,
                "url": url,
                "description": description,
                "position": position,
            })
        })
        .collect()
}

fn normalize_extract_results(payload: &Value, input_urls: &[String]) -> Vec<Value> {
    let items = extract_items(payload);
    let normalized = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let metadata = item.get("metadata").cloned().unwrap_or_else(|| json!({}));
            let url = value_string_with_fallback(
                &item,
                &metadata,
                &["url", "sourceURL"],
                input_urls.get(index),
            );
            let title = value_string_with_fallback(&item, &metadata, &["title"], None);
            let content = item
                .get("content")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    item.get("markdown")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
                .or_else(|| {
                    item.get("html")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
                .unwrap_or_default();
            let error = item
                .get("error")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    item.get("message")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                });
            json!({
                "url": url,
                "title": title,
                "content": content,
                "error": error,
            })
        })
        .collect::<Vec<_>>();

    if !normalized.is_empty() {
        return normalized;
    }

    input_urls
        .iter()
        .map(|url| {
            json!({
                "url": url,
                "title": "",
                "content": "",
                "error": "No content returned",
            })
        })
        .collect()
}

fn extract_items(payload: &Value) -> Vec<Value> {
    if let Some(items) = payload.get("results").and_then(Value::as_array) {
        return items.clone();
    }
    if let Some(data) = payload.get("data") {
        if let Some(items) = data.as_array() {
            return items.clone();
        }
        if data.is_object() {
            return vec![data.clone()];
        }
    }
    if let Some(items) = payload.as_array() {
        return items.clone();
    }
    if payload.is_object() {
        return vec![payload.clone()];
    }
    Vec::new()
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn bounded_integer(
    args: &Value,
    key: &str,
    default: i64,
    min: i64,
    max: i64,
) -> Result<i64, String> {
    let value = match args.get(key) {
        Some(value) => value
            .as_i64()
            .ok_or_else(|| format!("{key} must be an integer"))?,
        None => default,
    };
    if value < min || value > max {
        return Err(format!("{key} must be between {min} and {max}"));
    }
    Ok(value)
}

fn required_string_array(args: &Value, key: &str, max_items: usize) -> Result<Vec<String>, String> {
    let values = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array of strings"))?;
    if values.len() > max_items {
        return Err(format!("{key} supports at most {max_items} URLs"));
    }
    let mut items = Vec::new();
    for value in values {
        let Some(text) = value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Err(format!("{key} must contain only non-empty strings"));
        };
        items.push(text.to_string());
    }
    Ok(items)
}

fn value_string(item: &Value, keys: &[&str]) -> String {
    for key in keys {
        if let Some(value) = item.get(*key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    String::new()
}

fn value_string_with_fallback(
    item: &Value,
    metadata: &Value,
    keys: &[&str],
    fallback: Option<&String>,
) -> String {
    let direct = value_string(item, keys);
    if !direct.is_empty() {
        return direct;
    }
    let nested = value_string(metadata, keys);
    if !nested.is_empty() {
        return nested;
    }
    fallback.cloned().unwrap_or_default()
}

pub(crate) fn parse_http_url(url: &str) -> Result<Url, &'static str> {
    let parsed = Url::parse(url).map_err(|_| INVALID_URL_ERROR)?;
    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        _ => Err(INVALID_URL_ERROR),
    }
}

pub(crate) fn contains_embedded_secret(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return true;
    }

    if SECRET_PREFIXES.iter().any(|prefix| url.contains(prefix)) {
        return true;
    }

    parsed.query_pairs().any(|(key, value)| {
        SENSITIVE_QUERY_PARAMS
            .iter()
            .any(|sensitive| key.eq_ignore_ascii_case(sensitive))
            || SECRET_PREFIXES.iter().any(|prefix| value.contains(prefix))
    })
}

pub(crate) fn is_safe_url(url: &Url, runtime: &ToolRuntime) -> bool {
    let Some(host_text) = url.host_str() else {
        return false;
    };
    let host = normalize_host(host_text);
    if BLOCKED_HOSTNAMES.iter().any(|blocked| host == *blocked) {
        return false;
    }

    let allow_private = allow_private_urls(runtime);
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_metadata_ip(ip) {
            return false;
        }
        return allow_private || !is_blocked_ip(ip);
    }

    if allow_private {
        return true;
    }
    host != "localhost" && !host.ends_with(".localhost") && !host.ends_with(".local")
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.is_multicast()
                || is_cgnat(ip)
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

fn is_cgnat(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

fn is_metadata_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => matches!(
            ip,
            addr if addr == Ipv4Addr::new(169, 254, 169, 254)
                || addr == Ipv4Addr::new(169, 254, 170, 2)
                || addr == Ipv4Addr::new(169, 254, 169, 253)
                || addr == Ipv4Addr::new(100, 100, 100, 200)
        ),
        IpAddr::V6(ip) => ip == Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254),
    }
}

fn blocked_result(url: &str, error: impl Into<String>, blocked_by_policy: Option<Value>) -> Value {
    let mut result = json!({
        "url": url,
        "title": "",
        "content": "",
        "error": error.into(),
    });
    if let Some(blocked_by_policy) = blocked_by_policy
        && let Some(object) = result.as_object_mut()
    {
        object.insert("blocked_by_policy".to_string(), blocked_by_policy);
    }
    result
}

fn allow_private_urls(runtime: &ToolRuntime) -> bool {
    if let Ok(value) = std::env::var("HERMES_ALLOW_PRIVATE_URLS") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
        if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
            return false;
        }
    }

    let path = runtime.hermes_home().join("config.yaml");
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(root) = serde_yaml::from_str::<YamlValue>(&contents) else {
        return false;
    };
    let Some(security) = root
        .as_mapping()
        .and_then(|mapping| yaml_mapping(mapping, "security"))
    else {
        return false;
    };
    yaml_bool(security, "allow_private_urls").unwrap_or(false)
        || root
            .as_mapping()
            .and_then(|mapping| yaml_mapping(mapping, "browser"))
            .and_then(|mapping| yaml_bool(mapping, "allow_private_urls"))
            .unwrap_or(false)
}

pub(crate) fn check_website_access(url: &Url, runtime: &ToolRuntime) -> Option<WebsiteBlock> {
    let host = normalize_host(url.host_str()?);
    let policy = load_website_policy(runtime.hermes_home());
    if !policy.enabled {
        return None;
    }

    for rule in policy.rules {
        if host_matches_rule(&host, &rule.pattern) {
            return Some(WebsiteBlock {
                host,
                rule: rule.pattern,
                source: rule.source,
            });
        }
    }
    None
}

fn load_website_policy(hermes_home: &Path) -> WebsitePolicy {
    let config_path = hermes_home.join("config.yaml");
    let Ok(contents) = fs::read_to_string(&config_path) else {
        return WebsitePolicy::default();
    };
    let Ok(root) = serde_yaml::from_str::<YamlValue>(&contents) else {
        return WebsitePolicy::default();
    };
    let Some(security) = root
        .as_mapping()
        .and_then(|mapping| yaml_mapping(mapping, "security"))
    else {
        return WebsitePolicy::default();
    };
    let Some(blocklist) = yaml_mapping(security, "website_blocklist") else {
        return WebsitePolicy::default();
    };

    let enabled = yaml_bool(blocklist, "enabled").unwrap_or(false);
    let mut rules = Vec::new();

    if let Some(domains) = yaml_sequence(blocklist, "domains") {
        for domain in domains {
            if let Some(rule) = domain.as_str().and_then(normalize_rule) {
                rules.push(WebsiteRule {
                    pattern: rule,
                    source: "config".to_string(),
                });
            }
        }
    }

    if let Some(shared_files) = yaml_sequence(blocklist, "shared_files") {
        for shared_file in shared_files {
            let Some(path_text) = shared_file
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                continue;
            };
            let path = resolve_shared_path(hermes_home, path_text);
            let source = path.display().to_string();
            for rule in load_shared_rules(&path) {
                rules.push(WebsiteRule {
                    pattern: rule,
                    source: source.clone(),
                });
            }
        }
    }

    WebsitePolicy { enabled, rules }
}

fn load_shared_rules(path: &Path) -> Vec<String> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents.lines().filter_map(normalize_rule).collect()
}

fn resolve_shared_path(hermes_home: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value).expand_home();
    if path.is_absolute() {
        path
    } else {
        hermes_home.join(path)
    }
}

fn normalize_rule(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }

    let lowered = trimmed.to_ascii_lowercase();
    let host = if lowered.contains("://") {
        Url::parse(&lowered)
            .ok()
            .and_then(|parsed| parsed.host_str().map(normalize_host))
            .unwrap_or_else(|| normalize_host(lowered.split("://").nth(1).unwrap_or_default()))
    } else {
        normalize_host(trimmed)
    };
    let host = host.split('/').next().unwrap_or_default().to_string();
    let host = host.strip_prefix("www.").unwrap_or(&host).to_string();
    if host.is_empty() { None } else { Some(host) }
}

fn host_matches_rule(host: &str, rule: &str) -> bool {
    if let Some(suffix) = rule.strip_prefix("*.") {
        return host.ends_with(&format!(".{suffix}"));
    }
    host == rule || host.ends_with(&format!(".{rule}"))
}

fn normalize_host(host: &str) -> String {
    host.trim()
        .to_ascii_lowercase()
        .trim_end_matches('.')
        .to_string()
}

fn yaml_mapping<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Mapping> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_mapping)
}

fn yaml_sequence<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a [YamlValue]> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_sequence)
        .map(Vec::as_slice)
}

fn yaml_bool(mapping: &Mapping, key: &str) -> Option<bool> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_bool)
}

trait ExpandHome {
    fn expand_home(self) -> PathBuf;
}

impl ExpandHome for PathBuf {
    fn expand_home(self) -> PathBuf {
        if !self.starts_with("~") {
            return self;
        }
        match dirs::home_dir() {
            Some(home) if self == PathBuf::from("~") => home,
            Some(home) => match self.strip_prefix("~") {
                Ok(rest) => home.join(rest),
                Err(_) => self,
            },
            None => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;

    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn normalizes_search_results_from_firecrawl_json() {
        let payload = json!({
            "data": {
                "web": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "description": "hello",
                        "position": 7
                    }
                ]
            }
        });
        let results = normalize_search_results(&payload);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["title"], json!("Example"));
        assert_eq!(results[0]["position"], json!(7));
    }

    #[test]
    fn normalizes_extract_results_from_firecrawl_json() {
        let payload = json!({
            "data": [{
                "markdown": "# Title",
                "metadata": {
                    "sourceURL": "https://example.com",
                    "title": "Example"
                }
            }]
        });
        let results = normalize_extract_results(&payload, &[String::from("https://example.com")]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["url"], json!("https://example.com"));
        assert_eq!(results[0]["content"], json!("# Title"));
    }

    #[test]
    fn blocks_secret_bearing_urls() {
        let runtime = ToolRuntime::default();
        let result = handle_web_extract(
            &json!({"urls":["https://example.com/?token=sk-test-secret-token-value"]}),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(false));
        assert_eq!(parsed["error"], json!(BLOCKED_URL_SECRET_ERROR));
    }

    #[test]
    fn blocks_private_and_policy_urls_before_scrape() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - blocked.example\n",
        )
        .unwrap();
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());

        let result = handle_web_extract(
            &json!({"urls":["http://127.0.0.1:8080/", "https://blocked.example/docs"]}),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let items = parsed["results"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["error"], json!(PRIVATE_URL_ERROR));
        assert_eq!(
            items[1]["blocked_by_policy"]["host"],
            json!("blocked.example")
        );
    }

    #[test]
    fn web_tools_can_run_against_fake_firecrawl_cli() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let firecrawl = bin_dir.join("firecrawl");
        fs::write(
            &firecrawl,
            r#"#!/usr/bin/env bash
set -e
if [ "$1" = "--help" ]; then
  echo ok
  exit 0
fi
if [ "$1" = "search" ]; then
  echo '{"data":{"web":[{"title":"Example","url":"https://example.com","description":"search hit","position":1}]}}'
  exit 0
fi
if [ "$1" = "scrape" ]; then
  echo '{"data":[{"markdown":"hello world","metadata":{"sourceURL":"https://example.com","title":"Example"}}]}'
  exit 0
fi
echo "unexpected" >&2
exit 1
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&firecrawl).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&firecrawl, permissions).unwrap();

        let old_path = std::env::var("PATH").unwrap_or_default();
        unsafe {
            std::env::set_var("PATH", format!("{}:{}", bin_dir.display(), old_path));
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let search = handle_web_search(&json!({"query":"test","limit":1}), &runtime);
        let search_json: Value = serde_json::from_str(&search).unwrap();
        assert_eq!(search_json["success"], Value::Bool(true));
        assert_eq!(
            search_json["data"]["web"][0]["url"],
            json!("https://example.com")
        );

        let extract = handle_web_extract(&json!({"urls":["https://example.com"]}), &runtime);
        let extract_json: Value = serde_json::from_str(&extract).unwrap();
        assert_eq!(extract_json["results"][0]["title"], json!("Example"));
        assert_eq!(extract_json["results"][0]["content"], json!("hello world"));

        unsafe {
            std::env::set_var("PATH", old_path);
        }
    }
}
