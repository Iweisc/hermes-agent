//! SearXNG web search provider.
//!
//! Faithful Rust port of `tools/web_providers/searxng.py`.
//!
//! SearXNG is a free, self-hosted, privacy-respecting metasearch engine. It
//! implements the web *search* capability only — there is no extract
//! capability.
//!
//! Configuration (a URL, not a secret — set it in `config.yaml`, not `.env`):
//!
//! ```yaml
//! # ~/.hermes/config.yaml
//! SEARXNG_URL: http://localhost:8080
//!
//! # Use SearXNG for search, pair with any extract provider:
//! web:
//!   search_backend: "searxng"
//!   extract_backend: "firecrawl"
//! ```
//!
//! Public SearXNG instances are listed at <https://searx.space/> but
//! self-hosting is recommended for production use (rate limits and
//! availability vary per public instance).
//!
//! Uses the SearXNG JSON API (`/search?format=json`). Results are sorted by
//! SearXNG's own score (descending) and truncated to `limit`.

use std::time::Duration;

use serde_json::{json, Value};

/// HTTP request timeout, matching the Python `timeout=15`.
const SEARXNG_TIMEOUT_SECS: u64 = 15;

/// Search via a SearXNG instance.
///
/// Requires `SEARXNG_URL` to be set (e.g. `http://localhost:8080`). No API key
/// is needed — SearXNG is open-source and self-hosted.
///
/// The Python original reads `SEARXNG_URL` from the process environment inside
/// each method. We preserve that behavior with [`SearXNGSearchProvider::new`]
/// (env-backed), and additionally expose [`SearXNGSearchProvider::with_url`]
/// for tests / explicit configuration.
#[derive(Debug, Clone, Default)]
pub struct SearXNGSearchProvider {
    /// Explicit base URL override. When `None`, the value is read from the
    /// `SEARXNG_URL` environment variable on each access (matching Python).
    base_url_override: Option<String>,
}

impl SearXNGSearchProvider {
    /// Construct an env-backed provider (reads `SEARXNG_URL` from the
    /// environment), mirroring the Python provider.
    pub fn new() -> Self {
        Self {
            base_url_override: None,
        }
    }

    /// Construct a provider with an explicit base URL (no env access).
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            base_url_override: Some(url.into()),
        }
    }

    /// Read the raw, untrimmed `SEARXNG_URL` value (or the override).
    fn raw_url(&self) -> String {
        match &self.base_url_override {
            Some(u) => u.clone(),
            None => std::env::var("SEARXNG_URL").unwrap_or_default(),
        }
    }

    /// Short, human-readable provider name.
    pub fn provider_name(&self) -> String {
        "searxng".to_string()
    }

    /// Return `true` when `SEARXNG_URL` is set to a non-empty value.
    ///
    /// Mirrors `bool(os.getenv("SEARXNG_URL", "").strip())`.
    pub fn is_configured(&self) -> bool {
        !self.raw_url().trim().is_empty()
    }

    /// Convenience wrapper applying the Python default `limit=5`.
    pub fn search_default(&self, query: &str) -> Value {
        self.search(query, 5)
    }

    /// Execute a search against the configured SearXNG instance.
    ///
    /// On success returns:
    ///
    /// ```text
    /// {"success": true, "data": {"web": [
    ///     {"title": str, "url": str, "description": str, "position": int},
    ///     ...
    /// ]}}
    /// ```
    ///
    /// On failure returns `{"success": false, "error": str}`.
    ///
    /// This performs a blocking HTTP request. The default `limit` in Python is
    /// `5`; callers should pass `5` for parity (see [`search_default`]).
    ///
    /// [`search_default`]: SearXNGSearchProvider::search_default
    pub fn search(&self, query: &str, limit: usize) -> Value {
        // base_url = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
        let base_url = self.raw_url().trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return error("SEARXNG_URL is not set");
        }

        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(SEARXNG_TIMEOUT_SECS))
            .build()
        {
            Ok(c) => c,
            Err(exc) => {
                log::warn!("SearXNG request error: {exc}");
                return error(format!("Could not reach SearXNG at {base_url}: {exc}"));
            }
        };

        let url = format!("{base_url}/search");
        let resp = match client
            .get(&url)
            .header("Accept", "application/json")
            // params: q, format=json, pageno=1
            .query(&[("q", query), ("format", "json"), ("pageno", "1")])
            .send()
        {
            Ok(r) => r,
            Err(exc) => {
                // httpx.RequestError equivalent — connection/transport failure.
                log::warn!("SearXNG request error: {exc}");
                return error(format!("Could not reach SearXNG at {base_url}: {exc}"));
            }
        };

        // resp.raise_for_status() — httpx.HTTPStatusError equivalent.
        let resp = match resp.error_for_status() {
            Ok(r) => r,
            Err(exc) => {
                let status = exc
                    .status()
                    .map(|s| s.as_u16().to_string())
                    .unwrap_or_else(|| "error".to_string());
                log::warn!("SearXNG HTTP error: {exc}");
                return error(format!("SearXNG returned HTTP {status}"));
            }
        };

        // data = resp.json()
        let data: Value = match resp.json() {
            Ok(v) => v,
            Err(exc) => {
                log::warn!("SearXNG response parse error: {exc}");
                return error("Could not parse SearXNG response as JSON");
            }
        };

        parse_results(&data, query, limit)
    }
}

/// Build the normalized failure payload `{"success": false, "error": str}`.
fn error(msg: impl Into<String>) -> Value {
    json!({ "success": false, "error": msg.into() })
}

/// Coerce a JSON value into the float used for score-sorting.
///
/// Mirrors `float(r.get("score", 0))`: missing -> 0.0; numbers are used
/// directly; numeric strings parse; anything non-numeric is treated as 0.0.
/// (Python would raise on a non-numeric string, but SearXNG always emits a
/// numeric `score`; we degrade gracefully to keep results flowing.)
fn score_of(result: &Value) -> f64 {
    match result.get("score") {
        None | Some(Value::Null) => 0.0,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(0.0),
        Some(Value::Bool(b)) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Coerce a JSON value at `key` into the Python `str(r.get(key, ""))`.
///
/// Strings pass through verbatim; missing/null become `""`; other JSON scalars
/// are stringified.
fn str_field(result: &Value, key: &str) -> String {
    match result.get(key) {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Normalize the parsed SearXNG JSON body into the success envelope.
///
/// Split out from [`SearXNGSearchProvider::search`] so it can be unit-tested
/// without performing any network I/O.
fn parse_results(data: &Value, query: &str, limit: usize) -> Value {
    // raw_results = data.get("results", [])
    let empty: Vec<Value> = Vec::new();
    let raw_results: Vec<&Value> = match data.get("results") {
        Some(Value::Array(arr)) => arr.iter().collect(),
        _ => empty.iter().collect(),
    };
    let raw_count = raw_results.len();

    // Sort by score descending. Python's sorted() is stable; we preserve that
    // so equal-score items keep their original (insertion) order.
    let mut sorted_results: Vec<&Value> = raw_results;
    sorted_results.sort_by(|a, b| {
        // Descending: compare b vs a.
        score_of(b)
            .partial_cmp(&score_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let web_results: Vec<Value> = sorted_results
        .iter()
        .take(limit)
        .enumerate()
        .map(|(i, r)| {
            json!({
                "title": str_field(r, "title"),
                "url": str_field(r, "url"),
                "description": str_field(r, "content"),
                "position": (i + 1) as i64,
            })
        })
        .collect();

    log::info!(
        "SearXNG search '{}': {} results (from {} raw, limit {})",
        query,
        web_results.len(),
        raw_count,
        limit,
    );

    json!({ "success": true, "data": { "web": web_results } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_name_is_searxng() {
        assert_eq!(SearXNGSearchProvider::new().provider_name(), "searxng");
    }

    #[test]
    fn with_url_is_configured() {
        let p = SearXNGSearchProvider::with_url("http://localhost:8080");
        assert!(p.is_configured());
    }

    #[test]
    fn blank_url_not_configured() {
        let p = SearXNGSearchProvider::with_url("   ");
        assert!(!p.is_configured());
        let p2 = SearXNGSearchProvider::with_url("");
        assert!(!p2.is_configured());
    }

    #[test]
    fn search_errors_when_url_unset() {
        let p = SearXNGSearchProvider::with_url("");
        let v = p.search("rust", 5);
        assert_eq!(v["success"], false);
        assert_eq!(v["error"], "SEARXNG_URL is not set");
    }

    #[test]
    fn search_errors_when_url_only_slashes_after_trim() {
        // "  /  " trims to "/" then rstrip("/") -> "" -> not set.
        let p = SearXNGSearchProvider::with_url("  /  ");
        let v = p.search("rust", 5);
        assert_eq!(v["success"], false);
        assert_eq!(v["error"], "SEARXNG_URL is not set");
    }

    #[test]
    fn parse_sorts_by_score_descending_and_caps_to_limit() {
        let data = json!({
            "results": [
                {"title": "low", "url": "u1", "content": "c1", "score": 0.5},
                {"title": "high", "url": "u2", "content": "c2", "score": 9.0},
                {"title": "mid", "url": "u3", "content": "c3", "score": 3.0},
            ]
        });
        let v = parse_results(&data, "q", 2);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web.len(), 2);
        assert_eq!(web[0]["title"], "high");
        assert_eq!(web[0]["position"], 1);
        assert_eq!(web[1]["title"], "mid");
        assert_eq!(web[1]["position"], 2);
        // description comes from the "content" field.
        assert_eq!(web[0]["description"], "c2");
        assert_eq!(web[0]["url"], "u2");
    }

    #[test]
    fn parse_missing_score_defaults_to_zero_stable_order() {
        let data = json!({
            "results": [
                {"title": "a"},
                {"title": "b", "score": 0},
                {"title": "c", "score": 1.0},
            ]
        });
        let v = parse_results(&data, "q", 5);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web.len(), 3);
        // c has highest score; a and b tie at 0 and keep insertion order.
        assert_eq!(web[0]["title"], "c");
        assert_eq!(web[1]["title"], "a");
        assert_eq!(web[2]["title"], "b");
    }

    #[test]
    fn parse_missing_fields_become_empty_strings() {
        let data = json!({ "results": [ {"score": 1.0} ] });
        let v = parse_results(&data, "q", 5);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web[0]["title"], "");
        assert_eq!(web[0]["url"], "");
        assert_eq!(web[0]["description"], "");
        assert_eq!(web[0]["position"], 1);
    }

    #[test]
    fn parse_no_results_key_yields_empty_web() {
        let data = json!({ "other": 1 });
        let v = parse_results(&data, "q", 5);
        assert_eq!(v["success"], true);
        assert!(v["data"]["web"].as_array().unwrap().is_empty());
    }

    #[test]
    fn parse_string_score_is_parsed() {
        let data = json!({
            "results": [
                {"title": "a", "score": "0.1"},
                {"title": "b", "score": "5"},
            ]
        });
        let v = parse_results(&data, "q", 5);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web[0]["title"], "b");
        assert_eq!(web[1]["title"], "a");
    }

    #[test]
    fn parse_non_string_scalar_fields_are_stringified() {
        let data = json!({
            "results": [ {"title": 123, "url": true, "content": 4.5, "score": 1} ]
        });
        let v = parse_results(&data, "q", 5);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web[0]["title"], "123");
        assert_eq!(web[0]["url"], "true");
        assert_eq!(web[0]["description"], "4.5");
    }

    #[test]
    fn env_backed_provider_reads_searxng_url() {
        unsafe {
            std::env::set_var("SEARXNG_URL", "http://example.test:8080/");
        }
        let p = SearXNGSearchProvider::new();
        assert!(p.is_configured());
        unsafe {
            std::env::remove_var("SEARXNG_URL");
        }
        let p2 = SearXNGSearchProvider::new();
        assert!(!p2.is_configured());
    }
}
