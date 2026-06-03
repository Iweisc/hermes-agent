//! Abstract base traits for web capability providers.
//!
//! This is a faithful Rust port of `tools/web_providers/base.py`, which defines
//! the abstract interfaces for web search and web content-extraction backends
//! (Firecrawl, Tavily, Exa, etc.).
//!
//! In Python these are `abc.ABC` classes with `@abstractmethod` members. In Rust
//! the natural equivalent is a `trait` whose methods every concrete provider must
//! implement. Implementations live in sibling modules; the user selects a provider
//! via `hermes tools`, and the choice is persisted as
//! `config["web"]["search_backend"]` / `config["web"]["extract_backend"]`
//! (each falling back to `config["web"]["backend"]`).
//!
//! Both kinds of provider return results in a normalized JSON-shaped format.

use serde_json::{json, Value};

/// A single normalized web-search hit.
///
/// Mirrors the Python dict::
///
/// ```text
/// {"title": str, "url": str, "description": str, "position": int}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
    pub position: i64,
}

impl WebSearchResult {
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        description: impl Into<String>,
        position: i64,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            description: description.into(),
            position,
        }
    }

    /// Serialize to the normalized JSON object shape.
    pub fn to_json(&self) -> Value {
        json!({
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "position": self.position,
        })
    }
}

/// A single normalized extracted-content item.
///
/// Mirrors the Python dict::
///
/// ```text
/// {"url": str, "title": str, "content": str, "raw_content": str, "metadata": dict}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WebExtractResult {
    pub url: String,
    pub title: String,
    pub content: String,
    pub raw_content: String,
    /// Free-form metadata; defaults to an empty object.
    pub metadata: Value,
}

impl WebExtractResult {
    pub fn new(
        url: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
        raw_content: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            content: content.into(),
            raw_content: raw_content.into(),
            metadata: json!({}),
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Serialize to the normalized JSON object shape.
    pub fn to_json(&self) -> Value {
        json!({
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "raw_content": self.raw_content,
            "metadata": self.metadata,
        })
    }
}

/// Build the normalized success payload for a search provider::
///
/// ```text
/// {"success": True, "data": {"web": [ ... ]}}
/// ```
pub fn search_success(results: &[WebSearchResult]) -> Value {
    let web: Vec<Value> = results.iter().map(WebSearchResult::to_json).collect();
    json!({
        "success": true,
        "data": { "web": web },
    })
}

/// Build the normalized success payload for an extract provider::
///
/// ```text
/// {"success": True, "data": [ ... ]}
/// ```
pub fn extract_success(results: &[WebExtractResult]) -> Value {
    let data: Vec<Value> = results.iter().map(WebExtractResult::to_json).collect();
    json!({
        "success": true,
        "data": data,
    })
}

/// Build the normalized failure payload shared by both provider kinds::
///
/// ```text
/// {"success": False, "error": str}
/// ```
pub fn provider_error(error: impl Into<String>) -> Value {
    json!({
        "success": false,
        "error": error.into(),
    })
}

/// Interface for web search backends (Firecrawl, Tavily, Exa, etc.).
///
/// Search providers return results in a normalized format::
///
/// ```text
/// {
///     "success": True,
///     "data": {
///         "web": [
///             {"title": str, "url": str, "description": str, "position": int},
///             ...
///         ]
///     }
/// }
/// ```
///
/// On failure::
///
/// ```text
/// {"success": False, "error": str}
/// ```
pub trait WebSearchProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    ///
    /// Called at tool-registration time to gate availability.
    /// Must be cheap — no network calls.
    fn is_configured(&self) -> bool;

    /// Execute a web search and return normalized results.
    ///
    /// Matches the Python default of `limit = 5`; callers may use
    /// [`WebSearchProvider::search`] directly, or [`WebSearchProvider::search_default`]
    /// for the default-limit convenience.
    fn search(&self, query: &str, limit: usize) -> Value;

    /// Convenience wrapper applying the Python default `limit=5`.
    fn search_default(&self, query: &str) -> Value {
        self.search(query, 5)
    }
}

/// Interface for web content extraction backends.
///
/// Extract providers return results in a normalized format::
///
/// ```text
/// {
///     "success": True,
///     "data": [
///         {"url": str, "title": str, "content": str,
///          "raw_content": str, "metadata": dict},
///         ...
///     ]
/// }
/// ```
///
/// On failure::
///
/// ```text
/// {"success": False, "error": str}
/// ```
pub trait WebExtractProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    ///
    /// Called at tool-registration time to gate availability.
    /// Must be cheap — no network calls.
    fn is_configured(&self) -> bool;

    /// Extract content from the given URLs and return normalized results.
    ///
    /// The Python signature accepts arbitrary `**kwargs`; concrete Rust
    /// implementations should accept whatever extra options they need on the
    /// implementing type itself (e.g. via builder fields).
    fn extract(&self, urls: &[String]) -> Value;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummySearch {
        configured: bool,
    }

    impl WebSearchProvider for DummySearch {
        fn provider_name(&self) -> String {
            "dummy-search".to_string()
        }
        fn is_configured(&self) -> bool {
            self.configured
        }
        fn search(&self, query: &str, limit: usize) -> Value {
            if !self.configured {
                return provider_error("not configured");
            }
            let results: Vec<WebSearchResult> = (0..limit)
                .map(|i| {
                    WebSearchResult::new(
                        format!("title {i} for {query}"),
                        format!("https://example.com/{i}"),
                        format!("desc {i}"),
                        i as i64,
                    )
                })
                .collect();
            search_success(&results)
        }
    }

    struct DummyExtract;

    impl WebExtractProvider for DummyExtract {
        fn provider_name(&self) -> String {
            "dummy-extract".to_string()
        }
        fn is_configured(&self) -> bool {
            true
        }
        fn extract(&self, urls: &[String]) -> Value {
            let results: Vec<WebExtractResult> = urls
                .iter()
                .map(|u| {
                    WebExtractResult::new(u.clone(), "T", "body", "raw")
                        .with_metadata(json!({"len": "raw".len()}))
                })
                .collect();
            extract_success(&results)
        }
    }

    #[test]
    fn search_result_json_shape() {
        let r = WebSearchResult::new("T", "https://a", "d", 3);
        let v = r.to_json();
        assert_eq!(v["title"], "T");
        assert_eq!(v["url"], "https://a");
        assert_eq!(v["description"], "d");
        assert_eq!(v["position"], 3);
    }

    #[test]
    fn extract_result_default_metadata_is_object() {
        let r = WebExtractResult::new("https://a", "T", "c", "raw");
        assert!(r.metadata.is_object());
        assert_eq!(r.to_json()["metadata"], json!({}));
    }

    #[test]
    fn search_success_envelope() {
        let results = vec![
            WebSearchResult::new("a", "u1", "d1", 0),
            WebSearchResult::new("b", "u2", "d2", 1),
        ];
        let v = search_success(&results);
        assert_eq!(v["success"], true);
        let web = v["data"]["web"].as_array().unwrap();
        assert_eq!(web.len(), 2);
        assert_eq!(web[1]["position"], 1);
    }

    #[test]
    fn extract_success_envelope() {
        let results = vec![WebExtractResult::new("u", "t", "c", "r")];
        let v = extract_success(&results);
        assert_eq!(v["success"], true);
        assert!(v["data"].is_array());
        assert_eq!(v["data"][0]["url"], "u");
    }

    #[test]
    fn error_envelope() {
        let v = provider_error("boom");
        assert_eq!(v["success"], false);
        assert_eq!(v["error"], "boom");
    }

    #[test]
    fn search_default_uses_limit_five() {
        let p = DummySearch { configured: true };
        let v = p.search_default("q");
        assert_eq!(v["success"], true);
        assert_eq!(v["data"]["web"].as_array().unwrap().len(), 5);
        assert_eq!(p.provider_name(), "dummy-search");
    }

    #[test]
    fn search_provider_error_when_unconfigured() {
        let p = DummySearch { configured: false };
        assert!(!p.is_configured());
        let v = p.search("q", 3);
        assert_eq!(v["success"], false);
        assert_eq!(v["error"], "not configured");
    }

    #[test]
    fn extract_provider_roundtrip() {
        let p = DummyExtract;
        assert!(p.is_configured());
        let urls = vec!["https://a".to_string(), "https://b".to_string()];
        let v = p.extract(&urls);
        assert_eq!(v["success"], true);
        assert_eq!(v["data"].as_array().unwrap().len(), 2);
        assert_eq!(v["data"][1]["url"], "https://b");
        assert_eq!(v["data"][0]["metadata"]["len"], 3);
    }
}
