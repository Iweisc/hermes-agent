//! Abstract base classes for web capability providers.
//!
//! Port of `tools/web_providers/base.py`. The Python module defines two
//! `abc.ABC` interfaces — [`WebSearchProvider`] and [`WebExtractProvider`] —
//! plus the normalized result shapes their implementations must return.
//!
//! In Rust the abstract methods map onto trait methods. The normalized
//! success/failure envelopes are expressed both as JSON-construction helpers
//! (matching the exact dict shapes the Python code produces) and as typed
//! structs for ergonomic consumption by sibling modules.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A single web-search result entry.
///
/// Mirrors the per-item dict in the normalized search envelope::
///
/// ```text
/// {"title": str, "url": str, "description": str, "position": int}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Render this entry as the exact JSON object Python emits.
    pub fn to_json(&self) -> Value {
        json!({
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "position": self.position,
        })
    }
}

/// A single web-extract result entry.
///
/// Mirrors the per-item dict in the normalized extract envelope::
///
/// ```text
/// {"url": str, "title": str, "content": str, "raw_content": str, "metadata": dict}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebExtractResult {
    pub url: String,
    pub title: String,
    pub content: String,
    pub raw_content: String,
    /// Free-form metadata; Python stores an arbitrary dict here.
    #[serde(default)]
    pub metadata: Value,
}

impl WebExtractResult {
    pub fn new(
        url: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
        raw_content: impl Into<String>,
        metadata: Value,
    ) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            content: content.into(),
            raw_content: raw_content.into(),
            metadata,
        }
    }

    /// Render this entry as the exact JSON object Python emits.
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

/// Build the normalized success envelope for a web search.
///
/// Produces::
///
/// ```text
/// {"success": True, "data": {"web": [ ...results... ]}}
/// ```
pub fn search_success(results: &[WebSearchResult]) -> Value {
    let web: Vec<Value> = results.iter().map(WebSearchResult::to_json).collect();
    json!({
        "success": true,
        "data": { "web": web },
    })
}

/// Build the normalized success envelope for a web extract.
///
/// Produces::
///
/// ```text
/// {"success": True, "data": [ ...results... ]}
/// ```
pub fn extract_success(results: &[WebExtractResult]) -> Value {
    let data: Vec<Value> = results.iter().map(WebExtractResult::to_json).collect();
    json!({
        "success": true,
        "data": data,
    })
}

/// Build the normalized failure envelope shared by both provider kinds.
///
/// Produces::
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
/// Implementations live in sibling modules. The user selects a provider via
/// `hermes tools`; the choice is persisted as `config["web"]["search_backend"]`
/// (falling back to `config["web"]["backend"]`).
///
/// Search providers return results in the normalized format produced by
/// [`search_success`] / [`provider_error`].
pub trait WebSearchProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    ///
    /// Called at tool-registration time to gate availability. Must be cheap —
    /// no network calls.
    fn is_configured(&self) -> bool;

    /// Execute a web search and return normalized results.
    ///
    /// The Python signature defaults `limit` to 5; callers in Rust pass it
    /// explicitly (use [`DEFAULT_SEARCH_LIMIT`]).
    fn search(&self, query: &str, limit: u32) -> Value;
}

/// Default `limit` argument for [`WebSearchProvider::search`], matching the
/// Python default of `limit: int = 5`.
pub const DEFAULT_SEARCH_LIMIT: u32 = 5;

/// Interface for web content extraction backends.
///
/// Implementations live in sibling modules. The user selects a provider via
/// `hermes tools`; the choice is persisted as
/// `config["web"]["extract_backend"]` (falling back to
/// `config["web"]["backend"]`).
///
/// Extract providers return results in the normalized format produced by
/// [`extract_success`] / [`provider_error`].
pub trait WebExtractProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    ///
    /// Called at tool-registration time to gate availability. Must be cheap —
    /// no network calls.
    fn is_configured(&self) -> bool;

    /// Extract content from the given URLs and return normalized results.
    ///
    /// The Python signature accepts arbitrary `**kwargs`; in Rust those are
    /// passed as a JSON object (use [`serde_json::Value::Null`] when none).
    fn extract(&self, urls: &[String], kwargs: &Value) -> Value;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_success_shape_matches_python() {
        let results = vec![
            WebSearchResult::new("Title A", "https://a.example", "desc a", 1),
            WebSearchResult::new("Title B", "https://b.example", "desc b", 2),
        ];
        let env = search_success(&results);
        assert_eq!(env["success"], json!(true));
        assert_eq!(env["data"]["web"].as_array().unwrap().len(), 2);
        assert_eq!(env["data"]["web"][0]["title"], json!("Title A"));
        assert_eq!(env["data"]["web"][0]["url"], json!("https://a.example"));
        assert_eq!(env["data"]["web"][0]["description"], json!("desc a"));
        assert_eq!(env["data"]["web"][0]["position"], json!(1));
    }

    #[test]
    fn search_success_empty_has_empty_web_array() {
        let env = search_success(&[]);
        assert_eq!(env, json!({"success": true, "data": {"web": []}}));
    }

    #[test]
    fn extract_success_shape_matches_python() {
        let results = vec![WebExtractResult::new(
            "https://x.example",
            "X title",
            "main content",
            "<html>raw</html>",
            json!({"lang": "en"}),
        )];
        let env = extract_success(&results);
        assert_eq!(env["success"], json!(true));
        let data = env["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["url"], json!("https://x.example"));
        assert_eq!(data[0]["title"], json!("X title"));
        assert_eq!(data[0]["content"], json!("main content"));
        assert_eq!(data[0]["raw_content"], json!("<html>raw</html>"));
        assert_eq!(data[0]["metadata"], json!({"lang": "en"}));
    }

    #[test]
    fn error_shape_matches_python() {
        let env = provider_error("boom");
        assert_eq!(env, json!({"success": false, "error": "boom"}));
    }

    #[test]
    fn default_limit_is_five() {
        assert_eq!(DEFAULT_SEARCH_LIMIT, 5);
    }

    struct DummySearch;
    impl WebSearchProvider for DummySearch {
        fn provider_name(&self) -> String {
            "dummy".to_string()
        }
        fn is_configured(&self) -> bool {
            true
        }
        fn search(&self, query: &str, limit: u32) -> Value {
            let mut out = Vec::new();
            for i in 0..(limit as i64) {
                out.push(WebSearchResult::new(
                    format!("{query} {i}"),
                    format!("https://example/{i}"),
                    "d",
                    i,
                ));
            }
            search_success(&out)
        }
    }

    struct DummyExtract;
    impl WebExtractProvider for DummyExtract {
        fn provider_name(&self) -> String {
            "dummy-extract".to_string()
        }
        fn is_configured(&self) -> bool {
            false
        }
        fn extract(&self, urls: &[String], _kwargs: &Value) -> Value {
            if urls.is_empty() {
                return provider_error("no urls");
            }
            let results: Vec<WebExtractResult> = urls
                .iter()
                .map(|u| WebExtractResult::new(u.clone(), "", "", "", json!({})))
                .collect();
            extract_success(&results)
        }
    }

    #[test]
    fn search_trait_object_usable() {
        let p: &dyn WebSearchProvider = &DummySearch;
        assert_eq!(p.provider_name(), "dummy");
        assert!(p.is_configured());
        let env = p.search("q", DEFAULT_SEARCH_LIMIT);
        assert_eq!(env["data"]["web"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn extract_trait_object_usable() {
        let p: &dyn WebExtractProvider = &DummyExtract;
        assert_eq!(p.provider_name(), "dummy-extract");
        assert!(!p.is_configured());
        assert_eq!(p.extract(&[], &Value::Null), provider_error("no urls"));
        let env = p.extract(&["https://u".to_string()], &Value::Null);
        assert_eq!(env["data"].as_array().unwrap().len(), 1);
    }
}
