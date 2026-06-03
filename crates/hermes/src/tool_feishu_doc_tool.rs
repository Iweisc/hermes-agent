//! Feishu Document Tool -- read document content via Feishu/Lark API.
//!
//! Provides `feishu_doc_read` for reading document content as plain text.
//! This is a native Rust port of `tools/feishu_doc_tool.py`.
//!
//! The Python original relied on the `lark_oapi` SDK, which injects a
//! thread-local "lark client" (set by the feishu_comment handler) and issues a
//! `BaseRequest` to the documents `raw_content` endpoint. Since the SDK is not
//! ported, this module reproduces:
//!
//!   * the request construction (HTTP GET against the `raw_content` URI with the
//!     `document_id` path segment substituted, using a tenant access token), and
//!   * the response parsing (decode the JSON body, extract `data.content`).
//!
//! The "thread-local client" indirection of the Python code is modelled here as
//! an explicit [`FeishuClient`] that callers pass in (mirroring `set_client` /
//! `get_client`). A blocking-reqwest implementation, [`FeishuHttpClient`], is
//! provided that performs the real network call with exact API shapes.

use std::cell::RefCell;
use std::sync::Arc;

use serde_json::{json, Value};

/// The Lark/Feishu host used for open-api requests.
pub const FEISHU_OPEN_API_HOST: &str = "https://open.feishu.cn";

/// Raw-content URI template (matches the Python `_RAW_CONTENT_URI`).
///
/// `:document_id` is substituted with the document token at request time.
pub const RAW_CONTENT_URI: &str = "/open-apis/docx/v1/documents/:document_id/raw_content";

/// The 📄 emoji used when registering this tool.
pub const FEISHU_DOC_EMOJI: &str = "\u{1f4c4}";

// ---------------------------------------------------------------------------
// Tool schema
// ---------------------------------------------------------------------------

/// Returns the JSON schema describing the `feishu_doc_read` tool.
///
/// Mirrors `FEISHU_DOC_READ_SCHEMA` in the Python module.
pub fn feishu_doc_read_schema() -> Value {
    json!({
        "name": "feishu_doc_read",
        "description":
            "Read the full content of a Feishu/Lark document as plain text. \
             Useful when you need more context beyond the quoted text in a comment.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_token": {
                    "type": "string",
                    "description": "The document token (from the document URL or comment context).",
                },
            },
            "required": ["doc_token"],
        },
    })
}

// ---------------------------------------------------------------------------
// tool_result / tool_error helpers (match registry shapes)
// ---------------------------------------------------------------------------

/// Standard error payload, matching the registry's `tool_error` shape used by
/// other ported tools: `{"error": "<message>"}`.
pub fn tool_error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

/// Successful tool result carrying string content, matching the Python call
/// `tool_result(success=True, content=content)` => `{"success": true, "content": ...}`.
pub fn tool_result_content(content: &str) -> String {
    json!({ "success": true, "content": content }).to_string()
}

// ---------------------------------------------------------------------------
// Client abstraction (mirrors set_client / get_client thread-local)
// ---------------------------------------------------------------------------

/// A Feishu API response, modelling the relevant fields of the SDK's response
/// object: a numeric `code`, an optional `msg`, and the raw JSON body content.
#[derive(Debug, Clone)]
pub struct FeishuResponse {
    /// API status code; `0` means success.
    pub code: i64,
    /// Human-readable message (used in error reporting).
    pub msg: String,
    /// Raw response body as a JSON string, if available.
    pub raw_content: Option<String>,
}

/// Abstraction over a Lark/Feishu client capable of issuing the raw-content
/// request. Implemented by [`FeishuHttpClient`] for real network calls and by
/// fakes in tests.
pub trait FeishuClient: Send + Sync {
    /// Perform a GET against `RAW_CONTENT_URI` with `document_id` substituted by
    /// `doc_token`, using a tenant access token.
    fn read_raw_content(&self, doc_token: &str) -> Result<FeishuResponse, String>;
}

thread_local! {
    // Mirrors the Python module-level `threading.local()` holding the client.
    static THREAD_CLIENT: RefCell<Option<Arc<dyn FeishuClient>>> = const { RefCell::new(None) };
}

/// Store a Feishu client for the current thread (mirrors Python `set_client`).
pub fn set_client(client: Arc<dyn FeishuClient>) {
    THREAD_CLIENT.with(|c| *c.borrow_mut() = Some(client));
}

/// Clear the current thread's Feishu client.
pub fn clear_client() {
    THREAD_CLIENT.with(|c| *c.borrow_mut() = None);
}

/// Return the Feishu client for the current thread, or `None`
/// (mirrors Python `get_client`).
pub fn get_client() -> Option<Arc<dyn FeishuClient>> {
    THREAD_CLIENT.with(|c| c.borrow().clone())
}

// ---------------------------------------------------------------------------
// Response-body parsing
// ---------------------------------------------------------------------------

/// Extract the document content string from a raw JSON body, mirroring the
/// Python `body.get("data", {}).get("content", "")`.
///
/// Returns `Some(content)` when the body parses as a JSON object; missing or
/// non-string `data.content` collapses to an empty string (matching Python's
/// `.get(..., "")`). Returns `None` when the body is not valid JSON, letting the
/// caller fall through to the `data` fallback path.
pub fn extract_content_from_raw(raw_content: &str) -> Option<String> {
    let body: Value = serde_json::from_str(raw_content).ok()?;
    let content = body
        .get("data")
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Some(content)
}

// ---------------------------------------------------------------------------
// feishu_doc_read handler
// ---------------------------------------------------------------------------

/// Handle a `feishu_doc_read` invocation.
///
/// `args` is the parsed tool-call arguments object (expects a `doc_token`
/// string). Returns a JSON string (tool result or tool error), faithfully
/// reproducing `_handle_feishu_doc_read` in the Python module.
pub fn handle_feishu_doc_read(args: &Value) -> String {
    let doc_token = args
        .get("doc_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if doc_token.is_empty() {
        return tool_error("doc_token is required");
    }

    let client = match get_client() {
        Some(c) => c,
        None => {
            return tool_error(
                "Feishu client not available (not in a Feishu comment context)",
            )
        }
    };

    handle_feishu_doc_read_with_client(client.as_ref(), &doc_token)
}

/// Core logic of [`handle_feishu_doc_read`] with an explicit client, so callers
/// (and tests) can supply a client without touching thread-local state.
pub fn handle_feishu_doc_read_with_client(client: &dyn FeishuClient, doc_token: &str) -> String {
    let doc_token = doc_token.trim();
    if doc_token.is_empty() {
        return tool_error("doc_token is required");
    }

    let response = match client.read_raw_content(doc_token) {
        Ok(r) => r,
        Err(e) => return tool_error(&e),
    };

    // `code != 0` is a failure in the lark API convention.
    if response.code != 0 {
        return tool_error(&format!(
            "Failed to read document: code={} msg={}",
            response.code, response.msg
        ));
    }

    // Primary path: parse the raw JSON body and pull out data.content.
    if let Some(raw) = &response.raw_content {
        if let Some(content) = extract_content_from_raw(raw) {
            return tool_result_content(&content);
        }
        // If parsing failed, fall through to the (here vestigial) fallback.
    }

    // Fallback in Python tried response.data; with the HTTP client the body is
    // already captured in `raw_content`, so an unparseable/absent body means no
    // content was returned.
    tool_error("No content returned from document API")
}

// ---------------------------------------------------------------------------
// Real blocking HTTP client
// ---------------------------------------------------------------------------

/// Blocking-reqwest implementation of [`FeishuClient`].
///
/// Issues `GET {host}{RAW_CONTENT_URI}` with `:document_id` replaced by the
/// document token and an `Authorization: Bearer <tenant_access_token>` header,
/// matching the tenant-token request the SDK builds.
pub struct FeishuHttpClient {
    /// Open-api host, e.g. `https://open.feishu.cn`.
    pub host: String,
    /// A valid tenant access token.
    pub tenant_access_token: String,
    client: reqwest::blocking::Client,
}

impl FeishuHttpClient {
    /// Create a new HTTP client against the default Feishu host.
    pub fn new(tenant_access_token: impl Into<String>) -> Self {
        Self::with_host(FEISHU_OPEN_API_HOST, tenant_access_token)
    }

    /// Create a new HTTP client against a custom host (e.g. larksuite.com).
    pub fn with_host(host: impl Into<String>, tenant_access_token: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            tenant_access_token: tenant_access_token.into(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Build the full request URL for a given document token.
    pub fn build_url(&self, doc_token: &str) -> String {
        let path = RAW_CONTENT_URI.replace(":document_id", doc_token);
        format!("{}{}", self.host.trim_end_matches('/'), path)
    }
}

impl FeishuClient for FeishuHttpClient {
    fn read_raw_content(&self, doc_token: &str) -> Result<FeishuResponse, String> {
        let url = self.build_url(doc_token);
        let resp = self
            .client
            .get(&url)
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.tenant_access_token),
            )
            .send()
            .map_err(|e| format!("request failed: {e}"))?;

        let body = resp
            .text()
            .map_err(|e| format!("failed to read response body: {e}"))?;

        // The lark API returns a JSON envelope with `code`/`msg` plus `data`.
        let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let code = parsed.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        let msg = parsed
            .get("msg")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error")
            .to_string();

        Ok(FeishuResponse {
            code,
            msg,
            raw_content: Some(body),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeClient {
        resp: Result<FeishuResponse, String>,
    }

    impl FeishuClient for FakeClient {
        fn read_raw_content(&self, _doc_token: &str) -> Result<FeishuResponse, String> {
            self.resp.clone()
        }
    }

    fn ok_resp(raw: &str) -> FakeClient {
        FakeClient {
            resp: Ok(FeishuResponse {
                code: 0,
                msg: "ok".into(),
                raw_content: Some(raw.to_string()),
            }),
        }
    }

    #[test]
    fn schema_shape() {
        let s = feishu_doc_read_schema();
        assert_eq!(s["name"], "feishu_doc_read");
        assert_eq!(s["parameters"]["required"][0], "doc_token");
    }

    #[test]
    fn missing_doc_token_errors() {
        let client = ok_resp("{}");
        let out = handle_feishu_doc_read_with_client(&client, "   ");
        assert_eq!(out, tool_error("doc_token is required"));
    }

    #[test]
    fn no_client_in_context() {
        clear_client();
        let out = handle_feishu_doc_read(&json!({ "doc_token": "tok" }));
        assert!(out.contains("Feishu client not available"));
    }

    #[test]
    fn success_extracts_content() {
        let raw = json!({
            "code": 0,
            "msg": "success",
            "data": { "content": "Hello doc" }
        })
        .to_string();
        let client = ok_resp(&raw);
        let out = handle_feishu_doc_read_with_client(&client, "tok");
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["content"], "Hello doc");
    }

    #[test]
    fn missing_content_yields_empty_string() {
        // data present but no content key -> "" (matches Python .get(.., "")).
        let raw = json!({ "code": 0, "data": {} }).to_string();
        let client = ok_resp(&raw);
        let out = handle_feishu_doc_read_with_client(&client, "tok");
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["content"], "");
    }

    #[test]
    fn nonzero_code_errors() {
        let client = FakeClient {
            resp: Ok(FeishuResponse {
                code: 1254006,
                msg: "doc not found".into(),
                raw_content: Some("{}".into()),
            }),
        };
        let out = handle_feishu_doc_read_with_client(&client, "tok");
        assert!(out.contains("Failed to read document"));
        assert!(out.contains("code=1254006"));
        assert!(out.contains("doc not found"));
    }

    #[test]
    fn unparseable_body_no_content() {
        let client = FakeClient {
            resp: Ok(FeishuResponse {
                code: 0,
                msg: "ok".into(),
                raw_content: Some("not json".into()),
            }),
        };
        let out = handle_feishu_doc_read_with_client(&client, "tok");
        assert_eq!(out, tool_error("No content returned from document API"));
    }

    #[test]
    fn transport_error_propagates() {
        let client = FakeClient {
            resp: Err("request failed: boom".into()),
        };
        let out = handle_feishu_doc_read_with_client(&client, "tok");
        assert_eq!(out, tool_error("request failed: boom"));
    }

    #[test]
    fn thread_local_roundtrip() {
        clear_client();
        assert!(get_client().is_none());
        let raw = json!({ "code": 0, "data": { "content": "X" } }).to_string();
        set_client(Arc::new(ok_resp(&raw)));
        assert!(get_client().is_some());
        let out = handle_feishu_doc_read(&json!({ "doc_token": "tok" }));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["content"], "X");
        clear_client();
    }

    #[test]
    fn build_url_substitutes_token() {
        let c = FeishuHttpClient::new("tok");
        let url = c.build_url("DOC123");
        assert_eq!(
            url,
            "https://open.feishu.cn/open-apis/docx/v1/documents/DOC123/raw_content"
        );
    }
}
