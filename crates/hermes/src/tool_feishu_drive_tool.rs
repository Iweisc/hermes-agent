//! Feishu Drive Tools -- document comment operations via Feishu/Lark API.
//!
//! Native Rust port of `tools/feishu_drive_tool.py`. Provides tools for
//! listing, replying to, and adding document comments via the Feishu/Lark
//! `drive/v1` open-api endpoints.
//!
//! The Python original relied on the `lark_oapi` SDK, which injects a
//! thread-local "lark client" (set by the feishu_comment handler) and issues a
//! `BaseRequest` with a tenant access token. Since the SDK is not ported, this
//! module reproduces faithfully:
//!
//!   * the request construction (HTTP method, URI template with `:file_token` /
//!     `:comment_id` path segments substituted, the query list, and the JSON
//!     body), and
//!   * the response parsing ((`code`, `msg`, `data`) extracted from the JSON
//!     envelope, mirroring the SDK's `code`/`msg`/`raw.content` handling).
//!
//! The "thread-local client" indirection of the Python code is modelled here as
//! a [`FeishuClient`] trait that callers pass in (mirroring `set_client` /
//! `get_client`). A blocking-reqwest implementation, [`FeishuHttpClient`], is
//! provided that performs the real network call with exact API shapes.

use std::cell::RefCell;
use std::sync::Arc;

use serde_json::{json, Value};

/// The Lark/Feishu host used for open-api requests.
pub const FEISHU_OPEN_API_HOST: &str = "https://open.feishu.cn";

/// 💬 emoji used when registering the list tools (matches Python `\U0001f4ac`).
pub const FEISHU_DRIVE_COMMENT_EMOJI: &str = "\u{1f4ac}";

/// ✉️ emoji used when registering the reply/add tools (matches `✉️`).
pub const FEISHU_DRIVE_REPLY_EMOJI: &str = "\u{2709}\u{fe0f}";

// ---------------------------------------------------------------------------
// URI templates (match the Python module constants exactly)
// ---------------------------------------------------------------------------

/// List-comments URI template (`_LIST_COMMENTS_URI`).
pub const LIST_COMMENTS_URI: &str = "/open-apis/drive/v1/files/:file_token/comments";

/// List-replies URI template (`_LIST_REPLIES_URI`).
pub const LIST_REPLIES_URI: &str =
    "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies";

/// Reply-comment URI template (`_REPLY_COMMENT_URI`); same path as list replies.
pub const REPLY_COMMENT_URI: &str =
    "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies";

/// Add-comment URI template (`_ADD_COMMENT_URI`).
pub const ADD_COMMENT_URI: &str = "/open-apis/drive/v1/files/:file_token/new_comments";

// ---------------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------------

/// Schema for `feishu_drive_list_comments` (mirrors
/// `FEISHU_DRIVE_LIST_COMMENTS_SCHEMA`).
pub fn feishu_drive_list_comments_schema() -> Value {
    json!({
        "name": "feishu_drive_list_comments",
        "description":
            "List comments on a Feishu document. \
             Use is_whole=true to list whole-document comments only.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "The document file token.",
                },
                "file_type": {
                    "type": "string",
                    "description": "File type (default: docx).",
                    "default": "docx",
                },
                "is_whole": {
                    "type": "boolean",
                    "description": "If true, only return whole-document comments.",
                    "default": false,
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of comments per page (max 100).",
                    "default": 100,
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token for next page.",
                },
            },
            "required": ["file_token"],
        },
    })
}

/// Schema for `feishu_drive_list_comment_replies` (mirrors
/// `FEISHU_DRIVE_LIST_REPLIES_SCHEMA`).
pub fn feishu_drive_list_replies_schema() -> Value {
    json!({
        "name": "feishu_drive_list_comment_replies",
        "description": "List all replies in a comment thread on a Feishu document.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "The document file token.",
                },
                "comment_id": {
                    "type": "string",
                    "description": "The comment ID to list replies for.",
                },
                "file_type": {
                    "type": "string",
                    "description": "File type (default: docx).",
                    "default": "docx",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of replies per page (max 100).",
                    "default": 100,
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token for next page.",
                },
            },
            "required": ["file_token", "comment_id"],
        },
    })
}

/// Schema for `feishu_drive_reply_comment` (mirrors `FEISHU_DRIVE_REPLY_SCHEMA`).
pub fn feishu_drive_reply_schema() -> Value {
    json!({
        "name": "feishu_drive_reply_comment",
        "description":
            "Reply to a local comment thread on a Feishu document. \
             Use this for local (quoted-text) comments. \
             For whole-document comments, use feishu_drive_add_comment instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "The document file token.",
                },
                "comment_id": {
                    "type": "string",
                    "description": "The comment ID to reply to.",
                },
                "content": {
                    "type": "string",
                    "description": "The reply text content (plain text only, no markdown).",
                },
                "file_type": {
                    "type": "string",
                    "description": "File type (default: docx).",
                    "default": "docx",
                },
            },
            "required": ["file_token", "comment_id", "content"],
        },
    })
}

/// Schema for `feishu_drive_add_comment` (mirrors
/// `FEISHU_DRIVE_ADD_COMMENT_SCHEMA`).
pub fn feishu_drive_add_comment_schema() -> Value {
    json!({
        "name": "feishu_drive_add_comment",
        "description":
            "Add a new whole-document comment on a Feishu document. \
             Use this for whole-document comments or as a fallback when \
             reply_comment fails with code 1069302.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "The document file token.",
                },
                "content": {
                    "type": "string",
                    "description": "The comment text content (plain text only, no markdown).",
                },
                "file_type": {
                    "type": "string",
                    "description": "File type (default: docx).",
                    "default": "docx",
                },
            },
            "required": ["file_token", "content"],
        },
    })
}

// ---------------------------------------------------------------------------
// tool_result / tool_error helpers (match registry shapes)
// ---------------------------------------------------------------------------

/// Standard error payload: `{"error": "<message>"}`.
pub fn tool_error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

/// Successful tool result carrying a `data` payload, matching the Python call
/// `tool_result(data)` => the data object is returned directly.
///
/// The Python `tool_result(data)` returns the data dict serialised; we serialise
/// the data value directly to mirror that.
pub fn tool_result_data(data: &Value) -> String {
    data.to_string()
}

/// Successful tool result for mutating calls, matching the Python
/// `tool_result(success=True, data=data)` => `{"success": true, "data": ...}`.
pub fn tool_result_success(data: &Value) -> String {
    json!({ "success": true, "data": data.clone() }).to_string()
}

// ---------------------------------------------------------------------------
// Client abstraction (mirrors set_client / get_client thread-local)
// ---------------------------------------------------------------------------

/// HTTP method for a Feishu request. Mirrors the Python `method == "GET"`
/// branch that selects `HttpMethod.GET` else `HttpMethod.POST`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// HTTP GET.
    Get,
    /// HTTP POST (used for any non-GET method string).
    Post,
}

impl HttpMethod {
    /// Mirror Python: `HttpMethod.GET if method == "GET" else HttpMethod.POST`.
    pub fn from_str(method: &str) -> Self {
        if method == "GET" {
            HttpMethod::Get
        } else {
            HttpMethod::Post
        }
    }
}

/// A fully-specified Feishu API request, mirroring the `BaseRequest` the Python
/// `_do_request` builds: method, URI template, path substitutions, query list,
/// and optional JSON body. Always issued with a tenant access token.
#[derive(Debug, Clone)]
pub struct FeishuRequest {
    /// HTTP method.
    pub method: HttpMethod,
    /// URI template (e.g. [`LIST_COMMENTS_URI`]) with `:name` path segments.
    pub uri: String,
    /// Path substitutions, e.g. `{"file_token": "..."}`. Order-insensitive.
    pub paths: Vec<(String, String)>,
    /// Query parameters, preserved in insertion order (matters for fidelity).
    pub queries: Vec<(String, String)>,
    /// Optional JSON body.
    pub body: Option<Value>,
}

/// A parsed Feishu API response: numeric `code` (`0` == success), human-readable
/// `msg`, and the extracted `data` object.
#[derive(Debug, Clone)]
pub struct FeishuResponse {
    /// API status code; `0` means success.
    pub code: i64,
    /// Human-readable message (used in error reporting).
    pub msg: String,
    /// The `data` object from the response envelope (default empty object).
    pub data: Value,
}

/// Abstraction over a Lark/Feishu client capable of issuing a [`FeishuRequest`].
/// Implemented by [`FeishuHttpClient`] for real network calls and by fakes in
/// tests.
pub trait FeishuClient: Send + Sync {
    /// Execute `request` and return the parsed response (or a transport error).
    fn request(&self, request: &FeishuRequest) -> Result<FeishuResponse, String>;
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
// Argument helpers
// ---------------------------------------------------------------------------

/// Fetch a string arg, trimmed, defaulting to `""` when absent/non-string
/// (mirrors `args.get(key, "").strip()`).
fn arg_str_trimmed(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Fetch a string arg without trimming, defaulting to `""`
/// (mirrors `args.get(key, "")`).
fn arg_str(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Resolve the `file_type` arg with the `or "docx"` fallback used in Python:
/// `args.get("file_type", "docx") or "docx"`. A missing key, a JSON null, or an
/// empty string all collapse to `"docx"`.
fn arg_file_type(args: &Value) -> String {
    match args.get("file_type") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => "docx".to_string(),
    }
}

/// Resolve the `page_size` arg, mirroring `str(args.get("page_size", 100))`.
///
/// Integers render without a fractional part; the default is `100`.
fn arg_page_size(args: &Value) -> String {
    match args.get("page_size") {
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                // Float -> Python str() would keep a decimal; preserve as-is.
                n.to_string()
            }
        }
        Some(Value::String(s)) => s.clone(),
        _ => "100".to_string(),
    }
}

/// Resolve a truthy `is_whole` arg (mirrors `args.get("is_whole", False)`).
fn arg_is_whole(args: &Value) -> bool {
    matches!(args.get("is_whole"), Some(Value::Bool(true)))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Handle `feishu_drive_list_comments` using the current thread-local client.
pub fn handle_list_comments(args: &Value) -> String {
    match get_client() {
        Some(client) => handle_list_comments_with_client(client.as_ref(), args),
        None => tool_error("Feishu client not available"),
    }
}

/// Core logic of [`handle_list_comments`] with an explicit client.
pub fn handle_list_comments_with_client(client: &dyn FeishuClient, args: &Value) -> String {
    let file_token = arg_str_trimmed(args, "file_token");
    if file_token.is_empty() {
        return tool_error("file_token is required");
    }

    let file_type = arg_file_type(args);
    let is_whole = arg_is_whole(args);
    let page_size = arg_page_size(args);
    let page_token = arg_str(args, "page_token");

    let mut queries: Vec<(String, String)> = vec![
        ("file_type".into(), file_type),
        ("user_id_type".into(), "open_id".into()),
        ("page_size".into(), page_size),
    ];
    if is_whole {
        queries.push(("is_whole".into(), "true".into()));
    }
    if !page_token.is_empty() {
        queries.push(("page_token".into(), page_token));
    }

    let request = FeishuRequest {
        method: HttpMethod::Get,
        uri: LIST_COMMENTS_URI.to_string(),
        paths: vec![("file_token".into(), file_token)],
        queries,
        body: None,
    };

    match client.request(&request) {
        Ok(resp) => {
            if resp.code != 0 {
                tool_error(&format!(
                    "List comments failed: code={} msg={}",
                    resp.code, resp.msg
                ))
            } else {
                tool_result_data(&resp.data)
            }
        }
        Err(e) => tool_error(&e),
    }
}

/// Handle `feishu_drive_list_comment_replies` using the thread-local client.
pub fn handle_list_replies(args: &Value) -> String {
    match get_client() {
        Some(client) => handle_list_replies_with_client(client.as_ref(), args),
        None => tool_error("Feishu client not available"),
    }
}

/// Core logic of [`handle_list_replies`] with an explicit client.
pub fn handle_list_replies_with_client(client: &dyn FeishuClient, args: &Value) -> String {
    let file_token = arg_str_trimmed(args, "file_token");
    let comment_id = arg_str_trimmed(args, "comment_id");
    if file_token.is_empty() || comment_id.is_empty() {
        return tool_error("file_token and comment_id are required");
    }

    let file_type = arg_file_type(args);
    let page_size = arg_page_size(args);
    let page_token = arg_str(args, "page_token");

    let mut queries: Vec<(String, String)> = vec![
        ("file_type".into(), file_type),
        ("user_id_type".into(), "open_id".into()),
        ("page_size".into(), page_size),
    ];
    if !page_token.is_empty() {
        queries.push(("page_token".into(), page_token));
    }

    let request = FeishuRequest {
        method: HttpMethod::Get,
        uri: LIST_REPLIES_URI.to_string(),
        paths: vec![
            ("file_token".into(), file_token),
            ("comment_id".into(), comment_id),
        ],
        queries,
        body: None,
    };

    match client.request(&request) {
        Ok(resp) => {
            if resp.code != 0 {
                tool_error(&format!(
                    "List replies failed: code={} msg={}",
                    resp.code, resp.msg
                ))
            } else {
                tool_result_data(&resp.data)
            }
        }
        Err(e) => tool_error(&e),
    }
}

/// Handle `feishu_drive_reply_comment` using the thread-local client.
pub fn handle_reply_comment(args: &Value) -> String {
    match get_client() {
        Some(client) => handle_reply_comment_with_client(client.as_ref(), args),
        None => tool_error("Feishu client not available"),
    }
}

/// Core logic of [`handle_reply_comment`] with an explicit client.
pub fn handle_reply_comment_with_client(client: &dyn FeishuClient, args: &Value) -> String {
    let file_token = arg_str_trimmed(args, "file_token");
    let comment_id = arg_str_trimmed(args, "comment_id");
    let content = arg_str_trimmed(args, "content");
    if file_token.is_empty() || comment_id.is_empty() || content.is_empty() {
        return tool_error("file_token, comment_id, and content are required");
    }

    let file_type = arg_file_type(args);

    let body = json!({
        "content": {
            "elements": [
                {
                    "type": "text_run",
                    "text_run": { "text": content },
                }
            ]
        }
    });

    let request = FeishuRequest {
        method: HttpMethod::Post,
        uri: REPLY_COMMENT_URI.to_string(),
        paths: vec![
            ("file_token".into(), file_token),
            ("comment_id".into(), comment_id),
        ],
        queries: vec![("file_type".into(), file_type)],
        body: Some(body),
    };

    match client.request(&request) {
        Ok(resp) => {
            if resp.code != 0 {
                tool_error(&format!(
                    "Reply comment failed: code={} msg={}",
                    resp.code, resp.msg
                ))
            } else {
                tool_result_success(&resp.data)
            }
        }
        Err(e) => tool_error(&e),
    }
}

/// Handle `feishu_drive_add_comment` using the thread-local client.
pub fn handle_add_comment(args: &Value) -> String {
    match get_client() {
        Some(client) => handle_add_comment_with_client(client.as_ref(), args),
        None => tool_error("Feishu client not available"),
    }
}

/// Core logic of [`handle_add_comment`] with an explicit client.
pub fn handle_add_comment_with_client(client: &dyn FeishuClient, args: &Value) -> String {
    let file_token = arg_str_trimmed(args, "file_token");
    let content = arg_str_trimmed(args, "content");
    if file_token.is_empty() || content.is_empty() {
        return tool_error("file_token and content are required");
    }

    let file_type = arg_file_type(args);

    let body = json!({
        "file_type": file_type,
        "reply_elements": [
            { "type": "text", "text": content },
        ],
    });

    let request = FeishuRequest {
        method: HttpMethod::Post,
        uri: ADD_COMMENT_URI.to_string(),
        paths: vec![("file_token".into(), file_token)],
        queries: vec![],
        body: Some(body),
    };

    match client.request(&request) {
        Ok(resp) => {
            if resp.code != 0 {
                tool_error(&format!(
                    "Add comment failed: code={} msg={}",
                    resp.code, resp.msg
                ))
            } else {
                tool_result_success(&resp.data)
            }
        }
        Err(e) => tool_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Response-envelope parsing
// ---------------------------------------------------------------------------

/// Parse a raw JSON envelope string into `(code, msg, data)`, mirroring the
/// Python `_do_request` parsing: pull `code`/`msg` plus `data` (default `{}`).
///
/// When the body is not valid JSON, `code` defaults to `-1`, `msg` to
/// `"unknown error"`, and `data` to an empty object.
pub fn parse_envelope(body: &str) -> FeishuResponse {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let code = parsed.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    let msg = parsed
        .get("msg")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown error")
        .to_string();
    let data = parsed
        .get("data")
        .cloned()
        .unwrap_or_else(|| json!({}));
    FeishuResponse { code, msg, data }
}

// ---------------------------------------------------------------------------
// Real blocking HTTP client
// ---------------------------------------------------------------------------

/// Blocking-reqwest implementation of [`FeishuClient`].
///
/// Substitutes `:name` path segments into the URI template, attaches the query
/// list and an `Authorization: Bearer <tenant_access_token>` header, sends the
/// JSON body (for POST), and parses the JSON envelope into a [`FeishuResponse`].
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

    /// Substitute the request's path segments into its URI template and prefix
    /// the host. Each `paths` entry replaces `:<name>` literally.
    pub fn build_url(&self, request: &FeishuRequest) -> String {
        let mut path = request.uri.clone();
        for (name, value) in &request.paths {
            path = path.replace(&format!(":{name}"), value);
        }
        format!("{}{}", self.host.trim_end_matches('/'), path)
    }
}

impl FeishuClient for FeishuHttpClient {
    fn request(&self, request: &FeishuRequest) -> Result<FeishuResponse, String> {
        let url = self.build_url(request);

        let mut builder = match request.method {
            HttpMethod::Get => self.client.get(&url),
            HttpMethod::Post => self.client.post(&url),
        };

        builder = builder.header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.tenant_access_token),
        );

        if !request.queries.is_empty() {
            builder = builder.query(&request.queries);
        }

        if let Some(body) = &request.body {
            builder = builder.json(body);
        }

        let resp = builder.send().map_err(|e| format!("request failed: {e}"))?;
        let body = resp
            .text()
            .map_err(|e| format!("failed to read response body: {e}"))?;

        Ok(parse_envelope(&body))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A fake client capturing the last request and returning a canned response.
    struct FakeClient {
        resp: Result<FeishuResponse, String>,
        last: Mutex<Option<FeishuRequest>>,
    }

    impl FakeClient {
        fn ok(data: Value) -> Self {
            FakeClient {
                resp: Ok(FeishuResponse {
                    code: 0,
                    msg: "ok".into(),
                    data,
                }),
                last: Mutex::new(None),
            }
        }
        fn fail(code: i64, msg: &str) -> Self {
            FakeClient {
                resp: Ok(FeishuResponse {
                    code,
                    msg: msg.into(),
                    data: json!({}),
                }),
                last: Mutex::new(None),
            }
        }
        fn err(msg: &str) -> Self {
            FakeClient {
                resp: Err(msg.into()),
                last: Mutex::new(None),
            }
        }
        fn captured(&self) -> FeishuRequest {
            self.last.lock().unwrap().clone().unwrap()
        }
    }

    impl FeishuClient for FakeClient {
        fn request(&self, request: &FeishuRequest) -> Result<FeishuResponse, String> {
            *self.last.lock().unwrap() = Some(request.clone());
            self.resp.clone()
        }
    }

    #[test]
    fn schemas_have_expected_names_and_required() {
        assert_eq!(
            feishu_drive_list_comments_schema()["name"],
            "feishu_drive_list_comments"
        );
        assert_eq!(
            feishu_drive_list_replies_schema()["name"],
            "feishu_drive_list_comment_replies"
        );
        assert_eq!(
            feishu_drive_reply_schema()["name"],
            "feishu_drive_reply_comment"
        );
        assert_eq!(
            feishu_drive_add_comment_schema()["name"],
            "feishu_drive_add_comment"
        );
        assert_eq!(
            feishu_drive_reply_schema()["parameters"]["required"],
            json!(["file_token", "comment_id", "content"])
        );
    }

    #[test]
    fn list_comments_missing_token_errors() {
        let client = FakeClient::ok(json!({}));
        let out = handle_list_comments_with_client(&client, &json!({"file_token": "  "}));
        assert_eq!(out, tool_error("file_token is required"));
    }

    #[test]
    fn list_comments_builds_default_queries() {
        let client = FakeClient::ok(json!({"items": [1, 2]}));
        let out = handle_list_comments_with_client(&client, &json!({"file_token": "FT"}));
        // tool_result(data) returns the data dict directly.
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed, json!({"items": [1, 2]}));

        let req = client.captured();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.uri, LIST_COMMENTS_URI);
        assert_eq!(req.paths, vec![("file_token".to_string(), "FT".to_string())]);
        assert_eq!(
            req.queries,
            vec![
                ("file_type".to_string(), "docx".to_string()),
                ("user_id_type".to_string(), "open_id".to_string()),
                ("page_size".to_string(), "100".to_string()),
            ]
        );
        assert!(req.body.is_none());
    }

    #[test]
    fn list_comments_is_whole_and_page_token() {
        let client = FakeClient::ok(json!({}));
        let _ = handle_list_comments_with_client(
            &client,
            &json!({
                "file_token": "FT",
                "file_type": "sheet",
                "is_whole": true,
                "page_size": 50,
                "page_token": "PT"
            }),
        );
        let req = client.captured();
        assert_eq!(
            req.queries,
            vec![
                ("file_type".to_string(), "sheet".to_string()),
                ("user_id_type".to_string(), "open_id".to_string()),
                ("page_size".to_string(), "50".to_string()),
                ("is_whole".to_string(), "true".to_string()),
                ("page_token".to_string(), "PT".to_string()),
            ]
        );
    }

    #[test]
    fn list_comments_empty_file_type_falls_back_to_docx() {
        let client = FakeClient::ok(json!({}));
        let _ = handle_list_comments_with_client(
            &client,
            &json!({"file_token": "FT", "file_type": ""}),
        );
        let req = client.captured();
        assert_eq!(req.queries[0], ("file_type".to_string(), "docx".to_string()));
    }

    #[test]
    fn list_comments_nonzero_code_errors() {
        let client = FakeClient::fail(99, "boom");
        let out = handle_list_comments_with_client(&client, &json!({"file_token": "FT"}));
        assert!(out.contains("List comments failed"));
        assert!(out.contains("code=99"));
        assert!(out.contains("msg=boom"));
    }

    #[test]
    fn list_replies_requires_both_ids() {
        let client = FakeClient::ok(json!({}));
        let out = handle_list_replies_with_client(&client, &json!({"file_token": "FT"}));
        assert_eq!(out, tool_error("file_token and comment_id are required"));
    }

    #[test]
    fn list_replies_builds_request() {
        let client = FakeClient::ok(json!({"items": []}));
        let _ = handle_list_replies_with_client(
            &client,
            &json!({"file_token": "FT", "comment_id": "CID", "page_token": "PT"}),
        );
        let req = client.captured();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.uri, LIST_REPLIES_URI);
        assert_eq!(
            req.paths,
            vec![
                ("file_token".to_string(), "FT".to_string()),
                ("comment_id".to_string(), "CID".to_string()),
            ]
        );
        assert_eq!(
            req.queries,
            vec![
                ("file_type".to_string(), "docx".to_string()),
                ("user_id_type".to_string(), "open_id".to_string()),
                ("page_size".to_string(), "100".to_string()),
                ("page_token".to_string(), "PT".to_string()),
            ]
        );
    }

    #[test]
    fn reply_comment_requires_content() {
        let client = FakeClient::ok(json!({}));
        let out = handle_reply_comment_with_client(
            &client,
            &json!({"file_token": "FT", "comment_id": "CID"}),
        );
        assert_eq!(
            out,
            tool_error("file_token, comment_id, and content are required")
        );
    }

    #[test]
    fn reply_comment_builds_body_and_success() {
        let client = FakeClient::ok(json!({"reply_id": "r1"}));
        let out = handle_reply_comment_with_client(
            &client,
            &json!({"file_token": "FT", "comment_id": "CID", "content": "hi"}),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["data"], json!({"reply_id": "r1"}));

        let req = client.captured();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.uri, REPLY_COMMENT_URI);
        assert_eq!(
            req.queries,
            vec![("file_type".to_string(), "docx".to_string())]
        );
        assert_eq!(
            req.body,
            Some(json!({
                "content": {
                    "elements": [
                        {"type": "text_run", "text_run": {"text": "hi"}}
                    ]
                }
            }))
        );
    }

    #[test]
    fn add_comment_requires_token_and_content() {
        let client = FakeClient::ok(json!({}));
        let out = handle_add_comment_with_client(&client, &json!({"file_token": "FT"}));
        assert_eq!(out, tool_error("file_token and content are required"));
    }

    #[test]
    fn add_comment_builds_body_and_success() {
        let client = FakeClient::ok(json!({"comment_id": "c1"}));
        let out = handle_add_comment_with_client(
            &client,
            &json!({"file_token": "FT", "content": "note", "file_type": "doc"}),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["data"], json!({"comment_id": "c1"}));

        let req = client.captured();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.uri, ADD_COMMENT_URI);
        assert!(req.queries.is_empty());
        assert_eq!(
            req.paths,
            vec![("file_token".to_string(), "FT".to_string())]
        );
        assert_eq!(
            req.body,
            Some(json!({
                "file_type": "doc",
                "reply_elements": [{"type": "text", "text": "note"}]
            }))
        );
    }

    #[test]
    fn transport_error_propagates() {
        let client = FakeClient::err("request failed: boom");
        let out = handle_add_comment_with_client(
            &client,
            &json!({"file_token": "FT", "content": "x"}),
        );
        assert_eq!(out, tool_error("request failed: boom"));
    }

    #[test]
    fn no_client_in_context() {
        clear_client();
        let out = handle_list_comments(&json!({"file_token": "FT"}));
        assert_eq!(out, tool_error("Feishu client not available"));
    }

    #[test]
    fn thread_local_roundtrip() {
        clear_client();
        assert!(get_client().is_none());
        set_client(Arc::new(FakeClient::ok(json!({"ok": 1}))));
        assert!(get_client().is_some());
        let out = handle_list_comments(&json!({"file_token": "FT"}));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed, json!({"ok": 1}));
        clear_client();
    }

    #[test]
    fn http_method_from_str() {
        assert_eq!(HttpMethod::from_str("GET"), HttpMethod::Get);
        assert_eq!(HttpMethod::from_str("POST"), HttpMethod::Post);
        assert_eq!(HttpMethod::from_str("PUT"), HttpMethod::Post);
    }

    #[test]
    fn parse_envelope_extracts_data() {
        let env = json!({"code": 0, "msg": "success", "data": {"x": 1}}).to_string();
        let r = parse_envelope(&env);
        assert_eq!(r.code, 0);
        assert_eq!(r.msg, "success");
        assert_eq!(r.data, json!({"x": 1}));
    }

    #[test]
    fn parse_envelope_defaults_on_garbage() {
        let r = parse_envelope("not json");
        assert_eq!(r.code, -1);
        assert_eq!(r.msg, "unknown error");
        assert_eq!(r.data, json!({}));
    }

    #[test]
    fn build_url_substitutes_paths() {
        let c = FeishuHttpClient::new("tok");
        let req = FeishuRequest {
            method: HttpMethod::Get,
            uri: LIST_REPLIES_URI.to_string(),
            paths: vec![
                ("file_token".into(), "FT".into()),
                ("comment_id".into(), "CID".into()),
            ],
            queries: vec![],
            body: None,
        };
        assert_eq!(
            c.build_url(&req),
            "https://open.feishu.cn/open-apis/drive/v1/files/FT/comments/CID/replies"
        );
    }
}
