//! Feishu/Lark drive document comment handling.
//!
//! Port of `gateway/platforms/feishu_comment.py`.
//!
//! Processes `drive.notice.comment_add_v1` events and interacts with the
//! Drive v2 comment reaction API.  The Python original orchestrates an
//! `AIAgent` call and a live `lark_oapi` SDK client; here we port the
//! self-contained logic faithfully (event parsing, request construction,
//! response parsing, content extraction, document-link parsing, prompt
//! construction, text chunking, timeline selection and the in-process
//! session cache) and expose the network layer behind a small trait so the
//! Rust gateway can supply its own transport.
//!
//! Flow (handled by [`orchestrate_comment_event`] given a [`CommentApi`]):
//!   1. Parse event -> extract file_token, comment_id, reply_id, etc.
//!   2. Add OK reaction
//!   3. Fetch: doc meta + comment details (batch_query)
//!   4. Branch on is_whole:
//!        Whole -> list whole comments timeline
//!        Local -> list comment thread replies
//!   5. Build prompt (local or whole)
//!   6. Agent generates reply (caller-provided closure)
//!   7. Route reply:
//!        Whole -> add_whole_comment
//!        Local -> reply_to_comment (fallback to add_whole_comment on 1069302)

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// API request modelling
// ---------------------------------------------------------------------------

/// HTTP method for a lark request (only GET / POST are used).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

impl HttpMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
        }
    }

    /// Mirror Python: `HttpMethod.GET if method == "GET" else HttpMethod.POST`.
    pub fn from_method_str(method: &str) -> HttpMethod {
        if method == "GET" {
            HttpMethod::Get
        } else {
            HttpMethod::Post
        }
    }
}

/// A fully-described lark API request, equivalent to the dict that Python
/// passes to `_build_request` / `_exec_request`.
#[derive(Debug, Clone)]
pub struct ApiRequest {
    pub method: HttpMethod,
    pub uri: String,
    /// Path parameters, e.g. `{"file_token": "..."}`.
    pub paths: HashMap<String, String>,
    /// Query parameters as an ordered list of (key, value) pairs.
    /// Ordering is preserved to match the Python list-of-tuples behaviour.
    pub queries: Vec<(String, String)>,
    /// Optional JSON request body.
    pub body: Option<Value>,
}

impl ApiRequest {
    pub fn new(method: HttpMethod, uri: impl Into<String>) -> Self {
        ApiRequest {
            method,
            uri: uri.into(),
            paths: HashMap::new(),
            queries: Vec::new(),
            body: None,
        }
    }

    pub fn path(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.paths.insert(key.into(), value.into());
        self
    }

    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.queries.push((key.into(), value.into()));
        self
    }

    pub fn body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Substitute `:name` path placeholders in the URI with the value from
    /// `paths`.  This produces the concrete request path the lark SDK would
    /// dispatch.  Placeholders without a matching path entry are left intact.
    pub fn resolved_uri(&self) -> String {
        let mut out = String::new();
        for segment in self.uri.split('/') {
            if !out.is_empty() || self.uri.starts_with('/') {
                // Re-introduce the separator except before the very first
                // (possibly empty) segment when the URI starts with '/'.
            }
            out.push('/');
            if let Some(name) = segment.strip_prefix(':') {
                if let Some(v) = self.paths.get(name) {
                    out.push_str(v);
                    continue;
                }
            }
            out.push_str(segment);
        }
        // The loop above prepends a '/' for every segment; for a URI like
        // "/open-apis/..." the leading empty segment yields a doubled '/'.
        // Collapse a leading "//" back to "/".
        while out.starts_with("//") {
            out.remove(0);
        }
        out
    }
}

/// Outcome of an executed lark API request: `(code, msg, data)`.
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub code: i64,
    pub msg: String,
    pub data: Map<String, Value>,
}

impl ApiResponse {
    pub fn new(code: i64, msg: impl Into<String>, data: Map<String, Value>) -> Self {
        ApiResponse {
            code,
            msg: msg.into(),
            data,
        }
    }

    pub fn ok(&self) -> bool {
        self.code == 0
    }
}

/// Transport abstraction for executing lark API requests.
///
/// The Python module calls a live `lark_oapi` client via
/// `client.request(...)`; in Rust the gateway supplies an implementation
/// (e.g. backed by `reqwest::blocking`).  The high-level orchestration in
/// this module is generic over this trait.
pub trait CommentApi {
    /// Execute a request, returning the parsed `(code, msg, data)`.
    fn execute(&self, request: &ApiRequest) -> ApiResponse;

    /// Sleep helper used between retries (defaults to a real sleep).
    fn sleep_retry(&self) {
        std::thread::sleep(std::time::Duration::from_secs_f64(COMMENT_RETRY_DELAY_S));
    }
}

// ---------------------------------------------------------------------------
// URI constants
// ---------------------------------------------------------------------------

pub const REACTION_URI: &str = "/open-apis/drive/v2/files/:file_token/comments/reaction";
pub const BATCH_QUERY_META_URI: &str = "/open-apis/drive/v1/metas/batch_query";
pub const BATCH_QUERY_COMMENT_URI: &str =
    "/open-apis/drive/v1/files/:file_token/comments/batch_query";
pub const LIST_COMMENTS_URI: &str = "/open-apis/drive/v1/files/:file_token/comments";
pub const LIST_REPLIES_URI: &str =
    "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies";
pub const REPLY_COMMENT_URI: &str =
    "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies";
pub const ADD_COMMENT_URI: &str = "/open-apis/drive/v1/files/:file_token/new_comments";
pub const WIKI_GET_NODE_URI: &str = "/open-apis/wiki/v2/spaces/get_node";

pub const COMMENT_RETRY_LIMIT: usize = 6;
pub const COMMENT_RETRY_DELAY_S: f64 = 1.0;
pub const REPLY_CHUNK_SIZE: usize = 4000;
pub const PROMPT_TEXT_LIMIT: usize = 220;
pub const LOCAL_TIMELINE_LIMIT: usize = 20;
pub const WHOLE_TIMELINE_LIMIT: usize = 12;
pub const NO_REPLY_SENTINEL: &str = "NO_REPLY";

// ---------------------------------------------------------------------------
// Event parsing
// ---------------------------------------------------------------------------

/// Structured fields extracted from a `drive.notice.comment_add_v1` event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DriveCommentEvent {
    pub event_id: String,
    pub comment_id: String,
    pub reply_id: String,
    pub is_mentioned: bool,
    pub timestamp: String,
    pub file_token: String,
    pub file_type: String,
    pub notice_type: String,
    pub from_open_id: String,
    pub to_open_id: String,
}

/// Coerce a JSON value into a string the way Python's `str(x or "")` does:
/// `None`/`null`/`false`/`0`/empty become `""`, otherwise stringified.
fn str_or_empty(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => {
            // `str(s or "")` — empty string stays empty.
            s.clone()
        }
        Some(Value::Bool(b)) => {
            if *b {
                "True".to_string()
            } else {
                // `bool(False) or ""` -> "" then str("") -> ""
                String::new()
            }
        }
        Some(Value::Number(n)) => {
            // `n or ""` -> 0 is falsy -> ""
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Extract structured fields from a `drive.notice.comment_add_v1` payload.
///
/// `data` is the parsed event body; `parse_drive_comment_event` mirrors the
/// Python logic that reads `data.event` and the nested `notice_meta`.
/// Returns `None` when there is no `event` key (the malformed case).
pub fn parse_drive_comment_event(data: &Value) -> Option<DriveCommentEvent> {
    let event = data.get("event")?;
    if event.is_null() {
        return None;
    }
    let evt = event.as_object();
    let empty = Map::new();
    let evt = evt.unwrap_or(&empty);

    let notice_meta = evt
        .get("notice_meta")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let from_user = notice_meta
        .get("from_user_id")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let to_user = notice_meta
        .get("to_user_id")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let is_mentioned = match evt.get("is_mentioned") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Null) | None => false,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    };

    Some(DriveCommentEvent {
        event_id: str_or_empty(evt.get("event_id")),
        comment_id: str_or_empty(evt.get("comment_id")),
        reply_id: str_or_empty(evt.get("reply_id")),
        is_mentioned,
        timestamp: str_or_empty(evt.get("timestamp")),
        file_token: str_or_empty(notice_meta.get("file_token")),
        file_type: str_or_empty(notice_meta.get("file_type")),
        notice_type: str_or_empty(notice_meta.get("notice_type")),
        from_open_id: str_or_empty(from_user.get("open_id")),
        to_open_id: str_or_empty(to_user.get("open_id")),
    })
}

// ---------------------------------------------------------------------------
// Request builders (request-construction layer)
// ---------------------------------------------------------------------------

/// Build the request for an add/delete reaction call.
pub fn build_reaction_request(
    file_token: &str,
    file_type: &str,
    reply_id: &str,
    reaction_type: &str,
    action: &str,
) -> ApiRequest {
    ApiRequest::new(HttpMethod::Post, REACTION_URI)
        .path("file_token", file_token)
        .query("file_type", file_type)
        .body(json!({
            "action": action,
            "reply_id": reply_id,
            "reaction_type": reaction_type,
        }))
}

/// Build the document-meta batch_query request.
pub fn build_query_meta_request(file_token: &str, file_type: &str) -> ApiRequest {
    ApiRequest::new(HttpMethod::Post, BATCH_QUERY_META_URI).body(json!({
        "request_docs": [{"doc_token": file_token, "doc_type": file_type}],
        "with_url": true,
    }))
}

/// Build the comment batch_query request.
pub fn build_batch_query_comment_request(
    file_token: &str,
    file_type: &str,
    comment_id: &str,
) -> ApiRequest {
    ApiRequest::new(HttpMethod::Post, BATCH_QUERY_COMMENT_URI)
        .path("file_token", file_token)
        .query("file_type", file_type)
        .query("user_id_type", "open_id")
        .body(json!({ "comment_ids": [comment_id] }))
}

/// Build a list-whole-comments request for a given page.
pub fn build_list_whole_comments_request(
    file_token: &str,
    file_type: &str,
    page_token: &str,
) -> ApiRequest {
    let mut req = ApiRequest::new(HttpMethod::Get, LIST_COMMENTS_URI)
        .path("file_token", file_token)
        .query("file_type", file_type)
        .query("is_whole", "true")
        .query("page_size", "100")
        .query("user_id_type", "open_id");
    if !page_token.is_empty() {
        req = req.query("page_token", page_token);
    }
    req
}

/// Build a list-replies request for a given comment thread page.
pub fn build_list_replies_request(
    file_token: &str,
    file_type: &str,
    comment_id: &str,
    page_token: &str,
) -> ApiRequest {
    let mut req = ApiRequest::new(HttpMethod::Get, LIST_REPLIES_URI)
        .path("file_token", file_token)
        .path("comment_id", comment_id)
        .query("file_type", file_type)
        .query("page_size", "100")
        .query("user_id_type", "open_id");
    if !page_token.is_empty() {
        req = req.query("page_token", page_token);
    }
    req
}

/// Build a reply-to-local-comment request (text already sanitized by caller).
pub fn build_reply_comment_request(
    file_token: &str,
    file_type: &str,
    comment_id: &str,
    sanitized_text: &str,
) -> ApiRequest {
    ApiRequest::new(HttpMethod::Post, REPLY_COMMENT_URI)
        .path("file_token", file_token)
        .path("comment_id", comment_id)
        .query("file_type", file_type)
        .body(json!({
            "content": {
                "elements": [
                    {"type": "text_run", "text_run": {"text": sanitized_text}},
                ]
            }
        }))
}

/// Build an add-whole-comment request (text already sanitized by caller).
pub fn build_add_whole_comment_request(
    file_token: &str,
    file_type: &str,
    sanitized_text: &str,
) -> ApiRequest {
    ApiRequest::new(HttpMethod::Post, ADD_COMMENT_URI)
        .path("file_token", file_token)
        .body(json!({
            "file_type": file_type,
            "reply_elements": [
                {"type": "text", "text": sanitized_text},
            ],
        }))
}

/// Build the wiki get_node request (forward resolution of a wiki token).
pub fn build_wiki_get_node_request(wiki_token: &str) -> ApiRequest {
    ApiRequest::new(HttpMethod::Get, WIKI_GET_NODE_URI).query("token", wiki_token)
}

/// Build the wiki get_node reverse-lookup request (obj_token -> node).
pub fn build_wiki_reverse_lookup_request(obj_type: &str, obj_token: &str) -> ApiRequest {
    ApiRequest::new(HttpMethod::Get, WIKI_GET_NODE_URI)
        .query("token", obj_token)
        .query("obj_type", obj_type)
}

// ---------------------------------------------------------------------------
// API call layer (uses the CommentApi transport)
// ---------------------------------------------------------------------------

/// Add or delete an emoji reaction on a comment reply. Returns success.
pub fn update_comment_reaction<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    reply_id: &str,
    reaction_type: &str,
    action: &str,
) -> bool {
    let req = build_reaction_request(file_token, file_type, reply_id, reaction_type, action);
    api.execute(&req).ok()
}

pub fn add_comment_reaction<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    reply_id: &str,
    reaction_type: &str,
) -> bool {
    update_comment_reaction(api, file_token, file_type, reply_id, reaction_type, "add")
}

pub fn delete_comment_reaction<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    reply_id: &str,
    reaction_type: &str,
) -> bool {
    update_comment_reaction(api, file_token, file_type, reply_id, reaction_type, "delete")
}

/// Result of a document meta query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentMeta {
    pub title: String,
    pub url: String,
    pub doc_type: String,
}

/// Parse the `data` of a batch_query meta response into a [`DocumentMeta`].
///
/// Mirrors the Python alternate-shape handling: `metas` may be a list, or a
/// dict keyed by token.
pub fn parse_document_meta(data: &Map<String, Value>, file_token: &str, file_type: &str) -> Option<DocumentMeta> {
    let metas = data.get("metas");
    let meta: Map<String, Value> = match metas {
        Some(Value::Array(arr)) => {
            if arr.is_empty() {
                return None;
            }
            arr[0].as_object().cloned().unwrap_or_default()
        }
        Some(Value::Object(map)) => {
            // Alternate shape: dict keyed by token.
            map.get(file_token)
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default()
        }
        _ => return None,
    };

    Some(DocumentMeta {
        title: meta
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        url: meta
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        doc_type: meta
            .get("doc_type")
            .and_then(|v| v.as_str())
            .unwrap_or(file_type)
            .to_string(),
    })
}

/// Fetch document title and URL via batch_query meta API.
pub fn query_document_meta<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
) -> DocumentMeta {
    let req = build_query_meta_request(file_token, file_type);
    let resp = api.execute(&req);
    if !resp.ok() {
        return DocumentMeta::default();
    }
    parse_document_meta(&resp.data, file_token, file_type).unwrap_or_default()
}

/// Fetch comment details via batch_query comment API, retrying on failure.
///
/// Returns the first item (as a JSON object) or an empty object.
pub fn batch_query_comment<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    comment_id: &str,
) -> Map<String, Value> {
    let req = build_batch_query_comment_request(file_token, file_type, comment_id);
    let mut data = Map::new();
    for attempt in 0..COMMENT_RETRY_LIMIT {
        let resp = api.execute(&req);
        if resp.ok() {
            data = resp.data;
            break;
        }
        if attempt < COMMENT_RETRY_LIMIT - 1 {
            api.sleep_retry();
        } else {
            return Map::new();
        }
    }

    if let Some(Value::Array(items)) = data.get("items") {
        if let Some(first) = items.first() {
            if let Some(obj) = first.as_object() {
                return obj.clone();
            }
        }
    }
    Map::new()
}

/// List all whole-document comments (paginated, up to 5 pages).
pub fn list_whole_comments<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
) -> Vec<Value> {
    let mut all_comments: Vec<Value> = Vec::new();
    let mut page_token = String::new();

    for _ in 0..5 {
        let req = build_list_whole_comments_request(file_token, file_type, &page_token);
        let resp = api.execute(&req);
        if !resp.ok() {
            break;
        }
        if let Some(Value::Array(items)) = resp.data.get("items") {
            all_comments.extend(items.iter().cloned());
        }
        if !truthy(resp.data.get("has_more")) {
            break;
        }
        page_token = resp
            .data
            .get("page_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if page_token.is_empty() {
            break;
        }
    }
    all_comments
}

/// List all replies in a comment thread (paginated, up to 5 pages), retrying
/// up to 6 times if `expect_reply_id` is set and not yet present.
pub fn list_comment_replies<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    comment_id: &str,
    expect_reply_id: &str,
) -> Vec<Value> {
    let mut all_replies: Vec<Value> = Vec::new();

    for attempt in 0..COMMENT_RETRY_LIMIT {
        all_replies = Vec::new();
        let mut page_token = String::new();
        let mut fetch_ok = true;

        for _ in 0..5 {
            let req =
                build_list_replies_request(file_token, file_type, comment_id, &page_token);
            let resp = api.execute(&req);
            if !resp.ok() {
                fetch_ok = false;
                break;
            }
            if let Some(Value::Array(items)) = resp.data.get("items") {
                all_replies.extend(items.iter().cloned());
            }
            if !truthy(resp.data.get("has_more")) {
                break;
            }
            page_token = resp
                .data
                .get("page_token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if page_token.is_empty() {
                break;
            }
        }

        if expect_reply_id.is_empty() || !fetch_ok {
            break;
        }
        let found = all_replies.iter().any(|r| {
            r.get("reply_id")
                .and_then(|v| v.as_str())
                .map(|s| s == expect_reply_id)
                .unwrap_or(false)
        });
        if found {
            break;
        }
        if attempt < COMMENT_RETRY_LIMIT - 1 {
            api.sleep_retry();
        }
    }

    all_replies
}

/// Escape characters not allowed in Feishu comment text_run content.
pub fn sanitize_comment_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Post a reply to a local comment thread. Returns `(success, code)`.
pub fn reply_to_comment<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    comment_id: &str,
    text: &str,
) -> (bool, i64) {
    let sanitized = sanitize_comment_text(text);
    let req = build_reply_comment_request(file_token, file_type, comment_id, &sanitized);
    let resp = api.execute(&req);
    (resp.ok(), resp.code)
}

/// Add a new whole-document comment. Returns `true` on success.
pub fn add_whole_comment<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    text: &str,
) -> bool {
    let sanitized = sanitize_comment_text(text);
    let req = build_add_whole_comment_request(file_token, file_type, &sanitized);
    api.execute(&req).ok()
}

// ---------------------------------------------------------------------------
// Text chunking
// ---------------------------------------------------------------------------

/// Split text into chunks for delivery, preferring line breaks.
///
/// Operates on Unicode scalar values (chars) to mirror Python `str` slicing.
pub fn chunk_text(text: &str, limit: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return vec![text.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;
    let n = chars.len();

    while start < n {
        let remaining = n - start;
        if remaining <= limit {
            chunks.push(chars[start..].iter().collect());
            break;
        }
        // Find last newline within [start, start+limit) (Python rfind on the
        // window text[0:limit] of the current `text`).
        let window_end = start + limit;
        let mut cut: Option<usize> = None;
        // rfind in window (exclusive of window_end)
        let mut i = window_end;
        while i > start {
            i -= 1;
            if chars[i] == '\n' {
                cut = Some(i);
                break;
            }
        }
        // Python: cut = text.rfind("\n", 0, limit); offset relative to window
        // start. cut <= 0 means absent or at index 0 -> use limit.
        let cut_rel = match cut {
            Some(idx) => idx - start, // relative offset within current window
            None => usize::MAX,
        };
        let abs_cut = if cut_rel == usize::MAX || cut_rel == 0 {
            window_end
        } else {
            start + cut_rel
        };
        chunks.push(chars[start..abs_cut].iter().collect());
        // text = text[cut:].lstrip("\n")
        let mut next = abs_cut;
        while next < n && chars[next] == '\n' {
            next += 1;
        }
        start = next;
        if start >= n {
            break;
        }
    }
    chunks
}

/// Plan describing how `deliver_comment_reply` should route a single chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkRoute {
    Whole,
    Local,
}

/// Route an agent reply to the correct API, chunking long text.
///
/// - Whole comment -> add_whole_comment
/// - Local comment -> reply_to_comment, fallback to add_whole_comment on 1069302
pub fn deliver_comment_reply<A: CommentApi>(
    api: &A,
    file_token: &str,
    file_type: &str,
    comment_id: &str,
    text: &str,
    is_whole: bool,
) -> bool {
    let chunks = chunk_text(text, REPLY_CHUNK_SIZE);
    let mut all_ok = true;
    let mut is_whole = is_whole;

    for chunk in &chunks {
        let ok = if is_whole {
            add_whole_comment(api, file_token, file_type, chunk)
        } else {
            let (success, code) = reply_to_comment(api, file_token, file_type, comment_id, chunk);
            if success {
                true
            } else if code == 1069302 {
                let res = add_whole_comment(api, file_token, file_type, chunk);
                is_whole = true; // subsequent chunks also use add_comment
                res
            } else {
                false
            }
        };
        if !ok {
            all_ok = false;
            break;
        }
    }
    all_ok
}

// ---------------------------------------------------------------------------
// Comment content extraction helpers
// ---------------------------------------------------------------------------

/// Parse a reply's `content`: it may be an object, or a JSON string that
/// needs decoding. Returns the elements array (possibly empty) plus, when the
/// content was a string that failed to parse, the raw string fallback.
fn content_object(reply: &Value) -> Result<Value, String> {
    let content = reply.get("content").cloned().unwrap_or(Value::Null);
    match content {
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(v) => Ok(v),
            Err(_) => Err(s),
        },
        Value::Object(_) => Ok(content),
        Value::Null => Ok(json!({})),
        other => Ok(other),
    }
}

fn elements_of(content: &Value) -> Vec<Value> {
    content
        .get("elements")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

fn elem_str(elem: &Value, group: &str, field: &str) -> String {
    elem.get(group)
        .and_then(|g| g.get(field))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Extract plain text from a comment reply's content structure.
pub fn extract_reply_text(reply: &Value) -> String {
    let content = match content_object(reply) {
        Ok(c) => c,
        Err(raw) => return raw, // string content that wasn't JSON
    };
    let mut parts: Vec<String> = Vec::new();
    for elem in elements_of(&content) {
        match elem.get("type").and_then(|v| v.as_str()) {
            Some("text_run") => parts.push(elem_str(&elem, "text_run", "text")),
            Some("docs_link") => parts.push(elem_str(&elem, "docs_link", "url")),
            Some("person") => {
                let uid = elem
                    .get("person")
                    .and_then(|p| p.get("user_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                parts.push(format!("@{}", uid));
            }
            _ => {}
        }
    }
    parts.concat()
}

/// Extract `user_id` from a reply dict (open_id preferred).
pub fn get_reply_user_id(reply: &Value) -> String {
    match reply.get("user_id") {
        Some(Value::Object(map)) => {
            let open = map.get("open_id").and_then(|v| v.as_str()).unwrap_or("");
            if !open.is_empty() {
                open.to_string()
            } else {
                map.get("user_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
        }
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Extract semantic text from a reply, stripping self @mentions and
/// collapsing whitespace (Python `" ".join(text.split()).strip()`).
pub fn extract_semantic_text(reply: &Value, self_open_id: &str) -> String {
    let content = match content_object(reply) {
        Ok(c) => c,
        Err(raw) => return raw,
    };
    let mut parts: Vec<String> = Vec::new();
    for elem in elements_of(&content) {
        match elem.get("type").and_then(|v| v.as_str()) {
            Some("person") => {
                let uid = elem
                    .get("person")
                    .and_then(|p| p.get("user_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !self_open_id.is_empty() && uid == self_open_id {
                    continue;
                }
                parts.push(format!("@{}", uid));
            }
            Some("text_run") => parts.push(elem_str(&elem, "text_run", "text")),
            Some("docs_link") => parts.push(elem_str(&elem, "docs_link", "url")),
            _ => {}
        }
    }
    let joined = parts.concat();
    // " ".join(joined.split()).strip()
    joined.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Document link parsing and wiki resolution
// ---------------------------------------------------------------------------

/// A parsed/resolved document link extracted from comment replies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocLink {
    pub url: String,
    pub doc_type: String,
    pub token: String,
    pub resolved_type: Option<String>,
    pub resolved_token: Option<String>,
}

/// Match a feishu/lark document URL, returning `(doc_type, token)`.
///
/// Equivalent to the Python `_FEISHU_DOC_URL_RE.search`.
pub fn match_feishu_doc_url(url: &str) -> Option<(String, String)> {
    // Hosts
    const HOSTS: &[&str] = &[
        "feishu.cn",
        "larkoffice.com",
        "larksuite.com",
        "lark.suite.com",
    ];
    const TYPES: &[&str] = &[
        "wiki", "doc", "docx", "sheet", "sheets", "slides", "mindnote", "bitable", "base", "file",
    ];

    // Find a host occurrence, then attempt to parse "/<type>/<token>".
    for host in HOSTS {
        let mut search_from = 0usize;
        while let Some(pos) = url[search_from..].find(host) {
            let host_abs = search_from + pos;
            let after = &url[host_abs + host.len()..];
            if let Some(parsed) = parse_type_token(after, TYPES) {
                return Some(parsed);
            }
            search_from = host_abs + host.len();
            if search_from >= url.len() {
                break;
            }
        }
    }
    None
}

fn parse_type_token(after_host: &str, types: &[&str]) -> Option<(String, String)> {
    // Expect "/<doc_type>/<token>" immediately following the host.
    let rest = after_host.strip_prefix('/')?;
    // doc_type: one of the alternatives
    let mut matched_type: Option<&str> = None;
    for t in types {
        if rest.starts_with(t) {
            // Must be followed by '/'
            let after_type = &rest[t.len()..];
            if after_type.starts_with('/') {
                // Prefer the longest match (e.g. "sheets" over "sheet",
                // "docx" over "doc"); keep iterating to find longer one.
                if matched_type.map(|m| m.len()).unwrap_or(0) < t.len() {
                    matched_type = Some(t);
                }
            }
        }
    }
    let doc_type = matched_type?;
    let after_type = &rest[doc_type.len() + 1..]; // skip type and '/'
    // token: [A-Za-z0-9_-]{10,40}
    let token: String = after_type
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if token.len() < 10 || token.len() > 40 {
        return None;
    }
    Some((doc_type.to_string(), token))
}

/// Extract unique document links from a list of comment replies.
pub fn extract_docs_links(replies: &[Value]) -> Vec<DocLink> {
    let mut seen_tokens: HashSet<String> = HashSet::new();
    let mut links: Vec<DocLink> = Vec::new();
    for reply in replies {
        let content = match content_object(reply) {
            Ok(c) => c,
            Err(_) => continue, // string content that wasn't JSON -> skip
        };
        for elem in elements_of(&content) {
            let etype = elem.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if etype != "docs_link" && etype != "link" {
                continue;
            }
            let link_data = elem
                .get("docs_link")
                .or_else(|| elem.get("link"))
                .cloned()
                .unwrap_or(Value::Null);
            let url = link_data
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if url.is_empty() {
                continue;
            }
            let (doc_type, token) = match match_feishu_doc_url(&url) {
                Some(t) => t,
                None => continue,
            };
            if seen_tokens.contains(&token) {
                continue;
            }
            seen_tokens.insert(token.clone());
            links.push(DocLink {
                url,
                doc_type,
                token,
                resolved_type: None,
                resolved_token: None,
            });
        }
    }
    links
}

/// Reverse-lookup: given an obj_token, find its wiki node_token.
/// Returns the wiki_token if non-empty, else `None`.
pub fn reverse_lookup_wiki_token<A: CommentApi>(
    api: &A,
    obj_type: &str,
    obj_token: &str,
) -> Option<String> {
    let req = build_wiki_reverse_lookup_request(obj_type, obj_token);
    let resp = api.execute(&req);
    if resp.ok() {
        let wiki_token = resp
            .data
            .get("node")
            .and_then(|n| n.get("node_token"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if wiki_token.is_empty() {
            None
        } else {
            Some(wiki_token.to_string())
        }
    } else {
        None
    }
}

/// Resolve wiki links to their underlying document type and token, mutating
/// the `links` in place (sets `resolved_type`/`resolved_token`).
pub fn resolve_wiki_nodes<A: CommentApi>(api: &A, links: &mut [DocLink]) {
    let has_wiki = links.iter().any(|l| l.doc_type == "wiki");
    if !has_wiki {
        return;
    }
    for link in links.iter_mut() {
        if link.doc_type != "wiki" {
            continue;
        }
        let req = build_wiki_get_node_request(&link.token);
        let resp = api.execute(&req);
        if resp.ok() {
            let node = resp.data.get("node");
            let resolved_type = node
                .and_then(|n| n.get("obj_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let resolved_token = node
                .and_then(|n| n.get("obj_token"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !resolved_type.is_empty() && !resolved_token.is_empty() {
                link.resolved_type = Some(resolved_type.to_string());
                link.resolved_token = Some(resolved_token.to_string());
            }
        }
    }
}

/// Format resolved document links for prompt embedding.
pub fn format_referenced_docs(links: &[DocLink], current_file_token: &str) -> String {
    if links.is_empty() {
        return String::new();
    }
    let mut lines: Vec<String> = vec![String::new(), "Referenced documents in comments:".to_string()];
    for link in links {
        let rtype = link.resolved_type.as_deref().unwrap_or(&link.doc_type);
        let rtoken = link.resolved_token.as_deref().unwrap_or(&link.token);
        let is_current = rtoken == current_file_token;
        let suffix = if is_current {
            " (same as current document)"
        } else {
            ""
        };
        let url_trunc: String = link.url.chars().take(80).collect();
        lines.push(format!("- {}:{}{} ({})", rtype, rtoken, suffix, url_trunc));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// A single timeline entry: `(user_id, text, is_self)`.
pub type TimelineEntry = (String, String, bool);

/// Truncate text for prompt embedding (operates on chars).
pub fn truncate(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_string();
    }
    let mut s: String = chars[..limit].iter().collect();
    s.push_str("...");
    s
}

/// Select up to LOCAL_TIMELINE_LIMIT entries centered on `target_index`.
/// `target_index` of -1 (encoded as None) means "no target".
pub fn select_local_timeline(
    timeline: &[TimelineEntry],
    target_index: i64,
) -> Vec<TimelineEntry> {
    if timeline.len() <= LOCAL_TIMELINE_LIMIT {
        return timeline.to_vec();
    }
    let n = timeline.len() as i64;
    let mut selected: HashSet<i64> = HashSet::new();
    selected.insert(0);
    selected.insert(n - 1);
    if (0..n).contains(&target_index) {
        selected.insert(target_index);
    }
    let mut budget = LOCAL_TIMELINE_LIMIT as i64 - selected.len() as i64;
    let mut lo = target_index - 1;
    let mut hi = target_index + 1;
    while budget > 0 && (lo >= 0 || hi < n) {
        if lo >= 0 && !selected.contains(&lo) {
            selected.insert(lo);
            budget -= 1;
        }
        lo -= 1;
        if budget > 0 && hi < n && !selected.contains(&hi) {
            selected.insert(hi);
            budget -= 1;
        }
        hi += 1;
    }
    let mut idxs: Vec<i64> = selected.into_iter().collect();
    idxs.sort_unstable();
    idxs.into_iter().map(|i| timeline[i as usize].clone()).collect()
}

/// Select up to WHOLE_TIMELINE_LIMIT entries for whole-doc comments.
pub fn select_whole_timeline(
    timeline: &[TimelineEntry],
    current_index: i64,
    nearest_self_index: i64,
) -> Vec<TimelineEntry> {
    if timeline.len() <= WHOLE_TIMELINE_LIMIT {
        return timeline.to_vec();
    }
    let n = timeline.len() as i64;
    let mut selected: HashSet<i64> = HashSet::new();
    if (0..n).contains(&current_index) {
        selected.insert(current_index);
    }
    if (0..n).contains(&nearest_self_index) {
        selected.insert(nearest_self_index);
    }
    let mut budget = WHOLE_TIMELINE_LIMIT as i64 - selected.len() as i64;
    let mut lo = current_index - 1;
    let mut hi = current_index + 1;
    while budget > 0 && (lo >= 0 || hi < n) {
        if lo >= 0 && !selected.contains(&lo) {
            selected.insert(lo);
            budget -= 1;
        }
        lo -= 1;
        if budget > 0 && hi < n && !selected.contains(&hi) {
            selected.insert(hi);
            budget -= 1;
        }
        hi += 1;
    }
    if selected.is_empty() {
        // Fallback: take last N entries.
        let start = timeline.len().saturating_sub(WHOLE_TIMELINE_LIMIT);
        return timeline[start..].to_vec();
    }
    let mut idxs: Vec<i64> = selected.into_iter().collect();
    idxs.sort_unstable();
    idxs.into_iter().map(|i| timeline[i as usize].clone()).collect()
}

pub const COMMON_INSTRUCTIONS: &str = "This is a Feishu document comment thread, not an IM chat.\n\
Do NOT call feishu_drive_add_comment or feishu_drive_reply_comment yourself.\n\
Your reply will be posted automatically. Just output the reply text.\n\
Use the thread timeline above as the main context.\n\
If the quoted content is not enough, use feishu_doc_read to read nearby context.\n\
The quoted content is your primary anchor — insert/summarize/explain requests are about it.\n\
Do not guess document content you haven't read.\n\
Reply in the same language as the user's comment unless they request otherwise.\n\
Use plain text only. Do not use Markdown, headings, bullet lists, tables, or code blocks.\n\
Do not show your reasoning process. Do not start with \"I will\", \"Let me\", or \"I'll first\".\n\
Output only the final user-facing reply.\n\
If no reply is needed, output exactly NO_REPLY.";

/// Inputs for building a local (quoted-text) comment prompt.
pub struct LocalPromptInput<'a> {
    pub doc_title: &'a str,
    pub doc_url: &'a str,
    pub file_token: &'a str,
    pub file_type: &'a str,
    pub comment_id: &'a str,
    pub quote_text: &'a str,
    pub root_comment_text: &'a str,
    pub target_reply_text: &'a str,
    pub timeline: &'a [TimelineEntry],
    pub self_open_id: &'a str,
    pub target_index: i64,
    pub referenced_docs: &'a str,
}

/// Build the prompt for a local (quoted-text) comment.
pub fn build_local_comment_prompt(input: &LocalPromptInput) -> String {
    let selected = select_local_timeline(input.timeline, input.target_index);

    let mut lines: Vec<String> = vec![
        format!("The user added a reply in \"{}\".", input.doc_title),
        format!(
            "Current user comment text: \"{}\"",
            truncate(input.target_reply_text, PROMPT_TEXT_LIMIT)
        ),
        format!(
            "Original comment text: \"{}\"",
            truncate(input.root_comment_text, PROMPT_TEXT_LIMIT)
        ),
        format!("Quoted content: \"{}\"", truncate(input.quote_text, 500)),
        "This comment mentioned you (@mention is for routing, not task content).".to_string(),
        format!("Document link: {}", input.doc_url),
        "Current commented document:".to_string(),
        format!("- file_type={}", input.file_type),
        format!("- file_token={}", input.file_token),
        format!("- comment_id={}", input.comment_id),
        String::new(),
        format!(
            "Current comment card timeline ({}/{} entries):",
            selected.len(),
            input.timeline.len()
        ),
    ];

    for (user_id, text, is_self) in &selected {
        let marker = if *is_self { " <-- YOU" } else { "" };
        lines.push(format!(
            "[{}] {}{}",
            user_id,
            truncate(text, PROMPT_TEXT_LIMIT),
            marker
        ));
    }

    if !input.referenced_docs.is_empty() {
        lines.push(input.referenced_docs.to_string());
    }

    lines.push(String::new());
    lines.push(COMMON_INSTRUCTIONS.to_string());
    lines.join("\n")
}

/// Inputs for building a whole-document comment prompt.
pub struct WholePromptInput<'a> {
    pub doc_title: &'a str,
    pub doc_url: &'a str,
    pub file_token: &'a str,
    pub file_type: &'a str,
    pub comment_text: &'a str,
    pub timeline: &'a [TimelineEntry],
    pub self_open_id: &'a str,
    pub current_index: i64,
    pub nearest_self_index: i64,
    pub referenced_docs: &'a str,
}

/// Build the prompt for a whole-document comment.
pub fn build_whole_comment_prompt(input: &WholePromptInput) -> String {
    let selected =
        select_whole_timeline(input.timeline, input.current_index, input.nearest_self_index);

    let mut lines: Vec<String> = vec![
        format!("The user added a comment in \"{}\".", input.doc_title),
        format!(
            "Current user comment text: \"{}\"",
            truncate(input.comment_text, PROMPT_TEXT_LIMIT)
        ),
        "This is a whole-document comment.".to_string(),
        "This comment mentioned you (@mention is for routing, not task content).".to_string(),
        format!("Document link: {}", input.doc_url),
        "Current commented document:".to_string(),
        format!("- file_type={}", input.file_type),
        format!("- file_token={}", input.file_token),
        String::new(),
        format!(
            "Whole-document comment timeline ({}/{} entries):",
            selected.len(),
            input.timeline.len()
        ),
    ];

    for (user_id, text, is_self) in &selected {
        let marker = if *is_self { " <-- YOU" } else { "" };
        lines.push(format!(
            "[{}] {}{}",
            user_id,
            truncate(text, PROMPT_TEXT_LIMIT),
            marker
        ));
    }

    if !input.referenced_docs.is_empty() {
        lines.push(input.referenced_docs.to_string());
    }

    lines.push(String::new());
    lines.push(COMMON_INSTRUCTIONS.to_string());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Timeline building (the is_whole branches of handle_drive_comment_event)
// ---------------------------------------------------------------------------

/// Decode a `reply_list` field that may be an object or a JSON string into the
/// list of replies it contains.
fn replies_of_reply_list(reply_list: &Value) -> Vec<Value> {
    let obj = match reply_list {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or(json!({})),
        other => other.clone(),
    };
    obj.get("replies")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
}

/// Result of building the whole-document timeline.
pub struct WholeTimelineResult {
    pub timeline: Vec<TimelineEntry>,
    pub current_text: String,
    pub current_index: i64,
    pub nearest_self_index: i64,
    /// All raw replies across whole comments (for doc-link extraction).
    pub all_raw_replies: Vec<Value>,
}

/// Build the whole-document timeline from a list of whole comments.
pub fn build_whole_timeline(
    whole_comments: &[Value],
    self_open_id: &str,
    from_open_id: &str,
) -> WholeTimelineResult {
    let mut timeline: Vec<TimelineEntry> = Vec::new();
    let mut current_text = String::new();
    let mut current_index: i64 = -1;
    let mut nearest_self_index: i64 = -1;

    for wc in whole_comments {
        let reply_list = wc.get("reply_list").cloned().unwrap_or(json!({}));
        let replies = replies_of_reply_list(&reply_list);
        for r in &replies {
            let uid = get_reply_user_id(r);
            let text = extract_reply_text(r);
            let is_self = if !self_open_id.is_empty() {
                uid == self_open_id
            } else {
                false
            };
            let idx = timeline.len() as i64;
            timeline.push((uid.clone(), text, is_self));
            if uid == from_open_id {
                current_text = extract_semantic_text(r, self_open_id);
                current_index = idx;
            }
            if is_self {
                nearest_self_index = idx;
            }
        }
    }

    if current_text.is_empty() {
        for i in (0..timeline.len()).rev() {
            let (_uid, text, is_self) = &timeline[i];
            if !*is_self {
                current_text = text.clone();
                current_index = i as i64;
                break;
            }
        }
    }

    let mut all_raw_replies: Vec<Value> = Vec::new();
    for wc in whole_comments {
        let rl = wc.get("reply_list").cloned().unwrap_or(json!({}));
        all_raw_replies.extend(replies_of_reply_list(&rl));
    }

    WholeTimelineResult {
        timeline,
        current_text,
        current_index,
        nearest_self_index,
        all_raw_replies,
    }
}

/// Result of building the local comment thread timeline.
pub struct LocalTimelineResult {
    pub timeline: Vec<TimelineEntry>,
    pub root_text: String,
    pub target_text: String,
    pub target_index: i64,
}

/// Build the local comment timeline from a list of thread replies.
pub fn build_local_timeline(
    replies: &[Value],
    self_open_id: &str,
    from_open_id: &str,
    reply_id: &str,
) -> LocalTimelineResult {
    let mut timeline: Vec<TimelineEntry> = Vec::new();
    let mut root_text = String::new();
    let mut target_text = String::new();
    let mut target_index: i64 = -1;

    for (i, r) in replies.iter().enumerate() {
        let uid = get_reply_user_id(r);
        let text = extract_reply_text(r);
        let is_self = if !self_open_id.is_empty() {
            uid == self_open_id
        } else {
            false
        };
        timeline.push((uid.clone(), text, is_self));
        if i == 0 {
            root_text = extract_semantic_text(r, self_open_id);
        }
        let rid = r.get("reply_id").and_then(|v| v.as_str()).unwrap_or("");
        if !rid.is_empty() && rid == reply_id {
            target_text = extract_semantic_text(r, self_open_id);
            target_index = i as i64;
        }
    }

    if target_text.is_empty() && !timeline.is_empty() {
        for i in (0..timeline.len()).rev() {
            let (uid, text, _is_self) = &timeline[i];
            if uid == from_open_id {
                target_text = text.clone();
                target_index = i as i64;
                break;
            }
        }
    }

    LocalTimelineResult {
        timeline,
        root_text,
        target_text,
        target_index,
    }
}

// ---------------------------------------------------------------------------
// Session cache for cross-card memory within the same document
// ---------------------------------------------------------------------------

pub const SESSION_MAX_MESSAGES: usize = 50;
pub const SESSION_TTL_S: u64 = 3600;

struct SessionEntry {
    messages: Vec<Value>,
    last_access: f64,
}

static SESSION_CACHE: Mutex<Option<HashMap<String, SessionEntry>>> = Mutex::new(None);

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Session cache key for a document.
pub fn session_key(file_type: &str, file_token: &str) -> String {
    format!("comment-doc:{}:{}", file_type, file_token)
}

/// Load conversation history for a document session (honours TTL).
pub fn load_session_history(key: &str) -> Vec<Value> {
    let mut guard = SESSION_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    let now = now_secs();
    let expired = match cache.get(key) {
        None => return Vec::new(),
        Some(entry) => now - entry.last_access > SESSION_TTL_S as f64,
    };
    if expired {
        cache.remove(key);
        return Vec::new();
    }
    let entry = cache.get_mut(key).unwrap();
    entry.last_access = now;
    entry.messages.clone()
}

/// Save conversation history for a document session (keeps last N messages,
/// strips system messages and entries with empty content).
pub fn save_session_history(key: &str, messages: &[Value]) {
    let mut cleaned: Vec<Value> = messages
        .iter()
        .filter(|m| {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let has_content = truthy(m.get("content"));
            (role == "user" || role == "assistant") && has_content
        })
        .cloned()
        .collect();
    if cleaned.len() > SESSION_MAX_MESSAGES {
        let start = cleaned.len() - SESSION_MAX_MESSAGES;
        cleaned = cleaned[start..].to_vec();
    }
    let mut guard = SESSION_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    cache.insert(
        key.to_string(),
        SessionEntry {
            messages: cleaned,
            last_access: now_secs(),
        },
    );
}

/// Clear the entire session cache (test helper / lifecycle reset).
pub fn clear_session_cache() {
    let mut guard = SESSION_CACHE.lock().unwrap();
    *guard = Some(HashMap::new());
}

// ---------------------------------------------------------------------------
// Event filtering / orchestration helpers
// ---------------------------------------------------------------------------

/// notice_type values that are processed; anything else is skipped.
pub fn is_allowed_notice_type(notice_type: &str) -> bool {
    notice_type == "add_comment" || notice_type == "add_reply"
}

/// Reason an event was filtered out before processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterDecision {
    /// Event passes all filters and should be processed.
    Process,
    /// Self-authored event.
    SelfAuthored,
    /// Not addressed to this bot.
    NotAddressed,
    /// Disallowed notice type.
    DisallowedNoticeType,
    /// Missing required fields.
    MissingFields,
}

/// Apply the early filters from `handle_drive_comment_event`:
/// self-reply, receiver check, notice_type, required fields.
pub fn filter_comment_event(
    parsed: &DriveCommentEvent,
    self_open_id: &str,
) -> FilterDecision {
    let from_open_id = &parsed.from_open_id;
    let to_open_id = &parsed.to_open_id;
    let notice_type = &parsed.notice_type;

    if !from_open_id.is_empty() && !self_open_id.is_empty() && from_open_id == self_open_id {
        return FilterDecision::SelfAuthored;
    }
    if to_open_id.is_empty() || (!self_open_id.is_empty() && to_open_id != self_open_id) {
        return FilterDecision::NotAddressed;
    }
    if !notice_type.is_empty() && !is_allowed_notice_type(notice_type) {
        return FilterDecision::DisallowedNoticeType;
    }
    if parsed.file_token.is_empty() || parsed.file_type.is_empty() || parsed.comment_id.is_empty() {
        return FilterDecision::MissingFields;
    }
    FilterDecision::Process
}

/// Decide whether an agent response should be delivered.
/// Mirrors `if not response or _NO_REPLY_SENTINEL in response`.
pub fn should_deliver(response: &str) -> bool {
    !response.is_empty() && !response.contains(NO_REPLY_SENTINEL)
}

// ---------------------------------------------------------------------------
// shared: Python truthiness for serde_json::Value
// ---------------------------------------------------------------------------

/// Python-style truthiness for an optional JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event: Value) -> Value {
        json!({ "event": event })
    }

    #[test]
    fn parse_event_basic() {
        let data = ev(json!({
            "event_id": "e1",
            "comment_id": "c1",
            "reply_id": "r1",
            "is_mentioned": true,
            "timestamp": "123",
            "notice_meta": {
                "file_token": "ft",
                "file_type": "docx",
                "notice_type": "add_reply",
                "from_user_id": {"open_id": "ou_from"},
                "to_user_id": {"open_id": "ou_to"},
            }
        }));
        let p = parse_drive_comment_event(&data).unwrap();
        assert_eq!(p.event_id, "e1");
        assert_eq!(p.comment_id, "c1");
        assert_eq!(p.reply_id, "r1");
        assert!(p.is_mentioned);
        assert_eq!(p.file_token, "ft");
        assert_eq!(p.file_type, "docx");
        assert_eq!(p.notice_type, "add_reply");
        assert_eq!(p.from_open_id, "ou_from");
        assert_eq!(p.to_open_id, "ou_to");
    }

    #[test]
    fn parse_event_no_event_key() {
        assert!(parse_drive_comment_event(&json!({})).is_none());
        assert!(parse_drive_comment_event(&json!({"event": null})).is_none());
    }

    #[test]
    fn parse_event_missing_meta() {
        let p = parse_drive_comment_event(&ev(json!({"event_id": "x"}))).unwrap();
        assert_eq!(p.event_id, "x");
        assert_eq!(p.file_token, "");
        assert_eq!(p.from_open_id, "");
    }

    #[test]
    fn sanitize_escapes() {
        assert_eq!(
            sanitize_comment_text("a & b < c > d"),
            "a &amp; b &lt; c &gt; d"
        );
        // Order matters: & first so escaped entities are not double-escaped.
        assert_eq!(sanitize_comment_text("<&>"), "&lt;&amp;&gt;");
    }

    #[test]
    fn chunk_short_text() {
        assert_eq!(chunk_text("hello", 4000), vec!["hello".to_string()]);
    }

    #[test]
    fn chunk_prefers_newline() {
        let text = "aaa\nbbb\nccc";
        // limit 5 -> first cut at last newline within 5 chars: "aaa\nbbb" window
        let chunks = chunk_text(text, 5);
        // window "aaa\nb", rfind newline -> index 3; chunk "aaa", remainder "bbb\nccc"
        assert_eq!(chunks[0], "aaa");
        assert_eq!(chunks.join("|"), "aaa|bbb|ccc");
    }

    #[test]
    fn chunk_no_newline_hard_cut() {
        let text = "abcdefghij";
        let chunks = chunk_text(text, 4);
        assert_eq!(chunks, vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn extract_reply_text_elements() {
        let reply = json!({
            "content": {
                "elements": [
                    {"type": "text_run", "text_run": {"text": "Hello "}},
                    {"type": "person", "person": {"user_id": "u1"}},
                    {"type": "docs_link", "docs_link": {"url": "http://x"}},
                ]
            }
        });
        assert_eq!(extract_reply_text(&reply), "Hello @u1http://x");
    }

    #[test]
    fn extract_reply_text_string_content() {
        // content is a JSON string
        let reply = json!({
            "content": "{\"elements\":[{\"type\":\"text_run\",\"text_run\":{\"text\":\"hi\"}}]}"
        });
        assert_eq!(extract_reply_text(&reply), "hi");
        // content is a non-JSON string -> returned as-is
        let reply2 = json!({"content": "not json"});
        assert_eq!(extract_reply_text(&reply2), "not json");
    }

    #[test]
    fn semantic_strips_self_mention_and_whitespace() {
        let reply = json!({
            "content": {
                "elements": [
                    {"type": "person", "person": {"user_id": "self"}},
                    {"type": "text_run", "text_run": {"text": "  do   this  "}},
                    {"type": "person", "person": {"user_id": "other"}},
                ]
            }
        });
        // self mention skipped; whitespace collapsed
        assert_eq!(extract_semantic_text(&reply, "self"), "do this @other");
    }

    #[test]
    fn reply_user_id_variants() {
        assert_eq!(
            get_reply_user_id(&json!({"user_id": {"open_id": "o1"}})),
            "o1"
        );
        assert_eq!(
            get_reply_user_id(&json!({"user_id": {"user_id": "u2"}})),
            "u2"
        );
        assert_eq!(get_reply_user_id(&json!({"user_id": "plain"})), "plain");
        assert_eq!(get_reply_user_id(&json!({})), "");
    }

    #[test]
    fn doc_url_matching() {
        let (t, tok) =
            match_feishu_doc_url("https://example.feishu.cn/docx/abcdefghij1234").unwrap();
        assert_eq!(t, "docx");
        assert_eq!(tok, "abcdefghij1234");

        // sheets preferred over sheet (longest-match)
        let (t2, _) =
            match_feishu_doc_url("https://x.larksuite.com/sheets/abcdefghij12").unwrap();
        assert_eq!(t2, "sheets");

        // token too short
        assert!(match_feishu_doc_url("https://x.feishu.cn/doc/short").is_none());
        // unknown host
        assert!(match_feishu_doc_url("https://other.com/doc/abcdefghij").is_none());
    }

    #[test]
    fn extract_links_dedup() {
        let replies = vec![
            json!({"content": {"elements": [
                {"type": "docs_link", "docs_link": {"url": "https://a.feishu.cn/wiki/abcdefghij12"}}
            ]}}),
            json!({"content": {"elements": [
                {"type": "link", "link": {"url": "https://a.feishu.cn/wiki/abcdefghij12"}}
            ]}}),
            json!({"content": {"elements": [
                {"type": "docs_link", "docs_link": {"url": "https://a.feishu.cn/docx/zzzzzzzzzz99"}}
            ]}}),
        ];
        let links = extract_docs_links(&replies);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].doc_type, "wiki");
        assert_eq!(links[1].doc_type, "docx");
    }

    #[test]
    fn format_refs() {
        let links = vec![
            DocLink {
                url: "https://a.feishu.cn/docx/tok1234567".to_string(),
                doc_type: "docx".to_string(),
                token: "tok1234567".to_string(),
                resolved_type: None,
                resolved_token: None,
            },
            DocLink {
                url: "https://a.feishu.cn/wiki/wikitoken99".to_string(),
                doc_type: "wiki".to_string(),
                token: "wikitoken99".to_string(),
                resolved_type: Some("docx".to_string()),
                resolved_token: Some("cur".to_string()),
            },
        ];
        let out = format_referenced_docs(&links, "cur");
        assert!(out.contains("- docx:tok1234567 ("));
        assert!(out.contains("- docx:cur (same as current document)"));
    }

    #[test]
    fn truncate_chars() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 3), "hel...");
    }

    #[test]
    fn local_timeline_selection_keeps_endpoints() {
        let timeline: Vec<TimelineEntry> = (0..30)
            .map(|i| (format!("u{}", i), format!("t{}", i), false))
            .collect();
        let sel = select_local_timeline(&timeline, 15);
        assert_eq!(sel.len(), LOCAL_TIMELINE_LIMIT);
        assert_eq!(sel.first().unwrap().0, "u0");
        assert_eq!(sel.last().unwrap().0, "u29");
        assert!(sel.iter().any(|(u, _, _)| u == "u15"));
    }

    #[test]
    fn local_timeline_small_unchanged() {
        let timeline: Vec<TimelineEntry> =
            (0..5).map(|i| (format!("u{}", i), "t".to_string(), false)).collect();
        assert_eq!(select_local_timeline(&timeline, 2).len(), 5);
    }

    #[test]
    fn whole_timeline_selection() {
        let timeline: Vec<TimelineEntry> = (0..30)
            .map(|i| (format!("u{}", i), "t".to_string(), i == 3))
            .collect();
        let sel = select_whole_timeline(&timeline, 10, 3);
        assert_eq!(sel.len(), WHOLE_TIMELINE_LIMIT);
        assert!(sel.iter().any(|(u, _, _)| u == "u10"));
        assert!(sel.iter().any(|(u, _, _)| u == "u3"));
    }

    #[test]
    fn build_local_prompt_shape() {
        let timeline = vec![
            ("u1".to_string(), "first".to_string(), false),
            ("self".to_string(), "mine".to_string(), true),
        ];
        let input = LocalPromptInput {
            doc_title: "Doc",
            doc_url: "http://u",
            file_token: "ft",
            file_type: "docx",
            comment_id: "cid",
            quote_text: "quote",
            root_comment_text: "root",
            target_reply_text: "target",
            timeline: &timeline,
            self_open_id: "self",
            target_index: 1,
            referenced_docs: "",
        };
        let p = build_local_comment_prompt(&input);
        assert!(p.contains("The user added a reply in \"Doc\"."));
        assert!(p.contains("- file_token=ft"));
        assert!(p.contains("- comment_id=cid"));
        assert!(p.contains("[self] mine <-- YOU"));
        assert!(p.contains("Output only the final user-facing reply."));
    }

    #[test]
    fn build_whole_prompt_shape() {
        let timeline = vec![("u1".to_string(), "hi".to_string(), false)];
        let input = WholePromptInput {
            doc_title: "D",
            doc_url: "http://u",
            file_token: "ft",
            file_type: "docx",
            comment_text: "ctext",
            timeline: &timeline,
            self_open_id: "self",
            current_index: 0,
            nearest_self_index: -1,
            referenced_docs: "REFS",
        };
        let p = build_whole_comment_prompt(&input);
        assert!(p.contains("This is a whole-document comment."));
        assert!(p.contains("REFS"));
        assert!(!p.contains("- comment_id="));
    }

    #[test]
    fn filter_decisions() {
        let mut ev = DriveCommentEvent {
            file_token: "ft".into(),
            file_type: "docx".into(),
            comment_id: "c".into(),
            from_open_id: "other".into(),
            to_open_id: "me".into(),
            notice_type: "add_reply".into(),
            ..Default::default()
        };
        assert_eq!(filter_comment_event(&ev, "me"), FilterDecision::Process);

        ev.from_open_id = "me".into();
        assert_eq!(filter_comment_event(&ev, "me"), FilterDecision::SelfAuthored);

        ev.from_open_id = "other".into();
        ev.to_open_id = "someone".into();
        assert_eq!(filter_comment_event(&ev, "me"), FilterDecision::NotAddressed);

        ev.to_open_id = "me".into();
        ev.notice_type = "weird".into();
        assert_eq!(
            filter_comment_event(&ev, "me"),
            FilterDecision::DisallowedNoticeType
        );

        ev.notice_type = "add_comment".into();
        ev.file_token = "".into();
        assert_eq!(filter_comment_event(&ev, "me"), FilterDecision::MissingFields);
    }

    #[test]
    fn should_deliver_logic() {
        assert!(should_deliver("hello"));
        assert!(!should_deliver(""));
        assert!(!should_deliver("NO_REPLY"));
        assert!(!should_deliver("prefix NO_REPLY suffix"));
    }

    #[test]
    fn resolved_uri_substitution() {
        let req = build_list_replies_request("ft", "docx", "cid", "");
        assert_eq!(
            req.resolved_uri(),
            "/open-apis/drive/v1/files/ft/comments/cid/replies"
        );
        // queries preserved & ordered
        assert_eq!(
            req.queries,
            vec![
                ("file_type".to_string(), "docx".to_string()),
                ("page_size".to_string(), "100".to_string()),
                ("user_id_type".to_string(), "open_id".to_string()),
            ]
        );
    }

    // --- Mock CommentApi for higher-level flow tests ---

    struct MockApi {
        responses: Mutex<Vec<ApiResponse>>,
        calls: Mutex<Vec<ApiRequest>>,
    }

    impl MockApi {
        fn new(responses: Vec<ApiResponse>) -> Self {
            MockApi {
                responses: Mutex::new(responses),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl CommentApi for MockApi {
        fn execute(&self, request: &ApiRequest) -> ApiResponse {
            self.calls.lock().unwrap().push(request.clone());
            let mut r = self.responses.lock().unwrap();
            if r.is_empty() {
                ApiResponse::new(0, "", Map::new())
            } else {
                r.remove(0)
            }
        }
        fn sleep_retry(&self) {}
    }

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn query_meta_list_shape() {
        let resp = ApiResponse::new(
            0,
            "",
            obj(json!({"metas": [{"title": "T", "url": "http://u", "doc_type": "docx"}]})),
        );
        let api = MockApi::new(vec![resp]);
        let m = query_document_meta(&api, "ft", "docx");
        assert_eq!(m.title, "T");
        assert_eq!(m.url, "http://u");
        assert_eq!(m.doc_type, "docx");
    }

    #[test]
    fn query_meta_dict_shape() {
        let resp = ApiResponse::new(
            0,
            "",
            obj(json!({"metas": {"ft": {"title": "DT", "url": "u2"}}})),
        );
        let api = MockApi::new(vec![resp]);
        let m = query_document_meta(&api, "ft", "sheet");
        assert_eq!(m.title, "DT");
        assert_eq!(m.doc_type, "sheet"); // falls back to file_type
    }

    #[test]
    fn batch_query_retries_then_succeeds() {
        let fail = ApiResponse::new(1, "err", Map::new());
        let ok = ApiResponse::new(
            0,
            "",
            obj(json!({"items": [{"comment_id": "c", "is_whole": true}]})),
        );
        let api = MockApi::new(vec![fail.clone(), ok]);
        let item = batch_query_comment(&api, "ft", "docx", "c");
        assert_eq!(item.get("is_whole"), Some(&Value::Bool(true)));
        assert_eq!(api.calls.lock().unwrap().len(), 2);
    }

    #[test]
    fn batch_query_all_fail_returns_empty() {
        let fail = ApiResponse::new(1, "err", Map::new());
        let api = MockApi::new(vec![fail.clone(); COMMENT_RETRY_LIMIT]);
        let item = batch_query_comment(&api, "ft", "docx", "c");
        assert!(item.is_empty());
        assert_eq!(api.calls.lock().unwrap().len(), COMMENT_RETRY_LIMIT);
    }

    #[test]
    fn list_whole_paginates() {
        let p1 = ApiResponse::new(
            0,
            "",
            obj(json!({"items": [{"x": 1}], "has_more": true, "page_token": "pt2"})),
        );
        let p2 = ApiResponse::new(0, "", obj(json!({"items": [{"x": 2}], "has_more": false})));
        let api = MockApi::new(vec![p1, p2]);
        let all = list_whole_comments(&api, "ft", "docx");
        assert_eq!(all.len(), 2);
        // second request carried page_token
        let calls = api.calls.lock().unwrap();
        assert!(calls[1]
            .queries
            .iter()
            .any(|(k, v)| k == "page_token" && v == "pt2"));
    }

    #[test]
    fn deliver_fallback_on_1069302() {
        let local_fail = ApiResponse::new(1069302, "no", Map::new());
        let whole_ok = ApiResponse::new(0, "", Map::new());
        let api = MockApi::new(vec![local_fail, whole_ok]);
        let ok = deliver_comment_reply(&api, "ft", "docx", "cid", "text", false);
        assert!(ok);
        let calls = api.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].uri, REPLY_COMMENT_URI);
        assert_eq!(calls[1].uri, ADD_COMMENT_URI);
    }

    #[test]
    fn deliver_whole_uses_add_comment() {
        let api = MockApi::new(vec![ApiResponse::new(0, "", Map::new())]);
        let ok = deliver_comment_reply(&api, "ft", "docx", "cid", "hi", true);
        assert!(ok);
        assert_eq!(api.calls.lock().unwrap()[0].uri, ADD_COMMENT_URI);
    }

    #[test]
    fn whole_timeline_builds_current_from_user() {
        let whole = vec![json!({
            "reply_list": {"replies": [
                {"user_id": {"open_id": "self"}, "content": {"elements": [
                    {"type": "text_run", "text_run": {"text": "bot says"}}]}},
                {"user_id": {"open_id": "userA"}, "content": {"elements": [
                    {"type": "text_run", "text_run": {"text": "  hello there  "}}]}},
            ]}
        })];
        let r = build_whole_timeline(&whole, "self", "userA");
        assert_eq!(r.timeline.len(), 2);
        assert!(r.timeline[0].2); // first is self
        assert_eq!(r.current_index, 1);
        assert_eq!(r.current_text, "hello there");
        assert_eq!(r.nearest_self_index, 0);
        assert_eq!(r.all_raw_replies.len(), 2);
    }

    #[test]
    fn local_timeline_target_by_reply_id() {
        let replies = vec![
            json!({"reply_id": "r0", "user_id": {"open_id": "userA"},
                   "content": {"elements": [{"type": "text_run", "text_run": {"text": "root msg"}}]}}),
            json!({"reply_id": "r1", "user_id": {"open_id": "userA"},
                   "content": {"elements": [{"type": "text_run", "text_run": {"text": "target msg"}}]}}),
        ];
        let r = build_local_timeline(&replies, "self", "userA", "r1");
        assert_eq!(r.root_text, "root msg");
        assert_eq!(r.target_index, 1);
        assert_eq!(r.target_text, "target msg");
    }

    #[test]
    fn local_timeline_target_fallback_by_user() {
        let replies = vec![
            json!({"reply_id": "r0", "user_id": {"open_id": "userA"},
                   "content": {"elements": [{"type": "text_run", "text_run": {"text": "msg"}}]}}),
        ];
        // reply_id "missing" not found -> fallback to last reply from from_open_id
        let r = build_local_timeline(&replies, "self", "userA", "missing");
        assert_eq!(r.target_index, 0);
        assert_eq!(r.target_text, "msg");
    }

    #[test]
    fn session_cache_roundtrip() {
        clear_session_cache();
        let key = session_key("docx", "tokABC");
        assert_eq!(key, "comment-doc:docx:tokABC");
        assert!(load_session_history(&key).is_empty());

        let messages = vec![
            json!({"role": "system", "content": "ignore me"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
            json!({"role": "assistant", "content": ""}), // empty -> dropped
        ];
        save_session_history(&key, &messages);
        let loaded = load_session_history(&key);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0]["role"], "user");
        assert_eq!(loaded[1]["role"], "assistant");
    }

    #[test]
    fn session_cache_caps_messages() {
        clear_session_cache();
        let key = session_key("d", "t");
        let messages: Vec<Value> = (0..(SESSION_MAX_MESSAGES + 10))
            .map(|i| json!({"role": "user", "content": format!("m{}", i)}))
            .collect();
        save_session_history(&key, &messages);
        let loaded = load_session_history(&key);
        assert_eq!(loaded.len(), SESSION_MAX_MESSAGES);
        // kept the last N
        assert_eq!(
            loaded[0]["content"],
            json!(format!("m{}", 10))
        );
    }

    #[test]
    fn wiki_reverse_lookup() {
        let ok = ApiResponse::new(0, "", obj(json!({"node": {"node_token": "wiki123"}})));
        let api = MockApi::new(vec![ok]);
        assert_eq!(
            reverse_lookup_wiki_token(&api, "docx", "obj1"),
            Some("wiki123".to_string())
        );

        let empty = ApiResponse::new(0, "", obj(json!({"node": {"node_token": ""}})));
        let api2 = MockApi::new(vec![empty]);
        assert_eq!(reverse_lookup_wiki_token(&api2, "docx", "obj1"), None);

        let fail = ApiResponse::new(1, "err", Map::new());
        let api3 = MockApi::new(vec![fail]);
        assert_eq!(reverse_lookup_wiki_token(&api3, "docx", "obj1"), None);
    }

    #[test]
    fn resolve_wiki_mutates_links() {
        let mut links = vec![
            DocLink {
                url: "u".into(),
                doc_type: "wiki".into(),
                token: "wtok".into(),
                resolved_type: None,
                resolved_token: None,
            },
            DocLink {
                url: "u2".into(),
                doc_type: "docx".into(),
                token: "dtok".into(),
                resolved_type: None,
                resolved_token: None,
            },
        ];
        let ok = ApiResponse::new(
            0,
            "",
            obj(json!({"node": {"obj_type": "docx", "obj_token": "realtok"}})),
        );
        let api = MockApi::new(vec![ok]);
        resolve_wiki_nodes(&api, &mut links);
        assert_eq!(links[0].resolved_type.as_deref(), Some("docx"));
        assert_eq!(links[0].resolved_token.as_deref(), Some("realtok"));
        assert!(links[1].resolved_type.is_none());
        // only one API call (for the wiki link)
        assert_eq!(api.calls.lock().unwrap().len(), 1);
    }
}
