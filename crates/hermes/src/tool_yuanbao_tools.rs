//! Yuanbao platform tool set ("hermes-yuanbao" toolset).
//!
//! Native Rust port of `tools/yuanbao_tools.py` (the 元宝平台工具集).
//!
//! Provides the tool functions exposed to the LLM:
//!   * `get_group_info`      — query basic group info (name, owner, member count)
//!   * `query_group_members` — query members (find by name, list bots, list all)
//!   * `search_sticker`      — keyword-search the built-in sticker catalogue
//!   * `send_sticker`        — send a built-in sticker (TIMFaceElem) to a chat
//!   * `send_dm`             — send a private/direct message to a group member
//!
//! ## Adapter indirection
//!
//! In Python the "active adapter" is a runtime singleton living in
//! `gateway.platforms.yuanbao`, accessed via `get_active_adapter()`. The runtime
//! adapter performs the actual network operations (querying group info, member
//! lists, sending messages and stickers). Those operations are *not* statically
//! available in the ported Rust crate, so this module models the adapter as the
//! [`YuanbaoAdapter`] trait, which a caller supplies. When no adapter is wired
//! up, callers pass `None` and the functions return the same
//! `"Yuanbao adapter is not connected"` errors as the Python original.
//!
//! All result values are `serde_json::Value` objects whose key/value shapes
//! mirror the Python dicts exactly, so the registry's `tool_result` serialization
//! is byte-compatible with the Python tools.
//!
//! Sticker lookup/search and session-env resolution reuse the already-ported
//! `crate::gw_yuanbao_sticker` and `crate::gw_session_context` modules; the
//! `MEDIA:<path>` extraction reuses `crate::gw_platforms_base::extract_media`.

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::gw_platforms_base::extract_media;
use crate::gw_session_context::get_session_env;
use crate::gw_yuanbao_sticker::{
    get_random_sticker, get_sticker_by_id, get_sticker_by_name, search_stickers, Sticker,
};

// ---------------------------------------------------------------------------
// Role labels
// ---------------------------------------------------------------------------

/// Maps a numeric `user_type` to a human-readable role string.
///
/// Mirrors `_USER_TYPE_LABEL = {0: "unknown", 1: "user", 2: "yuanbao_ai", 3: "bot"}`.
pub fn user_type_label(user_type: i64) -> &'static str {
    match user_type {
        1 => "user",
        2 => "yuanbao_ai",
        3 => "bot",
        _ => "unknown",
    }
}

/// The `@mention` formatting hint returned when `mention=true`.
pub const MENTION_HINT: &str =
    "To @mention a user, you MUST use the format: space + @ + nickname + space (e.g. \" @Alice \").";

/// Image file extensions used for media dispatch (mirrors `MessageSender.IMAGE_EXTS`).
pub const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"];

/// Returns `true` if `ext` (a lowercase extension including the dot) is an image type.
pub fn is_image_ext(ext: &str) -> bool {
    IMAGE_EXTS.contains(&ext)
}

/// The toolset name under which these tools are registered.
pub const TOOLSET: &str = "hermes-yuanbao";

// ---------------------------------------------------------------------------
// Adapter abstraction (mirrors the gateway.platforms.yuanbao runtime singleton)
// ---------------------------------------------------------------------------

/// Result of an adapter send operation (mirrors the Python `SendResult` /
/// duck-typed object with `.success`, `.error`, `.message_id`).
#[derive(Debug, Clone, Default)]
pub struct AdapterSendResult {
    /// Whether the send succeeded.
    pub success: bool,
    /// Error message on failure (mirrors `result.error`).
    pub error: Option<String>,
    /// The delivered message id, if any (mirrors `result.message_id`).
    pub message_id: Option<String>,
}

impl AdapterSendResult {
    /// A successful result carrying an optional message id.
    pub fn ok(message_id: Option<String>) -> Self {
        Self {
            success: true,
            error: None,
            message_id,
        }
    }

    /// A failed result carrying an error string.
    pub fn err(error: impl Into<String>) -> Self {
        Self {
            success: false,
            error: Some(error.into()),
            message_id: None,
        }
    }
}

/// Raw group-info payload returned by `adapter.query_group_info`.
///
/// Mirrors the dict consumed by Python: `group_name`, `member_count`,
/// `owner_id`, `owner_nickname`.
#[derive(Debug, Clone, Default)]
pub struct GroupInfo {
    pub group_name: String,
    pub member_count: i64,
    pub owner_id: String,
    pub owner_nickname: String,
}

/// A raw group member as returned by `adapter.get_group_member_list`.
///
/// Mirrors the dict with `user_id`, `nickname`/`nick_name`, and
/// `user_type`/`role`. The `nickname` getter prefers `nickname` then
/// `nick_name`; `user_type` falls back to `role` then `0`.
#[derive(Debug, Clone, Default)]
pub struct RawMember {
    pub user_id: String,
    /// Primary nickname field (`nickname`).
    pub nickname: Option<String>,
    /// Alternate nickname field (`nick_name`).
    pub nick_name: Option<String>,
    /// Primary type field (`user_type`).
    pub user_type: Option<i64>,
    /// Alternate type field (`role`).
    pub role: Option<i64>,
}

impl RawMember {
    /// Resolve the display nickname: `nickname` -> `nick_name` -> `""`.
    pub fn resolved_nickname(&self) -> String {
        self.nickname
            .clone()
            .or_else(|| self.nick_name.clone())
            .unwrap_or_default()
    }

    /// Resolve the numeric user type: `user_type` -> `role` -> `0`.
    pub fn resolved_user_type(&self) -> i64 {
        self.user_type.or(self.role).unwrap_or(0)
    }
}

/// The runtime adapter surface used by the yuanbao tools.
///
/// This mirrors the methods the Python tools invoke on the active adapter
/// singleton. The methods are synchronous here (the Python originals are
/// `async`; the Rust port performs the equivalent blocking calls in the
/// adapter implementation).
pub trait YuanbaoAdapter {
    /// `adapter.query_group_info(group_code)` -> `Option<GroupInfo>`.
    fn query_group_info(&self, group_code: &str) -> Option<GroupInfo>;

    /// `adapter.get_group_member_list(group_code)` -> `Option<Vec<RawMember>>`.
    ///
    /// `None` models Python's `None` return (`raw is None`). An empty vec
    /// models `raw.get("members", [])` being empty.
    fn get_group_member_list(&self, group_code: &str) -> Option<Vec<RawMember>>;

    /// `adapter.send_sticker(chat_id, sticker_name, reply_to)`.
    fn send_sticker(
        &self,
        chat_id: &str,
        sticker_name: &str,
        reply_to: Option<&str>,
    ) -> AdapterSendResult;

    /// `adapter.send_dm(user_id, message, group_code=group_code)`.
    fn send_dm(&self, user_id: &str, message: &str, group_code: &str) -> AdapterSendResult;

    /// `adapter.send_image_file(chat_id, path, group_code=group_code)`.
    fn send_image_file(&self, chat_id: &str, path: &str, group_code: &str) -> AdapterSendResult;

    /// `adapter.send_document(chat_id, path, group_code=group_code)`.
    fn send_document(&self, chat_id: &str, path: &str, group_code: &str) -> AdapterSendResult;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Build a `{"success": false, "error": <msg>}` value (the common failure shape).
fn err(msg: impl Into<String>) -> Value {
    json!({ "success": false, "error": msg.into() })
}

/// Convert a `RawMember` into the public member dict
/// `{"user_id", "nickname", "role"}`.
fn member_to_value(m: &RawMember) -> Value {
    json!({
        "user_id": m.user_id,
        "nickname": m.resolved_nickname(),
        "role": user_type_label(m.resolved_user_type()),
    })
}

/// Insert the `mention_hint` key into `obj` when `mention` is true (mirrors the
/// Python `**hint` spread).
fn apply_mention_hint(obj: &mut Map<String, Value>, mention: bool) {
    if mention {
        obj.insert("mention_hint".to_string(), json!(MENTION_HINT));
    }
}

// ---------------------------------------------------------------------------
// get_group_info
// ---------------------------------------------------------------------------

/// Query basic group info (name, owner, member count).
///
/// Mirrors `get_group_info`. `adapter` of `None` models a disconnected adapter.
pub fn get_group_info(adapter: Option<&dyn YuanbaoAdapter>, group_code: &str) -> Value {
    if group_code.is_empty() {
        return err("group_code is required");
    }
    let adapter = match adapter {
        Some(a) => a,
        None => return err("Yuanbao adapter is not connected"),
    };

    match adapter.query_group_info(group_code) {
        None => err("query_group_info returned None"),
        Some(gi) => json!({
            "success": true,
            "group_code": group_code,
            "group_name": gi.group_name,
            "member_count": gi.member_count,
            "owner": {
                "user_id": gi.owner_id,
                "nickname": gi.owner_nickname,
            },
            "note": "The group is called \"派 (Pai)\" in the app.",
        }),
    }
}

// ---------------------------------------------------------------------------
// query_group_members
// ---------------------------------------------------------------------------

/// Unified group-member query (mirrors TS `query_session_members`).
///
/// `action` is one of `"find"`, `"list_bots"`, `"list_all"` (default).
pub fn query_group_members(
    adapter: Option<&dyn YuanbaoAdapter>,
    group_code: &str,
    action: &str,
    name: &str,
    mention: bool,
) -> Value {
    if group_code.is_empty() {
        return err("group_code is required");
    }
    let adapter = match adapter {
        Some(a) => a,
        None => return err("Yuanbao adapter is not connected"),
    };

    let raw = match adapter.get_group_member_list(group_code) {
        None => return err("get_group_member_list returned None"),
        Some(r) => r,
    };

    let all_members: Vec<Value> = raw.iter().map(member_to_value).collect();
    if all_members.is_empty() {
        return err("No members found in this group.");
    }

    // Helper to read a member value's role string.
    let role_of = |v: &Value| -> String {
        v.get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let nickname_of = |v: &Value| -> String {
        v.get("nickname")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };

    if action == "list_bots" {
        let bots: Vec<Value> = all_members
            .iter()
            .filter(|m| {
                let r = role_of(m);
                r == "yuanbao_ai" || r == "bot"
            })
            .cloned()
            .collect();
        if bots.is_empty() {
            return err("No bots found in this group.");
        }
        let mut obj = Map::new();
        obj.insert("success".to_string(), json!(true));
        obj.insert("msg".to_string(), json!(format!("Found {} bot(s).", bots.len())));
        obj.insert("members".to_string(), json!(bots));
        apply_mention_hint(&mut obj, mention);
        return Value::Object(obj);
    }

    if action == "find" {
        if !name.is_empty() {
            let filt = name.trim().to_lowercase();
            let matched: Vec<Value> = all_members
                .iter()
                .filter(|m| nickname_of(m).to_lowercase().contains(&filt))
                .cloned()
                .collect();
            let mut obj = Map::new();
            if !matched.is_empty() {
                obj.insert("success".to_string(), json!(true));
                obj.insert(
                    "msg".to_string(),
                    json!(format!("Found {} member(s) matching \"{}\".", matched.len(), name)),
                );
                obj.insert("members".to_string(), json!(matched));
            } else {
                obj.insert("success".to_string(), json!(false));
                obj.insert(
                    "msg".to_string(),
                    json!(format!("No match for \"{}\". All members listed below.", name)),
                );
                obj.insert("members".to_string(), json!(all_members));
            }
            apply_mention_hint(&mut obj, mention);
            return Value::Object(obj);
        }
        let mut obj = Map::new();
        obj.insert("success".to_string(), json!(true));
        obj.insert(
            "msg".to_string(),
            json!(format!("Found {} member(s).", all_members.len())),
        );
        obj.insert("members".to_string(), json!(all_members));
        apply_mention_hint(&mut obj, mention);
        return Value::Object(obj);
    }

    // list_all (default)
    let mut obj = Map::new();
    obj.insert("success".to_string(), json!(true));
    obj.insert(
        "msg".to_string(),
        json!(format!("Found {} member(s).", all_members.len())),
    );
    obj.insert("members".to_string(), json!(all_members));
    apply_mention_hint(&mut obj, mention);
    Value::Object(obj)
}

// ---------------------------------------------------------------------------
// search_sticker
// ---------------------------------------------------------------------------

/// Clamp a raw `limit` to `[1, 50]`, defaulting to 10 when `limit` is absent
/// (`None`) or zero/falsey. Mirrors `max(1, min(50, int(limit) if limit else 10))`.
pub fn clamp_sticker_limit(limit: Option<i64>) -> usize {
    let l = match limit {
        // `if limit else 10` — Python treats 0 as falsey.
        Some(0) | None => 10,
        Some(v) => v,
    };
    l.clamp(1, 50) as usize
}

/// Search the built-in sticker catalogue by keyword, returning the top-N
/// candidates. Mirrors `search_sticker`.
pub fn search_sticker(query: &str, limit: Option<i64>) -> Value {
    let safe_limit = clamp_sticker_limit(limit);
    let matches = search_stickers(query, safe_limit);
    let results: Vec<Value> = matches
        .iter()
        .map(|s| {
            json!({
                "sticker_id": s.sticker_id,
                "name": s.name,
                "description": s.description,
                "package_id": s.package_id,
            })
        })
        .collect();
    json!({
        "success": true,
        "query": query,
        "count": results.len(),
        "results": results,
    })
}

// ---------------------------------------------------------------------------
// send_sticker
// ---------------------------------------------------------------------------

/// Resolve a sticker spec to a catalogue [`Sticker`], mirroring the Python
/// resolution order: empty -> random; all-digits -> by id (then by name
/// fallback); otherwise -> by name.
fn resolve_sticker(raw: &str) -> Option<&'static Sticker> {
    if raw.is_empty() {
        return Some(get_random_sticker(None));
    }
    let mut found: Option<&'static Sticker> = None;
    if !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit()) {
        found = get_sticker_by_id(raw);
    }
    if found.is_none() {
        found = get_sticker_by_name(raw);
    }
    found
}

/// Send a built-in sticker (TIMFaceElem) to a chat.
///
/// `chat_id` empty -> falls back to the session `HERMES_SESSION_CHAT_ID`.
/// `sticker` empty -> a random sticker. Mirrors `send_sticker`.
pub fn send_sticker(
    adapter: Option<&dyn YuanbaoAdapter>,
    sticker: &str,
    chat_id: &str,
    reply_to: &str,
) -> Value {
    let trimmed_chat = chat_id.trim();
    let target = if !trimmed_chat.is_empty() {
        trimmed_chat.to_string()
    } else {
        get_session_env("HERMES_SESSION_CHAT_ID", "")
    };
    if target.is_empty() {
        return err("chat_id is required (no active yuanbao session detected)");
    }

    let adapter = match adapter {
        Some(a) => a,
        None => return err("Yuanbao adapter is not connected"),
    };

    let raw = sticker.trim();
    let sticker_obj = match resolve_sticker(raw) {
        Some(s) => s,
        None => {
            // `{raw!r}` is Python repr (single-quoted).
            return err(format!(
                "Sticker not found: '{}'. Use search_sticker first to discover available stickers.",
                raw
            ));
        }
    };

    let reply = if reply_to.is_empty() {
        None
    } else {
        Some(reply_to)
    };
    let result = adapter.send_sticker(&target, sticker_obj.name, reply);

    if result.success {
        json!({
            "success": true,
            "chat_id": target,
            "sticker": {
                "sticker_id": sticker_obj.sticker_id,
                "name": sticker_obj.name,
            },
            "message_id": result.message_id,
            "note": "Sticker delivered to the chat. If you have additional text to say, reply now; otherwise end your turn without generating text.",
        })
    } else {
        err(result.error.unwrap_or_else(|| "send_sticker failed".to_string()))
    }
}

// ---------------------------------------------------------------------------
// send_dm
// ---------------------------------------------------------------------------

/// A media file to send after the DM text: `(file_path, is_voice)`.
pub type MediaFile = (String, bool);

/// Send a private/direct message to a group member, with optional media.
///
/// Mirrors `send_dm`. When `user_id` is empty, resolves it from the group
/// member list by `name`.
pub fn send_dm(
    adapter: Option<&dyn YuanbaoAdapter>,
    group_code: &str,
    name: &str,
    message: &str,
    user_id: &str,
    media_files: &[MediaFile],
) -> Value {
    if message.is_empty() && media_files.is_empty() {
        return err("message or media_files is required");
    }

    let adapter = match adapter {
        Some(a) => a,
        None => return err("Yuanbao adapter is not connected"),
    };

    let mut resolved_user_id = user_id.trim().to_string();
    let mut resolved_nickname = name.trim().to_string();

    // Step 1: resolve user_id from the member list if not provided.
    if resolved_user_id.is_empty() {
        if group_code.is_empty() {
            return err("group_code is required when user_id is not provided");
        }
        if name.is_empty() {
            return err("name is required when user_id is not provided");
        }

        let members = match adapter.get_group_member_list(group_code) {
            None => return err("get_group_member_list returned None"),
            Some(m) => m,
        };

        let filt = name.trim().to_lowercase();
        let matched: Vec<&RawMember> = members
            .iter()
            .filter(|m| m.resolved_nickname().to_lowercase().contains(&filt))
            .collect();

        if matched.is_empty() {
            return err(format!(
                "No member matching \"{}\" found in group {}.",
                name, group_code
            ));
        }
        if matched.len() > 1 {
            let candidates: Vec<Value> = matched
                .iter()
                .map(|m| {
                    json!({
                        "user_id": m.user_id,
                        "nickname": m.resolved_nickname(),
                    })
                })
                .collect();
            return json!({
                "success": false,
                "error": format!("Multiple members match \"{}\". Please specify which one.", name),
                "candidates": candidates,
            });
        }

        let first = matched[0];
        resolved_user_id = first.user_id.clone();
        // Python: m.get("nickname", m.get("nick_name", name)) — fall back to
        // the passed-in `name` if both nickname fields are absent.
        resolved_nickname = first
            .nickname
            .clone()
            .or_else(|| first.nick_name.clone())
            .unwrap_or_else(|| name.to_string());
    }

    if resolved_user_id.is_empty() {
        return err("Could not resolve user_id");
    }

    // Step 2 + 3: send text DM and media.
    let chat_id = format!("direct:{}", resolved_user_id);
    let mut last_result: Option<AdapterSendResult> = None;
    let mut errors: Vec<String> = Vec::new();

    if !message.is_empty() && !message.trim().is_empty() {
        let r = adapter.send_dm(&resolved_user_id, message, group_code);
        if !r.success {
            errors.push(r.error.clone().unwrap_or_else(|| "text send failed".to_string()));
        }
        last_result = Some(r);
    }

    for (media_path, _is_voice) in media_files {
        let ext = Path::new(media_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e.to_lowercase()))
            .unwrap_or_default();
        let r = if is_image_ext(&ext) {
            adapter.send_image_file(&chat_id, media_path, group_code)
        } else {
            adapter.send_document(&chat_id, media_path, group_code)
        };
        if !r.success {
            errors.push(r.error.clone().unwrap_or_else(|| "media send failed".to_string()));
        }
        last_result = Some(r);
    }

    let last_result = match last_result {
        None => return err("No deliverable text or media remained"),
        Some(r) => r,
    };

    if !errors.is_empty() && !last_result.success {
        return err(errors.join("; "));
    }

    let mut note = format!("DM sent to \"{}\" successfully.", resolved_nickname);
    if !errors.is_empty() {
        note.push_str(&format!(" (partial failure: {})", errors.join("; ")));
    }

    json!({
        "success": true,
        "user_id": resolved_user_id,
        "nickname": resolved_nickname,
        "message_id": last_result.message_id,
        "note": note,
    })
}

// ---------------------------------------------------------------------------
// Toolset availability check
// ---------------------------------------------------------------------------

/// Toolset availability check — `true` when running in a yuanbao gateway session.
///
/// Mirrors `_check_yuanbao`: true if `HERMES_SESSION_PLATFORM == "yuanbao"`,
/// otherwise true if an adapter is present.
pub fn check_yuanbao(adapter_present: bool) -> bool {
    if get_session_env("HERMES_SESSION_PLATFORM", "") == "yuanbao" {
        return true;
    }
    adapter_present
}

// ---------------------------------------------------------------------------
// Handler arg parsing (mirrors the `_handle_yb_*` registry handlers)
// ---------------------------------------------------------------------------

/// Resolve `group_code` for `yb_send_dm`: prefer the explicit arg, otherwise
/// extract from the session `HERMES_SESSION_CHAT_ID` when it looks like
/// `group:<code>`. Mirrors the fallback logic in `_handle_yb_send_dm`.
pub fn resolve_send_dm_group_code(args: &Value) -> String {
    let explicit = args
        .get("group_code")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if !explicit.is_empty() {
        return explicit;
    }
    let chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "");
    if let Some(rest) = chat_id.strip_prefix("group:") {
        return rest.to_string();
    }
    String::new()
}

/// Parse the `media_files` argument into `(path, is_voice)` tuples and append any
/// `MEDIA:<path>` tags embedded in `message`. Returns `(media_files, cleaned_message)`.
///
/// Mirrors the `_handle_yb_send_dm` parsing: dict items use `path`/`is_voice`;
/// list/tuple items of length >= 2 use index 0 (path) and 1 (is_voice).
pub fn parse_send_dm_media(args: &Value, message: &str) -> (Vec<MediaFile>, String) {
    let mut media_files: Vec<MediaFile> = Vec::new();

    if let Some(raw_media) = args.get("media_files").and_then(Value::as_array) {
        for item in raw_media {
            if let Some(obj) = item.as_object() {
                let path = obj.get("path").and_then(Value::as_str).unwrap_or("").to_string();
                let is_voice = obj
                    .get("is_voice")
                    .map(value_truthy)
                    .unwrap_or(false);
                media_files.push((path, is_voice));
            } else if let Some(arr) = item.as_array() {
                if arr.len() >= 2 {
                    let path = json_to_str(&arr[0]);
                    let is_voice = value_truthy(&arr[1]);
                    media_files.push((path, is_voice));
                }
            }
        }
    }

    let (embedded_media, cleaned) = extract_media(message);
    media_files.extend(embedded_media);
    (media_files, cleaned)
}

/// Python-like truthiness for a JSON value (used for `is_voice` / `mention`).
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Stringify a JSON value the way `str(item[0])` would: strings without quotes.
fn json_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tool schemas (mirror the registry.register schema blocks)
// ---------------------------------------------------------------------------

/// Schema for `yb_query_group_info`.
pub fn yb_query_group_info_schema() -> Value {
    json!({
        "name": "yb_query_group_info",
        "description":
            "Query basic info about a group (called '派/Pai' in the app), \
             including group name, owner, and member count.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description": "The unique group identifier (group_code).",
                },
            },
            "required": ["group_code"],
        },
    })
}

/// Schema for `yb_query_group_members`.
pub fn yb_query_group_members_schema() -> Value {
    json!({
        "name": "yb_query_group_members",
        "description":
            "Query members of a group (called '派/Pai' in the app). \
             Use this tool when you need to @mention someone, find a user by name, \
             list bots (including Yuanbao AI), or list all members. \
             IMPORTANT: You MUST call this tool before @mentioning any user, \
             because you need the exact nickname to construct the @mention format.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description": "The unique group identifier (group_code).",
                },
                "action": {
                    "type": "string",
                    "enum": ["find", "list_bots", "list_all"],
                    "description":
                        "find — search a user by name (use when you need to @mention or look up someone); \
                         list_bots — list bots and Yuanbao AI assistants; \
                         list_all — list all members.",
                },
                "name": {
                    "type": "string",
                    "description":
                        "User name to search (partial match, case-insensitive). \
                         Required for 'find'. Use the name the user mentioned in the conversation.",
                },
                "mention": {
                    "type": "boolean",
                    "description":
                        "Set to true when you need to @mention/at someone in your reply. \
                         The response will include the exact @mention format to use.",
                },
            },
            "required": ["group_code", "action"],
        },
    })
}

/// Schema for `yb_send_dm`.
pub fn yb_send_dm_schema() -> Value {
    json!({
        "name": "yb_send_dm",
        "description":
            "Send a private/direct message (DM) to a user in a group, with optional media files. \
             This tool automatically looks up the user by name in the group member list \
             and sends the message. Use this when someone asks to privately message / 私信 / DM a user. \
             Supports text, images, and file attachments. \
             You can also provide user_id directly if already known.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description":
                        "The group where the target user belongs. \
                         Extract from chat_id: 'group:328306697' → '328306697'. \
                         Required when user_id is not provided.",
                },
                "name": {
                    "type": "string",
                    "description":
                        "Target user's display name (partial match, case-insensitive). \
                         Required when user_id is not provided.",
                },
                "message": {
                    "type": "string",
                    "description": "The message text to send as a DM. Can be empty if only sending media.",
                },
                "user_id": {
                    "type": "string",
                    "description":
                        "Target user's account ID. If provided, skips the member lookup. \
                         Usually obtained from a previous yb_query_group_members call.",
                },
                "media_files": {
                    "type": "array",
                    "description":
                        "Optional list of media files to send along with the DM. \
                         Images (.jpg/.png/.gif/.webp/.bmp) are sent as image messages; \
                         other files are sent as document attachments.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute local file path of the media to send.",
                            },
                            "is_voice": {
                                "type": "boolean",
                                "description": "Whether this file is a voice message (default false).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            "required": [],
        },
    })
}

/// Schema for `yb_search_sticker`.
pub fn yb_search_sticker_schema() -> Value {
    json!({
        "name": "yb_search_sticker",
        "description":
            "Search the built-in Yuanbao sticker (TIM face / 表情包) catalogue by keyword. \
             Returns the top matching candidates with sticker_id, name, and description. \
             Use this BEFORE yb_send_sticker to discover the right sticker_id. \
             Sticker = 贴纸 = TIM face — NOT a message reaction. \
             Prefer sending a sticker over bare Unicode emoji when reacting/expressing emotion.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description":
                        "Search keyword (Chinese or English, e.g. '666', '比心', 'cool', '吃瓜'). \
                         Empty string returns the first N stickers.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of candidates to return (default 10, max 50).",
                },
            },
            "required": [],
        },
    })
}

/// Schema for `yb_send_sticker`.
pub fn yb_send_sticker_schema() -> Value {
    json!({
        "name": "yb_send_sticker",
        "description":
            "Send a built-in sticker (TIMFaceElem / 贴纸表情) to the current Yuanbao chat. \
             Call yb_search_sticker first if you don't know the sticker_id/name. \
             Sticker = 贴纸 = TIM face — NOT a message reaction. \
             CRITICAL: Whenever the user asks you to send a sticker / 贴纸 / 表情包, you MUST \
             use this tool. DO NOT draw a PNG via execute_code / Pillow / matplotlib and \
             then call send_image_file — that produces a fake 'sticker' image instead of a \
             real TIM face and is the WRONG path. If no suitable sticker_id is known, call \
             yb_search_sticker first. When the recent thread shows users sending stickers, \
             prefer matching that tone by replying with a sticker instead of (or in \
             addition to) text.",
        "parameters": {
            "type": "object",
            "properties": {
                "sticker": {
                    "type": "string",
                    "description":
                        "Sticker name (e.g. '六六六', '比心', 'ok') or numeric sticker_id \
                         (e.g. '278'). Empty string sends a random built-in sticker.",
                },
                "chat_id": {
                    "type": "string",
                    "description":
                        "Target chat. Defaults to the current session. \
                         Format: 'direct:{account_id}', 'group:{group_code}', or bare account_id.",
                },
                "reply_to": {
                    "type": "string",
                    "description": "Optional ref_msg_id to quote-reply (group chat only).",
                },
            },
            "required": [],
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAdapter {
        group_info: Option<GroupInfo>,
        members: Option<Vec<RawMember>>,
    }

    impl YuanbaoAdapter for MockAdapter {
        fn query_group_info(&self, _group_code: &str) -> Option<GroupInfo> {
            self.group_info.clone()
        }
        fn get_group_member_list(&self, _group_code: &str) -> Option<Vec<RawMember>> {
            self.members.clone()
        }
        fn send_sticker(&self, _c: &str, _n: &str, _r: Option<&str>) -> AdapterSendResult {
            AdapterSendResult::ok(Some("msg-1".into()))
        }
        fn send_dm(&self, _u: &str, _m: &str, _g: &str) -> AdapterSendResult {
            AdapterSendResult::ok(Some("dm-1".into()))
        }
        fn send_image_file(&self, _c: &str, _p: &str, _g: &str) -> AdapterSendResult {
            AdapterSendResult::ok(Some("img-1".into()))
        }
        fn send_document(&self, _c: &str, _p: &str, _g: &str) -> AdapterSendResult {
            AdapterSendResult::ok(Some("doc-1".into()))
        }
    }

    fn member(uid: &str, nick: &str, ut: i64) -> RawMember {
        RawMember {
            user_id: uid.into(),
            nickname: Some(nick.into()),
            nick_name: None,
            user_type: Some(ut),
            role: None,
        }
    }

    #[test]
    fn user_type_label_mapping() {
        assert_eq!(user_type_label(0), "unknown");
        assert_eq!(user_type_label(1), "user");
        assert_eq!(user_type_label(2), "yuanbao_ai");
        assert_eq!(user_type_label(3), "bot");
        assert_eq!(user_type_label(99), "unknown");
    }

    #[test]
    fn group_info_requires_code_and_adapter() {
        assert_eq!(
            get_group_info(None, ""),
            json!({"success": false, "error": "group_code is required"})
        );
        assert_eq!(
            get_group_info(None, "g1"),
            json!({"success": false, "error": "Yuanbao adapter is not connected"})
        );
    }

    #[test]
    fn group_info_success() {
        let a = MockAdapter {
            group_info: Some(GroupInfo {
                group_name: "Pai".into(),
                member_count: 7,
                owner_id: "u9".into(),
                owner_nickname: "Boss".into(),
            }),
            members: None,
        };
        let v = get_group_info(Some(&a), "g1");
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["group_name"], json!("Pai"));
        assert_eq!(v["member_count"], json!(7));
        assert_eq!(v["owner"]["user_id"], json!("u9"));
        assert_eq!(v["owner"]["nickname"], json!("Boss"));
    }

    #[test]
    fn group_info_none_returns_error() {
        let a = MockAdapter { group_info: None, members: None };
        assert_eq!(
            get_group_info(Some(&a), "g1"),
            json!({"success": false, "error": "query_group_info returned None"})
        );
    }

    #[test]
    fn members_list_all_and_find() {
        let a = MockAdapter {
            group_info: None,
            members: Some(vec![
                member("u1", "Alice", 1),
                member("u2", "AliceBot", 3),
                member("u3", "Bob", 1),
            ]),
        };
        let all = query_group_members(Some(&a), "g1", "list_all", "", false);
        assert_eq!(all["success"], json!(true));
        assert_eq!(all["members"].as_array().unwrap().len(), 3);
        assert!(all.get("mention_hint").is_none());

        // find matches case-insensitively on substring
        let found = query_group_members(Some(&a), "g1", "find", "alice", true);
        assert_eq!(found["success"], json!(true));
        assert_eq!(found["members"].as_array().unwrap().len(), 2);
        assert_eq!(found["mention_hint"], json!(MENTION_HINT));

        // no match returns all members with success=false
        let none = query_group_members(Some(&a), "g1", "find", "zzz", false);
        assert_eq!(none["success"], json!(false));
        assert_eq!(none["members"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn members_list_bots() {
        let a = MockAdapter {
            group_info: None,
            members: Some(vec![member("u1", "Alice", 1), member("u2", "AI", 2), member("u3", "BotX", 3)]),
        };
        let bots = query_group_members(Some(&a), "g1", "list_bots", "", false);
        assert_eq!(bots["success"], json!(true));
        assert_eq!(bots["members"].as_array().unwrap().len(), 2);

        let no_bots = MockAdapter {
            group_info: None,
            members: Some(vec![member("u1", "Alice", 1)]),
        };
        let r = query_group_members(Some(&no_bots), "g1", "list_bots", "", false);
        assert_eq!(r, json!({"success": false, "error": "No bots found in this group."}));
    }

    #[test]
    fn members_empty_and_none() {
        let empty = MockAdapter { group_info: None, members: Some(vec![]) };
        assert_eq!(
            query_group_members(Some(&empty), "g1", "list_all", "", false),
            json!({"success": false, "error": "No members found in this group."})
        );
        let none = MockAdapter { group_info: None, members: None };
        assert_eq!(
            query_group_members(Some(&none), "g1", "list_all", "", false),
            json!({"success": false, "error": "get_group_member_list returned None"})
        );
    }

    #[test]
    fn nickname_fallback_to_nick_name() {
        let m = RawMember {
            user_id: "u1".into(),
            nickname: None,
            nick_name: Some("Fallback".into()),
            user_type: None,
            role: Some(3),
        };
        assert_eq!(m.resolved_nickname(), "Fallback");
        assert_eq!(m.resolved_user_type(), 3);
    }

    #[test]
    fn clamp_limit_rules() {
        assert_eq!(clamp_sticker_limit(None), 10);
        assert_eq!(clamp_sticker_limit(Some(0)), 10);
        assert_eq!(clamp_sticker_limit(Some(5)), 5);
        assert_eq!(clamp_sticker_limit(Some(999)), 50);
        assert_eq!(clamp_sticker_limit(Some(-3)), 1);
    }

    #[test]
    fn search_sticker_shape() {
        let v = search_sticker("666", Some(3));
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["query"], json!("666"));
        let results = v["results"].as_array().unwrap();
        assert!(results.len() <= 3);
        assert_eq!(v["count"], json!(results.len()));
        if let Some(first) = results.first() {
            assert!(first.get("sticker_id").is_some());
            assert!(first.get("name").is_some());
            assert!(first.get("description").is_some());
            assert!(first.get("package_id").is_some());
        }
    }

    #[test]
    fn send_sticker_requires_target_and_adapter() {
        // No chat_id and no session env -> error.
        let prev = std::env::var("HERMES_SESSION_CHAT_ID").ok();
        unsafe {
            std::env::remove_var("HERMES_SESSION_CHAT_ID");
        }
        assert_eq!(
            send_sticker(None, "", "", ""),
            json!({"success": false, "error": "chat_id is required (no active yuanbao session detected)"})
        );
        // Adapter missing with target present.
        assert_eq!(
            send_sticker(None, "", "group:1", ""),
            json!({"success": false, "error": "Yuanbao adapter is not connected"})
        );
        if let Some(p) = prev {
            unsafe {
                std::env::set_var("HERMES_SESSION_CHAT_ID", p);
            }
        }
    }

    #[test]
    fn send_sticker_not_found() {
        let a = MockAdapter { group_info: None, members: None };
        let v = send_sticker(Some(&a), "definitely-not-a-real-sticker-xyz", "group:1", "");
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("Sticker not found"));
    }

    #[test]
    fn send_sticker_random_success() {
        let a = MockAdapter { group_info: None, members: None };
        let v = send_sticker(Some(&a), "", "group:1", "");
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["chat_id"], json!("group:1"));
        assert_eq!(v["message_id"], json!("msg-1"));
        assert!(v["sticker"]["sticker_id"].is_string());
    }

    #[test]
    fn send_dm_requires_message_or_media() {
        assert_eq!(
            send_dm(None, "g1", "n", "", "", &[]),
            json!({"success": false, "error": "message or media_files is required"})
        );
    }

    #[test]
    fn send_dm_resolves_and_sends() {
        let a = MockAdapter {
            group_info: None,
            members: Some(vec![member("u1", "Alice", 1), member("u2", "Bob", 1)]),
        };
        let v = send_dm(Some(&a), "g1", "alice", "hi", "", &[]);
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["user_id"], json!("u1"));
        assert_eq!(v["nickname"], json!("Alice"));
        assert_eq!(v["message_id"], json!("dm-1"));
    }

    #[test]
    fn send_dm_multiple_matches() {
        let a = MockAdapter {
            group_info: None,
            members: Some(vec![member("u1", "Alice One", 1), member("u2", "Alice Two", 1)]),
        };
        let v = send_dm(Some(&a), "g1", "alice", "hi", "", &[]);
        assert_eq!(v["success"], json!(false));
        assert_eq!(v["candidates"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn send_dm_no_match() {
        let a = MockAdapter {
            group_info: None,
            members: Some(vec![member("u1", "Alice", 1)]),
        };
        let v = send_dm(Some(&a), "g1", "zzz", "hi", "", &[]);
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].as_str().unwrap().contains("No member matching"));
    }

    #[test]
    fn send_dm_with_user_id_and_media() {
        let a = MockAdapter { group_info: None, members: None };
        let media = vec![("photo.PNG".to_string(), false), ("file.bin".to_string(), false)];
        let v = send_dm(Some(&a), "", "Known", "hello", "u42", &media);
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["user_id"], json!("u42"));
        // last_result corresponds to the document send.
        assert_eq!(v["message_id"], json!("doc-1"));
    }

    #[test]
    fn parse_media_from_args_and_message() {
        let args = json!({
            "media_files": [
                {"path": "/a.png", "is_voice": false},
                {"path": "/b.ogg", "is_voice": true},
                ["/c.jpg", true],
            ],
        });
        let (media, cleaned) = parse_send_dm_media(&args, "plain text");
        assert_eq!(cleaned, "plain text");
        assert_eq!(media.len(), 3);
        assert_eq!(media[0], ("/a.png".to_string(), false));
        assert_eq!(media[1], ("/b.ogg".to_string(), true));
        assert_eq!(media[2], ("/c.jpg".to_string(), true));
    }

    #[test]
    fn resolve_group_code_from_session() {
        let prev = std::env::var("HERMES_SESSION_CHAT_ID").ok();
        unsafe {
            std::env::set_var("HERMES_SESSION_CHAT_ID", "group:328306697");
        }
        let args = json!({});
        assert_eq!(resolve_send_dm_group_code(&args), "328306697");
        // explicit arg wins
        let args2 = json!({"group_code": "explicit"});
        assert_eq!(resolve_send_dm_group_code(&args2), "explicit");
        match prev {
            Some(p) => unsafe { std::env::set_var("HERMES_SESSION_CHAT_ID", p) },
            None => unsafe { std::env::remove_var("HERMES_SESSION_CHAT_ID") },
        }
    }

    #[test]
    fn schemas_have_names() {
        assert_eq!(yb_query_group_info_schema()["name"], json!("yb_query_group_info"));
        assert_eq!(yb_query_group_members_schema()["name"], json!("yb_query_group_members"));
        assert_eq!(yb_send_dm_schema()["name"], json!("yb_send_dm"));
        assert_eq!(yb_search_sticker_schema()["name"], json!("yb_search_sticker"));
        assert_eq!(yb_send_sticker_schema()["name"], json!("yb_send_sticker"));
    }

    #[test]
    fn image_ext_detection() {
        assert!(is_image_ext(".jpg"));
        assert!(is_image_ext(".png"));
        assert!(!is_image_ext(".pdf"));
        assert!(!is_image_ext(""));
    }
}
