//! Slack platform adapter — native Rust port of `gateway/platforms/slack.py`.
//!
//! The original Python module is built on `slack-bolt` / `slack_sdk` Socket
//! Mode, whose async websocket lifecycle, event-dispatch decorators, and live
//! `AsyncWebClient` have no faithful equivalent in the crates available here.
//! This module therefore ports the **deterministic, side-effect-free logic**
//! that the rest of Hermes (and tests) depend on, reproducing the Python
//! behavior exactly:
//!
//! - [`extract_text_from_slack_blocks`] — walk Block Kit `rich_text` trees
//!   (Python `_extract_text_from_slack_blocks`), preserving quotes/lists/code.
//! - [`serialize_slack_blocks_for_agent`] — compact redacted JSON view of a
//!   non-rich-text Block Kit payload (Python `_serialize_slack_blocks_for_agent`).
//! - [`format_message`] — markdown → Slack mrkdwn conversion (Python
//!   `SlackAdapter.format_message`).
//! - [`describe_slack_api_error`] / [`describe_slack_download_failure_message`]
//!   — actionable attachment-failure text (Python `_describe_slack_*`).
//! - [`resolve_slack_proxy_url`] — proxy gating for Slack transport (Python
//!   `_resolve_slack_proxy_url`).
//! - Config / env gating: [`require_mention`], [`strict_mention`],
//!   [`free_response_channels`], [`reactions_enabled`], [`allow_bots_mode`],
//!   [`dm_top_level_threads_as_sessions`].
//! - Thread / routing helpers: [`resolve_thread_ts`], [`is_thread_reply`],
//!   [`is_mentioned`], [`strip_bot_mention`], [`thread_context_cache_key`].
//! - Slash-command routing: [`route_slash_command_text`],
//!   [`is_dm_channel`], slash-context TTL/match logic ([`SlashCommandContexts`]).
//! - Block Kit builders: [`build_exec_approval_blocks`],
//!   [`build_slash_confirm_blocks`], decision label maps, value parsing.
//! - Assistant-thread metadata extraction ([`extract_assistant_thread_metadata`]).
//! - Upload retry classification ([`is_retryable_upload_error`]).
//! - Thread-context formatting ([`format_thread_context`]).
//!
//! The live Socket Mode loop and network sends are out of scope; callers wire
//! those through the Python bridge. This module supplies the pure logic those
//! paths call into, and constructs/parses the exact API request shapes.

use serde_json::{json, Value};
use std::collections::HashMap;

// ─── Constants (mirror module-level Python constants) ───────────────────────

/// Slack API allows 40,000 chars; leave margin (Python `MAX_MESSAGE_LENGTH`).
pub const MAX_MESSAGE_LENGTH: usize = 39000;

/// Cap on tracked bot-message timestamps (Python `_BOT_TS_MAX`).
pub const BOT_TS_MAX: usize = 5000;

/// Cap on remembered mentioned-thread set (Python `_MENTIONED_THREADS_MAX`).
pub const MENTIONED_THREADS_MAX: usize = 5000;

/// Cap on cached assistant-thread metadata (Python `_ASSISTANT_THREADS_MAX`).
pub const ASSISTANT_THREADS_MAX: usize = 5000;

/// Thread-context cache TTL in seconds (Python `_THREAD_CACHE_TTL`).
pub const THREAD_CACHE_TTL: f64 = 60.0;

/// Slash-command context TTL in seconds (Python `_SLASH_CTX_TTL`).
pub const SLASH_CTX_TTL: f64 = 120.0;

/// Hosts whose proxy treatment is gated by `NO_PROXY` (Python `_SLACK_PROXY_HOSTS`).
pub const SLACK_PROXY_HOSTS: &[&str] = &["slack.com", "files.slack.com", "wss-primary.slack.com"];

// ─── Block Kit text extraction ──────────────────────────────────────────────

/// Render inline elements (text, link, channel, user, emoji, etc.) to a string.
/// Mirrors the inner `_render_inline_elements` of the Python helper.
fn render_inline_elements(elements: &[Value]) -> String {
    let mut pieces: Vec<String> = Vec::new();
    for el in elements {
        let el_type = el.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match el_type {
            "text" => pieces.push(el.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string()),
            "link" => {
                let url = el.get("url").and_then(|v| v.as_str()).unwrap_or("");
                let text = el.get("text").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(url);
                pieces.push(format!("{} ({})", text, url));
            }
            "channel" => {
                let cid = el.get("channel_id").and_then(|v| v.as_str()).unwrap_or("");
                pieces.push(format!("<#{}>", cid));
            }
            "user" => {
                let uid = el.get("user_id").and_then(|v| v.as_str()).unwrap_or("");
                pieces.push(format!("<@{}>", uid));
            }
            "usergroup" => {
                let gid = el.get("usergroup_id").and_then(|v| v.as_str()).unwrap_or("");
                pieces.push(format!("<!subteam^{}>", gid));
            }
            "emoji" => {
                let name = el.get("name").and_then(|v| v.as_str()).unwrap_or("");
                pieces.push(format!(":{}:", name));
            }
            "broadcast" => {
                let range = el.get("range").and_then(|v| v.as_str()).unwrap_or("here");
                pieces.push(format!("<!{}>", range));
            }
            "date" => {
                pieces.push(el.get("fallback").and_then(|v| v.as_str()).unwrap_or("").to_string());
            }
            _ => {}
        }
    }
    pieces.concat()
}

fn append_line(parts: &mut Vec<String>, text: &str, quote_depth: usize, bullet: &str) {
    if text.is_empty() || text.trim().is_empty() {
        return;
    }
    let prefix = if quote_depth > 0 {
        format!("{} ", ">".repeat(quote_depth))
    } else {
        String::new()
    };
    let combined = format!("{}{}{}", prefix, bullet, text);
    // Python `.rstrip()` strips trailing whitespace.
    parts.push(combined.trim_end().to_string());
}

fn walk_elements(parts: &mut Vec<String>, elements: &[Value], quote_depth: usize, bullet: &str) {
    for elem in elements {
        let elem_type = elem.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match elem_type {
            "rich_text_section" => {
                let inner = render_inline_elements(child_elements(elem));
                append_line(parts, &inner, quote_depth, bullet);
            }
            "rich_text_quote" => {
                walk_elements(parts, child_elements(elem), quote_depth + 1, "");
            }
            "rich_text_list" => {
                let list_style = elem.get("style").and_then(|v| v.as_str());
                for (idx, item) in child_elements(elem).iter().enumerate() {
                    let item_bullet = if list_style == Some("bullet") {
                        "\u{2022} ".to_string()
                    } else {
                        format!("{}. ", idx + 1)
                    };
                    walk_elements(parts, std::slice::from_ref(item), quote_depth, &item_bullet);
                }
            }
            "rich_text_preformatted" => {
                let mut code_lines: Vec<String> = Vec::new();
                for child in child_elements(elem) {
                    let child_type = child.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    let rendered = if child_type == "rich_text_section" {
                        render_inline_elements(child_elements(child))
                    } else {
                        render_inline_elements(std::slice::from_ref(child))
                    };
                    if !rendered.is_empty() {
                        code_lines.push(rendered);
                    }
                }
                let code_text = code_lines.join("\n");
                if !code_text.is_empty() {
                    let lang = elem.get("language").and_then(|v| v.as_str()).unwrap_or("");
                    let block = format!("```{}\n{}\n```", lang, code_text);
                    append_line(parts, &block, quote_depth, bullet);
                }
            }
            _ => {
                let rendered = render_inline_elements(std::slice::from_ref(elem));
                if !rendered.is_empty() {
                    append_line(parts, &rendered, quote_depth, bullet);
                }
            }
        }
    }
}

fn child_elements(v: &Value) -> &[Value] {
    v.get("elements").and_then(|e| e.as_array()).map(|a| a.as_slice()).unwrap_or(&[])
}

/// Extract readable text from Slack Block Kit blocks, including quoted/forwarded
/// content. Mirrors Python `_extract_text_from_slack_blocks`.
pub fn extract_text_from_slack_blocks(blocks: &[Value]) -> String {
    if blocks.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        let btype = block.get("type").and_then(|v| v.as_str());
        if btype == Some("rich_text") {
            walk_elements(&mut parts, child_elements(block), 0, "");
        }
    }
    parts.join("\n")
}

// ─── Block Kit JSON serialization for the agent ─────────────────────────────

const SCALAR_ALLOWLIST: &[&str] = &[
    "type", "block_id", "action_id", "style", "dispatch_action", "optional", "multiple", "emoji",
];
const RECURSIVE_ALLOWLIST: &[&str] = &[
    "text", "title", "description", "label", "placeholder", "accessory", "fields", "elements",
    "options", "option_groups", "confirm", "submit", "close", "hint",
];

fn is_empty_sentinel(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Object(m) => m.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

fn sanitize_block_value(value: &Value) -> Value {
    match value {
        Value::Array(arr) => {
            let items: Vec<Value> = arr
                .iter()
                .map(sanitize_block_value)
                .filter(|item| !is_empty_sentinel(item))
                .collect();
            Value::Array(items)
        }
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, item) in map {
                if SCALAR_ALLOWLIST.contains(&key.as_str()) {
                    out.insert(key.clone(), item.clone());
                } else if RECURSIVE_ALLOWLIST.contains(&key.as_str()) {
                    let cleaned = sanitize_block_value(item);
                    if !is_empty_sentinel(&cleaned) {
                        out.insert(key.clone(), cleaned);
                    }
                }
            }
            Value::Object(out)
        }
        // str/int/float/bool/None passthrough.
        _ => value.clone(),
    }
}

/// Return a compact, redacted JSON view of a non-rich-text Block Kit payload.
/// Mirrors Python `_serialize_slack_blocks_for_agent`.
pub fn serialize_slack_blocks_for_agent(blocks: &[Value], max_chars: usize) -> String {
    if blocks.is_empty() {
        return String::new();
    }
    // If every block is rich_text, the dedicated extractor already handled it.
    if blocks
        .iter()
        .all(|b| b.get("type").and_then(|v| v.as_str()) == Some("rich_text"))
    {
        return String::new();
    }

    let sanitized = Value::Array(blocks.iter().map(sanitize_block_value).collect());
    let mut payload = serde_json::to_string_pretty(&sanitized).unwrap_or_else(|_| format!("{:?}", blocks));

    if payload.len() > max_chars {
        // Python: payload[: max_chars - 18].rstrip() + "\n... [truncated]"
        let cut = max_chars.saturating_sub(18);
        let mut end = cut.min(payload.len());
        while !payload.is_char_boundary(end) {
            end -= 1;
        }
        let head = payload[..end].trim_end();
        payload = format!("{}\n... [truncated]", head);
    }

    format!("[Slack Block Kit payload for this message]\n```json\n{}\n```", payload)
}

/// Convenience wrapper with the Python default `max_chars=6000`.
pub fn serialize_slack_blocks_for_agent_default(blocks: &[Value]) -> String {
    serialize_slack_blocks_for_agent(blocks, 6000)
}

// ─── Proxy resolution ───────────────────────────────────────────────────────

/// Resolve a proxy URL that Slack SDK clients can safely use.
/// Mirrors Python `_resolve_slack_proxy_url`.
///
/// `resolved_proxy` is the output of the shared `resolve_proxy_url()` (None when
/// unset). `no_proxy_excludes` should report whether a given Slack host is
/// excluded by `NO_PROXY` (mirrors `is_host_excluded_by_no_proxy`).
pub fn resolve_slack_proxy_url(
    resolved_proxy: Option<&str>,
    no_proxy_excludes: impl Fn(&str) -> bool,
) -> Option<String> {
    let proxy_url = resolved_proxy?;
    if proxy_url.is_empty() {
        return None;
    }
    let normalized = proxy_url.to_lowercase();
    if !(normalized.starts_with("http://") || normalized.starts_with("https://")) {
        return None;
    }
    if SLACK_PROXY_HOSTS.iter().any(|h| no_proxy_excludes(h)) {
        return None;
    }
    Some(proxy_url.to_string())
}

// ─── Attachment-failure diagnostics ─────────────────────────────────────────

fn file_label(file_obj: Option<&Value>) -> String {
    let obj = file_obj.unwrap_or(&Value::Null);
    let name = obj.get("name").and_then(|v| v.as_str());
    let id = obj.get("id").and_then(|v| v.as_str());
    match (name, id) {
        (Some(n), _) if !n.is_empty() => n.to_string(),
        (_, Some(i)) if !i.is_empty() => i.to_string(),
        _ => "this attachment".to_string(),
    }
}

/// Convert Slack API auth/permission failures into actionable user-facing text.
/// Mirrors Python `_describe_slack_api_error`. `response` is the parsed Slack
/// API JSON response. Returns `None` when no actionable message applies.
pub fn describe_slack_api_error(response: Option<&Value>, file_obj: Option<&Value>) -> Option<String> {
    let response = response?;
    if !response.is_object() {
        return None;
    }
    let error = response.get("error").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if error.is_empty() {
        return None;
    }
    let label = file_label(file_obj);
    let needed = response.get("needed").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let provided = response.get("provided").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let reinstall_hint = " Update the Slack app scopes/settings and reinstall the app to the workspace.";
    let provided_hint = if !provided.is_empty() {
        format!(" Current bot scopes: {}.", provided)
    } else {
        String::new()
    };

    match error.as_str() {
        "missing_scope" => {
            let needed_hint = if !needed.is_empty() {
                format!("Missing scope: {}.", needed)
            } else {
                "Missing required Slack scope.".to_string()
            };
            Some(format!(
                "Slack attachment access failed for {}. {}{}{}",
                label, needed_hint, provided_hint, reinstall_hint
            ))
        }
        "not_authed" | "invalid_auth" | "account_inactive" | "token_revoked" => Some(format!(
            "Slack attachment access failed for {} because the bot token is not authorized ({}). Refresh the token/reinstall the app.",
            label, error
        )),
        "file_not_found" | "file_deleted" => {
            Some(format!("Slack attachment {} is no longer available ({}).", label, error))
        }
        "access_denied" | "file_access_denied" | "no_permission" | "not_allowed_token_type"
        | "restricted_action" => Some(format!(
            "Slack attachment access failed for {} because the bot does not have permission ({}). Check workspace permissions/scopes and reinstall if needed.",
            label, error
        )),
        _ => None,
    }
}

/// Translate a download failure (status code + message) into user-facing text.
/// Mirrors the HTTP / message branches of Python `_describe_slack_download_failure`
/// (the `response`-object branch is covered by [`describe_slack_api_error`]).
pub fn describe_slack_download_failure_message(
    status_code: Option<u16>,
    message: &str,
    file_obj: Option<&Value>,
) -> Option<String> {
    let label = file_label(file_obj);
    if let Some(status) = status_code {
        match status {
            401 => return Some(format!(
                "Slack attachment access failed for {} with HTTP 401. The bot token is not authorized for this file.",
                label
            )),
            403 => return Some(format!(
                "Slack attachment access failed for {} with HTTP 403. The bot likely lacks permission or scope to read this file.",
                label
            )),
            404 => return Some(format!(
                "Slack attachment {} returned HTTP 404 and is no longer reachable.",
                label
            )),
            _ => {}
        }
    }
    if message.contains("Slack returned HTML instead of media") || message.contains("non-image data") {
        return Some(format!(
            "Slack attachment access failed for {}: Slack returned an HTML/login or non-media response. This usually means a scope, auth, or file-permission problem.",
            label
        ));
    }
    None
}

// ─── Markdown → mrkdwn conversion ───────────────────────────────────────────

/// Convert standard markdown to Slack mrkdwn format.
/// Faithful port of Python `SlackAdapter.format_message` using the same
/// placeholder-protection ordering so code/inline-code/links survive escaping.
pub fn format_message(content: &str) -> String {
    use regex::Regex;
    if content.is_empty() {
        return content.to_string();
    }

    // Placeholders preserve insertion order; we restore in reverse.
    let mut placeholders: Vec<(String, String)> = Vec::new();
    // counter is the index used for the placeholder token.
    let mut counter: usize = 0;

    // Helper closure can't borrow `placeholders` mutably across regex callbacks
    // (the regex `replace_all` takes a Fn). We therefore implement each pass
    // with a manual scan that builds the output incrementally.

    macro_rules! stash {
        ($buf:expr, $val:expr) => {{
            let key = format!("\u{0}SL{}\u{0}", counter);
            counter += 1;
            placeholders.push((key.clone(), $val));
            key
        }};
    }

    let mut text = content.to_string();

    // 1) Protect fenced code blocks (``` ... ```)
    {
        let re = Regex::new(r"(?s)(```(?:[^\n]*\n)?.*?```)").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            out.push_str(&text[last..m.start()]);
            let key = stash!(out, m.as_str().to_string());
            out.push_str(&key);
            last = m.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 2) Protect inline code (`...`)
    {
        let re = Regex::new(r"(`[^`]+`)").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            out.push_str(&text[last..m.start()]);
            let key = stash!(out, m.as_str().to_string());
            out.push_str(&key);
            last = m.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 3) Convert markdown links [text](url) → <url|text>.
    // Python negative-lookbehind (?<!!) excludes image links ![...](...).
    {
        let re = Regex::new(r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&text) {
            let whole = caps.get(0).unwrap();
            // Manual (?<!!) lookbehind: skip when preceded by '!'.
            if whole.start() > 0 && text.as_bytes()[whole.start() - 1] == b'!' {
                continue;
            }
            out.push_str(&text[last..whole.start()]);
            let label = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let mut url = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("").to_string();
            if url.starts_with('<') && url.ends_with('>') {
                url = url[1..url.len() - 1].trim().to_string();
            }
            let key = stash!(out, format!("<{}|{}>", url, label));
            out.push_str(&key);
            last = whole.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 4) Protect existing Slack entities/manual links.
    {
        let re = Regex::new(r"(<(?:[@#!]|(?:https?|mailto|tel):)[^>\n]+>)").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            out.push_str(&text[last..m.start()]);
            let key = stash!(out, m.as_str().to_string());
            out.push_str(&key);
            last = m.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 5) Protect blockquote markers (multiline) before escaping.
    {
        let re = Regex::new(r"(?m)^(>+\s)").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            out.push_str(&text[last..m.start()]);
            let key = stash!(out, m.as_str().to_string());
            out.push_str(&key);
            last = m.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 6) Escape Slack control characters. Unescape first to avoid double-escape.
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">");
    text = text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");

    // 7) Convert headers (## Title) → *Title*.
    {
        let re = Regex::new(r"(?m)^#{1,6}\s+(.+)$").unwrap();
        let bold_re = Regex::new(r"(?s)\*\*(.+?)\*\*").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&text) {
            let whole = caps.get(0).unwrap();
            out.push_str(&text[last..whole.start()]);
            let inner = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let inner = bold_re.replace_all(inner, "$1").to_string();
            let key = stash!(out, format!("*{}*", inner));
            out.push_str(&key);
            last = whole.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 8) bold+italic ***text*** → *_text_*
    text = replace_capture(&text, r"(?s)\*\*\*(.+?)\*\*\*", |c| {
        format!("*_{}_*", c.get(1).map(|m| m.as_str()).unwrap_or(""))
    }, &mut placeholders, &mut counter);

    // 9) bold **text** → *text*
    text = replace_capture(&text, r"(?s)\*\*(.+?)\*\*", |c| {
        format!("*{}*", c.get(1).map(|m| m.as_str()).unwrap_or(""))
    }, &mut placeholders, &mut counter);

    // 10) italic *text* → _text_ (with surrounding non-* guards).
    {
        let re = Regex::new(r"\*(\S(?:[^*\n]*?\S)?)\*").unwrap();
        let mut out = String::new();
        let mut last = 0usize;
        let bytes = text.as_bytes();
        for caps in re.captures_iter(&text) {
            let whole = caps.get(0).unwrap();
            // (?<!\*) and (?!\*) manual lookarounds.
            if whole.start() > 0 && bytes[whole.start() - 1] == b'*' {
                continue;
            }
            if whole.end() < bytes.len() && bytes[whole.end()] == b'*' {
                continue;
            }
            out.push_str(&text[last..whole.start()]);
            let inner = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let key = format!("\u{0}SL{}\u{0}", counter);
            counter += 1;
            placeholders.push((key.clone(), format!("_{}_", inner)));
            out.push_str(&key);
            last = whole.end();
        }
        out.push_str(&text[last..]);
        text = out;
    }

    // 11) strikethrough ~~text~~ → ~text~
    text = replace_capture(&text, r"(?s)~~(.+?)~~", |c| {
        format!("~{}~", c.get(1).map(|m| m.as_str()).unwrap_or(""))
    }, &mut placeholders, &mut counter);

    // 13) Restore placeholders in reverse insertion order.
    for (key, val) in placeholders.iter().rev() {
        text = text.replace(key.as_str(), val);
    }

    text
}

/// Replace each match of `pattern` with a stashed placeholder whose value is
/// `f(captures)`. Shared by the bold/italic/strike passes of `format_message`.
fn replace_capture(
    text: &str,
    pattern: &str,
    f: impl Fn(&regex::Captures) -> String,
    placeholders: &mut Vec<(String, String)>,
    counter: &mut usize,
) -> String {
    let re = regex::Regex::new(pattern).unwrap();
    let mut out = String::new();
    let mut last = 0usize;
    for caps in re.captures_iter(text) {
        let whole = caps.get(0).unwrap();
        out.push_str(&text[last..whole.start()]);
        let key = format!("\u{0}SL{}\u{0}", *counter);
        *counter += 1;
        placeholders.push((key.clone(), f(&caps)));
        out.push_str(&key);
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

// ─── Config / env gating ────────────────────────────────────────────────────

fn extra_str<'a>(extra: &'a HashMap<String, Value>, key: &str) -> Option<&'a Value> {
    extra.get(key)
}

/// Parse a Slack `value` (config-extra or env) for explicit-false semantics:
/// returns true unless the value is one of false/0/no/off (case-insensitive).
fn truthy_unless_false(v: &str) -> bool {
    !matches!(v.trim().to_lowercase().as_str(), "false" | "0" | "no" | "off")
}

/// Parse for explicit-true semantics: true iff value is true/1/yes/on.
fn truthy_only_true(v: &str) -> bool {
    matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

/// Whether channel messages require an explicit bot mention.
/// Mirrors Python `_slack_require_mention`. `env_val` is `SLACK_REQUIRE_MENTION`.
pub fn require_mention(extra: &HashMap<String, Value>, env_val: Option<&str>) -> bool {
    if let Some(v) = extra_str(extra, "require_mention") {
        if !v.is_null() {
            return match v {
                Value::String(s) => truthy_unless_false(s),
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
                _ => true,
            };
        }
    }
    let raw = env_val.unwrap_or("true");
    truthy_unless_false(raw)
}

/// When true, channel threads require an explicit @-mention every message.
/// Mirrors Python `_slack_strict_mention`. `env_val` is `SLACK_STRICT_MENTION`.
pub fn strict_mention(extra: &HashMap<String, Value>, env_val: Option<&str>) -> bool {
    if let Some(v) = extra_str(extra, "strict_mention") {
        if !v.is_null() {
            return match v {
                Value::String(s) => truthy_only_true(s),
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                _ => false,
            };
        }
    }
    let raw = env_val.unwrap_or("false");
    truthy_only_true(raw)
}

/// Channel IDs where no @mention is required.
/// Mirrors Python `_slack_free_response_channels`.
/// `env_val` is `SLACK_FREE_RESPONSE_CHANNELS`.
pub fn free_response_channels(extra: &HashMap<String, Value>, env_val: Option<&str>) -> Vec<String> {
    let raw = extra.get("free_response_channels");
    let mut out: Vec<String> = Vec::new();
    let push_unique = |out: &mut Vec<String>, s: String| {
        if !s.is_empty() && !out.contains(&s) {
            out.push(s);
        }
    };

    match raw {
        Some(Value::Array(arr)) => {
            for part in arr {
                let s = value_to_scalar_string(part);
                push_unique(&mut out, s.trim().to_string());
            }
            return out;
        }
        Some(Value::Null) | None => {
            let s = env_val.unwrap_or("").trim().to_string();
            if !s.is_empty() {
                for part in s.split(',') {
                    push_unique(&mut out, part.trim().to_string());
                }
            }
            return out;
        }
        Some(other) => {
            let s = value_to_scalar_string(other);
            let s = s.trim().to_string();
            if !s.is_empty() {
                for part in s.split(',') {
                    push_unique(&mut out, part.trim().to_string());
                }
            }
            return out;
        }
    }
}

fn value_to_scalar_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => {
            // Python str(bool) → "True"/"False"
            if *b { "True".to_string() } else { "False".to_string() }
        }
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

/// Whether message reactions are enabled. Mirrors Python `_reactions_enabled`.
/// `env_val` is `SLACK_REACTIONS` (default "true").
pub fn reactions_enabled(env_val: Option<&str>) -> bool {
    let raw = env_val.unwrap_or("true").to_lowercase();
    !matches!(raw.as_str(), "false" | "0" | "no")
}

/// Bot-message handling mode. Mirrors the `allow_bots` resolution in
/// `_handle_slack_message`: config extra `allow_bots` then `SLACK_ALLOW_BOTS`
/// (default "none"). Returns the normalized lowercase/stripped mode.
pub fn allow_bots_mode(extra: &HashMap<String, Value>, env_val: Option<&str>) -> String {
    let mut allow = extra
        .get("allow_bots")
        .map(value_to_scalar_string)
        .unwrap_or_default();
    if allow.is_empty() {
        allow = env_val.unwrap_or("none").to_string();
    }
    allow.to_lowercase().trim().to_string()
}

/// Whether top-level Slack DMs get per-message session threads.
/// Mirrors Python `_dm_top_level_threads_as_sessions` (default True).
pub fn dm_top_level_threads_as_sessions(extra: &HashMap<String, Value>) -> bool {
    match extra.get("dm_top_level_threads_as_sessions") {
        None | Some(Value::Null) => true,
        Some(v) => {
            let s = value_to_scalar_string(v);
            matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
        }
    }
}

/// Whether `reply_broadcast` is enabled. Mirrors `config.extra["reply_broadcast"]`.
pub fn reply_broadcast(extra: &HashMap<String, Value>) -> bool {
    matches!(extra.get("reply_broadcast"), Some(Value::Bool(true)))
}

// ─── Thread / routing helpers ───────────────────────────────────────────────

/// Resolve the correct `thread_ts` for a Slack API call.
/// Faithful port of Python `_resolve_thread_ts`.
pub fn resolve_thread_ts(
    extra: &HashMap<String, Value>,
    reply_to: Option<&str>,
    metadata: Option<&HashMap<String, Value>>,
) -> Option<String> {
    let reply_in_thread = match extra.get("reply_in_thread") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => truthy_unless_false(s),
        Some(_) => true,
    };

    let md_get = |k: &str| -> Option<String> {
        metadata
            .and_then(|m| m.get(k))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    };

    if !reply_in_thread {
        let mut existing = md_get("thread_id").or_else(|| md_get("thread_ts"));
        if let (Some(ex), Some(rt)) = (existing.as_deref(), reply_to) {
            if ex == rt {
                existing = None;
            }
        }
        return existing;
    }

    if metadata.is_some() {
        if let Some(t) = md_get("thread_id") {
            return Some(t);
        }
        if let Some(t) = md_get("thread_ts") {
            return Some(t);
        }
    }
    reply_to.map(|s| s.to_string())
}

/// Whether an event's `thread_ts` indicates a real thread reply (not the root).
/// Mirrors `is_thread_reply = bool(event_thread_ts and event_thread_ts != ts)`.
pub fn is_thread_reply(event_thread_ts: Option<&str>, ts: &str) -> bool {
    match event_thread_ts {
        Some(t) if !t.is_empty() => t != ts,
        _ => false,
    }
}

/// Whether the routing text contains the bot mention `<@bot_uid>`.
/// Mirrors `is_mentioned = bot_uid and f"<@{bot_uid}>" in routing_text`.
pub fn is_mentioned(bot_uid: Option<&str>, routing_text: &str) -> bool {
    match bot_uid {
        Some(uid) if !uid.is_empty() => routing_text.contains(&format!("<@{}>", uid)),
        _ => false,
    }
}

/// Strip the bot mention token from text and trim, mirroring
/// `text.replace(f"<@{bot_uid}>", "").strip()`.
pub fn strip_bot_mention(text: &str, bot_uid: &str) -> String {
    if bot_uid.is_empty() {
        return text.trim().to_string();
    }
    text.replace(&format!("<@{}>", bot_uid), "").trim().to_string()
}

/// Cache key for thread-context lookups: `{channel}:{thread_ts}:{team_id}`.
pub fn thread_context_cache_key(channel_id: &str, thread_ts: &str, team_id: &str) -> String {
    format!("{}:{}:{}", channel_id, thread_ts, team_id)
}

/// Whether a channel ID denotes a DM (starts with `D`). Mirrors
/// `str(channel_id).startswith("D")`.
pub fn is_dm_channel(channel_id: &str) -> bool {
    channel_id.starts_with('D')
}

// ─── Slash-command routing ──────────────────────────────────────────────────

/// Compute the message text for a slash command invocation.
/// Faithful port of the routing logic in Python `_handle_slash_command`.
///
/// - `slash_name`: the command name with the leading `/` stripped.
/// - `text`: the trimmed argument string.
/// - `subcommand_map`: map of legacy `/hermes <sub>` words → canonical commands
///   (already including the `compact` → `/compress` override).
pub fn route_slash_command_text(
    slash_name: &str,
    text: &str,
    subcommand_map: &HashMap<String, String>,
) -> String {
    let text = text.trim();
    if slash_name == "hermes" || slash_name.is_empty() {
        if text.is_empty() {
            return "/help".to_string();
        }
        let first_word = text.split_whitespace().next().unwrap_or("");
        if let Some(mapped) = subcommand_map.get(first_word) {
            // rest = text[len(first_word):].strip()
            let rest = text[first_word.len()..].trim();
            if rest.is_empty() {
                return mapped.clone();
            }
            return format!("{} {}", mapped, rest).trim().to_string();
        }
        // Treat as a regular question.
        return text.to_string();
    }
    // Native slash — /<slash_name> [args].
    format!("/{} {}", slash_name, text).trim().to_string()
}

/// Build the regex pattern string used to match native slash commands.
/// Mirrors `^/(?:name1|name2|...)$` (or `^/hermes$` when empty).
pub fn slash_command_pattern(slash_names: &[String]) -> String {
    if slash_names.is_empty() {
        return r"^/hermes$".to_string();
    }
    let escaped: Vec<String> = slash_names.iter().map(|n| regex::escape(n)).collect();
    format!(r"^/(?:{})$", escaped.join("|"))
}

/// Stashed slash-command context: the response_url plus a monotonic timestamp.
#[derive(Debug, Clone)]
pub struct SlashContext {
    pub response_url: String,
    pub ts: f64,
}

/// Slash-command context store, keyed by `(channel_id, user_id)`.
/// Mirrors `_slash_command_contexts` plus `_pop_slash_context` semantics.
#[derive(Debug, Default)]
pub struct SlashCommandContexts {
    entries: HashMap<(String, String), SlashContext>,
}

impl SlashCommandContexts {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Stash a context. Mirrors the assignment in `_handle_slash_command`.
    pub fn stash(&mut self, channel_id: &str, user_id: &str, response_url: &str, now: f64) {
        if response_url.is_empty() || user_id.is_empty() || channel_id.is_empty() {
            return;
        }
        self.entries.insert(
            (channel_id.to_string(), user_id.to_string()),
            SlashContext { response_url: response_url.to_string(), ts: now },
        );
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return and remove the context for `chat_id`, if fresh. Faithful port of
    /// `_pop_slash_context`. `slash_user_id` is the ContextVar value (the
    /// invoking user); `None` triggers the channel-only fallback scan.
    pub fn pop(&mut self, chat_id: &str, slash_user_id: Option<&str>, now: f64) -> Option<SlashContext> {
        // Clean up stale entries first.
        let stale: Vec<(String, String)> = self
            .entries
            .iter()
            .filter(|(_, v)| now - v.ts > SLASH_CTX_TTL)
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale {
            self.entries.remove(&k);
        }

        // Precise (channel, user) match.
        if let Some(uid) = slash_user_id {
            if !uid.is_empty() {
                return self.entries.remove(&(chat_id.to_string(), uid.to_string()));
            }
        }

        // Channel-only fallback scan.
        let match_key = self.entries.keys().find(|(c, _)| c == chat_id).cloned();
        match_key.and_then(|k| self.entries.remove(&k))
    }
}

// ─── Block Kit builders ─────────────────────────────────────────────────────

/// Build the Block Kit blocks for an exec-approval prompt.
/// Faithful port of the `blocks` constructed in `send_exec_approval`.
pub fn build_exec_approval_blocks(command: &str, description: &str, session_key: &str) -> Value {
    let cmd_preview = preview_truncate(command, 2900);
    json!([
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": format!(":warning: *Command Approval Required*\n```{}```\nReason: {}", cmd_preview, description),
            },
        },
        {
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Allow Once"}, "style": "primary", "action_id": "hermes_approve_once", "value": session_key},
                {"type": "button", "text": {"type": "plain_text", "text": "Allow Session"}, "action_id": "hermes_approve_session", "value": session_key},
                {"type": "button", "text": {"type": "plain_text", "text": "Always Allow"}, "action_id": "hermes_approve_always", "value": session_key},
                {"type": "button", "text": {"type": "plain_text", "text": "Deny"}, "style": "danger", "action_id": "hermes_deny", "value": session_key},
            ],
        },
    ])
}

/// Build the fallback `text` for an exec-approval prompt
/// (`⚠️ Command approval required: {cmd_preview[:100]}`).
pub fn exec_approval_fallback_text(command: &str) -> String {
    let cmd_preview = preview_truncate(command, 2900);
    let head: String = cmd_preview.chars().take(100).collect();
    format!("\u{26A0}\u{FE0F} Command approval required: {}", head)
}

/// Build the Block Kit blocks for a slash-confirm prompt.
/// Faithful port of the `blocks` constructed in `send_slash_confirm`.
pub fn build_slash_confirm_blocks(title: &str, message: &str, session_key: &str, confirm_id: &str) -> Value {
    let body = preview_truncate(message, 2900);
    let value = format!("{}|{}", session_key, confirm_id);
    let title_disp = if title.is_empty() { "Confirm" } else { title };
    json!([
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": format!("*{}*\n\n{}", title_disp, body)},
        },
        {
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Approve Once"}, "style": "primary", "action_id": "hermes_confirm_once", "value": value},
                {"type": "button", "text": {"type": "plain_text", "text": "Always Approve"}, "action_id": "hermes_confirm_always", "value": value},
                {"type": "button", "text": {"type": "plain_text", "text": "Cancel"}, "style": "danger", "action_id": "hermes_confirm_cancel", "value": value},
            ],
        },
    ])
}

/// Fallback text for a slash-confirm prompt (`{title}: {body[:100]}`).
pub fn slash_confirm_fallback_text(title: &str, message: &str) -> String {
    let body = preview_truncate(message, 2900);
    let head: String = body.chars().take(100).collect();
    let title_disp = if title.is_empty() { "Confirm" } else { title };
    format!("{}: {}", title_disp, head)
}

/// `value[:N] + "..."` if longer than `N`, else `value`. Mirrors the Python
/// `x[:N] + "..." if len(x) > N else x` idiom (operating on chars).
fn preview_truncate(value: &str, n: usize) -> String {
    if value.chars().count() > n {
        let head: String = value.chars().take(n).collect();
        format!("{}...", head)
    } else {
        value.to_string()
    }
}

/// Map an approval `action_id` to its choice string. Mirrors `choice_map`.
pub fn approval_choice(action_id: &str) -> &'static str {
    match action_id {
        "hermes_approve_once" => "once",
        "hermes_approve_session" => "session",
        "hermes_approve_always" => "always",
        "hermes_deny" => "deny",
        _ => "deny",
    }
}

/// Approval decision label given choice + user name. Mirrors `label_map`.
pub fn approval_decision_label(choice: &str, user_name: &str) -> String {
    match choice {
        "once" => format!("\u{2705} Approved once by {}", user_name),
        "session" => format!("\u{2705} Approved for session by {}", user_name),
        "always" => format!("\u{2705} Approved permanently by {}", user_name),
        "deny" => format!("\u{274C} Denied by {}", user_name),
        _ => format!("Resolved by {}", user_name),
    }
}

/// Map a slash-confirm `action_id` to its choice string. Mirrors `choice_map`.
pub fn slash_confirm_choice(action_id: &str) -> &'static str {
    match action_id {
        "hermes_confirm_once" => "once",
        "hermes_confirm_always" => "always",
        "hermes_confirm_cancel" => "cancel",
        _ => "cancel",
    }
}

/// Slash-confirm decision label. Mirrors the slash-confirm `label_map`.
pub fn slash_confirm_decision_label(choice: &str, user_name: &str) -> String {
    match choice {
        "once" => format!("\u{2705} Approved once by {}", user_name),
        "always" => format!("\u{1F512} Always approved by {}", user_name),
        "cancel" => format!("\u{274C} Cancelled by {}", user_name),
        _ => format!("Resolved by {}", user_name),
    }
}

/// Parse `session_key|confirm_id` from a slash-confirm button value.
/// Returns `None` when no `|` is present (Python logs "Malformed").
pub fn parse_slash_confirm_value(value: &str) -> Option<(String, String)> {
    value.split_once('|').map(|(a, b)| (a.to_string(), b.to_string()))
}

/// Build the updated blocks shown after an approval/confirm decision.
/// Mirrors the `updated_blocks` construction in both button handlers.
pub fn build_decision_blocks(original_text: &str, fallback_section: &str, decision_text: &str) -> Value {
    let section_text = if original_text.is_empty() { fallback_section } else { original_text };
    json!([
        {"type": "section", "text": {"type": "mrkdwn", "text": section_text}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": decision_text}]},
    ])
}

/// Extract the first `section` block's text from a message's `blocks`.
/// Mirrors the `for block in message.get("blocks", [])` loop.
pub fn extract_section_text(blocks: &[Value]) -> String {
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("section") {
            return block
                .get("text")
                .and_then(|t| t.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
    }
    String::new()
}

/// Authorize a button click against `SLACK_ALLOWED_USERS`. Mirrors the
/// allowlist check shared by both button handlers: when the env var is set,
/// `*` allows everyone, otherwise the clicking user must be listed.
pub fn button_click_authorized(allowed_csv: &str, user_id: &str) -> bool {
    let trimmed = allowed_csv.trim();
    if trimmed.is_empty() {
        return true;
    }
    let allowed: Vec<&str> = trimmed.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    allowed.contains(&"*") || allowed.contains(&user_id)
}

// ─── Assistant-thread metadata ──────────────────────────────────────────────

/// Extract Slack Assistant thread identity data from an event payload.
/// Faithful port of Python `_extract_assistant_thread_metadata`.
pub fn extract_assistant_thread_metadata(event: &Value) -> HashMap<String, String> {
    let assistant_thread = event.get("assistant_thread").cloned().unwrap_or(Value::Null);
    let context = assistant_thread
        .get("context")
        .cloned()
        .or_else(|| event.get("context").cloned())
        .unwrap_or(Value::Null);

    let g = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());

    let channel_id = g(&assistant_thread, "channel_id")
        .or_else(|| g(event, "channel"))
        .or_else(|| g(&context, "channel_id"))
        .unwrap_or_default();
    let thread_ts = g(&assistant_thread, "thread_ts")
        .or_else(|| g(event, "thread_ts"))
        .or_else(|| g(event, "message_ts"))
        .unwrap_or_default();
    let user_id = g(&assistant_thread, "user_id")
        .or_else(|| g(event, "user"))
        .or_else(|| g(&context, "user_id"))
        .unwrap_or_default();
    let team_id = g(event, "team")
        .or_else(|| g(event, "team_id"))
        .or_else(|| g(&assistant_thread, "team_id"))
        .unwrap_or_default();
    let context_channel_id = g(&context, "channel_id").unwrap_or_default();

    let mut out = HashMap::new();
    out.insert("channel_id".to_string(), channel_id);
    out.insert("thread_ts".to_string(), thread_ts);
    out.insert("user_id".to_string(), user_id);
    out.insert("team_id".to_string(), team_id);
    out.insert("context_channel_id".to_string(), context_channel_id);
    out
}

// ─── Upload retry classification ────────────────────────────────────────────

/// Best-effort detection for transient Slack upload failures.
/// Faithful port of Python `_is_retryable_upload_error`. `status_code` is the
/// HTTP status if available; `message` is the lowercased composite error text.
pub fn is_retryable_upload_error(status_code: Option<u16>, message: &str) -> bool {
    if let Some(code) = status_code {
        return code == 429 || code >= 500;
    }
    let body = message.to_lowercase();
    if body.contains("rate_limited") || body.contains("ratelimited") || body.contains("429") {
        return true;
    }
    if body.contains("connection reset")
        || body.contains("service unavailable")
        || body.contains("temporarily unavailable")
    {
        return true;
    }
    // Fallback to the generic base classifier heuristic.
    is_retryable_error_heuristic(&body)
}

/// Generic transient-error heuristic mirroring the base adapter's
/// `_is_retryable_error` (timeouts, resets, 5xx, rate limits).
fn is_retryable_error_heuristic(body: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "timeout", "timed out", "connection reset", "connection aborted", "connection refused",
        "temporarily unavailable", "service unavailable", "rate limit", "rate_limited",
        "ratelimited", "429", "500", "502", "503", "504", "eof occurred", "broken pipe",
    ];
    NEEDLES.iter().any(|n| body.contains(n))
}

// ─── Thread-context formatting ──────────────────────────────────────────────

/// A single resolved thread message used to build the context block.
#[derive(Debug, Clone)]
pub struct ThreadContextEntry {
    pub display_name: String,
    pub text: String,
    pub is_parent: bool,
}

/// Format the prior-thread-history context string from already-resolved entries.
/// Mirrors the tail of `_fetch_thread_context` (the formatting only; the API
/// call/dedup/exclusion logic lives at the call site, which feeds in entries).
pub fn format_thread_context(entries: &[ThreadContextEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(entries.len());
    for e in entries {
        let prefix = if e.is_parent { "[thread parent] " } else { "" };
        parts.push(format!("{}{}: {}", prefix, e.display_name, e.text));
    }
    format!(
        "[Thread context — prior messages in this thread (not yet in conversation history):]\n{}\n[End of thread context]\n\n",
        parts.join("\n")
    )
}

/// Whether a thread-context cache entry is still fresh given a monotonic `now`.
pub fn cache_is_fresh(fetched_at: f64, now: f64) -> bool {
    now - fetched_at < THREAD_CACHE_TTL
}

// ─── Message-event classification helpers ───────────────────────────────────

/// Determine whether an event subtype should be ignored (edits / deletions).
/// Mirrors `if subtype in ("message_changed", "message_deleted"): return`.
pub fn is_ignored_subtype(subtype: Option<&str>) -> bool {
    matches!(subtype, Some("message_changed") | Some("message_deleted"))
}

/// Whether the event looks like a bot message. Mirrors
/// `event.get("bot_id") or event.get("subtype") == "bot_message"`.
pub fn is_bot_message(event: &Value) -> bool {
    event.get("bot_id").map(|v| !v.is_null()).unwrap_or(false)
        || event.get("subtype").and_then(|v| v.as_str()) == Some("bot_message")
}

/// Resolve channel_type → is_dm, applying the `D`-prefix fallback.
/// Mirrors the `channel_type` / `is_dm` block in `_handle_slack_message`.
pub fn resolve_is_dm(channel_type: &str, channel_id: &str) -> bool {
    let mut ct = channel_type.to_string();
    if ct.is_empty() && channel_id.starts_with('D') {
        ct = "im".to_string();
    }
    ct == "im" || ct == "mpim"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn blocks_text_extracts_quotes_and_lists() {
        let blocks = vec![json!({
            "type": "rich_text",
            "elements": [
                {"type": "rich_text_section", "elements": [{"type": "text", "text": "Hello"}]},
                {"type": "rich_text_quote", "elements": [
                    {"type": "rich_text_section", "elements": [{"type": "text", "text": "quoted line"}]}
                ]},
                {"type": "rich_text_list", "style": "bullet", "elements": [
                    {"type": "rich_text_section", "elements": [{"type": "text", "text": "item one"}]}
                ]}
            ]
        })];
        let out = extract_text_from_slack_blocks(&blocks);
        assert!(out.contains("Hello"));
        assert!(out.contains("> quoted line"));
        assert!(out.contains("\u{2022} item one"));
    }

    #[test]
    fn blocks_text_renders_link_and_user() {
        let blocks = vec![json!({
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [
                {"type": "text", "text": "see "},
                {"type": "link", "url": "https://x.test", "text": "X"},
                {"type": "text", "text": " ping "},
                {"type": "user", "user_id": "U1"}
            ]}]
        })];
        let out = extract_text_from_slack_blocks(&blocks);
        assert_eq!(out, "see X (https://x.test) ping <@U1>");
    }

    #[test]
    fn serialize_skips_pure_rich_text() {
        let blocks = vec![json!({"type": "rich_text", "elements": []})];
        assert_eq!(serialize_slack_blocks_for_agent_default(&blocks), "");
    }

    #[test]
    fn serialize_redacts_to_allowlist() {
        let blocks = vec![json!({
            "type": "section",
            "block_id": "b1",
            "secret": "should-drop",
            "text": {"type": "mrkdwn", "text": "hi"}
        })];
        let out = serialize_slack_blocks_for_agent_default(&blocks);
        assert!(out.contains("Slack Block Kit payload"));
        assert!(out.contains("\"block_id\""));
        assert!(out.contains("\"hi\""));
        assert!(!out.contains("secret"));
    }

    #[test]
    fn format_message_links_and_bold() {
        let out = format_message("see [docs](https://x.test) and **bold**");
        assert!(out.contains("<https://x.test|docs>"));
        assert!(out.contains("*bold*"));
    }

    #[test]
    fn format_message_protects_code() {
        let out = format_message("text `**not bold**` more");
        assert!(out.contains("`**not bold**`"));
    }

    #[test]
    fn format_message_header_to_bold() {
        let out = format_message("## Title");
        assert_eq!(out, "*Title*");
    }

    #[test]
    fn format_message_italic_single_star() {
        let out = format_message("an *emphasized* word");
        assert!(out.contains("_emphasized_"));
    }

    #[test]
    fn format_message_escapes_amp_lt_gt() {
        let out = format_message("a & b < c > d");
        assert!(out.contains("&amp;"));
        assert!(out.contains("&lt;"));
        assert!(out.contains("&gt;"));
    }

    #[test]
    fn describe_api_error_missing_scope() {
        let resp = json!({"error": "missing_scope", "needed": "files:read", "provided": "chat:write"});
        let file = json!({"name": "x.png"});
        let msg = describe_slack_api_error(Some(&resp), Some(&file)).unwrap();
        assert!(msg.contains("x.png"));
        assert!(msg.contains("Missing scope: files:read."));
        assert!(msg.contains("Current bot scopes: chat:write."));
    }

    #[test]
    fn describe_api_error_no_error_returns_none() {
        let resp = json!({"ok": true});
        assert!(describe_slack_api_error(Some(&resp), None).is_none());
    }

    #[test]
    fn download_failure_http_403() {
        let file = json!({"id": "F1"});
        let msg = describe_slack_download_failure_message(Some(403), "boom", Some(&file)).unwrap();
        assert!(msg.contains("HTTP 403"));
        assert!(msg.contains("F1"));
    }

    #[test]
    fn require_mention_defaults_and_overrides() {
        let mut extra = HashMap::new();
        assert!(require_mention(&extra, None));
        assert!(!require_mention(&extra, Some("false")));
        extra.insert("require_mention".to_string(), Value::Bool(false));
        assert!(!require_mention(&extra, None));
        extra.insert("require_mention".to_string(), Value::String("off".to_string()));
        assert!(!require_mention(&extra, None));
    }

    #[test]
    fn strict_mention_defaults_false() {
        let extra = HashMap::new();
        assert!(!strict_mention(&extra, None));
        assert!(strict_mention(&extra, Some("yes")));
    }

    #[test]
    fn free_response_channels_csv_and_list_and_numeric() {
        let mut extra = HashMap::new();
        assert_eq!(free_response_channels(&extra, Some("C1, C2 ,")), vec!["C1", "C2"]);
        extra.insert("free_response_channels".to_string(), json!(["C3", " C4 "]));
        assert_eq!(free_response_channels(&extra, None), vec!["C3", "C4"]);
        extra.insert("free_response_channels".to_string(), json!(1234567890i64));
        assert_eq!(free_response_channels(&extra, None), vec!["1234567890"]);
    }

    #[test]
    fn reactions_enabled_parsing() {
        assert!(reactions_enabled(None));
        assert!(!reactions_enabled(Some("false")));
        assert!(!reactions_enabled(Some("NO")));
    }

    #[test]
    fn allow_bots_mode_resolution() {
        let mut extra = HashMap::new();
        assert_eq!(allow_bots_mode(&extra, None), "none");
        assert_eq!(allow_bots_mode(&extra, Some("mentions")), "mentions");
        extra.insert("allow_bots".to_string(), Value::String("ALL".to_string()));
        assert_eq!(allow_bots_mode(&extra, Some("mentions")), "all");
    }

    #[test]
    fn dm_top_level_threads_default_true() {
        let mut extra = HashMap::new();
        assert!(dm_top_level_threads_as_sessions(&extra));
        extra.insert("dm_top_level_threads_as_sessions".to_string(), Value::String("false".to_string()));
        assert!(!dm_top_level_threads_as_sessions(&extra));
    }

    #[test]
    fn resolve_thread_ts_prefers_metadata() {
        let extra = HashMap::new();
        let mut md = HashMap::new();
        md.insert("thread_id".to_string(), json!("T1"));
        assert_eq!(resolve_thread_ts(&extra, Some("R1"), Some(&md)), Some("T1".to_string()));
        assert_eq!(resolve_thread_ts(&extra, Some("R1"), None), Some("R1".to_string()));
    }

    #[test]
    fn resolve_thread_ts_reply_in_thread_disabled() {
        let mut extra = HashMap::new();
        extra.insert("reply_in_thread".to_string(), Value::Bool(false));
        let mut md = HashMap::new();
        md.insert("thread_id".to_string(), json!("R1"));
        // synthetic thread (== reply_to) → None
        assert_eq!(resolve_thread_ts(&extra, Some("R1"), Some(&md)), None);
        // real thread differs from reply_to → kept
        md.insert("thread_id".to_string(), json!("T2"));
        assert_eq!(resolve_thread_ts(&extra, Some("R1"), Some(&md)), Some("T2".to_string()));
    }

    #[test]
    fn mention_helpers() {
        assert!(is_mentioned(Some("U1"), "hi <@U1> there"));
        assert!(!is_mentioned(Some("U1"), "no mention"));
        assert!(!is_mentioned(None, "<@U1>"));
        assert_eq!(strip_bot_mention("hi <@U1> there", "U1"), "hi  there");
        assert!(is_thread_reply(Some("T1"), "TS2"));
        assert!(!is_thread_reply(Some("TS"), "TS"));
    }

    #[test]
    fn route_slash_native() {
        let map = HashMap::new();
        assert_eq!(route_slash_command_text("stop", "now", &map), "/stop now");
        assert_eq!(route_slash_command_text("stop", "", &map), "/stop");
    }

    #[test]
    fn route_slash_hermes_subcommand_and_freeform() {
        let mut map = HashMap::new();
        map.insert("compact".to_string(), "/compress".to_string());
        assert_eq!(route_slash_command_text("hermes", "compact extra", &map), "/compress extra");
        assert_eq!(route_slash_command_text("hermes", "compact", &map), "/compress");
        assert_eq!(route_slash_command_text("hermes", "what is up", &map), "what is up");
        assert_eq!(route_slash_command_text("hermes", "", &map), "/help");
    }

    #[test]
    fn slash_pattern_build() {
        assert_eq!(slash_command_pattern(&[]), "^/hermes$");
        let names = vec!["btw".to_string(), "stop".to_string()];
        assert_eq!(slash_command_pattern(&names), "^/(?:btw|stop)$");
    }

    #[test]
    fn slash_context_pop_precise_and_fallback() {
        let mut ctx = SlashCommandContexts::new();
        ctx.stash("C1", "U1", "https://r1", 100.0);
        ctx.stash("C1", "U2", "https://r2", 100.0);
        // precise match
        let popped = ctx.pop("C1", Some("U2"), 101.0).unwrap();
        assert_eq!(popped.response_url, "https://r2");
        assert_eq!(ctx.len(), 1);
        // fallback (no uid) grabs remaining channel entry
        let popped2 = ctx.pop("C1", None, 101.0).unwrap();
        assert_eq!(popped2.response_url, "https://r1");
        assert!(ctx.is_empty());
    }

    #[test]
    fn slash_context_stale_eviction() {
        let mut ctx = SlashCommandContexts::new();
        ctx.stash("C1", "U1", "https://r1", 0.0);
        // now beyond TTL → evicted, pop returns None
        assert!(ctx.pop("C1", Some("U1"), SLASH_CTX_TTL + 1.0).is_none());
        assert!(ctx.is_empty());
    }

    #[test]
    fn exec_approval_blocks_shape() {
        let blocks = build_exec_approval_blocks("rm -rf /", "dangerous", "SK1");
        let arr = blocks.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        let buttons = arr[1]["elements"].as_array().unwrap();
        assert_eq!(buttons.len(), 4);
        assert_eq!(buttons[0]["action_id"], "hermes_approve_once");
        assert_eq!(buttons[0]["value"], "SK1");
        assert!(arr[0]["text"]["text"].as_str().unwrap().contains("```rm -rf /```"));
    }

    #[test]
    fn slash_confirm_blocks_and_value_parse() {
        let blocks = build_slash_confirm_blocks("Title", "do it?", "SK", "CID");
        let buttons = blocks[1]["elements"].as_array().unwrap();
        assert_eq!(buttons[0]["value"], "SK|CID");
        assert_eq!(parse_slash_confirm_value("SK|CID"), Some(("SK".to_string(), "CID".to_string())));
        assert_eq!(parse_slash_confirm_value("nopipe"), None);
    }

    #[test]
    fn decision_labels() {
        assert_eq!(approval_choice("hermes_approve_always"), "always");
        assert_eq!(approval_choice("unknown"), "deny");
        assert!(approval_decision_label("deny", "Bob").contains("Denied by Bob"));
        assert_eq!(slash_confirm_choice("hermes_confirm_cancel"), "cancel");
        assert!(slash_confirm_decision_label("always", "Ann").contains("Always approved by Ann"));
    }

    #[test]
    fn section_text_extraction() {
        let blocks = vec![
            json!({"type": "actions"}),
            json!({"type": "section", "text": {"type": "mrkdwn", "text": "original"}}),
        ];
        assert_eq!(extract_section_text(&blocks), "original");
    }

    #[test]
    fn button_auth() {
        assert!(button_click_authorized("", "U1"));
        assert!(button_click_authorized("*", "Uother"));
        assert!(button_click_authorized("U1, U2", "U2"));
        assert!(!button_click_authorized("U1, U2", "U3"));
    }

    #[test]
    fn assistant_metadata_extraction() {
        let event = json!({
            "assistant_thread": {"channel_id": "C1", "thread_ts": "T1", "user_id": "U1"},
            "team": "TEAM1",
            "context": {"channel_id": "CTX1"}
        });
        let md = extract_assistant_thread_metadata(&event);
        assert_eq!(md["channel_id"], "C1");
        assert_eq!(md["thread_ts"], "T1");
        assert_eq!(md["user_id"], "U1");
        assert_eq!(md["team_id"], "TEAM1");
        assert_eq!(md["context_channel_id"], "CTX1");
    }

    #[test]
    fn retryable_upload_errors() {
        assert!(is_retryable_upload_error(Some(429), ""));
        assert!(is_retryable_upload_error(Some(503), ""));
        assert!(!is_retryable_upload_error(Some(404), ""));
        assert!(is_retryable_upload_error(None, "Slack rate_limited"));
        assert!(is_retryable_upload_error(None, "connection reset by peer"));
        assert!(!is_retryable_upload_error(None, "invalid_arguments"));
    }

    #[test]
    fn thread_context_formatting() {
        let entries = vec![
            ThreadContextEntry { display_name: "alice".into(), text: "hello".into(), is_parent: true },
            ThreadContextEntry { display_name: "bob".into(), text: "hi".into(), is_parent: false },
        ];
        let out = format_thread_context(&entries);
        assert!(out.contains("[thread parent] alice: hello"));
        assert!(out.contains("bob: hi"));
        assert!(out.starts_with("[Thread context"));
        assert!(out.ends_with("[End of thread context]\n\n"));
        assert_eq!(format_thread_context(&[]), "");
    }

    #[test]
    fn proxy_resolution() {
        assert_eq!(resolve_slack_proxy_url(None, |_| false), None);
        assert_eq!(resolve_slack_proxy_url(Some("socks5://x"), |_| false), None);
        assert_eq!(
            resolve_slack_proxy_url(Some("http://proxy:8080"), |_| false),
            Some("http://proxy:8080".to_string())
        );
        // NO_PROXY excludes a slack host → None
        assert_eq!(resolve_slack_proxy_url(Some("http://proxy:8080"), |_| true), None);
    }

    #[test]
    fn subtype_and_bot_helpers() {
        assert!(is_ignored_subtype(Some("message_changed")));
        assert!(!is_ignored_subtype(Some("file_share")));
        assert!(is_bot_message(&json!({"bot_id": "B1"})));
        assert!(is_bot_message(&json!({"subtype": "bot_message"})));
        assert!(!is_bot_message(&json!({"user": "U1"})));
        assert!(resolve_is_dm("", "D123"));
        assert!(resolve_is_dm("mpim", "C1"));
        assert!(!resolve_is_dm("", "C1"));
    }

    #[test]
    fn cache_key_and_dm_helpers() {
        assert_eq!(thread_context_cache_key("C1", "T1", "TEAM"), "C1:T1:TEAM");
        assert!(is_dm_channel("D9"));
        assert!(!is_dm_channel("C9"));
        assert!(cache_is_fresh(0.0, 10.0));
        assert!(!cache_is_fresh(0.0, THREAD_CACHE_TTL + 1.0));
    }
}
