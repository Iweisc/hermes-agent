//! Telegram platform adapter — native Rust port of
//! `gateway/platforms/telegram.py`.
//!
//! The original Python module is built on the `python-telegram-bot` (PTB)
//! library, whose async `Application`/`Updater` polling+webhook lifecycle,
//! `httpx`-backed connection pools, inline-keyboard callback dispatch loop,
//! and media download coroutines have no faithful equivalent in the crates
//! available to this port. As with the sibling `gw_platforms_discord`
//! module, this file ports the **deterministic, side-effect-free logic**
//! that other Hermes code (and tests) depend on, reproducing the Python
//! behaviour exactly:
//!
//! - MarkdownV2 escaping / stripping: [`escape_mdv2`], [`strip_mdv2`].
//! - GFM-table rewriting for Telegram: [`is_table_row`],
//!   [`split_markdown_table_row`], [`render_table_block_for_telegram`],
//!   [`wrap_markdown_tables`].
//! - Full standard-markdown → MarkdownV2 conversion: [`format_message`].
//! - Thread-id mapping for the forum *General* topic:
//!   [`metadata_thread_id`], [`message_thread_id_for_send`],
//!   [`message_thread_id_for_typing`].
//! - Error classification: [`is_thread_not_found_error`],
//!   [`looks_like_polling_conflict`], [`looks_like_network_error`].
//! - Config / env gating: [`coerce_bool_extra`], [`reply_to_mode`],
//!   [`telegram_require_mention`], [`telegram_free_response_chats`],
//!   [`telegram_ignored_threads`], [`reactions_enabled`],
//!   [`disable_link_previews`], [`fallback_ips_from_config`],
//!   [`media_batch_delay_seconds`], [`text_batch_delay_seconds`],
//!   [`text_batch_split_delay_seconds`], the various `env_*` helpers, and the
//!   webhook config readers.
//! - Inline-button callback-data parsing/decisions:
//!   [`parse_callback_kind`], [`CallbackKind`], [`approval_label`],
//!   [`slash_confirm_label`], [`update_prompt_label`].
//! - Model-picker pagination: [`build_model_keyboard`], [`ModelKeyboard`],
//!   [`InlineButton`], [`model_short_label`].
//! - Mention / chunk-threading helpers: [`should_thread_reply`],
//!   [`clean_bot_trigger_text`], [`is_group_chat_type`],
//!   [`mention_matches`].
//! - Media helpers: [`missing_media_path_error`], [`is_supported_video_ext`],
//!   [`caption_truncate`], [`chunk_size_suffix_escape`].
//!
//! The PTB application bootstrap, polling/webhook reconnect ladder, and media
//! download/upload coroutines are out of scope — they require the live PTB
//! client. Callers wire those through the Python bridge; this module supplies
//! the pure logic those paths call into.

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;

use regex::Regex;

// ─── Constants (mirror class-level Python constants) ────────────────────────

/// Telegram single-message character limit (`MAX_MESSAGE_LENGTH`).
pub const MAX_MESSAGE_LENGTH: usize = 4096;

/// Threshold for detecting Telegram client-side message splits
/// (`_SPLIT_THRESHOLD`). When a chunk is near this limit, a continuation is
/// almost certain.
pub const SPLIT_THRESHOLD: usize = 4000;

/// Album/media-group debounce window in seconds (`MEDIA_GROUP_WAIT_SECONDS`).
pub const MEDIA_GROUP_WAIT_SECONDS: f64 = 0.8;

/// The forum *General* topic thread id is represented as `"1"` on the wire
/// (`_GENERAL_TOPIC_THREAD_ID`).
pub const GENERAL_TOPIC_THREAD_ID: &str = "1";

/// Page size for the interactive model picker (`_MODEL_PAGE_SIZE`).
pub const MODEL_PAGE_SIZE: usize = 8;

// ─── MarkdownV2 escaping / stripping ────────────────────────────────────────

/// Matches every character that MarkdownV2 requires to be backslash-escaped
/// when it appears outside a code span or fenced code block
/// (`_MDV2_ESCAPE_RE`). Characters: `_ * [ ] ( ) ~ \` > # + - = | { } . ! \`.
pub fn mdv2_escape_char(c: char) -> bool {
    matches!(
        c,
        '_' | '*'
            | '['
            | ']'
            | '('
            | ')'
            | '~'
            | '`'
            | '>'
            | '#'
            | '+'
            | '-'
            | '='
            | '|'
            | '{'
            | '}'
            | '.'
            | '!'
            | '\\'
    )
}

/// Escape Telegram MarkdownV2 special characters with a preceding backslash
/// (`_escape_mdv2`).
pub fn escape_mdv2(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if mdv2_escape_char(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Strip MarkdownV2 escape backslashes to produce clean plain text, and remove
/// MarkdownV2 formatting markers so the fallback doesn't show stray syntax
/// characters from `format_message` conversion (`_strip_mdv2`).
pub fn strip_mdv2(text: &str) -> String {
    // Remove escape backslashes before special characters.
    let re_escape = Regex::new(r"\\([_*\[\]()~`>#\+\-=|{}.!\\])").unwrap();
    let mut cleaned = re_escape.replace_all(text, "$1").into_owned();
    // Remove MarkdownV2 bold markers (*text* -> text).
    let re_bold = Regex::new(r"\*([^*]+)\*").unwrap();
    cleaned = re_bold.replace_all(&cleaned, "$1").into_owned();
    // Remove MarkdownV2 italic markers (_text_ -> text) without breaking
    // snake_case. Python uses (?<!\w)_([^_]+)_(?!\w); Rust's regex crate lacks
    // lookaround, so emulate the word-boundary semantics manually.
    cleaned = strip_italic_markers(&cleaned);
    // Remove strikethrough markers (~text~ -> text).
    let re_strike = Regex::new(r"~([^~]+)~").unwrap();
    cleaned = re_strike.replace_all(&cleaned, "$1").into_owned();
    // Remove spoiler markers (||text|| -> text).
    let re_spoiler = Regex::new(r"\|\|([^|]+)\|\|").unwrap();
    cleaned = re_spoiler.replace_all(&cleaned, "$1").into_owned();
    cleaned
}

/// Emulates Python's `(?<!\w)_([^_]+)_(?!\w)` italic-marker stripping, which
/// only removes a `_..._` pair when neither underscore is adjacent to a word
/// character (so `my_var_name` is left intact but `_italic_` is unwrapped).
fn strip_italic_markers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    while i < n {
        if chars[i] == '_' {
            // Left boundary: previous char must not be a word char (Python's _
            // counts as \w, but the opening `_` itself is being matched, so the
            // char before the opening underscore is what matters).
            let left_ok = i == 0 || !is_word(chars[i - 1]);
            if left_ok {
                // Find closing underscore: content [^_]+ (at least one char).
                let mut j = i + 1;
                while j < n && chars[j] != '_' {
                    j += 1;
                }
                if j < n && j > i + 1 {
                    // chars[j] is the closing underscore. Right boundary: char
                    // after closing underscore must not be a word char.
                    let right_ok = j + 1 >= n || !is_word(chars[j + 1]);
                    if right_ok {
                        // Emit the inner content unwrapped.
                        for &c in &chars[i + 1..j] {
                            out.push(c);
                        }
                        i = j + 1;
                        continue;
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// ─── GFM table → Telegram-friendly row groups ───────────────────────────────

/// Matches a GFM table delimiter row (`_TABLE_SEPARATOR_RE`): optional outer
/// pipes, cells of only dashes (with optional leading/trailing colons),
/// separated by `|`, requiring at least one internal `|`.
fn table_separator_re() -> Regex {
    Regex::new(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$").unwrap()
}

/// Return true if `line` could plausibly be a table data row (`_is_table_row`).
pub fn is_table_row(line: &str) -> bool {
    let stripped = line.trim();
    !stripped.is_empty() && stripped.contains('|')
}

/// Split a simple GFM table row into stripped cell values
/// (`_split_markdown_table_row`).
pub fn split_markdown_table_row(line: &str) -> Vec<String> {
    let mut stripped = line.trim().to_string();
    if stripped.starts_with('|') {
        stripped = stripped[1..].to_string();
    }
    if stripped.ends_with('|') {
        stripped = stripped[..stripped.len() - 1].to_string();
    }
    stripped.split('|').map(|c| c.trim().to_string()).collect()
}

/// Render a detected GFM table as Telegram-friendly row groups
/// (`_render_table_block_for_telegram`).
pub fn render_table_block_for_telegram(table_block: &[String]) -> String {
    if table_block.len() < 3 {
        return table_block.join("\n");
    }
    let headers = split_markdown_table_row(&table_block[0]);
    if headers.len() < 2 {
        return table_block.join("\n");
    }

    let mut rendered_rows: Vec<String> = Vec::new();
    for (index0, row) in table_block[2..].iter().enumerate() {
        let index = index0 + 1;
        let mut cells = split_markdown_table_row(row);
        if cells.len() < headers.len() {
            cells.resize(headers.len(), String::new());
        } else if cells.len() > headers.len() {
            cells.truncate(headers.len());
        }
        let heading = cells
            .iter()
            .find(|c| !c.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("Row {index}"));
        rendered_rows.push(format!("**{heading}**"));
        for (header, value) in headers.iter().zip(cells.iter()) {
            rendered_rows.push(format!("• {header}: {value}"));
        }
    }

    rendered_rows.join("\n\n")
}

/// Rewrite GFM-style pipe tables into Telegram-friendly bullet groups
/// (`_wrap_markdown_tables`). Tables inside existing fenced code blocks are
/// left untouched.
pub fn wrap_markdown_tables(text: &str) -> String {
    if !text.contains('|') || !text.contains('-') {
        return text.to_string();
    }
    let sep_re = table_separator_re();
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let stripped = line.trim_start();

        if stripped.starts_with("```") {
            in_fence = !in_fence;
            out.push(line.to_string());
            i += 1;
            continue;
        }
        if in_fence {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        if line.contains('|') && i + 1 < lines.len() && sep_re.is_match(lines[i + 1]) {
            let mut table_block: Vec<String> = vec![line.to_string(), lines[i + 1].to_string()];
            let mut j = i + 2;
            while j < lines.len() && is_table_row(lines[j]) {
                table_block.push(lines[j].to_string());
                j += 1;
            }
            out.push(render_table_block_for_telegram(&table_block));
            i = j;
            continue;
        }

        out.push(line.to_string());
        i += 1;
    }

    out.join("\n")
}

// ─── format_message: standard markdown → Telegram MarkdownV2 ─────────────────

/// Convert standard markdown to Telegram MarkdownV2 format
/// (`TelegramAdapter.format_message`).
///
/// Protected regions (code blocks, inline code) are extracted first so their
/// contents are never modified. Standard markdown constructs (headers, bold,
/// italic, links, strikethrough, spoilers, blockquotes) are translated to
/// MarkdownV2 syntax, and all remaining special characters are escaped.
pub fn format_message(content: &str) -> String {
    if content.is_empty() {
        return content.to_string();
    }

    let mut placeholders: Vec<(String, String)> = Vec::new();
    // Use a closure-free counter so we can stash placeholders deterministically.
    let mut counter: usize = 0;
    let mut stash = |placeholders: &mut Vec<(String, String)>, counter: &mut usize, value: String| -> String {
        let key = format!("\u{0}PH{}\u{0}", *counter);
        *counter += 1;
        placeholders.push((key.clone(), value));
        key
    };

    let mut text = content.to_string();

    // 0) Rewrite GFM-style pipe tables into Telegram-friendly row groups.
    text = wrap_markdown_tables(&text);

    // 1) Protect fenced code blocks (``` ... ```).
    {
        let re = Regex::new(r"(?s)(```(?:[^\n]*\n)?.*?```)").unwrap();
        let mut result = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            result.push_str(&text[last..m.start()]);
            let raw = m.as_str();
            // Find opening line end.
            let after_three = &raw[3..];
            let open_end = if let Some(pos) = after_three.find('\n') {
                3 + pos + 1
            } else {
                3
            };
            let opening = &raw[..open_end];
            let body_and_close = &raw[open_end..];
            // Strip trailing ```
            let body = &body_and_close[..body_and_close.len().saturating_sub(3)];
            let body = body.replace('\\', "\\\\").replace('`', "\\`");
            let value = format!("{opening}{body}```");
            let key = stash(&mut placeholders, &mut counter, value);
            result.push_str(&key);
            last = m.end();
        }
        result.push_str(&text[last..]);
        text = result;
    }

    // 2) Protect inline code (`...`).
    {
        let re = Regex::new(r"(`[^`]+`)").unwrap();
        let mut result = String::new();
        let mut last = 0usize;
        for m in re.find_iter(&text) {
            result.push_str(&text[last..m.start()]);
            let value = m.as_str().replace('\\', "\\\\");
            let key = stash(&mut placeholders, &mut counter, value);
            result.push_str(&key);
            last = m.end();
        }
        result.push_str(&text[last..]);
        text = result;
    }

    // 3) Convert markdown links.
    {
        let re = Regex::new(r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)").unwrap();
        let mut result = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&text) {
            let m = caps.get(0).unwrap();
            result.push_str(&text[last..m.start()]);
            let display = escape_mdv2(caps.get(1).map(|x| x.as_str()).unwrap_or(""));
            let url = caps
                .get(2)
                .map(|x| x.as_str())
                .unwrap_or("")
                .replace('\\', "\\\\")
                .replace(')', "\\)");
            let value = format!("[{display}]({url})");
            let key = stash(&mut placeholders, &mut counter, value);
            result.push_str(&key);
            last = m.end();
        }
        result.push_str(&text[last..]);
        text = result;
    }

    // 4) Convert markdown headers (## Title) → bold *Title*.
    {
        let re = Regex::new(r"(?m)^#{1,6}\s+(.+)$").unwrap();
        let bold_re = Regex::new(r"(?s)\*\*(.+?)\*\*").unwrap();
        let mut result = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&text) {
            let m = caps.get(0).unwrap();
            result.push_str(&text[last..m.start()]);
            let inner = caps.get(1).map(|x| x.as_str()).unwrap_or("").trim().to_string();
            let inner = bold_re.replace_all(&inner, "$1").into_owned();
            let value = format!("*{}*", escape_mdv2(&inner));
            let key = stash(&mut placeholders, &mut counter, value);
            result.push_str(&key);
            last = m.end();
        }
        result.push_str(&text[last..]);
        text = result;
    }

    // 5) Convert bold: **text** → *text*.
    text = replace_captured(&text, r"(?s)\*\*(.+?)\*\*", |inner| {
        format!("*{}*", escape_mdv2(inner))
    }, &mut placeholders, &mut counter, &mut stash);

    // 6) Convert italic: *text* (single asterisk) → _text_.
    text = replace_captured(&text, r"\*([^*\n]+)\*", |inner| {
        format!("_{}_", escape_mdv2(inner))
    }, &mut placeholders, &mut counter, &mut stash);

    // 7) Convert strikethrough: ~~text~~ → ~text~.
    text = replace_captured(&text, r"(?s)~~(.+?)~~", |inner| {
        format!("~{}~", escape_mdv2(inner))
    }, &mut placeholders, &mut counter, &mut stash);

    // 8) Convert spoiler: ||text|| → ||text||.
    text = replace_captured(&text, r"(?s)\|\|(.+?)\|\|", |inner| {
        format!("||{}||", escape_mdv2(inner))
    }, &mut placeholders, &mut counter, &mut stash);

    // 9) Convert blockquotes.
    {
        let re = Regex::new(r"(?m)^((?:\*\*)?>{1,3}) (.+)$").unwrap();
        let mut result = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&text) {
            let m = caps.get(0).unwrap();
            result.push_str(&text[last..m.start()]);
            let prefix = caps.get(1).map(|x| x.as_str()).unwrap_or("");
            let content_bq = caps.get(2).map(|x| x.as_str()).unwrap_or("");
            let value = if prefix.starts_with("**") && content_bq.ends_with("||") {
                let inner = &content_bq[..content_bq.len() - 2];
                format!("{prefix} {}||", escape_mdv2(inner))
            } else {
                format!("{prefix} {}", escape_mdv2(content_bq))
            };
            let key = stash(&mut placeholders, &mut counter, value);
            result.push_str(&key);
            last = m.end();
        }
        result.push_str(&text[last..]);
        text = result;
    }

    // 10) Escape remaining special characters in plain text.
    text = escape_mdv2(&text);

    // 11) Restore placeholders in reverse insertion order.
    for (key, value) in placeholders.iter().rev() {
        text = text.replace(key, value);
    }

    // 12) Safety net: escape unescaped ( ) { } that slipped through, while
    //     leaving content inside ``` or ` spans untouched.
    {
        let split_re = Regex::new(r"(?s)(```.*?```|`[^`]+`)").unwrap();
        let mut safe_parts: Vec<String> = Vec::new();
        let mut last = 0usize;
        let mut idx = 0usize;
        for m in split_re.find_iter(&text) {
            // Outside segment.
            let outside = &text[last..m.start()];
            safe_parts.push(escape_bare_brackets(outside));
            idx += 1;
            // Inside segment — untouched.
            safe_parts.push(m.as_str().to_string());
            idx += 1;
            last = m.end();
        }
        let _ = idx;
        safe_parts.push(escape_bare_brackets(&text[last..]));
        text = safe_parts.concat();
    }

    text
}

/// Helper for `format_message` steps 5–8: apply a regex over the whole text,
/// stash each transformed match behind a placeholder.
#[allow(clippy::too_many_arguments)]
fn replace_captured<F, S>(
    text: &str,
    pattern: &str,
    transform: F,
    placeholders: &mut Vec<(String, String)>,
    counter: &mut usize,
    stash: &mut S,
) -> String
where
    F: Fn(&str) -> String,
    S: FnMut(&mut Vec<(String, String)>, &mut usize, String) -> String,
{
    let re = Regex::new(pattern).unwrap();
    let mut result = String::new();
    let mut last = 0usize;
    for caps in re.captures_iter(text) {
        let m = caps.get(0).unwrap();
        result.push_str(&text[last..m.start()]);
        let inner = caps.get(1).map(|x| x.as_str()).unwrap_or("");
        let value = transform(inner);
        let key = stash(placeholders, counter, value);
        result.push_str(&key);
        last = m.end();
    }
    result.push_str(&text[last..]);
    result
}

/// Escape bare `( ) { }` outside code spans, mirroring `_esc_bare` in step 12
/// of `format_message`. Already-escaped chars, the `(` that opens a markdown
/// link `[text](url)`, and the `)` that closes a link URL are left as-is.
fn escape_bare_brackets(seg: &str) -> String {
    let chars: Vec<char> = seg.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(seg.len());
    let mut s = 0usize;
    while s < n {
        let ch = chars[s];
        if ch == '(' || ch == ')' || ch == '{' || ch == '}' {
            // Already escaped?
            if s > 0 && chars[s - 1] == '\\' {
                out.push(ch);
                s += 1;
                continue;
            }
            // ( that opens a MarkdownV2 link [text](url).
            if ch == '(' && s > 0 && chars[s - 1] == ']' {
                out.push(ch);
                s += 1;
                continue;
            }
            // ) that closes a link URL.
            if ch == ')' {
                let before: String = chars[..s].iter().collect();
                if before.contains("](http") || before.contains("](") {
                    let mut depth: i32 = 0;
                    let lower = s.saturating_sub(2000);
                    let mut handled = false;
                    let mut j = s as isize - 1;
                    while j >= lower as isize {
                        let jj = j as usize;
                        if chars[jj] == '(' {
                            depth -= 1;
                            if depth < 0 {
                                if jj > 0 && chars[jj - 1] == ']' {
                                    out.push(ch);
                                    handled = true;
                                }
                                break;
                            }
                        } else if chars[jj] == ')' {
                            depth += 1;
                        }
                        j -= 1;
                    }
                    if handled {
                        s += 1;
                        continue;
                    }
                }
            }
            out.push('\\');
            out.push(ch);
        } else {
            out.push(ch);
        }
        s += 1;
    }
    out
}

// ─── Thread-id mapping for the forum General topic ──────────────────────────

/// Extract a thread id from message metadata (`_metadata_thread_id`). Looks up
/// `thread_id` then `message_thread_id`.
pub fn metadata_thread_id(metadata: &HashMap<String, String>) -> Option<String> {
    if metadata.is_empty() {
        return None;
    }
    metadata
        .get("thread_id")
        .or_else(|| metadata.get("message_thread_id"))
        .cloned()
}

/// Map a thread id to the value used for `message_thread_id` on send
/// (`_message_thread_id_for_send`). The forum *General* topic (id `"1"`) is
/// represented as `None` on the wire; user-created topics keep their numeric
/// id.
pub fn message_thread_id_for_send(thread_id: Option<&str>) -> Option<i64> {
    match thread_id {
        None => None,
        Some(t) if t.is_empty() || t == GENERAL_TOPIC_THREAD_ID => None,
        Some(t) => t.parse::<i64>().ok(),
    }
}

/// Mirrors [`message_thread_id_for_send`] for typing indicators
/// (`_message_thread_id_for_typing`).
pub fn message_thread_id_for_typing(thread_id: Option<&str>) -> Option<i64> {
    message_thread_id_for_send(thread_id)
}

// ─── Error classification ───────────────────────────────────────────────────

/// Return whether an error message indicates a missing forum thread
/// (`_is_thread_not_found_error`).
pub fn is_thread_not_found_error(error: &str) -> bool {
    error.to_lowercase().contains("thread not found")
}

/// Detect a Telegram getUpdates polling conflict (`_looks_like_polling_conflict`).
///
/// `error_class_name` is the lowercased exception class name (PTB raises a
/// `Conflict`); `error_text` is the stringified error message.
pub fn looks_like_polling_conflict(error_class_name: &str, error_text: &str) -> bool {
    let text = error_text.to_lowercase();
    error_class_name.to_lowercase() == "conflict"
        || text.contains("terminated by other getupdates request")
        || text.contains("another bot instance is running")
}

/// Return true for transient network errors that warrant a reconnect attempt
/// (`_looks_like_network_error`).
///
/// `error_class_name` is the lowercased exception class name; `is_os_error`
/// signals that the underlying error is an `OSError`/`io::Error` equivalent.
pub fn looks_like_network_error(error_class_name: &str, is_os_error: bool) -> bool {
    let name = error_class_name.to_lowercase();
    matches!(name.as_str(), "networkerror" | "timedout" | "connectionerror") || is_os_error
}

// ─── Config / env helpers ───────────────────────────────────────────────────

/// Parse a boolean-ish value the way Python's `_coerce_bool_extra` does:
/// `None` → default; string interpreted case-insensitively
/// (`true/1/yes/on` → true, `false/0/no/off` → false, otherwise default);
/// any other value coerced via truthiness (here: non-empty string → true).
pub fn coerce_bool_value(value: Option<&str>, default: bool) -> bool {
    match value {
        None => default,
        Some(v) => {
            let lowered = v.trim().to_lowercase();
            match lowered.as_str() {
                "true" | "1" | "yes" | "on" => true,
                "false" | "0" | "no" | "off" => false,
                _ => default,
            }
        }
    }
}

/// Read a bool-ish key from a config `extra` map (`_coerce_bool_extra`).
pub fn coerce_bool_extra(extra: &HashMap<String, String>, key: &str, default: bool) -> bool {
    coerce_bool_value(extra.get(key).map(|s| s.as_str()), default)
}

/// Normalise the `reply_to_mode` config value, defaulting to `"first"`.
pub fn reply_to_mode(configured: Option<&str>) -> String {
    match configured {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => "first".to_string(),
    }
}

/// Determine whether this chunk should thread to the original message
/// (`_should_thread_reply`).
pub fn should_thread_reply(reply_to: Option<&str>, mode: &str, chunk_index: usize) -> bool {
    if reply_to.is_none() {
        return false;
    }
    match mode {
        "off" => false,
        "all" => true,
        _ => chunk_index == 0, // "first" (default)
    }
}

/// Whether group chats should require an explicit bot trigger
/// (`_telegram_require_mention`). `configured` is the `extra.require_mention`
/// value; falls back to `TELEGRAM_REQUIRE_MENTION` env when unset.
pub fn telegram_require_mention(configured: Option<&str>) -> bool {
    if let Some(v) = configured {
        return matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on");
    }
    let env_val = env::var("TELEGRAM_REQUIRE_MENTION").unwrap_or_default();
    matches!(env_val.to_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

/// Parse the `free_response_chats` set (`_telegram_free_response_chats`).
/// `raw` is the config value (comma-separated or list-joined); falls back to
/// `TELEGRAM_FREE_RESPONSE_CHATS` env when `None`.
pub fn telegram_free_response_chats(raw: Option<&str>) -> HashSet<String> {
    let value = match raw {
        Some(v) => v.to_string(),
        None => env::var("TELEGRAM_FREE_RESPONSE_CHATS").unwrap_or_default(),
    };
    value
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Parse the `ignored_threads` set (`_telegram_ignored_threads`). Non-numeric
/// entries are skipped (the Python logs a warning). Falls back to
/// `TELEGRAM_IGNORED_THREADS` env when `raw` is `None`.
pub fn telegram_ignored_threads(raw: Option<&str>) -> HashSet<i64> {
    let value = match raw {
        Some(v) => v.to_string(),
        None => env::var("TELEGRAM_IGNORED_THREADS").unwrap_or_default(),
    };
    let mut out = HashSet::new();
    for part in value.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(n) = t.parse::<i64>() {
            out.insert(n);
        }
    }
    out
}

/// Whether message reactions are enabled (`_reactions_enabled`). True unless
/// `TELEGRAM_REACTIONS` is one of `false/0/no` (case-insensitive).
pub fn reactions_enabled() -> bool {
    let v = env::var("TELEGRAM_REACTIONS")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase();
    !matches!(v.as_str(), "false" | "0" | "no")
}

/// Whether link previews should be disabled (`_disable_link_previews`).
pub fn disable_link_previews(extra: &HashMap<String, String>) -> bool {
    coerce_bool_extra(extra, "disable_link_previews", false)
}

/// Parse `int(os.getenv(name, default))` with the Python fallback-to-default
/// behaviour on parse failure.
pub fn env_int(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(v) => v.trim().parse::<i64>().unwrap_or(default),
        Err(_) => default,
    }
}

/// Parse `float(os.getenv(name, default))` with fallback-to-default on failure.
pub fn env_float(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(v) => v.trim().parse::<f64>().unwrap_or(default),
        Err(_) => default,
    }
}

/// Album/photo-burst debounce window (`HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS`).
pub fn media_batch_delay_seconds() -> f64 {
    env_float("HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS", 0.8)
}

/// Short text-batch debounce window (`HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS`).
pub fn text_batch_delay_seconds() -> f64 {
    env_float("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", 0.6)
}

/// Longer text-batch debounce window used when a chunk is near the split
/// threshold (`HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS`).
pub fn text_batch_split_delay_seconds() -> f64 {
    env_float("HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
}

/// Choose the text-batch flush delay based on the last chunk length
/// (`_flush_text_batch`): the longer split delay when at/over the threshold.
pub fn text_batch_flush_delay(last_chunk_len: usize) -> f64 {
    if last_chunk_len >= SPLIT_THRESHOLD {
        text_batch_split_delay_seconds()
    } else {
        text_batch_delay_seconds()
    }
}

/// Whether the fallback-IP transport is disabled
/// (`HERMES_TELEGRAM_DISABLE_FALLBACK_IPS`).
pub fn fallback_ips_disabled() -> bool {
    let v = env::var("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    matches!(v.as_str(), "1" | "true" | "yes" | "on")
}

/// HTTPX request-kwargs defaults read from env, mirroring `request_kwargs` in
/// `connect()`.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpRequestConfig {
    pub connection_pool_size: i64,
    pub pool_timeout: f64,
    pub connect_timeout: f64,
    pub read_timeout: f64,
    pub write_timeout: f64,
}

impl Default for HttpRequestConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HttpRequestConfig {
    /// Read the HTTP pool/timeout config from env with the Python defaults.
    pub fn from_env() -> Self {
        Self {
            connection_pool_size: env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
            pool_timeout: env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
            connect_timeout: env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
            read_timeout: env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
            write_timeout: env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
        }
    }
}

/// Webhook configuration derived from env, mirroring the webhook-mode block of
/// `connect()`.
#[derive(Debug, Clone, PartialEq)]
pub struct WebhookConfig {
    pub url: String,
    pub port: u16,
    pub secret: String,
    pub path: String,
}

/// Read `TELEGRAM_WEBHOOK_URL`; returns `None` if unset/blank (polling mode).
pub fn webhook_url() -> Option<String> {
    let v = env::var("TELEGRAM_WEBHOOK_URL").unwrap_or_default();
    let v = v.trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// Build the webhook config from env, mirroring `connect()`'s webhook branch.
///
/// Returns `Err` when `TELEGRAM_WEBHOOK_SECRET` is missing — the Python raises
/// a `RuntimeError` in that case (fail-closed, GHSA-3vpc-7q5r-276h).
pub fn webhook_config() -> Result<Option<WebhookConfig>, String> {
    let url = match webhook_url() {
        None => return Ok(None),
        Some(u) => u,
    };
    let port = env::var("TELEGRAM_WEBHOOK_PORT")
        .ok()
        .and_then(|p| p.trim().parse::<u16>().ok())
        .unwrap_or(8443);
    let secret = env::var("TELEGRAM_WEBHOOK_SECRET")
        .unwrap_or_default()
        .trim()
        .to_string();
    if secret.is_empty() {
        return Err(
            "TELEGRAM_WEBHOOK_SECRET is required when TELEGRAM_WEBHOOK_URL is set. Without it, \
             the webhook endpoint accepts forged updates from anyone who can reach it."
                .to_string(),
        );
    }
    let path = webhook_path_from_url(&url);
    Ok(Some(WebhookConfig {
        url,
        port,
        secret,
        path,
    }))
}

/// Extract the webhook URL path, defaulting to `/telegram` when empty
/// (mirrors `urlparse(webhook_url).path or "/telegram"`).
pub fn webhook_path_from_url(webhook_url: &str) -> String {
    // Mirror Python's `urlparse(webhook_url).path or "/telegram"`. urlparse
    // returns "" for a bare host (e.g. "https://app.fly.dev"), and the literal
    // path otherwise (including a lone "/"). The `url` crate normalises a bare
    // host to "/", so detect the "no path" case from the raw string rather than
    // from the parsed path.
    match url::Url::parse(webhook_url) {
        Ok(u) => {
            // Determine whether the source URL had an explicit path component.
            // Strip scheme, then authority; if nothing remains before a query
            // or fragment, urlparse would have produced "".
            let after_scheme = webhook_url.splitn(2, "://").nth(1).unwrap_or(webhook_url);
            // authority ends at the first '/', '?', or '#'.
            let auth_end = after_scheme
                .find(|c| c == '/' || c == '?' || c == '#')
                .unwrap_or(after_scheme.len());
            let has_explicit_path = after_scheme[auth_end..].starts_with('/');
            if has_explicit_path {
                u.path().to_string()
            } else {
                "/telegram".to_string()
            }
        }
        Err(_) => "/telegram".to_string(),
    }
}

/// Parse the validated fallback IPs from config (`_fallback_ips`). `configured`
/// may be a comma-separated string or a list rendered as comma-joined values.
pub fn fallback_ips_from_config(configured: Option<&str>) -> Vec<String> {
    crate::gw_telegram_network::parse_fallback_ip_env(configured)
}

// ─── Mention / chunk helpers ────────────────────────────────────────────────

/// Return true if the chat type string denotes a group/supergroup
/// (`_is_group_chat`). The Python splits on `.` and lowercases, so values like
/// `ChatType.SUPERGROUP` and `supergroup` both match.
pub fn is_group_chat_type(chat_type: &str) -> bool {
    let normalized = chat_type
        .rsplit('.')
        .next()
        .unwrap_or(chat_type)
        .to_lowercase();
    matches!(normalized.as_str(), "group" | "supergroup")
}

/// Strip a leading `@botname` trigger (with optional trailing punctuation) from
/// a message text (`_clean_bot_trigger_text`). Returns the cleaned text, or the
/// original when cleaning would empty it (or there is no bot username).
pub fn clean_bot_trigger_text(text: Option<&str>, bot_username: Option<&str>) -> Option<String> {
    let text = text?;
    let username = match bot_username {
        Some(u) if !u.is_empty() => u,
        _ => return Some(text.to_string()),
    };
    let pattern = format!(r"(?i)@{}\b[,:\-]*\s*", regex::escape(username));
    let re = Regex::new(&pattern).unwrap();
    let cleaned = re.replace_all(text, "").trim().to_string();
    if cleaned.is_empty() {
        Some(text.to_string())
    } else {
        Some(cleaned)
    }
}

/// Whether a Telegram `mention` entity span equals the expected `@botname`
/// (`_message_mentions_bot`, `mention` branch). `span` is the slice of message
/// text covered by the entity; `expected` is `@<lowercased-username>`.
pub fn mention_matches(span: &str, expected: &str) -> bool {
    !expected.is_empty() && span.trim().to_lowercase() == expected
}

/// Whether a `bot_command` entity span addresses this bot via the
/// `/cmd@botname` disambiguation form (`_message_mentions_bot`, `bot_command`
/// branch). Returns true when the `@...` suffix equals `expected`.
pub fn bot_command_addresses_bot(span: &str, expected: &str) -> bool {
    if expected.is_empty() {
        return false;
    }
    match span.find('@') {
        Some(at) => span[at..].trim().to_lowercase() == expected,
        None => false,
    }
}

// ─── Inline-button callback-data parsing ────────────────────────────────────

/// Decoded form of an inline-keyboard callback payload (`query.data`), mirroring
/// the dispatch ladder in `_handle_callback_query`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackKind {
    /// Model-picker callbacks: `mp:`, `mm:`, `mb`, `mx`, `mg:` and the
    /// `mx:noop` page-counter button.
    ModelPicker,
    /// Exec-approval callback `ea:<choice>:<id>` (choice ∈ once/session/always/deny).
    ExecApproval { choice: String, approval_id: i64 },
    /// Slash-confirm callback `sc:<choice>:<confirm_id>` (choice ∈ once/always/cancel).
    SlashConfirm { choice: String, confirm_id: String },
    /// Update-prompt callback `update_prompt:<y|n>`.
    UpdatePrompt { answer: String },
    /// Anything else (ignored by the handler).
    Unknown,
}

/// Parse a raw `query.data` callback payload into a [`CallbackKind`]
/// (`_handle_callback_query` dispatch). Returns [`CallbackKind::Unknown`] for
/// malformed exec-approval payloads where the id is non-numeric (the Python
/// answers "Invalid approval data." then returns; callers should surface that).
pub fn parse_callback_kind(data: &str) -> CallbackKind {
    // Model-picker prefixes — matches Python's startswith tuple
    // ("mp:", "mm:", "mb", "mx", "mg:").
    if data.starts_with("mp:")
        || data.starts_with("mm:")
        || data.starts_with("mb")
        || data.starts_with("mx")
        || data.starts_with("mg:")
    {
        return CallbackKind::ModelPicker;
    }

    if let Some(rest) = data.strip_prefix("ea:") {
        // Python: data.split(":", 2) — already consumed "ea:", so split rest
        // into [choice, id] (maxsplit semantics keep the id intact).
        let parts: Vec<&str> = rest.splitn(2, ':').collect();
        if parts.len() == 2 {
            if let Ok(approval_id) = parts[1].parse::<i64>() {
                return CallbackKind::ExecApproval {
                    choice: parts[0].to_string(),
                    approval_id,
                };
            }
            // Non-numeric id → "Invalid approval data."
            return CallbackKind::Unknown;
        }
        return CallbackKind::Unknown;
    }

    if let Some(rest) = data.strip_prefix("sc:") {
        let parts: Vec<&str> = rest.splitn(2, ':').collect();
        if parts.len() == 2 {
            return CallbackKind::SlashConfirm {
                choice: parts[0].to_string(),
                confirm_id: parts[1].to_string(),
            };
        }
        return CallbackKind::Unknown;
    }

    if let Some(rest) = data.strip_prefix("update_prompt:") {
        return CallbackKind::UpdatePrompt {
            answer: rest.to_string(),
        };
    }

    CallbackKind::Unknown
}

/// Human-readable label for an exec-approval choice (`label_map` in the `ea:`
/// branch). Unknown choices map to `"Resolved"`.
pub fn approval_label(choice: &str) -> &'static str {
    match choice {
        "once" => "✅ Approved once",
        "session" => "✅ Approved for session",
        "always" => "✅ Approved permanently",
        "deny" => "❌ Denied",
        _ => "Resolved",
    }
}

/// Human-readable label for a slash-confirm choice (`label_map` in the `sc:`
/// branch). Unknown choices map to `"Resolved"`.
pub fn slash_confirm_label(choice: &str) -> &'static str {
    match choice {
        "once" => "✅ Approved once",
        "always" => "🔒 Always approve",
        "cancel" => "❌ Cancelled",
        _ => "Resolved",
    }
}

/// Label for an update-prompt answer (`label = "Yes" if answer == "y" else
/// "No"`).
pub fn update_prompt_label(answer: &str) -> &'static str {
    if answer == "y" {
        "Yes"
    } else {
        "No"
    }
}

// ─── Model-picker pagination ─────────────────────────────────────────────────

/// One inline keyboard button: visible label plus its `callback_data`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineButton {
    pub text: String,
    pub callback_data: String,
}

impl InlineButton {
    pub fn new(text: impl Into<String>, callback_data: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            callback_data: callback_data.into(),
        }
    }
}

/// Result of [`build_model_keyboard`]: the row layout plus the page-info suffix
/// appended to the picker prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKeyboard {
    pub rows: Vec<Vec<InlineButton>>,
    pub page_info: String,
}

/// Shorten a model id for a button label (`mm:` label logic in
/// `_build_model_keyboard`): take the last `/`-segment, truncate to 35 chars +
/// "..." when over 38 chars.
pub fn model_short_label(model_id: &str) -> String {
    let short = if model_id.contains('/') {
        model_id.rsplit('/').next().unwrap_or(model_id)
    } else {
        model_id
    };
    if short.chars().count() > 38 {
        let truncated: String = short.chars().take(35).collect();
        format!("{truncated}...")
    } else {
        short.to_string()
    }
}

/// Build paginated model buttons (`_build_model_keyboard`). Returns the keyboard
/// rows (2 model buttons per row, plus optional pagination and a Back/Cancel
/// row) and the page-info suffix.
pub fn build_model_keyboard(models: &[String], page: usize) -> ModelKeyboard {
    let page_size = MODEL_PAGE_SIZE;
    let total = models.len();
    let total_pages = std::cmp::max(1, total.div_ceil(page_size));
    // page = max(0, min(page, total_pages - 1)) — page is usize so >= 0.
    let page = std::cmp::min(page, total_pages - 1);

    let start = page * page_size;
    let end = std::cmp::min(start + page_size, total);
    let page_models = &models[start..end];

    let mut buttons: Vec<InlineButton> = Vec::new();
    for (i, model_id) in page_models.iter().enumerate() {
        let abs_idx = start + i;
        let short = model_short_label(model_id);
        buttons.push(InlineButton::new(short, format!("mm:{abs_idx}")));
    }

    let mut rows: Vec<Vec<InlineButton>> = buttons
        .chunks(2)
        .map(|c| c.to_vec())
        .collect();

    if total_pages > 1 {
        let mut nav: Vec<InlineButton> = Vec::new();
        if page > 0 {
            nav.push(InlineButton::new("◀ Prev", format!("mg:{}", page - 1)));
        }
        nav.push(InlineButton::new(
            format!("{}/{}", page + 1, total_pages),
            "mx:noop",
        ));
        if page < total_pages - 1 {
            nav.push(InlineButton::new("Next ▶", format!("mg:{}", page + 1)));
        }
        rows.push(nav);
    }

    rows.push(vec![
        InlineButton::new("◀ Back", "mb"),
        InlineButton::new("✗ Cancel", "mx"),
    ]);

    let page_info = if total_pages > 1 {
        format!(" ({}–{} of {})", start + 1, end, total)
    } else {
        String::new()
    };

    ModelKeyboard { rows, page_info }
}

// ─── Media helpers ───────────────────────────────────────────────────────────

/// Build an actionable file-not-found error for gateway MEDIA delivery
/// (`_missing_media_path_error`). Sandbox-only paths get an extra hint.
pub fn missing_media_path_error(label: &str, path: &str) -> String {
    let mut error = format!("{label} file not found: {path}");
    if path.starts_with("/workspace/")
        || path.starts_with("/output/")
        || path.starts_with("/outputs/")
    {
        error.push_str(
            " (path may only exist inside the Docker sandbox. Bind-mount a host directory and \
             emit the host-visible path in MEDIA: for gateway file delivery.)",
        );
    }
    error
}

/// Truncate a caption to Telegram's 1024-char limit (caption slicing in the
/// various `send_*` methods). Operates on chars to match Python slicing.
pub fn caption_truncate(caption: Option<&str>) -> Option<String> {
    caption.map(|c| {
        if c.chars().count() > 1024 {
            c.chars().take(1024).collect()
        } else {
            c.to_string()
        }
    })
}

/// Classify which native voice/audio path a file extension takes in
/// `send_voice`: `.ogg`/`.opus` → voice; `.mp3`/`.m4a` → audio; otherwise the
/// document fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceSendKind {
    Voice,
    Audio,
    Document,
}

/// Determine the [`VoiceSendKind`] for a file path's extension (`send_voice`).
pub fn voice_send_kind(path: &str) -> VoiceSendKind {
    let ext = path
        .rsplit('.')
        .next()
        .map(|e| format!(".{}", e.to_lowercase()))
        .unwrap_or_default();
    match ext.as_str() {
        ".ogg" | ".opus" => VoiceSendKind::Voice,
        ".mp3" | ".m4a" => VoiceSendKind::Audio,
        _ => VoiceSendKind::Document,
    }
}

/// Escape the ` (n/m)` chunk-count suffix that `truncate_message` appends, so
/// the MarkdownV2-special parentheses don't break the send (the `re.sub` over
/// chunks in `send`). Returns the chunk with the suffix backslash-escaped.
pub fn chunk_size_suffix_escape(chunk: &str) -> String {
    let re = Regex::new(r" \((\d+)/(\d+)\)$").unwrap();
    re.replace(chunk, " \\($1/$2\\)").into_owned()
}

/// Whether a `send_photo` error string indicates Telegram rejected the image
/// for invalid dimensions (`send_image_file` `is_dim_error`).
pub fn is_photo_dimension_error(error: &str) -> bool {
    error.contains("Photo_invalid_dimensions") || error.contains("PHOTO_INVALID_DIMENSIONS")
}

/// Whether an edit error means the content was unchanged (`"not modified"`).
pub fn is_not_modified_error(error: &str) -> bool {
    error.to_lowercase().contains("not modified")
}

/// Whether an edit error means the message exceeded the length limit
/// (`message_too_long` / `too long`).
pub fn is_message_too_long_error(error: &str) -> bool {
    let lower = error.to_lowercase();
    lower.contains("message_too_long") || lower.contains("too long")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_mdv2() {
        assert_eq!(escape_mdv2("a.b"), "a\\.b");
        assert_eq!(escape_mdv2("(x)"), "\\(x\\)");
        assert_eq!(escape_mdv2("plain"), "plain");
        assert_eq!(escape_mdv2("a_b*c"), "a\\_b\\*c");
    }

    #[test]
    fn test_strip_mdv2_preserves_snake_case() {
        // _italic_ unwrapped, but my_var_name preserved.
        assert_eq!(strip_mdv2("_italic_"), "italic");
        assert_eq!(strip_mdv2("my_var_name"), "my_var_name");
        // Bold markers removed.
        assert_eq!(strip_mdv2("*bold*"), "bold");
        // Escape backslashes removed.
        assert_eq!(strip_mdv2("a\\.b"), "a.b");
        // Strikethrough + spoiler.
        assert_eq!(strip_mdv2("~struck~"), "struck");
        assert_eq!(strip_mdv2("||secret||"), "secret");
    }

    #[test]
    fn test_table_detection() {
        assert!(is_table_row("| a | b |"));
        assert!(!is_table_row("   "));
        assert!(!is_table_row("no pipes here"));
        let sep = table_separator_re();
        assert!(sep.is_match("| --- | --- |"));
        assert!(sep.is_match("|:--|--:|"));
        // Lone horizontal rule must NOT match (needs internal pipe).
        assert!(!sep.is_match("---"));
    }

    #[test]
    fn test_split_markdown_table_row() {
        assert_eq!(
            split_markdown_table_row("| a | b | c |"),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            split_markdown_table_row("x|y"),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_wrap_markdown_tables() {
        let input = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |";
        let out = wrap_markdown_tables(input);
        assert!(out.contains("**Alice**"));
        assert!(out.contains("• Name: Alice"));
        assert!(out.contains("• Age: 30"));
        assert!(out.contains("**Bob**"));
    }

    #[test]
    fn test_wrap_markdown_tables_no_table() {
        let input = "just some text without tables";
        assert_eq!(wrap_markdown_tables(input), input);
    }

    #[test]
    fn test_format_message_bold_italic() {
        // **bold** -> *bold*, *italic* -> _italic_.
        let out = format_message("**bold** and *italic*");
        assert!(out.contains("*bold*"));
        assert!(out.contains("_italic_"));
    }

    #[test]
    fn test_format_message_header() {
        let out = format_message("## Title");
        assert_eq!(out, "*Title*");
    }

    #[test]
    fn test_format_message_inline_code_protected() {
        let out = format_message("use `a.b()` here");
        // Inside inline code, the `.` is NOT escaped.
        assert!(out.contains("`a.b()`"));
        // Outside text "here" plain.
        assert!(out.contains("here"));
    }

    #[test]
    fn test_format_message_link() {
        let out = format_message("[Hi](https://x.com/a)");
        assert!(out.contains("[Hi](https://x.com/a)"));
    }

    #[test]
    fn test_format_message_empty() {
        assert_eq!(format_message(""), "");
    }

    #[test]
    fn test_format_message_escapes_plain_dot() {
        let out = format_message("hello world.");
        assert!(out.contains("world\\."));
    }

    #[test]
    fn test_thread_id_mapping() {
        assert_eq!(message_thread_id_for_send(None), None);
        assert_eq!(message_thread_id_for_send(Some("1")), None); // General topic
        assert_eq!(message_thread_id_for_send(Some("42")), Some(42));
        assert_eq!(message_thread_id_for_send(Some("")), None);
        assert_eq!(message_thread_id_for_typing(Some("1")), None);
        assert_eq!(message_thread_id_for_typing(Some("7")), Some(7));
    }

    #[test]
    fn test_metadata_thread_id() {
        let mut m = HashMap::new();
        assert_eq!(metadata_thread_id(&m), None);
        m.insert("message_thread_id".to_string(), "9".to_string());
        assert_eq!(metadata_thread_id(&m), Some("9".to_string()));
        m.insert("thread_id".to_string(), "5".to_string());
        // thread_id takes precedence.
        assert_eq!(metadata_thread_id(&m), Some("5".to_string()));
    }

    #[test]
    fn test_error_classification() {
        assert!(is_thread_not_found_error("Bad Request: thread not found"));
        assert!(!is_thread_not_found_error("some other error"));
        assert!(looks_like_polling_conflict("Conflict", ""));
        assert!(looks_like_polling_conflict(
            "SomeError",
            "terminated by other getUpdates request"
        ));
        assert!(!looks_like_polling_conflict("ValueError", "boom"));
        assert!(looks_like_network_error("NetworkError", false));
        assert!(looks_like_network_error("TimedOut", false));
        assert!(looks_like_network_error("Whatever", true));
        assert!(!looks_like_network_error("ValueError", false));
    }

    #[test]
    fn test_coerce_bool_value() {
        assert!(coerce_bool_value(Some("yes"), false));
        assert!(!coerce_bool_value(Some("off"), true));
        assert!(coerce_bool_value(None, true));
        assert!(!coerce_bool_value(None, false));
        // Unrecognised string falls back to default.
        assert!(coerce_bool_value(Some("maybe"), true));
        assert!(!coerce_bool_value(Some("maybe"), false));
    }

    #[test]
    fn test_reply_to_mode_and_threading() {
        assert_eq!(reply_to_mode(None), "first");
        assert_eq!(reply_to_mode(Some("")), "first");
        assert_eq!(reply_to_mode(Some("all")), "all");

        assert!(!should_thread_reply(None, "all", 0));
        assert!(!should_thread_reply(Some("5"), "off", 0));
        assert!(should_thread_reply(Some("5"), "all", 3));
        assert!(should_thread_reply(Some("5"), "first", 0));
        assert!(!should_thread_reply(Some("5"), "first", 1));
    }

    #[test]
    fn test_free_response_and_ignored_threads() {
        let chats = telegram_free_response_chats(Some(" 1, 2 ,3 "));
        assert!(chats.contains("1"));
        assert!(chats.contains("2"));
        assert!(chats.contains("3"));
        assert_eq!(chats.len(), 3);

        let threads = telegram_ignored_threads(Some("10, 20, bad, 30"));
        assert!(threads.contains(&10));
        assert!(threads.contains(&20));
        assert!(threads.contains(&30));
        assert_eq!(threads.len(), 3);
    }

    #[test]
    fn test_is_group_chat_type() {
        assert!(is_group_chat_type("group"));
        assert!(is_group_chat_type("supergroup"));
        assert!(is_group_chat_type("ChatType.SUPERGROUP"));
        assert!(!is_group_chat_type("private"));
        assert!(!is_group_chat_type("channel"));
    }

    #[test]
    fn test_clean_bot_trigger_text() {
        assert_eq!(
            clean_bot_trigger_text(Some("@hermes_bot hello"), Some("hermes_bot")),
            Some("hello".to_string())
        );
        assert_eq!(
            clean_bot_trigger_text(Some("@hermes_bot: hi there"), Some("hermes_bot")),
            Some("hi there".to_string())
        );
        // No username -> unchanged.
        assert_eq!(
            clean_bot_trigger_text(Some("@x hello"), None),
            Some("@x hello".to_string())
        );
        // Cleaning to empty -> return original.
        assert_eq!(
            clean_bot_trigger_text(Some("@hermes_bot"), Some("hermes_bot")),
            Some("@hermes_bot".to_string())
        );
    }

    #[test]
    fn test_mention_matches() {
        assert!(mention_matches("@HermesBot", "@hermesbot"));
        assert!(mention_matches("  @hermesbot  ", "@hermesbot"));
        assert!(!mention_matches("@other", "@hermesbot"));
        assert!(!mention_matches("@hermesbot", ""));
    }

    #[test]
    fn test_bot_command_addresses_bot() {
        assert!(bot_command_addresses_bot("/new@hermesbot", "@hermesbot"));
        assert!(!bot_command_addresses_bot("/new@other", "@hermesbot"));
        assert!(!bot_command_addresses_bot("/new", "@hermesbot"));
        assert!(!bot_command_addresses_bot("/new@hermesbot", ""));
    }

    #[test]
    fn test_parse_callback_kind() {
        assert_eq!(parse_callback_kind("mp:openai"), CallbackKind::ModelPicker);
        assert_eq!(parse_callback_kind("mm:3"), CallbackKind::ModelPicker);
        assert_eq!(parse_callback_kind("mb"), CallbackKind::ModelPicker);
        assert_eq!(parse_callback_kind("mx"), CallbackKind::ModelPicker);
        assert_eq!(parse_callback_kind("mx:noop"), CallbackKind::ModelPicker);
        assert_eq!(parse_callback_kind("mg:2"), CallbackKind::ModelPicker);

        assert_eq!(
            parse_callback_kind("ea:once:42"),
            CallbackKind::ExecApproval {
                choice: "once".to_string(),
                approval_id: 42
            }
        );
        // Non-numeric id -> Unknown.
        assert_eq!(parse_callback_kind("ea:once:abc"), CallbackKind::Unknown);

        assert_eq!(
            parse_callback_kind("sc:always:conf-9"),
            CallbackKind::SlashConfirm {
                choice: "always".to_string(),
                confirm_id: "conf-9".to_string()
            }
        );

        assert_eq!(
            parse_callback_kind("update_prompt:y"),
            CallbackKind::UpdatePrompt {
                answer: "y".to_string()
            }
        );

        assert_eq!(parse_callback_kind("garbage"), CallbackKind::Unknown);
    }

    #[test]
    fn test_callback_labels() {
        assert_eq!(approval_label("once"), "✅ Approved once");
        assert_eq!(approval_label("deny"), "❌ Denied");
        assert_eq!(approval_label("weird"), "Resolved");
        assert_eq!(slash_confirm_label("always"), "🔒 Always approve");
        assert_eq!(slash_confirm_label("cancel"), "❌ Cancelled");
        assert_eq!(update_prompt_label("y"), "Yes");
        assert_eq!(update_prompt_label("n"), "No");
    }

    #[test]
    fn test_model_short_label() {
        assert_eq!(model_short_label("anthropic/claude-3"), "claude-3");
        assert_eq!(model_short_label("gpt-4"), "gpt-4");
        let long = "x".repeat(50);
        let short = model_short_label(&long);
        assert_eq!(short.chars().count(), 38); // 35 + "..."
        assert!(short.ends_with("..."));
    }

    #[test]
    fn test_build_model_keyboard_single_page() {
        let models: Vec<String> = (0..5).map(|i| format!("m{i}")).collect();
        let kb = build_model_keyboard(&models, 0);
        // 5 buttons -> 3 rows of (2,2,1), no pagination, plus back/cancel row.
        assert_eq!(kb.page_info, "");
        // Last row is Back/Cancel.
        let last = kb.rows.last().unwrap();
        assert_eq!(last[0].callback_data, "mb");
        assert_eq!(last[1].callback_data, "mx");
        // First model button.
        assert_eq!(kb.rows[0][0].callback_data, "mm:0");
    }

    #[test]
    fn test_build_model_keyboard_pagination() {
        let models: Vec<String> = (0..20).map(|i| format!("m{i}")).collect();
        let kb = build_model_keyboard(&models, 1);
        // page 1 -> models 8..16, page_info " (9–16 of 20)".
        assert_eq!(kb.page_info, " (9–16 of 20)");
        // Has a pagination nav row with prev/counter/next.
        let has_prev = kb
            .rows
            .iter()
            .any(|r| r.iter().any(|b| b.callback_data == "mg:0"));
        let has_next = kb
            .rows
            .iter()
            .any(|r| r.iter().any(|b| b.callback_data == "mg:2"));
        assert!(has_prev);
        assert!(has_next);
        // Model index uses absolute idx (page 1 -> first button mm:8).
        assert_eq!(kb.rows[0][0].callback_data, "mm:8");
    }

    #[test]
    fn test_missing_media_path_error() {
        let e = missing_media_path_error("Image", "/tmp/x.png");
        assert_eq!(e, "Image file not found: /tmp/x.png");
        let e2 = missing_media_path_error("Video", "/workspace/x.mp4");
        assert!(e2.contains("Docker sandbox"));
    }

    #[test]
    fn test_caption_truncate() {
        assert_eq!(caption_truncate(None), None);
        assert_eq!(caption_truncate(Some("hi")), Some("hi".to_string()));
        let long = "a".repeat(2000);
        let t = caption_truncate(Some(&long)).unwrap();
        assert_eq!(t.chars().count(), 1024);
    }

    #[test]
    fn test_voice_send_kind() {
        assert_eq!(voice_send_kind("a.ogg"), VoiceSendKind::Voice);
        assert_eq!(voice_send_kind("a.OPUS"), VoiceSendKind::Voice);
        assert_eq!(voice_send_kind("a.mp3"), VoiceSendKind::Audio);
        assert_eq!(voice_send_kind("a.m4a"), VoiceSendKind::Audio);
        assert_eq!(voice_send_kind("a.wav"), VoiceSendKind::Document);
        assert_eq!(voice_send_kind("noext"), VoiceSendKind::Document);
    }

    #[test]
    fn test_chunk_size_suffix_escape() {
        assert_eq!(
            chunk_size_suffix_escape("hello (1/2)"),
            "hello \\(1/2\\)"
        );
        // Only escapes a trailing (n/m) suffix.
        assert_eq!(chunk_size_suffix_escape("hello (1/2) world"), "hello (1/2) world");
    }

    #[test]
    fn test_edit_error_classification() {
        assert!(is_photo_dimension_error("Bad: PHOTO_INVALID_DIMENSIONS"));
        assert!(!is_photo_dimension_error("rate limited"));
        assert!(is_not_modified_error("Message is not modified"));
        assert!(is_message_too_long_error("MESSAGE_TOO_LONG"));
        assert!(is_message_too_long_error("message is too long"));
    }

    #[test]
    fn test_text_batch_flush_delay() {
        // Below threshold -> short delay (default 0.6).
        assert_eq!(text_batch_flush_delay(10), text_batch_delay_seconds());
        // At/over threshold -> split delay (default 2.0).
        assert_eq!(
            text_batch_flush_delay(SPLIT_THRESHOLD),
            text_batch_split_delay_seconds()
        );
    }

    #[test]
    fn test_webhook_path_from_url() {
        assert_eq!(webhook_path_from_url("https://app.fly.dev/telegram"), "/telegram");
        assert_eq!(webhook_path_from_url("https://app.fly.dev"), "/telegram");
        assert_eq!(webhook_path_from_url("https://app.fly.dev/custom"), "/custom");
    }

    #[test]
    fn test_webhook_config_requires_secret() {
        // Wrap env mutation in unsafe per edition 2024.
        unsafe {
            env::set_var("TELEGRAM_WEBHOOK_URL", "https://app.fly.dev/telegram");
            env::remove_var("TELEGRAM_WEBHOOK_SECRET");
        }
        let res = webhook_config();
        assert!(res.is_err(), "missing secret must fail closed");

        unsafe {
            env::set_var("TELEGRAM_WEBHOOK_SECRET", "deadbeef");
        }
        let res = webhook_config().unwrap().unwrap();
        assert_eq!(res.path, "/telegram");
        assert_eq!(res.secret, "deadbeef");
        assert_eq!(res.port, 8443);

        unsafe {
            env::remove_var("TELEGRAM_WEBHOOK_URL");
            env::remove_var("TELEGRAM_WEBHOOK_SECRET");
        }
    }

    #[test]
    fn test_http_request_config_defaults() {
        // With nothing set, defaults apply.
        unsafe {
            env::remove_var("HERMES_TELEGRAM_HTTP_POOL_SIZE");
            env::remove_var("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT");
        }
        let cfg = HttpRequestConfig::from_env();
        assert_eq!(cfg.connection_pool_size, 512);
        assert_eq!(cfg.pool_timeout, 8.0);
        assert_eq!(cfg.connect_timeout, 10.0);
        assert_eq!(cfg.read_timeout, 20.0);
        assert_eq!(cfg.write_timeout, 20.0);
    }

    #[test]
    fn test_fallback_ips_from_config() {
        // Valid public IP passes, garbage filtered.
        let ips = fallback_ips_from_config(Some("149.154.167.220, not-an-ip"));
        assert!(ips.contains(&"149.154.167.220".to_string()));
    }
}
