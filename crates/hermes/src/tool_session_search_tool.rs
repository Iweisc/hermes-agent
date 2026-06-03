//! Session Search Tool - Long-Term Conversation Recall
//!
//! Ported from `tools/session_search_tool.py`.
//!
//! Searches past session transcripts in SQLite via FTS5, then summarizes the
//! top matching sessions using the configured auxiliary `session_search` model
//! (same pattern as `web_extract`). By default, auxiliary "auto" routing uses
//! the main chat provider/model unless the user overrides
//! `auxiliary.session_search`. Returns focused summaries of past conversations
//! rather than raw transcripts, keeping the main model's context window clean.
//!
//! Flow:
//!   1. FTS5 search finds matching messages ranked by relevance
//!   2. Groups by session, takes the top N unique sessions (default 3)
//!   3. Loads each session's conversation, truncates to ~100k chars centered on
//!      matches
//!   4. Sends to the configured auxiliary model with a focused summarization
//!      prompt
//!   5. Returns per-session summaries with metadata
//!
//! The Python module relied on an async auxiliary LLM client
//! (`agent.auxiliary_client.async_call_llm`) and a global tool registry. The
//! Rust port keeps the pure data-shaping logic faithful and abstracts the LLM
//! call behind the [`Summarizer`] trait so callers can wire in
//! `crate::ag_auxiliary_client` (or any other backend) without this module
//! depending on a not-yet-ported async runtime surface.

use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Map, Value};

/// Maximum characters of conversation transcript passed to the summarizer.
pub const MAX_SESSION_CHARS: usize = 100_000;
/// Maximum tokens requested from the summarizer model.
pub const MAX_SUMMARY_TOKENS: i64 = 10_000;

/// Sources excluded from session browsing/searching by default.
///
/// Third-party integrations (Paperclip agents, etc.) tag their sessions with
/// `HERMES_SESSION_SOURCE=tool` so they don't clutter the user's session
/// history.
pub const HIDDEN_SESSION_SOURCES: &[&str] = &["tool"];

/// Standard error payload, matching the registry's `tool_error` shape.
///
/// Mirrors `tools.registry.tool_error(..., success=False)`: a JSON object with
/// an `error` message and `success: false`.
pub fn tool_error(message: &str) -> String {
    json!({ "error": message, "success": false }).to_string()
}

// ---------------------------------------------------------------------------
// Database abstraction
// ---------------------------------------------------------------------------

/// Minimal database surface the session-search tool needs.
///
/// In production this is implemented for `crate::mod_hermes_state::SessionDB`
/// (see the blanket impl below). Tests provide an in-memory fake.
pub trait SessionStore {
    /// FTS5 search over messages; rows include at least `session_id` and may
    /// carry `source`, `model`, `session_started` metadata from the matching
    /// (possibly child) session.
    fn search_messages(
        &self,
        query: &str,
        role_filter: Option<&[String]>,
        exclude_sources: &[String],
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Map<String, Value>>, String>;

    /// Recent sessions with rich metadata (titles, previews, timestamps).
    fn list_sessions_rich(
        &self,
        limit: i64,
        exclude_sources: &[String],
        order_by_last_active: bool,
    ) -> Result<Vec<Map<String, Value>>, String>;

    /// Fetch a single session's metadata. `None` if it does not exist.
    fn get_session(&self, session_id: &str) -> Result<Option<Map<String, Value>>, String>;

    /// Load a session's conversation as a list of message dicts.
    fn get_messages_as_conversation(
        &self,
        session_id: &str,
    ) -> Result<Vec<Map<String, Value>>, String>;
}

// Concrete adapter for the ported state DB
// (`hermes_core::mod_hermes_state::SessionDB`). Gated behind the `state-db`
// feature so this module compiles standalone during the parallel port run even
// before the state module is wired into the crate graph; the integrator enables
// the feature once `mod_hermes_state` is available. The signatures here mirror
// `crate::mod_hermes_state::SessionDB::{search_messages, list_sessions_rich,
// get_session, get_messages_as_conversation}`.
#[cfg(feature = "state-db")]
impl SessionStore for hermes_core::mod_hermes_state::SessionDB {
    fn search_messages(
        &self,
        query: &str,
        role_filter: Option<&[String]>,
        exclude_sources: &[String],
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Map<String, Value>>, String> {
        hermes_core::mod_hermes_state::SessionDB::search_messages(
            self,
            query,
            None,
            Some(exclude_sources),
            role_filter,
            limit,
            offset,
        )
        .map_err(|e| e.to_string())
    }

    fn list_sessions_rich(
        &self,
        limit: i64,
        exclude_sources: &[String],
        order_by_last_active: bool,
    ) -> Result<Vec<Map<String, Value>>, String> {
        hermes_core::mod_hermes_state::SessionDB::list_sessions_rich(
            self,
            None,
            Some(exclude_sources),
            limit,
            0,
            false,
            false,
            order_by_last_active,
        )
        .map_err(|e| e.to_string())
    }

    fn get_session(&self, session_id: &str) -> Result<Option<Map<String, Value>>, String> {
        hermes_core::mod_hermes_state::SessionDB::get_session(self, session_id)
            .map_err(|e| e.to_string())
    }

    fn get_messages_as_conversation(
        &self,
        session_id: &str,
    ) -> Result<Vec<Map<String, Value>>, String> {
        hermes_core::mod_hermes_state::SessionDB::get_messages_as_conversation(
            self, session_id, false,
        )
        .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Summarizer abstraction
// ---------------------------------------------------------------------------

/// Result of asking the auxiliary model to summarize one session.
///
/// Mirrors the Python `_summarize_session` return: `Some(text)` on success,
/// `None` when no auxiliary model is available or the call failed after
/// retries.
pub trait Summarizer {
    /// Summarize a single conversation transcript focused on `query`.
    ///
    /// `system_prompt` and `user_prompt` are precomputed by
    /// [`build_summary_prompts`]. Implementations should request
    /// `temperature=0.1` and `max_tokens=MAX_SUMMARY_TOKENS`, and return the
    /// extracted text (via `extract_content_or_reasoning`) or `None`.
    fn summarize(&self, system_prompt: &str, user_prompt: &str) -> Option<String>;

    /// Maximum number of sessions summarized concurrently. The default mirrors
    /// `auxiliary.session_search.max_concurrency` (default 3, clamped [1, 5]).
    /// The Rust port performs the calls sequentially, so this is advisory.
    fn max_concurrency(&self) -> usize {
        3
    }
}

/// Clamp a raw `max_concurrency` configuration value into the supported range.
///
/// Mirrors `_get_session_search_max_concurrency`: `None`/non-int -> default,
/// otherwise clamp to `[1, 5]`.
pub fn clamp_max_concurrency(raw: Option<&Value>, default: usize) -> usize {
    let value = match raw {
        None | Some(Value::Null) => return default,
        Some(v) => {
            if let Some(i) = v.as_i64() {
                i
            } else if let Some(f) = v.as_f64() {
                f as i64
            } else if let Some(s) = v.as_str() {
                match s.trim().parse::<i64>() {
                    Ok(i) => i,
                    Err(_) => return default,
                }
            } else {
                return default;
            }
        }
    };
    value.clamp(1, 5) as usize
}

// ---------------------------------------------------------------------------
// Timestamp formatting
// ---------------------------------------------------------------------------

/// Convert a Unix timestamp (float/int) or ISO string to a human-readable date.
///
/// Mirrors `_format_timestamp`. Returns "unknown" for null, the formatted local
/// date for numerics / numeric-looking strings, the string itself for
/// non-numeric strings, and the raw stringified value if conversion fails.
pub fn format_timestamp(ts: &Value) -> String {
    match ts {
        Value::Null => "unknown".to_string(),
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                format_unix(f).unwrap_or_else(|| n.to_string())
            } else {
                n.to_string()
            }
        }
        Value::String(s) => {
            if is_numeric_string(s) {
                match s.parse::<f64>() {
                    Ok(f) => format_unix(f).unwrap_or_else(|| s.clone()),
                    Err(_) => s.clone(),
                }
            } else {
                s.clone()
            }
        }
        other => other.to_string(),
    }
}

/// True when a string is numeric in the Python sense used by `_format_timestamp`:
/// after stripping '.' and '-', the remaining characters are all digits (and
/// there is at least one).
fn is_numeric_string(s: &str) -> bool {
    let stripped: String = s.chars().filter(|c| *c != '.' && *c != '-').collect();
    !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit())
}

/// Format a Unix timestamp into "%B %d, %Y at %I:%M %p" in local time, matching
/// Python's `datetime.fromtimestamp(ts).strftime(...)`.
fn format_unix(ts: f64) -> Option<String> {
    use chrono::{Local, TimeZone};
    let secs = ts.trunc() as i64;
    let nanos = ((ts - ts.trunc()) * 1_000_000_000.0).round() as u32;
    match Local.timestamp_opt(secs, nanos) {
        chrono::offset::LocalResult::Single(dt) => Some(dt.format("%B %d, %Y at %I:%M %p").to_string()),
        chrono::offset::LocalResult::Ambiguous(dt, _) => {
            Some(dt.format("%B %d, %Y at %I:%M %p").to_string())
        }
        chrono::offset::LocalResult::None => None,
    }
}

// ---------------------------------------------------------------------------
// Conversation formatting
// ---------------------------------------------------------------------------

fn str_field(msg: &Map<String, Value>, key: &str) -> Option<String> {
    msg.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
}

/// Format session messages into a readable transcript for summarization.
///
/// Mirrors `_format_conversation`.
pub fn format_conversation(messages: &[Map<String, Value>]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_uppercase();
        // content = msg.get("content") or ""  -> falsy (null/empty/missing) => ""
        let mut content = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tool_name = str_field(msg, "tool_name");

        if role == "TOOL" && tool_name.as_deref().map(|n| !n.is_empty()).unwrap_or(false) {
            // Truncate long tool outputs
            if content.chars().count() > 500 {
                let chars: Vec<char> = content.chars().collect();
                let head: String = chars[..250].iter().collect();
                let tail: String = chars[chars.len() - 250..].iter().collect();
                content = format!("{head}\n...[truncated]...\n{tail}");
            }
            parts.push(format!("[TOOL:{}]: {}", tool_name.unwrap(), content));
        } else if role == "ASSISTANT" {
            let tool_calls = msg.get("tool_calls");
            let is_list = matches!(tool_calls, Some(Value::Array(_)));
            if is_list {
                let arr = tool_calls.and_then(|v| v.as_array()).unwrap();
                let mut tc_names: Vec<String> = Vec::new();
                for tc in arr {
                    if let Value::Object(obj) = tc {
                        // name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        let name = obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| {
                                obj.get("function")
                                    .and_then(|f| f.as_object())
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("?")
                                    .to_string()
                            });
                        tc_names.push(name);
                    }
                }
                if !tc_names.is_empty() {
                    parts.push(format!("[ASSISTANT]: [Called: {}]", tc_names.join(", ")));
                }
                if !content.is_empty() {
                    parts.push(format!("[ASSISTANT]: {content}"));
                }
            } else {
                parts.push(format!("[ASSISTANT]: {content}"));
            }
        } else {
            parts.push(format!("[{role}]: {content}"));
        }
    }
    parts.join("\n\n")
}

// ---------------------------------------------------------------------------
// Truncate around matches
// ---------------------------------------------------------------------------

/// Truncate a conversation transcript to `max_chars`, choosing a window that
/// maximises coverage of positions where `query` actually appears.
///
/// Mirrors `_truncate_around_matches`. Operates on Unicode scalar values
/// (chars), matching Python's string indexing semantics.
pub fn truncate_around_matches(full_text: &str, query: &str, max_chars: usize) -> String {
    let chars: Vec<char> = full_text.chars().collect();
    let text_len = chars.len();
    if text_len <= max_chars {
        return full_text.to_string();
    }

    let text_lower: Vec<char> = full_text.to_lowercase().chars().collect();
    // Lowercasing can change length for some scripts; Python lowercases the
    // whole text and indexes into that. Keep both representations consistent by
    // searching/slicing the lowercased form for positions, but slicing the
    // original for output as Python does (it slices full_text). To stay faithful
    // and avoid index drift, fall back to byte-stable behavior: if lengths
    // diverge, operate purely on the lowercased text for position finding and
    // map by min-length clamping. In practice transcripts are predominantly
    // ASCII where len is preserved.
    let lower_len = text_lower.len();
    let query_lower = query.to_lowercase();
    let query_lower = query_lower.trim();

    let lower_str: String = text_lower.iter().collect();

    let mut match_positions: Vec<usize> = Vec::new();

    // --- 1. Full-phrase search --------------------------------------------
    if !query_lower.is_empty() {
        match_positions = find_all_char_positions(&lower_str, query_lower);
    }

    // --- 2. Proximity co-occurrence of all terms (within 200 chars) -------
    if match_positions.is_empty() {
        let terms: Vec<&str> = query_lower.split_whitespace().collect();
        if terms.len() > 1 {
            let mut term_positions: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
            for t in &terms {
                term_positions.insert(*t, find_all_char_positions(&lower_str, t));
            }
            // rarest = min by number of occurrences; Python's min over the
            // list returns the first term achieving the minimum.
            let mut rarest = terms[0];
            let mut rarest_count = term_positions.get(rarest).map(|v| v.len()).unwrap_or(0);
            for t in &terms {
                let c = term_positions.get(*t).map(|v| v.len()).unwrap_or(0);
                if c < rarest_count {
                    rarest_count = c;
                    rarest = *t;
                }
            }
            let empty: Vec<usize> = Vec::new();
            let rarest_positions = term_positions.get(rarest).unwrap_or(&empty).clone();
            for pos in rarest_positions {
                let ok = terms.iter().filter(|t| **t != rarest).all(|t| {
                    term_positions
                        .get(*t)
                        .map(|ps| ps.iter().any(|p| (*p as i64 - pos as i64).abs() < 200))
                        .unwrap_or(false)
                });
                if ok {
                    match_positions.push(pos);
                }
            }
        }
    }

    // --- 3. Individual term positions (last resort) -----------------------
    if match_positions.is_empty() {
        let terms: Vec<&str> = query_lower.split_whitespace().collect();
        for t in &terms {
            for p in find_all_char_positions(&lower_str, t) {
                match_positions.push(p);
            }
        }
    }

    if match_positions.is_empty() {
        // Nothing at all — take from the start.
        let end = max_chars.min(text_len);
        let truncated: String = chars[..end].iter().collect();
        let suffix = if max_chars < text_len {
            "\n\n...[later conversation truncated]..."
        } else {
            ""
        };
        return format!("{truncated}{suffix}");
    }

    // --- Pick window that covers the most match positions -----------------
    match_positions.sort_unstable();

    let mut best_start: usize = 0;
    let mut best_count: usize = 0;
    let bias = max_chars / 4; // 25% before, 75% after
    for &candidate in &match_positions {
        let mut ws = candidate.saturating_sub(bias);
        let mut we = ws + max_chars;
        if we > lower_len {
            ws = lower_len.saturating_sub(max_chars);
            we = lower_len;
        }
        let count = match_positions.iter().filter(|&&p| ws <= p && p < we).count();
        if count > best_count {
            best_count = count;
            best_start = ws;
        }
    }

    // Python slices full_text[start:end]; positions came from lowercased text.
    // For ASCII (the common case) indices align. Clamp to the original length.
    let start = best_start.min(text_len);
    let end = text_len.min(start + max_chars);

    let truncated: String = chars[start..end].iter().collect();
    let prefix = if start > 0 {
        "...[earlier conversation truncated]...\n\n"
    } else {
        ""
    };
    let suffix = if end < text_len {
        "\n\n...[later conversation truncated]..."
    } else {
        ""
    };
    format!("{prefix}{truncated}{suffix}")
}

/// Find all starting positions (in char units) of `needle` within `haystack`.
/// Empty needle yields no positions.
fn find_all_char_positions(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return Vec::new();
    }
    let hay: Vec<char> = haystack.chars().collect();
    let nee: Vec<char> = needle.chars().collect();
    let mut out = Vec::new();
    if nee.len() > hay.len() {
        return out;
    }
    let last = hay.len() - nee.len();
    let mut i = 0;
    while i <= last {
        if hay[i..i + nee.len()] == nee[..] {
            out.push(i);
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Summarization prompt construction
// ---------------------------------------------------------------------------

/// Build the (system, user) prompt pair for summarizing a single session.
///
/// Mirrors `_summarize_session`'s prompt construction.
pub fn build_summary_prompts(
    conversation_text: &str,
    query: &str,
    session_meta: &Map<String, Value>,
) -> (String, String) {
    let system_prompt = "You are reviewing a past conversation transcript to help recall what happened. \
Summarize the conversation with a focus on the search topic. Include:\n\
1. What the user asked about or wanted to accomplish\n\
2. What actions were taken and what the outcomes were\n\
3. Key decisions, solutions found, or conclusions reached\n\
4. Any specific commands, files, URLs, or technical details that were important\n\
5. Anything left unresolved or notable\n\n\
Be thorough but concise. Preserve specific details (commands, paths, error messages) \
that would be useful to recall. Write in past tense as a factual recap."
        .to_string();

    let source = session_meta
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let started = format_timestamp(session_meta.get("started_at").unwrap_or(&Value::Null));

    let user_prompt = format!(
        "Search topic: {query}\n\
Session source: {source}\n\
Session date: {started}\n\n\
CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n\
Summarize this conversation with focus on: {query}"
    );

    (system_prompt, user_prompt)
}

// ---------------------------------------------------------------------------
// Recent sessions mode
// ---------------------------------------------------------------------------

/// Walk a session's parent chain to find the root (oldest ancestor) id.
///
/// Mirrors the lineage walk used in `_list_recent_sessions` and the inner
/// `_resolve_to_parent` of `session_search`. Cycle-safe via a visited set.
fn resolve_to_root<S: SessionStore>(db: &S, session_id: &str) -> String {
    let mut visited: HashSet<String> = HashSet::new();
    let mut sid = session_id.to_string();
    let mut root = session_id.to_string();
    while !sid.is_empty() && !visited.contains(&sid) {
        visited.insert(sid.clone());
        root = sid.clone();
        match db.get_session(&sid) {
            Ok(Some(s)) => {
                let parent = s
                    .get("parent_session_id")
                    .and_then(|v| v.as_str())
                    .filter(|p| !p.is_empty());
                match parent {
                    Some(p) => sid = p.to_string(),
                    None => break,
                }
            }
            _ => break,
        }
    }
    root
}

/// Return metadata for the most recent sessions (no LLM calls).
///
/// Mirrors `_list_recent_sessions`.
pub fn list_recent_sessions<S: SessionStore>(
    db: &S,
    limit: usize,
    current_session_id: Option<&str>,
) -> String {
    let exclude: Vec<String> = HIDDEN_SESSION_SOURCES.iter().map(|s| s.to_string()).collect();
    let sessions = match db.list_sessions_rich((limit + 5) as i64, &exclude, true) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Error listing recent sessions: {e}");
            return tool_error(&format!("Failed to list recent sessions: {e}"));
        }
    };

    // Resolve current session lineage to exclude it.
    let current_root: Option<String> = current_session_id.map(|cid| resolve_to_root(db, cid));

    let mut results: Vec<Value> = Vec::new();
    for s in &sessions {
        let sid = s.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if let Some(root) = &current_root {
            if sid == root || Some(sid) == current_session_id {
                continue;
            }
        }
        // Skip child/delegation sessions (they have parent_session_id).
        if s.get("parent_session_id")
            .and_then(|v| v.as_str())
            .map(|p| !p.is_empty())
            .unwrap_or(false)
        {
            continue;
        }
        let title = match s.get("title") {
            Some(Value::String(t)) if !t.is_empty() => Value::String(t.clone()),
            _ => Value::Null,
        };
        results.push(json!({
            "session_id": sid,
            "title": title,
            "source": s.get("source").and_then(|v| v.as_str()).unwrap_or(""),
            "started_at": s.get("started_at").cloned().unwrap_or(Value::String(String::new())),
            "last_active": s.get("last_active").cloned().unwrap_or(Value::String(String::new())),
            "message_count": s.get("message_count").cloned().unwrap_or(json!(0)),
            "preview": s.get("preview").and_then(|v| v.as_str()).unwrap_or(""),
        }));
        if results.len() >= limit {
            break;
        }
    }

    let count = results.len();
    json!({
        "success": true,
        "mode": "recent",
        "results": results,
        "count": count,
        "message": format!(
            "Showing {count} most recent sessions. Use a keyword query to search specific topics."
        ),
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Core search
// ---------------------------------------------------------------------------

/// Coerce a JSON `limit` argument into a clamped `[1, 5]` integer (default 3).
///
/// Mirrors the defensive coercion at the top of `session_search`.
pub fn coerce_limit(raw: &Value) -> i64 {
    let limit = if let Some(i) = raw.as_i64() {
        i
    } else if let Some(f) = raw.as_f64() {
        f as i64
    } else if let Some(s) = raw.as_str() {
        s.trim().parse::<i64>().unwrap_or(3)
    } else {
        // null / bool / object / array -> default
        3
    };
    limit.clamp(1, 5)
}

/// Search past sessions and return focused summaries of matching conversations.
///
/// Mirrors `session_search`. `summarizer` performs the auxiliary LLM call; pass
/// `None` to skip summarization entirely (every matched session falls back to a
/// raw preview, matching the Python behavior when the auxiliary model is
/// unavailable).
pub fn session_search<S: SessionStore>(
    query: &str,
    role_filter: Option<&str>,
    limit: &Value,
    db: Option<&S>,
    current_session_id: Option<&str>,
    summarizer: Option<&dyn Summarizer>,
) -> String {
    let db = match db {
        Some(d) => d,
        None => return tool_error("Session database not available."),
    };

    let limit = coerce_limit(limit) as usize;

    // Recent sessions mode: empty query -> metadata for recent sessions.
    if query.trim().is_empty() {
        return list_recent_sessions(db, limit, current_session_id);
    }
    let query = query.trim();

    // Parse role filter.
    let role_list: Option<Vec<String>> = role_filter.and_then(|rf| {
        let rf = rf.trim();
        if rf.is_empty() {
            None
        } else {
            let v: Vec<String> = rf
                .split(',')
                .map(|r| r.trim().to_string())
                .filter(|r| !r.is_empty())
                .collect();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
    });

    let exclude: Vec<String> = HIDDEN_SESSION_SOURCES.iter().map(|s| s.to_string()).collect();

    let raw_results = match db.search_messages(
        query,
        role_list.as_deref(),
        &exclude,
        50, // Get more matches to find unique sessions
        0,
    ) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Session search failed: {e}");
            return tool_error(&format!("Search failed: {e}"));
        }
    };

    if raw_results.is_empty() {
        return json!({
            "success": true,
            "query": query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        })
        .to_string();
    }

    let current_lineage_root: Option<String> =
        current_session_id.map(|cid| resolve_to_root(db, cid));

    // Group by resolved (parent) session_id, dedup, skip current lineage.
    // Preserve insertion order to match the Python dict iteration order.
    let mut seen_order: Vec<String> = Vec::new();
    let mut seen_sessions: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    for result in &raw_results {
        let raw_sid = match result.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let resolved_sid = resolve_to_root(db, &raw_sid);
        if let Some(root) = &current_lineage_root {
            if &resolved_sid == root {
                continue;
            }
        }
        if Some(raw_sid.as_str()) == current_session_id {
            continue;
        }
        if !seen_sessions.contains_key(&resolved_sid) {
            let mut entry = result.clone();
            entry.insert("session_id".to_string(), Value::String(resolved_sid.clone()));
            seen_sessions.insert(resolved_sid.clone(), entry);
            seen_order.push(resolved_sid.clone());
        }
        if seen_sessions.len() >= limit {
            break;
        }
    }

    // Prepare all sessions for summarization.
    struct Task {
        session_id: String,
        match_info: Map<String, Value>,
        conversation_text: String,
        session_meta: Map<String, Value>,
    }
    let mut tasks: Vec<Task> = Vec::new();
    for session_id in &seen_order {
        let match_info = seen_sessions.get(session_id).cloned().unwrap_or_default();
        let messages = match db.get_messages_as_conversation(session_id) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Failed to prepare session {session_id}: {e}");
                continue;
            }
        };
        if messages.is_empty() {
            continue;
        }
        let session_meta = db.get_session(session_id).ok().flatten().unwrap_or_default();
        let conversation_text = format_conversation(&messages);
        let conversation_text =
            truncate_around_matches(&conversation_text, query, MAX_SESSION_CHARS);
        tasks.push(Task {
            session_id: session_id.clone(),
            match_info,
            conversation_text,
            session_meta,
        });
    }

    // Summarize (sequentially in the Rust port; concurrency advisory).
    let mut summaries: Vec<Value> = Vec::new();
    for task in &tasks {
        let summary: Option<String> = match summarizer {
            Some(s) => {
                let (sys, user) =
                    build_summary_prompts(&task.conversation_text, query, &task.session_meta);
                s.summarize(&sys, &user)
            }
            None => None,
        };

        // Prefer resolved parent session metadata over FTS5 match metadata.
        let when = {
            let started = task.session_meta.get("started_at");
            let from_meta = matches!(started, Some(v) if !is_falsy(v));
            if from_meta {
                format_timestamp(started.unwrap())
            } else {
                format_timestamp(
                    task.match_info
                        .get("session_started")
                        .unwrap_or(&Value::Null),
                )
            }
        };
        let source = first_truthy_str(
            task.session_meta.get("source"),
            task.match_info.get("source"),
            "unknown",
        );
        let model = first_truthy_value(
            task.session_meta.get("model"),
            task.match_info.get("model"),
        );

        let summary_text = match summary {
            Some(text) if !text.is_empty() => text,
            _ => {
                // Fallback: raw preview so matched sessions aren't silently
                // dropped when the summarizer is unavailable.
                let preview = if !task.conversation_text.is_empty() {
                    let chars: Vec<char> = task.conversation_text.chars().collect();
                    let head_len = chars.len().min(500);
                    let head: String = chars[..head_len].iter().collect();
                    format!("{head}\n…[truncated]")
                } else {
                    "No preview available.".to_string()
                };
                format!("[Raw preview — summarization unavailable]\n{preview}")
            }
        };

        summaries.push(json!({
            "session_id": task.session_id,
            "when": when,
            "source": source,
            "model": model,
            "summary": summary_text,
        }));
    }

    let count = summaries.len();
    json!({
        "success": true,
        "query": query,
        "results": summaries,
        "count": count,
        "sessions_searched": seen_sessions.len(),
    })
    .to_string()
}

/// Python truthiness for the `or` chaining in metadata resolution: null, empty
/// string, 0, false, empty containers are falsy.
fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !*b,
        Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// `a or b or default` for string fields (Python truthiness on `a`/`b`).
fn first_truthy_str(a: Option<&Value>, b: Option<&Value>, default: &str) -> Value {
    if let Some(v) = a {
        if !is_falsy(v) {
            return v.clone();
        }
    }
    if let Some(v) = b {
        if !is_falsy(v) {
            return v.clone();
        }
    }
    Value::String(default.to_string())
}

/// `a or b` returning null when both are falsy (matches `session_meta.get("model") or match_info.get("model")`).
fn first_truthy_value(a: Option<&Value>, b: Option<&Value>) -> Value {
    if let Some(v) = a {
        if !is_falsy(v) {
            return v.clone();
        }
    }
    if let Some(v) = b {
        if !is_falsy(v) {
            return v.clone();
        }
    }
    Value::Null
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// The `session_search` tool JSON schema, matching `SESSION_SEARCH_SCHEMA`.
pub fn session_search_schema() -> Value {
    json!({
        "name": "session_search",
        "description":
            "Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- \
every past session is searchable, and this tool summarizes what happened.\n\n\
TWO MODES:\n\
1. Recent sessions (no query): Call with no arguments to see what was worked on recently. \
Returns titles, previews, and timestamps. Zero LLM cost, instant. \
Start here when the user asks what were we working on or what did we do recently.\n\
2. Keyword search (with query): Search for specific topics across all past sessions. \
Returns LLM-generated summaries of matching sessions.\n\n\
USE THIS PROACTIVELY when:\n\
- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n\
- The user asks about a topic you worked on before but don't have in current context\n\
- The user references a project, person, or concept that seems familiar but isn't in memory\n\
- You want to check if you've solved a similar problem before\n\
- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n\
Don't hesitate to search when it is actually cross-session -- it's fast and cheap. \
Better to search and confirm than to guess or ask the user to repeat themselves.\n\n\
Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), \
phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). \
IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses \
sessions that only mention some terms. If a broad OR query returns nothing, try individual \
keyword searches in parallel. Returns summaries of the top matching sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead (returns titles, previews, timestamps with no LLM cost).",
                },
                "role_filter": {
                    "type": "string",
                    "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max sessions to summarize (default: 3, max: 5).",
                    "default": 3,
                },
            },
            "required": [],
        },
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- in-memory fake store ----------------------------------------
    #[derive(Default)]
    struct FakeStore {
        search: Vec<Map<String, Value>>,
        recent: Vec<Map<String, Value>>,
        sessions: BTreeMap<String, Map<String, Value>>,
        conversations: BTreeMap<String, Vec<Map<String, Value>>>,
    }

    fn obj(pairs: &[(&str, Value)]) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    impl SessionStore for FakeStore {
        fn search_messages(
            &self,
            _query: &str,
            _role_filter: Option<&[String]>,
            _exclude_sources: &[String],
            _limit: i64,
            _offset: i64,
        ) -> Result<Vec<Map<String, Value>>, String> {
            Ok(self.search.clone())
        }
        fn list_sessions_rich(
            &self,
            _limit: i64,
            _exclude_sources: &[String],
            _order_by_last_active: bool,
        ) -> Result<Vec<Map<String, Value>>, String> {
            Ok(self.recent.clone())
        }
        fn get_session(&self, session_id: &str) -> Result<Option<Map<String, Value>>, String> {
            Ok(self.sessions.get(session_id).cloned())
        }
        fn get_messages_as_conversation(
            &self,
            session_id: &str,
        ) -> Result<Vec<Map<String, Value>>, String> {
            Ok(self.conversations.get(session_id).cloned().unwrap_or_default())
        }
    }

    struct FixedSummarizer(Option<String>);
    impl Summarizer for FixedSummarizer {
        fn summarize(&self, _s: &str, _u: &str) -> Option<String> {
            self.0.clone()
        }
    }

    #[test]
    fn test_format_timestamp_variants() {
        assert_eq!(format_timestamp(&Value::Null), "unknown");
        // numeric string passthrough as date
        let s = format_timestamp(&json!("1700000000"));
        assert!(s.contains("2023") || s.contains("at"), "got {s}");
        // non-numeric string returned verbatim
        assert_eq!(format_timestamp(&json!("2023-01-02T03:04:05")), "2023-01-02T03:04:05");
        // numeric value formats
        let n = format_timestamp(&json!(0));
        assert!(n.contains("at"), "got {n}");
    }

    #[test]
    fn test_is_numeric_string() {
        assert!(is_numeric_string("1700000000"));
        assert!(is_numeric_string("1700.5"));
        assert!(is_numeric_string("-12.3"));
        assert!(!is_numeric_string("2023-01-02T03:04"));
        assert!(!is_numeric_string(""));
        assert!(!is_numeric_string("abc"));
    }

    #[test]
    fn test_format_conversation_roles() {
        let msgs = vec![
            obj(&[("role", json!("user")), ("content", json!("hello"))]),
            obj(&[
                ("role", json!("assistant")),
                ("content", json!("hi there")),
                ("tool_calls", json!([{"name": "search"}, {"function": {"name": "read"}}])),
            ]),
            obj(&[
                ("role", json!("tool")),
                ("tool_name", json!("search")),
                ("content", json!("result")),
            ]),
        ];
        let out = format_conversation(&msgs);
        assert!(out.contains("[USER]: hello"));
        assert!(out.contains("[ASSISTANT]: [Called: search, read]"));
        assert!(out.contains("[ASSISTANT]: hi there"));
        assert!(out.contains("[TOOL:search]: result"));
    }

    #[test]
    fn test_format_conversation_tool_truncation() {
        let big = "x".repeat(1200);
        let msgs = vec![obj(&[
            ("role", json!("tool")),
            ("tool_name", json!("t")),
            ("content", json!(big)),
        ])];
        let out = format_conversation(&msgs);
        assert!(out.contains("...[truncated]..."));
        // 250 + marker + 250 < 1200
        assert!(out.len() < 1200 + 100);
    }

    #[test]
    fn test_format_conversation_null_content() {
        let msgs = vec![obj(&[("role", json!("user")), ("content", Value::Null)])];
        assert_eq!(format_conversation(&msgs), "[USER]: ");
    }

    #[test]
    fn test_truncate_short_returns_full() {
        let t = "short text";
        assert_eq!(truncate_around_matches(t, "x", 1000), t);
    }

    #[test]
    fn test_truncate_phrase_window() {
        // Build text with the phrase near the middle.
        let mut text = String::new();
        text.push_str(&"a".repeat(5000));
        text.push_str(" needle ");
        text.push_str(&"b".repeat(5000));
        let out = truncate_around_matches(&text, "needle", 1000);
        assert!(out.contains("needle"));
        assert!(out.contains("...[earlier conversation truncated]..."));
        assert!(out.contains("...[later conversation truncated]..."));
    }

    #[test]
    fn test_truncate_no_match_from_start() {
        let text = "z".repeat(5000);
        let out = truncate_around_matches(&text, "qqqq", 1000);
        assert!(out.starts_with("zzz"));
        assert!(out.contains("...[later conversation truncated]..."));
        assert!(!out.contains("earlier"));
    }

    #[test]
    fn test_find_positions() {
        assert_eq!(find_all_char_positions("ababab", "ab"), vec![0, 2, 4]);
        assert_eq!(find_all_char_positions("abc", ""), Vec::<usize>::new());
        assert_eq!(find_all_char_positions("a", "abc"), Vec::<usize>::new());
    }

    #[test]
    fn test_coerce_limit() {
        assert_eq!(coerce_limit(&json!(3)), 3);
        assert_eq!(coerce_limit(&json!(10)), 5);
        assert_eq!(coerce_limit(&json!(0)), 1);
        assert_eq!(coerce_limit(&Value::Null), 3);
        assert_eq!(coerce_limit(&json!("4")), 4);
        assert_eq!(coerce_limit(&json!("nope")), 3);
        assert_eq!(coerce_limit(&json!(2.9)), 2);
    }

    #[test]
    fn test_clamp_max_concurrency() {
        assert_eq!(clamp_max_concurrency(None, 3), 3);
        assert_eq!(clamp_max_concurrency(Some(&Value::Null), 3), 3);
        assert_eq!(clamp_max_concurrency(Some(&json!(2)), 3), 2);
        assert_eq!(clamp_max_concurrency(Some(&json!(99)), 3), 5);
        assert_eq!(clamp_max_concurrency(Some(&json!(0)), 3), 1);
        assert_eq!(clamp_max_concurrency(Some(&json!("4")), 3), 4);
        assert_eq!(clamp_max_concurrency(Some(&json!("x")), 3), 3);
    }

    #[test]
    fn test_session_search_no_db() {
        let out = session_search::<FakeStore>(
            "hi",
            None,
            &json!(3),
            None,
            None,
            None,
        );
        assert!(out.contains("Session database not available."));
        assert!(out.contains("\"success\":false"));
    }

    #[test]
    fn test_session_search_no_matches() {
        let store = FakeStore::default();
        let out = session_search(
            "hi",
            None,
            &json!(3),
            Some(&store),
            None,
            None,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["count"], json!(0));
        assert_eq!(v["message"], json!("No matching sessions found."));
    }

    #[test]
    fn test_session_search_recent_mode() {
        let mut store = FakeStore::default();
        store.recent = vec![
            obj(&[
                ("id", json!("s1")),
                ("title", json!("First")),
                ("source", json!("cli")),
                ("started_at", json!(1700000000.0)),
                ("message_count", json!(5)),
                ("preview", json!("hello world")),
            ]),
            // child session skipped
            obj(&[
                ("id", json!("s2")),
                ("parent_session_id", json!("s1")),
            ]),
        ];
        let out = session_search(
            "",
            None,
            &json!(3),
            Some(&store),
            None,
            None,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["mode"], json!("recent"));
        assert_eq!(v["count"], json!(1));
        assert_eq!(v["results"][0]["session_id"], json!("s1"));
        assert_eq!(v["results"][0]["title"], json!("First"));
    }

    #[test]
    fn test_session_search_with_summary_and_resolution() {
        let mut store = FakeStore::default();
        // FTS5 hit lives in child session "child", resolves to parent "root".
        store.search = vec![obj(&[
            ("session_id", json!("child")),
            ("source", json!("childsrc")),
            ("model", json!("childmodel")),
            ("session_started", json!(111.0)),
        ])];
        store.sessions.insert(
            "child".to_string(),
            obj(&[
                ("id", json!("child")),
                ("parent_session_id", json!("root")),
            ]),
        );
        store.sessions.insert(
            "root".to_string(),
            obj(&[
                ("id", json!("root")),
                ("source", json!("rootsrc")),
                ("model", json!("rootmodel")),
                ("started_at", json!(1700000000.0)),
            ]),
        );
        store.conversations.insert(
            "root".to_string(),
            vec![obj(&[("role", json!("user")), ("content", json!("about widgets"))])],
        );

        let summarizer = FixedSummarizer(Some("A summary about widgets".to_string()));
        let out = session_search(
            "widgets",
            None,
            &json!(3),
            Some(&store),
            None,
            Some(&summarizer),
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["count"], json!(1));
        let entry = &v["results"][0];
        assert_eq!(entry["session_id"], json!("root"));
        // Prefer parent metadata over child match info.
        assert_eq!(entry["source"], json!("rootsrc"));
        assert_eq!(entry["model"], json!("rootmodel"));
        assert_eq!(entry["summary"], json!("A summary about widgets"));
        assert_eq!(v["sessions_searched"], json!(1));
    }

    #[test]
    fn test_session_search_fallback_preview_when_no_summary() {
        let mut store = FakeStore::default();
        store.search = vec![obj(&[("session_id", json!("s1"))])];
        store
            .sessions
            .insert("s1".to_string(), obj(&[("id", json!("s1"))]));
        store.conversations.insert(
            "s1".to_string(),
            vec![obj(&[("role", json!("user")), ("content", json!("some content here"))])],
        );
        let out = session_search(
            "content",
            None,
            &json!(3),
            Some(&store),
            None,
            None, // no summarizer
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        let summary = v["results"][0]["summary"].as_str().unwrap();
        assert!(summary.contains("[Raw preview — summarization unavailable]"));
        assert!(summary.contains("some content here"));
        // source falls back to "unknown"
        assert_eq!(v["results"][0]["source"], json!("unknown"));
        assert_eq!(v["results"][0]["model"], Value::Null);
    }

    #[test]
    fn test_session_search_excludes_current_lineage() {
        let mut store = FakeStore::default();
        store.search = vec![obj(&[("session_id", json!("cur"))])];
        store
            .sessions
            .insert("cur".to_string(), obj(&[("id", json!("cur"))]));
        let out = session_search(
            "anything",
            None,
            &json!(3),
            Some(&store),
            Some("cur"),
            None,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        // Only result was the current session -> dropped.
        assert_eq!(v["count"], json!(0));
        assert_eq!(v["sessions_searched"], json!(0));
    }

    #[test]
    fn test_role_filter_parsing_via_search() {
        // Ensure empty role filter doesn't crash and search still runs.
        let store = FakeStore::default();
        let out = session_search(
            "q",
            Some(" , "),
            &json!(3),
            Some(&store),
            None,
            None,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["count"], json!(0));
    }

    #[test]
    fn test_schema_shape() {
        let s = session_search_schema();
        assert_eq!(s["name"], json!("session_search"));
        assert_eq!(s["parameters"]["properties"]["limit"]["default"], json!(3));
        assert_eq!(s["parameters"]["required"], json!([]));
    }
}
