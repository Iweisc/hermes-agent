//! Yuanbao platform adapter (native Rust port of `gateway/platforms/yuanbao.py`).
//!
//! Connects to the Yuanbao WebSocket gateway, handles authentication (AUTH_BIND),
//! heartbeat, reconnection, message receive (T05) and send (T06).
//!
//! This is a faithful port of the Python module. The original is heavily
//! `asyncio`-based; here the deterministic logic (Markdown chunking, sign-token
//! signing/caching, inbound field-extraction / decode middleware, message-body
//! construction, truncation, access policy, etc.) is ported exactly, while the
//! network surfaces (sign-token HTTP, resource download) use `reqwest::blocking`
//! and keep the API request/response shapes identical to the Python original.
//!
//! Cross-references:
//!   - [`crate::gw_yuanbao_proto`] — wire encode/decode
//!   - [`crate::gw_yuanbao_media`] — COS upload / download / mime helpers
//!   - [`crate::gw_yuanbao_sticker`] — sticker registry
//!
//! Configuration in config.yaml (or via env vars):
//! ```yaml
//! platforms:
//!   yuanbao:
//!     extra:
//!       app_id: "..."              # or YUANBAO_APP_ID
//!       app_secret: "..."          # or YUANBAO_APP_SECRET
//!       bot_id: "..."              # or YUANBAO_BOT_ID  (optional, returned by sign-token)
//!       ws_url: "wss://..."        # or YUANBAO_WS_URL
//!       api_domain: "https://..."  # or YUANBAO_API_DOMAIN
//! ```

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{FixedOffset, Utc};
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Version / platform constants (used in AUTH_BIND and sign-token headers)
// ---------------------------------------------------------------------------

/// App / bot version reported in sign-token + AUTH_BIND headers.
/// Python sources this from `hermes_cli.__version__` (falling back to "0.0.0").
pub const APP_VERSION: &str = "0.0.0";
/// Bot version reported in sign-token + AUTH_BIND headers.
pub const BOT_VERSION: &str = "0.0.0";

/// Single source instance id (mirrors `yuanbao_proto.HERMES_INSTANCE_ID`).
pub fn yuanbao_instance_id() -> String {
    crate::gw_yuanbao_proto::HERMES_INSTANCE_ID.to_string()
}

/// `sys.platform` equivalent for header `X-OperationSystem`.
pub fn operation_system() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        std::env::consts::OS
    }
}

// ---------------------------------------------------------------------------
// Module-level constants
// ---------------------------------------------------------------------------

pub const DEFAULT_WS_GATEWAY_URL: &str = "wss://bot-wss.yuanbao.tencent.com/wss/connection";
pub const DEFAULT_API_DOMAIN: &str = "https://bot.yuanbao.tencent.com";

pub const HEARTBEAT_INTERVAL_SECONDS: f64 = 30.0;
pub const CONNECT_TIMEOUT_SECONDS: f64 = 15.0;
pub const AUTH_TIMEOUT_SECONDS: f64 = 10.0;
pub const MAX_RECONNECT_ATTEMPTS: u32 = 100;
pub const DEFAULT_SEND_TIMEOUT: f64 = 30.0;

/// Close codes that indicate permanent errors — do NOT reconnect.
pub const NO_RECONNECT_CLOSE_CODES: &[u16] = &[4012, 4013, 4014, 4018, 4019, 4021];

/// Heartbeat timeout threshold — N consecutive missed pongs trigger reconnect.
pub const HEARTBEAT_TIMEOUT_THRESHOLD: u32 = 2;

/// Auth error codes — permanent auth failure, re-sign token.
pub const AUTH_FAILED_CODES: &[i64] = &[4001, 4002, 4003];
/// Auth error codes — transient, can retry with same token.
pub const AUTH_RETRYABLE_CODES: &[i64] = &[4010, 4011, 4099];

/// Reply Heartbeat: send RUNNING every N seconds.
pub const REPLY_HEARTBEAT_INTERVAL_S: f64 = 2.0;
/// Reply Heartbeat: auto-stop after N seconds of inactivity.
pub const REPLY_HEARTBEAT_TIMEOUT_S: f64 = 30.0;

/// Reference dedup TTL (5 minutes).
pub const REPLY_REF_TTL_S: f64 = 300.0;

/// Slow-response hint: push a waiting message when the agent produces no data
/// for this duration (seconds).
pub const SLOW_RESPONSE_TIMEOUT_S: f64 = 120.0;
pub const SLOW_RESPONSE_MESSAGE: &str = "任务有点复杂，正在努力处理中，请耐心等待...";

/// Observed-media backfill: how many recent transcript messages to scan.
pub const OBSERVED_MEDIA_BACKFILL_LOOKBACK: usize = 50;
/// Max number of resource references to resolve per inbound turn.
pub const OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN: usize = 12;

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ===========================================================================
// MarkdownProcessor — fence/table-aware Markdown chunking & sanitization
// ===========================================================================

/// Encapsulates all Markdown-related utilities for the Yuanbao platform.
pub struct MarkdownProcessor;

impl MarkdownProcessor {
    // -- Fence detection ---------------------------------------------------

    /// Detect whether the text has unclosed code block fences.
    pub fn has_unclosed_fence(text: &str) -> bool {
        let mut in_fence = false;
        for line in text.split('\n') {
            if line.starts_with("```") {
                in_fence = !in_fence;
            }
        }
        in_fence
    }

    // -- Table detection ---------------------------------------------------

    /// Detect whether the text ends with a table row (last non-empty line
    /// starts and ends with `|`).
    pub fn ends_with_table_row(text: &str) -> bool {
        let trimmed = text.trim_end();
        if trimmed.is_empty() {
            return false;
        }
        let last_line = trimmed.split('\n').last().unwrap_or("").trim();
        last_line.starts_with('|') && last_line.ends_with('|')
    }

    // -- Paragraph boundary splitting --------------------------------------

    /// Find the nearest paragraph boundary split point within `max_chars`,
    /// returning `(head, tail)` such that `head + tail == text`.
    ///
    /// Length is measured in Unicode scalar values (Python `len` over `str`).
    pub fn split_at_paragraph_boundary(text: &str, max_chars: usize) -> (String, String) {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() <= max_chars {
            return (text.to_string(), String::new());
        }

        // Window = first `max_chars` characters.
        let window: String = chars[..max_chars].iter().collect();

        // 1. Prefer the last blank line (\n\n) as paragraph boundary.
        if let Some(pos) = window.rfind("\n\n") {
            if pos > 0 {
                // pos is a byte offset within `window`; +2 to include "\n\n".
                let split_byte = pos + 2;
                return Self::split_text_at_byte(text, &window, split_byte);
            }
        }

        // 2. Then find the last newline after a sentence-ending punctuation.
        let mut best_end: Option<usize> = None;
        {
            // Iterate char boundaries: look for [。！？.!?] immediately followed by \n.
            let wchars: Vec<(usize, char)> = window.char_indices().collect();
            for i in 0..wchars.len() {
                let (_idx, ch) = wchars[i];
                let is_sentence_end =
                    matches!(ch, '。' | '！' | '？' | '.' | '!' | '?');
                if is_sentence_end {
                    // Next char must be '\n'.
                    if i + 1 < wchars.len() {
                        let (next_idx, next_ch) = wchars[i + 1];
                        if next_ch == '\n' {
                            // end position is just after the '\n'.
                            best_end = Some(next_idx + 1);
                        }
                    }
                }
            }
        }
        if let Some(end_byte) = best_end {
            if end_byte > 0 {
                return Self::split_text_at_byte(text, &window, end_byte);
            }
        }

        // 3. Fallback: find the last newline.
        if let Some(pos) = window.rfind('\n') {
            if pos > 0 {
                return Self::split_text_at_byte(text, &window, pos + 1);
            }
        }

        // 4. No valid split point found, force split at window boundary.
        let head: String = chars[..max_chars].iter().collect();
        let tail: String = chars[max_chars..].iter().collect();
        (head, tail)
    }

    /// Helper: given a byte offset within `window` (a prefix of `text`), split
    /// `text` into (head, tail) at that offset, counting characters so the
    /// split lands on a char boundary in the full text.
    fn split_text_at_byte(text: &str, window: &str, byte_off: usize) -> (String, String) {
        // Count chars in the window prefix up to byte_off.
        let prefix = &window[..byte_off];
        let nchars = prefix.chars().count();
        let chars: Vec<char> = text.chars().collect();
        let head: String = chars[..nchars].iter().collect();
        let tail: String = chars[nchars..].iter().collect();
        (head, tail)
    }

    // -- Atomic block helpers ---------------------------------------------

    /// Whether an atomic block is a code block (starts with ```).
    pub fn is_fence_atom(text: &str) -> bool {
        text.trim_start().starts_with("```")
    }

    /// Whether an atomic block is a table (first line starts/ends with `|`).
    pub fn is_table_atom(text: &str) -> bool {
        let first_line = text.split('\n').next().unwrap_or("").trim();
        first_line.starts_with('|') && first_line.ends_with('|')
    }

    fn is_table_line(line: &str) -> bool {
        let stripped = line.trim();
        stripped.starts_with('|') && stripped.ends_with('|')
    }

    /// Split text into a list of indivisible "atomic blocks".
    pub fn split_into_atoms(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut atoms: Vec<String> = Vec::new();
        let mut current_lines: Vec<String> = Vec::new();
        let mut in_fence = false;

        let flush = |current_lines: &mut Vec<String>, atoms: &mut Vec<String>| {
            if !current_lines.is_empty() {
                let atom = current_lines.join("\n");
                if !atom.trim().is_empty() {
                    atoms.push(atom);
                }
                current_lines.clear();
            }
        };

        for line in lines {
            if in_fence {
                current_lines.push(line.to_string());
                if line.starts_with("```") && current_lines.len() > 1 {
                    in_fence = false;
                    flush(&mut current_lines, &mut atoms);
                }
            } else if line.starts_with("```") {
                flush(&mut current_lines, &mut atoms);
                in_fence = true;
                current_lines.push(line.to_string());
            } else if Self::is_table_line(line) {
                if let Some(last) = current_lines.last() {
                    if !Self::is_table_line(last) {
                        flush(&mut current_lines, &mut atoms);
                    }
                }
                current_lines.push(line.to_string());
            } else if line.trim().is_empty() {
                flush(&mut current_lines, &mut atoms);
            } else {
                if let Some(last) = current_lines.last() {
                    if Self::is_table_line(last) {
                        flush(&mut current_lines, &mut atoms);
                    }
                }
                current_lines.push(line.to_string());
            }
        }

        flush(&mut current_lines, &mut atoms);
        atoms
    }

    fn clen(s: &str) -> usize {
        s.chars().count()
    }

    // -- Core: chunk splitting ---------------------------------------------

    /// Split Markdown text into multiple chunks by `max_chars`, keeping code
    /// blocks and table rows intact.
    pub fn chunk_markdown_text(text: &str, max_chars: usize) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        if Self::clen(text) <= max_chars {
            return vec![text.to_string()];
        }

        // Phase 1: Extract atomic blocks.
        let atoms = Self::split_into_atoms(text);

        // Phase 2: Greedy merge.
        let mut chunks: Vec<String> = Vec::new();
        let mut indivisible_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut current_parts: Vec<String> = Vec::new();
        let mut current_len: usize = 0;

        for atom in &atoms {
            let atom_len = Self::clen(atom);
            let mut sep_len = if current_parts.is_empty() { 0 } else { 2 };
            let projected_len = current_len + sep_len + atom_len;

            if projected_len > max_chars && !current_parts.is_empty() {
                chunks.push(current_parts.join("\n\n"));
                current_parts.clear();
                current_len = 0;
                sep_len = 0;
            }

            if current_parts.is_empty()
                && atom_len > max_chars
                && (Self::is_fence_atom(atom) || Self::is_table_atom(atom))
            {
                indivisible_set.insert(chunks.len());
                chunks.push(atom.clone());
                continue;
            }

            current_parts.push(atom.clone());
            current_len += sep_len + atom_len;
        }
        if !current_parts.is_empty() {
            chunks.push(current_parts.join("\n\n"));
        }

        // Phase 3: Post-processing — split still-oversized chunks.
        let mut result: Vec<String> = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            if Self::clen(chunk) <= max_chars {
                result.push(chunk.clone());
                continue;
            }
            if indivisible_set.contains(&idx) {
                result.push(chunk.clone());
                continue;
            }
            if Self::has_unclosed_fence(chunk) {
                result.push(chunk.clone());
                continue;
            }

            let mut remaining = chunk.clone();
            while Self::clen(&remaining) > max_chars {
                let (mut head, mut rem) =
                    Self::split_at_paragraph_boundary(&remaining, max_chars);
                if head.is_empty() {
                    let chars: Vec<char> = remaining.chars().collect();
                    head = chars[..max_chars].iter().collect();
                    rem = chars[max_chars..].iter().collect();
                }
                remaining = rem;
                if !head.is_empty() {
                    result.push(head);
                }
            }
            if !remaining.is_empty() {
                result.push(remaining);
            }
        }

        // Phase 4: Merge small chunks with neighbours.
        if result.len() > 1 {
            let mut merged: Vec<String> = vec![result[0].clone()];
            for chunk in &result[1..] {
                let prev = merged.last().unwrap().clone();
                let combined = format!("{prev}\n\n{chunk}");
                if Self::clen(&combined) <= max_chars {
                    *merged.last_mut().unwrap() = combined;
                } else {
                    merged.push(chunk.clone());
                }
            }
            result = merged;
        }

        result.into_iter().filter(|c| !c.is_empty()).collect()
    }

    // -- Block separator inference -----------------------------------------

    /// Infer the separator (`\n` or `\n\n`) to use between two split chunks.
    pub fn infer_block_separator(prev_chunk: &str, next_chunk: &str) -> &'static str {
        let prev_trimmed = prev_chunk.trim_end();
        let next_trimmed = next_chunk.trim_start();

        if prev_trimmed.ends_with("```") || next_trimmed.starts_with("```") {
            return "\n";
        }

        if Self::ends_with_table_row(prev_chunk) {
            let first_line = next_trimmed.split('\n').next().unwrap_or("").trim();
            if first_line.starts_with('|') && first_line.ends_with('|') {
                return "\n";
            }
        }

        "\n\n"
    }

    // -- Streaming fence merge ---------------------------------------------

    /// Stream-aware fence-conscious chunk merging.
    pub fn merge_block_streaming_fences(chunks: &[String]) -> Vec<String> {
        if chunks.is_empty() {
            return Vec::new();
        }
        let mut result: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < chunks.len() {
            let mut current = chunks[i].clone();
            while Self::has_unclosed_fence(&current) && i + 1 < chunks.len() {
                let sep = Self::infer_block_separator(&current, &chunks[i + 1]);
                current = format!("{current}{sep}{}", chunks[i + 1]);
                i += 1;
            }
            result.push(current);
            i += 1;
        }
        result
    }

    // -- Outer fence stripping ---------------------------------------------

    /// Strip an outer ```markdown ... ``` fence, if present.
    pub fn strip_outer_markdown_fence(text: &str) -> String {
        if text.is_empty() {
            return text.to_string();
        }
        let lines: Vec<&str> = text.split('\n').collect();
        if lines.len() < 3 {
            return text.to_string();
        }
        let first_line = lines[0].trim();
        let last_line = lines[lines.len() - 1].trim();

        // First line must be ```markdown / ```md (case-insensitive), nothing else.
        if !Self::is_markdown_fence_open(first_line) {
            return text.to_string();
        }
        if last_line != "```" {
            return text.to_string();
        }
        lines[1..lines.len() - 1].join("\n")
    }

    fn is_markdown_fence_open(line: &str) -> bool {
        // ^```(markdown|md)?\s*$  case-insensitive
        let l = line;
        if !l.starts_with("```") {
            return false;
        }
        let rest = &l[3..];
        let rest_trim = rest.trim_end();
        let lower = rest_trim.to_ascii_lowercase();
        lower.is_empty() || lower == "markdown" || lower == "md"
    }

    // -- Table sanitization ------------------------------------------------

    /// Sanitize common AI-generated Markdown table formatting issues.
    pub fn sanitize_markdown_table(text: &str) -> String {
        if !text.contains('|') {
            return text.to_string();
        }
        let lines: Vec<&str> = text.split('\n').collect();
        let mut result_lines: Vec<String> = Vec::new();

        for line in lines {
            let stripped = line.trim();
            if stripped.starts_with('|') && stripped.ends_with('|') {
                if Self::is_separator_row(stripped) {
                    let cells: Vec<&str> = stripped.split('|').collect();
                    let normalized = cells
                        .iter()
                        .map(|cell| {
                            let c = cell.trim();
                            if c.is_empty() {
                                (*cell).to_string()
                            } else {
                                c.to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("|");
                    result_lines.push(normalized);
                } else if stripped == "||"
                    || stripped.replace('|', "").trim().is_empty()
                {
                    continue;
                } else {
                    result_lines.push(stripped.to_string());
                }
            } else {
                result_lines.push(line.to_string());
            }
        }

        result_lines.join("\n")
    }

    /// Match `^\|[\s\-:]+(\|[\s\-:]+)+\|$`.
    fn is_separator_row(s: &str) -> bool {
        if !s.starts_with('|') || !s.ends_with('|') {
            return false;
        }
        let inner = &s[1..s.len() - 1];
        // Must have at least one '|' (i.e. >= 2 cells).
        if !inner.contains('|') {
            return false;
        }
        for cell in inner.split('|') {
            if cell.is_empty() {
                return false; // each `[\s\-:]+` must be non-empty
            }
            if !cell
                .chars()
                .all(|c| c.is_whitespace() || c == '-' || c == ':')
            {
                return false;
            }
        }
        true
    }

    // -- Markdown hint prompt ----------------------------------------------

    /// Markdown rendering hint appended to the system prompt.
    pub fn markdown_hint_system_prompt() -> String {
        concat!(
            "The current platform supports Markdown rendering. You can use the following formats:\n",
            "- Code blocks: ```language\\ncode\\n```\n",
            "- Tables: | col1 | col2 |\\n|---|---|\\n| val1 | val2 |\n",
            "- Bold: **text** / Italic: *text*\n",
            "Please use Markdown formatting when appropriate to improve readability."
        )
        .to_string()
    }
}

// ===========================================================================
// SignManager — sign-token acquisition / caching / signature
// ===========================================================================

/// A cached sign-token entry.
#[derive(Debug, Clone, Default)]
pub struct TokenData {
    pub token: String,
    pub bot_id: String,
    pub duration: i64,
    pub product: String,
    pub source: String,
    pub expire_ts: f64,
}

impl TokenData {
    /// Convert to the dict shape the Python code returns from `get_token`.
    pub fn to_json(&self) -> Value {
        json!({
            "token": self.token,
            "bot_id": self.bot_id,
            "duration": self.duration,
            "product": self.product,
            "source": self.source,
            "expire_ts": self.expire_ts,
        })
    }
}

/// Sign-token signing + caching logic. Static methods match the Python
/// `SignManager` classmethods; the process-wide cache is held in a `Mutex`.
pub struct SignManager;

impl SignManager {
    pub const TOKEN_PATH: &'static str = "/api/v5/robotLogic/sign-token";
    pub const RETRYABLE_CODE: i64 = 10099;
    pub const MAX_RETRIES: u32 = 3;
    pub const RETRY_DELAY_S: f64 = 1.0;
    /// Early refresh margin (seconds).
    pub const CACHE_REFRESH_MARGIN_S: f64 = 60.0;
    pub const HTTP_TIMEOUT_S: f64 = 10.0;

    fn cache() -> &'static Mutex<HashMap<String, TokenData>> {
        use std::sync::OnceLock;
        static CACHE: OnceLock<Mutex<HashMap<String, TokenData>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Compute HMAC-SHA256 signature (aligned with the TypeScript original).
    ///
    /// `plain = nonce + timestamp + app_key + app_secret`;
    /// `signature = HMAC-SHA256(key=app_secret, msg=plain).hexdigest()`.
    pub fn compute_signature(
        nonce: &str,
        timestamp: &str,
        app_key: &str,
        app_secret: &str,
    ) -> String {
        let plain = format!("{nonce}{timestamp}{app_key}{app_secret}");
        let mut mac = HmacSha256::new_from_slice(app_secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(plain.as_bytes());
        let result = mac.finalize().into_bytes();
        hex_lower(&result)
    }

    /// Build a Beijing-time ISO-8601 timestamp (no milliseconds).
    /// Format: `2006-01-02T15:04:05+08:00`.
    pub fn build_timestamp() -> String {
        let tz = FixedOffset::east_opt(8 * 3600).unwrap();
        let bjtime = Utc::now().with_timezone(&tz);
        bjtime.format("%Y-%m-%dT%H:%M:%S+08:00").to_string()
    }

    /// Whether the cache entry is valid (not expired with margin).
    pub fn is_cache_valid(entry: &TokenData) -> bool {
        entry.expire_ts - now_secs() > Self::CACHE_REFRESH_MARGIN_S
    }

    /// Remove all expired entries from the token cache; returns count purged.
    pub fn purge_expired() -> usize {
        let now = now_secs();
        let mut cache = Self::cache().lock().unwrap();
        let before = cache.len();
        cache.retain(|_, v| !(now - v.expire_ts > 0.0));
        before - cache.len()
    }

    /// Clear cached entries (e.g. on disconnect).
    pub fn clear_cache() {
        Self::cache().lock().unwrap().clear();
    }

    /// Send a sign-ticket HTTP request with auto-retry (blocking).
    ///
    /// Returns the parsed `data` object from the API response.
    pub fn fetch(
        app_key: &str,
        app_secret: &str,
        api_domain: &str,
        route_env: &str,
    ) -> Result<Value, String> {
        let url = format!("{}{}", api_domain.trim_end_matches('/'), Self::TOKEN_PATH);
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(Self::HTTP_TIMEOUT_S))
            .build()
            .map_err(|e| format!("http client build failed: {e}"))?;

        for attempt in 0..=Self::MAX_RETRIES {
            let nonce = random_hex(16);
            let timestamp = Self::build_timestamp();
            let signature = Self::compute_signature(&nonce, &timestamp, app_key, app_secret);

            let payload = json!({
                "app_key": app_key,
                "nonce": nonce,
                "signature": signature,
                "timestamp": timestamp,
            });

            let mut req = client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("X-AppVersion", APP_VERSION)
                .header("X-OperationSystem", operation_system())
                .header("X-Instance-Id", yuanbao_instance_id())
                .header("X-Bot-Version", BOT_VERSION);
            if !route_env.is_empty() {
                req = req.header("X-Route-Env", route_env);
            }

            log::info!(
                "Sign token request: url={url}{}",
                if attempt > 0 {
                    format!(" (retry {attempt}/{})", Self::MAX_RETRIES)
                } else {
                    String::new()
                }
            );

            let response = req
                .json(&payload)
                .send()
                .map_err(|e| format!("sign-token request failed: {e}"))?;

            let status = response.status();
            if status.as_u16() != 200 {
                let body = response.text().unwrap_or_default();
                let snippet: String = body.chars().take(200).collect();
                return Err(format!("Sign token API returned {status}: {snippet}"));
            }

            let result_data: Value = response
                .json()
                .map_err(|e| format!("Sign token response parse error: {e}"))?;

            let code = result_data.get("code").and_then(Value::as_i64);
            if code == Some(0) {
                let data = result_data.get("data");
                match data {
                    Some(d) if d.is_object() => {
                        log::info!(
                            "Sign token success: bot_id={}",
                            d.get("bot_id").and_then(Value::as_str).unwrap_or("")
                        );
                        return Ok(d.clone());
                    }
                    _ => {
                        return Err(format!(
                            "Sign token response missing 'data' field: {result_data}"
                        ));
                    }
                }
            }

            if code == Some(Self::RETRYABLE_CODE) && attempt < Self::MAX_RETRIES {
                log::warn!(
                    "Sign token retryable: code={:?}, retrying in {}s (attempt={}/{})",
                    code,
                    Self::RETRY_DELAY_S,
                    attempt + 1,
                    Self::MAX_RETRIES
                );
                std::thread::sleep(std::time::Duration::from_secs_f64(Self::RETRY_DELAY_S));
                continue;
            }

            let msg = result_data
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or("");
            return Err(format!("Sign token error: code={code:?}, msg={msg}"));
        }

        Err("Sign token failed: max retries exceeded".to_string())
    }

    fn store_from_data(app_key: &str, data: &Value) -> TokenData {
        let duration = data.get("duration").and_then(Value::as_i64).unwrap_or(0);
        let expire_ts = if duration > 0 {
            now_secs() + duration as f64
        } else {
            now_secs() + 3600.0
        };
        let entry = TokenData {
            token: data
                .get("token")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            bot_id: data
                .get("bot_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            duration,
            product: data
                .get("product")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            source: data
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            expire_ts,
        };
        Self::cache()
            .lock()
            .unwrap()
            .insert(app_key.to_string(), entry.clone());
        entry
    }

    /// Get a WS auth token, using the cache when valid.
    pub fn get_token(
        app_key: &str,
        app_secret: &str,
        api_domain: &str,
        route_env: &str,
    ) -> Result<TokenData, String> {
        Self::purge_expired();

        if let Some(cached) = Self::cache().lock().unwrap().get(app_key).cloned() {
            if Self::is_cache_valid(&cached) {
                let remain = (cached.expire_ts - now_secs()) as i64;
                log::info!("Using cached token ({remain}s remaining)");
                return Ok(cached);
            }
        }

        // Double-check under the implicit lock pattern (Python uses an asyncio
        // lock; the cache Mutex protects the store).
        if let Some(cached) = Self::cache().lock().unwrap().get(app_key).cloned() {
            if Self::is_cache_valid(&cached) {
                return Ok(cached);
            }
        }

        let data = Self::fetch(app_key, app_secret, api_domain, route_env)?;
        Ok(Self::store_from_data(app_key, &data))
    }

    /// Force refresh token (clear cache and re-sign).
    pub fn force_refresh(
        app_key: &str,
        app_secret: &str,
        api_domain: &str,
        route_env: &str,
    ) -> Result<TokenData, String> {
        let tail: String = app_key.chars().rev().take(4).collect::<Vec<_>>()
            .into_iter().rev().collect();
        log::warn!("[force-refresh] Clearing cache and re-signing token: app_key=****{tail}");
        Self::cache().lock().unwrap().remove(app_key);
        let data = Self::fetch(app_key, app_secret, api_domain, route_env)?;
        Ok(Self::store_from_data(app_key, &data))
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Generate `n` random bytes rendered as lowercase hex (mirrors
/// `secrets.token_hex(n)` → 2n hex chars).
fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    if getrandom::fill(&mut buf).is_err() {
        // Fallback deterministic-ish source if the OS RNG is unavailable.
        let seed = now_secs().to_bits();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (seed.wrapping_add(i as u64) & 0xff) as u8;
        }
    }
    hex_lower(&buf)
}

// ===========================================================================
// AccessPolicy — DM / Group access control
// ===========================================================================

/// Platform-level DM / Group access control policy.
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    dm_policy: String,
    dm_allow_from: Vec<String>,
    group_policy: String,
    group_allow_from: Vec<String>,
}

impl AccessPolicy {
    pub fn new(
        dm_policy: impl Into<String>,
        dm_allow_from: Vec<String>,
        group_policy: impl Into<String>,
        group_allow_from: Vec<String>,
    ) -> Self {
        AccessPolicy {
            dm_policy: dm_policy.into(),
            dm_allow_from,
            group_policy: group_policy.into(),
            group_allow_from,
        }
    }

    /// Platform-level DM inbound filter (open / allowlist / disabled).
    pub fn is_dm_allowed(&self, sender_id: &str) -> bool {
        if self.dm_policy == "disabled" {
            return false;
        }
        if self.dm_policy == "allowlist" {
            return self
                .dm_allow_from
                .iter()
                .any(|x| x == sender_id.trim());
        }
        true
    }

    /// Platform-level group chat inbound filter (open / allowlist / disabled).
    pub fn is_group_allowed(&self, group_code: &str) -> bool {
        if self.group_policy == "disabled" {
            return false;
        }
        if self.group_policy == "allowlist" {
            return self
                .group_allow_from
                .iter()
                .any(|x| x == group_code.trim());
        }
        true
    }

    pub fn dm_policy(&self) -> &str {
        &self.dm_policy
    }

    pub fn group_policy(&self) -> &str {
        &self.group_policy
    }
}

// ===========================================================================
// Inbound decode / field-extraction helpers (DecodeMiddleware etc.)
// ===========================================================================

/// Normalized push payload (matches `parse_json_push` / `decode_inbound_push`).
#[derive(Debug, Clone, Default)]
pub struct PushPayload {
    pub callback_command: String,
    pub from_account: String,
    pub to_account: String,
    pub sender_nickname: String,
    pub group_code: String,
    pub group_name: String,
    pub msg_seq: i64,
    pub msg_id: String,
    /// Each element is `{"msg_type": str, "msg_content": object}`.
    pub msg_body: Vec<Value>,
    pub cloud_custom_data: String,
    pub bot_owner_id: String,
    pub recall_msg_seq_list: Option<Value>,
    pub trace_id: String,
}

/// Decode helpers (the JSON/protobuf push parsing of `DecodeMiddleware`).
pub struct DecodeMiddleware;

impl DecodeMiddleware {
    /// Normalize a raw JSON `msg_body` array to `[{"msg_type", "msg_content"}]`.
    pub fn convert_json_msg_body(raw_body: &Value) -> Vec<Value> {
        let mut result = Vec::new();
        let arr = match raw_body.as_array() {
            Some(a) => a,
            None => return result,
        };
        for item in arr {
            if !item.is_object() {
                continue;
            }
            let msg_type = item
                .get("msg_type")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| item.get("MsgType").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();

            let mut msg_content = item
                .get("msg_content")
                .filter(|v| !v.is_null() && !is_empty_value(v))
                .cloned()
                .or_else(|| item.get("MsgContent").cloned())
                .unwrap_or_else(|| json!({}));

            if let Some(s) = msg_content.as_str() {
                msg_content = match serde_json::from_str::<Value>(s) {
                    Ok(v) => v,
                    Err(_) => json!({ "text": s }),
                };
            }
            // Python: `msg_content or {}` — a null/falsey content becomes {}.
            let final_content = if is_empty_value(&msg_content) {
                json!({})
            } else {
                msg_content
            };
            result.push(json!({ "msg_type": msg_type, "msg_content": final_content }));
        }
        result
    }

    /// Convert a JSON-format push to a [`PushPayload`], or `None`.
    pub fn parse_json_push(raw_json: &Value) -> Option<PushPayload> {
        if raw_json.is_null() || !raw_json.is_object() {
            return None;
        }

        let s = |key: &str, alt: &str| -> String {
            let v = raw_json.get(key).and_then(Value::as_str).unwrap_or("");
            if !v.is_empty() {
                return v.to_string();
            }
            raw_json
                .get(alt)
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };

        let from_account = s("from_account", "From_Account");
        let group_code = {
            let v = raw_json.get("group_code").and_then(Value::as_str).unwrap_or("");
            if !v.is_empty() {
                v.to_string()
            } else {
                let g = raw_json.get("GroupId").and_then(Value::as_str).unwrap_or("");
                if !g.is_empty() {
                    g.to_string()
                } else {
                    raw_json
                        .get("group_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                }
            }
        };

        let msg_body_raw = {
            let v = raw_json.get("msg_body");
            match v {
                Some(b) if b.as_array().map(|a| !a.is_empty()).unwrap_or(false) => b.clone(),
                _ => raw_json.get("MsgBody").cloned().unwrap_or(Value::Null),
            }
        };
        let msg_body = Self::convert_json_msg_body(&msg_body_raw);

        let callback_command = raw_json
            .get("callback_command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        if from_account.is_empty() && msg_body.is_empty() && callback_command.is_empty() {
            return None;
        }

        let sender_nickname = s("sender_nickname", "nick_name");

        let msg_seq = {
            let v = raw_json.get("msg_seq").and_then(Value::as_i64).unwrap_or(0);
            if v != 0 {
                v
            } else {
                raw_json.get("MsgSeq").and_then(Value::as_i64).unwrap_or(0)
            }
        };

        let msg_id = {
            // explicit precedence chain matching the Python `a or b or c`
            let a = raw_json.get("msg_id").and_then(Value::as_str).unwrap_or("");
            if !a.is_empty() {
                a.to_string()
            } else {
                let b = raw_json.get("msg_key").and_then(Value::as_str).unwrap_or("");
                if !b.is_empty() {
                    b.to_string()
                } else {
                    raw_json
                        .get("MsgKey")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                }
            }
        };

        let cloud_custom_data = s("cloud_custom_data", "CloudCustomData");
        let bot_owner_id = s("bot_owner_id", "botOwnerId");

        let recall_msg_seq_list = raw_json
            .get("recall_msg_seq_list")
            .filter(|v| !v.is_null())
            .cloned();

        let trace_id = raw_json
            .get("log_ext")
            .and_then(|le| le.as_object())
            .and_then(|o| o.get("trace_id"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        Some(PushPayload {
            callback_command,
            from_account,
            to_account: s("to_account", "To_Account"),
            sender_nickname,
            group_code,
            group_name: raw_json
                .get("group_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            msg_seq,
            msg_id,
            msg_body,
            cloud_custom_data,
            bot_owner_id,
            recall_msg_seq_list,
            trace_id,
        })
    }

    /// Decode a single raw frame into `(push, decoded_via)`.
    ///
    /// `decoded_via` is `"json"` or `"protobuf"`; returns `None` when no valid
    /// message could be extracted.
    pub fn decode_single(data: &[u8]) -> Option<(PushPayload, &'static str)> {
        // Try JSON first.
        if let Ok(text) = std::str::from_utf8(data) {
            if let Ok(conn_json) = serde_json::from_str::<Value>(text) {
                if conn_json.is_object() {
                    if let Some(push) = Self::parse_json_push(&conn_json) {
                        return Some((push, "json"));
                    }
                    return None;
                }
            }
        }
        // Fall back to protobuf.
        if let Some(push) = crate::gw_yuanbao_proto::decode_inbound_push(data) {
            return Some((inbound_push_to_payload(&push), "protobuf"));
        }
        None
    }

    /// Merge a list of decoded frames into a single aggregated [`PushPayload`].
    ///
    /// Returns `(push, decoded_via)` for the first valid frame, with extra
    /// frames' `msg_body` appended (separated by a synthetic newline elem).
    pub fn merge_frames(frames: &[Vec<u8>]) -> Option<(PushPayload, &'static str)> {
        let mut merged: Option<PushPayload> = None;
        let mut decoded_via = "";

        for data in frames {
            let decoded = Self::decode_single(data);
            let (push, via) = match decoded {
                Some(p) => p,
                None => continue,
            };
            match &mut merged {
                None => {
                    merged = Some(push);
                    decoded_via = via;
                }
                Some(base) => {
                    if !push.msg_body.is_empty() {
                        let sep = json!({
                            "msg_type": "TIMTextElem",
                            "msg_content": { "text": "\n" }
                        });
                        base.msg_body.push(sep);
                        base.msg_body.extend(push.msg_body);
                    }
                }
            }
        }

        merged.map(|p| (p, decoded_via))
    }
}

fn is_empty_value(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Bool(b) => !*b,
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// Convert a decoded protobuf [`crate::gw_yuanbao_proto::InboundPush`] into the
/// platform-level [`PushPayload`] shape.
pub fn inbound_push_to_payload(push: &crate::gw_yuanbao_proto::InboundPush) -> PushPayload {
    use crate::gw_yuanbao_proto::MsgContent;

    fn content_to_json(c: &MsgContent) -> Value {
        // Mirror the Python decoder: include only non-empty/non-zero fields so
        // downstream `content.get(...)` lookups behave identically.
        let mut obj = serde_json::Map::new();
        if !c.text.is_empty() {
            obj.insert("text".into(), Value::from(c.text.clone()));
        }
        if !c.uuid.is_empty() {
            obj.insert("uuid".into(), Value::from(c.uuid.clone()));
        }
        if !c.data.is_empty() {
            obj.insert("data".into(), Value::from(c.data.clone()));
        }
        if !c.desc.is_empty() {
            obj.insert("desc".into(), Value::from(c.desc.clone()));
        }
        if !c.ext.is_empty() {
            obj.insert("ext".into(), Value::from(c.ext.clone()));
        }
        if !c.sound.is_empty() {
            obj.insert("sound".into(), Value::from(c.sound.clone()));
        }
        if !c.url.is_empty() {
            obj.insert("url".into(), Value::from(c.url.clone()));
        }
        if !c.file_name.is_empty() {
            obj.insert("file_name".into(), Value::from(c.file_name.clone()));
        }
        if c.image_format != 0 {
            obj.insert("image_format".into(), Value::from(c.image_format));
        }
        if c.index != 0 {
            obj.insert("index".into(), Value::from(c.index));
        }
        if c.file_size != 0 {
            obj.insert("file_size".into(), Value::from(c.file_size));
        }
        if !c.image_info_array.is_empty() {
            let arr: Vec<Value> = c
                .image_info_array
                .iter()
                .map(|ii| {
                    json!({
                        "type": ii.kind,
                        "size": ii.size,
                        "width": ii.width,
                        "height": ii.height,
                        "url": ii.url,
                    })
                })
                .collect();
            obj.insert("image_info_array".into(), Value::Array(arr));
        }
        Value::Object(obj)
    }

    let msg_body: Vec<Value> = push
        .msg_body
        .iter()
        .map(|e| {
            json!({
                "msg_type": e.msg_type,
                "msg_content": content_to_json(&e.msg_content),
            })
        })
        .collect();

    let recall_json = push.recall_msg_seq_list.as_ref().map(|list| {
        let arr: Vec<Value> = list
            .iter()
            .map(|s| json!({ "msg_seq": s.msg_seq, "msg_id": s.msg_id }))
            .collect();
        Value::Array(arr)
    });

    // Prefer msg_id, falling back to msg_key (matches Python field selection).
    let msg_id = if !push.msg_id.is_empty() {
        push.msg_id.clone()
    } else {
        push.msg_key.clone()
    };

    PushPayload {
        callback_command: push.callback_command.clone(),
        from_account: push.from_account.clone(),
        to_account: push.to_account.clone(),
        sender_nickname: push.sender_nickname.clone(),
        group_code: push.group_code.clone(),
        group_name: push.group_name.clone(),
        msg_seq: push.msg_seq as i64,
        msg_id,
        msg_body,
        cloud_custom_data: push.cloud_custom_data.clone(),
        bot_owner_id: push.bot_owner_id.clone(),
        recall_msg_seq_list: recall_json,
        trace_id: push.trace_id.clone(),
    }
}

// ===========================================================================
// Content extraction (ExtractContentMiddleware)
// ===========================================================================

/// An inbound media reference extracted from the msg_body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRef {
    pub kind: String, // "image" | "file"
    pub url: String,
    pub name: Option<String>,
}

/// Content-extraction helpers.
pub struct ExtractContentMiddleware;

impl ExtractContentMiddleware {
    pub const CARD_CONTENT_MAX_LENGTH: usize = 1000;

    /// Format elem_type 1010 (share card) into bracket-placeholder text.
    pub fn format_shared_link(custom: &Value) -> String {
        let title = custom.get("title").and_then(Value::as_str).unwrap_or("");
        let link = custom.get("link").and_then(Value::as_str).unwrap_or("");
        let header = if !link.is_empty() {
            format!("[share_card: {title} | {link}]")
        } else {
            format!("[share_card: {title}]")
        };
        let mut lines = vec![header];
        let max_len = Self::CARD_CONTENT_MAX_LENGTH;
        for field in ["card_content", "wechat_des"] {
            if let Some(val) = custom.get(field).and_then(Value::as_str) {
                if !val.is_empty() {
                    let preview = if val.chars().count() > max_len {
                        let truncated: String = val.chars().take(max_len).collect();
                        format!("{truncated}...(truncated)")
                    } else {
                        val.to_string()
                    };
                    lines.push(format!("Preview: {preview}"));
                    break;
                }
            }
        }
        if !link.is_empty() {
            lines.push("[visit link for full content]".to_string());
        }
        lines.join("\n")
    }

    /// Format elem_type 1007 (link understanding card) into placeholder text.
    pub fn format_link_understanding(custom: &Value) -> Option<String> {
        let content = custom.get("content")?;
        let content_str = content.as_str()?;
        if content_str.is_empty() {
            return None;
        }
        let parsed: Value = serde_json::from_str(content_str).ok()?;
        let link = parsed.get("link").and_then(Value::as_str)?;
        if link.is_empty() {
            return None;
        }
        Some(format!("[link: {link} | visit link for full content]"))
    }

    /// Extract plain text content from a msg_body array.
    pub fn extract_text(msg_body: &[Value]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for elem in msg_body {
            let elem_type = elem.get("msg_type").and_then(Value::as_str).unwrap_or("");
            let content = elem
                .get("msg_content")
                .cloned()
                .unwrap_or_else(|| json!({}));

            match elem_type {
                "TIMTextElem" => {
                    let text = content.get("text").and_then(Value::as_str).unwrap_or("");
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
                "TIMImageElem" => parts.push("[image]".to_string()),
                "TIMFileElem" => {
                    let filename = content
                        .get("file_name")
                        .and_then(Value::as_str)
                        .or_else(|| content.get("fileName").and_then(Value::as_str))
                        .or_else(|| content.get("filename").and_then(Value::as_str))
                        .unwrap_or("");
                    if !filename.is_empty() {
                        parts.push(format!("[file: {filename}]"));
                    } else {
                        parts.push("[file]".to_string());
                    }
                }
                "TIMSoundElem" => parts.push("[voice]".to_string()),
                "TIMVideoFileElem" => parts.push("[video]".to_string()),
                "TIMCustomElem" => {
                    let data_val = content.get("data").and_then(Value::as_str).unwrap_or("");
                    if !data_val.is_empty() {
                        match serde_json::from_str::<Value>(data_val) {
                            Ok(custom) if custom.is_object() => {
                                let ctype = custom.get("elem_type").and_then(Value::as_i64);
                                match ctype {
                                    Some(1002) => {
                                        let t = custom
                                            .get("text")
                                            .and_then(Value::as_str)
                                            .unwrap_or("[mention]");
                                        parts.push(t.to_string());
                                    }
                                    Some(1010) => {
                                        parts.push(Self::format_shared_link(&custom));
                                    }
                                    Some(1007) => {
                                        match Self::format_link_understanding(&custom) {
                                            Some(t) => parts.push(t),
                                            None => parts
                                                .push("[unsupported message type]".to_string()),
                                        }
                                    }
                                    _ => parts.push("[unsupported message type]".to_string()),
                                }
                            }
                            Ok(_non_obj) => {
                                parts.push("[unsupported message type]".to_string());
                            }
                            Err(_) => {
                                parts.push(data_val.to_string());
                            }
                        }
                    } else {
                        parts.push("[unsupported message type]".to_string());
                    }
                }
                "TIMFaceElem" => {
                    let raw_data = content.get("data").and_then(Value::as_str).unwrap_or("");
                    let mut face_name = String::new();
                    if !raw_data.is_empty() {
                        if let Ok(face_data) = serde_json::from_str::<Value>(raw_data) {
                            face_name = face_data
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .trim()
                                .to_string();
                        }
                    }
                    if !face_name.is_empty() {
                        parts.push(format!("[emoji: {face_name}]"));
                    } else {
                        parts.push("[emoji]".to_string());
                    }
                }
                "" => {}
                other => {
                    parts.push(format!("[{other}]"));
                }
            }
        }
        parts.join(" ")
    }

    /// Normalize input: strip whitespace and convert a leading full-width
    /// slash (`／`) to ASCII `/`.
    pub fn rewrite_slash_command(text: &str) -> String {
        let trimmed = text.trim();
        if let Some(rest) = trimmed.strip_prefix('\u{ff0f}') {
            format!("/{rest}")
        } else {
            trimmed.to_string()
        }
    }

    /// Extract inbound image/file references from a TIM msg_body.
    pub fn extract_inbound_media_refs(msg_body: &[Value]) -> Vec<MediaRef> {
        let mut refs = Vec::new();
        for elem in msg_body {
            if !elem.is_object() {
                continue;
            }
            let msg_type = elem.get("msg_type").and_then(Value::as_str).unwrap_or("");
            let content = match elem.get("msg_content") {
                Some(c) if c.is_object() => c,
                _ => continue,
            };

            if msg_type == "TIMImageElem" {
                let image_info_array = content
                    .get("image_info_array")
                    .and_then(Value::as_array);
                let image_info: Option<&Value> = match image_info_array {
                    Some(arr) if arr.len() > 1 && arr[1].is_object() => Some(&arr[1]),
                    Some(arr) if !arr.is_empty() && arr[0].is_object() => Some(&arr[0]),
                    _ => None,
                };
                let image_url = image_info
                    .and_then(|i| i.get("url"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !image_url.is_empty() {
                    refs.push(MediaRef {
                        kind: "image".to_string(),
                        url: image_url,
                        name: None,
                    });
                }
                continue;
            }

            if msg_type == "TIMFileElem" {
                let file_url = content.get("url").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let file_name = content
                    .get("file_name")
                    .and_then(Value::as_str)
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .or_else(|| {
                        content.get("fileName").and_then(Value::as_str).map(|s| s.trim()).filter(|s| !s.is_empty())
                    })
                    .or_else(|| {
                        content.get("filename").and_then(Value::as_str).map(|s| s.trim()).filter(|s| !s.is_empty())
                    })
                    .map(|s| s.to_string());
                if !file_url.is_empty() {
                    refs.push(MediaRef {
                        kind: "file".to_string(),
                        url: file_url,
                        name: file_name,
                    });
                }
            }
        }
        refs
    }

    /// Extract link URLs from share-card (1010) and link-understanding (1007).
    pub fn extract_link_urls(msg_body: &[Value]) -> Vec<String> {
        let mut urls = Vec::new();
        for elem in msg_body {
            if !elem.is_object()
                || elem.get("msg_type").and_then(Value::as_str) != Some("TIMCustomElem")
            {
                continue;
            }
            let data_str = elem
                .get("msg_content")
                .and_then(|c| c.get("data"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if data_str.is_empty() {
                continue;
            }
            let custom: Value = match serde_json::from_str(data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if !custom.is_object() {
                continue;
            }
            match custom.get("elem_type").and_then(Value::as_i64) {
                Some(1010) => {
                    if let Some(link) = custom.get("link").and_then(Value::as_str) {
                        if !link.is_empty() {
                            urls.push(link.to_string());
                        }
                    }
                }
                Some(1007) => {
                    if let Some(content) = custom.get("content").and_then(Value::as_str) {
                        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
                            if let Some(link) = parsed.get("link").and_then(Value::as_str) {
                                if !link.is_empty() {
                                    urls.push(link.to_string());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        urls
    }
}

// ===========================================================================
// PlaceholderFilterMiddleware
// ===========================================================================

/// Pure-placeholder filter (e.g. "[image]" with no media).
pub struct PlaceholderFilter;

impl PlaceholderFilter {
    pub const SKIPPABLE: &'static [&'static str] = &[
        "[image]", "[图片]", "[file]", "[文件]", "[video]", "[视频]", "[voice]", "[语音]",
    ];

    /// Detect whether the message is a pure placeholder (should be skipped).
    pub fn is_skippable_placeholder(text: &str, media_count: usize) -> bool {
        if media_count > 0 {
            return false;
        }
        let stripped = text.trim();
        Self::SKIPPABLE.contains(&stripped)
    }
}

// ===========================================================================
// OwnerCommandMiddleware
// ===========================================================================

/// Owner slash-command detection.
pub struct OwnerCommand;

impl OwnerCommand {
    pub const ALLOWLIST: &'static [&'static str] = &[
        "/new", "/reset", "/retry", "/undo", "/stop", "/approve", "/deny", "/background", "/bg",
        "/btw", "/queue", "/q",
    ];

    /// Normalize a leading full-width slash and strip whitespace.
    pub fn rewrite_slash_command(text: &str) -> String {
        ExtractContentMiddleware::rewrite_slash_command(text)
    }

    /// Identify allowlisted slash commands and determine sender identity.
    ///
    /// Returns `(cmd, cmd_line, is_owner)`:
    ///   - `(None, None, false)`: not an allowlisted command
    ///   - `(Some(cmd), Some(line), true)`: owner match
    ///   - `(Some(cmd), Some(line), false)`: allowlisted but sender not owner
    pub fn detect_owner_command(
        push: &PushPayload,
        msg_body: &[Value],
        chat_type: &str,
        from_account: &str,
    ) -> (Option<String>, Option<String>, bool) {
        if chat_type != "group" {
            return (None, None, false);
        }

        let text_elems: Vec<&Value> = msg_body
            .iter()
            .filter(|e| e.get("msg_type").and_then(Value::as_str) == Some("TIMTextElem"))
            .collect();
        if text_elems.len() != 1 {
            return (None, None, false);
        }

        let text = text_elems[0]
            .get("msg_content")
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let cmd_line = Self::rewrite_slash_command(text);
        if !cmd_line.starts_with('/') {
            return (None, None, false);
        }
        let cmd = cmd_line
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        if !Self::ALLOWLIST.contains(&cmd.as_str()) {
            return (None, None, false);
        }

        let owner_id = push.bot_owner_id.trim();
        let is_owner = !owner_id.is_empty() && owner_id == from_account;
        (Some(cmd), Some(cmd_line), is_owner)
    }
}

// ===========================================================================
// ChatRouting / SkipSelf / classification helpers
// ===========================================================================

/// Routing result for an inbound message.
#[derive(Debug, Clone, Default)]
pub struct ChatRoute {
    pub chat_id: String,
    pub chat_type: String,
    pub chat_name: String,
}

/// Derive `chat_id`, `chat_type`, `chat_name` from push fields.
pub fn route_chat(
    group_code: &str,
    group_name: &str,
    from_account: &str,
    sender_nickname: &str,
) -> ChatRoute {
    if !group_code.is_empty() {
        ChatRoute {
            chat_id: format!("group:{group_code}"),
            chat_type: "group".to_string(),
            chat_name: if group_name.is_empty() {
                group_code.to_string()
            } else {
                group_name.to_string()
            },
        }
    } else {
        ChatRoute {
            chat_id: format!("direct:{from_account}"),
            chat_type: "dm".to_string(),
            chat_name: if sender_nickname.is_empty() {
                from_account.to_string()
            } else {
                sender_nickname.to_string()
            },
        }
    }
}

/// Detect whether the message is from the bot itself.
pub fn is_self_reference(from_account: &str, bot_id: Option<&str>) -> bool {
    match bot_id {
        Some(b) if !from_account.is_empty() && !b.is_empty() => from_account == b,
        _ => false,
    }
}

/// Message classification mirroring `gateway::MessageType`.
pub fn classify_message_type(text: &str, msg_body: &[Value]) -> crate::gateway::MessageType {
    use crate::gateway::MessageType;
    if text.starts_with('/') {
        return MessageType::Command;
    }
    for elem in msg_body {
        match elem.get("msg_type").and_then(Value::as_str) {
            Some("TIMImageElem") => return MessageType::Photo,
            Some("TIMSoundElem") => return MessageType::Voice,
            Some("TIMVideoFileElem") => return MessageType::Video,
            Some("TIMFileElem") => return MessageType::Document,
            _ => {}
        }
    }
    MessageType::Text
}

// ===========================================================================
// GroupAtGuard / GroupAttribution helpers
// ===========================================================================

/// Group-chat @bot detection and attribution helpers.
pub struct GroupAtGuard;

impl GroupAtGuard {
    /// Detect whether the message @-mentions the bot.
    pub fn is_at_bot(msg_body: &[Value], bot_id: Option<&str>) -> bool {
        let bot_id = match bot_id {
            Some(b) if !b.is_empty() => b,
            _ => return false,
        };
        for elem in msg_body {
            if elem.get("msg_type").and_then(Value::as_str) != Some("TIMCustomElem") {
                continue;
            }
            let data_str = elem
                .get("msg_content")
                .and_then(|c| c.get("data"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if data_str.is_empty() {
                continue;
            }
            if let Ok(custom) = serde_json::from_str::<Value>(data_str) {
                if custom.get("elem_type").and_then(Value::as_i64) == Some(1002)
                    && custom.get("user_id").and_then(Value::as_str) == Some(bot_id)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Extract the display text used to @-mention this bot (e.g. `@yuanbao-bot`).
    pub fn extract_bot_mention_text(msg_body: &[Value], bot_id: Option<&str>) -> String {
        let bot_id = match bot_id {
            Some(b) if !b.is_empty() => b,
            _ => return String::new(),
        };
        for elem in msg_body {
            if elem.get("msg_type").and_then(Value::as_str) != Some("TIMCustomElem") {
                continue;
            }
            let data_str = elem
                .get("msg_content")
                .and_then(|c| c.get("data"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if data_str.is_empty() {
                continue;
            }
            if let Ok(custom) = serde_json::from_str::<Value>(data_str) {
                if custom.get("elem_type").and_then(Value::as_i64) == Some(1002)
                    && custom.get("user_id").and_then(Value::as_str) == Some(bot_id)
                {
                    let mention_text = custom
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !mention_text.is_empty() {
                        return mention_text;
                    }
                }
            }
        }
        String::new()
    }

    /// Build a per-turn group-chat prompt highlighting which message to answer.
    pub fn build_group_channel_prompt(msg_body: &[Value], bot_id: Option<&str>) -> String {
        let bid = bot_id.filter(|b| !b.is_empty()).unwrap_or("unknown");
        let mention = Self::extract_bot_mention_text(msg_body, bot_id);
        let bot_mention = if mention.is_empty() { "unknown".to_string() } else { mention };
        format!(
            "You are handling a Yuanbao group chat message.\n\
             - Your identity: user_id={bid}, @-mention name in this group={bot_mention}\n\
             - Lines in history prefixed with `[nickname|user_id]` are observed group context \
             and are not necessarily addressed to you.\n\
             - Treat only the current new message as a request explicitly directed at you, \
             and answer it directly."
        )
    }
}

/// Build the `[nickname|user_id]\n<content>` attribution for a group message.
pub fn build_group_attribution(sender_display: &str, user_id: &str, text: &str) -> String {
    format!("[{sender_display}|{user_id}]\n{text}")
}

// ===========================================================================
// QuoteContextMiddleware
// ===========================================================================

/// Extract quote/reply context from `cloud_custom_data`.
///
/// Returns `(reply_to_message_id, reply_to_text)`.
pub fn extract_quote_context(cloud_custom_data: &str) -> (Option<String>, Option<String>) {
    if cloud_custom_data.is_empty() {
        return (None, None);
    }
    let parsed: Value = match serde_json::from_str(cloud_custom_data) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let quote = match parsed.get("quote") {
        Some(q) if q.is_object() => q,
        _ => return (None, None),
    };

    let quote_type = quote.get("type").and_then(value_as_int).unwrap_or(0);
    let mut desc = quote
        .get("desc")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if quote_type == 2 && desc.is_empty() {
        desc = "[image]".to_string();
    }
    if desc.is_empty() {
        return (None, None);
    }

    let quote_id = quote
        .get("id")
        .and_then(Value::as_str)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let sender = quote
        .get("sender_nickname")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| quote.get("sender_id").and_then(Value::as_str))
        .unwrap_or("")
        .trim()
        .to_string();

    let quote_text = if sender.is_empty() {
        desc
    } else {
        format!("{sender}: {desc}")
    };
    (quote_id, Some(quote_text))
}

fn value_as_int(v: &Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        Some(i)
    } else if let Some(f) = v.as_f64() {
        Some(f as i64)
    } else if let Some(s) = v.as_str() {
        s.trim().parse::<i64>().ok()
    } else {
        None
    }
}

// ===========================================================================
// MediaResolveMiddleware — resource resolution helpers
// ===========================================================================

/// Media-resolution helpers (resource-id → direct download URL etc.).
pub struct MediaResolve;

impl MediaResolve {
    /// Guess an image extension from a URL path.
    pub fn guess_image_ext_from_url(url: &str) -> String {
        let path = url::Url::parse(url)
            .map(|u| u.path().to_string())
            .unwrap_or_else(|_| url.to_string());
        let ext = match path.rfind('.') {
            Some(idx) => path[idx..].to_ascii_lowercase(),
            None => String::new(),
        };
        const ALLOWED: &[&str] = &[
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff",
        ];
        if ALLOWED.contains(&ext.as_str()) {
            ext
        } else {
            ".jpg".to_string()
        }
    }

    /// Parse a Yuanbao resource-placeholder URL into a `resourceId`, if any.
    pub fn extract_resource_id(url: &str) -> Option<String> {
        let parsed = url::Url::parse(url).ok()?;
        for (k, v) in parsed.query_pairs() {
            if k.eq_ignore_ascii_case("resourceid") {
                let id = v.trim().to_string();
                if !id.is_empty() {
                    return Some(id);
                }
            }
        }
        None
    }

    /// Exchange a `resourceId` for a direct download URL via the business API.
    ///
    /// Performs a single 401-retry with token force-refresh. Returns the
    /// resolved URL, or an error string.
    pub fn fetch_resource_url(
        api_domain: &str,
        app_key: &str,
        app_secret: &str,
        bot_id: &str,
        route_env: &str,
        token_data: &TokenData,
        resource_id: &str,
    ) -> Result<String, String> {
        let resource_id = resource_id.trim();
        if resource_id.is_empty() {
            return Err("missing resource_id".to_string());
        }

        let mut token = token_data.token.trim().to_string();
        let mut source = {
            let s = token_data.source.trim();
            if s.is_empty() { "web".to_string() } else { s.to_string() }
        };
        let mut id = {
            let b = token_data.bot_id.trim();
            if !b.is_empty() {
                b.to_string()
            } else if !bot_id.is_empty() {
                bot_id.to_string()
            } else {
                app_key.to_string()
            }
        };
        if token.is_empty() || id.is_empty() {
            return Err("missing token or bot_id for resource download".to_string());
        }

        let api_url = format!("{api_domain}/api/resource/v1/download");
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| format!("http client build failed: {e}"))?;

        for attempt in 0..2 {
            let resp = client
                .get(&api_url)
                .query(&[("resourceId", resource_id)])
                .header("Content-Type", "application/json")
                .header("X-ID", id.clone())
                .header("X-Token", token.clone())
                .header("X-Source", source.clone())
                .send()
                .map_err(|e| format!("resource/v1/download request failed: {e}"))?;

            if resp.status().as_u16() == 401 && attempt == 0 {
                let refreshed =
                    SignManager::force_refresh(app_key, app_secret, api_domain, route_env)?;
                token = refreshed.token.trim().to_string();
                source = {
                    let s = refreshed.source.trim();
                    if !s.is_empty() {
                        s.to_string()
                    } else if !source.is_empty() {
                        source.clone()
                    } else {
                        "web".to_string()
                    }
                };
                id = {
                    let b = refreshed.bot_id.trim();
                    if !b.is_empty() {
                        b.to_string()
                    } else if !bot_id.is_empty() {
                        bot_id.to_string()
                    } else {
                        app_key.to_string()
                    }
                };
                if token.is_empty() || id.is_empty() {
                    break;
                }
                continue;
            }

            let status = resp.status();
            if !status.is_success() {
                return Err(format!("resource/v1/download HTTP error: {status}"));
            }
            let payload: Value = resp
                .json()
                .map_err(|e| format!("resource/v1/download parse error: {e}"))?;

            let code = payload.get("code").and_then(Value::as_i64);
            if !matches!(code, None | Some(0)) {
                let msg = payload.get("msg").and_then(Value::as_str).unwrap_or("");
                return Err(format!("resource/v1/download failed: code={code:?}, msg={msg}"));
            }
            let data = match payload.get("data") {
                Some(d) if d.is_object() => d.clone(),
                _ => payload.clone(),
            };
            let real_url = data
                .get("url")
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .or_else(|| data.get("realUrl").and_then(Value::as_str))
                .unwrap_or("")
                .trim()
                .to_string();
            if !real_url.is_empty() {
                return Ok(real_url);
            }
            return Err("resource/v1/download missing url/realUrl".to_string());
        }

        Err("resource/v1/download did not return a URL".to_string())
    }
}

// ===========================================================================
// MessageSender — text truncation / cron stripping / msg_body builders
// ===========================================================================

/// Outbound message-construction helpers (the deterministic parts of
/// `MessageSender`). The async dispatch/lock surfaces are represented by
/// [`YuanbaoAdapter`] state; encoding is delegated to `gw_yuanbao_proto`.
pub struct MessageSender;

impl MessageSender {
    pub const IMAGE_EXTS: &'static [&'static str] =
        &[".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"];
    pub const CHAT_DICT_MAX_SIZE: usize = 1000;

    /// Media pre-validation. Returns an error description if invalid, else `None`.
    pub fn validate_media(
        file_bytes: Option<&[u8]>,
        filename: &str,
        max_size_mb: u64,
    ) -> Option<String> {
        match file_bytes {
            None => return Some(format!("Empty file: {filename}")),
            Some(b) if b.is_empty() => return Some(format!("Empty file: {filename}")),
            Some(b) => {
                let max_bytes = (max_size_mb * 1024 * 1024) as usize;
                if b.len() > max_bytes {
                    let size_mb = b.len() as f64 / 1024.0 / 1024.0;
                    return Some(format!(
                        "File too large: {filename} ({size_mb:.1}MB > {max_size_mb}MB)"
                    ));
                }
            }
        }
        None
    }

    /// Split a long message into chunks with table-awareness.
    ///
    /// Delegates core splitting to [`MarkdownProcessor::chunk_markdown_text`]
    /// and strips page indicators like `(1/3)` from the output.
    pub fn truncate_message(content: &str, max_length: usize) -> Vec<String> {
        if content.chars().count() <= max_length {
            return vec![content.to_string()];
        }
        let chunks = MarkdownProcessor::chunk_markdown_text(content, max_length);
        let stripped: Vec<String> = chunks.iter().map(|c| strip_indicator(c)).collect();
        if stripped.is_empty() {
            vec![content.to_string()]
        } else {
            stripped
        }
    }

    /// Strip the scheduler cron header/footer wrapper for cleaner output.
    pub fn strip_cron_wrapper(content: &str) -> String {
        if !content.starts_with("Cronjob Response: ") {
            return content.to_string();
        }
        let divider = "\n-------------\n\n";
        let footer_prefix =
            "\n\nTo stop or manage this job, send me a new message (e.g. \"stop reminder ";
        let divider_pos = content.find(divider);
        let footer_pos = content.rfind(footer_prefix);
        let (divider_pos, footer_pos) = match (divider_pos, footer_pos) {
            (Some(d), Some(f)) if f > d => (d, f),
            _ => return content.to_string(),
        };
        let header = &content[..divider_pos];
        if !header.contains("\n(job_id: ") {
            return content.to_string();
        }
        let body_start = divider_pos + divider.len();
        let body = content[body_start..footer_pos].trim();
        if body.is_empty() {
            content.to_string()
        } else {
            body.to_string()
        }
    }

    /// Build a C2C text msg_body for `send_c2c_message`.
    pub fn text_msg_body(text: &str) -> Vec<Value> {
        vec![json!({ "msg_type": "TIMTextElem", "msg_content": { "text": text } })]
    }

    /// Parse `@nickname` patterns into a mixed TIMTextElem + TIMCustomElem
    /// msg_body using the supplied member list (group `@mention` resolution).
    ///
    /// `members` is the cached member list for the group (each a JSON object
    /// with `nickname`/`nick_name` and `user_id`). When no members are
    /// available a single plain text elem is returned.
    pub fn build_msg_body_with_mentions(text: &str, members: &[Value]) -> Vec<Value> {
        if members.is_empty() {
            return Self::text_msg_body(text);
        }

        // nickname(lower) -> (real_nick, uid)
        let mut nickname_to_uid: HashMap<String, (String, String)> = HashMap::new();
        for m in members {
            let nick = m
                .get("nickname")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| m.get("nick_name").and_then(Value::as_str))
                .unwrap_or("");
            let uid = m.get("user_id").and_then(Value::as_str).unwrap_or("");
            if !nick.is_empty() && !uid.is_empty() {
                nickname_to_uid
                    .insert(nick.to_lowercase(), (nick.to_string(), uid.to_string()));
            }
        }

        let matches = find_at_mentions(text);
        let mut msg_body: Vec<Value> = Vec::new();
        let mut last_idx = 0usize;

        for (start, end, nickname) in &matches {
            if *start > last_idx {
                let seg = text[last_idx..*start].trim();
                if !seg.is_empty() {
                    msg_body.push(json!({
                        "msg_type": "TIMTextElem",
                        "msg_content": { "text": seg }
                    }));
                }
            }
            match nickname_to_uid.get(&nickname.to_lowercase()) {
                Some((real_nick, uid)) => {
                    let data = json!({
                        "elem_type": 1002,
                        "text": format!("@{real_nick}"),
                        "user_id": uid,
                    });
                    msg_body.push(json!({
                        "msg_type": "TIMCustomElem",
                        "msg_content": { "data": data.to_string() }
                    }));
                }
                None => {
                    msg_body.push(json!({
                        "msg_type": "TIMTextElem",
                        "msg_content": { "text": format!("@{nickname}") }
                    }));
                }
            }
            last_idx = *end;
        }

        if last_idx < text.len() {
            let tail = text[last_idx..].trim();
            if !tail.is_empty() {
                msg_body.push(json!({
                    "msg_type": "TIMTextElem",
                    "msg_content": { "text": tail }
                }));
            }
        }

        if msg_body.is_empty() {
            msg_body.push(json!({
                "msg_type": "TIMTextElem",
                "msg_content": { "text": text }
            }));
        }

        msg_body
    }
}

/// Strip a trailing page indicator like ` (1/3)` from a chunk.
fn strip_indicator(text: &str) -> String {
    // regex: \s*\(\d+/\d+\)$
    let trimmed_end = text.trim_end_matches(|c: char| c.is_whitespace());
    if let Some(open) = trimmed_end.rfind('(') {
        let candidate = &trimmed_end[open..];
        if candidate.ends_with(')') {
            let inner = &candidate[1..candidate.len() - 1];
            if let Some((a, b)) = inner.split_once('/') {
                if !a.is_empty()
                    && !b.is_empty()
                    && a.chars().all(|c| c.is_ascii_digit())
                    && b.chars().all(|c| c.is_ascii_digit())
                {
                    // Strip from `open`, plus any leading whitespace before it.
                    let before = &text[..open];
                    return before.trim_end().to_string();
                }
            }
        }
    }
    text.to_string()
}

/// Find `@nickname` mentions: `(?:(?<=\s)|(?<=^))@(\S+?)(?=\s|$)` (MULTILINE).
///
/// Returns `(start_byte, end_byte, nickname)` for each match. The `@` must be
/// preceded by whitespace or be at the start of a line, and the nickname runs
/// (non-greedily) until whitespace or end-of-line.
fn find_at_mentions(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;

    while i < n {
        // Find next '@'.
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        // Check the lookbehind: start-of-string, start-of-line (after \n in
        // MULTILINE mode `^` matches after newline), or preceded by whitespace.
        let preceded_ok = if i == 0 {
            true
        } else {
            let prev = bytes[i - 1];
            prev == b'\n' || (prev as char).is_whitespace()
        };
        if !preceded_ok {
            i += 1;
            continue;
        }
        // Match \S+? non-greedily until whitespace or end-of-line/string.
        let name_start = i + 1;
        let mut j = name_start;
        while j < n {
            let c = bytes[j] as char;
            if c.is_whitespace() {
                break;
            }
            j += 1;
        }
        if j == name_start {
            // `@` followed immediately by whitespace/EOL → \S+ requires >=1 char
            i += 1;
            continue;
        }
        // The lookahead `(?=\s|$)` is satisfied because we stopped at whitespace
        // or end-of-string. Nickname is text[name_start..j].
        let nickname = text[name_start..j].to_string();
        out.push((i, j, nickname));
        i = j;
    }

    out
}

// ===========================================================================
// YuanbaoAdapter — configuration + adapter state
// ===========================================================================

/// Yuanbao AI Bot adapter configuration & state.
///
/// This mirrors `YuanbaoAdapter.__init__`: it parses credentials/endpoints
/// from the platform config `extra` map (with env-var fallbacks for the access
/// policy and auto-sethome), and holds the deduplicator, member cache and
/// per-session bookkeeping.
pub struct YuanbaoAdapter {
    pub app_key: String,
    pub app_secret: String,
    pub bot_id: Option<String>,
    pub ws_url: String,
    pub api_domain: String,
    pub route_env: String,

    pub access_policy: AccessPolicy,

    /// group_code -> (updated_ts, member list)
    pub member_cache: HashMap<String, (f64, Vec<Value>)>,
    pub member_cache_ttl_s: f64,

    pub dedup: crate::gw_helpers::MessageDeduplicator,

    /// session_key -> msg_id currently being processed.
    pub processing_msg_ids: HashMap<String, String>,
    pub processing_msg_texts: HashMap<String, String>,
    /// Bounded cache of msg_id -> attributed content.
    pub msg_content_cache: HashMap<String, String>,

    pub auto_sethome_done: bool,
    pub running: bool,
}

impl YuanbaoAdapter {
    pub const MAX_TEXT_CHUNK: usize = 4000;
    pub const MEDIA_MAX_SIZE_MB: u64 = 50;
    pub const REPLY_REF_MAX_ENTRIES: usize = 500;
    pub const DM_MAX_CHARS: usize = 10000;

    /// Build an adapter from the platform config `extra` map plus the existing
    /// configured home channel (used for auto-sethome bookkeeping).
    ///
    /// `extra` is the platform's `extra` object; `home_channel_chat_id` is the
    /// configured home channel chat id (or empty).
    pub fn new(extra: &Value, home_channel_chat_id: &str) -> Self {
        let get_str = |key: &str| -> String {
            extra
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };

        let app_key = get_str("app_id").trim().to_string();
        let app_secret = get_str("app_secret").trim().to_string();
        let bot_id = {
            let b = get_str("bot_id");
            if b.is_empty() { None } else { Some(b) }
        };
        let ws_url = {
            let u = get_str("ws_url");
            if u.is_empty() {
                DEFAULT_WS_GATEWAY_URL.to_string()
            } else {
                u.trim().to_string()
            }
        };
        let api_domain = {
            let d = get_str("api_domain");
            let base = if d.is_empty() {
                DEFAULT_API_DOMAIN.to_string()
            } else {
                d
            };
            base.trim_end_matches('/').to_string()
        };
        let route_env = get_str("route_env").trim().to_string();

        // -- Access control policy --
        let dm_policy = {
            let v = get_str("dm_policy");
            let v = if v.is_empty() {
                std::env::var("YUANBAO_DM_POLICY").unwrap_or_else(|_| "open".to_string())
            } else {
                v
            };
            v.trim().to_lowercase()
        };
        let dm_allow_from = {
            let raw = {
                let v = get_str("dm_allow_from");
                if v.is_empty() {
                    std::env::var("YUANBAO_DM_ALLOW_FROM").unwrap_or_default()
                } else {
                    v
                }
            };
            split_csv(&raw)
        };
        let group_policy = {
            let v = get_str("group_policy");
            let v = if v.is_empty() {
                std::env::var("YUANBAO_GROUP_POLICY").unwrap_or_else(|_| "open".to_string())
            } else {
                v
            };
            v.trim().to_lowercase()
        };
        let group_allow_from = {
            let raw = {
                let v = get_str("group_allow_from");
                if v.is_empty() {
                    std::env::var("YUANBAO_GROUP_ALLOW_FROM").unwrap_or_default()
                } else {
                    v
                }
            };
            split_csv(&raw)
        };

        let access_policy =
            AccessPolicy::new(dm_policy, dm_allow_from, group_policy, group_allow_from);

        // -- Auto-sethome bookkeeping --
        let existing_home = {
            let env_home = std::env::var("YUANBAO_HOME_CHANNEL").unwrap_or_default();
            if !env_home.is_empty() {
                env_home
            } else {
                home_channel_chat_id.to_string()
            }
        };
        let auto_sethome_done =
            !existing_home.is_empty() && !existing_home.starts_with("group:");

        YuanbaoAdapter {
            app_key,
            app_secret,
            bot_id,
            ws_url,
            api_domain,
            route_env,
            access_policy,
            member_cache: HashMap::new(),
            member_cache_ttl_s: 300.0,
            dedup: crate::gw_helpers::MessageDeduplicator::with_config(2000, 300.0),
            processing_msg_ids: HashMap::new(),
            processing_msg_texts: HashMap::new(),
            msg_content_cache: HashMap::new(),
            auto_sethome_done,
            running: false,
        }
    }

    /// Get the current valid sign token (using the module-level cache).
    pub fn get_cached_token(&self) -> Result<TokenData, String> {
        SignManager::get_token(&self.app_key, &self.app_secret, &self.api_domain, &self.route_env)
    }

    /// Return basic chat metadata derived from the chat_id prefix.
    pub fn get_chat_info(&self, chat_id: &str) -> Value {
        if chat_id.starts_with("group:") {
            json!({ "name": chat_id, "type": "group" })
        } else {
            json!({ "name": chat_id, "type": "dm" })
        }
    }

    /// Return a snapshot of the connection status fields (the non-runtime parts).
    pub fn get_status(&self, connected: bool, connect_id: Option<&str>, reconnect_attempts: u32) -> Value {
        json!({
            "connected": connected,
            "bot_id": self.bot_id,
            "connect_id": connect_id,
            "reconnect_attempts": reconnect_attempts,
            "ws_url": self.ws_url,
        })
    }

    /// Cache the most-recent member list for a group (used by @mention).
    pub fn cache_members(&mut self, group_code: &str, members: Vec<Value>) {
        if !members.is_empty() {
            self.member_cache
                .insert(group_code.to_string(), (now_secs(), members));
        }
    }

    /// Return the cached member list for a group if still fresh, else `&[]`.
    pub fn fresh_members(&self, group_code: &str) -> Vec<Value> {
        if let Some((ts, members)) = self.member_cache.get(group_code) {
            if now_secs() - ts < self.member_cache_ttl_s {
                return members.clone();
            }
        }
        Vec::new()
    }

    /// Bounded insert into the msg_id → content cache (cap 200, drop oldest).
    pub fn cache_msg_content(&mut self, msg_id: &str, content: &str) {
        if msg_id.is_empty() || content.is_empty() {
            return;
        }
        self.msg_content_cache
            .insert(msg_id.to_string(), content.to_string());
        if self.msg_content_cache.len() > 200 {
            // Python deletes the oldest insertion-order keys; HashMap has no
            // order, so trim arbitrary surplus to keep the cap.
            let surplus = self.msg_content_cache.len() - 200;
            let keys: Vec<String> = self
                .msg_content_cache
                .keys()
                .take(surplus)
                .cloned()
                .collect();
            for k in keys {
                self.msg_content_cache.remove(&k);
            }
        }
    }
}

/// Split a comma-separated string into trimmed non-empty parts.
fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect()
}

// ===========================================================================
// GroupQueryService — AI-tool-facing wrappers (filtering / mention hint)
// ===========================================================================

/// Group query result filtering (the deterministic parts of
/// `GroupQueryService.query_session_members`).
pub struct GroupQueryService;

impl GroupQueryService {
    /// Filter + summarize a member list for the `query_session_members` tool.
    ///
    /// `action` is `"find"` | `"list_bots"` | `"list_all"`; `name` is the
    /// search keyword used when `action == "find"`.
    pub fn filter_members(members: &[Value], action: &str, name: Option<&str>) -> Value {
        let mut filtered: Vec<Value> = members.to_vec();

        if action == "find" {
            if let Some(name) = name {
                let query = name.to_lowercase();
                filtered.retain(|m| {
                    let nick = m
                        .get("nickname")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_lowercase();
                    let card = m
                        .get("name_card")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_lowercase();
                    let uid = m
                        .get("user_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_lowercase();
                    nick.contains(&query) || card.contains(&query) || uid.contains(&query)
                });
            }
        } else if action == "list_bots" {
            filtered.retain(|m| {
                m.get("nickname")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_lowercase()
                    .contains("bot")
            });
        }

        let total = filtered.len();
        let mut mention_hint = String::new();
        if !filtered.is_empty() && filtered.len() <= 10 {
            let names: Vec<String> = filtered
                .iter()
                .map(|m| {
                    m.get("name_card")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .or_else(|| m.get("nickname").and_then(Value::as_str))
                        .filter(|s| !s.is_empty())
                        .or_else(|| m.get("user_id").and_then(Value::as_str))
                        .unwrap_or("")
                        .to_string()
                })
                .collect();
            mention_hint = format!("Mention with @name: {}", names.join(", "));
        }

        let limited: Vec<Value> = filtered.into_iter().take(50).collect();
        json!({
            "members": limited,
            "total": total,
            "mentionHint": mention_hint,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_detection() {
        assert!(MarkdownProcessor::has_unclosed_fence("```rust\nfn x() {}"));
        assert!(!MarkdownProcessor::has_unclosed_fence("```rust\nfn x() {}\n```"));
        assert!(!MarkdownProcessor::has_unclosed_fence("plain text"));
    }

    #[test]
    fn table_row_detection() {
        assert!(MarkdownProcessor::ends_with_table_row("| a | b |\n| c | d |"));
        assert!(!MarkdownProcessor::ends_with_table_row("just text"));
        assert!(!MarkdownProcessor::ends_with_table_row(""));
    }

    #[test]
    fn separator_row_normalization() {
        let out = MarkdownProcessor::sanitize_markdown_table("| --- | --- |");
        assert_eq!(out, "|---|---|");
    }

    #[test]
    fn empty_table_row_skipped() {
        let out = MarkdownProcessor::sanitize_markdown_table("text\n||\nmore");
        assert_eq!(out, "text\nmore");
    }

    #[test]
    fn chunk_short_text_unchanged() {
        let chunks = MarkdownProcessor::chunk_markdown_text("hello world", 4000);
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn chunk_long_text_splits() {
        let para = "abcde".repeat(100); // 500 chars
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = MarkdownProcessor::chunk_markdown_text(&text, 600);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            // Each individual paragraph fits; combined would exceed.
            assert!(c.chars().count() <= 1200);
        }
    }

    #[test]
    fn strip_outer_fence() {
        let text = "```markdown\nhello\nworld\n```";
        assert_eq!(MarkdownProcessor::strip_outer_markdown_fence(text), "hello\nworld");
        let no_fence = "plain\ntext\nhere";
        assert_eq!(MarkdownProcessor::strip_outer_markdown_fence(no_fence), no_fence);
    }

    #[test]
    fn infer_separator() {
        assert_eq!(MarkdownProcessor::infer_block_separator("text ```", "more"), "\n");
        assert_eq!(MarkdownProcessor::infer_block_separator("a", "```b"), "\n");
        assert_eq!(
            MarkdownProcessor::infer_block_separator("| a |", "| b |"),
            "\n"
        );
        assert_eq!(MarkdownProcessor::infer_block_separator("text", "more"), "\n\n");
    }

    #[test]
    fn merge_streaming_fences() {
        let chunks = vec![
            "```rust\nfn x() {".to_string(),
            "```".to_string(),
        ];
        let merged = MarkdownProcessor::merge_block_streaming_fences(&chunks);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].contains("```rust"));
    }

    #[test]
    fn signature_is_hmac_sha256_hex() {
        let sig = SignManager::compute_signature("n", "t", "key", "secret");
        // plain = "ntkeysecret", key = "secret"
        assert_eq!(sig.len(), 64);
        // deterministic
        let sig2 = SignManager::compute_signature("n", "t", "key", "secret");
        assert_eq!(sig, sig2);
    }

    #[test]
    fn timestamp_format() {
        let ts = SignManager::build_timestamp();
        assert!(ts.ends_with("+08:00"));
        assert_eq!(ts.len(), "2006-01-02T15:04:05+08:00".len());
    }

    #[test]
    fn token_cache_validity() {
        let mut entry = TokenData::default();
        entry.expire_ts = now_secs() + 120.0;
        assert!(SignManager::is_cache_valid(&entry));
        entry.expire_ts = now_secs() + 10.0;
        assert!(!SignManager::is_cache_valid(&entry));
    }

    #[test]
    fn access_policy_dm() {
        let p = AccessPolicy::new("open", vec![], "open", vec![]);
        assert!(p.is_dm_allowed("anyone"));

        let p = AccessPolicy::new("disabled", vec![], "open", vec![]);
        assert!(!p.is_dm_allowed("anyone"));

        let p = AccessPolicy::new(
            "allowlist",
            vec!["alice".to_string()],
            "open",
            vec![],
        );
        assert!(p.is_dm_allowed("alice"));
        assert!(!p.is_dm_allowed("bob"));
    }

    #[test]
    fn access_policy_group() {
        let p = AccessPolicy::new(
            "open",
            vec![],
            "allowlist",
            vec!["g1".to_string()],
        );
        assert!(p.is_group_allowed("g1"));
        assert!(!p.is_group_allowed("g2"));
    }

    #[test]
    fn parse_json_push_basic() {
        let raw = json!({
            "from_account": "u123",
            "group_code": "g1",
            "msg_body": [
                { "msg_type": "TIMTextElem", "msg_content": { "text": "hi" } }
            ],
            "msg_id": "m1",
        });
        let push = DecodeMiddleware::parse_json_push(&raw).unwrap();
        assert_eq!(push.from_account, "u123");
        assert_eq!(push.group_code, "g1");
        assert_eq!(push.msg_id, "m1");
        assert_eq!(push.msg_body.len(), 1);
    }

    #[test]
    fn parse_json_push_pascal_case() {
        let raw = json!({
            "From_Account": "u9",
            "GroupId": "gx",
            "MsgBody": [
                { "MsgType": "TIMTextElem", "MsgContent": { "text": "yo" } }
            ],
            "MsgKey": "k9",
        });
        let push = DecodeMiddleware::parse_json_push(&raw).unwrap();
        assert_eq!(push.from_account, "u9");
        assert_eq!(push.group_code, "gx");
        assert_eq!(push.msg_id, "k9");
        assert_eq!(push.msg_body[0]["msg_type"], "TIMTextElem");
    }

    #[test]
    fn parse_json_push_empty_returns_none() {
        let raw = json!({ "to_account": "x" });
        assert!(DecodeMiddleware::parse_json_push(&raw).is_none());
    }

    #[test]
    fn extract_text_variants() {
        let body = vec![
            json!({ "msg_type": "TIMTextElem", "msg_content": { "text": "hello" } }),
            json!({ "msg_type": "TIMImageElem", "msg_content": {} }),
            json!({ "msg_type": "TIMFileElem", "msg_content": { "file_name": "a.pdf" } }),
        ];
        assert_eq!(
            ExtractContentMiddleware::extract_text(&body),
            "hello [image] [file: a.pdf]"
        );
    }

    #[test]
    fn extract_text_custom_mention() {
        let data = json!({ "elem_type": 1002, "text": "@bob" }).to_string();
        let body = vec![json!({
            "msg_type": "TIMCustomElem",
            "msg_content": { "data": data }
        })];
        assert_eq!(ExtractContentMiddleware::extract_text(&body), "@bob");
    }

    #[test]
    fn rewrite_fullwidth_slash() {
        assert_eq!(ExtractContentMiddleware::rewrite_slash_command("  ／new  "), "/new");
        assert_eq!(ExtractContentMiddleware::rewrite_slash_command("/reset"), "/reset");
    }

    #[test]
    fn inbound_media_refs() {
        let body = vec![
            json!({
                "msg_type": "TIMImageElem",
                "msg_content": {
                    "image_info_array": [
                        { "url": "http://a/small.jpg" },
                        { "url": "http://a/medium.jpg" }
                    ]
                }
            }),
            json!({
                "msg_type": "TIMFileElem",
                "msg_content": { "url": "http://a/doc.pdf", "file_name": "doc.pdf" }
            }),
        ];
        let refs = ExtractContentMiddleware::extract_inbound_media_refs(&body);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].kind, "image");
        // medium (index 1) preferred
        assert_eq!(refs[0].url, "http://a/medium.jpg");
        assert_eq!(refs[1].kind, "file");
        assert_eq!(refs[1].name.as_deref(), Some("doc.pdf"));
    }

    #[test]
    fn link_url_extraction() {
        let share = json!({ "elem_type": 1010, "link": "http://share" }).to_string();
        let body = vec![json!({
            "msg_type": "TIMCustomElem",
            "msg_content": { "data": share }
        })];
        let urls = ExtractContentMiddleware::extract_link_urls(&body);
        assert_eq!(urls, vec!["http://share".to_string()]);
    }

    #[test]
    fn placeholder_filter() {
        assert!(PlaceholderFilter::is_skippable_placeholder("[image]", 0));
        assert!(!PlaceholderFilter::is_skippable_placeholder("[image]", 1));
        assert!(!PlaceholderFilter::is_skippable_placeholder("hello", 0));
    }

    #[test]
    fn owner_command_detection() {
        let push = PushPayload {
            bot_owner_id: "owner1".to_string(),
            ..Default::default()
        };
        let body = vec![json!({
            "msg_type": "TIMTextElem",
            "msg_content": { "text": "/stop now" }
        })];
        let (cmd, line, is_owner) =
            OwnerCommand::detect_owner_command(&push, &body, "group", "owner1");
        assert_eq!(cmd.as_deref(), Some("/stop"));
        assert_eq!(line.as_deref(), Some("/stop now"));
        assert!(is_owner);

        // Non-owner
        let (cmd2, _l2, is_owner2) =
            OwnerCommand::detect_owner_command(&push, &body, "group", "someone_else");
        assert_eq!(cmd2.as_deref(), Some("/stop"));
        assert!(!is_owner2);

        // Not allowlisted
        let body2 = vec![json!({
            "msg_type": "TIMTextElem",
            "msg_content": { "text": "/unknown" }
        })];
        let (cmd3, _, _) = OwnerCommand::detect_owner_command(&push, &body2, "group", "owner1");
        assert!(cmd3.is_none());
    }

    #[test]
    fn owner_command_dm_skipped() {
        let push = PushPayload::default();
        let body = vec![json!({
            "msg_type": "TIMTextElem",
            "msg_content": { "text": "/stop" }
        })];
        let (cmd, _, _) = OwnerCommand::detect_owner_command(&push, &body, "dm", "u1");
        assert!(cmd.is_none());
    }

    #[test]
    fn route_chat_group_and_dm() {
        let g = route_chat("g1", "Group One", "u1", "Nick");
        assert_eq!(g.chat_id, "group:g1");
        assert_eq!(g.chat_type, "group");
        assert_eq!(g.chat_name, "Group One");

        let d = route_chat("", "", "u1", "Nick");
        assert_eq!(d.chat_id, "direct:u1");
        assert_eq!(d.chat_type, "dm");
        assert_eq!(d.chat_name, "Nick");
    }

    #[test]
    fn self_reference() {
        assert!(is_self_reference("bot1", Some("bot1")));
        assert!(!is_self_reference("bot1", Some("bot2")));
        assert!(!is_self_reference("bot1", None));
        assert!(!is_self_reference("", Some("bot1")));
    }

    #[test]
    fn classify_types() {
        use crate::gateway::MessageType;
        assert_eq!(classify_message_type("/cmd", &[]), MessageType::Command);
        let img = vec![json!({ "msg_type": "TIMImageElem" })];
        assert_eq!(classify_message_type("x", &img), MessageType::Photo);
        let file = vec![json!({ "msg_type": "TIMFileElem" })];
        assert_eq!(classify_message_type("x", &file), MessageType::Document);
        assert_eq!(classify_message_type("plain", &[]), MessageType::Text);
    }

    #[test]
    fn at_bot_detection() {
        let data = json!({ "elem_type": 1002, "user_id": "bot1", "text": "@bot" }).to_string();
        let body = vec![json!({
            "msg_type": "TIMCustomElem",
            "msg_content": { "data": data }
        })];
        assert!(GroupAtGuard::is_at_bot(&body, Some("bot1")));
        assert!(!GroupAtGuard::is_at_bot(&body, Some("bot2")));
        assert_eq!(
            GroupAtGuard::extract_bot_mention_text(&body, Some("bot1")),
            "@bot"
        );
    }

    #[test]
    fn channel_prompt_contains_identity() {
        let data = json!({ "elem_type": 1002, "user_id": "bot1", "text": "@yb" }).to_string();
        let body = vec![json!({
            "msg_type": "TIMCustomElem",
            "msg_content": { "data": data }
        })];
        let prompt = GroupAtGuard::build_group_channel_prompt(&body, Some("bot1"));
        assert!(prompt.contains("user_id=bot1"));
        assert!(prompt.contains("@yb"));
    }

    #[test]
    fn quote_context_extraction() {
        let ccd = json!({
            "quote": { "type": 1, "desc": "hi there", "id": "q1", "sender_nickname": "Alice" }
        })
        .to_string();
        let (id, text) = extract_quote_context(&ccd);
        assert_eq!(id.as_deref(), Some("q1"));
        assert_eq!(text.as_deref(), Some("Alice: hi there"));

        // image-type quote with no desc → placeholder
        let ccd2 = json!({ "quote": { "type": 2, "id": "q2" } }).to_string();
        let (_id2, text2) = extract_quote_context(&ccd2);
        assert_eq!(text2.as_deref(), Some("[image]"));

        // empty
        let (id3, text3) = extract_quote_context("");
        assert!(id3.is_none());
        assert!(text3.is_none());
    }

    #[test]
    fn guess_image_ext() {
        assert_eq!(MediaResolve::guess_image_ext_from_url("http://a/x.png"), ".png");
        assert_eq!(MediaResolve::guess_image_ext_from_url("http://a/x.webp"), ".webp");
        assert_eq!(MediaResolve::guess_image_ext_from_url("http://a/x.exe"), ".jpg");
        assert_eq!(MediaResolve::guess_image_ext_from_url("http://a/noext"), ".jpg");
    }

    #[test]
    fn extract_resource_id() {
        assert_eq!(
            MediaResolve::extract_resource_id(
                "https://h.tencent.com/api/resource/download?resourceId=abc123"
            ),
            Some("abc123".to_string())
        );
        assert_eq!(
            MediaResolve::extract_resource_id("https://h.tencent.com/x"),
            None
        );
    }

    #[test]
    fn validate_media_cases() {
        assert!(MessageSender::validate_media(None, "f", 50).is_some());
        assert!(MessageSender::validate_media(Some(&[]), "f", 50).is_some());
        assert!(MessageSender::validate_media(Some(&[1, 2, 3]), "f", 50).is_none());
        let big = vec![0u8; 2 * 1024 * 1024];
        assert!(MessageSender::validate_media(Some(&big), "f", 1).is_some());
    }

    #[test]
    fn truncate_short_message() {
        assert_eq!(
            MessageSender::truncate_message("short", 4000),
            vec!["short".to_string()]
        );
    }

    #[test]
    fn strip_cron_wrapper_cases() {
        let content = "Cronjob Response: \n(job_id: 5)\n-------------\n\nthe actual body\n\nTo stop or manage this job, send me a new message (e.g. \"stop reminder 5\")";
        assert_eq!(MessageSender::strip_cron_wrapper(content), "the actual body");

        // Not a cron wrapper
        assert_eq!(MessageSender::strip_cron_wrapper("regular"), "regular");
    }

    #[test]
    fn strip_page_indicator() {
        assert_eq!(strip_indicator("Hello world (1/3)"), "Hello world");
        assert_eq!(strip_indicator("No indicator here"), "No indicator here");
        assert_eq!(strip_indicator("Text (abc)"), "Text (abc)");
    }

    #[test]
    fn mentions_without_members_plain() {
        let body = MessageSender::build_msg_body_with_mentions("hi @bob", &[]);
        assert_eq!(body.len(), 1);
        assert_eq!(body[0]["msg_content"]["text"], "hi @bob");
    }

    #[test]
    fn mentions_with_members() {
        let members = vec![json!({ "nickname": "Bob", "user_id": "u_bob" })];
        let body = MessageSender::build_msg_body_with_mentions("hello @Bob there", &members);
        // Expect: text "hello", custom @Bob, text "there"
        assert!(body.iter().any(|e| e["msg_type"] == "TIMCustomElem"));
        let custom = body
            .iter()
            .find(|e| e["msg_type"] == "TIMCustomElem")
            .unwrap();
        let data: Value =
            serde_json::from_str(custom["msg_content"]["data"].as_str().unwrap()).unwrap();
        assert_eq!(data["elem_type"], 1002);
        assert_eq!(data["user_id"], "u_bob");
        assert_eq!(data["text"], "@Bob");
    }

    #[test]
    fn find_at_mentions_positions() {
        let m = find_at_mentions("hi @bob and @alice");
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].2, "bob");
        assert_eq!(m[1].2, "alice");

        // email-like @ not preceded by whitespace should be skipped
        let m2 = find_at_mentions("user@host");
        assert!(m2.is_empty());
    }

    #[test]
    fn group_query_filter_find() {
        let members = vec![
            json!({ "nickname": "Alice", "user_id": "u1" }),
            json!({ "nickname": "Bob", "user_id": "u2" }),
        ];
        let result = GroupQueryService::filter_members(&members, "find", Some("ali"));
        assert_eq!(result["total"], 1);
        assert_eq!(result["members"][0]["nickname"], "Alice");
        assert!(result["mentionHint"].as_str().unwrap().contains("Alice"));
    }

    #[test]
    fn group_query_filter_bots() {
        let members = vec![
            json!({ "nickname": "ChatBot", "user_id": "u1" }),
            json!({ "nickname": "Human", "user_id": "u2" }),
        ];
        let result = GroupQueryService::filter_members(&members, "list_bots", None);
        assert_eq!(result["total"], 1);
        assert_eq!(result["members"][0]["nickname"], "ChatBot");
    }

    #[test]
    fn adapter_new_from_extra() {
        let extra = json!({
            "app_id": " key123 ",
            "app_secret": " secret ",
            "bot_id": "bot42",
            "dm_policy": "allowlist",
            "dm_allow_from": "alice, bob",
        });
        // Ensure env doesn't interfere.
        unsafe {
            std::env::remove_var("YUANBAO_DM_POLICY");
            std::env::remove_var("YUANBAO_DM_ALLOW_FROM");
            std::env::remove_var("YUANBAO_GROUP_POLICY");
            std::env::remove_var("YUANBAO_GROUP_ALLOW_FROM");
            std::env::remove_var("YUANBAO_HOME_CHANNEL");
        }
        let a = YuanbaoAdapter::new(&extra, "");
        assert_eq!(a.app_key, "key123");
        assert_eq!(a.app_secret, "secret");
        assert_eq!(a.bot_id.as_deref(), Some("bot42"));
        assert_eq!(a.ws_url, DEFAULT_WS_GATEWAY_URL);
        assert_eq!(a.api_domain, DEFAULT_API_DOMAIN);
        assert_eq!(a.access_policy.dm_policy(), "allowlist");
        assert!(a.access_policy.is_dm_allowed("alice"));
        assert!(!a.access_policy.is_dm_allowed("carol"));
    }

    #[test]
    fn adapter_auto_sethome_done_flag() {
        unsafe {
            std::env::remove_var("YUANBAO_HOME_CHANNEL");
        }
        // existing dm home → done
        let a = YuanbaoAdapter::new(&json!({}), "direct:u1");
        assert!(a.auto_sethome_done);
        // existing group home → not done (eligible for upgrade)
        let b = YuanbaoAdapter::new(&json!({}), "group:g1");
        assert!(!b.auto_sethome_done);
        // no home → not done
        let c = YuanbaoAdapter::new(&json!({}), "");
        assert!(!c.auto_sethome_done);
    }

    #[test]
    fn adapter_member_cache() {
        let mut a = YuanbaoAdapter::new(&json!({}), "");
        a.cache_members("g1", vec![json!({ "nickname": "X", "user_id": "u" })]);
        let fresh = a.fresh_members("g1");
        assert_eq!(fresh.len(), 1);
        assert!(a.fresh_members("nonexistent").is_empty());
    }

    #[test]
    fn adapter_chat_info() {
        let a = YuanbaoAdapter::new(&json!({}), "");
        assert_eq!(a.get_chat_info("group:g1")["type"], "group");
        assert_eq!(a.get_chat_info("direct:u1")["type"], "dm");
    }

    #[test]
    fn merge_frames_aggregates() {
        let frame1 = serde_json::to_vec(&json!({
            "from_account": "u1",
            "msg_body": [{ "msg_type": "TIMTextElem", "msg_content": { "text": "a" } }]
        }))
        .unwrap();
        let frame2 = serde_json::to_vec(&json!({
            "from_account": "u1",
            "msg_body": [{ "msg_type": "TIMTextElem", "msg_content": { "text": "b" } }]
        }))
        .unwrap();
        let (push, via) = DecodeMiddleware::merge_frames(&[frame1, frame2]).unwrap();
        assert_eq!(via, "json");
        // base body + separator + extra body
        assert_eq!(push.msg_body.len(), 3);
        assert_eq!(push.msg_body[1]["msg_content"]["text"], "\n");
    }
}
