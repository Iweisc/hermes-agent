//! Matrix platform adapter — native Rust port of
//! `gateway/platforms/matrix.py`.
//!
//! The original Python module connects to a Matrix homeserver through the
//! `mautrix` async SDK (HTTP long-poll sync, OlmMachine E2EE, to-device key
//! shares, `aiohttp` sessions). That live client lifecycle and the libolm
//! cryptography have no faithful equivalent in the crates available to this
//! port, so the websocket/sync loop, OlmMachine bootstrap, and media
//! upload/download paths are left to the Python bridge.
//!
//! What is ported here is the **deterministic, side-effect-free logic** that
//! other Hermes code and tests depend on, reproducing Python behaviour
//! exactly:
//!
//! - [`check_matrix_requirements`] — env-driven precondition gate.
//! - [`MatrixConfig`] — env/extra parsing done once in `__init__`.
//! - [`looks_like_matrix_image_filename`] — drop transport filenames as bodies.
//! - [`mxc_to_http`] — `mxc://` → homeserver download URL.
//! - Mention machinery: [`extract_outbound_mentions`],
//!   [`inject_outbound_mention_links`], [`protect_outbound_mention_regions`],
//!   [`is_bot_mentioned`], [`strip_mention`].
//! - Message content builder: [`build_text_message_content`],
//!   [`build_edit_message_content`], [`build_reaction_content`],
//!   [`build_media_message_content`].
//! - Markdown → Matrix HTML fallback: [`markdown_to_html_fallback`],
//!   [`sanitize_link_url`], [`html_escape`].
//! - Sender gating: [`is_self_sender`], [`is_system_or_bridge_sender`].
//! - Reply-fallback stripping: [`strip_reply_fallback`].
//! - Event dedup: [`EventDedup`].
//! - Text batching: [`TextBatchConfig`].
//! - Reaction-approval bookkeeping: [`MatrixApprovalPrompt`],
//!   [`ApprovalRegistry`], [`approval_reaction_choice`].
//! - Presence validation: [`is_valid_presence_state`].

use std::collections::{HashSet, VecDeque};

use regex::Regex;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Matrix message size limit (4000 chars practical).
pub const MAX_MESSAGE_LENGTH: usize = 4000;

/// Threshold for detecting Matrix client-side message splits.
pub const SPLIT_THRESHOLD: usize = 3900;

/// Grace period: ignore messages older than this many seconds before startup.
pub const STARTUP_GRACE_SECONDS: f64 = 5.0;

/// Hint shown when E2EE dependencies are missing.
pub const E2EE_INSTALL_HINT: &str =
    "Install with: pip install 'mautrix[encryption]'  (requires libolm C library)";

/// Image filename extensions recognised in Matrix `m.image` bodies.
pub const MATRIX_IMAGE_FILENAME_EXTS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic", ".heif", ".avif",
];

/// Valid presence states accepted by `set_presence`.
pub const VALID_PRESENCE_STATES: &[&str] = &["online", "offline", "unavailable"];

// ─── Regexes (lazily compiled) ──────────────────────────────────────────────

thread_local! {
    static OUTBOUND_MENTION_RE: Regex = Regex::new(
        r"(?P<lead>^|[^\w/])(?P<mxid>@[0-9A-Za-z._=/-]+:[0-9A-Za-z.-]+(?::\d+)?)",
    )
    .unwrap();
}

fn with_outbound_mention_re<R>(f: impl FnOnce(&Regex) -> R) -> R {
    OUTBOUND_MENTION_RE.with(|re| f(re))
}

// ─── Boolean env parsing ────────────────────────────────────────────────────

/// Python truthy: lowercased value in {"true","1","yes"}.
pub fn env_truthy(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "true" | "1" | "yes")
}

/// Python falsy-gate: lowercased value NOT in {"false","0","no"} → true.
/// Used for defaults-true flags like MATRIX_REQUIRE_MENTION / MATRIX_REACTIONS.
pub fn env_not_falsy(value: &str) -> bool {
    !matches!(value.trim().to_lowercase().as_str(), "false" | "0" | "no")
}

fn env_or_empty(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

// ─── Requirements gate ──────────────────────────────────────────────────────

/// Result of [`check_matrix_requirements`] explaining the gating decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementsCheck {
    /// All preconditions satisfied (mautrix availability assumed by caller).
    Ok,
    /// Neither access token nor password configured.
    MissingCredentials,
    /// Homeserver URL not configured.
    MissingHomeserver,
    /// MATRIX_ENCRYPTION=true but E2EE deps unavailable.
    EncryptionRequestedButUnavailable,
}

impl RequirementsCheck {
    pub fn is_ok(&self) -> bool {
        matches!(self, RequirementsCheck::Ok)
    }
}

/// Pure form of Python `check_matrix_requirements`.
///
/// `e2ee_deps_available` is the result the Python code obtains from importing
/// `mautrix.crypto.OlmMachine`; the caller supplies it because libolm cannot
/// be probed from this port. Note Python does NOT check homeserver before the
/// credentials check — it returns False on missing credentials first.
pub fn check_matrix_requirements_from(
    access_token: &str,
    password: &str,
    homeserver: &str,
    encryption_env: &str,
    e2ee_deps_available: bool,
) -> RequirementsCheck {
    if access_token.is_empty() && password.is_empty() {
        return RequirementsCheck::MissingCredentials;
    }
    if homeserver.is_empty() {
        return RequirementsCheck::MissingHomeserver;
    }
    if env_truthy(encryption_env) && !e2ee_deps_available {
        return RequirementsCheck::EncryptionRequestedButUnavailable;
    }
    RequirementsCheck::Ok
}

/// Convenience wrapper reading from environment (assumes E2EE deps present
/// unless `e2ee_deps_available` says otherwise).
pub fn check_matrix_requirements(e2ee_deps_available: bool) -> RequirementsCheck {
    check_matrix_requirements_from(
        &env_or_empty("MATRIX_ACCESS_TOKEN"),
        &env_or_empty("MATRIX_PASSWORD"),
        &env_or_empty("MATRIX_HOMESERVER"),
        &env_or_empty("MATRIX_ENCRYPTION"),
        e2ee_deps_available,
    )
}

// ─── Config (mirrors MatrixAdapter.__init__ env parsing) ────────────────────

/// All adapter settings derived from env vars / config extra at construction.
#[derive(Debug, Clone)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub access_token: String,
    pub user_id: String,
    pub password: String,
    pub encryption: bool,
    pub device_id: String,
    pub require_mention: bool,
    pub free_rooms: HashSet<String>,
    pub auto_thread: bool,
    pub dm_auto_thread: bool,
    pub dm_mention_threads: bool,
    pub reactions_enabled: bool,
    pub allowed_user_ids: HashSet<String>,
    pub text_batch_delay_seconds: f64,
    pub text_batch_split_delay_seconds: f64,
}

fn split_csv_set(raw: &str) -> HashSet<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn parse_f64_or(raw: &str, default: f64) -> f64 {
    let t = raw.trim();
    if t.is_empty() {
        return default;
    }
    t.parse::<f64>().unwrap_or(default)
}

impl MatrixConfig {
    /// Build from the environment, mirroring the Python `__init__`. The
    /// homeserver has any trailing slashes stripped.
    pub fn from_env() -> Self {
        let homeserver = env_or_empty("MATRIX_HOMESERVER")
            .trim_end_matches('/')
            .to_string();
        let encryption = env_truthy(&env_or_empty("MATRIX_ENCRYPTION"));

        let require_mention = match std::env::var("MATRIX_REQUIRE_MENTION") {
            Ok(v) => env_not_falsy(&v),
            Err(_) => env_not_falsy("true"),
        };
        let auto_thread = match std::env::var("MATRIX_AUTO_THREAD") {
            Ok(v) => env_truthy(&v),
            Err(_) => env_truthy("true"),
        };
        let reactions_enabled = match std::env::var("MATRIX_REACTIONS") {
            Ok(v) => env_not_falsy(&v),
            Err(_) => env_not_falsy("true"),
        };

        MatrixConfig {
            homeserver,
            access_token: env_or_empty("MATRIX_ACCESS_TOKEN"),
            user_id: env_or_empty("MATRIX_USER_ID"),
            password: env_or_empty("MATRIX_PASSWORD"),
            encryption,
            device_id: env_or_empty("MATRIX_DEVICE_ID"),
            require_mention,
            free_rooms: split_csv_set(&env_or_empty("MATRIX_FREE_RESPONSE_ROOMS")),
            auto_thread,
            dm_auto_thread: env_truthy(&env_or_empty("MATRIX_DM_AUTO_THREAD")),
            dm_mention_threads: env_truthy(&env_or_empty("MATRIX_DM_MENTION_THREADS")),
            reactions_enabled,
            allowed_user_ids: split_csv_set(&env_or_empty("MATRIX_ALLOWED_USERS")),
            text_batch_delay_seconds: parse_f64_or(
                &env_or_empty("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS"),
                0.6,
            ),
            text_batch_split_delay_seconds: parse_f64_or(
                &env_or_empty("HERMES_MATRIX_TEXT_BATCH_SPLIT_DELAY_SECONDS"),
                2.0,
            ),
        }
    }
}

// ─── Image-filename heuristic ───────────────────────────────────────────────

fn basename(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

fn suffix_lower(name: &str) -> Option<String> {
    // Python pathlib `.suffix`: the final dotted component, but not a leading
    // dot (e.g. ".bashrc" has no suffix).
    let bytes = name.as_bytes();
    let dot = name.rfind('.')?;
    if dot == 0 {
        return None;
    }
    // Ensure the dot is part of the filename, not preceded only by dots.
    if bytes[dot - 1] == b'.' && name[..dot].chars().all(|c| c == '.') {
        return None;
    }
    if dot + 1 >= name.len() {
        return None;
    }
    Some(name[dot..].to_lowercase())
}

/// Guess whether `suffix` corresponds to an image mimetype, mirroring the
/// subset of `mimetypes.guess_type` Python relies on plus the explicit
/// extension allowlist.
fn suffix_is_image(suffix: &str) -> bool {
    let guessed_image = matches!(
        suffix,
        ".jpg"
            | ".jpeg"
            | ".jpe"
            | ".png"
            | ".gif"
            | ".bmp"
            | ".webp"
            | ".tiff"
            | ".tif"
            | ".ico"
            | ".svg"
    );
    guessed_image || MATRIX_IMAGE_FILENAME_EXTS.contains(&suffix)
}

/// Port of Python `_looks_like_matrix_image_filename`.
pub fn looks_like_matrix_image_filename(text: &str) -> bool {
    let candidate = text.trim();
    if candidate.is_empty() || candidate.contains('\n') || candidate.ends_with('/') {
        return false;
    }
    let name = basename(candidate);
    if name.is_empty() || name != candidate {
        return false;
    }
    match suffix_lower(name) {
        Some(s) => suffix_is_image(&s),
        None => false,
    }
}

// ─── mxc:// → HTTP ──────────────────────────────────────────────────────────

/// Port of Python `_mxc_to_http`. `homeserver` should already be trailing-slash
/// trimmed (as stored on the adapter).
pub fn mxc_to_http(homeserver: &str, mxc_url: &str) -> String {
    if !mxc_url.starts_with("mxc://") {
        return mxc_url.to_string();
    }
    let parts = &mxc_url[6..];
    format!(
        "{}/_matrix/client/v1/media/download/{}",
        homeserver, parts
    )
}

// ─── HTML escaping (Python html.escape, quote=True default) ─────────────────

/// Port of Python `html.escape(s, quote=True)`.
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Port of Python `_sanitize_link_url`.
pub fn sanitize_link_url(url: &str) -> String {
    let stripped = url.trim();
    let scheme = if stripped.contains(':') {
        stripped
            .splitn(2, ':')
            .next()
            .unwrap_or("")
            .to_lowercase()
            .trim()
            .to_string()
    } else {
        String::new()
    };
    if matches!(scheme.as_str(), "javascript" | "data" | "vbscript") {
        return String::new();
    }
    stripped.replace('"', "&quot;")
}

// ─── Outbound mention protection / extraction / injection ───────────────────

/// Port of Python `_protect_outbound_mention_regions`. Returns the protected
/// text (with placeholder sentinels) and the list of original protected
/// fragments in order.
pub fn protect_outbound_mention_regions(text: &str) -> (String, Vec<String>) {
    let mut placeholders: Vec<String> = Vec::new();

    // ```...``` fenced blocks (DOTALL).
    let fenced = Regex::new(r"(?s)```.*?```").unwrap();
    let inline = Regex::new(r"`[^`\n]+`").unwrap();
    let links = Regex::new(r"\[[^\]]+\]\([^)]+\)").unwrap();

    let step = |re: &Regex, input: &str, placeholders: &mut Vec<String>| -> String {
        let mut result = String::new();
        let mut last = 0usize;
        for m in re.find_iter(input) {
            result.push_str(&input[last..m.start()]);
            let ph = {
                let idx = placeholders.len();
                placeholders.push(m.as_str().to_string());
                format!("\u{0}MENTION_PROTECTED{}\u{0}", idx)
            };
            result.push_str(&ph);
            last = m.end();
        }
        result.push_str(&input[last..]);
        result
    };

    let s1 = step(&fenced, text, &mut placeholders);
    let s2 = step(&inline, &s1, &mut placeholders);
    let s3 = step(&links, &s2, &mut placeholders);

    (s3, placeholders)
}

/// Port of Python `_extract_outbound_mentions`.
pub fn extract_outbound_mentions(text: &str) -> Vec<String> {
    let (protected, _) = protect_outbound_mention_regions(text);
    let mut seen: HashSet<String> = HashSet::new();
    let mut mentions: Vec<String> = Vec::new();
    with_outbound_mention_re(|re| {
        for caps in re.captures_iter(&protected) {
            let user_id = caps.name("mxid").unwrap().as_str().to_string();
            if seen.insert(user_id.clone()) {
                mentions.push(user_id);
            }
        }
    });
    mentions
}

/// Port of Python `_inject_outbound_mention_links`.
pub fn inject_outbound_mention_links(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let (protected, placeholders) = protect_outbound_mention_regions(text);

    let linked = with_outbound_mention_re(|re| {
        let mut result = String::new();
        let mut last = 0usize;
        for caps in re.captures_iter(&protected) {
            let whole = caps.get(0).unwrap();
            let lead = caps.name("lead").map(|m| m.as_str()).unwrap_or("");
            let mxid = caps.name("mxid").unwrap().as_str();
            result.push_str(&protected[last..whole.start()]);
            result.push_str(lead);
            result.push_str(&format!("[{}](https://matrix.to/#/{})", mxid, mxid));
            last = whole.end();
        }
        result.push_str(&protected[last..]);
        result
    });

    let mut out = linked;
    for (idx, original) in placeholders.iter().enumerate() {
        out = out.replace(&format!("\u{0}MENTION_PROTECTED{}\u{0}", idx), original);
    }
    out
}

// ─── Bot mention detection / stripping ──────────────────────────────────────

/// Localpart of an MXID `@local:server` → `local`; otherwise None.
fn mxid_localpart(user_id: &str) -> Option<String> {
    if !user_id.contains(':') {
        return None;
    }
    let before_colon = user_id.split(':').next().unwrap_or("");
    let lp = before_colon.trim_start_matches('@');
    if lp.is_empty() {
        None
    } else {
        Some(lp.to_string())
    }
}

/// Port of Python `_is_bot_mentioned`.
pub fn is_bot_mentioned(
    user_id: &str,
    body: &str,
    formatted_body: Option<&str>,
    mention_user_ids: Option<&[String]>,
) -> bool {
    // m.mentions.user_ids — authoritative per MSC3952 / Matrix v1.7.
    if let Some(ids) = mention_user_ids {
        if !user_id.is_empty() && ids.iter().any(|id| id == user_id) {
            return true;
        }
    }
    if body.is_empty() && formatted_body.map(|f| f.is_empty()).unwrap_or(true) {
        return false;
    }
    if !user_id.is_empty() && body.contains(user_id) {
        return true;
    }
    if let Some(localpart) = mxid_localpart(user_id) {
        let pattern = format!(r"(?i)\b{}\b", regex::escape(&localpart));
        if let Ok(re) = Regex::new(&pattern) {
            if re.is_match(body) {
                return true;
            }
        }
    }
    if let Some(fb) = formatted_body {
        if !user_id.is_empty() && fb.contains(&format!("matrix.to/#/{}", user_id)) {
            return true;
        }
    }
    false
}

/// Port of Python `_strip_mention`.
pub fn strip_mention(user_id: &str, body: &str) -> String {
    if body.is_empty() {
        return String::new();
    }
    let mut body = body.to_string();

    if !user_id.is_empty() {
        body = body.replace(user_id, "");
    }

    if let Some(localpart) = mxid_localpart(user_id) {
        let pattern = format!(r"(?i)(?P<pre>^|[^\w])@{}\b", regex::escape(&localpart));
        if let Ok(re) = Regex::new(&pattern) {
            // Replace the @localpart token but keep the leading non-word char.
            let mut result = String::new();
            let mut last = 0usize;
            for caps in re.captures_iter(&body) {
                let whole = caps.get(0).unwrap();
                let pre = caps.name("pre").map(|m| m.as_str()).unwrap_or("");
                result.push_str(&body[last..whole.start()]);
                result.push_str(pre);
                last = whole.end();
            }
            result.push_str(&body[last..]);
            body = result;
        }
    }

    // Normalize spacing after mention removal.
    let collapse_ws = Regex::new(r"[ \t]{2,}").unwrap();
    body = collapse_ws.replace_all(&body, " ").to_string();
    let before_punct = Regex::new(r"\s+([,.;:!?])").unwrap();
    body = before_punct.replace_all(&body, "$1").to_string();
    body.trim().to_string()
}

// ─── Display name fallback ──────────────────────────────────────────────────

/// Port of the fallback branch of Python `_get_display_name`: strip the
/// `@localpart:server` form to just the localpart, else return as-is.
pub fn display_name_fallback(user_id: &str) -> String {
    if user_id.starts_with('@') && user_id.contains(':') {
        user_id[1..].split(':').next().unwrap_or("").to_string()
    } else {
        user_id.to_string()
    }
}

// ─── Sender gating ──────────────────────────────────────────────────────────

/// Port of Python `_is_self_sender`. When `own_user_id` is empty the bot
/// cannot prove a sender is not itself, so returns true defensively.
pub fn is_self_sender(own_user_id: &str, sender: &str) -> bool {
    let own = own_user_id.trim().to_lowercase();
    if own.is_empty() {
        return true;
    }
    sender.trim().to_lowercase() == own
}

/// Port of Python `_is_system_or_bridge_sender`.
pub fn is_system_or_bridge_sender(sender: &str) -> bool {
    let mut s = sender.trim().to_string();
    if s.is_empty() {
        return true;
    }
    if s.starts_with('@') {
        s = s[1..].to_string();
    }
    let localpart = if s.contains(':') {
        s.split(':').next().unwrap_or("").to_string()
    } else {
        s
    };
    if localpart.is_empty() {
        return true;
    }
    localpart.starts_with('_')
}

// ─── Startup grace ──────────────────────────────────────────────────────────

/// Port of the startup-grace check in `_on_room_message`. `event_ts` and
/// `startup_ts` are in seconds. Returns true when the event should be dropped
/// as too old. A zero/absent `event_ts` is never dropped.
pub fn is_before_startup_grace(event_ts: f64, startup_ts: f64) -> bool {
    event_ts != 0.0 && event_ts < startup_ts - STARTUP_GRACE_SECONDS
}

// ─── Reply-fallback stripping ───────────────────────────────────────────────

/// Port of the reply-fallback stripping in `_handle_text_message`. Only strips
/// when `reply_to` is set and the body begins with `"> "`.
pub fn strip_reply_fallback(body: &str, reply_to: bool) -> String {
    if !reply_to || !body.starts_with("> ") {
        return body.to_string();
    }
    let lines: Vec<&str> = body.split('\n').collect();
    let mut stripped: Vec<&str> = Vec::new();
    let mut past_fallback = false;
    for line in &lines {
        if !past_fallback {
            if line.starts_with("> ") || *line == ">" {
                continue;
            }
            if *line == "" {
                past_fallback = true;
                continue;
            }
            past_fallback = true;
        }
        stripped.push(line);
    }
    if stripped.is_empty() {
        body.to_string()
    } else {
        stripped.join("\n")
    }
}

// ─── Event deduplication ────────────────────────────────────────────────────

/// Bounded event-id dedup matching Python's deque(maxlen=1000) + set.
pub struct EventDedup {
    order: VecDeque<String>,
    seen: HashSet<String>,
    maxlen: usize,
}

impl EventDedup {
    pub fn new(maxlen: usize) -> Self {
        EventDedup {
            order: VecDeque::with_capacity(maxlen),
            seen: HashSet::new(),
            maxlen,
        }
    }

    /// Port of `_is_duplicate_event`: returns true when already processed,
    /// otherwise records the id and returns false.
    pub fn is_duplicate(&mut self, event_id: &str) -> bool {
        if event_id.is_empty() {
            return false;
        }
        if self.seen.contains(event_id) {
            return true;
        }
        if self.order.len() == self.maxlen {
            if let Some(evicted) = self.order.pop_front() {
                self.seen.remove(&evicted);
            }
        }
        self.order.push_back(event_id.to_string());
        self.seen.insert(event_id.to_string());
        false
    }
}

impl Default for EventDedup {
    fn default() -> Self {
        EventDedup::new(1000)
    }
}

// ─── Text batching ──────────────────────────────────────────────────────────

/// Port of the text-batch delay selection logic.
#[derive(Debug, Clone, Copy)]
pub struct TextBatchConfig {
    pub delay_seconds: f64,
    pub split_delay_seconds: f64,
}

impl TextBatchConfig {
    /// Batching is active when delay > 0 (Python checks `> 0`).
    pub fn enabled(&self) -> bool {
        self.delay_seconds > 0.0
    }

    /// Delay for a batch whose last chunk was `last_chunk_len` chars: the
    /// split delay when at/over the split threshold, else the base delay.
    pub fn delay_for(&self, last_chunk_len: usize) -> f64 {
        if last_chunk_len >= SPLIT_THRESHOLD {
            self.split_delay_seconds
        } else {
            self.delay_seconds
        }
    }
}

// ─── Message-content builders ───────────────────────────────────────────────

use serde_json::{json, Map, Value};

/// Strip image markdown the way `format_message` does (media uploaded
/// separately): `![alt](url)` → `url`.
pub fn format_message(content: &str) -> String {
    let re = Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").unwrap();
    re.replace_all(content, "$2").to_string()
}

/// Port of Python `_build_text_message_content`. `markdown_html` should be the
/// result of converting `inject_outbound_mention_links(text)` through the
/// markdown→HTML pipeline (use [`markdown_to_html_fallback`] when no markdown
/// library is available).
pub fn build_text_message_content(text: &str, msgtype: &str) -> Value {
    let mut content = Map::new();
    content.insert("msgtype".into(), Value::String(msgtype.to_string()));
    content.insert("body".into(), Value::String(text.to_string()));

    let mention_user_ids = extract_outbound_mentions(text);
    if !mention_user_ids.is_empty() {
        content.insert(
            "m.mentions".into(),
            json!({ "user_ids": mention_user_ids }),
        );
    }

    let html_source = inject_outbound_mention_links(text);
    let html = markdown_to_html_fallback(&html_source);
    if !html.is_empty() && html != text {
        content.insert(
            "format".into(),
            Value::String("org.matrix.custom.html".into()),
        );
        content.insert("formatted_body".into(), Value::String(html));
    }

    Value::Object(content)
}

/// Port of Python `edit_message` content construction (the `m.replace` body).
/// `formatted` is the already format_message'd content.
pub fn build_edit_message_content(formatted: &str, message_id: &str) -> Value {
    let new_content = build_text_message_content(formatted, "m.text");
    let new_obj = new_content.as_object().unwrap();

    let mut content = Map::new();
    content.insert("msgtype".into(), Value::String("m.text".into()));
    content.insert("body".into(), Value::String(format!("* {}", formatted)));
    content.insert("m.new_content".into(), new_content.clone());

    if let Some(mentions) = new_obj.get("m.mentions") {
        content.insert("m.mentions".into(), mentions.clone());
    }
    if let Some(Value::String(fb)) = new_obj.get("formatted_body") {
        content.insert(
            "format".into(),
            Value::String("org.matrix.custom.html".into()),
        );
        content.insert(
            "formatted_body".into(),
            Value::String(format!("* {}", fb)),
        );
    }
    content.insert(
        "m.relates_to".into(),
        json!({ "rel_type": "m.replace", "event_id": message_id }),
    );

    Value::Object(content)
}

/// Apply reply-to and thread relation onto a message content object, mirroring
/// the relation logic in `send` / `_upload_and_send`.
pub fn apply_relations(
    content: &mut Value,
    reply_to: Option<&str>,
    thread_id: Option<&str>,
) {
    let obj = match content.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    if let Some(rt) = reply_to {
        obj.insert(
            "m.relates_to".into(),
            json!({ "m.in_reply_to": { "event_id": rt } }),
        );
    }

    if let Some(tid) = thread_id {
        let mut relates_to = obj
            .get("m.relates_to")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        relates_to.insert("rel_type".into(), Value::String("m.thread".into()));
        relates_to.insert("event_id".into(), Value::String(tid.to_string()));
        relates_to.insert("is_falling_back".into(), Value::Bool(true));
        if let Some(rt) = reply_to {
            if !relates_to.contains_key("m.in_reply_to") {
                relates_to.insert(
                    "m.in_reply_to".into(),
                    json!({ "event_id": rt }),
                );
            }
        }
        obj.insert("m.relates_to".into(), Value::Object(relates_to));
    }
}

/// Port of Python `_send_reaction` content construction.
pub fn build_reaction_content(event_id: &str, emoji: &str) -> Value {
    json!({
        "m.relates_to": {
            "rel_type": "m.annotation",
            "event_id": event_id,
            "key": emoji,
        }
    })
}

/// Port of Python `_upload_and_send` content construction (sans relations,
/// which the caller applies via [`apply_relations`]).
pub fn build_media_message_content(
    msgtype: &str,
    body: &str,
    content_type: &str,
    raw_size: usize,
    mxc_url: &str,
    encrypted_file: Option<Value>,
    is_voice: bool,
) -> Value {
    let mut content = Map::new();
    content.insert("msgtype".into(), Value::String(msgtype.to_string()));
    content.insert("body".into(), Value::String(body.to_string()));
    content.insert(
        "info".into(),
        json!({ "mimetype": content_type, "size": raw_size }),
    );

    match encrypted_file {
        Some(Value::Object(mut file_payload)) => {
            file_payload.insert("url".into(), Value::String(mxc_url.to_string()));
            content.insert("file".into(), Value::Object(file_payload));
        }
        _ => {
            content.insert("url".into(), Value::String(mxc_url.to_string()));
        }
    }

    if is_voice {
        content.insert("org.matrix.msc3245.voice".into(), json!({}));
    }

    Value::Object(content)
}

// ─── Presence ───────────────────────────────────────────────────────────────

/// Port of the validity check in `set_presence`.
pub fn is_valid_presence_state(state: &str) -> bool {
    VALID_PRESENCE_STATES.contains(&state)
}

// ─── Reaction-approval bookkeeping ──────────────────────────────────────────

/// Port of Python `_MatrixApprovalPrompt`.
#[derive(Debug, Clone)]
pub struct MatrixApprovalPrompt {
    pub session_key: String,
    pub chat_id: String,
    pub message_id: String,
    pub resolved: bool,
    /// emoji → reaction event_id for the bot's own seed reactions.
    pub bot_reaction_events: std::collections::HashMap<String, String>,
}

impl MatrixApprovalPrompt {
    pub fn new(session_key: &str, chat_id: &str, message_id: &str) -> Self {
        MatrixApprovalPrompt {
            session_key: session_key.to_string(),
            chat_id: chat_id.to_string(),
            message_id: message_id.to_string(),
            resolved: false,
            bot_reaction_events: std::collections::HashMap::new(),
        }
    }
}

/// Map an approval reaction emoji to its choice, mirroring
/// `_approval_reaction_map`.
pub fn approval_reaction_choice(emoji: &str) -> Option<&'static str> {
    match emoji {
        "\u{2705}" => Some("once"), // ✅
        "\u{274e}" => Some("deny"), // ❎
        _ => None,
    }
}

/// Bookkeeping for pending reaction-based exec approvals, mirroring the two
/// adapter dicts (`_approval_prompts_by_event`, `_approval_prompt_by_session`).
#[derive(Default)]
pub struct ApprovalRegistry {
    by_event: std::collections::HashMap<String, MatrixApprovalPrompt>,
    by_session: std::collections::HashMap<String, String>,
}

impl ApprovalRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a fresh prompt, dropping any prior prompt for the same session
    /// (Python pops the old event before inserting).
    pub fn register(&mut self, prompt: MatrixApprovalPrompt) {
        let session_key = prompt.session_key.clone();
        let message_id = prompt.message_id.clone();
        if let Some(old_event) = self.by_session.get(&session_key).cloned() {
            self.by_event.remove(&old_event);
        }
        self.by_event.insert(message_id.clone(), prompt);
        self.by_session.insert(session_key, message_id);
    }

    pub fn get_by_event(&self, event_id: &str) -> Option<&MatrixApprovalPrompt> {
        self.by_event.get(event_id)
    }

    pub fn get_by_event_mut(&mut self, event_id: &str) -> Option<&mut MatrixApprovalPrompt> {
        self.by_event.get_mut(event_id)
    }

    /// Resolve and remove the prompt for `event_id`, mirroring the post-resolve
    /// pops in `_on_reaction`. Returns the removed prompt (with `resolved=true`).
    pub fn resolve(&mut self, event_id: &str) -> Option<MatrixApprovalPrompt> {
        let mut prompt = self.by_event.remove(event_id)?;
        prompt.resolved = true;
        self.by_session.remove(&prompt.session_key);
        Some(prompt)
    }
}

// ─── Markdown → Matrix HTML (regex fallback) ────────────────────────────────

/// Reproduce Python's horizontal-rule regex `^[\s]*([-*_])\s*\1\s*\1[\s\-*_]*$`
/// (the `regex` crate lacks the `\1` backreference).
fn is_horizontal_rule(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0usize;
    // ^[\s]*
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    // ([-*_])
    if i >= chars.len() || !matches!(chars[i], '-' | '*' | '_') {
        return false;
    }
    let marker = chars[i];
    i += 1;
    // \s* \1
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() || chars[i] != marker {
        return false;
    }
    i += 1;
    // \s* \1
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() || chars[i] != marker {
        return false;
    }
    i += 1;
    // [\s\-*_]* $
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() || matches!(c, '-' | '*' | '_') {
            i += 1;
        } else {
            return false;
        }
    }
    true
}

fn protect_html(placeholders: &mut Vec<String>, fragment: String) -> String {
    let idx = placeholders.len();
    placeholders.push(fragment);
    format!("\u{0}PROTECTED{}\u{0}", idx)
}

/// Port of Python `_markdown_to_html_fallback`.
pub fn markdown_to_html_fallback(text: &str) -> String {
    let mut placeholders: Vec<String> = Vec::new();

    // Fenced code blocks: ```lang\n...\n```  (DOTALL)
    let fenced = Regex::new(r"(?s)```(\w*)\n(.*?)```").unwrap();
    let mut result = String::new();
    {
        let mut last = 0usize;
        for caps in fenced.captures_iter(text) {
            let whole = caps.get(0).unwrap();
            result.push_str(&text[last..whole.start()]);
            let lang = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let code = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            let frag = if !lang.is_empty() {
                format!(
                    "<pre><code class=\"language-{}\">{}</code></pre>",
                    html_escape(lang),
                    html_escape(code)
                )
            } else {
                format!("<pre><code>{}</code></pre>", html_escape(code))
            };
            result.push_str(&protect_html(&mut placeholders, frag));
            last = whole.end();
        }
        result.push_str(&text[last..]);
    }

    // Inline code: `code`
    {
        let inline = Regex::new(r"`([^`\n]+)`").unwrap();
        let mut next = String::new();
        let mut last = 0usize;
        for caps in inline.captures_iter(&result) {
            let whole = caps.get(0).unwrap();
            next.push_str(&result[last..whole.start()]);
            let code = caps.get(1).unwrap().as_str();
            let frag = format!("<code>{}</code>", html_escape(code));
            next.push_str(&protect_html(&mut placeholders, frag));
            last = whole.end();
        }
        next.push_str(&result[last..]);
        result = next;
    }

    // Markdown links: [text](url)
    {
        let links = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
        let mut next = String::new();
        let mut last = 0usize;
        for caps in links.captures_iter(&result) {
            let whole = caps.get(0).unwrap();
            next.push_str(&result[last..whole.start()]);
            let label = caps.get(1).unwrap().as_str();
            let url = caps.get(2).unwrap().as_str();
            let frag = format!(
                "<a href=\"{}\">{}</a>",
                sanitize_link_url(url),
                html_escape(label)
            );
            next.push_str(&protect_html(&mut placeholders, frag));
            last = whole.end();
        }
        next.push_str(&result[last..]);
        result = next;
    }

    // HTML-escape remaining text (split on PROTECTED sentinels).
    {
        let split_re = Regex::new(r"(\x00PROTECTED\d+\x00)").unwrap();
        let mut parts: Vec<String> = Vec::new();
        let mut last = 0usize;
        for m in split_re.find_iter(&result) {
            parts.push(result[last..m.start()].to_string());
            parts.push(m.as_str().to_string());
            last = m.end();
        }
        parts.push(result[last..].to_string());
        for part in parts.iter_mut() {
            if !part.starts_with("\u{0}PROTECTED") {
                *part = html_escape(part);
            }
        }
        result = parts.concat();
    }

    // Block-level transforms (line-oriented).
    let hdr_re = Regex::new(r"^(#{1,6})\s+(.+)$").unwrap();
    let ul_re = Regex::new(r"^[\s]*[-*+]\s+(.+)$").unwrap();
    let ol_re = Regex::new(r"^[\s]*\d+[.)]\s+(.+)$").unwrap();

    let lines: Vec<String> = result.split('\n').map(|s| s.to_string()).collect();
    let mut out_lines: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = &lines[i];

        if is_horizontal_rule(line) {
            out_lines.push("<hr>".to_string());
            i += 1;
            continue;
        }

        if let Some(caps) = hdr_re.captures(line) {
            let level = caps.get(1).unwrap().as_str().len();
            let body = caps.get(2).unwrap().as_str().trim();
            out_lines.push(format!("<h{0}>{1}</h{0}>", level, body));
            i += 1;
            continue;
        }

        if line.starts_with("&gt; ")
            || line == "&gt;"
            || line.starts_with("> ")
            || line == ">"
        {
            let mut bq_lines: Vec<String> = Vec::new();
            while i < lines.len()
                && (lines[i].starts_with("&gt; ")
                    || lines[i] == "&gt;"
                    || lines[i].starts_with("> ")
                    || lines[i] == ">")
            {
                let ln = &lines[i];
                if let Some(rest) = ln.strip_prefix("&gt; ") {
                    bq_lines.push(rest.to_string());
                } else if let Some(rest) = ln.strip_prefix("> ") {
                    bq_lines.push(rest.to_string());
                } else {
                    bq_lines.push(String::new());
                }
                i += 1;
            }
            out_lines.push(format!("<blockquote>{}</blockquote>", bq_lines.join("<br>")));
            continue;
        }

        if ul_re.is_match(line) {
            let mut items: Vec<String> = Vec::new();
            while i < lines.len() {
                if let Some(caps) = ul_re.captures(&lines[i]) {
                    items.push(caps.get(1).unwrap().as_str().to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            let li: String = items.iter().map(|it| format!("<li>{}</li>", it)).collect();
            out_lines.push(format!("<ul>{}</ul>", li));
            continue;
        }

        if ol_re.is_match(line) {
            let mut items: Vec<String> = Vec::new();
            while i < lines.len() {
                if let Some(caps) = ol_re.captures(&lines[i]) {
                    items.push(caps.get(1).unwrap().as_str().to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            let li: String = items.iter().map(|it| format!("<li>{}</li>", it)).collect();
            out_lines.push(format!("<ol>{}</ol>", li));
            continue;
        }

        out_lines.push(line.clone());
        i += 1;
    }

    result = out_lines.join("\n");

    // Inline transforms (DOTALL).
    let bold = Regex::new(r"(?s)\*\*(.+?)\*\*").unwrap();
    result = bold.replace_all(&result, "<strong>$1</strong>").to_string();
    let bold_us = Regex::new(r"(?s)__(.+?)__").unwrap();
    result = bold_us.replace_all(&result, "<strong>$1</strong>").to_string();
    let italic = Regex::new(r"(?s)\*(.+?)\*").unwrap();
    result = italic.replace_all(&result, "<em>$1</em>").to_string();
    let italic_us = Regex::new(r"(?s)(?P<pre>^|[^\w])_(.+?)_(?P<post>[^\w]|$)").unwrap();
    // Python uses lookarounds (?<!\w)_..._(?!\w); emulate by capturing &
    // re-emitting the boundary chars.
    result = replace_italic_underscore(&italic_us, &result);
    let strike = Regex::new(r"(?s)~~(.+?)~~").unwrap();
    result = strike.replace_all(&result, "<del>$1</del>").to_string();

    // Newline → <br>\n then unwrap block tags.
    let nl = Regex::new(r"\n").unwrap();
    result = nl.replace_all(&result, "<br>\n").to_string();
    let unwrap_open =
        Regex::new(r"<br>\n(</?(?:pre|blockquote|h[1-6]|ul|ol|li|hr))").unwrap();
    result = unwrap_open.replace_all(&result, "\n$1").to_string();
    let unwrap_close =
        Regex::new(r"(</(?:pre|blockquote|h[1-6]|ul|ol|li)>)<br>").unwrap();
    result = unwrap_close.replace_all(&result, "$1").to_string();

    // Restore protected regions.
    for (idx, original) in placeholders.iter().enumerate() {
        result = result.replace(&format!("\u{0}PROTECTED{}\u{0}", idx), original);
    }

    result
}

/// Emulate Python's `(?<!\w)_(.+?)_(?!\w)` italic substitution using captured
/// boundary characters.
fn replace_italic_underscore(re: &Regex, input: &str) -> String {
    let mut result = String::new();
    let mut last = 0usize;
    for caps in re.captures_iter(input) {
        let whole = caps.get(0).unwrap();
        let pre = caps.name("pre").map(|m| m.as_str()).unwrap_or("");
        let inner = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let post = caps.name("post").map(|m| m.as_str()).unwrap_or("");
        result.push_str(&input[last..whole.start()]);
        result.push_str(pre);
        result.push_str(&format!("<em>{}</em>", inner));
        result.push_str(post);
        last = whole.end();
    }
    result.push_str(&input[last..]);
    result
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_filename_heuristic() {
        assert!(looks_like_matrix_image_filename("photo.jpg"));
        assert!(looks_like_matrix_image_filename("IMG_1234.PNG"));
        assert!(looks_like_matrix_image_filename("art.avif"));
        // A caption with words is not a filename.
        assert!(!looks_like_matrix_image_filename("here is photo.jpg now"));
        assert!(!looks_like_matrix_image_filename("dir/photo.jpg")); // name != candidate
        assert!(!looks_like_matrix_image_filename("notes.txt"));
        assert!(!looks_like_matrix_image_filename(""));
        assert!(!looks_like_matrix_image_filename("line1\nline2.png"));
        assert!(!looks_like_matrix_image_filename("trailing/"));
        assert!(!looks_like_matrix_image_filename("noext"));
    }

    #[test]
    fn mxc_conversion() {
        assert_eq!(
            mxc_to_http("https://hs.example", "mxc://hs.example/abcd"),
            "https://hs.example/_matrix/client/v1/media/download/hs.example/abcd"
        );
        assert_eq!(mxc_to_http("https://hs", "https://other"), "https://other");
    }

    #[test]
    fn html_escaping() {
        assert_eq!(
            html_escape("a & b <c> \"d\" 'e'"),
            "a &amp; b &lt;c&gt; &quot;d&quot; &#x27;e&#x27;"
        );
    }

    #[test]
    fn link_sanitization() {
        assert_eq!(sanitize_link_url("javascript:alert(1)"), "");
        assert_eq!(sanitize_link_url("DATA:text/html,x"), "");
        assert_eq!(sanitize_link_url("https://x.com"), "https://x.com");
        assert_eq!(sanitize_link_url("https://x.com/\"q"), "https://x.com/&quot;q");
    }

    #[test]
    fn outbound_mention_extraction() {
        let text = "hi @bob:example.org and @bob:example.org again, code `@no:body` and [a](b)";
        let mentions = extract_outbound_mentions(text);
        assert_eq!(mentions, vec!["@bob:example.org".to_string()]);
    }

    #[test]
    fn outbound_mention_injection() {
        let text = "ping @bob:example.org";
        let out = inject_outbound_mention_links(text);
        assert_eq!(
            out,
            "ping [@bob:example.org](https://matrix.to/#/@bob:example.org)"
        );
        // Mentions inside code spans are left literal.
        let code = "see `@bob:example.org`";
        assert_eq!(inject_outbound_mention_links(code), code);
    }

    #[test]
    fn bot_mention_detection() {
        let uid = "@hermes:example.org";
        assert!(is_bot_mentioned(uid, "hey @hermes:example.org", None, None));
        // localpart word boundary
        assert!(is_bot_mentioned(uid, "hello hermes!", None, None));
        // not mentioned
        assert!(!is_bot_mentioned(uid, "totally unrelated", None, None));
        // authoritative m.mentions
        let ids = vec![uid.to_string()];
        assert!(is_bot_mentioned(uid, "", None, Some(&ids)));
        // formatted_body pill
        assert!(is_bot_mentioned(
            uid,
            "click",
            Some("<a href=\"https://matrix.to/#/@hermes:example.org\">x</a>"),
            None
        ));
    }

    #[test]
    fn strip_mention_keeps_words() {
        let uid = "@hermes:example.org";
        // Full MXID stripped.
        assert_eq!(strip_mention(uid, "hi @hermes:example.org there"), "hi there");
        // @localpart stripped.
        assert_eq!(strip_mention(uid, "@hermes do this"), "do this");
        // Bare word "Hermes Agent" NOT mangled into "Agent".
        assert_eq!(strip_mention(uid, "Hermes Agent"), "Hermes Agent");
    }

    #[test]
    fn self_and_system_senders() {
        assert!(is_self_sender("@bot:srv", "@BOT:SRV"));
        assert!(is_self_sender("", "@anyone:srv")); // defensive
        assert!(!is_self_sender("@bot:srv", "@user:srv"));

        assert!(is_system_or_bridge_sender("@_telegram_1:srv"));
        assert!(is_system_or_bridge_sender("@:srv"));
        assert!(is_system_or_bridge_sender(":srv"));
        assert!(is_system_or_bridge_sender(""));
        assert!(!is_system_or_bridge_sender("@alice:srv"));
    }

    #[test]
    fn display_name_fallback_strips() {
        assert_eq!(display_name_fallback("@alice:srv"), "alice");
        assert_eq!(display_name_fallback("plain"), "plain");
    }

    #[test]
    fn reply_fallback_stripping() {
        let body = "> <@a:s> original\n> more quote\n\nactual reply";
        assert_eq!(strip_reply_fallback(body, true), "actual reply");
        // No reply → unchanged.
        assert_eq!(strip_reply_fallback("hello", false), "hello");
        // reply_to but body has no fallback marker → unchanged.
        assert_eq!(strip_reply_fallback("hello", true), "hello");
    }

    #[test]
    fn startup_grace() {
        assert!(is_before_startup_grace(100.0, 110.0)); // 100 < 110 - 5
        assert!(!is_before_startup_grace(108.0, 110.0)); // within grace
        assert!(!is_before_startup_grace(0.0, 110.0)); // zero never dropped
    }

    #[test]
    fn dedup_bounded() {
        let mut d = EventDedup::new(2);
        assert!(!d.is_duplicate("a"));
        assert!(d.is_duplicate("a"));
        assert!(!d.is_duplicate("b"));
        assert!(!d.is_duplicate("c")); // evicts "a"
        assert!(!d.is_duplicate("a")); // "a" no longer remembered
        assert!(!d.is_duplicate("")); // empty never dedup'd
    }

    #[test]
    fn text_batch_delays() {
        let cfg = TextBatchConfig {
            delay_seconds: 0.6,
            split_delay_seconds: 2.0,
        };
        assert!(cfg.enabled());
        assert_eq!(cfg.delay_for(10), 0.6);
        assert_eq!(cfg.delay_for(SPLIT_THRESHOLD), 2.0);
        let off = TextBatchConfig {
            delay_seconds: 0.0,
            split_delay_seconds: 2.0,
        };
        assert!(!off.enabled());
    }

    #[test]
    fn requirements_gate() {
        assert_eq!(
            check_matrix_requirements_from("", "", "", "", true),
            RequirementsCheck::MissingCredentials
        );
        assert_eq!(
            check_matrix_requirements_from("tok", "", "", "", true),
            RequirementsCheck::MissingHomeserver
        );
        assert_eq!(
            check_matrix_requirements_from("tok", "", "https://hs", "true", false),
            RequirementsCheck::EncryptionRequestedButUnavailable
        );
        assert!(
            check_matrix_requirements_from("tok", "", "https://hs", "false", false).is_ok()
        );
        assert!(
            check_matrix_requirements_from("tok", "", "https://hs", "true", true).is_ok()
        );
    }

    #[test]
    fn presence_validation() {
        assert!(is_valid_presence_state("online"));
        assert!(is_valid_presence_state("offline"));
        assert!(is_valid_presence_state("unavailable"));
        assert!(!is_valid_presence_state("away"));
    }

    #[test]
    fn approval_choices() {
        assert_eq!(approval_reaction_choice("\u{2705}"), Some("once"));
        assert_eq!(approval_reaction_choice("\u{274e}"), Some("deny"));
        assert_eq!(approval_reaction_choice("\u{1f600}"), None);
    }

    #[test]
    fn approval_registry_flow() {
        let mut reg = ApprovalRegistry::new();
        reg.register(MatrixApprovalPrompt::new("sess1", "!room", "$evt1"));
        assert!(reg.get_by_event("$evt1").is_some());
        // New prompt for same session drops old event mapping.
        reg.register(MatrixApprovalPrompt::new("sess1", "!room", "$evt2"));
        assert!(reg.get_by_event("$evt1").is_none());
        assert!(reg.get_by_event("$evt2").is_some());
        let resolved = reg.resolve("$evt2").unwrap();
        assert!(resolved.resolved);
        assert!(reg.get_by_event("$evt2").is_none());
    }

    #[test]
    fn text_content_with_mention() {
        let content = build_text_message_content("ping @bob:example.org", "m.text");
        assert_eq!(content["msgtype"], "m.text");
        assert_eq!(content["body"], "ping @bob:example.org");
        assert_eq!(content["m.mentions"]["user_ids"][0], "@bob:example.org");
        assert_eq!(content["format"], "org.matrix.custom.html");
        assert!(content["formatted_body"]
            .as_str()
            .unwrap()
            .contains("matrix.to/#/@bob:example.org"));
    }

    #[test]
    fn edit_content_shape() {
        let content = build_edit_message_content("hello", "$orig");
        assert_eq!(content["body"], "* hello");
        assert_eq!(content["m.relates_to"]["rel_type"], "m.replace");
        assert_eq!(content["m.relates_to"]["event_id"], "$orig");
        assert_eq!(content["m.new_content"]["body"], "hello");
    }

    #[test]
    fn reaction_content_shape() {
        let c = build_reaction_content("$e", "\u{2705}");
        assert_eq!(c["m.relates_to"]["rel_type"], "m.annotation");
        assert_eq!(c["m.relates_to"]["event_id"], "$e");
        assert_eq!(c["m.relates_to"]["key"], "\u{2705}");
    }

    #[test]
    fn relations_applied() {
        let mut c = build_text_message_content("hi", "m.text");
        apply_relations(&mut c, Some("$r"), Some("$t"));
        let rel = &c["m.relates_to"];
        assert_eq!(rel["rel_type"], "m.thread");
        assert_eq!(rel["event_id"], "$t");
        assert_eq!(rel["is_falling_back"], true);
        assert_eq!(rel["m.in_reply_to"]["event_id"], "$r");
    }

    #[test]
    fn media_content_plain_and_encrypted() {
        let plain = build_media_message_content(
            "m.image", "cap", "image/png", 42, "mxc://hs/x", None, false,
        );
        assert_eq!(plain["url"], "mxc://hs/x");
        assert_eq!(plain["info"]["size"], 42);
        assert!(plain.get("file").is_none());

        let enc_file = json!({"v": "v2", "key": {"k": "abc"}});
        let enc = build_media_message_content(
            "m.audio", "v", "audio/ogg", 7, "mxc://hs/y", Some(enc_file), true,
        );
        assert_eq!(enc["file"]["url"], "mxc://hs/y");
        assert!(enc.get("url").is_none());
        assert!(enc.get("org.matrix.msc3245.voice").is_some());
    }

    #[test]
    fn format_message_strips_image_md() {
        assert_eq!(format_message("see ![alt](http://x/y.png) ok"), "see http://x/y.png ok");
    }

    #[test]
    fn markdown_fallback_basics() {
        let html = markdown_to_html_fallback("**bold** and `code`");
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<code>code</code>"));

        let fenced = markdown_to_html_fallback("```py\nprint(1)\n```");
        assert!(fenced.contains("<pre><code class=\"language-py\">"));
        assert!(fenced.contains("print(1)"));

        let link = markdown_to_html_fallback("[x](https://y.com)");
        assert!(link.contains("<a href=\"https://y.com\">x</a>"));

        let xss = markdown_to_html_fallback("[x](javascript:alert(1))");
        assert!(xss.contains("<a href=\"\">x</a>"));

        let hdr = markdown_to_html_fallback("# Title");
        assert!(hdr.contains("<h1>Title</h1>"));

        let ul = markdown_to_html_fallback("- a\n- b");
        assert!(ul.contains("<ul><li>a</li><li>b</li></ul>"));

        // HTML in plain text gets escaped.
        let esc = markdown_to_html_fallback("a < b & c");
        assert!(esc.contains("a &lt; b &amp; c"));
    }

    #[test]
    fn markdown_fallback_blockquote() {
        let bq = markdown_to_html_fallback("> quoted\n> line2");
        assert!(bq.contains("<blockquote>quoted<br>line2</blockquote>"));
    }
}
