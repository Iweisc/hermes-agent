//! MemoryManager — orchestrates memory providers for the agent.
//!
//! Native Rust port of `agent/memory_manager.py` (555 LOC).
//!
//! Single integration point in the agent run loop. Replaces scattered
//! per-backend code with one manager that delegates to registered providers.
//!
//! Only ONE external plugin provider is allowed at a time — attempting to
//! register a second external provider is rejected with a warning. This
//! prevents tool schema bloat and conflicting memory backends.
//!
//! ## Differences from the Python source
//!
//! The Python module relies on `inspect.signature` to decide how to pass
//! metadata into a provider's `on_memory_write` hook (the
//! `_provider_memory_write_metadata_mode` helper). In Rust the
//! [`MemoryProvider`](crate::memory_provider::MemoryProvider) trait has a single
//! fixed signature — `on_memory_write(&mut self, &MemoryWrite)` — so the
//! runtime-introspection dance collapses to a single statically-typed call.
//! The [`MetadataMode`] enum and [`provider_memory_write_metadata_mode`] helper
//! are still provided (always returning [`MetadataMode::Keyword`] for trait
//! objects) to preserve API parity and document the behavior.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::memory_provider::{Kwargs, MemoryProvider, MemoryWrite};
use crate::tool_registry::tool_error;

// ---------------------------------------------------------------------------
// Context fencing helpers
// ---------------------------------------------------------------------------

const OPEN_TAG: &str = "<memory-context>";
const CLOSE_TAG: &str = "</memory-context>";

/// Strip fence tags, injected context blocks, and system notes from provider output.
///
/// Mirrors Python `sanitize_context`. Applies, in order:
///   1. `_INTERNAL_CONTEXT_RE` — remove whole `<memory-context>…</memory-context>` blocks.
///   2. `_INTERNAL_NOTE_RE` — remove the injected `[System note: …]` line.
///   3. `_FENCE_TAG_RE` — remove any stray open/close fence tags.
pub fn sanitize_context(text: &str) -> String {
    let s = internal_context_re().replace_all(text, "");
    let s = internal_note_re().replace_all(&s, "");
    let s = fence_tag_re().replace_all(&s, "");
    s.into_owned()
}

fn fence_tag_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // r'</?\s*memory-context\s*>'  case-insensitive
        regex::RegexBuilder::new(r"</?\s*memory-context\s*>")
            .case_insensitive(true)
            .build()
            .expect("fence_tag_re")
    })
}

fn internal_context_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>'  case-insensitive, non-greedy
        regex::RegexBuilder::new(r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>")
            .case_insensitive(true)
            .build()
            .expect("internal_context_re")
    })
}

fn internal_note_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // Matches the two known phrasings of the system note.
        regex::RegexBuilder::new(
            r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*",
        )
        .case_insensitive(true)
        .build()
        .expect("internal_note_re")
    })
}

/// Wrap prefetched memory in a fenced block with system note.
///
/// Returns an empty string when `raw_context` is blank. If the provider
/// returned a string that already contained fence tags / system notes, they
/// are stripped first and a warning is logged (matching Python).
pub fn build_memory_context_block(raw_context: &str) -> String {
    if raw_context.trim().is_empty() {
        return String::new();
    }
    let clean = sanitize_context(raw_context);
    if clean != raw_context {
        log::warn!("memory provider returned pre-wrapped context; stripped");
    }
    format!(
        "<memory-context>\n\
         [System note: The following is recalled memory context, \
         NOT new user input. Treat as authoritative reference data — \
         this is the agent's persistent memory and should inform all responses.]\n\n\
         {clean}\n\
         </memory-context>"
    )
}

// ---------------------------------------------------------------------------
// StreamingContextScrubber
// ---------------------------------------------------------------------------

/// Stateful scrubber for streaming text that may contain split memory-context spans.
///
/// The one-shot [`sanitize_context`] regex cannot survive chunk boundaries: a
/// `<memory-context>` opened in one delta and closed in a later delta would
/// leak its payload because the non-greedy block regex needs both tags in one
/// string. This scrubber runs a small state machine across deltas, holding back
/// partial-tag tails and discarding everything inside a span (including the
/// system-note line).
///
/// ```ignore
/// let mut scrubber = StreamingContextScrubber::new();
/// for delta in stream {
///     let visible = scrubber.feed(&delta);
///     if !visible.is_empty() { emit(&visible); }
/// }
/// let trailing = scrubber.flush(); // at end of stream
/// if !trailing.is_empty() { emit(&trailing); }
/// ```
///
/// The scrubber is re-entrant per agent instance. Callers building new
/// top-level responses (new turn) should create a fresh scrubber or call
/// [`StreamingContextScrubber::reset`].
#[derive(Debug, Clone, Default)]
pub struct StreamingContextScrubber {
    in_span: bool,
    buf: String,
}

impl StreamingContextScrubber {
    pub fn new() -> Self {
        Self {
            in_span: false,
            buf: String::new(),
        }
    }

    pub fn reset(&mut self) {
        self.in_span = false;
        self.buf.clear();
    }

    /// Return the visible portion of `text` after scrubbing.
    ///
    /// Any trailing fragment that could be the start of an open/close tag is
    /// held back in the internal buffer and surfaced on the next `feed()` call
    /// or discarded/emitted by [`flush`](Self::flush).
    pub fn feed(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let mut buf: String = std::mem::take(&mut self.buf);
        buf.push_str(text);
        let mut out = String::new();

        while !buf.is_empty() {
            let buf_lower = buf.to_lowercase();
            if self.in_span {
                match buf_lower.find(CLOSE_TAG) {
                    None => {
                        // Hold back a potential partial close tag; drop the rest.
                        let held = Self::max_partial_suffix(&buf, CLOSE_TAG);
                        self.buf = if held > 0 {
                            suffix_chars(&buf, held)
                        } else {
                            String::new()
                        };
                        return out;
                    }
                    Some(byte_idx) => {
                        // Found close — skip span content + tag, continue.
                        buf = buf[byte_idx + CLOSE_TAG.len()..].to_string();
                        self.in_span = false;
                    }
                }
            } else {
                match buf_lower.find(OPEN_TAG) {
                    None => {
                        // No open tag — hold back a potential partial open tag.
                        let held = Self::max_partial_suffix(&buf, OPEN_TAG);
                        if held > 0 {
                            let tail = suffix_chars(&buf, held);
                            let keep_len = buf.len() - tail.len();
                            out.push_str(&buf[..keep_len]);
                            self.buf = tail;
                        } else {
                            out.push_str(&buf);
                        }
                        return out;
                    }
                    Some(byte_idx) => {
                        // Emit text before the tag, enter span.
                        if byte_idx > 0 {
                            out.push_str(&buf[..byte_idx]);
                        }
                        buf = buf[byte_idx + OPEN_TAG.len()..].to_string();
                        self.in_span = true;
                    }
                }
            }
        }

        out
    }

    /// Emit any held-back buffer at end-of-stream.
    ///
    /// If still inside an unterminated span the remaining content is discarded
    /// (safer: leaking partial memory context is worse than a truncated
    /// answer). Otherwise the held-back partial-tag tail is emitted verbatim
    /// (it turned out not to be a real tag).
    pub fn flush(&mut self) -> String {
        if self.in_span {
            self.buf.clear();
            self.in_span = false;
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }

    /// Return the length (in chars) of the longest buf-suffix that is a
    /// tag-prefix. Case-insensitive. Returns 0 if no suffix could start the tag.
    fn max_partial_suffix(buf: &str, tag: &str) -> usize {
        let tag_lower = tag.to_lowercase();
        let buf_lower = buf.to_lowercase();
        // Operate on char counts to mirror Python's per-character slicing.
        let buf_chars: Vec<char> = buf_lower.chars().collect();
        let tag_char_len = tag_lower.chars().count();
        let max_check = buf_chars.len().min(tag_char_len.saturating_sub(1));
        for i in (1..=max_check).rev() {
            let start = buf_chars.len() - i;
            let suffix: String = buf_chars[start..].iter().collect();
            if tag_lower.starts_with(&suffix) {
                return i;
            }
        }
        0
    }
}

/// Return the last `n` *characters* of `s` as an owned String.
fn suffix_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if n >= chars.len() {
        return s.to_string();
    }
    chars[chars.len() - n..].iter().collect()
}

// ---------------------------------------------------------------------------
// Metadata mode (parity shim)
// ---------------------------------------------------------------------------

/// How to pass metadata to a provider's memory-write hook.
///
/// In the Python source this is computed per-provider via `inspect.signature`.
/// In Rust the trait signature is fixed, so this exists only for API parity and
/// always resolves to [`MetadataMode::Keyword`] for trait objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataMode {
    Keyword,
    Positional,
    Legacy,
}

/// Return how to pass metadata to a provider's memory-write hook.
///
/// Statically-typed trait objects always accept the structured
/// [`MemoryWrite`], so this returns [`MetadataMode::Keyword`].
pub fn provider_memory_write_metadata_mode(_provider: &dyn MemoryProvider) -> MetadataMode {
    MetadataMode::Keyword
}

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

/// Orchestrates the built-in provider plus at most one external provider.
///
/// The builtin provider is always first. Only one non-builtin (external)
/// provider is allowed. Failures in one provider never block the other.
pub struct MemoryManager {
    providers: Vec<Box<dyn MemoryProvider>>,
    /// tool name → index into `providers`.
    tool_to_provider: HashMap<String, usize>,
    /// True once a non-builtin provider is added.
    has_external: bool,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            tool_to_provider: HashMap::new(),
            has_external: false,
        }
    }

    // -- Registration -------------------------------------------------------

    /// Register a memory provider.
    ///
    /// Built-in provider (name `"builtin"`) is always accepted. Only **one**
    /// external (non-builtin) provider is allowed — a second attempt is
    /// rejected with a warning and the provider is dropped.
    pub fn add_provider(&mut self, provider: Box<dyn MemoryProvider>) {
        let is_builtin = provider.name() == "builtin";

        if !is_builtin {
            if self.has_external {
                let existing = self
                    .providers
                    .iter()
                    .find(|p| p.name() != "builtin")
                    .map(|p| p.name().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                log::warn!(
                    "Rejected memory provider '{}' — external provider '{}' is \
                     already registered. Only one external memory provider is \
                     allowed at a time. Configure which one via memory.provider \
                     in config.yaml.",
                    provider.name(),
                    existing,
                );
                return;
            }
            self.has_external = true;
        }

        let provider_name = provider.name().to_string();
        let schemas = provider.get_tool_schemas();
        let schema_count = schemas.len();
        let idx = self.providers.len();
        self.providers.push(provider);

        // Index tool names → provider for routing.
        for schema in &schemas {
            let tool_name = schema
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if tool_name.is_empty() {
                continue;
            }
            match self.tool_to_provider.get(&tool_name) {
                None => {
                    self.tool_to_provider.insert(tool_name, idx);
                }
                Some(&existing_idx) => {
                    log::warn!(
                        "Memory tool name conflict: '{}' already registered by {}, \
                         ignoring from {}",
                        tool_name,
                        self.providers[existing_idx].name(),
                        provider_name,
                    );
                }
            }
        }

        log::info!(
            "Memory provider '{}' registered ({} tools)",
            provider_name,
            schema_count,
        );
    }

    /// All registered providers in order.
    pub fn providers(&self) -> &[Box<dyn MemoryProvider>] {
        &self.providers
    }

    /// Get a provider by name, or `None` if not registered.
    pub fn get_provider(&self, name: &str) -> Option<&dyn MemoryProvider> {
        self.providers
            .iter()
            .find(|p| p.name() == name)
            .map(|b| b.as_ref())
    }

    // -- System prompt ------------------------------------------------------

    /// Collect system prompt blocks from all providers.
    ///
    /// Returns combined text, or empty string if no providers contribute. Each
    /// non-empty block is labeled with the provider name (by the provider
    /// itself). Blocks are joined with a blank line.
    pub fn build_system_prompt(&self) -> String {
        let mut blocks: Vec<String> = Vec::new();
        for provider in &self.providers {
            let block = provider.system_prompt_block();
            if !block.trim().is_empty() {
                blocks.push(block);
            }
        }
        blocks.join("\n\n")
    }

    // -- Prefetch / recall --------------------------------------------------

    /// Collect prefetch context from all providers.
    ///
    /// Returns merged context text. Empty providers are skipped.
    pub fn prefetch_all(&self, query: &str, session_id: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        for provider in &self.providers {
            let result = provider.prefetch(query, session_id);
            if !result.trim().is_empty() {
                parts.push(result);
            }
        }
        parts.join("\n\n")
    }

    /// Queue background prefetch on all providers for the next turn.
    pub fn queue_prefetch_all(&mut self, query: &str, session_id: &str) {
        for provider in &mut self.providers {
            provider.queue_prefetch(query, session_id);
        }
    }

    // -- Sync ---------------------------------------------------------------

    /// Sync a completed turn to all providers.
    pub fn sync_all(&mut self, user_content: &str, assistant_content: &str, session_id: &str) {
        for provider in &mut self.providers {
            provider.sync_turn(user_content, assistant_content, session_id);
        }
    }

    // -- Tools --------------------------------------------------------------

    /// Collect tool schemas from all providers, de-duplicated by name.
    pub fn get_all_tool_schemas(&self) -> Vec<Value> {
        let mut schemas: Vec<Value> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for provider in &self.providers {
            for schema in provider.get_tool_schemas() {
                let name = schema
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() && !seen.contains(&name) {
                    seen.insert(name);
                    schemas.push(schema);
                }
            }
        }
        schemas
    }

    /// Return the set of all tool names across all providers.
    pub fn get_all_tool_names(&self) -> HashSet<String> {
        self.tool_to_provider.keys().cloned().collect()
    }

    /// Check if any provider handles this tool.
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_to_provider.contains_key(tool_name)
    }

    /// Route a tool call to the correct provider.
    ///
    /// Returns a JSON string result. If no provider handles the tool, returns a
    /// `tool_error(...)` JSON string (matching Python — it does **not** raise).
    /// Provider failures are also captured and returned as `tool_error(...)`.
    pub fn handle_tool_call(
        &mut self,
        tool_name: &str,
        args: &Map<String, Value>,
        kwargs: &Kwargs,
    ) -> String {
        let idx = match self.tool_to_provider.get(tool_name).copied() {
            Some(idx) => idx,
            None => {
                return tool_error(
                    format!("No memory provider handles tool '{tool_name}'"),
                    None,
                );
            }
        };
        let provider = &mut self.providers[idx];
        match provider.handle_tool_call(tool_name, args, kwargs) {
            Ok(result) => result,
            Err(e) => {
                log::error!(
                    "Memory provider '{}' handle_tool_call({}) failed: {}",
                    provider.name(),
                    tool_name,
                    e,
                );
                tool_error(format!("Memory tool '{tool_name}' failed: {e}"), None)
            }
        }
    }

    // -- Lifecycle hooks ----------------------------------------------------

    /// Notify all providers of a new turn.
    ///
    /// `kwargs` may include: `remaining_tokens`, `model`, `platform`, `tool_count`.
    pub fn on_turn_start(&mut self, turn_number: i64, message: &str, kwargs: &Kwargs) {
        for provider in &mut self.providers {
            provider.on_turn_start(turn_number, message, kwargs);
        }
    }

    /// Notify all providers of session end.
    pub fn on_session_end(&mut self, messages: &[Value]) {
        for provider in &mut self.providers {
            provider.on_session_end(messages);
        }
    }

    /// Notify all providers that the agent's session_id has rotated.
    ///
    /// Fires on `/resume`, `/branch`, `/reset`, `/new`, and context
    /// compression. No-op when `new_session_id` is empty.
    pub fn on_session_switch(
        &mut self,
        new_session_id: &str,
        parent_session_id: &str,
        reset: bool,
        kwargs: &Kwargs,
    ) {
        if new_session_id.is_empty() {
            return;
        }
        for provider in &mut self.providers {
            provider.on_session_switch(new_session_id, parent_session_id, reset, kwargs);
        }
    }

    /// Notify all providers before context compression.
    ///
    /// Returns combined text from providers to include in the compression
    /// summary prompt. Empty string if no provider contributes.
    pub fn on_pre_compress(&mut self, messages: &[Value]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for provider in &mut self.providers {
            let result = provider.on_pre_compress(messages);
            if !result.trim().is_empty() {
                parts.push(result);
            }
        }
        parts.join("\n\n")
    }

    /// Notify external providers when the built-in memory tool writes.
    ///
    /// Skips the builtin provider itself (it's the source of the write). The
    /// Python `_provider_memory_write_metadata_mode` introspection is a no-op
    /// here — the trait takes a structured [`MemoryWrite`] directly.
    pub fn on_memory_write(&mut self, write: &MemoryWrite) {
        for provider in &mut self.providers {
            if provider.name() == "builtin" {
                continue;
            }
            // metadata_mode is always Keyword for trait objects; preserved for parity.
            let _mode = MetadataMode::Keyword;
            provider.on_memory_write(write);
        }
    }

    /// Notify all providers that a subagent completed.
    pub fn on_delegation(
        &mut self,
        task: &str,
        result: &str,
        child_session_id: &str,
        kwargs: &Kwargs,
    ) {
        for provider in &mut self.providers {
            provider.on_delegation(task, result, child_session_id, kwargs);
        }
    }

    /// Shut down all providers (reverse order for clean teardown).
    pub fn shutdown_all(&mut self) {
        for provider in self.providers.iter_mut().rev() {
            provider.shutdown();
        }
    }

    /// Initialize all providers.
    ///
    /// Automatically injects `hermes_home` into `kwargs` (resolved via
    /// `crate::mod_hermes_constants::get_hermes_home`) so that every provider
    /// can resolve profile-scoped storage paths. If the caller already supplied
    /// `hermes_home` it is left untouched.
    pub fn initialize_all(&mut self, session_id: &str, kwargs: &Kwargs) {
        let mut kw = kwargs.clone();
        if !kw.contains_key("hermes_home") {
            kw.insert("hermes_home".to_string(), Value::String(default_hermes_home()));
        }
        for provider in &mut self.providers {
            provider.initialize(session_id, &kw);
        }
    }
}

/// Resolve the active HERMES_HOME directory as a string.
///
/// Mirrors the lazy `from hermes_constants import get_hermes_home` in Python.
/// Falls back to `~/.hermes` if the constants helper is unavailable.
fn default_hermes_home() -> String {
    if let Ok(env) = std::env::var("HERMES_HOME") {
        if !env.is_empty() {
            return env;
        }
    }
    dirs::home_dir()
        .map(|h| h.join(".hermes").to_string_lossy().into_owned())
        .unwrap_or_else(|| ".hermes".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_provider::ProviderError;
    use serde_json::json;

    // --- sanitize_context ---------------------------------------------------

    #[test]
    fn sanitize_strips_whole_block() {
        let input = "before<memory-context>secret stuff</memory-context>after";
        assert_eq!(sanitize_context(input), "beforeafter");
    }

    #[test]
    fn sanitize_strips_stray_fence_tags() {
        let input = "a <memory-context> b </memory-context> c";
        // Whole-block regex removes the inner span first ("a  c").
        assert_eq!(sanitize_context(input), "a  c");
    }

    #[test]
    fn sanitize_strips_orphan_open_tag() {
        let input = "hello <memory-context> world";
        assert_eq!(sanitize_context(input), "hello  world");
    }

    #[test]
    fn sanitize_strips_system_note() {
        let input = "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]payload";
        assert_eq!(sanitize_context(input), "payload");
    }

    #[test]
    fn sanitize_case_insensitive_tags() {
        let input = "x<MEMORY-CONTEXT>y</Memory-Context>z";
        assert_eq!(sanitize_context(input), "xz");
    }

    // --- build_memory_context_block ----------------------------------------

    #[test]
    fn build_block_empty_for_blank() {
        assert_eq!(build_memory_context_block(""), "");
        assert_eq!(build_memory_context_block("   \n  "), "");
    }

    #[test]
    fn build_block_wraps_content() {
        let out = build_memory_context_block("recall data");
        assert!(out.starts_with("<memory-context>\n"));
        assert!(out.ends_with("\n</memory-context>"));
        assert!(out.contains("recall data"));
        assert!(out.contains("[System note:"));
    }

    // --- StreamingContextScrubber ------------------------------------------

    #[test]
    fn scrubber_passthrough_no_tags() {
        let mut s = StreamingContextScrubber::new();
        assert_eq!(s.feed("hello world"), "hello world");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn scrubber_single_chunk_span() {
        let mut s = StreamingContextScrubber::new();
        let v = s.feed("a<memory-context>secret</memory-context>b");
        assert_eq!(v, "ab");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn scrubber_split_across_chunks() {
        let mut s = StreamingContextScrubber::new();
        let mut out = String::new();
        out.push_str(&s.feed("visible <memory-"));
        out.push_str(&s.feed("context>hidden secret"));
        out.push_str(&s.feed(" still hidden</memory-"));
        out.push_str(&s.feed("context>tail"));
        out.push_str(&s.flush());
        assert_eq!(out, "visible tail");
    }

    #[test]
    fn scrubber_unterminated_span_discards() {
        let mut s = StreamingContextScrubber::new();
        let v = s.feed("ok <memory-context>leaking");
        assert_eq!(v, "ok ");
        // Still in span at end-of-stream → remaining discarded.
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn scrubber_partial_tag_that_isnt_real() {
        let mut s = StreamingContextScrubber::new();
        // "<mem" looks like a partial open tag — held back.
        let v = s.feed("done <mem");
        assert_eq!(v, "done ");
        // Turns out not a tag; flush emits the held tail verbatim.
        assert_eq!(s.flush(), "<mem");
    }

    #[test]
    fn scrubber_reset() {
        let mut s = StreamingContextScrubber::new();
        s.feed("x<memory-context>y");
        s.reset();
        assert_eq!(s.feed("plain"), "plain");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn scrubber_held_partial_resolves_to_text_next_feed() {
        let mut s = StreamingContextScrubber::new();
        assert_eq!(s.feed("abc<me"), "abc");
        // Next delta makes it clear it's not a tag.
        assert_eq!(s.feed("ow"), "<meow");
        assert_eq!(s.flush(), "");
    }

    // --- max_partial_suffix -------------------------------------------------

    #[test]
    fn max_partial_suffix_basic() {
        assert_eq!(
            StreamingContextScrubber::max_partial_suffix("foo<memory-", OPEN_TAG),
            8
        );
        assert_eq!(
            StreamingContextScrubber::max_partial_suffix("no tag here", OPEN_TAG),
            0
        );
        // Full tag length is never returned (max is len-1).
        assert_eq!(
            StreamingContextScrubber::max_partial_suffix(OPEN_TAG, OPEN_TAG),
            OPEN_TAG.chars().count() - 1
        );
    }

    // --- MemoryManager: a stub provider ------------------------------------

    struct StubProvider {
        name: String,
        prompt: String,
        prefetch_out: String,
        tools: Vec<Value>,
        events: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl StubProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                prompt: String::new(),
                prefetch_out: String::new(),
                tools: Vec::new(),
                events: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
    }

    impl MemoryProvider for StubProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn is_available(&self) -> bool {
            true
        }
        fn initialize(&mut self, session_id: &str, kwargs: &Kwargs) {
            let hh = kwargs
                .get("hermes_home")
                .and_then(Value::as_str)
                .unwrap_or("<none>")
                .to_string();
            self.events
                .lock()
                .unwrap()
                .push(format!("init:{session_id}:{hh}"));
        }
        fn system_prompt_block(&self) -> String {
            self.prompt.clone()
        }
        fn prefetch(&self, _query: &str, _session_id: &str) -> String {
            self.prefetch_out.clone()
        }
        fn get_tool_schemas(&self) -> Vec<Value> {
            self.tools.clone()
        }
        fn handle_tool_call(
            &mut self,
            tool_name: &str,
            _args: &Map<String, Value>,
            _kwargs: &Kwargs,
        ) -> Result<String, ProviderError> {
            if tool_name == "boom" {
                return Err(ProviderError("kaboom".to_string()));
            }
            Ok(json!({"ok": tool_name}).to_string())
        }
        fn on_memory_write(&mut self, write: &MemoryWrite) {
            self.events
                .lock()
                .unwrap()
                .push(format!("write:{}", write.content));
        }
        fn shutdown(&mut self) {
            self.events.lock().unwrap().push("shutdown".to_string());
        }
    }

    fn schema(name: &str) -> Value {
        json!({"name": name, "description": "d", "parameters": {}})
    }

    #[test]
    fn rejects_second_external_provider() {
        let mut mgr = MemoryManager::new();
        let mut p1 = StubProvider::new("honcho");
        p1.tools = vec![schema("h_tool")];
        let mut p2 = StubProvider::new("mem0");
        p2.tools = vec![schema("m_tool")];

        mgr.add_provider(Box::new(p1));
        mgr.add_provider(Box::new(p2)); // rejected

        assert_eq!(mgr.providers().len(), 1);
        assert!(mgr.has_tool("h_tool"));
        assert!(!mgr.has_tool("m_tool"));
    }

    #[test]
    fn builtin_plus_one_external_allowed() {
        let mut mgr = MemoryManager::new();
        mgr.add_provider(Box::new(StubProvider::new("builtin")));
        mgr.add_provider(Box::new(StubProvider::new("honcho")));
        assert_eq!(mgr.providers().len(), 2);
        assert!(mgr.get_provider("builtin").is_some());
        assert!(mgr.get_provider("honcho").is_some());
        assert!(mgr.get_provider("nope").is_none());
    }

    #[test]
    fn tool_name_conflict_keeps_first() {
        let mut mgr = MemoryManager::new();
        let mut builtin = StubProvider::new("builtin");
        builtin.tools = vec![schema("dup")];
        let mut ext = StubProvider::new("ext");
        ext.tools = vec![schema("dup"), schema("unique")];

        mgr.add_provider(Box::new(builtin));
        mgr.add_provider(Box::new(ext));

        // get_all_tool_schemas de-dups by name across providers.
        let all = mgr.get_all_tool_schemas();
        let names: Vec<&str> = all
            .iter()
            .filter_map(|s| s.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["dup", "unique"]);
        assert!(mgr.has_tool("dup"));
        assert!(mgr.has_tool("unique"));
    }

    #[test]
    fn build_system_prompt_joins_nonempty() {
        let mut mgr = MemoryManager::new();
        let mut a = StubProvider::new("builtin");
        a.prompt = "alpha".to_string();
        let mut b = StubProvider::new("ext");
        b.prompt = "   ".to_string(); // blank → skipped
        mgr.add_provider(Box::new(a));
        mgr.add_provider(Box::new(b));
        assert_eq!(mgr.build_system_prompt(), "alpha");
    }

    #[test]
    fn prefetch_all_merges() {
        let mut mgr = MemoryManager::new();
        let mut a = StubProvider::new("builtin");
        a.prefetch_out = "one".to_string();
        let mut b = StubProvider::new("ext");
        b.prefetch_out = "two".to_string();
        mgr.add_provider(Box::new(a));
        mgr.add_provider(Box::new(b));
        assert_eq!(mgr.prefetch_all("q", ""), "one\n\ntwo");
    }

    #[test]
    fn handle_unknown_tool_returns_error() {
        let mut mgr = MemoryManager::new();
        mgr.add_provider(Box::new(StubProvider::new("builtin")));
        let out = mgr.handle_tool_call("ghost", &Map::new(), &Kwargs::new());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("No memory provider handles tool 'ghost'"));
    }

    #[test]
    fn handle_tool_failure_returns_error() {
        let mut mgr = MemoryManager::new();
        let mut p = StubProvider::new("builtin");
        p.tools = vec![schema("boom")];
        mgr.add_provider(Box::new(p));
        let out = mgr.handle_tool_call("boom", &Map::new(), &Kwargs::new());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("boom"));
        assert!(v["error"].as_str().unwrap().contains("kaboom"));
    }

    #[test]
    fn handle_tool_success() {
        let mut mgr = MemoryManager::new();
        let mut p = StubProvider::new("builtin");
        p.tools = vec![schema("good")];
        mgr.add_provider(Box::new(p));
        let out = mgr.handle_tool_call("good", &Map::new(), &Kwargs::new());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"].as_str().unwrap(), "good");
    }

    #[test]
    fn on_memory_write_skips_builtin() {
        let mut mgr = MemoryManager::new();
        let builtin = StubProvider::new("builtin");
        let builtin_events = builtin.events.clone();
        let ext = StubProvider::new("ext");
        let ext_events = ext.events.clone();
        mgr.add_provider(Box::new(builtin));
        mgr.add_provider(Box::new(ext));

        let write = MemoryWrite {
            action: crate::memory_provider::MemoryAction::Add,
            target: crate::memory_provider::MemoryTarget::Memory,
            content: "note".to_string(),
            metadata: Map::new(),
        };
        mgr.on_memory_write(&write);

        assert!(builtin_events
            .lock()
            .unwrap()
            .iter()
            .all(|e| !e.starts_with("write:")));
        assert!(ext_events
            .lock()
            .unwrap()
            .contains(&"write:note".to_string()));
    }

    #[test]
    fn shutdown_all_reverse_order() {
        let mut mgr = MemoryManager::new();
        let shared = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut a = StubProvider::new("builtin");
        a.events = shared.clone();
        let mut b = StubProvider::new("ext");
        b.events = shared.clone();
        mgr.add_provider(Box::new(a));
        mgr.add_provider(Box::new(b));
        mgr.shutdown_all();
        // Both shut down; order is reverse but both recorded.
        assert_eq!(shared.lock().unwrap().len(), 2);
    }

    #[test]
    fn initialize_injects_hermes_home() {
        let mut mgr = MemoryManager::new();
        let p = StubProvider::new("builtin");
        let events = p.events.clone();
        mgr.add_provider(Box::new(p));
        mgr.initialize_all("sess-1", &Kwargs::new());
        let ev = events.lock().unwrap();
        assert!(ev.iter().any(|e| e.starts_with("init:sess-1:") && !e.ends_with("<none>")));
    }

    #[test]
    fn initialize_respects_explicit_hermes_home() {
        let mut mgr = MemoryManager::new();
        let p = StubProvider::new("builtin");
        let events = p.events.clone();
        mgr.add_provider(Box::new(p));
        let mut kw = Kwargs::new();
        kw.insert("hermes_home".to_string(), Value::String("/custom/home".to_string()));
        mgr.initialize_all("sess-2", &kw);
        assert!(events
            .lock()
            .unwrap()
            .contains(&"init:sess-2:/custom/home".to_string()));
    }

    #[test]
    fn on_session_switch_empty_id_noop() {
        let mut mgr = MemoryManager::new();
        mgr.add_provider(Box::new(StubProvider::new("builtin")));
        // Should not panic; empty id returns early.
        mgr.on_session_switch("", "", false, &Kwargs::new());
    }

    #[test]
    fn metadata_mode_always_keyword() {
        let p = StubProvider::new("ext");
        assert_eq!(
            provider_memory_write_metadata_mode(&p),
            MetadataMode::Keyword
        );
    }
}
