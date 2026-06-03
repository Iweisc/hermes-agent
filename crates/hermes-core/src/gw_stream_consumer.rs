//! Gateway streaming consumer — bridges sync agent callbacks to async platform delivery.
//!
//! Native Rust port of `gateway/stream_consumer.py`.
//!
//! The agent fires `stream_delta_callback(text)` synchronously from its worker
//! thread. [`GatewayStreamConsumer`]:
//!   1. Receives deltas via [`GatewayStreamConsumer::on_delta`] (thread-safe, sync)
//!   2. Queues them via an [`std::sync::mpsc`] channel
//!   3. The async [`GatewayStreamConsumer::run`] task buffers, rate-limits, and
//!      progressively edits a single message on the target platform.
//!
//! Design: uses the edit transport (send initial message, then `edit_message`).
//! This is universally supported across Telegram, Discord, and Slack.
//!
//! Credit: jobless0x (#774, #1312), OutThisLife (#798), clicksingh (#697).

use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use regex::Regex;

/// Boxed future returned by async adapter methods. We hand-roll this instead of
/// pulling in the `async-trait` crate so the trait stays object-safe
/// (`Box<dyn StreamAdapter>`) while remaining dependency-free.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ── Constants ──────────────────────────────────────────────────────────────

/// After this many consecutive flood-control failures, permanently disable
/// progressive edits for the remainder of the stream.
const MAX_FLOOD_STRIKES: u32 = 3;

/// Guard against creating a standalone message whose only visible content is a
/// handful of characters alongside the streaming cursor.
const MIN_NEW_MSG_CHARS: usize = 4;

/// Sentinel message id used when a platform accepted a send but did not return
/// an editable message id (e.g. Signal, webhook github_comment delivery).
pub const NO_EDIT_SENTINEL: &str = "__no_edit__";

/// Reasoning/thinking tags that models emit inline in content.
/// Must stay in sync with `cli.py` `_OPEN_TAGS`/`_CLOSE_TAGS` and
/// `run_agent.py` `_strip_think_blocks()` tag variants.
pub const OPEN_THINK_TAGS: &[&str] = &[
    "<REASONING_SCRATCHPAD>",
    "<think>",
    "<reasoning>",
    "<THINKING>",
    "<thinking>",
    "<thought>",
];

pub const CLOSE_THINK_TAGS: &[&str] = &[
    "</REASONING_SCRATCHPAD>",
    "</think>",
    "</reasoning>",
    "</THINKING>",
    "</thinking>",
    "</thought>",
];

// ── Queue items ─────────────────────────────────────────────────────────────

/// Items placed on the consumer's internal queue by the sync callbacks.
#[derive(Debug, Clone)]
enum QueueItem {
    /// A text delta to accumulate.
    Text(String),
    /// Stream is complete.
    Done,
    /// Tool boundary — finalize current message, start a new one.
    NewSegment,
    /// A completed assistant commentary message between API/tool iterations.
    Commentary(String),
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Runtime config for a single stream consumer instance.
#[derive(Debug, Clone)]
pub struct StreamConsumerConfig {
    pub edit_interval: f64,
    pub buffer_threshold: usize,
    pub cursor: String,
    pub buffer_only: bool,
    /// When > 0, the final edit for a streamed response is delivered as a fresh
    /// message if the original preview has been visible for at least this many
    /// seconds. Makes the platform's visible timestamp reflect completion time
    /// instead of first-token time for long-running responses. Ported from
    /// openclaw/openclaw#72038. Default 0 = always edit in place (legacy).
    pub fresh_final_after_seconds: f64,
}

impl Default for StreamConsumerConfig {
    fn default() -> Self {
        StreamConsumerConfig {
            edit_interval: 1.0,
            buffer_threshold: 40,
            cursor: " ▉".to_string(),
            buffer_only: false,
            fresh_final_after_seconds: 0.0,
        }
    }
}

// ── Adapter abstraction ──────────────────────────────────────────────────────

/// Result of a platform send/edit operation. Mirrors the Python `SendResult`.
#[derive(Debug, Clone, Default)]
pub struct SendResult {
    pub success: bool,
    pub message_id: Option<String>,
    pub error: Option<String>,
}

impl SendResult {
    pub fn ok(message_id: Option<String>) -> Self {
        SendResult {
            success: true,
            message_id,
            error: None,
        }
    }

    pub fn fail(error: impl Into<String>) -> Self {
        SendResult {
            success: false,
            message_id: None,
            error: Some(error.into()),
        }
    }
}

/// Platform adapter the consumer drives. Concrete gateway adapters implement
/// this trait. Mirrors the duck-typed `adapter` object in Python.
///
/// Async methods return [`BoxFuture`] so the trait remains object-safe
/// (`Box<dyn StreamAdapter>`) without the `async-trait` crate.
pub trait StreamAdapter: Send + Sync {
    /// Maximum message length (Python: `MAX_MESSAGE_LENGTH`, default 4096).
    fn max_message_length(&self) -> usize {
        4096
    }

    /// Whether the adapter requires an explicit `finalize=true` edit to close
    /// out the streaming UI (Python: `REQUIRES_EDIT_FINALIZE`). Default false.
    fn requires_edit_finalize(&self) -> bool {
        false
    }

    /// Split a message into properly sized chunks with word/code-fence
    /// boundaries and chunk indicators like "(1/2)". Mirrors
    /// `adapter.truncate_message`.
    fn truncate_message(&self, text: &str, limit: usize) -> Vec<String> {
        GatewayStreamConsumer::split_text_chunks(text, limit)
    }

    /// Send a new message. `reply_to` threads to a previous message.
    fn send<'a>(
        &'a self,
        chat_id: &'a str,
        content: &'a str,
        reply_to: Option<&'a str>,
        metadata: Option<&'a serde_json::Value>,
    ) -> BoxFuture<'a, SendResult>;

    /// Edit an existing message.
    fn edit_message<'a>(
        &'a self,
        chat_id: &'a str,
        message_id: &'a str,
        content: &'a str,
        finalize: bool,
    ) -> BoxFuture<'a, SendResult>;

    /// Best-effort delete of a stale preview. Returns whether the platform
    /// supports deletion at all (Python checks `getattr(adapter,
    /// "delete_message", None)`). Default: not supported.
    fn supports_delete(&self) -> bool {
        false
    }

    /// Delete a message (only called when [`StreamAdapter::supports_delete`]).
    fn delete_message<'a>(
        &'a self,
        _chat_id: &'a str,
        _message_id: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }
}

/// Callback fired whenever a fresh content bubble is created on the platform.
pub type OnNewMessage = Box<dyn Fn() + Send + Sync>;

// ── Consumer ──────────────────────────────────────────────────────────────────

/// Async consumer that progressively edits a platform message with streamed
/// tokens.
pub struct GatewayStreamConsumer {
    adapter: Box<dyn StreamAdapter>,
    chat_id: String,
    cfg: StreamConsumerConfig,
    metadata: Option<serde_json::Value>,
    on_new_message: Option<OnNewMessage>,

    // Sender is cloneable and used by the sync callbacks; Receiver is drained
    // by `run`.
    tx: Sender<QueueItem>,
    rx: Mutex<Option<Receiver<QueueItem>>>,

    accumulated: String,
    message_id: Option<String>,
    message_created_ts: Option<Instant>,
    already_sent: bool,
    edit_supported: bool,
    last_edit_time: Option<Instant>,
    last_sent_text: String,
    fallback_final_send: bool,
    fallback_prefix: String,
    flood_strikes: u32,
    current_edit_interval: f64,
    final_response_sent: bool,
    adapter_requires_finalize: bool,

    // Think-block filter state.
    in_think_block: bool,
    think_buffer: String,
}

impl GatewayStreamConsumer {
    pub fn new(
        adapter: Box<dyn StreamAdapter>,
        chat_id: impl Into<String>,
        config: Option<StreamConsumerConfig>,
        metadata: Option<serde_json::Value>,
        on_new_message: Option<OnNewMessage>,
    ) -> Self {
        let cfg = config.unwrap_or_default();
        let (tx, rx) = mpsc::channel();
        let adapter_requires_finalize = adapter.requires_edit_finalize();
        let current_edit_interval = cfg.edit_interval;
        GatewayStreamConsumer {
            adapter,
            chat_id: chat_id.into(),
            cfg,
            metadata,
            on_new_message,
            tx,
            rx: Mutex::new(Some(rx)),
            accumulated: String::new(),
            message_id: None,
            message_created_ts: None,
            already_sent: false,
            edit_supported: true,
            last_edit_time: None,
            last_sent_text: String::new(),
            fallback_final_send: false,
            fallback_prefix: String::new(),
            flood_strikes: 0,
            current_edit_interval,
            final_response_sent: false,
            adapter_requires_finalize,
            in_think_block: false,
            think_buffer: String::new(),
        }
    }

    /// A cheap, cloneable handle for the sync callbacks. The agent's worker
    /// thread holds this to call [`DeltaSink::on_delta`] etc.
    pub fn sink(&self) -> DeltaSink {
        DeltaSink {
            tx: self.tx.clone(),
        }
    }

    /// True if at least one message was sent or edited during the run.
    pub fn already_sent(&self) -> bool {
        self.already_sent
    }

    /// True when the consumer delivered the final assistant reply.
    pub fn final_response_sent(&self) -> bool {
        self.final_response_sent
    }

    // ── Sync callbacks (also exposed via DeltaSink) ──────────────────────

    /// Finalize the current stream segment and start a fresh message.
    pub fn on_segment_break(&self) {
        let _ = self.tx.send(QueueItem::NewSegment);
    }

    /// Queue a completed interim assistant commentary message.
    pub fn on_commentary(&self, text: &str) {
        if !text.is_empty() {
            let _ = self.tx.send(QueueItem::Commentary(text.to_string()));
        }
    }

    /// Thread-safe callback called from the agent's worker thread.
    ///
    /// When `text` is `None`, signals a tool boundary: the current message is
    /// finalized and subsequent text is sent as a new message.
    pub fn on_delta(&self, text: Option<&str>) {
        match text {
            Some(t) if !t.is_empty() => {
                let _ = self.tx.send(QueueItem::Text(t.to_string()));
            }
            None => self.on_segment_break(),
            _ => {}
        }
    }

    /// Signal that the stream is complete.
    pub fn finish(&self) {
        let _ = self.tx.send(QueueItem::Done);
    }

    fn notify_new_message(&self) {
        if let Some(cb) = &self.on_new_message {
            cb();
        }
    }

    fn reset_segment_state(&mut self, preserve_no_edit: bool) {
        if preserve_no_edit && self.message_id.as_deref() == Some(NO_EDIT_SENTINEL) {
            return;
        }
        self.message_id = None;
        self.message_created_ts = None;
        self.accumulated.clear();
        self.last_sent_text.clear();
        self.fallback_final_send = false;
        self.fallback_prefix.clear();
    }

    // ── Think-block filtering ────────────────────────────────────────────

    /// Add a text delta to the accumulated buffer, suppressing think blocks.
    pub fn filter_and_accumulate(&mut self, text: &str) {
        let mut buf = std::mem::take(&mut self.think_buffer);
        buf.push_str(text);

        loop {
            if buf.is_empty() {
                return;
            }
            if self.in_think_block {
                // Look for the earliest closing tag.
                let mut best_idx: Option<usize> = None;
                let mut best_len = 0usize;
                for tag in CLOSE_THINK_TAGS {
                    if let Some(idx) = buf.find(tag) {
                        if best_idx.is_none() || idx < best_idx.unwrap() {
                            best_idx = Some(idx);
                            best_len = tag.len();
                        }
                    }
                }
                if best_len > 0 {
                    let idx = best_idx.unwrap();
                    self.in_think_block = false;
                    buf = buf[idx + best_len..].to_string();
                } else {
                    // No closing tag yet — hold tail that could be a partial
                    // closing tag prefix, discard the rest.
                    let max_tag = CLOSE_THINK_TAGS.iter().map(|t| t.len()).max().unwrap_or(0);
                    self.think_buffer = if buf.len() > max_tag {
                        tail_chars(&buf, max_tag)
                    } else {
                        buf.clone()
                    };
                    return;
                }
            } else {
                // Look for the earliest opening tag at a block boundary.
                let mut best_idx: Option<usize> = None;
                let mut best_len = 0usize;
                for tag in OPEN_THINK_TAGS {
                    let mut search_start = 0usize;
                    loop {
                        let found = buf[search_start..].find(tag).map(|i| i + search_start);
                        let idx = match found {
                            Some(i) => i,
                            None => break,
                        };
                        let is_boundary = if idx == 0 {
                            self.accumulated.is_empty() || self.accumulated.ends_with('\n')
                        } else {
                            let preceding = &buf[..idx];
                            match preceding.rfind('\n') {
                                None => {
                                    (self.accumulated.is_empty()
                                        || self.accumulated.ends_with('\n'))
                                        && preceding.trim().is_empty()
                                }
                                Some(last_nl) => preceding[last_nl + 1..].trim().is_empty(),
                            }
                        };
                        if is_boundary && (best_idx.is_none() || idx < best_idx.unwrap()) {
                            best_idx = Some(idx);
                            best_len = tag.len();
                            break; // first boundary hit for this tag is enough
                        }
                        search_start = idx + 1;
                    }
                }
                if best_len > 0 {
                    let idx = best_idx.unwrap();
                    self.accumulated.push_str(&buf[..idx]);
                    self.in_think_block = true;
                    buf = buf[idx + best_len..].to_string();
                } else {
                    // No opening tag — check for a partial tag at the tail.
                    let mut held_back = 0usize;
                    for tag in OPEN_THINK_TAGS {
                        let tag_chars: Vec<char> = tag.chars().collect();
                        for i in 1..tag_chars.len() {
                            let prefix: String = tag_chars[..i].iter().collect();
                            if buf.ends_with(&prefix) && prefix.len() > held_back {
                                held_back = prefix.len();
                            }
                        }
                    }
                    if held_back > 0 {
                        let split = buf.len() - held_back;
                        self.accumulated.push_str(&buf[..split]);
                        self.think_buffer = buf[split..].to_string();
                    } else {
                        self.accumulated.push_str(&buf);
                    }
                    return;
                }
            }
        }
    }

    /// Flush any held-back partial-tag buffer into accumulated text on stream
    /// end so trailing held-back text is not lost.
    fn flush_think_buffer(&mut self) {
        if !self.think_buffer.is_empty() && !self.in_think_block {
            let buf = std::mem::take(&mut self.think_buffer);
            self.accumulated.push_str(&buf);
        }
    }

    // ── Run loop ──────────────────────────────────────────────────────────

    /// Async task that drains the queue and edits the platform message.
    pub async fn run(&mut self) {
        let raw_limit = self.adapter.max_message_length();
        let safe_limit = std::cmp::max(
            500,
            raw_limit.saturating_sub(self.cfg.cursor.len() + 100),
        );

        // Take ownership of the receiver for this run.
        let rx = match self.rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => return, // already consumed
        };

        loop {
            // Drain all available items from the queue.
            let mut got_done = false;
            let mut got_segment_break = false;
            let mut commentary_text: Option<String> = None;
            loop {
                match rx.try_recv() {
                    Ok(QueueItem::Done) => {
                        got_done = true;
                        break;
                    }
                    Ok(QueueItem::NewSegment) => {
                        got_segment_break = true;
                        break;
                    }
                    Ok(QueueItem::Commentary(t)) => {
                        commentary_text = Some(t);
                        break;
                    }
                    Ok(QueueItem::Text(t)) => {
                        self.filter_and_accumulate(&t);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        // All senders dropped without a Done. Treat like done
                        // so the task can terminate (defensive — Python relies
                        // on finish() always being called).
                        got_done = true;
                        break;
                    }
                }
            }

            if got_done {
                self.flush_think_buffer();
            }

            // Decide whether to flush an edit.
            let now = Instant::now();
            let elapsed = match self.last_edit_time {
                Some(t) => now.duration_since(t).as_secs_f64(),
                // Python initializes _last_edit_time to 0.0 (epoch monotonic),
                // so the first elapsed is effectively very large => edit fires.
                None => f64::INFINITY,
            };
            let mut should_edit = got_done || got_segment_break || commentary_text.is_some();
            if !self.cfg.buffer_only {
                should_edit = should_edit
                    || (elapsed >= self.current_edit_interval && !self.accumulated.is_empty())
                    || self.accumulated.len() >= self.cfg.buffer_threshold;
            }

            let mut current_update_visible = false;
            if should_edit && !self.accumulated.is_empty() {
                // Split overflow: accumulated exceeds limit and no message to
                // edit yet (first message or after a segment break).
                if self.accumulated.len() > safe_limit && self.message_id.is_none() {
                    let chunks = self
                        .adapter
                        .truncate_message(&self.accumulated, safe_limit);
                    let reply_to = self.message_id.clone();
                    for chunk in chunks {
                        self.send_new_chunk(&chunk, reply_to.as_deref()).await;
                    }
                    self.accumulated.clear();
                    self.last_sent_text.clear();
                    self.last_edit_time = Some(Instant::now());
                    if got_done {
                        self.final_response_sent = self.already_sent;
                        return;
                    }
                    if got_segment_break {
                        self.message_id = None;
                        self.fallback_final_send = false;
                        self.fallback_prefix.clear();
                    }
                    continue;
                }

                // Existing message: edit it with the first chunk, then start a
                // new message for the overflow remainder.
                while self.accumulated.len() > safe_limit
                    && self.message_id.is_some()
                    && self.edit_supported
                {
                    let mut split_at = self.accumulated[..safe_limit].rfind('\n').unwrap_or(0);
                    if split_at < safe_limit / 2 {
                        split_at = safe_limit;
                    }
                    let chunk = self.accumulated[..split_at].to_string();
                    let ok = self.send_or_edit(&chunk, false).await;
                    if self.fallback_final_send || !ok {
                        break;
                    }
                    self.accumulated = self.accumulated[split_at..]
                        .trim_start_matches('\n')
                        .to_string();
                    self.message_id = None;
                    self.last_sent_text.clear();
                }

                let mut display_text = self.accumulated.clone();
                if !got_done && !got_segment_break && commentary_text.is_none() {
                    display_text.push_str(&self.cfg.cursor);
                }

                current_update_visible =
                    self.send_or_edit(&display_text, got_segment_break).await;
                self.last_edit_time = Some(Instant::now());
            }

            if got_done {
                if !self.accumulated.is_empty() {
                    if self.fallback_final_send {
                        let acc = self.accumulated.clone();
                        self.send_fallback_final(&acc).await;
                    } else if current_update_visible && !self.adapter_requires_finalize {
                        self.final_response_sent = true;
                    } else if self.message_id.is_some() {
                        let acc = self.accumulated.clone();
                        self.final_response_sent = self.send_or_edit(&acc, true).await;
                    } else if !self.already_sent {
                        let acc = self.accumulated.clone();
                        self.final_response_sent = self.send_or_edit(&acc, false).await;
                    }
                }
                return;
            }

            if let Some(text) = commentary_text {
                self.reset_segment_state(false);
                self.send_commentary(&text).await;
                self.last_edit_time = Some(Instant::now());
                self.reset_segment_state(false);
            }

            if got_segment_break {
                if !self.accumulated.is_empty()
                    && !current_update_visible
                    && self.message_id.is_some()
                    && self.message_id.as_deref() != Some(NO_EDIT_SENTINEL)
                {
                    self.flush_segment_tail_on_edit_failure().await;
                }
                self.reset_segment_state(true);
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Best-effort final edit, intended to be called when the run task is being
    /// cancelled. Mirrors the `asyncio.CancelledError` branch.
    pub async fn on_cancelled(&mut self) {
        let mut best_effort_ok = false;
        if !self.accumulated.is_empty() && self.message_id.is_some() {
            let acc = self.accumulated.clone();
            best_effort_ok = self.send_or_edit(&acc, false).await;
        }
        if best_effort_ok && !self.final_response_sent {
            self.final_response_sent = true;
        }
    }

    // ── Display cleanup ────────────────────────────────────────────────────

    /// Strip `MEDIA:<path>` directives and internal markers before display.
    pub fn clean_for_display(text: &str) -> String {
        if !text.contains("MEDIA:") && !text.contains("[[audio_as_voice]]") {
            return text.to_string();
        }
        let mut cleaned = text.replace("[[audio_as_voice]]", "");
        cleaned = media_re().replace_all(&cleaned, "").to_string();
        cleaned = blank_lines_re().replace_all(&cleaned, "\n\n").to_string();
        cleaned.trim_end().to_string()
    }

    async fn send_new_chunk(&mut self, text: &str, reply_to_id: Option<&str>) -> Option<String> {
        let text = Self::clean_for_display(text);
        if text.trim().is_empty() {
            return reply_to_id.map(|s| s.to_string());
        }
        let meta = self.metadata.clone();
        let result = self
            .adapter
            .send(&self.chat_id, &text, reply_to_id, meta.as_ref())
            .await;
        if result.success {
            if let Some(mid) = result.message_id {
                self.message_id = Some(mid.clone());
                self.already_sent = true;
                self.last_sent_text = text;
                self.notify_new_message();
                return Some(mid);
            }
        }
        self.edit_supported = false;
        reply_to_id.map(|s| s.to_string())
    }

    fn visible_prefix(&self) -> String {
        let mut prefix = self.last_sent_text.clone();
        if !self.cfg.cursor.is_empty() && prefix.ends_with(&self.cfg.cursor) {
            prefix.truncate(prefix.len() - self.cfg.cursor.len());
        }
        Self::clean_for_display(&prefix)
    }

    fn continuation_text(&self, final_text: &str) -> String {
        let prefix = if !self.fallback_prefix.is_empty() {
            self.fallback_prefix.clone()
        } else {
            self.visible_prefix()
        };
        if !prefix.is_empty() && final_text.starts_with(&prefix) {
            return final_text[prefix.len()..].trim_start().to_string();
        }
        final_text.to_string()
    }

    /// Split text into reasonably sized chunks for fallback sends.
    pub fn split_text_chunks(text: &str, limit: usize) -> Vec<String> {
        if text.len() <= limit {
            return vec![text.to_string()];
        }
        let mut chunks = Vec::new();
        let mut remaining = text.to_string();
        while remaining.len() > limit {
            let mut split_at = remaining[..limit].rfind('\n').unwrap_or(0);
            if split_at < limit / 2 {
                split_at = limit;
            }
            chunks.push(remaining[..split_at].to_string());
            remaining = remaining[split_at..].trim_start_matches('\n').to_string();
        }
        if !remaining.is_empty() {
            chunks.push(remaining);
        }
        chunks
    }

    async fn send_fallback_final(&mut self, text: &str) {
        let final_text = Self::clean_for_display(text);
        let mut continuation = self.continuation_text(&final_text);
        self.fallback_final_send = false;

        if continuation.trim().is_empty() {
            if !final_text.trim().is_empty() && final_text != self.visible_prefix() {
                continuation = final_text.clone();
            } else {
                // Defence-in-depth: strip a possibly-stuck cursor.
                if self.message_id.is_some()
                    && !self.last_sent_text.is_empty()
                    && !self.cfg.cursor.is_empty()
                    && self.last_sent_text.ends_with(&self.cfg.cursor)
                {
                    let clean_text = self.last_sent_text
                        [..self.last_sent_text.len() - self.cfg.cursor.len()]
                        .to_string();
                    let mid = self.message_id.clone().unwrap();
                    let result = self
                        .adapter
                        .edit_message(&self.chat_id, &mid, &clean_text, false)
                        .await;
                    if result.success {
                        self.last_sent_text = clean_text;
                    }
                }
                self.already_sent = true;
                self.final_response_sent = true;
                return;
            }
        }

        let raw_limit = self.adapter.max_message_length();
        let safe_limit = std::cmp::max(500, raw_limit.saturating_sub(100));
        let chunks = Self::split_text_chunks(&continuation, safe_limit);

        let mut last_message_id: Option<String> = None;
        let mut last_successful_chunk = String::new();
        let mut sent_any_chunk = false;
        let meta = self.metadata.clone();

        for chunk in &chunks {
            let mut result: Option<SendResult> = None;
            for attempt in 0..2 {
                let r = self
                    .adapter
                    .send(&self.chat_id, chunk, None, meta.as_ref())
                    .await;
                let success = r.success;
                let is_flood = Self::is_flood_error(&r);
                result = Some(r);
                if success {
                    break;
                }
                if attempt == 0 && is_flood {
                    tokio::time::sleep(Duration::from_secs_f64(3.0)).await;
                } else {
                    break;
                }
            }

            let success = result.as_ref().map(|r| r.success).unwrap_or(false);
            if !success {
                if sent_any_chunk {
                    self.already_sent = true;
                    self.final_response_sent = true;
                    self.message_id = last_message_id;
                    self.last_sent_text = last_successful_chunk;
                    self.fallback_prefix.clear();
                    return;
                }
                self.already_sent = false;
                self.message_id = None;
                self.last_sent_text.clear();
                self.fallback_prefix.clear();
                return;
            }
            sent_any_chunk = true;
            last_successful_chunk = chunk.clone();
            if let Some(mid) = result.and_then(|r| r.message_id) {
                last_message_id = Some(mid);
            }
            self.notify_new_message();
        }

        self.message_id = last_message_id;
        self.already_sent = true;
        self.final_response_sent = true;
        self.last_sent_text = chunks.last().cloned().unwrap_or_default();
        self.fallback_prefix.clear();
    }

    fn is_flood_error(result: &SendResult) -> bool {
        let err = result.error.clone().unwrap_or_default().to_lowercase();
        err.contains("flood") || err.contains("retry after") || err.contains("rate")
    }

    async fn flush_segment_tail_on_edit_failure(&mut self) {
        if !self.fallback_final_send {
            self.try_strip_cursor().await;
        }
        let visible = if !self.fallback_prefix.is_empty() {
            self.fallback_prefix.clone()
        } else {
            self.visible_prefix()
        };
        let mut tail = self.accumulated.clone();
        if !visible.is_empty() && tail.starts_with(&visible) {
            tail = tail[visible.len()..].trim_start().to_string();
        }
        tail = Self::clean_for_display(&tail);
        if tail.trim().is_empty() {
            return;
        }
        let meta = self.metadata.clone();
        let result = self
            .adapter
            .send(&self.chat_id, &tail, None, meta.as_ref())
            .await;
        if result.success {
            self.already_sent = true;
        }
    }

    async fn try_strip_cursor(&mut self) {
        match self.message_id.as_deref() {
            None => return,
            Some(NO_EDIT_SENTINEL) => return,
            Some(_) => {}
        }
        let prefix = self.visible_prefix();
        if prefix.trim().is_empty() {
            return;
        }
        let mid = self.message_id.clone().unwrap();
        let _result = self
            .adapter
            .edit_message(&self.chat_id, &mid, &prefix, false)
            .await;
        // Python sets _last_sent_text = prefix unconditionally after the await
        // (it's inside the try, before any failure can be observed since the
        // success isn't checked). Match: assignment happens regardless of
        // success as long as no exception. With SendResult we never "throw",
        // so assign unconditionally.
        self.last_sent_text = prefix;
    }

    async fn send_commentary(&mut self, text: &str) -> bool {
        let text = Self::clean_for_display(text);
        if text.trim().is_empty() {
            return false;
        }
        let meta = self.metadata.clone();
        let result = self
            .adapter
            .send(&self.chat_id, &text, None, meta.as_ref())
            .await;
        // Note: do NOT set already_sent = true here (interim status messages).
        if result.success {
            self.notify_new_message();
        }
        result.success
    }

    fn should_send_fresh_final(&self) -> bool {
        let threshold = self.cfg.fresh_final_after_seconds;
        if threshold <= 0.0 {
            return false;
        }
        match self.message_id.as_deref() {
            None => return false,
            Some(NO_EDIT_SENTINEL) => return false,
            Some(_) => {}
        }
        let created = match self.message_created_ts {
            Some(t) => t,
            None => return false,
        };
        Instant::now().duration_since(created).as_secs_f64() >= threshold
    }

    async fn try_fresh_final(&mut self, text: &str) -> bool {
        let old_message_id = self.message_id.clone();
        let meta = self.metadata.clone();
        let result = self
            .adapter
            .send(&self.chat_id, text, None, meta.as_ref())
            .await;
        if !result.success {
            return false;
        }
        // Best-effort delete of the stale preview.
        if let Some(ref old) = old_message_id {
            if old != NO_EDIT_SENTINEL && self.adapter.supports_delete() {
                self.adapter.delete_message(&self.chat_id, old).await;
            }
        }
        if let Some(new_id) = result.message_id {
            self.message_id = Some(new_id);
            self.message_created_ts = Some(Instant::now());
        } else {
            self.message_id = Some(NO_EDIT_SENTINEL.to_string());
            self.message_created_ts = None;
        }
        self.already_sent = true;
        self.last_sent_text = text.to_string();
        self.final_response_sent = true;
        true
    }

    /// Send or edit the streaming message. Returns true if the text was
    /// successfully delivered. `finalize` is true when this is the last edit.
    async fn send_or_edit(&mut self, text: &str, finalize: bool) -> bool {
        let text = Self::clean_for_display(text);

        let visible_without_cursor = if self.cfg.cursor.is_empty() {
            text.clone()
        } else {
            text.replace(&self.cfg.cursor, "")
        };
        let visible_stripped = visible_without_cursor.trim();
        if visible_stripped.is_empty() {
            return true; // cursor-only / whitespace-only update
        }
        if text.trim().is_empty() {
            return true;
        }

        // Guard against tiny standalone cursor-only first messages.
        if self.message_id.is_none()
            && !self.cfg.cursor.is_empty()
            && text.contains(&self.cfg.cursor)
            && visible_stripped.chars().count() < MIN_NEW_MSG_CHARS
        {
            return true;
        }

        if self.message_id.is_some() {
            if self.edit_supported {
                // Skip if identical to last sent (unless finalize required).
                if text == self.last_sent_text
                    && !(finalize && self.adapter_requires_finalize)
                {
                    return true;
                }
                // Fresh-final for long-lived previews.
                if finalize && self.should_send_fresh_final() {
                    if self.try_fresh_final(&text).await {
                        return true;
                    }
                }
                let mid = self.message_id.clone().unwrap();
                let result = self
                    .adapter
                    .edit_message(&self.chat_id, &mid, &text, finalize)
                    .await;
                if result.success {
                    self.already_sent = true;
                    self.last_sent_text = text;
                    self.flood_strikes = 0;
                    return true;
                }
                // Edit failed.
                if Self::is_flood_error(&result) {
                    self.flood_strikes += 1;
                    self.current_edit_interval =
                        (self.current_edit_interval * 2.0).min(10.0);
                    if self.flood_strikes < MAX_FLOOD_STRIKES {
                        self.last_edit_time = Some(Instant::now());
                        return false;
                    }
                }
                // Non-flood error OR strikes exhausted: enter fallback mode.
                self.fallback_prefix = self.visible_prefix();
                self.fallback_final_send = true;
                self.edit_supported = false;
                self.already_sent = true;
                self.try_strip_cursor().await;
                false
            } else {
                // Editing not supported — skip intermediate updates.
                false
            }
        } else {
            // First message — send new.
            let meta = self.metadata.clone();
            let result = self
                .adapter
                .send(&self.chat_id, &text, None, meta.as_ref())
                .await;
            if result.success {
                if let Some(mid) = result.message_id.clone() {
                    self.message_id = Some(mid);
                    self.message_created_ts = Some(Instant::now());
                } else {
                    self.edit_supported = false;
                }
                self.already_sent = true;
                self.last_sent_text = text.clone();
                if result.message_id.is_none() {
                    self.fallback_prefix = self.visible_prefix();
                    self.fallback_final_send = true;
                    self.message_id = Some(NO_EDIT_SENTINEL.to_string());
                }
                self.notify_new_message();
                true
            } else {
                self.edit_supported = false;
                false
            }
        }
    }
}

/// Cheap, cloneable, thread-safe handle used by the agent's worker thread to
/// push deltas without holding the consumer itself.
#[derive(Clone)]
pub struct DeltaSink {
    tx: Sender<QueueItem>,
}

impl DeltaSink {
    /// Push a text delta. `None` signals a tool boundary (segment break).
    pub fn on_delta(&self, text: Option<&str>) {
        match text {
            Some(t) if !t.is_empty() => {
                let _ = self.tx.send(QueueItem::Text(t.to_string()));
            }
            None => {
                let _ = self.tx.send(QueueItem::NewSegment);
            }
            _ => {}
        }
    }

    pub fn on_segment_break(&self) {
        let _ = self.tx.send(QueueItem::NewSegment);
    }

    pub fn on_commentary(&self, text: &str) {
        if !text.is_empty() {
            let _ = self.tx.send(QueueItem::Commentary(text.to_string()));
        }
    }

    pub fn finish(&self) {
        let _ = self.tx.send(QueueItem::Done);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Take the last `n` bytes of `s`, snapping to a char boundary so we never
/// slice mid-codepoint. Python slices by character; for the partial-tag tail
/// (ASCII tags) this matters only when multibyte content precedes the cut.
fn tail_chars(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut start = s.len() - n;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_string()
}

fn media_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"[`"']?MEDIA:\s*\S+[`"']?"#).unwrap())
}

fn blank_lines_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n{3,}").unwrap())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Recording adapter for tests.
    struct RecordingAdapter {
        sends: Arc<Mutex<Vec<String>>>,
        edits: Arc<Mutex<Vec<(String, bool)>>>,
        next_id: AtomicUsize,
        return_message_id: bool,
        max_len: usize,
        edit_fails_with: Option<String>,
    }

    impl RecordingAdapter {
        fn new() -> Self {
            RecordingAdapter {
                sends: Arc::new(Mutex::new(Vec::new())),
                edits: Arc::new(Mutex::new(Vec::new())),
                next_id: AtomicUsize::new(0),
                return_message_id: true,
                max_len: 4096,
                edit_fails_with: None,
            }
        }
    }

    impl StreamAdapter for RecordingAdapter {
        fn max_message_length(&self) -> usize {
            self.max_len
        }
        fn send<'a>(
            &'a self,
            _chat_id: &'a str,
            content: &'a str,
            _reply_to: Option<&'a str>,
            _metadata: Option<&'a serde_json::Value>,
        ) -> BoxFuture<'a, SendResult> {
            Box::pin(async move {
                self.sends.lock().unwrap().push(content.to_string());
                if self.return_message_id {
                    let n = self.next_id.fetch_add(1, Ordering::SeqCst);
                    SendResult::ok(Some(format!("msg_{}", n)))
                } else {
                    SendResult::ok(None)
                }
            })
        }
        fn edit_message<'a>(
            &'a self,
            _chat_id: &'a str,
            _message_id: &'a str,
            content: &'a str,
            finalize: bool,
        ) -> BoxFuture<'a, SendResult> {
            Box::pin(async move {
                self.edits
                    .lock()
                    .unwrap()
                    .push((content.to_string(), finalize));
                if let Some(ref err) = self.edit_fails_with {
                    return SendResult::fail(err.clone());
                }
                SendResult::ok(Some("edit".to_string()))
            })
        }
    }

    fn consumer_with(adapter: RecordingAdapter) -> GatewayStreamConsumer {
        GatewayStreamConsumer::new(Box::new(adapter), "chat1", None, None, None)
    }

    #[test]
    fn clean_for_display_strips_media() {
        let out = GatewayStreamConsumer::clean_for_display("Hi MEDIA:/tmp/a.png there");
        assert!(!out.contains("MEDIA:"));
        assert!(out.contains("Hi"));
        assert!(out.contains("there"));
    }

    #[test]
    fn clean_for_display_strips_audio_marker_and_collapses_blanks() {
        let out =
            GatewayStreamConsumer::clean_for_display("a[[audio_as_voice]]\n\n\n\nb\n\n  ");
        assert!(!out.contains("[[audio_as_voice]]"));
        assert!(out.contains("\n\n"));
        assert!(!out.contains("\n\n\n"));
        assert!(!out.ends_with(' '));
    }

    #[test]
    fn clean_for_display_passthrough() {
        let s = "plain text no markers";
        assert_eq!(GatewayStreamConsumer::clean_for_display(s), s);
    }

    #[test]
    fn think_block_suppressed() {
        // Tag at a block boundary (line start / after newline) is suppressed.
        let mut c = consumer_with(RecordingAdapter::new());
        c.filter_and_accumulate("Hello\n<think>secret reasoning</think> world");
        assert_eq!(c.accumulated, "Hello\n world");
    }

    #[test]
    fn think_block_at_start_suppressed() {
        let mut c = consumer_with(RecordingAdapter::new());
        c.filter_and_accumulate("<think>hidden</think>visible");
        assert_eq!(c.accumulated, "visible");
    }

    #[test]
    fn think_block_split_across_deltas() {
        // Opening tag at the very start of the stream (boundary), split across
        // deltas; closing tag also split.
        let mut c = consumer_with(RecordingAdapter::new());
        c.filter_and_accumulate("<thi");
        c.filter_and_accumulate("nk>hidden</thi");
        c.filter_and_accumulate("nk>B");
        assert_eq!(c.accumulated, "B");
    }

    #[test]
    fn think_tag_in_prose_not_suppressed() {
        // A <think> mid-line preceded by non-whitespace prose is not a boundary.
        let mut c = consumer_with(RecordingAdapter::new());
        c.filter_and_accumulate("the <think> tag is documented");
        assert_eq!(c.accumulated, "the <think> tag is documented");
    }

    #[test]
    fn flush_think_buffer_recovers_partial() {
        let mut c = consumer_with(RecordingAdapter::new());
        // Trailing partial-open-tag prefix held back.
        c.filter_and_accumulate("text<th");
        assert_eq!(c.accumulated, "text");
        assert_eq!(c.think_buffer, "<th");
        c.flush_think_buffer();
        assert_eq!(c.accumulated, "text<th");
    }

    #[test]
    fn split_text_chunks_short() {
        assert_eq!(
            GatewayStreamConsumer::split_text_chunks("hi", 100),
            vec!["hi".to_string()]
        );
    }

    #[test]
    fn split_text_chunks_splits_on_newline() {
        let text = format!("{}\n{}", "a".repeat(60), "b".repeat(60));
        let chunks = GatewayStreamConsumer::split_text_chunks(&text, 100);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a".repeat(60));
        assert_eq!(chunks[1], "b".repeat(60));
    }

    #[test]
    fn is_flood_error_detection() {
        assert!(GatewayStreamConsumer::is_flood_error(&SendResult::fail(
            "Flood control exceeded"
        )));
        assert!(GatewayStreamConsumer::is_flood_error(&SendResult::fail(
            "Too Many Requests: retry after 5"
        )));
        assert!(GatewayStreamConsumer::is_flood_error(&SendResult::fail(
            "rate limited"
        )));
        assert!(!GatewayStreamConsumer::is_flood_error(&SendResult::fail(
            "internal server error"
        )));
        assert!(!GatewayStreamConsumer::is_flood_error(&SendResult::ok(None)));
    }

    #[tokio::test]
    async fn basic_stream_sends_then_finalizes() {
        let adapter = RecordingAdapter::new();
        let sends = adapter.sends.clone();
        let edits = adapter.edits.clone();
        let mut c = consumer_with(adapter);
        let sink = c.sink();
        sink.on_delta(Some("Hello world this is a stream"));
        sink.finish();
        c.run().await;
        // First send happened.
        assert_eq!(sends.lock().unwrap().len(), 1);
        assert!(c.already_sent());
        assert!(c.final_response_sent());
        // Final delivered content has no cursor.
        let last = if !edits.lock().unwrap().is_empty() {
            edits.lock().unwrap().last().unwrap().0.clone()
        } else {
            sends.lock().unwrap().last().unwrap().clone()
        };
        assert!(!last.contains('▉'));
    }

    #[tokio::test]
    async fn no_edit_sentinel_when_platform_returns_no_id() {
        let mut adapter = RecordingAdapter::new();
        adapter.return_message_id = false;
        let mut c = consumer_with(adapter);
        let sink = c.sink();
        sink.on_delta(Some("A meaningful first message that is long enough"));
        sink.finish();
        c.run().await;
        assert!(c.already_sent());
        // Fallback mode delivered the final response.
        assert!(c.final_response_sent());
    }

    #[tokio::test]
    async fn commentary_does_not_set_already_sent_via_run_only() {
        let adapter = RecordingAdapter::new();
        let sends = adapter.sends.clone();
        let mut c = consumer_with(adapter);
        let sink = c.sink();
        sink.on_commentary("Using browser tool...");
        sink.finish();
        c.run().await;
        // Commentary was sent.
        assert_eq!(sends.lock().unwrap().len(), 1);
        // No accumulated content => not final.
        assert!(!c.final_response_sent());
    }

    #[tokio::test]
    async fn edit_flood_backoff_then_fallback() {
        let mut adapter = RecordingAdapter::new();
        adapter.edit_fails_with = Some("Flood control exceeded".to_string());
        let mut c = consumer_with(adapter);

        // Establish a first message.
        c.send_or_edit("Initial chunk of text here", false).await;
        assert!(c.message_id.is_some());
        assert_eq!(c.flood_strikes, 0);

        // Subsequent edits flood: first strikes back off, then fallback.
        let r1 = c.send_or_edit("Initial chunk of text here plus more", false).await;
        assert!(!r1);
        assert_eq!(c.flood_strikes, 1);
        assert!(c.edit_supported);

        c.send_or_edit("more 2", false).await;
        c.send_or_edit("more 3", false).await;
        // After MAX strikes, fallback mode engaged.
        assert!(c.fallback_final_send || !c.edit_supported);
    }

    #[test]
    fn config_defaults() {
        let cfg = StreamConsumerConfig::default();
        assert_eq!(cfg.edit_interval, 1.0);
        assert_eq!(cfg.buffer_threshold, 40);
        assert_eq!(cfg.cursor, " ▉");
        assert!(!cfg.buffer_only);
        assert_eq!(cfg.fresh_final_after_seconds, 0.0);
    }
}
