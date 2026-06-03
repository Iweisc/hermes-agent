//! Stateful scrubber for reasoning/thinking blocks in streamed assistant text.
//!
//! The regex-based whole-string stripper used elsewhere is correct for a
//! complete string, but when run *per-delta* it destroys the state that
//! downstream consumers rely on. Concretely, when a model streams
//!
//! ```text
//! delta1 = "<think>"
//! delta2 = "Let me check their config"
//! delta3 = "</think>"
//! ```
//!
//! a per-delta regex erases `delta1` entirely (unterminated-open at the
//! boundary matches `^<think>...`), so the downstream state machine never
//! sees the open tag, treats `delta2` as regular content, and leaks
//! reasoning to the user. Consumers that don't run their own state machine
//! never had any defence at all.
//!
//! This module centralises the tag-suppression state machine at the upstream
//! layer so every stream-delta callback sees text that has already had
//! reasoning blocks removed. Partial tags at delta boundaries are held back
//! until the next delta resolves them, and end-of-stream flushing surfaces
//! any held-back prose that turned out not to be a real tag.
//!
//! Usage:
//!
//! ```
//! use hermes_core::think_scrubber::StreamingThinkScrubber;
//! let mut scrubber = StreamingThinkScrubber::new();
//! let mut out = String::new();
//! for delta in ["<think>", "secret", "</think>", "hello"] {
//!     out.push_str(&scrubber.feed(delta));
//! }
//! out.push_str(&scrubber.flush());
//! assert_eq!(out, "hello");
//! ```
//!
//! The scrubber is re-entrant per agent instance. Call [`StreamingThinkScrubber::reset`]
//! at the top of each new turn so a hung block from an interrupted prior
//! stream cannot taint the next turn's output.
//!
//! Tag variants handled (case-insensitive): `<think>`, `<thinking>`,
//! `<reasoning>`, `<thought>`, `<REASONING_SCRATCHPAD>`.
//!
//! Block-boundary rule for opens: an opening tag is only treated as a
//! reasoning-block opener when it appears at the start of the stream, after a
//! newline (optionally followed by whitespace), or when only whitespace has
//! been emitted on the current line. This prevents prose that *mentions* the
//! tag name (e.g. `"use <think> tags here"`) from being incorrectly
//! suppressed. Closed pairs (`<think>X</think>`) are always suppressed
//! regardless of boundary; a closed pair is an intentional, bounded
//! construct.

/// The reasoning/thinking tag base names handled (case-insensitive).
const OPEN_TAG_NAMES: &[&str] = &[
    "think",
    "thinking",
    "reasoning",
    "thought",
    "REASONING_SCRATCHPAD",
];

/// Stateful scrubber for streaming reasoning/thinking blocks.
///
/// State machine:
///   - `in_block`: `true` while inside an opened block, waiting for a close
///     tag. All text inside is discarded.
///   - `buf`: held-back partial-tag tail. Emitted / discarded on the next
///     [`feed`](StreamingThinkScrubber::feed) call or by
///     [`flush`](StreamingThinkScrubber::flush).
///   - `last_emitted_ended_newline`: `true` iff the most recent emission to
///     the consumer ended with `\n`, or nothing has been emitted yet
///     (start-of-stream counts as a boundary). Used to decide whether an open
///     tag at buffer position 0 is at a block boundary.
pub struct StreamingThinkScrubber {
    in_block: bool,
    /// Held-back partial-tag tail, stored as chars for code-point indexing
    /// matching the Python implementation.
    buf: Vec<char>,
    last_emitted_ended_newline: bool,
    open_tags: Vec<Vec<char>>,
    close_tags: Vec<Vec<char>>,
    /// Lowercased forms, parallel to `open_tags` / `close_tags`.
    open_tags_lower: Vec<Vec<char>>,
    close_tags_lower: Vec<Vec<char>>,
    max_tag_len: usize,
}

impl Default for StreamingThinkScrubber {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingThinkScrubber {
    /// Construct a fresh scrubber in the start-of-stream state.
    pub fn new() -> Self {
        let open_tags: Vec<Vec<char>> = OPEN_TAG_NAMES
            .iter()
            .map(|name| format!("<{}>", name).chars().collect())
            .collect();
        let close_tags: Vec<Vec<char>> = OPEN_TAG_NAMES
            .iter()
            .map(|name| format!("</{}>", name).chars().collect())
            .collect();
        let open_tags_lower: Vec<Vec<char>> =
            open_tags.iter().map(|t| to_lower(t)).collect();
        let close_tags_lower: Vec<Vec<char>> =
            close_tags.iter().map(|t| to_lower(t)).collect();
        let max_tag_len = open_tags
            .iter()
            .chain(close_tags.iter())
            .map(|t| t.len())
            .max()
            .unwrap_or(0);
        Self {
            in_block: false,
            buf: Vec::new(),
            last_emitted_ended_newline: true,
            open_tags,
            close_tags,
            open_tags_lower,
            close_tags_lower,
            max_tag_len,
        }
    }

    /// Reset all state. Call at the top of every new turn.
    pub fn reset(&mut self) {
        self.in_block = false;
        self.buf.clear();
        self.last_emitted_ended_newline = true;
    }

    /// Feed one delta; return the scrubbed visible portion.
    ///
    /// May return an empty string when the entire delta is reasoning content
    /// or is being held back pending resolution of a partial tag at the
    /// boundary.
    pub fn feed(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        // buf = self._buf + text; self._buf = ""
        let mut buf: Vec<char> = Vec::with_capacity(self.buf.len() + text.len());
        buf.append(&mut self.buf); // drains self.buf, leaving it empty
        buf.extend(text.chars());
        let mut start: usize = 0; // index into buf representing remaining slice

        let mut out: Vec<char> = Vec::new();

        while start < buf.len() {
            let cur = &buf[start..];
            if self.in_block {
                // Hunt for the earliest close tag.
                let (close_idx, close_len) =
                    find_first_tag(cur, &self.close_tags, &self.close_tags_lower);
                if close_idx == usize::MAX {
                    // No close yet — hold back a potential partial close-tag
                    // prefix; discard everything else.
                    let held = self.max_partial_suffix(cur, &self.close_tags_lower);
                    self.buf = if held > 0 {
                        cur[cur.len() - held..].to_vec()
                    } else {
                        Vec::new()
                    };
                    return out.into_iter().collect();
                }
                // Found close: discard block content + tag, continue.
                start += close_idx + close_len;
                self.in_block = false;
            } else {
                // Priority 1 — closed <tag>X</tag> pair anywhere in buf.
                let pair = self.find_earliest_closed_pair(cur);
                // Priority 2 — unterminated open tag at a block boundary.
                let (open_idx, open_len) = self.find_open_at_boundary(cur, &out);

                // Pick whichever match comes earliest in the buffer.
                if let Some((p_start, p_end)) = pair {
                    if open_idx == usize::MAX || p_start <= open_idx {
                        let preceding = &cur[..p_start];
                        if !preceding.is_empty() {
                            let preceding = self.strip_orphan_close_tags(preceding);
                            if !preceding.is_empty() {
                                self.last_emitted_ended_newline =
                                    ends_with_newline(&preceding);
                                out.extend(preceding);
                            }
                        }
                        start += p_end;
                        continue;
                    }
                }

                if open_idx != usize::MAX {
                    // Unterminated open at boundary — emit preceding, enter
                    // block, continue loop with remainder.
                    let preceding = &cur[..open_idx];
                    if !preceding.is_empty() {
                        let preceding = self.strip_orphan_close_tags(preceding);
                        if !preceding.is_empty() {
                            self.last_emitted_ended_newline = ends_with_newline(&preceding);
                            out.extend(preceding);
                        }
                    }
                    self.in_block = true;
                    start += open_idx + open_len;
                    continue;
                }

                // No resolvable tag structure in buf. Hold back any
                // partial-tag prefix at the tail so a split tag across deltas
                // isn't missed, then emit the rest.
                let held_open = self.max_partial_suffix(cur, &self.open_tags_lower);
                let held_close = self.max_partial_suffix(cur, &self.close_tags_lower);
                let held = held_open.max(held_close);
                let emit_text: Vec<char>;
                if held > 0 {
                    emit_text = cur[..cur.len() - held].to_vec();
                    self.buf = cur[cur.len() - held..].to_vec();
                } else {
                    emit_text = cur.to_vec();
                    self.buf = Vec::new();
                }
                if !emit_text.is_empty() {
                    let emit_text = self.strip_orphan_close_tags(&emit_text);
                    if !emit_text.is_empty() {
                        self.last_emitted_ended_newline = ends_with_newline(&emit_text);
                        out.extend(emit_text);
                    }
                }
                return out.into_iter().collect();
            }
        }

        out.into_iter().collect()
    }

    /// End-of-stream flush.
    ///
    /// If still inside an unterminated block, held-back content is discarded —
    /// leaking partial reasoning is worse than a truncated answer. Otherwise
    /// the held-back partial-tag tail is emitted verbatim (it turned out not
    /// to be a real tag prefix).
    pub fn flush(&mut self) -> String {
        if self.in_block {
            self.buf.clear();
            self.in_block = false;
            return String::new();
        }
        let tail = std::mem::take(&mut self.buf);
        if tail.is_empty() {
            return String::new();
        }
        let tail = self.strip_orphan_close_tags(&tail);
        if !tail.is_empty() {
            self.last_emitted_ended_newline = ends_with_newline(&tail);
        }
        tail.into_iter().collect()
    }

    // ── internal helpers ───────────────────────────────────────────────

    /// Return `(start_idx, end_idx)` of the earliest closed pair, else `None`.
    ///
    /// A closed pair is `<tag>...</tag>` of any variant. Matches are
    /// case-insensitive and non-greedy (the closest close tag after an open
    /// tag wins). When two tag variants could both match, the one whose open
    /// tag appears earlier wins.
    fn find_earliest_closed_pair(&self, buf: &[char]) -> Option<(usize, usize)> {
        let buf_lower = to_lower(buf);
        let mut best: Option<(usize, usize)> = None;
        for (open_lower, close_lower) in
            self.open_tags_lower.iter().zip(self.close_tags_lower.iter())
        {
            let open_idx = match find_sub(&buf_lower, open_lower, 0) {
                Some(i) => i,
                None => continue,
            };
            let close_idx =
                match find_sub(&buf_lower, close_lower, open_idx + open_lower.len()) {
                    Some(i) => i,
                    None => continue,
                };
            let end_idx = close_idx + close_lower.len();
            if best.is_none() || open_idx < best.unwrap().0 {
                best = Some((open_idx, end_idx));
            }
        }
        best
    }

    /// Return the earliest block-boundary open-tag `(idx, len)`.
    ///
    /// Returns `(usize::MAX, 0)` if no boundary-legal opener is present.
    fn find_open_at_boundary(&self, buf: &[char], already_emitted: &[char]) -> (usize, usize) {
        let buf_lower = to_lower(buf);
        let mut best_idx = usize::MAX;
        let mut best_len = 0usize;
        for (tag, tag_lower) in self.open_tags.iter().zip(self.open_tags_lower.iter()) {
            let mut search_start = 0usize;
            loop {
                let idx = match find_sub(&buf_lower, tag_lower, search_start) {
                    Some(i) => i,
                    None => break,
                };
                if self.is_block_boundary(buf, idx, already_emitted) {
                    if best_idx == usize::MAX || idx < best_idx {
                        best_idx = idx;
                        best_len = tag.len();
                    }
                    break; // first boundary hit for this tag is enough
                }
                search_start = idx + 1;
            }
        }
        (best_idx, best_len)
    }

    /// `true` iff position `idx` in `buf` is a block boundary.
    ///
    /// A block boundary is:
    ///   - buf position 0 AND the most recent emission ended with a newline
    ///     (or nothing has been emitted yet)
    ///   - any position whose preceding text on the current line (since the
    ///     last newline in buf) is whitespace-only, AND if there is no newline
    ///     in the preceding buf portion, the most recent prior emission ended
    ///     with a newline
    fn is_block_boundary(&self, buf: &[char], idx: usize, already_emitted: &[char]) -> bool {
        if idx == 0 {
            // already_emitted is the accumulated `out` for this feed() call.
            // Python tracks it as a list of chunks and checks the last chunk;
            // here we flatten to chars, so "non-empty" + ends-with-newline is
            // equivalent for the boundary test.
            if !already_emitted.is_empty() {
                return ends_with_newline(already_emitted);
            }
            return self.last_emitted_ended_newline;
        }
        let preceding = &buf[..idx];
        let last_nl = rfind_char(preceding, '\n');
        match last_nl {
            None => {
                let prior_newline = if !already_emitted.is_empty() {
                    ends_with_newline(already_emitted)
                } else {
                    self.last_emitted_ended_newline
                };
                prior_newline && is_whitespace_only(preceding)
            }
            Some(nl) => is_whitespace_only(&preceding[nl + 1..]),
        }
    }

    /// Return the longest buf-suffix that is a prefix of any tag.
    ///
    /// Only prefixes strictly shorter than the tag itself count (full-length
    /// suffixes are the tag and are handled as matches, not held-back
    /// partials). Case-insensitive. `tags_lower` must be the lowercased tag
    /// forms.
    fn max_partial_suffix(&self, buf: &[char], tags_lower: &[Vec<char>]) -> usize {
        if buf.is_empty() {
            return 0;
        }
        let buf_lower = to_lower(buf);
        let max_check = buf_lower.len().min(self.max_tag_len.saturating_sub(1));
        let mut i = max_check;
        while i >= 1 {
            let suffix = &buf_lower[buf_lower.len() - i..];
            for tag_lower in tags_lower {
                if tag_lower.len() > i && tag_lower.starts_with(suffix) {
                    return i;
                }
            }
            i -= 1;
        }
        0
    }

    /// Remove any close tags from `text` (orphan-close handling).
    ///
    /// An orphan close tag has no matching open in the current scrubber state;
    /// it's always noise, stripped with any trailing whitespace so the
    /// surrounding prose flows naturally.
    fn strip_orphan_close_tags(&self, text: &[char]) -> Vec<char> {
        // Quick check for "</" anywhere.
        let has_open_close = text
            .windows(2)
            .any(|w| w[0] == '<' && w[1] == '/');
        if !has_open_close {
            return text.to_vec();
        }
        let text_lower = to_lower(text);
        let mut out: Vec<char> = Vec::with_capacity(text.len());
        let mut i = 0usize;
        while i < text.len() {
            let mut matched = false;
            if i + 2 <= text_lower.len()
                && text_lower[i] == '<'
                && text_lower[i + 1] == '/'
            {
                for tag_lower in &self.close_tags_lower {
                    let tag_len = tag_lower.len();
                    if i + tag_len <= text_lower.len()
                        && &text_lower[i..i + tag_len] == tag_lower.as_slice()
                    {
                        // Skip the tag and any trailing whitespace.
                        let mut j = i + tag_len;
                        while j < text.len() && matches!(text[j], ' ' | '\t' | '\n' | '\r') {
                            j += 1;
                        }
                        i = j;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                out.push(text[i]);
                i += 1;
            }
        }
        out
    }
}

// ── free-function helpers ───────────────────────────────────────────────

/// Lowercase a char slice. ASCII-affecting; mirrors Python `str.lower` for the
/// ASCII tag content. Uses Rust's Unicode-aware lowercasing for non-ASCII
/// (which only matters for content chars, never tag matching).
fn to_lower(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    for c in s {
        for lc in c.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

/// Return `(earliest_index, tag_length)` over `tags` (using parallel
/// `tags_lower` for matching), or `(usize::MAX, 0)`. Case-insensitive.
fn find_first_tag(
    buf: &[char],
    tags: &[Vec<char>],
    tags_lower: &[Vec<char>],
) -> (usize, usize) {
    let buf_lower = to_lower(buf);
    let mut best_idx = usize::MAX;
    let mut best_len = 0usize;
    for (tag, tag_lower) in tags.iter().zip(tags_lower.iter()) {
        if let Some(idx) = find_sub(&buf_lower, tag_lower, 0) {
            if best_idx == usize::MAX || idx < best_idx {
                best_idx = idx;
                best_len = tag.len();
            }
        }
    }
    (best_idx, best_len)
}

/// Find `needle` in `haystack` starting at `from`, returning the start index
/// (in char units). Equivalent to Python `str.find(sub, start)`.
fn find_sub(haystack: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from.min(haystack.len()));
    }
    if from > haystack.len() || needle.len() > haystack.len() {
        return None;
    }
    let last = haystack.len() - needle.len();
    let mut i = from;
    while i <= last {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Index of the last occurrence of `c`, like Python `str.rfind`.
fn rfind_char(s: &[char], c: char) -> Option<usize> {
    s.iter().rposition(|&x| x == c)
}

/// `true` if the slice contains only Python-`str.strip`-whitespace (i.e.
/// `char::is_whitespace`), matching `preceding.strip() == ""`.
fn is_whitespace_only(s: &[char]) -> bool {
    s.iter().all(|c| c.is_whitespace())
}

/// `true` if the slice ends with `\n`.
fn ends_with_newline(s: &[char]) -> bool {
    s.last() == Some(&'\n')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed all deltas then flush; return the concatenated visible output.
    fn run(deltas: &[&str]) -> String {
        let mut s = StreamingThinkScrubber::new();
        let mut out = String::new();
        for d in deltas {
            out.push_str(&s.feed(d));
        }
        out.push_str(&s.flush());
        out
    }

    #[test]
    fn split_open_block_across_deltas() {
        // The motivating case: split <think> across deltas must not leak.
        let out = run(&["<think>", "Let me check their config", "</think>"]);
        assert_eq!(out, "");
    }

    #[test]
    fn closed_pair_inline_is_stripped_without_boundary() {
        // Closed pairs are always suppressed regardless of boundary.
        let out = run(&["use <think>secret</think> done"]);
        assert_eq!(out, "use  done");
    }

    #[test]
    fn open_tag_mentioned_in_prose_not_stripped() {
        // An unterminated open tag mid-line (not at a boundary) is prose.
        let mut s = StreamingThinkScrubber::new();
        let out = s.feed("use <think> tags here");
        // It's held back partially or emitted; flush surfaces remainder.
        let mut full = out;
        full.push_str(&s.flush());
        assert_eq!(full, "use <think> tags here");
    }

    #[test]
    fn open_at_start_of_stream_is_boundary() {
        let mut s = StreamingThinkScrubber::new();
        let out = s.feed("<think>secret");
        assert_eq!(out, "");
        // Still in block, flush discards.
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn open_after_newline_is_boundary() {
        let out = run(&["hello\n<think>secret</think>"]);
        assert_eq!(out, "hello\n");
    }

    #[test]
    fn unterminated_block_discarded_on_flush() {
        let mut s = StreamingThinkScrubber::new();
        assert_eq!(s.feed("<think>partial reasoning"), "");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn plain_text_passes_through() {
        let out = run(&["hello world"]);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn case_insensitive_tags() {
        let out = run(&["<THINK>secret</THINK>after"]);
        assert_eq!(out, "after");
    }

    #[test]
    fn all_tag_variants() {
        assert_eq!(run(&["<thinking>x</thinking>a"]), "a");
        assert_eq!(run(&["<reasoning>x</reasoning>a"]), "a");
        assert_eq!(run(&["<thought>x</thought>a"]), "a");
        assert_eq!(run(&["<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>a"]), "a");
    }

    #[test]
    fn partial_tag_held_back_then_resolved() {
        let mut s = StreamingThinkScrubber::new();
        // "<thi" looks like a partial open tag → held back.
        let out1 = s.feed("<thi");
        assert_eq!(out1, "");
        // Completes to <think> at a boundary (start of stream).
        let out2 = s.feed("nk>secret</think>done");
        assert_eq!(out2, "done");
    }

    #[test]
    fn partial_tag_turns_out_to_be_prose() {
        let mut s = StreamingThinkScrubber::new();
        let out1 = s.feed("hello <thi");
        // "hello " emitted, "<thi" held.
        assert_eq!(out1, "hello ");
        let out2 = s.feed(" s a tag");
        assert_eq!(out2, "<thi s a tag");
    }

    #[test]
    fn orphan_close_tag_stripped() {
        let out = run(&["hello </think> world"]);
        // Orphan close tag + trailing whitespace stripped.
        assert_eq!(out, "hello world");
    }

    #[test]
    fn close_split_across_deltas_inside_block() {
        let mut s = StreamingThinkScrubber::new();
        assert_eq!(s.feed("<think>secret</thi"), "");
        assert_eq!(s.feed("nk>visible"), "visible");
    }

    #[test]
    fn reset_clears_hung_block() {
        let mut s = StreamingThinkScrubber::new();
        assert_eq!(s.feed("<think>hung"), "");
        s.reset();
        // Fresh turn: plain text passes.
        assert_eq!(s.feed("clean"), "clean");
    }

    #[test]
    fn empty_feed_returns_empty() {
        let mut s = StreamingThinkScrubber::new();
        assert_eq!(s.feed(""), "");
    }

    #[test]
    fn newline_boundary_tracked_across_feeds() {
        let mut s = StreamingThinkScrubber::new();
        // Emit prose ending without newline.
        assert_eq!(s.feed("answer:"), "answer:");
        // Next feed: open tag at position 0 is NOT a boundary (prior emission
        // did not end with newline), so it's treated as prose.
        let mut full = s.feed("<think>");
        full.push_str(&s.flush());
        assert_eq!(full, "<think>");
    }

    #[test]
    fn newline_then_open_at_position_zero_is_boundary() {
        let mut s = StreamingThinkScrubber::new();
        assert_eq!(s.feed("line\n"), "line\n");
        // Open tag at position 0, prior emission ended with newline → boundary.
        assert_eq!(s.feed("<think>secret"), "");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn multibyte_content_does_not_panic() {
        let out = run(&["héllo <think>résumé</think> wörld"]);
        assert_eq!(out, "héllo  wörld");
    }

    #[test]
    fn whitespace_indented_open_is_boundary() {
        // After a newline with only whitespace before the tag.
        let out = run(&["text\n  <think>secret</think>"]);
        assert_eq!(out, "text\n  ");
    }
}
