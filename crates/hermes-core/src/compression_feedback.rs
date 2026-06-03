//! User-facing summaries for manual compression commands.
//!
//! Native Rust port of `agent/manual_compression_feedback.py`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Consistent user-facing feedback for a manual compression operation.
///
/// Mirrors the dict returned by the Python `summarize_manual_compression`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManualCompressionFeedback {
    /// True when compression produced no change to the message list.
    pub noop: bool,
    /// Primary headline shown to the user.
    pub headline: String,
    /// Line describing the approximate request size in tokens.
    pub token_line: String,
    /// Optional explanatory note (e.g. when fewer messages still grow the
    /// token estimate). `None` when no note applies.
    pub note: Option<String>,
}

/// Format an integer with comma thousands separators, matching Python's
/// `f"{n:,}"` for non-negative and negative integers.
fn comma_format(n: i64) -> String {
    let negative = n < 0;
    // Use absolute value via i128 to avoid overflow on i64::MIN.
    let mut digits = (n as i128).unsigned_abs().to_string();

    // Insert commas every three digits from the right.
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    let bytes = digits.as_bytes();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    digits.clear();

    if negative {
        format!("-{out}")
    } else {
        out
    }
}

/// Return consistent user-facing feedback for manual compression.
///
/// `before_messages` / `after_messages` are the message lists before and
/// after compression; equality between them (element-by-element) determines
/// whether the operation was a no-op. Token counts are the approximate
/// request sizes before and after.
pub fn summarize_manual_compression(
    before_messages: &[Value],
    after_messages: &[Value],
    before_tokens: i64,
    after_tokens: i64,
) -> ManualCompressionFeedback {
    let before_count = before_messages.len();
    let after_count = after_messages.len();
    let noop = after_messages == before_messages;

    let headline;
    let token_line;

    if noop {
        headline = format!("No changes from compression: {before_count} messages");
        if after_tokens == before_tokens {
            token_line = format!(
                "Approx request size: ~{} tokens (unchanged)",
                comma_format(before_tokens)
            );
        } else {
            token_line = format!(
                "Approx request size: ~{} → ~{} tokens",
                comma_format(before_tokens),
                comma_format(after_tokens)
            );
        }
    } else {
        headline = format!("Compressed: {before_count} → {after_count} messages");
        token_line = format!(
            "Approx request size: ~{} → ~{} tokens",
            comma_format(before_tokens),
            comma_format(after_tokens)
        );
    }

    let note = if !noop && after_count < before_count && after_tokens > before_tokens {
        Some(
            "Note: fewer messages can still raise this estimate when \
compression rewrites the transcript into denser summaries."
                .to_string(),
        )
    } else {
        None
    };

    ManualCompressionFeedback {
        noop,
        headline,
        token_line,
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| json!({"role": "user", "content": format!("m{i}")}))
            .collect()
    }

    #[test]
    fn comma_format_matches_python() {
        assert_eq!(comma_format(0), "0");
        assert_eq!(comma_format(5), "5");
        assert_eq!(comma_format(999), "999");
        assert_eq!(comma_format(1_000), "1,000");
        assert_eq!(comma_format(12_345), "12,345");
        assert_eq!(comma_format(1_234_567), "1,234,567");
        assert_eq!(comma_format(-1_234_567), "-1,234,567");
    }

    #[test]
    fn noop_unchanged_tokens() {
        let m = msgs(3);
        let fb = summarize_manual_compression(&m, &m, 1500, 1500);
        assert!(fb.noop);
        assert_eq!(fb.headline, "No changes from compression: 3 messages");
        assert_eq!(
            fb.token_line,
            "Approx request size: ~1,500 tokens (unchanged)"
        );
        assert!(fb.note.is_none());
    }

    #[test]
    fn noop_changed_tokens() {
        let m = msgs(2);
        let fb = summarize_manual_compression(&m, &m, 1000, 1200);
        assert!(fb.noop);
        assert_eq!(fb.headline, "No changes from compression: 2 messages");
        assert_eq!(
            fb.token_line,
            "Approx request size: ~1,000 → ~1,200 tokens"
        );
        assert!(fb.note.is_none());
    }

    #[test]
    fn compressed_with_token_reduction() {
        let before = msgs(10);
        let after = msgs(4);
        let fb = summarize_manual_compression(&before, &after, 5000, 2000);
        assert!(!fb.noop);
        assert_eq!(fb.headline, "Compressed: 10 → 4 messages");
        assert_eq!(
            fb.token_line,
            "Approx request size: ~5,000 → ~2,000 tokens"
        );
        assert!(fb.note.is_none());
    }

    #[test]
    fn fewer_messages_more_tokens_emits_note() {
        let before = msgs(10);
        let after = msgs(3);
        let fb = summarize_manual_compression(&before, &after, 2000, 2500);
        assert!(!fb.noop);
        assert_eq!(fb.headline, "Compressed: 10 → 3 messages");
        assert_eq!(
            fb.token_line,
            "Approx request size: ~2,000 → ~2,500 tokens"
        );
        assert_eq!(
            fb.note.as_deref(),
            Some(
                "Note: fewer messages can still raise this estimate when \
compression rewrites the transcript into denser summaries."
            )
        );
    }

    #[test]
    fn more_messages_no_note() {
        // after_count >= before_count: note suppressed even if tokens grew.
        let before = msgs(3);
        let after = msgs(5);
        let fb = summarize_manual_compression(&before, &after, 100, 200);
        assert!(!fb.noop);
        assert!(fb.note.is_none());
    }

    #[test]
    fn different_content_same_length_is_not_noop() {
        let before = vec![json!({"role": "user", "content": "a"})];
        let after = vec![json!({"role": "user", "content": "b"})];
        let fb = summarize_manual_compression(&before, &after, 100, 100);
        assert!(!fb.noop);
        assert_eq!(fb.headline, "Compressed: 1 → 1 messages");
    }
}
