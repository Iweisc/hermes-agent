//! Trajectory saving utilities and static helpers.
//!
//! Native Rust port of `agent/trajectory.py`.
//!
//! `_convert_to_trajectory_format` stays in the agent layer (batch_runner
//! calls `agent._convert_to_trajectory_format`). Only the static helpers and
//! the file-write logic live here.

use std::fs::OpenOptions;
use std::io::Write as _;

use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Convert `<REASONING_SCRATCHPAD>` tags to `<think>` tags.
///
/// Mirrors the Python short-circuit: if the content is empty or has no
/// opening `<REASONING_SCRATCHPAD>` tag, it is returned unchanged.
pub fn convert_scratchpad_to_think(content: &str) -> String {
    if content.is_empty() || !content.contains("<REASONING_SCRATCHPAD>") {
        return content.to_string();
    }
    content
        .replace("<REASONING_SCRATCHPAD>", "<think>")
        .replace("</REASONING_SCRATCHPAD>", "</think>")
}

/// Check if content has an opening `<REASONING_SCRATCHPAD>` without a closing tag.
pub fn has_incomplete_scratchpad(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    content.contains("<REASONING_SCRATCHPAD>") && !content.contains("</REASONING_SCRATCHPAD>")
}

/// A single trajectory file entry, serialised as one JSONL line.
///
/// `conversations` holds the ShareGPT-format conversation list. Each turn is
/// kept as an opaque `serde_json::Value` so the exact shape produced by the
/// agent layer round-trips byte-for-byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryEntry {
    pub conversations: Vec<Value>,
    pub timestamp: String,
    pub model: String,
    pub completed: bool,
}

/// Build the default output filename based on completion status.
///
/// Mirrors the Python default: `trajectory_samples.jsonl` when completed,
/// otherwise `failed_trajectories.jsonl`.
pub fn default_trajectory_filename(completed: bool) -> &'static str {
    if completed {
        "trajectory_samples.jsonl"
    } else {
        "failed_trajectories.jsonl"
    }
}

/// Produce the local-time ISO-8601 timestamp matching Python's
/// `datetime.now().isoformat()` (no timezone offset, microsecond precision).
fn now_isoformat() -> String {
    // Python's datetime.isoformat() emits e.g. "2026-06-03T14:30:00.123456"
    // (microseconds, naive local time, no offset). chrono's %.6f renders the
    // fractional second with a leading dot and 6 digits.
    Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

/// Build a trajectory entry without writing it (timestamp captured now).
pub fn build_trajectory_entry(
    trajectory: Vec<Value>,
    model: &str,
    completed: bool,
) -> TrajectoryEntry {
    TrajectoryEntry {
        conversations: trajectory,
        timestamp: now_isoformat(),
        model: model.to_string(),
        completed,
    }
}

/// Append a trajectory entry to a JSONL file.
///
/// * `trajectory` - the ShareGPT-format conversation list.
/// * `model` - model name for metadata.
/// * `completed` - whether the conversation completed successfully.
/// * `filename` - override output filename. Defaults to
///   `trajectory_samples.jsonl` or `failed_trajectories.jsonl` based on
///   `completed`.
///
/// Like the Python, this never propagates I/O errors: failures are logged at
/// warn level and swallowed. Returns `true` when the line was written.
pub fn save_trajectory(
    trajectory: Vec<Value>,
    model: &str,
    completed: bool,
    filename: Option<&str>,
) -> bool {
    let filename = filename
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_trajectory_filename(completed).to_string());

    let entry = build_trajectory_entry(trajectory, model, completed);

    match write_entry(&filename, &entry) {
        Ok(()) => {
            log::info!("Trajectory saved to {}", filename);
            true
        }
        Err(e) => {
            log::warn!("Failed to save trajectory: {}", e);
            false
        }
    }
}

/// Serialise `entry` to a JSONL line and append it to `filename`.
///
/// Separated out so tests can exercise the write path and surface errors that
/// `save_trajectory` deliberately swallows.
pub fn write_entry(filename: &str, entry: &TrajectoryEntry) -> Result<(), String> {
    // ensure_ascii=False in the Python: serde_json already emits UTF-8 without
    // escaping non-ASCII, matching that behaviour.
    let line = serde_json::to_string(entry).map_err(|e| e.to_string())?;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename)
        .map_err(|e| e.to_string())?;
    f.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    f.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_replaces_both_tags() {
        let input = "<REASONING_SCRATCHPAD>hello</REASONING_SCRATCHPAD>";
        assert_eq!(convert_scratchpad_to_think(input), "<think>hello</think>");
    }

    #[test]
    fn convert_no_tag_returns_unchanged() {
        assert_eq!(convert_scratchpad_to_think("plain text"), "plain text");
    }

    #[test]
    fn convert_empty_returns_empty() {
        assert_eq!(convert_scratchpad_to_think(""), "");
    }

    #[test]
    fn convert_handles_multiple_occurrences() {
        let input = "<REASONING_SCRATCHPAD>a</REASONING_SCRATCHPAD> mid <REASONING_SCRATCHPAD>b</REASONING_SCRATCHPAD>";
        assert_eq!(
            convert_scratchpad_to_think(input),
            "<think>a</think> mid <think>b</think>"
        );
    }

    #[test]
    fn convert_only_opening_tag_still_replaces() {
        // Python replaces unconditionally once the opening tag is present.
        assert_eq!(
            convert_scratchpad_to_think("<REASONING_SCRATCHPAD>partial"),
            "<think>partial"
        );
    }

    #[test]
    fn incomplete_true_when_open_no_close() {
        assert!(has_incomplete_scratchpad("<REASONING_SCRATCHPAD>oops"));
    }

    #[test]
    fn incomplete_false_when_balanced() {
        assert!(!has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>ok</REASONING_SCRATCHPAD>"
        ));
    }

    #[test]
    fn incomplete_false_when_no_tags() {
        assert!(!has_incomplete_scratchpad("nothing here"));
    }

    #[test]
    fn incomplete_false_when_empty() {
        assert!(!has_incomplete_scratchpad(""));
    }

    #[test]
    fn default_filename_by_completion() {
        assert_eq!(default_trajectory_filename(true), "trajectory_samples.jsonl");
        assert_eq!(
            default_trajectory_filename(false),
            "failed_trajectories.jsonl"
        );
    }

    #[test]
    fn timestamp_has_iso_shape() {
        let ts = now_isoformat();
        // e.g. 2026-06-03T14:30:00.123456 -> 'T' separator, '.' fractional sep.
        assert_eq!(ts.as_bytes()[10], b'T');
        assert!(ts.contains('.'), "timestamp should have fractional seconds: {ts}");
        // No timezone offset suffix (matches naive datetime.now()).
        assert!(!ts.ends_with('Z'));
        assert!(!ts[11..].contains('+'));
    }

    #[test]
    fn build_entry_preserves_fields_and_payload() {
        let convo = vec![
            json!({"from": "human", "value": "hi"}),
            json!({"from": "gpt", "value": "héllo 你好"}),
        ];
        let entry = build_trajectory_entry(convo.clone(), "test-model", true);
        assert_eq!(entry.model, "test-model");
        assert!(entry.completed);
        assert_eq!(entry.conversations, convo);
    }

    #[test]
    fn write_entry_appends_jsonl_with_unicode_unescaped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traj.jsonl");
        let path_str = path.to_str().unwrap();

        let e1 = build_trajectory_entry(vec![json!({"value": "你好"})], "m1", true);
        let e2 = build_trajectory_entry(vec![json!({"value": "bye"})], "m2", false);
        write_entry(path_str, &e1).unwrap();
        write_entry(path_str, &e2).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        // Non-ASCII is preserved (ensure_ascii=False parity).
        assert!(lines[0].contains("你好"));

        // Each line round-trips back into a TrajectoryEntry.
        let parsed1: TrajectoryEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed1.model, "m1");
        assert!(parsed1.completed);
        let parsed2: TrajectoryEntry = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(parsed2.model, "m2");
        assert!(!parsed2.completed);
    }

    #[test]
    fn save_trajectory_default_completed_filename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory_samples.jsonl");
        // Use an explicit path so the test does not touch CWD.
        let ok = save_trajectory(
            vec![json!({"value": "x"})],
            "model-x",
            true,
            Some(path.to_str().unwrap()),
        );
        assert!(ok);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 1);
        let parsed: TrajectoryEntry = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        assert_eq!(parsed.model, "model-x");
    }
}
