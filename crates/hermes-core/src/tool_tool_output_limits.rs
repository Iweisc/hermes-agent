//! Configurable tool-output truncation limits.
//!
//! Ported from `tools/tool_output_limits.py` (originally from
//! anomalyco/opencode PR #23770, *feat(truncate): allow configuring tool
//! output truncation limits*).
//!
//! OpenCode hardcoded `MAX_LINES = 2000` and `MAX_BYTES = 50 * 1024` as
//! tool-output truncation thresholds. Hermes-agent had the same hardcoded
//! constants in two places:
//!
//! * `tools/terminal_tool.py` — `MAX_OUTPUT_CHARS = 50000` (terminal
//!   stdout/stderr cap)
//! * `tools/file_operations.py` — `MAX_LINES = 2000` / `MAX_LINE_LENGTH = 2000`
//!   (read_file pagination cap + per-line cap)
//!
//! This module centralises those values behind a single config section
//! (`tool_output` in `config.yaml`) so power users can tune them without
//! patching the source. The existing hardcoded numbers remain as defaults, so
//! behaviour is unchanged when the config key is absent.
//!
//! Example `config.yaml`:
//!
//! ```yaml
//! tool_output:
//!   max_bytes: 100000        # terminal output cap (chars)
//!   max_lines: 5000          # read_file pagination + truncation cap
//!   max_line_length: 2000    # per-line length cap before '... [truncated]'
//! ```
//!
//! The limits reader is defensive: any error (missing config file, invalid
//! value type, etc.) falls back to the built-in defaults so tools never fail
//! because of a malformed config.

use serde_yaml::Value;

/// Terminal output cap (chars). Matches `terminal_tool.MAX_OUTPUT_CHARS`.
pub const DEFAULT_MAX_BYTES: i64 = 50_000;
/// read_file pagination + truncation cap. Matches `file_operations.MAX_LINES`.
pub const DEFAULT_MAX_LINES: i64 = 2000;
/// Per-line length cap. Matches `file_operations.MAX_LINE_LENGTH`.
pub const DEFAULT_MAX_LINE_LENGTH: i64 = 2000;

/// Resolved tool-output limits.
///
/// Mirrors the dict returned by the Python `get_tool_output_limits()`, with
/// keys `max_bytes`, `max_lines`, `max_line_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolOutputLimits {
    pub max_bytes: i64,
    pub max_lines: i64,
    pub max_line_length: i64,
}

impl Default for ToolOutputLimits {
    fn default() -> Self {
        ToolOutputLimits {
            max_bytes: DEFAULT_MAX_BYTES,
            max_lines: DEFAULT_MAX_LINES,
            max_line_length: DEFAULT_MAX_LINE_LENGTH,
        }
    }
}

/// Return `value` coerced to a positive int, or `default` on any issue.
///
/// Faithful port of Python's `_coerce_positive_int`: it accepts ints, floats
/// (truncated toward zero like Python's `int()`), and numeric strings. Any
/// value that fails to parse, or that resolves to `<= 0`, yields `default`.
fn coerce_positive_int(value: Option<&Value>, default: i64) -> i64 {
    let iv = match value {
        None | Some(Value::Null) => return default,
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(u) = n.as_u64() {
                // Clamp values that overflow i64; treat as "too big to be sane".
                if u > i64::MAX as u64 {
                    return default;
                }
                u as i64
            } else if let Some(f) = n.as_f64() {
                // Python int() truncates toward zero; NaN/inf -> default.
                if f.is_finite() {
                    f.trunc() as i64
                } else {
                    return default;
                }
            } else {
                return default;
            }
        }
        Some(Value::String(s)) => {
            // Python's int(str) parses a base-10 integer literal (optionally
            // surrounded by whitespace). It does NOT accept floats like "1.5".
            match s.trim().parse::<i64>() {
                Ok(i) => i,
                Err(_) => return default,
            }
        }
        Some(Value::Bool(_)) | Some(Value::Sequence(_)) | Some(Value::Mapping(_)) => {
            return default;
        }
        Some(Value::Tagged(_)) => return default,
    };
    if iv <= 0 { default } else { iv }
}

/// Resolve tool-output limits from the `tool_output` section of `config`.
///
/// `config` is the parsed `config.yaml` document (or `None` if unavailable).
/// Missing or invalid entries fall through to the `DEFAULT_*` constants. This
/// function NEVER panics — it is the analog of Python's
/// `get_tool_output_limits()`, but takes the loaded config as a parameter
/// rather than importing `hermes_cli.config` directly (so callers stay in
/// control of how/when the config document is loaded).
pub fn get_tool_output_limits(config: Option<&Value>) -> ToolOutputLimits {
    // Locate the `tool_output` mapping; anything non-mapping is treated as empty.
    let section: Option<&Value> = match config {
        Some(Value::Mapping(map)) => map.get(Value::String("tool_output".to_string())),
        _ => None,
    };
    let section_map = match section {
        Some(Value::Mapping(map)) => Some(map),
        _ => None,
    };

    let get = |key: &str| -> Option<&Value> {
        section_map.and_then(|m| m.get(Value::String(key.to_string())))
    };

    ToolOutputLimits {
        max_bytes: coerce_positive_int(get("max_bytes"), DEFAULT_MAX_BYTES),
        max_lines: coerce_positive_int(get("max_lines"), DEFAULT_MAX_LINES),
        max_line_length: coerce_positive_int(get("max_line_length"), DEFAULT_MAX_LINE_LENGTH),
    }
}

/// Shortcut for terminal-tool callers that only need the byte cap.
pub fn get_max_bytes(config: Option<&Value>) -> i64 {
    get_tool_output_limits(config).max_bytes
}

/// Shortcut for file-ops callers that only need the line cap.
pub fn get_max_lines(config: Option<&Value>) -> i64 {
    get_tool_output_limits(config).max_lines
}

/// Shortcut for file-ops callers that only need the per-line cap.
pub fn get_max_line_length(config: Option<&Value>) -> i64 {
    get_tool_output_limits(config).max_line_length
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::from_str;

    fn yaml(s: &str) -> Value {
        from_str(s).unwrap()
    }

    #[test]
    fn defaults_when_no_config() {
        let limits = get_tool_output_limits(None);
        assert_eq!(limits, ToolOutputLimits::default());
        assert_eq!(limits.max_bytes, 50_000);
        assert_eq!(limits.max_lines, 2000);
        assert_eq!(limits.max_line_length, 2000);
    }

    #[test]
    fn defaults_when_section_missing() {
        let cfg = yaml("foo: bar\n");
        let limits = get_tool_output_limits(Some(&cfg));
        assert_eq!(limits, ToolOutputLimits::default());
    }

    #[test]
    fn defaults_when_section_not_mapping() {
        let cfg = yaml("tool_output: 42\n");
        assert_eq!(get_tool_output_limits(Some(&cfg)), ToolOutputLimits::default());
    }

    #[test]
    fn reads_valid_overrides() {
        let cfg = yaml("tool_output:\n  max_bytes: 100000\n  max_lines: 5000\n  max_line_length: 1000\n");
        let limits = get_tool_output_limits(Some(&cfg));
        assert_eq!(limits.max_bytes, 100_000);
        assert_eq!(limits.max_lines, 5000);
        assert_eq!(limits.max_line_length, 1000);
    }

    #[test]
    fn partial_overrides_fall_back_per_key() {
        let cfg = yaml("tool_output:\n  max_lines: 9999\n");
        let limits = get_tool_output_limits(Some(&cfg));
        assert_eq!(limits.max_bytes, DEFAULT_MAX_BYTES);
        assert_eq!(limits.max_lines, 9999);
        assert_eq!(limits.max_line_length, DEFAULT_MAX_LINE_LENGTH);
    }

    #[test]
    fn non_positive_falls_back() {
        let cfg = yaml("tool_output:\n  max_bytes: 0\n  max_lines: -5\n");
        let limits = get_tool_output_limits(Some(&cfg));
        assert_eq!(limits.max_bytes, DEFAULT_MAX_BYTES);
        assert_eq!(limits.max_lines, DEFAULT_MAX_LINES);
    }

    #[test]
    fn invalid_types_fall_back() {
        let cfg = yaml("tool_output:\n  max_bytes: \"not-a-number\"\n  max_lines: [1, 2]\n  max_line_length: {a: 1}\n");
        let limits = get_tool_output_limits(Some(&cfg));
        assert_eq!(limits, ToolOutputLimits::default());
    }

    #[test]
    fn numeric_string_is_coerced() {
        let cfg = yaml("tool_output:\n  max_bytes: \"  4096  \"\n");
        assert_eq!(get_max_bytes(Some(&cfg)), 4096);
    }

    #[test]
    fn float_truncates_toward_zero() {
        let cfg = yaml("tool_output:\n  max_lines: 12.9\n");
        assert_eq!(get_max_lines(Some(&cfg)), 12);
    }

    #[test]
    fn bool_falls_back() {
        // Note: Python int(True) == 1, but YAML bools are not numbers here;
        // we treat bool as invalid -> default, which is the conservative path.
        let cfg = yaml("tool_output:\n  max_line_length: true\n");
        assert_eq!(get_max_line_length(Some(&cfg)), DEFAULT_MAX_LINE_LENGTH);
    }

    #[test]
    fn shortcuts_match_struct() {
        let cfg = yaml("tool_output:\n  max_bytes: 7\n  max_lines: 8\n  max_line_length: 9\n");
        assert_eq!(get_max_bytes(Some(&cfg)), 7);
        assert_eq!(get_max_lines(Some(&cfg)), 8);
        assert_eq!(get_max_line_length(Some(&cfg)), 9);
    }
}
