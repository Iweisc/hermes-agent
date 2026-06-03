//! Strip ANSI escape sequences from subprocess output.
//!
//! Used by terminal_tool, code_execution_tool, and process_registry to clean
//! command output before returning it to the model. This prevents ANSI codes
//! from entering the model's context — which is the root cause of models
//! copying escape sequences into file writes.
//!
//! Covers the full ECMA-48 spec: CSI (including private-mode `?` prefix,
//! colon-separated params, intermediate bytes), OSC (BEL and ST terminators),
//! DCS/SOS/PM/APC string sequences, nF multi-byte escapes, Fp/Fe/Fs
//! single-byte escapes, and 8-bit C1 control characters.
//!
//! Faithful port of `tools/ansi_strip.py`.

use regex::Regex;
use std::sync::OnceLock;

/// The full ANSI-escape matcher.
///
/// Mirrors the Python `_ANSI_ESCAPE_RE`. Notes on the Rust translation:
/// - `\x1b` is the ESC byte (U+001B).
/// - `[\s\S]` in Python (with `re.DOTALL`) matches *any* character including
///   newlines; the Rust equivalent is `(?s:.)`. We enable `(?s)` (dotall) so
///   `.` matches newlines, matching Python's `re.DOTALL`.
/// - The 8-bit C1 controls `\x80-\x9f` and the dedicated bytes `\x9b`/`\x9c`/
///   `\x9d` are Unicode scalar values here (the `regex` crate operates on
///   `&str`), exactly as Python's `str` regex treats them as code points.
fn ansi_escape_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            // dotall so `.` matches newlines (Python re.DOTALL).
            r"(?s)",
            r"\x1b",
            r"(?:",
            // CSI sequence: ESC [ params intermediates final
            r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]",
            // OSC (BEL or ST terminator)
            r"|\].*?(?:\x07|\x1b\\)",
            // DCS/SOS/PM/APC strings (ST terminator)
            r"|[PX^_].*?(?:\x1b\\)",
            // nF escape sequences
            r"|[\x20-\x2f]+[\x30-\x7e]",
            // Fp/Fe/Fs single-byte
            r"|[\x30-\x7e]",
            r")",
            // 8-bit CSI
            r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]",
            // 8-bit OSC
            r"|\x9d.*?(?:\x07|\x9c)",
            // Other 8-bit C1 controls
            r"|[\x80-\x9f]",
        ))
        .expect("ANSI escape regex must compile")
    })
}

/// Fast-path check — skip the full regex when no escape-like bytes are present.
///
/// Mirrors Python's `_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")`.
fn has_escape_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\x1b\x80-\x9f]").expect("has-escape regex must compile"))
}

/// Remove ANSI escape sequences from `text`.
///
/// Returns the input unchanged (fast path) when no ESC or C1 bytes are
/// present. Safe to call on any string — clean text passes through with
/// negligible overhead.
///
/// Returns a borrowed `Cow` on the fast path (no allocation) and an owned
/// `Cow` when stripping actually occurs.
pub fn strip_ansi(text: &str) -> std::borrow::Cow<'_, str> {
    if text.is_empty() || !has_escape_re().is_match(text) {
        return std::borrow::Cow::Borrowed(text);
    }
    ansi_escape_re().replace_all(text, "")
}

/// Convenience wrapper returning an owned `String`.
pub fn strip_ansi_owned(text: &str) -> String {
    strip_ansi(text).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_clean_pass_through() {
        assert_eq!(strip_ansi(""), "");
        assert_eq!(strip_ansi("hello world"), "hello world");
        // No allocation on the fast path.
        assert!(matches!(
            strip_ansi("plain"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn strips_basic_color_csi() {
        // \x1b[31mRED\x1b[0m
        let input = "\u{1b}[31mRED\u{1b}[0m";
        assert_eq!(strip_ansi(input), "RED");
    }

    #[test]
    fn strips_private_mode_and_colon_params() {
        // CSI with private-mode '?' prefix.
        assert_eq!(strip_ansi("\u{1b}[?25lhi\u{1b}[?25h"), "hi");
        // Colon-separated params (\x3a is ':') with intermediate byte.
        assert_eq!(strip_ansi("\u{1b}[38:2:255:0:0mX"), "X");
    }

    #[test]
    fn strips_csi_with_intermediate_bytes() {
        // intermediate bytes \x20-\x2f then final \x40-\x7e
        assert_eq!(strip_ansi("\u{1b}[1 qZ"), "Z");
    }

    #[test]
    fn strips_osc_bel_terminated() {
        // OSC set window title, BEL terminated.
        assert_eq!(strip_ansi("\u{1b}]0;title\u{07}body"), "body");
    }

    #[test]
    fn strips_osc_st_terminated() {
        // OSC terminated by ST (ESC \).
        assert_eq!(strip_ansi("\u{1b}]8;;http://x\u{1b}\\link"), "link");
    }

    #[test]
    fn strips_dcs_sos_pm_apc() {
        // DCS string (ESC P ... ESC \)
        assert_eq!(strip_ansi("\u{1b}Pdata\u{1b}\\after"), "after");
        // APC string (ESC _ ... ESC \)
        assert_eq!(strip_ansi("\u{1b}_apc\u{1b}\\end"), "end");
    }

    #[test]
    fn strips_nf_escape() {
        // nF: ESC, intermediate bytes (\x20-\x2f), final (\x30-\x7e).
        // ESC ( B selects ASCII charset.
        assert_eq!(strip_ansi("\u{1b}(Btext"), "text");
    }

    #[test]
    fn strips_single_byte_fe_fp_fs() {
        // ESC c (full reset) — single-byte Fs.
        assert_eq!(strip_ansi("\u{1b}cclean"), "clean");
    }

    #[test]
    fn strips_8bit_csi() {
        // 8-bit CSI is \x9b.
        assert_eq!(strip_ansi("\u{9b}31mY\u{9b}0m"), "Y");
    }

    #[test]
    fn strips_8bit_osc() {
        // 8-bit OSC \x9d ... ST \x9c
        assert_eq!(strip_ansi("\u{9d}0;t\u{9c}data"), "data");
        // 8-bit OSC \x9d ... BEL
        assert_eq!(strip_ansi("\u{9d}0;t\u{07}data"), "data");
    }

    #[test]
    fn strips_other_c1_controls() {
        // Lone C1 control bytes get removed.
        assert_eq!(strip_ansi("a\u{80}b\u{9f}c"), "abc");
    }

    #[test]
    fn handles_multiline_dotall() {
        // OSC payload spanning a newline must still be consumed (DOTALL).
        let input = "\u{1b}]0;line1\nline2\u{07}done";
        assert_eq!(strip_ansi(input), "done");
    }

    #[test]
    fn owned_wrapper_works() {
        assert_eq!(strip_ansi_owned("\u{1b}[31mRED\u{1b}[0m"), "RED".to_string());
    }
}
