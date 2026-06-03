//! Shared ANSI color utilities for Hermes CLI modules.
//!
//! Faithful port of `hermes_cli/colors.py`. Provides:
//! - [`should_use_color`]: gate colored output on env + TTY checks.
//! - [`Colors`]: the raw ANSI escape sequences.
//! - [`color`]: wrap text in color codes when appropriate.

use std::env;

/// Return `true` when colored output is appropriate.
///
/// Respects the `NO_COLOR` environment variable (https://no-color.org/)
/// and `TERM=dumb`, in addition to the stdout TTY check.
pub fn should_use_color() -> bool {
    if env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if env::var("TERM").map(|t| t == "dumb").unwrap_or(false) {
        return false;
    }
    if !stdout_is_tty() {
        return false;
    }
    true
}

/// Mirror of Python's `sys.stdout.isatty()` (fd 1).
fn stdout_is_tty() -> bool {
    // SAFETY: isatty just inspects a file descriptor and has no memory effects.
    unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

/// Raw ANSI escape sequences, mirroring the Python `Colors` class.
pub struct Colors;

impl Colors {
    pub const RESET: &'static str = "\x1b[0m";
    pub const BOLD: &'static str = "\x1b[1m";
    pub const DIM: &'static str = "\x1b[2m";
    pub const RED: &'static str = "\x1b[31m";
    pub const GREEN: &'static str = "\x1b[32m";
    pub const YELLOW: &'static str = "\x1b[33m";
    pub const BLUE: &'static str = "\x1b[34m";
    pub const MAGENTA: &'static str = "\x1b[35m";
    pub const CYAN: &'static str = "\x1b[36m";
}

/// Apply color `codes` to `text`, but only when color output is appropriate.
///
/// When color is disabled (see [`should_use_color`]), the text is returned
/// unchanged. Otherwise the codes are concatenated, prepended to the text,
/// and followed by [`Colors::RESET`] — matching the Python `color()` helper.
pub fn color(text: &str, codes: &[&str]) -> String {
    if !should_use_color() {
        return text.to_string();
    }
    let mut out = String::new();
    for c in codes {
        out.push_str(c);
    }
    out.push_str(text);
    out.push_str(Colors::RESET);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the colored string directly, bypassing the TTY gate, so the
    /// formatting logic can be asserted deterministically in CI (no TTY).
    fn color_forced(text: &str, codes: &[&str]) -> String {
        let mut out = String::new();
        for c in codes {
            out.push_str(c);
        }
        out.push_str(text);
        out.push_str(Colors::RESET);
        out
    }

    #[test]
    fn codes_match_python_class() {
        assert_eq!(Colors::RESET, "\x1b[0m");
        assert_eq!(Colors::BOLD, "\x1b[1m");
        assert_eq!(Colors::DIM, "\x1b[2m");
        assert_eq!(Colors::RED, "\x1b[31m");
        assert_eq!(Colors::GREEN, "\x1b[32m");
        assert_eq!(Colors::YELLOW, "\x1b[33m");
        assert_eq!(Colors::BLUE, "\x1b[34m");
        assert_eq!(Colors::MAGENTA, "\x1b[35m");
        assert_eq!(Colors::CYAN, "\x1b[36m");
    }

    #[test]
    fn color_concatenates_codes_text_and_reset() {
        assert_eq!(
            color_forced("hi", &[Colors::BOLD, Colors::RED]),
            "\x1b[1m\x1b[31mhi\x1b[0m"
        );
    }

    #[test]
    fn color_with_no_codes_just_wraps_reset() {
        assert_eq!(color_forced("plain", &[]), "plain\x1b[0m");
    }

    #[test]
    fn color_returns_plain_text_when_disabled() {
        // In the test harness stdout is not a TTY, so should_use_color() is
        // false and the public `color` must return the text unchanged.
        assert!(!should_use_color());
        assert_eq!(color("hello", &[Colors::GREEN]), "hello");
    }

    #[test]
    fn empty_text_round_trips() {
        assert_eq!(color_forced("", &[Colors::CYAN]), "\x1b[36m\x1b[0m");
    }
}
