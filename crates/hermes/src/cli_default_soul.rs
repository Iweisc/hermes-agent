//! Default SOUL.md template seeded into HERMES_HOME on first run.
//!
//! Native Rust port of `hermes_cli/default_soul.py`. The Python module exposed a
//! single module-level string, `DEFAULT_SOUL_MD`. We reproduce that exact text
//! verbatim as a `pub const`, plus a small accessor for ergonomic use from other
//! modules.

/// The default `SOUL.md` contents written into `HERMES_HOME` the first time the
/// agent runs and no soul file exists yet.
///
/// This must remain byte-for-byte identical to the Python source so that the
/// seeded file is stable across the Python and native implementations.
pub const DEFAULT_SOUL_MD: &str = "You are Hermes Agent, an intelligent AI assistant created by Nous Research. \
You are helpful, knowledgeable, and direct. You assist users with a wide \
range of tasks including answering questions, writing and editing code, \
analyzing information, creative work, and executing actions via your tools. \
You communicate clearly, admit uncertainty when appropriate, and prioritize \
being genuinely useful over being verbose unless otherwise directed below. \
Be targeted and efficient in your exploration and investigations.";

/// Returns the default SOUL.md template text.
///
/// Equivalent to reading the module-level `DEFAULT_SOUL_MD` constant in the
/// original Python module; provided as a function for callers that prefer an
/// accessor over a bare constant.
pub fn default_soul_md() -> &'static str {
    DEFAULT_SOUL_MD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_source_exactly() {
        // Reconstruct the string from the exact fragments in default_soul.py to
        // guard against accidental edits (whitespace, wording).
        let expected = String::new()
            + "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
            + "You are helpful, knowledgeable, and direct. You assist users with a wide "
            + "range of tasks including answering questions, writing and editing code, "
            + "analyzing information, creative work, and executing actions via your tools. "
            + "You communicate clearly, admit uncertainty when appropriate, and prioritize "
            + "being genuinely useful over being verbose unless otherwise directed below. "
            + "Be targeted and efficient in your exploration and investigations.";
        assert_eq!(DEFAULT_SOUL_MD, expected);
    }

    #[test]
    fn accessor_matches_constant() {
        assert_eq!(default_soul_md(), DEFAULT_SOUL_MD);
    }

    #[test]
    fn starts_and_ends_expected() {
        assert!(DEFAULT_SOUL_MD.starts_with("You are Hermes Agent, an intelligent AI assistant created by Nous Research."));
        assert!(DEFAULT_SOUL_MD.ends_with("Be targeted and efficient in your exploration and investigations."));
        // Single-line template: no embedded newlines.
        assert!(!DEFAULT_SOUL_MD.contains('\n'));
    }
}
