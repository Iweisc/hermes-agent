//! Shared CLI output helpers for Hermes CLI modules.
//!
//! Native Rust port of `hermes_cli/cli_output.py`.
//!
//! Extracts the identical `print_info/success/warning/error` and `prompt()`
//! functions previously duplicated across setup.py, tools_config.py,
//! mcp_config.py, and memory_setup.py.

use std::io::{self, BufRead, IsTerminal, Write};

use crate::cli_colors::{color, Colors};

// ─── Print Helpers ──────────────────────────────────────────────────────────

/// Format a dim informational message (with two leading spaces).
pub fn format_info(text: &str) -> String {
    color(&format!("  {}", text), &[Colors::DIM])
}

/// Format a green success message with `✓` prefix.
pub fn format_success(text: &str) -> String {
    color(&format!("✓ {}", text), &[Colors::GREEN])
}

/// Format a yellow warning message with `⚠` prefix.
pub fn format_warning(text: &str) -> String {
    color(&format!("⚠ {}", text), &[Colors::YELLOW])
}

/// Format a red error message with `✗` prefix.
pub fn format_error(text: &str) -> String {
    color(&format!("✗ {}", text), &[Colors::RED])
}

/// Format a bold yellow header (with leading newline and two spaces).
pub fn format_header(text: &str) -> String {
    color(&format!("\n  {}", text), &[Colors::YELLOW])
}

/// Print a dim informational message.
pub fn print_info(text: &str) {
    println!("{}", format_info(text));
}

/// Print a green success message with `✓` prefix.
pub fn print_success(text: &str) {
    println!("{}", format_success(text));
}

/// Print a yellow warning message with `⚠` prefix.
pub fn print_warning(text: &str) {
    println!("{}", format_warning(text));
}

/// Print a red error message with `✗` prefix.
pub fn print_error(text: &str) {
    println!("{}", format_error(text));
}

/// Print a bold yellow header.
pub fn print_header(text: &str) {
    println!("{}", format_header(text));
}

// ─── Input Prompts ────────────────────────────────────────────────────────────

/// Build the colored prompt display string for a question with an optional
/// default. Mirrors the Python suffix/format logic exactly.
pub fn prompt_display(question: &str, default: Option<&str>) -> String {
    let suffix = match default {
        Some(d) if !d.is_empty() => format!(" [{}]", d),
        _ => String::new(),
    };
    color(&format!("  {}{}: ", question, suffix), &[Colors::YELLOW])
}

/// Prompt the user for input with an optional default and password masking.
///
/// Returns the user's input (stripped), or *default* if the user presses Enter.
/// Returns empty string on EOF (or read error), mirroring the Python behavior
/// of returning `""` on `KeyboardInterrupt`/`EOFError`.
///
/// Note: Python's `default` defaulting to `None` becomes `None` here; a
/// `Some("")` is treated the same as `None` for the displayed suffix (matching
/// Python's `if default` truthiness) but is still returned on empty input.
pub fn prompt(question: &str, default: Option<&str>, password: bool) -> String {
    let display = prompt_display(question, default);

    let value = if password {
        read_password(&display)
    } else {
        read_line(&display)
    };

    match value {
        Some(v) => {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                trimmed.to_string()
            } else {
                default.unwrap_or("").to_string()
            }
        }
        None => {
            // EOF / read failure: match Python's print() then return "".
            println!();
            String::new()
        }
    }
}

/// Prompt for a yes/no answer. Returns bool.
///
/// Default of `true` shows the hint `Y/n`; `false` shows `y/N`. Empty input
/// returns the default; otherwise true iff the answer (lowercased) starts with
/// `y`.
pub fn prompt_yes_no(question: &str, default: bool) -> bool {
    let hint = if default { "Y/n" } else { "y/N" };
    let answer = prompt(&format!("{} ({})", question, hint), None, false);
    if answer.is_empty() {
        return default;
    }
    answer.to_lowercase().starts_with('y')
}

// ─── Low-level input ──────────────────────────────────────────────────────────

/// Read a single line from stdin after writing `display` to stdout.
/// Returns `None` on EOF or error.
fn read_line(display: &str) -> Option<String> {
    print!("{}", display);
    let _ = io::stdout().flush();

    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => None, // EOF with no bytes
        Ok(_) => {
            // Strip a single trailing newline (and optional CR) as input() does.
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            Some(line)
        }
        Err(_) => None,
    }
}

/// Read a password without echoing if stdin is a TTY. Falls back to a plain
/// line read when not attached to a terminal (mirroring getpass behavior of
/// falling back to a non-echoing-unavailable read).
fn read_password(display: &str) -> Option<String> {
    if io::stdin().is_terminal() {
        match read_password_tty(display) {
            Ok(v) => Some(v),
            Err(_) => None,
        }
    } else {
        // No TTY: getpass reads from stdin without masking.
        read_line(display)
    }
}

/// Disable terminal echo, read a line, restore echo. Unix-only via libc termios.
#[cfg(unix)]
fn read_password_tty(display: &str) -> io::Result<String> {
    use std::os::unix::io::AsRawFd;

    print!("{}", display);
    io::stdout().flush()?;

    let fd = io::stdin().as_raw_fd();

    // Save current termios, then clear ECHO.
    let mut term: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut term) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let original = term;
    term.c_lflag &= !libc::ECHO;
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &term) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let stdin = io::stdin();
    let mut line = String::new();
    let read_res = stdin.lock().read_line(&mut line);

    // Always restore the original terminal settings.
    unsafe {
        libc::tcsetattr(fd, libc::TCSANOW, &original);
    }
    // getpass echoes a newline since the user's Enter was swallowed.
    println!();

    let n = read_res?;
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
    }
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    Ok(line)
}

#[cfg(not(unix))]
fn read_password_tty(display: &str) -> io::Result<String> {
    // Non-unix fallback: no masking available, behave like a normal read.
    match read_line(display) {
        Some(v) => Ok(v),
        None => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests assume color output is disabled (NO_COLOR) so we can assert
    // on the raw text content rather than ANSI escapes.
    fn no_color() {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
    }

    #[test]
    fn format_helpers_prefixes() {
        no_color();
        assert_eq!(format_info("hi"), "  hi");
        assert_eq!(format_success("ok"), "✓ ok");
        assert_eq!(format_warning("careful"), "⚠ careful");
        assert_eq!(format_error("bad"), "✗ bad");
        assert_eq!(format_header("Title"), "\n  Title");
    }

    #[test]
    fn prompt_display_with_default() {
        no_color();
        assert_eq!(prompt_display("Name", Some("bob")), "  Name [bob]: ");
    }

    #[test]
    fn prompt_display_without_default() {
        no_color();
        assert_eq!(prompt_display("Name", None), "  Name: ");
    }

    #[test]
    fn prompt_display_empty_default_treated_as_none() {
        no_color();
        // Python: `if default` is falsy for "", so no suffix.
        assert_eq!(prompt_display("Name", Some("")), "  Name: ");
    }
}
