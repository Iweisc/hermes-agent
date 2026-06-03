//! Shared terminal-UI components for Hermes CLI.
//!
//! Faithful native-Rust port of `hermes_cli/curses_ui.py`.
//!
//! Used by `hermes tools` and `hermes skills` for interactive checklists.
//! Provides a fullscreen multi-select with keyboard navigation, plus a
//! text-based numbered fallback for terminals without interactive support.
//!
//! Python's `curses` module has no direct stdlib equivalent in Rust, so the
//! interactive path here is implemented directly on top of raw-mode termios +
//! ANSI escape sequences (an "alternate screen" full-redraw loop), which
//! reproduces the same keyboard navigation, scrolling, color attributes, and
//! cancel semantics as the original `curses.wrapper()` based code. When the
//! terminal cannot be put into raw mode (or stdin is not a TTY) the numbered
//! fallbacks — direct ports of the Python `_numbered_*` helpers — are used.

use std::collections::BTreeSet;
use std::io::{self, Read, Write};

use crate::cli_colors::{color, Colors};

// ---------------------------------------------------------------------------
// TTY / raw-mode helpers (replacing Python's `curses` + `termios`)
// ---------------------------------------------------------------------------

/// Mirror of Python's `sys.stdin.isatty()` (fd 0).
fn stdin_is_tty() -> bool {
    // SAFETY: isatty only inspects a file descriptor; no memory effects.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Flush any stray bytes from the stdin input buffer.
///
/// Direct port of the Python `flush_stdin`. Must be called after the
/// interactive raw-mode loop returns, **before** any subsequent line read.
/// `termios.TCIFLUSH` discards data received but not read; leftover
/// escape-sequence bytes (arrow keys, mode-switch responses, rapid
/// keypresses) would otherwise corrupt the next line input.
///
/// On non-TTY stdin (piped, redirected) this is a no-op.
pub fn flush_stdin() {
    if !stdin_is_tty() {
        return;
    }
    // SAFETY: tcflush operates on a file descriptor and the libc TCIFLUSH
    // queue selector; it has no Rust-visible memory effects.
    unsafe {
        let _ = libc::tcflush(libc::STDIN_FILENO, libc::TCIFLUSH);
    }
}

/// RAII guard that puts the terminal into cbreak/raw-ish mode for the
/// lifetime of the interactive loop and restores the prior settings on drop.
///
/// Mirrors what `curses.wrapper()` does: disable canonical mode and echo so
/// keypresses arrive immediately and are not printed, then restore on exit.
struct RawMode {
    fd: libc::c_int,
    orig: libc::termios,
}

impl RawMode {
    /// Enter raw mode. Returns `None` when the terminal cannot be configured
    /// (e.g. not a real TTY), which signals callers to use the fallback path.
    fn enter() -> Option<RawMode> {
        let fd = libc::STDIN_FILENO;
        // SAFETY: zeroed termios is a valid initial buffer for tcgetattr.
        let mut orig: libc::termios = unsafe { std::mem::zeroed() };
        // SAFETY: tcgetattr fills `orig`; returns non-zero on error.
        if unsafe { libc::tcgetattr(fd, &mut orig) } != 0 {
            return None;
        }
        let mut raw = orig;
        // Disable canonical mode (line buffering) and echo.
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        // Read returns as soon as 1 byte is available, no inter-byte timer.
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        // SAFETY: applying our modified termios copy to the same fd.
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return None;
        }
        Some(RawMode { fd, orig })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // SAFETY: restoring the previously captured termios on the same fd.
        unsafe {
            let _ = libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig);
        }
    }
}

/// Logical keypresses decoded from the raw-mode byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    Up,
    Down,
    Enter,
    Space,
    /// ESC or 'q' — cancel.
    Cancel,
    Char(u8),
    Other,
}

/// Read and decode one logical key from stdin in raw mode.
///
/// Reproduces the curses key mapping:
/// - `KEY_UP` / `k`            -> Up
/// - `KEY_DOWN` / `j`          -> Down
/// - `KEY_ENTER` / `\n` / `\r` -> Enter
/// - space                     -> Space
/// - ESC (bare) / `q`          -> Cancel
///
/// Arrow keys arrive as the CSI sequences `ESC [ A` (up) / `ESC [ B` (down).
/// A bare ESC (no following bytes ready) maps to Cancel, matching curses
/// returning `27` for the escape key.
fn read_key(stdin: &mut impl Read) -> Key {
    let mut b = [0u8; 1];
    if stdin.read(&mut b).unwrap_or(0) == 0 {
        return Key::Cancel; // EOF behaves like cancel.
    }
    match b[0] {
        0x1b => {
            // Possible escape sequence. Peek at the next byte non-blockingly.
            match read_byte_nonblocking(stdin) {
                Some(b'[') | Some(b'O') => match read_byte_nonblocking(stdin) {
                    Some(b'A') => Key::Up,
                    Some(b'B') => Key::Down,
                    Some(_) => Key::Other,
                    None => Key::Cancel,
                },
                Some(_) => Key::Other,
                None => Key::Cancel, // bare ESC
            }
        }
        b'\n' | b'\r' => Key::Enter,
        b' ' => Key::Space,
        b'k' => Key::Up,
        b'j' => Key::Down,
        b'q' => Key::Cancel,
        other => Key::Char(other),
    }
}

/// Try to read one more byte without blocking, used to disambiguate a bare
/// ESC from a CSI escape sequence. Returns `None` when nothing is buffered.
fn read_byte_nonblocking(stdin: &mut impl Read) -> Option<u8> {
    let fd = libc::STDIN_FILENO;
    // Use poll() with a tiny timeout so a lone ESC is not mistaken for the
    // start of an arrow-key sequence (terminals deliver the whole sequence
    // atomically, so a few ms is plenty).
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll on a single pollfd we own; timeout in milliseconds.
    let ready = unsafe { libc::poll(&mut pfd, 1, 2) };
    if ready <= 0 {
        return None;
    }
    let mut b = [0u8; 1];
    match stdin.read(&mut b) {
        Ok(1) => Some(b[0]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Screen drawing helpers (replacing curses addnstr/refresh)
// ---------------------------------------------------------------------------

/// Best-effort terminal size `(rows, cols)`, mirroring `stdscr.getmaxyx()`.
/// Falls back to a sane 24x80 when the ioctl is unavailable.
fn term_size() -> (usize, usize) {
    // SAFETY: winsize is a plain repr(C) struct; zeroed is a valid input.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: TIOCGWINSZ writes the window size into `ws` for the given fd.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) };
    if rc == 0 && ws.ws_row > 0 && ws.ws_col > 0 {
        (ws.ws_row as usize, ws.ws_col as usize)
    } else {
        (24, 80)
    }
}

/// True display width of `s` in terminal cells (counts Unicode scalar values,
/// not bytes), so multi-byte glyphs like `✓`/`→` occupy one column as they do
/// under curses' `addnstr` width accounting.
fn display_len(s: &str) -> usize {
    s.chars().count()
}

/// Truncate `s` to at most `max` display columns (`curses.addnstr` semantics).
fn truncate_cols(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max {
            break;
        }
        out.push(ch);
    }
    out
}

/// Accumulates ANSI output for one full-screen frame, then flushes atomically.
struct Frame {
    buf: String,
    use_color: bool,
}

impl Frame {
    fn new(use_color: bool) -> Frame {
        Frame {
            buf: String::new(),
            use_color,
        }
    }

    /// Clear the screen and move the cursor home (curses `stdscr.clear()`).
    fn clear(&mut self) {
        self.buf.push_str("\x1b[2J\x1b[H");
    }

    /// Write `text` (truncated to `max_x` columns) at 0-based `(y, x)` with the
    /// given attribute codes, mirroring `stdscr.addnstr(y, x, text, max_x, attr)`.
    fn add(&mut self, y: usize, x: usize, text: &str, max_x: usize, attrs: &[&str]) {
        // Move to 1-based cursor position.
        self.buf.push_str(&format!("\x1b[{};{}H", y + 1, x + 1));
        let truncated = truncate_cols(text, max_x);
        if self.use_color && !attrs.is_empty() {
            for a in attrs {
                self.buf.push_str(a);
            }
            self.buf.push_str(&truncated);
            self.buf.push_str(Colors::RESET);
        } else {
            self.buf.push_str(&truncated);
        }
    }

    /// Flush the frame to stdout.
    fn refresh(&mut self) {
        let mut out = io::stdout();
        let _ = out.write_all(self.buf.as_bytes());
        let _ = out.flush();
        self.buf.clear();
    }
}

/// Enter the alternate screen and hide the cursor (curses `wrapper` setup).
fn screen_setup() {
    let mut out = io::stdout();
    let _ = out.write_all(b"\x1b[?1049h\x1b[?25l");
    let _ = out.flush();
}

/// Leave the alternate screen and restore the cursor (curses `endwin`).
fn screen_teardown() {
    let mut out = io::stdout();
    let _ = out.write_all(b"\x1b[?25h\x1b[?1049l");
    let _ = out.flush();
}

/// Should attributes/colors be emitted? Gate on the same color policy used by
/// the rest of the CLI (this is the analogue of curses `has_colors()`).
fn has_colors() -> bool {
    crate::cli_colors::should_use_color()
}

// ---------------------------------------------------------------------------
// curses_checklist — multi-select
// ---------------------------------------------------------------------------

/// Multi-select checklist. Returns the set of selected indices.
///
/// Faithful port of Python `curses_checklist`.
///
/// - `title`: header line displayed above the checklist.
/// - `items`: display labels for each row.
/// - `selected`: indices that start checked (pre-selected).
/// - `cancel_returns`: returned on ESC/q. Defaults to the original `selected`.
/// - `status_fn`: optional callback `f(chosen) -> String` rendered on the
///   bottom row (e.g. live token counts). An empty string draws nothing.
pub fn curses_checklist(
    title: &str,
    items: &[String],
    selected: &BTreeSet<usize>,
    cancel_returns: Option<&BTreeSet<usize>>,
    status_fn: Option<&dyn Fn(&BTreeSet<usize>) -> String>,
) -> BTreeSet<usize> {
    let cancel: BTreeSet<usize> = match cancel_returns {
        Some(c) => c.clone(),
        None => selected.clone(),
    };

    // Safety: interactive mode hangs/spins when stdin is not a terminal
    // (e.g. subprocess pipe). Return defaults immediately.
    if !stdin_is_tty() {
        return cancel;
    }

    if items.is_empty() {
        // curses uses `% len(items)` which would divide by zero; the Python
        // code would raise and fall through to the numbered fallback, which
        // simply confirms an empty set. Match that quietly.
        return cancel.clone();
    }

    match checklist_interactive(title, items, selected, &cancel, status_fn) {
        Some(result) => result,
        None => numbered_fallback(title, items, selected, &cancel, status_fn),
    }
}

/// Returns `Some(result)` when the interactive loop ran to completion, or
/// `None` when raw mode could not be entered (caller falls back).
fn checklist_interactive(
    title: &str,
    items: &[String],
    selected: &BTreeSet<usize>,
    cancel_returns: &BTreeSet<usize>,
    status_fn: Option<&dyn Fn(&BTreeSet<usize>) -> String>,
) -> Option<BTreeSet<usize>> {
    let _raw = RawMode::enter()?;
    screen_setup();

    let colored = has_colors();
    // color_pair(1) green, color_pair(2) yellow, color_pair(3) dim gray.
    let mut chosen: BTreeSet<usize> = selected.clone();
    let mut cursor: usize = 0;
    let mut scroll_offset: usize = 0;
    let mut stdin = io::stdin();

    let result: BTreeSet<usize> = loop {
        let mut frame = Frame::new(colored);
        frame.clear();
        let (max_y, max_x) = term_size();
        let footer_rows = if status_fn.is_some() { 1 } else { 0 };

        // Header (bold + yellow) and help line (dim).
        let mut hattr: Vec<&str> = vec![Colors::BOLD];
        if colored {
            hattr.push(Colors::YELLOW);
        }
        frame.add(0, 0, title, max_x.saturating_sub(1), &hattr);
        frame.add(
            1,
            0,
            "  \u{2191}\u{2193} navigate  SPACE toggle  ENTER confirm  ESC cancel",
            max_x.saturating_sub(1),
            &[Colors::DIM],
        );

        // Scrollable item list.
        let visible_rows = max_y.saturating_sub(3 + footer_rows);
        if cursor < scroll_offset {
            scroll_offset = cursor;
        } else if visible_rows > 0 && cursor >= scroll_offset + visible_rows {
            scroll_offset = cursor - visible_rows + 1;
        }

        let end = items.len().min(scroll_offset + visible_rows);
        for (draw_i, i) in (scroll_offset..end).enumerate() {
            let y = draw_i + 3;
            if y >= max_y.saturating_sub(1 + footer_rows) {
                break;
            }
            let check = if chosen.contains(&i) { "\u{2713}" } else { " " };
            let arrow = if i == cursor { "\u{2192}" } else { " " };
            let line = format!(" {} [{}] {}", arrow, check, items[i]);
            let mut attr: Vec<&str> = Vec::new();
            if i == cursor {
                attr.push(Colors::BOLD);
                if colored {
                    attr.push(Colors::GREEN);
                }
            }
            frame.add(y, 0, &line, max_x.saturating_sub(1), &attr);
        }

        // Status bar (bottom row, right-aligned, dim gray).
        if let Some(sf) = status_fn {
            let status_text = sf(&chosen);
            if !status_text.is_empty() {
                let sx = max_x.saturating_sub(display_len(&status_text) + 1);
                frame.add(
                    max_y.saturating_sub(1),
                    sx,
                    &status_text,
                    max_x.saturating_sub(sx + 1),
                    &[Colors::DIM],
                );
            }
        }

        frame.refresh();

        match read_key(&mut stdin) {
            Key::Up => cursor = (cursor + items.len() - 1) % items.len(),
            Key::Down => cursor = (cursor + 1) % items.len(),
            Key::Space => {
                if chosen.contains(&cursor) {
                    chosen.remove(&cursor);
                } else {
                    chosen.insert(cursor);
                }
            }
            Key::Enter => break chosen.clone(),
            Key::Cancel => break cancel_returns.clone(),
            _ => {}
        }
    };

    screen_teardown();
    flush_stdin();
    Some(result)
}

// ---------------------------------------------------------------------------
// curses_radiolist — single-select with optional description
// ---------------------------------------------------------------------------

/// Single-select radio list. Returns the selected index.
///
/// Faithful port of Python `curses_radiolist`.
///
/// - `selected`: index that starts selected (pre-selected).
/// - `cancel_returns`: returned on ESC/q. Defaults to the original `selected`.
/// - `description`: optional multi-line text shown between the title and the
///   item list.
pub fn curses_radiolist(
    title: &str,
    items: &[String],
    selected: usize,
    cancel_returns: Option<usize>,
    description: Option<&str>,
) -> usize {
    let cancel = cancel_returns.unwrap_or(selected);

    if !stdin_is_tty() {
        return cancel;
    }

    if items.is_empty() {
        return cancel;
    }

    let desc_lines: Vec<&str> = match description {
        Some(d) => d.split('\n').collect(),
        None => Vec::new(),
    };

    match radiolist_interactive(title, items, selected, cancel, &desc_lines) {
        Some(r) => r,
        None => radio_numbered_fallback(title, items, selected, cancel),
    }
}

fn radiolist_interactive(
    title: &str,
    items: &[String],
    selected: usize,
    cancel_returns: usize,
    desc_lines: &[&str],
) -> Option<usize> {
    let _raw = RawMode::enter()?;
    screen_setup();

    let colored = has_colors();
    let mut cursor: usize = selected;
    let mut scroll_offset: usize = 0;
    let mut stdin = io::stdin();

    let result: usize = loop {
        let mut frame = Frame::new(colored);
        frame.clear();
        let (max_y, max_x) = term_size();

        let mut row: usize = 0;

        // Header.
        let mut hattr: Vec<&str> = vec![Colors::BOLD];
        if colored {
            hattr.push(Colors::YELLOW);
        }
        frame.add(row, 0, title, max_x.saturating_sub(1), &hattr);
        row += 1;

        // Description lines.
        for dline in desc_lines {
            if row >= max_y.saturating_sub(1) {
                break;
            }
            frame.add(row, 0, dline, max_x.saturating_sub(1), &[]);
            row += 1;
        }

        frame.add(
            row,
            0,
            "  \u{2191}\u{2193} navigate  ENTER/SPACE select  ESC cancel",
            max_x.saturating_sub(1),
            &[Colors::DIM],
        );
        row += 1;

        // Scrollable item list.
        let items_start = row + 1;
        let visible_rows = max_y.saturating_sub(items_start + 1);
        if cursor < scroll_offset {
            scroll_offset = cursor;
        } else if visible_rows > 0 && cursor >= scroll_offset + visible_rows {
            scroll_offset = cursor - visible_rows + 1;
        }

        let end = items.len().min(scroll_offset + visible_rows);
        for (draw_i, i) in (scroll_offset..end).enumerate() {
            let y = draw_i + items_start;
            if y >= max_y.saturating_sub(1) {
                break;
            }
            let radio = if i == selected { "\u{25cf}" } else { "\u{25cb}" };
            let arrow = if i == cursor { "\u{2192}" } else { " " };
            let line = format!(" {} ({}) {}", arrow, radio, items[i]);
            let mut attr: Vec<&str> = Vec::new();
            if i == cursor {
                attr.push(Colors::BOLD);
                if colored {
                    attr.push(Colors::GREEN);
                }
            }
            frame.add(y, 0, &line, max_x.saturating_sub(1), &attr);
        }

        frame.refresh();

        match read_key(&mut stdin) {
            Key::Up => cursor = (cursor + items.len() - 1) % items.len(),
            Key::Down => cursor = (cursor + 1) % items.len(),
            Key::Space | Key::Enter => break cursor,
            Key::Cancel => break cancel_returns,
            _ => {}
        }
    };

    screen_teardown();
    flush_stdin();
    Some(result)
}

/// Text-based numbered fallback for radio selection.
/// Direct port of Python `_radio_numbered_fallback`.
fn radio_numbered_fallback(
    title: &str,
    items: &[String],
    selected: usize,
    cancel_returns: usize,
) -> usize {
    println!("{}", color(&format!("\n  {}", title), &[Colors::YELLOW]));
    println!(
        "{}",
        color("  Select by number, Enter to confirm.\n", &[Colors::DIM])
    );

    for (i, label) in items.iter().enumerate() {
        let marker = if i == selected {
            color("(\u{25cf})", &[Colors::GREEN])
        } else {
            "(\u{25cb})".to_string()
        };
        println!("  {} {:>2}. {}", marker, i + 1, label);
    }
    println!();

    let prompt = color(
        &format!("  Choice [default {}]: ", selected + 1),
        &[Colors::DIM],
    );
    match read_line(&prompt) {
        Some(val) => {
            let val = val.trim();
            if val.is_empty() {
                return selected;
            }
            match val.parse::<i64>() {
                Ok(n) => {
                    let idx = n - 1;
                    if idx >= 0 && (idx as usize) < items.len() {
                        idx as usize
                    } else {
                        selected
                    }
                }
                Err(_) => cancel_returns,
            }
        }
        None => cancel_returns, // EOF / interrupt
    }
}

// ---------------------------------------------------------------------------
// curses_single_select — single-select menu with implicit Cancel row
// ---------------------------------------------------------------------------

/// Single-select menu. Returns the selected index, or `None` on cancel.
///
/// Faithful port of Python `curses_single_select`. A `cancel_label` row is
/// appended to the list; selecting it (or pressing ESC/q) returns `None`.
pub fn curses_single_select(
    title: &str,
    items: &[String],
    default_index: usize,
    cancel_label: &str,
) -> Option<usize> {
    if !stdin_is_tty() {
        return None;
    }

    let mut all_items: Vec<String> = items.to_vec();
    all_items.push(cancel_label.to_string());
    let cancel_idx = items.len();

    let selected = match single_select_interactive(title, &all_items, default_index) {
        Some(r) => r,
        None => {
            return numbered_single_fallback(title, &all_items, cancel_idx);
        }
    };

    match selected {
        Some(idx) if idx >= cancel_idx => None,
        other => other,
    }
}

/// Outer `Option` distinguishes "raw mode unavailable" (`None`) from a
/// completed loop; the inner `Option<usize>` is the curses result holder
/// (`None` for ESC/q, `Some(idx)` for Enter).
fn single_select_interactive(
    title: &str,
    all_items: &[String],
    default_index: usize,
) -> Option<Option<usize>> {
    let _raw = RawMode::enter()?;
    screen_setup();

    let colored = has_colors();
    let mut cursor: usize = default_index.min(all_items.len() - 1);
    let mut scroll_offset: usize = 0;
    let mut stdin = io::stdin();

    let result: Option<usize> = loop {
        let mut frame = Frame::new(colored);
        frame.clear();
        let (max_y, max_x) = term_size();

        let mut hattr: Vec<&str> = vec![Colors::BOLD];
        if colored {
            hattr.push(Colors::YELLOW);
        }
        frame.add(0, 0, title, max_x.saturating_sub(1), &hattr);
        frame.add(
            1,
            0,
            "  \u{2191}\u{2193} navigate  ENTER confirm  ESC/q cancel",
            max_x.saturating_sub(1),
            &[Colors::DIM],
        );

        let visible_rows = max_y.saturating_sub(3);
        if cursor < scroll_offset {
            scroll_offset = cursor;
        } else if visible_rows > 0 && cursor >= scroll_offset + visible_rows {
            scroll_offset = cursor - visible_rows + 1;
        }

        let end = all_items.len().min(scroll_offset + visible_rows);
        for (draw_i, i) in (scroll_offset..end).enumerate() {
            let y = draw_i + 3;
            if y >= max_y.saturating_sub(1) {
                break;
            }
            let arrow = if i == cursor { "\u{2192}" } else { " " };
            let line = format!(" {} {}", arrow, all_items[i]);
            let mut attr: Vec<&str> = Vec::new();
            if i == cursor {
                attr.push(Colors::BOLD);
                if colored {
                    attr.push(Colors::GREEN);
                }
            }
            frame.add(y, 0, &line, max_x.saturating_sub(1), &attr);
        }

        frame.refresh();

        match read_key(&mut stdin) {
            Key::Up => cursor = (cursor + all_items.len() - 1) % all_items.len(),
            Key::Down => cursor = (cursor + 1) % all_items.len(),
            Key::Enter => break Some(cursor),
            Key::Cancel => break None,
            _ => {}
        }
    };

    screen_teardown();
    flush_stdin();
    Some(result)
}

/// Text-based numbered fallback for single-select.
/// Direct port of Python `_numbered_single_fallback`.
fn numbered_single_fallback(
    title: &str,
    items: &[String],
    cancel_idx: usize,
) -> Option<usize> {
    println!("\n  {}\n", title);
    for (i, label) in items.iter().enumerate() {
        println!("  {}. {}", i + 1, label);
    }
    println!();

    let prompt = format!("  Choice [1-{}]: ", items.len());
    if let Some(val) = read_line(&prompt) {
        let val = val.trim();
        if val.is_empty() {
            return None;
        }
        if let Ok(n) = val.parse::<i64>() {
            let idx = n - 1;
            if idx >= 0 && (idx as usize) < items.len() && (idx as usize) < cancel_idx {
                return Some(idx as usize);
            }
            if idx >= 0 && idx as usize == cancel_idx {
                return None;
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// _numbered_fallback — text-based toggle fallback for checklist
// ---------------------------------------------------------------------------

/// Text-based toggle fallback for terminals without interactive support.
/// Direct port of Python `_numbered_fallback`.
fn numbered_fallback(
    title: &str,
    items: &[String],
    selected: &BTreeSet<usize>,
    cancel_returns: &BTreeSet<usize>,
    status_fn: Option<&dyn Fn(&BTreeSet<usize>) -> String>,
) -> BTreeSet<usize> {
    let mut chosen: BTreeSet<usize> = selected.clone();
    println!("{}", color(&format!("\n  {}", title), &[Colors::YELLOW]));
    println!(
        "{}",
        color("  Toggle by number, Enter to confirm.\n", &[Colors::DIM])
    );

    loop {
        for (i, label) in items.iter().enumerate() {
            let marker = if chosen.contains(&i) {
                color("[\u{2713}]", &[Colors::GREEN])
            } else {
                "[ ]".to_string()
            };
            println!("  {} {:>2}. {}", marker, i + 1, label);
        }
        if let Some(sf) = status_fn {
            let status_text = sf(&chosen);
            if !status_text.is_empty() {
                println!(
                    "{}",
                    color(&format!("\n  {}", status_text), &[Colors::DIM])
                );
            }
        }
        println!();

        let prompt = color("  Toggle # (or Enter to confirm): ", &[Colors::DIM]);
        match read_line(&prompt) {
            Some(val) => {
                let val = val.trim();
                if val.is_empty() {
                    break;
                }
                match val.parse::<i64>() {
                    Ok(n) => {
                        let idx = n - 1;
                        if idx >= 0 && (idx as usize) < items.len() {
                            let i = idx as usize;
                            if chosen.contains(&i) {
                                chosen.remove(&i);
                            } else {
                                chosen.insert(i);
                            }
                        }
                    }
                    Err(_) => return cancel_returns.clone(),
                }
            }
            None => return cancel_returns.clone(), // EOF / interrupt
        }
        println!();
    }

    chosen
}

/// Print `prompt` (no newline) and read one line from stdin.
///
/// Returns `None` on EOF (mirrors Python's `EOFError`); a successful read
/// returns the line **without** the trailing newline. A failed numeric parse
/// is handled by callers (mapping to Python's `ValueError` branch).
fn read_line(prompt: &str) -> Option<String> {
    {
        let mut out = io::stdout();
        let _ = out.write_all(prompt.as_bytes());
        let _ = out.flush();
    }
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => {
            // Strip a single trailing newline (and CR), leaving inner content
            // for the caller's `.trim()`.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_cols_respects_char_width() {
        assert_eq!(truncate_cols("hello", 3), "hel");
        assert_eq!(truncate_cols("hello", 10), "hello");
        assert_eq!(truncate_cols("hello", 0), "");
        // Multi-byte glyphs count as one column each.
        assert_eq!(truncate_cols("\u{2192}\u{2713}x", 2), "\u{2192}\u{2713}");
    }

    #[test]
    fn display_len_counts_scalars_not_bytes() {
        assert_eq!(display_len("abc"), 3);
        // → and ✓ are multi-byte but single-column.
        assert_eq!(display_len("\u{2192}\u{2713}"), 2);
    }

    #[test]
    fn non_tty_checklist_returns_cancel_default() {
        // In CI stdin is not a TTY: must return the cancel default immediately
        // without attempting any interactive work.
        let items = vec!["a".to_string(), "b".to_string()];
        let selected: BTreeSet<usize> = [0usize].into_iter().collect();
        let got = curses_checklist("Title", &items, &selected, None, None);
        assert_eq!(got, selected);
    }

    #[test]
    fn non_tty_checklist_uses_explicit_cancel_returns() {
        let items = vec!["a".to_string(), "b".to_string()];
        let selected: BTreeSet<usize> = [0usize].into_iter().collect();
        let cancel: BTreeSet<usize> = [1usize].into_iter().collect();
        let got = curses_checklist("Title", &items, &selected, Some(&cancel), None);
        assert_eq!(got, cancel);
    }

    #[test]
    fn non_tty_radiolist_returns_selected_default() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(curses_radiolist("T", &items, 2, None, None), 2);
        // Explicit cancel_returns override.
        assert_eq!(curses_radiolist("T", &items, 2, Some(0), None), 0);
    }

    #[test]
    fn non_tty_single_select_returns_none() {
        let items = vec!["a".to_string(), "b".to_string()];
        assert_eq!(curses_single_select("T", &items, 0, "Cancel"), None);
    }

    #[test]
    fn flush_stdin_is_noop_off_tty() {
        // Must not panic when stdin is not a TTY (CI environment).
        flush_stdin();
    }

    #[test]
    fn wrap_arithmetic_matches_python_modulo() {
        // Reproduce the cursor wrap semantics: (cursor - 1) mod len and
        // (cursor + 1) mod len, computed the way the loops do.
        let len = 3usize;
        let up = |c: usize| (c + len - 1) % len;
        let down = |c: usize| (c + 1) % len;
        assert_eq!(up(0), 2);
        assert_eq!(up(1), 0);
        assert_eq!(down(2), 0);
        assert_eq!(down(0), 1);
    }

    #[test]
    fn empty_items_checklist_returns_cancel() {
        // Off-TTY path returns cancel; the empty-items guard also returns cancel.
        let items: Vec<String> = Vec::new();
        let selected: BTreeSet<usize> = BTreeSet::new();
        let got = curses_checklist("T", &items, &selected, None, None);
        assert!(got.is_empty());
    }
}
