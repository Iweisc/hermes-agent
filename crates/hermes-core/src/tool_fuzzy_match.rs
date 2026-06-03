//! Fuzzy Matching Module for File Operations
//!
//! Port of `tools/fuzzy_match.py`.
//!
//! Implements a multi-strategy matching chain to robustly find and replace
//! text, accommodating variations in whitespace, indentation, and escaping
//! common in LLM-generated code.
//!
//! The strategy chain (inspired by OpenCode), tried in order:
//! 1. Exact match - Direct string comparison
//! 2. Line-trimmed - Strip leading/trailing whitespace per line
//! 3. Whitespace normalized - Collapse multiple spaces/tabs to single space
//! 4. Indentation flexible - Ignore indentation differences entirely
//! 5. Escape normalized - Convert `\n` literals to actual newlines
//! 6. Trimmed boundary - Trim first/last line whitespace only
//! 7. Unicode normalized - smart quotes / dashes / ellipsis -> ASCII
//! 8. Block anchor - Match first+last lines, use similarity for middle
//! 9. Context-aware - 50% line similarity threshold
//!
//! Multi-occurrence matching is handled via the `replace_all` flag.
//!
//! NOTE ON INDEXING: Python strings index by Unicode code point. To reproduce
//! the exact `(start, end)` offsets the Python implementation produces, this
//! port operates on `Vec<char>` (code points) for all position arithmetic and
//! converts back to `String` only at the boundaries. All returned `(start,
//! end)` positions are therefore **character** offsets, matching Python.

use std::collections::HashMap;

/// Unicode -> ASCII replacement map. Order matters only in that all entries
/// are applied; replacements never produce new map keys.
const UNICODE_MAP: &[(char, &str)] = &[
    ('\u{201c}', "\""), // smart left double quote
    ('\u{201d}', "\""), // smart right double quote
    ('\u{2018}', "'"),  // smart left single quote
    ('\u{2019}', "'"),  // smart right single quote
    ('\u{2014}', "--"), // em dash
    ('\u{2013}', "-"),  // en dash
    ('\u{2026}', "..."), // ellipsis
    ('\u{00a0}', " "),  // non-breaking space
];

fn unicode_repl(c: char) -> Option<&'static str> {
    UNICODE_MAP
        .iter()
        .find(|(k, _)| *k == c)
        .map(|(_, v)| *v)
}

/// Normalizes Unicode characters to their standard ASCII equivalents.
pub fn unicode_normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match unicode_repl(c) {
            Some(repl) => out.push_str(repl),
            None => out.push(c),
        }
    }
    out
}

/// Result of a fuzzy find-and-replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyResult {
    /// Resulting content (modified on success, original on failure).
    pub content: String,
    /// Number of replacements performed.
    pub match_count: usize,
    /// Name of the strategy that matched, if any.
    pub strategy: Option<String>,
    /// Error description, if the operation failed.
    pub error: Option<String>,
}

/// Convenience: did the operation succeed (a match was applied)?
impl FuzzyResult {
    pub fn is_ok(&self) -> bool {
        self.error.is_none() && self.match_count > 0
    }
}

// ============================================================================
// Public entry point
// ============================================================================

/// Find and replace text using a chain of increasingly fuzzy matching
/// strategies.
///
/// On success returns the modified content, the number of replacements, the
/// strategy used, and `None` error. On failure returns the original content,
/// `0`, `None` strategy, and an error description.
pub fn fuzzy_find_and_replace(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> FuzzyResult {
    let fail = |err: &str| FuzzyResult {
        content: content.to_string(),
        match_count: 0,
        strategy: None,
        error: Some(err.to_string()),
    };

    if old_string.is_empty() {
        return fail("old_string cannot be empty");
    }
    if old_string == new_string {
        return fail("old_string and new_string are identical");
    }

    let content_chars: Vec<char> = content.chars().collect();

    type StrategyFn = fn(&str, &str) -> Vec<(usize, usize)>;
    let strategies: &[(&str, StrategyFn)] = &[
        ("exact", strategy_exact),
        ("line_trimmed", strategy_line_trimmed),
        ("whitespace_normalized", strategy_whitespace_normalized),
        ("indentation_flexible", strategy_indentation_flexible),
        ("escape_normalized", strategy_escape_normalized),
        ("trimmed_boundary", strategy_trimmed_boundary),
        ("unicode_normalized", strategy_unicode_normalized),
        ("block_anchor", strategy_block_anchor),
        ("context_aware", strategy_context_aware),
    ];

    for (strategy_name, strategy_fn) in strategies {
        let matches = strategy_fn(content, old_string);
        if matches.is_empty() {
            continue;
        }

        if matches.len() > 1 && !replace_all {
            return fail(&format!(
                "Found {} matches for old_string. Provide more context to make it unique, or use replace_all=True.",
                matches.len()
            ));
        }

        // Escape-drift guard for non-exact strategies.
        if *strategy_name != "exact" {
            if let Some(drift_err) =
                detect_escape_drift(&content_chars, &matches, old_string, new_string)
            {
                return fail(&drift_err);
            }
        }

        let new_content = apply_replacements(&content_chars, &matches, new_string);
        return FuzzyResult {
            content: new_content,
            match_count: matches.len(),
            strategy: Some((*strategy_name).to_string()),
            error: None,
        };
    }

    fail("Could not find a match for old_string in the file")
}

/// Detect tool-call escape-drift artifacts in `new_string`.
///
/// Looks for `\'` or `\"` sequences present in both `old_string` and
/// `new_string` but absent from the matched region of the file — a signal
/// that the transport layer inserted spurious shell-style escapes.
fn detect_escape_drift(
    content_chars: &[char],
    matches: &[(usize, usize)],
    old_string: &str,
    new_string: &str,
) -> Option<String> {
    // Cheap pre-check.
    if !new_string.contains("\\'") && !new_string.contains("\\\"") {
        return None;
    }

    // Aggregate matched regions (character slices) of the file.
    let mut matched_regions = String::new();
    for &(start, end) in matches {
        let s = start.min(content_chars.len());
        let e = end.min(content_chars.len());
        if s < e {
            matched_regions.extend(content_chars[s..e].iter());
        }
    }

    for suspect in ["\\'", "\\\""] {
        if new_string.contains(suspect)
            && old_string.contains(suspect)
            && !matched_regions.contains(suspect)
        {
            // suspect[1] in Python: the bare quote char.
            let plain = &suspect[1..];
            return Some(format!(
                "Escape-drift detected: old_string and new_string contain the literal sequence '{}' but the matched region of the file does not. This is almost always a tool-call serialization artifact where an apostrophe or quote got prefixed with a spurious backslash. Re-read the file with read_file and pass old_string/new_string without backslash-escaping '{}' characters.",
                suspect, plain
            ));
        }
    }
    None
}

/// Apply replacements at the given character positions, working end-to-start
/// so earlier positions remain valid.
fn apply_replacements(
    content_chars: &[char],
    matches: &[(usize, usize)],
    new_string: &str,
) -> String {
    let mut sorted: Vec<(usize, usize)> = matches.to_vec();
    sorted.sort_by(|a, b| b.0.cmp(&a.0)); // descending by start

    let mut result: Vec<char> = content_chars.to_vec();
    let new_chars: Vec<char> = new_string.chars().collect();
    for (start, end) in sorted {
        let s = start.min(result.len());
        let e = end.min(result.len());
        let (s, e) = if s <= e { (s, e) } else { (s, s) };
        result.splice(s..e, new_chars.iter().cloned());
    }
    result.into_iter().collect()
}

// ============================================================================
// Matching Strategies
// ============================================================================

/// Strategy 1: Exact string match. Returns character offsets.
fn strategy_exact(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let content_chars: Vec<char> = content.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    char_find_all(&content_chars, &pattern_chars)
}

/// Find all occurrences of `pattern` in `content` (both char slices),
/// advancing by 1 char after each match (mirrors Python `start = pos + 1`).
fn char_find_all(content: &[char], pattern: &[char]) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    if pattern.is_empty() {
        return matches;
    }
    let n = content.len();
    let m = pattern.len();
    if m > n {
        return matches;
    }
    let mut start = 0usize;
    while start + m <= n {
        let mut found = None;
        let mut i = start;
        while i + m <= n {
            if content[i..i + m] == *pattern {
                found = Some(i);
                break;
            }
            i += 1;
        }
        match found {
            Some(pos) => {
                matches.push((pos, pos + m));
                start = pos + 1;
            }
            None => break,
        }
    }
    matches
}

/// Strategy 2: Match with line-by-line whitespace trimming.
fn strategy_line_trimmed(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_lines: Vec<String> = pattern.split('\n').map(|l| py_strip(l)).collect();
    let pattern_normalized = pattern_lines.join("\n");

    let content_lines: Vec<&str> = content.split('\n').collect();
    let content_normalized_lines: Vec<String> =
        content_lines.iter().map(|l| py_strip(l)).collect();

    find_normalized_matches(
        content,
        &content_lines,
        &content_normalized_lines,
        &pattern_normalized,
    )
}

/// Strategy 3: Collapse multiple whitespace (spaces/tabs) to single space.
fn strategy_whitespace_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_normalized = collapse_ws(pattern);
    let content_normalized = collapse_ws(content);

    let cn_chars: Vec<char> = content_normalized.chars().collect();
    let pn_chars: Vec<char> = pattern_normalized.chars().collect();
    let matches_in_normalized = char_find_all(&cn_chars, &pn_chars);
    if matches_in_normalized.is_empty() {
        return Vec::new();
    }
    map_normalized_positions(content, &content_normalized, &matches_in_normalized)
}

/// Collapse runs of spaces/tabs to a single space (preserving newlines), like
/// Python `re.sub(r'[ \t]+', ' ', s)`.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            in_ws = false;
            out.push(c);
        }
    }
    out
}

/// Strategy 4: Ignore indentation differences entirely (lstrip each line).
fn strategy_indentation_flexible(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let content_lines: Vec<&str> = content.split('\n').collect();
    let content_stripped_lines: Vec<String> =
        content_lines.iter().map(|l| py_lstrip(l)).collect();
    let pattern_lines: Vec<String> = pattern.split('\n').map(|l| py_lstrip(l)).collect();
    let pattern_normalized = pattern_lines.join("\n");

    find_normalized_matches(
        content,
        &content_lines,
        &content_stripped_lines,
        &pattern_normalized,
    )
}

/// Strategy 5: Convert escape sequences (`\n`, `\t`, `\r`) to real chars.
fn strategy_escape_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_unescaped = pattern
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r");

    if pattern_unescaped == pattern {
        return Vec::new();
    }
    strategy_exact(content, &pattern_unescaped)
}

/// Strategy 6: Trim whitespace from first and last lines only.
fn strategy_trimmed_boundary(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let mut pattern_lines: Vec<String> = pattern.split('\n').map(|s| s.to_string()).collect();
    if pattern_lines.is_empty() {
        return Vec::new();
    }
    let n_lines = pattern_lines.len();
    pattern_lines[0] = py_strip(&pattern_lines[0]);
    if n_lines > 1 {
        let last = n_lines - 1;
        pattern_lines[last] = py_strip(&pattern_lines[last]);
    }
    let modified_pattern = pattern_lines.join("\n");

    let content_lines: Vec<&str> = content.split('\n').collect();
    let content_len = content.chars().count();

    let mut matches = Vec::new();
    let pattern_line_count = n_lines;
    if content_lines.len() < pattern_line_count {
        return matches;
    }
    for i in 0..=(content_lines.len() - pattern_line_count) {
        let mut check_lines: Vec<String> = content_lines[i..i + pattern_line_count]
            .iter()
            .map(|s| s.to_string())
            .collect();
        check_lines[0] = py_strip(&check_lines[0]);
        if check_lines.len() > 1 {
            let last = check_lines.len() - 1;
            check_lines[last] = py_strip(&check_lines[last]);
        }
        if check_lines.join("\n") == modified_pattern {
            let (start_pos, end_pos) = calculate_line_positions(
                &content_lines,
                i,
                i + pattern_line_count,
                content_len,
            );
            matches.push((start_pos, end_pos));
        }
    }
    matches
}

/// Build a list mapping each original char index to its normalized index,
/// accounting for UNICODE_MAP expansions. Length = `original.len() + 1`.
fn build_orig_to_norm_map(original_chars: &[char]) -> Vec<usize> {
    let mut result = Vec::with_capacity(original_chars.len() + 1);
    let mut norm_pos = 0usize;
    for &c in original_chars {
        result.push(norm_pos);
        norm_pos += match unicode_repl(c) {
            Some(repl) => repl.chars().count(),
            None => 1,
        };
    }
    result.push(norm_pos); // sentinel
    result
}

/// Convert (start, end) positions in the normalized string to original
/// positions using the orig->norm map.
fn map_positions_norm_to_orig(
    orig_to_norm: &[usize],
    norm_matches: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    // norm_pos -> first original position with that norm_pos.
    let mut norm_to_orig_start: HashMap<usize, usize> = HashMap::new();
    let orig_len = orig_to_norm.len() - 1; // number of original chars
    for (orig_pos, &norm_pos) in orig_to_norm[..orig_len].iter().enumerate() {
        norm_to_orig_start.entry(norm_pos).or_insert(orig_pos);
    }

    let mut results = Vec::new();
    for &(norm_start, norm_end) in norm_matches {
        let orig_start = match norm_to_orig_start.get(&norm_start) {
            Some(&p) => p,
            None => continue,
        };
        let mut orig_end = orig_start;
        while orig_end < orig_len && orig_to_norm[orig_end] < norm_end {
            orig_end += 1;
        }
        results.push((orig_start, orig_end));
    }
    results
}

/// Strategy 7: Unicode normalisation (smart quotes / dashes / ellipsis / nbsp).
fn strategy_unicode_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let norm_pattern = unicode_normalize(pattern);
    let norm_content = unicode_normalize(content);
    if norm_content == content && norm_pattern == pattern {
        return Vec::new();
    }

    let mut norm_matches = strategy_exact(&norm_content, &norm_pattern);
    if norm_matches.is_empty() {
        norm_matches = strategy_line_trimmed(&norm_content, &norm_pattern);
    }
    if norm_matches.is_empty() {
        return Vec::new();
    }

    let content_chars: Vec<char> = content.chars().collect();
    let orig_to_norm = build_orig_to_norm_map(&content_chars);
    map_positions_norm_to_orig(&orig_to_norm, &norm_matches)
}

/// Strategy 8: Match by anchoring on first and last lines, similarity middle.
fn strategy_block_anchor(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let norm_pattern = unicode_normalize(pattern);
    let norm_content = unicode_normalize(content);

    let pattern_lines: Vec<&str> = norm_pattern.split('\n').collect();
    if pattern_lines.len() < 2 {
        return Vec::new();
    }
    let first_line = py_strip(pattern_lines[0]);
    let last_line = py_strip(pattern_lines[pattern_lines.len() - 1]);

    let norm_content_lines: Vec<&str> = norm_content.split('\n').collect();
    let orig_content_lines: Vec<&str> = content.split('\n').collect();
    let content_len = content.chars().count();

    let pattern_line_count = pattern_lines.len();
    if norm_content_lines.len() < pattern_line_count {
        return Vec::new();
    }

    let mut potential_matches = Vec::new();
    for i in 0..=(norm_content_lines.len() - pattern_line_count) {
        if py_strip(norm_content_lines[i]) == first_line
            && py_strip(norm_content_lines[i + pattern_line_count - 1]) == last_line
        {
            potential_matches.push(i);
        }
    }

    let candidate_count = potential_matches.len();
    let threshold = if candidate_count == 1 { 0.50 } else { 0.70 };

    let mut matches = Vec::new();
    for &i in &potential_matches {
        let similarity = if pattern_line_count <= 2 {
            1.0
        } else {
            let content_middle = norm_content_lines[i + 1..i + pattern_line_count - 1].join("\n");
            let pattern_middle = pattern_lines[1..pattern_lines.len() - 1].join("\n");
            seq_ratio(&content_middle, &pattern_middle)
        };
        if similarity >= threshold {
            let (start_pos, end_pos) = calculate_line_positions(
                &orig_content_lines,
                i,
                i + pattern_line_count,
                content_len,
            );
            matches.push((start_pos, end_pos));
        }
    }
    matches
}

/// Strategy 9: Line-by-line similarity with 50% threshold.
fn strategy_context_aware(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_lines: Vec<&str> = pattern.split('\n').collect();
    let content_lines: Vec<&str> = content.split('\n').collect();
    if pattern_lines.is_empty() {
        return Vec::new();
    }
    let content_len = content.chars().count();
    let pattern_line_count = pattern_lines.len();
    if content_lines.len() < pattern_line_count {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for i in 0..=(content_lines.len() - pattern_line_count) {
        let block_lines = &content_lines[i..i + pattern_line_count];
        let mut high_similarity_count = 0usize;
        // zip(pattern_lines, block_lines) — equal length here.
        for (p_line, c_line) in pattern_lines.iter().zip(block_lines.iter()) {
            let sim = seq_ratio(&py_strip(p_line), &py_strip(c_line));
            if sim >= 0.80 {
                high_similarity_count += 1;
            }
        }
        if (high_similarity_count as f64) >= (pattern_lines.len() as f64) * 0.5 {
            let (start_pos, end_pos) = calculate_line_positions(
                &content_lines,
                i,
                i + pattern_line_count,
                content_len,
            );
            matches.push((start_pos, end_pos));
        }
    }
    matches
}

// ============================================================================
// Helper functions
// ============================================================================

/// Calculate start/end char positions from line indices, where lines were
/// produced by splitting on `\n`. Mirrors Python `_calculate_line_positions`.
fn calculate_line_positions(
    content_lines: &[&str],
    start_line: usize,
    end_line: usize,
    content_length: usize,
) -> (usize, usize) {
    let line_len = |s: &str| s.chars().count();
    let start_pos: usize = content_lines[..start_line]
        .iter()
        .map(|l| line_len(l) + 1)
        .sum();
    let mut end_pos: isize = content_lines[..end_line]
        .iter()
        .map(|l| line_len(l) as isize + 1)
        .sum::<isize>()
        - 1;
    if end_pos >= content_length as isize {
        end_pos = content_length as isize;
    }
    let end_pos = if end_pos < 0 { 0 } else { end_pos as usize };
    (start_pos, end_pos)
}

/// Find matches in normalized content (line-based) and map back to original
/// char positions. Mirrors Python `_find_normalized_matches`.
fn find_normalized_matches(
    content: &str,
    content_lines: &[&str],
    content_normalized_lines: &[String],
    pattern_normalized: &str,
) -> Vec<(usize, usize)> {
    let pattern_norm_lines: Vec<&str> = pattern_normalized.split('\n').collect();
    let num_pattern_lines = pattern_norm_lines.len();
    let content_len = content.chars().count();

    let mut matches = Vec::new();
    if content_normalized_lines.len() < num_pattern_lines {
        return matches;
    }
    for i in 0..=(content_normalized_lines.len() - num_pattern_lines) {
        let block = content_normalized_lines[i..i + num_pattern_lines].join("\n");
        if block == pattern_normalized {
            let (start_pos, end_pos) =
                calculate_line_positions(content_lines, i, i + num_pattern_lines, content_len);
            matches.push((start_pos, end_pos));
        }
    }
    matches
}

/// Map positions from whitespace-normalized string back to original. Mirrors
/// Python `_map_normalized_positions` (best-effort for ws normalization).
fn map_normalized_positions(
    original: &str,
    normalized: &str,
    normalized_matches: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    if normalized_matches.is_empty() {
        return Vec::new();
    }
    let orig: Vec<char> = original.chars().collect();
    let norm: Vec<char> = normalized.chars().collect();
    let orig_n = orig.len();
    let norm_n = norm.len();

    let mut orig_to_norm: Vec<usize> = Vec::with_capacity(orig_n);
    let mut orig_idx = 0usize;
    let mut norm_idx = 0usize;

    let is_sp_tab = |c: char| c == ' ' || c == '\t';

    while orig_idx < orig_n && norm_idx < norm_n {
        if orig[orig_idx] == norm[norm_idx] {
            orig_to_norm.push(norm_idx);
            orig_idx += 1;
            norm_idx += 1;
        } else if is_sp_tab(orig[orig_idx]) && norm[norm_idx] == ' ' {
            orig_to_norm.push(norm_idx);
            orig_idx += 1;
            if orig_idx < orig_n && !is_sp_tab(orig[orig_idx]) {
                norm_idx += 1;
            }
        } else if is_sp_tab(orig[orig_idx]) {
            orig_to_norm.push(norm_idx);
            orig_idx += 1;
        } else {
            // Mismatch - shouldn't happen with our normalization.
            orig_to_norm.push(norm_idx);
            orig_idx += 1;
        }
    }
    // Fill remaining.
    while orig_idx < orig_n {
        orig_to_norm.push(norm_n);
        orig_idx += 1;
    }

    // Reverse mapping.
    let mut norm_to_orig_start: HashMap<usize, usize> = HashMap::new();
    let mut norm_to_orig_end: HashMap<usize, usize> = HashMap::new();
    for (orig_pos, &norm_pos) in orig_to_norm.iter().enumerate() {
        norm_to_orig_start.entry(norm_pos).or_insert(orig_pos);
        norm_to_orig_end.insert(norm_pos, orig_pos);
    }

    let mut original_matches = Vec::new();
    for &(norm_start, norm_end) in normalized_matches {
        let orig_start = match norm_to_orig_start.get(&norm_start) {
            Some(&p) => p,
            None => {
                // Find nearest: min index i where orig_to_norm[i] >= norm_start.
                match orig_to_norm.iter().position(|&n| n >= norm_start) {
                    Some(p) => p,
                    None => continue,
                }
            }
        };

        let mut orig_end = if norm_end >= 1 {
            match norm_to_orig_end.get(&(norm_end - 1)) {
                Some(&p) => p + 1,
                None => orig_start + (norm_end - norm_start),
            }
        } else {
            // norm_end == 0: Python would look up key (-1); never present.
            orig_start + (norm_end - norm_start)
        };

        // Expand to include trailing whitespace that was normalized.
        while orig_end < orig_n && is_sp_tab(orig[orig_end]) {
            orig_end += 1;
        }

        original_matches.push((orig_start, orig_end.min(orig_n)));
    }
    original_matches
}

/// Find lines in content most similar to old_string for "did you mean?"
/// feedback. Returns a formatted string, or empty when nothing useful.
pub fn find_closest_lines(
    old_string: &str,
    content: &str,
    context_lines: usize,
    max_results: usize,
) -> String {
    if old_string.is_empty() || content.is_empty() {
        return String::new();
    }

    let old_lines: Vec<&str> = py_splitlines(old_string);
    let content_lines: Vec<&str> = py_splitlines(content);
    if old_lines.is_empty() || content_lines.is_empty() {
        return String::new();
    }

    // Anchor = first non-blank stripped line (preferring line 0).
    let mut anchor = py_strip(old_lines[0]);
    if anchor.is_empty() {
        let candidate = old_lines
            .iter()
            .map(|l| py_strip(l))
            .find(|s| !s.is_empty());
        match candidate {
            Some(c) => anchor = c,
            None => return String::new(),
        }
    }

    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (i, line) in content_lines.iter().enumerate() {
        let stripped = py_strip(line);
        if stripped.is_empty() {
            continue;
        }
        let ratio = seq_ratio(&anchor, &stripped);
        if ratio > 0.3 {
            scored.push((ratio, i));
        }
    }
    if scored.is_empty() {
        return String::new();
    }

    // Sort by descending ratio. Python's sort is stable; ties keep input order.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(f64, usize)> = scored.into_iter().take(max_results).collect();

    let mut parts: Vec<String> = Vec::new();
    let mut seen_ranges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for (_, line_idx) in top {
        let start = line_idx.saturating_sub(context_lines);
        let end = content_lines
            .len()
            .min(line_idx + old_lines.len() + context_lines);
        let key = (start, end);
        if seen_ranges.contains(&key) {
            continue;
        }
        seen_ranges.insert(key);
        let mut snippet_lines = Vec::new();
        for j in 0..(end - start) {
            snippet_lines.push(format!("{:4}| {}", start + j + 1, content_lines[start + j]));
        }
        parts.push(snippet_lines.join("\n"));
    }

    if parts.is_empty() {
        return String::new();
    }
    parts.join("\n---\n")
}

/// Return a "Did you mean..." snippet for plain no-match errors, gated so it
/// only fires for actual "Could not find" failures.
pub fn format_no_match_hint(
    error: Option<&str>,
    match_count: usize,
    old_string: &str,
    content: &str,
) -> String {
    if match_count != 0 {
        return String::new();
    }
    match error {
        Some(e) if e.starts_with("Could not find") => {}
        _ => return String::new(),
    }
    let hint = find_closest_lines(old_string, content, 2, 3);
    if hint.is_empty() {
        return String::new();
    }
    format!("\n\nDid you mean one of these sections?\n{}", hint)
}

// ============================================================================
// String helpers reproducing Python str semantics
// ============================================================================

/// Python `str.strip()` — strips Unicode whitespace from both ends.
fn py_strip(s: &str) -> String {
    s.trim_matches(|c: char| c.is_whitespace()).to_string()
}

/// Python `str.lstrip()` — strips Unicode whitespace from the left.
fn py_lstrip(s: &str) -> String {
    s.trim_start_matches(|c: char| c.is_whitespace()).to_string()
}

/// Python `str.splitlines()` — splits on universal newlines, no trailing
/// empty element. We support the common `\n`, `\r\n`, `\r` cases.
fn py_splitlines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let bytes = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        let b = bytes[i];
        if b == b'\n' {
            lines.push(&s[start..i]);
            i += 1;
            start = i;
        } else if b == b'\r' {
            lines.push(&s[start..i]);
            if i + 1 < n && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < n {
        lines.push(&s[start..n]);
    }
    lines
}

// ============================================================================
// difflib.SequenceMatcher.ratio() port
// ============================================================================

/// Compute `difflib.SequenceMatcher(None, a, b).ratio()`.
///
/// ratio = 2.0 * M / T, where T = len(a) + len(b) and M is the total number
/// of matched characters found by the recursive longest-matching-block
/// algorithm (with autojunk for the `b` sequence, matching CPython defaults).
pub fn seq_ratio(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let length = a_chars.len() + b_chars.len();
    if length == 0 {
        return 1.0;
    }
    let matches = SequenceMatcher::new(&a_chars, &b_chars).total_matches();
    2.0 * (matches as f64) / (length as f64)
}

struct SequenceMatcher<'a> {
    a: &'a [char],
    b: &'a [char],
    /// b2j: char -> indices in b (after junk/popular removal).
    b2j: HashMap<char, Vec<usize>>,
}

impl<'a> SequenceMatcher<'a> {
    fn new(a: &'a [char], b: &'a [char]) -> Self {
        let mut sm = SequenceMatcher {
            a,
            b,
            b2j: HashMap::new(),
        };
        sm.chain_b();
        sm
    }

    /// Build b2j, applying CPython's "autojunk" heuristic: when b has more than
    /// 200 elements, any element occurring more than 1% of the time (strictly:
    /// count > n/100 + 1) is treated as popular and removed from b2j.
    fn chain_b(&mut self) {
        let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, &elt) in self.b.iter().enumerate() {
            b2j.entry(elt).or_default().push(i);
        }

        let n = self.b.len();
        // No explicit isjunk function here (None), so only autojunk applies.
        if n >= 200 {
            let ntest = n / 100 + 1;
            let popular: Vec<char> = b2j
                .iter()
                .filter(|(_, idxs)| idxs.len() > ntest)
                .map(|(&c, _)| c)
                .collect();
            for c in popular {
                b2j.remove(&c);
            }
        }
        self.b2j = b2j;
    }

    /// Find the longest matching block in a[alo..ahi] and b[blo..bhi].
    /// Returns (i, j, k): a[i..i+k] == b[j..j+k].
    fn find_longest_match(
        &self,
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
    ) -> (usize, usize, usize) {
        let mut besti = alo;
        let mut bestj = blo;
        let mut bestsize = 0usize;

        // j2len: for the current i, maps j -> length of match ending at (i, j).
        let mut j2len: HashMap<usize, usize> = HashMap::new();

        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(&self.a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = if j == 0 {
                        1
                    } else {
                        j2len.get(&(j - 1)).copied().unwrap_or(0) + 1
                    };
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }

        // Note: CPython extends the match over junk on the boundaries here, but
        // with isjunk=None and elements removed via autojunk treated as
        // "popular" (not junk for this extension), there is no junk set, so the
        // junk-extension loops are no-ops. We keep the bare block.
        (besti, bestj, bestsize)
    }

    /// Total number of matched chars across all matching blocks.
    fn total_matches(&self) -> usize {
        let mut total = 0usize;
        // Iterative stack to mirror get_matching_blocks recursion.
        let mut queue: Vec<(usize, usize, usize, usize)> =
            vec![(0, self.a.len(), 0, self.b.len())];
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.find_longest_match(alo, ahi, blo, bhi);
            if k > 0 {
                total += k;
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        total
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(r: &FuzzyResult) -> &str {
        assert!(r.error.is_none(), "unexpected error: {:?}", r.error);
        &r.content
    }

    #[test]
    fn test_empty_old_string() {
        let r = fuzzy_find_and_replace("abc", "", "x", false);
        assert_eq!(r.error.as_deref(), Some("old_string cannot be empty"));
        assert_eq!(r.match_count, 0);
    }

    #[test]
    fn test_identical_strings() {
        let r = fuzzy_find_and_replace("abc", "a", "a", false);
        assert_eq!(
            r.error.as_deref(),
            Some("old_string and new_string are identical")
        );
    }

    #[test]
    fn test_exact_match() {
        let r = fuzzy_find_and_replace("def foo():\n    pass", "def foo():", "def bar():", false);
        assert_eq!(r.strategy.as_deref(), Some("exact"));
        assert_eq!(r.match_count, 1);
        assert_eq!(ok(&r), "def bar():\n    pass");
    }

    #[test]
    fn test_multiple_matches_no_replace_all() {
        let r = fuzzy_find_and_replace("a\na\na", "a", "b", false);
        assert!(r.error.as_deref().unwrap().starts_with("Found 3 matches"));
        assert_eq!(r.match_count, 0);
        assert_eq!(r.content, "a\na\na");
    }

    #[test]
    fn test_multiple_matches_replace_all() {
        let r = fuzzy_find_and_replace("a\na\na", "a", "b", true);
        assert_eq!(r.match_count, 3);
        assert_eq!(ok(&r), "b\nb\nb");
    }

    #[test]
    fn test_no_match() {
        let r = fuzzy_find_and_replace("hello world", "zzz", "q", false);
        assert_eq!(
            r.error.as_deref(),
            Some("Could not find a match for old_string in the file")
        );
    }

    #[test]
    fn test_line_trimmed() {
        let content = "    def foo():\n        pass";
        let r = fuzzy_find_and_replace(content, "def foo():\n    pass", "X", false);
        assert_eq!(r.strategy.as_deref(), Some("line_trimmed"));
        assert_eq!(r.match_count, 1);
        // Replaces the whole matched block (both lines).
        assert_eq!(ok(&r), "X");
    }

    #[test]
    fn test_whitespace_normalized() {
        let content = "foo    bar baz";
        let r = fuzzy_find_and_replace(content, "foo bar baz", "X", false);
        assert_eq!(r.strategy.as_deref(), Some("whitespace_normalized"));
        assert_eq!(r.match_count, 1);
    }

    #[test]
    fn test_indentation_flexible() {
        let content = "        result = compute()";
        let r = fuzzy_find_and_replace(content, "result = compute()", "X", false);
        // Could match via line_trimmed first; both strip. Verify it matched.
        assert!(r.match_count >= 1);
        assert!(r.error.is_none());
    }

    #[test]
    fn test_escape_normalized() {
        let content = "line1\nline2";
        let r = fuzzy_find_and_replace(content, "line1\\nline2", "X", false);
        assert_eq!(r.strategy.as_deref(), Some("escape_normalized"));
        assert_eq!(r.match_count, 1);
        assert_eq!(ok(&r), "X");
    }

    #[test]
    fn test_unicode_normalized() {
        // Content has smart quotes; pattern uses ASCII quotes.
        let content = "say \u{201c}hi\u{201d} now";
        let r = fuzzy_find_and_replace(content, "say \"hi\" now", "X", false);
        assert_eq!(r.strategy.as_deref(), Some("unicode_normalized"));
        assert_eq!(r.match_count, 1);
        assert_eq!(ok(&r), "X");
    }

    #[test]
    fn test_unicode_expansion_offsets() {
        // em-dash expands to '--'; ensure offsets map correctly back.
        let content = "a\u{2014}b extra";
        let r = fuzzy_find_and_replace(content, "a--b", "Z", false);
        assert_eq!(r.match_count, 1);
        assert_eq!(ok(&r), "Z extra");
    }

    #[test]
    fn test_block_anchor() {
        let content = "def f():\n    x = 1\n    y = 2\n    return x";
        // First and last lines match exactly; middle differs slightly.
        let pattern = "def f():\n    x = 1\n    y = 99\n    return x";
        let r = fuzzy_find_and_replace(content, pattern, "REPLACED", false);
        // Should fall to block_anchor (middle similarity high enough, single candidate).
        assert!(r.error.is_none(), "err: {:?}", r.error);
        assert_eq!(r.match_count, 1);
    }

    #[test]
    fn test_escape_drift_detected() {
        // Non-exact strategy match where new_string carries spurious \'.
        let content = "    value = compute()";
        let old = "value = compute()\\'";
        let new = "value = other()\\'";
        let r = fuzzy_find_and_replace(content, old, new, false);
        // old won't match anything (it has the escape and trailing junk), so
        // this likely yields a no-match; the drift guard is unit-tested below.
        let _ = r;
    }

    #[test]
    fn test_detect_escape_drift_unit() {
        let content: Vec<char> = "name = 'bob'".chars().collect();
        let matches = vec![(0usize, content.len())];
        // new/old both contain \" but content doesn't.
        let err = detect_escape_drift(&content, &matches, "x\\\"y", "z\\\"w");
        assert!(err.is_some());
        assert!(err.unwrap().contains("Escape-drift detected"));

        // If matched region already contains the escape, no drift.
        let content2: Vec<char> = "a\\\"b".chars().collect();
        let matches2 = vec![(0usize, content2.len())];
        let err2 = detect_escape_drift(&content2, &matches2, "x\\\"y", "z\\\"w");
        assert!(err2.is_none());

        // No suspect escape in new_string -> None.
        let err3 = detect_escape_drift(&content, &matches, "x'y", "z'w");
        assert!(err3.is_none());
    }

    #[test]
    fn test_seq_ratio_basic() {
        // Matches Python difflib examples.
        assert!((seq_ratio("abcd", "abcd") - 1.0).abs() < 1e-9);
        assert!((seq_ratio("", "") - 1.0).abs() < 1e-9);
        // "abcd" vs "abce": 3 matches, T=8 -> 0.75
        assert!((seq_ratio("abcd", "abce") - 0.75).abs() < 1e-9);
        // difflib: ratio('abcde','abXde') -> matches 'ab' + 'de' = 4, T=10 -> 0.8
        assert!((seq_ratio("abcde", "abXde") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_seq_ratio_known() {
        // Python: SequenceMatcher(None, "GESTALT PATTERN MATCHING".lower()...).
        // Use a documented example: ratio of "pineapple" / "applesauce".
        let r = seq_ratio("pineapple", "applesauce");
        // Python difflib returns 0.5263157894736842 for this pair.
        assert!((r - 0.5263157894736842).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_calculate_line_positions() {
        let lines = vec!["abc", "de", "fghi"];
        // content "abc\nde\nfghi" length = 11
        let (s, e) = calculate_line_positions(&lines, 0, 1, 11);
        assert_eq!((s, e), (0, 3));
        let (s2, e2) = calculate_line_positions(&lines, 1, 2, 11);
        assert_eq!((s2, e2), (4, 6));
        let (s3, e3) = calculate_line_positions(&lines, 0, 3, 11);
        assert_eq!((s3, e3), (0, 11)); // clamped to content_length
    }

    #[test]
    fn test_find_closest_lines() {
        let content = "fn alpha() {}\nfn beta() {}\nfn gamma() {}";
        let hint = find_closest_lines("fn alpa() {}", content, 1, 3);
        assert!(hint.contains("alpha"), "hint was: {}", hint);
    }

    #[test]
    fn test_format_no_match_hint_gating() {
        let content = "fn alpha() {}";
        // Wrong error prefix -> empty.
        let h = format_no_match_hint(Some("Found 2 matches"), 0, "fn alpha() {}", content);
        assert_eq!(h, "");
        // match_count != 0 -> empty.
        let h2 = format_no_match_hint(Some("Could not find"), 1, "x", content);
        assert_eq!(h2, "");
        // Correct gating -> non-empty hint.
        let h3 = format_no_match_hint(
            Some("Could not find a match for old_string in the file"),
            0,
            "fn alpa() {}",
            content,
        );
        assert!(h3.starts_with("\n\nDid you mean"));
    }

    #[test]
    fn test_py_splitlines() {
        assert_eq!(py_splitlines("a\nb\nc"), vec!["a", "b", "c"]);
        assert_eq!(py_splitlines("a\nb\n"), vec!["a", "b"]);
        assert_eq!(py_splitlines("a\r\nb"), vec!["a", "b"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
    }

    #[test]
    fn test_context_aware() {
        // Lines similar but not exact; >=50% lines >=0.80 similarity.
        let content = "alpha beta gamma\ndelta epsilon zeta";
        let pattern = "alpha beta gamme\ndelta epsilon zeta";
        let r = fuzzy_find_and_replace(content, pattern, "DONE", false);
        assert!(r.error.is_none(), "err: {:?}", r.error);
        assert_eq!(r.match_count, 1);
    }
}
