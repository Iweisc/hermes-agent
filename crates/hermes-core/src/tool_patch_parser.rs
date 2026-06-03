//! V4A Patch Format Parser (native Rust port of `tools/patch_parser.py`).
//!
//! Parses the V4A patch format used by codex, cline, and other coding agents,
//! and applies the resulting operations through a pluggable file-operations
//! interface using a two-phase validate-then-apply strategy.
//!
//! V4A Format:
//! ```text
//! *** Begin Patch
//! *** Update File: path/to/file.py
//! @@ optional context hint @@
//!  context line (space prefix)
//! -removed line (minus prefix)
//! +added line (plus prefix)
//! *** Add File: path/to/new.py
//! +new file content
//! +line 2
//! *** Delete File: path/to/old.py
//! *** Move File: old/path.py -> new/path.py
//! *** End Patch
//! ```
//!
//! The Python original deferred to `tools.fuzzy_match.fuzzy_find_and_replace`
//! (and an optional `format_no_match_hint`). That module has not been ported to
//! Rust yet, so the fuzzy-matching dependency is expressed here as the
//! [`FuzzyMatcher`] trait. The integration layer supplies a concrete matcher;
//! tests use a simple exact-match implementation.

use std::sync::OnceLock;

use regex::Regex;

/// The kind of operation a single V4A directive represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    Add,
    Update,
    Delete,
    Move,
}

impl OperationType {
    /// String tag matching the Python `Enum` values.
    pub fn as_str(self) -> &'static str {
        match self {
            OperationType::Add => "add",
            OperationType::Update => "update",
            OperationType::Delete => "delete",
            OperationType::Move => "move",
        }
    }
}

/// A single line in a patch hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HunkLine {
    /// One of `' '`, `'-'`, or `'+'`.
    pub prefix: char,
    pub content: String,
}

impl HunkLine {
    pub fn new(prefix: char, content: impl Into<String>) -> Self {
        HunkLine {
            prefix,
            content: content.into(),
        }
    }
}

/// A group of changes within a file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hunk {
    pub context_hint: Option<String>,
    pub lines: Vec<HunkLine>,
}

impl Hunk {
    pub fn new() -> Self {
        Hunk::default()
    }

    pub fn with_hint(hint: Option<String>) -> Self {
        Hunk {
            context_hint: hint,
            lines: Vec::new(),
        }
    }
}

/// A single operation in a V4A patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchOperation {
    pub operation: OperationType,
    pub file_path: String,
    /// For move operations.
    pub new_path: Option<String>,
    pub hunks: Vec<Hunk>,
    /// For add-file operations (unused by the parser, kept for parity).
    pub content: Option<String>,
}

impl PatchOperation {
    pub fn new(operation: OperationType, file_path: impl Into<String>) -> Self {
        PatchOperation {
            operation,
            file_path: file_path.into(),
            new_path: None,
            hunks: Vec::new(),
            content: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pluggable dependencies (file operations + fuzzy matching)
// ---------------------------------------------------------------------------

/// Result of a single file-operation call, mirroring the Python `FileOpResult`
/// duck-typed shape (`.content`, `.error`).
#[derive(Debug, Clone, Default)]
pub struct FileOpResult {
    pub content: String,
    pub error: Option<String>,
}

impl FileOpResult {
    pub fn ok(content: impl Into<String>) -> Self {
        FileOpResult {
            content: content.into(),
            error: None,
        }
    }

    pub fn err(error: impl Into<String>) -> Self {
        FileOpResult {
            content: String::new(),
            error: Some(error.into()),
        }
    }

    pub fn is_err(&self) -> bool {
        self.error.is_some()
    }
}

/// The file-operations interface the apply phase depends on. Mirrors the Python
/// `file_ops` object that exposes `read_file_raw`, `write_file`, `delete_file`,
/// `move_file`, and an optional `_check_lint`.
pub trait FileOps {
    /// Read raw file content (no line-number prefixes / truncation).
    fn read_file_raw(&self, path: &str) -> FileOpResult;
    /// Write `content` to `path`, creating parent directories as needed.
    fn write_file(&mut self, path: &str, content: &str) -> FileOpResult;
    /// Delete the file at `path`.
    fn delete_file(&mut self, path: &str) -> FileOpResult;
    /// Move/rename `src` to `dst`.
    fn move_file(&mut self, src: &str, dst: &str) -> FileOpResult;
    /// Optional lint check; returns the lint result serialized to a string
    /// (the Python original calls `.to_dict()`). `None` means lint is not
    /// available for this `file_ops` implementation.
    fn check_lint(&self, _path: &str) -> Option<String> {
        None
    }
}

/// Outcome of a fuzzy find-and-replace, mirroring the 4-tuple returned by
/// `tools.fuzzy_match.fuzzy_find_and_replace`:
/// `(new_content, count, strategy, error)`.
#[derive(Debug, Clone)]
pub struct FuzzyResult {
    pub new_content: String,
    pub count: usize,
    pub strategy: Option<String>,
    pub error: Option<String>,
}

/// The fuzzy-matching dependency. `fuzzy_match.py` has not been ported yet, so
/// callers inject an implementation. The `replace_all` flag matches the Python
/// signature.
pub trait FuzzyMatcher {
    fn fuzzy_find_and_replace(
        &self,
        content: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> FuzzyResult;

    /// Optional human-readable hint appended to "no match" errors, mirroring
    /// `tools.fuzzy_match.format_no_match_hint`. Default appends nothing.
    fn format_no_match_hint(
        &self,
        _error: Option<&str>,
        _count: usize,
        _search_pattern: &str,
        _content: &str,
    ) -> String {
        String::new()
    }
}

/// Final result of applying a set of operations. Mirrors the Python
/// `tools.file_operations.PatchResult` shape used by callers.
#[derive(Debug, Clone, Default)]
pub struct PatchResult {
    pub success: bool,
    pub diff: String,
    pub files_modified: Vec<String>,
    pub files_created: Vec<String>,
    pub files_deleted: Vec<String>,
    /// Map of file path -> serialized lint result, when any lint ran.
    pub lint: Option<Vec<(String, String)>>,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Regex helpers (compiled once)
// ---------------------------------------------------------------------------

fn re_update() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\*\*\*\s*Update\s+File:\s*(.+)").unwrap())
}

fn re_add() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\*\*\*\s*Add\s+File:\s*(.+)").unwrap())
}

fn re_delete() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\*\*\*\s*Delete\s+File:\s*(.+)").unwrap())
}

fn re_move() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)").unwrap())
}

fn re_hint() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^@@\s*(.+?)\s*@@").unwrap())
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a V4A format patch.
///
/// Returns `(operations, error_message)`:
/// - On success: `(operations, None)`.
/// - On failure: `(vec![], Some(error))`.
///
/// An empty (but well-formed) patch is not an error; callers get an empty list.
pub fn parse_v4a_patch(patch_content: &str) -> (Vec<PatchOperation>, Option<String>) {
    let lines: Vec<&str> = patch_content.split('\n').collect();
    let mut operations: Vec<PatchOperation> = Vec::new();

    // Find patch boundaries.
    let mut start_idx: Option<usize> = None;
    let mut end_idx: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        if line.contains("*** Begin Patch") || line.contains("***Begin Patch") {
            start_idx = Some(i);
        } else if line.contains("*** End Patch") || line.contains("***End Patch") {
            end_idx = Some(i);
            break;
        }
    }

    // `start_idx is None` -> -1 in Python so the scan begins at index 0.
    // We model the "starting line index" as a signed concept by tracking
    // whether a marker was found.
    let scan_start: usize = match start_idx {
        Some(s) => s + 1,
        None => 0,
    };
    let end_idx = end_idx.unwrap_or(lines.len());

    let mut current_op: Option<PatchOperation> = None;
    let mut current_hunk: Option<Hunk> = None;

    let mut i = scan_start;
    while i < end_idx {
        let line = lines[i];

        let update_match = re_update().captures(line);
        let add_match = re_add().captures(line);
        let delete_match = re_delete().captures(line);
        let move_match = re_move().captures(line);

        if let Some(caps) = update_match {
            flush_op(&mut operations, &mut current_op, &mut current_hunk);
            current_op = Some(PatchOperation::new(
                OperationType::Update,
                caps.get(1).map(|m| m.as_str().trim()).unwrap_or(""),
            ));
            current_hunk = None;
        } else if let Some(caps) = add_match {
            flush_op(&mut operations, &mut current_op, &mut current_hunk);
            current_op = Some(PatchOperation::new(
                OperationType::Add,
                caps.get(1).map(|m| m.as_str().trim()).unwrap_or(""),
            ));
            current_hunk = Some(Hunk::new());
        } else if let Some(caps) = delete_match {
            flush_op(&mut operations, &mut current_op, &mut current_hunk);
            let op = PatchOperation::new(
                OperationType::Delete,
                caps.get(1).map(|m| m.as_str().trim()).unwrap_or(""),
            );
            operations.push(op);
            current_op = None;
            current_hunk = None;
        } else if let Some(caps) = move_match {
            flush_op(&mut operations, &mut current_op, &mut current_hunk);
            let mut op = PatchOperation::new(
                OperationType::Move,
                caps.get(1).map(|m| m.as_str().trim()).unwrap_or(""),
            );
            op.new_path = Some(caps.get(2).map(|m| m.as_str().trim()).unwrap_or("").to_string());
            operations.push(op);
            current_op = None;
            current_hunk = None;
        } else if line.starts_with("@@") {
            // Context hint / hunk marker.
            if current_op.is_some() {
                // Flush the in-progress hunk into the current op.
                if let (Some(op), Some(hunk)) = (current_op.as_mut(), current_hunk.take()) {
                    if !hunk.lines.is_empty() {
                        op.hunks.push(hunk);
                    }
                }
                let hint = re_hint()
                    .captures(line)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string());
                current_hunk = Some(Hunk::with_hint(hint));
            }
        } else if current_op.is_some() && !line.is_empty() {
            // Parse a hunk line.
            if current_hunk.is_none() {
                current_hunk = Some(Hunk::new());
            }
            let hunk = current_hunk.as_mut().unwrap();
            if let Some(rest) = line.strip_prefix('+') {
                hunk.lines.push(HunkLine::new('+', rest));
            } else if let Some(rest) = line.strip_prefix('-') {
                hunk.lines.push(HunkLine::new('-', rest));
            } else if let Some(rest) = line.strip_prefix(' ') {
                hunk.lines.push(HunkLine::new(' ', rest));
            } else if line.starts_with('\\') {
                // "\ No newline at end of file" marker - skip.
            } else {
                // Treat as context line (implicit space prefix).
                hunk.lines.push(HunkLine::new(' ', line));
            }
        }

        i += 1;
    }

    // Don't forget the last operation.
    flush_op(&mut operations, &mut current_op, &mut current_hunk);

    // Validate the parsed result.
    if operations.is_empty() {
        // Empty patch is not an error — callers get [] and can decide.
        return (operations, None);
    }

    let mut parse_errors: Vec<String> = Vec::new();
    for op in &operations {
        if op.file_path.is_empty() {
            parse_errors.push("Operation with empty file path".to_string());
        }
        if op.operation == OperationType::Update && op.hunks.is_empty() {
            parse_errors.push(format!("UPDATE '{}': no hunks found", op.file_path));
        }
        if op.operation == OperationType::Move && op.new_path.is_none() {
            parse_errors.push(format!(
                "MOVE '{}': missing destination path (expected 'src -> dst')",
                op.file_path
            ));
        }
    }

    if !parse_errors.is_empty() {
        return (Vec::new(), Some(format!("Parse error: {}", parse_errors.join("; "))));
    }

    (operations, None)
}

/// Flush the in-progress operation (and its in-progress hunk) into `operations`.
/// Mirrors the repeated save-previous-operation block in the Python source.
fn flush_op(
    operations: &mut Vec<PatchOperation>,
    current_op: &mut Option<PatchOperation>,
    current_hunk: &mut Option<Hunk>,
) {
    if let Some(mut op) = current_op.take() {
        if let Some(hunk) = current_hunk.take() {
            if !hunk.lines.is_empty() {
                op.hunks.push(hunk);
            }
        }
        operations.push(op);
    } else {
        // No current op: still clear any dangling hunk to match Python flow
        // where current_hunk is reset alongside current_op transitions.
        *current_hunk = None;
    }
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

/// Count non-overlapping occurrences of `pattern` in `text`, advancing by one
/// byte after each match — a faithful port of the Python `_count_occurrences`
/// (which steps `start = pos + 1`, allowing overlapping-style recount).
pub fn count_occurrences(text: &str, pattern: &str) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut start = 0usize;
    while start <= text.len() {
        match text[start..].find(pattern) {
            Some(rel) => {
                let pos = start + rel;
                count += 1;
                start = pos + 1;
            }
            None => break,
        }
    }
    count
}

/// Validate all operations without writing any files. Returns a list of error
/// strings; an empty list means the apply phase can proceed safely.
///
/// For UPDATE operations, hunks are simulated in order so that later hunks
/// validate against post-earlier-hunk content (matching apply order).
pub fn validate_operations<F: FileOps, M: FuzzyMatcher>(
    operations: &[PatchOperation],
    file_ops: &F,
    fuzzy: &M,
) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    for op in operations {
        match op.operation {
            OperationType::Update => {
                let read_result = file_ops.read_file_raw(&op.file_path);
                if let Some(err) = &read_result.error {
                    errors.push(format!("{}: {}", op.file_path, err));
                    continue;
                }

                let mut simulated = read_result.content;
                for hunk in &op.hunks {
                    let search_lines: Vec<&str> = hunk
                        .lines
                        .iter()
                        .filter(|l| l.prefix == ' ' || l.prefix == '-')
                        .map(|l| l.content.as_str())
                        .collect();

                    if search_lines.is_empty() {
                        // Addition-only hunk: validate context hint uniqueness.
                        if let Some(hint) = &hunk.context_hint {
                            let occurrences = count_occurrences(&simulated, hint);
                            if occurrences == 0 {
                                errors.push(format!(
                                    "{}: addition-only hunk context hint '{}' not found",
                                    op.file_path, hint
                                ));
                            } else if occurrences > 1 {
                                errors.push(format!(
                                    "{}: addition-only hunk context hint '{}' is ambiguous ({} occurrences)",
                                    op.file_path, hint, occurrences
                                ));
                            }
                        }
                        continue;
                    }

                    let search_pattern = search_lines.join("\n");
                    let replace_lines: Vec<&str> = hunk
                        .lines
                        .iter()
                        .filter(|l| l.prefix == ' ' || l.prefix == '+')
                        .map(|l| l.content.as_str())
                        .collect();
                    let replacement = replace_lines.join("\n");

                    let res = fuzzy.fuzzy_find_and_replace(
                        &simulated,
                        &search_pattern,
                        &replacement,
                        false,
                    );
                    if res.count == 0 {
                        let label = match &hunk.context_hint {
                            Some(h) => format!("'{}'", h),
                            None => "(no hint)".to_string(),
                        };
                        let mut msg = format!("{}: hunk {} not found", op.file_path, label);
                        if let Some(me) = &res.error {
                            msg.push_str(&format!(" — {}", me));
                        }
                        msg.push_str(&fuzzy.format_no_match_hint(
                            res.error.as_deref(),
                            res.count,
                            &search_pattern,
                            &simulated,
                        ));
                        errors.push(msg);
                    } else {
                        // Advance simulation so subsequent hunks validate correctly.
                        simulated = res.new_content;
                    }
                }
            }
            OperationType::Delete => {
                let read_result = file_ops.read_file_raw(&op.file_path);
                if read_result.is_err() {
                    errors.push(format!("{}: file not found for deletion", op.file_path));
                }
            }
            OperationType::Move => {
                let new_path = match &op.new_path {
                    Some(p) => p,
                    None => {
                        errors.push(format!(
                            "{}: MOVE operation missing destination path",
                            op.file_path
                        ));
                        continue;
                    }
                };
                let src_result = file_ops.read_file_raw(&op.file_path);
                if src_result.is_err() {
                    errors.push(format!("{}: source file not found for move", op.file_path));
                }
                let dst_result = file_ops.read_file_raw(new_path);
                if !dst_result.is_err() {
                    errors.push(format!(
                        "{}: destination already exists — move would overwrite",
                        new_path
                    ));
                }
            }
            OperationType::Add => {
                // ADD: parent directory creation handled by write_file.
            }
        }
    }

    errors
}

/// Apply V4A patch operations using a file-operations interface.
///
/// Two-phase validate-then-apply: if validation finds any error, no files are
/// touched. Apply-phase failures are reported with a note to run `git diff`.
pub fn apply_v4a_operations<F: FileOps, M: FuzzyMatcher>(
    operations: &[PatchOperation],
    file_ops: &mut F,
    fuzzy: &M,
) -> PatchResult {
    // ---- Phase 1: validate ----
    let validation_errors = validate_operations(operations, file_ops, fuzzy);
    if !validation_errors.is_empty() {
        let body: Vec<String> = validation_errors.iter().map(|e| format!("  • {}", e)).collect();
        return PatchResult {
            success: false,
            error: Some(format!(
                "Patch validation failed (no files were modified):\n{}",
                body.join("\n")
            )),
            ..Default::default()
        };
    }

    // ---- Phase 2: apply ----
    let mut files_modified: Vec<String> = Vec::new();
    let mut files_created: Vec<String> = Vec::new();
    let mut files_deleted: Vec<String> = Vec::new();
    let mut all_diffs: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for op in operations {
        match op.operation {
            OperationType::Add => match apply_add(op, file_ops) {
                Ok(diff) => {
                    files_created.push(op.file_path.clone());
                    all_diffs.push(diff);
                }
                Err(e) => errors.push(format!("Failed to add {}: {}", op.file_path, e)),
            },
            OperationType::Delete => match apply_delete(op, file_ops) {
                Ok(diff) => {
                    files_deleted.push(op.file_path.clone());
                    all_diffs.push(diff);
                }
                Err(e) => errors.push(format!("Failed to delete {}: {}", op.file_path, e)),
            },
            OperationType::Move => match apply_move(op, file_ops) {
                Ok(diff) => {
                    files_modified.push(format!(
                        "{} -> {}",
                        op.file_path,
                        op.new_path.as_deref().unwrap_or("")
                    ));
                    all_diffs.push(diff);
                }
                Err(e) => errors.push(format!("Failed to move {}: {}", op.file_path, e)),
            },
            OperationType::Update => match apply_update(op, file_ops, fuzzy) {
                Ok(diff) => {
                    files_modified.push(op.file_path.clone());
                    all_diffs.push(diff);
                }
                Err(e) => errors.push(format!("Failed to update {}: {}", op.file_path, e)),
            },
        }
    }

    // Run lint on all modified/created files.
    let mut lint_results: Vec<(String, String)> = Vec::new();
    for f in files_modified.iter().chain(files_created.iter()) {
        if let Some(lint) = file_ops.check_lint(f) {
            lint_results.push((f.clone(), lint));
        }
    }
    let lint = if lint_results.is_empty() {
        None
    } else {
        Some(lint_results)
    };

    let combined_diff = all_diffs.join("\n");

    if !errors.is_empty() {
        let body: Vec<String> = errors.iter().map(|e| format!("  • {}", e)).collect();
        return PatchResult {
            success: false,
            diff: combined_diff,
            files_modified,
            files_created,
            files_deleted,
            lint,
            error: Some(format!(
                "Apply phase failed (state may be inconsistent — run `git diff` to assess):\n{}",
                body.join("\n")
            )),
        };
    }

    PatchResult {
        success: true,
        diff: combined_diff,
        files_modified,
        files_created,
        files_deleted,
        lint,
        error: None,
    }
}

/// Apply an add-file operation. Returns `Ok(diff)` or `Err(message)`.
fn apply_add<F: FileOps>(op: &PatchOperation, file_ops: &mut F) -> Result<String, String> {
    // Extract content from hunks (all + lines).
    let content_lines: Vec<&str> = op
        .hunks
        .iter()
        .flat_map(|h| h.lines.iter())
        .filter(|l| l.prefix == '+')
        .map(|l| l.content.as_str())
        .collect();

    let content = content_lines.join("\n");

    let result = file_ops.write_file(&op.file_path, &content);
    if let Some(err) = result.error {
        return Err(err);
    }

    let mut diff = format!("--- /dev/null\n+++ b/{}\n", op.file_path);
    let added: Vec<String> = content_lines.iter().map(|l| format!("+{}", l)).collect();
    diff.push_str(&added.join("\n"));

    Ok(diff)
}

/// Apply a delete-file operation. Returns `Ok(diff)` or `Err(message)`.
fn apply_delete<F: FileOps>(op: &PatchOperation, file_ops: &mut F) -> Result<String, String> {
    // Read before deleting so we can produce a real unified diff.
    let read_result = file_ops.read_file_raw(&op.file_path);
    if read_result.is_err() {
        return Err(format!("Cannot delete {}: file not found", op.file_path));
    }

    let result = file_ops.delete_file(&op.file_path);
    if let Some(err) = result.error {
        return Err(err);
    }

    let removed_lines = splitlines_keepends(&read_result.content);
    let diff = unified_diff(
        &removed_lines,
        &[],
        &format!("a/{}", op.file_path),
        "/dev/null",
    );
    if diff.is_empty() {
        Ok(format!("# Deleted: {}", op.file_path))
    } else {
        Ok(diff)
    }
}

/// Apply a move-file operation. Returns `Ok(diff)` or `Err(message)`.
fn apply_move<F: FileOps>(op: &PatchOperation, file_ops: &mut F) -> Result<String, String> {
    let new_path = op.new_path.clone().unwrap_or_default();
    let result = file_ops.move_file(&op.file_path, &new_path);
    if let Some(err) = result.error {
        return Err(err);
    }
    Ok(format!("# Moved: {} -> {}", op.file_path, new_path))
}

/// Apply an update-file operation. Returns `Ok(diff)` or `Err(message)`.
fn apply_update<F: FileOps, M: FuzzyMatcher>(
    op: &PatchOperation,
    file_ops: &mut F,
    fuzzy: &M,
) -> Result<String, String> {
    let read_result = file_ops.read_file_raw(&op.file_path);
    if let Some(err) = read_result.error {
        return Err(format!("Cannot read file: {}", err));
    }

    let current_content = read_result.content;
    let mut new_content = current_content.clone();

    for hunk in &op.hunks {
        let mut search_lines: Vec<&str> = Vec::new();
        let mut replace_lines: Vec<&str> = Vec::new();

        for line in &hunk.lines {
            match line.prefix {
                ' ' => {
                    search_lines.push(&line.content);
                    replace_lines.push(&line.content);
                }
                '-' => search_lines.push(&line.content),
                '+' => replace_lines.push(&line.content),
                _ => {}
            }
        }

        if !search_lines.is_empty() {
            let search_pattern = search_lines.join("\n");
            let replacement = replace_lines.join("\n");

            let mut res =
                fuzzy.fuzzy_find_and_replace(&new_content, &search_pattern, &replacement, false);
            new_content = res.new_content.clone();
            let mut error = res.error.clone();
            let mut count = res.count;

            if error.is_some() && count == 0 {
                // Try with context hint if available.
                if let Some(hint) = &hunk.context_hint {
                    if let Some(hint_pos) = new_content.find(hint.as_str()) {
                        // Search in a window around the hint.
                        let window_start = hint_pos.saturating_sub(500);
                        let window_end = std::cmp::min(new_content.len(), hint_pos + 2000);
                        // Clamp to char boundaries to keep slicing valid.
                        let window_start = floor_char_boundary(&new_content, window_start);
                        let window_end = ceil_char_boundary(&new_content, window_end);
                        let window = new_content[window_start..window_end].to_string();

                        res = fuzzy.fuzzy_find_and_replace(
                            &window,
                            &search_pattern,
                            &replacement,
                            false,
                        );
                        count = res.count;
                        error = res.error.clone();

                        if count > 0 {
                            new_content = format!(
                                "{}{}{}",
                                &new_content[..window_start],
                                res.new_content,
                                &new_content[window_end..]
                            );
                            error = None;
                        }
                    }
                }

                if let Some(err) = error {
                    let mut err_msg = format!("Could not apply hunk: {}", err);
                    err_msg.push_str(&fuzzy.format_no_match_hint(
                        Some(&err),
                        0,
                        &search_pattern,
                        &new_content,
                    ));
                    return Err(err_msg);
                }
            }
        } else {
            // Addition-only hunk (no context or removed lines).
            // Insert at the location indicated by the context hint, or at end.
            let insert_text = replace_lines.join("\n");
            if let Some(hint) = &hunk.context_hint {
                let occurrences = count_occurrences(&new_content, hint);
                if occurrences == 0 {
                    // Hint not found — append at end as a safe fallback.
                    new_content = format!(
                        "{}\n{}\n",
                        new_content.trim_end_matches('\n'),
                        insert_text
                    );
                } else if occurrences > 1 {
                    return Err(format!(
                        "Addition-only hunk: context hint '{}' is ambiguous ({} occurrences) — provide a more unique hint",
                        hint, occurrences
                    ));
                } else {
                    let hint_pos = new_content.find(hint.as_str()).unwrap();
                    // Insert after the line containing the context hint.
                    match new_content[hint_pos..].find('\n') {
                        Some(rel) => {
                            let eol = hint_pos + rel;
                            new_content = format!(
                                "{}{}\n{}",
                                &new_content[..eol + 1],
                                insert_text,
                                &new_content[eol + 1..]
                            );
                        }
                        None => {
                            new_content = format!("{}\n{}", new_content, insert_text);
                        }
                    }
                }
            } else {
                new_content = format!(
                    "{}\n{}\n",
                    new_content.trim_end_matches('\n'),
                    insert_text
                );
            }
        }
    }

    let write_result = file_ops.write_file(&op.file_path, &new_content);
    if let Some(err) = write_result.error {
        return Err(err);
    }

    let diff = unified_diff(
        &splitlines_keepends(&current_content),
        &splitlines_keepends(&new_content),
        &format!("a/{}", op.file_path),
        &format!("b/{}", op.file_path),
    );

    Ok(diff)
}

// ---------------------------------------------------------------------------
// difflib-compatible helpers
// ---------------------------------------------------------------------------

/// Split `text` into lines keeping their line terminators, mirroring Python's
/// `str.splitlines(keepends=True)` (for the `\n` family used by patches).
pub fn splitlines_keepends(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if ch == '\n' {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Produce a unified diff equivalent to `difflib.unified_diff(a, b, ...)`
/// joined with the empty string (the lines already carry their newlines).
/// Default context of 3 lines, matching Python's `n=3`.
pub fn unified_diff(a: &[String], b: &[String], fromfile: &str, tofile: &str) -> String {
    let groups = grouped_opcodes(a, b, 3);
    if groups.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str(&format!("--- {}\n", fromfile));
    out.push_str(&format!("+++ {}\n", tofile));

    for group in &groups {
        let (first, last) = (group.first().unwrap(), group.last().unwrap());
        let i1 = first.i1;
        let i2 = last.i2;
        let j1 = first.j1;
        let j2 = last.j2;
        out.push_str(&format!(
            "@@ -{} +{} @@\n",
            format_range(i1, i2),
            format_range(j1, j2)
        ));
        for op in group {
            match op.tag {
                Tag::Equal => {
                    for line in &a[op.i1..op.i2] {
                        out.push_str(" ");
                        out.push_str(line);
                    }
                }
                Tag::Replace | Tag::Delete => {
                    for line in &a[op.i1..op.i2] {
                        out.push_str("-");
                        out.push_str(line);
                    }
                    if op.tag == Tag::Replace {
                        for line in &b[op.j1..op.j2] {
                            out.push_str("+");
                            out.push_str(line);
                        }
                    }
                }
                Tag::Insert => {
                    for line in &b[op.j1..op.j2] {
                        out.push_str("+");
                        out.push_str(line);
                    }
                }
            }
        }
    }

    out
}

fn format_range(start: usize, stop: usize) -> String {
    // Mirrors difflib._format_range_unified.
    let length = stop.saturating_sub(start);
    let beginning = if length == 0 { start } else { start + 1 };
    if length == 1 {
        format!("{}", beginning)
    } else {
        format!("{},{}", beginning, length)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tag {
    Equal,
    Replace,
    Delete,
    Insert,
}

#[derive(Debug, Clone)]
struct Opcode {
    tag: Tag,
    i1: usize,
    i2: usize,
    j1: usize,
    j2: usize,
}

/// Compute opcodes via a longest-common-subsequence diff over whole lines.
/// This is sufficient for the unified-diff output the Python code emits.
fn diff_opcodes(a: &[String], b: &[String]) -> Vec<Opcode> {
    let n = a.len();
    let m = b.len();
    // LCS DP table.
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    // Backtrack into a sequence of matched/unmatched runs, then coalesce.
    let mut raw: Vec<(Tag, usize, usize, usize, usize)> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if a[i] == b[j] {
            raw.push((Tag::Equal, i, i + 1, j, j + 1));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            raw.push((Tag::Delete, i, i + 1, j, j));
            i += 1;
        } else {
            raw.push((Tag::Insert, i, i, j, j + 1));
            j += 1;
        }
    }
    while i < n {
        raw.push((Tag::Delete, i, i + 1, j, j));
        i += 1;
    }
    while j < m {
        raw.push((Tag::Insert, i, i, j, j + 1));
        j += 1;
    }

    // Coalesce consecutive runs and merge delete+insert into replace.
    let mut ops: Vec<Opcode> = Vec::new();
    for (tag, i1, i2, j1, j2) in raw {
        if let Some(last) = ops.last_mut() {
            if last.tag == tag {
                last.i2 = i2.max(last.i2);
                last.j2 = j2.max(last.j2);
                last.i1 = last.i1.min(i1);
                last.j1 = last.j1.min(j1);
                continue;
            }
            // Merge adjacent delete + insert (in either order) into replace.
            if (last.tag == Tag::Delete && tag == Tag::Insert)
                || (last.tag == Tag::Insert && tag == Tag::Delete)
            {
                last.tag = Tag::Replace;
                last.i1 = last.i1.min(i1);
                last.i2 = last.i2.max(i2);
                last.j1 = last.j1.min(j1);
                last.j2 = last.j2.max(j2);
                continue;
            }
            if last.tag == Tag::Replace && (tag == Tag::Delete || tag == Tag::Insert) {
                last.i2 = last.i2.max(i2);
                last.j2 = last.j2.max(j2);
                continue;
            }
        }
        ops.push(Opcode {
            tag,
            i1,
            i2,
            j1,
            j2,
        });
    }

    ops
}

/// Group opcodes with `n` lines of context, mirroring
/// `SequenceMatcher.get_grouped_opcodes`.
fn grouped_opcodes(a: &[String], b: &[String], n: usize) -> Vec<Vec<Opcode>> {
    let mut codes = diff_opcodes(a, b);
    if codes.is_empty() {
        codes.push(Opcode {
            tag: Tag::Equal,
            i1: 0,
            i2: 1.min(a.len()),
            j1: 0,
            j2: 1.min(b.len()),
        });
    }

    // Trim leading/trailing equal blocks.
    if let Some(first) = codes.first().cloned() {
        if first.tag == Tag::Equal {
            let i1 = first.i2.saturating_sub(n).max(first.i1);
            let j1 = first.j2.saturating_sub(n).max(first.j1);
            codes[0] = Opcode {
                tag: Tag::Equal,
                i1,
                i2: first.i2,
                j1,
                j2: first.j2,
            };
        }
    }
    if let Some(last) = codes.last().cloned() {
        if last.tag == Tag::Equal {
            let idx = codes.len() - 1;
            codes[idx] = Opcode {
                tag: Tag::Equal,
                i1: last.i1,
                i2: (last.i1 + n).min(last.i2),
                j1: last.j1,
                j2: (last.j1 + n).min(last.j2),
            };
        }
    }

    let mut groups: Vec<Vec<Opcode>> = Vec::new();
    let mut group: Vec<Opcode> = Vec::new();
    let max_gap = n * 2;
    for code in codes {
        let Opcode {
            tag,
            mut i1,
            i2,
            mut j1,
            j2,
        } = code;
        // End the current group and start a new one whenever there is a large
        // range with no changes.
        if tag == Tag::Equal && i2.saturating_sub(i1) > max_gap {
            group.push(Opcode {
                tag,
                i1,
                i2: (i1 + n).min(i2),
                j1,
                j2: (j1 + n).min(j2),
            });
            groups.push(std::mem::take(&mut group));
            i1 = i2.saturating_sub(n);
            j1 = j2.saturating_sub(n);
            group.push(Opcode {
                tag,
                i1,
                i2,
                j1,
                j2,
            });
            continue;
        }
        group.push(Opcode {
            tag,
            i1,
            i2,
            j1,
            j2,
        });
    }

    // Drop a trailing group that contains only context.
    if !group.is_empty()
        && !(group.len() == 1 && group[0].tag == Tag::Equal)
    {
        groups.push(group);
    } else if !group.is_empty() && groups.is_empty() {
        // Keep nothing: no real changes.
    }

    groups
}

// char-boundary helpers (replacements for unstable std methods).
fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(s: &str, mut index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    while index < s.len() && !s.is_char_boundary(index) {
        index += 1;
    }
    index
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory file ops for testing.
    struct MemFs {
        files: HashMap<String, String>,
    }

    impl MemFs {
        fn new() -> Self {
            MemFs {
                files: HashMap::new(),
            }
        }
    }

    impl FileOps for MemFs {
        fn read_file_raw(&self, path: &str) -> FileOpResult {
            match self.files.get(path) {
                Some(c) => FileOpResult::ok(c.clone()),
                None => FileOpResult::err(format!("File not found: {}", path)),
            }
        }
        fn write_file(&mut self, path: &str, content: &str) -> FileOpResult {
            self.files.insert(path.to_string(), content.to_string());
            FileOpResult::default()
        }
        fn delete_file(&mut self, path: &str) -> FileOpResult {
            self.files.remove(path);
            FileOpResult::default()
        }
        fn move_file(&mut self, src: &str, dst: &str) -> FileOpResult {
            if let Some(c) = self.files.remove(src) {
                self.files.insert(dst.to_string(), c);
                FileOpResult::default()
            } else {
                FileOpResult::err(format!("File not found: {}", src))
            }
        }
    }

    /// Exact-match fuzzy implementation used to exercise apply/validate logic.
    struct ExactMatcher;

    impl FuzzyMatcher for ExactMatcher {
        fn fuzzy_find_and_replace(
            &self,
            content: &str,
            old_string: &str,
            new_string: &str,
            replace_all: bool,
        ) -> FuzzyResult {
            if old_string.is_empty() {
                return FuzzyResult {
                    new_content: content.to_string(),
                    count: 0,
                    strategy: None,
                    error: Some("old_string cannot be empty".to_string()),
                };
            }
            let occ = content.matches(old_string).count();
            if occ == 0 {
                return FuzzyResult {
                    new_content: content.to_string(),
                    count: 0,
                    strategy: None,
                    error: Some("not found".to_string()),
                };
            }
            if occ > 1 && !replace_all {
                return FuzzyResult {
                    new_content: content.to_string(),
                    count: 0,
                    strategy: None,
                    error: Some("ambiguous".to_string()),
                };
            }
            let new_content = if replace_all {
                content.replace(old_string, new_string)
            } else {
                content.replacen(old_string, new_string, 1)
            };
            FuzzyResult {
                new_content,
                count: occ,
                strategy: Some("exact".to_string()),
                error: None,
            }
        }
    }

    #[test]
    fn parse_update_with_hunk() {
        let patch = "*** Begin Patch\n\
                     *** Update File: src/foo.py\n\
                     @@ def foo @@\n\
                      context\n\
                     -old line\n\
                     +new line\n\
                     *** End Patch";
        let (ops, err) = parse_v4a_patch(patch);
        assert!(err.is_none(), "err: {:?}", err);
        assert_eq!(ops.len(), 1);
        let op = &ops[0];
        assert_eq!(op.operation, OperationType::Update);
        assert_eq!(op.file_path, "src/foo.py");
        assert_eq!(op.hunks.len(), 1);
        assert_eq!(op.hunks[0].context_hint.as_deref(), Some("def foo"));
        assert_eq!(
            op.hunks[0].lines,
            vec![
                HunkLine::new(' ', "context"),
                HunkLine::new('-', "old line"),
                HunkLine::new('+', "new line"),
            ]
        );
    }

    #[test]
    fn parse_add_delete_move() {
        let patch = "*** Begin Patch\n\
                     *** Add File: new.txt\n\
                     +hello\n\
                     +world\n\
                     *** Delete File: gone.txt\n\
                     *** Move File: a.txt -> b.txt\n\
                     *** End Patch";
        let (ops, err) = parse_v4a_patch(patch);
        assert!(err.is_none(), "err: {:?}", err);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].operation, OperationType::Add);
        assert_eq!(ops[0].file_path, "new.txt");
        assert_eq!(ops[0].hunks.len(), 1);
        assert_eq!(ops[1].operation, OperationType::Delete);
        assert_eq!(ops[1].file_path, "gone.txt");
        assert_eq!(ops[2].operation, OperationType::Move);
        assert_eq!(ops[2].file_path, "a.txt");
        assert_eq!(ops[2].new_path.as_deref(), Some("b.txt"));
    }

    #[test]
    fn parse_without_begin_marker() {
        let patch = "*** Update File: x.py\n\
                     -a\n\
                     +b\n";
        let (ops, err) = parse_v4a_patch(patch);
        assert!(err.is_none(), "err: {:?}", err);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].operation, OperationType::Update);
    }

    #[test]
    fn parse_empty_is_not_error() {
        let (ops, err) = parse_v4a_patch("");
        assert!(ops.is_empty());
        assert!(err.is_none());
    }

    #[test]
    fn parse_update_without_hunks_is_error() {
        let patch = "*** Begin Patch\n*** Update File: x.py\n*** End Patch";
        let (ops, err) = parse_v4a_patch(patch);
        assert!(ops.is_empty());
        let e = err.unwrap();
        assert!(e.contains("no hunks found"), "{}", e);
    }

    #[test]
    fn count_occurrences_overlapping_step() {
        // "aa" in "aaa": Python finds 2 with start=pos+1.
        assert_eq!(count_occurrences("aaa", "aa"), 2);
        assert_eq!(count_occurrences("abcabc", "abc"), 2);
        assert_eq!(count_occurrences("abc", "xyz"), 0);
        assert_eq!(count_occurrences("abc", ""), 0);
    }

    #[test]
    fn apply_add_creates_file() {
        let (ops, err) = parse_v4a_patch(
            "*** Begin Patch\n*** Add File: new.txt\n+line1\n+line2\n*** End Patch",
        );
        assert!(err.is_none());
        let mut fs = MemFs::new();
        let res = apply_v4a_operations(&ops, &mut fs, &ExactMatcher);
        assert!(res.success, "error: {:?}", res.error);
        assert_eq!(res.files_created, vec!["new.txt".to_string()]);
        assert_eq!(fs.files.get("new.txt").unwrap(), "line1\nline2");
    }

    #[test]
    fn apply_update_replaces_content() {
        let mut fs = MemFs::new();
        fs.files
            .insert("f.txt".to_string(), "alpha\nold\nbeta\n".to_string());
        let (ops, err) = parse_v4a_patch(
            "*** Begin Patch\n*** Update File: f.txt\n@@ @@\n alpha\n-old\n+new\n beta\n*** End Patch",
        );
        assert!(err.is_none());
        let res = apply_v4a_operations(&ops, &mut fs, &ExactMatcher);
        assert!(res.success, "error: {:?}", res.error);
        assert_eq!(fs.files.get("f.txt").unwrap(), "alpha\nnew\nbeta\n");
        assert!(res.diff.contains("-old"));
        assert!(res.diff.contains("+new"));
    }

    #[test]
    fn validate_fails_on_missing_update_file() {
        let fs = MemFs::new();
        let (ops, _) = parse_v4a_patch(
            "*** Begin Patch\n*** Update File: missing.txt\n@@ @@\n-x\n+y\n*** End Patch",
        );
        let errors = validate_operations(&ops, &fs, &ExactMatcher);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("missing.txt"));
    }

    #[test]
    fn apply_delete_removes_file() {
        let mut fs = MemFs::new();
        fs.files.insert("d.txt".to_string(), "a\nb\n".to_string());
        let (ops, _) =
            parse_v4a_patch("*** Begin Patch\n*** Delete File: d.txt\n*** End Patch");
        let res = apply_v4a_operations(&ops, &mut fs, &ExactMatcher);
        assert!(res.success, "error: {:?}", res.error);
        assert_eq!(res.files_deleted, vec!["d.txt".to_string()]);
        assert!(!fs.files.contains_key("d.txt"));
    }

    #[test]
    fn move_overwrite_validation() {
        let mut fs = MemFs::new();
        fs.files.insert("a.txt".to_string(), "a".to_string());
        fs.files.insert("b.txt".to_string(), "b".to_string());
        let (ops, _) =
            parse_v4a_patch("*** Begin Patch\n*** Move File: a.txt -> b.txt\n*** End Patch");
        let errors = validate_operations(&ops, &fs, &ExactMatcher);
        assert!(errors.iter().any(|e| e.contains("would overwrite")));
    }

    #[test]
    fn addition_only_hunk_inserts_after_hint() {
        let mut fs = MemFs::new();
        fs.files
            .insert("f.txt".to_string(), "header\nbody\nfooter\n".to_string());
        // Addition-only hunk: only '+' lines, with a context hint.
        let mut op = PatchOperation::new(OperationType::Update, "f.txt");
        let mut hunk = Hunk::with_hint(Some("header".to_string()));
        hunk.lines.push(HunkLine::new('+', "inserted"));
        op.hunks.push(hunk);
        let res = apply_v4a_operations(&[op], &mut fs, &ExactMatcher);
        assert!(res.success, "error: {:?}", res.error);
        assert_eq!(
            fs.files.get("f.txt").unwrap(),
            "header\ninserted\nbody\nfooter\n"
        );
    }

    #[test]
    fn unified_diff_basic() {
        let a = splitlines_keepends("a\nb\nc\n");
        let b = splitlines_keepends("a\nx\nc\n");
        let d = unified_diff(&a, &b, "a/f", "b/f");
        assert!(d.starts_with("--- a/f\n+++ b/f\n"), "{}", d);
        assert!(d.contains("-b\n"), "{}", d);
        assert!(d.contains("+x\n"), "{}", d);
        assert!(d.contains(" a\n"), "{}", d);
    }

    #[test]
    fn splitlines_keepends_matches_python() {
        assert_eq!(
            splitlines_keepends("a\nb\n"),
            vec!["a\n".to_string(), "b\n".to_string()]
        );
        assert_eq!(
            splitlines_keepends("a\nb"),
            vec!["a\n".to_string(), "b".to_string()]
        );
        assert!(splitlines_keepends("").is_empty());
    }
}
