//! Skills Hub CLI — unified interface for the Hermes Skills Hub.
//!
//! Native Rust port of `hermes_cli/skills_hub.py`.
//!
//! Powers both:
//!   - `hermes skills <subcommand>` (CLI entry point)
//!   - `/skills <subcommand>` (slash command in the interactive chat)
//!
//! All logic lives in shared `do_*` functions. The CLI entry point and slash
//! command handler are thin wrappers that parse args and delegate.
//!
//! In the original Python, the heavy lifting (registries, source routing,
//! quarantine, lock file, taps) lives in `tools.skills_hub`. Here those concerns
//! are abstracted behind the [`SkillSource`] trait and the [`HubBackend`] trait,
//! so this module reproduces the *presentation + control flow* faithfully while
//! delegating IO-heavy concerns to a backend (which the native hub
//! implementation in `crate::skills_cmd` can satisfy, or tests can mock).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

// ---------------------------------------------------------------------------
// Console abstraction (mirrors rich.Console.print). Renders markup-stripped
// text to a sink so the control flow and emitted strings can be tested.
// ---------------------------------------------------------------------------

/// Output sink that mirrors `rich.console.Console.print`. The original code
/// emits Rich markup like `[bold]...[/]`; we keep the strings byte-identical so
/// behaviour is faithful, and provide a quiet variant for programmatic callers
/// (mirrors the `_Q` shim in `inspect_skill`).
pub trait Console {
    fn print(&self, msg: &str);
}

/// Standard console that writes lines to stdout.
#[derive(Default)]
pub struct StdoutConsole;

impl Console for StdoutConsole {
    fn print(&self, msg: &str) {
        println!("{msg}");
    }
}

/// Quiet console that discards output (mirrors the `_Q` class used by
/// `inspect_skill`).
#[derive(Default)]
pub struct QuietConsole;

impl Console for QuietConsole {
    fn print(&self, _msg: &str) {}
}

/// Console that records every printed message (used by tests).
#[derive(Default)]
pub struct CaptureConsole {
    pub lines: std::cell::RefCell<Vec<String>>,
}

impl Console for CaptureConsole {
    fn print(&self, msg: &str) {
        self.lines.borrow_mut().push(msg.to_string());
    }
}

impl CaptureConsole {
    pub fn joined(&self) -> String {
        self.lines.borrow().join("\n")
    }
    pub fn contains(&self, needle: &str) -> bool {
        self.lines.borrow().iter().any(|l| l.contains(needle))
    }
}

// ---------------------------------------------------------------------------
// Prompt abstraction (mirrors builtin input()). EOF/cancel => None.
// ---------------------------------------------------------------------------

/// Supplies interactive answers. Mirrors Python's `input()`; returning `None`
/// reproduces an `EOFError`/`KeyboardInterrupt`.
pub trait Prompter {
    fn ask(&self, label: &str) -> Option<String>;
}

/// Prompter that always cancels (returns `None`) — equivalent to a
/// non-interactive surface where `input()` would raise EOFError.
#[derive(Default)]
pub struct NoPrompter;

impl Prompter for NoPrompter {
    fn ask(&self, _label: &str) -> Option<String> {
        None
    }
}

/// Prompter that replays a fixed queue of answers (for tests / scripted flows).
pub struct ScriptedPrompter {
    answers: std::cell::RefCell<std::collections::VecDeque<Option<String>>>,
}

impl ScriptedPrompter {
    pub fn new<I: IntoIterator<Item = Option<String>>>(answers: I) -> Self {
        Self {
            answers: std::cell::RefCell::new(answers.into_iter().collect()),
        }
    }
}

impl Prompter for ScriptedPrompter {
    fn ask(&self, _label: &str) -> Option<String> {
        self.answers.borrow_mut().pop_front().flatten()
    }
}

/// Prompter that reads from stdin (real interactive use).
#[derive(Default)]
pub struct StdinPrompter;

impl Prompter for StdinPrompter {
    fn ask(&self, label: &str) -> Option<String> {
        use std::io::Write;
        print!("{label}");
        let _ = std::io::stdout().flush();
        let mut buf = String::new();
        match std::io::stdin().read_line(&mut buf) {
            Ok(0) => None, // EOF
            Ok(_) => Some(buf),
            Err(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types (mirror tools.skills_hub.SkillResult / SkillMeta / SkillBundle).
// ---------------------------------------------------------------------------

/// A single search/browse hit. Mirrors `tools.skills_hub.SkillResult`.
#[derive(Debug, Clone)]
pub struct SkillResult {
    pub name: String,
    pub description: String,
    pub source: String,
    pub trust_level: String,
    pub identifier: String,
}

/// Skill metadata (`tools.skills_hub.SkillMeta`).
#[derive(Debug, Clone, Default)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub source: String,
    pub trust_level: String,
    pub identifier: String,
    pub path: String,
    pub tags: Vec<String>,
    pub extra: BTreeMap<String, ExtraValue>,
}

/// A fetched skill bundle (`tools.skills_hub.SkillBundle`).
#[derive(Debug, Clone, Default)]
pub struct SkillBundle {
    pub name: String,
    pub source: String,
    pub trust_level: String,
    pub identifier: String,
    /// File name -> contents. SKILL.md content lives here.
    pub files: BTreeMap<String, Vec<u8>>,
    pub metadata: BTreeMap<String, ExtraValue>,
}

/// A loosely-typed metadata value (the Python dicts mix strings/ints/bools and
/// nested maps for `security_audits`).
#[derive(Debug, Clone, PartialEq)]
pub enum ExtraValue {
    Str(String),
    Int(i64),
    Bool(bool),
    Map(BTreeMap<String, String>),
    Null,
}

impl ExtraValue {
    fn as_truthy_str(&self) -> Option<&str> {
        match self {
            ExtraValue::Str(s) if !s.is_empty() => Some(s),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Source / backend traits (mirror src.search/inspect/fetch and module-level
// helpers from tools.skills_hub).
// ---------------------------------------------------------------------------

/// One registry adapter (mirrors a source object from `create_source_router`).
pub trait SkillSource {
    fn source_id(&self) -> String;
    fn search(&self, query: &str, limit: usize) -> Vec<SkillResult>;
    fn inspect(&self, identifier: &str) -> Option<SkillMeta>;
    fn fetch(&self, identifier: &str) -> Option<SkillBundle>;
    /// GitHub API rate-limit flag (`src.is_rate_limited` / `src.github.is_rate_limited`).
    fn is_rate_limited(&self) -> bool {
        false
    }
}

/// Per-source search budget, mirroring the dicts used in `do_browse`/`browse_skills`.
pub fn default_browse_per_source_limit(source_id: &str) -> usize {
    match source_id {
        "official" => 200,
        "skills-sh" => 200,
        "well-known" => 50,
        "github" => 200,
        "clawhub" => 500,
        "claude-marketplace" => 100,
        "lobehub" => 500,
        _ => 50,
    }
}

/// Per-source budget for the programmatic `browse_skills` (smaller, mirrors
/// the second dict literal in Python).
pub fn programmatic_browse_per_source_limit(source_id: &str) -> usize {
    match source_id {
        "official" => 100,
        "skills-sh" => 100,
        "well-known" => 25,
        "github" => 100,
        "clawhub" => 50,
        "claude-marketplace" => 50,
        "lobehub" => 50,
        _ => 50,
    }
}

/// Result of a parallel multi-source browse fetch.
pub struct ParallelSearchOutcome {
    pub results: Vec<SkillResult>,
    pub source_counts: BTreeMap<String, usize>,
    pub timed_out: Vec<String>,
}

/// Backend that supplies the source router + the stateful hub operations that
/// `tools.skills_hub` / `tools.skills_guard` / `agent.*` provide in Python.
///
/// A native implementation can wire these to `crate::skills_cmd`. The trait keeps
/// `cli_skills_hub` self-contained and testable.
pub trait HubBackend {
    fn sources(&self) -> Vec<Box<dyn SkillSource>>;

    /// `tools.skills_hub.unified_search`.
    fn unified_search(&self, query: &str, source_filter: &str, limit: usize) -> Vec<SkillResult>;

    /// `tools.skills_hub.parallel_search_sources` (browse path).
    fn parallel_search_sources(
        &self,
        query: &str,
        source_filter: &str,
        overall_timeout_secs: u64,
    ) -> ParallelSearchOutcome;

    /// `SKILLS_DIR` (`~/.hermes/skills`).
    fn skills_dir(&self) -> PathBuf {
        crate::mod_hermes_constants::get_skills_dir()
    }

    /// `hermes_constants.display_hermes_home()`.
    fn display_hermes_home(&self) -> String {
        crate::mod_hermes_constants::display_hermes_home()
    }

    fn ensure_hub_dirs(&self) {}

    /// `HubLockFile.list_installed()`.
    fn list_installed(&self) -> Vec<InstalledEntry>;
    /// `HubLockFile.get_installed(name)`.
    fn get_installed(&self, name: &str) -> Option<InstalledEntry> {
        self.list_installed().into_iter().find(|e| e.name == name)
    }

    /// `TapsManager.list_taps()`.
    fn list_taps(&self) -> Vec<TapEntry>;
    /// `TapsManager.add(repo, path?)` -> True if newly added.
    fn tap_add(&self, repo: &str, path: &str) -> bool;
    /// `TapsManager.remove(repo)` -> True if removed.
    fn tap_remove(&self, repo: &str) -> bool;

    /// `tools.skills_guard.scan_skill`.
    fn scan_skill(&self, path: &Path, source: &str) -> ScanResult;
    /// `tools.skills_guard.should_allow_install` -> (allowed, reason).
    fn should_allow_install(&self, result: &ScanResult, force: bool) -> (bool, String);
    /// `tools.skills_guard.format_scan_report`.
    fn format_scan_report(&self, result: &ScanResult) -> String;

    /// `tools.skills_hub.quarantine_bundle` -> quarantine path (or error).
    fn quarantine_bundle(&self, bundle: &SkillBundle) -> Result<PathBuf, String>;
    /// `tools.skills_hub.install_from_quarantine` -> install dir (or error).
    fn install_from_quarantine(
        &self,
        q_path: &Path,
        name: &str,
        category: &str,
        bundle: &SkillBundle,
        result: &ScanResult,
    ) -> Result<PathBuf, String>;
    /// `shutil.rmtree(path, ignore_errors=True)`.
    fn rmtree(&self, path: &Path) {
        let _ = std::fs::remove_dir_all(path);
    }
    /// `tools.skills_hub.append_audit_log`.
    fn append_audit_log(
        &self,
        action: &str,
        name: &str,
        source: &str,
        trust_level: &str,
        verdict: &str,
        detail: &str,
    );

    /// `tools.skills_hub.uninstall_skill` -> (success, message).
    fn uninstall_skill(&self, name: &str) -> (bool, String);

    /// `tools.skills_sync.reset_bundled_skill`.
    fn reset_bundled_skill(&self, name: &str, restore: bool) -> ResetResult;

    /// `tools.skills_hub.check_for_skill_updates`.
    fn check_for_skill_updates(&self, name: Option<&str>) -> Vec<UpdateEntry>;

    /// Clear the skills system-prompt cache (`agent.prompt_builder.clear_skills_system_prompt_cache`).
    fn clear_skills_system_prompt_cache(&self) {}

    /// `tools.skills_sync._read_manifest()` — bundled/builtin skill names.
    fn read_manifest(&self) -> Vec<String> {
        Vec::new()
    }
    /// `tools.skills_tool._find_all_skills(skip_disabled=True)`.
    fn find_all_skills(&self) -> Vec<DiscoveredSkill> {
        Vec::new()
    }
    /// `agent.skill_utils.get_disabled_skill_names()`.
    fn disabled_skill_names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// `HubLockFile` installed-entry view.
#[derive(Debug, Clone, Default)]
pub struct InstalledEntry {
    pub name: String,
    pub source: String,
    pub identifier: String,
    pub trust_level: String,
    pub install_path: String,
}

/// `TapsManager` tap view.
#[derive(Debug, Clone, Default)]
pub struct TapEntry {
    pub repo: String,
    pub name: String,
    pub path: String,
}

/// `tools.skills_guard.scan_skill` result view.
#[derive(Debug, Clone, Default)]
pub struct ScanResult {
    pub verdict: String,
    pub findings_count: usize,
}

/// `tools.skills_sync.reset_bundled_skill` result view.
#[derive(Debug, Clone, Default)]
pub struct ResetResult {
    pub ok: bool,
    pub message: String,
    pub copied: Vec<String>,
    pub updated: Vec<String>,
}

/// `check_for_skill_updates` entry.
#[derive(Debug, Clone, Default)]
pub struct UpdateEntry {
    pub name: String,
    pub source: String,
    pub status: String,
    pub identifier: String,
}

/// `_find_all_skills` row.
#[derive(Debug, Clone, Default)]
pub struct DiscoveredSkill {
    pub name: String,
    pub category: String,
}

// ---------------------------------------------------------------------------
// Trust-style maps (mirror the inline dicts in the Python).
// ---------------------------------------------------------------------------

fn trust_style(trust_level: &str) -> &'static str {
    match trust_level {
        "builtin" => "bright_cyan",
        "trusted" => "green",
        "community" => "yellow",
        _ => "dim",
    }
}

fn trust_style_with_local(trust: &str) -> &'static str {
    match trust {
        "builtin" => "bright_cyan",
        "trusted" => "green",
        "community" => "yellow",
        "local" => "dim",
        _ => "dim",
    }
}

fn trust_rank(trust_level: &str) -> i32 {
    match trust_level {
        "builtin" => 3,
        "trusted" => 2,
        "community" => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Mirrors `_resolve_short_name`: resolve a short skill name to a full
/// identifier via search. Returns empty string when ambiguous / not found.
pub fn resolve_short_name(name: &str, backend: &dyn HubBackend, c: &dyn Console) -> String {
    c.print(&format!("[dim]Resolving '{name}'...[/]"));

    let results = backend.unified_search(name, "all", 20);

    let name_lc = name.to_lowercase();
    let exact: Vec<&SkillResult> = results
        .iter()
        .filter(|r| r.name.to_lowercase() == name_lc)
        .collect();

    if exact.len() == 1 {
        c.print(&format!("[dim]Resolved to: {}[/]", exact[0].identifier));
        return exact[0].identifier.clone();
    }

    if exact.len() > 1 {
        c.print(&format!("\n[yellow]Multiple skills named '{name}' found:[/]"));
        // Table: Source / Trust / Identifier
        for r in &exact {
            let ts = trust_style(&r.trust_level);
            let trust_label = if r.source == "official" {
                "official"
            } else {
                r.trust_level.as_str()
            };
            c.print(&format!(
                "{}  [{ts}]{trust_label}[/]  [bold cyan]{}[/]",
                r.source, r.identifier
            ));
        }
        c.print("[bold]Use the full identifier to install a specific one.[/]\n");
        return String::new();
    }

    if !results.is_empty() {
        c.print(&format!(
            "[yellow]No exact match for '{name}'. Did you mean one of these?[/]"
        ));
        for r in results.iter().take(5) {
            c.print(&format!("  [cyan]{}[/] — {}", r.name, r.identifier));
        }
        c.print("");
        return String::new();
    }

    c.print(&format!(
        "[bold red]Error:[/] No skill named '{name}' found in any source.\n"
    ));
    String::new()
}

/// Mirrors `_format_extra_metadata_lines`.
pub fn format_extra_metadata_lines(extra: &BTreeMap<String, ExtraValue>) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    if extra.is_empty() {
        return lines;
    }

    if let Some(v) = extra.get("repo_url").and_then(ExtraValue::as_truthy_str) {
        lines.push(format!("[bold]Repo:[/] {v}"));
    }
    if let Some(v) = extra.get("detail_url").and_then(ExtraValue::as_truthy_str) {
        lines.push(format!("[bold]Detail Page:[/] {v}"));
    }
    if let Some(v) = extra.get("index_url").and_then(ExtraValue::as_truthy_str) {
        lines.push(format!("[bold]Index:[/] {v}"));
    }
    if let Some(v) = extra.get("endpoint").and_then(ExtraValue::as_truthy_str) {
        lines.push(format!("[bold]Endpoint:[/] {v}"));
    }
    if let Some(v) = extra
        .get("install_command")
        .and_then(ExtraValue::as_truthy_str)
    {
        lines.push(format!("[bold]Install Command:[/] {v}"));
    }
    // installs: not None (0 is shown, only None hides it).
    if let Some(v) = extra.get("installs") {
        if !matches!(v, ExtraValue::Null) {
            lines.push(format!("[bold]Installs:[/] {}", render_extra_scalar(v)));
        }
    }
    // weekly_installs: truthy
    if let Some(v) = extra.get("weekly_installs") {
        if is_truthy(v) {
            lines.push(format!(
                "[bold]Weekly Installs:[/] {}",
                render_extra_scalar(v)
            ));
        }
    }

    if let Some(ExtraValue::Map(security)) = extra.get("security_audits") {
        if !security.is_empty() {
            // sorted(security.items())
            let ordered = security
                .iter() // BTreeMap is already sorted by key
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("[bold]Security:[/] {ordered}"));
        }
    }

    lines
}

fn render_extra_scalar(v: &ExtraValue) -> String {
    match v {
        ExtraValue::Str(s) => s.clone(),
        ExtraValue::Int(i) => i.to_string(),
        ExtraValue::Bool(b) => {
            // Python str(True) == "True"
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        ExtraValue::Map(_) => String::new(),
        ExtraValue::Null => "None".to_string(),
    }
}

fn is_truthy(v: &ExtraValue) -> bool {
    match v {
        ExtraValue::Str(s) => !s.is_empty(),
        ExtraValue::Int(i) => *i != 0,
        ExtraValue::Bool(b) => *b,
        ExtraValue::Map(m) => !m.is_empty(),
        ExtraValue::Null => false,
    }
}

/// Mirrors `_resolve_source_meta_and_bundle`. Returns (meta, bundle).
pub fn resolve_source_meta_and_bundle(
    identifier: &str,
    sources: &[Box<dyn SkillSource>],
) -> (Option<SkillMeta>, Option<SkillBundle>) {
    let mut meta: Option<SkillMeta> = None;
    let mut bundle: Option<SkillBundle> = None;

    for src in sources {
        if meta.is_none() {
            meta = src.inspect(identifier);
            // matched_source bookkeeping omitted (unused by callers here).
        }
        bundle = src.fetch(identifier);
        if bundle.is_some() {
            if meta.is_none() {
                meta = src.inspect(identifier);
            }
            break;
        }
    }

    (meta, bundle)
}

/// Mirrors `_derive_category_from_install_path`.
pub fn derive_category_from_install_path(install_path: &str) -> String {
    let path = Path::new(install_path);
    let parent = path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    if parent == "." || parent.is_empty() {
        // Python Path("x").parent == Path(".") -> "."; we normalize both to "".
        // But Path("a/b").parent == "a".
        if path.components().count() <= 1 {
            return String::new();
        }
    }
    if parent == "." {
        String::new()
    } else {
        parent
    }
}

// ---------------------------------------------------------------------------
// Interactive name/category resolution for URL-installed skills.
// ---------------------------------------------------------------------------

fn valid_name_re() -> Regex {
    Regex::new(r"^[a-z][a-z0-9_-]*$").unwrap()
}

fn valid_category_re() -> Regex {
    Regex::new(r"^[a-z][a-z0-9_/-]*$").unwrap()
}

/// Mirrors `_is_valid_installed_skill_name`.
pub fn is_valid_installed_skill_name(name: &str) -> bool {
    let candidate = name.trim().to_lowercase();
    if candidate.is_empty() {
        return false;
    }
    if matches!(
        candidate.as_str(),
        "skill" | "readme" | "index" | "unnamed-skill"
    ) {
        return false;
    }
    valid_name_re().is_match(&candidate)
}

/// Mirrors `_existing_categories`. Lists sorted subdir names under skills dir
/// that look like category buckets (no top-level SKILL.md, but at least one
/// nested SKILL.md). Hidden dirs are skipped.
pub fn existing_categories(skills_dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let entries = match std::fs::read_dir(skills_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if path.join("SKILL.md").exists() {
            continue;
        }
        if has_nested_skill_md(&path) {
            out.push(name);
        }
    }
    out.sort();
    out.dedup();
    out
}

fn has_nested_skill_md(dir: &Path) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let rd = match std::fs::read_dir(&cur) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if entry.file_name() == "SKILL.md" {
                return true;
            }
        }
    }
    false
}

/// Mirrors `_prompt_for_skill_name`. Returns None on cancel/EOF/invalid.
pub fn prompt_for_skill_name(
    c: &dyn Console,
    prompter: &dyn Prompter,
    url: &str,
    default: &str,
) -> Option<String> {
    c.print("");
    c.print(&format!(
        "[yellow]The SKILL.md at {url} doesn't declare a `name:` in its frontmatter,[/]\n[yellow]and the URL path doesn't produce a valid identifier either.[/]"
    ));
    let default_hint = if default.is_empty() {
        String::new()
    } else {
        format!(" [{default}]")
    };
    c.print(&format!(
        "[bold]Enter a skill name{default_hint}:[/] [dim](lowercase letters, digits, hyphens, underscores; starts with a letter)[/]"
    ));
    let raw = prompter.ask("Name: ")?;
    let mut answer = raw.trim().to_string();
    if answer.is_empty() && !default.is_empty() {
        answer = default.to_string();
    }
    if !is_valid_installed_skill_name(&answer) {
        c.print(&format!(
            "[bold red]Invalid name:[/] {answer:?}. Aborting install.\n"
        ));
        return None;
    }
    Some(answer)
}

/// Mirrors `_prompt_for_category`. Empty/None/invalid => flat install ("").
pub fn prompt_for_category(c: &dyn Console, prompter: &dyn Prompter, existing: &[String]) -> String {
    c.print("");
    if !existing.is_empty() {
        c.print(
            "[bold]Pick a category[/] [dim](reuse an existing bucket, type a new one, or press Enter to install flat)[/]",
        );
        c.print(&format!("[dim]Existing: {}[/]", existing.join(", ")));
    } else {
        c.print(
            "[bold]Category[/] [dim](optional — press Enter to install flat at ~/.hermes/skills/<name>/)[/]",
        );
    }
    let raw = match prompter.ask("Category: ") {
        Some(r) => r,
        None => return String::new(),
    };
    let answer = raw.trim().to_string();
    if answer.is_empty() {
        return String::new();
    }
    if !valid_category_re().is_match(&answer) {
        c.print(&format!(
            "[dim]Invalid category {answer:?} — installing flat.[/]"
        ));
        return String::new();
    }
    answer
}

// ---------------------------------------------------------------------------
// do_search
// ---------------------------------------------------------------------------

/// Mirrors `do_search`.
pub fn do_search(
    backend: &dyn HubBackend,
    query: &str,
    source: &str,
    limit: usize,
    c: &dyn Console,
) {
    c.print(&format!("\n[bold]Searching for:[/] {query}"));

    let results = backend.unified_search(query, source, limit);

    if results.is_empty() {
        c.print("[dim]No skills found matching your query.[/]\n");
        return;
    }

    c.print(&format!(
        "[Skills Hub — {} result(s)]",
        results.len()
    ));
    for r in &results {
        let ts = trust_style(&r.trust_level);
        let trust_label = if r.source == "official" {
            "official"
        } else {
            r.trust_level.as_str()
        };
        let desc = truncate_with_ellipsis(&r.description, 60);
        c.print(&format!(
            "{name} | {desc} | {source} | [{ts}]{trust_label}[/] | {ident}",
            name = r.name,
            source = r.source,
            ident = r.identifier
        ));
    }

    c.print(
        "[dim]Use: hermes skills inspect <identifier> to preview, hermes skills install <identifier> to install[/]\n",
    );
}

fn truncate_with_ellipsis(s: &str, max: usize) -> String {
    // Python: s[:max] + ("..." if len(s) > max else "")  (char-based slice).
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > max {
        let head: String = chars[..max].iter().collect();
        format!("{head}...")
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// do_browse
// ---------------------------------------------------------------------------

/// Mirrors `do_browse`.
pub fn do_browse(backend: &dyn HubBackend, page: i64, page_size: i64, source: &str, c: &dyn Console) {
    // Clamp page_size to [1, 100]
    let page_size = page_size.clamp(1, 100);

    let outcome = backend.parallel_search_sources(/*query*/ "", source, 30);
    let all_results = outcome.results;
    let source_counts = outcome.source_counts;
    let timed_out = outcome.timed_out;

    if all_results.is_empty() {
        c.print("[dim]No skills found in the Skills Hub.[/]\n");
        return;
    }

    let deduped = dedupe_and_sort(all_results);

    let total = deduped.len() as i64;
    let total_pages = std::cmp::max(1, (total + page_size - 1) / page_size);
    let page = page.clamp(1, total_pages);
    let start = (page - 1) * page_size;
    let end = std::cmp::min(start + page_size, total);
    let page_items = &deduped[start as usize..end as usize];

    let official_count = deduped.iter().filter(|r| r.source == "official").count();

    let source_label = if source != "all" {
        format!("— {source}")
    } else {
        "— all sources".to_string()
    };
    let mut loaded_label = format!("{total} skills loaded");
    if !timed_out.is_empty() {
        loaded_label += &format!(", {} source(s) still loading", timed_out.len());
    }
    c.print(&format!(
        "\n[bold]Skills Hub — Browse {source_label}[/]  [dim]({loaded_label}, page {page}/{total_pages})[/]"
    ));
    if official_count > 0 && page == 1 {
        c.print(&format!(
            "[bright_cyan]★ {official_count} official optional skill(s) from Nous Research[/]"
        ));
    }
    c.print("");

    for (offset, r) in page_items.iter().enumerate() {
        let i = start + 1 + offset as i64;
        let ts = trust_style(&r.trust_level);
        let trust_label = if r.source == "official" {
            "★ official".to_string()
        } else {
            r.trust_level.clone()
        };
        let desc = truncate_with_ellipsis(&r.description, 50);
        c.print(&format!(
            "{i} | {name} | {desc} | {source} | [{ts}]{trust_label}[/]",
            name = r.name,
            source = r.source
        ));
    }

    // Navigation hints
    let mut nav_parts: Vec<String> = Vec::new();
    if page > 1 {
        nav_parts.push(format!("[cyan]--page {}[/] ← prev", page - 1));
    }
    if page < total_pages {
        nav_parts.push(format!("[cyan]--page {}[/] → next", page + 1));
    }
    if !nav_parts.is_empty() {
        c.print(&format!("  {}", nav_parts.join(" | ")));
    }

    if source == "all" && !source_counts.is_empty() {
        // sorted(source_counts.items()) — BTreeMap iterates sorted.
        let parts: Vec<String> = source_counts
            .iter()
            .map(|(sid, ct)| format!("{sid}: {ct}"))
            .collect();
        c.print(&format!("  [dim]Sources: {}[/]", parts.join(", ")));
    }

    if !timed_out.is_empty() {
        c.print(&format!(
            "  [yellow]⚡ Slow sources skipped: {} — run again for cached results[/]",
            timed_out.join(", ")
        ));
    }

    c.print(
        "[dim]Tip: 'hermes skills search <query>' searches deeper across all registries[/]\n",
    );
}

fn dedupe_and_sort(all_results: Vec<SkillResult>) -> Vec<SkillResult> {
    // Deduplicate by name, preferring higher trust. Python dict preserves
    // insertion order; we mirror that with an ordered keys list.
    let mut order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashMap<String, SkillResult> = std::collections::HashMap::new();
    for r in all_results {
        let rank = trust_rank(&r.trust_level);
        match seen.get(&r.name) {
            Some(existing) if rank <= trust_rank(&existing.trust_level) => {}
            Some(_) => {
                seen.insert(r.name.clone(), r);
            }
            None => {
                order.push(r.name.clone());
                seen.insert(r.name.clone(), r);
            }
        }
    }
    let mut deduped: Vec<SkillResult> = order
        .into_iter()
        .filter_map(|k| seen.remove(&k))
        .collect();

    // Sort: -trust_rank, source != "official", name.lower()
    deduped.sort_by(|a, b| {
        let ka = (
            -trust_rank(&a.trust_level),
            a.source != "official",
            a.name.to_lowercase(),
        );
        let kb = (
            -trust_rank(&b.trust_level),
            b.source != "official",
            b.name.to_lowercase(),
        );
        ka.cmp(&kb)
    });
    deduped
}

// ---------------------------------------------------------------------------
// do_install
// ---------------------------------------------------------------------------

/// Options for [`do_install`], mirroring the Python keyword args.
pub struct InstallOptions<'a> {
    pub category: String,
    pub force: bool,
    pub skip_confirm: bool,
    pub invalidate_cache: bool,
    pub name_override: String,
    pub prompter: &'a dyn Prompter,
}

impl<'a> Default for InstallOptions<'a> {
    fn default() -> Self {
        InstallOptions {
            category: String::new(),
            force: false,
            skip_confirm: false,
            invalidate_cache: true,
            name_override: String::new(),
            prompter: &NO_PROMPTER,
        }
    }
}

static NO_PROMPTER: NoPrompter = NoPrompter;

/// Mirrors `do_install`.
pub fn do_install(backend: &dyn HubBackend, identifier: &str, opts: InstallOptions, c: &dyn Console) {
    backend.ensure_hub_dirs();

    let mut identifier = identifier.to_string();
    let mut category = opts.category.clone();

    let sources = backend.sources();

    // Short-name resolution
    if !identifier.contains('/') {
        identifier = resolve_short_name(&identifier, backend, c);
        if identifier.is_empty() {
            return;
        }
    }

    c.print(&format!("\n[bold]Fetching:[/] {identifier}"));

    let (mut meta, bundle) = resolve_source_meta_and_bundle(&identifier, &sources);

    let mut bundle = match bundle {
        Some(b) => b,
        None => {
            let rate_limited = sources.iter().any(|s| s.is_rate_limited());
            c.print(&format!(
                "[bold red]Error:[/] Could not fetch '{identifier}' from any source."
            ));
            if rate_limited {
                c.print(
                    "[yellow]Hint:[/] GitHub API rate limit exhausted (unauthenticated: 60 requests/hour).\nSet [bold]GITHUB_TOKEN[/] in your .env or install the [bold]gh[/] CLI and run [bold]gh auth login[/] to raise the limit to 5,000/hr.\n",
                );
            } else {
                c.print("");
            }
            return;
        }
    };

    // URL-sourced skills: resolve missing name.
    let awaiting_name = matches!(
        bundle.metadata.get("awaiting_name"),
        Some(ExtraValue::Bool(true))
    );
    if bundle.source == "url" && (bundle.name.is_empty() || awaiting_name) {
        if !opts.name_override.is_empty() && is_valid_installed_skill_name(&opts.name_override) {
            bundle.name = opts.name_override.trim().to_string();
            bundle
                .metadata
                .insert("awaiting_name".to_string(), ExtraValue::Bool(false));
        } else if !opts.name_override.is_empty() {
            c.print(&format!(
                "[bold red]Invalid --name:[/] {:?}. Must be a lowercase identifier (letters, digits, hyphens, underscores; starts with a letter).\n",
                opts.name_override
            ));
            return;
        } else if opts.skip_confirm {
            let url = bundle
                .metadata
                .get("url")
                .and_then(ExtraValue::as_truthy_str)
                .unwrap_or(&identifier)
                .to_string();
            c.print(&format!(
                "[bold red]Cannot install from URL:[/] {url}\n[yellow]The SKILL.md has no `name:` in its frontmatter, and the URL path doesn't produce a valid identifier.[/]\n\nRetry with an explicit name:\n  [bold]/skills install {url} --name <your-name>[/]\n  [bold]hermes skills install {url} --name <your-name>[/]\n\n[dim]Or ask the SKILL.md's author to add a `name:` field to its YAML frontmatter.[/]\n"
            ));
            return;
        } else {
            let url = bundle
                .metadata
                .get("url")
                .and_then(ExtraValue::as_truthy_str)
                .unwrap_or(&identifier)
                .to_string();
            let chosen = prompt_for_skill_name(c, opts.prompter, &url, "");
            match chosen {
                Some(name) => {
                    bundle.name = name;
                    bundle
                        .metadata
                        .insert("awaiting_name".to_string(), ExtraValue::Bool(false));
                }
                None => {
                    c.print("[dim]Installation cancelled.[/]\n");
                    return;
                }
            }
        }
        if let Some(m) = meta.as_mut() {
            m.name = bundle.name.clone();
            m.path = bundle.name.clone();
        }
    }

    // URL-sourced: interactive category pick.
    if bundle.source == "url" && category.is_empty() && !opts.skip_confirm {
        let existing = existing_categories(&backend.skills_dir());
        category = prompt_for_category(c, opts.prompter, &existing);
    }

    // Auto-detect category for official skills.
    if bundle.source == "official" && category.is_empty() {
        let parts: Vec<&str> = bundle.identifier.split('/').collect();
        if parts.len() >= 3 {
            category = parts[1].to_string();
        }
    }

    // Already installed?
    if let Some(existing) = backend.get_installed(&bundle.name) {
        c.print(&format!(
            "[yellow]Warning:[/] '{}' is already installed at {}",
            bundle.name, existing.install_path
        ));
        if !opts.force {
            c.print("Use --force to reinstall.\n");
            return;
        }
    }

    // Merge extra metadata: meta.extra updated with bundle.metadata.
    let mut extra_metadata: BTreeMap<String, ExtraValue> =
        meta.as_ref().map(|m| m.extra.clone()).unwrap_or_default();
    for (k, v) in &bundle.metadata {
        extra_metadata.insert(k.clone(), v.clone());
    }

    // Quarantine
    let q_path = match backend.quarantine_bundle(&bundle) {
        Ok(p) => p,
        Err(exc) => {
            c.print(&format!("[bold red]Installation blocked:[/] {exc}\n"));
            backend.append_audit_log(
                "BLOCKED",
                &bundle.name,
                &bundle.source,
                &bundle.trust_level,
                "invalid_path",
                &exc,
            );
            return;
        }
    };
    c.print(&format!(
        "[dim]Quarantined to {}[/]",
        quarantine_display(&q_path)
    ));

    // Scan
    c.print("[bold]Running security scan...[/]");
    let scan_source = if !bundle.identifier.is_empty() {
        bundle.identifier.clone()
    } else if let Some(m) = meta.as_ref().filter(|m| !m.identifier.is_empty()) {
        m.identifier.clone()
    } else {
        identifier.clone()
    };
    let result = backend.scan_skill(&q_path, &scan_source);
    c.print(&backend.format_scan_report(&result));

    // Install policy
    let (allowed, reason) = backend.should_allow_install(&result, opts.force);
    if !allowed {
        c.print(&format!("\n[bold red]Installation blocked:[/] {reason}"));
        backend.rmtree(&q_path);
        backend.append_audit_log(
            "BLOCKED",
            &bundle.name,
            &bundle.source,
            &bundle.trust_level,
            &result.verdict,
            &format!("{}_findings", result.findings_count),
        );
        return;
    }

    if !extra_metadata.is_empty() {
        let metadata_lines = format_extra_metadata_lines(&extra_metadata);
        if !metadata_lines.is_empty() {
            c.print(&panel(
                &metadata_lines.join("\n"),
                "Upstream Metadata",
                "blue",
            ));
        }
    }

    // Confirm with user
    if !opts.force && !opts.skip_confirm {
        c.print("");
        let home = backend.display_hermes_home();
        let cat_seg = if category.is_empty() {
            String::new()
        } else {
            format!("{category}/")
        };
        if bundle.source == "official" {
            c.print(&panel(
                &format!(
                    "[bold bright_cyan]This is an official optional skill maintained by Nous Research.[/]\n\nIt ships with hermes-agent but is not activated by default.\nInstalling will copy it to your skills directory where the agent can use it.\n\nFiles will be at: [cyan]{home}/skills/{cat_seg}{}/[/]",
                    bundle.name
                ),
                "Official Skill",
                "bright_cyan",
            ));
        } else {
            c.print(&panel(
                &format!(
                    "[bold yellow]You are installing a third-party skill at your own risk.[/]\n\nExternal skills can contain instructions that influence agent behavior,\nshell commands, and scripts. Even after automated scanning, you should\nreview the installed files before use.\n\nFiles will be at: [cyan]{home}/skills/{cat_seg}{}/[/]",
                    bundle.name
                ),
                "Disclaimer",
                "yellow",
            ));
        }
        c.print(&format!("[bold]Install '{}'?[/]", bundle.name));
        let answer = opts
            .prompter
            .ask("Confirm [y/N]: ")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_else(|| "n".to_string());
        if answer != "y" && answer != "yes" {
            c.print("[dim]Installation cancelled.[/]\n");
            backend.rmtree(&q_path);
            return;
        }
    }

    // Install
    let install_dir =
        match backend.install_from_quarantine(&q_path, &bundle.name, &category, &bundle, &result) {
            Ok(d) => d,
            Err(exc) => {
                c.print(&format!("[bold red]Installation blocked:[/] {exc}\n"));
                backend.rmtree(&q_path);
                backend.append_audit_log(
                    "BLOCKED",
                    &bundle.name,
                    &bundle.source,
                    &bundle.trust_level,
                    "invalid_path",
                    &exc,
                );
                return;
            }
        };

    let skills_dir = backend.skills_dir();
    let rel = install_dir
        .strip_prefix(&skills_dir)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| install_dir.to_string_lossy().to_string());
    c.print(&format!("[bold green]Installed:[/] {rel}"));
    let files: Vec<&str> = bundle.files.keys().map(|s| s.as_str()).collect();
    c.print(&format!("[dim]Files: {}[/]\n", files.join(", ")));

    if opts.invalidate_cache {
        backend.clear_skills_system_prompt_cache();
    } else {
        c.print("[dim]Skill will be available in your next session.[/]");
        c.print(
            "[dim]Use /reset to start a new session now, or --now to activate immediately (invalidates prompt cache).[/]\n",
        );
    }
}

fn quarantine_display(q_path: &Path) -> String {
    // Python: q_path.relative_to(q_path.parent.parent.parent)
    let base = q_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent());
    match base {
        Some(b) => q_path
            .strip_prefix(b)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| q_path.to_string_lossy().to_string()),
        None => q_path.to_string_lossy().to_string(),
    }
}

/// Renders a rich Panel as plain text (title + body). Keeps body content exact.
fn panel(body: &str, title: &str, border_style: &str) -> String {
    format!("[panel border={border_style} title={title}]\n{body}")
}

// ---------------------------------------------------------------------------
// do_inspect
// ---------------------------------------------------------------------------

/// Mirrors `do_inspect`.
pub fn do_inspect(backend: &dyn HubBackend, identifier: &str, c: &dyn Console) {
    let sources = backend.sources();
    let mut identifier = identifier.to_string();

    if !identifier.contains('/') {
        identifier = resolve_short_name(&identifier, backend, c);
        if identifier.is_empty() {
            return;
        }
    }

    let (meta, bundle) = resolve_source_meta_and_bundle(&identifier, &sources);

    let meta = match meta {
        Some(m) => m,
        None => {
            c.print(&format!(
                "[bold red]Error:[/] Could not find '{identifier}' in any source.\n"
            ));
            return;
        }
    };

    c.print("");
    let ts = trust_style(&meta.trust_level);
    let trust_label = if meta.source == "official" {
        "official"
    } else {
        meta.trust_level.as_str()
    };

    let mut info_lines = vec![
        format!("[bold]Name:[/] {}", meta.name),
        format!("[bold]Description:[/] {}", meta.description),
        format!("[bold]Source:[/] {}", meta.source),
        format!("[bold]Trust:[/] [{ts}]{trust_label}[/]"),
        format!("[bold]Identifier:[/] {}", meta.identifier),
    ];
    if !meta.tags.is_empty() {
        info_lines.push(format!("[bold]Tags:[/] {}", meta.tags.join(", ")));
    }
    info_lines.extend(format_extra_metadata_lines(&meta.extra));

    c.print(&panel(
        &info_lines.join("\n"),
        &format!("Skill: {}", meta.name),
        "",
    ));

    if let Some(b) = bundle.as_ref() {
        if let Some(content_bytes) = b.files.get("SKILL.md") {
            let content = String::from_utf8_lossy(content_bytes);
            let lines: Vec<&str> = content.split('\n').collect();
            let mut preview = lines.iter().take(50).cloned().collect::<Vec<_>>().join("\n");
            if lines.len() > 50 {
                preview += &format!("\n\n... ({} more lines)", lines.len() - 50);
            }
            c.print(&panel(&preview, "SKILL.md Preview", ""));
        }
    }

    c.print("");
}

// ---------------------------------------------------------------------------
// browse_skills (programmatic) / inspect_skill (programmatic)
// ---------------------------------------------------------------------------

/// One programmatic browse item.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrowseItem {
    pub name: String,
    pub description: String,
    pub source: String,
    pub trust: String,
}

/// `browse_skills` programmatic result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrowsePage {
    pub items: Vec<BrowseItem>,
    pub page: i64,
    pub total_pages: i64,
    pub total: i64,
}

/// Mirrors `browse_skills`.
pub fn browse_skills(backend: &dyn HubBackend, page: i64, page_size: i64, source: &str) -> BrowsePage {
    let page_size = page_size.clamp(1, 100);
    let sources = backend.sources();
    let mut all_results: Vec<SkillResult> = Vec::new();
    for src in &sources {
        let sid = src.source_id();
        if source != "all" && sid != source && sid != "official" {
            continue;
        }
        let limit = programmatic_browse_per_source_limit(&sid);
        all_results.extend(src.search("", limit));
    }
    if all_results.is_empty() {
        return BrowsePage {
            items: Vec::new(),
            page: 1,
            total_pages: 1,
            total: 0,
        };
    }
    let deduped = dedupe_and_sort(all_results);
    let total = deduped.len() as i64;
    let total_pages = std::cmp::max(1, (total + page_size - 1) / page_size);
    let page = page.clamp(1, total_pages);
    let start = (page - 1) * page_size;
    let end = std::cmp::min(start + page_size, total);
    let page_items = &deduped[start as usize..end as usize];
    BrowsePage {
        items: page_items
            .iter()
            .map(|r| BrowseItem {
                name: r.name.clone(),
                description: r.description.clone(),
                source: r.source.clone(),
                trust: r.trust_level.clone(),
            })
            .collect(),
        page,
        total_pages,
        total,
    }
}

/// `inspect_skill` programmatic result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InspectInfo {
    pub name: String,
    pub description: String,
    pub source: String,
    pub identifier: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_md_preview: Option<String>,
}

/// Mirrors `inspect_skill` (programmatic, quiet).
pub fn inspect_skill(backend: &dyn HubBackend, identifier: &str) -> Option<InspectInfo> {
    let quiet = QuietConsole;
    let sources = backend.sources();
    let mut ident = identifier.to_string();
    if !ident.contains('/') {
        ident = resolve_short_name(&ident, backend, &quiet);
        if ident.is_empty() {
            return None;
        }
    }
    let (meta, bundle) = resolve_source_meta_and_bundle(&ident, &sources);
    let meta = meta?;
    let mut info = InspectInfo {
        name: meta.name.clone(),
        description: meta.description.clone(),
        source: meta.source.clone(),
        identifier: meta.identifier.clone(),
        tags: meta.tags.clone(),
        skill_md_preview: None,
    };
    if let Some(b) = bundle.as_ref() {
        if let Some(content_bytes) = b.files.get("SKILL.md") {
            let content = String::from_utf8_lossy(content_bytes);
            let lines: Vec<&str> = content.split('\n').collect();
            let mut preview = lines.iter().take(50).cloned().collect::<Vec<_>>().join("\n");
            if lines.len() > 50 {
                preview += &format!("\n\n... ({} more lines)", lines.len() - 50);
            }
            info.skill_md_preview = Some(preview);
        }
    }
    Some(info)
}

// ---------------------------------------------------------------------------
// do_list
// ---------------------------------------------------------------------------

/// Mirrors `do_list`.
pub fn do_list(backend: &dyn HubBackend, source_filter: &str, enabled_only: bool, c: &dyn Console) {
    backend.ensure_hub_dirs();
    let hub_installed: std::collections::HashMap<String, InstalledEntry> = backend
        .list_installed()
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();
    let builtin_names: std::collections::HashSet<String> =
        backend.read_manifest().into_iter().collect();

    let all_skills = backend.find_all_skills();
    let disabled_names: std::collections::HashSet<String> =
        backend.disabled_skill_names().into_iter().collect();

    let mut title = "Installed Skills".to_string();
    if enabled_only {
        title += " (enabled only)";
    }
    c.print(&format!("[{title}]"));

    let mut hub_count = 0usize;
    let mut builtin_count = 0usize;
    let mut local_count = 0usize;
    let mut enabled_count = 0usize;
    let mut disabled_count = 0usize;

    // sorted by (category or "", name)
    let mut skills = all_skills;
    skills.sort_by(|a, b| {
        (a.category.clone(), a.name.clone()).cmp(&(b.category.clone(), b.name.clone()))
    });

    for skill in &skills {
        let name = &skill.name;
        let category = &skill.category;
        let hub_entry = hub_installed.get(name);

        let (source_type, source_display, trust): (&str, String, String) =
            if let Some(entry) = hub_entry {
                (
                    "hub",
                    if entry.source.is_empty() {
                        "hub".to_string()
                    } else {
                        entry.source.clone()
                    },
                    if entry.trust_level.is_empty() {
                        "community".to_string()
                    } else {
                        entry.trust_level.clone()
                    },
                )
            } else if builtin_names.contains(name) {
                ("builtin", "builtin".to_string(), "builtin".to_string())
            } else {
                ("local", "local".to_string(), "local".to_string())
            };

        if source_filter != "all" && source_filter != source_type {
            continue;
        }

        let is_enabled = !disabled_names.contains(name);
        if enabled_only && !is_enabled {
            continue;
        }

        match source_type {
            "hub" => hub_count += 1,
            "builtin" => builtin_count += 1,
            _ => local_count += 1,
        }

        let status_cell = if is_enabled {
            enabled_count += 1;
            "[bold green]enabled[/]"
        } else {
            disabled_count += 1;
            "[dim red]disabled[/]"
        };

        let ts = trust_style_with_local(&trust);
        let trust_label = if source_display == "official" {
            "official".to_string()
        } else {
            trust.clone()
        };
        c.print(&format!(
            "{name} | {category} | {source_display} | [{ts}]{trust_label}[/] | {status_cell}"
        ));
    }

    let mut summary = format!(
        "[dim]{hub_count} hub-installed, {builtin_count} builtin, {local_count} local"
    );
    if enabled_only {
        summary += &format!(" — {enabled_count} enabled shown");
    } else {
        summary += &format!(" — {enabled_count} enabled, {disabled_count} disabled");
    }
    summary += "[/]\n";
    c.print(&summary);
}

// ---------------------------------------------------------------------------
// do_check / do_update / do_audit
// ---------------------------------------------------------------------------

/// Mirrors `do_check`.
pub fn do_check(backend: &dyn HubBackend, name: Option<&str>, c: &dyn Console) {
    let results = backend.check_for_skill_updates(name);
    if results.is_empty() {
        c.print("[dim]No hub-installed skills to check.[/]\n");
        return;
    }
    c.print("[Skill Updates]");
    for entry in &results {
        c.print(&format!(
            "{} | {} | {}",
            entry.name, entry.source, entry.status
        ));
    }
    let update_count = results
        .iter()
        .filter(|e| e.status == "update_available")
        .count();
    c.print(&format!(
        "[dim]{update_count} update(s) available across {} checked skill(s)[/]\n",
        results.len()
    ));
}

/// Mirrors `do_update`.
pub fn do_update(backend: &dyn HubBackend, name: Option<&str>, c: &dyn Console) {
    let updates: Vec<UpdateEntry> = backend
        .check_for_skill_updates(name)
        .into_iter()
        .filter(|e| e.status == "update_available")
        .collect();
    if updates.is_empty() {
        c.print("[dim]No updates available.[/]\n");
        return;
    }
    for entry in &updates {
        let installed = backend.get_installed(&entry.name);
        let category = installed
            .map(|i| derive_category_from_install_path(&i.install_path))
            .unwrap_or_default();
        c.print(&format!("[bold]Updating:[/] {}", entry.name));
        do_install(
            backend,
            &entry.identifier,
            InstallOptions {
                category,
                force: true,
                ..Default::default()
            },
            c,
        );
    }
    c.print(&format!(
        "[bold green]Updated {} skill(s).[/]\n",
        updates.len()
    ));
}

/// Mirrors `do_audit`.
pub fn do_audit(backend: &dyn HubBackend, name: Option<&str>, c: &dyn Console) {
    let installed = backend.list_installed();
    if installed.is_empty() {
        c.print("[dim]No hub-installed skills to audit.[/]\n");
        return;
    }

    let targets: Vec<InstalledEntry> = if let Some(n) = name {
        let t: Vec<InstalledEntry> = installed.into_iter().filter(|e| e.name == n).collect();
        if t.is_empty() {
            c.print(&format!(
                "[bold red]Error:[/] '{n}' is not a hub-installed skill.\n"
            ));
            return;
        }
        t
    } else {
        installed
    };

    c.print(&format!("\n[bold]Auditing {} skill(s)...[/]\n", targets.len()));

    let skills_dir = backend.skills_dir();
    for entry in &targets {
        let skill_path = skills_dir.join(&entry.install_path);
        if !skill_path.exists() {
            c.print(&format!(
                "[yellow]Warning:[/] {} — path missing: {}",
                entry.name, entry.install_path
            ));
            continue;
        }
        let source = if entry.identifier.is_empty() {
            entry.source.clone()
        } else {
            entry.identifier.clone()
        };
        let result = backend.scan_skill(&skill_path, &source);
        c.print(&backend.format_scan_report(&result));
        c.print("");
    }
}

// ---------------------------------------------------------------------------
// do_uninstall / do_reset
// ---------------------------------------------------------------------------

/// Mirrors `do_uninstall`.
pub fn do_uninstall(
    backend: &dyn HubBackend,
    name: &str,
    c: &dyn Console,
    prompter: &dyn Prompter,
    skip_confirm: bool,
    invalidate_cache: bool,
) {
    if !skip_confirm {
        c.print(&format!("\n[bold]Uninstall '{name}'?[/]"));
        let answer = prompter
            .ask("Confirm [y/N]: ")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_else(|| "n".to_string());
        if answer != "y" && answer != "yes" {
            c.print("[dim]Cancelled.[/]\n");
            return;
        }
    }

    let (success, msg) = backend.uninstall_skill(name);
    if success {
        c.print(&format!("[bold green]{msg}[/]\n"));
        if invalidate_cache {
            backend.clear_skills_system_prompt_cache();
        } else {
            c.print("[dim]Change will take effect in your next session.[/]");
            c.print(
                "[dim]Use /reset to start a new session now, or --now to apply immediately (invalidates prompt cache).[/]\n",
            );
        }
    } else {
        c.print(&format!("[bold red]Error:[/] {msg}\n"));
    }
}

/// Mirrors `do_reset`.
pub fn do_reset(
    backend: &dyn HubBackend,
    name: &str,
    restore: bool,
    c: &dyn Console,
    prompter: &dyn Prompter,
    skip_confirm: bool,
    invalidate_cache: bool,
) {
    if !skip_confirm && restore {
        c.print(&format!("\n[bold]Restore '{name}' from bundled source?[/]"));
        c.print("[dim]This will DELETE your current copy and re-copy the bundled version.[/]");
        let answer = prompter
            .ask("Confirm [y/N]: ")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_else(|| "n".to_string());
        if answer != "y" && answer != "yes" {
            c.print("[dim]Cancelled.[/]\n");
            return;
        }
    }

    let result = backend.reset_bundled_skill(name, restore);

    if !result.ok {
        c.print(&format!("[bold red]Error:[/] {}\n", result.message));
        return;
    }

    c.print(&format!("[bold green]{}[/]", result.message));
    if !result.copied.is_empty() {
        c.print(&format!("[dim]Copied: {}[/]", result.copied.join(", ")));
    }
    if !result.updated.is_empty() {
        c.print(&format!("[dim]Updated: {}[/]", result.updated.join(", ")));
    }
    c.print("");

    if invalidate_cache {
        backend.clear_skills_system_prompt_cache();
    } else {
        c.print("[dim]Change will take effect in your next session.[/]");
        c.print(
            "[dim]Use /reset to start a new session now, or --now to apply immediately (invalidates prompt cache).[/]\n",
        );
    }
}

// ---------------------------------------------------------------------------
// do_tap
// ---------------------------------------------------------------------------

/// Mirrors `do_tap`.
pub fn do_tap(backend: &dyn HubBackend, action: &str, repo: &str, c: &dyn Console) {
    match action {
        "list" => {
            let taps = backend.list_taps();
            if taps.is_empty() {
                c.print("[dim]No custom taps configured. Using default sources only.[/]\n");
                return;
            }
            c.print("[Configured Taps]");
            for t in &taps {
                // label = repo or name or path or "unknown"
                let label = if !t.repo.is_empty() {
                    t.repo.clone()
                } else if !t.name.is_empty() {
                    t.name.clone()
                } else if !t.path.is_empty() {
                    t.path.clone()
                } else {
                    "unknown".to_string()
                };
                let path = if t.path.is_empty() {
                    "skills/".to_string()
                } else {
                    t.path.clone()
                };
                c.print(&format!("{label} | {path}"));
            }
            c.print("");
        }
        "add" => {
            if repo.is_empty() {
                c.print(
                    "[bold red]Error:[/] Repo required. Usage: hermes skills tap add owner/repo\n",
                );
                return;
            }
            if backend.tap_add(repo, "skills/") {
                c.print(&format!("[bold green]Added tap:[/] {repo}\n"));
            } else {
                c.print(&format!("[yellow]Tap already exists:[/] {repo}\n"));
            }
        }
        "remove" => {
            if repo.is_empty() {
                c.print(
                    "[bold red]Error:[/] Repo required. Usage: hermes skills tap remove owner/repo\n",
                );
                return;
            }
            if backend.tap_remove(repo) {
                c.print(&format!("[bold green]Removed tap:[/] {repo}\n"));
            } else {
                c.print(&format!("[bold red]Error:[/] Tap not found: {repo}\n"));
            }
        }
        _ => {
            c.print(&format!(
                "[bold red]Unknown tap action:[/] {action}. Use: list, add, remove\n"
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// do_publish + _github_publish
// ---------------------------------------------------------------------------

/// Minimal GitHub auth shim (mirrors `tools.skills_hub.GitHubAuth`).
pub trait GitHubAuth {
    fn is_authenticated(&self) -> bool;
    /// Headers for GitHub API requests (`auth.get_headers()`).
    fn headers(&self) -> Vec<(String, String)>;
}

/// Mirrors `do_publish`.
pub fn do_publish(
    backend: &dyn HubBackend,
    auth: &dyn GitHubAuth,
    skill_path: &str,
    target: &str,
    repo: &str,
    c: &dyn Console,
) {
    let mut path = PathBuf::from(skill_path);
    if !path.is_absolute() {
        path = backend.skills_dir().join(skill_path);
    }
    if !path.exists() || !path.join("SKILL.md").exists() {
        c.print(&format!(
            "[bold red]Error:[/] No SKILL.md found at {}\n",
            path.display()
        ));
        return;
    }

    // Parse frontmatter for name/description
    let skill_md = std::fs::read_to_string(path.join("SKILL.md")).unwrap_or_default();
    let fm = parse_frontmatter(&skill_md);
    let name = fm
        .get("name")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
        });
    let description = fm.get("description").cloned().unwrap_or_default();
    if description.is_empty() {
        c.print("[bold red]Error:[/] SKILL.md must have a 'description' in frontmatter.\n");
        return;
    }

    c.print(&format!("[bold]Scanning '{name}' before publish...[/]"));
    let result = backend.scan_skill(&path, "self");
    c.print(&backend.format_scan_report(&result));
    if result.verdict == "dangerous" {
        c.print("[bold red]Cannot publish a skill with DANGEROUS verdict.[/]\n");
        return;
    }

    match target {
        "github" => {
            if repo.is_empty() {
                c.print(
                    "[bold red]Error:[/] --repo required for GitHub publish.\nUsage: hermes skills publish <path> --to github --repo owner/repo\n",
                );
                return;
            }
            if !auth.is_authenticated() {
                c.print(&format!(
                    "[bold red]Error:[/] GitHub authentication required.\nSet GITHUB_TOKEN in {}/.env or run 'gh auth login'.\n",
                    backend.display_hermes_home()
                ));
                return;
            }
            c.print(&format!("[bold]Publishing '{name}' to {repo}...[/]"));
            let (success, msg) = github_publish(&path, &name, repo, auth);
            if success {
                c.print(&format!("[bold green]{msg}[/]\n"));
            } else {
                c.print(&format!("[bold red]Error:[/] {msg}\n"));
            }
        }
        "clawhub" => {
            c.print(
                "[yellow]ClawHub publishing is not yet supported. Submit manually at https://clawhub.ai/submit[/]\n",
            );
        }
        other => {
            c.print(&format!(
                "[bold red]Unknown target:[/] {other}. Use 'github' or 'clawhub'.\n"
            ));
        }
    }
}

/// Parse YAML frontmatter `name:`/`description:` from a SKILL.md.
/// Mirrors the Python: only if the doc starts with `---`, find the closing
/// `\n---\s*\n`, then `yaml.safe_load` that block.
fn parse_frontmatter(skill_md: &str) -> BTreeMap<String, String> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    if !skill_md.starts_with("---") {
        return out;
    }
    let rest = &skill_md[3..];
    // re.search(r'\n---\s*\n', rest)
    let re = Regex::new(r"\n---[ \t]*\n").unwrap();
    let m = match re.find(rest) {
        Some(m) => m,
        None => return out,
    };
    let block = &rest[..m.start()];
    if let Ok(serde_yaml::Value::Mapping(map)) = serde_yaml::from_str::<serde_yaml::Value>(block) {
        for (k, v) in map {
            if let (Some(ks), Some(vs)) = (k.as_str(), v.as_str()) {
                out.insert(ks.to_string(), vs.to_string());
            }
        }
    }
    out
}

/// Mirrors `_github_publish`. Returns (success, message). Constructs the same
/// fork -> branch -> upload -> PR request sequence with reqwest::blocking.
pub fn github_publish(
    skill_path: &Path,
    skill_name: &str,
    target_repo: &str,
    auth: &dyn GitHubAuth,
) -> (bool, String) {
    use reqwest::blocking::Client;
    use std::time::Duration;

    let client = match Client::builder().build() {
        Ok(c) => c,
        Err(e) => return (false, format!("Network error: {e}")),
    };

    let header_map = || {
        let mut hm = reqwest::header::HeaderMap::new();
        for (k, v) in auth.headers() {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(&v),
            ) {
                hm.insert(name, val);
            }
        }
        // GitHub requires a User-Agent
        hm.entry(reqwest::header::USER_AGENT)
            .or_insert(reqwest::header::HeaderValue::from_static("hermes-skills-hub"));
        hm
    };

    // 1. Fork the repo
    let fork_repo: String;
    {
        let resp = client
            .post(format!(
                "https://api.github.com/repos/{target_repo}/forks"
            ))
            .headers(header_map())
            .timeout(Duration::from_secs(30))
            .send();
        match resp {
            Ok(r) => {
                let status = r.status().as_u16();
                if status == 200 || status == 202 {
                    let fork: serde_json::Value = match r.json() {
                        Ok(v) => v,
                        Err(e) => return (false, format!("Network error forking repo: {e}")),
                    };
                    fork_repo = fork
                        .get("full_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                } else if status == 403 {
                    return (false, "GitHub token lacks permission to fork repos".to_string());
                } else {
                    return (false, format!("Failed to fork {target_repo}: {status}"));
                }
            }
            Err(e) => return (false, format!("Network error forking repo: {e}")),
        }
    }

    // 2. Get default branch
    let default_branch = client
        .get(format!("https://api.github.com/repos/{target_repo}"))
        .headers(header_map())
        .timeout(Duration::from_secs(15))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| {
            v.get("default_branch")
                .and_then(|b| b.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "main".to_string());

    // 3. Get the base tree SHA
    let base_sha = {
        let resp = client
            .get(format!(
                "https://api.github.com/repos/{fork_repo}/git/refs/heads/{default_branch}"
            ))
            .headers(header_map())
            .timeout(Duration::from_secs(15))
            .send();
        let val: serde_json::Value = match resp.and_then(|r| r.json()) {
            Ok(v) => v,
            Err(e) => return (false, format!("Failed to get base branch: {e}")),
        };
        match val
            .get("object")
            .and_then(|o| o.get("sha"))
            .and_then(|s| s.as_str())
        {
            Some(s) => s.to_string(),
            None => {
                return (
                    false,
                    "Failed to get base branch: missing object.sha".to_string(),
                )
            }
        }
    };

    // 4. Create a new branch
    let branch_name = format!("add-skill-{skill_name}");
    if let Err(e) = client
        .post(format!("https://api.github.com/repos/{fork_repo}/git/refs"))
        .headers(header_map())
        .timeout(Duration::from_secs(15))
        .json(&serde_json::json!({
            "ref": format!("refs/heads/{branch_name}"),
            "sha": base_sha,
        }))
        .send()
    {
        return (false, format!("Failed to create branch: {e}"));
    }

    // 5. Upload skill files
    for f in walk_files(skill_path) {
        let rel = match f.strip_prefix(skill_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        let upload_path = format!("skills/{skill_name}/{rel}");
        let bytes = match std::fs::read(&f) {
            Ok(b) => b,
            Err(e) => return (false, format!("Failed to upload {rel}: {e}")),
        };
        use base64::Engine;
        let content_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        if let Err(e) = client
            .put(format!(
                "https://api.github.com/repos/{fork_repo}/contents/{upload_path}"
            ))
            .headers(header_map())
            .timeout(Duration::from_secs(15))
            .json(&serde_json::json!({
                "message": format!("Add {skill_name} skill: {rel}"),
                "content": content_b64,
                "branch": branch_name,
            }))
            .send()
        {
            return (false, format!("Failed to upload {rel}: {e}"));
        }
    }

    // 6. Create PR
    let fork_owner = fork_repo.split('/').next().unwrap_or("");
    let resp = client
        .post(format!("https://api.github.com/repos/{target_repo}/pulls"))
        .headers(header_map())
        .timeout(Duration::from_secs(15))
        .json(&serde_json::json!({
            "title": format!("Add skill: {skill_name}"),
            "body": format!(
                "Submitting the `{skill_name}` skill via Hermes Skills Hub.\n\nThis skill was scanned by the Hermes Skills Guard before submission."
            ),
            "head": format!("{fork_owner}:{branch_name}"),
            "base": default_branch,
        }))
        .send();
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if status == 201 {
                let v: serde_json::Value = r.json().unwrap_or(serde_json::Value::Null);
                let pr_url = v
                    .get("html_url")
                    .and_then(|u| u.as_str())
                    .unwrap_or_default();
                (true, format!("PR created: {pr_url}"))
            } else {
                let text = r.text().unwrap_or_default();
                let snippet: String = text.chars().take(200).collect();
                (false, format!("Failed to create PR: {status} {snippet}"))
            }
        }
        Err(e) => (false, format!("Network error creating PR: {e}")),
    }
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let rd = match std::fs::read_dir(&cur) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// snapshot export / import
// ---------------------------------------------------------------------------

/// Mirrors `do_snapshot_export`. Returns the JSON payload (so the CLI can write
/// it or print to stdout when `output_path == "-"`).
pub fn do_snapshot_export(backend: &dyn HubBackend, output_path: &str, c: &dyn Console) {
    let installed = backend.list_installed();
    let tap_list = backend.list_taps();

    let skills: Vec<serde_json::Value> = installed
        .iter()
        .map(|entry| {
            let category = if entry.install_path.contains('/') {
                // str(Path(install_path).parent)
                Path::new(&entry.install_path)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            serde_json::json!({
                "name": entry.name,
                "source": entry.source,
                "identifier": entry.identifier,
                "category": category,
            })
        })
        .collect();

    let taps_json: Vec<serde_json::Value> = tap_list
        .iter()
        .map(|t| {
            serde_json::json!({
                "repo": t.repo,
                "name": t.name,
                "path": t.path,
            })
        })
        .collect();

    let snapshot = serde_json::json!({
        "hermes_version": "0.1.0",
        "exported_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false),
        "skills": skills,
        "taps": taps_json,
    });

    let payload = serde_json::to_string_pretty(&snapshot).unwrap_or_default() + "\n";

    if output_path == "-" {
        print!("{payload}");
    } else {
        let out = PathBuf::from(output_path);
        if std::fs::write(&out, &payload).is_ok() {
            c.print(&format!("[bold green]Snapshot exported:[/] {}", out.display()));
            c.print(&format!(
                "[dim]{} skill(s), {} tap(s)[/]\n",
                installed.len(),
                tap_list.len()
            ));
        }
    }
}

/// Mirrors `do_snapshot_import`.
pub fn do_snapshot_import(backend: &dyn HubBackend, input_path: &str, force: bool, c: &dyn Console) {
    let inp = PathBuf::from(input_path);
    if !inp.exists() {
        c.print(&format!(
            "[bold red]Error:[/] File not found: {}\n",
            inp.display()
        ));
        return;
    }

    let text = match std::fs::read_to_string(&inp) {
        Ok(t) => t,
        Err(_) => {
            c.print(&format!(
                "[bold red]Error:[/] Invalid JSON in {}\n",
                inp.display()
            ));
            return;
        }
    };
    let snapshot: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            c.print(&format!(
                "[bold red]Error:[/] Invalid JSON in {}\n",
                inp.display()
            ));
            return;
        }
    };

    // Restore taps first
    let taps = snapshot
        .get("taps")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if !taps.is_empty() {
        for tap in &taps {
            let repo = tap.get("repo").and_then(|v| v.as_str()).unwrap_or("");
            if !repo.is_empty() {
                let path = tap
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("skills/");
                backend.tap_add(repo, path);
            }
        }
        c.print(&format!("[dim]Restored {} tap(s)[/]", taps.len()));
    }

    let skills = snapshot
        .get("skills")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if skills.is_empty() {
        c.print("[dim]No skills in snapshot to install.[/]\n");
        return;
    }

    c.print(&format!(
        "[bold]Importing {} skill(s) from snapshot...[/]\n",
        skills.len()
    ));
    for entry in &skills {
        let identifier = entry
            .get("identifier")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let category = entry
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if identifier.is_empty() {
            let nm = entry.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            c.print(&format!(
                "[yellow]Skipping entry with no identifier: {nm}[/]"
            ));
            continue;
        }
        let display_name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(identifier);
        c.print(&format!("[bold]--- {display_name} ---[/]"));
        do_install(
            backend,
            identifier,
            InstallOptions {
                category,
                force,
                ..Default::default()
            },
            c,
        );
    }

    c.print("[bold green]Snapshot import complete.[/]\n");
}

// ---------------------------------------------------------------------------
// Slash command entry point (/skills in chat)
// ---------------------------------------------------------------------------

/// Mirrors `handle_skills_slash`. The original always uses skip_confirm=True for
/// install/uninstall/reset (prompt_toolkit context), so a [`NoPrompter`] is fine.
pub fn handle_skills_slash(
    backend: &dyn HubBackend,
    auth: &dyn GitHubAuth,
    cmd: &str,
    c: &dyn Console,
) {
    let mut parts: Vec<String> = cmd.trim().split_whitespace().map(|s| s.to_string()).collect();

    // Strip leading "/skills"
    if !parts.is_empty() && parts[0].to_lowercase() == "/skills" {
        parts.remove(0);
    }

    if parts.is_empty() {
        print_skills_help(c);
        return;
    }

    let action = parts[0].to_lowercase();
    let args: Vec<String> = parts[1..].to_vec();

    match action.as_str() {
        "browse" => {
            let mut page: i64 = 1;
            let mut page_size: i64 = 20;
            let mut source = "all".to_string();
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--page" && i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<i64>() {
                        page = p;
                    }
                    i += 2;
                } else if args[i] == "--size" && i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<i64>() {
                        page_size = p;
                    }
                    i += 2;
                } else if args[i] == "--source" && i + 1 < args.len() {
                    source = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            do_browse(backend, page, page_size, &source, c);
        }
        "search" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills search <query> [--source skills-sh|well-known|github|official] [--limit N]\n");
                return;
            }
            let mut source = "all".to_string();
            let mut limit: usize = 10;
            let mut query_parts: Vec<String> = Vec::new();
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--source" && i + 1 < args.len() {
                    source = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--limit" && i + 1 < args.len() {
                    if let Ok(l) = args[i + 1].parse::<usize>() {
                        limit = l;
                    }
                    i += 2;
                } else {
                    query_parts.push(args[i].clone());
                    i += 1;
                }
            }
            do_search(backend, &query_parts.join(" "), &source, limit, c);
        }
        "install" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills install <identifier-or-url> [--name <name>] [--category <cat>] [--force] [--now]\n");
                return;
            }
            let identifier = args[0].clone();
            let mut category = String::new();
            let mut name_override = String::new();
            let force = args.iter().any(|a| a == "--force");
            let invalidate_cache = args.iter().any(|a| a == "--now");
            for (i, a) in args.iter().enumerate() {
                if a == "--category" && i + 1 < args.len() {
                    category = args[i + 1].clone();
                } else if a == "--name" && i + 1 < args.len() {
                    name_override = args[i + 1].clone();
                }
            }
            do_install(
                backend,
                &identifier,
                InstallOptions {
                    category,
                    force,
                    skip_confirm: true,
                    invalidate_cache,
                    name_override,
                    prompter: &NO_PROMPTER,
                },
                c,
            );
        }
        "inspect" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills inspect <identifier>\n");
                return;
            }
            do_inspect(backend, &args[0], c);
        }
        "list" => {
            let mut source_filter = "all".to_string();
            let enabled_only =
                args.iter().any(|a| a == "--enabled-only") || args.iter().any(|a| a == "--enabled");
            if let Some(idx) = args.iter().position(|a| a == "--source") {
                if idx + 1 < args.len() {
                    source_filter = args[idx + 1].clone();
                }
            }
            do_list(backend, &source_filter, enabled_only, c);
        }
        "check" => {
            let name = args.first().map(|s| s.as_str());
            do_check(backend, name, c);
        }
        "update" => {
            let name = args.first().map(|s| s.as_str());
            do_update(backend, name, c);
        }
        "audit" => {
            let name = args.first().map(|s| s.as_str());
            do_audit(backend, name, c);
        }
        "uninstall" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills uninstall <name> [--now]\n");
                return;
            }
            let invalidate_cache = args.iter().any(|a| a == "--now");
            do_uninstall(backend, &args[0], c, &NO_PROMPTER, true, invalidate_cache);
        }
        "reset" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills reset <name> [--restore] [--now]\n");
                c.print("[dim]Clears the bundled-skills manifest entry so future updates stop marking it as user-modified.[/]");
                c.print("[dim]Pass --restore to also replace the current copy with the bundled version.[/]\n");
                return;
            }
            let name = args[0].clone();
            let restore = args.iter().any(|a| a == "--restore");
            let invalidate_cache = args.iter().any(|a| a == "--now");
            do_reset(backend, &name, restore, c, &NO_PROMPTER, true, invalidate_cache);
        }
        "publish" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills publish <skill-path> [--to github] [--repo owner/repo]\n");
                return;
            }
            let skill_path = args[0].clone();
            let mut target = "github".to_string();
            let mut repo = String::new();
            for (i, a) in args.iter().enumerate() {
                if a == "--to" && i + 1 < args.len() {
                    target = args[i + 1].clone();
                }
                if a == "--repo" && i + 1 < args.len() {
                    repo = args[i + 1].clone();
                }
            }
            do_publish(backend, auth, &skill_path, &target, &repo, c);
        }
        "snapshot" => {
            if args.is_empty() {
                c.print("[bold red]Usage:[/] /skills snapshot export <file> | /skills snapshot import <file>\n");
                return;
            }
            let snap_action = args[0].as_str();
            if snap_action == "export" && args.len() > 1 {
                do_snapshot_export(backend, &args[1], c);
            } else if snap_action == "import" && args.len() > 1 {
                let force = args.iter().any(|a| a == "--force");
                do_snapshot_import(backend, &args[1], force, c);
            } else {
                c.print("[bold red]Usage:[/] /skills snapshot export <file> | /skills snapshot import <file>\n");
            }
        }
        "tap" => {
            if args.is_empty() {
                do_tap(backend, "list", "", c);
                return;
            }
            let tap_action = args[0].as_str();
            let repo = args.get(1).cloned().unwrap_or_default();
            do_tap(backend, tap_action, &repo, c);
        }
        "help" | "--help" | "-h" => {
            print_skills_help(c);
        }
        other => {
            c.print(&format!("[bold red]Unknown action:[/] {other}"));
            print_skills_help(c);
        }
    }
}

/// Mirrors `_print_skills_help`.
pub fn print_skills_help(c: &dyn Console) {
    c.print(&panel(
        "[bold]Skills Hub Commands:[/]\n\n  [cyan]browse[/] [--source official]   Browse all available skills (paginated)\n  [cyan]search[/] <query>              Search registries for skills\n  [cyan]install[/] <identifier>        Install a skill (with security scan)\n  [cyan]inspect[/] <identifier>        Preview a skill without installing\n  [cyan]list[/] [--source hub|builtin|local] [--enabled-only]\n       List installed skills; --enabled-only filters to the active profile's live set\n  [cyan]check[/] [name]                Check hub skills for upstream updates\n  [cyan]update[/] [name]               Update hub skills with upstream changes\n  [cyan]audit[/] [name]                Re-scan hub skills for security\n  [cyan]uninstall[/] <name>            Remove a hub-installed skill\n  [cyan]reset[/] <name> [--restore]    Reset bundled-skill tracking (fix 'user-modified' flag)\n  [cyan]publish[/] <path> --repo <r>   Publish a skill to GitHub via PR\n  [cyan]snapshot[/] export|import      Export/import skill configurations\n  [cyan]tap[/] list|add|remove         Manage skill sources\n",
        "/skills",
        "",
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    // ---- pure helpers ----

    #[test]
    fn valid_skill_names() {
        assert!(is_valid_installed_skill_name("my-skill"));
        assert!(is_valid_installed_skill_name("foo_bar"));
        assert!(is_valid_installed_skill_name("a"));
        assert!(!is_valid_installed_skill_name(""));
        assert!(!is_valid_installed_skill_name("skill"));
        assert!(!is_valid_installed_skill_name("README"));
        assert!(!is_valid_installed_skill_name("index"));
        assert!(!is_valid_installed_skill_name("unnamed-skill"));
        assert!(!is_valid_installed_skill_name("1bad"));
        assert!(!is_valid_installed_skill_name("Bad Name"));
        // case folded then validated; uppercase folds to lowercase identifier
        assert!(is_valid_installed_skill_name("My-Skill"));
    }

    #[test]
    fn truncate_behavior() {
        assert_eq!(truncate_with_ellipsis("short", 60), "short");
        let long = "a".repeat(70);
        let t = truncate_with_ellipsis(&long, 60);
        assert_eq!(t.len(), 63);
        assert!(t.ends_with("..."));
    }

    #[test]
    fn derive_category() {
        assert_eq!(derive_category_from_install_path("foo"), "");
        assert_eq!(derive_category_from_install_path("cat/foo"), "cat");
        assert_eq!(derive_category_from_install_path("a/b/foo"), "a/b");
    }

    #[test]
    fn extra_metadata_lines_order_and_security() {
        let mut extra: BTreeMap<String, ExtraValue> = BTreeMap::new();
        extra.insert("repo_url".into(), ExtraValue::Str("https://x/y".into()));
        extra.insert("installs".into(), ExtraValue::Int(0));
        let mut sec = BTreeMap::new();
        sec.insert("zeta".into(), "pass".into());
        sec.insert("alpha".into(), "fail".into());
        extra.insert("security_audits".into(), ExtraValue::Map(sec));
        let lines = format_extra_metadata_lines(&extra);
        assert_eq!(lines[0], "[bold]Repo:[/] https://x/y");
        // installs=0 is shown (not None)
        assert!(lines.iter().any(|l| l == "[bold]Installs:[/] 0"));
        // security sorted by key
        assert!(lines
            .iter()
            .any(|l| l == "[bold]Security:[/] alpha=fail, zeta=pass"));
    }

    #[test]
    fn parse_frontmatter_extracts_name_desc() {
        let md = "---\nname: demo\ndescription: A demo skill\n---\nbody text\n";
        let fm = parse_frontmatter(md);
        assert_eq!(fm.get("name").map(String::as_str), Some("demo"));
        assert_eq!(
            fm.get("description").map(String::as_str),
            Some("A demo skill")
        );
    }

    #[test]
    fn parse_frontmatter_no_frontmatter() {
        assert!(parse_frontmatter("no frontmatter here").is_empty());
        assert!(parse_frontmatter("---\nname: x\nno closing").is_empty());
    }

    #[test]
    fn dedupe_prefers_higher_trust() {
        let results = vec![
            SkillResult {
                name: "dup".into(),
                description: "".into(),
                source: "github".into(),
                trust_level: "community".into(),
                identifier: "a/dup".into(),
            },
            SkillResult {
                name: "dup".into(),
                description: "".into(),
                source: "official".into(),
                trust_level: "builtin".into(),
                identifier: "official/dup".into(),
            },
        ];
        let out = dedupe_and_sort(results);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].identifier, "official/dup");
    }

    // ---- mock backend ----

    #[derive(Default)]
    struct MockSource {
        results: Vec<SkillResult>,
        meta: Option<SkillMeta>,
        bundle: Option<SkillBundle>,
    }
    impl SkillSource for MockSource {
        fn source_id(&self) -> String {
            "mock".into()
        }
        fn search(&self, _q: &str, _l: usize) -> Vec<SkillResult> {
            self.results.clone()
        }
        fn inspect(&self, _id: &str) -> Option<SkillMeta> {
            self.meta.clone()
        }
        fn fetch(&self, _id: &str) -> Option<SkillBundle> {
            self.bundle.clone()
        }
    }

    #[derive(Default)]
    struct MockBackend {
        search_results: Vec<SkillResult>,
        installed: RefCell<Vec<InstalledEntry>>,
        taps: RefCell<Vec<TapEntry>>,
        meta: Option<SkillMeta>,
        bundle: Option<SkillBundle>,
        cache_cleared: RefCell<bool>,
    }

    impl HubBackend for MockBackend {
        fn sources(&self) -> Vec<Box<dyn SkillSource>> {
            vec![Box::new(MockSource {
                results: self.search_results.clone(),
                meta: self.meta.clone(),
                bundle: self.bundle.clone(),
            })]
        }
        fn unified_search(&self, _q: &str, _sf: &str, _l: usize) -> Vec<SkillResult> {
            self.search_results.clone()
        }
        fn parallel_search_sources(&self, _q: &str, _sf: &str, _t: u64) -> ParallelSearchOutcome {
            ParallelSearchOutcome {
                results: self.search_results.clone(),
                source_counts: BTreeMap::new(),
                timed_out: Vec::new(),
            }
        }
        fn skills_dir(&self) -> PathBuf {
            std::env::temp_dir().join("hermes-test-skills-nonexistent")
        }
        fn display_hermes_home(&self) -> String {
            "/tmp/hermes".into()
        }
        fn list_installed(&self) -> Vec<InstalledEntry> {
            self.installed.borrow().clone()
        }
        fn list_taps(&self) -> Vec<TapEntry> {
            self.taps.borrow().clone()
        }
        fn tap_add(&self, repo: &str, path: &str) -> bool {
            if self.taps.borrow().iter().any(|t| t.repo == repo) {
                return false;
            }
            self.taps.borrow_mut().push(TapEntry {
                repo: repo.into(),
                name: String::new(),
                path: path.into(),
            });
            true
        }
        fn tap_remove(&self, repo: &str) -> bool {
            let before = self.taps.borrow().len();
            self.taps.borrow_mut().retain(|t| t.repo != repo);
            self.taps.borrow().len() != before
        }
        fn scan_skill(&self, _p: &Path, _s: &str) -> ScanResult {
            ScanResult {
                verdict: "safe".into(),
                findings_count: 0,
            }
        }
        fn should_allow_install(&self, _r: &ScanResult, _f: bool) -> (bool, String) {
            (true, "ok".into())
        }
        fn format_scan_report(&self, r: &ScanResult) -> String {
            format!("[scan {}]", r.verdict)
        }
        fn quarantine_bundle(&self, _b: &SkillBundle) -> Result<PathBuf, String> {
            Ok(PathBuf::from("/tmp/a/b/c/quar"))
        }
        fn install_from_quarantine(
            &self,
            _q: &Path,
            name: &str,
            category: &str,
            _b: &SkillBundle,
            _r: &ScanResult,
        ) -> Result<PathBuf, String> {
            let mut p = self.skills_dir();
            if !category.is_empty() {
                p = p.join(category);
            }
            Ok(p.join(name))
        }
        fn append_audit_log(&self, _a: &str, _n: &str, _s: &str, _t: &str, _v: &str, _d: &str) {}
        fn uninstall_skill(&self, _name: &str) -> (bool, String) {
            (true, "Uninstalled".into())
        }
        fn reset_bundled_skill(&self, _name: &str, _restore: bool) -> ResetResult {
            ResetResult {
                ok: true,
                message: "reset".into(),
                ..Default::default()
            }
        }
        fn check_for_skill_updates(&self, _name: Option<&str>) -> Vec<UpdateEntry> {
            Vec::new()
        }
        fn clear_skills_system_prompt_cache(&self) {
            *self.cache_cleared.borrow_mut() = true;
        }
    }

    #[test]
    fn search_empty_prints_no_results() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        do_search(&backend, "kube", "all", 10, &c);
        assert!(c.contains("No skills found matching your query."));
    }

    #[test]
    fn search_renders_results() {
        let backend = MockBackend {
            search_results: vec![SkillResult {
                name: "pptx".into(),
                description: "make slides".into(),
                source: "official".into(),
                trust_level: "builtin".into(),
                identifier: "official/pptx".into(),
            }],
            ..Default::default()
        };
        let c = CaptureConsole::default();
        do_search(&backend, "ppt", "all", 10, &c);
        assert!(c.contains("pptx"));
        assert!(c.contains("official/pptx"));
        // official => label "official"
        assert!(c.joined().contains("]official[/]"));
    }

    #[test]
    fn resolve_short_name_single_match() {
        let backend = MockBackend {
            search_results: vec![SkillResult {
                name: "pptx".into(),
                description: "".into(),
                source: "official".into(),
                trust_level: "builtin".into(),
                identifier: "official/pptx".into(),
            }],
            ..Default::default()
        };
        let c = CaptureConsole::default();
        let id = resolve_short_name("pptx", &backend, &c);
        assert_eq!(id, "official/pptx");
    }

    #[test]
    fn resolve_short_name_ambiguous() {
        let backend = MockBackend {
            search_results: vec![
                SkillResult {
                    name: "x".into(),
                    description: "".into(),
                    source: "github".into(),
                    trust_level: "community".into(),
                    identifier: "a/x".into(),
                },
                SkillResult {
                    name: "x".into(),
                    description: "".into(),
                    source: "official".into(),
                    trust_level: "builtin".into(),
                    identifier: "official/x".into(),
                },
            ],
            ..Default::default()
        };
        let c = CaptureConsole::default();
        let id = resolve_short_name("x", &backend, &c);
        assert_eq!(id, "");
        assert!(c.contains("Multiple skills named 'x' found:"));
    }

    #[test]
    fn install_url_no_name_skip_confirm_errors() {
        let mut bundle = SkillBundle {
            source: "url".into(),
            ..Default::default()
        };
        bundle
            .metadata
            .insert("url".into(), ExtraValue::Str("https://e/SKILL.md".into()));
        let backend = MockBackend {
            bundle: Some(bundle),
            ..Default::default()
        };
        let c = CaptureConsole::default();
        do_install(
            &backend,
            "https://e/SKILL.md",
            InstallOptions {
                skip_confirm: true,
                ..Default::default()
            },
            &c,
        );
        assert!(c.contains("Cannot install from URL: https://e/SKILL.md"));
    }

    #[test]
    fn install_url_with_name_override_proceeds() {
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), b"---\nname: x\n---\nbody".to_vec());
        let bundle = SkillBundle {
            source: "url".into(),
            files,
            ..Default::default()
        };
        let backend = MockBackend {
            bundle: Some(bundle),
            ..Default::default()
        };
        let c = CaptureConsole::default();
        do_install(
            &backend,
            "https://e/SKILL.md",
            InstallOptions {
                skip_confirm: true,
                name_override: "my-skill".into(),
                ..Default::default()
            },
            &c,
        );
        assert!(c.contains("[bold green]Installed:[/] my-skill"));
        assert!(*backend.cache_cleared.borrow());
    }

    #[test]
    fn install_not_found_reports_error() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        do_install(
            &backend,
            "owner/missing",
            InstallOptions {
                skip_confirm: true,
                ..Default::default()
            },
            &c,
        );
        assert!(c.contains("Could not fetch 'owner/missing' from any source."));
    }

    #[test]
    fn tap_list_empty() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        do_tap(&backend, "list", "", &c);
        assert!(c.contains("No custom taps configured. Using default sources only."));
    }

    #[test]
    fn tap_add_and_remove() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        do_tap(&backend, "add", "owner/repo", &c);
        assert!(c.contains("[bold green]Added tap:[/] owner/repo"));
        let c2 = CaptureConsole::default();
        do_tap(&backend, "add", "owner/repo", &c2);
        assert!(c2.contains("[yellow]Tap already exists:[/] owner/repo"));
        let c3 = CaptureConsole::default();
        do_tap(&backend, "remove", "owner/repo", &c3);
        assert!(c3.contains("[bold green]Removed tap:[/] owner/repo"));
    }

    #[test]
    fn uninstall_confirmed() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        let p = ScriptedPrompter::new(vec![Some("y".into())]);
        do_uninstall(&backend, "foo", &c, &p, false, true);
        assert!(c.contains("[bold green]Uninstalled[/]"));
        assert!(*backend.cache_cleared.borrow());
    }

    #[test]
    fn uninstall_cancelled_on_eof() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        let p = NoPrompter;
        do_uninstall(&backend, "foo", &c, &p, false, true);
        assert!(c.contains("[dim]Cancelled.[/]\n"));
    }

    #[test]
    fn browse_skills_programmatic_paginates() {
        let mut results = Vec::new();
        for i in 0..25 {
            results.push(SkillResult {
                name: format!("skill{i:02}"),
                description: "".into(),
                source: "github".into(),
                trust_level: "community".into(),
                identifier: format!("a/skill{i:02}"),
            });
        }
        let backend = MockBackend {
            search_results: results,
            ..Default::default()
        };
        let page = browse_skills(&backend, 1, 20, "all");
        assert_eq!(page.total, 25);
        assert_eq!(page.total_pages, 2);
        assert_eq!(page.items.len(), 20);
    }

    #[test]
    fn snapshot_import_missing_file() {
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        do_snapshot_import(&backend, "/nonexistent/snapshot.json", false, &c);
        assert!(c.contains("File not found:"));
    }

    #[test]
    fn slash_unknown_action_shows_help() {
        struct NoAuth;
        impl GitHubAuth for NoAuth {
            fn is_authenticated(&self) -> bool {
                false
            }
            fn headers(&self) -> Vec<(String, String)> {
                Vec::new()
            }
        }
        let backend = MockBackend::default();
        let c = CaptureConsole::default();
        handle_skills_slash(&backend, &NoAuth, "/skills frobnicate", &c);
        assert!(c.contains("[bold red]Unknown action:[/] frobnicate"));
        assert!(c.contains("Skills Hub Commands:"));
    }

    #[test]
    fn slash_search_parses_flags() {
        struct NoAuth;
        impl GitHubAuth for NoAuth {
            fn is_authenticated(&self) -> bool {
                false
            }
            fn headers(&self) -> Vec<(String, String)> {
                Vec::new()
            }
        }
        let backend = MockBackend {
            search_results: vec![SkillResult {
                name: "k8s".into(),
                description: "kubernetes".into(),
                source: "github".into(),
                trust_level: "community".into(),
                identifier: "a/k8s".into(),
            }],
            ..Default::default()
        };
        let c = CaptureConsole::default();
        handle_skills_slash(&backend, &NoAuth, "/skills search kube --limit 5", &c);
        assert!(c.contains("Searching for:"));
        assert!(c.joined().contains("kube"));
    }
}
