//! hermes fallback — manage the fallback provider chain.
//!
//! Faithful native Rust port of `hermes_cli/fallback_cmd.py`.
//!
//! Fallback providers are tried in order when the primary model fails with
//! rate-limit, overload, or connection errors. See:
//! <https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers>
//!
//! Subcommands:
//!   * `hermes fallback [list]`  — show the current fallback chain (default).
//!   * `hermes fallback add`     — pick provider + model via the same picker as
//!                                 `hermes model`, then append the selection.
//!   * `hermes fallback remove`  — pick an entry to delete from the chain.
//!   * `hermes fallback clear`   — remove all fallback entries.
//!
//! Storage: `fallback_providers` in `~/.hermes/config.yaml` (top-level, list of
//! `{provider, model, base_url?, api_mode?}` mappings). The legacy single-dict
//! `fallback_model` format is migrated to the new list format on first add.
//!
//! Integration notes
//! ------------------
//! Config load/save is delegated to `hermes_core::cli_config` (`serde_yaml::Value`).
//! The auth-store snapshot/restore is delegated to `crate::cli_auth`
//! (`serde_json::Value`). Both are looked up lazily through the [`FallbackEnv`]
//! trait so the pure logic (chain read/write, formatting, extraction, dispatch)
//! is fully unit-testable without touching disk, a TTY, or a real picker.

use std::io::{self, BufRead, Write};

use serde_yaml::{Mapping, Sequence, Value as Yaml};

// ---------------------------------------------------------------------------
// Plain Rust model of a fallback entry
// ---------------------------------------------------------------------------

/// A normalized `{provider, model, base_url?, api_mode?}` fallback entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FallbackEntry {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
    pub api_mode: Option<String>,
}

impl FallbackEntry {
    /// Convert into a YAML mapping suitable for storing under
    /// `fallback_providers`. Only present optional keys are emitted, matching
    /// the Python dict construction order (provider, model, base_url, api_mode).
    pub fn to_yaml(&self) -> Yaml {
        let mut m = Mapping::new();
        m.insert(Yaml::String("provider".into()), Yaml::String(self.provider.clone()));
        m.insert(Yaml::String("model".into()), Yaml::String(self.model.clone()));
        if let Some(b) = &self.base_url {
            m.insert(Yaml::String("base_url".into()), Yaml::String(b.clone()));
        }
        if let Some(a) = &self.api_mode {
            m.insert(Yaml::String("api_mode".into()), Yaml::String(a.clone()));
        }
        Yaml::Mapping(m)
    }
}

// ---------------------------------------------------------------------------
// Small YAML helpers
// ---------------------------------------------------------------------------

fn map_get<'a>(m: &'a Yaml, key: &str) -> Option<&'a Yaml> {
    m.as_mapping().and_then(|map| map.get(&Yaml::String(key.to_string())))
}

/// `(value or "").strip()` — return the trimmed string, or empty when the
/// value is absent / not a non-empty string.
fn str_or_empty(v: Option<&Yaml>) -> String {
    match v {
        Some(Yaml::String(s)) => s.trim().to_string(),
        _ => String::new(),
    }
}

fn map_remove(m: &mut Mapping, key: &str) {
    m.remove(&Yaml::String(key.to_string()));
}

// ---------------------------------------------------------------------------
// Helpers (port of the `_*` private functions)
// ---------------------------------------------------------------------------

/// Return the normalized fallback chain.
///
/// Accepts both the new list format (`fallback_providers`) and the legacy
/// single-dict / list `fallback_model` format. The returned vec is a fresh
/// owned copy — callers can mutate freely without touching the config.
pub fn read_chain(config: &Yaml) -> Vec<FallbackEntry> {
    // Helper: keep only mappings that have non-empty provider + model.
    fn collect(seq: &Sequence) -> Vec<FallbackEntry> {
        seq.iter()
            .filter_map(|e| {
                let provider = e.as_mapping().and_then(|m| m.get(&Yaml::String("provider".into())));
                let model = e.as_mapping().and_then(|m| m.get(&Yaml::String("model".into())));
                let provider = match provider {
                    Some(Yaml::String(s)) if !s.is_empty() => s.clone(),
                    _ => return None,
                };
                let model = match model {
                    Some(Yaml::String(s)) if !s.is_empty() => s.clone(),
                    _ => return None,
                };
                Some(FallbackEntry {
                    provider,
                    model,
                    base_url: opt_str(e, "base_url"),
                    api_mode: opt_str(e, "api_mode"),
                })
            })
            .collect()
    }

    fn opt_str(e: &Yaml, key: &str) -> Option<String> {
        match map_get(e, key) {
            Some(Yaml::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    }

    if let Some(Yaml::Sequence(seq)) = map_get(config, "fallback_providers") {
        let result = collect(seq);
        if !result.is_empty() {
            return result;
        }
    }

    match map_get(config, "fallback_model") {
        Some(Yaml::Mapping(_)) => {
            // Single-dict legacy form: wrap in a one-element sequence and reuse
            // the same provider/model filtering.
            let legacy = map_get(config, "fallback_model").cloned().unwrap();
            let seq = vec![legacy];
            collect(&seq)
        }
        Some(Yaml::Sequence(seq)) => collect(seq),
        _ => Vec::new(),
    }
}

/// Persist the chain to `fallback_providers` and drop the legacy key so there
/// is only one source of truth.
pub fn write_chain(config: &mut Yaml, chain: &[FallbackEntry]) {
    if config.as_mapping().is_none() {
        *config = Yaml::Mapping(Mapping::new());
    }
    let map = config.as_mapping_mut().expect("mapping");
    let seq: Sequence = chain.iter().map(FallbackEntry::to_yaml).collect();
    map.insert(Yaml::String("fallback_providers".into()), Yaml::Sequence(seq));
    map_remove(map, "fallback_model");
}

/// One-line human-readable rendering of a fallback entry.
pub fn format_entry(entry: &FallbackEntry) -> String {
    let provider = if entry.provider.is_empty() { "?" } else { &entry.provider };
    let model = if entry.model.is_empty() { "?" } else { &entry.model };
    let suffix = match &entry.base_url {
        Some(b) if !b.is_empty() => format!("  [{}]", b),
        _ => String::new(),
    };
    format!("{}  (via {}){}", model, provider, suffix)
}

/// Pull the `{provider, model, base_url?, api_mode?}` entry from a
/// `config["model"]` snapshot. The picker writes the selection to
/// `model.default`; fall back to `model.model`.
pub fn extract_fallback_from_model_cfg(model_cfg: Option<&Yaml>) -> Option<FallbackEntry> {
    let model_cfg = model_cfg?;
    if model_cfg.as_mapping().is_none() {
        return None;
    }
    let provider = str_or_empty(map_get(model_cfg, "provider"));
    let mut model = str_or_empty(map_get(model_cfg, "default"));
    if model.is_empty() {
        model = str_or_empty(map_get(model_cfg, "model"));
    }
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    let base_url = {
        let b = str_or_empty(map_get(model_cfg, "base_url"));
        if b.is_empty() { None } else { Some(b) }
    };
    let api_mode = {
        let a = str_or_empty(map_get(model_cfg, "api_mode"));
        if a.is_empty() { None } else { Some(a) }
    };
    Some(FallbackEntry { provider, model, base_url, api_mode })
}

/// One-line description of the primary model for display purposes.
pub fn describe_primary(config: &Yaml) -> Option<String> {
    match map_get(config, "model") {
        Some(m) if m.as_mapping().is_some() => {
            let provider = {
                let p = str_or_empty(map_get(m, "provider"));
                if p.is_empty() { "?".to_string() } else { p }
            };
            let model = {
                let mut s = str_or_empty(map_get(m, "default"));
                if s.is_empty() {
                    s = str_or_empty(map_get(m, "model"));
                }
                if s.is_empty() { "?".to_string() } else { s }
            };
            Some(format!("{}  (via {})", model, provider))
        }
        Some(Yaml::String(s)) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    }
}

/// Pluralize "entry"/"entries" exactly as the Python source.
fn entries_word(n: usize) -> &'static str {
    if n == 1 { "entry" } else { "entries" }
}

// ---------------------------------------------------------------------------
// Environment abstraction
// ---------------------------------------------------------------------------

/// Outcome of running the provider/model picker for `fallback add`.
#[derive(Debug)]
pub enum PickerOutcome {
    /// The picker completed; `model_after` is the post-picker `config["model"]`
    /// snapshot (may be `None` if the key was removed).
    Completed { model_after: Option<Yaml> },
    /// The picker cancelled / bailed without finishing.
    Cancelled,
    /// A provider auth flow called `sys.exit(...)`; restore state and propagate.
    Exit(i32),
}

/// Side-effecting dependencies of the fallback commands. Production code wires
/// these to the real config / auth / picker helpers; tests provide fakes.
pub trait FallbackEnv {
    fn load_config(&self) -> Yaml;
    fn save_config(&mut self, config: &Yaml) -> Result<(), String>;

    /// Snapshot the auth store's `active_provider`, or `Null`/sentinel when
    /// unavailable. Mirrors `_snapshot_auth_active_provider`.
    fn snapshot_active_provider(&self) -> Yaml {
        Yaml::Null
    }
    /// Best-effort restore of a previously snapshotted `active_provider`.
    fn restore_active_provider(&mut self, _value: &Yaml) {}

    /// Require an interactive TTY for the given action; returns an error message
    /// (to be surfaced and treated as fatal) when not interactive.
    fn require_tty(&self, _action: &str) -> Result<(), String> {
        Ok(())
    }

    /// Launch the provider+model picker (same one used by `hermes model`). The
    /// picker is expected to mutate `config["model"]` on disk; the returned
    /// outcome reports what happened.
    fn run_picker(&mut self) -> PickerOutcome;
}

// ---------------------------------------------------------------------------
// `_restore_model_cfg`
// ---------------------------------------------------------------------------

/// Restore `config["model"]` to a previously-captured snapshot.
pub fn restore_model_cfg<E: FallbackEnv>(env: &mut E, model_before: Option<&Yaml>) {
    let mut cfg = env.load_config();
    if cfg.as_mapping().is_none() {
        cfg = Yaml::Mapping(Mapping::new());
    }
    let map = cfg.as_mapping_mut().expect("mapping");
    match model_before {
        None => {
            map_remove(map, "model");
        }
        Some(v) => {
            map.insert(Yaml::String("model".into()), v.clone());
        }
    }
    let _ = env.save_config(&cfg);
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

/// Print the current fallback chain.
pub fn cmd_fallback_list<E: FallbackEnv, W: Write>(env: &E, out: &mut W) {
    let config = env.load_config();
    let chain = read_chain(&config);

    let _ = writeln!(out);
    if chain.is_empty() {
        let _ = writeln!(out, "  No fallback providers configured.");
        let _ = writeln!(out);
        let _ = writeln!(out, "  Add one with:  hermes fallback add");
        let _ = writeln!(out);
        return;
    }

    if let Some(primary) = describe_primary(&config) {
        let _ = writeln!(out, "  Primary:   {}", primary);
        let _ = writeln!(out);
    }
    let _ = writeln!(out, "  Fallback chain ({} {}):", chain.len(), entries_word(chain.len()));
    for (i, entry) in chain.iter().enumerate() {
        let _ = writeln!(out, "    {}. {}", i + 1, format_entry(entry));
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "  Tried in order when the primary fails (rate-limit, 5xx, connection errors).");
    let _ = writeln!(
        out,
        "  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers"
    );
    let _ = writeln!(out);
}

/// Result of `cmd_fallback_add` / the dispatcher, so the caller can map an
/// in-band `SystemExit` to a process exit code.
#[derive(Debug, PartialEq, Eq)]
pub enum CmdOutcome {
    Ok,
    Exit(i32),
}

/// Launch the same picker as `hermes model`, then append the selection.
pub fn cmd_fallback_add<E: FallbackEnv, W: Write>(env: &mut E, out: &mut W) -> CmdOutcome {
    if let Err(msg) = env.require_tty("fallback add") {
        let _ = writeln!(out, "{}", msg);
        return CmdOutcome::Exit(1);
    }

    // Snapshot BEFORE the picker runs so we can distinguish "user actually
    // picked something" from "user cancelled" by comparing before/after.
    let before_cfg = env.load_config();
    let model_before = map_get(&before_cfg, "model").cloned();
    let active_provider_before = env.snapshot_active_provider();

    let _ = writeln!(out);
    let _ = writeln!(out, "  Adding a fallback provider.  The picker below is the same one used by");
    let _ = writeln!(out, "  `hermes model` — select the provider + model you want as a fallback.");
    let _ = writeln!(out);

    let model_after = match env.run_picker() {
        PickerOutcome::Exit(code) => {
            // Some provider flows exit on auth failure — restore state and
            // propagate the exit.
            restore_model_cfg(env, model_before.as_ref());
            env.restore_active_provider(&active_provider_before);
            return CmdOutcome::Exit(code);
        }
        PickerOutcome::Cancelled => None,
        PickerOutcome::Completed { model_after } => model_after,
    };

    let new_entry = extract_fallback_from_model_cfg(model_after.as_ref());
    let new_entry = match new_entry {
        None => {
            // Picker didn't complete (user cancelled or flow bailed).
            restore_model_cfg(env, model_before.as_ref());
            env.restore_active_provider(&active_provider_before);
            let _ = writeln!(out);
            let _ = writeln!(out, "  No fallback added.");
            return CmdOutcome::Ok;
        }
        Some(e) => e,
    };

    // Picker picked the same thing that's already the primary → nothing to add.
    let primary_entry = extract_fallback_from_model_cfg(model_before.as_ref());
    if let Some(p) = &primary_entry {
        if p.provider == new_entry.provider && p.model == new_entry.model {
            restore_model_cfg(env, model_before.as_ref());
            env.restore_active_provider(&active_provider_before);
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "  Selected model matches the current primary ({}).",
                format_entry(&new_entry)
            );
            let _ = writeln!(out, "  A provider cannot be a fallback for itself — no change.");
            return CmdOutcome::Ok;
        }
    }

    // Re-load with primary restored, then append the new entry. We deliberately
    // re-load (rather than mutating after_cfg) because the picker may have
    // touched other top-level keys we want to keep.
    restore_model_cfg(env, model_before.as_ref());
    env.restore_active_provider(&active_provider_before);

    let mut final_cfg = env.load_config();
    let mut chain = read_chain(&final_cfg);

    // Reject exact-duplicate fallback entries.
    for existing in &chain {
        if existing.provider == new_entry.provider && existing.model == new_entry.model {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "  {} is already in the fallback chain — skipped.",
                format_entry(&new_entry)
            );
            return CmdOutcome::Ok;
        }
    }

    chain.push(new_entry.clone());
    write_chain(&mut final_cfg, &chain);
    let _ = env.save_config(&final_cfg);

    let _ = writeln!(out);
    let _ = writeln!(out, "  Added fallback: {}", format_entry(&new_entry));
    let _ = writeln!(out, "  Chain is now {} {} long.", chain.len(), entries_word(chain.len()));
    let _ = writeln!(out);
    let _ = writeln!(out, "  Run `hermes fallback list` to view, or `hermes fallback remove` to delete.");
    CmdOutcome::Ok
}

/// A chooser that returns a selected index from a list of labelled choices, or
/// `None` when cancelled. Mirrors the curses / numbered-pick fallback in Python.
pub trait Chooser {
    fn choose(&mut self, question: &str, choices: &[String], default: usize) -> Option<usize>;
}

/// Pick an entry from the chain and remove it.
pub fn cmd_fallback_remove<E: FallbackEnv, C: Chooser, W: Write>(
    env: &mut E,
    chooser: &mut C,
    out: &mut W,
) {
    let mut config = env.load_config();
    let mut chain = read_chain(&config);

    if chain.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "  No fallback providers configured — nothing to remove.");
        let _ = writeln!(out);
        return;
    }

    let mut choices: Vec<String> = chain.iter().map(format_entry).collect();
    choices.push("Cancel".to_string());

    let idx = chooser.choose("Select a fallback to remove:", &choices, 0);

    // `idx is None or idx < 0 or idx >= len(chain)` → the "Cancel" sentinel
    // (last index == chain.len()) and out-of-range both cancel.
    let idx = match idx {
        Some(i) if i < chain.len() => i,
        _ => {
            let _ = writeln!(out);
            let _ = writeln!(out, "  Cancelled — no change.");
            return;
        }
    };

    let removed = chain.remove(idx);
    write_chain(&mut config, &chain);
    let _ = env.save_config(&config);

    let _ = writeln!(out);
    let _ = writeln!(out, "  Removed fallback: {}", format_entry(&removed));
    if !chain.is_empty() {
        let _ = writeln!(out, "  Chain is now {} {} long.", chain.len(), entries_word(chain.len()));
    } else {
        let _ = writeln!(out, "  Fallback chain is now empty.");
    }
    let _ = writeln!(out);
}

/// Remove all fallback entries (with confirmation). `confirm` returns the raw
/// user response line, or `None` on KeyboardInterrupt/EOF.
pub fn cmd_fallback_clear<E: FallbackEnv, W: Write>(
    env: &mut E,
    out: &mut W,
    confirm: impl FnOnce(&str) -> Option<String>,
) {
    let mut config = env.load_config();
    let chain = read_chain(&config);

    if chain.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "  No fallback providers configured — nothing to clear.");
        let _ = writeln!(out);
        return;
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "  Current fallback chain ({} {}):", chain.len(), entries_word(chain.len()));
    for (i, entry) in chain.iter().enumerate() {
        let _ = writeln!(out, "    {}. {}", i + 1, format_entry(entry));
    }
    let _ = writeln!(out);

    let resp = match confirm("  Clear all entries? [y/N]: ") {
        None => {
            // KeyboardInterrupt / EOF
            let _ = writeln!(out);
            let _ = writeln!(out, "  Cancelled.");
            return;
        }
        Some(s) => s.trim().to_lowercase(),
    };

    if resp != "y" && resp != "yes" {
        let _ = writeln!(out, "  Cancelled — no change.");
        return;
    }

    write_chain(&mut config, &[]);
    let _ = env.save_config(&config);
    let _ = writeln!(out);
    let _ = writeln!(out, "  Fallback chain cleared.");
    let _ = writeln!(out);
}

/// Fallback numbered-list picker when curses is unavailable. `read_line` yields
/// the next input line, or `None` on EOF / interrupt.
pub fn numbered_pick<W: Write>(
    question: &str,
    choices: &[String],
    out: &mut W,
    mut read_line: impl FnMut() -> Option<String>,
) -> Option<usize> {
    let _ = writeln!(out, "{}", question);
    for (i, c) in choices.iter().enumerate() {
        let _ = writeln!(out, "  {}. {}", i + 1, c);
    }
    let _ = writeln!(out);
    loop {
        let _ = write!(out, "Choice [1-{}]: ", choices.len());
        let _ = out.flush();
        match read_line() {
            None => {
                let _ = writeln!(out);
                return None;
            }
            Some(line) => {
                let val = line.trim();
                if val.is_empty() {
                    return None;
                }
                match val.parse::<i64>() {
                    Ok(n) => {
                        let idx = n - 1;
                        if idx >= 0 && (idx as usize) < choices.len() {
                            return Some(idx as usize);
                        }
                        let _ = writeln!(out, "Please enter 1-{}", choices.len());
                    }
                    Err(_) => {
                        let _ = writeln!(out, "Please enter a number");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Normalized fallback subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackSub {
    List,
    Add,
    Remove,
    Clear,
}

/// Map a raw subcommand string (as in `args.fallback_command`) to a
/// [`FallbackSub`]. `None`/""/"list"/"ls" → List; "rm" → Remove. Unknown
/// subcommands return `Err(raw)`.
pub fn parse_sub(sub: Option<&str>) -> Result<FallbackSub, String> {
    match sub.map(str::trim) {
        None | Some("") | Some("list") | Some("ls") => Ok(FallbackSub::List),
        Some("add") => Ok(FallbackSub::Add),
        Some("remove") | Some("rm") => Ok(FallbackSub::Remove),
        Some("clear") => Ok(FallbackSub::Clear),
        Some(other) => Err(other.to_string()),
    }
}

/// Top-level dispatcher for `hermes fallback [subcommand]`.
///
/// `chooser` and `confirm` are only consulted by the remove / clear paths; the
/// caller wires them to the real curses/stdin prompts in production.
pub fn cmd_fallback<E, C, W>(
    env: &mut E,
    sub: Option<&str>,
    chooser: &mut C,
    out: &mut W,
    confirm: impl FnOnce(&str) -> Option<String>,
) -> CmdOutcome
where
    E: FallbackEnv,
    C: Chooser,
    W: Write,
{
    match parse_sub(sub) {
        Ok(FallbackSub::List) => {
            cmd_fallback_list(env, out);
            CmdOutcome::Ok
        }
        Ok(FallbackSub::Add) => cmd_fallback_add(env, out),
        Ok(FallbackSub::Remove) => {
            cmd_fallback_remove(env, chooser, out);
            CmdOutcome::Ok
        }
        Ok(FallbackSub::Clear) => {
            cmd_fallback_clear(env, out, confirm);
            CmdOutcome::Ok
        }
        Err(other) => {
            let _ = writeln!(out, "Unknown fallback subcommand: {}", other);
            let _ = writeln!(out, "Use one of: list, add, remove, clear");
            CmdOutcome::Exit(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Production wiring helpers
// ---------------------------------------------------------------------------

/// Read a single line from stdin, returning `None` on EOF (mirrors Python's
/// `input()` raising EOFError, treated as cancel).
pub fn stdin_read_line() -> Option<String> {
    let mut line = String::new();
    let stdin = io::stdin();
    let n = stdin.lock().read_line(&mut line).ok()?;
    if n == 0 {
        return None;
    }
    // Strip a single trailing newline (and optional CR) to mirror input().
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    Some(line)
}

/// A [`Chooser`] backed by the numbered-pick fallback over stdin. Production
/// code may try curses first and fall back to this.
pub struct NumberedChooser;

impl Chooser for NumberedChooser {
    fn choose(&mut self, question: &str, choices: &[String], _default: usize) -> Option<usize> {
        let mut out = io::stdout();
        numbered_pick(question, choices, &mut out, stdin_read_line)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value as Yaml;

    fn ystr(s: &str) -> Yaml {
        Yaml::String(s.to_string())
    }

    fn yaml_map(pairs: &[(&str, Yaml)]) -> Yaml {
        let mut m = Mapping::new();
        for (k, v) in pairs {
            m.insert(ystr(k), v.clone());
        }
        Yaml::Mapping(m)
    }

    fn entry(p: &str, m: &str) -> Yaml {
        yaml_map(&[("provider", ystr(p)), ("model", ystr(m))])
    }

    // ---- read_chain -------------------------------------------------------

    #[test]
    fn read_chain_new_format() {
        let cfg = yaml_map(&[(
            "fallback_providers",
            Yaml::Sequence(vec![entry("openai", "gpt-4o"), entry("anthropic", "claude")]),
        )]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].provider, "openai");
        assert_eq!(chain[0].model, "gpt-4o");
        assert_eq!(chain[1].provider, "anthropic");
    }

    #[test]
    fn read_chain_filters_invalid_entries() {
        let cfg = yaml_map(&[(
            "fallback_providers",
            Yaml::Sequence(vec![
                entry("openai", "gpt-4o"),
                yaml_map(&[("provider", ystr("x"))]), // missing model
                ystr("not a map"),
            ]),
        )]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "openai");
    }

    #[test]
    fn read_chain_legacy_single_dict() {
        let cfg = yaml_map(&[("fallback_model", entry("openai", "gpt-4o"))]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].model, "gpt-4o");
    }

    #[test]
    fn read_chain_legacy_list() {
        let cfg = yaml_map(&[(
            "fallback_model",
            Yaml::Sequence(vec![entry("a", "m1"), entry("b", "m2")]),
        )]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn read_chain_prefers_new_over_legacy() {
        let cfg = yaml_map(&[
            ("fallback_providers", Yaml::Sequence(vec![entry("new", "n")])),
            ("fallback_model", entry("old", "o")),
        ]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "new");
    }

    #[test]
    fn read_chain_empty() {
        let cfg = yaml_map(&[]);
        assert!(read_chain(&cfg).is_empty());
    }

    #[test]
    fn read_chain_empty_new_falls_back_to_legacy() {
        // Empty/invalid fallback_providers must NOT short-circuit; legacy wins.
        let cfg = yaml_map(&[
            ("fallback_providers", Yaml::Sequence(vec![])),
            ("fallback_model", entry("old", "o")),
        ]);
        let chain = read_chain(&cfg);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "old");
    }

    // ---- write_chain ------------------------------------------------------

    #[test]
    fn write_chain_drops_legacy_key() {
        let mut cfg = yaml_map(&[("fallback_model", entry("old", "o"))]);
        let chain = vec![FallbackEntry {
            provider: "p".into(),
            model: "m".into(),
            base_url: Some("https://x".into()),
            api_mode: None,
        }];
        write_chain(&mut cfg, &chain);
        assert!(map_get(&cfg, "fallback_model").is_none());
        let fp = map_get(&cfg, "fallback_providers").unwrap();
        let seq = fp.as_sequence().unwrap();
        assert_eq!(seq.len(), 1);
        assert_eq!(map_get(&seq[0], "base_url"), Some(&ystr("https://x")));
        assert!(map_get(&seq[0], "api_mode").is_none());
    }

    #[test]
    fn write_then_read_roundtrip() {
        let mut cfg = yaml_map(&[]);
        let chain = vec![
            FallbackEntry { provider: "a".into(), model: "m1".into(), base_url: None, api_mode: Some("chat".into()) },
            FallbackEntry { provider: "b".into(), model: "m2".into(), base_url: Some("u".into()), api_mode: None },
        ];
        write_chain(&mut cfg, &chain);
        let back = read_chain(&cfg);
        assert_eq!(back, chain);
    }

    // ---- format_entry -----------------------------------------------------

    #[test]
    fn format_entry_basic() {
        let e = FallbackEntry { provider: "openai".into(), model: "gpt-4o".into(), base_url: None, api_mode: None };
        assert_eq!(format_entry(&e), "gpt-4o  (via openai)");
    }

    #[test]
    fn format_entry_with_base_url() {
        let e = FallbackEntry { provider: "openai".into(), model: "gpt-4o".into(), base_url: Some("https://x".into()), api_mode: None };
        assert_eq!(format_entry(&e), "gpt-4o  (via openai)  [https://x]");
    }

    #[test]
    fn format_entry_missing_uses_question_mark() {
        let e = FallbackEntry::default();
        assert_eq!(format_entry(&e), "?  (via ?)");
    }

    // ---- extract_fallback_from_model_cfg ----------------------------------

    #[test]
    fn extract_uses_default_then_model() {
        let cfg = yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))]);
        let e = extract_fallback_from_model_cfg(Some(&cfg)).unwrap();
        assert_eq!(e.model, "gpt-4o");

        let cfg2 = yaml_map(&[("provider", ystr("openai")), ("model", ystr("legacy"))]);
        let e2 = extract_fallback_from_model_cfg(Some(&cfg2)).unwrap();
        assert_eq!(e2.model, "legacy");
    }

    #[test]
    fn extract_requires_provider_and_model() {
        assert!(extract_fallback_from_model_cfg(Some(&yaml_map(&[("provider", ystr("x"))]))).is_none());
        assert!(extract_fallback_from_model_cfg(Some(&yaml_map(&[("default", ystr("m"))]))).is_none());
        assert!(extract_fallback_from_model_cfg(Some(&ystr("string"))).is_none());
        assert!(extract_fallback_from_model_cfg(None).is_none());
    }

    #[test]
    fn extract_optional_fields() {
        let cfg = yaml_map(&[
            ("provider", ystr(" openai ")),
            ("default", ystr(" gpt-4o ")),
            ("base_url", ystr(" https://x ")),
            ("api_mode", ystr(" chat ")),
        ]);
        let e = extract_fallback_from_model_cfg(Some(&cfg)).unwrap();
        assert_eq!(e.provider, "openai");
        assert_eq!(e.model, "gpt-4o");
        assert_eq!(e.base_url.as_deref(), Some("https://x"));
        assert_eq!(e.api_mode.as_deref(), Some("chat"));
    }

    // ---- describe_primary -------------------------------------------------

    #[test]
    fn describe_primary_map() {
        let cfg = yaml_map(&[("model", yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))]))]);
        assert_eq!(describe_primary(&cfg).as_deref(), Some("gpt-4o  (via openai)"));
    }

    #[test]
    fn describe_primary_string() {
        let cfg = yaml_map(&[("model", ystr("  some-model  "))]);
        assert_eq!(describe_primary(&cfg).as_deref(), Some("some-model"));
    }

    #[test]
    fn describe_primary_none() {
        assert!(describe_primary(&yaml_map(&[])).is_none());
    }

    #[test]
    fn describe_primary_empty_map_uses_qmarks() {
        let cfg = yaml_map(&[("model", yaml_map(&[]))]);
        assert_eq!(describe_primary(&cfg).as_deref(), Some("?  (via ?)"));
    }

    // ---- parse_sub --------------------------------------------------------

    #[test]
    fn parse_sub_variants() {
        assert_eq!(parse_sub(None), Ok(FallbackSub::List));
        assert_eq!(parse_sub(Some("")), Ok(FallbackSub::List));
        assert_eq!(parse_sub(Some("list")), Ok(FallbackSub::List));
        assert_eq!(parse_sub(Some("ls")), Ok(FallbackSub::List));
        assert_eq!(parse_sub(Some("add")), Ok(FallbackSub::Add));
        assert_eq!(parse_sub(Some("remove")), Ok(FallbackSub::Remove));
        assert_eq!(parse_sub(Some("rm")), Ok(FallbackSub::Remove));
        assert_eq!(parse_sub(Some("clear")), Ok(FallbackSub::Clear));
        assert_eq!(parse_sub(Some("bogus")), Err("bogus".to_string()));
    }

    // ---- numbered_pick ----------------------------------------------------

    #[test]
    fn numbered_pick_valid_choice() {
        let choices = vec!["a".to_string(), "b".to_string(), "Cancel".to_string()];
        let mut out = Vec::new();
        let mut inputs = vec!["2".to_string()].into_iter();
        let idx = numbered_pick("Pick:", &choices, &mut out, || inputs.next());
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn numbered_pick_empty_cancels() {
        let choices = vec!["a".to_string()];
        let mut out = Vec::new();
        let mut inputs = vec!["".to_string()].into_iter();
        assert_eq!(numbered_pick("Pick:", &choices, &mut out, || inputs.next()), None);
    }

    #[test]
    fn numbered_pick_retries_on_bad_input() {
        let choices = vec!["a".to_string(), "b".to_string()];
        let mut out = Vec::new();
        let mut inputs = vec!["x".to_string(), "9".to_string(), "1".to_string()].into_iter();
        assert_eq!(numbered_pick("Pick:", &choices, &mut out, || inputs.next()), Some(0));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Please enter a number"));
        assert!(text.contains("Please enter 1-2"));
    }

    #[test]
    fn numbered_pick_eof_cancels() {
        let choices = vec!["a".to_string()];
        let mut out = Vec::new();
        assert_eq!(numbered_pick("Pick:", &choices, &mut out, || None), None);
    }

    // ---- Fake env for command-level tests ---------------------------------

    struct FakeEnv {
        config: Yaml,
        active_provider: Yaml,
        require_tty_ok: bool,
        picker: Option<PickerOutcome>,
    }

    impl FakeEnv {
        fn new(config: Yaml) -> Self {
            FakeEnv { config, active_provider: Yaml::Null, require_tty_ok: true, picker: None }
        }
    }

    impl FallbackEnv for FakeEnv {
        fn load_config(&self) -> Yaml {
            self.config.clone()
        }
        fn save_config(&mut self, config: &Yaml) -> Result<(), String> {
            self.config = config.clone();
            Ok(())
        }
        fn snapshot_active_provider(&self) -> Yaml {
            self.active_provider.clone()
        }
        fn restore_active_provider(&mut self, value: &Yaml) {
            self.active_provider = value.clone();
        }
        fn require_tty(&self, _action: &str) -> Result<(), String> {
            if self.require_tty_ok { Ok(()) } else { Err("requires a TTY".into()) }
        }
        fn run_picker(&mut self) -> PickerOutcome {
            self.picker.take().unwrap_or(PickerOutcome::Cancelled)
        }
    }

    struct StaticChooser(Option<usize>);
    impl Chooser for StaticChooser {
        fn choose(&mut self, _q: &str, _c: &[String], _d: usize) -> Option<usize> {
            self.0
        }
    }

    #[test]
    fn list_empty() {
        let env = FakeEnv::new(yaml_map(&[]));
        let mut out = Vec::new();
        cmd_fallback_list(&env, &mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("No fallback providers configured."));
        assert!(text.contains("hermes fallback add"));
    }

    #[test]
    fn list_with_entries_and_primary() {
        let cfg = yaml_map(&[
            ("model", yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))])),
            ("fallback_providers", Yaml::Sequence(vec![entry("anthropic", "claude")])),
        ]);
        let env = FakeEnv::new(cfg);
        let mut out = Vec::new();
        cmd_fallback_list(&env, &mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Primary:   gpt-4o  (via openai)"));
        assert!(text.contains("Fallback chain (1 entry):"));
        assert!(text.contains("1. claude  (via anthropic)"));
    }

    #[test]
    fn add_requires_tty() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        env.require_tty_ok = false;
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Exit(1));
    }

    #[test]
    fn add_cancelled_picker() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        env.picker = Some(PickerOutcome::Cancelled);
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Ok);
        assert!(String::from_utf8(out).unwrap().contains("No fallback added."));
        assert!(read_chain(&env.config).is_empty());
    }

    #[test]
    fn add_appends_entry() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        env.picker = Some(PickerOutcome::Completed {
            model_after: Some(yaml_map(&[("provider", ystr("anthropic")), ("default", ystr("claude"))])),
        });
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Ok);
        let chain = read_chain(&env.config);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "anthropic");
        assert_eq!(chain[0].model, "claude");
        // model_before was None → restore should have removed the model key.
        assert!(map_get(&env.config, "model").is_none());
        assert!(String::from_utf8(out).unwrap().contains("Added fallback: claude  (via anthropic)"));
    }

    #[test]
    fn add_rejects_self_as_primary() {
        let cfg = yaml_map(&[("model", yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))]))]);
        let mut env = FakeEnv::new(cfg);
        env.picker = Some(PickerOutcome::Completed {
            model_after: Some(yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))])),
        });
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Ok);
        assert!(read_chain(&env.config).is_empty());
        // model restored to before snapshot
        assert!(map_get(&env.config, "model").is_some());
        assert!(String::from_utf8(out).unwrap().contains("cannot be a fallback for itself"));
    }

    #[test]
    fn add_rejects_duplicate() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("anthropic", "claude")]))]);
        let mut env = FakeEnv::new(cfg);
        env.picker = Some(PickerOutcome::Completed {
            model_after: Some(yaml_map(&[("provider", ystr("anthropic")), ("default", ystr("claude"))])),
        });
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Ok);
        assert_eq!(read_chain(&env.config).len(), 1);
        assert!(String::from_utf8(out).unwrap().contains("already in the fallback chain"));
    }

    #[test]
    fn add_picker_exit_propagates_and_restores() {
        let cfg = yaml_map(&[("model", yaml_map(&[("provider", ystr("openai")), ("default", ystr("gpt-4o"))]))]);
        let mut env = FakeEnv::new(cfg);
        env.active_provider = ystr("openai");
        env.picker = Some(PickerOutcome::Exit(3));
        let mut out = Vec::new();
        assert_eq!(cmd_fallback_add(&mut env, &mut out), CmdOutcome::Exit(3));
        // primary + active provider restored
        assert!(map_get(&env.config, "model").is_some());
        assert_eq!(env.active_provider, ystr("openai"));
    }

    #[test]
    fn remove_cancel_index() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1"), entry("b", "m2")]))]);
        let mut env = FakeEnv::new(cfg);
        // chooser returns the Cancel sentinel index (== chain.len() == 2)
        let mut chooser = StaticChooser(Some(2));
        let mut out = Vec::new();
        cmd_fallback_remove(&mut env, &mut chooser, &mut out);
        assert_eq!(read_chain(&env.config).len(), 2);
        assert!(String::from_utf8(out).unwrap().contains("Cancelled — no change."));
    }

    #[test]
    fn remove_entry() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1"), entry("b", "m2")]))]);
        let mut env = FakeEnv::new(cfg);
        let mut chooser = StaticChooser(Some(0));
        let mut out = Vec::new();
        cmd_fallback_remove(&mut env, &mut chooser, &mut out);
        let chain = read_chain(&env.config);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "b");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Removed fallback: m1  (via a)"));
        assert!(text.contains("Chain is now 1 entry long."));
    }

    #[test]
    fn remove_last_entry_reports_empty() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1")]))]);
        let mut env = FakeEnv::new(cfg);
        let mut chooser = StaticChooser(Some(0));
        let mut out = Vec::new();
        cmd_fallback_remove(&mut env, &mut chooser, &mut out);
        assert!(read_chain(&env.config).is_empty());
        assert!(String::from_utf8(out).unwrap().contains("Fallback chain is now empty."));
    }

    #[test]
    fn remove_empty_chain() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        let mut chooser = StaticChooser(Some(0));
        let mut out = Vec::new();
        cmd_fallback_remove(&mut env, &mut chooser, &mut out);
        assert!(String::from_utf8(out).unwrap().contains("nothing to remove"));
    }

    #[test]
    fn clear_confirmed() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1")]))]);
        let mut env = FakeEnv::new(cfg);
        let mut out = Vec::new();
        cmd_fallback_clear(&mut env, &mut out, |_| Some("y".to_string()));
        assert!(read_chain(&env.config).is_empty());
        assert!(String::from_utf8(out).unwrap().contains("Fallback chain cleared."));
    }

    #[test]
    fn clear_declined() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1")]))]);
        let mut env = FakeEnv::new(cfg);
        let mut out = Vec::new();
        cmd_fallback_clear(&mut env, &mut out, |_| Some("n".to_string()));
        assert_eq!(read_chain(&env.config).len(), 1);
        assert!(String::from_utf8(out).unwrap().contains("Cancelled — no change."));
    }

    #[test]
    fn clear_interrupt() {
        let cfg = yaml_map(&[("fallback_providers", Yaml::Sequence(vec![entry("a", "m1")]))]);
        let mut env = FakeEnv::new(cfg);
        let mut out = Vec::new();
        cmd_fallback_clear(&mut env, &mut out, |_| None);
        assert_eq!(read_chain(&env.config).len(), 1);
        assert!(String::from_utf8(out).unwrap().contains("Cancelled."));
    }

    #[test]
    fn clear_empty_chain() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        let mut out = Vec::new();
        cmd_fallback_clear(&mut env, &mut out, |_| Some("y".to_string()));
        assert!(String::from_utf8(out).unwrap().contains("nothing to clear"));
    }

    #[test]
    fn dispatch_unknown_subcommand() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        let mut chooser = StaticChooser(None);
        let mut out = Vec::new();
        let r = cmd_fallback(&mut env, Some("frobnicate"), &mut chooser, &mut out, |_| None);
        assert_eq!(r, CmdOutcome::Exit(2));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Unknown fallback subcommand: frobnicate"));
        assert!(text.contains("Use one of: list, add, remove, clear"));
    }

    #[test]
    fn dispatch_list_default() {
        let mut env = FakeEnv::new(yaml_map(&[]));
        let mut chooser = StaticChooser(None);
        let mut out = Vec::new();
        let r = cmd_fallback(&mut env, None, &mut chooser, &mut out, |_| None);
        assert_eq!(r, CmdOutcome::Ok);
        assert!(String::from_utf8(out).unwrap().contains("No fallback providers configured."));
    }
}
