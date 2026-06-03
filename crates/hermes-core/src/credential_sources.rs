//! Unified removal contract for every credential source Hermes reads from.
//!
//! Native Rust port of `agent/credential_sources.py`.
//!
//! Hermes seeds its credential pool from many places:
//!
//! ```text
//!     env:<VAR>     — os.environ / ~/.hermes/.env
//!     claude_code   — ~/.claude/.credentials.json
//!     hermes_pkce   — ~/.hermes/.anthropic_oauth.json
//!     device_code   — auth.json providers.<provider> (nous, openai-codex, ...)
//!     qwen-cli      — ~/.qwen/oauth_creds.json
//!     gh_cli        — gh auth token
//!     config:<name> — custom_providers config entry
//!     model_config  — model.api_key when model.provider == "custom"
//!     manual        — user ran `hermes auth add`
//! ```
//!
//! Each source has its own *reader* elsewhere (the `_seed_from_*` helpers,
//! not ported here). What this module unifies is **removal**: every source
//! registers a [`RemovalStep`] that, in the same shape:
//!
//! 1. cleans up the externally-readable state the source reads from
//!    (`.env` line, `auth.json` block, OAuth file, ...);
//! 2. suppresses the `(provider, source_id)` so the corresponding seeding
//!    branch skips the upsert on re-load;
//! 3. returns a [`RemovalResult`] describing what was cleaned plus any
//!    diagnostic hints the user should see.
//!
//! ## Side effects via [`RemovalContext`]
//!
//! The Python `remove_fn`s lazily import and call helpers that live in other
//! modules (`hermes_cli.config.remove_env_value`,
//! `hermes_cli.auth.suppress_credential_source`, the auth-store mutators,
//! `get_hermes_home`). Those modules are not all ported yet, so to keep this
//! module self-contained, compilable, and testable, every external side
//! effect is funnelled through the [`RemovalContext`] trait.
//! [`DefaultRemovalContext`] implements the real filesystem/env behaviour and
//! defers the not-yet-ported auth-store / `.env` / suppression operations
//! onto [`DefaultRemovalContext::pending_ops`] for the caller to dispatch.
//! Tests inject a mock.

use std::env;
use std::path::PathBuf;

/// The removed pool entry handed to a [`RemovalStep`]'s remove function.
///
/// Mirrors the relevant slice of Python's `PooledCredential`: only the
/// `source` string is consulted by the removal logic.
#[derive(Debug, Clone)]
pub struct RemovedEntry {
    /// Source identifier as it appeared in `PooledCredential.source`
    /// (`"claude_code"`, `"env:XAI_API_KEY"`, `"manual:device_code"`, ...).
    pub source: String,
}

impl RemovedEntry {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }
}

/// Outcome of removing a credential source.
///
/// Mirrors the Python `RemovalResult` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovalResult {
    /// Short strings describing external state that was actually mutated
    /// (`"Cleared XAI_API_KEY from .env"`). Printed verbatim to the user.
    pub cleaned: Vec<String>,
    /// Diagnostic lines ABOUT state the user may need to clean up themselves
    /// or that is deliberately left intact (shell-exported env var, Claude
    /// Code credential file we don't delete, ...). Always non-destructive.
    pub hints: Vec<String>,
    /// Whether to suppress the source after cleanup so future `load_pool`
    /// calls skip it. Defaults to `true`; the only legitimate `false` is
    /// `manual` entries, which aren't seeded from anywhere external.
    pub suppress: bool,
}

impl Default for RemovalResult {
    fn default() -> Self {
        Self::new()
    }
}

impl RemovalResult {
    /// Empty result with `suppress = true` (the dataclass default).
    pub fn new() -> Self {
        Self {
            cleaned: Vec::new(),
            hints: Vec::new(),
            suppress: true,
        }
    }

    /// Construct with an explicit suppress flag (rare; `manual` only).
    pub fn with_suppress(suppress: bool) -> Self {
        Self {
            cleaned: Vec::new(),
            hints: Vec::new(),
            suppress,
        }
    }

    /// Convenience constructor mirroring `RemovalResult(hints=[...])`.
    pub fn from_hints<I, S>(hints: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut result = Self::new();
        result.hints = hints.into_iter().map(Into::into).collect();
        result
    }
}

/// External side effects a removal step may perform.
///
/// Abstracted so this module stays self-contained: the real implementation
/// ([`DefaultRemovalContext`]) does filesystem/env work; the auth-store and
/// suppression operations are wired by the caller; tests inject a mock.
pub trait RemovalContext {
    /// `os.getenv(name)` — current-process environment lookup.
    fn getenv(&self, name: &str) -> Option<String>;

    /// Lines of `~/.hermes/.env` (`get_env_path().read_text().splitlines()`),
    /// or `None` if the file does not exist / cannot be read.
    fn read_env_lines(&self) -> Option<Vec<String>>;

    /// `hermes_cli.config.remove_env_value` — clear `name` from `.env`.
    /// Returns `true` if a value was actually cleared.
    fn remove_env_value(&self, name: &str) -> bool;

    /// Delete `~/.hermes/.anthropic_oauth.json`. Returns:
    /// * `Ok(true)`  — file existed and was deleted,
    /// * `Ok(false)` — file did not exist,
    /// * `Err(msg)`  — deletion failed (message for a diagnostic hint).
    fn delete_hermes_oauth_file(&self) -> Result<bool, String>;

    /// `_clear_auth_store_provider` — delete `auth_store.providers[provider]`.
    /// Returns `true` if the entry existed and was deleted.
    fn clear_auth_store_provider(&self, provider: &str) -> bool;

    /// `hermes_cli.auth.suppress_credential_source(provider, source_id)`.
    fn suppress_credential_source(&self, provider: &str, source_id: &str);
}

/// How to remove one specific credential source cleanly.
///
/// Mirrors the Python `RemovalStep` dataclass.
pub struct RemovalStep {
    /// Provider pool key (`"xai"`, `"anthropic"`, `"nous"`, ...). The special
    /// value `"*"` matches any provider (used for `manual`-style sources).
    pub provider: &'static str,
    /// Source identifier or prefix (`"claude_code"`, `"env:"`, `"config:"`).
    pub source_id: &'static str,
    /// One-line human-readable description for docs / tests.
    pub description: &'static str,
    /// Optional predicate overriding literal `source_id` matching. Receives
    /// the removed entry's source string.
    match_fn: Option<Box<dyn Fn(&str) -> bool + Send + Sync>>,
    /// `(ctx, provider, removed) -> RemovalResult` — does the cleanup.
    #[allow(clippy::type_complexity)]
    remove_fn:
        Box<dyn Fn(&dyn RemovalContext, &str, &RemovedEntry) -> RemovalResult + Send + Sync>,
}

impl RemovalStep {
    /// `RemovalStep.matches`: wildcard provider, optional predicate, else
    /// literal `source_id` equality.
    pub fn matches(&self, provider: &str, source: &str) -> bool {
        if self.provider != "*" && self.provider != provider {
            return false;
        }
        match &self.match_fn {
            Some(predicate) => predicate(source),
            None => source == self.source_id,
        }
    }

    /// Run the step's removal function.
    pub fn remove(
        &self,
        ctx: &dyn RemovalContext,
        provider: &str,
        removed: &RemovedEntry,
    ) -> RemovalResult {
        (self.remove_fn)(ctx, provider, removed)
    }
}

impl std::fmt::Debug for RemovalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemovalStep")
            .field("provider", &self.provider)
            .field("source_id", &self.source_id)
            .field("description", &self.description)
            .field("has_match_fn", &self.match_fn.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Individual remove_fn implementations — one per source.
// ---------------------------------------------------------------------------

/// `env:<VAR>` — the most common case.
fn remove_env_source(
    ctx: &dyn RemovalContext,
    provider: &str,
    removed: &RemovedEntry,
) -> RemovalResult {
    let mut result = RemovalResult::new();
    let env_var = removed.source.strip_prefix("env:").unwrap_or("");
    if env_var.is_empty() {
        return result;
    }

    // Detect shell vs .env BEFORE remove_env_value pops the process env.
    let env_in_process = ctx.getenv(env_var).map(|v| !v.is_empty()).unwrap_or(false);
    let env_in_dotenv = ctx
        .read_env_lines()
        .map(|lines| {
            lines
                .iter()
                .any(|line| line.trim_start().starts_with(&format!("{env_var}=")))
        })
        .unwrap_or(false);
    let shell_exported = env_in_process && !env_in_dotenv;

    if ctx.remove_env_value(env_var) {
        result.cleaned.push(format!("Cleared {env_var} from .env"));
    }

    if shell_exported {
        result.hints.extend([
            format!(
                "Note: {env_var} is still set in your shell environment \
                 (not in ~/.hermes/.env)."
            ),
            "  Unset it there (shell profile, systemd EnvironmentFile, \
             launchd plist, etc.) or it will keep being visible to Hermes."
                .to_string(),
            format!(
                "  The pool entry is now suppressed — Hermes will ignore \
                 {env_var} until you run `hermes auth add {provider}`."
            ),
        ]);
    } else {
        result.hints.push(format!(
            "Suppressed env:{env_var} — it will not be re-seeded even \
             if the variable is re-exported later."
        ));
    }
    result
}

/// `~/.claude/.credentials.json` is owned by Claude Code — suppress, don't delete.
fn remove_claude_code(
    _ctx: &dyn RemovalContext,
    _provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    RemovalResult::from_hints([
        "Suppressed claude_code credential — it will not be re-seeded.",
        "Note: Claude Code credentials still live in ~/.claude/.credentials.json",
        "Run `hermes auth add anthropic` to re-enable if needed.",
    ])
}

/// `~/.hermes/.anthropic_oauth.json` is ours — delete it outright.
fn remove_hermes_pkce(
    ctx: &dyn RemovalContext,
    _provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    let mut result = RemovalResult::new();
    match ctx.delete_hermes_oauth_file() {
        Ok(true) => result
            .cleaned
            .push("Cleared Hermes Anthropic OAuth credentials".to_string()),
        Ok(false) => {}
        Err(msg) => result.hints.push(msg),
    }
    result
}

/// Nous OAuth lives in `auth.json providers.nous` — clear it and suppress.
fn remove_nous_device_code(
    ctx: &dyn RemovalContext,
    provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    let mut result = RemovalResult::new();
    if ctx.clear_auth_store_provider(provider) {
        result
            .cleaned
            .push(format!("Cleared {provider} OAuth tokens from auth store"));
    }
    result
}

/// MiniMax OAuth lives in `auth.json providers.minimax-oauth` — clear it.
fn remove_minimax_oauth(
    ctx: &dyn RemovalContext,
    provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    let mut result = RemovalResult::new();
    if ctx.clear_auth_store_provider(provider) {
        result
            .cleaned
            .push(format!("Cleared {provider} OAuth tokens from auth store"));
    }
    result
}

/// Codex tokens live in TWO places: our auth store AND `~/.codex/auth.json`.
/// Clear the auth store and suppress the canonical `device_code` re-seed key.
fn remove_codex_device_code(
    ctx: &dyn RemovalContext,
    provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    let mut result = RemovalResult::new();
    if ctx.clear_auth_store_provider(provider) {
        result
            .cleaned
            .push(format!("Cleared {provider} OAuth tokens from auth store"));
    }
    // Suppress the canonical re-seed source, not just whatever source the
    // removed entry had — otherwise `manual:device_code` removals wouldn't
    // block the `device_code` re-seed path.
    ctx.suppress_credential_source(provider, "device_code");
    result.hints.extend([
        "Suppressed openai-codex device_code source — it will not be re-seeded.".to_string(),
        "Note: Codex CLI credentials still live in ~/.codex/auth.json".to_string(),
        "Run `hermes auth add openai-codex` to re-enable if needed.".to_string(),
    ]);
    result
}

/// `~/.qwen/oauth_creds.json` is owned by the Qwen CLI — suppress, don't delete.
fn remove_qwen_cli(
    _ctx: &dyn RemovalContext,
    _provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    RemovalResult::from_hints([
        "Suppressed qwen-cli credential — it will not be re-seeded.",
        "Note: Qwen CLI credentials still live in ~/.qwen/oauth_creds.json",
        "Run `hermes auth add qwen-oauth` to re-enable if needed.",
    ])
}

/// Copilot token comes from `gh auth token` or
/// `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN`. Suppress ALL known
/// copilot sources so removal is stable regardless of which entry was clicked.
fn remove_copilot_gh(
    ctx: &dyn RemovalContext,
    provider: &str,
    _removed: &RemovedEntry,
) -> RemovalResult {
    ctx.suppress_credential_source(provider, "gh_cli");
    for env_var in ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"] {
        ctx.suppress_credential_source(provider, &format!("env:{env_var}"));
    }
    RemovalResult::from_hints([
        "Suppressed all copilot token sources (gh_cli + env vars) — they will not be re-seeded.",
        "Note: Your gh CLI / shell environment is unchanged.",
        "Run `hermes auth add copilot` to re-enable if needed.",
    ])
}

/// Custom provider pools come from `custom_providers` config or
/// `model.api_key` — both in `config.yaml`. Suppress; don't touch the file.
fn remove_custom_config(
    _ctx: &dyn RemovalContext,
    _provider: &str,
    removed: &RemovedEntry,
) -> RemovalResult {
    RemovalResult::from_hints([
        format!("Suppressed {} — it will not be re-seeded.", removed.source),
        "Note: The underlying value in config.yaml is unchanged.  Edit it \
         directly if you want to remove the credential from disk."
            .to_string(),
    ])
}

// ---------------------------------------------------------------------------
// Registry construction.
// ---------------------------------------------------------------------------

/// Build the ordered list of every registered [`RemovalStep`].
///
/// Mirrors `_register_all_sources()`. **ORDER MATTERS** — [`find_removal_step`]
/// returns the first match, so provider-specific steps precede the generic
/// `env:*` / `config:*` steps.
pub fn registry() -> Vec<RemovalStep> {
    vec![
        RemovalStep {
            provider: "copilot",
            source_id: "gh_cli",
            description: "gh auth token / COPILOT_GITHUB_TOKEN / GH_TOKEN",
            match_fn: Some(Box::new(|src: &str| src == "gh_cli" || src.starts_with("env:"))),
            remove_fn: Box::new(remove_copilot_gh),
        },
        RemovalStep {
            provider: "*",
            source_id: "env:",
            description: "Any env-seeded credential (XAI_API_KEY, DEEPSEEK_API_KEY, etc.)",
            match_fn: Some(Box::new(|src: &str| src.starts_with("env:"))),
            remove_fn: Box::new(remove_env_source),
        },
        RemovalStep {
            provider: "anthropic",
            source_id: "claude_code",
            description: "~/.claude/.credentials.json",
            match_fn: None,
            remove_fn: Box::new(remove_claude_code),
        },
        RemovalStep {
            provider: "anthropic",
            source_id: "hermes_pkce",
            description: "~/.hermes/.anthropic_oauth.json",
            match_fn: None,
            remove_fn: Box::new(remove_hermes_pkce),
        },
        RemovalStep {
            provider: "nous",
            source_id: "device_code",
            description: "auth.json providers.nous",
            match_fn: None,
            remove_fn: Box::new(remove_nous_device_code),
        },
        RemovalStep {
            provider: "openai-codex",
            source_id: "device_code",
            description: "auth.json providers.openai-codex + ~/.codex/auth.json",
            match_fn: Some(Box::new(|src: &str| {
                src == "device_code" || src.ends_with(":device_code")
            })),
            remove_fn: Box::new(remove_codex_device_code),
        },
        RemovalStep {
            provider: "qwen-oauth",
            source_id: "qwen-cli",
            description: "~/.qwen/oauth_creds.json",
            match_fn: None,
            remove_fn: Box::new(remove_qwen_cli),
        },
        RemovalStep {
            provider: "minimax-oauth",
            source_id: "oauth",
            description: "auth.json providers.minimax-oauth",
            match_fn: None,
            remove_fn: Box::new(remove_minimax_oauth),
        },
        RemovalStep {
            provider: "*",
            source_id: "config:",
            description: "Custom provider config.yaml api_key field",
            match_fn: Some(Box::new(|src: &str| {
                src.starts_with("config:") || src == "model_config"
            })),
            remove_fn: Box::new(remove_custom_config),
        },
    ]
}

/// Return the first [`RemovalStep`] matching `(provider, source)`, or `None`.
///
/// Mirrors `find_removal_step`. Unregistered sources fall through to the
/// default remove path (the pool entry is already gone; no external cleanup,
/// no suppression) — correct for `manual` entries.
pub fn find_removal_step<'a>(
    steps: &'a [RemovalStep],
    provider: &str,
    source: &str,
) -> Option<&'a RemovalStep> {
    steps.iter().find(|step| step.matches(provider, source))
}

/// Convenience: build the default registry and return all steps.
pub fn all_steps() -> Vec<RemovalStep> {
    registry()
}

// ---------------------------------------------------------------------------
// DefaultRemovalContext — real filesystem/env implementation.
// ---------------------------------------------------------------------------

/// A side effect the [`DefaultRemovalContext`] defers to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingOp {
    /// `remove_env_value(name)`
    RemoveEnvValue(String),
    /// `clear_auth_store_provider(provider)`
    ClearAuthStoreProvider(String),
    /// `suppress_credential_source(provider, source_id)`
    SuppressCredentialSource { provider: String, source_id: String },
}

/// Real-environment [`RemovalContext`].
///
/// `getenv` / `read_env_lines` / `delete_hermes_oauth_file` do genuine work
/// against the process environment and `~/.hermes`. `remove_env_value`,
/// `clear_auth_store_provider`, and `suppress_credential_source` belong to
/// the not-yet-ported config/auth modules; until those are wired in, the
/// default implementation records the requested operations on `pending_ops`
/// so the caller (the main loop) can dispatch them, rather than silently
/// no-op'ing. Real wiring can replace this with a context that calls the
/// ported helpers directly.
#[derive(Debug, Default)]
pub struct DefaultRemovalContext {
    /// Operations the default context could not perform itself, recorded for
    /// the caller to dispatch into the real auth/config subsystems.
    pub pending_ops: std::cell::RefCell<Vec<PendingOp>>,
}

impl DefaultRemovalContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// `~/.hermes` honouring `HERMES_HOME`. Mirrors `get_hermes_home()`.
    fn hermes_home(&self) -> PathBuf {
        if let Ok(val) = env::var("HERMES_HOME") {
            let val = val.trim();
            if !val.is_empty() {
                return PathBuf::from(val);
            }
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
    }

    fn env_path(&self) -> PathBuf {
        self.hermes_home().join(".env")
    }

    fn oauth_path(&self) -> PathBuf {
        self.hermes_home().join(".anthropic_oauth.json")
    }
}

impl RemovalContext for DefaultRemovalContext {
    fn getenv(&self, name: &str) -> Option<String> {
        env::var(name).ok()
    }

    fn read_env_lines(&self) -> Option<Vec<String>> {
        let path = self.env_path();
        if !path.exists() {
            return None;
        }
        // mirror read_text(errors="replace"): lossy decode, then splitlines.
        let bytes = std::fs::read(&path).ok()?;
        let text = String::from_utf8_lossy(&bytes);
        Some(text.lines().map(str::to_string).collect())
    }

    fn remove_env_value(&self, name: &str) -> bool {
        // Deferred: the real `.env` mutator lives in the config module.
        self.pending_ops
            .borrow_mut()
            .push(PendingOp::RemoveEnvValue(name.to_string()));
        false
    }

    fn delete_hermes_oauth_file(&self) -> Result<bool, String> {
        let path = self.oauth_path();
        if !path.exists() {
            return Ok(false);
        }
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) => Err(format!("Could not delete {}: {err}", path.display())),
        }
    }

    fn clear_auth_store_provider(&self, provider: &str) -> bool {
        self.pending_ops
            .borrow_mut()
            .push(PendingOp::ClearAuthStoreProvider(provider.to_string()));
        false
    }

    fn suppress_credential_source(&self, provider: &str, source_id: &str) {
        self.pending_ops
            .borrow_mut()
            .push(PendingOp::SuppressCredentialSource {
                provider: provider.to_string(),
                source_id: source_id.to_string(),
            });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    #[derive(Default)]
    struct MockContext {
        env: HashMap<String, String>,
        env_lines: Option<Vec<String>>,
        remove_env_returns: bool,
        oauth_result: Option<Result<bool, String>>,
        clear_store_returns: bool,
        removed_env: RefCell<Vec<String>>,
        cleared_providers: RefCell<Vec<String>>,
        suppressed: RefCell<Vec<(String, String)>>,
    }

    impl RemovalContext for MockContext {
        fn getenv(&self, name: &str) -> Option<String> {
            self.env.get(name).cloned()
        }
        fn read_env_lines(&self) -> Option<Vec<String>> {
            self.env_lines.clone()
        }
        fn remove_env_value(&self, name: &str) -> bool {
            self.removed_env.borrow_mut().push(name.to_string());
            self.remove_env_returns
        }
        fn delete_hermes_oauth_file(&self) -> Result<bool, String> {
            self.oauth_result.clone().unwrap_or(Ok(false))
        }
        fn clear_auth_store_provider(&self, provider: &str) -> bool {
            self.cleared_providers
                .borrow_mut()
                .push(provider.to_string());
            self.clear_store_returns
        }
        fn suppress_credential_source(&self, provider: &str, source_id: &str) {
            self.suppressed
                .borrow_mut()
                .push((provider.to_string(), source_id.to_string()));
        }
    }

    fn find<'a>(steps: &'a [RemovalStep], provider: &str, source: &str) -> &'a RemovalStep {
        find_removal_step(steps, provider, source)
            .unwrap_or_else(|| panic!("no step for {provider}/{source}"))
    }

    #[test]
    fn registry_order_and_count() {
        let steps = registry();
        assert_eq!(steps.len(), 9);
        // copilot must precede the generic env step.
        assert_eq!(steps[0].provider, "copilot");
        assert_eq!(steps[1].source_id, "env:");
        assert_eq!(steps[8].source_id, "config:");
    }

    #[test]
    fn matches_wildcard_and_literal() {
        let steps = registry();
        // env wildcard matches any provider.
        assert!(find_removal_step(&steps, "xai", "env:XAI_API_KEY").is_some());
        // literal claude_code only under anthropic.
        let claude = find(&steps, "anthropic", "claude_code");
        assert_eq!(claude.source_id, "claude_code");
        assert!(find_removal_step(&steps, "openai", "claude_code").is_none());
    }

    #[test]
    fn copilot_env_routes_to_copilot_not_generic_env() {
        let steps = registry();
        // copilot's GH_TOKEN (env:) must hit the copilot step (registered first).
        let step = find(&steps, "copilot", "env:GH_TOKEN");
        assert_eq!(step.provider, "copilot");
    }

    #[test]
    fn unregistered_source_returns_none() {
        let steps = registry();
        assert!(find_removal_step(&steps, "someprovider", "manual").is_none());
        assert!(find_removal_step(&steps, "openai", "manual:device_code").is_none());
    }

    #[test]
    fn codex_matches_manual_variant() {
        let steps = registry();
        let s1 = find(&steps, "openai-codex", "device_code");
        let s2 = find(&steps, "openai-codex", "manual:device_code");
        assert_eq!(s1.provider, "openai-codex");
        assert_eq!(s2.provider, "openai-codex");
    }

    #[test]
    fn env_source_dotenv_only_clears_and_suppresses() {
        let steps = registry();
        let step = find(&steps, "xai", "env:XAI_API_KEY");
        let ctx = MockContext {
            env_lines: Some(vec!["XAI_API_KEY=abc".to_string()]),
            remove_env_returns: true,
            ..MockContext::default()
        };
        // Not in process env -> not shell-exported.
        let result = step.remove(&ctx, "xai", &RemovedEntry::new("env:XAI_API_KEY"));
        assert_eq!(result.cleaned, vec!["Cleared XAI_API_KEY from .env"]);
        assert_eq!(
            result.hints,
            vec![
                "Suppressed env:XAI_API_KEY — it will not be re-seeded even \
                 if the variable is re-exported later."
            ]
        );
        assert!(result.suppress);
        assert_eq!(*ctx.removed_env.borrow(), vec!["XAI_API_KEY"]);
    }

    #[test]
    fn env_source_shell_exported_hints_about_shell() {
        let steps = registry();
        let step = find(&steps, "xai", "env:XAI_API_KEY");
        let mut env = HashMap::new();
        env.insert("XAI_API_KEY".to_string(), "secret".to_string());
        let ctx = MockContext {
            env,
            env_lines: None, // not in .env
            remove_env_returns: false,
            ..MockContext::default()
        };
        let result = step.remove(&ctx, "xai", &RemovedEntry::new("env:XAI_API_KEY"));
        assert!(result.cleaned.is_empty());
        assert_eq!(result.hints.len(), 3);
        assert!(result.hints[0].contains("still set in your shell"));
        assert!(result.hints[2].contains("hermes auth add xai"));
    }

    #[test]
    fn env_source_empty_var_noop() {
        let steps = registry();
        let step = find(&steps, "xai", "env:");
        let ctx = MockContext::default();
        let result = step.remove(&ctx, "xai", &RemovedEntry::new("env:"));
        assert!(result.cleaned.is_empty());
        assert!(result.hints.is_empty());
        assert!(ctx.removed_env.borrow().is_empty());
    }

    #[test]
    fn nous_clears_auth_store() {
        let steps = registry();
        let step = find(&steps, "nous", "device_code");
        let ctx = MockContext {
            clear_store_returns: true,
            ..MockContext::default()
        };
        let result = step.remove(&ctx, "nous", &RemovedEntry::new("device_code"));
        assert_eq!(
            result.cleaned,
            vec!["Cleared nous OAuth tokens from auth store"]
        );
        assert_eq!(*ctx.cleared_providers.borrow(), vec!["nous"]);
    }

    #[test]
    fn codex_clears_store_and_suppresses_canonical() {
        let steps = registry();
        let step = find(&steps, "openai-codex", "manual:device_code");
        let ctx = MockContext {
            clear_store_returns: true,
            ..MockContext::default()
        };
        let result = step.remove(
            &ctx,
            "openai-codex",
            &RemovedEntry::new("manual:device_code"),
        );
        assert_eq!(
            result.cleaned,
            vec!["Cleared openai-codex OAuth tokens from auth store"]
        );
        // Canonical "device_code" key suppressed, not the manual variant.
        assert_eq!(
            *ctx.suppressed.borrow(),
            vec![("openai-codex".to_string(), "device_code".to_string())]
        );
        assert_eq!(result.hints.len(), 3);
    }

    #[test]
    fn copilot_suppresses_all_variants() {
        let steps = registry();
        let step = find(&steps, "copilot", "gh_cli");
        let ctx = MockContext::default();
        let _ = step.remove(&ctx, "copilot", &RemovedEntry::new("gh_cli"));
        let suppressed = ctx.suppressed.borrow();
        assert_eq!(suppressed.len(), 4);
        assert_eq!(suppressed[0], ("copilot".to_string(), "gh_cli".to_string()));
        assert_eq!(
            suppressed[1],
            (
                "copilot".to_string(),
                "env:COPILOT_GITHUB_TOKEN".to_string()
            )
        );
        assert_eq!(
            suppressed[3],
            ("copilot".to_string(), "env:GITHUB_TOKEN".to_string())
        );
    }

    #[test]
    fn hermes_pkce_delete_outcomes() {
        let steps = registry();
        let step = find(&steps, "anthropic", "hermes_pkce");

        let ctx = MockContext {
            oauth_result: Some(Ok(true)),
            ..MockContext::default()
        };
        let r = step.remove(&ctx, "anthropic", &RemovedEntry::new("hermes_pkce"));
        assert_eq!(r.cleaned, vec!["Cleared Hermes Anthropic OAuth credentials"]);

        let ctx2 = MockContext {
            oauth_result: Some(Ok(false)),
            ..MockContext::default()
        };
        let r2 = step.remove(&ctx2, "anthropic", &RemovedEntry::new("hermes_pkce"));
        assert!(r2.cleaned.is_empty() && r2.hints.is_empty());

        let ctx3 = MockContext {
            oauth_result: Some(Err("boom".to_string())),
            ..MockContext::default()
        };
        let r3 = step.remove(&ctx3, "anthropic", &RemovedEntry::new("hermes_pkce"));
        assert_eq!(r3.hints, vec!["boom"]);
    }

    #[test]
    fn custom_config_uses_source_label() {
        let steps = registry();
        let step = find(&steps, "myprov", "config:mypool");
        let ctx = MockContext::default();
        let r = step.remove(&ctx, "myprov", &RemovedEntry::new("config:mypool"));
        assert_eq!(
            r.hints[0],
            "Suppressed config:mypool — it will not be re-seeded."
        );
        // model_config also routes here.
        assert!(find_removal_step(&steps, "myprov", "model_config").is_some());
    }

    #[test]
    fn claude_code_and_qwen_hint_only() {
        let steps = registry();
        let claude = find(&steps, "anthropic", "claude_code");
        let r = claude.remove(
            &MockContext::default(),
            "anthropic",
            &RemovedEntry::new("claude_code"),
        );
        assert!(r.cleaned.is_empty());
        assert_eq!(r.hints.len(), 3);
        assert!(r.suppress);

        let qwen = find(&steps, "qwen-oauth", "qwen-cli");
        let r2 = qwen.remove(
            &MockContext::default(),
            "qwen-oauth",
            &RemovedEntry::new("qwen-cli"),
        );
        assert!(r2.hints[1].contains("~/.qwen/oauth_creds.json"));
    }

    #[test]
    fn default_context_defers_ops() {
        let ctx = DefaultRemovalContext::new();
        ctx.remove_env_value("FOO");
        ctx.clear_auth_store_provider("nous");
        ctx.suppress_credential_source("copilot", "gh_cli");
        let ops = ctx.pending_ops.borrow();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0], PendingOp::RemoveEnvValue("FOO".to_string()));
        assert_eq!(
            ops[1],
            PendingOp::ClearAuthStoreProvider("nous".to_string())
        );
        assert_eq!(
            ops[2],
            PendingOp::SuppressCredentialSource {
                provider: "copilot".to_string(),
                source_id: "gh_cli".to_string()
            }
        );
    }

    #[test]
    fn removal_result_defaults_suppress_true() {
        assert!(RemovalResult::new().suppress);
        assert!(!RemovalResult::with_suppress(false).suppress);
        assert!(RemovalResult::from_hints(["a"]).suppress);
    }
}
