//! Environment variable passthrough registry.
//!
//! Skills that declare `required_environment_variables` in their frontmatter
//! need those vars available in sandboxed execution environments
//! (execute_code, terminal). By default both sandboxes strip secrets from the
//! child process environment for security. This module provides a
//! session-scoped allowlist so skill-declared vars (and user-configured
//! overrides) pass through.
//!
//! Two sources feed the allowlist:
//!
//! 1. **Skill declarations** — when a skill is loaded via `skill_view`, its
//!    `required_environment_variables` are registered here automatically.
//! 2. **User config** — `terminal.env_passthrough` in config.yaml lets users
//!    explicitly allowlist vars for non-skill use cases.
//!
//! Both `code_execution` and the local terminal environment consult
//! [`is_env_passthrough`] before stripping a variable.
//!
//! # Port notes
//!
//! The Python original backed the skill-scoped allowlist with a `ContextVar`
//! to prevent cross-session data bleed in the gateway pipeline. Rust does not
//! have task-local storage that mirrors `contextvars` cleanly across async
//! await points, so the allowlist is stored in a thread-local set. Each
//! gateway worker thread therefore gets its own isolated allowlist, which
//! preserves the no-cross-session-bleed guarantee for the common
//! thread-per-session execution model. Callers that need an explicit,
//! independently-owned allowlist can construct an [`EnvPassthroughRegistry`]
//! directly.
//!
//! The Hermes provider-credential blocklist (the
//! `_HERMES_PROVIDER_ENV_BLOCKLIST` from `tools/environments/local.py`) is not
//! yet ported to native Rust. The Python code imported it lazily and treated a
//! failed import as "no blocklist" (returning `False`). To faithfully
//! reproduce the security behaviour without that dependency, this module
//! exposes the blocklist as an injectable resource via
//! [`set_provider_credential_blocklist`] and an embedded built-in fallback
//! ([`builtin_provider_env_blocklist`]) that mirrors the static entries from
//! the Python `_build_provider_env_blocklist` literal. This keeps the
//! GHSA-rhgp-j443-p4rf protection in force by default rather than degrading to
//! the import-failure path.

use std::collections::HashSet;
use std::sync::{OnceLock, RwLock};

/// A single env-passthrough allowlist plus an optional process-wide config
/// cache.
///
/// This is the explicit, owned form of the registry. Most callers will use the
/// free functions ([`register_env_passthrough`], [`is_env_passthrough`], etc.)
/// which operate on a thread-local registry instance, matching the
/// session-isolation semantics of the original `ContextVar`-backed Python
/// module.
#[derive(Debug, Default, Clone)]
pub struct EnvPassthroughRegistry {
    /// Skill/user registered names (session-scoped, mutable).
    allowed: HashSet<String>,
}

impl EnvPassthroughRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Register environment variable names as allowed in sandboxed
    /// environments.
    ///
    /// Mirrors [`register_env_passthrough`]; see that function for the
    /// GHSA-rhgp-j443-p4rf rationale. Names are trimmed; empty names are
    /// skipped; Hermes-managed provider credentials are refused.
    pub fn register<I, S>(&mut self, var_names: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for raw in var_names {
            let name = raw.as_ref().trim();
            if name.is_empty() {
                continue;
            }
            if is_hermes_provider_credential(name) {
                log::warn!(
                    "env passthrough: refusing to register Hermes provider \
                     credential {:?} (blocked by provider env blocklist). \
                     Skills must not override the execute_code sandbox's \
                     credential scrubbing; see GHSA-rhgp-j443-p4rf.",
                    name
                );
                continue;
            }
            self.allowed.insert(name.to_string());
            log::debug!("env passthrough: registered {}", name);
        }
    }

    /// Check whether `var_name` is allowed to pass through to sandboxes.
    ///
    /// Returns `true` if the variable was registered (by a skill or user) here,
    /// or appears in the cached config-based allowlist.
    pub fn is_passthrough(&self, var_name: &str) -> bool {
        if self.allowed.contains(var_name) {
            return true;
        }
        load_config_passthrough().contains(var_name)
    }

    /// Return the union of registered and config-based passthrough vars.
    pub fn all(&self) -> HashSet<String> {
        let mut out = self.allowed.clone();
        out.extend(load_config_passthrough().iter().cloned());
        out
    }

    /// Reset the skill-scoped allowlist (e.g. on session reset).
    ///
    /// Does not affect the process-wide config-based allowlist.
    pub fn clear(&mut self) {
        self.allowed.clear();
    }
}

thread_local! {
    static THREAD_REGISTRY: std::cell::RefCell<EnvPassthroughRegistry> =
        std::cell::RefCell::new(EnvPassthroughRegistry::new());
}

// ---------------------------------------------------------------------------
// Provider-credential blocklist (GHSA-rhgp-j443-p4rf)
// ---------------------------------------------------------------------------

/// Process-wide override for the provider-credential blocklist. When unset, the
/// [`builtin_provider_env_blocklist`] is used.
static PROVIDER_BLOCKLIST: OnceLock<HashSet<String>> = OnceLock::new();

/// Install the authoritative Hermes provider-credential blocklist for this
/// process.
///
/// Intended to be called once during startup by whatever module ends up owning
/// the port of `_build_provider_env_blocklist` (derived from the provider
/// registry, optional env-var metadata, and the static literal). If never
/// called, [`builtin_provider_env_blocklist`] is used as a safe default.
///
/// Returns `Err` with the supplied set if a blocklist was already installed
/// (the existing one is retained).
pub fn set_provider_credential_blocklist(
    blocklist: HashSet<String>,
) -> Result<(), HashSet<String>> {
    PROVIDER_BLOCKLIST.set(blocklist)
}

/// The active provider-credential blocklist for this process.
fn active_blocklist() -> &'static HashSet<String> {
    PROVIDER_BLOCKLIST.get_or_init(builtin_provider_env_blocklist)
}

/// True if `name` is a Hermes-managed provider credential (API key, token, or
/// similar) per the active provider blocklist.
///
/// Skill-declared `required_environment_variables` frontmatter must not be able
/// to override this list — that was the bypass in GHSA-rhgp-j443-p4rf where a
/// malicious skill registered `ANTHROPIC_TOKEN` / `OPENAI_API_KEY` as
/// passthrough and received the credential in the `execute_code` child process,
/// defeating the sandbox's scrubbing guarantee.
///
/// Non-Hermes API keys (`TENOR_API_KEY`, `NOTION_TOKEN`, etc.) are NOT in the
/// blocklist and remain legitimately registerable — skills that wrap
/// third-party APIs still work.
pub fn is_hermes_provider_credential(name: &str) -> bool {
    active_blocklist().contains(name)
}

/// The built-in static portion of `_HERMES_PROVIDER_ENV_BLOCKLIST`.
///
/// This mirrors the static `blocked.update({...})` literal from
/// `tools/environments/local.py::_build_provider_env_blocklist`. The
/// dynamically-derived entries (from `PROVIDER_REGISTRY` and `OPTIONAL_ENV_VARS`)
/// are added by the owner of those ports via
/// [`set_provider_credential_blocklist`].
pub fn builtin_provider_env_blocklist() -> HashSet<String> {
    [
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "COHERE_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "HELICONE_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_ALLOWED_USERS",
        "WHATSAPP_ENABLED",
        "WHATSAPP_MODE",
        "WHATSAPP_ALLOWED_USERS",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_IGNORE_STORIES",
        "HASS_TOKEN",
        "HASS_URL",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "GATEWAY_ALLOWED_USERS",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "VERCEL_OIDC_TOKEN",
        "VERCEL_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ---------------------------------------------------------------------------
// Config-based passthrough (terminal.env_passthrough)
// ---------------------------------------------------------------------------

/// Cache for the config-based allowlist (loaded once per process).
static CONFIG_PASSTHROUGH: OnceLock<HashSet<String>> = OnceLock::new();

/// Optional override for how the config-based passthrough list is sourced.
///
/// The Python code read `terminal.env_passthrough` from `config.yaml` via
/// `read_raw_config()` + `cfg_get`. That config plumbing is not yet wired into
/// this native module, so the source is injectable. Install a provider once at
/// startup with [`set_config_passthrough_source`]; the first call to
/// [`is_env_passthrough`]/[`get_all_passthrough`] resolves and caches it.
static CONFIG_SOURCE: RwLock<Option<fn() -> Vec<String>>> = RwLock::new(None);

/// Install the function used to source `terminal.env_passthrough` from config.
///
/// Must be called before the config passthrough cache is first read; later
/// calls have no effect on the (already-cached) result. The function should
/// return the raw list of var names from `terminal.env_passthrough`; trimming
/// and empty-filtering are handled here.
pub fn set_config_passthrough_source(source: fn() -> Vec<String>) {
    if let Ok(mut guard) = CONFIG_SOURCE.write() {
        *guard = Some(source);
    }
}

/// Load `terminal.env_passthrough` from config (cached for the process
/// lifetime).
///
/// Mirrors `_load_config_passthrough`: trims each entry, drops empties, and
/// caches the resulting frozen set. If no source is installed, the result is
/// empty (matching the Python behaviour when the config read raised — it logged
/// at debug and returned an empty set).
fn load_config_passthrough() -> &'static HashSet<String> {
    CONFIG_PASSTHROUGH.get_or_init(|| {
        let source = CONFIG_SOURCE.read().ok().and_then(|g| *g);
        let mut result: HashSet<String> = HashSet::new();
        if let Some(src) = source {
            for item in src() {
                let trimmed = item.trim();
                if !trimmed.is_empty() {
                    result.insert(trimmed.to_string());
                }
            }
        } else {
            log::debug!(
                "Could not read terminal.env_passthrough from config: no source installed"
            );
        }
        result
    })
}

// ---------------------------------------------------------------------------
// Free functions operating on the thread-local registry
// ---------------------------------------------------------------------------

/// Register environment variable names as allowed in sandboxed environments.
///
/// Typically called when a skill declares `required_environment_variables`.
///
/// Variables that are Hermes-managed provider credentials (from the active
/// provider blocklist) are rejected here to preserve the `execute_code`
/// sandbox's credential-scrubbing guarantee per GHSA-rhgp-j443-p4rf. A skill
/// that needs to talk to a Hermes-managed provider should do so via the
/// agent's main-process tools (web_search, web_extract, etc.) where the
/// credential remains safely in the main process.
///
/// Non-Hermes third-party API keys (`TENOR_API_KEY`, `NOTION_TOKEN`, etc.) pass
/// through normally — they were never in the sandbox scrub list.
pub fn register_env_passthrough<I, S>(var_names: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    THREAD_REGISTRY.with(|r| r.borrow_mut().register(var_names));
}

/// Check whether `var_name` is allowed to pass through to sandboxes.
///
/// Returns `true` if the variable was registered by a skill or listed in the
/// user's `terminal.env_passthrough` config.
pub fn is_env_passthrough(var_name: &str) -> bool {
    THREAD_REGISTRY.with(|r| r.borrow().is_passthrough(var_name))
}

/// Return the union of skill-registered and config-based passthrough vars.
pub fn get_all_passthrough() -> HashSet<String> {
    THREAD_REGISTRY.with(|r| r.borrow().all())
}

/// Reset the skill-scoped allowlist (e.g. on session reset).
pub fn clear_env_passthrough() {
    THREAD_REGISTRY.with(|r| r.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_check_roundtrip() {
        let mut reg = EnvPassthroughRegistry::new();
        reg.register(["TENOR_API_KEY", "NOTION_TOKEN"]);
        assert!(reg.is_passthrough("TENOR_API_KEY"));
        assert!(reg.is_passthrough("NOTION_TOKEN"));
        assert!(!reg.is_passthrough("UNREGISTERED_VAR"));
    }

    #[test]
    fn names_are_trimmed_and_empties_skipped() {
        let mut reg = EnvPassthroughRegistry::new();
        reg.register(["  PADDED_VAR  ", "", "   ", "\tTABBED\n"]);
        assert!(reg.is_passthrough("PADDED_VAR"));
        assert!(reg.is_passthrough("TABBED"));
        // Empty/whitespace-only entries never register.
        assert!(!reg.is_passthrough(""));
        assert!(!reg.is_passthrough("   "));
        let all = reg.all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn hermes_provider_credentials_are_refused() {
        let mut reg = EnvPassthroughRegistry::new();
        // These are in the built-in blocklist and must be refused.
        reg.register([
            "ANTHROPIC_TOKEN",
            "OPENAI_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ]);
        assert!(!reg.is_passthrough("ANTHROPIC_TOKEN"));
        assert!(!reg.is_passthrough("OPENAI_API_KEY"));
        assert!(!reg.is_passthrough("CLAUDE_CODE_OAUTH_TOKEN"));
    }

    #[test]
    fn non_hermes_keys_still_register_alongside_blocked() {
        let mut reg = EnvPassthroughRegistry::new();
        reg.register(["OPENAI_API_KEY", "TENOR_API_KEY"]);
        assert!(!reg.is_passthrough("OPENAI_API_KEY"));
        assert!(reg.is_passthrough("TENOR_API_KEY"));
    }

    #[test]
    fn clear_resets_registered() {
        let mut reg = EnvPassthroughRegistry::new();
        reg.register(["MY_VAR"]);
        assert!(reg.is_passthrough("MY_VAR"));
        reg.clear();
        assert!(!reg.is_passthrough("MY_VAR"));
    }

    #[test]
    fn all_is_union_of_registered() {
        let mut reg = EnvPassthroughRegistry::new();
        reg.register(["A", "B"]);
        let all = reg.all();
        assert!(all.contains("A"));
        assert!(all.contains("B"));
    }

    #[test]
    fn builtin_blocklist_contains_expected_entries() {
        let bl = builtin_provider_env_blocklist();
        assert!(bl.contains("ANTHROPIC_TOKEN"));
        assert!(bl.contains("OPENAI_API_KEY"));
        assert!(bl.contains("VERCEL_TOKEN"));
        assert!(bl.contains("GH_TOKEN"));
        // A third-party key must NOT be in the blocklist.
        assert!(!bl.contains("TENOR_API_KEY"));
        assert!(!bl.contains("NOTION_TOKEN"));
    }

    #[test]
    fn is_hermes_provider_credential_uses_active_blocklist() {
        // Default (no override installed at the point this test runs) should
        // reflect the built-in list for clearly-blocked entries.
        assert!(is_hermes_provider_credential("ANTHROPIC_TOKEN"));
        assert!(!is_hermes_provider_credential("DEFINITELY_NOT_A_PROVIDER_KEY"));
    }

    #[test]
    fn thread_local_free_functions_isolated_per_thread() {
        clear_env_passthrough();
        register_env_passthrough(["THREAD_MAIN_VAR"]);
        assert!(is_env_passthrough("THREAD_MAIN_VAR"));

        let seen_in_other = std::thread::spawn(|| {
            // Fresh thread -> fresh registry; should not see main thread's var.
            let leaked = is_env_passthrough("THREAD_MAIN_VAR");
            register_env_passthrough(["OTHER_THREAD_VAR"]);
            (leaked, is_env_passthrough("OTHER_THREAD_VAR"))
        })
        .join()
        .unwrap();

        assert!(!seen_in_other.0, "var leaked across threads");
        assert!(seen_in_other.1);
        // Main thread does not see the other thread's registration.
        assert!(!is_env_passthrough("OTHER_THREAD_VAR"));
        clear_env_passthrough();
    }
}
