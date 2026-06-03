//! Platform Adapter Registry
//!
//! Native Rust port of `gateway/platform_registry.py`.
//!
//! Allows platform adapters (built-in and plugin) to self-register so the
//! gateway can discover and instantiate them without hardcoded `if/elif`
//! chains.
//!
//! Built-in adapters continue to use the existing `if/elif` in
//! `_create_adapter()` for now. Plugin adapters register here via
//! `PluginContext.register_platform()` and are looked up first -- if nothing is
//! found the gateway falls through to the legacy code path.
//!
//! # Usage (plugin side)
//!
//! ```ignore
//! use hermes_core::gw_platform_registry::{platform_registry, PlatformEntry};
//!
//! let mut entry = PlatformEntry::new(
//!     "irc",
//!     "IRC",
//!     |cfg| Some(Box::new(IRCAdapter::new(cfg))),
//!     || check_requirements(),
//! );
//! entry.validate_config = Some(Box::new(|cfg| cfg.has_server()));
//! entry.required_env = vec!["IRC_SERVER".into()];
//! entry.install_hint = "pip install irc".into();
//! platform_registry().lock().unwrap().register(entry);
//! ```
//!
//! # Usage (gateway side)
//!
//! ```ignore
//! let adapter = platform_registry().lock().unwrap().create_adapter("irc", cfg);
//! ```
//!
//! ## Design notes
//!
//! The Python module stores arbitrary Python callables (factories, predicates,
//! setup functions) that take/return dynamically-typed objects. To reproduce
//! this faithfully in Rust without committing to a concrete adapter/config type,
//! this port is generic over a configuration type `C` and an adapter type `A`.
//! The default singleton uses boxed dynamically-typed values
//! (`Box<dyn Any + Send>`) so that callers operating across module boundaries
//! can use it exactly like the Python module-level `platform_registry`
//! singleton.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Boxed factory: receives a config value, returns an adapter instance (or
/// `None` if construction fails — the Python factory may raise, which we model
/// as returning `None`).
pub type AdapterFactory<C, A> = Box<dyn Fn(&C) -> Option<A> + Send>;

/// Boxed dependency-availability predicate (Python `check_fn`).
pub type CheckFn = Box<dyn Fn() -> bool + Send>;

/// Boxed config predicate (Python `validate_config` / `is_connected`).
pub type ConfigPredicate<C> = Box<dyn Fn(&C) -> bool + Send>;

/// Boxed interactive setup function (Python `setup_fn`).
pub type SetupFn = Box<dyn Fn() + Send>;

/// Metadata and factory for a single platform adapter.
///
/// Mirrors the Python `PlatformEntry` dataclass. Generic over the config type
/// `C` passed to predicates/factory and the adapter type `A` produced.
pub struct PlatformEntry<C, A> {
    /// Identifier used in config.yaml (e.g. "irc", "viber").
    pub name: String,

    /// Human-readable label (e.g. "IRC", "Viber").
    pub label: String,

    /// Factory callable: receives a config, returns an adapter instance.
    pub adapter_factory: AdapterFactory<C, A>,

    /// Returns `true` when the platform's dependencies are available.
    pub check_fn: CheckFn,

    /// Optional: given a config, is it properly configured? If `None`, the
    /// registry skips config validation and lets the adapter fail at
    /// connect() time with a descriptive error.
    pub validate_config: Option<ConfigPredicate<C>>,

    /// Optional: given a config, is the platform connected/enabled? Used by
    /// `GatewayConfig.get_connected_platforms()` and setup UI status. If
    /// `None`, falls back to `validate_config` or `check_fn`.
    pub is_connected: Option<ConfigPredicate<C>>,

    /// Env vars this platform needs (for `hermes setup` display).
    pub required_env: Vec<String>,

    /// Hint shown when `check_fn` returns `false`.
    pub install_hint: String,

    /// Optional setup function for interactive configuration.
    pub setup_fn: Option<SetupFn>,

    /// "builtin" or "plugin".
    pub source: String,

    /// Name of the plugin manifest that registered this entry (empty for
    /// built-ins).
    pub plugin_name: String,

    /// E.g. "IRC_ALLOWED_USERS" — checked for comma-separated user IDs.
    pub allowed_users_env: String,

    /// E.g. "IRC_ALLOW_ALL_USERS" — if truthy, all users authorized.
    pub allow_all_env: String,

    /// Max message length for smart-chunking. 0 = no limit.
    pub max_message_length: i64,

    /// If `true`, session descriptions redact PII (phone numbers, etc.).
    pub pii_safe: bool,

    /// Emoji for CLI/gateway display (e.g. "💬").
    pub emoji: String,

    /// Whether this platform should appear in `_UPDATE_ALLOWED_PLATFORMS`
    /// (allows `/update` command from this platform).
    pub allow_update_command: bool,

    /// Platform hint injected into the system prompt. Empty string = no hint.
    pub platform_hint: String,
}

impl<C, A> PlatformEntry<C, A> {
    /// Construct an entry with the required fields set and all optional fields
    /// at their Python dataclass defaults.
    ///
    /// Defaults mirror the Python dataclass:
    /// - `validate_config = None`, `is_connected = None`, `setup_fn = None`
    /// - `required_env = []`
    /// - `install_hint = ""`, `plugin_name = ""`, `allowed_users_env = ""`,
    ///   `allow_all_env = ""`, `platform_hint = ""`
    /// - `source = "plugin"`
    /// - `max_message_length = 0`
    /// - `pii_safe = false`
    /// - `emoji = "🔌"`
    /// - `allow_update_command = true`
    pub fn new<F, K>(name: impl Into<String>, label: impl Into<String>, adapter_factory: F, check_fn: K) -> Self
    where
        F: Fn(&C) -> Option<A> + Send + 'static,
        K: Fn() -> bool + Send + 'static,
    {
        PlatformEntry {
            name: name.into(),
            label: label.into(),
            adapter_factory: Box::new(adapter_factory),
            check_fn: Box::new(check_fn),
            validate_config: None,
            is_connected: None,
            required_env: Vec::new(),
            install_hint: String::new(),
            setup_fn: None,
            source: "plugin".to_string(),
            plugin_name: String::new(),
            allowed_users_env: String::new(),
            allow_all_env: String::new(),
            max_message_length: 0,
            pii_safe: false,
            emoji: "🔌".to_string(),
            allow_update_command: true,
            platform_hint: String::new(),
        }
    }
}

/// Central registry of platform adapters.
///
/// Mirrors the Python `PlatformRegistry`. The Python class relies on the GIL
/// for read safety and assumes writes happen at startup; in Rust, wrap the
/// registry in a `Mutex` (see [`platform_registry`]) for thread-safe access.
pub struct PlatformRegistry<C, A> {
    entries: HashMap<String, PlatformEntry<C, A>>,
}

impl<C, A> Default for PlatformRegistry<C, A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C, A> PlatformRegistry<C, A> {
    /// Create an empty registry.
    pub fn new() -> Self {
        PlatformRegistry {
            entries: HashMap::new(),
        }
    }

    /// Register a platform adapter entry.
    ///
    /// If an entry with the same name exists, it is replaced (last writer wins
    /// -- this lets plugins override built-in adapters if desired).
    pub fn register(&mut self, entry: PlatformEntry<C, A>) {
        if let Some(prev) = self.entries.get(&entry.name) {
            log::info!(
                "Platform '{}' re-registered (was {}, now {})",
                entry.name,
                prev.source,
                entry.source,
            );
        }
        log::debug!(
            "Registered platform adapter: {} ({})",
            entry.name,
            entry.source
        );
        self.entries.insert(entry.name.clone(), entry);
    }

    /// Remove a platform entry. Returns `true` if it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Look up a platform entry by name.
    pub fn get(&self, name: &str) -> Option<&PlatformEntry<C, A>> {
        self.entries.get(name)
    }

    /// Mutable lookup by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut PlatformEntry<C, A>> {
        self.entries.get_mut(name)
    }

    /// Return all registered platform entries.
    pub fn all_entries(&self) -> Vec<&PlatformEntry<C, A>> {
        self.entries.values().collect()
    }

    /// Return only plugin-registered platform entries.
    pub fn plugin_entries(&self) -> Vec<&PlatformEntry<C, A>> {
        self.entries
            .values()
            .filter(|e| e.source == "plugin")
            .collect()
    }

    /// Whether a platform with the given name is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Create an adapter instance for the given platform name.
    ///
    /// Returns `None` if:
    /// - No entry registered for `name`
    /// - `check_fn()` returns `false` (missing deps)
    /// - `validate_config()` returns `false` (misconfigured)
    /// - The factory fails to build the adapter
    pub fn create_adapter(&self, name: &str, config: &C) -> Option<A> {
        let entry = self.entries.get(name)?;

        if !(entry.check_fn)() {
            let hint = if entry.install_hint.is_empty() {
                String::new()
            } else {
                format!(" ({})", entry.install_hint)
            };
            log::warn!("Platform '{}' requirements not met{}", entry.label, hint);
            return None;
        }

        if let Some(validate) = &entry.validate_config {
            if !validate(config) {
                log::warn!("Platform '{}' config validation failed", entry.label);
                return None;
            }
        }

        match (entry.adapter_factory)(config) {
            Some(adapter) => Some(adapter),
            None => {
                log::error!("Failed to create adapter for platform '{}'", entry.label);
                None
            }
        }
    }
}

/// The dynamically-typed entry used by the module-level singleton, matching the
/// Python `platform_registry`'s ability to hold adapters/configs of any type.
pub type AnyPlatformEntry = PlatformEntry<Box<dyn Any + Send>, Box<dyn Any + Send>>;

/// The dynamically-typed registry used by the module-level singleton.
pub type AnyPlatformRegistry = PlatformRegistry<Box<dyn Any + Send>, Box<dyn Any + Send>>;

/// Module-level singleton, mirroring Python's module-level
/// `platform_registry = PlatformRegistry()`.
///
/// Wrapped in a `Mutex` for thread-safe access. Acquire the lock to register
/// or query entries.
pub fn platform_registry() -> &'static Mutex<AnyPlatformRegistry> {
    static REGISTRY: OnceLock<Mutex<AnyPlatformRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(PlatformRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple concrete config + adapter for testing the generic registry.
    #[derive(Clone)]
    struct TestConfig {
        server: Option<String>,
    }

    #[derive(Debug, PartialEq)]
    struct TestAdapter {
        name: String,
    }

    fn make_entry(name: &str, check: bool) -> PlatformEntry<TestConfig, TestAdapter> {
        let nm = name.to_string();
        PlatformEntry::new(
            name,
            name.to_uppercase(),
            move |_cfg: &TestConfig| {
                Some(TestAdapter {
                    name: nm.clone(),
                })
            },
            move || check,
        )
    }

    #[test]
    fn defaults_match_python_dataclass() {
        let e = make_entry("irc", true);
        assert_eq!(e.source, "plugin");
        assert_eq!(e.emoji, "🔌");
        assert!(e.allow_update_command);
        assert!(!e.pii_safe);
        assert_eq!(e.max_message_length, 0);
        assert!(e.install_hint.is_empty());
        assert!(e.required_env.is_empty());
        assert!(e.validate_config.is_none());
        assert!(e.is_connected.is_none());
        assert!(e.setup_fn.is_none());
        assert!(e.platform_hint.is_empty());
    }

    #[test]
    fn register_get_and_is_registered() {
        let mut reg = PlatformRegistry::new();
        assert!(!reg.is_registered("irc"));
        reg.register(make_entry("irc", true));
        assert!(reg.is_registered("irc"));
        assert_eq!(reg.get("irc").unwrap().label, "IRC");
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn register_last_writer_wins() {
        let mut reg = PlatformRegistry::new();
        let mut first = make_entry("irc", true);
        first.label = "First".into();
        reg.register(first);

        let mut second = make_entry("irc", true);
        second.label = "Second".into();
        reg.register(second);

        // Only one entry, the later one.
        assert_eq!(reg.all_entries().len(), 1);
        assert_eq!(reg.get("irc").unwrap().label, "Second");
    }

    #[test]
    fn unregister_returns_existence() {
        let mut reg = PlatformRegistry::new();
        reg.register(make_entry("irc", true));
        assert!(reg.unregister("irc"));
        assert!(!reg.unregister("irc"));
        assert!(!reg.is_registered("irc"));
    }

    #[test]
    fn plugin_entries_filters_by_source() {
        let mut reg = PlatformRegistry::new();
        reg.register(make_entry("irc", true)); // default source = plugin

        let mut builtin = make_entry("telegram", true);
        builtin.source = "builtin".into();
        reg.register(builtin);

        assert_eq!(reg.all_entries().len(), 2);
        let plugins = reg.plugin_entries();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "irc");
    }

    #[test]
    fn create_adapter_missing_returns_none() {
        let reg: PlatformRegistry<TestConfig, TestAdapter> = PlatformRegistry::new();
        let cfg = TestConfig { server: None };
        assert!(reg.create_adapter("nope", &cfg).is_none());
    }

    #[test]
    fn create_adapter_check_fn_false_returns_none() {
        let mut reg = PlatformRegistry::new();
        let mut e = make_entry("irc", false);
        e.install_hint = "pip install irc".into();
        reg.register(e);
        let cfg = TestConfig { server: None };
        assert!(reg.create_adapter("irc", &cfg).is_none());
    }

    #[test]
    fn create_adapter_validate_config_false_returns_none() {
        let mut reg = PlatformRegistry::new();
        let mut e = make_entry("irc", true);
        e.validate_config = Some(Box::new(|cfg: &TestConfig| cfg.server.is_some()));
        reg.register(e);

        let bad = TestConfig { server: None };
        assert!(reg.create_adapter("irc", &bad).is_none());

        let good = TestConfig {
            server: Some("chat.example".into()),
        };
        let adapter = reg.create_adapter("irc", &good);
        assert_eq!(adapter, Some(TestAdapter { name: "irc".into() }));
    }

    #[test]
    fn create_adapter_success_no_validator() {
        let mut reg = PlatformRegistry::new();
        reg.register(make_entry("irc", true));
        let cfg = TestConfig { server: None };
        let adapter = reg.create_adapter("irc", &cfg);
        assert_eq!(adapter, Some(TestAdapter { name: "irc".into() }));
    }

    #[test]
    fn create_adapter_factory_failure_returns_none() {
        let mut reg: PlatformRegistry<TestConfig, TestAdapter> = PlatformRegistry::new();
        let e = PlatformEntry::new(
            "broken",
            "Broken",
            |_cfg: &TestConfig| None, // factory "raises" -> None
            || true,
        );
        reg.register(e);
        let cfg = TestConfig { server: None };
        assert!(reg.create_adapter("broken", &cfg).is_none());
    }

    #[test]
    fn singleton_is_usable() {
        let reg = platform_registry();
        let mut guard = reg.lock().unwrap();
        let before = guard.all_entries().len();
        let entry: AnyPlatformEntry = PlatformEntry::new(
            "singleton_test_platform",
            "Singleton Test",
            |_cfg: &Box<dyn Any + Send>| {
                let b: Box<dyn Any + Send> = Box::new(123u32);
                Some(b)
            },
            || true,
        );
        guard.register(entry);
        assert!(guard.is_registered("singleton_test_platform"));
        assert_eq!(guard.all_entries().len(), before + 1);
        // cleanup so repeated test runs stay deterministic across this process
        guard.unregister("singleton_test_platform");
    }
}
