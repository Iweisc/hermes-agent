//! Image Generation Provider Registry - faithful port of
//! `agent/image_gen_registry.py`.
//!
//! Central map of registered providers. Populated by plugins at import-time via
//! `PluginContext.register_image_gen_provider()`; consumed by the
//! `image_generate` tool to dispatch each call to the active backend.
//!
//! # Active selection
//! The active provider is chosen by `image_gen.provider` in `config.yaml`.
//! In this Rust port the configured value is supplied to
//! [`get_active_provider`] as a parameter (the caller reads it from config),
//! keeping the registry decoupled from config loading. When unset, the same
//! fallback logic as the Python module applies:
//!
//! 1. If exactly one provider is registered, use it.
//! 2. Otherwise if a provider named `fal` is registered, use it (legacy
//!    default - matches pre-plugin behavior).
//! 3. Otherwise return `None`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::agent_image_gen_provider::ImageGenProvider;

/// A thread-shareable provider handle. The registry stores trait objects so
/// any backend implementing [`ImageGenProvider`] can be registered.
pub type ProviderHandle = Arc<dyn ImageGenProvider + Send + Sync>;

/// Process-wide registry of image generation providers, keyed by name.
fn providers() -> &'static Mutex<BTreeMap<String, ProviderHandle>> {
    static PROVIDERS: OnceLock<Mutex<BTreeMap<String, ProviderHandle>>> = OnceLock::new();
    PROVIDERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Error type mirroring the Python `TypeError`/`ValueError` raised by
/// `register_provider`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterError {
    /// The provider's `.name` was empty or whitespace-only.
    EmptyName,
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterError::EmptyName => {
                write!(f, "Image gen provider .name must be a non-empty string")
            }
        }
    }
}

impl std::error::Error for RegisterError {}

/// Register an image generation provider.
///
/// Re-registration (same `name`) overwrites the previous entry and logs a
/// debug message - this makes hot-reload scenarios (tests, dev loops) behave
/// predictably.
///
/// Returns [`RegisterError::EmptyName`] when the provider's name is empty or
/// whitespace-only (mirrors the Python `ValueError`). The type-check from the
/// Python implementation is enforced statically by the trait bound here.
pub fn register_provider(provider: ProviderHandle) -> Result<(), RegisterError> {
    let name = provider.name();
    if name.trim().is_empty() {
        return Err(RegisterError::EmptyName);
    }

    let existed = {
        let mut map = providers().lock().expect("provider registry poisoned");
        let existing = map.remove(&name);
        map.insert(name.clone(), provider);
        existing.is_some()
    };

    if existed {
        log::debug!("Image gen provider '{name}' re-registered");
    } else {
        log::debug!("Registered image gen provider '{name}'");
    }
    Ok(())
}

/// Return all registered providers, sorted by name.
///
/// The backing store is a `BTreeMap` keyed by name, so iteration order is
/// already sorted - matching the Python `sorted(items, key=lambda p: p.name)`.
pub fn list_providers() -> Vec<ProviderHandle> {
    let map = providers().lock().expect("provider registry poisoned");
    map.values().cloned().collect()
}

/// Return the provider registered under `name` (trimmed), or `None`.
pub fn get_provider(name: &str) -> Option<ProviderHandle> {
    let map = providers().lock().expect("provider registry poisoned");
    map.get(name.trim()).cloned()
}

/// Resolve the currently-active provider.
///
/// `configured` is the value of `image_gen.provider` read from config by the
/// caller (already trimmed/normalised, or `None` when unset/blank). Fallback
/// logic mirrors the module docs:
///
/// 1. If `configured` names a registered provider, return it.
/// 2. Otherwise, if exactly one provider is registered, return it.
/// 3. Otherwise, if a provider named `fal` is registered, return it.
/// 4. Otherwise return `None`.
pub fn get_active_provider(configured: Option<&str>) -> Option<ProviderHandle> {
    let snapshot: BTreeMap<String, ProviderHandle> = {
        let map = providers().lock().expect("provider registry poisoned");
        map.clone()
    };

    // Normalise the configured value the way the Python loader does: a value is
    // only meaningful when it is a non-empty (after trimming) string.
    let configured = configured.map(str::trim).filter(|s| !s.is_empty());

    if let Some(name) = configured {
        if let Some(provider) = snapshot.get(name) {
            return Some(provider.clone());
        }
        log::debug!(
            "image_gen.provider='{name}' configured but not registered; falling back"
        );
    }

    // Fallback: single-provider case.
    if snapshot.len() == 1 {
        return snapshot.values().next().cloned();
    }

    // Fallback: prefer legacy FAL for backward compat.
    if let Some(provider) = snapshot.get("fal") {
        return Some(provider.clone());
    }

    None
}

/// Clear the registry. **Test-only.**
pub fn reset_for_tests() {
    let mut map = providers().lock().expect("provider registry poisoned");
    map.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_image_gen_provider::DEFAULT_ASPECT_RATIO;
    use serde_json::{Map, Value};
    use std::sync::{Mutex as StdMutex, MutexGuard};

    // The registry is process-global; serialise tests so they don't clobber
    // each other's state.
    static TEST_GUARD: StdMutex<()> = StdMutex::new(());

    fn guard() -> MutexGuard<'static, ()> {
        let g = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_tests();
        g
    }

    struct FakeProvider {
        name: String,
    }

    impl FakeProvider {
        fn handle(name: &str) -> ProviderHandle {
            Arc::new(FakeProvider {
                name: name.to_string(),
            })
        }
    }

    impl ImageGenProvider for FakeProvider {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn generate(
            &self,
            prompt: &str,
            aspect_ratio: &str,
            _kwargs: &Map<String, Value>,
        ) -> Value {
            let mut map = Map::new();
            map.insert("success".to_string(), Value::Bool(true));
            map.insert("prompt".to_string(), Value::String(prompt.to_string()));
            map.insert(
                "aspect_ratio".to_string(),
                Value::String(aspect_ratio.to_string()),
            );
            map.insert("provider".to_string(), Value::String(self.name.clone()));
            Value::Object(map)
        }
    }

    #[test]
    fn register_rejects_empty_name() {
        let _g = guard();
        let err = register_provider(FakeProvider::handle("   ")).unwrap_err();
        assert_eq!(err, RegisterError::EmptyName);
        assert!(list_providers().is_empty());
    }

    #[test]
    fn register_and_get_roundtrip() {
        let _g = guard();
        register_provider(FakeProvider::handle("openai")).unwrap();
        let got = get_provider("openai").expect("provider present");
        assert_eq!(got.name(), "openai");
        // Trimming applies on lookup.
        assert!(get_provider("  openai  ").is_some());
        assert!(get_provider("missing").is_none());
    }

    #[test]
    fn reregister_overwrites() {
        let _g = guard();
        register_provider(FakeProvider::handle("fal")).unwrap();
        register_provider(FakeProvider::handle("fal")).unwrap();
        assert_eq!(list_providers().len(), 1);
    }

    #[test]
    fn list_providers_sorted_by_name() {
        let _g = guard();
        register_provider(FakeProvider::handle("zeta")).unwrap();
        register_provider(FakeProvider::handle("alpha")).unwrap();
        register_provider(FakeProvider::handle("mid")).unwrap();
        let names: Vec<String> = list_providers().iter().map(|p| p.name()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn active_prefers_configured() {
        let _g = guard();
        register_provider(FakeProvider::handle("a")).unwrap();
        register_provider(FakeProvider::handle("b")).unwrap();
        let active = get_active_provider(Some("b")).expect("active");
        assert_eq!(active.name(), "b");
        // Whitespace-padded config value is trimmed.
        assert_eq!(get_active_provider(Some("  a  ")).unwrap().name(), "a");
    }

    #[test]
    fn active_falls_back_to_single_provider() {
        let _g = guard();
        register_provider(FakeProvider::handle("only")).unwrap();
        // Unset config.
        assert_eq!(get_active_provider(None).unwrap().name(), "only");
        // Configured-but-missing also falls through to the single provider.
        assert_eq!(get_active_provider(Some("nope")).unwrap().name(), "only");
        // Blank config behaves like unset.
        assert_eq!(get_active_provider(Some("   ")).unwrap().name(), "only");
    }

    #[test]
    fn active_falls_back_to_fal_legacy() {
        let _g = guard();
        register_provider(FakeProvider::handle("openai")).unwrap();
        register_provider(FakeProvider::handle("fal")).unwrap();
        register_provider(FakeProvider::handle("replicate")).unwrap();
        // No config, multiple providers -> legacy fal.
        assert_eq!(get_active_provider(None).unwrap().name(), "fal");
    }

    #[test]
    fn active_none_when_ambiguous_no_fal() {
        let _g = guard();
        register_provider(FakeProvider::handle("openai")).unwrap();
        register_provider(FakeProvider::handle("replicate")).unwrap();
        assert!(get_active_provider(None).is_none());
        assert!(get_active_provider(Some("missing")).is_none());
    }

    #[test]
    fn active_none_when_empty_registry() {
        let _g = guard();
        assert!(get_active_provider(None).is_none());
    }

    #[test]
    fn generate_passes_through() {
        let _g = guard();
        register_provider(FakeProvider::handle("fal")).unwrap();
        let p = get_active_provider(None).unwrap();
        let out = p.generate("a cat", DEFAULT_ASPECT_RATIO, &Map::new());
        assert_eq!(out["prompt"], "a cat");
        assert_eq!(out["aspect_ratio"], DEFAULT_ASPECT_RATIO);
        assert_eq!(out["success"], true);
    }
}
