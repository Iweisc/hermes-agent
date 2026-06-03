//! Configurable budget constants for tool result persistence.
//!
//! Overridable at the RL environment level via `HermesAgentEnvConfig` fields.
//! Per-tool resolution: pinned > config overrides > registry > default.
//!
//! Faithful native Rust port of `tools/budget_config.py`.

use std::collections::HashMap;

use crate::tool_registry::ToolRegistry;

/// Default fallback for per-result persistence threshold, in characters.
///
/// Matches the hardcoded value in `tool_result_storage.py`. Kept here as the
/// single source of truth; the storage layer reads from these constants.
pub const DEFAULT_RESULT_SIZE_CHARS: i64 = 100_000;

/// Default aggregate per-turn char budget across all tool results.
pub const DEFAULT_TURN_BUDGET_CHARS: i64 = 200_000;

/// Default inline preview snippet size after persistence, in characters.
pub const DEFAULT_PREVIEW_SIZE_CHARS: i64 = 1_500;

/// Returns the table of tools whose thresholds must never be overridden.
///
/// `read_file = inf` prevents infinite persist -> read -> persist loops.
///
/// In Python this is a module-level `Dict[str, float]`. Because the only
/// pinned value is `+inf`, the map stores `f64` values.
pub fn pinned_thresholds() -> HashMap<&'static str, f64> {
    let mut m = HashMap::new();
    m.insert("read_file", f64::INFINITY);
    m
}

/// Looks up a pinned threshold for `tool_name`, if any.
pub fn pinned_threshold(tool_name: &str) -> Option<f64> {
    match tool_name {
        "read_file" => Some(f64::INFINITY),
        _ => None,
    }
}

/// Immutable budget constants for the 3-layer tool result persistence system.
///
/// * Layer 2 (per-result): [`BudgetConfig::resolve_threshold`] -> threshold in chars.
/// * Layer 3 (per-turn):   `turn_budget` -> aggregate char budget across all tool
///   results in a single assistant turn.
/// * Preview:              `preview_size` -> inline snippet size after persistence.
///
/// Mirrors the frozen Python `@dataclass`. Instances are intended to be treated
/// as immutable; clone instead of mutating in place.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetConfig {
    pub default_result_size: i64,
    pub turn_budget: i64,
    pub preview_size: i64,
    pub tool_overrides: HashMap<String, i64>,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            default_result_size: DEFAULT_RESULT_SIZE_CHARS,
            turn_budget: DEFAULT_TURN_BUDGET_CHARS,
            preview_size: DEFAULT_PREVIEW_SIZE_CHARS,
            tool_overrides: HashMap::new(),
        }
    }
}

impl BudgetConfig {
    /// Constructs a config matching the Python dataclass defaults.
    ///
    /// Equivalent to `BudgetConfig()` in Python.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves the persistence threshold for a tool, in characters.
    ///
    /// Priority: pinned -> `tool_overrides` -> registry per-tool -> default.
    ///
    /// The return value is an `f64` because pinned thresholds may be `+inf`
    /// (mirroring Python's `int | float` return type, where `read_file`
    /// returns `float("inf")`). Finite values are exact char counts.
    pub fn resolve_threshold(&self, tool_name: &str, registry: &ToolRegistry) -> f64 {
        if let Some(pinned) = pinned_threshold(tool_name) {
            return pinned;
        }
        if let Some(&over) = self.tool_overrides.get(tool_name) {
            return over as f64;
        }
        registry.get_max_result_size(tool_name, Some(self.default_result_size as f64))
    }
}

/// The default config -- matches current hardcoded behavior exactly.
///
/// Python exposes this as a module-level singleton `DEFAULT_BUDGET`. In Rust we
/// expose it as a function to avoid a non-`const`-constructible static
/// (`HashMap::new` is not `const` on stable across all toolchains).
pub fn default_budget() -> BudgetConfig {
    BudgetConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::{Handler, RegisterOptions, ToolRegistry};
    use serde_json::{json, Value};
    use std::sync::Arc;

    #[test]
    fn defaults_match_python() {
        let cfg = BudgetConfig::default();
        assert_eq!(cfg.default_result_size, 100_000);
        assert_eq!(cfg.turn_budget, 200_000);
        assert_eq!(cfg.preview_size, 1_500);
        assert!(cfg.tool_overrides.is_empty());
    }

    #[test]
    fn default_budget_helper_equals_default() {
        assert_eq!(default_budget(), BudgetConfig::default());
        assert_eq!(BudgetConfig::new(), BudgetConfig::default());
    }

    #[test]
    fn pinned_read_file_is_infinity() {
        assert_eq!(pinned_threshold("read_file"), Some(f64::INFINITY));
        assert_eq!(pinned_threshold("other_tool"), None);

        let table = pinned_thresholds();
        assert!(table["read_file"].is_infinite());
        assert!(table["read_file"].is_sign_positive());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn resolve_pinned_wins_over_everything() {
        let mut cfg = BudgetConfig::default();
        // Even with an override present, pinned takes priority.
        cfg.tool_overrides.insert("read_file".to_string(), 42);
        let reg = ToolRegistry::new();
        let t = cfg.resolve_threshold("read_file", &reg);
        assert!(t.is_infinite());
    }

    #[test]
    fn resolve_override_wins_over_registry_and_default() {
        let mut cfg = BudgetConfig::default();
        cfg.tool_overrides.insert("my_tool".to_string(), 555);
        let reg = ToolRegistry::new();
        assert_eq!(cfg.resolve_threshold("my_tool", &reg), 555.0);
    }

    #[test]
    fn resolve_falls_back_to_default_when_unknown() {
        let cfg = BudgetConfig::default();
        let reg = ToolRegistry::new();
        // Unregistered tool, no override -> default_result_size.
        assert_eq!(
            cfg.resolve_threshold("unregistered", &reg),
            100_000.0
        );
    }

    #[test]
    fn resolve_uses_registry_per_tool_value() {
        let cfg = BudgetConfig::default();
        let reg = ToolRegistry::new();
        let handler: Handler = Arc::new(|_a: &Value, _k: &Value| String::new());
        reg.register(
            RegisterOptions::new("sized_tool", "tset", json!({"name": "sized_tool"}), handler)
                .max_result_size_chars(Some(7_000.0)),
        );
        assert_eq!(cfg.resolve_threshold("sized_tool", &reg), 7_000.0);
    }
}
