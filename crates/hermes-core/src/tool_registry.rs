//! Central registry for all hermes-agent tools.
//!
//! Native Rust port of `tools/registry.py`. Each tool registers its schema,
//! handler, toolset membership, and availability check at startup. Higher-level
//! code queries the registry instead of maintaining parallel data structures.
//!
//! Key differences from the Python original (idiomatic adaptations that preserve
//! observable behaviour):
//!
//! * Handlers and check functions are boxed closures stored as trait objects
//!   ([`Handler`] / [`CheckFn`]).
//! * The `check_fn` TTL cache is keyed by a stable, per-entry id (the tool name's
//!   check) rather than by Python object identity, because Rust closures have no
//!   stable identity. The cache key is supplied at registration as the tool name;
//!   tools sharing a toolset-level check still get deduplicated within a single
//!   [`ToolRegistry::get_definitions`] pass via the per-call cache.
//! * `dispatch` mirrors the Python contract: unknown tools and handler panics /
//!   errors are serialised into `{"error": "..."}` JSON strings.
//!
//! Thread-safety: a single [`std::sync::Mutex`] guards all mutable state, exactly
//! like the Python `threading.RLock`. Readers take coherent snapshots.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

/// TTL for cached `check_fn` results. Matches the Python `_CHECK_FN_TTL_SECONDS`.
pub const CHECK_FN_TTL_SECONDS: f64 = 30.0;

/// A tool handler: takes the parsed arguments object plus arbitrary extra kwargs
/// (as a JSON object) and returns a JSON-encoded result string.
pub type Handler = Arc<dyn Fn(&Value, &Value) -> String + Send + Sync>;

/// A toolset/availability check. Returns `true` when the tool is usable.
pub type CheckFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Metadata for a single registered tool. Port of the Python `ToolEntry`.
#[derive(Clone)]
pub struct ToolEntry {
    pub name: String,
    pub toolset: String,
    pub schema: Value,
    pub handler: Handler,
    pub check_fn: Option<CheckFn>,
    pub requires_env: Vec<String>,
    pub is_async: bool,
    pub description: String,
    pub emoji: String,
    pub max_result_size_chars: Option<f64>,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("name", &self.name)
            .field("toolset", &self.toolset)
            .field("schema", &self.schema)
            .field("has_check_fn", &self.check_fn.is_some())
            .field("requires_env", &self.requires_env)
            .field("is_async", &self.is_async)
            .field("description", &self.description)
            .field("emoji", &self.emoji)
            .field("max_result_size_chars", &self.max_result_size_chars)
            .finish()
    }
}

/// Options for registering a tool. Mirrors the keyword arguments of the Python
/// `ToolRegistry.register`. All optional fields default to the Python defaults.
pub struct RegisterOptions {
    pub name: String,
    pub toolset: String,
    pub schema: Value,
    pub handler: Handler,
    pub check_fn: Option<CheckFn>,
    pub requires_env: Vec<String>,
    pub is_async: bool,
    pub description: String,
    pub emoji: String,
    pub max_result_size_chars: Option<f64>,
}

impl RegisterOptions {
    /// Construct minimal options with required fields; the rest take Python defaults.
    pub fn new(
        name: impl Into<String>,
        toolset: impl Into<String>,
        schema: Value,
        handler: Handler,
    ) -> Self {
        Self {
            name: name.into(),
            toolset: toolset.into(),
            schema,
            handler,
            check_fn: None,
            requires_env: Vec::new(),
            is_async: false,
            description: String::new(),
            emoji: String::new(),
            max_result_size_chars: None,
        }
    }

    pub fn check_fn(mut self, f: CheckFn) -> Self {
        self.check_fn = Some(f);
        self
    }
    pub fn requires_env(mut self, env: Vec<String>) -> Self {
        self.requires_env = env;
        self
    }
    pub fn is_async(mut self, v: bool) -> Self {
        self.is_async = v;
        self
    }
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
    pub fn emoji(mut self, e: impl Into<String>) -> Self {
        self.emoji = e.into();
        self
    }
    pub fn max_result_size_chars(mut self, v: Option<f64>) -> Self {
        self.max_result_size_chars = v;
        self
    }
}

/// Default fallback for `get_max_result_size` when no per-tool / caller default is
/// supplied. Mirrors `tools.budget_config.DEFAULT_RESULT_SIZE_CHARS`.
pub const DEFAULT_RESULT_SIZE_CHARS: f64 = 100_000.0;

/// Default emoji used by [`ToolRegistry::get_emoji`] when none is set.
pub const DEFAULT_EMOJI: &str = "\u{26a1}"; // ⚡

// ---------------------------------------------------------------------------
// check_fn TTL cache
//
// Keyed by the tool name (used as the cache identity). The Python version keyed
// on the callable object; here the natural stable identity is the tool name.
// ---------------------------------------------------------------------------

struct CheckFnCache {
    entries: HashMap<String, (Instant, bool)>,
}

impl CheckFnCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &str, now: Instant) -> Option<bool> {
        if let Some((ts, value)) = self.entries.get(key) {
            if now.duration_since(*ts) < Duration::from_secs_f64(CHECK_FN_TTL_SECONDS) {
                return Some(*value);
            }
        }
        None
    }

    fn put(&mut self, key: String, now: Instant, value: bool) {
        self.entries.insert(key, (now, value));
    }
}

/// Run `fn`, TTL-caching the boolean result against `key`. Swallows panics as
/// `false`, matching the Python `except Exception: value = False`.
fn check_fn_cached(cache: &Mutex<CheckFnCache>, key: &str, f: &CheckFn) -> bool {
    let now = Instant::now();
    {
        let guard = cache.lock().unwrap();
        if let Some(value) = guard.get(key, now) {
            return value;
        }
    }
    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f())).unwrap_or(false);
    cache.lock().unwrap().put(key.to_string(), now, value);
    value
}

// ---------------------------------------------------------------------------
// Toolset requirement / availability descriptors (mirror the dict shapes the
// Python helpers returned).
// ---------------------------------------------------------------------------

/// Per-toolset metadata for UI display. Port of `get_available_toolsets()` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolsetDisplay {
    pub available: bool,
    pub tools: Vec<String>,
    pub description: String,
    pub requirements: Vec<String>,
}

/// Backwards-compatible `TOOLSET_REQUIREMENTS` entry. Port of
/// `get_toolset_requirements()` values. `has_check_fn` replaces the Python
/// `check_fn` callable reference (callables aren't serialisable here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolsetRequirements {
    pub name: String,
    pub env_vars: Vec<String>,
    pub has_check_fn: bool,
    pub setup_url: Option<String>,
    pub tools: Vec<String>,
}

/// Info about a toolset whose requirements are not met. Port of the dicts in
/// `check_tool_availability()`'s `unavailable` list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnavailableToolset {
    pub name: String,
    pub env_vars: Vec<String>,
    pub tools: Vec<String>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

struct Inner {
    tools: HashMap<String, ToolEntry>,
    toolset_checks: HashMap<String, CheckFn>,
    toolset_aliases: HashMap<String, String>,
    generation: u64,
}

/// Singleton-style registry that collects tool schemas + handlers. Port of the
/// Python `ToolRegistry`. Cheap to `clone` (shares the same underlying state).
#[derive(Clone)]
pub struct ToolRegistry {
    inner: Arc<Mutex<Inner>>,
    check_cache: Arc<Mutex<CheckFnCache>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                tools: HashMap::new(),
                toolset_checks: HashMap::new(),
                toolset_aliases: HashMap::new(),
                generation: 0,
            })),
            check_cache: Arc::new(Mutex::new(CheckFnCache::new())),
        }
    }

    /// Current mutation generation counter; bumped on every mutation. External
    /// callers can memoise against it.
    pub fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// Drop all cached `check_fn` results. Call after config changes that affect
    /// tool availability. Mirrors module-level `invalidate_check_fn_cache`.
    pub fn invalidate_check_fn_cache(&self) {
        self.check_cache.lock().unwrap().entries.clear();
    }

    fn snapshot_entries(&self) -> Vec<ToolEntry> {
        self.inner.lock().unwrap().tools.values().cloned().collect()
    }

    fn snapshot_state(&self) -> (Vec<ToolEntry>, HashMap<String, CheckFn>) {
        let guard = self.inner.lock().unwrap();
        (
            guard.tools.values().cloned().collect(),
            guard.toolset_checks.clone(),
        )
    }

    fn evaluate_toolset_check(toolset: &str, check: Option<&CheckFn>) -> bool {
        match check {
            None => true,
            Some(c) => {
                let _ = toolset;
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c())).unwrap_or(false)
            }
        }
    }

    /// Return a registered tool entry by name, or `None`.
    pub fn get_entry(&self, name: &str) -> Option<ToolEntry> {
        self.inner.lock().unwrap().tools.get(name).cloned()
    }

    /// Sorted unique toolset names present in the registry.
    pub fn get_registered_toolset_names(&self) -> Vec<String> {
        let set: BTreeSet<String> = self
            .snapshot_entries()
            .into_iter()
            .map(|e| e.toolset)
            .collect();
        set.into_iter().collect()
    }

    /// Sorted tool names registered under a given toolset.
    pub fn get_tool_names_for_toolset(&self, toolset: &str) -> Vec<String> {
        let set: BTreeSet<String> = self
            .snapshot_entries()
            .into_iter()
            .filter(|e| e.toolset == toolset)
            .map(|e| e.name)
            .collect();
        set.into_iter().collect()
    }

    /// Register an explicit alias for a canonical toolset name.
    pub fn register_toolset_alias(&self, alias: &str, toolset: &str) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(existing) = guard.toolset_aliases.get(alias) {
            if existing != toolset {
                log::warn!(
                    "Toolset alias collision: '{}' ({}) overwritten by {}",
                    alias,
                    existing,
                    toolset
                );
            }
        }
        guard
            .toolset_aliases
            .insert(alias.to_string(), toolset.to_string());
        guard.generation += 1;
    }

    /// Snapshot of `{alias: canonical_toolset}` mappings.
    pub fn get_registered_toolset_aliases(&self) -> HashMap<String, String> {
        self.inner.lock().unwrap().toolset_aliases.clone()
    }

    /// Canonical toolset name for an alias, or `None`.
    pub fn get_toolset_alias_target(&self, alias: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .toolset_aliases
            .get(alias)
            .cloned()
    }

    // ------------------------------------------------------------------
    // Registration
    // ------------------------------------------------------------------

    /// Register a tool. Called at startup by each tool module.
    ///
    /// Returns `true` if the registration was applied, `false` if it was rejected
    /// because it would shadow an existing tool from a different (non-MCP) toolset.
    pub fn register(&self, opts: RegisterOptions) -> bool {
        let mut guard = self.inner.lock().unwrap();
        if let Some(existing) = guard.tools.get(&opts.name) {
            if existing.toolset != opts.toolset {
                let both_mcp =
                    existing.toolset.starts_with("mcp-") && opts.toolset.starts_with("mcp-");
                if both_mcp {
                    log::debug!(
                        "Tool '{}': MCP toolset '{}' overwriting MCP toolset '{}'",
                        opts.name,
                        opts.toolset,
                        existing.toolset
                    );
                } else {
                    log::error!(
                        "Tool registration REJECTED: '{}' (toolset '{}') would shadow \
                         existing tool from toolset '{}'. Deregister the existing tool \
                         first if this is intentional.",
                        opts.name,
                        opts.toolset,
                        existing.toolset
                    );
                    return false;
                }
            }
        }

        // description default: `description or schema.get("description", "")`
        let description = if !opts.description.is_empty() {
            opts.description.clone()
        } else {
            opts.schema
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };

        let entry = ToolEntry {
            name: opts.name.clone(),
            toolset: opts.toolset.clone(),
            schema: opts.schema,
            handler: opts.handler,
            check_fn: opts.check_fn.clone(),
            requires_env: opts.requires_env,
            is_async: opts.is_async,
            description,
            emoji: opts.emoji,
            max_result_size_chars: opts.max_result_size_chars,
        };

        guard.tools.insert(opts.name.clone(), entry);

        if let Some(check) = opts.check_fn {
            guard
                .toolset_checks
                .entry(opts.toolset.clone())
                .or_insert(check);
        }
        guard.generation += 1;
        true
    }

    /// Remove a tool from the registry, cleaning up toolset check + aliases when
    /// it was the last tool in its toolset.
    pub fn deregister(&self, name: &str) {
        {
            let mut guard = self.inner.lock().unwrap();
            let entry = match guard.tools.remove(name) {
                Some(e) => e,
                None => return,
            };
            let toolset_still_exists =
                guard.tools.values().any(|e| e.toolset == entry.toolset);
            if !toolset_still_exists {
                guard.toolset_checks.remove(&entry.toolset);
                guard
                    .toolset_aliases
                    .retain(|_, target| *target != entry.toolset);
            }
            guard.generation += 1;
        }
        log::debug!("Deregistered tool: {}", name);
    }

    // ------------------------------------------------------------------
    // Schema retrieval
    // ------------------------------------------------------------------

    /// Return OpenAI-format tool schemas for the requested tool names. Only tools
    /// whose `check_fn()` returns `true` (or which have none) are included.
    /// Results are TTL-cached for ~30 s.
    pub fn get_definitions(&self, tool_names: &BTreeSet<String>, quiet: bool) -> Vec<Value> {
        let mut result = Vec::new();
        // Per-call cache keyed by tool name, layered on top of the TTL cache.
        let mut check_results: HashMap<String, bool> = HashMap::new();
        let entries = self.snapshot_entries();
        let by_name: HashMap<String, ToolEntry> =
            entries.into_iter().map(|e| (e.name.clone(), e)).collect();

        // sorted(tool_names) — BTreeSet already iterates in sorted order.
        for name in tool_names {
            let entry = match by_name.get(name) {
                Some(e) => e,
                None => continue,
            };
            if let Some(check) = &entry.check_fn {
                let ok = *check_results.entry(name.clone()).or_insert_with(|| {
                    check_fn_cached(&self.check_cache, name, check)
                });
                if !ok {
                    if !quiet {
                        log::debug!("Tool {} unavailable (check failed)", name);
                    }
                    continue;
                }
            }
            // Ensure schema always has a "name" field — use entry.name as fallback.
            let mut schema_with_name = match &entry.schema {
                Value::Object(m) => m.clone(),
                other => {
                    let mut m = Map::new();
                    if !other.is_null() {
                        m.insert("_schema".to_string(), other.clone());
                    }
                    m
                }
            };
            schema_with_name.insert("name".to_string(), Value::String(entry.name.clone()));
            result.push(json!({
                "type": "function",
                "function": Value::Object(schema_with_name),
            }));
        }
        result
    }

    // ------------------------------------------------------------------
    // Dispatch
    // ------------------------------------------------------------------

    /// Execute a tool handler by name. Unknown tools and handler panics are
    /// serialised to `{"error": "..."}`. The `kwargs` value is passed through to
    /// the handler (Python `**kwargs`).
    ///
    /// Note: async handler bridging (`is_async`) is the caller's responsibility in
    /// the Rust runtime; all registered handlers here are synchronous closures.
    pub fn dispatch(&self, name: &str, args: &Value, kwargs: &Value) -> String {
        let entry = match self.get_entry(name) {
            Some(e) => e,
            None => return json!({ "error": format!("Unknown tool: {}", name) }).to_string(),
        };
        let handler = entry.handler.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handler(args, kwargs)
        }));
        match result {
            Ok(s) => s,
            Err(_) => {
                log::error!("Tool {} dispatch error", name);
                json!({ "error": format!("Tool execution failed: {}", name) }).to_string()
            }
        }
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Per-tool max result size, or `default` (or the global default).
    pub fn get_max_result_size(&self, name: &str, default: Option<f64>) -> f64 {
        if let Some(entry) = self.get_entry(name) {
            if let Some(v) = entry.max_result_size_chars {
                return v;
            }
        }
        default.unwrap_or(DEFAULT_RESULT_SIZE_CHARS)
    }

    /// Sorted list of all registered tool names.
    pub fn get_all_tool_names(&self) -> Vec<String> {
        let set: BTreeSet<String> =
            self.snapshot_entries().into_iter().map(|e| e.name).collect();
        set.into_iter().collect()
    }

    /// Return a tool's raw schema dict, bypassing check_fn filtering.
    pub fn get_schema(&self, name: &str) -> Option<Value> {
        self.get_entry(name).map(|e| e.schema)
    }

    /// Return the toolset a tool belongs to, or `None`.
    pub fn get_toolset_for_tool(&self, name: &str) -> Option<String> {
        self.get_entry(name).map(|e| e.toolset)
    }

    /// Return the emoji for a tool, or `default` if unset.
    pub fn get_emoji(&self, name: &str, default: &str) -> String {
        match self.get_entry(name) {
            Some(e) if !e.emoji.is_empty() => e.emoji,
            _ => default.to_string(),
        }
    }

    /// Return `{tool_name: toolset_name}` for every registered tool.
    pub fn get_tool_to_toolset_map(&self) -> HashMap<String, String> {
        self.snapshot_entries()
            .into_iter()
            .map(|e| (e.name, e.toolset))
            .collect()
    }

    /// Check if a toolset's requirements are met. Returns `false` on check panic.
    pub fn is_toolset_available(&self, toolset: &str) -> bool {
        let check = self.inner.lock().unwrap().toolset_checks.get(toolset).cloned();
        Self::evaluate_toolset_check(toolset, check.as_ref())
    }

    /// Return `{toolset: available_bool}` for every toolset.
    pub fn check_toolset_requirements(&self) -> BTreeMap<String, bool> {
        let (entries, toolset_checks) = self.snapshot_state();
        let toolsets: BTreeSet<String> = entries.into_iter().map(|e| e.toolset).collect();
        toolsets
            .into_iter()
            .map(|ts| {
                let avail = Self::evaluate_toolset_check(&ts, toolset_checks.get(&ts));
                (ts, avail)
            })
            .collect()
    }

    /// Return toolset metadata for UI display. Insertion order follows entry
    /// iteration order, like the Python dict.
    pub fn get_available_toolsets(&self) -> HashMap<String, ToolsetDisplay> {
        let (entries, toolset_checks) = self.snapshot_state();
        let mut toolsets: HashMap<String, ToolsetDisplay> = HashMap::new();
        for entry in &entries {
            let ts = entry.toolset.clone();
            let disp = toolsets.entry(ts.clone()).or_insert_with(|| ToolsetDisplay {
                available: Self::evaluate_toolset_check(&ts, toolset_checks.get(&ts)),
                tools: Vec::new(),
                description: String::new(),
                requirements: Vec::new(),
            });
            disp.tools.push(entry.name.clone());
            for env in &entry.requires_env {
                if !disp.requirements.contains(env) {
                    disp.requirements.push(env.clone());
                }
            }
        }
        toolsets
    }

    /// Build a `TOOLSET_REQUIREMENTS`-compatible map for backward compat.
    pub fn get_toolset_requirements(&self) -> HashMap<String, ToolsetRequirements> {
        let (entries, toolset_checks) = self.snapshot_state();
        let mut result: HashMap<String, ToolsetRequirements> = HashMap::new();
        for entry in &entries {
            let ts = entry.toolset.clone();
            let req = result.entry(ts.clone()).or_insert_with(|| ToolsetRequirements {
                name: ts.clone(),
                env_vars: Vec::new(),
                has_check_fn: toolset_checks.contains_key(&ts),
                setup_url: None,
                tools: Vec::new(),
            });
            if !req.tools.contains(&entry.name) {
                req.tools.push(entry.name.clone());
            }
            for env in &entry.requires_env {
                if !req.env_vars.contains(env) {
                    req.env_vars.push(env.clone());
                }
            }
        }
        result
    }

    /// Return `(available_toolsets, unavailable_info)`. Toolsets are processed in
    /// first-seen entry order, matching the Python iteration.
    pub fn check_tool_availability(&self) -> (Vec<String>, Vec<UnavailableToolset>) {
        let mut available = Vec::new();
        let mut unavailable = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let (entries, toolset_checks) = self.snapshot_state();
        for entry in &entries {
            let ts = entry.toolset.clone();
            if seen.contains(&ts) {
                continue;
            }
            seen.insert(ts.clone());
            if Self::evaluate_toolset_check(&ts, toolset_checks.get(&ts)) {
                available.push(ts);
            } else {
                unavailable.push(UnavailableToolset {
                    name: ts.clone(),
                    env_vars: entry.requires_env.clone(),
                    tools: entries
                        .iter()
                        .filter(|e| e.toolset == ts)
                        .map(|e| e.name.clone())
                        .collect(),
                });
            }
        }
        (available, unavailable)
    }
}

// ---------------------------------------------------------------------------
// Helpers for tool response serialization (module-level, like the Python ones).
// ---------------------------------------------------------------------------

/// Return a JSON error string for tool handlers. `extra` fields are merged in.
///
/// ```
/// # use hermes_core::tool_registry::tool_error;
/// # use serde_json::json;
/// assert_eq!(tool_error("file not found", None), r#"{"error":"file not found"}"#);
/// ```
pub fn tool_error(message: impl AsRef<str>, extra: Option<Value>) -> String {
    let mut map = Map::new();
    map.insert(
        "error".to_string(),
        Value::String(message.as_ref().to_string()),
    );
    if let Some(Value::Object(extra_map)) = extra {
        for (k, v) in extra_map {
            map.insert(k, v);
        }
    }
    Value::Object(map).to_string()
}

/// Return a JSON result string for tool handlers from a data value.
pub fn tool_result(data: Value) -> String {
    data.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_handler() -> Handler {
        Arc::new(|_args: &Value, _kw: &Value| "{}".to_string())
    }

    fn opts(name: &str, toolset: &str) -> RegisterOptions {
        RegisterOptions::new(
            name,
            toolset,
            json!({"description": "d", "parameters": {}}),
            noop_handler(),
        )
    }

    #[test]
    fn register_and_lookup() {
        let reg = ToolRegistry::new();
        assert!(reg.register(opts("read", "fs")));
        let e = reg.get_entry("read").unwrap();
        assert_eq!(e.toolset, "fs");
        // description falls back to schema description
        assert_eq!(e.description, "d");
        assert_eq!(reg.get_all_tool_names(), vec!["read".to_string()]);
        assert_eq!(reg.get_toolset_for_tool("read"), Some("fs".to_string()));
    }

    #[test]
    fn explicit_description_wins() {
        let reg = ToolRegistry::new();
        reg.register(opts("read", "fs").description("explicit"));
        assert_eq!(reg.get_entry("read").unwrap().description, "explicit");
    }

    #[test]
    fn generation_bumps_on_mutation() {
        let reg = ToolRegistry::new();
        let g0 = reg.generation();
        reg.register(opts("a", "x"));
        let g1 = reg.generation();
        assert!(g1 > g0);
        reg.register_toolset_alias("alias", "x");
        assert!(reg.generation() > g1);
    }

    #[test]
    fn shadowing_rejected_for_non_mcp() {
        let reg = ToolRegistry::new();
        assert!(reg.register(opts("t", "builtin")));
        // different toolset, non-mcp -> rejected
        assert!(!reg.register(opts("t", "plugin")));
        assert_eq!(reg.get_entry("t").unwrap().toolset, "builtin");
    }

    #[test]
    fn mcp_to_mcp_overwrite_allowed() {
        let reg = ToolRegistry::new();
        assert!(reg.register(opts("t", "mcp-server-a")));
        assert!(reg.register(opts("t", "mcp-server-b")));
        assert_eq!(reg.get_entry("t").unwrap().toolset, "mcp-server-b");
    }

    #[test]
    fn same_toolset_overwrite_allowed() {
        let reg = ToolRegistry::new();
        assert!(reg.register(opts("t", "fs")));
        assert!(reg.register(opts("t", "fs").description("v2")));
        assert_eq!(reg.get_entry("t").unwrap().description, "v2");
    }

    #[test]
    fn deregister_cleans_toolset_check_and_aliases() {
        let reg = ToolRegistry::new();
        let check: CheckFn = Arc::new(|| true);
        reg.register(opts("only", "ts").check_fn(check));
        reg.register_toolset_alias("al", "ts");
        assert!(reg.is_toolset_available("ts"));
        assert_eq!(reg.get_toolset_alias_target("al"), Some("ts".to_string()));
        reg.deregister("only");
        // toolset check and alias gone
        assert!(reg.is_toolset_available("ts")); // no check => available
        assert_eq!(reg.get_toolset_alias_target("al"), None);
        assert!(reg.get_entry("only").is_none());
    }

    #[test]
    fn deregister_keeps_check_if_other_tool_remains() {
        let reg = ToolRegistry::new();
        let check: CheckFn = Arc::new(|| false);
        reg.register(opts("a", "ts").check_fn(check.clone()));
        reg.register(opts("b", "ts"));
        reg.deregister("a");
        // still has the check from the first registration -> unavailable
        assert!(!reg.is_toolset_available("ts"));
    }

    #[test]
    fn get_definitions_filters_unavailable() {
        let reg = ToolRegistry::new();
        let yes: CheckFn = Arc::new(|| true);
        let no: CheckFn = Arc::new(|| false);
        reg.register(opts("ok", "ts1").check_fn(yes));
        reg.register(opts("blocked", "ts2").check_fn(no));
        reg.register(opts("free", "ts3")); // no check

        let names: BTreeSet<String> =
            ["ok", "blocked", "free", "missing"].iter().map(|s| s.to_string()).collect();
        let defs = reg.get_definitions(&names, true);
        let got: BTreeSet<String> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            got,
            ["free", "ok"].iter().map(|s| s.to_string()).collect()
        );
        // schema carries name + original fields
        let okdef = defs
            .iter()
            .find(|d| d["function"]["name"] == "ok")
            .unwrap();
        assert_eq!(okdef["type"], "function");
        assert_eq!(okdef["function"]["description"], "d");
    }

    #[test]
    fn check_fn_ttl_cache_caches_first_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let reg = ToolRegistry::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let check: CheckFn = Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            true
        });
        reg.register(opts("x", "ts").check_fn(check));
        let names: BTreeSet<String> = ["x"].iter().map(|s| s.to_string()).collect();
        reg.get_definitions(&names, true);
        reg.get_definitions(&names, true);
        // cached across calls within TTL
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        reg.invalidate_check_fn_cache();
        reg.get_definitions(&names, true);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn dispatch_unknown_tool() {
        let reg = ToolRegistry::new();
        let out = reg.dispatch("nope", &json!({}), &json!({}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], "Unknown tool: nope");
    }

    #[test]
    fn dispatch_runs_handler() {
        let reg = ToolRegistry::new();
        let handler: Handler =
            Arc::new(|args: &Value, _kw: &Value| json!({"echo": args["v"]}).to_string());
        reg.register(RegisterOptions::new("echo", "ts", json!({}), handler));
        let out = reg.dispatch("echo", &json!({"v": 5}), &json!({}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["echo"], 5);
    }

    #[test]
    fn dispatch_panic_becomes_error() {
        let reg = ToolRegistry::new();
        let handler: Handler = Arc::new(|_a: &Value, _k: &Value| panic!("boom"));
        reg.register(RegisterOptions::new("bad", "ts", json!({}), handler));
        let out = reg.dispatch("bad", &json!({}), &json!({}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("Tool execution failed"));
    }

    #[test]
    fn max_result_size_resolution() {
        let reg = ToolRegistry::new();
        reg.register(opts("a", "ts").max_result_size_chars(Some(42.0)));
        reg.register(opts("b", "ts"));
        assert_eq!(reg.get_max_result_size("a", None), 42.0);
        assert_eq!(reg.get_max_result_size("b", Some(7.0)), 7.0);
        assert_eq!(reg.get_max_result_size("b", None), DEFAULT_RESULT_SIZE_CHARS);
        assert_eq!(reg.get_max_result_size("missing", None), DEFAULT_RESULT_SIZE_CHARS);
    }

    #[test]
    fn emoji_default() {
        let reg = ToolRegistry::new();
        reg.register(opts("a", "ts").emoji("\u{1f600}"));
        reg.register(opts("b", "ts"));
        assert_eq!(reg.get_emoji("a", DEFAULT_EMOJI), "\u{1f600}");
        assert_eq!(reg.get_emoji("b", DEFAULT_EMOJI), DEFAULT_EMOJI);
        assert_eq!(reg.get_emoji("missing", DEFAULT_EMOJI), DEFAULT_EMOJI);
    }

    #[test]
    fn alias_collision_overwrites() {
        let reg = ToolRegistry::new();
        reg.register_toolset_alias("al", "ts1");
        reg.register_toolset_alias("al", "ts2");
        assert_eq!(reg.get_toolset_alias_target("al"), Some("ts2".to_string()));
        let aliases = reg.get_registered_toolset_aliases();
        assert_eq!(aliases.get("al"), Some(&"ts2".to_string()));
    }

    #[test]
    fn toolset_queries() {
        let reg = ToolRegistry::new();
        reg.register(opts("b", "ts1").requires_env(vec!["KEY".to_string()]));
        reg.register(opts("a", "ts1"));
        reg.register(opts("c", "ts2"));
        assert_eq!(
            reg.get_registered_toolset_names(),
            vec!["ts1".to_string(), "ts2".to_string()]
        );
        assert_eq!(
            reg.get_tool_names_for_toolset("ts1"),
            vec!["a".to_string(), "b".to_string()]
        );
        let reqs = reg.get_toolset_requirements();
        assert_eq!(reqs["ts1"].env_vars, vec!["KEY".to_string()]);
        let map = reg.get_tool_to_toolset_map();
        assert_eq!(map["c"], "ts2");
    }

    #[test]
    fn check_tool_availability_split() {
        let reg = ToolRegistry::new();
        let no: CheckFn = Arc::new(|| false);
        reg.register(opts("good", "ts_ok"));
        reg.register(opts("bad", "ts_bad").requires_env(vec!["X".into()]).check_fn(no));
        let (avail, unavail) = reg.check_tool_availability();
        assert!(avail.contains(&"ts_ok".to_string()));
        assert_eq!(unavail.len(), 1);
        assert_eq!(unavail[0].name, "ts_bad");
        assert_eq!(unavail[0].env_vars, vec!["X".to_string()]);
    }

    #[test]
    fn tool_error_and_result() {
        assert_eq!(tool_error("nope", None), r#"{"error":"nope"}"#);
        let e = tool_error("bad", Some(json!({"code": 404})));
        let v: Value = serde_json::from_str(&e).unwrap();
        assert_eq!(v["error"], "bad");
        assert_eq!(v["code"], 404);
        assert_eq!(tool_result(json!({"k": "v"})), r#"{"k":"v"}"#);
    }

    #[test]
    fn available_toolsets_metadata() {
        let reg = ToolRegistry::new();
        reg.register(opts("a", "ts").requires_env(vec!["E1".into(), "E2".into()]));
        reg.register(opts("b", "ts").requires_env(vec!["E1".into()]));
        let ts = reg.get_available_toolsets();
        let entry = &ts["ts"];
        assert!(entry.available); // no check_fn
        assert_eq!(entry.tools.len(), 2);
        assert_eq!(entry.requirements, vec!["E1".to_string(), "E2".to_string()]);
    }
}
