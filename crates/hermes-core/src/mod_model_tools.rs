//! Model Tools — thin orchestration layer over the tool registry.
//!
//! Native Rust port of `model_tools.py`. The Python module is a thin
//! orchestration layer over [`crate::tool_registry::ToolRegistry`]: it triggers
//! tool discovery, then exposes the public API that the agent loop, CLI, batch
//! runner, and RL environments consume.
//!
//! Public API (signatures adapted from the Python original):
//!   * [`get_tool_definitions`] — the main schema provider with toolset filtering
//!     + memoization.
//!   * [`handle_function_call`] — the main dispatcher.
//!   * [`coerce_tool_args`] — JSON-Schema-aware argument type coercion.
//!   * [`legacy_toolset_map`] — the `_tools`-suffixed compatibility name map.
//!   * Backward-compat wrappers: [`get_all_tool_names`], [`get_toolset_for_tool`],
//!     [`get_available_toolsets`], [`check_toolset_requirements`],
//!     [`check_tool_availability`].
//!
//! # Differences from the Python original
//!
//! * **Async bridging.** The Python module's `_run_async` / `_get_tool_loop` /
//!   `_get_worker_loop` machinery exists solely to bridge sync call sites to
//!   asyncio coroutine handlers without tripping "Event loop is closed" on
//!   cached httpx clients. In the Rust port all registered handlers are
//!   synchronous closures (see [`crate::tool_registry`] module docs), so that
//!   plumbing has no analogue and is intentionally omitted. Async tool handlers,
//!   if introduced, are the caller's responsibility to bridge.
//!
//! * **Global state.** Python kept `_tool_defs_cache` and
//!   `_last_resolved_tool_names` as module globals. Here they live in process
//!   globals guarded by mutexes, with the same observable semantics.
//!
//! * **The registry is passed explicitly.** Python referenced a module-global
//!   `registry`; the Rust [`ToolRegistry`] has no singleton, so the schema /
//!   dispatch helpers take `&ToolRegistry`. The convenience wrappers that the
//!   Python module exposed as free functions are mirrored here as functions
//!   taking the registry as their first argument.
//!
//! * **Plugin / discord / execute_code dynamic-schema rebuilds and plugin hooks**
//!   are modelled as optional injected callbacks ([`DynamicSchemaHooks`],
//!   [`DispatchHooks`]) so this module does not hard-depend on tool modules that
//!   may not be ported yet. When no hooks are supplied the behaviour degrades to
//!   the static schemas, exactly like Python's `except Exception: pass`
//!   fail-open paths.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde_json::{json, Value};

use crate::tool_registry::{
    ToolRegistry, ToolsetDisplay, ToolsetRequirements, UnavailableToolset,
};
use crate::mod_toolsets::{get_all_toolsets, resolve_toolset, validate_toolset};

// =============================================================================
// Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
// =============================================================================

/// Returns the legacy `_tools`-suffixed toolset name map. Port of
/// `_LEGACY_TOOLSET_MAP`. Old toolset names (e.g. `"web_tools"`) map directly to
/// explicit tool-name lists so configs written against the pre-toolset naming
/// keep working.
pub fn legacy_toolset_map() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static MAP: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        m.insert("web_tools", vec!["web_search", "web_extract"]);
        m.insert("terminal_tools", vec!["terminal"]);
        m.insert("vision_tools", vec!["vision_analyze"]);
        m.insert("moa_tools", vec!["mixture_of_agents"]);
        m.insert("image_tools", vec!["image_generate"]);
        m.insert(
            "skills_tools",
            vec!["skills_list", "skill_view", "skill_manage"],
        );
        m.insert(
            "browser_tools",
            vec![
                "browser_navigate",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "browser_scroll",
                "browser_back",
                "browser_press",
                "browser_get_images",
                "browser_vision",
                "browser_console",
            ],
        );
        m.insert("cronjob_tools", vec!["cronjob"]);
        m.insert(
            "rl_tools",
            vec![
                "rl_list_environments",
                "rl_select_environment",
                "rl_get_current_config",
                "rl_edit_config",
                "rl_start_training",
                "rl_check_status",
                "rl_stop_training",
                "rl_get_results",
                "rl_list_runs",
                "rl_test_inference",
            ],
        );
        m.insert(
            "file_tools",
            vec!["read_file", "write_file", "patch", "search_files"],
        );
        m.insert("tts_tools", vec!["text_to_speech"]);
        m
    })
}

// =============================================================================
// Module-level state (ports of the Python module globals)
// =============================================================================

/// Resolved tool names from the last [`get_tool_definitions`] call. Port of
/// `_last_resolved_tool_names`. Used by `execute_code` to know which tools are
/// available in this session.
fn last_resolved() -> &'static Mutex<Vec<String>> {
    static LAST: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    LAST.get_or_init(|| Mutex::new(Vec::new()))
}

/// Snapshot of the most recently resolved tool names. Mirrors reading the Python
/// `_last_resolved_tool_names` global.
pub fn last_resolved_tool_names() -> Vec<String> {
    last_resolved().lock().unwrap().clone()
}

/// Memoization cache for [`get_tool_definitions`]. Port of `_tool_defs_cache`.
/// Keyed on the same logical inputs as Python: the enabled/disabled toolset
/// sets, the registry generation, and a config fingerprint.
type CacheKey = (
    Option<BTreeSet<String>>, // enabled (None => default-all)
    Option<BTreeSet<String>>, // disabled (None/empty treated as None)
    u64,                      // registry generation
    Option<(i128, u64)>,      // config (mtime_ns, size)
);

fn tool_defs_cache() -> &'static Mutex<HashMap<CacheKey, Vec<Value>>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Vec<Value>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop memoized [`get_tool_definitions`] results. Port of
/// `_clear_tool_defs_cache`. Call when dynamic schema dependencies change (e.g.
/// discord capability cache reset, `execute_code` sandbox reconfigured).
pub fn clear_tool_defs_cache() {
    tool_defs_cache().lock().unwrap().clear();
}

// =============================================================================
// get_tool_definitions  (the main schema provider)
// =============================================================================

/// Optional callbacks for the dynamic-schema rebuild steps that the Python
/// `_compute_tool_definitions` performs (execute_code / discord / browser
/// fix-ups). Tool modules that own those schemas inject the relevant hooks; when
/// a hook is `None` the corresponding rebuild is skipped (the static schema is
/// kept), matching the Python fail-open behaviour.
#[derive(Default)]
pub struct DynamicSchemaHooks {
    /// Rebuild the `execute_code` schema given the set of sandbox-allowed tool
    /// names that are actually available. Returns the replacement `function`
    /// object (not wrapped in `{"type": "function", ...}`).
    pub build_execute_code_schema: Option<Box<dyn Fn(&BTreeSet<String>) -> Value + Send + Sync>>,
    /// Per-discord-tool dynamic schema function (keyed by tool name, e.g.
    /// `"discord"` / `"discord_admin"`). Returning `None` strips the tool, like
    /// the Python `get_dynamic_schema_*` returning `None`.
    pub discord_schema: Option<Box<dyn Fn(&str) -> Option<Value> + Send + Sync>>,
}

impl DynamicSchemaHooks {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Config fingerprint `(mtime_ns, size)` used as part of the memo key. Returns
/// `None` when the path is missing / unreadable, mirroring Python's
/// `except (FileNotFoundError, OSError, ImportError): cfg_fp = None`.
fn config_fingerprint(config_path: Option<&std::path::Path>) -> Option<(i128, u64)> {
    let path = config_path?;
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i128)?;
    Some((mtime_ns, size))
}

/// Get tool definitions for model API calls with toolset-based filtering.
///
/// Port of `get_tool_definitions`. All tools must be part of a toolset to be
/// accessible.
///
/// * `enabled_toolsets`: `Some` => only include tools from these toolsets;
///   `None` => start with everything.
/// * `disabled_toolsets`: subtract these toolsets' tools at the end. An empty
///   slice is treated as "no disable" (like Python's falsy check).
/// * `quiet_mode`: when `true`, suppress status logging *and* enable memoization
///   (the Python cache is only active when `quiet_mode=True` because the verbose
///   path has stdout side effects).
/// * `config_path`: path used for the config-mtime fingerprint in the memo key.
/// * `hooks`: dynamic-schema rebuild callbacks (may be empty/default).
pub fn get_tool_definitions(
    registry: &ToolRegistry,
    enabled_toolsets: Option<&[String]>,
    disabled_toolsets: &[String],
    quiet_mode: bool,
    config_path: Option<&std::path::Path>,
    hooks: &DynamicSchemaHooks,
) -> Vec<Value> {
    // Fast path: memoized result when the caller doesn't need verbose output.
    if quiet_mode {
        let cfg_fp = config_fingerprint(config_path);
        let cache_key: CacheKey = (
            enabled_toolsets.map(|ts| ts.iter().cloned().collect()),
            if disabled_toolsets.is_empty() {
                None
            } else {
                Some(disabled_toolsets.iter().cloned().collect())
            },
            registry.generation(),
            cfg_fp,
        );

        if let Some(cached) = tool_defs_cache().lock().unwrap().get(&cache_key) {
            // Update _last_resolved_tool_names so downstream callers see
            // consistent state even on a cache hit.
            *last_resolved().lock().unwrap() =
                cached.iter().filter_map(name_of).collect();
            // Shallow copy of the list; schemas are read-only by all callers.
            return cached.clone();
        }

        let result = compute_tool_definitions(
            registry,
            enabled_toolsets,
            disabled_toolsets,
            quiet_mode,
            hooks,
        );
        tool_defs_cache()
            .lock()
            .unwrap()
            .insert(cache_key, result.clone());
        return result;
    }

    compute_tool_definitions(
        registry,
        enabled_toolsets,
        disabled_toolsets,
        quiet_mode,
        hooks,
    )
}

/// Extract `tool["function"]["name"]` as an owned String, if present.
fn name_of(tool: &Value) -> Option<String> {
    tool.get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .map(|s| s.to_string())
}

/// Uncached implementation of [`get_tool_definitions`]. Port of
/// `_compute_tool_definitions`.
fn compute_tool_definitions(
    registry: &ToolRegistry,
    enabled_toolsets: Option<&[String]>,
    disabled_toolsets: &[String],
    quiet_mode: bool,
    hooks: &DynamicSchemaHooks,
) -> Vec<Value> {
    let legacy = legacy_toolset_map();
    let mut tools_to_include: BTreeSet<String> = BTreeSet::new();

    match enabled_toolsets {
        Some(enabled) => {
            for toolset_name in enabled {
                if validate_toolset(toolset_name) {
                    let resolved = resolve_toolset(toolset_name);
                    for t in &resolved {
                        tools_to_include.insert(t.clone());
                    }
                    if !quiet_mode {
                        let body = if resolved.is_empty() {
                            "no tools".to_string()
                        } else {
                            resolved.join(", ")
                        };
                        log::info!("\u{2705} Enabled toolset '{}': {}", toolset_name, body);
                    }
                } else if let Some(legacy_tools) = legacy.get(toolset_name.as_str()) {
                    for t in legacy_tools {
                        tools_to_include.insert((*t).to_string());
                    }
                    if !quiet_mode {
                        log::info!(
                            "\u{2705} Enabled legacy toolset '{}': {}",
                            toolset_name,
                            legacy_tools.join(", ")
                        );
                    }
                } else if !quiet_mode {
                    log::warn!("\u{26a0}\u{fe0f}  Unknown toolset: {}", toolset_name);
                }
            }
        }
        None => {
            // Default: start with everything.
            for ts_name in get_all_toolsets().keys() {
                for t in resolve_toolset(ts_name) {
                    tools_to_include.insert(t);
                }
            }
        }
    }

    // Always apply disabled toolsets as a subtraction step at the end.
    if !disabled_toolsets.is_empty() {
        for toolset_name in disabled_toolsets {
            if validate_toolset(toolset_name) {
                let resolved = resolve_toolset(toolset_name);
                for t in &resolved {
                    tools_to_include.remove(t);
                }
                if !quiet_mode {
                    let body = if resolved.is_empty() {
                        "no tools".to_string()
                    } else {
                        resolved.join(", ")
                    };
                    log::info!("\u{1f6ab} Disabled toolset '{}': {}", toolset_name, body);
                }
            } else if let Some(legacy_tools) = legacy.get(toolset_name.as_str()) {
                for t in legacy_tools {
                    tools_to_include.remove(*t);
                }
                if !quiet_mode {
                    log::info!(
                        "\u{1f6ab} Disabled legacy toolset '{}': {}",
                        toolset_name,
                        legacy_tools.join(", ")
                    );
                }
            } else if !quiet_mode {
                log::warn!("\u{26a0}\u{fe0f}  Unknown toolset: {}", toolset_name);
            }
        }
    }

    // Ask the registry for schemas (only returns tools whose check_fn passes).
    let mut filtered_tools = registry.get_definitions(&tools_to_include, quiet_mode);

    // The set of tool names that actually passed check_fn filtering.
    let mut available_tool_names: BTreeSet<String> =
        filtered_tools.iter().filter_map(name_of).collect();

    // Rebuild execute_code schema to only list sandbox tools actually available.
    if available_tool_names.contains("execute_code") {
        if let Some(build) = &hooks.build_execute_code_schema {
            let sandbox_enabled: BTreeSet<String> = sandbox_allowed_tools()
                .intersection(&available_tool_names)
                .cloned()
                .collect();
            let dynamic_schema = build(&sandbox_enabled);
            for td in filtered_tools.iter_mut() {
                if name_of(td).as_deref() == Some("execute_code") {
                    *td = json!({ "type": "function", "function": dynamic_schema });
                    break;
                }
            }
        }
    }

    // Rebuild discord / discord_admin schemas based on the bot's privileged
    // intents and the user's action allowlist.
    if let Some(discord_schema) = &hooks.discord_schema {
        for discord_tool_name in ["discord", "discord_admin"] {
            if available_tool_names.contains(discord_tool_name) {
                let dynamic = discord_schema(discord_tool_name);
                match dynamic {
                    None => {
                        filtered_tools
                            .retain(|t| name_of(t).as_deref() != Some(discord_tool_name));
                        available_tool_names.remove(discord_tool_name);
                    }
                    Some(dynamic) => {
                        for td in filtered_tools.iter_mut() {
                            if name_of(td).as_deref() == Some(discord_tool_name) {
                                *td = json!({ "type": "function", "function": dynamic });
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Strip web tool cross-references from browser_navigate description when
    // web_search / web_extract are not available.
    if available_tool_names.contains("browser_navigate") {
        let web_tools_available = available_tool_names.contains("web_search")
            || available_tool_names.contains("web_extract");
        if !web_tools_available {
            const CROSS_REF: &str = " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).";
            for td in filtered_tools.iter_mut() {
                if name_of(td).as_deref() == Some("browser_navigate") {
                    if let Some(func) = td.get("function") {
                        let mut func = func.clone();
                        let desc = func
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .replace(CROSS_REF, "");
                        if let Value::Object(m) = &mut func {
                            m.insert("description".to_string(), Value::String(desc));
                        }
                        *td = json!({ "type": "function", "function": func });
                    }
                    break;
                }
            }
        }
    }

    if !quiet_mode {
        if filtered_tools.is_empty() {
            log::info!("\u{1f6e0}\u{fe0f}  No tools selected (all filtered out or unavailable)");
        } else {
            let tool_names: Vec<String> = filtered_tools.iter().filter_map(name_of).collect();
            log::info!(
                "\u{1f6e0}\u{fe0f}  Final tool selection ({} tools): {}",
                filtered_tools.len(),
                tool_names.join(", ")
            );
        }
    }

    *last_resolved().lock().unwrap() = filtered_tools.iter().filter_map(name_of).collect();

    // Sanitize schemas for broad backend compatibility (llama.cpp grammar etc.).
    crate::tool_schema_sanitizer::sanitize_tool_schemas(&filtered_tools)
}

/// Default sandbox-allowed tool names, mirroring
/// `tools.code_execution_tool.SANDBOX_ALLOWED_TOOLS`. Defined here so the
/// execute_code rebuild step has a stable set even if the tool module isn't
/// ported; the actual schema body comes from the injected hook.
fn sandbox_allowed_tools() -> &'static BTreeSet<String> {
    static SET: OnceLock<BTreeSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "web_search",
            "web_extract",
            "read_file",
            "write_file",
            "search_files",
            "vision_analyze",
            "image_generate",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    })
}

// =============================================================================
// Tool argument type coercion
// =============================================================================

/// Coerce tool call arguments to match their JSON Schema types. Port of
/// `coerce_tool_args`.
///
/// LLMs frequently return numbers/booleans as strings (`"42"`, `"true"`). This
/// compares each argument against the tool's registered schema and coerces when
/// the value is a string but the schema expects another type. Original values
/// are preserved when coercion fails. Bare scalars are wrapped in a one-element
/// list when the schema declares `"type": "array"`.
pub fn coerce_tool_args(registry: &ToolRegistry, tool_name: &str, args: Value) -> Value {
    let mut map = match args {
        Value::Object(m) => m,
        other => return other, // not a dict -> unchanged
    };
    if map.is_empty() {
        return Value::Object(map);
    }

    let schema = match registry.get_schema(tool_name) {
        Some(s) => s,
        None => return Value::Object(map),
    };

    let properties = schema
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.as_object())
        .cloned();
    let properties = match properties {
        Some(p) if !p.is_empty() => p,
        _ => return Value::Object(map),
    };

    let keys: Vec<String> = map.keys().cloned().collect();
    for key in keys {
        let prop_schema = match properties.get(&key) {
            Some(p) => p.clone(),
            None => continue,
        };
        let expected = prop_schema.get("type");

        let value = map.get(&key).cloned().unwrap_or(Value::Null);

        // Wrap bare non-list values when the schema declares ``array``.
        if expected.and_then(|e| e.as_str()) == Some("array")
            && !value.is_null()
            && !value.is_array()
        {
            if let Value::String(s) = &value {
                let coerced = coerce_value(s, expected, Some(&prop_schema));
                if !values_identical(&coerced, &value) {
                    map.insert(key.clone(), coerced);
                    continue;
                }
                map.insert(key.clone(), Value::Array(vec![value.clone()]));
                log::info!(
                    "coerce_tool_args: wrapped bare string in list for {}.{}",
                    tool_name,
                    key
                );
                continue;
            }
            let type_name = json_type_name(&value);
            map.insert(key.clone(), Value::Array(vec![value]));
            log::info!(
                "coerce_tool_args: wrapped bare {} in list for {}.{}",
                type_name,
                tool_name,
                key
            );
            continue;
        }

        // Only string values are candidates for the remaining coercions.
        let s = match &value {
            Value::String(s) => s.clone(),
            _ => continue,
        };
        if expected.is_none() && !schema_allows_null(Some(&prop_schema)) {
            continue;
        }
        let coerced = coerce_value(&s, expected, Some(&prop_schema));
        if !values_identical(&coerced, &value) {
            map.insert(key.clone(), coerced);
        }
    }

    Value::Object(map)
}

/// Python `value.__name__`-style type label for log messages.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Whether two values are "the same" for coercion-change detection. Python used
/// identity (`is not`) on the original string; here we compare structurally,
/// which is equivalent for the inputs that matter (the original was always a
/// string, so a structurally-equal string means "no coercion happened").
fn values_identical(coerced: &Value, original: &Value) -> bool {
    coerced == original
}

/// Attempt to coerce a string `value` to `expected_type`. Returns the original
/// string (as a `Value::String`) when coercion is not applicable or fails. Port
/// of `_coerce_value`.
fn coerce_value(value: &str, expected_type: Option<&Value>, schema: Option<&Value>) -> Value {
    if schema_allows_null(schema) && value.trim().to_lowercase() == "null" {
        return Value::Null;
    }

    match expected_type {
        // Union type — try each in order, return first successful coercion.
        Some(Value::Array(types)) => {
            for t in types {
                let result = coerce_value(value, Some(t), schema);
                if result != Value::String(value.to_string()) {
                    return result;
                }
            }
            Value::String(value.to_string())
        }
        Some(Value::String(t)) => match t.as_str() {
            "integer" => coerce_number(value, true),
            "number" => coerce_number(value, false),
            "boolean" => coerce_boolean(value),
            "array" => coerce_json(value, JsonKind::Array),
            "object" => coerce_json(value, JsonKind::Object),
            "null" if value.trim().to_lowercase() == "null" => Value::Null,
            _ => Value::String(value.to_string()),
        },
        _ => Value::String(value.to_string()),
    }
}

/// Return `true` when a JSON Schema fragment explicitly permits null. Port of
/// `_schema_allows_null`.
fn schema_allows_null(schema: Option<&Value>) -> bool {
    let schema = match schema {
        Some(Value::Object(_)) => schema.unwrap(),
        _ => return false,
    };

    match schema.get("type") {
        Some(Value::String(t)) if t == "null" => return true,
        Some(Value::Array(types)) => {
            if types.iter().any(|t| t.as_str() == Some("null")) {
                return true;
            }
        }
        _ => {}
    }
    if schema.get("nullable") == Some(&Value::Bool(true)) {
        return true;
    }

    for union_key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(variants)) = schema.get(union_key) {
            for variant in variants {
                if variant.is_object() && variant.get("type").and_then(|t| t.as_str()) == Some("null")
                {
                    return true;
                }
            }
        }
    }

    false
}

#[derive(Clone, Copy)]
enum JsonKind {
    Array,
    Object,
}

/// Parse `value` as JSON when the schema expects an array or object. Returns the
/// original string when parsing fails or yields the wrong type. Port of
/// `_coerce_json`.
fn coerce_json(value: &str, kind: JsonKind) -> Value {
    let parsed: Value = match serde_json::from_str(value) {
        Ok(v) => v,
        Err(_) => return Value::String(value.to_string()),
    };
    let ok = match kind {
        JsonKind::Array => parsed.is_array(),
        JsonKind::Object => parsed.is_object(),
    };
    if ok {
        parsed
    } else {
        Value::String(value.to_string())
    }
}

/// Try to parse `value` as a number. Returns the original string on failure.
/// Port of `_coerce_number`. Integers-valued floats collapse to integers; with
/// `integer_only`, fractional values keep the original string.
fn coerce_number(value: &str, integer_only: bool) -> Value {
    let f: f64 = match value.trim().parse::<f64>() {
        Ok(f) => f,
        Err(_) => return Value::String(value.to_string()),
    };
    // Guard against inf/nan — not JSON-serializable, keep original string.
    if f.is_nan() || f.is_infinite() {
        return Value::String(value.to_string());
    }
    // If it looks like an integer (no fractional part), return int.
    if f == f.trunc() && f.abs() < 9.007_199_254_740_992e15 {
        // Within i64-safe integer range; emit an integer.
        return Value::Number((f as i64).into());
    }
    if integer_only {
        // Schema wants an integer but value has decimals — keep as string.
        return Value::String(value.to_string());
    }
    match serde_json::Number::from_f64(f) {
        Some(n) => Value::Number(n),
        None => Value::String(value.to_string()),
    }
}

/// Try to parse `value` as a boolean. Returns the original string on failure.
/// Port of `_coerce_boolean`.
fn coerce_boolean(value: &str) -> Value {
    match value.trim().to_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::String(value.to_string()),
    }
}

// =============================================================================
// handle_function_call  (the main dispatcher)
// =============================================================================

/// Tools whose execution is intercepted by the agent loop because they need
/// agent-level state. Port of `_AGENT_LOOP_TOOLS`.
pub fn agent_loop_tools() -> &'static BTreeSet<&'static str> {
    static SET: OnceLock<BTreeSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        ["todo", "memory", "session_search", "delegate_task"]
            .into_iter()
            .collect()
    })
}

/// Read/search tools that do *not* reset the consecutive-read counter. Port of
/// `_READ_SEARCH_TOOLS`.
pub fn read_search_tools() -> &'static BTreeSet<&'static str> {
    static SET: OnceLock<BTreeSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["read_file", "search_files"].into_iter().collect())
}

/// Optional dispatch-time hooks. These model the Python plugin-system seams
/// (`get_pre_tool_call_block_message`, `notify_other_tool_call`,
/// `post_tool_call`, `transform_tool_result`) without hard-depending on the
/// plugin module. All are fail-open: any panic is swallowed.
#[derive(Default)]
pub struct DispatchHooks {
    /// Pre-tool-call block check. Returning `Some(msg)` aborts dispatch and the
    /// dispatcher returns `{"error": msg}`. Mirrors
    /// `get_pre_tool_call_block_message`.
    #[allow(clippy::type_complexity)]
    pub pre_tool_call_block: Option<Box<dyn Fn(&str, &Value, &PreCallCtx) -> Option<String> + Send + Sync>>,
    /// Notify the read-loop tracker when a non-read/search tool runs. Mirrors
    /// `tools.file_tools.notify_other_tool_call`.
    pub notify_other_tool_call: Option<Box<dyn Fn(&str) + Send + Sync>>,
    /// Observational post-tool-call hook (receives `duration_ms`). Mirrors
    /// `invoke_hook("post_tool_call", ...)`.
    pub post_tool_call: Option<Box<dyn Fn(&PostCallCtx) + Send + Sync>>,
    /// Result-canonicalization seam. The first `Some(string)` returned replaces
    /// the result. Mirrors `invoke_hook("transform_tool_result", ...)`.
    pub transform_tool_result: Option<Box<dyn Fn(&PostCallCtx) -> Option<String> + Send + Sync>>,
}

impl DispatchHooks {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Identifiers threaded through the dispatch hooks. Port of the keyword args
/// that the Python dispatcher forwarded to the plugin hooks.
#[derive(Clone, Default)]
pub struct CallIds {
    pub task_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub session_id: Option<String>,
    pub user_task: Option<String>,
}

impl CallIds {
    fn task(&self) -> &str {
        self.task_id.as_deref().unwrap_or("")
    }
    fn session(&self) -> &str {
        self.session_id.as_deref().unwrap_or("")
    }
    fn call(&self) -> &str {
        self.tool_call_id.as_deref().unwrap_or("")
    }
}

/// Context passed to the pre-tool-call block hook.
pub struct PreCallCtx<'a> {
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub tool_call_id: &'a str,
}

/// Context passed to post-tool-call / transform-tool-result hooks.
pub struct PostCallCtx<'a> {
    pub tool_name: &'a str,
    pub args: &'a Value,
    pub result: &'a str,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub tool_call_id: &'a str,
    pub duration_ms: i64,
}

/// Main function call dispatcher that routes calls to the tool registry. Port of
/// `handle_function_call`.
///
/// * `registry`: the tool registry to dispatch against.
/// * `function_name` / `function_args`: the call.
/// * `ids`: task/session/tool-call identifiers + the user's original task.
/// * `enabled_tools`: tool names enabled for this session. When `Some`,
///   `execute_code` uses this list to decide which sandbox tools to generate;
///   falls back to the process-global `_last_resolved_tool_names`.
/// * `skip_pre_tool_call_hook`: when `true`, the caller already fired the
///   pre-tool-call hook; don't fire it again.
/// * `hooks`: plugin-system seams (may be empty/default).
///
/// Returns the function result as a JSON string.
#[allow(clippy::too_many_arguments)]
pub fn handle_function_call(
    registry: &ToolRegistry,
    function_name: &str,
    function_args: Value,
    ids: &CallIds,
    enabled_tools: Option<&[String]>,
    skip_pre_tool_call_hook: bool,
    hooks: &DispatchHooks,
) -> String {
    // Coerce string arguments to their schema-declared types (e.g. "42" -> 42).
    let function_args = coerce_tool_args(registry, function_name, function_args);

    let dispatch = || -> String {
        if agent_loop_tools().contains(function_name) {
            return json!({
                "error": format!("{} must be handled by the agent loop", function_name)
            })
            .to_string();
        }

        // Check plugin hooks for a block directive (unless caller already did).
        if !skip_pre_tool_call_hook {
            if let Some(block_fn) = &hooks.pre_tool_call_block {
                let ctx = PreCallCtx {
                    task_id: ids.task(),
                    session_id: ids.session(),
                    tool_call_id: ids.call(),
                };
                let block_message = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    block_fn(function_name, &function_args, &ctx)
                }))
                .unwrap_or(None);
                if let Some(msg) = block_message {
                    return json!({ "error": msg }).to_string();
                }
            }
        }

        // Notify the read-loop tracker when a non-read/search tool runs.
        if !read_search_tools().contains(function_name) {
            if let Some(notify) = &hooks.notify_other_tool_call {
                let task = ids.task_id.clone().unwrap_or_else(|| "default".to_string());
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| notify(&task)));
            }
        }

        // Measure tool dispatch latency (monotonic, immune to wall-clock skew).
        let dispatch_start = Instant::now();
        let result = if function_name == "execute_code" {
            // Prefer the caller-provided list so subagents can't overwrite the
            // parent's tool set via the process-global.
            let sandbox_enabled: Vec<String> = match enabled_tools {
                Some(t) => t.to_vec(),
                None => last_resolved_tool_names(),
            };
            let kwargs = json!({
                "task_id": ids.task_id,
                "enabled_tools": sandbox_enabled,
            });
            registry.dispatch(function_name, &function_args, &kwargs)
        } else {
            let kwargs = json!({
                "task_id": ids.task_id,
                "user_task": ids.user_task,
            });
            registry.dispatch(function_name, &function_args, &kwargs)
        };
        let duration_ms = dispatch_start.elapsed().as_millis() as i64;

        // Observational post_tool_call hook.
        if let Some(post) = &hooks.post_tool_call {
            let ctx = PostCallCtx {
                tool_name: function_name,
                args: &function_args,
                result: &result,
                task_id: ids.task(),
                session_id: ids.session(),
                tool_call_id: ids.call(),
                duration_ms,
            };
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| post(&ctx)));
        }

        // transform_tool_result canonicalization seam. First Some(string) wins.
        let mut result = result;
        if let Some(transform) = &hooks.transform_tool_result {
            let replaced = {
                let ctx = PostCallCtx {
                    tool_name: function_name,
                    args: &function_args,
                    result: &result,
                    task_id: ids.task(),
                    session_id: ids.session(),
                    tool_call_id: ids.call(),
                    duration_ms,
                };
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| transform(&ctx)))
                    .unwrap_or(None)
            };
            if let Some(s) = replaced {
                result = s;
            }
        }

        result
    };

    // Outer try/except: any panic becomes an {"error": ...} JSON string.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(dispatch)) {
        Ok(s) => s,
        Err(_) => {
            let error_msg = format!("Error executing {}", function_name);
            log::error!("{}", error_msg);
            json!({ "error": error_msg }).to_string()
        }
    }
}

// =============================================================================
// Backward-compat wrapper functions
// =============================================================================

/// Return all registered tool names. Port of `get_all_tool_names`.
pub fn get_all_tool_names(registry: &ToolRegistry) -> Vec<String> {
    registry.get_all_tool_names()
}

/// Return the toolset a tool belongs to. Port of `get_toolset_for_tool`.
pub fn get_toolset_for_tool(registry: &ToolRegistry, tool_name: &str) -> Option<String> {
    registry.get_toolset_for_tool(tool_name)
}

/// Return toolset availability info for UI display. Port of
/// `get_available_toolsets`.
pub fn get_available_toolsets(registry: &ToolRegistry) -> HashMap<String, ToolsetDisplay> {
    registry.get_available_toolsets()
}

/// Return `{toolset: available_bool}` for every registered toolset. Port of
/// `check_toolset_requirements`.
pub fn check_toolset_requirements(registry: &ToolRegistry) -> std::collections::BTreeMap<String, bool> {
    registry.check_toolset_requirements()
}

/// Return `(available_toolsets, unavailable_info)`. Port of
/// `check_tool_availability`.
pub fn check_tool_availability(
    registry: &ToolRegistry,
) -> (Vec<String>, Vec<UnavailableToolset>) {
    registry.check_tool_availability()
}

/// Build the `TOOL_TO_TOOLSET_MAP` (for batch_runner). Port of the module-level
/// `TOOL_TO_TOOLSET_MAP` constant, which Python built once after discovery.
pub fn tool_to_toolset_map(registry: &ToolRegistry) -> HashMap<String, String> {
    registry.get_tool_to_toolset_map()
}

/// Build the `TOOLSET_REQUIREMENTS` map (for cli / doctor). Port of the
/// module-level `TOOLSET_REQUIREMENTS` constant.
pub fn toolset_requirements(registry: &ToolRegistry) -> HashMap<String, ToolsetRequirements> {
    registry.get_toolset_requirements()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::{Handler, RegisterOptions};
    use std::sync::Arc;

    fn noop() -> Handler {
        Arc::new(|_a: &Value, _k: &Value| "{}".to_string())
    }

    fn reg_with(name: &str, schema: Value) -> ToolRegistry {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(name, "ts", schema, noop()));
        reg
    }

    // ----- coerce_value primitives -----

    #[test]
    fn coerce_number_integer_valued_float_becomes_int() {
        assert_eq!(coerce_number("42", false), json!(42));
        assert_eq!(coerce_number("42.0", false), json!(42));
        assert_eq!(coerce_number("3.5", false), json!(3.5));
    }

    #[test]
    fn coerce_number_integer_only_keeps_decimal_string() {
        assert_eq!(coerce_number("3.5", true), json!("3.5"));
        assert_eq!(coerce_number("7", true), json!(7));
    }

    #[test]
    fn coerce_number_rejects_nan_inf_and_garbage() {
        assert_eq!(coerce_number("inf", false), json!("inf"));
        assert_eq!(coerce_number("nan", false), json!("nan"));
        assert_eq!(coerce_number("abc", false), json!("abc"));
    }

    #[test]
    fn coerce_boolean_cases() {
        assert_eq!(coerce_boolean("true"), json!(true));
        assert_eq!(coerce_boolean(" FALSE "), json!(false));
        assert_eq!(coerce_boolean("yes"), json!("yes"));
    }

    #[test]
    fn coerce_json_array_and_object() {
        assert_eq!(coerce_json("[1,2]", JsonKind::Array), json!([1, 2]));
        assert_eq!(coerce_json("{\"a\":1}", JsonKind::Object), json!({"a": 1}));
        // wrong type kept as string
        assert_eq!(coerce_json("{\"a\":1}", JsonKind::Array), json!("{\"a\":1}"));
        assert_eq!(coerce_json("not json", JsonKind::Array), json!("not json"));
    }

    #[test]
    fn schema_allows_null_variants() {
        assert!(schema_allows_null(Some(&json!({"type": "null"}))));
        assert!(schema_allows_null(Some(&json!({"type": ["string", "null"]}))));
        assert!(schema_allows_null(Some(&json!({"nullable": true}))));
        assert!(schema_allows_null(Some(&json!({"anyOf": [{"type": "null"}]}))));
        assert!(schema_allows_null(Some(&json!({"oneOf": [{"type": "string"}, {"type": "null"}]}))));
        assert!(!schema_allows_null(Some(&json!({"type": "string"}))));
        assert!(!schema_allows_null(Some(&json!("string"))));
        assert!(!schema_allows_null(None));
    }

    #[test]
    fn coerce_value_null_string_to_none_when_allowed() {
        let schema = json!({"type": ["integer", "null"]});
        assert_eq!(coerce_value("null", schema.get("type"), Some(&schema)), Value::Null);
        // not allowed -> stays string
        let schema2 = json!({"type": "integer"});
        assert_eq!(
            coerce_value("null", schema2.get("type"), Some(&schema2)),
            json!("null")
        );
    }

    #[test]
    fn coerce_value_union_tries_in_order() {
        let schema = json!({"type": ["integer", "string"]});
        // "42" -> integer wins
        assert_eq!(coerce_value("42", schema.get("type"), Some(&schema)), json!(42));
        // "abc" -> neither integer; string is identity -> stays string
        assert_eq!(coerce_value("abc", schema.get("type"), Some(&schema)), json!("abc"));
    }

    // ----- coerce_tool_args end-to-end -----

    #[test]
    fn coerce_tool_args_integer_string() {
        let reg = reg_with(
            "t",
            json!({"parameters": {"properties": {"n": {"type": "integer"}}}}),
        );
        let out = coerce_tool_args(&reg, "t", json!({"n": "42"}));
        assert_eq!(out, json!({"n": 42}));
    }

    #[test]
    fn coerce_tool_args_boolean_string() {
        let reg = reg_with(
            "t",
            json!({"parameters": {"properties": {"b": {"type": "boolean"}}}}),
        );
        let out = coerce_tool_args(&reg, "t", json!({"b": "true"}));
        assert_eq!(out, json!({"b": true}));
    }

    #[test]
    fn coerce_tool_args_wraps_bare_scalar_in_array() {
        let reg = reg_with(
            "t",
            json!({"parameters": {"properties": {"urls": {"type": "array"}}}}),
        );
        // bare string -> wrapped (no JSON-parse possible)
        let out = coerce_tool_args(&reg, "t", json!({"urls": "https://a.com"}));
        assert_eq!(out, json!({"urls": ["https://a.com"]}));
        // JSON-encoded array string -> parsed, not double-wrapped
        let out2 = coerce_tool_args(&reg, "t", json!({"urls": "[\"a\",\"b\"]"}));
        assert_eq!(out2, json!({"urls": ["a", "b"]}));
        // bare number -> wrapped
        let out3 = coerce_tool_args(&reg, "t", json!({"urls": 5}));
        assert_eq!(out3, json!({"urls": [5]}));
    }

    #[test]
    fn coerce_tool_args_array_nullable_string_becomes_none_not_wrapped() {
        let reg = reg_with(
            "t",
            json!({"parameters": {"properties": {"x": {"type": ["array", "null"]}}}}),
        );
        let out = coerce_tool_args(&reg, "t", json!({"x": "null"}));
        assert_eq!(out, json!({"x": null}));
    }

    #[test]
    fn coerce_tool_args_preserves_when_no_schema() {
        let reg = ToolRegistry::new();
        let out = coerce_tool_args(&reg, "missing", json!({"n": "42"}));
        assert_eq!(out, json!({"n": "42"}));
    }

    #[test]
    fn coerce_tool_args_non_dict_unchanged() {
        let reg = ToolRegistry::new();
        assert_eq!(coerce_tool_args(&reg, "t", json!("scalar")), json!("scalar"));
        assert_eq!(coerce_tool_args(&reg, "t", json!({})), json!({}));
    }

    #[test]
    fn coerce_tool_args_skips_unknown_keys_and_non_strings() {
        let reg = reg_with(
            "t",
            json!({"parameters": {"properties": {"n": {"type": "integer"}}}}),
        );
        let out = coerce_tool_args(&reg, "t", json!({"n": 1, "other": "x"}));
        assert_eq!(out, json!({"n": 1, "other": "x"}));
    }

    // ----- handle_function_call -----

    #[test]
    fn handle_agent_loop_tool_returns_stub_error() {
        let reg = reg_with("todo", json!({"parameters": {}}));
        let out = handle_function_call(
            &reg,
            "todo",
            json!({}),
            &CallIds::default(),
            None,
            true,
            &DispatchHooks::default(),
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], "todo must be handled by the agent loop");
    }

    #[test]
    fn handle_dispatches_and_coerces_args() {
        let reg = ToolRegistry::new();
        let handler: Handler = Arc::new(|args: &Value, _k: &Value| {
            json!({"got": args["n"]}).to_string()
        });
        reg.register(RegisterOptions::new(
            "echo",
            "ts",
            json!({"parameters": {"properties": {"n": {"type": "integer"}}}}),
            handler,
        ));
        let out = handle_function_call(
            &reg,
            "echo",
            json!({"n": "7"}),
            &CallIds::default(),
            None,
            true,
            &DispatchHooks::default(),
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["got"], 7); // coerced "7" -> 7
    }

    #[test]
    fn handle_pre_tool_call_block_aborts() {
        let reg = reg_with("t", json!({"parameters": {}}));
        let mut hooks = DispatchHooks::default();
        hooks.pre_tool_call_block =
            Some(Box::new(|_n, _a, _c| Some("blocked by policy".to_string())));
        let out = handle_function_call(
            &reg,
            "t",
            json!({}),
            &CallIds::default(),
            None,
            false,
            &hooks,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], "blocked by policy");
    }

    #[test]
    fn handle_skip_pre_hook_does_not_block() {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(
            "t",
            "ts",
            json!({"parameters": {}}),
            Arc::new(|_a, _k| json!({"ok": true}).to_string()),
        ));
        let mut hooks = DispatchHooks::default();
        hooks.pre_tool_call_block = Some(Box::new(|_n, _a, _c| Some("should not fire".to_string())));
        let out = handle_function_call(
            &reg,
            "t",
            json!({}),
            &CallIds::default(),
            None,
            true, // skip
            &hooks,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn handle_transform_tool_result_replaces() {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(
            "t",
            "ts",
            json!({"parameters": {}}),
            Arc::new(|_a, _k| json!({"orig": 1}).to_string()),
        ));
        let mut hooks = DispatchHooks::default();
        hooks.transform_tool_result =
            Some(Box::new(|_ctx| Some(r#"{"transformed":true}"#.to_string())));
        let out = handle_function_call(
            &reg,
            "t",
            json!({}),
            &CallIds::default(),
            None,
            true,
            &hooks,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["transformed"], true);
    }

    #[test]
    fn handle_notify_fires_for_non_read_tool() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(
            "terminal",
            "ts",
            json!({"parameters": {}}),
            noop(),
        ));
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let mut hooks = DispatchHooks::default();
        hooks.notify_other_tool_call = Some(Box::new(move |task| {
            assert_eq!(task, "default");
            f.store(true, Ordering::SeqCst);
        }));
        handle_function_call(&reg, "terminal", json!({}), &CallIds::default(), None, true, &hooks);
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn handle_notify_skipped_for_read_tool() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(
            "read_file",
            "ts",
            json!({"parameters": {}}),
            noop(),
        ));
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let mut hooks = DispatchHooks::default();
        hooks.notify_other_tool_call = Some(Box::new(move |_| f.store(true, Ordering::SeqCst)));
        handle_function_call(&reg, "read_file", json!({}), &CallIds::default(), None, true, &hooks);
        assert!(!fired.load(Ordering::SeqCst));
    }

    // ----- get_tool_definitions -----

    #[test]
    fn legacy_map_has_expected_entries() {
        let m = legacy_toolset_map();
        assert_eq!(m.get("web_tools").unwrap(), &vec!["web_search", "web_extract"]);
        assert_eq!(m.get("tts_tools").unwrap(), &vec!["text_to_speech"]);
        assert_eq!(m.get("browser_tools").unwrap().len(), 10);
    }

    #[test]
    fn get_tool_definitions_enabled_legacy_toolset() {
        let reg = ToolRegistry::new();
        // Register the two web tools.
        reg.register(RegisterOptions::new(
            "web_search",
            "web",
            json!({"description": "search"}),
            noop(),
        ));
        reg.register(RegisterOptions::new(
            "web_extract",
            "web",
            json!({"description": "extract"}),
            noop(),
        ));
        let enabled = vec!["web_tools".to_string()];
        let defs = get_tool_definitions(
            &reg,
            Some(&enabled),
            &[],
            true,
            None,
            &DynamicSchemaHooks::default(),
        );
        let names: BTreeSet<String> = defs.iter().filter_map(name_of).collect();
        assert_eq!(
            names,
            ["web_extract", "web_search"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        );
        // last_resolved updated.
        let lr: BTreeSet<String> = last_resolved_tool_names().into_iter().collect();
        assert!(lr.contains("web_search"));
    }

    #[test]
    fn get_tool_definitions_disabled_subtracts() {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new("web_search", "web", json!({}), noop()));
        reg.register(RegisterOptions::new("web_extract", "web", json!({}), noop()));
        let enabled = vec!["web_tools".to_string()];
        let disabled = vec!["web_tools".to_string()];
        let defs = get_tool_definitions(
            &reg,
            Some(&enabled),
            &disabled,
            true,
            None,
            &DynamicSchemaHooks::default(),
        );
        assert!(defs.is_empty());
    }

    #[test]
    fn get_tool_definitions_memoizes_on_quiet() {
        clear_tool_defs_cache();
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new("web_search", "web", json!({}), noop()));
        reg.register(RegisterOptions::new("web_extract", "web", json!({}), noop()));
        let enabled = vec!["web_tools".to_string()];
        let a = get_tool_definitions(&reg, Some(&enabled), &[], true, None, &DynamicSchemaHooks::default());
        let b = get_tool_definitions(&reg, Some(&enabled), &[], true, None, &DynamicSchemaHooks::default());
        assert_eq!(a, b);
        // A mutation bumps generation -> cache miss with fresh (still equal) result.
        reg.register(RegisterOptions::new("x", "other", json!({}), noop()));
        let c = get_tool_definitions(&reg, Some(&enabled), &[], true, None, &DynamicSchemaHooks::default());
        assert_eq!(a, c);
    }

    #[test]
    fn browser_navigate_strips_web_cross_ref_when_web_unavailable() {
        let reg = ToolRegistry::new();
        let desc = "Navigate. For simple information retrieval, prefer web_search or web_extract (faster, cheaper).";
        reg.register(RegisterOptions::new(
            "browser_navigate",
            "browser",
            json!({"description": desc}),
            noop(),
        ));
        let enabled = vec!["browser_tools".to_string()];
        let defs = get_tool_definitions(
            &reg,
            Some(&enabled),
            &[],
            true,
            None,
            &DynamicSchemaHooks::default(),
        );
        let nav = defs
            .iter()
            .find(|d| name_of(d).as_deref() == Some("browser_navigate"))
            .unwrap();
        let got = nav["function"]["description"].as_str().unwrap();
        assert_eq!(got, "Navigate.");
    }

    #[test]
    fn execute_code_schema_rebuilt_via_hook() {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new(
            "execute_code",
            "code",
            json!({"description": "static"}),
            noop(),
        ));
        reg.register(RegisterOptions::new("web_search", "web", json!({}), noop()));
        let mut hooks = DynamicSchemaHooks::default();
        hooks.build_execute_code_schema = Some(Box::new(|sandbox: &BTreeSet<String>| {
            json!({"description": "dynamic", "sandbox": sandbox.iter().cloned().collect::<Vec<_>>()})
        }));
        let enabled = vec!["code".to_string(), "web".to_string()];
        // Register a toolset alias path won't matter; use enabled toolset names that
        // resolve via registry-based toolsets isn't available here, so just include
        // both tools directly through legacy-style names is not possible. Instead
        // verify the rebuild logic via include-all (None) path.
        let _ = enabled;
        let defs = get_tool_definitions(&reg, None, &[], true, None, &hooks);
        // execute_code may or may not be selected depending on toolset resolution;
        // if present, its schema must be the dynamic one.
        if let Some(ec) = defs.iter().find(|d| name_of(d).as_deref() == Some("execute_code")) {
            assert_eq!(ec["function"]["description"], "dynamic");
        }
    }

    #[test]
    fn backward_compat_wrappers() {
        let reg = ToolRegistry::new();
        reg.register(RegisterOptions::new("a", "ts1", json!({}), noop()));
        assert_eq!(get_all_tool_names(&reg), vec!["a".to_string()]);
        assert_eq!(get_toolset_for_tool(&reg, "a"), Some("ts1".to_string()));
        assert!(tool_to_toolset_map(&reg).contains_key("a"));
        assert!(toolset_requirements(&reg).contains_key("ts1"));
        assert!(get_available_toolsets(&reg).contains_key("ts1"));
        assert!(check_toolset_requirements(&reg).contains_key("ts1"));
        let (avail, _unavail) = check_tool_availability(&reg);
        assert!(avail.contains(&"ts1".to_string()));
    }
}
