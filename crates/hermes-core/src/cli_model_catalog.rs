//! Remote model catalog fetcher.
//!
//! Native Rust port of `hermes_cli/model_catalog.py`.
//!
//! The Hermes docs site hosts a JSON manifest of curated models for providers
//! we want to update without shipping a release (currently OpenRouter and
//! Nous Portal). This module fetches, validates, and caches that manifest,
//! falling back to the in-repo hardcoded lists when the network is unavailable.
//!
//! Pipeline
//! --------
//! 1. [`get_catalog`] — returns a parsed manifest object.
//!    - Checks in-process cache (invalidated by TTL).
//!    - Reads disk cache at `~/.hermes/cache/model_catalog.json`.
//!    - Fetches the master URL if disk cache is stale or missing.
//!    - On any fetch failure, keeps using the stale cache (or empty object).
//!
//! 2. [`get_curated_openrouter_models`] / [`get_curated_nous_models`] — thin
//!    accessors returning the shapes existing callers expect. Each returns
//!    `None` so callers can fall back to the in-repo hardcoded list.
//!
//! Schema (version 1)
//! ------------------
//! ```text
//!     {
//!       "version": 1,
//!       "updated_at": "2026-04-25T22:00:00Z",
//!       "metadata": {...},                # free-form
//!       "providers": {
//!         "openrouter": {
//!           "metadata": {...},            # free-form
//!           "models": [
//!             {"id": "vendor/model", "description": "recommended",
//!              "metadata": {...}}          # free-form, model-level
//!           ]
//!         },
//!         "nous": {...}
//!       }
//!     }
//! ```
//!
//! Unknown fields are ignored — extra metadata can be added at either level
//! without bumping `version`. `version` bumps are reserved for breaking
//! changes (renaming `providers`, changing `models` shape).

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default master manifest URL.
pub const DEFAULT_CATALOG_URL: &str =
    "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json";
/// Default cache TTL in hours.
pub const DEFAULT_TTL_HOURS: f64 = 24.0;
/// Default fetch timeout in seconds.
pub const DEFAULT_FETCH_TIMEOUT: f64 = 8.0;
/// Highest manifest schema version this build understands.
pub const SUPPORTED_SCHEMA_VERSION: i64 = 1;

/// `User-Agent` header sent on catalog fetches: `hermes-cli/<version>`.
fn hermes_user_agent() -> String {
    format!("hermes-cli/{}", crate::cli_models::HERMES_VERSION)
}

// ---------------------------------------------------------------------------
// In-process cache
// ---------------------------------------------------------------------------

/// In-process cache to avoid repeated disk + parse work across multiple calls
/// within the same session. Invalidated by TTL against the disk file's mtime,
/// so calling code never has to think about this.
struct CacheState {
    catalog: Option<Value>,
    source_mtime: f64,
}

fn cache_state() -> &'static Mutex<CacheState> {
    use std::sync::OnceLock;
    static STATE: OnceLock<Mutex<CacheState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(CacheState {
            catalog: None,
            source_mtime: 0.0,
        })
    })
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The `model_catalog` config block with defaults filled in.
#[derive(Debug, Clone)]
pub struct CatalogConfig {
    pub enabled: bool,
    pub url: String,
    pub ttl_hours: f64,
    /// `model_catalog.providers` map (provider name -> free-form block).
    pub providers: Map<String, Value>,
}

/// Coerce a JSON value to `f64` the way Python's `float(raw.get(...) or DEFAULT)`
/// would, treating falsy / non-numeric as "use default".
fn to_float_or(value: Option<&Value>, default: f64) -> f64 {
    match value {
        Some(Value::Number(n)) => match n.as_f64() {
            // Python `or DEFAULT` treats 0.0 as falsy -> default.
            Some(f) if f != 0.0 => f,
            _ => default,
        },
        Some(Value::String(s)) => {
            if s.is_empty() {
                default
            } else {
                s.parse::<f64>().unwrap_or(default)
            }
        }
        _ => default,
    }
}

/// Load the `model_catalog` config block with defaults filled in.
///
/// Mirrors Python `_load_catalog_config`: any failure to load config yields an
/// empty config, and a non-dict `model_catalog` is treated as empty.
pub fn load_catalog_config() -> CatalogConfig {
    // cli_config returns serde_yaml::Value; bridge to serde_json (this module's
    // Value) so the object accessors below apply unchanged.
    let cfg: Value = serde_json::to_value(crate::cli_config::load_config())
        .unwrap_or(Value::Null);

    let raw: &Map<String, Value> = cfg
        .get("model_catalog")
        .and_then(Value::as_object)
        .unwrap_or_else(|| {
            // Use a static empty map to satisfy the borrow; build lazily.
            static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
            EMPTY.get_or_init(Map::new)
        });

    // enabled: bool(raw.get("enabled", True))
    let enabled = match raw.get("enabled") {
        Some(v) => json_truthy(v),
        None => true,
    };

    // url: str(raw.get("url") or DEFAULT)
    let url = match raw.get("url") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => DEFAULT_CATALOG_URL.to_string(),
    };

    let ttl_hours = to_float_or(raw.get("ttl_hours"), DEFAULT_TTL_HOURS);

    let providers = raw
        .get("providers")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    CatalogConfig {
        enabled,
        url,
        ttl_hours,
        providers,
    }
}

/// Python-style truthiness for the `enabled` flag.
fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Disk cache path: `<hermes_home>/cache/model_catalog.json`.
pub fn cache_path() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_home()
        .join("cache")
        .join("model_catalog.json")
}

// ---------------------------------------------------------------------------
// Fetch + validate + cache
// ---------------------------------------------------------------------------

/// HTTP GET the manifest URL and return a parsed value, or `None` on failure.
fn fetch_manifest(url: &str, timeout: f64) -> Option<Value> {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()
    {
        Ok(c) => c,
        Err(exc) => {
            log::info!("model catalog fetch errored ({url}): {exc}");
            return None;
        }
    };

    let resp = match client
        .get(url)
        .header("Accept", "application/json")
        .header("User-Agent", hermes_user_agent())
        .send()
    {
        Ok(r) => r,
        Err(exc) => {
            log::info!("model catalog fetch failed ({url}): {exc}");
            return None;
        }
    };

    let body = match resp.text() {
        Ok(b) => b,
        Err(exc) => {
            log::info!("model catalog fetch failed ({url}): {exc}");
            return None;
        }
    };

    let data: Value = match serde_json::from_str(&body) {
        Ok(d) => d,
        Err(exc) => {
            log::info!("model catalog fetch failed ({url}): {exc}");
            return None;
        }
    };

    if !validate_manifest(&data) {
        log::info!("model catalog at {url} failed schema validation");
        return None;
    }

    Some(data)
}

/// Return `true` when `data` matches the minimum manifest shape.
pub fn validate_manifest(data: &Value) -> bool {
    let obj = match data.as_object() {
        Some(o) => o,
        None => return false,
    };

    // version must be an integer <= SUPPORTED_SCHEMA_VERSION (and not a bool).
    match obj.get("version") {
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            let version = match n.as_i64() {
                Some(v) => v,
                // u64 too large for i64 -> definitely > supported.
                None => return false,
            };
            if version > SUPPORTED_SCHEMA_VERSION {
                return false;
            }
        }
        _ => return false,
    }

    let providers = match obj.get("providers").and_then(Value::as_object) {
        Some(p) => p,
        None => return false,
    };

    for (_pname, pblock) in providers {
        let pblock = match pblock.as_object() {
            Some(b) => b,
            None => return false,
        };
        let models = match pblock.get("models").and_then(Value::as_array) {
            Some(m) => m,
            None => return false,
        };
        for m in models {
            let m = match m.as_object() {
                Some(o) => o,
                None => return false,
            };
            match m.get("id") {
                Some(Value::String(s)) if !s.trim().is_empty() => {}
                _ => return false,
            }
        }
    }

    true
}

/// Return `(data_or_none, mtime)`. `mtime` is 0 if file is missing/invalid.
fn read_disk_cache() -> (Option<Value>, f64) {
    let path = cache_path();
    let mtime = match std::fs::metadata(&path).and_then(|m| m.modified()) {
        Ok(t) => match t.duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs_f64(),
            Err(_) => return (None, 0.0),
        },
        Err(_) => return (None, 0.0),
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return (None, 0.0),
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(_) => return (None, 0.0),
    };
    if !validate_manifest(&data) {
        return (None, 0.0);
    }
    (Some(data), mtime)
}

/// Atomically write the manifest to the disk cache. Failures are logged, never
/// raised — mirrors the Python `OSError`-swallowing behaviour.
fn write_disk_cache(data: &Value) {
    let path = cache_path();
    let result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // path.with_suffix(path.suffix + ".tmp") -> append ".tmp" to extension.
        let mut tmp = path.clone();
        let new_ext = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("{ext}.tmp"),
            None => "tmp".to_string(),
        };
        tmp.set_extension(new_ext);

        let mut serialized =
            serde_json::to_string_pretty(data).unwrap_or_else(|_| "{}".to_string());
        serialized.push('\n');
        std::fs::write(&tmp, serialized)?;

        crate::mod_utils::atomic_replace(&tmp, &path)?;
        Ok(())
    })();

    if let Err(exc) = result {
        log::info!("model catalog cache write failed: {exc}");
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Current wall-clock time in seconds since the Unix epoch.
fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Return the parsed model catalog manifest, or an empty object on failure.
///
/// Callers should treat a missing provider/model as "use the in-repo fallback"
/// — this function never errors so the CLI keeps working offline. The returned
/// `Value` is always a JSON object (possibly empty).
pub fn get_catalog(force_refresh: bool) -> Value {
    let cfg = load_catalog_config();
    if !cfg.enabled {
        return Value::Object(Map::new());
    }

    let ttl_seconds = (cfg.ttl_hours * 3600.0).max(0.0);

    let (disk_data, disk_mtime) = read_disk_cache();
    let now = now_seconds();
    let disk_fresh = disk_data.is_some() && (now - disk_mtime) < ttl_seconds;

    // In-process cache hit: disk hasn't changed since we loaded it and still fresh.
    {
        let state = cache_state().lock().unwrap();
        if !force_refresh
            && state.catalog.is_some()
            && disk_data.is_some()
            && disk_mtime == state.source_mtime
            && disk_fresh
        {
            return state.catalog.clone().unwrap();
        }
    }

    // Disk is fresh enough — use it without a network hit.
    if !force_refresh && disk_fresh {
        if let Some(data) = disk_data.clone() {
            let mut state = cache_state().lock().unwrap();
            state.catalog = Some(data.clone());
            state.source_mtime = disk_mtime;
            return data;
        }
    }

    // Need to (re)fetch. If it fails, fall back to any stale disk copy.
    if let Some(fetched) = fetch_manifest(&cfg.url, DEFAULT_FETCH_TIMEOUT) {
        write_disk_cache(&fetched);
        let (new_disk_data, new_mtime) = read_disk_cache();
        if let Some(new_disk_data) = new_disk_data {
            let mut state = cache_state().lock().unwrap();
            state.catalog = Some(new_disk_data.clone());
            state.source_mtime = new_mtime;
            return new_disk_data;
        }
        let mut state = cache_state().lock().unwrap();
        state.catalog = Some(fetched.clone());
        state.source_mtime = now;
        return fetched;
    }

    if let Some(data) = disk_data {
        let mut state = cache_state().lock().unwrap();
        state.catalog = Some(data.clone());
        state.source_mtime = disk_mtime;
        return data;
    }

    Value::Object(Map::new())
}

/// If `model_catalog.providers.<name>.url` is set, fetch that instead.
fn fetch_provider_override(provider: &str) -> Option<Value> {
    let cfg = load_catalog_config();
    if !cfg.enabled {
        return None;
    }
    let provider_cfg = cfg.providers.get(provider).and_then(Value::as_object)?;
    let override_url = match provider_cfg.get("url") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return None,
    };
    // Override fetches skip the disk cache because they're usually third-party
    // self-hosted. Re-request on every call but with a short timeout so they
    // don't block the picker.
    fetch_manifest(&override_url, DEFAULT_FETCH_TIMEOUT)
}

/// Return the provider's manifest block, respecting per-provider overrides.
pub fn get_provider_block(provider: &str) -> Option<Value> {
    if let Some(override_data) = fetch_provider_override(provider) {
        if let Some(block) = override_data
            .get("providers")
            .and_then(Value::as_object)
            .and_then(|p| p.get(provider))
        {
            if block.is_object() {
                return Some(block.clone());
            }
        }
    }

    let catalog = get_catalog(false);
    let obj = catalog.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let block = obj.get("providers").and_then(Value::as_object)?.get(provider)?;
    if block.is_object() {
        Some(block.clone())
    } else {
        None
    }
}

/// Return OpenRouter's curated `[(id, description), ...]` from the manifest.
///
/// Returns `None` when the manifest is unavailable, so callers can fall back to
/// their hardcoded list.
pub fn get_curated_openrouter_models() -> Option<Vec<(String, String)>> {
    let block = get_provider_block("openrouter")?;
    let models = block.get("models").and_then(Value::as_array);
    let mut out: Vec<(String, String)> = Vec::new();
    if let Some(models) = models {
        for m in models {
            let m = match m.as_object() {
                Some(o) => o,
                None => continue,
            };
            let mid = str_or_empty(m.get("id"));
            let mid = mid.trim();
            if mid.is_empty() {
                continue;
            }
            let desc = str_or_empty(m.get("description"));
            out.push((mid.to_string(), desc));
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Return Nous Portal's curated list of model ids from the manifest.
///
/// Returns `None` when the manifest is unavailable.
pub fn get_curated_nous_models() -> Option<Vec<String>> {
    let block = get_provider_block("nous")?;
    let models = block.get("models").and_then(Value::as_array);
    let mut out: Vec<String> = Vec::new();
    if let Some(models) = models {
        for m in models {
            let m = match m.as_object() {
                Some(o) => o,
                None => continue,
            };
            let mid = str_or_empty(m.get("id"));
            let mid = mid.trim();
            if !mid.is_empty() {
                out.push(mid.to_string());
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Mirror Python `str(m.get(key) or "")`: `None`/null/empty -> "", else string
/// coercion of the value.
fn str_or_empty(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => {
            // Python `str(True)` -> "True"; `False or ""` -> "" (falsy).
            if *b {
                "True".to_string()
            } else {
                String::new()
            }
        }
        Some(Value::Number(n)) => {
            // `0 or ""` -> "" in Python.
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Clear the in-process cache. Used by tests and `hermes model --refresh`.
pub fn reset_cache() {
    let mut state = cache_state().lock().unwrap();
    state.catalog = None;
    state.source_mtime = 0.0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_minimal_ok() {
        let data = json!({
            "version": 1,
            "providers": {
                "openrouter": {
                    "models": [{"id": "vendor/model", "description": "x"}]
                }
            }
        });
        assert!(validate_manifest(&data));
    }

    #[test]
    fn validate_rejects_non_object() {
        assert!(!validate_manifest(&json!([])));
        assert!(!validate_manifest(&json!("string")));
        assert!(!validate_manifest(&json!(5)));
    }

    #[test]
    fn validate_rejects_future_version() {
        let data = json!({
            "version": 2,
            "providers": {}
        });
        assert!(!validate_manifest(&data));
    }

    #[test]
    fn validate_rejects_missing_or_low_version() {
        // Missing version.
        assert!(!validate_manifest(&json!({"providers": {}})));
        // version < 1 is still an int <= SUPPORTED, so the Python check
        // (version > SUPPORTED) accepts it. Mirror that exactly.
        assert!(validate_manifest(&json!({"version": 0, "providers": {}})));
    }

    #[test]
    fn validate_rejects_bool_version() {
        // bool is a Number-like in Python but isinstance(True, int) is True;
        // however in JSON a bool is Value::Bool, not a Number, so we reject.
        // This is acceptable since manifests use a real integer.
        assert!(!validate_manifest(&json!({"version": true, "providers": {}})));
    }

    #[test]
    fn validate_rejects_non_dict_provider() {
        let data = json!({
            "version": 1,
            "providers": {"openrouter": []}
        });
        assert!(!validate_manifest(&data));
    }

    #[test]
    fn validate_rejects_models_not_list() {
        let data = json!({
            "version": 1,
            "providers": {"openrouter": {"models": {}}}
        });
        assert!(!validate_manifest(&data));
    }

    #[test]
    fn validate_rejects_model_without_id() {
        let data = json!({
            "version": 1,
            "providers": {"openrouter": {"models": [{"description": "x"}]}}
        });
        assert!(!validate_manifest(&data));
    }

    #[test]
    fn validate_rejects_blank_id() {
        let data = json!({
            "version": 1,
            "providers": {"openrouter": {"models": [{"id": "   "}]}}
        });
        assert!(!validate_manifest(&data));
    }

    #[test]
    fn validate_empty_providers_ok() {
        let data = json!({"version": 1, "providers": {}});
        assert!(validate_manifest(&data));
    }

    #[test]
    fn str_or_empty_behaviour() {
        assert_eq!(str_or_empty(None), "");
        assert_eq!(str_or_empty(Some(&json!(null))), "");
        assert_eq!(str_or_empty(Some(&json!(""))), "");
        assert_eq!(str_or_empty(Some(&json!("hi"))), "hi");
        assert_eq!(str_or_empty(Some(&json!(0))), "");
        assert_eq!(str_or_empty(Some(&json!(42))), "42");
        assert_eq!(str_or_empty(Some(&json!(false))), "");
        assert_eq!(str_or_empty(Some(&json!(true))), "True");
    }

    #[test]
    fn to_float_or_behaviour() {
        assert_eq!(to_float_or(None, 24.0), 24.0);
        assert_eq!(to_float_or(Some(&json!(0)), 24.0), 24.0);
        assert_eq!(to_float_or(Some(&json!(12)), 24.0), 12.0);
        assert_eq!(to_float_or(Some(&json!(12.5)), 24.0), 12.5);
        assert_eq!(to_float_or(Some(&json!("")), 24.0), 24.0);
        assert_eq!(to_float_or(Some(&json!("6")), 24.0), 6.0);
        assert_eq!(to_float_or(Some(&json!("bad")), 24.0), 24.0);
    }

    #[test]
    fn json_truthy_behaviour() {
        assert!(!json_truthy(&json!(null)));
        assert!(!json_truthy(&json!(false)));
        assert!(json_truthy(&json!(true)));
        assert!(!json_truthy(&json!(0)));
        assert!(json_truthy(&json!(1)));
        assert!(!json_truthy(&json!("")));
        assert!(json_truthy(&json!("x")));
        assert!(!json_truthy(&json!([])));
        assert!(!json_truthy(&json!({})));
    }

    #[test]
    fn user_agent_shape() {
        assert!(hermes_user_agent().starts_with("hermes-cli/"));
    }

    #[test]
    fn extract_openrouter_from_block() {
        // Exercise the extraction logic directly via a fabricated block by
        // mirroring get_curated_openrouter_models on a known block.
        let block = json!({
            "models": [
                {"id": "a/b", "description": "rec"},
                {"id": "  ", "description": "skip-blank"},
                {"id": "c/d"},
                {"description": "no-id"},
                "not-an-object"
            ]
        });
        let models = block.get("models").and_then(Value::as_array).unwrap();
        let mut out: Vec<(String, String)> = Vec::new();
        for m in models {
            let m = match m.as_object() {
                Some(o) => o,
                None => continue,
            };
            let mid = str_or_empty(m.get("id"));
            let mid = mid.trim();
            if mid.is_empty() {
                continue;
            }
            let desc = str_or_empty(m.get("description"));
            out.push((mid.to_string(), desc));
        }
        assert_eq!(
            out,
            vec![
                ("a/b".to_string(), "rec".to_string()),
                ("c/d".to_string(), String::new()),
            ]
        );
    }
}
