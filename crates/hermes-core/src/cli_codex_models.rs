//! Codex model discovery from API, local cache, and config.
//!
//! Faithful native Rust port of `hermes_cli/codex_models.py`.
//!
//! Resolution order in [`get_codex_model_ids`]: live API (if a token is
//! provided) > `config.toml` default model > local `models_cache.json` >
//! hardcoded defaults. Synthetic forward-compat Codex slugs are appended so
//! newer GPT-5 Codex variants surface even when live discovery omits them.
//!
//! Note: the Python source reads `config.toml` via `tomllib`. There is no TOML
//! crate available in the target crate, so [`read_default_model`] extracts the
//! top-level `model = "..."` key with a small regex. This matches the only
//! shape the Python code consumes (`payload.get("model")` on the root table).

use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

/// Hardcoded fallback catalog, highest-priority first.
pub const DEFAULT_CODEX_MODELS: &[&str] = &[
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
];

/// `(synthetic_model, template_models)` pairs used by [`add_forward_compat_models`].
///
/// If `synthetic_model` is absent but any of its `template_models` is present,
/// the synthetic slug is appended. Mirrors Clawdbot's synthetic catalog /
/// forward-compat behaviour for GPT-5 Codex variants.
const FORWARD_COMPAT_TEMPLATE_MODELS: &[(&str, &[&str])] = &[
    ("gpt-5.5", &["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"]),
    ("gpt-5.4-mini", &["gpt-5.3-codex", "gpt-5.2-codex"]),
    ("gpt-5.4", &["gpt-5.3-codex", "gpt-5.2-codex"]),
    ("gpt-5.3-codex", &["gpt-5.2-codex"]),
];

/// Default rank applied when an entry has no numeric `priority`.
const DEFAULT_RANK: i64 = 10_000;

/// Add Clawdbot-style synthetic forward-compat Codex models.
///
/// Dedupes the incoming ids (first occurrence wins), then appends any synthetic
/// slug whose template models are present.
pub fn add_forward_compat_models<I, S>(model_ids: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut ordered: Vec<String> = Vec::new();
    for model_id in model_ids {
        let model_id = model_id.into();
        if !ordered.iter().any(|m| m == &model_id) {
            ordered.push(model_id);
        }
    }

    for (synthetic_model, template_models) in FORWARD_COMPAT_TEMPLATE_MODELS {
        if ordered.iter().any(|m| m == synthetic_model) {
            continue;
        }
        if template_models
            .iter()
            .any(|template| ordered.iter().any(|m| m == template))
        {
            ordered.push((*synthetic_model).to_string());
        }
    }

    ordered
}

/// Extract a sortable `(rank, slug)` from a JSON model entry, applying the same
/// visibility / `supported_in_api` filtering the Python code uses.
///
/// Returns `None` when the entry should be skipped.
fn extract_sortable_entry(item: &Value) -> Option<(i64, String)> {
    let obj = item.as_object()?;

    let slug = obj.get("slug")?.as_str()?;
    let slug = slug.trim();
    if slug.is_empty() {
        return None;
    }

    // `supported_in_api is False` -> skip. Only an explicit boolean `false`
    // filters; missing / non-bool values pass through.
    if obj.get("supported_in_api").and_then(Value::as_bool) == Some(false) {
        return None;
    }

    if let Some(visibility) = obj.get("visibility").and_then(Value::as_str) {
        let v = visibility.trim().to_lowercase();
        if v == "hide" || v == "hidden" {
            return None;
        }
    }

    // int(priority) when priority is int|float, else DEFAULT_RANK.
    let rank = match obj.get("priority") {
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                // Mirror Python int(float) truncation toward zero.
                f.trunc() as i64
            } else {
                DEFAULT_RANK
            }
        }
        _ => DEFAULT_RANK,
    };

    Some((rank, slug.to_string()))
}

/// Sort `(rank, slug)` pairs by `(rank, slug)`, dedupe by slug (first wins),
/// and return the slugs. Shared between API and cache parsing.
fn sort_and_dedupe(mut sortable: Vec<(i64, String)>) -> Vec<String> {
    sortable.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut deduped: Vec<String> = Vec::new();
    for (_, slug) in sortable {
        if !deduped.iter().any(|s| s == &slug) {
            deduped.push(slug);
        }
    }
    deduped
}

/// Parse the API/cache response body shape: an object with a `models` array.
/// Returns visible model slugs sorted by priority. Exposed for testing.
pub fn parse_models_response(data: &Value) -> Vec<String> {
    let entries = data
        .as_object()
        .and_then(|o| o.get("models"))
        .and_then(Value::as_array);

    let mut sortable: Vec<(i64, String)> = Vec::new();
    if let Some(entries) = entries {
        for item in entries {
            if let Some(pair) = extract_sortable_entry(item) {
                sortable.push(pair);
            }
        }
    }
    sort_and_dedupe(sortable)
}

/// Fetch available models from the Codex API. Returns visible models sorted by
/// priority, or an empty vec on any failure (non-200, network error, bad JSON).
///
/// Note: the Python version applies `_add_forward_compat_models` to the result
/// here; the public [`get_codex_model_ids`] also applies it again to the API
/// result. We keep this function's output as the raw sorted slugs (matching the
/// Python intermediate) and let the caller apply forward-compat, so behaviour
/// matches `get_codex_model_ids`.
pub fn fetch_models_from_api(access_token: &str) -> Vec<String> {
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(exc) => {
            log::debug!("Failed to build Codex models HTTP client: {exc}");
            return Vec::new();
        }
    };

    let resp = match client
        .get("https://chatgpt.com/backend-api/codex/models?client_version=1.0.0")
        .header("Authorization", format!("Bearer {access_token}"))
        .send()
    {
        Ok(r) => r,
        Err(exc) => {
            log::debug!("Failed to fetch Codex models from API: {exc}");
            return Vec::new();
        }
    };

    if resp.status().as_u16() != 200 {
        return Vec::new();
    }

    let data: Value = match resp.json() {
        Ok(v) => v,
        Err(exc) => {
            log::debug!("Failed to fetch Codex models from API: {exc}");
            return Vec::new();
        }
    };

    parse_models_response(&data)
}

/// Read the top-level `model = "..."` key from `<codex_home>/config.toml`.
///
/// Returns `None` when the file is absent, unreadable, or has no usable `model`
/// key. We do a minimal, top-level-only extraction (no nested tables), which is
/// all the Python source consumes.
pub fn read_default_model(codex_home: &Path) -> Option<String> {
    let config_path = codex_home.join("config.toml");
    if !config_path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&config_path).ok()?;
    parse_default_model_from_toml(&text)
}

/// Extract the first top-level `model = "..."` assignment from TOML text.
///
/// Skips lines once a `[table]` header is encountered so we only read the root
/// table (matching `payload.get("model")` on the parsed root dict). Exposed for
/// testing.
pub fn parse_default_model_from_toml(text: &str) -> Option<String> {
    // Quoted-string form: model = "value" or model = 'value'.
    let quoted = Regex::new(r#"^\s*model\s*=\s*("([^"]*)"|'([^']*)')\s*$"#).ok()?;

    for raw_line in text.lines() {
        let line = strip_toml_comment(raw_line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Stop at the first table/array-of-tables header: keys after this are
        // no longer in the root table.
        if trimmed.starts_with('[') {
            break;
        }
        if let Some(caps) = quoted.captures(trimmed) {
            let value = caps
                .get(2)
                .or_else(|| caps.get(3))
                .map(|m| m.as_str())
                .unwrap_or("");
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
            return None;
        }
    }
    None
}

/// Remove an unquoted trailing `# comment` from a TOML line.
///
/// Quotes are respected so `#` inside a string literal is not treated as a
/// comment start.
fn strip_toml_comment(line: &str) -> String {
    let mut in_single = false;
    let mut in_double = false;
    let mut out = String::with_capacity(line.len());
    for ch in line.chars() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => break,
            _ => {}
        }
        out.push(ch);
    }
    out
}

/// Read and rank model slugs from `<codex_home>/models_cache.json`.
///
/// Returns an empty vec when the file is absent or unparseable.
pub fn read_cache_models(codex_home: &Path) -> Vec<String> {
    let cache_path = codex_home.join("models_cache.json");
    if !cache_path.exists() {
        return Vec::new();
    }
    let text = match std::fs::read_to_string(&cache_path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let raw: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    // Same shape/filtering as the API response.
    parse_models_response(&raw)
}

/// Resolve the Codex home directory: `$CODEX_HOME` (trimmed, if set and
/// non-empty) else `~/.codex`.
pub fn codex_home() -> PathBuf {
    let from_env = std::env::var("CODEX_HOME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let base = match from_env {
        Some(s) => s,
        None => {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            return home.join(".codex");
        }
    };
    expand_user(&base)
}

/// Expand a leading `~` in a path string to the user's home directory,
/// approximating Python's `Path.expanduser()` for the common cases.
fn expand_user(path: &str) -> PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

/// Return available Codex model IDs, trying the live API first, then local
/// sources.
///
/// Resolution order: API (live, if `access_token` provided) > `config.toml`
/// default > local cache > hardcoded defaults. The final list always passes
/// through [`add_forward_compat_models`].
pub fn get_codex_model_ids(access_token: Option<&str>) -> Vec<String> {
    let home = codex_home();

    // Try live API if we have a (non-empty) token.
    if let Some(token) = access_token {
        if !token.is_empty() {
            let api_models = fetch_models_from_api(token);
            if !api_models.is_empty() {
                return add_forward_compat_models(api_models);
            }
        }
    }

    // Fall back to local sources.
    let mut ordered: Vec<String> = Vec::new();

    if let Some(default_model) = read_default_model(&home) {
        ordered.push(default_model);
    }

    for model_id in read_cache_models(&home) {
        if !ordered.iter().any(|m| m == &model_id) {
            ordered.push(model_id);
        }
    }

    for model_id in DEFAULT_CODEX_MODELS {
        if !ordered.iter().any(|m| m == model_id) {
            ordered.push((*model_id).to_string());
        }
    }

    add_forward_compat_models(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn forward_compat_appends_missing_synthetic() {
        // gpt-5.4 present, gpt-5.5 absent -> gpt-5.5 appended.
        let out = add_forward_compat_models(vec!["gpt-5.4".to_string()]);
        assert!(out.contains(&"gpt-5.5".to_string()));
        // gpt-5.4 also triggers gpt-5.4-mini (templates include gpt-5.3-codex/
        // gpt-5.2-codex only) -> gpt-5.4-mini NOT triggered by gpt-5.4 alone.
        assert!(!out.contains(&"gpt-5.4-mini".to_string()));
    }

    #[test]
    fn forward_compat_dedupes_input_first_wins() {
        let out = add_forward_compat_models(vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
        ]);
        assert_eq!(out, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn forward_compat_skips_present_synthetic() {
        let out = add_forward_compat_models(vec![
            "gpt-5.5".to_string(),
            "gpt-5.4".to_string(),
        ]);
        // gpt-5.5 already present, so not re-added; count of gpt-5.5 == 1.
        assert_eq!(out.iter().filter(|m| m.as_str() == "gpt-5.5").count(), 1);
    }

    #[test]
    fn forward_compat_single_pass_from_5_2() {
        // gpt-5.2-codex triggers gpt-5.4-mini, gpt-5.4, gpt-5.3-codex (their
        // templates include gpt-5.2-codex). The loop is a SINGLE pass over the
        // template table, so gpt-5.5 — whose templates only become present
        // during this same pass — is evaluated FIRST and therefore NOT added.
        let out = add_forward_compat_models(vec!["gpt-5.2-codex".to_string()]);
        assert_eq!(out[0], "gpt-5.2-codex");
        assert!(out.contains(&"gpt-5.4-mini".to_string()));
        assert!(out.contains(&"gpt-5.4".to_string()));
        assert!(out.contains(&"gpt-5.3-codex".to_string()));
        assert!(!out.contains(&"gpt-5.5".to_string()));
        // Exact ordering: input, then table order for triggered synthetics.
        assert_eq!(
            out,
            vec![
                "gpt-5.2-codex".to_string(),
                "gpt-5.4-mini".to_string(),
                "gpt-5.4".to_string(),
                "gpt-5.3-codex".to_string(),
            ]
        );
    }

    #[test]
    fn parse_models_sorts_by_priority_then_slug() {
        let data = json!({
            "models": [
                {"slug": "zeta", "priority": 1},
                {"slug": "alpha", "priority": 1},
                {"slug": "first", "priority": 0},
            ]
        });
        let out = parse_models_response(&data);
        assert_eq!(out, vec!["first", "alpha", "zeta"]);
    }

    #[test]
    fn parse_models_filters_visibility_and_unsupported() {
        let data = json!({
            "models": [
                {"slug": "ok", "priority": 0},
                {"slug": "hidden-one", "priority": 0, "visibility": "Hidden"},
                {"slug": "hide-one", "priority": 0, "visibility": "hide"},
                {"slug": "no-api", "priority": 0, "supported_in_api": false},
                {"slug": "yes-api", "priority": 0, "supported_in_api": true},
            ]
        });
        let out = parse_models_response(&data);
        assert!(out.contains(&"ok".to_string()));
        assert!(out.contains(&"yes-api".to_string()));
        assert!(!out.contains(&"hidden-one".to_string()));
        assert!(!out.contains(&"hide-one".to_string()));
        assert!(!out.contains(&"no-api".to_string()));
    }

    #[test]
    fn parse_models_skips_blank_and_nonstring_slug() {
        let data = json!({
            "models": [
                {"slug": "   "},
                {"slug": 123},
                {"nope": "x"},
                "not-a-dict",
                {"slug": "  good  "},
            ]
        });
        let out = parse_models_response(&data);
        assert_eq!(out, vec!["good"]);
    }

    #[test]
    fn parse_models_default_rank_for_missing_priority() {
        let data = json!({
            "models": [
                {"slug": "no-prio"},
                {"slug": "low", "priority": 5},
            ]
        });
        // priority 5 < default 10000 -> "low" first.
        let out = parse_models_response(&data);
        assert_eq!(out, vec!["low", "no-prio"]);
    }

    #[test]
    fn parse_models_float_priority_truncated() {
        let data = json!({
            "models": [
                {"slug": "b", "priority": 2.9},
                {"slug": "a", "priority": 1.1},
            ]
        });
        let out = parse_models_response(&data);
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn parse_models_empty_when_not_object() {
        assert!(parse_models_response(&json!([])).is_empty());
        assert!(parse_models_response(&json!("x")).is_empty());
        assert!(parse_models_response(&json!({"other": 1})).is_empty());
    }

    #[test]
    fn toml_default_model_quoted() {
        assert_eq!(
            parse_default_model_from_toml("model = \"gpt-5.4\"\n"),
            Some("gpt-5.4".to_string())
        );
        assert_eq!(
            parse_default_model_from_toml("model = 'gpt-5.3-codex'"),
            Some("gpt-5.3-codex".to_string())
        );
    }

    #[test]
    fn toml_default_model_with_comment_and_other_keys() {
        let text = "approval = \"auto\"\nmodel = \"gpt-5.5\"  # the model\nother = 1\n";
        assert_eq!(
            parse_default_model_from_toml(text),
            Some("gpt-5.5".to_string())
        );
    }

    #[test]
    fn toml_default_model_stops_at_table_header() {
        // model only appears inside a sub-table -> not read (root only).
        let text = "[profile]\nmodel = \"gpt-5.4\"\n";
        assert_eq!(parse_default_model_from_toml(text), None);
    }

    #[test]
    fn toml_default_model_blank_value_is_none() {
        assert_eq!(parse_default_model_from_toml("model = \"\"\n"), None);
    }

    #[test]
    fn toml_default_model_absent() {
        assert_eq!(parse_default_model_from_toml("approval = \"auto\"\n"), None);
    }

    #[test]
    fn read_cache_models_missing_file_is_empty() {
        let dir = std::env::temp_dir().join("hermes_codex_models_test_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_cache_models(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_cache_models_parses_file() {
        let dir =
            std::env::temp_dir().join(format!("hermes_codex_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let body = json!({
            "models": [
                {"slug": "b", "priority": 2},
                {"slug": "a", "priority": 1},
            ]
        });
        std::fs::write(dir.join("models_cache.json"), body.to_string()).unwrap();
        assert_eq!(read_cache_models(&dir), vec!["a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_default_model_reads_config_toml() {
        let dir =
            std::env::temp_dir().join(format!("hermes_codex_conf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "model = \"gpt-5.4\"\n").unwrap();
        assert_eq!(read_default_model(&dir), Some("gpt-5.4".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_codex_model_ids_no_token_includes_defaults() {
        // With no token and a (likely) empty codex home, defaults + forward
        // compat should be present.
        let out = get_codex_model_ids(None);
        assert!(out.contains(&"gpt-5.5".to_string()));
        assert!(out.contains(&"gpt-5.1-codex-mini".to_string()));
    }

    #[test]
    fn get_codex_model_ids_empty_token_falls_back() {
        let out = get_codex_model_ids(Some(""));
        assert!(out.contains(&"gpt-5.5".to_string()));
    }
}
