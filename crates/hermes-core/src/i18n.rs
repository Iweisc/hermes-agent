//! Lightweight internationalization (i18n) for Hermes static user-facing
//! messages.
//!
//! Scope (thin slice, by design): only the highest-impact static strings shown
//! to the user by Hermes itself -- approval prompts, a handful of gateway slash
//! command replies, restart-drain notices. Agent-generated output, log lines,
//! error tracebacks, tool outputs, and slash-command descriptions all stay in
//! English.
//!
//! Catalog files live under `locales/<lang>.yaml` at the repo root. Each
//! catalog is a flat dict keyed by dotted paths (e.g. `approval.choose` or
//! `gateway.approval_expired`). Missing keys fall back to English; if English
//! is missing too, the key path itself is returned so a broken catalog never
//! crashes the agent.
//!
//! Usage:
//!
//! ```ignore
//! use hermes_core::i18n;
//! println!("{}", i18n::t("approval.choose_long"));                  // current lang
//! println!("{}", i18n::t_fmt("gateway.draining", None, &[("count", "3")])); // {count}
//! println!("{}", i18n::t_lang("approval.choose_long", Some("zh")));        // override
//! ```
//!
//! Language resolution order:
//!   1. Explicit `lang` argument passed to [`t_lang`] / [`t_fmt`]
//!   2. `HERMES_LANGUAGE` environment variable (for tests / quick override)
//!   3. `display.language` from config.yaml
//!   4. `"en"` (baseline)
//!
//! Supported languages: en, zh, ja, de, es, fr, tr, uk. Unknown values fall
//! back to en.
//!
//! This is a native Rust port of `agent/i18n.py`.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde_yaml::Value as YamlValue;

/// Languages this build ships catalogs for. Anything else normalizes to
/// [`DEFAULT_LANGUAGE`].
pub const SUPPORTED_LANGUAGES: [&str; 8] = ["en", "zh", "ja", "de", "es", "fr", "tr", "uk"];

/// Baseline language used when nothing else resolves and as the fallback
/// catalog for missing keys.
pub const DEFAULT_LANGUAGE: &str = "en";

/// Accept a few natural aliases so users who type "chinese" / "zh-CN" / "jp"
/// get the right catalog instead of silently falling back to English.
const LANGUAGE_ALIASES: &[(&str, &str)] = &[
    ("english", "en"),
    ("en-us", "en"),
    ("en-gb", "en"),
    ("chinese", "zh"),
    ("mandarin", "zh"),
    ("zh-cn", "zh"),
    ("zh-tw", "zh"),
    ("zh-hans", "zh"),
    ("zh-hant", "zh"),
    ("japanese", "ja"),
    ("jp", "ja"),
    ("ja-jp", "ja"),
    ("german", "de"),
    ("deutsch", "de"),
    ("de-de", "de"),
    ("spanish", "es"),
    ("español", "es"),
    ("espanol", "es"),
    ("es-es", "es"),
    ("es-mx", "es"),
    ("french", "fr"),
    ("français", "fr"),
    ("france", "fr"),
    ("fr-fr", "fr"),
    ("fr-be", "fr"),
    ("fr-ca", "fr"),
    ("fr-ch", "fr"),
    ("ukrainian", "uk"),
    ("ukrainisch", "uk"),
    ("українська", "uk"),
    ("uk-ua", "uk"),
    ("ua", "uk"),
    ("turkish", "tr"),
    ("türkçe", "tr"),
    ("tr-tr", "tr"),
];

/// Per-language flattened catalog cache, keyed by `(lang, path)` like the
/// Python implementation.
fn catalog_cache() -> &'static Mutex<HashMap<(String, String), HashMap<String, String>>> {
    static CACHE: OnceLock<Mutex<HashMap<(String, String), HashMap<String, String>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Cached resolution of `display.language` from config.yaml (mirrors the
/// Python `lru_cache(maxsize=1)`). `None` slot inside means "computed and the
/// answer was no config language".
fn config_language_cache() -> &'static Mutex<Option<Option<String>>> {
    static CACHE: OnceLock<Mutex<Option<Option<String>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Return the directory containing locale YAML files.
///
/// `HERMES_LOCALES_DIR` overrides the location (used by tests and by installs
/// that relocate the catalogs); otherwise it lives next to the repo root, the
/// same place `agent/i18n.py` looks (`<repo>/locales`).
fn locales_dir() -> PathBuf {
    if let Some(dir) = env::var_os("HERMES_LOCALES_DIR") {
        return PathBuf::from(dir);
    }
    // crates/hermes-core -> crates -> repo root, then /locales.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("..").join("..");
    let root = root.canonicalize().unwrap_or(root);
    root.join("locales")
}

/// Normalize a user-supplied language value to a supported code.
///
/// Accepts supported codes directly, common aliases (`chinese` -> `zh`), and
/// case-insensitive regional tags (`zh-CN` -> `zh`). Returns the default
/// language for unknown / empty values.
pub fn normalize_lang(value: &str) -> String {
    let key = value.trim().to_lowercase();
    if key.is_empty() {
        return DEFAULT_LANGUAGE.to_string();
    }
    if SUPPORTED_LANGUAGES.contains(&key.as_str()) {
        return key;
    }
    if let Some((_, code)) = LANGUAGE_ALIASES.iter().find(|(alias, _)| *alias == key) {
        return (*code).to_string();
    }
    // Try stripping a region suffix (e.g. "pt-br" -> "pt" won't be supported,
    // but "zh-CN" -> "zh" will).
    let base = key.split('-').next().unwrap_or(&key);
    if SUPPORTED_LANGUAGES.contains(&base) {
        return base.to_string();
    }
    DEFAULT_LANGUAGE.to_string()
}

/// Recursively flatten a YAML node into a dotted-key, text-only map.
/// Non-string, non-mapping leaves are ignored -- catalogs are text-only.
fn flatten_into(node: &YamlValue, prefix: &str, out: &mut HashMap<String, String>) {
    match node {
        YamlValue::Mapping(map) => {
            for (key, value) in map {
                // Mirror Python's `str(key)`: scalars use their textual form.
                let key_str = match key {
                    YamlValue::String(s) => s.clone(),
                    YamlValue::Bool(b) => b.to_string(),
                    YamlValue::Number(n) => n.to_string(),
                    YamlValue::Null => "null".to_string(),
                    _ => continue,
                };
                let child_key = if prefix.is_empty() {
                    key_str
                } else {
                    format!("{prefix}.{key_str}")
                };
                flatten_into(value, &child_key, out);
            }
        }
        YamlValue::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        _ => {}
    }
}

/// Load and flatten one locale YAML file into a dotted-key map.
///
/// YAML files can be nested for human readability; this produces the flat key
/// space [`t`] expects. Cached per-language for the process. Missing or broken
/// catalogs yield an empty map (and are cached as such) rather than erroring.
fn load_catalog(lang: &str) -> HashMap<String, String> {
    let path = locales_dir().join(format!("{lang}.yaml"));
    let cache_key = (lang.to_string(), path.to_string_lossy().to_string());

    {
        let cache = catalog_cache().lock().unwrap();
        if let Some(cached) = cache.get(&cache_key) {
            return cached.clone();
        }
    }

    if !path.is_file() {
        log::debug!("i18n catalog missing for {lang} at {}", path.display());
        catalog_cache()
            .lock()
            .unwrap()
            .insert(cache_key, HashMap::new());
        return HashMap::new();
    }

    let flat = match std::fs::read_to_string(&path) {
        Ok(text) => match serde_yaml::from_str::<YamlValue>(&text) {
            Ok(raw) => {
                let mut flat = HashMap::new();
                flatten_into(&raw, "", &mut flat);
                flat
            }
            Err(exc) => {
                log::warn!("Failed to load i18n catalog {}: {exc}", path.display());
                HashMap::new()
            }
        },
        Err(exc) => {
            log::warn!("Failed to load i18n catalog {}: {exc}", path.display());
            HashMap::new()
        }
    };

    catalog_cache()
        .lock()
        .unwrap()
        .insert(cache_key, flat.clone());
    flat
}

/// Default Hermes home (`$HERMES_HOME`, else `~/.hermes`, else `.hermes`).
fn default_hermes_home() -> PathBuf {
    env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

/// Read `display.language` from config.yaml, normalized, or `None` if unset /
/// unreadable. Any failure is swallowed (like the Python try/except) so i18n
/// never crashes a caller.
fn read_config_language() -> Option<String> {
    let path = default_hermes_home().join("config.yaml");
    let text = std::fs::read_to_string(&path).ok()?;
    let doc: YamlValue = serde_yaml::from_str(&text).ok()?;
    let lang = doc.get("display")?.get("language")?;
    let lang = lang.as_str()?;
    if lang.is_empty() {
        return None;
    }
    Some(normalize_lang(lang))
}

/// Read `display.language` from config.yaml once per process.
///
/// Cached because [`t`] is called in hot paths (every approval prompt, every
/// gateway reply) and re-reading YAML each call would be wasteful.
/// [`reset_language_cache`] clears this when config changes at runtime.
fn config_language_cached() -> Option<String> {
    let mut slot = config_language_cache().lock().unwrap();
    if let Some(cached) = slot.as_ref() {
        return cached.clone();
    }
    let resolved = read_config_language();
    *slot = Some(resolved.clone());
    resolved
}

/// Invalidate cached language resolution and catalogs.
///
/// Call after the config's `display.language` changes if a running process
/// needs to pick it up without restart.
pub fn reset_language_cache() {
    *config_language_cache().lock().unwrap() = None;
    catalog_cache().lock().unwrap().clear();
}

/// Resolve the active language using env > config > default order.
pub fn get_language() -> String {
    if let Some(env_lang) = env::var_os("HERMES_LANGUAGE") {
        let env_lang = env_lang.to_string_lossy();
        if !env_lang.is_empty() {
            return normalize_lang(&env_lang);
        }
    }
    if let Some(cfg_lang) = config_language_cached() {
        return cfg_lang;
    }
    DEFAULT_LANGUAGE.to_string()
}

/// Resolve a catalog value for `key` in `target`, falling back to English, then
/// to the bare key. Returns the raw (un-formatted) string.
fn lookup(key: &str, target: &str) -> String {
    let catalog = load_catalog(target);
    if let Some(value) = catalog.get(key) {
        return value.clone();
    }
    if target != DEFAULT_LANGUAGE {
        // Fall through to English rather than showing a key path to the user.
        if let Some(value) = load_catalog(DEFAULT_LANGUAGE).get(key) {
            return value.clone();
        }
    }
    // Last-ditch: return the key itself. A broken catalog should not crash
    // anything; it just looks ugly until someone fixes it.
    log::debug!("i18n miss: key={key:?} lang={target:?}");
    key.to_string()
}

/// Substitute Python-`str.format`-style named placeholders (`{name}`) in
/// `template` from `kwargs`.
///
/// Mirrors enough of `str.format` for the catalog's needs: named `{field}`
/// substitution and literal `{{` / `}}` escaping. If a referenced field is
/// missing, or braces are unbalanced, this returns `Err` so the caller can fall
/// back to the unformatted template -- matching the Python behavior of catching
/// `KeyError`/`IndexError`/`ValueError` and returning the raw value.
fn apply_format(template: &str, kwargs: &[(&str, &str)]) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    out.push('{');
                    continue;
                }
                let mut field = String::new();
                let mut closed = false;
                for nc in chars.by_ref() {
                    if nc == '}' {
                        closed = true;
                        break;
                    }
                    field.push(nc);
                }
                if !closed {
                    return Err(format!("unterminated field: {{{field}"));
                }
                // Strip any `:spec`/`!conv` suffix; catalogs use plain names but
                // be lenient so a `{count:>3}` style entry never panics.
                let name = field
                    .split(|ch| ch == ':' || ch == '!')
                    .next()
                    .unwrap_or("")
                    .trim();
                if name.is_empty() {
                    // Positional `{}` is unsupported by these catalogs.
                    return Err("positional field {} unsupported".to_string());
                }
                match kwargs.iter().find(|(k, _)| *k == name) {
                    Some((_, v)) => out.push_str(v),
                    None => return Err(format!("missing key: {name}")),
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    out.push('}');
                } else {
                    return Err("single '}' in format string".to_string());
                }
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Translate a dotted key to the active (env/config/default) language.
///
/// Equivalent to Python `t(key)` with no `lang` and no format kwargs.
pub fn t(key: &str) -> String {
    lookup(key, &get_language())
}

/// Translate a dotted key with an optional explicit language override.
///
/// `lang` (when `Some` and non-empty) takes precedence over env + config, after
/// normalization. Equivalent to Python `t(key, lang=...)`.
pub fn t_lang(key: &str, lang: Option<&str>) -> String {
    let target = resolve_target(lang);
    lookup(key, &target)
}

/// Translate a dotted key with optional language override and `str.format`-style
/// named substitution arguments.
///
/// `t_fmt("gateway.draining", None, &[("count", "3")])` expects a catalog entry
/// with a `{count}` placeholder. If formatting fails (missing key, bad braces),
/// the raw, unformatted catalog value is returned -- matching the Python
/// fallback. Equivalent to Python `t(key, lang=..., **kwargs)`.
pub fn t_fmt(key: &str, lang: Option<&str>, kwargs: &[(&str, &str)]) -> String {
    let target = resolve_target(lang);
    let value = lookup(key, &target);
    if kwargs.is_empty() {
        return value;
    }
    match apply_format(&value, kwargs) {
        Ok(formatted) => formatted,
        Err(exc) => {
            log::warn!(
                "i18n format failed for key={key:?} lang={target:?} kwargs={kwargs:?}: {exc}"
            );
            value
        }
    }
}

/// Decide the target language for an optional override, mirroring
/// `_normalize_lang(lang) if lang else get_language()`.
fn resolve_target(lang: Option<&str>) -> String {
    match lang {
        Some(l) if !l.is_empty() => normalize_lang(l),
        _ => get_language(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests: they mutate process-global env + shared caches.
    static TEST_GUARD: StdMutex<()> = StdMutex::new(());

    struct Env {
        _guard: std::sync::MutexGuard<'static, ()>,
        dir: std::path::PathBuf,
    }

    impl Env {
        fn new() -> Self {
            let guard = TEST_GUARD.lock().unwrap_or_else(|p| p.into_inner());
            let dir = std::env::temp_dir().join(format!("hermes_i18n_test_{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            unsafe {
                std::env::set_var("HERMES_LOCALES_DIR", &dir);
                std::env::remove_var("HERMES_LANGUAGE");
                std::env::set_var("HERMES_HOME", &dir);
            }
            reset_language_cache();
            Env { _guard: guard, dir }
        }

        fn write_catalog(&self, lang: &str, body: &str) {
            std::fs::write(self.dir.join(format!("{lang}.yaml")), body).unwrap();
            reset_language_cache();
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            for lang in SUPPORTED_LANGUAGES {
                let _ = std::fs::remove_file(self.dir.join(format!("{lang}.yaml")));
            }
            let _ = std::fs::remove_file(self.dir.join("config.yaml"));
            unsafe {
                std::env::remove_var("HERMES_LOCALES_DIR");
                std::env::remove_var("HERMES_LANGUAGE");
                std::env::remove_var("HERMES_HOME");
            }
            reset_language_cache();
        }
    }

    #[test]
    fn normalize_supported_alias_and_region() {
        assert_eq!(normalize_lang("en"), "en");
        assert_eq!(normalize_lang("ZH"), "zh");
        assert_eq!(normalize_lang("  Ja  "), "ja");
        // Alias table.
        assert_eq!(normalize_lang("chinese"), "zh");
        assert_eq!(normalize_lang("jp"), "ja");
        assert_eq!(normalize_lang("español"), "es");
        // Region-tag stripping for a supported base.
        assert_eq!(normalize_lang("zh-CN"), "zh");
        assert_eq!(normalize_lang("fr-FR"), "fr");
        // Unknown / empty -> default.
        assert_eq!(normalize_lang(""), "en");
        assert_eq!(normalize_lang("   "), "en");
        assert_eq!(normalize_lang("pt-br"), "en");
        assert_eq!(normalize_lang("klingon"), "en");
    }

    #[test]
    fn flattens_nested_catalog() {
        let env = Env::new();
        env.write_catalog(
            "en",
            "approval:\n  choose: \"pick one\"\ngateway:\n  draining: \"draining {count}\"\n",
        );
        assert_eq!(t_lang("approval.choose", Some("en")), "pick one");
        assert_eq!(
            t_fmt("gateway.draining", Some("en"), &[("count", "3")]),
            "draining 3"
        );
    }

    #[test]
    fn falls_back_to_english_then_key() {
        let env = Env::new();
        env.write_catalog("en", "approval:\n  choose: \"english choose\"\n");
        env.write_catalog("zh", "approval:\n  other: \"中文\"\n");
        // Present only in en -> english fallback.
        assert_eq!(t_lang("approval.choose", Some("zh")), "english choose");
        // Present in neither -> bare key.
        assert_eq!(t_lang("approval.missing", Some("zh")), "approval.missing");
    }

    #[test]
    fn explicit_lang_wins() {
        let env = Env::new();
        env.write_catalog("en", "k: \"hello\"\n");
        env.write_catalog("zh", "k: \"你好\"\n");
        assert_eq!(t_lang("k", Some("zh")), "你好");
        assert_eq!(t_lang("k", Some("en")), "hello");
        // Alias override also works.
        assert_eq!(t_lang("k", Some("chinese")), "你好");
    }

    #[test]
    fn env_overrides_config_and_default() {
        let env = Env::new();
        env.write_catalog("en", "k: \"en\"\n");
        env.write_catalog("de", "k: \"de\"\n");
        unsafe {
            std::env::set_var("HERMES_LANGUAGE", "de");
        }
        reset_language_cache();
        assert_eq!(get_language(), "de");
        assert_eq!(t("k"), "de");
        unsafe {
            std::env::remove_var("HERMES_LANGUAGE");
        }
        reset_language_cache();
        assert_eq!(get_language(), "en");
    }

    #[test]
    fn config_language_used_when_no_env() {
        let env = Env::new();
        env.write_catalog("en", "k: \"en\"\n");
        env.write_catalog("fr", "k: \"fr\"\n");
        std::fs::write(
            env.dir.join("config.yaml"),
            "display:\n  language: fr\n",
        )
        .unwrap();
        reset_language_cache();
        assert_eq!(get_language(), "fr");
        assert_eq!(t("k"), "fr");
    }

    #[test]
    fn format_failure_returns_raw_value() {
        let env = Env::new();
        env.write_catalog("en", "k: \"needs {missing}\"\n");
        // Provide a different kwarg -> format fails -> raw value returned.
        assert_eq!(
            t_fmt("k", Some("en"), &[("count", "3")]),
            "needs {missing}"
        );
    }

    #[test]
    fn format_handles_escaped_braces() {
        assert_eq!(
            apply_format("a {x} b {{lit}}", &[("x", "1")]).unwrap(),
            "a 1 b {lit}"
        );
        assert!(apply_format("{unterminated", &[]).is_err());
        assert!(apply_format("oops }", &[]).is_err());
    }

    #[test]
    fn missing_catalog_yields_key() {
        let env = Env::new();
        let _ = env; // no catalogs written
        assert_eq!(t_lang("anything.here", Some("ja")), "anything.here");
    }
}
