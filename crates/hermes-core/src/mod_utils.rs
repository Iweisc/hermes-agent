//! Shared utility functions for hermes-agent.
//!
//! Native Rust port of `utils.py`. Provides truthy coercion, environment
//! variable helpers, atomic JSON/YAML file writes (symlink- and
//! permission-preserving), safe JSON parsing, proxy URL normalisation, and
//! base-URL hostname matching helpers.

use std::env;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The project's shared set of strings treated as boolean-true.
///
/// Mirrors `TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})`.
pub const TRUTHY_STRINGS: [&str; 4] = ["1", "true", "yes", "on"];

/// A loosely-typed value, used to faithfully reproduce Python's `is_truthy_value`
/// which accepts `None`, `bool`, `str`, or arbitrary objects.
#[derive(Debug, Clone, PartialEq)]
pub enum TruthyInput {
    /// Python `None`.
    None,
    /// Python `bool`.
    Bool(bool),
    /// Python `str`.
    Str(String),
    /// Any other object — coerced via Python's `bool(value)` truthiness.
    /// `true` means a truthy object, `false` a falsy one.
    Other(bool),
}

impl From<bool> for TruthyInput {
    fn from(b: bool) -> Self {
        TruthyInput::Bool(b)
    }
}

impl From<&str> for TruthyInput {
    fn from(s: &str) -> Self {
        TruthyInput::Str(s.to_string())
    }
}

impl From<String> for TruthyInput {
    fn from(s: String) -> Self {
        TruthyInput::Str(s)
    }
}

/// Return true when `s`, after trim+lowercase, is one of the shared truthy
/// strings. This is the core comparison used by [`is_truthy_value`] for the
/// string branch.
pub fn is_truthy_str(s: &str) -> bool {
    let normalized = s.trim().to_lowercase();
    TRUTHY_STRINGS.contains(&normalized.as_str())
}

/// Coerce bool-ish values using the project's shared truthy string set.
///
/// Faithful port of Python's `is_truthy_value(value, default=False)`:
/// - `None` -> returns `default`
/// - `bool` -> returns the bool as-is
/// - `str`  -> trim+lowercase, membership test against [`TRUTHY_STRINGS`]
/// - other  -> Python `bool(value)` truthiness (carried in `Other`)
pub fn is_truthy_value(value: &TruthyInput, default: bool) -> bool {
    match value {
        TruthyInput::None => default,
        TruthyInput::Bool(b) => *b,
        TruthyInput::Str(s) => is_truthy_str(s),
        TruthyInput::Other(b) => *b,
    }
}

/// Return true when an environment variable is set to a truthy value.
///
/// Mirrors `env_var_enabled(name, default="")`: reads the env var (falling
/// back to `default` when unset) and evaluates it as a string truthy value
/// with an effective default of `false`.
pub fn env_var_enabled(name: &str, default: &str) -> bool {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    is_truthy_str(&raw)
}

/// Capture the permission bits (`S_IMODE`) of `path` if it exists, else `None`.
///
/// Mirrors `_preserve_file_mode`. Returns `None` on any I/O error or when the
/// path does not exist.
fn preserve_file_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match fs::metadata(path) {
            Ok(meta) => Some(meta.permissions().mode() & 0o7777),
            Err(_) => None,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// Re-apply `mode` to `path` after an atomic replace.
///
/// Mirrors `_restore_file_mode`. `tempfile`-style creation yields restrictive
/// `0o600` permissions; after the replace the target would inherit those,
/// breaking Docker/NAS mounts. Restores the captured mode, swallowing errors.
fn restore_file_mode(path: &Path, mode: Option<u32>) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(m) = mode {
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(m));
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
}

/// Atomically move `tmp_path` onto `target`, preserving symlinks.
///
/// Faithful port of `atomic_replace`. When `target` is a symlink, the symlink
/// is resolved first so the rename writes through to the real file in-place and
/// the symlink survives (GitHub #16743). For non-symlink and non-existent paths
/// behaviour is identical to a plain rename.
///
/// Returns the resolved real path used for the replace, so callers that need to
/// re-apply permissions can target it instead of the symlink.
pub fn atomic_replace(tmp_path: &Path, target: &Path) -> std::io::Result<PathBuf> {
    let is_link = fs::symlink_metadata(target)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let real_path: PathBuf = if is_link {
        // os.path.realpath canonicalises the symlink chain.
        fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf())
    } else {
        target.to_path_buf()
    };
    fs::rename(tmp_path, &real_path)?;
    Ok(real_path)
}

/// Create a uniquely-named temp file in `dir` with the given `prefix`/`suffix`.
///
/// Mirrors `tempfile.mkstemp(dir=..., prefix=..., suffix=...)`. Returns the
/// path of an exclusively-created file. On Unix the file is created with
/// `0o600` permissions, matching `mkstemp`.
fn mkstemp(dir: &Path, prefix: &str, suffix: &str) -> std::io::Result<PathBuf> {
    use std::fs::OpenOptions;
    // Seed from process id + a monotonic-ish counter for uniqueness without a
    // new dependency. Retry on collision.
    let pid = std::process::id();
    for attempt in 0..10_000u64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!("{prefix}{pid}_{nanos}_{attempt}{suffix}");
        let candidate = dir.join(name);
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&candidate) {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not create unique temp file",
    ))
}

/// The file stem (filename without final extension) used for the temp prefix.
fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Write serialized text to a file atomically (temp file + fsync + rename),
/// preserving symlinks and the original file's permission bits.
///
/// Shared engine for [`atomic_json_write`] and [`atomic_yaml_write`].
fn atomic_write_bytes(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let original_mode = preserve_file_mode(path);

    let stem = file_stem(path);
    let prefix = format!(".{stem}_");
    let tmp_path = mkstemp(parent, &prefix, ".tmp")?;

    // Closure so we can guarantee temp-file cleanup on any error (the Python
    // code catches BaseException to clean up before re-raising).
    let result = (|| -> std::io::Result<PathBuf> {
        {
            let mut f = fs::OpenOptions::new().write(true).open(&tmp_path)?;
            f.write_all(contents)?;
            f.flush()?;
            f.sync_all()?; // os.fsync
        }
        // Preserve symlinks — swap in-place on the real file (GitHub #16743).
        let real_path = atomic_replace(&tmp_path, path)?;
        Ok(real_path)
    })();

    match result {
        Ok(real_path) => {
            restore_file_mode(&real_path, original_mode);
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Write JSON data to a file atomically.
///
/// Faithful port of `atomic_json_write`. Serializes `data` with the given
/// `indent` (using `ensure_ascii=False` equivalent — Rust's `serde_json`
/// pretty/compact serializers do not escape non-ASCII), then writes via temp
/// file + fsync + rename, preserving symlinks and permissions.
///
/// `indent == 0` produces compact output (matching `json.dump(indent=0)`'s
/// near-compact behaviour is approximated as compact here for `indent <= 0`).
pub fn atomic_json_write(
    path: &Path,
    data: &serde_json::Value,
    indent: usize,
) -> std::io::Result<()> {
    let serialized = if indent == 0 {
        serde_json::to_string(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
    } else {
        // serde_json's pretty printer uses 2-space indent. For the project's
        // default (indent=2) this is exact. For other indents we build a custom
        // formatter.
        if indent == 2 {
            serde_json::to_string_pretty(data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        } else {
            let indent_bytes = " ".repeat(indent);
            let formatter =
                serde_json::ser::PrettyFormatter::with_indent(indent_bytes.as_bytes());
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            use serde::Serialize;
            data.serialize(&mut ser)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            String::from_utf8(buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        }
    };
    atomic_write_bytes(path, serialized.as_bytes())
}

/// Write YAML data to a file atomically.
///
/// Faithful port of `atomic_yaml_write`. Serializes `data` to YAML and writes
/// via temp file + fsync + rename, preserving symlinks and permissions.
/// `extra_content`, when supplied, is appended verbatim after the YAML dump.
///
/// Note: `serde_yaml` always emits block style and does not sort keys by
/// default (it preserves mapping order), matching the Python call's
/// `default_flow_style=False, sort_keys=False` defaults.
pub fn atomic_yaml_write(
    path: &Path,
    data: &serde_yaml::Value,
    extra_content: Option<&str>,
) -> std::io::Result<()> {
    let mut serialized = serde_yaml::to_string(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(extra) = extra_content {
        serialized.push_str(extra);
    }
    atomic_write_bytes(path, serialized.as_bytes())
}

// ─── JSON Helpers ─────────────────────────────────────────────────────────────

/// Parse JSON, returning `default` on any parse error.
///
/// Faithful port of `safe_json_loads(text, default=None)`. In Rust the
/// "default" is supplied by the caller; `None` is represented by passing
/// `serde_json::Value::Null` (or use [`safe_json_loads_opt`] for an `Option`).
pub fn safe_json_loads(text: &str, default: serde_json::Value) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or(default)
}

/// Parse JSON, returning `None` on any parse error.
///
/// Convenience variant of [`safe_json_loads`] matching the common Python call
/// `safe_json_loads(text)` where the default is `None`.
pub fn safe_json_loads_opt(text: &str) -> Option<serde_json::Value> {
    serde_json::from_str(text).ok()
}

// ─── Environment Variable Helpers ─────────────────────────────────────────────

/// Read an environment variable as an integer, with fallback.
///
/// Faithful port of `env_int(key, default=0)`: reads the var, trims it, and on
/// empty or non-integer falls back to `default`.
pub fn env_int(key: &str, default: i64) -> i64 {
    let raw = env::var(key).unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return default;
    }
    raw.parse::<i64>().unwrap_or(default)
}

/// Read an environment variable as a boolean.
///
/// Faithful port of `env_bool(key, default=False)`: reads the var as a string
/// truthy value, returning `default` when the var is unset/empty.
pub fn env_bool(key: &str, default: bool) -> bool {
    // Python: `is_truthy_value(os.getenv(key, ""), default=default)`.
    // `os.getenv(key, "")` never returns None, so the str branch of
    // is_truthy_value always runs and `default` only ever matters if the env
    // value were None (impossible here). An unset var thus yields
    // is_truthy_str("") == false, NOT `default`. Reproduce exactly; `default`
    // is retained for signature/API parity.
    let _ = default;
    match env::var(key) {
        Ok(v) => is_truthy_str(&v),
        Err(_) => false,
    }
}

// ─── Proxy Helpers ────────────────────────────────────────────────────────────

/// Supported proxy environment variable keys, in the Python ordering.
pub const PROXY_ENV_KEYS: [&str; 6] = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
];

/// Normalize proxy URLs for httpx/aiohttp compatibility.
///
/// Faithful port of `normalize_proxy_url`. Trims the input; an empty/`None`
/// value yields `None`. A `socks://` prefix (case-insensitive) is rewritten to
/// the explicit `socks5://` scheme; everything else is returned unchanged.
pub fn normalize_proxy_url(proxy_url: Option<&str>) -> Option<String> {
    let candidate = proxy_url.unwrap_or("").trim();
    if candidate.is_empty() {
        return None;
    }
    if candidate.to_lowercase().starts_with("socks://") {
        // Preserve the original case of the remainder; only the scheme is
        // rewritten (matches slicing `candidate[len("socks://"):]`).
        let rest = &candidate["socks://".len()..];
        return Some(format!("socks5://{rest}"));
    }
    Some(candidate.to_string())
}

/// Rewrite supported proxy env vars to canonical URL forms in-place.
///
/// Faithful port of `normalize_proxy_env_vars`. For each key in
/// [`PROXY_ENV_KEYS`], normalises the current value and, when the normalised
/// form is non-empty and differs from the original, writes it back to the
/// process environment.
///
/// # Safety
/// Mutates the process environment via `std::env::set_var`, which is not
/// thread-safe on some platforms; call during single-threaded startup, as the
/// Python original does.
pub fn normalize_proxy_env_vars() {
    for key in PROXY_ENV_KEYS {
        let value = env::var(key).unwrap_or_default();
        if let Some(normalized) = normalize_proxy_url(Some(&value)) {
            if normalized != value {
                // SAFETY: edition-2024 requires `unsafe` for env mutation. As
                // the Python original documents, this must run during
                // single-threaded startup before other threads read the env.
                unsafe { env::set_var(key, &normalized); }
            }
        }
    }
}

// ─── URL Parsing Helpers ──────────────────────────────────────────────────────

/// Return the lowercased hostname for a base URL, or `""` if absent.
///
/// Faithful port of `base_url_hostname`. Trims the input; bare hosts (no
/// `://`) are parsed by prefixing `//`. The hostname is lowercased and any
/// trailing dots stripped. Returns `""` when no hostname can be extracted.
///
/// Use exact-hostname comparisons against known provider hosts instead of
/// substring matches on the raw URL (see the Python docstring for the
/// security rationale).
pub fn base_url_hostname(base_url: &str) -> String {
    let raw = base_url.trim();
    if raw.is_empty() {
        return String::new();
    }
    let to_parse = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("//{raw}")
    };
    match url::Url::parse(&to_parse) {
        Ok(parsed) => parsed
            .host_str()
            .map(|h| h.to_lowercase().trim_end_matches('.').to_string())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

/// Return true when the base URL's hostname is `domain` or a subdomain.
///
/// Faithful port of `base_url_host_matches`. Safer counterpart to a substring
/// `domain in base_url` check. Accepts bare hosts, full URLs, and URLs with
/// paths. The match is exact-host or proper-subdomain only.
pub fn base_url_host_matches(base_url: &str, domain: &str) -> bool {
    let hostname = base_url_hostname(base_url);
    if hostname.is_empty() {
        return false;
    }
    let domain = domain.trim().to_lowercase();
    let domain = domain.trim_end_matches('.');
    if domain.is_empty() {
        return false;
    }
    hostname == domain || hostname.ends_with(&format!(".{domain}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise env-mutating tests to avoid cross-test interference.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn truthy_strings_membership() {
        assert!(is_truthy_str("1"));
        assert!(is_truthy_str("TRUE"));
        assert!(is_truthy_str("  Yes  "));
        assert!(is_truthy_str("on"));
        assert!(!is_truthy_str("0"));
        assert!(!is_truthy_str("false"));
        assert!(!is_truthy_str(""));
        assert!(!is_truthy_str("enabled"));
    }

    #[test]
    fn is_truthy_value_branches() {
        assert!(!is_truthy_value(&TruthyInput::None, false));
        assert!(is_truthy_value(&TruthyInput::None, true));
        assert!(is_truthy_value(&TruthyInput::Bool(true), false));
        assert!(!is_truthy_value(&TruthyInput::Bool(false), true));
        assert!(is_truthy_value(&TruthyInput::Str("yes".into()), false));
        assert!(!is_truthy_value(&TruthyInput::Str("nope".into()), false));
        assert!(is_truthy_value(&TruthyInput::Other(true), false));
        assert!(!is_truthy_value(&TruthyInput::Other(false), true));
    }

    #[test]
    fn env_helpers() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_TEST_INT", "  42 "); }
        assert_eq!(env_int("HERMES_TEST_INT", 7), 42);
        unsafe { env::set_var("HERMES_TEST_INT", "notnum"); }
        assert_eq!(env_int("HERMES_TEST_INT", 7), 7);
        unsafe { env::set_var("HERMES_TEST_INT", "   "); }
        assert_eq!(env_int("HERMES_TEST_INT", 7), 7);
        unsafe { env::remove_var("HERMES_TEST_INT"); }
        assert_eq!(env_int("HERMES_TEST_INT", 9), 9);

        unsafe { env::set_var("HERMES_TEST_BOOL", "on"); }
        assert!(env_bool("HERMES_TEST_BOOL", false));
        unsafe { env::set_var("HERMES_TEST_BOOL", "off"); }
        assert!(!env_bool("HERMES_TEST_BOOL", true));
        unsafe { env::remove_var("HERMES_TEST_BOOL"); }
        // Unset -> getenv("") -> not truthy -> false (default unused per Python).
        assert!(!env_bool("HERMES_TEST_BOOL", true));

        unsafe { env::set_var("HERMES_TEST_EN", "1"); }
        assert!(env_var_enabled("HERMES_TEST_EN", ""));
        unsafe { env::remove_var("HERMES_TEST_EN"); }
        assert!(!env_var_enabled("HERMES_TEST_EN", ""));
        assert!(env_var_enabled("HERMES_TEST_EN", "true"));
    }

    #[test]
    fn safe_json() {
        assert_eq!(
            safe_json_loads("{\"a\":1}", serde_json::Value::Null),
            serde_json::json!({"a": 1})
        );
        assert_eq!(
            safe_json_loads("not json", serde_json::json!("fallback")),
            serde_json::json!("fallback")
        );
        assert!(safe_json_loads_opt("[1,2]").is_some());
        assert!(safe_json_loads_opt("{bad").is_none());
    }

    #[test]
    fn proxy_normalisation() {
        assert_eq!(normalize_proxy_url(None), None);
        assert_eq!(normalize_proxy_url(Some("   ")), None);
        assert_eq!(
            normalize_proxy_url(Some("socks://127.0.0.1:1080")),
            Some("socks5://127.0.0.1:1080".to_string())
        );
        assert_eq!(
            normalize_proxy_url(Some("SOCKS://Host:9")),
            Some("socks5://Host:9".to_string())
        );
        assert_eq!(
            normalize_proxy_url(Some("http://proxy:8080")),
            Some("http://proxy:8080".to_string())
        );
        assert_eq!(
            normalize_proxy_url(Some("  https://p:1  ")),
            Some("https://p:1".to_string())
        );
    }

    #[test]
    fn proxy_env_rewrite() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("ALL_PROXY", "socks://1.2.3.4:1080"); }
        normalize_proxy_env_vars();
        assert_eq!(env::var("ALL_PROXY").unwrap(), "socks5://1.2.3.4:1080");
        unsafe { env::remove_var("ALL_PROXY"); }
    }

    #[test]
    fn hostname_extraction() {
        assert_eq!(base_url_hostname(""), "");
        assert_eq!(base_url_hostname("   "), "");
        assert_eq!(
            base_url_hostname("https://API.OpenAI.com/v1"),
            "api.openai.com"
        );
        assert_eq!(base_url_hostname("api.x.ai"), "api.x.ai");
        assert_eq!(base_url_hostname("api.anthropic.com."), "api.anthropic.com");
        assert_eq!(
            base_url_hostname("https://proxy.test/api.openai.com/v1"),
            "proxy.test"
        );
    }

    #[test]
    fn host_matches() {
        assert!(base_url_host_matches(
            "https://api.moonshot.ai/v1",
            "moonshot.ai"
        ));
        assert!(base_url_host_matches("https://moonshot.ai", "moonshot.ai"));
        assert!(!base_url_host_matches(
            "https://evil.com/moonshot.ai/v1",
            "moonshot.ai"
        ));
        assert!(!base_url_host_matches(
            "https://moonshot.ai.evil/v1",
            "moonshot.ai"
        ));
        assert!(!base_url_host_matches("", "moonshot.ai"));
        assert!(!base_url_host_matches("https://moonshot.ai", ""));
    }

    #[test]
    fn atomic_json_roundtrip() {
        let dir = std::env::temp_dir().join(format!("hermes_utils_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("nested").join("data.json");
        let value = serde_json::json!({"name": "héllo", "n": 3});
        atomic_json_write(&target, &value, 2).unwrap();
        let read = fs::read_to_string(&target).unwrap();
        // ensure_ascii=False equivalent: non-ASCII preserved literally.
        assert!(read.contains("héllo"));
        let parsed: serde_json::Value = serde_json::from_str(&read).unwrap();
        assert_eq!(parsed, value);
        // No leftover temp files.
        let leftovers: Vec<_> = fs::read_dir(target.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_yaml_roundtrip() {
        let dir = std::env::temp_dir().join(format!("hermes_utils_yaml_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("conf.yaml");
        let value: serde_yaml::Value =
            serde_yaml::from_str("key: value\nlist:\n  - a\n  - b\n").unwrap();
        atomic_yaml_write(&target, &value, Some("# trailing comment\n")).unwrap();
        let read = fs::read_to_string(&target).unwrap();
        assert!(read.contains("key: value"));
        assert!(read.ends_with("# trailing comment\n"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_replace_preserves_symlink() {
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!("hermes_utils_link_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real.json");
        fs::write(&real, "{\"old\":1}").unwrap();
        let link = dir.join("link.json");
        symlink(&real, &link).unwrap();

        atomic_json_write(&link, &serde_json::json!({"new": 2}), 2).unwrap();

        // The symlink must still be a symlink.
        let meta = fs::symlink_metadata(&link).unwrap();
        assert!(meta.file_type().is_symlink());
        // The real file received the new content.
        let parsed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&real).unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!({"new": 2}));
        let _ = fs::remove_dir_all(&dir);
    }
}
