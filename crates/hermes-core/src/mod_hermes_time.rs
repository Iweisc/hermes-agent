//! Timezone-aware clock for Hermes.
//!
//! Provides a single [`now`] helper that returns a timezone-aware datetime
//! based on the user's configured IANA timezone (e.g. `Asia/Kolkata`).
//!
//! Resolution order:
//!   1. `HERMES_TIMEZONE` environment variable
//!   2. `timezone` key in `~/.hermes/config.yaml`
//!   3. Falls back to the server's local time (`Local::now()`).
//!
//! Invalid timezone values log a warning and fall back safely — Hermes never
//! crashes due to a bad timezone string.
//!
//! Port of `hermes_time.py`. The Python module returns a single
//! `datetime` (tz-aware). Rust has no single type that can hold either a
//! fixed-offset local time or a named-zone time interchangeably, so we model
//! the result with [`HermesTime`], a small enum exposing the common operations
//! callers need (formatting, naive/utc conversion, offset).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, FixedOffset, Local, TimeZone, Utc};
use chrono_tz::Tz;

// ── Config path resolution ──────────────────────────────────────────────────
//
// Mirrors `hermes_constants.get_config_path()`. The skill utilities module
// (`crate::ag_skill_utils::default_config_path`) already reproduces this; we
// reference it when available but keep a local fallback so this module never
// blocks on a not-yet-ported dependency.

/// Resolve `~/.hermes/config.yaml`, matching `hermes_constants.get_config_path()`.
fn config_path() -> PathBuf {
    // Prefer the shared helper from ag_skill_utils when present.
    // (It is a flat sibling module in this crate.)
    #[allow(unused_imports)]
    {
        // If ag_skill_utils::default_config_path exists, use it. We can't
        // conditionally compile on the presence of an item, so we replicate
        // the identical layout here. Both yield `~/.hermes/config.yaml`.
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes").join("config.yaml")
}

// ── Cached resolution state ──────────────────────────────────────────────────
//
// Python caches the resolved ZoneInfo in module globals and re-resolves only
// after `reset_cache()`. We use a process-global `Mutex` to match that
// once-and-reuse semantics.

#[derive(Debug, Clone)]
struct CacheState {
    resolved: bool,
    tz: Option<Tz>,
    tz_name: String,
}

impl CacheState {
    const fn empty() -> Self {
        CacheState {
            resolved: false,
            tz: None,
            tz_name: String::new(),
        }
    }
}

static CACHE: Mutex<CacheState> = Mutex::new(CacheState::empty());

// ── Timezone-name resolution ─────────────────────────────────────────────────

/// Read the configured IANA timezone string (or empty string).
///
/// Does file I/O when falling through to `config.yaml`, so callers should cache
/// the result rather than calling on every [`now`].
pub fn resolve_timezone_name() -> String {
    resolve_timezone_name_at(&config_path())
}

/// Testable variant: resolve against an explicit config path.
fn resolve_timezone_name_at(config: &Path) -> String {
    // 1. Environment variable (highest priority — set by Supervisor, etc.)
    if let Ok(tz_env) = std::env::var("HERMES_TIMEZONE") {
        let trimmed = tz_env.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // 2. config.yaml `timezone` key.
    if config.exists() {
        if let Ok(text) = std::fs::read_to_string(config) {
            if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&text) {
                if let Some(tz_cfg) = value
                    .as_mapping()
                    .and_then(|m| m.get(&serde_yaml::Value::String("timezone".to_string())))
                    .and_then(|v| v.as_str())
                {
                    let trimmed = tz_cfg.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }

    String::new()
}

/// Validate and return a [`Tz`], or `None` if the name is empty or invalid.
///
/// Invalid names log a warning (matching the Python behaviour) and return
/// `None`, signalling "fall back to server-local time".
pub fn get_zoneinfo(name: &str) -> Option<Tz> {
    if name.is_empty() {
        return None;
    }
    match name.parse::<Tz>() {
        Ok(tz) => Some(tz),
        Err(err) => {
            log::warn!(
                "Invalid timezone '{}': {}. Falling back to server local time.",
                name,
                err
            );
            None
        }
    }
}

/// Return the user's configured timezone, or `None` (meaning server-local).
///
/// Resolved once and cached. Call [`reset_cache`] after config changes.
pub fn get_timezone() -> Option<Tz> {
    let mut cache = CACHE.lock().expect("hermes_time cache poisoned");
    if !cache.resolved {
        cache.tz_name = resolve_timezone_name();
        cache.tz = get_zoneinfo(&cache.tz_name);
        cache.resolved = true;
    }
    cache.tz
}

/// Force re-resolution on the next [`get_timezone`] / [`now`] call.
///
/// Mirrors the Python `reset_cache()` referenced in the docstrings.
pub fn reset_cache() {
    let mut cache = CACHE.lock().expect("hermes_time cache poisoned");
    *cache = CacheState::empty();
}

// ── The tz-aware "now" value ─────────────────────────────────────────────────

/// A timezone-aware instant, equivalent to the Python tz-aware `datetime`
/// returned by [`now`].
///
/// Either a named IANA zone (when one is configured) or the server's local
/// fixed offset (the fallback). Both variants carry the same UTC instant and
/// expose the wall-clock representation in their respective zone.
#[derive(Debug, Clone, Copy)]
pub enum HermesTime {
    /// Wall-clock time in a configured IANA zone.
    Zoned(DateTime<Tz>),
    /// Server-local time (fixed offset), the no-timezone-configured fallback.
    Local(DateTime<Local>),
}

impl HermesTime {
    /// The instant in UTC.
    pub fn to_utc(&self) -> DateTime<Utc> {
        match self {
            HermesTime::Zoned(dt) => dt.with_timezone(&Utc),
            HermesTime::Local(dt) => dt.with_timezone(&Utc),
        }
    }

    /// The current UTC offset, e.g. `+05:30`.
    pub fn offset(&self) -> FixedOffset {
        match self {
            HermesTime::Zoned(dt) => dt.offset().fix(),
            HermesTime::Local(dt) => *dt.offset(),
        }
    }

    /// Render with a chrono `strftime` format string, in the local wall clock.
    pub fn format(&self, fmt: &str) -> String {
        match self {
            HermesTime::Zoned(dt) => dt.format(fmt).to_string(),
            HermesTime::Local(dt) => dt.format(fmt).to_string(),
        }
    }

    /// ISO-8601 / RFC-3339 representation including the offset.
    pub fn to_rfc3339(&self) -> String {
        match self {
            HermesTime::Zoned(dt) => dt.to_rfc3339(),
            HermesTime::Local(dt) => dt.to_rfc3339(),
        }
    }
}

/// Return the current time as a timezone-aware [`HermesTime`].
///
/// If a valid timezone is configured, returns wall-clock time in that zone.
/// Otherwise returns the server's local time (still offset-aware).
pub fn now() -> HermesTime {
    match get_timezone() {
        Some(tz) => HermesTime::Zoned(Utc::now().with_timezone(&tz)),
        None => HermesTime::Local(Local::now()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex as StdMutex;

    // Serialise tests that touch the HERMES_TIMEZONE env var / global cache.
    static ENV_GUARD: StdMutex<()> = StdMutex::new(());

    fn clear_env() {
        unsafe {
            std::env::remove_var("HERMES_TIMEZONE");
        }
    }

    #[test]
    fn env_var_takes_priority() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe {
            std::env::set_var("HERMES_TIMEZONE", "  Asia/Kolkata  ");
        }
        // Even with a config file present, env wins (and is trimmed).
        let dir = std::env::temp_dir().join("hermes_time_test_env");
        let _ = std::fs::create_dir_all(&dir);
        let cfg = dir.join("config.yaml");
        let mut f = std::fs::File::create(&cfg).unwrap();
        writeln!(f, "timezone: America/New_York").unwrap();

        assert_eq!(resolve_timezone_name_at(&cfg), "Asia/Kolkata");
        clear_env();
    }

    #[test]
    fn reads_config_yaml_when_no_env() {
        let _g = ENV_GUARD.lock().unwrap();
        clear_env();
        let dir = std::env::temp_dir().join("hermes_time_test_cfg");
        let _ = std::fs::create_dir_all(&dir);
        let cfg = dir.join("config.yaml");
        let mut f = std::fs::File::create(&cfg).unwrap();
        writeln!(f, "timezone: '  Europe/London  '").unwrap();

        assert_eq!(resolve_timezone_name_at(&cfg), "Europe/London");
    }

    #[test]
    fn empty_when_no_env_and_no_config() {
        let _g = ENV_GUARD.lock().unwrap();
        clear_env();
        let cfg = std::env::temp_dir().join("hermes_time_test_missing_xyz.yaml");
        let _ = std::fs::remove_file(&cfg);
        assert_eq!(resolve_timezone_name_at(&cfg), "");
    }

    #[test]
    fn config_with_blank_timezone_yields_empty() {
        let _g = ENV_GUARD.lock().unwrap();
        clear_env();
        let dir = std::env::temp_dir().join("hermes_time_test_blank");
        let _ = std::fs::create_dir_all(&dir);
        let cfg = dir.join("config.yaml");
        let mut f = std::fs::File::create(&cfg).unwrap();
        writeln!(f, "timezone: '   '").unwrap();
        assert_eq!(resolve_timezone_name_at(&cfg), "");
    }

    #[test]
    fn valid_zone_parses() {
        let tz = get_zoneinfo("Asia/Kolkata").expect("valid zone");
        assert_eq!(tz, Tz::Asia__Kolkata);
    }

    #[test]
    fn invalid_zone_returns_none() {
        assert!(get_zoneinfo("Not/A_Zone").is_none());
        assert!(get_zoneinfo("").is_none());
    }

    #[test]
    fn now_falls_back_to_local() {
        let _g = ENV_GUARD.lock().unwrap();
        clear_env();
        reset_cache();
        // With no env var and (presumably) no kolkata config, now() should be
        // some valid instant. We just assert it produces a coherent UTC value.
        let t = now();
        let utc = t.to_utc();
        // Round-trips through UTC without panicking; offset is well-defined.
        let _ = t.offset();
        assert!(utc.timestamp() > 0);
        reset_cache();
    }

    #[test]
    fn now_uses_configured_zone_via_env() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe {
            std::env::set_var("HERMES_TIMEZONE", "Asia/Kolkata");
        }
        reset_cache();
        let t = now();
        // Asia/Kolkata is a fixed +05:30 offset (no DST).
        assert_eq!(t.offset(), FixedOffset::east_opt(5 * 3600 + 30 * 60).unwrap());
        clear_env();
        reset_cache();
    }
}
