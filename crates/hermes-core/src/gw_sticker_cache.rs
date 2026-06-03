//! Sticker description cache for Telegram.
//!
//! Native Rust port of `gateway/sticker_cache.py`.
//!
//! When users send stickers, we describe them via the vision tool and cache
//! the descriptions keyed by `file_unique_id` so we don't re-analyze the same
//! sticker image on every send. Descriptions are concise (1-2 sentences).
//!
//! Cache location: `~/.hermes/sticker_cache.json` (honours `HERMES_HOME` via
//! [`crate::mod_hermes_constants::get_hermes_home`]).
//!
//! Behavioural notes vs. the Python original:
//! * `_load_cache` swallows JSON decode / IO errors and returns an empty map,
//!   matching the Python `except (json.JSONDecodeError, OSError)` behaviour.
//! * `cached_at` is stored as a floating-point Unix timestamp (seconds), exactly
//!   like Python's `time.time()`.
//! * JSON is written with two-space indentation and `ensure_ascii=False`
//!   (`serde_json::to_string_pretty` already preserves non-ASCII characters).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::mod_hermes_constants::get_hermes_home;

/// Vision prompt for describing stickers -- kept concise to save tokens.
pub const STICKER_VISION_PROMPT: &str = "Describe this sticker in 1-2 sentences. \
Focus on what it depicts -- character, action, emotion. Be concise and objective.";

/// One cached sticker description entry.
///
/// Mirrors the Python dict with keys `{description, emoji, set_name, cached_at}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StickerEntry {
    pub description: String,
    #[serde(default)]
    pub emoji: String,
    #[serde(default)]
    pub set_name: String,
    pub cached_at: f64,
}

/// Resolve the on-disk cache path: `<hermes_home>/sticker_cache.json`.
///
/// Evaluated lazily (not a module-level constant) so it always reflects the
/// current `HERMES_HOME` environment, matching how the Python module behaves
/// across test profiles.
pub fn cache_path() -> PathBuf {
    get_hermes_home().join("sticker_cache.json")
}

/// Load the sticker cache from disk.
///
/// Returns an empty map if the file is missing, unreadable, or contains
/// invalid JSON (mirrors the Python `except (JSONDecodeError, OSError)`).
fn load_cache() -> BTreeMap<String, StickerEntry> {
    let path = cache_path();
    if !path.exists() {
        return BTreeMap::new();
    }
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return BTreeMap::new(),
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Save the sticker cache to disk, creating the parent directory if needed.
fn save_cache(cache: &BTreeMap<String, StickerEntry>) -> std::io::Result<()> {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(cache)
        .unwrap_or_else(|_| "{}".to_string());
    std::fs::write(&path, json)
}

/// Current Unix time in seconds as a float, like Python's `time.time()`.
fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Look up a cached sticker description.
///
/// Returns the [`StickerEntry`] for `file_unique_id` or `None`.
pub fn get_cached_description(file_unique_id: &str) -> Option<StickerEntry> {
    let cache = load_cache();
    cache.get(file_unique_id).cloned()
}

/// Store a sticker description in the cache.
///
/// * `file_unique_id` - Telegram's stable sticker identifier.
/// * `description`     - Vision-generated description text.
/// * `emoji`           - Associated emoji (e.g. `"😀"`).
/// * `set_name`        - Sticker set name if available.
pub fn cache_sticker_description(
    file_unique_id: &str,
    description: &str,
    emoji: &str,
    set_name: &str,
) -> std::io::Result<()> {
    let mut cache = load_cache();
    cache.insert(
        file_unique_id.to_string(),
        StickerEntry {
            description: description.to_string(),
            emoji: emoji.to_string(),
            set_name: set_name.to_string(),
            cached_at: now_seconds(),
        },
    );
    save_cache(&cache)
}

/// Build the warm-style injection text for a sticker description.
///
/// Returns a string like:
/// `[The user sent a sticker 😀 from "MyPack"~ It shows: "A cat waving" (=^.w.^=)]`
pub fn build_sticker_injection(description: &str, emoji: &str, set_name: &str) -> String {
    let context = if !set_name.is_empty() && !emoji.is_empty() {
        format!(" {emoji} from \"{set_name}\"")
    } else if !emoji.is_empty() {
        format!(" {emoji}")
    } else {
        String::new()
    };

    format!("[The user sent a sticker{context}~ It shows: \"{description}\" (=^.w.^=)]")
}

/// Build injection text for animated/video stickers we can't analyze.
pub fn build_animated_sticker_injection(emoji: &str) -> String {
    if !emoji.is_empty() {
        format!(
            "[The user sent an animated sticker {emoji}~ \
I can't see animated ones yet, but the emoji suggests: {emoji}]"
        )
    } else {
        "[The user sent an animated sticker~ I can't see animated ones yet]".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate HERMES_HOME so they don't race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn with_temp_home<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let unique = std::env::temp_dir().join(format!(
            "hermes_sticker_test_{}_{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&unique).unwrap();
        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &unique);
        }
        f();
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
        let _ = std::fs::remove_dir_all(&unique);
    }

    #[test]
    fn test_injection_with_set_and_emoji() {
        let s = build_sticker_injection("A cat waving", "😀", "MyPack");
        assert_eq!(
            s,
            "[The user sent a sticker 😀 from \"MyPack\"~ It shows: \"A cat waving\" (=^.w.^=)]"
        );
    }

    #[test]
    fn test_injection_emoji_only() {
        let s = build_sticker_injection("A dog", "🐶", "");
        assert_eq!(s, "[The user sent a sticker 🐶~ It shows: \"A dog\" (=^.w.^=)]");
    }

    #[test]
    fn test_injection_set_without_emoji_is_ignored() {
        // Python: set_name only matters when emoji is also present.
        let s = build_sticker_injection("Plain", "", "SomePack");
        assert_eq!(s, "[The user sent a sticker~ It shows: \"Plain\" (=^.w.^=)]");
    }

    #[test]
    fn test_animated_injection() {
        assert_eq!(
            build_animated_sticker_injection("🎉"),
            "[The user sent an animated sticker 🎉~ I can't see animated ones yet, but the emoji suggests: 🎉]"
        );
        assert_eq!(
            build_animated_sticker_injection(""),
            "[The user sent an animated sticker~ I can't see animated ones yet]"
        );
    }

    #[test]
    fn test_cache_roundtrip() {
        with_temp_home(|| {
            assert!(get_cached_description("uid123").is_none());

            cache_sticker_description("uid123", "A waving cat", "😺", "Cats").unwrap();

            let entry = get_cached_description("uid123").expect("entry present");
            assert_eq!(entry.description, "A waving cat");
            assert_eq!(entry.emoji, "😺");
            assert_eq!(entry.set_name, "Cats");
            assert!(entry.cached_at > 0.0);

            assert!(get_cached_description("missing").is_none());
        });
    }

    #[test]
    fn test_corrupt_cache_returns_empty() {
        with_temp_home(|| {
            std::fs::write(cache_path(), b"{ not valid json").unwrap();
            assert!(get_cached_description("anything").is_none());
            // A subsequent write should still succeed (overwrites corrupt file).
            cache_sticker_description("x", "desc", "", "").unwrap();
            assert!(get_cached_description("x").is_some());
        });
    }
}
