//! Sticker description cache for Telegram.
//!
//! Faithful port of `gateway/sticker_cache.py`: caches vision-generated sticker
//! descriptions keyed by Telegram `file_unique_id` in
//! `<hermes_home>/sticker_cache.json`, plus the warm-style injection-text
//! builders. Pure JSON-file + string formatting.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

/// Vision prompt for describing stickers — kept concise to save tokens.
pub const STICKER_VISION_PROMPT: &str = "Describe this sticker in 1-2 sentences. Focus on what it depicts -- character, action, emotion. Be concise and objective.";

fn cache_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join("sticker_cache.json")
}

fn load_cache(hermes_home: &Path) -> Map<String, Value> {
    let path = cache_path(hermes_home);
    if !path.exists() {
        return Map::new();
    }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn save_cache(hermes_home: &Path, cache: &Map<String, Value>) -> std::io::Result<()> {
    let path = cache_path(hermes_home);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Python writes indent=2, ensure_ascii=False.
    let text = serde_json::to_string_pretty(&Value::Object(cache.clone()))?;
    std::fs::write(&path, text)
}

/// Look up a cached sticker description by `file_unique_id`. Returns the stored
/// `{description, emoji, set_name, cached_at}` object, or None.
pub fn get_cached_description(hermes_home: &Path, file_unique_id: &str) -> Option<Value> {
    load_cache(hermes_home).get(file_unique_id).cloned()
}

/// Store a sticker description in the cache. `now` is the cache timestamp
/// (`time.time()` seconds); callers pass it so the function stays pure-ish.
pub fn cache_sticker_description(
    hermes_home: &Path,
    file_unique_id: &str,
    description: &str,
    emoji: &str,
    set_name: &str,
    now: f64,
) -> std::io::Result<()> {
    let mut cache = load_cache(hermes_home);
    cache.insert(
        file_unique_id.to_string(),
        json!({
            "description": description,
            "emoji": emoji,
            "set_name": set_name,
            "cached_at": now,
        }),
    );
    save_cache(hermes_home, &cache)
}

/// Convenience wrapper that stamps `cached_at` with the current time.
pub fn cache_sticker_description_now(
    hermes_home: &Path,
    file_unique_id: &str,
    description: &str,
    emoji: &str,
    set_name: &str,
) -> std::io::Result<()> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    cache_sticker_description(hermes_home, file_unique_id, description, emoji, set_name, now)
}

/// Build the warm-style injection text for a sticker description.
/// Port of `build_sticker_injection`.
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

/// Build injection text for animated/video stickers that can't be analyzed.
/// Port of `build_animated_sticker_injection`.
pub fn build_animated_sticker_injection(emoji: &str) -> String {
    if !emoji.is_empty() {
        format!(
            "[The user sent an animated sticker {emoji}~ I can't see animated ones yet, but the emoji suggests: {emoji}]"
        )
    } else {
        "[The user sent an animated sticker~ I can't see animated ones yet]".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn cache_round_trip() {
        let temp = TempDir::new().unwrap();
        let home = temp.path();
        assert!(get_cached_description(home, "abc").is_none());
        cache_sticker_description(home, "abc", "A cat waving", "😀", "MyPack", 1234.5).unwrap();
        let got = get_cached_description(home, "abc").unwrap();
        assert_eq!(got["description"], "A cat waving");
        assert_eq!(got["emoji"], "😀");
        assert_eq!(got["set_name"], "MyPack");
        assert_eq!(got["cached_at"], 1234.5);
        // a second key coexists
        cache_sticker_description(home, "def", "dog", "", "", 1.0).unwrap();
        assert!(get_cached_description(home, "abc").is_some());
        assert_eq!(get_cached_description(home, "def").unwrap()["description"], "dog");
    }

    #[test]
    fn injection_text_variants() {
        assert_eq!(
            build_sticker_injection("A cat waving", "😀", "MyPack"),
            "[The user sent a sticker 😀 from \"MyPack\"~ It shows: \"A cat waving\" (=^.w.^=)]"
        );
        assert_eq!(
            build_sticker_injection("hi", "😀", ""),
            "[The user sent a sticker 😀~ It shows: \"hi\" (=^.w.^=)]"
        );
        assert_eq!(
            build_sticker_injection("hi", "", ""),
            "[The user sent a sticker~ It shows: \"hi\" (=^.w.^=)]"
        );
    }

    #[test]
    fn animated_injection_variants() {
        assert_eq!(
            build_animated_sticker_injection("🎉"),
            "[The user sent an animated sticker 🎉~ I can't see animated ones yet, but the emoji suggests: 🎉]"
        );
        assert_eq!(
            build_animated_sticker_injection(""),
            "[The user sent an animated sticker~ I can't see animated ones yet]"
        );
    }
}
