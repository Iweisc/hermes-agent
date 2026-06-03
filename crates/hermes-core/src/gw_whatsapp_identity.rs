//! Shared helpers for canonicalising WhatsApp sender identity.
//!
//! WhatsApp's bridge can surface the same human under two different JID shapes
//! within a single conversation:
//!
//! - LID form: `999999999999999@lid`
//! - Phone form: `15551234567@s.whatsapp.net`
//!
//! Both the authorisation path (gateway/run) and the session-key path
//! (gateway/session) need to collapse these aliases to a single stable
//! identity. This module is the single source of truth for that resolution so
//! the two paths can never drift apart.
//!
//! Public helpers:
//!
//! - [`normalize_whatsapp_identifier`] — strip JID/LID/device/plus syntax
//!   down to the bare numeric identifier.
//! - [`canonical_whatsapp_identifier`] — walk the bridge's
//!   `lid-mapping-*.json` files and return a stable canonical identity
//!   across phone/LID variants.
//! - [`expand_whatsapp_aliases`] — return the full alias set for an
//!   identifier. Used by authorisation code that needs to match any known
//!   form of a sender against an allow-list.
//!
//! Plugins that need per-sender behaviour on WhatsApp (role-based routing,
//! per-contact authorisation, policy gating in a gateway hook) should use
//! [`canonical_whatsapp_identifier`] so their bookkeeping lines up with
//! Hermes' own session keys.
//!
//! Ported from `gateway/whatsapp_identity.py`. The Python module derives the
//! WhatsApp session directory from `hermes_constants.get_hermes_home()`. In
//! line with the other ported gateway modules (e.g. `gateway_mirror`), the
//! Rust port takes `hermes_home: &Path` as an explicit parameter on the
//! filesystem-touching helpers rather than resolving it implicitly.

use std::collections::HashSet;
use std::collections::VecDeque;
use std::path::Path;
use std::path::PathBuf;

use regex::Regex;
use std::sync::OnceLock;

/// WhatsApp JIDs are numeric (or plus-prefixed numeric) with optional
/// `@`, `.` and `:` separators. `\w` is pinned to ASCII so full-width digits /
/// Unicode word chars can't sneak through (we use an explicit ASCII class
/// here, matching the Python `[A-Za-z0-9@.+\-]`).
fn safe_identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9@.+\-]+$").expect("valid identifier regex"))
}

/// Strip WhatsApp JID/LID syntax down to its stable numeric identifier.
///
/// Accepts any of the identifier shapes the WhatsApp bridge may emit:
/// `"60123456789@s.whatsapp.net"`, `"60123456789:47@s.whatsapp.net"`,
/// `"60123456789@lid"`, or a bare `"+601****6789"` / `"60123456789"`.
/// Returns just the numeric identifier (`"60123456789"`) suitable for equality
/// comparisons.
///
/// Mirrors the Python pipeline:
/// `str(value or "").strip().replace("+", "", 1).split(":", 1)[0].split("@", 1)[0]`.
pub fn normalize_whatsapp_identifier(value: &str) -> String {
    // .strip() — trim leading/trailing whitespace.
    let trimmed = value.trim();

    // .replace("+", "", 1) — remove the first '+' only.
    let without_plus: String = match trimmed.find('+') {
        Some(idx) => {
            let mut s = String::with_capacity(trimmed.len() - 1);
            s.push_str(&trimmed[..idx]);
            // skip the single '+' at idx (one byte, ASCII)
            s.push_str(&trimmed[idx + 1..]);
            s
        }
        None => trimmed.to_string(),
    };

    // .split(":", 1)[0] — keep everything before the first ':'.
    let before_colon = match without_plus.split_once(':') {
        Some((head, _)) => head,
        None => without_plus.as_str(),
    };

    // .split("@", 1)[0] — keep everything before the first '@'.
    let before_at = match before_colon.split_once('@') {
        Some((head, _)) => head,
        None => before_colon,
    };

    before_at.to_string()
}

/// Path to the bridge's WhatsApp session directory:
/// `<hermes_home>/whatsapp/session`.
fn session_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("whatsapp").join("session")
}

/// Resolve WhatsApp phone/LID aliases via bridge session mapping files.
///
/// Returns the set of all identifiers transitively reachable through the
/// bridge's `<hermes_home>/whatsapp/session/lid-mapping-*.json` files, starting
/// from `identifier`. The result always includes the normalized input itself,
/// so callers can safely membership-check against the return value without a
/// separate fallback branch.
///
/// Returns an empty set if `identifier` normalizes to empty.
pub fn expand_whatsapp_aliases(hermes_home: &Path, identifier: &str) -> HashSet<String> {
    let normalized = normalize_whatsapp_identifier(identifier);
    if normalized.is_empty() {
        return HashSet::new();
    }

    let dir = session_dir(hermes_home);
    let mut resolved: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    queue.push_back(normalized);

    while let Some(current) = queue.pop_front() {
        if current.is_empty() || resolved.contains(&current) {
            continue;
        }
        // Defense-in-depth: reject identifiers that could sneak path
        // separators / traversal segments into the `lid-mapping-{current}`
        // filename below. The hardcoded `lid-mapping-` prefix already prevents
        // escape via the path component split, but this keeps the identifier
        // space to the characters WhatsApp JIDs actually use.
        if !safe_identifier_re().is_match(&current) {
            continue;
        }

        resolved.insert(current.clone());

        for suffix in ["", "_reverse"] {
            let file_name = format!("lid-mapping-{current}{suffix}.json");
            let mapping_path = dir.join(&file_name);
            if !mapping_path.exists() {
                continue;
            }
            let raw = match std::fs::read_to_string(&mapping_path) {
                Ok(raw) => raw,
                Err(exc) => {
                    log::debug!(
                        "whatsapp_identity: failed to read {}: {}",
                        mapping_path.display(),
                        exc
                    );
                    continue;
                }
            };
            // Python: json.loads(...) then normalize_whatsapp_identifier(...).
            // The mapping file holds a single JSON value (the aliased
            // identifier string). We parse it, coerce to the same shape Python
            // would (str(value or "")), then normalize.
            let parsed: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(exc) => {
                    log::debug!(
                        "whatsapp_identity: failed to read {}: {}",
                        mapping_path.display(),
                        exc
                    );
                    continue;
                }
            };
            let mapped = normalize_whatsapp_identifier(&json_value_to_identifier(&parsed));
            if !mapped.is_empty() && !resolved.contains(&mapped) {
                queue.push_back(mapped);
            }
        }
    }

    resolved
}

/// Coerce a parsed JSON value into the string the Python `str(value or "")`
/// path inside `normalize_whatsapp_identifier` would have produced.
///
/// WhatsApp bridge mapping files store the aliased identifier as a JSON string,
/// but we coerce defensively for robustness. A JSON `null`, `false`, `0`, or
/// empty string is falsy in Python and `str(value or "")` yields `""`; we
/// reproduce that so those degrade to the empty (ignored) alias.
fn json_value_to_identifier(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(false) => String::new(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Number(n) => {
            // Python `0`/`0.0` are falsy -> "".
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        // Containers are truthy if non-empty; their str() form contains no
        // safe-identifier chars beyond brackets/quotes, so they won't match
        // the safe-identifier regex anyway. Reproduce falsy-empty behaviour.
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                String::new()
            } else {
                value.to_string()
            }
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                String::new()
            } else {
                value.to_string()
            }
        }
    }
}

/// Return a stable WhatsApp sender identity across phone-JID/LID variants.
///
/// WhatsApp may surface the same person under either a phone-format JID
/// (`60123456789@s.whatsapp.net`) or a LID (`1234567890@lid`). This applies to
/// a DM `chat_id` *and* to the `participant_id` of a member inside a group
/// chat — both represent a user identity, and the bridge may flip between the
/// two for the same human.
///
/// This helper reads the bridge's `whatsapp/session/lid-mapping-*.json` files,
/// walks the mapping transitively, and picks the shortest (numeric-preferred)
/// alias as the canonical identity.
///
/// Returns an empty string if `identifier` normalizes to empty. If no mapping
/// files exist yet (fresh bridge install), returns the normalized input
/// unchanged.
pub fn canonical_whatsapp_identifier(hermes_home: &Path, identifier: &str) -> String {
    let normalized = normalize_whatsapp_identifier(identifier);
    if normalized.is_empty() {
        return String::new();
    }

    // expand_whatsapp_aliases always includes `normalized` itself in the
    // returned set, so the min below degrades gracefully to `normalized` when
    // no lid-mapping files are present.
    let aliases = expand_whatsapp_aliases(hermes_home, &normalized);

    // Python: min(aliases, key=lambda c: (len(c), c)) — shortest first, then
    // lexicographic. Python `len` is the number of Unicode code points and
    // string comparison is by code point; for the ASCII-only identifiers the
    // safe-identifier regex permits, char-count and char-order match exactly.
    aliases
        .into_iter()
        .min_by(|a, b| {
            (a.chars().count(), a.as_str()).cmp(&(b.chars().count(), b.as_str()))
        })
        // expand_whatsapp_aliases returns a non-empty set here (it includes
        // the non-empty `normalized`), but fall back defensively.
        .unwrap_or(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalize_strips_phone_jid() {
        assert_eq!(
            normalize_whatsapp_identifier("60123456789@s.whatsapp.net"),
            "60123456789"
        );
    }

    #[test]
    fn normalize_strips_device_suffix() {
        assert_eq!(
            normalize_whatsapp_identifier("60123456789:47@s.whatsapp.net"),
            "60123456789"
        );
    }

    #[test]
    fn normalize_strips_lid() {
        assert_eq!(
            normalize_whatsapp_identifier("60123456789@lid"),
            "60123456789"
        );
    }

    #[test]
    fn normalize_strips_leading_plus_only_once() {
        assert_eq!(normalize_whatsapp_identifier("+60123456789"), "60123456789");
        // Only the first '+' is removed (Python .replace("+", "", 1)).
        assert_eq!(normalize_whatsapp_identifier("++60"), "+60");
    }

    #[test]
    fn normalize_trims_whitespace() {
        assert_eq!(
            normalize_whatsapp_identifier("  +60123456789@s.whatsapp.net  "),
            "60123456789"
        );
    }

    #[test]
    fn normalize_empty_inputs() {
        assert_eq!(normalize_whatsapp_identifier(""), "");
        assert_eq!(normalize_whatsapp_identifier("   "), "");
    }

    #[test]
    fn normalize_bare_numeric_passthrough() {
        assert_eq!(normalize_whatsapp_identifier("60123456789"), "60123456789");
    }

    #[test]
    fn expand_empty_for_empty_input() {
        let tmp = std::env::temp_dir().join("hermes_wa_id_empty_test");
        assert!(expand_whatsapp_aliases(&tmp, "").is_empty());
        assert!(expand_whatsapp_aliases(&tmp, "  @lid").is_empty());
    }

    #[test]
    fn expand_includes_self_when_no_mapping() {
        let tmp = std::env::temp_dir().join("hermes_wa_id_self_test_nonexistent");
        let aliases = expand_whatsapp_aliases(&tmp, "60123456789@s.whatsapp.net");
        let expected: HashSet<String> = ["60123456789".to_string()].into_iter().collect();
        assert_eq!(aliases, expected);
    }

    #[test]
    fn expand_follows_mapping_transitively() {
        let base = std::env::temp_dir().join(format!(
            "hermes_wa_id_map_test_{}",
            std::process::id()
        ));
        let dir = session_dir(&base);
        fs::create_dir_all(&dir).unwrap();
        // 60123456789 -> 99999@lid form -> normalizes to 99999
        fs::write(
            dir.join("lid-mapping-60123456789.json"),
            "\"99999@lid\"",
        )
        .unwrap();
        // reverse maps 99999 back to the phone form
        fs::write(
            dir.join("lid-mapping-99999_reverse.json"),
            "\"60123456789@s.whatsapp.net\"",
        )
        .unwrap();

        let aliases = expand_whatsapp_aliases(&base, "60123456789@s.whatsapp.net");
        let expected: HashSet<String> = ["60123456789".to_string(), "99999".to_string()]
            .into_iter()
            .collect();
        assert_eq!(aliases, expected);

        // Canonical picks the shortest, then lexicographically.
        let canonical = canonical_whatsapp_identifier(&base, "60123456789@s.whatsapp.net");
        assert_eq!(canonical, "99999");

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn canonical_empty_for_empty() {
        let tmp = std::env::temp_dir().join("hermes_wa_id_canon_empty");
        assert_eq!(canonical_whatsapp_identifier(&tmp, ""), "");
    }

    #[test]
    fn canonical_passthrough_when_no_mapping() {
        let tmp = std::env::temp_dir().join("hermes_wa_id_canon_passthrough_nonexistent");
        assert_eq!(
            canonical_whatsapp_identifier(&tmp, "+60123456789@lid"),
            "60123456789"
        );
    }

    #[test]
    fn expand_skips_malformed_json() {
        let base = std::env::temp_dir().join(format!(
            "hermes_wa_id_bad_json_{}",
            std::process::id()
        ));
        let dir = session_dir(&base);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("lid-mapping-555.json"), "{ not json").unwrap();
        let aliases = expand_whatsapp_aliases(&base, "555");
        let expected: HashSet<String> = ["555".to_string()].into_iter().collect();
        assert_eq!(aliases, expected);
        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn safe_identifier_regex_rejects_traversal() {
        assert!(!safe_identifier_re().is_match("../etc"));
        assert!(!safe_identifier_re().is_match("a/b"));
        assert!(safe_identifier_re().is_match("60123456789"));
        assert!(safe_identifier_re().is_match("60123456789@s.whatsapp.net"));
    }

    #[test]
    fn canonical_shortest_then_lexicographic() {
        let base = std::env::temp_dir().join(format!(
            "hermes_wa_id_tie_{}",
            std::process::id()
        ));
        let dir = session_dir(&base);
        fs::create_dir_all(&dir).unwrap();
        // same length -> lexicographic; "111" < "999"
        fs::write(dir.join("lid-mapping-999.json"), "\"111\"").unwrap();
        let canonical = canonical_whatsapp_identifier(&base, "999");
        assert_eq!(canonical, "111");
        fs::remove_dir_all(&base).ok();
    }
}
