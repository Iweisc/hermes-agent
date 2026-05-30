//! Shared helpers for canonicalising WhatsApp sender identity.
//!
//! Faithful port of `gateway/whatsapp_identity.py`. WhatsApp's bridge can
//! surface the same human under a LID form (`999...@lid`) or a phone form
//! (`1555...@s.whatsapp.net`); both the auth path and the session-key path must
//! collapse these aliases to one stable identity. Pure string/JSON/file logic.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

/// WhatsApp JIDs are numeric (or plus-prefixed numeric) with optional `@`, `.`,
/// `:`, `-` separators. ASCII-pinned so full-width digits can't sneak through.
fn safe_identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9@.+\-]+$").expect("valid identifier regex"))
}

/// Strip WhatsApp JID/LID syntax down to its stable numeric identifier.
/// Port of `normalize_whatsapp_identifier`: trim, drop a single leading `+`,
/// then take the part before the first `:` and before the first `@`.
pub fn normalize_whatsapp_identifier(value: &str) -> String {
    let trimmed = value.trim();
    // Replace the FIRST '+' only (Python str.replace("+", "", 1)).
    let plus_stripped = match trimmed.find('+') {
        Some(idx) => {
            let mut s = String::with_capacity(trimmed.len() - 1);
            s.push_str(&trimmed[..idx]);
            s.push_str(&trimmed[idx + 1..]);
            s
        }
        None => trimmed.to_string(),
    };
    let before_colon = plus_stripped.split(':').next().unwrap_or("");
    before_colon.split('@').next().unwrap_or("").to_string()
}

/// Resolve WhatsApp phone/LID aliases via the bridge's session mapping files.
/// Port of `expand_whatsapp_aliases`: BFS over
/// `<hermes_home>/whatsapp/session/lid-mapping-<id>{,_reverse}.json`, always
/// including the normalized input. Empty set when the input normalizes empty.
pub fn expand_whatsapp_aliases(hermes_home: &Path, identifier: &str) -> BTreeSet<String> {
    let normalized = normalize_whatsapp_identifier(identifier);
    if normalized.is_empty() {
        return BTreeSet::new();
    }

    let session_dir = hermes_home.join("whatsapp").join("session");
    let mut resolved: BTreeSet<String> = BTreeSet::new();
    let mut queue: Vec<String> = vec![normalized];

    while !queue.is_empty() {
        let current = queue.remove(0);
        if current.is_empty() || resolved.contains(&current) {
            continue;
        }
        // Defense-in-depth: reject identifiers that could inject path
        // separators into the lid-mapping filename.
        if !safe_identifier_re().is_match(&current) {
            continue;
        }
        resolved.insert(current.clone());
        for suffix in ["", "_reverse"] {
            let mapping_path = session_dir.join(format!("lid-mapping-{current}{suffix}.json"));
            if !mapping_path.exists() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&mapping_path) else {
                continue;
            };
            // The mapping file is a JSON string value (the aliased identifier).
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let raw = match &value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let mapped = normalize_whatsapp_identifier(&raw);
            if !mapped.is_empty() && !resolved.contains(&mapped) {
                queue.push(mapped);
            }
        }
    }

    resolved
}

/// Return a stable WhatsApp sender identity across phone-JID/LID variants.
/// Port of `canonical_whatsapp_identifier`: the shortest (numeric-preferred,
/// then lexical) alias from the transitive mapping. Empty input -> "".
pub fn canonical_whatsapp_identifier(hermes_home: &Path, identifier: &str) -> String {
    let normalized = normalize_whatsapp_identifier(identifier);
    if normalized.is_empty() {
        return String::new();
    }
    let aliases = expand_whatsapp_aliases(hermes_home, &normalized);
    // min by (len, lexical); aliases always includes `normalized`.
    aliases
        .into_iter()
        .min_by(|a, b| (a.len(), a).cmp(&(b.len(), b)))
        .unwrap_or(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn normalize_strips_jid_lid_device_plus() {
        assert_eq!(normalize_whatsapp_identifier("60123456789@s.whatsapp.net"), "60123456789");
        assert_eq!(normalize_whatsapp_identifier("60123456789:47@s.whatsapp.net"), "60123456789");
        assert_eq!(normalize_whatsapp_identifier("999999999999999@lid"), "999999999999999");
        assert_eq!(normalize_whatsapp_identifier("+60123456789"), "60123456789");
        assert_eq!(normalize_whatsapp_identifier("  60123456789  "), "60123456789");
        assert_eq!(normalize_whatsapp_identifier(""), "");
    }

    #[test]
    fn expand_includes_self_without_mapping_files() {
        let temp = TempDir::new().unwrap();
        let aliases = expand_whatsapp_aliases(temp.path(), "60123456789@s.whatsapp.net");
        assert_eq!(aliases.len(), 1);
        assert!(aliases.contains("60123456789"));
        // canonical degrades to the normalized input
        assert_eq!(
            canonical_whatsapp_identifier(temp.path(), "60123456789@lid"),
            "60123456789"
        );
    }

    #[test]
    fn expand_walks_mapping_files_and_canonical_picks_shortest() {
        let temp = TempDir::new().unwrap();
        let session = temp.path().join("whatsapp").join("session");
        fs::create_dir_all(&session).unwrap();
        // LID 999... maps to phone 60123 (shorter); phone maps back.
        fs::write(
            session.join("lid-mapping-999999999999999.json"),
            "\"60123@s.whatsapp.net\"",
        )
        .unwrap();
        fs::write(
            session.join("lid-mapping-60123.json"),
            "\"999999999999999@lid\"",
        )
        .unwrap();

        let aliases = expand_whatsapp_aliases(temp.path(), "999999999999999@lid");
        assert!(aliases.contains("999999999999999"));
        assert!(aliases.contains("60123"));
        // canonical = shortest -> "60123"
        assert_eq!(
            canonical_whatsapp_identifier(temp.path(), "999999999999999@lid"),
            "60123"
        );
    }

    #[test]
    fn empty_identifier_yields_empty() {
        let temp = TempDir::new().unwrap();
        assert!(expand_whatsapp_aliases(temp.path(), "  ").is_empty());
        assert_eq!(canonical_whatsapp_identifier(temp.path(), "@lid"), "");
    }
}
