//! Hermes-managed Camofox state helpers.
//!
//! Provides profile-scoped identity and state directory paths for Camofox
//! persistent browser profiles. When managed persistence is enabled, Hermes
//! sends a deterministic `userId` derived from the active profile so that
//! Camofox can map it to the same persistent browser profile directory
//! across restarts.
//!
//! This is a native Rust port of `tools/browser_camofox_state.py`.

use std::collections::HashMap;
use std::path::PathBuf;

use sha1::{Digest, Sha1};

use crate::mod_hermes_constants::get_hermes_home;

/// Name of the profile-scoped root subdirectory holding browser auth state.
pub const CAMOFOX_STATE_DIR_NAME: &str = "browser_auth";
/// Name of the Camofox-specific subdirectory under the auth root.
pub const CAMOFOX_STATE_SUBDIR: &str = "camofox";

/// The RFC 4122 `NAMESPACE_URL` UUID, as used by Python's `uuid.NAMESPACE_URL`.
///
/// `6ba7b811-9dad-11d1-80b4-00c04fd430c8`
const NAMESPACE_URL_BYTES: [u8; 16] = [
    0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
];

/// Compute a UUIDv5 (SHA-1 based, name-based) from a namespace UUID and a name,
/// returning its canonical 32-character lowercase hex representation **without
/// dashes** — matching Python's `uuid.uuid5(...).hex`.
fn uuid5_hex(namespace: &[u8; 16], name: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(namespace);
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();

    // Take the first 16 bytes of the 20-byte SHA-1 digest.
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);

    // Set the version to 5 (the high nibble of byte 6).
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    // Set the variant to RFC 4122 (the two high bits of byte 8).
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    let mut hex = String::with_capacity(32);
    for b in bytes.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Return the profile-scoped root directory for Camofox persistence.
pub fn get_camofox_state_dir() -> PathBuf {
    get_hermes_home()
        .join(CAMOFOX_STATE_DIR_NAME)
        .join(CAMOFOX_STATE_SUBDIR)
}

/// Return the stable Hermes-managed Camofox identity for this profile.
///
/// The user identity is profile-scoped (same Hermes profile = same `user_id`).
/// The session key is scoped to the logical browser task so newly created tabs
/// within the same profile reuse the same identity contract.
///
/// Returns a map with two keys: `"user_id"` and `"session_key"`, mirroring the
/// Python `Dict[str, str]` return value.
pub fn get_camofox_identity(task_id: Option<&str>) -> HashMap<String, String> {
    let scope_root = get_camofox_state_dir().to_string_lossy().into_owned();
    let logical_scope = task_id.unwrap_or("default");

    let user_digest_full = uuid5_hex(&NAMESPACE_URL_BYTES, &format!("camofox-user:{scope_root}"));
    let user_digest = &user_digest_full[..10];

    let session_digest_full = uuid5_hex(
        &NAMESPACE_URL_BYTES,
        &format!("camofox-session:{scope_root}:{logical_scope}"),
    );
    let session_digest = &session_digest_full[..16];

    let mut out = HashMap::with_capacity(2);
    out.insert("user_id".to_string(), format!("hermes_{user_digest}"));
    out.insert("session_key".to_string(), format!("task_{session_digest}"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer test for UUIDv5 against Python's reference output:
    /// `uuid.uuid5(uuid.NAMESPACE_URL, "python.org").hex`.
    #[test]
    fn uuid5_matches_python_reference() {
        let h = uuid5_hex(&NAMESPACE_URL_BYTES, "python.org");
        assert_eq!(h, "7af94e2b4dd950f09c9a8a48519bdef0");
    }

    #[test]
    fn uuid5_is_deterministic() {
        let a = uuid5_hex(&NAMESPACE_URL_BYTES, "camofox-user:/some/path");
        let b = uuid5_hex(&NAMESPACE_URL_BYTES, "camofox-user:/some/path");
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn state_dir_ends_with_expected_subdirs() {
        let dir = get_camofox_state_dir();
        let mut comps = dir.components().rev();
        assert_eq!(
            comps.next().unwrap().as_os_str().to_string_lossy(),
            CAMOFOX_STATE_SUBDIR
        );
        assert_eq!(
            comps.next().unwrap().as_os_str().to_string_lossy(),
            CAMOFOX_STATE_DIR_NAME
        );
    }

    #[test]
    fn identity_has_expected_prefixes_and_lengths() {
        let id = get_camofox_identity(None);
        let user_id = id.get("user_id").expect("user_id present");
        let session_key = id.get("session_key").expect("session_key present");

        assert!(user_id.starts_with("hermes_"));
        // "hermes_" (7) + 10 hex chars
        assert_eq!(user_id.len(), 7 + 10);

        assert!(session_key.starts_with("task_"));
        // "task_" (5) + 16 hex chars
        assert_eq!(session_key.len(), 5 + 16);
    }

    #[test]
    fn user_id_is_profile_scoped_session_key_is_task_scoped() {
        let default_id = get_camofox_identity(None);
        let explicit_default = get_camofox_identity(Some("default"));
        // None and Some("default") are equivalent.
        assert_eq!(default_id, explicit_default);

        let other = get_camofox_identity(Some("task-abc"));
        // Same profile => same user_id regardless of task.
        assert_eq!(default_id.get("user_id"), other.get("user_id"));
        // Different task => different session_key.
        assert_ne!(default_id.get("session_key"), other.get("session_key"));
    }
}
