//! DM Pairing System
//!
//! Code-based approval flow for authorizing new users on messaging platforms.
//! Instead of static allowlists with user IDs, unknown users receive a one-time
//! pairing code that the bot owner approves via the CLI.
//!
//! Security features (based on OWASP + NIST SP 800-63-4 guidance):
//!   - 8-char codes from 32-char unambiguous alphabet (no 0/O/1/I)
//!   - Cryptographic randomness via getrandom
//!   - 1-hour code expiry
//!   - Max 3 pending codes per platform
//!   - Rate limiting: 1 request per user per 10 minutes
//!   - Lockout after 5 failed approval attempts (1 hour)
//!   - File permissions: chmod 0600 on all data files
//!   - Codes are never logged to stdout
//!
//! Storage: ~/.hermes/platforms/pairing/ (or legacy ~/.hermes/pairing/)
//!
//! Faithful Rust port of `gateway/pairing.py`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::mod_hermes_constants::get_hermes_dir;
use crate::mod_utils::atomic_replace;

/// Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion.
pub const ALPHABET: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
pub const CODE_LENGTH: usize = 8;

/// Codes expire after 1 hour.
pub const CODE_TTL_SECONDS: f64 = 3600.0;
/// 1 request per user per 10 minutes.
pub const RATE_LIMIT_SECONDS: f64 = 600.0;
/// Lockout duration after too many failures.
pub const LOCKOUT_SECONDS: f64 = 3600.0;

/// Max pending codes per platform.
pub const MAX_PENDING_PER_PLATFORM: usize = 3;
/// Failed approvals before lockout.
pub const MAX_FAILED_ATTEMPTS: i64 = 5;

/// Resolve the pairing directory, mirroring
/// `get_hermes_dir("platforms/pairing", "pairing")`.
pub fn pairing_dir() -> PathBuf {
    get_hermes_dir("platforms/pairing", "pairing")
}

/// Wall-clock seconds since the Unix epoch, matching Python's `time.time()`.
fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Fill `buf` with cryptographically secure random bytes, falling back to a
/// time-seeded LCG only if `getrandom` is unavailable at runtime.
fn getrandom_bytes(buf: &mut [u8]) {
    if getrandom::fill(buf).is_err() {
        let mut state = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
            ^ (std::process::id() as u64);
        for b in buf.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (state >> 33) as u8;
        }
    }
}

/// Cryptographically uniform choice index into a slice of length `n` using
/// rejection sampling to avoid modulo bias (mirrors `secrets.choice`).
fn secure_index(n: usize) -> usize {
    debug_assert!(n > 0);
    let n = n as u64;
    // Largest multiple of n that fits in u64, used as the rejection threshold.
    let limit = u64::MAX - (u64::MAX % n);
    loop {
        let mut bytes = [0u8; 8];
        getrandom_bytes(&mut bytes);
        let val = u64::from_le_bytes(bytes);
        if val < limit {
            return (val % n) as usize;
        }
    }
}

/// Generate a single random code from [`ALPHABET`].
fn generate_random_code() -> String {
    let chars: Vec<char> = ALPHABET.chars().collect();
    let mut out = String::with_capacity(CODE_LENGTH);
    for _ in 0..CODE_LENGTH {
        out.push(chars[secure_index(chars.len())]);
    }
    out
}

/// Write `data` to `path` with restrictive permissions (owner read/write only).
///
/// Uses a temp-file + atomic rename so readers always see either the old
/// complete file or the new one — never a partial write. Mirrors `_secure_write`.
fn secure_write(path: &Path, data: &str) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    // Create a uniquely-named temp file in the same directory (tempfile.mkstemp).
    let tmp_path = make_temp(parent, ".tmp")?;

    let result = (|| -> std::io::Result<()> {
        {
            let mut f = fs::OpenOptions::new().write(true).open(&tmp_path)?;
            f.write_all(data.as_bytes())?;
            f.flush()?;
            f.sync_all()?; // os.fsync
        }
        let real_path = atomic_replace(&tmp_path, path)?;
        // os.chmod(path, 0o600); Windows may not support it, so ignore errors.
        set_mode_600(&real_path);
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

#[cfg(unix)]
fn set_mode_600(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_mode_600(_path: &Path) {}

/// Create a uniquely-named temp file in `dir` with the given `suffix`, returning
/// its path. Mirrors `tempfile.mkstemp(dir=..., suffix=...)`.
fn make_temp(dir: &Path, suffix: &str) -> std::io::Result<PathBuf> {
    let pid = std::process::id();
    for attempt in 0..10_000u64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let candidate = dir.join(format!("tmp{pid}_{nanos}_{attempt}{suffix}"));
        let mut opts = fs::OpenOptions::new();
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

/// Result of approving a pairing code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedUser {
    pub user_id: String,
    pub user_name: String,
}

/// A pending pairing request as surfaced by [`PairingStore::list_pending`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingEntry {
    pub platform: String,
    pub code: String,
    pub user_id: String,
    pub user_name: String,
    pub age_minutes: i64,
}

/// An approved-user record as surfaced by [`PairingStore::list_approved`].
#[derive(Debug, Clone)]
pub struct ApprovedEntry {
    pub platform: String,
    pub user_id: String,
    pub user_name: String,
    pub approved_at: f64,
    /// Any additional fields stored alongside the known ones.
    pub extra: Map<String, Value>,
}

/// Manages pairing codes and approved user lists.
///
/// Data files per platform:
///   - `{platform}-pending.json`   : pending pairing requests
///   - `{platform}-approved.json`  : approved (paired) users
///   - `_rate_limits.json`         : rate limit tracking
pub struct PairingStore {
    dir: PathBuf,
    /// Protects all read-modify-write cycles. The gateway runs multiple platform
    /// adapters concurrently sharing one PairingStore. A reentrant lock is not
    /// needed because no public method recurses into another locked method.
    lock: Mutex<()>,
}

impl Default for PairingStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PairingStore {
    /// Create a store rooted at the default pairing directory.
    pub fn new() -> Self {
        Self::with_dir(pairing_dir())
    }

    /// Create a store rooted at an explicit directory (used for tests/injection).
    pub fn with_dir(dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&dir);
        Self {
            dir,
            lock: Mutex::new(()),
        }
    }

    fn pending_path(&self, platform: &str) -> PathBuf {
        self.dir.join(format!("{platform}-pending.json"))
    }

    fn approved_path(&self, platform: &str) -> PathBuf {
        self.dir.join(format!("{platform}-approved.json"))
    }

    fn rate_limit_path(&self) -> PathBuf {
        self.dir.join("_rate_limits.json")
    }

    /// Load a JSON object from `path`, returning an empty map on any error or
    /// non-object content (mirrors `_load_json` which always returns a dict).
    fn load_json(&self, path: &Path) -> Map<String, Value> {
        if path.exists() {
            if let Ok(text) = fs::read_to_string(path) {
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) {
                    return map;
                }
            }
        }
        Map::new()
    }

    /// Serialize `data` and write it securely. Mirrors `_save_json` which uses
    /// `json.dumps(data, indent=2, ensure_ascii=False)`.
    fn save_json(&self, path: &Path, data: &Map<String, Value>) {
        let text = serialize_pretty(data);
        let _ = secure_write(path, &text);
    }

    // ----- Approved users -----

    /// Check if a user is approved (paired) on a platform.
    pub fn is_approved(&self, platform: &str, user_id: &str) -> bool {
        let approved = self.load_json(&self.approved_path(platform));
        approved.contains_key(user_id)
    }

    /// List approved users, optionally filtered by platform.
    pub fn list_approved(&self, platform: Option<&str>) -> Vec<ApprovedEntry> {
        let mut results = Vec::new();
        let platforms = match platform {
            Some(p) => vec![p.to_string()],
            None => self.all_platforms("approved"),
        };
        for p in platforms {
            let approved = self.load_json(&self.approved_path(&p));
            for (uid, info) in approved.iter() {
                let obj = info.as_object().cloned().unwrap_or_default();
                let user_name = obj
                    .get("user_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let approved_at = obj
                    .get("approved_at")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let mut extra = obj.clone();
                extra.remove("user_name");
                extra.remove("approved_at");
                results.push(ApprovedEntry {
                    platform: p.clone(),
                    user_id: uid.clone(),
                    user_name,
                    approved_at,
                    extra,
                });
            }
        }
        results
    }

    /// Add a user to the approved list. Must be called under the lock.
    fn approve_user(&self, platform: &str, user_id: &str, user_name: &str) {
        let path = self.approved_path(platform);
        let mut approved = self.load_json(&path);
        let mut entry = Map::new();
        entry.insert("user_name".into(), Value::String(user_name.to_string()));
        entry.insert(
            "approved_at".into(),
            Value::from(now_secs()),
        );
        approved.insert(user_id.to_string(), Value::Object(entry));
        self.save_json(&path, &approved);
    }

    /// Remove a user from the approved list. Returns true if found.
    pub fn revoke(&self, platform: &str, user_id: &str) -> bool {
        let path = self.approved_path(platform);
        let _guard = self.lock.lock().unwrap();
        let mut approved = self.load_json(&path);
        if approved.remove(user_id).is_some() {
            self.save_json(&path, &approved);
            return true;
        }
        false
    }

    // ----- Pending codes -----

    /// Generate a pairing code for a new user.
    ///
    /// Returns the code string, or `None` if:
    ///   - User is rate-limited (too recent request)
    ///   - Max pending codes reached for this platform
    ///   - User/platform is in lockout due to failed attempts
    pub fn generate_code(
        &self,
        platform: &str,
        user_id: &str,
        user_name: &str,
    ) -> Option<String> {
        let _guard = self.lock.lock().unwrap();
        self.cleanup_expired_locked(platform);

        // Check lockout.
        if self.is_locked_out_locked(platform) {
            return None;
        }

        // Check rate limit for this specific user.
        if self.is_rate_limited_locked(platform, user_id) {
            return None;
        }

        // Check max pending.
        let pending_path = self.pending_path(platform);
        let mut pending = self.load_json(&pending_path);
        if pending.len() >= MAX_PENDING_PER_PLATFORM {
            return None;
        }

        // Generate cryptographically random code.
        let code = generate_random_code();

        // Store pending request.
        let mut entry = Map::new();
        entry.insert("user_id".into(), Value::String(user_id.to_string()));
        entry.insert("user_name".into(), Value::String(user_name.to_string()));
        entry.insert("created_at".into(), Value::from(now_secs()));
        pending.insert(code.clone(), Value::Object(entry));
        self.save_json(&pending_path, &pending);

        // Record rate limit.
        self.record_rate_limit_locked(platform, user_id);

        Some(code)
    }

    /// Approve a pairing code. Adds the user to the approved list.
    ///
    /// Returns [`ApprovedUser`] on success, `None` if code is invalid/expired.
    pub fn approve_code(&self, platform: &str, code: &str) -> Option<ApprovedUser> {
        let _guard = self.lock.lock().unwrap();
        self.cleanup_expired_locked(platform);
        let code = code.to_uppercase();
        let code = code.trim();

        let pending_path = self.pending_path(platform);
        let mut pending = self.load_json(&pending_path);
        if !pending.contains_key(code) {
            self.record_failed_attempt_locked(platform);
            return None;
        }

        let entry = pending.remove(code).unwrap();
        self.save_json(&pending_path, &pending);

        let obj = entry.as_object().cloned().unwrap_or_default();
        let user_id = obj
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let user_name = obj
            .get("user_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Add to approved list.
        self.approve_user(platform, &user_id, &user_name);

        Some(ApprovedUser {
            user_id,
            user_name,
        })
    }

    /// List pending pairing requests, optionally filtered by platform.
    pub fn list_pending(&self, platform: Option<&str>) -> Vec<PendingEntry> {
        let mut results = Vec::new();
        let platforms = match platform {
            Some(p) => vec![p.to_string()],
            None => self.all_platforms("pending"),
        };
        for p in platforms {
            // cleanup_expired performs its own load/save; not under the lock,
            // matching the Python which calls it without holding self._lock here.
            self.cleanup_expired(&p);
            let pending = self.load_json(&self.pending_path(&p));
            for (code, info) in pending.iter() {
                let obj = info.as_object().cloned().unwrap_or_default();
                let created_at = obj
                    .get("created_at")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let age_min = ((now_secs() - created_at) / 60.0) as i64;
                let user_id = obj
                    .get("user_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let user_name = obj
                    .get("user_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                results.push(PendingEntry {
                    platform: p.clone(),
                    code: code.clone(),
                    user_id,
                    user_name,
                    age_minutes: age_min,
                });
            }
        }
        results
    }

    /// Clear all pending requests. Returns count removed.
    pub fn clear_pending(&self, platform: Option<&str>) -> usize {
        let _guard = self.lock.lock().unwrap();
        let mut count = 0;
        let platforms = match platform {
            Some(p) => vec![p.to_string()],
            None => self.all_platforms("pending"),
        };
        for p in platforms {
            let path = self.pending_path(&p);
            let pending = self.load_json(&path);
            count += pending.len();
            self.save_json(&path, &Map::new());
        }
        count
    }

    // ----- Rate limiting and lockout -----

    /// Check if a user has requested a code too recently.
    fn is_rate_limited_locked(&self, platform: &str, user_id: &str) -> bool {
        let limits = self.load_json(&self.rate_limit_path());
        let key = format!("{platform}:{user_id}");
        let last_request = limits.get(&key).and_then(|v| v.as_f64()).unwrap_or(0.0);
        (now_secs() - last_request) < RATE_LIMIT_SECONDS
    }

    /// Record the time of a pairing request for rate limiting.
    fn record_rate_limit_locked(&self, platform: &str, user_id: &str) {
        let path = self.rate_limit_path();
        let mut limits = self.load_json(&path);
        let key = format!("{platform}:{user_id}");
        limits.insert(key, Value::from(now_secs()));
        self.save_json(&path, &limits);
    }

    /// Check if a platform is in lockout due to failed approval attempts.
    fn is_locked_out_locked(&self, platform: &str) -> bool {
        let limits = self.load_json(&self.rate_limit_path());
        let lockout_key = format!("_lockout:{platform}");
        let lockout_until = limits
            .get(&lockout_key)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        now_secs() < lockout_until
    }

    /// Record a failed approval attempt. Triggers lockout after
    /// `MAX_FAILED_ATTEMPTS`.
    fn record_failed_attempt_locked(&self, platform: &str) {
        let path = self.rate_limit_path();
        let mut limits = self.load_json(&path);
        let fail_key = format!("_failures:{platform}");
        let fails = limits.get(&fail_key).and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        limits.insert(fail_key.clone(), Value::from(fails));
        if fails >= MAX_FAILED_ATTEMPTS {
            let lockout_key = format!("_lockout:{platform}");
            limits.insert(lockout_key, Value::from(now_secs() + LOCKOUT_SECONDS));
            limits.insert(fail_key, Value::from(0i64)); // Reset counter.
            // Codes are never logged; this message intentionally omits any code.
            log::warn!(
                "[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s \
                 after {MAX_FAILED_ATTEMPTS} failed attempts"
            );
        }
        self.save_json(&path, &limits);
    }

    // ----- Cleanup -----

    /// Remove expired pending codes (acquires the lock).
    fn cleanup_expired(&self, platform: &str) {
        let _guard = self.lock.lock().unwrap();
        self.cleanup_expired_locked(platform);
    }

    /// Remove expired pending codes. Caller must hold the lock.
    fn cleanup_expired_locked(&self, platform: &str) {
        let path = self.pending_path(platform);
        let mut pending = self.load_json(&path);
        let now = now_secs();
        let expired: Vec<String> = pending
            .iter()
            .filter(|(_, info)| {
                let created_at = info
                    .as_object()
                    .and_then(|o| o.get("created_at"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                (now - created_at) > CODE_TTL_SECONDS
            })
            .map(|(code, _)| code.clone())
            .collect();
        if !expired.is_empty() {
            for code in &expired {
                pending.remove(code);
            }
            self.save_json(&path, &pending);
        }
    }

    /// List all platforms that have data files of a given suffix.
    fn all_platforms(&self, suffix: &str) -> Vec<String> {
        let mut platforms = Vec::new();
        let needle = format!("-{suffix}.json");
        let entries = match fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return platforms,
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(&needle) {
                let platform = name.trim_end_matches(&needle as &str).to_string();
                if !platform.starts_with('_') {
                    platforms.push(platform);
                }
            }
        }
        platforms
    }
}

/// Serialize a JSON object with 2-space indentation and `ensure_ascii=False`
/// semantics. `serde_json::to_string_pretty` uses 2-space indent and emits raw
/// UTF-8, matching `json.dumps(data, indent=2, ensure_ascii=False)`.
///
/// To keep deterministic key ordering identical to Python's insertion order is
/// not guaranteed by `serde_json::Map` unless the `preserve_order` feature is
/// enabled; we sort keys for stable output. The on-disk data is only ever read
/// back by this module, so ordering is not behaviourally significant.
fn serialize_pretty(data: &Map<String, Value>) -> String {
    let sorted: BTreeMap<&String, &Value> = data.iter().collect();
    serde_json::to_string_pretty(&sorted).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_store() -> PairingStore {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("hermes_pairing_test_{nanos}_{n}"));
        PairingStore::with_dir(dir)
    }

    #[test]
    fn code_format_and_alphabet() {
        let code = generate_random_code();
        assert_eq!(code.len(), CODE_LENGTH);
        for c in code.chars() {
            assert!(ALPHABET.contains(c), "char {c} not in alphabet");
        }
        // Forbidden ambiguous characters.
        for c in ['0', 'O', '1', 'I'] {
            assert!(!ALPHABET.contains(c));
        }
    }

    #[test]
    fn generate_then_approve_roundtrip() {
        let store = temp_store();
        let code = store
            .generate_code("telegram", "u123", "Alice")
            .expect("should produce a code");
        assert_eq!(code.len(), CODE_LENGTH);
        assert!(!store.is_approved("telegram", "u123"));

        // Lower-case + whitespace should still match (upper().strip()).
        let lowered = format!("  {}  ", code.to_lowercase());
        let res = store.approve_code("telegram", &lowered).expect("approved");
        assert_eq!(res.user_id, "u123");
        assert_eq!(res.user_name, "Alice");
        assert!(store.is_approved("telegram", "u123"));

        // Code consumed; second approve is a failed attempt.
        assert!(store.approve_code("telegram", &code).is_none());
    }

    #[test]
    fn rate_limit_blocks_second_request() {
        let store = temp_store();
        assert!(store.generate_code("slack", "user", "").is_some());
        // Immediate second request for same user is rate-limited.
        assert!(store.generate_code("slack", "user", "").is_none());
        // Different user still allowed.
        assert!(store.generate_code("slack", "other", "").is_some());
    }

    #[test]
    fn max_pending_enforced() {
        let store = temp_store();
        for i in 0..MAX_PENDING_PER_PLATFORM {
            assert!(
                store
                    .generate_code("discord", &format!("u{i}"), "")
                    .is_some()
            );
        }
        // Pool full; a new distinct user is rejected.
        assert!(store.generate_code("discord", "extra", "").is_none());
        assert_eq!(store.list_pending(Some("discord")).len(), MAX_PENDING_PER_PLATFORM);
    }

    #[test]
    fn lockout_after_failed_attempts() {
        let store = temp_store();
        // First create a pending code so the platform dir exists; not required.
        for _ in 0..MAX_FAILED_ATTEMPTS {
            assert!(store.approve_code("matrix", "BADCODE9").is_none());
        }
        // Now locked out: generate_code returns None even for a fresh user.
        assert!(store.generate_code("matrix", "freshuser", "").is_none());
    }

    #[test]
    fn revoke_and_list_approved() {
        let store = temp_store();
        let code = store.generate_code("feishu", "uX", "Bob").unwrap();
        store.approve_code("feishu", &code).unwrap();

        let approved = store.list_approved(Some("feishu"));
        assert_eq!(approved.len(), 1);
        assert_eq!(approved[0].user_id, "uX");
        assert_eq!(approved[0].user_name, "Bob");

        assert!(store.revoke("feishu", "uX"));
        assert!(!store.revoke("feishu", "uX")); // already gone
        assert!(!store.is_approved("feishu", "uX"));
    }

    #[test]
    fn clear_pending_counts() {
        let store = temp_store();
        store.generate_code("wecom", "a", "").unwrap();
        store.generate_code("wecom", "b", "").unwrap();
        let removed = store.clear_pending(Some("wecom"));
        assert_eq!(removed, 2);
        assert_eq!(store.list_pending(Some("wecom")).len(), 0);
    }

    #[test]
    fn all_platforms_skips_underscore_files() {
        let store = temp_store();
        store.generate_code("signal", "z", "").unwrap();
        // _rate_limits.json exists but must not be treated as a platform.
        let platforms = store.all_platforms("pending");
        assert_eq!(platforms, vec!["signal".to_string()]);
    }

    #[test]
    fn secure_index_is_in_range() {
        for _ in 0..1000 {
            let idx = secure_index(32);
            assert!(idx < 32);
        }
    }
}
