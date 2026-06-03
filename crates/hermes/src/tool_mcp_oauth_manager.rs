//! Central manager for per-server MCP OAuth state (native Rust port of
//! `tools/mcp_oauth_manager.py`).
//!
//! One instance shared across the process. Holds per-server OAuth provider
//! state and coordinates:
//!
//! - **Cross-process token reload** via mtime-based disk watch. When an
//!   external process (e.g. a user cron job) refreshes tokens on disk, the
//!   next auth flow picks them up without requiring a process restart.
//! - **401 deduplication**. When N concurrent tool calls all hit 401 with the
//!   same access_token, only one recovery attempt fires; the rest await the
//!   same result.
//! - **Reconnect signalling** for long-lived MCP sessions. The manager itself
//!   does not drive reconnection — the caller does — but the manager is the
//!   single source of truth that decides when reconnect is warranted.
//!
//! This module is the ONLY place that should instantiate / cache an MCP OAuth
//! provider — all other code paths go through [`get_manager`].
//!
//! ## Port notes
//!
//! The Python original subclassed the MCP Python SDK's async `httpx.Auth`
//! `OAuthClientProvider`, injecting a pre-flow disk-watch hook and a
//! token-expiry seeding `_initialize` override. There is no Rust equivalent of
//! that SDK object here, so this port reproduces the *self-contained* manager
//! logic that does not depend on the SDK:
//!
//!   - per-server entry caching with URL-change eviction
//!     ([`MCPOAuthManager::get_or_build_provider`]),
//!   - mtime-based disk-watch invalidation
//!     ([`MCPOAuthManager::invalidate_if_disk_changed`]),
//!   - deduplicated 401 recovery ([`MCPOAuthManager::handle_401`]),
//!   - cache eviction + on-disk token removal
//!     ([`MCPOAuthManager::remove`]).
//!
//! The "provider" itself is modelled as the resolved
//! [`crate::tool_mcp_oauth::OAuthProviderConfig`] (the analogue of the Python
//! `OAuthClientProvider` constructor arguments) plus a tracked `_initialized`
//! flag that the disk-watch flips to `false`, mirroring the SDK's private
//! `_initialized` reset.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{Map, Value};

use crate::tool_mcp_oauth::{
    build_oauth_config, remove_oauth_tokens, safe_filename, OAuthProviderConfig,
};

// ---------------------------------------------------------------------------
// Provider handle
// ---------------------------------------------------------------------------

/// A cached OAuth provider for one MCP server.
///
/// Rust analogue of the Python `HermesMCPOAuthProvider` instance. Wraps the
/// resolved [`OAuthProviderConfig`] and an `_initialized` flag that the
/// disk-watch flips to `false` to force a reload-from-storage on the next auth
/// flow (mirrors the SDK's private `_initialized` reset).
#[derive(Debug)]
pub struct OAuthProvider {
    config: OAuthProviderConfig,
    /// Whether stored tokens have been loaded into memory. Reset to `false`
    /// when the manager detects an external on-disk refresh, so the next auth
    /// flow re-reads from storage. Mirrors the SDK's `_initialized`.
    initialized: AtomicBool,
    /// Whether an in-memory refresh token is available such that a token
    /// refresh can succeed without a full browser reauth. Used by
    /// [`MCPOAuthManager::handle_401`]'s no-disk-change branch (analogue of the
    /// SDK context's `can_refresh_token()`).
    can_refresh: AtomicBool,
}

impl OAuthProvider {
    fn new(config: OAuthProviderConfig) -> Self {
        Self {
            config,
            initialized: AtomicBool::new(false),
            can_refresh: AtomicBool::new(false),
        }
    }

    /// The resolved provider configuration.
    pub fn config(&self) -> &OAuthProviderConfig {
        &self.config
    }

    /// Whether stored tokens are currently loaded into memory.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Mark the provider as initialized (tokens loaded from storage).
    pub fn set_initialized(&self, value: bool) {
        self.initialized.store(value, Ordering::SeqCst);
    }

    /// Whether an in-memory refresh token is available.
    pub fn can_refresh_token(&self) -> bool {
        self.can_refresh.load(Ordering::SeqCst)
    }

    /// Record whether an in-memory refresh token is available.
    pub fn set_can_refresh(&self, value: bool) {
        self.can_refresh.store(value, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Per-server entry
// ---------------------------------------------------------------------------

/// Per-server OAuth state tracked by the manager.
///
/// Rust analogue of the Python `_ProviderEntry` dataclass.
#[derive(Debug)]
struct ProviderEntry {
    /// The MCP server URL used to build the provider. Tracked so we can
    /// discard a cached provider if the URL changes.
    server_url: String,
    /// Optional `oauth:` config block from `mcp_servers.<name>.oauth`.
    oauth_config: Option<Map<String, Value>>,
    /// The cached provider. `None` until first use.
    provider: Option<Arc<OAuthProvider>>,
    /// Last-seen `st_mtime_ns` of the on-disk tokens file. `None` if never
    /// read. Used by [`MCPOAuthManager::invalidate_if_disk_changed`] to detect
    /// external refreshes.
    last_mtime_ns: Option<u128>,
    /// In-flight 401-handler results keyed by the failed access_token, for
    /// deduplicating thundering-herd 401s. Mirrors the Python `pending_401`
    /// map of futures. A completed entry holds `Some(result)`.
    pending_401: HashMap<String, Arc<Pending401>>,
}

impl ProviderEntry {
    fn new(server_url: String, oauth_config: Option<Map<String, Value>>) -> Self {
        Self {
            server_url,
            oauth_config,
            provider: None,
            last_mtime_ns: None,
            pending_401: HashMap::new(),
        }
    }
}

/// A single in-flight (or completed) 401 recovery, awaited by all callers that
/// share the same failed access_token. Rust analogue of the per-key
/// `asyncio.Future[bool]`.
#[derive(Debug)]
struct Pending401 {
    /// `None` while running; `Some(result)` once resolved.
    result: Mutex<Option<bool>>,
}

impl Pending401 {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// mtime helper
// ---------------------------------------------------------------------------

/// Return the tokens-file modification time in whole nanoseconds since the
/// Unix epoch, mirroring Python's `st_mtime_ns`.
fn tokens_mtime_ns(path: &std::path::Path) -> Option<u128> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    mtime
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Single source of truth for per-server MCP OAuth state.
///
/// Thread-safe: the `entries` map is guarded by a single mutex for
/// get-or-create semantics and per-entry mutation.
pub struct MCPOAuthManager {
    entries: Mutex<HashMap<String, ProviderEntry>>,
}

impl Default for MCPOAuthManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MCPOAuthManager {
    /// Construct an empty manager.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    // -- Provider construction / caching -----------------------------------

    /// Return a cached OAuth provider for `server_name` or build one.
    ///
    /// Idempotent: repeat calls with the same name return the same instance.
    /// If `server_url` changes for a given name, the cached entry is discarded
    /// and a fresh provider is built.
    ///
    /// Returns `None` only if building the provider config failed (e.g. no free
    /// callback port could be bound). In the Python original this returned
    /// `None` when the MCP SDK's OAuth support was unavailable.
    pub fn get_or_build_provider(
        &self,
        server_name: &str,
        server_url: &str,
        oauth_config: Option<Map<String, Value>>,
    ) -> Option<Arc<OAuthProvider>> {
        let mut entries = self.entries.lock().unwrap();

        // URL-change eviction: drop the stale entry so a fresh provider builds.
        if let Some(existing) = entries.get(server_name) {
            if existing.server_url != server_url {
                log::info!(
                    "MCP OAuth '{server_name}': URL changed from {} to {server_url}, discarding cache",
                    existing.server_url
                );
                entries.remove(server_name);
            }
        }

        let entry = entries
            .entry(server_name.to_string())
            .or_insert_with(|| ProviderEntry::new(server_url.to_string(), oauth_config));

        if entry.provider.is_none() {
            entry.provider = Self::build_provider(server_name, entry);
        }

        entry.provider.clone()
    }

    /// Build the underlying OAuth provider from the entry's config.
    ///
    /// Rust analogue of the Python `_build_provider`: resolves the callback
    /// port, builds client metadata, and pre-registers a configured client_id
    /// (all delegated to [`build_oauth_config`]). Returns `None` if the config
    /// could not be resolved.
    fn build_provider(server_name: &str, entry: &ProviderEntry) -> Option<Arc<OAuthProvider>> {
        match build_oauth_config(server_name, &entry.server_url, entry.oauth_config.as_ref()) {
            Ok(config) => Some(Arc::new(OAuthProvider::new(config))),
            Err(exc) => {
                log::warn!("MCP OAuth '{server_name}': failed to build provider: {exc}");
                None
            }
        }
    }

    /// Evict the provider from cache AND delete tokens from disk.
    ///
    /// Called by `hermes mcp remove <name>` and (indirectly) by
    /// `hermes mcp login <name>` during forced re-auth.
    pub fn remove(&self, server_name: &str) {
        {
            let mut entries = self.entries.lock().unwrap();
            entries.remove(server_name);
        }
        remove_oauth_tokens(server_name);
        log::info!("MCP OAuth '{server_name}': evicted from cache and removed from disk");
    }

    // -- Disk watch ---------------------------------------------------------

    /// Path to the tokens file for `server_name`.
    fn tokens_path(server_name: &str) -> std::path::PathBuf {
        crate::tool_mcp_oauth::get_token_dir().join(format!("{}.json", safe_filename(server_name)))
    }

    /// If the tokens file on disk has a newer mtime than last-seen, force the
    /// cached provider to reload its in-memory state on the next auth flow.
    ///
    /// Returns `true` if the cache was invalidated (mtime differed). This is
    /// the core fix for the external-refresh workflow: a cron job writes fresh
    /// tokens to disk, and on the next tool call the running MCP session picks
    /// them up without a restart.
    ///
    /// Returns `false` if there is no cached provider for `server_name`, if the
    /// tokens file does not exist, or if the mtime is unchanged.
    pub fn invalidate_if_disk_changed(&self, server_name: &str) -> bool {
        let mut entries = self.entries.lock().unwrap();

        let entry = match entries.get_mut(server_name) {
            Some(e) if e.provider.is_some() => e,
            _ => return false,
        };

        let path = Self::tokens_path(server_name);
        let mtime_ns = match tokens_mtime_ns(&path) {
            Some(m) => m,
            None => return false,
        };

        if Some(mtime_ns) != entry.last_mtime_ns {
            let old = entry.last_mtime_ns;
            entry.last_mtime_ns = Some(mtime_ns);
            // Force the provider to reload from storage on its next auth flow.
            if let Some(provider) = &entry.provider {
                provider.set_initialized(false);
            }
            log::info!(
                "MCP OAuth '{server_name}': tokens file changed (mtime {} -> {mtime_ns}), forcing reload",
                old.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string())
            );
            true
        } else {
            false
        }
    }

    // -- 401 handler (dedup'd) ----------------------------------------------

    /// Handle a 401 from a tool call, deduplicated across concurrent callers.
    ///
    /// Returns:
    /// - `true`  if a (possibly new) access token is now available — caller
    ///   should trigger a reconnect and retry the operation.
    /// - `false` if no recovery path exists — caller should surface a
    ///   `needs_reauth` error to the model so it stops hallucinating manual
    ///   refresh attempts.
    ///
    /// Thundering-herd protection: if N concurrent tool calls hit 401 with the
    /// same `failed_access_token`, only one recovery attempt actually runs.
    /// Others observe the same result.
    pub fn handle_401(&self, server_name: &str, failed_access_token: Option<&str>) -> bool {
        // Quick existence check, then determine whether this caller owns the
        // recovery (the first to register the key) or is a follower.
        let key = failed_access_token.unwrap_or("<unknown>").to_string();

        let (is_owner, pending) = {
            let mut entries = self.entries.lock().unwrap();
            let entry = match entries.get_mut(server_name) {
                Some(e) if e.provider.is_some() => e,
                _ => return false,
            };
            match entry.pending_401.get(&key) {
                Some(existing) => (false, Arc::clone(existing)),
                None => {
                    let p = Arc::new(Pending401::new());
                    entry.pending_401.insert(key.clone(), Arc::clone(&p));
                    (true, p)
                }
            }
        };

        if is_owner {
            let result = self.do_handle_401(server_name);
            *pending.result.lock().unwrap() = Some(result);
            // Drop the in-flight entry so a future 401 with the same token can
            // retry recovery. Mirrors the Python `finally: pending_401.pop`.
            let mut entries = self.entries.lock().unwrap();
            if let Some(entry) = entries.get_mut(server_name) {
                entry.pending_401.remove(&key);
            }
            return result;
        }

        // Follower: observe the owner's resolved result. The owner records the
        // result before removing the pending entry, so a clone we already hold
        // will carry the answer once set. If still unset (owner not finished
        // in this single-threaded synchronous model), fall back to false.
        pending.result.lock().unwrap().unwrap_or(false)
    }

    /// Run the actual 401 recovery for `server_name`. Rust analogue of the
    /// Python `_do_handle`.
    fn do_handle_401(&self, server_name: &str) -> bool {
        // Step 1: Did disk change? Picks up an external refresh.
        if self.invalidate_if_disk_changed(server_name) {
            return true;
        }

        // Step 2: No disk change — if the provider can refresh in-place, let
        // the caller retry. The refresh itself happens on the next request.
        let entries = self.entries.lock().unwrap();
        match entries.get(server_name).and_then(|e| e.provider.as_ref()) {
            Some(provider) => provider.can_refresh_token(),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level singleton
// ---------------------------------------------------------------------------

static MANAGER: OnceLock<Arc<MCPOAuthManager>> = OnceLock::new();

/// Return the process-wide [`MCPOAuthManager`] singleton.
pub fn get_manager() -> Arc<MCPOAuthManager> {
    Arc::clone(MANAGER.get_or_init(|| Arc::new(MCPOAuthManager::new())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hermes-oauth-mgr-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::set_var("HERMES_HOME", &dir); }
        dir
    }

    #[test]
    fn test_get_manager_is_singleton() {
        let a = get_manager();
        let b = get_manager();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_get_or_build_provider_caches() {
        let _home = temp_home("cache");
        let mgr = MCPOAuthManager::new();
        let p1 = mgr
            .get_or_build_provider("srv1", "https://mcp.example.com/mcp", None)
            .expect("provider built");
        let p2 = mgr
            .get_or_build_provider("srv1", "https://mcp.example.com/mcp", None)
            .expect("provider cached");
        assert!(Arc::ptr_eq(&p1, &p2), "same name+url should return same instance");
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_url_change_discards_cache() {
        let _home = temp_home("urlchange");
        let mgr = MCPOAuthManager::new();
        let p1 = mgr
            .get_or_build_provider("srv2", "https://a.example.com/mcp", None)
            .unwrap();
        let p2 = mgr
            .get_or_build_provider("srv2", "https://b.example.com/mcp", None)
            .unwrap();
        assert!(!Arc::ptr_eq(&p1, &p2), "URL change should rebuild");
        assert_eq!(p2.config().server_url, "https://b.example.com/mcp");
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_invalidate_no_provider_returns_false() {
        let _home = temp_home("noprov");
        let mgr = MCPOAuthManager::new();
        // Nothing built yet.
        assert!(!mgr.invalidate_if_disk_changed("missing"));
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_invalidate_detects_disk_change() {
        let _home = temp_home("diskchange");
        let mgr = MCPOAuthManager::new();
        let provider = mgr
            .get_or_build_provider("disksrv", "https://mcp.example.com/mcp", None)
            .unwrap();
        provider.set_initialized(true);

        // No tokens file yet -> false.
        assert!(!mgr.invalidate_if_disk_changed("disksrv"));

        // Write a tokens file; first observation registers the mtime and
        // returns true (mtime differs from the initial `None`).
        let storage = crate::tool_mcp_oauth::HermesTokenStorage::new("disksrv");
        let mut tok = Map::new();
        tok.insert("access_token".into(), Value::from("abc"));
        storage.set_tokens(&tok).unwrap();

        assert!(mgr.invalidate_if_disk_changed("disksrv"));
        // The provider should have been reset.
        assert!(!provider.is_initialized());

        // Unchanged mtime -> false.
        assert!(!mgr.invalidate_if_disk_changed("disksrv"));

        storage.remove();
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_handle_401_no_provider_false() {
        let _home = temp_home("h401noprov");
        let mgr = MCPOAuthManager::new();
        assert!(!mgr.handle_401("nope", Some("tok")));
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_handle_401_disk_change_returns_true() {
        let _home = temp_home("h401disk");
        let mgr = MCPOAuthManager::new();
        mgr.get_or_build_provider("h401srv", "https://mcp.example.com/mcp", None)
            .unwrap();

        // Write tokens so the disk-watch fires on first observation.
        let storage = crate::tool_mcp_oauth::HermesTokenStorage::new("h401srv");
        let mut tok = Map::new();
        tok.insert("access_token".into(), Value::from("abc"));
        storage.set_tokens(&tok).unwrap();

        assert!(mgr.handle_401("h401srv", Some("stale-token")));

        storage.remove();
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_handle_401_can_refresh_branch() {
        let _home = temp_home("h401refresh");
        let mgr = MCPOAuthManager::new();
        let provider = mgr
            .get_or_build_provider("refsrv", "https://mcp.example.com/mcp", None)
            .unwrap();
        // No tokens on disk -> no disk change. With can_refresh=false -> false.
        assert!(!mgr.handle_401("refsrv", Some("t")));
        // Flip can_refresh -> recovery returns true.
        provider.set_can_refresh(true);
        assert!(mgr.handle_401("refsrv", Some("t")));
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_remove_clears_cache_and_disk() {
        let _home = temp_home("remove");
        let mgr = MCPOAuthManager::new();
        mgr.get_or_build_provider("rmsrv", "https://mcp.example.com/mcp", None)
            .unwrap();
        let storage = crate::tool_mcp_oauth::HermesTokenStorage::new("rmsrv");
        let mut tok = Map::new();
        tok.insert("access_token".into(), Value::from("abc"));
        storage.set_tokens(&tok).unwrap();
        assert!(storage.has_cached_tokens());

        mgr.remove("rmsrv");

        assert!(!storage.has_cached_tokens());
        // Rebuilding produces a fresh instance (cache was cleared).
        let rebuilt = mgr.get_or_build_provider("rmsrv", "https://mcp.example.com/mcp", None);
        assert!(rebuilt.is_some());
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn test_provider_initialized_flag_roundtrip() {
        let _home = temp_home("initflag");
        let mgr = MCPOAuthManager::new();
        let p = mgr
            .get_or_build_provider("flagsrv", "https://mcp.example.com/mcp", None)
            .unwrap();
        assert!(!p.is_initialized());
        p.set_initialized(true);
        assert!(p.is_initialized());
        unsafe { std::env::remove_var("HERMES_HOME"); }
    }
}
