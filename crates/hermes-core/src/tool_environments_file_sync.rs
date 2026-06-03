//! Shared file sync manager for remote execution backends.
//!
//! Tracks local file changes via mtime+size, detects deletions, and
//! syncs to remote environments transactionally. Used by SSH, Modal,
//! and Daytona. Docker and Singularity use bind mounts (live host FS
//! view) and don't need this.
//!
//! Native Rust port of `tools/environments/file_sync.py`.

use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// `(mtime_secs_as_f64, size_bytes)` cache key for change detection.
pub type FileKey = (f64, u64);

/// Default seconds between rate-limited sync cycles.
pub const SYNC_INTERVAL_SECONDS: f64 = 5.0;

/// Environment variable that, when set (truthy / non-empty), forces every
/// `sync()` call to run regardless of the rate-limit interval.
pub const FORCE_SYNC_ENV: &str = "HERMES_FORCE_FILE_SYNC";

const SYNC_BACK_MAX_RETRIES: usize = 3;
/// Seconds to wait between sync-back retries.
const SYNC_BACK_BACKOFF: [u64; 3] = [2, 4, 8];
/// 2 GiB — refuse to extract larger tars.
const SYNC_BACK_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Return `(mtime, size)` for cache comparison, or `None` if unreadable.
///
/// Mirrors `tools.environments.base._file_mtime_key`.
pub fn file_mtime_key(host_path: &str) -> Option<FileKey> {
    let meta = fs::metadata(host_path).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Some((mtime, meta.len()))
}

/// Quote a single argument for safe inclusion in a POSIX shell command.
///
/// Mirrors Python's `shlex.quote`.
fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // shlex.quote: safe chars are [a-zA-Z0-9_@%+=:,./-]
    let safe = s.bytes().all(|b| {
        matches!(b,
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'
            | b'_' | b'@' | b'%' | b'+' | b'=' | b':' | b',' | b'.' | b'/' | b'-')
    });
    if safe {
        return s.to_string();
    }
    // Wrap in single quotes; escape embedded single quotes.
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

/// Parent directory of a remote path rendered as a string, matching
/// Python's `str(Path(p).parent)`.
fn path_parent_str(p: &str) -> String {
    Path::new(p)
        .parent()
        .map(|pp| pp.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            // Python's PurePosixPath("foo").parent == "." ;  Path("/").parent is "/"
            if p.starts_with('/') {
                "/".to_string()
            } else {
                ".".to_string()
            }
        })
}

/// Build a shell `rm -f` command for a batch of remote paths.
pub fn quoted_rm_command(remote_paths: &[String]) -> String {
    let parts: Vec<String> = remote_paths.iter().map(|p| shlex_quote(p)).collect();
    format!("rm -f {}", parts.join(" "))
}

/// Build a shell `mkdir -p` command for a batch of directories.
pub fn quoted_mkdir_command(dirs: &[String]) -> String {
    let parts: Vec<String> = dirs.iter().map(|d| shlex_quote(d)).collect();
    format!("mkdir -p {}", parts.join(" "))
}

/// Extract sorted unique parent directories from `(host, remote)` pairs.
pub fn unique_parent_dirs(files: &[(String, String)]) -> Vec<String> {
    let set: BTreeSet<String> = files
        .iter()
        .map(|(_, remote)| path_parent_str(remote))
        .collect();
    set.into_iter().collect()
}

/// Return the hex SHA-256 digest of a file.
pub fn sha256_file(path: &str) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Enumerate all files that should be synced to a remote environment.
///
/// In Python this combines credentials, skills, and cache via a late
/// import of `tools.credential_files`. The credential-enumeration logic is
/// not yet ported to Rust, so this is parameterised: the caller supplies the
/// three enumerators. Each returns `(host_path, remote_path)` pairs.
///
/// Credential remote paths are remapped from the hardcoded `/root/.hermes`
/// prefix to `container_base` (only the first occurrence, matching Python's
/// `str.replace(old, new, 1)`).
pub fn iter_sync_files<C, S, K>(
    container_base: &str,
    credential_mounts: C,
    skills_files: S,
    cache_files: K,
) -> Vec<(String, String)>
where
    C: IntoIterator<Item = (String, String)>,
    S: IntoIterator<Item = (String, String)>,
    K: IntoIterator<Item = (String, String)>,
{
    let mut files: Vec<(String, String)> = Vec::new();
    for (host_path, container_path) in credential_mounts {
        let remote = replace_first(&container_path, "/root/.hermes", container_base);
        files.push((host_path, remote));
    }
    for (host_path, container_path) in skills_files {
        files.push((host_path, container_path));
    }
    for (host_path, container_path) in cache_files {
        files.push((host_path, container_path));
    }
    files
}

/// Replace only the first occurrence of `from` with `to` (Python `replace(..,1)`).
fn replace_first(s: &str, from: &str, to: &str) -> String {
    match s.find(from) {
        Some(idx) => {
            let mut out = String::with_capacity(s.len() - from.len() + to.len());
            out.push_str(&s[..idx]);
            out.push_str(to);
            out.push_str(&s[idx + from.len()..]);
            out
        }
        None => s.to_string(),
    }
}

/// Errors raised by transport callbacks or sync operations.
pub type SyncError = Box<dyn std::error::Error + Send + Sync>;

/// Transport callbacks provided by each backend.
///
/// Field names mirror the Python callable parameter names. Bulk variants are
/// optional; when absent the per-file `upload_fn` is used instead.
pub struct Transport {
    /// `(host_path, remote_path) -> Result`. Uploads a single file.
    pub upload_fn: Box<dyn FnMut(&str, &str) -> Result<(), SyncError>>,
    /// `[(host_path, remote_path), ...] -> Result`. Uploads a batch.
    pub bulk_upload_fn: Option<Box<dyn FnMut(&[(String, String)]) -> Result<(), SyncError>>>,
    /// `(dest_tar_path) -> Result`. Writes a tar archive of remote `.hermes/`.
    pub bulk_download_fn: Option<Box<dyn FnMut(&Path) -> Result<(), SyncError>>>,
    /// `(remote_paths) -> Result`. Deletes remote files.
    pub delete_fn: Box<dyn FnMut(&[String]) -> Result<(), SyncError>>,
    /// `() -> [(host_path, remote_path), ...]`. Current file source.
    pub get_files_fn: Box<dyn FnMut() -> Vec<(String, String)>>,
}

/// Tracks local file changes and syncs to a remote environment.
///
/// Backends instantiate this with [`Transport`] callbacks. The manager
/// handles mtime-based change detection, deletion tracking, rate limiting,
/// and transactional state.
///
/// Not used by bind-mount backends (Docker, Singularity).
pub struct FileSyncManager {
    transport: Transport,
    /// remote_path -> (mtime, size)
    synced_files: HashMap<String, FileKey>,
    /// remote_path -> sha256 hex digest
    pushed_hashes: HashMap<String, String>,
    /// monotonic seconds; `None` ensures the first sync runs.
    last_sync_time: Option<std::time::Instant>,
    sync_interval: f64,
    /// Hook for tests / patching retry sleeps; defaults to `std::thread::sleep`.
    sleep_fn: Box<dyn FnMut(u64)>,
}

impl FileSyncManager {
    /// Create a manager from a [`Transport`] using the default sync interval.
    pub fn new(transport: Transport) -> Self {
        Self::with_interval(transport, SYNC_INTERVAL_SECONDS)
    }

    /// Create a manager with an explicit sync interval (seconds).
    pub fn with_interval(transport: Transport, sync_interval: f64) -> Self {
        FileSyncManager {
            transport,
            synced_files: HashMap::new(),
            pushed_hashes: HashMap::new(),
            last_sync_time: None,
            sync_interval,
            sleep_fn: Box::new(|secs| std::thread::sleep(std::time::Duration::from_secs(secs))),
        }
    }

    /// Override the retry-sleep hook (used by tests to avoid real waits).
    pub fn set_sleep_fn(&mut self, f: Box<dyn FnMut(u64)>) {
        self.sleep_fn = f;
    }

    /// Read-only view of the committed synced-files cache.
    pub fn synced_files(&self) -> &HashMap<String, FileKey> {
        &self.synced_files
    }

    /// Read-only view of the pushed content-hash cache.
    pub fn pushed_hashes(&self) -> &HashMap<String, String> {
        &self.pushed_hashes
    }

    fn force_sync_env_set() -> bool {
        std::env::var(FORCE_SYNC_ENV)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Run a sync cycle: upload changed files, delete removed files.
    ///
    /// Rate-limited to once per `sync_interval` unless `force` is true or
    /// `HERMES_FORCE_FILE_SYNC` is set to a non-empty value.
    ///
    /// Transactional: state is only committed if ALL operations succeed. On
    /// failure, state rolls back so the next cycle retries everything.
    pub fn sync(&mut self, force: bool) {
        if !force && !Self::force_sync_env_set() {
            if let Some(last) = self.last_sync_time {
                if last.elapsed().as_secs_f64() < self.sync_interval {
                    return;
                }
            }
        }

        let current_files = (self.transport.get_files_fn)();
        let current_remote_paths: BTreeSet<String> =
            current_files.iter().map(|(_, r)| r.clone()).collect();

        // --- Uploads: new or changed files ---
        let mut to_upload: Vec<(String, String)> = Vec::new();
        let mut new_files = self.synced_files.clone();
        for (host_path, remote_path) in &current_files {
            let file_key = match file_mtime_key(host_path) {
                Some(k) => k,
                None => continue,
            };
            if self.synced_files.get(remote_path) == Some(&file_key) {
                continue;
            }
            to_upload.push((host_path.clone(), remote_path.clone()));
            new_files.insert(remote_path.clone(), file_key);
        }

        // --- Deletes: synced paths no longer in current set ---
        let to_delete: Vec<String> = self
            .synced_files
            .keys()
            .filter(|p| !current_remote_paths.contains(*p))
            .cloned()
            .collect();

        if to_upload.is_empty() && to_delete.is_empty() {
            self.last_sync_time = Some(std::time::Instant::now());
            return;
        }

        // Snapshot for rollback (only when there's work to do).
        let prev_files = self.synced_files.clone();
        let prev_hashes = self.pushed_hashes.clone();

        if !to_upload.is_empty() {
            log::debug!("file_sync: uploading {} file(s)", to_upload.len());
        }
        if !to_delete.is_empty() {
            log::debug!(
                "file_sync: deleting {} stale remote file(s)",
                to_delete.len()
            );
        }

        let result = self.do_sync_transport(&to_upload, &to_delete);

        match result {
            Ok(()) => {
                // --- Commit (all succeeded) ---
                for (host_path, remote_path) in &to_upload {
                    if let Ok(digest) = sha256_file(host_path) {
                        self.pushed_hashes.insert(remote_path.clone(), digest);
                    } else {
                        // Python would raise on read failure inside try and
                        // roll back. _sha256_file opens the file; if it fails
                        // here we surface it as a rollback to match semantics.
                        self.synced_files = prev_files;
                        self.pushed_hashes = prev_hashes;
                        self.last_sync_time = Some(std::time::Instant::now());
                        log::warn!(
                            "file_sync: sync failed, rolled back state: could not hash {host_path}"
                        );
                        return;
                    }
                }

                for p in &to_delete {
                    new_files.remove(p);
                    self.pushed_hashes.remove(p);
                }

                self.synced_files = new_files;
                self.last_sync_time = Some(std::time::Instant::now());
            }
            Err(exc) => {
                self.synced_files = prev_files;
                self.pushed_hashes = prev_hashes;
                self.last_sync_time = Some(std::time::Instant::now());
                log::warn!("file_sync: sync failed, rolled back state: {exc}");
            }
        }
    }

    fn do_sync_transport(
        &mut self,
        to_upload: &[(String, String)],
        to_delete: &[String],
    ) -> Result<(), SyncError> {
        if !to_upload.is_empty() {
            if let Some(bulk) = self.transport.bulk_upload_fn.as_mut() {
                bulk(to_upload)?;
                log::debug!("file_sync: bulk-uploaded {} file(s)", to_upload.len());
            } else {
                for (host_path, remote_path) in to_upload {
                    (self.transport.upload_fn)(host_path, remote_path)?;
                    log::debug!("file_sync: uploaded {host_path} -> {remote_path}");
                }
            }
        }

        if !to_delete.is_empty() {
            (self.transport.delete_fn)(to_delete)?;
            log::debug!("file_sync: deleted {to_delete:?}");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Sync-back: pull remote changes to host on teardown
    // ------------------------------------------------------------------

    /// Pull remote changes back to the host filesystem.
    ///
    /// Downloads the remote `.hermes/` directory as a tar archive, unpacks
    /// it, and applies only files that differ from what was originally
    /// pushed (based on SHA-256 content hashes).
    ///
    /// `hermes_home` defaults to the supplied value, or the caller should
    /// pass the result of `crate::mod_hermes_constants::get_hermes_home()`.
    /// Retries with exponential backoff. (SIGINT deferral and cross-process
    /// file locking from the Python original are platform-specific and
    /// handled at the call site; this port serializes only logically.)
    pub fn sync_back(&mut self, hermes_home: PathBuf) {
        if self.transport.bulk_download_fn.is_none() {
            return;
        }

        // Nothing was ever committed through this manager — skip to avoid
        // retry storms against an uninitialized remote .hermes/ directory.
        if self.pushed_hashes.is_empty() && self.synced_files.is_empty() {
            log::debug!("sync_back: no prior push state — skipping");
            return;
        }

        let lock_path = hermes_home.join(".sync.lock");
        if let Some(parent) = lock_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let mut last_exc: Option<SyncError> = None;
        for attempt in 0..SYNC_BACK_MAX_RETRIES {
            match self.sync_back_once(&lock_path) {
                Ok(()) => return,
                Err(exc) => {
                    if attempt < SYNC_BACK_MAX_RETRIES - 1 {
                        let delay = SYNC_BACK_BACKOFF[attempt];
                        log::warn!(
                            "sync_back: attempt {} failed ({}), retrying in {}s",
                            attempt + 1,
                            exc,
                            delay
                        );
                        (self.sleep_fn)(delay);
                    }
                    last_exc = Some(exc);
                }
            }
        }

        match last_exc {
            Some(exc) => log::warn!(
                "sync_back: all {SYNC_BACK_MAX_RETRIES} attempts failed: {exc}"
            ),
            None => log::warn!("sync_back: all {SYNC_BACK_MAX_RETRIES} attempts failed"),
        }
    }

    /// Single sync-back attempt. The Python original adds SIGINT deferral and
    /// an `flock`-based file lock around this; both are environment-specific
    /// concerns delegated to the call site here.
    fn sync_back_once(&mut self, lock_path: &Path) -> Result<(), SyncError> {
        // Acquire an exclusive lock via the lock file's presence. We open the
        // file to mirror the Python contract of creating the lock path; true
        // cross-process flock is left to the integration layer.
        let _lock = File::create(lock_path).map_err(|e| -> SyncError { Box::new(e) })?;
        self.sync_back_impl()
    }

    fn sync_back_impl(&mut self) -> Result<(), SyncError> {
        if self.transport.bulk_download_fn.is_none() {
            return Err("_sync_back_impl called without bulk_download_fn".into());
        }

        // Cache file mapping once to avoid O(n*m) from repeated iteration.
        let file_mapping: Vec<(String, String)> = (self.transport.get_files_fn)();

        // Create a temp tar file path.
        let tar_path = temp_path_with_suffix(".tar");
        let download_res = {
            let f = self.transport.bulk_download_fn.as_mut().unwrap();
            f(&tar_path)
        };
        if let Err(e) = download_res {
            let _ = fs::remove_file(&tar_path);
            return Err(e);
        }

        let result = self.extract_and_apply(&tar_path, &file_mapping);
        let _ = fs::remove_file(&tar_path);
        result
    }

    fn extract_and_apply(
        &self,
        tar_path: &Path,
        file_mapping: &[(String, String)],
    ) -> Result<(), SyncError> {
        // Defensive size cap.
        let tar_size = fs::metadata(tar_path).map(|m| m.len()).unwrap_or(0);
        if tar_size > SYNC_BACK_MAX_BYTES {
            log::warn!(
                "sync_back: remote tar is {tar_size} bytes (cap {SYNC_BACK_MAX_BYTES}) — skipping extraction"
            );
            return Ok(());
        }

        // Staging directory.
        let staging = temp_dir_with_prefix("hermes-sync-back-")?;

        let extract_res = (|| -> Result<(), SyncError> {
            let tf = File::open(tar_path)?;
            let mut archive = tar::Archive::new(tf);
            // filter="data": tar extraction is unpacked into staging.
            archive.unpack(&staging)?;

            let mut applied = 0u64;
            for staged_file in walk_files(&staging)? {
                let rel = staged_file
                    .strip_prefix(&staging)
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|_| staged_file.clone());
                let remote_path = format!("/{}", rel.to_string_lossy());
                let staged_str = staged_file.to_string_lossy().into_owned();

                let pushed_hash = self.pushed_hashes.get(&remote_path).cloned();

                // Skip hashing for files unchanged from push.
                if let Some(ref ph) = pushed_hash {
                    let remote_hash = sha256_file(&staged_str)?;
                    if &remote_hash == ph {
                        continue;
                    }
                }

                // Resolve host path from cached mapping.
                let host_path = match self.resolve_host_path(&remote_path, file_mapping) {
                    Some(h) => h,
                    None => match self.infer_host_path(&remote_path, file_mapping) {
                        Some(h) => h,
                        None => {
                            log::debug!("sync_back: skipping {remote_path} (no host mapping)");
                            continue;
                        }
                    },
                };

                if Path::new(&host_path).exists() {
                    if let Some(ref ph) = pushed_hash {
                        if let Ok(host_hash) = sha256_file(&host_path) {
                            if &host_hash != ph {
                                log::warn!(
                                    "sync_back: conflict on {remote_path} — host modified \
                                     since push, remote also changed. Applying remote \
                                     version (last-write-wins)."
                                );
                            }
                        }
                    }
                }

                if let Some(parent) = Path::new(&host_path).parent() {
                    fs::create_dir_all(parent)?;
                }
                copy2(&staged_str, &host_path)?;
                applied += 1;
            }

            if applied > 0 {
                log::info!("sync_back: applied {applied} changed file(s)");
            } else {
                log::debug!("sync_back: no remote changes detected");
            }
            Ok(())
        })();

        let _ = fs::remove_dir_all(&staging);
        extract_res
    }

    /// Find the host path for a known remote path from the file mapping.
    pub fn resolve_host_path(
        &self,
        remote_path: &str,
        file_mapping: &[(String, String)],
    ) -> Option<String> {
        for (host, remote) in file_mapping {
            if remote == remote_path {
                return Some(host.clone());
            }
        }
        None
    }

    /// Infer a host path for a new remote file by matching path prefixes.
    ///
    /// Uses the existing file mapping to find a remote->host directory pair,
    /// then applies the same prefix substitution to the new file.
    pub fn infer_host_path(
        &self,
        remote_path: &str,
        file_mapping: &[(String, String)],
    ) -> Option<String> {
        for (host, remote) in file_mapping {
            let remote_dir = path_parent_str(remote);
            let prefix = format!("{remote_dir}/");
            if remote_path.starts_with(&prefix) {
                let host_dir = path_parent_str(host);
                let suffix = &remote_path[remote_dir.len()..];
                return Some(format!("{host_dir}{suffix}"));
            }
        }
        None
    }
}

/// Recursively list all regular files under `root` (mirrors `os.walk`).
fn walk_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Copy file contents and permissions/timestamps best-effort (`shutil.copy2`).
fn copy2(src: &str, dst: &str) -> std::io::Result<()> {
    fs::copy(src, dst)?;
    // copy2 preserves metadata; fs::copy already preserves permission bits on
    // Unix. Timestamp preservation is best-effort and omitted (not relied on
    // for correctness in callers).
    Ok(())
}

/// Generate a unique temp file path with the given suffix without creating it.
fn temp_path_with_suffix(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let uniq = format!(
        "hermes-sync-{}-{}{}",
        std::process::id(),
        unique_counter(),
        suffix
    );
    p.push(uniq);
    p
}

/// Create a unique temp directory with the given prefix.
fn temp_dir_with_prefix(prefix: &str) -> std::io::Result<PathBuf> {
    let mut p = std::env::temp_dir();
    let uniq = format!("{}{}-{}", prefix, std::process::id(), unique_counter());
    p.push(uniq);
    fs::create_dir_all(&p)?;
    Ok(p)
}

fn unique_counter() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(2654435761)
}

#[allow(dead_code)]
fn write_all_helper(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn shlex_quote_safe_and_unsafe() {
        assert_eq!(shlex_quote("abc/def.txt"), "abc/def.txt");
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn quoted_rm_and_mkdir() {
        let paths = vec!["/root/a b".to_string(), "/root/c".to_string()];
        assert_eq!(quoted_rm_command(&paths), "rm -f '/root/a b' /root/c");
        let dirs = vec!["/x".to_string(), "/y z".to_string()];
        assert_eq!(quoted_mkdir_command(&dirs), "mkdir -p /x '/y z'");
    }

    #[test]
    fn unique_parent_dirs_sorted_unique() {
        let files = vec![
            ("/h/a".to_string(), "/root/.hermes/x/a".to_string()),
            ("/h/b".to_string(), "/root/.hermes/x/b".to_string()),
            ("/h/c".to_string(), "/root/.hermes/y/c".to_string()),
        ];
        assert_eq!(
            unique_parent_dirs(&files),
            vec!["/root/.hermes/x".to_string(), "/root/.hermes/y".to_string()]
        );
    }

    #[test]
    fn iter_sync_files_remaps_credential_prefix_once() {
        let creds = vec![(
            "/host/cred".to_string(),
            "/root/.hermes/creds/.env".to_string(),
        )];
        let skills = vec![("/host/s".to_string(), "/home/user/.hermes/skills/a".to_string())];
        let cache: Vec<(String, String)> = vec![];
        let files = iter_sync_files("/home/user/.hermes", creds, skills, cache);
        assert_eq!(files[0].1, "/home/user/.hermes/creds/.env");
        assert_eq!(files[1].1, "/home/user/.hermes/skills/a");
    }

    #[test]
    fn replace_first_only_first() {
        assert_eq!(
            replace_first("/root/.hermes/root/.hermes", "/root/.hermes", "/x"),
            "/x/root/.hermes"
        );
    }

    #[test]
    fn infer_host_path_prefix_substitution() {
        let t = make_noop_transport();
        let mgr = FileSyncManager::new(t);
        let mapping = vec![(
            "/home/me/.hermes/skills/a.md".to_string(),
            "/root/.hermes/skills/a.md".to_string(),
        )];
        let got = mgr.infer_host_path("/root/.hermes/skills/b.md", &mapping);
        assert_eq!(got, Some("/home/me/.hermes/skills/b.md".to_string()));
    }

    #[test]
    fn resolve_host_path_exact() {
        let t = make_noop_transport();
        let mgr = FileSyncManager::new(t);
        let mapping = vec![("/h/a".to_string(), "/r/a".to_string())];
        assert_eq!(mgr.resolve_host_path("/r/a", &mapping), Some("/h/a".to_string()));
        assert_eq!(mgr.resolve_host_path("/r/z", &mapping), None);
    }

    #[test]
    fn sha256_file_matches_known() {
        let dir = temp_dir_with_prefix("hermes-test-").unwrap();
        let p = dir.join("f.txt");
        write_all_helper(&p, b"hello").unwrap();
        let digest = sha256_file(&p.to_string_lossy()).unwrap();
        // sha256("hello")
        assert_eq!(
            digest,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn make_noop_transport() -> Transport {
        Transport {
            upload_fn: Box::new(|_h, _r| Ok(())),
            bulk_upload_fn: None,
            bulk_download_fn: None,
            delete_fn: Box::new(|_p| Ok(())),
            get_files_fn: Box::new(Vec::new),
        }
    }

    #[test]
    fn sync_uploads_new_files_and_commits() {
        let dir = temp_dir_with_prefix("hermes-sync-test-").unwrap();
        let host = dir.join("a.txt");
        write_all_helper(&host, b"data").unwrap();
        let host_str = host.to_string_lossy().into_owned();

        let uploads: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));
        let up = uploads.clone();

        let host_for_files = host_str.clone();
        let t = Transport {
            upload_fn: Box::new(move |h, r| {
                up.borrow_mut().push((h.to_string(), r.to_string()));
                Ok(())
            }),
            bulk_upload_fn: None,
            bulk_download_fn: None,
            delete_fn: Box::new(|_p| Ok(())),
            get_files_fn: Box::new(move || {
                vec![(host_for_files.clone(), "/root/.hermes/a.txt".to_string())]
            }),
        };
        let mut mgr = FileSyncManager::new(t);
        mgr.sync(true);

        assert_eq!(uploads.borrow().len(), 1);
        assert_eq!(uploads.borrow()[0].1, "/root/.hermes/a.txt");
        // Committed state.
        assert!(mgr.synced_files().contains_key("/root/.hermes/a.txt"));
        assert!(mgr.pushed_hashes().contains_key("/root/.hermes/a.txt"));

        // Second sync (force) with no change => no new upload.
        mgr.sync(true);
        assert_eq!(uploads.borrow().len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sync_rolls_back_on_upload_failure() {
        let dir = temp_dir_with_prefix("hermes-sync-fail-").unwrap();
        let host = dir.join("a.txt");
        write_all_helper(&host, b"data").unwrap();
        let host_for_files = host.to_string_lossy().into_owned();

        let t = Transport {
            upload_fn: Box::new(|_h, _r| Err("boom".into())),
            bulk_upload_fn: None,
            bulk_download_fn: None,
            delete_fn: Box::new(|_p| Ok(())),
            get_files_fn: Box::new(move || {
                vec![(host_for_files.clone(), "/root/.hermes/a.txt".to_string())]
            }),
        };
        let mut mgr = FileSyncManager::new(t);
        mgr.sync(true);
        // Rolled back: nothing committed.
        assert!(mgr.synced_files().is_empty());
        assert!(mgr.pushed_hashes().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sync_deletes_stale_remote_files() {
        // First commit a file, then have get_files return empty -> delete.
        let dir = temp_dir_with_prefix("hermes-sync-del-").unwrap();
        let host = dir.join("a.txt");
        write_all_helper(&host, b"data").unwrap();
        let host_str = host.to_string_lossy().into_owned();

        let present: Rc<RefCell<bool>> = Rc::new(RefCell::new(true));
        let deleted: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let del = deleted.clone();
        let present_for_files = present.clone();
        let host_for_files = host_str.clone();

        let t = Transport {
            upload_fn: Box::new(|_h, _r| Ok(())),
            bulk_upload_fn: None,
            bulk_download_fn: None,
            delete_fn: Box::new(move |p| {
                del.borrow_mut().extend(p.iter().cloned());
                Ok(())
            }),
            get_files_fn: Box::new(move || {
                if *present_for_files.borrow() {
                    vec![(host_for_files.clone(), "/root/.hermes/a.txt".to_string())]
                } else {
                    vec![]
                }
            }),
        };
        let mut mgr = FileSyncManager::new(t);
        mgr.sync(true);
        assert!(mgr.synced_files().contains_key("/root/.hermes/a.txt"));

        *present.borrow_mut() = false;
        mgr.sync(true);
        assert_eq!(deleted.borrow().as_slice(), ["/root/.hermes/a.txt"]);
        assert!(mgr.synced_files().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sync_back_skips_without_prior_state() {
        let called: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
        let c = called.clone();
        let t = Transport {
            upload_fn: Box::new(|_h, _r| Ok(())),
            bulk_upload_fn: None,
            bulk_download_fn: Some(Box::new(move |_p| {
                *c.borrow_mut() = true;
                Ok(())
            })),
            delete_fn: Box::new(|_p| Ok(())),
            get_files_fn: Box::new(Vec::new),
        };
        let mut mgr = FileSyncManager::new(t);
        mgr.sync_back(std::env::temp_dir());
        // No prior push state -> download never called.
        assert!(!*called.borrow());
    }

    #[test]
    fn sync_back_noop_without_bulk_download() {
        let t = make_noop_transport();
        let mut mgr = FileSyncManager::new(t);
        mgr.pushed_hashes.insert("/r/a".to_string(), "x".to_string());
        // bulk_download_fn is None -> early return, no panic.
        mgr.sync_back(std::env::temp_dir());
    }

    #[test]
    fn path_parent_handles_root_and_relative() {
        assert_eq!(path_parent_str("/root/.hermes/a"), "/root/.hermes");
        assert_eq!(path_parent_str("/a"), "/");
        assert_eq!(path_parent_str("a"), ".");
    }
}
