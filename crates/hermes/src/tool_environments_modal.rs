//! Modal cloud execution environment using the native Modal SDK directly.
//!
//! Native Rust port of `tools/environments/modal.py`.
//!
//! The Python original drives Modal's async SDK (`Sandbox.create()` +
//! `Sandbox.exec()`) from a dedicated background-thread event loop
//! (`_AsyncWorker`), while preserving Hermes' persistent filesystem-snapshot
//! behaviour across sessions.
//!
//! There is no native Rust Modal SDK, so the SDK surface is abstracted behind
//! the [`ModalSandbox`] trait (mirroring the daytona port's `DaytonaApi`
//! pattern). The environment logic — snapshot-store persistence, image-spec
//! resolution, upload/download/delete command construction, tar archive
//! building, stdin chunking, `_run_bash` output formatting, and the
//! create/restore/snapshot lifecycle — is reproduced faithfully and is testable
//! without a live Modal account.
//!
//! Faithful behaviours preserved:
//!   * Snapshot store layout in `{hermes_home}/modal_snapshots.json` with the
//!     `direct:{task_id}` namespaced key plus legacy-key fallback/migration.
//!   * `_resolve_modal_image`: `im-` ids -> `Image.from_id`; ubuntu/debian
//!     registry refs get an `add_python` apt step prepended; the ensurepip
//!     dockerfile command is always present.
//!   * `_modal_upload`: single file base64-piped through `mkdir -p && base64 -d`.
//!   * `_modal_bulk_upload`: gzipped tar streamed through
//!     `<mkdir> && base64 -d | tar xzf - -C /`, 1 MiB stdin chunks.
//!   * `_modal_bulk_download`: `tar cf - -C / root/.hermes`.
//!   * `_modal_delete`: batched `rm -f`.
//!   * `_run_bash`: `bash -c` / `bash -l -c`, stdout/stderr merge rule, exit
//!     code propagation, terminate-based cancel.
//!   * `cleanup`: sync_back, then snapshot (persistent) + terminate.

use std::sync::Arc;

use serde_json::Value;

// ---------------------------------------------------------------------------
// shell quoting (mirrors Python shlex.quote)
// ---------------------------------------------------------------------------

/// POSIX shell single-quote a string, matching Python's `shlex.quote`.
///
/// Empty string becomes `''`; strings free of "unsafe" characters are returned
/// unchanged; otherwise the string is single-quoted with embedded single quotes
/// escaped as `'"'"'`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = |c: char| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    };
    if s.chars().all(safe) {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    out.push_str(&s.replace('\'', "'\"'\"'"));
    out.push('\'');
    out
}

/// Parent directory of a path rendered as a string, matching Python's
/// `str(Path(p).parent)`.
fn parent_dir(path: &str) -> String {
    let trimmed = {
        let t = path.trim_end_matches('/');
        if t.is_empty() && path.starts_with('/') {
            return "/".to_string();
        }
        t
    };
    match trimmed.rfind('/') {
        Some(0) => "/".to_string(),
        Some(idx) => trimmed[..idx].to_string(),
        None => ".".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Snapshot store constants & helpers
// ---------------------------------------------------------------------------

/// Filename of the snapshot store under the hermes home directory. Mirrors
/// `_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"`.
pub const SNAPSHOT_STORE_FILENAME: &str = "modal_snapshots.json";

/// Namespace prefix for direct-SDK snapshot keys. Mirrors
/// `_DIRECT_SNAPSHOT_NAMESPACE`.
pub const DIRECT_SNAPSHOT_NAMESPACE: &str = "direct";

/// Modal SDK stdin buffer chunk size (1 MiB). Mirrors `_STDIN_CHUNK_SIZE`.
pub const STDIN_CHUNK_SIZE: usize = 1024 * 1024;

/// Compute the absolute path to the snapshot store given a hermes-home dir.
/// Mirrors `_SNAPSHOT_STORE`.
pub fn snapshot_store_path(hermes_home: &std::path::Path) -> std::path::PathBuf {
    hermes_home.join(SNAPSHOT_STORE_FILENAME)
}

/// Build the namespaced snapshot key. Mirrors `_direct_snapshot_key`.
pub fn direct_snapshot_key(task_id: &str) -> String {
    format!("{DIRECT_SNAPSHOT_NAMESPACE}:{task_id}")
}

/// Load the snapshot store JSON object. Mirrors `_load_snapshots`.
pub fn load_snapshots(store_path: &std::path::Path) -> serde_json::Map<String, Value> {
    crate::tool_environments_base::load_json_store(store_path)
}

/// Persist the snapshot store JSON object. Mirrors `_save_snapshots`.
pub fn save_snapshots(
    store_path: &std::path::Path,
    data: &serde_json::Map<String, Value>,
) -> std::io::Result<()> {
    crate::tool_environments_base::save_json_store(store_path, data)
}

/// Resolve a snapshot id to restore for a task.
///
/// Returns `(snapshot_id, from_legacy_key)`. Mirrors
/// `_get_snapshot_restore_candidate`: prefer the namespaced `direct:{task}`
/// key, then fall back to a bare `task_id` legacy key (flagged so the caller can
/// migrate it).
pub fn get_snapshot_restore_candidate(
    store_path: &std::path::Path,
    task_id: &str,
) -> (Option<String>, bool) {
    let snapshots = load_snapshots(store_path);
    let namespaced_key = direct_snapshot_key(task_id);
    if let Some(Value::String(sid)) = snapshots.get(&namespaced_key) {
        if !sid.is_empty() {
            return (Some(sid.clone()), false);
        }
    }
    if let Some(Value::String(sid)) = snapshots.get(task_id) {
        if !sid.is_empty() {
            return (Some(sid.clone()), true);
        }
    }
    (None, false)
}

/// Store a snapshot id under the namespaced key and drop any legacy bare key.
/// Mirrors `_store_direct_snapshot`.
pub fn store_direct_snapshot(
    store_path: &std::path::Path,
    task_id: &str,
    snapshot_id: &str,
) -> std::io::Result<()> {
    let mut snapshots = load_snapshots(store_path);
    snapshots.insert(direct_snapshot_key(task_id), Value::String(snapshot_id.to_string()));
    snapshots.remove(task_id);
    save_snapshots(store_path, &snapshots)
}

/// Delete the snapshot entry/entries for a task.
///
/// Mirrors `_delete_direct_snapshot`: when `snapshot_id` is `None`, both the
/// namespaced and legacy keys are removed; when provided, a key is only removed
/// if its value matches. Persists only when something changed.
pub fn delete_direct_snapshot(
    store_path: &std::path::Path,
    task_id: &str,
    snapshot_id: Option<&str>,
) -> std::io::Result<()> {
    let mut snapshots = load_snapshots(store_path);
    let mut updated = false;
    let namespaced = direct_snapshot_key(task_id);
    for key in [namespaced.as_str(), task_id] {
        let value = match snapshots.get(key) {
            Some(v) => v.clone(),
            None => continue,
        };
        // Python: `if value is None: continue` — a JSON null is treated as None.
        if value.is_null() {
            continue;
        }
        let matches = match snapshot_id {
            None => true,
            Some(sid) => value.as_str() == Some(sid),
        };
        if matches {
            snapshots.remove(key);
            updated = true;
        }
    }
    if updated {
        save_snapshots(store_path, &snapshots)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Image spec resolution
// ---------------------------------------------------------------------------

/// A resolved Modal image reference, the native-Rust analogue of the Modal
/// `Image` objects returned by `_resolve_modal_image`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalImage {
    /// `Image.from_id(<id>)` — a snapshot/image id beginning with `im-`.
    FromId(String),
    /// `Image.from_registry(ref, setup_dockerfile_commands=...)`.
    FromRegistry {
        reference: String,
        setup_dockerfile_commands: Vec<String>,
    },
}

/// The ensurepip dockerfile command always appended by `_resolve_modal_image`.
const ENSUREPIP_COMMAND: &str = "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; \
python -m ensurepip --upgrade --default-pip 2>/dev/null || true";

/// The apt python install command prepended for ubuntu/debian images.
const ADD_PYTHON_COMMAND: &str =
    "RUN apt-get update -qq && apt-get install -y -qq python3 python3-venv > /dev/null 2>&1 || true";

/// Convert a registry reference or snapshot id into a [`ModalImage`].
///
/// Faithful port of `_resolve_modal_image` (string-input branch). Non-string
/// image specs in Python are passed through untouched; in this typed port the
/// caller already holds a `&str`, so that pass-through is the caller's concern.
///
/// Includes the PR-4511 `add_python` behaviour: ubuntu/debian registry refs get
/// an apt `python3`/`python3-venv` install step prepended before the ensurepip
/// step.
pub fn resolve_modal_image(image_spec: &str) -> ModalImage {
    if image_spec.starts_with("im-") {
        return ModalImage::FromId(image_spec.to_string());
    }

    let lower = image_spec.to_lowercase();
    let add_python = ["ubuntu", "debian"].iter().any(|base| lower.contains(base));

    let mut setup_commands = vec![ENSUREPIP_COMMAND.to_string()];
    if add_python {
        setup_commands.insert(0, ADD_PYTHON_COMMAND.to_string());
    }

    ModalImage::FromRegistry {
        reference: image_spec.to_string(),
        setup_dockerfile_commands: setup_commands,
    }
}

// ---------------------------------------------------------------------------
// Modal SDK surface (abstracted for testability)
// ---------------------------------------------------------------------------

/// Outcome of a sandbox `exec` call. Mirrors the pieces the Python code reads
/// off a Modal process object (`stdout`, `stderr`, `wait()` exit code).
#[derive(Debug, Clone, Default)]
pub struct ExecOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

/// Error type for sandbox operations.
pub type ModalError = Box<dyn std::error::Error + Send + Sync>;

/// Abstracted Modal sandbox surface.
///
/// Each method corresponds to an async Modal SDK call the Python module issues
/// via `_AsyncWorker.run_coroutine`. Implementations talk to the real Modal
/// service; the environment logic here is transport-agnostic.
pub trait ModalSandbox: Send {
    /// `sandbox.exec("bash", "-c", cmd, ...)` with stdin written then EOF.
    ///
    /// `args` is the full argv (e.g. `["bash", "-c", cmd]` or
    /// `["bash", "-l", "-c", cmd]`). `stdin` (if any) is written in
    /// [`STDIN_CHUNK_SIZE`] chunks then EOF'd before reading output. `timeout`
    /// is the per-exec timeout in seconds (Modal's `exec(timeout=...)`); `None`
    /// means unset.
    fn exec(
        &self,
        args: &[String],
        stdin: Option<&[u8]>,
        timeout: Option<i32>,
    ) -> Result<ExecOutput, ModalError>;

    /// `sandbox.snapshot_filesystem()` -> the resulting image/object id.
    fn snapshot_filesystem(&self) -> Result<String, ModalError>;

    /// `sandbox.terminate()`.
    fn terminate(&self) -> Result<(), ModalError>;
}

// ---------------------------------------------------------------------------
// Command construction helpers (pure)
// ---------------------------------------------------------------------------

/// Build the `bash -c` command string used by [`modal_upload`]. Mirrors the
/// `cmd` string assembled in `_modal_upload`.
pub fn upload_command(remote_path: &str) -> String {
    let container_dir = parent_dir(remote_path);
    format!(
        "mkdir -p {} && base64 -d > {}",
        shlex_quote(&container_dir),
        shlex_quote(remote_path)
    )
}

/// Build the `bash -c` command string used by [`modal_bulk_upload`]. Mirrors
/// the `cmd` string assembled in `_modal_bulk_upload`.
pub fn bulk_upload_command(files: &[(String, String)]) -> String {
    let parents = crate::tool_environments_file_sync::unique_parent_dirs(files);
    let mkdir_part = crate::tool_environments_file_sync::quoted_mkdir_command(&parents);
    format!("{mkdir_part} && base64 -d | tar xzf - -C /")
}

/// The fixed bulk-download command. Mirrors `_modal_bulk_download`.
pub const BULK_DOWNLOAD_COMMAND: &str = "tar cf - -C / root/.hermes";

/// Build the argv for `_run_bash`. Mirrors the `args` list construction.
pub fn run_bash_args(cmd_string: &str, login: bool) -> Vec<String> {
    if login {
        vec![
            "bash".to_string(),
            "-l".to_string(),
            "-c".to_string(),
            cmd_string.to_string(),
        ]
    } else {
        vec!["bash".to_string(), "-c".to_string(), cmd_string.to_string()]
    }
}

/// Merge stdout/stderr the way `_run_bash`'s `_do` coroutine does.
///
/// Mirrors:
/// ```text
/// output = stdout
/// if stderr:
///     output = f"{stdout}\n{stderr}" if stdout else stderr
/// ```
/// Both byte streams are decoded UTF-8-lossily (`errors="replace"`).
pub fn merge_stdout_stderr(stdout: &[u8], stderr: &[u8]) -> String {
    let out = String::from_utf8_lossy(stdout);
    let err = String::from_utf8_lossy(stderr);
    if err.is_empty() {
        out.into_owned()
    } else if out.is_empty() {
        err.into_owned()
    } else {
        format!("{out}\n{err}")
    }
}

// ---------------------------------------------------------------------------
// tar archive building (mirrors tarfile w:gz + arcname=remote.lstrip("/"))
// ---------------------------------------------------------------------------

/// Build a gzipped tar archive of `files`, using `remote.lstrip("/")` as each
/// member's arcname. Mirrors the in-memory `tarfile.open(mode="w:gz")` build in
/// `_modal_bulk_upload`. Returns the raw gzip bytes (the caller base64-encodes).
pub fn build_bulk_tar_gz(files: &[(String, String)]) -> std::io::Result<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let buf: Vec<u8> = Vec::new();
    let enc = GzEncoder::new(buf, Compression::default());
    let mut tar = tar::Builder::new(enc);
    for (host_path, remote_path) in files {
        let arcname = remote_path.trim_start_matches('/');
        tar.append_path_with_name(host_path, arcname)?;
    }
    let enc = tar.into_inner()?;
    enc.finish()
}

// ---------------------------------------------------------------------------
// Modal environment
// ---------------------------------------------------------------------------

/// Modal cloud execution via native Modal sandboxes.
///
/// Mirrors `ModalEnvironment`. Construction (image resolution, snapshot restore
/// selection, sandbox creation) is performed via [`ModalEnvironment::create`];
/// the heavy lifting of creating the sandbox is delegated to a caller-supplied
/// factory so this type stays independent of the (absent) native Modal SDK.
pub struct ModalEnvironment {
    pub task_id: String,
    pub persistent: bool,
    pub cwd: String,
    pub timeout: i32,
    store_path: std::path::PathBuf,
    sandbox: Option<Arc<dyn ModalSandbox>>,
    sync_manager: Option<crate::tool_environments_file_sync::FileSyncManager>,
}

/// Stdin embedding mode. Modal uses heredoc. Mirrors `_stdin_mode = "heredoc"`.
pub const STDIN_MODE: crate::tool_environments_base::StdinMode =
    crate::tool_environments_base::StdinMode::Heredoc;

/// Snapshot-creation timeout (seconds). Mirrors `_snapshot_timeout = 60`.
pub const SNAPSHOT_TIMEOUT: i32 = 60;

/// What [`ModalEnvironment::plan_creation`] decided about restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreationPlan {
    /// The image spec that will be used for the (first) creation attempt:
    /// the restored snapshot id if any, else the base image.
    pub target_image: ModalImage,
    /// The restored snapshot id, if a candidate was found.
    pub restored_snapshot_id: Option<String>,
    /// Whether the restore candidate came from the legacy bare key.
    pub restored_from_legacy_key: bool,
}

impl ModalEnvironment {
    /// Decide which image to create the sandbox from.
    ///
    /// Mirrors the restore-candidate logic at the top of `__init__`: when
    /// persistent, look up the snapshot store; the target image is the restored
    /// snapshot (resolved via [`resolve_modal_image`]) or the base `image`.
    pub fn plan_creation(
        store_path: &std::path::Path,
        image: &str,
        persistent: bool,
        task_id: &str,
    ) -> CreationPlan {
        let mut restored_snapshot_id = None;
        let mut restored_from_legacy_key = false;
        if persistent {
            let (sid, legacy) = get_snapshot_restore_candidate(store_path, task_id);
            restored_snapshot_id = sid;
            restored_from_legacy_key = legacy;
            if let Some(ref sid) = restored_snapshot_id {
                let short: String = sid.chars().take(20).collect();
                log::info!("Modal: restoring from snapshot {short}");
            }
        }

        let target_spec = restored_snapshot_id.as_deref().unwrap_or(image);
        let target_image = resolve_modal_image(target_spec);

        CreationPlan {
            target_image,
            restored_snapshot_id,
            restored_from_legacy_key,
        }
    }

    /// Construct a [`ModalEnvironment`] given an already-created sandbox.
    ///
    /// The Python `__init__` creates the sandbox inline; this port separates the
    /// SDK-bound creation step (handled by the caller, who supplies the
    /// resulting [`ModalSandbox`]) from the host-side state setup, mirroring the
    /// post-creation work: storing the migrated legacy snapshot key, building
    /// the [`FileSyncManager`], the initial forced sync, and `init_session`.
    ///
    /// On a successful create from a legacy-key restore, the key is migrated to
    /// the namespaced form (mirrors the `else` branch of `__init__`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_sandbox(
        store_path: std::path::PathBuf,
        sandbox: Arc<dyn ModalSandbox>,
        cwd: String,
        timeout: i32,
        persistent: bool,
        task_id: String,
        plan: &CreationPlan,
        restored_with_target_image: bool,
        sync_manager: crate::tool_environments_file_sync::FileSyncManager,
    ) -> Self {
        // Migrate a legacy-key restore to the namespaced key, but only when the
        // create succeeded against the restored snapshot image (the Python
        // `else` branch runs only if the first attempt didn't raise).
        if restored_with_target_image {
            if let (Some(sid), true) =
                (&plan.restored_snapshot_id, plan.restored_from_legacy_key)
            {
                let _ = store_direct_snapshot(&store_path, &task_id, sid);
            }
        }

        log::info!("Modal: sandbox created (task={task_id})");

        ModalEnvironment {
            task_id,
            persistent,
            cwd,
            timeout,
            store_path,
            sandbox: Some(sandbox),
            sync_manager: Some(sync_manager),
        }
    }

    /// Access the live sandbox handle, if any.
    pub fn sandbox(&self) -> Option<&Arc<dyn ModalSandbox>> {
        self.sandbox.as_ref()
    }

    /// Path to this environment's snapshot store.
    pub fn store_path(&self) -> &std::path::Path {
        &self.store_path
    }

    /// Upload a single file via base64 piped through stdin. Mirrors
    /// `_modal_upload`.
    pub fn modal_upload(&self, host_path: &str, remote_path: &str) -> Result<(), ModalError> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| -> ModalError { "Modal: no sandbox".into() })?;
        let content = std::fs::read(host_path)?;
        let b64 = base64_encode(&content);
        let cmd = upload_command(remote_path);
        let args = vec!["bash".to_string(), "-c".to_string(), cmd];
        sandbox.exec(&args, Some(b64.as_bytes()), None)?;
        Ok(())
    }

    /// Upload many files via a gzipped tar archive piped through stdin. Mirrors
    /// `_modal_bulk_upload`.
    pub fn modal_bulk_upload(&self, files: &[(String, String)]) -> Result<(), ModalError> {
        if files.is_empty() {
            return Ok(());
        }
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| -> ModalError { "Modal: no sandbox".into() })?;

        let tar_gz = build_bulk_tar_gz(files)?;
        let payload = base64_encode(&tar_gz);
        let cmd = bulk_upload_command(files);
        let args = vec!["bash".to_string(), "-c".to_string(), cmd];

        let out = sandbox.exec(&args, Some(payload.as_bytes()), None)?;
        if out.exit_code != 0 {
            let stderr_text = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "Modal bulk upload failed (exit {}): {}",
                out.exit_code, stderr_text
            )
            .into());
        }
        Ok(())
    }

    /// Download the remote `/root/.hermes` directory as a tar archive to `dest`.
    /// Mirrors `_modal_bulk_download`.
    pub fn modal_bulk_download(&self, dest: &std::path::Path) -> Result<(), ModalError> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| -> ModalError { "Modal: no sandbox".into() })?;
        let args = vec![
            "bash".to_string(),
            "-c".to_string(),
            BULK_DOWNLOAD_COMMAND.to_string(),
        ];
        let out = sandbox.exec(&args, None, None)?;
        if out.exit_code != 0 {
            return Err(
                format!("Modal bulk download failed (exit {})", out.exit_code).into(),
            );
        }
        std::fs::write(dest, &out.stdout)?;
        Ok(())
    }

    /// Batch-delete remote files. Mirrors `_modal_delete`.
    pub fn modal_delete(&self, remote_paths: &[String]) -> Result<(), ModalError> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| -> ModalError { "Modal: no sandbox".into() })?;
        let rm_cmd = crate::tool_environments_file_sync::quoted_rm_command(remote_paths);
        let args = vec!["bash".to_string(), "-c".to_string(), rm_cmd];
        sandbox.exec(&args, None, None)?;
        Ok(())
    }

    /// Run a bash command in the sandbox and return `(output, exit_code)`.
    ///
    /// Mirrors `_run_bash`'s `_do` coroutine: build argv (login adds `-l`),
    /// exec with `timeout`, read stdout/stderr, wait for exit code, then merge
    /// the streams per [`merge_stdout_stderr`].
    pub fn run_bash(
        &self,
        cmd_string: &str,
        login: bool,
        timeout: i32,
    ) -> Result<(String, i32), ModalError> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| -> ModalError { "Modal: no sandbox".into() })?;
        let args = run_bash_args(cmd_string, login);
        let out = sandbox.exec(&args, None, Some(timeout))?;
        let output = merge_stdout_stderr(&out.stdout, &out.stderr);
        Ok((output, out.exit_code))
    }

    /// Terminate the sandbox (used as the cancel function for interrupts).
    /// Mirrors `_run_bash`'s `cancel`.
    pub fn cancel(&self) -> Result<(), ModalError> {
        if let Some(sandbox) = self.sandbox.as_ref() {
            sandbox.terminate()?;
        }
        Ok(())
    }

    /// Sync files to the sandbox (rate-limited internally). Mirrors
    /// `_before_execute`.
    pub fn before_execute(&mut self) {
        if let Some(mgr) = self.sync_manager.as_mut() {
            mgr.sync(false);
        }
    }

    /// Mutable access to the file-sync manager (for the initial forced sync).
    pub fn sync_manager_mut(
        &mut self,
    ) -> Option<&mut crate::tool_environments_file_sync::FileSyncManager> {
        self.sync_manager.as_mut()
    }

    /// Snapshot the filesystem (if persistent) then stop the sandbox. Mirrors
    /// `cleanup`.
    ///
    /// `hermes_home` is needed for the file-sync sync-back lock path; pass the
    /// same directory used to build [`store_path`].
    ///
    /// [`store_path`]: ModalEnvironment::store_path
    pub fn cleanup(&mut self, hermes_home: std::path::PathBuf) {
        if self.sandbox.is_none() {
            return;
        }

        if let Some(mgr) = self.sync_manager.as_mut() {
            log::info!("Modal: syncing files from sandbox...");
            mgr.sync_back(hermes_home);
        }

        if self.persistent {
            // Snapshot the filesystem; swallow errors (snapshot_id = None).
            let snapshot_id: Option<String> = self
                .sandbox
                .as_ref()
                .and_then(|sb| sb.snapshot_filesystem().ok())
                .filter(|s| !s.is_empty());

            if let Some(sid) = snapshot_id {
                if store_direct_snapshot(&self.store_path, &self.task_id, &sid).is_ok() {
                    let short: String = sid.chars().take(20).collect();
                    log::info!(
                        "Modal: saved filesystem snapshot {} for task {}",
                        short,
                        self.task_id
                    );
                }
            }
        }

        // terminate (best-effort), then drop state.
        if let Some(sb) = self.sandbox.as_ref() {
            let _ = sb.terminate();
        }
        self.sandbox = None;
    }
}

// ---------------------------------------------------------------------------
// base64 (avoid pulling a specific base64 crate API version into this file)
// ---------------------------------------------------------------------------

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut chunks = data.chunks_exact(3);
    for chunk in &mut chunks {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        out.push(ALPHABET[(n & 63) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn tmp_store() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let uniq = format!(
            "hermes-modal-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        p.push(uniq);
        p
    }

    #[test]
    fn direct_key_format() {
        assert_eq!(direct_snapshot_key("abc"), "direct:abc");
        assert_eq!(direct_snapshot_key("default"), "direct:default");
    }

    #[test]
    fn restore_candidate_prefers_namespaced() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("direct:t1".into(), Value::String("im-new".into()));
        m.insert("t1".into(), Value::String("im-legacy".into()));
        save_snapshots(&path, &m).unwrap();

        let (sid, legacy) = get_snapshot_restore_candidate(&path, "t1");
        assert_eq!(sid.as_deref(), Some("im-new"));
        assert!(!legacy);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn restore_candidate_legacy_fallback() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("t2".into(), Value::String("im-legacy".into()));
        save_snapshots(&path, &m).unwrap();

        let (sid, legacy) = get_snapshot_restore_candidate(&path, "t2");
        assert_eq!(sid.as_deref(), Some("im-legacy"));
        assert!(legacy);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn restore_candidate_empty_string_ignored() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("direct:t3".into(), Value::String(String::new()));
        save_snapshots(&path, &m).unwrap();
        let (sid, legacy) = get_snapshot_restore_candidate(&path, "t3");
        assert_eq!(sid, None);
        assert!(!legacy);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn restore_candidate_missing() {
        let path = tmp_store();
        let (sid, legacy) = get_snapshot_restore_candidate(&path, "nope");
        assert_eq!(sid, None);
        assert!(!legacy);
    }

    #[test]
    fn store_direct_snapshot_migrates_legacy() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("t4".into(), Value::String("im-old".into()));
        save_snapshots(&path, &m).unwrap();

        store_direct_snapshot(&path, "t4", "im-fresh").unwrap();
        let loaded = load_snapshots(&path);
        assert_eq!(
            loaded.get("direct:t4").and_then(|v| v.as_str()),
            Some("im-fresh")
        );
        assert!(loaded.get("t4").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_direct_snapshot_all_keys() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("direct:t5".into(), Value::String("im-a".into()));
        m.insert("t5".into(), Value::String("im-b".into()));
        save_snapshots(&path, &m).unwrap();

        delete_direct_snapshot(&path, "t5", None).unwrap();
        let loaded = load_snapshots(&path);
        assert!(loaded.get("direct:t5").is_none());
        assert!(loaded.get("t5").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_direct_snapshot_matching_only() {
        let path = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("direct:t6".into(), Value::String("im-a".into()));
        m.insert("t6".into(), Value::String("im-b".into()));
        save_snapshots(&path, &m).unwrap();

        // Only delete entries equal to "im-a".
        delete_direct_snapshot(&path, "t6", Some("im-a")).unwrap();
        let loaded = load_snapshots(&path);
        assert!(loaded.get("direct:t6").is_none());
        assert_eq!(loaded.get("t6").and_then(|v| v.as_str()), Some("im-b"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_direct_snapshot_no_match_no_write() {
        let path = tmp_store();
        // Store file does not exist; delete with no matches must not create it.
        delete_direct_snapshot(&path, "ghost", Some("im-x")).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn resolve_image_from_id() {
        assert_eq!(
            resolve_modal_image("im-12345"),
            ModalImage::FromId("im-12345".into())
        );
    }

    #[test]
    fn resolve_image_ubuntu_adds_python() {
        let img = resolve_modal_image("ubuntu:22.04");
        match img {
            ModalImage::FromRegistry {
                reference,
                setup_dockerfile_commands,
            } => {
                assert_eq!(reference, "ubuntu:22.04");
                assert_eq!(setup_dockerfile_commands.len(), 2);
                assert_eq!(setup_dockerfile_commands[0], ADD_PYTHON_COMMAND);
                assert_eq!(setup_dockerfile_commands[1], ENSUREPIP_COMMAND);
            }
            _ => panic!("expected FromRegistry"),
        }
    }

    #[test]
    fn resolve_image_debian_case_insensitive() {
        let img = resolve_modal_image("DEBIAN:bookworm");
        match img {
            ModalImage::FromRegistry {
                setup_dockerfile_commands,
                ..
            } => assert_eq!(setup_dockerfile_commands.len(), 2),
            _ => panic!("expected FromRegistry"),
        }
    }

    #[test]
    fn resolve_image_other_registry_no_python_step() {
        let img = resolve_modal_image("python:3.12-slim");
        match img {
            ModalImage::FromRegistry {
                reference,
                setup_dockerfile_commands,
            } => {
                assert_eq!(reference, "python:3.12-slim");
                assert_eq!(setup_dockerfile_commands.len(), 1);
                assert_eq!(setup_dockerfile_commands[0], ENSUREPIP_COMMAND);
            }
            _ => panic!("expected FromRegistry"),
        }
    }

    #[test]
    fn upload_command_quotes() {
        assert_eq!(
            upload_command("/root/.hermes/a b.txt"),
            "mkdir -p /root/.hermes && base64 -d > '/root/.hermes/a b.txt'"
        );
    }

    #[test]
    fn bulk_upload_command_shape() {
        let files = vec![
            ("/h/a".to_string(), "/root/.hermes/x/a".to_string()),
            ("/h/b".to_string(), "/root/.hermes/y/b".to_string()),
        ];
        let cmd = bulk_upload_command(&files);
        assert!(cmd.starts_with("mkdir -p /root/.hermes/x /root/.hermes/y"));
        assert!(cmd.ends_with("&& base64 -d | tar xzf - -C /"));
    }

    #[test]
    fn run_bash_args_login_toggle() {
        assert_eq!(
            run_bash_args("echo hi", false),
            vec!["bash", "-c", "echo hi"]
        );
        assert_eq!(
            run_bash_args("echo hi", true),
            vec!["bash", "-l", "-c", "echo hi"]
        );
    }

    #[test]
    fn merge_streams_rules() {
        assert_eq!(merge_stdout_stderr(b"out", b""), "out");
        assert_eq!(merge_stdout_stderr(b"", b"err"), "err");
        assert_eq!(merge_stdout_stderr(b"out", b"err"), "out\nerr");
        assert_eq!(merge_stdout_stderr(b"", b""), "");
    }

    #[test]
    fn merge_streams_lossy_utf8() {
        // Invalid UTF-8 bytes become the replacement char (errors="replace").
        let s = merge_stdout_stderr(&[0xff, 0xfe], b"");
        assert_eq!(s, "\u{fffd}\u{fffd}");
    }

    #[test]
    fn base64_matches_known() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn build_tar_gz_roundtrips() {
        // Write a host file, archive it, decompress + untar, verify member name
        // and content (arcname = remote.lstrip("/")).
        let dir = std::env::temp_dir().join(format!("hermes-modal-tar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let host = dir.join("payload.txt");
        std::fs::write(&host, b"hello tar").unwrap();

        let files = vec![(
            host.to_string_lossy().into_owned(),
            "/root/.hermes/payload.txt".to_string(),
        )];
        let gz = build_bulk_tar_gz(&files).unwrap();

        let dec = flate2::read::GzDecoder::new(&gz[..]);
        let mut ar = tar::Archive::new(dec);
        let mut found = false;
        for entry in ar.entries().unwrap() {
            let mut e = entry.unwrap();
            let path = e.path().unwrap().to_string_lossy().into_owned();
            assert_eq!(path, "root/.hermes/payload.txt");
            let mut content = String::new();
            use std::io::Read;
            e.read_to_string(&mut content).unwrap();
            assert_eq!(content, "hello tar");
            found = true;
        }
        assert!(found);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Sandbox stub for exec-driven paths -----------------------------

    #[derive(Default)]
    struct StubSandbox {
        calls: StdMutex<Vec<(Vec<String>, Option<Vec<u8>>, Option<i32>)>>,
        exec_result: ExecOutput,
        snapshot_id: String,
        terminated: StdMutex<bool>,
    }

    impl ModalSandbox for StubSandbox {
        fn exec(
            &self,
            args: &[String],
            stdin: Option<&[u8]>,
            timeout: Option<i32>,
        ) -> Result<ExecOutput, ModalError> {
            self.calls
                .lock()
                .unwrap()
                .push((args.to_vec(), stdin.map(|s| s.to_vec()), timeout));
            Ok(self.exec_result.clone())
        }
        fn snapshot_filesystem(&self) -> Result<String, ModalError> {
            Ok(self.snapshot_id.clone())
        }
        fn terminate(&self) -> Result<(), ModalError> {
            *self.terminated.lock().unwrap() = true;
            Ok(())
        }
    }

    fn make_env(sandbox: Arc<StubSandbox>, persistent: bool) -> (ModalEnvironment, std::path::PathBuf) {
        let store = tmp_store();
        let sync = crate::tool_environments_file_sync::FileSyncManager::new(
            crate::tool_environments_file_sync::Transport {
                upload_fn: Box::new(|_h, _r| Ok(())),
                bulk_upload_fn: None,
                bulk_download_fn: None,
                delete_fn: Box::new(|_p| Ok(())),
                get_files_fn: Box::new(Vec::new),
            },
        );
        let plan = ModalEnvironment::plan_creation(&store, "ubuntu:22.04", persistent, "task");
        let env = ModalEnvironment::from_sandbox(
            store.clone(),
            sandbox,
            "/root".into(),
            60,
            persistent,
            "task".into(),
            &plan,
            true,
            sync,
        );
        (env, store)
    }

    #[test]
    fn run_bash_builds_argv_and_merges() {
        let sb = Arc::new(StubSandbox {
            exec_result: ExecOutput {
                stdout: b"hi".to_vec(),
                stderr: b"warn".to_vec(),
                exit_code: 3,
            },
            ..Default::default()
        });
        let (env, store) = make_env(sb.clone(), false);
        let (out, code) = env.run_bash("echo hi", true, 45).unwrap();
        assert_eq!(out, "hi\nwarn");
        assert_eq!(code, 3);
        let calls = sb.calls.lock().unwrap();
        assert_eq!(calls[0].0, vec!["bash", "-l", "-c", "echo hi"]);
        assert_eq!(calls[0].2, Some(45));
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn modal_delete_quotes_and_execs() {
        let sb = Arc::new(StubSandbox::default());
        let (env, store) = make_env(sb.clone(), false);
        env.modal_delete(&["/root/.hermes/a b".to_string()]).unwrap();
        let calls = sb.calls.lock().unwrap();
        assert_eq!(calls[0].0[0], "bash");
        assert_eq!(calls[0].0[1], "-c");
        assert_eq!(calls[0].0[2], "rm -f '/root/.hermes/a b'");
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn bulk_upload_failure_surfaces() {
        let sb = Arc::new(StubSandbox {
            exec_result: ExecOutput {
                stdout: vec![],
                stderr: b"disk full".to_vec(),
                exit_code: 1,
            },
            ..Default::default()
        });
        let (env, store) = make_env(sb, false);
        let dir = std::env::temp_dir().join(format!("hermes-bulk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let host = dir.join("f.txt");
        std::fs::write(&host, b"x").unwrap();
        let files = vec![(
            host.to_string_lossy().into_owned(),
            "/root/.hermes/f.txt".to_string(),
        )];
        let err = env.modal_bulk_upload(&files).unwrap_err();
        assert!(err.to_string().contains("exit 1"));
        assert!(err.to_string().contains("disk full"));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn cleanup_snapshots_and_terminates() {
        let sb = Arc::new(StubSandbox {
            snapshot_id: "im-snap-result".to_string(),
            ..Default::default()
        });
        let (mut env, store) = make_env(sb.clone(), true);
        // sync_manager has no prior push state -> sync_back is a no-op.
        env.cleanup(std::env::temp_dir());
        assert!(*sb.terminated.lock().unwrap());
        let loaded = load_snapshots(&store);
        assert_eq!(
            loaded.get("direct:task").and_then(|v| v.as_str()),
            Some("im-snap-result")
        );
        assert!(env.sandbox().is_none());
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn cleanup_non_persistent_no_snapshot() {
        let sb = Arc::new(StubSandbox {
            snapshot_id: "im-should-not-store".to_string(),
            ..Default::default()
        });
        let (mut env, store) = make_env(sb.clone(), false);
        env.cleanup(std::env::temp_dir());
        assert!(*sb.terminated.lock().unwrap());
        // Non-persistent: nothing written to the store.
        let loaded = load_snapshots(&store);
        assert!(loaded.get("direct:task").is_none());
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn cleanup_idempotent_after_first() {
        let sb = Arc::new(StubSandbox::default());
        let (mut env, store) = make_env(sb, false);
        env.cleanup(std::env::temp_dir());
        // Second call returns early (sandbox is None) without panicking.
        env.cleanup(std::env::temp_dir());
        let _ = std::fs::remove_file(&store);
    }

    #[test]
    fn from_sandbox_migrates_legacy_key() {
        let store = tmp_store();
        let mut m = serde_json::Map::new();
        m.insert("mtask".into(), Value::String("im-legacy".into()));
        save_snapshots(&store, &m).unwrap();

        let plan = ModalEnvironment::plan_creation(&store, "ubuntu:22.04", true, "mtask");
        assert_eq!(plan.restored_snapshot_id.as_deref(), Some("im-legacy"));
        assert!(plan.restored_from_legacy_key);
        assert_eq!(plan.target_image, ModalImage::FromId("im-legacy".into()));

        let sync = crate::tool_environments_file_sync::FileSyncManager::new(
            crate::tool_environments_file_sync::Transport {
                upload_fn: Box::new(|_h, _r| Ok(())),
                bulk_upload_fn: None,
                bulk_download_fn: None,
                delete_fn: Box::new(|_p| Ok(())),
                get_files_fn: Box::new(Vec::new),
            },
        );
        let _env = ModalEnvironment::from_sandbox(
            store.clone(),
            Arc::new(StubSandbox::default()),
            "/root".into(),
            60,
            true,
            "mtask".into(),
            &plan,
            true, // created against restored image
            sync,
        );
        // Legacy key migrated to namespaced.
        let loaded = load_snapshots(&store);
        assert_eq!(
            loaded.get("direct:mtask").and_then(|v| v.as_str()),
            Some("im-legacy")
        );
        assert!(loaded.get("mtask").is_none());
        let _ = std::fs::remove_file(&store);
    }
}
