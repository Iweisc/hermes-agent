//! Singularity/Apptainer persistent container environment.
//!
//! Security-hardened with `--containall`, `--no-home`, capability dropping.
//! Supports configurable resource limits and optional filesystem persistence
//! via writable overlay directories that survive across sessions.
//!
//! Native Rust port of `tools/environments/singularity.py`. The port preserves
//! the command-construction and JSON snapshot-store behavior of the original.
//! The spawn-per-call session machinery in the Python `BaseEnvironment` is not
//! reproduced verbatim here; instead the building blocks (instance lifecycle,
//! bash command construction, overlay/snapshot persistence) are exposed as a
//! `SingularityEnvironment` struct with the same observable side effects.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// hermes_home resolution
// ---------------------------------------------------------------------------

/// Resolve the Hermes home directory (default: `~/.hermes`).
///
/// Mirrors `hermes_constants.get_hermes_home`. The Rust port of that module
/// (`hermes-core/src/mod_hermes_constants.rs`) keeps `get_hermes_home` behind a
/// private `mod`, so it is reimplemented here with identical behavior to avoid
/// editing shared crate files. If `mod_hermes_constants` is later re-exported,
/// this single call-site can be swapped to delegate to it.
fn hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    home.join(".hermes")
}

/// Path to the snapshot store JSON file (`{HERMES_HOME}/singularity_snapshots.json`).
pub fn snapshot_store_path() -> PathBuf {
    hermes_home().join("singularity_snapshots.json")
}

// ---------------------------------------------------------------------------
// Shared JSON store helpers (mirrors tools.environments.base)
// ---------------------------------------------------------------------------

/// Load a JSON file as an object map, returning an empty map on any error.
pub fn load_json_store(path: &Path) -> Map<String, Value> {
    if path.exists() {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) {
                return map;
            }
        }
    }
    Map::new()
}

/// Write *data* as pretty-printed (indent=2) JSON to *path*.
pub fn save_json_store(path: &Path, data: &Map<String, Value>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let value = Value::Object(data.clone());
    let text = serde_json::to_string_pretty(&value)?;
    std::fs::write(path, text)
}

fn load_snapshots() -> Map<String, Value> {
    load_json_store(&snapshot_store_path())
}

fn save_snapshots(data: &Map<String, Value>) -> std::io::Result<()> {
    save_json_store(&snapshot_store_path(), data)
}

// ---------------------------------------------------------------------------
// Executable discovery / preflight
// ---------------------------------------------------------------------------

/// Locate an executable on PATH (equivalent of `shutil.which`).
fn which(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && (meta.permissions().mode() & 0o111 != 0),
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Locate the apptainer or singularity CLI binary.
///
/// Returns the bare command name (`"apptainer"` or `"singularity"`) — matching
/// the Python implementation which returns the name, not the full path.
pub fn find_singularity_executable() -> Result<String, String> {
    if which("apptainer").is_some() {
        return Ok("apptainer".to_string());
    }
    if which("singularity").is_some() {
        return Ok("singularity".to_string());
    }
    Err("Neither 'apptainer' nor 'singularity' was found in PATH. \
         Install Apptainer (https://apptainer.org/docs/admin/main/installation.html) \
         or Singularity and ensure the CLI is available."
        .to_string())
}

/// Preflight check: resolve the executable and verify it responds to `version`.
pub fn ensure_singularity_available() -> Result<String, String> {
    let exe = find_singularity_executable()?;

    let output = run_with_timeout(
        Command::new(&exe).arg("version"),
        Duration::from_secs(10),
    );

    match output {
        Ok(out) => {
            if !out.status.success() {
                let code = out
                    .status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let mut stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                if stderr.len() > 200 {
                    stderr.truncate(200);
                }
                return Err(format!(
                    "'{exe} version' failed (exit code {code}): {stderr}"
                ));
            }
            Ok(exe)
        }
        Err(RunError::NotFound) => Err(format!(
            "Singularity backend selected but '{exe}' could not be executed."
        )),
        Err(RunError::Timeout) => Err(format!("'{exe} version' timed out.")),
        Err(RunError::Other(e)) => Err(format!("'{exe} version' failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Subprocess execution with timeout
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum RunError {
    NotFound,
    Timeout,
    Other(String),
}

/// Run a command, capturing stdout/stderr, killing it if it exceeds *timeout*.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<std::process::Output, RunError> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.stdin(Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(RunError::NotFound),
        Err(e) => return Err(RunError::Other(e.to_string())),
    };

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                return child
                    .wait_with_output()
                    .map_err(|e| RunError::Other(e.to_string()));
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RunError::Timeout);
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(RunError::Other(e.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch / cache directory resolution
// ---------------------------------------------------------------------------

fn get_sandbox_dir() -> std::io::Result<PathBuf> {
    let p = if let Some(custom) = std::env::var_os("TERMINAL_SANDBOX_DIR") {
        PathBuf::from(custom)
    } else {
        hermes_home().join("sandboxes")
    };
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Resolve the scratch directory used for sandboxes / overlays / SIF cache.
pub fn get_scratch_dir() -> std::io::Result<PathBuf> {
    if let Some(custom) = std::env::var_os("TERMINAL_SCRATCH_DIR") {
        let scratch_path = PathBuf::from(custom);
        std::fs::create_dir_all(&scratch_path)?;
        return Ok(scratch_path);
    }

    let sandbox = get_sandbox_dir()?.join("singularity");

    let scratch = Path::new("/scratch");
    if scratch.exists() && is_writable(scratch) {
        let user = std::env::var("USER").unwrap_or_else(|_| "hermes".to_string());
        let user_scratch = scratch.join(user).join("hermes-agent");
        std::fs::create_dir_all(&user_scratch)?;
        log::info!("Using /scratch for sandboxes: {}", user_scratch.display());
        return Ok(user_scratch);
    }

    std::fs::create_dir_all(&sandbox)?;
    Ok(sandbox)
}

#[cfg(unix)]
fn is_writable(path: &Path) -> bool {
    use std::ffi::CString;
    let Some(s) = path.to_str() else { return false };
    let Ok(cstr) = CString::new(s) else { return false };
    // libc::W_OK == 2
    unsafe { libc::access(cstr.as_ptr(), libc::W_OK) == 0 }
}

#[cfg(not(unix))]
fn is_writable(path: &Path) -> bool {
    // Best-effort fallback on non-unix: writable if it exists.
    path.exists()
}

/// Resolve the apptainer cache directory.
pub fn get_apptainer_cache_dir() -> std::io::Result<PathBuf> {
    if let Some(cache_dir) = std::env::var_os("APPTAINER_CACHEDIR") {
        let cache_path = PathBuf::from(cache_dir);
        std::fs::create_dir_all(&cache_path)?;
        return Ok(cache_path);
    }
    let scratch = get_scratch_dir()?;
    let cache_path = scratch.join(".apptainer");
    std::fs::create_dir_all(&cache_path)?;
    Ok(cache_path)
}

// ---------------------------------------------------------------------------
// SIF build
// ---------------------------------------------------------------------------

/// Global lock guarding one-time SIF builds (mirrors the Python module lock).
static SIF_BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Resolve *image* to a buildable/usable reference, building a SIF cache file
/// for `docker://` images when possible.
///
/// Returns the path/URL string that should be passed to apptainer.
pub fn get_or_build_sif(image: &str, executable: &str) -> String {
    if image.ends_with(".sif") && Path::new(image).exists() {
        return image.to_string();
    }
    if !image.starts_with("docker://") {
        return image.to_string();
    }

    let image_name = image
        .replacen("docker://", "", 1)
        .replace('/', "-")
        .replace(':', "-");

    let cache_dir = match get_apptainer_cache_dir() {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Could not resolve apptainer cache dir: {e}; using docker:// URL");
            return image.to_string();
        }
    };
    let sif_path = cache_dir.join(format!("{image_name}.sif"));

    if sif_path.exists() {
        return sif_path.to_string_lossy().into_owned();
    }

    let _guard = SIF_BUILD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if sif_path.exists() {
        return sif_path.to_string_lossy().into_owned();
    }

    log::info!("Building SIF image (one-time setup)...");
    log::info!("  Source: {image}");
    log::info!("  Target: {}", sif_path.display());

    let tmp_dir = cache_dir.join("tmp");
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        log::warn!("SIF build error: {e}, falling back to docker:// URL");
        return image.to_string();
    }

    let mut cmd = Command::new(executable);
    cmd.arg("build")
        .arg(&sif_path)
        .arg(image)
        .env("APPTAINER_TMPDIR", &tmp_dir)
        .env("APPTAINER_CACHEDIR", &cache_dir);

    match run_with_timeout(&mut cmd, Duration::from_secs(600)) {
        Ok(out) => {
            if !out.status.success() {
                let mut stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                if stderr.len() > 500 {
                    stderr.truncate(500);
                }
                log::warn!("SIF build failed, falling back to docker:// URL");
                log::warn!("  Error: {stderr}");
                return image.to_string();
            }
            log::info!("SIF image built successfully");
            sif_path.to_string_lossy().into_owned()
        }
        Err(RunError::Timeout) => {
            log::warn!("SIF build timed out, falling back to docker:// URL");
            if sif_path.exists() {
                let _ = std::fs::remove_file(&sif_path);
            }
            image.to_string()
        }
        Err(e) => {
            log::warn!("SIF build error: {e:?}, falling back to docker:// URL");
            image.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Credential / skills mounts
// ---------------------------------------------------------------------------

/// A host->container bind-mount entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    pub host_path: String,
    pub container_path: String,
}

/// Provider trait for credential/skills bind-mounts, so callers that have a
/// ported `credential_files` module can plug it in. The default implementation
/// returns no mounts (matching the Python `except` fallback that logs and
/// continues).
pub trait MountProvider {
    fn credential_file_mounts(&self) -> Vec<MountEntry> {
        Vec::new()
    }
    fn skills_directory_mounts(&self) -> Vec<MountEntry> {
        Vec::new()
    }
}

/// A no-op mount provider (equivalent to the Python failure fallback).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoMounts;

impl MountProvider for NoMounts {}

// ---------------------------------------------------------------------------
// SingularityEnvironment
// ---------------------------------------------------------------------------

/// Hardened Singularity/Apptainer container with resource limits and persistence.
///
/// Spawn-per-call: every command spawns a fresh `apptainer exec ... bash -c`
/// process. Session env-var preservation across calls is the responsibility of
/// the higher-level base environment; this struct owns the instance lifecycle
/// and command construction.
pub struct SingularityEnvironment {
    pub executable: String,
    pub image: String,
    pub instance_id: String,
    pub cwd: String,
    pub timeout: u64,
    instance_started: bool,
    persistent: bool,
    task_id: String,
    overlay_dir: Option<PathBuf>,
    cpu: f64,
    memory: i64,
}

/// Generate a random 12-char hex instance id, matching `uuid4().hex[:12]`.
fn random_instance_id() -> String {
    // 6 random bytes -> 12 hex chars.
    let mut bytes = [0u8; 6];
    fill_random(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("hermes_{hex}")
}

fn fill_random(buf: &mut [u8]) {
    // Use a simple OS-entropy source; on unix read from /dev/urandom, else
    // derive from time + address entropy as a best-effort fallback.
    #[cfg(unix)]
    {
        if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
            use std::io::Read;
            if f.read_exact(buf).is_ok() {
                return;
            }
        }
    }
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e3779b97f4a7c15)
        ^ (buf.as_ptr() as u64);
    for b in buf.iter_mut() {
        // xorshift64
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *b = (seed & 0xff) as u8;
    }
}

impl SingularityEnvironment {
    /// Construct and start a new Singularity environment.
    ///
    /// Performs the same setup the Python `__init__` does: preflight, image
    /// resolution, overlay directory creation (when persistent), and instance
    /// start. Returns an error string instead of raising.
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: MountProvider>(
        image: &str,
        cwd: &str,
        timeout: u64,
        cpu: f64,
        memory: i64,
        _disk: i64,
        persistent_filesystem: bool,
        task_id: &str,
        mounts: &M,
    ) -> Result<Self, String> {
        let executable = ensure_singularity_available()?;
        let resolved_image = get_or_build_sif(image, &executable);
        let instance_id = random_instance_id();

        let mut overlay_dir: Option<PathBuf> = None;
        if persistent_filesystem {
            let scratch = get_scratch_dir().map_err(|e| e.to_string())?;
            let overlay_base = scratch.join("hermes-overlays");
            std::fs::create_dir_all(&overlay_base).map_err(|e| e.to_string())?;
            let dir = overlay_base.join(format!("overlay-{task_id}"));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            overlay_dir = Some(dir);
        }

        let mut env = SingularityEnvironment {
            executable,
            image: resolved_image,
            instance_id,
            cwd: cwd.to_string(),
            timeout,
            instance_started: false,
            persistent: persistent_filesystem,
            task_id: task_id.to_string(),
            overlay_dir,
            cpu,
            memory,
        };

        env.start_instance(mounts)?;
        Ok(env)
    }

    /// Build the `instance start` argument vector (without the executable).
    ///
    /// Exposed for testing and reuse; mirrors the Python `_start_instance`
    /// command construction exactly.
    pub fn build_start_args<M: MountProvider>(&self, mounts: &M) -> Vec<String> {
        let mut cmd: Vec<String> = vec!["instance".into(), "start".into()];
        cmd.push("--containall".into());
        cmd.push("--no-home".into());

        if self.persistent {
            if let Some(overlay) = &self.overlay_dir {
                cmd.push("--overlay".into());
                cmd.push(overlay.to_string_lossy().into_owned());
            } else {
                cmd.push("--writable-tmpfs".into());
            }
        } else {
            cmd.push("--writable-tmpfs".into());
        }

        for m in mounts.credential_file_mounts() {
            cmd.push("--bind".into());
            cmd.push(format!("{}:{}:ro", m.host_path, m.container_path));
        }
        for m in mounts.skills_directory_mounts() {
            cmd.push("--bind".into());
            cmd.push(format!("{}:{}:ro", m.host_path, m.container_path));
        }

        if self.memory > 0 {
            cmd.push("--memory".into());
            cmd.push(format!("{}M", self.memory));
        }
        if self.cpu > 0.0 {
            cmd.push("--cpus".into());
            cmd.push(format_cpus(self.cpu));
        }

        cmd.push(self.image.clone());
        cmd.push(self.instance_id.clone());
        cmd
    }

    fn start_instance<M: MountProvider>(&mut self, mounts: &M) -> Result<(), String> {
        let args = self.build_start_args(mounts);
        let mut cmd = Command::new(&self.executable);
        cmd.args(&args);

        match run_with_timeout(&mut cmd, Duration::from_secs(120)) {
            Ok(out) => {
                if !out.status.success() {
                    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                    return Err(format!("Failed to start instance: {stderr}"));
                }
                self.instance_started = true;
                log::info!(
                    "Singularity instance {} started (persistent={})",
                    self.instance_id,
                    self.persistent
                );
                Ok(())
            }
            Err(RunError::Timeout) => Err("Instance start timed out".to_string()),
            Err(RunError::NotFound) => {
                Err(format!("'{}' could not be executed.", self.executable))
            }
            Err(RunError::Other(e)) => Err(format!("Failed to start instance: {e}")),
        }
    }

    /// Build the `exec` argument vector for running a bash command inside the
    /// instance (without the executable). Mirrors Python `_run_bash`.
    pub fn build_bash_args(&self, cmd_string: &str, login: bool) -> Vec<String> {
        let mut cmd: Vec<String> = vec![
            "exec".into(),
            format!("instance://{}", self.instance_id),
        ];
        if login {
            cmd.push("bash".into());
            cmd.push("-l".into());
            cmd.push("-c".into());
            cmd.push(cmd_string.to_string());
        } else {
            cmd.push("bash".into());
            cmd.push("-c".into());
            cmd.push(cmd_string.to_string());
        }
        cmd
    }

    /// Spawn a bash process inside the Singularity instance.
    ///
    /// Returns the spawned child handle (with stdout piped, stderr merged into
    /// stdout, stdin piped when `stdin_data` is provided), matching the
    /// `_popen_bash` semantics of the Python base module.
    pub fn run_bash(
        &self,
        cmd_string: &str,
        login: bool,
        stdin_data: Option<&str>,
    ) -> Result<std::process::Child, String> {
        if !self.instance_started {
            return Err("Singularity instance not started".to_string());
        }
        let args = self.build_bash_args(cmd_string, login);
        let mut cmd = Command::new(&self.executable);
        cmd.args(&args);
        cmd.stdout(Stdio::piped());
        // stderr merged into stdout (subprocess.STDOUT)
        cmd.stderr(Stdio::piped());
        if stdin_data.is_some() {
            cmd.stdin(Stdio::piped());
        } else {
            cmd.stdin(Stdio::null());
        }

        let mut child = cmd.spawn().map_err(|e| e.to_string())?;

        if let Some(data) = stdin_data {
            if let Some(mut stdin) = child.stdin.take() {
                let data = data.to_string();
                std::thread::spawn(move || {
                    use std::io::Write;
                    let _ = stdin.write_all(data.as_bytes());
                    // dropping stdin closes the pipe
                });
            }
        }
        Ok(child)
    }

    /// Stop the instance. If persistent, the overlay dir survives and is
    /// recorded in the snapshot store keyed by task id.
    pub fn cleanup(&mut self) {
        if self.instance_started {
            let mut cmd = Command::new(&self.executable);
            cmd.arg("instance").arg("stop").arg(&self.instance_id);
            match run_with_timeout(&mut cmd, Duration::from_secs(30)) {
                Ok(_) => {
                    log::info!("Singularity instance {} stopped", self.instance_id);
                }
                Err(e) => {
                    log::warn!(
                        "Failed to stop Singularity instance {}: {:?}",
                        self.instance_id,
                        e
                    );
                }
            }
            self.instance_started = false;
        }

        if self.persistent {
            if let Some(overlay) = &self.overlay_dir {
                let mut snapshots = load_snapshots();
                snapshots.insert(
                    self.task_id.clone(),
                    Value::String(overlay.to_string_lossy().into_owned()),
                );
                if let Err(e) = save_snapshots(&snapshots) {
                    log::warn!("Failed to save singularity snapshot store: {e}");
                }
            }
        }
    }

    pub fn instance_started(&self) -> bool {
        self.instance_started
    }

    pub fn overlay_dir(&self) -> Option<&Path> {
        self.overlay_dir.as_deref()
    }
}

impl Drop for SingularityEnvironment {
    fn drop(&mut self) {
        // Best-effort cleanup; safe to call even if already cleaned up.
        self.cleanup();
    }
}

/// Format a cpu float the way Python's `str(float)` would for the common
/// cases used here (e.g. `2.0` -> "2.0", `1.5` -> "1.5").
fn format_cpus(cpu: f64) -> String {
    // Python passes str(self._cpu) where cpu may be float. e.g. str(2.0)=="2.0".
    if cpu.fract() == 0.0 {
        format!("{cpu:.1}")
    } else {
        // Trim, but keep natural representation.
        let s = format!("{cpu}");
        s
    }
}

// ---------------------------------------------------------------------------
// Snapshot store accessors (public convenience)
// ---------------------------------------------------------------------------

/// Read all recorded overlay snapshots as a task_id -> overlay_path map.
pub fn list_snapshots() -> BTreeMap<String, String> {
    load_snapshots()
        .into_iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeMounts {
        creds: Vec<MountEntry>,
        skills: Vec<MountEntry>,
    }
    impl MountProvider for FakeMounts {
        fn credential_file_mounts(&self) -> Vec<MountEntry> {
            self.creds.clone()
        }
        fn skills_directory_mounts(&self) -> Vec<MountEntry> {
            self.skills.clone()
        }
    }

    fn base_env(persistent: bool, overlay: Option<PathBuf>) -> SingularityEnvironment {
        SingularityEnvironment {
            executable: "apptainer".into(),
            image: "/cache/img.sif".into(),
            instance_id: "hermes_abc123def456".into(),
            cwd: "~".into(),
            timeout: 60,
            instance_started: false,
            persistent,
            task_id: "default".into(),
            overlay_dir: overlay,
            cpu: 0.0,
            memory: 0,
        }
    }

    #[test]
    fn start_args_non_persistent_uses_writable_tmpfs() {
        let env = base_env(false, None);
        let args = env.build_start_args(&NoMounts);
        assert_eq!(args[0], "instance");
        assert_eq!(args[1], "start");
        assert!(args.contains(&"--containall".to_string()));
        assert!(args.contains(&"--no-home".to_string()));
        assert!(args.contains(&"--writable-tmpfs".to_string()));
        assert!(!args.contains(&"--overlay".to_string()));
        // image and instance id are last two
        assert_eq!(args[args.len() - 2], "/cache/img.sif");
        assert_eq!(args[args.len() - 1], "hermes_abc123def456");
    }

    #[test]
    fn start_args_persistent_uses_overlay() {
        let env = base_env(true, Some(PathBuf::from("/scratch/ovl/overlay-t1")));
        let args = env.build_start_args(&NoMounts);
        let i = args.iter().position(|a| a == "--overlay").expect("overlay");
        assert_eq!(args[i + 1], "/scratch/ovl/overlay-t1");
        assert!(!args.contains(&"--writable-tmpfs".to_string()));
    }

    #[test]
    fn start_args_persistent_without_overlay_falls_back_to_tmpfs() {
        let env = base_env(true, None);
        let args = env.build_start_args(&NoMounts);
        assert!(args.contains(&"--writable-tmpfs".to_string()));
    }

    #[test]
    fn start_args_resource_limits() {
        let mut env = base_env(false, None);
        env.memory = 512;
        env.cpu = 2.0;
        let args = env.build_start_args(&NoMounts);
        let mi = args.iter().position(|a| a == "--memory").unwrap();
        assert_eq!(args[mi + 1], "512M");
        let ci = args.iter().position(|a| a == "--cpus").unwrap();
        assert_eq!(args[ci + 1], "2.0");
    }

    #[test]
    fn start_args_no_resource_limits_when_zero() {
        let env = base_env(false, None);
        let args = env.build_start_args(&NoMounts);
        assert!(!args.contains(&"--memory".to_string()));
        assert!(!args.contains(&"--cpus".to_string()));
    }

    #[test]
    fn start_args_mounts_are_readonly_binds() {
        let env = base_env(false, None);
        let mounts = FakeMounts {
            creds: vec![MountEntry {
                host_path: "/h/cred".into(),
                container_path: "/c/cred".into(),
            }],
            skills: vec![MountEntry {
                host_path: "/h/skills".into(),
                container_path: "/root/.hermes/skills".into(),
            }],
        };
        let args = env.build_start_args(&mounts);
        assert!(args.contains(&"/h/cred:/c/cred:ro".to_string()));
        assert!(args.contains(&"/h/skills:/root/.hermes/skills:ro".to_string()));
        // both preceded by --bind
        let bind_count = args.iter().filter(|a| *a == "--bind").count();
        assert_eq!(bind_count, 2);
    }

    #[test]
    fn bash_args_non_login() {
        let env = base_env(false, None);
        let args = env.build_bash_args("echo hi", false);
        assert_eq!(args[0], "exec");
        assert_eq!(args[1], "instance://hermes_abc123def456");
        assert_eq!(&args[2..], &["bash", "-c", "echo hi"]);
    }

    #[test]
    fn bash_args_login() {
        let env = base_env(false, None);
        let args = env.build_bash_args("echo hi", true);
        assert_eq!(&args[2..], &["bash", "-l", "-c", "echo hi"]);
    }

    #[test]
    fn get_or_build_sif_passthrough_non_docker() {
        // Plain image name that is not docker:// and not an existing .sif file.
        assert_eq!(get_or_build_sif("ubuntu:22.04", "apptainer"), "ubuntu:22.04");
    }

    #[test]
    fn get_or_build_sif_nonexistent_sif_is_passthrough() {
        // Ends with .sif but does not exist -> not docker:// either -> passthrough.
        let p = "/definitely/not/here.sif";
        assert_eq!(get_or_build_sif(p, "apptainer"), p);
    }

    #[test]
    fn json_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "sing_test_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("snap.json");
        let _ = std::fs::remove_file(&path);

        // missing -> empty
        assert!(load_json_store(&path).is_empty());

        let mut m = Map::new();
        m.insert("t1".into(), Value::String("/over/t1".into()));
        save_json_store(&path, &m).unwrap();

        let loaded = load_json_store(&path);
        assert_eq!(loaded.get("t1").unwrap().as_str().unwrap(), "/over/t1");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn json_store_corrupt_returns_empty() {
        let dir = std::env::temp_dir().join(format!(
            "sing_test_corrupt_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.json");
        std::fs::write(&path, "{not valid json").unwrap();
        assert!(load_json_store(&path).is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn random_instance_id_format() {
        let id = random_instance_id();
        assert!(id.starts_with("hermes_"));
        let hex = &id["hermes_".len()..];
        assert_eq!(hex.len(), 12);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn format_cpus_integers_and_fractions() {
        assert_eq!(format_cpus(2.0), "2.0");
        assert_eq!(format_cpus(1.5), "1.5");
    }

    #[test]
    fn scratch_dir_respects_custom_env() {
        let dir = std::env::temp_dir().join(format!(
            "sing_scratch_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        unsafe {
            std::env::set_var("TERMINAL_SCRATCH_DIR", &dir);
        }
        let resolved = get_scratch_dir().unwrap();
        assert_eq!(resolved, dir);
        assert!(dir.exists());
        unsafe {
            std::env::remove_var("TERMINAL_SCRATCH_DIR");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
