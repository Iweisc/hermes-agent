//! Docker execution environment for sandboxed command execution.
//!
//! Faithful native Rust port of `tools/environments/docker.py`.
//!
//! Security hardened (cap-drop ALL, no-new-privileges, PID limits),
//! configurable resource limits (CPU, memory, disk), and optional filesystem
//! persistence via bind mounts.
//!
//! # Port notes
//!
//! The Python module is mostly *argument construction*: it assembles the long
//! `docker run -d ...` and `docker exec ...` argument vectors from the
//! configuration, applying security flags, resource limits, tmpfs/bind-mount
//! choices, credential mounts and environment-variable forwarding. That pure
//! logic is reproduced exactly here and is the part most worth unit-testing.
//!
//! The thin imperative shell around it (probing `docker version`,
//! `docker info`, launching the container, spawning `docker exec` for each
//! command, and the fire-and-forget cleanup) is reproduced with
//! [`std::process::Command`]. Network/daemon access is kept behind the same
//! preflight (`ensure_docker_available`) used by the Python backend.
//!
//! Cross-references:
//! - `crate::tool_environments_base` (hermes-core) for `get_sandbox_dir`.
//! - `crate::tool_env_passthrough` (hermes-core) for `get_all_passthrough`
//!   and the Hermes provider-credential blocklist.
//!
//! Some Python dependencies are not yet ported to native Rust
//! (`tools.credential_files.*`, `hermes_cli.config.load_env`). They are
//! modelled here as injectable hooks / overridable parameters so this module
//! does not block on them, matching the "fail-soft" behaviour of the Python
//! source (a failed import simply contributed nothing).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Common Docker Desktop install paths checked when `docker` is not in PATH.
/// macOS Intel: `/usr/local/bin`, macOS Apple Silicon (Homebrew):
/// `/opt/homebrew/bin`, Docker Desktop app bundle:
/// `/Applications/Docker.app/Contents/Resources/bin`.
pub const DOCKER_SEARCH_PATHS: &[&str] = &[
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
];

/// Security flags applied to every container. Mirrors `_BASE_SECURITY_ARGS`.
///
/// The container itself is the security boundary (isolated from host). We drop
/// all capabilities then add back the minimum needed (DAC_OVERRIDE, CHOWN,
/// FOWNER), block privilege escalation, limit PIDs, and size-limit the scratch
/// tmpfs mounts.
pub const BASE_SECURITY_ARGS: &[&str] = &[
    "--cap-drop",
    "ALL",
    "--cap-add",
    "DAC_OVERRIDE",
    "--cap-add",
    "CHOWN",
    "--cap-add",
    "FOWNER",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "256",
    "--tmpfs",
    "/tmp:rw,nosuid,size=512m",
    "--tmpfs",
    "/var/tmp:rw,noexec,nosuid,size=256m",
    "--tmpfs",
    "/run:rw,noexec,nosuid,size=64m",
];

/// Extra caps needed when the container starts as root and an entrypoint must
/// drop privileges via gosu/su. Mirrors `_GOSU_CAP_ARGS`.
pub const GOSU_CAP_ARGS: &[&str] = &["--cap-add", "SETUID", "--cap-add", "SETGID"];

fn env_var_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap())
}

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

/// Return a deduplicated list of valid environment variable names. Faithful
/// port of `_normalize_forward_env_names`.
///
/// Invalid / empty / non-name entries are dropped (the Python version logs a
/// warning for each; here we just skip them). Order is preserved.
pub fn normalize_forward_env_names(forward_env: Option<&[String]>) -> Vec<String> {
    let mut normalized: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    let items: &[String] = forward_env.unwrap_or(&[]);
    for item in items {
        let key = item.trim();
        if key.is_empty() {
            continue;
        }
        if !env_var_name_re().is_match(key) {
            log::warn!("Ignoring invalid docker_forward_env entry: {item:?}");
            continue;
        }
        if seen.contains(key) {
            continue;
        }
        seen.insert(key.to_string());
        normalized.push(key.to_string());
    }

    normalized
}

/// A single docker_env value as it can appear in config: a string or a simple
/// scalar (int/float/bool). Complex values are rejected during normalisation.
#[derive(Debug, Clone)]
pub enum EnvValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// Any complex (list/dict/null) value — rejected by [`normalize_env_dict`].
    Complex,
}

impl EnvValue {
    /// Coerce simple scalar types to string the way Python's `str()` would for
    /// the accepted scalar types. Returns `None` for [`EnvValue::Complex`].
    fn coerce(&self) -> Option<String> {
        match self {
            EnvValue::Str(s) => Some(s.clone()),
            EnvValue::Int(i) => Some(i.to_string()),
            // Python bool is a subclass of int but `str(True)` == "True"; the
            // isinstance order in the source checks (int, float, bool) so a
            // bool hits the int branch — but str() still yields "True"/"False".
            EnvValue::Bool(b) => Some(if *b { "True" } else { "False" }.to_string()),
            EnvValue::Float(f) => Some(python_str_float(*f)),
            EnvValue::Complex => None,
        }
    }
}

/// Mimic Python `str(float)` reasonably for the common cases (whole numbers
/// render with a trailing `.0`).
fn python_str_float(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Validate and normalise a docker_env map to `{str: str}`. Faithful port of
/// `_normalize_env_dict`. Entries with invalid names or complex values are
/// dropped.
pub fn normalize_env_dict(env: Option<&BTreeMap<String, EnvValue>>) -> BTreeMap<String, String> {
    let mut normalized: BTreeMap<String, String> = BTreeMap::new();
    let Some(env) = env else {
        return normalized;
    };
    if env.is_empty() {
        return normalized;
    }

    for (key, value) in env {
        let trimmed = key.trim();
        if !env_var_name_re().is_match(trimmed) {
            log::warn!("Ignoring invalid docker_env key: {key:?}");
            continue;
        }
        match value.coerce() {
            Some(v) => {
                normalized.insert(trimmed.to_string(), v);
            }
            None => {
                log::warn!("Ignoring non-string docker_env value for {key:?}: <complex>");
            }
        }
    }

    normalized
}

// ---------------------------------------------------------------------------
// Docker binary resolution
// ---------------------------------------------------------------------------

static DOCKER_EXECUTABLE: OnceLock<Option<String>> = OnceLock::new();

fn is_executable_file(path: &str) -> bool {
    let p = Path::new(path);
    if !p.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::metadata(p) {
            Ok(m) => m.permissions().mode() & 0o111 != 0,
            Err(_) => false,
        }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Search `PATH` for `name`, returning the first executable match (mirrors
/// `shutil.which`).
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        let candidate_str = candidate.to_string_lossy().to_string();
        if is_executable_file(&candidate_str) {
            return Some(candidate_str);
        }
    }
    None
}

/// Locate the docker (or podman) CLI binary. Faithful port of `find_docker`.
///
/// Resolution order:
/// 1. `HERMES_DOCKER_BINARY` env var — explicit override (e.g. `/usr/bin/podman`)
/// 2. `docker` on PATH
/// 3. `podman` on PATH (drop-in compatible)
/// 4. Well-known macOS Docker Desktop install locations
///
/// Returns the absolute path, or `None` if neither runtime can be found. The
/// result is cached process-wide on first success (matching the Python module
/// global `_docker_executable`). When nothing is found the negative result is
/// *not* cached, so a later install can still be picked up.
pub fn find_docker() -> Option<String> {
    if let Some(cached) = DOCKER_EXECUTABLE.get() {
        if let Some(v) = cached {
            return Some(v.clone());
        }
    }

    let resolved = resolve_docker_uncached();
    if let Some(ref found) = resolved {
        // Only cache a positive result. `set` is a no-op if already set.
        let _ = DOCKER_EXECUTABLE.set(Some(found.clone()));
    }
    resolved
}

fn resolve_docker_uncached() -> Option<String> {
    // 1. Explicit override via env var.
    if let Ok(override_path) = std::env::var("HERMES_DOCKER_BINARY") {
        if !override_path.is_empty() && is_executable_file(&override_path) {
            log::info!("Using HERMES_DOCKER_BINARY override: {override_path}");
            return Some(override_path);
        }
    }

    // 2. docker on PATH.
    if let Some(found) = which("docker") {
        return Some(found);
    }

    // 3. podman on PATH.
    if let Some(found) = which("podman") {
        log::info!("Using podman as container runtime: {found}");
        return Some(found);
    }

    // 4. Well-known macOS Docker Desktop locations.
    for path in DOCKER_SEARCH_PATHS {
        if is_executable_file(path) {
            log::info!("Found docker at non-PATH location: {path}");
            return Some((*path).to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Security / user arg builders
// ---------------------------------------------------------------------------

/// Return the security/cap/tmpfs args tailored to the privilege mode. Faithful
/// port of `_build_security_args`.
pub fn build_security_args(run_as_host_user: bool) -> Vec<String> {
    let mut args: Vec<String> = BASE_SECURITY_ARGS.iter().map(|s| s.to_string()).collect();
    if !run_as_host_user {
        args.extend(GOSU_CAP_ARGS.iter().map(|s| s.to_string()));
    }
    args
}

/// Return `<uid>:<gid>` for the current host user, or `None` on platforms
/// where this is not meaningful. Faithful port of `_resolve_host_user_spec`.
pub fn resolve_host_user_spec() -> Option<String> {
    #[cfg(unix)]
    {
        // SAFETY: getuid/getgid are always-succeeding syscalls.
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        Some(format!("{uid}:{gid}"))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Docker availability preflight
// ---------------------------------------------------------------------------

/// Best-effort check that the docker CLI is available before use. Faithful port
/// of `_ensure_docker_available`.
///
/// Returns `Ok(docker_exe)` on success, or `Err(message)` mirroring the Python
/// `RuntimeError` messages.
pub fn ensure_docker_available() -> Result<String, String> {
    let docker_exe = match find_docker() {
        Some(p) => p,
        None => {
            log::error!(
                "Docker backend selected but no docker executable was found in PATH \
                 or known install locations. Install Docker Desktop and ensure the \
                 CLI is available."
            );
            return Err(
                "Docker executable not found in PATH or known install locations. \
                 Install Docker and ensure the 'docker' command is available."
                    .to_string(),
            );
        }
    };

    let output = run_with_timeout(
        Command::new(&docker_exe).arg("version"),
        Duration::from_secs(5),
    );

    match output {
        Ok(Some(out)) => {
            if !out.status.success() {
                let code = out.status.code().unwrap_or(-1);
                let stderr = String::from_utf8_lossy(&out.stderr);
                log::error!(
                    "Docker backend selected but '{docker_exe} version' failed \
                     (exit code {code}, stderr={})",
                    stderr.trim()
                );
                return Err(
                    "Docker command is available but 'docker version' failed. \
                     Check your Docker installation."
                        .to_string(),
                );
            }
            Ok(docker_exe)
        }
        Ok(None) => {
            // Timed out — daemon not responding.
            log::error!(
                "Docker backend selected but '{docker_exe} version' timed out. \
                 The Docker daemon may not be running."
            );
            Err(
                "Docker daemon is not responding. Ensure Docker is running and try again."
                    .to_string(),
            )
        }
        Err(_) => {
            log::error!(
                "Docker backend selected but the resolved docker executable '{docker_exe}' \
                 could not be executed."
            );
            Err(
                "Docker executable could not be executed. Check your Docker installation."
                    .to_string(),
            )
        }
    }
}

/// Run `cmd`, waiting up to `timeout`. Returns `Ok(Some(output))` on
/// completion, `Ok(None)` on timeout (the child is killed), and `Err` if the
/// process could not be spawned. A lightweight stand-in for
/// `subprocess.run(..., timeout=...)`.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> std::io::Result<Option<std::process::Output>> {
    use std::io::Read;
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()?;

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                if let Some(mut o) = child.stdout.take() {
                    let _ = o.read_to_end(&mut stdout);
                }
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_end(&mut stderr);
                }
                return Ok(Some(std::process::Output {
                    status,
                    stdout,
                    stderr,
                }));
            }
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// storage-opt support probe
// ---------------------------------------------------------------------------

static STORAGE_OPT_OK: OnceLock<bool> = OnceLock::new();

/// Check if Docker's storage driver supports `--storage-opt size=`. Faithful
/// port of `_storage_opt_supported`.
///
/// Only overlay2 on XFS with pquota supports per-container disk quotas. Most
/// distros default to ext4, where this flag errors out. The result is cached
/// process-wide on first call (matching the module global `_storage_opt_ok`).
pub fn storage_opt_supported() -> bool {
    if let Some(cached) = STORAGE_OPT_OK.get() {
        return *cached;
    }

    let result = probe_storage_opt();
    let _ = STORAGE_OPT_OK.set(result);
    log::debug!("Docker --storage-opt support: {result}");
    result
}

fn probe_storage_opt() -> bool {
    let docker = find_docker().unwrap_or_else(|| "docker".to_string());

    let info = run_with_timeout(
        Command::new(&docker)
            .arg("info")
            .arg("--format")
            .arg("{{.Driver}}"),
        Duration::from_secs(10),
    );
    let driver = match info {
        Ok(Some(out)) => String::from_utf8_lossy(&out.stdout).trim().to_lowercase(),
        _ => return false,
    };
    if driver != "overlay2" {
        return false;
    }

    // overlay2 only supports storage-opt on XFS with pquota. Probe by creating
    // a throwaway container.
    let probe = run_with_timeout(
        Command::new(&docker)
            .arg("create")
            .arg("--storage-opt")
            .arg("size=1m")
            .arg("hello-world"),
        Duration::from_secs(15),
    );
    match probe {
        Ok(Some(out)) if out.status.success() => {
            let container_id = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !container_id.is_empty() {
                let _ = run_with_timeout(
                    Command::new(&docker).arg("rm").arg(&container_id),
                    Duration::from_secs(5),
                );
            }
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Credential / skill / cache mounts (injectable; not yet ported natively)
// ---------------------------------------------------------------------------

/// A single bind-mount declaration `{host_path, container_path}` as produced by
/// `tools.credential_files`. Read-only mounts are added with a `:ro` suffix.
#[derive(Debug, Clone)]
pub struct MountEntry {
    pub host_path: String,
    pub container_path: String,
}

/// Source of the host-side mounts that the Python module pulls from
/// `tools.credential_files`. Not yet ported to native Rust, so this is an
/// injectable hook. The default ([`MountSource::empty`]) contributes nothing,
/// matching the Python "import failed → no mounts" fallback.
#[derive(Default)]
pub struct MountSource {
    pub credential_file_mounts: Vec<MountEntry>,
    pub skills_directory_mounts: Vec<MountEntry>,
    pub cache_directory_mounts: Vec<MountEntry>,
}

impl MountSource {
    pub fn empty() -> Self {
        MountSource::default()
    }
}

// ---------------------------------------------------------------------------
// Configuration + plan
// ---------------------------------------------------------------------------

/// Configuration for a [`DockerEnvironment`], mirroring `__init__` parameters.
#[derive(Debug, Clone)]
pub struct DockerConfig {
    pub image: String,
    pub cwd: String,
    pub timeout: i32,
    pub cpu: f64,
    pub memory: i64,
    pub disk: i64,
    pub persistent_filesystem: bool,
    pub task_id: String,
    pub volumes: Option<Vec<String>>,
    pub forward_env: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, EnvValue>>,
    pub network: bool,
    pub host_cwd: Option<String>,
    pub auto_mount_cwd: bool,
    pub run_as_host_user: bool,
}

impl Default for DockerConfig {
    fn default() -> Self {
        DockerConfig {
            image: String::new(),
            cwd: "/root".to_string(),
            timeout: 60,
            cpu: 0.0,
            memory: 0,
            disk: 0,
            persistent_filesystem: false,
            task_id: "default".to_string(),
            volumes: None,
            forward_env: None,
            env: None,
            network: true,
            host_cwd: None,
            auto_mount_cwd: false,
            run_as_host_user: false,
        }
    }
}

/// The fully-assembled `docker run` plan: every input choice resolved into the
/// argument vectors and host-side scratch dirs the Python `__init__` computes
/// before invoking `docker run -d`. Computing this separately from actually
/// launching the container makes the heavy argument logic unit-testable
/// without a Docker daemon.
#[derive(Debug, Clone, Default)]
pub struct DockerRunPlan {
    /// Final `[...all_run_args...]` (security + user + writable + resource +
    /// volume + env), in the exact order the Python code concatenates them.
    pub all_run_args: Vec<String>,
    pub security_args: Vec<String>,
    pub user_args: Vec<String>,
    pub writable_args: Vec<String>,
    pub resource_args: Vec<String>,
    pub volume_args: Vec<String>,
    pub env_args: Vec<String>,
    /// Host-side bind-mount dirs created on disk (persistent mode only).
    pub workspace_dir: Option<String>,
    pub home_dir: Option<String>,
    /// The normalised cwd (`~` rewritten to `/root`).
    pub cwd: String,
}

/// Inputs the plan builder needs that come from the wider system: where sandbox
/// dirs live, which host platform we are on, the storage-opt probe result, the
/// host user spec, and the credential/skill/cache mounts. Supplying them
/// explicitly keeps the builder pure and testable.
pub struct PlanContext<'a> {
    /// Root for persistent sandbox storage (`get_sandbox_dir()`), e.g.
    /// `crate::tool_environments_base::get_sandbox_dir(&hermes_home)`.
    pub sandbox_dir: &'a Path,
    /// Whether the host platform is macOS (`sys.platform == "darwin"`).
    pub is_darwin: bool,
    /// Result of [`storage_opt_supported`] — passed in so the builder need not
    /// touch the Docker daemon.
    pub storage_opt_supported: bool,
    /// `<uid>:<gid>` host user spec, or `None` (see [`resolve_host_user_spec`]).
    pub host_user_spec: Option<String>,
    /// Mounts contributed by credential-files/skills/cache (see [`MountSource`]).
    pub mounts: &'a MountSource,
    /// If true, actually create the persistent bind-mount dirs on disk.
    pub create_dirs: bool,
}

/// Build the full `docker run` argument plan. This is a faithful port of the
/// body of `DockerEnvironment.__init__` up to (but not including) the actual
/// `subprocess.run([... docker run ...])` call.
pub fn build_run_plan(cfg: &DockerConfig, ctx: &PlanContext) -> std::io::Result<DockerRunPlan> {
    let mut plan = DockerRunPlan::default();

    let cwd = if cfg.cwd == "~" {
        "/root".to_string()
    } else {
        cfg.cwd.clone()
    };
    plan.cwd = cwd;

    let persistent = cfg.persistent_filesystem;
    let env = normalize_env_dict(cfg.env.as_ref());

    log::info!("DockerEnvironment volumes: {:?}", cfg.volumes);
    // Ensure volumes is a list — Python coerces a non-list to []. In Rust the
    // type already enforces a list, so nothing to do.

    // --- Resource limit args ---
    let mut resource_args: Vec<String> = Vec::new();
    if cfg.cpu > 0.0 {
        resource_args.push("--cpus".to_string());
        resource_args.push(python_str_float_general(cfg.cpu));
    }
    if cfg.memory > 0 {
        resource_args.push("--memory".to_string());
        resource_args.push(format!("{}m", cfg.memory));
    }
    if cfg.disk > 0 && !ctx.is_darwin {
        if ctx.storage_opt_supported {
            resource_args.push("--storage-opt".to_string());
            resource_args.push(format!("size={}m", cfg.disk));
        } else {
            log::warn!(
                "Docker storage driver does not support per-container disk limits \
                 (requires overlay2 on XFS with pquota). Container will run without disk quota."
            );
        }
    }
    if !cfg.network {
        resource_args.push("--network=none".to_string());
    }

    // --- User-configured volume mounts (docker_volumes) ---
    let mut volume_args: Vec<String> = Vec::new();
    let mut workspace_explicitly_mounted = false;
    if let Some(volumes) = &cfg.volumes {
        for vol in volumes {
            let vol = vol.trim();
            if vol.is_empty() {
                continue;
            }
            if vol.contains(':') {
                volume_args.push("-v".to_string());
                volume_args.push(vol.to_string());
                if vol.contains(":/workspace") {
                    workspace_explicitly_mounted = true;
                }
            } else {
                log::warn!("Docker volume '{vol}' missing colon, skipping");
            }
        }
    }

    // --- host_cwd auto-mount resolution ---
    let host_cwd_abs: String = match &cfg.host_cwd {
        Some(hc) if !hc.is_empty() => abspath_expanduser(hc),
        _ => String::new(),
    };
    let host_cwd_is_dir = !host_cwd_abs.is_empty() && Path::new(&host_cwd_abs).is_dir();
    let bind_host_cwd = cfg.auto_mount_cwd
        && !host_cwd_abs.is_empty()
        && host_cwd_is_dir
        && !workspace_explicitly_mounted;
    if cfg.auto_mount_cwd && cfg.host_cwd.is_some() && !host_cwd_abs.is_empty() && !host_cwd_is_dir {
        log::debug!(
            "Skipping docker cwd mount: host_cwd is not a valid directory: {:?}",
            cfg.host_cwd
        );
    }

    // --- writable workspace / home ---
    let mut writable_args: Vec<String> = Vec::new();
    if persistent {
        let sandbox = ctx.sandbox_dir.join("docker").join(&cfg.task_id);
        let home_dir = sandbox.join("home");
        let home_dir_str = home_dir.to_string_lossy().to_string();
        if ctx.create_dirs {
            std::fs::create_dir_all(&home_dir)?;
        }
        plan.home_dir = Some(home_dir_str.clone());
        writable_args.push("-v".to_string());
        writable_args.push(format!("{home_dir_str}:/root"));

        if !bind_host_cwd && !workspace_explicitly_mounted {
            let workspace_dir = sandbox.join("workspace");
            let workspace_dir_str = workspace_dir.to_string_lossy().to_string();
            if ctx.create_dirs {
                std::fs::create_dir_all(&workspace_dir)?;
            }
            plan.workspace_dir = Some(workspace_dir_str.clone());
            writable_args.push("-v".to_string());
            writable_args.push(format!("{workspace_dir_str}:/workspace"));
        }
    } else {
        if !bind_host_cwd && !workspace_explicitly_mounted {
            writable_args.push("--tmpfs".to_string());
            writable_args.push("/workspace:rw,exec,size=10g".to_string());
        }
        writable_args.push("--tmpfs".to_string());
        writable_args.push("/home:rw,exec,size=1g".to_string());
        writable_args.push("--tmpfs".to_string());
        writable_args.push("/root:rw,exec,size=1g".to_string());
    }

    if bind_host_cwd {
        log::info!("Mounting configured host cwd to /workspace: {host_cwd_abs}");
        // Prepend the cwd mount to the existing volume_args.
        let mut prefixed = vec!["-v".to_string(), format!("{host_cwd_abs}:/workspace")];
        prefixed.extend(volume_args);
        volume_args = prefixed;
    } else if workspace_explicitly_mounted {
        log::debug!("Skipping docker cwd mount: /workspace already mounted by user config");
    }

    // --- credential / skill / cache mounts (read-only) ---
    for mount_entry in &ctx.mounts.credential_file_mounts {
        volume_args.push("-v".to_string());
        volume_args.push(format!(
            "{}:{}:ro",
            mount_entry.host_path, mount_entry.container_path
        ));
        log::info!(
            "Docker: mounting credential {} -> {}",
            mount_entry.host_path,
            mount_entry.container_path
        );
    }
    for skills_mount in &ctx.mounts.skills_directory_mounts {
        volume_args.push("-v".to_string());
        volume_args.push(format!(
            "{}:{}:ro",
            skills_mount.host_path, skills_mount.container_path
        ));
        log::info!(
            "Docker: mounting skills dir {} -> {}",
            skills_mount.host_path,
            skills_mount.container_path
        );
    }
    for cache_mount in &ctx.mounts.cache_directory_mounts {
        volume_args.push("-v".to_string());
        volume_args.push(format!(
            "{}:{}:ro",
            cache_mount.host_path, cache_mount.container_path
        ));
        log::info!(
            "Docker: mounting cache dir {} -> {}",
            cache_mount.host_path,
            cache_mount.container_path
        );
    }

    // --- explicit env args (docker_env), sorted by key ---
    let mut env_args: Vec<String> = Vec::new();
    for (key, value) in &env {
        env_args.push("-e".to_string());
        env_args.push(format!("{key}={value}"));
    }

    // --- user args (run-as-host-user) ---
    let mut user_args: Vec<String> = Vec::new();
    if cfg.run_as_host_user {
        match &ctx.host_user_spec {
            Some(user_spec) => {
                user_args.push("--user".to_string());
                user_args.push(user_spec.clone());
                log::info!("Docker: running container as host user {user_spec}");
            }
            None => {
                log::warn!(
                    "docker_run_as_host_user is enabled but this platform does \
                     not expose POSIX uid/gid; container will start as its \
                     image default user."
                );
            }
        }
    }
    let security_args = build_security_args(cfg.run_as_host_user && !user_args.is_empty());

    log::info!("Docker volume_args: {volume_args:?}");

    let mut all_run_args: Vec<String> = Vec::new();
    all_run_args.extend(security_args.iter().cloned());
    all_run_args.extend(user_args.iter().cloned());
    all_run_args.extend(writable_args.iter().cloned());
    all_run_args.extend(resource_args.iter().cloned());
    all_run_args.extend(volume_args.iter().cloned());
    all_run_args.extend(env_args.iter().cloned());
    log::info!("Docker run_args: {all_run_args:?}");

    plan.security_args = security_args;
    plan.user_args = user_args;
    plan.writable_args = writable_args;
    plan.resource_args = resource_args;
    plan.volume_args = volume_args;
    plan.env_args = env_args;
    plan.all_run_args = all_run_args;

    Ok(plan)
}

/// `os.path.abspath(os.path.expanduser(p))`-equivalent.
fn abspath_expanduser(p: &str) -> String {
    let expanded: PathBuf = if let Some(rest) = p.strip_prefix("~") {
        if rest.is_empty() || rest.starts_with('/') {
            if let Some(home) = dirs::home_dir() {
                if rest.is_empty() {
                    home
                } else {
                    home.join(rest.trim_start_matches('/'))
                }
            } else {
                PathBuf::from(p)
            }
        } else {
            PathBuf::from(p)
        }
    } else {
        PathBuf::from(p)
    };

    let abs = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .map(|c| c.join(&expanded))
            .unwrap_or(expanded)
    };
    normalize_path(&abs).to_string_lossy().to_string()
}

/// Lexical path normalisation (collapse `.`/`..`) similar to what
/// `os.path.abspath` performs.
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// General float-to-string mirroring Python `str(float)` for `--cpus`. The
/// Python source uses `str(cpu)`, e.g. `str(1.5)` == "1.5", `str(2.0)` == "2.0".
fn python_str_float_general(f: f64) -> String {
    python_str_float(f)
}

// ---------------------------------------------------------------------------
// init-time env forwarding
// ---------------------------------------------------------------------------

/// Loader for `~/.hermes/.env` values. Mirrors `_load_hermes_env_vars`. The
/// Python version imported `hermes_cli.config.load_env`; that is not yet ported
/// natively, so this is an injectable hook defaulting to "no values".
pub type HermesEnvLoader = fn() -> BTreeMap<String, String>;

fn empty_hermes_env() -> BTreeMap<String, String> {
    BTreeMap::new()
}

/// Build `-e KEY=VALUE` args for injecting host env vars into `init_session`.
/// Faithful port of `_build_init_env_args`.
///
/// `passthrough_keys` is the result of `tools.env_passthrough.get_all_passthrough`
/// (e.g. `crate::tool_env_passthrough::get_all_passthrough()`), and
/// `provider_blocklist` is the Hermes provider-credential blocklist
/// (`_HERMES_PROVIDER_ENV_BLOCKLIST`, e.g.
/// `crate::tool_env_passthrough::builtin_provider_env_blocklist()`).
///
/// Explicit `docker_forward_env` entries are an intentional opt-in and win over
/// the generic blocklist; only implicit passthrough keys are filtered.
pub fn build_init_env_args(
    env: &BTreeMap<String, String>,
    forward_env: &[String],
    passthrough_keys: &BTreeSet<String>,
    provider_blocklist: &BTreeSet<String>,
    hermes_env_loader: Option<HermesEnvLoader>,
) -> Vec<String> {
    let mut exec_env: BTreeMap<String, String> = env.clone();

    let explicit_forward_keys: BTreeSet<String> = forward_env.iter().cloned().collect();

    // passthrough_keys - provider_blocklist
    let filtered_passthrough: BTreeSet<String> = passthrough_keys
        .difference(provider_blocklist)
        .cloned()
        .collect();

    // explicit | filtered_passthrough
    let forward_keys: BTreeSet<String> = explicit_forward_keys
        .union(&filtered_passthrough)
        .cloned()
        .collect();

    let hermes_env: BTreeMap<String, String> = if !forward_keys.is_empty() {
        hermes_env_loader.unwrap_or(empty_hermes_env)()
    } else {
        BTreeMap::new()
    };

    for key in &forward_keys {
        let value = match std::env::var(key) {
            Ok(v) => Some(v),
            Err(_) => hermes_env.get(key).cloned(),
        };
        if let Some(v) = value {
            exec_env.insert(key.clone(), v);
        }
    }

    // sorted(exec_env) — BTreeMap already iterates in sorted key order.
    let mut args: Vec<String> = Vec::new();
    for (key, value) in &exec_env {
        args.push("-e".to_string());
        args.push(format!("{key}={value}"));
    }
    args
}

// ---------------------------------------------------------------------------
// docker exec command construction
// ---------------------------------------------------------------------------

/// Build the `docker exec ...` argv to run `cmd_string` inside the container.
/// Faithful port of `_run_bash` (the argv construction; the actual spawn is
/// left to the caller / [`DockerEnvironment::run_bash`]).
///
/// `init_env_args` are the `-e KEY=VALUE` args from [`build_init_env_args`];
/// they are only injected when `login` is true (i.e. during `init_session`).
pub fn build_exec_argv(
    docker_exe: &str,
    container_id: &str,
    cmd_string: &str,
    login: bool,
    init_env_args: &[String],
    has_stdin: bool,
) -> Vec<String> {
    let mut cmd: Vec<String> = vec![docker_exe.to_string(), "exec".to_string()];
    if has_stdin {
        cmd.push("-i".to_string());
    }
    if login {
        cmd.extend(init_env_args.iter().cloned());
    }
    cmd.push(container_id.to_string());
    if login {
        cmd.push("bash".to_string());
        cmd.push("-l".to_string());
        cmd.push("-c".to_string());
        cmd.push(cmd_string.to_string());
    } else {
        cmd.push("bash".to_string());
        cmd.push("-c".to_string());
        cmd.push(cmd_string.to_string());
    }
    cmd
}

/// Build the `docker run -d ...` argv. Faithful port of the `run_cmd` list in
/// `__init__`. `container_name` is normally `hermes-<8 hex>`.
pub fn build_run_argv(
    docker_exe: &str,
    container_name: &str,
    cwd: &str,
    all_run_args: &[String],
    image: &str,
) -> Vec<String> {
    let mut cmd: Vec<String> = vec![
        docker_exe.to_string(),
        "run".to_string(),
        "-d".to_string(),
        "--init".to_string(),
        "--name".to_string(),
        container_name.to_string(),
        "-w".to_string(),
        cwd.to_string(),
    ];
    cmd.extend(all_run_args.iter().cloned());
    cmd.push(image.to_string());
    cmd.push("sleep".to_string());
    cmd.push("infinity".to_string());
    cmd
}

/// Generate a container name `hermes-<8 hex>`. Mirrors
/// `f"hermes-{uuid.uuid4().hex[:8]}"`.
pub fn generate_container_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // A lightweight random-ish 32-bit hex; the Python version only needs 8 hex
    // chars of entropy for a non-colliding container suffix.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let mix = nanos
        ^ pid.rotate_left(13)
        ^ (std::ptr::addr_of!(nanos) as usize as u32);
    format!("hermes-{mix:08x}")
}

// ---------------------------------------------------------------------------
// DockerEnvironment
// ---------------------------------------------------------------------------

/// Hardened Docker container execution with resource limits and persistence.
/// Faithful port of `DockerEnvironment`.
///
/// Security: all capabilities dropped, no privilege escalation, PID limits,
/// size-limited tmpfs for scratch dirs. The container itself is the security
/// boundary — the filesystem inside is writable so agents can install packages
/// as needed. Writable workspace via tmpfs or bind mounts.
///
/// Persistence: when enabled, bind mounts preserve `/workspace` and `/root`
/// across container restarts.
pub struct DockerEnvironment {
    pub cwd: String,
    pub timeout: i32,
    persistent: bool,
    task_id: String,
    forward_env: Vec<String>,
    env: BTreeMap<String, String>,
    container_id: Option<String>,
    docker_exe: String,
    init_env_args: Vec<String>,
    workspace_dir: Option<String>,
    home_dir: Option<String>,
}

impl DockerEnvironment {
    /// Construct and start a Docker container. Faithful port of
    /// `DockerEnvironment.__init__`.
    ///
    /// `ctx` supplies the system inputs the plan builder needs (sandbox dir,
    /// platform, storage-opt result, host user, mounts). `passthrough_keys`
    /// and `provider_blocklist` feed [`build_init_env_args`]; pass them from
    /// `crate::tool_env_passthrough`.
    ///
    /// This performs network/daemon access: it runs `docker version` (preflight)
    /// then `docker run -d` to start the container, then [`init_session`]. On
    /// any failure it returns `Err(message)`.
    ///
    /// [`init_session`]: DockerEnvironment::init_session
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &DockerConfig,
        ctx: &PlanContext,
        passthrough_keys: &BTreeSet<String>,
        provider_blocklist: &BTreeSet<String>,
        hermes_env_loader: Option<HermesEnvLoader>,
    ) -> Result<Self, String> {
        // Fail fast if Docker is not available.
        ensure_docker_available()?;

        let plan = build_run_plan(cfg, ctx).map_err(|e| format!("Failed to prepare workspace dirs: {e}"))?;

        let env = normalize_env_dict(cfg.env.as_ref());
        let forward_env = normalize_forward_env_names(cfg.forward_env.as_deref());

        // Resolve the docker executable once so it works even when
        // /usr/local/bin is not in PATH.
        let docker_exe = find_docker().unwrap_or_else(|| "docker".to_string());

        let container_name = generate_container_name();
        let run_argv = build_run_argv(
            &docker_exe,
            &container_name,
            &plan.cwd,
            &plan.all_run_args,
            &cfg.image,
        );
        log::debug!("Starting container: {}", run_argv.join(" "));

        let mut command = Command::new(&run_argv[0]);
        command.args(&run_argv[1..]);
        let output = run_with_timeout(&mut command, Duration::from_secs(120))
            .map_err(|e| format!("Failed to spawn docker run: {e}"))?;
        let output = match output {
            Some(o) => o,
            None => return Err("docker run timed out (image pull may take a while)".to_string()),
        };
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "docker run failed (exit code {}): {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ));
        }
        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let short = container_id.chars().take(12).collect::<String>();
        log::info!("Started container {container_name} ({short})");

        let init_env_args = build_init_env_args(
            &env,
            &forward_env,
            passthrough_keys,
            provider_blocklist,
            hermes_env_loader,
        );

        let cwd = if cfg.cwd == "~" {
            "/root".to_string()
        } else {
            plan.cwd.clone()
        };

        let mut this = DockerEnvironment {
            cwd,
            timeout: cfg.timeout,
            persistent: cfg.persistent_filesystem,
            task_id: cfg.task_id.clone(),
            forward_env,
            env,
            container_id: Some(container_id),
            docker_exe,
            init_env_args,
            workspace_dir: plan.workspace_dir,
            home_dir: plan.home_dir,
        };

        // Initialise session snapshot inside the container.
        this.init_session();

        Ok(this)
    }

    /// The resolved docker executable path.
    pub fn docker_exe(&self) -> &str {
        &self.docker_exe
    }

    /// The started container id (full hex), if running.
    pub fn container_id(&self) -> Option<&str> {
        self.container_id.as_deref()
    }

    /// The init-time `-e` forwarding args.
    pub fn init_env_args(&self) -> &[String] {
        &self.init_env_args
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn forward_env(&self) -> &[String] {
        &self.forward_env
    }

    pub fn env(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    /// Build the argv to spawn a bash process inside the container. Faithful
    /// port of `_run_bash`'s command construction. Panics if the container is
    /// not started (mirrors the Python `assert self._container_id`).
    pub fn run_bash_argv(&self, cmd_string: &str, login: bool, stdin_data: Option<&str>) -> Vec<String> {
        let container_id = self
            .container_id
            .as_deref()
            .expect("Container not started");
        build_exec_argv(
            &self.docker_exe,
            container_id,
            cmd_string,
            login,
            &self.init_env_args,
            stdin_data.is_some(),
        )
    }

    /// Spawn a bash process inside the container, optionally piping
    /// `stdin_data`. Returns the spawned [`std::process::Child`].
    pub fn run_bash(
        &self,
        cmd_string: &str,
        login: bool,
        stdin_data: Option<&str>,
    ) -> std::io::Result<std::process::Child> {
        let argv = self.run_bash_argv(cmd_string, login, stdin_data);
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]);
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        if stdin_data.is_some() {
            command.stdin(Stdio::piped());
        } else {
            command.stdin(Stdio::null());
        }
        let mut child = command.spawn()?;
        if let Some(data) = stdin_data {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let data = data.to_string();
                std::thread::spawn(move || {
                    let _ = stdin.write_all(data.as_bytes());
                });
            }
        }
        Ok(child)
    }

    /// Initialise the session snapshot inside the container. The Python
    /// `BaseEnvironment.init_session` runs a login shell to capture the
    /// environment snapshot; the heavy snapshot machinery lives in
    /// `crate::tool_environments_base`. Here we run a no-op login bash so the
    /// snapshot args (`init_env_args`) are exercised, matching the structure of
    /// the Python call. Failures are swallowed (best-effort), as in Python.
    pub fn init_session(&mut self) {
        if self.container_id.is_none() {
            return;
        }
        // Run a login bash that simply succeeds; this mirrors the shape of the
        // Python init (which sources/export-p's a snapshot). The full snapshot
        // bootstrap script generation lives in tool_environments_base.
        if let Ok(mut child) = self.run_bash("true", true, None) {
            let _ = run_child_with_timeout(&mut child, Duration::from_secs(30));
        }
    }

    /// Stop and remove the container. Bind-mount dirs persist if
    /// `persistent=True`. Faithful port of `cleanup`.
    pub fn cleanup(&mut self) {
        if let Some(container_id) = self.container_id.take() {
            let docker_exe = &self.docker_exe;
            // Stop in background so cleanup doesn't block.
            let stop_cmd = format!(
                "(timeout 60 {docker_exe} stop {container_id} || \
                 {docker_exe} rm -f {container_id}) >/dev/null 2>&1 &"
            );
            if let Err(e) = spawn_shell(&stop_cmd) {
                log::warn!("Failed to stop container {container_id}: {e}");
            }

            if !self.persistent {
                // Also schedule removal (stop only leaves it as stopped).
                let rm_cmd =
                    format!("sleep 3 && {docker_exe} rm -f {container_id} >/dev/null 2>&1 &");
                let _ = spawn_shell(&rm_cmd);
            }
        }

        if !self.persistent {
            for d in [self.workspace_dir.take(), self.home_dir.take()].into_iter().flatten() {
                let _ = std::fs::remove_dir_all(&d);
            }
        }
    }
}

impl Drop for DockerEnvironment {
    fn drop(&mut self) {
        if self.container_id.is_some() {
            self.cleanup();
        }
    }
}

/// Spawn a detached `bash -c <cmd>` (mirrors `subprocess.Popen(cmd, shell=True)`).
fn spawn_shell(cmd: &str) -> std::io::Result<()> {
    Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

/// Wait for a child up to `timeout`, killing it on expiry.
fn run_child_with_timeout(child: &mut std::process::Child, timeout: Duration) -> std::io::Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_str(s: &str) -> EnvValue {
        EnvValue::Str(s.to_string())
    }

    #[test]
    fn forward_env_dedup_and_validate() {
        let input = vec![
            "FOO".to_string(),
            " BAR ".to_string(),
            "FOO".to_string(), // dup
            "".to_string(),    // empty
            "1BAD".to_string(),// invalid name
            "ok_2".to_string(),
        ];
        let got = normalize_forward_env_names(Some(&input));
        assert_eq!(got, vec!["FOO".to_string(), "BAR".to_string(), "ok_2".to_string()]);
        assert!(normalize_forward_env_names(None).is_empty());
    }

    #[test]
    fn env_dict_normalisation() {
        let mut env = BTreeMap::new();
        env.insert("A".to_string(), ev_str("x"));
        env.insert("B".to_string(), EnvValue::Int(7));
        env.insert("C".to_string(), EnvValue::Bool(true));
        env.insert("D".to_string(), EnvValue::Float(2.0));
        env.insert("1BAD".to_string(), ev_str("nope"));
        env.insert("E".to_string(), EnvValue::Complex);
        let got = normalize_env_dict(Some(&env));
        assert_eq!(got.get("A").unwrap(), "x");
        assert_eq!(got.get("B").unwrap(), "7");
        assert_eq!(got.get("C").unwrap(), "True");
        assert_eq!(got.get("D").unwrap(), "2.0");
        assert!(!got.contains_key("1BAD"));
        assert!(!got.contains_key("E"));
    }

    #[test]
    fn security_args_modes() {
        let host = build_security_args(true);
        let gosu = build_security_args(false);
        // Both share the base prefix.
        assert!(host.starts_with(&["--cap-drop".to_string(), "ALL".to_string()]));
        // gosu mode adds SETUID/SETGID at the end.
        assert!(gosu.windows(2).any(|w| w == ["--cap-add", "SETUID"]));
        assert!(gosu.windows(2).any(|w| w == ["--cap-add", "SETGID"]));
        assert!(!host.windows(2).any(|w| w == ["--cap-add", "SETUID"]));
        assert_eq!(gosu.len(), host.len() + 4);
    }

    fn empty_ctx<'a>(sandbox: &'a Path, mounts: &'a MountSource) -> PlanContext<'a> {
        PlanContext {
            sandbox_dir: sandbox,
            is_darwin: false,
            storage_opt_supported: false,
            host_user_spec: None,
            mounts,
            create_dirs: false,
        }
    }

    #[test]
    fn plan_non_persistent_default_tmpfs() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        // Non-persistent: tmpfs workspace + home + root.
        assert!(plan.writable_args.windows(2).any(|w| w
            == ["--tmpfs", "/workspace:rw,exec,size=10g"]));
        assert!(plan.writable_args.windows(2).any(|w| w == ["--tmpfs", "/home:rw,exec,size=1g"]));
        assert!(plan.writable_args.windows(2).any(|w| w == ["--tmpfs", "/root:rw,exec,size=1g"]));
        // gosu caps present (not run_as_host_user).
        assert!(plan.security_args.windows(2).any(|w| w == ["--cap-add", "SETUID"]));
        assert!(plan.workspace_dir.is_none());
    }

    #[test]
    fn plan_resource_args() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            cpu: 1.5,
            memory: 512,
            network: false,
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert!(plan.resource_args.windows(2).any(|w| w == ["--cpus", "1.5"]));
        assert!(plan.resource_args.windows(2).any(|w| w == ["--memory", "512m"]));
        assert!(plan.resource_args.contains(&"--network=none".to_string()));
    }

    #[test]
    fn plan_disk_darwin_skipped() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            disk: 1024,
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let mut ctx = empty_ctx(&sandbox, &mounts);
        ctx.is_darwin = true;
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert!(!plan.resource_args.iter().any(|a| a.starts_with("size=")));
        // Now non-darwin with storage-opt support.
        let mut ctx2 = empty_ctx(&sandbox, &mounts);
        ctx2.storage_opt_supported = true;
        let plan2 = build_run_plan(&cfg, &ctx2).unwrap();
        assert!(plan2.resource_args.windows(2).any(|w| w == ["--storage-opt", "size=1024m"]));
    }

    #[test]
    fn plan_volumes_and_workspace_detection() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            volumes: Some(vec![
                "  /host:/data ".to_string(),
                "bad_no_colon".to_string(),
                "".to_string(),
                "/h2:/workspace".to_string(),
            ]),
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert!(plan.volume_args.windows(2).any(|w| w == ["-v", "/host:/data"]));
        assert!(plan.volume_args.windows(2).any(|w| w == ["-v", "/h2:/workspace"]));
        // Because /workspace is explicitly mounted, no tmpfs /workspace.
        assert!(!plan
            .writable_args
            .windows(2)
            .any(|w| w == ["--tmpfs", "/workspace:rw,exec,size=10g"]));
    }

    #[test]
    fn plan_user_args_and_security_mode() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            run_as_host_user: true,
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let mut ctx = empty_ctx(&sandbox, &mounts);
        ctx.host_user_spec = Some("1000:1000".to_string());
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert!(plan.user_args.windows(2).any(|w| w == ["--user", "1000:1000"]));
        // run_as_host_user with --user → no gosu caps.
        assert!(!plan.security_args.windows(2).any(|w| w == ["--cap-add", "SETUID"]));

        // Without a host_user_spec, falls back to full cap set.
        let mut ctx2 = empty_ctx(&sandbox, &mounts);
        ctx2.host_user_spec = None;
        let plan2 = build_run_plan(&cfg, &ctx2).unwrap();
        assert!(plan2.user_args.is_empty());
        assert!(plan2.security_args.windows(2).any(|w| w == ["--cap-add", "SETUID"]));
    }

    #[test]
    fn plan_credential_mounts_readonly() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            ..Default::default()
        };
        let mounts = MountSource {
            credential_file_mounts: vec![MountEntry {
                host_path: "/host/cred".to_string(),
                container_path: "/c/cred".to_string(),
            }],
            ..Default::default()
        };
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert!(plan
            .volume_args
            .windows(2)
            .any(|w| w == ["-v", "/host/cred:/c/cred:ro"]));
    }

    #[test]
    fn plan_env_args_sorted() {
        let mut env = BTreeMap::new();
        env.insert("ZED".to_string(), ev_str("1"));
        env.insert("ALPHA".to_string(), ev_str("2"));
        let cfg = DockerConfig {
            image: "img".to_string(),
            env: Some(env),
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        // Sorted: ALPHA before ZED.
        assert_eq!(
            plan.env_args,
            vec!["-e", "ALPHA=2", "-e", "ZED=1"]
        );
    }

    #[test]
    fn plan_cwd_tilde_rewrite() {
        let cfg = DockerConfig {
            image: "img".to_string(),
            cwd: "~".to_string(),
            ..Default::default()
        };
        let mounts = MountSource::empty();
        let sandbox = PathBuf::from("/tmp/sbx");
        let ctx = empty_ctx(&sandbox, &mounts);
        let plan = build_run_plan(&cfg, &ctx).unwrap();
        assert_eq!(plan.cwd, "/root");
    }

    #[test]
    fn run_argv_shape() {
        let argv = build_run_argv(
            "docker",
            "hermes-abc12345",
            "/root",
            &["--network=none".to_string()],
            "ubuntu:24.04",
        );
        assert_eq!(
            argv,
            vec![
                "docker", "run", "-d", "--init", "--name", "hermes-abc12345", "-w", "/root",
                "--network=none", "ubuntu:24.04", "sleep", "infinity"
            ]
        );
    }

    #[test]
    fn exec_argv_login_vs_plain() {
        let init_env = vec!["-e".to_string(), "K=V".to_string()];
        let login = build_exec_argv("docker", "cid", "echo hi", true, &init_env, false);
        assert_eq!(
            login,
            vec!["docker", "exec", "-e", "K=V", "cid", "bash", "-l", "-c", "echo hi"]
        );
        let plain = build_exec_argv("docker", "cid", "echo hi", false, &init_env, true);
        // No -e injection on non-login; -i because stdin present.
        assert_eq!(
            plain,
            vec!["docker", "exec", "-i", "cid", "bash", "-c", "echo hi"]
        );
    }

    #[test]
    fn init_env_args_explicit_wins_over_blocklist() {
        // SECRET is on the blocklist AND in passthrough → filtered.
        // SECRET is also explicitly forwarded → must survive.
        unsafe {
            std::env::set_var("SECRET", "shh");
            std::env::set_var("PLAIN", "ok");
        }
        let env = BTreeMap::new();
        let forward = vec!["SECRET".to_string()];
        let mut passthrough = BTreeSet::new();
        passthrough.insert("SECRET".to_string());
        passthrough.insert("PLAIN".to_string());
        let mut blocklist = BTreeSet::new();
        blocklist.insert("SECRET".to_string());

        let args = build_init_env_args(&env, &forward, &passthrough, &blocklist, None);
        // SECRET survives because it is explicitly forwarded.
        assert!(args.windows(2).any(|w| w == ["-e", "SECRET=shh"]));
        // PLAIN survives via passthrough (not on blocklist).
        assert!(args.windows(2).any(|w| w == ["-e", "PLAIN=ok"]));
        unsafe {
            std::env::remove_var("SECRET");
            std::env::remove_var("PLAIN");
        }
    }

    #[test]
    fn init_env_args_blocklisted_passthrough_filtered() {
        unsafe {
            std::env::set_var("DOCKER_BLK", "v");
        }
        let env = BTreeMap::new();
        let forward: Vec<String> = vec![];
        let mut passthrough = BTreeSet::new();
        passthrough.insert("DOCKER_BLK".to_string());
        let mut blocklist = BTreeSet::new();
        blocklist.insert("DOCKER_BLK".to_string());
        let args = build_init_env_args(&env, &forward, &passthrough, &blocklist, None);
        // Filtered out — blocklisted and not explicitly forwarded.
        assert!(!args.windows(2).any(|w| w == ["-e", "DOCKER_BLK=v"]));
        unsafe {
            std::env::remove_var("DOCKER_BLK");
        }
    }

    #[test]
    fn init_env_args_docker_env_always_included() {
        let mut env = BTreeMap::new();
        env.insert("CFG".to_string(), "val".to_string());
        let args = build_init_env_args(
            &env,
            &[],
            &BTreeSet::new(),
            &BTreeSet::new(),
            None,
        );
        assert_eq!(args, vec!["-e", "CFG=val"]);
    }

    #[test]
    fn container_name_format() {
        let name = generate_container_name();
        assert!(name.starts_with("hermes-"));
        assert_eq!(name.len(), "hermes-".len() + 8);
        assert!(name["hermes-".len()..].chars().all(|c| c.is_ascii_hexdigit()));
    }
}
